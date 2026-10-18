"""SDK observations preserve totals, final answers and shutdown ownership."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import threading
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import pytest

from cayu.deadlines import ExecutionDeadline
from cayu.external_agents import ExternalAgentContext, ExternalAgentLaunch
from cayu.external_agents.claude import ClaudeAgentAdapter
from cayu.external_agents.codex import CodexAgentAdapter


@pytest.mark.parametrize("model_alias", [False, True])
def test_claude_observations_deduplicate_usage_and_gate_every_tool(
    monkeypatch, model_alias, capsys
):
    import cayu.external_agents.claude as module

    @dataclass
    class Text:
        text: str

    @dataclass
    class Assistant:
        content: list
        usage: dict
        message_id: str = "message"
        uuid: str | None = None
        model: str = "claude"

    @dataclass
    class Result:
        model_usage: dict = field(
            default_factory=lambda: {
                ("claude-alias" if model_alias else "claude"): {
                    "inputTokens": 10,
                    "outputTokens": 6,
                }
            }
        )
        usage: dict = field(default_factory=dict)
        total_cost_usd: float = 0.2
        is_error: bool = False
        structured_output: dict = field(default_factory=lambda: {"answer": 42})
        result: str = "unused"

    @dataclass
    class Stream:
        event: dict
        parent_tool_use_id: str | None = None

    options_seen = []
    authorization = []
    closed = []

    async def query(*, prompt, options):
        options_seen.append(options)
        assert (await anext(prompt))["message"]["content"] == "review"
        hook = options.hooks["PreToolUse"][0].hooks[0]
        for call_id in ("call", "call"):
            result = await hook({"tool_name": "Read", "tool_input": {"path": "a"}}, call_id, {})
            assert result["hookSpecificOutput"]["permissionDecision"] == "allow"
        denied = await hook({"tool_name": "Read", "tool_input": {"path": "other"}}, "call", {})
        assert denied["hookSpecificOutput"]["permissionDecision"] == "deny"
        assert options.include_partial_messages
        assert callable(options.stderr)
        options.stderr("private guest diagnostic must not enter service logs")
        yield Stream(
            {
                "type": "message_start",
                "message": {
                    "id": "message",
                    "model": "claude",
                    "usage": {"input_tokens": 10, "output_tokens": 1},
                },
            }
        )
        yield Assistant(
            [Text("first block")], {"input_tokens": 10, "output_tokens": 1}, uuid="block-one"
        )
        yield Assistant(
            [Text("second block")], {"input_tokens": 10, "output_tokens": 1}, uuid="block-two"
        )
        yield Stream({"type": "message_delta", "usage": {"output_tokens": 6}})
        yield Stream({"type": "message_stop"})
        yield Result()

    class Client:
        def __init__(self, *, options):
            self.options = options

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def query(self, prompt):
            self.prompt = prompt

        def receive_response(self):
            return query(prompt=self.prompt, options=self.options)

    sdk = SimpleNamespace(
        ClaudeSDKClient=Client,
        ClaudeAgentOptions=SimpleNamespace,
        HookMatcher=SimpleNamespace,
        AssistantMessage=Assistant,
        ResultMessage=Result,
        TextBlock=Text,
        ToolUseBlock=type("ToolUse", (), {}),
        UserMessage=type("User", (), {}),
        StreamEvent=Stream,
        PermissionResultAllow=SimpleNamespace,
        PermissionResultDeny=SimpleNamespace,
    )
    monkeypatch.setattr(module, "import_module", lambda name: sdk)
    monkeypatch.setenv("HOST_ONLY_SECRET", "private")

    async def scenario():
        async def close():
            closed.append(True)

        @asynccontextmanager
        async def launch(context):
            yield ExternalAgentLaunch("/sandbox/claude", "/checkout", close)

        async def authorize(name, args, call_id):
            authorization.append((name, args, call_id))
            return True

        adapter = ClaudeAgentAdapter(model="claude", launch=launch)
        events = [
            event
            async for event in adapter.stream(
                ExternalAgentContext(
                    "session",
                    "review",
                    ExecutionDeadline.after(20),
                    {"type": "object"},
                    authorize,
                )
            )
        ]
        usage = [event.payload for event in events if event.kind == "usage"]
        assert sum(item.get("input_tokens", 0) for item in usage) == 10
        assert sum(item.get("output_tokens", 0) for item in usage) == 6
        assert usage[-1]["cost"] == "0.200000000"
        text = [event for event in events if event.kind == "text"]
        assert len({event.id for event in text}) == 2
        assert [event.payload["text"] for event in text] == ["first block", "second block"]
        final_usage = next(event for event in events if event.kind == "usage")
        assert final_usage.payload["output_tokens"] == 6
        assert events.index(final_usage) < next(
            i for i, event in enumerate(events) if event.kind == "result"
        )
        # Feed actual SDK-shaped observations through the recorder and normal cost projection.
        from decimal import Decimal

        from cayu import CayuApp, RunRequest
        from cayu.budgets import ModelPrice, PriceBook
        from cayu.external_agents import run_external_agent
        from cayu.tasks import InMemoryTaskStore

        class Observations:
            name, provider_name, model = "claude", "anthropic", "claude"

            async def stream(self, context):
                for event in events:
                    yield event

        app = CayuApp(task_store=InMemoryTaskStore(), enable_logging=False)
        recorded = await run_external_agent(
            app,
            RunRequest(
                agent_name="claude", messages=[], execution_deadline=ExecutionDeadline.after(20)
            ),
            Observations(),
            prompt="review",
        )
        cost = await app.get_session_cost(
            recorded.session_id,
            pricing=PriceBook(
                prices=(
                    ModelPrice.fixed(
                        provider_name="anthropic",
                        model="claude",
                        input_per_million=Decimal("1000"),
                        output_per_million=Decimal("1000"),
                    ),
                )
            ),
        )
        assert cost.total_cost == Decimal("0.016")
        from datetime import UTC, datetime, timedelta

        from cayu.sessions.usage import UsageRollupQuery

        summary = await app.session_store.aggregate_usage(
            UsageRollupQuery(
                start_at=datetime.now(UTC) - timedelta(minutes=1),
                end_at=datetime.now(UTC) + timedelta(minutes=1),
            )
        )
        assert summary.reported_costs.records[0].cost == "0.200000000"
        assert summary.reported_costs.records[0].status == "reported"
        await app.aclose()
        assert events[-1].payload == {"output": {"answer": 42}}
        assert len(authorization) == 1 and closed == [True]
        options = options_seen[0]
        assert options.setting_sources == [] and options.allowed_tools == []
        assert options.env["HOST_ONLY_SECRET"] == ""
        assert "private guest diagnostic" not in capsys.readouterr().err

    asyncio.run(scenario())


def test_docker_launcher_keeps_capabilities_out_of_host_argv(tmp_path, monkeypatch):
    import cayu.external_agents.environment as module

    async def scenario():
        output = tmp_path / "captured.json"
        docker = tmp_path / "docker"
        docker.write_text(
            f"#!{sys.executable}\nimport json, os, sys\n"
            f"open({str(output)!r}, 'w').write(json.dumps({{'argv': sys.argv, 'environment': dict(os.environ)}}))\n"
        )
        docker.chmod(0o700)
        closed = []

        async def close():
            closed.append(True)

        async def create(*args, **kwargs):
            return SimpleNamespace(
                container_id="immutable-container", docker_path=str(docker), close=close
            )

        async def network(*args):
            pass

        monkeypatch.setattr(module.DockerRunner, "create", create)
        monkeypatch.setattr(module, "_require_internal_network", network)
        monkeypatch.setattr(module.shutil, "which", lambda name: str(docker))
        monkeypatch.setenv("PRIVATE_SERVICE_KEY", "must-not-inherit")
        token = "run-capability-must-not-appear-in-argv"
        async with module.docker_agent_launch(
            image="test",
            cli="/usr/bin/agent",
            workspace=tmp_path,
            network="internal",
            environment={"CAYU_MODEL_PROXY_TOKEN": token},
        ) as launch:
            process = await asyncio.create_subprocess_exec(
                launch.executable, "--version", env={**os.environ, **launch.environment}
            )
            assert await process.wait() == 0
            captured = json.loads(output.read_text())
            assert token not in str(captured["argv"])
            assert token not in Path(launch.executable).read_text()
            assert captured["environment"]["CAYU_MODEL_PROXY_TOKEN"] == token
            assert "PRIVATE_SERVICE_KEY" not in captured["environment"]
            assert captured["argv"][captured["argv"].index("-e") + 1] == "CAYU_MODEL_PROXY_TOKEN"
            assert "env -i" in str(captured["argv"])
        assert closed == [True]

    asyncio.run(scenario())


def test_claude_stream_close_joins_sdk_disconnect_before_launcher_cleanup(monkeypatch):
    import cayu.external_agents.claude as module

    async def scenario():
        disconnecting, release = asyncio.Event(), asyncio.Event()
        order = []
        sdk_owner = []

        @dataclass
        class Text:
            text: str

        @dataclass
        class Assistant:
            content: list
            uuid: str = "one"

        class Client:
            def __init__(self, *, options):
                pass

            async def __aenter__(self):
                sdk_owner.append(asyncio.current_task())
                return self

            async def __aexit__(self, *args):
                assert asyncio.current_task() is sdk_owner[0]
                disconnecting.set()
                await release.wait()
                order.append("sdk")

            async def query(self, prompt):
                pass

            async def receive_response(self):
                yield Assistant([Text("first")])
                await asyncio.Event().wait()

        sdk = SimpleNamespace(
            ClaudeSDKClient=Client,
            ClaudeAgentOptions=SimpleNamespace,
            HookMatcher=SimpleNamespace,
            AssistantMessage=Assistant,
            TextBlock=Text,
            ToolUseBlock=type("ToolUse", (), {}),
        )
        monkeypatch.setattr(module, "import_module", lambda name: sdk)

        async def close():
            order.append("launcher")

        @asynccontextmanager
        async def launch(context):
            yield ExternalAgentLaunch("/sandbox/claude", "/checkout", close)

        adapter = ClaudeAgentAdapter(model="claude", launch=launch)
        stream = adapter.stream(ExternalAgentContext("test", "prompt", ExecutionDeadline.after(20)))
        assert (await anext(stream)).kind == "text"
        closing = asyncio.create_task(stream.aclose())
        await asyncio.wait_for(disconnecting.wait(), 5)
        assert not closing.done() and not order
        release.set()
        await asyncio.wait_for(closing, 5)
        assert order == ["sdk", "launcher"]
        assert sdk_owner[0].done()

    asyncio.run(scenario())


@pytest.mark.parametrize("sandbox", ["read-only", "external"])
def test_codex_records_incremental_openai_usage_and_only_final_answer(monkeypatch, sandbox):
    import cayu.external_agents.codex as module

    def notice(method, payload):
        return SimpleNamespace(
            method=method, payload=SimpleNamespace(model_dump=lambda **kwargs: payload)
        )

    notifications = [
        notice(
            "thread/tokenUsage/updated",
            {
                "tokenUsage": {
                    "total": {"inputTokens": 100, "outputTokens": 20, "cachedInputTokens": 40}
                }
            },
        ),
        notice(
            "thread/tokenUsage/updated",
            {
                "tokenUsage": {
                    "total": {"inputTokens": 150, "outputTokens": 30, "cachedInputTokens": 60}
                }
            },
        ),
        notice(
            "item/completed",
            {"item": {"type": "agentMessage", "phase": "commentary", "text": "thinking"}},
        ),
        notice(
            "item/completed",
            {"item": {"type": "agentMessage", "phase": "final_answer", "text": '{"answer":42}'}},
        ),
        notice("turn/completed", {"turn": {"status": "completed"}}),
    ]
    clients = []
    closed = []

    class Client:
        def __init__(self, config, approval_handler):
            self.config, self.approval = config, approval_handler
            self.closed = False
            clients.append(self)

        def start(self):
            self.start_thread = threading.get_ident()

        def initialize(self):
            assert self.approval("unknown/action", {}) == {"decision": "decline"}
            assert self.approval("item/commandExecution/requestApproval", {"itemId": "cmd"}) == {
                "decision": "decline"
            }

        def thread_start(self, params):
            assert params["ephemeral"] is True and params["approvalPolicy"] == "untrusted"
            assert params["sandbox"] == ("danger-full-access" if sandbox == "external" else sandbox)
            return SimpleNamespace(thread=SimpleNamespace(id="thread"))

        def turn_start(self, *args):
            if sandbox == "external":
                assert args[2]["sandboxPolicy"] == {
                    "type": "externalSandbox",
                    "networkAccess": "enabled",
                }
            else:
                assert "sandboxPolicy" not in args[2]
            return SimpleNamespace(turn=SimpleNamespace(id="turn"))

        def next_turn_notification(self, turn):
            return notifications.pop(0)

        def close(self):
            self.closed = True

    monkeypatch.setattr(
        module,
        "import_module",
        lambda name: SimpleNamespace(CodexClient=Client, CodexConfig=SimpleNamespace),
    )

    async def scenario():
        async def close():
            closed.append(True)

        @asynccontextmanager
        async def launch(context):
            yield ExternalAgentLaunch("/sandbox/codex", "/checkout", close)

        adapter = CodexAgentAdapter(
            model="codex", base_url="http://gateway/models/openai", launch=launch, sandbox=sandbox
        )
        events = [
            e
            async for e in adapter.stream(
                ExternalAgentContext(
                    "session", "review", ExecutionDeadline.after(20), {"type": "object"}
                )
            )
        ]
        usage = [e.payload for e in events if e.kind == "usage"]
        assert sum(u["input_tokens"] for u in usage) == 150
        assert sum(u["input_tokens_details"]["cached_tokens"] for u in usage) == 60
        assert all("cache_read_input_tokens" not in u for u in usage)
        assert events[-1].payload["output"] == {"answer": 42}
        assert clients[0].start_thread == threading.get_ident()
        assert clients[0].closed and closed == [True]

    asyncio.run(scenario())


@pytest.mark.parametrize("phase", ["initialize", "approval"])
def test_codex_cancellation_joins_sdk_waiters_and_permission_callbacks(monkeypatch, phase):
    import cayu.external_agents.codex as module

    async def scenario():
        entered, authorized_closed = asyncio.Event(), asyncio.Event()
        stopped = threading.Event()
        closed = []
        loop = asyncio.get_running_loop()

        class Client:
            def __init__(self, config, approval_handler):
                self.approval = approval_handler

            def start(self):
                pass

            def initialize(self):
                if phase == "approval":
                    assert self.approval(
                        "item/commandExecution/requestApproval", {"itemId": "pending"}
                    ) == {"decision": "decline"}
                else:
                    loop.call_soon_threadsafe(entered.set)
                    stopped.wait(10)
                raise RuntimeError("transport closed")

            def close(self):
                stopped.set()

        monkeypatch.setattr(
            module,
            "import_module",
            lambda name: SimpleNamespace(CodexClient=Client, CodexConfig=SimpleNamespace),
        )

        async def close():
            closed.append(True)

        @asynccontextmanager
        async def launch(context):
            yield ExternalAgentLaunch("/sandbox/codex", "/checkout", close)

        async def authorize(*args):
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                await asyncio.sleep(0)
                authorized_closed.set()

        adapter = CodexAgentAdapter(model="codex", base_url="http://gateway", launch=launch)

        async def run():
            return [
                e
                async for e in adapter.stream(
                    ExternalAgentContext(
                        "session", "review", ExecutionDeadline.after(20), authorize=authorize
                    )
                )
            ]

        task = asyncio.create_task(run())
        await asyncio.wait_for(entered.wait(), 5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 5)
        assert stopped.is_set() and closed == [True]
        if phase == "approval":
            assert authorized_closed.is_set()

    asyncio.run(scenario())


@pytest.mark.parametrize("deadline", [False, True])
@pytest.mark.parametrize("guest_cleanup_fails", [False, True])
def test_codex_guest_teardown_unblocks_a_buffered_sdk_write(
    monkeypatch, deadline, guest_cleanup_fails
):
    import cayu.external_agents.codex as module
    from cayu.external_agents._cleanup import ExternalCleanupIncomplete

    async def scenario():
        entered = asyncio.Event()
        stopped = threading.Event()
        buffer_lock = threading.Lock()
        watchdog_fired = threading.Event()
        actions = []
        loop = asyncio.get_running_loop()

        class Client:
            def __init__(self, config, approval_handler):
                pass

            def start(self):
                self._proc = SimpleNamespace(kill=stopped.set, wait=stopped.wait)

            def initialize(self):
                with buffer_lock:
                    loop.call_soon_threadsafe(entered.set)
                    stopped.wait()
                actions.append("writer joined")
                raise BrokenPipeError("guest stopped")

            def close(self):
                with buffer_lock:
                    actions.append("sdk closed")

        monkeypatch.setattr(
            module,
            "import_module",
            lambda name: SimpleNamespace(CodexClient=Client, CodexConfig=SimpleNamespace),
        )

        async def close():
            actions.append("guest closed")
            if guest_cleanup_fails and actions.count("guest closed") == 1:
                raise OSError("guest removal must be retried")
            stopped.set()

        @asynccontextmanager
        async def launch(context):
            yield ExternalAgentLaunch("/sandbox/codex", "/checkout", close)

        adapter = CodexAgentAdapter(model="codex", base_url="http://gateway", launch=launch)

        async def run():
            async with asyncio.timeout(0.1 if deadline else None):
                return [
                    event
                    async for event in adapter.stream(
                        ExternalAgentContext(
                            "session", "x" * (2 * 1024 * 1024), ExecutionDeadline.after(10)
                        )
                    )
                ]

        def unblock_failed_test():
            watchdog_fired.set()
            stopped.set()

        # Bound the regression's blocked worker even when testing the old order.
        watchdog = threading.Timer(3, unblock_failed_test)
        watchdog.start()
        try:
            task = asyncio.create_task(run())
            await asyncio.wait_for(entered.wait(), 2)
            if not deadline:
                task.cancel()
            expected = (
                ExternalCleanupIncomplete
                if guest_cleanup_fails
                else TimeoutError
                if deadline
                else asyncio.CancelledError
            )
            with pytest.raises(expected) as caught:
                await task
            assert not watchdog_fired.is_set()
            assert actions.count("guest closed") == actions.count("sdk closed") == 1
            assert actions.index("guest closed") < actions.index("sdk closed")
            assert "writer joined" in actions
            if guest_cleanup_fails:
                await caught.value.retry()
                assert actions.count("guest closed") == 2
                assert actions.count("sdk closed") == 1
        finally:
            stopped.set()
            watchdog.cancel()
            watchdog.join()

    asyncio.run(scenario())

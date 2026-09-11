from __future__ import annotations

import asyncio
import json

import pytest
from pydantic import SecretStr
from tests.core.test_tool_round_execution_identities import _SequencedProvider

from cayu import (
    AgentSpec,
    CayuApp,
    ExecutionProfileBehaviorIdentity,
    InMemorySessionStore,
    Message,
    PostgresSessionStore,
    ResumeRequest,
    RunRequest,
    SessionQuery,
    SessionStatus,
    SQLiteSessionStore,
    SubagentSpec,
    SubagentTool,
    ToolApprovalDecision,
)
from cayu.core import ToolResultPart
from cayu.core.runtime_authority import SessionRunFenced
from cayu.providers import ModelStreamEvent
from cayu.runtime import (
    IncompleteSessionRecoveryRequest,
    IncompleteSessionsRecoveryRequest,
    ToolApprovalRequest,
    UserInputResponse,
)
from cayu.runtime._session_control import SessionControl
from cayu.runtime._tool_effect_state import ToolEffectStateOwner
from cayu.runtime.hooks import BeforeToolCallDecision, RuntimeHook
from cayu.runtime.public_authority import PublicAuthorityAliasCodec, PublicAuthorityAliasKeyring
from cayu.runtime.tool_policy import AlwaysRequireApprovalToolPolicy
from cayu.storage.migrations import SchemaMode
from cayu.tools.user_input import UserInputTool
from cayu.vaults import SecretRedactor


def _identity(name):
    return ExecutionProfileBehaviorIdentity(
        name=f"tests:foreground-recovery:{name}",
        behavior_version="1",
        implementation_version="1",
    )


class _Provider(_SequencedProvider):
    @property
    def execution_profile_identity(self):
        return _identity("provider")


class _TaskModifier(RuntimeHook):
    @property
    def execution_profile_identity(self):
        return _identity("task-modifier")

    async def before_tool_call(self, context):
        return BeforeToolCallDecision(
            action="proceed_modified",
            modified_arguments={**context.arguments, "task": "modified task"},
        )


def _app(
    store,
    provider,
    *,
    modify_task=False,
    secret_redactor=None,
    result_max_chars=12,
    gate="ordinary",
    child_tools=(),
):
    app = CayuApp(session_store=store, secret_redactor=secret_redactor, enable_logging=False)
    app.register_provider(provider, default=True)
    tool = SubagentTool(
        app,
        agents={"child": SubagentSpec(agent_name="child", result_max_chars=result_max_chars)},
        execution_profile_identity=_identity("subagent"),
    )
    app.register_agent(
        AgentSpec(name="parent", model="test"),
        tools=[tool, UserInputTool()] if gate == "user_input" else [tool],
        runtime_hooks=[_TaskModifier()] if modify_task else [],
        tool_policy=AlwaysRequireApprovalToolPolicy(tools=["subagent"])
        if gate == "approval"
        else None,
    )
    app.register_agent(AgentSpec(name="child", model="test"), tools=list(child_tools))
    return app


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("modify_task", [False, True])
@pytest.mark.parametrize("answer", ["", "short answer", "Unicode 😀é日本語 " * 4, None])
def test_foreground_terminal_result_survives_lost_parent_return(
    tmp_path, monkeypatch, backend, answer, modify_task
):
    async def scenario():
        database = tmp_path / "sessions.sqlite"
        store = InMemorySessionStore() if backend == "memory" else SQLiteSessionStore(database)
        provider = _Provider(
            [
                [
                    ModelStreamEvent.tool_call(
                        id="spawn", name="subagent", arguments={"agent": "child", "task": "answer"}
                    ),
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ],
                [ModelStreamEvent.error("child failed")]
                if answer is None
                else [
                    ModelStreamEvent.text_delta(answer),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ],
                [
                    ModelStreamEvent.text_delta("parent done"),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ],
            ]
        )
        original = SubagentTool.run
        selected = []

        async def lose_return(tool, context, arguments):
            result = await original(tool, context, arguments)
            selected.append(result)
            raise ConnectionError("parent lost the child's completed return")

        try:
            with monkeypatch.context() as patch:
                patch.setattr(SubagentTool, "run", lose_return)
                first = [
                    event
                    async for event in _app(store, provider, modify_task=modify_task).run(
                        RunRequest(
                            session_id="parent",
                            agent_name="parent",
                            messages=[Message.text("user", "go")],
                        )
                    )
                ]
            assert first[-1].type.value == "session.interrupted"
            assert len(selected) == 1
            assert selected[0].is_error is (answer is None)
            assert len(provider.requests) == 2
            if isinstance(store, SQLiteSessionStore):
                await store.close()
                store = SQLiteSessionStore(database)
            recovered = [
                event
                async for event in _app(store, provider, modify_task=modify_task).resume(
                    ResumeRequest(session_id="parent", messages=[Message.text("user", "continue")])
                )
            ]
            assert recovered[-1].type.value == "session.completed"
            assert len(provider.requests) == 3  # Only parent continuation, never child replay.
            events = await store.load_events("parent")
            terminal = [
                event
                for event in events
                if event.type in {"tool.call.completed", "tool.call.failed"}
            ]
            assert len(terminal) == 1
            assert terminal[0].payload["result"] == selected[0].model_dump(mode="json")
            transcript = await store.load_transcript("parent")
            assert len([message for message in transcript if message.role.value == "tool"]) == 1
        finally:
            if isinstance(store, SQLiteSessionStore):
                await store.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
def test_foreground_recovery_rejects_conflicting_store_linkage_before_reading_result(
    tmp_path, monkeypatch, backend, request
):
    dsn = request.getfixturevalue("postgres_dsn") if backend == "postgres" else None
    parent_id = "linkage-parent"

    async def scenario():
        database = tmp_path / "linkage.sqlite"
        store = (
            InMemorySessionStore()
            if backend == "memory"
            else SQLiteSessionStore(database)
            if backend == "sqlite"
            else PostgresSessionStore(dsn, schema_mode=SchemaMode.CREATE)
        )
        provider = _Provider(
            [
                [
                    ModelStreamEvent.tool_call(
                        id="spawn", name="subagent", arguments={"agent": "child", "task": "answer"}
                    ),
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ],
                [
                    ModelStreamEvent.text_delta("child answer"),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ],
                [
                    ModelStreamEvent.text_delta("parent done"),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ],
            ]
        )
        original_run = SubagentTool.run
        selected = []

        async def lose_return(tool, context, arguments):
            selected.append(await original_run(tool, context, arguments))
            raise ConnectionError("parent lost terminal child return")

        async def resume():
            return [
                event
                async for event in _app(store, provider).resume(
                    ResumeRequest(session_id=parent_id, messages=[Message.text("user", "continue")])
                )
            ]

        try:
            with monkeypatch.context() as patch:
                patch.setattr(SubagentTool, "run", lose_return)
                initial = [
                    event
                    async for event in _app(store, provider).run(
                        RunRequest(
                            session_id=parent_id,
                            agent_name="parent",
                            messages=[Message.text("user", "go")],
                        )
                    )
                ]
            assert initial[-1].type == "session.interrupted"
            assert len(selected) == 1
            children = (
                await store.list_sessions(SessionQuery(parent_session_id=parent_id))
            ).sessions
            assert len(children) == 1
            child = children[0]
            metadata = child.model_dump(mode="json")["metadata"]
            query_transcript = type(store).query_transcript
            list_sessions = type(store).list_sessions
            child_reads = []
            conflicting_metadata = None

            async def conflicting_children(target_store, query):
                page = await list_sessions(target_store, query)
                if conflicting_metadata is None:
                    return page
                return page.model_copy(
                    update={
                        "sessions": [
                            item.model_copy(update={"metadata": conflicting_metadata})
                            if item.id == child.id
                            else item
                            for item in page.sessions
                        ]
                    }
                )

            async def observe_child_reads(target_store, query, **kwargs):
                if query.session_id == child.id:
                    child_reads.append(query)
                return await query_transcript(target_store, query, **kwargs)

            monkeypatch.setattr(type(store), "query_transcript", observe_child_reads)
            monkeypatch.setattr(type(store), "list_sessions", conflicting_children)
            for field, conflicting in (
                ("idempotency_key", "another-round-key"),
                ("tool_call_id", "another-call"),
                ("parent_session_id", "another-parent"),
                ("spawn_fingerprint", "sha256:" + "0" * 64),
                ("agent", "another-agent"),
                ("mode", True),
            ):
                conflicting_metadata = {
                    **metadata,
                    "subagent": {**metadata["subagent"], field: conflicting},
                }
                # Public metadata writes cannot forge runtime lineage. Exercise
                # conflicting backend readback separately, without weakening it.
                with pytest.raises(ValueError, match="runtime-owned"):
                    await store.update_metadata(child.id, conflicting_metadata)
                if isinstance(store, SQLiteSessionStore):
                    await store.close()
                    store = SQLiteSessionStore(database)
                rejected = await resume()
                assert rejected[-1].type == "session.interrupted", field
                assert not child_reads, field
                assert len(provider.requests) == 2, field
                assert not any(
                    event.type in {"tool.call.completed", "tool.call.failed"}
                    for event in await store.load_events(parent_id)
                ), field
            conflicting_metadata = None
            assert await store.load(child.id) == child
            accepted = await resume()
            assert accepted[-1].type == "session.completed"
            assert child_reads
            assert len(provider.requests) == 3
            terminals = [
                event
                for event in await store.load_events(parent_id)
                if event.type in {"tool.call.completed", "tool.call.failed"}
            ]
            assert len(terminals) == 1
            assert terminals[0].payload["result"] == selected[0].model_dump(mode="json")
        finally:
            if isinstance(store, (SQLiteSessionStore, PostgresSessionStore)):
                await store.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("child_failed", [False, True])
@pytest.mark.parametrize("gate", ["ordinary", "approval", "user_input"])
def test_reconstructed_foreground_result_preserves_redaction_and_taint(
    tmp_path, monkeypatch, backend, child_failed, gate, caplog, capsys, recwarn
):
    secret = "FOREGROUND_CREDENTIAL_CANARY_484"
    answer = f"SAFE_CHILD_ANSWER 😀 {secret} 日本語"

    async def scenario():
        codec = PublicAuthorityAliasCodec(
            PublicAuthorityAliasKeyring(active_key_id="test", keys={"test": SecretStr("A" * 43)})
        )

        def make_store():
            return (
                InMemorySessionStore(public_authority_alias_codec=codec)
                if backend == "memory"
                else SQLiteSessionStore(
                    tmp_path / "secrets.sqlite", public_authority_alias_codec=codec
                )
            )

        store = make_store()
        calls = [
            ModelStreamEvent.tool_call(
                id="spawn", name="subagent", arguments={"agent": "child", "task": "answer"}
            )
        ]
        if gate == "user_input":
            calls.append(
                ModelStreamEvent.tool_call(
                    id="question", name="ask_user", arguments={"question": "Continue?"}
                )
            )
        provider = _Provider(
            [
                [
                    *calls,
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ],
                [ModelStreamEvent.error(answer)]
                if child_failed
                else [
                    ModelStreamEvent.text_delta(answer),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ],
                [
                    ModelStreamEvent.text_delta("parent done"),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ],
            ]
        )

        def build_app():
            return _app(
                store,
                provider,
                secret_redactor=SecretRedactor(secret),
                result_max_chars=64,
                gate=gate,
            )

        original = SubagentTool.run
        selected = []

        async def lose_return(tool, context, arguments):
            result = await original(tool, context, arguments)
            selected.append(result.model_dump(mode="json"))
            raise ConnectionError("parent lost completed child return")

        try:
            with monkeypatch.context() as patch:
                patch.setattr(SubagentTool, "run", lose_return)
                initial_app = build_app()
                first = [
                    event
                    async for event in initial_app.run(
                        RunRequest(
                            session_id="parent",
                            agent_name="parent",
                            messages=[Message.text("user", "go")],
                            metadata={"cayu:taint_labels": ["untrusted_web"]},
                        )
                    )
                ]
                if gate == "approval":
                    approval = next(
                        event for event in first if event.type == "tool.call.approval_requested"
                    )
                    resolution = ToolApprovalRequest(
                        session_id="parent",
                        approval_id=approval.payload["approval"]["approval_id"],
                        tool_round_id=approval.payload["tool_round_id"],
                        tool_call_id=approval.payload["tool_call_id"],
                        decision=ToolApprovalDecision.APPROVE,
                    )
                    first.extend(
                        [event async for event in initial_app.resolve_tool_approval(resolution)]
                    )
                elif gate == "user_input":
                    awaiting = next(
                        event for event in first if event.type == "session.awaiting_user_input"
                    )
                    response = UserInputResponse(
                        session_id="parent", input_id=awaiting.payload["input_id"], answer="yes"
                    )
                    first.extend(
                        [event async for event in initial_app.resolve_user_input(response)]
                    )
            assert first[-1].type == "session.interrupted"
            assert len(selected) == 1
            child_id = selected[0]["structured"]["child_session_id"]
            child = await store.load(child_id)
            assert child is not None
            assert child.metadata["cayu:taint_labels"] == ["untrusted_web"]
            assert child.parent_session_id == "parent"
            if backend == "sqlite":
                await store.close()
                store = make_store()
            recovered_app = build_app()
            continuation = (
                recovered_app.resolve_tool_approval(resolution)
                if gate == "approval"
                else recovered_app.resolve_user_input(response)
                if gate == "user_input"
                else recovered_app.resume(
                    ResumeRequest(session_id="parent", messages=[Message.text("user", "continue")])
                )
            )
            recovered = [event async for event in continuation]
            assert recovered[-1].type == "session.completed"
            assert len(provider.requests) == 3
            assert await store.load(child_id) == child
            parent = await store.load("parent")
            assert parent is not None and parent.metadata["cayu:taint_labels"] == ["untrusted_web"]
            events = await store.load_events("parent")
            terminals = [
                event
                for event in events
                if event.tool_name == "subagent"
                and event.type in {"tool.call.completed", "tool.call.failed"}
            ]
            assert len(terminals) == 1 and terminals[0].payload["result"] == selected[0]
            effect = await ToolEffectStateOwner(store).resolve_call(
                parent,
                tool_round_id=terminals[0].payload["tool_round_id"],
                tool_call_id=terminals[0].payload["tool_call_id"],
            )
            assert effect is not None and effect.terminal is not None
            assert effect.terminal.event_id == terminals[0].id
            assert secret not in effect.model_dump_json()
            assert selected[0]["is_error"] is child_failed
            assert "SAFE_CHILD_ANSWER" in selected[0]["content"]
            surfaces = [
                *first,
                *recovered,
                *events,
                *await store.load_events(child_id),
                *await store.load_transcript("parent"),
                *await store.load_transcript(child_id),
            ]
            assert secret not in json.dumps([value.model_dump(mode="json") for value in surfaces])
            assert secret not in json.dumps(selected)
            generic_recovery = [
                event
                for event in recovered
                if str(event.type).startswith(("recovery.", "session."))
            ]
            assert "SAFE_CHILD_ANSWER" not in json.dumps(
                [event.model_dump(mode="json") for event in generic_recovery]
            )
        finally:
            if backend == "sqlite":
                await store.close()

    asyncio.run(scenario())
    captured = capsys.readouterr()
    assert secret not in caplog.text + captured.out + captured.err
    assert all(secret not in str(warning.message) for warning in recwarn)


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_foreground_recovery_retains_mcp_only_secret_redaction_after_reconstruction(
    backend, tmp_path, monkeypatch, caplog, capsys, recwarn
):
    from tests.core.test_mcp import FakeMcpSession, _fake_server_spec, _fake_tool_definitions

    from cayu.mcp.base import McpToolResult
    from cayu.mcp.tools import McpToolAdapter, McpToolset
    from cayu.vaults import REDACTED_SECRET

    secret = "foreground-mcp-transport-only-credential"

    async def scenario():
        database = tmp_path / "mcp-child.sqlite"
        store = InMemorySessionStore() if backend == "memory" else SQLiteSessionStore(database)
        calls = []
        definitions = _fake_tool_definitions("echo")
        provider = _Provider(
            [
                [
                    ModelStreamEvent.tool_call(
                        id="spawn", name="subagent", arguments={"agent": "child", "task": "answer"}
                    ),
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ],
                [
                    ModelStreamEvent.tool_call(
                        id="echo", name="mcp__local-mcp__echo", arguments={"text": "go"}
                    ),
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ],
                [
                    ModelStreamEvent.text_delta("safe child answer"),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ],
                [
                    ModelStreamEvent.text_delta("parent done"),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ],
            ]
        )

        def build_app(*, reconstructed=False):
            class Session(FakeMcpSession):
                @property
                def secret_redactor(self):
                    return SecretRedactor() if reconstructed else SecretRedactor(secret)

                async def call_tool(self, name, arguments):
                    calls.append((name, arguments))
                    return McpToolResult(
                        content=[{"type": "text", "text": secret}],
                        structured_content={"echo": secret},
                    )

            adapter = McpToolAdapter(
                toolset=McpToolset(
                    server=_fake_server_spec().model_copy(
                        update={"connection_id": "foreground-child-mcp"}
                    ),
                    session=Session(definitions=definitions),
                    definitions=definitions,
                ),
                definition=definitions[0],
            )
            adapter.spec = adapter.spec.model_copy(
                update={"execution_profile_identity": _identity("child-mcp")}
            )
            return _app(store, provider, secret_redactor=SecretRedactor(), child_tools=[adapter])

        selected = []
        run = SubagentTool.run

        async def lose_return(tool, context, arguments):
            selected.append(await run(tool, context, arguments))
            raise ConnectionError("parent lost sanitized child result")

        try:
            with monkeypatch.context() as patch:
                patch.setattr(SubagentTool, "run", lose_return)
                initial = [
                    event
                    async for event in build_app().run(
                        RunRequest(
                            session_id="parent",
                            agent_name="parent",
                            messages=[Message.text("user", "go")],
                        )
                    )
                ]
            assert initial[-1].type == "session.interrupted"
            assert len(calls) == 1 and len(provider.requests) == 3, selected
            child = (await store.list_sessions(SessionQuery(parent_session_id="parent"))).sessions[
                0
            ]
            child_events = await store.load_events(child.id)
            tool_event = next(
                event
                for event in child_events
                if event.tool_name == "mcp__local-mcp__echo" and event.type == "tool.call.completed"
            )
            assert REDACTED_SECRET in tool_event.payload["result"]["content"]
            assert secret not in json.dumps(
                [event.model_dump(mode="json") for event in child_events]
            )
            if isinstance(store, SQLiteSessionStore):
                await store.close()
                store = SQLiteSessionStore(database)
            recovered = [
                event
                async for event in build_app(reconstructed=True).resume(
                    ResumeRequest(session_id="parent", messages=[Message.text("user", "continue")])
                )
            ]
            assert recovered[-1].type == "session.completed"
            assert len(calls) == 1 and len(provider.requests) == 4
            terminal = [
                event
                for event in await store.load_events("parent")
                if event.type in {"tool.call.completed", "tool.call.failed"}
            ]
            assert len(terminal) == 1
            assert terminal[0].payload["result"] == selected[0].model_dump(mode="json")
            surfaces = [
                *initial,
                *recovered,
                *await store.load_events("parent"),
                *await store.load_events(child.id),
                *await store.load_transcript("parent"),
                *await store.load_transcript(child.id),
            ]
            assert secret not in json.dumps([value.model_dump(mode="json") for value in surfaces])
            assert secret not in json.dumps(
                [request.model_dump(mode="json") for request in provider.requests]
            )
        finally:
            if isinstance(store, SQLiteSessionStore):
                await store.close()

    asyncio.run(scenario())
    captured = capsys.readouterr()
    assert secret not in caplog.text + captured.out + captured.err
    assert all(secret not in str(warning.message) for warning in recwarn)


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
def test_foreground_recovery_claim_fences_competing_application(
    backend, tmp_path, monkeypatch, request
):
    dsn = request.getfixturevalue("postgres_dsn") if backend == "postgres" else None

    async def scenario():
        def new_store():
            if backend == "memory":
                return InMemorySessionStore()
            if backend == "sqlite":
                return SQLiteSessionStore(tmp_path / "contended.sqlite")
            return PostgresSessionStore(dsn, schema_mode=SchemaMode.CREATE)

        store = new_store()
        second_store = store if backend == "memory" else new_store()
        provider = _Provider(
            [
                [
                    ModelStreamEvent.tool_call(
                        id="spawn", name="subagent", arguments={"agent": "child", "task": "answer"}
                    ),
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ],
                [
                    ModelStreamEvent.text_delta("saved answer"),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ],
                [
                    ModelStreamEvent.text_delta("parent done"),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ],
            ]
        )
        selected = []
        original_run = SubagentTool.run
        first_task = None
        release = asyncio.Event()

        async def lose_return(tool, context, arguments):
            result = await original_run(tool, context, arguments)
            selected.append(result.model_dump(mode="json"))
            raise ConnectionError("parent lost completed child return")

        try:
            with monkeypatch.context() as patch:
                patch.setattr(SubagentTool, "run", lose_return)
                events = [
                    event
                    async for event in _app(store, provider).run(
                        RunRequest(
                            session_id="parent",
                            agent_name="parent",
                            messages=[Message.text("user", "go")],
                        )
                    )
                ]
            assert events[-1].type == "session.interrupted"
            assert len(selected) == 1 and len(provider.requests) == 2
            claimed = asyncio.Event()
            reserve = type(store).reserve_stalled_run_recovery

            async def hold_claim(target, *args, **kwargs):
                result = await reserve(target, *args, **kwargs)
                if result is not None and not claimed.is_set():
                    claimed.set()
                    await release.wait()
                return result

            monkeypatch.setattr(type(store), "reserve_stalled_run_recovery", hold_claim)
            first_app, second_app = _app(store, provider), _app(second_store, provider)
            recovery = IncompleteSessionRecoveryRequest(session_id="parent")
            first_task = asyncio.create_task(first_app.recover_incomplete_session(recovery))
            await asyncio.wait_for(claimed.wait(), timeout=15)
            competing = await asyncio.wait_for(
                second_app.recover_incomplete_session(recovery), timeout=15
            )
            assert competing.actions == ("skipped_active",)
            assert not any(
                event.type in {"tool.call.completed", "tool.call.failed"}
                for event in await second_store.load_events("parent")
            )
            release.set()
            await asyncio.wait_for(first_task, timeout=30)
            first_task = None
            terminals = [
                event
                for event in await store.load_events("parent")
                if event.type in {"tool.call.completed", "tool.call.failed"}
            ]
            assert len(terminals) == 1 and terminals[0].payload["result"] == selected[0]
            await second_app.recover_incomplete_session(recovery)
            continued = [
                event
                async for event in second_app.resume(
                    ResumeRequest(session_id="parent", messages=[Message.text("user", "continue")])
                )
            ]
            assert continued[-1].type == "session.completed"
            assert len(provider.requests) == 3
            assert [
                event
                for event in await second_store.load_events("parent")
                if event.type in {"tool.call.completed", "tool.call.failed"}
            ] == terminals
            assert (
                sum(
                    message.role == "tool"
                    for message in await second_store.load_transcript("parent")
                )
                == 1
            )
        finally:
            release.set()
            if first_task is not None:
                first_task.cancel()
                await asyncio.gather(first_task, return_exceptions=True)
            if backend != "memory":
                await store.close()
                await second_store.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_foreground_round_retry_preserves_committed_sibling(tmp_path, monkeypatch, backend):
    async def scenario():
        database = tmp_path / "siblings.sqlite"
        store = InMemorySessionStore() if backend == "memory" else SQLiteSessionStore(database)
        provider = _Provider(
            [
                [
                    *[
                        ModelStreamEvent.tool_call(
                            id=call, name="subagent", arguments={"agent": "child", "task": call}
                        )
                        for call in ("first", "second")
                    ],
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ],
                [ModelStreamEvent.text_delta("first answer"), ModelStreamEvent.completed()],
                [ModelStreamEvent.text_delta("second answer"), ModelStreamEvent.completed()],
                [ModelStreamEvent.text_delta("parent done"), ModelStreamEvent.completed()],
            ]
        )
        original_run = SubagentTool.run
        selected = {}

        async def lose_second_return(tool, context, arguments):
            result = await original_run(tool, context, arguments)
            call = context.metadata["tool_call_id"]
            selected[call] = result.model_dump(mode="json")
            if call == "second":
                raise ConnectionError("second child's return lost")
            return result

        async def terminals():
            return [
                event
                for event in await store.load_events("parent")
                if event.type in {"tool.call.completed", "tool.call.failed"}
            ]

        try:
            with monkeypatch.context() as patch:
                patch.setattr(SubagentTool, "run", lose_second_return)
                first = [
                    event
                    async for event in _app(store, provider).run(
                        RunRequest(
                            session_id="parent",
                            agent_name="parent",
                            messages=[Message.text("user", "go")],
                        )
                    )
                ]
            assert first[-1].type == "session.interrupted"
            assert set(selected) == {"first", "second"}
            assert len(provider.requests) == 3
            # The first result is staged with the interrupted round, not yet
            # published. Recovery must retain its progress if the second fails.
            assert await terminals() == []
            publish = type(store).publish_session_operation
            fault_fired = False

            async def fail_second_publication(target, session_id, **kwargs):
                nonlocal fault_fired
                if not fault_fired and any(
                    event.type in {"tool.call.completed", "tool.call.failed"}
                    and event.payload.get("tool_call_id") == "second"
                    for event in kwargs.get("events", ())
                ):
                    fault_fired = True
                    raise ConnectionError("second native terminal publication unavailable")
                return await publish(target, session_id, **kwargs)

            with monkeypatch.context() as patch:
                patch.setattr(type(store), "publish_session_operation", fail_second_publication)
                failed = [
                    event
                    async for event in _app(store, provider).resume(
                        ResumeRequest(
                            session_id="parent", messages=[Message.text("user", "continue")]
                        )
                    )
                ]
            assert fault_fired and failed[-1].type == "session.failed"
            prior = await terminals()
            assert len(prior) == 1 and prior[0].payload["tool_call_id"] == "first"
            assert prior[0].payload["result"] == selected["first"]
            assert len(provider.requests) == 3
            if backend == "sqlite":
                await store.close()
                store = SQLiteSessionStore(database)
            continued = [
                event
                async for event in _app(store, provider).resume(
                    ResumeRequest(session_id="parent", messages=[Message.text("user", "retry")])
                )
            ]
            assert continued[-1].type == "session.completed"
            final = await terminals()
            assert len(final) == 2 and final[0] == prior[0]
            assert {
                event.payload["tool_call_id"]: event.payload["result"] for event in final
            } == selected
            assert len(provider.requests) == 4
            parts = [
                part
                for message in await store.load_transcript("parent")
                for part in message.content
                if isinstance(part, ToolResultPart)
            ]
            assert len(parts) == 2 and {part.tool_call_id for part in parts} == {"first", "second"}
        finally:
            if backend == "sqlite":
                await store.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_foreground_loss_before_child_creation_retains_manual_obligation(
    tmp_path, monkeypatch, backend
):
    async def scenario():
        database = tmp_path / "missing-child.sqlite"
        store = InMemorySessionStore() if backend == "memory" else SQLiteSessionStore(database)
        provider = _Provider(
            [
                [
                    ModelStreamEvent.tool_call(
                        id="spawn", name="subagent", arguments={"agent": "child", "task": "answer"}
                    ),
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ]
            ]
        )

        async def lose_dispatch(tool, context, arguments):
            raise ConnectionError("foreground adapter lost before child creation")

        try:
            with monkeypatch.context() as patch:
                patch.setattr(SubagentTool, "run", lose_dispatch)
                first = [
                    event
                    async for event in _app(store, provider).run(
                        RunRequest(
                            session_id="parent",
                            agent_name="parent",
                            messages=[Message.text("user", "go")],
                        )
                    )
                ]
            assert first[-1].type == "session.interrupted"
            if isinstance(store, SQLiteSessionStore):
                await store.close()
                store = SQLiteSessionStore(database)
            app = _app(store, provider)
            recovery = await app.recover_incomplete_session(
                IncompleteSessionRecoveryRequest(session_id="parent")
            )
            assert "pending_tool_effect" in recovery.actions
            assert recovery.pending_subagent_session_ids == ()
            retried = [
                event
                async for event in app.resume(
                    ResumeRequest(session_id="parent", messages=[Message.text("user", "continue")])
                )
            ]
            assert retried[-1].type == "session.interrupted"
            assert len(provider.requests) == 1
            assert not (
                await store.list_sessions(SessionQuery(parent_session_id="parent"))
            ).sessions
            assert not any(
                event.type in {"tool.call.completed", "tool.call.failed"}
                for event in await store.load_events("parent")
            )
            checkpoint = await store.load_checkpoint("parent")
            assert checkpoint is not None and "pending_tool_round" in checkpoint
        finally:
            if isinstance(store, SQLiteSessionStore):
                await store.close()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("gate", "recovery_mode"),
    [
        (gate, mode)
        for gate in ("ordinary", "approval", "user_input")
        for mode in ("report", "continue", "redacted_ids")
    ]
    + [
        (gate, mode)
        for gate in ("ordinary", "approval", "user_input")
        for mode in ("ack_lost", "readback_lost", "cancelled_publication")
    ]
    + [("ordinary", "cancelled_classification_failure")],
)
@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_pending_foreground_child_is_reported_without_consuming_human_gate(
    gate, recovery_mode, backend, tmp_path, monkeypatch
):
    async def scenario():
        codec = (
            PublicAuthorityAliasCodec(
                PublicAuthorityAliasKeyring(
                    active_key_id="test", keys={"test": SecretStr("A" * 43)}
                )
            )
            if recovery_mode == "redacted_ids"
            else None
        )
        store = (
            InMemorySessionStore(public_authority_alias_codec=codec)
            if backend == "memory"
            else SQLiteSessionStore(tmp_path / "gated.sqlite", public_authority_alias_codec=codec)
        )
        retained_streams = []
        child_ids = []
        publication_committed = asyncio.Event()
        release_publication = asyncio.Event()
        continuation_task = None
        cancel_publication = recovery_mode in {
            "cancelled_publication",
            "cancelled_classification_failure",
        }
        classification_failure = OSError("interruption classification read failed")
        classification_failed = False
        calls = [
            ModelStreamEvent.tool_call(
                id="spawn", name="subagent", arguments={"agent": "child", "task": "answer"}
            )
        ]
        if gate == "user_input":
            calls.append(
                ModelStreamEvent.tool_call(
                    id="question", name="ask_user", arguments={"question": "Continue?"}
                )
            )
        provider = _Provider(
            [
                [*calls, ModelStreamEvent.completed({"finish_reason": "tool_calls"})],
                [
                    ModelStreamEvent.text_delta("parent done"),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ],
            ]
        )

        class LostChildStream:
            """Lose the adapter stream after real child creation; retain it for owned cleanup."""

            def __init__(self, app):
                self.app = app
                self.session_store = store

            async def run(self, request):
                stream = self.app.run(request)
                retained_streams.append(stream)
                child_ids.append(request.session_id)
                event = await anext(stream)
                assert event.type.value == "session.started"
                yield event
                raise ConnectionError("child stream transport lost after durable creation")

            def interrupt_session(self, request):
                return self.app.interrupt_session(request)

            async def _submit_durable_subagent(self, **kwargs):
                raise AssertionError("Foreground recovery must not submit durable tasks")

            async def _reconcile_durable_subagent(self, **kwargs):
                raise AssertionError("Foreground recovery must not reconcile durable tasks")

        def build_app():
            app = CayuApp(
                session_store=store,
                secret_redactor=SecretRedactor("cayu-child")
                if recovery_mode == "redacted_ids"
                else None,
                enable_logging=False,
            )
            app.register_provider(provider, default=True)
            subagent = SubagentTool(
                LostChildStream(app),
                agents={"child": SubagentSpec(agent_name="child")},
                execution_profile_identity=_identity("gated-subagent"),
            )
            app.register_agent(
                AgentSpec(name="parent", model="test"),
                tools=[subagent, UserInputTool()] if gate == "user_input" else [subagent],
                tool_policy=AlwaysRequireApprovalToolPolicy(tools=["subagent"])
                if gate == "approval"
                else None,
            )
            app.register_agent(AgentSpec(name="child", model="test"))
            return app

        try:
            app = build_app()
            events = [
                event
                async for event in app.run(
                    RunRequest(
                        session_id="gated-parent",
                        agent_name="parent",
                        messages=[Message.text("user", "go")],
                    )
                )
            ]
            if gate == "approval":
                approval = next(
                    event for event in events if event.type.value == "tool.call.approval_requested"
                )
                request = ToolApprovalRequest(
                    session_id="gated-parent",
                    approval_id=approval.payload["approval"]["approval_id"],
                    tool_round_id=approval.payload["tool_round_id"],
                    tool_call_id=approval.payload["tool_call_id"],
                    decision=ToolApprovalDecision.APPROVE,
                )
                _ = [event async for event in app.resolve_tool_approval(request)]
            elif gate == "user_input":
                awaiting = next(
                    event for event in events if event.type.value == "session.awaiting_user_input"
                )
                response = UserInputResponse(
                    session_id="gated-parent", input_id=awaiting.payload["input_id"], answer="yes"
                )
                _ = [event async for event in app.resolve_user_input(response)]
            assert len(child_ids) == 1
            child = await store.load(child_ids[0])
            assert child is not None
            assert child.status.value in {"pending", "running", "interrupting"}, child.status
            before = await store.load_checkpoint("gated-parent")
            recovered = await build_app().recover_incomplete_session(
                IncompleteSessionRecoveryRequest(session_id="gated-parent")
            )
            assert "pending_subagent" in recovered.actions
            expected_ids = tuple(
                build_app().project_session_id_for_exposure(value) for value in child_ids
            )
            assert recovered.pending_subagent_session_ids == expected_ids
            if recovery_mode == "redacted_ids":
                assert expected_ids != tuple(child_ids)
                assert "cayu-child" not in recovered.model_dump_json()
                if gate == "ordinary":
                    page = await build_app().recover_incomplete_sessions(
                        IncompleteSessionsRecoveryRequest(statuses={SessionStatus.INTERRUPTED})
                    )
                    assert len(page.results) == 1
                    assert page.results[0].pending_subagent_session_ids == expected_ids
                    assert "cayu-child" not in page.model_dump_json()
            if gate != "ordinary":
                assert (
                    recovered.pending_approval_id
                    if gate == "approval"
                    else recovered.pending_user_input_id
                ) is not None
            after = await store.load_checkpoint("gated-parent")
            if gate != "ordinary":
                gate_key = "pending_tool_approval" if gate == "approval" else "pending_user_input"
                assert after is not None and before is not None
                assert after[gate_key] == before[gate_key]
            assert len(provider.requests) == 1
            assert not any(
                event.type in {"tool.call.completed", "tool.call.failed"}
                and event.tool_name == "subagent"
                for event in await store.load_events("gated-parent")
            )
            retry_app = build_app()
            retry = (
                retry_app.resolve_tool_approval(request)
                if gate == "approval"
                else retry_app.resolve_user_input(response)
                if gate == "user_input"
                else retry_app.resume(
                    ResumeRequest(
                        session_id="gated-parent", messages=[Message.text("user", "continue")]
                    )
                )
            )
            still_pending = [event async for event in retry]
            assert still_pending[-1].type.value == "session.interrupted"
            if recovery_mode == "redacted_ids":
                assert "cayu-child" not in still_pending[-1].model_dump_json()
            else:
                assert still_pending[-1].payload["pending_subagent_session_ids"] == child_ids
            assert len(child_ids) == len(provider.requests) == 1
            if recovery_mode != "report":
                for stream in retained_streams:
                    await stream.aclose()
                retained_streams.clear()
                terminal_child = await store.load(child_ids[0])
                assert terminal_child is not None and terminal_child.status.value == "interrupted"
                if recovery_mode == "redacted_ids":
                    if isinstance(store, SQLiteSessionStore):
                        await store.close()
                        store = SQLiteSessionStore(
                            tmp_path / "gated.sqlite", public_authority_alias_codec=codec
                        )
                    child_recovery = await build_app().recover_incomplete_session(
                        IncompleteSessionRecoveryRequest(session_id=expected_ids[0])
                    )
                    assert child_recovery.session_id == expected_ids[0]
                faults = {"publication": False, "readback": False}
                if recovery_mode in {"ack_lost", "readback_lost"} or cancel_publication:
                    publish = type(store).publish_session_operation
                    readback = type(store).load_session_operation

                    async def lose_ack(target_store, session_id, **kwargs):
                        result = await publish(target_store, session_id, **kwargs)
                        if not faults["publication"] and any(
                            event.tool_name == "subagent"
                            and event.type in {"tool.call.completed", "tool.call.failed"}
                            for event in kwargs.get("events", ())
                        ):
                            faults["publication"] = True
                            if cancel_publication:
                                publication_committed.set()
                                await release_publication.wait()
                                return result
                            raise ConnectionError("native terminal acknowledgement lost")
                        return result

                    async def lose_readback(target_store, session_id, idempotency_key, **kwargs):
                        if (
                            recovery_mode == "readback_lost"
                            and faults["publication"]
                            and not faults["readback"]
                            and idempotency_key.startswith("tool-effect:v1:")
                        ):
                            faults["readback"] = True
                            raise ConnectionError("native terminal readback unavailable")
                        return await readback(target_store, session_id, idempotency_key, **kwargs)

                    monkeypatch.setattr(type(store), "publish_session_operation", lose_ack)
                    monkeypatch.setattr(type(store), "load_session_operation", lose_readback)
                if recovery_mode == "cancelled_classification_failure":
                    interrupt_requested = SessionControl.interrupt_requested

                    async def fail_classification(control, session_id):
                        nonlocal classification_failed
                        if publication_committed.is_set() and not classification_failed:
                            classification_failed = True
                            raise classification_failure
                        return await interrupt_requested(control, session_id)

                    monkeypatch.setattr(SessionControl, "interrupt_requested", fail_classification)
                continuation_app = build_app()

                def continue_stream(app):
                    return (
                        app.resolve_tool_approval(request)
                        if gate == "approval"
                        else app.resolve_user_input(response)
                        if gate == "user_input"
                        else app.resume(
                            ResumeRequest(
                                session_id="gated-parent",
                                messages=[Message.text("user", "continue")],
                            )
                        )
                    )

                selected_before_retry = None
                if cancel_publication:
                    cancellations = []

                    async def consume_continuation():
                        try:
                            return [event async for event in continue_stream(continuation_app)]
                        except asyncio.CancelledError as exc:
                            cancellations.append(exc)
                            raise

                    continuation_task = asyncio.create_task(consume_continuation())
                    await asyncio.wait_for(publication_committed.wait(), timeout=30)
                    assert continuation_task.cancel()
                    assert continuation_task.cancel()
                    assert continuation_task.cancelling() == 2
                    release_publication.set()
                    with pytest.raises(asyncio.CancelledError):
                        await continuation_task
                    assert continuation_task.cancelled()
                    assert continuation_task.cancelling() == 2
                    assert len(cancellations) == 1
                    if recovery_mode == "cancelled_classification_failure":
                        assert classification_failed
                        cleanup = cancellations[0].__cause__
                        assert isinstance(cleanup, ExceptionGroup)
                        assert len(cleanup.exceptions) == 1
                        assert isinstance(cleanup.exceptions[0], SessionRunFenced)
                        assert cleanup.__cause__ is classification_failure
                        assert classification_failure.__cause__ is None
                    assert len(provider.requests) == 1
                    selected_before_retry = [
                        event
                        for event in await store.load_events("gated-parent")
                        if event.tool_name == "subagent"
                        and event.type in {"tool.call.failed", "tool.call.completed"}
                    ]
                    assert len(selected_before_retry) == 1
                    if isinstance(store, SQLiteSessionStore):
                        await store.close()
                        store = SQLiteSessionStore(tmp_path / "gated.sqlite")
                    recovered_app = build_app()
                    await recovered_app.recover_incomplete_session(
                        IncompleteSessionRecoveryRequest(session_id="gated-parent")
                    )
                    continued = [event async for event in continue_stream(recovered_app)]
                else:
                    continued = [event async for event in continue_stream(continuation_app)]
                if recovery_mode == "readback_lost":
                    assert faults == {"publication": True, "readback": True}
                    assert len(provider.requests) == 1
                    assert continued[-1].type == (
                        "session.failed" if gate == "ordinary" else "session.interrupted"
                    )
                    types = continued[-1].payload["failure_evidence"]["exception_types"]
                    assert types.count("ConnectionError") == 2
                    assert types.count("ExceptionGroup") == 1
                    selected_before_retry = [
                        event
                        for event in await store.load_events("gated-parent")
                        if event.tool_name == "subagent"
                        and event.type in {"tool.call.failed", "tool.call.completed"}
                    ]
                    assert len(selected_before_retry) == 1
                    if isinstance(store, SQLiteSessionStore):
                        await store.close()
                        store = SQLiteSessionStore(tmp_path / "gated.sqlite")
                    continued = [event async for event in continue_stream(build_app())]
                if recovery_mode == "ack_lost":
                    assert faults["publication"], continued[-1].payload
                assert continued[-1].type.value == "session.completed", continued[-1].payload
                assert len(provider.requests) == 2  # Parent only; child never replayed.
                terminal = [
                    event
                    for event in await store.load_events("gated-parent")
                    if event.tool_name == "subagent"
                    and event.type in {"tool.call.failed", "tool.call.completed"}
                ]
                assert len(terminal) == 1
                if selected_before_retry is not None:
                    assert terminal == selected_before_retry
                assert terminal[0].payload["result"]["is_error"] is True
        finally:
            release_publication.set()
            if continuation_task is not None and not continuation_task.done():
                continuation_task.cancel()
                await asyncio.gather(continuation_task, return_exceptions=True)
            for stream in retained_streams:
                await stream.aclose()
            if isinstance(store, SQLiteSessionStore):
                await store.close()

    asyncio.run(scenario())

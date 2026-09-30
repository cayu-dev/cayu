from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from cayu import (
    AgentSpec,
    BeforeStopAction,
    BeforeStopDecision,
    CayuApp,
    ExecutionProfileBehaviorIdentity,
    InMemorySessionStore,
    LoopPolicy,
    Message,
    ModelStreamEvent,
    RequireFinalTool,
    ResumeRequest,
    RunRequest,
    ScriptedModelProvider,
    SessionStatus,
    SQLiteSessionStore,
    Tool,
    ToolApprovalDecision,
    ToolApprovalRequest,
    ToolContext,
    ToolPolicy,
    ToolPolicyDecision,
    ToolPolicyResult,
    ToolResult,
    ToolSpec,
)

EMPTY = [ModelStreamEvent.completed({"finish_reason": "stop"})]


def _call(name: str, call_id: str) -> list[ModelStreamEvent]:
    return [
        ModelStreamEvent.tool_call(name=name, arguments={}, id=call_id),
        ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
    ]


class _Tool(Tool):
    def __init__(self, name: str, *, failures: int = 0) -> None:
        self.spec = ToolSpec(
            name=name,
            description=f"Test tool {name}.",
            input_schema={"type": "object", "properties": {}},
            execution_profile_identity=ExecutionProfileBehaviorIdentity(
                name=f"tests:{name}", behavior_version="1", implementation_version="1"
            ),
        )
        super().__init__()
        self.failures = failures
        self.calls = 0

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        self.calls += 1
        if self.calls <= self.failures:
            return ToolResult(content="service unavailable", is_error=True)
        return ToolResult(content=f"{self.spec.name} done")


class _ApprovalPolicy(ToolPolicy):
    @property
    def execution_profile_identity(self):
        return ExecutionProfileBehaviorIdentity(
            name="tests:require-final-tool:approval",
            behavior_version="1",
            implementation_version="1",
        )

    async def authorize(self, request):
        return ToolPolicyResult(decision=ToolPolicyDecision.REQUIRE_APPROVAL, reason="approval")


def _app(store, provider, *tools, tool_policy=None):
    app = CayuApp(session_store=store, enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="support", model="scripted-model"),
        tools=list(tools) or [_Tool("find_orders"), _Tool("ask_customer")],
        tool_policy=tool_policy,
    )
    return app


async def _run(app, policy, session_id="s", **request):
    return [
        event
        async for event in app.run(
            RunRequest(
                agent_name="support",
                session_id=session_id,
                messages=[Message.text("user", "My order never arrived.")],
                loop_policies=(policy,),
                **request,
            )
        )
    ]


def _reminders(transcript, policy_text):
    return [
        message
        for message in transcript
        if message.role == "user" and message.content[0].text == policy_text
    ]


REMINDER = "You have not finished this turn. End it by calling one of: ask_customer."


def test_empty_stop_before_final_tool_gets_a_reminder_and_the_turn_completes() -> None:
    async def scenario():
        store = InMemorySessionStore()
        provider = ScriptedModelProvider(
            [_call("find_orders", "c1"), EMPTY, _call("ask_customer", "c2"), EMPTY]
        )
        events = await _run(_app(store, provider), RequireFinalTool(["ask_customer"]))
        return events, await store.load("s"), await store.load_transcript("s"), provider

    events, session, transcript, provider = asyncio.run(scenario())

    assert session.status == SessionStatus.COMPLETED
    assert len(provider.requests) == 4
    assert len(_reminders(transcript, REMINDER)) == 1
    selected = [e for e in events if e.type == "custom.loop.before_stop.selected"]
    assert [e.payload["action"] for e in selected] == ["continue"]
    assert selected[0].payload["metadata"]["reminder"] == 1


def test_stop_after_final_tool_completes_without_reminder() -> None:
    async def scenario():
        store = InMemorySessionStore()
        provider = ScriptedModelProvider([_call("ask_customer", "c1"), EMPTY])
        events = await _run(_app(store, provider), RequireFinalTool(["ask_customer"]))
        return events, await store.load("s"), await store.load_transcript("s"), provider

    events, session, transcript, provider = asyncio.run(scenario())

    assert session.status == SessionStatus.COMPLETED
    assert len(provider.requests) == 2
    assert not _reminders(transcript, REMINDER)
    assert not [e for e in events if e.type == "custom.loop.before_stop.selected"]


def test_text_answer_without_final_tool_also_gets_a_reminder() -> None:
    async def scenario():
        store = InMemorySessionStore()
        provider = ScriptedModelProvider(
            [
                [
                    ModelStreamEvent.text_delta("Your order is on its way."),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ],
                _call("ask_customer", "c1"),
                EMPTY,
            ]
        )
        await _run(_app(store, provider), RequireFinalTool(["ask_customer"]))
        return await store.load("s"), provider

    session, provider = asyncio.run(scenario())

    assert session.status == SessionStatus.COMPLETED
    assert len(provider.requests) == 3


def test_failed_final_tool_does_not_count() -> None:
    async def scenario():
        store = InMemorySessionStore()
        ask = _Tool("ask_customer", failures=1)
        provider = ScriptedModelProvider(
            [_call("ask_customer", "c1"), EMPTY, _call("ask_customer", "c2"), EMPTY]
        )
        await _run(_app(store, provider, ask), RequireFinalTool(["ask_customer"]))
        return await store.load("s"), await store.load_transcript("s"), ask

    session, transcript, ask = asyncio.run(scenario())

    assert session.status == SessionStatus.COMPLETED
    assert ask.calls == 2
    assert len(_reminders(transcript, REMINDER)) == 1


@pytest.mark.parametrize("max_reminders", [1, 2])
def test_exhausted_reminders_interrupt_with_a_reason(max_reminders) -> None:
    async def scenario():
        store = InMemorySessionStore()
        provider = ScriptedModelProvider([EMPTY] * (max_reminders + 1))
        policy = RequireFinalTool(["ask_customer"], max_reminders=max_reminders)
        events = await _run(_app(store, provider), policy)
        return events, await store.load("s"), await store.load_transcript("s"), provider

    events, session, transcript, provider = asyncio.run(scenario())

    assert session.status == SessionStatus.INTERRUPTED
    assert len(provider.requests) == max_reminders + 1
    assert len(_reminders(transcript, REMINDER)) == max_reminders
    interrupted = [e for e in events if e.type == "session.interrupted"][-1]
    assert "ask_customer" in interrupted.payload["reason"]
    assert interrupted.payload["policy_metadata"] == {
        "reminders": max_reminders,
        "tool_names": ["ask_customer"],
    }


def test_exhausted_reminders_can_fail_instead() -> None:
    async def scenario():
        store = InMemorySessionStore()
        provider = ScriptedModelProvider([EMPTY])
        policy = RequireFinalTool(
            ["ask_customer"], max_reminders=0, on_exhausted=BeforeStopAction.FAIL
        )
        events = await _run(_app(store, provider), policy)
        return events, await store.load("s")

    events, session = asyncio.run(scenario())

    assert session.status == SessionStatus.FAILED
    assert events[-1].type == "session.failed"
    assert "ask_customer" in events[-1].payload["error"]


def test_reminder_provenance_preserves_protocol_keys_under_secret_redaction() -> None:
    from cayu.runtime._checkpoint_redaction import durable_value_contains_secret
    from cayu.runtime._loop_policy_continuations import BEFORE_STOP_CONTINUATIONS_CHECKPOINT_KEY
    from cayu.vaults.redaction import SecretRedactor

    async def scenario():
        store = InMemorySessionStore()
        await _run(
            _app(store, ScriptedModelProvider([EMPTY, EMPTY])),
            RequireFinalTool(["ask_customer"], max_reminders=1),
        )
        checkpoint = await store.load_checkpoint("s")
        return {
            BEFORE_STOP_CONTINUATIONS_CHECKPOINT_KEY: checkpoint[
                BEFORE_STOP_CONTINUATIONS_CHECKPOINT_KEY
            ]
        }

    checkpoint = asyncio.run(scenario())
    marker = checkpoint[BEFORE_STOP_CONTINUATIONS_CHECKPOINT_KEY]
    for secret in (
        "before_stop_continuations",
        "runtime_authored_anchors",
        "anchor_transcript_index",
        "policy_sha256",
        marker["session_sha256"][:8],
        marker["policy_sha256"][:8],
        marker["runtime_authored_anchors"][0]["user_message_sha256"][:8],
    ):
        assert not durable_value_contains_secret(checkpoint, redactor=SecretRedactor(secret))
    marker["runtime_authored_anchors"][0]["user_message_sha256"] = "private-canary"
    assert durable_value_contains_secret(checkpoint, redactor=SecretRedactor("private-canary"))


def test_a_new_user_message_starts_a_new_turn() -> None:
    async def scenario():
        store = InMemorySessionStore()
        provider = ScriptedModelProvider([EMPTY, _call("ask_customer", "c1"), EMPTY])
        app = _app(store, provider)
        policy = RequireFinalTool(["ask_customer"], max_reminders=0)
        await _run(app, policy)
        first = (await store.load("s")).status
        events = [
            event
            async for event in app.resume(
                ResumeRequest(
                    session_id="s",
                    messages=[Message.text("user", "It was the boots.")],
                    loop_policies=(policy,),
                )
            )
        ]
        return first, (await store.load("s")).status, events

    first, second, events = asyncio.run(scenario())

    assert first == SessionStatus.INTERRUPTED
    assert second == SessionStatus.COMPLETED
    assert not [e for e in events if e.type == "custom.loop.before_stop.selected"]


@pytest.mark.parametrize("store_kind", ["memory", "sqlite"])
@pytest.mark.parametrize("max_reminders", [0, 1])
def test_user_message_matching_reminder_starts_a_new_turn(
    store_kind, max_reminders, tmp_path
) -> None:
    async def scenario():
        store = (
            InMemorySessionStore()
            if store_kind == "memory"
            else SQLiteSessionStore(tmp_path / "sessions.sqlite")
        )
        try:
            provider = ScriptedModelProvider(
                [EMPTY] * max_reminders
                + [_call("ask_customer", "c1"), EMPTY]
                + [EMPTY] * (max_reminders + 1)
            )
            app = _app(store, provider)
            policy = RequireFinalTool(
                ["ask_customer"], reminder="Please continue.", max_reminders=max_reminders
            )
            await _run(app, policy)
            assert (await store.load("s")).status == SessionStatus.COMPLETED
            events = [
                event
                async for event in app.resume(
                    ResumeRequest(
                        session_id="s",
                        messages=[Message.text("user", "Please continue.")],
                        loop_policies=(policy,),
                    )
                )
            ]
            assert (await store.load("s")).status == SessionStatus.INTERRUPTED
            assert events[-1].payload["policy_metadata"]["reminders"] == max_reminders
            assert len(provider.requests) == 3 + 2 * max_reminders
        finally:
            if isinstance(store, SQLiteSessionStore):
                await store.close()

    asyncio.run(scenario())


def test_another_policy_cannot_supply_the_final_tool_reminder() -> None:
    class ContinueOnce(LoopPolicy):
        async def before_stop(self, context):
            if context.step == 2:
                return BeforeStopDecision.continue_with(Message.text("user", REMINDER))
            return BeforeStopDecision.complete()

    async def scenario():
        store = InMemorySessionStore()
        provider = ScriptedModelProvider([_call("ask_customer", "c1"), EMPTY, EMPTY])
        app = _app(store, provider)
        events = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="support",
                    session_id="s",
                    messages=[Message.text("user", "My order never arrived.")],
                    loop_policies=(
                        ContinueOnce(),
                        RequireFinalTool(["ask_customer"], max_reminders=0),
                    ),
                )
            )
        ]
        assert (await store.load("s")).status == SessionStatus.INTERRUPTED
        assert events[-1].payload["policy_metadata"]["reminders"] == 0

    asyncio.run(scenario())


def test_initial_user_message_matching_reminder_does_not_consume_a_reminder() -> None:
    async def scenario():
        store = InMemorySessionStore()
        provider = ScriptedModelProvider([EMPTY, EMPTY])
        policy = RequireFinalTool(
            ["ask_customer"], reminder="My order never arrived.", max_reminders=1
        )
        events = await _run(_app(store, provider), policy)
        assert len(provider.requests) == 2
        assert (await store.load("s")).status == SessionStatus.INTERRUPTED
        assert events[-1].payload["policy_metadata"]["reminders"] == 1

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("names", "options", "error"),
    [
        ((), {}, ValueError),
        (("ask", "ask"), {}, ValueError),
        ("ask", {}, TypeError),
        (("ask",), {"max_reminders": -1}, ValueError),
        (("ask",), {"on_exhausted": BeforeStopAction.CONTINUE}, ValueError),
        (("ask",), {"reminder": " "}, ValueError),
    ],
)
def test_invalid_configuration_is_rejected(names, options, error) -> None:
    with pytest.raises(error):
        RequireFinalTool(names, **options)


def test_profile_identity_follows_configuration() -> None:
    one = RequireFinalTool(["ask_customer", "propose"])
    same = RequireFinalTool(("ask_customer", "propose"))
    other = RequireFinalTool(["ask_customer", "propose"], max_reminders=3)

    assert one.execution_profile_identity == same.execution_profile_identity
    assert one.execution_profile_identity != other.execution_profile_identity
    assert one.adoption_replay_identity != other.adoption_replay_identity


def test_reminders_survive_a_new_process(tmp_path) -> None:
    path = tmp_path / "sessions.sqlite"
    first = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import asyncio, json, sys
from cayu import SQLiteSessionStore, ScriptedModelProvider, RequireFinalTool
from tests.core.test_require_final_tool import _app, _run, _call, _ApprovalPolicy, EMPTY
async def run():
    store = SQLiteSessionStore(sys.argv[1])
    try:
        provider = ScriptedModelProvider([EMPTY, _call("find_orders", "c1")])
        events = await _run(
            _app(store, provider, tool_policy=_ApprovalPolicy()),
            RequireFinalTool(["ask_customer"], max_reminders=1),
        )
        approval = next(e for e in events if e.type == "tool.call.approval_requested")
        print(json.dumps({
            "approval_id": approval.payload["approval"]["approval_id"],
            "tool_round_id": approval.payload["tool_round_id"],
            "tool_call_id": approval.payload["tool_call_id"],
        }))
    finally:
        await store.close()
asyncio.run(run())
""",
            str(path),
        ],
        cwd=Path(__file__).resolve().parents[2],
        env={**os.environ, "PYTHONPATH": "src:."},
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    approval = json.loads(first.stdout)

    async def second_process():
        store = SQLiteSessionStore(path)
        try:
            provider = ScriptedModelProvider([EMPTY])
            app = _app(store, provider, tool_policy=_ApprovalPolicy())
            events = [
                event
                async for event in app.resolve_tool_approval(
                    ToolApprovalRequest(
                        session_id="s",
                        **approval,
                        decision=ToolApprovalDecision.APPROVE,
                        loop_policies=(RequireFinalTool(["ask_customer"], max_reminders=1),),
                    )
                )
            ]
            return (await store.load("s")).status, events, provider
        finally:
            await store.close()

    status, events, provider = asyncio.run(second_process())

    assert status == SessionStatus.INTERRUPTED
    assert len(provider.requests) == 1
    assert events[-1].type == "session.interrupted"
    assert events[-1].payload["policy_metadata"]["reminders"] == 1

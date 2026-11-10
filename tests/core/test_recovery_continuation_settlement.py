"""Every exit of a human-gate continuation settles the invocation it admitted.

Resolving a user input or a tool approval admits an invocation, commits the
gate's close and then hands the closed round to the engine. Whatever ends the
continuation after the close (an ordinary failure, a cancellation or a closed
stream) must leave the session terminal and resumable, never RUNNING with no
owner. Each case states who settles it and checks that it was settled
exactly once: one terminal event and one terminal hook.
"""

import asyncio

import pytest
from tests.core._recovery_settlement_observers import InterruptionHook, observe_decisions
from tests.runtime.runtime_provider_support import FakeProvider
from tests.runtime.runtime_tool_support import PureRecordingTool, RequireApprovalPolicy

from cayu.agents import AgentSpec
from cayu.applications import CayuApp
from cayu.approvals.tools import (
    ToolApprovalDecision,
    ToolApprovalRecoveryOutcome,
    ToolApprovalRecoveryRequest,
    ToolApprovalRequest,
)
from cayu.approvals.user_input import UserInputResponse
from cayu.events import EventType
from cayu.messages import Message
from cayu.providers import ModelStreamEvent
from cayu.sessions.memory import InMemorySessionStore
from cayu.sessions.records import SessionStatus
from cayu.sessions.requests import ResumeRequest, RunRequest
from cayu.storage.sqlite import SQLiteSessionStore
from cayu.tools.user_input import UserInputTool

_CLOSE_KINDS = {
    "user-input": "user-input-close",
    "approval": "approval-close",
    "approval-recovery": "approval-close",
}


class _ContinuationFault:
    """Fail the n-th checkpoint read after the gate's close commits.

    Manual recovery first needs a call whose terminal event was never stored,
    so the first tool completion fails once during its setup.
    """

    invocation_lifecycle_command_version = 1
    close_kind = ""
    fault = ""
    nth = 0
    phase = False
    fail_tool_completion = False
    reads: int | None = None
    injected = False
    closed = False

    async def append_events(self, session_id, events):
        if self.fail_tool_completion and any(
            event.type is EventType.TOOL_CALL_COMPLETED for event in events
        ):
            self.fail_tool_completion = False
            raise RuntimeError("terminal tool event unavailable")
        await super().append_events(session_id, events)

    async def publish_runtime_publication(self, session_id, **kwargs):
        result = await super().publish_runtime_publication(session_id, **kwargs)
        if self.phase and kwargs["request"].kind == self.close_kind:
            self.closed = True
            self.reads = None if self.fault == "close" else 0
        return result

    async def load_checkpoint(self, session_id):
        if self.reads is not None:
            self.reads += 1
        if self.reads == self.nth:
            self.reads = None
            self.injected = True
            if self.fault == "cancel":
                raise asyncio.CancelledError()
            raise OSError("continuation checkpoint read unavailable")
        return await super().load_checkpoint(session_id)


class _MemoryStore(_ContinuationFault, InMemorySessionStore):
    invocation_lifecycle_command_version = 1


class _SQLiteStore(_ContinuationFault, SQLiteSessionStore):
    invocation_lifecycle_command_version = 1


def _app(store, gate: str, hook: InterruptionHook) -> tuple[CayuApp, FakeProvider]:
    ask = ModelStreamEvent.tool_call(
        id="call_ask", name="ask_user", arguments={"question": "Continue?"}
    )
    call = ModelStreamEvent.tool_call(id="call_1", name="side_effect", arguments={})
    calls, tools, policy = {
        "user-input": ([ask], [UserInputTool()], None),
        "approval": ([call], [PureRecordingTool()], RequireApprovalPolicy()),
        "approval-recovery": ([call], [PureRecordingTool()], RequireApprovalPolicy()),
    }[gate]
    done = [ModelStreamEvent.text_delta("done"), ModelStreamEvent.completed()]
    provider = FakeProvider(
        [[*calls, ModelStreamEvent.completed({"finish_reason": "tool_calls"})], done, done]
    )
    app = CayuApp(session_store=store, runtime_hooks=[hook], enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"), tools=tools, tool_policy=policy
    )
    return app, provider


def _resolution(app: CayuApp, gate: str, session_id: str, paused: list):
    if gate == "user-input":
        awaiting = next(e for e in paused if e.type is EventType.SESSION_AWAITING_USER_INPUT)
        return app.resolve_user_input(
            UserInputResponse(
                session_id=session_id, input_id=awaiting.payload["input_id"], answer="yes"
            )
        )
    approval = next(e for e in paused if e.type is EventType.TOOL_CALL_APPROVAL_REQUESTED)
    return app.resolve_tool_approval(
        ToolApprovalRequest(
            session_id=session_id,
            approval_id=approval.payload["approval_id"],
            tool_round_id=approval.payload["tool_round_id"],
            tool_call_id=approval.payload["tool_call_id"],
            decision=ToolApprovalDecision.APPROVE,
        )
    )


def _manual_recovery(app: CayuApp, session_id: str, paused: list):
    approval = next(e for e in paused if e.type is EventType.TOOL_CALL_APPROVAL_REQUESTED)
    return app.recover_tool_approval(
        ToolApprovalRecoveryRequest(
            session_id=session_id,
            approval_id=approval.payload["approval_id"],
            tool_round_id=approval.payload["tool_round_id"],
            tool_call_id="call_1",
            outcome=ToolApprovalRecoveryOutcome.COMPLETED,
            message="verified outside",
        )
    )


async def _drive(stream, store) -> BaseException | None:
    try:
        async for _event in stream:
            if store.fault == "close" and store.closed:
                break
    except (Exception, asyncio.CancelledError) as error:
        return error
    finally:
        await stream.aclose()
    return None


# How each exit ends. ``owner``: the continuation's owner settles it as
# abandoned (its ledger decides). ``run``: the admitted run's own exit handler
# interrupts it. ``failed``: the run renders the failure as a failed session.
def _expected(gate: str, fault: str, nth: int) -> str:
    if fault == "close" or (fault, nth) == ("cancel", 1):
        return "owner"
    if fault == "cancel":
        return "run"
    # Only the user-input continuation reads outside the run before it starts;
    # every other read failure here happens inside the run.
    return "owner" if (gate, nth) == ("user-input", 1) else "failed"


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("gate", sorted(_CLOSE_KINDS))
@pytest.mark.parametrize(
    ("fault", "nth"), [("fail", 1), ("fail", 2), ("cancel", 1), ("cancel", 2), ("close", 0)]
)
def test_post_close_continuation_exit_settles_the_admitted_invocation(
    backend, gate, fault, nth, tmp_path, monkeypatch
):
    decisions = observe_decisions(monkeypatch)
    expected = _expected(gate, fault, nth)

    async def scenario():
        store = _MemoryStore() if backend == "memory" else _SQLiteStore(tmp_path / "s.db")
        store.close_kind, store.fault, store.nth = _CLOSE_KINDS[gate], fault, nth
        store.fail_tool_completion = gate == "approval-recovery"
        hook = InterruptionHook()
        app, provider = _app(store, gate, hook)
        session_id = f"continuation-{gate}-{fault}-{nth}"
        try:
            paused = [
                event
                async for event in app.run(
                    RunRequest(
                        session_id=session_id,
                        agent_name="assistant",
                        messages=[Message.text("user", "go")],
                    )
                )
            ]
            stream = _resolution(app, gate, session_id, paused)
            if gate == "approval-recovery":
                failed = [event.type async for event in stream]
                assert EventType.SESSION_INTERRUPTED in failed
                stream = _manual_recovery(app, session_id, paused)
            before = len(await store.load_events(session_id))
            hooks_before = hook.calls
            store.phase = True
            error = await _drive(stream, store)
            assert store.closed
            assert store.injected is (fault != "close")
            if expected == "failed" or fault == "close":
                assert error is None
            else:
                assert type(error) is (OSError if fault == "fail" else asyncio.CancelledError)
            assert len(provider.requests) == 1
            current = await store.load(session_id)
            assert current is not None
            phase = [event.type for event in (await store.load_events(session_id))[before:]]
            if expected == "failed":
                assert current.status is SessionStatus.FAILED
                assert phase.count(EventType.SESSION_FAILED) == 1
                assert EventType.SESSION_INTERRUPTED not in phase
                assert hook.calls == hooks_before
            else:
                # Settled exactly once: one terminal event and one terminal hook.
                assert current.status is SessionStatus.INTERRUPTED
                assert phase.count(EventType.SESSION_INTERRUPTED) == 1
                assert hook.calls - hooks_before == 1
            assert decisions == (["recovered"] if expected == "owner" else [])

            resumed = [
                event
                async for event in app.resume(
                    ResumeRequest(session_id=session_id, messages=[Message.text("user", "again")])
                )
            ]
            assert resumed[-1].type is EventType.SESSION_COMPLETED
            assert len(provider.requests) == 2
        finally:
            await app.aclose()

    asyncio.run(scenario())

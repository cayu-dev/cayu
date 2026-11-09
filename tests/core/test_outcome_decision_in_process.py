"""A failed or paused turn whose own publication ends early is finished in process.

The run commits the failed or interrupted transition with its decision, then its
TURN_COMPLETED or session event is cut short: a cancellation lands in that
write, environment finalization fails before the session event is saved, or the
event writer fails. The run's settlement must publish what the decision owes and
release the fence, so the session can be resumed or its approval answered at
once, without waiting for a recovery sweep. When the process stops instead, the
server's scheduled orphan reconciliation finishes the outcome.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator
from datetime import timedelta
from typing import Any

import pytest
from tests.core.test_tool_completion import FinalTool
from tests.fault_injection.harness import ManualClock

from cayu import AgentSpec, CayuApp, EventType, Message, ResumeRequest, RunRequest, SessionStatus
from cayu.approvals import ToolApprovalDecision, ToolApprovalRequest
from cayu.evals import ScriptedModelProvider
from cayu.providers import ModelProvider, ModelRequest, ModelStreamEvent
from cayu.runtime._environment_lifecycle import EnvironmentLifecycle
from cayu.runtime.execution_identity import ExecutionProfileBehaviorIdentity
from cayu.sessions import InMemorySessionStore
from cayu.storage.sqlite import SQLiteSessionStore
from cayu.tools.policy import AlwaysRequireApprovalToolPolicy
from cayu.tools.user_input import UserInputTool

_SESSION = "owed-in-process"
_OUTCOME = {
    "failure": (EventType.SESSION_FAILED, SessionStatus.FAILED),
    "approval": (EventType.SESSION_INTERRUPTED, SessionStatus.INTERRUPTED),
}


class _FailsOnce(ModelProvider):
    name = "fails-once"
    execution_profile_identity = ExecutionProfileBehaviorIdentity(
        name="tests:owed-in-process:fails-once", behavior_version="1", implementation_version="1"
    )

    def __init__(self) -> None:
        self.calls = 0

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        del request
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("provider failed")
        yield ModelStreamEvent.text_delta("done")
        yield ModelStreamEvent.completed({"finish_reason": "stop"})


def _provider(outcome: str) -> ModelProvider:
    if outcome == "failure":
        return _FailsOnce()
    return ScriptedModelProvider(
        [
            [
                ModelStreamEvent.tool_call(name="ask_customer", arguments={}, id="question"),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.text_delta("done"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )


def _app(store: Any, outcome: str, provider: ModelProvider | None = None) -> CayuApp:
    app = CayuApp(session_store=store, enable_logging=False)
    app.register_provider(provider or _provider(outcome), default=True)
    if outcome == "failure":
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))
    else:
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[FinalTool()],
            tool_policy=AlwaysRequireApprovalToolPolicy(),
        )
    return app


def _cut_short(
    monkeypatch: pytest.MonkeyPatch, store: Any, trigger: str, session_event: EventType
) -> None:
    """Break the outcome's own publication once, as ``trigger`` describes."""

    fired = False

    if trigger == "finalize":
        finalize = EnvironmentLifecycle.finalize_terminal_event

        async def failing_finalize(self, *, event, **kwargs):
            nonlocal fired
            if event.type is session_event and not fired:
                fired = True
                raise OSError("environment finalization failed")
            return await finalize(self, event=event, **kwargs)

        monkeypatch.setattr(EnvironmentLifecycle, "finalize_terminal_event", failing_finalize)
        return
    target = EventType.TURN_COMPLETED if trigger == "writer" else session_event
    append = type(store).append_event_and_claim_side_effect

    async def broken_append(self, session_id, event, **kwargs):
        nonlocal fired
        if event.type is target and not fired:
            fired = True
            if trigger == "writer":
                raise OSError("event store unavailable")
            current = asyncio.current_task()
            assert current is not None
            current.cancel()
            await asyncio.sleep(0)
        return await append(self, session_id, event, **kwargs)

    monkeypatch.setattr(type(store), "append_event_and_claim_side_effect", broken_append)


async def _continue(app: CayuApp, store: Any, outcome: str) -> list[Any]:
    if outcome == "failure":
        stream = app.resume(
            ResumeRequest(session_id=_SESSION, messages=[Message.text("user", "again")])
        )
    else:
        pending = (await store.load_checkpoint(_SESSION))["pending_tool_approval"]
        stream = app.resolve_tool_approval(
            ToolApprovalRequest(
                session_id=_SESSION,
                approval_id=pending["approval_id"],
                tool_round_id=pending["tool_round_id"],
                tool_call_id=pending["tool_call_id"],
                decision=ToolApprovalDecision.APPROVE,
            )
        )
    return [event async for event in stream]


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("outcome", sorted(_OUTCOME))
@pytest.mark.parametrize("trigger", ["cancel", "finalize", "writer"])
def test_an_outcome_cut_short_in_process_is_finished_before_its_release(
    tmp_path, monkeypatch, backend: str, outcome: str, trigger: str
) -> None:
    session_event, status = _OUTCOME[outcome]

    async def scenario() -> None:
        store = (
            InMemorySessionStore()
            if backend == "memory"
            else SQLiteSessionStore(tmp_path / "owed.sqlite")
        )
        app = _app(store, outcome)
        try:
            with monkeypatch.context() as patch:
                _cut_short(patch, store, trigger, session_event)

                async def run() -> None:
                    async for _ in app.run(
                        RunRequest(
                            agent_name="assistant",
                            session_id=_SESSION,
                            messages=[Message.text("user", "go")],
                        )
                    ):
                        pass

                # The run reports the failure it was cut short by, if any.
                with contextlib.suppress(BaseException):
                    await asyncio.create_task(run())
            assert await app.drain_recovery_cleanups(timeout_s=5)
            assert await app.drain_environment_cleanups(timeout_s=5)
            assert (await store.load(_SESSION)).status is status
            owed = [event.type for event in await store.load_events(_SESSION)]
            assert owed.count(session_event) == 1, owed
            assert owed.count(EventType.TURN_COMPLETED) == 1, owed

            # No recovery sweep runs: the session continues at once.
            continued = await _continue(app, store, outcome)
            assert continued[-1].type is EventType.SESSION_COMPLETED
        finally:
            await app.aclose()
            if backend == "sqlite":
                await store.close()

    asyncio.run(scenario())


class _ProcessStopped(BaseException):
    """A process-level stop: the run's settlement leaves the session to recovery."""


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("outcome", sorted(_OUTCOME))
def test_the_servers_orphan_reconciliation_finishes_an_owed_outcome(
    tmp_path, monkeypatch, backend: str, outcome: str
) -> None:
    session_event, status = _OUTCOME[outcome]

    async def scenario() -> None:
        clock = ManualClock()
        store = (
            InMemorySessionStore(ownership_clock=clock)
            if backend == "memory"
            else SQLiteSessionStore(tmp_path / "owed.sqlite", ownership_clock=clock)
        )
        # The restarted app runs the same provider, as a redeployed worker would.
        provider = _provider(outcome)
        stopped = _app(store, outcome, provider)
        append = type(store).append_event_and_claim_side_effect

        async def stop_at_session_event(self, session_id, event, **kwargs):
            if event.type is session_event:
                raise _ProcessStopped
            return await append(self, session_id, event, **kwargs)

        with monkeypatch.context() as patch:
            patch.setattr(type(store), "append_event_and_claim_side_effect", stop_at_session_event)
            with contextlib.suppress(_ProcessStopped):
                async for _ in stopped.run(
                    RunRequest(
                        agent_name="assistant",
                        session_id=_SESSION,
                        messages=[Message.text("user", "go")],
                    )
                ):
                    pass
        assert (await store.load(_SESSION)).status is status
        assert session_event not in [event.type for event in await store.load_events(_SESSION)]
        clock.advance(timedelta(days=1))
        app = _app(store, outcome, provider)
        try:
            await app.reconcile_orphaned_sessions(inactive_for_seconds=0)
            owed = [event.type for event in await store.load_events(_SESSION)]
            assert owed.count(session_event) == 1, owed
            continued = await _continue(app, store, outcome)
            assert continued[-1].type is EventType.SESSION_COMPLETED
        finally:
            await app.aclose()
            if backend == "sqlite":
                await store.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_a_pause_whose_transition_fails_is_settled_in_process(
    tmp_path, monkeypatch, backend: str
) -> None:
    """Nothing is owed before the pause commits, so the run itself settles the session."""

    async def scenario() -> None:
        store = (
            InMemorySessionStore()
            if backend == "memory"
            else SQLiteSessionStore(tmp_path / "owed.sqlite")
        )
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(
            ScriptedModelProvider(
                [
                    [
                        ModelStreamEvent.tool_call(
                            id="ask", name="ask_user", arguments={"question": "Which one?"}
                        ),
                        ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                    ]
                ]
            ),
            default=True,
        )
        app.register_agent(AgentSpec(name="assistant", model="fake-model"), tools=[UserInputTool()])
        settle = type(store).settle_session_invocation

        async def refuse_pause(self, command):
            if command.transition.to_status is SessionStatus.INTERRUPTED:
                raise OSError("pause transition failed")
            return await settle(self, command)

        try:
            with monkeypatch.context() as patch:
                patch.setattr(type(store), "settle_session_invocation", refuse_pause)
                with contextlib.suppress(BaseException):
                    async for _ in app.run(
                        RunRequest(
                            agent_name="assistant",
                            session_id=_SESSION,
                            messages=[Message.text("user", "go")],
                        )
                    ):
                        pass
            assert await app.drain_recovery_cleanups(timeout_s=5)
            assert (await store.load(_SESSION)).status is not SessionStatus.RUNNING
        finally:
            await app.aclose()
            if backend == "sqlite":
                await store.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_answering_a_pause_whose_process_stopped_finishes_it_first(
    tmp_path, monkeypatch, backend: str
) -> None:
    """Approving a pause whose process stopped before its release settles it first."""

    async def scenario() -> None:
        clock = ManualClock()
        store = (
            InMemorySessionStore(ownership_clock=clock)
            if backend == "memory"
            else SQLiteSessionStore(tmp_path / "owed.sqlite", ownership_clock=clock)
        )
        provider = _provider("approval")
        stopped = _app(store, "approval", provider)

        async def stop_before_release(self, command):
            raise _ProcessStopped

        # The pause's events are durable; only its fence release, which
        # settles its decision, is lost.
        with monkeypatch.context() as patch:
            patch.setattr(type(store), "release_session_invocation", stop_before_release)
            with contextlib.suppress(_ProcessStopped):
                async for _ in stopped.run(
                    RunRequest(
                        agent_name="assistant",
                        session_id=_SESSION,
                        messages=[Message.text("user", "go")],
                    )
                ):
                    pass
        clock.advance(timedelta(days=1))
        app = _app(store, "approval", provider)
        try:
            continued = await _continue(app, store, "approval")
            assert continued[-1].type is EventType.SESSION_COMPLETED
            owed = [event.type for event in await store.load_events(_SESSION)]
            assert owed.count(EventType.SESSION_INTERRUPTED) == 1, owed
            assert owed.count(EventType.TURN_COMPLETED) == 2, owed
        finally:
            await app.aclose()
            if backend == "sqlite":
                await store.close()

    asyncio.run(scenario())

"""Session-store contracts for the completed terminal decision a completion installs."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from tests.core.session_store_conformance.support import _close_store, _open_store

from cayu.agents import AgentSpec
from cayu.applications import CayuApp
from cayu.evals.testing import ScriptedModelProvider
from cayu.events import EventType
from cayu.messages import Message
from cayu.providers import ModelStreamEvent
from cayu.runtime._checkpoint_store import runtime_checkpoint_session_store
from cayu.runtime._environment_lifecycle import EnvironmentLifecycle
from cayu.runtime._invocation_lifecycle import _release_invocation_command_with_cleanup_authority
from cayu.sessions._execution_profile_checkpoint import (
    active_invocation_execution_profile_from_checkpoint,
)
from cayu.sessions._invocation_lifecycle import ReleaseInvocationCommand
from cayu.sessions._invocation_terminal_decision import (
    build_completed_invocation_terminal_decision,
    invocation_terminal_decision_from_checkpoint,
)
from cayu.sessions._paired_writes import completed_decision_install_mutation
from cayu.sessions.authority import SessionRunFenced
from cayu.sessions.checkpoints import (
    INVOCATION_TERMINAL_DECISION_CHECKPOINT_KEY,
    SETTLED_INVOCATION_TERMINAL_DECISION_CHECKPOINT_KEY,
)
from cayu.sessions.messaging import EnqueueSessionMessageRequest, SessionMessageDeliveryMode
from cayu.sessions.records import SessionStatus
from cayu.sessions.requests import ResumeRequest, RunRequest

_SESSION_ID = "completed-decision"


def _answer() -> list[ModelStreamEvent]:
    return [
        ModelStreamEvent.text_delta("done"),
        ModelStreamEvent.completed({"finish_reason": "stop"}),
    ]


def _app(store: Any, turns: int) -> CayuApp:
    app = CayuApp(session_store=store, enable_logging=False)
    app.register_provider(ScriptedModelProvider([_answer() for _ in range(turns)]), default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))
    return app


async def _consume(stream: Any) -> BaseException | None:
    try:
        async for _ in stream:
            pass
    except Exception as error:
        return error
    return None


def _fail_owed_turn(store: Any, monkeypatch: pytest.MonkeyPatch, failures: list[int]) -> None:
    original = store.append_event

    async def append_event(session_id: str, event: Any) -> None:
        if event.type is EventType.TURN_COMPLETED and failures[0]:
            failures[0] -= 1
            raise OSError("turn completion store failure")
        await original(session_id, event)

    monkeypatch.setattr(store, "append_event", append_event)


def _fail_completion_finalization(monkeypatch: pytest.MonkeyPatch, failures: list[int]) -> None:
    original = EnvironmentLifecycle.finalize_terminal_event

    async def finalize_terminal_event(self: EnvironmentLifecycle, **kwargs: Any) -> Any:
        if kwargs["event"].type is EventType.SESSION_COMPLETED and failures[0]:
            failures[0] -= 1
            raise OSError("environment binding finalization failure")
        return await original(self, **kwargs)

    monkeypatch.setattr(EnvironmentLifecycle, "finalize_terminal_event", finalize_terminal_event)


async def _assert_completed_once_and_resumable(store: Any, app: CayuApp) -> None:
    session = await store.load(_SESSION_ID)
    assert session is not None and session.status is SessionStatus.COMPLETED
    checkpoint = await store.load_checkpoint(_SESSION_ID) or {}
    assert INVOCATION_TERMINAL_DECISION_CHECKPOINT_KEY not in checkpoint
    assert SETTLED_INVOCATION_TERMINAL_DECISION_CHECKPOINT_KEY not in checkpoint
    types = [event.type for event in await store.load_events(_SESSION_ID)]
    assert EventType.SESSION_FAILED not in types
    assert types.count(EventType.SESSION_COMPLETED) == 1
    assert types.count(EventType.TURN_COMPLETED) == 1
    resumed = [
        event
        async for event in app.resume(
            ResumeRequest(session_id=_SESSION_ID, messages=[Message.text("user", "again")])
        )
    ]
    assert resumed[-1].type is EventType.SESSION_COMPLETED


@pytest.mark.parametrize("fault", ["turn-store-error", "binding-finalize-error"])
def test_session_store_conformance_failure_after_completion_is_finished_by_recovery(
    session_store_case, monkeypatch, fault: str
) -> None:
    """An error after the completion commits leaves it owed, and recovery finishes it.

    The error is reported, never turned into a failed session. The publication is
    not retried in the failing process, so environment finalization never runs twice.
    """

    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            failures = [1]
            if fault == "turn-store-error":
                _fail_owed_turn(store, monkeypatch, failures)
            else:
                _fail_completion_finalization(monkeypatch, failures)
            app = _app(store, turns=2)
            error = await _consume(
                app.run(
                    RunRequest(
                        agent_name="assistant",
                        session_id=_SESSION_ID,
                        messages=[Message.text("user", "answer")],
                    )
                )
            )
            assert isinstance(error, OSError) and failures == [0]
            assert await app.drain_recovery_cleanups(timeout_s=5)
            assert await app.drain_environment_cleanups(timeout_s=5)
            session = await store.load(_SESSION_ID)
            assert session is not None and session.status is SessionStatus.COMPLETED
            assert INVOCATION_TERMINAL_DECISION_CHECKPOINT_KEY in (
                await store.load_checkpoint(_SESSION_ID) or {}
            )
            await app.aclose()
            recovery = _app(store, turns=1)
            await recovery.recover_abandoned_sessions(inactive_for_seconds=0)
            await _assert_completed_once_and_resumable(store, recovery)
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_lasting_failure_after_completion_is_recovered(
    session_store_case, monkeypatch
) -> None:
    """A completion that keeps failing stays completed and owed, and the sweep finishes it."""

    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            failures = [1_000]
            _fail_owed_turn(store, monkeypatch, failures)
            app = _app(store, turns=1)
            error = await _consume(
                app.run(
                    RunRequest(
                        agent_name="assistant",
                        session_id=_SESSION_ID,
                        messages=[Message.text("user", "answer")],
                    )
                )
            )
            assert isinstance(error, OSError)
            assert await app.drain_recovery_cleanups(timeout_s=5)
            await app.aclose()
            session = await store.load(_SESSION_ID)
            assert session is not None and session.status is SessionStatus.COMPLETED
            assert INVOCATION_TERMINAL_DECISION_CHECKPOINT_KEY in (
                await store.load_checkpoint(_SESSION_ID) or {}
            )

            failures[0] = 0
            recovery = _app(store, turns=1)
            await recovery.recover_abandoned_sessions(inactive_for_seconds=0)
            await _assert_completed_once_and_resumable(store, recovery)
        finally:
            await _close_store(store)

    asyncio.run(run())


def _capture_completion_installs(
    store: Any, monkeypatch: pytest.MonkeyPatch, before_first: Any = None
) -> list[Any]:
    original = type(store).settle_session_invocation
    installs: list[Any] = []

    async def settle_session_invocation(self: Any, command: Any) -> Any:
        transition = command.transition
        if transition.to_status is SessionStatus.COMPLETED and transition.checkpoint_mutation:
            installs.append(command)
            if before_first is not None and len(installs) == 1:
                await before_first()
        return await original(self, command)

    monkeypatch.setattr(type(store), "settle_session_invocation", settle_session_invocation)
    return installs


def test_session_store_conformance_completed_decision_store_rules(
    session_store_case, monkeypatch
) -> None:
    """Install with the COMPLETED transition only, replay exactly, refuse leaving or releasing."""

    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            installs = _capture_completion_installs(store, monkeypatch)
            _fail_owed_turn(store, monkeypatch, [1_000])
            app = _app(store, turns=1)
            assert isinstance(
                await _consume(
                    app.run(
                        RunRequest(
                            agent_name="assistant",
                            session_id=_SESSION_ID,
                            messages=[Message.text("user", "answer")],
                        )
                    )
                ),
                OSError,
            )
            assert await app.drain_recovery_cleanups(timeout_s=5)
            [install] = installs
            session = await store.load(_SESSION_ID)
            checkpoint = await store.load_checkpoint(_SESSION_ID) or {}
            decision = invocation_terminal_decision_from_checkpoint(checkpoint)
            assert session is not None and session.status is SessionStatus.COMPLETED
            assert decision is not None and decision.outcome.value == "completed"
            assert decision.interaction_event_id == install.transition.event.id

            # Exact replay returns the receipt and installs nothing again.
            replayed = await store.settle_session_invocation(install)
            assert replayed.replayed is True
            assert (
                invocation_terminal_decision_from_checkpoint(
                    await store.load_checkpoint(_SESSION_ID)
                )
                == decision
            )

            # Another decision for the same completion event is refused.
            conflicting = build_completed_invocation_terminal_decision(
                session_id=decision.session_id,
                session_instance_id=decision.session_instance_id,
                run_epoch=decision.run_epoch,
                profile_interaction_id=decision.profile_interaction_id,
                interaction_id=decision.interaction_id,
                execution_profile_fingerprint=decision.execution_profile_fingerprint,
                interaction_event_id=decision.interaction_event_id,
                terminal_event_id=decision.terminal_event_id,
                observed_at=decision.observed_at,
                terminal_payload={"reason": "another"},
                turn_completed_payload=decision.turn_completed_payload or {},
            )
            with pytest.raises(ValueError, match="reused with different data"):
                await store.settle_session_invocation(
                    install.model_copy(
                        update={
                            "transition": install.transition.model_copy(
                                update={
                                    "checkpoint_mutation": completed_decision_install_mutation(
                                        conflicting
                                    )
                                }
                            )
                        }
                    )
                )

            # The session cannot leave COMPLETED while its completion is owed.
            with pytest.raises(SessionRunFenced, match="keeps its status"):
                await store.transition_status(
                    _SESSION_ID,
                    from_statuses={SessionStatus.COMPLETED},
                    to_status=SessionStatus.FAILED,
                )

            # Nor release its fence before SESSION_COMPLETED is durable.
            active = active_invocation_execution_profile_from_checkpoint(checkpoint)
            assert active is not None
            with pytest.raises(SessionRunFenced, match="held until its completed outcome"):
                await runtime_checkpoint_session_store(store).apply_invocation_lifecycle_command(
                    _release_invocation_command_with_cleanup_authority(
                        ReleaseInvocationCommand(
                            session_id=_SESSION_ID,
                            expected_session_instance_id=session.instance_id,
                            expected_run_epoch=active.run_epoch,
                            expected_active_profile=active,
                            settlement_transition=install.transition,
                        )
                    )
                )
            assert (await store.load(_SESSION_ID)).run_epoch == session.run_epoch
            await app.aclose()

            # Recovery publishes it and the release settles the decision with the fence.
            monkeypatch.undo()
            recovery = _app(store, turns=1)
            await recovery.recover_abandoned_sessions(inactive_for_seconds=0)
            released = await store.load(_SESSION_ID)
            assert released is not None and released.run_epoch > session.run_epoch
            await _assert_completed_once_and_resumable(store, recovery)
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_queued_completion_installs_no_decision(
    session_store_case, monkeypatch
) -> None:
    """Queued input keeps the session running, so its completion installs nothing."""

    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            app = _app(store, turns=2)

            async def queue_next_turn() -> None:
                await app.enqueue_session_message(
                    EnqueueSessionMessageRequest(
                        session_id=_SESSION_ID,
                        idempotency_key="queued-before-completion",
                        content="one more",
                        delivery_mode=SessionMessageDeliveryMode.NEXT_TURN,
                    )
                )

            installs = _capture_completion_installs(store, monkeypatch, queue_next_turn)
            assert (
                await _consume(
                    app.run(
                        RunRequest(
                            agent_name="assistant",
                            session_id=_SESSION_ID,
                            messages=[Message.text("user", "answer")],
                        )
                    )
                )
                is None
            )
            assert len(installs) == 2
            queued, completed = installs
            receipt = await store.load_interaction_transition_receipt(
                _SESSION_ID, transition=queued.transition
            )
            assert receipt is not None and receipt.status_changed is False
            checkpoint = await store.load_checkpoint(_SESSION_ID) or {}
            assert INVOCATION_TERMINAL_DECISION_CHECKPOINT_KEY not in checkpoint
            completed_receipt = await store.load_interaction_transition_receipt(
                _SESSION_ID, transition=completed.transition
            )
            assert completed_receipt is not None and completed_receipt.status_changed is True
            types = [event.type for event in await store.load_events(_SESSION_ID)]
            assert types.count(EventType.SESSION_COMPLETED) == 1
        finally:
            await _close_store(store)

    asyncio.run(run())

"""Shared session-store finalization contract cases."""

from __future__ import annotations

import asyncio
import copy
from datetime import UTC, datetime, timedelta

import pytest
from tests.core.session_store_conformance.support import (
    _close_store,
    _collect_events,
    _identity,
    _open_peer_store,
    _open_store,
    _private_event_for_public_event,
    _reopen_store,
    _SimulatedProcessLoss,
    _UserInputRecoveryProvider,
)

import cayu.runtime._recovery_coordinator as recovery_coordinator_module
import cayu.runtime._session_control as session_control_module
from cayu._exception_groups import iter_exception_tree
from cayu.agents import AgentSpec
from cayu.applications import CayuApp
from cayu.events import (
    Event,
    EventType,
)
from cayu.messages import (
    Message,
)
from cayu.runtime import _recovery_ownership as recovery_ownership_module
from cayu.runtime._interruption_coordinator import suppress_interruption_cascade
from cayu.runtime.authority import SessionRunFenced
from cayu.runtime.execution_profiles import (
    active_invocation_execution_profile_from_checkpoint,
)
from cayu.sessions._checkpoint_mutations import (
    SessionRuntimePublicationConflict,
)
from cayu.sessions._interaction_transitions import (
    InteractionTransitionSpec,
)
from cayu.sessions.event_queries import EventQuery
from cayu.sessions.memory import InMemorySessionStore
from cayu.sessions.records import Session, SessionStatus
from cayu.sessions.requests import (
    InterruptSessionRequest,
    RunRequest,
)
from cayu.sessions.store import SessionStore
from cayu.storage.sqlite import SQLiteSessionStore
from cayu.tools.user_input import UserInputTool

pytestmark = pytest.mark.postgres


def test_session_store_conformance_claim_loss_does_not_run_unclaimed_abandonment(
    session_store_case,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        session_id = f"terminal-finalization-claim-loss-{session_store_case[0]}"
        transition_started = asyncio.Event()
        never_release_transition = asyncio.Event()
        try:
            app = CayuApp(session_store=store, enable_logging=False)
            app.register_provider(_UserInputRecoveryProvider(), default=True)
            app.register_agent(
                AgentSpec(name="assistant", model="fake-model"),
                tools=[UserInputTool()],
            )
            await _collect_events(
                app.run(
                    RunRequest(
                        session_id=session_id,
                        agent_name="assistant",
                        messages=[Message.text("user", "ask a question")],
                    )
                )
            )
            request = InterruptSessionRequest(
                session_id=session_id,
                reason="operator supersedes the paused question",
            )
            recovery = app._recovery_coordinator
            original_heartbeat = recovery._recovery_ownership.heartbeat
            original_publish = store.publish_interaction_transition
            original_status_transition = store.transition_status_and_checkpoint
            original_finalize_abandoned = app._session_finalization.finalize_abandoned_session_run
            abandoned_finalization_calls = 0

            async def block_claimed_transition(*args, **kwargs):
                if kwargs.get("expected_recovery_claim_id") is not None:
                    transition_started.set()
                    await never_release_transition.wait()
                return await original_publish(*args, **kwargs)

            async def block_claimed_status_transition(*args, **kwargs):
                if kwargs.get("to_status") is SessionStatus.INTERRUPTED:
                    transition_started.set()
                    await never_release_transition.wait()
                return await original_status_transition(*args, **kwargs)

            async def observe_abandoned_finalization(request):
                nonlocal abandoned_finalization_calls
                abandoned_finalization_calls += 1
                return await original_finalize_abandoned(request)

            async def replace_claim_while_transition_waits(
                *,
                session_id: str,
                claim_id: str,
                local_lease_deadline: float,
                stop: asyncio.Event,
            ) -> None:
                del local_lease_deadline, stop
                await asyncio.wait_for(transition_started.wait(), timeout=10)
                replacement_now = datetime.now(UTC)

                def replace_claim(_session, checkpoint):
                    assert checkpoint is not None
                    updated = copy.deepcopy(checkpoint)
                    assert updated["incomplete_session_recovery_claim"]["claim_id"] == claim_id
                    updated["incomplete_session_recovery_claim"] = {
                        "version": 1,
                        "claim_id": "replacement-finalization-owner",
                        "claimed_at": replacement_now.isoformat(),
                        "claim_expires_at": (replacement_now + timedelta(minutes=5)).isoformat(),
                    }
                    return updated

                await store.transform_checkpoint(session_id, replace_claim)
                raise recovery_coordinator_module._IncompleteRecoveryClaimLost(
                    "peer replaced the terminal finalization claim"
                )

            store.publish_interaction_transition = block_claimed_transition  # type: ignore[method-assign]
            store.transition_status_and_checkpoint = (  # type: ignore[method-assign]
                block_claimed_status_transition
            )
            app._session_finalization.finalize_abandoned_session_run = (  # type: ignore[method-assign]
                observe_abandoned_finalization
            )
            monkeypatch.setattr(
                recovery._recovery_ownership,
                "heartbeat",
                replace_claim_while_transition_waits,
            )

            with (
                suppress_interruption_cascade(),
                pytest.raises(
                    recovery_coordinator_module._IncompleteRecoveryClaimLost,
                    match="peer replaced",
                ),
            ):
                await _collect_events(app.interrupt_session(request))

            assert abandoned_finalization_calls == 0
            retained_session = await store.load(session_id)
            retained_checkpoint = await store.load_checkpoint(session_id)
            assert retained_session is not None
            assert retained_session.status is SessionStatus.INTERRUPTING
            assert retained_checkpoint is not None
            assert retained_checkpoint["incomplete_session_recovery_claim"]["claim_id"] == (
                "replacement-finalization-owner"
            )
            terminal_records = await store.query_events(
                EventQuery(
                    session_id=session_id,
                    event_type=EventType.SESSION_INTERRUPTED,
                )
            )
            assert [
                record.event.payload.get("interruption_type") for record in terminal_records
            ] == ["user_input_required"]

            def expire_replacement_claim(_session, checkpoint):
                assert checkpoint is not None
                updated = copy.deepcopy(checkpoint)
                expired_at = datetime.now(UTC) - timedelta(minutes=5)
                marker = updated["incomplete_session_recovery_claim"]
                assert marker["claim_id"] == "replacement-finalization-owner"
                marker["claimed_at"] = (expired_at - timedelta(minutes=5)).isoformat()
                marker["claim_expires_at"] = expired_at.isoformat()
                return updated

            await store.transform_checkpoint(session_id, expire_replacement_claim)
            store.publish_interaction_transition = original_publish  # type: ignore[method-assign]
            store.transition_status_and_checkpoint = (  # type: ignore[method-assign]
                original_status_transition
            )
            app._session_finalization.finalize_abandoned_session_run = (  # type: ignore[method-assign]
                original_finalize_abandoned
            )
            monkeypatch.setattr(
                recovery._recovery_ownership,
                "heartbeat",
                original_heartbeat,
            )
            with suppress_interruption_cascade():
                retried = await _collect_events(app.interrupt_session(request))
            assert retried[-1].type is EventType.SESSION_INTERRUPTED
            assert retried[-1].payload["interruption_type"] == "operator_requested"
        finally:
            never_release_transition.set()
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_unclaimed_interruption_cancellation_is_finalized(
    session_store_case,
    monkeypatch,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        session_id = f"unclaimed-interruption-cancellation-{session_store_case[0]}"
        transition_started = asyncio.Event()
        hold_first_transition = asyncio.Event()
        try:
            monkeypatch.setattr(
                session_control_module,
                "ACTIVE_INTERRUPTED_EVENT_WAIT_ATTEMPTS",
                2,
            )
            monkeypatch.setattr(
                session_control_module,
                "ACTIVE_INTERRUPTED_EVENT_WAIT_INTERVAL_S",
                0.001,
            )
            app = CayuApp(session_store=store, enable_logging=False)
            app.register_provider(_UserInputRecoveryProvider(), default=True)
            app.register_agent(AgentSpec(name="assistant", model="fake-model"))
            await store.create(
                RunRequest(
                    session_id=session_id,
                    agent_name="assistant",
                    messages=[Message.text("user", "start")],
                ),
                identity=_identity(),
            )
            original_transition = store.update_status
            original_finalize = (
                app._session_engine._session_finalization.finalize_abandoned_session_run
            )
            transition_calls = 0
            finalization_calls = 0

            async def block_first_transition(*args, **kwargs):
                nonlocal transition_calls
                transition_calls += 1
                if transition_calls == 1:
                    transition_started.set()
                    await hold_first_transition.wait()
                return await original_transition(*args, **kwargs)

            async def observe_finalization(request):
                nonlocal finalization_calls
                finalization_calls += 1
                return await original_finalize(request)

            store.update_status = block_first_transition  # type: ignore[method-assign]
            app._session_engine._session_finalization.finalize_abandoned_session_run = (  # type: ignore[method-assign]
                observe_finalization
            )
            task = asyncio.create_task(
                _collect_events(
                    app.interrupt_session(
                        InterruptSessionRequest(
                            session_id=session_id,
                            reason="operator stop",
                        )
                    )
                )
            )
            await asyncio.wait_for(transition_started.wait(), timeout=10)
            task.cancel()
            assert task.cancelling() == 1
            with pytest.raises(asyncio.CancelledError):
                await task
            assert task.cancelled()
            assert finalization_calls == 1
            retained = await store.load(session_id)
            assert retained is not None
            assert retained.status is SessionStatus.INTERRUPTED
        finally:
            hold_first_transition.set()
            await _close_store(store)

    asyncio.run(run())


@pytest.mark.parametrize(
    "claim_boundary",
    [
        pytest.param("ack-loss", id="commit-then-lose-acknowledgement"),
        pytest.param("cancellation", id="cancel-after-commit"),
        pytest.param(
            "cancellation-read-failure",
            id="cancel-after-commit-before-checkpoint-read-failure",
        ),
        pytest.param("generator-exit", id="generator-exit-after-commit"),
        pytest.param(
            "generator-exit-read-failure",
            id="generator-exit-after-commit-before-checkpoint-read-failure",
        ),
        pytest.param("grouped-control", id="grouped-process-control-after-commit"),
        pytest.param("payload-conflict", id="conflicting-payload-before-claim"),
    ],
)
def test_session_store_conformance_reconciles_terminal_finalization_claim_transfer(
    session_store_case,
    claim_boundary: str,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        session_id = f"terminal-finalization-claim-{claim_boundary}-{session_store_case[0]}"
        release_acknowledgement = asyncio.Event()
        try:
            app = CayuApp(session_store=store, enable_logging=False)
            app.register_provider(_UserInputRecoveryProvider(), default=True)
            app.register_agent(
                AgentSpec(name="assistant", model="fake-model"),
                tools=[UserInputTool()],
            )
            await _collect_events(
                app.run(
                    RunRequest(
                        session_id=session_id,
                        agent_name="assistant",
                        messages=[Message.text("user", "ask a question")],
                    )
                )
            )
            request = InterruptSessionRequest(
                session_id=session_id,
                reason="operator supersedes the paused question",
            )

            original_transition = store.transition_status_and_checkpoint

            async def commit_supersession_then_stop(*args, **kwargs):
                transitioned = await original_transition(*args, **kwargs)
                if kwargs.get("to_status") is SessionStatus.INTERRUPTING:
                    raise _SimulatedProcessLoss("stop after supersession transition")
                return transitioned

            store.transition_status_and_checkpoint = (  # type: ignore[method-assign]
                commit_supersession_then_stop
            )
            with (
                suppress_interruption_cascade(),
                pytest.raises(_SimulatedProcessLoss, match="supersession transition"),
            ):
                await _collect_events(app.interrupt_session(request))
            store.transition_status_and_checkpoint = original_transition  # type: ignore[method-assign]

            def release_abandoned_claim(_session: Session, checkpoint):
                updated = dict({} if checkpoint is None else checkpoint)
                updated.pop("incomplete_session_recovery_claim", None)
                return updated

            await store.transform_checkpoint(session_id, release_abandoned_claim)
            prepared = await store.load_checkpoint(session_id)
            assert prepared is not None
            assert "pending_session_interrupt" in prepared
            assert "incomplete_session_recovery_claim" not in prepared

            original_transform = store.transform_checkpoint_with_store_time
            original_load_checkpoint = store.load_checkpoint
            claim_committed = asyncio.Event()
            injected = False
            fail_next_checkpoint_read = False

            async def fail_checkpoint_read_after_transfer(loaded_session_id: str):
                nonlocal fail_next_checkpoint_read
                if fail_next_checkpoint_read:
                    fail_next_checkpoint_read = False
                    raise OSError("post-claim checkpoint read failed")
                return await original_load_checkpoint(loaded_session_id)

            async def interrupt_claim_acknowledgement(
                transformed_session_id: str,
                transform,
            ):
                nonlocal injected
                nonlocal fail_next_checkpoint_read
                if injected:
                    return await original_transform(transformed_session_id, transform)
                before = await original_load_checkpoint(transformed_session_id)
                claim_transfer_candidate = (
                    not injected
                    and before is not None
                    and "incomplete_session_recovery_claim" not in before
                    and "pending_session_interrupt" in before
                )
                if claim_boundary == "payload-conflict" and claim_transfer_candidate:
                    injected = True

                    def replace_pending_payload(
                        _session: Session,
                        checkpoint,
                        _store_now: datetime,
                    ):
                        updated = dict({} if checkpoint is None else checkpoint)
                        pending_interrupt = dict(updated["pending_session_interrupt"])
                        pending_interrupt["reason"] = "conflicting concurrent interrupt"
                        updated["pending_session_interrupt"] = pending_interrupt
                        return updated

                    await original_transform(
                        transformed_session_id,
                        replace_pending_payload,
                    )
                result = await original_transform(transformed_session_id, transform)
                after = await original_load_checkpoint(transformed_session_id)
                installed_now = (
                    not injected
                    and before is not None
                    and "incomplete_session_recovery_claim" not in before
                    and after is not None
                    and "incomplete_session_recovery_claim" in after
                    and "pending_session_interrupt" in after
                )
                if not installed_now:
                    return result
                injected = True
                claim_committed.set()
                if claim_boundary == "ack-loss":
                    raise OSError("terminal claim acknowledgement lost")
                if claim_boundary in {
                    "generator-exit",
                    "generator-exit-read-failure",
                }:
                    fail_next_checkpoint_read = claim_boundary.endswith("read-failure")
                    raise GeneratorExit("terminal claim transfer stopped")
                if claim_boundary == "grouped-control":
                    raise BaseExceptionGroup(
                        "terminal claim transfer controls",
                        [
                            GeneratorExit("terminal claim transfer stopped"),
                            RuntimeError("terminal claim transfer cleanup failed"),
                        ],
                    )
                await release_acknowledgement.wait()
                fail_next_checkpoint_read = claim_boundary.endswith("read-failure")
                return result

            store.load_checkpoint = fail_checkpoint_read_after_transfer  # type: ignore[method-assign]
            store.transform_checkpoint_with_store_time = (  # type: ignore[method-assign]
                interrupt_claim_acknowledgement
            )
            if claim_boundary == "ack-loss":
                with suppress_interruption_cascade():
                    events = await _collect_events(app.interrupt_session(request))
                assert events[-1].type is EventType.SESSION_INTERRUPTED
            elif claim_boundary == "payload-conflict":
                with (
                    suppress_interruption_cascade(),
                    pytest.raises(
                        SessionRuntimePublicationConflict,
                        match="interrupt identity changed",
                    ),
                ):
                    await _collect_events(app.interrupt_session(request))
                conflicted = await store.load_checkpoint(session_id)
                assert conflicted is not None
                assert "pending_session_interrupt" in conflicted
                assert "incomplete_session_recovery_claim" not in conflicted
                records = [
                    record
                    for record in await store.query_events(
                        EventQuery(
                            session_id=session_id,
                            event_type=EventType.SESSION_INTERRUPTED,
                        )
                    )
                    if record.event.payload.get("interruption_type") == "operator_requested"
                ]
                assert records == []
                return
            elif claim_boundary in {
                "generator-exit",
                "generator-exit-read-failure",
                "grouped-control",
            }:
                with (
                    suppress_interruption_cascade(),
                    pytest.raises(
                        GeneratorExit,
                        match="terminal claim transfer stopped",
                    ) as process_control,
                ):
                    await _collect_events(app.interrupt_session(request))
                if claim_boundary == "grouped-control":
                    assert process_control.value.__cause__ is not None
                    retained_failures = tuple(
                        candidate
                        for candidate in iter_exception_tree(process_control.value.__cause__)
                        if not isinstance(candidate, BaseExceptionGroup)
                    )
                    assert (
                        sum(
                            isinstance(candidate, RuntimeError)
                            and str(candidate) == "terminal claim transfer cleanup failed"
                            for candidate in retained_failures
                        )
                        == 1
                    )
                elif claim_boundary == "generator-exit-read-failure":
                    assert process_control.value.__cause__ is not None
                    retained_failures = tuple(
                        candidate
                        for candidate in iter_exception_tree(process_control.value.__cause__)
                        if not isinstance(candidate, BaseExceptionGroup)
                    )
                    assert (
                        sum(
                            isinstance(candidate, OSError)
                            and str(candidate) == "post-claim checkpoint read failed"
                            for candidate in retained_failures
                        )
                        == 1
                    )
                store.load_checkpoint = original_load_checkpoint  # type: ignore[method-assign]
                store.transform_checkpoint_with_store_time = original_transform  # type: ignore[method-assign]
                with suppress_interruption_cascade():
                    events = await _collect_events(app.interrupt_session(request))
                assert events[-1].type is EventType.SESSION_INTERRUPTED
            else:
                with suppress_interruption_cascade():
                    retrying = asyncio.create_task(_collect_events(app.interrupt_session(request)))
                    await asyncio.wait_for(claim_committed.wait(), timeout=5)
                    retrying.cancel("cancel terminal claim acknowledgement")
                    assert retrying.cancelling() == 1
                    release_acknowledgement.set()
                    with pytest.raises(asyncio.CancelledError) as cancellation:
                        await retrying
                    assert cancellation.value.args == ("cancel terminal claim acknowledgement",)
                    assert retrying.cancelled() is True
                    if claim_boundary == "cancellation-read-failure":
                        assert cancellation.value.__cause__ is not None
                        retained_failures = tuple(
                            candidate
                            for candidate in iter_exception_tree(cancellation.value.__cause__)
                            if not isinstance(candidate, BaseExceptionGroup)
                        )
                        assert (
                            sum(
                                isinstance(candidate, OSError)
                                and str(candidate) == "post-claim checkpoint read failed"
                                for candidate in retained_failures
                            )
                            == 1
                        )
                store.load_checkpoint = original_load_checkpoint  # type: ignore[method-assign]
                after_cancellation = await original_load_checkpoint(session_id)
                assert after_cancellation is not None
                if claim_boundary == "cancellation-read-failure":
                    assert "pending_session_interrupt" in after_cancellation
                else:
                    assert "pending_session_interrupt" not in after_cancellation
                assert "incomplete_session_recovery_claim" not in after_cancellation
                store.transform_checkpoint_with_store_time = original_transform  # type: ignore[method-assign]
                store = await _reopen_store(session_store_case, store)
                recovery_app = CayuApp(session_store=store, enable_logging=False)
                recovery_app.register_provider(_UserInputRecoveryProvider(), default=True)
                recovery_app.register_agent(
                    AgentSpec(name="assistant", model="fake-model"),
                    tools=[UserInputTool()],
                )
                with suppress_interruption_cascade():
                    events = await _collect_events(recovery_app.interrupt_session(request))
                assert events[-1].type is EventType.SESSION_INTERRUPTED

            assert injected is True
            final_checkpoint = await store.load_checkpoint(session_id)
            assert final_checkpoint is not None
            assert "pending_session_interrupt" not in final_checkpoint
            assert "incomplete_session_recovery_claim" not in final_checkpoint
            records = [
                record
                for record in await store.query_events(
                    EventQuery(
                        session_id=session_id,
                        event_type=EventType.SESSION_INTERRUPTED,
                    )
                )
                if record.event.payload.get("interruption_type") == "operator_requested"
            ]
            assert len(records) == 1
        finally:
            release_acknowledgement.set()
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_settles_replacement_claim_when_terminal_event_wins(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        session_id = f"terminal-event-wins-replacement-claim-{session_store_case[0]}"
        try:
            app = CayuApp(session_store=store, enable_logging=False)
            app.register_provider(_UserInputRecoveryProvider(), default=True)
            app.register_agent(
                AgentSpec(name="assistant", model="fake-model"),
                tools=[UserInputTool()],
            )
            await _collect_events(
                app.run(
                    RunRequest(
                        session_id=session_id,
                        agent_name="assistant",
                        messages=[Message.text("user", "ask a question")],
                    )
                )
            )
            request = InterruptSessionRequest(
                session_id=session_id,
                reason="operator supersedes the paused question",
            )
            engine = app._session_engine
            original_transition = store.transition_status_and_checkpoint

            async def commit_supersession_then_stop(*args, **kwargs):
                transitioned = await original_transition(*args, **kwargs)
                if kwargs.get("to_status") is SessionStatus.INTERRUPTING:
                    raise _SimulatedProcessLoss("stop after supersession transition")
                return transitioned

            store.transition_status_and_checkpoint = (  # type: ignore[method-assign]
                commit_supersession_then_stop
            )
            with (
                suppress_interruption_cascade(),
                pytest.raises(_SimulatedProcessLoss, match="supersession transition"),
            ):
                await _collect_events(app.interrupt_session(request))
            store.transition_status_and_checkpoint = original_transition  # type: ignore[method-assign]

            def release_abandoned_claim(_session: Session, checkpoint):
                updated = dict({} if checkpoint is None else checkpoint)
                updated.pop("incomplete_session_recovery_claim", None)
                return updated

            await store.transform_checkpoint(session_id, release_abandoned_claim)

            interrupting = await store.load(session_id)
            checkpoint = await store.load_checkpoint(session_id)
            assert interrupting is not None
            assert interrupting.status is SessionStatus.INTERRUPTING
            assert checkpoint is not None
            expected_payload = checkpoint["pending_session_interrupt"]
            assert "incomplete_session_recovery_claim" not in checkpoint

            recovery = app._recovery_coordinator
            original_renewal = recovery.terminal_finalization.renew_claim
            persisted_event: Event | None = None

            async def publish_terminal_after_replacement_renewal(*args, **kwargs):
                nonlocal persisted_event
                renewal = await original_renewal(*args, **kwargs)
                if renewal is not None and persisted_event is None:
                    terminal_session = await store.update_status(
                        session_id,
                        SessionStatus.INTERRUPTED,
                    )
                    repair_event = recovery.terminal_finalization._repair_event(
                        session=terminal_session,
                        terminal_run_epoch=terminal_session.run_epoch,
                        terminal_timestamp=terminal_session.updated_at,
                        pending_interrupt_payload=expected_payload,
                        pending_action_interrupt_payload=None,
                        run_operation=None,
                    )
                    persisted_event = await recovery.terminal_finalization._persist_repair_event(
                        repair_event
                    )
                return renewal

            original_cancel_active_runs = engine._session_control.cancel_active_runs
            recovery.terminal_finalization.renew_claim = (  # type: ignore[method-assign]
                publish_terminal_after_replacement_renewal
            )
            engine._session_control.cancel_active_runs = (  # type: ignore[method-assign]
                lambda _session_id: True
            )
            try:
                with suppress_interruption_cascade():
                    replayed = await _collect_events(app.interrupt_session(request))
            finally:
                recovery.terminal_finalization.renew_claim = (  # type: ignore[method-assign]
                    original_renewal
                )
                engine._session_control.cancel_active_runs = (  # type: ignore[method-assign]
                    original_cancel_active_runs
                )

            assert persisted_event is not None
            assert (
                await _private_event_for_public_event(store, replayed[-1])
            ).id == persisted_event.id
            settled = await store.load_checkpoint(session_id)
            assert settled is not None
            assert "pending_session_interrupt" not in settled
            assert "incomplete_session_recovery_claim" not in settled
            assert not any(
                task.get_name() == f"cayu-terminal-finalization-heartbeat:{session_id}"
                and not task.done()
                for task in asyncio.all_tasks()
            )
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_user_input_supersession_retry_joins_live_finalizer(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        peer_store: SessionStore | None = None
        owner_release = asyncio.Event()
        allow_read_handoff = asyncio.Event()
        tasks: list[asyncio.Task[list[Event]]] = []
        session_id = f"user-input-supersession-live-finalizer-{session_store_case[0]}"
        try:
            app = CayuApp(session_store=store, enable_logging=False)
            app.register_provider(_UserInputRecoveryProvider(), default=True)
            app.register_agent(
                AgentSpec(name="assistant", model="fake-model"),
                tools=[UserInputTool()],
            )
            await _collect_events(
                app.run(
                    RunRequest(
                        session_id=session_id,
                        agent_name="assistant",
                        messages=[Message.text("user", "ask a question")],
                    )
                )
            )

            owner_started = asyncio.Event()
            original_terminal_preparation = (
                app._session_finalization.prepare_terminal_event_for_atomic_transition
            )

            async def hold_owner_before_terminal_event(*args, **kwargs):
                owner_started.set()
                await owner_release.wait()
                return await original_terminal_preparation(*args, **kwargs)

            app._session_finalization.prepare_terminal_event_for_atomic_transition = (  # type: ignore[method-assign]
                hold_owner_before_terminal_event
            )
            request = InterruptSessionRequest(
                session_id=session_id,
                reason="operator supersedes the paused question",
            )
            with suppress_interruption_cascade():
                owner_task = asyncio.create_task(_collect_events(app.interrupt_session(request)))
                tasks.append(owner_task)
                await asyncio.wait_for(owner_started.wait(), timeout=10)

                checkpoint = await store.load_checkpoint(session_id)
                assert checkpoint is not None
                original_claim = checkpoint["incomplete_session_recovery_claim"]
                assert "pending_session_interrupt" in checkpoint

                peer_store = await _open_peer_store(session_store_case, store)
                peer_app = CayuApp(
                    session_store=peer_store,
                    enable_logging=False,
                    clock=lambda: datetime(2200, 1, 1, tzinfo=UTC),
                )
                peer_app.register_provider(_UserInputRecoveryProvider(), default=True)
                peer_app.register_agent(
                    AgentSpec(name="assistant", model="fake-model"),
                    tools=[UserInputTool()],
                )
                peer_claim_finished = False
                join_read_started = asyncio.Event()
                original_peer_claim = peer_app._recovery_ownership.claim
                original_peer_load_checkpoint = peer_store.load_checkpoint

                async def observe_live_owner(*args, **kwargs):
                    nonlocal peer_claim_finished
                    claimed = await original_peer_claim(*args, **kwargs)
                    assert claimed is None
                    peer_claim_finished = True
                    return claimed

                async def complete_owner_between_join_reads(loaded_session_id: str):
                    checkpoint_before_handoff = await original_peer_load_checkpoint(
                        loaded_session_id
                    )
                    if (
                        peer_claim_finished
                        and not join_read_started.is_set()
                        and checkpoint_before_handoff is not None
                        and "pending_session_interrupt" in checkpoint_before_handoff
                    ):
                        join_read_started.set()
                        await allow_read_handoff.wait()
                        owner_release.set()
                        for _attempt in range(100):
                            checkpoint_after_handoff = await original_peer_load_checkpoint(
                                loaded_session_id
                            )
                            if (
                                checkpoint_after_handoff is not None
                                and "pending_session_interrupt" not in checkpoint_after_handoff
                            ):
                                return checkpoint_after_handoff
                            await asyncio.sleep(0.01)
                        raise TimeoutError("Live terminal finalizer did not clear its marker.")
                    return checkpoint_before_handoff

                peer_app._recovery_ownership.claim = (  # type: ignore[method-assign]
                    observe_live_owner
                )
                peer_store.load_checkpoint = (  # type: ignore[method-assign]
                    complete_owner_between_join_reads
                )
                retry_task = asyncio.create_task(
                    _collect_events(peer_app.interrupt_session(request))
                )
                tasks.append(retry_task)
                await asyncio.wait_for(join_read_started.wait(), timeout=10)
                assert not owner_task.done()
                assert not retry_task.done()
                retained = await original_peer_load_checkpoint(session_id)
                assert retained is not None
                assert retained["incomplete_session_recovery_claim"] == original_claim

                allow_read_handoff.set()
                owner_events, retry_events = await asyncio.wait_for(
                    asyncio.gather(owner_task, retry_task),
                    timeout=10,
                )
                assert owner_events[-1].id == retry_events[-1].id

            final_checkpoint = await store.load_checkpoint(session_id)
            assert final_checkpoint is not None
            assert "pending_session_interrupt" not in final_checkpoint
            assert "incomplete_session_recovery_claim" not in final_checkpoint
            terminal_records = [
                record
                for record in await store.query_events(
                    EventQuery(
                        session_id=session_id,
                        event_type=EventType.SESSION_INTERRUPTED,
                    )
                )
                if record.event.payload.get("interruption_type") == "operator_requested"
            ]
            assert len(terminal_records) == 1
            assert (
                terminal_records[0].event.id
                == (await _private_event_for_public_event(store, owner_events[-1])).id
            )
        finally:
            allow_read_handoff.set()
            owner_release.set()
            for task in tasks:
                if not task.done():
                    task.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            if peer_store is not None and peer_store is not store:
                await _close_store(peer_store)
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_does_not_dispatch_expired_terminal_finalization_claim(
    session_store_case,
    monkeypatch,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        session_id = f"expired-terminal-finalization-claim-{session_store_case[0]}"
        try:
            monkeypatch.setattr(
                recovery_ownership_module,
                "_INCOMPLETE_RECOVERY_CLAIM_LEASE",
                timedelta(milliseconds=100),
            )
            app = CayuApp(
                session_store=store,
                enable_logging=False,
            )
            app.register_provider(_UserInputRecoveryProvider(), default=True)
            app.register_agent(
                AgentSpec(name="assistant", model="fake-model"),
                tools=[UserInputTool()],
            )
            await _collect_events(
                app.run(
                    RunRequest(
                        session_id=session_id,
                        agent_name="assistant",
                        messages=[Message.text("user", "ask a question")],
                    )
                )
            )
            request = InterruptSessionRequest(
                session_id=session_id,
                reason="operator supersedes a paused question",
            )
            original_transition = store.transition_status_and_checkpoint
            transition_delayed = False
            expired_claim_observations: list[datetime] = []

            async def expire_claim_after_atomic_transition(*args, **kwargs):
                nonlocal transition_delayed
                transitioned = await original_transition(*args, **kwargs)
                if kwargs.get("to_status") is SessionStatus.INTERRUPTING and not transition_delayed:
                    transition_delayed = True

                    def expire_claim(_session, checkpoint, store_now):
                        assert checkpoint is not None
                        updated = copy.deepcopy(checkpoint)
                        claim = updated["incomplete_session_recovery_claim"]
                        expired_claim_observations.append(store_now)
                        claim["claimed_at"] = (store_now - timedelta(minutes=10)).isoformat()
                        claim["claim_expires_at"] = (store_now - timedelta(minutes=5)).isoformat()
                        return updated

                    await store.transform_checkpoint_with_store_time(
                        session_id,
                        expire_claim,
                    )
                return transitioned

            store.transition_status_and_checkpoint = (  # type: ignore[method-assign]
                expire_claim_after_atomic_transition
            )
            with (
                suppress_interruption_cascade(),
                pytest.raises(
                    recovery_coordinator_module._IncompleteRecoveryClaimLost,
                    match=(
                        "expired before live dispatch|stopped before its store lease|"
                        "acknowledgement consumed its lease"
                    ),
                ),
            ):
                await _collect_events(app.interrupt_session(request))
            assert transition_delayed is True

            retained_session = await store.load(session_id)
            retained = await store.load_checkpoint(session_id)
            assert retained_session is not None
            assert retained_session.status is SessionStatus.INTERRUPTING
            assert retained is not None
            assert "pending_session_interrupt" in retained
            retained_claim = retained["incomplete_session_recovery_claim"]
            assert len(expired_claim_observations) == 1
            assert (
                datetime.fromisoformat(retained_claim["claim_expires_at"])
                < (expired_claim_observations[0])
            )
            terminal_records = [
                record
                for record in await store.query_events(
                    EventQuery(
                        session_id=session_id,
                        event_type=EventType.SESSION_INTERRUPTED,
                    )
                )
                if record.event.payload.get("interruption_type") == "operator_requested"
            ]
            assert terminal_records == []

            store.transition_status_and_checkpoint = original_transition  # type: ignore[method-assign]
            monkeypatch.setattr(
                recovery_ownership_module,
                "_INCOMPLETE_RECOVERY_CLAIM_LEASE",
                timedelta(minutes=5),
            )
            with suppress_interruption_cascade():
                retried = await _collect_events(app.interrupt_session(request))
            assert retried[-1].type is EventType.SESSION_INTERRUPTED
            final_checkpoint = await store.load_checkpoint(session_id)
            assert final_checkpoint is not None
            assert "pending_session_interrupt" not in final_checkpoint
            assert "incomplete_session_recovery_claim" not in final_checkpoint
            terminal_records = [
                record
                for record in await store.query_events(
                    EventQuery(
                        session_id=session_id,
                        event_type=EventType.SESSION_INTERRUPTED,
                    )
                )
                if record.event.payload.get("interruption_type") == "operator_requested"
            ]
            assert len(terminal_records) == 1
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_replaces_expired_recovery_claim_during_pause_supersession(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        session_id = f"expired-pause-supersession-claim-{session_store_case[0]}"
        try:
            app = CayuApp(session_store=store, enable_logging=False)
            app.register_provider(_UserInputRecoveryProvider(), default=True)
            app.register_agent(
                AgentSpec(name="assistant", model="fake-model"),
                tools=[UserInputTool()],
            )
            await _collect_events(
                app.run(
                    RunRequest(
                        session_id=session_id,
                        agent_name="assistant",
                        messages=[Message.text("user", "ask a question")],
                    )
                )
            )
            paused = await store.load(session_id)
            assert paused is not None
            assert paused.status is SessionStatus.INTERRUPTED
            expired_at = datetime.now(UTC) - timedelta(minutes=5)

            def retain_expired_claim(_session, checkpoint):
                assert checkpoint is not None
                updated = copy.deepcopy(checkpoint)
                updated["incomplete_session_recovery_claim"] = {
                    "version": 1,
                    "claim_id": "expired-predecessor",
                    "claimed_at": (expired_at - timedelta(minutes=5)).isoformat(),
                    "claim_expires_at": expired_at.isoformat(),
                }
                return updated

            await store.transform_checkpoint(session_id, retain_expired_claim)
            request = InterruptSessionRequest(
                session_id=session_id,
                reason="operator supersedes expired recovery owner",
            )
            with suppress_interruption_cascade():
                events = await _collect_events(app.interrupt_session(request))
            assert events[-1].type is EventType.SESSION_INTERRUPTED
            assert events[-1].payload["interruption_type"] == "operator_requested"
            checkpoint = await store.load_checkpoint(session_id)
            assert checkpoint is not None
            assert "pending_user_input" not in checkpoint
            assert "pending_session_interrupt" not in checkpoint
            assert "incomplete_session_recovery_claim" not in checkpoint
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_status_only_finalization_requires_exact_live_claim(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        session_id = f"status-only-finalization-claim-{session_store_case[0]}"
        try:
            created = await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "create")],
                ),
                identity=_identity(),
            )
            interrupting = await store.update_status(created.id, SessionStatus.INTERRUPTING)

            def install_claim(claim_id: str, *, expired: bool = False):
                def transform(_session, checkpoint, store_now):
                    updated = {} if checkpoint is None else copy.deepcopy(checkpoint)
                    updated["incomplete_session_recovery_claim"] = {
                        "version": 1,
                        "claim_id": claim_id,
                        "claimed_at": (
                            store_now - timedelta(minutes=10) if expired else store_now
                        ).isoformat(),
                        "claim_expires_at": (
                            store_now - timedelta(minutes=5)
                            if expired
                            else store_now + timedelta(minutes=5)
                        ).isoformat(),
                    }
                    return updated

                return transform

            await store.transform_checkpoint_with_store_time(
                session_id,
                install_claim("stale-finalizer"),
            )
            stale_session = interrupting.model_copy(deep=True)
            await store.transform_checkpoint_with_store_time(
                session_id,
                install_claim("replacement-finalizer"),
            )
            app = CayuApp(
                session_store=store,
                enable_logging=False,
                clock=lambda: datetime(2200, 1, 1, tzinfo=UTC),
            )

            with pytest.raises(
                SessionRunFenced,
                match="lost its exact terminal recovery claim",
            ):
                await app._session_finalization.publish_interaction_transition(
                    session=stale_session,
                    agent_name="assistant",
                    environment_name=None,
                    to_status=SessionStatus.INTERRUPTED,
                    expected_recovery_claim_id="stale-finalizer",
                )
            retained = await store.load(session_id)
            assert retained is not None
            assert retained.status is SessionStatus.INTERRUPTING

            (
                finalized,
                event,
                status_changed,
            ) = await app._session_finalization.publish_interaction_transition(
                session=stale_session,
                agent_name="assistant",
                environment_name=None,
                to_status=SessionStatus.INTERRUPTED,
                expected_recovery_claim_id="replacement-finalizer",
            )
            assert finalized.status is SessionStatus.INTERRUPTED
            assert event is None
            assert status_changed is True

            expired_session_id = f"{session_id}-expired"
            expired_created = await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id=expired_session_id,
                    messages=[Message.text("user", "create expired")],
                ),
                identity=_identity(),
            )
            expired_interrupting = await store.update_status(
                expired_created.id,
                SessionStatus.INTERRUPTING,
            )
            await store.transform_checkpoint_with_store_time(
                expired_session_id,
                install_claim("expired-finalizer", expired=True),
            )
            past_clock_app = CayuApp(
                session_store=store,
                enable_logging=False,
                clock=lambda: datetime(2000, 1, 1, tzinfo=UTC),
            )
            with pytest.raises(
                SessionRunFenced,
                match="lost its exact terminal recovery claim",
            ):
                await past_clock_app._session_finalization.publish_interaction_transition(
                    session=expired_interrupting,
                    agent_name="assistant",
                    environment_name=None,
                    to_status=SessionStatus.INTERRUPTED,
                    expected_recovery_claim_id="expired-finalizer",
                )
            retained_expired = await store.load(expired_session_id)
            assert retained_expired is not None
            assert retained_expired.status is SessionStatus.INTERRUPTING
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_interaction_claim_uses_store_clock(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        session_id = f"interaction-claim-clock-{session_store_case[0]}"
        interaction_id = f"interaction-claim-clock-{session_store_case[0]}"
        try:
            started = Event(
                id=f"evt-{interaction_id}-started",
                type=EventType.INTERACTION_STARTED,
                session_id=session_id,
                interaction_id=interaction_id,
            )
            await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "start")],
                ),
                identity=_identity(),
                interaction_started_event=started,
                interaction_source_messages=[Message.text("user", "start")],
            )
            claim_times: list[datetime] = []

            def install_claim(_session, checkpoint, store_now):
                claim_times.append(store_now)
                return {
                    **({} if checkpoint is None else checkpoint),
                    "incomplete_session_recovery_claim": {
                        "version": 1,
                        "claim_id": "store-clock-claim",
                        "claimed_at": store_now.isoformat(),
                        "claim_expires_at": (store_now + timedelta(minutes=5)).isoformat(),
                    },
                }

            await store.transform_checkpoint_with_store_time(
                session_id,
                install_claim,
            )
            assert len(claim_times) == 1
            interrupted = Event(
                id=f"evt-{interaction_id}-interrupted",
                type=EventType.INTERACTION_INTERRUPTED,
                session_id=session_id,
                interaction_id=interaction_id,
            )
            transition = InteractionTransitionSpec(
                event=interrupted,
                from_statuses=(SessionStatus.RUNNING,),
                to_status=SessionStatus.INTERRUPTED,
            )
            with pytest.raises(
                SessionRunFenced,
                match="owned by another terminal recovery claim",
            ):
                await store.publish_interaction_transition(
                    session_id,
                    event=interrupted,
                    from_statuses={SessionStatus.RUNNING},
                    to_status=SessionStatus.INTERRUPTED,
                )
            retained_before_claimed_publication = await store.load(session_id)
            assert retained_before_claimed_publication is not None
            assert retained_before_claimed_publication.status is SessionStatus.RUNNING
            assert (
                await store.query_events(EventQuery(session_id=session_id, event_id=interrupted.id))
                == []
            )
            published = await store.publish_interaction_transition(
                session_id,
                event=interrupted,
                from_statuses={SessionStatus.RUNNING},
                to_status=SessionStatus.INTERRUPTED,
                expected_recovery_claim_id="store-clock-claim",
            )
            assert published.replayed is False
            assert published.session.status is SessionStatus.INTERRUPTED
            loaded_receipt = await store.load_interaction_transition_receipt(
                session_id,
                transition=transition,
                expected_recovery_claim_id="store-clock-claim",
            )
            assert loaded_receipt is not None
            assert loaded_receipt.transition == transition

            with pytest.raises(
                SessionRunFenced,
                match="another recovery authority",
            ):
                await store.publish_interaction_transition(
                    session_id,
                    event=interrupted,
                    from_statuses={SessionStatus.RUNNING},
                    to_status=SessionStatus.INTERRUPTED,
                )
            with pytest.raises(
                SessionRunFenced,
                match="another recovery authority",
            ):
                await store.load_interaction_transition_receipt(
                    session_id,
                    transition=transition,
                )
            with pytest.raises(
                SessionRunFenced,
                match="another recovery authority",
            ):
                await store.publish_interaction_transition(
                    session_id,
                    event=interrupted,
                    from_statuses={SessionStatus.RUNNING},
                    to_status=SessionStatus.INTERRUPTED,
                    expected_recovery_claim_id="substituted-claim",
                )

            def expire_original_claim(_session, checkpoint, store_now):
                assert checkpoint is not None
                updated = copy.deepcopy(checkpoint)
                updated["incomplete_session_recovery_claim"] = {
                    "version": 1,
                    "claim_id": "store-clock-claim",
                    "claimed_at": (store_now - timedelta(minutes=10)).isoformat(),
                    "claim_expires_at": (store_now - timedelta(minutes=5)).isoformat(),
                }
                return updated

            await store.transform_checkpoint_with_store_time(
                session_id,
                expire_original_claim,
            )
            replayed = await store.publish_interaction_transition(
                session_id,
                event=interrupted,
                from_statuses={SessionStatus.RUNNING},
                to_status=SessionStatus.INTERRUPTED,
                expected_recovery_claim_id="store-clock-claim",
            )
            assert replayed.replayed is True
            assert replayed.event == interrupted

            def replace_claim(_session, checkpoint, store_now):
                assert checkpoint is not None
                updated = copy.deepcopy(checkpoint)
                updated["incomplete_session_recovery_claim"] = {
                    "version": 1,
                    "claim_id": "replacement-claim",
                    "claimed_at": store_now.isoformat(),
                    "claim_expires_at": (store_now + timedelta(minutes=5)).isoformat(),
                }
                return updated

            await store.transform_checkpoint_with_store_time(session_id, replace_claim)
            with pytest.raises(
                SessionRunFenced,
                match="superseded by another recovery authority",
            ):
                await store.load_interaction_transition_receipt(
                    session_id,
                    transition=transition,
                    expected_recovery_claim_id="store-clock-claim",
                )
            with pytest.raises(
                SessionRunFenced,
                match="superseded by another recovery authority",
            ):
                await store.publish_interaction_transition(
                    session_id,
                    event=interrupted,
                    from_statuses={SessionStatus.RUNNING},
                    to_status=SessionStatus.INTERRUPTED,
                    expected_recovery_claim_id="store-clock-claim",
                )

            expired_session_id = f"{session_id}-expired"
            expired_interaction_id = f"{interaction_id}-expired"
            expired_started = Event(
                id=f"evt-{expired_interaction_id}-started",
                type=EventType.INTERACTION_STARTED,
                session_id=expired_session_id,
                interaction_id=expired_interaction_id,
            )
            await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id=expired_session_id,
                    messages=[Message.text("user", "expired claim")],
                ),
                identity=_identity(),
                interaction_started_event=expired_started,
                interaction_source_messages=[Message.text("user", "expired claim")],
            )

            def install_expired_claim(_session, checkpoint, store_now):
                return {
                    **({} if checkpoint is None else checkpoint),
                    "incomplete_session_recovery_claim": {
                        "version": 1,
                        "claim_id": "expired-store-clock-claim",
                        "claimed_at": (store_now - timedelta(minutes=10)).isoformat(),
                        "claim_expires_at": (store_now - timedelta(minutes=5)).isoformat(),
                    },
                }

            await store.transform_checkpoint_with_store_time(
                expired_session_id,
                install_expired_claim,
            )
            expired_event = Event(
                id=f"evt-{expired_interaction_id}-interrupted",
                type=EventType.INTERACTION_INTERRUPTED,
                session_id=expired_session_id,
                interaction_id=expired_interaction_id,
            )
            with pytest.raises(
                SessionRunFenced,
                match="lost its exact terminal recovery claim",
            ):
                await store.publish_interaction_transition(
                    expired_session_id,
                    event=expired_event,
                    from_statuses={SessionStatus.RUNNING},
                    to_status=SessionStatus.INTERRUPTED,
                    expected_recovery_claim_id="expired-store-clock-claim",
                )
            retained_expired = await store.load(expired_session_id)
            assert retained_expired is not None
            assert retained_expired.status is SessionStatus.RUNNING
            assert (
                await store.query_events(
                    EventQuery(
                        session_id=expired_session_id,
                        event_id=expired_event.id,
                    )
                )
                == []
            )
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_sqlite_interaction_claim_store_clock_fatal_signal_releases_writer_transaction(
    tmp_path,
) -> None:
    class FatalClaimClockSignal(BaseException):
        pass

    async def run() -> None:
        database = tmp_path / "fatal-interaction-claim-clock.sqlite"
        observed_at = datetime(2040, 1, 1, tzinfo=UTC)
        fail_clock = {"value": False}

        def ownership_clock() -> datetime:
            if fail_clock["value"]:
                raise FatalClaimClockSignal("claim clock stopped")
            return observed_at

        store = SQLiteSessionStore(database, ownership_clock=ownership_clock)
        peer: SQLiteSessionStore | None = None
        session_id = "fatal-interaction-claim-clock"
        interaction_id = "fatal-interaction-claim-clock"
        try:
            started = Event(
                id=f"evt-{interaction_id}-started",
                type=EventType.INTERACTION_STARTED,
                session_id=session_id,
                interaction_id=interaction_id,
            )
            await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "start")],
                ),
                identity=_identity(),
                interaction_started_event=started,
                interaction_source_messages=[Message.text("user", "start")],
            )
            await store.checkpoint(
                session_id,
                {
                    "incomplete_session_recovery_claim": {
                        "version": 1,
                        "claim_id": "fatal-clock-claim",
                        "claimed_at": observed_at.isoformat(),
                        "claim_expires_at": (observed_at + timedelta(minutes=5)).isoformat(),
                    }
                },
            )
            interrupted = Event(
                id=f"evt-{interaction_id}-interrupted",
                type=EventType.INTERACTION_INTERRUPTED,
                session_id=session_id,
                interaction_id=interaction_id,
            )

            fail_clock["value"] = True
            with pytest.raises(FatalClaimClockSignal, match="claim clock stopped"):
                await store.publish_interaction_transition(
                    session_id,
                    event=interrupted,
                    from_statuses={SessionStatus.RUNNING},
                    to_status=SessionStatus.INTERRUPTED,
                    expected_recovery_claim_id="fatal-clock-claim",
                )
            assert store._connection.in_transaction is False

            fail_clock["value"] = False
            published = await store.publish_interaction_transition(
                session_id,
                event=interrupted,
                from_statuses={SessionStatus.RUNNING},
                to_status=SessionStatus.INTERRUPTED,
                expected_recovery_claim_id="fatal-clock-claim",
            )
            assert published.session.status is SessionStatus.INTERRUPTED
            assert published.replayed is False

            peer = SQLiteSessionStore(
                database,
                public_authority_alias_codec=store.public_authority_alias_codec,
            )
            await peer.checkpoint(session_id, {"peer_writer_completed": True})
            assert await peer.load_checkpoint(session_id) == {"peer_writer_completed": True}
        finally:
            if peer is not None:
                await peer.close()
            await store.close()

    asyncio.run(run())


@pytest.mark.parametrize(
    "claim_boundary",
    [
        pytest.param("status-transition", id="atomic-interrupt-claim-installation"),
        pytest.param("checkpoint-transform", id="claim-acquisition-or-renewal"),
    ],
)
def test_sqlite_terminal_claim_callback_fatal_signal_releases_writer_transaction(
    tmp_path,
    claim_boundary: str,
) -> None:
    class FatalClaimCallbackSignal(BaseException):
        pass

    async def run() -> None:
        database = tmp_path / f"fatal-{claim_boundary}.sqlite"
        store = SQLiteSessionStore(database)
        peer: SQLiteSessionStore | None = None
        session_id = f"fatal-{claim_boundary}"
        interaction_id = f"fatal-{claim_boundary}"
        try:
            started = Event(
                id=f"evt-{interaction_id}-started",
                type=EventType.INTERACTION_STARTED,
                session_id=session_id,
                interaction_id=interaction_id,
            )
            await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "start")],
                ),
                identity=_identity(),
                interaction_started_event=started,
                interaction_source_messages=[Message.text("user", "start")],
            )

            def fail_claim_callback(_session, _checkpoint):
                raise FatalClaimCallbackSignal("terminal claim callback stopped")

            with pytest.raises(
                FatalClaimCallbackSignal,
                match="terminal claim callback stopped",
            ):
                if claim_boundary == "status-transition":
                    await store.transition_status_and_checkpoint(
                        session_id,
                        from_statuses={SessionStatus.RUNNING},
                        to_status=SessionStatus.INTERRUPTING,
                        checkpoint_transform=fail_claim_callback,
                    )
                else:
                    await store.transform_checkpoint(session_id, fail_claim_callback)
            assert store._connection.in_transaction is False

            if claim_boundary == "status-transition":
                transitioned = await store.transition_status_and_checkpoint(
                    session_id,
                    from_statuses={SessionStatus.RUNNING},
                    to_status=SessionStatus.INTERRUPTING,
                    checkpoint_transform=lambda _session, checkpoint: checkpoint,
                )
                assert transitioned.status is SessionStatus.INTERRUPTING
            else:
                await store.transform_checkpoint(
                    session_id,
                    lambda _session, checkpoint: {
                        **({} if checkpoint is None else checkpoint),
                        "claim_callback_retried": True,
                    },
                )

            peer = SQLiteSessionStore(
                database,
                public_authority_alias_codec=store.public_authority_alias_codec,
            )
            await peer.checkpoint(session_id, {"peer_writer_completed": True})
            assert await peer.load_checkpoint(session_id) == {"peer_writer_completed": True}
        finally:
            if peer is not None:
                await peer.close()
            await store.close()

    asyncio.run(run())


def test_sqlite_public_interrupt_fatal_claim_callback_releases_writer_transaction(
    tmp_path,
) -> None:
    class FatalClaimCallbackSignal(BaseException):
        pass

    async def run() -> None:
        database = tmp_path / "fatal-public-interrupt-claim.sqlite"
        store = SQLiteSessionStore(database)
        peer: SQLiteSessionStore | None = None
        session_id = "fatal-public-interrupt-claim"
        try:
            app = CayuApp(session_store=store, enable_logging=False)
            app.register_provider(_UserInputRecoveryProvider(), default=True)
            app.register_agent(AgentSpec(name="assistant", model="fake-model"))
            await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "start")],
                ),
                identity=_identity(),
            )
            original_transition = store.transition_status_and_checkpoint

            async def fail_public_claim_callback(*args, **kwargs):
                transform_key = (
                    "store_time_checkpoint_transform"
                    if kwargs.get("store_time_checkpoint_transform") is not None
                    else "checkpoint_transform"
                )
                checkpoint_transform = kwargs[transform_key]

                def fail_after_claim_validation(*transform_args):
                    checkpoint_transform(*transform_args)
                    raise FatalClaimCallbackSignal("public terminal claim callback stopped")

                return await original_transition(
                    *args,
                    **{
                        **kwargs,
                        transform_key: fail_after_claim_validation,
                    },
                )

            store.transition_status_and_checkpoint = (  # type: ignore[method-assign]
                fail_public_claim_callback
            )
            with (
                suppress_interruption_cascade(),
                pytest.raises(
                    FatalClaimCallbackSignal,
                    match="public terminal claim callback stopped",
                ),
            ):
                await _collect_events(
                    app.interrupt_session(
                        InterruptSessionRequest(
                            session_id=session_id,
                            reason="operator stop",
                        )
                    )
                )
            assert store._connection.in_transaction is False

            store.transition_status_and_checkpoint = original_transition  # type: ignore[method-assign]
            with suppress_interruption_cascade():
                events = await _collect_events(
                    app.interrupt_session(
                        InterruptSessionRequest(
                            session_id=session_id,
                            reason="operator stop",
                        )
                    )
                )
            assert events[-1].type is EventType.SESSION_INTERRUPTED

            peer = SQLiteSessionStore(
                database,
                public_authority_alias_codec=store.public_authority_alias_codec,
            )
            await peer.checkpoint(session_id, {"peer_writer_completed": True})
            assert await peer.load_checkpoint(session_id) == {"peer_writer_completed": True}
        finally:
            if peer is not None:
                await peer.close()
            await store.close()

    asyncio.run(run())


def test_incomplete_recovery_success_does_not_suppress_simultaneous_claim_loss(
    monkeypatch,
) -> None:
    async def run() -> None:
        store = InMemorySessionStore()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(_UserInputRecoveryProvider(), default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[UserInputTool()],
        )
        session_id = "simultaneous-terminal-recovery-claim-loss"
        await _collect_events(
            app.run(
                RunRequest(
                    session_id=session_id,
                    agent_name="assistant",
                    messages=[Message.text("user", "ask a question")],
                )
            )
        )
        session = await store.load(session_id)
        assert session is not None
        recovery = app._recovery_coordinator
        release = asyncio.Event()

        async def lose_claim_with_recovery_completion(**_kwargs):
            await release.wait()
            raise recovery_coordinator_module._IncompleteRecoveryClaimLost(
                "peer replaced the recovery claim"
            )

        monkeypatch.setattr(
            recovery._recovery_ownership,
            "heartbeat",
            lose_claim_with_recovery_completion,
        )

        async def finish_recovery() -> bool:
            release.set()
            await asyncio.sleep(0)
            return True

        claim = await recovery._recovery_ownership.claim(
            session=session,
            inactive_for_seconds=None,
            execution_profile_snapshot=active_invocation_execution_profile_from_checkpoint(
                await store.load_checkpoint(session.id)
            ),
        )
        assert claim is not None
        with pytest.raises(
            recovery_coordinator_module._IncompleteRecoveryClaimLost,
            match="peer replaced",
        ):
            await recovery._recovery_ownership.run_with_heartbeat(
                claim=claim,
                recovery=finish_recovery,
            )

    asyncio.run(run())


@pytest.mark.parametrize("with_diagnostics", [False, True])
def test_session_store_conformance_stops_live_finalizer_after_terminal_claim_loss(
    session_store_case,
    monkeypatch,
    with_diagnostics,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        peer_store: SessionStore | None = None
        session_id = f"lost-live-terminal-finalization-claim-{session_store_case[0]}"
        release_handler = asyncio.Event()
        tasks: list[asyncio.Task[list[Event]]] = []
        try:
            monkeypatch.setattr(
                recovery_ownership_module,
                "_INCOMPLETE_RECOVERY_CLAIM_LEASE",
                timedelta(seconds=2),
            )
            monkeypatch.setattr(
                recovery_ownership_module,
                "_INCOMPLETE_RECOVERY_CLAIM_HEARTBEAT_INTERVAL_SECONDS",
                0.02,
            )
            monkeypatch.setattr(
                recovery_ownership_module,
                "_INCOMPLETE_RECOVERY_CLAIM_HEARTBEAT_RETRY_SECONDS",
                0.01,
            )
            awaiting_event_id: str | None = None
            original_publish = store.publish_runtime_publication
            original_side_effect_claim = store.claim_persisted_event_side_effect
            fan_out_started = asyncio.Event()

            async def remember_awaiting_event(session_id: str, **kwargs):
                nonlocal awaiting_event_id
                result = await original_publish(session_id, **kwargs)
                if kwargs["request"].kind == "user-input-open":
                    awaiting_event_id = next(
                        event.id
                        for event in kwargs["request"].events
                        if event.type is EventType.SESSION_AWAITING_USER_INPUT
                    )
                return result

            async def block_awaiting_fan_out(*, session_id: str, event_id: str):
                if event_id == awaiting_event_id:
                    fan_out_started.set()
                    await asyncio.Event().wait()
                return await original_side_effect_claim(
                    session_id=session_id,
                    event_id=event_id,
                )

            store.publish_runtime_publication = remember_awaiting_event  # type: ignore[method-assign]
            store.claim_persisted_event_side_effect = block_awaiting_fan_out  # type: ignore[method-assign]
            app = CayuApp(session_store=store, enable_logging=False)
            app.register_provider(_UserInputRecoveryProvider(), default=True)
            app.register_agent(
                AgentSpec(name="assistant", model="fake-model"),
                tools=[UserInputTool()],
            )
            handler_entered = asyncio.Event()
            original_handler = app._session_finalization.handle_session_interrupted

            async def hold_live_handler(*args, **kwargs):
                if (
                    kwargs["session"].id
                    in app._session_engine._session_control._terminal_finalization_claim_handoffs
                ):
                    handler_entered.set()
                    await release_handler.wait()
                if with_diagnostics:
                    kwargs["interaction_transition_failures"] = (
                        {"error_type": "RuntimeError", "message": "transition cleanup failed"},
                    )
                async for event in original_handler(*args, **kwargs):
                    yield event

            app._session_finalization.handle_session_interrupted = (  # type: ignore[method-assign]
                hold_live_handler
            )
            running = asyncio.create_task(
                _collect_events(
                    app.run(
                        RunRequest(
                            session_id=session_id,
                            agent_name="assistant",
                            messages=[Message.text("user", "ask a question")],
                        )
                    )
                )
            )
            tasks.append(running)
            await asyncio.wait_for(fan_out_started.wait(), timeout=10)
            request = InterruptSessionRequest(
                session_id=session_id,
                reason="operator supersedes active pause",
            )
            with suppress_interruption_cascade():
                interrupting = asyncio.create_task(_collect_events(app.interrupt_session(request)))
                tasks.append(interrupting)
                await asyncio.wait_for(handler_entered.wait(), timeout=10)
                checkpoint = await store.load_checkpoint(session_id)
                assert checkpoint is not None
                lost_claim = checkpoint["incomplete_session_recovery_claim"]
                lost_claim_id = lost_claim["claim_id"]

                recovery = app._recovery_coordinator
                original_renew = recovery._recovery_ownership.renew_claim

                async def lose_live_claim(
                    renewed_session_id: str,
                    renewed_claim_id: str,
                ):
                    if renewed_session_id == session_id and renewed_claim_id == lost_claim_id:
                        return None
                    return await original_renew(
                        renewed_session_id,
                        renewed_claim_id,
                    )

                recovery._recovery_ownership.renew_claim = (  # type: ignore[method-assign]
                    lose_live_claim
                )
                handoff = (
                    app._session_engine._session_control._terminal_finalization_claim_handoffs[
                        session_id
                    ]
                )
                with pytest.raises(recovery_coordinator_module._IncompleteRecoveryClaimLost):
                    await asyncio.wait_for(
                        asyncio.shield(handoff.heartbeat_task),
                        timeout=5,
                    )
                await asyncio.sleep(2.1)

                peer_store = await _open_peer_store(session_store_case, store)
                peer_app = CayuApp(session_store=peer_store, enable_logging=False)
                peer_app.register_provider(_UserInputRecoveryProvider(), default=True)
                peer_app.register_agent(
                    AgentSpec(name="assistant", model="fake-model"),
                    tools=[UserInputTool()],
                )
                peer = asyncio.create_task(_collect_events(peer_app.interrupt_session(request)))
                tasks.append(peer)
                peer_events = await asyncio.wait_for(peer, timeout=10)
                assert peer_events[-1].type is EventType.SESSION_INTERRUPTED
                peer_checkpoint = await store.load_checkpoint(session_id)
                peer_session = await store.load(session_id)
                peer_records = await store.load_events(session_id)

                release_handler.set()
                run_outcome, interrupt_events = await asyncio.wait_for(
                    asyncio.gather(running, interrupting, return_exceptions=True),
                    timeout=10,
                )
            assert isinstance(run_outcome, BaseException), repr(
                [(event.id, event.type, event.payload) for event in run_outcome[-2:]]
            )
            assert isinstance(
                interrupt_events, recovery_coordinator_module._IncompleteRecoveryClaimLost
            )
            settled_session = await store.load(session_id)
            assert peer_session is not None and settled_session is not None
            # The original execution lease is released when its run unwinds;
            # that fence increment must be the only session mutation.
            assert settled_session.run_epoch == peer_session.run_epoch + 1
            assert settled_session.model_copy(update={"run_epoch": peer_session.run_epoch}) == (
                peer_session
            )
            assert await store.load_events(session_id) == peer_records
            final_checkpoint = await store.load_checkpoint(session_id)
            assert final_checkpoint is not None
            assert peer_checkpoint is not None
            assert {
                key: value
                for key, value in final_checkpoint.items()
                if key != "invocation_lifecycle_receipt"
            } == {
                key: value
                for key, value in peer_checkpoint.items()
                if key != "invocation_lifecycle_receipt"
            }
            prior_receipts = peer_checkpoint["invocation_lifecycle_receipt"]["receipts"]
            final_receipts = final_checkpoint["invocation_lifecycle_receipt"]["receipts"]
            assert all(receipt in final_receipts for receipt in prior_receipts)
            [release_receipt] = [
                receipt for receipt in final_receipts if receipt not in prior_receipts
            ]
            assert release_receipt["kind"] == "release"
            assert release_receipt["session_id"] == session_id
            assert release_receipt["session_instance_id"] == peer_session.instance_id
            assert (
                release_receipt["active_profile"]
                == peer_checkpoint["active_invocation_execution_profile"]
            )
            assert release_receipt["result_session"] == settled_session.model_dump(mode="json")
            assert "pending_session_interrupt" not in final_checkpoint
            assert "incomplete_session_recovery_claim" not in final_checkpoint
            assert (
                session_id
                not in app._session_engine._session_control._terminal_finalization_claim_handoffs
            )
            terminal_records = [
                record
                for record in await store.query_events(
                    EventQuery(
                        session_id=session_id,
                        event_type=EventType.SESSION_INTERRUPTED,
                    )
                )
                if record.event.payload.get("interruption_type") == "operator_requested"
            ]
            assert len(terminal_records) == 1
        finally:
            release_handler.set()
            for task in tasks:
                if not task.done():
                    task.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            if peer_store is not None and peer_store is not store:
                await _close_store(peer_store)
            await _close_store(store)

    asyncio.run(run())

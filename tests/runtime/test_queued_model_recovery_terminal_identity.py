"""A lost queued model dispatch keeps its exact receipt and acknowledgement identity."""

from __future__ import annotations

import asyncio
import os
from datetime import UTC, datetime, timedelta

import pytest
from tests.runtime.runtime_provider_support import VersionedFakeProvider
from tests.runtime.test_orphaned_session_reconciliation import _orphan, _request, _slot
from tests.runtime.test_orphaned_session_reconciliation import database_url as database_url

from cayu import AgentSpec, CayuApp, EventType, Message, SessionStatus
from cayu.providers.base import ModelStreamEvent
from cayu.sessions._terminal_evidence import (
    _queued_dispatch_terminal_receipts_from_checkpoint,
    _session_run_operation_from_checkpoint,
)
from cayu.storage import open_application_stores
from cayu.tasks import TaskStatus, reconcile_abandoned_tasks
from cayu.tasks.dispatch import (
    DispatchRequest,
    DispatchStatus,
    TaskStoreDispatcher,
    _existing_queued_dispatch_envelope,
    _QueuedDispatchSettlementState,
)


class _QueuedModel(VersionedFakeProvider):
    def __init__(self, *, crash=False):
        super().__init__([])
        self.crash = crash
        self.calls = 0

    async def stream(self, request):
        self.calls += 1
        if not self.crash:
            raise AssertionError("Recovery must not redispatch the queued model")
        if self.calls == 2:
            os._exit(137)
        yield ModelStreamEvent.text_delta("initial response")
        yield ModelStreamEvent.completed({"finish_reason": "stop"})


def _app(stores, *, crash=False):
    dispatcher = TaskStoreDispatcher(stores.task_store, lease_seconds=1)
    app = CayuApp(
        session_store=stores.session_store,
        task_store=stores.task_store,
        dispatcher=dispatcher,
        enable_logging=False,
    )
    provider = _QueuedModel(crash=crash)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="scanner", model="fake-model"))
    slot = _slot()
    app.register_environment_factory(slot.environment.spec, slot)
    return app, provider, dispatcher


def _crash_queued_model():
    async def scenario():
        stores = open_application_stores(
            os.environ["CAYU_ORPHAN_TEST_DATABASE_URL"],
            sqlite_path=os.environ["CAYU_ORPHAN_TEST_STORE"],
            event_watchers=False,
        )
        app, _, dispatcher = _app(stores, crash=True)
        async for _ in app.run(_request("scan")):
            pass
        submitted = await app.dispatch(
            DispatchRequest(
                session_id="scan",
                dispatch_id="queued-model",
                messages=[Message.text("user", "queued request")],
            )
        )
        assert submitted.status is DispatchStatus.SUBMITTED
        await dispatcher.process_next(app, worker_id="original")
        raise AssertionError("Queued dispatch did not reach the crashing model")

    asyncio.run(scenario())


def _crash_after_queued_model_settlement():
    import cayu.runtime._recovery_ownership as recovery_ownership

    recovery_ownership._INCOMPLETE_RECOVERY_CLAIM_LEASE = timedelta(seconds=1)

    async def scenario():
        stores = open_application_stores(
            os.environ["CAYU_ORPHAN_TEST_DATABASE_URL"],
            sqlite_path=os.environ["CAYU_ORPHAN_TEST_STORE"],
            event_watchers=False,
        )
        app, _, _ = _app(stores)
        await stores.task_store.reclaim_expired()
        fanout = app._event_writer.fan_out_persisted

        async def die_after_commit(events, *args, **kwargs):
            if any(event.type is EventType.INTERACTION_INTERRUPTED for event in events):
                assert (await stores.session_store.load("scan")).status is SessionStatus.INTERRUPTED
                os._exit(137)
            return await fanout(events, *args, **kwargs)

        app._event_writer.fan_out_persisted = die_after_commit
        await app.reconcile_orphaned_sessions()
        raise AssertionError("Queued model settlement did not reach native commit")

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "failure",
    ["normal", "lost_ack", "restart", "receipt_transfer", "operation_conflict", "event_conflict"],
)
def test_queued_model_worker_loss_retains_terminal_identity_and_acknowledges(
    tmp_path, database_url, monkeypatch, failure
):
    _orphan(
        tmp_path / "cayu.db",
        "scan",
        database_url=database_url,
        module=__name__,
        entry="_crash_queued_model",
    )

    if failure == "restart":
        _orphan(
            tmp_path / "cayu.db",
            "scan",
            database_url=database_url,
            module=__name__,
            entry="_crash_after_queued_model_settlement",
        )

    async def scenario():
        from cayu.sessions._checkpoint_preservation import (
            _invocation_lifecycle_authority_mutation_scope,
        )
        from cayu.sessions._invocation_lifecycle import SettleInvocationCommand
        from cayu.sessions._invocation_terminal_decision import (
            InvocationTerminalOutcome,
            invocation_terminal_event_id,
        )
        from cayu.sessions._model_stage import SessionModelCompletionStageConflict
        from cayu.sessions.recovery import _incomplete_recovery_claim_from_checkpoint

        stores = open_application_stores(
            database_url, sqlite_path=tmp_path / "cayu.db", event_watchers=False
        )
        app, provider, dispatcher = _app(stores)
        try:
            checkpoint = await stores.session_store.load_checkpoint("scan")
            operation = _session_run_operation_from_checkpoint(checkpoint)
            assert operation is not None and operation.queue_task_id is not None
            assert (await stores.session_store.load("scan")).status is (
                SessionStatus.INTERRUPTED if failure == "restart" else SessionStatus.RUNNING
            )
            async with asyncio.timeout(10):
                while True:
                    claim = _incomplete_recovery_claim_from_checkpoint(
                        await stores.session_store.load_checkpoint("scan")
                    )
                    if claim is None or claim[1] <= datetime.now(UTC):
                        break
                    await asyncio.sleep(0.02)
            async with asyncio.timeout(10):
                while True:
                    task = await stores.task_store.load_task(operation.queue_task_id)
                    if task.lease_expires_at <= datetime.now(UTC):
                        break
                    await asyncio.sleep(0.02)
            envelope = _existing_queued_dispatch_envelope(task, task_type=task.type)
            assert envelope is not None
            # Dispatched work is conservatively cancellation-required on lease
            # loss. Do not manufacture task quiescence to test terminal evidence.
            await stores.task_store.reclaim_expired()
            native_apply = app._runtime_session_store.apply_invocation_lifecycle_command
            settle = app._manual_model_completion_recovery.recover_model_completion_stage
            observed = []

            async def lose_commit_ack(command):
                result = await native_apply(command)
                if (
                    isinstance(command, SettleInvocationCommand)
                    and command.transition.model_completion_stage_settlement is not None
                ):
                    raise OSError("queued model settlement committed before acknowledgement")
                return result

            async def inspect_settlement(request, **kwargs):
                result = await settle(request, **kwargs)
                observed.append(result)
                context = kwargs["invocation_context"]
                if failure == "receipt_transfer":
                    await app._terminal_finalization.repair(
                        session=result.session,
                        terminal_run_epoch=result.session.run_epoch,
                        terminal_timestamp=result.session.updated_at,
                        previous_status=SessionStatus.RUNNING,
                        claim_id=context.recovery_claim_id,
                    )
                    assert (
                        _session_run_operation_from_checkpoint(
                            await stores.session_store.load_checkpoint("scan")
                        )
                        is None
                    )
                    assert (
                        operation.operation_id
                        in _queued_dispatch_terminal_receipts_from_checkpoint(
                            await stores.session_store.load_checkpoint("scan")
                        )
                    )
                    replay = await settle(request, **kwargs)
                    assert replay.replayed
                    observed.append(replay)
                elif failure in {"operation_conflict", "event_conflict"}:
                    original_checkpoint = await stores.session_store.load_checkpoint("scan")
                    changed = dict(original_checkpoint)
                    marker = dict(changed["session_run_operation"])
                    if failure == "operation_conflict":
                        marker["operation_id"] = "a" * 64
                        marker["terminal_event_id"] = "cayu-queued-dispatch-terminal-" + "a" * 64
                    else:
                        from cayu.sessions._invocation_terminal_decision import (
                            invocation_terminal_decision_from_checkpoint,
                        )

                        owed = invocation_terminal_decision_from_checkpoint(original_checkpoint)
                        marker["terminal_event_id"] = invocation_terminal_event_id(
                            outcome=InvocationTerminalOutcome.INTERRUPTED,
                            session_id=result.session.id,
                            session_instance_id=result.session.instance_id,
                            run_epoch=result.session.run_epoch,
                            interaction_id=owed.interaction_id,
                            source_id=owed.interaction_event_id,
                            event_kind="session",
                        )
                    changed["session_run_operation"] = marker
                    with _invocation_lifecycle_authority_mutation_scope():
                        await stores.session_store.checkpoint("scan", changed)
                    before = await stores.session_store.load_events("scan")
                    try:
                        with pytest.raises(SessionModelCompletionStageConflict):
                            await settle(request, **kwargs)
                        assert await stores.session_store.load_events("scan") == before
                        assert await stores.session_store.load_checkpoint("scan") == changed
                    finally:
                        with _invocation_lifecycle_authority_mutation_scope():
                            await stores.session_store.checkpoint("scan", original_checkpoint)
                return result

            if failure == "lost_ack":
                monkeypatch.setattr(
                    app._runtime_session_store,
                    "apply_invocation_lifecycle_command",
                    lose_commit_ack,
                )
            monkeypatch.setattr(
                app._manual_model_completion_recovery,
                "recover_model_completion_stage",
                inspect_settlement,
            )
            await app.reconcile_orphaned_sessions()
            if failure != "restart":
                assert observed, "The claimed queued model owner was not reached"
            if failure == "lost_ack":
                assert observed[0].replayed
            if failure == "receipt_transfer":
                assert len(observed) == 2
            assert (await stores.session_store.load("scan")).status is SessionStatus.INTERRUPTED
            receipts = _queued_dispatch_terminal_receipts_from_checkpoint(
                await stores.session_store.load_checkpoint("scan")
            )
            assert receipts[operation.operation_id].terminal_event_id == operation.terminal_event_id
            settlement = await app._queued_dispatch_settlement_state(envelope)
            assert settlement.state is _QueuedDispatchSettlementState.TERMINAL_EVIDENCE_DURABLE
            assert settlement.terminal_status is DispatchStatus.INTERRUPTED
            replayed = [event async for event in app._dispatch_queued(envelope)]
            assert len(replayed) == 1 and replayed[0].type is EventType.SESSION_INTERRUPTED
            # The subprocess exit is positive handler-quiescence evidence. Use
            # the documented validator entrance, then the actual replacement
            # dispatcher's receipt consumer, without writing a synthetic status.
            [cancelled] = await reconcile_abandoned_tasks(
                stores.task_store,
                live_worker_ids=["replacement"],
                validator_id="tests.exited-queued-process",
                validator_version="1",
            )
            assert (
                cancelled.id == operation.queue_task_id and cancelled.status is TaskStatus.CANCELLED
            )
            assert await dispatcher.process_next(app, worker_id="replacement") is None
            events = await stores.session_store.load_events("scan")
            terminal = [event for event in events if event.type is EventType.SESSION_INTERRUPTED]
            assert len(terminal) == 1
            assert terminal[0].id == operation.terminal_event_id
            assert terminal[0].payload["session_run_operation_id"] == operation.operation_id
            assert not _queued_dispatch_terminal_receipts_from_checkpoint(
                await stores.session_store.load_checkpoint("scan")
            )
            assert provider.calls == 0
            await app._acknowledge_queued_dispatch(
                envelope, dispatch_status=DispatchStatus.INTERRUPTED
            )
            assert [event.id async for event in app._dispatch_queued(envelope)] == [
                event.id for event in replayed
            ]
            assert await app.reconcile_orphaned_sessions() == ()
            assert await stores.session_store.load_events("scan") == events
        finally:
            await app.aclose()
            await stores.close()

    asyncio.run(scenario())

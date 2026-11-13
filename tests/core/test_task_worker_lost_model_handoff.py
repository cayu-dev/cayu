"""A real worker loss during model dispatch settles through the task handoff owner."""

from __future__ import annotations

import asyncio
import os

import pytest
from tests.runtime.runtime_provider_support import VersionedFakeProvider
from tests.runtime.test_orphaned_session_reconciliation import (
    _TOOL,
    _application,
    _budget,
    _CrashingProvider,
    _orphan,
    _request,
    _slot,
)
from tests.runtime.test_orphaned_session_reconciliation import database_url as database_url

from cayu import EventType, SessionStatus, Tool
from cayu.runtime._model_completion_contracts import ModelCompletionManualRecoveryRequired
from cayu.sessions.recovery import IncompleteSessionRecoveryRequest
from cayu.storage import open_application_stores
from cayu.tasks import TaskCreate, run_task_worker_until_idle
from cayu.tasks.records import TaskStatus
from cayu.tasks.worker import _recover_expired_interrupted_task_handoffs, run_task_worker


class _NoDispatch(Tool):
    spec = _TOOL

    def __init__(self):
        self.calls = 0

    async def run(self, ctx, args):
        self.calls += 1
        raise AssertionError("Recovery must not redispatch the original external effect")


def _crash_attached_worker() -> None:
    async def scenario():
        stores = open_application_stores(
            os.environ["CAYU_ORPHAN_TEST_DATABASE_URL"],
            sqlite_path=os.environ["CAYU_ORPHAN_TEST_STORE"],
            event_watchers=False,
        )
        app = _application(
            stores.session_store,
            task_store=stores.task_store,
            environment=_slot(),
            budget_policy=_budget() if os.environ["CAYU_ORPHAN_TEST_CRASH"] == "model" else None,
            provider=_CrashingProvider if os.environ["CAYU_ORPHAN_TEST_CRASH"] == "model" else None,
        )
        await app.create_task(TaskCreate(task_id="job", type="scan"))

        async def handler(app, task, worker_id):
            request = _request("scan").model_copy(
                update={
                    "task_id": task.id,
                    "task_worker_id": worker_id,
                    "task_lease_expires_at": task.lease_expires_at,
                }
            )
            async for _ in app.run(request):
                pass

        await run_task_worker(
            app,
            stores.task_store,
            handler,
            worker_id="lost-worker",
            lease_seconds=1,
            max_tasks=1,
            reclaim=False,
        )
        raise AssertionError("The native worker did not dispatch its model")

    asyncio.run(scenario())


@pytest.mark.parametrize("crash", ["model", "tool"])
@pytest.mark.parametrize("cancel", [False, True])
def test_expired_attached_model_dispatch_settles_without_manual_recovery(
    tmp_path, database_url, cancel, crash, monkeypatch
):
    path = tmp_path / "cayu.db"
    _orphan(
        path,
        "scan",
        database_url=database_url,
        crash=crash,
        entry="_crash_attached_worker",
        module=__name__,
    )

    async def scenario():
        stores = open_application_stores(database_url, sqlite_path=path, event_watchers=False)
        provider = VersionedFakeProvider([])
        tool = _NoDispatch()
        app = _application(
            stores.session_store,
            task_store=stores.task_store,
            environment=_slot(),
            budget_policy=_budget() if crash == "model" else None,
            provider=lambda _: provider,
            tool=tool,
        )
        try:
            task = await stores.task_store.load_task("job")
            session = await stores.session_store.load("scan")
            assert task.session_id == session.id and task.session_instance_id == session.instance_id
            assert session.status is SessionStatus.RUNNING
            stage = await stores.session_store.load_active_model_completion_stage("scan")
            assert (stage is not None) is (crash == "model")
            # Use real store-clock expiry, never a manually constructed lost-owner state.
            async with asyncio.timeout(10):
                while (
                    await stores.task_store.load_expired_interrupted_task_handoff_candidate("job")
                    is None
                ):
                    await asyncio.sleep(0.02)
            if cancel:
                await stores.task_store.cancel_task("job", {"code": "operator_cancelled"})
            request = IncompleteSessionRecoveryRequest(
                session_id="scan",
                inactive_for_seconds=None,
                reason="expired_attached_task_owner",
                metadata={"recovery_scope": "attached_task"},
            )
            # Caller-supplied identical values never enable internal unknown-model settlement.
            if crash == "model":
                with pytest.raises(ModelCompletionManualRecoveryRequired):
                    await app.recover_incomplete_session(request)
                # Public recovery may acquire/release its own epoch, but cannot
                # settle the unknown model or release its reserved budget.
                assert (await stores.session_store.load("scan")).status is SessionStatus.RUNNING
                assert (
                    await stores.session_store.load_model_completion_stage_settlement(
                        "scan",
                        stage.stage.stage_id,
                    )
                    is None
                )
                assert (
                    await stores.session_store.load_active_model_completion_stage("scan")
                    is not None
                )
            # Lose the task-store response after its durable commit. The existing
            # handoff receipt owner must read it back without another model call.
            recover = stores.task_store.recover_interrupted_task_worker
            committed_requests = []

            async def commit_without_ack(command):
                await recover(command)
                committed_requests.append(command)
                raise OSError("handoff committed but response lost")

            if not cancel:
                monkeypatch.setattr(
                    stores.task_store, "recover_interrupted_task_worker", commit_without_ack
                )
            if cancel:

                async def no_handler(*_args):
                    raise AssertionError("A cancelled task must not redispatch its handler")

                result = await asyncio.wait_for(
                    run_task_worker_until_idle(
                        app,
                        stores.task_store,
                        no_handler,
                        worker_id="replacement",
                        reclaim=False,
                    ),
                    15,
                )
                assert result.idle and result.handled == 0 and not result.attention_required
            else:
                page = await _recover_expired_interrupted_task_handoffs(
                    app,
                    stores.task_store,
                    worker_id="replacement",
                    after=None,
                    limit=10,
                    stop=None,
                )
                assert page.recovered == 1 and not page.failed and not page.manual_recovery_required
            current = await stores.session_store.load("scan")
            task = await stores.task_store.load_task("job")
            assert current.status is SessionStatus.INTERRUPTED
            assert (
                current.instance_id == session.instance_id and current.run_epoch > session.run_epoch
            )
            assert task.status is (TaskStatus.CANCELLED if cancel else TaskStatus.RUNNING)
            assert task.worker_id is None and task.lease_expires_at is None
            assert cancel or task.interrupted_handoff_id is not None
            if crash == "model":
                settlement = await stores.session_store.load_model_completion_stage_settlement(
                    "scan",
                    stage.stage.stage_id,
                )
                assert settlement is not None
                assert settlement.disposition.value == "provider_effect_outcome_unknown"
                assert stage.stage.reservation_ids
                for reservation_id in stage.stage.reservation_ids:
                    reservation = await stores.budget_ledger.load_reservation(reservation_id)
                    assert reservation.status == "reconciled"
            events = await stores.session_store.load_events("scan")
            if crash == "tool":
                assert any(event.type is EventType.TOOL_EFFECT_OUTCOME_UNKNOWN for event in events)
            if not cancel:
                assert len(committed_requests) == 1
                receipt = await stores.task_store.load_interrupted_task_handoff_receipt(
                    task.id,
                    task.interrupted_handoff_id,
                )
                assert receipt is not None
                assert await recover(committed_requests[0]) == receipt
            assert sum(event.type is EventType.SESSION_INTERRUPTED for event in events) == 1
            assert provider.requests == [] and tool.calls == 0
            replay = await _recover_expired_interrupted_task_handoffs(
                app,
                stores.task_store,
                worker_id="replacement",
                after=None,
                limit=10,
                stop=None,
            )
            assert replay.recovered == 0 and not replay.failed
            assert await stores.session_store.load_events("scan") == events
            assert await stores.task_store.load_task("job") == task
        finally:
            await app.aclose()
            await stores.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("crash", ["model", "tool"])
@pytest.mark.parametrize("cancel", [False, True])
def test_expired_attached_dispatch_without_environment_is_not_left_running(
    tmp_path,
    database_url,
    crash,
    cancel,
):
    path = tmp_path / "cayu.db"
    _orphan(
        path,
        "scan",
        database_url=database_url,
        crash=crash,
        entry="_crash_attached_worker",
        module=__name__,
    )

    async def scenario():
        stores = open_application_stores(database_url, sqlite_path=path, event_watchers=False)
        provider, tool = VersionedFakeProvider([]), _NoDispatch()
        app = _application(
            stores.session_store,
            task_store=stores.task_store,
            provider=lambda _: provider,
            tool=tool,
            budget_policy=_budget() if crash == "model" else None,
        )
        try:
            async with asyncio.timeout(10):
                while (
                    await stores.task_store.load_expired_interrupted_task_handoff_candidate("job")
                    is None
                ):
                    await asyncio.sleep(0.02)
            original = await stores.session_store.load("scan")
            stage = await stores.session_store.load_active_model_completion_stage("scan")
            with pytest.raises(KeyError):
                await app.recover_incomplete_session(
                    IncompleteSessionRecoveryRequest(
                        session_id="scan",
                        inactive_for_seconds=None,
                        reason="expired_attached_task_owner",
                        metadata={"recovery_scope": "attached_task"},
                    )
                )
            assert (await stores.session_store.load("scan")).status is SessionStatus.RUNNING
            if cancel:
                await stores.task_store.cancel_task("job", {"code": "operator_cancelled"})
            await _recover_expired_interrupted_task_handoffs(
                app,
                stores.task_store,
                worker_id="replacement",
                after=None,
                limit=10,
                stop=None,
            )
            current = await stores.session_store.load("scan")
            task = await stores.task_store.load_task("job")
            assert current.status is SessionStatus.INTERRUPTED
            assert current.environment_name == original.environment_name
            assert current.instance_id == original.instance_id
            assert task.status is (TaskStatus.CANCELLED if cancel else TaskStatus.RUNNING)
            assert task.worker_id is None and task.lease_expires_at is None
            if not cancel:
                assert task.interrupted_handoff_id is not None
                assert (
                    await stores.task_store.load_interrupted_task_handoff_receipt(
                        task.id, task.interrupted_handoff_id
                    )
                    is not None
                )
            events = await stores.session_store.load_events("scan")
            assert sum(event.type is EventType.SESSION_INTERRUPTED for event in events) == 1
            if crash == "model":
                settlement = await stores.session_store.load_model_completion_stage_settlement(
                    "scan", stage.stage.stage_id
                )
                assert settlement.disposition.value == "provider_effect_outcome_unknown"
                for reservation_id in stage.stage.reservation_ids:
                    assert (
                        await stores.budget_ledger.load_reservation(reservation_id)
                    ).status == "reconciled"
            else:
                assert any(event.type is EventType.TOOL_EFFECT_OUTCOME_UNKNOWN for event in events)
            replay = await _recover_expired_interrupted_task_handoffs(
                app, stores.task_store, worker_id="replacement", after=None, limit=10, stop=None
            )
            assert replay.recovered == 0 and not replay.failed
            assert await stores.session_store.load_events("scan") == events
            assert await stores.task_store.load_task("job") == task
            assert provider.requests == [] and tool.calls == 0
        finally:
            await app.aclose()
            await stores.close()

    asyncio.run(scenario())

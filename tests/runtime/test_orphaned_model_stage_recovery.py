"""Replacement recovery ends lost model work with its environment still registered."""

from __future__ import annotations

import asyncio
import os

import pytest
from tests.runtime.test_orphaned_model_work_reconciliation import (
    _application,
    _budget_settled_once,
    _unfinished,
    _worker_died_in,
)
from tests.runtime.test_orphaned_session_reconciliation import (
    _orphan,
    _slot,
    _stores,
)
from tests.runtime.test_orphaned_session_reconciliation import database_url as database_url

from cayu import EventType, Message, RunRequest, SessionStatus
from cayu.storage import open_application_stores
from cayu.tasks import TaskCreate, run_task_worker, run_task_worker_until_idle
from cayu.tasks.records import TaskStatus


@pytest.mark.parametrize("mode", ["compaction", "auxiliary"])
def test_lost_nested_model_work_is_settled_with_registered_environment(
    tmp_path, database_url, mode
):
    _worker_died_in(tmp_path, database_url, mode)

    async def scenario():
        stores = _stores(database_url, tmp_path / "cayu.db")
        app = _application(stores.session_store, mode, environment=_slot())
        try:
            stage = await _unfinished(stores.session_store, mode)
            await app.reconcile_orphaned_sessions()
            session = await stores.session_store.load("scan")
            assert session.status is SessionStatus.INTERRUPTED
            assert await stores.session_store.load_active_model_completion_stage("scan") is None
            await _budget_settled_once(stores, stage, conservative=True)
            events = await stores.session_store.load_events("scan")
            assert [event.type for event in events].count(EventType.SESSION_INTERRUPTED) == 1
            assert await app.reconcile_orphaned_sessions() == ()
            assert await stores.session_store.load_events("scan") == events
        finally:
            await app.aclose()
            await stores.close()

    asyncio.run(scenario())


def _crash_attached_nested_model() -> None:
    async def scenario():
        stores = open_application_stores(
            os.environ["CAYU_ORPHAN_TEST_DATABASE_URL"],
            sqlite_path=os.environ["CAYU_ORPHAN_TEST_STORE"],
            event_watchers=False,
        )
        mode = os.environ["CAYU_ORPHAN_TEST_CRASH"]
        if mode in {"empty", "prepared", "answered_tools"}:

            async def die_before_publication(*_args, **_kwargs):
                os._exit(137)

            boundary = {
                "empty": "prepare_model_completion_stage",
                "prepared": "mark_model_completion_stage_dispatched",
                "answered_tools": "promote_model_completion_stage",
            }[mode]
            setattr(type(stores.session_store), boundary, die_before_publication)
        app = _application(
            stores.session_store,
            mode,
            environment=_slot(),
            task_store=stores.task_store,
            crash=True,
        )
        await app.create_task(TaskCreate(task_id="job", type="scan"))

        async def handler(app, task, worker_id):
            messages = (
                [
                    Message.text("user", "old"),
                    Message.text("assistant", "old answer"),
                    Message.text("user", "go"),
                ]
                if mode == "compaction"
                else [Message.text("user", "go")]
            )
            async for _ in app.run(
                RunRequest(
                    session_id="scan",
                    agent_name="scanner",
                    environment_name=_slot().environment.spec.name,
                    messages=messages,
                    task_id=task.id,
                    task_worker_id=worker_id,
                    task_lease_expires_at=task.lease_expires_at,
                )
            ):
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
        raise AssertionError("The worker did not die during nested model dispatch")

    asyncio.run(scenario())


@pytest.mark.parametrize("mode", ["compaction", "auxiliary", "empty", "prepared", "answered_tools"])
def test_expired_attached_nested_model_without_environment_is_interrupted(
    tmp_path, database_url, mode
):
    path = tmp_path / "cayu.db"
    _orphan(
        path,
        "scan",
        database_url=database_url,
        crash=mode,
        module=__name__,
        entry="_crash_attached_nested_model",
    )

    async def scenario():
        stores = open_application_stores(database_url, sqlite_path=path, event_watchers=False)
        app = _application(stores.session_store, mode, task_store=stores.task_store)
        try:
            if mode == "prepared":
                active = await stores.session_store.load_active_model_completion_stage("scan")
                stage = active.stage
                assert stage.state == "in_flight"
                assert (
                    await stores.session_store.load_model_completion_stage_dispatch(
                        "scan", stage.stage_id
                    )
                    is None
                )
            else:
                stage = None if mode == "empty" else await _unfinished(stores.session_store, mode)
            if stage is None:
                assert await stores.session_store.load_active_model_completion_stage("scan") is None
                original_reservations = await stores.budget_ledger._scan_reservation_records(
                    session_id="scan"
                )
                assert original_reservations
                assert all(
                    reservation.status == "active" and reservation.dispatch_id is None
                    for reservation in original_reservations
                )
            original = await stores.session_store.load("scan")
            task = await stores.task_store.load_task("job")
            assert task is not None
            assert task.worker_id == "lost-worker"
            # The short worker lease can expire while the child is still
            # entering its crash point, leaving durable cancellation intent.
            # Request against that exact claim; replay must preserve the fence.
            cancellation = await stores.task_store.request_claimed_task_cancellation(
                "job", task.worker_id, claim_generation=task.claim_generation
            )
            assert (
                await stores.task_store.request_claimed_task_cancellation(
                    "job", task.worker_id, claim_generation=task.claim_generation
                )
                == cancellation
            )
            async with asyncio.timeout(10):
                while not await stores.task_store.list_expired_interrupted_task_handoff_candidates(
                    limit=10
                ):
                    await asyncio.sleep(0.02)

            async def no_handler(*_args):
                raise AssertionError("Recovery must not dispatch another handler")

            result = await asyncio.wait_for(
                run_task_worker_until_idle(
                    app, stores.task_store, no_handler, worker_id="replacement", reclaim=False
                ),
                15,
            )
            session = await stores.session_store.load("scan")
            assert session.status is SessionStatus.INTERRUPTED
            assert session.environment_name == original.environment_name
            assert result.handled == 0
            assert (await stores.task_store.load_task("job")).status is TaskStatus.CANCELLED
            assert await stores.session_store.load_active_model_completion_stage("scan") is None
            if mode == "prepared":
                for reservation_id in stage.reservation_ids:
                    reservation = await stores.budget_ledger.load_reservation(reservation_id)
                    assert reservation.status == "released"
            elif stage is not None:
                await _budget_settled_once(stores, stage, conservative=mode != "answered_tools")
            else:
                reservations = await stores.budget_ledger._scan_reservation_records(
                    session_id="scan"
                )
                # Before a stage exists, the ledger's renewable pre-dispatch
                # lease owns reclamation. Session repair must not free capacity
                # without that owner's authority or drop its durable hold.
                assert reservations == original_reservations
            events = await stores.session_store.load_events("scan")
            assert [event.type for event in events].count(EventType.SESSION_INTERRUPTED) == 1
            assert await app.reconcile_orphaned_sessions() == ()
            assert await stores.session_store.load_events("scan") == events
        finally:
            await app.aclose()
            await stores.close()

    asyncio.run(scenario())

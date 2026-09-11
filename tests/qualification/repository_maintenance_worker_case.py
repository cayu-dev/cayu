"""Managed-task completion boundary; not a restart or deployment qualification."""

import asyncio
import importlib

from cayu import (
    CodingProductArtifactRepository,
    CodingProductRunner,
    EventQuery,
    EventType,
    SessionStatus,
    TaskQuery,
    TaskStatus,
    run_task_worker,
)


async def exercise_coding_worker(
    application, task, queued, monkeypatch, *, deadline, reservations, rejected=False
):
    """Keep the outer claim until the sealed product independently verifies."""
    store = application.app.task_store
    assert queued.id == task.task_id and queued.type == "maintenance.coding"
    assert queued.status is TaskStatus.PENDING and queued.session_id is None
    entered, release = asyncio.Event(), asyncio.Event()
    original = CodingProductRunner._compile_and_publish

    async def paused_publication(runner, *args, **kwargs):
        entered.set()
        await release.wait()
        return await original(runner, *args, **kwargs)

    monkeypatch.setattr(CodingProductRunner, "_compile_and_publish", paused_publication)

    async def handler(_app, claimed, worker_id):
        assert claimed.id == task.task_id
        assert claimed.invocation == queued.invocation
        operation = importlib.import_module("operations.maintenance_worker")
        await operation.handle_coding_task(application, reservations, claimed, worker_id)

    query = TaskQuery(type="maintenance.coding")
    worker = asyncio.create_task(
        run_task_worker(
            application.app,
            store,
            handler,
            worker_id="coding-owner",
            query=query,
            # Allow the full coding fixture a realistic lease; test renewal
            # across a complete lease lifetime at the controlled barrier below.
            lease_seconds=10,
            poll_interval_s=0.001,
            reclaim=False,
            max_tasks=1,
        )
    )
    try:
        await asyncio.wait_for(entered.wait(), timeout=60)
        session = await application.app.session_store.load(task.session_id)
        assert session is not None and session.status is SessionStatus.COMPLETED
        assert session.parent_session_id == task.parent_session_id
        root = await application.app.session_store.load(task.parent_session_id)
        # A workflow root is a terminal journal anchor, not an active agent
        # session. Its workflow events, not Session.status, describe progress.
        assert root is not None and root.status is SessionStatus.COMPLETED
        assert (
            root.execution_deadline.model_dump()
            == session.execution_deadline.model_dump()
            == deadline.model_dump()
        )
        assert (
            len(
                await application.app.session_store.query_events(
                    EventQuery(session_id=root.id, event_type=EventType.WORKFLOW_STARTED)
                )
            )
            == 1
        )
        assert (
            await application.app.session_store.query_events(
                EventQuery(session_id=root.id, event_type=EventType.WORKFLOW_COMPLETED)
            )
            == []
        )
        before = await store.load_task(task.task_id)
        assert before is not None and before.status is TaskStatus.CLAIMED
        assert before.session_id is None
        assert before.invocation == queued.invocation
        assert before.worker_id == "coding-owner"
        assert not before.result and not worker.done()
        # Cross the original claim lifetime while publication is blocked. The
        # existing worker heartbeat, not an application lease, retains ownership.
        await asyncio.sleep(10.05)
        current = await store.load_task(task.task_id)
        assert current is not None and current.status is TaskStatus.CLAIMED
        assert current.lease_expires_at > before.lease_expires_at
        assert await store.reclaim_expired(query=query) == []
        assert await store.claim_task("replacement", query, lease_seconds=1) is None
        release.set()
        assert await asyncio.wait_for(worker, timeout=30) == 1
        terminal = await store.load_task(task.task_id)
        expected_status = TaskStatus.FAILED if rejected else TaskStatus.COMPLETED
        assert terminal is not None and terminal.status is expected_status
        assert terminal.worker_id is None and terminal.lease_expires_at is None
        assert terminal.invocation == queued.invocation
        root = await application.app.session_store.load(task.parent_session_id)
        assert root is not None and root.status is SessionStatus.COMPLETED
        completed = await application.app.session_store.query_events(
            EventQuery(session_id=root.id, event_type=EventType.WORKFLOW_COMPLETED)
        )
        assert len(completed) == 1
        assert completed[0].event.payload["verdict"] == ("rejected" if rejected else "verified")
        repository = CodingProductArtifactRepository(application.artifact_store)
        admitted = await repository.load_request(task.product_run_id, session_id=task.session_id)
        publication = await repository.load_publication(
            request_fingerprint=admitted.fingerprint,
            digest=completed[0].event.payload["result_digest"],
        )
        if rejected:
            assert not terminal.result
            assert terminal.error["error"] == "MaintenanceAcceptanceRejected"
        else:
            assert terminal.result == {
                "product_run_id": task.product_run_id,
                "result_digest": publication.result_reference.digest,
            }
            # Downstream delivery reconstructs from the durable reservation and
            # completed task, not this fixture's in-process publication variable.
            identities = importlib.import_module("operations.maintenance_runs")
            reopened = identities.SQLiteMaintenanceRunStore(reservations.path)
            identity = await reopened.load_for_task(task.task_id)
            results = importlib.import_module("operations.maintenance_results")
            observed_task, observed_publication = await results.load_verified_coding_result(
                application, reopened, identity
            )
            assert observed_task == task
            assert observed_publication == publication
            return observed_publication
        return publication
    finally:
        release.set()
        if not worker.done():
            worker.cancel()
        await asyncio.gather(worker, return_exceptions=True)

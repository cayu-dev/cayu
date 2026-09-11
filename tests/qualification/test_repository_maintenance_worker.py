"""Real Runtime worker failure settlement before coding dispatch."""

import asyncio
import importlib

import pytest

from cayu import ExecutionDeadline, TaskQuery, TaskStatus, run_task_worker
from tests.cli.test_scaffold_coding_budget import denial_policy
from tests.qualification.test_repository_maintenance_application import project as project
from tests.qualification.test_repository_maintenance_request import consumer as consumer


@pytest.mark.parametrize("failure", ["expired", "configuration_changed", "missing_reservation"])
@pytest.mark.parametrize("consumer", ["memory", "sqlite"], indirect=True)
def test_worker_settles_rejected_request_without_dispatch(consumer, tmp_path, failure):
    application, task, provider, domain, requests, _workflow = consumer

    async def scenario():
        identities = importlib.import_module("domain.maintenance_identity")
        stores = importlib.import_module("operations.maintenance_runs")
        intake = importlib.import_module("operations.maintenance_intake")
        worker = importlib.import_module("operations.maintenance_worker")
        registry = stores.SQLiteMaintenanceRunStore(tmp_path / "worker-reservations.sqlite")
        await registry.initialize()
        accepted = await requests.capture_accepted_request(application, task)
        deadline = ExecutionDeadline.after(0 if failure == "expired" else 180)
        assert deadline.expires_at is not None
        identity = await registry.reserve(
            identities.MaintenanceRunIntent(
                tenant="fixture",
                subject="operator",
                idempotency_key="worker-failure",
                request_json=domain.encode_request(accepted),
            ),
            coding_expires_at=deadline.expires_at.isoformat(),
        )
        await intake.ensure_coding_task(application.app, registry, identity)
        if failure == "configuration_changed":
            application.app.budget_policy = denial_policy()
        if failure == "missing_reservation":
            # A different, initialized registry cannot authorize the queued task.
            registry = stores.SQLiteMaintenanceRunStore(tmp_path / "empty.sqlite")
            await registry.initialize()

        async def handler(_app, claimed, worker_id):
            await worker.handle_coding_task(application, registry, claimed, worker_id)

        assert (
            await run_task_worker(
                application.app,
                application.app.task_store,
                handler,
                worker_id="maintenance-worker",
                query=TaskQuery(type="maintenance.coding"),
                max_tasks=1,
                reclaim=False,
                lease_seconds=10,
            )
            == 1
        )
        terminal = await application.app.task_store.load_task(identity.task_id)
        assert terminal is not None and terminal.status is TaskStatus.FAILED
        assert terminal.worker_id is None and terminal.lease_expires_at is None
        assert terminal.error and not terminal.result
        assert (
            terminal.error["error"]
            == {
                "expired": "ExecutionDeadlineExceeded",
                "configuration_changed": "ValueError",
                "missing_reservation": "MaintenanceTaskConflict",
            }[failure]
        )
        assert not provider.requests
        assert await application.app.session_store.load(identity.session_id) is None
        listed = await application.artifact_store.list(session_id=identity.session_id)
        assert not listed.artifacts and listed.total_count == 0
        if failure != "configuration_changed":
            assert await application.app.session_store.load(identity.workflow_session_id) is None
        assert await registry.load_for_task(identity.task_id) == (
            None if failure == "missing_reservation" else identity
        )

    asyncio.run(scenario())

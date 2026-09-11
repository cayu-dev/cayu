"""Saved consent through Runtime worker/native connector; upstream evidence controlled."""

import asyncio
import importlib
import json
from dataclasses import replace
from datetime import UTC, datetime

import pytest

from cayu import (
    GitHubCheckBundle,
    GitHubCheckObservation,
    GitHubDeliveryApproval,
    TaskQuery,
    TaskStatus,
    github_connector_behavior_fingerprint,
    run_task_worker,
)
from tests.core.test_github_delivery import FakeTransport, _connector, _pr
from tests.qualification.test_repository_maintenance_github_intake import configuration, enqueue
from tests.qualification.test_repository_maintenance_github_intake import (
    github_intake as github_intake,
)
from tests.qualification.test_repository_maintenance_intake import intake as intake


@pytest.mark.parametrize("cancel", [False, True])
def test_saved_approval_execution_retains_task_until_connector_settles(
    github_intake, monkeypatch, tmp_path, cancel
):
    intake_module, application, reservations, identity = github_intake
    worker_module = importlib.import_module("operations.maintenance_github")
    config = configuration()
    config["requested_at"] = datetime.now(UTC).isoformat()
    config["limits"] = {"poll_interval_seconds": 1, "timeout_seconds": 1, "max_elapsed_seconds": 30}
    config["security"].update(
        {
            "connector_id": "github-connector",
            "connector_behavior_fingerprint": github_connector_behavior_fingerprint(),
            "credential_profile_id": "github-token",
            "egress_profile_id": "github-only",
        }
    )
    monkeypatch.setenv("CAYU_MAINTENANCE_GITHUB_JSON", json.dumps(config))
    monkeypatch.setattr(
        worker_module, "load_verified_git_result", intake_module.load_verified_git_result
    )

    async def verify(*args):
        return None

    application.verify = verify

    async def scenario():
        queued = await enqueue(github_intake)
        native = await intake_module.load_github_approval_request(
            application, reservations, identity
        )
        saved = GitHubDeliveryApproval.model_validate_json(queued.input["approval_json"])
        transport = FakeTransport(native)
        transport.check_bundles.append(
            GitHubCheckBundle(
                head_commit=native.repository.head_commit,
                checks=(
                    GitHubCheckObservation(
                        provider_id="test",
                        name="test",
                        head_commit=native.repository.head_commit,
                        status="completed",
                        conclusion="success",
                    ),
                ),
            )
        )
        connector, _config = _connector(tmp_path, native, transport)
        connector.profile = replace(
            connector.profile,
            repositories={
                "github": replace(_config, repository_id=native.repository.repository_id)
            },
        )
        connector.clock = lambda: datetime.now(UTC)
        entered, release = asyncio.Event(), asyncio.Event()
        factories = 0

        def factory():
            nonlocal factories
            factories += 1
            return connector

        async def create(config, request):
            transport.create_calls += 1
            entered.set()
            await release.wait()
            transport.pull_request = _pr(request)
            return transport.pull_request, "worker-create"

        monkeypatch.setattr(transport, "create_pull_request", create)

        async def handle(_app, task, worker_id):
            await worker_module.handle_github_delivery_task(
                application, reservations, task, worker_id, factory
            )

        store = application.app.task_store
        query = TaskQuery(type="maintenance.github_delivery")
        owner = asyncio.create_task(
            run_task_worker(
                application.app,
                store,
                handle,
                worker_id="worker",
                query=query,
                lease_seconds=1,
                reclaim=False,
                max_tasks=1,
                recover_interrupted_handoffs=False,
            )
        )
        try:
            await asyncio.wait_for(entered.wait(), 5)
            if cancel:
                owner.cancel("stop-github-worker")
                for _ in range(100):
                    if (await store.load_task(queued.id)).status_reason == "cancellation_requested":
                        break
                    await asyncio.sleep(0.01)
                else:
                    pytest.fail("Missing cancellation fence")
                await asyncio.sleep(1.05)
                assert not owner.done()
                assert await store.reclaim_expired(query=query) == []
                assert await store.claim_task("replacement", query, lease_seconds=1) is None
            release.set()
            done, _ = await asyncio.wait((owner,), timeout=5)
            assert owner in done
            if cancel:
                with pytest.raises(asyncio.CancelledError, match="stop-github-worker"):
                    await owner
                assert owner.cancelled() and owner.cancelling() == 1
            else:
                assert await owner == 1
            final = await store.load_task(queued.id)
            assert final.status is (TaskStatus.CANCELLED if cancel else TaskStatus.COMPLETED)
            assert final.worker_id is None and final.lease_expires_at is None
            assert factories == 1 and transport.create_calls == 1
            assert await connector.aclose(timeout_s=0) is True
            if cancel:
                assert not final.result
            else:
                publication = await connector.repository.latest(native)
                assert publication.result.approval_fingerprint == saved.fingerprint
                assert final.result == {
                    "request_fingerprint": native.fingerprint,
                    "result_digest": publication.artifact.sha256,
                }
        finally:
            release.set()
            if not owner.done():
                owner.cancel()
            await asyncio.gather(owner, return_exceptions=True)
            assert await connector.aclose(timeout_s=5) is True

    asyncio.run(scenario())


def test_changed_configuration_does_not_construct_connector(github_intake, monkeypatch):
    intake_module, application, reservations, _identity = github_intake
    worker_module = importlib.import_module("operations.maintenance_github")
    monkeypatch.setattr(
        worker_module, "load_verified_git_result", intake_module.load_verified_git_result
    )
    conflict = importlib.import_module("operations.maintenance_intake").MaintenanceTaskConflict

    def forbidden():
        pytest.fail("Conflicting request constructed connector")

    async def scenario():
        await enqueue(github_intake)
        claimed = await application.app.task_store.claim_task(
            "worker", TaskQuery(type="maintenance.github_delivery")
        )
        config = configuration()
        config["metadata"]["title"] = "Changed after consent"
        monkeypatch.setenv("CAYU_MAINTENANCE_GITHUB_JSON", json.dumps(config))
        with pytest.raises(conflict):
            await worker_module.handle_github_delivery_task(
                application, reservations, claimed, "worker", forbidden
            )
        assert await application.app.task_store.load_task(claimed.id) == claimed

    asyncio.run(scenario())

"""Approved Runtime task execution; native effects are controlled in these cases."""

import asyncio
import importlib
import json
from types import SimpleNamespace

import pytest

from cayu import RemoteGitDeliveryApproval, TaskQuery, TaskStatus, run_task_worker
from tests.qualification.test_repository_maintenance_git_approval import (
    approval_context as approval_context,
)
from tests.qualification.test_repository_maintenance_git_configuration import arguments
from tests.qualification.test_repository_maintenance_git_intake import git_intake as git_intake
from tests.qualification.test_repository_maintenance_intake import intake as intake


def test_delivery_claim_restores_saved_approval(approval_context):
    module, application, reservations, identity, native, _pending, _receipts, options = (
        approval_context
    )
    conflict = importlib.import_module("operations.maintenance_intake").MaintenanceTaskConflict

    async def scenario():
        queued = await module.ensure_git_delivery_task(
            application, reservations, identity, **options
        )
        store = application.app.task_store
        with pytest.raises(conflict):
            await module.load_claimed_git_delivery(application.app, reservations, queued, "worker")
        claimed = await store.claim_task("worker", TaskQuery(type="maintenance.git_delivery"))
        restored, request, approval = await module.load_claimed_git_delivery(
            application.app, reservations, claimed, "worker"
        )
        assert restored == identity and request == native
        assert approval == RemoteGitDeliveryApproval.model_validate_json(
            queued.input["approval_json"]
        )
        with pytest.raises(conflict):
            await module.load_claimed_git_delivery(application.app, reservations, claimed, "other")
        # Identical claim identity cannot authorize changed decision-bearing fields.
        for field, value in {
            "approval_id": "other",
            "request_fingerprint": "sha256:" + "0" * 64,
            "prepared_tree": "0" * 40,
            "policy_fingerprint": "sha256:" + "0" * 64,
            "commit_approved": False,
            "push_approved": False,
        }.items():
            altered = claimed.model_copy(deep=True)
            raw = json.loads(altered.input["approval_json"])
            raw[field] = value
            altered.input["approval_json"] = json.dumps(raw, sort_keys=True, separators=(",", ":"))
            with pytest.raises(conflict):
                await module.load_claimed_git_delivery(
                    application.app, reservations, altered, "worker"
                )
        assert await store.load_task(claimed.id) == claimed

    asyncio.run(scenario())


@pytest.mark.parametrize("bad", ["duplicate", "oversize", "boolean", "policy", "denied"])
def test_corrupt_stored_approval_is_not_authority(approval_context, monkeypatch, bad):
    module, application, reservations, identity, _native, _pending, _receipts, options = (
        approval_context
    )
    conflict = importlib.import_module("operations.maintenance_intake").MaintenanceTaskConflict

    async def scenario():
        await module.ensure_git_delivery_task(application, reservations, identity, **options)
        claimed = await application.app.task_store.claim_task(
            "worker", TaskQuery(type="maintenance.git_delivery")
        )
        corrupt = claimed.model_copy(deep=True)
        if bad == "duplicate":
            encoded = '{"approval_id":"a","approval_id":"b"}'
        elif bad == "oversize":
            encoded = "x" * 65537
        elif bad == "boolean":
            encoded = True
        else:
            raw = json.loads(corrupt.input["approval_json"])
            if bad == "policy":
                raw["policy_fingerprint"] = "sha256:" + "0" * 64
            else:
                raw["push_approved"] = False
            encoded = json.dumps(raw, sort_keys=True, separators=(",", ":"))
        corrupt.input["approval_json"] = encoded

        async def stored(_task_id):
            return corrupt

        monkeypatch.setattr(application.app.task_store, "load_task", stored)
        with pytest.raises(conflict):
            await module.load_claimed_git_delivery(application.app, reservations, corrupt, "worker")

    asyncio.run(scenario())


@pytest.mark.parametrize("cancel", [False, True])
def test_delivery_worker_passes_saved_approval_and_retains_owner(
    approval_context, monkeypatch, tmp_path, cancel
):
    module, application, reservations, identity, native, _pending, _receipts, options = (
        approval_context
    )
    worker_module = importlib.import_module("operations.maintenance_git_worker")
    integration = importlib.import_module("integrations.remote_git")
    broker_options = arguments(tmp_path)
    broker_options["remote_url"] = str(tmp_path / "remote")
    broker = integration.build_remote_git_delivery_broker(**broker_options)
    application.artifact_store = broker_options["artifact_store"]
    monkeypatch.setattr(
        worker_module, "load_verified_coding_result", module.load_verified_coding_result
    )
    monkeypatch.setattr(worker_module, "_configured_git_request", lambda *args: native)

    async def scenario():
        queued = await module.ensure_git_delivery_task(
            application, reservations, identity, **options
        )
        saved = RemoteGitDeliveryApproval.model_validate_json(queued.input["approval_json"])
        entered, release = asyncio.Event(), asyncio.Event()
        calls = []

        async def dispatched(_application, _task, _product, given_broker, request, *, approval):
            assert given_broker is broker and request == native
            assert approval == saved
            calls.append(approval)
            entered.set()
            await release.wait()
            return SimpleNamespace(artifact=SimpleNamespace(sha256="sha256:" + "a" * 64))

        monkeypatch.setattr(worker_module, "run_verified_git_delivery", dispatched)

        async def handle(_app, claimed, worker_id):
            await worker_module.handle_git_delivery_task(
                application, reservations, claimed, worker_id, broker
            )

        store = application.app.task_store
        query = TaskQuery(type="maintenance.git_delivery")
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
                owner.cancel("stop-approved-delivery")
                for _ in range(100):
                    if (await store.load_task(queued.id)).status_reason == "cancellation_requested":
                        break
                    await asyncio.sleep(0.01)
                else:
                    pytest.fail("Worker did not retain cancellation fence")
                await asyncio.sleep(1.05)
                assert not owner.done()
                assert await store.reclaim_expired(query=query) == []
                assert await store.claim_task("replacement", query, lease_seconds=1) is None
            release.set()
            done, _ = await asyncio.wait((owner,), timeout=5)
            assert owner in done
            if cancel:
                with pytest.raises(asyncio.CancelledError, match="stop-approved-delivery"):
                    await owner
                assert owner.cancelled() and owner.cancelling() == 1
            else:
                assert await owner == 1
            final = await store.load_task(queued.id)
            assert final.status is (TaskStatus.CANCELLED if cancel else TaskStatus.COMPLETED)
            assert final.worker_id is None and final.lease_expires_at is None
            assert len(calls) == 1
            if cancel:
                assert not final.result
            else:
                assert final.result == {
                    "request_fingerprint": native.fingerprint,
                    "result_digest": "sha256:" + "a" * 64,
                }
        finally:
            release.set()
            if not owner.done():
                owner.cancel()
            await asyncio.gather(owner, return_exceptions=True)

    asyncio.run(scenario())

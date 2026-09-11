"""Task/native evidence joins with real task stores and controlled artifact reads."""

import asyncio
import importlib

import pytest

from cayu import RemoteGitDeliveryApproval, RemoteGitDeliveryState, TaskQuery
from tests.qualification.test_repository_maintenance_git_approval import (
    approval_context as approval_context,
)
from tests.qualification.test_repository_maintenance_git_intake import git_intake as git_intake
from tests.qualification.test_repository_maintenance_intake import intake as intake


@pytest.fixture
def pushed(approval_context):
    module, application, reservations, identity, native, publication, receipts, options = (
        approval_context
    )

    async def setup():
        queued = await module.ensure_git_delivery_task(
            application, reservations, identity, **options
        )
        approval = RemoteGitDeliveryApproval.model_validate_json(queued.input["approval_json"])
        claimed = await application.app.task_store.claim_task(
            "worker", TaskQuery(type="maintenance.git_delivery")
        )
        completed = await application.app.task_store.complete_task(
            claimed.id,
            {
                "request_fingerprint": native.fingerprint,
                "result_digest": publication.artifact.sha256,
            },
            worker_id="worker",
            lease_expires_at=claimed.lease_expires_at,
        )
        publication.result.state = RemoteGitDeliveryState.PUSHED
        publication.result.cleanup_settled = True
        publication.result.approval_id = approval.approval_id
        publication.result.approval_fingerprint = approval.fingerprint
        receipts[-1].state = RemoteGitDeliveryState.PUSHED
        return completed

    completed = asyncio.run(setup())
    return module, application, reservations, identity, publication, receipts, completed


def test_exact_completed_push_is_read_without_task_mutation(pushed):
    module, application, reservations, identity, publication, _receipts, completed = pushed

    async def scenario():
        _task, _product, restored = await module.load_verified_git_result(
            application, reservations, identity
        )
        assert restored is publication
        assert await application.app.task_store.load_task(completed.id) == completed

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "fault",
    [
        "state",
        "cleanup",
        "approval_id",
        "approval_fingerprint",
        "tree",
        "receipt_state",
        "receipt_digest",
        "receipt_absent",
        "config",
    ],
)
def test_conflicting_push_evidence_cannot_feed_github(pushed, monkeypatch, fault):
    module, application, reservations, identity, publication, receipts, completed = pushed
    conflict = importlib.import_module("operations.maintenance_intake").MaintenanceTaskConflict
    if fault == "state":
        publication.result.state = RemoteGitDeliveryState.AMBIGUOUS
    elif fault == "cleanup":
        publication.result.cleanup_settled = False
    elif fault in {"approval_id", "approval_fingerprint", "tree"}:
        setattr(publication.result, fault, "different")
    elif fault == "receipt_state":
        receipts[-1].state = RemoteGitDeliveryState.APPROVAL_REQUIRED
    elif fault == "receipt_digest":
        receipts[-1].evidence_sha256 = "sha256:" + "0" * 64
    elif fault == "receipt_absent":
        receipts.clear()
    else:
        monkeypatch.setattr(module, "_configured_git_request", lambda *args: None)

    async def scenario():
        with pytest.raises(conflict):
            await module.load_verified_git_result(application, reservations, identity)
        assert await application.app.task_store.load_task(completed.id) == completed

    asyncio.run(scenario())


@pytest.mark.parametrize("state", ["missing", "pending", "claimed", "cancelled"])
def test_noncompleted_delivery_does_not_read_remote_result(approval_context, monkeypatch, state):
    module, application, reservations, identity, _native, _publication, _receipts, options = (
        approval_context
    )
    conflict = importlib.import_module("operations.maintenance_intake").MaintenanceTaskConflict

    async def scenario():
        if state != "missing":
            await module.ensure_git_delivery_task(application, reservations, identity, **options)
        if state == "claimed":
            await application.app.task_store.claim_task(
                "worker", TaskQuery(type="maintenance.git_delivery")
            )
        elif state == "cancelled":
            await application.app.task_store.cancel_task(identity.git_delivery_task_id)
        before = await application.app.task_store.load_task(identity.git_delivery_task_id)

        def forbidden(*args):
            pytest.fail("Unproven completion reached native Git evidence")

        monkeypatch.setattr(module, "RemoteGitDeliveryRepository", forbidden)
        with pytest.raises(conflict):
            await module.load_verified_git_result(application, reservations, identity)
        assert await application.app.task_store.load_task(identity.git_delivery_task_id) == before

    asyncio.run(scenario())


def test_cancel_native_readback_preserves_completed_task(pushed, monkeypatch):
    module, application, reservations, identity, _publication, _receipts, completed = pushed

    async def scenario():
        entered = asyncio.Event()

        async def wait(*args):
            entered.set()
            await asyncio.Event().wait()

        monkeypatch.setattr(module.RemoteGitDeliveryRepository, "load_result", wait)
        owner = asyncio.create_task(
            module.load_verified_git_result(application, reservations, identity)
        )
        try:
            await asyncio.wait_for(entered.wait(), 5)
            owner.cancel("stop-git-readback")
            with pytest.raises(asyncio.CancelledError, match="stop-git-readback"):
                await owner
            assert owner.cancelled() and owner.cancelling() == 1
            assert await application.app.task_store.load_task(completed.id) == completed
        finally:
            if not owner.done():
                owner.cancel()
            await asyncio.gather(owner, return_exceptions=True)

    asyncio.run(scenario())


@pytest.mark.parametrize("bad", ["boolean", "digest", "fingerprint", "extra"])
def test_completed_delivery_requires_exact_receipt_shape(pushed, monkeypatch, bad):
    module, application, reservations, identity, _publication, _receipts, completed = pushed
    conflict = importlib.import_module("operations.maintenance_intake").MaintenanceTaskConflict
    corrupt = completed.model_copy(deep=True)
    if bad == "boolean":
        corrupt.result["result_digest"] = True
    elif bad == "digest":
        corrupt.result["result_digest"] = "a" * 64
    elif bad == "fingerprint":
        corrupt.result["request_fingerprint"] = "sha256:" + "0" * 64
    else:
        corrupt.result["extra"] = "not-authority"

    async def stored(_task_id):
        return corrupt

    def forbidden(*args):
        pytest.fail("Malformed receipt reached Git artifact readback")

    monkeypatch.setattr(application.app.task_store, "load_task", stored)
    monkeypatch.setattr(module, "RemoteGitDeliveryRepository", forbidden)

    async def scenario():
        with pytest.raises(conflict):
            await module.load_verified_git_result(application, reservations, identity)

    asyncio.run(scenario())

"""Exact approval queue with real task stores and controlled native evidence reads."""

import asyncio
import importlib
from types import SimpleNamespace

import pytest

from cayu import (
    RemoteGitDeliveryApproval,
    RemoteGitDeliveryRequest,
    RemoteGitDeliveryState,
    TaskQuery,
)
from tests.qualification.test_repository_maintenance_git_intake import git_intake as git_intake
from tests.qualification.test_repository_maintenance_intake import intake as intake


@pytest.fixture
def approval_context(git_intake, monkeypatch):
    module, application, reservations, identity = git_intake
    digest = "sha256:" + "f" * 64
    application.artifact_store = object()

    async def setup():
        queued = await module.ensure_git_preparation_task(
            application, reservations, identity, actor_subject="preparer"
        )
        claimed = await application.app.task_store.claim_task(
            "preparer", TaskQuery(type="maintenance.git_preparation")
        )
        native = RemoteGitDeliveryRequest.model_validate_json(queued.input["request_json"])
        await application.app.task_store.complete_task(
            queued.id,
            {"request_fingerprint": native.fingerprint, "result_digest": digest},
            worker_id="preparer",
            lease_expires_at=claimed.lease_expires_at,
        )
        return native

    native = asyncio.run(setup())
    prepared = SimpleNamespace(request_fingerprint=native.fingerprint, tree="a" * 40)
    pending = SimpleNamespace(
        result=SimpleNamespace(state=RemoteGitDeliveryState.APPROVAL_REQUIRED, tree=prepared.tree),
        artifact=SimpleNamespace(sha256=digest),
    )
    receipts = [
        SimpleNamespace(state=RemoteGitDeliveryState.APPROVAL_REQUIRED, evidence_sha256=digest)
    ]

    class ReadOnlyRepository:
        def __init__(self, store):
            assert store is application.artifact_store

        async def load_result(self, request, expected_digest):
            assert request == native and expected_digest == digest
            return pending

        async def load_prepared(self, request):
            assert request == native
            return prepared

        async def load_lifecycle(self, request):
            assert request == native
            return receipts

    monkeypatch.setattr(module, "RemoteGitDeliveryRepository", ReadOnlyRepository)
    options = dict(
        actor_subject="approver",
        expected_request_fingerprint=native.fingerprint,
        expected_tree=prepared.tree,
        approval_id="decision-1",
    )
    return module, application, reservations, identity, native, pending, receipts, options


def test_exact_approval_survives_enqueue_ack_loss_and_native_progress(
    approval_context, monkeypatch
):
    module, application, reservations, identity, native, _pending, receipts, options = (
        approval_context
    )
    create = application.app.create_task
    calls = 0

    async def lose(request):
        nonlocal calls
        calls += 1
        await create(request)
        raise ConnectionError("ack lost")

    monkeypatch.setattr(application.app, "create_task", lose)

    async def scenario():
        with pytest.raises(ConnectionError):
            await module.ensure_git_delivery_task(application, reservations, identity, **options)
        original = await application.app.task_store.load_task(identity.git_delivery_task_id)
        receipts[-1].state = RemoteGitDeliveryState.PUSHED
        replay = await module.ensure_git_delivery_task(
            application, reservations, identity, **options
        )
        assert replay == original and calls == 1
        assert replay.invocation.origin.subject == "approver"
        approval = RemoteGitDeliveryApproval.model_validate_json(replay.input["approval_json"])
        assert approval.request_fingerprint == native.fingerprint
        assert approval.prepared_tree == options["expected_tree"]
        assert approval.commit_approved is True and approval.push_approved is True

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "field", ["expected_request_fingerprint", "expected_tree", "approval_id", "actor_subject"]
)
def test_changed_expected_approval_cannot_replace_task(approval_context, field):
    module, application, reservations, identity, _native, _pending, _receipts, options = (
        approval_context
    )
    conflict = importlib.import_module("operations.maintenance_intake").MaintenanceTaskConflict

    async def scenario():
        original = await module.ensure_git_delivery_task(
            application, reservations, identity, **options
        )
        changed = dict(options)
        changed[field] = "different"
        with pytest.raises(conflict):
            await module.ensure_git_delivery_task(application, reservations, identity, **changed)
        assert await application.app.task_store.load_task(identity.git_delivery_task_id) == original

    asyncio.run(scenario())


@pytest.mark.parametrize("case", ["advanced", "digest", "state", "tree"])
def test_unproven_pending_evidence_cannot_enqueue(approval_context, case):
    module, application, reservations, identity, _native, pending, receipts, options = (
        approval_context
    )
    conflict = importlib.import_module("operations.maintenance_intake").MaintenanceTaskConflict
    if case == "advanced":
        receipts[-1].state = RemoteGitDeliveryState.PUSHED
    elif case == "digest":
        receipts[-1].evidence_sha256 = "sha256:" + "0" * 64
    elif case == "state":
        pending.result.state = RemoteGitDeliveryState.CONFLICT
    else:
        pending.result.tree = "0" * 40

    async def scenario():
        with pytest.raises(conflict):
            await module.ensure_git_delivery_task(application, reservations, identity, **options)
        assert await application.app.task_store.load_task(identity.git_delivery_task_id) is None

    asyncio.run(scenario())


def test_approval_readback_real_cancellation_cannot_enqueue(approval_context, monkeypatch):
    module, application, reservations, identity, _native, _pending, _receipts, options = (
        approval_context
    )

    async def scenario():
        entered = asyncio.Event()

        async def wait(*args):
            entered.set()
            await asyncio.Event().wait()

        monkeypatch.setattr(module, "load_git_approval_request", wait)
        owner = asyncio.create_task(
            module.ensure_git_delivery_task(application, reservations, identity, **options)
        )
        await entered.wait()
        owner.cancel("stop-approval")
        with pytest.raises(asyncio.CancelledError, match="stop-approval"):
            await owner
        assert owner.cancelling() == 1 and owner.cancelled()
        assert await application.app.task_store.load_task(identity.git_delivery_task_id) is None

    asyncio.run(scenario())

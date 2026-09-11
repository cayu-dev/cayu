"""Real task-store enqueue; coding verification is controlled in these unit cases."""

import asyncio
import importlib
import json
from types import SimpleNamespace

import pytest

from cayu import (
    RemoteGitDeliveryRequest,
    RemoteGitSourceAuthority,
    TaskQuery,
    TaskStatus,
    run_task_worker,
)
from tests.qualification.test_repository_maintenance_delivery_configuration import authority
from tests.qualification.test_repository_maintenance_git_configuration import arguments
from tests.qualification.test_repository_maintenance_intake import intake as intake


@pytest.fixture
def git_intake(intake, monkeypatch):
    _coding, _domain, reservations, app, identity = intake
    module = importlib.import_module("operations.maintenance_git_intake")
    monkeypatch.setenv("CAYU_MAINTENANCE_GIT_JSON", json.dumps(authority()))
    application = SimpleNamespace(app=app)

    async def verified(given, registry, expected):
        assert given is application and registry is reservations
        assert expected == identity
        return None, object()

    def request(_publication, **kwargs):
        return RemoteGitDeliveryRequest(
            **kwargs,
            source=RemoteGitSourceAuthority(
                product_result_artifact_id="product-artifact",
                product_result_sha256="a" * 64,
                product_request_fingerprint="b" * 64,
                product_run_id=identity.product_run_id,
                source_workspace_id="workspace",
                final_source_revision="c" * 64,
                diff_artifact_id="diff-artifact",
                diff_sha256="d" * 64,
                check_evidence_sha256="e" * 64,
            ),
        )

    monkeypatch.setattr(module, "load_verified_coding_result", verified)
    monkeypatch.setattr(module, "remote_git_delivery_request", request)
    return module, application, reservations, identity


def test_preparation_task_retains_exact_authority_and_operator(git_intake):
    module, application, reservations, identity = git_intake

    async def scenario():
        created = await module.ensure_git_preparation_task(
            application, reservations, identity, actor_subject="maintenance-operator"
        )
        assert created.id == identity.git_preparation_task_id
        assert created.status is TaskStatus.PENDING
        assert created.invocation.origin.subject == "maintenance-operator"
        assert created.invocation.origin.subject != identity.intent.subject
        assert created.invocation.origin.tenant == identity.intent.tenant
        native = RemoteGitDeliveryRequest.model_validate_json(created.input["request_json"])
        assert native.delivery_id == native.idempotency_key == identity.git_delivery_task_id
        assert native.session_id == identity.session_id
        assert native.source.product_run_id == identity.product_run_id
        assert native.commit.authored_at == "2026-09-10T00:00:00+00:00"
        claimed = await application.app.task_store.claim_task(
            "worker", TaskQuery(type="maintenance.git_preparation")
        )
        terminal = await application.app.task_store.complete_task(
            created.id,
            {"phase": "prepared"},
            worker_id="worker",
            lease_expires_at=claimed.lease_expires_at,
        )
        assert (
            await module.ensure_git_preparation_task(
                application, reservations, identity, actor_subject="maintenance-operator"
            )
            == terminal
        )
        assert terminal.invocation == created.invocation

    asyncio.run(scenario())


@pytest.mark.parametrize("change", ["actor", "commit", "destination", "limit"])
def test_changed_preparation_authority_conflicts(git_intake, monkeypatch, change):
    module, application, reservations, identity = git_intake
    conflict = importlib.import_module("operations.maintenance_intake").MaintenanceTaskConflict

    async def scenario():
        created = await module.ensure_git_preparation_task(
            application, reservations, identity, actor_subject="operator"
        )
        value = authority()
        if change == "commit":
            value["commit"]["authored_at"] = "2026-09-11T00:00:00+00:00"
        elif change == "destination":
            value["repository"]["destination_ref"] = "refs/heads/cayu/other"
        elif change == "limit":
            value["limits"]["timeout_seconds"] = 121
        monkeypatch.setenv("CAYU_MAINTENANCE_GIT_JSON", json.dumps(value))
        with pytest.raises(conflict):
            await module.ensure_git_preparation_task(
                application,
                reservations,
                identity,
                actor_subject="other" if change == "actor" else "operator",
            )
        assert await application.app.task_store.load_task(created.id) == created

    asyncio.run(scenario())


def test_committed_creation_ack_loss_replays_once(git_intake, monkeypatch):
    module, application, reservations, identity = git_intake
    original = application.app.create_task
    calls = 0

    async def lose(request):
        nonlocal calls
        calls += 1
        await original(request)
        raise ConnectionError("acknowledgement lost")

    monkeypatch.setattr(application.app, "create_task", lose)

    async def scenario():
        with pytest.raises(ConnectionError):
            await module.ensure_git_preparation_task(
                application, reservations, identity, actor_subject="operator"
            )
        observed = await application.app.task_store.load_task(identity.git_preparation_task_id)
        assert (
            await module.ensure_git_preparation_task(
                application, reservations, identity, actor_subject="operator"
            )
            == observed
        )
        assert calls == 1

    asyncio.run(scenario())


def test_real_readback_cancellation_does_not_enqueue(git_intake, monkeypatch):
    module, application, reservations, identity = git_intake

    async def scenario():
        entered = asyncio.Event()

        async def wait(*args):
            entered.set()
            await asyncio.Event().wait()

        monkeypatch.setattr(module, "load_verified_coding_result", wait)
        owner = asyncio.create_task(
            module.ensure_git_preparation_task(
                application, reservations, identity, actor_subject="operator"
            )
        )
        await entered.wait()
        owner.cancel("stop-preparation")
        with pytest.raises(asyncio.CancelledError, match="stop-preparation"):
            await owner
        assert owner.cancelling() == 1 and owner.cancelled()
        assert await application.app.task_store.load_task(identity.git_preparation_task_id) is None

    asyncio.run(scenario())


def test_unverified_result_and_invalid_actor_cannot_enqueue(git_intake, monkeypatch):
    module, application, reservations, identity = git_intake
    calls = 0

    async def rejected(*args):
        nonlocal calls
        calls += 1
        raise ValueError("unverified")

    monkeypatch.setattr(module, "load_verified_coding_result", rejected)

    async def scenario():
        with pytest.raises(ValueError, match="preparation actor"):
            await module.ensure_git_preparation_task(
                application, reservations, identity, actor_subject=True
            )
        assert calls == 0
        with pytest.raises(ValueError, match="unverified"):
            await module.ensure_git_preparation_task(
                application, reservations, identity, actor_subject="operator"
            )
        assert calls == 1
        assert await application.app.task_store.load_task(identity.git_preparation_task_id) is None

    asyncio.run(scenario())


def test_concurrent_preparation_intake_has_one_winner(git_intake, monkeypatch):
    module, application, reservations, identity = git_intake
    store = application.app.task_store
    original = store.load_task

    async def scenario():
        both = asyncio.Event()
        initial_reads = 0

        async def barrier(task_id):
            nonlocal initial_reads
            result = await original(task_id)
            if result is None and initial_reads < 2:
                initial_reads += 1
                if initial_reads == 2:
                    both.set()
                await both.wait()
            return result

        monkeypatch.setattr(store, "load_task", barrier)
        first, second = await asyncio.gather(
            *(
                module.ensure_git_preparation_task(
                    application, reservations, identity, actor_subject="operator"
                )
                for _ in range(2)
            )
        )
        assert first == second
        assert initial_reads == 2
        assert await original(identity.git_preparation_task_id) == first

    asyncio.run(scenario())


def test_claimed_preparation_reconstructs_store_authority(git_intake):
    module, application, reservations, identity = git_intake
    conflict = importlib.import_module("operations.maintenance_intake").MaintenanceTaskConflict

    async def scenario():
        queued = await module.ensure_git_preparation_task(
            application, reservations, identity, actor_subject="operator"
        )
        with pytest.raises(conflict):
            await module.load_claimed_git_preparation(
                application.app, reservations, queued, "worker"
            )
        claimed = await application.app.task_store.claim_task(
            "worker", TaskQuery(type="maintenance.git_preparation")
        )
        restored, native = await module.load_claimed_git_preparation(
            application.app, reservations, claimed, "worker"
        )
        assert restored == identity
        assert native == RemoteGitDeliveryRequest.model_validate_json(queued.input["request_json"])
        with pytest.raises(conflict):
            await module.load_claimed_git_preparation(
                application.app, reservations, claimed, "other"
            )
        altered = claimed.model_copy(deep=True)
        altered.input["maintenance_run_id"] = "other"
        with pytest.raises(conflict):
            await module.load_claimed_git_preparation(
                application.app, reservations, altered, "worker"
            )
        assert await application.app.task_store.load_task(claimed.id) == claimed

    asyncio.run(scenario())


@pytest.mark.parametrize("bad", ["duplicate", "oversize", "wrong-type", "identity"])
def test_claimed_preparation_rejects_corrupt_durable_input(git_intake, monkeypatch, bad):
    module, application, reservations, identity = git_intake
    conflict = importlib.import_module("operations.maintenance_intake").MaintenanceTaskConflict

    async def scenario():
        await module.ensure_git_preparation_task(
            application, reservations, identity, actor_subject="operator"
        )
        claimed = await application.app.task_store.claim_task(
            "worker", TaskQuery(type="maintenance.git_preparation")
        )
        corrupt = claimed.model_copy(deep=True)
        if bad == "duplicate":
            corrupt.input["request_json"] = '{"delivery_id":"a","delivery_id":"b"}'
        elif bad == "oversize":
            corrupt.input["request_json"] = "x" * 65537
        elif bad == "wrong-type":
            corrupt.input["request_json"] = True
        else:
            raw = json.loads(corrupt.input["request_json"])
            raw["source"]["product_run_id"] = "different-product"
            corrupt.input["request_json"] = json.dumps(raw, sort_keys=True, separators=(",", ":"))

        async def stored(_task_id):
            return corrupt

        monkeypatch.setattr(application.app.task_store, "load_task", stored)
        with pytest.raises(conflict):
            await module.load_claimed_git_preparation(
                application.app, reservations, corrupt, "worker"
            )

    asyncio.run(scenario())


def test_preparation_worker_cancellation_retains_dispatched_owner(
    git_intake, monkeypatch, tmp_path
):
    module, application, reservations, identity = git_intake
    worker_module = importlib.import_module("operations.maintenance_git_worker")
    integration = importlib.import_module("integrations.remote_git")
    options = arguments(tmp_path)
    options["remote_url"] = str(tmp_path / "remote")
    broker = integration.build_remote_git_delivery_broker(**options)
    application.artifact_store = options["artifact_store"]
    monkeypatch.setattr(
        worker_module, "load_verified_coding_result", module.load_verified_coding_result
    )

    async def scenario():
        queued = await module.ensure_git_preparation_task(
            application, reservations, identity, actor_subject="operator"
        )
        native = RemoteGitDeliveryRequest.model_validate_json(queued.input["request_json"])
        monkeypatch.setattr(
            worker_module, "_configured_git_request", lambda *args, **kwargs: native
        )
        entered, release = asyncio.Event(), asyncio.Event()
        calls = 0

        async def dispatched(*args, approval=None):
            assert approval is None
            nonlocal calls
            calls += 1
            entered.set()
            await release.wait()
            return SimpleNamespace(artifact=SimpleNamespace(sha256="a" * 64))

        monkeypatch.setattr(worker_module, "run_verified_git_delivery", dispatched)

        async def handle(_app, claimed, worker_id):
            await worker_module.handle_git_preparation_task(
                application, reservations, claimed, worker_id, broker
            )

        store = application.app.task_store
        query = TaskQuery(type="maintenance.git_preparation")
        worker = asyncio.create_task(
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
            await asyncio.wait_for(entered.wait(), timeout=5)
            worker.cancel("stop-preparation-owner")
            for _ in range(100):
                current = await store.load_task(queued.id)
                if current.status_reason == "cancellation_requested":
                    break
                await asyncio.sleep(0.01)
            else:
                pytest.fail("Worker did not retain cancellation fence")
            await asyncio.sleep(1.05)
            assert not worker.done()
            assert await store.reclaim_expired(query=query) == []
            assert await store.claim_task("replacement", query, lease_seconds=1) is None
            release.set()
            done, _ = await asyncio.wait((worker,), timeout=5)
            assert worker in done
            with pytest.raises(asyncio.CancelledError, match="stop-preparation-owner"):
                await worker
            assert worker.cancelled() and worker.cancelling() == 1
            final = await store.load_task(queued.id)
            assert final.status is TaskStatus.CANCELLED and not final.result
            assert final.worker_id is None and final.lease_expires_at is None
            assert calls == 1
        finally:
            release.set()
            if not worker.done():
                worker.cancel()
            await asyncio.gather(worker, return_exceptions=True)

    asyncio.run(scenario())

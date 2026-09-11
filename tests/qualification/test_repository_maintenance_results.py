"""Delivery cannot invent a completed coding result from a reservation or task."""

import asyncio
import importlib
import warnings

import pytest

from cayu import TaskQuery
from tests.cli.test_scaffold_coding_budget import denial_policy
from tests.qualification.test_repository_maintenance_application import project as project
from tests.qualification.test_repository_maintenance_request import consumer as consumer


@pytest.fixture
def result_context(consumer, tmp_path):
    application, task, provider, domain, requests, _workflow = consumer
    identities = importlib.import_module("domain.maintenance_identity")
    stores = importlib.import_module("operations.maintenance_runs")
    intake = importlib.import_module("operations.maintenance_intake")
    results = importlib.import_module("operations.maintenance_results")
    registry = stores.SQLiteMaintenanceRunStore(tmp_path / "result-reservations.sqlite")

    async def prepare():
        await registry.initialize()
        accepted = await requests.capture_accepted_request(application, task)
        return await registry.reserve(
            identities.MaintenanceRunIntent(
                tenant="tenant",
                subject="operator",
                idempotency_key="readback",
                request_json=domain.encode_request(accepted),
            )
        )

    identity = asyncio.run(prepare())
    return application, registry, identity, intake, results, provider


def forbid_artifacts(monkeypatch, results):
    def forbidden(*args, **kwargs):
        pytest.fail("Unproven completion reached artifact readback")

    monkeypatch.setattr(results, "CodingProductArtifactRepository", forbidden)


@pytest.mark.parametrize("consumer", ["memory", "sqlite"], indirect=True)
@pytest.mark.parametrize("state", ["missing", "pending", "claimed", "failed", "cancelled"])
def test_non_completed_task_cannot_start_delivery(result_context, monkeypatch, state):
    application, registry, identity, intake, results, provider = result_context
    forbid_artifacts(monkeypatch, results)

    async def scenario():
        store = application.app.task_store
        if state != "missing":
            await intake.ensure_coding_task(application.app, registry, identity)
        if state == "cancelled":
            await store.cancel_task(identity.task_id)
        if state in {"claimed", "failed"}:
            claimed = await store.claim_task("worker", TaskQuery(type="maintenance.coding"))
            if state == "failed":
                await store.fail_task(
                    identity.task_id,
                    {"error": "rejected"},
                    worker_id="worker",
                    lease_expires_at=claimed.lease_expires_at,
                )
        before = await store.load_task(identity.task_id)
        with pytest.raises(results.MaintenanceResultUnavailable):
            await results.load_verified_coding_result(application, registry, identity)
        assert await store.load_task(identity.task_id) == before
        assert not provider.requests

    asyncio.run(scenario())


async def complete(context, result):
    application, registry, identity, intake, _results, _provider = context
    store = application.app.task_store
    await intake.ensure_coding_task(application.app, registry, identity)
    claimed = await store.claim_task("worker", TaskQuery(type="maintenance.coding"))
    return await store.complete_task(
        identity.task_id, result, worker_id="worker", lease_expires_at=claimed.lease_expires_at
    )


@pytest.mark.parametrize("consumer", ["memory", "sqlite"], indirect=True)
@pytest.mark.parametrize("fault", ["missing", "extra", "product", "boolean", "digest"])
def test_completed_task_requires_exact_result_shape(result_context, monkeypatch, fault):
    application, registry, identity, _intake, results, provider = result_context
    forbid_artifacts(monkeypatch, results)
    value = {"product_run_id": identity.product_run_id, "result_digest": "a" * 64}
    if fault == "missing":
        value.pop("result_digest")
    elif fault == "extra":
        value["extra"] = "not-authority"
    elif fault == "product":
        value["product_run_id"] = identity.public_id
    elif fault == "boolean":
        value["result_digest"] = True
    else:
        value["result_digest"] = "sha256:" + "a" * 64

    async def scenario():
        before = await complete(result_context, value)
        with pytest.raises(results.MaintenanceResultUnavailable):
            await results.load_verified_coding_result(application, registry, identity)
        assert await application.app.task_store.load_task(identity.task_id) == before
        assert not provider.requests

    asyncio.run(scenario())


def test_current_configuration_must_match_accepted_result(result_context, monkeypatch):
    application, registry, identity, _intake, results, provider = result_context
    forbid_artifacts(monkeypatch, results)

    async def scenario():
        await complete(
            result_context, {"product_run_id": identity.product_run_id, "result_digest": "a" * 64}
        )
        application.app.budget_policy = denial_policy()
        with pytest.raises(results.MaintenanceResultUnavailable):
            await results.load_verified_coding_result(application, registry, identity)
        assert not provider.requests

    asyncio.run(scenario())


def test_changed_expected_reservation_rejects_before_task_lookup(result_context, monkeypatch):
    application, registry, identity, _intake, results, _provider = result_context

    async def forbidden(*args, **kwargs):
        pytest.fail("Conflicting reservation reached task lookup")

    monkeypatch.setattr(application.app.task_store, "load_task", forbidden)

    async def scenario():
        changed = identity.model_copy(
            update={"intent": identity.intent.model_copy(update={"subject": "different"})}
        )
        with pytest.raises(results.MaintenanceResultUnavailable):
            await results.load_verified_coding_result(application, registry, changed)

    asyncio.run(scenario())


def test_corrupt_result_diagnostics_do_not_serialize_values(
    result_context, monkeypatch, caplog, capsys
):
    application, registry, identity, _intake, results, _provider = result_context
    forbid_artifacts(monkeypatch, results)
    canary = "private-result-canary"

    class Hostile:
        def __repr__(self):
            return canary

    async def scenario():
        task = await complete(
            result_context, {"product_run_id": identity.product_run_id, "result_digest": "a" * 64}
        )
        task.result = {"product_run_id": canary, "result_digest": Hostile()}

        async def corrupted(_id):
            return task

        monkeypatch.setattr(application.app.task_store, "load_task", corrupted)
        with pytest.raises(results.MaintenanceResultUnavailable) as caught:
            await results.load_verified_coding_result(application, registry, identity)
        assert canary not in str(caught.value) + repr(caught.value)

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        asyncio.run(scenario())
    assert all(canary not in str(item.message) for item in captured)
    output = capsys.readouterr()
    assert canary not in caplog.text + output.out + output.err


def test_readback_cancellation_preserves_task_and_original_signal(result_context, monkeypatch):
    application, registry, identity, intake, results, provider = result_context

    async def scenario():
        before = await intake.ensure_coding_task(application.app, registry, identity)
        entered = asyncio.Event()

        async def blocked(**kwargs):
            entered.set()
            await asyncio.Event().wait()

        monkeypatch.setattr(registry, "load_owned", blocked)
        owner = asyncio.create_task(
            results.load_verified_coding_result(application, registry, identity)
        )
        try:
            await asyncio.wait_for(entered.wait(), timeout=5)
            owner.cancel("stop-delivery-readback")
            with pytest.raises(asyncio.CancelledError, match="stop-delivery-readback"):
                await owner
            assert owner.cancelled() and owner.cancelling() == 1
        finally:
            if not owner.done():
                owner.cancel()
            await asyncio.gather(owner, return_exceptions=True)
        assert await application.app.task_store.load_task(identity.task_id) == before
        assert not provider.requests

    asyncio.run(scenario())

"""Public reservation/task handoff, not HTTP authentication or coding execution."""

import asyncio
import importlib
import warnings

import pytest

from cayu import (
    CayuApp,
    InMemoryTaskStore,
    InvocationOriginClaim,
    TaskCreate,
    TaskQuery,
    TaskStatus,
)
from cayu.cli.project import project_context
from cayu.storage.sqlite import SQLiteTaskStore
from tests.qualification.repository_maintenance_application import maintenance_project_files


@pytest.fixture(params=["memory", "sqlite"])
def intake(request, tmp_path):
    for name, content in maintenance_project_files().items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    with project_context(tmp_path):
        module = importlib.import_module("operations.maintenance_intake")
        domain = importlib.import_module("domain.maintenance_identity")
        stores = importlib.import_module("operations.maintenance_runs")
        reservations = stores.SQLiteMaintenanceRunStore(tmp_path / "reservations.sqlite")
        task_store = (
            InMemoryTaskStore()
            if request.param == "memory"
            else SQLiteTaskStore(tmp_path / "tasks.sqlite")
        )
        app = CayuApp(task_store=task_store, enable_logging=False)
        try:
            asyncio.run(reservations.initialize())
            record = asyncio.run(
                reservations.reserve(
                    domain.MaintenanceRunIntent(
                        tenant="tenant-a",
                        subject="subject-a",
                        idempotency_key="request-1",
                        request_json='{"case":"fixed","configuration":"pinned"}',
                    )
                )
            )
            yield module, domain, reservations, app, record
        finally:
            if isinstance(task_store, SQLiteTaskStore):
                asyncio.run(task_store.close())


def test_create_and_terminal_replay_preserve_task_state(intake):
    module, _domain, reservations, app, record = intake

    async def scenario():
        created = await module.ensure_coding_task(app, reservations, record)
        assert created.id == record.task_id and created.status is TaskStatus.PENDING
        assert created.invocation.origin.trust.value == "host_asserted"
        assert created.invocation.origin.subject == "subject-a"
        assert created.invocation.origin.tenant == "tenant-a"
        assert await module.ensure_coding_task(app, reservations, record) == created
        claimed = await app.task_store.claim_task("worker", TaskQuery(type="maintenance.coding"))
        terminal = await app.task_store.complete_task(
            created.id,
            {"phase": "done"},
            worker_id="worker",
            lease_expires_at=claimed.lease_expires_at,
        )
        assert await module.ensure_coding_task(app, reservations, record) == terminal
        assert terminal.invocation == created.invocation

    asyncio.run(scenario())


def test_claimed_lookup_uses_saved_identity_without_creating_tasks(intake, monkeypatch):
    module, _domain, reservations, app, record = intake

    async def scenario():
        pending = await module.ensure_coding_task(app, reservations, record)
        with pytest.raises(module.MaintenanceTaskConflict):
            await module.load_claimed_coding_identity(app, reservations, pending, "worker")
        claimed = await app.task_store.claim_task("worker", TaskQuery(type="maintenance.coding"))

        async def forbidden_create(_request):
            raise AssertionError("Claimed lookup must never recreate a task.")

        monkeypatch.setattr(app, "create_task", forbidden_create)
        assert (
            await module.load_claimed_coding_identity(app, reservations, claimed, "worker")
            == record
        )
        with pytest.raises(module.MaintenanceTaskConflict):
            await module.load_claimed_coding_identity(app, reservations, claimed, "another-worker")
        for changed in (
            claimed.model_copy(update={"id": record.git_delivery_task_id}),
            claimed.model_copy(update={"input": {"maintenance_run_id": "wrong"}}),
        ):
            with pytest.raises(module.MaintenanceTaskConflict):
                await module.load_claimed_coding_identity(app, reservations, changed, "worker")

        async def missing(_task_id):
            return None

        monkeypatch.setattr(app.task_store, "load_task", missing)
        with pytest.raises(module.MaintenanceTaskConflict):
            await module.load_claimed_coding_identity(app, reservations, claimed, "worker")

    asyncio.run(scenario())


def test_cancelled_claim_lookup_keeps_task_unchanged(intake, monkeypatch):
    module, _domain, reservations, app, record = intake

    async def scenario():
        await module.ensure_coding_task(app, reservations, record)
        claimed = await app.task_store.claim_task("worker", TaskQuery(type="maintenance.coding"))
        entered = asyncio.Event()

        async def paused(_task_id):
            entered.set()
            await asyncio.Event().wait()

        monkeypatch.setattr(reservations, "load_for_task", paused)
        owner = asyncio.create_task(
            module.load_claimed_coding_identity(app, reservations, claimed, "worker")
        )
        try:
            await asyncio.wait_for(entered.wait(), timeout=5)
            owner.cancel("lookup stopped")
            with pytest.raises(asyncio.CancelledError, match="lookup stopped"):
                await owner
            assert owner.cancelled() and owner.cancelling() == 1
            assert await app.task_store.load_task(claimed.id) == claimed
        finally:
            if not owner.done():
                owner.cancel()
            await asyncio.gather(owner, return_exceptions=True)

    asyncio.run(scenario())


def test_concurrent_creation_race_reconciles_exact_task(intake, monkeypatch):
    module, _domain, reservations, app, record = intake

    async def scenario():
        original = app.task_store.load_task
        arrivals = 0
        both = asyncio.Event()

        async def synchronized_load(task_id):
            nonlocal arrivals
            value = await original(task_id)
            if value is None:
                arrivals += 1
                if arrivals == 2:
                    both.set()
                await asyncio.wait_for(both.wait(), timeout=5)
            return value

        monkeypatch.setattr(app.task_store, "load_task", synchronized_load)
        first, second = await asyncio.gather(
            module.ensure_coding_task(app, reservations, record),
            module.ensure_coding_task(app, reservations, record),
        )
        assert arrivals == 2 and first == second
        assert first.invocation.root_invocation_id == second.invocation.root_invocation_id

    asyncio.run(scenario())


def test_committed_task_response_loss_replays_without_creation(intake, monkeypatch):
    module, _domain, reservations, app, record = intake

    async def scenario():
        original = app.create_task
        committed = []

        async def lost_response(request):
            committed.append(await original(request))
            raise ConnectionError("committed response lost")

        monkeypatch.setattr(app, "create_task", lost_response)
        with pytest.raises(ConnectionError):
            await module.ensure_coding_task(app, reservations, record)
        replay = await module.ensure_coding_task(app, reservations, record)
        assert replay == committed[0] and len(committed) == 1

    asyncio.run(scenario())


@pytest.mark.parametrize("changed", ["title", "input", "metadata", "origin"])
def test_same_task_id_with_conflicting_authority_is_rejected(intake, changed):
    module, _domain, reservations, app, record = intake

    async def scenario():
        values = dict(
            task_id=record.task_id,
            type="maintenance.coding",
            title="Repository maintenance coding",
            input={"maintenance_run_id": record.public_id},
            metadata={"maintenance_intent_fingerprint": record.intent.fingerprint},
            invocation_origin=InvocationOriginClaim(subject="subject-a", tenant="tenant-a"),
        )
        if changed == "origin":
            values["invocation_origin"] = None
        else:
            values[changed] = "wrong" if changed == "title" else {"wrong": "value"}
        existing = await app.create_task(TaskCreate.model_validate(values))
        with pytest.raises(module.MaintenanceTaskConflict):
            await module.ensure_coding_task(app, reservations, record)
        assert await app.task_store.load_task(record.task_id) == existing

    asyncio.run(scenario())


def test_wrong_tenant_rejected_before_task_lookup(intake, monkeypatch):
    module, domain, reservations, app, record = intake
    forged = domain.MaintenanceRunIdentity.model_validate(
        {**record.model_dump(), "intent": {**record.intent.model_dump(), "tenant": "tenant-b"}}
    )

    async def forbidden_lookup(_task_id):
        raise AssertionError("Wrong tenant reached Runtime lookup.")

    monkeypatch.setattr(app.task_store, "load_task", forbidden_lookup)
    with pytest.raises(module.MaintenanceTaskConflict):
        asyncio.run(module.ensure_coding_task(app, reservations, forged))


def test_corrupt_task_field_does_not_render_secret(intake, monkeypatch, caplog, capsys):
    module, _domain, reservations, app, record = intake

    class Hostile:
        def __repr__(self):
            raise AssertionError("secret-canary-repr")

        def __eq__(self, _other):
            raise AssertionError("secret-canary-equality")

    async def scenario():
        task = await module.ensure_coding_task(app, reservations, record)
        damaged = task.model_copy(update={"input": {"maintenance_run_id": Hostile()}})

        async def corrupt_read(_task_id):
            return damaged

        monkeypatch.setattr(app.task_store, "load_task", corrupt_read)
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            with pytest.raises(module.MaintenanceTaskConflict) as error:
                await module.ensure_coding_task(app, reservations, record)
            assert "secret-canary" not in str(error.value)
        assert not captured

    asyncio.run(scenario())
    assert "secret-canary" not in caplog.text
    output = capsys.readouterr()
    assert not output.out and not output.err

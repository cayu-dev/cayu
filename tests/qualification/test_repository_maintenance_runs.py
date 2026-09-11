"""Emitted reservation-store conformance, separate from product-route authorization."""

import asyncio
import importlib
import os
import warnings
from concurrent.futures import ThreadPoolExecutor
from uuid import uuid4

import pytest

from cayu.cli.project import project_context
from tests.qualification.repository_maintenance_application import maintenance_project_files


@pytest.fixture(params=["sqlite", pytest.param("postgres", marks=pytest.mark.postgres)])
def reservation(request, tmp_path):
    backend = request.param
    if backend == "postgres" and not (
        os.environ.get("CAYU_TEST_POSTGRES_DSN") or os.environ.get("CAYU_REQUIRE_POSTGRES")
    ):
        pytest.skip("PostgreSQL reservation conformance needs the required PostgreSQL lane")
    for name, content in maintenance_project_files().items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    with project_context(tmp_path):
        module = importlib.import_module("operations.maintenance_runs")
        identity = importlib.import_module("domain.maintenance_identity")
        if backend == "sqlite":
            path = tmp_path / "reservations.sqlite"

            def factory():
                return module.SQLiteMaintenanceRunStore(path)

            yield module, identity, factory
        else:
            import psycopg
            from psycopg import sql
            from psycopg.conninfo import make_conninfo

            dsn = request.getfixturevalue("postgres_dsn")
            schema = "maintenance_" + uuid4().hex
            with psycopg.connect(dsn, autocommit=True) as connection:
                connection.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
            isolated = make_conninfo(dsn, options="-csearch_path=" + schema)
            try:
                yield module, identity, lambda: module.PostgresMaintenanceRunStore(isolated)
            finally:
                with psycopg.connect(dsn, autocommit=True) as connection:
                    connection.execute(
                        sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema))
                    )


def _intent(identity, **changes):
    return identity.MaintenanceRunIntent(
        **{
            "tenant": "tenant-a",
            "subject": "user-a",
            "idempotency_key": "request-1",
            "request_json": '{"case":"fixed","configuration":"pinned"}',
            **changes,
        }
    )


def test_rejected_input_never_opens_storage(reservation, monkeypatch, caplog, capsys):
    _module, identity, factory = reservation

    class Hostile:
        def __repr__(self):
            raise AssertionError("secret-canary-repr")

    store = factory()

    def unexpected_transaction(**_kwargs):
        raise AssertionError("Rejected input reached database dispatch.")

    monkeypatch.setattr(store, "_transaction", unexpected_transaction)
    damaged = _intent(identity, subject="secret-canary-sibling")
    object.__setattr__(damaged, "request_json", Hostile())

    async def scenario():
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            with pytest.raises(ValueError) as error:
                await store.reserve(damaged)
            assert "secret-canary" not in str(error.value)
            with pytest.raises(ValueError):
                await store.load_owned(tenant="tenant-a", public_id=None)
            with pytest.raises(ValueError):
                await store.load_for_task(True)
            for root in (True, "not-a-uuid", "", Hostile()):
                with pytest.raises(ValueError):
                    await store.reserve(_intent(identity), workflow_session_id=root)
            for expiry in (True, "not-a-date", "2000-01-01", Hostile()):
                with pytest.raises(ValueError):
                    await store.reserve(_intent(identity), coding_expires_at=expiry)
        assert not captured

    asyncio.run(scenario())
    assert "secret-canary" not in caplog.text
    output = capsys.readouterr()
    assert not output.out and not output.err


def test_initialize_replay_and_tenant_task_lookup(reservation):
    module, identity, factory = reservation

    async def scenario():
        store = factory()
        with pytest.raises(module.ReservationUnavailable):
            await store.check_ready()
        if isinstance(store, module.SQLiteMaintenanceRunStore):
            assert not store.path.exists()
        await store.initialize()
        await store.initialize()
        await store.check_ready()
        expected = _intent(identity)
        record = await store.reserve(expected)
        reopened = factory()
        assert await reopened.reserve(expected) == record
        assert await reopened.load_owned(tenant="tenant-a", public_id=record.public_id) == record
        assert await reopened.load_owned(tenant="tenant-b", public_id=record.public_id) is None
        assert await reopened.load_owned(tenant="tenant-a", public_id=str(uuid4())) is None
        for phase in identity.MaintenanceTaskPhase:
            assert await reopened.load_for_task(identity.task_id_for(record, phase)) == record
        assert await reopened.load_for_task(str(uuid4())) is None
        other = await reopened.reserve(_intent(identity, tenant="tenant-b"))
        assert other.public_id != record.public_id

    asyncio.run(scenario())


def test_explicit_workflow_root_replays_and_conflicts_without_rebinding(reservation):
    module, identity, factory = reservation

    async def scenario():
        await factory().initialize()
        intent = _intent(identity)
        root = str(uuid4())
        record = await factory().reserve(intent, workflow_session_id=root)
        assert record.workflow_session_id == root
        assert await factory().reserve(intent, workflow_session_id=root) == record
        # An ordinary intake retry retrieves the stored root rather than replacing it.
        assert await factory().reserve(intent) == record
        with pytest.raises(module.ReservationConflict):
            await factory().reserve(intent, workflow_session_id=str(uuid4()))
        assert (
            await factory().load_owned(tenant=intent.tenant, public_id=record.public_id) == record
        )
        with pytest.raises(module.ReservationUnavailable):
            await factory().reserve(
                _intent(identity, idempotency_key="another"), workflow_session_id=root
            )
        async with factory()._transaction() as db:
            rows = await db.fetch("SELECT public_id FROM maintenance_runs_v1")
            assert rows == [{"public_id": record.public_id}]

    asyncio.run(scenario())


def test_expired_reservation_replays_without_renewing_coding_time(reservation):
    module, identity, factory = reservation

    async def scenario():
        await factory().initialize()
        intent = _intent(identity)
        expiry = "2000-01-01T00:00:00+00:00"
        record = await factory().reserve(intent, coding_expires_at=expiry)
        assert record.coding_deadline().expired
        assert await factory().reserve(intent) == record
        assert await factory().reserve(intent, coding_expires_at=expiry) == record
        for changed in ("1999-01-01T00:00:00Z", "2099-01-01T00:00:00Z"):
            with pytest.raises(module.ReservationConflict):
                await factory().reserve(intent, coding_expires_at=changed)
        assert await factory().load_for_task(record.task_id) == record
        assert (
            await factory().load_owned(tenant=intent.tenant, public_id=record.public_id) == record
        )

    asyncio.run(scenario())


def test_concurrent_expiries_do_not_rebind_the_winning_reservation(reservation):
    module, identity, factory = reservation
    asyncio.run(factory().initialize())
    intent = _intent(identity)
    expiries = ["2000-01-01T00:00:00+00:00", "2001-01-01T00:00:00+00:00"]

    def reserve(expiry):
        try:
            return asyncio.run(factory().reserve(intent, coding_expires_at=expiry))
        except module.ReservationConflict:
            return None

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(reserve, expiries))
    assert sum(record is None for record in outcomes) == 1
    winner = next(record for record in outcomes if record is not None)
    assert winner.coding_expires_at in expiries
    assert asyncio.run(factory().reserve(intent)) == winner


def test_concurrent_workflow_roots_do_not_rebind_the_winning_reservation(reservation):
    module, identity, factory = reservation
    asyncio.run(factory().initialize())
    intent = _intent(identity)
    roots = [str(uuid4()), str(uuid4())]

    def reserve(root):
        try:
            return asyncio.run(factory().reserve(intent, workflow_session_id=root))
        except module.ReservationConflict:
            return None

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(reserve, roots))
    assert sum(record is None for record in outcomes) == 1
    winner = next(record for record in outcomes if record is not None)
    assert winner.workflow_session_id in roots
    assert asyncio.run(factory().reserve(intent)) == winner


def test_partial_schema_is_rejected_without_repair(reservation):
    module, _identity, factory = reservation

    async def scenario():
        store = factory()
        await store.initialize()
        async with store._transaction(write=True) as db:
            await db.execute("DROP TABLE maintenance_phase_tasks_v1")
        with pytest.raises(module.ReservationUnavailable):
            await store.initialize()
        with pytest.raises(module.ReservationUnavailable):
            await store.check_ready()
        async with store._transaction() as db:
            assert await db.tables() == {"maintenance_runs_v1"}

    asyncio.run(scenario())


def test_concurrent_conflicting_reservations_preserve_one_intent(reservation):
    module, identity, factory = reservation
    asyncio.run(factory().initialize())
    choices = [_intent(identity), _intent(identity, subject="different")]

    def reserve(choice):
        try:
            return asyncio.run(factory().reserve(choice))
        except module.ReservationConflict:
            return None

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(reserve, choices))
    assert sum(outcome is None for outcome in outcomes) == 1
    winner = next(outcome for outcome in outcomes if outcome is not None)
    assert asyncio.run(factory().reserve(winner.intent)) == winner


def test_cancellation_after_sql_dispatch_rolls_back_owned_transaction(reservation, monkeypatch):
    module, identity, factory = reservation
    adapter = (
        module._SQLiteSQL
        if isinstance(factory(), module.SQLiteMaintenanceRunStore)
        else module._PostgresSQL
    )
    original = adapter.execute

    async def scenario():
        store = factory()
        await store.initialize()
        dispatched = asyncio.Event()
        release = asyncio.Event()
        allocated = []

        async def pause_after_insert(self, sql, parameters: tuple = ()):
            await original(self, sql, parameters)
            if sql.startswith("INSERT INTO maintenance_phase_tasks_v1") and not allocated:
                allocated.append(parameters[1])
                dispatched.set()
                await release.wait()

        with monkeypatch.context() as patch:
            patch.setattr(adapter, "execute", pause_after_insert)
            owner = asyncio.create_task(store.reserve(_intent(identity)))
            try:
                await asyncio.wait_for(dispatched.wait(), timeout=5)
                assert await factory().load_owned(tenant="tenant-a", public_id=allocated[0]) is None
                owner.cancel("cancel reservation")
                with pytest.raises(asyncio.CancelledError) as error:
                    await owner
                assert error.value.args == ("cancel reservation",)
                assert owner.cancelled() and owner.cancelling() == 1
            finally:
                release.set()
                if not owner.done():
                    owner.cancel()
                await asyncio.gather(owner, return_exceptions=True)
        async with store._transaction() as db:
            assert await db.fetch("SELECT public_id FROM maintenance_runs_v1") == []
            assert await db.fetch("SELECT task_id FROM maintenance_phase_tasks_v1") == []
        replacement = await factory().reserve(_intent(identity))
        assert replacement.public_id != allocated[0]

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "change", [{"subject": "another"}, {"request_json": '{"configuration":"changed"}'}]
)
def test_same_tenant_key_conflicts_on_full_intent(reservation, change):
    module, identity, factory = reservation

    async def scenario():
        store = factory()
        await store.initialize()
        record = await store.reserve(_intent(identity))
        with pytest.raises(module.ReservationConflict):
            await store.reserve(_intent(identity, **change))
        assert await store.load_owned(tenant="tenant-a", public_id=record.public_id) == record

    asyncio.run(scenario())


def test_concurrent_identical_reservations_have_one_identity(reservation):
    _module, identity, factory = reservation
    asyncio.run(factory().initialize())
    expected = _intent(identity)
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(asyncio.run, factory().reserve(expected)) for _ in range(4)]
        records = [future.result(timeout=10) for future in futures]
    assert all(record == records[0] for record in records)


def test_partial_phase_insert_rolls_back(reservation, monkeypatch):
    module, identity, factory = reservation
    adapter = (
        module._SQLiteSQL
        if isinstance(factory(), module.SQLiteMaintenanceRunStore)
        else module._PostgresSQL
    )
    original = adapter.execute
    count = 0

    async def fail_second_phase(self, sql, parameters=()):
        nonlocal count
        if sql.startswith("INSERT INTO maintenance_phase_tasks_v1"):
            count += 1
            if count == 2:
                raise RuntimeError("injected phase-index failure")
        await original(self, sql, parameters)

    async def scenario():
        store = factory()
        await store.initialize()
        with monkeypatch.context() as patch:
            patch.setattr(adapter, "execute", fail_second_phase)
            with pytest.raises(RuntimeError, match="injected phase-index"):
                await store.reserve(_intent(identity))
        async with store._transaction() as db:
            assert await db.fetch("SELECT public_id FROM maintenance_runs_v1") == []
            assert await db.fetch("SELECT task_id FROM maintenance_phase_tasks_v1") == []
        await factory().reserve(_intent(identity))

    asyncio.run(scenario())


def test_lost_response_after_commit_reconciles_exactly(reservation):
    _module, identity, factory = reservation

    async def scenario():
        store = factory()
        await store.initialize()
        observed = []

        async def lose_response():
            observed.append(await store.reserve(_intent(identity)))
            raise ConnectionError("response lost after commit")

        with pytest.raises(ConnectionError):
            await lose_response()
        assert await factory().reserve(_intent(identity)) == observed[0]

    asyncio.run(scenario())


@pytest.mark.parametrize("corruption", ["missing_phase", "changed_index", "changed_json"])
def test_corrupt_reservation_cannot_authorize_lookup(reservation, corruption):
    module, identity, factory = reservation

    async def scenario():
        store = factory()
        await store.initialize()
        record = await store.reserve(_intent(identity))
        async with store._transaction(write=True) as db:
            if corruption == "missing_phase":
                await db.execute(
                    "DELETE FROM maintenance_phase_tasks_v1 WHERE task_id = %s",
                    (record.git_delivery_task_id,),
                )
            elif corruption == "changed_index":
                await db.execute("UPDATE maintenance_runs_v1 SET subject = %s", ("another",))
            else:
                await db.execute(
                    "UPDATE maintenance_runs_v1 SET identity_json = %s", ('{"secret":"canary"}',)
                )
        with pytest.raises(module.ReservationUnavailable) as error:
            await store.load_owned(tenant="tenant-a", public_id=record.public_id)
        assert "canary" not in str(error.value)
        # A wrong-tenant lookup must not even decode the corrupt private row.
        assert await store.load_owned(tenant="tenant-b", public_id=record.public_id) is None

    asyncio.run(scenario())

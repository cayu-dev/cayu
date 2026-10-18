"""The durable budget ledger ``open_application_stores`` builds beside its session store."""

from __future__ import annotations

import asyncio
from decimal import Decimal
from pathlib import Path

import pytest
from tests.core._execution_unit_fixtures import model_attempt_identity

from cayu import CayuApp
from cayu.budgets import BudgetLimit, BudgetReservation, InMemoryBudgetLedger
from cayu.budgets.base import BudgetLedger
from cayu.budgets.pricing import ModelPrice, PriceBook
from cayu.sessions import InMemorySessionStore
from cayu.storage.application import open_application_stores

_UNREACHABLE = "postgresql://app@127.0.0.1:1/unreachable"


@pytest.fixture(autouse=True)
def _clear_database_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "CAYU_DATABASE_URL",
        "CAYU_DATABASE_DIRECT_URL",
        "CAYU_DATABASE_POOL_MAX",
        "CAYU_REQUIRE_POSTGRES",
    ):
        monkeypatch.delenv(name, raising=False)


def _limit() -> BudgetLimit:
    return BudgetLimit.usd(
        1,
        pricing=PriceBook(
            prices=(
                ModelPrice.fixed(
                    provider_name="provider",
                    model="model",
                    match="exact",
                    input_per_million=Decimal("1"),
                    output_per_million=Decimal("1"),
                ),
            ),
        ),
        reservation=BudgetReservation(max_input_tokens=1000, max_output_tokens=1000),
    )


async def _reserve(ledger: BudgetLedger, session_id: str) -> str:
    result = await ledger.reserve(
        limit=_limit(),
        session_id=session_id,
        agent_name="assistant",
        provider_name="provider",
        model="model",
        model_attempt_identity=model_attempt_identity(),
    )
    assert result.accepted is True
    assert result.record is not None
    return result.record.reservation_id


def _app(stores, **kwargs) -> CayuApp:
    return CayuApp(session_store=stores.session_store, enable_logging=False, **kwargs)


def test_sqlite_stores_build_a_ledger_in_the_same_file_that_apps_use(tmp_path: Path) -> None:
    from cayu.storage import SQLiteBudgetLedger

    database = tmp_path / "data" / "cayu.db"

    async def scenario() -> str:
        async with open_application_stores(None, sqlite_path=database) as stores:
            assert isinstance(stores.budget_ledger, SQLiteBudgetLedger)
            assert stores.budget_ledger.path == database
            assert stores.connection_budget == 0
            assert "budget_ledger=True" in repr(stores)
            app = _app(stores)
            assert app.budget_ledger is stores.budget_ledger
            return await _reserve(app.budget_ledger, "session-before-restart")

    reservation_id = asyncio.run(scenario())

    async def after_restart() -> None:
        async with open_application_stores(None, sqlite_path=database) as stores:
            app = _app(stores)
            record = await app.budget_ledger.load_reservation(reservation_id)
            assert record is not None
            assert record.session_id == "session-before-restart"
            assert record.status == "active"

    asyncio.run(after_restart())


def test_an_app_supplied_ledger_wins_and_other_stores_keep_the_in_memory_default(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        async with open_application_stores(None, sqlite_path=tmp_path / "cayu.db") as stores:
            own = InMemoryBudgetLedger()
            assert _app(stores, budget_ledger=own).budget_ledger is own
        assert type(CayuApp(enable_logging=False).budget_ledger) is InMemoryBudgetLedger
        unrelated = CayuApp(session_store=InMemorySessionStore(), enable_logging=False)
        assert type(unrelated.budget_ledger) is InMemoryBudgetLedger

    asyncio.run(scenario())


class _UnhashableSessionStore(InMemorySessionStore):
    """A valid custom store that defines equality without a hash."""

    def __eq__(self, other: object) -> bool:
        return self is other

    __hash__ = None  # ty: ignore[invalid-assignment]


def test_an_unhashable_custom_session_store_keeps_the_in_memory_default() -> None:
    store = _UnhashableSessionStore()

    app = CayuApp(session_store=store, enable_logging=False)

    assert type(app.budget_ledger) is InMemoryBudgetLedger


def test_a_ledger_recorded_beside_an_unhashable_store_is_found_by_identity() -> None:
    from cayu.storage.application import ApplicationStores
    from cayu.storage.targets import SessionStoreBackend

    store = _UnhashableSessionStore()
    ledger = InMemoryBudgetLedger()
    ApplicationStores(
        backend=SessionStoreBackend.SQLITE,
        session_store=store,
        task_store=None,
        knowledge_store=None,
        budget_ledger=ledger,
    )

    assert CayuApp(session_store=store, enable_logging=False).budget_ledger is ledger
    # An equal-looking but distinct store does not inherit it.
    other = _UnhashableSessionStore()
    assert CayuApp(session_store=other, enable_logging=False).budget_ledger is not ledger


def test_a_record_is_dropped_with_its_session_store() -> None:
    import gc

    from cayu.budgets import _store_ledgers

    store = InMemorySessionStore()
    key = id(store)
    _store_ledgers.attach_session_store_budget_ledger(store, InMemoryBudgetLedger())
    assert key in _store_ledgers._LEDGERS

    del store
    gc.collect()

    assert key not in _store_ledgers._LEDGERS


def test_the_ledger_can_be_left_out(tmp_path: Path) -> None:
    async def scenario() -> None:
        async with open_application_stores(
            None, sqlite_path=tmp_path / "cayu.db", budget_ledger=False
        ) as stores:
            assert stores.budget_ledger is None
            assert "budget_ledger=False" in repr(stores)
            assert type(_app(stores).budget_ledger) is InMemoryBudgetLedger

    asyncio.run(scenario())
    with pytest.raises(TypeError, match="budget_ledger must be a bool"):
        open_application_stores(
            None,
            sqlite_path=tmp_path / "cayu.db",
            budget_ledger="yes",  # ty: ignore[invalid-argument-type]
        )


def test_closing_the_stores_closes_the_ledger(tmp_path: Path) -> None:
    stores = open_application_stores(None, sqlite_path=tmp_path / "cayu.db")
    ledger = stores.budget_ledger
    assert ledger is not None
    asyncio.run(stores.close())
    with pytest.raises(Exception, match="closed"):
        asyncio.run(_reserve(ledger, "after-close"))


def test_diagnostic_inspection_builds_no_ledger(tmp_path: Path) -> None:
    from cayu.storage._diagnostic_inspection import diagnostic_store_inspection

    missing = tmp_path / "missing" / "cayu.db"
    with diagnostic_store_inspection() as inspection:
        stores = open_application_stores(None, sqlite_path=missing)
        assert stores.budget_ledger is None
        asyncio.run(stores.close())
        postgres = open_application_stores(_UNREACHABLE, sqlite_path=missing)
        assert postgres.budget_ledger is None
        asyncio.run(postgres.close())
    inspection.verify()
    assert not missing.exists()


def test_postgres_ledger_shares_the_pool_without_connecting(tmp_path: Path) -> None:
    from cayu.storage import PostgresBudgetLedger
    from cayu.storage.migrations import SchemaMode

    stores = open_application_stores(_UNREACHABLE, sqlite_path=tmp_path / "cayu.db")
    ledger = stores.budget_ledger
    assert isinstance(ledger, PostgresBudgetLedger)
    assert ledger._pool is stores.session_store._pool
    assert ledger._owns_pool is False
    assert ledger._schema_mode is SchemaMode.VALIDATE
    assert stores.connection_budget == stores.pool_max_size + 1
    assert _app(stores).budget_ledger is ledger
    assert not (tmp_path / "cayu.db").exists()
    asyncio.run(stores.close())


def test_postgres_reservations_survive_a_restart(
    migrated_postgres_url: str, tmp_path: Path
) -> None:
    async def before_restart() -> str:
        async with open_application_stores(
            migrated_postgres_url, sqlite_path=tmp_path / "cayu.db"
        ) as stores:
            app = _app(stores)
            assert app.budget_ledger is stores.budget_ledger
            return await _reserve(app.budget_ledger, "postgres-before-restart")

    reservation_id = asyncio.run(before_restart())

    async def after_restart() -> None:
        async with open_application_stores(
            migrated_postgres_url, sqlite_path=tmp_path / "cayu.db"
        ) as stores:
            record = await _app(stores).budget_ledger.load_reservation(reservation_id)
            assert record is not None
            assert record.session_id == "postgres-before-restart"
            assert record.status == "active"

    asyncio.run(after_restart())
    assert not (tmp_path / "cayu.db").exists()


def _fork_caps_available() -> bool:
    import inspect

    from cayu.fork_controllers import ForkController

    return "fork_budget_limit" in inspect.signature(ForkController).parameters


@pytest.fixture(params=["sqlite", "postgres"])
def application_database_url(request: pytest.FixtureRequest) -> str | None:
    if request.param == "sqlite":
        return None
    return request.getfixturevalue("migrated_postgres_url")


@pytest.mark.skipif(
    not _fork_caps_available(), reason="Per-fork caps land with cayu-tech/cayu#2554."
)
def test_a_capped_forks_reservations_survive_a_restart(
    application_database_url: str | None, tmp_path: Path
) -> None:
    """An abandoned fork step stays reserved in the stores' ledger after a restart.

    The app gets no ``budget_ledger``; it uses the one ``open_application_stores``
    built. Abandoned steps write no cost event, so only the durable reservations
    keep the cap from admitting more spend once the process restarts.
    """

    from contextlib import aclosing

    from tests.core.test_session_fork_overlays import (
        SUBJECT,
        ScriptedProvider,
        _completed,
        _controller_toolset,
        _drain,
        _fork_request,
        _host_app,
        _run_trunk,
        fork_budget_limit,
    )

    from cayu import EventType, Message, ResumeRequest

    child = "compass-child"
    sqlite_path = tmp_path / "cayu.db"

    def invocation() -> ResumeRequest:
        return ResumeRequest(session_id=child, messages=[Message.text("user", "Investigate.")])

    async def abandon(stream) -> bool:
        async with aclosing(stream) as events:
            async for event in events:
                if event.type is EventType.MODEL_TEXT_DELTA:
                    return True
        return False

    def host(stores, provider):
        toolset, _server = _controller_toolset()
        app, _journal = _host_app(
            stores.session_store,
            provider,
            toolset=toolset,
            budget_limit=fork_budget_limit("0.30"),
            budget_ledger=None,
        )
        assert app.budget_ledger is stores.budget_ledger
        return app

    async def before_restart() -> list[bool]:
        async with open_application_stores(
            application_database_url, sqlite_path=sqlite_path
        ) as stores:
            app = host(stores, ScriptedProvider([_completed("closed"), *[_completed("x")] * 2]))
            try:
                await _run_trunk(app)
                controller = app.fork_controller_access(SUBJECT)
                await _drain(controller.fork(_fork_request(child)))
                # Two abandoned steps, 0.15 USD each, reserve the whole cap.
                return [await abandon(controller.resume(invocation())) for _ in range(2)]
            finally:
                await app.aclose()

    assert asyncio.run(before_restart()) == [True, True]

    async def after_restart():
        async with open_application_stores(
            application_database_url, sqlite_path=sqlite_path
        ) as stores:
            provider = ScriptedProvider([_completed("never")])
            app = host(stores, provider)
            try:
                events = await _drain(app.fork_controller_access(SUBJECT).resume(invocation()))
            finally:
                await app.aclose()
            return events, provider

    events, provider = asyncio.run(after_restart())
    assert [event for event in events if event.type is EventType.BUDGET_LIMIT_REACHED]
    assert provider.requests == []

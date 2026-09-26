"""Reservation inventory includes work whose event was never published."""

from decimal import Decimal
from uuid import uuid4

import pytest
from tests.core._execution_unit_fixtures import model_attempt_identity
from tests.core.test_budget_binding import _limit

from cayu.budgets.base import BudgetBindingRegistrationConflict, InMemoryBudgetLedger
from cayu.storage.budget_ledger import SQLiteBudgetLedger
from cayu.storage.migrations import SchemaMode
from cayu.storage.postgres import PostgresBudgetLedger


@pytest.mark.parametrize(
    "method",
    [
        "reserve",
        "reserve_batch",
        "release",
        "reconcile",
        "mark_dispatched",
        "load_reservation",
        "load_settlement",
        "_scan_reservation_records",
    ],
)
def test_budget_readback_does_not_inherit_qualification_over_changed_owners(method):
    async def changed_owner(*args, **kwargs):
        raise AssertionError("Opaque owner must not be invoked")

    opaque = type("OpaqueLedger", (InMemoryBudgetLedger,), {method: changed_owner})()
    assert not opaque._supports_producer_budget_readback()
    opaque._producer_budget_readback_version = 1
    assert not opaque._supports_producer_budget_readback()


def test_sqlite_reservation_inventory_uses_exact_session_index(tmp_path):
    from cayu.storage import _sqlite_support

    path = tmp_path / "index.sqlite"
    connection = _sqlite_support.connect(path)
    try:
        _sqlite_support.reconcile_schema(connection)
        _sqlite_support._validate_reservation_inventory_index(connection)
        plan = connection.execute(
            "EXPLAIN QUERY PLAN SELECT reservation_id FROM cayu_budget_reservations "
            "WHERE session_id = ? AND reservation_id > ? ORDER BY reservation_id LIMIT ?",
            ("producer", "", 2),
        ).fetchall()
        assert any("idx_cayu_budget_reservations_session_identity" in row[3] for row in plan)
        connection.execute("DROP INDEX idx_cayu_budget_reservations_session_identity")
        with pytest.raises(RuntimeError, match="inventory index"):
            _sqlite_support._validate_reservation_inventory_index(connection)
        connection.execute(
            "CREATE INDEX idx_cayu_budget_reservations_session_identity "
            "ON cayu_budget_reservations(reservation_id)"
        )
        with pytest.raises(RuntimeError, match="inventory index"):
            _sqlite_support._validate_reservation_inventory_index(connection)
    finally:
        connection.close()


@pytest.mark.parametrize("allowance", [True, 1.0, "1", None])
def test_binding_readback_rejects_malformed_durable_allowance(allowance):
    from cayu.budgets._reservation_scan import require_binding_readback

    with pytest.raises(BudgetBindingRegistrationConflict):
        require_binding_readback(("a" * 64, allowance), ("a" * 64, 1))


@pytest.mark.anyio
@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
async def test_inventory_retains_active_and_terminal_unpublished_reservations(
    backend, tmp_path, request
):
    address = request.getfixturevalue("postgres_dsn") if backend == "postgres" else None

    def open_ledger():
        if backend == "memory":
            return InMemoryBudgetLedger(reservation_ttl_seconds=None)
        if backend == "sqlite":
            return SQLiteBudgetLedger(tmp_path / "inventory.sqlite", reservation_ttl_seconds=None)
        return PostgresBudgetLedger(
            address, schema_mode=SchemaMode.CREATE, reservation_ttl_seconds=None
        )

    ledger = open_ledger()
    assert ledger._supports_producer_budget_readback()
    prefix = "inventory-" + uuid4().hex
    expected = {}
    try:
        binding = dict(binding_id=prefix, authority_digest="a" * 64, allowance=3)
        for _ in range(2):
            with pytest.raises(LookupError):
                await ledger._require_registered_budget_binding(**binding)
        await ledger.register_budget_binding(**binding)
        await ledger._require_registered_budget_binding(**binding)
        for changes in ({"authority_digest": "b" * 64}, {"allowance": 4}):
            with pytest.raises(BudgetBindingRegistrationConflict):
                await ledger._require_registered_budget_binding(**(binding | changes))
        with pytest.raises(ValueError):
            await ledger._require_registered_budget_binding(**(binding | {"allowance": True}))
        for number in range(4):
            result = await ledger.reserve(
                reservation_id=f"{prefix}-{number}",
                limit=_limit().model_copy(update={"key": prefix}),
                session_id=prefix if number < 3 else prefix + ":other",
                agent_name="producer",
                provider_name="provider",
                model="model",
                model_attempt_identity=model_attempt_identity(),
                settlement_event_payload={"interaction_id": prefix},
            )
            assert result.accepted and result.record is not None
            identifier = result.record.reservation_id
            if number == 1:
                await ledger.release(reservation_id=identifier, reason="no dispatch")
            elif number == 2:
                await ledger.mark_dispatched(reservation_ids=(identifier,), dispatch_id=prefix)
                await ledger.reconcile(reservation_id=identifier, actual_amount=Decimal("0.000001"))
            if number < 3:
                expected[identifier] = await ledger.load_reservation(identifier)
        if backend != "memory":
            await ledger.close()
            ledger = open_ledger()
        await ledger._require_registered_budget_binding(**binding)
        after = None
        found = {}
        while page := await ledger._scan_reservation_records(
            session_id=prefix, after=after, limit=2
        ):
            assert len(page) <= 2
            for record in page:
                assert after is None or record.reservation_id > after
                assert record.session_id == prefix
                found[record.reservation_id] = record
                after = record.reservation_id
        assert found == expected
        assert {record.status for record in found.values()} == {"active", "released", "reconciled"}
        for limit in (True, 0, 129):
            with pytest.raises(ValueError):
                await ledger._scan_reservation_records(session_id=prefix, limit=limit)
        with pytest.raises(TypeError):
            await ledger._scan_reservation_records(session_id=prefix, after=True)
        assert await ledger._scan_reservation_records(session_id=prefix + ":missing") == ()
    finally:
        if backend != "memory":
            await ledger.close()

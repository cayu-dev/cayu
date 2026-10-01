from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest

from cayu import Event, EventType, Message, RunRequest
from cayu.budgets.reported import ReportedCostObservation, reported_cost_observation
from cayu.sessions.base import (
    InMemorySessionStore,
    SessionAggregateFilter,
    SessionIdentity,
    UsageRollupQuery,
)
from cayu.storage.sqlite import SQLiteSessionStore

pytestmark = pytest.mark.anyio
NOW = datetime(2026, 1, 1, tzinfo=UTC)


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
@pytest.mark.parametrize("count", [0, 99, 100, 101])
async def test_reported_costs_are_bounded_filtered_and_reconstructable(
    backend, count, tmp_path, request
):
    sid = "reported-" + uuid4().hex
    if backend == "memory":
        store = InMemorySessionStore()

        def factory():
            return store
    elif backend == "sqlite":

        def factory():
            return SQLiteSessionStore(tmp_path / "costs.sqlite")

        store = factory()
    else:
        from cayu.storage.migrations import SchemaMode
        from cayu.storage.postgres import PostgresSessionStore

        dsn = request.getfixturevalue("postgres_dsn")

        def factory():
            return PostgresSessionStore(dsn, schema_mode=SchemaMode.CREATE)

        store = factory()
    query = UsageRollupQuery(
        start_at=NOW,
        end_at=NOW + timedelta(days=1),
        sessions=SessionAggregateFilter(labels={"report": sid}),
    )
    try:
        await store.create(
            RunRequest(
                session_id=sid,
                agent_name="assistant",
                labels={"report": sid},
                messages=[Message.text("user", "hello")],
            ),
            identity=SessionIdentity(provider_name="cayu_gateway", model="example/model"),
        )
        events = [
            Event(
                id=f"event-{i:03}",
                type=EventType.MODEL_COMPLETED,
                session_id=sid,
                timestamp=NOW + timedelta(seconds=i),
                payload={
                    "provider_name": "cayu_gateway",
                    "model": "example/model",
                    "id": "same-request",
                    "usage": {
                        "cost": "0.000000000" if i % 2 else None,
                        "cost_status": "reported" if i % 2 else "pending",
                        "cost_currency": "USD",
                    },
                },
            )
            for i in range(count)
        ]
        await store.append_events(sid, events)
        result = await store.aggregate_usage(query)
        assert result.reported_costs is not None
        rows = result.reported_costs.records
        assert len(rows) == min(count, 100)
        assert result.reported_costs.truncated is (count > 100)
        assert [row.event_id for row in rows] == [e.id for e in reversed(events)][:100]
        assert all(
            row.cost == ("0.000000000" if row.status == "reported" else None) for row in rows
        )
        # A repeated provider request ID remains separate event observations,
        # never a blindly summed bill or extra local budget consumption.
        assert all(row.request_id == "same-request" for row in rows)
        if backend != "memory":
            await store.close()
            store = factory()
        assert (await store.aggregate_usage(query)).reported_costs == result.reported_costs
        empty = query.model_copy(update={"start_at": NOW + timedelta(hours=1)})
        assert not (await store.aggregate_usage(empty)).reported_costs.records
        malformed = Event(
            type=EventType.MODEL_COMPLETED,
            session_id=sid,
            timestamp=NOW + timedelta(hours=2),
            payload={"usage": {"cost": False, "cost_status": "pending", "cost_currency": "USD"}},
        )
        await store.append_event(sid, malformed)
        invalid = (await store.aggregate_usage(query)).reported_costs.records[0]
        assert invalid.event_id == malformed.id
        assert invalid.status == "unavailable" and invalid.cost is None
    finally:
        await store.delete_session(sid)
        if backend != "memory":
            await store.close()


@pytest.mark.parametrize(
    "cost,status,currency,expected",
    [
        ("0.000000001", "reported", "USD", "reported"),
        ("0.000000000", "reported", "USD", "reported"),
        (None, "pending", "USD", "pending"),
        (None, "unavailable", "USD", "unavailable"),
        (True, "reported", "USD", "unavailable"),
        (0, "reported", "USD", "unavailable"),
        ("NaN", "reported", "USD", "unavailable"),
        ("-1.000000000", "reported", "USD", "unavailable"),
        ("0.000000000\n", "reported", "USD", "unavailable"),
        ("0.000000001", "reported", "EUR", "unavailable"),
        ("0.000000001", "future-status", "USD", "unavailable"),
        ("0.000000001", "pending", "USD", "unavailable"),
    ],
)
async def test_reported_cost_validation_never_turns_unknown_into_zero(
    cost, status, currency, expected
):
    row = reported_cost_observation(
        session_id="session",
        event_id="event",
        timestamp=NOW,
        values={"cost": cost, "status": status, "currency": currency},
    )
    assert row.status == expected
    assert row.cost == (cost if expected == "reported" else None)


@pytest.mark.parametrize(
    "changes",
    [
        {"cost": "0.000000000", "status": "pending"},
        {"cost": "NaN", "status": "reported"},
        {"cost": "0.000000000", "currency": None, "status": "reported"},
        {"timestamp": NOW.replace(tzinfo=None)},
    ],
)
async def test_custom_report_projection_rejects_inconsistent_observations(changes):
    values = dict(
        session_id="session",
        event_id="event",
        timestamp=NOW,
        request_id=None,
        provider_name=None,
        model=None,
        cost=None,
        currency="USD",
        status="pending",
    )
    with pytest.raises(ValueError):
        ReportedCostObservation.model_validate(values | changes)

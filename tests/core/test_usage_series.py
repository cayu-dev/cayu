"""Event-window series through the native aggregate store boundary."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import pytest

from cayu.events import Event, EventType
from cayu.messages import Message
from cayu.sessions.memory import InMemorySessionStore
from cayu.sessions.records import SessionIdentity
from cayu.sessions.requests import RunRequest
from cayu.sessions.usage import UsageRollupQuery
from cayu.storage.sqlite import SQLiteSessionStore


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_series_partitions_activity_without_summing_distinct_sessions(backend, tmp_path):
    async def run():
        store = (
            InMemorySessionStore()
            if backend == "memory"
            else SQLiteSessionStore(tmp_path / "series.sqlite")
        )
        start = datetime(2026, 10, 1, tzinfo=UTC)
        try:
            await store.create(
                RunRequest(
                    session_id="older",
                    agent_name="writer",
                    messages=[Message.text("user", "hello")],
                ),
                identity=SessionIdentity(provider_name="configured", model="configured-model"),
            )
            for i, tokens in enumerate((10, 20, 40)):
                await store.append_event(
                    "older",
                    Event(
                        id=f"e{i}",
                        session_id="older",
                        type=EventType.MODEL_COMPLETED,
                        timestamp=start + timedelta(days=i),
                        payload={
                            "usage_metrics": {
                                "provider_name": "actual",
                                "model": "actual-model",
                                "input_tokens": tokens,
                                "output_tokens": 0,
                                "total_tokens": tokens,
                            }
                        },
                    ),
                )
            result = await store.aggregate_usage(
                UsageRollupQuery(
                    start_at=start,
                    end_at=start + timedelta(days=2),
                    series={"unit": "day", "timezone": "UTC"},
                )
            )
            assert result.totals.session_count == 1
            assert result.totals.usage.total_tokens == 30
            assert result.series is not None
            assert [b.totals.session_count for b in result.series.buckets] == [1, 1]
            assert [b.totals.usage.total_tokens for b in result.series.buckets] == [10, 20]
            assert [b.start_at for b in result.series.buckets] == [start, start + timedelta(days=1)]
            assert result.series.buckets[-1].end_at == result.end_at
        finally:
            if isinstance(store, SQLiteSessionStore):
                await store.close()

    asyncio.run(run())


@pytest.mark.parametrize(
    "zone,start,end,hours",
    [
        ("America/New_York", "2026-03-08T05:00:00+00:00", "2026-03-10T04:00:00+00:00", [23, 24]),
        ("America/New_York", "2026-11-01T04:00:00+00:00", "2026-11-03T05:00:00+00:00", [25, 24]),
        ("Asia/Kathmandu", "2026-10-01T18:15:00+00:00", "2026-10-03T18:15:00+00:00", [24, 24]),
    ],
)
def test_day_series_uses_calendar_boundaries(zone, start, end, hours):
    from cayu._time_buckets import TimeSeriesSpec, time_buckets

    buckets = time_buckets(
        datetime.fromisoformat(start),
        datetime.fromisoformat(end),
        TimeSeriesSpec(unit="day", timezone=zone),
    )
    assert [(b.end_at - b.start_at).total_seconds() / 3600 for b in buckets] == hours


def test_hour_fold_clipping_week_and_limits():
    from cayu._time_buckets import TimeSeriesSpec, time_buckets

    start = datetime(2026, 11, 1, 4, 30, tzinfo=UTC)
    buckets = time_buckets(
        start, start + timedelta(hours=4), TimeSeriesSpec(unit="hour", timezone="America/New_York")
    )
    assert len(buckets) == 5
    assert buckets[0].start_at == start
    assert [b.end_at - b.start_at for b in buckets] == [
        timedelta(minutes=30),
        timedelta(hours=1),
        timedelta(hours=1),
        timedelta(hours=1),
        timedelta(minutes=30),
    ]
    aligned = datetime(2026, 1, 1, tzinfo=UTC)
    assert (
        len(time_buckets(aligned, aligned + timedelta(hours=512), TimeSeriesSpec(unit="hour")))
        == 512
    )
    with pytest.raises(ValueError, match="512"):
        time_buckets(aligned, aligned + timedelta(hours=513), TimeSeriesSpec(unit="hour"))
    weekly = time_buckets(aligned, aligned + timedelta(days=16), TimeSeriesSpec(unit="week"))
    assert [b.end_at - b.start_at for b in weekly] == [
        timedelta(days=7),
        timedelta(days=7),
        timedelta(days=2),
    ]


@pytest.mark.parametrize(
    "series",
    [
        {"unit": True},
        {"unit": "month"},
        {"unit": "day", "timezone": "../secret"},
        {"unit": "day", "timezone": "No/SuchZone"},
    ],
)
def test_invalid_series_is_rejected_before_store_work(series):
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        UsageRollupQuery(
            start_at=datetime(2026, 1, 1, tzinfo=UTC),
            end_at=datetime(2026, 1, 2, tzinfo=UTC),
            series=series,
        )


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_empty_series_and_custom_result_query_validation(backend, tmp_path):
    from cayu.budgets.aggregates import UsageRollupInconsistent

    async def run():
        store = (
            InMemorySessionStore()
            if backend == "memory"
            else SQLiteSessionStore(tmp_path / "empty.sqlite")
        )
        try:
            query = UsageRollupQuery(
                start_at=datetime(2026, 1, 1, tzinfo=UTC),
                end_at=datetime(2026, 1, 3, tzinfo=UTC),
                series={"unit": "day"},
            )
            result = await store.aggregate_usage(query)
            result.validate_for_query(query)
            assert result.series is not None and len(result.series.buckets) == 2
            assert all(
                b.totals.model_steps == b.totals.session_count == b.totals.usage.total_tokens == 0
                for b in result.series.buckets
            )
            with pytest.raises(UsageRollupInconsistent):
                result.model_copy(update={"series": None}).validate_for_query(query)
            with pytest.raises(UsageRollupInconsistent):
                result.validate_for_query(query.model_copy(update={"series": None}))
            wrong = result.series.model_copy(update={"timezone": "Asia/Bishkek"})
            with pytest.raises(UsageRollupInconsistent):
                result.model_copy(update={"series": wrong}).validate_for_query(query)
        finally:
            if isinstance(store, SQLiteSessionStore):
                await store.close()

    asyncio.run(run())


def test_postgres_series_matches_memory(postgres_dsn):
    """CI-only persistent backend parity; isolate against shared fixture rows."""
    from uuid import uuid4

    from cayu.sessions.queries import SessionAggregateFilter
    from cayu.storage.migrations import SchemaMode
    from cayu.storage.postgres import PostgresSessionStore

    async def run():
        memory = InMemorySessionStore()
        postgres = PostgresSessionStore(
            postgres_dsn, min_size=1, max_size=2, schema_mode=SchemaMode.CREATE
        )
        scope = uuid4().hex
        start = datetime(2026, 10, 1, tzinfo=UTC)
        try:
            for store in (memory, postgres):
                await store.create(
                    RunRequest(
                        session_id=scope,
                        agent_name="writer",
                        messages=[Message.text("user", "hello")],
                        labels={"series-scope": scope},
                    ),
                    identity=SessionIdentity(provider_name="configured", model="configured-model"),
                )
                for i in (0, 1, 3):
                    await store.append_event(
                        scope,
                        Event(
                            id=f"{scope}-{i}",
                            session_id=scope,
                            type=EventType.MODEL_COMPLETED,
                            timestamp=start + timedelta(days=i),
                            payload={
                                "usage_metrics": {
                                    "input_tokens": 20,
                                    "output_tokens": 3,
                                    "total_tokens": 23,
                                }
                            },
                        ),
                    )
            query = UsageRollupQuery(
                start_at=start,
                end_at=start + timedelta(days=3),
                sessions=SessionAggregateFilter(labels={"series-scope": scope}),
                series={"unit": "day"},
            )
            actual = await postgres.aggregate_usage(query)
            expected = await memory.aggregate_usage(query)
            actual.validate_for_query(query)
            assert actual.series == expected.series
            assert actual.totals == expected.totals
        finally:
            await postgres.close()

    asyncio.run(run())


def test_sqlite_many_buckets_use_one_additional_aggregate_scan(tmp_path, monkeypatch):
    from cayu.storage import _sqlite_aggregates

    original = _sqlite_aggregates.usage_series_statement
    statements = []

    def observed(**kwargs):
        sql, params = original(**kwargs)
        statements.append((sql, params))
        return sql, params

    monkeypatch.setattr(_sqlite_aggregates, "usage_series_statement", observed)

    async def run():
        store = SQLiteSessionStore(tmp_path / "bounded.sqlite")
        try:
            start = datetime(2026, 1, 1, tzinfo=UTC)
            result = await store.aggregate_usage(
                UsageRollupQuery(
                    start_at=start, end_at=start + timedelta(hours=512), series={"unit": "hour"}
                )
            )
            assert len(result.series.buckets) == 512
            assert len(statements) == 1
            sql, params = statements[0]
            plan = await store._run_read(
                lambda connection: list(connection.execute("EXPLAIN QUERY PLAN " + sql, params))
            )
            event_reads = [
                row[3] for row in plan if "event" in row[3].lower() and "index" in row[3].lower()
            ]
            assert len(event_reads) == 1, plan
            assert "timestamp>?" in event_reads[0] and "timestamp<?" in event_reads[0]
        finally:
            await store.close()

    asyncio.run(run())


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_series_covers_tools_hosted_resources_auxiliary_usage_and_restart(backend, tmp_path):
    from cayu.sessions.queries import SessionAggregateFilter

    async def run():
        path = tmp_path / "families.sqlite"
        store = InMemorySessionStore() if backend == "memory" else SQLiteSessionStore(path)
        start = datetime(2026, 10, 1, tzinfo=UTC)
        try:
            await store.create(
                RunRequest(
                    session_id="families",
                    agent_name="writer",
                    messages=[Message.text("user", "hello")],
                ),
                identity=SessionIdentity(provider_name="configured", model="configured-model"),
            )
            payloads = [
                (
                    EventType.MODEL_COMPLETED,
                    {"usage_metrics": {"input_tokens": 2**53 + 1, "total_tokens": 2**53 + 1}},
                ),
                (
                    EventType.MODEL_AUXILIARY_ATTEMPT_SETTLED,
                    {"usage_metrics": {"input_tokens": 3, "total_tokens": 3}},
                ),
                (EventType.TOOL_CALL_STARTED, {}),
                (
                    EventType.MODEL_HOSTED_TOOL_CALL,
                    {"tool_type": "web_search", "status": "outcome_unknown"},
                ),
            ]
            for i, (event_type, payload) in enumerate(payloads):
                await store.append_event(
                    "families",
                    Event(
                        id=f"f-{i}",
                        session_id="families",
                        type=event_type,
                        timestamp=start + timedelta(hours=i),
                        payload=payload,
                    ),
                )
            query = UsageRollupQuery(
                start_at=start,
                end_at=start + timedelta(hours=5),
                sessions=SessionAggregateFilter(agent_name="writer", provider_name="configured"),
                series={"unit": "hour"},
            )
            result = await store.aggregate_usage(query)
            result.validate_for_query(query)
            assert result.totals.model_steps == 1
            assert result.totals.tool_calls == 1
            assert result.totals.usage.total_tokens == 2**53 + 4
            assert result.totals.usage.hosted_tools.web_search_outcome_unknown == 1
            assert [b.totals.model_steps for b in result.series.buckets] == [1, 0, 0, 0, 0]
            assert [b.totals.tool_calls for b in result.series.buckets] == [0, 0, 1, 0, 0]
            assert result.series.model_dump(mode="json")["buckets"][0]["totals"]["usage"][
                "total_tokens"
            ] == str(2**53 + 1)
            if backend == "sqlite":
                await store.close()
                store = SQLiteSessionStore(path)
                assert (await store.aggregate_usage(query)).series == result.series
        finally:
            if isinstance(store, SQLiteSessionStore):
                await store.close()

    asyncio.run(run())


def test_cancelling_native_series_while_waiting_for_owner_publishes_nothing():
    async def run():
        store = InMemorySessionStore()
        query = UsageRollupQuery(
            start_at=datetime(2026, 1, 1, tzinfo=UTC),
            end_at=datetime(2026, 1, 2, tzinfo=UTC),
            series={"unit": "hour"},
        )
        async with store._read_session_query_state():
            task = asyncio.create_task(store.aggregate_usage(query))
            await asyncio.sleep(0)
            task.cancel()
            assert task.cancelling() == 1
            with pytest.raises(asyncio.CancelledError):
                await task
            assert task.cancelled()
        result = await store.aggregate_usage(query)
        assert len(result.series.buckets) == 24

    asyncio.run(run())


def test_skipped_calendar_date_and_datetime_extremes():
    from cayu._time_buckets import TimeSeriesSpec, time_buckets

    buckets = time_buckets(
        datetime(2011, 12, 29, 10, tzinfo=UTC),
        datetime(2011, 12, 31, 10, tzinfo=UTC),
        TimeSeriesSpec(unit="day", timezone="Pacific/Apia"),
    )
    assert len(buckets) == 2
    assert all(b.end_at - b.start_at == timedelta(days=1) for b in buckets)
    with pytest.raises(ValueError, match="represented"):
        time_buckets(
            datetime.min.replace(tzinfo=UTC),
            datetime(1, 1, 2, tzinfo=UTC),
            TimeSeriesSpec(unit="day", timezone="America/New_York"),
        )
    maximum = datetime.max.replace(tzinfo=UTC)
    clipped = time_buckets(maximum - timedelta(hours=1), maximum, TimeSeriesSpec(unit="day"))
    assert clipped[-1].end_at == maximum


async def _append_hosted_progress_and_terminal(store, scope, terminal_status):
    """Use the ordinary runtime-supported progress/terminal event payload shape."""
    start = datetime(2026, 10, 1, tzinfo=UTC)
    await store.create(
        RunRequest(
            session_id=scope,
            agent_name="writer",
            messages=[Message.text("user", "search")],
            labels={"hosted-scope": scope},
        ),
        identity=SessionIdentity(provider_name="configured", model="configured-model"),
    )
    for i, status in enumerate(("in_progress", "searching", terminal_status)):
        await store.append_event(
            scope,
            Event(
                id=f"{scope}-{i}",
                session_id=scope,
                type=EventType.MODEL_HOSTED_TOOL_CALL,
                timestamp=start + timedelta(minutes=10 * i if i < 2 else 70),
                payload={
                    "tool_type": "web_search",
                    "call_id": "search-call",
                    "status": status,
                    "provider_name": "actual",
                    "model": "actual-model",
                },
            ),
        )
    for event_type, payload in (
        (
            EventType.MODEL_COMPLETED,
            {
                "usage_metrics": {
                    "provider_name": "actual",
                    "model": "actual-model",
                    "input_tokens": 5,
                    "output_tokens": 2,
                    "total_tokens": 7,
                }
            },
        ),
        (EventType.TOOL_CALL_STARTED, {}),
    ):
        await store.append_event(
            scope,
            Event(
                id=f"{scope}-{event_type}",
                session_id=scope,
                type=event_type,
                timestamp=start + timedelta(minutes=80),
                payload=payload,
            ),
        )
    return start


async def _assert_hosted_progress_excluded(store, terminal_status, *, with_series):
    from uuid import uuid4

    from cayu.sessions.queries import SessionAggregateFilter

    scope = uuid4().hex
    start = await _append_hosted_progress_and_terminal(store, scope, terminal_status)
    for hours, session_limit in ((1, None), (1, 10), (2, None), (2, 10)):
        query = UsageRollupQuery(
            start_at=start,
            end_at=start + timedelta(hours=hours),
            sessions=SessionAggregateFilter(labels={"hosted-scope": scope}),
            session_group_limit=session_limit,
            include_pricing_inputs=True,
            **({"series": {"unit": "hour"}} if with_series else {}),
        )
        result = await store.aggregate_usage(query)
        result.validate_for_query(query)
        has_terminal = int(hours == 2)
        assert result.totals.session_count == has_terminal
        assert result.totals.model_steps == has_terminal
        assert result.totals.tool_calls == has_terminal
        assert result.totals.usage.total_tokens == 7 * has_terminal
        resources = result.totals.usage.hosted_tools
        assert resources.web_search_calls == has_terminal * (terminal_status != "outcome_unknown")
        assert resources.web_search_outcome_unknown == has_terminal * (
            terminal_status == "outcome_unknown"
        )
        if session_limit is None:
            assert result.session_breakdown is None
            assert result.session_pricing_inputs == ()
        else:
            assert result.session_breakdown is not None
            assert result.session_breakdown.groups[0].totals == result.totals
            assert len(result.session_pricing_inputs) == 2 * has_terminal
        for breakdown in (result.provider_breakdown, result.model_breakdown):
            assert len(breakdown.groups) == has_terminal
            if has_terminal:
                assert breakdown.groups[0].totals.session_count == 1
                assert breakdown.groups[0].totals.usage == result.totals.usage
        assert len(result.pricing_inputs) == 2 * has_terminal
        if with_series:
            assert result.series is not None
            assert [b.totals.session_count for b in result.series.buckets] == list(range(hours))
            assert result.series.buckets[0].totals.usage.total_tokens == 0
            assert result.series.buckets[-1].totals == result.totals


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize(
    "terminal_status", ["completed", "incomplete", "failed", "outcome_unknown"]
)
@pytest.mark.parametrize("with_series", [False, True])
def test_hosted_progress_is_not_usage_activity(backend, terminal_status, with_series, tmp_path):
    async def run():
        store = (
            InMemorySessionStore()
            if backend == "memory"
            else SQLiteSessionStore(tmp_path / "hosted.sqlite")
        )
        try:
            await _assert_hosted_progress_excluded(store, terminal_status, with_series=with_series)
        finally:
            if isinstance(store, SQLiteSessionStore):
                await store.close()

    asyncio.run(run())


def test_postgres_hosted_progress_is_not_usage_activity(postgres_dsn):
    from cayu.storage.migrations import SchemaMode
    from cayu.storage.postgres import PostgresSessionStore

    async def run():
        store = PostgresSessionStore(
            postgres_dsn, min_size=1, max_size=2, schema_mode=SchemaMode.CREATE
        )
        try:
            for status in ("completed", "incomplete", "failed", "outcome_unknown"):
                for with_series in (False, True):
                    await _assert_hosted_progress_excluded(store, status, with_series=with_series)
        finally:
            await store.close()

    asyncio.run(run())


@pytest.mark.parametrize("hours", [1, 2, 512])
def test_postgres_series_statement_has_valid_cte_boundary(hours):
    """Catch malformed CTE composition locally; native execution remains in PG CI."""
    import re

    from cayu.storage._postgres_aggregates import usage_series_statement
    from cayu.storage._session_store_sql import SessionQuerySqlPlan

    start = datetime(2026, 1, 1, tzinfo=UTC)
    plan = SessionQuerySqlPlan("", (), "", (), "", "")
    sql, params = usage_series_statement(
        session_plan=plan,
        query=UsageRollupQuery(
            start_at=start, end_at=start + timedelta(hours=hours), series={"unit": "hour"}
        ),
    )
    assert re.search(r"\),\s*SELECT\s+bucket_index", sql) is None
    assert sql.count("%s") == len(params)
    assert len(params) == 4 + hours - 1

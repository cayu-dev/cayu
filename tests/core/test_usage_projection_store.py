"""Stored usage and tool totals equal a recompute from events.

``cayu_session_usage``, ``cayu_usage_hourly`` and ``cayu_tool_usage`` are kept by
triggers. These tests generate random histories on SQLite and Postgres, change
them the ways retention and repair do (whole-session deletion, single-event
deletion, payload rewrites), and check that every rollup and tool total read
from the stored totals equals the event-level reference, for random windows,
filters, series and price books. They also check the migration's backfill,
saturated counters, price groups that span a tier, and the indexed tool-call
listing.
"""

from __future__ import annotations

import json
import random
import sqlite3
from datetime import UTC, datetime, timedelta
from typing import LiteralString, cast

import pytest

from cayu import Event, EventType, Message, RunRequest
from cayu._time_buckets import TimeSeriesSpec
from cayu.budgets.pricing import default_price_book
from cayu.sessions.queries import SessionAggregateFilter
from cayu.sessions.records import SessionIdentity
from cayu.sessions.tool_usage import ToolUsageReader
from cayu.sessions.usage import UsageRollupQuery
from cayu.sessions.usage_recompute import recompute_usage
from cayu.storage._usage_projection_schema import (
    POSTGRES_PENDING_TABLES,
    PROJECTION_TABLES,
    USAGE_PROJECTION_REVISION,
)
from cayu.storage.migrations import SchemaMode
from cayu.storage.sqlite import SQLiteSessionStore

pytestmark = pytest.mark.anyio
BASE = datetime(2026, 3, 1, 0, tzinfo=UTC)
MODELS = (("anthropic", "claude-x"), ("anthropic", "claude-y"), ("openai", "gpt-x"))
TOOLS = ("exec_command", "read_file", "write_file", "rare_tool")


class _Backend:
    def __init__(self, name: str, tmp_path, request) -> None:
        self.name = name
        if name == "sqlite":
            self.path = tmp_path / "usage-projection.sqlite"
        else:
            self.dsn = request.getfixturevalue("postgres_dsn")

    async def open(self, schema_mode: SchemaMode = SchemaMode.CREATE):
        if self.name == "sqlite":
            return SQLiteSessionStore(self.path, schema_mode=schema_mode)
        from cayu.storage.postgres import PostgresSessionStore

        store = PostgresSessionStore(self.dsn, schema_mode=schema_mode)
        try:
            await store.ensure_schema()
        except BaseException:
            await store.close()
            raise
        return store

    async def execute(self, sql: str, params: tuple = ()) -> list[tuple]:
        if self.name == "sqlite":
            connection = sqlite3.connect(self.path)
            try:
                rows = [tuple(row) for row in connection.execute(sql, params).fetchall()]
                connection.commit()
                return rows
            finally:
                connection.close()
        import psycopg

        async with await psycopg.AsyncConnection.connect(self.dsn, autocommit=True) as conn:
            cursor = await conn.execute(cast("LiteralString", sql.replace("?", "%s")), params)
            return [tuple(row) for row in await cursor.fetchall()] if cursor.description else []


@pytest.fixture(params=["sqlite", "postgres"])
def backend(request, tmp_path) -> _Backend:
    return _Backend(request.param, tmp_path, request)


def _metrics(rng: random.Random, provider: str, model: str) -> dict:
    input_tokens = rng.choice((0, 7, 1200, 40_000, 150_000, 210_000, 300_000))
    output = rng.randrange(0, 5000)
    metrics: dict = {
        "provider_name": provider,
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output,
        "total_tokens": input_tokens + output,
    }
    if rng.random() < 0.6:
        metrics["cache"] = {
            "read_tokens": rng.randrange(0, 900),
            "write_tokens": rng.randrange(0, 50),
        }
    if rng.random() < 0.3:
        metrics["reasoning_output_tokens"] = rng.randrange(0, 400)
    if rng.random() < 0.2:
        metrics["billing_identity"] = {
            "provider_name": provider,
            "model": model,
            "service_tier": rng.choice(("standard", "priority")),
        }
    if rng.random() < 0.05:
        metrics["output_tokens"] = "not a count"
    return metrics


def _history(rng: random.Random, session_id: str, start: datetime) -> list[Event]:
    events: list[Event] = []
    at = start
    agent = rng.choice(("assistant", "reviewer", None))
    for index in range(rng.randrange(1, 40)):
        at += timedelta(minutes=rng.choice((1, 3, 17, 45, 95)), seconds=rng.randrange(60))
        provider, model = rng.choice(MODELS)
        kind = rng.random()
        payload: dict
        tool_name = None
        if kind < 0.35:
            event_type = EventType.MODEL_COMPLETED
            payload = {"usage_metrics": _metrics(rng, provider, model), "transcript_cursor": index}
            if rng.random() < 0.1:
                payload["usage_normalization_failed"] = True
            if rng.random() < 0.15:
                payload["usage"] = {
                    "cost": "0.01",
                    "cost_currency": "USD",
                    "cost_status": "reported",
                }
            if rng.random() < 0.05:
                payload = {"usage": {"input_tokens": 3}}
        elif kind < 0.45:
            event_type = EventType.MODEL_AUXILIARY_ATTEMPT_SETTLED
            payload = {"usage_metrics": _metrics(rng, provider, model)}
        elif kind < 0.55:
            event_type = EventType.MODEL_HOSTED_TOOL_CALL
            payload = {
                "tool_type": rng.choice(("web_search", "web_search", "file_search")),
                "status": rng.choice(("completed", "failed", "outcome_unknown", "in_progress")),
                "provider_name": "openai",
                "model": "gpt-x",
            }
        else:
            tool_name = rng.choice(TOOLS[:3]) if rng.random() < 0.95 else TOOLS[3]
            event_type = rng.choice(
                (
                    EventType.TOOL_CALL_STARTED,
                    EventType.TOOL_CALL_COMPLETED,
                    EventType.TOOL_CALL_FAILED,
                    EventType.TOOL_CALL_APPROVED,
                    EventType.TOOL_CALL_APPROVAL_DENIED,
                )
            )
            payload = {"interrupted": True} if rng.random() < 0.1 else {}
        events.append(
            Event(
                id=f"{session_id}-{index}",
                session_id=session_id,
                type=event_type,
                timestamp=at,
                agent_name=agent,
                environment_name=rng.choice(("prod", None)),
                tool_name=tool_name,
                payload=payload,
            )
        )
    return events


async def _populate(store, rng: random.Random, prefix: str, sessions: int) -> list[str]:
    ids = []
    for number in range(sessions):
        session_id = f"{prefix}-{number}"
        await store.create(
            RunRequest(
                session_id=session_id,
                agent_name=rng.choice(("assistant", "reviewer")),
                labels={"team": rng.choice(("a", "b")), "run": prefix},
                messages=[Message.text("user", "hello")],
            ),
            identity=SessionIdentity(provider_name="anthropic", model="claude-x"),
        )
        for event in _history(rng, session_id, BASE + timedelta(hours=rng.randrange(0, 60))):
            await store.append_event(session_id, event)
        ids.append(session_id)
    return ids


async def _mutate(backend: _Backend, store, rng: random.Random, ids: list[str]) -> None:
    """Delete sessions and events and rewrite payloads, as retention and repair do."""

    for session_id in rng.sample(ids, k=max(1, len(ids) // 6)):
        await store.delete_session(session_id)
        ids.remove(session_id)
    rows = await backend.execute(
        "SELECT sequence, event_type FROM cayu_events WHERE session_id IN ("
        + ", ".join("?" for _ in ids)
        + ")",
        tuple(ids),
    )
    for sequence, _ in rng.sample(rows, k=min(len(rows), 15)):
        await backend.execute("DELETE FROM cayu_events WHERE sequence = ?", (sequence,))
    remaining = await backend.execute(
        "SELECT sequence FROM cayu_events WHERE event_type = 'model.completed'"
    )
    for (sequence,) in rng.sample(remaining, k=min(len(remaining), 6)):
        metrics = _metrics(rng, *rng.choice(MODELS))
        payload = json.dumps({"usage_metrics": metrics})
        if backend.name == "sqlite":
            await backend.execute(
                "UPDATE cayu_events SET payload_json = ? WHERE sequence = ?", (payload, sequence)
            )
        else:
            # Postgres keeps the payload in both columns; accounting reads ``event``.
            await backend.execute(
                "UPDATE cayu_events SET payload = ?::jsonb, "
                "event = jsonb_set(event, '{payload}', ?::jsonb) WHERE sequence = ?",
                (payload, payload, sequence),
            )


def _queries(rng: random.Random, prefix: str) -> list[UsageRollupQuery]:
    book = default_price_book()
    boundaries = book.input_token_tier_boundaries()
    queries = []
    for _ in range(14):
        start = BASE + timedelta(minutes=rng.randrange(-120, 60 * 60))
        if rng.random() < 0.4:
            start = start.replace(minute=0, second=0, microsecond=0)
        end = start + timedelta(
            minutes=rng.choice((7, 60, 240, 1440, 4000)), seconds=rng.randrange(60)
        )
        session_filter = rng.choice(
            (
                # Every session: the hourly totals answer the sums.
                SessionAggregateFilter(),
                SessionAggregateFilter(labels={"run": prefix}),
                SessionAggregateFilter(labels={"run": prefix, "team": "a"}),
                SessionAggregateFilter(labels={"run": prefix}, agent_name="reviewer"),
            )
        )
        series = rng.choice(
            (
                None,
                TimeSeriesSpec(unit="hour", timezone="UTC"),
                TimeSeriesSpec(unit="day", timezone="America/Los_Angeles"),
                TimeSeriesSpec(unit="hour", timezone="Asia/Kolkata"),
            )
        )
        if series is not None and series.unit == "hour" and end - start > timedelta(days=2):
            series = TimeSeriesSpec(unit="day", timezone=series.timezone)
        queries.append(
            UsageRollupQuery(
                start_at=start,
                end_at=end,
                sessions=session_filter,
                series=series,
                group_limit=rng.choice((1, 2, 20)),
                session_group_limit=rng.choice((None, 1, 3, 50)),
                include_pricing_inputs=rng.random() < 0.8,
                pricing_tier_boundaries=rng.choice((boundaries, (150_000,), ())),
            )
        )
    return queries


async def _assert_stored_equals_recompute(store, queries: list[UsageRollupQuery]) -> None:
    for query in queries:
        report = await recompute_usage(store, query)
        assert report.matches, (query, report.differences)


def _summary_differences(stored: dict, reference: dict) -> dict[str, object]:
    def leaves(value: object, prefix: str = "") -> dict[str, object]:
        if isinstance(value, dict):
            found: dict[str, object] = {}
            for key, item in value.items():
                found.update(leaves(item, f"{prefix}{key}."))
            return found
        return {prefix.rstrip("."): value}

    differences: dict[str, object] = {}
    for session_id in sorted(stored.keys() | reference.keys()):
        left = leaves(stored[session_id].model_dump()) if session_id in stored else {}
        right = leaves(reference[session_id].model_dump()) if session_id in reference else {}
        changed = {
            path: (left.get(path), right.get(path))
            for path in left.keys() | right.keys()
            if left.get(path) != right.get(path)
        }
        if changed:
            differences[session_id] = changed
    return differences


async def _assert_session_totals_equal_reference(
    backend: _Backend, store, rng: random.Random, ids: list[str]
) -> None:
    from cayu.budgets.usage import SessionUsageSummary
    from cayu.sessions.store import SessionStore

    def nonempty(summaries: dict) -> dict:
        # The reducer also lists a session whose only usage-bearing events are
        # hosted calls still in progress, with nothing counted.
        return {
            key: value
            for key, value in summaries.items()
            if value != SessionUsageSummary(session_id=key)
        }

    windows = [(None, None)]
    for _ in range(4):
        start = BASE + timedelta(minutes=rng.randrange(0, 60 * 60))
        windows.append((start, start + timedelta(minutes=rng.choice((13, 120, 2000)))))
    for since, until in windows:
        stored = await store.read_session_usage_totals(ids, since=since, until=until)
        # A fresh store: the accounting cache is invalidated by deletions only, and
        # the test also rewrites payloads with SQL.
        fresh = await backend.open()
        try:
            reference = await SessionStore.read_session_usage_totals(
                fresh, ids, since=since, until=until
            )
        finally:
            await fresh.close()
        assert nonempty(stored) == nonempty(reference), (
            since,
            until,
            _summary_differences(nonempty(stored), nonempty(reference)),
        )


@pytest.mark.parametrize("seed", [3, 11, 29])
async def test_stored_totals_equal_a_recompute_of_random_histories(backend, seed) -> None:
    rng = random.Random(seed)
    prefix = f"p{seed}-{backend.name}"
    store = await backend.open()
    try:
        ids = await _populate(store, rng, prefix, sessions=12)
        await _assert_stored_equals_recompute(store, _queries(rng, prefix))
        await _assert_session_totals_equal_reference(backend, store, rng, ids)
        await _mutate(backend, store, rng, ids)
        await _assert_stored_equals_recompute(store, _queries(rng, prefix))
        await _assert_session_totals_equal_reference(backend, store, rng, ids)
        for session_id in ids:
            await store.delete_session(session_id)
        assert await backend.execute(
            "SELECT COUNT(*) FROM cayu_session_usage WHERE session_id LIKE ?", (prefix + "-%",)
        ) == [(0,)]
    finally:
        await store.close()


async def test_migration_fills_stored_totals_from_retained_events(backend) -> None:
    rng = random.Random(5)
    prefix = f"migrate-{backend.name}"
    store = await backend.open()
    try:
        await _populate(store, rng, prefix, sessions=8)
    finally:
        await store.close()
    for table in (
        *PROJECTION_TABLES,
        *(POSTGRES_PENDING_TABLES if backend.name == "postgres" else ()),
    ):
        await backend.execute(f"DELETE FROM {table}")
    await backend.execute(
        "DELETE FROM cayu_schema_migrations WHERE revision >= ?", (USAGE_PROJECTION_REVISION,)
    )
    store = await backend.open(SchemaMode.MIGRATE)
    try:
        assert (await backend.execute("SELECT COUNT(*) FROM cayu_session_usage"))[0][0] > 0
        await _assert_stored_equals_recompute(store, _queries(rng, prefix))
    finally:
        await store.close()


async def test_saturated_counters_fall_back_to_events(tmp_path) -> None:
    store = SQLiteSessionStore(tmp_path / "saturate.sqlite")
    try:
        await store.create(
            RunRequest(
                session_id="big", agent_name="assistant", messages=[Message.text("user", "x")]
            ),
            identity=SessionIdentity(provider_name="anthropic", model="claude-x"),
        )
        huge = 2**63 - 10
        for index in range(2):
            await store.append_event(
                "big",
                Event(
                    id=f"big-{index}",
                    session_id="big",
                    type=EventType.MODEL_COMPLETED,
                    timestamp=BASE + timedelta(minutes=index),
                    payload={
                        "usage_metrics": {
                            "provider_name": "anthropic",
                            "model": "claude-x",
                            "input_tokens": huge,
                            "output_tokens": 0,
                            "total_tokens": huge,
                        }
                    },
                ),
            )
        connection = sqlite3.connect(tmp_path / "saturate.sqlite")
        assert connection.execute("SELECT MAX(overflow) FROM cayu_session_usage").fetchone() == (1,)
        connection.close()
        query = UsageRollupQuery(
            start_at=BASE - timedelta(hours=1),
            end_at=BASE + timedelta(hours=2),
            session_group_limit=5,
            include_pricing_inputs=True,
        )
        result = await store.aggregate_usage(query)
        assert result.totals.usage.input_tokens == 2 * huge
        assert (await recompute_usage(store, query)).matches
    finally:
        await store.close()


async def test_tool_calls_by_name_reach_a_rare_tool_past_common_ones(backend) -> None:
    store = await backend.open()
    try:
        await store.create(
            RunRequest(
                session_id="tools", agent_name="assistant", messages=[Message.text("user", "x")]
            ),
            identity=SessionIdentity(provider_name="anthropic", model="claude-x"),
        )
        events = [
            Event(
                id="rare",
                session_id="tools",
                type=EventType.TOOL_CALL_COMPLETED,
                timestamp=BASE,
                tool_name="seldom_used",
            )
        ] + [
            Event(
                id=f"common-{index}",
                session_id="tools",
                type=EventType.TOOL_CALL_COMPLETED,
                timestamp=BASE + timedelta(seconds=index + 1),
                tool_name="frequent_tool",
                payload={"interrupted": True} if index == 0 else {},
            )
            for index in range(250)
        ]
        await store.append_events("tools", events)
        reader = ToolUsageReader(
            store,
            names={"seldom_used": "seldom_used", "frequent_tool": "frequent_tool"},
            project_event=lambda record: {
                "session_id": record.event.session_id,
                "id": record.event.id,
            },
        )
        rare = await reader.recent(name="seldom_used", cursor=None, limit=20)
        assert [call["event_id"] for call in rare["calls"]] == ["rare"]
        assert rare["next_cursor"] is None
        first = await reader.recent(name="frequent_tool", cursor=None, limit=100)
        assert len(first["calls"]) == 100
        calls = list(first["calls"])
        cursor = first["next_cursor"]
        while cursor is not None:
            page = await reader.recent(name="frequent_tool", cursor=cursor, limit=100)
            calls.extend(page["calls"])
            cursor = page["next_cursor"]
        assert [call["event_id"] for call in calls] == [
            f"common-{index}" for index in reversed(range(250))
        ]
        assert {call["event_id"] for call in calls if call["outcome"] == "interrupted"} == {
            "common-0"
        }
        newest = await reader.recent(name=None, cursor=None, limit=100)
        assert newest["calls"][0]["event_id"] == "common-249"
        totals = await reader.totals()
        by_name = {row["name"]: row for row in totals["tools"]}
        assert by_name["frequent_tool"]["calls"] == 250
        assert by_name["seldom_used"]["calls"] == 1
    finally:
        await store.close()


async def test_postgres_rollup_ignores_deltas_that_cancel_a_row(postgres_dsn) -> None:
    """A reader whose compaction was skipped sees a row and the deltas cancelling it."""

    import psycopg

    from cayu.storage._usage_projection_schema import _COMPACTION_LOCK
    from cayu.storage.postgres import PostgresSessionStore

    backend = _Backend.__new__(_Backend)
    backend.name, backend.dsn = "postgres", postgres_dsn
    rng = random.Random(7)
    store = await backend.open()
    try:
        ids = await _populate(store, rng, "cancel", sessions=3)
        book = default_price_book()
        query = UsageRollupQuery(
            start_at=BASE - timedelta(hours=1),
            end_at=BASE + timedelta(days=3),
            include_pricing_inputs=True,
            pricing_tier_boundaries=book.input_token_tier_boundaries(),
        )
        await store.aggregate_usage(query)  # folds every delta into the shared rows
        async with await psycopg.AsyncConnection.connect(postgres_dsn) as holder:
            # Another reader is compacting: this one reads the pending deltas.
            await holder.execute("SELECT pg_advisory_xact_lock(%s)", (_COMPACTION_LOCK,))
            await store.delete_session(ids[0])
            pending = await backend.execute("SELECT COUNT(*) FROM cayu_usage_hourly_pending")
            assert pending[0][0] > 0
            stored = await store.aggregate_usage(query)
            reference = await store.aggregate_usage(query, from_events=True)
            await holder.rollback()
        assert stored.model_dump(exclude={"as_of"}) == reference.model_dump(exclude={"as_of"})
    finally:
        await store.close()
    reader = PostgresSessionStore(postgres_dsn, read_only=True)
    try:
        # A read-only store reads the pending deltas without folding them.
        assert (await reader.aggregate_usage(query)).model_dump(
            exclude={"as_of"}
        ) == reference.model_dump(exclude={"as_of"})
        await reader.tool_usage_totals()
    finally:
        await reader.close()


@pytest.mark.parametrize("mutation", ["delete", "rewrite"])
async def test_postgres_concurrent_price_identities_keep_their_event_binding(
    postgres_dsn, mutation
) -> None:
    """An invisible equivalent identity cannot block, or change a later removal's group."""

    from uuid import uuid4

    import psycopg

    from cayu.storage._usage_projection_schema import POSTGRES_USAGE_PROJECTION_REBUILD
    from cayu.storage.postgres import PostgresSessionStore

    dsn = psycopg.conninfo.make_conninfo(postgres_dsn, options="-c lock_timeout=3000")
    backend = _Backend.__new__(_Backend)
    backend.name, backend.dsn = "postgres", dsn
    model = f"concurrent-{uuid4()}"
    store = PostgresSessionStore(dsn, schema_mode=SchemaMode.CREATE)

    def event(session_id, tokens=7):
        return Event(
            type=EventType.MODEL_COMPLETED,
            session_id=session_id,
            timestamp=BASE,
            payload={
                "usage_metrics": {
                    "provider_name": "openai",
                    "model": model,
                    "input_tokens": tokens,
                    "output_tokens": 3,
                    "total_tokens": tokens + 3,
                }
            },
        )

    try:
        sessions = [
            await store.create(
                RunRequest(agent_name="assistant", messages=[]),
                identity=SessionIdentity(provider_name="openai", model=model),
            )
            for _ in range(2)
        ]
        earlier, later = sessions
        async with await psycopg.AsyncConnection.connect(dsn) as held:
            async with held.cursor() as cur:
                await cur.execute("SELECT pg_current_xact_id()")
                await store._append_events_with_cursor(
                    cur, earlier.id, [event(earlier.id)], expected_run_epoch=None
                )
                await store.append_event(later.id, event(later.id))
            await held.commit()
        bindings = dict(
            await backend.execute(
                "SELECT session_id, projection_price_identity_id FROM cayu_usage_events "
                "WHERE session_id IN (?, ?) AND event_type = 'model.completed'",
                (earlier.id, later.id),
            )
        )
        assert set(bindings) == {earlier.id, later.id}
        assert all(identity > 0 for identity in bindings.values())
        assert bindings[earlier.id] != bindings[later.id]
        queries = [
            UsageRollupQuery(
                start_at=BASE - timedelta(hours=1),
                end_at=BASE + timedelta(hours=2),
                include_pricing_inputs=True,
                sessions=selection,
            )
            for selection in (
                SessionAggregateFilter(),
                SessionAggregateFilter(agent_name="assistant"),
            )
        ]
        await _assert_stored_equals_recompute(store, queries)
        if mutation == "delete":
            await backend.execute(
                "DELETE FROM cayu_events WHERE session_id = ? AND event_type = 'model.completed'",
                (later.id,),
            )
        else:
            payload = json.dumps(event(later.id, 11).payload)
            await backend.execute(
                "UPDATE cayu_events SET payload = ?::jsonb, "
                "event = jsonb_set(event, '{payload}', ?::jsonb) "
                "WHERE session_id = ? AND event_type = 'model.completed'",
                (payload, payload, later.id),
            )
        await _assert_stored_equals_recompute(store, queries)
        for statement in POSTGRES_USAGE_PROJECTION_REBUILD:
            await backend.execute(statement)
        await _assert_stored_equals_recompute(store, queries)
        for session in sessions:
            await store.delete_session(session.id)
        await _assert_stored_equals_recompute(store, queries)
    finally:
        await store.close()

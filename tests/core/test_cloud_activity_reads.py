"""The Cloud activity page reads the store a fixed number of times, whatever the history.

Cayu Cloud polls ``GET /cloud/v1/activity``. A page counts each run's steps and
scans event heads instead of reading every event of its sessions, so its store
reads stay the same as runs and retained history grow.
"""

from __future__ import annotations

import asyncio
from collections import Counter
from contextlib import contextmanager

import pytest
from tests.core.session_store_conformance.support import (
    _close_store,
    _open_store,
    conformance_postgres_dsn,
    session_store_case,
)
from tests.core.test_cloud_reporting import PRICE, _app, _deny, _drain, _Record, _start

from cayu import AgentSpec, Message, ResumeRequest
from cayu.cloud_reporting import read_cloud_activity
from cayu.events import EventType
from cayu.sessions.event_queries import EventOrder, EventQuery
from cayu.sessions.records import EventHead
from cayu.sessions.summaries import SessionStepCounts

__all__ = ["conformance_postgres_dsn", "session_store_case"]


def _activity_app(store=None):
    app = _app(store)
    # Takes notes without asking, so its runs make tool calls.
    app.register_agent(AgentSpec(name="recorder", model=PRICE.model), tools=[_Record()])
    return app


async def _workload(app, prefix: str, *, touch: bool = True) -> list[str]:
    """Three runs: a reply, a tool call, and a denied approval or another reply.

    Classifying a touch reads its own evidence, so pages compared for their
    store reads hold the same touches.
    """

    await _start(app, f"{prefix}-write", "writer")
    await _start(app, f"{prefix}-record", "recorder")
    if not touch:
        await _start(app, f"{prefix}-reply", "writer")
        return [f"{prefix}-write", f"{prefix}-record", f"{prefix}-reply"]
    await _start(app, f"{prefix}-deny", "approver")
    await _deny(app, f"{prefix}-deny")
    return [f"{prefix}-write", f"{prefix}-record", f"{prefix}-deny"]


@pytest.mark.store_semantics
def test_step_counts_and_event_heads_match_the_full_reads(session_store_case) -> None:
    async def scenario():
        store = await _open_store(session_store_case)
        try:
            app = _activity_app(store)
            ids = await _workload(app, "heads")
            counts = await store.count_session_steps([*ids, "heads-missing"])
            usage = await store.read_usage_accounting(
                EventQuery(session_ids=tuple(ids)), by_session=True
            )
            expected = {
                summary.session_id: SessionStepCounts(summary.model_steps, summary.tool_calls)
                for summary in usage.session_summaries
            }
            queries = [
                EventQuery(
                    session_ids=tuple(ids),
                    event_types=(EventType.MODEL_COMPLETED, EventType.SESSION_INTERRUPTED),
                    after_sequence=0,
                    limit=50,
                    order_by=EventOrder.SEQUENCE_ASC,
                ),
                EventQuery(session_id=ids[1], limit=7, order_by=EventOrder.SEQUENCE_DESC),
            ]
            heads = [
                (await store.query_event_heads(query), await store.query_events(query))
                for query in queries
            ]
            return ids, counts, expected, heads
        finally:
            await _close_store(store)

    ids, counts, expected, heads = asyncio.run(scenario())
    assert set(counts) == set(ids)
    assert counts == expected
    assert counts[ids[1]].tool_calls == 1
    assert counts[ids[0]] == SessionStepCounts(model_steps=1, tool_calls=0)
    for found, records in heads:
        assert records
        assert found == [EventHead.of(record) for record in records]


@pytest.fixture
def fresh_dsn(postgres_dsn):
    """A database of its own, so the page reads only this test's runs."""

    from uuid import uuid4

    import psycopg
    from psycopg import sql
    from psycopg.conninfo import make_conninfo

    database = f"cayu_activity_{uuid4().hex}"
    with psycopg.connect(postgres_dsn, autocommit=True) as connection:
        connection.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(database)))
    try:
        yield make_conninfo(postgres_dsn, dbname=database)
    finally:
        with psycopg.connect(postgres_dsn, autocommit=True) as connection:
            connection.execute(
                sql.SQL("DROP DATABASE {} WITH (FORCE)").format(sql.Identifier(database))
            )


@contextmanager
def _counted_statements():
    """Count every statement any Postgres cursor sends while the block runs."""

    import psycopg

    counts: Counter[str] = Counter()
    originals = (psycopg.AsyncCursor.execute, psycopg.AsyncServerCursor.execute)

    def counted(original):
        async def execute(self, query, params=None, **kwargs):
            counts[" ".join(str(query).split())[:60]] += 1
            return await original(self, query, params, **kwargs)

        return execute

    psycopg.AsyncCursor.execute = counted(originals[0])
    psycopg.AsyncServerCursor.execute = counted(originals[1])
    try:
        yield counts
    finally:
        psycopg.AsyncCursor.execute, psycopg.AsyncServerCursor.execute = originals


def test_postgres_activity_page_reads_do_not_grow_with_runs_or_history(
    fresh_dsn, monkeypatch
) -> None:
    pytest.importorskip("psycopg")
    from cayu.storage import PostgresSessionStore
    from cayu.storage.migrations import SchemaMode

    # Every page prices its runs, so pages of different runs read the same way.
    _page_without_cost_cache(monkeypatch)

    async def scenario():
        store = PostgresSessionStore(fresh_dsn, schema_mode=SchemaMode.CREATE)
        app = _activity_app(store)
        try:

            async def page(cursor=None):
                with _counted_statements() as counts:
                    read = await read_cloud_activity(app, cursor=cursor)
                assert read.has_more is False
                return read, sum(counts.values()), counts

            await _workload(app, "warm")
            await page()  # Open the pool and the store identity before counting.
            small = await page()
            for index in range(3):
                await _workload(app, f"more-{index}", touch=False)
            large = await page()
            # The same work after a short and after a long retained history.
            await _workload(app, "after-short")
            short = await page(large[0].next_cursor)
            for index in range(10):
                await _workload(app, f"history-{index}", touch=False)
            tail = await page(short[0].next_cursor)
            await _workload(app, "after-long")
            long = await page(tail[0].next_cursor)
            return small, large, short, long
        finally:
            await app.aclose()

    small, large, short, long = asyncio.run(scenario())
    assert (len(small[0].runs), len(large[0].runs)) == (3, 12)
    assert len(small[0].touches) == len(large[0].touches) == 1
    assert small[1] == large[1], (small[2] - large[2]) + (large[2] - small[2])
    assert [len(read.runs) for read, _, _ in (short, long)] == [3, 3]
    assert short[1] == long[1], (short[2] - long[2]) + (long[2] - short[2])
    by_session = {run.session_id: run for run in large[0].runs}
    assert by_session["more-0-record"].tool_calls == 1
    assert by_session["more-0-write"].model_steps == 1


@pytest.mark.store_semantics
def test_lineages_and_event_positions_match_the_full_reads(session_store_case) -> None:
    from cayu.runtime._cost_accounting import COST_EVENT_TYPES
    from cayu.sessions import _page_reads as page_reads
    from cayu.sessions.records import SessionLineage

    async def scenario():
        store = await _open_store(session_store_case)
        try:
            app = _activity_app(store)
            ids = [*await _workload(app, "lineage"), "lineage-missing"]
            lineages = await store.load_session_lineages(ids)
            sessions = await store.load_sessions(ids)
            positions = await store.summarize_event_positions(ids, COST_EVENT_TYPES)
            one_by_one = await page_reads.summarize_event_positions_one_by_one(
                store, ids, COST_EVENT_TYPES
            )
            return lineages, sessions, positions, one_by_one
        finally:
            await _close_store(store)

    lineages, sessions, positions, one_by_one = asyncio.run(scenario())
    assert lineages == {
        session_id: SessionLineage.of(session) for session_id, session in sessions.items()
    }
    assert positions == one_by_one
    assert set(positions) == set(sessions)
    assert all(position.count >= 1 for position in positions.values())


def _page_without_cost_cache(monkeypatch):
    """Read pages as before the cost cache: every run priced from its events."""

    from cayu import cloud_reporting
    from cayu.sessions.overview import SessionCostCache

    monkeypatch.setattr(cloud_reporting, "_cost_cache", lambda _app, _store_id: SessionCostCache())


@pytest.mark.store_semantics
def test_a_cached_run_cost_is_the_cost_read_from_its_events(
    session_store_case, monkeypatch
) -> None:
    async def scenario():
        store = await _open_store(session_store_case)
        try:
            app = _activity_app(store)
            await _workload(app, "cost")
            first = await read_cloud_activity(app)
            # The same runs, after another turn of one of them and a new run.
            await _drain(
                app.resume(
                    ResumeRequest(
                        session_id="cost-record", messages=[Message.text("user", "again")]
                    )
                )
            )
            await _workload(app, "later", touch=False)
            cached = await read_cloud_activity(app)
            again = await read_cloud_activity(app, cursor=first.next_cursor)
            with monkeypatch.context() as patch:
                _page_without_cost_cache(patch)
                fresh = await read_cloud_activity(app)
                fresh_again = await read_cloud_activity(app, cursor=first.next_cursor)
            return first, cached, again, fresh, fresh_again
        finally:
            await _close_store(store)

    first, cached, again, fresh, fresh_again = asyncio.run(scenario())
    assert cached == fresh
    assert again == fresh_again
    costs = {run.session_id: run.cost for run in cached.runs}
    assert (
        costs["cost-record"].amount
        > {run.session_id: run.cost for run in first.runs}["cost-record"].amount
    )


@contextmanager
def _counted_page_work():
    """Count, per page: session row reads, anchor reads, priced events and loop decodes."""

    import threading

    from cayu.runtime import _cost_accounting
    from cayu.sessions import _supervision_projection as projection
    from cayu.storage import _postgres_page_reads as pg_page_reads
    from cayu.storage import _postgres_support as pg_support

    counts: Counter[str] = Counter()
    add = _cost_accounting.CostAccountingReducer.add
    session_from_row = pg_support.session_from_row
    load_sessions = pg_page_reads.load_sessions
    retained = projection._retained_anchor_records

    def counted_add(self, sequence, event):
        counts["priced events"] += 1
        return add(self, sequence, event)

    def counted_session(row, **kwargs):
        if threading.current_thread() is threading.main_thread():
            counts["session rows decoded on the event loop"] += 1
        return session_from_row(row, **kwargs)

    async def counted_sessions(cur, session_ids):
        counts["session row reads"] += 1
        return await load_sessions(cur, session_ids)

    async def counted_retained(store, anchors):
        counts["anchor reads"] += 1
        return await retained(store, anchors)

    _cost_accounting.CostAccountingReducer.add = counted_add
    pg_support.session_from_row = counted_session
    pg_page_reads.load_sessions = counted_sessions
    projection._retained_anchor_records = counted_retained
    try:
        yield counts
    finally:
        _cost_accounting.CostAccountingReducer.add = add
        pg_support.session_from_row = session_from_row
        pg_page_reads.load_sessions = load_sessions
        projection._retained_anchor_records = retained


def test_postgres_activity_page_work_per_run_and_touch(fresh_dsn) -> None:
    pytest.importorskip("psycopg")
    from cayu.storage import PostgresSessionStore
    from cayu.storage.migrations import SchemaMode

    async def scenario():
        store = PostgresSessionStore(fresh_dsn, schema_mode=SchemaMode.CREATE)
        app = _activity_app(store)
        try:

            async def page(cursor=None):
                with _counted_page_work() as work:
                    read = await read_cloud_activity(app, cursor=cursor)
                return read, dict(work)

            await _workload(app, "warm")
            await page()
            for index in range(3):
                await _workload(app, f"touch-{index}")
            touches = await page()
            # The same runs again, unchanged since they were priced.
            repeat = await page()
            return touches, repeat
        finally:
            await app.aclose()

    touches, repeat = asyncio.run(scenario())
    read, work = touches
    assert len(read.runs) == 12 and len(read.touches) == 4
    # One read of whole session rows per page, decoded off the event loop.
    assert work["session row reads"] == 1
    assert work.get("session rows decoded on the event loop", 0) == 0
    # The page's anchors are read together.
    assert work["anchor reads"] == 1
    # A page whose runs' costs did not change prices no event.
    assert repeat[1].get("priced events", 0) == 0
    assert repeat[0] == read

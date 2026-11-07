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
from tests.core.test_cloud_reporting import PRICE, _app, _deny, _Record, _start

from cayu import AgentSpec
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


def test_postgres_activity_page_reads_do_not_grow_with_runs_or_history(fresh_dsn) -> None:
    pytest.importorskip("psycopg")
    from cayu.storage import PostgresSessionStore
    from cayu.storage.migrations import SchemaMode

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

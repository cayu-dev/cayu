"""Session lists and counts filter, sort and count in SQL on summary rows.

Lists narrowed by overview state, eval and fork lineage, or text return full
pages until the list ends; counts are grouped reads that equal the reference
reader's counts across random histories; the total count is opt-in; and each
list query sorts on an index rather than visiting every session.
"""

from __future__ import annotations

import asyncio
import contextlib
import random
import sqlite3
from datetime import UTC, datetime, timedelta

import httpx
import pytest
from tests.core.session_store_conformance.support import (
    _close_store,
    _open_store,
    conformance_postgres_dsn,
    session_store_case,
)
from tests.core.test_cloud_reporting import _app, _drain, _start
from tests.server.test_session_summary_rows import _all_sessions, _service, _step

from cayu import Message, RunRequest
from cayu._time_buckets import TimeSeriesSpec
from cayu.budgets.pricing import default_price_book
from cayu.server import ServerConfig, create_server
from cayu.sessions import ForkSessionRequest
from cayu.sessions._run_identity import eval_trial_scope
from cayu.sessions.overview import SessionOverviewReader
from cayu.sessions.queries import SessionAggregateFilter, SessionQuery
from cayu.sessions.summary_rows import SessionOverviewFilter
from cayu.storage import _session_store_sql as session_store_sql
from cayu.storage._session_summary_rows import overview_filter_clause
from cayu.storage._sqlite_session_queries import SQL_DIALECT

__all__ = ["conformance_postgres_dsn", "session_store_case"]


async def _populate(app) -> None:
    for index in range(7):
        await _start(app, f"done-{index}", "writer")
    for index in range(3):
        await _start(app, f"ask-{index}", "asker")
    await _start(app, "approve-0", "approver")
    async with eval_trial_scope():
        await _start(app, "eval-0", "asker")
    await _drain(
        app.fork_session(ForkSessionRequest(source_session_id="done-1", session_id="fork-0"))
    )
    await _drain(
        app.run(
            RunRequest(
                agent_name="asker",
                session_id="child-0",
                parent_session_id="done-0",
                messages=[Message.text("user", "go")],
            )
        )
    )


async def _walk(client: httpx.AsyncClient, params: dict) -> list[list[str]]:
    pages: list[list[str]] = []
    cursor = None
    while True:
        response = await client.get(
            "/api/sessions", params={**params, **({"cursor": cursor} if cursor else {})}
        )
        assert response.status_code == 200, response.text
        body = response.json()
        pages.append([row["id"] for row in body["sessions"]])
        assert body["total_count"] is None or "total" in params.get("include", [])
        cursor = body["next_cursor"]
        if cursor is None:
            return pages
        assert len(pages) < 50


FILTERS = [
    {"overview_state": "waiting_input"},
    {"overview_state": ["waiting_approval", "waiting_input"]},
    {"overview_state": "failed"},
    {"include_eval": "false", "include_forks": "false"},
    {"include_eval": "false", "overview_state": "waiting_input"},
    {"q": "ask"},
    {"q": "ask", "include_forks": "false"},
    {"q": "e"},
]


@pytest.mark.store_semantics
def test_filtered_lists_return_full_pages_until_they_end(session_store_case):
    async def scenario():
        store = await _open_store(session_store_case)
        try:
            app = _app(store)
            await _populate(app)
            server = create_server(app, config=ServerConfig.local_development())
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=server), base_url="http://test"
            ) as client:
                everything = (
                    await client.get("/api/sessions", params={"include": "overview", "limit": 100})
                ).json()
                rows = {
                    session["id"]: (session, overview)
                    for session, overview in zip(
                        everything["sessions"], everything["overviews"], strict=True
                    )
                }
                order = [session["id"] for session in everything["sessions"]]
                for params in FILTERS:
                    # A fresh write leaves a row unsettled; it must still be listed right.
                    await _start(app, "late", "asker")
                    expected = [
                        session_id for session_id in order if _matches(params, *rows[session_id])
                    ]
                    for limit in (1, 2, 5):
                        pages = await _walk(
                            client, {**params, "include": "overview", "limit": limit}
                        )
                        listed = [session_id for page in pages for session_id in page]
                        assert [i for i in listed if i in rows] == expected, params
                        # Every page but the last is full where the store keeps summary
                        # rows; the in-memory store filters each store page on its own.
                        if session_store_case[0] != "memory":
                            assert all(len(page) == limit for page in pages[:-1]), (params, pages)
                    await store.delete_session("late")
        finally:
            await _close_store(store)

    asyncio.run(scenario())


def _matches(params: dict, session: dict, overview: dict) -> bool:
    states = params.get("overview_state")
    if states is not None:
        states = [states] if isinstance(states, str) else states
        if overview["state"] not in states:
            return False
    if params.get("include_eval") == "false" and overview["eval"]:
        return False
    if params.get("include_forks") == "false" and overview["forked"]:
        return False
    text = params.get("q")
    if text is not None:
        values = [
            overview["title"],
            *(session.get(field) for field in ("id", "agent_name", "provider_name", "model")),
            *(session.get(field) for field in ("environment_name", "parent_session_id")),
            session.get("causal_budget_id"),
            *session["labels"].keys(),
            *session["labels"].values(),
        ]
        return any(
            text.casefold() in value.casefold() for value in values if isinstance(value, str)
        )
    return True


@pytest.mark.store_semantics
def test_total_count_is_returned_only_when_asked(session_store_case):
    async def scenario():
        store = await _open_store(session_store_case)
        try:
            app = _app(store)
            for index in range(3):
                await _start(app, f"s{index}", "writer")
            server = create_server(app, config=ServerConfig.local_development())
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=server), base_url="http://test"
            ) as client:
                plain = (await client.get("/api/sessions", params={"limit": 2})).json()
                assert plain["total_count"] is None
                total = (
                    await client.get("/api/sessions", params={"limit": 2, "include": "total"})
                ).json()
                assert total["total_count"] == 3
        finally:
            await _close_store(store)

    asyncio.run(scenario())


@pytest.mark.store_semantics
@pytest.mark.parametrize("seed", range(3))
def test_counts_equal_the_reference_across_random_histories(session_store_case, seed):
    async def scenario():
        store = await _open_store(session_store_case)
        try:
            app = _app(store)
            rng = random.Random(seed)
            names: list[str] = []
            counter = [0]
            for _ in range(14):
                with contextlib.suppress(KeyError, ValueError, RuntimeError):
                    await _step(app, rng, names, counter)
                if rng.random() < 0.5:
                    await _assert_counts_match(app)
            await _assert_counts_match(app)
        finally:
            await _close_store(store)

    asyncio.run(scenario())


async def _assert_counts_match(app) -> None:
    now = datetime.now(UTC)
    pricing = default_price_book()
    service = _service(app)
    sessions = await _all_sessions(app.session_store)
    agents = sorted({session.agent_name for session in sessions}) or ["writer"]
    for since, until in (
        (now - timedelta(days=1), now + timedelta(minutes=1)),
        (now - timedelta(seconds=2), now + timedelta(minutes=1)),
        (now + timedelta(minutes=1), now + timedelta(minutes=2)),
    ):
        for include_eval in (False, True):
            for include_forks in (False, True):
                for filters in (
                    SessionAggregateFilter(),
                    SessionAggregateFilter(agent_name=agents[0]),
                ):
                    series = TimeSeriesSpec(unit="hour", timezone="UTC")
                    arguments = dict(
                        since=since,
                        until=until,
                        include_eval=include_eval,
                        include_forks=include_forks,
                        filters=filters,
                        series=series,
                    )
                    reference = await SessionOverviewReader(
                        app.session_store, pricing=pricing
                    ).counts(**arguments)
                    summarized = await service.summaries.counts(**arguments, pricing=pricing)
                    if summarized is None:
                        return  # No summary rows: the route reads the reference.
                    assert summarized == reference, arguments


@pytest.mark.store_semantics
def test_list_queries_sort_on_an_index(session_store_case):
    """Plans must not sort every session; a later change can't drop an index silently."""

    if session_store_case[0] != "sqlite":
        pytest.skip("The SQLite plan is checked here; Postgres has its own test below.")

    async def scenario():
        store = await _open_store(session_store_case)
        try:
            app = _app(store)
            await _populate(app)
            path = session_store_case[1] / "sessions.sqlite"
            return path
        finally:
            await _close_store(store)

    path = asyncio.run(scenario())
    with contextlib.closing(sqlite3.connect(path)) as connection:
        for query, overview, index in _PLANS:
            plan = _sqlite_plan(connection, query, overview)
            assert index in plan, (query, plan)
            assert "USE TEMP B-TREE FOR ORDER BY" not in plan, (query, plan)


_PLANS = [
    (SessionQuery(limit=50), SessionOverviewFilter(), "idx_cayu_sessions_updated_id"),
    (
        SessionQuery(limit=50, agent_name="asker"),
        SessionOverviewFilter(),
        "idx_cayu_sessions_agent_updated_id",
    ),
    (
        SessionQuery(limit=50, environment_name="env"),
        SessionOverviewFilter(),
        "idx_cayu_sessions_environment_updated_id",
    ),
    (
        SessionQuery(limit=50, parent_session_id="done-0"),
        SessionOverviewFilter(),
        "idx_cayu_sessions_parent_updated_id",
    ),
    (
        SessionQuery(limit=50, order_by="created_at_desc"),
        SessionOverviewFilter(),
        "idx_cayu_sessions_created_id",
    ),
    (
        SessionQuery(limit=50),
        SessionOverviewFilter(states=frozenset({"failed"})),
        "idx_cayu_session_summaries_state",
    ),
    (
        SessionQuery(limit=50),
        SessionOverviewFilter(include_eval=False, include_forks=False),
        "idx_cayu_sessions_updated_id",
    ),
]


def _sqlite_plan(connection, query: SessionQuery, overview: SessionOverviewFilter) -> str:
    plan = session_store_sql.build_session_query_sql(
        query,
        dialect=SQL_DIALECT,
        access_clause=overview_filter_clause(overview, postgres=False),
    )
    rows = connection.execute(
        "EXPLAIN QUERY PLAN SELECT cayu_sessions.id FROM cayu_sessions "
        f"{plan.page_where_sql} ORDER BY {plan.order_sql} {plan.pagination_sql}",
        plan.page_params,
    ).fetchall()
    return "\n".join(str(row[-1]) for row in rows)


@pytest.mark.store_semantics
def test_postgres_list_queries_can_sort_on_an_index(session_store_case):
    """With sequential scans disabled, every list plan must use its index, not sort."""

    if session_store_case[0] != "postgres":
        pytest.skip("The Postgres plan is checked here; SQLite has its own test above.")
    import psycopg

    from cayu.storage._postgres_session_queries import SQL_DIALECT as POSTGRES_DIALECT

    async def scenario():
        store = await _open_store(session_store_case)
        try:
            await _populate(_app(store))
        finally:
            await _close_store(store)
        async with (
            await psycopg.AsyncConnection.connect(session_store_case[2]) as conn,
            conn.cursor() as cur,
        ):
            await cur.execute("SET enable_seqscan = off")
            await cur.execute("SET enable_bitmapscan = off")
            await cur.execute("SET enable_sort = off")
            for query, overview, index in _PLANS:
                plan = session_store_sql.build_session_query_sql(
                    query,
                    dialect=POSTGRES_DIALECT,
                    access_clause=overview_filter_clause(overview, postgres=True),
                )
                expected = (
                    "idx_cayu_sessions_updated_id"
                    if index == "idx_cayu_session_summaries_state"
                    else index
                )
                # The table is a few rows on one page, so a filtered list costs the same on
                # the plain updated-at index (with a filter) as on its own index, and the
                # planner picks either. Take the plain one away, inside a savepoint, so the
                # plan shows whether the list's own index can serve it without a sort.
                await cur.execute("SAVEPOINT plan")
                if expected != "idx_cayu_sessions_updated_id":
                    await cur.execute("DROP INDEX idx_cayu_sessions_updated_id")
                await cur.execute(
                    "EXPLAIN SELECT cayu_sessions.id FROM cayu_sessions "
                    f"{plan.page_where_sql} ORDER BY {plan.order_sql} {plan.pagination_sql}",
                    plan.page_params,
                )
                text = "\n".join(row[0] for row in await cur.fetchall())
                await cur.execute("ROLLBACK TO SAVEPOINT plan")
                assert expected in text, (query, text)
                assert "Sort  (" not in text and "Sort Key" not in text, (query, text)

    asyncio.run(scenario())


@pytest.mark.store_semantics
def test_counts_derive_unsettled_rows_a_batch_at_a_time(session_store_case, monkeypatch):
    """Many never-listed sessions are derived in batches, not one page read of them all."""

    from cayu.server import _session_summaries

    monkeypatch.setattr(_session_summaries, "_USAGE_PAGE", 2)

    async def scenario():
        store = await _open_store(session_store_case)
        try:
            app = _app(store)
            for index in range(5):
                await _start(app, f"bulk-{index}", "writer")
            loads: list[int] = []
            load = store.load_session_summaries

            async def counted(session_ids):
                loads.append(len(session_ids))
                return await load(session_ids)

            monkeypatch.setattr(store, "load_session_summaries", counted)
            arguments = dict(
                since=None,
                until=None,
                include_eval=False,
                include_forks=False,
                filters=SessionAggregateFilter(),
                series=None,
            )
            summarized = await _service(app).summaries.counts(**arguments, pricing=None)
            if summarized is None:
                return  # No summary rows: the route reads the reference.
            assert summarized.total_count == 5
            assert loads and max(loads) <= 2
        finally:
            await _close_store(store)

    asyncio.run(scenario())

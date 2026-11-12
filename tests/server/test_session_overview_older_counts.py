"""Counts find every older session that can change them, without paging history."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import pytest
from tests.core.session_store_conformance.support import (
    _close_store,
    _open_store,
    conformance_postgres_dsn,
    session_store_case,
)
from tests.core.test_cloud_reporting import _app, _delegating_app, _start

from cayu.sessions import overview
from cayu.sessions.event_queries import EventQuery
from cayu.sessions.overview import SessionOverviewReader
from cayu.sessions.queries import SessionAggregateFilter, session_query_from_aggregate_filter

__all__ = ["conformance_postgres_dsn", "session_store_case"]
UNFILTERED = session_query_from_aggregate_filter(SessionAggregateFilter())


async def _by_agent(store) -> dict[str, str]:
    from cayu.sessions.queries import SessionQuery

    listed = await store.list_sessions(SessionQuery(limit=100))
    return {session.agent_name: session.id for session in listed.sessions}


@pytest.mark.store_semantics
@pytest.mark.parametrize("paged", [False, True])
def test_older_parents_waiting_on_subagents_are_found_by_creation_paging(
    session_store_case, monkeypatch, paged
):
    async def scenario():
        store = await _open_store(session_store_case)
        try:
            app = _delegating_app(store)
            await _start(app, "root", "root")
            agents = await _by_agent(store)
            if paged:
                # A pending read that needs another page pages by mutable
                # activity time, so the whole listing page is kept instead.
                original = store.query_pending_actions_for_sessions

                async def more(session_ids, query=None, **kwargs):
                    page = await original(session_ids, query, **kwargs)
                    return page.model_copy(update={"has_more": True, "next_cursor": "next"})

                monkeypatch.setattr(store, "query_pending_actions_for_sessions", more)
            since = datetime.now(UTC) + timedelta(seconds=1)
            candidates = await SessionOverviewReader(store)._older_count_candidates(
                UNFILTERED, since, since + timedelta(hours=1)
            )
            # Root and middle each wait on a subagent; the leaf waits on a person.
            expected = {"root", agents["middle"]}
            if paged:
                expected.add(agents["leaf"])
            assert set(candidates) == expected
        finally:
            await _close_store(store)

    asyncio.run(scenario())


@pytest.mark.store_semantics
def test_older_usage_is_found_by_split_time_ranges_without_sequence_cursors(
    session_store_case, monkeypatch
):
    async def scenario():
        store = await _open_store(session_store_case)
        try:
            app = _app(store)
            await _start(app, "before", "writer")
            since = (await store.load("before")).created_at + timedelta(microseconds=1)
            for name in ("older-a", "older-b", "older-c"):
                await _start(app, name, "writer")
            until = datetime.now(UTC) + timedelta(seconds=1)
            # Four runs record a usage-bearing event each, so a page of two fills
            # and the window is split until every range holds fewer.
            monkeypatch.setattr(overview, "_USAGE_HEAD_PAGE", 2)
            queries: list[EventQuery] = []
            original = store.query_event_heads

            async def observed(query):
                queries.append(query)
                return await original(query)

            monkeypatch.setattr(store, "query_event_heads", observed)
            reader = SessionOverviewReader(store)
            found: dict[str, None] = {}
            await reader._usage_sessions(since, until, found)
            assert set(found) == {"before", "older-a", "older-b", "older-c"}
            assert len(queries) > 1
            assert all(q.after_sequence is None and q.before_sequence is None for q in queries)
            # The split ranges tile the window exactly once.
            covered = sorted((q.since, q.until) for q in queries if q.since and q.until)
            assert covered[0][0] == since and max(end for _, end in covered) == until
            usage = await store.read_usage_accounting(
                EventQuery(
                    session_ids=("before", "older-a", "older-b", "older-c"),
                    since=since,
                    until=until,
                )
            )
            counts = await SessionOverviewReader(store).counts(since=since, until=until)
            assert counts.input_tokens == usage.summary.usage.input_tokens > 0
        finally:
            await _close_store(store)

    asyncio.run(scenario())


@pytest.mark.store_semantics
def test_usage_events_sharing_one_instant_are_read_from_one_snapshot(
    session_store_case, monkeypatch
):
    from cayu.events import Event, EventType
    from cayu.sessions.records import SessionIdentity
    from cayu.sessions.requests import RunRequest

    async def scenario():
        store = await _open_store(session_store_case)
        try:
            names = ("dense-a", "dense-b", "dense-c")
            for name in names:
                await store.create(
                    RunRequest(session_id=name, agent_name="writer", messages=[]),
                    identity=SessionIdentity(provider_name="test", model="test"),
                )
            since = datetime.now(UTC) + timedelta(seconds=1)
            instant = since + timedelta(minutes=1)
            for name in names:
                await store.append_event(
                    name,
                    Event(
                        session_id=name,
                        timestamp=instant,
                        type=EventType.MODEL_COMPLETED,
                        payload={
                            "usage_metrics": {
                                "provider_name": "test",
                                "model": "test",
                                "input_tokens": 100,
                                "total_tokens": 100,
                            }
                        },
                    ),
                )
            # Three events at one instant fill every page of two, however far
            # the range is split, so the snapshot read must name the sessions.
            monkeypatch.setattr(overview, "_USAGE_HEAD_PAGE", 2)
            snapshots: list[EventQuery] = []
            original = store.read_usage_accounting

            async def observed(query, **kwargs):
                snapshots.append(query)
                return await original(query, **kwargs)

            monkeypatch.setattr(store, "read_usage_accounting", observed)
            reader = SessionOverviewReader(store)
            until = since + timedelta(hours=1)
            found: dict[str, None] = {}
            await reader._usage_sessions(since, until, found)
            assert set(found) == set(names)
            assert snapshots and all(
                q.until - q.since <= timedelta(milliseconds=1) for q in snapshots
            )
            # The older sessions' usage in the window is counted; none of them
            # was created in it.
            counts = await SessionOverviewReader(store).counts(since=since, until=until)
            assert counts.total_count == 0
            assert counts.input_tokens == 300
        finally:
            await _close_store(store)

    asyncio.run(scenario())

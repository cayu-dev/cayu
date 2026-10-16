"""The supervision candidate scan stays on the event-type/timestamp index."""

from __future__ import annotations

import asyncio
import sqlite3
from datetime import UTC, datetime, timedelta

from tests.core.session_store_conformance.support import (
    _close_store,
    _open_store,
    conformance_postgres_dsn,
)

from cayu.events import Event, EventType
from cayu.messages import Message
from cayu.sessions import SupervisionEventQuery
from cayu.sessions._supervision_projection import supervision_candidate_query
from cayu.sessions.event_queries import EventQuery
from cayu.sessions.records import SessionIdentity
from cayu.sessions.requests import RunRequest
from cayu.storage import _postgres_session_queries as postgres_queries
from cayu.storage import _session_store_sql as session_store_sql
from cayu.storage import _sqlite_session_queries as sqlite_queries

SESSIONS = 40
NOISE_PER_SESSION = 60
WINDOW = SupervisionEventQuery(
    start_at=datetime.now(UTC) - timedelta(days=1),
    end_at=datetime.now(UTC) + timedelta(days=1),
)
TYPE_TIMESTAMP_INDEX = "idx_cayu_events_type_timestamp"

__all__ = ["conformance_postgres_dsn"]


async def _populate(store) -> None:
    """Many tool and model events per session, with one approval anchor each."""

    identity = SessionIdentity(provider_name="fake", model="fake-model")
    for index in range(SESSIONS):
        session_id = f"sess_plan_{index}"
        await store.create(
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "task")],
            ),
            identity=identity,
        )
        noise = [
            Event(
                type=EventType.TOOL_CALL_STARTED if n % 2 else EventType.MODEL_STARTED,
                session_id=session_id,
                agent_name="assistant",
                payload={},
            )
            for n in range(NOISE_PER_SESSION)
        ]
        await store.append_events(session_id, noise)
        await store.append_event(
            session_id,
            Event(
                type=EventType.TOOL_CALL_APPROVAL_EXPIRED,
                session_id=session_id,
                agent_name="assistant",
                payload={"approval_id": f"approval_{index}"},
            ),
        )


def _candidate_query() -> EventQuery:
    return supervision_candidate_query(WINDOW).model_copy(update={"after_sequence": 0})


def _candidate_plan(dialect) -> session_store_sql.EventQuerySqlPlan:
    return session_store_sql.build_event_query_sql(_candidate_query(), dialect=dialect)


def test_sqlite_candidate_scan_uses_the_type_timestamp_index(tmp_path):
    case = ("sqlite", tmp_path, None)

    async def run():
        store = await _open_store(case)
        try:
            await _populate(store)
        finally:
            await _close_store(store)

    asyncio.run(run())
    plan = _candidate_plan(sqlite_queries.SQL_DIALECT)
    connection = sqlite3.connect(tmp_path / "sessions.sqlite")
    try:
        connection.execute("ANALYZE")
        rows = connection.execute(
            f"""
            EXPLAIN QUERY PLAN
            SELECT cayu_events.sequence
            FROM {sqlite_queries._event_query_source(_candidate_query())}
            JOIN cayu_sessions ON cayu_sessions.id = cayu_events.session_id
            {plan.where_sql}
            ORDER BY cayu_events.sequence {plan.order_direction}
            LIMIT ?
            """,
            [*plan.params, 100],
        ).fetchall()
    finally:
        connection.close()
    details = [row[3] for row in rows]

    assert any(TYPE_TIMESTAMP_INDEX in detail for detail in details), details
    assert not any(detail.startswith("SCAN cayu_events") for detail in details), details


def test_postgres_candidate_scan_can_use_the_type_timestamp_index(
    tmp_path, conformance_postgres_dsn
):
    import psycopg

    postgres_dsn = conformance_postgres_dsn
    case = ("postgres", tmp_path, postgres_dsn)

    async def run():
        store = await _open_store(case)
        try:
            await _populate(store)
        finally:
            await _close_store(store)
        plan = _candidate_plan(postgres_queries.SQL_DIALECT)
        async with (
            await psycopg.AsyncConnection.connect(postgres_dsn) as connection,
            connection.cursor() as cursor,
        ):
            await cursor.execute("ANALYZE cayu_events")
            await cursor.execute("SET enable_seqscan = off")
            await cursor.execute(
                f"""
                EXPLAIN
                SELECT cayu_events.sequence
                FROM cayu_events
                JOIN cayu_sessions ON cayu_sessions.id = cayu_events.session_id
                {plan.where_sql}
                ORDER BY cayu_events.sequence {plan.order_direction}
                LIMIT %s
                """,
                [*plan.params, 100],
            )
            return [row[0] for row in await cursor.fetchall()]

    details = asyncio.run(run())

    assert any(TYPE_TIMESTAMP_INDEX in detail for detail in details), details


def _expired(session_id: str, approval_id: str) -> Event:
    return Event(
        type=EventType.TOOL_CALL_APPROVAL_EXPIRED,
        session_id=session_id,
        agent_name="assistant",
        payload={"approval_id": approval_id},
    )


def test_postgres_page_never_passes_an_uncommitted_lower_sequence(
    tmp_path, conformance_postgres_dsn
):
    import psycopg

    postgres_dsn = conformance_postgres_dsn
    case = ("postgres", tmp_path, postgres_dsn)
    identity = SessionIdentity(provider_name="fake", model="fake-model")

    async def run():
        store = await _open_store(case)
        try:
            for session_id in ("sess_cutoff_early", "sess_cutoff_late"):
                await store.create(
                    RunRequest(
                        agent_name="assistant",
                        session_id=session_id,
                        messages=[Message.text("user", "task")],
                    ),
                    identity=identity,
                )
            await store.append_event("sess_cutoff_early", _expired("sess_cutoff_early", "first"))
            async with await psycopg.AsyncConnection.connect(postgres_dsn) as open_writer:
                async with open_writer.cursor() as cursor:
                    await cursor.execute(
                        "SELECT column_name FROM information_schema.columns "
                        "WHERE table_schema = current_schema() AND table_name = 'cayu_events' "
                        "AND column_name NOT IN ('sequence', 'insert_xid')"
                    )
                    columns = [row[0] for row in await cursor.fetchall()]
                    copied = ", ".join(
                        {
                            "event_id": "'uncommitted-event'",
                            "session_order": "session_order + 1000",
                            "payload": "jsonb_set(payload, '{approval_id}', '\"uncommitted\"')",
                            "event": (
                                "jsonb_set(jsonb_set(event, '{id}', '\"uncommitted-event\"'), "
                                "'{payload,approval_id}', '\"uncommitted\"')"
                            ),
                        }.get(column, column)
                        for column in columns
                    )
                    await cursor.execute(
                        f"INSERT INTO cayu_events ({', '.join(columns)}) "
                        f"SELECT {copied} FROM cayu_events WHERE session_id = 'sess_cutoff_early'"
                    )
                    await store.append_event(
                        "sess_cutoff_late", _expired("sess_cutoff_late", "third")
                    )
                    before_commit = await store.query_supervision_events(WINDOW)
                await open_writer.commit()
            after_commit = await store.query_supervision_events(
                WINDOW.model_copy(update={"cursor": before_commit.next_cursor})
            )
            return before_commit, after_commit
        finally:
            await _close_store(store)

    before_commit, after_commit = asyncio.run(run())

    assert [event.approval_id for event in before_commit.events] == ["first"]
    assert [event.approval_id for event in after_commit.events] == ["uncommitted", "third"]

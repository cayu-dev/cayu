"""Reads and settlements of ``cayu_session_summaries`` for both native stores.

Each read is one statement for a whole page of sessions. A settlement stores a
page of derived overviews in one transaction, each only if the session's
incarnation and source generation are still the ones it was computed from, so
it can never overwrite a newer write's invalidation, together with the stamps
of its dependencies, which a later read compares with their rows.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any

from cayu.sessions.summary_rows import (
    SessionSummaryRow,
    SessionSummaryStamp,
    SessionUsageDeltas,
    SummarySettlement,
)
from cayu.storage import _sqlite_records as sqlite_records
from cayu.storage._session_summary_schema import dependencies_current_sql

SUMMARY_COLUMNS = (
    "session_id, instance_id, parent_session_id, agent_name, environment_name, "
    "provider_name, model, causal_budget_id, status, eval, title, created_at, updated_at, "
    "last_activity_at, usage_epoch, source_generation, "
    "derived_generation, state, forked, artifact_count, derived_json"
)
_SQLITE_PAGE = "SELECT CAST(value AS TEXT) FROM json_each(?)"


def _loaded_columns(*, postgres: bool) -> str:
    current = dependencies_current_sql("summary", postgres=postgres)
    return f"{SUMMARY_COLUMNS}, {current} AS dependencies_current"


def _unsettled_sql(*, postgres: bool) -> str:
    return (
        "(summary.derived_generation IS NULL "
        "OR summary.derived_generation <> summary.source_generation "
        f"OR NOT {dependencies_current_sql('summary', postgres=postgres)})"
    )


#: Session IDs per page of an ``--all`` walk or a backfill.
MAX_SUMMARY_PAGE = 500


def _row(values: Sequence[Any], *, parse_time, parse_json) -> SessionSummaryRow:
    (
        session_id,
        instance_id,
        parent_session_id,
        agent_name,
        environment_name,
        provider_name,
        model,
        causal_budget_id,
        status,
        eval_mark,
        title,
        created_at,
        updated_at,
        last_activity_at,
        usage_epoch,
        source_generation,
        derived_generation,
        state,
        forked,
        artifact_count,
        derived,
        dependencies_current,
    ) = values
    return SessionSummaryRow(
        session_id=session_id,
        instance_id=instance_id,
        parent_session_id=parent_session_id,
        agent_name=agent_name,
        environment_name=environment_name,
        provider_name=provider_name,
        model=model,
        causal_budget_id=causal_budget_id,
        status=status,
        eval=bool(eval_mark),
        title=title,
        created_at=parse_time(created_at),
        updated_at=parse_time(updated_at),
        last_activity_at=parse_time(last_activity_at),
        usage_epoch=int(usage_epoch),
        source_generation=int(source_generation),
        derived_generation=None if derived_generation is None else int(derived_generation),
        state=state,
        forked=None if forked is None else bool(forked),
        artifact_count=int(artifact_count),
        derived=None if derived is None else parse_json(derived),
        dependencies_current=bool(dependencies_current),
    )


def _ordered(found: dict[str, Any], session_ids: Sequence[str]) -> dict[str, Any]:
    return {session_id: found[session_id] for session_id in session_ids if session_id in found}


def _derived_text(derived: dict[str, Any]) -> str:
    return json.dumps(derived, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


# SQLite


def sqlite_load(
    connection: sqlite3.Connection, session_ids: Sequence[str]
) -> dict[str, SessionSummaryRow]:
    if not session_ids:
        return {}
    rows = connection.execute(
        f"SELECT {_loaded_columns(postgres=False)} FROM cayu_session_summaries AS summary "
        f"WHERE summary.session_id IN ({_SQLITE_PAGE})",
        (sqlite_records.json_dumps(list(session_ids)),),
    ).fetchall()
    found = {
        row[0]: _row(tuple(row), parse_time=sqlite_records.parse_datetime, parse_json=json.loads)
        for row in rows
    }
    return _ordered(found, session_ids)


def sqlite_stamps(
    connection: sqlite3.Connection, session_ids: Sequence[str]
) -> dict[str, SessionSummaryStamp]:
    if not session_ids:
        return {}
    rows = connection.execute(
        "SELECT session_id, instance_id, source_generation FROM cayu_session_summaries "
        f"WHERE session_id IN ({_SQLITE_PAGE})",
        (sqlite_records.json_dumps(list(session_ids)),),
    ).fetchall()
    found = {row[0]: (row[1], int(row[2])) for row in rows}
    return {session_id: found.get(session_id) for session_id in dict.fromkeys(session_ids)}


def _dependency_rows(settled: Sequence[SummarySettlement]) -> list[tuple[Any, ...]]:
    return [
        (
            settlement.session_id,
            depends_on,
            None if stamp is None else stamp[0],
            None if stamp is None else stamp[1],
        )
        for settlement in settled
        for depends_on, stamp in settlement.dependencies.items()
    ]


def sqlite_settle(
    connection: sqlite3.Connection, settlements: Sequence[SummarySettlement]
) -> frozenset[str]:
    """Store each settlement whose row is unchanged, with its dependencies, in one transaction."""

    if not settlements:
        return frozenset()
    connection.execute("BEGIN IMMEDIATE")
    try:
        own = sqlite_stamps(connection, [settlement.session_id for settlement in settlements])
        settled = [
            settlement
            for settlement in settlements
            if own.get(settlement.session_id) == (settlement.instance_id, settlement.generation)
        ]
        connection.executemany(
            "UPDATE cayu_session_summaries SET derived_generation = ?, state = ?, forked = ?, "
            "derived_json = ? WHERE session_id = ?",
            [
                (
                    settlement.generation,
                    settlement.state,
                    int(settlement.forked),
                    _derived_text(settlement.derived),
                    settlement.session_id,
                )
                for settlement in settled
            ],
        )
        connection.executemany(
            "DELETE FROM cayu_session_summary_dependencies WHERE session_id = ?",
            [(settlement.session_id,) for settlement in settled],
        )
        connection.executemany(
            "INSERT INTO cayu_session_summary_dependencies "
            "(session_id, depends_on, instance_id, source_generation) VALUES (?, ?, ?, ?)",
            _dependency_rows(settled),
        )
        connection.commit()
        return frozenset(settlement.session_id for settlement in settled)
    except BaseException:
        connection.rollback()
        raise


def sqlite_usage_deltas(
    connection: sqlite3.Connection, afters: Mapping[str, int]
) -> SessionUsageDeltas:
    """Each session's usage after its own sequence, through its newest event, in one snapshot.

    A session's appends are serialized, so every later event of the session has
    a higher sequence than the newest one this snapshot holds.
    """

    from cayu.budgets.usage import USAGE_BEARING_EVENT_TYPES
    from cayu.runtime._usage_accounting import USAGE_ACCOUNTING_PAGE_SIZE, UsageAccountingReducer
    from cayu.sessions.event_queries import EventQuery
    from cayu.sessions.records import EventRecord

    event_types = [str(event_type) for event_type in USAGE_BEARING_EVENT_TYPES]
    event_columns = ", ".join(f"event.{name}" for name in sqlite_records.EVENT_COLUMN_NAMES)
    reducer = UsageAccountingReducer(EventQuery(session_ids=tuple(afters)), by_session=True)
    with connection:
        connection.execute("BEGIN")
        heads = {
            row[0]: int(row[1])
            for row in connection.execute(
                "SELECT bound.value, (SELECT MAX(sequence) FROM cayu_events "
                "WHERE session_id = bound.value) FROM json_each(?) AS bound",
                (sqlite_records.json_dumps(list(afters)),),
            ).fetchall()
            if row[1] is not None
        }
        bounds = {
            session_id: [after, heads[session_id]]
            for session_id, after in afters.items()
            if heads.get(session_id, 0) > after
        }
        cursor = connection.execute(
            "SELECT event.sequence, "
            f"{event_columns} FROM json_each(?) AS bound "
            "JOIN cayu_events AS event ON event.session_id = bound.key "
            "AND event.sequence > json_extract(bound.value, '$[0]') "
            "AND event.sequence <= json_extract(bound.value, '$[1]') "
            f"WHERE event.event_type IN ({', '.join('?' for _ in event_types)}) "
            "ORDER BY event.sequence",
            (sqlite_records.json_dumps(bounds), *event_types),
        )
        try:
            while rows := cursor.fetchmany(USAGE_ACCOUNTING_PAGE_SIZE):
                reducer.add_page(
                    [
                        EventRecord(sequence=row[0], event=sqlite_records.event_from_row(row))
                        for row in rows
                    ]
                )
        finally:
            cursor.close()
    snapshot = reducer.snapshot()
    return SessionUsageDeltas(
        through_sequences={
            session_id: max(after, heads.get(session_id, 0)) for session_id, after in afters.items()
        },
        summaries={summary.session_id: summary for summary in snapshot.session_summaries},
    )


def sqlite_page(
    connection: sqlite3.Connection, *, after: str | None, limit: int, unsettled: bool
) -> list[str]:
    filters = ["summary.session_id > ?"]
    if unsettled:
        filters.append(_unsettled_sql(postgres=False))
    rows = connection.execute(
        "SELECT summary.session_id FROM cayu_session_summaries AS summary "
        f"WHERE {' AND '.join(filters)} ORDER BY summary.session_id LIMIT ?",
        ("" if after is None else after, limit),
    ).fetchall()
    return [row[0] for row in rows]


# Postgres


def _pg_json(value: Any) -> Any:
    return json.loads(value) if isinstance(value, str | bytes) else value


def _pg_time(value: datetime) -> datetime:
    return value


async def postgres_load(cur: Any, session_ids: Sequence[str]) -> dict[str, SessionSummaryRow]:
    if not session_ids:
        return {}
    await cur.execute(
        f"SELECT {_loaded_columns(postgres=True)} "  # type: ignore[arg-type]
        "FROM cayu_session_summaries AS summary WHERE summary.session_id = ANY(%s::text[])",
        (list(session_ids),),
    )
    found = {
        row[0]: _row(tuple(row), parse_time=_pg_time, parse_json=_pg_json)
        for row in await cur.fetchall()
    }
    return _ordered(found, session_ids)


async def postgres_stamps(cur: Any, session_ids: Sequence[str]) -> dict[str, SessionSummaryStamp]:
    if not session_ids:
        return {}
    await cur.execute(
        "SELECT session_id, instance_id, source_generation FROM cayu_session_summaries "
        "WHERE session_id = ANY(%s::text[])",
        (list(session_ids),),
    )
    found = {row[0]: (row[1], int(row[2])) for row in await cur.fetchall()}
    return {session_id: found.get(session_id) for session_id in dict.fromkeys(session_ids)}


async def postgres_settle(cur: Any, settlements: Sequence[SummarySettlement]) -> frozenset[str]:
    """Settle inside the caller's transaction, which commits on return.

    The rows are locked with SKIP LOCKED: a writer holding one is about to
    advance its generation, so its settlement is skipped. A settlement that
    never waits can never be part of a deadlock with writers.
    """

    if not settlements:
        return frozenset()
    await cur.execute(
        "SELECT session_id, instance_id, source_generation FROM cayu_session_summaries "
        "WHERE session_id = ANY(%s::text[]) ORDER BY session_id FOR UPDATE SKIP LOCKED",
        (sorted({settlement.session_id for settlement in settlements}),),
    )
    own = {row[0]: (row[1], int(row[2])) for row in await cur.fetchall()}
    settled = [
        settlement
        for settlement in settlements
        if own.get(settlement.session_id) == (settlement.instance_id, settlement.generation)
    ]
    if not settled:
        return frozenset()
    ids = [settlement.session_id for settlement in settled]
    await cur.execute(
        "UPDATE cayu_session_summaries AS summary SET derived_generation = settled.generation, "
        "state = settled.state, forked = settled.forked, derived_json = settled.derived::jsonb "
        "FROM unnest(%s::text[], %s::bigint[], %s::text[], %s::boolean[], %s::text[]) "
        "AS settled(session_id, generation, state, forked, derived) "
        "WHERE summary.session_id = settled.session_id",
        (
            ids,
            [settlement.generation for settlement in settled],
            [settlement.state for settlement in settled],
            [settlement.forked for settlement in settled],
            [_derived_text(settlement.derived) for settlement in settled],
        ),
    )
    await cur.execute(
        "DELETE FROM cayu_session_summary_dependencies WHERE session_id = ANY(%s::text[])",
        (ids,),
    )
    rows = _dependency_rows(settled)
    if rows:
        await cur.execute(
            "INSERT INTO cayu_session_summary_dependencies "
            "(session_id, depends_on, instance_id, source_generation) "
            "SELECT * FROM unnest(%s::text[], %s::text[], %s::text[], %s::bigint[])",
            tuple([row[index] for row in rows] for index in range(4)),
        )
    return frozenset(ids)


async def postgres_usage_deltas(cur: Any, afters: Mapping[str, int]) -> SessionUsageDeltas:
    """Each session's usage after its own sequence, through its newest event, in one snapshot.

    A session's appends are serialized by its event counter row, so every later
    event of the session has a higher sequence than the newest one this
    snapshot holds, whatever other sessions' open writers have reserved.
    """

    from cayu.budgets.usage import USAGE_BEARING_EVENT_TYPES
    from cayu.events import Event
    from cayu.runtime._usage_accounting import USAGE_ACCOUNTING_PAGE_SIZE, UsageAccountingReducer
    from cayu.sessions.event_queries import EventQuery
    from cayu.sessions.records import EventRecord
    from cayu.storage import _postgres_support as pg_support

    await cur.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
    await cur.execute(
        "SELECT bound.session_id, (SELECT MAX(sequence) FROM cayu_events "
        "WHERE session_id = bound.session_id) FROM unnest(%s::text[]) AS bound(session_id)",
        (list(afters),),
    )
    heads = {row[0]: int(row[1]) for row in await cur.fetchall() if row[1] is not None}
    bounded = [
        (session_id, after, heads[session_id])
        for session_id, after in afters.items()
        if heads.get(session_id, 0) > after
    ]
    reducer = UsageAccountingReducer(EventQuery(session_ids=tuple(afters)), by_session=True)
    if bounded:
        await cur.execute(
            "SELECT event.sequence, event.event "
            "FROM unnest(%s::text[], %s::bigint[], %s::bigint[]) "
            "AS bound(session_id, after_sequence, head) "
            "JOIN cayu_events AS event ON event.session_id = bound.session_id "
            "AND event.sequence > bound.after_sequence AND event.sequence <= bound.head "
            "WHERE event.event_type = ANY(%s::text[]) ORDER BY event.sequence",
            (
                [row[0] for row in bounded],
                [row[1] for row in bounded],
                [row[2] for row in bounded],
                [str(event_type) for event_type in USAGE_BEARING_EVENT_TYPES],
            ),
        )
        while rows := await cur.fetchmany(USAGE_ACCOUNTING_PAGE_SIZE):
            reducer.add_page(
                [
                    EventRecord(sequence=row[0], event=Event(**pg_support._json_obj(row[1])))
                    for row in rows
                ]
            )
    snapshot = reducer.snapshot()
    return SessionUsageDeltas(
        through_sequences={
            session_id: max(after, heads.get(session_id, 0)) for session_id, after in afters.items()
        },
        summaries={summary.session_id: summary for summary in snapshot.session_summaries},
    )


async def postgres_page(cur: Any, *, after: str | None, limit: int, unsettled: bool) -> list[str]:
    filters = ['summary.session_id COLLATE "C" > %s']
    if unsettled:
        filters.append(_unsettled_sql(postgres=True))
    await cur.execute(
        "SELECT summary.session_id FROM cayu_session_summaries AS summary "  # type: ignore[arg-type]
        f'WHERE {" AND ".join(filters)} ORDER BY summary.session_id COLLATE "C" LIMIT %s',
        ("" if after is None else after, limit),
    )
    return [row[0] for row in await cur.fetchall()]

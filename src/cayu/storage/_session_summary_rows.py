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

from cayu.budgets.usage import USAGE_BEARING_EVENT_TYPES
from cayu.sessions.summary_rows import (
    SessionOverviewFilter,
    SessionSummaryRow,
    SessionSummaryStamp,
    SessionUsageDeltas,
    SummaryCounts,
    SummarySettlement,
)
from cayu.storage import _sqlite_records as sqlite_records
from cayu.storage._session_store_sql import SqlClause, combine_clauses
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


def _stale_dependencies_sql(*, postgres: bool) -> str:
    """The summary rows whose dependencies moved, as one uncorrelated subquery."""

    distinct = "IS DISTINCT FROM" if postgres else "IS NOT"
    return (
        "SELECT dependency.session_id FROM cayu_session_summary_dependencies AS dependency "
        "LEFT JOIN cayu_session_summaries AS other ON other.session_id = dependency.depends_on "
        f"WHERE other.instance_id {distinct} dependency.instance_id "
        f"OR other.source_generation {distinct} dependency.source_generation"
    )


def _listed_unsettled_sql(*, postgres: bool) -> str:
    """``_unsettled_sql`` for set reads: the dependency check is one uncorrelated subquery.

    A correlated check per row keeps list and count reads from using their indexes.
    """

    return (
        "(summary.derived_generation IS NULL "
        "OR summary.derived_generation <> summary.source_generation "
        f"OR summary.session_id IN ({_stale_dependencies_sql(postgres=postgres)}))"
    )


def _unsettled_sql(*, postgres: bool) -> str:
    return (
        "(summary.derived_generation IS NULL "
        "OR summary.derived_generation <> summary.source_generation "
        f"OR NOT {dependencies_current_sql('summary', postgres=postgres)})"
    )


_USAGE_BEARING = tuple(str(event_type) for event_type in USAGE_BEARING_EVENT_TYPES)
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


def overview_filter_clause(overview: SessionOverviewFilter, *, postgres: bool) -> SqlClause:
    """The ``cayu_sessions`` condition a session list adds for ``overview``.

    The summary's eval mark is copied from the session row, so it is exact. A
    stored state or fork lineage counts only on a settled row; an unsettled row
    stays a candidate for the caller to derive.
    """

    mark = "%s" if postgres else "?"
    false = "false" if postgres else "0"
    conditions: list[str] = []
    params: list[object] = []
    if overview.states:
        states = sorted(overview.states)
        conditions.append(f"summary.state IN ({', '.join(mark for _ in states)})")
        params.extend(states)
    if not overview.include_forks:
        conditions.append(f"summary.forked = {false}")
    derived = " AND ".join(conditions)
    eval_only = "" if overview.include_eval else f" AND summary.eval = {false}"
    clauses = []
    if derived:
        # Three indexed selections: unsettled rows (a partial index), rows whose
        # dependencies moved, and settled rows matching the stored columns.
        if postgres:
            # PostgreSQL filters an ordered index scan by one hashed subquery.
            candidates = (
                "SELECT summary.session_id FROM cayu_session_summaries AS summary "
                f"WHERE ({_listed_unsettled_sql(postgres=True)} OR ({derived})){eval_only}"
            )
            return SqlClause(f"cayu_sessions.id IN ({candidates})", tuple(params))
        stale = _stale_dependencies_sql(postgres=False)
        candidates = (
            "SELECT summary.session_id FROM cayu_session_summaries AS summary "
            "WHERE (summary.derived_generation IS NULL "
            f"OR summary.derived_generation <> summary.source_generation){eval_only} "
            "UNION SELECT summary.session_id FROM cayu_session_summaries AS summary "
            f"WHERE summary.session_id IN ({stale}){eval_only} "
            "UNION SELECT summary.session_id FROM cayu_session_summaries AS summary "
            f"WHERE {derived}{eval_only}"
        )
        # A membership test (the unary plus keeps SQLite from driving the list from
        # the candidates), so pages still read in sort-index order.
        clauses.append(SqlClause(f"+cayu_sessions.id IN ({candidates})", tuple(params)))
    elif not overview.include_eval:
        clauses.append(
            SqlClause(
                "cayu_sessions.id IN (SELECT summary.session_id "
                f"FROM cayu_session_summaries AS summary WHERE summary.eval = {false})",
                (),
            )
        )
    return combine_clauses(*clauses) or SqlClause("1 = 1")


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


# Overview counts


_ACTIVE_STATUSES = ("pending", "running", "interrupting")


def counts_statements(
    *,
    where_sql: str,
    where_params: tuple[object, ...],
    include_eval: bool,
    include_forks: bool,
    since: object,
    until: object,
    bucket_starts: Sequence[object],
    postgres: bool,
) -> tuple[tuple[str, tuple[object, ...]], ...]:
    """The statements ``read_counts`` runs, in one read snapshot.

    Candidates are the sessions matching the filter created in [since, until),
    and the older ones counts can still include: active ones, interrupted ones
    waiting on a subagent, and ones with usage-bearing events in the window.
    """

    mark = "%s" if postgres else "?"
    false = "false" if postgres else "0"
    if postgres:
        complete = "COALESCE((summary.derived_json ->> 'pending_complete')::boolean, false)"
        delegated = (
            "COALESCE((summary.derived_json -> 'waiting' ->> 'delegated_action')::bigint, 0)"
        )
    else:
        complete = "COALESCE(json_extract(summary.derived_json, '$.pending_complete'), 0) = 1"
        delegated = "COALESCE(json_extract(summary.derived_json, '$.waiting.delegated_action'), 0)"
    settled = f"(NOT {_listed_unsettled_sql(postgres=postgres)})"
    usage_types = ", ".join(f"'{event_type}'" for event_type in _USAGE_BEARING)
    used = (
        "summary.session_id IN (SELECT usage.session_id FROM cayu_usage_events AS usage "
        f"WHERE usage.event_type IN ({usage_types}) "
        f"AND usage.timestamp >= {mark} AND usage.timestamp < {mark})"
    )
    active = ", ".join(f"'{status}'" for status in _ACTIVE_STATUSES)
    eval_sql = "" if include_eval else f" AND summary.eval = {false}"
    fork_ok = "TRUE" if include_forks else f"forked = {false}"
    # The filter's columns are the session row's; it selects session IDs on its own.
    session_filter = (
        f"AND summary.session_id IN (SELECT cayu_sessions.id FROM cayu_sessions {where_sql})"
        if where_sql.strip()
        else ""
    )
    base = f"""
        WITH candidates AS (
            SELECT
                summary.session_id,
                summary.created_at >= {mark} AS in_window,
                summary.status,
                summary.state,
                summary.forked,
                summary.created_at,
                {settled} AS settled,
                {complete} AS complete,
                {used} AS used
            FROM cayu_session_summaries AS summary
            WHERE summary.created_at < {mark}{eval_sql}
              {session_filter}
              AND (
                  summary.created_at >= {mark}
                  OR summary.status IN ({active})
                  OR (summary.status = 'interrupted' AND (NOT {settled} OR {delegated} > 0))
                  OR {used}
              )
        )
    """
    params = (since, since, until, until, *where_params, since, since, until)
    counted = f"settled AND {fork_ok}"
    statements: list[tuple[str, tuple[object, ...]]] = [
        (
            base + f"SELECT in_window, state, COUNT(*), "
            f"SUM(CASE WHEN complete THEN 0 ELSE 1 END) FROM candidates WHERE {counted} "
            "GROUP BY in_window, state",
            params,
        ),
        (base + "SELECT session_id, in_window, used FROM candidates WHERE NOT settled", params),
        (base + f"SELECT session_id FROM candidates WHERE {counted} AND used", params),
    ]
    if bucket_starts:
        cases = " ".join(
            f"WHEN created_at < {mark} THEN {index}" for index in range(len(bucket_starts) - 1)
        )
        bucket = f"CASE {cases} ELSE {len(bucket_starts) - 1} END" if cases else "0"
        statements.append(
            (
                base + f"SELECT {bucket} AS bucket, COUNT(*), "
                "SUM(CASE WHEN state = 'failed' THEN 1 ELSE 0 END) "
                f"FROM candidates WHERE {counted} AND in_window GROUP BY 1",
                (*params, *bucket_starts[1:]),
            )
        )
    return tuple(statements)


def summary_counts_from_rows(rows: Sequence[Sequence[Sequence[Any]]]) -> SummaryCounts:
    grouped, unsettled, used, *bucketed = rows
    by_state: dict[str, int] = {}
    running = incomplete = 0
    for in_window, state, count, missing in grouped:
        incomplete += int(missing or 0)
        if state == "running":
            running += int(count)
        if in_window:
            by_state[state] = by_state.get(state, 0) + int(count)
    buckets = {
        int(row[0]): (int(row[1]), int(row[2] or 0)) for row in (bucketed[0] if bucketed else ())
    }
    return SummaryCounts(
        by_state=by_state,
        running=running,
        incomplete=incomplete,
        buckets=buckets,
        usage_session_ids=tuple(row[0] for row in used),
        unsettled=tuple((row[0], bool(row[1]), bool(row[2])) for row in unsettled),
    )

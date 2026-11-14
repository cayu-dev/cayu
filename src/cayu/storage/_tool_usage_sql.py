"""Retained tool usage tallies in one SQL aggregate, shared by SQLite and Postgres.

The statement counts what ``cayu.tool_inventory.tool_usage_contribution`` counts:
each ``tool.call.completed`` or ``tool.call.failed`` event with a tool name is one
call (failed ones also a failure), each ``tool.call.approved`` or
``tool.call.approval_denied`` event with a tool name is one approval decision, and
each ``model.hosted_tool_call`` whose payload ``tool_type`` is ``web_search`` and
whose ``status`` is a terminal string is one ``openai_web_search`` call. Only
hosted calls read the payload; every other event is tallied from its columns. The
statement reads ``cayu_usage_events``, the narrow copy of those events, under the
``cayu_events`` name so event query plans and access bounds apply unchanged. It is
the reference for ``STORED_TOOL_USAGE_TOTALS_SQL``, which sums the trigger-kept
``cayu_tool_usage`` rows instead.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from cayu import tool_inventory
from cayu.events import EventType
from cayu.sessions.event_queries import EventQuery
from cayu.storage import _session_store_sql as session_store_sql
from cayu.tool_inventory import TOOL_USAGE_EVENT_TYPES

HOSTED_WEB_SEARCH_TOOL_NAME = "openai_web_search"
_HOSTED_TERMINAL_STATUSES = tuple(sorted(tool_inventory._HOSTED_TERMINAL_STATUSES))


#: Postgres also adds the deltas not yet folded into ``cayu_tool_usage``.
POSTGRES_STORED_TOOL_USAGE_TOTALS_SQL = """
    SELECT tool_name AS tool, SUM(calls) AS calls, SUM(failures) AS failures,
           SUM(approved) AS approved, SUM(rejected) AS rejected,
           MAX(last_used_at) AS last_used_at
    FROM (
        SELECT tool_name, calls, failures, approved, rejected, last_used_at
        FROM cayu_tool_usage
        UNION ALL
        SELECT tool_name, calls, failures, approved, rejected, last_used_at
        FROM cayu_tool_usage_pending
    ) AS totals
    GROUP BY tool_name
    HAVING SUM(calls) > 0 OR SUM(failures) > 0 OR SUM(approved) > 0 OR SUM(rejected) > 0
"""

#: Whether a removed call leaves a Postgres tool row's newest call to recompute.
POSTGRES_TOOL_USAGE_RECHECK_SQL = """
    SELECT EXISTS (SELECT 1 FROM cayu_tool_usage_pending WHERE recheck)
        OR EXISTS (SELECT 1 FROM cayu_tool_usage WHERE recheck)
"""

STORED_TOOL_USAGE_TOTALS_SQL = """
    SELECT tool_name AS tool, SUM(calls) AS calls, SUM(failures) AS failures,
           SUM(approved) AS approved, SUM(rejected) AS rejected,
           MAX(last_used_at) AS last_used_at
    FROM cayu_tool_usage
    GROUP BY tool_name
"""


@dataclass(frozen=True)
class ToolUsageTotalsSql:
    newest: session_store_sql.EventQuerySqlPlan
    totals_sql: str
    totals_params: tuple[object, ...]


def _quoted(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def tool_usage_totals_sql(
    *,
    dialect: session_store_sql.SessionStoreSqlDialect,
    payload_text: Callable[[str], str],
    payload_is_string: Callable[[str], str],
) -> ToolUsageTotalsSql:
    """Build the newest-sequence plan and the grouped tally statement.

    ``payload_text(key)`` renders a payload member as SQL text and
    ``payload_is_string(key)`` tests that member is a JSON string. Both receive
    fixed keys, never caller input; access bounds come from the event query plan.
    """

    hosted = _quoted(str(EventType.MODEL_HOSTED_TOOL_CALL))
    completed = _quoted(str(EventType.TOOL_CALL_COMPLETED))
    failed = _quoted(str(EventType.TOOL_CALL_FAILED))
    approved = _quoted(str(EventType.TOOL_CALL_APPROVED))
    denied = _quoted(str(EventType.TOOL_CALL_APPROVAL_DENIED))
    statuses = ", ".join(_quoted(status) for status in _HOSTED_TERMINAL_STATUSES)
    counted = session_store_sql.SqlClause(
        f"""(
            (cayu_events.event_type <> {hosted} AND cayu_events.tool_name IS NOT NULL)
            OR (
                cayu_events.event_type = {hosted}
                AND {payload_is_string("tool_type")}
                AND {payload_text("tool_type")} = 'web_search'
                AND {payload_is_string("status")}
                AND {payload_text("status")} IN ({statuses})
            )
        )"""
    )
    plan = session_store_sql.build_event_query_sql(
        EventQuery(limit=1, event_types=TOOL_USAGE_EVENT_TYPES),
        dialect=dialect,
        extra_after_sequence_clauses=(counted,),
    )
    call = f"cayu_events.event_type IN ({completed}, {failed}, {hosted})"
    totals_sql = f"""
        SELECT
            CASE
                WHEN cayu_events.event_type = {hosted}
                THEN {_quoted(HOSTED_WEB_SEARCH_TOOL_NAME)}
                ELSE cayu_events.tool_name
            END AS tool,
            SUM(CASE WHEN {call} THEN 1 ELSE 0 END) AS calls,
            SUM(
                CASE
                    WHEN cayu_events.event_type = {failed} THEN 1
                    WHEN cayu_events.event_type = {hosted}
                     AND {payload_text("status")} = 'failed' THEN 1
                    ELSE 0
                END
            ) AS failures,
            SUM(CASE WHEN cayu_events.event_type = {approved} THEN 1 ELSE 0 END) AS approved,
            SUM(CASE WHEN cayu_events.event_type = {denied} THEN 1 ELSE 0 END) AS rejected,
            MAX(CASE WHEN {call} THEN cayu_events.timestamp END) AS last_used_at
        FROM cayu_usage_events AS cayu_events
        JOIN cayu_sessions ON cayu_sessions.id = cayu_events.session_id
        {plan.where_sql}
        GROUP BY 1
    """
    newest = session_store_sql.build_event_query_sql(EventQuery(limit=1), dialect=dialect)
    return ToolUsageTotalsSql(newest=newest, totals_sql=totals_sql, totals_params=plan.params)


def tool_call_heads_sql(
    *,
    dialect: session_store_sql.SessionStoreSqlDialect,
    raw_names: tuple[str, ...] | None,
    before_sequence: int | None,
    limit: int,
    payload_text: Callable[[str], str],
    payload_is_string: Callable[[str], str],
) -> tuple[str, tuple[object, ...]]:
    """The newest terminal tool calls, newest first, from ``cayu_usage_events``.

    With ``raw_names``, each name is one ``(tool_name, sequence)`` index range
    read newest first and only ``limit`` rows deep, so a rare tool costs as much
    as a common one. ``openai_web_search`` also reads hosted web searches. With
    ``None``, every call event is read newest first. ``payload_text`` and
    ``payload_is_string`` render the copy's payload members as in
    ``tool_usage_totals_sql``.
    """

    marker = dialect.placeholder
    completed = _quoted(str(EventType.TOOL_CALL_COMPLETED))
    failed = _quoted(str(EventType.TOOL_CALL_FAILED))
    hosted = _quoted(str(EventType.MODEL_HOSTED_TOOL_CALL))
    statuses = ", ".join(_quoted(status) for status in _HOSTED_TERMINAL_STATUSES)
    hosted_call = (
        f"(event_type = {hosted} AND {payload_is_string('tool_type')} "
        f"AND {payload_text('tool_type')} = 'web_search' AND {payload_is_string('status')} "
        f"AND {payload_text('status')} IN ({statuses}))"
    )
    columns = (
        "sequence, session_id, event_id, "
        f"CASE WHEN event_type = {hosted} THEN {_quoted(HOSTED_WEB_SEARCH_TOOL_NAME)} "
        f"ELSE tool_name END AS tool_name"
    )
    before = "" if before_sequence is None else f" AND sequence < {marker}"
    before_params: tuple[object, ...] = () if before_sequence is None else (before_sequence,)
    parts: list[str] = []
    params: list[object] = []
    if raw_names is None:
        # Read newest first by sequence and stop at the limit. Without statistics
        # SQLite would otherwise read every call through the event-type index and
        # sort them; the unary plus keeps it off that index.
        kind = "+event_type" if dialect.contains_style == "sqlite_nocase_like" else "event_type"
        parts.append(
            f"SELECT {columns} FROM cayu_usage_events "
            f"WHERE ({kind} IN ({completed}, {failed}) OR {hosted_call}){before} "
            f"ORDER BY sequence DESC LIMIT {marker}"
        )
        params.extend((*before_params, limit))
    else:
        for name in raw_names:
            parts.append(
                f"SELECT {columns} FROM cayu_usage_events "
                f"WHERE tool_name = {marker} AND event_type IN ({completed}, {failed}){before} "
                f"ORDER BY sequence DESC LIMIT {marker}"
            )
            params.extend((name, *before_params, limit))
        if HOSTED_WEB_SEARCH_TOOL_NAME in raw_names:
            parts.append(
                f"SELECT {columns} FROM cayu_usage_events "
                f"WHERE {hosted_call}{before} "
                f"ORDER BY sequence DESC LIMIT {marker}"
            )
            params.extend((*before_params, limit))
    if not parts:
        return "SELECT NULL WHERE 1 = 0", ()
    union = " UNION ALL ".join(
        f"SELECT * FROM ({part}) AS part_{index}" for index, part in enumerate(parts)
    )
    return (
        f"SELECT * FROM ({union}) AS heads ORDER BY sequence DESC LIMIT {marker}",
        (*params, limit),
    )

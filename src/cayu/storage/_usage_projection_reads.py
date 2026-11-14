"""Usage rollup statements over the stored usage totals, for SQLite and Postgres.

A rollup window rarely starts and ends on whole UTC hours, and a series in a
time zone with a half-hour offset cuts hours in two. Each statement therefore
reads two sources and adds them up in one snapshot:

- ``cayu_session_usage`` rows for the whole UTC hours inside the window that no
  bucket boundary cuts, and
- ``cayu_usage_events`` rows, one per event, for the rest of the window: the
  partial hours at its edges and every hour a bucket boundary cuts.

The statements return the same sections and columns as the event-level
statements in ``_sqlite_aggregates`` and ``_postgres_aggregates``, which stay
the reference: ``aggregate_session_usage_from_events`` there, and
``cayu usage rollup --recompute`` compares the two.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Any, Literal, Protocol

from cayu._time_buckets import time_buckets
from cayu.budgets.aggregates import (
    MAX_USAGE_COST_CANDIDATES,
    MAX_USAGE_PRICING_INPUT_BYTES,
    MAX_USAGE_PRICING_RAW_CANDIDATES,
    MAX_USAGE_ROLLUP_SESSION_ID_BYTES,
    BoundedUsagePricingInputAccumulator,
    _usage_pricing_event_type,
    _UsagePricingInputTooLarge,
    pricing_usage_metrics_from_event_payload,
)
from cayu.budgets.usage import UsageMetrics
from cayu.events import EventType
from cayu.sessions.usage import UsageRollupQuery
from cayu.storage._session_store_sql import SessionQuerySqlPlan
from cayu.storage._usage_projection_schema import COUNTER_COLUMNS


class _NamedRow(Protocol):
    def __getitem__(self, key: str, /) -> Any: ...


Dialect = Literal["sqlite", "postgres"]
_HOUR = timedelta(hours=1)
_TOKEN_COLUMNS = COUNTER_COLUMNS[:11]
_PRICING_COUNTERS = (
    *_TOKEN_COLUMNS,
    "metrics_web_search_calls",
    "metrics_web_search_outcome_unknown",
)


class StoredTotalsUnavailable(Exception):
    """The stored totals cannot answer this read exactly; use the event-level reference."""


def _floor_hour(value: datetime) -> datetime:
    return value.astimezone(UTC).replace(minute=0, second=0, microsecond=0)


def _ceil_hour(value: datetime) -> datetime:
    floor = _floor_hour(value)
    return floor if floor == value else floor + _HOUR


@dataclass(frozen=True, slots=True)
class WindowSplit:
    """Which part of a window the stored hours answer, and which the events do."""

    hours: tuple[datetime, datetime] | None
    cut_hours: tuple[datetime, ...]
    event_ranges: tuple[tuple[datetime, datetime], ...]


def split_window(start: datetime, end: datetime, cuts: Iterable[datetime] = ()) -> WindowSplit:
    """Split [start, end) into whole stored hours and event-level ranges.

    ``cuts`` are instants (series bucket boundaries) that must not fall inside a
    stored hour; an hour one cuts is read from events.
    """

    start, end = start.astimezone(UTC), end.astimezone(UTC)
    first, last = _ceil_hour(start), _floor_hour(end)
    if first >= last:
        return WindowSplit(None, (), ((start, end),))
    ranges: list[tuple[datetime, datetime]] = []
    if start < first:
        ranges.append((start, first))
    cut_hours = sorted(
        {
            _floor_hour(cut)
            for cut in cuts
            if _floor_hour(cut) != cut.astimezone(UTC) and first <= _floor_hour(cut) < last
        }
    )
    ranges.extend((hour, hour + _HOUR) for hour in cut_hours)
    if last < end:
        ranges.append((last, end))
    return WindowSplit((first, last), tuple(cut_hours), tuple(ranges))


class _Sql:
    """Dialect details of the projection statements."""

    def __init__(self, dialect: Dialect) -> None:
        self.dialect = dialect
        self.ph = "?" if dialect == "sqlite" else "%s"

    def time(self, value: datetime) -> object:
        value = value.astimezone(UTC)
        return value.isoformat() if self.dialect == "sqlite" else value

    def time_param(self) -> str:
        return "?" if self.dialect == "sqlite" else "%s::timestamptz"

    def date_of(self, column: str) -> str:
        if self.dialect == "sqlite":
            return f"substr({column}, 1, 10)"
        return f"({column} AT TIME ZONE 'UTC')::date"


def _event_columns(sql: _Sql) -> str:
    """Per-event usage columns of ``cayu_usage_events AS event`` with rollup include rules."""

    if sql.dialect == "sqlite":
        model_types = "event.event_type IN ('model.completed', 'model.auxiliary.attempt_settled')"
        hosted = "event.event_type = 'model.hosted_tool_call'"
        tokens = [
            f"CASE WHEN {model_types} THEN event.{column} ELSE 0 END AS {column}"
            for column in _TOKEN_COLUMNS
        ]
        searches = [
            f"CASE WHEN {hosted} THEN event.{column} ELSE 0 END AS {column}"
            for column in COUNTER_COLUMNS[11:]
        ]
        return ",\n            ".join(
            (
                "event.session_id AS session_id",
                "event.event_type AS event_type",
                "event.provider_name AS provider_name",
                "event.model AS model",
                "1 AS event_count",
                "event.has_usage AS has_usage_count",
                *tokens,
                *searches,
                "0 AS overflow",
                "event.timestamp AS at",
            )
        )
    from cayu.storage._usage_projection_schema import _postgres_projection_select

    return (
        ",\n            ".join(
            (
                "event.session_id AS session_id",
                "event.event_type AS event_type",
                "projected.provider_name AS provider_name",
                "projected.model AS model",
                "1 AS event_count",
                "projected.has_usage AS has_usage_count",
                *(f"projected.{column} AS {column}" for column in COUNTER_COLUMNS),
                "0 AS overflow",
                "event.timestamp AS at",
            )
        )
        + f"""
        FROM cayu_usage_events AS event
        CROSS JOIN LATERAL (
            SELECT {_postgres_projection_select()}
        ) AS projected"""
    )


_HOURLY_COLUMNS = (
    "hour_start",
    "agent_name",
    "environment_name",
    "event_type",
    "price_identity_id",
    "input_band",
    "provider_name",
    "model",
    "event_count",
    "has_usage_count",
    *COUNTER_COLUMNS,
    "metrics_web_search_calls",
    "metrics_web_search_outcome_unknown",
    "min_input_tokens",
    "max_input_tokens",
    "max_pricing_input_bytes",
    "overflow",
)


def _hourly_source(sql: _Sql) -> str:
    """``cayu_usage_hourly AS u``; on Postgres with the deltas not yet folded into it."""

    if sql.dialect == "sqlite":
        return "cayu_usage_hourly AS u"
    columns = ", ".join(_HOURLY_COLUMNS)
    key = _HOURLY_COLUMNS[:6]
    folded = ", ".join(
        (
            *key,
            "MIN(provider_name) AS provider_name",
            "MIN(model) AS model",
            *(
                f"SUM({column}) AS {column}"
                for column in (
                    "event_count",
                    "has_usage_count",
                    *COUNTER_COLUMNS,
                    "metrics_web_search_calls",
                    "metrics_web_search_outcome_unknown",
                )
            ),
            "MIN(min_input_tokens) AS min_input_tokens",
            "MAX(max_input_tokens) AS max_input_tokens",
            "MAX(max_pricing_input_bytes) AS max_pricing_input_bytes",
            "MAX(overflow) AS overflow",
        )
    )
    # A row and pending deltas that cancel it (its events were deleted after
    # the last compaction) fold to nothing, as the compacted table would hold.
    return (
        f"(SELECT {folded} FROM (SELECT {columns} FROM cayu_usage_hourly "
        f"UNION ALL SELECT {columns} FROM cayu_usage_hourly_pending) AS unfolded "
        f"GROUP BY {', '.join(key)} HAVING SUM(event_count) > 0) AS u"
    )


def unfiltered(session_plan: SessionQuerySqlPlan) -> bool:
    """Whether every session matches, so the hourly totals can answer the sums."""

    return not session_plan.filter_where_sql.strip() and not session_plan.filter_params


def _usage_rows(
    sql: _Sql, split: WindowSplit, *, hourly: bool = False, every_session: bool = False
) -> tuple[str, list[object]]:
    """The ``usage_rows`` CTE body and its parameters, after ``matched_sessions``.

    ``hourly`` reads the stored hours from ``cayu_usage_hourly``, whose rows
    carry no session (``session_id`` is NULL); ``_count_rows`` then names the
    sessions. ``every_session`` (no session filter) skips the join to
    ``matched_sessions``: usage rows and events go with their session.
    """

    matched = (
        "" if every_session else "\n        JOIN matched_sessions AS session ON session.id = {}"
    )

    params: list[object] = []
    parts: list[str] = []
    columns = ", ".join(
        (
            "NULL AS session_id" if hourly else "u.session_id",
            "u.event_type",
            "u.provider_name",
            "u.model",
            "u.event_count",
            "u.has_usage_count",
            *(f"u.{column}" for column in COUNTER_COLUMNS),
            "u.overflow",
            "u.hour_start AS at",
        )
    )
    if split.hours is not None:
        cut = ""
        if split.cut_hours:
            cut = (
                " AND u.hour_start NOT IN ("
                + ", ".join(sql.time_param() for _ in split.cut_hours)
                + ")"
            )
        source = (
            _hourly_source(sql)
            if hourly
            else "cayu_session_usage AS u" + matched.format("u.session_id")
        )
        parts.append(
            f"""SELECT {columns}
        FROM {source}
        WHERE u.hour_start >= {sql.time_param()} AND u.hour_start < {sql.time_param()}{cut}"""
        )
        params.extend((sql.time(split.hours[0]), sql.time(split.hours[1])))
        params.extend(sql.time(hour) for hour in split.cut_hours)
    if split.event_ranges:
        ranges = " OR ".join(
            f"(event.timestamp >= {sql.time_param()} AND event.timestamp < {sql.time_param()})"
            for _ in split.event_ranges
        )
        source = (
            f"SELECT {_event_columns(sql)} FROM cayu_usage_events AS event"
            if sql.dialect == "sqlite"
            else f"SELECT {_event_columns(sql)}"
        )
        eligible = (
            "event.usage_eligible = 1"
            if sql.dialect == "sqlite"
            else "projected.usage_eligible = 1"
        )
        parts.append(
            f"""{source}{matched.format("event.session_id")}
        WHERE ({ranges})
          AND event.event_type IN (
              'model.completed', 'model.auxiliary.attempt_settled',
              'tool.call.started', 'model.hosted_tool_call'
          )
          AND {eligible}"""
        )
        for low, high in split.event_ranges:
            params.extend((sql.time(low), sql.time(high)))
    if not parts:
        raise AssertionError("A window always has stored hours or event ranges.")
    return "\n        UNION ALL\n        ".join(parts), params


def _count_rows(sql: _Sql, split: WindowSplit, *, hourly: bool) -> tuple[str, list[object]]:
    """The ``count_rows`` CTE body: one row per session, type and identity that has usage.

    With ``hourly``, the stored hours come from the narrow
    ``idx_cayu_session_usage_hour`` index; otherwise from ``usage_rows``.
    """

    if not hourly:
        return (
            "SELECT session_id, event_type, provider_name, model, at FROM usage_rows",
            [],
        )
    params: list[object] = []
    parts: list[str] = []
    if split.hours is not None:
        cut = ""
        if split.cut_hours:
            cut = (
                " AND u.hour_start NOT IN ("
                + ", ".join(sql.time_param() for _ in split.cut_hours)
                + ")"
            )
        parts.append(
            "SELECT u.session_id, u.event_type, u.provider_name, u.model, u.hour_start AS at "
            "FROM cayu_session_usage AS u "
            f"WHERE u.hour_start >= {sql.time_param()} AND u.hour_start < {sql.time_param()}{cut}"
        )
        params.extend((sql.time(split.hours[0]), sql.time(split.hours[1])))
        params.extend(sql.time(hour) for hour in split.cut_hours)
    if split.event_ranges:
        parts.append(
            "SELECT session_id, event_type, provider_name, model, at FROM usage_rows "
            "WHERE session_id IS NOT NULL"
        )
    return " UNION ALL ".join(parts), params


def _sums(prefix: str = "") -> str:
    return ",\n        ".join(
        f"COALESCE(SUM({prefix}{column}), 0) AS {column}" for column in COUNTER_COLUMNS
    )


def _totals(prefix: str = "") -> str:
    return f"""SUM(CASE WHEN {prefix}event_type = 'model.completed' THEN {prefix}event_count ELSE 0 END)
            AS model_steps,
        SUM(CASE WHEN {prefix}event_type = 'model.completed' THEN {prefix}has_usage_count ELSE 0 END)
            AS model_steps_with_usage,
        {_sums(prefix)}"""


def _group_sections(dimension: str, keys: tuple[str, ...], order: str, *, same: str) -> str:
    key_list = ", ".join(keys)
    match = " AND ".join(f"row.{key} {same} ranked.{key}" for key in keys)
    joined = " AND ".join(f"sums.{key} {same} counts.{key}" for key in keys)
    sum_keys = ", ".join(f"sums.{key}" for key in keys)
    return f"""
{dimension}_sums AS (
    SELECT {key_list}, {_totals()}
    FROM usage_rows
    WHERE event_type <> 'tool.call.started'
    GROUP BY {key_list}
),
{dimension}_counts AS (
    SELECT {key_list}, COUNT(DISTINCT session_id) AS session_count
    FROM count_rows
    WHERE event_type <> 'tool.call.started'
    GROUP BY {key_list}
),
{dimension}_grouped AS (
    SELECT {sum_keys}, COALESCE(counts.session_count, 0) AS session_count,
           sums.model_steps, sums.model_steps_with_usage,
           {", ".join(f"sums.{column}" for column in COUNTER_COLUMNS)}
    FROM {dimension}_sums AS sums
    LEFT JOIN {dimension}_counts AS counts ON {joined}
),
{dimension}_ranked AS (
    SELECT *, ROW_NUMBER() OVER (ORDER BY {order}) AS group_rank,
           COUNT(*) OVER () AS total_groups
    FROM {dimension}_grouped
),
{dimension}_remainder_sums AS (
    SELECT {_totals("row.")},
           MAX(ranked.total_groups) - (SELECT group_limit FROM scope) AS group_count,
           COUNT(*) AS row_count
    FROM usage_rows AS row
    JOIN {dimension}_ranked AS ranked ON {match}
    WHERE row.event_type <> 'tool.call.started'
      AND ranked.group_rank > (SELECT group_limit FROM scope)
),
{dimension}_remainder AS (
    SELECT (
        SELECT COUNT(DISTINCT row.session_id)
        FROM count_rows AS row
        JOIN {dimension}_ranked AS ranked ON {match}
        WHERE row.event_type <> 'tool.call.started'
          AND ranked.group_rank > (SELECT group_limit FROM scope)
    ) AS session_count, sums.*
    FROM {dimension}_remainder_sums AS sums
    WHERE sums.row_count > 0
)"""


def _matched_sessions(where: str) -> str:
    return f"SELECT id, status FROM cayu_sessions {where}"


def _materialized(session_plan: SessionQuerySqlPlan) -> str:
    """How ``matched_sessions`` is declared.

    A filter is evaluated once, into a materialized set. Without one, every
    session matches, and a materialized copy would read every ``cayu_sessions``
    row (they carry kilobytes of metadata each); inlined, the joins and counts
    use the session indexes instead.
    """

    return "AS NOT MATERIALIZED" if unfiltered(session_plan) else "AS MATERIALIZED"


def _display_columns(sql: _Sql, alias: str) -> str:
    """A session's bounded agent name and title, read only for retained sessions."""

    if sql.dialect == "sqlite":
        title = (
            f"CASE WHEN json_type({alias}.metadata_json, '$.\"cayu:title\"') = 'text' "
            f"AND length(json_extract({alias}.metadata_json, '$.\"cayu:title\"')) <= 256 "
            f"THEN json_extract({alias}.metadata_json, '$.\"cayu:title\"') END"
        )
    else:
        title = (
            f"CASE WHEN jsonb_typeof({alias}.metadata -> 'cayu:title') = 'string' "
            f"AND length({alias}.metadata ->> 'cayu:title') <= 256 "
            f"THEN {alias}.metadata ->> 'cayu:title' END"
        )
    return (
        f"CASE WHEN length({alias}.agent_name) <= 256 THEN {alias}.agent_name END AS agent_name, "
        f"{title} AS title"
    )


def rollup_statement(
    dialect: Dialect, *, session_plan: SessionQuerySqlPlan, query: UsageRollupQuery
) -> tuple[str, tuple[object, ...]]:
    """Totals plus the provider and model breakdowns, in the reference's row shape."""

    sql = _Sql(dialect)
    split = split_window(query.start_at, query.end_at)
    hourly = unfiltered(session_plan)
    rows, row_params = _usage_rows(sql, split, hourly=hourly)
    counted, count_params = _count_rows(sql, split, hourly=hourly)
    zeros = ", ".join(f"overall.{column}" for column in COUNTER_COLUMNS)

    def section(name: str, source: str, provider: str, model: str, group_count: str) -> str:
        counters = ", ".join(COUNTER_COLUMNS)
        tail = (
            "NULL AS effective_on, 0 AS occurrences, NULL AS usage_metrics_json, "
            "NULL AS billing_identity_json, 0 AS matching_session_count, "
            "0 AS active_session_count, NULL AS as_of"
            if dialect == "sqlite"
            else "NULL AS effective_on, 0 AS occurrences, NULL AS usage_metrics, "
            "NULL AS billing_identity, 0 AS matching_session_count, 0 AS active_session_count"
        )
        return (
            f"SELECT '{name}' AS section, {provider} AS provider_name, {model} AS model, "
            f"session_count, model_steps, model_steps_with_usage, 0 AS tool_calls, {counters}, "
            f"{group_count} AS group_count, {tail}, 0 AS overflow FROM {source}"
        )

    as_of = "strftime('%Y-%m-%dT%H:%M:%fZ', 'now')"
    identity_names = (
        ("usage_metrics_json", "billing_identity_json")
        if dialect == "sqlite"
        else ("usage_metrics", "billing_identity")
    )
    overall_tail = (
        f"NULL AS effective_on, 0 AS occurrences, NULL AS {identity_names[0]}, "
        f"NULL AS {identity_names[1]}, "
        "(SELECT COUNT(*) FROM matched_sessions) AS matching_session_count, "
        "(SELECT COUNT(*) FROM matched_sessions WHERE status IN "
        "('pending', 'running', 'interrupting')) AS active_session_count"
        + (f", {as_of} AS as_of" if dialect == "sqlite" else "")
    )
    materialized = "AS MATERIALIZED"
    matched = _materialized(session_plan)
    same = "IS" if dialect == "sqlite" else "IS NOT DISTINCT FROM"
    statement = f"""
WITH
scope(group_limit) AS (SELECT {sql.ph}),
matched_sessions {matched} ({_matched_sessions(session_plan.filter_where_sql)}),
usage_rows {materialized} (
        {rows}
),
count_rows {materialized} (
        {counted}
),
overall AS (
    SELECT (SELECT COUNT(DISTINCT session_id) FROM count_rows) AS session_count,
        COALESCE(SUM(CASE WHEN event_type = 'model.completed' THEN event_count ELSE 0 END), 0)
            AS model_steps,
        COALESCE(SUM(CASE WHEN event_type = 'model.completed' THEN has_usage_count ELSE 0 END), 0)
            AS model_steps_with_usage,
        COALESCE(SUM(CASE WHEN event_type = 'tool.call.started' THEN event_count ELSE 0 END), 0)
            AS tool_calls,
        {_sums()},
        COALESCE(MAX(overflow), 0) AS overflow
    FROM usage_rows
),{_group_sections("provider", ("provider_name",), "total_tokens DESC, model_steps DESC, provider_name IS NULL, provider_name ASC", same=same)},{_group_sections("model", ("provider_name", "model"), "total_tokens DESC, model_steps DESC, provider_name IS NULL, provider_name ASC, model IS NULL, model ASC", same=same)}
SELECT 'overall' AS section, NULL AS provider_name, NULL AS model, session_count, model_steps,
       model_steps_with_usage, tool_calls, {zeros.replace("overall.", "")}, 0 AS group_count,
       {overall_tail}, overflow AS overflow
FROM overall
UNION ALL
{section("provider", "provider_ranked WHERE group_rank <= (SELECT group_limit FROM scope)", "provider_name", "NULL", "total_groups")}
UNION ALL
{section("provider_remainder", "provider_remainder", "NULL", "NULL", "group_count")}
UNION ALL
{section("model", "model_ranked WHERE group_rank <= (SELECT group_limit FROM scope)", "provider_name", "model", "total_groups")}
UNION ALL
{section("model_remainder", "model_remainder", "NULL", "NULL", "group_count")}
"""
    if dialect == "postgres":
        statement = statement.replace(
            "scope(group_limit) AS (SELECT %s)", "scope(group_limit) AS (SELECT %s::integer)"
        )
    return statement, (
        query.group_limit,
        *session_plan.filter_params,
        *row_params,
        *count_params,
    )


def series_statement(
    dialect: Dialect, *, session_plan: SessionQuerySqlPlan, query: UsageRollupQuery
) -> tuple[str, tuple[object, ...]]:
    if query.series is None:
        raise ValueError("A usage series must be requested.")
    sql = _Sql(dialect)
    buckets = time_buckets(query.start_at, query.end_at, query.series)
    boundaries = [bucket.end_at for bucket in buckets[:-1]]
    split = split_window(query.start_at, query.end_at, boundaries)
    hourly = unfiltered(session_plan)
    rows, row_params = _usage_rows(sql, split, hourly=hourly)
    counted, count_params = _count_rows(sql, split, hourly=hourly)
    cases = " ".join(
        f"WHEN at < {sql.time_param()} THEN {index}" for index in range(len(boundaries))
    )
    bucket = f"CASE {cases} ELSE {len(boundaries)} END" if cases else "0"
    statement = f"""
WITH
matched_sessions {_materialized(session_plan)} ({_matched_sessions(session_plan.filter_where_sql)}),
usage_rows AS MATERIALIZED (
        {rows}
),
count_rows AS MATERIALIZED (
        {counted}
),
bucketed AS (SELECT *, {bucket} AS bucket_index FROM usage_rows),
bucket_sessions AS (
    SELECT bucket_index, COUNT(DISTINCT session_id) AS session_count
    FROM (SELECT session_id, {bucket} AS bucket_index FROM count_rows) AS counted
    GROUP BY bucket_index
),
bucket_sums AS (
    SELECT bucket_index,
        COALESCE(SUM(CASE WHEN event_type = 'model.completed' THEN event_count ELSE 0 END), 0)
            AS model_steps,
        COALESCE(SUM(CASE WHEN event_type = 'model.completed' THEN has_usage_count ELSE 0 END), 0)
            AS model_steps_with_usage,
        COALESCE(SUM(CASE WHEN event_type = 'tool.call.started' THEN event_count ELSE 0 END), 0)
            AS tool_calls,
        {_sums()},
        COALESCE(MAX(overflow), 0) AS overflow
    FROM bucketed
    GROUP BY bucket_index
)
SELECT sums.bucket_index, COALESCE(counts.session_count, 0) AS session_count,
    sums.model_steps, sums.model_steps_with_usage, sums.tool_calls,
    {", ".join(f"sums.{column}" for column in COUNTER_COLUMNS)}, sums.overflow
FROM bucket_sums AS sums
LEFT JOIN bucket_sessions AS counts ON counts.bucket_index = sums.bucket_index
"""
    bucket_params = [sql.time(boundary) for boundary in boundaries]
    return statement, (
        *session_plan.filter_params,
        *row_params,
        *count_params,
        *bucket_params,
        *bucket_params,
    )


def session_breakdown_statement(
    dialect: Dialect, *, session_plan: SessionQuerySqlPlan, query: UsageRollupQuery
) -> tuple[str, tuple[object, ...]]:
    """The per-session ranking and its exact remainder, in the reference's row shape.

    Only sessions with usage in the window are grouped; the matched sessions
    without any rank after them by session id, as zero rows, so the first
    ``limit`` of them are enough, and the remainder is the overall totals less
    the retained rows.
    """

    limit = query.retained_session_limit
    if limit is None:
        raise ValueError("session_group_limit is required for a session breakdown.")
    sql = _Sql(dialect)
    split = split_window(query.start_at, query.end_at)
    rows, row_params = _usage_rows(sql, split, every_session=unfiltered(session_plan))
    zero_counters = ", ".join(f"0 AS {column}" for column in COUNTER_COLUMNS)
    counters = ", ".join(COUNTER_COLUMNS)
    differences = ",\n        ".join(
        f"COALESCE(total.{column}, 0) - COALESCE(kept.{column}, 0) AS {column}"
        for column in (
            "session_count",
            "model_steps",
            "model_steps_with_usage",
            "tool_calls",
            *COUNTER_COLUMNS,
        )
    )
    kept_sums = ",\n        ".join(
        f"SUM({column}) AS {column}"
        for column in (
            "session_count",
            "model_steps",
            "model_steps_with_usage",
            "tool_calls",
            *COUNTER_COLUMNS,
        )
    )
    limit_param = sql.ph if dialect == "sqlite" else "%s::integer"
    active = "status IN ('pending', 'running', 'interrupting')"
    active_value = f"CASE WHEN {active} THEN 1 ELSE 0 END"
    statement = f"""
WITH
scope(session_group_limit) AS (SELECT {limit_param}),
matched_sessions {_materialized(session_plan)} ({_matched_sessions(session_plan.filter_where_sql)}),
usage_rows AS MATERIALIZED (
        {rows}
),
session_grouped AS MATERIALIZED (
    SELECT session_id, 1 AS session_count,
        COALESCE(SUM(CASE WHEN event_type = 'model.completed' THEN event_count ELSE 0 END), 0)
            AS model_steps,
        COALESCE(SUM(CASE WHEN event_type = 'model.completed' THEN has_usage_count ELSE 0 END), 0)
            AS model_steps_with_usage,
        COALESCE(SUM(CASE WHEN event_type = 'tool.call.started' THEN event_count ELSE 0 END), 0)
            AS tool_calls,
        {_sums()},
        MAX(overflow) AS overflow
    FROM usage_rows
    GROUP BY session_id
),
candidates AS (
    SELECT grouped.session_id,
           grouped.session_count, grouped.model_steps, grouped.model_steps_with_usage,
           grouped.tool_calls, {", ".join(f"grouped.{column}" for column in COUNTER_COLUMNS)}
    FROM session_grouped AS grouped
    UNION ALL
    SELECT idle.id, 0, 0, 0, 0, {zero_counters}
    FROM (
        SELECT id
        FROM matched_sessions AS session
        WHERE NOT EXISTS (
            SELECT 1 FROM session_grouped AS grouped WHERE grouped.session_id = session.id
        )
        ORDER BY id ASC
        LIMIT (SELECT session_group_limit FROM scope)
    ) AS idle
),
ranked AS MATERIALIZED (
    SELECT * FROM candidates
    ORDER BY total_tokens DESC, model_steps DESC, session_id ASC
    LIMIT (SELECT session_group_limit FROM scope)
),
-- Status, agent name and title are read only for the retained sessions.
retained AS MATERIALIZED (
    SELECT ranked.*, session.status,
           {active_value.replace("status", "session.status")} AS active,
           {_display_columns(sql, "session")}
    FROM ranked
    JOIN cayu_sessions AS session ON session.id = ranked.session_id
),
total AS (
    SELECT COUNT(*) AS session_count, {", ".join(f"SUM({c}) AS {c}" for c in ("model_steps", "model_steps_with_usage", "tool_calls", *COUNTER_COLUMNS))}
    FROM session_grouped
),
kept AS (
    SELECT {kept_sums}, COUNT(*) AS group_count, SUM(active) AS active_session_count
    FROM retained
),
remainder AS (
    SELECT {differences},
        (SELECT COUNT(*) FROM matched_sessions) - kept.group_count AS group_count,
        (SELECT COUNT(*) FROM matched_sessions WHERE {active})
            - COALESCE(kept.active_session_count, 0) AS active_session_count
    FROM total CROSS JOIN kept
    WHERE (SELECT COUNT(*) FROM matched_sessions) > kept.group_count
),
identity_bound AS (
    SELECT EXISTS (
        SELECT 1 FROM retained
        WHERE {"length(CAST(session_id AS BLOB))" if dialect == "sqlite" else "octet_length(session_id)"}
              > {MAX_USAGE_ROLLUP_SESSION_ID_BYTES}
    ) AS exceeded
)
SELECT 'session' AS section,
    CASE WHEN {"length(CAST(session_id AS BLOB))" if dialect == "sqlite" else "octet_length(session_id)"}
              <= {MAX_USAGE_ROLLUP_SESSION_ID_BYTES} THEN session_id END AS session_id,
    (SELECT exceeded FROM identity_bound) AS session_id_too_large,
    status, active, session_count, model_steps, model_steps_with_usage, tool_calls, {counters},
    0 AS group_count, 0 AS active_session_count, agent_name, title,
    (SELECT COALESCE(MAX(overflow), 0) FROM session_grouped) AS overflow
FROM retained
UNION ALL
SELECT 'session_remainder', NULL, (SELECT exceeded FROM identity_bound), NULL, NULL,
    session_count, model_steps, model_steps_with_usage, tool_calls, {counters},
    group_count, active_session_count, NULL, NULL,
    (SELECT COALESCE(MAX(overflow), 0) FROM session_grouped)
FROM remainder
"""
    return statement, (limit, *session_plan.filter_params, *row_params)


def pricing_statement(
    dialect: Dialect,
    *,
    session_plan: SessionQuerySqlPlan,
    hours: tuple[datetime, datetime],
    session_ids: tuple[str, ...] | None = None,
) -> tuple[str, tuple[object, ...]]:
    """Price groups of the stored whole hours, one row per group (and session)."""

    sql = _Sql(dialect)
    session_column = "u.session_id" if session_ids is not None else "NULL"
    requested = ""
    params: list[object] = [*session_plan.filter_params]
    if session_ids is not None:
        if not session_ids or len(session_ids) > MAX_USAGE_COST_CANDIDATES:
            raise ValueError("session_ids must hold 1 to 1000 values.")
        requested = " AND u.session_id IN (" + ", ".join(sql.ph for _ in session_ids) + ")"
    sums = ", ".join(f"SUM(u.{column}) AS {column}" for column in _PRICING_COUNTERS)
    metrics, billing = (
        ("identity.usage_metrics_json", "identity.billing_identity_json")
        if dialect == "sqlite"
        else ("identity.usage_metrics", "identity.billing_identity")
    )
    group = [
        "u.event_type",
        sql.date_of("u.hour_start"),
        *(("u.session_id",) if session_ids is not None else ()),
        "u.price_identity_id",
        "u.input_band",
        metrics,
        billing,
    ]
    hourly = session_ids is None and unfiltered(session_plan)
    source = (
        _hourly_source(sql)
        if hourly
        else "cayu_session_usage AS u\nJOIN matched_sessions AS session ON session.id = u.session_id"
    )
    statement = f"""
WITH matched_sessions {_materialized(session_plan)} ({_matched_sessions(session_plan.filter_where_sql)})
SELECT u.event_type AS event_type, {sql.date_of("u.hour_start")} AS effective_on,
       {session_column} AS session_id, u.price_identity_id AS price_identity_id,
       SUM(u.event_count) AS occurrences, {sums},
       MIN(u.min_input_tokens) AS min_input_tokens, MAX(u.max_input_tokens) AS max_input_tokens,
       MAX(u.max_pricing_input_bytes) AS max_pricing_input_bytes, MAX(u.overflow) AS overflow,
       {metrics} AS usage_metrics, {billing} AS billing_identity
FROM {source}
LEFT JOIN cayu_usage_price_identities AS identity ON identity.id = u.price_identity_id
WHERE u.hour_start >= {sql.time_param()} AND u.hour_start < {sql.time_param()}
  AND u.event_type IN ('model.completed', 'model.auxiliary.attempt_settled', 'model.hosted_tool_call'){requested}
GROUP BY {", ".join(group)}
HAVING SUM(u.event_count) > 0
"""
    params.extend((sql.time(hours[0]), sql.time(hours[1])))
    if session_ids is not None:
        params.extend(session_ids)
    return statement, tuple(params)


def _json_value(value: str | dict[str, Any] | None) -> object:
    if value is None or isinstance(value, dict):
        return value
    return json.loads(value)


def add_stored_pricing_rows(
    pricing: BoundedUsagePricingInputAccumulator,
    rows: Iterable[Mapping[str, Any]],
    *,
    with_session: bool,
) -> None:
    """Fold stored price-group rows into ``pricing``.

    Raises ``StoredTotalsUnavailable`` when a row spans one of the request's
    tier boundaries or saturated a counter, so the caller prices from events.
    """

    rows = list(rows)
    if len(rows) > MAX_USAGE_PRICING_RAW_CANDIDATES:
        pricing.reject_candidate_row_overflow(limit=MAX_USAGE_PRICING_RAW_CANDIDATES)
        return
    parsed: dict[tuple[EventType, object], UsageMetrics | None] = {}
    for row in rows:
        if pricing.truncated:
            return
        if int(row["overflow"] or 0):
            raise StoredTotalsUnavailable("A stored usage counter saturated.")
        session_id = row["session_id"] if with_session else None
        size = int(row["max_pricing_input_bytes"]) + (
            len(session_id.encode("utf-8")) if session_id is not None else 0
        )
        if size > MAX_USAGE_PRICING_INPUT_BYTES:
            pricing.reject_oversized_candidate()
            return
        event_type = EventType(row["event_type"])
        # Rows of one price identity share their metrics; parse them once.
        identity = (event_type, row["price_identity_id"])
        if identity in parsed:
            metrics = parsed[identity]
        else:
            payload: dict[str, object] = {}
            metrics_value = _json_value(row["usage_metrics"])
            billing_value = _json_value(row["billing_identity"])
            if metrics_value is not None:
                payload["usage_metrics"] = metrics_value
            if billing_value is not None:
                payload["billing_identity"] = billing_value
            try:
                metrics = (
                    None
                    if not payload
                    else pricing_usage_metrics_from_event_payload(
                        payload, event_type=event_type, max_bytes=pricing.max_bytes
                    )
                )
            except _UsagePricingInputTooLarge:
                pricing.reject_oversized_candidate()
                return
            parsed[identity] = metrics
        low, high = int(row["min_input_tokens"]), int(row["max_input_tokens"])
        if metrics is not None and pricing.spans_tier(low, high):
            raise StoredTotalsUnavailable("A stored price group spans a tier boundary.")
        effective_on = row["effective_on"]
        if not isinstance(effective_on, date):
            effective_on = date.fromisoformat(str(effective_on))
        pricing.add_metrics_totals(
            event_type=_usage_pricing_event_type(event_type),
            session_id=session_id,
            effective_on=effective_on,
            occurrences=int(row["occurrences"]),
            metrics=metrics,
            counters=tuple(int(row[column]) for column in _PRICING_COUNTERS),
            min_input_tokens=low,
            max_input_tokens=high,
        )


def stored_hours(query: UsageRollupQuery) -> WindowSplit:
    """The pricing split: stored whole hours plus the window's partial edges."""

    return split_window(query.start_at, query.end_at)


def edge_queries(query: UsageRollupQuery, split: WindowSplit) -> tuple[UsageRollupQuery, ...]:
    """Copies of ``query`` for each event-level range, for the reference statements."""

    return tuple(
        query.model_copy(update={"start_at": low, "end_at": high, "series": None})
        for low, high in split.event_ranges
    )


def row_overflowed(rows: Iterable[Mapping[str, Any]]) -> bool:
    return any(int(row["overflow"] or 0) for row in rows)


__all__ = [
    "StoredTotalsUnavailable",
    "WindowSplit",
    "add_stored_pricing_rows",
    "edge_queries",
    "pricing_statement",
    "rollup_statement",
    "row_overflowed",
    "series_statement",
    "session_breakdown_statement",
    "split_window",
    "stored_hours",
]


_SESSION_TOTAL_COLUMNS = (
    *_TOKEN_COLUMNS,
    "metrics_web_search_calls",
    "metrics_web_search_outcome_unknown",
)


def session_totals_statement(
    dialect: Dialect,
    *,
    session_ids: tuple[str, ...],
    since: datetime | None,
    until: datetime | None,
) -> tuple[str, tuple[object, ...]]:
    """Per-session usage rows, by event type and identity, for ``SessionUsageSummary``.

    Without a window, every stored row of the sessions; with one, the stored
    whole hours plus the window's partial edges read from events.
    """

    if not session_ids or len(session_ids) > MAX_USAGE_COST_CANDIDATES:
        raise ValueError("session_ids must hold 1 to 1000 values.")
    sql = _Sql(dialect)
    requested = (
        "requested(session_id) AS (VALUES " + ", ".join("(?)" for _ in session_ids) + ")"
        if dialect == "sqlite"
        else "requested(session_id) AS (SELECT unnest(%s::text[]))"
    )
    params: list[object] = list(session_ids) if dialect == "sqlite" else [list(session_ids)]
    stored = ", ".join(
        (
            "u.session_id",
            "u.event_type",
            "u.provider_name",
            "u.model",
            "u.event_count",
            "u.has_usage_count",
            "u.legacy_count",
            *(f"u.{column}" for column in _SESSION_TOTAL_COLUMNS),
            "u.first_sequence",
            "u.first_sequence_stale",
            "u.overflow",
        )
    )
    parts = [
        f"SELECT {stored} FROM cayu_session_usage AS u "
        "JOIN requested ON requested.session_id = u.session_id"
    ]
    split = None
    if since is not None or until is not None:
        start = since or datetime(1970, 1, 1, tzinfo=UTC)
        end = until or datetime(9999, 1, 1, tzinfo=UTC)
        split = split_window(start, end)
        if split.hours is None:
            parts = []
        else:
            parts[0] += (
                f" WHERE u.hour_start >= {sql.time_param()} AND u.hour_start < {sql.time_param()}"
            )
            params.extend((sql.time(split.hours[0]), sql.time(split.hours[1])))
    if split is not None and split.event_ranges:
        ranges = " OR ".join(
            f"(x.timestamp >= {sql.time_param()} AND x.timestamp < {sql.time_param()})"
            for _ in split.event_ranges
        )
        for low, high in split.event_ranges:
            params.extend((sql.time(low), sql.time(high)))
        delta_columns = ", ".join(
            (
                "d.session_id AS session_id",
                "d.event_type AS event_type",
                "d.provider_name AS provider_name",
                "d.model AS model",
                "1 AS event_count",
                ("d.has_usage_count" if dialect == "sqlite" else "d.has_usage")
                + " AS has_usage_count",
                ("d.legacy_count" if dialect == "sqlite" else "d.legacy") + " AS legacy_count",
                *(f"d.{column} AS {column}" for column in _SESSION_TOTAL_COLUMNS),
                "d.sequence AS first_sequence",
                "0 AS first_sequence_stale",
                "0 AS overflow",
            )
        )
        if dialect == "sqlite":
            from cayu.storage._usage_projection_schema import (
                _SQLITE_EVENT_COLUMNS,
                sqlite_usage_delta,
            )

            source = (
                "SELECT "
                + ", ".join(f"x.{column} AS {column}" for column in _SQLITE_EVENT_COLUMNS)
                + " FROM cayu_usage_events AS x JOIN requested ON requested.session_id = x.session_id"
                + f" WHERE {ranges}"
            )
            parts.append(f"SELECT {delta_columns} FROM ({sqlite_usage_delta(source)}) AS d")
        else:
            parts.append(
                f"""SELECT {delta_columns} FROM (
                    SELECT x.session_id, x.event_type, x.sequence, p.*
                    FROM cayu_usage_events AS x
                    JOIN requested ON requested.session_id = x.session_id
                    CROSS JOIN LATERAL cayu_usage_event_projection(x) AS p
                    WHERE ({ranges})
                      AND x.event_type IN ('model.completed', 'model.auxiliary.attempt_settled',
                                           'model.hosted_tool_call', 'tool.call.started')
                      AND p.usage_eligible = 1
                ) AS d"""
            )
    if not parts:
        return "", ()
    sums = ", ".join(f"SUM({column}) AS {column}" for column in _SESSION_TOTAL_COLUMNS)
    statement = f"""
WITH {requested},
usage_rows AS (
    {" UNION ALL ".join(parts)}
)
SELECT session_id, event_type, provider_name, model, SUM(event_count) AS event_count,
       SUM(has_usage_count) AS has_usage_count, SUM(legacy_count) AS legacy_count, {sums},
       MIN(first_sequence) AS first_sequence,
       MAX(first_sequence_stale) AS first_sequence_stale, MAX(overflow) AS overflow
FROM usage_rows
GROUP BY session_id, event_type, provider_name, model
"""
    return statement, tuple(params)


def session_summaries_from_rows(
    rows: Iterable[_NamedRow],
) -> tuple[dict[str, Any], set[str]]:
    """``SessionUsageSummary`` per session, and the sessions to read from events instead.

    A session needs its events when a row saturated a counter, or when it has a
    model step without normalized metrics that the reference may normalize from
    its raw usage.
    """

    from cayu.budgets.usage import SessionUsageSummary, build_aggregate_usage_metrics

    grouped: dict[str, list[_NamedRow]] = {}
    for row in rows:
        grouped.setdefault(str(row["session_id"]), []).append(row)
    summaries: dict[str, Any] = {}
    fallback: set[str] = set()
    for session_id, session_rows in grouped.items():
        # Saturated counters, legacy steps, and a removed first event (whose
        # successor the rows don't know) are read from events.
        if any(
            int(row["overflow"] or 0)
            or int(row["legacy_count"] or 0)
            or int(row["first_sequence_stale"] or 0)
            for row in session_rows
        ):
            fallback.add(session_id)
            continue
        model_rows = [
            row
            for row in session_rows
            if row["event_type"] in ("model.completed", "model.auxiliary.attempt_settled")
        ]
        totals = {
            column: sum(int(row[column]) for row in session_rows)
            for column in _SESSION_TOTAL_COLUMNS
        }

        def first_seen(column: str, session_rows=session_rows) -> list[str]:
            first: dict[str, int] = {}
            for row in session_rows:
                value = row[column]
                if value is None:
                    continue
                sequence = int(row["first_sequence"])
                if value not in first or sequence < first[value]:
                    first[value] = sequence
            return sorted(first, key=lambda value: first[value])

        summaries[session_id] = SessionUsageSummary(
            session_id=session_id,
            model_steps=sum(
                int(row["event_count"])
                for row in session_rows
                if row["event_type"] == "model.completed"
            ),
            unmeasured_model_attempts=sum(
                int(row["event_count"]) - int(row["has_usage_count"]) for row in model_rows
            ),
            tool_calls=sum(
                int(row["event_count"])
                for row in session_rows
                if row["event_type"] == "tool.call.started"
            ),
            provider_names=first_seen("provider_name"),
            models=first_seen("model"),
            usage=build_aggregate_usage_metrics(
                input_tokens=totals["input_tokens"],
                output_tokens=totals["output_tokens"],
                total_tokens=totals["total_tokens"],
                reasoning_output_tokens=totals["reasoning_output_tokens"],
                cache_read_tokens=totals["cache_read_tokens"],
                cache_write_tokens=totals["cache_write_tokens"],
                cache_write_5m_tokens=totals["cache_write_5m_tokens"],
                cache_write_1h_tokens=totals["cache_write_1h_tokens"],
                cache_write_unknown_ttl_tokens=totals["cache_write_unknown_ttl_tokens"],
                cached_input_tokens=totals["cached_input_tokens"],
                uncached_input_tokens=totals["uncached_input_tokens"],
                web_search_calls=totals["metrics_web_search_calls"],
                web_search_outcome_unknown=totals["metrics_web_search_outcome_unknown"],
            ),
        )
    return summaries, fallback

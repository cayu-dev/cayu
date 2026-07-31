from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from typing import Any

from cayu.runtime.aggregates import (
    _BEDROCK_AGGREGATE_COMPLETION_EVIDENCE,
    _BEDROCK_AGGREGATE_REQUEST_EVIDENCE,
    AGGREGATE_IDENTITY_TRIM_CHARACTERS,
    EXACT_AGGREGATE,
    MAX_AGGREGATE_USAGE_COUNTER,
    MAX_USAGE_PRICING_INPUT_BYTES,
    MAX_USAGE_PRICING_RAW_CANDIDATES,
    MAX_USAGE_ROLLUP_SESSION_ID_BYTES,
    AggregateAccuracy,
    AggregateAccuracyKind,
    BoundedUsagePricingInputAccumulator,
    UsageAggregateBreakdown,
    UsageAggregateGroup,
    UsageAggregateRemainder,
    UsageAggregateTotals,
    UsagePricingInput,
    UsageRollupResultTooLarge,
    UsageRollupStoreResult,
    UsageSessionAggregateBreakdown,
    UsageSessionAggregateGroup,
    UsageSessionAggregateRemainder,
    aggregate_identity_value,
    build_aggregate_usage_metrics,
)
from cayu.runtime.sessions import UsageRollupQuery
from cayu.storage._session_store_sql import SessionQuerySqlPlan

_RESULT_SQL = """
WITH
scope(group_limit, as_of, identity_trim) AS (
    SELECT ?, strftime('%Y-%m-%dT%H:%M:%fZ', 'now'), ?
),
matched_sessions AS MATERIALIZED (
    SELECT id, status
    FROM cayu_sessions
    {session_where_sql}
),
usage_events AS MATERIALIZED (
    SELECT
        event.session_id,
        event.event_type,
        event.timestamp,
        CASE
            WHEN json_type(
                     event.payload_json,
                     '$.usage_normalization_failed'
                 ) IS NOT 'true'
             AND json_type(event.payload_json, '$.usage_metrics') = 'object'
            THEN 1
            ELSE 0
        END AS has_usage,
        CASE
            WHEN json_type(
                     event.payload_json,
                     '$.usage_normalization_failed'
                 ) IS NOT 'true'
             AND json_type(event.payload_json, '$.usage_metrics.provider_name') = 'text'
             AND trim(
                 json_extract(event.payload_json, '$.usage_metrics.provider_name'),
                 (SELECT identity_trim FROM scope)
             ) = json_extract(event.payload_json, '$.usage_metrics.provider_name')
             AND json_extract(event.payload_json, '$.usage_metrics.provider_name') <> ''
            THEN json_extract(event.payload_json, '$.usage_metrics.provider_name')
        END AS provider_name,
        CASE
            WHEN json_type(
                     event.payload_json,
                     '$.usage_normalization_failed'
                 ) IS NOT 'true'
             AND json_type(event.payload_json, '$.usage_metrics.model') = 'text'
             AND trim(
                 json_extract(event.payload_json, '$.usage_metrics.model'),
                 (SELECT identity_trim FROM scope)
             ) = json_extract(event.payload_json, '$.usage_metrics.model')
             AND json_extract(event.payload_json, '$.usage_metrics.model') <> ''
            THEN json_extract(event.payload_json, '$.usage_metrics.model')
        END AS model,
        {input_tokens} AS input_tokens,
        {output_tokens} AS output_tokens,
        {total_tokens} AS total_tokens,
        {reasoning_tokens} AS reasoning_output_tokens,
        {cache_read_tokens} AS cache_read_tokens,
        {cache_write_tokens} AS cache_write_tokens,
        {cache_write_5m_tokens} AS cache_write_5m_tokens,
        {cache_write_1h_tokens} AS cache_write_1h_tokens,
        {cache_write_unknown_tokens} AS cache_write_unknown_ttl_tokens,
        {cached_input_tokens} AS cached_input_tokens,
        {uncached_input_tokens} AS uncached_input_tokens
    FROM cayu_events AS event
    JOIN matched_sessions AS session ON session.id = event.session_id
    CROSS JOIN scope
    WHERE event.timestamp >= ?
      AND event.timestamp < ?
      AND event.event_type IN ('model.completed', 'tool.call.started')
),
overall_grouped AS (
    SELECT
        COUNT(DISTINCT session_id) AS session_count,
        COALESCE(SUM(event_type = 'model.completed'), 0) AS model_steps,
        COALESCE(SUM(event_type = 'model.completed' AND has_usage = 1), 0)
            AS model_steps_with_usage,
        COALESCE(SUM(event_type = 'tool.call.started'), 0) AS tool_calls,
        {overall_usage_sum} AS usage_sums
    FROM usage_events
),
overall AS (
    SELECT
        session_count, model_steps, model_steps_with_usage, tool_calls,
        {usage_sum_projection}
    FROM overall_grouped
),
provider_grouped_raw AS (
    SELECT
        provider_name,
        COUNT(DISTINCT session_id) AS session_count,
        COUNT(*) AS model_steps,
        SUM(has_usage) AS model_steps_with_usage,
        {group_usage_sum} AS usage_sums
    FROM usage_events
    WHERE event_type = 'model.completed'
    GROUP BY provider_name
),
provider_grouped AS (
    SELECT
        provider_name, session_count, model_steps, model_steps_with_usage,
        {usage_sum_projection}
    FROM provider_grouped_raw
),
provider_ranked AS (
    SELECT
        *,
        ROW_NUMBER() OVER (
            ORDER BY total_tokens DESC, model_steps DESC,
                     provider_name IS NULL, provider_name ASC
        ) AS group_rank,
        COUNT(*) OVER () AS total_groups
    FROM provider_grouped
),
provider_remainder_raw AS (
    SELECT
        COUNT(DISTINCT event.session_id) AS session_count,
        COUNT(*) AS model_steps,
        SUM(event.has_usage) AS model_steps_with_usage,
        {event_usage_sum} AS usage_sums,
        MAX(ranked.total_groups) - (SELECT group_limit FROM scope) AS group_count
    FROM usage_events AS event
    JOIN provider_ranked AS ranked ON event.provider_name IS ranked.provider_name
    WHERE event.event_type = 'model.completed'
      AND ranked.group_rank > (SELECT group_limit FROM scope)
    HAVING COUNT(*) > 0
),
provider_remainder AS (
    SELECT
        session_count, model_steps, model_steps_with_usage,
        {usage_sum_projection}, group_count
    FROM provider_remainder_raw
),
model_grouped_raw AS (
    SELECT
        provider_name,
        model,
        COUNT(DISTINCT session_id) AS session_count,
        COUNT(*) AS model_steps,
        SUM(has_usage) AS model_steps_with_usage,
        {group_usage_sum} AS usage_sums
    FROM usage_events
    WHERE event_type = 'model.completed'
    GROUP BY provider_name, model
),
model_grouped AS (
    SELECT
        provider_name, model, session_count, model_steps, model_steps_with_usage,
        {usage_sum_projection}
    FROM model_grouped_raw
),
model_ranked AS (
    SELECT
        *,
        ROW_NUMBER() OVER (
            ORDER BY total_tokens DESC, model_steps DESC,
                     provider_name IS NULL, provider_name ASC,
                     model IS NULL, model ASC
        ) AS group_rank,
        COUNT(*) OVER () AS total_groups
    FROM model_grouped
),
model_remainder_raw AS (
    SELECT
        COUNT(DISTINCT event.session_id) AS session_count,
        COUNT(*) AS model_steps,
        SUM(event.has_usage) AS model_steps_with_usage,
        {event_usage_sum} AS usage_sums,
        MAX(ranked.total_groups) - (SELECT group_limit FROM scope) AS group_count
    FROM usage_events AS event
    JOIN model_ranked AS ranked
      ON event.provider_name IS ranked.provider_name
     AND event.model IS ranked.model
    WHERE event.event_type = 'model.completed'
      AND ranked.group_rank > (SELECT group_limit FROM scope)
    HAVING COUNT(*) > 0
),
model_remainder AS (
    SELECT
        session_count, model_steps, model_steps_with_usage,
        {usage_sum_projection}, group_count
    FROM model_remainder_raw
)
SELECT
    'overall' AS section,
    NULL AS provider_name,
    NULL AS model,
    overall.session_count,
    overall.model_steps,
    overall.model_steps_with_usage,
    overall.tool_calls,
    overall.input_tokens,
    overall.output_tokens,
    overall.total_tokens,
    overall.reasoning_output_tokens,
    overall.cache_read_tokens,
    overall.cache_write_tokens,
    overall.cache_write_5m_tokens,
    overall.cache_write_1h_tokens,
    overall.cache_write_unknown_ttl_tokens,
    overall.cached_input_tokens,
    overall.uncached_input_tokens,
    0 AS group_count,
    NULL AS effective_on,
    0 AS occurrences,
    NULL AS usage_metrics_json,
    NULL AS billing_identity_json,
    (SELECT COUNT(*) FROM matched_sessions) AS matching_session_count,
    (
        SELECT COUNT(*)
        FROM matched_sessions
        WHERE status IN ('pending', 'running', 'interrupting')
    ) AS active_session_count,
    (SELECT as_of FROM scope) AS as_of
FROM overall
UNION ALL
SELECT
    'provider', provider_name, NULL, session_count, model_steps,
    model_steps_with_usage, 0, input_tokens, output_tokens, total_tokens,
    reasoning_output_tokens, cache_read_tokens, cache_write_tokens,
    cache_write_5m_tokens, cache_write_1h_tokens, cache_write_unknown_ttl_tokens,
    cached_input_tokens, uncached_input_tokens, total_groups, NULL, 0, NULL, NULL, 0, 0, NULL
FROM provider_ranked
WHERE group_rank <= (SELECT group_limit FROM scope)
UNION ALL
SELECT
    'provider_remainder', NULL, NULL, session_count, model_steps,
    model_steps_with_usage, 0, input_tokens, output_tokens, total_tokens,
    reasoning_output_tokens, cache_read_tokens, cache_write_tokens,
    cache_write_5m_tokens, cache_write_1h_tokens, cache_write_unknown_ttl_tokens,
    cached_input_tokens, uncached_input_tokens, group_count, NULL, 0, NULL, NULL, 0, 0, NULL
FROM provider_remainder
UNION ALL
SELECT
    'model', provider_name, model, session_count, model_steps,
    model_steps_with_usage, 0, input_tokens, output_tokens, total_tokens,
    reasoning_output_tokens, cache_read_tokens, cache_write_tokens,
    cache_write_5m_tokens, cache_write_1h_tokens, cache_write_unknown_ttl_tokens,
    cached_input_tokens, uncached_input_tokens, total_groups, NULL, 0, NULL, NULL, 0, 0, NULL
FROM model_ranked
WHERE group_rank <= (SELECT group_limit FROM scope)
UNION ALL
SELECT
    'model_remainder', NULL, NULL, session_count, model_steps,
    model_steps_with_usage, 0, input_tokens, output_tokens, total_tokens,
    reasoning_output_tokens, cache_read_tokens, cache_write_tokens,
    cache_write_5m_tokens, cache_write_1h_tokens, cache_write_unknown_ttl_tokens,
    cached_input_tokens, uncached_input_tokens, group_count, NULL, 0, NULL, NULL, 0, 0, NULL
FROM model_remainder
"""

_PRICING_INPUT_SQL = """
WITH
scope(max_input_bytes, identity_trim) AS (
    SELECT ?, ?
),
matched_sessions AS MATERIALIZED (
    SELECT id
    FROM cayu_sessions
    {session_where_sql}
),
pricing_candidates AS (
    SELECT
        substr(event.timestamp, 1, 10) AS effective_on,
        {usage_metrics_projection} AS usage_metrics_json,
        {billing_identity_projection} AS billing_identity_json
    FROM cayu_events AS event
    JOIN matched_sessions AS session ON session.id = event.session_id
    WHERE event.timestamp >= ?
      AND event.timestamp < ?
      AND event.event_type = 'model.completed'
),
measured_candidates AS (
    SELECT
        *,
        COALESCE(length(CAST(usage_metrics_json AS BLOB)), 0)
            + COALESCE(length(CAST(billing_identity_json AS BLOB)), 0) AS input_bytes
    FROM pricing_candidates
),
bounded_candidates AS (
    SELECT
        effective_on,
        input_bytes > (SELECT max_input_bytes FROM scope) AS input_oversized,
        CASE
            WHEN input_bytes <= (SELECT max_input_bytes FROM scope)
            THEN usage_metrics_json
        END AS usage_metrics_json,
        CASE
            WHEN input_bytes <= (SELECT max_input_bytes FROM scope)
            THEN billing_identity_json
        END AS billing_identity_json
    FROM measured_candidates
)
SELECT
    effective_on,
    usage_metrics_json,
    billing_identity_json,
    input_oversized,
    occurrences,
    COUNT(*) OVER () AS raw_group_count
FROM (
    SELECT
        effective_on,
        usage_metrics_json,
        billing_identity_json,
        input_oversized,
        COUNT(*) AS occurrences
    FROM bounded_candidates
    GROUP BY effective_on, usage_metrics_json, billing_identity_json, input_oversized
)
LIMIT ?
"""

_SESSION_PRICING_INPUT_SQL = """
WITH
scope(max_input_bytes, identity_trim) AS (
    SELECT ?, ?
),
requested_sessions(session_id) AS (
    VALUES {requested_session_values}
),
matched_sessions AS MATERIALIZED (
    SELECT cayu_sessions.id
    FROM cayu_sessions
    JOIN requested_sessions AS requested ON requested.session_id = cayu_sessions.id
    {session_where_sql}
),
pricing_candidates AS (
    SELECT
        event.session_id,
        substr(event.timestamp, 1, 10) AS effective_on,
        {usage_metrics_projection} AS usage_metrics_json,
        {billing_identity_projection} AS billing_identity_json
    FROM cayu_events AS event
    JOIN matched_sessions AS session ON session.id = event.session_id
    WHERE event.timestamp >= ?
      AND event.timestamp < ?
      AND event.event_type = 'model.completed'
),
measured_candidates AS (
    SELECT
        *,
        length(CAST(session_id AS BLOB))
            + COALESCE(length(CAST(usage_metrics_json AS BLOB)), 0)
            + COALESCE(length(CAST(billing_identity_json AS BLOB)), 0) AS input_bytes
    FROM pricing_candidates
),
bounded_candidates AS (
    SELECT
        session_id,
        effective_on,
        input_bytes > (SELECT max_input_bytes FROM scope) AS input_oversized,
        CASE
            WHEN input_bytes <= (SELECT max_input_bytes FROM scope)
            THEN usage_metrics_json
        END AS usage_metrics_json,
        CASE
            WHEN input_bytes <= (SELECT max_input_bytes FROM scope)
            THEN billing_identity_json
        END AS billing_identity_json
    FROM measured_candidates
)
SELECT
    session_id,
    effective_on,
    usage_metrics_json,
    billing_identity_json,
    input_oversized,
    occurrences,
    COUNT(*) OVER () AS raw_group_count
FROM (
    SELECT
        session_id,
        effective_on,
        usage_metrics_json,
        billing_identity_json,
        input_oversized,
        COUNT(*) AS occurrences
    FROM bounded_candidates
    GROUP BY
        session_id,
        effective_on,
        usage_metrics_json,
        billing_identity_json,
        input_oversized
)
LIMIT ?
"""

_SESSION_BREAKDOWN_SQL = """
WITH
scope(session_group_limit) AS (
    SELECT ?
),
matched_sessions AS MATERIALIZED (
    SELECT id, status
    FROM cayu_sessions
    {session_where_sql}
),
usage_events AS MATERIALIZED (
    SELECT
        event.session_id,
        event.event_type,
        CASE
            WHEN json_type(
                     event.payload_json,
                     '$.usage_normalization_failed'
                 ) IS NOT 'true'
             AND json_type(event.payload_json, '$.usage_metrics') = 'object'
            THEN 1
            ELSE 0
        END AS has_usage,
        {input_tokens} AS input_tokens,
        {output_tokens} AS output_tokens,
        {total_tokens} AS total_tokens,
        {reasoning_tokens} AS reasoning_output_tokens,
        {cache_read_tokens} AS cache_read_tokens,
        {cache_write_tokens} AS cache_write_tokens,
        {cache_write_5m_tokens} AS cache_write_5m_tokens,
        {cache_write_1h_tokens} AS cache_write_1h_tokens,
        {cache_write_unknown_tokens} AS cache_write_unknown_ttl_tokens,
        {cached_input_tokens} AS cached_input_tokens,
        {uncached_input_tokens} AS uncached_input_tokens
    FROM cayu_events AS event
    JOIN matched_sessions AS session ON session.id = event.session_id
    WHERE event.timestamp >= ?
      AND event.timestamp < ?
      AND event.event_type IN ('model.completed', 'tool.call.started')
),
session_grouped_raw AS (
    SELECT
        session.id AS session_id,
        session.status,
        session.status IN ('pending', 'running', 'interrupting') AS active,
        CASE WHEN COUNT(event.session_id) > 0 THEN 1 ELSE 0 END AS session_count,
        COALESCE(SUM(event.event_type = 'model.completed'), 0) AS model_steps,
        COALESCE(SUM(
            event.event_type = 'model.completed' AND event.has_usage = 1
        ), 0) AS model_steps_with_usage,
        COALESCE(SUM(event.event_type = 'tool.call.started'), 0) AS tool_calls,
        {session_usage_sum} AS usage_sums
    FROM matched_sessions AS session
    LEFT JOIN usage_events AS event ON event.session_id = session.id
    GROUP BY session.id, session.status
),
session_grouped AS (
    SELECT
        session_id, status, active, session_count, model_steps,
        model_steps_with_usage, tool_calls,
        {usage_sum_projection}
    FROM session_grouped_raw
),
retained_sessions AS MATERIALIZED (
    SELECT
        *
    FROM session_grouped
    ORDER BY total_tokens DESC, model_steps DESC, session_id ASC
    LIMIT (SELECT session_group_limit FROM scope)
),
session_identity_bound AS (
    SELECT EXISTS (
        SELECT 1
        FROM retained_sessions
        WHERE length(CAST(session_id AS BLOB)) > {session_id_max_bytes}
    ) AS exceeded
),
session_remainder_raw AS (
    SELECT
        COALESCE(SUM(session.session_count), 0) AS session_count,
        COALESCE(SUM(session.model_steps), 0) AS model_steps,
        COALESCE(SUM(session.model_steps_with_usage), 0) AS model_steps_with_usage,
        COALESCE(SUM(session.tool_calls), 0) AS tool_calls,
        {remainder_group_usage_sum} AS usage_sums,
        COUNT(*) AS group_count,
        COALESCE(SUM(session.active), 0) AS active_session_count
    FROM session_grouped AS session
    WHERE NOT EXISTS (
        SELECT 1
        FROM retained_sessions AS retained
        WHERE retained.session_id = session.session_id
    )
    HAVING COUNT(*) > 0
),
session_remainder AS (
    SELECT
        session_count, model_steps, model_steps_with_usage, tool_calls,
        {usage_sum_projection}, group_count, active_session_count
    FROM session_remainder_raw
)
SELECT
    'session' AS section,
    CASE
        WHEN length(CAST(session_id AS BLOB)) <= {session_id_max_bytes}
        THEN session_id
    END AS session_id,
    (SELECT exceeded FROM session_identity_bound) AS session_id_too_large,
    status,
    active,
    session_count,
    model_steps,
    model_steps_with_usage,
    tool_calls,
    input_tokens,
    output_tokens,
    total_tokens,
    reasoning_output_tokens,
    cache_read_tokens,
    cache_write_tokens,
    cache_write_5m_tokens,
    cache_write_1h_tokens,
    cache_write_unknown_ttl_tokens,
    cached_input_tokens,
    uncached_input_tokens,
    0 AS group_count,
    0 AS active_session_count
FROM retained_sessions
UNION ALL
SELECT
    'session_remainder',
    NULL,
    (SELECT exceeded FROM session_identity_bound),
    NULL,
    NULL,
    session_count,
    model_steps,
    model_steps_with_usage,
    tool_calls,
    input_tokens,
    output_tokens,
    total_tokens,
    reasoning_output_tokens,
    cache_read_tokens,
    cache_write_tokens,
    cache_write_5m_tokens,
    cache_write_1h_tokens,
    cache_write_unknown_ttl_tokens,
    cached_input_tokens,
    uncached_input_tokens,
    group_count,
    active_session_count
FROM session_remainder
"""


_TOKEN_PATHS = {
    "input_tokens": "$.usage_metrics.input_tokens",
    "output_tokens": "$.usage_metrics.output_tokens",
    "total_tokens": "$.usage_metrics.total_tokens",
    "reasoning_tokens": "$.usage_metrics.reasoning_output_tokens",
    "cache_read_tokens": "$.usage_metrics.cache.read_tokens",
    "cache_write_tokens": "$.usage_metrics.cache.write_tokens",
    "cache_write_5m_tokens": "$.usage_metrics.cache.write_5m_tokens",
    "cache_write_1h_tokens": "$.usage_metrics.cache.write_1h_tokens",
    "cache_write_unknown_tokens": "$.usage_metrics.cache.write_unknown_ttl_tokens",
    "cached_input_tokens": "$.usage_metrics.cache.cached_input_tokens",
    "uncached_input_tokens": "$.usage_metrics.cache.uncached_input_tokens",
}

_USAGE_COUNTER_COLUMNS = (
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "reasoning_output_tokens",
    "cache_read_tokens",
    "cache_write_tokens",
    "cache_write_5m_tokens",
    "cache_write_1h_tokens",
    "cache_write_unknown_ttl_tokens",
    "cached_input_tokens",
    "uncached_input_tokens",
)
_EXACT_SUM_ZERO = "0" * 38


def aggregate_session_usage(
    connection: sqlite3.Connection,
    *,
    session_plan: SessionQuerySqlPlan,
    query: UsageRollupQuery,
) -> UsageRollupStoreResult:
    sql, params = usage_rollup_statement(session_plan=session_plan, query=query)
    owns_transaction = not connection.in_transaction
    if owns_transaction:
        connection.execute("BEGIN")
    try:
        rows = connection.execute(sql, params).fetchall()
        session_breakdown = None
        if query.session_group_limit is not None:
            session_sql, session_params = session_breakdown_statement(
                session_plan=session_plan,
                query=query,
            )
            session_rows = connection.execute(session_sql, session_params).fetchall()
            session_breakdown = _session_breakdown_from_rows(
                session_rows,
                limit=query.session_group_limit,
            )
        pricing = BoundedUsagePricingInputAccumulator(query.pricing_input_limit)
        if query.include_pricing_inputs:
            pricing_sql, pricing_params = pricing_input_statement(
                session_plan=session_plan,
                query=query,
            )
            cursor = connection.execute(pricing_sql, pricing_params)
            for row in cursor:
                _add_pricing_input_from_row(pricing, row)
                if pricing.truncated:
                    break
        session_pricing = BoundedUsagePricingInputAccumulator(query.pricing_input_limit)
        if (
            query.include_pricing_inputs
            and session_breakdown is not None
            and session_breakdown.groups
        ):
            session_pricing_sql, session_pricing_params = session_pricing_input_statement(
                session_plan=session_plan,
                query=query,
                session_ids=tuple(group.session_id for group in session_breakdown.groups),
            )
            cursor = connection.execute(session_pricing_sql, session_pricing_params)
            for row in cursor:
                _add_pricing_input_from_row(
                    session_pricing,
                    row,
                    session_id=row["session_id"],
                )
                if session_pricing.truncated:
                    break
        result = _result_from_rows(
            rows,
            query=query,
            pricing=pricing.result(),
            session_pricing=session_pricing.result(),
            session_breakdown=session_breakdown,
        )
        if owns_transaction:
            connection.commit()
        return result
    except Exception:
        if owns_transaction:
            connection.rollback()
        raise


def usage_rollup_statement(
    *,
    session_plan: SessionQuerySqlPlan,
    query: UsageRollupQuery,
) -> tuple[str, tuple[object, ...]]:
    """Build the exact SQLite statement used by the aggregate read path."""

    sql = _RESULT_SQL.format(
        session_where_sql=session_plan.filter_where_sql,
        **{name: _nonnegative_json_integer(path) for name, path in _TOKEN_PATHS.items()},
        overall_usage_sum=_exact_usage_sum(
            "CASE WHEN event_type = 'model.completed' THEN {column} ELSE 0 END"
        ),
        group_usage_sum=_exact_usage_sum("{column}"),
        event_usage_sum=_exact_usage_sum("event.{column}"),
        usage_sum_projection=_usage_sum_projection(),
    )
    return (
        sql,
        (
            query.group_limit,
            AGGREGATE_IDENTITY_TRIM_CHARACTERS,
            *session_plan.filter_params,
            query.start_at.isoformat(),
            query.end_at.isoformat(),
        ),
    )


def session_breakdown_statement(
    *,
    session_plan: SessionQuerySqlPlan,
    query: UsageRollupQuery,
) -> tuple[str, tuple[object, ...]]:
    if query.session_group_limit is None:
        raise ValueError("session_group_limit is required for a session breakdown.")
    sql = _SESSION_BREAKDOWN_SQL.format(
        session_where_sql=session_plan.filter_where_sql,
        **{name: _nonnegative_json_integer(path) for name, path in _TOKEN_PATHS.items()},
        session_usage_sum=_exact_usage_sum(
            "CASE WHEN event.event_type = 'model.completed' THEN event.{column} ELSE 0 END"
        ),
        remainder_group_usage_sum=_exact_usage_sum("session.{column}"),
        usage_sum_projection=_usage_sum_projection(),
        session_id_max_bytes=MAX_USAGE_ROLLUP_SESSION_ID_BYTES,
    )
    return (
        sql,
        (
            query.session_group_limit,
            *session_plan.filter_params,
            query.start_at.isoformat(),
            query.end_at.isoformat(),
        ),
    )


def pricing_input_statement(
    *,
    session_plan: SessionQuerySqlPlan,
    query: UsageRollupQuery,
    max_input_bytes: int = MAX_USAGE_PRICING_INPUT_BYTES,
) -> tuple[str, tuple[object, ...]]:
    if type(max_input_bytes) is not int or max_input_bytes < 1:
        raise ValueError("max_input_bytes must be a positive integer.")
    return (
        _PRICING_INPUT_SQL.format(
            session_where_sql=session_plan.filter_where_sql,
            usage_metrics_projection=_sqlite_usage_metrics_projection(),
            billing_identity_projection=_sqlite_billing_identity_projection("$.billing_identity"),
        ),
        (
            max_input_bytes,
            AGGREGATE_IDENTITY_TRIM_CHARACTERS,
            *session_plan.filter_params,
            query.start_at.isoformat(),
            query.end_at.isoformat(),
            MAX_USAGE_PRICING_RAW_CANDIDATES,
        ),
    )


def session_pricing_input_statement(
    *,
    session_plan: SessionQuerySqlPlan,
    query: UsageRollupQuery,
    session_ids: tuple[str, ...],
    max_input_bytes: int = MAX_USAGE_PRICING_INPUT_BYTES,
) -> tuple[str, tuple[object, ...]]:
    if type(max_input_bytes) is not int or max_input_bytes < 1:
        raise ValueError("max_input_bytes must be a positive integer.")
    if not session_ids:
        raise ValueError("session_ids cannot be empty.")
    if len(session_ids) > 100 or len(set(session_ids)) != len(session_ids):
        raise ValueError("session_ids must contain at most 100 distinct values.")
    return (
        _SESSION_PRICING_INPUT_SQL.format(
            requested_session_values=", ".join("(?)" for _ in session_ids),
            session_where_sql=session_plan.filter_where_sql,
            usage_metrics_projection=_sqlite_usage_metrics_projection(),
            billing_identity_projection=_sqlite_billing_identity_projection("$.billing_identity"),
        ),
        (
            max_input_bytes,
            AGGREGATE_IDENTITY_TRIM_CHARACTERS,
            *session_ids,
            *session_plan.filter_params,
            query.start_at.isoformat(),
            query.end_at.isoformat(),
            MAX_USAGE_PRICING_RAW_CANDIDATES + 1,
        ),
    )


def _sqlite_usage_metrics_projection() -> str:
    metrics_path = "$.usage_metrics"
    identity_path = f"{metrics_path}.billing_identity"
    metrics = f"json_extract(event.payload_json, '{metrics_path}')"
    identity = _sqlite_billing_identity_projection(identity_path)
    return f"""
        CASE
            WHEN json_type(
                     event.payload_json,
                     '$.usage_normalization_failed'
                 ) IS NOT 'true'
             AND json_type(event.payload_json, '{metrics_path}') = 'object'
            THEN CASE
                WHEN json_type(event.payload_json, '{identity_path}') = 'object'
                THEN json_set(
                    {metrics},
                    '$.billing_identity',
                    json({identity})
                )
                ELSE {metrics}
            END
        END
    """.strip()


def _sqlite_billing_identity_projection(identity_path: str) -> str:
    identity = f"json_extract(event.payload_json, '{identity_path}')"
    request_evidence = _sqlite_text_evidence_projection(
        f"{identity_path}.request_evidence",
        _BEDROCK_AGGREGATE_REQUEST_EVIDENCE,
    )
    completion_evidence = _sqlite_text_evidence_projection(
        f"{identity_path}.completion_evidence",
        _BEDROCK_AGGREGATE_COMPLETION_EVIDENCE,
    )
    return f"""
        CASE
            WHEN json_type(event.payload_json, '{identity_path}') = 'object'
            THEN CASE
                WHEN json_extract(
                    event.payload_json,
                    '{identity_path}.provider_name'
                ) = 'bedrock'
                THEN json_patch(
                    json_remove({identity}, '$.request_evidence', '$.completion_evidence'),
                    json_object(
                        'request_evidence', CASE
                            WHEN json({request_evidence}) <> json('{{}}')
                            THEN json({request_evidence})
                        END,
                        'completion_evidence', CASE
                            WHEN json({completion_evidence}) <> json('{{}}')
                            THEN json({completion_evidence})
                        END
                    )
                )
                ELSE json_remove({identity}, '$.request_evidence', '$.completion_evidence')
            END
        END
    """.strip()


def _sqlite_text_evidence_projection(evidence_path: str, fields: tuple[str, ...]) -> str:
    projected = "'{}'"
    for field in fields:
        path = f"{evidence_path}.{field}"
        value = f"json_extract(event.payload_json, '{path}')"
        projected = f"""
            json_patch(
                {projected},
                CASE
                    WHEN json_type(event.payload_json, '{path}') = 'text'
                     AND {value} <> ''
                     AND trim(
                         {value},
                         (SELECT identity_trim FROM scope)
                     ) = {value}
                    THEN json_object('{field}', {value})
                    ELSE '{{}}'
                END
            )
        """.strip()
    return projected


def _nonnegative_json_integer(path: str) -> str:
    return f"""
        CASE
            WHEN json_type(
                     event.payload_json,
                     '$.usage_normalization_failed'
                 ) IS NOT 'true'
             AND json_type(event.payload_json, '{path}') = 'integer'
             AND json_extract(event.payload_json, '{path}') >= 0
             AND json_extract(event.payload_json, '{path}') <= {MAX_AGGREGATE_USAGE_COUNTER}
            THEN json_extract(event.payload_json, '{path}')
            ELSE 0
        END
    """.strip()


def _exact_usage_sum(expression: str) -> str:
    arguments = ",\n            ".join(
        expression.format(column=column) for column in _USAGE_COUNTER_COLUMNS
    )
    return f"cayu_exact_usage_sum(\n            {arguments}\n        )"


def _usage_sum_projection() -> str:
    return ",\n        ".join(
        (f"COALESCE(json_extract(usage_sums, '$[{index}]'), '{_EXACT_SUM_ZERO}') AS {column}")
        for index, column in enumerate(_USAGE_COUNTER_COLUMNS)
    )


def _result_from_rows(
    rows: list[sqlite3.Row],
    *,
    query: UsageRollupQuery,
    pricing: tuple[tuple[UsagePricingInput, ...], int, AggregateAccuracy],
    session_pricing: tuple[tuple[UsagePricingInput, ...], int, AggregateAccuracy],
    session_breakdown: UsageSessionAggregateBreakdown | None,
) -> UsageRollupStoreResult:
    overall = next(row for row in rows if row["section"] == "overall")
    provider_rows = _ordered_group_rows([row for row in rows if row["section"] == "provider"])
    model_rows = _ordered_group_rows([row for row in rows if row["section"] == "model"])
    provider_remainder = next(
        (row for row in rows if row["section"] == "provider_remainder"),
        None,
    )
    model_remainder = next(
        (row for row in rows if row["section"] == "model_remainder"),
        None,
    )
    pricing_inputs, pricing_group_count, pricing_accuracy = pricing
    (
        session_pricing_inputs,
        session_pricing_group_count,
        session_pricing_accuracy,
    ) = session_pricing

    return UsageRollupStoreResult(
        as_of=datetime.fromisoformat(overall["as_of"]).astimezone(UTC),
        start_at=query.start_at,
        end_at=query.end_at,
        totals=_totals_from_row(overall),
        provider_breakdown=_breakdown_from_rows(
            provider_rows,
            remainder=provider_remainder,
            limit=query.group_limit,
            dimension="provider",
        ),
        model_breakdown=_breakdown_from_rows(
            model_rows,
            remainder=model_remainder,
            limit=query.group_limit,
            dimension="model",
        ),
        session_breakdown=session_breakdown,
        pricing_inputs=tuple(pricing_inputs),
        pricing_inputs_included=query.include_pricing_inputs,
        pricing_input_group_count=pricing_group_count,
        pricing_inputs_accuracy=pricing_accuracy,
        session_pricing_inputs=tuple(session_pricing_inputs),
        session_pricing_inputs_included=(
            query.include_pricing_inputs and query.session_group_limit is not None
        ),
        session_pricing_input_group_count=session_pricing_group_count,
        session_pricing_inputs_accuracy=session_pricing_accuracy,
        active_session_count=overall["active_session_count"],
        matching_session_count=overall["matching_session_count"],
    )


def _session_breakdown_from_rows(
    rows: list[sqlite3.Row],
    *,
    limit: int,
) -> UsageSessionAggregateBreakdown:
    if any(bool(row["session_id_too_large"]) for row in rows):
        raise UsageRollupResultTooLarge("A session identity exceeds the usage-rollup byte limit.")
    groups = tuple(
        UsageSessionAggregateGroup(
            session_id=row["session_id"],
            status=row["status"],
            active=bool(row["active"]),
            totals=_totals_from_row(row),
        )
        for row in sorted(
            (row for row in rows if row["section"] == "session"),
            key=lambda row: (
                -int(row["total_tokens"]),
                -int(row["model_steps"]),
                str(row["session_id"]),
            ),
        )
    )
    remainder_row = next(
        (row for row in rows if row["section"] == "session_remainder"),
        None,
    )
    remainder = (
        None
        if remainder_row is None
        else UsageSessionAggregateRemainder(
            group_count=_row_int(remainder_row, "group_count"),
            active_session_count=_row_int(remainder_row, "active_session_count"),
            totals=_totals_from_row(remainder_row),
        )
    )
    return UsageSessionAggregateBreakdown(
        groups=groups,
        remainder=remainder,
        accuracy=(
            EXACT_AGGREGATE.model_copy()
            if remainder is None
            else AggregateAccuracy(
                kind=AggregateAccuracyKind.TRUNCATED,
                reason="Matching sessions exceed session_group_limit.",
                limit=limit,
            )
        ),
    )


def _ordered_group_rows(rows: list[sqlite3.Row]) -> list[sqlite3.Row]:
    return sorted(
        rows,
        key=lambda row: (
            -int(row["total_tokens"]),
            -int(row["model_steps"]),
            row["provider_name"] is None,
            row["provider_name"] or "",
            row["model"] is None,
            row["model"] or "",
        ),
    )


def _breakdown_from_rows(
    rows: list[sqlite3.Row],
    *,
    remainder: sqlite3.Row | None,
    limit: int,
    dimension: str,
) -> UsageAggregateBreakdown:
    groups = tuple(
        UsageAggregateGroup(
            provider_name=aggregate_identity_value(row["provider_name"]),
            model=aggregate_identity_value(row["model"]),
            totals=_totals_from_row(row),
        )
        for row in rows
    )
    if remainder is None:
        return UsageAggregateBreakdown(
            groups=groups,
            remainder=None,
            accuracy=EXACT_AGGREGATE.model_copy(),
        )
    return UsageAggregateBreakdown(
        groups=groups,
        remainder=UsageAggregateRemainder(
            group_count=remainder["group_count"],
            totals=_totals_from_row(remainder),
        ),
        accuracy=AggregateAccuracy(
            kind=AggregateAccuracyKind.TRUNCATED,
            reason=f"Distinct {dimension} groups exceed group_limit.",
            limit=limit,
        ),
    )


def _totals_from_row(row: sqlite3.Row) -> UsageAggregateTotals:
    return UsageAggregateTotals(
        session_count=_row_int(row, "session_count"),
        model_steps=_row_int(row, "model_steps"),
        model_steps_with_usage=_row_int(row, "model_steps_with_usage"),
        tool_calls=_row_int(row, "tool_calls"),
        usage=build_aggregate_usage_metrics(
            input_tokens=_row_int(row, "input_tokens"),
            output_tokens=_row_int(row, "output_tokens"),
            total_tokens=_row_int(row, "total_tokens"),
            reasoning_output_tokens=_row_int(row, "reasoning_output_tokens"),
            cache_read_tokens=_row_int(row, "cache_read_tokens"),
            cache_write_tokens=_row_int(row, "cache_write_tokens"),
            cache_write_5m_tokens=_row_int(row, "cache_write_5m_tokens"),
            cache_write_1h_tokens=_row_int(row, "cache_write_1h_tokens"),
            cache_write_unknown_ttl_tokens=_row_int(row, "cache_write_unknown_ttl_tokens"),
            cached_input_tokens=_row_int(row, "cached_input_tokens"),
            uncached_input_tokens=_row_int(row, "uncached_input_tokens"),
        ),
    )


def _row_int(row: sqlite3.Row, column: str) -> int:
    value = row[column]
    return 0 if value is None else int(value)


def _add_pricing_input_from_row(
    pricing: BoundedUsagePricingInputAccumulator,
    row: sqlite3.Row,
    *,
    session_id: str | None = None,
) -> None:
    if row["input_oversized"]:
        pricing.reject_oversized_candidate()
        return
    if row["raw_group_count"] > MAX_USAGE_PRICING_RAW_CANDIDATES:
        pricing.reject_candidate_row_overflow(limit=MAX_USAGE_PRICING_RAW_CANDIDATES)
        return
    payload: dict[str, Any] = {}
    if row["usage_metrics_json"] is not None:
        payload["usage_metrics"] = json.loads(row["usage_metrics_json"])
    if row["billing_identity_json"] is not None:
        payload["billing_identity"] = json.loads(row["billing_identity_json"])
    pricing.add_payload(
        session_id=session_id,
        effective_on=row["effective_on"],
        occurrences=row["occurrences"],
        payload=payload,
    )

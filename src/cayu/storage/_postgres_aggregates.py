from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, LiteralString, cast

from cayu.runtime.aggregates import (
    AGGREGATE_IDENTITY_TRIM_CHARACTERS,
    EXACT_AGGREGATE,
    MAX_AGGREGATE_USAGE_COUNTER,
    MAX_USAGE_PRICING_INPUT_BYTES,
    MAX_USAGE_PRICING_RAW_CANDIDATES,
    AggregateAccuracy,
    AggregateAccuracyKind,
    BoundedUsagePricingInputAccumulator,
    UsageAggregateBreakdown,
    UsageAggregateGroup,
    UsageAggregateRemainder,
    UsageAggregateTotals,
    UsagePricingInput,
    UsageRollupStoreResult,
    aggregate_identity_value,
)
from cayu.runtime.sessions import UsageRollupQuery
from cayu.runtime.usage import CacheUsageMetrics, UsageMetrics
from cayu.storage._session_store_sql import SessionQuerySqlPlan

_RESULT_SQL = """
WITH
scope(group_limit, identity_trim) AS (
    SELECT %s::integer, %s::text
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
            WHEN jsonb_typeof(event.payload -> 'usage_metrics') = 'object' THEN 1
            ELSE 0
        END AS has_usage,
        (CASE
            WHEN jsonb_typeof(event.payload #> '{{usage_metrics,provider_name}}') = 'string'
             AND btrim(
                 event.payload #>> '{{usage_metrics,provider_name}}',
                 (SELECT identity_trim FROM scope)
             ) = event.payload #>> '{{usage_metrics,provider_name}}'
             AND event.payload #>> '{{usage_metrics,provider_name}}' <> ''
            THEN event.payload #>> '{{usage_metrics,provider_name}}'
        END) COLLATE "C" AS provider_name,
        (CASE
            WHEN jsonb_typeof(event.payload #> '{{usage_metrics,model}}') = 'string'
             AND btrim(
                 event.payload #>> '{{usage_metrics,model}}',
                 (SELECT identity_trim FROM scope)
             ) = event.payload #>> '{{usage_metrics,model}}'
             AND event.payload #>> '{{usage_metrics,model}}' <> ''
            THEN event.payload #>> '{{usage_metrics,model}}'
        END) COLLATE "C" AS model,
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
    WHERE event.timestamp >= %s::timestamptz
      AND event.timestamp < %s::timestamptz
      AND event.event_type IN ('model.completed', 'tool.call.started')
),
overall AS (
    SELECT
        COUNT(DISTINCT session_id) AS session_count,
        COUNT(*) FILTER (WHERE event_type = 'model.completed') AS model_steps,
        COUNT(*) FILTER (
            WHERE event_type = 'model.completed' AND has_usage = 1
        ) AS model_steps_with_usage,
        COUNT(*) FILTER (WHERE event_type = 'tool.call.started') AS tool_calls,
        COALESCE(SUM(input_tokens) FILTER (WHERE event_type = 'model.completed'), 0)
            AS input_tokens,
        COALESCE(SUM(output_tokens) FILTER (WHERE event_type = 'model.completed'), 0)
            AS output_tokens,
        COALESCE(SUM(total_tokens) FILTER (WHERE event_type = 'model.completed'), 0)
            AS total_tokens,
        COALESCE(SUM(reasoning_output_tokens) FILTER (WHERE event_type = 'model.completed'), 0)
            AS reasoning_output_tokens,
        COALESCE(SUM(cache_read_tokens) FILTER (WHERE event_type = 'model.completed'), 0)
            AS cache_read_tokens,
        COALESCE(SUM(cache_write_tokens) FILTER (WHERE event_type = 'model.completed'), 0)
            AS cache_write_tokens,
        COALESCE(SUM(cache_write_5m_tokens) FILTER (WHERE event_type = 'model.completed'), 0)
            AS cache_write_5m_tokens,
        COALESCE(SUM(cache_write_1h_tokens) FILTER (WHERE event_type = 'model.completed'), 0)
            AS cache_write_1h_tokens,
        COALESCE(SUM(cache_write_unknown_ttl_tokens) FILTER (WHERE event_type = 'model.completed'), 0)
            AS cache_write_unknown_ttl_tokens,
        COALESCE(SUM(cached_input_tokens) FILTER (WHERE event_type = 'model.completed'), 0)
            AS cached_input_tokens,
        COALESCE(SUM(uncached_input_tokens) FILTER (WHERE event_type = 'model.completed'), 0)
            AS uncached_input_tokens
    FROM usage_events
),
provider_grouped AS (
    SELECT
        provider_name,
        COUNT(DISTINCT session_id) AS session_count,
        COUNT(*) AS model_steps,
        SUM(has_usage) AS model_steps_with_usage,
        SUM(input_tokens) AS input_tokens,
        SUM(output_tokens) AS output_tokens,
        SUM(total_tokens) AS total_tokens,
        SUM(reasoning_output_tokens) AS reasoning_output_tokens,
        SUM(cache_read_tokens) AS cache_read_tokens,
        SUM(cache_write_tokens) AS cache_write_tokens,
        SUM(cache_write_5m_tokens) AS cache_write_5m_tokens,
        SUM(cache_write_1h_tokens) AS cache_write_1h_tokens,
        SUM(cache_write_unknown_ttl_tokens) AS cache_write_unknown_ttl_tokens,
        SUM(cached_input_tokens) AS cached_input_tokens,
        SUM(uncached_input_tokens) AS uncached_input_tokens
    FROM usage_events
    WHERE event_type = 'model.completed'
    GROUP BY provider_name
),
provider_ranked AS (
    SELECT
        *,
        ROW_NUMBER() OVER (
            ORDER BY total_tokens DESC, model_steps DESC,
                     provider_name ASC NULLS LAST
        ) AS group_rank,
        COUNT(*) OVER () AS total_groups
    FROM provider_grouped
),
provider_remainder AS (
    SELECT
        COUNT(DISTINCT event.session_id) AS session_count,
        COUNT(*) AS model_steps,
        SUM(event.has_usage) AS model_steps_with_usage,
        SUM(event.input_tokens) AS input_tokens,
        SUM(event.output_tokens) AS output_tokens,
        SUM(event.total_tokens) AS total_tokens,
        SUM(event.reasoning_output_tokens) AS reasoning_output_tokens,
        SUM(event.cache_read_tokens) AS cache_read_tokens,
        SUM(event.cache_write_tokens) AS cache_write_tokens,
        SUM(event.cache_write_5m_tokens) AS cache_write_5m_tokens,
        SUM(event.cache_write_1h_tokens) AS cache_write_1h_tokens,
        SUM(event.cache_write_unknown_ttl_tokens) AS cache_write_unknown_ttl_tokens,
        SUM(event.cached_input_tokens) AS cached_input_tokens,
        SUM(event.uncached_input_tokens) AS uncached_input_tokens,
        MAX(ranked.total_groups) - (SELECT group_limit FROM scope) AS group_count
    FROM usage_events AS event
    JOIN provider_ranked AS ranked
      ON event.provider_name IS NOT DISTINCT FROM ranked.provider_name
    WHERE event.event_type = 'model.completed'
      AND ranked.group_rank > (SELECT group_limit FROM scope)
    HAVING COUNT(*) > 0
),
model_grouped AS (
    SELECT
        provider_name,
        model,
        COUNT(DISTINCT session_id) AS session_count,
        COUNT(*) AS model_steps,
        SUM(has_usage) AS model_steps_with_usage,
        SUM(input_tokens) AS input_tokens,
        SUM(output_tokens) AS output_tokens,
        SUM(total_tokens) AS total_tokens,
        SUM(reasoning_output_tokens) AS reasoning_output_tokens,
        SUM(cache_read_tokens) AS cache_read_tokens,
        SUM(cache_write_tokens) AS cache_write_tokens,
        SUM(cache_write_5m_tokens) AS cache_write_5m_tokens,
        SUM(cache_write_1h_tokens) AS cache_write_1h_tokens,
        SUM(cache_write_unknown_ttl_tokens) AS cache_write_unknown_ttl_tokens,
        SUM(cached_input_tokens) AS cached_input_tokens,
        SUM(uncached_input_tokens) AS uncached_input_tokens
    FROM usage_events
    WHERE event_type = 'model.completed'
    GROUP BY provider_name, model
),
model_ranked AS (
    SELECT
        *,
        ROW_NUMBER() OVER (
            ORDER BY total_tokens DESC, model_steps DESC,
                     provider_name ASC NULLS LAST, model ASC NULLS LAST
        ) AS group_rank,
        COUNT(*) OVER () AS total_groups
    FROM model_grouped
),
model_remainder AS (
    SELECT
        COUNT(DISTINCT event.session_id) AS session_count,
        COUNT(*) AS model_steps,
        SUM(event.has_usage) AS model_steps_with_usage,
        SUM(event.input_tokens) AS input_tokens,
        SUM(event.output_tokens) AS output_tokens,
        SUM(event.total_tokens) AS total_tokens,
        SUM(event.reasoning_output_tokens) AS reasoning_output_tokens,
        SUM(event.cache_read_tokens) AS cache_read_tokens,
        SUM(event.cache_write_tokens) AS cache_write_tokens,
        SUM(event.cache_write_5m_tokens) AS cache_write_5m_tokens,
        SUM(event.cache_write_1h_tokens) AS cache_write_1h_tokens,
        SUM(event.cache_write_unknown_ttl_tokens) AS cache_write_unknown_ttl_tokens,
        SUM(event.cached_input_tokens) AS cached_input_tokens,
        SUM(event.uncached_input_tokens) AS uncached_input_tokens,
        MAX(ranked.total_groups) - (SELECT group_limit FROM scope) AS group_count
    FROM usage_events AS event
    JOIN model_ranked AS ranked
      ON event.provider_name IS NOT DISTINCT FROM ranked.provider_name
     AND event.model IS NOT DISTINCT FROM ranked.model
    WHERE event.event_type = 'model.completed'
      AND ranked.group_rank > (SELECT group_limit FROM scope)
    HAVING COUNT(*) > 0
)
SELECT
    'overall'::text AS section,
    NULL::text AS provider_name,
    NULL::text AS model,
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
    0::bigint AS group_count,
    NULL::date AS effective_on,
    0::bigint AS occurrences,
    NULL::jsonb AS usage_metrics,
    NULL::jsonb AS billing_identity,
    (SELECT COUNT(*) FROM matched_sessions) AS matching_session_count,
    (
        SELECT COUNT(*)
        FROM matched_sessions
        WHERE status IN ('pending', 'running', 'interrupting')
    ) AS active_session_count
FROM overall
UNION ALL
SELECT
    'provider', provider_name, NULL, session_count, model_steps,
    model_steps_with_usage, 0, input_tokens, output_tokens, total_tokens,
    reasoning_output_tokens, cache_read_tokens, cache_write_tokens,
    cache_write_5m_tokens, cache_write_1h_tokens, cache_write_unknown_ttl_tokens,
    cached_input_tokens, uncached_input_tokens, total_groups, NULL, 0, NULL, NULL, 0, 0
FROM provider_ranked
WHERE group_rank <= (SELECT group_limit FROM scope)
UNION ALL
SELECT
    'provider_remainder', NULL, NULL, session_count, model_steps,
    model_steps_with_usage, 0, input_tokens, output_tokens, total_tokens,
    reasoning_output_tokens, cache_read_tokens, cache_write_tokens,
    cache_write_5m_tokens, cache_write_1h_tokens, cache_write_unknown_ttl_tokens,
    cached_input_tokens, uncached_input_tokens, group_count, NULL, 0, NULL, NULL, 0, 0
FROM provider_remainder
UNION ALL
SELECT
    'model', provider_name, model, session_count, model_steps,
    model_steps_with_usage, 0, input_tokens, output_tokens, total_tokens,
    reasoning_output_tokens, cache_read_tokens, cache_write_tokens,
    cache_write_5m_tokens, cache_write_1h_tokens, cache_write_unknown_ttl_tokens,
    cached_input_tokens, uncached_input_tokens, total_groups, NULL, 0, NULL, NULL, 0, 0
FROM model_ranked
WHERE group_rank <= (SELECT group_limit FROM scope)
UNION ALL
SELECT
    'model_remainder', NULL, NULL, session_count, model_steps,
    model_steps_with_usage, 0, input_tokens, output_tokens, total_tokens,
    reasoning_output_tokens, cache_read_tokens, cache_write_tokens,
    cache_write_5m_tokens, cache_write_1h_tokens, cache_write_unknown_ttl_tokens,
    cached_input_tokens, uncached_input_tokens, group_count, NULL, 0, NULL, NULL, 0, 0
FROM model_remainder
"""

_PRICING_INPUT_SQL = """
WITH
scope(max_input_bytes) AS (
    SELECT %s::bigint
),
matched_sessions AS MATERIALIZED (
    SELECT id
    FROM cayu_sessions
    {session_where_sql}
),
pricing_candidates AS (
    SELECT
        (event.timestamp AT TIME ZONE 'UTC')::date AS effective_on,
        CASE
            WHEN jsonb_typeof(event.payload -> 'usage_metrics') = 'object'
            THEN CASE
                WHEN jsonb_typeof(
                    event.payload #> '{{usage_metrics,billing_identity}}'
                ) = 'object'
                THEN jsonb_set(
                    event.payload -> 'usage_metrics',
                    '{{billing_identity}}',
                    (event.payload #> '{{usage_metrics,billing_identity}}')
                        - 'request_evidence' - 'completion_evidence',
                    false
                )
                ELSE event.payload -> 'usage_metrics'
            END
        END AS usage_metrics,
        CASE
            WHEN jsonb_typeof(event.payload -> 'billing_identity') = 'object'
            THEN (event.payload -> 'billing_identity')
                 - 'request_evidence' - 'completion_evidence'
        END AS billing_identity
    FROM cayu_events AS event
    JOIN matched_sessions AS session ON session.id = event.session_id
    WHERE event.timestamp >= %s::timestamptz
      AND event.timestamp < %s::timestamptz
      AND event.event_type = 'model.completed'
),
measured_candidates AS (
    SELECT
        *,
        COALESCE(octet_length(usage_metrics::text), 0)
            + COALESCE(octet_length(billing_identity::text), 0) AS input_bytes
    FROM pricing_candidates
),
bounded_candidates AS (
    SELECT
        effective_on,
        input_bytes > (SELECT max_input_bytes FROM scope) AS input_oversized,
        CASE
            WHEN input_bytes <= (SELECT max_input_bytes FROM scope)
            THEN usage_metrics
        END AS usage_metrics,
        CASE
            WHEN input_bytes <= (SELECT max_input_bytes FROM scope)
            THEN billing_identity
        END AS billing_identity
    FROM measured_candidates
),
pricing_groups AS (
    SELECT
        effective_on,
        usage_metrics,
        billing_identity,
        input_oversized,
        COUNT(*) AS occurrences
    FROM bounded_candidates
    GROUP BY effective_on, usage_metrics, billing_identity, input_oversized
)
SELECT
    effective_on,
    usage_metrics,
    billing_identity,
    input_oversized,
    occurrences,
    COUNT(*) OVER () AS raw_group_count
FROM pricing_groups
LIMIT %s
"""


_TOKEN_PATHS = {
    "input_tokens": "{usage_metrics,input_tokens}",
    "output_tokens": "{usage_metrics,output_tokens}",
    "total_tokens": "{usage_metrics,total_tokens}",
    "reasoning_tokens": "{usage_metrics,reasoning_output_tokens}",
    "cache_read_tokens": "{usage_metrics,cache,read_tokens}",
    "cache_write_tokens": "{usage_metrics,cache,write_tokens}",
    "cache_write_5m_tokens": "{usage_metrics,cache,write_5m_tokens}",
    "cache_write_1h_tokens": "{usage_metrics,cache,write_1h_tokens}",
    "cache_write_unknown_tokens": "{usage_metrics,cache,write_unknown_ttl_tokens}",
    "cached_input_tokens": "{usage_metrics,cache,cached_input_tokens}",
    "uncached_input_tokens": "{usage_metrics,cache,uncached_input_tokens}",
}

_ROW_COLUMNS = (
    "section",
    "provider_name",
    "model",
    "session_count",
    "model_steps",
    "model_steps_with_usage",
    "tool_calls",
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
    "group_count",
    "effective_on",
    "occurrences",
    "usage_metrics",
    "billing_identity",
    "matching_session_count",
    "active_session_count",
)


async def aggregate_session_usage(
    connection: Any,
    *,
    session_plan: SessionQuerySqlPlan,
    query: UsageRollupQuery,
    as_of: datetime,
) -> UsageRollupStoreResult:
    sql, params = usage_rollup_statement(session_plan=session_plan, query=query)
    async with connection.cursor() as cursor:
        await cursor.execute(cast("LiteralString", sql), params)
        raw_rows = await cursor.fetchall()
    rows = [dict(zip(_ROW_COLUMNS, row, strict=True)) for row in raw_rows]
    pricing = BoundedUsagePricingInputAccumulator(query.pricing_input_limit)
    if query.include_pricing_inputs:
        pricing_sql, pricing_params = pricing_input_statement(
            session_plan=session_plan,
            query=query,
        )
        async with connection.cursor(name="cayu_usage_rollup_pricing_inputs") as cursor:
            await cursor.execute(cast("LiteralString", pricing_sql), pricing_params)
            while not pricing.truncated:
                row = await cursor.fetchone()
                if row is None:
                    break
                _add_pricing_input_from_values(pricing, row)
    return _result_from_rows(
        rows,
        query=query,
        as_of=as_of,
        pricing=pricing.result(),
    )


def usage_rollup_statement(
    *,
    session_plan: SessionQuerySqlPlan,
    query: UsageRollupQuery,
) -> tuple[str, tuple[object, ...]]:
    """Build the exact PostgreSQL statement used by the aggregate read path."""

    sql = _RESULT_SQL.format(
        session_where_sql=session_plan.filter_where_sql,
        **{name: _nonnegative_json_integer(path) for name, path in _TOKEN_PATHS.items()},
    )
    return (
        sql,
        (
            query.group_limit,
            AGGREGATE_IDENTITY_TRIM_CHARACTERS,
            *session_plan.filter_params,
            query.start_at,
            query.end_at,
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
        _PRICING_INPUT_SQL.format(session_where_sql=session_plan.filter_where_sql),
        (
            max_input_bytes,
            *session_plan.filter_params,
            query.start_at,
            query.end_at,
            MAX_USAGE_PRICING_RAW_CANDIDATES,
        ),
    )


def _nonnegative_json_integer(path: str) -> str:
    return f"""
        CASE
            WHEN jsonb_typeof(event.payload #> '{path}') = 'number'
             AND (event.payload #>> '{path}') ~ '^[0-9]+$'
             AND (
                 length(event.payload #>> '{path}') < 19
                 OR (
                     length(event.payload #>> '{path}') = 19
                     AND (event.payload #>> '{path}') COLLATE "C"
                         <= '{MAX_AGGREGATE_USAGE_COUNTER}'
                 )
             )
            THEN (event.payload #>> '{path}')::numeric
            ELSE 0::numeric
        END
    """.strip()


def _result_from_rows(
    rows: list[dict[str, Any]],
    *,
    query: UsageRollupQuery,
    as_of: datetime,
    pricing: tuple[tuple[UsagePricingInput, ...], int, AggregateAccuracy],
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

    return UsageRollupStoreResult(
        as_of=as_of.astimezone(UTC),
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
        pricing_inputs=tuple(pricing_inputs),
        pricing_inputs_included=query.include_pricing_inputs,
        pricing_input_group_count=pricing_group_count,
        pricing_inputs_accuracy=pricing_accuracy,
        active_session_count=int(overall["active_session_count"]),
        matching_session_count=int(overall["matching_session_count"]),
    )


def _ordered_group_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
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
    rows: list[dict[str, Any]],
    *,
    remainder: dict[str, Any] | None,
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
            group_count=int(remainder["group_count"]),
            totals=_totals_from_row(remainder),
        ),
        accuracy=AggregateAccuracy(
            kind=AggregateAccuracyKind.TRUNCATED,
            reason=f"Distinct {dimension} groups exceed group_limit.",
            limit=limit,
        ),
    )


def _totals_from_row(row: dict[str, Any]) -> UsageAggregateTotals:
    return UsageAggregateTotals(
        session_count=int(row["session_count"]),
        model_steps=int(row["model_steps"]),
        model_steps_with_usage=int(row["model_steps_with_usage"]),
        tool_calls=int(row["tool_calls"]),
        usage=UsageMetrics(
            input_tokens=int(row["input_tokens"]),
            output_tokens=int(row["output_tokens"]),
            total_tokens=int(row["total_tokens"]),
            reasoning_output_tokens=int(row["reasoning_output_tokens"]),
            cache=CacheUsageMetrics(
                read_tokens=int(row["cache_read_tokens"]),
                write_tokens=int(row["cache_write_tokens"]),
                write_5m_tokens=int(row["cache_write_5m_tokens"]),
                write_1h_tokens=int(row["cache_write_1h_tokens"]),
                write_unknown_ttl_tokens=int(row["cache_write_unknown_ttl_tokens"]),
                cached_input_tokens=int(row["cached_input_tokens"]),
                uncached_input_tokens=int(row["uncached_input_tokens"]),
            ),
        ),
    )


def _add_pricing_input_from_row(
    pricing: BoundedUsagePricingInputAccumulator,
    row: dict[str, Any],
) -> None:
    if row["input_oversized"]:
        pricing.reject_oversized_candidate()
        return
    if int(row["raw_group_count"]) > MAX_USAGE_PRICING_RAW_CANDIDATES:
        pricing.reject_candidate_row_overflow(limit=MAX_USAGE_PRICING_RAW_CANDIDATES)
        return
    payload: dict[str, Any] = {}
    if row["usage_metrics"] is not None:
        payload["usage_metrics"] = row["usage_metrics"]
    if row["billing_identity"] is not None:
        payload["billing_identity"] = row["billing_identity"]
    pricing.add_payload(
        effective_on=row["effective_on"],
        occurrences=int(row["occurrences"]),
        payload=payload,
    )


def _add_pricing_input_from_values(
    pricing: BoundedUsagePricingInputAccumulator,
    row: tuple[object, ...],
) -> None:
    (
        effective_on,
        usage_metrics,
        billing_identity,
        input_oversized,
        occurrences,
        raw_group_count,
    ) = row
    _add_pricing_input_from_row(
        pricing,
        {
            "effective_on": effective_on,
            "usage_metrics": usage_metrics,
            "billing_identity": billing_identity,
            "input_oversized": input_oversized,
            "occurrences": occurrences,
            "raw_group_count": raw_group_count,
        },
    )

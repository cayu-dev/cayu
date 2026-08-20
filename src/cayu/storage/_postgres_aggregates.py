from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, LiteralString, cast

from cayu.core.events import EventType
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
    UsageSessionStatus,
    aggregate_identity_value,
    build_aggregate_usage_metrics,
)
from cayu.runtime.sessions import UsageRollupQuery
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
            WHEN event.payload -> 'usage_normalization_failed'
                     IS DISTINCT FROM 'true'::jsonb
             AND jsonb_typeof(event.payload -> 'usage_metrics') = 'object'
            THEN 1
            ELSE 0
        END AS has_usage,
        (CASE
            WHEN event.payload -> 'usage_normalization_failed'
                     IS DISTINCT FROM 'true'::jsonb
             AND jsonb_typeof(
                     event.payload #> '{{usage_metrics,provider_name}}'
                 ) = 'string'
             AND btrim(
                 event.payload #>> '{{usage_metrics,provider_name}}',
                 (SELECT identity_trim FROM scope)
             ) = event.payload #>> '{{usage_metrics,provider_name}}'
             AND event.payload #>> '{{usage_metrics,provider_name}}' <> ''
            THEN event.payload #>> '{{usage_metrics,provider_name}}'
            WHEN event.event_type = 'model.hosted_tool_call'
             AND jsonb_typeof(event.payload -> 'provider_name') = 'string'
             AND btrim(
                 event.payload ->> 'provider_name',
                 (SELECT identity_trim FROM scope)
             ) = event.payload ->> 'provider_name'
             AND event.payload ->> 'provider_name' <> ''
            THEN event.payload ->> 'provider_name'
        END) COLLATE "C" AS provider_name,
        (CASE
            WHEN event.payload -> 'usage_normalization_failed'
                     IS DISTINCT FROM 'true'::jsonb
             AND jsonb_typeof(event.payload #> '{{usage_metrics,model}}') = 'string'
             AND btrim(
                 event.payload #>> '{{usage_metrics,model}}',
                 (SELECT identity_trim FROM scope)
             ) = event.payload #>> '{{usage_metrics,model}}'
             AND event.payload #>> '{{usage_metrics,model}}' <> ''
            THEN event.payload #>> '{{usage_metrics,model}}'
            WHEN event.event_type = 'model.hosted_tool_call'
             AND jsonb_typeof(event.payload -> 'model') = 'string'
             AND btrim(
                 event.payload ->> 'model',
                 (SELECT identity_trim FROM scope)
             ) = event.payload ->> 'model'
             AND event.payload ->> 'model' <> ''
            THEN event.payload ->> 'model'
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
        {uncached_input_tokens} AS uncached_input_tokens,
        {web_search_calls} AS web_search_calls,
        {web_search_outcome_unknown} AS web_search_outcome_unknown
    FROM cayu_events AS event
    JOIN matched_sessions AS session ON session.id = event.session_id
    CROSS JOIN scope
    WHERE event.timestamp >= %s::timestamptz
      AND event.timestamp < %s::timestamptz
      AND event.event_type IN (
          'model.completed', 'tool.call.started', 'model.hosted_tool_call'
      )
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
            AS uncached_input_tokens,
        COALESCE(SUM(web_search_calls), 0) AS web_search_calls,
        COALESCE(SUM(web_search_outcome_unknown), 0) AS web_search_outcome_unknown
    FROM usage_events
),
provider_grouped AS (
    SELECT
        provider_name,
        COUNT(DISTINCT session_id) AS session_count,
        COUNT(*) FILTER (WHERE event_type = 'model.completed') AS model_steps,
        COUNT(*) FILTER (
            WHERE event_type = 'model.completed' AND has_usage = 1
        ) AS model_steps_with_usage,
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
        SUM(uncached_input_tokens) AS uncached_input_tokens,
        SUM(web_search_calls) AS web_search_calls,
        SUM(web_search_outcome_unknown) AS web_search_outcome_unknown
    FROM usage_events
    WHERE event_type = 'model.completed'
       OR web_search_calls > 0
       OR web_search_outcome_unknown > 0
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
        COUNT(*) FILTER (WHERE event.event_type = 'model.completed') AS model_steps,
        COUNT(*) FILTER (
            WHERE event.event_type = 'model.completed' AND event.has_usage = 1
        ) AS model_steps_with_usage,
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
        SUM(event.web_search_calls) AS web_search_calls,
        SUM(event.web_search_outcome_unknown) AS web_search_outcome_unknown,
        MAX(ranked.total_groups) - (SELECT group_limit FROM scope) AS group_count
    FROM usage_events AS event
    JOIN provider_ranked AS ranked
      ON event.provider_name IS NOT DISTINCT FROM ranked.provider_name
    WHERE (event.event_type = 'model.completed'
           OR event.web_search_calls > 0
           OR event.web_search_outcome_unknown > 0)
      AND ranked.group_rank > (SELECT group_limit FROM scope)
    HAVING COUNT(*) > 0
),
model_grouped AS (
    SELECT
        provider_name,
        model,
        COUNT(DISTINCT session_id) AS session_count,
        COUNT(*) FILTER (WHERE event_type = 'model.completed') AS model_steps,
        COUNT(*) FILTER (
            WHERE event_type = 'model.completed' AND has_usage = 1
        ) AS model_steps_with_usage,
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
        SUM(uncached_input_tokens) AS uncached_input_tokens,
        SUM(web_search_calls) AS web_search_calls,
        SUM(web_search_outcome_unknown) AS web_search_outcome_unknown
    FROM usage_events
    WHERE event_type = 'model.completed'
       OR web_search_calls > 0
       OR web_search_outcome_unknown > 0
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
        COUNT(*) FILTER (WHERE event.event_type = 'model.completed') AS model_steps,
        COUNT(*) FILTER (
            WHERE event.event_type = 'model.completed' AND event.has_usage = 1
        ) AS model_steps_with_usage,
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
        SUM(event.web_search_calls) AS web_search_calls,
        SUM(event.web_search_outcome_unknown) AS web_search_outcome_unknown,
        MAX(ranked.total_groups) - (SELECT group_limit FROM scope) AS group_count
    FROM usage_events AS event
    JOIN model_ranked AS ranked
      ON event.provider_name IS NOT DISTINCT FROM ranked.provider_name
     AND event.model IS NOT DISTINCT FROM ranked.model
    WHERE (event.event_type = 'model.completed'
           OR event.web_search_calls > 0
           OR event.web_search_outcome_unknown > 0)
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
    overall.web_search_calls,
    overall.web_search_outcome_unknown,
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
    cached_input_tokens, uncached_input_tokens, web_search_calls,
    web_search_outcome_unknown, total_groups, NULL, 0, NULL, NULL, 0, 0
FROM provider_ranked
WHERE group_rank <= (SELECT group_limit FROM scope)
UNION ALL
SELECT
    'provider_remainder', NULL, NULL, session_count, model_steps,
    model_steps_with_usage, 0, input_tokens, output_tokens, total_tokens,
    reasoning_output_tokens, cache_read_tokens, cache_write_tokens,
    cache_write_5m_tokens, cache_write_1h_tokens, cache_write_unknown_ttl_tokens,
    cached_input_tokens, uncached_input_tokens, web_search_calls,
    web_search_outcome_unknown, group_count, NULL, 0, NULL, NULL, 0, 0
FROM provider_remainder
UNION ALL
SELECT
    'model', provider_name, model, session_count, model_steps,
    model_steps_with_usage, 0, input_tokens, output_tokens, total_tokens,
    reasoning_output_tokens, cache_read_tokens, cache_write_tokens,
    cache_write_5m_tokens, cache_write_1h_tokens, cache_write_unknown_ttl_tokens,
    cached_input_tokens, uncached_input_tokens, web_search_calls,
    web_search_outcome_unknown, total_groups, NULL, 0, NULL, NULL, 0, 0
FROM model_ranked
WHERE group_rank <= (SELECT group_limit FROM scope)
UNION ALL
SELECT
    'model_remainder', NULL, NULL, session_count, model_steps,
    model_steps_with_usage, 0, input_tokens, output_tokens, total_tokens,
    reasoning_output_tokens, cache_read_tokens, cache_write_tokens,
    cache_write_5m_tokens, cache_write_1h_tokens, cache_write_unknown_ttl_tokens,
    cached_input_tokens, uncached_input_tokens, web_search_calls,
    web_search_outcome_unknown, group_count, NULL, 0, NULL, NULL, 0, 0
FROM model_remainder
"""

_PRICING_INPUT_SQL = """
WITH
scope(max_input_bytes, identity_trim) AS (
    SELECT %s::bigint, %s::text
),
matched_sessions AS MATERIALIZED (
    SELECT id
    FROM cayu_sessions
    {session_where_sql}
),
pricing_candidates AS (
    SELECT
        event.event_type,
        (event.timestamp AT TIME ZONE 'UTC')::date AS effective_on,
        {usage_metrics_projection} AS usage_metrics,
        {billing_identity_projection} AS billing_identity
    FROM cayu_events AS event
    JOIN matched_sessions AS session ON session.id = event.session_id
    WHERE event.timestamp >= %s::timestamptz
      AND event.timestamp < %s::timestamptz
      AND (
          event.event_type = 'model.completed'
          OR (
              event.event_type = 'model.hosted_tool_call'
              AND event.payload ->> 'tool_type' = 'web_search'
              AND event.payload ->> 'status' IN (
                  'completed', 'incomplete', 'failed', 'outcome_unknown'
              )
          )
      )
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
        event_type,
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
        event_type,
        effective_on,
        usage_metrics,
        billing_identity,
        input_oversized,
        COUNT(*) AS occurrences
    FROM bounded_candidates
    GROUP BY event_type, effective_on, usage_metrics, billing_identity, input_oversized
)
SELECT
    event_type,
    effective_on,
    usage_metrics,
    billing_identity,
    input_oversized,
    occurrences,
    COUNT(*) OVER () AS raw_group_count
FROM pricing_groups
LIMIT %s
"""

_SESSION_PRICING_INPUT_SQL = """
WITH
scope(max_input_bytes, identity_trim) AS (
    SELECT %s::bigint, %s::text
),
requested_sessions(session_id) AS (
    SELECT unnest(%s::text[])
),
matched_sessions AS MATERIALIZED (
    SELECT cayu_sessions.id
    FROM cayu_sessions
    JOIN requested_sessions AS requested
      ON requested.session_id = cayu_sessions.id
    {session_where_sql}
),
pricing_candidates AS (
    SELECT
        event.event_type,
        event.session_id,
        (event.timestamp AT TIME ZONE 'UTC')::date AS effective_on,
        {usage_metrics_projection} AS usage_metrics,
        {billing_identity_projection} AS billing_identity
    FROM cayu_events AS event
    JOIN matched_sessions AS session ON session.id = event.session_id
    WHERE event.timestamp >= %s::timestamptz
      AND event.timestamp < %s::timestamptz
      AND (
          event.event_type = 'model.completed'
          OR (
              event.event_type = 'model.hosted_tool_call'
              AND event.payload ->> 'tool_type' = 'web_search'
              AND event.payload ->> 'status' IN (
                  'completed', 'incomplete', 'failed', 'outcome_unknown'
              )
          )
      )
),
measured_candidates AS (
    SELECT
        *,
        octet_length(session_id)
            + COALESCE(octet_length(usage_metrics::text), 0)
            + COALESCE(octet_length(billing_identity::text), 0) AS input_bytes
    FROM pricing_candidates
),
bounded_candidates AS (
    SELECT
        event_type,
        session_id,
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
        event_type,
        session_id,
        effective_on,
        usage_metrics,
        billing_identity,
        input_oversized,
        COUNT(*) AS occurrences
    FROM bounded_candidates
    GROUP BY
        event_type,
        session_id,
        effective_on,
        usage_metrics,
        billing_identity,
        input_oversized
)
SELECT
    event_type,
    session_id,
    effective_on,
    usage_metrics,
    billing_identity,
    input_oversized,
    occurrences,
    COUNT(*) OVER () AS raw_group_count
FROM pricing_groups
LIMIT %s
"""

_SESSION_BREAKDOWN_SQL = """
WITH
scope(session_group_limit) AS (
    SELECT %s::integer
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
            WHEN event.payload -> 'usage_normalization_failed'
                     IS DISTINCT FROM 'true'::jsonb
             AND jsonb_typeof(event.payload -> 'usage_metrics') = 'object'
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
        {uncached_input_tokens} AS uncached_input_tokens,
        {web_search_calls} AS web_search_calls,
        {web_search_outcome_unknown} AS web_search_outcome_unknown
    FROM cayu_events AS event
    JOIN matched_sessions AS session ON session.id = event.session_id
    WHERE event.timestamp >= %s::timestamptz
      AND event.timestamp < %s::timestamptz
      AND event.event_type IN (
          'model.completed', 'tool.call.started', 'model.hosted_tool_call'
      )
),
session_grouped AS (
    SELECT
        session.id AS session_id,
        session.status,
        session.status IN ('pending', 'running', 'interrupting') AS active,
        CASE WHEN COUNT(event.session_id) > 0 THEN 1 ELSE 0 END AS session_count,
        COUNT(*) FILTER (WHERE event.event_type = 'model.completed') AS model_steps,
        COUNT(*) FILTER (
            WHERE event.event_type = 'model.completed' AND event.has_usage = 1
        ) AS model_steps_with_usage,
        COUNT(*) FILTER (WHERE event.event_type = 'tool.call.started') AS tool_calls,
        COALESCE(SUM(input_tokens) FILTER (
            WHERE event.event_type = 'model.completed'
        ), 0) AS input_tokens,
        COALESCE(SUM(output_tokens) FILTER (
            WHERE event.event_type = 'model.completed'
        ), 0) AS output_tokens,
        COALESCE(SUM(total_tokens) FILTER (
            WHERE event.event_type = 'model.completed'
        ), 0) AS total_tokens,
        COALESCE(SUM(reasoning_output_tokens) FILTER (
            WHERE event.event_type = 'model.completed'
        ), 0) AS reasoning_output_tokens,
        COALESCE(SUM(cache_read_tokens) FILTER (
            WHERE event.event_type = 'model.completed'
        ), 0) AS cache_read_tokens,
        COALESCE(SUM(cache_write_tokens) FILTER (
            WHERE event.event_type = 'model.completed'
        ), 0) AS cache_write_tokens,
        COALESCE(SUM(cache_write_5m_tokens) FILTER (
            WHERE event.event_type = 'model.completed'
        ), 0) AS cache_write_5m_tokens,
        COALESCE(SUM(cache_write_1h_tokens) FILTER (
            WHERE event.event_type = 'model.completed'
        ), 0) AS cache_write_1h_tokens,
        COALESCE(SUM(cache_write_unknown_ttl_tokens) FILTER (
            WHERE event.event_type = 'model.completed'
        ), 0) AS cache_write_unknown_ttl_tokens,
        COALESCE(SUM(cached_input_tokens) FILTER (
            WHERE event.event_type = 'model.completed'
        ), 0) AS cached_input_tokens,
        COALESCE(SUM(uncached_input_tokens) FILTER (
            WHERE event.event_type = 'model.completed'
        ), 0) AS uncached_input_tokens,
        COALESCE(SUM(web_search_calls), 0) AS web_search_calls,
        COALESCE(SUM(web_search_outcome_unknown), 0) AS web_search_outcome_unknown
    FROM matched_sessions AS session
    LEFT JOIN usage_events AS event ON event.session_id = session.id
    GROUP BY session.id, session.status
),
retained_sessions AS MATERIALIZED (
    SELECT
        *
    FROM session_grouped
    ORDER BY
        total_tokens DESC,
        model_steps DESC,
        session_id COLLATE "C" ASC
    LIMIT (SELECT session_group_limit FROM scope)
),
session_identity_bound AS (
    SELECT EXISTS (
        SELECT 1
        FROM retained_sessions
        WHERE octet_length(session_id) > {session_id_max_bytes}
    ) AS exceeded
),
session_remainder AS (
    SELECT
        COALESCE(SUM(session.session_count), 0) AS session_count,
        COALESCE(SUM(session.model_steps), 0) AS model_steps,
        COALESCE(SUM(session.model_steps_with_usage), 0) AS model_steps_with_usage,
        COALESCE(SUM(session.tool_calls), 0) AS tool_calls,
        COALESCE(SUM(session.input_tokens), 0) AS input_tokens,
        COALESCE(SUM(session.output_tokens), 0) AS output_tokens,
        COALESCE(SUM(session.total_tokens), 0) AS total_tokens,
        COALESCE(SUM(session.reasoning_output_tokens), 0) AS reasoning_output_tokens,
        COALESCE(SUM(session.cache_read_tokens), 0) AS cache_read_tokens,
        COALESCE(SUM(session.cache_write_tokens), 0) AS cache_write_tokens,
        COALESCE(SUM(session.cache_write_5m_tokens), 0) AS cache_write_5m_tokens,
        COALESCE(SUM(session.cache_write_1h_tokens), 0) AS cache_write_1h_tokens,
        COALESCE(SUM(session.cache_write_unknown_ttl_tokens), 0)
            AS cache_write_unknown_ttl_tokens,
        COALESCE(SUM(session.cached_input_tokens), 0) AS cached_input_tokens,
        COALESCE(SUM(session.uncached_input_tokens), 0) AS uncached_input_tokens,
        COALESCE(SUM(session.web_search_calls), 0) AS web_search_calls,
        COALESCE(SUM(session.web_search_outcome_unknown), 0)
            AS web_search_outcome_unknown,
        COUNT(*) AS group_count,
        COUNT(*) FILTER (WHERE session.active) AS active_session_count
    FROM session_grouped AS session
    WHERE NOT EXISTS (
        SELECT 1
        FROM retained_sessions AS retained
        WHERE retained.session_id = session.session_id
    )
    HAVING COUNT(*) > 0
)
SELECT
    'session'::text AS section,
    CASE
        WHEN octet_length(session_id) <= {session_id_max_bytes}
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
    web_search_calls,
    web_search_outcome_unknown,
    0::bigint AS group_count,
    0::bigint AS active_session_count
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
    web_search_calls,
    web_search_outcome_unknown,
    group_count,
    active_session_count
FROM session_remainder
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
    "web_search_calls",
    "web_search_outcome_unknown",
    "group_count",
    "effective_on",
    "occurrences",
    "usage_metrics",
    "billing_identity",
    "matching_session_count",
    "active_session_count",
)
_SESSION_ROW_COLUMNS = (
    "section",
    "session_id",
    "session_id_too_large",
    "status",
    "active",
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
    "web_search_calls",
    "web_search_outcome_unknown",
    "group_count",
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
    session_breakdown = None
    if query.session_group_limit is not None:
        session_sql, session_params = session_breakdown_statement(
            session_plan=session_plan,
            query=query,
        )
        async with connection.cursor() as cursor:
            await cursor.execute(cast("LiteralString", session_sql), session_params)
            raw_session_rows = await cursor.fetchall()
        session_breakdown = _session_breakdown_from_rows(
            [dict(zip(_SESSION_ROW_COLUMNS, row, strict=True)) for row in raw_session_rows],
            limit=query.session_group_limit,
        )
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
    session_pricing = BoundedUsagePricingInputAccumulator(query.pricing_input_limit)
    if query.include_pricing_inputs and session_breakdown is not None and session_breakdown.groups:
        session_pricing_sql, session_pricing_params = session_pricing_input_statement(
            session_plan=session_plan,
            query=query,
            session_ids=tuple(group.session_id for group in session_breakdown.groups),
        )
        async with connection.cursor(name="cayu_usage_rollup_session_pricing_inputs") as cursor:
            await cursor.execute(
                cast("LiteralString", session_pricing_sql),
                session_pricing_params,
            )
            while not session_pricing.truncated:
                row = await cursor.fetchone()
                if row is None:
                    break
                _add_session_pricing_input_from_values(session_pricing, row)
    return _result_from_rows(
        rows,
        query=query,
        as_of=as_of,
        pricing=pricing.result(),
        session_pricing=session_pricing.result(),
        session_breakdown=session_breakdown,
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
        web_search_calls=_web_search_terminal_count(),
        web_search_outcome_unknown=_web_search_unknown_count(),
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


def session_breakdown_statement(
    *,
    session_plan: SessionQuerySqlPlan,
    query: UsageRollupQuery,
) -> tuple[str, tuple[object, ...]]:
    if query.session_group_limit is None:
        raise ValueError("session_group_limit is required for a session breakdown.")
    return (
        _SESSION_BREAKDOWN_SQL.format(
            session_where_sql=session_plan.filter_where_sql,
            session_id_max_bytes=MAX_USAGE_ROLLUP_SESSION_ID_BYTES,
            **{name: _nonnegative_json_integer(path) for name, path in _TOKEN_PATHS.items()},
            web_search_calls=_web_search_terminal_count(),
            web_search_outcome_unknown=_web_search_unknown_count(),
        ),
        (
            query.session_group_limit,
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
        _PRICING_INPUT_SQL.format(
            session_where_sql=session_plan.filter_where_sql,
            usage_metrics_projection=_postgres_usage_metrics_projection(),
            billing_identity_projection=_postgres_billing_identity_projection(
                ("billing_identity",)
            ),
        ),
        (
            max_input_bytes,
            AGGREGATE_IDENTITY_TRIM_CHARACTERS,
            *session_plan.filter_params,
            query.start_at,
            query.end_at,
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
            session_where_sql=session_plan.filter_where_sql,
            usage_metrics_projection=_postgres_usage_metrics_projection(),
            billing_identity_projection=_postgres_billing_identity_projection(
                ("billing_identity",)
            ),
        ),
        (
            max_input_bytes,
            AGGREGATE_IDENTITY_TRIM_CHARACTERS,
            list(session_ids),
            *session_plan.filter_params,
            query.start_at,
            query.end_at,
            MAX_USAGE_PRICING_RAW_CANDIDATES + 1,
        ),
    )


def _postgres_usage_metrics_projection() -> str:
    metrics_path = ("usage_metrics",)
    identity_path = (*metrics_path, "billing_identity")
    metrics = f"event.payload #> {_postgres_json_path(metrics_path)}"
    identity = _postgres_billing_identity_projection(identity_path)
    projected_metrics = f"""
        CASE
            WHEN jsonb_typeof(
                event.payload #> {_postgres_json_path(identity_path)}
            ) = 'object'
            THEN jsonb_set(
                {metrics},
                '{{billing_identity}}',
                {identity},
                false
            )
            ELSE {metrics}
        END
    """.strip()
    return f"""
        CASE
            WHEN event.event_type = 'model.hosted_tool_call'
             AND event.payload ->> 'tool_type' = 'web_search'
             AND event.payload ->> 'status' IN (
                 'completed', 'incomplete', 'failed', 'outcome_unknown'
             )
            THEN jsonb_build_object(
                'provider_name', event.payload -> 'provider_name',
                'model', event.payload -> 'model',
                'hosted_tools', jsonb_build_object(
                    'web_search_calls', CASE
                        WHEN event.payload ->> 'status' = 'outcome_unknown'
                        THEN 0 ELSE 1 END,
                    'web_search_outcome_unknown', CASE
                        WHEN event.payload ->> 'status' = 'outcome_unknown'
                        THEN 1 ELSE 0 END
                )
            )
            WHEN event.event_type = 'model.completed'
             AND event.payload -> 'usage_normalization_failed'
                     IS DISTINCT FROM 'true'::jsonb
             AND jsonb_typeof({metrics}) = 'object'
            THEN {projected_metrics}
        END
    """.strip()


def _postgres_billing_identity_projection(identity_path: tuple[str, ...]) -> str:
    identity = f"(event.payload #> {_postgres_json_path(identity_path)})"
    request_evidence = _postgres_text_evidence_projection(
        (*identity_path, "request_evidence"),
        _BEDROCK_AGGREGATE_REQUEST_EVIDENCE,
    )
    completion_evidence = _postgres_text_evidence_projection(
        (*identity_path, "completion_evidence"),
        _BEDROCK_AGGREGATE_COMPLETION_EVIDENCE,
    )
    provider_path = _postgres_json_path((*identity_path, "provider_name"))
    return f"""
        CASE
            WHEN jsonb_typeof({identity}) = 'object'
            THEN CASE
                WHEN event.payload #>> {provider_path} = 'bedrock'
                THEN ({identity} - 'request_evidence' - 'completion_evidence')
                     || jsonb_strip_nulls(
                         jsonb_build_object(
                             'request_evidence', NULLIF(
                                 {request_evidence},
                                 '{{}}'::jsonb
                             ),
                             'completion_evidence', NULLIF(
                                 {completion_evidence},
                                 '{{}}'::jsonb
                             )
                         )
                     )
                ELSE {identity} - 'request_evidence' - 'completion_evidence'
            END
        END
    """.strip()


def _postgres_text_evidence_projection(
    evidence_path: tuple[str, ...],
    fields: tuple[str, ...],
) -> str:
    entries: list[str] = []
    for field in fields:
        path = _postgres_json_path((*evidence_path, field))
        value = f"event.payload #>> {path}"
        entries.append(
            f"""
                '{field}', CASE
                    WHEN jsonb_typeof(event.payload #> {path}) = 'string'
                     AND {value} <> ''
                     AND btrim(
                         {value},
                         (SELECT identity_trim FROM scope)
                     ) = {value}
                    THEN event.payload #> {path}
                END
            """.strip()
        )
    return f"jsonb_strip_nulls(jsonb_build_object({', '.join(entries)}))"


def _postgres_json_path(parts: tuple[str, ...]) -> str:
    return "'{" + ",".join(parts) + "}'"


def _nonnegative_json_integer(path: str) -> str:
    return f"""
        CASE
            WHEN event.payload -> 'usage_normalization_failed'
                     IS DISTINCT FROM 'true'::jsonb
             AND jsonb_typeof(event.payload #> '{path}') = 'number'
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


def _web_search_unknown_count() -> str:
    return """
        CASE
            WHEN event.event_type = 'model.hosted_tool_call'
             AND event.payload ->> 'tool_type' = 'web_search'
             AND event.payload ->> 'status' = 'outcome_unknown'
            THEN 1::numeric
            ELSE 0::numeric
        END
    """.strip()


def _web_search_terminal_count() -> str:
    return """
        CASE
            WHEN event.event_type = 'model.hosted_tool_call'
             AND event.payload ->> 'tool_type' = 'web_search'
             AND event.payload ->> 'status' IN ('completed', 'incomplete', 'failed')
            THEN 1::numeric
            ELSE 0::numeric
        END
    """.strip()


def _result_from_rows(
    rows: list[dict[str, Any]],
    *,
    query: UsageRollupQuery,
    as_of: datetime,
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
        active_session_count=int(overall["active_session_count"]),
        matching_session_count=int(overall["matching_session_count"]),
    )


def _session_breakdown_from_rows(
    rows: list[dict[str, Any]],
    *,
    limit: int,
) -> UsageSessionAggregateBreakdown:
    if any(bool(row["session_id_too_large"]) for row in rows):
        raise UsageRollupResultTooLarge("A session identity exceeds the usage-rollup byte limit.")
    groups = tuple(
        UsageSessionAggregateGroup(
            session_id=str(row["session_id"]),
            status=cast("UsageSessionStatus", str(row["status"])),
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
            group_count=int(remainder_row["group_count"]),
            active_session_count=int(remainder_row["active_session_count"]),
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
        usage=build_aggregate_usage_metrics(
            input_tokens=int(row["input_tokens"]),
            output_tokens=int(row["output_tokens"]),
            total_tokens=int(row["total_tokens"]),
            reasoning_output_tokens=int(row["reasoning_output_tokens"]),
            cache_read_tokens=int(row["cache_read_tokens"]),
            cache_write_tokens=int(row["cache_write_tokens"]),
            cache_write_5m_tokens=int(row["cache_write_5m_tokens"]),
            cache_write_1h_tokens=int(row["cache_write_1h_tokens"]),
            cache_write_unknown_ttl_tokens=int(row["cache_write_unknown_ttl_tokens"]),
            cached_input_tokens=int(row["cached_input_tokens"]),
            uncached_input_tokens=int(row["uncached_input_tokens"]),
            web_search_calls=int(row["web_search_calls"]),
            web_search_outcome_unknown=int(row["web_search_outcome_unknown"]),
        ),
    )


def _add_pricing_input_from_row(
    pricing: BoundedUsagePricingInputAccumulator,
    row: dict[str, Any],
    *,
    session_id: str | None = None,
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
        event_type=EventType(row["event_type"]),
        session_id=session_id,
        effective_on=row["effective_on"],
        occurrences=int(row["occurrences"]),
        payload=payload,
    )


def _add_pricing_input_from_values(
    pricing: BoundedUsagePricingInputAccumulator,
    row: tuple[object, ...],
) -> None:
    (
        event_type,
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
            "event_type": event_type,
            "effective_on": effective_on,
            "usage_metrics": usage_metrics,
            "billing_identity": billing_identity,
            "input_oversized": input_oversized,
            "occurrences": occurrences,
            "raw_group_count": raw_group_count,
        },
    )


def _add_session_pricing_input_from_values(
    pricing: BoundedUsagePricingInputAccumulator,
    row: tuple[object, ...],
) -> None:
    (
        event_type,
        session_id,
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
            "event_type": event_type,
            "effective_on": effective_on,
            "usage_metrics": usage_metrics,
            "billing_identity": billing_identity,
            "input_oversized": input_oversized,
            "occurrences": occurrences,
            "raw_group_count": raw_group_count,
        },
        session_id=str(session_id),
    )

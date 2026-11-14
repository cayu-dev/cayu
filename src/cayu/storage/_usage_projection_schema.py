"""Stored usage and tool totals, maintained by triggers on ``cayu_usage_events``.

Usage and Tools pages used to add up events on every request. Storage revision
``USAGE_PROJECTION_REVISION`` keeps the sums they show as rows, written in the
same transaction as the event by triggers on ``cayu_usage_events`` (itself kept
by triggers on ``cayu_events``), so every writer, older binaries included, keeps
them current:

``cayu_session_usage``
    One row per session, UTC hour, event type, price identity and input band,
    with the event count and every rollup counter. Rollups filtered by session
    attributes, the per-session ranking and per-session totals read it.
``cayu_usage_hourly``
    The same sums per UTC hour, event agent, event environment, event type,
    price identity and input band. Unfiltered rollups read it.
``cayu_tool_usage``
    Calls, failures and approval decisions per tool name, event agent and UTC
    hour, with the newest call's time. Tools reads it.
``cayu_usage_price_identities``
    Interned price identities: the event's normalized usage metrics with
    every valid counter reduced to 0 or 1, plus its billing identity. Rows that
    share one fall into one price group of
    ``BoundedUsagePricingInputAccumulator``, so cost priced at read time from
    their summed counters equals cost priced event by event. Postgres may retain
    equivalent identities created by concurrent uncommitted writers. Each copy
    row binds the identity it used, so removal reverses the same group.

A row's input band counts the ``PRICE_INPUT_BANDS`` boundaries below its
events' input tokens, so with the usual price books no row spans a price tier.
Readers check each row's input-token range against the request's tier
boundaries and price a row that spans one from its events.

SQLite counters saturate at the int64 maximum and set ``overflow``; readers then
use the event-level reference. Postgres stores counters as NUMERIC.

The event-level rollup that ``_sqlite_aggregates`` and ``_postgres_aggregates``
ran before stays as the reference implementation, and ``cayu usage rollup
--recompute`` compares the two.
"""

from __future__ import annotations

from cayu.budgets.aggregates import MAX_AGGREGATE_USAGE_COUNTER
from cayu.storage._usage_event_schema import (
    USAGE_EVENT_TYPES,
    postgres_usage_event_function,
    postgres_usage_event_update_trigger,
    sqlite_usage_event_triggers,
)

#: The storage revision that adds the usage projections. Several sessions add
#: revisions at once, so keep this the only place that names the number.
USAGE_PROJECTION_REVISION = 157

#: Input-token boundaries that split projection rows. They are the tier
#: boundaries of the price books Cayu ships and documents; a request whose book
#: has another boundary prices the rows that span it from their events.
PRICE_INPUT_BANDS: tuple[int, ...] = (32_768, 100_000, 128_000, 200_000, 272_000)

#: Counters every projection row sums, in ``_USAGE_COUNTER_COLUMNS`` order. Token
#: counters count model steps and auxiliary attempts only, and web search counts
#: count hosted calls only, so a plain SUM gives what the rollup reports.
COUNTER_COLUMNS: tuple[str, ...] = (
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
)
_TOKEN_COLUMNS = COUNTER_COLUMNS[:11]
_WEB_SEARCH_COLUMNS = COUNTER_COLUMNS[11:]

#: The hosted web search counters a model step's own usage metrics carry. Pricing
#: reads them; rollups count web searches from hosted calls only.
METRICS_WEB_SEARCH_COLUMNS: tuple[str, ...] = (
    "metrics_web_search_calls",
    "metrics_web_search_outcome_unknown",
)

#: Every counter path of a usage-metrics object, in ``_usage_pricing_counters`` order.
PRICING_COUNTER_PATHS: tuple[tuple[str, ...], ...] = (
    ("input_tokens",),
    ("output_tokens",),
    ("total_tokens",),
    ("reasoning_output_tokens",),
    ("cache", "read_tokens"),
    ("cache", "write_tokens"),
    ("cache", "write_5m_tokens"),
    ("cache", "write_1h_tokens"),
    ("cache", "write_unknown_ttl_tokens"),
    ("cache", "cached_input_tokens"),
    ("cache", "uncached_input_tokens"),
    ("hosted_tools", "web_search_calls"),
    ("hosted_tools", "web_search_outcome_unknown"),
)

USAGE_TYPES: tuple[str, ...] = (
    "model.completed",
    "model.auxiliary.attempt_settled",
    "model.hosted_tool_call",
    "tool.call.started",
)
PRICED_TYPES: tuple[str, ...] = USAGE_TYPES[:3]
MODEL_TYPES: tuple[str, ...] = USAGE_TYPES[:2]
HOSTED = "model.hosted_tool_call"
HOSTED_WEB_SEARCH_TOOL_NAME = "openai_web_search"
_HOSTED_TERMINAL_STATUSES = ("completed", "failed", "incomplete", "outcome_unknown")

PROJECTION_TABLES: tuple[str, ...] = (
    "cayu_session_usage",
    "cayu_usage_hourly",
    "cayu_tool_usage",
    "cayu_usage_price_identities",
)
#: Postgres-only delta tables (see ``_postgres_compact_function``).
POSTGRES_PENDING_TABLES: tuple[str, ...] = ("cayu_usage_hourly_pending", "cayu_tool_usage_pending")

_MAX = MAX_AGGREGATE_USAGE_COUNTER
#: The transaction advisory lock key of Postgres compaction ("cayu usage").
_COMPACTION_LOCK = 0x6361797575736167
_COMMA_LINE = ",\n        "
_COMMA_SHORT = ",\n    "


def _sql_list(values: tuple[str, ...]) -> str:
    return ", ".join("'" + value.replace("'", "''") + "'" for value in values)


_USAGE_TYPES_SQL = _sql_list(USAGE_TYPES)
_PRICED_TYPES_SQL = _sql_list(PRICED_TYPES)
_MODEL_TYPES_SQL = _sql_list(MODEL_TYPES)
_ALL_COPIED_TYPES_SQL = _sql_list(USAGE_EVENT_TYPES)
_TERMINAL_SQL = _sql_list(_HOSTED_TERMINAL_STATUSES)

_SESSION_KEY = ("session_id", "hour_start", "event_type", "price_identity_id", "input_band")
_HOURLY_KEY = (
    "hour_start",
    "agent_name",
    "environment_name",
    "event_type",
    "price_identity_id",
    "input_band",
)
_TOOL_KEY = ("tool_name", "agent_name", "hour_start")
_TOOL_COUNTERS = ("calls", "failures", "approved", "rejected")

# Every column of the usage-projection ``cayu_usage_events`` copy, in SQLite.
_SQLITE_EVENT_COLUMNS = (
    "sequence",
    "session_id",
    "event_id",
    "event_type",
    "timestamp",
    "tool_name",
    "payload_json",
    "agent_name",
    "environment_name",
    "usage_eligible",
    "has_usage",
    "provider_name",
    "model",
    *_TOKEN_COLUMNS,
    *_WEB_SEARCH_COLUMNS,
    "usage_metrics_json",
    "billing_identity_json",
    "reported_cost",
)


def _band_sql(input_tokens: str, *, postgres: bool = False) -> str:
    cast = "::integer" if postgres else ""
    return (
        "(" + " + ".join(f"({input_tokens} > {bound}){cast}" for bound in PRICE_INPUT_BANDS) + ")"
    )


# --- SQLite -----------------------------------------------------------------


def _sqlite_row(alias: str) -> str:
    """A one-row SELECT of every copy column of trigger row ``alias`` (NEW or OLD)."""

    return "SELECT " + ", ".join(
        f"{alias}.{column} AS {column}" for column in _SQLITE_EVENT_COLUMNS
    )


def _sqlite_identity_metrics(metrics: str) -> str:
    """``metrics`` with every valid counter replaced by 0 or 1, and invalid ones by 'x'.

    A valid counter is a nonnegative JSON integer; strict validation rejects any
    other value, so ``'x'`` keeps the identity invalid exactly when an event's
    metrics are. Absent counters stay absent and validate as zero.
    """

    replacements = []
    for path in PRICING_COUNTER_PATHS:
        json_path = "'$." + ".".join(path) + "'"
        replacements.append(
            f"{json_path}, CASE WHEN json_type({metrics}, {json_path}) = 'integer' "
            f"AND json_extract({metrics}, {json_path}) >= 0 "
            f"AND json_extract({metrics}, {json_path}) <= {_MAX} "
            f"THEN min(json_extract({metrics}, {json_path}), 1) ELSE 'x' END"
        )
    return f"json_replace({metrics}, " + ", ".join(replacements) + ")"


def _sqlite_identity_key(alias: str) -> str:
    metrics = f"{alias}.usage_metrics_json"
    billing = f"{alias}.billing_identity_json"
    return (
        f"CASE WHEN {alias}.event_type IN ({_PRICED_TYPES_SQL}) "
        f"AND ({metrics} IS NOT NULL OR {billing} IS NOT NULL) "
        f"THEN COALESCE({_sqlite_identity_metrics(metrics)}, '~') || char(31) "
        f"|| COALESCE({billing}, '~') END"
    )


def _sqlite_nonnegative(path: str, payload: str) -> str:
    return f"""CASE
            WHEN json_type({payload}, '$.usage_normalization_failed') IS NOT 'true'
             AND json_type({payload}, '{path}') = 'integer'
             AND json_extract({payload}, '{path}') >= 0
            THEN json_extract({payload}, '{path}')
            ELSE 0
        END"""


def sqlite_usage_delta(source: str) -> str:
    """Project each usage row of ``source`` (copy columns, aliased ``e``) to its keys.

    One output row per usage-bearing event, with the projection keys and the
    amounts it adds to its rows.
    """

    tokens = ",\n        ".join(
        f"CASE WHEN e.event_type IN ({_MODEL_TYPES_SQL}) THEN e.{column} ELSE 0 END AS {column}"
        for column in _TOKEN_COLUMNS
    )
    searches = ",\n        ".join(
        f"CASE WHEN e.event_type = '{HOSTED}' THEN e.{column} ELSE 0 END AS {column}"
        for column in _WEB_SEARCH_COLUMNS
    )
    metric_searches = ",\n        ".join(
        f"CASE WHEN e.event_type = '{HOSTED}' THEN e.{column} "
        f"WHEN e.event_type IN ({_MODEL_TYPES_SQL}) "
        f"THEN {_sqlite_nonnegative(f'$.usage_metrics.hosted_tools.{column}', 'e.payload_json')} "
        f"ELSE 0 END AS metrics_{column}"
        for column in _WEB_SEARCH_COLUMNS
    )
    return f"""SELECT
        e.session_id AS session_id,
        substr(e.timestamp, 1, 13) || ':00:00+00:00' AS hour_start,
        e.event_type AS event_type,
        COALESCE((
            SELECT identity.id FROM cayu_usage_price_identities AS identity
            WHERE identity.identity_key = {_sqlite_identity_key("e")}
        ), 0) AS price_identity_id,
        {_band_sql("e.input_tokens")} AS input_band,
        e.provider_name AS provider_name,
        e.model AS model,
        COALESCE(e.agent_name, '') AS agent_name,
        COALESCE(e.environment_name, '') AS environment_name,
        e.has_usage AS has_usage_count,
        CASE WHEN e.event_type IN ({_MODEL_TYPES_SQL})
              AND json_type(e.payload_json, '$.usage_metrics') IS NULL
              AND json_type(e.payload_json, '$.usage_normalization_failed') IS NOT 'true'
             THEN 1 ELSE 0 END AS legacy_count,
        {tokens},
        {searches},
        {metric_searches},
        e.input_tokens AS event_input_tokens,
        COALESCE(length(CAST(e.usage_metrics_json AS BLOB)), 0)
            + COALESCE(length(CAST(e.billing_identity_json AS BLOB)), 0) AS pricing_input_bytes,
        e.sequence AS sequence
    FROM ({source}) AS e
    WHERE e.event_type IN ({_USAGE_TYPES_SQL}) AND e.usage_eligible = 1"""


def sqlite_tool_delta(source: str) -> str:
    """Project each tool usage row of ``source`` (copy columns, aliased ``e``)."""

    hosted_counted = (
        f"e.event_type = '{HOSTED}' "
        "AND json_type(e.payload_json, '$.tool_type') = 'text' "
        "AND json_extract(e.payload_json, '$.tool_type') = 'web_search' "
        "AND json_type(e.payload_json, '$.status') = 'text' "
        f"AND json_extract(e.payload_json, '$.status') IN ({_TERMINAL_SQL})"
    )
    call = f"e.event_type IN ('tool.call.completed', 'tool.call.failed', '{HOSTED}')"
    return f"""SELECT
        CASE WHEN e.event_type = '{HOSTED}' THEN '{HOSTED_WEB_SEARCH_TOOL_NAME}'
             ELSE e.tool_name END AS tool_name,
        COALESCE(e.agent_name, '') AS agent_name,
        substr(e.timestamp, 1, 13) || ':00:00+00:00' AS hour_start,
        CASE WHEN {call} THEN 1 ELSE 0 END AS calls,
        CASE WHEN e.event_type = 'tool.call.failed' THEN 1
             WHEN e.event_type = '{HOSTED}'
              AND json_extract(e.payload_json, '$.status') = 'failed' THEN 1
             ELSE 0 END AS failures,
        CASE WHEN e.event_type = 'tool.call.approved' THEN 1 ELSE 0 END AS approved,
        CASE WHEN e.event_type = 'tool.call.approval_denied' THEN 1 ELSE 0 END AS rejected,
        CASE WHEN {call} THEN e.timestamp END AS last_used_at
    FROM ({source}) AS e
    WHERE (
        (e.event_type IN ('tool.call.completed', 'tool.call.failed',
                          'tool.call.approved', 'tool.call.approval_denied')
         AND e.tool_name IS NOT NULL)
        OR ({hosted_counted})
    )"""


def _sqlite_saturating_add(column: str) -> str:
    return (
        f"{column} = CASE WHEN {column} > {_MAX} - excluded.{column} "
        f"THEN {_MAX} ELSE {column} + excluded.{column} END"
    )


def _sqlite_overflow(columns: tuple[str, ...]) -> str:
    return " OR ".join(f"{column} > {_MAX} - excluded.{column}" for column in columns)


_SUMMED = (*COUNTER_COLUMNS, *METRICS_WEB_SEARCH_COLUMNS)


def _sqlite_usage_upsert(table: str, key: tuple[str, ...], delta: str, *, session: bool) -> str:
    attributes = ("provider_name", "model")
    extremes = ("min_input_tokens", "max_input_tokens", "max_pricing_input_bytes")
    session_columns = ("legacy_count", "first_sequence") if session else ()
    columns = (
        *key,
        *attributes,
        "event_count",
        "has_usage_count",
        *(("legacy_count",) if session else ()),
        *_SUMMED,
        *extremes,
        *(("first_sequence",) if session else ()),
        "overflow",
    )
    values = (
        *(f"d.{column}" for column in key),
        "d.provider_name",
        "d.model",
        "1",
        "d.has_usage_count",
        *(("d.legacy_count",) if session else ()),
        *(f"d.{column}" for column in _SUMMED),
        "d.event_input_tokens",
        "d.event_input_tokens",
        "d.pricing_input_bytes",
        *(("d.sequence",) if session else ()),
        "0",
    )
    updates = [
        "event_count = event_count + 1",
        "has_usage_count = has_usage_count + excluded.has_usage_count",
        *(
            ("legacy_count = legacy_count + excluded.legacy_count",)
            if "legacy_count" in session_columns
            else ()
        ),
        f"overflow = overflow OR {_sqlite_overflow(_SUMMED)}",
        *(_sqlite_saturating_add(column) for column in _SUMMED),
        "min_input_tokens = min(min_input_tokens, excluded.min_input_tokens)",
        "max_input_tokens = max(max_input_tokens, excluded.max_input_tokens)",
        "max_pricing_input_bytes = max(max_pricing_input_bytes, excluded.max_pricing_input_bytes)",
        *(("first_sequence = min(first_sequence, excluded.first_sequence)",) if session else ()),
    ]
    return f"""INSERT INTO {table} ({", ".join(columns)})
    SELECT {", ".join(values)} FROM ({delta}) AS d WHERE true
    ON CONFLICT ({", ".join(key)}) DO UPDATE SET
        {_COMMA_LINE.join(updates)}"""


def _sqlite_usage_decrement(table: str, key: tuple[str, ...], delta: str, *, session: bool) -> str:
    match = " AND ".join(f"{table}.{column} = d.{column}" for column in key)
    updates = [
        f"event_count = {table}.event_count - 1",
        f"has_usage_count = {table}.has_usage_count - d.has_usage_count",
        *((f"legacy_count = {table}.legacy_count - d.legacy_count",) if session else ()),
        *(f"{column} = max({table}.{column} - d.{column}, 0)" for column in _SUMMED),
    ]
    return f"""UPDATE {table} SET
        {_COMMA_LINE.join(updates)}
    FROM ({delta}) AS d WHERE {match};
    DELETE FROM {table}
    WHERE {table}.event_count <= 0 AND EXISTS (
        SELECT 1 FROM ({delta}) AS d WHERE {match}
    )"""


def _sqlite_next_hour(hour: str) -> str:
    return f"strftime('%Y-%m-%dT%H:00:00+00:00', substr({hour}, 1, 19), '+1 hour')"


def _sqlite_session_first_stale(delta: str) -> str:
    """Flag a session row whose first event is being removed.

    Recomputing it would re-project the session's remaining events in that
    hour on every removal, so deleting a session would take time quadratic in
    its events. A flagged row's session reads its per-session totals from
    events. The input-token range is left as it was: it still covers every
    remaining event, and a range that is only too wide can only make a read
    price from events.
    """

    match = " AND ".join(f"cayu_session_usage.{column} = d.{column}" for column in _SESSION_KEY)
    return f"""UPDATE cayu_session_usage SET first_sequence_stale = 1
    FROM ({delta}) AS d
    WHERE {match} AND d.sequence <= cayu_session_usage.first_sequence"""


def _sqlite_tool_upsert(delta: str) -> str:
    return f"""INSERT INTO cayu_tool_usage (
        tool_name, agent_name, hour_start, calls, failures, approved, rejected, last_used_at
    )
    SELECT d.tool_name, d.agent_name, d.hour_start, d.calls, d.failures, d.approved,
           d.rejected, d.last_used_at
    FROM ({delta}) AS d WHERE true
    ON CONFLICT (tool_name, agent_name, hour_start) DO UPDATE SET
        calls = calls + excluded.calls,
        failures = failures + excluded.failures,
        approved = approved + excluded.approved,
        rejected = rejected + excluded.rejected,
        last_used_at = CASE
            WHEN last_used_at IS NULL THEN excluded.last_used_at
            WHEN excluded.last_used_at IS NULL THEN last_used_at
            ELSE max(last_used_at, excluded.last_used_at)
        END"""


def _sqlite_tool_last_call(row: str) -> str:
    """The newest call left for tool row ``row`` (alias of ``cayu_tool_usage``)."""

    hour = f"x.timestamp >= {row}.hour_start AND x.timestamp < {_sqlite_next_hour(row + '.hour_start')}"
    agent = f"COALESCE(x.agent_name, '') = {row}.agent_name"
    hosted = (
        f"SELECT MAX(x.timestamp) FROM cayu_usage_events AS x "
        f"WHERE {row}.tool_name = '{HOSTED_WEB_SEARCH_TOOL_NAME}' AND x.event_type = '{HOSTED}' "
        f"AND {hour} AND {agent} "
        "AND json_type(x.payload_json, '$.tool_type') = 'text' "
        "AND json_extract(x.payload_json, '$.tool_type') = 'web_search' "
        "AND json_type(x.payload_json, '$.status') = 'text' "
        f"AND json_extract(x.payload_json, '$.status') IN ({_TERMINAL_SQL})"
    )
    named = (
        "SELECT MAX(x.timestamp) FROM cayu_usage_events AS x "
        f"WHERE x.tool_name = {row}.tool_name "
        "AND x.event_type IN ('tool.call.completed', 'tool.call.failed') "
        f"AND {hour} AND {agent}"
    )
    return f"NULLIF(max(COALESCE(({hosted}), ''), COALESCE(({named}), '')), '')"


def _sqlite_tool_decrement(delta: str) -> str:
    match = " AND ".join(f"cayu_tool_usage.{column} = d.{column}" for column in _TOOL_KEY)
    return f"""UPDATE cayu_tool_usage SET
        calls = cayu_tool_usage.calls - d.calls,
        failures = cayu_tool_usage.failures - d.failures,
        approved = cayu_tool_usage.approved - d.approved,
        rejected = cayu_tool_usage.rejected - d.rejected
    FROM ({delta}) AS d WHERE {match};
    DELETE FROM cayu_tool_usage
    WHERE cayu_tool_usage.calls <= 0 AND cayu_tool_usage.failures <= 0 AND cayu_tool_usage.approved <= 0 AND cayu_tool_usage.rejected <= 0
      AND EXISTS (SELECT 1 FROM ({delta}) AS d WHERE {match});
    UPDATE cayu_tool_usage SET last_used_at = {_sqlite_tool_last_call("cayu_tool_usage")}
    FROM ({delta}) AS d
    WHERE {match} AND d.last_used_at IS NOT NULL AND d.last_used_at >= cayu_tool_usage.last_used_at"""


def _sqlite_identity_intern(source: str) -> str:
    """Record the price identity of each priced row of ``source`` (copy columns)."""

    # Not INSERT OR IGNORE: inside a trigger the outer statement's conflict
    # algorithm (the copy's INSERT OR REPLACE) would override it and renumber
    # the identity. A new key is inserted only when absent.
    return f"""INSERT INTO cayu_usage_price_identities (
        identity_key, usage_metrics_json, billing_identity_json
    )
    SELECT identity_key, MIN(usage_metrics_json), MIN(billing_identity_json) FROM (
        SELECT {_sqlite_identity_key("e")} AS identity_key,
               {_sqlite_identity_metrics("e.usage_metrics_json")} AS usage_metrics_json,
               e.billing_identity_json AS billing_identity_json
        FROM ({source}) AS e
        WHERE e.usage_eligible = 1
    ) AS candidate
    WHERE identity_key IS NOT NULL AND NOT EXISTS (
        SELECT 1 FROM cayu_usage_price_identities AS known
        WHERE known.identity_key = candidate.identity_key
    )
    GROUP BY identity_key"""


def _sqlite_add(source: str) -> str:
    usage = sqlite_usage_delta(source)
    return ";\n    ".join(
        (
            _sqlite_usage_upsert("cayu_session_usage", _SESSION_KEY, usage, session=True),
            _sqlite_usage_upsert("cayu_usage_hourly", _HOURLY_KEY, usage, session=False),
            _sqlite_tool_upsert(sqlite_tool_delta(source)),
        )
    )


def _sqlite_remove(source: str) -> str:
    usage = sqlite_usage_delta(source)
    return ";\n    ".join(
        (
            _sqlite_session_first_stale(usage),
            _sqlite_usage_decrement("cayu_session_usage", _SESSION_KEY, usage, session=True),
            _sqlite_usage_decrement("cayu_usage_hourly", _HOURLY_KEY, usage, session=False),
            _sqlite_tool_decrement(sqlite_tool_delta(source)),
        )
    )


_SQLITE_EXISTING = (
    "SELECT "
    + ", ".join(f"x.{column} AS {column}" for column in _SQLITE_EVENT_COLUMNS)
    + " FROM cayu_usage_events AS x WHERE x.sequence = NEW.sequence"
)


def _sqlite_usage_table(table: str, key: tuple[str, ...], *, session: bool) -> str:
    key_columns = {
        "session_id": "session_id TEXT NOT NULL",
        "hour_start": "hour_start TEXT NOT NULL",
        "agent_name": "agent_name TEXT NOT NULL",
        "environment_name": "environment_name TEXT NOT NULL",
        "event_type": "event_type TEXT NOT NULL",
        "price_identity_id": "price_identity_id INTEGER NOT NULL",
        "input_band": "input_band INTEGER NOT NULL",
    }
    counters = ",\n    ".join(f"{column} INTEGER NOT NULL" for column in _SUMMED)
    session_columns = (
        "legacy_count INTEGER NOT NULL,\n    first_sequence INTEGER NOT NULL,\n    "
        "first_sequence_stale INTEGER NOT NULL DEFAULT 0,\n    "
        if session
        else ""
    )
    return f"""CREATE TABLE IF NOT EXISTS {table} (
    {_COMMA_SHORT.join(key_columns[column] for column in key)},
    provider_name TEXT,
    model TEXT,
    event_count INTEGER NOT NULL,
    has_usage_count INTEGER NOT NULL,
    {session_columns}{counters},
    min_input_tokens INTEGER NOT NULL,
    max_input_tokens INTEGER NOT NULL,
    max_pricing_input_bytes INTEGER NOT NULL,
    overflow INTEGER NOT NULL,
    PRIMARY KEY ({", ".join(key)})
) WITHOUT ROWID;"""


#: Columns the usage-projection revision adds to ``cayu_usage_events`` (``_MIGRATION_ADD_COLUMNS``).
SQLITE_USAGE_EVENT_ADDED_COLUMNS: tuple[tuple[str, str, str], ...] = (
    ("cayu_usage_events", "agent_name", "TEXT"),
    ("cayu_usage_events", "environment_name", "TEXT"),
)

_SQLITE_ALL_EVENTS = (
    "SELECT "
    + ", ".join(f"x.{column} AS {column}" for column in _SQLITE_EVENT_COLUMNS)
    + " FROM cayu_usage_events AS x ORDER BY x.sequence"
)

#: Refill the stored totals from ``cayu_usage_events``, inside one write
#: transaction. The migration runs it; so can a tool that rewrites events with
#: SQL, such as the dashboard benchmark's store copy.
SQLITE_USAGE_PROJECTION_REBUILD = f"""
DELETE FROM cayu_session_usage;
DELETE FROM cayu_usage_hourly;
DELETE FROM cayu_tool_usage;
{_sqlite_identity_intern(_SQLITE_ALL_EVENTS)};
{_sqlite_add(_SQLITE_ALL_EVENTS)};
"""

SQLITE_USAGE_PROJECTION_DDL = f"""
UPDATE cayu_usage_events SET
    agent_name = (SELECT event.agent_name FROM cayu_events AS event
                  WHERE event.sequence = cayu_usage_events.sequence),
    environment_name = (SELECT event.environment_name FROM cayu_events AS event
                        WHERE event.sequence = cayu_usage_events.sequence);

DROP TRIGGER IF EXISTS cayu_usage_events_insert;
DROP TRIGGER IF EXISTS cayu_usage_events_delete;
DROP TRIGGER IF EXISTS cayu_usage_events_update;
{sqlite_usage_event_triggers(attributed=True)}
CREATE INDEX IF NOT EXISTS idx_cayu_usage_events_session_timestamp
    ON cayu_usage_events(session_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_cayu_usage_events_tool_sequence
    ON cayu_usage_events(tool_name, sequence) WHERE tool_name IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_cayu_usage_events_tool_timestamp
    ON cayu_usage_events(tool_name, timestamp) WHERE tool_name IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_cayu_usage_events_reported_cost
    ON cayu_usage_events(timestamp)
    WHERE event_type = 'model.completed' AND reported_cost = 1;

CREATE TABLE IF NOT EXISTS cayu_usage_price_identities (
    id INTEGER PRIMARY KEY,
    identity_key TEXT NOT NULL UNIQUE,
    usage_metrics_json TEXT,
    billing_identity_json TEXT
);

{_sqlite_usage_table("cayu_session_usage", _SESSION_KEY, session=True)}
CREATE INDEX IF NOT EXISTS idx_cayu_session_usage_hour
    ON cayu_session_usage(hour_start, event_type, provider_name, model, session_id);

{_sqlite_usage_table("cayu_usage_hourly", _HOURLY_KEY, session=False)}

CREATE TABLE IF NOT EXISTS cayu_tool_usage (
    tool_name TEXT NOT NULL,
    agent_name TEXT NOT NULL,
    hour_start TEXT NOT NULL,
    calls INTEGER NOT NULL,
    failures INTEGER NOT NULL,
    approved INTEGER NOT NULL,
    rejected INTEGER NOT NULL,
    last_used_at TEXT,
    PRIMARY KEY (tool_name, agent_name, hour_start)
) WITHOUT ROWID;

{SQLITE_USAGE_PROJECTION_REBUILD}

CREATE TRIGGER IF NOT EXISTS cayu_usage_projection_replace
BEFORE INSERT ON cayu_usage_events
FOR EACH ROW
WHEN NEW.event_type IN ({_ALL_COPIED_TYPES_SQL})
 AND EXISTS (SELECT 1 FROM cayu_usage_events WHERE sequence = NEW.sequence)
BEGIN
    {_sqlite_remove(_SQLITE_EXISTING)};
END;

CREATE TRIGGER IF NOT EXISTS cayu_usage_projection_insert
AFTER INSERT ON cayu_usage_events
FOR EACH ROW
WHEN NEW.event_type IN ({_ALL_COPIED_TYPES_SQL})
BEGIN
    {_sqlite_identity_intern(_sqlite_row("NEW"))};
    {_sqlite_add(_sqlite_row("NEW"))};
END;

CREATE TRIGGER IF NOT EXISTS cayu_usage_projection_delete
AFTER DELETE ON cayu_usage_events
FOR EACH ROW
WHEN OLD.event_type IN ({_ALL_COPIED_TYPES_SQL})
BEGIN
    {_sqlite_remove(_sqlite_row("OLD"))};
END;

CREATE TRIGGER IF NOT EXISTS cayu_usage_projection_update
AFTER UPDATE ON cayu_usage_events
FOR EACH ROW
WHEN OLD.event_type IN ({_ALL_COPIED_TYPES_SQL}) OR NEW.event_type IN ({_ALL_COPIED_TYPES_SQL})
BEGIN
    {_sqlite_remove(_sqlite_row("OLD"))};
    {_sqlite_identity_intern(_sqlite_row("NEW"))};
    {_sqlite_add(_sqlite_row("NEW"))};
END;
"""


# --- Postgres ---------------------------------------------------------------


def _postgres_trim() -> str:
    from cayu.budgets.aggregates import AGGREGATE_IDENTITY_TRIM_CHARACTERS

    return "(" + " || ".join(f"chr({ord(c)})" for c in AGGREGATE_IDENTITY_TRIM_CHARACTERS) + ")"


def _postgres_trimmed(expression: str) -> str:
    return expression.replace("(SELECT identity_trim FROM scope)", _postgres_trim())


def _postgres_identity_text(path: str) -> str:
    """A trimmed, nonblank payload string at ``path`` (``{a,b}``), else NULL."""

    return f"""CASE
            WHEN event.payload -> 'usage_normalization_failed' IS DISTINCT FROM 'true'::jsonb
             AND jsonb_typeof(event.payload #> '{path}') = 'string'
             AND btrim(event.payload #>> '{path}', {_postgres_trim()}) = event.payload #>> '{path}'
             AND event.payload #>> '{path}' <> ''
            THEN event.payload #>> '{path}'
        END"""


def _postgres_hosted_text(key: str) -> str:
    return f"""CASE
            WHEN event.event_type = '{HOSTED}'
             AND jsonb_typeof(event.payload -> '{key}') = 'string'
             AND btrim(event.payload ->> '{key}', {_postgres_trim()}) = event.payload ->> '{key}'
             AND event.payload ->> '{key}' <> ''
            THEN event.payload ->> '{key}'
        END"""


def _postgres_valid_counter(value: str) -> str:
    """Whether jsonb ``value`` is an integer that strict usage metrics accept."""

    text = f"({value} #>> '{{}}')"
    return f"""(jsonb_typeof({value}) = 'number'
             AND {text} ~ '^[0-9]+$'
             AND (length({text}) < 19
                  OR (length({text}) = 19 AND {text} COLLATE "C" <= '{_MAX}')))"""


def _postgres_identity_metrics(metrics: str) -> str:
    expression = metrics
    for path in PRICING_COUNTER_PATHS:
        pg_path = "'{" + ",".join(path) + "}'"
        value = f"({metrics} #> {pg_path})"
        expression = (
            f"jsonb_set({expression}, {pg_path}, CASE WHEN {_postgres_valid_counter(value)} "
            f"THEN to_jsonb(LEAST(({value} #>> '{{}}')::numeric, 1)) ELSE '\"x\"'::jsonb END, false)"
        )
    return expression


_POSTGRES_PROJECTION_TYPE_COLUMNS = (
    ("usage_eligible", "integer"),
    ("has_usage", "integer"),
    ("legacy", "integer"),
    ("provider_name", "text"),
    ("model", "text"),
    *((column, "numeric") for column in COUNTER_COLUMNS),
    *((column, "numeric") for column in METRICS_WEB_SEARCH_COLUMNS),
    ("event_input_tokens", "numeric"),
    ("pricing_input_bytes", "bigint"),
    ("usage_metrics", "jsonb"),
    ("billing_identity", "jsonb"),
)


def _postgres_projection_select() -> str:
    """Every projected value of one copy row ``event`` (payload, event_type)."""

    from cayu.storage import _postgres_aggregates as pg

    tokens = {
        column: pg._nonnegative_json_integer(path)
        for column, path in zip(_TOKEN_COLUMNS, pg._TOKEN_PATHS.values(), strict=True)
    }
    model_types = f"event.event_type IN ({_MODEL_TYPES_SQL})"
    metrics = _postgres_trimmed(pg._postgres_usage_metrics_projection())
    billing = _postgres_trimmed(pg._postgres_billing_identity_projection(("billing_identity",)))
    values = {
        "usage_eligible": f"CASE WHEN {pg._HOSTED_USAGE_ELIGIBILITY_SQL} THEN 1 ELSE 0 END",
        "has_usage": """CASE
            WHEN event.payload -> 'usage_normalization_failed' IS DISTINCT FROM 'true'::jsonb
             AND jsonb_typeof(event.payload -> 'usage_metrics') = 'object'
            THEN 1 ELSE 0 END""",
        "legacy": f"""CASE WHEN {model_types}
             AND NOT (event.payload ? 'usage_metrics')
             AND event.payload -> 'usage_normalization_failed' IS DISTINCT FROM 'true'::jsonb
            THEN 1 ELSE 0 END""",
        "provider_name": "COALESCE("
        + _postgres_identity_text("{usage_metrics,provider_name}")
        + ", "
        + _postgres_hosted_text("provider_name")
        + ")",
        "model": "COALESCE("
        + _postgres_identity_text("{usage_metrics,model}")
        + ", "
        + _postgres_hosted_text("model")
        + ")",
        **{
            column: f"CASE WHEN {model_types} THEN {expression} ELSE 0::numeric END"
            for column, expression in tokens.items()
        },
        "web_search_calls": pg._web_search_terminal_count(),
        "web_search_outcome_unknown": pg._web_search_unknown_count(),
        "metrics_web_search_calls": f"CASE WHEN event.event_type = '{HOSTED}' "
        f"THEN {pg._web_search_terminal_count()} WHEN {model_types} THEN "
        f"{pg._nonnegative_json_integer('{usage_metrics,hosted_tools,web_search_calls}')} "
        "ELSE 0::numeric END",
        "metrics_web_search_outcome_unknown": f"CASE WHEN event.event_type = '{HOSTED}' "
        f"THEN {pg._web_search_unknown_count()} WHEN {model_types} THEN "
        f"{pg._nonnegative_json_integer('{usage_metrics,hosted_tools,web_search_outcome_unknown}')} "
        "ELSE 0::numeric END",
        "event_input_tokens": pg._nonnegative_json_integer("{usage_metrics,input_tokens}"),
        "pricing_input_bytes": "0",
        "usage_metrics": f"CASE WHEN event.event_type IN ({_PRICED_TYPES_SQL}) THEN {metrics} END",
        "billing_identity": f"CASE WHEN event.event_type IN ({_PRICED_TYPES_SQL}) THEN {billing} END",
    }
    # The hosted-call projection has no tokens, so its input band is 0.
    values["event_input_tokens"] = (
        f"CASE WHEN {model_types} THEN {values['event_input_tokens']} ELSE 0::numeric END"
    )
    names = [name for name, _ in _POSTGRES_PROJECTION_TYPE_COLUMNS]
    if set(values) != set(names):
        raise RuntimeError("Postgres usage projections do not match their columns.")
    return ",\n            ".join(f"{values[name]} AS {name}" for name in names)


def _postgres_price_identity_function() -> str:
    """Reuse visible identities without waiting on another session's uncommitted row."""

    from cayu.storage import _postgres_aggregates as pg

    metrics = _postgres_trimmed(pg._postgres_usage_metrics_projection())
    billing = _postgres_trimmed(pg._postgres_billing_identity_projection(("billing_identity",)))
    return f"""
    CREATE OR REPLACE FUNCTION cayu_intern_usage_price_identity(ev cayu_usage_events)
    RETURNS bigint AS $$
    DECLARE
        projected record;
        identity_metrics jsonb;
        v_hash text;
        identity_id bigint;
    BEGIN
        IF ev.event_type NOT IN ({_PRICED_TYPES_SQL}) THEN
            RETURN 0;
        END IF;
        SELECT {metrics} AS usage_metrics, {billing} AS billing_identity
        INTO projected
        FROM (SELECT ev.payload AS payload, ev.event_type AS event_type) AS event;
        IF projected.usage_metrics IS NULL AND projected.billing_identity IS NULL THEN
            RETURN 0;
        END IF;
        identity_metrics := {_postgres_identity_metrics("projected.usage_metrics")};
        v_hash := encode(sha256(convert_to(
            COALESCE(identity_metrics::text, '~') || chr(31)
                || COALESCE(projected.billing_identity::text, '~'),
            'UTF8'
        )), 'hex');
        SELECT known.id INTO identity_id FROM cayu_usage_price_identities AS known
        WHERE known.identity_hash = v_hash
        ORDER BY known.id LIMIT 1;
        IF identity_id IS NULL THEN
            -- A unique content key would wait for an invisible writer to commit.
            -- Duplicate equivalents are safe because each event retains its binding.
            INSERT INTO cayu_usage_price_identities (
                identity_hash, usage_metrics, billing_identity
            ) VALUES (v_hash, identity_metrics, projected.billing_identity)
            RETURNING id INTO identity_id;
        END IF;
        RETURN identity_id;
    END;
    $$ LANGUAGE plpgsql
    """


def _postgres_projection_function() -> str:
    columns = ",\n        ".join(
        f"{name} {kind}" for name, kind in _POSTGRES_PROJECTION_TYPE_COLUMNS
    )
    return f"""
    CREATE OR REPLACE FUNCTION cayu_usage_event_projection(
        ev cayu_usage_events
    )
    RETURNS TABLE (
        price_identity_id bigint,
        input_band integer,
        {columns}
    ) AS $$
    #variable_conflict use_column
    DECLARE
        projected record;
    BEGIN
        SELECT
            {_postgres_projection_select()}
        INTO projected
        FROM (SELECT ev.payload AS payload, ev.event_type AS event_type) AS event;
        price_identity_id := COALESCE(ev.projection_price_identity_id, 0);
        input_band := {_band_sql("projected.event_input_tokens", postgres=True)};
        usage_eligible := projected.usage_eligible;
        has_usage := projected.has_usage;
        legacy := projected.legacy;
        provider_name := projected.provider_name;
        model := projected.model;
        {" ".join(f"{column} := projected.{column};" for column in (*_SUMMED, "event_input_tokens"))}
        pricing_input_bytes := COALESCE(octet_length(projected.usage_metrics::text), 0)
            + COALESCE(octet_length(projected.billing_identity::text), 0);
        usage_metrics := projected.usage_metrics;
        billing_identity := projected.billing_identity;
        RETURN NEXT;
    END;
    $$ LANGUAGE plpgsql
    """


def _postgres_hour(timestamp: str) -> str:
    return f"(date_trunc('hour', {timestamp} AT TIME ZONE 'UTC') AT TIME ZONE 'UTC')"


def _postgres_usage_upsert(
    table: str, key: tuple[str, ...], values: dict[str, str], *, session: bool
) -> str:
    columns = (
        *key,
        "provider_name",
        "model",
        "event_count",
        "has_usage_count",
        *(("legacy_count",) if session else ()),
        *_SUMMED,
        "min_input_tokens",
        "max_input_tokens",
        "max_pricing_input_bytes",
        *(("first_sequence",) if session else ()),
        "overflow",
    )
    row = {
        **values,
        "provider_name": "p.provider_name",
        "model": "p.model",
        "event_count": "1",
        "has_usage_count": "p.has_usage",
        "legacy_count": "p.legacy",
        **{column: f"p.{column}" for column in _SUMMED},
        "min_input_tokens": "p.event_input_tokens",
        "max_input_tokens": "p.event_input_tokens",
        "max_pricing_input_bytes": "p.pricing_input_bytes",
        "first_sequence": "ev.sequence",
        "overflow": "0",
    }
    updates = [
        f"event_count = {table}.event_count + 1",
        f"has_usage_count = {table}.has_usage_count + EXCLUDED.has_usage_count",
        *((f"legacy_count = {table}.legacy_count + EXCLUDED.legacy_count",) if session else ()),
        *(f"{column} = {table}.{column} + EXCLUDED.{column}" for column in _SUMMED),
        f"min_input_tokens = LEAST({table}.min_input_tokens, EXCLUDED.min_input_tokens)",
        f"max_input_tokens = GREATEST({table}.max_input_tokens, EXCLUDED.max_input_tokens)",
        f"max_pricing_input_bytes = GREATEST({table}.max_pricing_input_bytes, "
        "EXCLUDED.max_pricing_input_bytes)",
        *(
            (f"first_sequence = LEAST({table}.first_sequence, EXCLUDED.first_sequence)",)
            if session
            else ()
        ),
    ]
    return f"""INSERT INTO {table} ({", ".join(columns)})
            VALUES ({", ".join(row[column] for column in columns)})
            ON CONFLICT ({", ".join(key)}) DO UPDATE SET
                {_COMMA_LINE.join(updates)};"""


def _postgres_usage_decrement(
    table: str, key: tuple[str, ...], values: dict[str, str], *, session: bool
) -> str:
    match = " AND ".join(f"{table}.{column} = {values[column]}" for column in key)
    updates = [
        f"event_count = {table}.event_count - 1",
        f"has_usage_count = {table}.has_usage_count - p.has_usage",
        *((f"legacy_count = {table}.legacy_count - p.legacy",) if session else ()),
        *(f"{column} = {table}.{column} - p.{column}" for column in _SUMMED),
    ]
    return f"""UPDATE {table} SET
                {_COMMA_LINE.join(updates)}
            WHERE {match};
            DELETE FROM {table} WHERE {match} AND {table}.event_count <= 0;"""


def _postgres_tool_counted() -> str:
    hosted_counted = (
        f"ev.event_type = '{HOSTED}' "
        "AND jsonb_typeof(ev.payload -> 'tool_type') = 'string' "
        "AND ev.payload ->> 'tool_type' = 'web_search' "
        "AND jsonb_typeof(ev.payload -> 'status') = 'string' "
        f"AND ev.payload ->> 'status' IN ({_TERMINAL_SQL})"
    )
    return (
        "((ev.event_type IN ('tool.call.completed', 'tool.call.failed', "
        "'tool.call.approved', 'tool.call.approval_denied') AND ev.tool_name IS NOT NULL)"
        f" OR ({hosted_counted}))"
    )


def _postgres_tool_values() -> str:
    call = f"ev.event_type IN ('tool.call.completed', 'tool.call.failed', '{HOSTED}')"
    return f"""
        tool_counted := {_postgres_tool_counted()};
        tool_key := CASE WHEN ev.event_type = '{HOSTED}' THEN '{HOSTED_WEB_SEARCH_TOOL_NAME}'
                         ELSE ev.tool_name END;
        tool_calls := CASE WHEN {call} THEN 1 ELSE 0 END;
        tool_failures := CASE WHEN ev.event_type = 'tool.call.failed' THEN 1
                              WHEN ev.event_type = '{HOSTED}'
                               AND ev.payload ->> 'status' = 'failed' THEN 1 ELSE 0 END;
        tool_approved := CASE WHEN ev.event_type = 'tool.call.approved' THEN 1 ELSE 0 END;
        tool_rejected := CASE WHEN ev.event_type = 'tool.call.approval_denied' THEN 1 ELSE 0 END;"""


def _postgres_tool_last_call(row: str) -> str:
    """The newest call left for tool row ``row`` (alias of ``cayu_tool_usage``)."""

    hour = f"x.timestamp >= {row}.hour_start AND x.timestamp < {row}.hour_start + interval '1 hour'"
    agent = f"COALESCE(x.agent_name, '') = {row}.agent_name"
    return f"""GREATEST(
                (SELECT MAX(x.timestamp) FROM cayu_usage_events AS x
                 WHERE {row}.tool_name = '{HOSTED_WEB_SEARCH_TOOL_NAME}'
                   AND x.event_type = '{HOSTED}'
                   AND {hour} AND {agent}
                   AND jsonb_typeof(x.payload -> 'tool_type') = 'string'
                   AND x.payload ->> 'tool_type' = 'web_search'
                   AND jsonb_typeof(x.payload -> 'status') = 'string'
                   AND x.payload ->> 'status' IN ({_TERMINAL_SQL})),
                (SELECT MAX(x.timestamp) FROM cayu_usage_events AS x
                 WHERE x.tool_name = {row}.tool_name
                   AND x.event_type IN ('tool.call.completed', 'tool.call.failed')
                   AND {hour} AND {agent})
            )"""


def _postgres_pending_delta(values: dict[str, str], *, sign: str) -> str:
    """Append one event's hourly delta (``sign`` ``-`` removes it) to the pending table."""

    columns = (
        *_HOURLY_KEY,
        "provider_name",
        "model",
        "event_count",
        "has_usage_count",
        *_SUMMED,
        "min_input_tokens",
        "max_input_tokens",
        "max_pricing_input_bytes",
    )
    row = {
        **values,
        "provider_name": "p.provider_name",
        "model": "p.model",
        "event_count": f"{sign}1",
        "has_usage_count": f"{sign}p.has_usage",
        **{column: f"{sign}p.{column}" for column in _SUMMED},
        "min_input_tokens": "NULL" if sign else "p.event_input_tokens",
        "max_input_tokens": "NULL" if sign else "p.event_input_tokens",
        "max_pricing_input_bytes": "NULL" if sign else "p.pricing_input_bytes",
    }
    return f"""INSERT INTO cayu_usage_hourly_pending ({", ".join(columns)})
            VALUES ({", ".join(row[column] for column in columns)});"""


def _postgres_compact_function() -> str:
    """Fold pending hourly and tool deltas into their shared rows, one caller at a time.

    Event writers only append deltas, so the shared rows are written here alone,
    under a transaction advisory lock; a caller that finds it taken skips.
    Readers add the pending rows they see to the shared rows, so compacting
    never changes a result.
    """

    key = ", ".join(_HOURLY_KEY)
    sums = ", ".join(f"SUM({column})" for column in ("event_count", "has_usage_count", *_SUMMED))
    added = ",\n                ".join(
        f"{column} = cayu_usage_hourly.{column} + EXCLUDED.{column}"
        for column in ("event_count", "has_usage_count", *_SUMMED)
    )
    columns = (
        *_HOURLY_KEY,
        "provider_name",
        "model",
        "event_count",
        "has_usage_count",
        *_SUMMED,
        "min_input_tokens",
        "max_input_tokens",
        "max_pricing_input_bytes",
    )
    return f"""
    CREATE OR REPLACE FUNCTION cayu_compact_usage_projection()
    RETURNS void AS $$
    BEGIN
        -- A read-only store cannot fold; its readers include the pending deltas.
        IF current_setting('transaction_read_only')::boolean
           OR NOT pg_try_advisory_xact_lock({_COMPACTION_LOCK}) THEN
            RETURN;
        END IF;
        WITH moved AS (DELETE FROM cayu_usage_hourly_pending RETURNING *)
        INSERT INTO cayu_usage_hourly ({", ".join(columns)})
        SELECT {key}, MIN(provider_name), MIN(model), {sums},
               COALESCE(MIN(min_input_tokens), 0), COALESCE(MAX(max_input_tokens), 0),
               COALESCE(MAX(max_pricing_input_bytes), 0)
        FROM moved
        GROUP BY {key}
        ON CONFLICT ({key}) DO UPDATE SET
                {added},
                min_input_tokens = LEAST(cayu_usage_hourly.min_input_tokens, EXCLUDED.min_input_tokens),
                max_input_tokens = GREATEST(cayu_usage_hourly.max_input_tokens, EXCLUDED.max_input_tokens),
                max_pricing_input_bytes = GREATEST(
                    cayu_usage_hourly.max_pricing_input_bytes, EXCLUDED.max_pricing_input_bytes
                );
        DELETE FROM cayu_usage_hourly WHERE event_count <= 0;
        WITH moved AS (DELETE FROM cayu_tool_usage_pending RETURNING *)
        INSERT INTO cayu_tool_usage (
            tool_name, agent_name, hour_start, calls, failures, approved, rejected,
            last_used_at, recheck
        )
        SELECT tool_name, agent_name, hour_start, SUM(calls), SUM(failures), SUM(approved),
               SUM(rejected), MAX(last_used_at), bool_or(recheck)
        FROM moved
        GROUP BY tool_name, agent_name, hour_start
        ON CONFLICT (tool_name, agent_name, hour_start) DO UPDATE SET
            calls = cayu_tool_usage.calls + EXCLUDED.calls,
            failures = cayu_tool_usage.failures + EXCLUDED.failures,
            approved = cayu_tool_usage.approved + EXCLUDED.approved,
            rejected = cayu_tool_usage.rejected + EXCLUDED.rejected,
            last_used_at = GREATEST(cayu_tool_usage.last_used_at, EXCLUDED.last_used_at),
            recheck = cayu_tool_usage.recheck OR EXCLUDED.recheck;
        DELETE FROM cayu_tool_usage
        WHERE calls <= 0 AND failures <= 0 AND approved <= 0 AND rejected <= 0;
        UPDATE cayu_tool_usage AS row SET recheck = false, last_used_at = {_postgres_tool_last_call("row")}
        WHERE row.recheck;
    END;
    $$ LANGUAGE plpgsql
    """


def _postgres_apply_function() -> str:
    session_values = {
        "session_id": "ev.session_id",
        "hour_start": "hour",
        "event_type": "ev.event_type",
        "price_identity_id": "p.price_identity_id",
        "input_band": "p.input_band",
    }
    hourly_values = {
        **session_values,
        "agent_name": "agent",
        "environment_name": "environment",
    }
    session_match = " AND ".join(
        f"cayu_session_usage.{column} = {value}" for column, value in session_values.items()
    )
    return f"""
    CREATE OR REPLACE FUNCTION cayu_apply_usage_projection(
        ev cayu_usage_events, direction integer
    )
    RETURNS void AS $$
    DECLARE
        p record;
        hour timestamptz := {_postgres_hour("ev.timestamp")};
        agent text := COALESCE(ev.agent_name, '');
        environment text := COALESCE(ev.environment_name, '');
        tool_counted boolean;
        tool_key text;
        tool_calls integer;
        tool_failures integer;
        tool_approved integer;
        tool_rejected integer;
    BEGIN
        IF ev.event_type IN ({_USAGE_TYPES_SQL}) THEN
            SELECT * INTO p FROM cayu_usage_event_projection(ev);
            IF p.usage_eligible = 1 AND direction > 0 THEN
                {_postgres_usage_upsert("cayu_session_usage", _SESSION_KEY, session_values, session=True)}
                {_postgres_pending_delta(hourly_values, sign="")}
            ELSIF p.usage_eligible = 1 THEN
                {_postgres_usage_decrement("cayu_session_usage", _SESSION_KEY, session_values, session=True)}
                -- See _sqlite_session_first_stale: flagged, not recomputed.
                UPDATE cayu_session_usage SET first_sequence_stale = 1
                WHERE {session_match} AND ev.sequence <= cayu_session_usage.first_sequence;
                {_postgres_pending_delta(hourly_values, sign="-")}
            END IF;
        END IF;
        {_postgres_tool_values()}
        IF tool_counted THEN
            -- Shared rows are written only by cayu_compact_usage_projection, so
            -- concurrent event writers never wait on or deadlock over them.
            INSERT INTO cayu_tool_usage_pending (
                tool_name, agent_name, hour_start, calls, failures, approved, rejected,
                last_used_at, recheck
            )
            VALUES (
                tool_key, agent, hour, direction * tool_calls, direction * tool_failures,
                direction * tool_approved, direction * tool_rejected,
                CASE WHEN direction > 0 AND tool_calls > 0 THEN ev.timestamp END,
                direction < 0 AND tool_calls > 0
            );
        END IF;
    END;
    $$ LANGUAGE plpgsql
    """


def _postgres_usage_table(table: str, key: tuple[str, ...], *, session: bool) -> str:
    key_columns = {
        "session_id": 'session_id TEXT COLLATE "C" NOT NULL',
        "hour_start": "hour_start TIMESTAMPTZ NOT NULL",
        "agent_name": 'agent_name TEXT COLLATE "C" NOT NULL',
        "environment_name": 'environment_name TEXT COLLATE "C" NOT NULL',
        "event_type": "event_type TEXT NOT NULL",
        "price_identity_id": "price_identity_id BIGINT NOT NULL",
        "input_band": "input_band INTEGER NOT NULL",
    }
    counters = _COMMA_LINE.join(f"{column} NUMERIC NOT NULL" for column in _SUMMED)
    session_columns = (
        "legacy_count BIGINT NOT NULL,\n        first_sequence BIGINT NOT NULL,\n        "
        "first_sequence_stale INTEGER NOT NULL DEFAULT 0,\n        "
        if session
        else ""
    )
    return f"""CREATE TABLE IF NOT EXISTS {table} (
        {_COMMA_LINE.join(key_columns[column] for column in key)},
        provider_name TEXT COLLATE "C",
        model TEXT COLLATE "C",
        event_count BIGINT NOT NULL,
        has_usage_count BIGINT NOT NULL,
        {session_columns}{counters},
        min_input_tokens NUMERIC NOT NULL,
        max_input_tokens NUMERIC NOT NULL,
        max_pricing_input_bytes BIGINT NOT NULL,
        overflow INTEGER NOT NULL DEFAULT 0,
        PRIMARY KEY ({", ".join(key)})
    )"""


def _postgres_pending_table() -> str:
    """Hourly deltas event writers append; ``cayu_compact_usage_projection`` folds them."""

    key_columns = {
        "hour_start": "hour_start TIMESTAMPTZ NOT NULL",
        "agent_name": 'agent_name TEXT COLLATE "C" NOT NULL',
        "environment_name": 'environment_name TEXT COLLATE "C" NOT NULL',
        "event_type": "event_type TEXT NOT NULL",
        "price_identity_id": "price_identity_id BIGINT NOT NULL",
        "input_band": "input_band INTEGER NOT NULL",
    }
    counters = _COMMA_LINE.join(f"{column} NUMERIC NOT NULL" for column in _SUMMED)
    return f"""CREATE TABLE IF NOT EXISTS cayu_usage_hourly_pending (
        id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
        {_COMMA_LINE.join(key_columns[column] for column in _HOURLY_KEY)},
        provider_name TEXT COLLATE "C",
        model TEXT COLLATE "C",
        event_count BIGINT NOT NULL,
        has_usage_count BIGINT NOT NULL,
        {counters},
        min_input_tokens NUMERIC,
        max_input_tokens NUMERIC,
        max_pricing_input_bytes BIGINT,
        overflow INTEGER NOT NULL DEFAULT 0
    )"""


def _postgres_usage_backfill(table: str, key: tuple[str, ...], *, session: bool) -> str:
    group = {
        "session_id": "ev.session_id",
        "hour_start": _postgres_hour("ev.timestamp"),
        "agent_name": "COALESCE(ev.agent_name, '')",
        "environment_name": "COALESCE(ev.environment_name, '')",
        "event_type": "ev.event_type",
        "price_identity_id": "p.price_identity_id",
        "input_band": "p.input_band",
    }
    columns = (
        *key,
        "provider_name",
        "model",
        "event_count",
        "has_usage_count",
        *(("legacy_count",) if session else ()),
        *_SUMMED,
        "min_input_tokens",
        "max_input_tokens",
        "max_pricing_input_bytes",
        *(("first_sequence",) if session else ()),
    )
    aggregates = (
        *(group[column] for column in key),
        "MIN(p.provider_name)",
        "MIN(p.model)",
        "COUNT(*)",
        "SUM(p.has_usage)",
        *(("SUM(p.legacy)",) if session else ()),
        *(f"SUM(p.{column})" for column in _SUMMED),
        "MIN(p.event_input_tokens)",
        "MAX(p.event_input_tokens)",
        "MAX(p.pricing_input_bytes)",
        *(("MIN(ev.sequence)",) if session else ()),
    )
    return f"""
    INSERT INTO {table} ({", ".join(columns)})
    SELECT {", ".join(aggregates)}
    FROM cayu_usage_events AS ev
    CROSS JOIN LATERAL cayu_usage_event_projection(ev) AS p
    WHERE ev.event_type IN ({_USAGE_TYPES_SQL}) AND p.usage_eligible = 1
    GROUP BY {", ".join(group[column] for column in key)}
    """


def _postgres_tool_backfill() -> str:
    call = f"ev.event_type IN ('tool.call.completed', 'tool.call.failed', '{HOSTED}')"
    return f"""
    INSERT INTO cayu_tool_usage (
        tool_name, agent_name, hour_start, calls, failures, approved, rejected, last_used_at
    )
    SELECT
        CASE WHEN ev.event_type = '{HOSTED}' THEN '{HOSTED_WEB_SEARCH_TOOL_NAME}'
             ELSE ev.tool_name END,
        COALESCE(ev.agent_name, ''),
        {_postgres_hour("ev.timestamp")},
        SUM(CASE WHEN {call} THEN 1 ELSE 0 END),
        SUM(CASE WHEN ev.event_type = 'tool.call.failed' THEN 1
                 WHEN ev.event_type = '{HOSTED}' AND ev.payload ->> 'status' = 'failed' THEN 1
                 ELSE 0 END),
        SUM(CASE WHEN ev.event_type = 'tool.call.approved' THEN 1 ELSE 0 END),
        SUM(CASE WHEN ev.event_type = 'tool.call.approval_denied' THEN 1 ELSE 0 END),
        MAX(CASE WHEN {call} THEN ev.timestamp END)
    FROM cayu_usage_events AS ev
    WHERE {_postgres_tool_counted()}
    GROUP BY 1, 2, 3
    """


#: Refill the stored totals from ``cayu_usage_events`` (see the SQLite twin).
POSTGRES_USAGE_PROJECTION_REBUILD: tuple[str, ...] = (
    # Binding-only updates do not run the source-column projection triggers.
    "UPDATE cayu_usage_events AS event SET projection_price_identity_id = "
    "cayu_intern_usage_price_identity(event)",
    "TRUNCATE cayu_session_usage, cayu_usage_hourly, cayu_usage_hourly_pending, "
    "cayu_tool_usage, cayu_tool_usage_pending",
    _postgres_usage_backfill("cayu_session_usage", _SESSION_KEY, session=True),
    _postgres_usage_backfill("cayu_usage_hourly", _HOURLY_KEY, session=False),
    _postgres_tool_backfill(),
)

POSTGRES_USAGE_PROJECTION_DDL: tuple[str, ...] = (
    # Event writers wait until the copy and the projections agree.
    "LOCK TABLE cayu_events, cayu_usage_events IN SHARE ROW EXCLUSIVE MODE",
    "ALTER TABLE cayu_usage_events ADD COLUMN IF NOT EXISTS agent_name TEXT",
    "ALTER TABLE cayu_usage_events ADD COLUMN IF NOT EXISTS environment_name TEXT",
    "ALTER TABLE cayu_usage_events ADD COLUMN IF NOT EXISTS projection_price_identity_id BIGINT",
    """
    UPDATE cayu_usage_events AS copy SET
        agent_name = event.agent_name,
        environment_name = event.environment_name
    FROM cayu_events AS event
    WHERE event.sequence = copy.sequence
    """,
    postgres_usage_event_function(attributed=True),
    "DROP TRIGGER IF EXISTS cayu_usage_events_update ON cayu_events",
    postgres_usage_event_update_trigger(attributed=True),
    """CREATE INDEX IF NOT EXISTS idx_cayu_usage_events_session_timestamp
        ON cayu_usage_events(session_id, timestamp)""",
    """CREATE INDEX IF NOT EXISTS idx_cayu_usage_events_tool_sequence
        ON cayu_usage_events(tool_name, sequence) WHERE tool_name IS NOT NULL""",
    """CREATE INDEX IF NOT EXISTS idx_cayu_usage_events_tool_timestamp
        ON cayu_usage_events(tool_name, timestamp) WHERE tool_name IS NOT NULL""",
    """CREATE INDEX IF NOT EXISTS idx_cayu_usage_events_reported_cost
        ON cayu_usage_events(timestamp)
        WHERE event_type = 'model.completed' AND payload #> '{usage,cost_status}' IS NOT NULL""",
    """CREATE TABLE IF NOT EXISTS cayu_usage_price_identities (
        id BIGINT GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
        identity_hash TEXT COLLATE "C" NOT NULL,
        usage_metrics JSONB,
        billing_identity JSONB
    )""",
    "ALTER TABLE cayu_usage_price_identities "
    "DROP CONSTRAINT IF EXISTS cayu_usage_price_identities_hash_key",
    "CREATE INDEX IF NOT EXISTS idx_cayu_usage_price_identities_hash "
    "ON cayu_usage_price_identities(identity_hash, id)",
    _postgres_usage_table("cayu_session_usage", _SESSION_KEY, session=True),
    """CREATE INDEX IF NOT EXISTS idx_cayu_session_usage_hour
        ON cayu_session_usage(hour_start, event_type, provider_name, model, session_id)""",
    _postgres_usage_table("cayu_usage_hourly", _HOURLY_KEY, session=False),
    _postgres_pending_table(),
    """CREATE TABLE IF NOT EXISTS cayu_tool_usage (
        tool_name TEXT COLLATE "C" NOT NULL,
        agent_name TEXT COLLATE "C" NOT NULL,
        hour_start TIMESTAMPTZ NOT NULL,
        calls BIGINT NOT NULL,
        failures BIGINT NOT NULL,
        approved BIGINT NOT NULL,
        rejected BIGINT NOT NULL,
        last_used_at TIMESTAMPTZ,
        recheck BOOLEAN NOT NULL DEFAULT false,
        PRIMARY KEY (tool_name, agent_name, hour_start)
    )""",
    """CREATE TABLE IF NOT EXISTS cayu_tool_usage_pending (
        id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
        tool_name TEXT COLLATE "C" NOT NULL,
        agent_name TEXT COLLATE "C" NOT NULL,
        hour_start TIMESTAMPTZ NOT NULL,
        calls BIGINT NOT NULL,
        failures BIGINT NOT NULL,
        approved BIGINT NOT NULL,
        rejected BIGINT NOT NULL,
        last_used_at TIMESTAMPTZ,
        recheck BOOLEAN NOT NULL
    )""",
    _postgres_price_identity_function(),
    _postgres_projection_function(),
    _postgres_apply_function(),
    _postgres_compact_function(),
    """
    CREATE OR REPLACE FUNCTION cayu_bind_usage_price_identity()
    RETURNS TRIGGER AS $$
    BEGIN
        NEW.projection_price_identity_id := cayu_intern_usage_price_identity(NEW);
        RETURN NEW;
    END;
    $$ LANGUAGE plpgsql
    """,
    "DROP TRIGGER IF EXISTS cayu_usage_price_identity ON cayu_usage_events",
    """
    CREATE TRIGGER cayu_usage_price_identity
    BEFORE INSERT OR UPDATE OF sequence, session_id, event_id, event_type, timestamp,
        tool_name, payload, agent_name, environment_name ON cayu_usage_events
    FOR EACH ROW EXECUTE FUNCTION cayu_bind_usage_price_identity()
    """,
    """
    CREATE OR REPLACE FUNCTION cayu_track_usage_projection()
    RETURNS TRIGGER AS $$
    BEGIN
        IF TG_OP IN ('UPDATE', 'DELETE') THEN
            PERFORM cayu_apply_usage_projection(OLD, -1);
        END IF;
        IF TG_OP IN ('INSERT', 'UPDATE') THEN
            PERFORM cayu_apply_usage_projection(NEW, 1);
        END IF;
        RETURN NULL;
    END;
    $$ LANGUAGE plpgsql
    """,
    "DROP TRIGGER IF EXISTS cayu_usage_projection ON cayu_usage_events",
    """
    CREATE TRIGGER cayu_usage_projection
    AFTER INSERT OR UPDATE OF sequence, session_id, event_id, event_type, timestamp,
        tool_name, payload, agent_name, environment_name OR DELETE ON cayu_usage_events
    FOR EACH ROW EXECUTE FUNCTION cayu_track_usage_projection()
    """,
    *POSTGRES_USAGE_PROJECTION_REBUILD,
)

"""A narrow, trigger-maintained copy of the events that usage and tool aggregates read.

Aggregates over ``cayu_events`` would read each row's whole payload, so an
aggregate over a month of model steps and tool calls reads gigabytes of unrelated
content on a large store.
``cayu_usage_events`` keeps one row per aggregated event with the indexed columns
the aggregates filter on and a payload reduced to the top-level members they
read. Triggers on ``cayu_events`` keep it in step with every insert, update and
delete, including those of older binaries, and the migration copies the events
already retained.

PostgreSQL aggregate statements read the copy through the same column names and
payload expressions they used against ``cayu_events``. SQLite reparses JSON text
for every member it reads, so its triggers also store each rollup projection of
the reduced payload, and its statements read those columns.
"""

from __future__ import annotations

from cayu.events import EventType

#: Every event type an aggregate reads from ``cayu_usage_events``.
USAGE_EVENT_TYPES: tuple[str, ...] = tuple(
    str(event_type)
    for event_type in (
        EventType.MODEL_COMPLETED,
        EventType.MODEL_AUXILIARY_ATTEMPT_SETTLED,
        EventType.MODEL_HOSTED_TOOL_CALL,
        EventType.TOOL_CALL_STARTED,
        EventType.TOOL_CALL_COMPLETED,
        EventType.TOOL_CALL_FAILED,
        EventType.TOOL_CALL_APPROVED,
        EventType.TOOL_CALL_APPROVAL_DENIED,
    )
)

#: The event types whose payload the aggregates read. Tool lifecycle events are
#: tallied from their columns, so their copy keeps an empty payload.
PAYLOAD_EVENT_TYPES: tuple[str, ...] = tuple(
    str(event_type)
    for event_type in (
        EventType.MODEL_COMPLETED,
        EventType.MODEL_AUXILIARY_ATTEMPT_SETTLED,
        EventType.MODEL_HOSTED_TOOL_CALL,
    )
)

#: The top-level payload members usage rollups, pricing inputs, reported costs
#: and hosted tool tallies read. Each is copied whole when present.
PAYLOAD_KEYS: tuple[str, ...] = (
    "billing_identity",
    "id",
    "model",
    "provider_name",
    "status",
    "tool_type",
    "usage",
    "usage_metrics",
    "usage_normalization_failed",
)


def _sql_list(values: tuple[str, ...]) -> str:
    return ", ".join("'" + value.replace("'", "''") + "'" for value in values)


_USAGE_TYPES_SQL = _sql_list(USAGE_EVENT_TYPES)
_PAYLOAD_TYPES_SQL = _sql_list(PAYLOAD_EVENT_TYPES)


def _sqlite_payload_member(source: str, key: str) -> str:
    path = f"'$.{key}'"
    kind = f"json_type({source}, {path})"
    # json_quote leaves objects and arrays from json_extract unchanged, quotes
    # text and renders numbers and null; booleans keep their JSON spelling.
    return f"""CASE
                WHEN {kind} IS NULL THEN ''
                WHEN {kind} IN ('true', 'false') THEN '"{key}":' || {kind} || ','
                ELSE '"{key}":' || json_quote(json_extract({source}, {path})) || ','
            END"""


def sqlite_usage_payload(source: str, event_type: str) -> str:
    """SQL reducing ``source`` payload JSON text to the members aggregates read."""

    members = "\n            || ".join(_sqlite_payload_member(source, key) for key in PAYLOAD_KEYS)
    return f"""CASE
        WHEN {event_type} IN ({_PAYLOAD_TYPES_SQL}) THEN json(
            '{{' || rtrim(
            {members},
            ','
            ) || '}}'
        )
        ELSE '{{}}'
    END"""


def _sqlite_identity_trim() -> str:
    from cayu.budgets.aggregates import AGGREGATE_IDENTITY_TRIM_CHARACTERS

    return "char(" + ", ".join(str(ord(c)) for c in AGGREGATE_IDENTITY_TRIM_CHARACTERS) + ")"


def _sqlite_projections() -> tuple[tuple[str, str], ...]:
    from cayu.storage._sqlite_aggregates import usage_event_projection_sql

    return usage_event_projection_sql(identity_trim=_sqlite_identity_trim())


_SQLITE_PROJECTION_COLUMNS = (
    ("usage_eligible", "INTEGER NOT NULL"),
    ("has_usage", "INTEGER NOT NULL"),
    ("provider_name", "TEXT"),
    ("model", "TEXT"),
    ("input_tokens", "INTEGER NOT NULL"),
    ("output_tokens", "INTEGER NOT NULL"),
    ("total_tokens", "INTEGER NOT NULL"),
    ("reasoning_output_tokens", "INTEGER NOT NULL"),
    ("cache_read_tokens", "INTEGER NOT NULL"),
    ("cache_write_tokens", "INTEGER NOT NULL"),
    ("cache_write_5m_tokens", "INTEGER NOT NULL"),
    ("cache_write_1h_tokens", "INTEGER NOT NULL"),
    ("cache_write_unknown_ttl_tokens", "INTEGER NOT NULL"),
    ("cached_input_tokens", "INTEGER NOT NULL"),
    ("uncached_input_tokens", "INTEGER NOT NULL"),
    ("web_search_calls", "INTEGER NOT NULL"),
    ("web_search_outcome_unknown", "INTEGER NOT NULL"),
    ("usage_metrics_json", "TEXT"),
    ("billing_identity_json", "TEXT"),
    ("reported_cost", "INTEGER NOT NULL"),
)


def _sqlite_copy(row: str, *, attributed: bool = False) -> str:
    """Copy ``row`` of ``cayu_events``, storing each rollup projection of its payload.

    SQLite reparses JSON text for every member it reads, so the projections run
    once here, over the reduced payload, instead of in every rollup statement.
    ``attributed`` copies the event's agent and environment names too, which the
    usage-projection copy keeps for the usage projections (see
    ``_usage_projection_schema``); revision 149's copy has no such columns.
    """

    projections = _sqlite_projections()
    if tuple(name for name, _ in projections) != tuple(
        name for name, _ in _SQLITE_PROJECTION_COLUMNS
    ):
        raise RuntimeError("SQLite usage-event projections do not match the copy's columns.")
    names = ", ".join(name for name, _ in projections)
    expressions = ",\n            ".join(expression for _, expression in projections)
    attribution = ", agent_name, environment_name" if attributed else ""
    attribution_values = ", event.agent_name, event.environment_name" if attributed else ""
    attribution_source = (
        f"{row}.agent_name AS agent_name, {row}.environment_name AS environment_name,"
        if attributed
        else ""
    )
    return f"""INSERT OR REPLACE INTO cayu_usage_events (
            sequence, session_id, event_id, event_type, timestamp, tool_name, payload_json,
            {names}{attribution}
        )
        SELECT
            event.sequence, event.session_id, event.event_id, event.event_type,
            event.timestamp, event.tool_name, event.payload_json,
            {expressions}{attribution_values}
        FROM (
            SELECT
                {row}.sequence AS sequence, {row}.session_id AS session_id,
                {row}.event_id AS event_id, {row}.event_type AS event_type,
                {row}.timestamp AS timestamp, {row}.tool_name AS tool_name,
                {attribution_source}
                {sqlite_usage_payload(f"{row}.payload_json", f"{row}.event_type")}
                    AS payload_json"""


_SQLITE_COLUMN_DDL = ",\n    ".join(f"{name} {kind}" for name, kind in _SQLITE_PROJECTION_COLUMNS)


def sqlite_usage_event_triggers(*, attributed: bool) -> str:
    """The ``cayu_events`` triggers that keep ``cayu_usage_events`` in step."""

    update_columns = (
        "sequence, session_id, event_id, event_type, timestamp, tool_name, payload_json"
        + (", agent_name, environment_name" if attributed else "")
    )
    return f"""
CREATE TRIGGER IF NOT EXISTS cayu_usage_events_insert
AFTER INSERT ON cayu_events
FOR EACH ROW
WHEN NEW.event_type IN ({_USAGE_TYPES_SQL})
BEGIN
    {_sqlite_copy("NEW", attributed=attributed)}
        ) AS event;
END;

CREATE TRIGGER IF NOT EXISTS cayu_usage_events_delete
AFTER DELETE ON cayu_events
FOR EACH ROW
WHEN OLD.event_type IN ({_USAGE_TYPES_SQL})
BEGIN
    DELETE FROM cayu_usage_events WHERE sequence = OLD.sequence;
END;

CREATE TRIGGER IF NOT EXISTS cayu_usage_events_update
AFTER UPDATE OF {update_columns}
ON cayu_events
FOR EACH ROW
WHEN OLD.event_type IN ({_USAGE_TYPES_SQL}) OR NEW.event_type IN ({_USAGE_TYPES_SQL})
BEGIN
    DELETE FROM cayu_usage_events WHERE sequence = OLD.sequence;
    {_sqlite_copy("NEW", attributed=attributed)}
            WHERE NEW.event_type IN ({_USAGE_TYPES_SQL})
        ) AS event;
END;
"""


SQLITE_USAGE_EVENTS_DDL = f"""
CREATE TABLE IF NOT EXISTS cayu_usage_events (
    sequence INTEGER PRIMARY KEY,
    session_id TEXT NOT NULL,
    event_id TEXT NOT NULL,
    event_type TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    tool_name TEXT,
    payload_json TEXT NOT NULL,
    {_SQLITE_COLUMN_DDL}
);

CREATE INDEX IF NOT EXISTS idx_cayu_usage_events_type_timestamp
    ON cayu_usage_events(event_type, timestamp);
{sqlite_usage_event_triggers(attributed=False)}
{_sqlite_copy("source")}
            FROM cayu_events AS source
            WHERE source.event_type IN ({_USAGE_TYPES_SQL})
        ) AS event;
"""


def postgres_usage_payload(source: str, event_type: str) -> str:
    """SQL reducing ``source`` payload JSONB to the members aggregates read."""

    return f"""CASE
        WHEN {event_type} IN ({_PAYLOAD_TYPES_SQL}) THEN COALESCE(
            (
                SELECT jsonb_object_agg(member.key, member.value)
                FROM jsonb_each({source}) AS member
                WHERE member.key IN ({_sql_list(PAYLOAD_KEYS)})
            ),
            '{{}}'::jsonb
        )
        ELSE '{{}}'::jsonb
    END"""


def postgres_usage_event_function(*, attributed: bool) -> str:
    """The trigger function that keeps ``cayu_usage_events`` in step."""

    columns = "sequence, session_id, event_id, event_type, timestamp, tool_name, payload"
    values = (
        "NEW.sequence, NEW.session_id, NEW.event_id, NEW.event_type,\n"
        "                NEW.timestamp, NEW.tool_name,\n"
        f"                {postgres_usage_payload('NEW.payload', 'NEW.event_type')}"
    )
    updates = """session_id = EXCLUDED.session_id,
                event_id = EXCLUDED.event_id,
                event_type = EXCLUDED.event_type,
                timestamp = EXCLUDED.timestamp,
                tool_name = EXCLUDED.tool_name,
                payload = EXCLUDED.payload"""
    if attributed:
        columns += ", agent_name, environment_name"
        values += ",\n                NEW.agent_name, NEW.environment_name"
        updates += """,
                agent_name = EXCLUDED.agent_name,
                environment_name = EXCLUDED.environment_name"""
    return f"""
    CREATE OR REPLACE FUNCTION cayu_track_usage_event()
    RETURNS TRIGGER AS $$
    BEGIN
        IF TG_OP IN ('UPDATE', 'DELETE') THEN
            DELETE FROM cayu_usage_events WHERE sequence = OLD.sequence;
        END IF;
        IF TG_OP IN ('INSERT', 'UPDATE') AND NEW.event_type IN ({_USAGE_TYPES_SQL}) THEN
            INSERT INTO cayu_usage_events (
                {columns}
            )
            VALUES (
                {values}
            )
            ON CONFLICT (sequence) DO UPDATE SET
                {updates};
        END IF;
        RETURN NULL;
    END;
    $$ LANGUAGE plpgsql
    """


def postgres_usage_event_update_trigger(*, attributed: bool) -> str:
    columns = "sequence, session_id, event_id, event_type, timestamp, tool_name, payload" + (
        ", agent_name, environment_name" if attributed else ""
    )
    return f"""
    CREATE TRIGGER cayu_usage_events_update
    AFTER UPDATE OF {columns}
    ON cayu_events
    FOR EACH ROW
    WHEN (OLD.event_type IN ({_USAGE_TYPES_SQL}) OR NEW.event_type IN ({_USAGE_TYPES_SQL}))
    EXECUTE FUNCTION cayu_track_usage_event()
    """


POSTGRES_USAGE_EVENTS_DDL: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS cayu_usage_events (
        sequence BIGINT PRIMARY KEY,
        session_id TEXT NOT NULL,
        event_id TEXT NOT NULL,
        event_type TEXT NOT NULL,
        timestamp TIMESTAMPTZ NOT NULL,
        tool_name TEXT,
        payload JSONB NOT NULL
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_cayu_usage_events_type_timestamp
        ON cayu_usage_events(event_type, timestamp)
    """,
    postgres_usage_event_function(attributed=False),
    "DROP TRIGGER IF EXISTS cayu_usage_events_insert ON cayu_events",
    f"""
    CREATE TRIGGER cayu_usage_events_insert
    AFTER INSERT ON cayu_events
    FOR EACH ROW
    WHEN (NEW.event_type IN ({_USAGE_TYPES_SQL}))
    EXECUTE FUNCTION cayu_track_usage_event()
    """,
    "DROP TRIGGER IF EXISTS cayu_usage_events_delete ON cayu_events",
    f"""
    CREATE TRIGGER cayu_usage_events_delete
    AFTER DELETE ON cayu_events
    FOR EACH ROW
    WHEN (OLD.event_type IN ({_USAGE_TYPES_SQL}))
    EXECUTE FUNCTION cayu_track_usage_event()
    """,
    "DROP TRIGGER IF EXISTS cayu_usage_events_update ON cayu_events",
    postgres_usage_event_update_trigger(attributed=False),
    f"""
    INSERT INTO cayu_usage_events (
        sequence, session_id, event_id, event_type, timestamp, tool_name, payload
    )
    SELECT
        event.sequence, event.session_id, event.event_id, event.event_type,
        event.timestamp, event.tool_name,
        {postgres_usage_payload("event.payload", "event.event_type")}
    FROM cayu_events AS event
    WHERE event.event_type IN ({_USAGE_TYPES_SQL})
    ON CONFLICT (sequence) DO NOTHING
    """,
)

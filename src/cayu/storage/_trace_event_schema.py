"""A narrow, trigger-maintained copy of what the session trace reads from each event.

The trace waterfall places every event of a session under its turn, model step,
tool round, tool call, command or wait, from a few identity members of the
payload. Reading those from ``cayu_events`` means reading and decoding every
whole payload of the session, tool results included, on every trace read.

``cayu_trace_events`` keeps one row per event, keyed by its sequence, with the
payload's size and the payload reduced to the members
:mod:`cayu.sessions.trace_events` names. Readers join it to the event's own
columns on ``cayu_events``, which sit before the payload in the row, so a trace
or an event-head page reads no event payload. Triggers on ``cayu_events`` keep
the copy in step with every insert, update and delete, including those of older
binaries, and the migration copies the events already retained. Postgres reads
the payload from the ``event`` document, as its event reads decode it.

The members are fixed here rather than read from the event schema, because a
revision's DDL must not change after it ships. A test checks that the schema
still asks for no member outside them.
"""

from __future__ import annotations

from cayu._trace_payload import (
    TRACE_PAYLOAD_KEYS,
    TRACE_RESULT_STRUCTURED_EVENT_TYPES,
    TRACE_RESULT_STRUCTURED_KEYS,
    TRACE_WHOLE_MEMBERS,
)

#: The storage revision that adds ``cayu_trace_events``. Session stores require it.
TRACE_EVENTS_REVISION = 154


def _sql_text(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _sql_list(values: tuple[str, ...]) -> str:
    return ", ".join(_sql_text(value) for value in values)


def _sqlite_member(source: str, path: str, key: str) -> str:
    """``"key":value,`` for the JSON value at ``path`` of ``source``, or ``''`` when absent."""

    kind = f"json_type({source}, '{path}')"
    # json_quote keeps objects and arrays from json_extract as JSON, quotes text
    # and renders numbers and null; booleans keep their JSON spelling.
    return (
        f"CASE WHEN {kind} IS NULL THEN '' "
        f"WHEN {kind} IN ('true', 'false') THEN '\"{key}\":' || {kind} || ',' "
        f"ELSE '\"{key}\":' || json_quote(json_extract({source}, '{path}')) || ',' END"
    )


def sqlite_trace_payload(source: str, event_type: str) -> str:
    """SQL reducing ``source`` payload JSON text to the members the trace reads."""

    members = [_sqlite_member(source, f"$.{key}", key) for key in TRACE_PAYLOAD_KEYS]
    for owner, keys in TRACE_WHOLE_MEMBERS.items():
        for key in keys:
            members.append(
                f"CASE WHEN {event_type} = {_sql_text(owner)} "
                f"THEN {_sqlite_member(source, f'$.{key}', key)} ELSE '' END"
            )
    structured = " || ".join(
        _sqlite_member(source, f"$.result.structured.{key}", key)
        for key in TRACE_RESULT_STRUCTURED_KEYS
    )
    members.append(
        f"CASE WHEN {event_type} IN ({_sql_list(TRACE_RESULT_STRUCTURED_EVENT_TYPES)}) "
        f"AND json_type({source}, '$.result.structured') = 'object' "
        f"THEN '\"result\":{{\"structured\":{{' || rtrim({structured}, ',') || '}}}},' "
        "ELSE '' END"
    )
    joined = "\n            || ".join(members)
    # A payload that is not a JSON object keeps an empty copy: the copy must
    # never refuse a write the event table accepts.
    return (
        f"CASE WHEN json_valid({source}) AND json_type({source}) = 'object' "
        f"THEN json('{{' || rtrim(\n            {joined},\n            ','\n        ) || '}}') "
        "ELSE '{}' END"
    )


def _sqlite_copy(row: str) -> str:
    return f"""INSERT OR REPLACE INTO cayu_trace_events (
            sequence, session_id, payload_bytes, payload_json
        )
        SELECT
            {row}.sequence, {row}.session_id,
            length(CAST({row}.payload_json AS BLOB)),
            {sqlite_trace_payload(f"{row}.payload_json", f"{row}.event_type")}"""


SQLITE_TRACE_EVENTS_DDL = f"""
CREATE TABLE IF NOT EXISTS cayu_trace_events (
    sequence INTEGER PRIMARY KEY,
    session_id TEXT NOT NULL,
    payload_bytes INTEGER NOT NULL,
    payload_json TEXT NOT NULL
);

CREATE TRIGGER IF NOT EXISTS cayu_trace_events_insert
AFTER INSERT ON cayu_events
FOR EACH ROW
BEGIN
    {_sqlite_copy("NEW")};
END;

CREATE TRIGGER IF NOT EXISTS cayu_trace_events_delete
AFTER DELETE ON cayu_events
FOR EACH ROW
BEGIN
    DELETE FROM cayu_trace_events WHERE sequence = OLD.sequence;
END;

CREATE TRIGGER IF NOT EXISTS cayu_trace_events_update
AFTER UPDATE OF sequence, session_id, event_type, payload_json ON cayu_events
FOR EACH ROW
BEGIN
    DELETE FROM cayu_trace_events WHERE sequence = OLD.sequence;
    {_sqlite_copy("NEW")};
END;

{_sqlite_copy("source")}
    FROM cayu_events AS source;
"""


def postgres_trace_payload(source: str, event_type: str) -> str:
    """SQL reducing ``source`` payload JSONB to the members the trace reads."""

    whole = " OR ".join(
        f"(member.key IN ({_sql_list(keys)}) AND {event_type} = {_sql_text(owner)})"
        for owner, keys in TRACE_WHOLE_MEMBERS.items()
    )
    return f"""CASE WHEN jsonb_typeof({source}) IS DISTINCT FROM 'object' THEN '{{}}'::jsonb ELSE (
        COALESCE(
            (
                SELECT jsonb_object_agg(member.key, member.value)
                FROM jsonb_each({source}) AS member
                WHERE member.key IN ({_sql_list(TRACE_PAYLOAD_KEYS)}) OR {whole}
            ),
            '{{}}'::jsonb
        )
        || CASE
            WHEN {event_type} IN ({_sql_list(TRACE_RESULT_STRUCTURED_EVENT_TYPES)})
                AND jsonb_typeof({source} -> 'result' -> 'structured') = 'object'
            THEN jsonb_build_object(
                'result',
                jsonb_build_object(
                    'structured',
                    COALESCE(
                        (
                            SELECT jsonb_object_agg(member.key, member.value)
                            FROM jsonb_each({source} -> 'result' -> 'structured') AS member
                            WHERE member.key IN ({_sql_list(TRACE_RESULT_STRUCTURED_KEYS)})
                        ),
                        '{{}}'::jsonb
                    )
                )
            )
            ELSE '{{}}'::jsonb
        END
    ) END"""


POSTGRES_TRACE_EVENTS_DDL: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS cayu_trace_events (
        sequence BIGINT PRIMARY KEY,
        session_id TEXT NOT NULL,
        payload_bytes BIGINT NOT NULL,
        payload JSONB NOT NULL
    )
    """,
    f"""
    CREATE OR REPLACE FUNCTION cayu_track_trace_event()
    RETURNS TRIGGER AS $$
    BEGIN
        IF TG_OP IN ('UPDATE', 'DELETE') THEN
            DELETE FROM cayu_trace_events WHERE sequence = OLD.sequence;
        END IF;
        IF TG_OP IN ('INSERT', 'UPDATE') THEN
            INSERT INTO cayu_trace_events (sequence, session_id, payload_bytes, payload)
            VALUES (
                NEW.sequence, NEW.session_id, octet_length((NEW.event -> 'payload')::text),
                {postgres_trace_payload("(NEW.event -> 'payload')", "NEW.event_type")}
            )
            ON CONFLICT (sequence) DO UPDATE SET
                session_id = EXCLUDED.session_id,
                payload_bytes = EXCLUDED.payload_bytes,
                payload = EXCLUDED.payload;
        END IF;
        RETURN NULL;
    END;
    $$ LANGUAGE plpgsql
    """,
    "DROP TRIGGER IF EXISTS cayu_trace_events_insert ON cayu_events",
    """
    CREATE TRIGGER cayu_trace_events_insert
    AFTER INSERT ON cayu_events
    FOR EACH ROW
    EXECUTE FUNCTION cayu_track_trace_event()
    """,
    "DROP TRIGGER IF EXISTS cayu_trace_events_delete ON cayu_events",
    """
    CREATE TRIGGER cayu_trace_events_delete
    AFTER DELETE ON cayu_events
    FOR EACH ROW
    EXECUTE FUNCTION cayu_track_trace_event()
    """,
    "DROP TRIGGER IF EXISTS cayu_trace_events_update ON cayu_events",
    """
    CREATE TRIGGER cayu_trace_events_update
    AFTER UPDATE OF sequence, session_id, event_type, payload, event ON cayu_events
    FOR EACH ROW
    EXECUTE FUNCTION cayu_track_trace_event()
    """,
    f"""
    INSERT INTO cayu_trace_events (sequence, session_id, payload_bytes, payload)
    SELECT
        event.sequence, event.session_id, octet_length((event.event -> 'payload')::text),
        {postgres_trace_payload("(event.event -> 'payload')", "event.event_type")}
    FROM cayu_events AS event
    ON CONFLICT (sequence) DO NOTHING
    """,
)

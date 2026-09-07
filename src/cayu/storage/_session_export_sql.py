"""Content-free source-size preflight inside a session export read transaction."""

from __future__ import annotations


def export_size_statement(*, postgres: bool) -> tuple[str, int]:
    placeholder = "%s" if postgres else "?"
    tables = [
        (
            "cayu_sessions",
            "id",
            ("invocation", "metadata") if postgres else ("invocation_json", "metadata_json"),
        ),
        ("cayu_session_labels", "session_id", ("key", "value")),
        (
            "cayu_events",
            "session_id",
            ("event",)
            if postgres
            else (
                "payload_json",
                "event_id",
                "session_id",
                "agent_name",
                "environment_name",
                "timestamp",
                "event_type",
            ),
        ),
        ("cayu_transcript_messages", "session_id", ("message",) if postgres else ("message_json",)),
        ("cayu_checkpoints", "session_id", ("state",) if postgres else ("state_json",)),
        (
            "cayu_deferred_interaction_inputs",
            "session_id",
            ("source_messages",) if postgres else ("source_messages_json",),
        ),
        ("cayu_targeted_tool_grants", "session_id", ("record",) if postgres else ("record_json",)),
        (
            "cayu_targeted_tool_grant_uses",
            "session_id",
            ("record",) if postgres else ("record_json",),
        ),
    ]
    selects = []
    for table, identity, columns in tables:
        lengths = [
            f"COALESCE(octet_length({column}::text), 0)"
            if postgres
            else f"COALESCE(length(CAST({column} AS BLOB)), 0)"
            for column in columns
        ]
        selects.append(
            f"SELECT {' + '.join(lengths)} AS size FROM {table} WHERE {identity} = {placeholder}"
        )
    return (
        "SELECT COALESCE(SUM(size), 0), COALESCE(MAX(size), 0) FROM ("
        + " UNION ALL ".join(selects)
        + ") AS export_source_sizes",
        len(tables),
    )

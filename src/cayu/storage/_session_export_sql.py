"""Content-free source-size preflight inside a session export read transaction."""

from __future__ import annotations

from cayu.storage._session_closure_sql import stored_column_size


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
        # message_bytes is the full message size; a tool result stored in its
        # tool event is read back into the message, so count it here too.
        ("cayu_transcript_messages", "session_id", ("message_bytes",)),
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
        lengths = [stored_column_size(column, postgres=postgres) for column in columns]
        selects.append(
            f"SELECT {' + '.join(lengths)} AS size FROM {table} WHERE {identity} = {placeholder}"
        )
    return (
        "SELECT COALESCE(SUM(size), 0), COALESCE(MAX(size), 0) FROM ("
        + " UNION ALL ".join(selects)
        + ") AS export_source_sizes",
        len(tables),
    )

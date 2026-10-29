"""SQLite reads of transcript messages whose tool results live in their tool event.

See :mod:`cayu.storage._transcript_tool_results` for the stored form.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable, Iterable, Sequence
from typing import Any

from cayu.messages import Message
from cayu.storage import _transcript_tool_results as tool_results

_CHUNK = 400


def event_contents(
    connection: sqlite3.Connection, session_id: str, event_ids: Iterable[str]
) -> dict[str, str]:
    """Load ``result.content`` of the named tool terminal events in one session."""

    ids = sorted(set(event_ids))
    contents: dict[str, str] = {}
    for start in range(0, len(ids), _CHUNK):
        chunk = ids[start : start + _CHUNK]
        placeholders = ", ".join("?" for _ in chunk)
        for row in connection.execute(
            "SELECT event_id, json_extract(payload_json, '$.result.content') "
            "FROM cayu_events WHERE session_id = ? "
            f"AND event_id IN ({placeholders}) AND event_type IN (?, ?) "
            "AND json_type(payload_json, '$.result.content') = 'text'",
            (session_id, *chunk, *tool_results.TOOL_RESULT_EVENT_TYPES),
        ):
            contents[str(row[0])] = row[1]
    return contents


def read_snapshot(connection: sqlite3.Connection, read: Callable[[sqlite3.Connection], Any]) -> Any:
    """Run ``read`` in one read transaction unless the caller already holds one.

    A transcript read and the tool-event read that restores its results must
    see the same database state, or a concurrent compaction between them
    would make valid rows fail their digest check.
    """

    if connection.in_transaction:
        return read(connection)
    connection.execute("BEGIN")
    try:
        return read(connection)
    finally:
        connection.rollback()


def _parse(raw: Any) -> Any:
    return json.loads(raw) if isinstance(raw, str | bytes) else raw


def document_decoder(
    connection: sqlite3.Connection, session_id: str, raw_messages: Iterable[Any]
) -> Callable[[Any], Any]:
    """Return a function that decodes these stored messages of one session.

    The tool events the messages reference are loaded once, up front.
    """

    event_ids = [
        event_id
        for raw in raw_messages
        if tool_results.may_reference(raw)
        for event_id in tool_results.referenced_event_ids(_parse(raw))
    ]
    contents = event_contents(connection, session_id, event_ids) if event_ids else {}

    def decode(raw: Any) -> Any:
        document = _parse(raw)
        if tool_results.may_reference(raw):
            document = tool_results.resolve_message_document(document, contents)
        return document

    return decode


def decoder(
    connection: sqlite3.Connection, session_id: str, raw_messages: Iterable[Any]
) -> Callable[[Any], Message]:
    """Like :func:`document_decoder`, returning validated Messages."""

    decode = document_decoder(connection, session_id, raw_messages)
    return lambda raw: Message.model_validate(decode(raw))


def messages(
    connection: sqlite3.Connection, session_id: str, raw_messages: Sequence[Any]
) -> list[Message]:
    """Decode stored message JSON of one session into Messages."""

    decode = decoder(connection, session_id, raw_messages)
    return [decode(raw) for raw in raw_messages]


def message(connection: sqlite3.Connection, session_id: str, raw_message: Any) -> Message:
    return messages(connection, session_id, (raw_message,))[0]


def messages_by_session(
    connection: sqlite3.Connection, rows: Sequence[tuple[str, Any]]
) -> list[Message]:
    """Decode ``(session_id, message_json)`` rows that may span sessions."""

    by_session: dict[str, list[Any]] = {}
    for session_id, raw in rows:
        by_session.setdefault(session_id, []).append(raw)
    decoders = {
        session_id: decoder(connection, session_id, raws) for session_id, raws in by_session.items()
    }
    return [decoders[session_id](raw) for session_id, raw in rows]


def search_document_message(raw_message: Any) -> Message:
    """Decode a stored message for narrative-text projection without loading events.

    Tool results never contribute narrative text, so references are dropped.
    """

    document = _parse(raw_message)
    if tool_results.may_reference(raw_message):
        document = tool_results.without_references(document)
    return Message.model_validate(document)


__all__ = [
    "decoder",
    "document_decoder",
    "event_contents",
    "message",
    "messages",
    "messages_by_session",
    "read_snapshot",
    "search_document_message",
]

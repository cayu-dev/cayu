"""Postgres reads of transcript messages whose tool results live in their tool event.

See :mod:`cayu.storage._transcript_tool_results` for the stored form.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import Any

from cayu.messages import Message
from cayu.storage import _transcript_tool_results as tool_results
from cayu.storage._postgres_support import _json_obj

_CHUNK = 400


async def event_contents(cur: Any, session_id: str, event_ids: Iterable[str]) -> dict[str, str]:
    """Load ``result.content`` of the named tool terminal events in one session."""

    ids = sorted(set(event_ids))
    contents: dict[str, str] = {}
    for start in range(0, len(ids), _CHUNK):
        await cur.execute(
            "SELECT event_id, payload -> 'result' ->> 'content' FROM cayu_events "
            "WHERE session_id = %s AND event_id = ANY(%s) AND event_type = ANY(%s) "
            "AND jsonb_typeof(payload -> 'result' -> 'content') = 'string'",
            (
                session_id,
                ids[start : start + _CHUNK],
                list(tool_results.TOOL_RESULT_EVENT_TYPES),
            ),
        )
        for event_id, content in await cur.fetchall():
            contents[str(event_id)] = content
    return contents


async def document_decoder(
    cur: Any, session_id: str, raw_messages: Iterable[Any]
) -> Callable[[Any], Any]:
    """Return a function that decodes these stored messages of one session.

    The tool events the messages reference are loaded once, up front, so the
    caller must have fetched its own rows before calling this.
    """

    event_ids = [
        event_id
        for raw in raw_messages
        if tool_results.may_reference(raw)
        for event_id in tool_results.referenced_event_ids(_json_obj(raw))
    ]
    contents = await event_contents(cur, session_id, event_ids) if event_ids else {}

    def decode(raw: Any) -> Any:
        document = _json_obj(raw)
        if tool_results.may_reference(raw):
            document = tool_results.resolve_message_document(document, contents)
        return document

    return decode


async def decoder(
    cur: Any, session_id: str, raw_messages: Iterable[Any]
) -> Callable[[Any], Message]:
    """Like :func:`document_decoder`, returning validated Messages."""

    decode = await document_decoder(cur, session_id, raw_messages)
    return lambda raw: Message.model_validate(decode(raw))


async def message_documents(cur: Any, session_id: str, raw_messages: Sequence[Any]) -> list[Any]:
    decode = await document_decoder(cur, session_id, raw_messages)
    return [decode(raw) for raw in raw_messages]


async def messages(cur: Any, session_id: str, raw_messages: Sequence[Any]) -> list[Message]:
    """Decode stored message JSON of one session into Messages."""

    decode = await decoder(cur, session_id, raw_messages)
    return [decode(raw) for raw in raw_messages]


async def message(cur: Any, session_id: str, raw_message: Any) -> Message:
    return (await messages(cur, session_id, (raw_message,)))[0]


__all__ = [
    "decoder",
    "document_decoder",
    "event_contents",
    "message",
    "message_documents",
    "messages",
]

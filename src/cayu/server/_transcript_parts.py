"""Transcript pages with long parts cut short, and one whole message by index.

A tool result can be megabytes and the first user message hundreds of
kilobytes, so a transcript page that carries them whole is slow to send and to
render. With ``max_part_bytes`` a page cuts each longer text or tool-result part
to that many UTF-8 bytes after redaction, leaves out a tool result's longer
structured data and artifacts, and marks the part with ``truncated`` and its
``full_bytes``. ``GET /sessions/{session_id}/transcript/{message_index}``
returns that message whole.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from typing import Annotated, Any

from fastapi import APIRouter, HTTPException, Path
from pydantic import StringConstraints

from cayu._validation import MAX_DURABLE_JSON_INTEGER
from cayu.server.contracts import ApiTranscriptMessage
from cayu.sessions.store import SessionStore

NonBlankString = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]

#: The member each part type keeps its long text in.
_LONG_TEXT_FIELDS = {"text": "text", "tool_result": "content"}
#: Tool-result members dropped, as their empty value, when longer than the cut.
_LONG_DATA_FIELDS = {"structured": None, "artifacts": []}
#: The smallest cut a page may ask for, so a truncated part still shows its start.
MIN_PART_BYTES = 256


def _json_bytes(value: Any) -> int:
    return len(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode())


def truncate_message_parts(message: dict[str, Any], max_bytes: int) -> dict[str, Any]:
    """Bound each text and tool-result part of a projected message by ``max_bytes``.

    A longer text, or tool-result ``content``, is cut on a character boundary.
    A tool result's ``structured`` data or ``artifacts`` longer than
    ``max_bytes`` as JSON are left out, as ``null`` and ``[]``. A part changed
    either way carries ``truncated: true`` and ``full_bytes``, the size of the
    whole projected part as JSON. Other parts are unchanged.
    """

    content = []
    for part in message["content"]:
        part_type = part.get("type") if isinstance(part, dict) else ""
        field = _LONG_TEXT_FIELDS.get(part_type)
        if field is None:
            content.append(part)
            continue
        cut = dict(part)
        value = part.get(field)
        if isinstance(value, str) and len(value) > max_bytes // 4:
            encoded = value.encode()
            if len(encoded) > max_bytes:
                cut[field] = encoded[:max_bytes].decode(errors="ignore")
        if part_type == "tool_result":
            for name, empty in _LONG_DATA_FIELDS.items():
                if part.get(name) and _json_bytes(part[name]) > max_bytes:
                    cut[name] = empty
        if cut != part:
            cut["truncated"] = True
            cut["full_bytes"] = _json_bytes(part)
        content.append(cut)
    return {**message, "content": content}


def register_transcript_message(
    router: APIRouter,
    *,
    protected: list[Any],
    store: SessionStore,
    resolve_session: Callable[[str], Awaitable[str]],
    serialize_message: Callable[..., dict[str, Any]],
) -> None:
    @router.get(
        "/sessions/{session_id}/transcript/{message_index}",
        response_model=ApiTranscriptMessage,
        dependencies=protected,
        description=(
            "Return one transcript message whole, by its index, including any part a "
            "transcript page cut short with `max_part_bytes`."
        ),
    )
    async def get_session_transcript_message(
        session_id: NonBlankString,
        message_index: Annotated[int, Path(ge=0, le=MAX_DURABLE_JSON_INTEGER)],
    ):
        session_id = await resolve_session(session_id)
        if await store.load_state(session_id) is None:
            raise HTTPException(status_code=404, detail="Session not found")
        # The index is absolute: retention removes messages without renumbering them.
        window = await store.load_transcript_window(session_id, start_index=message_index, limit=1)
        record = next((item for item in window.records if item.index == message_index), None)
        if record is None:
            raise HTTPException(status_code=404, detail="Transcript message not found")
        return serialize_message(session_id, record.index, record.message, record.interaction_id)

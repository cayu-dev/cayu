"""SSE event serialization for the cayu server."""

from __future__ import annotations

import json
from typing import Any

from cayu.core.events import Event


def event_to_sse_data(event: Event) -> str:
    """Serialize a runtime Event to a JSON string for SSE."""
    data: dict[str, Any] = {
        "id": event.id,
        "type": str(event.type),
        "session_id": event.session_id,
        "agent_name": event.agent_name,
        "tool_name": event.tool_name,
        "payload": event.payload,
        "timestamp": event.timestamp.isoformat(),
    }
    if event.environment_name is not None:
        data["environment_name"] = event.environment_name
    if event.workflow_name is not None:
        data["workflow_name"] = event.workflow_name
    return json.dumps(data)


def sse_event_id(event: Event) -> str:
    """Stable SSE ``id:`` field for an event: ``<session_id>:<event_id>``.

    Carrying the session id lets a reconnecting client's ``Last-Event-ID`` name
    both the session and the last event it saw, so the server can replay the
    persisted events it missed.
    """
    return f"{event.session_id}:{event.id}"


def event_to_sse_message(event: Event) -> dict[str, str]:
    """Serialize a runtime Event to an SSE message with a resumable ``id:``."""
    return {"id": sse_event_id(event), "data": event_to_sse_data(event)}


def error_to_sse_message(error: BaseException) -> dict[str, str]:
    """Terminal structured ``error`` SSE frame for a failed event stream.

    Emitted as the last frame instead of aborting the connection, so consumers
    can distinguish a runtime failure from a transport drop.
    """
    return {
        "event": "error",
        "data": json.dumps(
            {
                "type": "stream.error",
                "error": str(error),
                "error_type": type(error).__name__,
            }
        ),
    }


def parse_last_event_id(value: str) -> tuple[str, str] | None:
    """Parse a ``Last-Event-ID`` header (``<session_id>:<event_id>``).

    Returns ``None`` when the value does not carry both parts.
    """
    session_id, sep, event_id = value.partition(":")
    session_id = session_id.strip()
    event_id = event_id.strip()
    if not sep or not session_id or not event_id:
        return None
    return session_id, event_id

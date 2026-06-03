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
    return json.dumps(data)

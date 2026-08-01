from __future__ import annotations

import hashlib
import json


def tool_idempotency_key(
    *,
    session_id: str,
    tool_call_id: str,
    tool_round_id: str | None = None,
    approval_id: str | None = None,
    pause_id: str | None = None,
) -> str:
    """Return the stable content-addressed identity for one tool execution."""

    components = (
        "cayu-tool-idempotency-v1",
        session_id,
        tool_round_id or "",
        approval_id or "",
        pause_id or "",
        tool_call_id,
    )
    material = json.dumps(components, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return "cayu-tool:v1:" + hashlib.sha256(material).hexdigest()

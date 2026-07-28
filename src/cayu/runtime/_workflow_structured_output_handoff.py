"""Process-local handoff for workflow child structured output."""

from __future__ import annotations

from typing import Any

from cayu._validation import copy_json_value, require_clean_nonblank

_MISSING = object()
_PENDING = object()


class WorkflowStructuredOutputHandoff:
    """Track one opt-in structured-output handoff per live child session."""

    def __init__(self) -> None:
        self._entries: dict[str, object] = {}

    def prepare(self, session_id: str) -> None:
        session_id = require_clean_nonblank(session_id, "session_id")
        if session_id in self._entries:
            raise RuntimeError("Workflow structured-output capture is already active.")
        self._entries[session_id] = _PENDING

    def take(self, session_id: str) -> tuple[bool, Any]:
        session_id = require_clean_nonblank(session_id, "session_id")
        output = self._entries.pop(session_id, _MISSING)
        if output is _MISSING or output is _PENDING:
            return False, None
        return True, output

    def discard(self, session_id: str) -> None:
        session_id = require_clean_nonblank(session_id, "session_id")
        self._entries.pop(session_id, None)

    def record(self, session_id: str, output: Any) -> None:
        if session_id not in self._entries:
            return
        self._entries[session_id] = copy_json_value(
            output,
            "workflow_structured_output",
        )


__all__ = ["WorkflowStructuredOutputHandoff"]

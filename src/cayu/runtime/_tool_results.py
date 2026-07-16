from __future__ import annotations

from copy import deepcopy
from typing import Any

from cayu._validation import copy_json_value
from cayu.core.events import Event
from cayu.core.tools import ToolResult
from cayu.vaults import SecretRedactor


def normalize_tool_result(result: ToolResult) -> ToolResult:
    if result.is_error and not result.content.strip():
        return result.model_copy(update={"content": "Tool returned an error without details."})
    return result


def validate_tool_result(result: ToolResult) -> ToolResult:
    if type(result) is not ToolResult:
        raise TypeError("Tool results must be ToolResult instances.")
    return ToolResult(
        content=result.content,
        structured=copy_json_value(result.structured, "structured"),
        artifacts=copy_json_value(result.artifacts, "artifacts"),
        is_error=result.is_error,
    )


def redact_tool_result(result: ToolResult, redactor: SecretRedactor) -> ToolResult:
    if type(result) is not ToolResult:
        raise TypeError("Tool results must be ToolResult instances.")
    if not isinstance(redactor, SecretRedactor):
        raise TypeError("redactor must be a SecretRedactor.")
    if not redactor.has_values:
        return result
    return ToolResult(
        content=redactor.redact_text(result.content),
        structured=redactor.redact_json(result.structured),
        artifacts=redactor.redact_json(result.artifacts),
        is_error=result.is_error,
    )


def redact_tool_result_event(
    *,
    event: Event,
    result: ToolResult,
    redactor: SecretRedactor,
) -> tuple[Event, ToolResult]:
    """Redact a terminal event payload and keep its result field synchronized."""

    redacted_result = redact_tool_result(result, redactor)
    if not redactor.has_values:
        return event, redacted_result
    payload = redactor.redact_json(event.payload)
    if type(payload) is not dict:
        raise AssertionError("Event payload redaction returned non-object payload.")
    payload["result"] = redacted_result.model_dump()
    return event.model_copy(update={"payload": payload}), redacted_result


def tool_result_from_payload(payload: dict[str, Any]) -> ToolResult:
    return normalize_tool_result(validate_tool_result(ToolResult(**deepcopy(payload))))


def exception_message(exc: Exception) -> str:
    message = str(exc).strip()
    if message:
        return message
    return f"{type(exc).__name__}: tool execution failed"

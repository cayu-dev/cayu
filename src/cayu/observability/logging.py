from __future__ import annotations

import logging
from typing import Any

from cayu.core.events import Event, EventType
from cayu.runtime.event_sinks import EventSink
from cayu.vaults import SecretRedactor

DEFAULT_CAYU_LOGGER_NAME = "cayu"
DEFAULT_ERROR_SUMMARY_LIMIT = 200

_DEBUG_EVENTS = {
    EventType.MODEL_TEXT_DELTA,
    EventType.HOOK_STARTED,
    EventType.HOOK_COMPLETED,
}
_WARNING_EVENTS = {
    EventType.SESSION_INTERRUPTED,
    EventType.SESSION_LIMIT_REACHED,
    EventType.MODEL_ERROR,
    EventType.MODEL_RETRY,
    EventType.TOOL_CALL_BLOCKED,
    EventType.TOOL_CALL_APPROVAL_DENIED,
    EventType.RUNTIME_SINK_FAILED,
}
_ERROR_EVENTS = {
    EventType.SESSION_FAILED,
    EventType.TOOL_CALL_FAILED,
    EventType.TASK_FAILED,
    EventType.HOOK_FAILED,
    EventType.CONTEXT_COMPACTION_FAILED,
}


class LoggingEventSink(EventSink):
    """Emit concise runtime event summaries through Python logging.

    The sink does not configure global handlers, process logging levels, or
    formatter state. Applications stay responsible for logging configuration.
    """

    def __init__(
        self,
        *,
        logger: logging.Logger | None = None,
        logger_name: str = DEFAULT_CAYU_LOGGER_NAME,
        error_summary_limit: int = DEFAULT_ERROR_SUMMARY_LIMIT,
        redactor: SecretRedactor | None = None,
    ) -> None:
        if logger is not None and not isinstance(logger, logging.Logger):
            raise TypeError("LoggingEventSink logger must be a logging.Logger.")
        if type(logger_name) is not str or not logger_name.strip():
            raise ValueError("LoggingEventSink logger_name must be a non-empty string.")
        if type(error_summary_limit) is not int:
            raise TypeError("LoggingEventSink error_summary_limit must be an integer.")
        if error_summary_limit <= 0:
            raise ValueError("LoggingEventSink error_summary_limit must be greater than zero.")
        if redactor is not None and not isinstance(redactor, SecretRedactor):
            raise TypeError("LoggingEventSink redactor must be a SecretRedactor.")
        self.logger = logger if logger is not None else logging.getLogger(logger_name)
        self.error_summary_limit = error_summary_limit
        self.redactor = redactor if redactor is not None else SecretRedactor()

    async def emit(self, event: Event) -> None:
        if type(event) is not Event:
            raise TypeError("LoggingEventSink requires Event instances.")
        level = _level_for(event.type)
        if not self.logger.isEnabledFor(level):
            return
        self.logger.log(
            level,
            "%-30s | %s",
            _clean(str(event.type), redactor=self.redactor),
            _summarize_event(
                event,
                error_summary_limit=self.error_summary_limit,
                redactor=self.redactor,
            ),
        )


def _level_for(event_type: EventType | str) -> int:
    if event_type in _DEBUG_EVENTS:
        return logging.DEBUG
    if event_type in _ERROR_EVENTS:
        return logging.ERROR
    if event_type in _WARNING_EVENTS:
        return logging.WARNING
    return logging.INFO


def _summarize_event(
    event: Event,
    *,
    error_summary_limit: int,
    redactor: SecretRedactor,
) -> str:
    parts = [_identity(event, redactor=redactor)]
    payload = event.payload
    event_type = event.type
    if event_type == EventType.SESSION_STARTED or event_type == EventType.SESSION_RESUMED:
        _append(parts, "agent", event.agent_name, redactor=redactor)
        _append(parts, "environment", event.environment_name, redactor=redactor)
    elif event_type == EventType.MODEL_STARTED:
        _append(parts, "provider", payload.get("provider"), redactor=redactor)
        _append(parts, "model", payload.get("model"), redactor=redactor)
        _append(parts, "step", payload.get("step"), redactor=redactor)
    elif event_type == EventType.MODEL_COMPLETED:
        _append_usage(
            parts,
            payload.get("usage_metrics") or payload.get("usage"),
            redactor=redactor,
        )
    elif event_type == EventType.MODEL_RETRY:
        _append(parts, "provider", payload.get("provider"), redactor=redactor)
        _append(parts, "model", payload.get("model"), redactor=redactor)
        _append(parts, "step", payload.get("step"), redactor=redactor)
        _append(parts, "attempt", payload.get("attempt"), redactor=redactor)
        _append(parts, "next_attempt", payload.get("next_attempt"), redactor=redactor)
        _append(parts, "max_attempts", payload.get("max_attempts"), redactor=redactor)
        _append(parts, "reason", payload.get("reason"), redactor=redactor)
        _append(parts, "delay_seconds", payload.get("delay_seconds"), redactor=redactor)
        _append_error(parts, payload, limit=error_summary_limit, redactor=redactor)
    elif event_type == EventType.SESSION_LIMIT_REACHED:
        _append(parts, "limit", payload.get("limit"), redactor=redactor)
        _append(parts, "actual", payload.get("actual"), redactor=redactor)
        _append(parts, "maximum", payload.get("maximum"), redactor=redactor)
    elif event_type in {
        EventType.TOOL_CALL_STARTED,
        EventType.TOOL_CALL_COMPLETED,
        EventType.TOOL_CALL_FAILED,
        EventType.TOOL_CALL_BLOCKED,
        EventType.TOOL_CALL_APPROVAL_REQUESTED,
        EventType.TOOL_CALL_APPROVED,
        EventType.TOOL_CALL_APPROVAL_DENIED,
    }:
        _append(parts, "tool", event.tool_name, redactor=redactor)
        _append(parts, "call", payload.get("tool_call_id"), redactor=redactor)
        _append_error(parts, payload, limit=error_summary_limit, redactor=redactor)
    elif event_type in {
        EventType.SESSION_FAILED,
        EventType.MODEL_ERROR,
        EventType.TASK_FAILED,
        EventType.HOOK_FAILED,
        EventType.RUNTIME_SINK_FAILED,
        EventType.CONTEXT_COMPACTION_FAILED,
    }:
        _append_error(parts, payload, limit=error_summary_limit, redactor=redactor)
    elif event_type in {
        EventType.TASK_CREATED,
        EventType.TASK_STARTED,
        EventType.TASK_COMPLETED,
        EventType.TASK_CANCELLED,
    }:
        _append(parts, "task", payload.get("task_id"), redactor=redactor)
    elif event_type == EventType.CONTEXT_COMPACTION_COMPLETED:
        _append(parts, "messages_before", payload.get("messages_before"), redactor=redactor)
        _append(parts, "messages_after", payload.get("messages_after"), redactor=redactor)
        _append(parts, "summary_chars", payload.get("summary_chars"), redactor=redactor)
    elif event_type == EventType.SESSION_CHECKPOINTED:
        _append(parts, "checkpoint", payload.get("checkpoint"), redactor=redactor)
    return " | ".join(parts)


def _identity(event: Event, *, redactor: SecretRedactor) -> str:
    parts = [_clean(event.session_id, redactor=redactor)]
    if event.agent_name:
        parts.append(f"agent={_clean(event.agent_name, redactor=redactor)}")
    if event.environment_name:
        parts.append(f"env={_clean(event.environment_name, redactor=redactor)}")
    return " ".join(parts)


def _append(
    parts: list[str],
    key: str,
    value: Any,
    *,
    redactor: SecretRedactor,
) -> None:
    if value is None:
        return
    if type(value) in {str, int, float, bool}:
        parts.append(f"{key}={_clean(str(value), redactor=redactor)}")


def _append_usage(parts: list[str], usage: Any, *, redactor: SecretRedactor) -> None:
    if type(usage) is not dict:
        return
    _append(parts, "input_tokens", usage.get("input_tokens"), redactor=redactor)
    _append(parts, "output_tokens", usage.get("output_tokens"), redactor=redactor)
    _append(parts, "total_tokens", usage.get("total_tokens"), redactor=redactor)
    _append(
        parts,
        "reasoning_output_tokens",
        usage.get("reasoning_output_tokens"),
        redactor=redactor,
    )
    cache = usage.get("cache")
    if type(cache) is dict:
        _append(parts, "cache_read_tokens", cache.get("read_tokens"), redactor=redactor)
        _append(parts, "cache_write_tokens", cache.get("write_tokens"), redactor=redactor)
        _append(
            parts,
            "cached_input_tokens",
            cache.get("cached_input_tokens"),
            redactor=redactor,
        )
        return
    input_details = usage.get("input_tokens_details")
    if type(input_details) is dict:
        _append(parts, "cached_tokens", input_details.get("cached_tokens"), redactor=redactor)


def _append_error(
    parts: list[str],
    payload: dict[str, Any],
    *,
    limit: int,
    redactor: SecretRedactor,
) -> None:
    error_type = payload.get("error_type")
    error = payload.get("error")
    if type(error_type) is str and error_type:
        parts.append(f"error_type={_clean(error_type, redactor=redactor)}")
    if type(error) is str and error:
        parts.append(f"error={_truncate(_clean(error, redactor=redactor), limit)}")


def _clean(value: str, *, redactor: SecretRedactor) -> str:
    return redactor.redact_text(value).encode("unicode_escape").decode("ascii")


def _truncate(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return f"{value[:limit]}..."

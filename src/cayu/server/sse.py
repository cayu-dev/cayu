"""SSE event serialization for the cayu server."""

from __future__ import annotations

import json
from typing import Any, Literal

from cayu._validation import json_utf8_size_within_limit, require_clean_nonblank
from cayu.core.events import EVENT_ID_MAX_CHARS, Event

SSE_EVENT_DATA_MAX_BYTES = 2 * 1024 * 1024
SSE_ERROR_TEXT_MAX_BYTES = 512
SSE_ERROR_TYPE_MAX_BYTES = 128
SSE_ERROR_SESSION_ID_MAX_BYTES = 512
SSE_OBSERVER_MAX_FRAMES = 256
SSE_OBSERVER_MAX_BYTES = 2 * 1024 * 1024
SSE_REPLAY_PAGE_EVENTS = 32
SSE_REPLAY_START_MARKER_FORMAT = "session_id:"
SSE_SEND_TIMEOUT_SECONDS = 30.0

SseErrorKind = Literal["runtime", "observer"]
SseErrorCode = Literal[
    "runtime_failed",
    "observer_lagged",
    "event_frame_too_large",
    "replay_idle_timeout",
]

_SSE_ERROR_KINDS = frozenset({"runtime", "observer"})
_SSE_ERROR_CODES = frozenset(
    {
        "runtime_failed",
        "observer_lagged",
        "event_frame_too_large",
        "replay_idle_timeout",
    }
)
_TRUNCATED_SUFFIX = "... [truncated]"


class SseEventFrameTooLargeError(ValueError):
    """A durable event cannot be represented inside the live SSE frame limit."""

    def __init__(
        self,
        *,
        session_id: str,
        max_bytes: int,
        actual_bytes: int | None = None,
    ) -> None:
        self.session_id = require_clean_nonblank(session_id, "session_id")
        self.actual_bytes = actual_bytes
        self.max_bytes = max_bytes
        size = "exceeds" if actual_bytes is None else f"is {actual_bytes} bytes and exceeds"
        super().__init__(
            f"SSE event for session {self.session_id} {size} the {max_bytes}-byte "
            "live frame limit. Use durable history instead."
        )


class SseObserverLaggedError(RuntimeError):
    """The transient observer buffer filled while durable execution continued."""


def _event_to_sse_payload(event: Event) -> dict[str, Any]:
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
    return data


def event_to_sse_data(event: Event) -> str:
    """Serialize a runtime Event to a compact JSON string for SSE."""
    return json.dumps(_event_to_sse_payload(event), separators=(",", ":"))


def sse_event_id(event: Event) -> str:
    """Stable SSE ``id:`` field for an event: ``<session_id>:<event_id>``.

    Carrying the session id lets a reconnecting client's ``Last-Event-ID`` name
    both the session and the last event it saw, so the server can replay the
    persisted events it missed.
    """
    return f"{event.session_id}:{event.id}"


def event_to_sse_message(
    event: Event,
    *,
    max_data_bytes: int = SSE_EVENT_DATA_MAX_BYTES,
) -> dict[str, str]:
    """Serialize a bounded runtime event with a resumable ``id:``.

    Durable events are never truncated to fit the live transport. An oversized
    event raises a typed observer error so the caller can direct the client to
    bounded durable-history reads without silently changing event data.
    """
    if type(max_data_bytes) is not int or max_data_bytes <= 0:
        raise ValueError("max_data_bytes must be a positive integer.")
    payload = _event_to_sse_payload(event)
    if not json_utf8_size_within_limit(payload, max_data_bytes, ensure_ascii=True):
        raise SseEventFrameTooLargeError(
            session_id=event.session_id,
            max_bytes=max_data_bytes,
        )
    data = json.dumps(payload, separators=(",", ":"))
    actual_bytes = len(data.encode("utf-8"))
    # Defensive verification keeps the wire ceiling authoritative if JSON
    # encoding behavior ever diverges from the allocation-free preflight.
    if actual_bytes > max_data_bytes:
        raise SseEventFrameTooLargeError(
            session_id=event.session_id,
            max_bytes=max_data_bytes,
            actual_bytes=actual_bytes,
        )
    return {"id": sse_event_id(event), "data": data}


def sse_message_data_bytes(message: dict[str, str]) -> int:
    """Return the UTF-8 byte size of one serialized SSE ``data`` value."""
    data = message.get("data")
    if type(data) is not str:
        raise ValueError("SSE message must contain a string data field.")
    return len(data.encode("utf-8"))


def error_to_sse_message(
    error: BaseException,
    *,
    kind: SseErrorKind,
    code: SseErrorCode,
    retryable: bool,
    session_id: str | None = None,
    error_text: str | None = None,
    max_error_bytes: int = SSE_ERROR_TEXT_MAX_BYTES,
) -> dict[str, str]:
    """Serialize a bounded, classified terminal ``error`` SSE frame.

    ``error_text`` is supplied separately so the server route can apply the
    application's configured secret redactor before this transport-level size
    bound. When it is omitted, the serializer uses a generic message rather
    than exposing the raw exception string.
    """
    if kind not in _SSE_ERROR_KINDS:
        raise ValueError(f"Unsupported SSE error kind: {kind!r}.")
    if code not in _SSE_ERROR_CODES:
        raise ValueError(f"Unsupported SSE error code: {code!r}.")
    if type(retryable) is not bool:
        raise TypeError("retryable must be a boolean.")
    if type(max_error_bytes) is not int or max_error_bytes <= 0:
        raise ValueError("max_error_bytes must be a positive integer.")
    if session_id is not None:
        if type(session_id) is not str:
            require_clean_nonblank(session_id, "session_id")
        elif not _valid_utf8_within_limit(session_id, SSE_ERROR_SESSION_ID_MAX_BYTES):
            session_id = None
        else:
            session_id = require_clean_nonblank(session_id, "session_id")
    text = f"{type(error).__name__}: stream failed." if error_text is None else error_text
    if type(text) is not str:
        raise TypeError("error_text must be a string or None.")
    text = _truncate_utf8(text, max_error_bytes).strip()
    if not text:
        text = _truncate_utf8(f"{type(error).__name__}: stream failed.", max_error_bytes)
    error_type = _truncate_utf8(type(error).__name__, SSE_ERROR_TYPE_MAX_BYTES)
    return {
        "event": "error",
        "data": json.dumps(
            {
                "type": "stream.error",
                "kind": kind,
                "code": code,
                "error": text,
                "error_type": error_type,
                "retryable": retryable,
                "session_id": session_id,
            },
            separators=(",", ":"),
        ),
    }


def _truncate_utf8(value: str, max_bytes: int) -> str:
    characters: list[tuple[str, int]] = []
    used_bytes = 0
    for character in value:
        try:
            encoded = character.encode("utf-8")
        except UnicodeEncodeError:
            encoded = "�".encode()
        size = len(encoded)
        if used_bytes + size > max_bytes:
            break
        characters.append((encoded.decode("utf-8"), size))
        used_bytes += size
    else:
        return "".join(character for character, _ in characters)

    suffix_bytes = len(_TRUNCATED_SUFFIX.encode("utf-8"))
    include_suffix = suffix_bytes < max_bytes
    prefix_limit = max_bytes - suffix_bytes if include_suffix else max_bytes
    while characters and used_bytes > prefix_limit:
        _, size = characters.pop()
        used_bytes -= size
    prefix = "".join(character for character, _ in characters)
    return prefix + _TRUNCATED_SUFFIX if include_suffix else prefix


def _valid_utf8_within_limit(value: str, max_bytes: int) -> bool:
    used_bytes = 0
    for character in value:
        try:
            used_bytes += len(character.encode("utf-8"))
        except UnicodeEncodeError:
            return False
        if used_bytes > max_bytes:
            return False
    return True


def parse_last_event_id(
    value: str,
    *,
    expected_session_id: str | None = None,
) -> tuple[str, str | None] | None:
    """Parse an event marker or the explicit ``<session_id>:`` start marker.

    A ``None`` event id means replay from the start of the named session. Supplying
    ``expected_session_id`` preserves existing session identities that contain a
    colon by removing the exact known prefix instead of guessing a split point.
    Marker components are intentionally strict because a reconnect request must
    never fall back from a malformed or unknown boundary to an ambiguous mutation.
    """
    if type(value) is not str or value != value.strip():
        return None
    expected_prefix = f"{expected_session_id}:" if expected_session_id is not None else ""
    if expected_session_id is not None and value.startswith(expected_prefix):
        session_id = expected_session_id
        event_id = value[len(expected_prefix) :]
        sep = ":"
    else:
        session_id, sep, event_id = value.partition(":")
    if not sep or not _valid_sse_marker_component(session_id):
        return None
    if not event_id:
        return session_id, None
    if not _valid_sse_marker_component(event_id) or len(event_id) > EVENT_ID_MAX_CHARS:
        return None
    return session_id, event_id


def _valid_sse_marker_component(value: str) -> bool:
    return (
        bool(value)
        and value == value.strip()
        and not any(ord(character) < 0x20 or ord(character) == 0x7F for character in value)
    )

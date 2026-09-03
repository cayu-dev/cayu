"""Shared SSE-to-JSON parsing for streaming provider transports.

Providers that stream over Server-Sent Events (OpenAI Responses, Chat
Completions) decode through this single parser, so heartbeat handling and the
idle-timeout timer cannot drift between adapters.
"""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import AsyncIterator, Mapping
from typing import Any

from cayu._validation import require_finite
from cayu.providers.deadlines import (
    ProviderDeadlineKind,
    ProviderStreamDeadlineController,
)

DEFAULT_SSE_MAX_EVENT_BYTES = 16 * 1024 * 1024
DEFAULT_SSE_MAX_EVENT_LINES = 4_096
DEFAULT_SSE_EVENT_DURATION_MULTIPLIER = 4.0
_SSE_LINE_ENDING = re.compile(rb"[\r\n]")
_SSE_LINE_CHECKPOINT_INTERVAL = 256


class _SseByteActivity:
    """Internal signal that raw response bytes arrived before a complete line."""

    __slots__ = ("line_in_progress",)

    def __init__(self, *, line_in_progress: bool) -> None:
        self.line_in_progress = line_in_progress


_SSE_BYTE_ACTIVITY = _SseByteActivity(line_in_progress=False)
_SSE_PENDING_LINE_ACTIVITY = _SseByteActivity(line_in_progress=True)


class SseEventTimeoutError(TimeoutError):
    """Raised when one unterminated SSE event exceeds its duration ceiling."""


class SseEventLimitError(RuntimeError):
    """Raised when one SSE event exceeds a fixed size or line-count ceiling."""


async def _aiter_bounded_sse_lines(
    chunks: AsyncIterator[bytes],
    *,
    max_line_bytes: int,
    provider_label: str,
    emit_byte_activity: bool = False,
    deadline_controller: ProviderStreamDeadlineController | None = None,
) -> AsyncIterator[str | _SseByteActivity]:
    """Decode UTF-8 SSE lines without allowing unbounded line accumulation.

    ``httpx.Response.aiter_lines`` buffers until it encounters a line ending,
    so an upstream can otherwise grow an unterminated line without reaching
    the event-level checks below. SSE is UTF-8 and recognizes LF, CRLF, and CR
    line endings. Invalid UTF-8 retains httpx's replacement-decoding behavior.
    """
    if type(max_line_bytes) is not int or max_line_bytes <= 0:
        raise ValueError("max_line_bytes must be a positive integer.")

    limit_message = f"{provider_label} SSE line exceeded the {max_line_bytes}-byte limit."
    line = bytearray()
    discard_line_feed = False
    lines_since_checkpoint = 0

    async for chunk in chunks:
        chunk_had_bytes = bool(chunk)
        if chunk_had_bytes and deadline_controller is not None:
            deadline_controller.observe_transport()
        yielded_line = False
        chunk_view = memoryview(chunk)
        offset = 0
        while offset < len(chunk):
            if discard_line_feed:
                discard_line_feed = False
                if chunk[offset] == 0x0A:
                    offset += 1
                    continue

            # Search for either delimiter once so delimiter-dense chunks do not
            # rescan the remaining suffix for the delimiter that is absent.
            delimiter_match = _SSE_LINE_ENDING.search(chunk, offset)
            delimiter = len(chunk) if delimiter_match is None else delimiter_match.start()
            segment_bytes = delimiter - offset
            if len(line) + segment_bytes > max_line_bytes:
                line.clear()
                # Match objects retain their source bytes. Drop every local
                # reference before the bounded failure crosses the provider boundary.
                delimiter_match = None
                chunk_view.release()
                chunk = b""
                raise SseEventLimitError(limit_message)
            line.extend(chunk_view[offset:delimiter])
            if delimiter == len(chunk):
                break

            decoded_line = line.decode("utf-8", errors="replace")
            line.clear()
            discard_line_feed = chunk[delimiter] == 0x0D
            offset = delimiter + 1
            yield decoded_line
            decoded_line = ""
            yielded_line = True
            lines_since_checkpoint += 1
            if lines_since_checkpoint >= _SSE_LINE_CHECKPOINT_INTERVAL:
                lines_since_checkpoint = 0
                # Async-generator yields do not guarantee a scheduler turn when
                # the consumer immediately requests the next already-buffered line.
                await asyncio.sleep(0)

        chunk_view.release()
        if emit_byte_activity and chunk_had_bytes:
            if line:
                # The bytes after the final delimiter have not produced a line
                # yet, so surface their activity without exposing them to the
                # provider parser.
                yield _SSE_PENDING_LINE_ACTIVITY
            elif not yielded_line:
                # A split CRLF can consume the LF without producing a second
                # logical line. It still counts as received transport activity.
                yield _SSE_BYTE_ACTIVITY

    if line:
        decoded_line = line.decode("utf-8", errors="replace")
        line.clear()
        yield decoded_line


async def aiter_sse_json_events(
    lines: AsyncIterator[str | _SseByteActivity],
    *,
    deadline_controller: ProviderStreamDeadlineController,
    provider_label: str,
    protocol_error: type[Exception],
    max_event_bytes: int = DEFAULT_SSE_MAX_EVENT_BYTES,
    max_event_lines: int = DEFAULT_SSE_MAX_EVENT_LINES,
    max_event_duration_s: float | None = None,
) -> AsyncIterator[Mapping[str, Any]]:
    """Decode an SSE line stream into JSON data objects.

    Raw bytes refresh only the transport clock. Only complete, valid JSON data
    events refresh the protocol clock; comments, incomplete lines, and malformed
    events do not. A separate duration ceiling bounds one incomplete line/event
    even while raw data keeps arriving.
    Error messages are prefixed with ``provider_label``; malformed SSE data
    raises ``protocol_error``.
    """
    if type(deadline_controller) is not ProviderStreamDeadlineController:
        raise TypeError("deadline_controller must be ProviderStreamDeadlineController.")
    if type(max_event_bytes) is not int or max_event_bytes <= 0:
        raise ValueError("max_event_bytes must be a positive integer.")
    if type(max_event_lines) is not int or max_event_lines <= 0:
        raise ValueError("max_event_lines must be a positive integer.")
    if max_event_duration_s is None:
        max_event_duration_s = (
            deadline_controller.deadlines.protocol_idle_timeout_s
            * DEFAULT_SSE_EVENT_DURATION_MULTIPLIER
        )
    if type(max_event_duration_s) not in {int, float}:
        raise ValueError("max_event_duration_s must be greater than zero.")
    max_event_duration_s = require_finite(float(max_event_duration_s), "max_event_duration_s")
    if max_event_duration_s <= 0:
        raise ValueError("max_event_duration_s must be greater than zero.")
    event_timeout_message = (
        f"{provider_label} streaming response did not finish one SSE event "
        f"within {max_event_duration_s:g} seconds."
    )
    event_bytes_message = f"{provider_label} SSE event exceeded the {max_event_bytes}-byte limit."
    event_lines_message = f"{provider_label} SSE event exceeded the {max_event_lines}-line limit."

    def decode(data: str) -> Mapping[str, Any]:
        try:
            decoded = json.loads(data)
        except ValueError as exc:
            raise protocol_error(f"{provider_label} SSE data was not valid JSON.") from exc
        if not isinstance(decoded, Mapping):
            raise protocol_error(f"{provider_label} SSE data must decode to a JSON object.")
        return decoded

    iterator = lines.__aiter__()
    loop = asyncio.get_running_loop()
    event_started_at: float | None = None
    pending_line_started_at: float | None = None
    event_bytes = 0
    event_lines = 0
    has_data = False
    data_lines: list[str] = []

    while True:
        now = loop.time()
        duration_started_at = (
            event_started_at if event_started_at is not None else pending_line_started_at
        )
        event_remaining: float | None = None
        if duration_started_at is not None:
            event_remaining = max_event_duration_s - (now - duration_started_at)
            if event_remaining <= 0:
                raise SseEventTimeoutError(event_timeout_message)
        read = deadline_controller.wait_for(
            iterator.__anext__(),
            kinds=(
                ProviderDeadlineKind.TRANSPORT_IDLE,
                ProviderDeadlineKind.PROTOCOL_IDLE,
                ProviderDeadlineKind.ABSOLUTE,
            ),
        )
        deadline = asyncio.timeout(event_remaining) if event_remaining is not None else None
        try:
            if deadline is None:
                line = await read
            else:
                async with deadline:
                    line = await read
        except StopAsyncIteration:
            break
        except TimeoutError:
            if deadline is None or not deadline.expired():
                raise
            raise SseEventTimeoutError(event_timeout_message) from None

        received_at = loop.time()

        if isinstance(line, _SseByteActivity):
            if (
                line.line_in_progress
                and event_started_at is None
                and pending_line_started_at is None
            ):
                pending_line_started_at = received_at
            continue

        if line.startswith(":"):
            # Comments are heartbeats, not event fields. If an event is already
            # open they keep the connection active without extending its
            # independent event-duration deadline.
            if event_started_at is None:
                pending_line_started_at = None
            continue
        if line == "":
            if event_started_at is None:
                pending_line_started_at = None
                continue
            data = "\n".join(data_lines)
            data_lines = []
            event_started_at = None
            pending_line_started_at = None
            event_bytes = 0
            event_lines = 0
            if not has_data:
                continue
            has_data = False
            if data == "[DONE]":
                deadline_controller.observe_protocol()
                break
            decoded = decode(data)
            deadline_controller.observe_protocol()
            pause_started = deadline_controller.idle_pause_started()
            yield decoded
            deadline_controller.exclude_idle_pause(
                pause_started,
                kinds=(
                    ProviderDeadlineKind.TRANSPORT_IDLE,
                    ProviderDeadlineKind.PROTOCOL_IDLE,
                ),
            )
            continue
        if event_started_at is None:
            event_started_at = (
                pending_line_started_at if pending_line_started_at is not None else received_at
            )
        pending_line_started_at = None
        event_lines += 1
        if event_lines > max_event_lines:
            raise SseEventLimitError(event_lines_message)
        event_bytes += len(line.encode("utf-8")) + (1 if event_lines > 1 else 0)
        if event_bytes > max_event_bytes:
            raise SseEventLimitError(event_bytes_message)
        if line.startswith("data:"):
            data = line[5:]
            if data.startswith(" "):
                data = data[1:]
            has_data = True
            data_lines.append(data)
            continue

    if data_lines:
        data = "\n".join(data_lines)
        if data != "[DONE]":
            decoded = decode(data)
            deadline_controller.observe_protocol()
            pause_started = deadline_controller.idle_pause_started()
            yield decoded
            deadline_controller.exclude_idle_pause(
                pause_started,
                kinds=(
                    ProviderDeadlineKind.TRANSPORT_IDLE,
                    ProviderDeadlineKind.PROTOCOL_IDLE,
                ),
            )


__all__ = [
    "DEFAULT_SSE_EVENT_DURATION_MULTIPLIER",
    "DEFAULT_SSE_MAX_EVENT_BYTES",
    "DEFAULT_SSE_MAX_EVENT_LINES",
    "SseEventLimitError",
    "SseEventTimeoutError",
    "aiter_sse_json_events",
]

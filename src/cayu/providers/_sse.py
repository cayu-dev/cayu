"""Shared SSE-to-JSON parsing for streaming provider transports.

Providers that stream over Server-Sent Events (OpenAI Responses, Chat
Completions) decode through this single parser, so heartbeat handling and the
idle-timeout timer cannot drift between adapters.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Mapping
from typing import Any


async def aiter_sse_json_events(
    lines: AsyncIterator[str],
    *,
    idle_timeout_s: float,
    provider_label: str,
    protocol_error: type[Exception],
) -> AsyncIterator[Mapping[str, Any]]:
    """Decode an SSE line stream into JSON data objects.

    SSE comment lines (starting with ``:``) are keep-alive heartbeats and count
    as stream activity, so a slow-but-alive stream is not killed by the idle
    timeout. Error messages are prefixed with ``provider_label``; malformed SSE
    data raises ``protocol_error``.
    """
    if idle_timeout_s <= 0:
        raise ValueError("idle_timeout_s must be greater than zero.")
    idle_message = (
        f"{provider_label} streaming response produced no SSE events "
        f"for {idle_timeout_s:g} seconds."
    )

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
    last_event_at = loop.time()
    data_lines: list[str] = []

    while True:
        elapsed = loop.time() - last_event_at
        remaining = idle_timeout_s - elapsed
        if remaining <= 0:
            raise TimeoutError(idle_message)
        try:
            line = await asyncio.wait_for(iterator.__anext__(), timeout=remaining)
        except StopAsyncIteration:
            break
        except TimeoutError:
            raise TimeoutError(idle_message) from None

        if line.startswith(":"):
            # SSE comment / keep-alive heartbeat: count it as stream activity so a
            # slow-but-alive stream is not killed by the idle timeout.
            last_event_at = loop.time()
            continue
        if line == "":
            if not data_lines:
                continue
            data = "\n".join(data_lines)
            data_lines = []
            if data == "[DONE]":
                break
            decoded = decode(data)
            last_event_at = loop.time()
            yield decoded
            continue
        if line.startswith("data:"):
            data = line[5:]
            if data.startswith(" "):
                data = data[1:]
            data_lines.append(data)
            continue

    if data_lines:
        data = "\n".join(data_lines)
        if data != "[DONE]":
            yield decode(data)


__all__ = ["aiter_sse_json_events"]

"""Network-free check of the request prefixes both providers would cache.

Real ``OpenAIProvider`` and ``AnthropicProvider`` instances serialize every turn
through recording transports. The check asserts the serialized prefix of the
late-system variant is append-only across turns and that the default Anthropic
cache markers sit on the stable history. Token counts are simulated from the
serialized payloads (about four characters per token) with each provider's
documented prefix rule, and are labeled as such.
"""

from __future__ import annotations

import itertools
import json
from collections.abc import AsyncIterator, Mapping
from pathlib import Path
from typing import Any, cast

from examples._advanced_support import ScenarioResult
from examples.late_system_message_caching.scenario import (
    SCENARIO,
    VARIANTS,
    run_conversation,
    summarize,
)

from cayu import AnthropicProvider, OpenAIProvider

_CHARS_PER_TOKEN = 4
_OPENAI_MIN_CACHED_TOKENS = 1024
_OPENAI_CACHE_INCREMENT = 128


def _tokens(text: str) -> int:
    return len(text) // _CHARS_PER_TOKEN


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _without_markers(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _without_markers(item) for key, item in value.items() if key != "cache_control"
        }
    if isinstance(value, list):
        return [_without_markers(item) for item in value]
    return value


def _common_prefix(left: str, right: str) -> int:
    limit = min(len(left), len(right))
    index = 0
    while index < limit and left[index] == right[index]:
        index += 1
    return index


class _SimulatedResponses:
    """OpenAI Responses transport: automatic caching of the longest shared prefix."""

    def __init__(self) -> None:
        self.payloads: list[dict[str, Any]] = []
        self._seen: list[str] = []

    async def stream_response_events(
        self, *, payload: Mapping[str, Any], **_: Any
    ) -> AsyncIterator[Mapping[str, Any]]:
        self.payloads.append(dict(payload))
        # Cached prefix order: instructions, tools, then input items.
        prompt = _canonical(
            [payload.get("instructions"), payload.get("tools", []), *payload["input"]]
        )
        shared = max((_common_prefix(prompt, earlier) for earlier in self._seen), default=0)
        self._seen.append(prompt)
        shared_tokens = _tokens(prompt[:shared])
        cached = (
            shared_tokens // _OPENAI_CACHE_INCREMENT * _OPENAI_CACHE_INCREMENT
            if shared_tokens >= _OPENAI_MIN_CACHED_TOKENS
            else 0
        )
        usage = {
            "input_tokens": _tokens(prompt),
            "input_tokens_details": {"cached_tokens": cached},
            "output_tokens": 8,
            "total_tokens": _tokens(prompt) + 8,
        }
        yield {"type": "response.created", "response": {"id": "resp-simulated"}}
        yield {"type": "response.output_text.delta", "delta": "Simulated answer."}
        yield {
            "type": "response.completed",
            "response": {
                "id": "resp-simulated",
                "model": payload["model"],
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "id": f"msg-simulated-{len(self.payloads)}",
                        "status": "completed",
                        "role": "assistant",
                        "content": [
                            {"type": "output_text", "text": "Simulated answer.", "annotations": []}
                        ],
                    }
                ],
                "usage": usage,
            },
        }

    async def aclose(self) -> None:
        return None


# Anthropic checks each marker and up to 20 earlier block positions for an
# entry an earlier request wrote at one of its markers.
_ANTHROPIC_LOOKBACK_BLOCKS = 20


def _anthropic_blocks(payload: Mapping[str, Any]) -> list[Any]:
    """Prompt blocks in Anthropic's cache order: tools, system, then messages."""

    blocks: list[Any] = [*payload.get("tools", [])]
    system = payload.get("system")
    if isinstance(system, list):
        blocks.extend(system)
    elif system:
        blocks.append({"type": "text", "text": system})
    for message in payload["messages"]:
        for block in message["content"]:
            blocks.append({"role": message["role"], **block})
    return blocks


class _SimulatedMessages:
    """Anthropic Messages transport: only prefixes ending at a marker are cached."""

    def __init__(self) -> None:
        self.payloads: list[dict[str, Any]] = []
        self._written: set[str] = set()

    async def stream_message_events(
        self, *, payload: Mapping[str, Any], **_: Any
    ) -> AsyncIterator[Mapping[str, Any]]:
        self.payloads.append(dict(payload))
        blocks = _anthropic_blocks(payload)
        prefixes = [
            _canonical(_without_markers(blocks[: index + 1])) for index in range(len(blocks))
        ]
        markers = [index for index, block in enumerate(blocks) if "cache_control" in block]
        read = 0
        for marker in markers:
            for index in range(marker, max(-1, marker - _ANTHROPIC_LOOKBACK_BLOCKS), -1):
                if prefixes[index] in self._written:
                    read = max(read, len(prefixes[index]))
                    break
        written = max((len(prefixes[marker]) for marker in markers), default=0)
        self._written.update(prefixes[marker] for marker in markers)
        whole = prefixes[-1]
        read_tokens = _tokens(whole[:read])
        write_tokens = max(0, _tokens(whole[:written]) - read_tokens)
        uncached = max(0, _tokens(whole) - read_tokens - write_tokens)
        yield {
            "type": "message_start",
            "message": {
                "id": f"msg-simulated-{len(self.payloads)}",
                "model": payload["model"],
                "usage": {
                    "input_tokens": uncached,
                    "cache_read_input_tokens": read_tokens,
                    "cache_creation_input_tokens": write_tokens,
                },
            },
        }
        yield {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""},
        }
        yield {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "Simulated answer."},
        }
        yield {"type": "content_block_stop", "index": 0}
        yield {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn"},
            "usage": {"output_tokens": 8},
        }
        yield {"type": "message_stop"}

    async def aclose(self) -> None:
        return None


def _openai_prefix_is_append_only(payloads: list[dict[str, Any]]) -> bool:
    for previous, current in itertools.pairwise(payloads):
        previous_items = previous["input"][:-1]  # drop the changing developer item
        if (
            current.get("instructions") != previous.get("instructions")
            or current["input"][: len(previous_items)] != previous_items
            or previous["input"][-1].get("role") != "developer"
        ):
            return False
    return True


def _anthropic_prefix_is_append_only(payloads: list[dict[str, Any]]) -> bool:
    def blocks(payload: dict[str, Any]) -> list[Any]:
        return [
            (message["role"], _without_markers(block))
            for message in payload["messages"]
            for block in message["content"]
        ]

    for previous, current in itertools.pairwise(payloads):
        previous_blocks = blocks(previous)[:-1]  # drop the changing <system> note
        if (
            _without_markers(current.get("system")) != _without_markers(previous.get("system"))
            or blocks(current)[: len(previous_blocks)] != previous_blocks
        ):
            return False
    return True


def _anthropic_markers_on_stable_history(payloads: list[dict[str, Any]]) -> bool:
    for payload in payloads:
        system = payload.get("system")
        if not (isinstance(system, list) and "cache_control" in system[-1]):
            return False
        marked = [
            block
            for message in payload["messages"]
            for block in message["content"]
            if "cache_control" in block
        ]
        if len(marked) != 1 or "Controller state" in json.dumps(marked[0]):
            return False
        if any("cache_control" in block for block in payload["messages"][-1]["content"]):
            return False
    return True


async def run(root: Path) -> ScenarioResult:
    openai_transport = _SimulatedResponses()
    anthropic_transport = _SimulatedMessages()
    # The simulated transports implement only the streaming calls this fixture uses.
    openai = OpenAIProvider(api_key="simulated-key", transport=cast("Any", openai_transport))
    anthropic = AnthropicProvider(
        api_key="simulated-key", transport=cast("Any", anthropic_transport)
    )
    measurements: dict[str, Any] = {}
    late_payloads: dict[str, list[dict[str, Any]]] = {}
    for label, provider, transport, model in (
        ("openai", openai, openai_transport, "gpt-6-luna"),
        ("anthropic", anthropic, anthropic_transport, "claude-haiku-4-5"),
    ):
        runs = []
        for variant in VARIANTS:
            start = len(transport.payloads)
            runs.append(
                await run_conversation(
                    provider,
                    provider_name=label,
                    model=model,
                    variant=variant,
                    options={},
                    nonce=f"fixture-{variant}",
                )
            )
            if variant == "late_system":
                late_payloads[label] = transport.payloads[start:]
        measurements[label] = summarize(runs)

    late_cached = {
        label: summary["late_system"]["turns_2_plus_cached_share"]
        for label, summary in measurements.items()
    }
    assertions = {
        "openai_late_system_prefix_is_append_only": _openai_prefix_is_append_only(
            late_payloads["openai"]
        ),
        "openai_sends_one_prompt_cache_key": len(
            {payload.get("prompt_cache_key") for payload in late_payloads["openai"]}
        )
        == 1
        and late_payloads["openai"][0].get("prompt_cache_key") is not None,
        "anthropic_late_system_prefix_is_append_only": _anthropic_prefix_is_append_only(
            late_payloads["anthropic"]
        ),
        "anthropic_default_markers_on_stable_history": _anthropic_markers_on_stable_history(
            late_payloads["anthropic"]
        ),
        "simulated_late_system_turns_2_plus_mostly_cached": all(
            share >= 0.9 for share in late_cached.values()
        ),
    }
    result = ScenarioResult(
        scenario=SCENARIO,
        mode="deterministic",
        status="verified" if all(assertions.values()) else "failed",
        assertions=assertions,
        sessions=[],
        metrics={
            "token_counts": "simulated from serialized payloads; not provider-reported",
            "providers": measurements,
        },
    )
    result.write(root)
    result.require_verified()
    return result

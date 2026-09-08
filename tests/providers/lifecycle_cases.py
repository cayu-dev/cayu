"""Synthetic native frames for the registered public-provider lifecycle matrix."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Literal

Protocol = Literal["responses", "messages", "chat", "bedrock"]


def lifecycle_events(protocol: Protocol, scenario: str) -> list[dict[str, Any]] | None:
    if not scenario.startswith("lifecycle_"):
        return None
    start, text, terminal, tail, tool, reasoning = _frames(protocol)
    prefix = [start, *text]
    clean = [*prefix, *terminal]
    case = scenario.removeprefix("lifecycle_")
    if case == "clean":
        return clean
    if case == "omitted_terminal_identity":
        if protocol == "responses":
            terminal[-1]["response"].pop("id")
            terminal[-1]["response"].pop("model")
        elif protocol == "chat":
            terminal[-1].pop("id")
            terminal[-1].pop("model")
        return [*prefix, *terminal]
    if case == "missing_start":
        return [*text, *terminal]
    if case == "unfinished":
        return prefix
    if case == "terminal_before_content":
        return [*terminal, *text]
    if case == "start_after_terminal":
        return [*clean, deepcopy(start)]
    if case == "repeated_terminal":
        return [*clean, deepcopy(terminal[-1])]
    if case == "excess_terminal":
        return [*clean, deepcopy(terminal[-1]), deepcopy(terminal[-1])]
    if case == "conflicting_terminal":
        repeated = deepcopy(terminal[-1])
        if protocol == "chat":
            repeated["choices"][0]["finish_reason"] = "length"
        elif protocol == "bedrock":
            repeated["messageStop"]["stopReason"] = "max_tokens"
        elif protocol == "responses":
            repeated["type"] = "response.incomplete"
            repeated["response"]["status"] = "incomplete"
        return [*clean, repeated]
    if case == "conflicting_terminal_metadata":
        repeated = deepcopy(terminal[-1])
        if protocol == "chat":
            repeated["usage"] = {"prompt_tokens": 1, "completion_tokens": 99}
        elif protocol == "bedrock":
            repeated["messageStop"]["additionalModelResponseFields"] = {"changed": True}
        elif protocol == "responses":
            repeated["response"]["usage"]["output_tokens"] = 99
        return [*clean, repeated]
    if case == "postterminal_text":
        return [*clean, *deepcopy(text)]
    if case == "postterminal_tool":
        return [*clean, *tool]
    if case == "postterminal_reasoning":
        return [*clean, *reasoning]
    if case == "tail":
        return [*clean, *tail]
    if case == "excess_tail":
        return [*clean, *tail, *deepcopy(tail)]
    if case in {"response_conflict", "model_conflict", "invalid_response", "invalid_model"}:
        changed = deepcopy(start)
        key = "id" if case in {"response_conflict", "invalid_response"} else "model"
        value = True if case.startswith("invalid_") else "foreign"
        if protocol == "responses":
            changed["type"] = "response.in_progress"
            changed["response"][key] = value
        elif protocol == "messages":
            changed["message"][key] = value
        elif protocol == "chat":
            changed[key] = value
        else:
            raise AssertionError("Converse has no streamed response/model identity.")
        return [*prefix, changed, *terminal]
    raise AssertionError(f"Missing lifecycle fixture for {scenario}.")


def _frames(
    protocol: Protocol,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    if protocol == "responses":
        response = {"id": "response-lifecycle", "model": "gpt-conformance"}
        return (
            {"type": "response.created", "response": {**response, "status": "in_progress"}},
            [{"type": "response.output_text.delta", "delta": "hello"}],
            [
                {
                    "type": "response.completed",
                    "response": {
                        **response,
                        "status": "completed",
                        "output": [],
                        "usage": {"input_tokens": 1, "output_tokens": 1},
                    },
                }
            ],
            [
                {
                    "type": "response.in_progress",
                    "response": {**response, "usage": {"input_tokens": 1}},
                }
            ],
            [
                {
                    "type": "response.output_item.added",
                    "output_index": 0,
                    "item": {
                        "type": "function_call",
                        "id": "fc-lifecycle",
                        "call_id": "call-lifecycle",
                        "name": "unsafe",
                        "arguments": "",
                        "status": "in_progress",
                    },
                }
            ],
            [{"type": "response.reasoning_text.delta", "delta": "late-reasoning"}],
        )
    if protocol == "messages":
        return (
            {
                "type": "message_start",
                "message": {
                    "id": "message-lifecycle",
                    "model": "claude-conformance",
                    "usage": {"input_tokens": 1},
                },
            },
            [
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "text", "text": "hello"},
                },
                {"type": "content_block_stop", "index": 0},
            ],
            [
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "end_turn"},
                    "usage": {"output_tokens": 1},
                },
                {"type": "message_stop"},
            ],
            [{"type": "message_delta", "usage": {"output_tokens": 2}}],
            [
                {
                    "type": "content_block_start",
                    "index": 1,
                    "content_block": {"type": "tool_use", "id": "call-lifecycle", "name": "unsafe"},
                }
            ],
            [
                {
                    "type": "content_block_start",
                    "index": 1,
                    "content_block": {"type": "thinking", "thinking": "late-reasoning"},
                }
            ],
        )
    if protocol == "chat":

        def chunk(delta: dict[str, Any], finish: str | None = None) -> dict[str, Any]:
            return {
                "id": "chat-lifecycle",
                "model": "chat-conformance",
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
            }

        return (
            chunk({"role": "assistant"}),
            [chunk({"content": "hello"})],
            [chunk({}, "stop")],
            [
                {
                    "id": "chat-lifecycle",
                    "model": "chat-conformance",
                    "choices": [],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                }
            ],
            [
                chunk(
                    {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call-lifecycle",
                                "type": "function",
                                "function": {"name": "unsafe", "arguments": "{}"},
                            }
                        ]
                    }
                )
            ],
            [chunk({"reasoning_content": "late-reasoning"})],
        )
    return (
        {"messageStart": {"role": "assistant"}},
        [{"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": "hello"}}}],
        [{"messageStop": {"stopReason": "end_turn"}}],
        [{"metadata": {"usage": {"inputTokens": 1, "outputTokens": 1}}}],
        [
            {
                "contentBlockStart": {
                    "contentBlockIndex": 1,
                    "start": {"toolUse": {"toolUseId": "call-lifecycle", "name": "unsafe"}},
                }
            }
        ],
        [
            {
                "contentBlockDelta": {
                    "contentBlockIndex": 1,
                    "delta": {"reasoningContent": {"text": "late-reasoning"}},
                }
            }
        ],
    )

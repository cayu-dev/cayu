"""Live local context-pressure calibration.

OpenAI:
    CAYU_PROVIDER=openai PYTHONPATH=src .venv/bin/python examples/context_pressure_calibration_live.py
    CAYU_PROVIDER=openai CAYU_CONTEXT_CALIBRATION_MODE=large-text CAYU_APPROX_INPUT_TOKENS=20000 \
        PYTHONPATH=src .venv/bin/python examples/context_pressure_calibration_live.py

Anthropic:
    CAYU_PROVIDER=anthropic PYTHONPATH=src .venv/bin/python examples/context_pressure_calibration_live.py

The script builds one realistic model request with:
- prior user/assistant/tool messages
- a tool schema
- tool-call arguments
- a tool-result payload
- a small resolved image attachment
- provider options

It then compares Cayu's local context-pressure estimate with:
- official provider input-token count, when available
- actual provider-reported input tokens from the real model call
"""

from __future__ import annotations

import asyncio
import json
import os
from base64 import standard_b64encode
from collections.abc import AsyncIterator
from typing import Any

from cayu import (
    Message,
    ObservedDeltaContextEstimator,
    ToolSpec,
    estimate_model_request_context_pressure,
    file_attachment,
)
from cayu.artifacts import RESOLVED_FILE_ATTACHMENTS_OPTION
from cayu.providers import (
    AnthropicProvider,
    ModelProvider,
    ModelRequest,
    ModelStreamEvent,
    ModelStreamEventType,
    OpenAIProvider,
)

ATTACHMENT_ID = "calibration_document"
ATTACHMENT_BYTES = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
    b"\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4"
    b"\x89\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05"
    b"\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
)


async def main() -> None:
    provider_name = _provider_name()
    model = _model(provider_name)
    _require_api_key(provider_name)
    provider = _provider(provider_name)

    mode = _calibration_mode()
    chars_per_token = _chars_per_token()
    request = _model_request(model, mode=mode, chars_per_token=chars_per_token)
    estimator = ObservedDeltaContextEstimator(chars_per_token=chars_per_token)
    estimate = estimate_model_request_context_pressure(
        model_request=request,
        image_min_tokens=provider.context_pressure_profile.image_min_tokens,
        document_min_tokens=provider.context_pressure_profile.document_min_tokens,
        document_bytes_per_token=provider.context_pressure_profile.document_bytes_per_token,
        tool_schema_chars_per_token=provider.context_pressure_profile.tool_schema_chars_per_token,
        reserved_output_tokens=_reserved_output_tokens(),
        estimator=estimator,
    )

    print("provider", provider_name)
    print("model", model)
    print("mode", mode)
    _print_json("LOCAL_ESTIMATE", estimate.model_dump(mode="json"))

    official_count = await provider.count_input_tokens(request)
    if official_count is None:
        _print_json("OFFICIAL_COUNT", {"available": False})
    else:
        _print_json("OFFICIAL_COUNT", official_count.model_dump(mode="json"))

    actual = await _run_model(provider, request)
    _print_json("ACTUAL_USAGE", actual)

    actual_input_tokens = actual.get("input_tokens")
    if not isinstance(actual_input_tokens, int) or actual_input_tokens <= 0:
        raise SystemExit(f"provider did not report input tokens: {_json(actual)}")

    _print_json(
        "LOCAL_VS_ACTUAL",
        _comparison(
            estimated_input_tokens=estimate.estimated_context_input_tokens,
            actual_input_tokens=actual_input_tokens,
        ),
    )
    if official_count is not None and official_count.input_tokens is not None:
        _print_json(
            "OFFICIAL_VS_ACTUAL",
            _comparison(
                estimated_input_tokens=official_count.input_tokens,
                actual_input_tokens=actual_input_tokens,
            ),
        )
    print("status ok")


def _model_request(model: str, *, mode: str, chars_per_token: int) -> ModelRequest:
    if mode == "large-text":
        return _large_text_model_request(model, chars_per_token=chars_per_token)
    return _mixed_model_request(model)


def _mixed_model_request(model: str) -> ModelRequest:
    attachment = file_attachment(
        artifact_id=ATTACHMENT_ID,
        kind="image",
        filename="calibration-pixel.png",
        content_type="image/png",
        size_bytes=len(ATTACHMENT_BYTES),
        metadata={"source": "context-pressure-calibration"},
    )
    resolved_attachment = {
        "artifact_id": ATTACHMENT_ID,
        "kind": "image",
        "filename": "calibration-pixel.png",
        "content_type": "image/png",
        "data_base64": standard_b64encode(ATTACHMENT_BYTES).decode("ascii"),
        "metadata": {"source": "context-pressure-calibration"},
    }
    messages = [
        Message.text(
            "system",
            "You are a terse assistant. Answer from the supplied policy evidence only.",
        ),
        Message.text(
            "user",
            "We need to push code to GitHub from a remote sandbox. What is the safe auth boundary?",
        ),
        Message.tool_call(
            tool_call_id="call_policy_lookup",
            tool_name="lookup_security_policy",
            arguments={
                "query": "remote sandbox github push credential boundary",
                "namespace": "project:cayu",
                "labels": {"area": "sandbox-git", "project": "cayu"},
                "include_evidence": True,
            },
        ),
        Message.tool_result(
            tool_call_id="call_policy_lookup",
            tool_name="lookup_security_policy",
            content=(
                "Found policy entry remote_git_credentials. It recommends a brokered "
                "Git HTTP proxy and forbids raw GitHub tokens inside sandbox process "
                "state or files."
            ),
            structured={
                "entry_id": "remote_git_credentials",
                "confidence": "high",
                "labels": {"area": "sandbox-git", "project": "cayu"},
                "evidence": [
                    "brokered Git HTTP proxy",
                    "credential injected outside sandbox",
                    "no token in env/files/argv/output",
                ],
            },
            artifacts=[attachment],
        ),
        Message.text(
            "user",
            "Give the recommendation in one sentence.",
        ),
    ]
    tools = [
        ToolSpec(
            name="lookup_security_policy",
            description=(
                "Search scoped security policy knowledge. Use this for credential "
                "boundary and sandbox authorization guidance."
            ),
            input_schema={
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language search query.",
                    },
                    "namespace": {
                        "type": "string",
                        "description": "Knowledge namespace.",
                    },
                    "labels": {
                        "type": "object",
                        "additionalProperties": {"type": "string"},
                    },
                    "include_evidence": {
                        "type": "boolean",
                    },
                },
                "required": ["query", "namespace"],
            },
        ).model_dump(mode="json")
    ]
    return ModelRequest(
        model=model,
        messages=messages,
        tools=tools,
        options={
            "max_output_tokens": 80,
            "temperature": 0,
            RESOLVED_FILE_ATTACHMENTS_OPTION: {
                ATTACHMENT_ID: resolved_attachment,
            },
        },
    )


def _large_text_model_request(model: str, *, chars_per_token: int) -> ModelRequest:
    approx_tokens = _approx_input_tokens()
    calibration_text = _large_calibration_text(
        approx_tokens,
        chars_per_token=chars_per_token,
    )
    messages = [
        Message.text(
            "system",
            "You are a calibration assistant. Return only the requested marker.",
        ),
        Message.text(
            "user",
            (
                "The following policy corpus is for context-window calibration. "
                "Do not summarize it. After reading it, answer exactly: calibration-ok.\n\n"
                f"{calibration_text}"
            ),
        ),
    ]
    return ModelRequest(
        model=model,
        messages=messages,
        tools=[],
        options={
            "max_output_tokens": 8,
            "temperature": 0,
        },
    )


def _large_calibration_text(approx_tokens: int, *, chars_per_token: int) -> str:
    sentence = (
        "Remote sandbox Git authentication should use a brokered Git HTTP proxy; "
        "the trusted Cayu side injects credentials outside the sandbox boundary, "
        "and raw GitHub tokens must not appear in environment variables, files, "
        "process arguments, or command output. "
    )
    # Build slightly above the requested size so fixed prompt overhead does not
    # dominate the run.
    target_chars = approx_tokens * chars_per_token
    repetitions = max(1, (target_chars // len(sentence)) + 1)
    return sentence * repetitions


async def _run_model(provider: ModelProvider, request: ModelRequest) -> dict[str, int]:
    completed_payload: dict[str, Any] | None = None
    async for event in _model_events(provider, request):
        if event.type == ModelStreamEventType.ERROR:
            raise SystemExit(f"model error: {_json(event.payload)}")
        if event.type == ModelStreamEventType.COMPLETED:
            completed_payload = event.payload
    if completed_payload is None:
        raise SystemExit("provider stream ended without completion")
    usage = completed_payload.get("usage")
    if not isinstance(usage, dict):
        raise SystemExit(f"completion did not include usage: {_json(completed_payload)}")
    input_tokens = usage.get("input_tokens")
    output_tokens = usage.get("output_tokens")
    total_tokens = usage.get("total_tokens")
    result: dict[str, int] = {}
    if isinstance(input_tokens, int):
        result["input_tokens"] = input_tokens
    if isinstance(output_tokens, int):
        result["output_tokens"] = output_tokens
    if isinstance(total_tokens, int):
        result["total_tokens"] = total_tokens
    return result


async def _model_events(
    provider: ModelProvider,
    request: ModelRequest,
) -> AsyncIterator[ModelStreamEvent]:
    async for event in provider.stream(request):
        yield event


def _comparison(
    *,
    estimated_input_tokens: int,
    actual_input_tokens: int,
) -> dict[str, float | int]:
    delta_tokens = actual_input_tokens - estimated_input_tokens
    return {
        "estimated_input_tokens": estimated_input_tokens,
        "actual_input_tokens": actual_input_tokens,
        "delta_tokens": delta_tokens,
        "relative_error": delta_tokens / actual_input_tokens,
        "absolute_relative_error": abs(delta_tokens) / actual_input_tokens,
    }


def _reserved_output_tokens() -> int:
    value = os.environ.get("CAYU_RESERVED_OUTPUT_TOKENS", "80")
    try:
        parsed = int(value)
    except ValueError as exc:
        raise SystemExit("CAYU_RESERVED_OUTPUT_TOKENS must be an integer.") from exc
    if parsed < 0:
        raise SystemExit("CAYU_RESERVED_OUTPUT_TOKENS must be non-negative.")
    return parsed


def _chars_per_token() -> int:
    value = os.environ.get("CAYU_ESTIMATE_CHARS_PER_TOKEN", "5")
    try:
        parsed = int(value)
    except ValueError as exc:
        raise SystemExit("CAYU_ESTIMATE_CHARS_PER_TOKEN must be an integer.") from exc
    if parsed < 1:
        raise SystemExit("CAYU_ESTIMATE_CHARS_PER_TOKEN must be positive.")
    return parsed


def _calibration_mode() -> str:
    mode = os.environ.get("CAYU_CONTEXT_CALIBRATION_MODE", "mixed").strip().lower()
    if mode in {"mixed", "large-text"}:
        return mode
    raise SystemExit("CAYU_CONTEXT_CALIBRATION_MODE must be mixed or large-text.")


def _approx_input_tokens() -> int:
    value = os.environ.get("CAYU_APPROX_INPUT_TOKENS", "20000")
    try:
        parsed = int(value)
    except ValueError as exc:
        raise SystemExit("CAYU_APPROX_INPUT_TOKENS must be an integer.") from exc
    if parsed < 1000:
        raise SystemExit("CAYU_APPROX_INPUT_TOKENS must be at least 1000.")
    return parsed


def _provider_name() -> str:
    requested = os.environ.get("CAYU_PROVIDER")
    if requested is not None:
        requested = requested.strip().lower()
        if requested in {"openai", "anthropic"}:
            return requested
        raise SystemExit("CAYU_PROVIDER must be openai or anthropic.")
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    return "openai"


def _model(provider_name: str) -> str:
    if provider_name == "openai":
        return os.environ.get("CAYU_OPENAI_MODEL", "gpt-5.5")
    return os.environ.get("CAYU_ANTHROPIC_MODEL", "claude-sonnet-4-6")


def _provider(provider_name: str) -> ModelProvider:
    if provider_name == "openai":
        return OpenAIProvider()
    return AnthropicProvider()


def _require_api_key(provider_name: str) -> None:
    if provider_name == "openai" and not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("Set OPENAI_API_KEY or choose CAYU_PROVIDER=anthropic.")
    if provider_name == "anthropic" and not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("Set ANTHROPIC_API_KEY or choose CAYU_PROVIDER=openai.")


def _print_json(label: str, payload: object) -> None:
    print(label, _json(payload))


def _json(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, default=str)


if __name__ == "__main__":
    asyncio.run(main())

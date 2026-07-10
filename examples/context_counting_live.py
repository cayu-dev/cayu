"""Demo-only live provider-token-counting smoke test.

OpenAI:
    CAYU_PROVIDER=openai PYTHONPATH=src .venv/bin/python examples/context_counting_live.py

Anthropic:
    CAYU_PROVIDER=anthropic PYTHONPATH=src .venv/bin/python examples/context_counting_live.py
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any

from cayu import (
    AgentSpec,
    CayuApp,
    ContextCountingConfig,
    ContextCountingMode,
    Event,
    EventType,
    Message,
    RunRequest,
)
from cayu.providers import (
    AnthropicProvider,
    InputTokenCountConfidence,
    InputTokenCountMethod,
    InputTokenCountResult,
    ModelProvider,
    ModelRequest,
    OpenAIProvider,
)

PROMPT = "Reply with exactly this sentence and no extra text: token counting live check"


async def main() -> None:
    provider_name = _provider_name()
    model = _model(provider_name)
    _require_api_key(provider_name)

    provider = _provider(provider_name)

    print("provider", provider_name)
    print("model", model)

    direct_count = await _run_direct_count(provider, model)
    _print_json("DIRECT_COUNT", direct_count.model_dump(mode="json"))
    _validate_direct_count(direct_count)

    app = CayuApp(
        context_counting=ContextCountingConfig(mode=ContextCountingMode.OBSERVE),
        enable_logging=False,
    )
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model=model))

    counted: dict[str, Any] | None = None
    reconciled: dict[str, Any] | None = None
    completed: dict[str, Any] | None = None

    request = RunRequest(
        agent_name="assistant",
        session_id=f"demo_{provider_name}_context_counting_live",
        max_steps=1,
        messages=[Message.text("user", PROMPT)],
    )

    async for event in app.run(request):
        if event.type in _INTERESTING_EVENTS:
            _print_event(event)
        if event.type == EventType.CONTEXT_COUNTED:
            counted = event.payload
        elif event.type == EventType.CONTEXT_COUNT_RECONCILED:
            reconciled = event.payload
        elif event.type == EventType.MODEL_COMPLETED:
            completed = event.payload
        elif event.type == EventType.CONTEXT_COUNT_FAILED:
            raise SystemExit(f"context count failed: {_json(event.payload)}")
        elif event.type == EventType.MODEL_ERROR:
            raise SystemExit(f"model call failed: {_json(event.payload)}")

    _validate_runtime_events(
        counted=counted,
        reconciled=reconciled,
        completed=completed,
    )
    print("status ok")


async def _run_direct_count(
    provider: ModelProvider,
    model: str,
) -> InputTokenCountResult:
    request = ModelRequest(
        model=model,
        messages=[Message.text("user", PROMPT)],
    )
    result = await provider.count_input_tokens(request)
    if result is None:
        raise SystemExit("provider returned no input-token count")
    return result


def _validate_direct_count(result: InputTokenCountResult) -> None:
    if result.method != InputTokenCountMethod.OFFICIAL:
        raise SystemExit(f"expected official count method, got {result.method!s}")
    if result.confidence != InputTokenCountConfidence.HIGH:
        raise SystemExit(f"expected high count confidence, got {result.confidence!s}")
    if not isinstance(result.input_tokens, int) or result.input_tokens <= 0:
        raise SystemExit(f"expected positive input_tokens, got {result.input_tokens!r}")
    endpoint = result.metadata.get("endpoint")
    if endpoint not in {"responses/input_tokens", "messages/count_tokens"}:
        raise SystemExit(f"unexpected count endpoint metadata: {endpoint!r}")


def _validate_runtime_events(
    *,
    counted: dict[str, Any] | None,
    reconciled: dict[str, Any] | None,
    completed: dict[str, Any] | None,
) -> None:
    if counted is None:
        raise SystemExit("runtime did not emit context.counted")
    if completed is None:
        raise SystemExit("runtime did not emit model.completed")
    if reconciled is None:
        raise SystemExit("runtime did not emit context.count.reconciled")

    count = counted.get("count")
    if not isinstance(count, dict):
        raise SystemExit(f"context.counted has invalid count payload: {_json(counted)}")
    if count.get("method") != "official":
        raise SystemExit(f"runtime count method is not official: {_json(count)}")
    if count.get("confidence") != "high":
        raise SystemExit(f"runtime count confidence is not high: {_json(count)}")
    if not isinstance(count.get("input_tokens"), int) or count["input_tokens"] <= 0:
        raise SystemExit(f"runtime count input_tokens is invalid: {_json(count)}")

    if reconciled.get("observation_id") != counted.get("observation_id"):
        raise SystemExit("reconciled event did not match counted observation_id")
    if reconciled.get("reconciled") is not True:
        raise SystemExit(f"context count did not reconcile: {_json(reconciled)}")
    if not isinstance(reconciled.get("actual_input_tokens"), int):
        raise SystemExit(f"missing actual input token usage: {_json(reconciled)}")
    if not isinstance(reconciled.get("delta_tokens"), int):
        raise SystemExit(f"missing token delta: {_json(reconciled)}")


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


_INTERESTING_EVENTS = {
    EventType.CONTEXT_COUNTED,
    EventType.CONTEXT_COUNT_FAILED,
    EventType.CONTEXT_COUNT_RECONCILED,
    EventType.MODEL_STARTED,
    EventType.MODEL_COMPLETED,
    EventType.MODEL_ERROR,
    EventType.SESSION_COMPLETED,
    EventType.SESSION_FAILED,
}


def _print_event(event: Event) -> None:
    _print_json(str(event.type), event.payload)


def _print_json(label: str, payload: object) -> None:
    print(label, _json(payload))


def _json(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, default=str)


if __name__ == "__main__":
    asyncio.run(main())

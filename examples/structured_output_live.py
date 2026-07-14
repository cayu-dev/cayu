"""Verified live structured-output provider contract."""

from __future__ import annotations

import asyncio
import json
import os
import re

from _live_checks import require, require_positive_model_usage, require_successful_terminal
from cayu import (
    AgentSpec,
    AnthropicProvider,
    CayuApp,
    Event,
    EventType,
    Message,
    OpenAIProvider,
    RunRequest,
    StructuredOutputSpec,
    StructuredOutputStrategy,
)
from cayu.providers import ModelProvider

EVIDENCE_PREFIX = "CAYU_NIGHTLY_EVIDENCE="

INVOICE_SCHEMA = {
    "type": "object",
    "properties": {
        "invoice": {
            "type": "object",
            "properties": {
                "number": {"type": "string"},
                "vendor": {"type": "string"},
                "status": {"type": "string", "enum": ["paid", "unpaid", "unknown"]},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "line_items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "description": {
                                "type": "string",
                                "enum": ["Managed hosting"],
                            },
                            "amount": {"type": "number"},
                        },
                        "required": ["description", "amount"],
                        "additionalProperties": False,
                    },
                    "minItems": 1,
                },
            },
            "required": ["number", "vendor", "status", "confidence", "line_items"],
            "additionalProperties": False,
        }
    },
    "required": ["invoice"],
    "additionalProperties": False,
}

EXPECTED_OUTPUT = {
    "invoice": {
        "number": "INV-1042",
        "vendor": "Acme Cloud",
        "status": "paid",
        "confidence": 0.91,
        "line_items": [{"description": "Managed hosting", "amount": 125.50}],
    }
}


async def main() -> None:
    provider_name = _provider_name()
    model = _model(provider_name)
    strategy = _strategy()
    if strategy == StructuredOutputStrategy.NATIVE and provider_name != "openai":
        raise RuntimeError("Native structured output in this example currently requires OpenAI.")
    _require_api_key(provider_name)

    print("provider", provider_name)
    print("model", model)
    print("strategy", strategy.value)

    evidence = await _run_contract(
        provider=_provider(provider_name),
        provider_name=provider_name,
        model=model,
        strategy=strategy,
    )
    print(EVIDENCE_PREFIX + json.dumps(evidence, sort_keys=True))
    print("status ok")


async def _run_contract(
    *,
    provider: ModelProvider,
    provider_name: str,
    model: str,
    strategy: StructuredOutputStrategy,
) -> dict[str, object]:
    app = CayuApp(enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(
            name="assistant",
            model=model,
            system_prompt=_system_prompt(strategy),
        )
    )

    request = RunRequest(
        agent_name="assistant",
        session_id=f"contract_{provider_name}_structured_output",
        max_steps=3,
        messages=[
            Message.text(
                "user",
                (
                    "Invoice INV-1042 from Acme Cloud is marked PAID. "
                    "It has one line item: Managed hosting for 125.50 USD. "
                    "Use the exact line-item description 'Managed hosting'. "
                    "Return the invoice status with confidence 0.91."
                ),
            )
        ],
        structured_output=StructuredOutputSpec(
            name="invoice_status",
            json_schema=INVOICE_SCHEMA,
            max_retries=2,
            strategy=strategy,
            repair_prompt=(
                "Call the structured-output tool again with an `output` object "
                "that exactly matches the invoice schema."
                if strategy == StructuredOutputStrategy.TOOL
                else "Return only valid JSON that exactly matches the invoice schema."
            ),
        ),
    )

    events: list[Event] = []
    async for event in app.run(request):
        events.append(event)
        print(
            event.type,
            event.environment_name or "-",
            event.tool_name or "-",
            event.payload,
        )
    return _validate_runtime_events(
        events,
        provider_name=provider_name,
        model=model,
        strategy=strategy,
    )


def _validate_runtime_events(
    events: list[Event],
    *,
    provider_name: str,
    model: str,
    strategy: StructuredOutputStrategy,
) -> dict[str, object]:
    require_successful_terminal(events)
    completed_events = require_positive_model_usage(events)
    validated_events = [
        event for event in events if event.type == EventType.STRUCTURED_OUTPUT_VALIDATED
    ]
    require(
        len(validated_events) == 1,
        f"expected one structured_output.validated event, got {len(validated_events)}",
    )
    require(
        validated_events[0].payload.get("output") == EXPECTED_OUTPUT,
        "validated invoice output did not match the requested invoice facts: "
        f"{validated_events[0].payload.get('output')!r}",
    )

    total_tokens = 0
    resolved_models: set[str] = set()
    for event in completed_events:
        usage = event.payload["usage_metrics"]
        require(
            usage.get("provider_name") == provider_name,
            "completed usage provider did not match the configured provider",
        )
        require(
            usage.get("requested_model") == model,
            "completed usage model did not match the requested model",
        )
        resolved_model = usage.get("model")
        require(
            isinstance(resolved_model, str) and bool(resolved_model.strip()),
            f"completed usage did not report a resolved model: {resolved_model!r}",
        )
        require(
            _resolved_model_matches(model, resolved_model),
            f"provider resolved model {resolved_model!r} does not match requested model {model!r}",
        )
        resolved_models.add(resolved_model)
        total_tokens += usage["total_tokens"]
    require(
        len(resolved_models) == 1,
        f"provider reported inconsistent resolved models: {sorted(resolved_models)!r}",
    )

    return {
        "provider": provider_name,
        "model": model,
        "resolved_model": next(iter(resolved_models)),
        "strategy": strategy.value,
        "invoice_number": EXPECTED_OUTPUT["invoice"]["number"],
        "invoice_status": EXPECTED_OUTPUT["invoice"]["status"],
        "total_tokens": total_tokens,
    }


def _resolved_model_matches(requested_model: str, resolved_model: str) -> bool:
    if resolved_model == requested_model:
        return True
    versioned_model = rf"{re.escape(requested_model)}-(?:\d{{8}}|\d{{4}}-\d{{2}}-\d{{2}})"
    return re.fullmatch(versioned_model, resolved_model) is not None


def _provider_name() -> str:
    requested = os.environ.get("CAYU_PROVIDER")
    if requested is not None:
        requested = requested.strip().lower()
        if requested in {"openai", "anthropic"}:
            return requested
        raise RuntimeError("CAYU_PROVIDER must be openai or anthropic.")
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    return "openai"


def _model(provider_name: str) -> str:
    if provider_name == "openai":
        return os.environ.get("CAYU_OPENAI_MODEL", "gpt-5.6")
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


def _strategy() -> StructuredOutputStrategy:
    strategy = os.environ.get("CAYU_STRUCTURED_OUTPUT_STRATEGY", "tool").strip().lower()
    if strategy not in {"tool", "native"}:
        raise RuntimeError("CAYU_STRUCTURED_OUTPUT_STRATEGY must be tool or native.")
    return StructuredOutputStrategy(strategy)


def _system_prompt(strategy: StructuredOutputStrategy) -> str:
    if strategy == StructuredOutputStrategy.NATIVE:
        return "Extract invoice facts."
    return (
        "Extract invoice facts. Use the structured-output tool when the final answer "
        "is ready. Do not return final JSON as plain text."
    )


if __name__ == "__main__":
    asyncio.run(main())

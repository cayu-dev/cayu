from __future__ import annotations

import os
from collections.abc import AsyncIterator
from typing import Any

from examples._advanced_support.results import SessionEvidence

from cayu import (
    AnthropicProvider,
    CayuApp,
    ChatCompletionsProvider,
    Event,
    EventType,
    ForkSessionRequest,
    OpenAIProvider,
    RunLimits,
    SessionStatus,
    StructuredOutputSpec,
    session_usage_summary,
    usage_metrics_from_event_payload,
)
from cayu.providers import ModelProvider, ModelStreamEvent
from cayu.runtime.structured_output import STRUCTURED_OUTPUT_TOOL_NAME

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"


def advanced_run_limits() -> RunLimits:
    return RunLimits(
        max_total_tokens=50_000,
        max_tool_calls=12,
        max_elapsed_seconds=180,
        scope="run",
    )


def stable_output_spec(name: str, schema: dict[str, Any]) -> StructuredOutputSpec:
    return StructuredOutputSpec(
        name=name,
        json_schema=schema,
        max_retries=2,
        repair_prompt=(
            "Call the structured-output tool exactly once and satisfy the supplied schema. "
            "Keep every field concise and do not answer in plain text."
        ),
    )


def completed_batch(
    text: str,
    *,
    input_tokens: int = 20,
    output_tokens: int = 5,
) -> list[ModelStreamEvent]:
    return [
        ModelStreamEvent.text_delta(text),
        ModelStreamEvent.completed(
            {
                "finish_reason": "stop",
                "usage": {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": input_tokens + output_tokens,
                },
            }
        ),
    ]


def structured_batch(
    output: dict[str, Any],
    *,
    call_id: str,
    input_tokens: int = 30,
    output_tokens: int = 10,
) -> list[ModelStreamEvent]:
    return [
        ModelStreamEvent.tool_call(
            id=call_id,
            name=STRUCTURED_OUTPUT_TOOL_NAME,
            arguments={"output": output},
        ),
        ModelStreamEvent.completed(
            {
                "finish_reason": "tool_calls",
                "usage": {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": input_tokens + output_tokens,
                },
            }
        ),
    ]


async def collect_events(events: AsyncIterator[Event]) -> list[Event]:
    return [event async for event in events]


def _runtime_failure_summary(events: list[Event]) -> list[dict[str, Any]]:
    failure_types = {
        EventType.MODEL_ERROR,
        EventType.CONTEXT_COMPACTION_FAILED,
        EventType.SESSION_FAILED,
    }
    diagnostic_keys = (
        "error_type",
        "error",
        "provider_error_type",
        "status_code",
        "retryable",
        "purpose",
        "reason",
    )
    return [
        {
            "type": str(event.type),
            **{key: event.payload[key] for key in diagnostic_keys if key in event.payload},
        }
        for event in events
        if event.type in failure_types
    ][-5:]


async def count_model_completions(app: CayuApp, session_ids: list[str]) -> int:
    total = 0
    for session_id in session_ids:
        events = await app.session_store.load_events(session_id)
        total += sum(event.type == EventType.MODEL_COMPLETED for event in events)
    return total


async def session_evidence(
    app: CayuApp,
    roles: dict[str, str],
) -> list[SessionEvidence]:
    evidence: list[SessionEvidence] = []
    for session_id, role in roles.items():
        session = await app.session_store.load(session_id)
        if session is None:
            raise RuntimeError(f"Session disappeared: {session_id}")
        events = await app.session_store.load_events(session_id)
        if session.status != SessionStatus.COMPLETED:
            failures = _runtime_failure_summary(events)
            raise RuntimeError(
                f"Session {session_id} did not complete: {session.status}; failures={failures!r}"
            )
        usage = session_usage_summary(session_id, events)
        manual_recovery = any(event.payload.get("manual_recovery") is True for event in events)
        interrupted = any(event.type == EventType.SESSION_INTERRUPTED for event in events)
        receipt_ids = sorted(
            {
                value
                for event in events
                if event.type == EventType.TOOL_CALL_COMPLETED
                for value in [event.payload.get("idempotency_key")]
                if isinstance(value, str) and value
            }
        )
        taint_labels = session.metadata.get("taint_labels", [])
        if not isinstance(taint_labels, list):
            taint_labels = []
        evidence.append(
            SessionEvidence(
                session_id=session.id,
                role=role,
                parent_session_id=session.parent_session_id,
                causal_budget_id=session.causal_budget_id or session.id,
                status=session.status.value,
                usage={
                    "input_tokens": usage.usage.input_tokens,
                    "output_tokens": usage.usage.output_tokens,
                    "total_tokens": usage.usage.total_tokens,
                },
                model_steps=usage.model_steps,
                tool_calls=usage.tool_calls,
                recovery_state=(
                    "manually-reconciled"
                    if manual_recovery
                    else "resumed-after-interruption"
                    if interrupted
                    else "not-required"
                ),
                taint_labels=sorted(str(label) for label in taint_labels),
                compaction_count=sum(
                    event.type == EventType.CONTEXT_COMPACTION_COMPLETED for event in events
                ),
                receipt_ids=receipt_ids,
            )
        )
    return evidence


async def first_model_input_tokens(app: CayuApp, session_id: str) -> int:
    events = await app.session_store.load_events(session_id)
    first_completion = next(
        (event for event in events if event.type == EventType.MODEL_COMPLETED),
        None,
    )
    if first_completion is None:
        raise RuntimeError(f"Session {session_id} has no completed model attempt.")
    usage = usage_metrics_from_event_payload(first_completion.payload)
    if usage is None or usage.input_tokens <= 0:
        raise RuntimeError(f"Session {session_id} has no provider-reported input tokens.")
    return usage.input_tokens


async def fork_session(
    app: CayuApp,
    *,
    source_session_id: str,
    session_id: str,
    agent_name: str,
) -> Event:
    events = await collect_events(
        app.fork_session(
            ForkSessionRequest(
                source_session_id=source_session_id,
                session_id=session_id,
                agent_name=agent_name,
            )
        )
    )
    fork_events = [event for event in events if event.type == EventType.SESSION_FORKED]
    if len(fork_events) != 1:
        raise RuntimeError(f"Expected one session.forked event, got {events!r}")
    return fork_events[0]


def validated_output(events: list[Event]) -> dict[str, Any]:
    validated = [event for event in events if event.type == EventType.STRUCTURED_OUTPUT_VALIDATED]
    if len(validated) != 1:
        summary = [
            {
                "type": str(event.type),
                **{
                    key: event.payload[key]
                    for key in ("errors", "error", "reason")
                    if key in event.payload
                },
            }
            for event in events
            if event.type
            in {
                EventType.STRUCTURED_OUTPUT_FAILED,
                EventType.SESSION_FAILED,
                EventType.SESSION_INTERRUPTED,
            }
        ]
        raise RuntimeError(
            f"Expected one structured output, got {len(validated)}; terminal={summary!r}"
        )
    output = validated[0].payload.get("output")
    if not isinstance(output, dict):
        raise RuntimeError("Structured output was not an object.")
    return output


def live_provider(provider_name: str | None = None) -> tuple[ModelProvider, str]:
    selected = (provider_name or os.environ.get("CAYU_ADVANCED_PROVIDER", "gemini")).strip().lower()
    if selected == "gemini":
        if not os.environ.get("GEMINI_API_KEY"):
            raise RuntimeError("Set GEMINI_API_KEY to run the live Gemini examples.")
        return (
            ChatCompletionsProvider(
                name="google",
                api_key_env="GEMINI_API_KEY",
                base_url=GEMINI_BASE_URL,
                document_encoding="image_url",
                timeout_s=60,
                stream_idle_timeout_s=30,
            ),
            os.environ.get("CAYU_GEMINI_MODEL", "gemini-3.1-flash-lite"),
        )
    if selected == "openai":
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("Set OPENAI_API_KEY to run the live OpenAI examples.")
        return OpenAIProvider(), os.environ.get("CAYU_OPENAI_MODEL", "gpt-5.6-luna")
    if selected == "anthropic":
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError("Set ANTHROPIC_API_KEY to run the live Anthropic examples.")
        return AnthropicProvider(), os.environ.get("CAYU_ANTHROPIC_MODEL", "claude-sonnet-4-6")
    raise ValueError("Live provider must be gemini, openai, or anthropic.")

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from decimal import Decimal

import pytest

import cayu.runtime.context as runtime_context_module
from cayu.core import AgentSpec, Event, EventType, Message
from cayu.providers import ModelProvider, ModelRequest, ModelStreamEvent, UsageDialect
from cayu.runtime import (
    BudgetLimit,
    CayuApp,
    CheckpointCompactionContextPolicy,
    ModelCompactor,
    ModelPrice,
    PriceBook,
    RunRequest,
    estimate_session_cost,
)
from cayu.runtime.budgets import budget_actual_cost_for_event


class FakeProvider(ModelProvider):
    name = "fake"

    def __init__(
        self,
        events: list[ModelStreamEvent] | list[list[ModelStreamEvent]],
    ) -> None:
        if events and isinstance(events[0], list):
            self.event_batches = events  # type: ignore[assignment]
        else:
            self.event_batches = [events]  # type: ignore[list-item]
        self.requests: list[ModelRequest] = []

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        self.requests.append(request)
        batch_index = len(self.requests) - 1
        if batch_index >= len(self.event_batches):
            raise AssertionError(f"No fake provider event batch for request {batch_index}")
        for event in self.event_batches[batch_index]:
            yield event


async def collect_events(app: CayuApp, request: RunRequest) -> list[Event]:
    return [event async for event in app.run(request)]


def test_compaction_provider_metadata_strips_runtime_owned_execution_identity() -> None:
    provider_identity = {
        "model_step_id": f"mstep_{'a' * 32}",
        "model_attempt_id": f"matt_{'b' * 32}",
        "tool_round_id": f"tround_{'c' * 32}",
        "budget_limit_id": f"blim_{'d' * 64}",
        "reservation_id": f"bres_{'e' * 32}",
    }

    sanitized = runtime_context_module._provider_completed_metadata(
        {
            **provider_identity,
            "usage": {"input_tokens": 8, "output_tokens": 2},
        }
    )

    assert sanitized == {"usage": {"input_tokens": 8, "output_tokens": 2}}


def test_compaction_preserves_normalized_usage_when_raw_usage_is_null() -> None:
    payload = runtime_context_module._compaction_model_completed_payload(
        completed_payload={
            "usage": None,
            "usage_metrics": {
                "provider_name": "normalized-provider",
                "model": "summary-model",
                "input_tokens": 5,
                "output_tokens": 1,
                "total_tokens": 6,
            },
        },
        provider_name="normalized-provider",
        fallback_model="summary-model",
        compactor="ModelCompactor",
    )

    assert payload["usage_metrics"]["input_tokens"] == 5
    assert "usage_normalization_failed" not in payload


@pytest.mark.parametrize(
    "nullable_usage",
    [
        {"input_tokens_details": None},
        {"prompt_tokens_details": None},
        {"output_tokens_details": None},
        {"completion_tokens_details": None},
        {"cache_creation": None},
        {"input_tokens_details": {"cached_tokens": None}},
        {"prompt_tokens_details": {"cached_tokens": None}},
        {"output_tokens_details": {"reasoning_tokens": None}},
        {"completion_tokens_details": {"reasoning_tokens": None}},
        {"output_tokens_details": {"thinking_tokens": None}},
        {"completion_tokens_details": {"thinking_tokens": None}},
    ],
    ids=(
        "input-details",
        "prompt-details",
        "output-details",
        "completion-details",
        "cache-creation",
        "input-cached-counter",
        "prompt-cached-counter",
        "output-reasoning-counter",
        "completion-reasoning-counter",
        "output-thinking-counter",
        "completion-thinking-counter",
    ),
)
def test_compaction_nullable_usage_details_remain_priced_after_sanitization(
    nullable_usage: dict[str, object],
) -> None:
    payload = runtime_context_module._compaction_model_completed_payload(
        completed_payload={
            "model": "summary-model",
            "usage": {
                "input_tokens": 100,
                "output_tokens": 20,
                **nullable_usage,
            },
        },
        provider_name="renamed-openai",
        fallback_model="summary-model",
        compactor="ModelCompactor",
        usage_dialect=UsageDialect.OPENAI,
    )
    sanitized = runtime_context_module.sanitize_context_compaction_telemetry(
        runtime_context_module.ContextCompactionTelemetry(
            event_type=EventType.MODEL_COMPLETED,
            payload=payload,
        )
    )
    event = Event(
        type=EventType.MODEL_COMPLETED,
        session_id="compaction_nullable_usage",
        payload=sanitized.payload,
    )
    pricing = PriceBook(
        prices=(
            ModelPrice.fixed(
                provider_name="renamed-openai",
                model="summary-model",
                input_per_million=Decimal("10"),
                output_per_million=Decimal("20"),
            ),
        )
    )
    cost = estimate_session_cost(
        session_id=event.session_id,
        events=[event],
        pricing=pricing,
    )
    actual = budget_actual_cost_for_event(
        limit=BudgetLimit(
            scope="app",
            max_estimated_cost=Decimal("1"),
            pricing=pricing,
        ),
        event=event,
    )

    assert sanitized.payload["usage_metrics"]["input_tokens"] == 100
    assert sanitized.payload["usage_metrics"]["output_tokens"] == 20
    assert sanitized.payload["usage_metrics"]["cache"]["read_tokens"] == 0
    assert sanitized.payload["usage_metrics"]["cache"]["uncached_input_tokens"] == 100
    assert "usage_normalization_failed" not in sanitized.payload
    assert "usage_unavailable_reason" not in sanitized.payload
    assert cost.priced_model_steps == 1
    assert cost.unpriced_model_steps == 0
    assert cost.total_cost == Decimal("0.0014")
    assert actual.amount == cost.total_cost


def test_compaction_preserves_rejected_openai_usage_through_sanitization() -> None:
    raw_usage = {
        "input_tokens": 100,
        "output_tokens": 0,
        "input_tokens_details": {"cached_tokens": 80},
        "cache_read_input_tokens": 70,
    }
    payload = runtime_context_module._compaction_model_completed_payload(
        completed_payload={
            "usage": raw_usage,
            "usage_unavailable_reason": "provider supplied reason",
        },
        provider_name="renamed-openai",
        fallback_model="summary-model",
        compactor="ModelCompactor",
        usage_dialect=UsageDialect.OPENAI,
    )

    sanitized = runtime_context_module.sanitize_context_compaction_telemetry(
        runtime_context_module.ContextCompactionTelemetry(
            event_type=EventType.MODEL_COMPLETED,
            payload=payload,
        )
    )

    assert sanitized.payload["usage"] == raw_usage
    assert sanitized.payload["usage_normalization_failed"] is True
    assert "usage_metrics" not in sanitized.payload
    assert sanitized.payload["usage_unavailable_reason"] == ("invalid compaction usage telemetry")


@pytest.mark.parametrize("invalid_text", ["100\x00", "\ud800"], ids=["nul", "surrogate"])
@pytest.mark.parametrize(
    "invalid_primary_counter",
    [True, False],
    ids=["invalid-primary", "invalid-extra-field"],
)
def test_automatic_compaction_persists_undurable_usage_as_unpriced_evidence(
    invalid_text: str,
    invalid_primary_counter: bool,
) -> None:
    class UndurableOpenAICompactionProvider(FakeProvider):
        name = "renamed-openai"
        usage_dialect = UsageDialect.OPENAI

    compactor_provider = UndurableOpenAICompactionProvider(
        [
            ModelStreamEvent.text_delta("model summary"),
            ModelStreamEvent.completed(
                {
                    "model": "summary-model",
                    "usage": (
                        {
                            "input_tokens": invalid_text,
                            "output_tokens": 1,
                        }
                        if invalid_primary_counter
                        else {
                            "input_tokens": 10,
                            "output_tokens": 1,
                            "provider_note": invalid_text,
                        }
                    ),
                }
            ),
        ]
    )
    runtime_provider = FakeProvider(
        [
            ModelStreamEvent.text_delta("done"),
            ModelStreamEvent.completed(
                {
                    "model": "fake-model",
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                }
            ),
        ]
    )
    pricing = PriceBook(
        prices=(
            ModelPrice.fixed(
                provider_name="renamed-openai",
                model="summary-model",
                input_per_million=Decimal("1"),
                output_per_million=Decimal("1"),
            ),
            ModelPrice.fixed(
                provider_name="fake",
                model="fake-model",
                input_per_million=Decimal("1"),
                output_per_million=Decimal("1"),
            ),
        )
    )
    app = CayuApp(enable_logging=False)
    app.register_provider(runtime_provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        context_policy=CheckpointCompactionContextPolicy(
            compactor=ModelCompactor(
                provider=compactor_provider,
                model="summary-model",
            ),
            max_user_turns=1,
            compact_after_messages=2,
        ),
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id=(
                    "automatic-undurable-compaction-"
                    f"{invalid_primary_counter}-{ord(invalid_text[-1])}"
                ),
                messages=[
                    Message.text("user", "old"),
                    Message.text("assistant", "old answer"),
                    Message.text("user", "current"),
                ],
            ),
        )
    )
    completed = next(
        event
        for event in events
        if event.type == EventType.MODEL_COMPLETED
        and event.payload.get("purpose") == "context_compaction"
    )
    cost = asyncio.run(app.get_session_cost(completed.session_id, pricing))

    assert "usage" not in completed.payload
    assert "usage_metrics" not in completed.payload
    assert completed.payload["usage_normalization_failed"] is True
    assert completed.payload["usage_unavailable_reason"] == ("invalid compaction usage telemetry")
    assert cost.model_steps == 1
    assert cost.priced_model_steps == 0
    assert cost.unpriced_model_steps == 1
    assert runtime_provider.requests == []
    assert events[-1].type == EventType.SESSION_FAILED


@pytest.mark.parametrize(
    ("raw_usage", "dialect", "preserves_raw_usage"),
    [
        (
            {"input_tokens": 100, "total_tokens": 100},
            UsageDialect.OPENAI,
            True,
        ),
        (
            {"output_tokens": 20, "total_tokens": 20},
            UsageDialect.OPENAI,
            True,
        ),
        (
            {
                "input_tokens": 100,
                "output_tokens": 20,
                "input_tokens_details": {"cached_tokens": 80},
                "cache_read_input_tokens": 70,
            },
            UsageDialect.AUTO,
            True,
        ),
        (
            {
                "input_tokens": 100,
                "output_tokens": 20,
                "output_tokens_details": {"reasoning_tokens": "10"},
            },
            UsageDialect.OPENAI,
            False,
        ),
        (
            {
                "input_tokens": "100",
                "output_tokens": 20,
            },
            UsageDialect.AUTO,
            False,
        ),
        (
            {
                "input_tokens": 100,
                "output_tokens": 20,
                "input_tokens_details": {"cached_tokens": 80},
                "cache_read_input_tokens": 80,
            },
            UsageDialect.AUTO,
            True,
        ),
        (
            {
                "input_tokens": 100,
                "output_tokens": 20,
                "cache_read_input_tokens": "80",
            },
            UsageDialect.AUTO,
            False,
        ),
    ],
    ids=(
        "missing-output",
        "missing-input",
        "mixed-cache-conflict",
        "malformed-reasoning",
        "malformed-input-without-details",
        "ambiguous-matching-cache-counters",
        "malformed-inferred-cache-read",
    ),
)
def test_compaction_rejects_incomplete_or_mixed_usage_before_pricing(
    raw_usage: dict[str, object],
    dialect: UsageDialect,
    preserves_raw_usage: bool,
) -> None:
    payload = runtime_context_module._compaction_model_completed_payload(
        completed_payload={"usage": raw_usage},
        provider_name="gateway",
        fallback_model="summary-model",
        compactor="ModelCompactor",
        usage_dialect=dialect,
    )
    sanitized = runtime_context_module.sanitize_context_compaction_telemetry(
        runtime_context_module.ContextCompactionTelemetry(
            event_type=EventType.MODEL_COMPLETED,
            payload=payload,
        )
    )

    if preserves_raw_usage:
        assert sanitized.payload["usage"] == raw_usage
    else:
        assert "usage" not in sanitized.payload
    assert sanitized.payload["usage_normalization_failed"] is True
    assert "usage_metrics" not in sanitized.payload
    assert sanitized.payload["usage_unavailable_reason"] == ("invalid compaction usage telemetry")


def test_compaction_preserves_conflicting_cache_write_usage_as_bounded_evidence() -> None:
    raw_usage = {
        "input_tokens": 100,
        "output_tokens": 20,
        "cache_creation_input_tokens": 5,
        "cache_creation": {"ephemeral_5m_input_tokens": 6},
    }
    payload = runtime_context_module._compaction_model_completed_payload(
        completed_payload={"usage": raw_usage},
        provider_name="gateway",
        fallback_model="summary-model",
        compactor="ModelCompactor",
        usage_dialect=UsageDialect.ANTHROPIC,
    )
    sanitized = runtime_context_module.sanitize_context_compaction_telemetry(
        runtime_context_module.ContextCompactionTelemetry(
            event_type=EventType.MODEL_COMPLETED,
            payload=payload,
        )
    )

    assert sanitized.payload["usage"] == {
        "input_tokens": 100,
        "output_tokens": 20,
        "cache_creation_input_tokens": 5,
        "cache_creation": {"bounded_total": 6},
    }
    assert sanitized.payload["usage_normalization_failed"] is True
    assert "usage_metrics" not in sanitized.payload
    assert sanitized.payload["usage_unavailable_reason"] == ("invalid compaction usage telemetry")

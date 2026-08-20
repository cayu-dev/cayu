from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from examples._advanced_support.runtime import (
    _runtime_failure_summary,
    runtime_evidence_for_roles,
)
from examples.prompt_cache_compaction.deterministic import run
from examples.prompt_cache_compaction.live import _thinking_for_model
from examples.prompt_cache_compaction.scenario import (
    LoadStableContextTool,
    _model_step_attempt_events,
    _paired_cost_quality_report,
    _retention_quality,
    _usage_snapshot_payload,
)

from cayu import (
    CayuApp,
    Event,
    EventType,
    InMemorySessionStore,
    Message,
    RunRequest,
    SessionIdentity,
    SessionStatus,
    ToolEffect,
)


def test_stable_context_loader_declares_its_read_only_effect() -> None:
    assert LoadStableContextTool.spec.effect is ToolEffect.NONE


def test_runtime_failure_summary_keeps_provider_diagnostics_without_request_data() -> None:
    events = [
        Event(
            type=EventType.MODEL_ERROR,
            session_id="failed-session",
            payload={
                "error_type": "AnthropicAPIError",
                "error": "invalid request",
                "provider_error_type": "invalid_request_error",
                "status_code": 400,
                "request": {"secret": "must not be reported"},
            },
        ),
        Event(
            type=EventType.SESSION_FAILED,
            session_id="failed-session",
            payload={"error_type": "AnthropicAPIError", "error": "invalid request"},
        ),
    ]

    assert _runtime_failure_summary(events) == [
        {
            "type": "model.error",
            "error_type": "AnthropicAPIError",
            "error": "invalid request",
            "provider_error_type": "invalid_request_error",
            "status_code": 400,
        },
        {
            "type": "session.failed",
            "error_type": "AnthropicAPIError",
            "error": "invalid request",
        },
    ]


def test_runtime_evidence_for_roles_includes_safe_failure_diagnostics() -> None:
    async def scenario() -> None:
        store = InMemorySessionStore()
        await store.create(
            RunRequest(
                agent_name="agent",
                session_id="failed-session",
                messages=[Message.text("user", "request body")],
            ),
            identity=SessionIdentity(provider_name="provider", model="model"),
        )
        await store.append_event(
            "failed-session",
            Event(
                type=EventType.MODEL_ERROR,
                session_id="failed-session",
                payload={
                    "error_type": "ProviderError",
                    "error": "bad request",
                    "request": {"secret": "must not be reported"},
                },
            ),
        )
        await store.update_status("failed-session", SessionStatus.FAILED)
        with pytest.raises(RuntimeError) as caught:
            await runtime_evidence_for_roles(
                CayuApp(session_store=store, enable_logging=False),
                {"failed-session": "test"},
            )
        assert str(caught.value) == (
            "Session failed-session did not complete: failed; "
            "failures=[{'type': 'model.error', 'error_type': 'ProviderError', "
            "'error': 'bad request'}]"
        )

    asyncio.run(scenario())


def test_live_thinking_configuration_matches_anthropic_model_capability() -> None:
    haiku = _thinking_for_model("claude-haiku-4-5-20251001")
    sonnet = _thinking_for_model("claude-sonnet-4-6")
    explicit_budgeted = _thinking_for_model("custom-model", mode="budgeted")

    assert haiku.max_tokens == 1024
    assert haiku.effort is None
    assert sonnet.max_tokens is None
    assert sonnet.effort == "low"
    assert explicit_budgeted.max_tokens == 1024


def test_paired_usage_evidence_distinguishes_missing_counters_from_zero() -> None:
    assert _usage_snapshot_payload({}) == {
        "usage_available": False,
        "cache_usage_available": False,
        "input_tokens": None,
        "output_tokens": None,
        "reasoning_output_tokens": None,
        "cache_read_tokens": None,
        "cache_write_tokens": None,
        "uncached_input_tokens": None,
    }
    assert (
        _usage_snapshot_payload(
            {
                "usage": {"input_tokens": 10, "output_tokens": 2},
                "usage_metrics": {
                    "input_tokens": 10,
                    "output_tokens": 2,
                    "reasoning_output_tokens": 0,
                    "cache": {
                        "read_tokens": 0,
                        "write_tokens": 0,
                        "cached_input_tokens": 0,
                        "uncached_input_tokens": 10,
                    },
                },
            }
        )["cache_usage_available"]
        is False
    )
    assert (
        _usage_snapshot_payload(
            {
                "usage": {
                    "input_tokens": 10,
                    "output_tokens": 2,
                    "cache_read_input_tokens": 0,
                    "cache_creation_input_tokens": 0,
                },
                "usage_metrics": {
                    "input_tokens": 10,
                    "output_tokens": 2,
                    "reasoning_output_tokens": 0,
                    "cache": {
                        "read_tokens": 0,
                        "write_tokens": 0,
                        "cached_input_tokens": 0,
                        "uncached_input_tokens": 10,
                    },
                },
            }
        )["cache_usage_available"]
        is True
    )


def test_prompt_cache_pair_retains_failed_retry_attempt() -> None:
    candidate_session_id = "candidate-session"
    model_step_id = "mstep_11111111111111111111111111111111"
    first_attempt_id = "matt_11111111111111111111111111111111"
    second_attempt_id = "matt_22222222222222222222222222222222"
    shared = {
        "provider": "provider",
        "model": "model",
        "purpose": "context_compaction",
        "step": 1,
        "max_attempts": 2,
        "model_step_id": model_step_id,
    }
    candidate_events = [
        Event(
            type=EventType.MODEL_STARTED,
            session_id=candidate_session_id,
            payload={**shared, "attempt": 1, "model_attempt_id": first_attempt_id},
        ),
        Event(
            type=EventType.MODEL_RETRY,
            session_id=candidate_session_id,
            payload={
                **shared,
                "attempt": 1,
                "next_attempt": 2,
                "model_attempt_id": first_attempt_id,
            },
        ),
        Event(
            type=EventType.MODEL_ATTEMPT_DISCARDED,
            session_id=candidate_session_id,
            payload={
                **shared,
                "attempt": 1,
                "next_attempt": 2,
                "model_attempt_id": first_attempt_id,
            },
        ),
        Event(
            type=EventType.MODEL_STARTED,
            session_id=candidate_session_id,
            payload={**shared, "attempt": 2, "model_attempt_id": second_attempt_id},
        ),
        Event(
            type=EventType.MODEL_COMPLETED,
            session_id=candidate_session_id,
            payload={
                **shared,
                "attempt": 2,
                "model_attempt_id": second_attempt_id,
                "provider_name": "provider",
                "usage_metrics": {
                    "provider_name": "provider",
                    "model": "model",
                    "input_tokens": 50,
                    "output_tokens": 0,
                    "total_tokens": 50,
                    "reasoning_output_tokens": 0,
                    "cache": {
                        "read_tokens": 0,
                        "write_tokens": 0,
                        "cached_input_tokens": 0,
                        "uncached_input_tokens": 50,
                    },
                },
            },
        ),
    ]
    baseline_event = Event(
        type=EventType.MODEL_COMPLETED,
        session_id="baseline-session",
        payload=candidate_events[-1].payload,
    )

    report = _paired_cost_quality_report(
        candidate_events=_model_step_attempt_events(candidate_events, candidate_events[-1]),
        baseline_events=(baseline_event,),
        source_checkpoint_id="checkpoint-1",
        baseline_quality=_retention_quality(
            passed=True,
            reference="session://baseline/summary",
        ),
        candidate_quality=_retention_quality(
            passed=True,
            reference="session://candidate/summary",
        ),
        price_book=None,
    )

    assert report["status"] == "unavailable"
    candidate = report["pairs"][0]["candidate"]
    assert candidate["whole_harness"]["attempt_count"] == 2
    assert candidate["whole_harness"]["retry_attempt_count"] == 1
    assert candidate["whole_harness"]["missing_usage_attempt_count"] == 1


def test_prompt_cache_compaction_preserves_prefix_then_bounds_the_delta(tmp_path: Path) -> None:
    result = asyncio.run(run(tmp_path))

    assert result.status == "verified"
    assert result.scenario == "prompt-cache-compaction"
    assert result.assertions == {
        "tool_session_exercised": True,
        "first_compaction_extended_exact_request_prefix": True,
        "first_compaction_reused_provider_cache": True,
        "bounded_baseline_used_same_compactable_source": True,
        "bounded_baseline_stripped_exact_request_shape": True,
        "paired_cache_counters_reported_separately": True,
        "paired_model_configuration_matches": True,
        "paired_cost_claim_is_provenance_gated": True,
        "paired_summaries_pass_quality_floor": True,
        "second_compaction_used_bounded_delta": True,
        "compaction_modes_recorded": True,
        "compaction_spend_persisted": True,
        "comparison_spend_reported_separately": True,
        "session_completed": True,
    }
    assert result.metrics["model_requests"] == 7
    assert result.metrics["compaction_model_steps"] == 2
    assert result.metrics["first_compaction_cache_read_tokens"] == 1200
    assert result.metrics["first_compaction_attempt"] == {
        "usage_available": True,
        "cache_usage_available": True,
        "input_tokens": 1240,
        "output_tokens": 8,
        "reasoning_output_tokens": 0,
        "cache_read_tokens": 1200,
        "cache_write_tokens": 0,
        "uncached_input_tokens": 40,
    }
    assert result.metrics["bounded_baseline_first_compaction_attempt"] == {
        "usage_available": True,
        "cache_usage_available": True,
        "input_tokens": 1220,
        "output_tokens": 8,
        "reasoning_output_tokens": 0,
        "cache_read_tokens": 0,
        "cache_write_tokens": 0,
        "uncached_input_tokens": 1220,
    }
    cost_report = result.metrics["paired_first_compaction_cost"]
    assert cost_report["schema_version"] == 3
    assert cost_report["status"] == "verified"
    assert cost_report["aggregate"]["eligible_pair_ids"] == ["first-prompt-cache-compaction"]
    pair = cost_report["pairs"][0]
    assert pair["baseline_cost"] == "0.00378"
    assert pair["candidate_cost"] == "0.00060"
    assert pair["savings"] == "0.00318"
    assert pair["savings_percentage"] == "84.13"
    assert [item["operation"] for item in pair["baseline"]["operations"]] == ["comparison_control"]
    assert [item["operation"] for item in pair["candidate"]["operations"]] == ["compaction"]
    assert pair["candidate"]["whole_harness"]["cache_read_input_tokens"] == 1200
    assert pair["candidate"]["pricing_provenance"] == [
        {
            "source": "deterministic fixture; not provider pricing",
            "url": "https://example.invalid/cayu/prompt-cache-pricing-fixture",
            "as_of": "2026-01-01",
        }
    ]
    assert result.metrics["retry_inclusive_candidate_session"] == {
        "input_tokens": 3877,
        "output_tokens": 40,
        "model_steps": 7,
        "provider_attempts": 7,
        "cost_status": "priced",
        "cost": "0.008991",
        "unpriced_or_missing_usage_attempts": 0,
    }
    assert result.metrics["benchmark_harness"] == {
        "input_tokens": 5097,
        "output_tokens": 48,
        "model_steps": 8,
        "provider_attempts": 8,
        "cost_status": "priced",
        "cost": "0.012771",
        "unpriced_or_missing_usage_attempts": 0,
    }
    assert result.metrics["provider_attempts"] == 7
    assert result.metrics["provider_attempts_beyond_completed_steps"] == 0
    assert result.metrics["comparison_provider_attempts"] == 1
    assert result.metrics["total_provider_attempts_including_comparison"] == 8
    assert result.sessions[0].model_steps == 7
    assert result.sessions[0].tool_calls == 1
    assert result.sessions[0].compaction_count == 2
    assert result.output_path is not None
    assert result.output_path.exists()

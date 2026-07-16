from __future__ import annotations

import asyncio
from pathlib import Path

from examples._advanced_support.runtime import _runtime_failure_summary
from examples.prompt_cache_compaction.deterministic import run
from examples.prompt_cache_compaction.live import _thinking_for_model
from examples.prompt_cache_compaction.scenario import LoadStableContextTool, _usage_snapshot_payload

from cayu import Event, EventType, ToolEffect


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
    assert result.metrics["paired_first_compaction_cost"] == {
        "status": "priced",
        "currency": "USD",
        "candidate_cost": "0.00060",
        "bounded_baseline_cost": "0.00378",
        "savings": "0.00318",
        "savings_percent": "84.13",
        "price_book_version": "deterministic-fixture-v1",
        "price_book_generated_at": "2026-01-01T00:00:00Z",
        "pricing_provider_name": "scripted",
        "pricing_model": "scripted-model",
        "pricing_match": "prefix",
        "pricing_tier_max_input_tokens": None,
        "pricing_provenance": {
            "source": "deterministic fixture; not provider pricing",
            "url": "https://example.invalid/cayu/prompt-cache-pricing-fixture",
            "as_of": "2026-01-01",
        },
    }
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

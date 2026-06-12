from __future__ import annotations

from cayu.core import Event, EventType
from cayu.runtime.stop_policy import (
    RunLimits,
    StopLimit,
    first_reached_limit,
    has_run_limits,
)
from cayu.runtime.usage import (
    normalize_usage_metrics,
    session_usage_summary,
    usage_metrics_from_event_payload,
)


def test_normalize_openai_usage_metrics() -> None:
    metrics = normalize_usage_metrics(
        provider_name="openai",
        model="gpt-5.5",
        raw_usage={
            "input_tokens": 100,
            "input_tokens_details": {"cached_tokens": 60},
            "output_tokens": 20,
            "output_tokens_details": {"reasoning_tokens": 5},
            "total_tokens": 120,
        },
    )

    assert metrics is not None
    assert metrics.provider_name == "openai"
    assert metrics.model == "gpt-5.5"
    assert metrics.input_tokens == 100
    assert metrics.output_tokens == 20
    assert metrics.total_tokens == 120
    assert metrics.reasoning_output_tokens == 5
    assert metrics.cache.read_tokens == 60
    assert metrics.cache.write_tokens == 0
    assert metrics.cache.cached_input_tokens == 60
    assert metrics.cache.uncached_input_tokens == 40


def test_normalize_openai_chat_usage_shape() -> None:
    metrics = normalize_usage_metrics(
        provider_name="openai",
        model="gpt-5.5",
        raw_usage={
            "prompt_tokens": 100,
            "prompt_tokens_details": {"cached_tokens": 40},
            "completion_tokens": 10,
            "completion_tokens_details": {"reasoning_tokens": 3},
        },
    )

    assert metrics is not None
    assert metrics.input_tokens == 100
    assert metrics.output_tokens == 10
    assert metrics.total_tokens == 110
    assert metrics.reasoning_output_tokens == 3
    assert metrics.cache.read_tokens == 40
    assert metrics.cache.cached_input_tokens == 40
    assert metrics.cache.uncached_input_tokens == 60


def test_normalize_anthropic_usage_metrics() -> None:
    metrics = normalize_usage_metrics(
        provider_name="anthropic",
        model="claude-sonnet-4-6",
        raw_usage={
            "input_tokens": 100,
            "output_tokens": 20,
            "cache_read_input_tokens": 70,
            "cache_creation_input_tokens": 0,
            "cache_creation": {
                "ephemeral_5m_input_tokens": 2,
                "ephemeral_1h_input_tokens": 3,
            },
        },
    )

    assert metrics is not None
    assert metrics.provider_name == "anthropic"
    assert metrics.model == "claude-sonnet-4-6"
    assert metrics.input_tokens == 175
    assert metrics.output_tokens == 20
    assert metrics.total_tokens == 195
    assert metrics.cache.read_tokens == 70
    assert metrics.cache.write_tokens == 5
    assert metrics.cache.cached_input_tokens == 70
    assert metrics.cache.uncached_input_tokens == 100


def test_normalize_anthropic_top_level_cache_write_counter() -> None:
    metrics = normalize_usage_metrics(
        provider_name="anthropic",
        model="claude-sonnet-4-6",
        raw_usage={
            "input_tokens": 100,
            "output_tokens": 20,
            "cache_creation_input_tokens": 10,
        },
    )

    assert metrics is not None
    assert metrics.cache.write_tokens == 10


def test_session_usage_summary_aggregates_model_steps_and_tools() -> None:
    events = [
        Event(
            type=EventType.MODEL_COMPLETED,
            session_id="session_1",
            payload={
                "usage_metrics": {
                    "provider_name": "openai",
                    "model": "gpt-5.5",
                    "input_tokens": 100,
                    "output_tokens": 20,
                    "total_tokens": 120,
                    "reasoning_output_tokens": 4,
                    "cache": {
                        "read_tokens": 60,
                        "write_tokens": 0,
                        "cached_input_tokens": 60,
                        "uncached_input_tokens": 40,
                    },
                }
            },
        ),
        Event(type=EventType.TOOL_CALL_STARTED, session_id="session_1", tool_name="read_file"),
        Event(
            type=EventType.MODEL_COMPLETED,
            session_id="session_1",
            payload={
                "usage": {
                    "input_tokens": 50,
                    "output_tokens": 10,
                    "cache_read_input_tokens": 5,
                },
                "provider_name": "anthropic",
                "model": "claude-sonnet-4-6",
            },
        ),
    ]

    summary = session_usage_summary("session_1", events)

    assert summary.session_id == "session_1"
    assert summary.model_steps == 2
    assert summary.tool_calls == 1
    assert summary.provider_names == ["openai", "anthropic"]
    assert summary.models == ["gpt-5.5", "claude-sonnet-4-6"]
    assert summary.usage.input_tokens == 155
    assert summary.usage.output_tokens == 30
    assert summary.usage.total_tokens == 185
    assert summary.usage.reasoning_output_tokens == 4
    assert summary.usage.cache.read_tokens == 65
    assert summary.usage.cache.cached_input_tokens == 65
    assert summary.usage.cache.uncached_input_tokens == 90


def test_usage_metrics_from_event_payload_rejects_non_usage_payload() -> None:
    assert usage_metrics_from_event_payload({"usage": "bad"}) is None
    assert usage_metrics_from_event_payload({"usage": {}}) is None


def test_run_limits_detect_reached_token_budget() -> None:
    summary = session_usage_summary(
        "session_1",
        [
            Event(
                type=EventType.MODEL_COMPLETED,
                session_id="session_1",
                payload={
                    "usage_metrics": {
                        "input_tokens": 7,
                        "output_tokens": 3,
                        "total_tokens": 10,
                        "reasoning_output_tokens": 0,
                        "cache": {
                            "read_tokens": 0,
                            "write_tokens": 0,
                            "cached_input_tokens": 0,
                            "uncached_input_tokens": 7,
                        },
                    }
                },
            )
        ],
    )

    decision = first_reached_limit(
        limits=RunLimits(max_total_tokens=10),
        usage=summary,
        elapsed_seconds=0,
    )

    assert decision is not None
    assert decision.limit == StopLimit.TOTAL_TOKENS
    assert decision.maximum == 10
    assert decision.actual == 10


def test_run_limits_allow_tool_call_until_capacity_is_exceeded() -> None:
    summary = session_usage_summary(
        "session_1",
        [Event(type=EventType.TOOL_CALL_STARTED, session_id="session_1")],
    )

    allowed = first_reached_limit(
        limits=RunLimits(max_tool_calls=2),
        usage=summary,
        elapsed_seconds=0,
        pending_tool_calls=1,
    )
    blocked = first_reached_limit(
        limits=RunLimits(max_tool_calls=2),
        usage=summary,
        elapsed_seconds=0,
        pending_tool_calls=2,
    )

    assert allowed is None
    assert blocked is not None
    assert blocked.limit == StopLimit.TOOL_CALLS
    assert blocked.actual == 3


def test_has_run_limits_detects_empty_and_configured_limits() -> None:
    assert not has_run_limits(RunLimits())
    assert has_run_limits(RunLimits(max_elapsed_seconds=1))

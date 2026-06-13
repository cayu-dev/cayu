from __future__ import annotations

import asyncio
from decimal import Decimal

from cayu.core import Event, EventType, Message
from cayu.runtime import CayuApp, ModelPricing, PricingCatalog, RunRequest, SessionIdentity
from cayu.runtime.costs import estimate_session_cost
from cayu.runtime.stop_policy import (
    RunLimits,
    StopLimit,
    copy_run_limits,
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


def test_estimate_session_cost_prices_each_model_step() -> None:
    events = [
        Event(
            type=EventType.MODEL_COMPLETED,
            session_id="session_1",
            payload={
                "usage_metrics": {
                    "provider_name": "openai",
                    "model": "gpt-5.5-2026-04-23",
                    "input_tokens": 1000,
                    "output_tokens": 200,
                    "total_tokens": 1200,
                    "reasoning_output_tokens": 0,
                    "cache": {
                        "read_tokens": 400,
                        "write_tokens": 0,
                        "cached_input_tokens": 400,
                        "uncached_input_tokens": 600,
                    },
                }
            },
        ),
        Event(
            type=EventType.MODEL_COMPLETED,
            session_id="session_1",
            payload={
                "usage_metrics": {
                    "provider_name": "anthropic",
                    "model": "claude-sonnet-4-6",
                    "input_tokens": 1000,
                    "output_tokens": 100,
                    "total_tokens": 1100,
                    "reasoning_output_tokens": 0,
                    "cache": {
                        "read_tokens": 300,
                        "write_tokens": 200,
                        "cached_input_tokens": 300,
                        "uncached_input_tokens": 500,
                    },
                }
            },
        ),
    ]
    pricing = PricingCatalog(
        prices=(
            ModelPricing(
                provider_name="openai",
                model="gpt-5.5",
                match="prefix",
                input_per_million=Decimal("2"),
                output_per_million=Decimal("8"),
                cache_read_input_per_million=Decimal("0.5"),
            ),
            ModelPricing(
                provider_name="anthropic",
                model="claude-sonnet-4-6",
                input_per_million=Decimal("3"),
                output_per_million=Decimal("15"),
                cache_read_input_per_million=Decimal("0.3"),
                cache_write_input_per_million=Decimal("3.75"),
            ),
        )
    )

    summary = estimate_session_cost(session_id="session_1", events=events, pricing=pricing)

    assert summary.model_steps == 2
    assert summary.priced_model_steps == 2
    assert summary.unpriced_model_steps == 0
    assert summary.line_items[0].pricing_model == "gpt-5.5"
    assert summary.line_items[0].pricing_match == "prefix"
    assert summary.line_items[0].input_cost == Decimal("0.0012")
    assert summary.line_items[0].cache_read_input_cost == Decimal("0.0002")
    assert summary.line_items[0].output_cost == Decimal("0.0016")
    assert summary.line_items[0].total_cost == Decimal("0.0030")
    assert summary.line_items[1].input_cost == Decimal("0.0015")
    assert summary.line_items[1].cache_read_input_cost == Decimal("0.00009")
    assert summary.line_items[1].cache_write_input_cost == Decimal("0.00075")
    assert summary.line_items[1].output_cost == Decimal("0.0015")
    assert summary.line_items[1].total_cost == Decimal("0.00384")
    assert summary.total_cost == Decimal("0.00684")


def test_estimate_session_cost_reports_unpriced_model_steps() -> None:
    events = [
        Event(
            type=EventType.MODEL_COMPLETED,
            session_id="session_1",
            payload={
                "usage_metrics": {
                    "provider_name": "openai",
                    "model": "gpt-unknown",
                    "input_tokens": 100,
                    "output_tokens": 10,
                    "total_tokens": 110,
                    "reasoning_output_tokens": 0,
                    "cache": {
                        "read_tokens": 0,
                        "write_tokens": 0,
                        "cached_input_tokens": 0,
                        "uncached_input_tokens": 100,
                    },
                }
            },
        ),
        Event(type=EventType.MODEL_COMPLETED, session_id="session_1", payload={}),
    ]
    pricing = PricingCatalog(
        prices=(
            ModelPricing(
                provider_name="openai",
                model="gpt-5.5",
                input_per_million=Decimal("1"),
                output_per_million=Decimal("2"),
            ),
        )
    )

    summary = estimate_session_cost(session_id="session_1", events=events, pricing=pricing)

    assert summary.total_cost == Decimal("0")
    assert summary.priced_model_steps == 0
    assert summary.unpriced_model_steps == 2
    assert summary.line_items[0].missing_pricing_reason == "no matching model pricing"
    assert (
        summary.line_items[1].missing_pricing_reason
        == "model.completed event has no token usage metrics"
    )


def test_estimate_session_cost_rejects_currency_mismatch() -> None:
    events = [
        Event(
            type=EventType.MODEL_COMPLETED,
            session_id="session_1",
            payload={
                "usage_metrics": {
                    "provider_name": "openai",
                    "model": "gpt-5.5",
                    "input_tokens": 100,
                    "output_tokens": 10,
                    "total_tokens": 110,
                    "reasoning_output_tokens": 0,
                    "cache": {
                        "read_tokens": 0,
                        "write_tokens": 0,
                        "cached_input_tokens": 0,
                        "uncached_input_tokens": 100,
                    },
                }
            },
        )
    ]
    pricing = PricingCatalog(
        prices=(
            ModelPricing(
                provider_name="openai",
                model="gpt-5.5",
                input_per_million=Decimal("1"),
                output_per_million=Decimal("2"),
                currency="EUR",
            ),
        )
    )

    summary = estimate_session_cost(
        session_id="session_1",
        events=events,
        pricing=pricing,
        currency="USD",
    )

    assert summary.total_cost == Decimal("0")
    assert summary.unpriced_model_steps == 1
    assert summary.line_items[0].missing_pricing_reason == (
        "pricing currency EUR does not match requested USD"
    )


def test_estimate_session_cost_prefers_exact_pricing_over_prefix_pricing() -> None:
    events = [
        Event(
            type=EventType.MODEL_COMPLETED,
            session_id="session_1",
            payload={
                "usage_metrics": {
                    "provider_name": "openai",
                    "model": "gpt-5.5-special",
                    "input_tokens": 1000,
                    "output_tokens": 0,
                    "total_tokens": 1000,
                    "reasoning_output_tokens": 0,
                    "cache": {
                        "read_tokens": 0,
                        "write_tokens": 0,
                        "cached_input_tokens": 0,
                        "uncached_input_tokens": 1000,
                    },
                }
            },
        )
    ]
    pricing = PricingCatalog(
        prices=(
            ModelPricing(
                provider_name="openai",
                model="gpt-5.5",
                match="prefix",
                input_per_million=Decimal("1"),
                output_per_million=Decimal("1"),
            ),
            ModelPricing(
                provider_name="openai",
                model="gpt-5.5-special",
                match="exact",
                input_per_million=Decimal("10"),
                output_per_million=Decimal("1"),
            ),
        )
    )

    summary = estimate_session_cost(
        session_id="session_1",
        events=events,
        pricing=pricing,
        currency="usd",
    )

    assert summary.currency == "USD"
    assert summary.line_items[0].pricing_match == "exact"
    assert summary.total_cost == Decimal("0.01")


def test_estimate_session_cost_respects_explicit_zero_cache_prices() -> None:
    events = [
        Event(
            type=EventType.MODEL_COMPLETED,
            session_id="session_1",
            payload={
                "usage_metrics": {
                    "provider_name": "openai",
                    "model": "gpt-5.5",
                    "input_tokens": 1000,
                    "output_tokens": 0,
                    "total_tokens": 1000,
                    "reasoning_output_tokens": 0,
                    "cache": {
                        "read_tokens": 800,
                        "write_tokens": 100,
                        "cached_input_tokens": 800,
                        "uncached_input_tokens": 100,
                    },
                }
            },
        )
    ]
    pricing = PricingCatalog(
        prices=(
            ModelPricing(
                provider_name="openai",
                model="gpt-5.5",
                input_per_million=Decimal("10"),
                output_per_million=Decimal("10"),
                cache_read_input_per_million=Decimal("0"),
                cache_write_input_per_million=Decimal("0"),
            ),
        )
    )

    summary = estimate_session_cost(session_id="session_1", events=events, pricing=pricing)

    assert summary.line_items[0].input_cost == Decimal("0.001")
    assert summary.line_items[0].cache_read_input_cost == Decimal("0")
    assert summary.line_items[0].cache_write_input_cost == Decimal("0")
    assert summary.total_cost == Decimal("0.001")


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


def test_run_limits_scope_defaults_to_session() -> None:
    assert RunLimits().scope == "session"


def test_copy_run_limits_preserves_scope() -> None:
    copied = copy_run_limits(RunLimits(scope="run", max_tool_calls=3))
    assert copied.scope == "run"
    assert copied.max_tool_calls == 3


def test_run_limits_scope_alone_is_not_a_limit() -> None:
    assert not has_run_limits(RunLimits(scope="run"))


def test_cayu_app_get_session_cost_uses_durable_events() -> None:
    app = CayuApp()
    asyncio.run(
        app.session_store.create(
            RunRequest(
                agent_name="assistant",
                messages=[Message.text("user", "hi")],
                session_id="cost_session",
            ),
            identity=SessionIdentity(
                provider_name="openai",
                model="gpt-5.5",
                runtime_name="cayu",
                runtime_version=None,
            ),
        )
    )
    asyncio.run(
        app.session_store.append_event(
            "cost_session",
            Event(
                type=EventType.MODEL_COMPLETED,
                session_id="cost_session",
                payload={
                    "usage_metrics": {
                        "provider_name": "openai",
                        "model": "gpt-5.5",
                        "input_tokens": 1000,
                        "output_tokens": 100,
                        "total_tokens": 1100,
                        "reasoning_output_tokens": 0,
                        "cache": {
                            "read_tokens": 0,
                            "write_tokens": 0,
                            "cached_input_tokens": 0,
                            "uncached_input_tokens": 1000,
                        },
                    }
                },
            ),
        )
    )
    pricing = PricingCatalog(
        prices=(
            ModelPricing(
                provider_name="openai",
                model="gpt-5.5",
                input_per_million=Decimal("1"),
                output_per_million=Decimal("10"),
            ),
        )
    )

    summary = asyncio.run(app.get_session_cost("cost_session", pricing))

    assert summary.total_cost == Decimal("0.002")

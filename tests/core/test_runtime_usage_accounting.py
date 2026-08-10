from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from decimal import Decimal

import pytest

from cayu.core import AgentSpec, Event, EventType, Message
from cayu.providers import (
    ModelProvider,
    ModelRequest,
    ModelStreamEvent,
    UsageDialect,
    bedrock_billing_identity,
)
from cayu.runtime import (
    BillingIdentity,
    BudgetLimit,
    BudgetPolicy,
    BudgetReservation,
    CayuApp,
    InMemoryBudgetLedger,
    InMemorySessionStore,
    ModelPrice,
    PriceBook,
    PriceSchedule,
    PriceTier,
    PricingContextSelector,
    Provenance,
    RunRequest,
    TieredPricing,
)
from cayu.storage import SQLiteSessionStore


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


def fake_budget_limit(max_estimated_cost: str) -> BudgetLimit:
    return BudgetLimit(
        max_estimated_cost=Decimal(max_estimated_cost),
        pricing=PriceBook(
            prices=(
                ModelPrice.fixed(
                    provider_name="fake",
                    model="fake-model",
                    input_per_million=Decimal("1"),
                    output_per_million=Decimal("10"),
                ),
            )
        ),
        scope="session",
    )


@pytest.mark.parametrize("invalid_text", ["resource\x00id", "\ud800"], ids=["nul", "surrogate"])
def test_undurable_provider_billing_identity_fails_before_reservation_or_dispatch(
    invalid_text: str,
) -> None:
    invalid_identity = BillingIdentity(
        provider_name="fake",
        resource_id="fake-model",
    ).model_copy(update={"resource_id": invalid_text})

    class UndurableBillingIdentityProvider(FakeProvider):
        async def billing_identity_for_request(
            self,
            request: ModelRequest,
        ) -> BillingIdentity:
            return invalid_identity

    provider = UndurableBillingIdentityProvider(
        [
            ModelStreamEvent.completed(
                {
                    "model": "fake-model",
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                }
            )
        ]
    )
    pricing = fake_budget_limit("10").pricing
    app = CayuApp(
        budget_policy=BudgetPolicy(
            limits=(
                BudgetLimit(
                    scope="app",
                    max_estimated_cost=Decimal("10"),
                    pricing=pricing,
                    reservation=BudgetReservation(
                        max_input_tokens=1_000_000,
                        max_output_tokens=0,
                    ),
                ),
            )
        ),
        budget_ledger=InMemoryBudgetLedger(),
        enable_logging=False,
    )
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id=f"undurable-billing-identity-{ord(invalid_text[-1])}",
                messages=[Message.text("user", "hello")],
            ),
        )
    )

    assert provider.requests == []
    assert EventType.BUDGET_RESERVED not in {event.type for event in events}
    assert EventType.MODEL_STARTED not in {event.type for event in events}
    model_error = next(event for event in events if event.type == EventType.MODEL_ERROR)
    assert model_error.payload["stage"] == "billing_identity_for_request"
    assert model_error.payload["provider_error_code"] == "billing_identity_resolution_failed"
    assert events[-1].type == EventType.SESSION_FAILED


@pytest.mark.parametrize(
    ("cache_read_price", "expected_amount"),
    [(None, "10"), (Decimal("0"), "2")],
)
def test_openai_cached_input_uses_one_price_rule_for_reservation_and_reconciliation(
    cache_read_price: Decimal | None,
    expected_amount: str,
) -> None:
    class RenamedOpenAIBudgetProvider(FakeProvider):
        name = "company-gateway"
        usage_dialect = UsageDialect.OPENAI

    provider = RenamedOpenAIBudgetProvider(
        [
            ModelStreamEvent.completed(
                {
                    "model": "gpt-test",
                    "usage": {
                        "input_tokens": 1_000_000,
                        "output_tokens": 0,
                        "input_tokens_details": {"cached_tokens": 800_000},
                    },
                }
            )
        ]
    )
    pricing = PriceBook(
        prices=(
            ModelPrice.fixed(
                provider_name="company-gateway",
                model="gpt-test",
                input_per_million=Decimal("10"),
                output_per_million=Decimal("10"),
                cache_read_input_per_million=cache_read_price,
            ),
        )
    )
    app = CayuApp(
        budget_policy=BudgetPolicy(
            limits=(
                BudgetLimit(
                    scope="app",
                    max_estimated_cost=Decimal("20"),
                    pricing=pricing,
                    reservation=BudgetReservation(
                        max_input_tokens=200_000,
                        max_output_tokens=0,
                        max_cache_read_input_tokens=800_000,
                    ),
                ),
            )
        ),
        budget_ledger=InMemoryBudgetLedger(),
    )
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="gpt-test"))

    session_id = f"sess_openai_cache_{expected_amount}"
    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "hello")],
            ),
        )
    )
    cost = asyncio.run(app.get_session_cost(session_id, pricing))

    reserved = next(event for event in events if event.type == EventType.BUDGET_RESERVED)
    completed = next(event for event in events if event.type == EventType.MODEL_COMPLETED)
    reconciled = next(event for event in events if event.type == EventType.BUDGET_RECONCILED)
    metrics = completed.payload["usage_metrics"]
    assert metrics["cache"]["read_tokens"] == 800_000
    assert metrics["cache"]["uncached_input_tokens"] == 200_000
    assert reserved.payload["requested"] == expected_amount
    assert reconciled.payload["actual_amount"] == expected_amount
    assert cost.total_cost == Decimal(expected_amount)
    assert cost.line_items[0].cache_read_input_tokens == 800_000
    assert cost.line_items[0].uncached_input_tokens == 200_000


def test_bedrock_cached_usage_with_provider_total_remains_priced() -> None:
    identity = bedrock_billing_identity(
        invoked_model="us.anthropic.claude-sonnet-4-6-v1",
        source_region="us-east-1",
        resource_type="foundation_model",
        requested_service_tier="default",
    )

    class CanonicalBedrockUsageProvider(FakeProvider):
        name = "bedrock"
        billing_provider_name = "bedrock"
        usage_dialect = UsageDialect.ANTHROPIC

        async def billing_identity_for_request(
            self,
            request: ModelRequest,
        ) -> BillingIdentity:
            assert request.model == identity.resource_id
            return identity

    provider = CanonicalBedrockUsageProvider(
        [
            ModelStreamEvent.completed(
                {
                    "model": identity.resource_id,
                    "usage": {
                        "input_tokens": 11,
                        "output_tokens": 2,
                        "total_tokens": 13,
                        "cache_read_input_tokens": 3,
                        "cache_creation_input_tokens": 1,
                        "cache_details": [{"ttl": "5m", "input_tokens": 1}],
                    },
                }
            )
        ]
    )
    pricing = PriceBook(
        prices=(
            ModelPrice(
                provider_name="bedrock",
                model=identity.resource_id,
                match="exact",
                pricing_context=PricingContextSelector(
                    dimensions={
                        "source_region": ("us-east-1",),
                        "service_tier": ("default",),
                    }
                ),
                cache_write_ttls=("5m",),
                schedules=(
                    PriceSchedule(
                        pricing=TieredPricing(
                            standard=(
                                PriceTier(
                                    input_per_million=Decimal("1"),
                                    output_per_million=Decimal("1"),
                                    cache_read_input_per_million=Decimal("0.1"),
                                ),
                            ),
                            cache_write_5m_per_million=Decimal("2"),
                        ),
                        provenance=Provenance(
                            source="official",
                            url="https://example.test/pricing",
                            as_of="2026-07-23",
                        ),
                    ),
                ),
            ),
        )
    )
    app = CayuApp()
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model=identity.resource_id))

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_bedrock_cached_provider_total",
                messages=[Message.text("user", "hello")],
            ),
        )
    )
    cost = asyncio.run(app.get_session_cost("sess_bedrock_cached_provider_total", pricing))

    completed = next(event for event in events if event.type == EventType.MODEL_COMPLETED)
    assert "usage_normalization_failed" not in completed.payload
    assert completed.payload["usage_metrics"]["input_tokens"] == 15
    assert completed.payload["usage_metrics"]["total_tokens"] == 17
    assert completed.payload["usage_metrics"]["cache"]["read_tokens"] == 3
    assert completed.payload["usage_metrics"]["cache"]["write_tokens"] == 1
    assert completed.payload["usage_metrics"]["cache"]["uncached_input_tokens"] == 11
    assert cost.priced_model_steps == 1
    assert cost.unpriced_model_steps == 0
    assert cost.line_items[0].input_tokens == 15
    assert cost.line_items[0].cache_read_input_tokens == 3
    assert cost.line_items[0].cache_write_input_tokens == 1
    assert cost.line_items[0].uncached_input_tokens == 11
    assert cost.total_cost == Decimal("0.0000153")


@pytest.mark.parametrize(
    ("usage_dialect", "session_suffix"),
    [
        (UsageDialect.OPENAI, "openai"),
        (UsageDialect.ANTHROPIC, "anthropic"),
        (UsageDialect.GEMINI, "gemini"),
        (UsageDialect.GENERIC, "generic"),
    ],
)
def test_authoritative_explicit_zero_usage_releases_the_reservation(
    usage_dialect: UsageDialect,
    session_suffix: str,
) -> None:
    class ZeroUsageProvider(FakeProvider):
        pass

    provider = ZeroUsageProvider(
        [
            ModelStreamEvent.completed(
                {
                    "model": "fake-model",
                    "usage": {
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "total_tokens": 0,
                    },
                }
            )
        ]
    )
    provider.usage_dialect = usage_dialect
    pricing = fake_budget_limit("10").pricing
    app = CayuApp(
        budget_policy=BudgetPolicy(
            limits=(
                BudgetLimit(
                    scope="app",
                    max_estimated_cost=Decimal("10"),
                    pricing=pricing,
                    reservation=BudgetReservation(
                        max_input_tokens=1_000_000,
                        max_output_tokens=0,
                    ),
                ),
            )
        ),
        budget_ledger=InMemoryBudgetLedger(),
    )
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id=f"sess_{session_suffix}_zero_usage",
                messages=[Message.text("user", "hello")],
            ),
        )
    )
    cost = asyncio.run(app.get_session_cost(f"sess_{session_suffix}_zero_usage", pricing))

    reserved = next(event for event in events if event.type == EventType.BUDGET_RESERVED)
    completed = next(event for event in events if event.type == EventType.MODEL_COMPLETED)
    reconciled = next(event for event in events if event.type == EventType.BUDGET_RECONCILED)
    assert completed.payload["usage_metrics"]["input_tokens"] == 0
    assert completed.payload["usage_metrics"]["output_tokens"] == 0
    assert "usage_normalization_failed" not in completed.payload
    assert reconciled.payload["actual_amount"] == "0"
    assert reconciled.payload["released_amount"] == reserved.payload["requested"]
    assert cost.priced_model_steps == 1
    assert cost.unpriced_model_steps == 0
    assert cost.total_cost == Decimal("0")
    assert events[-1].type == EventType.SESSION_COMPLETED


@pytest.mark.parametrize(
    ("raw_usage", "dialect"),
    [
        (
            {
                "input_tokens": 1_000_000,
                "output_tokens": 0,
                "input_tokens_details": {"cached_tokens": 800_000},
                "cache_read_input_tokens": 700_000,
            },
            UsageDialect.OPENAI,
        ),
        (
            {
                "input_tokens": 500_000,
                "prompt_tokens": 1_000_000,
                "output_tokens": 0,
            },
            UsageDialect.OPENAI,
        ),
        (
            {
                "input_tokens": 500_000,
                "prompt_tokens": 1_000_000,
                "output_tokens": 0,
                "total_tokens": 1_000_000,
                "input_tokens_details": {"cached_tokens": 400_000},
            },
            UsageDialect.AUTO,
        ),
        (
            {
                "input_tokens": 1_000_000,
                "total_tokens": 1_000_000,
            },
            UsageDialect.OPENAI,
        ),
        (
            {
                "output_tokens": 1,
                "total_tokens": 1,
            },
            UsageDialect.OPENAI,
        ),
        (
            {
                "input_tokens": 500_000,
                "output_tokens": 0,
                "input_tokens_details": {"cached_tokens": 400_000},
                "cache_read_input_tokens": 300_000,
            },
            UsageDialect.AUTO,
        ),
        (
            {
                "input_tokens": 1_000_000,
                "output_tokens": 0,
                "output_tokens_details": {"reasoning_tokens": "10"},
            },
            UsageDialect.OPENAI,
        ),
        (
            {
                "input_tokens": "1000000",
                "output_tokens": 0,
            },
            UsageDialect.AUTO,
        ),
        (
            {
                "input_tokens": 1_000_000,
                "output_tokens": 0,
                "input_tokens_details": {"cached_tokens": 800_000},
                "cache_read_input_tokens": 800_000,
            },
            UsageDialect.AUTO,
        ),
        (
            {
                "input_tokens": 1_000_000,
                "output_tokens": 0,
                "cache_read_input_tokens": "800000",
            },
            UsageDialect.AUTO,
        ),
        (
            {
                "input_tokens": 500_000,
                "output_tokens": 0,
                "cache_creation_input_tokens": 100_000,
                "cache_creation": {"ephemeral_5m_input_tokens": 200_000},
            },
            UsageDialect.ANTHROPIC,
        ),
        (
            {
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "output_tokens_details": {"thinking_tokens": 1},
            },
            UsageDialect.GENERIC,
        ),
    ],
    ids=(
        "contradictory-cache-read",
        "declared-openai-input-aliases",
        "generic-openai-shaped-input-aliases",
        "declared-openai-missing-output",
        "declared-openai-missing-input",
        "auto-mixed-cache-counters",
        "declared-openai-malformed-reasoning",
        "auto-malformed-input-without-details",
        "auto-ambiguous-matching-cache-counters",
        "auto-malformed-inferred-cache-read",
        "anthropic-cache-write-conflict",
        "generic-thinking-conflict",
    ),
)
def test_malformed_cached_usage_charges_the_reserved_amount(
    raw_usage: dict[str, object],
    dialect: UsageDialect,
) -> None:
    class GatewayBudgetProvider(FakeProvider):
        name = "company-gateway"
        usage_dialect = dialect

    provider = GatewayBudgetProvider(
        [
            ModelStreamEvent.completed(
                {
                    "model": "gpt-test",
                    "usage": raw_usage,
                }
            )
        ]
    )
    pricing = PriceBook(
        prices=(
            ModelPrice.fixed(
                provider_name="company-gateway",
                model="gpt-test",
                input_per_million=Decimal("10"),
                output_per_million=Decimal("10"),
            ),
        )
    )
    app = CayuApp(
        budget_policy=BudgetPolicy(
            limits=(
                BudgetLimit(
                    scope="app",
                    max_estimated_cost=Decimal("10"),
                    pricing=pricing,
                    reservation=BudgetReservation(
                        max_input_tokens=1_000_000,
                        max_output_tokens=0,
                    ),
                ),
            )
        ),
        budget_ledger=InMemoryBudgetLedger(),
    )
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="gpt-test"))

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_malformed_cached_usage",
                messages=[Message.text("user", "hello")],
            ),
        )
    )

    completed = next(event for event in events if event.type == EventType.MODEL_COMPLETED)
    reconciled = next(event for event in events if event.type == EventType.BUDGET_RECONCILED)
    assert completed.payload["usage"] == raw_usage
    assert "usage_metrics" not in completed.payload
    assert completed.payload["usage_normalization_failed"] is True
    assert reconciled.payload["actual_amount"] == "10"
    assert reconciled.payload["reason"] == (
        "model completed without priced usage; charged reserved amount"
    )
    assert events[-1].type == EventType.SESSION_INTERRUPTED


@pytest.mark.parametrize(
    ("raw_usage", "dialect"),
    [
        (
            {
                "input_tokens": 500_000,
                "prompt_tokens": 1_000_000,
                "output_tokens": 0,
                "total_tokens": 1_000_000,
                "input_tokens_details": {"cached_tokens": 400_000},
            },
            UsageDialect.AUTO,
        ),
        (
            {
                "output_tokens": 1,
                "total_tokens": 1,
            },
            UsageDialect.OPENAI,
        ),
        (
            {
                "input_tokens": 500_000,
                "output_tokens": 0,
                "input_tokens_details": {"cached_tokens": 400_000},
                "cache_read_input_tokens": 300_000,
            },
            UsageDialect.AUTO,
        ),
        (
            {
                "input_tokens": "500000",
                "output_tokens": 0,
            },
            UsageDialect.AUTO,
        ),
        (
            {
                "input_tokens": 500_000,
                "output_tokens": 0,
                "input_tokens_details": {"cached_tokens": 400_000},
                "cache_read_input_tokens": 400_000,
            },
            UsageDialect.AUTO,
        ),
        (
            {
                "input_tokens": 500_000,
                "output_tokens": 0,
                "cache_read_input_tokens": "400000",
            },
            UsageDialect.AUTO,
        ),
        (
            {
                "input_tokens": 500_000,
                "output_tokens": 0,
                "cache_creation_input_tokens": 100_000,
                "cache_creation": {"ephemeral_5m_input_tokens": 200_000},
            },
            UsageDialect.ANTHROPIC,
        ),
        (
            {
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "output_tokens_details": {"thinking_tokens": 1},
            },
            UsageDialect.GENERIC,
        ),
    ],
    ids=(
        "generic-alias-conflict",
        "openai-missing-input",
        "mixed-cache-conflict",
        "auto-malformed-input-without-details",
        "auto-ambiguous-matching-cache-counters",
        "auto-malformed-inferred-cache-read",
        "anthropic-cache-write-conflict",
        "generic-thinking-conflict",
    ),
)
def test_malformed_usage_is_unpriced_and_blocks_later_dispatch(
    raw_usage: dict[str, object],
    dialect: UsageDialect,
) -> None:
    class GenericGatewayProvider(FakeProvider):
        name = "company-gateway"
        usage_dialect = dialect

    provider = GenericGatewayProvider(
        [
            ModelStreamEvent.completed(
                {
                    "model": "gpt-test",
                    "usage": raw_usage,
                }
            )
        ]
    )
    pricing = PriceBook(
        prices=(
            ModelPrice.fixed(
                provider_name="company-gateway",
                model="gpt-test",
                input_per_million=Decimal("10"),
                output_per_million=Decimal("10"),
            ),
        )
    )
    store = InMemorySessionStore()
    budget_policy = BudgetPolicy(
        limits=(
            BudgetLimit(
                scope="app",
                max_estimated_cost=Decimal("100"),
                pricing=pricing,
            ),
        )
    )
    app = CayuApp(session_store=store, budget_policy=budget_policy)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="gpt-test"))

    first_events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_generic_usage_contradiction_first",
                messages=[Message.text("user", "first")],
            ),
        )
    )
    cost = asyncio.run(app.get_session_cost("sess_generic_usage_contradiction_first", pricing))
    restarted_app = CayuApp(session_store=store, budget_policy=budget_policy)
    restarted_app.register_provider(provider, default=True)
    restarted_app.register_agent(AgentSpec(name="assistant", model="gpt-test"))
    second_events = asyncio.run(
        collect_events(
            restarted_app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_generic_usage_contradiction_second",
                messages=[Message.text("user", "second")],
            ),
        )
    )

    completed = next(event for event in first_events if event.type == EventType.MODEL_COMPLETED)
    assert completed.payload["usage"] == raw_usage
    assert "usage_metrics" not in completed.payload
    assert completed.payload["usage_normalization_failed"] is True
    assert cost.priced_model_steps == 0
    assert cost.unpriced_model_steps == 1
    assert any(event.type == EventType.BUDGET_LIMIT_REACHED for event in first_events)
    assert any(event.type == EventType.BUDGET_LIMIT_REACHED for event in second_events)
    assert len(provider.requests) == 1


@pytest.mark.parametrize(
    ("registered_name", "declared_dialect", "raw_usage", "case_name"),
    [
        (
            "anthropic-gateway",
            UsageDialect.ANTHROPIC,
            {
                "input_tokens": 500_000,
                "output_tokens": 0,
                "cache_creation_input_tokens": 100_000,
                "cache_details": [{"ttl": "5m", "input_tokens": 200_000}],
            },
            "cache_write",
        ),
        (
            "generic-gateway",
            UsageDialect.GENERIC,
            {
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "output_tokens_details": {"thinking_tokens": 1},
            },
            "thinking",
        ),
    ],
    ids=("cache-write-conflict", "generic-thinking-conflict"),
)
def test_conflicting_usage_remains_unpriced_after_sqlite_reopen(
    tmp_path,
    registered_name: str,
    declared_dialect: UsageDialect,
    raw_usage: dict[str, object],
    case_name: str,
) -> None:
    class GatewayProvider(FakeProvider):
        name = registered_name
        usage_dialect = declared_dialect

    provider = GatewayProvider(
        [
            ModelStreamEvent.completed(
                {
                    "model": "test-model",
                    "usage": raw_usage,
                    "usage_metrics": {
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "total_tokens": 0,
                    },
                }
            )
        ]
    )
    pricing = PriceBook(
        prices=(
            ModelPrice.fixed(
                provider_name=registered_name,
                model="test-model",
                input_per_million=Decimal("10"),
                output_per_million=Decimal("10"),
            ),
        )
    )
    path = tmp_path / f"contradictory-{case_name}-usage.sqlite"
    session_id = f"sess_contradictory_{case_name}_usage_sqlite"

    async def run():
        store = SQLiteSessionStore(path)
        app = CayuApp(session_store=store)
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="test-model"))
        events = await collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "hello")],
            ),
        )
        await store.close()

        reopened = SQLiteSessionStore(path)
        try:
            durable_events = await reopened.load_events(session_id)
            reopened_app = CayuApp(session_store=reopened)
            cost = await reopened_app.get_session_cost(session_id, pricing)
            return events, durable_events, cost
        finally:
            await reopened.close()

    events, durable_events, cost = asyncio.run(run())

    completed = next(event for event in events if event.type == EventType.MODEL_COMPLETED)
    durable_completed = next(
        event for event in durable_events if event.type == EventType.MODEL_COMPLETED
    )
    assert completed.payload["usage"] == raw_usage
    assert "usage_metrics" not in completed.payload
    assert completed.payload["usage_normalization_failed"] is True
    assert durable_completed.payload["usage"] == raw_usage
    assert "usage_metrics" not in durable_completed.payload
    assert durable_completed.payload["usage_normalization_failed"] is True
    assert cost.priced_model_steps == 0
    assert cost.unpriced_model_steps == 1
    assert cost.total_cost == Decimal("0")


def test_cayu_app_strips_nested_provider_supplied_billing_identity_without_raw_usage() -> None:
    forged_identity = bedrock_billing_identity(
        invoked_model="global.anthropic.claude-sonnet-4-6",
        source_region="us-east-1",
        resource_type="inference_profile",
        profile_scope="global",
        effective_service_tier="default",
    ).model_dump(mode="json")
    provider = FakeProvider(
        [
            ModelStreamEvent.completed(
                {
                    "model": "fake-model",
                    "usage": None,
                    "usage_metrics": {
                        "provider_name": "fake",
                        "model": "fake-model",
                        "billing_identity": forged_identity,
                        "input_tokens": 250_000,
                        "output_tokens": 0,
                        "total_tokens": 250_000,
                    },
                }
            )
        ]
    )
    ledger = InMemoryBudgetLedger()
    limit = fake_budget_limit("10").model_copy(
        update={
            "scope": "app",
            "reservation": BudgetReservation(
                max_input_tokens=1_000_000,
                max_output_tokens=0,
            ),
        }
    )
    app = CayuApp(
        budget_policy=BudgetPolicy(limits=(limit,)),
        budget_ledger=ledger,
    )
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_untrusted_nested_billing_identity",
                messages=[Message.text("user", "hello")],
            ),
        )
    )

    completed = next(event for event in events if event.type == EventType.MODEL_COMPLETED)
    reconciled = next(event for event in events if event.type == EventType.BUDGET_RECONCILED)
    assert "billing_identity" not in completed.payload
    assert "billing_identity" not in completed.payload["usage_metrics"]
    assert reconciled.payload["actual_amount"] == "0.25"
    assert reconciled.payload["billing_identity"] is None
    assert events[-1].type == EventType.SESSION_COMPLETED


def test_runtime_preserves_raw_openai_usage_when_cache_counters_are_contradictory() -> None:
    class RenamedOpenAIProvider(FakeProvider):
        name = "renamed-openai"
        usage_dialect = UsageDialect.OPENAI

    raw_usage = {
        "input_tokens": 12,
        "input_tokens_details": {"cached_tokens": 5},
        "cache_read_input_tokens": 4,
        "output_tokens": 3,
    }
    provider = RenamedOpenAIProvider(
        [
            ModelStreamEvent.completed(
                {
                    "model": "fake-model-version",
                    "usage": raw_usage,
                    "usage_metrics": {
                        "input_tokens": 1,
                        "cache": {"uncached_input_tokens": 1},
                    },
                }
            )
        ]
    )
    app = CayuApp(enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="usage_runtime_contradictory_cache",
                messages=[Message.text("user", "hello")],
            ),
        )
    )

    completed = next(event for event in events if event.type == EventType.MODEL_COMPLETED)
    assert completed.payload["usage"] == raw_usage
    assert "usage_metrics" not in completed.payload
    assert completed.payload["usage_normalization_failed"] is True
    summary = asyncio.run(app.get_session_usage("usage_runtime_contradictory_cache"))
    assert summary.usage.input_tokens == 0
    assert summary.usage.cache.read_tokens == 0


@pytest.mark.parametrize("invalid_text", ["note\x00", "\ud800"], ids=["nul", "surrogate"])
def test_runtime_fails_closed_when_normalized_usage_loses_undurable_raw_evidence(
    invalid_text: str,
) -> None:
    class RenamedOpenAIProvider(FakeProvider):
        name = "actual-provider"
        usage_dialect = UsageDialect.OPENAI

    provider = RenamedOpenAIProvider(
        [
            ModelStreamEvent.text_delta("done"),
            ModelStreamEvent.completed(
                {
                    "provider_name": "provider-controlled-spoof",
                    "model": "served-model",
                    "usage": {
                        "input_tokens": 12,
                        "input_tokens_details": {"cached_tokens": 5},
                        "output_tokens": 3,
                        "provider_note": invalid_text,
                    },
                }
            ),
        ]
    )
    pricing = PriceBook(
        prices=(
            ModelPrice.fixed(
                provider_name="actual-provider",
                model="served-model",
                input_per_million=Decimal("1"),
                output_per_million=Decimal("1"),
            ),
        )
    )
    app = CayuApp(enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="requested-model"))
    session_id = f"usage-runtime-undurable-extra-{ord(invalid_text[-1])}"

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "hello")],
            ),
        )
    )
    completed = next(event for event in events if event.type == EventType.MODEL_COMPLETED)
    cost = asyncio.run(app.get_session_cost(session_id, pricing))

    assert "usage" not in completed.payload
    assert "usage_metrics" not in completed.payload
    assert completed.payload["usage_normalization_failed"] is True
    assert completed.payload["usage_unavailable_reason"] == (
        "invalid model completion usage telemetry"
    )
    assert completed.payload["provider_name"] == "actual-provider"
    assert completed.payload["requested_model"] == "requested-model"
    assert completed.payload["model"] == "served-model"
    assert cost.priced_model_steps == 0
    assert cost.unpriced_model_steps == 1
    assert cost.line_items[0].provider_name == "actual-provider"

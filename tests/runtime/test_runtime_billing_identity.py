"""Billing identity contracts."""

from __future__ import annotations

import asyncio
import json
from decimal import Decimal
from typing import Any

import pytest
from tests.runtime.runtime_budget_support import fake_budget_limit
from tests.runtime.runtime_provider_support import FakeProvider
from tests.runtime.runtime_session_support import collect_events

from cayu.agents import AgentSpec
from cayu.applications import CayuApp
from cayu.budgets.base import BudgetLimit, BudgetPolicy, BudgetReservation, InMemoryBudgetLedger
from cayu.budgets.billing import BillingIdentity, PricingContext
from cayu.budgets.pricing import ContextualPricingRequirement, ModelPrice, PriceBook
from cayu.context.base import CheckpointCompactionContextPolicy, ModelCompactor
from cayu.events import Event, EventType
from cayu.messages import Message
from cayu.providers.base import ModelProviderError, ModelRequest, ModelStreamEvent
from cayu.providers.bedrock import (
    BedrockProvider,
    bedrock_billing_identity,
    completed_bedrock_billing_identity,
)
from cayu.sessions.requests import RunRequest


def test_cayu_app_uses_bedrock_identity_for_budget_preflight_and_reconciliation() -> None:
    requested_identity = bedrock_billing_identity(
        invoked_model="global.anthropic.claude-sonnet-4-6",
        source_region="us-east-1",
        resource_type="inference_profile",
        profile_scope="global",
    )
    completed_identity = completed_bedrock_billing_identity(
        requested_identity,
        effective_service_tier="default",
    )

    class ContextualBillingProvider(FakeProvider):
        name = "bedrock"
        billing_provider_name = "bedrock"

        async def billing_identity_for_request(
            self,
            request: ModelRequest,
        ) -> BillingIdentity:
            assert request.model == requested_identity.resource_id
            return requested_identity

        def billing_identity_for_completion(
            self,
            identity: BillingIdentity | None,
            payload: dict[str, Any],
        ) -> BillingIdentity | None:
            assert identity == requested_identity
            assert payload["bedrock_service_tier"] == "default"
            return completed_identity

    provider = ContextualBillingProvider(
        [
            ModelStreamEvent.completed(
                {
                    "model": requested_identity.resource_id,
                    "usage": {"input_tokens": 500_000, "output_tokens": 100_000},
                    "bedrock_service_tier": "default",
                }
            )
        ]
    )
    pricing = PriceBook(
        prices=(
            ModelPrice.fixed(
                provider_name="bedrock",
                model=requested_identity.resource_id,
                match="exact",
                input_per_million=Decimal("3"),
                output_per_million=Decimal("15"),
                pricing_context={
                    "source_region": ("us-east-1",),
                    "service_tier": ("default",),
                },
            ),
        )
    )
    ledger = InMemoryBudgetLedger()
    app = CayuApp(
        budget_policy=BudgetPolicy(
            limits=(
                BudgetLimit(
                    scope="app",
                    max_estimated_cost=Decimal("20"),
                    pricing=pricing,
                    reservation=BudgetReservation(
                        max_input_tokens=1_000_000,
                        max_output_tokens=1_000_000,
                    ),
                ),
            )
        ),
        budget_ledger=ledger,
    )
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model=requested_identity.resource_id))

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_bedrock_budget_identity",
                messages=[Message.text("user", "hello")],
            ),
        )
    )

    assert len(provider.requests) == 1
    reserved = next(event for event in events if event.type == EventType.BUDGET_RESERVED)
    completed = next(event for event in events if event.type == EventType.MODEL_COMPLETED)
    reconciled = next(event for event in events if event.type == EventType.BUDGET_RECONCILED)
    assert reserved.payload["billing_identity"] == requested_identity.model_dump(mode="json")
    assert completed.payload["billing_identity"] == completed_identity.model_dump(mode="json")
    assert "billing_identity" not in completed.payload["usage_metrics"]
    assert len([event for event in events if event.type == EventType.BUDGET_CHECKED]) == 1
    assert Decimal(reconciled.payload["actual_amount"]) == Decimal("3")
    assert reconciled.payload["billing_identity"] == completed.payload["billing_identity"]
    assert next(iter(ledger._records.values())).billing_identity == completed_identity


def test_cayu_app_rejects_completion_billing_identity_rewrite_without_budget() -> None:
    requested_identity = bedrock_billing_identity(
        invoked_model="global.anthropic.claude-sonnet-4-6",
        source_region="us-east-1",
        resource_type="inference_profile",
        profile_scope="global",
    )

    class RewritingBillingProvider(FakeProvider):
        name = "bedrock"
        billing_provider_name = "bedrock"

        async def billing_identity_for_request(
            self,
            request: ModelRequest,
        ) -> BillingIdentity:
            return requested_identity

        def billing_identity_for_completion(
            self,
            identity: BillingIdentity | None,
            payload: dict[str, Any],
        ) -> BillingIdentity | None:
            return requested_identity.model_copy(
                update={
                    "request_evidence": {
                        **requested_identity.request_evidence,
                        "source_region": "eu-west-1",
                    }
                }
            )

    provider = RewritingBillingProvider(
        [
            ModelStreamEvent.completed(
                {
                    "model": requested_identity.resource_id,
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                }
            )
        ]
    )
    app = CayuApp()
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model=requested_identity.resource_id))

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_rewritten_completion_billing_identity",
                messages=[Message.text("user", "hello")],
            ),
        )
    )

    assert len(provider.requests) == 1
    completed = next(event for event in events if event.type == EventType.MODEL_COMPLETED)
    assert completed.payload["completion_outcome"] == "billing_identity_resolution_failed"
    assert completed.payload["usage_metrics"]["total_tokens"] == 2
    assert completed.payload["billing_identity"] == requested_identity.model_dump(mode="json")
    assert (
        completed.payload["completion_error"]["error"]
        == "Model provider billing identity resolution failed"
    )
    assert EventType.MODEL_ERROR not in [event.type for event in events]
    assert events[-1].type == EventType.SESSION_FAILED
    assert events[-1].payload["error"] == "Model provider billing identity resolution failed"


def test_cayu_app_rechecks_request_budget_after_resolving_bedrock_identity() -> None:
    identity = bedrock_billing_identity(
        invoked_model="global.anthropic.claude-sonnet-4-6",
        source_region="eu-west-1",
        resource_type="inference_profile",
        profile_scope="global",
    )

    class WrongRegionBedrockProvider(FakeProvider):
        name = "bedrock"
        billing_provider_name = "bedrock"

        async def billing_identity_for_request(
            self,
            request: ModelRequest,
        ) -> BillingIdentity:
            assert request.model == identity.resource_id
            return identity

    provider = WrongRegionBedrockProvider(
        [
            ModelStreamEvent.completed(
                {
                    "model": identity.resource_id,
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                }
            )
        ]
    )
    pricing = PriceBook(
        prices=(
            ModelPrice.fixed(
                provider_name="bedrock",
                model=identity.resource_id,
                match="exact",
                input_per_million=Decimal("3"),
                output_per_million=Decimal("15"),
                pricing_context={
                    "source_region": ("us-east-1",),
                    "service_tier": ("default",),
                },
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
                session_id="sess_bedrock_request_budget_identity",
                messages=[Message.text("user", "hello")],
                budget_limits=(
                    BudgetLimit(
                        scope="run",
                        max_estimated_cost=Decimal("20"),
                        pricing=pricing,
                    ),
                ),
            ),
        )
    )

    assert provider.requests == []
    reached = next(event for event in events if event.type == EventType.SESSION_LIMIT_REACHED)
    assert "no matching model pricing" in reached.payload["message"]
    assert events[-1].type == EventType.SESSION_INTERRUPTED


def test_cayu_app_rejects_unreservable_bedrock_tier_as_budget_limit() -> None:
    identity = bedrock_billing_identity(
        invoked_model="global.anthropic.claude-sonnet-4-6",
        source_region="us-east-1",
        resource_type="inference_profile",
        profile_scope="global",
        requested_service_tier="reserved",
    )

    class ReservedTierBedrockProvider(FakeProvider):
        name = "bedrock"
        billing_provider_name = "bedrock"

        async def billing_identity_for_request(
            self,
            request: ModelRequest,
        ) -> BillingIdentity:
            return identity

    provider = ReservedTierBedrockProvider(
        [
            ModelStreamEvent.completed(
                {
                    "model": identity.resource_id,
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                    "bedrock_service_tier": "default",
                }
            )
        ]
    )
    pricing = PriceBook(
        prices=(
            ModelPrice.fixed(
                provider_name="bedrock",
                model=identity.resource_id,
                match="exact",
                input_per_million=Decimal("3"),
                output_per_million=Decimal("15"),
                pricing_context={
                    "source_region": ("us-east-1",),
                    "service_tier": ("default",),
                },
            ),
        )
    )
    ledger = InMemoryBudgetLedger()
    app = CayuApp(
        budget_policy=BudgetPolicy(
            limits=(
                BudgetLimit(
                    scope="app",
                    max_estimated_cost=Decimal("20"),
                    pricing=pricing,
                    reservation=BudgetReservation(
                        max_input_tokens=1_000_000,
                        max_output_tokens=1_000_000,
                    ),
                ),
            )
        ),
        budget_ledger=ledger,
    )
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model=identity.resource_id))

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_unreservable_bedrock_tier",
                messages=[Message.text("user", "hello")],
            ),
        )
    )

    assert provider.requests == []
    assert ledger._records == {}
    reached = next(event for event in events if event.type == EventType.BUDGET_LIMIT_REACHED)
    assert "no matching model pricing" in reached.payload["message"]
    assert events[-1].type == EventType.SESSION_INTERRUPTED


def test_cayu_app_defers_request_budget_for_renamed_bedrock_provider() -> None:
    identity = bedrock_billing_identity(
        invoked_model="global.anthropic.claude-sonnet-4-6",
        source_region="us-east-1",
        resource_type="inference_profile",
        profile_scope="global",
    )

    class RenamedBedrockProvider(FakeProvider):
        name = "bedrock-us-east"
        billing_provider_name = "bedrock"

        async def billing_identity_for_request(
            self,
            request: ModelRequest,
        ) -> BillingIdentity:
            return identity

    provider = RenamedBedrockProvider(
        [
            ModelStreamEvent.completed(
                {
                    "model": identity.resource_id,
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                }
            )
        ]
    )
    pricing = PriceBook(
        prices=(
            ModelPrice.fixed(
                provider_name="bedrock",
                model=identity.resource_id,
                match="exact",
                input_per_million=Decimal("3"),
                output_per_million=Decimal("15"),
                pricing_context={
                    "source_region": ("us-east-1",),
                    "service_tier": ("default",),
                },
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
                session_id="sess_renamed_bedrock_request_budget",
                messages=[Message.text("user", "hello")],
                budget_limits=(
                    BudgetLimit(
                        scope="run",
                        max_estimated_cost=Decimal("20"),
                        pricing=pricing,
                    ),
                ),
            ),
        )
    )

    assert len(provider.requests) == 1
    assert events[-1].type == EventType.SESSION_COMPLETED


def test_cayu_app_prices_provider_neutral_contextual_identity() -> None:
    identity = BillingIdentity(
        provider_name="commercial-cloud",
        resource_id="opaque-resource-1",
        request_evidence={"tenant": "tenant-a"},
        pricing_contexts=(PricingContext(dimensions={"zone": "north"}),),
    )

    class ContextualCommercialProvider(FakeProvider):
        name = "renamed-commercial-cloud"
        billing_provider_name = "commercial-cloud"

        async def billing_identity_for_request(
            self,
            request: ModelRequest,
        ) -> BillingIdentity:
            assert request.model == identity.resource_id
            return identity

    provider = ContextualCommercialProvider(
        [
            ModelStreamEvent.completed(
                {
                    "model": identity.resource_id,
                    "usage": {"input_tokens": 1_000_000, "output_tokens": 1_000_000},
                }
            )
        ]
    )
    pricing = PriceBook(
        contextual_pricing_requirements=(
            ContextualPricingRequirement(
                provider_name="commercial-cloud",
                dimensions=("zone",),
            ),
        ),
        prices=(
            ModelPrice.fixed(
                provider_name="commercial-cloud",
                model=identity.resource_id,
                match="exact",
                pricing_context={"zone": ("north",)},
                input_per_million=Decimal("2"),
                output_per_million=Decimal("6"),
            ),
        ),
    )
    app = CayuApp()
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model=identity.resource_id))

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_neutral_contextual_price",
                messages=[Message.text("user", "hello")],
                budget_limits=(
                    BudgetLimit(
                        scope="app",
                        max_estimated_cost=Decimal("10"),
                        pricing=pricing,
                        reservation=BudgetReservation(
                            max_input_tokens=1_000_000,
                            max_output_tokens=1_000_000,
                        ),
                    ),
                ),
            ),
        )
    )

    completed = next(event for event in events if event.type == EventType.MODEL_COMPLETED)
    reconciled = next(event for event in events if event.type == EventType.BUDGET_RECONCILED)
    assert completed.payload["billing_identity"] == identity.model_dump(mode="json")
    assert reconciled.payload["actual_amount"] == "8"
    assert events[-1].type == EventType.SESSION_COMPLETED


def test_cayu_app_records_billing_identity_resolution_failure_as_model_error() -> None:
    class InvalidBillingIdentityProvider(FakeProvider):
        billing_provider_name = "bedrock"

        async def billing_identity_for_request(
            self,
            request: ModelRequest,
        ) -> BillingIdentity:
            raise ValueError("source region is unavailable")

    provider = InvalidBillingIdentityProvider(
        [ModelStreamEvent.completed({"usage": {"input_tokens": 1, "output_tokens": 0}})]
    )
    app = CayuApp()
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_billing_identity_hook_error",
                messages=[Message.text("user", "hello")],
            ),
        )
    )

    assert provider.requests == []
    model_error = next(event for event in events if event.type == EventType.MODEL_ERROR)
    assert model_error.payload["stage"] == "billing_identity_for_request"
    assert model_error.payload["provider_error_code"] == "billing_identity_resolution_failed"
    assert model_error.payload["retryable"] is False
    assert events[-1].type == EventType.SESSION_FAILED


@pytest.mark.parametrize("stage", ["request", "completion"])
@pytest.mark.parametrize("typed", [False, True], ids=["plain", "typed"])
def test_cayu_app_drops_custom_billing_hook_credentials(
    stage: str,
    typed: bool,
) -> None:
    canary = f"provider-custom-billing-{stage}-{typed}-canary-0123456789"

    def credential_failure() -> Exception:
        if typed:
            error: Exception = ModelProviderError(
                f"credential resolver failed near {canary}",
                provider=canary,
                status_code=401,
                error_type=canary,
                error_code=canary,
                request_id=canary,
                retryable=False,
                response_body=canary,
            )
        else:
            error = RuntimeError(f"credential resolver failed near {canary}")
        error.headers = {"Authorization": canary}  # type: ignore[attr-defined]
        error.add_note(f"resolver retained {canary}")
        return error

    class FailingBillingProvider(FakeProvider):
        async def billing_identity_for_request(
            self,
            request: ModelRequest,
        ) -> BillingIdentity | None:
            if stage == "request":
                raise credential_failure()
            return None

        def billing_identity_for_completion(
            self,
            identity: BillingIdentity | None,
            payload: dict[str, Any],
        ) -> BillingIdentity | None:
            if stage == "completion":
                raise credential_failure()
            return identity

    provider = FailingBillingProvider(
        [ModelStreamEvent.completed({"usage": {"input_tokens": 1, "output_tokens": 0}})]
    )
    app = CayuApp(enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id=f"sess_custom_billing_{stage}_{typed}",
                messages=[Message.text("user", "hello")],
            ),
        )
    )

    retained = json.dumps([event.model_dump(mode="json") for event in events], sort_keys=True)
    assert canary not in retained
    if stage == "request":
        model_error = next(event for event in events if event.type == EventType.MODEL_ERROR)
        error_payload = model_error.payload
    else:
        completed = next(event for event in events if event.type == EventType.MODEL_COMPLETED)
        assert completed.payload["completion_outcome"] == "billing_identity_resolution_failed"
        assert completed.payload["usage_metrics"]["input_tokens"] == 1
        assert completed.payload["usage_metrics"]["output_tokens"] == 0
        assert EventType.MODEL_ERROR not in {event.type for event in events}
        error_payload = completed.payload["completion_error"]
    assert error_payload["stage"] == f"billing_identity_for_{stage}"
    assert error_payload["error"] == "Model provider billing identity resolution failed"
    assert error_payload["provider"] == "fake"
    assert error_payload["provider_error_type"] == "BillingIdentityResolutionError"
    assert error_payload["provider_error_code"] == "billing_identity_resolution_failed"
    assert events[-1].type == EventType.SESSION_FAILED


@pytest.mark.parametrize("stage", ["request", "completion"])
def test_cayu_app_classifies_provider_created_billing_cancellation_as_failure(
    stage: str,
) -> None:
    canary = f"provider-custom-billing-cancel-{stage}-canary-0123456789"

    def credential_cancellation() -> asyncio.CancelledError:
        cancellation = asyncio.CancelledError(f"cancelled near {canary}")
        cancellation.headers = {"Authorization": canary}  # type: ignore[attr-defined]
        cancellation.add_note(f"resolver retained {canary}")
        return cancellation

    class CancellingBillingProvider(FakeProvider):
        def __repr__(self) -> str:
            return f"CancellingBillingProvider(api_key={canary!r})"

        async def billing_identity_for_request(
            self,
            request: ModelRequest,
        ) -> BillingIdentity | None:
            if stage == "request":
                raise credential_cancellation()
            return None

        def billing_identity_for_completion(
            self,
            identity: BillingIdentity | None,
            payload: dict[str, Any],
        ) -> BillingIdentity | None:
            if stage == "completion":
                raise credential_cancellation()
            return identity

    provider = CancellingBillingProvider(
        [ModelStreamEvent.completed({"usage": {"input_tokens": 1, "output_tokens": 0}})]
    )
    app = CayuApp(enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id=f"sess_custom_billing_cancel_{stage}",
                messages=[Message.text("user", "hello")],
            ),
        )
    )

    retained = json.dumps([event.model_dump(mode="json") for event in events], sort_keys=True)
    assert canary not in retained
    assert events[-1].type == EventType.SESSION_FAILED
    if stage == "request":
        failure_payload = next(
            event.payload for event in events if event.type == EventType.MODEL_ERROR
        )
    else:
        completed = next(event for event in events if event.type == EventType.MODEL_COMPLETED)
        failure_payload = completed.payload["completion_error"]
    assert failure_payload["error"] == "Model provider billing identity resolution failed"
    assert failure_payload["provider_error_type"] == "BillingIdentityResolutionError"


def test_cayu_app_classifies_grouped_provider_cancellation_without_task_request_as_failure() -> (
    None
):
    canary = "provider-grouped-synthetic-cancellation-canary"

    class GroupedCancellingBillingProvider(FakeProvider):
        def __init__(self) -> None:
            super().__init__([])
            self.cancellation_count: int | None = None

        async def billing_identity_for_request(
            self,
            request: ModelRequest,
        ) -> BillingIdentity | None:
            del request
            task = asyncio.current_task()
            self.cancellation_count = None if task is None else task.cancelling()
            raise BaseExceptionGroup(
                "provider group",
                [asyncio.CancelledError(f"synthetic cancellation near {canary}")],
            )

    provider = GroupedCancellingBillingProvider()
    app = CayuApp(enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_grouped_synthetic_billing_cancel",
                messages=[Message.text("user", "hello")],
            ),
        )
    )

    assert provider.cancellation_count == 0
    assert events[-1].type == EventType.SESSION_FAILED
    model_error = next(event for event in events if event.type == EventType.MODEL_ERROR)
    assert model_error.payload["error"] == "Model provider billing identity resolution failed"
    retained = json.dumps(
        [event.model_dump(mode="json") for event in events],
        sort_keys=True,
    )
    assert canary not in retained


def test_cayu_app_does_not_authenticate_historical_cancellation_as_billing_authority() -> None:
    class CancellingBillingProvider(FakeProvider):
        def __init__(self) -> None:
            super().__init__([])
            self.child_cancellation_count: int | None = None

        async def billing_identity_for_request(
            self,
            request: ModelRequest,
        ) -> BillingIdentity | None:
            del request
            task = asyncio.current_task()
            self.child_cancellation_count = None if task is None else task.cancelling()
            raise asyncio.CancelledError("provider-created cancellation")

    provider = CancellingBillingProvider()
    app = CayuApp(enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    async def run() -> tuple[list[Event], int, bool]:
        task = asyncio.current_task()
        assert task is not None
        task.cancel("already handled")
        with pytest.raises(asyncio.CancelledError):
            await asyncio.sleep(0)
        assert task.cancelling() == 1

        events = await collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_historical_billing_cancellation",
                messages=[Message.text("user", "hello")],
            ),
        )
        await asyncio.sleep(0)
        return events, task.cancelling(), task.cancelled()

    events, cancellation_count, cancelled = asyncio.run(run())

    assert provider.child_cancellation_count == 0
    assert cancellation_count == 1
    assert cancelled is False
    assert events[-1].type == EventType.SESSION_FAILED
    assert EventType.SESSION_INTERRUPTED not in {event.type for event in events}


def test_cayu_app_never_persists_bedrock_credential_resolution_failure() -> None:
    canary = "provider-bedrock-billing-canary-0123456789"

    class FailingBillingProvider(BedrockProvider):
        async def _get_client(self) -> Any:
            error = RuntimeError(f"AWS credential resolution failed near {canary}")
            error.headers = {"Authorization": canary}
            error.add_note(f"resolver retained {canary}")
            raise error

    app = CayuApp(enable_logging=False)
    app.register_provider(FailingBillingProvider(), default=True)
    app.register_agent(AgentSpec(name="assistant", model="anthropic.claude-test"))

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_bedrock_billing_credential_failure",
                messages=[Message.text("user", "hello")],
            ),
        )
    )

    retained = json.dumps([event.model_dump(mode="json") for event in events], sort_keys=True)
    assert canary not in retained
    model_error = next(event for event in events if event.type == EventType.MODEL_ERROR)
    assert model_error.payload["stage"] == "billing_identity_for_request"
    assert model_error.payload["error"] == "Model provider billing identity resolution failed"
    assert model_error.payload["provider_error_code"] == "billing_identity_resolution_failed"
    assert events[-1].type == EventType.SESSION_FAILED


def test_cayu_app_contextual_budget_fails_closed_when_provider_has_no_identity() -> None:
    model = "global.anthropic.claude-sonnet-4-6"
    provider = FakeProvider(
        [
            ModelStreamEvent.completed(
                {"model": model, "usage": {"input_tokens": 1, "output_tokens": 1}}
            )
        ]
    )
    pricing = PriceBook(
        prices=(
            ModelPrice.fixed(
                provider_name="bedrock",
                model=model,
                match="exact",
                input_per_million=Decimal("3"),
                output_per_million=Decimal("15"),
                pricing_context={
                    "source_region": ("us-east-1",),
                    "service_tier": ("default",),
                },
            ),
        )
    )
    app = CayuApp()
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model=model))

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_contextual_budget_without_identity",
                messages=[Message.text("user", "hello")],
                budget_limits=(
                    BudgetLimit(
                        scope="run",
                        max_estimated_cost=Decimal("20"),
                        pricing=pricing,
                    ),
                ),
            ),
        )
    )

    assert provider.requests == []
    assert EventType.SESSION_LIMIT_REACHED in [event.type for event in events]
    assert events[-1].type == EventType.SESSION_INTERRUPTED


def test_cayu_app_rejects_unresolved_contextual_budget_before_compaction() -> None:
    model = "global.anthropic.claude-sonnet-4-6"
    compactor_provider = FakeProvider(
        [
            ModelStreamEvent.text_delta("summary"),
            ModelStreamEvent.completed(
                {"model": "summary-model", "usage": {"input_tokens": 1, "output_tokens": 1}}
            ),
        ]
    )
    runtime_provider = FakeProvider(
        [
            ModelStreamEvent.completed(
                {"model": model, "usage": {"input_tokens": 1, "output_tokens": 1}}
            )
        ]
    )
    pricing = PriceBook(
        prices=(
            ModelPrice.fixed(
                provider_name="bedrock",
                model=model,
                match="exact",
                input_per_million=Decimal("3"),
                output_per_million=Decimal("15"),
                pricing_context={
                    "source_region": ("us-east-1",),
                    "service_tier": ("default",),
                },
            ),
        )
    )
    app = CayuApp()
    app.register_provider(runtime_provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model=model),
        context_policy=CheckpointCompactionContextPolicy(
            compactor=ModelCompactor(provider=compactor_provider, model="summary-model"),
            max_user_turns=1,
            compact_after_messages=2,
        ),
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_unresolved_budget_before_compaction",
                messages=[
                    Message.text("user", "old"),
                    Message.text("assistant", "old answer"),
                    Message.text("user", "current"),
                ],
                budget_limits=(
                    BudgetLimit(
                        scope="run",
                        max_estimated_cost=Decimal("20"),
                        pricing=pricing,
                    ),
                ),
            ),
        )
    )

    assert compactor_provider.requests == []
    assert runtime_provider.requests == []
    assert EventType.CONTEXT_COMPACTION_STARTED not in [event.type for event in events]
    assert events[-1].type == EventType.SESSION_INTERRUPTED


@pytest.mark.parametrize(
    "raw_billing_identity",
    [
        bedrock_billing_identity(
            invoked_model="global.anthropic.claude-sonnet-4-6",
            source_region="us-east-1",
            resource_type="inference_profile",
            profile_scope="global",
            effective_service_tier="default",
        ).model_dump(mode="json"),
        {"provider": "bedrock"},
    ],
    ids=("valid-looking", "malformed"),
)
def test_cayu_app_strips_provider_supplied_billing_identity(
    raw_billing_identity: dict[str, Any],
) -> None:
    provider = FakeProvider(
        [
            ModelStreamEvent.completed(
                {
                    "model": "fake-model",
                    "usage": {"input_tokens": 250_000, "output_tokens": 0},
                    "billing_identity": raw_billing_identity,
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
                session_id="sess_untrusted_billing_identity",
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
    assert next(iter(ledger._records.values())).billing_identity is None
    assert events[-1].type == EventType.SESSION_COMPLETED

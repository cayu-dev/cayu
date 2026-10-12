"""Compaction billing contracts."""

from __future__ import annotations

import asyncio
import json
import traceback
from decimal import Decimal
from typing import Any

import pytest
from tests.provider_traceback_assertions import is_cayu_source_filename
from tests.runtime.runtime_compaction_support import compaction_price_book
from tests.runtime.runtime_environment_support import RecordingWorkspaceBinding
from tests.runtime.runtime_provider_support import FakeProvider
from tests.runtime.runtime_session_support import (
    _expected_failure_evidence,
    _test_session,
    collect_events,
)

from cayu.agents import AgentSpec
from cayu.applications import CayuApp
from cayu.budgets.base import BudgetLimit, BudgetPolicy, BudgetReservation, InMemoryBudgetLedger
from cayu.budgets.billing import BillingIdentity
from cayu.budgets.pricing import ModelPrice, PriceBook
from cayu.budgets.run_limits import RunLimits
from cayu.context.base import CheckpointCompactionContextPolicy, CompactionRequest, ModelCompactor
from cayu.environments.base import Environment, EnvironmentSpec
from cayu.events import EventType
from cayu.messages import Message
from cayu.providers.base import ModelProviderError, ModelRequest, ModelStreamEvent
from cayu.providers.bedrock import bedrock_billing_identity, completed_bedrock_billing_identity
from cayu.runtime._model_errors import (
    _BillingIdentityResolutionCancelled,
    detach_billing_identity_cancellation_group,
)
from cayu.sessions.records import Session
from cayu.sessions.requests import RunRequest


@pytest.mark.parametrize("with_reservation", [False, True])
def test_automatic_compaction_uses_its_bedrock_billing_identity(
    with_reservation: bool,
) -> None:
    requested_identity = bedrock_billing_identity(
        invoked_model="global.anthropic.claude-sonnet-4-6",
        source_region="us-east-1",
        resource_type="inference_profile",
        profile_scope="global",
        requested_service_tier="reserved",
    )
    completed_identity = completed_bedrock_billing_identity(
        requested_identity,
        effective_service_tier="default",
    )

    class ContextualCompactionProvider(FakeProvider):
        name = "bedrock"

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

    compactor_provider = ContextualCompactionProvider(
        [
            ModelStreamEvent.text_delta("model summary"),
            ModelStreamEvent.completed(
                {
                    "model": requested_identity.resource_id,
                    "usage": {"input_tokens": 1_000_000, "output_tokens": 0},
                    "bedrock_service_tier": "default",
                }
            ),
        ]
    )
    runtime_provider = FakeProvider(
        [
            ModelStreamEvent.completed(
                {
                    "model": "fake-model",
                    "usage": {"input_tokens": 1, "output_tokens": 0},
                }
            )
        ]
    )
    pricing = PriceBook(
        prices=(
            ModelPrice.fixed(
                provider_name="fake",
                model="fake-model",
                input_per_million=Decimal("1"),
                output_per_million=Decimal("1"),
            ),
            ModelPrice.fixed(
                provider_name="bedrock",
                model=requested_identity.resource_id,
                match="exact",
                input_per_million=Decimal("2"),
                output_per_million=Decimal("1"),
                pricing_context={
                    "source_region": ("us-east-1",),
                    "service_tier": ("reserved",),
                },
            ),
            ModelPrice.fixed(
                provider_name="bedrock",
                model=requested_identity.resource_id,
                match="exact",
                input_per_million=Decimal("3"),
                output_per_million=Decimal("1"),
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
                    reservation=(
                        BudgetReservation(
                            max_input_tokens=1_000_000,
                            max_output_tokens=0,
                        )
                        if with_reservation
                        else None
                    ),
                ),
            )
        ),
        budget_ledger=ledger,
    )
    app.register_provider(runtime_provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        context_policy=CheckpointCompactionContextPolicy(
            compactor=ModelCompactor(
                provider=compactor_provider,
                model=requested_identity.resource_id,
                options={"bedrock": {"serviceTier": {"type": "reserved"}}},
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
                session_id=f"sess_bedrock_compaction_identity_{with_reservation}",
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
    assert completed.payload["billing_identity"] == completed_identity.model_dump(mode="json")
    assert "billing_identity" not in completed.payload["usage_metrics"]
    assert events[-1].type == EventType.SESSION_COMPLETED
    if with_reservation:
        reserved = next(
            event
            for event in events
            if event.type == EventType.BUDGET_RESERVED
            and event.payload["model"] == requested_identity.resource_id
        )
        reconciled = next(
            event
            for event in events
            if event.type == EventType.BUDGET_RECONCILED
            and event.payload["reservation_id"] == reserved.payload["reservation_id"]
        )
        assert Decimal(reserved.payload["requested"]) == Decimal("3")
        assert reserved.payload["billing_identity"] == requested_identity.model_dump(mode="json")
        assert Decimal(reconciled.payload["actual_amount"]) == Decimal("3")
        assert reconciled.payload["billing_identity"] == completed.payload["billing_identity"]
        assert (
            next(
                record
                for record in ledger._records.values()
                if record.model == requested_identity.resource_id
            ).billing_identity
            == completed_identity
        )
    else:
        assert EventType.BUDGET_RESERVED not in [event.type for event in events]
        assert EventType.BUDGET_RECONCILED not in [event.type for event in events]
        assert ledger._records == {}


@pytest.mark.parametrize("scope", ["app", "run"])
def test_automatic_contextual_budget_counts_prior_hierarchy_dispatches(scope) -> None:
    requested_identity = bedrock_billing_identity(
        invoked_model="global.anthropic.claude-sonnet-4-6",
        source_region="us-east-1",
        resource_type="inference_profile",
        profile_scope="global",
        requested_service_tier="default",
    )
    completed_identity = completed_bedrock_billing_identity(
        requested_identity,
        effective_service_tier="default",
    )

    class ContextualHierarchyProvider(FakeProvider):
        name = "bedrock-adapter"
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
            assert identity == requested_identity
            return completed_identity

    completed_batch = [
        ModelStreamEvent.text_delta("short summary"),
        ModelStreamEvent.completed(
            {
                "model": requested_identity.resource_id,
                "usage": {"input_tokens": 1, "output_tokens": 0},
            }
        ),
    ]
    compactor_provider = ContextualHierarchyProvider([completed_batch for _index in range(8)])
    runtime_provider = FakeProvider(
        [ModelStreamEvent.completed({"usage": {"input_tokens": 1, "output_tokens": 0}})]
    )
    pricing = PriceBook(
        prices=(
            ModelPrice.fixed(
                provider_name="fake",
                model="fake-model",
                input_per_million=Decimal("1"),
                output_per_million=Decimal("1"),
            ),
            ModelPrice.fixed(
                provider_name="bedrock",
                model=requested_identity.resource_id,
                match="exact",
                input_per_million=Decimal("3"),
                output_per_million=Decimal("1"),
                pricing_context={
                    "source_region": ("us-east-1",),
                    "service_tier": ("default",),
                },
            ),
        )
    )
    app = CayuApp(
        budget_policy=BudgetPolicy(
            limits=(
                BudgetLimit(
                    scope="app",
                    max_estimated_cost=Decimal("0.000004"),
                    pricing=pricing,
                ),
            )
            if scope == "app"
            else ()
        )
    )
    app.register_provider(runtime_provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        context_policy=CheckpointCompactionContextPolicy(
            compactor=ModelCompactor(
                provider=compactor_provider,
                model=requested_identity.resource_id,
                max_input_chars=1000,
            ),
            max_user_turns=1,
            compact_after_messages=1,
        ),
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_automatic_contextual_hierarchy_budget",
                budget_limits=(
                    BudgetLimit(
                        scope="run", max_estimated_cost=Decimal("0.000004"), pricing=pricing
                    ),
                )
                if scope == "run"
                else (),
                messages=[
                    Message.text("user", "oversized " + "x" * 5000),
                    Message.text("user", "current"),
                ],
            ),
        )
    )

    assert len(compactor_provider.requests) == 2
    assert runtime_provider.requests == []
    completions = [
        event
        for event in events
        if event.type == EventType.MODEL_COMPLETED
        and event.payload.get("purpose") == "context_compaction"
    ]
    assert len(completions) == 2
    assert EventType.BUDGET_RESERVATION_FAILED in {event.type for event in events}
    assert events[-1].type == EventType.SESSION_INTERRUPTED


def test_automatic_hierarchy_enforces_run_limit_before_every_dispatch() -> None:
    compactor_provider = FakeProvider(
        [
            [
                ModelStreamEvent.text_delta("short summary"),
                ModelStreamEvent.completed(
                    {
                        "model": "summary-model",
                        "usage": {
                            "input_tokens": 7,
                            "output_tokens": 3,
                            "total_tokens": 10,
                        },
                    }
                ),
            ]
        ]
    )
    runtime_provider = FakeProvider([ModelStreamEvent.completed({})])
    app = CayuApp(enable_logging=False)
    app.register_provider(runtime_provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        context_policy=CheckpointCompactionContextPolicy(
            compactor=ModelCompactor(
                provider=compactor_provider,
                model="summary-model",
                max_input_chars=1000,
            ),
            max_user_turns=1,
            compact_after_messages=1,
        ),
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_automatic_hierarchy_run_limit",
                messages=[
                    Message.text("user", "oversized " + "x" * 5000),
                    Message.text("user", "current"),
                ],
                limits=RunLimits(max_total_tokens=10),
            ),
        )
    )

    assert len(compactor_provider.requests) == 1
    assert runtime_provider.requests == []
    assert (
        sum(
            event.type == EventType.MODEL_COMPLETED
            and event.payload.get("purpose") == "context_compaction"
            for event in events
        )
        == 1
    )
    assert EventType.SESSION_LIMIT_REACHED in {event.type for event in events}
    assert events[-1].type == EventType.SESSION_INTERRUPTED


def test_automatic_hierarchy_enforces_static_budget_before_every_dispatch() -> None:
    compactor_provider = FakeProvider(
        [
            [
                ModelStreamEvent.text_delta("short summary"),
                ModelStreamEvent.completed(
                    {
                        "model": "summary-model",
                        "usage": {
                            "input_tokens": 10,
                            "output_tokens": 0,
                            "total_tokens": 10,
                        },
                    }
                ),
            ]
        ]
    )
    runtime_provider = FakeProvider([ModelStreamEvent.completed({})])
    app = CayuApp(
        enable_logging=False,
        budget_policy=BudgetPolicy(
            limits=(
                BudgetLimit(
                    scope="app",
                    max_estimated_cost=Decimal("0.000010"),
                    pricing=compaction_price_book(),
                ),
            )
        ),
    )
    app.register_provider(runtime_provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        context_policy=CheckpointCompactionContextPolicy(
            compactor=ModelCompactor(
                provider=compactor_provider,
                model="summary-model",
                max_input_chars=1000,
            ),
            max_user_turns=1,
            compact_after_messages=1,
        ),
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_automatic_hierarchy_static_budget",
                messages=[
                    Message.text("user", "oversized " + "x" * 5000),
                    Message.text("user", "current"),
                ],
            ),
        )
    )

    assert len(compactor_provider.requests) == 1
    assert runtime_provider.requests == []
    assert (
        sum(
            event.type == EventType.MODEL_COMPLETED
            and event.payload.get("purpose") == "context_compaction"
            for event in events
        )
        == 1
    )
    assert EventType.BUDGET_LIMIT_REACHED in {event.type for event in events}
    assert events[-1].type == EventType.SESSION_INTERRUPTED


def test_automatic_compaction_request_budget_prices_compactor_model_before_dispatch() -> None:
    compactor_provider = FakeProvider(
        [
            ModelStreamEvent.text_delta("must not be dispatched"),
            ModelStreamEvent.completed(
                {
                    "model": "summary-model",
                    "usage": {"input_tokens": 10, "output_tokens": 0},
                }
            ),
        ]
    )
    runtime_provider = FakeProvider([ModelStreamEvent.completed({})])
    session_only_pricing = PriceBook(
        prices=(
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
            compact_after_messages=1,
        ),
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_compaction_request_budget_model_preflight",
                messages=[
                    Message.text("user", "old"),
                    Message.text("assistant", "old answer"),
                    Message.text("user", "current"),
                ],
                budget_limits=(
                    BudgetLimit(
                        scope="run",
                        max_estimated_cost=Decimal("1"),
                        pricing=session_only_pricing,
                    ),
                ),
            ),
        )
    )

    assert compactor_provider.requests == []
    assert runtime_provider.requests == []
    limit_event = next(event for event in events if event.type == EventType.SESSION_LIMIT_REACHED)
    assert limit_event.payload["message"] == (
        "Budget cannot be verified because fake/summary-model: no matching model pricing."
    )
    assert EventType.MODEL_COMPLETED not in {event.type for event in events}
    assert events[-1].type == EventType.SESSION_INTERRUPTED


@pytest.mark.parametrize("stage", ["request", "completion"])
@pytest.mark.parametrize("typed", [False, True], ids=["plain", "typed"])
def test_model_compactor_drops_custom_billing_hook_credentials(
    stage: str,
    typed: bool,
) -> None:
    canary = f"provider-compaction-billing-{stage}-{typed}-canary-0123456789"

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

    class FailingCompactionProvider(FakeProvider):
        def __repr__(self) -> str:
            return f"FailingCompactionProvider(api_key={canary!r})"

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

    provider = FailingCompactionProvider(
        [
            ModelStreamEvent.text_delta("summary"),
            ModelStreamEvent.completed({"usage": {"input_tokens": 1, "output_tokens": 1}}),
        ]
    )
    compactor = ModelCompactor(provider=provider, model="summary-model")

    with pytest.raises(ModelProviderError) as exc_info:
        asyncio.run(
            compactor.compact(
                CompactionRequest(
                    session=_test_session(),
                    agent=AgentSpec(name="assistant", model="fake-model"),
                    messages=[Message.text("user", "old request")],
                )
            )
        )

    retained = str(exc_info.value) + repr(exc_info.value) + repr(vars(exc_info.value))
    assert canary not in retained
    assert str(exc_info.value) == "Model provider billing identity resolution failed"
    assert exc_info.value.provider == "fake"
    assert exc_info.value.error_type == "BillingIdentityResolutionError"
    assert exc_info.value.error_code == "billing_identity_resolution_failed"
    captured = traceback.TracebackException.from_exception(
        exc_info.value,
        capture_locals=True,
    )
    cayu_frames = [frame for frame in captured.stack if is_cayu_source_filename(frame.filename)]
    leaking_locals = [
        (frame.name, name)
        for frame in cayu_frames
        for name, value in (frame.locals or {}).items()
        if canary in value
    ]
    assert leaking_locals == []
    assert all(not frame.name.startswith("billing_identity_for_") for frame in captured.stack)
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None


@pytest.mark.parametrize("stage", ["request", "completion"])
def test_model_compactor_drops_custom_billing_hook_cancellation_credentials(
    stage: str,
) -> None:
    canary = f"provider-compaction-billing-cancel-{stage}-canary-0123456789"

    class CancellingCompactionProvider(FakeProvider):
        def __repr__(self) -> str:
            return f"CancellingCompactionProvider(api_key={canary!r})"

        async def billing_identity_for_request(
            self,
            request: ModelRequest,
        ) -> BillingIdentity | None:
            if stage == "request":
                raise asyncio.CancelledError(f"cancelled near {canary}")
            return None

        def billing_identity_for_completion(
            self,
            identity: BillingIdentity | None,
            payload: dict[str, Any],
        ) -> BillingIdentity | None:
            if stage == "completion":
                raise asyncio.CancelledError(f"cancelled near {canary}")
            return identity

    provider = CancellingCompactionProvider(
        [
            ModelStreamEvent.text_delta("summary"),
            ModelStreamEvent.completed({"usage": {"input_tokens": 1, "output_tokens": 1}}),
        ]
    )

    with pytest.raises(ModelProviderError) as exc_info:
        asyncio.run(
            ModelCompactor(provider=provider, model="summary-model").compact(
                CompactionRequest(
                    session=_test_session(),
                    agent=AgentSpec(name="assistant", model="fake-model"),
                    messages=[Message.text("user", "old request")],
                )
            )
        )

    retained = str(exc_info.value) + repr(exc_info.value) + repr(vars(exc_info.value))
    assert str(exc_info.value) == "Model provider billing identity resolution failed"
    assert exc_info.value.provider == "fake"
    assert exc_info.value.error_type == "BillingIdentityResolutionError"
    assert exc_info.value.error_code == "billing_identity_resolution_failed"
    assert exc_info.value.retryable is False
    assert canary not in retained
    captured = traceback.TracebackException.from_exception(
        exc_info.value,
        capture_locals=True,
    )
    cayu_frames = [frame for frame in captured.stack if is_cayu_source_filename(frame.filename)]
    leaking_locals = [
        (frame.name, name)
        for frame in cayu_frames
        for name, value in (frame.locals or {}).items()
        if canary in value
    ]
    assert leaking_locals == []
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None


@pytest.mark.parametrize("stage", ["request", "completion"])
def test_cayu_app_drops_automatic_compaction_billing_cancellation_credentials(
    stage: str,
) -> None:
    canary = f"automatic-compaction-billing-cancel-{stage}-canary-0123456789"

    class CancellingAutomaticCompactionProvider(FakeProvider):
        def __repr__(self) -> str:
            return f"CancellingAutomaticCompactionProvider(api_key={canary!r})"

        async def billing_identity_for_request(
            self,
            request: ModelRequest,
        ) -> BillingIdentity | None:
            if stage == "request":
                raise asyncio.CancelledError(f"cancelled near {canary}")
            return None

        def billing_identity_for_completion(
            self,
            identity: BillingIdentity | None,
            payload: dict[str, Any],
        ) -> BillingIdentity | None:
            if stage == "completion":
                raise asyncio.CancelledError(f"cancelled near {canary}")
            return identity

    provider = CancellingAutomaticCompactionProvider(
        [
            ModelStreamEvent.text_delta("summary"),
            ModelStreamEvent.completed({"usage": {"input_tokens": 1, "output_tokens": 1}}),
        ]
    )
    app = CayuApp(enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        context_policy=CheckpointCompactionContextPolicy(
            compactor=ModelCompactor(provider=provider, model="summary-model"),
            max_user_turns=1,
            compact_after_messages=2,
        ),
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id=f"sess_automatic_compaction_billing_cancel_{stage}",
                messages=[
                    Message.text("user", "old"),
                    Message.text("assistant", "old answer"),
                    Message.text("user", "current"),
                ],
            ),
        )
    )

    retained = json.dumps([event.model_dump(mode="json") for event in events], sort_keys=True)
    assert canary not in retained
    assert events[-1].type is EventType.SESSION_FAILED
    assert EventType.SESSION_INTERRUPTED not in {event.type for event in events}
    expected_terminal: dict[str, Any] = {
        "failure_evidence": _expected_failure_evidence(
            f"sess_automatic_compaction_billing_cancel_{stage}",
            "ModelProviderError",
            "ContextBuildError",
        ),
        "error": "Model provider billing identity resolution failed",
        "error_type": "ModelProviderError",
    }
    if stage == "completion":
        failure = next(
            event for event in events if event.type == EventType.CONTEXT_COMPACTION_FAILED
        )
        expected_disposition = {
            "phase": "provider_dispatch",
            "reason": "internal_failed",
            "retryable": False,
            "provider_dispatch_disposition": "unknown",
            "recovery_action": "reconcile_completion",
        }
        assert {key: failure.payload[key] for key in expected_disposition} == expected_disposition
        assert type(failure.payload["elapsed_ms"]) is int
        expected_terminal["compaction_failure"] = {
            **expected_disposition,
            "elapsed_ms": failure.payload["elapsed_ms"],
        }
    assert events[-1].payload == expected_terminal


def test_cayu_app_detaches_grouped_compaction_billing_cancellation_credentials() -> None:
    provider_canary = "automatic-compaction-grouped-cancel-canary-0123456789"
    cleanup_marker = "automatic compaction workspace cleanup failed"

    class CancellingCompactionProvider(FakeProvider):
        def __repr__(self) -> str:
            return f"CancellingCompactionProvider(api_key={provider_canary!r})"

        async def billing_identity_for_request(
            self,
            request: ModelRequest,
        ) -> BillingIdentity | None:
            raise asyncio.CancelledError(f"cancelled near {provider_canary}")

    provider = CancellingCompactionProvider([])
    binding = RecordingWorkspaceBinding(
        fail_finalize=True,
        finalize_error=RuntimeError(cleanup_marker),
    )
    app = CayuApp(enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_environment(
        Environment(EnvironmentSpec(name="local"), binding=binding),
        default=True,
    )
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        context_policy=CheckpointCompactionContextPolicy(
            compactor=ModelCompactor(provider=provider, model="summary-model"),
            max_user_turns=1,
            compact_after_messages=2,
        ),
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_automatic_compaction_grouped_billing_cancel",
                messages=[
                    Message.text("user", "old"),
                    Message.text("assistant", "old answer"),
                    Message.text("user", "current"),
                ],
            ),
        )
    )

    retained = json.dumps([event.model_dump(mode="json") for event in events], sort_keys=True)
    assert provider_canary not in retained
    assert cleanup_marker in retained
    assert events[-1].type is EventType.SESSION_FAILED
    assert EventType.SESSION_INTERRUPTED not in {event.type for event in events}
    assert events[-1].payload["error"] == "Model provider billing identity resolution failed"
    assert events[-1].payload["error_type"] == "ModelProviderError"


@pytest.mark.parametrize("grouped", [False, True])
def test_cayu_app_detaches_billing_cancellation_before_fallible_interruption_checks(
    grouped: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    canary = f"billing-cancellation-transition-failure-{grouped}-canary-0123456789"

    class CredentialBearingProvider(FakeProvider):
        def __repr__(self) -> str:
            return f"CredentialBearingProvider(api_key={canary!r})"

    provider = CredentialBearingProvider([])
    app = CayuApp(enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))
    private_cancellation_raised = False

    async def failing_session(**kwargs: Any):
        nonlocal private_cancellation_raised
        del kwargs
        private_cancellation_raised = True
        cancellation = _BillingIdentityResolutionCancelled(f"cancelled near {canary}")
        if grouped:
            raise BaseExceptionGroup(
                "credential-bearing group",
                [cancellation, RuntimeError(f"cleanup failed near {canary}")],
            )
        raise cancellation
        yield

    app._session_engine._run_session = failing_session  # type: ignore[method-assign]

    if grouped:

        async def unexpected_load(_store: object, _session_id: str) -> Session | None:
            if private_cancellation_raised:
                raise AssertionError("billing detachment must not perform an unauthenticated load")
            return None

        monkeypatch.setattr(type(app.session_store), "load", unexpected_load)
    else:

        async def unexpected_interrupt_requested(_session_id: str) -> bool:
            raise AssertionError(
                "billing detachment must not perform an unauthenticated interruption check"
            )

        app._session_control.interrupt_requested = (  # type: ignore[method-assign]
            unexpected_interrupt_requested
        )

    expected_failure = BaseExceptionGroup if grouped else asyncio.CancelledError
    with pytest.raises(expected_failure) as exc_info:
        asyncio.run(
            collect_events(
                app,
                RunRequest(
                    agent_name="assistant",
                    session_id=f"sess_billing_cancellation_transition_failure_{grouped}",
                    messages=[Message.text("user", "hello")],
                ),
            )
        )

    retained = str(exc_info.value) + repr(exc_info.value) + repr(vars(exc_info.value))
    assert canary not in retained
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None
    captured = traceback.TracebackException.from_exception(
        exc_info.value,
        capture_locals=True,
    )
    cayu_frames = [frame for frame in captured.stack if is_cayu_source_filename(frame.filename)]
    assert canary not in repr([(frame.name, frame.locals) for frame in cayu_frames])


@pytest.mark.parametrize("signal_type", [KeyboardInterrupt, SystemExit, GeneratorExit])
def test_detached_billing_cancellation_group_preserves_fatal_signal_types(
    signal_type: type[BaseException],
) -> None:
    canary = f"concurrent-{signal_type.__name__}-canary-0123456789"
    original = BaseExceptionGroup(
        "outer group",
        [
            _BillingIdentityResolutionCancelled(f"cancelled near {canary}"),
            BaseExceptionGroup(
                "nested group",
                [
                    signal_type(f"fatal signal near {canary}"),
                ],
            ),
        ],
    )

    detached = detach_billing_identity_cancellation_group(original)

    assert detached is not None
    assert isinstance(detached.exceptions[0], asyncio.CancelledError)
    nested = detached.exceptions[1]
    assert isinstance(nested, BaseExceptionGroup)
    fatal = next(child for child in nested.exceptions if isinstance(child, signal_type))
    assert canary not in (str(fatal) + repr(fatal) + repr(vars(fatal)))
    assert fatal.__traceback__ is None
    assert fatal.__cause__ is None
    assert fatal.__context__ is None


def test_detached_billing_cancellation_group_preserves_integer_system_exit_code() -> None:
    original = BaseExceptionGroup(
        "outer group",
        [
            _BillingIdentityResolutionCancelled("caller cancellation"),
            SystemExit(17),
        ],
    )

    detached = detach_billing_identity_cancellation_group(original)

    assert detached is not None
    fatal = next(child for child in detached.exceptions if isinstance(child, SystemExit))
    assert fatal.code == 17


@pytest.mark.parametrize("limit_source", ["policy", "request"])
@pytest.mark.parametrize(
    ("resolved_identity", "session_suffix"),
    [
        pytest.param(
            bedrock_billing_identity(
                invoked_model="global.anthropic.claude-sonnet-4-6",
                source_region="eu-west-1",
                resource_type="inference_profile",
                profile_scope="global",
            ),
            "wrong_region",
            id="mismatched-context",
        ),
        pytest.param(None, "missing_identity", id="missing-identity"),
    ],
)
def test_automatic_compaction_strict_contextual_budget_rejects_before_dispatch(
    limit_source: str,
    resolved_identity: BillingIdentity | None,
    session_suffix: str,
) -> None:
    invoked_model = "global.anthropic.claude-sonnet-4-6"

    class ContextualCompactionProvider(FakeProvider):
        name = "bedrock-adapter"
        billing_provider_name = "bedrock"

        async def billing_identity_for_request(
            self,
            request: ModelRequest,
        ) -> BillingIdentity | None:
            assert request.model == invoked_model
            return resolved_identity

    compactor_provider = ContextualCompactionProvider(
        [
            ModelStreamEvent.text_delta("model summary"),
            ModelStreamEvent.completed(
                {
                    "model": invoked_model,
                    "usage": {"input_tokens": 1, "output_tokens": 0},
                }
            ),
        ]
    )
    runtime_provider = FakeProvider(
        [
            ModelStreamEvent.completed(
                {
                    "model": "fake-model",
                    "usage": {"input_tokens": 1, "output_tokens": 0},
                }
            )
        ]
    )
    pricing = PriceBook(
        prices=(
            ModelPrice.fixed(
                provider_name="fake",
                model="fake-model",
                input_per_million=Decimal("1"),
                output_per_million=Decimal("1"),
            ),
            ModelPrice.fixed(
                provider_name="bedrock",
                model=invoked_model,
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
    limit = BudgetLimit(
        scope="app" if limit_source == "policy" else "run",
        max_estimated_cost=Decimal("20"),
        pricing=pricing,
    )
    app = CayuApp(
        budget_policy=(BudgetPolicy(limits=(limit,)) if limit_source == "policy" else None)
    )
    app.register_provider(runtime_provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        context_policy=CheckpointCompactionContextPolicy(
            compactor=ModelCompactor(
                provider=compactor_provider,
                model=invoked_model,
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
                session_id=(f"sess_automatic_compaction_{limit_source}_{session_suffix}"),
                messages=[
                    Message.text("user", "old"),
                    Message.text("assistant", "old answer"),
                    Message.text("user", "current"),
                ],
                budget_limits=(limit,) if limit_source == "request" else (),
            ),
        )
    )

    assert compactor_provider.requests == []
    assert runtime_provider.requests == []
    failures = [event for event in events if event.type == EventType.BUDGET_RESERVATION_FAILED]
    assert len(failures) == 1
    assert "no matching model pricing" in failures[0].payload["message"]
    assert EventType.MODEL_COMPLETED not in [event.type for event in events]
    assert events[-1].type == EventType.SESSION_INTERRUPTED

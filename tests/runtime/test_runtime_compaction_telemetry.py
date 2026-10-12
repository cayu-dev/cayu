"""Compaction telemetry contracts."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from decimal import Decimal
from typing import Any

import pytest
from tests.runtime.runtime_compaction_support import (
    _expected_compaction_failure,
    compaction_price_book,
)
from tests.runtime.runtime_provider_support import FakeProvider
from tests.runtime.runtime_session_support import (
    _expected_failure_evidence,
    _private_events_for_public_events,
    checkpoint_without_active_invocation_profile,
    collect_events,
)

import cayu.context.base as runtime_context_module
from cayu.agents import AgentSpec
from cayu.applications import CayuApp
from cayu.budgets.base import BudgetLimit, BudgetPolicy, BudgetWindow, InMemoryBudgetStore
from cayu.budgets.billing import BillingIdentity
from cayu.budgets.pricing import ModelPrice, PriceBook
from cayu.context.base import (
    CheckpointCompactionContextPolicy,
    CompactionRequest,
    CompactionResult,
    ContextCompactor,
    ModelCompactor,
)
from cayu.context.footprints import RequestFootprintConfig
from cayu.events import EventType
from cayu.messages import Message
from cayu.providers.base import ModelProvider, ModelRequest, ModelStreamEvent, ModelStreamEventType
from cayu.sessions.checkpoints import (
    CHECKPOINT_SCHEMA_VERSION_KEY,
    CURRENT_CHECKPOINT_SCHEMA_VERSION,
)
from cayu.sessions.memory import InMemorySessionStore
from cayu.sessions.records import Session
from cayu.sessions.requests import RunRequest
from cayu.tasks.creation import TaskCreate
from cayu.tasks.memory import InMemoryTaskStore
from cayu.tasks.records import TaskStatus


def test_automatic_compaction_rejects_unattributed_custom_telemetry_without_leaking():
    class AdversarialAutomaticCompactor(ContextCompactor):
        def provider_budget_identity(self, _session: Session) -> None:
            return None

        async def compact(self, request: CompactionRequest) -> CompactionResult:
            identity = BillingIdentity(
                provider_name="adversarial-compactor",
                resource_id="summary-model-v1",
            )
            return CompactionResult(
                summary="private summary",
                covered_message_count=len(request.messages),
                source_chunk_mode="vendor-private-mode",
                metadata={
                    "instructions": "metadata instruction secret",
                    "provider_state": {"secret": "metadata provider secret"},
                },
                model_completed_payloads=[
                    {
                        "provider_name": "adversarial-compactor",
                        "requested_model": "summary-model",
                        "model": "summary-model-v1",
                        "billing_identity": identity.model_dump(mode="json"),
                        "usage_metrics": {
                            "input_tokens": 8,
                            "output_tokens": 2,
                            "total_tokens": 10,
                            "provider_state": {"secret": "nested metrics secret"},
                            "arbitrary": "q" * 100_000,
                        },
                        "summary": "model event summary secret",
                        "instructions": "model event instruction secret",
                        "provider_state": {"secret": "opaque continuation secret"},
                        "arbitrary": "p" * 100_000,
                    }
                ],
            )

    async def run() -> None:
        session_id = "sess_automatic_adversarial_compaction_telemetry"
        store = InMemorySessionStore()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(FakeProvider([ModelStreamEvent.completed({})]), default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=AdversarialAutomaticCompactor(),
                max_user_turns=1,
                compact_after_messages=1,
            ),
        )
        await collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[
                    Message.text("user", "old"),
                    Message.text("assistant", "old answer"),
                    Message.text("user", "current"),
                ],
            ),
        )
        events = await store.load_events(session_id)
        assert not any(
            event.type == EventType.MODEL_COMPLETED
            and event.payload.get("purpose") == "context_compaction"
            for event in events
        )
        assert not any(event.type == EventType.CONTEXT_COMPACTION_COMPLETED for event in events)
        assert events[-1].type == EventType.SESSION_FAILED
        assert (
            events[-1].payload["error"]
            == "Compaction completion was observed outside its provider dispatch."
        )
        serialized = json.dumps(
            [event.model_dump(mode="json") for event in events],
            ensure_ascii=False,
        )
        for secret in (
            "metadata instruction secret",
            "metadata provider secret",
            "model event summary secret",
            "model event instruction secret",
            "opaque continuation secret",
            "nested metrics secret",
            "p" * 1000,
            "q" * 1000,
            "vendor-private-mode",
        ):
            assert secret not in serialized

    asyncio.run(run())


def test_compaction_telemetry_rejects_conflicting_billing_identities():
    event_identity = BillingIdentity(
        provider_name="event-provider",
        resource_id="event-model",
    )
    metrics_identity = BillingIdentity(
        provider_name="metrics-provider",
        resource_id="metrics-model",
    )
    telemetry = runtime_context_module.ContextCompactionTelemetry(
        event_type=EventType.MODEL_COMPLETED,
        payload={
            "billing_identity": event_identity.model_dump(mode="json"),
            "usage_metrics": {
                "billing_identity": metrics_identity.model_dump(mode="json"),
                "input_tokens": 8,
                "output_tokens": 2,
                "total_tokens": 10,
            },
        },
    )

    with pytest.raises(ValueError, match="billing identities do not match"):
        runtime_context_module.sanitize_context_compaction_telemetry(telemetry)


@pytest.mark.parametrize(
    "payload",
    [
        {"billing_identity": "not-an-object"},
        {
            "usage_metrics": {
                "billing_identity": {"provider_name": "missing-resource-id"},
                "input_tokens": 1,
            }
        },
    ],
)
def test_compaction_telemetry_rejects_malformed_billing_identity(payload):
    telemetry = runtime_context_module.ContextCompactionTelemetry(
        event_type=EventType.MODEL_COMPLETED,
        payload=payload,
    )

    with pytest.raises(ValueError, match="billing identity"):
        runtime_context_module.sanitize_context_compaction_telemetry(telemetry)


def test_compaction_telemetry_preserves_exact_maximum_usage_counter():
    maximum = runtime_context_module._COMPACTION_EVENT_INTEGER_MAX
    sanitized = runtime_context_module.sanitize_context_compaction_telemetry(
        runtime_context_module.ContextCompactionTelemetry(
            event_type=EventType.MODEL_COMPLETED,
            payload={
                "provider_name": "provider",
                "model": "model",
                "usage_metrics": {"input_tokens": maximum},
            },
        )
    )

    assert sanitized.payload["usage_metrics"]["input_tokens"] == maximum
    assert "usage_unavailable_reason" not in sanitized.payload


def test_compaction_telemetry_does_not_turn_empty_metrics_into_zero_usage():
    sanitized = runtime_context_module.sanitize_context_compaction_telemetry(
        runtime_context_module.ContextCompactionTelemetry(
            event_type=EventType.MODEL_COMPLETED,
            payload={
                "provider_name": "provider",
                "model": "model",
                "usage_metrics": {},
            },
        )
    )

    assert "usage_metrics" not in sanitized.payload


def test_compaction_telemetry_preserves_identity_from_empty_metrics():
    identity = BillingIdentity(provider_name="provider", resource_id="model")
    sanitized = runtime_context_module.sanitize_context_compaction_telemetry(
        runtime_context_module.ContextCompactionTelemetry(
            event_type=EventType.MODEL_COMPLETED,
            payload={
                "usage_metrics": {
                    "billing_identity": identity.model_dump(mode="json"),
                },
            },
        )
    )

    assert "usage_metrics" not in sanitized.payload
    assert sanitized.payload["billing_identity"] == identity.model_dump(mode="json")


def test_compaction_telemetry_preserves_provider_model_from_empty_metrics():
    sanitized = runtime_context_module.sanitize_context_compaction_telemetry(
        runtime_context_module.ContextCompactionTelemetry(
            event_type=EventType.MODEL_COMPLETED,
            payload={
                "usage_metrics": {
                    "provider_name": "nested-provider",
                    "requested_model": "requested-model",
                    "model": "served-model",
                },
            },
        )
    )

    assert "usage_metrics" not in sanitized.payload
    assert sanitized.payload["provider_name"] == "nested-provider"
    assert sanitized.payload["requested_model"] == "requested-model"
    assert sanitized.payload["model"] == "served-model"


def test_compaction_telemetry_rejects_conflicting_identity_with_empty_metrics():
    event_identity = BillingIdentity(provider_name="event-provider", resource_id="event-model")
    metrics_identity = BillingIdentity(
        provider_name="metrics-provider",
        resource_id="metrics-model",
    )
    telemetry = runtime_context_module.ContextCompactionTelemetry(
        event_type=EventType.MODEL_COMPLETED,
        payload={
            "billing_identity": event_identity.model_dump(mode="json"),
            "usage_metrics": {
                "billing_identity": metrics_identity.model_dump(mode="json"),
            },
        },
    )

    with pytest.raises(ValueError, match="billing identities do not match"):
        runtime_context_module.sanitize_context_compaction_telemetry(telemetry)


@pytest.mark.parametrize(
    "usage_payload",
    [
        {"usage_metrics": {"input_tokens": 2**63}},
        {"usage_metrics": {"input_tokens": -1}},
        {"usage_metrics": {"input_tokens": True}},
        {"usage_metrics": {"cache": {"read_tokens": "1"}}},
        {"usage_metrics_rejected": True},
        {"usage": {"input_tokens": 2**63}},
        {
            "usage": {
                "cache_creation": {
                    "ephemeral_5m": 2**63 - 1,
                    "ephemeral_1h": 1,
                }
            }
        },
    ],
)
def test_compaction_telemetry_marks_present_invalid_usage_unavailable(usage_payload):
    sanitized = runtime_context_module.sanitize_context_compaction_telemetry(
        runtime_context_module.ContextCompactionTelemetry(
            event_type=EventType.MODEL_COMPLETED,
            payload={
                "provider_name": "provider",
                "model": "model",
                **usage_payload,
            },
        )
    )

    assert "usage" not in sanitized.payload
    assert "usage_metrics" not in sanitized.payload
    assert sanitized.payload["usage_unavailable_reason"] == ("invalid compaction usage telemetry")


def test_automatic_compaction_rejects_conflicting_billing_identity_before_checkpoint():
    event_identity = BillingIdentity(
        provider_name="event-provider",
        resource_id="event-model",
    )
    metrics_identity = BillingIdentity(
        provider_name="metrics-provider",
        resource_id="metrics-model",
    )

    class ConflictingIdentityCompactor(ContextCompactor):
        def provider_budget_identity(self, _session: Session) -> None:
            return None

        async def compact(self, request: CompactionRequest) -> CompactionResult:
            return CompactionResult(
                summary="must not be checkpointed",
                covered_message_count=len(request.messages),
                model_completed_payloads=[
                    {
                        "billing_identity": event_identity.model_dump(mode="json"),
                        "usage_metrics": {
                            "billing_identity": metrics_identity.model_dump(mode="json"),
                            "input_tokens": 8,
                            "output_tokens": 2,
                            "total_tokens": 10,
                        },
                    }
                ],
            )

    async def run() -> None:
        session_id = "sess_conflicting_compaction_billing_identity"
        store = InMemorySessionStore()
        provider = FakeProvider([ModelStreamEvent.completed({})])
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=ConflictingIdentityCompactor(),
                max_user_turns=1,
                compact_after_messages=1,
            ),
        )

        events = await collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[
                    Message.text("user", "old"),
                    Message.text("assistant", "old answer"),
                    Message.text("user", "current"),
                ],
            ),
        )

        assert provider.requests == []
        assert EventType.MODEL_COMPLETED not in {event.type for event in events}
        assert EventType.CONTEXT_COMPACTION_COMPLETED not in {event.type for event in events}
        assert events[-1].type == EventType.SESSION_FAILED
        checkpoint = await store.load_checkpoint(session_id)
        assert checkpoint is None or "context_compaction" not in checkpoint

    asyncio.run(run())


def test_cayu_app_keeps_invalid_summary_spend_before_rejecting_custom_recovery():
    class RecoveringCompactor(ContextCompactor):
        def __init__(self, provider: ModelProvider) -> None:
            self.provider = provider

        async def compact(self, request: CompactionRequest) -> CompactionResult:
            try:
                await ModelCompactor(
                    provider=self.provider,
                    model="invalid-summary-model",
                ).compact(request)
            except ValueError:
                return CompactionResult(
                    summary="recovered summary",
                    covered_message_count=len(request.messages),
                    model_completed_payloads=[
                        {
                            "model": "fallback-summary-model",
                            "provider_name": "custom",
                            "requested_model": "fallback-summary-model",
                            "purpose": "context_compaction",
                            "compactor": type(self).__name__,
                            "usage_metrics": {
                                "provider_name": "custom",
                                "requested_model": "fallback-summary-model",
                                "model": "fallback-summary-model",
                                "input_tokens": 40,
                                "output_tokens": 2,
                                "total_tokens": 42,
                                "reasoning_output_tokens": 0,
                                "cache": {
                                    "read_tokens": 0,
                                    "write_tokens": 0,
                                    "cached_input_tokens": 0,
                                    "uncached_input_tokens": 40,
                                },
                            },
                        }
                    ],
                )
            raise AssertionError("the first compaction unexpectedly returned a valid summary")

    store = InMemorySessionStore()
    budget_store = InMemoryBudgetStore()
    compactor_provider = FakeProvider(
        [
            ModelStreamEvent.completed(
                {
                    "model": "invalid-summary-model",
                    "usage": {"input_tokens": 100, "output_tokens": 0},
                }
            )
        ]
    )
    runtime_provider = FakeProvider(
        [
            ModelStreamEvent.text_delta("done"),
            ModelStreamEvent.completed({"usage": {"input_tokens": 3, "output_tokens": 1}}),
        ]
    )
    app = CayuApp(
        session_store=store,
        budget_store=budget_store,
        request_footprint=RequestFootprintConfig(enabled=False),
    )
    app.register_provider(runtime_provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        context_policy=CheckpointCompactionContextPolicy(
            compactor=RecoveringCompactor(compactor_provider),
            max_user_turns=1,
            compact_after_messages=2,
        ),
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_recovered_invalid_compaction_summary",
                messages=[
                    Message.text("user", "old"),
                    Message.text("assistant", "old answer"),
                    Message.text("user", "current"),
                ],
            ),
        )
    )

    durable_events = asyncio.run(store.load_events("sess_recovered_invalid_compaction_summary"))
    completed_events = [
        event for event in durable_events if event.type == EventType.MODEL_COMPLETED
    ]
    assert len(completed_events) == 1
    invalid_attempt = completed_events[0]
    assert invalid_attempt.payload["compaction_outcome"] == "invalid_summary"
    assert invalid_attempt.payload["usage_metrics"]["input_tokens"] == 100
    assert "compaction_attempt_id" not in invalid_attempt.payload
    assert invalid_attempt.payload["model_step_id"].startswith("mstep_")
    assert invalid_attempt.payload["model_attempt_id"].startswith("matt_")
    assert events[-1].type == EventType.SESSION_FAILED
    assert (
        events[-1].payload["error"]
        == "Compaction completion was observed outside its provider dispatch."
    )
    assert runtime_provider.requests == []

    usage = asyncio.run(app.get_session_usage("sess_recovered_invalid_compaction_summary"))
    assert usage.model_steps == 1
    assert usage.usage.input_tokens == 100
    assert usage.usage.output_tokens == 0
    budget_events = asyncio.run(
        budget_store.load_events_for_budget(
            scope="app",
            key=None,
            window=BudgetWindow.all_time(),
        )
    )
    assert [event.id for event in budget_events] == [invalid_attempt.id]


def test_cayu_app_keeps_successful_inner_compaction_spend_when_wrapper_fails():
    class FailingWrapperCompactor(ContextCompactor):
        def __init__(self, provider: ModelProvider) -> None:
            self.provider = provider

        async def compact(self, request: CompactionRequest) -> CompactionResult:
            await ModelCompactor(provider=self.provider, model="summary-model").compact(request)
            raise RuntimeError("wrapper post-processing failed")

    store = InMemorySessionStore()
    budget_store = InMemoryBudgetStore()
    compactor_provider = FakeProvider(
        [
            ModelStreamEvent.text_delta("valid inner summary"),
            ModelStreamEvent.completed(
                {
                    "model": "summary-model",
                    "usage": {"input_tokens": 10, "output_tokens": 2},
                }
            ),
        ]
    )
    runtime_provider = FakeProvider([ModelStreamEvent.completed({})])
    app = CayuApp(
        session_store=store,
        budget_store=budget_store,
        request_footprint=RequestFootprintConfig(enabled=False),
    )
    app.register_provider(runtime_provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        context_policy=CheckpointCompactionContextPolicy(
            compactor=FailingWrapperCompactor(compactor_provider),
            max_user_turns=1,
            compact_after_messages=2,
        ),
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_failed_compaction_wrapper",
                messages=[
                    Message.text("user", "old"),
                    Message.text("assistant", "old answer"),
                    Message.text("user", "current"),
                ],
            ),
        )
    )

    assert [event.type for event in events] == [
        EventType.SESSION_STARTED,
        EventType.CONTEXT_COMPACTION_STARTED,
        EventType.MODEL_STARTED,
        EventType.MODEL_COMPLETED,
        EventType.CONTEXT_COMPACTION_FAILED,
        EventType.TURN_COMPLETED,
        EventType.SESSION_FAILED,
    ]
    completed = events[3]
    assert completed.payload["usage_metrics"]["input_tokens"] == 10
    assert completed.payload["usage_metrics"]["output_tokens"] == 2
    assert events[4].payload["error_type"] == "RuntimeError"
    assert "error" not in events[4].payload
    assert runtime_provider.requests == []
    usage = asyncio.run(app.get_session_usage("sess_failed_compaction_wrapper"))
    assert usage.model_steps == 1
    assert usage.usage.total_tokens == 12
    budget_events = asyncio.run(
        budget_store.load_events_for_budget(
            scope="app",
            key=None,
            window=BudgetWindow.all_time(),
        )
    )
    private_completed = asyncio.run(
        _private_events_for_public_events(
            app.session_store,
            "sess_failed_compaction_wrapper",
            [completed],
        )
    )[0]
    assert [event.id for event in budget_events] == [private_completed.id]


def test_cayu_app_keeps_compaction_completion_when_stream_fails_after_completed():
    class CompletedThenFailingProvider(ModelProvider):
        name = "fake"

        async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
            del request
            yield ModelStreamEvent.text_delta("completed summary")
            yield ModelStreamEvent.completed(
                {
                    "model": "summary-model",
                    "usage": {"input_tokens": 14, "output_tokens": 2},
                }
            )
            raise RuntimeError("stream teardown failed")

    budget_store = InMemoryBudgetStore()
    runtime_provider = FakeProvider([ModelStreamEvent.completed({})])
    app = CayuApp(budget_store=budget_store)
    app.register_provider(runtime_provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        context_policy=CheckpointCompactionContextPolicy(
            compactor=ModelCompactor(
                provider=CompletedThenFailingProvider(),
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
                session_id="sess_compaction_stream_teardown_failure",
                messages=[
                    Message.text("user", "old"),
                    Message.text("assistant", "old answer"),
                    Message.text("user", "current"),
                ],
            ),
        )
    )

    completed_events = [event for event in events if event.type == EventType.MODEL_COMPLETED]
    assert len(completed_events) == 1
    completed = completed_events[0]
    assert completed.payload["usage_metrics"]["input_tokens"] == 14
    assert completed.payload["usage_metrics"]["output_tokens"] == 2
    assert "compaction_attempt_id" not in completed.payload
    failed = next(event for event in events if event.type == EventType.CONTEXT_COMPACTION_FAILED)
    assert failed.payload["error_type"] == "RuntimeError"
    assert "error" not in failed.payload
    assert runtime_provider.requests == []
    usage = asyncio.run(app.get_session_usage("sess_compaction_stream_teardown_failure"))
    assert usage.model_steps == 1
    assert usage.usage.total_tokens == 16
    budget_events = asyncio.run(
        budget_store.load_events_for_budget(
            scope="app",
            key=None,
            window=BudgetWindow.all_time(),
        )
    )
    private_completed = asyncio.run(
        _private_events_for_public_events(
            app.session_store,
            "sess_compaction_stream_teardown_failure",
            [completed],
        )
    )[0]
    assert [event.id for event in budget_events] == [private_completed.id]


def test_cayu_app_preserves_provider_order_when_wrapper_recovers_earlier_draft():
    class DraftRecoveringCompactor(ContextCompactor):
        def __init__(self, provider: ModelProvider) -> None:
            self.provider = provider

        async def compact(self, request: CompactionRequest) -> CompactionResult:
            draft = await ModelCompactor(
                provider=self.provider,
                model="draft-model",
            ).compact(request)
            try:
                await ModelCompactor(
                    provider=self.provider,
                    model="refinement-model",
                ).compact(request)
            except ValueError:
                return draft
            raise AssertionError("the refinement unexpectedly returned a valid summary")

    budget_store = InMemoryBudgetStore()
    compactor_provider = FakeProvider(
        [
            [
                ModelStreamEvent.text_delta("usable draft"),
                ModelStreamEvent.completed(
                    {
                        "model": "draft-model",
                        "usage": {"input_tokens": 10, "output_tokens": 1},
                    }
                ),
            ],
            [
                ModelStreamEvent.text_delta(" \n\t"),
                ModelStreamEvent.completed(
                    {
                        "model": "refinement-model",
                        "usage": {"input_tokens": 20, "output_tokens": 2},
                    }
                ),
            ],
        ]
    )
    runtime_provider = FakeProvider(
        [
            ModelStreamEvent.text_delta("done"),
            ModelStreamEvent.completed({"usage": {"input_tokens": 3, "output_tokens": 1}}),
        ]
    )
    app = CayuApp(
        budget_store=budget_store,
        request_footprint=RequestFootprintConfig(enabled=False),
    )
    app.register_provider(runtime_provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        context_policy=CheckpointCompactionContextPolicy(
            compactor=DraftRecoveringCompactor(compactor_provider),
            max_user_turns=1,
            compact_after_messages=2,
        ),
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_recovered_compaction_draft",
                messages=[
                    Message.text("user", "old"),
                    Message.text("assistant", "old answer"),
                    Message.text("user", "current"),
                ],
            ),
        )
    )

    completed_events = [event for event in events if event.type == EventType.MODEL_COMPLETED]
    assert len(completed_events) == 3
    draft, refinement, runtime_completed = completed_events
    assert draft.payload["requested_model"] == "draft-model"
    assert draft.payload["usage_metrics"]["input_tokens"] == 10
    assert refinement.payload["requested_model"] == "refinement-model"
    assert refinement.payload["compaction_outcome"] == "invalid_summary"
    assert refinement.payload["usage_metrics"]["input_tokens"] == 20
    assert runtime_completed.payload["usage_metrics"]["input_tokens"] == 3
    assert events[-1].type == EventType.SESSION_COMPLETED
    usage = asyncio.run(app.get_session_usage("sess_recovered_compaction_draft"))
    assert usage.model_steps == 3
    assert usage.usage.input_tokens == 33
    assert usage.usage.output_tokens == 4
    budget_events = asyncio.run(
        budget_store.load_events_for_budget(
            scope="app",
            key=None,
            window=BudgetWindow.all_time(),
        )
    )
    private_completed_events = asyncio.run(
        _private_events_for_public_events(
            app.session_store,
            "sess_recovered_compaction_draft",
            completed_events,
        )
    )
    assert [event.id for event in budget_events] == [event.id for event in private_completed_events]


def test_cayu_app_rejects_returned_only_completion_and_keeps_anchored_inner_call():
    class ReturnedFirstCompactor(ContextCompactor):
        def __init__(self, provider: ModelProvider) -> None:
            self.provider = provider

        async def compact(self, request: CompactionRequest) -> CompactionResult:
            inner = await ModelCompactor(
                provider=self.provider,
                model="inner-model",
            ).compact(request)
            return CompactionResult(
                summary=inner.summary,
                covered_message_count=inner.covered_message_count,
                model_completed_payloads=[
                    {
                        "model": "custom-model",
                        "provider_name": "custom",
                        "requested_model": "custom-model",
                        "purpose": "context_compaction",
                        "compactor": type(self).__name__,
                        "usage_metrics": {
                            "provider_name": "custom",
                            "requested_model": "custom-model",
                            "model": "custom-model",
                            "input_tokens": 10,
                            "output_tokens": 1,
                            "total_tokens": 11,
                            "reasoning_output_tokens": 0,
                            "cache": {
                                "read_tokens": 0,
                                "write_tokens": 0,
                                "cached_input_tokens": 0,
                                "uncached_input_tokens": 10,
                            },
                        },
                    },
                    *inner.model_completed_payloads,
                ],
            )

    budget_store = InMemoryBudgetStore()
    compactor_provider = FakeProvider(
        [
            ModelStreamEvent.text_delta("inner summary"),
            ModelStreamEvent.completed(
                {
                    "model": "inner-model",
                    "usage": {"input_tokens": 20, "output_tokens": 2},
                }
            ),
        ]
    )
    runtime_provider = FakeProvider(
        [
            ModelStreamEvent.text_delta("done"),
            ModelStreamEvent.completed({"usage": {"input_tokens": 3, "output_tokens": 1}}),
        ]
    )
    app = CayuApp(
        budget_store=budget_store,
        request_footprint=RequestFootprintConfig(enabled=False),
    )
    app.register_provider(runtime_provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        context_policy=CheckpointCompactionContextPolicy(
            compactor=ReturnedFirstCompactor(compactor_provider),
            max_user_turns=1,
            compact_after_messages=2,
        ),
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_returned_completion_before_inner",
                messages=[
                    Message.text("user", "old"),
                    Message.text("assistant", "old answer"),
                    Message.text("user", "current"),
                ],
            ),
        )
    )

    durable_events = asyncio.run(
        app.session_store.load_events("sess_returned_completion_before_inner")
    )
    completed_events = [
        event for event in durable_events if event.type == EventType.MODEL_COMPLETED
    ]
    assert len(completed_events) == 1
    inner = completed_events[0]
    assert inner.payload["requested_model"] == "inner-model"
    assert inner.payload["usage_metrics"]["input_tokens"] == 20
    assert inner.payload["model_step_id"].startswith("mstep_")
    assert inner.payload["model_attempt_id"].startswith("matt_")
    assert events[-1].type == EventType.SESSION_FAILED
    assert (
        events[-1].payload["error"]
        == "Compaction completion was observed outside its provider dispatch."
    )
    assert runtime_provider.requests == []
    usage = asyncio.run(app.get_session_usage("sess_returned_completion_before_inner"))
    assert usage.model_steps == 1
    assert usage.usage.input_tokens == 20
    assert usage.usage.output_tokens == 2
    budget_events = asyncio.run(
        budget_store.load_events_for_budget(
            scope="app",
            key=None,
            window=BudgetWindow.all_time(),
        )
    )
    assert [event.id for event in budget_events] == [inner.id]


@pytest.mark.parametrize(
    ("invalid_value", "invalid_location", "expected_error_code", "expected_error_path"),
    [
        ("timeout\x00workload-secret-value", "top-level", "nul_character", "$/#2"),
        (float("nan"), "top-level", "non_finite_number", "$/#2"),
        ("timeout\x00workload-secret-value", "usage", "nul_character", "$/#1/#2"),
        (
            "timeout\x00workload-secret-value",
            "model-and-usage",
            "nul_character",
            "$/#0",
        ),
    ],
    ids=["nul", "nan", "nested-usage-aux", "invalid-model-and-nested-usage-aux"],
)
def test_cayu_app_retains_only_authoritative_usage_when_completion_metadata_is_not_durable(
    invalid_value: object,
    invalid_location: str,
    expected_error_code: str,
    expected_error_path: str,
):
    store = InMemorySessionStore()
    budget_store = InMemoryBudgetStore()
    secret_key = "workload-secret-key"
    usage = {"input_tokens": 30, "output_tokens": 4}
    payload: dict[str, Any] = {
        "model": "summary-model",
        "usage": usage,
    }
    if invalid_location == "usage":
        usage[secret_key] = invalid_value
    elif invalid_location == "model-and-usage":
        payload["model"] = invalid_value
        usage[secret_key] = invalid_value
    else:
        payload[secret_key] = invalid_value
    compactor_provider = FakeProvider(
        [
            ModelStreamEvent.text_delta("valid summary"),
            ModelStreamEvent.model_construct(
                type=ModelStreamEventType.COMPLETED,
                delta="",
                payload=payload,
                completion=None,
            ),
        ]
    )
    runtime_provider = FakeProvider([ModelStreamEvent.completed({})])
    app = CayuApp(session_store=store, budget_store=budget_store)
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
                session_id="sess_unsafe_compaction_metadata",
                messages=[
                    Message.text("user", "old"),
                    Message.text("assistant", "old answer"),
                    Message.text("user", "current"),
                ],
            ),
        )
    )

    assert [event.type for event in events] == [
        EventType.SESSION_STARTED,
        EventType.CONTEXT_COMPACTION_STARTED,
        EventType.MODEL_STARTED,
        EventType.MODEL_COMPLETED,
        EventType.CONTEXT_COMPACTION_FAILED,
        EventType.TURN_COMPLETED,
        EventType.SESSION_FAILED,
    ]
    completed = events[3]
    assert completed.payload["provider_name"] == "fake"
    assert completed.payload["requested_model"] == "summary-model"
    assert completed.payload["model"] == "summary-model"
    if invalid_location == "top-level":
        assert completed.payload["usage_metrics"]["input_tokens"] == 30
        assert completed.payload["usage_metrics"]["output_tokens"] == 4
        assert completed.payload["usage"] == {"input_tokens": 30, "output_tokens": 4}
        assert "usage_normalization_failed" not in completed.payload
    else:
        assert "usage" not in completed.payload
        assert "usage_metrics" not in completed.payload
        assert completed.payload["usage_normalization_failed"] is True
        assert completed.payload["usage_unavailable_reason"] == (
            "invalid compaction usage telemetry"
        )
    assert "provider_note" not in completed.payload
    assert completed.payload["compaction_outcome"] == "invalid_completion_metadata"
    assert secret_key not in completed.payload
    assert "compaction_attempt_id" not in completed.payload
    assert events[4].payload["error_type"] == "DurableValueError"
    assert "error" not in events[4].payload
    assert events[-1].payload == {
        "compaction_failure": _expected_compaction_failure(events[4]),
        "failure_evidence": _expected_failure_evidence(
            "sess_unsafe_compaction_metadata", "DurableValueError", "ContextBuildError"
        ),
        "error": "Operation failed with a non-portable diagnostic.",
        "error_type": "DurableValueError",
        "durable_value_error_code": expected_error_code,
        "durable_value_error_path": expected_error_path,
    }
    rendered = json.dumps(
        [event.model_dump(mode="json") for event in events],
        allow_nan=False,
    )
    assert secret_key not in rendered
    assert "timeout" not in rendered
    assert "workload-secret-value" not in rendered
    assert len(compactor_provider.requests) == 1
    assert runtime_provider.requests == []
    usage = asyncio.run(app.get_session_usage("sess_unsafe_compaction_metadata"))
    assert usage.model_steps == 1
    assert usage.usage.total_tokens == (34 if invalid_location == "top-level" else 0)
    budget_events = asyncio.run(
        budget_store.load_events_for_budget(
            scope="app",
            key=None,
            window=BudgetWindow.all_time(),
        )
    )
    private_completed = asyncio.run(
        _private_events_for_public_events(
            store,
            "sess_unsafe_compaction_metadata",
            [completed],
        )
    )[0]
    assert [event.id for event in budget_events] == [private_completed.id]


@pytest.mark.parametrize(
    ("summary_events", "expected_error", "expected_error_type", "expected_error_code"),
    [
        ([], "`summary` cannot be blank.", "ValueError", None),
        (
            [ModelStreamEvent.text_delta(" \n\t")],
            "`summary` cannot be blank.",
            "ValueError",
            None,
        ),
        (
            [ModelStreamEvent.text_delta("invalid\x00summary")],
            "`summary` must not contain NUL characters. [code=nul_character; path=$]",
            "DurableValueError",
            "nul_character",
        ),
        (
            [ModelStreamEvent.text_delta("invalid\ud800summary")],
            "`summary` must not contain Unicode surrogate code points. "
            "[code=unicode_surrogate; path=$]",
            "DurableValueError",
            "unicode_surrogate",
        ),
    ],
    ids=["blank", "whitespace", "nul", "surrogate"],
)
def test_cayu_app_counts_completed_compaction_with_invalid_summary(
    summary_events: list[ModelStreamEvent],
    expected_error: str,
    expected_error_type: str,
    expected_error_code: str | None,
):
    store = InMemorySessionStore()
    task_store = InMemoryTaskStore()
    asyncio.run(
        task_store.create_task(
            TaskCreate(
                task_id="task_invalid_compaction_summary",
                type="compaction-recovery",
                assigned_agent_name="assistant",
            )
        )
    )
    budget_store = InMemoryBudgetStore()
    compactor_provider = FakeProvider(
        [
            *summary_events,
            ModelStreamEvent.completed(
                {
                    "model": "summary-model",
                    "usage": {"input_tokens": 100, "output_tokens": 0},
                }
            ),
        ]
    )
    runtime_provider = FakeProvider(
        [
            ModelStreamEvent.text_delta("unused"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ]
    )
    app = CayuApp(
        session_store=store,
        task_store=task_store,
        budget_store=budget_store,
    )
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
                session_id="sess_invalid_compaction_summary",
                causal_budget_id="job_invalid_compaction_summary",
                task_id="task_invalid_compaction_summary",
                messages=[
                    Message.text("user", "old"),
                    Message.text("assistant", "old answer"),
                    Message.text("user", "current"),
                ],
            ),
        )
    )

    assert [event.type for event in events] == [
        EventType.SESSION_STARTED,
        EventType.TASK_STARTED,
        EventType.CONTEXT_COMPACTION_STARTED,
        EventType.MODEL_STARTED,
        EventType.MODEL_COMPLETED,
        EventType.CONTEXT_COMPACTION_FAILED,
        EventType.TASK_FAILED,
        EventType.TURN_COMPLETED,
        EventType.SESSION_FAILED,
    ]
    completed = events[4]
    assert completed.payload["purpose"] == "context_compaction"
    assert completed.payload["compactor"] == "ModelCompactor"
    assert completed.payload["compaction_outcome"] == "invalid_summary"
    assert completed.payload["provider_name"] == "fake"
    assert completed.payload["requested_model"] == "summary-model"
    assert completed.payload["model"] == "summary-model"
    assert completed.payload["usage"] == {"input_tokens": 100, "output_tokens": 0}
    assert completed.payload["usage_metrics"] == {
        "provider_name": "fake",
        "requested_model": "summary-model",
        "model": "summary-model",
        "input_tokens": 100,
        "output_tokens": 0,
        "total_tokens": 100,
        "reasoning_output_tokens": 0,
        "cache": {
            "read_tokens": 0,
            "write_tokens": 0,
            "cached_input_tokens": 0,
            "uncached_input_tokens": 100,
        },
    }
    failure = events[5]
    assert failure.payload["error_type"] == expected_error_type
    assert "error" not in failure.payload
    expected_disposition = {
        "phase": "provider_dispatch",
        "reason": "internal_failed",
        "retryable": False,
        "provider_dispatch_disposition": "dispatched",
        "recovery_action": "reconcile_completion",
    }
    assert {key: failure.payload[key] for key in expected_disposition} == expected_disposition
    assert type(failure.payload["elapsed_ms"]) is int
    compaction_failure = {
        **expected_disposition,
        "elapsed_ms": failure.payload["elapsed_ms"],
    }
    assert events[7].payload["step_count"] == 1
    assert events[7].payload["token_usage"]["input_tokens"] == 100
    terminal = events[8]
    assert terminal.payload["error_type"] == expected_error_type
    assert terminal.payload["compaction_failure"] == compaction_failure
    assert type(terminal.payload["execution_profile_fingerprint"]) is str
    assert type(terminal.payload["runtime_task_failure_id"]) is str
    if expected_error_type == "DurableValueError":
        assert expected_error_code is not None
        assert terminal.payload["error"] == "Operation failed with a non-portable diagnostic."
        assert terminal.payload["durable_value_error_code"] == expected_error_code
        assert terminal.payload["durable_value_error_path"] == "$"
    else:
        assert terminal.payload["error"] == expected_error
    task = asyncio.run(task_store.load_task("task_invalid_compaction_summary"))
    assert task is not None
    assert task.status == TaskStatus.FAILED
    assert task.error is not None
    assert task.error["compaction_failure"] == compaction_failure
    assert runtime_provider.requests == []
    assert checkpoint_without_active_invocation_profile(
        asyncio.run(store.load_checkpoint("sess_invalid_compaction_summary"))
    ) == {CHECKPOINT_SCHEMA_VERSION_KEY: CURRENT_CHECKPOINT_SCHEMA_VERSION}

    usage = asyncio.run(app.get_session_usage("sess_invalid_compaction_summary"))
    assert usage.model_steps == 1
    assert usage.usage.input_tokens == 100
    assert usage.usage.total_tokens == 100
    causal_usage = asyncio.run(app.get_causal_budget_usage("job_invalid_compaction_summary"))
    assert causal_usage.model_steps == 1
    assert causal_usage.usage.input_tokens == 100

    pricing = PriceBook(
        prices=(
            ModelPrice.fixed(
                provider_name="fake",
                model="summary-model",
                input_per_million=Decimal("1"),
                output_per_million=Decimal("10"),
            ),
        )
    )
    cost = asyncio.run(
        app.get_session_cost(
            "sess_invalid_compaction_summary",
            pricing,
        )
    )
    assert cost.model_steps == 1
    assert cost.priced_model_steps == 1
    assert cost.total_cost == Decimal("0.0001")
    causal_cost = asyncio.run(app.get_causal_budget_cost("job_invalid_compaction_summary", pricing))
    assert causal_cost.model_steps == 1
    assert causal_cost.priced_model_steps == 1
    assert causal_cost.total_cost == Decimal("0.0001")

    budget_events = asyncio.run(
        budget_store.load_events_for_budget(
            scope="app",
            key=None,
            window=BudgetWindow.all_time(),
        )
    )
    private_completed = asyncio.run(
        _private_events_for_public_events(
            store,
            "sess_invalid_compaction_summary",
            [completed],
        )
    )[0]
    assert [event.id for event in budget_events] == [private_completed.id]


def test_cayu_app_counts_completed_compaction_rejected_for_tool_call():
    store = InMemorySessionStore()
    budget_store = InMemoryBudgetStore()
    compactor_provider = FakeProvider(
        [
            ModelStreamEvent.tool_call(
                id="call_compaction_tool",
                name="unexpected_tool",
                arguments={},
            ),
            ModelStreamEvent.completed(
                {
                    "model": "summary-model",
                    "finish_reason": "tool_calls",
                    "usage": {"input_tokens": 80, "output_tokens": 6},
                }
            ),
        ]
    )
    runtime_provider = FakeProvider(
        [
            ModelStreamEvent.text_delta("unused"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ]
    )
    app = CayuApp(session_store=store, budget_store=budget_store)
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
                session_id="sess_rejected_compaction_tool_call",
                causal_budget_id="job_rejected_compaction_tool_call",
                messages=[
                    Message.text("user", "old"),
                    Message.text("assistant", "old answer"),
                    Message.text("user", "current"),
                ],
            ),
        )
    )

    assert [event.type for event in events] == [
        EventType.SESSION_STARTED,
        EventType.CONTEXT_COMPACTION_STARTED,
        EventType.MODEL_STARTED,
        EventType.MODEL_COMPLETED,
        EventType.CONTEXT_COMPACTION_FAILED,
        EventType.TURN_COMPLETED,
        EventType.SESSION_FAILED,
    ]
    completed = events[3]
    assert completed.payload["purpose"] == "context_compaction"
    assert completed.payload["compactor"] == "ModelCompactor"
    assert completed.payload["compaction_outcome"] == "rejected_tool_call"
    assert completed.payload["finish_reason"] == "tool_calls"
    assert completed.payload["usage_metrics"]["input_tokens"] == 80
    assert completed.payload["usage_metrics"]["output_tokens"] == 6
    assert "compaction_attempt_id" not in completed.payload
    assert events[4].payload["error_type"] == "_CompactionToolCallError"
    assert "error" not in events[4].payload
    assert events[5].payload["step_count"] == 1
    assert events[5].payload["token_usage"]["total_tokens"] == 86
    assert events[6].payload["error"] == "Compaction model must not call tools."
    assert runtime_provider.requests == []
    assert checkpoint_without_active_invocation_profile(
        asyncio.run(store.load_checkpoint("sess_rejected_compaction_tool_call"))
    ) == {CHECKPOINT_SCHEMA_VERSION_KEY: CURRENT_CHECKPOINT_SCHEMA_VERSION}

    usage = asyncio.run(app.get_session_usage("sess_rejected_compaction_tool_call"))
    assert usage.model_steps == 1
    assert usage.usage.input_tokens == 80
    assert usage.usage.output_tokens == 6
    causal_usage = asyncio.run(app.get_causal_budget_usage("job_rejected_compaction_tool_call"))
    assert causal_usage.model_steps == 1
    assert causal_usage.usage.total_tokens == 86

    pricing = PriceBook(
        prices=(
            ModelPrice.fixed(
                provider_name="fake",
                model="summary-model",
                input_per_million=Decimal("1"),
                output_per_million=Decimal("10"),
            ),
        )
    )
    cost = asyncio.run(app.get_session_cost("sess_rejected_compaction_tool_call", pricing))
    assert cost.model_steps == 1
    assert cost.priced_model_steps == 1
    assert cost.total_cost == Decimal("0.00014")
    budget_events = asyncio.run(
        budget_store.load_events_for_budget(
            scope="app",
            key=None,
            window=BudgetWindow.all_time(),
        )
    )
    private_completed = asyncio.run(
        _private_events_for_public_events(
            store,
            "sess_rejected_compaction_tool_call",
            [completed],
        )
    )[0]
    assert [event.id for event in budget_events] == [private_completed.id]


@pytest.mark.parametrize(
    ("compactor_events", "expected_error", "expected_outcome"),
    [
        (
            [ModelStreamEvent.text_delta("partial summary")],
            "Compaction model stream ended without a completed event.",
            "unfinished_stream",
        ),
        (
            [
                ModelStreamEvent.tool_call(
                    id="call_compaction_tool",
                    name="unexpected_tool",
                    arguments={},
                )
            ],
            "Compaction model must not call tools.",
            "rejected_tool_call",
        ),
    ],
    ids=["partial-text", "tool-call"],
)
def test_cayu_app_does_not_invent_usage_for_unfinished_compaction_stream(
    compactor_events: list[ModelStreamEvent],
    expected_error: str,
    expected_outcome: str,
):
    store = InMemorySessionStore()
    budget_store = InMemoryBudgetStore()
    compactor_provider = FakeProvider(compactor_events)
    runtime_provider = FakeProvider(
        [
            ModelStreamEvent.text_delta("unused"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ]
    )
    app = CayuApp(session_store=store, budget_store=budget_store)
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
                session_id="sess_unfinished_compaction",
                messages=[
                    Message.text("user", "old"),
                    Message.text("assistant", "old answer"),
                    Message.text("user", "current"),
                ],
            ),
        )
    )

    assert [event.type for event in events] == [
        EventType.SESSION_STARTED,
        EventType.CONTEXT_COMPACTION_STARTED,
        EventType.MODEL_STARTED,
        EventType.MODEL_COMPLETED,
        EventType.CONTEXT_COMPACTION_FAILED,
        EventType.TURN_COMPLETED,
        EventType.SESSION_FAILED,
    ]
    uncertain = events[3]
    assert uncertain.payload["compaction_outcome"] == expected_outcome
    assert uncertain.payload["usage_unavailable_reason"]
    assert "usage_metrics" not in uncertain.payload
    assert events[4].payload["error_type"] in {"RuntimeError", "_CompactionToolCallError"}
    assert "error" not in events[4].payload
    assert events[5].payload["step_count"] == 1
    assert events[6].payload["error"] == expected_error
    assert runtime_provider.requests == []
    assert checkpoint_without_active_invocation_profile(
        asyncio.run(store.load_checkpoint("sess_unfinished_compaction"))
    ) == {CHECKPOINT_SCHEMA_VERSION_KEY: CURRENT_CHECKPOINT_SCHEMA_VERSION}

    usage = asyncio.run(app.get_session_usage("sess_unfinished_compaction"))
    assert usage.model_steps == 1
    assert usage.usage.total_tokens == 0
    budget_events = asyncio.run(
        budget_store.load_events_for_budget(
            scope="app",
            key=None,
            window=BudgetWindow.all_time(),
        )
    )
    private_uncertain = asyncio.run(
        _private_events_for_public_events(
            store,
            "sess_unfinished_compaction",
            [uncertain],
        )
    )[0]
    assert [event.id for event in budget_events] == [private_uncertain.id]


def test_unfinished_automatic_compaction_fails_closed_after_restart() -> None:
    class UnfinishedThenSuccessfulProvider(ModelProvider):
        name = "fake"

        def __init__(self) -> None:
            self.requests: list[ModelRequest] = []

        async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
            self.requests.append(request)
            if len(self.requests) == 1:
                yield ModelStreamEvent.text_delta("partial summary")
                return
            yield ModelStreamEvent.text_delta("must not run")
            yield ModelStreamEvent.completed(
                {
                    "model": request.model,
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                }
            )

    async def run() -> None:
        store = InMemorySessionStore()
        budget_store = InMemoryBudgetStore()
        compactor_provider = UnfinishedThenSuccessfulProvider()
        policy = BudgetPolicy(
            limits=(
                BudgetLimit(
                    scope="app",
                    max_estimated_cost=Decimal("1"),
                    pricing=compaction_price_book(),
                ),
            )
        )

        def new_app() -> CayuApp:
            app = CayuApp(
                session_store=store,
                budget_store=budget_store,
                budget_policy=policy,
                enable_logging=False,
            )
            app.register_provider(FakeProvider([ModelStreamEvent.completed({})]), default=True)
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
            return app

        first_events = await collect_events(
            new_app(),
            RunRequest(
                agent_name="assistant",
                session_id="sess_unfinished_compaction_first_process",
                messages=[
                    Message.text("user", "old"),
                    Message.text("assistant", "old answer"),
                    Message.text("user", "current"),
                ],
            ),
        )
        assert first_events[-1].type == EventType.SESSION_FAILED
        uncertain = next(
            event
            for event in first_events
            if event.type == EventType.MODEL_COMPLETED
            and event.payload.get("purpose") == "context_compaction"
        )
        assert uncertain.payload["compaction_outcome"] == "unfinished_stream"
        assert uncertain.payload["usage_unavailable_reason"]

        restart_events = await collect_events(
            new_app(),
            RunRequest(
                agent_name="assistant",
                session_id="sess_unfinished_compaction_second_process",
                messages=[
                    Message.text("user", "old"),
                    Message.text("assistant", "old answer"),
                    Message.text("user", "current"),
                ],
            ),
        )

        assert len(compactor_provider.requests) == 1
        assert any(event.type == EventType.BUDGET_LIMIT_REACHED for event in restart_events)
        assert not any(
            event.type == EventType.MODEL_COMPLETED
            and event.payload.get("purpose") == "context_compaction"
            for event in restart_events
        )
        durable_budget_events = await budget_store.load_events_for_budget(
            scope="app",
            key=None,
            window=BudgetWindow.all_time(),
        )
        private_uncertain = (
            await _private_events_for_public_events(
                store,
                "sess_unfinished_compaction_first_process",
                [uncertain],
            )
        )[0]
        assert [event.id for event in durable_budget_events] == [private_uncertain.id]

    asyncio.run(run())

"""Context overflow contracts."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from decimal import Decimal
from typing import Literal

import pytest
from tests.runtime.runtime_compaction_support import RecordingCompactor
from tests.runtime.runtime_provider_support import ContextOverflowProvider, FakeProvider
from tests.runtime.runtime_session_support import (
    _private_events_for_public_events,
    checkpoint_without_model_step_publication,
    collect_events,
    collect_resume_events,
)

from cayu.agents import AgentSpec
from cayu.applications import CayuApp
from cayu.budgets.base import BudgetLimit, BudgetPolicy, BudgetReservation
from cayu.budgets.pricing import ModelPrice, PriceBook
from cayu.budgets.run_limits import RunLimits
from cayu.context.base import (
    CheckpointCompactionContextPolicy,
    MessageWindowContextPolicy,
    ModelCompactor,
    PromptCacheCompactor,
    RecentTurnsContextPolicy,
)
from cayu.events import EventType
from cayu.messages import Message
from cayu.providers.base import (
    ModelContextOverflowError,
    ModelProvider,
    ModelRequest,
    ModelStreamEvent,
)
from cayu.sessions.checkpoints import (
    CHECKPOINT_SCHEMA_VERSION_KEY,
    CURRENT_CHECKPOINT_SCHEMA_VERSION,
)
from cayu.sessions.memory import InMemorySessionStore
from cayu.sessions.requests import ResumeRequest, RunRequest


class MessageCountOverflowProvider(ModelProvider):
    """Overflows full history but accepts a bounded two-message rebuild."""

    name = "message-count-overflow"

    def __init__(self) -> None:
        self.requests: list[ModelRequest] = []

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        self.requests.append(request)
        if len(request.messages) >= 3:
            raise ModelContextOverflowError(
                "context too large",
                provider=self.name,
                status_code=400,
                error_code="context_length_exceeded",
            )
        yield ModelStreamEvent.text_delta("bounded response")
        yield ModelStreamEvent.completed(
            {
                "finish_reason": "stop",
                "usage": {"input_tokens": 10, "output_tokens": 2},
            }
        )


class EventFlattenedOverflowProvider(ModelProvider):
    """Contract-violating provider: flattens overflow into an error event.

    `stream()` must raise `ModelContextOverflowError`, but a provider that
    wraps every failure with `ModelStreamEvent.error(..., cause=exc)` still
    carries the typed overflow identity in the payload. The runtime must route
    that into context-overflow recovery instead of generic retries.
    """

    name = "overflow-event"

    def __init__(self) -> None:
        self.requests: list[ModelRequest] = []

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        self.requests.append(request)
        if len(self.requests) == 1:
            yield ModelStreamEvent.error(
                "context too large",
                cause=ModelContextOverflowError(
                    "context too large",
                    provider=self.name,
                    status_code=400,
                    error_code="context_length_exceeded",
                ),
            )
            return
        yield ModelStreamEvent.text_delta("recovered")
        yield ModelStreamEvent.completed({"finish_reason": "stop"})


def test_context_overflow_policy_rebuilds_context_and_retries_once():
    provider = ContextOverflowProvider()
    app = CayuApp()
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        context_overflow_policy=RecentTurnsContextPolicy(max_user_turns=1),
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="context_overflow_recovery",
                messages=[
                    Message.text("user", "old request"),
                    Message.text("user", "new request"),
                ],
            ),
        )
    )

    assert [request.messages[0].content[0].text for request in provider.requests] == [
        "old request",
        "new request",
    ]
    assert len(provider.requests[0].messages) == 2
    assert len(provider.requests[1].messages) == 1
    assert [
        event.type
        for event in events
        if event.type
        in {
            EventType.CONTEXT_OVERFLOW_DETECTED,
            EventType.CONTEXT_OVERFLOW_RECOVERING,
            EventType.CONTEXT_OVERFLOW_FAILED,
            EventType.SESSION_COMPLETED,
        }
    ] == [
        EventType.CONTEXT_OVERFLOW_DETECTED,
        EventType.CONTEXT_OVERFLOW_RECOVERING,
        EventType.SESSION_COMPLETED,
    ]
    first_started = next(event for event in events if event.type == EventType.MODEL_STARTED)
    model_error = next(event for event in events if event.type == EventType.MODEL_ERROR)
    assert model_error.payload == {
        "error": "context too large",
        "error_type": "ModelContextOverflowError",
        "provider_name": "overflow",
        "requested_model": "fake-model",
        "stage": "provider_dispatch",
        "context_overflow": True,
        "provider": "overflow",
        "status_code": 400,
        "provider_error_code": "context_length_exceeded",
        "retryable": False,
        "step": 1,
        "attempt": 1,
        "max_attempts": 5,
        "model_step_id": first_started.payload["model_step_id"],
        "model_attempt_id": first_started.payload["model_attempt_id"],
        "execution_profile_fingerprint": first_started.payload["execution_profile_fingerprint"],
    }
    detected = next(event for event in events if event.type == EventType.CONTEXT_OVERFLOW_DETECTED)
    assert detected.payload == {
        "step": 1,
        "phase": "initial",
        "error": "context too large",
        "error_type": "ModelContextOverflowError",
        "provider": "overflow",
        "original_message_count": 2,
        "status_code": 400,
        "provider_error_code": "context_length_exceeded",
        "model_step_id": first_started.payload["model_step_id"],
        "model_attempt_id": first_started.payload["model_attempt_id"],
        "execution_profile_fingerprint": first_started.payload["execution_profile_fingerprint"],
    }
    recovering = next(
        event for event in events if event.type == EventType.CONTEXT_OVERFLOW_RECOVERING
    )
    assert recovering.payload == {
        "step": 1,
        "original_message_count": 2,
        "recovery_message_count": 1,
        "policy": "RecentTurnsContextPolicy",
        "model_step_id": first_started.payload["model_step_id"],
        "model_attempt_id": first_started.payload["model_attempt_id"],
        "execution_profile_fingerprint": first_started.payload["execution_profile_fingerprint"],
    }
    transcript = asyncio.run(app.session_store.load_transcript("context_overflow_recovery"))
    assert [message.content[0].text for message in transcript if message.role == "user"] == [
        "old request",
        "new request",
    ]


def test_context_overflow_recovery_from_error_stream_event():
    provider = EventFlattenedOverflowProvider()
    app = CayuApp()
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        context_overflow_policy=RecentTurnsContextPolicy(max_user_turns=1),
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="context_overflow_event_recovery",
                messages=[
                    Message.text("user", "old request"),
                    Message.text("user", "new request"),
                ],
            ),
        )
    )

    assert len(provider.requests) == 2
    assert len(provider.requests[1].messages) == 1
    # The typed overflow is rehydrated before the generic retry path. It still
    # receives one terminal model.error so durable budget evidence can join the
    # dispatched attempt exactly.
    assert [event.type for event in events].count(EventType.MODEL_ERROR) == 1
    assert [
        event.type
        for event in events
        if event.type
        in {
            EventType.CONTEXT_OVERFLOW_DETECTED,
            EventType.CONTEXT_OVERFLOW_RECOVERING,
            EventType.CONTEXT_OVERFLOW_FAILED,
            EventType.SESSION_COMPLETED,
        }
    ] == [
        EventType.CONTEXT_OVERFLOW_DETECTED,
        EventType.CONTEXT_OVERFLOW_RECOVERING,
        EventType.SESSION_COMPLETED,
    ]
    first_started = next(event for event in events if event.type == EventType.MODEL_STARTED)
    model_error = next(event for event in events if event.type == EventType.MODEL_ERROR)
    assert {
        "model_step_id": model_error.payload["model_step_id"],
        "model_attempt_id": model_error.payload["model_attempt_id"],
    } == {
        "model_step_id": first_started.payload["model_step_id"],
        "model_attempt_id": first_started.payload["model_attempt_id"],
    }
    assert model_error.payload["stage"] == "provider_dispatch"
    assert model_error.payload["context_overflow"] is True
    assert model_error.payload["retryable"] is False
    detected = next(event for event in events if event.type == EventType.CONTEXT_OVERFLOW_DETECTED)
    assert detected.payload == {
        "step": 1,
        "phase": "initial",
        "error": "context too large",
        "error_type": "ModelContextOverflowError",
        "provider": "overflow-event",
        "original_message_count": 2,
        "status_code": 400,
        "provider_error_code": "context_length_exceeded",
        "model_step_id": first_started.payload["model_step_id"],
        "model_attempt_id": first_started.payload["model_attempt_id"],
        "execution_profile_fingerprint": first_started.payload["execution_profile_fingerprint"],
    }


def test_context_overflow_error_stream_event_without_policy_fails_typed():
    provider = EventFlattenedOverflowProvider()
    app = CayuApp()
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="context_overflow_event_no_policy",
                messages=[Message.text("user", "too much")],
            ),
        )
    )

    assert len(provider.requests) == 1
    assert [event.type for event in events].count(EventType.MODEL_ERROR) == 1
    failed = events[-1]
    assert failed.type == EventType.SESSION_FAILED
    assert failed.payload["error"] == "context too large"
    assert failed.payload["error_type"] == "ModelContextOverflowError"


def test_context_overflow_policy_can_checkpoint_compaction_before_retry():
    provider = ContextOverflowProvider(
        success_events=[
            ModelStreamEvent.text_delta("recovered"),
            ModelStreamEvent.completed({"usage": {"input_tokens": 1, "output_tokens": 0}}),
        ]
    )
    compactor = RecordingCompactor()
    app = CayuApp(
        budget_policy=BudgetPolicy(
            limits=(
                BudgetLimit(
                    scope="app",
                    max_estimated_cost=Decimal("10"),
                    pricing=PriceBook(
                        prices=(
                            ModelPrice.fixed(
                                provider_name="overflow",
                                model="fake-model",
                                input_per_million=Decimal("1"),
                                output_per_million=Decimal("10"),
                            ),
                        )
                    ),
                ),
            )
        )
    )
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        context_overflow_policy=CheckpointCompactionContextPolicy(
            compactor=compactor,
            max_user_turns=1,
            compact_after_messages=1,
        ),
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="context_overflow_compaction_recovery",
                messages=[
                    Message.text("user", "old request"),
                    Message.text("user", "new request"),
                ],
            ),
        )
    )

    assert len(compactor.requests) == 1
    assert [message.content[0].text for message in compactor.requests[0].messages] == [
        "old request"
    ]
    assert len(provider.requests) == 2
    assert len(provider.requests[1].messages) == 2
    assert (
        provider.requests[1]
        .messages[0]
        .content[0]
        .text.startswith("Previous session context summary:")
    )
    assert provider.requests[1].messages[1].content[0].text == "new request"
    assert EventType.SESSION_CHECKPOINTED in [event.type for event in events]
    # Deterministic compaction has no provider spend, so it must not add a
    # third budget boundary between the initial check and post-model check.
    assert [event.type for event in events].count(EventType.BUDGET_CHECKED) == 2
    checkpoint = asyncio.run(
        app.session_store.load_checkpoint("context_overflow_compaction_recovery")
    )
    assert checkpoint_without_model_step_publication(checkpoint) == {
        CHECKPOINT_SCHEMA_VERSION_KEY: CURRENT_CHECKPOINT_SCHEMA_VERSION,
        "context_compaction": {
            "version": 2,
            "summary": "old request",
            "compacted_transcript_cursor": 1,
            "metadata": {"request_count": 1},
        },
    }


@pytest.mark.parametrize("scope", ["run", "session"])
def test_context_overflow_model_compaction_stops_before_recovery_at_token_limit(
    scope: Literal["run", "session"],
) -> None:
    store = InMemorySessionStore()
    runtime_provider = ContextOverflowProvider()
    compactor_provider = FakeProvider(
        [
            ModelStreamEvent.text_delta("model summary"),
            ModelStreamEvent.completed(
                {
                    "model": "summary-model",
                    "usage": {"input_tokens": 16, "output_tokens": 4},
                }
            ),
        ]
    )
    app = CayuApp(session_store=store)
    app.register_provider(runtime_provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        context_overflow_policy=CheckpointCompactionContextPolicy(
            compactor=ModelCompactor(
                provider=compactor_provider,
                model="summary-model",
            ),
            max_user_turns=1,
            compact_after_messages=1,
        ),
    )
    session_id = f"context_overflow_compaction_{scope}_token_limit"

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[
                    Message.text("user", "old request"),
                    Message.text("user", "new request"),
                ],
                limits=RunLimits(max_total_tokens=15, scope=scope),
            ),
        )
    )

    assert len(runtime_provider.requests) == 1
    assert len(compactor_provider.requests) == 1
    assert EventType.CONTEXT_OVERFLOW_RECOVERING not in [event.type for event in events]
    assert [
        event.type
        for event in events
        if event.type
        in {
            EventType.CONTEXT_OVERFLOW_DETECTED,
            EventType.CONTEXT_COMPACTION_STARTED,
            EventType.MODEL_COMPLETED,
            EventType.CONTEXT_COMPACTION_COMPLETED,
            EventType.SESSION_CHECKPOINTED,
            EventType.SESSION_LIMIT_REACHED,
            EventType.TURN_COMPLETED,
            EventType.SESSION_INTERRUPTED,
        }
    ] == [
        EventType.CONTEXT_OVERFLOW_DETECTED,
        EventType.CONTEXT_COMPACTION_STARTED,
        EventType.MODEL_COMPLETED,
        EventType.CONTEXT_COMPACTION_COMPLETED,
        EventType.SESSION_CHECKPOINTED,
        EventType.SESSION_LIMIT_REACHED,
        EventType.TURN_COMPLETED,
        EventType.SESSION_INTERRUPTED,
    ]
    limit_event = next(event for event in events if event.type == EventType.SESSION_LIMIT_REACHED)
    assert limit_event.payload["limit"] == "total_tokens"
    assert limit_event.payload["actual"] == 20
    assert limit_event.payload["maximum"] == 15
    checkpoint = asyncio.run(store.load_checkpoint(session_id))
    assert checkpoint is not None
    assert checkpoint["context_compaction"]["summary"] == "model summary"
    usage = asyncio.run(app.get_session_usage(session_id))
    assert usage.model_steps == 1
    assert usage.usage.total_tokens == 20


def test_context_overflow_hierarchy_enforces_run_limit_before_every_dispatch() -> None:
    store = InMemorySessionStore()
    runtime_provider = ContextOverflowProvider()
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
    app = CayuApp(session_store=store, enable_logging=False)
    app.register_provider(runtime_provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        context_overflow_policy=CheckpointCompactionContextPolicy(
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
                session_id="context_overflow_hierarchy_run_limit",
                messages=[
                    Message.text("user", "oversized " + "x" * 5000),
                    Message.text("user", "current"),
                ],
                limits=RunLimits(max_total_tokens=10),
            ),
        )
    )

    assert len(runtime_provider.requests) == 1
    assert len(compactor_provider.requests) == 1
    event_types = [event.type for event in events]
    assert EventType.CONTEXT_OVERFLOW_RECOVERING not in event_types
    assert event_types.index(EventType.CONTEXT_COMPACTION_FAILED) < event_types.index(
        EventType.SESSION_LIMIT_REACHED
    )
    assert events[-1].type == EventType.SESSION_INTERRUPTED
    checkpoint = asyncio.run(store.load_checkpoint("context_overflow_hierarchy_run_limit"))
    assert checkpoint is None or "context_compaction" not in checkpoint


def test_context_overflow_model_compaction_settles_dispatch_before_budget_gate() -> None:
    store = InMemorySessionStore()
    runtime_provider = ContextOverflowProvider()
    compactor_provider = FakeProvider(
        [
            ModelStreamEvent.text_delta("model summary"),
            ModelStreamEvent.completed(
                {
                    "model": "summary-model",
                    "usage": {"input_tokens": 20, "output_tokens": 0},
                }
            ),
        ]
    )
    pricing = PriceBook(
        prices=(
            ModelPrice.fixed(
                provider_name="overflow",
                model="fake-model",
                input_per_million=Decimal("1"),
                output_per_million=Decimal("10"),
            ),
            ModelPrice.fixed(
                provider_name="fake",
                model="summary-model",
                input_per_million=Decimal("1"),
                output_per_million=Decimal("10"),
            ),
        )
    )
    app = CayuApp(
        session_store=store,
        budget_policy=BudgetPolicy(
            limits=(
                BudgetLimit(
                    scope="app",
                    max_estimated_cost=Decimal("0.00002"),
                    pricing=pricing,
                    reservation=BudgetReservation(
                        max_input_tokens=10,
                        max_output_tokens=0,
                    ),
                ),
            )
        ),
    )
    app.register_provider(runtime_provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        context_overflow_policy=CheckpointCompactionContextPolicy(
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
                session_id="context_overflow_compaction_budget_limit",
                messages=[
                    Message.text("user", "old request"),
                    Message.text("user", "new request"),
                ],
            ),
        )
    )

    event_types = [event.type for event in events]
    assert len(runtime_provider.requests) == 1
    assert len(compactor_provider.requests) == 1
    assert event_types.count(EventType.BUDGET_RESERVED) == 2
    assert EventType.CONTEXT_OVERFLOW_RECOVERING not in event_types
    checkpoint_index = event_types.index(EventType.SESSION_CHECKPOINTED)
    reconciliations = [
        (index, event)
        for index, event in enumerate(events)
        if event.type == EventType.BUDGET_RECONCILED
    ]
    compaction_reconciliation_index, compaction_reconciliation = reconciliations[0]
    dispatch_reconciliation_index, dispatch_reconciliation = reconciliations[1]
    reached_index = event_types.index(EventType.BUDGET_LIMIT_REACHED)
    assert compaction_reconciliation_index < checkpoint_index
    assert checkpoint_index < dispatch_reconciliation_index < reached_index
    assert Decimal(compaction_reconciliation.payload["actual_amount"]) == Decimal("0.00002")
    assert compaction_reconciliation.payload["reason"] == (
        "automatic context compaction model completed"
    )
    assert Decimal(dispatch_reconciliation.payload["actual_amount"]) == Decimal("0.00001")
    assert (
        dispatch_reconciliation.payload["reason"]
        == "provider usage unknown after dispatch; charged reserved amount"
    )
    reached = events[reached_index]
    assert Decimal(reached.payload["actual"]) == Decimal("0.00002")
    assert reached.payload["scope"] == "app"
    assert events[-1].type == EventType.SESSION_INTERRUPTED
    cost = asyncio.run(app.get_session_cost("context_overflow_compaction_budget_limit", pricing))
    assert cost.model_steps == 1
    assert cost.total_cost == Decimal("0.00002")


def test_context_overflow_compaction_reservation_denial_settles_before_terminal() -> None:
    runtime_provider = ContextOverflowProvider()
    compactor_provider = FakeProvider([ModelStreamEvent.completed({})])
    pricing = PriceBook(
        prices=(
            ModelPrice.fixed(
                provider_name="overflow",
                model="fake-model",
                input_per_million=Decimal("1"),
                output_per_million=Decimal("10"),
            ),
            ModelPrice.fixed(
                provider_name="fake",
                model="summary-model",
                input_per_million=Decimal("1"),
                output_per_million=Decimal("10"),
            ),
        )
    )
    app = CayuApp(
        budget_policy=BudgetPolicy(
            limits=(
                BudgetLimit(
                    scope="app",
                    max_estimated_cost=Decimal("0.000019"),
                    pricing=pricing,
                    reservation=BudgetReservation(
                        max_input_tokens=10,
                        max_output_tokens=0,
                    ),
                ),
            )
        )
    )
    app.register_provider(runtime_provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        context_overflow_policy=CheckpointCompactionContextPolicy(
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
                session_id="context_overflow_compaction_reservation_denied",
                messages=[
                    Message.text("user", "old request"),
                    Message.text("user", "new request"),
                ],
            ),
        )
    )

    event_types = [event.type for event in events]
    assert len(runtime_provider.requests) == 1
    assert compactor_provider.requests == []
    assert event_types.index(EventType.BUDGET_RECONCILED) < event_types.index(
        EventType.BUDGET_LIMIT_REACHED
    )
    assert events[-1].type == EventType.SESSION_INTERRUPTED


def test_prompt_cache_compactor_bounds_context_overflow_recovery() -> None:
    provider = MessageCountOverflowProvider()
    app = CayuApp()
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        context_overflow_policy=CheckpointCompactionContextPolicy(
            compactor=PromptCacheCompactor(provider=provider),
            max_user_turns=1,
            compact_after_messages=1,
        ),
    )

    first_events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="prompt_cache_overflow_recovery",
                messages=[Message.text("user", "old request")],
            ),
        )
    )
    assert first_events[-1].type == EventType.SESSION_COMPLETED

    recovery_events = asyncio.run(
        collect_resume_events(
            app,
            ResumeRequest(
                session_id="prompt_cache_overflow_recovery",
                messages=[Message.text("user", "new request")],
            ),
        )
    )

    assert recovery_events[-1].type == EventType.SESSION_COMPLETED
    assert [len(request.messages) for request in provider.requests] == [1, 3, 2, 2]
    completed = [
        event for event in recovery_events if event.type == EventType.CONTEXT_COMPACTION_COMPLETED
    ]
    assert len(completed) == 1
    assert completed[0].payload["chunk_mode"] == "single_request"


def test_context_overflow_without_policy_fails_without_retry():
    provider = ContextOverflowProvider()
    app = CayuApp()
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="context_overflow_no_policy",
                messages=[Message.text("user", "too much")],
            ),
        )
    )

    assert len(provider.requests) == 1
    assert EventType.CONTEXT_OVERFLOW_DETECTED not in [event.type for event in events]
    assert [event.type for event in events].count(EventType.MODEL_ERROR) == 1
    failed = events[-1]
    assert failed.type == EventType.SESSION_FAILED
    assert failed.payload["error"] == "context too large"
    assert failed.payload["error_type"] == "ModelContextOverflowError"


def test_context_overflow_policy_fails_cleanly_when_recovery_overflows():
    provider = ContextOverflowProvider(overflow_requests=2)
    app = CayuApp()
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        context_overflow_policy=MessageWindowContextPolicy(max_messages=1),
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="context_overflow_recovery_failed",
                messages=[
                    Message.text("user", "old request"),
                    Message.text("user", "new request"),
                ],
            ),
        )
    )

    assert len(provider.requests) == 2
    model_errors = [event for event in events if event.type == EventType.MODEL_ERROR]
    assert len(model_errors) == 2
    private_events = asyncio.run(
        _private_events_for_public_events(
            app.session_store,
            "context_overflow_recovery_failed",
            events,
        )
    )
    private_model_errors = [
        event for event in private_events if event.type == EventType.MODEL_ERROR
    ]
    assert (
        private_model_errors[0].payload["model_step_id"]
        == private_model_errors[1].payload["model_step_id"]
    )
    assert (
        private_model_errors[0].payload["model_attempt_id"]
        != private_model_errors[1].payload["model_attempt_id"]
    )
    assert [
        event.type
        for event in events
        if event.type
        in {
            EventType.CONTEXT_OVERFLOW_DETECTED,
            EventType.CONTEXT_OVERFLOW_RECOVERING,
            EventType.CONTEXT_OVERFLOW_FAILED,
        }
    ] == [
        EventType.CONTEXT_OVERFLOW_DETECTED,
        EventType.CONTEXT_OVERFLOW_RECOVERING,
        EventType.CONTEXT_OVERFLOW_FAILED,
    ]
    recovery_started = [event for event in private_events if event.type == EventType.MODEL_STARTED][
        -1
    ]
    failed = next(
        event for event in private_events if event.type == EventType.CONTEXT_OVERFLOW_FAILED
    )
    assert failed.payload == {
        "step": 1,
        "phase": "recovery",
        "error": "context too large",
        "error_type": "ModelContextOverflowError",
        "provider": "overflow",
        "original_message_count": 2,
        "status_code": 400,
        "provider_error_code": "context_length_exceeded",
        "recovery_message_count": 1,
        "model_step_id": recovery_started.payload["model_step_id"],
        "model_attempt_id": recovery_started.payload["model_attempt_id"],
        "execution_profile_fingerprint": recovery_started.payload["execution_profile_fingerprint"],
    }
    assert events[-1].type == EventType.SESSION_FAILED

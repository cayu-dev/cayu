"""Compaction budget admission contracts."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from datetime import datetime
from decimal import Decimal
from typing import Literal

import pytest
from tests.runtime.runtime_compaction_support import RecordingCompactor, compaction_price_book
from tests.runtime.runtime_provider_support import FakeProvider
from tests.runtime.runtime_session_support import (
    _AllowForkProfileAdoption,
    collect_events,
    collect_resume_events,
)

import cayu.runtime._run_limits as run_limits_module
from cayu._exception_groups import exception_cause, iter_exception_tree
from cayu.agents import AgentSpec
from cayu.applications import CayuApp
from cayu.approvals.tools import ResolutionActor, ResolutionActorSource
from cayu.budgets.base import BudgetLimit, BudgetPolicy, BudgetReservation, InMemoryBudgetLedger
from cayu.budgets.run_limits import RunLimits
from cayu.context.base import (
    CheckpointCompactionContextPolicy,
    CompactionRequest,
    CompactionResult,
    ContextCompactor,
    MessageWindowContextPolicy,
    ModelCompactor,
    PromptCacheCompactor,
    UsageTriggeredContextPolicy,
)
from cayu.events import EventType
from cayu.messages import Message
from cayu.providers.base import (
    ModelContextOverflowError,
    ModelProvider,
    ModelProviderError,
    ModelRequest,
    ModelStreamEvent,
)
from cayu.providers.retry_policy import RetryPolicy
from cayu.runtime._model_completion_contracts import ModelCompletionManualRecoveryRequired
from cayu.runtime.execution_profiles import ExecutionProfileAdoptionIntent
from cayu.sessions.memory import InMemorySessionStore
from cayu.sessions.records import Session, SessionStatus
from cayu.sessions.recovery import IncompleteSessionRecoveryRequest
from cayu.sessions.requests import ResumeRequest, RunRequest


@pytest.mark.parametrize("scope", ["run", "session"])
def test_cayu_app_stops_before_main_model_when_compaction_reaches_token_limit(
    scope: Literal["run", "session"],
):
    store = InMemorySessionStore()
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
    runtime_provider = FakeProvider([ModelStreamEvent.completed({})])
    app = CayuApp(session_store=store)
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
    session_id = f"sess_compaction_{scope}_token_limit"

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[
                    Message.text("user", "old"),
                    Message.text("assistant", "old answer"),
                    Message.text("user", "current"),
                ],
                limits=RunLimits(max_total_tokens=15, scope=scope),
            ),
        )
    )

    assert [event.type for event in events] == [
        EventType.SESSION_STARTED,
        EventType.CONTEXT_COMPACTION_STARTED,
        EventType.MODEL_STARTED,
        EventType.MODEL_COMPLETED,
        EventType.CONTEXT_COMPACTION_COMPLETED,
        EventType.SESSION_CHECKPOINTED,
        EventType.SESSION_LIMIT_REACHED,
        EventType.TURN_COMPLETED,
        EventType.SESSION_INTERRUPTED,
    ]
    assert events[6].payload["limit"] == "total_tokens"
    assert events[6].payload["actual"] == 20
    assert events[6].payload["maximum"] == 15
    assert events[7].payload["token_usage"]["total_tokens"] == 20
    assert len(compactor_provider.requests) == 1
    assert runtime_provider.requests == []
    checkpoint = asyncio.run(store.load_checkpoint(session_id))
    assert checkpoint is not None
    assert checkpoint["context_compaction"]["summary"] == "model summary"
    usage = asyncio.run(app.get_session_usage(session_id))
    assert usage.model_steps == 1
    assert usage.usage.total_tokens == 20


def test_resume_after_compaction_limit_stop_uses_latest_agent_completion_for_context() -> None:
    store = InMemorySessionStore()
    compactor_provider = FakeProvider(
        [
            [
                ModelStreamEvent.text_delta("first model summary"),
                ModelStreamEvent.completed(
                    {
                        "model": "summary-model",
                        "usage": {
                            "input_tokens": 16,
                            "output_tokens": 4,
                            "total_tokens": 20,
                        },
                    }
                ),
            ],
            [
                ModelStreamEvent.text_delta("second model summary"),
                ModelStreamEvent.completed(
                    {
                        "model": "summary-model",
                        "usage": {
                            "input_tokens": 16,
                            "output_tokens": 4,
                            "total_tokens": 20,
                        },
                    }
                ),
            ],
        ]
    )
    runtime_provider = FakeProvider(
        [
            [
                ModelStreamEvent.text_delta("first answer"),
                ModelStreamEvent.completed(
                    {
                        "usage": {
                            "input_tokens": 8,
                            "output_tokens": 2,
                            "total_tokens": 10,
                        }
                    }
                ),
            ],
            [
                ModelStreamEvent.text_delta("resumed answer"),
                ModelStreamEvent.completed(
                    {
                        "usage": {
                            "input_tokens": 8,
                            "output_tokens": 2,
                            "total_tokens": 10,
                        }
                    }
                ),
            ],
        ]
    )
    policy = UsageTriggeredContextPolicy(
        base_policy=CheckpointCompactionContextPolicy(
            compactor=ModelCompactor(
                provider=compactor_provider,
                model="summary-model",
            ),
            max_user_turns=1,
            compact_after_messages=2,
        ),
        triggered_policy=MessageWindowContextPolicy(max_messages=1),
        min_total_tokens=15,
        sticky=False,
    )
    app = CayuApp(
        session_store=store,
        execution_profile_policy=_AllowForkProfileAdoption(),
        enable_logging=False,
    )
    app.register_provider(runtime_provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        context_policy=policy,
    )
    session_id = "sess_resume_after_compaction_limit_stop"
    limits = RunLimits(max_total_tokens=15, scope="run")

    first_events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "first request")],
                limits=limits,
            ),
        )
    )
    stopped_events = asyncio.run(
        collect_resume_events(
            app,
            ResumeRequest(
                session_id=session_id,
                messages=[Message.text("user", "second request")],
                limits=limits,
            ),
        )
    )

    assert first_events[-1].type == EventType.SESSION_COMPLETED
    assert stopped_events[-1].type == EventType.SESSION_INTERRUPTED
    assert any(event.type == EventType.SESSION_LIMIT_REACHED for event in stopped_events)
    assert len(runtime_provider.requests) == 1
    assert len(compactor_provider.requests) == 1

    resumed_events = asyncio.run(
        collect_resume_events(
            app,
            ResumeRequest(
                session_id=session_id,
                messages=[Message.text("user", "third request")],
                profile_adoption=ExecutionProfileAdoptionIntent(
                    idempotency_key="resume-after-compaction-limit-stop",
                    reason="The operator authorizes a fresh run without the prior run limit.",
                    requested_by=ResolutionActor(
                        subject="test-operator",
                        source=ResolutionActorSource.REQUEST,
                    ),
                ),
            ),
        )
    )

    assert resumed_events[-1].type == EventType.SESSION_COMPLETED
    assert len(compactor_provider.requests) == 2
    assert len(runtime_provider.requests) == 2
    resumed_projection = json.dumps(
        [message.model_dump(mode="json") for message in runtime_provider.requests[-1].messages]
    )
    assert "second model summary" in resumed_projection
    assert "third request" in resumed_projection


def test_cayu_app_stops_before_main_model_when_compaction_reaches_app_budget():
    store = InMemorySessionStore()
    pricing = compaction_price_book()
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
    runtime_provider = FakeProvider([ModelStreamEvent.completed({})])
    app = CayuApp(
        session_store=store,
        budget_policy=BudgetPolicy(
            limits=(
                BudgetLimit(
                    scope="app",
                    max_estimated_cost=Decimal("0.00002"),
                    pricing=pricing,
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
                session_id="sess_compaction_app_budget",
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
        EventType.BUDGET_CHECKED,
        EventType.CONTEXT_COMPACTION_STARTED,
        EventType.BUDGET_CHECKED,
        EventType.MODEL_STARTED,
        EventType.MODEL_COMPLETED,
        EventType.CONTEXT_COMPACTION_COMPLETED,
        EventType.SESSION_CHECKPOINTED,
        EventType.BUDGET_CHECKED,
        EventType.BUDGET_LIMIT_REACHED,
        EventType.SESSION_LIMIT_REACHED,
        EventType.TURN_COMPLETED,
        EventType.SESSION_INTERRUPTED,
    ]
    assert Decimal(events[8].payload["actual"]) == Decimal("0.00002")
    assert events[9].payload["scope"] == "app"
    assert events[10].payload["limit"] == "estimated_cost"
    assert runtime_provider.requests == []
    checkpoint = asyncio.run(store.load_checkpoint("sess_compaction_app_budget"))
    assert checkpoint is not None
    cost = asyncio.run(app.get_session_cost("sess_compaction_app_budget", pricing))
    assert cost.model_steps == 1
    assert cost.priced_model_steps == 1
    assert cost.total_cost == Decimal("0.00002")


@pytest.mark.parametrize(
    ("maximum", "main_provider_calls", "terminal_event"),
    [
        pytest.param(
            "0.000029",
            0,
            EventType.SESSION_INTERRUPTED,
            id="combined-reservation-above-cap",
        ),
        pytest.param(
            "0.000030",
            1,
            EventType.SESSION_INTERRUPTED,
            id="combined-reservation-at-cap",
        ),
        pytest.param(
            "0.000031",
            1,
            EventType.SESSION_COMPLETED,
            id="combined-reservation-below-cap",
        ),
    ],
)
def test_cayu_app_compaction_spend_participates_in_strict_main_admission(
    maximum: str,
    main_provider_calls: int,
    terminal_event: EventType,
) -> None:
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
    runtime_provider = FakeProvider(
        [
            ModelStreamEvent.text_delta("final answer"),
            ModelStreamEvent.completed({"usage": {"input_tokens": 10, "output_tokens": 0}}),
        ]
    )
    app = CayuApp(
        budget_policy=BudgetPolicy(
            limits=(
                BudgetLimit(
                    scope="app",
                    max_estimated_cost=Decimal(maximum),
                    pricing=compaction_price_book(),
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
                session_id=f"sess_compaction_strict_admission_{maximum}",
                messages=[
                    Message.text("user", "old"),
                    Message.text("assistant", "old answer"),
                    Message.text("user", "current"),
                ],
            ),
        )
    )

    assert len(compactor_provider.requests) == 1
    assert len(runtime_provider.requests) == main_provider_calls
    reconciliations = [event for event in events if event.type == EventType.BUDGET_RECONCILED]
    assert Decimal(reconciliations[0].payload["actual_amount"]) == Decimal("0.00002")
    if main_provider_calls == 0:
        failure = next(
            event for event in events if event.type == EventType.BUDGET_RESERVATION_FAILED
        )
        assert Decimal(failure.payload["actual"]) == Decimal("0.00003")
        assert Decimal(failure.payload["requested"]) == Decimal("0.00001")
    else:
        assert [Decimal(event.payload["actual_amount"]) for event in reconciliations] == [
            Decimal("0.00002"),
            Decimal("0.00001"),
        ]
    assert events[-1].type == terminal_event


def test_cayu_app_rejects_automatic_compaction_before_provider_dispatch() -> None:
    compactor_provider = FakeProvider([ModelStreamEvent.completed({})])
    runtime_provider = FakeProvider([ModelStreamEvent.completed({})])
    app = CayuApp(
        budget_policy=BudgetPolicy(
            limits=(
                BudgetLimit(
                    scope="app",
                    max_estimated_cost=Decimal("0.000009"),
                    pricing=compaction_price_book(),
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
                session_id="sess_compaction_reservation_rejected",
                messages=[
                    Message.text("user", "old"),
                    Message.text("assistant", "old answer"),
                    Message.text("user", "current"),
                ],
            ),
        )
    )

    assert compactor_provider.requests == []
    assert runtime_provider.requests == []
    assert EventType.REQUEST_FOOTPRINT_RECORDED not in [event.type for event in events]
    assert EventType.MODEL_STARTED not in [event.type for event in events]
    assert EventType.BUDGET_RESERVATION_FAILED in [event.type for event in events]
    assert EventType.BUDGET_RESERVED not in [event.type for event in events]
    assert EventType.BUDGET_RECONCILED not in [event.type for event in events]
    assert events[-1].type == EventType.SESSION_INTERRUPTED


def test_cayu_app_automatic_compaction_failure_charges_reserved_amount() -> None:
    compactor_provider = FakeProvider([ModelStreamEvent.error("compactor unavailable")])
    runtime_provider = FakeProvider([ModelStreamEvent.completed({})])
    app = CayuApp(
        budget_policy=BudgetPolicy(
            limits=(
                BudgetLimit(
                    scope="app",
                    max_estimated_cost=Decimal("1"),
                    pricing=compaction_price_book(),
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
                session_id="sess_compaction_provider_failure_budget",
                messages=[
                    Message.text("user", "old"),
                    Message.text("assistant", "old answer"),
                    Message.text("user", "current"),
                ],
            ),
        )
    )

    assert len(compactor_provider.requests) == 1
    assert runtime_provider.requests == []
    reconciliation = next(event for event in events if event.type == EventType.BUDGET_RECONCILED)
    assert Decimal(reconciliation.payload["actual_amount"]) == Decimal("0.00001")
    assert reconciliation.payload["reason"] == (
        "automatic context compaction dispatch has uncertain usage; charged reserved amount"
    )
    assert events[-1].type == EventType.SESSION_FAILED


def test_automatic_compaction_predispatch_failure_does_not_charge_budget(monkeypatch) -> None:
    compactor_provider = FakeProvider([ModelStreamEvent.completed({})])
    runtime_provider = FakeProvider([ModelStreamEvent.completed({})])

    def fail_before_request(request: CompactionRequest, **kwargs) -> str:
        del request, kwargs
        raise ValueError("prompt construction failed")

    monkeypatch.setattr("cayu.context.base._bounded_default_compaction_prompt", fail_before_request)

    app = CayuApp(
        budget_policy=BudgetPolicy(
            limits=(
                BudgetLimit(
                    scope="app",
                    max_estimated_cost=Decimal("1"),
                    pricing=compaction_price_book(),
                    reservation=BudgetReservation(max_input_tokens=10, max_output_tokens=0),
                ),
            )
        )
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
                session_id="sess_compaction_predispatch_failure_budget",
                messages=[
                    Message.text("user", "old"),
                    Message.text("assistant", "old answer"),
                    Message.text("user", "current"),
                ],
            ),
        )
    )

    assert compactor_provider.requests == []
    assert runtime_provider.requests == []
    assert EventType.BUDGET_RESERVED not in [event.type for event in events]
    assert EventType.BUDGET_RECONCILED not in [event.type for event in events]
    assert EventType.BUDGET_RESERVATION_RELEASED not in [event.type for event in events]
    assert events[-1].type == EventType.SESSION_FAILED


def test_automatic_compaction_supports_non_expiring_custom_ledger() -> None:
    class NonExpiringLedger(InMemoryBudgetLedger):
        def __init__(self) -> None:
            super().__init__(reservation_ttl_seconds=None)

        async def heartbeat(self, *, reservation_id: str) -> bool:
            del reservation_id
            raise AssertionError("non-expiring ledgers must not be heartbeated")

    compactor_provider = FakeProvider(
        [
            ModelStreamEvent.text_delta("summary"),
            ModelStreamEvent.completed({"model": "summary-model", "usage": {"input_tokens": 1}}),
        ]
    )
    runtime_provider = FakeProvider(
        [ModelStreamEvent.completed({"usage": {"input_tokens": 1, "output_tokens": 0}})]
    )
    app = CayuApp(
        budget_policy=BudgetPolicy(
            limits=(
                BudgetLimit(
                    scope="app",
                    max_estimated_cost=Decimal("1"),
                    pricing=compaction_price_book(),
                    reservation=BudgetReservation(max_input_tokens=10, max_output_tokens=0),
                ),
            )
        ),
        budget_ledger=NonExpiringLedger(),
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
                session_id="sess_non_expiring_compaction_budget_ledger",
                messages=[
                    Message.text("user", "old"),
                    Message.text("assistant", "old answer"),
                    Message.text("user", "current"),
                ],
            ),
        )
    )

    assert len(compactor_provider.requests) == 1
    assert len(runtime_provider.requests) == 1
    assert events[-1].type == EventType.SESSION_COMPLETED


@pytest.mark.parametrize(
    ("maximum", "compactor_calls", "runtime_calls", "terminal_event"),
    [
        pytest.param(
            "0.000019",
            1,
            0,
            EventType.SESSION_INTERRUPTED,
            id="retry-reservation-denied",
        ),
        pytest.param(
            "1",
            1,
            0,
            EventType.SESSION_INTERRUPTED,
            id="retry-uncertain-attempt-fails-closed",
        ),
    ],
)
def test_automatic_compaction_retry_requires_an_independent_reservation(
    maximum: str,
    compactor_calls: int,
    runtime_calls: int,
    terminal_event: EventType,
) -> None:
    class InheritedModelCompactor(ModelCompactor):
        pass

    class RetryCompactionProvider(ModelProvider):
        name = "fake"

        def __init__(self) -> None:
            self.requests: list[ModelRequest] = []

        async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
            self.requests.append(request)
            if len(self.requests) == 1:
                raise ModelProviderError(
                    "provider overloaded",
                    provider=self.name,
                    status_code=503,
                    retryable=True,
                )
            yield ModelStreamEvent.text_delta("summary")
            yield ModelStreamEvent.completed(
                {"model": "summary-model", "usage": {"input_tokens": 1}}
            )

    compactor_provider = RetryCompactionProvider()
    runtime_provider = FakeProvider(
        [ModelStreamEvent.completed({"usage": {"input_tokens": 1, "output_tokens": 0}})]
    )
    app = CayuApp(
        budget_policy=BudgetPolicy(
            limits=(
                BudgetLimit(
                    scope="app",
                    max_estimated_cost=Decimal(maximum),
                    pricing=compaction_price_book(),
                    reservation=BudgetReservation(max_input_tokens=10, max_output_tokens=0),
                ),
            )
        )
    )
    app.register_provider(runtime_provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        context_policy=CheckpointCompactionContextPolicy(
            compactor=InheritedModelCompactor(
                provider=compactor_provider,
                model="summary-model",
                retry_policy=RetryPolicy(max_attempts=2, initial_delay_s=0.0),
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
                session_id=f"sess_compaction_retry_budget_{maximum}",
                messages=[
                    Message.text("user", "old"),
                    Message.text("assistant", "old answer"),
                    Message.text("user", "current"),
                ],
            ),
        )
    )

    assert len(compactor_provider.requests) == compactor_calls
    assert len(runtime_provider.requests) == runtime_calls
    reconciliations = [event for event in events if event.type == EventType.BUDGET_RECONCILED]
    assert Decimal(reconciliations[0].payload["actual_amount"]) == Decimal("0.00001")
    failure = next(event for event in events if event.type == EventType.BUDGET_RESERVATION_FAILED)
    assert Decimal(failure.payload["actual"]) == Decimal("0")
    assert failure.payload["message"] == (
        "Budget cannot be verified because 1 model step(s) have missing or invalid completion usage."
    )
    assert events[-1].type == terminal_event


def test_automatic_compaction_does_not_retry_after_settlement_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailingReconcileLedger(InMemoryBudgetLedger):
        async def reconcile(
            self,
            *,
            reservation_id: str,
            actual_amount: Decimal,
            reason: str | None = None,
            occurred_at: datetime | None = None,
            **kwargs,
        ):
            del reservation_id, actual_amount, reason, occurred_at, kwargs
            raise RuntimeError("ledger reconciliation failed")

    class RetryableFailureProvider(ModelProvider):
        name = "fake"

        def __init__(self) -> None:
            self.requests: list[ModelRequest] = []

        async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
            self.requests.append(request)
            raise ModelProviderError(
                "provider overloaded",
                provider=self.name,
                status_code=503,
                retryable=True,
            )
            yield

    compactor_provider = RetryableFailureProvider()
    runtime_provider = FakeProvider([ModelStreamEvent.completed({})])
    ledger = FailingReconcileLedger()
    app = CayuApp(
        budget_policy=BudgetPolicy(
            limits=(
                BudgetLimit(
                    scope="app",
                    max_estimated_cost=Decimal("1"),
                    pricing=compaction_price_book(),
                    reservation=BudgetReservation(max_input_tokens=10, max_output_tokens=0),
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
                model="summary-model",
                retry_policy=RetryPolicy(max_attempts=2, initial_delay_s=0.0),
            ),
            max_user_turns=1,
            compact_after_messages=2,
        ),
    )

    session_id = "sess_compaction_settlement_failure_no_retry"

    async def run():
        events = []
        with pytest.raises(ModelProviderError, match="provider overloaded") as exc_info:
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[
                        Message.text("user", "old"),
                        Message.text("assistant", "old answer"),
                        Message.text("user", "current"),
                    ],
                )
            ):
                events.append(event)
        return (
            events,
            exc_info.value,
            await app.session_store.load(session_id),
            await app.session_store.load_active_model_completion_stage(session_id),
        )

    events, failure, session, active = asyncio.run(run())

    assert len(compactor_provider.requests) == 1
    assert runtime_provider.requests == []
    assert EventType.SESSION_FAILED not in [event.type for event in events]
    assert session is not None and session.status == SessionStatus.RUNNING
    assert failure.__notes__ == [
        "Automatic compaction budget settlement also failed: "
        "RuntimeError: ledger reconciliation failed"
    ]
    failure_cause = exception_cause(failure)
    assert isinstance(failure_cause, BaseExceptionGroup)
    cause_failures = tuple(iter_exception_tree(failure_cause))
    assert (
        sum(
            type(item) is RuntimeError and str(item) == "ledger reconciliation failed"
            for item in cause_failures
        )
        == 1
    )
    assert (
        sum(
            isinstance(
                item,
                run_limits_module.ModelCompletionBudgetSettlementPending,
            )
            for item in cause_failures
        )
        == 1
    )
    assert active is not None and active.stage.state == "completed"

    assert active.stage.publication is not None
    [staged_error] = [
        event for event in active.stage.publication.events if event.type is EventType.MODEL_ERROR
    ]
    assert staged_error.payload["status_code"] == 503
    assert staged_error.payload["provider_retryable"] is True
    assert staged_error.payload["retry"] is False
    assert staged_error.payload["retry_disposition"] == "suppressed"
    assert staged_error.payload["retry_suppression"] == "automatic_retry_disabled"

    with pytest.raises(RuntimeError, match="ledger reconciliation failed"):
        asyncio.run(
            app.recover_incomplete_session(
                IncompleteSessionRecoveryRequest(
                    session_id=session_id,
                    inactive_for_seconds=0,
                )
            )
        )

    recovered_active = asyncio.run(app.session_store.load_active_model_completion_stage(session_id))
    assert recovered_active == active
    assert [record.status for record in ledger._records.values()] == ["active"]
    recovered_events = asyncio.run(app.session_store.load_events(session_id))
    assert not any(
        event.type is EventType.MODEL_COMPLETED
        and event.payload.get("purpose") == "context_compaction"
        for event in recovered_events
    )
    assert len(compactor_provider.requests) == 1
    assert runtime_provider.requests == []

    monkeypatch.setattr(ledger, "reconcile", InMemoryBudgetLedger.reconcile.__get__(ledger))
    with pytest.raises(ModelCompletionManualRecoveryRequired, match="without a durable context"):
        asyncio.run(
            app.recover_incomplete_session(
                IncompleteSessionRecoveryRequest(session_id=session_id, inactive_for_seconds=0)
            )
        )
    recovered_events = asyncio.run(app.session_store.load_events(session_id))
    [recovered_error] = [event for event in recovered_events if event.type is EventType.MODEL_ERROR]
    assert recovered_error.id == staged_error.id
    assert recovered_error.payload["status_code"] == 503
    assert [record.status for record in ledger._records.values()] == ["reconciled"]
    assert len(compactor_provider.requests) == 1
    assert runtime_provider.requests == []


def test_prompt_cache_compaction_does_not_fallback_after_settlement_failure() -> None:
    class FailSecondReconcileLedger(InMemoryBudgetLedger):
        def __init__(self) -> None:
            super().__init__()
            self.reconcile_calls = 0

        async def reconcile(
            self,
            *,
            reservation_id: str,
            actual_amount: Decimal,
            reason: str | None = None,
            occurred_at: datetime | None = None,
            **kwargs,
        ):
            self.reconcile_calls += 1
            if self.reconcile_calls == 2:
                raise RuntimeError("ledger reconciliation failed")
            return await super().reconcile(
                reservation_id=reservation_id,
                actual_amount=actual_amount,
                reason=reason,
                occurred_at=occurred_at,
                **kwargs,
            )

    overflow = ModelContextOverflowError(
        "context too large",
        provider="fake",
        status_code=400,
        error_code="context_length_exceeded",
    )
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.text_delta("first answer"),
                ModelStreamEvent.completed({"usage": {"input_tokens": 1, "output_tokens": 0}}),
            ],
            [ModelStreamEvent.error(str(overflow), cause=overflow)],
        ]
    )
    ledger = FailSecondReconcileLedger()
    app = CayuApp(
        budget_policy=BudgetPolicy(
            limits=(
                BudgetLimit(
                    scope="app",
                    max_estimated_cost=Decimal("1"),
                    pricing=compaction_price_book(),
                    reservation=BudgetReservation(max_input_tokens=10, max_output_tokens=0),
                ),
            )
        ),
        budget_ledger=ledger,
    )
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        context_policy=CheckpointCompactionContextPolicy(
            compactor=PromptCacheCompactor(provider=provider),
            max_user_turns=1,
            compact_after_messages=2,
        ),
    )
    first_events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_prompt_cache_settlement_failure",
                messages=[Message.text("user", "first")],
            ),
        )
    )
    assert first_events[-1].type == EventType.SESSION_COMPLETED

    with pytest.raises(ModelContextOverflowError, match="context too large") as exc_info:
        asyncio.run(
            collect_resume_events(
                app,
                ResumeRequest(
                    session_id="sess_prompt_cache_settlement_failure",
                    messages=[Message.text("user", "second")],
                ),
            )
        )

    assert len(provider.requests) == 2
    assert ledger.reconcile_calls == 2
    failure = exc_info.value
    assert failure.__notes__ == [
        "Automatic compaction budget settlement also failed: "
        "RuntimeError: ledger reconciliation failed"
    ]
    failure_cause = exception_cause(failure)
    assert isinstance(failure_cause, BaseExceptionGroup)
    cause_failures = tuple(iter_exception_tree(failure_cause))
    assert (
        sum(
            type(item) is RuntimeError and str(item) == "ledger reconciliation failed"
            for item in cause_failures
        )
        == 1
    )
    assert (
        sum(
            isinstance(
                item,
                run_limits_module.ModelCompletionBudgetSettlementPending,
            )
            for item in cause_failures
        )
        == 1
    )
    active = asyncio.run(
        app.session_store.load_active_model_completion_stage("sess_prompt_cache_settlement_failure")
    )
    assert active is not None and active.stage.state == "completed"


def test_inherited_prompt_cache_compactor_fails_closed_before_unpriced_fallback() -> None:
    class InheritedPromptCacheCompactor(PromptCacheCompactor):
        pass

    overflow = ModelContextOverflowError(
        "context too large",
        provider="fake",
        status_code=400,
        error_code="context_length_exceeded",
    )
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.text_delta("first answer"),
                ModelStreamEvent.completed(
                    {
                        "model": "fake-model",
                        "usage": {"input_tokens": 1, "output_tokens": 0},
                    }
                ),
            ],
            [ModelStreamEvent.error(str(overflow), cause=overflow)],
            [
                ModelStreamEvent.text_delta("bounded summary"),
                ModelStreamEvent.completed(
                    {
                        "model": "fake-model",
                        "usage": {"input_tokens": 1, "output_tokens": 0},
                    }
                ),
            ],
        ]
    )
    app = CayuApp(
        budget_policy=BudgetPolicy(
            limits=(
                BudgetLimit(
                    scope="app",
                    max_estimated_cost=Decimal("0.000019"),
                    pricing=compaction_price_book(),
                    reservation=BudgetReservation(max_input_tokens=10, max_output_tokens=0),
                ),
            )
        )
    )
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        context_policy=CheckpointCompactionContextPolicy(
            compactor=InheritedPromptCacheCompactor(provider=provider),
            max_user_turns=1,
            compact_after_messages=2,
        ),
    )

    first_events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_inherited_prompt_cache_budget",
                messages=[Message.text("user", "first")],
            ),
        )
    )
    assert first_events[-1].type == EventType.SESSION_COMPLETED

    resume_events = asyncio.run(
        collect_resume_events(
            app,
            ResumeRequest(
                session_id="sess_inherited_prompt_cache_budget",
                messages=[Message.text("user", "second")],
            ),
        )
    )

    assert len(provider.requests) == 2
    assert [event.type for event in resume_events].count(EventType.BUDGET_RESERVED) == 1
    failures = [
        event for event in resume_events if event.type == EventType.BUDGET_RESERVATION_FAILED
    ]
    assert len(failures) == 1, [event.type for event in resume_events]
    failure = failures[0]
    assert Decimal(failure.payload["actual"]) == Decimal("0.000001")
    assert failure.payload["message"] == (
        "Budget cannot be verified because 1 model step(s) have missing or invalid completion usage."
    )
    assert resume_events[-1].type == EventType.SESSION_INTERRUPTED


@pytest.mark.parametrize("override_target", ["prompt_once", "oversized_atomic_unit"])
def test_automatic_compaction_rejects_model_compactor_helper_overrides_before_dispatch(
    override_target: str,
) -> None:
    class PromptOnceOverrideCompactor(ModelCompactor):
        async def _compact_prompt_once(self, *args, **kwargs) -> CompactionResult:
            del args, kwargs
            raise AssertionError("overridden prompt helper must not execute")

    class OversizedOverrideCompactor(ModelCompactor):
        async def _compact_oversized_atomic_unit(
            self,
            request: CompactionRequest,
        ) -> CompactionResult:
            del request
            raise AssertionError("overridden hierarchy helper must not execute")

    provider = FakeProvider(
        [
            ModelStreamEvent.text_delta("must not dispatch"),
            ModelStreamEvent.completed({"usage": {"input_tokens": 1}}),
        ]
    )
    compactor_type = (
        PromptOnceOverrideCompactor
        if override_target == "prompt_once"
        else OversizedOverrideCompactor
    )
    app = CayuApp(enable_logging=False)
    app.register_provider(FakeProvider([ModelStreamEvent.completed({})]), default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        context_policy=CheckpointCompactionContextPolicy(
            compactor=compactor_type(
                provider=provider,
                model="summary-model",
                max_input_chars=1000,
            ),
            max_user_turns=1,
            compact_after_messages=1,
        ),
    )
    old_message = "oversized " + "x" * 5000 if override_target == "oversized_atomic_unit" else "old"

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id=f"sess_compaction_helper_override_{override_target}",
                messages=[
                    Message.text("user", old_message),
                    Message.text("user", "current"),
                ],
                limits=RunLimits(max_total_tokens=100),
            ),
        )
    )

    assert provider.requests == []
    assert events[-1].type == EventType.SESSION_FAILED
    assert "cannot safely run opaque provider-backed compactor" in events[-1].payload["error"]


def test_automatic_compaction_rejects_provider_backed_deterministic_identity_claim() -> None:
    class MisdeclaredModelCompactor(ModelCompactor):
        def provider_budget_identity(self, session: Session) -> None:
            del session
            return None

    compactor_provider = FakeProvider(
        [
            ModelStreamEvent.text_delta("must not dispatch"),
            ModelStreamEvent.completed({"usage": {"input_tokens": 1}}),
        ]
    )
    runtime_provider = FakeProvider([ModelStreamEvent.completed({})])
    app = CayuApp(enable_logging=False)
    app.register_provider(runtime_provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        context_policy=CheckpointCompactionContextPolicy(
            compactor=MisdeclaredModelCompactor(
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
                session_id="sess_compaction_misdeclared_deterministic_identity",
                messages=[
                    Message.text("user", "old"),
                    Message.text("assistant", "old answer"),
                    Message.text("user", "current"),
                ],
                limits=RunLimits(max_total_tokens=10),
            ),
        )
    )

    assert compactor_provider.requests == []
    assert runtime_provider.requests == []
    assert events[-1].type == EventType.SESSION_FAILED
    assert "cannot declare a deterministic budget identity" in events[-1].payload["error"]


@pytest.mark.parametrize(
    ("reservation", "expected_compactor_calls", "expected_terminal_event"),
    [
        pytest.param(
            BudgetReservation(max_input_tokens=10, max_output_tokens=0),
            0,
            EventType.SESSION_FAILED,
            id="strict-reservation-fails-closed",
        ),
        pytest.param(
            None,
            0,
            EventType.SESSION_FAILED,
            id="event-backed-budget-also-fails-closed",
        ),
    ],
)
def test_automatic_compaction_opaque_multi_dispatch_budget_boundary(
    reservation: BudgetReservation | None,
    expected_compactor_calls: int,
    expected_terminal_event: EventType,
) -> None:
    class OpaqueMultiDispatchCompactor(ContextCompactor):
        def __init__(self, provider: ModelProvider) -> None:
            self.provider = provider
            self.calls = 0

        def provider_budget_identity(self, session: Session) -> tuple[str, str]:
            del session
            return "fake", "summary-model"

        async def compact(self, request: CompactionRequest) -> CompactionResult:
            self.calls += 1
            model_request = ModelRequest(
                model="summary-model",
                messages=[Message.text("user", "summarize")],
            )
            for _ in range(2):
                async for _event in self.provider.stream(model_request):
                    pass
            return CompactionResult(
                summary="summary",
                covered_message_count=len(request.messages),
                model_completed_payloads=[
                    {
                        "provider_name": "fake",
                        "requested_model": "summary-model",
                        "model": "summary-model",
                        "purpose": "context_compaction",
                        "usage_metrics": {
                            "provider_name": "fake",
                            "requested_model": "summary-model",
                            "model": "summary-model",
                            "input_tokens": 10,
                            "output_tokens": 0,
                            "total_tokens": 10,
                        },
                    },
                    {
                        "provider_name": "fake",
                        "requested_model": "summary-model",
                        "model": "summary-model",
                        "purpose": "context_compaction",
                        "usage_metrics": {
                            "provider_name": "fake",
                            "requested_model": "summary-model",
                            "model": "summary-model",
                            "input_tokens": 10,
                            "output_tokens": 0,
                            "total_tokens": 10,
                        },
                    },
                ],
            )

    compactor_provider = FakeProvider(
        [
            [ModelStreamEvent.completed({"usage": {"input_tokens": 10}})],
            [ModelStreamEvent.completed({"usage": {"input_tokens": 10}})],
        ]
    )
    compactor = OpaqueMultiDispatchCompactor(compactor_provider)
    runtime_provider = FakeProvider([ModelStreamEvent.completed({})])
    app = CayuApp(
        budget_policy=BudgetPolicy(
            limits=(
                BudgetLimit(
                    scope="app",
                    max_estimated_cost=Decimal("0.000019"),
                    pricing=compaction_price_book(),
                    reservation=reservation,
                ),
            )
        )
    )
    app.register_provider(runtime_provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        context_policy=CheckpointCompactionContextPolicy(
            compactor=compactor,
            max_user_turns=1,
            compact_after_messages=2,
        ),
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id=f"sess_compaction_opaque_budget_{reservation is not None}",
                messages=[
                    Message.text("user", "old"),
                    Message.text("assistant", "old answer"),
                    Message.text("user", "current"),
                ],
            ),
        )
    )

    assert compactor.calls == expected_compactor_calls
    assert len(compactor_provider.requests) == expected_compactor_calls * 2
    assert runtime_provider.requests == []
    assert events[-1].type == expected_terminal_event
    assert EventType.BUDGET_RESERVED not in [event.type for event in events]
    assert EventType.MODEL_COMPLETED not in [event.type for event in events]
    assert "cannot safely run opaque provider-backed compactor" in events[-1].payload["error"]


def test_deterministic_automatic_compaction_does_not_consume_strict_reservation() -> None:
    class DeterministicRecordingCompactor(RecordingCompactor):
        def provider_budget_identity(self, session: Session) -> None:
            del session
            return None

    compactor = DeterministicRecordingCompactor()
    runtime_provider = FakeProvider(
        [ModelStreamEvent.completed({"usage": {"input_tokens": 1, "output_tokens": 0}})]
    )
    app = CayuApp(
        budget_policy=BudgetPolicy(
            limits=(
                BudgetLimit(
                    scope="app",
                    max_estimated_cost=Decimal("0.00002"),
                    pricing=compaction_price_book(),
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
        context_policy=CheckpointCompactionContextPolicy(
            compactor=compactor,
            max_user_turns=1,
            compact_after_messages=2,
        ),
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_deterministic_compaction_strict_budget",
                messages=[
                    Message.text("user", "old"),
                    Message.text("assistant", "old answer"),
                    Message.text("user", "current"),
                ],
            ),
        )
    )

    assert len(compactor.requests) == 1
    assert len(runtime_provider.requests) == 1
    assert [event.type for event in events].count(EventType.BUDGET_RESERVED) == 1
    assert [event.type for event in events].count(EventType.BUDGET_RECONCILED) == 1
    assert events[-1].type == EventType.SESSION_COMPLETED


def test_prompt_cache_deterministic_fallback_does_not_consume_compaction_reservation() -> None:
    provider = FakeProvider(
        [ModelStreamEvent.completed({"usage": {"input_tokens": 1, "output_tokens": 0}})]
    )
    app = CayuApp(
        budget_policy=BudgetPolicy(
            limits=(
                BudgetLimit(
                    scope="app",
                    max_estimated_cost=Decimal("0.00002"),
                    pricing=compaction_price_book(),
                    reservation=BudgetReservation(
                        max_input_tokens=10,
                        max_output_tokens=0,
                    ),
                ),
            )
        )
    )
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        context_policy=CheckpointCompactionContextPolicy(
            compactor=PromptCacheCompactor(provider=provider),
            max_user_turns=1,
            compact_after_messages=2,
        ),
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_prompt_cache_deterministic_budget_fallback",
                messages=[
                    Message.text("user", "old"),
                    Message.text("assistant", "old answer"),
                    Message.text("user", "current"),
                ],
            ),
        )
    )

    assert len(provider.requests) == 1
    assert [event.type for event in events].count(EventType.BUDGET_RESERVED) == 1
    assert [event.type for event in events].count(EventType.BUDGET_RECONCILED) == 1
    compaction_completed = next(
        event for event in events if event.type == EventType.CONTEXT_COMPACTION_COMPLETED
    )
    assert compaction_completed.payload["chunk_mode"] == "digest_prefix"
    assert events[-1].type == EventType.SESSION_COMPLETED


def test_cayu_app_compaction_request_budget_notify_is_nonblocking_and_deduplicated():
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
    runtime_provider = FakeProvider(
        [
            ModelStreamEvent.text_delta("final answer"),
            ModelStreamEvent.completed({"usage": {"input_tokens": 1, "output_tokens": 0}}),
        ]
    )
    app = CayuApp()
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
                session_id="sess_compaction_request_budget_notify",
                messages=[
                    Message.text("user", "old"),
                    Message.text("assistant", "old answer"),
                    Message.text("user", "current"),
                ],
                budget_limits=(
                    BudgetLimit(
                        scope="run",
                        max_estimated_cost=Decimal("0.00002"),
                        pricing=compaction_price_book(),
                        action="notify",
                    ),
                ),
            ),
        )
    )

    notify_events = [event for event in events if event.type == EventType.BUDGET_LIMIT_REACHED]
    assert len(notify_events) == 1
    assert notify_events[0].payload["action"] == "notify"
    assert Decimal(notify_events[0].payload["actual"]) == Decimal("0.00002")
    assert len(compactor_provider.requests) == 1
    assert len(runtime_provider.requests) == 1
    assert events[-1].type == EventType.SESSION_COMPLETED
    usage = asyncio.run(app.get_session_usage("sess_compaction_request_budget_notify"))
    assert usage.model_steps == 2
    assert usage.usage.total_tokens == 21


def test_cayu_app_non_model_compaction_does_not_add_budget_boundary():
    compactor = RecordingCompactor()
    provider = FakeProvider(
        [ModelStreamEvent.completed({"usage": {"input_tokens": 1, "output_tokens": 0}})]
    )
    app = CayuApp(
        budget_policy=BudgetPolicy(
            limits=(
                BudgetLimit(
                    scope="app",
                    max_estimated_cost=Decimal("10"),
                    pricing=compaction_price_book(),
                ),
            )
        )
    )
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        context_policy=CheckpointCompactionContextPolicy(
            compactor=compactor,
            max_user_turns=1,
            compact_after_messages=2,
        ),
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_non_model_compaction_budget_boundaries",
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
        EventType.BUDGET_CHECKED,
        EventType.CONTEXT_COMPACTION_STARTED,
        EventType.CONTEXT_COMPACTION_COMPLETED,
        EventType.SESSION_CHECKPOINTED,
        EventType.MODEL_STARTED,
        EventType.MODEL_COMPLETED,
        EventType.BUDGET_CHECKED,
        EventType.TURN_COMPLETED,
        EventType.SESSION_COMPLETED,
    ]
    assert len(compactor.requests) == 1
    assert len(provider.requests) == 1


def test_cayu_app_model_compaction_policy_without_compaction_keeps_normal_boundaries():
    compactor_provider = FakeProvider([ModelStreamEvent.completed({})])
    runtime_provider = FakeProvider(
        [ModelStreamEvent.completed({"usage": {"input_tokens": 1, "output_tokens": 0}})]
    )
    app = CayuApp(
        budget_policy=BudgetPolicy(
            limits=(
                BudgetLimit(
                    scope="app",
                    max_estimated_cost=Decimal("10"),
                    pricing=compaction_price_book(),
                ),
            )
        )
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
                session_id="sess_no_compaction_budget_boundaries",
                messages=[Message.text("user", "current")],
            ),
        )
    )

    assert [event.type for event in events] == [
        EventType.SESSION_STARTED,
        EventType.BUDGET_CHECKED,
        EventType.MODEL_STARTED,
        EventType.MODEL_COMPLETED,
        EventType.BUDGET_CHECKED,
        EventType.TURN_COMPLETED,
        EventType.SESSION_COMPLETED,
    ]
    assert compactor_provider.requests == []
    assert len(runtime_provider.requests) == 1

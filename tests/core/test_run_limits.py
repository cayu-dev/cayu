"""Focused contracts for the concrete run-limit controller."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Callable
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

import pytest
from pydantic import ValidationError

import cayu.runtime._run_limits as run_limits_module
from cayu._validation import MAX_DURABLE_JSON_INTEGER
from cayu.core import (
    AgentSpec,
    Event,
    EventType,
    Message,
    Tool,
    ToolContext,
    ToolEffect,
    ToolResult,
    ToolSpec,
)
from cayu.core.billing import BillingIdentity
from cayu.providers import (
    ModelProvider,
    ModelRequest,
    ModelStreamEvent,
    bedrock_billing_identity,
)
from cayu.runtime import AlwaysRequireApprovalToolPolicy, CayuApp
from cayu.runtime._event_projection import public_event_sequence
from cayu.runtime._event_writer import RuntimeEventWriter
from cayu.runtime._run_limits import (
    BudgetedOperationFailed,
    BudgetedOperationRejected,
    BudgetedOperationSucceeded,
    BudgetEvaluation,
    BudgetReservationLeaseLost,
    LimitEvaluation,
    RunLimitController,
    RunLimitGate,
)
from cayu.runtime.budgets import (
    BudgetLedger,
    BudgetLimit,
    BudgetPolicy,
    BudgetReservation,
    BudgetReservationIdentityConflict,
    BudgetReservationResult,
    BudgetSettlementFallback,
    InMemoryBudgetLedger,
    SessionBudgetStore,
    has_deferred_contextual_price,
)
from cayu.runtime.costs import ModelPrice, PriceBook
from cayu.runtime.execution_units import (
    ModelAttemptIdentity,
    new_model_step_identity,
)
from cayu.runtime.sessions import (
    EventQuery,
    InMemorySessionStore,
    ResumeRequest,
    RunRequest,
    Session,
    SessionIdentity,
    SessionRunFenced,
    SessionStatus,
)
from cayu.runtime.stop_policy import RunLimits, StopLimit


def _controller(
    store: InMemorySessionStore,
    *,
    ledger: BudgetLedger | None = None,
    clock: Callable[[], datetime] | None = None,
) -> RunLimitController:
    budget_store = SessionBudgetStore(store)
    return RunLimitController(
        session_store=store,
        budget_store=budget_store,
        budget_ledger=ledger if ledger is not None else InMemoryBudgetLedger(),
        event_writer=RuntimeEventWriter(
            session_store=store,
            budget_store=budget_store,
            event_sinks=(),
        ),
        clock=clock or (lambda: datetime(2026, 1, 1, tzinfo=UTC)),
    )


def _pricing() -> PriceBook:
    return PriceBook(
        prices=(
            ModelPrice.fixed(
                provider_name="fake",
                model="fake-model",
                input_per_million=Decimal("1"),
                output_per_million=Decimal("10"),
            ),
        )
    )


def _reserved_limit(maximum: str) -> BudgetLimit:
    return BudgetLimit(
        scope="app",
        max_estimated_cost=Decimal(maximum),
        pricing=_pricing(),
        reservation=BudgetReservation(
            max_input_tokens=1_000_000,
            max_output_tokens=0,
        ),
    )


def _model_attempt_identity() -> ModelAttemptIdentity:
    return new_model_step_identity().new_attempt()


def test_deferred_bedrock_price_does_not_shadow_direct_gateway_price() -> None:
    pricing = PriceBook(
        prices=(
            ModelPrice.fixed(
                provider_name="gateway",
                model="shared-model",
                match="exact",
                input_per_million=Decimal("1"),
                output_per_million=Decimal("2"),
            ),
            ModelPrice.fixed(
                provider_name="bedrock",
                model="shared-model",
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

    assert not has_deferred_contextual_price(
        pricing,
        provider_name="gateway",
        model="shared-model",
    )
    assert has_deferred_contextual_price(
        pricing,
        provider_name="bedrock",
        model="shared-model",
    )


class _RecordingProvider(ModelProvider):
    name = "fake"

    def __init__(self) -> None:
        self.calls = 0

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        self.calls += 1
        yield ModelStreamEvent.completed(
            {
                "finish_reason": "stop",
                "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
            }
        )


class _ApprovalTool(Tool):
    spec = ToolSpec(
        name="approval_tool",
        input_schema={"type": "object", "properties": {}},
        effect=ToolEffect.EXTERNAL,
    )

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        del ctx, args
        return ToolResult(content="done")


class _ApprovalProvider(ModelProvider):
    name = "fake"

    def __init__(self) -> None:
        self.calls = 0

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        del request
        self.calls += 1
        yield ModelStreamEvent.tool_call(
            id="approval-call",
            name="approval_tool",
            arguments={},
        )
        yield ModelStreamEvent.completed({"finish_reason": "tool_calls"})


class _LoseLeaseOnSecondHeartbeat(InMemoryBudgetLedger):
    def __init__(self) -> None:
        super().__init__(reservation_ttl_seconds=1)
        self.heartbeat_calls = 0

    async def heartbeat(self, *, reservation_id: str) -> bool:
        self.heartbeat_calls += 1
        if self.heartbeat_calls == 2:
            return False
        return await super().heartbeat(reservation_id=reservation_id)


class _CancelSecondReservationLedger(InMemoryBudgetLedger):
    def __init__(self) -> None:
        super().__init__()
        self.reserve_calls = 0
        self.reservation_ids: list[str] = []

    async def reserve(
        self,
        *,
        reservation_id: str | None = None,
        limit: BudgetLimit,
        session_id: str,
        agent_name: str,
        provider_name: str,
        model: str,
        model_attempt_identity: ModelAttemptIdentity,
        environment_name: str | None = None,
        settlement_event_payload: dict[str, Any] | None = None,
        settlement_fallback: BudgetSettlementFallback | None = None,
        effective_at: datetime | None = None,
    ) -> BudgetReservationResult:
        self.reserve_calls += 1
        if self.reserve_calls == 2:
            raise asyncio.CancelledError
        result = await super().reserve(
            reservation_id=reservation_id,
            limit=limit,
            session_id=session_id,
            agent_name=agent_name,
            provider_name=provider_name,
            model=model,
            model_attempt_identity=model_attempt_identity,
            environment_name=environment_name,
            settlement_event_payload=settlement_event_payload,
            settlement_fallback=settlement_fallback,
            effective_at=effective_at,
        )
        if result.record is not None:
            self.reservation_ids.append(result.record.reservation_id)
        return result


class _FailSecondReleaseLedger(InMemoryBudgetLedger):
    def __init__(self) -> None:
        super().__init__()
        self.release_calls = 0

    async def release(
        self,
        *,
        reservation_id: str,
        reason: str,
        occurred_at: datetime | None = None,
    ):
        self.release_calls += 1
        if self.release_calls == 2:
            raise RuntimeError("simulated second release failure")
        return await super().release(
            reservation_id=reservation_id,
            reason=reason,
            occurred_at=occurred_at,
        )


class _CancelFirstHeartbeatLedger(InMemoryBudgetLedger):
    def __init__(self) -> None:
        super().__init__()
        self.reservation_ids: list[str] = []
        self.release_calls = 0

    async def reserve(
        self,
        *,
        reservation_id: str | None = None,
        limit: BudgetLimit,
        session_id: str,
        agent_name: str,
        provider_name: str,
        model: str,
        model_attempt_identity: ModelAttemptIdentity,
        environment_name: str | None = None,
        settlement_event_payload: dict[str, Any] | None = None,
        settlement_fallback: BudgetSettlementFallback | None = None,
        effective_at: datetime | None = None,
    ) -> BudgetReservationResult:
        result = await super().reserve(
            reservation_id=reservation_id,
            limit=limit,
            session_id=session_id,
            agent_name=agent_name,
            provider_name=provider_name,
            model=model,
            model_attempt_identity=model_attempt_identity,
            environment_name=environment_name,
            settlement_event_payload=settlement_event_payload,
            settlement_fallback=settlement_fallback,
            effective_at=effective_at,
        )
        if result.record is not None:
            self.reservation_ids.append(result.record.reservation_id)
        return result

    async def heartbeat(self, *, reservation_id: str) -> bool:
        raise asyncio.CancelledError

    async def release(
        self,
        *,
        reservation_id: str,
        reason: str,
        occurred_at: datetime | None = None,
    ):
        self.release_calls += 1
        return await super().release(
            reservation_id=reservation_id,
            reason=reason,
            occurred_at=occurred_at,
        )


class _FailSecondReconcileLedger(InMemoryBudgetLedger):
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
            raise RuntimeError("simulated second reconciliation failure")
        return await super().reconcile(
            reservation_id=reservation_id,
            actual_amount=actual_amount,
            reason=reason,
            occurred_at=occurred_at,
            **kwargs,
        )


async def _running_session(store: InMemorySessionStore, session_id: str) -> Session:
    session = await store.create(
        RunRequest(
            agent_name="assistant",
            session_id=session_id,
            messages=[Message.text("user", "hello")],
        ),
        identity=SessionIdentity(provider_name="fake", model="fake-model"),
    )
    return await store.update_status(session.id, SessionStatus.RUNNING)


@pytest.mark.parametrize(
    "field_name",
    [
        "max_input_tokens",
        "max_output_tokens",
        "max_total_tokens",
        "max_tool_calls",
        "max_elapsed_seconds",
    ],
)
def test_run_limits_reject_nonportable_integer_values(field_name: str) -> None:
    with pytest.raises(ValidationError):
        RunLimits(**{field_name: MAX_DURABLE_JSON_INTEGER + 1})


def test_app_revalidates_run_limits_before_provider_or_durable_mutation() -> None:
    store = InMemorySessionStore()
    provider = _ApprovalProvider()
    app = CayuApp(session_store=store, enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[_ApprovalTool()],
        tool_policy=AlwaysRequireApprovalToolPolicy(),
    )
    invalid_limits = RunLimits.model_construct(max_total_tokens=MAX_DURABLE_JSON_INTEGER + 1)
    request = RunRequest(
        agent_name="assistant",
        session_id="sess_nonportable_run_limit",
        messages=[Message.text("user", "call the tool")],
        limits=invalid_limits,
    )

    async def scenario():
        with pytest.raises(ValidationError):
            async for _ in app.run(request):
                pass
        session = await store.load("sess_nonportable_run_limit")
        records = await store.query_events(
            EventQuery(session_id="sess_nonportable_run_limit", limit=100)
        )
        return session, records

    session, records = asyncio.run(scenario())

    assert provider.calls == 0
    assert session is None
    assert records == []


def test_session_run_limit_publishes_usage_beyond_int64_without_provider_call() -> None:
    store = InMemorySessionStore()
    provider = _RecordingProvider()
    app = CayuApp(session_store=store, enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))
    maximum = MAX_DURABLE_JSON_INTEGER
    expected = maximum * 2

    async def scenario():
        await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="sess_aggregate_run_limit",
                messages=[Message.text("user", "initial")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        await store.append_events(
            "sess_aggregate_run_limit",
            [
                Event(
                    type=EventType.MODEL_COMPLETED,
                    session_id="sess_aggregate_run_limit",
                    payload={
                        "usage_metrics": {
                            "input_tokens": maximum,
                            "output_tokens": maximum,
                            "total_tokens": maximum,
                        }
                    },
                )
                for _ in range(2)
            ],
        )
        await store.update_status("sess_aggregate_run_limit", SessionStatus.COMPLETED)
        events = [
            event
            async for event in app.resume(
                ResumeRequest(
                    session_id="sess_aggregate_run_limit",
                    messages=[Message.text("user", "do not dispatch")],
                    limits=RunLimits(
                        max_total_tokens=maximum,
                        scope="session",
                    ),
                )
            )
        ]
        records = await store.query_events(
            EventQuery(session_id="sess_aggregate_run_limit", limit=100)
        )
        return events, records

    events, records = asyncio.run(scenario())

    assert provider.calls == 0
    limit_event = next(event for event in events if event.type == EventType.SESSION_LIMIT_REACHED)
    assert limit_event.payload["actual"] == str(expected)
    assert limit_event.payload["usage_summary"]["usage"]["total_tokens"] == str(expected)
    turn = next(event for event in events if event.type == EventType.TURN_COMPLETED)
    assert turn.payload["token_usage"]["total_tokens"] == 0
    limit_sequence = public_event_sequence(limit_event.id)
    assert limit_sequence is not None
    assert any(
        record.sequence == limit_sequence and record.event.payload == limit_event.payload
        for record in records
    )


def test_controller_returns_typed_limit_decision_without_finalizing_session():
    store = InMemorySessionStore()
    controller = _controller(store)

    async def scenario() -> tuple[LimitEvaluation, Session]:
        session = await _running_session(store, "sess_controller_limit")
        await store.append_events(
            session.id,
            [
                Event(
                    type=EventType.MODEL_COMPLETED,
                    session_id=session.id,
                    agent_name="assistant",
                    payload={
                        "provider_name": "fake",
                        "model": "fake-model",
                        "usage": {
                            "input_tokens": 7,
                            "output_tokens": 4,
                            "total_tokens": 11,
                        },
                    },
                )
            ],
        )
        result = await controller.evaluate_request_limits(
            session=session,
            agent_name="assistant",
            environment_name=None,
            limits=RunLimits(max_total_tokens=10),
            budget_limits=(),
            run_started_at=time.monotonic(),
        )
        loaded = await store.load(session.id)
        assert loaded is not None
        return result, loaded

    result, session = asyncio.run(scenario())

    assert type(result) is LimitEvaluation
    assert result.decision is not None
    assert result.decision.limit == StopLimit.TOTAL_TOKENS
    assert result.usage_summary.usage.total_tokens == 11
    assert result.events == ()
    assert session.status == SessionStatus.RUNNING


def test_controller_session_elapsed_paths_share_injected_clock():
    store = InMemorySessionStore()
    injected_now = datetime(2030, 1, 1, tzinfo=UTC)
    controller = _controller(store, clock=lambda: injected_now)

    async def scenario():
        await _running_session(store, "sess_controller_elapsed_clock")
        store._sessions["sess_controller_elapsed_clock"].created_at = injected_now - timedelta(
            seconds=2
        )
        session = await store.load("sess_controller_elapsed_clock")
        assert session is not None
        limits = RunLimits(max_elapsed_seconds=1, scope="session")

        request_evaluation = await controller.evaluate_request_limits(
            session=session,
            agent_name="assistant",
            environment_name=None,
            limits=limits,
            budget_limits=(),
            run_started_at=time.monotonic(),
        )
        operation_decision = await controller.evaluate_operation_run_limit(
            session=session,
            limits=limits,
            operation_events=[],
            operation_started_at=time.monotonic(),
        )
        return request_evaluation.decision, operation_decision

    request_decision, operation_decision = asyncio.run(scenario())

    assert request_decision is not None
    assert request_decision.limit is StopLimit.ELAPSED_SECONDS
    assert operation_decision is not None
    assert operation_decision.limit is StopLimit.ELAPSED_SECONDS


def test_cayu_app_session_elapsed_limit_uses_injected_clock_before_provider_dispatch():
    store = InMemorySessionStore()
    provider = _RecordingProvider()
    injected_now = datetime(2030, 1, 1, tzinfo=UTC)
    app = CayuApp(
        session_store=store,
        enable_logging=False,
        clock=lambda: injected_now,
    )
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    async def scenario():
        session = await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="sess_app_elapsed_clock",
                messages=[Message.text("user", "initial")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        store._sessions[session.id].created_at = injected_now - timedelta(seconds=2)
        await store.update_status(session.id, SessionStatus.COMPLETED)

        events = [
            event
            async for event in app.resume(
                ResumeRequest(
                    session_id=session.id,
                    messages=[Message.text("user", "do not dispatch")],
                    limits=RunLimits(max_elapsed_seconds=1, scope="session"),
                )
            )
        ]
        loaded = await store.load(session.id)
        assert loaded is not None
        return events, loaded

    events, session = asyncio.run(scenario())

    assert provider.calls == 0
    assert [event.type for event in events] == [
        EventType.INTERACTION_STARTED,
        EventType.SESSION_RESUMED,
        EventType.SESSION_LIMIT_REACHED,
        EventType.INTERACTION_INTERRUPTED,
        EventType.TURN_COMPLETED,
        EventType.SESSION_INTERRUPTED,
    ]
    assert events[2].payload["limit"] == "elapsed_seconds"
    assert session.status is SessionStatus.INTERRUPTED


def test_controller_run_elapsed_paths_keep_monotonic_invocation_time(monkeypatch):
    monotonic_now = {"value": 100.0}
    monkeypatch.setattr(run_limits_module.time, "monotonic", lambda: monotonic_now["value"])
    store = InMemorySessionStore()
    controller = _controller(
        store,
        clock=lambda: datetime(2030, 1, 1, tzinfo=UTC),
    )

    async def evaluate(session_id: str):
        session = await _running_session(store, session_id)
        limits = RunLimits(max_elapsed_seconds=1, scope="run")
        request_evaluation = await controller.evaluate_request_limits(
            session=session,
            agent_name="assistant",
            environment_name=None,
            limits=limits,
            budget_limits=(),
            run_started_at=100.0,
        )
        operation_decision = await controller.evaluate_operation_run_limit(
            session=session,
            limits=limits,
            operation_events=[],
            operation_started_at=100.0,
        )
        return request_evaluation.decision, operation_decision

    before_request, before_operation = asyncio.run(evaluate("sess_controller_run_elapsed_before"))
    assert before_request is None
    assert before_operation is None

    monotonic_now["value"] = 101.0
    after_request, after_operation = asyncio.run(evaluate("sess_controller_run_elapsed_after"))
    assert after_request is not None
    assert after_request.limit is StopLimit.ELAPSED_SECONDS
    assert after_operation is not None
    assert after_operation.limit is StopLimit.ELAPSED_SECONDS


def test_controller_fails_closed_on_malformed_normalized_usage_without_crashing():
    store = InMemorySessionStore()
    controller = _controller(store)

    async def scenario() -> tuple[LimitEvaluation, Session]:
        session = await _running_session(store, "sess_malformed_usage_budget")
        await store.append_event(
            session.id,
            Event(
                type=EventType.MODEL_COMPLETED,
                session_id=session.id,
                agent_name="assistant",
                payload={
                    "usage_metrics": {
                        "provider_name": " fake ",
                        "model": "fake-model",
                        "input_tokens": 7,
                        "output_tokens": 3,
                        "total_tokens": 10,
                        "reasoning_output_tokens": "not-an-integer",
                    }
                },
            ),
        )
        result = await controller.evaluate_request_limits(
            session=session,
            agent_name="assistant",
            environment_name=None,
            limits=RunLimits(),
            budget_limits=(
                BudgetLimit(
                    scope="session",
                    max_estimated_cost=Decimal("1"),
                    pricing=_pricing(),
                ),
            ),
            run_started_at=time.monotonic(),
        )
        loaded = await store.load(session.id)
        assert loaded is not None
        return result, loaded

    result, session = asyncio.run(scenario())

    assert result.decision is not None
    assert result.decision.limit is StopLimit.ESTIMATED_COST
    assert "cannot be verified" in result.decision.message
    assert result.cost_summary is not None
    assert result.cost_summary.unpriced_model_steps == 1
    assert result.cost_summary.line_items[0].missing_pricing_reason == (
        "model.completed event has no valid normalized usage metrics"
    )
    assert session.status is SessionStatus.RUNNING


def test_operation_session_limit_deduplicates_persisted_operation_events():
    store = InMemorySessionStore()
    controller = _controller(store)

    async def scenario():
        session = await _running_session(store, "sess_operation_limit_deduplication")
        completion = Event(
            type=EventType.MODEL_COMPLETED,
            session_id=session.id,
            agent_name="assistant",
            payload={
                "provider_name": "fake",
                "model": "fake-model",
                "usage": {
                    "input_tokens": 8,
                    "output_tokens": 2,
                    "total_tokens": 10,
                },
            },
        )
        await store.append_event(session.id, completion)

        below_limit = await controller.evaluate_operation_run_limit(
            session=session,
            limits=RunLimits(max_total_tokens=15, scope="session"),
            operation_events=[completion],
            operation_started_at=time.monotonic(),
        )
        at_limit = await controller.evaluate_operation_run_limit(
            session=session,
            limits=RunLimits(max_total_tokens=10, scope="session"),
            operation_events=[completion],
            operation_started_at=time.monotonic(),
        )
        return below_limit, at_limit

    below_limit, at_limit = asyncio.run(scenario())

    assert below_limit is None
    assert at_limit is not None
    assert at_limit.limit == StopLimit.TOTAL_TOKENS
    assert at_limit.actual == 10


def test_run_limit_gate_reuses_incremental_usage_without_finalizing_session():
    store = InMemorySessionStore()
    controller = _controller(store)

    async def scenario() -> tuple[LimitEvaluation, LimitEvaluation, Session]:
        session = await _running_session(store, "sess_gate_incremental")
        gate = RunLimitGate(
            controller,
            session=session,
            agent_name="assistant",
            environment_name=None,
            limits=RunLimits(max_total_tokens=10),
            budget_limits=(),
            run_started_at=time.monotonic(),
            run_baseline=None,
            budget_baseline_events=[],
            budget_notify_events=[],
        )
        await store.append_event(
            session.id,
            Event(
                type=EventType.MODEL_COMPLETED,
                session_id=session.id,
                agent_name="assistant",
                payload={
                    "provider_name": "fake",
                    "model": "fake-model",
                    "usage": {
                        "input_tokens": 3,
                        "output_tokens": 2,
                        "total_tokens": 5,
                    },
                },
            ),
        )
        first = await gate.evaluate_limits()
        await store.append_event(
            session.id,
            Event(
                type=EventType.MODEL_COMPLETED,
                session_id=session.id,
                agent_name="assistant",
                payload={
                    "provider_name": "fake",
                    "model": "fake-model",
                    "usage": {
                        "input_tokens": 4,
                        "output_tokens": 2,
                        "total_tokens": 6,
                    },
                },
            ),
        )
        second = await gate.evaluate_limits()
        loaded = await store.load(session.id)
        assert loaded is not None
        return first, second, loaded

    first, second, session = asyncio.run(scenario())

    assert first.decision is None
    assert first.usage_summary.usage.total_tokens == 5
    assert second.decision is not None
    assert second.decision.limit == StopLimit.TOTAL_TOKENS
    assert second.usage_summary.usage.total_tokens == 11
    assert session.status == SessionStatus.RUNNING


def test_controller_fails_closed_for_unpriced_policy_without_finalizing_session():
    store = InMemorySessionStore()
    policy = BudgetPolicy(
        limits=(
            BudgetLimit(
                scope="app",
                max_estimated_cost=Decimal("1"),
                pricing=PriceBook(
                    prices=(
                        ModelPrice.fixed(
                            provider_name="other",
                            model="other-model",
                            input_per_million=Decimal("1"),
                            output_per_million=Decimal("1"),
                        ),
                    )
                ),
            ),
        )
    )
    controller = _controller(store)

    async def scenario() -> tuple[BudgetEvaluation, Session]:
        session = await _running_session(store, "sess_controller_unpriced")
        result = await controller.evaluate_policy_budgets(
            session=session,
            agent_name="assistant",
            environment_name=None,
            budget_policy=policy,
        )
        loaded = await store.load(session.id)
        assert loaded is not None
        return result, loaded

    result, session = asyncio.run(scenario())

    assert type(result) is BudgetEvaluation
    assert result.check is not None
    assert result.check.limit_reached is True
    assert "cannot be verified" in result.check.message
    assert "pricing" in result.check.message
    assert [event.type for event in result.events] == [EventType.BUDGET_CHECKED]
    assert session.status == SessionStatus.RUNNING


def test_app_policy_replacement_is_used_by_the_next_run():
    store = InMemorySessionStore()
    provider = _RecordingProvider()
    app = CayuApp(session_store=store, enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))
    app.budget_policy = BudgetPolicy(
        limits=(
            BudgetLimit(
                scope="app",
                max_estimated_cost=Decimal("1"),
                pricing=PriceBook(
                    prices=(
                        ModelPrice.fixed(
                            provider_name="other",
                            model="other-model",
                            input_per_million=Decimal("1"),
                            output_per_million=Decimal("1"),
                        ),
                    )
                ),
            ),
        )
    )

    async def scenario() -> list[Event]:
        return [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="sess_replaced_budget_policy",
                    messages=[Message.text("user", "hello")],
                )
            )
        ]

    events = asyncio.run(scenario())

    assert provider.calls == 0
    assert [event.type for event in events] == [
        EventType.INTERACTION_STARTED,
        EventType.SESSION_STARTED,
        EventType.BUDGET_CHECKED,
        EventType.BUDGET_LIMIT_REACHED,
        EventType.SESSION_LIMIT_REACHED,
        EventType.INTERACTION_INTERRUPTED,
        EventType.TURN_COMPLETED,
        EventType.SESSION_INTERRUPTED,
    ]


def test_app_releases_reservations_when_initial_renewal_is_cancelled():
    store = InMemorySessionStore()
    ledger = _CancelFirstHeartbeatLedger()
    provider = _RecordingProvider()
    app = CayuApp(
        session_store=store,
        budget_ledger=ledger,
        budget_policy=BudgetPolicy(limits=(_reserved_limit("3"),)),
        enable_logging=False,
    )
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    async def scenario():
        with pytest.raises(asyncio.CancelledError):
            async for _ in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="sess_initial_renewal_cancelled",
                    messages=[Message.text("user", "hello")],
                )
            ):
                pass
        assert len(ledger.reservation_ids) == 1
        renewed = await InMemoryBudgetLedger.heartbeat(
            ledger,
            reservation_id=ledger.reservation_ids[0],
        )
        records = await store.query_events(
            EventQuery(session_id="sess_initial_renewal_cancelled", limit=100)
        )
        return renewed, [item.event.type for item in records]

    renewed, event_types = asyncio.run(scenario())

    assert provider.calls == 0
    assert ledger.release_calls == 1
    assert renewed is False
    assert EventType.BUDGET_RESERVED in event_types
    assert EventType.BUDGET_RESERVATION_RELEASED in event_types


def test_controller_releases_prior_operation_reservations_when_later_limit_rejects():
    store = InMemorySessionStore()
    ledger = InMemoryBudgetLedger()
    controller = _controller(store, ledger=ledger)

    async def scenario():
        await _running_session(store, "sess_operation_rejection")
        setup = await controller.reserve_operation_budgets(
            budget_limits=(_reserved_limit("2"), _reserved_limit("0.5")),
            session_id="sess_operation_rejection",
            agent_name="assistant",
            provider_name="fake",
            model="fake-model",
            model_attempt_identity=_model_attempt_identity(),
            rejection_release_reason="later reservation rejected",
            accepted_record_error="accepted reservation missing record",
        )
        first_record = setup.results[0].record
        assert first_record is not None
        renewed = await ledger.heartbeat(reservation_id=first_record.reservation_id)
        return setup, renewed

    setup, renewed = asyncio.run(scenario())

    assert [result.accepted for result in setup.results] == [True, False]
    assert setup.failure is setup.results[-1]
    assert setup.error is None
    assert setup.reservations == ()
    assert len(setup.releases) == 1
    assert setup.releases[0].status == "released"
    assert setup.releases[0].reason == "later reservation rejected"
    assert renewed is False


def test_controller_rejects_duplicate_operation_reservation_ids():
    class DuplicateReservationIdLedger(InMemoryBudgetLedger):
        first_reservation_id: str | None = None

        async def reserve(self, **kwargs):
            result = await super().reserve(**kwargs)
            assert result.record is not None
            if self.first_reservation_id is None:
                self.first_reservation_id = result.record.reservation_id
                return result
            return result.model_copy(
                update={
                    "record": result.record.model_copy(
                        update={"reservation_id": self.first_reservation_id}
                    )
                }
            )

    store = InMemorySessionStore()
    ledger = DuplicateReservationIdLedger()
    controller = _controller(store, ledger=ledger)

    async def scenario():
        await _running_session(store, "sess_duplicate_operation_reservation_identity")
        setup = await controller.reserve_operation_budgets(
            budget_limits=(_reserved_limit("3"), _reserved_limit("3")),
            session_id="sess_duplicate_operation_reservation_identity",
            agent_name="assistant",
            provider_name="fake",
            model="fake-model",
            model_attempt_identity=_model_attempt_identity(),
            rejection_release_reason="reservation rejected",
            accepted_record_error="accepted reservation missing record",
        )
        reservations = list(setup.reservations)
        releases = [
            reconciliation
            async for reconciliation in controller.release_operation_reservations(
                reservations,
                reason="duplicate reservation identity",
            )
        ]
        return setup, reservations, releases

    setup, reservations, releases = asyncio.run(scenario())

    assert len(setup.results) == 1
    assert len(setup.reservations) == 1
    assert isinstance(setup.error, RuntimeError)
    assert str(setup.error) == "Budget ledger reused a reservation identity."
    assert reservations == []
    assert len(releases) == 1
    assert releases[0].status == "released"


def test_controller_returns_uniquely_accepted_reservation_when_identity_claim_fails():
    class RecordingLedger(InMemoryBudgetLedger):
        reservation_id: str | None = None

        async def reserve(self, **kwargs):
            result = await super().reserve(**kwargs)
            assert result.record is not None
            self.reservation_id = result.record.reservation_id
            return result

    class FailingIdentityGuard:
        async def claim(self, *args, **kwargs):
            del args, kwargs
            raise ConnectionError("reservation identity claim acknowledgement lost")

    store = InMemorySessionStore()
    ledger = RecordingLedger()
    controller = _controller(store, ledger=ledger)

    async def scenario():
        await _running_session(store, "sess_operation_identity_claim_failure")
        setup = await controller.reserve_operation_budgets(
            budget_limits=(_reserved_limit("3"),),
            session_id="sess_operation_identity_claim_failure",
            agent_name="assistant",
            provider_name="fake",
            model="fake-model",
            model_attempt_identity=_model_attempt_identity(),
            rejection_release_reason="reservation rejected",
            accepted_record_error="accepted reservation missing record",
            reservation_identity_guard=FailingIdentityGuard(),
        )
        assert ledger.reservation_id is not None
        active = await ledger.heartbeat(reservation_id=ledger.reservation_id)
        reservations = list(setup.reservations)
        releases = [
            reconciliation
            async for reconciliation in controller.release_operation_reservations(
                reservations,
                reason="identity claim failed before provider dispatch",
            )
        ]
        still_active = await ledger.heartbeat(reservation_id=ledger.reservation_id)
        return setup, active, still_active, reservations, releases

    setup, active, still_active, reservations, releases = asyncio.run(scenario())

    assert isinstance(setup.error, ConnectionError)
    assert setup.results == ()
    assert setup.events == ()
    assert len(setup.reservations) == 1
    assert active is True
    assert still_active is False
    assert reservations == []
    assert len(releases) == 1
    assert releases[0].status == "released"


def test_controller_returns_accepted_reservation_when_event_factory_fails():
    store = InMemorySessionStore()
    ledger = InMemoryBudgetLedger()
    controller = _controller(store, ledger=ledger)

    def failing_event_factory(result: BudgetReservationResult) -> Event:
        assert result.accepted
        raise RuntimeError("reservation event construction failed")

    async def scenario():
        await _running_session(store, "sess_operation_event_factory_failure")
        setup = await controller.reserve_operation_budgets(
            budget_limits=(_reserved_limit("3"),),
            session_id="sess_operation_event_factory_failure",
            agent_name="assistant",
            provider_name="fake",
            model="fake-model",
            model_attempt_identity=_model_attempt_identity(),
            rejection_release_reason="reservation rejected",
            accepted_record_error="accepted reservation missing record",
            reservation_event_factory=failing_event_factory,
        )
        active = list(setup.reservations)
        releases = [
            reconciliation
            async for reconciliation in controller.release_operation_reservations(
                active,
                reason="event construction failed before publication",
            )
        ]
        return setup, active, releases

    setup, active, releases = asyncio.run(scenario())

    assert isinstance(setup.error, RuntimeError)
    assert str(setup.error) == "reservation event construction failed"
    assert setup.results == ()
    assert len(setup.reservations) == 1
    assert active == []
    assert len(releases) == 1
    assert releases[0].status == "released"


@pytest.mark.parametrize(
    "conflict_type",
    [BudgetReservationIdentityConflict, SessionRunFenced],
)
def test_controller_does_not_return_reservation_after_proven_identity_conflict(
    conflict_type,
):
    class ConflictingIdentityGuard:
        async def claim(self, *args, **kwargs):
            del args, kwargs
            raise conflict_type("reservation ownership rejected")

    store = InMemorySessionStore()
    ledger = InMemoryBudgetLedger()
    controller = _controller(store, ledger=ledger)

    async def scenario():
        await _running_session(store, "sess_operation_proven_identity_conflict")
        return await controller.reserve_operation_budgets(
            budget_limits=(_reserved_limit("3"),),
            session_id="sess_operation_proven_identity_conflict",
            agent_name="assistant",
            provider_name="fake",
            model="fake-model",
            model_attempt_identity=_model_attempt_identity(),
            rejection_release_reason="reservation rejected",
            accepted_record_error="accepted reservation missing record",
            reservation_identity_guard=ConflictingIdentityGuard(),
        )

    setup = asyncio.run(scenario())

    assert isinstance(setup.error, conflict_type)
    assert setup.reservations == ()
    assert setup.results == ()
    assert setup.events == ()


def test_controller_releases_model_reservation_when_event_attestation_fails(monkeypatch):
    store = InMemorySessionStore()
    ledger = InMemoryBudgetLedger()
    controller = _controller(store, ledger=ledger)
    original = run_limits_module._event_with_budget_authority

    def failing_attestation(event: Event, **kwargs) -> Event:
        if event.type == EventType.BUDGET_RESERVED:
            raise RuntimeError("reservation event attestation failed")
        return original(event, **kwargs)

    monkeypatch.setattr(
        run_limits_module,
        "_event_with_budget_authority",
        failing_attestation,
    )

    async def scenario():
        session = await _running_session(store, "sess_model_event_attestation_failure")
        setup = await controller.reserve_for_model_step(
            session=session,
            agent_name="assistant",
            provider_name="fake",
            environment_name=None,
            model_attempt_identity=_model_attempt_identity(),
            budget_policy=BudgetPolicy(limits=(_reserved_limit("3"),)),
        )
        record = next(iter(ledger._records.values()))
        active = await ledger.heartbeat(reservation_id=record.reservation_id)
        events = await store.query_events(EventQuery(session_id=session.id, limit=100))
        return setup, active, [item.event.type for item in events]

    setup, active, event_types = asyncio.run(scenario())

    assert isinstance(setup.error, RuntimeError)
    assert str(setup.error) == "reservation event attestation failed"
    assert setup.reservations == ()
    assert active is False
    assert EventType.BUDGET_RESERVED not in event_types
    assert EventType.BUDGET_RESERVATION_RELEASED in event_types


@pytest.mark.parametrize("failure_type", [ConnectionError, asyncio.CancelledError])
def test_model_reservation_claim_failure_releases_before_dispatch(failure_type):
    class RecordingLedger(InMemoryBudgetLedger):
        reservation_id: str | None = None

        async def reserve(self, **kwargs):
            result = await super().reserve(**kwargs)
            assert result.record is not None
            self.reservation_id = result.record.reservation_id
            return result

    class FailingIdentityGuard:
        async def claim(self, *args, **kwargs):
            del args, kwargs
            raise failure_type("reservation identity claim failed")

    store = InMemorySessionStore()
    ledger = RecordingLedger()
    controller = _controller(store, ledger=ledger)

    async def scenario():
        session = await _running_session(store, "sess_model_identity_claim_failure")
        setup = None
        cancellation = None
        try:
            setup = await controller.reserve_for_model_step(
                session=session,
                agent_name="assistant",
                provider_name="fake",
                environment_name=None,
                model_attempt_identity=_model_attempt_identity(),
                budget_policy=BudgetPolicy(limits=(_reserved_limit("3"),)),
                reservation_identity_guard=FailingIdentityGuard(),
            )
        except asyncio.CancelledError as exc:
            cancellation = exc
        assert ledger.reservation_id is not None
        active = await ledger.heartbeat(reservation_id=ledger.reservation_id)
        records = await store.query_events(
            EventQuery(session_id=session.id, limit=100),
        )
        return setup, cancellation, active, [record.event.type for record in records]

    setup, cancellation, active, event_types = asyncio.run(scenario())

    assert active is False
    assert EventType.BUDGET_RESERVED not in event_types
    assert EventType.BUDGET_RESERVATION_RELEASED in event_types
    if failure_type is asyncio.CancelledError:
        assert setup is None
        assert isinstance(cancellation, asyncio.CancelledError)
    else:
        assert cancellation is None
        assert setup is not None
        assert isinstance(setup.error, ConnectionError)
        assert setup.reservations == ()


def test_controller_returns_partial_reservations_when_setup_is_cancelled():
    store = InMemorySessionStore()
    ledger = _CancelSecondReservationLedger()
    controller = _controller(store, ledger=ledger)

    async def scenario():
        await _running_session(store, "sess_operation_setup_cancelled")
        setup = await controller.reserve_operation_budgets(
            budget_limits=(_reserved_limit("3"), _reserved_limit("3")),
            session_id="sess_operation_setup_cancelled",
            agent_name="assistant",
            provider_name="fake",
            model="fake-model",
            model_attempt_identity=_model_attempt_identity(),
            rejection_release_reason="reservation rejected",
            accepted_record_error="accepted reservation missing record",
        )
        active = list(setup.reservations)
        releases = [
            reconciliation
            async for reconciliation in controller.release_operation_reservations(
                active,
                reason="cancelled setup cleanup",
            )
        ]
        assert active == []
        return setup, releases

    setup, releases = asyncio.run(scenario())

    assert len(setup.results) == 1
    assert len(setup.reservations) == 1
    assert isinstance(setup.error, asyncio.CancelledError)
    assert len(releases) == 1
    assert releases[0].status == "released"


def test_controller_releases_model_reservations_before_propagating_cancellation():
    store = InMemorySessionStore()
    ledger = _CancelSecondReservationLedger()
    controller = _controller(store, ledger=ledger)

    async def scenario():
        session = await _running_session(store, "sess_model_setup_cancelled")
        with pytest.raises(asyncio.CancelledError):
            await controller.reserve_for_model_step(
                session=session,
                agent_name="assistant",
                provider_name="fake",
                environment_name=None,
                budget_policy=BudgetPolicy(limits=(_reserved_limit("3"),)),
                request_budget_limits=(_reserved_limit("3"),),
                model_attempt_identity=_model_attempt_identity(),
            )
        assert len(ledger.reservation_ids) == 1
        renewed = await ledger.heartbeat(reservation_id=ledger.reservation_ids[0])
        records = await store.query_events(EventQuery(session_id=session.id, limit=100))
        return renewed, [record.event.type for record in records]

    renewed, event_types = asyncio.run(scenario())

    assert renewed is False
    assert event_types == [
        EventType.BUDGET_RESERVED,
        EventType.BUDGET_RESERVATION_RELEASED,
    ]


def test_controller_preserves_partial_release_progress_after_later_failure():
    store = InMemorySessionStore()
    ledger = _FailSecondReleaseLedger()
    controller = _controller(store, ledger=ledger)

    async def scenario():
        await _running_session(store, "sess_partial_operation_release")
        setup = await controller.reserve_operation_budgets(
            budget_limits=(
                _reserved_limit("4"),
                _reserved_limit("4"),
                _reserved_limit("4"),
                _reserved_limit("0.5"),
            ),
            session_id="sess_partial_operation_release",
            agent_name="assistant",
            provider_name="fake",
            model="fake-model",
            model_attempt_identity=_model_attempt_identity(),
            rejection_release_reason="later reservation rejected",
            accepted_record_error="accepted reservation missing record",
        )
        remaining = list(setup.reservations)
        cleanup = [
            reconciliation
            async for reconciliation in controller.release_operation_reservations(
                remaining,
                reason="fallback cleanup",
            )
        ]
        return setup, remaining, cleanup

    setup, remaining, cleanup = asyncio.run(scenario())

    assert setup.failure is setup.results[-1]
    assert isinstance(setup.error, RuntimeError)
    assert str(setup.error) == "simulated second release failure"
    assert len(setup.releases) == 1
    assert setup.releases[0].reason == "later reservation rejected"
    assert len(setup.reservations) == 2
    assert remaining == []
    assert len(cleanup) == 2
    assert all(result.reason == "fallback cleanup" for result in cleanup)


def test_controller_preserves_partial_model_release_progress_after_later_failure():
    store = InMemorySessionStore()
    ledger = _FailSecondReleaseLedger()
    controller = _controller(store, ledger=ledger)

    async def scenario():
        session = await _running_session(store, "sess_partial_model_release")
        setup = await controller.reserve_for_model_step(
            session=session,
            agent_name="assistant",
            provider_name="fake",
            environment_name=None,
            budget_policy=BudgetPolicy(limits=(_reserved_limit("4"),)),
            request_budget_limits=(
                _reserved_limit("4"),
                _reserved_limit("4"),
                _reserved_limit("0.5"),
            ),
            model_attempt_identity=_model_attempt_identity(),
        )
        records = await store.query_events(EventQuery(session_id=session.id, limit=100))
        events = [record.event for record in records]
        reservation_ids = [
            event.payload["reservation_id"]
            for event in events
            if event.type == EventType.BUDGET_RESERVED
        ]
        active_reservations = [
            await ledger.heartbeat(reservation_id=reservation_id)
            for reservation_id in reservation_ids
        ]
        return setup, events, active_reservations

    setup, events, active_reservations = asyncio.run(scenario())

    assert setup.failure is not None
    assert setup.failure.accepted is False
    assert isinstance(setup.error, RuntimeError)
    assert str(setup.error) == "simulated second release failure"
    assert setup.reservations == ()
    assert ledger.release_calls == 4
    assert active_reservations == [False, False, False]
    assert [event.type for event in events] == [
        EventType.BUDGET_RESERVED,
        EventType.BUDGET_RESERVED,
        EventType.BUDGET_RESERVED,
        EventType.BUDGET_RESERVATION_FAILED,
        EventType.BUDGET_RESERVATION_RELEASED,
        EventType.BUDGET_RESERVATION_RELEASED,
        EventType.BUDGET_RESERVATION_RELEASED,
    ]
    assert {
        event.payload["reason"]
        for event in events
        if event.type == EventType.BUDGET_RESERVATION_RELEASED
    } == {"reservation failed"}


def test_controller_reconciles_operation_reservations_with_priced_actuals():
    store = InMemorySessionStore()
    ledger = InMemoryBudgetLedger()
    controller = _controller(store, ledger=ledger)
    completion_identity = BillingIdentity(
        provider_name="fake",
        resource_id="fake-model",
        completion_evidence={"effective_tier": "standard"},
    )

    async def scenario():
        await _running_session(store, "sess_operation_reconcile")
        setup = await controller.reserve_operation_budgets(
            budget_limits=(_reserved_limit("2"),),
            session_id="sess_operation_reconcile",
            agent_name="assistant",
            provider_name="fake",
            model="fake-model",
            model_attempt_identity=_model_attempt_identity(),
            rejection_release_reason="reservation rejected",
            accepted_record_error="accepted reservation missing record",
        )
        reservations = list(setup.reservations)
        reconciliations = [
            reconciliation
            async for reconciliation in controller.reconcile_operation_reservations(
                reservations,
                model_completed_events=[
                    Event(
                        type=EventType.MODEL_COMPLETED,
                        session_id="sess_operation_reconcile",
                        agent_name="assistant",
                        payload={
                            "provider_name": "fake",
                            "model": "fake-model",
                            "billing_identity": completion_identity.model_dump(mode="json"),
                            "usage": {
                                "input_tokens": 250_000,
                                "output_tokens": 0,
                                "total_tokens": 250_000,
                            },
                        },
                    )
                ],
                completed_reason="operation model completed",
                missing_usage_reason="operation usage missing",
            )
        ]
        assert reservations == []
        return setup, reconciliations

    setup, reconciliations = asyncio.run(scenario())

    assert setup.failure is None
    assert setup.error is None
    assert len(reconciliations) == 1
    assert reconciliations[0].status == "reconciled"
    assert reconciliations[0].actual_amount == Decimal("0.25")
    assert reconciliations[0].reason == "operation model completed"
    assert reconciliations[0].pricing_provider_name == "fake"
    assert reconciliations[0].pricing_model == "fake-model"
    assert reconciliations[0].billing_identity == completion_identity
    assert next(iter(ledger._records.values())).billing_identity is None


def test_controller_arbitrates_operation_heartbeat_lease_loss():
    store = InMemorySessionStore()
    ledger = _LoseLeaseOnSecondHeartbeat()
    controller = _controller(store, ledger=ledger)

    async def scenario() -> None:
        await _running_session(store, "sess_operation_heartbeat")
        setup = await controller.reserve_operation_budgets(
            budget_limits=(_reserved_limit("2"),),
            session_id="sess_operation_heartbeat",
            agent_name="assistant",
            provider_name="fake",
            model="fake-model",
            model_attempt_identity=_model_attempt_identity(),
            rejection_release_reason="reservation rejected",
            accepted_record_error="accepted reservation missing record",
        )

        async def operation() -> str:
            await asyncio.sleep(2)
            return "completed"

        with pytest.raises(BudgetReservationLeaseLost):
            await controller.run_operation_with_reservation_heartbeat(
                operation,
                reservations=list(setup.reservations),
                authoritative_failure_types=(),
                lease_lost_before_dispatch_message="lease lost before operation",
                authoritative_failure_note="lease lost as operation failed",
                concurrent_failure_note="operation failed while lease loss was handled",
            )

    asyncio.run(scenario())

    assert ledger.heartbeat_calls == 2


@pytest.mark.parametrize("operation_failure_type", [RuntimeError, asyncio.CancelledError])
def test_controller_preserves_completed_metadata_when_caller_cancellation_wins(
    operation_failure_type: type[BaseException],
) -> None:
    store = InMemorySessionStore()
    ledger = InMemoryBudgetLedger(reservation_ttl_seconds=30)
    controller = _controller(store, ledger=ledger)
    completed_metadata = {
        "model": "fake-model",
        "usage": {"input_tokens": 8, "output_tokens": 2},
    }

    async def scenario() -> None:
        await _running_session(store, "sess_operation_caller_cancel_metadata")
        setup = await controller.reserve_operation_budgets(
            budget_limits=(_reserved_limit("2"),),
            session_id="sess_operation_caller_cancel_metadata",
            agent_name="assistant",
            provider_name="fake",
            model="fake-model",
            model_attempt_identity=_model_attempt_identity(),
            rejection_release_reason="reservation rejected",
            accepted_record_error="accepted reservation missing record",
        )
        operation_started = asyncio.Event()

        async def operation() -> str:
            operation_started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError as exc:
                operation_failure = operation_failure_type("operation ended after completion")
                operation_failure.__dict__["completed_metadata"] = completed_metadata
                raise operation_failure from exc

        task = asyncio.create_task(
            controller.run_operation_with_reservation_heartbeat(
                operation,
                reservations=list(setup.reservations),
                authoritative_failure_types=(),
                lease_lost_before_dispatch_message="lease lost before operation",
                authoritative_failure_note="lease lost as operation failed",
                concurrent_failure_note="operation failed while lease loss was handled",
            )
        )
        await operation_started.wait()
        task.cancel("caller cancelled")

        with pytest.raises(asyncio.CancelledError, match="caller cancelled") as raised:
            await task

        assert raised.value.__dict__["completed_metadata"] == completed_metadata
        assert raised.value.__dict__["completed_metadata"] is not completed_metadata
        assert task.cancelling() == 1
        assert task.cancelled()

    asyncio.run(scenario())


@pytest.mark.parametrize("operation_failure_type", [RuntimeError, asyncio.CancelledError])
def test_controller_preserves_completed_metadata_when_heartbeat_lease_loss_wins(
    operation_failure_type: type[BaseException],
) -> None:
    store = InMemorySessionStore()
    ledger = _LoseLeaseOnSecondHeartbeat()
    controller = _controller(store, ledger=ledger)
    completed_metadata = {
        "model": "fake-model",
        "usage": {"input_tokens": 8, "output_tokens": 2},
    }

    async def scenario() -> None:
        await _running_session(store, "sess_operation_lease_loss_metadata")
        setup = await controller.reserve_operation_budgets(
            budget_limits=(_reserved_limit("2"),),
            session_id="sess_operation_lease_loss_metadata",
            agent_name="assistant",
            provider_name="fake",
            model="fake-model",
            model_attempt_identity=_model_attempt_identity(),
            rejection_release_reason="reservation rejected",
            accepted_record_error="accepted reservation missing record",
        )

        async def operation() -> str:
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError as exc:
                operation_failure = operation_failure_type("operation ended after completion")
                operation_failure.__dict__["completed_metadata"] = completed_metadata
                raise operation_failure from exc

        with pytest.raises(BudgetReservationLeaseLost) as raised:
            await controller.run_operation_with_reservation_heartbeat(
                operation,
                reservations=list(setup.reservations),
                authoritative_failure_types=(),
                lease_lost_before_dispatch_message="lease lost before operation",
                authoritative_failure_note="lease lost as operation failed",
                concurrent_failure_note="operation failed while lease loss was handled",
            )

        assert raised.value.__dict__["completed_metadata"] == completed_metadata
        assert raised.value.__dict__["completed_metadata"] is not completed_metadata

    asyncio.run(scenario())

    assert ledger.heartbeat_calls == 2


def test_controller_returns_typed_success_for_budgeted_compactor_dispatch():
    class PricingInstantLedger(InMemoryBudgetLedger):
        def __init__(self) -> None:
            super().__init__()
            self.effective_at: list[datetime | None] = []

        async def reserve(self, **kwargs):
            self.effective_at.append(kwargs.get("effective_at"))
            return await super().reserve(**kwargs)

    store = InMemorySessionStore()
    ledger = PricingInstantLedger()
    controller = _controller(store, ledger=ledger)

    async def scenario():
        session = await _running_session(store, "sess_budgeted_dispatch_success")
        completion_events = [
            Event(
                type=EventType.MODEL_COMPLETED,
                session_id=session.id,
                agent_name="assistant",
                payload={
                    "provider_name": "fake",
                    "model": "fake-model",
                    "usage": {
                        "input_tokens": 250_000,
                        "output_tokens": 0,
                        "total_tokens": 250_000,
                    },
                },
            ),
            Event(
                type=EventType.MODEL_COMPLETED,
                session_id=session.id,
                agent_name="assistant",
                payload={
                    "provider_name": "fake",
                    "model": "fake-model",
                    "usage_unavailable_reason": "provider omitted usage",
                },
            ),
        ]
        return await controller.run_automatic_compaction_dispatch(
            lambda: asyncio.sleep(0, result="summary"),
            completed_events=lambda: completion_events,
            budget_limits=(_reserved_limit("3"),),
            session=session,
            agent_name="assistant",
            environment_name=None,
            provider_name="fake",
            model="fake-model",
            model_attempt_identity=_model_attempt_identity(),
            authoritative_failure_types=(),
        )

    outcome = asyncio.run(scenario())

    assert isinstance(outcome, BudgetedOperationSucceeded)
    assert outcome.result == "summary"
    assert ledger.effective_at == [datetime(2026, 1, 1, tzinfo=UTC)]
    assert [event.type for event in outcome.events] == [
        EventType.BUDGET_RESERVED,
        EventType.BUDGET_RECONCILED,
    ]
    assert Decimal(outcome.events[-1].payload["actual_amount"]) == Decimal("1.25")
    assert outcome.events[-1].payload["reason"] == (
        "automatic context compaction completed with partially uncertain usage; "
        "charged known cost plus reserved amount per uncertain completion"
    )


def test_controller_rejects_rewritten_compactor_reservation_amount_before_dispatch():
    class UnderReservingLedger(InMemoryBudgetLedger):
        async def reserve(self, **kwargs):
            result = await super().reserve(**kwargs)
            assert result.record is not None
            rewritten_record = result.record.model_copy(update={"reserved_amount": Decimal("0")})
            return result.model_copy(
                update={
                    "requested": Decimal("0"),
                    "record": rewritten_record,
                }
            )

    store = InMemorySessionStore()
    controller = _controller(store, ledger=UnderReservingLedger())
    dispatches = 0

    async def scenario():
        nonlocal dispatches
        session = await _running_session(store, "sess_under_reserved_compactor_dispatch")

        async def operation() -> str:
            nonlocal dispatches
            dispatches += 1
            return "summary"

        return await controller.run_automatic_compaction_dispatch(
            operation,
            completed_events=list,
            budget_limits=(_reserved_limit("3"),),
            session=session,
            agent_name="assistant",
            environment_name=None,
            provider_name="fake",
            model="fake-model",
            model_attempt_identity=_model_attempt_identity(),
            authoritative_failure_types=(),
        )

    outcome = asyncio.run(scenario())

    assert isinstance(outcome, BudgetedOperationFailed)
    assert dispatches == 0
    assert outcome.events == ()
    assert str(outcome.error) == "Budget ledger reservation result changed its requested amount."


def test_controller_releases_compactor_reservation_when_dispatch_observation_fails():
    store = InMemorySessionStore()
    ledger = InMemoryBudgetLedger(reservation_ttl_seconds=None)
    controller = _controller(store, ledger=ledger)
    observations = 0
    dispatches = 0

    async def scenario():
        nonlocal observations, dispatches
        session = await _running_session(store, "sess_compactor_observation_failure")

        async def observe() -> None:
            nonlocal observations
            observations += 1
            raise RuntimeError("footprint publication failed")

        async def operation() -> str:
            nonlocal dispatches
            dispatches += 1
            return "must not run"

        outcome = await controller.run_automatic_compaction_dispatch(
            operation,
            completed_events=list,
            budget_limits=(_reserved_limit("3"),),
            session=session,
            agent_name="assistant",
            environment_name=None,
            provider_name="fake",
            model="fake-model",
            model_attempt_identity=_model_attempt_identity(),
            authoritative_failure_types=(),
            before_provider_dispatch=observe,
        )
        assert len(ledger._records) == 1
        record = next(iter(ledger._records.values()))
        return outcome, record

    outcome, record = asyncio.run(scenario())

    assert isinstance(outcome, BudgetedOperationFailed)
    assert observations == 1
    assert dispatches == 0
    assert str(outcome.error) == "footprint publication failed"
    assert record.status == "released"
    assert [event.type for event in outcome.events] == [
        EventType.BUDGET_RESERVED,
        EventType.BUDGET_RESERVATION_RELEASED,
    ]
    assert outcome.events[-1].payload["reason"] == "reservation setup failed"


def test_controller_releases_compactor_reservation_when_observation_is_cancelled():
    store = InMemorySessionStore()
    ledger = InMemoryBudgetLedger(reservation_ttl_seconds=None)
    controller = _controller(store, ledger=ledger)

    async def scenario():
        session = await _running_session(store, "sess_compactor_observation_cancelled")
        observation_started = asyncio.Event()
        dispatches = 0

        async def observe() -> None:
            observation_started.set()
            await asyncio.Event().wait()

        async def operation() -> str:
            nonlocal dispatches
            dispatches += 1
            return "must not run"

        async def invoke() -> None:
            outcome = await controller.run_automatic_compaction_dispatch(
                operation,
                completed_events=list,
                budget_limits=(_reserved_limit("3"),),
                session=session,
                agent_name="assistant",
                environment_name=None,
                provider_name="fake",
                model="fake-model",
                model_attempt_identity=_model_attempt_identity(),
                authoritative_failure_types=(),
                before_provider_dispatch=observe,
            )
            assert isinstance(outcome, BudgetedOperationFailed)
            raise outcome.error

        task = asyncio.create_task(invoke())
        await observation_started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        records = await store.query_events(EventQuery(session_id=session.id, limit=100))
        return task, dispatches, [record.event for record in records]

    task, dispatches, events = asyncio.run(scenario())

    assert task.cancelling() == 1
    assert task.cancelled()
    assert dispatches == 0
    assert len(ledger._records) == 1
    assert next(iter(ledger._records.values())).status == "released"
    assert [event.type for event in events] == [
        EventType.BUDGET_RESERVED,
        EventType.BUDGET_RESERVATION_RELEASED,
    ]
    assert events[-1].payload["reason"] == "reservation setup cancelled"


def test_controller_rejects_reservation_id_reuse_across_compactor_dispatches():
    class ReusingReservationIdLedger(InMemoryBudgetLedger):
        first_reservation_id: str | None = None

        def __init__(self) -> None:
            super().__init__(reservation_ttl_seconds=None)

        async def reserve(self, **kwargs):
            result = await super().reserve(**kwargs)
            assert result.record is not None
            if self.first_reservation_id is None:
                self.first_reservation_id = result.record.reservation_id
                return result
            return result.model_copy(
                update={
                    "record": result.record.model_copy(
                        update={"reservation_id": self.first_reservation_id}
                    )
                }
            )

    store = InMemorySessionStore()
    controller = _controller(store, ledger=ReusingReservationIdLedger())
    dispatches = 0

    async def scenario():
        nonlocal dispatches
        session = await _running_session(store, "sess_reused_compactor_reservation_identity")

        async def operation() -> str:
            nonlocal dispatches
            dispatches += 1
            return "summary"

        def completed_events() -> list[Event]:
            return [
                Event(
                    type=EventType.MODEL_COMPLETED,
                    session_id=session.id,
                    agent_name="assistant",
                    payload={
                        "provider_name": "fake",
                        "model": "fake-model",
                        "usage": {
                            "input_tokens": 250_000,
                            "output_tokens": 0,
                            "total_tokens": 250_000,
                        },
                    },
                )
            ]

        first = await controller.run_automatic_compaction_dispatch(
            operation,
            completed_events=completed_events,
            budget_limits=(_reserved_limit("3"),),
            session=session,
            agent_name="assistant",
            environment_name=None,
            provider_name="fake",
            model="fake-model",
            model_attempt_identity=_model_attempt_identity(),
            authoritative_failure_types=(),
        )
        second = await controller.run_automatic_compaction_dispatch(
            operation,
            completed_events=completed_events,
            budget_limits=(_reserved_limit("3"),),
            session=session,
            agent_name="assistant",
            environment_name=None,
            provider_name="fake",
            model="fake-model",
            model_attempt_identity=_model_attempt_identity(),
            authoritative_failure_types=(),
        )
        return first, second

    first, second = asyncio.run(scenario())

    assert isinstance(first, BudgetedOperationSucceeded)
    assert isinstance(second, BudgetedOperationFailed)
    assert dispatches == 1
    assert second.events == ()
    assert str(second.error) == "Budget ledger reused a reservation identity."


def test_concurrent_compactor_collision_does_not_release_the_winner():
    class CoordinatedDuplicateLedger(InMemoryBudgetLedger):
        def __init__(self) -> None:
            super().__init__(reservation_ttl_seconds=None)
            self.first_reservation_id: str | None = None
            self.second_reserved = asyncio.Event()
            self.release_second_result = asyncio.Event()

        async def reserve(self, **kwargs):
            result = await super().reserve(**kwargs)
            assert result.record is not None
            if kwargs["session_id"] == "sess_compactor_collision_winner":
                self.first_reservation_id = result.record.reservation_id
                await self.second_reserved.wait()
                return result
            self.second_reserved.set()
            await self.release_second_result.wait()
            assert self.first_reservation_id is not None
            return result.model_copy(
                update={
                    "record": result.record.model_copy(
                        update={"reservation_id": self.first_reservation_id}
                    )
                }
            )

    async def scenario():
        store = InMemorySessionStore()
        ledger = CoordinatedDuplicateLedger()
        winner_controller = _controller(store, ledger=ledger)
        loser_controller = _controller(store, ledger=ledger)
        winner = await _running_session(store, "sess_compactor_collision_winner")
        loser = await _running_session(store, "sess_compactor_collision_loser")
        operation_started = asyncio.Event()
        finish_operation = asyncio.Event()

        async def winner_operation() -> str:
            operation_started.set()
            await finish_operation.wait()
            return "summary"

        async def loser_operation() -> str:
            raise AssertionError("colliding compactor dispatch must not start")

        def completed_events(session: Session) -> list[Event]:
            return [
                Event(
                    type=EventType.MODEL_COMPLETED,
                    session_id=session.id,
                    agent_name="assistant",
                    payload={
                        "provider_name": "fake",
                        "model": "fake-model",
                        "usage": {
                            "input_tokens": 250_000,
                            "output_tokens": 0,
                            "total_tokens": 250_000,
                        },
                    },
                )
            ]

        winner_task = asyncio.create_task(
            winner_controller.run_automatic_compaction_dispatch(
                winner_operation,
                completed_events=lambda: completed_events(winner),
                budget_limits=(_reserved_limit("3"),),
                session=winner,
                agent_name="assistant",
                environment_name=None,
                provider_name="fake",
                model="fake-model",
                model_attempt_identity=_model_attempt_identity(),
                authoritative_failure_types=(),
            )
        )
        loser_task = asyncio.create_task(
            loser_controller.run_automatic_compaction_dispatch(
                loser_operation,
                completed_events=lambda: completed_events(loser),
                budget_limits=(_reserved_limit("3"),),
                session=loser,
                agent_name="assistant",
                environment_name=None,
                provider_name="fake",
                model="fake-model",
                model_attempt_identity=_model_attempt_identity(),
                authoritative_failure_types=(),
            )
        )
        await asyncio.wait_for(operation_started.wait(), timeout=5)
        ledger.release_second_result.set()
        loser_outcome = await asyncio.wait_for(loser_task, timeout=5)
        assert ledger.first_reservation_id is not None
        winner_still_active = await ledger.heartbeat(reservation_id=ledger.first_reservation_id)
        finish_operation.set()
        winner_outcome = await asyncio.wait_for(winner_task, timeout=5)
        return winner_outcome, loser_outcome, winner_still_active

    winner, loser, winner_still_active = asyncio.run(scenario())

    assert winner_still_active is True
    assert isinstance(winner, BudgetedOperationSucceeded)
    assert isinstance(loser, BudgetedOperationFailed)
    assert [event.type for event in winner.events] == [
        EventType.BUDGET_RESERVED,
        EventType.BUDGET_RECONCILED,
    ]
    assert loser.events == ()
    assert str(loser.error) == "Budget ledger reused a reservation identity."


def test_controller_returns_typed_rejection_without_dispatching_compactor():
    store = InMemorySessionStore()
    controller = _controller(store)
    dispatched = False

    async def scenario():
        session = await _running_session(store, "sess_budgeted_dispatch_rejected")

        async def operation() -> str:
            nonlocal dispatched
            dispatched = True
            return "must not run"

        return await controller.run_automatic_compaction_dispatch(
            operation,
            completed_events=list,
            budget_limits=(_reserved_limit("0.5"),),
            session=session,
            agent_name="assistant",
            environment_name=None,
            provider_name="fake",
            model="fake-model",
            model_attempt_identity=_model_attempt_identity(),
            authoritative_failure_types=(),
        )

    outcome = asyncio.run(scenario())

    assert isinstance(outcome, BudgetedOperationRejected)
    assert dispatched is False
    assert outcome.failure.accepted is False
    assert [event.type for event in outcome.events] == [EventType.BUDGET_RESERVATION_FAILED]


def test_controller_returns_typed_rejection_for_unreservable_bedrock_tier():
    store = InMemorySessionStore()
    ledger = InMemoryBudgetLedger()
    controller = _controller(store, ledger=ledger)
    dispatched = False
    model = "global.anthropic.claude-sonnet-4-6"
    identity = bedrock_billing_identity(
        invoked_model=model,
        source_region="us-east-1",
        resource_type="inference_profile",
        profile_scope="global",
        requested_service_tier="reserved",
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

    async def scenario():
        session = await _running_session(store, "sess_unreservable_compactor_tier")

        async def operation() -> str:
            nonlocal dispatched
            dispatched = True
            return "must not run"

        return await controller.run_automatic_compaction_dispatch(
            operation,
            completed_events=list,
            budget_limits=(
                BudgetLimit(
                    scope="app",
                    max_estimated_cost=Decimal("20"),
                    pricing=pricing,
                    reservation=BudgetReservation(
                        max_input_tokens=1_000_000,
                        max_output_tokens=1_000_000,
                    ),
                ),
            ),
            session=session,
            agent_name="assistant",
            environment_name=None,
            provider_name="bedrock",
            model=model,
            model_attempt_identity=_model_attempt_identity(),
            billing_identity=identity,
            authoritative_failure_types=(),
        )

    outcome = asyncio.run(scenario())

    assert isinstance(outcome, BudgetedOperationRejected)
    assert dispatched is False
    assert ledger._records == {}
    assert "no matching model pricing" in outcome.failure.message
    assert [event.type for event in outcome.events] == [EventType.BUDGET_RESERVATION_FAILED]


def test_controller_settles_partial_setup_before_returning_typed_cancellation():
    store = InMemorySessionStore()
    ledger = _CancelSecondReservationLedger()
    controller = _controller(store, ledger=ledger)

    async def scenario():
        session = await _running_session(store, "sess_budgeted_dispatch_setup_cancelled")
        return await controller.run_automatic_compaction_dispatch(
            lambda: asyncio.sleep(0, result="must not run"),
            completed_events=list,
            budget_limits=(_reserved_limit("3"), _reserved_limit("3")),
            session=session,
            agent_name="assistant",
            environment_name=None,
            provider_name="fake",
            model="fake-model",
            model_attempt_identity=_model_attempt_identity(),
            authoritative_failure_types=(),
        )

    outcome = asyncio.run(scenario())

    assert isinstance(outcome, BudgetedOperationFailed)
    assert isinstance(outcome.error, asyncio.CancelledError)
    assert [event.type for event in outcome.events] == [
        EventType.BUDGET_RESERVED,
        EventType.BUDGET_RESERVATION_RELEASED,
    ]
    assert outcome.events[-1].payload["reason"] == "reservation setup cancelled"


def test_controller_attempts_every_compactor_settlement_after_one_limit_fails():
    store = InMemorySessionStore()
    ledger = _FailSecondReconcileLedger()
    controller = _controller(store, ledger=ledger)

    async def scenario():
        session = await _running_session(store, "sess_budgeted_dispatch_partial_settlement")
        completion_events = [
            Event(
                type=EventType.MODEL_COMPLETED,
                session_id=session.id,
                agent_name="assistant",
                payload={
                    "provider_name": "fake",
                    "model": "fake-model",
                    "usage": {
                        "input_tokens": 250_000,
                        "output_tokens": 0,
                        "total_tokens": 250_000,
                    },
                },
            )
        ]
        return await controller.run_automatic_compaction_dispatch(
            lambda: asyncio.sleep(0, result="summary"),
            completed_events=lambda: completion_events,
            budget_limits=(
                _reserved_limit("4"),
                _reserved_limit("4"),
                _reserved_limit("4"),
            ),
            session=session,
            agent_name="assistant",
            environment_name=None,
            provider_name="fake",
            model="fake-model",
            model_attempt_identity=_model_attempt_identity(),
            authoritative_failure_types=(),
        )

    outcome = asyncio.run(scenario())

    assert isinstance(outcome, BudgetedOperationFailed)
    assert str(outcome.error) == "simulated second reconciliation failure"
    assert ledger.reconcile_calls == 3
    assert [event.type for event in outcome.events].count(EventType.BUDGET_RECONCILED) == 2

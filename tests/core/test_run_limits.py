"""Focused contracts for the concrete run-limit controller."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from decimal import Decimal

import pytest

from cayu.core import AgentSpec, Event, EventType, Message
from cayu.providers import ModelProvider, ModelRequest, ModelStreamEvent
from cayu.runtime import CayuApp
from cayu.runtime._event_writer import RuntimeEventWriter
from cayu.runtime._run_limits import (
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
    BudgetReservationResult,
    InMemoryBudgetLedger,
    SessionBudgetStore,
)
from cayu.runtime.costs import ModelPrice, PriceBook
from cayu.runtime.sessions import (
    EventQuery,
    InMemorySessionStore,
    RunRequest,
    Session,
    SessionIdentity,
    SessionStatus,
)
from cayu.runtime.stop_policy import RunLimits, StopLimit


def _controller(
    store: InMemorySessionStore,
    *,
    ledger: BudgetLedger | None = None,
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
        clock=lambda: datetime(2026, 1, 1, tzinfo=UTC),
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
        limit: BudgetLimit,
        session_id: str,
        agent_name: str,
        provider_name: str,
        model: str,
    ) -> BudgetReservationResult:
        self.reserve_calls += 1
        if self.reserve_calls == 2:
            raise asyncio.CancelledError
        result = await super().reserve(
            limit=limit,
            session_id=session_id,
            agent_name=agent_name,
            provider_name=provider_name,
            model=model,
        )
        if result.record is not None:
            self.reservation_ids.append(result.record.reservation_id)
        return result


class _FailSecondReleaseLedger(InMemoryBudgetLedger):
    def __init__(self) -> None:
        super().__init__()
        self.release_calls = 0

    async def release(self, *, reservation_id: str, reason: str):
        self.release_calls += 1
        if self.release_calls == 2:
            raise RuntimeError("simulated second release failure")
        return await super().release(reservation_id=reservation_id, reason=reason)


class _CancelFirstHeartbeatLedger(InMemoryBudgetLedger):
    def __init__(self) -> None:
        super().__init__()
        self.reservation_ids: list[str] = []
        self.release_calls = 0

    async def reserve(
        self,
        *,
        limit: BudgetLimit,
        session_id: str,
        agent_name: str,
        provider_name: str,
        model: str,
    ) -> BudgetReservationResult:
        result = await super().reserve(
            limit=limit,
            session_id=session_id,
            agent_name=agent_name,
            provider_name=provider_name,
            model=model,
        )
        if result.record is not None:
            self.reservation_ids.append(result.record.reservation_id)
        return result

    async def heartbeat(self, *, reservation_id: str) -> bool:
        raise asyncio.CancelledError

    async def release(self, *, reservation_id: str, reason: str):
        self.release_calls += 1
        return await super().release(reservation_id=reservation_id, reason=reason)


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
        EventType.SESSION_STARTED,
        EventType.BUDGET_CHECKED,
        EventType.BUDGET_LIMIT_REACHED,
        EventType.SESSION_LIMIT_REACHED,
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
        setup = await controller.reserve_operation_budgets(
            budget_limits=(_reserved_limit("2"), _reserved_limit("0.5")),
            session_id="sess_operation_rejection",
            agent_name="assistant",
            provider_name="fake",
            model="fake-model",
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


def test_controller_returns_partial_reservations_when_setup_is_cancelled():
    store = InMemorySessionStore()
    ledger = _CancelSecondReservationLedger()
    controller = _controller(store, ledger=ledger)

    async def scenario():
        setup = await controller.reserve_operation_budgets(
            budget_limits=(_reserved_limit("3"), _reserved_limit("3")),
            session_id="sess_operation_setup_cancelled",
            agent_name="assistant",
            provider_name="fake",
            model="fake-model",
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
    controller = _controller(store)

    async def scenario():
        setup = await controller.reserve_operation_budgets(
            budget_limits=(_reserved_limit("2"),),
            session_id="sess_operation_reconcile",
            agent_name="assistant",
            provider_name="fake",
            model="fake-model",
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


def test_controller_arbitrates_operation_heartbeat_lease_loss():
    store = InMemorySessionStore()
    ledger = _LoseLeaseOnSecondHeartbeat()
    controller = _controller(store, ledger=ledger)

    async def scenario() -> None:
        setup = await controller.reserve_operation_budgets(
            budget_limits=(_reserved_limit("2"),),
            session_id="sess_operation_heartbeat",
            agent_name="assistant",
            provider_name="fake",
            model="fake-model",
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

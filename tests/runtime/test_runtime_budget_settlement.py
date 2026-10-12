"""Budget settlement contracts."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from decimal import Decimal

import pytest
from tests.runtime.runtime_budget_support import BlockingBudgetProvider, fake_budget_limit
from tests.runtime.runtime_provider_support import FakeProvider
from tests.runtime.runtime_session_support import _private_record_for_public_event, collect_events

import cayu.runtime._live_model_attempt as live_model_attempt_module
import cayu.runtime._model_step_executor as model_step_executor_module
from cayu.agents import AgentSpec
from cayu.applications import CayuApp
from cayu.budgets.base import (
    BudgetLimit,
    BudgetPolicy,
    BudgetReservation,
    InMemoryBudgetLedger,
    budget_settlement_id,
)
from cayu.configuration import CayuConfig, RunDefaults
from cayu.context.counting import ContextCountingConfig, ContextCountingMode
from cayu.context.footprints import RequestFootprintConfig
from cayu.events import Event, EventType
from cayu.messages import Message
from cayu.providers.base import (
    InputTokenCountConfidence,
    InputTokenCountMethod,
    InputTokenCountResult,
    ModelProvider,
    ModelRequest,
    ModelStreamEvent,
)
from cayu.providers.retry_policy import RetryPolicy
from cayu.sessions.event_queries import EventQuery
from cayu.sessions.memory import InMemorySessionStore
from cayu.sessions.records import Session, SessionStatus
from cayu.sessions.requests import RunRequest


def test_cayu_app_conservatively_reconciles_unknown_post_dispatch_spend():
    store = InMemorySessionStore()
    provider = FakeProvider(
        [
            [ModelStreamEvent.error("provider down")],
            [
                ModelStreamEvent.completed(
                    {
                        "finish_reason": "stop",
                        "usage": {
                            "input_tokens": 1,
                            "output_tokens": 0,
                            "total_tokens": 1,
                        },
                    }
                )
            ],
        ]
    )
    app = CayuApp(
        session_store=store,
        budget_policy=BudgetPolicy(
            limits=(
                BudgetLimit(
                    scope="app",
                    max_estimated_cost=Decimal("1"),
                    pricing=fake_budget_limit("10").pricing,
                    reservation=BudgetReservation(
                        max_input_tokens=1_000_000,
                        max_output_tokens=0,
                    ),
                ),
            )
        ),
    )
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    failed_events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_budget_reservation_failed",
                messages=[Message.text("user", "first")],
            ),
        )
    )
    retry_events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_budget_reservation_retry",
                messages=[Message.text("user", "second")],
            ),
        )
    )

    failed_event_types = [event.type for event in failed_events]
    assert EventType.BUDGET_RESERVATION_RELEASED not in failed_event_types
    assert failed_events[-1].type == EventType.SESSION_FAILED
    reconciliation = next(
        event for event in failed_events if event.type == EventType.BUDGET_RECONCILED
    )
    assert reconciliation.payload["actual_amount"] == "1"
    assert reconciliation.payload["reason"] == (
        "provider usage unknown after dispatch; charged reserved amount"
    )
    assert "provider down" not in json.dumps(reconciliation.payload)

    retry_event_types = [event.type for event in retry_events]
    assert EventType.BUDGET_RESERVATION_FAILED in retry_event_types
    assert EventType.BUDGET_RESERVED not in retry_event_types
    assert retry_events[-1].type == EventType.SESSION_INTERRUPTED
    assert len(provider.requests) == 1


def test_cayu_app_reconciles_reserved_amount_when_provider_raises_after_dispatch() -> None:
    class RaisingBudgetProvider(ModelProvider):
        name = "fake"

        def __init__(self) -> None:
            self.requests: list[ModelRequest] = []

        async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
            self.requests.append(request)
            raise RuntimeError("provider transport failed")
            yield  # pragma: no cover

    provider = RaisingBudgetProvider()
    app = CayuApp(
        budget_policy=BudgetPolicy(
            limits=(
                BudgetLimit(
                    scope="app",
                    max_estimated_cost=Decimal("1"),
                    pricing=fake_budget_limit("10").pricing,
                    reservation=BudgetReservation(
                        max_input_tokens=1_000_000,
                        max_output_tokens=0,
                    ),
                ),
            )
        )
    )
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_budget_provider_raised",
                messages=[Message.text("user", "hello")],
            ),
        )
    )

    assert len(provider.requests) == 1
    assert events[-1].type == EventType.SESSION_FAILED
    assert events[-1].payload["error"] == "provider transport failed"
    reconciliation = next(event for event in events if event.type == EventType.BUDGET_RECONCILED)
    assert reconciliation.payload["actual_amount"] == "1"
    assert reconciliation.payload["reason"] == (
        "provider usage unknown after dispatch; charged reserved amount"
    )


def test_cayu_app_treats_provider_exception_group_as_an_ordinary_failure() -> None:
    class GroupRaisingProvider(ModelProvider):
        name = "group-raising"

        async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
            raise ExceptionGroup(
                "provider failures",
                [RuntimeError("transport failed"), ValueError("invalid response")],
            )
            yield  # pragma: no cover

    async def run() -> tuple[list[Event], Session | None]:
        store = InMemorySessionStore()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(GroupRaisingProvider(), default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        events = await collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_provider_exception_group",
                messages=[Message.text("user", "hello")],
            ),
        )
        return events, await store.load("sess_provider_exception_group")

    events, session = asyncio.run(run())

    assert session is not None
    assert session.status is SessionStatus.FAILED
    assert events[-1].type is EventType.SESSION_FAILED
    assert events[-1].payload["error_type"] == "ExceptionGroup"


def test_cayu_app_releases_reservation_for_failure_before_provider_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = FakeProvider([ModelStreamEvent.completed({"finish_reason": "stop"})])
    app = CayuApp(
        request_footprint=RequestFootprintConfig(enabled=False),
        budget_policy=BudgetPolicy(
            limits=(
                BudgetLimit(
                    scope="app",
                    max_estimated_cost=Decimal("1"),
                    pricing=fake_budget_limit("10").pricing,
                    reservation=BudgetReservation(
                        max_input_tokens=1_000_000,
                        max_output_tokens=0,
                    ),
                ),
            )
        ),
    )
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    def fail_before_dispatch(*_args, **_kwargs):
        raise RuntimeError("request preparation failed")

    monkeypatch.setattr(
        model_step_executor_module,
        "analyze_request_context_pressure",
        fail_before_dispatch,
    )
    monkeypatch.setattr(
        live_model_attempt_module,
        "analyze_request_context_pressure",
        fail_before_dispatch,
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_budget_pre_dispatch_failure",
                messages=[Message.text("user", "hello")],
            ),
        )
    )

    assert provider.requests == []
    assert EventType.BUDGET_RECONCILED not in [event.type for event in events]
    released = next(
        event for event in events if event.type == EventType.BUDGET_RESERVATION_RELEASED
    )
    assert released.payload["reason"] == "model step failed before provider dispatch"
    assert events[-1].type == EventType.SESSION_FAILED
    assert events[-1].payload["error"] == "request preparation failed"


def test_cayu_app_uses_durable_completion_usage_when_provider_fails_after_completion() -> None:
    provider = FakeProvider(
        [
            ModelStreamEvent.completed(
                {
                    "finish_reason": "stop",
                    "usage": {
                        "input_tokens": 250_000,
                        "output_tokens": 0,
                        "total_tokens": 250_000,
                    },
                }
            ),
            ModelStreamEvent.error("late provider failure"),
        ]
    )
    app = CayuApp(
        budget_policy=BudgetPolicy(
            limits=(
                BudgetLimit(
                    scope="app",
                    max_estimated_cost=Decimal("1"),
                    pricing=fake_budget_limit("10").pricing,
                    reservation=BudgetReservation(
                        max_input_tokens=1_000_000,
                        max_output_tokens=0,
                    ),
                ),
            )
        )
    )
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_budget_failure_after_completion",
                messages=[Message.text("user", "hello")],
            ),
        )
    )

    reconciliations = [event for event in events if event.type == EventType.BUDGET_RECONCILED]
    assert len(reconciliations) == 1
    assert reconciliations[0].payload["actual_amount"] == "0.25"
    assert reconciliations[0].payload["reason"] == "model completed"
    assert events[-1].type == EventType.SESSION_FAILED
    assert events[-1].payload["error"] == "Model provider emitted event after completed: error"


@pytest.mark.parametrize(
    (
        "first_attempt_completed",
        "budget_maximum",
        "expected_request_count",
        "expected_amounts",
        "expected_final_reason",
        "expected_final_pricing_none",
    ),
    [
        pytest.param(
            False,
            "2",
            2,
            ["1", "1"],
            "provider usage unknown after dispatch; charged reserved amount",
            True,
            id="two-unknown-attempts",
        ),
        pytest.param(
            True,
            "1.25",
            1,
            ["0.25"],
            "model completed",
            False,
            id="authoritative-completion-is-not-retried",
        ),
    ],
)
def test_cayu_app_accounts_for_each_dispatched_retry(
    first_attempt_completed: bool,
    budget_maximum: str,
    expected_request_count: int,
    expected_amounts: list[str],
    expected_final_reason: str,
    expected_final_pricing_none: bool,
) -> None:
    class RetriedBudgetProvider(ModelProvider):
        name = "fake"

        def __init__(self) -> None:
            self.requests: list[ModelRequest] = []

        async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
            self.requests.append(request)
            if len(self.requests) == 1 and first_attempt_completed:
                yield ModelStreamEvent.completed(
                    {
                        "finish_reason": "stop",
                        "usage": {
                            "input_tokens": 250_000,
                            "output_tokens": 0,
                            "total_tokens": 250_000,
                        },
                    }
                )
            raise TimeoutError("stream idle timeout")

    provider = RetriedBudgetProvider()
    app = CayuApp(
        config=CayuConfig(
            run=RunDefaults(retry_policy=RetryPolicy(max_attempts=2, initial_delay_s=0.0))
        ),
        budget_policy=BudgetPolicy(
            limits=(
                BudgetLimit(
                    scope="app",
                    max_estimated_cost=Decimal(budget_maximum),
                    pricing=fake_budget_limit("10").pricing,
                    reservation=BudgetReservation(
                        max_input_tokens=1_000_000,
                        max_output_tokens=0,
                    ),
                ),
            )
        ),
    )
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id=f"sess_budget_retried_{first_attempt_completed}",
                messages=[Message.text("user", "hello")],
            ),
        )
    )

    assert len(provider.requests) == expected_request_count
    reconciliations = [event for event in events if event.type == EventType.BUDGET_RECONCILED]
    assert [event.payload["actual_amount"] for event in reconciliations] == expected_amounts
    assert reconciliations[-1].payload["reason"] == expected_final_reason
    assert (reconciliations[-1].payload["pricing"] is None) is expected_final_pricing_none
    assert events[-1].type == EventType.SESSION_FAILED


@pytest.mark.parametrize("failed_operation", ["reconcile", "reserve"])
def test_cayu_app_retry_accounting_failure_does_not_mask_provider_failure(
    failed_operation: str,
) -> None:
    class FailingRetryPreparationLedger(InMemoryBudgetLedger):
        def __init__(self) -> None:
            super().__init__()
            self.reserve_calls = 0

        async def reserve(self, **kwargs):
            self.reserve_calls += 1
            if failed_operation == "reserve" and self.reserve_calls == 2:
                raise RuntimeError("budget ledger unavailable")
            return await super().reserve(**kwargs)

        async def reconcile(self, **kwargs):
            if failed_operation == "reconcile":
                raise RuntimeError("budget ledger unavailable")
            return await super().reconcile(**kwargs)

    class RetriedBudgetProvider(ModelProvider):
        name = "fake"

        def __init__(self) -> None:
            self.requests: list[ModelRequest] = []

        async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
            self.requests.append(request)
            raise TimeoutError("authoritative provider timeout")
            yield  # pragma: no cover

    session_id = f"sess_budget_retry_{failed_operation}_failed"
    store = InMemorySessionStore()
    ledger = FailingRetryPreparationLedger()
    provider = RetriedBudgetProvider()
    app = CayuApp(
        session_store=store,
        config=CayuConfig(
            run=RunDefaults(retry_policy=RetryPolicy(max_attempts=2, initial_delay_s=0.0))
        ),
        budget_policy=BudgetPolicy(
            limits=(
                BudgetLimit(
                    scope="app",
                    max_estimated_cost=Decimal("2"),
                    pricing=fake_budget_limit("10").pricing,
                    reservation=BudgetReservation(
                        max_input_tokens=1_000_000,
                        max_output_tokens=0,
                    ),
                ),
            )
        ),
        budget_ledger=ledger,
    )
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    async def run():
        events = []
        request = RunRequest(
            agent_name="assistant",
            session_id=session_id,
            messages=[Message.text("user", "hello")],
        )
        if failed_operation == "reconcile":
            with pytest.raises(TimeoutError, match="authoritative provider timeout"):
                async for event in app.run(request):
                    events.append(event)
        else:
            events = await collect_events(app, request)
        return (
            events,
            await store.load(session_id),
            await store.load_active_model_completion_stage(session_id),
        )

    events, session, active_stage = asyncio.run(run())

    assert len(provider.requests) == 1
    assert [event.type for event in events].count(EventType.MODEL_STARTED) == 1
    assert session is not None
    if failed_operation == "reconcile":
        assert session.status == SessionStatus.RUNNING
        assert active_stage is not None
        record = next(iter(ledger._records.values()))
        assert active_stage.stage.reservation_ids == (record.reservation_id,)
        assert record.status == "active"
        assert (
            asyncio.run(ledger.load_settlement(budget_settlement_id(record.reservation_id))) is None
        )
        assert EventType.SESSION_FAILED not in [event.type for event in events]
    else:
        assert session.status == SessionStatus.FAILED
        assert active_stage is None
        assert events[-1].type == EventType.SESSION_FAILED
        assert events[-1].payload["error"] == "authoritative provider timeout"
        assert events[-1].payload["error_type"] == "TimeoutError"


def test_cayu_app_releases_partial_retry_reservations_when_later_reserve_raises() -> None:
    class PartiallyFailingReservationLedger(InMemoryBudgetLedger):
        def __init__(self) -> None:
            super().__init__()
            self.reserve_calls = 0

        async def reserve(self, **kwargs):
            self.reserve_calls += 1
            if self.reserve_calls == 4:
                raise RuntimeError("second retry reservation failed")
            return await super().reserve(**kwargs)

    class RetriedBudgetProvider(ModelProvider):
        name = "fake"

        def __init__(self) -> None:
            self.requests: list[ModelRequest] = []

        async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
            self.requests.append(request)
            raise TimeoutError("authoritative provider timeout")
            yield  # pragma: no cover

    async def run():
        session_id = "sess_budget_partial_retry_reservation_failure"
        store = InMemorySessionStore()
        ledger = PartiallyFailingReservationLedger()
        provider = RetriedBudgetProvider()
        reservation = BudgetReservation(max_input_tokens=1_000_000, max_output_tokens=0)
        pricing = fake_budget_limit("10").pricing
        app = CayuApp(
            session_store=store,
            config=CayuConfig(
                run=RunDefaults(retry_policy=RetryPolicy(max_attempts=2, initial_delay_s=0.0))
            ),
            budget_policy=BudgetPolicy(
                limits=(
                    BudgetLimit(
                        scope="app",
                        max_estimated_cost=Decimal("3"),
                        pricing=pricing,
                        reservation=reservation,
                    ),
                    BudgetLimit(
                        scope="agent",
                        key="assistant",
                        max_estimated_cost=Decimal("3"),
                        pricing=pricing,
                        reservation=reservation,
                    ),
                )
            ),
            budget_ledger=ledger,
        )
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))

        events = await collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "hello")],
            ),
        )
        return (
            events,
            await store.query_events(EventQuery(session_id=session_id, limit=100)),
            tuple(ledger._records.values()),
            provider,
        )

    events, stored_records, records, provider = asyncio.run(run())
    stored_events = [record.event for record in stored_records]

    assert len(provider.requests) == 1
    assert [record.status for record in records] == ["reconciled", "reconciled", "released"]
    assert records[-1].reason == "reservation setup failed"
    released = [
        event for event in stored_events if event.type == EventType.BUDGET_RESERVATION_RELEASED
    ]
    streamed_releases = [
        event for event in events if event.type == EventType.BUDGET_RESERVATION_RELEASED
    ]
    assert len(released) == 1
    assert len(streamed_releases) == 1
    assert released[0].payload["reservation_id"] == records[-1].reservation_id
    assert (
        _private_record_for_public_event(stored_records, streamed_releases[0]).event == released[0]
    )
    assert events[-1].type == EventType.SESSION_FAILED
    assert events[-1].payload["error"] == "authoritative provider timeout"


def test_cayu_app_reconciles_dispatched_budget_when_run_stream_is_abandoned() -> None:
    async def run():
        session_id = "sess_budget_abandoned_after_provider_error"
        store = InMemorySessionStore()
        ledger = InMemoryBudgetLedger()
        app = CayuApp(
            session_store=store,
            budget_policy=BudgetPolicy(
                limits=(
                    BudgetLimit(
                        scope="app",
                        max_estimated_cost=Decimal("1"),
                        pricing=fake_budget_limit("10").pricing,
                        reservation=BudgetReservation(
                            max_input_tokens=1_000_000,
                            max_output_tokens=0,
                        ),
                    ),
                )
            ),
            budget_ledger=ledger,
        )
        app.register_provider(
            FakeProvider([ModelStreamEvent.error("authoritative provider failure")]),
            default=True,
        )
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))

        stream = app.run(
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "hello")],
            )
        )
        async for event in stream:
            if event.type == EventType.MODEL_ERROR:
                await stream.aclose()
                break

        session = await store.load(session_id)
        stored_events = await store.load_events(session_id)
        return session, stored_events, tuple(ledger._records.values())

    session, stored_events, records = asyncio.run(run())

    assert session is not None
    assert session.status == SessionStatus.INTERRUPTED
    assert len(records) == 1
    assert records[0].status == "reconciled"
    assert records[0].actual_amount == Decimal("1")
    assert records[0].reason == ("provider usage unknown after dispatch; charged reserved amount")
    reconciliation = next(
        event for event in stored_events if event.type == EventType.BUDGET_RECONCILED
    )
    assert reconciliation.payload["reservation_id"] == records[0].reservation_id
    assert stored_events[-1].type == EventType.SESSION_INTERRUPTED
    assert stored_events[-1].payload["abandoned"] is True


def test_cayu_app_releases_pending_budget_when_run_stream_is_abandoned() -> None:
    async def run():
        session_id = "sess_budget_abandoned_before_provider_dispatch"
        store = InMemorySessionStore()
        ledger = InMemoryBudgetLedger()
        provider = FakeProvider([ModelStreamEvent.completed({"finish_reason": "stop"})])
        app = CayuApp(
            session_store=store,
            budget_policy=BudgetPolicy(
                limits=(
                    BudgetLimit(
                        scope="app",
                        max_estimated_cost=Decimal("1"),
                        pricing=fake_budget_limit("10").pricing,
                        reservation=BudgetReservation(
                            max_input_tokens=1_000_000,
                            max_output_tokens=0,
                        ),
                    ),
                )
            ),
            budget_ledger=ledger,
        )
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))

        stream = app.run(
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "hello")],
            )
        )
        async for event in stream:
            if event.type == EventType.BUDGET_RESERVED:
                await stream.aclose()
                break

        session = await store.load(session_id)
        stored_events = await store.load_events(session_id)
        return session, stored_events, tuple(ledger._records.values()), provider.requests

    session, stored_events, records, provider_requests = asyncio.run(run())

    assert session is not None
    assert session.status == SessionStatus.INTERRUPTED
    assert provider_requests == []
    assert len(records) == 1
    assert records[0].status == "released"
    assert records[0].reason == "model step abandoned before provider dispatch"
    assert EventType.BUDGET_RECONCILED not in [event.type for event in stored_events]
    released = next(
        event for event in stored_events if event.type == EventType.BUDGET_RESERVATION_RELEASED
    )
    assert released.payload["reservation_id"] == records[0].reservation_id


def test_cayu_app_abandonment_retains_recovery_authority_when_budget_settlement_fails() -> None:
    class FailingReconciliationLedger(InMemoryBudgetLedger):
        async def reconcile(self, **_kwargs):
            raise RuntimeError("budget ledger unavailable")

    async def run():
        session_id = "sess_budget_abandoned_settlement_failed"
        store = InMemorySessionStore()
        ledger = FailingReconciliationLedger(reservation_ttl_seconds=None)
        app = CayuApp(
            session_store=store,
            budget_policy=BudgetPolicy(
                limits=(
                    BudgetLimit(
                        scope="app",
                        max_estimated_cost=Decimal("1"),
                        pricing=fake_budget_limit("10").pricing,
                        reservation=BudgetReservation(
                            max_input_tokens=1_000_000,
                            max_output_tokens=0,
                        ),
                    ),
                )
            ),
            budget_ledger=ledger,
        )
        app.register_provider(
            FakeProvider([ModelStreamEvent.error("authoritative provider failure")]),
            default=True,
        )
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))

        stream = app.run(
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "hello")],
            )
        )
        async for event in stream:
            if event.type == EventType.MODEL_ERROR:
                await stream.aclose()
                break
        record = next(iter(ledger._records.values()))
        return (
            await store.load(session_id),
            await store.load_events(session_id),
            await store.load_active_model_completion_stage(session_id),
            record,
            await ledger.load_settlement(budget_settlement_id(record.reservation_id)),
        )

    session, stored_events, active_stage, record, settlement = asyncio.run(run())

    assert session is not None
    assert session.status == SessionStatus.RUNNING
    assert active_stage is not None
    assert active_stage.stage.reservation_ids == (record.reservation_id,)
    assert record.status == "active"
    assert settlement is None
    assert EventType.SESSION_INTERRUPTED not in [event.type for event in stored_events]


def test_cayu_app_persists_successful_reconciliation_before_later_limit_fails() -> None:
    class PartiallyFailingLedger(InMemoryBudgetLedger):
        def __init__(self) -> None:
            super().__init__()
            self.reconcile_calls = 0

        async def reconcile(self, **kwargs):
            self.reconcile_calls += 1
            if self.reconcile_calls == 2:
                raise RuntimeError("second reconciliation failed")
            return await super().reconcile(**kwargs)

    async def run():
        session_id = "sess_budget_partial_reconciliation_failure"
        store = InMemorySessionStore()
        ledger = PartiallyFailingLedger()
        reservation = BudgetReservation(max_input_tokens=1_000_000, max_output_tokens=0)
        pricing = fake_budget_limit("10").pricing
        app = CayuApp(
            session_store=store,
            budget_policy=BudgetPolicy(
                limits=(
                    BudgetLimit(
                        scope="app",
                        max_estimated_cost=Decimal("2"),
                        pricing=pricing,
                        reservation=reservation,
                    ),
                    BudgetLimit(
                        scope="agent",
                        key="assistant",
                        max_estimated_cost=Decimal("2"),
                        pricing=pricing,
                        reservation=reservation,
                    ),
                )
            ),
            budget_ledger=ledger,
        )
        app.register_provider(
            FakeProvider([ModelStreamEvent.error("authoritative provider failure")]),
            default=True,
        )
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))

        events = []
        with pytest.raises(RuntimeError, match="authoritative provider failure"):
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "hello")],
                )
            ):
                events.append(event)
        return (
            events,
            await store.load(session_id),
            await store.load_active_model_completion_stage(session_id),
            await store.query_events(EventQuery(session_id=session_id, limit=100)),
            tuple(ledger._records.values()),
        )

    events, session, active_stage, stored_records, records = asyncio.run(run())
    stored_events = [record.event for record in stored_records]

    reconciliations = [
        event for event in stored_events if event.type == EventType.BUDGET_RECONCILED
    ]
    assert [record.status for record in records] == ["reconciled", "active"]
    assert session is not None and session.status == SessionStatus.RUNNING
    assert active_stage is not None
    assert active_stage.stage.reservation_ids == tuple(record.reservation_id for record in records)
    assert len(reconciliations) == 1
    assert reconciliations[0].payload["reservation_id"] == records[0].reservation_id
    streamed_reconciliations = [
        event for event in events if event.type == EventType.BUDGET_RECONCILED
    ]
    assert [
        _private_record_for_public_event(stored_records, event).event.id
        for event in streamed_reconciliations
    ] == [reconciliations[0].id]
    assert EventType.SESSION_FAILED not in [event.type for event in events]


def test_cayu_app_retries_only_unsettled_limit_after_partial_reconciliation() -> None:
    class TransientlyFailingLedger(InMemoryBudgetLedger):
        def __init__(self) -> None:
            super().__init__()
            self.reconcile_calls = 0

        async def reconcile(self, **kwargs):
            self.reconcile_calls += 1
            if self.reconcile_calls == 2:
                raise RuntimeError("second reconciliation failed once")
            return await super().reconcile(**kwargs)

    class RetriedBudgetProvider(ModelProvider):
        name = "fake"

        def __init__(self) -> None:
            self.requests: list[ModelRequest] = []

        async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
            self.requests.append(request)
            raise TimeoutError("authoritative provider timeout")
            yield  # pragma: no cover

    async def run():
        session_id = "sess_budget_partial_reconciliation_retry"
        store = InMemorySessionStore()
        ledger = TransientlyFailingLedger()
        provider = RetriedBudgetProvider()
        reservation = BudgetReservation(max_input_tokens=1_000_000, max_output_tokens=0)
        pricing = fake_budget_limit("10").pricing
        app = CayuApp(
            session_store=store,
            config=CayuConfig(
                run=RunDefaults(retry_policy=RetryPolicy(max_attempts=2, initial_delay_s=0.0))
            ),
            budget_policy=BudgetPolicy(
                limits=(
                    BudgetLimit(
                        scope="app",
                        max_estimated_cost=Decimal("2"),
                        pricing=pricing,
                        reservation=reservation,
                    ),
                    BudgetLimit(
                        scope="agent",
                        key="assistant",
                        max_estimated_cost=Decimal("2"),
                        pricing=pricing,
                        reservation=reservation,
                    ),
                )
            ),
            budget_ledger=ledger,
        )
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))

        events = await collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "hello")],
            ),
        )
        return (
            events,
            await store.query_events(EventQuery(session_id=session_id, limit=100)),
            tuple(ledger._records.values()),
            provider.requests,
            ledger.reconcile_calls,
        )

    events, stored_records, records, provider_requests, reconcile_calls = asyncio.run(run())
    stored_events = [record.event for record in stored_records]

    reconciliations = [
        event for event in stored_events if event.type == EventType.BUDGET_RECONCILED
    ]
    assert len(provider_requests) == 1
    assert reconcile_calls == 3
    assert [record.status for record in records] == ["reconciled", "reconciled"]
    assert len(reconciliations) == 2
    assert {event.payload["reservation_id"] for event in reconciliations} == {
        record.reservation_id for record in records
    }
    streamed_reconciliations = [
        event for event in events if event.type == EventType.BUDGET_RECONCILED
    ]
    assert [
        _private_record_for_public_event(stored_records, event).event.id
        for event in streamed_reconciliations
    ] == [event.id for event in reconciliations]
    assert events[-1].type == EventType.SESSION_FAILED
    assert events[-1].payload["error"] == "authoritative provider timeout"


def test_cayu_app_does_not_dispatch_retry_without_an_additional_reservation() -> None:
    class RetriedBudgetProvider(ModelProvider):
        name = "fake"

        def __init__(self) -> None:
            self.requests: list[ModelRequest] = []
            self.count_requests: list[ModelRequest] = []

        async def count_input_tokens(
            self,
            request: ModelRequest,
        ) -> InputTokenCountResult:
            self.count_requests.append(request)
            return InputTokenCountResult(
                input_tokens=1,
                method=InputTokenCountMethod.OFFICIAL,
                confidence=InputTokenCountConfidence.HIGH,
            )

        async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
            self.requests.append(request)
            raise TimeoutError("stream idle timeout")
            yield  # pragma: no cover

    provider = RetriedBudgetProvider()
    app = CayuApp(
        config=CayuConfig(
            run=RunDefaults(retry_policy=RetryPolicy(max_attempts=2, initial_delay_s=0.0))
        ),
        budget_policy=BudgetPolicy(
            limits=(
                BudgetLimit(
                    scope="app",
                    max_estimated_cost=Decimal("1"),
                    pricing=fake_budget_limit("10").pricing,
                    reservation=BudgetReservation(
                        max_input_tokens=1_000_000,
                        max_output_tokens=0,
                    ),
                ),
            )
        ),
        context_counting=ContextCountingConfig(mode=ContextCountingMode.OBSERVE),
    )
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_budget_retry_reservation_denied",
                messages=[Message.text("user", "hello")],
            ),
        )
    )

    event_types = [event.type for event in events]
    assert len(provider.requests) == 1
    assert len(provider.count_requests) == 1
    assert event_types.count(EventType.MODEL_STARTED) == 1
    assert EventType.BUDGET_RESERVATION_FAILED in event_types
    assert EventType.BUDGET_RECONCILED in event_types
    assert events[-1].type == EventType.SESSION_INTERRUPTED


def test_cayu_app_pauses_budget_heartbeat_during_retry_reconciliation() -> None:
    class SlowReconciliationLedger(InMemoryBudgetLedger):
        def __init__(self) -> None:
            super().__init__(reservation_ttl_seconds=1)

        async def reconcile(self, **kwargs):
            reconciliation = await super().reconcile(**kwargs)
            await asyncio.sleep(0.5)
            return reconciliation

    class RetriedBudgetProvider(ModelProvider):
        name = "fake"

        def __init__(self) -> None:
            self.requests: list[ModelRequest] = []

        async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
            self.requests.append(request)
            if len(self.requests) == 1:
                raise TimeoutError("stream idle timeout")
            yield ModelStreamEvent.completed(
                {
                    "finish_reason": "stop",
                    "usage": {
                        "input_tokens": 250_000,
                        "output_tokens": 0,
                        "total_tokens": 250_000,
                    },
                }
            )

    provider = RetriedBudgetProvider()
    app = CayuApp(
        config=CayuConfig(
            run=RunDefaults(retry_policy=RetryPolicy(max_attempts=2, initial_delay_s=0.0))
        ),
        budget_policy=BudgetPolicy(
            limits=(
                BudgetLimit(
                    scope="app",
                    max_estimated_cost=Decimal("2"),
                    pricing=fake_budget_limit("10").pricing,
                    reservation=BudgetReservation(
                        max_input_tokens=1_000_000,
                        max_output_tokens=0,
                    ),
                ),
            )
        ),
        budget_ledger=SlowReconciliationLedger(),
    )
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_budget_retry_slow_reconciliation",
                messages=[Message.text("user", "hello")],
            ),
        )
    )

    assert len(provider.requests) == 2
    assert events[-1].type == EventType.SESSION_COMPLETED
    assert len([event for event in events if event.type == EventType.BUDGET_RECONCILED]) == 2


def test_cayu_app_budget_settlement_failure_retains_model_recovery_authority() -> None:
    class FailingReconciliationLedger(InMemoryBudgetLedger):
        async def reconcile(self, **_kwargs):
            raise RuntimeError("budget ledger unavailable")

    session_id = "sess_budget_settlement_failed"
    store = InMemorySessionStore()
    ledger = FailingReconciliationLedger(reservation_ttl_seconds=None)
    provider = FakeProvider([ModelStreamEvent.error("authoritative provider failure")])
    app = CayuApp(
        session_store=store,
        budget_policy=BudgetPolicy(
            limits=(
                BudgetLimit(
                    scope="app",
                    max_estimated_cost=Decimal("1"),
                    pricing=fake_budget_limit("10").pricing,
                    reservation=BudgetReservation(
                        max_input_tokens=1_000_000,
                        max_output_tokens=0,
                    ),
                ),
            )
        ),
        budget_ledger=ledger,
    )
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    async def run():
        events = []
        with pytest.raises(RuntimeError, match="authoritative provider failure") as exc_info:
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "hello")],
                )
            ):
                events.append(event)
        return (
            events,
            exc_info.value,
            await store.load(session_id),
            await store.load_active_model_completion_stage(session_id),
        )

    events, failure, session, active_stage = asyncio.run(run())

    assert EventType.SESSION_FAILED not in [event.type for event in events]
    assert session is not None and session.status == SessionStatus.RUNNING
    assert active_stage is not None
    record = next(iter(ledger._records.values()))
    assert active_stage.stage.reservation_ids == (record.reservation_id,)
    assert record.status == "active"
    assert record.dispatch_id == active_stage.stage.stage_id
    assert asyncio.run(ledger.load_settlement(budget_settlement_id(record.reservation_id))) is None
    assert failure.__notes__ == [
        "Budget settlement also failed: RuntimeError: budget ledger unavailable"
    ]


def test_cayu_app_budget_settlement_failure_does_not_mask_cancellation() -> None:
    class FailingReconciliationLedger(InMemoryBudgetLedger):
        async def reconcile(self, **_kwargs):
            raise RuntimeError("budget ledger unavailable")

    async def run() -> None:
        session_id = "sess_budget_cancel_settlement_failed"
        store = InMemorySessionStore()
        ledger = FailingReconciliationLedger(reservation_ttl_seconds=None)
        provider = BlockingBudgetProvider()
        app = CayuApp(
            session_store=store,
            budget_policy=BudgetPolicy(
                limits=(
                    BudgetLimit(
                        scope="app",
                        max_estimated_cost=Decimal("1"),
                        pricing=fake_budget_limit("10").pricing,
                        reservation=BudgetReservation(
                            max_input_tokens=1_000_000,
                            max_output_tokens=0,
                        ),
                    ),
                )
            ),
            budget_ledger=ledger,
        )
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))

        task = asyncio.create_task(
            collect_events(
                app,
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "hello")],
                ),
            )
        )
        await asyncio.wait_for(provider.started.wait(), timeout=10)
        task.cancel()
        with pytest.raises(asyncio.CancelledError) as exc_info:
            await task

        assert provider.cancelled.is_set()
        assert exc_info.value.args == ("Provider operation cancelled",)
        assert type(exc_info.value.__cause__) is RuntimeError
        assert str(exc_info.value.__cause__) == (
            "Session interruption finalization failed after provider cancellation."
        )
        assert exc_info.value.__context__ is None
        session = await store.load(session_id)
        active_stage = await store.load_active_model_completion_stage(session_id)
        assert session is not None and session.status == SessionStatus.RUNNING
        assert active_stage is not None
        record = next(iter(ledger._records.values()))
        assert active_stage.stage.reservation_ids == (record.reservation_id,)
        assert record.status == "active"
        assert await ledger.load_settlement(budget_settlement_id(record.reservation_id)) is None

    asyncio.run(run())


def test_cayu_app_cancellation_during_provider_failure_settlement_propagates() -> None:
    class BlockingReconciliationLedger(InMemoryBudgetLedger):
        def __init__(self) -> None:
            super().__init__()
            self.reconciliation_started = asyncio.Event()

        async def reconcile(self, **_kwargs):
            self.reconciliation_started.set()
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

    async def run() -> None:
        ledger = BlockingReconciliationLedger()
        provider = FakeProvider([ModelStreamEvent.error("provider failed first")])
        app = CayuApp(
            budget_policy=BudgetPolicy(
                limits=(
                    BudgetLimit(
                        scope="app",
                        max_estimated_cost=Decimal("1"),
                        pricing=fake_budget_limit("10").pricing,
                        reservation=BudgetReservation(
                            max_input_tokens=1_000_000,
                            max_output_tokens=0,
                        ),
                    ),
                )
            ),
            budget_ledger=ledger,
        )
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))

        task = asyncio.create_task(
            collect_events(
                app,
                RunRequest(
                    agent_name="assistant",
                    session_id="sess_budget_settlement_cancelled",
                    messages=[Message.text("user", "hello")],
                ),
            )
        )
        await asyncio.wait_for(ledger.reconciliation_started.wait(), timeout=10)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(run())

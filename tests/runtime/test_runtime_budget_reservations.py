"""Budget reservations contracts."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

import pytest
from tests.core._execution_unit_fixtures import model_attempt_identity
from tests.runtime.runtime_budget_support import BlockingBudgetProvider, fake_budget_limit
from tests.runtime.runtime_provider_support import FakeProvider
from tests.runtime.runtime_session_support import (
    collect_events,
    collect_fork_events,
    collect_resume_events,
)

import cayu.runtime._run_limits as run_limits_module
from cayu.agents import AgentSpec
from cayu.applications import CayuApp
from cayu.budgets.base import BudgetLimit, BudgetPolicy, BudgetReservation, InMemoryBudgetLedger
from cayu.budgets.pricing import (
    ModelPrice,
    PriceBook,
    PriceSchedule,
    PriceTier,
    Provenance,
    TieredPricing,
)
from cayu.context.counting import ContextCountingConfig, ContextCountingMode
from cayu.events import Event, EventType
from cayu.messages import Message
from cayu.providers.base import ModelRequest, ModelStreamEvent
from cayu.runtime._event_projection import PRIVATE_EVENT_AUTHORITY
from cayu.sessions.memory import InMemorySessionStore
from cayu.sessions.records import SessionStatus
from cayu.sessions.requests import ForkSessionRequest, ResumeRequest, RunRequest
from cayu.vaults.redaction import SecretRedactor


class CancellationCleanupFailingBudgetProvider(BlockingBudgetProvider):
    def __init__(self, cleanup_message: str = "provider cancellation cleanup failed") -> None:
        super().__init__()
        self.cleanup_message = cleanup_message

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        self.requests.append(request)
        self.started.set()
        try:
            await self.release.wait()
        except asyncio.CancelledError:
            self.cancelled.set()
            raise RuntimeError(self.cleanup_message) from None
        yield ModelStreamEvent.completed({"finish_reason": "stop"})


class RejectingHeartbeatBudgetLedger(InMemoryBudgetLedger):
    def __init__(
        self,
        *,
        successful_heartbeat_calls: int = 2,
        reject_after: asyncio.Event | None = None,
    ) -> None:
        # These tests inject heartbeat rejection; wall-clock expiry must not
        # select a different failure path while a CI worker is descheduled.
        now = datetime.now(UTC)
        super().__init__(clock=lambda: now, reservation_ttl_seconds=1)
        self.successful_heartbeat_calls = successful_heartbeat_calls
        self.reject_after = reject_after
        self.heartbeat_calls = 0
        self.rejected = asyncio.Event()

    async def heartbeat(self, *, reservation_id: str) -> bool:
        self.heartbeat_calls += 1
        if self.heartbeat_calls <= self.successful_heartbeat_calls or (
            self.reject_after is not None and not self.reject_after.is_set()
        ):
            return await super().heartbeat(reservation_id=reservation_id)
        self.rejected.set()
        return False


class ObservingHeartbeatBudgetLedger(InMemoryBudgetLedger):
    def __init__(self) -> None:
        self.now = datetime.now(UTC)
        super().__init__(clock=lambda: self.now, reservation_ttl_seconds=1)
        self.heartbeat_calls = 0
        self.renewed_past_initial_ttl = asyncio.Event()

    async def heartbeat(self, *, reservation_id: str) -> bool:
        # Exercise the real heartbeat task across more than one lease lifetime
        # without making an unrelated loaded-runner pause expire the lease.
        self.now += timedelta(milliseconds=250)
        renewed = await super().heartbeat(reservation_id=reservation_id)
        assert renewed
        self.heartbeat_calls += 1
        if self.heartbeat_calls >= 6:
            self.renewed_past_initial_ttl.set()
        return renewed


def test_cayu_app_budget_reservation_reconciles_model_step():
    store = InMemorySessionStore()
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
            )
        ]
    )
    app = CayuApp(
        session_store=store,
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
        budget_ledger=InMemoryBudgetLedger(),
    )
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_budget_reservation",
                messages=[Message.text("user", "hello")],
            ),
        )
    )

    assert [event.type for event in events] == [
        EventType.SESSION_STARTED,
        EventType.BUDGET_CHECKED,
        EventType.BUDGET_RESERVED,
        EventType.MODEL_STARTED,
        EventType.MODEL_COMPLETED,
        EventType.BUDGET_RECONCILED,
        EventType.BUDGET_CHECKED,
        EventType.TURN_COMPLETED,
        EventType.SESSION_COMPLETED,
    ]
    reserved = next(event for event in events if event.type == EventType.BUDGET_RESERVED)
    reconciled = next(event for event in events if event.type == EventType.BUDGET_RECONCILED)
    assert reserved.payload["requested"] == "1"
    assert reserved.payload["actual"] == "1"
    assert reconciled.payload["actual_amount"] == "0.25"
    assert reconciled.payload["released_amount"] == "0.75"
    assert reconciled.payload["pricing"] == {
        "provider_name": "fake",
        "model": "fake-model",
        "match": "prefix",
        "provenance": {
            "source": "application",
            "url": "application://price-book",
            "as_of": "unspecified",
        },
        "effective_from": None,
        "effective_through": None,
        "tier_max_input_tokens": None,
    }


def test_cayu_app_heartbeats_silent_live_budget_reservation() -> None:
    async def run():
        provider = BlockingBudgetProvider()
        ledger = ObservingHeartbeatBudgetLedger()
        limit = BudgetLimit(
            scope="app",
            max_estimated_cost=Decimal("1"),
            pricing=fake_budget_limit("10").pricing,
            reservation=BudgetReservation(
                max_input_tokens=1_000_000,
                max_output_tokens=0,
            ),
        )
        app = CayuApp(
            budget_policy=BudgetPolicy(limits=(limit,)),
            budget_ledger=ledger,
        )
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))

        run_task = asyncio.create_task(
            collect_events(
                app,
                RunRequest(
                    agent_name="assistant",
                    session_id="sess_live_reservation",
                    messages=[Message.text("user", "hello")],
                ),
            )
        )
        await provider.started.wait()
        await asyncio.wait_for(ledger.renewed_past_initial_ttl.wait(), timeout=30)
        blocked = await ledger.reserve(
            limit=limit,
            session_id="sess_concurrent",
            agent_name="assistant",
            provider_name="fake",
            model="fake-model",
            model_attempt_identity=model_attempt_identity(),
        )
        provider.release.set()
        events = await run_task
        return blocked, events

    blocked, events = asyncio.run(run())

    assert blocked.accepted is False
    assert events[-1].type == EventType.SESSION_COMPLETED
    assert EventType.BUDGET_RECONCILED in [event.type for event in events]


def test_cayu_app_does_not_start_model_after_reservation_expires_during_event_pause() -> None:
    async def run():
        provider = BlockingBudgetProvider()
        now = [datetime.now(UTC)]
        ledger = InMemoryBudgetLedger(
            clock=lambda: now[0],
            reservation_ttl_seconds=1,
        )
        limit = BudgetLimit(
            scope="app",
            max_estimated_cost=Decimal("1"),
            pricing=fake_budget_limit("10").pricing,
            reservation=BudgetReservation(
                max_input_tokens=1_000_000,
                max_output_tokens=0,
            ),
        )
        app = CayuApp(
            budget_policy=BudgetPolicy(limits=(limit,)),
            budget_ledger=ledger,
        )
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))

        stream = app.run(
            RunRequest(
                agent_name="assistant",
                session_id="sess_expired_before_model",
                messages=[Message.text("user", "hello")],
            )
        )
        events: list[Event] = []
        async for event in stream:
            events.append(event)
            if event.type == EventType.BUDGET_RESERVED:
                break

        assert provider.started.is_set() is False
        now[0] += timedelta(seconds=2)
        replacement = await ledger.reserve(
            limit=limit,
            session_id="sess_replacement",
            agent_name="assistant",
            provider_name="fake",
            model="fake-model",
            model_attempt_identity=model_attempt_identity(),
        )
        async for event in stream:
            events.append(event)
        return provider, replacement, events

    provider, replacement, events = asyncio.run(run())

    assert replacement.accepted is True
    assert provider.started.is_set() is False
    assert EventType.BUDGET_RESERVATION_RELEASED in [event.type for event in events]
    assert events[-1].type == EventType.SESSION_FAILED


@pytest.mark.parametrize(
    "pause_at",
    [
        pytest.param(EventType.MODEL_STARTED, id="model-started"),
        pytest.param(EventType.CONTEXT_PRESSURE_ESTIMATED, id="context-pressure"),
    ],
)
def test_cayu_app_validates_budget_lease_immediately_before_provider_dispatch(
    pause_at: EventType,
) -> None:
    class AbandonmentAcknowledgementLostStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1

        def __init__(self) -> None:
            super().__init__()
            self.abandonment_calls = 0
            self.lost_acknowledgement = False

        async def _abandon_model_completion_stage_atomic(self, prepared):
            self.abandonment_calls += 1
            result = await super()._abandon_model_completion_stage_atomic(prepared)
            if not self.lost_acknowledgement and result.replayed is False:
                self.lost_acknowledgement = True
                raise ConnectionError("stage abandonment acknowledgement lost")
            return result

    async def run():
        provider = BlockingBudgetProvider()
        store = AbandonmentAcknowledgementLostStore()
        session_id = f"sess_expired_at_{pause_at.value}"
        now = [datetime.now(UTC)]
        ledger = InMemoryBudgetLedger(
            clock=lambda: now[0],
            reservation_ttl_seconds=1,
        )
        limit = BudgetLimit(
            scope="app",
            max_estimated_cost=Decimal("1"),
            pricing=fake_budget_limit("10").pricing,
            reservation=BudgetReservation(
                max_input_tokens=1_000_000,
                max_output_tokens=0,
            ),
        )
        app = CayuApp(
            session_store=store,
            budget_policy=BudgetPolicy(limits=(limit,)),
            budget_ledger=ledger,
            context_counting=ContextCountingConfig(mode=ContextCountingMode.OBSERVE),
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
        events: list[Event] = []
        async for event in stream:
            events.append(event)
            if event.type == pause_at:
                break

        assert provider.started.is_set() is False
        now[0] += timedelta(seconds=2)
        replacement = await ledger.reserve(
            limit=limit,
            session_id=f"sess_replacement_for_{pause_at.value}",
            agent_name="assistant",
            provider_name="fake",
            model="fake-model",
            model_attempt_identity=model_attempt_identity(),
        )
        async for event in stream:
            events.append(event)
        return (
            provider,
            replacement,
            events,
            await store.load_active_model_completion_stage(session_id),
            store.abandonment_calls,
            store.lost_acknowledgement,
        )

    (
        provider,
        replacement,
        events,
        active_stage,
        abandonment_calls,
        lost_acknowledgement,
    ) = asyncio.run(run())

    event_types = [event.type for event in events]
    assert replacement.accepted is True
    assert provider.started.is_set() is False
    assert EventType.BUDGET_RESERVATION_RELEASED in event_types
    assert EventType.BUDGET_RECONCILED not in event_types
    assert events[-1].type == EventType.SESSION_FAILED
    assert active_stage is None
    assert abandonment_calls == 2
    assert lost_acknowledgement is True


def test_cayu_app_does_not_advance_provider_when_budget_heartbeat_is_terminal() -> None:
    async def run():
        provider = BlockingBudgetProvider()
        ledger = RejectingHeartbeatBudgetLedger(successful_heartbeat_calls=1)
        limit = BudgetLimit(
            scope="app",
            max_estimated_cost=Decimal("1"),
            pricing=fake_budget_limit("10").pricing,
            reservation=BudgetReservation(
                max_input_tokens=1_000_000,
                max_output_tokens=0,
            ),
        )
        app = CayuApp(
            budget_policy=BudgetPolicy(limits=(limit,)),
            budget_ledger=ledger,
        )
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))

        stream = app.run(
            RunRequest(
                agent_name="assistant",
                session_id="sess_terminal_heartbeat_before_dispatch",
                messages=[Message.text("user", "hello")],
            )
        )
        events: list[Event] = []
        async for event in stream:
            events.append(event)
            if event.type == EventType.MODEL_STARTED:
                break

        assert provider.started.is_set() is False
        await asyncio.wait_for(ledger.rejected.wait(), timeout=5)
        async for event in stream:
            events.append(event)
        return provider, ledger, events

    provider, ledger, events = asyncio.run(run())

    event_types = [event.type for event in events]
    assert ledger.heartbeat_calls == 2
    assert provider.started.is_set() is False
    assert EventType.BUDGET_RESERVATION_RELEASED in event_types
    assert EventType.BUDGET_RECONCILED not in event_types
    assert events[-1].type == EventType.SESSION_FAILED


def test_cayu_app_reconciles_all_limits_before_yielding_reconciliation_events() -> None:
    async def run():
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
                )
            ]
        )
        now = [datetime.now(UTC)]
        ledger = InMemoryBudgetLedger(
            clock=lambda: now[0],
            reservation_ttl_seconds=1,
        )
        reservation = BudgetReservation(
            max_input_tokens=1_000_000,
            max_output_tokens=0,
        )
        pricing = fake_budget_limit("10").pricing
        app_limit = BudgetLimit(
            scope="app",
            max_estimated_cost=Decimal("1"),
            pricing=pricing,
            reservation=reservation,
        )
        agent_limit = BudgetLimit(
            scope="agent",
            key="assistant",
            max_estimated_cost=Decimal("1"),
            pricing=pricing,
            reservation=reservation,
        )
        app = CayuApp(
            budget_policy=BudgetPolicy(limits=(app_limit, agent_limit)),
            budget_ledger=ledger,
        )
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))

        stream = app.run(
            RunRequest(
                agent_name="assistant",
                session_id="sess_multi_limit_reconciliation",
                messages=[Message.text("user", "hello")],
            )
        )
        events: list[Event] = []
        async for event in stream:
            events.append(event)
            if event.type == EventType.BUDGET_RECONCILED:
                break

        now[0] += timedelta(seconds=2)
        blocked = await ledger.reserve(
            limit=agent_limit,
            session_id="sess_concurrent",
            agent_name="assistant",
            provider_name="fake",
            model="fake-model",
            model_attempt_identity=model_attempt_identity(),
        )
        async for event in stream:
            events.append(event)
        return blocked, events

    blocked, events = asyncio.run(run())

    assert blocked.accepted is False
    assert blocked.actual == Decimal("1.25")
    assert events[-1].type == EventType.SESSION_COMPLETED
    assert [event.type for event in events].count(EventType.BUDGET_RECONCILED) == 2


def test_cayu_app_fails_closed_and_charges_reservation_when_heartbeat_is_lost() -> None:
    async def run():
        provider = BlockingBudgetProvider()
        ledger = RejectingHeartbeatBudgetLedger(reject_after=provider.started)
        limit = BudgetLimit(
            scope="app",
            max_estimated_cost=Decimal("1"),
            pricing=fake_budget_limit("10").pricing,
            reservation=BudgetReservation(
                max_input_tokens=1_000_000,
                max_output_tokens=0,
            ),
        )
        app = CayuApp(
            budget_policy=BudgetPolicy(limits=(limit,)),
            budget_ledger=ledger,
        )
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))

        events = await asyncio.wait_for(
            collect_events(
                app,
                RunRequest(
                    agent_name="assistant",
                    session_id="sess_lost_reservation",
                    messages=[Message.text("user", "hello")],
                ),
            ),
            timeout=30,
        )
        return provider, ledger, events

    provider, ledger, events = asyncio.run(run())

    assert provider.started.is_set()
    assert provider.cancelled.is_set()
    assert ledger.heartbeat_calls >= 3
    assert ledger.rejected.is_set()
    assert events[-1].type == EventType.SESSION_FAILED
    assert events[-1].payload["error_type"] == "BudgetReservationLeaseLost"
    assert EventType.BUDGET_RESERVATION_RELEASED not in [event.type for event in events]
    reconciliation = next(event for event in events if event.type == EventType.BUDGET_RECONCILED)
    assert reconciliation.payload["actual_amount"] == "1"
    assert reconciliation.payload["reason"] == (
        "reservation heartbeat lost; charged reserved amount"
    )


def test_cayu_app_keeps_lease_loss_authoritative_when_provider_cleanup_fails() -> None:
    async def run():
        cleanup_secret = "provider-cleanup-secret-canary"
        provider = CancellationCleanupFailingBudgetProvider(
            f"provider cancellation cleanup failed: {cleanup_secret}"
        )
        ledger = RejectingHeartbeatBudgetLedger(reject_after=provider.started)
        limit = BudgetLimit(
            scope="app",
            max_estimated_cost=Decimal("1"),
            pricing=fake_budget_limit("10").pricing,
            reservation=BudgetReservation(
                max_input_tokens=1_000_000,
                max_output_tokens=0,
            ),
        )
        app = CayuApp(
            budget_policy=BudgetPolicy(limits=(limit,)),
            budget_ledger=ledger,
            secret_redactor=SecretRedactor(cleanup_secret),
        )
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))

        unhandled: list[dict[str, Any]] = []
        loop = asyncio.get_running_loop()
        previous_handler = loop.get_exception_handler()
        loop.set_exception_handler(lambda _loop, context: unhandled.append(context))
        try:
            events = await collect_events(
                app,
                RunRequest(
                    agent_name="assistant",
                    session_id="sess_provider_cleanup_failed_after_lease_loss",
                    messages=[Message.text("user", "hello")],
                ),
            )
            await asyncio.sleep(0)
        finally:
            loop.set_exception_handler(previous_handler)
        return provider, ledger, events, unhandled

    provider, ledger, events, unhandled = asyncio.run(run())

    event_types = [event.type for event in events]
    assert provider.started.is_set()
    assert provider.cancelled.is_set()
    assert ledger.heartbeat_calls >= 3
    assert ledger.rejected.is_set()
    assert events[-1].type == EventType.SESSION_FAILED
    assert events[-1].payload["error_type"] == "BudgetReservationLeaseLost"
    assert events[-1].payload["error"].startswith("Budget reservation lease was lost:")
    assert events[-1].payload["provider_cleanup_failure"] == {
        "error": "Provider stream cleanup did not complete normally.",
        "error_type": "ProviderIteratorCleanupError",
        "phase": "provider_iterator_cleanup",
    }
    assert "provider-cleanup-secret-canary" not in repr(events[-1].payload)
    assert event_types.count(EventType.BUDGET_RECONCILED) == 1
    assert EventType.BUDGET_RESERVATION_RELEASED not in event_types
    reconciliation = next(event for event in events if event.type == EventType.BUDGET_RECONCILED)
    assert reconciliation.payload["actual_amount"] == "1"
    assert reconciliation.payload["reason"] == (
        "reservation heartbeat lost; charged reserved amount"
    )
    assert unhandled == []


def test_cayu_app_preserves_completed_usage_when_heartbeat_fails_in_same_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_wait = asyncio.wait

    async def force_terminal_item_and_heartbeat_into_same_done_set(
        tasks,
        *,
        return_when=asyncio.ALL_COMPLETED,
        timeout=None,
    ):
        done, pending = await real_wait(tasks, return_when=return_when, timeout=timeout)
        for task in done:
            if task.cancelled() or task.exception() is not None:
                continue
            result = task.result()
            if (
                isinstance(result, tuple)
                and result
                and isinstance(result[0], Event)
                and result[0].type == EventType.MODEL_COMPLETED
            ):
                await asyncio.gather(*pending, return_exceptions=True)
                return set(tasks), set()
        return done, pending

    monkeypatch.setattr(
        run_limits_module.asyncio,
        "wait",
        force_terminal_item_and_heartbeat_into_same_done_set,
    )

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
            )
        ]
    )
    ledger = RejectingHeartbeatBudgetLedger()
    limit = BudgetLimit(
        scope="app",
        max_estimated_cost=Decimal("1"),
        pricing=fake_budget_limit("10").pricing,
        reservation=BudgetReservation(
            max_input_tokens=1_000_000,
            max_output_tokens=0,
        ),
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
                session_id="sess_same_turn_completion_and_lease_loss",
                messages=[Message.text("user", "hello")],
            ),
        )
    )

    assert EventType.MODEL_COMPLETED in [event.type for event in events]
    assert events[-1].type == EventType.SESSION_FAILED
    reconciliations = [event for event in events if event.type == EventType.BUDGET_RECONCILED]
    assert len(reconciliations) == 1
    assert reconciliations[0].payload["actual_amount"] == "0.25"
    assert reconciliations[0].payload["reason"] == "model completed"
    assert EventType.BUDGET_RESERVATION_RELEASED not in [event.type for event in events]


def test_cayu_app_tiered_causal_reservation_counts_cached_input_for_tier() -> None:
    class AnthropicBudgetProvider(FakeProvider):
        name = "anthropic"

    provider = AnthropicBudgetProvider(
        [
            ModelStreamEvent.completed(
                {
                    "finish_reason": "stop",
                    "usage": {
                        "input_tokens": 100_000,
                        "output_tokens": 0,
                        "cache_read_input_tokens": 150_000,
                    },
                }
            )
        ]
    )
    pricing = PriceBook(
        price_book_version="test",
        generated_at="2026-01-01",
        prices=(
            ModelPrice(
                provider_name="anthropic",
                model="claude-test",
                schedules=(
                    PriceSchedule(
                        pricing=TieredPricing(
                            standard=(
                                PriceTier(
                                    max_input_tokens=200_000,
                                    input_per_million=Decimal("1"),
                                    output_per_million=Decimal("2"),
                                    cache_read_input_per_million=Decimal("0.1"),
                                ),
                                PriceTier(
                                    max_input_tokens=None,
                                    input_per_million=Decimal("10"),
                                    output_per_million=Decimal("20"),
                                    cache_read_input_per_million=Decimal("1"),
                                ),
                            )
                        ),
                        provenance=Provenance(
                            source="official",
                            url="https://example.test/pricing",
                            as_of="2026-01-01",
                        ),
                    ),
                ),
            ),
        ),
    )
    app = CayuApp(
        session_store=InMemorySessionStore(),
        budget_policy=BudgetPolicy(
            limits=(
                BudgetLimit(
                    scope="causal",
                    key="job-tiered",
                    max_estimated_cost=Decimal("2"),
                    pricing=pricing,
                    reservation=BudgetReservation(
                        max_input_tokens=100_000,
                        max_output_tokens=0,
                        max_cache_read_input_tokens=150_000,
                    ),
                ),
            )
        ),
        budget_ledger=InMemoryBudgetLedger(),
    )
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="claude-test-2026-07-01"))

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_tiered_cache_reservation",
                causal_budget_id="job-tiered",
                messages=[Message.text("user", "hello")],
            ),
        )
    )

    reserved = next(event for event in events if event.type == EventType.BUDGET_RESERVED)
    completed = next(event for event in events if event.type == EventType.MODEL_COMPLETED)
    reconciled = next(event for event in events if event.type == EventType.BUDGET_RECONCILED)
    assert reserved.payload["requested"] == "1.15"
    assert completed.payload["usage_metrics"]["input_tokens"] == 250_000
    assert reconciled.payload["actual_amount"] == "1.15"
    assert reconciled.payload["released_amount"] == "0"


def test_cayu_app_budget_reservation_stops_before_provider_when_capacity_is_unavailable():
    store = InMemorySessionStore()
    provider = FakeProvider(
        [
            ModelStreamEvent.completed(
                {
                    "finish_reason": "stop",
                    "usage": {
                        "input_tokens": 750_000,
                        "output_tokens": 0,
                        "total_tokens": 750_000,
                    },
                }
            )
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

    first_events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_budget_reservation_first",
                messages=[Message.text("user", "first")],
            ),
        )
    )
    second_events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_budget_reservation_second",
                messages=[Message.text("user", "second")],
            ),
        )
    )

    assert first_events[-1].type == EventType.SESSION_COMPLETED
    assert [event.type for event in second_events] == [
        EventType.SESSION_STARTED,
        EventType.BUDGET_CHECKED,
        EventType.BUDGET_RESERVATION_FAILED,
        EventType.BUDGET_LIMIT_REACHED,
        EventType.SESSION_LIMIT_REACHED,
        EventType.TURN_COMPLETED,
        EventType.SESSION_INTERRUPTED,
    ]
    failed = next(
        event for event in second_events if event.type == EventType.BUDGET_RESERVATION_FAILED
    )
    assert failed.payload["accepted"] is False
    assert failed.payload["requested"] == "1"
    assert failed.payload["actual"] == "1.75"
    assert len(provider.requests) == 1


def test_cayu_app_request_budget_reservation_reconciles_model_step():
    store = InMemorySessionStore()
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
            )
        ]
    )
    app = CayuApp(session_store=store, budget_ledger=InMemoryBudgetLedger())
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_request_budget_reservation",
                messages=[Message.text("user", "hello")],
                budget_limits=(
                    BudgetLimit(
                        scope="app",
                        max_estimated_cost=Decimal("2"),
                        pricing=fake_budget_limit("10").pricing,
                        reservation=BudgetReservation(
                            max_input_tokens=1_000_000,
                            max_output_tokens=0,
                        ),
                    ),
                ),
            ),
        )
    )

    assert events[-1].type == EventType.SESSION_COMPLETED
    reserved = next(event for event in events if event.type == EventType.BUDGET_RESERVED)
    reconciled = next(event for event in events if event.type == EventType.BUDGET_RECONCILED)
    assert reserved.payload["requested"] == "1"
    assert reserved.payload["actual"] == "1"
    assert reconciled.payload["actual_amount"] == "0.25"
    assert reconciled.payload["released_amount"] == "0.75"


def test_cayu_app_request_budget_reservation_stops_second_session_sharing_the_budget():
    store = InMemorySessionStore()
    provider = FakeProvider(
        [
            ModelStreamEvent.completed(
                {
                    "finish_reason": "stop",
                    "usage": {
                        "input_tokens": 750_000,
                        "output_tokens": 0,
                        "total_tokens": 750_000,
                    },
                }
            )
        ]
    )
    app = CayuApp(session_store=store, budget_ledger=InMemoryBudgetLedger())
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))
    shared_limit = BudgetLimit(
        scope="agent",
        key="assistant",
        max_estimated_cost=Decimal("1"),
        pricing=fake_budget_limit("10").pricing,
        reservation=BudgetReservation(
            max_input_tokens=1_000_000,
            max_output_tokens=0,
        ),
    )

    first_events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_request_reservation_first",
                messages=[Message.text("user", "first")],
                budget_limits=(shared_limit,),
            ),
        )
    )
    second_events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_request_reservation_second",
                messages=[Message.text("user", "second")],
                budget_limits=(shared_limit,),
            ),
        )
    )

    assert first_events[-1].type == EventType.SESSION_COMPLETED
    assert second_events[-1].type == EventType.SESSION_INTERRUPTED
    failed = next(
        event for event in second_events if event.type == EventType.BUDGET_RESERVATION_FAILED
    )
    assert failed.payload["accepted"] is False
    assert failed.payload["requested"] == "1"
    assert failed.payload["actual"] == "1.75"
    assert len(provider.requests) == 1


def test_cayu_app_causal_budget_is_shared_by_forked_sessions():
    store = InMemorySessionStore()
    provider = FakeProvider(
        [
            ModelStreamEvent.completed(
                {
                    "finish_reason": "stop",
                    "usage": {
                        "input_tokens": 750_000,
                        "output_tokens": 0,
                        "total_tokens": 750_000,
                    },
                }
            )
        ]
    )
    app = CayuApp(
        session_store=store,
        budget_policy=BudgetPolicy(
            limits=(
                BudgetLimit(
                    scope="causal",
                    key="job_shared",
                    max_estimated_cost=Decimal("1"),
                    pricing=fake_budget_limit("10").pricing,
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

    parent_events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_causal_budget_parent",
                causal_budget_id="job_shared",
                messages=[Message.text("user", "first")],
            ),
        )
    )
    fork_events = asyncio.run(
        collect_fork_events(
            app,
            ForkSessionRequest(
                source_session_id="sess_causal_budget_parent",
                session_id="sess_causal_budget_child",
            ),
        )
    )
    child_events = asyncio.run(
        collect_resume_events(
            app,
            ResumeRequest(
                session_id="sess_causal_budget_child",
                messages=[Message.text("user", "continue")],
            ),
        )
    )
    child_session = asyncio.run(store.load("sess_causal_budget_child"))

    assert parent_events[-1].type == EventType.SESSION_COMPLETED
    assert fork_events[0].type == EventType.SESSION_FORKED
    assert fork_events[0].payload["causal_budget_id"] == PRIVATE_EVENT_AUTHORITY
    assert child_session is not None
    assert child_session.causal_budget_id == "job_shared"
    assert [event.type for event in child_events] == [
        EventType.SESSION_RESUMED,
        EventType.BUDGET_CHECKED,
        EventType.BUDGET_RESERVATION_FAILED,
        EventType.BUDGET_LIMIT_REACHED,
        EventType.SESSION_LIMIT_REACHED,
        EventType.TURN_COMPLETED,
        EventType.SESSION_INTERRUPTED,
    ]
    failed = next(
        event for event in child_events if event.type == EventType.BUDGET_RESERVATION_FAILED
    )
    assert failed.payload["scope"] == "causal"
    assert failed.payload["key"] == "job_shared"
    assert failed.payload["actual"] == "1.75"
    assert child_session is not None
    assert child_session.causal_budget_id == "job_shared"
    assert child_session.status == SessionStatus.INTERRUPTED
    assert len(provider.requests) == 1


def test_cayu_app_request_causal_budget_limit_applies_to_matching_causal_history():
    store = InMemorySessionStore()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.text_delta("first"),
                ModelStreamEvent.completed(
                    {
                        "finish_reason": "stop",
                        "usage": {
                            "input_tokens": 1_000_000,
                            "output_tokens": 0,
                            "total_tokens": 1_000_000,
                        },
                    }
                ),
            ],
            [
                ModelStreamEvent.text_delta("should not run"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    app = CayuApp(session_store=store)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    first_events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_request_causal_budget_first",
                causal_budget_id="job_request",
                messages=[Message.text("user", "first")],
            ),
        )
    )
    second_events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_request_causal_budget_second",
                causal_budget_id="job_request",
                messages=[Message.text("user", "second")],
                budget_limits=(
                    BudgetLimit(
                        scope="causal",
                        key="job_request",
                        max_estimated_cost=Decimal("1"),
                        pricing=fake_budget_limit("10").pricing,
                    ),
                ),
            ),
        )
    )

    assert first_events[-1].type == EventType.SESSION_COMPLETED
    assert [event.type for event in second_events] == [
        EventType.SESSION_STARTED,
        EventType.SESSION_LIMIT_REACHED,
        EventType.TURN_COMPLETED,
        EventType.SESSION_INTERRUPTED,
    ]
    assert second_events[1].payload["actual"] == "1"
    assert len(provider.requests) == 1

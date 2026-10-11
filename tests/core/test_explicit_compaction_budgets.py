"""Explicit compaction budget admission, reservation settlement and release."""

from __future__ import annotations

import asyncio
from decimal import Decimal
from typing import Any

import pytest
from tests.core._explicit_compaction_fixtures import (
    ContextualBedrockCompactionProvider,
    FinalRenewalFailureBudgetLedger,
    RecordingCompactor,
    UsageCompactionProvider,
    _create_profiled_session,
)

from cayu._validation import MAX_DURABLE_JSON_INTEGER
from cayu.agents import AgentSpec
from cayu.applications import CayuApp
from cayu.budgets.base import BudgetLimit, BudgetPolicy, BudgetReservation, InMemoryBudgetLedger
from cayu.budgets.pricing import ModelPrice, PriceBook
from cayu.budgets.run_limits import RunLimits
from cayu.context.base import (
    CheckpointCompactionContextPolicy,
    CompactionRequest,
    CompactionResult,
    ContextBuildError,
    ContextCompactor,
    ModelCompactor,
    PromptCacheCompactor,
)
from cayu.events import Event, EventType
from cayu.messages import Message
from cayu.providers.base import ModelProvider, ModelRequest, ModelStreamEvent
from cayu.providers.bedrock import bedrock_billing_identity
from cayu.runtime._event_projection import public_event_id, public_event_sequence
from cayu.sessions.event_queries import EventQuery
from cayu.sessions.memory import InMemorySessionStore
from cayu.sessions.records import SessionIdentity, SessionStatus
from cayu.sessions.requests import CompactSessionRequest, RunRequest


async def _private_events_for_public_events(
    store: Any,
    session_id: str,
    events: list[Event],
) -> list[Event]:
    sequences = [public_event_sequence(event.id) for event in events]
    assert all(sequence is not None for sequence in sequences)
    records = await store.query_events(EventQuery(session_id=session_id, limit=5000))
    event_by_sequence = {record.sequence: record.event for record in records}
    return [event_by_sequence[sequence] for sequence in sequences]


class ReservationInspectingCompactionProvider(ModelProvider):
    name = "compaction-provider"

    def __init__(self, store: InMemorySessionStore) -> None:
        self.store = store
        self.session_id: str | None = None
        self.calls = 0
        self.durable_types_before_dispatch: list[EventType] | None = None

    async def stream(self, request: ModelRequest):
        self.calls += 1
        assert self.session_id is not None
        self.durable_types_before_dispatch = [
            record.event.type
            for record in await self.store.query_events(
                EventQuery(session_id=self.session_id, limit=100)
            )
        ]
        yield ModelStreamEvent.text_delta("provider summary")
        yield ModelStreamEvent.completed({"usage": {"input_tokens": 8, "output_tokens": 2}})


class InitialRenewalFailureBudgetLedger(InMemoryBudgetLedger):
    async def heartbeat(self, *, reservation_id: str) -> bool:
        return False


class FailingSecondReservationBudgetLedger(InMemoryBudgetLedger):
    def __init__(self) -> None:
        super().__init__()
        self.reserve_calls = 0
        self.release_calls = 0

    async def reserve(self, **kwargs):
        self.reserve_calls += 1
        if self.reserve_calls == 2:
            raise RuntimeError("simulated reservation store failure")
        return await super().reserve(**kwargs)

    async def release(self, **kwargs):
        self.release_calls += 1
        return await super().release(**kwargs)


class FailingSecondReleaseBudgetLedger(InMemoryBudgetLedger):
    def __init__(self) -> None:
        super().__init__()
        self.release_calls = 0
        self.reservation_ids: list[str] = []

    async def reserve(self, **kwargs):
        result = await super().reserve(**kwargs)
        if result.record is not None:
            self.reservation_ids.append(result.record.reservation_id)
        return result

    async def release(self, **kwargs):
        self.release_calls += 1
        if self.release_calls == 2:
            raise RuntimeError("simulated second release failure")
        return await super().release(**kwargs)


class UndeclaredProviderCompactor(ContextCompactor):
    def __init__(self) -> None:
        self.calls = 0

    async def compact(self, request: CompactionRequest) -> CompactionResult:
        self.calls += 1
        return CompactionResult(
            summary="undeclared provider summary",
            covered_message_count=len(request.messages),
            model_completed_payloads=[
                {
                    "provider_name": "compaction-provider",
                    "model": "summary-model",
                    "usage": {"input_tokens": 8, "output_tokens": 2},
                }
            ],
        )


class UnsafeProviderIdentityCompactor(ContextCompactor):
    def __init__(self, identity: tuple[str, str]) -> None:
        self.identity = identity
        self.calls = 0

    def provider_budget_identity(self, _session) -> tuple[str, str]:
        return self.identity

    async def compact(self, request: CompactionRequest) -> CompactionResult:
        self.calls += 1
        return CompactionResult(
            summary="must not execute", covered_message_count=len(request.messages)
        )


def test_compact_session_admits_each_static_priced_hierarchy_dispatch() -> None:
    async def run() -> None:
        pricing = PriceBook(
            prices=(
                ModelPrice.fixed(
                    provider_name="compaction-provider",
                    model="summary-model",
                    input_per_million=Decimal("1"),
                    output_per_million=Decimal("2"),
                ),
            )
        )
        provider = UsageCompactionProvider()
        ledger = InMemoryBudgetLedger()
        store = InMemorySessionStore()
        app = CayuApp(
            session_store=store,
            budget_policy=BudgetPolicy(
                limits=(
                    BudgetLimit(
                        scope="app",
                        max_estimated_cost=Decimal("0.000035"),
                        pricing=pricing,
                        reservation=BudgetReservation(
                            max_input_tokens=10,
                            max_output_tokens=10,
                        ),
                    ),
                )
            ),
            budget_ledger=ledger,
            enable_logging=False,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=ModelCompactor(
                    provider=provider,
                    model="summary-model",
                    max_input_chars=1000,
                ),
                max_user_turns=1,
            ),
        )
        created = await _create_profiled_session(
            app,
            store,
            RunRequest(
                agent_name="assistant",
                session_id="sess_static_hierarchy_dispatch_budget",
                messages=[Message.text("user", "create only")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        transcript = [
            Message.text("user", "oversized " + "x" * 5000),
            Message.text("user", "current"),
        ]
        await store.append_transcript_messages(created.id, transcript)
        completed = await store.update_status(created.id, SessionStatus.COMPLETED)
        events: list[Event] = []

        with pytest.raises(RuntimeError, match="budget reservation failed"):
            async for event in app.compact_session(
                CompactSessionRequest(
                    session_id=created.id,
                    idempotency_key="static-hierarchy-dispatch-budget",
                    expected_run_epoch=completed.run_epoch,
                    expected_transcript_cursor=len(transcript),
                )
            ):
                events.append(event)

        assert provider.calls == 1
        assert sum(event.type == EventType.BUDGET_RESERVED for event in events) == 1
        assert sum(event.type == EventType.BUDGET_RECONCILED for event in events) == 1
        assert sum(event.type == EventType.BUDGET_RESERVATION_FAILED for event in events) == 1
        completions = [
            event
            for event in events
            if event.type == EventType.MODEL_COMPLETED
            and event.payload.get("purpose") == "context_compaction"
        ]
        assert len(completions) == 1
        assert completions[0].payload["usage_metrics"]["total_tokens"] == 10
        checkpoint = await store.load_checkpoint(created.id)
        assert checkpoint is not None
        assert "context_compaction" not in checkpoint

    asyncio.run(run())


def test_compact_session_admits_each_hierarchy_dispatch_against_run_limits() -> None:
    async def run() -> None:
        provider = UsageCompactionProvider()
        store = InMemorySessionStore()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=ModelCompactor(
                    provider=provider,
                    model="summary-model",
                    max_input_chars=1000,
                ),
                max_user_turns=1,
            ),
        )
        created = await _create_profiled_session(
            app,
            store,
            RunRequest(
                agent_name="assistant",
                session_id="sess_explicit_hierarchy_run_limit",
                messages=[],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        transcript = [
            Message.text("user", "oversized " + "x" * 5000),
            Message.text("user", "current"),
        ]
        await store.append_transcript_messages(created.id, transcript)
        completed = await store.update_status(created.id, SessionStatus.COMPLETED)
        events: list[Event] = []

        with pytest.raises(RuntimeError, match="Compaction limit reached"):
            async for event in app.compact_session(
                CompactSessionRequest(
                    session_id=created.id,
                    idempotency_key="explicit-hierarchy-run-limit",
                    expected_run_epoch=completed.run_epoch,
                    expected_transcript_cursor=len(transcript),
                    limits=RunLimits(max_total_tokens=10),
                )
            ):
                events.append(event)

        assert provider.calls == 1
        completions = [
            event
            for event in events
            if event.type == EventType.MODEL_COMPLETED
            and event.payload.get("purpose") == "context_compaction"
        ]
        assert len(completions) == 1
        assert completions[0].payload["usage_metrics"]["total_tokens"] == 10
        assert sum(event.type == EventType.CONTEXT_COMPACTION_FAILED for event in events) == 1
        checkpoint = await store.load_checkpoint(created.id)
        assert checkpoint is not None
        assert "context_compaction" not in checkpoint

    asyncio.run(run())


def test_compact_session_admits_each_hierarchy_dispatch_against_static_budget() -> None:
    async def run() -> None:
        pricing = PriceBook(
            prices=(
                ModelPrice.fixed(
                    provider_name="compaction-provider",
                    model="summary-model",
                    input_per_million=Decimal("1"),
                    output_per_million=Decimal("1"),
                ),
            )
        )
        provider = UsageCompactionProvider()
        store = InMemorySessionStore()
        app = CayuApp(
            session_store=store,
            budget_policy=BudgetPolicy(
                limits=(
                    BudgetLimit(
                        scope="app",
                        max_estimated_cost=Decimal("0.000010"),
                        pricing=pricing,
                    ),
                )
            ),
            enable_logging=False,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=ModelCompactor(
                    provider=provider,
                    model="summary-model",
                    max_input_chars=1000,
                ),
                max_user_turns=1,
            ),
        )
        created = await _create_profiled_session(
            app,
            store,
            RunRequest(
                agent_name="assistant",
                session_id="sess_explicit_hierarchy_static_budget",
                messages=[],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        transcript = [
            Message.text("user", "oversized " + "x" * 5000),
            Message.text("user", "current"),
        ]
        await store.append_transcript_messages(created.id, transcript)
        completed = await store.update_status(created.id, SessionStatus.COMPLETED)
        events: list[Event] = []

        with pytest.raises(RuntimeError, match="Compaction budget limit reached"):
            async for event in app.compact_session(
                CompactSessionRequest(
                    session_id=created.id,
                    idempotency_key="explicit-hierarchy-static-budget",
                    expected_run_epoch=completed.run_epoch,
                    expected_transcript_cursor=len(transcript),
                )
            ):
                events.append(event)

        assert provider.calls == 1
        assert EventType.BUDGET_RESERVED not in {event.type for event in events}
        assert EventType.BUDGET_LIMIT_REACHED in {event.type for event in events}
        completions = [
            event
            for event in events
            if event.type == EventType.MODEL_COMPLETED
            and event.payload.get("purpose") == "context_compaction"
        ]
        assert len(completions) == 1
        assert completions[0].payload["usage_metrics"]["total_tokens"] == 10
        assert sum(event.type == EventType.CONTEXT_COMPACTION_FAILED for event in events) == 1
        checkpoint = await store.load_checkpoint(created.id)
        assert checkpoint is not None
        assert "context_compaction" not in checkpoint

    asyncio.run(run())


def test_compact_session_contextual_budget_counts_prior_hierarchy_dispatches() -> None:
    async def run() -> None:
        identity = bedrock_billing_identity(
            invoked_model="summary-model",
            source_region="us-east-1",
            resource_type="foundation_model",
        )
        provider = ContextualBedrockCompactionProvider(identity)
        pricing = PriceBook(
            prices=(
                ModelPrice.fixed(
                    provider_name="bedrock",
                    model="summary-model",
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
        store = InMemorySessionStore()
        app = CayuApp(
            session_store=store,
            budget_policy=BudgetPolicy(
                limits=(
                    BudgetLimit(
                        scope="app",
                        max_estimated_cost=Decimal("0.0001"),
                        pricing=pricing,
                    ),
                )
            ),
            enable_logging=False,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=ModelCompactor(
                    provider=provider,
                    model="summary-model",
                    max_input_chars=1000,
                ),
                max_user_turns=1,
            ),
        )
        created = await _create_profiled_session(
            app,
            store,
            RunRequest(
                agent_name="assistant",
                session_id="sess_contextual_hierarchy_dispatch_budget",
                messages=[],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        transcript = [
            Message.text("user", "oversized " + "x" * 5000),
            Message.text("user", "current"),
        ]
        await store.append_transcript_messages(created.id, transcript)
        completed = await store.update_status(created.id, SessionStatus.COMPLETED)
        events: list[Event] = []

        with pytest.raises(RuntimeError, match="budget limit reached"):
            async for event in app.compact_session(
                CompactSessionRequest(
                    session_id=created.id,
                    idempotency_key="contextual-hierarchy-dispatch-budget",
                    expected_run_epoch=completed.run_epoch,
                    expected_transcript_cursor=len(transcript),
                )
            ):
                events.append(event)

        assert provider.calls == 2
        completions = [event for event in events if event.type == EventType.MODEL_COMPLETED]
        assert len(completions) == 2
        assert sum(event.payload["usage_metrics"]["total_tokens"] for event in completions) == 20
        assert EventType.BUDGET_LIMIT_REACHED in {event.type for event in events}
        checkpoint = await store.load_checkpoint(created.id)
        assert checkpoint is not None
        assert "context_compaction" not in checkpoint

    asyncio.run(run())


def test_compact_session_usage_counts_against_supplied_cost_budget() -> None:
    async def run() -> None:
        store = InMemorySessionStore()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=ModelCompactor(
                    provider=UsageCompactionProvider(),
                    model="summary-model",
                ),
                max_user_turns=1,
            ),
        )
        created = await _create_profiled_session(
            app,
            store,
            RunRequest(
                agent_name="assistant",
                session_id="sess_compact_budget",
                messages=[Message.text("user", "create only")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        transcript = [
            Message.text("user", "old request"),
            Message.text("assistant", "old answer"),
            Message.text("user", "current request"),
            Message.text("assistant", "current answer"),
        ]
        await store.append_transcript_messages(created.id, transcript)
        completed = await store.update_status(created.id, SessionStatus.COMPLETED)
        pricing = PriceBook(
            prices=(
                ModelPrice.fixed(
                    provider_name="compaction-provider",
                    model="summary-model",
                    input_per_million=Decimal("1"),
                    output_per_million=Decimal("2"),
                ),
            )
        )
        events = []

        with pytest.raises(RuntimeError, match="budget limit reached"):
            async for event in app.compact_session(
                CompactSessionRequest(
                    session_id=created.id,
                    idempotency_key="compact-budget-1",
                    expected_run_epoch=completed.run_epoch,
                    expected_transcript_cursor=len(transcript),
                    budget_limits=(
                        BudgetLimit(
                            scope="run",
                            max_estimated_cost=Decimal("0.00001"),
                            pricing=pricing,
                        ),
                    ),
                )
            ):
                events.append(event)

        assert [event.type for event in events] == [
            EventType.CONTEXT_COMPACTION_STARTED,
            EventType.REQUEST_FOOTPRINT_RECORDED,
            EventType.MODEL_STARTED,
            EventType.MODEL_COMPLETED,
            EventType.BUDGET_LIMIT_REACHED,
            EventType.CONTEXT_COMPACTION_FAILED,
        ]
        budget_event = events[-2]
        assert budget_event.payload["actual"] == "0.000012"
        assert budget_event.payload["operation_id"]
        assert await store.load_transcript(created.id) == transcript
        checkpoint = await store.load_checkpoint(created.id)
        assert checkpoint is not None
        assert "context_compaction" not in checkpoint

    asyncio.run(run())


def test_compact_session_applies_app_policy_and_reserves_before_provider_work() -> None:
    async def run() -> None:
        pricing = PriceBook(
            prices=(
                ModelPrice.fixed(
                    provider_name="compaction-provider",
                    model="summary-model",
                    input_per_million=Decimal("1"),
                    output_per_million=Decimal("2"),
                ),
            )
        )
        provider = UsageCompactionProvider()
        store = InMemorySessionStore()
        app = CayuApp(
            session_store=store,
            budget_policy=BudgetPolicy(
                limits=(
                    BudgetLimit(
                        scope="app",
                        max_estimated_cost=Decimal("0.00001"),
                        pricing=pricing,
                        reservation=BudgetReservation(
                            max_input_tokens=8,
                            max_output_tokens=2,
                        ),
                    ),
                )
            ),
            enable_logging=False,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=ModelCompactor(provider=provider, model="summary-model"),
                max_user_turns=1,
            ),
        )
        created = await _create_profiled_session(
            app,
            store,
            RunRequest(
                agent_name="assistant",
                session_id="sess_compact_policy_reservation",
                messages=[Message.text("user", "create only")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        transcript = [
            Message.text("user", "old request"),
            Message.text("assistant", "old answer"),
            Message.text("user", "current request"),
            Message.text("assistant", "current answer"),
        ]
        await store.append_transcript_messages(created.id, transcript)
        completed = await store.update_status(created.id, SessionStatus.COMPLETED)
        events = []

        with pytest.raises(RuntimeError, match="reservation failed"):
            async for event in app.compact_session(
                CompactSessionRequest(
                    session_id=created.id,
                    idempotency_key="compact-policy-reservation",
                    expected_run_epoch=completed.run_epoch,
                    expected_transcript_cursor=len(transcript),
                )
            ):
                events.append(event)

        assert [event.type for event in events] == [
            EventType.CONTEXT_COMPACTION_STARTED,
            EventType.BUDGET_CHECKED,
            EventType.BUDGET_RESERVATION_FAILED,
            EventType.BUDGET_LIMIT_REACHED,
            EventType.CONTEXT_COMPACTION_FAILED,
        ]
        assert provider.calls == 0
        assert all(event.payload["operation_id"] for event in events[1:])
        profile_fingerprints = {
            event.payload.get("execution_profile_fingerprint") for event in events
        }
        assert None not in profile_fingerprints
        assert len(profile_fingerprints) == 1

    asyncio.run(run())


def test_compact_session_stops_on_an_exhausted_app_budget_before_provider_work() -> None:
    async def run() -> None:
        pricing = PriceBook(
            prices=(
                ModelPrice.fixed(
                    provider_name="compaction-provider",
                    model="summary-model",
                    input_per_million=Decimal("1"),
                    output_per_million=Decimal("2"),
                ),
            )
        )
        provider = UsageCompactionProvider()
        store = InMemorySessionStore()
        app = CayuApp(
            session_store=store,
            budget_policy=BudgetPolicy(
                limits=(
                    BudgetLimit(
                        scope="app",
                        max_estimated_cost=Decimal("0.00001"),
                        pricing=pricing,
                    ),
                )
            ),
            enable_logging=False,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=ModelCompactor(provider=provider, model="summary-model"),
                max_user_turns=1,
            ),
        )
        created = await _create_profiled_session(
            app,
            store,
            RunRequest(
                agent_name="assistant",
                session_id="sess_compact_exhausted_policy",
                messages=[Message.text("user", "create only")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        transcript = [
            Message.text("user", "old request"),
            Message.text("assistant", "old answer"),
            Message.text("user", "current request"),
            Message.text("assistant", "current answer"),
        ]
        await store.append_transcript_messages(created.id, transcript)
        await store.append_events(
            created.id,
            [
                Event(
                    type=EventType.MODEL_COMPLETED,
                    session_id=created.id,
                    agent_name="assistant",
                    payload={
                        "provider_name": "compaction-provider",
                        "model": "summary-model",
                        "usage_metrics": {
                            "input_tokens": 8,
                            "output_tokens": 2,
                            "total_tokens": 10,
                        },
                    },
                )
            ],
        )
        completed = await store.update_status(created.id, SessionStatus.COMPLETED)
        events = []

        with pytest.raises(RuntimeError, match="budget limit reached"):
            async for event in app.compact_session(
                CompactSessionRequest(
                    session_id=created.id,
                    idempotency_key="compact-exhausted-policy",
                    expected_run_epoch=completed.run_epoch,
                    expected_transcript_cursor=len(transcript),
                )
            ):
                events.append(event)

        assert [event.type for event in events] == [
            EventType.CONTEXT_COMPACTION_STARTED,
            EventType.BUDGET_CHECKED,
            EventType.BUDGET_LIMIT_REACHED,
            EventType.CONTEXT_COMPACTION_FAILED,
        ]
        assert provider.calls == 0

    asyncio.run(run())


def test_compact_session_releases_partial_reservations_when_later_acquisition_fails() -> None:
    async def run() -> None:
        pricing = PriceBook(
            prices=(
                ModelPrice.fixed(
                    provider_name="compaction-provider",
                    model="summary-model",
                    input_per_million=Decimal("1"),
                    output_per_million=Decimal("2"),
                ),
            )
        )
        reservation = BudgetReservation(max_input_tokens=10, max_output_tokens=10)
        provider = UsageCompactionProvider()
        ledger = FailingSecondReservationBudgetLedger()
        store = InMemorySessionStore()
        app = CayuApp(
            session_store=store,
            budget_policy=BudgetPolicy(
                limits=(
                    BudgetLimit(
                        scope="app",
                        max_estimated_cost=Decimal("0.001"),
                        pricing=pricing,
                        reservation=reservation,
                    ),
                )
            ),
            budget_ledger=ledger,
            enable_logging=False,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=ModelCompactor(provider=provider, model="summary-model"),
                max_user_turns=1,
            ),
        )
        created = await _create_profiled_session(
            app,
            store,
            RunRequest(
                agent_name="assistant",
                session_id="sess_compact_partial_reservation_failure",
                messages=[Message.text("user", "create only")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        transcript = [
            Message.text("user", "old request"),
            Message.text("assistant", "old answer"),
            Message.text("user", "current request"),
            Message.text("assistant", "current answer"),
        ]
        await store.append_transcript_messages(created.id, transcript)
        completed = await store.update_status(created.id, SessionStatus.COMPLETED)
        events = []

        with pytest.raises(RuntimeError, match="simulated reservation store failure"):
            async for event in app.compact_session(
                CompactSessionRequest(
                    session_id=created.id,
                    idempotency_key="compact-partial-reservation-failure",
                    expected_run_epoch=completed.run_epoch,
                    expected_transcript_cursor=len(transcript),
                    budget_limits=(
                        BudgetLimit(
                            scope="agent",
                            key="assistant",
                            max_estimated_cost=Decimal("0.001"),
                            pricing=pricing,
                            reservation=reservation,
                        ),
                    ),
                )
            ):
                events.append(event)

        assert [event.type for event in events] == [
            EventType.CONTEXT_COMPACTION_STARTED,
            EventType.BUDGET_CHECKED,
            EventType.BUDGET_RESERVED,
            EventType.BUDGET_RESERVATION_RELEASED,
            EventType.CONTEXT_COMPACTION_FAILED,
        ]
        reservation_sequence = public_event_sequence(events[2].id)
        assert reservation_sequence is not None
        reservation_record = next(
            record
            for record in await store.query_events(
                EventQuery(
                    session_id=created.id,
                    after_sequence=reservation_sequence - 1,
                    limit=1,
                )
            )
            if record.sequence == reservation_sequence
        )
        reservation_id = reservation_record.event.payload["reservation_id"]
        assert ledger.reserve_calls == 2
        assert ledger.release_calls == 1
        assert not await ledger.heartbeat(reservation_id=reservation_id)
        assert provider.calls == 0

    asyncio.run(run())


def test_compact_session_preserves_partial_cleanup_and_releases_remaining_reservations() -> None:
    async def run() -> None:
        pricing = PriceBook(
            prices=(
                ModelPrice.fixed(
                    provider_name="compaction-provider",
                    model="summary-model",
                    input_per_million=Decimal("1"),
                    output_per_million=Decimal("2"),
                ),
            )
        )
        reservation = BudgetReservation(max_input_tokens=10, max_output_tokens=10)
        provider = UsageCompactionProvider()
        ledger = FailingSecondReleaseBudgetLedger()
        store = InMemorySessionStore()
        app = CayuApp(
            session_store=store,
            budget_policy=BudgetPolicy(
                limits=(
                    BudgetLimit(
                        scope="app",
                        max_estimated_cost=Decimal("0.001"),
                        pricing=pricing,
                        reservation=reservation,
                    ),
                    BudgetLimit(
                        scope="agent",
                        key="assistant",
                        max_estimated_cost=Decimal("0.001"),
                        pricing=pricing,
                        reservation=reservation,
                    ),
                    BudgetLimit(
                        scope="causal",
                        key="compact-partial-cleanup-causal",
                        max_estimated_cost=Decimal("0.001"),
                        pricing=pricing,
                        reservation=reservation,
                    ),
                )
            ),
            budget_ledger=ledger,
            enable_logging=False,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=ModelCompactor(provider=provider, model="summary-model"),
                max_user_turns=1,
            ),
        )
        created = await _create_profiled_session(
            app,
            store,
            RunRequest(
                agent_name="assistant",
                session_id="sess_compact_partial_cleanup_failure",
                causal_budget_id="compact-partial-cleanup-causal",
                messages=[Message.text("user", "create only")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        transcript = [
            Message.text("user", "old request"),
            Message.text("assistant", "old answer"),
            Message.text("user", "current request"),
            Message.text("assistant", "current answer"),
        ]
        await store.append_transcript_messages(created.id, transcript)
        completed = await store.update_status(created.id, SessionStatus.COMPLETED)
        events = []

        with pytest.raises(RuntimeError, match="simulated second release failure"):
            async for event in app.compact_session(
                CompactSessionRequest(
                    session_id=created.id,
                    idempotency_key="compact-partial-cleanup-failure",
                    expected_run_epoch=completed.run_epoch,
                    expected_transcript_cursor=len(transcript),
                    budget_limits=(
                        BudgetLimit(
                            scope="app",
                            max_estimated_cost=Decimal("0.000001"),
                            pricing=pricing,
                            reservation=reservation,
                        ),
                    ),
                )
            ):
                events.append(event)

        assert [event.type for event in events] == [
            EventType.CONTEXT_COMPACTION_STARTED,
            EventType.BUDGET_CHECKED,
            EventType.BUDGET_CHECKED,
            EventType.BUDGET_CHECKED,
            EventType.BUDGET_RESERVED,
            EventType.BUDGET_RESERVED,
            EventType.BUDGET_RESERVED,
            EventType.BUDGET_RESERVATION_FAILED,
            EventType.BUDGET_RESERVATION_RELEASED,
            EventType.BUDGET_RESERVATION_RELEASED,
            EventType.BUDGET_RESERVATION_RELEASED,
            EventType.CONTEXT_COMPACTION_FAILED,
        ]
        assert len(ledger.reservation_ids) == 3
        private_releases = await _private_events_for_public_events(
            store,
            created.id,
            events[8:11],
        )
        assert {event.payload["reservation_id"] for event in private_releases} == set(
            ledger.reservation_ids
        )
        assert ledger.release_calls == 4
        for reservation_id in ledger.reservation_ids:
            assert not await ledger.heartbeat(reservation_id=reservation_id)
        assert provider.calls == 0

    asyncio.run(run())


def test_compact_session_reconciles_reservations_and_replays_budget_events() -> None:
    async def run() -> None:
        pricing = PriceBook(
            prices=(
                ModelPrice.fixed(
                    provider_name="compaction-provider",
                    model="summary-model",
                    input_per_million=Decimal("1"),
                    output_per_million=Decimal("2"),
                ),
            )
        )
        provider = UsageCompactionProvider()
        store = InMemorySessionStore()
        app = CayuApp(
            session_store=store,
            budget_policy=BudgetPolicy(
                limits=(
                    BudgetLimit(
                        scope="app",
                        max_estimated_cost=Decimal("0.001"),
                        pricing=pricing,
                        reservation=BudgetReservation(
                            max_input_tokens=10,
                            max_output_tokens=10,
                        ),
                    ),
                )
            ),
            enable_logging=False,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=ModelCompactor(provider=provider, model="summary-model"),
                max_user_turns=1,
            ),
        )
        created = await _create_profiled_session(
            app,
            store,
            RunRequest(
                agent_name="assistant",
                session_id="sess_compact_policy_reconcile",
                messages=[Message.text("user", "create only")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        transcript = [
            Message.text("user", "old request"),
            Message.text("assistant", "old answer"),
            Message.text("user", "current request"),
            Message.text("assistant", "current answer"),
        ]
        await store.append_transcript_messages(created.id, transcript)
        completed = await store.update_status(created.id, SessionStatus.COMPLETED)
        request = CompactSessionRequest(
            session_id=created.id,
            idempotency_key="compact-policy-reconcile",
            expected_run_epoch=completed.run_epoch,
            expected_transcript_cursor=len(transcript),
        )

        first = [event async for event in app.compact_session(request)]
        replay = [event async for event in app.compact_session(request)]

        assert [event.type for event in first] == [
            EventType.CONTEXT_COMPACTION_STARTED,
            EventType.BUDGET_CHECKED,
            EventType.BUDGET_RESERVED,
            EventType.REQUEST_FOOTPRINT_RECORDED,
            EventType.MODEL_STARTED,
            EventType.MODEL_COMPLETED,
            EventType.BUDGET_RECONCILED,
            EventType.BUDGET_CHECKED,
            EventType.CONTEXT_COMPACTION_COMPLETED,
            EventType.SESSION_CHECKPOINTED,
        ]
        assert [event.id for event in replay] == [event.id for event in first]
        assert provider.calls == 1
        reconciled = first[6]
        assert reconciled.payload["actual_amount"] == "0.000012"
        assert reconciled.payload["operation_id"]

    asyncio.run(run())


def test_compact_session_settles_usage_overflow_at_reserved_amount_once() -> None:
    class OverflowUsageCompactionProvider(ModelProvider):
        name = "compaction-provider"

        def __init__(self) -> None:
            self.calls = 0

        async def stream(self, request: ModelRequest):
            self.calls += 1
            yield ModelStreamEvent.text_delta("provider summary")
            yield ModelStreamEvent.completed(
                {
                    "model": request.model,
                    "usage": {
                        "input_tokens": MAX_DURABLE_JSON_INTEGER,
                        "output_tokens": MAX_DURABLE_JSON_INTEGER,
                    },
                }
            )

    async def run() -> None:
        pricing = PriceBook(
            prices=(
                ModelPrice.fixed(
                    provider_name="compaction-provider",
                    model="summary-model",
                    input_per_million=Decimal("1"),
                    output_per_million=Decimal("1"),
                ),
            )
        )
        provider = OverflowUsageCompactionProvider()
        ledger = InMemoryBudgetLedger(reservation_ttl_seconds=60)
        store = InMemorySessionStore()
        app = CayuApp(
            session_store=store,
            budget_ledger=ledger,
            budget_policy=BudgetPolicy(
                limits=(
                    BudgetLimit(
                        scope="app",
                        max_estimated_cost=Decimal("1"),
                        pricing=pricing,
                        reservation=BudgetReservation(
                            max_input_tokens=10,
                            max_output_tokens=10,
                        ),
                    ),
                )
            ),
            enable_logging=False,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=ModelCompactor(provider=provider, model="summary-model"),
                max_user_turns=1,
            ),
        )
        created = await _create_profiled_session(
            app,
            store,
            RunRequest(
                agent_name="assistant",
                session_id="sess_compact_usage_overflow_reconcile",
                messages=[],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        transcript = [
            Message.text("user", "old request"),
            Message.text("assistant", "old answer"),
            Message.text("user", "current request"),
        ]
        await store.append_transcript_messages(created.id, transcript)
        completed = await store.update_status(created.id, SessionStatus.COMPLETED)

        request = CompactSessionRequest(
            session_id=created.id,
            idempotency_key="compact-usage-overflow-reconcile",
            expected_run_epoch=completed.run_epoch,
            expected_transcript_cursor=len(transcript),
        )
        first: list[Event] = []
        with pytest.raises(ContextBuildError, match="integer_out_of_range"):
            async for event in app.compact_session(request):
                first.append(event)
        replay = [event async for event in app.compact_session(request)]

        events = [
            record.event
            for record in await store.query_events(EventQuery(session_id=created.id, limit=100))
        ]
        reservations = [event for event in events if event.type == EventType.BUDGET_RESERVED]
        completions = [
            event
            for event in events
            if event.type == EventType.MODEL_COMPLETED
            and event.payload.get("purpose") == "context_compaction"
        ]
        reconciliations = [event for event in events if event.type == EventType.BUDGET_RECONCILED]

        assert provider.calls == 1
        assert [event.id for event in replay] == [event.id for event in first]
        assert len(reservations) == 1
        assert len(completions) == 1
        assert completions[0].payload["usage_metrics_rejected"] is True
        assert "usage" not in completions[0].payload
        assert "usage_metrics" not in completions[0].payload
        assert len(reconciliations) == 1
        assert reconciliations[0].payload["actual_amount"] == "0.00002"
        record = next(iter(ledger._records.values()))
        assert record.status == "reconciled"
        assert record.actual_amount == Decimal("0.00002")
        assert record.reason == (
            "compaction completed without priced usage; charged reserved amount"
        )
        assert not await ledger.heartbeat(reservation_id=reservations[0].payload["reservation_id"])

    asyncio.run(run())


def test_compact_session_reserves_prompt_cache_compactor_with_session_model() -> None:
    async def run() -> None:
        pricing = PriceBook(
            prices=(
                ModelPrice.fixed(
                    provider_name="compaction-provider",
                    model="summary-model",
                    input_per_million=Decimal("1"),
                    output_per_million=Decimal("2"),
                ),
            )
        )
        provider = UsageCompactionProvider()
        store = InMemorySessionStore()
        app = CayuApp(
            session_store=store,
            budget_policy=BudgetPolicy(
                limits=(
                    BudgetLimit(
                        scope="app",
                        max_estimated_cost=Decimal("0.001"),
                        pricing=pricing,
                        reservation=BudgetReservation(
                            max_input_tokens=10,
                            max_output_tokens=10,
                        ),
                    ),
                )
            ),
            enable_logging=False,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="summary-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=PromptCacheCompactor(provider=provider),
                max_user_turns=1,
            ),
        )
        created = await _create_profiled_session(
            app,
            store,
            RunRequest(
                agent_name="assistant",
                session_id="sess_compact_prompt_cache_reservation",
                messages=[Message.text("user", "create only")],
            ),
            identity=SessionIdentity(
                provider_name="compaction-provider",
                model="summary-model",
            ),
        )
        transcript = [
            Message.text("user", "old request"),
            Message.text("assistant", "old answer"),
            Message.text("user", "current request"),
            Message.text("assistant", "current answer"),
        ]
        await store.append_transcript_messages(created.id, transcript)
        completed = await store.update_status(created.id, SessionStatus.COMPLETED)

        events = [
            event
            async for event in app.compact_session(
                CompactSessionRequest(
                    session_id=created.id,
                    idempotency_key="compact-prompt-cache-reservation",
                    expected_run_epoch=completed.run_epoch,
                    expected_transcript_cursor=len(transcript),
                )
            )
        ]

        assert EventType.BUDGET_RESERVED in [event.type for event in events]
        assert EventType.BUDGET_RECONCILED in [event.type for event in events]
        assert events[-1].type == EventType.SESSION_CHECKPOINTED
        assert provider.calls == 1

    asyncio.run(run())


def test_compact_session_preserves_usage_when_final_reservation_renewal_fails() -> None:
    async def run() -> None:
        pricing = PriceBook(
            prices=(
                ModelPrice.fixed(
                    provider_name="compaction-provider",
                    model="summary-model",
                    input_per_million=Decimal("1"),
                    output_per_million=Decimal("2"),
                ),
            )
        )
        provider = UsageCompactionProvider()
        ledger = FinalRenewalFailureBudgetLedger()
        store = InMemorySessionStore()
        app = CayuApp(
            session_store=store,
            budget_policy=BudgetPolicy(
                limits=(
                    BudgetLimit(
                        scope="app",
                        max_estimated_cost=Decimal("0.001"),
                        pricing=pricing,
                        reservation=BudgetReservation(
                            max_input_tokens=10,
                            max_output_tokens=10,
                        ),
                    ),
                )
            ),
            budget_ledger=ledger,
            enable_logging=False,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=ModelCompactor(provider=provider, model="summary-model"),
                max_user_turns=1,
            ),
        )
        created = await _create_profiled_session(
            app,
            store,
            RunRequest(
                agent_name="assistant",
                session_id="sess_compact_final_renewal",
                messages=[Message.text("user", "create only")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        transcript = [
            Message.text("user", "old request"),
            Message.text("assistant", "old answer"),
            Message.text("user", "current request"),
            Message.text("assistant", "current answer"),
        ]
        await store.append_transcript_messages(created.id, transcript)
        completed = await store.update_status(created.id, SessionStatus.COMPLETED)
        events = []

        with pytest.raises(RuntimeError, match="lease was lost"):
            async for event in app.compact_session(
                CompactSessionRequest(
                    session_id=created.id,
                    idempotency_key="compact-final-renewal",
                    expected_run_epoch=completed.run_epoch,
                    expected_transcript_cursor=len(transcript),
                )
            ):
                events.append(event)

        assert [event.type for event in events] == [
            EventType.CONTEXT_COMPACTION_STARTED,
            EventType.BUDGET_CHECKED,
            EventType.BUDGET_RESERVED,
            EventType.REQUEST_FOOTPRINT_RECORDED,
            EventType.MODEL_STARTED,
            EventType.MODEL_COMPLETED,
            EventType.BUDGET_RECONCILED,
            EventType.CONTEXT_COMPACTION_FAILED,
        ]
        assert events[6].payload["actual_amount"] == "0.000012"
        assert provider.calls == 1
        assert ledger.heartbeat_calls == 2

    asyncio.run(run())


def test_compact_session_releases_reservation_when_initial_renewal_fails() -> None:
    async def run() -> None:
        pricing = PriceBook(
            prices=(
                ModelPrice.fixed(
                    provider_name="compaction-provider",
                    model="summary-model",
                    input_per_million=Decimal("1"),
                    output_per_million=Decimal("2"),
                ),
            )
        )
        provider = UsageCompactionProvider()
        store = InMemorySessionStore()
        app = CayuApp(
            session_store=store,
            budget_policy=BudgetPolicy(
                limits=(
                    BudgetLimit(
                        scope="app",
                        max_estimated_cost=Decimal("0.001"),
                        pricing=pricing,
                        reservation=BudgetReservation(
                            max_input_tokens=10,
                            max_output_tokens=10,
                        ),
                    ),
                )
            ),
            budget_ledger=InitialRenewalFailureBudgetLedger(),
            enable_logging=False,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=ModelCompactor(provider=provider, model="summary-model"),
                max_user_turns=1,
            ),
        )
        created = await _create_profiled_session(
            app,
            store,
            RunRequest(
                agent_name="assistant",
                session_id="sess_compact_initial_renewal",
                messages=[Message.text("user", "create only")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        transcript = [
            Message.text("user", "old request"),
            Message.text("assistant", "old answer"),
            Message.text("user", "current request"),
            Message.text("assistant", "current answer"),
        ]
        await store.append_transcript_messages(created.id, transcript)
        completed = await store.update_status(created.id, SessionStatus.COMPLETED)
        events = []

        with pytest.raises(RuntimeError, match="before provider dispatch"):
            async for event in app.compact_session(
                CompactSessionRequest(
                    session_id=created.id,
                    idempotency_key="compact-initial-renewal",
                    expected_run_epoch=completed.run_epoch,
                    expected_transcript_cursor=len(transcript),
                )
            ):
                events.append(event)

        assert [event.type for event in events] == [
            EventType.CONTEXT_COMPACTION_STARTED,
            EventType.BUDGET_CHECKED,
            EventType.BUDGET_RESERVED,
            EventType.BUDGET_RESERVATION_RELEASED,
            EventType.CONTEXT_COMPACTION_FAILED,
        ]
        assert events[3].payload["actual_amount"] is None
        assert provider.calls == 0

    asyncio.run(run())


def test_compact_session_skips_provider_reservations_for_deterministic_compaction() -> None:
    async def run() -> None:
        pricing = PriceBook(
            prices=(
                ModelPrice.fixed(
                    provider_name="fake",
                    model="fake-model",
                    input_per_million=Decimal("1"),
                    output_per_million=Decimal("2"),
                ),
            )
        )
        store = InMemorySessionStore()
        app = CayuApp(
            session_store=store,
            budget_policy=BudgetPolicy(
                limits=(
                    BudgetLimit(
                        scope="app",
                        max_estimated_cost=Decimal("0.001"),
                        pricing=pricing,
                        reservation=BudgetReservation(
                            max_input_tokens=10,
                            max_output_tokens=10,
                        ),
                    ),
                )
            ),
            enable_logging=False,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=RecordingCompactor(),
                max_user_turns=1,
            ),
        )
        created = await _create_profiled_session(
            app,
            store,
            RunRequest(
                agent_name="assistant",
                session_id="sess_deterministic_compaction_budget",
                messages=[Message.text("user", "create only")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        transcript = [
            Message.text("user", "old request"),
            Message.text("assistant", "old answer"),
            Message.text("user", "current request"),
            Message.text("assistant", "current answer"),
        ]
        await store.append_transcript_messages(created.id, transcript)
        completed = await store.update_status(created.id, SessionStatus.COMPLETED)

        events = [
            event
            async for event in app.compact_session(
                CompactSessionRequest(
                    session_id=created.id,
                    idempotency_key="compact-deterministic-budget",
                    expected_run_epoch=completed.run_epoch,
                    expected_transcript_cursor=len(transcript),
                )
            )
        ]

        assert events[-1].type == EventType.SESSION_CHECKPOINTED
        assert not any(str(event.type).startswith("budget.") for event in events)

    asyncio.run(run())


def test_compact_session_rejects_unknown_compactor_budget_identity_before_work() -> None:
    async def run() -> None:
        pricing = PriceBook(
            prices=(
                ModelPrice.fixed(
                    provider_name="compaction-provider",
                    model="summary-model",
                    input_per_million=Decimal("1"),
                    output_per_million=Decimal("2"),
                ),
            )
        )
        compactor = UndeclaredProviderCompactor()
        store = InMemorySessionStore()
        app = CayuApp(
            session_store=store,
            budget_policy=BudgetPolicy(
                limits=(
                    BudgetLimit(
                        scope="app",
                        max_estimated_cost=Decimal("0.001"),
                        pricing=pricing,
                    ),
                )
            ),
            enable_logging=False,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=compactor,
                max_user_turns=1,
            ),
        )
        created = await _create_profiled_session(
            app,
            store,
            RunRequest(
                agent_name="assistant",
                session_id="sess_unknown_compactor_budget_identity",
                messages=[Message.text("user", "create only")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        transcript = [
            Message.text("user", "old request"),
            Message.text("assistant", "old answer"),
            Message.text("user", "current request"),
            Message.text("assistant", "current answer"),
        ]
        await store.append_transcript_messages(created.id, transcript)
        completed = await store.update_status(created.id, SessionStatus.COMPLETED)

        with pytest.raises(RuntimeError, match="declare provider_budget_identity"):
            async for _event in app.compact_session(
                CompactSessionRequest(
                    session_id=created.id,
                    idempotency_key="compact-unknown-budget-identity",
                    expected_run_epoch=completed.run_epoch,
                    expected_transcript_cursor=len(transcript),
                )
            ):
                pass

        assert compactor.calls == 0
        assert await store.load_checkpoint(created.id) is None

    asyncio.run(run())


def test_compact_session_rejects_provider_backed_deterministic_identity_claim() -> None:
    class NeverDispatchedProvider(ModelProvider):
        name = "compaction-provider"

        def __init__(self) -> None:
            self.requests: list[ModelRequest] = []

        async def stream(self, request: ModelRequest):
            self.requests.append(request)
            yield ModelStreamEvent.completed(
                {
                    "model": request.model,
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                }
            )

    class MisdeclaredModelCompactor(ModelCompactor):
        def provider_budget_identity(self, session) -> None:
            del session
            return None

    async def run() -> None:
        store = InMemorySessionStore()
        provider = NeverDispatchedProvider()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=MisdeclaredModelCompactor(
                    provider=provider,
                    model="summary-model",
                ),
                max_user_turns=1,
            ),
        )
        created = await _create_profiled_session(
            app,
            store,
            RunRequest(
                agent_name="assistant",
                session_id="sess_misdeclared_compactor_budget_identity",
                messages=[Message.text("user", "create only")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        transcript = [
            Message.text("user", "old request"),
            Message.text("assistant", "old answer"),
            Message.text("user", "current request"),
            Message.text("assistant", "current answer"),
        ]
        await store.append_transcript_messages(created.id, transcript)
        completed = await store.update_status(created.id, SessionStatus.COMPLETED)

        with pytest.raises(
            RuntimeError,
            match="cannot declare a deterministic budget identity",
        ):
            async for _event in app.compact_session(
                CompactSessionRequest(
                    session_id=created.id,
                    idempotency_key="compact-misdeclared-budget-identity",
                    expected_run_epoch=completed.run_epoch,
                    expected_transcript_cursor=len(transcript),
                    limits=RunLimits(max_total_tokens=10),
                )
            ):
                pass

        assert provider.requests == []
        assert await store.load_checkpoint(created.id) is None

    asyncio.run(run())


@pytest.mark.parametrize(
    "identity",
    [
        ("compaction-provider\x00suffix", "summary-model"),
        ("compaction-provider", "summary\nmodel"),
        ("compaction-provider", "summary-\ud800-model"),
        ("compaction-provider", "m" * 513),
    ],
)
def test_compact_session_rejects_unsafe_compactor_budget_identity_before_work(
    identity: tuple[str, str],
) -> None:
    async def run() -> None:
        pricing = PriceBook(
            prices=(
                ModelPrice.fixed(
                    provider_name="compaction-provider",
                    model="summary-model",
                    input_per_million=Decimal("1"),
                    output_per_million=Decimal("2"),
                ),
            )
        )
        compactor = UnsafeProviderIdentityCompactor(identity)
        store = InMemorySessionStore()
        app = CayuApp(
            session_store=store,
            budget_policy=BudgetPolicy(
                limits=(
                    BudgetLimit(
                        scope="app",
                        max_estimated_cost=Decimal("0.001"),
                        pricing=pricing,
                    ),
                )
            ),
            enable_logging=False,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=compactor,
                max_user_turns=1,
            ),
        )
        created = await _create_profiled_session(
            app,
            store,
            RunRequest(
                agent_name="assistant",
                session_id="sess_unsafe_compactor_budget_identity",
                messages=[Message.text("user", "create only")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        transcript = [
            Message.text("user", "old request"),
            Message.text("assistant", "old answer"),
            Message.text("user", "current request"),
            Message.text("assistant", "current answer"),
        ]
        await store.append_transcript_messages(created.id, transcript)
        completed = await store.update_status(created.id, SessionStatus.COMPLETED)

        with pytest.raises(ValueError, match="without control characters"):
            async for _event in app.compact_session(
                CompactSessionRequest(
                    session_id=created.id,
                    idempotency_key="compact-unsafe-budget-identity",
                    expected_run_epoch=completed.run_epoch,
                    expected_transcript_cursor=len(transcript),
                )
            ):
                pass

        assert compactor.calls == 0
        assert await store.load_checkpoint(created.id) is None

    asyncio.run(run())


def test_compact_session_persists_reservation_lifecycle_before_provider_work() -> None:
    async def run() -> None:
        pricing = PriceBook(
            prices=(
                ModelPrice.fixed(
                    provider_name="compaction-provider",
                    model="summary-model",
                    input_per_million=Decimal("1"),
                    output_per_million=Decimal("2"),
                ),
            )
        )
        store = InMemorySessionStore()
        provider = ReservationInspectingCompactionProvider(store)
        app = CayuApp(
            session_store=store,
            budget_policy=BudgetPolicy(
                limits=(
                    BudgetLimit(
                        scope="app",
                        max_estimated_cost=Decimal("0.001"),
                        pricing=pricing,
                        reservation=BudgetReservation(
                            max_input_tokens=10,
                            max_output_tokens=10,
                        ),
                    ),
                )
            ),
            enable_logging=False,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=ModelCompactor(provider=provider, model="summary-model"),
                max_user_turns=1,
            ),
        )
        created = await _create_profiled_session(
            app,
            store,
            RunRequest(
                agent_name="assistant",
                session_id="sess_compact_persisted_reservation",
                messages=[Message.text("user", "create only")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        provider.session_id = created.id
        transcript = [
            Message.text("user", "old request"),
            Message.text("assistant", "old answer"),
            Message.text("user", "current request"),
            Message.text("assistant", "current answer"),
        ]
        await store.append_transcript_messages(created.id, transcript)
        completed = await store.update_status(created.id, SessionStatus.COMPLETED)
        stream = app.compact_session(
            CompactSessionRequest(
                session_id=created.id,
                idempotency_key="compact-persisted-reservation",
                expected_run_epoch=completed.run_epoch,
                expected_transcript_cursor=len(transcript),
            )
        )

        started = await anext(stream)
        checked = await anext(stream)
        reserved = await anext(stream)
        assert [started.type, checked.type, reserved.type] == [
            EventType.CONTEXT_COMPACTION_STARTED,
            EventType.BUDGET_CHECKED,
            EventType.BUDGET_RESERVED,
        ]
        assert provider.calls == 1
        assert provider.durable_types_before_dispatch == [
            EventType.CONTEXT_COMPACTION_STARTED,
            EventType.BUDGET_CHECKED,
            EventType.BUDGET_RESERVED,
            EventType.REQUEST_FOOTPRINT_RECORDED,
            EventType.MODEL_STARTED,
        ]
        durable_records_before_close = await store.query_events(
            EventQuery(session_id=created.id, limit=100)
        )
        durable_before_close = [record.event for record in durable_records_before_close]
        assert [
            public_event_id(record.sequence) for record in durable_records_before_close[:3]
        ] == [
            started.id,
            checked.id,
            reserved.id,
        ]
        assert [event.type for event in durable_before_close] == [
            EventType.CONTEXT_COMPACTION_STARTED,
            EventType.BUDGET_CHECKED,
            EventType.BUDGET_RESERVED,
            EventType.REQUEST_FOOTPRINT_RECORDED,
            EventType.MODEL_STARTED,
            EventType.MODEL_COMPLETED,
            EventType.BUDGET_RECONCILED,
            EventType.BUDGET_CHECKED,
            EventType.CONTEXT_COMPACTION_COMPLETED,
            EventType.SESSION_CHECKPOINTED,
        ]

        await stream.aclose()

        durable_after_close = [
            record.event
            for record in await store.query_events(EventQuery(session_id=created.id, limit=100))
        ]
        assert [event.id for event in durable_after_close] == [
            event.id for event in durable_before_close
        ]
        checkpoint = await store.load_checkpoint(created.id)
        assert checkpoint is not None
        record = await store.load_session_operation(
            created.id,
            "compact-persisted-reservation",
        )
        assert record is not None
        assert record["event_ids"] == [event.id for event in durable_after_close]
        assert record["status"] == "completed"

    asyncio.run(run())

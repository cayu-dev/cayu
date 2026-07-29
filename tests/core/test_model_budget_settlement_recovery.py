from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from cayu.core import AgentSpec, Event, EventType, Message
from cayu.core.billing import BillingIdentity, PricingContext
from cayu.providers import ModelProvider, ModelRequest, ModelStreamEvent
from cayu.runtime import (
    BudgetLimit,
    BudgetPolicy,
    BudgetReservation,
    BudgetWindow,
    CayuApp,
    CheckpointCompactionContextPolicy,
    IncompleteSessionRecoveryRequest,
    InMemoryBudgetLedger,
    InMemorySessionStore,
    ModelCompactor,
    RunRequest,
)
from cayu.runtime.budgets import (
    MODEL_COMPLETION_BUDGET_SETTLEMENTS_KEY,
    PUBLICATION_FALLBACK_BUDGET_REASON,
    budget_reservation_payload,
)
from cayu.runtime.costs import (
    ContextualPricingRequirement,
    ModelPrice,
    PriceBook,
    Provenance,
)
from cayu.runtime.execution_units import ModelAttemptIdentity
from cayu.runtime.sessions import EventQuery
from cayu.vaults import REDACTED_SECRET, SecretRedactor


class _MutableClock:
    def __init__(self, value: datetime) -> None:
        self.value = value

    def __call__(self) -> datetime:
        return self.value


class _SimulatedProcessLoss(BaseException):
    pass


class _CompletedProvider(ModelProvider):
    name = "fake"

    def __init__(self) -> None:
        self.requests: list[ModelRequest] = []

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        self.requests.append(request)
        yield ModelStreamEvent.text_delta("durable answer")
        yield ModelStreamEvent.completed(
            {
                "usage": {
                    "input_tokens": 7,
                    "output_tokens": 3,
                    "total_tokens": 10,
                }
            }
        )


class _MillionTokenCompletedProvider(_CompletedProvider):
    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        self.requests.append(request)
        yield ModelStreamEvent.text_delta("durable expensive answer")
        yield ModelStreamEvent.completed(
            {
                "usage": {
                    "input_tokens": 1_000_000,
                    "output_tokens": 0,
                    "total_tokens": 1_000_000,
                }
            }
        )


class _DelayedMillionTokenCompletedProvider(_MillionTokenCompletedProvider):
    def __init__(self, clock: _MutableClock) -> None:
        super().__init__()
        self._clock = clock

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        self.requests.append(request)
        self._clock.value += timedelta(seconds=61)
        yield ModelStreamEvent.completed(
            {
                "usage": {
                    "input_tokens": 1_000_000,
                    "output_tokens": 0,
                    "total_tokens": 1_000_000,
                }
            }
        )


class _BillingIdentityCompletedProvider(_CompletedProvider):
    def __init__(self, identity: BillingIdentity) -> None:
        super().__init__()
        self._identity = identity

    async def billing_identity_for_request(
        self,
        request: ModelRequest,
    ) -> BillingIdentity:
        del request
        return self._identity


class _CrashBeforeFirstReconciliation(InMemoryBudgetLedger):
    def __init__(self, *, clock=None) -> None:
        super().__init__(clock=clock, reservation_ttl_seconds=1)
        self.crash_enabled = True

    async def reconcile(self, **kwargs):
        if self.crash_enabled:
            self.crash_enabled = False
            raise _SimulatedProcessLoss("process lost after model publication")
        return await super().reconcile(**kwargs)


class _CrashBeforeFirstSettlementEventStore(InMemorySessionStore):
    def __init__(self) -> None:
        super().__init__()
        self.crash_enabled = True

    async def append_event(self, session_id, event):
        if self.crash_enabled and event.type == EventType.BUDGET_RECONCILED:
            self.crash_enabled = False
            raise _SimulatedProcessLoss("process lost after ledger settlement")
        await super().append_event(session_id, event)


class _LoseFirstReconciliationAcknowledgement(InMemoryBudgetLedger):
    def __init__(self) -> None:
        super().__init__(reservation_ttl_seconds=1)
        self.crash_enabled = True

    async def reconcile(self, **kwargs):
        reconciliation = await super().reconcile(**kwargs)
        if self.crash_enabled:
            self.crash_enabled = False
            raise _SimulatedProcessLoss("ledger acknowledgement lost after commit")
        return reconciliation


class _LoseEveryDispatchFenceAcknowledgement(InMemoryBudgetLedger):
    async def mark_dispatched(self, **kwargs):
        await super().mark_dispatched(**kwargs)
        raise RuntimeError("dispatch fence acknowledgement remains unavailable")


class _LoseFirstSettlementEventAcknowledgementStore(InMemorySessionStore):
    def __init__(self) -> None:
        super().__init__()
        self.acknowledgement_loss_enabled = True

    async def append_event(self, session_id, event):
        if self.acknowledgement_loss_enabled and event.type == EventType.BUDGET_RECONCILED:
            self.acknowledgement_loss_enabled = False
            await super().append_event(session_id, event)
            raise RuntimeError("session event acknowledgement lost after commit")
        await super().append_event(session_id, event)


class _MutatedSettlementLoadLedger(InMemoryBudgetLedger):
    async def load_settlement(self, settlement_id):
        settlement = await super().load_settlement(settlement_id)
        if settlement is not None:
            settlement.event.payload["actual_amount"] = "999"
        return settlement


class _MutatedSettlementAcknowledgementLedger(InMemoryBudgetLedger):
    async def mark_settlement_event_published(self, **kwargs):
        settlement = await super().mark_settlement_event_published(**kwargs)
        settlement.event.payload["actual_amount"] = "999"
        return settlement


class _MutatedSettlementPageLedger(InMemoryBudgetLedger):
    async def list_pending_settlements(self, **kwargs):
        settlements = await super().list_pending_settlements(**kwargs)
        if settlements:
            settlements[0].event.payload["actual_amount"] = "999"
        return settlements


def _budget_policy(
    *,
    provenance: Provenance | None = None,
) -> BudgetPolicy:
    return BudgetPolicy(
        limits=(
            BudgetLimit(
                scope="app",
                max_estimated_cost=Decimal("1"),
                pricing=PriceBook(
                    prices=(
                        ModelPrice.fixed(
                            provider_name="fake",
                            model="fake-model",
                            input_per_million=Decimal("1"),
                            output_per_million=Decimal("10"),
                            provenance=provenance,
                        ),
                    )
                ),
                reservation=BudgetReservation(
                    max_input_tokens=1_000_000,
                    max_output_tokens=0,
                ),
            ),
        )
    )


def _app(
    store: InMemorySessionStore,
    ledger: InMemoryBudgetLedger,
    provider: _CompletedProvider,
    *,
    clock=None,
    budget_policy: BudgetPolicy | None = None,
    secret_redactor: SecretRedactor | None = None,
) -> CayuApp:
    kwargs = {} if clock is None else {"clock": clock}
    app = CayuApp(
        session_store=store,
        budget_ledger=ledger,
        budget_policy=budget_policy if budget_policy is not None else _budget_policy(),
        secret_redactor=secret_redactor,
        enable_logging=False,
        **kwargs,
    )
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))
    return app


async def _run_until_process_loss(app: CayuApp, session_id: str) -> None:
    with pytest.raises(_SimulatedProcessLoss):
        async for _event in app.run(
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "answer once")],
            )
        ):
            pass


async def _budget_events(
    store: InMemorySessionStore,
    session_id: str,
) -> list:
    records = await store.query_events(
        EventQuery(
            session_id=session_id,
            event_types=(
                EventType.BUDGET_RECONCILED,
                EventType.BUDGET_RESERVATION_RELEASED,
            ),
            limit=100,
        )
    )
    return [record.event for record in records]


async def _create_terminal_session(
    store: InMemorySessionStore,
    session_id: str,
) -> None:
    provider = _CompletedProvider()
    app = CayuApp(session_store=store, enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))
    async for _event in app.run(
        RunRequest(
            agent_name="assistant",
            session_id=session_id,
            messages=[Message.text("user", "create an audit owner")],
        )
    ):
        pass


async def _seed_pending_release(
    store: InMemorySessionStore,
    ledger: InMemoryBudgetLedger,
    *,
    session_id: str,
) -> Event:
    await _create_terminal_session(store, session_id)
    limit = _budget_policy().limits[0]
    result = await ledger.reserve(
        limit=limit,
        session_id=session_id,
        agent_name="assistant",
        provider_name="fake",
        model="fake-model",
        model_attempt_identity=ModelAttemptIdentity(
            model_step_id="mstep_" + "1" * 32,
            model_attempt_id="matt_" + "2" * 32,
        ),
    )
    assert result.record is not None
    event = Event(
        type=EventType.BUDGET_RESERVED,
        session_id=session_id,
        agent_name="assistant",
        payload=budget_reservation_payload(result),
    )
    claim = {
        "reservation_id": result.record.reservation_id,
        "publication_session_id": session_id,
        "publication_id": event.id,
    }
    await store.claim_budget_reservation_identity(**claim)
    await ledger.claim_reservation_identity(**claim)
    await store.append_event(session_id, event)
    side_effect_claim = await store.claim_persisted_event_side_effect(
        session_id=session_id,
        event_id=event.id,
    )
    assert side_effect_claim is not None
    await store.mark_persisted_event_side_effect_delivered(side_effect_claim)
    reconciliation = await ledger.release(
        reservation_id=result.record.reservation_id,
        reason="provider not dispatched",
    )
    settlement = await ledger.load_settlement(reconciliation.settlement_id)
    assert settlement is not None
    return settlement.event


def _secret_pricing_policy(secret: str) -> BudgetPolicy:
    return _budget_policy(
        provenance=Provenance(
            source=secret,
            url="application://price-book",
            as_of="2026-07-29",
        )
    )


def _decimal_collision_policy(*, reserved_input_tokens: int) -> BudgetPolicy:
    return BudgetPolicy(
        limits=(
            BudgetLimit(
                scope="app",
                max_estimated_cost=Decimal("24691356"),
                pricing=PriceBook(
                    prices=(
                        ModelPrice.fixed(
                            provider_name="fake",
                            model="fake-model",
                            input_per_million=Decimal("12345678"),
                            output_per_million=Decimal("0"),
                        ),
                    )
                ),
                reservation=BudgetReservation(
                    max_input_tokens=reserved_input_tokens,
                    max_output_tokens=0,
                ),
            ),
        )
    )


def test_secret_colliding_reserved_amount_fails_before_ledger_mutation() -> None:
    async def scenario() -> None:
        session_id = "sess_budget_reserved_amount_secret_collision"
        secret = "12345678"
        store = InMemorySessionStore()
        ledger = InMemoryBudgetLedger()
        provider = _MillionTokenCompletedProvider()
        events = [
            event
            async for event in _app(
                store,
                ledger,
                provider,
                budget_policy=_decimal_collision_policy(
                    reserved_input_tokens=1_000_000,
                ),
                secret_redactor=SecretRedactor(secret),
            ).run(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "reject unsafe accounting authority")],
                )
            )
        ]

        assert provider.requests == []
        assert ledger._records == {}
        assert ledger._settlements == {}
        assert EventType.BUDGET_RESERVED not in {event.type for event in events}
        assert EventType.MODEL_STARTED not in {event.type for event in events}
        assert events[-1].type == EventType.SESSION_FAILED
        assert all(secret not in event.model_dump_json() for event in events)

    asyncio.run(scenario())


def test_secret_colliding_actual_amount_uses_prevalidated_conservative_settlement() -> None:
    async def scenario() -> None:
        session_id = "sess_budget_actual_amount_secret_collision"
        secret = "12345678"
        store = InMemorySessionStore()
        ledger = InMemoryBudgetLedger()
        provider = _MillionTokenCompletedProvider()
        events = [
            event
            async for event in _app(
                store,
                ledger,
                provider,
                budget_policy=_decimal_collision_policy(
                    reserved_input_tokens=2_000_000,
                ),
                secret_redactor=SecretRedactor(secret),
            ).run(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "retain completed usage safely")],
                )
            )
        ]

        assert len(provider.requests) == 1
        assert events[-1].type == EventType.SESSION_COMPLETED
        completion = next(event for event in events if event.type == EventType.MODEL_COMPLETED)
        assert completion.payload["usage"]["input_tokens"] == 1_000_000
        evidence = completion.payload[MODEL_COMPLETION_BUDGET_SETTLEMENTS_KEY][0]
        assert evidence["settlement_kind"] == "conservative"
        assert evidence["actual_amount"] == "24691356"
        assert evidence["reason"] == PUBLICATION_FALLBACK_BUDGET_REASON
        reconciled = next(event for event in events if event.type == EventType.BUDGET_RECONCILED)
        assert reconciled.payload["settlement_kind"] == "conservative"
        assert reconciled.payload["actual_amount"] == "24691356"
        assert reconciled.payload["reason"] == PUBLICATION_FALLBACK_BUDGET_REASON
        record = next(iter(ledger._records.values()))
        assert record.status == "reconciled"
        assert record.actual_amount == Decimal("24691356")
        settlement = next(iter(ledger._settlements.values()))
        assert settlement.event == reconciled
        assert settlement.event_published is True
        assert all(secret not in event.model_dump_json() for event in events)

    asyncio.run(scenario())


def test_redacted_request_billing_identity_recovers_original_reservation() -> None:
    async def scenario() -> None:
        secret = "billing-workload-secret"
        session_id = "sess_model_budget_redacted_request_identity"
        identity = BillingIdentity(
            provider_name="fake",
            resource_id="fake-model",
            request_evidence={"opaque": secret},
            pricing_contexts=(PricingContext(dimensions={"billing_tenant": secret}),),
        )
        durable_identity = identity.model_copy(
            update={
                "request_evidence": {"opaque": REDACTED_SECRET},
                "pricing_contexts": (
                    PricingContext(dimensions={"billing_tenant": REDACTED_SECRET}),
                ),
            },
            deep=True,
        )
        policy = BudgetPolicy(
            limits=(
                BudgetLimit(
                    scope="app",
                    max_estimated_cost=Decimal("1"),
                    pricing=PriceBook(
                        contextual_pricing_requirements=(
                            ContextualPricingRequirement(
                                provider_name="fake",
                                dimensions=("billing_tenant",),
                            ),
                        ),
                        prices=(
                            ModelPrice.fixed(
                                provider_name="fake",
                                model="fake-model",
                                match="exact",
                                input_per_million=Decimal("1"),
                                output_per_million=Decimal("10"),
                                pricing_context={
                                    "billing_tenant": (secret,),
                                },
                            ),
                        ),
                    ),
                    reservation=BudgetReservation(
                        max_input_tokens=1_000_000,
                        max_output_tokens=0,
                    ),
                ),
            )
        )
        store = InMemorySessionStore()
        ledger = _CrashBeforeFirstReconciliation()
        provider = _BillingIdentityCompletedProvider(identity)
        redactor = SecretRedactor(secret)
        await _run_until_process_loss(
            _app(
                store,
                ledger,
                provider,
                budget_policy=policy,
                secret_redactor=redactor,
            ),
            session_id,
        )

        active = next(iter(ledger._records.values()))
        assert active.status == "active"
        assert active.dispatch_id is not None
        assert active.billing_identity == durable_identity
        completion_records = await store.query_events(
            EventQuery(
                session_id=session_id,
                event_type=EventType.MODEL_COMPLETED,
                limit=1,
            )
        )
        assert len(completion_records) == 1
        completion = completion_records[0].event
        assert completion.payload["billing_identity"] == durable_identity.model_dump(mode="json")
        evidence = completion.payload[MODEL_COMPLETION_BUDGET_SETTLEMENTS_KEY][0]
        assert evidence["billing_identity"] == durable_identity.model_dump(mode="json")
        assert secret not in completion.model_dump_json()

        await _app(
            store,
            ledger,
            provider,
            budget_policy=policy,
            secret_redactor=redactor,
        ).recover_incomplete_session(IncompleteSessionRecoveryRequest(session_id=session_id))

        assert len(provider.requests) == 1
        reconciled = next(iter(ledger._records.values()))
        assert reconciled.status == "reconciled"
        assert reconciled.actual_amount == Decimal("0.000037")
        assert reconciled.billing_identity == durable_identity
        budget_events = await _budget_events(store, session_id)
        assert len(budget_events) == 1
        assert budget_events[0].payload["billing_identity"] == durable_identity.model_dump(
            mode="json"
        )
        assert next(iter(ledger._settlements.values())).event_published is True

    asyncio.run(scenario())


def test_publication_fallback_uses_completion_time_for_rolling_budget() -> None:
    async def scenario() -> None:
        clock = _MutableClock(datetime(2026, 7, 29, tzinfo=UTC))
        completion_time = clock.value + timedelta(seconds=61)
        limit = (
            _decimal_collision_policy(
                reserved_input_tokens=2_000_000,
            )
            .limits[0]
            .model_copy(
                update={"window": BudgetWindow.rolling(seconds=60)},
                deep=True,
            )
        )
        ledger = InMemoryBudgetLedger(clock=clock)
        provider = _DelayedMillionTokenCompletedProvider(clock)
        events = [
            event
            async for event in _app(
                InMemorySessionStore(),
                ledger,
                provider,
                clock=clock,
                budget_policy=BudgetPolicy(limits=(limit,)),
                secret_redactor=SecretRedactor(
                    [
                        "12345678",
                        completion_time.isoformat(),
                    ]
                ),
            ).run(
                RunRequest(
                    agent_name="assistant",
                    session_id="sess_budget_completion_window",
                    messages=[Message.text("user", "complete after the rolling boundary")],
                )
            )
        ]

        assert events[-1].type == EventType.SESSION_COMPLETED
        completion = next(event for event in events if event.type == EventType.MODEL_COMPLETED)
        evidence = completion.payload[MODEL_COMPLETION_BUDGET_SETTLEMENTS_KEY][0]
        assert evidence["settlement_kind"] == "conservative"
        assert type(evidence["settled_at_unix_us"]) is int
        assert "settled_at" not in evidence
        assert completion_time.isoformat() not in completion.model_dump_json()
        first = next(iter(ledger._records.values()))
        assert first.status == "reconciled"
        assert first.actual_amount == Decimal("24691356")
        assert first.updated_at == completion_time
        settlement = next(iter(ledger._settlements.values()))
        assert settlement.reconciliation.settled_at == completion_time
        assert settlement.event_published is True
        clock.value += timedelta(microseconds=1)
        second = await ledger.reserve(
            limit=limit,
            session_id="sess_budget_completion_window_second",
            agent_name="assistant",
            provider_name="fake",
            model="fake-model",
            model_attempt_identity=ModelAttemptIdentity(
                model_step_id="mstep_" + "1" * 32,
                model_attempt_id="matt_" + "2" * 32,
            ),
            effective_at=clock.value,
        )
        assert second.accepted is False

    asyncio.run(scenario())


def test_publication_fallback_is_isolated_to_the_unpublishable_limit() -> None:
    def limit(
        *,
        scope: str,
        maximum: str,
        input_price: str,
        max_input_tokens: int,
    ) -> BudgetLimit:
        return BudgetLimit(
            scope=scope,
            key="assistant" if scope == "agent" else None,
            max_estimated_cost=Decimal(maximum),
            pricing=PriceBook(
                prices=(
                    ModelPrice.fixed(
                        provider_name="fake",
                        model="fake-model",
                        input_per_million=Decimal(input_price),
                        output_per_million=Decimal("0"),
                    ),
                )
            ),
            reservation=BudgetReservation(
                max_input_tokens=max_input_tokens,
                max_output_tokens=0,
            ),
        )

    async def scenario() -> None:
        ledger = InMemoryBudgetLedger()
        policy = BudgetPolicy(
            limits=(
                limit(
                    scope="app",
                    maximum="24691356",
                    input_price="12345678",
                    max_input_tokens=2_000_000,
                ),
                limit(
                    scope="agent",
                    maximum="10",
                    input_price="1",
                    max_input_tokens=10_000_000,
                ),
            )
        )
        events = [
            event
            async for event in _app(
                InMemorySessionStore(),
                ledger,
                _MillionTokenCompletedProvider(),
                budget_policy=policy,
                secret_redactor=SecretRedactor("12345678"),
            ).run(
                RunRequest(
                    agent_name="assistant",
                    session_id="sess_budget_per_limit_fallback",
                    messages=[Message.text("user", "settle both limits")],
                )
            )
        ]

        assert events[-1].type == EventType.SESSION_COMPLETED
        records = {record.scope: record for record in ledger._records.values()}
        assert records["app"].reserved_amount == Decimal("24691356")
        assert records["app"].actual_amount == Decimal("24691356")
        assert records["agent"].reserved_amount == Decimal("10")
        assert records["agent"].actual_amount == Decimal("1")
        settlements = {
            settlement.reconciliation.budget_limit_id: settlement.reconciliation
            for settlement in ledger._settlements.values()
        }
        assert settlements[records["app"].budget_limit_id].settlement_kind == "conservative"
        assert settlements[records["agent"].budget_limit_id].settlement_kind == "completed"

    asyncio.run(scenario())


def test_model_settlement_redacts_dynamic_pricing_before_ledger_commit() -> None:
    async def scenario() -> None:
        secret = "settlement-pricing-provenance-secret"
        session_id = "sess_model_budget_redacted_pricing"
        store = InMemorySessionStore()
        ledger = InMemoryBudgetLedger()
        provider = _CompletedProvider()
        events = [
            event
            async for event in _app(
                store,
                ledger,
                provider,
                budget_policy=_secret_pricing_policy(secret),
                secret_redactor=SecretRedactor(secret),
            ).run(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "publish safe accounting")],
                )
            )
        ]

        assert len(provider.requests) == 1
        assert events[-1].type == EventType.SESSION_COMPLETED
        completion = next(event for event in events if event.type == EventType.MODEL_COMPLETED)
        settlement_evidence = completion.payload[MODEL_COMPLETION_BUDGET_SETTLEMENTS_KEY][0]
        assert settlement_evidence["pricing"]["provenance"]["source"] == REDACTED_SECRET
        reconciled = next(event for event in events if event.type == EventType.BUDGET_RECONCILED)
        assert reconciled.payload["pricing"]["provenance"]["source"] == REDACTED_SECRET
        committed = next(iter(ledger._settlements.values()))
        assert committed.event == reconciled
        assert committed.event_published is True

    asyncio.run(scenario())


def test_recovery_replays_redacted_dynamic_pricing_without_redispatch() -> None:
    async def scenario() -> None:
        secret = "recovered-settlement-pricing-secret"
        session_id = "sess_model_budget_redacted_pricing_recovery"
        store = InMemorySessionStore()
        ledger = _CrashBeforeFirstReconciliation()
        provider = _CompletedProvider()
        policy = _secret_pricing_policy(secret)
        redactor = SecretRedactor(secret)
        await _run_until_process_loss(
            _app(
                store,
                ledger,
                provider,
                budget_policy=policy,
                secret_redactor=redactor,
            ),
            session_id,
        )

        completion_records = await store.query_events(
            EventQuery(
                session_id=session_id,
                event_type=EventType.MODEL_COMPLETED,
                limit=1,
            )
        )
        assert len(completion_records) == 1
        settlement_evidence = completion_records[0].event.payload[
            MODEL_COMPLETION_BUDGET_SETTLEMENTS_KEY
        ][0]
        assert settlement_evidence["pricing"]["provenance"]["source"] == REDACTED_SECRET

        await _app(
            store,
            ledger,
            provider,
            budget_policy=policy,
            secret_redactor=redactor,
        ).recover_incomplete_session(IncompleteSessionRecoveryRequest(session_id=session_id))

        assert len(provider.requests) == 1
        budget_events = await _budget_events(store, session_id)
        assert len(budget_events) == 1
        assert budget_events[0].payload["pricing"]["provenance"]["source"] == REDACTED_SECRET
        committed = next(iter(ledger._settlements.values()))
        assert committed.event == budget_events[0]
        assert committed.event_published is True

    asyncio.run(scenario())


def test_automatic_compaction_settlement_redacts_dynamic_pricing() -> None:
    secret = "compaction-settlement-pricing-secret"
    compactor_provider = _CompletedProvider()
    runtime_provider = _CompletedProvider()
    ledger = InMemoryBudgetLedger()
    secret_policy = _secret_pricing_policy(secret)
    policy = BudgetPolicy(
        limits=(
            secret_policy.limits[0].model_copy(
                update={"max_estimated_cost": Decimal("2")},
                deep=True,
            ),
        )
    )
    app = CayuApp(
        budget_ledger=ledger,
        budget_policy=policy,
        secret_redactor=SecretRedactor(secret),
        enable_logging=False,
    )
    app.register_provider(runtime_provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        context_policy=CheckpointCompactionContextPolicy(
            compactor=ModelCompactor(
                provider=compactor_provider,
                model="fake-model",
            ),
            max_user_turns=1,
            compact_after_messages=2,
        ),
    )

    async def scenario() -> list[Event]:
        return [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="sess_compaction_redacted_settlement_pricing",
                    messages=[
                        Message.text("user", "old"),
                        Message.text("assistant", "old answer"),
                        Message.text("user", "current"),
                    ],
                )
            )
        ]

    events = asyncio.run(scenario())

    assert len(compactor_provider.requests) == 1
    assert len(runtime_provider.requests) == 1
    assert events[-1].type == EventType.SESSION_COMPLETED
    settlements = tuple(ledger._settlements.values())
    assert len(settlements) == 2
    assert all(settlement.event_published for settlement in settlements)
    assert all(
        settlement.event.payload["pricing"]["provenance"]["source"] == REDACTED_SECRET
        for settlement in settlements
    )


def test_recovery_settles_durable_model_completion_without_redispatch() -> None:
    async def scenario() -> None:
        session_id = "sess_model_budget_reconcile_recovery"
        store = InMemorySessionStore()
        clock = _MutableClock(datetime(2026, 7, 29, tzinfo=UTC))
        ledger = _CrashBeforeFirstReconciliation(clock=clock)
        provider = _CompletedProvider()
        await _run_until_process_loss(_app(store, ledger, provider), session_id)

        records = tuple(ledger._records.values())
        assert len(records) == 1
        assert records[0].status == "active"
        assert records[0].dispatch_id is not None
        assert await _budget_events(store, session_id) == []

        clock.value += timedelta(seconds=2)
        async for _event in _app(store, ledger, provider).run(
            RunRequest(
                agent_name="assistant",
                session_id="sess_model_budget_blocked_while_unsettled",
                messages=[Message.text("user", "must not dispatch")],
            )
        ):
            pass
        assert len(provider.requests) == 1

        await _app(store, ledger, provider).recover_incomplete_session(
            IncompleteSessionRecoveryRequest(session_id=session_id)
        )

        assert len(provider.requests) == 1
        settled = next(iter(ledger._records.values()))
        assert settled.status == "reconciled"
        assert settled.actual_amount == Decimal("0.000037")
        budget_events = await _budget_events(store, session_id)
        assert len(budget_events) == 1
        assert budget_events[0].type == EventType.BUDGET_RECONCILED
        assert budget_events[0].payload["settlement_kind"] == "completed"

    asyncio.run(scenario())


def test_recovery_reconstructs_event_after_ledger_commit_without_redispatch() -> None:
    async def scenario() -> None:
        session_id = "sess_model_budget_event_recovery"
        store = _CrashBeforeFirstSettlementEventStore()
        ledger = InMemoryBudgetLedger(reservation_ttl_seconds=1)
        provider = _CompletedProvider()
        await _run_until_process_loss(_app(store, ledger, provider), session_id)

        record = next(iter(ledger._records.values()))
        assert record.status == "reconciled"
        settlement = next(iter(ledger._settlements.values()))
        assert settlement.event_published is False
        assert await _budget_events(store, session_id) == []

        await _app(store, ledger, provider).recover_incomplete_session(
            IncompleteSessionRecoveryRequest(session_id=session_id)
        )

        assert len(provider.requests) == 1
        budget_events = await _budget_events(store, session_id)
        assert budget_events == [settlement.event]
        recovered = await ledger.load_settlement(settlement.settlement_id)
        assert recovered is not None
        assert recovered.event_published is True

    asyncio.run(scenario())


def test_recovery_handles_lost_ledger_acknowledgement_without_redispatch() -> None:
    async def scenario() -> None:
        session_id = "sess_model_budget_ledger_ack_loss"
        store = InMemorySessionStore()
        ledger = _LoseFirstReconciliationAcknowledgement()
        provider = _CompletedProvider()
        await _run_until_process_loss(_app(store, ledger, provider), session_id)

        settlement = next(iter(ledger._settlements.values()))
        assert settlement.event_published is False

        await _app(store, ledger, provider).recover_incomplete_session(
            IncompleteSessionRecoveryRequest(session_id=session_id)
        )

        assert len(provider.requests) == 1
        assert await _budget_events(store, session_id) == [settlement.event]
        recovered = await ledger.load_settlement(settlement.settlement_id)
        assert recovered is not None
        assert recovered.event_published is True

    asyncio.run(scenario())


def test_exact_event_replay_handles_lost_session_store_acknowledgement() -> None:
    async def scenario() -> None:
        session_id = "sess_model_budget_event_ack_loss"
        store = _LoseFirstSettlementEventAcknowledgementStore()
        ledger = InMemoryBudgetLedger()
        provider = _CompletedProvider()

        async for _event in _app(store, ledger, provider).run(
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "answer once")],
            )
        ):
            pass

        assert len(provider.requests) == 1
        budget_events = await _budget_events(store, session_id)
        assert len(budget_events) == 1
        settlement = next(iter(ledger._settlements.values()))
        assert budget_events == [settlement.event]
        assert settlement.event_published is True

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "ledger_type",
    [_MutatedSettlementLoadLedger, _MutatedSettlementAcknowledgementLedger],
)
def test_runtime_revalidates_custom_ledger_settlements_before_publication(
    ledger_type,
) -> None:
    async def scenario() -> None:
        session_id = f"sess_untrusted_{ledger_type.__name__.lower()}"
        store = InMemorySessionStore()
        ledger = ledger_type()
        provider = _CompletedProvider()

        events = [
            event
            async for event in _app(store, ledger, provider).run(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "reject forged accounting")],
                )
            )
        ]

        assert len(provider.requests) == 1
        assert events[-1].type == EventType.SESSION_FAILED
        persisted = await _budget_events(store, session_id)
        assert all(event.payload.get("actual_amount") != "999" for event in persisted)
        committed = next(iter(ledger._settlements.values()))
        assert committed.event.payload["actual_amount"] == "0.000037"

    asyncio.run(scenario())


def test_runtime_revalidates_custom_ledger_pending_pages_before_publication() -> None:
    async def scenario() -> None:
        store = InMemorySessionStore()
        ledger = _MutatedSettlementPageLedger()
        owner_session_id = "sess_mutated_pending_settlement_page_owner"
        await _seed_pending_release(
            store,
            ledger,
            session_id=owner_session_id,
        )

        provider = _CompletedProvider()
        events = [
            event
            async for event in _app(store, ledger, provider).run(
                RunRequest(
                    agent_name="assistant",
                    session_id="sess_mutated_pending_settlement_page_trigger",
                    messages=[Message.text("user", "reject the forged page")],
                )
            )
        ]

        assert len(provider.requests) == 0
        assert events[-1].type == EventType.SESSION_FAILED
        assert await _budget_events(store, owner_session_id) == []
        committed = next(iter(ledger._settlements.values()))
        assert committed.event.payload["actual_amount"] is None

    asyncio.run(scenario())


def test_session_deletion_waits_for_reachable_budget_outbox_publication() -> None:
    async def scenario() -> None:
        store = InMemorySessionStore()
        ledger = InMemoryBudgetLedger()
        deleted_session_id = "sess_deleted_pending_budget_outbox"
        expected_event = await _seed_pending_release(
            store,
            ledger,
            session_id=deleted_session_id,
        )
        with pytest.raises(
            ValueError,
            match="budget settlement audit event is pending",
        ):
            await store.delete_session(deleted_session_id)

        provider = _CompletedProvider()
        events = [
            event
            async for event in _app(store, ledger, provider).run(
                RunRequest(
                    agent_name="assistant",
                    session_id="sess_after_deleted_budget_outbox",
                    messages=[Message.text("user", "must still dispatch")],
                )
            )
        ]

        assert len(provider.requests) == 1
        assert events[-1].type != EventType.SESSION_FAILED
        assert await _budget_events(store, deleted_session_id) == [expected_event]
        assert await ledger.list_pending_settlements(session_id=deleted_session_id) == []

        await store.delete_session(deleted_session_id)
        assert await store.load(deleted_session_id) is None

    asyncio.run(scenario())


def test_shared_ledger_routes_pending_outbox_to_its_session_store_owner() -> None:
    async def scenario() -> None:
        ledger = InMemoryBudgetLedger()
        owner_store = InMemorySessionStore()
        owner_session_id = "sess_foreign_pending_budget_outbox"
        expected_event = await _seed_pending_release(
            owner_store,
            ledger,
            session_id=owner_session_id,
        )

        foreign_provider = _CompletedProvider()
        foreign_store = InMemorySessionStore()
        async for _event in _app(foreign_store, ledger, foreign_provider).run(
            RunRequest(
                agent_name="assistant",
                session_id="sess_foreign_budget_worker",
                messages=[Message.text("user", "do not claim another store's outbox")],
            )
        ):
            pass
        assert len(foreign_provider.requests) == 1
        assert await _budget_events(owner_store, owner_session_id) == []

        owner_provider = _CompletedProvider()
        owner_limit = (
            _budget_policy()
            .limits[0]
            .model_copy(
                update={"max_estimated_cost": Decimal("2")},
                deep=True,
            )
        )
        owner_app = CayuApp(
            session_store=owner_store,
            budget_ledger=ledger,
            budget_policy=BudgetPolicy(limits=(owner_limit,)),
            enable_logging=False,
        )
        owner_app.register_provider(owner_provider, default=True)
        owner_app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        async for _event in owner_app.run(
            RunRequest(
                agent_name="assistant",
                session_id="sess_owner_budget_worker",
                messages=[Message.text("user", "publish the reachable outbox")],
            )
        ):
            pass

        assert len(owner_provider.requests) == 1
        assert await _budget_events(owner_store, owner_session_id) == [expected_event]
        assert await ledger.list_pending_settlements(session_id=owner_session_id) == []

    asyncio.run(scenario())


def test_unresolved_dispatch_fence_acknowledgement_retains_recovery_stage() -> None:
    async def scenario() -> None:
        session_id = "sess_model_budget_dispatch_fence_unresolved"
        store = InMemorySessionStore()
        ledger = _LoseEveryDispatchFenceAcknowledgement()
        provider = _CompletedProvider()
        app = _app(store, ledger, provider)

        async for _event in app.run(
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "do not lose provenance")],
            )
        ):
            pass

        assert len(provider.requests) == 0
        active = await store.load_active_model_completion_stage(session_id)
        assert active is not None
        record = next(iter(ledger._records.values()))
        assert record.status == "active"
        assert record.dispatch_id == active.stage.stage_id
        assert ledger._settlements == {}

        async for _event in app.run(
            RunRequest(
                agent_name="assistant",
                session_id="sess_model_budget_blocked_by_unresolved_fence",
                messages=[Message.text("user", "must remain blocked")],
            )
        ):
            pass
        assert len(provider.requests) == 0
        assert await store.load_active_model_completion_stage(session_id) == active

    asyncio.run(scenario())


def test_automatic_compaction_rejects_rewritten_settlement_classification() -> None:
    class _RewritingSettlementLedger(InMemoryBudgetLedger):
        async def reconcile(self, **kwargs):
            reconciliation = await super().reconcile(**kwargs)
            return reconciliation.model_copy(update={"settlement_kind": "conservative"})

    compactor_provider = _CompletedProvider()
    runtime_provider = _CompletedProvider()
    ledger = _RewritingSettlementLedger()
    app = CayuApp(
        budget_ledger=ledger,
        budget_policy=_budget_policy(),
        enable_logging=False,
    )
    app.register_provider(runtime_provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        context_policy=CheckpointCompactionContextPolicy(
            compactor=ModelCompactor(
                provider=compactor_provider,
                model="fake-model",
            ),
            max_user_turns=1,
            compact_after_messages=2,
        ),
    )

    async def scenario():
        return [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="sess_compaction_rewritten_settlement_kind",
                    messages=[
                        Message.text("user", "old"),
                        Message.text("assistant", "old answer"),
                        Message.text("user", "current"),
                    ],
                )
            )
        ]

    events = asyncio.run(scenario())

    assert len(compactor_provider.requests) == 1
    assert len(runtime_provider.requests) == 0
    assert EventType.BUDGET_RECONCILED not in {event.type for event in events}
    assert events[-1].type == EventType.SESSION_FAILED
    assert events[-1].payload["error"] == (
        "Budget ledger settlement changed its requested outcome."
    )


def test_concurrent_recovery_settles_once_without_redispatch() -> None:
    async def scenario() -> None:
        session_id = "sess_model_budget_concurrent_recovery"
        store = InMemorySessionStore()
        ledger = _CrashBeforeFirstReconciliation()
        provider = _CompletedProvider()
        await _run_until_process_loss(_app(store, ledger, provider), session_id)

        results = await asyncio.gather(
            _app(store, ledger, provider).recover_incomplete_session(
                IncompleteSessionRecoveryRequest(session_id=session_id)
            ),
            _app(store, ledger, provider).recover_incomplete_session(
                IncompleteSessionRecoveryRequest(session_id=session_id)
            ),
            return_exceptions=True,
        )

        assert all(not isinstance(result, BaseException) for result in results), results
        assert len(provider.requests) == 1
        budget_events = await _budget_events(store, session_id)
        assert len(budget_events) == 1
        assert len(ledger._settlements) == 1

    asyncio.run(scenario())

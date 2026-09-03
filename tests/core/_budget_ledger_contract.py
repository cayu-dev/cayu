from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Protocol
from unittest.mock import patch
from uuid import UUID

import pytest
from tests.core._execution_unit_fixtures import model_attempt_identity

from cayu._validation import MAX_DURABLE_JSON_INTEGER, DurableValueError
from cayu.core import AgentSpec, EventType, Message
from cayu.core.billing import BillingIdentity, PricingContext
from cayu.providers import ModelProvider, ModelRequest, ModelStreamEvent
from cayu.runtime import (
    BudgetLedger,
    BudgetLimit,
    BudgetPolicy,
    BudgetReservation,
    BudgetSettlementCursor,
    BudgetSettlementFallback,
    CayuApp,
    InMemorySessionStore,
    ModelPrice,
    PriceBook,
    RunRequest,
)
from cayu.runtime._event_writer import RuntimeEventWriter
from cayu.runtime._run_limits import RunLimitController
from cayu.runtime.budgets import (
    BudgetReservationIdentityConflict,
    BudgetReservationRecoveryContext,
    SessionBudgetStore,
    budget_reservation_authority_sha256,
    budget_settlement_id,
)
from cayu.runtime.costs import ContextualPricingRequirement
from cayu.runtime.sessions import SessionIdentity, SessionStatus
from cayu.vaults import REDACTED_SECRET


class MutableClock(Protocol):
    value: datetime

    def __call__(self) -> datetime: ...


async def assert_runtime_manual_model_recovery_settles_dispatched_reservation(
    ledger: BudgetLedger,
    limit: BudgetLimit,
) -> None:
    """Exercise the runtime-owned conservative settlement path against one ledger."""

    store = InMemorySessionStore()
    session = await store.create(
        RunRequest(
            agent_name="assistant",
            session_id="sess_manual_model_recovery_ledger_contract",
            messages=[Message.text("user", "hello")],
        ),
        identity=SessionIdentity(provider_name="fake", model="fake-model"),
    )
    session = await store.update_status(session.id, SessionStatus.RUNNING)
    attempt_identity = model_attempt_identity()
    reserved = await ledger.reserve(
        limit=limit,
        session_id=session.id,
        agent_name=session.agent_name,
        provider_name="fake",
        model=session.model,
        model_attempt_identity=attempt_identity,
        settlement_event_payload={
            "interaction_id": "interaction_manual_model_recovery_ledger_contract"
        },
    )
    assert reserved.accepted is True
    assert reserved.record is not None
    dispatch_id = "model-stage:manual-model-recovery-ledger-contract"
    await ledger.mark_dispatched(
        reservation_ids=(reserved.record.reservation_id,),
        dispatch_id=dispatch_id,
    )
    budget_store = SessionBudgetStore(store)
    controller = RunLimitController(
        session_store=store,
        budget_store=budget_store,
        budget_ledger=ledger,
        event_writer=RuntimeEventWriter(
            session_store=store,
            budget_store=budget_store,
            event_sinks=(),
        ),
        clock=lambda: reserved.record.created_at + timedelta(seconds=1),
    )
    context = BudgetReservationRecoveryContext(
        reservation_id=reserved.record.reservation_id,
        budget_limit_id=reserved.record.budget_limit_id,
        reservation_authority_sha256=budget_reservation_authority_sha256(reserved.record),
    )

    events = await controller.reconcile_manual_model_completion_reservations(
        reservation_ids=(reserved.record.reservation_id,),
        recovery_contexts=(context,),
        session=session,
        provider_name="fake",
        model=session.model,
        model_attempt_identity=attempt_identity,
        dispatch_id=dispatch_id,
    )
    await controller.require_model_completion_reservation_settlements(
        reservation_ids=(reserved.record.reservation_id,),
        recovery_contexts=(context,),
        dispatch_id=dispatch_id,
    )

    assert [event.type for event in events] == [EventType.BUDGET_RECONCILED]
    reconciled = await ledger.load_reservation(reserved.record.reservation_id)
    assert reconciled is not None
    assert reconciled.status == "reconciled"
    assert reconciled.actual_amount == reserved.record.reserved_amount
    settlement = await ledger.load_settlement(budget_settlement_id(reserved.record.reservation_id))
    assert settlement is not None
    assert settlement.settlement_kind == "conservative"
    assert settlement.event_published is True


async def assert_prepriced_reservation_stores_only_durable_billing_identity(
    ledger: BudgetLedger,
) -> None:
    """Accept a prepriced amount while persisting only normalized authority."""

    raw_identity = BillingIdentity(
        provider_name="fake",
        resource_id="fake-model",
        request_evidence={"opaque": "raw-workload-value"},
        pricing_contexts=(PricingContext(dimensions={"billing_tenant": "raw-workload-value"}),),
    )
    durable_identity = raw_identity.model_copy(
        update={
            "request_evidence": {"opaque": REDACTED_SECRET},
            "pricing_contexts": (PricingContext(dimensions={"billing_tenant": REDACTED_SECRET}),),
        },
        deep=True,
    )
    limit = BudgetLimit(
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
                    output_per_million=Decimal("0"),
                    pricing_context={"billing_tenant": ("raw-workload-value",)},
                ),
            ),
        ),
        reservation=BudgetReservation(
            max_input_tokens=1_000_000,
            max_output_tokens=0,
        ),
    )
    result = await ledger.reserve(
        limit=limit,
        session_id="sess_transient_pricing_identity",
        agent_name="assistant",
        provider_name="fake",
        model="fake-model",
        model_attempt_identity=model_attempt_identity(),
        requested_amount=Decimal("1"),
        billing_identity=durable_identity,
    )

    assert result.accepted is True
    assert result.requested == Decimal("1")
    assert result.record is not None
    assert result.record.billing_identity == durable_identity
    assert "raw-workload-value" not in result.record.model_dump_json()


class _CompletedProvider(ModelProvider):
    name = "fake"

    def __init__(self) -> None:
        self.requests = 0

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        del request
        self.requests += 1
        yield ModelStreamEvent.text_delta("done")
        yield ModelStreamEvent.completed(
            {"usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}}
        )


class _DelegatingBudgetLedger(BudgetLedger):
    def __init__(self, ledger: BudgetLedger) -> None:
        self.ledger = ledger

    @property
    def reservation_ttl_seconds(self) -> int | None:
        return self.ledger.reservation_ttl_seconds

    async def claim_reservation_identity(self, **kwargs) -> None:
        await self.ledger.claim_reservation_identity(**kwargs)

    async def reserve(self, **kwargs):
        return await self.ledger.reserve(**kwargs)

    async def heartbeat(self, **kwargs):
        return await self.ledger.heartbeat(**kwargs)

    async def mark_dispatched(self, **kwargs):
        return await self.ledger.mark_dispatched(**kwargs)

    async def release_pre_provider_dispatch(self, **kwargs):
        return await self.ledger.release_pre_provider_dispatch(**kwargs)

    async def reconcile(self, **kwargs):
        return await self.ledger.reconcile(**kwargs)

    async def release(self, **kwargs):
        return await self.ledger.release(**kwargs)

    async def load_reservation(self, reservation_id):
        return await self.ledger.load_reservation(reservation_id)

    async def load_settlement(self, settlement_id):
        return await self.ledger.load_settlement(settlement_id)

    async def list_pending_settlements(self, **kwargs):
        return await self.ledger.list_pending_settlements(**kwargs)

    async def mark_settlement_event_published(self, **kwargs):
        return await self.ledger.mark_settlement_event_published(**kwargs)


class _LoseFirstDispatchAcknowledgement(_DelegatingBudgetLedger):
    def __init__(self, ledger: BudgetLedger) -> None:
        super().__init__(ledger)
        self.mark_dispatched_calls = 0
        self.dispatched_reservation_ids: tuple[str, ...] = ()

    async def mark_dispatched(self, **kwargs):
        records = await self.ledger.mark_dispatched(**kwargs)
        self.mark_dispatched_calls += 1
        self.dispatched_reservation_ids = tuple(record.reservation_id for record in records)
        if self.mark_dispatched_calls == 1:
            raise RuntimeError("dispatch fence acknowledgement lost after commit")
        return records


class _LoseFirstReservationAcknowledgement(_DelegatingBudgetLedger):
    def __init__(self, ledger: BudgetLedger) -> None:
        super().__init__(ledger)
        self.lose_acknowledgement = True
        self.lost_reservation_id: str | None = None

    async def reserve(self, **kwargs):
        result = await self.ledger.reserve(**kwargs)
        if self.lose_acknowledgement:
            self.lose_acknowledgement = False
            assert result.record is not None
            self.lost_reservation_id = result.record.reservation_id
            raise RuntimeError("reservation acknowledgement lost after commit")
        return result


async def assert_load_reservation_reconstructs_exact_record(
    ledger: BudgetLedger,
    limit: BudgetLimit,
) -> None:
    """Load one exact active and reconciled reservation by durable identity."""

    result = await ledger.reserve(
        limit=limit,
        session_id="sess_provider_operation_reservation_recovery",
        agent_name="assistant",
        provider_name="fake",
        model="fake-model",
        model_attempt_identity=model_attempt_identity(),
    )
    assert result.accepted is True
    assert result.record is not None
    loaded = await ledger.load_reservation(result.record.reservation_id)
    assert loaded == result.record
    assert loaded is not result.record

    await ledger.reconcile(
        reservation_id=result.record.reservation_id,
        actual_amount=Decimal("0.01"),
        reason="provider operation recovered",
    )
    reconciled = await ledger.load_reservation(result.record.reservation_id)
    assert reconciled is not None
    assert reconciled.status == "reconciled"
    assert reconciled.actual_amount == Decimal("0.01")
    assert await ledger.load_reservation("missing-provider-operation-reservation") is None


async def assert_runtime_reconstructs_dispatch_fence_acknowledgement(
    ledger: BudgetLedger,
    limit: BudgetLimit,
) -> None:
    """Prove runtime replay against one concrete ledger implementation."""

    wrapped = _LoseFirstDispatchAcknowledgement(ledger)
    provider = _CompletedProvider()
    store = InMemorySessionStore()
    app = CayuApp(
        session_store=store,
        budget_ledger=wrapped,
        budget_policy=BudgetPolicy(limits=(limit,)),
        enable_logging=False,
    )
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    events = [
        event
        async for event in app.run(
            RunRequest(
                agent_name="assistant",
                session_id="sess_dispatch_fence_ack_reconstruction",
                messages=[Message.text("user", "run exactly once")],
            )
        )
    ]

    assert wrapped.mark_dispatched_calls == 2
    assert provider.requests == 1
    assert len(wrapped.dispatched_reservation_ids) == 1
    settlement = await ledger.load_settlement(
        budget_settlement_id(wrapped.dispatched_reservation_ids[0])
    )
    assert settlement is not None
    assert settlement.reconciliation.status == "reconciled"
    assert settlement.event_published is True
    assert (
        await store.load_active_model_completion_stage("sess_dispatch_fence_ack_reconstruction")
        is None
    )
    assert sum(event.type == EventType.BUDGET_RECONCILED for event in events) == 1


async def assert_runtime_publishes_cross_session_ttl_release(
    ledger: BudgetLedger,
    limit: BudgetLimit,
    *,
    clock: MutableClock,
    ttl_seconds: int,
) -> None:
    """Prove that one session can publish another session's reaped release."""

    wrapped = _LoseFirstReservationAcknowledgement(ledger)
    provider = _CompletedProvider()
    store = InMemorySessionStore()
    app = CayuApp(
        session_store=store,
        budget_ledger=wrapped,
        budget_policy=BudgetPolicy(limits=(limit,)),
        clock=clock,
        enable_logging=False,
    )
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    abandoned_session_id = "sess_cross_session_ttl_release_source"
    async for _event in app.run(
        RunRequest(
            agent_name="assistant",
            session_id=abandoned_session_id,
            messages=[Message.text("user", "fail after reservation commit")],
        )
    ):
        pass
    abandoned = await store.load(abandoned_session_id)
    assert abandoned is not None
    assert abandoned.status.value == "failed"
    assert provider.requests == 0
    assert wrapped.lost_reservation_id is not None

    clock.value += timedelta(seconds=ttl_seconds + 1)
    async for _event in app.run(
        RunRequest(
            agent_name="assistant",
            session_id="sess_cross_session_ttl_release_trigger",
            messages=[Message.text("user", "trigger the reap")],
        )
    ):
        pass

    assert provider.requests == 1
    settlement = await ledger.load_settlement(budget_settlement_id(wrapped.lost_reservation_id))
    assert settlement is not None
    assert settlement.reconciliation.status == "released"
    assert settlement.event_published is True
    source_events = await store.load_events(abandoned_session_id)
    assert [
        event for event in source_events if event.type == EventType.BUDGET_RESERVATION_RELEASED
    ] == [settlement.event]
    assert await ledger.list_pending_settlements(session_id=abandoned_session_id) == []


async def assert_reservation_identity_collision_is_rejected(
    ledger: BudgetLedger,
    limit: BudgetLimit,
) -> None:
    first_identity = model_attempt_identity()
    second_identity = model_attempt_identity()
    fixed_uuid = UUID("11111111-1111-1111-1111-111111111111")
    reserve_kwargs = {
        "limit": limit,
        "agent_name": "assistant",
        "provider_name": "fake",
        "model": "fake-model",
    }

    with patch("cayu.runtime.budgets.uuid4", return_value=fixed_uuid):
        first = await ledger.reserve(
            session_id="sess_reservation_identity_winner",
            model_attempt_identity=first_identity,
            **reserve_kwargs,
        )
        assert first.accepted is True
        assert first.record is not None

        with pytest.raises(
            BudgetReservationIdentityConflict,
            match="reused a reservation identity",
        ):
            await ledger.reserve(
                session_id="sess_reservation_identity_collision",
                model_attempt_identity=second_identity,
                **reserve_kwargs,
            )

    reconciled = await ledger.reconcile(
        reservation_id=first.record.reservation_id,
        actual_amount=Decimal("0.01"),
        reason="winner retained after collision",
    )
    assert (reconciled.model_step_id, reconciled.model_attempt_id) == (
        first_identity.model_step_id,
        first_identity.model_attempt_id,
    )


async def assert_portable_text_boundaries(
    ledger: BudgetLedger,
    limit: BudgetLimit,
) -> None:
    for invalid_text, code in (
        ("workload-secret-value\x00", "nul_character"),
        ("workload-secret-value\ud800", "unicode_surrogate"),
    ):
        with pytest.raises(DurableValueError) as invalid_reservation:
            await ledger.reserve(
                limit=limit,
                session_id=invalid_text,
                agent_name="assistant",
                provider_name="fake",
                model="fake-model",
                model_attempt_identity=model_attempt_identity(),
            )
        assert invalid_reservation.value.code == code
        assert "workload-secret-value" not in str(invalid_reservation.value)

    reserved = await ledger.reserve(
        limit=limit,
        session_id="sess_portable_text",
        agent_name="assistant",
        provider_name="fake",
        model="fake-model",
        model_attempt_identity=model_attempt_identity(),
    )
    assert reserved.accepted is True
    assert reserved.record is not None

    for invalid_text, code in (
        ("workload-secret-value\x00", "nul_character"),
        ("workload-secret-value\ud800", "unicode_surrogate"),
    ):
        with pytest.raises(DurableValueError) as invalid_release:
            await ledger.release(
                reservation_id=reserved.record.reservation_id,
                reason=invalid_text,
            )
        assert invalid_release.value.code == code
        assert "workload-secret-value" not in str(invalid_release.value)
        assert await ledger.heartbeat(reservation_id=reserved.record.reservation_id) is True

    released = await ledger.release(
        reservation_id=reserved.record.reservation_id,
        reason="portable boundary verified",
    )
    assert released.status == "released"


async def assert_idempotent_terminal_settlements(
    ledger: BudgetLedger,
    limit: BudgetLimit,
    *,
    clock: MutableClock,
) -> None:
    first_identity = model_attempt_identity()
    first = await ledger.reserve(
        limit=limit,
        session_id="sess_idempotent_reconcile",
        agent_name="assistant",
        provider_name="fake",
        model="fake-model",
        model_attempt_identity=first_identity,
    )
    assert first.accepted is True
    assert first.record is not None
    assert (first.model_step_id, first.model_attempt_id) == (
        first_identity.model_step_id,
        first_identity.model_attempt_id,
    )
    assert (first.record.model_step_id, first.record.model_attempt_id) == (
        first_identity.model_step_id,
        first_identity.model_attempt_id,
    )

    occurred_at = clock.value
    reconciled, concurrent_reconciliation = await asyncio.gather(
        ledger.reconcile(
            reservation_id=first.record.reservation_id,
            actual_amount=Decimal("0.01"),
            reason="actual usage",
            occurred_at=occurred_at,
        ),
        ledger.reconcile(
            reservation_id=first.record.reservation_id,
            actual_amount=Decimal("0.01"),
            reason="actual usage",
            occurred_at=occurred_at,
        ),
    )
    assert concurrent_reconciliation == reconciled
    assert (reconciled.model_step_id, reconciled.model_attempt_id) == (
        first_identity.model_step_id,
        first_identity.model_attempt_id,
    )
    reconciliation_retry = await ledger.reconcile(
        reservation_id=first.record.reservation_id,
        actual_amount=Decimal("0.01"),
        reason="actual usage",
        occurred_at=occurred_at + timedelta(seconds=40),
    )
    assert reconciliation_retry == reconciled

    with pytest.raises(ValueError, match="conflicting reconciliation"):
        await ledger.reconcile(
            reservation_id=first.record.reservation_id,
            actual_amount=Decimal("0.02"),
            reason="different outcome",
        )
    with pytest.raises(ValueError, match="not active"):
        await ledger.release(
            reservation_id=first.record.reservation_id,
            reason="cannot release charged spend",
        )

    clock.value = occurred_at + timedelta(seconds=1)
    second_identity = model_attempt_identity()
    second = await ledger.reserve(
        limit=limit,
        session_id="sess_idempotent_release",
        agent_name="assistant",
        provider_name="fake",
        model="fake-model",
        model_attempt_identity=second_identity,
    )
    assert second.accepted is True
    assert second.actual == Decimal("0.23")
    assert second.record is not None
    assert (second.model_step_id, second.model_attempt_id) == (
        second_identity.model_step_id,
        second_identity.model_attempt_id,
    )

    released, concurrent_release = await asyncio.gather(
        ledger.release(
            reservation_id=second.record.reservation_id,
            reason="provider not dispatched",
        ),
        ledger.release(
            reservation_id=second.record.reservation_id,
            reason="provider not dispatched",
        ),
    )
    assert concurrent_release == released
    assert (released.model_step_id, released.model_attempt_id) == (
        second_identity.model_step_id,
        second_identity.model_attempt_id,
    )
    release_retry = await ledger.release(
        reservation_id=second.record.reservation_id,
        reason="provider not dispatched",
    )
    assert release_retry == released

    with pytest.raises(ValueError, match="conflicting release"):
        await ledger.release(
            reservation_id=second.record.reservation_id,
            reason="different outcome",
        )
    with pytest.raises(ValueError, match="not active"):
        await ledger.reconcile(
            reservation_id=second.record.reservation_id,
            actual_amount=Decimal("0.22"),
            reason="cannot charge explicitly released reservation",
        )

    # The original reconciliation is now outside the rolling window. If the
    # idempotent retry had moved its accounting timestamp forward, its charge
    # would still contribute to this admission result.
    clock.value = occurred_at + timedelta(seconds=90)
    third = await ledger.reserve(
        limit=limit,
        session_id="sess_after_idempotent_settlements",
        agent_name="assistant",
        provider_name="fake",
        model="fake-model",
        model_attempt_identity=model_attempt_identity(),
    )
    assert third.accepted is True
    assert third.actual == Decimal("0.22")


async def assert_crash_safe_dispatch_and_settlement_outbox(
    ledger: BudgetLedger,
    limit: BudgetLimit,
    *,
    clock: MutableClock,
    ttl_seconds: int,
) -> None:
    """Exercise the shared dispatch fence and terminal audit-outbox contract."""

    completion_identity = model_attempt_identity()
    completion_fallback = BudgetSettlementFallback(
        settled_at=clock.value,
        reconciliation_reason="prevalidated conservative completion",
        release_reason="prevalidated predispatch release",
        expiration_reason=f"prevalidated expiry after {ttl_seconds}s",
    )
    reserved = await ledger.reserve(
        limit=limit,
        session_id="sess_crash_safe_completion",
        agent_name="assistant",
        environment_name="sandbox",
        settlement_event_payload={"audit_context": "trusted"},
        provider_name="fake",
        model="fake-model",
        model_attempt_identity=completion_identity,
        settlement_fallback=completion_fallback,
    )
    assert reserved.accepted is True
    assert reserved.record is not None
    assert reserved.record.settlement_fallback == completion_fallback
    reserved.record.settlement_event_payload["forged"] = "caller mutation"
    dispatch_id = f"{completion_identity.model_step_id}:dispatch:0"
    dispatched_records = await ledger.mark_dispatched(
        reservation_ids=(reserved.record.reservation_id,),
        dispatch_id=dispatch_id,
        dispatched_at=clock.value,
    )
    assert len(dispatched_records) == 1
    dispatched = dispatched_records[0]
    assert dispatched.dispatch_id == dispatch_id
    assert dispatched.settlement_fallback == completion_fallback
    assert await ledger.mark_dispatched(
        reservation_ids=(reserved.record.reservation_id,),
        dispatch_id=dispatch_id,
        dispatched_at=clock.value + timedelta(seconds=1),
    ) == (dispatched,)
    with pytest.raises(ValueError, match="Dispatched budget reservation cannot be released"):
        await ledger.release(
            reservation_id=reserved.record.reservation_id,
            reason="must retain post-dispatch capacity",
        )

    clock.value += timedelta(seconds=ttl_seconds)
    blocked = await ledger.reserve(
        limit=limit,
        session_id="sess_crash_safe_blocked",
        agent_name="assistant",
        provider_name="fake",
        model="fake-model",
        model_attempt_identity=model_attempt_identity(),
    )
    assert blocked.accepted is False

    reconciled_at = clock.value
    reconciled, duplicate = await asyncio.gather(
        ledger.reconcile(
            reservation_id=reserved.record.reservation_id,
            actual_amount=Decimal("0.01"),
            settlement_kind="completed",
            reason="model completed",
            occurred_at=reconciled_at,
        ),
        ledger.reconcile(
            reservation_id=reserved.record.reservation_id,
            actual_amount=Decimal("0.01"),
            settlement_kind="completed",
            reason="model completed",
            occurred_at=reconciled_at,
        ),
    )
    assert duplicate == reconciled
    assert reconciled.settlement_kind == "completed"
    settlement = await ledger.load_settlement(reconciled.settlement_id)
    assert settlement is not None
    assert settlement.reconciliation == reconciled
    assert settlement.event.payload["settlement_id"] == reconciled.settlement_id
    assert settlement.event.payload["actual_amount"] == "0.01"
    assert settlement.event.payload["audit_context"] == "trusted"
    assert "forged" not in settlement.event.payload
    assert settlement.event.environment_name == "sandbox"
    assert settlement.event_published is False
    assert await ledger.list_pending_settlements(
        session_id=reserved.record.session_id,
    ) == [settlement]

    with pytest.raises(ValueError, match="conflicting settlement"):
        await ledger.reconcile(
            reservation_id=reserved.record.reservation_id,
            actual_amount=Decimal("0.01"),
            settlement_kind="conservative",
            reason="model completed",
            occurred_at=reconciled_at,
        )

    published = await ledger.mark_settlement_event_published(
        settlement_id=settlement.settlement_id,
        event_id=settlement.event.id,
    )
    assert published.event_published is True
    assert (
        await ledger.mark_settlement_event_published(
            settlement_id=settlement.settlement_id,
            event_id=settlement.event.id,
        )
        == published
    )
    assert await ledger.list_pending_settlements(session_id=reserved.record.session_id) == []

    expiration_fallback = BudgetSettlementFallback(
        settled_at=clock.value,
        reconciliation_reason="unused conservative fallback",
        release_reason="unused predispatch release",
        expiration_reason=f"prevalidated expiry after {ttl_seconds}s",
    )
    pending = await ledger.reserve(
        limit=limit,
        session_id="sess_crash_safe_predispatch",
        agent_name="assistant",
        provider_name="fake",
        model="fake-model",
        model_attempt_identity=model_attempt_identity(),
        settlement_fallback=expiration_fallback,
    )
    assert pending.accepted is True
    assert pending.record is not None
    clock.value += timedelta(seconds=ttl_seconds)
    replacement = await ledger.reserve(
        limit=limit,
        session_id="sess_crash_safe_replacement",
        agent_name="assistant",
        provider_name="fake",
        model="fake-model",
        model_attempt_identity=model_attempt_identity(),
    )
    assert replacement.accepted is True
    pending_releases = await ledger.list_pending_settlements(
        session_id=pending.record.session_id,
    )
    assert len(pending_releases) == 1, pending_releases
    assert await ledger.list_pending_settlements() == pending_releases
    assert pending_releases[0].settlement_kind == "released"
    assert pending_releases[0].reconciliation.reason == expiration_fallback.expiration_reason
    assert pending_releases[0].reconciliation.settled_at == clock.value

    audit_limit = limit.model_copy(
        update={"max_estimated_cost": Decimal("2")},
        deep=True,
    )
    conservative = await ledger.reserve(
        limit=audit_limit,
        session_id="sess_conservative_settlement",
        agent_name="assistant",
        provider_name="fake",
        model="fake-model",
        model_attempt_identity=model_attempt_identity(),
    )
    assert conservative.record is not None
    await ledger.mark_dispatched(
        reservation_ids=(conservative.record.reservation_id,),
        dispatch_id="dispatch:conservative",
    )
    conservative_reconciliation = await ledger.reconcile(
        reservation_id=conservative.record.reservation_id,
        actual_amount=conservative.record.reserved_amount,
        settlement_kind="conservative",
        reason="provider usage unknown after dispatch; charged reserved amount",
        occurred_at=clock.value,
    )
    assert (
        await ledger.reconcile(
            reservation_id=conservative.record.reservation_id,
            actual_amount=conservative.record.reserved_amount,
            settlement_kind="conservative",
            reason="provider usage unknown after dispatch; charged reserved amount",
            occurred_at=clock.value + timedelta(seconds=1),
        )
        == conservative_reconciliation
    )
    conservative_settlement = await ledger.load_settlement(
        conservative_reconciliation.settlement_id
    )
    assert conservative_settlement is not None
    assert conservative_settlement.settlement_kind == "conservative"
    assert conservative_settlement.event_published is False

    pre_provider_identity = model_attempt_identity()
    pre_provider_first = await ledger.reserve(
        limit=audit_limit,
        session_id="sess_pre_provider_release",
        agent_name="assistant",
        provider_name="fake",
        model="fake-model",
        model_attempt_identity=pre_provider_identity,
    )
    pre_provider_second = await ledger.reserve(
        limit=audit_limit,
        session_id="sess_pre_provider_release",
        agent_name="assistant",
        provider_name="fake",
        model="fake-model",
        model_attempt_identity=pre_provider_identity,
    )
    assert pre_provider_first.record is not None
    assert pre_provider_second.record is not None
    pre_provider_dispatch_id = "dispatch:pre-provider-release"
    await ledger.mark_dispatched(
        reservation_ids=(pre_provider_first.record.reservation_id,),
        dispatch_id=pre_provider_dispatch_id,
        dispatched_at=clock.value,
    )
    pre_provider_ids = (
        pre_provider_first.record.reservation_id,
        pre_provider_second.record.reservation_id,
    )
    pre_provider_releases = await ledger.release_pre_provider_dispatch(
        reservation_ids=pre_provider_ids,
        dispatch_id=pre_provider_dispatch_id,
        reason="provider dispatch barrier was not crossed",
        occurred_at=clock.value,
    )
    assert tuple(item.reservation_id for item in pre_provider_releases) == pre_provider_ids
    assert all(item.status == "released" for item in pre_provider_releases)
    assert all(item.settlement_kind == "released" for item in pre_provider_releases)
    assert (
        await ledger.release_pre_provider_dispatch(
            reservation_ids=pre_provider_ids,
            dispatch_id=pre_provider_dispatch_id,
            reason="provider dispatch barrier was not crossed",
            occurred_at=clock.value + timedelta(seconds=1),
        )
        == pre_provider_releases
    )

    conflicting_terminal = await ledger.reserve(
        limit=audit_limit,
        session_id="sess_pre_provider_terminal_conflict",
        agent_name="assistant",
        provider_name="fake",
        model="fake-model",
        model_attempt_identity=model_attempt_identity(),
    )
    assert conflicting_terminal.record is not None
    ordinary_release = await ledger.release(
        reservation_id=conflicting_terminal.record.reservation_id,
        reason="ordinary unrelated release",
        occurred_at=clock.value,
    )
    with pytest.raises(ValueError, match="conflicting release"):
        await ledger.release_pre_provider_dispatch(
            reservation_ids=(conflicting_terminal.record.reservation_id,),
            dispatch_id="dispatch:different-stage",
            reason="provider dispatch barrier was not crossed",
            occurred_at=clock.value + timedelta(seconds=1),
        )
    unchanged_settlement = await ledger.load_settlement(ordinary_release.settlement_id)
    assert unchanged_settlement is not None
    assert unchanged_settlement.reconciliation == ordinary_release

    conflicting_first = await ledger.reserve(
        limit=audit_limit,
        session_id="sess_pre_provider_release_conflict",
        agent_name="assistant",
        provider_name="fake",
        model="fake-model",
        model_attempt_identity=model_attempt_identity(),
    )
    conflicting_second = await ledger.reserve(
        limit=audit_limit,
        session_id="sess_pre_provider_release_conflict",
        agent_name="assistant",
        provider_name="fake",
        model="fake-model",
        model_attempt_identity=model_attempt_identity(),
    )
    assert conflicting_first.record is not None
    assert conflicting_second.record is not None
    await ledger.mark_dispatched(
        reservation_ids=(conflicting_first.record.reservation_id,),
        dispatch_id="dispatch:conflicting-owner",
    )
    with pytest.raises(ValueError, match="conflicting dispatch"):
        await ledger.release_pre_provider_dispatch(
            reservation_ids=(
                conflicting_first.record.reservation_id,
                conflicting_second.record.reservation_id,
            ),
            dispatch_id="dispatch:wrong-owner",
            reason="must fail atomically",
        )
    still_active = await ledger.load_reservation(conflicting_second.record.reservation_id)
    assert still_active is not None and still_active.status == "active"
    assert (
        await ledger.load_settlement(budget_settlement_id(conflicting_second.record.reservation_id))
        is None
    )

    releasable = await ledger.reserve(
        limit=audit_limit,
        session_id="sess_release_settlement",
        agent_name="assistant",
        provider_name="fake",
        model="fake-model",
        model_attempt_identity=model_attempt_identity(),
    )
    assert releasable.record is not None
    released = await ledger.release(
        reservation_id=releasable.record.reservation_id,
        reason="provider not dispatched",
    )
    assert (
        await ledger.release(
            reservation_id=releasable.record.reservation_id,
            reason="provider not dispatched",
        )
        == released
    )
    release_settlement = await ledger.load_settlement(released.settlement_id)
    assert release_settlement is not None
    assert release_settlement.settlement_kind == "released"
    assert release_settlement.event_published is False

    expected_pending = await ledger.list_pending_settlements()
    paged_pending = []
    cursor = None
    while True:
        page = await ledger.list_pending_settlements(after=cursor, limit=1)
        if not page:
            break
        assert len(page) == 1
        paged_pending.extend(page)
        cursor = BudgetSettlementCursor(
            settled_at=page[0].reconciliation.settled_at,
            settlement_id=page[0].settlement_id,
        )
    assert paged_pending == expected_pending

    atomic_limit = limit.model_copy(
        update={"max_estimated_cost": Decimal("1")},
        deep=True,
    )
    first = await ledger.reserve(
        limit=atomic_limit,
        session_id="sess_atomic_dispatch_first",
        agent_name="assistant",
        provider_name="fake",
        model="fake-model",
        model_attempt_identity=model_attempt_identity(),
    )
    second = await ledger.reserve(
        limit=atomic_limit,
        session_id="sess_atomic_dispatch_second",
        agent_name="assistant",
        provider_name="fake",
        model="fake-model",
        model_attempt_identity=model_attempt_identity(),
    )
    assert first.record is not None
    assert second.record is not None
    await ledger.mark_dispatched(
        reservation_ids=(first.record.reservation_id,),
        dispatch_id="dispatch:atomic:first",
    )
    with pytest.raises(ValueError, match="conflicting dispatch"):
        await ledger.mark_dispatched(
            reservation_ids=(
                first.record.reservation_id,
                second.record.reservation_id,
            ),
            dispatch_id="dispatch:atomic:combined",
        )
    clock.value += timedelta(seconds=ttl_seconds)
    trigger_reap = await ledger.reserve(
        limit=atomic_limit,
        session_id="sess_atomic_dispatch_trigger",
        agent_name="assistant",
        provider_name="fake",
        model="fake-model",
        model_attempt_identity=model_attempt_identity(),
    )
    assert trigger_reap.accepted is True
    second_releases = await ledger.list_pending_settlements(
        session_id=second.record.session_id,
    )
    assert len(second_releases) == 1
    assert second_releases[0].reservation_id == second.record.reservation_id


async def assert_budget_reservation_store_time_conformance(
    ledger: BudgetLedger,
    limit: BudgetLimit,
    *,
    clock: MutableClock,
    ttl_seconds: int,
    contender_ledger: BudgetLedger | None = None,
) -> None:
    """Exercise expiry, dispatch, heartbeat, reaping, and exact replay at store time."""

    expiring = await ledger.reserve(
        limit=limit,
        session_id="sess_store_time_expiring",
        agent_name="assistant",
        provider_name="fake",
        model="fake-model",
        model_attempt_identity=model_attempt_identity(),
    )
    assert expiring.record is not None

    replay_limit = limit.model_copy(
        update={"max_estimated_cost": Decimal("1")},
        deep=True,
    )
    replayable = await ledger.reserve(
        limit=replay_limit,
        session_id="sess_store_time_replay",
        agent_name="assistant",
        provider_name="fake",
        model="fake-model",
        model_attempt_identity=model_attempt_identity(),
    )
    assert replayable.record is not None
    dispatched = await ledger.mark_dispatched(
        reservation_ids=(replayable.record.reservation_id,),
        dispatch_id="dispatch:store-time-replay",
    )

    clock.value += timedelta(seconds=ttl_seconds)
    assert await ledger.heartbeat(reservation_id=expiring.record.reservation_id) is False
    with pytest.raises(ValueError, match="expired"):
        await ledger.mark_dispatched(
            reservation_ids=(expiring.record.reservation_id,),
            dispatch_id="dispatch:too-late",
        )
    assert (
        await ledger.mark_dispatched(
            reservation_ids=(replayable.record.reservation_id,),
            dispatch_id="dispatch:store-time-replay",
        )
        == dispatched
    )

    replacement = await ledger.reserve(
        limit=limit,
        session_id="sess_store_time_replacement",
        agent_name="assistant",
        provider_name="fake",
        model="fake-model",
        model_attempt_identity=model_attempt_identity(),
    )
    assert replacement.accepted is True
    settlements = await ledger.list_pending_settlements(
        session_id=expiring.record.session_id,
    )
    assert len(settlements) == 1
    assert settlements[0].reconciliation.settled_at == clock.value

    race_limit = limit.model_copy(
        update={"scope": "agent", "key": "store-time-race"},
        deep=True,
    )
    race_owner = await ledger.reserve(
        limit=race_limit,
        session_id="sess_store_time_race_owner",
        agent_name="assistant",
        provider_name="fake",
        model="fake-model",
        model_attempt_identity=model_attempt_identity(),
    )
    assert race_owner.record is not None

    clock.value += timedelta(seconds=ttl_seconds - 1)
    competing_ledger = contender_ledger or ledger
    live_heartbeat, live_competitor = await asyncio.gather(
        ledger.heartbeat(reservation_id=race_owner.record.reservation_id),
        competing_ledger.reserve(
            limit=race_limit,
            session_id="sess_store_time_race_live_competitor",
            agent_name="assistant",
            provider_name="fake",
            model="fake-model",
            model_attempt_identity=model_attempt_identity(),
        ),
    )
    assert live_heartbeat is True
    assert live_competitor.accepted is False

    clock.value += timedelta(seconds=ttl_seconds)
    expired_heartbeat, replacement_race = await asyncio.gather(
        ledger.heartbeat(reservation_id=race_owner.record.reservation_id),
        competing_ledger.reserve(
            limit=race_limit,
            session_id="sess_store_time_race_replacement",
            agent_name="assistant",
            provider_name="fake",
            model="fake-model",
            model_attempt_identity=model_attempt_identity(),
        ),
    )
    assert expired_heartbeat is False
    assert replacement_race.accepted is True
    race_settlements = await ledger.list_pending_settlements(
        session_id=race_owner.record.session_id,
    )
    assert len(race_settlements) == 1
    assert race_settlements[0].reservation_id == race_owner.record.reservation_id


async def assert_maximum_reservation_ttl_preserves_minimum_timestamp(
    ledger: BudgetLedger,
    limit: BudgetLimit,
) -> None:
    """A maximum TTL must not expire a record stamped at ``datetime.min``."""

    assert ledger.reservation_ttl_seconds == MAX_DURABLE_JSON_INTEGER
    owner = await ledger.reserve(
        limit=limit,
        session_id="sess_minimum_timestamp_owner",
        agent_name="assistant",
        provider_name="fake",
        model="fake-model",
        model_attempt_identity=model_attempt_identity(),
    )
    assert owner.record is not None
    assert owner.record.updated_at == datetime.min.replace(tzinfo=owner.record.updated_at.tzinfo)
    assert await ledger.heartbeat(reservation_id=owner.record.reservation_id) is True

    contender = await ledger.reserve(
        limit=limit,
        session_id="sess_minimum_timestamp_contender",
        agent_name="assistant",
        provider_name="fake",
        model="fake-model",
        model_attempt_identity=model_attempt_identity(),
    )
    assert contender.accepted is False
    assert await ledger.list_pending_settlements(session_id=owner.record.session_id) == []

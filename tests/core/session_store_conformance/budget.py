"""Shared session-store budget contract cases."""

from __future__ import annotations

import asyncio
import contextvars
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest
from tests.core.session_store_conformance.support import (
    _close_store,
    _identity,
    _open_store,
    _reopen_store,
)

import cayu.runtime._model_step_executor as model_step_executor_module
from cayu.agents import AgentSpec
from cayu.applications import CayuApp
from cayu.budgets.aggregates import estimate_usage_rollup_cost
from cayu.budgets.base import (
    BudgetLimit,
    BudgetPolicy,
    BudgetReservation,
    BudgetReservationIdentityConflict,
)
from cayu.budgets.billing import BillingIdentity
from cayu.budgets.pricing import ModelPrice, PriceBook
from cayu.budgets.usage import UsageMetrics
from cayu.events import (
    Event,
    EventType,
)
from cayu.messages import (
    Message,
)
from cayu.providers.base import ModelProvider, ModelStreamEvent, UsageDialect
from cayu.providers.bedrock import bedrock_billing_identity, completed_bedrock_billing_identity
from cayu.runtime.authority import SessionRunFenced
from cayu.sessions.records import Session, SessionStatus
from cayu.sessions.requests import (
    RunRequest,
)
from cayu.sessions.store import SessionStore
from cayu.sessions.usage import UsageRollupQuery

pytestmark = pytest.mark.postgres


def test_session_store_conformance_rejects_store_wide_budget_reservation_reuse(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            first = await store.create(
                RunRequest(
                    session_id=f"reservation-identity-first-{session_store_case[0]}",
                    agent_name="assistant",
                    messages=[Message.text("user", "first")],
                ),
                identity=_identity(),
            )
            second = await store.create(
                RunRequest(
                    session_id=f"reservation-identity-second-{session_store_case[0]}",
                    agent_name="assistant",
                    messages=[Message.text("user", "second")],
                ),
                identity=_identity(),
            )
            reservation_id = "bres_shared_conformance_identity"
            await store.append_event(
                first.id,
                Event(
                    type=EventType.BUDGET_RESERVED,
                    session_id=first.id,
                    payload={"reservation_id": reservation_id},
                ),
            )

            with pytest.raises(
                BudgetReservationIdentityConflict,
                match="Budget ledger reused a reservation identity",
            ):
                await store.append_event(
                    second.id,
                    Event(
                        type=EventType.BUDGET_RESERVED,
                        session_id=second.id,
                        payload={"reservation_id": reservation_id},
                    ),
                )

            assert [event.type for event in await store.load_events(first.id)] == [
                EventType.BUDGET_RESERVED
            ]
            assert await store.load_events(second.id) == []

            concurrent_sessions = [
                await store.create(
                    RunRequest(
                        session_id=(f"reservation-identity-race-{index}-{session_store_case[0]}"),
                        agent_name="assistant",
                        messages=[Message.text("user", f"race {index}")],
                    ),
                    identity=_identity(),
                )
                for index in range(2)
            ]
            concurrent_reservation_id = "bres_shared_conformance_race_identity"
            results = await asyncio.gather(
                *(
                    store.append_event(
                        session.id,
                        Event(
                            type=EventType.BUDGET_RESERVED,
                            session_id=session.id,
                            payload={"reservation_id": concurrent_reservation_id},
                        ),
                    )
                    for session in concurrent_sessions
                ),
                return_exceptions=True,
            )

            assert sum(result is None for result in results) == 1
            conflicts = [
                result
                for result in results
                if isinstance(result, BudgetReservationIdentityConflict)
            ]
            assert len(conflicts) == 1
            concurrent_event_counts = [
                len(await store.load_events(session.id)) for session in concurrent_sessions
            ]
            assert sum(concurrent_event_counts) == 1
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_retains_reservation_identity_after_session_deletion(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            first = await store.create(
                RunRequest(
                    session_id=f"reservation-delete-first-{session_store_case[0]}",
                    agent_name="assistant",
                    messages=[Message.text("user", "first")],
                ),
                identity=_identity(),
            )
            reservation_id = "bres_shared_conformance_deleted_session"
            reserved = Event(
                type=EventType.BUDGET_RESERVED,
                session_id=first.id,
                payload={"reservation_id": reservation_id},
            )
            await store.append_event(
                first.id,
                reserved,
            )
            reserved_claim = await store.claim_persisted_event_side_effect(
                session_id=first.id,
                event_id=reserved.id,
            )
            assert reserved_claim is not None
            await store.mark_persisted_event_side_effect_delivered(reserved_claim)
            released = Event(
                type=EventType.BUDGET_RESERVATION_RELEASED,
                session_id=first.id,
                payload={"reservation_id": reservation_id},
            )
            await store.append_event(first.id, released)
            released_claim = await store.claim_persisted_event_side_effect(
                session_id=first.id,
                event_id=released.id,
            )
            assert released_claim is not None
            await store.mark_persisted_event_side_effect_delivered(released_claim)
            await store.delete_session(first.id)
            store = await _reopen_store(session_store_case, store)

            second = await store.create(
                RunRequest(
                    session_id=f"reservation-delete-second-{session_store_case[0]}",
                    agent_name="assistant",
                    messages=[Message.text("user", "second")],
                ),
                identity=_identity(),
            )
            with pytest.raises(
                BudgetReservationIdentityConflict,
                match="Budget ledger reused a reservation identity",
            ):
                await store.append_event(
                    second.id,
                    Event(
                        type=EventType.BUDGET_RESERVED,
                        session_id=second.id,
                        payload={"reservation_id": reservation_id},
                    ),
                )
            assert await store.load_events(second.id) == []
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_claims_reservation_publication_idempotently(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            session = await store.create(
                RunRequest(
                    session_id=f"reservation-claim-{session_store_case[0]}",
                    agent_name="assistant",
                    messages=[Message.text("user", "claim")],
                ),
                identity=_identity(),
            )
            event = Event(
                type=EventType.BUDGET_RESERVED,
                session_id=session.id,
                payload={"reservation_id": "bres_shared_conformance_claim"},
            )
            await store.claim_budget_reservation_identity(
                reservation_id="bres_shared_conformance_claim",
                publication_session_id=session.id,
                publication_id=event.id,
            )
            await store.claim_budget_reservation_identity(
                reservation_id="bres_shared_conformance_claim",
                publication_session_id=session.id,
                publication_id=event.id,
            )
            with pytest.raises(
                BudgetReservationIdentityConflict,
                match="Budget ledger reused a reservation identity",
            ):
                await store.claim_budget_reservation_identity(
                    reservation_id="bres_shared_conformance_claim",
                    publication_session_id=session.id,
                    publication_id="evt_conflicting_reservation_publication",
                )
            conflicting_session = await store.create(
                RunRequest(
                    session_id=f"reservation-claim-conflict-{session_store_case[0]}",
                    agent_name="assistant",
                    messages=[Message.text("user", "conflict")],
                ),
                identity=_identity(),
            )
            with pytest.raises(
                BudgetReservationIdentityConflict,
                match="Budget ledger reused a reservation identity",
            ):
                await store.claim_budget_reservation_identity(
                    reservation_id="bres_shared_conformance_claim",
                    publication_session_id=conflicting_session.id,
                    publication_id=event.id,
                )
            with pytest.raises(KeyError, match="Session not found"):
                await store.claim_budget_reservation_identity(
                    reservation_id="bres_shared_conformance_missing_session",
                    publication_session_id="sess_missing_reservation_publication",
                    publication_id=event.id,
                )

            await store.append_event(session.id, event)
            assert [stored.id for stored in await store.load_events(session.id)] == [event.id]

            concurrent_results = await asyncio.gather(
                *(
                    store.claim_budget_reservation_identity(
                        reservation_id="bres_shared_conformance_concurrent_claim",
                        publication_session_id=session.id,
                        publication_id=publication_id,
                    )
                    for publication_id in (
                        "evt_concurrent_claim_first",
                        "evt_concurrent_claim_second",
                    )
                ),
                return_exceptions=True,
            )
            assert sum(result is None for result in concurrent_results) == 1
            assert (
                sum(
                    isinstance(result, BudgetReservationIdentityConflict)
                    for result in concurrent_results
                )
                == 1
            )
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_classifies_exact_budget_event_replay_as_duplicate(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            session = await store.create(
                RunRequest(
                    session_id=f"reservation-event-replay-{session_store_case[0]}",
                    agent_name="assistant",
                    messages=[Message.text("user", "publish")],
                ),
                identity=_identity(),
            )
            event = Event(
                id="evt_shared_conformance_reservation_replay",
                type=EventType.BUDGET_RESERVED,
                session_id=session.id,
                payload={"reservation_id": "bres_shared_conformance_reservation_replay"},
            )

            await store.append_event(session.id, event)

            with pytest.raises(ValueError, match="Event already exists for session"):
                await store.append_event(session.id, event)

            assert [stored.id for stored in await store.load_events(session.id)] == [event.id]
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_enforces_reservation_claims_in_atomic_publication(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            session = await store.create(
                RunRequest(
                    session_id=f"reservation-atomic-publication-{session_store_case[0]}",
                    agent_name="assistant",
                    messages=[Message.text("user", "publish")],
                ),
                identity=_identity(),
            )
            await store.checkpoint(session.id, {"state": "before"})

            reservation_id = "bres_shared_conformance_atomic_publication"
            rightful_event = Event(
                type=EventType.BUDGET_RESERVED,
                session_id=session.id,
                payload={"reservation_id": reservation_id},
            )
            conflicting_event = Event(
                type=EventType.BUDGET_RESERVED,
                session_id=session.id,
                payload={"reservation_id": reservation_id},
            )
            await store.claim_budget_reservation_identity(
                reservation_id=reservation_id,
                publication_session_id=session.id,
                publication_id=rightful_event.id,
            )

            with pytest.raises(
                BudgetReservationIdentityConflict,
                match="Budget ledger reused a reservation identity",
            ):
                await store.publish_checkpoint_and_events(
                    session.id,
                    checkpoint_transform=lambda _session, _checkpoint: {"state": "conflicting"},
                    events=[conflicting_event],
                )

            assert await store.load_checkpoint(session.id) == {"state": "before"}
            assert await store.load_events(session.id) == []

            await store.publish_checkpoint_and_events(
                session.id,
                checkpoint_transform=lambda _session, _checkpoint: {"state": "published"},
                events=[rightful_event],
            )

            assert await store.load_checkpoint(session.id) == {"state": "published"}
            assert [event.id for event in await store.load_events(session.id)] == [
                rightful_event.id
            ]
            with pytest.raises(ValueError, match="Event already exists for session"):
                await store.publish_checkpoint_and_events(
                    session.id,
                    checkpoint_transform=lambda _session, _checkpoint: {"state": "duplicate"},
                    events=[rightful_event],
                )

            assert await store.load_checkpoint(session.id) == {"state": "published"}
            assert [event.id for event in await store.load_events(session.id)] == [
                rightful_event.id
            ]
            await store.claim_budget_reservation_identity(
                reservation_id=reservation_id,
                publication_session_id=session.id,
                publication_id=rightful_event.id,
            )
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_fences_stale_reservation_claims(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        replacement_ready = asyncio.Event()
        allow_replacement_claim = asyncio.Event()
        try:
            created = await store.create(
                RunRequest(
                    session_id=f"reservation-claim-fence-{session_store_case[0]}",
                    agent_name="assistant",
                    messages=[Message.text("user", "claim")],
                ),
                identity=_identity(),
            )
            owned = await store.transition_status(
                created.id,
                from_statuses={SessionStatus.PENDING},
                to_status=SessionStatus.RUNNING,
            )

            async def replace_owner_and_claim() -> Session:
                replacement = await store.fence_stalled_run(
                    created.id,
                    statuses={SessionStatus.RUNNING},
                    inactive_for_seconds=0,
                )
                assert replacement is not None
                assert replacement.run_epoch == owned.run_epoch + 1
                replacement_ready.set()
                await allow_replacement_claim.wait()
                await store.claim_budget_reservation_identity(
                    reservation_id="bres_shared_conformance_fenced_claim",
                    publication_session_id=created.id,
                    publication_id="evt_replacement_reservation_publication",
                )
                await store.release_run_fence(created.id)
                return replacement

            replacement_task = asyncio.create_task(
                replace_owner_and_claim(),
                context=contextvars.Context(),
            )
            await asyncio.wait_for(replacement_ready.wait(), timeout=10)
            try:
                with pytest.raises(
                    SessionRunFenced,
                    match="Session run epoch no longer owns",
                ):
                    await store.claim_budget_reservation_identity(
                        reservation_id="bres_shared_conformance_fenced_claim",
                        publication_session_id=created.id,
                        publication_id="evt_stale_reservation_publication",
                    )
            finally:
                allow_replacement_claim.set()
            await replacement_task
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_inspection_uses_tolerant_usage_aggregates(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        session_id = f"inspection-usage-{session_store_case[0]}"
        timestamp = datetime(2026, 7, 20, 12, 0, tzinfo=UTC)
        try:
            await store.create(
                RunRequest(
                    session_id=session_id,
                    agent_name="assistant",
                    messages=[Message.text("user", "inspect")],
                ),
                identity=_identity(),
            )
            await store.append_event(
                session_id,
                Event(
                    type=EventType.MODEL_COMPLETED,
                    session_id=session_id,
                    timestamp=timestamp,
                    payload={
                        "usage_metrics": {
                            "provider_name": " fake ",
                            "model": "valid-model",
                            "input_tokens": 7,
                            "output_tokens": 3,
                            "total_tokens": 10,
                            "reasoning_output_tokens": "not-an-integer",
                            "cache": {
                                "read_tokens": 4,
                                "write_tokens": -1,
                            },
                        }
                    },
                ),
            )
            await store.append_events(
                session_id,
                [
                    Event(
                        type=EventType.MODEL_HOSTED_TOOL_CALL,
                        session_id=session_id,
                        timestamp=timestamp + timedelta(milliseconds=50),
                        payload={
                            "tool_type": "web_search",
                            "status": "failed",
                            "provider_name": "openai",
                            "model": "gpt-5.6",
                        },
                    ),
                    Event(
                        type=EventType.MODEL_HOSTED_TOOL_CALL,
                        session_id=session_id,
                        timestamp=timestamp + timedelta(milliseconds=100),
                        payload={
                            "tool_type": "web_search",
                            "status": "outcome_unknown",
                            "provider_name": "openai",
                            "model": "gpt-5.6",
                        },
                    ),
                ],
            )

            inspection = await store.inspect_summary(session_id)
            native = await store.aggregate_usage(
                UsageRollupQuery(
                    start_at=timestamp - timedelta(seconds=1),
                    end_at=timestamp + timedelta(seconds=1),
                )
            )

            assert inspection.model_calls == native.totals.model_steps == 1
            assert inspection.model_calls_with_usage == 1
            assert inspection.model_calls_with_usage == native.totals.model_steps_with_usage
            assert inspection.usage.usage == native.totals.usage
            assert inspection.usage.provider_names == ["openai"]
            assert inspection.usage.models == ["valid-model", "gpt-5.6"]
            assert inspection.usage.usage.input_tokens == 7
            assert inspection.usage.usage.output_tokens == 3
            assert inspection.usage.usage.reasoning_output_tokens == 0
            assert inspection.usage.usage.cache.read_tokens == 4
            assert inspection.usage.usage.cache.write_tokens == 0
            assert inspection.usage.usage.hosted_tools.web_search_calls == 1
            assert inspection.usage.usage.hosted_tools.web_search_outcome_unknown == 1
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_usage_normalization_failure_is_authoritative(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        session_id = f"normalization-failed-usage-{session_store_case[0]}"
        timestamp = datetime(2026, 7, 20, 13, 0, tzinfo=UTC)
        try:
            await store.create(
                RunRequest(
                    session_id=session_id,
                    agent_name="assistant",
                    messages=[Message.text("user", "inspect")],
                ),
                identity=_identity(),
            )
            await store.append_event(
                session_id,
                Event(
                    type=EventType.MODEL_COMPLETED,
                    session_id=session_id,
                    timestamp=timestamp,
                    payload={
                        "provider_name": "raw-provider",
                        "model": "raw-model",
                        "usage_normalization_failed": True,
                        "usage": {
                            "input_tokens": 70,
                            "output_tokens": 30,
                            "total_tokens": 100,
                        },
                        "usage_metrics": {
                            "provider_name": "normalized-provider",
                            "model": "normalized-model",
                            "input_tokens": 7,
                            "output_tokens": 3,
                            "total_tokens": 10,
                        },
                    },
                ),
            )

            inspection = await store.inspect_summary(session_id)
            native = await store.aggregate_usage(
                UsageRollupQuery(
                    start_at=timestamp - timedelta(seconds=1),
                    end_at=timestamp + timedelta(seconds=1),
                    include_pricing_inputs=True,
                )
            )
            cost = estimate_usage_rollup_cost(
                native,
                PriceBook(
                    prices=(
                        ModelPrice.fixed(
                            provider_name="normalized-provider",
                            model="normalized-model",
                            input_per_million=Decimal("1"),
                            output_per_million=Decimal("1"),
                        ),
                    )
                ),
            )

            assert inspection.model_calls == native.totals.model_steps == 1
            assert inspection.model_calls_with_usage == 0
            assert native.totals.model_steps_with_usage == 0
            assert inspection.usage.provider_names == []
            assert inspection.usage.models == []
            assert inspection.usage.usage.total_tokens == 0
            assert native.totals.usage.total_tokens == 0
            assert native.pricing_input_group_count == 1
            assert native.pricing_inputs[0].metrics is None
            assert native.pricing_inputs[0].occurrences == 1
            assert cost.priced_model_steps == 0
            assert cost.unpriced_model_steps == 1
            assert cost.unpriced_reasons[0].reason == (
                "model.completed event has no valid normalized usage metrics"
            )
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_inspection_fails_closed_for_malformed_budget_evidence(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        store_kind = session_store_case[0]
        try:
            for index, case in enumerate(
                (
                    "invalid-attempt",
                    "whitespace-reservation",
                    "invalid-amount",
                    "outstanding",
                    "invalid-pricing",
                ),
                start=1,
            ):
                session_id = f"inspection-budget-{case}-{store_kind}"
                budget_limit_id = f"blim_{index:064x}"
                identity = {
                    "model_step_id": f"mstep_{index:032x}",
                    "model_attempt_id": f"matt_{index:032x}",
                }
                reservation_id = (
                    " reservation-with-whitespace "
                    if case == "whitespace-reservation"
                    else f"reservation-{case}"
                )
                reservation_identity = (
                    {**identity, "model_attempt_id": "bad"}
                    if case == "invalid-attempt"
                    else identity
                )

                await store.create(
                    RunRequest(
                        session_id=session_id,
                        agent_name="assistant",
                        messages=[Message.text("user", "inspect")],
                    ),
                    identity=_identity(),
                )
                await store.append_event(
                    session_id,
                    Event(
                        type=EventType.BUDGET_RESERVED,
                        session_id=session_id,
                        payload={
                            "reservation_id": reservation_id,
                            "budget_limit_id": budget_limit_id,
                            **reservation_identity,
                            "scope": "session",
                            "key": session_id,
                            "window": "all_time",
                            "currency": "USD",
                            "maximum": "1",
                            "action": "interrupt",
                        },
                    ),
                )
                if case not in {"invalid-attempt", "outstanding"}:
                    pricing: object = (
                        {"provider_name": 42}
                        if case == "invalid-pricing"
                        else {
                            "provider_name": "fake",
                            "model": "model",
                            "match": "exact",
                            "provenance": {
                                "source": "application",
                                "url": "application://test-price-book",
                                "as_of": "2026-07-25",
                            },
                            "effective_from": None,
                            "effective_through": None,
                            "tier_max_input_tokens": None,
                        }
                    )
                    await store.append_event(
                        session_id,
                        Event(
                            type=EventType.BUDGET_RECONCILED,
                            session_id=session_id,
                            payload={
                                "reservation_id": reservation_id,
                                "budget_limit_id": budget_limit_id,
                                **identity,
                                "actual_amount": (
                                    "not-a-number" if case == "invalid-amount" else "0.25"
                                ),
                                "pricing": pricing,
                            },
                        ),
                    )

                inspection = await store.inspect_summary(session_id)

                assert inspection.budget.cost_state == "partial"
                assert inspection.budget.amount is None
                assert inspection.budget.currency is None
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_inspection_preserves_budget_event_order(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        store_kind = session_store_case[0]
        pricing = {
            "provider_name": "fake",
            "model": "model",
            "match": "exact",
            "provenance": {
                "source": "application",
                "url": "application://test-price-book",
                "as_of": "2026-08-10",
            },
            "effective_from": None,
            "effective_through": None,
            "tier_max_input_tokens": None,
        }
        try:
            case_index = 0
            for terminal_kind in ("reconciled", "released"):
                for reordered in (False, True):
                    case_index += 1
                    order_name = "reordered" if reordered else "ordered"
                    session_id = f"inspection-budget-{terminal_kind}-{order_name}-{store_kind}"
                    budget_limit_id = f"blim_{case_index:064x}"
                    reservation_id = f"reservation-{terminal_kind}-{case_index}"
                    identity = {
                        "model_step_id": f"mstep_{case_index:032x}",
                        "model_attempt_id": f"matt_{case_index:032x}",
                    }
                    await store.create(
                        RunRequest(
                            session_id=session_id,
                            agent_name="assistant",
                            messages=[Message.text("user", "inspect")],
                        ),
                        identity=_identity(),
                    )
                    reservation = Event(
                        type=EventType.BUDGET_RESERVED,
                        session_id=session_id,
                        payload={
                            "reservation_id": reservation_id,
                            "budget_limit_id": budget_limit_id,
                            **identity,
                            "scope": "session",
                            "key": None,
                            "window": "all_time",
                            "currency": "USD",
                            "maximum": "1",
                            "action": "interrupt",
                        },
                    )
                    if terminal_kind == "reconciled":
                        terminal = Event(
                            type=EventType.BUDGET_RECONCILED,
                            session_id=session_id,
                            payload={
                                "reservation_id": reservation_id,
                                "settlement_kind": "completed",
                                "budget_limit_id": budget_limit_id,
                                **identity,
                                "actual_amount": "0.25",
                                "pricing": pricing,
                            },
                        )
                        model_terminal = Event(
                            type=EventType.MODEL_COMPLETED,
                            session_id=session_id,
                            payload=identity,
                        )
                        expected_amount = "0.25"
                    else:
                        terminal = Event(
                            type=EventType.BUDGET_RESERVATION_RELEASED,
                            session_id=session_id,
                            payload={
                                "reservation_id": reservation_id,
                                "settlement_kind": "released",
                                "budget_limit_id": budget_limit_id,
                                **identity,
                            },
                        )
                        model_terminal = None
                        expected_amount = "0"
                    events = [terminal, reservation] if reordered else [reservation, terminal]
                    if model_terminal is not None:
                        events.append(model_terminal)
                    await store.append_events(session_id, events)

                    inspection = await store.inspect_summary(session_id)

                    if reordered:
                        assert inspection.budget.cost_state == "partial"
                        assert inspection.budget.amount is None
                        assert inspection.budget.currency is None
                    else:
                        assert inspection.budget.cost_state == "priced"
                        assert inspection.budget.amount == expected_amount
                        assert inspection.budget.currency == "USD"
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_inspection_requires_exact_model_budget_join(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        store_kind = session_store_case[0]
        pricing = {
            "provider_name": "fake",
            "model": "model",
            "match": "exact",
            "provenance": {
                "source": "application",
                "url": "application://test-price-book",
                "as_of": "2026-07-27",
            },
            "effective_from": None,
            "effective_through": None,
            "tier_max_input_tokens": None,
        }
        try:
            for index, case in enumerate(("matching", "conflicting", "missing"), start=1):
                session_id = f"inspection-model-budget-{case}-{store_kind}"
                model_step_id = f"mstep_{index:032x}"
                model_attempt_id = f"matt_{index:032x}"
                budget_step_id = (
                    f"mstep_{index + 10:032x}" if case == "conflicting" else model_step_id
                )
                await store.create(
                    RunRequest(
                        session_id=session_id,
                        agent_name="assistant",
                        messages=[Message.text("user", "inspect")],
                    ),
                    identity=_identity(),
                )
                events = []
                if case != "missing":
                    events.append(
                        Event(
                            type=EventType.MODEL_COMPLETED,
                            session_id=session_id,
                            payload={
                                "model_step_id": model_step_id,
                                "model_attempt_id": model_attempt_id,
                                "usage_metrics": {
                                    "input_tokens": 7,
                                    "output_tokens": 3,
                                    "total_tokens": 10,
                                },
                            },
                        )
                    )
                events.extend(
                    [
                        Event(
                            type=EventType.BUDGET_RESERVED,
                            session_id=session_id,
                            payload={
                                "reservation_id": f"reservation-{case}",
                                "budget_limit_id": f"blim_{index:064x}",
                                "model_step_id": budget_step_id,
                                "model_attempt_id": model_attempt_id,
                                "scope": "session",
                                "key": None,
                                "window": "all_time",
                                "currency": "USD",
                                "maximum": "1",
                                "action": "interrupt",
                            },
                        ),
                        Event(
                            type=EventType.BUDGET_RECONCILED,
                            session_id=session_id,
                            payload={
                                "reservation_id": f"reservation-{case}",
                                "settlement_kind": "completed",
                                "budget_limit_id": f"blim_{index:064x}",
                                "model_step_id": budget_step_id,
                                "model_attempt_id": model_attempt_id,
                                "actual_amount": "0.25",
                                "pricing": pricing,
                            },
                        ),
                    ]
                )
                await store.append_events(session_id, events)

                inspection = await store.inspect_summary(session_id)

                if case == "matching":
                    assert inspection.budget.cost_state == "priced"
                    assert inspection.budget.amount == "0.25"
                    assert inspection.budget.currency == "USD"
                else:
                    assert inspection.budget.cost_state == "partial"
                    assert inspection.budget.amount is None
                    assert inspection.budget.currency is None
        finally:
            await _close_store(store)

    asyncio.run(run())


@pytest.mark.parametrize("invalid_text", ["100\x00", "\ud800"], ids=["nul", "surrogate"])
@pytest.mark.parametrize(
    "invalid_primary_counter",
    [True, False],
    ids=["invalid-primary", "invalid-extra-field"],
)
@pytest.mark.parametrize("with_reservation", [False, True], ids=["strict", "reservation"])
def test_session_store_conformance_preserves_undurable_completion_spend(
    session_store_case,
    invalid_text: str,
    invalid_primary_counter: bool,
    with_reservation: bool,
) -> None:
    class MalformedUsageProvider(ModelProvider):
        name = "renamed-openai"
        usage_dialect = UsageDialect.OPENAI

        def __init__(self) -> None:
            self.calls = 0

        async def stream(self, request):
            del request
            self.calls += 1
            yield ModelStreamEvent.text_delta("done")
            yield ModelStreamEvent.completed(
                {
                    "provider_name": "provider-controlled-spoof",
                    "model": "gpt-test",
                    "usage": (
                        {
                            "input_tokens": invalid_text,
                            "output_tokens": 1,
                        }
                        if invalid_primary_counter
                        else {
                            "input_tokens": 10,
                            "output_tokens": 1,
                            "provider_note": invalid_text,
                        }
                    ),
                }
            )

    async def run() -> None:
        store = await _open_store(session_store_case)
        provider = MalformedUsageProvider()
        pricing = PriceBook(
            prices=(
                ModelPrice.fixed(
                    provider_name=provider.name,
                    model="gpt-test",
                    input_per_million=Decimal("10"),
                    output_per_million=Decimal("10"),
                ),
            )
        )
        reservation = (
            BudgetReservation(
                max_input_tokens=1_000_000,
                max_output_tokens=0,
            )
            if with_reservation
            else None
        )
        maximum = Decimal("10") if with_reservation else Decimal("100")
        policy = BudgetPolicy(
            limits=(
                BudgetLimit(
                    scope="app",
                    max_estimated_cost=maximum,
                    pricing=pricing,
                    reservation=reservation,
                ),
            )
        )

        def build_app(current_store: SessionStore) -> CayuApp:
            app = CayuApp(
                session_store=current_store,
                budget_policy=policy,
                enable_logging=False,
            )
            app.register_provider(provider, default=True)
            app.register_agent(AgentSpec(name="assistant", model="gpt-test"))
            return app

        store_kind = session_store_case[0]
        case_suffix = f"{store_kind}-{with_reservation}-{invalid_primary_counter}"
        first_session_id = f"undurable-completion-{case_suffix}-first"
        second_session_id = f"undurable-completion-{case_suffix}-second"
        try:
            app = build_app(store)
            first_events = [
                event
                async for event in app.run(
                    RunRequest(
                        session_id=first_session_id,
                        agent_name="assistant",
                        messages=[Message.text("user", "first")],
                    )
                )
            ]
            completed = next(
                event for event in first_events if event.type == EventType.MODEL_COMPLETED
            )
            assert "usage" not in completed.payload
            assert "usage_metrics" not in completed.payload
            assert completed.payload["usage_normalization_failed"] is True
            assert completed.payload["usage_unavailable_reason"] == (
                "invalid model completion usage telemetry"
            )
            assert completed.payload["provider_name"] == provider.name

            cost = await app.get_session_cost(first_session_id, pricing)
            assert cost.model_steps == 1
            assert cost.priced_model_steps == 0
            assert cost.unpriced_model_steps == 1
            assert cost.line_items[0].provider_name == provider.name
            if with_reservation:
                reconciliation = next(
                    event for event in first_events if event.type == EventType.BUDGET_RECONCILED
                )
                assert reconciliation.payload["actual_amount"] == "10"
                assert reconciliation.payload["reason"] == (
                    "model completed without priced usage; charged reserved amount"
                )

            store = await _reopen_store(session_store_case, store)
            restarted_app = build_app(store)
            second_events = [
                event
                async for event in restarted_app.run(
                    RunRequest(
                        session_id=second_session_id,
                        agent_name="assistant",
                        messages=[Message.text("user", "second")],
                    )
                )
            ]
            assert provider.calls == 1
            assert EventType.MODEL_STARTED not in {event.type for event in second_events}
            assert EventType.BUDGET_LIMIT_REACHED in {event.type for event in second_events}
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_preserves_usage_for_invalid_provider_state(
    session_store_case,
) -> None:
    class InvalidProviderStateProvider(ModelProvider):
        name = "fake"

        def __init__(self) -> None:
            self.calls = 0

        async def stream(self, request):
            del request
            self.calls += 1
            yield ModelStreamEvent.completed(
                {
                    "model": "fake-model",
                    "usage": {
                        "input_tokens": 7,
                        "output_tokens": 3,
                        "total_tokens": 10,
                    },
                    "provider_state": {},
                }
            )

    async def assert_durable_result(
        store: SessionStore,
        session_id: str,
    ) -> None:
        events = await store.load_events(session_id)
        completed = next(event for event in events if event.type == EventType.MODEL_COMPLETED)
        assert completed.payload["usage_metrics"]["input_tokens"] == 7
        assert completed.payload["usage_metrics"]["output_tokens"] == 3
        assert completed.payload["usage_metrics"]["total_tokens"] == 10
        assert completed.payload["completion_outcome"] == "invalid_transcript_state"
        assert completed.payload["completion_error"]["provider_error_code"] == (
            "invalid_model_completion_transcript"
        )
        assert completed.payload["transcript_cursor"] == 1
        assert "provider_state" not in completed.payload

        app = CayuApp(session_store=store, enable_logging=False)
        usage = await app.get_session_usage(session_id)
        assert usage.model_steps == 1
        assert usage.usage.input_tokens == 7
        assert usage.usage.output_tokens == 3
        assert usage.usage.total_tokens == 10
        assert await store.load_transcript(session_id) == [Message.text("user", "hello")]
        session = await store.load(session_id)
        assert session is not None
        assert session.status == SessionStatus.FAILED

    async def run() -> None:
        store = await _open_store(session_store_case)
        provider = InvalidProviderStateProvider()
        session_id = f"invalid-provider-state-{session_store_case[0]}"
        try:
            app = CayuApp(session_store=store, enable_logging=False)
            app.register_provider(provider, default=True)
            app.register_agent(AgentSpec(name="assistant", model="fake-model"))
            events = [
                event
                async for event in app.run(
                    RunRequest(
                        session_id=session_id,
                        agent_name="assistant",
                        messages=[Message.text("user", "hello")],
                    )
                )
            ]

            assert provider.calls == 1
            assert EventType.MODEL_RETRY not in {event.type for event in events}
            assert EventType.MODEL_ERROR not in {event.type for event in events}
            assert events[-1].type == EventType.SESSION_FAILED
            await assert_durable_result(store, session_id)

            store = await _reopen_store(session_store_case, store)
            await assert_durable_result(store, session_id)
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_preserves_only_safe_bedrock_aggregate_evidence(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            session_id = "sess_bedrock_aggregate_evidence"
            start = datetime(2026, 7, 1, tzinfo=UTC)
            invoked_model = "global.anthropic.claude-sonnet-4-6"
            await store.create(
                RunRequest(
                    session_id=session_id,
                    agent_name="assistant",
                    messages=[Message.text("user", "price this run")],
                ),
                identity=_identity(),
            )

            def identity_for_region(region: str) -> BillingIdentity:
                completed = completed_bedrock_billing_identity(
                    bedrock_billing_identity(
                        invoked_model=invoked_model,
                        source_region=region,
                        resource_type="inference_profile",
                        profile_scope="global",
                        requested_service_tier="default",
                    ),
                    effective_service_tier="default",
                )
                return BillingIdentity(
                    provider_name=completed.provider_name,
                    resource_id=completed.resource_id,
                    request_evidence={
                        **completed.request_evidence,
                        "customer_secret": "must-not-cross-the-aggregate-boundary",
                    },
                    completion_evidence={
                        **completed.completion_evidence,
                        "provider_trace": "must-also-remain-redacted",
                    },
                    pricing_contexts=completed.pricing_contexts,
                )

            nested_identity = identity_for_region("us-east-1")
            root_identity = identity_for_region("us-west-2")
            nested_metrics = UsageMetrics(
                provider_name="bedrock",
                model=invoked_model,
                billing_identity=nested_identity,
                input_tokens=1,
                total_tokens=1,
            )
            root_metrics = UsageMetrics(
                provider_name="bedrock",
                model=invoked_model,
                input_tokens=1,
                total_tokens=1,
            )
            await store.append_events(
                session_id,
                [
                    Event(
                        id="bedrock-nested-aggregate-evidence",
                        type=EventType.MODEL_COMPLETED,
                        session_id=session_id,
                        timestamp=start,
                        payload={"usage_metrics": nested_metrics.model_dump(mode="json")},
                    ),
                    Event(
                        id="bedrock-root-aggregate-evidence",
                        type=EventType.MODEL_COMPLETED,
                        session_id=session_id,
                        timestamp=start + timedelta(minutes=1),
                        payload={
                            "usage_metrics": root_metrics.model_dump(mode="json"),
                            "billing_identity": root_identity.model_dump(mode="json"),
                        },
                    ),
                ],
            )

            result = await store.aggregate_usage(
                UsageRollupQuery(
                    start_at=start,
                    end_at=start + timedelta(days=1),
                    include_pricing_inputs=True,
                )
            )

            assert result.pricing_inputs_accuracy.kind == "exact"
            assert len(result.pricing_inputs) == 2
            projected_by_region: dict[str, BillingIdentity] = {}
            for item in result.pricing_inputs:
                assert item.metrics is not None
                projected = item.metrics.billing_identity
                assert projected is not None
                region = projected.request_evidence.get("source_region")
                assert region is not None
                projected_by_region[region] = projected
            assert set(projected_by_region) == {"us-east-1", "us-west-2"}
            for region, projected in projected_by_region.items():
                assert projected.request_evidence == {
                    "source_region": region,
                    "resource_type": "inference_profile",
                    "profile_scope": "global",
                    "requested_service_tier": "default",
                }
                assert projected.completion_evidence == {"effective_service_tier": "default"}
                assert "customer_secret" not in projected.request_evidence
                assert "provider_trace" not in projected.completion_evidence
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_reconstructs_gemini_thinking_usage(
    session_store_case,
) -> None:
    class GeminiUsageProvider(ModelProvider):
        name = "gemini"
        usage_dialect = UsageDialect.GEMINI

        async def stream(self, request):
            del request
            yield ModelStreamEvent.text_delta("OK.")
            yield ModelStreamEvent.completed(
                {
                    "model": "gemini-3.5-flash",
                    "usage": {
                        "prompt_tokens": 5,
                        "completion_tokens": 2,
                        "total_tokens": 60,
                    },
                }
            )

    async def assert_usage(store: SessionStore, session_id: str) -> None:
        inspection = await store.inspect_summary(session_id)
        assert inspection.model_calls == 1
        assert inspection.model_calls_with_usage == 1
        assert inspection.usage.provider_names == ["gemini"]
        assert inspection.usage.models == ["gemini-3.5-flash"]
        assert inspection.usage.usage.input_tokens == 5
        assert inspection.usage.usage.output_tokens == 55
        assert inspection.usage.usage.reasoning_output_tokens == 53
        assert inspection.usage.usage.total_tokens == 60

    async def run() -> None:
        store = await _open_store(session_store_case)
        session_id = f"gemini-thinking-usage-{session_store_case[0]}"
        try:
            app = CayuApp(session_store=store, enable_logging=False)
            app.register_provider(GeminiUsageProvider(), default=True)
            app.register_agent(AgentSpec(name="assistant", model="gemini-3.5-flash"))
            events = [
                event
                async for event in app.run(
                    RunRequest(
                        session_id=session_id,
                        agent_name="assistant",
                        messages=[Message.text("user", "Reply with OK.")],
                    )
                )
            ]
            completed = next(event for event in events if event.type == EventType.MODEL_COMPLETED)
            assert "usage_normalization_failed" not in completed.payload
            await assert_usage(store, session_id)

            store = await _reopen_store(session_store_case, store)
            await assert_usage(store, session_id)
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_context_usage_pages_past_compaction(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        session_id = f"context-usage-pagination-{session_store_case[0]}"
        try:
            await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "hello")],
                ),
                identity=_identity(),
            )
            await store.append_events(
                session_id,
                [
                    Event(
                        type=EventType.MODEL_COMPLETED,
                        session_id=session_id,
                        payload={
                            "model": "fake-model",
                            "transcript_cursor": 2,
                            "usage": {
                                "input_tokens": 8,
                                "output_tokens": 2,
                                "total_tokens": 10,
                            },
                        },
                    ),
                    *[
                        Event(
                            type=EventType.MODEL_COMPLETED,
                            session_id=session_id,
                            payload={
                                **(
                                    {"purpose": "context_compaction"}
                                    if index % 3 == 0
                                    else (
                                        {"purpose": "future_auxiliary_call"}
                                        if index % 3 == 1
                                        else {}
                                    )
                                ),
                                "model": "summary-model",
                                "usage": {
                                    "input_tokens": index,
                                    "output_tokens": 1,
                                    "total_tokens": index + 1,
                                },
                            },
                        )
                        for index in range(1, 102)
                    ],
                ],
            )

            usage = await model_step_executor_module._context_usage_state_for_session(
                session_store=store,
                session_id=session_id,
            )

            assert usage.last_input_tokens == 8
            assert usage.last_output_tokens == 2
            assert usage.last_total_tokens == 10
            assert usage.last_transcript_cursor == 2
            assert usage.last_model == "fake-model"
        finally:
            await _close_store(store)

    asyncio.run(run())

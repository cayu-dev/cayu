from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from typing import Generic, TypeVar

from cayu._validation import require_clean_nonblank
from cayu.core.events import Event, EventType
from cayu.providers import ModelProviderError
from cayu.runtime._event_writer import RuntimeEventWriter
from cayu.runtime._session_queries import query_all_event_records
from cayu.runtime.budgets import (
    BudgetCheck,
    BudgetLedger,
    BudgetLimit,
    BudgetPolicy,
    BudgetReconciliation,
    BudgetReservationRecord,
    BudgetReservationResult,
    BudgetStore,
    budget_actual_cost_for_event,
    budget_check_from_events,
    budget_check_payload,
    budget_limits_for_session,
    budget_price,
    budget_reconciliation_payload,
    budget_reconciliation_with_pricing,
    budget_reservation_payload,
    events_for_budget_window,
    request_budget_limits_for_session,
)
from cayu.runtime.costs import SessionCostSummary, estimate_session_cost
from cayu.runtime.sessions import EventQuery, EventRecord, Session, SessionStore
from cayu.runtime.stop_policy import (
    RunLimits,
    StopDecision,
    StopLimit,
    first_reached_limit,
    has_run_limits,
)
from cayu.runtime.usage import (
    USAGE_BEARING_EVENT_TYPES,
    SessionUsageSummary,
    UsageMetrics,
    session_usage_summary,
)

UNKNOWN_POST_DISPATCH_BUDGET_REASON = (
    "provider usage unknown after dispatch; charged reserved amount"
)

_OperationResultT = TypeVar("_OperationResultT")
_StreamResultT = TypeVar("_StreamResultT")


class BudgetReservationLeaseLost(RuntimeError):
    """Raised when a live model step can no longer prove its budget reservation."""


class BudgetReservationLeaseLostBeforeModelDispatch(BudgetReservationLeaseLost):
    """Raised when lease loss is detected before any provider attempt starts."""


class BudgetDispatchReservationFailed(RuntimeError):
    def __init__(self, result: BudgetReservationResult) -> None:
        self.result = result
        super().__init__(result.message)


def add_budget_failure_note(
    authoritative_failure: BaseException,
    *,
    operation: str,
    accounting_failure: Exception,
) -> None:
    note = (
        f"Budget {operation} also failed: {type(accounting_failure).__name__}: {accounting_failure}"
    )
    if note not in getattr(authoritative_failure, "__notes__", ()):
        authoritative_failure.add_note(note)


def budget_heartbeat_task_failure(task: asyncio.Task[None]) -> BaseException:
    if task.cancelled():
        return BudgetReservationLeaseLost(
            "Budget reservation heartbeat was cancelled unexpectedly."
        )
    failure = task.exception()
    if failure is None:
        return BudgetReservationLeaseLost("Budget reservation heartbeat stopped unexpectedly.")
    return failure


async def _next_model_step_item(
    iterator: AsyncIterator[tuple[Event | None, _StreamResultT | None]],
) -> tuple[Event | None, _StreamResultT | None]:
    return await anext(iterator)


@dataclass(frozen=True)
class BudgetStepReservation:
    limit: BudgetLimit
    record: BudgetReservationRecord


@dataclass
class BudgetProviderDispatch:
    reservations: tuple[BudgetStepReservation, ...]
    completion: Event | None = None
    settled_reservation_ids: set[str] = field(default_factory=set)

    @property
    def settled(self) -> bool:
        return self.settled_reservation_ids == {
            reservation.record.reservation_id for reservation in self.reservations
        }


@dataclass
class BudgetModelStepLifecycle:
    dispatches: list[BudgetProviderDispatch] = field(default_factory=list)
    pending_reservations: tuple[BudgetStepReservation, ...] | None = None
    reservation_transition_lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    @property
    def provider_dispatch_may_have_occurred(self) -> bool:
        return bool(self.dispatches)

    def prepare_provider_dispatch(
        self,
        reservations: list[BudgetStepReservation],
    ) -> None:
        if self.pending_reservations is not None:
            raise RuntimeError("Provider dispatch already has prepared budget reservations.")
        self.pending_reservations = tuple(reservations)

    def mark_provider_dispatch(self) -> None:
        if self.pending_reservations is None:
            raise RuntimeError("Provider dispatch has no prepared budget reservations.")
        self.dispatches.append(BudgetProviderDispatch(self.pending_reservations))
        self.pending_reservations = None

    def record_model_completion(self, event: Event) -> None:
        if not self.dispatches:
            raise RuntimeError("Model completed before provider dispatch was recorded.")
        if self.dispatches[-1].completion is not None:
            raise RuntimeError("Provider dispatch produced more than one model completion.")
        self.dispatches[-1].completion = event


@dataclass(frozen=True)
class LimitEvaluation:
    decision: StopDecision | None
    usage_summary: SessionUsageSummary
    cost_summary: SessionCostSummary | None
    events: tuple[Event, ...] = ()


@dataclass(frozen=True)
class BudgetEvaluation:
    check: BudgetCheck | None
    events: tuple[Event, ...] = ()


@dataclass(frozen=True)
class BudgetReservationSetup:
    reservations: tuple[BudgetStepReservation, ...]
    failure: BudgetReservationResult | None
    events: tuple[Event, ...]
    error: Exception | None


@dataclass(frozen=True)
class OperationReservationSetup:
    reservations: tuple[BudgetStepReservation, ...]
    results: tuple[BudgetReservationResult, ...]
    releases: tuple[BudgetReconciliation, ...]
    failure: BudgetReservationResult | None
    error: BaseException | None


@dataclass(frozen=True)
class OperationBudgetCheck:
    limit: BudgetLimit
    check: BudgetCheck


@dataclass(frozen=True)
class BudgetedOperationSucceeded(Generic[_OperationResultT]):
    result: _OperationResultT
    events: tuple[Event, ...]


@dataclass(frozen=True)
class BudgetedOperationRejected:
    failure: BudgetReservationResult
    events: tuple[Event, ...]


@dataclass(frozen=True)
class BudgetedOperationFailed:
    error: BaseException
    cause: BaseException | None
    events: tuple[Event, ...]


@dataclass
class _BudgetedOperationLifecycle:
    reservations: list[BudgetStepReservation] = field(default_factory=list)
    events: list[Event] = field(default_factory=list)
    provider_dispatch_started: bool = False
    settled: bool = False
    predispatch_release_reason: str = "context compaction reservation lost before provider dispatch"


@dataclass(frozen=True)
class _BudgetLimitOutcome:
    decision: StopDecision
    check: BudgetCheck


class SessionUsageTracker:
    """Incrementally accumulate usage with one ordered tail query per refresh."""

    def __init__(self, session_store: SessionStore, *, session_id: str) -> None:
        self._session_store = session_store
        self._session_id = require_clean_nonblank(session_id, "session_id")
        self._after_sequence: int | None = None
        self._events: list[Event] = []

    async def _new_usage_records(self) -> list[EventRecord]:
        # One multi-type query and one shared watermark are essential. Separate
        # per-type reads can skip spend appended between queries.
        return await query_all_event_records(
            self._session_store,
            EventQuery(
                session_id=self._session_id,
                event_types=USAGE_BEARING_EVENT_TYPES,
                after_sequence=self._after_sequence,
            ),
        )

    async def mark_current_position(self) -> None:
        new_records = await self._new_usage_records()
        if new_records:
            self._after_sequence = new_records[-1].sequence

    async def usage_events(self) -> list[Event]:
        new_records = await self._new_usage_records()
        if new_records:
            self._events.extend(record.event for record in new_records)
            self._after_sequence = new_records[-1].sequence
        return self._events


class RunLimitController:
    """Evaluate run limits and own durable budget-accounting lifecycle."""

    def __init__(
        self,
        *,
        session_store: SessionStore,
        budget_store: BudgetStore,
        budget_ledger: BudgetLedger,
        event_writer: RuntimeEventWriter,
        clock: Callable[[], datetime],
    ) -> None:
        self._session_store = session_store
        self._budget_store = budget_store
        self._budget_ledger = budget_ledger
        self._event_writer = event_writer
        self._clock = clock

    def usage_tracker(self, session_id: str) -> SessionUsageTracker:
        return SessionUsageTracker(self._session_store, session_id=session_id)

    @property
    def reservation_ttl_seconds(self) -> int | None:
        return self._budget_ledger.reservation_ttl_seconds

    async def session_usage_events(self, session_id: str) -> list[Event]:
        records = await query_all_event_records(
            self._session_store,
            EventQuery(
                session_id=session_id,
                event_types=USAGE_BEARING_EVENT_TYPES,
            ),
        )
        return [record.event for record in records]

    async def evaluate_operation_run_limit(
        self,
        *,
        session: Session,
        limits: RunLimits,
        operation_events: list[Event],
        operation_started_at: float,
    ) -> StopDecision | None:
        """Evaluate limits for a bounded operation such as explicit compaction."""

        if not has_run_limits(limits):
            return None
        usage_events = list(operation_events)
        if limits.scope == "session":
            usage_events = [
                *await self.session_usage_events(session.id),
                *usage_events,
            ]
            created_at = session.created_at
            if created_at.tzinfo is None:
                created_at = created_at.replace(tzinfo=UTC)
            elapsed_seconds = max(
                0,
                int((self._clock() - created_at.astimezone(UTC)).total_seconds()),
            )
        else:
            elapsed_seconds = max(0, int(time.monotonic() - operation_started_at))
        return first_reached_limit(
            limits=limits,
            usage=session_usage_summary(session.id, usage_events),
            elapsed_seconds=elapsed_seconds,
        )

    async def evaluate_operation_budgets(
        self,
        *,
        session: Session,
        budget_limits: tuple[BudgetLimit, ...],
        operation_events: list[Event],
        provider_name: str | None,
        model: str | None,
    ) -> tuple[OperationBudgetCheck, ...]:
        """Evaluate scopes while including an operation's uncommitted events."""

        checks: list[OperationBudgetCheck] = []
        for limit in budget_limits:
            if limit.scope in {"app", "agent", "causal"}:
                existing_events = await self._budget_store.load_events_for_budget(
                    scope=limit.scope,
                    key=limit.key,
                    window=limit.window,
                )
            elif limit.scope == "session":
                existing_events = await self.session_usage_events(session.id)
            elif limit.scope == "run":
                existing_events = []
            else:
                raise ValueError(f"Unsupported request budget scope: {limit.scope}")
            events = events_for_budget_window(
                [*existing_events, *operation_events],
                limit.window,
                now=self._clock(),
            )
            event_provider_name, event_model = _latest_model_event_identity(operation_events)
            effective_provider_name = event_provider_name or provider_name
            effective_model = event_model or model
            if effective_provider_name is None or effective_model is None:
                summary = estimate_session_cost(
                    session_id=session.id,
                    events=events,
                    pricing=limit.pricing,
                    currency=limit.currency,
                )
                if (
                    summary.unpriced_model_steps == 0
                    and summary.total_cost < limit.max_estimated_cost
                ):
                    continue
            check = budget_check_from_events(
                limit=limit,
                events=events,
                provider_name=effective_provider_name,
                model=effective_model,
                effective_at=self._clock(),
            )
            checks.append(OperationBudgetCheck(limit=limit, check=check))
        return tuple(checks)

    async def evaluate_request_limits(
        self,
        *,
        session: Session,
        agent_name: str,
        environment_name: str | None,
        limits: RunLimits,
        budget_limits: tuple[BudgetLimit, ...],
        run_started_at: float,
        run_baseline: SessionUsageSummary | None = None,
        budget_baseline_events: list[Event] | None = None,
        pending_tool_calls: int = 0,
        budget_notify_events: list[Event] | None = None,
        usage_tracker: SessionUsageTracker | None = None,
    ) -> LimitEvaluation:
        budget_limits = request_budget_limits_for_session(
            limits=budget_limits,
            agent_name=agent_name,
            causal_budget_id=session.causal_budget_id,
        )
        if not has_run_limits(limits) and not budget_limits:
            return LimitEvaluation(
                decision=None,
                usage_summary=SessionUsageSummary(session_id=session.id),
                cost_summary=None,
            )
        events = (
            await usage_tracker.usage_events()
            if usage_tracker is not None
            else await self.session_usage_events(session.id)
        )
        usage_summary = session_usage_summary(session.id, events)
        usage_for_limits = usage_summary
        if limits.scope == "run" and run_baseline is not None:
            current, baseline = usage_summary.usage, run_baseline.usage
            usage_for_limits = SessionUsageSummary(
                session_id=session.id,
                tool_calls=max(0, usage_summary.tool_calls - run_baseline.tool_calls),
                usage=UsageMetrics(
                    input_tokens=max(0, current.input_tokens - baseline.input_tokens),
                    output_tokens=max(0, current.output_tokens - baseline.output_tokens),
                    total_tokens=max(0, current.total_tokens - baseline.total_tokens),
                ),
            )
        if limits.scope == "session":
            created_at = session.created_at
            if created_at.tzinfo is None:
                created_at = created_at.replace(tzinfo=UTC)
            elapsed_seconds = max(
                0,
                int((datetime.now(UTC) - created_at.astimezone(UTC)).total_seconds()),
            )
        else:
            elapsed_seconds = max(0, int(time.monotonic() - run_started_at))
        decision = first_reached_limit(
            limits=limits,
            usage=usage_for_limits,
            elapsed_seconds=elapsed_seconds,
            pending_tool_calls=pending_tool_calls,
        )
        if decision is not None:
            return LimitEvaluation(
                decision=decision,
                usage_summary=usage_summary,
                cost_summary=None,
            )

        cost_summary: SessionCostSummary | None = None
        emitted_events: list[Event] = []
        for budget_limit in budget_limits:
            budget_events = events
            budget_baseline: SessionCostSummary | None = None
            budget_window_now = self._clock()
            if budget_limit.scope in {"app", "agent", "causal"}:
                budget_events = await self._budget_store.load_events_for_budget(
                    scope=budget_limit.scope,
                    key=budget_limit.key,
                    window=budget_limit.window,
                )
            elif budget_limit.scope == "run":
                budget_events = events_for_budget_window(
                    events,
                    budget_limit.window,
                    now=budget_window_now,
                )
                budget_baseline = estimate_session_cost(
                    session_id=session.id,
                    events=events_for_budget_window(
                        budget_baseline_events or [],
                        budget_limit.window,
                        now=budget_window_now,
                    ),
                    pricing=budget_limit.pricing,
                    currency=budget_limit.currency,
                )
            elif budget_limit.scope == "session":
                budget_events = events_for_budget_window(
                    events,
                    budget_limit.window,
                    now=budget_window_now,
                )
            else:
                raise ValueError(f"Unsupported request budget scope: {budget_limit.scope}")

            cost_summary = estimate_session_cost(
                session_id=session.id,
                events=budget_events,
                pricing=budget_limit.pricing,
                currency=budget_limit.currency,
            )
            budget_outcome = _first_budget_limit_outcome(
                session=session,
                limit=budget_limit,
                cost_summary=cost_summary,
                cost_baseline=budget_baseline,
                effective_at=budget_window_now,
            )
            if budget_outcome is None:
                continue
            if budget_limit.action == "notify":
                if not _budget_notify_already_emitted_in_invocation(
                    budget_notify_events or [],
                    check=budget_outcome.check,
                ):
                    event = await self._emit_budget_limit_reached(
                        session=session,
                        agent_name=agent_name,
                        environment_name=environment_name,
                        check=budget_outcome.check,
                    )
                    emitted_events.append(event)
                    if budget_notify_events is not None:
                        budget_notify_events.append(event)
                continue
            return LimitEvaluation(
                decision=budget_outcome.decision,
                usage_summary=usage_summary,
                cost_summary=cost_summary,
                events=tuple(emitted_events),
            )
        return LimitEvaluation(
            decision=None,
            usage_summary=usage_summary,
            cost_summary=cost_summary,
            events=tuple(emitted_events),
        )

    async def evaluate_policy_budgets(
        self,
        *,
        session: Session,
        agent_name: str,
        environment_name: str | None,
        budget_policy: BudgetPolicy | None,
    ) -> BudgetEvaluation:
        limits = budget_limits_for_session(
            policy=budget_policy,
            agent_name=agent_name,
            causal_budget_id=session.causal_budget_id,
        )
        if not limits:
            return BudgetEvaluation(check=None)
        emitted_events: list[Event] = []
        for limit in limits:
            events = await self._budget_store.load_events_for_budget(
                scope=limit.scope,
                key=limit.key,
                window=limit.window,
            )
            check = budget_check_from_events(
                limit=limit,
                events=events,
                provider_name=session.provider_name,
                model=session.model,
                effective_at=self._clock(),
            )
            emitted_events.append(
                await self._event_writer.emit(
                    Event(
                        type=EventType.BUDGET_CHECKED,
                        session_id=session.id,
                        agent_name=agent_name,
                        environment_name=environment_name,
                        payload=budget_check_payload(check),
                    )
                )
            )
            if not check.limit_reached:
                continue
            if limit.action == "notify":
                if not await self._budget_notify_already_emitted(limit=limit, check=check):
                    emitted_events.append(
                        await self._emit_budget_limit_reached(
                            session=session,
                            agent_name=agent_name,
                            environment_name=environment_name,
                            check=check,
                        )
                    )
                continue
            return BudgetEvaluation(check=check, events=tuple(emitted_events))
        return BudgetEvaluation(check=None, events=tuple(emitted_events))

    async def _budget_notify_already_emitted(
        self,
        *,
        limit: BudgetLimit,
        check: BudgetCheck,
    ) -> bool:
        if type(limit) is not BudgetLimit:
            raise TypeError("limit must be a BudgetLimit instance.")
        if type(check) is not BudgetCheck:
            raise TypeError("check must be a BudgetCheck instance.")
        if limit.action != "notify":
            return False

        since, until = limit.window.bounds()
        agent_name: str | None = None
        causal_budget_id: str | None = None
        if limit.scope == "agent":
            agent_name = require_clean_nonblank(limit.key or "", "key")
        elif limit.scope == "causal":
            causal_budget_id = require_clean_nonblank(limit.key or "", "key")
        elif limit.scope != "app":
            return False

        records = await query_all_event_records(
            self._session_store,
            EventQuery(
                causal_budget_id=causal_budget_id,
                event_type=EventType.BUDGET_LIMIT_REACHED,
                agent_name=agent_name,
                since=since,
                until=until,
                limit=5000,
            ),
        )
        return any(
            _budget_limit_reached_payload_matches(record.event.payload, check=check)
            for record in records
        )

    async def _emit_budget_limit_reached(
        self,
        *,
        session: Session,
        agent_name: str,
        environment_name: str | None,
        check: BudgetCheck,
    ) -> Event:
        return await self._event_writer.emit(
            Event(
                type=EventType.BUDGET_LIMIT_REACHED,
                session_id=session.id,
                agent_name=agent_name,
                environment_name=environment_name,
                payload=budget_limit_reached_payload(check),
            )
        )

    async def reserve_for_model_step(
        self,
        *,
        session: Session,
        agent_name: str,
        provider_name: str,
        environment_name: str | None,
        budget_policy: BudgetPolicy | None,
        request_budget_limits: tuple[BudgetLimit, ...] = (),
    ) -> BudgetReservationSetup:
        limits = self.provider_reservation_limits(
            session=session,
            agent_name=agent_name,
            budget_policy=budget_policy,
            request_budget_limits=request_budget_limits,
        )
        if not limits:
            return BudgetReservationSetup((), None, (), None)

        reservations: list[BudgetStepReservation] = []
        emitted_events: list[Event] = []
        reservation_failure: BudgetReservationResult | None = None
        release_reason = "reservation setup failed"
        try:
            for limit in limits:
                result = await self._budget_ledger.reserve(
                    limit=limit,
                    session_id=session.id,
                    agent_name=agent_name,
                    provider_name=provider_name,
                    model=session.model,
                )
                if result.accepted:
                    if result.record is None:
                        raise RuntimeError("Accepted budget reservation did not return a record.")
                    reservations.append(BudgetStepReservation(limit=limit, record=result.record))
                emitted_events.append(
                    await self._event_writer.emit(
                        Event(
                            type=(
                                EventType.BUDGET_RESERVED
                                if result.accepted
                                else EventType.BUDGET_RESERVATION_FAILED
                            ),
                            session_id=session.id,
                            agent_name=agent_name,
                            environment_name=environment_name,
                            payload=budget_reservation_payload(result),
                        )
                    )
                )
                if not result.accepted:
                    reservation_failure = result
                    release_reason = "reservation failed"
                    async for event in self.release_reservations(
                        reservations,
                        session=session,
                        agent_name=agent_name,
                        environment_name=environment_name,
                        reason=release_reason,
                    ):
                        emitted_events.append(event)
                    return BudgetReservationSetup(
                        tuple(reservations),
                        result,
                        tuple(emitted_events),
                        None,
                    )
        except BaseException as reservation_exc:
            async for event in self.settlement_events_preserving_failure(
                self.release_reservations(
                    reservations,
                    session=session,
                    agent_name=agent_name,
                    environment_name=environment_name,
                    reason=release_reason,
                ),
                authoritative_failure=reservation_exc,
            ):
                emitted_events.append(event)
            if not isinstance(reservation_exc, Exception):
                raise
            return BudgetReservationSetup(
                tuple(reservations),
                reservation_failure,
                tuple(emitted_events),
                reservation_exc,
            )
        return BudgetReservationSetup(tuple(reservations), None, tuple(emitted_events), None)

    def provider_reservation_limits(
        self,
        *,
        session: Session,
        agent_name: str,
        budget_policy: BudgetPolicy | None,
        request_budget_limits: tuple[BudgetLimit, ...] = (),
    ) -> tuple[BudgetLimit, ...]:
        return tuple(
            limit
            for limit in (
                *budget_limits_for_session(
                    policy=budget_policy,
                    agent_name=agent_name,
                    causal_budget_id=session.causal_budget_id,
                ),
                *request_budget_limits,
            )
            if limit.reservation is not None
        )

    async def reconcile_dispatched_reservations(
        self,
        reservations: list[BudgetStepReservation],
        *,
        lifecycle: BudgetModelStepLifecycle,
        session: Session,
        agent_name: str,
        environment_name: str | None,
        unknown_reason: str,
    ) -> AsyncIterator[Event]:
        if not lifecycle.dispatches:
            raise ValueError("Cannot reconcile a model step with no provider dispatch.")
        active_lifecycle_reservation_ids = {
            reservation.record.reservation_id
            for dispatch in lifecycle.dispatches
            for reservation in dispatch.reservations
            if reservation.record.reservation_id not in dispatch.settled_reservation_ids
        }
        if lifecycle.pending_reservations is not None:
            active_lifecycle_reservation_ids.update(
                reservation.record.reservation_id for reservation in lifecycle.pending_reservations
            )
        if active_lifecycle_reservation_ids != {
            reservation.record.reservation_id for reservation in reservations
        }:
            raise RuntimeError("Budget dispatch lifecycle lost a reservation.")

        for dispatch in lifecycle.dispatches:
            if dispatch.settled:
                continue
            emitted_events: list[Event] = []
            settlement_failure: Exception | None = None
            async with lifecycle.reservation_transition_lock:
                for reservation in dispatch.reservations:
                    reservation_id = reservation.record.reservation_id
                    if reservation_id in dispatch.settled_reservation_ids:
                        continue
                    priced_actual = None
                    if dispatch.completion is None:
                        actual_amount = reservation.record.reserved_amount
                        reason = unknown_reason
                        reconciled_at = self._clock()
                    else:
                        reconciled_at = dispatch.completion.timestamp
                        try:
                            priced_actual = budget_actual_cost_for_event(
                                limit=reservation.limit,
                                event=dispatch.completion,
                            )
                        except ValueError:
                            actual_amount = reservation.record.reserved_amount
                            reason = "model completed without priced usage; charged reserved amount"
                        else:
                            actual_amount = priced_actual.amount
                            reason = "model completed"
                    try:
                        reconciliation = await self._budget_ledger.reconcile(
                            reservation_id=reservation_id,
                            actual_amount=actual_amount,
                            reason=reason,
                            occurred_at=reconciled_at,
                        )
                        if priced_actual is not None:
                            reconciliation = budget_reconciliation_with_pricing(
                                reconciliation,
                                priced_actual.line_item,
                            )
                        emitted_events.append(
                            await self._event_writer.emit(
                                Event(
                                    type=EventType.BUDGET_RECONCILED,
                                    session_id=session.id,
                                    agent_name=agent_name,
                                    environment_name=environment_name,
                                    payload=budget_reconciliation_payload(reconciliation),
                                )
                            )
                        )
                    except Exception as exc:
                        settlement_failure = exc
                        break
                    dispatch.settled_reservation_ids.add(reservation_id)
                    reservations[:] = [
                        active
                        for active in reservations
                        if active.record.reservation_id != reservation_id
                    ]

            for event in emitted_events:
                yield event
            if settlement_failure is not None:
                raise settlement_failure

    async def settle_after_model_failure(
        self,
        reservations: list[BudgetStepReservation],
        *,
        lifecycle: BudgetModelStepLifecycle,
        session: Session,
        agent_name: str,
        environment_name: str | None,
        release_reason: str,
        unknown_reason: str = UNKNOWN_POST_DISPATCH_BUDGET_REASON,
    ) -> AsyncIterator[Event]:
        if lifecycle.provider_dispatch_may_have_occurred:
            async for event in self.reconcile_dispatched_reservations(
                reservations,
                lifecycle=lifecycle,
                session=session,
                agent_name=agent_name,
                environment_name=environment_name,
                unknown_reason=unknown_reason,
            ):
                yield event

        if lifecycle.pending_reservations is not None:
            async for event in self.release_reservations(
                list(lifecycle.pending_reservations),
                session=session,
                agent_name=agent_name,
                environment_name=environment_name,
                reason=release_reason,
            ):
                yield event

    async def settlement_events_preserving_failure(
        self,
        settlement_events: AsyncIterator[Event],
        *,
        authoritative_failure: BaseException,
    ) -> AsyncIterator[Event]:
        try:
            async for event in settlement_events:
                yield event
        except Exception as accounting_exc:
            add_budget_failure_note(
                authoritative_failure,
                operation="settlement",
                accounting_failure=accounting_exc,
            )

    async def before_provider_dispatch(
        self,
        reservations: list[BudgetStepReservation],
        *,
        lifecycle: BudgetModelStepLifecycle,
    ) -> None:
        async with lifecycle.reservation_transition_lock:
            if reservations and self._budget_ledger.reservation_ttl_seconds is not None:
                try:
                    await self.renew_reservations(reservations)
                except BudgetReservationLeaseLost as exc:
                    if not lifecycle.provider_dispatch_may_have_occurred:
                        raise BudgetReservationLeaseLostBeforeModelDispatch(
                            "Budget reservation lease was lost before model dispatch."
                        ) from exc
                    raise
            lifecycle.mark_provider_dispatch()

    async def model_step_events_with_heartbeat(
        self,
        model_step_events: AsyncIterator[tuple[Event | None, _StreamResultT | None]],
        *,
        reservations: list[BudgetStepReservation],
        lifecycle: BudgetModelStepLifecycle,
    ) -> AsyncIterator[tuple[Event | None, _StreamResultT | None]]:
        ttl_seconds = self._budget_ledger.reservation_ttl_seconds
        if not reservations or ttl_seconds is None:
            async for item in model_step_events:
                yield item
            return

        heartbeat_task = asyncio.create_task(
            self.heartbeat_reservations(
                reservations,
                reservation_transition_lock=lifecycle.reservation_transition_lock,
                interval_seconds=ttl_seconds / 3,
            )
        )
        iterator = model_step_events.__aiter__()
        next_item_task: asyncio.Task[tuple[Event | None, _StreamResultT | None]] | None = None
        exhausted = False
        try:
            while True:
                if heartbeat_task.done():
                    heartbeat_failure = budget_heartbeat_task_failure(heartbeat_task)
                    if not lifecycle.provider_dispatch_may_have_occurred:
                        raise BudgetReservationLeaseLostBeforeModelDispatch(
                            "Budget reservation lease was lost before model dispatch."
                        ) from heartbeat_failure
                    raise heartbeat_failure

                next_item_task = asyncio.create_task(_next_model_step_item(iterator))
                done, _ = await asyncio.wait(
                    {next_item_task, heartbeat_task},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if heartbeat_task in done:
                    heartbeat_failure = budget_heartbeat_task_failure(heartbeat_task)
                    completed_item: tuple[Event | None, _StreamResultT | None] | None = None
                    provider_failure: Exception | None = None
                    if not next_item_task.done():
                        next_item_task.cancel()
                        try:
                            await next_item_task
                        except asyncio.CancelledError:
                            pass
                        except Exception as exc:
                            provider_failure = exc
                    else:
                        try:
                            completed_item = next_item_task.result()
                        except (StopAsyncIteration, asyncio.CancelledError):
                            pass
                        except Exception as exc:
                            provider_failure = exc
                    if provider_failure is not None:
                        heartbeat_failure.add_note(
                            "Provider iterator failed while budget lease loss was being "
                            f"handled: {type(provider_failure).__name__}: {provider_failure}"
                        )
                    if completed_item is not None:
                        next_item_task = None
                        yield completed_item
                    if not lifecycle.provider_dispatch_may_have_occurred:
                        raise BudgetReservationLeaseLostBeforeModelDispatch(
                            "Budget reservation lease was lost before model dispatch."
                        ) from heartbeat_failure
                    raise heartbeat_failure
                try:
                    item = next_item_task.result()
                except StopAsyncIteration:
                    exhausted = True
                    break
                next_item_task = None
                yield item

            heartbeat_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await heartbeat_task
            async with lifecycle.reservation_transition_lock:
                await self.renew_reservations(reservations)
        finally:
            if next_item_task is not None and not next_item_task.done():
                next_item_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await next_item_task
            if not heartbeat_task.done():
                heartbeat_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await heartbeat_task
            if not exhausted:
                close = getattr(iterator, "aclose", None)
                if close is not None:
                    with contextlib.suppress(Exception):
                        await close()

    async def heartbeat_reservations(
        self,
        reservations: list[BudgetStepReservation],
        *,
        reservation_transition_lock: asyncio.Lock | None,
        interval_seconds: float,
    ) -> None:
        while True:
            await asyncio.sleep(interval_seconds)
            if reservation_transition_lock is None:
                await self.renew_reservations(reservations)
            else:
                async with reservation_transition_lock:
                    await self.renew_reservations(reservations)

    async def renew_reservations(
        self,
        reservations: list[BudgetStepReservation],
    ) -> None:
        for reservation in reservations:
            reservation_id = reservation.record.reservation_id
            try:
                renewed = await self._budget_ledger.heartbeat(
                    reservation_id=reservation_id,
                )
            except Exception as exc:
                raise BudgetReservationLeaseLost(
                    f"Could not renew budget reservation: {reservation_id}"
                ) from exc
            if not renewed:
                raise BudgetReservationLeaseLost(
                    f"Budget reservation lease was lost: {reservation_id}"
                )

    async def run_operation_with_reservation_heartbeat(
        self,
        operation: Callable[[], Awaitable[_OperationResultT]],
        *,
        reservations: list[BudgetStepReservation],
        authoritative_failure_types: tuple[type[BaseException], ...],
        lease_lost_before_dispatch_message: str,
        authoritative_failure_note: str,
        concurrent_failure_note: str,
    ) -> tuple[_OperationResultT, BaseException | None]:
        if not reservations:
            return await operation(), None
        ttl_seconds = self._budget_ledger.reservation_ttl_seconds
        if ttl_seconds is None:
            return await operation(), None
        try:
            await self.renew_reservations(reservations)
        except BudgetReservationLeaseLost as exc:
            raise BudgetReservationLeaseLostBeforeModelDispatch(
                lease_lost_before_dispatch_message
            ) from exc

        async def await_operation() -> _OperationResultT:
            return await operation()

        operation_task = asyncio.create_task(await_operation())
        heartbeat_task = asyncio.create_task(
            self.heartbeat_reservations(
                reservations,
                reservation_transition_lock=None,
                interval_seconds=ttl_seconds / 3,
            )
        )
        try:
            done, _ = await asyncio.wait(
                {operation_task, heartbeat_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if heartbeat_task in done:
                failure = budget_heartbeat_task_failure(heartbeat_task)
                if operation_task.done():
                    try:
                        return operation_task.result(), failure
                    except BaseException as operation_failure:
                        if isinstance(operation_failure, authoritative_failure_types):
                            operation_failure.add_note(f"{authoritative_failure_note}: {failure}")
                            raise operation_failure from failure
                        failure.add_note(
                            f"{concurrent_failure_note}: "
                            f"{type(operation_failure).__name__}: {operation_failure}"
                        )
                        raise failure from operation_failure
                operation_task.cancel()
                try:
                    return await operation_task, failure
                except asyncio.CancelledError:
                    pass
                except BaseException as operation_failure:
                    if isinstance(operation_failure, authoritative_failure_types):
                        operation_failure.add_note(f"{authoritative_failure_note}: {failure}")
                        raise operation_failure from failure
                    failure.add_note(
                        f"{concurrent_failure_note}: "
                        f"{type(operation_failure).__name__}: {operation_failure}"
                    )
                raise failure
            result = operation_task.result()
            heartbeat_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await heartbeat_task
            try:
                await self.renew_reservations(reservations)
            except BudgetReservationLeaseLost as exc:
                return result, exc
            return result, None
        finally:
            if not operation_task.done():
                operation_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await operation_task
            if not heartbeat_task.done():
                heartbeat_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await heartbeat_task

    async def run_automatic_compaction_dispatch(
        self,
        operation: Callable[[], Awaitable[_OperationResultT]],
        *,
        completed_events: Callable[[], list[Event]],
        budget_limits: tuple[BudgetLimit, ...],
        session: Session,
        agent_name: str,
        environment_name: str | None,
        provider_name: str,
        model: str,
        authoritative_failure_types: tuple[type[BaseException], ...],
    ) -> (
        BudgetedOperationSucceeded[_OperationResultT]
        | BudgetedOperationRejected
        | BudgetedOperationFailed
    ):
        """Run one observable compactor dispatch under strict budget accounting."""

        lifecycle = _BudgetedOperationLifecycle()
        result: _OperationResultT | None = None
        reservation_failure: BudgetReservationResult | None = None
        authoritative_failure: BaseException | None = None
        authoritative_cause: BaseException | None = None
        lease_failure: BaseException | None = None

        try:
            setup = await self.reserve_operation_budgets(
                budget_limits=budget_limits,
                session_id=session.id,
                agent_name=agent_name,
                provider_name=provider_name,
                model=model,
                rejection_release_reason="reservation failed",
                accepted_record_error=(
                    "Accepted automatic compaction budget reservation has no record."
                ),
            )
            lifecycle.reservations.extend(setup.reservations)
            for reservation_result in setup.results:
                lifecycle.events.append(
                    await self._event_writer.emit(
                        Event(
                            type=(
                                EventType.BUDGET_RESERVED
                                if reservation_result.accepted
                                else EventType.BUDGET_RESERVATION_FAILED
                            ),
                            session_id=session.id,
                            agent_name=agent_name,
                            environment_name=environment_name,
                            payload=budget_reservation_payload(reservation_result),
                        )
                    )
                )
            for reconciliation in setup.releases:
                lifecycle.events.append(
                    await self._event_writer.emit(
                        Event(
                            type=EventType.BUDGET_RESERVATION_RELEASED,
                            session_id=session.id,
                            agent_name=agent_name,
                            environment_name=environment_name,
                            payload=budget_reconciliation_payload(reconciliation),
                        )
                    )
                )
            if setup.error is not None:
                raise setup.error
            if setup.failure is not None:
                reservation_failure = setup.failure
            else:

                async def run_dispatched_operation() -> _OperationResultT:
                    lifecycle.provider_dispatch_started = True
                    return await operation()

                result, lease_failure = await self.run_operation_with_reservation_heartbeat(
                    run_dispatched_operation,
                    reservations=lifecycle.reservations,
                    authoritative_failure_types=authoritative_failure_types,
                    lease_lost_before_dispatch_message=(
                        "Compaction budget reservation lease was lost before provider dispatch."
                    ),
                    authoritative_failure_note=(
                        "Budget reservation lease was also lost as compaction failed"
                    ),
                    concurrent_failure_note=(
                        "Compactor also failed while reservation lease loss was handled"
                    ),
                )
        except BaseException as exc:
            authoritative_failure = exc
            authoritative_cause = exc.__cause__
            if not lifecycle.provider_dispatch_started:
                lifecycle.predispatch_release_reason = (
                    "reservation setup cancelled"
                    if isinstance(exc, asyncio.CancelledError)
                    else "reservation setup failed"
                )

        completion_events: list[Event] = []
        if lifecycle.provider_dispatch_started:
            try:
                completion_events = completed_events()
            except BaseException as evidence_failure:
                if authoritative_failure is None:
                    authoritative_failure = evidence_failure
                    authoritative_cause = evidence_failure.__cause__
                else:
                    authoritative_failure.add_note(
                        "Automatic compaction completion evidence also failed: "
                        f"{type(evidence_failure).__name__}: {evidence_failure}"
                    )

        (
            settlement_cancellation,
            settlement_failure,
        ) = await self._settle_budgeted_operation_resisting_cancellation(
            lifecycle=lifecycle,
            completed_events=completion_events,
            session=session,
            agent_name=agent_name,
            environment_name=environment_name,
        )

        if settlement_cancellation is not None:
            propagated_cancellation = (
                authoritative_failure
                if isinstance(authoritative_failure, asyncio.CancelledError)
                else settlement_cancellation
            )
            cancellation_cause: BaseException | None = None
            if settlement_failure is not None:
                propagated_cancellation.add_note(
                    "Automatic compaction budget settlement also failed: "
                    f"{type(settlement_failure).__name__}: {settlement_failure}"
                )
                cancellation_cause = settlement_failure
            elif (
                authoritative_failure is not None
                and authoritative_failure is not propagated_cancellation
            ):
                propagated_cancellation.add_note(
                    "Automatic compaction had already failed before cancellation: "
                    f"{type(authoritative_failure).__name__}: {authoritative_failure}"
                )
                cancellation_cause = authoritative_failure
            return BudgetedOperationFailed(
                error=propagated_cancellation,
                cause=cancellation_cause,
                events=tuple(lifecycle.events),
            )

        if settlement_failure is not None:
            if authoritative_failure is not None:
                authoritative_failure.__dict__["_cayu_compaction_budget_settlement_failed"] = True
                if isinstance(authoritative_failure, ModelProviderError):
                    authoritative_failure.retryable = False
                authoritative_failure.add_note(
                    "Automatic compaction budget settlement also failed: "
                    f"{type(settlement_failure).__name__}: {settlement_failure}"
                )
                return BudgetedOperationFailed(
                    error=authoritative_failure,
                    cause=settlement_failure,
                    events=tuple(lifecycle.events),
                )
            return BudgetedOperationFailed(
                error=settlement_failure,
                cause=None,
                events=tuple(lifecycle.events),
            )

        if authoritative_failure is not None:
            return BudgetedOperationFailed(
                error=authoritative_failure,
                cause=authoritative_cause,
                events=tuple(lifecycle.events),
            )
        if lease_failure is not None:
            return BudgetedOperationFailed(
                error=lease_failure,
                cause=lease_failure.__cause__,
                events=tuple(lifecycle.events),
            )
        if reservation_failure is not None:
            return BudgetedOperationRejected(
                failure=reservation_failure,
                events=tuple(lifecycle.events),
            )
        if result is None:
            return BudgetedOperationFailed(
                error=RuntimeError("Automatic compaction completed without a result."),
                cause=None,
                events=tuple(lifecycle.events),
            )
        return BudgetedOperationSucceeded(result=result, events=tuple(lifecycle.events))

    async def _settle_budgeted_operation_resisting_cancellation(
        self,
        *,
        lifecycle: _BudgetedOperationLifecycle,
        completed_events: list[Event],
        session: Session,
        agent_name: str,
        environment_name: str | None,
    ) -> tuple[asyncio.CancelledError | None, BaseException | None]:
        settlement_task = asyncio.create_task(
            self._settle_budgeted_operation(
                lifecycle=lifecycle,
                completed_events=completed_events,
                session=session,
                agent_name=agent_name,
                environment_name=environment_name,
            )
        )
        cancellation: asyncio.CancelledError | None = None
        settlement_failure: BaseException | None = None
        while not settlement_task.done():
            try:
                await asyncio.shield(settlement_task)
            except asyncio.CancelledError as exc:
                if settlement_task.cancelled():
                    settlement_failure = exc
                    break
                if cancellation is None:
                    cancellation = exc
            except BaseException as exc:
                settlement_failure = exc
                break
        if settlement_failure is None:
            try:
                settlement_task.result()
            except BaseException as exc:
                settlement_failure = exc
        return cancellation, settlement_failure

    async def _settle_budgeted_operation(
        self,
        *,
        lifecycle: _BudgetedOperationLifecycle,
        completed_events: list[Event],
        session: Session,
        agent_name: str,
        environment_name: str | None,
    ) -> None:
        if lifecycle.settled:
            return
        if not lifecycle.reservations:
            lifecycle.settled = True
            return
        settlement_failures: list[tuple[str, Exception]] = []

        def raise_settlement_failure() -> None:
            if not settlement_failures:
                return
            first_reservation_id, first_failure = settlement_failures[0]
            first_failure.add_note(
                "Automatic compaction budget settlement failed for reservation "
                f"{first_reservation_id}."
            )
            for reservation_id, failure in settlement_failures[1:]:
                first_failure.add_note(
                    "Additional automatic compaction budget settlement failure for "
                    f"reservation {reservation_id}: {type(failure).__name__}: {failure}"
                )
            raise first_failure

        if not lifecycle.provider_dispatch_started:
            for reservation in list(lifecycle.reservations):
                try:
                    working_reservations = [reservation]
                    async for reconciliation in self.release_operation_reservations(
                        working_reservations,
                        reason=lifecycle.predispatch_release_reason,
                    ):
                        lifecycle.events.append(
                            await self._event_writer.emit(
                                Event(
                                    type=EventType.BUDGET_RESERVATION_RELEASED,
                                    session_id=session.id,
                                    agent_name=agent_name,
                                    environment_name=environment_name,
                                    payload=budget_reconciliation_payload(reconciliation),
                                )
                            )
                        )
                except Exception as exc:
                    settlement_failures.append((reservation.record.reservation_id, exc))
            lifecycle.settled = True
            raise_settlement_failure()
            return

        for reservation in list(lifecycle.reservations):
            try:
                priced_actuals = []
                uncertain_completion_count = 0
                for event in completed_events:
                    try:
                        priced_actuals.append(
                            budget_actual_cost_for_event(limit=reservation.limit, event=event)
                        )
                    except ValueError:
                        uncertain_completion_count += 1
                if not completed_events:
                    actual_amount = reservation.record.reserved_amount
                    reason = (
                        "automatic context compaction dispatch has uncertain usage; "
                        "charged reserved amount"
                    )
                else:
                    actual_amount = sum(
                        (priced.amount for priced in priced_actuals),
                        start=(reservation.record.reserved_amount * uncertain_completion_count),
                    )
                    if uncertain_completion_count:
                        reason = (
                            "automatic context compaction completed with partially uncertain "
                            "usage; charged known cost plus reserved amount per uncertain "
                            "completion"
                        )
                    else:
                        reason = "automatic context compaction model completed"
                reconciliation = await self._budget_ledger.reconcile(
                    reservation_id=reservation.record.reservation_id,
                    actual_amount=actual_amount,
                    reason=reason,
                    occurred_at=(
                        completed_events[-1].timestamp if completed_events else self._clock()
                    ),
                )
                if len(completed_events) == 1 and len(priced_actuals) == 1:
                    reconciliation = budget_reconciliation_with_pricing(
                        reconciliation,
                        priced_actuals[0].line_item,
                    )
                lifecycle.events.append(
                    await self._event_writer.emit(
                        Event(
                            type=EventType.BUDGET_RECONCILED,
                            session_id=session.id,
                            agent_name=agent_name,
                            environment_name=environment_name,
                            payload=budget_reconciliation_payload(reconciliation),
                        )
                    )
                )
            except Exception as exc:
                settlement_failures.append((reservation.record.reservation_id, exc))
        lifecycle.settled = True
        raise_settlement_failure()

    async def reserve_operation_budgets(
        self,
        *,
        budget_limits: tuple[BudgetLimit, ...],
        session_id: str,
        agent_name: str,
        provider_name: str | None,
        model: str | None,
        rejection_release_reason: str,
        accepted_record_error: str,
    ) -> OperationReservationSetup:
        limits = [limit for limit in budget_limits if limit.reservation is not None]
        if not limits or provider_name is None or model is None:
            return OperationReservationSetup((), (), (), None, None)

        reservations: list[BudgetStepReservation] = []
        results: list[BudgetReservationResult] = []
        releases: list[BudgetReconciliation] = []
        for limit in limits:
            try:
                result = await self._budget_ledger.reserve(
                    limit=limit,
                    session_id=session_id,
                    agent_name=agent_name,
                    provider_name=provider_name,
                    model=model,
                )
                results.append(result)
                if not result.accepted:
                    async for reconciliation in self.release_operation_reservations(
                        reservations,
                        reason=rejection_release_reason,
                    ):
                        releases.append(reconciliation)
                    return OperationReservationSetup(
                        reservations=(),
                        results=tuple(results),
                        releases=tuple(releases),
                        failure=result,
                        error=None,
                    )
                if result.record is None:
                    raise RuntimeError(accepted_record_error)
                reservations.append(BudgetStepReservation(limit=limit, record=result.record))
            except BaseException as exc:
                return OperationReservationSetup(
                    reservations=tuple(reservations),
                    results=tuple(results),
                    releases=tuple(releases),
                    failure=(results[-1] if results and not results[-1].accepted else None),
                    error=exc,
                )
        return OperationReservationSetup(
            reservations=tuple(reservations),
            results=tuple(results),
            releases=(),
            failure=None,
            error=None,
        )

    async def reconcile_operation_reservations(
        self,
        reservations: list[BudgetStepReservation],
        *,
        model_completed_events: list[Event],
        completed_reason: str,
        missing_usage_reason: str,
    ) -> AsyncIterator[BudgetReconciliation]:
        for reservation in list(reservations):
            reconciliation = await self._reconcile_operation_reservation(
                reservation,
                model_completed_events=model_completed_events,
                completed_reason=completed_reason,
                missing_usage_reason=missing_usage_reason,
            )
            _remove_reservation(reservations, reservation)
            yield reconciliation

    async def _reconcile_operation_reservation(
        self,
        reservation: BudgetStepReservation,
        *,
        model_completed_events: list[Event],
        completed_reason: str,
        missing_usage_reason: str,
    ) -> BudgetReconciliation:
        priced_actuals = []
        try:
            priced_actuals = [
                budget_actual_cost_for_event(limit=reservation.limit, event=event)
                for event in model_completed_events
            ]
            if not priced_actuals:
                raise ValueError("Operation completed without model usage.")
            actual_amount = sum(
                (priced.amount for priced in priced_actuals),
                start=Decimal("0"),
            )
            reason = completed_reason
        except ValueError:
            actual_amount = reservation.record.reserved_amount
            reason = missing_usage_reason
        reconciliation = await self._budget_ledger.reconcile(
            reservation_id=reservation.record.reservation_id,
            actual_amount=actual_amount,
            reason=reason,
            occurred_at=(
                model_completed_events[-1].timestamp if model_completed_events else self._clock()
            ),
        )
        if len(priced_actuals) == 1:
            reconciliation = budget_reconciliation_with_pricing(
                reconciliation,
                priced_actuals[0].line_item,
            )
        return reconciliation

    async def reconcile_uncertain_operation_reservations(
        self,
        reservations: list[BudgetStepReservation],
        *,
        reason: str,
    ) -> AsyncIterator[BudgetReconciliation]:
        for reservation in list(reservations):
            reconciliation = await self._budget_ledger.reconcile(
                reservation_id=reservation.record.reservation_id,
                actual_amount=reservation.record.reserved_amount,
                reason=reason,
                occurred_at=self._clock(),
            )
            _remove_reservation(reservations, reservation)
            yield reconciliation

    async def release_operation_reservations(
        self,
        reservations: list[BudgetStepReservation],
        *,
        reason: str,
    ) -> AsyncIterator[BudgetReconciliation]:
        for reservation in list(reservations):
            reconciliation = await self._budget_ledger.release(
                reservation_id=reservation.record.reservation_id,
                reason=reason,
            )
            _remove_reservation(reservations, reservation)
            yield reconciliation

    async def release_reservations(
        self,
        reservations: list[BudgetStepReservation],
        *,
        session: Session,
        agent_name: str,
        environment_name: str | None,
        reason: str,
    ) -> AsyncIterator[Event]:
        for reservation in list(reservations):
            reconciliation = await self._budget_ledger.release(
                reservation_id=reservation.record.reservation_id,
                reason=reason,
            )
            event = await self._event_writer.emit(
                Event(
                    type=EventType.BUDGET_RESERVATION_RELEASED,
                    session_id=session.id,
                    agent_name=agent_name,
                    environment_name=environment_name,
                    payload=budget_reconciliation_payload(reconciliation),
                )
            )
            _remove_reservation(reservations, reservation)
            yield event


class RunLimitGate:
    """Retain one run's limit inputs and incremental usage watermark."""

    def __init__(
        self,
        controller: RunLimitController,
        *,
        session: Session,
        agent_name: str,
        environment_name: str | None,
        limits: RunLimits,
        budget_limits: tuple[BudgetLimit, ...],
        run_started_at: float,
        run_baseline: SessionUsageSummary | None,
        budget_baseline_events: list[Event],
        budget_notify_events: list[Event],
    ) -> None:
        self._controller = controller
        self._session = session
        self._agent_name = agent_name
        self._environment_name = environment_name
        self._limits = limits
        self._budget_limits = budget_limits
        self._run_started_at = run_started_at
        self._run_baseline = run_baseline
        self._budget_baseline_events = budget_baseline_events
        self._budget_notify_events = budget_notify_events
        self._usage_tracker = controller.usage_tracker(session.id)

    async def evaluate_limits(self, *, pending_tool_calls: int = 0) -> LimitEvaluation:
        return await self._controller.evaluate_request_limits(
            session=self._session,
            agent_name=self._agent_name,
            environment_name=self._environment_name,
            limits=self._limits,
            budget_limits=self._budget_limits,
            run_started_at=self._run_started_at,
            run_baseline=self._run_baseline,
            budget_baseline_events=self._budget_baseline_events,
            pending_tool_calls=pending_tool_calls,
            budget_notify_events=self._budget_notify_events,
            usage_tracker=self._usage_tracker,
        )

    async def evaluate_budget(self, budget_policy: BudgetPolicy | None) -> BudgetEvaluation:
        return await self._controller.evaluate_policy_budgets(
            session=self._session,
            agent_name=self._agent_name,
            environment_name=self._environment_name,
            budget_policy=budget_policy,
        )


def _latest_model_event_identity(events: list[Event]) -> tuple[str | None, str | None]:
    for event in reversed(events):
        if event.type != EventType.MODEL_COMPLETED:
            continue
        provider_name = event.payload.get("provider_name")
        model = event.payload.get("model") or event.payload.get("requested_model")
        return (
            provider_name if type(provider_name) is str else None,
            model if type(model) is str else None,
        )
    return None, None


def _remove_reservation(
    reservations: list[BudgetStepReservation],
    settled: BudgetStepReservation,
) -> None:
    reservation_id = settled.record.reservation_id
    reservations[:] = [
        reservation
        for reservation in reservations
        if reservation.record.reservation_id != reservation_id
    ]


def budget_limit_reached_payload(check: BudgetCheck) -> dict[str, object]:
    if type(check) is not BudgetCheck:
        raise TypeError("check must be a BudgetCheck.")
    return budget_check_payload(check)


def _budget_limit_reached_payload_matches(
    payload: dict[str, object],
    *,
    check: BudgetCheck,
) -> bool:
    if not isinstance(payload, dict):
        return False
    if type(check) is not BudgetCheck:
        raise TypeError("check must be a BudgetCheck.")
    return (
        payload.get("scope") == check.scope
        and payload.get("key") == check.key
        and payload.get("window") == check.window.storage_key
        and payload.get("currency") == check.currency
        and payload.get("maximum") == str(check.maximum)
        and payload.get("action") == check.action
    )


def _budget_notify_already_emitted_in_invocation(
    events: list[Event],
    *,
    check: BudgetCheck,
) -> bool:
    if type(check) is not BudgetCheck:
        raise TypeError("check must be a BudgetCheck.")
    return any(
        event.type == EventType.BUDGET_LIMIT_REACHED
        and _budget_limit_reached_payload_matches(event.payload, check=check)
        for event in events
    )


def _first_budget_limit_outcome(
    *,
    session: Session,
    limit: BudgetLimit,
    cost_summary: SessionCostSummary,
    cost_baseline: SessionCostSummary | None,
    effective_at: datetime,
) -> _BudgetLimitOutcome | None:
    if type(session) is not Session:
        raise TypeError("session must be a Session instance.")
    if type(limit) is not BudgetLimit:
        raise TypeError("limit must be a BudgetLimit instance.")
    if type(cost_summary) is not SessionCostSummary:
        raise TypeError("cost_summary must be a SessionCostSummary.")
    if cost_baseline is not None and type(cost_baseline) is not SessionCostSummary:
        raise TypeError("cost_baseline must be a SessionCostSummary.")

    actual_cost = cost_summary.total_cost
    unpriced_model_steps = cost_summary.unpriced_model_steps
    if limit.scope == "run" and cost_baseline is not None:
        actual_cost = max(
            cost_summary.total_cost - cost_baseline.total_cost,
            Decimal("0"),
        )
        unpriced_model_steps = max(
            unpriced_model_steps - cost_baseline.unpriced_model_steps,
            0,
        )

    if unpriced_model_steps > 0 and not limit.allow_unpriced:
        decision = StopDecision(
            limit=StopLimit.ESTIMATED_COST,
            maximum=limit.max_estimated_cost,
            actual=actual_cost,
            message=(
                "Estimated cost budget cannot be verified because "
                f"{unpriced_model_steps} model step(s) have no matching pricing."
            ),
        )
        return _BudgetLimitOutcome(
            decision=decision,
            check=_budget_check_from_stop_decision(
                limit=limit,
                decision=decision,
                cost_summary=cost_summary,
                unpriced_model_steps=unpriced_model_steps,
            ),
        )
    preflight_error = _budget_limit_preflight_error(
        session=session,
        limit=limit,
        effective_at=effective_at,
    )
    if preflight_error is not None:
        decision = StopDecision(
            limit=StopLimit.ESTIMATED_COST,
            maximum=limit.max_estimated_cost,
            actual=actual_cost,
            message=preflight_error,
        )
        return _BudgetLimitOutcome(
            decision=decision,
            check=_budget_check_from_stop_decision(
                limit=limit,
                decision=decision,
                cost_summary=cost_summary,
                unpriced_model_steps=unpriced_model_steps,
            ),
        )
    if actual_cost >= limit.max_estimated_cost:
        decision = StopDecision(
            limit=StopLimit.ESTIMATED_COST,
            maximum=limit.max_estimated_cost,
            actual=actual_cost,
            message=(
                "Estimated cost budget reached: "
                f"{actual_cost} >= {limit.max_estimated_cost} {limit.currency}."
            ),
        )
        return _BudgetLimitOutcome(
            decision=decision,
            check=_budget_check_from_stop_decision(
                limit=limit,
                decision=decision,
                cost_summary=cost_summary,
                unpriced_model_steps=unpriced_model_steps,
            ),
        )
    return None


def _budget_check_from_stop_decision(
    *,
    limit: BudgetLimit,
    decision: StopDecision,
    cost_summary: SessionCostSummary,
    unpriced_model_steps: int,
) -> BudgetCheck:
    if decision.limit != StopLimit.ESTIMATED_COST:
        raise ValueError("Budget checks can only be created for estimated-cost decisions.")
    if type(decision.actual) is not Decimal:
        raise TypeError("Estimated-cost decisions must use Decimal actual values.")
    return BudgetCheck(
        scope=limit.scope,
        key=limit.key,
        window=limit.window,
        currency=limit.currency,
        maximum=limit.max_estimated_cost,
        actual=decision.actual,
        action=limit.action,
        model_steps=cost_summary.model_steps,
        unpriced_model_steps=unpriced_model_steps,
        limit_reached=True,
        message=decision.message,
        cost_summary=cost_summary,
    )


def _budget_limit_preflight_error(
    *,
    session: Session,
    limit: BudgetLimit,
    effective_at: datetime,
) -> str | None:
    if limit.allow_unpriced:
        return None
    price = budget_price(
        limit,
        provider_name=session.provider_name,
        model=session.model,
        effective_at=effective_at,
    )
    if price is None:
        return (
            "Estimated cost budget cannot be verified because "
            f"{session.provider_name}/{session.model} has no matching pricing."
        )
    if price.currency.upper() != limit.currency.upper():
        return (
            "Estimated cost budget cannot be verified because "
            f"{session.provider_name}/{session.model} pricing currency {price.currency} "
            f"does not match requested {limit.currency}."
        )
    return None

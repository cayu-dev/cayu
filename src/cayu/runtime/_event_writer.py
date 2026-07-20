from __future__ import annotations

import logging
from collections.abc import Iterable

from cayu.core.events import Event, EventType
from cayu.runtime.budgets import BudgetStore
from cayu.runtime.event_sinks import EventSink
from cayu.runtime.sessions import (
    EventQuery,
    PersistedEventSideEffectClaim,
    PersistedEventSideEffectClaimLost,
    PersistedEventSideEffectDelivery,
    PersistedEventSideEffectStatus,
    SessionStore,
)

_PERSISTED_SIDE_EFFECT_MAX_ATTEMPTS = 3
_PERSISTED_SIDE_EFFECT_RETRY_DELAY_SECONDS = 30.0

logger = logging.getLogger(__name__)


class RuntimeEventWriter:
    """Persist runtime events and fan them out to configured sinks."""

    def __init__(
        self,
        *,
        session_store: SessionStore,
        budget_store: BudgetStore,
        event_sinks: Iterable[EventSink],
    ) -> None:
        self._session_store = session_store
        self._budget_store = budget_store
        self._event_sinks = tuple(event_sinks)

    async def emit(self, event: Event) -> Event:
        await self._session_store.append_event(event.session_id, event)
        claim = await self._session_store.claim_persisted_event_side_effect(
            session_id=event.session_id,
            event_id=event.id,
        )
        if claim is None:
            await self._handle_unclaimed_persisted_side_effect(event)
            return event.model_copy(deep=True)
        delivered_event, _ = await self._deliver_persisted_side_effect_claim(claim)
        return delivered_event

    async def persist(self, event: Event) -> Event:
        """Commit an event to the durable side-effect handoff without delivering it.

        This is reserved for failure evidence that must become durable before
        caller cancellation is redelivered. The store atomically creates the
        pending side-effect record, so normal recovery retains ownership of
        budget and sink delivery.
        """

        await self._session_store.append_event(event.session_id, event)
        return event.model_copy(deep=True)

    async def is_persisted(self, event: Event) -> bool:
        """Return whether this exact event reached the durable event handoff."""

        records = await self._session_store.query_events(
            EventQuery(session_id=event.session_id, event_id=event.id, limit=1)
        )
        return any(record.event.id == event.id for record in records)

    async def emit_many(self, session_id: str, events: list[Event]) -> list[Event]:
        """Persist and fan out a defensive copy of one event batch.

        Batch events use the same durable budget and sink handoff as ``emit``.
        This matters for store-atomic publications that include cost-bearing
        events, such as explicit compaction recovery.
        """
        copied_events = await self.persist_many(session_id, events)
        await self.fan_out_persisted(copied_events)
        return copied_events

    async def persist_many(self, session_id: str, events: list[Event]) -> list[Event]:
        """Persist a defensive event batch without delivering its side effects.

        Callers that must distinguish a failed durable append from failed
        post-commit delivery can persist first and then call
        :meth:`fan_out_persisted`. The store-owned side-effect handoff keeps a
        committed batch recoverable if fan-out is interrupted or fails.
        """
        if type(events) is not list:
            raise TypeError("Runtime events must be a list.")
        copied_events: list[Event] = []
        for event in events:
            if type(event) is not Event:
                raise TypeError("Runtime events must be Event instances.")
            if event.session_id != session_id:
                raise ValueError("Event session_id does not match target session.")
            copied_events.append(event.model_copy(deep=True))

        await self._session_store.append_events(session_id, copied_events)
        return copied_events

    async def fan_out_persisted(self, events: list[Event]) -> list[Event]:
        """Apply budget/sink side effects after a store-owned atomic publication."""

        if type(events) is not list:
            raise TypeError("Runtime events must be a list.")
        copied_events: list[Event] = []
        for event in events:
            if type(event) is not Event:
                raise TypeError("Runtime events must be Event instances.")
            copied = event.model_copy(deep=True)
            claim = await self._session_store.claim_persisted_event_side_effect(
                session_id=copied.session_id,
                event_id=copied.id,
            )
            if claim is None:
                await self._handle_unclaimed_persisted_side_effect(copied)
            else:
                await self._deliver_persisted_side_effect_claim(claim)
            copied_events.append(copied)
        return copied_events

    async def recover_persisted_side_effects(self, *, limit: int = 100) -> list[Event]:
        """Deliver a bounded batch of committed event side effects after a crash."""

        if type(limit) is not int or not 1 <= limit <= 1000:
            raise ValueError("limit must be between 1 and 1000.")
        candidates = await self._session_store.list_persisted_event_side_effect_deliveries(
            statuses={
                PersistedEventSideEffectStatus.PENDING,
                PersistedEventSideEffectStatus.FAILED,
                PersistedEventSideEffectStatus.LEASED,
            },
            claimable_only=True,
            limit=limit,
        )
        recovered: list[Event] = []
        for candidate in candidates:
            claim = await self._session_store.claim_persisted_event_side_effect(
                session_id=candidate.session_id,
                event_id=candidate.event_id,
            )
            if claim is None:
                continue
            try:
                event, delivered = await self._deliver_persisted_side_effect_claim(claim)
            except Exception as exc:
                logger.error(
                    "Persisted event side-effect recovery failed: "
                    "session_id=%s event_id=%s event_type=%s error_type=%s",
                    claim.session_id,
                    claim.event_id,
                    claim.event.type,
                    type(exc).__name__,
                )
                continue
            if delivered:
                recovered.append(event)
        return recovered

    async def _deliver_persisted_side_effect_claim(
        self,
        claim: PersistedEventSideEffectClaim,
    ) -> tuple[Event, bool]:
        event = claim.event.model_copy(deep=True)
        try:
            await self._forward_budget_event_if_required(event)
        except Exception as exc:
            try:
                await self._mark_claim_failed(
                    claim,
                    error=f"{type(exc).__name__}: {exc}",
                )
            except Exception as bookkeeping_exc:
                exc.add_note(
                    "Persisted event side-effect failure bookkeeping also failed: "
                    f"{type(bookkeeping_exc).__name__}: {bookkeeping_exc}"
                )
            raise
        sink_failures = await self._emit_to_sinks(event)
        if sink_failures:
            failure_summary = "; ".join(_exception_summary(failure) for failure in sink_failures)
            try:
                await self._mark_claim_failed(
                    claim,
                    error=failure_summary,
                )
            except Exception as bookkeeping_exc:
                primary_failure = sink_failures[0]
                primary_failure.add_note(
                    "Persisted event side-effect failure bookkeeping also failed: "
                    f"{type(bookkeeping_exc).__name__}: {bookkeeping_exc}"
                )
                if len(sink_failures) > 1:
                    primary_failure.add_note(
                        "Additional persisted event sink failures: "
                        + "; ".join(_exception_summary(failure) for failure in sink_failures[1:])
                    )
                raise primary_failure from bookkeeping_exc
            return event, False
        try:
            await self._session_store.mark_persisted_event_side_effect_delivered(claim)
        except PersistedEventSideEffectClaimLost:
            return event, False
        except Exception as exc:
            logger.error(
                "Persisted event side-effect delivery acknowledgement failed; "
                "leaving the durable claim for recovery: "
                "session_id=%s event_id=%s event_type=%s error_type=%s",
                claim.session_id,
                claim.event_id,
                claim.event.type,
                type(exc).__name__,
                exc_info=True,
            )
            return event, False
        return event, True

    async def _forward_budget_event_if_required(self, event: Event) -> None:
        if event.type == EventType.MODEL_COMPLETED:
            await self._budget_store.append_event(event.model_copy(deep=True))

    async def _handle_unclaimed_persisted_side_effect(self, event: Event) -> None:
        delivery = await self._session_store.get_persisted_event_side_effect_delivery(
            session_id=event.session_id,
            event_id=event.id,
        )
        if delivery is None:
            raise RuntimeError("Persisted event side-effect delivery was not found.")
        if delivery.status in {
            PersistedEventSideEffectStatus.PENDING,
            PersistedEventSideEffectStatus.LEASED,
        }:
            # PostgreSQL can expose the pre-claim PENDING row while another
            # transaction owns its update. This idempotent fallback closes the
            # accounting race; the durable handoff still owns delivery/retry.
            try:
                await self._forward_budget_event_if_required(event)
            except Exception:
                return
            return
        if delivery.status in {
            PersistedEventSideEffectStatus.FAILED,
            PersistedEventSideEffectStatus.DELIVERED,
            PersistedEventSideEffectStatus.DEAD_LETTERED,
        }:
            return
        raise RuntimeError(
            "Persisted event side-effect claim unexpectedly returned no claim "
            f"for {delivery.status.value} delivery."
        )

    async def _mark_claim_failed(
        self,
        claim: PersistedEventSideEffectClaim,
        *,
        error: str,
    ) -> PersistedEventSideEffectDelivery | None:
        try:
            delivery = await self._session_store.mark_persisted_event_side_effect_failed(
                claim,
                error=error,
                max_attempts=_PERSISTED_SIDE_EFFECT_MAX_ATTEMPTS,
                retry_delay_seconds=_PERSISTED_SIDE_EFFECT_RETRY_DELAY_SECONDS,
            )
        except PersistedEventSideEffectClaimLost:
            return None
        if delivery.status is PersistedEventSideEffectStatus.DEAD_LETTERED:
            logger.error(
                "Persisted event side effect dead-lettered: "
                "session_id=%s event_id=%s event_type=%s attempts=%s budget_effect=%s",
                claim.session_id,
                claim.event_id,
                claim.event.type,
                delivery.attempts,
                claim.event.type == EventType.MODEL_COMPLETED,
            )
        return delivery

    async def _emit_to_sinks(self, event: Event) -> list[Exception]:
        failures: list[Exception] = []
        for sink in self._event_sinks:
            try:
                await sink.emit(event.model_copy(deep=True))
            except Exception as exc:
                try:
                    await self._session_store.append_event(
                        event.session_id,
                        Event(
                            type=EventType.RUNTIME_SINK_FAILED,
                            session_id=event.session_id,
                            agent_name=event.agent_name,
                            environment_name=event.environment_name,
                            payload={
                                "sink": type(sink).__name__,
                                "error": str(exc),
                                "error_type": type(exc).__name__,
                                "event_id": event.id,
                                "event_type": str(event.type),
                            },
                        ),
                    )
                except Exception as diagnostic_exc:
                    exc.add_note(
                        "runtime.sink.failed persistence failed: "
                        f"{type(diagnostic_exc).__name__}: {diagnostic_exc}"
                    )
                failures.append(exc)
        return failures


def _exception_summary(exc: Exception) -> str:
    parts = [f"{type(exc).__name__}: {exc}"]
    parts.extend(getattr(exc, "__notes__", ()))
    return "; ".join(parts)

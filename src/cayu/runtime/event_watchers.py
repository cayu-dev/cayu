from __future__ import annotations

import asyncio
import inspect
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator

from cayu._task_wait import await_shielded_task_outcome
from cayu._validation import (
    MAX_DURABLE_JSON_INTEGER,
    require_durable_nonblank,
    require_positive_timedelta_seconds,
)
from cayu._validation import (
    require_durable_clean_nonblank as require_clean_nonblank,
)
from cayu.runtime._diagnostics import exception_diagnostic
from cayu.runtime.sessions import EventOrder, EventQuery, EventRecord, copy_event_query
from cayu.vaults import SecretRedactor

EVENT_WATCHER_QUERY_PAGE_LIMIT = 5000


class EventWatcherDeliveryStatus(StrEnum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    DEAD_LETTERED = "dead_lettered"
    LEASED = "leased"
    LEASE_LOST = "lease_lost"
    PUBLICATION_FAILED = "publication_failed"


class EventWatcherState(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    watcher_name: str
    cursor_sequence: StrictInt = Field(default=0, ge=0, le=MAX_DURABLE_JSON_INTEGER)
    pending_event_id: str | None = None
    pending_event_sequence: StrictInt | None = Field(
        default=None, ge=1, le=MAX_DURABLE_JSON_INTEGER
    )
    pending_attempt: StrictInt = Field(default=0, ge=0, le=MAX_DURABLE_JSON_INTEGER)
    pending_claim_id: str | None = None
    delivery_status: EventWatcherDeliveryStatus | None = None
    lease_expires_at: datetime | None = None
    last_error: str | None = None
    dead_lettered_count: StrictInt = Field(default=0, ge=0, le=MAX_DURABLE_JSON_INTEGER)
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @field_validator("watcher_name", "pending_event_id", "pending_claim_id", "last_error")
    @classmethod
    def validate_optional_clean_strings(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        if info.field_name == "last_error":
            return require_durable_nonblank(value, "last_error")
        return require_clean_nonblank(value, info.field_name)

    @field_validator("lease_expires_at", "updated_at")
    @classmethod
    def normalize_timestamp(cls, value: datetime | None, info) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError(f"{info.field_name} must be timezone-aware.")
        return value.astimezone(UTC)


class EventWatcherClaim(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    watcher_name: str
    event_id: str
    event_sequence: StrictInt = Field(ge=1, le=MAX_DURABLE_JSON_INTEGER)
    attempt: StrictInt = Field(ge=1, le=MAX_DURABLE_JSON_INTEGER)
    claim_id: str = Field(default_factory=lambda: str(uuid4()))
    lease_expires_at: datetime

    @field_validator("watcher_name", "event_id", "claim_id")
    @classmethod
    def validate_clean_strings(cls, value: str, info) -> str:
        return require_clean_nonblank(value, info.field_name)

    @field_validator("lease_expires_at")
    @classmethod
    def normalize_lease_expires_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("lease_expires_at must be timezone-aware.")
        return value.astimezone(UTC)


class EventWatcherDelivery(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    watcher_name: str
    event_id: str
    event_sequence: StrictInt = Field(ge=1, le=MAX_DURABLE_JSON_INTEGER)
    status: EventWatcherDeliveryStatus
    attempt: StrictInt = Field(ge=1, le=MAX_DURABLE_JSON_INTEGER)
    cursor_sequence: StrictInt = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)
    error: str | None = None

    @field_validator("watcher_name", "event_id", "error")
    @classmethod
    def validate_optional_clean_strings(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        if info.field_name == "error":
            return require_durable_nonblank(value, "error")
        return require_clean_nonblank(value, info.field_name)


class EventWatcherDeadLetter(BaseModel):
    """A durable record of one event that exhausted its delivery attempts.

    Persisting these (rather than only bumping a counter and overwriting a single
    ``last_error`` on the watcher state) keeps every dead-lettered event
    individually inspectable and replayable: the durable event log remains the
    source of truth, and ``event_sequence`` + ``event_id`` point back at the exact
    event a handler can be re-dispatched against.
    """

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    watcher_name: str
    event_id: str
    event_sequence: StrictInt = Field(ge=1, le=MAX_DURABLE_JSON_INTEGER)
    attempts: StrictInt = Field(ge=1, le=MAX_DURABLE_JSON_INTEGER)
    error: str
    dead_lettered_at: datetime
    resolved_at: datetime | None = None

    @field_validator("watcher_name", "event_id")
    @classmethod
    def validate_clean_strings(cls, value: str, info) -> str:
        return require_clean_nonblank(value, info.field_name)

    @field_validator("error")
    @classmethod
    def validate_error(cls, value: str) -> str:
        return require_durable_nonblank(value, "error")

    @field_validator("dead_lettered_at", "resolved_at")
    @classmethod
    def normalize_timestamp(cls, value: datetime | None, info) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError(f"{info.field_name} must be timezone-aware.")
        return value.astimezone(UTC)


class EventWatcherContext(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    watcher_name: str
    record: EventRecord
    attempt: StrictInt = Field(ge=1, le=MAX_DURABLE_JSON_INTEGER)

    @field_validator("watcher_name")
    @classmethod
    def validate_watcher_name(cls, value: str) -> str:
        return require_clean_nonblank(value, "watcher_name")

    @field_validator("record")
    @classmethod
    def copy_record(cls, value: EventRecord) -> EventRecord:
        return copy_event_watcher_record(value)


EventWatcherHandler = Callable[[EventWatcherContext], Awaitable[None] | None]


@dataclass(frozen=True)
class EventWatcher:
    """Trusted app-code handler for durable runtime events."""

    name: str
    query: EventQuery
    handler: EventWatcherHandler
    max_attempts: int = 3
    batch_size: int = 100
    lease_seconds: float = 300.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", require_clean_nonblank(self.name, "name"))
        if not callable(self.handler):
            raise TypeError("handler must be callable.")
        query = copy_event_query(self.query)
        if query.after_sequence is not None:
            raise ValueError("EventWatcher query must not set after_sequence.")
        object.__setattr__(self, "query", query)
        if type(self.max_attempts) is not int or self.max_attempts < 1:
            raise ValueError("max_attempts must be an integer greater than or equal to 1.")
        if type(self.batch_size) is not int or self.batch_size < 1:
            raise ValueError("batch_size must be an integer greater than or equal to 1.")
        object.__setattr__(
            self,
            "lease_seconds",
            require_positive_timedelta_seconds(self.lease_seconds, "lease_seconds"),
        )


class EventWatcherRunResult(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    watcher_name: str
    deliveries: list[EventWatcherDelivery] = Field(default_factory=list)
    blocked_by_active_lease: bool = False
    error: str | None = None

    @field_validator("watcher_name")
    @classmethod
    def validate_watcher_name(cls, value: str) -> str:
        return require_clean_nonblank(value, "watcher_name")

    @field_validator("deliveries")
    @classmethod
    def copy_deliveries(
        cls,
        value: list[EventWatcherDelivery],
    ) -> list[EventWatcherDelivery]:
        if type(value) is not list:
            raise TypeError("deliveries must be a list.")
        return [
            delivery.model_copy(deep=True)
            if type(delivery) is EventWatcherDelivery
            else EventWatcherDelivery.model_validate(delivery)
            for delivery in value
        ]


class EventWatcherStore(ABC):
    """Durable delivery state for event watchers."""

    @abstractmethod
    async def load_state(self, watcher_name: str) -> EventWatcherState:
        """Load watcher cursor and pending attempt state."""

    @abstractmethod
    async def claim_event(
        self,
        *,
        watcher_name: str,
        record: EventRecord,
        lease_seconds: float,
        max_attempts: int = 3,
    ) -> EventWatcherClaim | EventWatcherDelivery | None:
        """Claim one event for at-least-once processing.

        Returns ``None`` when another live claim owns the watcher or the cursor
        already passed the event. Returns a dead-letter delivery when reclaim
        exhausts the maximum; no handler may run for that result.
        """

    async def renew_claim(
        self, claim: EventWatcherClaim, *, lease_seconds: float
    ) -> EventWatcherClaim:
        """Renew a live exact claim under authoritative store time.

        Custom stores must implement this before Runtime dispatches a handler.
        Expired or replaced claims raise EventWatcherLeaseLost.
        """
        raise NotImplementedError("Event watcher lease renewal is required by this store contract.")

    @abstractmethod
    async def mark_success(self, claim: EventWatcherClaim) -> EventWatcherDelivery:
        """Mark a claimed event handled and advance the watcher cursor."""

    @abstractmethod
    async def mark_failure(
        self,
        claim: EventWatcherClaim,
        *,
        error: str,
        max_attempts: int,
    ) -> EventWatcherDelivery:
        """Mark a claimed event failed or dead-lettered.

        When the final attempt is exhausted the store also persists a durable
        :class:`EventWatcherDeadLetter` record for the event (see
        :meth:`list_dead_letters`) so it can be inspected and replayed later.
        """

    async def list_dead_letters(
        self,
        watcher_name: str,
        *,
        include_resolved: bool = False,
        limit: int = 100,
    ) -> list[EventWatcherDeadLetter]:
        """Return persisted dead-letter records for a watcher, oldest first.

        Unresolved records are returned by default; pass ``include_resolved`` to
        also surface ones already marked handled via :meth:`resolve_dead_letter`.
        """
        raise NotImplementedError("Event watcher dead letters are not supported by this store.")

    async def resolve_dead_letter(
        self,
        watcher_name: str,
        event_sequence: int,
    ) -> EventWatcherDeadLetter:
        """Mark a dead-letter record handled (e.g. after a successful replay).

        Raises :class:`ValueError` when no such record exists for the watcher.
        """
        raise NotImplementedError("Event watcher dead letters are not supported by this store.")


class EventWatcherLeaseLost(ValueError):
    """The store can no longer authorize this watcher delivery claim."""


class EventWatcherSettlement(BaseModel):
    """Immutable receipt for one exact handler acknowledgement."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    claim: EventWatcherClaim
    operation: Literal["success", "failure"]
    error: str | None = None
    max_attempts: StrictInt | None = None
    delivery: EventWatcherDelivery


def replay_event_watcher_settlement(
    receipt: EventWatcherSettlement,
    claim: EventWatcherClaim,
    *,
    error: str | None,
    max_attempts: int | None,
) -> EventWatcherDelivery:
    if (
        receipt.claim.model_dump(exclude={"lease_expires_at"})
        != claim.model_dump(exclude={"lease_expires_at"})
        or receipt.operation != ("success" if error is None else "failure")
        or receipt.error != error
        or receipt.max_attempts != max_attempts
    ):
        raise EventWatcherLeaseLost("Watcher acknowledgement conflicts with its durable receipt.")
    return receipt.delivery.model_copy(deep=True)


@dataclass(frozen=True)
class EventWatcherTransition:
    state: EventWatcherState
    outcome: EventWatcherClaim | EventWatcherDelivery | None
    dead_letter: EventWatcherDeadLetter | None = None


def claim_event_watcher_transition(
    state: EventWatcherState,
    record: EventRecord,
    *,
    now: datetime,
    lease_seconds: float,
    max_attempts: int,
) -> EventWatcherTransition:
    """Apply admission/exhaustion under the adapter's transaction and clock."""

    max_attempts = _validate_max_attempts(max_attempts)
    lease_seconds = require_positive_timedelta_seconds(
        lease_seconds, "lease_seconds", relative_to=now
    )
    if state.cursor_sequence >= record.sequence:
        return EventWatcherTransition(state, None)
    if (
        state.delivery_status is EventWatcherDeliveryStatus.LEASED
        and state.lease_expires_at is not None
        and state.lease_expires_at > now
    ):
        return EventWatcherTransition(state, None)
    pending = state.pending_event_id is not None
    if pending and (
        state.pending_event_id != record.event.id or state.pending_event_sequence != record.sequence
    ):
        raise EventWatcherLeaseLost("An earlier pending watcher event must settle first.")
    if pending and state.pending_attempt >= max_attempts:
        exhausted = EventWatcherClaim(
            watcher_name=state.watcher_name,
            event_id=record.event.id,
            event_sequence=record.sequence,
            attempt=state.pending_attempt,
            claim_id=state.pending_claim_id or str(uuid4()),
            lease_expires_at=state.lease_expires_at or now,
        )
        error = "Event watcher attempt limit exhausted before durable handler acknowledgement."
        updated = _terminal_watcher_state(
            state,
            exhausted,
            status=EventWatcherDeliveryStatus.DEAD_LETTERED,
            error=error,
            now=now,
        )
        return EventWatcherTransition(
            updated,
            _delivery_from_claim(
                exhausted,
                status=EventWatcherDeliveryStatus.DEAD_LETTERED,
                cursor_sequence=updated.cursor_sequence,
                error=error,
            ),
            _dead_letter_from_claim(exhausted, error=error, now=now),
        )
    claim = EventWatcherClaim(
        watcher_name=state.watcher_name,
        event_id=record.event.id,
        event_sequence=record.sequence,
        attempt=state.pending_attempt + 1 if pending else 1,
        lease_expires_at=now + timedelta(seconds=lease_seconds),
    )
    return EventWatcherTransition(
        state.model_copy(
            update={
                "pending_event_id": claim.event_id,
                "pending_event_sequence": claim.event_sequence,
                "pending_attempt": claim.attempt,
                "pending_claim_id": claim.claim_id,
                "delivery_status": EventWatcherDeliveryStatus.LEASED,
                "lease_expires_at": claim.lease_expires_at,
                "last_error": None,
                "updated_at": now,
            },
            deep=True,
        ),
        claim,
    )


def renew_event_watcher_transition(
    state: EventWatcherState, claim: EventWatcherClaim, *, now: datetime, lease_seconds: float
) -> EventWatcherTransition:
    _matching_claim_state(state, claim, now=now)
    lease_seconds = require_positive_timedelta_seconds(
        lease_seconds, "lease_seconds", relative_to=now
    )
    assert state.lease_expires_at is not None
    expires = max(state.lease_expires_at, now + timedelta(seconds=lease_seconds))
    renewed = claim.model_copy(update={"lease_expires_at": expires}, deep=True)
    return EventWatcherTransition(
        state.model_copy(update={"lease_expires_at": expires, "updated_at": now}, deep=True),
        renewed,
    )


def settle_event_watcher_transition(
    state: EventWatcherState,
    claim: EventWatcherClaim,
    *,
    now: datetime,
    error: str | None,
    max_attempts: int | None,
) -> EventWatcherTransition:
    _matching_claim_state(state, claim, now=now)
    dead_letter = None
    if error is None:
        status = EventWatcherDeliveryStatus.SUCCEEDED
        updated = _terminal_watcher_state(state, claim, status=status, error=None, now=now)
    elif claim.attempt >= _validate_max_attempts(max_attempts):
        status = EventWatcherDeliveryStatus.DEAD_LETTERED
        updated = _terminal_watcher_state(state, claim, status=status, error=error, now=now)
        dead_letter = _dead_letter_from_claim(claim, error=error, now=now)
    else:
        status = EventWatcherDeliveryStatus.FAILED
        updated = state.model_copy(
            update={
                "delivery_status": status,
                "pending_claim_id": None,
                "lease_expires_at": None,
                "last_error": error,
                "updated_at": now,
            },
            deep=True,
        )
    return EventWatcherTransition(
        updated,
        _delivery_from_claim(
            claim, status=status, cursor_sequence=updated.cursor_sequence, error=error
        ),
        dead_letter,
    )


def _terminal_watcher_state(
    state: EventWatcherState,
    claim: EventWatcherClaim,
    *,
    status: EventWatcherDeliveryStatus,
    error: str | None,
    now: datetime,
) -> EventWatcherState:
    return state.model_copy(
        update={
            "cursor_sequence": claim.event_sequence,
            "pending_event_id": None,
            "pending_event_sequence": None,
            "pending_attempt": 0,
            "pending_claim_id": None,
            "delivery_status": status,
            "lease_expires_at": None,
            "last_error": error,
            "dead_lettered_count": state.dead_lettered_count
            + int(status is EventWatcherDeliveryStatus.DEAD_LETTERED),
            "updated_at": now,
        },
        deep=True,
    )


class InMemoryEventWatcherStore(EventWatcherStore):
    """In-process watcher state for tests, examples, and single-process apps."""

    def __init__(self, *, clock: Callable[[], datetime] | None = None) -> None:
        self._lock = asyncio.Lock()
        self._states: dict[str, EventWatcherState] = {}
        self._dead_letters: dict[str, dict[int, EventWatcherDeadLetter]] = {}
        self._clock = _clock_or_utc_now(clock)
        self._settlements: dict[tuple[str, str], EventWatcherSettlement] = {}

    async def load_state(self, watcher_name: str) -> EventWatcherState:
        watcher_name = require_clean_nonblank(watcher_name, "watcher_name")
        async with self._lock:
            state = self._states.get(watcher_name)
            if state is None:
                return EventWatcherState(watcher_name=watcher_name)
            return state.model_copy(deep=True)

    async def claim_event(
        self,
        *,
        watcher_name: str,
        record: EventRecord,
        lease_seconds: float,
        max_attempts: int = 3,
    ) -> EventWatcherClaim | EventWatcherDelivery | None:
        watcher_name = require_clean_nonblank(watcher_name, "watcher_name")
        record = copy_event_watcher_record(record)
        max_attempts = _validate_max_attempts(max_attempts)
        async with self._lock:
            now = self._clock()
            state = self._states.get(watcher_name)
            if state is None:
                state = EventWatcherState(watcher_name=watcher_name, updated_at=now)
            now = max(now, state.updated_at)
            transition = claim_event_watcher_transition(
                state,
                record,
                now=now,
                lease_seconds=lease_seconds,
                max_attempts=max_attempts,
            )
            self._states[watcher_name] = transition.state
            if transition.dead_letter is not None:
                self._dead_letters.setdefault(watcher_name, {})[record.sequence] = (
                    transition.dead_letter
                )
            return transition.outcome

    async def renew_claim(
        self, claim: EventWatcherClaim, *, lease_seconds: float
    ) -> EventWatcherClaim:
        claim = copy_event_watcher_claim(claim)
        async with self._lock:
            state = self._states.get(claim.watcher_name)
            if state is None:
                raise EventWatcherLeaseLost("Watcher claim not found.")
            transition = renew_event_watcher_transition(
                state, claim, now=max(self._clock(), state.updated_at), lease_seconds=lease_seconds
            )
            self._states[claim.watcher_name] = transition.state
            assert isinstance(transition.outcome, EventWatcherClaim)
            return transition.outcome

    async def mark_success(self, claim: EventWatcherClaim) -> EventWatcherDelivery:
        return await self._settle(claim, error=None, max_attempts=None)

    async def mark_failure(
        self,
        claim: EventWatcherClaim,
        *,
        error: str,
        max_attempts: int,
    ) -> EventWatcherDelivery:
        return await self._settle(
            claim, error=_clean_error(error), max_attempts=_validate_max_attempts(max_attempts)
        )

    async def _settle(
        self,
        claim: EventWatcherClaim,
        *,
        error: str | None,
        max_attempts: int | None,
    ) -> EventWatcherDelivery:
        claim = copy_event_watcher_claim(claim)
        key = (claim.watcher_name, claim.claim_id)
        async with self._lock:
            receipt = self._settlements.get(key)
            if receipt is not None:
                return replay_event_watcher_settlement(
                    receipt, claim, error=error, max_attempts=max_attempts
                )
            state = self._states.get(claim.watcher_name)
            if state is None:
                raise EventWatcherLeaseLost("Watcher claim not found.")
            transition = settle_event_watcher_transition(
                state,
                claim,
                now=max(self._clock(), state.updated_at),
                error=error,
                max_attempts=max_attempts,
            )
            assert isinstance(transition.outcome, EventWatcherDelivery)
            receipt = EventWatcherSettlement(
                claim=claim,
                operation="success" if error is None else "failure",
                error=error,
                max_attempts=max_attempts,
                delivery=transition.outcome,
            )
            self._states[claim.watcher_name] = transition.state
            if transition.dead_letter is not None:
                self._dead_letters.setdefault(claim.watcher_name, {})[claim.event_sequence] = (
                    transition.dead_letter
                )
            self._settlements[key] = receipt
            return transition.outcome.model_copy(deep=True)

    async def list_dead_letters(
        self,
        watcher_name: str,
        *,
        include_resolved: bool = False,
        limit: int = 100,
    ) -> list[EventWatcherDeadLetter]:
        watcher_name = require_clean_nonblank(watcher_name, "watcher_name")
        limit = _validate_dead_letter_limit(limit)
        async with self._lock:
            records = self._dead_letters.get(watcher_name, {})
            selected = [
                record.model_copy(deep=True)
                for _, record in sorted(records.items())
                if include_resolved or record.resolved_at is None
            ]
            return selected[:limit]

    async def resolve_dead_letter(
        self,
        watcher_name: str,
        event_sequence: int,
    ) -> EventWatcherDeadLetter:
        watcher_name = require_clean_nonblank(watcher_name, "watcher_name")
        event_sequence = _validate_event_sequence(event_sequence)
        now = self._clock()
        async with self._lock:
            record = self._dead_letters.get(watcher_name, {}).get(event_sequence)
            if record is None:
                raise ValueError(
                    f"No dead-letter record for watcher {watcher_name!r} "
                    f"at sequence {event_sequence}."
                )
            resolved = record.model_copy(
                update={"resolved_at": record.resolved_at or now},
                deep=True,
            )
            self._dead_letters[watcher_name][event_sequence] = resolved
            return resolved.model_copy(deep=True)


async def run_event_watcher_handler(
    watcher: EventWatcher,
    context: EventWatcherContext,
) -> None:
    if inspect.iscoroutinefunction(watcher.handler) or inspect.iscoroutinefunction(
        type(watcher.handler).__call__
    ):
        result = watcher.handler(context)
    else:
        work = asyncio.create_task(asyncio.to_thread(watcher.handler, context))
        outcome = await await_shielded_task_outcome(work)
        if outcome.cancellation is not None:
            raise outcome.cancellation
        if outcome.error is not None:
            raise outcome.error
        result = outcome.result
    if inspect.isawaitable(result):
        await result


def event_query_after_cursor(
    query: EventQuery,
    cursor_sequence: int,
    *,
    limit: int = 1,
) -> EventQuery:
    query = copy_event_query(query)
    if type(limit) is not int or limit < 1:
        raise ValueError("limit must be an integer greater than or equal to 1.")
    return copy_event_query(
        query,
        update={
            "after_sequence": cursor_sequence,
            "limit": min(limit, EVENT_WATCHER_QUERY_PAGE_LIMIT),
            "order_by": EventOrder.SEQUENCE_ASC,
        },
    )


def copy_event_watcher_record(record: EventRecord) -> EventRecord:
    if type(record) is not EventRecord:
        raise TypeError("record must be an EventRecord.")
    return EventRecord(sequence=record.sequence, event=record.event)


def copy_event_watcher_state(state: EventWatcherState) -> EventWatcherState:
    if type(state) is not EventWatcherState:
        raise TypeError("state must be an EventWatcherState.")
    return EventWatcherState.model_validate(state.model_dump(mode="python"))


def copy_event_watcher_claim(claim: EventWatcherClaim) -> EventWatcherClaim:
    if type(claim) is not EventWatcherClaim:
        raise TypeError("claim must be an EventWatcherClaim.")
    return EventWatcherClaim.model_validate(claim.model_dump(mode="python"))


def copy_event_watcher_delivery(delivery: EventWatcherDelivery) -> EventWatcherDelivery:
    if type(delivery) is not EventWatcherDelivery:
        raise TypeError("delivery must be an EventWatcherDelivery.")
    return EventWatcherDelivery.model_validate(delivery.model_dump(mode="python"))


def copy_event_watcher_dead_letter(
    dead_letter: EventWatcherDeadLetter,
) -> EventWatcherDeadLetter:
    if type(dead_letter) is not EventWatcherDeadLetter:
        raise TypeError("dead_letter must be an EventWatcherDeadLetter.")
    return EventWatcherDeadLetter.model_validate(dead_letter.model_dump(mode="python"))


def event_watcher_error_payload(
    error: BaseException,
    *,
    redactor: SecretRedactor | None = None,
) -> str:
    return exception_diagnostic(
        error,
        empty_message="event watcher failed",
        nonportable_message="Event watcher failed with a non-portable diagnostic.",
        redactor=redactor,
    ).message


def _matching_claim_state(
    state: EventWatcherState | None,
    claim: EventWatcherClaim,
    *,
    now: datetime,
) -> EventWatcherState:
    if state is None:
        raise EventWatcherLeaseLost(f"Watcher claim not found: {claim.watcher_name}")
    if state.pending_claim_id != claim.claim_id:
        raise EventWatcherLeaseLost("Watcher claim is no longer active.")
    if state.pending_event_id != claim.event_id:
        raise EventWatcherLeaseLost("Watcher claim event_id does not match active claim.")
    if state.pending_event_sequence != claim.event_sequence:
        raise EventWatcherLeaseLost("Watcher claim sequence does not match active claim.")
    if state.pending_attempt != claim.attempt:
        raise EventWatcherLeaseLost("Watcher claim attempt does not match active claim.")
    if (
        state.delivery_status is not EventWatcherDeliveryStatus.LEASED
        or state.lease_expires_at is None
        or state.lease_expires_at <= now
    ):
        raise EventWatcherLeaseLost("Watcher claim lease has expired.")
    return state


def _delivery_from_claim(
    claim: EventWatcherClaim,
    *,
    status: EventWatcherDeliveryStatus,
    cursor_sequence: int,
    error: str | None = None,
) -> EventWatcherDelivery:
    return EventWatcherDelivery(
        watcher_name=claim.watcher_name,
        event_id=claim.event_id,
        event_sequence=claim.event_sequence,
        status=status,
        attempt=claim.attempt,
        cursor_sequence=cursor_sequence,
        error=error,
    )


def _dead_letter_from_claim(
    claim: EventWatcherClaim,
    *,
    error: str,
    now: datetime,
) -> EventWatcherDeadLetter:
    return EventWatcherDeadLetter(
        watcher_name=claim.watcher_name,
        event_id=claim.event_id,
        event_sequence=claim.event_sequence,
        attempts=claim.attempt,
        error=error,
        dead_lettered_at=now,
    )


def _validate_dead_letter_limit(value: int) -> int:
    if type(value) is not int or value < 1:
        raise ValueError("limit must be an integer greater than or equal to 1.")
    return value


def _validate_event_sequence(value: int) -> int:
    if type(value) is not int or value < 1:
        raise ValueError("event_sequence must be an integer greater than or equal to 1.")
    return value


def _clock_or_utc_now(clock: Callable[[], datetime] | None) -> Callable[[], datetime]:
    if clock is None:
        return lambda: datetime.now(UTC)
    if not callable(clock):
        raise TypeError("clock must be callable.")

    def wrapped() -> datetime:
        value = clock()
        if not isinstance(value, datetime):
            raise TypeError("clock must return a datetime.")
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("clock must return a timezone-aware datetime.")
        return value.astimezone(UTC)

    return wrapped


def _validate_max_attempts(value: int | None) -> int:
    if type(value) is not int or value < 1:
        raise ValueError("max_attempts must be an integer greater than or equal to 1.")
    return value


def _clean_error(value: str) -> str:
    value = require_durable_nonblank(value, "error")
    encoded = value.encode("utf-8")
    if len(encoded) <= 4096:
        return value
    return (
        encoded[:4096].decode("utf-8", errors="ignore").strip()
        or "Event watcher diagnostic truncated."
    )

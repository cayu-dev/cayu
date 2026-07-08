from __future__ import annotations

import asyncio
import inspect
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator

from cayu._validation import copy_json_value, require_clean_nonblank
from cayu.runtime.sessions import EventQuery, EventRecord, copy_event_query

EVENT_WATCHER_QUERY_PAGE_LIMIT = 5000


class EventWatcherDeliveryStatus(StrEnum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    DEAD_LETTERED = "dead_lettered"
    LEASED = "leased"


class EventWatcherState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    watcher_name: str
    cursor_sequence: StrictInt = Field(default=0, ge=0)
    pending_event_id: str | None = None
    pending_event_sequence: StrictInt | None = Field(default=None, ge=1)
    pending_attempt: StrictInt = Field(default=0, ge=0)
    pending_claim_id: str | None = None
    delivery_status: EventWatcherDeliveryStatus | None = None
    lease_expires_at: datetime | None = None
    last_error: str | None = None
    dead_lettered_count: StrictInt = Field(default=0, ge=0)
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @field_validator("watcher_name", "pending_event_id", "pending_claim_id", "last_error")
    @classmethod
    def validate_optional_clean_strings(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        if info.field_name == "last_error":
            if not value.strip():
                raise ValueError("last_error cannot be blank.")
            return value
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
    model_config = ConfigDict(extra="forbid")

    watcher_name: str
    event_id: str
    event_sequence: StrictInt = Field(ge=1)
    attempt: StrictInt = Field(ge=1)
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
    model_config = ConfigDict(extra="forbid")

    watcher_name: str
    event_id: str
    event_sequence: StrictInt = Field(ge=1)
    status: EventWatcherDeliveryStatus
    attempt: StrictInt = Field(ge=1)
    cursor_sequence: StrictInt = Field(ge=0)
    error: str | None = None

    @field_validator("watcher_name", "event_id", "error")
    @classmethod
    def validate_optional_clean_strings(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        if info.field_name == "error":
            if not value.strip():
                raise ValueError("error cannot be blank.")
            return value
        return require_clean_nonblank(value, info.field_name)


class EventWatcherDeadLetter(BaseModel):
    """A durable record of one event that exhausted its delivery attempts.

    Persisting these (rather than only bumping a counter and overwriting a single
    ``last_error`` on the watcher state) keeps every dead-lettered event
    individually inspectable and replayable: the durable event log remains the
    source of truth, and ``event_sequence`` + ``event_id`` point back at the exact
    event a handler can be re-dispatched against.
    """

    model_config = ConfigDict(extra="forbid")

    watcher_name: str
    event_id: str
    event_sequence: StrictInt = Field(ge=1)
    attempts: StrictInt = Field(ge=1)
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
        if not value.strip():
            raise ValueError("error cannot be blank.")
        return value

    @field_validator("dead_lettered_at", "resolved_at")
    @classmethod
    def normalize_timestamp(cls, value: datetime | None, info) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError(f"{info.field_name} must be timezone-aware.")
        return value.astimezone(UTC)


class EventWatcherContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    watcher_name: str
    record: EventRecord
    attempt: StrictInt = Field(ge=1)

    @field_validator("watcher_name")
    @classmethod
    def validate_watcher_name(cls, value: str) -> str:
        return require_clean_nonblank(value, "watcher_name")

    @field_validator("record")
    @classmethod
    def copy_record(cls, value: EventRecord) -> EventRecord:
        if type(value) is not EventRecord:
            raise TypeError("record must be an EventRecord.")
        return EventRecord(sequence=value.sequence, event=value.event)


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
        if type(self.lease_seconds) not in {int, float} or self.lease_seconds <= 0:
            raise ValueError("lease_seconds must be greater than 0.")
        object.__setattr__(self, "lease_seconds", float(self.lease_seconds))


class EventWatcherRunResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    watcher_name: str
    deliveries: list[EventWatcherDelivery] = Field(default_factory=list)
    blocked_by_active_lease: bool = False

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
    ) -> EventWatcherClaim | None:
        """Claim one event for at-least-once processing.

        Returns ``None`` when another live claim still owns the watcher.
        """

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


class InMemoryEventWatcherStore(EventWatcherStore):
    """In-process watcher state for tests, examples, and single-process apps."""

    def __init__(self, *, clock: Callable[[], datetime] | None = None) -> None:
        self._lock = asyncio.Lock()
        self._states: dict[str, EventWatcherState] = {}
        self._dead_letters: dict[str, dict[int, EventWatcherDeadLetter]] = {}
        self._clock = _clock_or_utc_now(clock)

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
    ) -> EventWatcherClaim | None:
        watcher_name = require_clean_nonblank(watcher_name, "watcher_name")
        record = _copy_event_record(record)
        lease_seconds = _validate_lease_seconds(lease_seconds)
        now = self._clock()
        async with self._lock:
            state = self._states.get(watcher_name)
            if state is None:
                state = EventWatcherState(watcher_name=watcher_name, updated_at=now)
            if state.cursor_sequence >= record.sequence:
                self._states[watcher_name] = state
                return None
            if (
                state.delivery_status is EventWatcherDeliveryStatus.LEASED
                and state.lease_expires_at is not None
                and state.lease_expires_at > now
            ):
                return None

            attempt = (
                state.pending_attempt + 1
                if state.pending_event_id == record.event.id
                and state.pending_event_sequence == record.sequence
                else 1
            )
            claim = EventWatcherClaim(
                watcher_name=watcher_name,
                event_id=record.event.id,
                event_sequence=record.sequence,
                attempt=attempt,
                lease_expires_at=now + timedelta(seconds=lease_seconds),
            )
            self._states[watcher_name] = state.model_copy(
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
            )
            return claim.model_copy(deep=True)

    async def mark_success(self, claim: EventWatcherClaim) -> EventWatcherDelivery:
        claim = _copy_claim(claim)
        now = self._clock()
        async with self._lock:
            state = _matching_claim_state(self._states.get(claim.watcher_name), claim)
            updated = state.model_copy(
                update={
                    "cursor_sequence": claim.event_sequence,
                    "pending_event_id": None,
                    "pending_event_sequence": None,
                    "pending_attempt": 0,
                    "pending_claim_id": None,
                    "delivery_status": EventWatcherDeliveryStatus.SUCCEEDED,
                    "lease_expires_at": None,
                    "last_error": None,
                    "updated_at": now,
                },
                deep=True,
            )
            self._states[claim.watcher_name] = updated
            return _delivery_from_claim(
                claim,
                status=EventWatcherDeliveryStatus.SUCCEEDED,
                cursor_sequence=updated.cursor_sequence,
            )

    async def mark_failure(
        self,
        claim: EventWatcherClaim,
        *,
        error: str,
        max_attempts: int,
    ) -> EventWatcherDelivery:
        claim = _copy_claim(claim)
        error = _clean_error(error)
        max_attempts = _validate_max_attempts(max_attempts)
        now = self._clock()
        async with self._lock:
            state = _matching_claim_state(self._states.get(claim.watcher_name), claim)
            if claim.attempt >= max_attempts:
                updated = state.model_copy(
                    update={
                        "cursor_sequence": claim.event_sequence,
                        "pending_event_id": None,
                        "pending_event_sequence": None,
                        "pending_attempt": 0,
                        "pending_claim_id": None,
                        "delivery_status": EventWatcherDeliveryStatus.DEAD_LETTERED,
                        "lease_expires_at": None,
                        "last_error": error,
                        "dead_lettered_count": state.dead_lettered_count + 1,
                        "updated_at": now,
                    },
                    deep=True,
                )
                status = EventWatcherDeliveryStatus.DEAD_LETTERED
                dead_letter = _dead_letter_from_claim(claim, error=error, now=now)
                self._dead_letters.setdefault(claim.watcher_name, {})[
                    dead_letter.event_sequence
                ] = dead_letter
            else:
                updated = state.model_copy(
                    update={
                        "delivery_status": EventWatcherDeliveryStatus.FAILED,
                        "pending_claim_id": None,
                        "lease_expires_at": None,
                        "last_error": error,
                        "updated_at": now,
                    },
                    deep=True,
                )
                status = EventWatcherDeliveryStatus.FAILED
            self._states[claim.watcher_name] = updated
            return _delivery_from_claim(
                claim,
                status=status,
                cursor_sequence=updated.cursor_sequence,
                error=error,
            )

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
    result = watcher.handler(context)
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
    return EventQuery(
        session_id=query.session_id,
        session_ids=query.session_ids,
        causal_budget_id=query.causal_budget_id,
        event_type=query.event_type,
        event_types=query.event_types,
        agent_name=query.agent_name,
        environment_name=query.environment_name,
        workflow_name=query.workflow_name,
        tool_name=query.tool_name,
        since=query.since,
        until=query.until,
        after_sequence=cursor_sequence,
        limit=min(limit, EVENT_WATCHER_QUERY_PAGE_LIMIT),
    )


def copy_event_watcher_state(state: EventWatcherState) -> EventWatcherState:
    if type(state) is not EventWatcherState:
        raise TypeError("state must be an EventWatcherState.")
    return state.model_copy(deep=True)


def copy_event_watcher_claim(claim: EventWatcherClaim) -> EventWatcherClaim:
    return _copy_claim(claim)


def copy_event_watcher_delivery(delivery: EventWatcherDelivery) -> EventWatcherDelivery:
    if type(delivery) is not EventWatcherDelivery:
        raise TypeError("delivery must be an EventWatcherDelivery.")
    return delivery.model_copy(deep=True)


def copy_event_watcher_dead_letter(
    dead_letter: EventWatcherDeadLetter,
) -> EventWatcherDeadLetter:
    if type(dead_letter) is not EventWatcherDeadLetter:
        raise TypeError("dead_letter must be an EventWatcherDeadLetter.")
    return dead_letter.model_copy(deep=True)


def event_watcher_error_payload(error: BaseException) -> str:
    message = str(error).strip()
    if message:
        return message
    return type(error).__name__


def _copy_event_record(record: EventRecord) -> EventRecord:
    if type(record) is not EventRecord:
        raise TypeError("record must be an EventRecord.")
    return EventRecord(sequence=record.sequence, event=record.event)


def _copy_claim(claim: EventWatcherClaim) -> EventWatcherClaim:
    if type(claim) is not EventWatcherClaim:
        raise TypeError("claim must be an EventWatcherClaim.")
    return claim.model_copy(deep=True)


def _matching_claim_state(
    state: EventWatcherState | None,
    claim: EventWatcherClaim,
) -> EventWatcherState:
    if state is None:
        raise ValueError(f"Watcher claim not found: {claim.watcher_name}")
    if state.pending_claim_id != claim.claim_id:
        raise ValueError("Watcher claim is no longer active.")
    if state.pending_event_id != claim.event_id:
        raise ValueError("Watcher claim event_id does not match active claim.")
    if state.pending_event_sequence != claim.event_sequence:
        raise ValueError("Watcher claim sequence does not match active claim.")
    if state.pending_attempt != claim.attempt:
        raise ValueError("Watcher claim attempt does not match active claim.")
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


def _validate_lease_seconds(value: float) -> float:
    if type(value) not in {int, float} or value <= 0:
        raise ValueError("lease_seconds must be greater than 0.")
    return float(value)


def _validate_max_attempts(value: int) -> int:
    if type(value) is not int or value < 1:
        raise ValueError("max_attempts must be an integer greater than or equal to 1.")
    return value


def _clean_error(value: str) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError("error must be a non-empty string.")
    return copy_json_value(value, "error")

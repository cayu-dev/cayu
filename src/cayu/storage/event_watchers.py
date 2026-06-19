from __future__ import annotations

import asyncio
import sqlite3
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path

from cayu._validation import require_clean_nonblank, require_nonblank
from cayu.runtime.event_watchers import (
    EventWatcherClaim,
    EventWatcherDelivery,
    EventWatcherDeliveryStatus,
    EventWatcherState,
    EventWatcherStore,
)
from cayu.runtime.sessions import EventRecord
from cayu.storage import migrations as schema

from . import _sqlite_support as sqlite_support


class SQLiteEventWatcherStore(EventWatcherStore):
    """SQLite-backed durable delivery state for event watchers."""

    def __init__(
        self,
        path: str | Path,
        *,
        clock: Callable[[], datetime] | None = None,
        schema_mode: schema.SchemaMode = schema.SchemaMode.CREATE,
    ) -> None:
        if isinstance(path, Path):
            db_path = path
        elif type(path) is str:
            db_path = Path(require_nonblank(path, "path"))
        else:
            raise TypeError("SQLiteEventWatcherStore path must be a string or Path.")
        if not isinstance(schema_mode, schema.SchemaMode):
            raise TypeError("schema_mode must be a SchemaMode.")
        self.path = db_path
        self._lock = asyncio.Lock()
        self._clock = _clock_or_utc_now(clock)
        self._connection = sqlite_support.connect(db_path)
        sqlite_support.reconcile_schema(self._connection, schema_mode)

    async def load_state(self, watcher_name: str) -> EventWatcherState:
        watcher_name = require_clean_nonblank(watcher_name, "watcher_name")
        async with self._lock:
            row = self._connection.execute(
                """
                SELECT *
                FROM cayu_event_watcher_state
                WHERE watcher_name = ?
                """,
                (watcher_name,),
            ).fetchone()
            if row is None:
                return EventWatcherState(watcher_name=watcher_name)
            return _state_from_row(row)

    async def claim_event(
        self,
        *,
        watcher_name: str,
        record: EventRecord,
        lease_seconds: float,
    ) -> EventWatcherClaim | None:
        watcher_name = require_clean_nonblank(watcher_name, "watcher_name")
        if type(record) is not EventRecord:
            raise TypeError("record must be an EventRecord.")
        lease_seconds = _validate_positive_float(lease_seconds, "lease_seconds")
        now = self._clock()
        async with self._lock:
            try:
                self._connection.execute("BEGIN IMMEDIATE")
                state = self._load_state_unlocked(watcher_name)
                if state.cursor_sequence >= record.sequence:
                    self._connection.commit()
                    return None
                if (
                    state.delivery_status is EventWatcherDeliveryStatus.LEASED
                    and state.lease_expires_at is not None
                    and state.lease_expires_at > now
                ):
                    self._connection.commit()
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
                self._upsert_state_unlocked(
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
                    )
                )
                self._connection.commit()
                return claim
            except Exception:
                self._connection.rollback()
                raise

    async def mark_success(self, claim: EventWatcherClaim) -> EventWatcherDelivery:
        if type(claim) is not EventWatcherClaim:
            raise TypeError("claim must be an EventWatcherClaim.")
        now = self._clock()
        async with self._lock:
            try:
                self._connection.execute("BEGIN IMMEDIATE")
                state = self._matching_state_unlocked(claim)
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
                self._upsert_state_unlocked(updated)
                self._connection.commit()
                return _delivery_from_claim(
                    claim,
                    status=EventWatcherDeliveryStatus.SUCCEEDED,
                    cursor_sequence=updated.cursor_sequence,
                )
            except Exception:
                self._connection.rollback()
                raise

    async def mark_failure(
        self,
        claim: EventWatcherClaim,
        *,
        error: str,
        max_attempts: int,
    ) -> EventWatcherDelivery:
        if type(claim) is not EventWatcherClaim:
            raise TypeError("claim must be an EventWatcherClaim.")
        error = _clean_error(error)
        if type(max_attempts) is not int or max_attempts < 1:
            raise ValueError("max_attempts must be an integer greater than or equal to 1.")
        now = self._clock()
        async with self._lock:
            try:
                self._connection.execute("BEGIN IMMEDIATE")
                state = self._matching_state_unlocked(claim)
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
                self._upsert_state_unlocked(updated)
                self._connection.commit()
                return _delivery_from_claim(
                    claim,
                    status=status,
                    cursor_sequence=updated.cursor_sequence,
                    error=error,
                )
            except Exception:
                self._connection.rollback()
                raise

    async def close(self) -> None:
        async with self._lock:
            self._connection.close()

    def _load_state_unlocked(self, watcher_name: str) -> EventWatcherState:
        row = self._connection.execute(
            """
            SELECT *
            FROM cayu_event_watcher_state
            WHERE watcher_name = ?
            """,
            (watcher_name,),
        ).fetchone()
        if row is None:
            return EventWatcherState(watcher_name=watcher_name, updated_at=self._clock())
        return _state_from_row(row)

    def _matching_state_unlocked(self, claim: EventWatcherClaim) -> EventWatcherState:
        state = self._load_state_unlocked(claim.watcher_name)
        if state.pending_claim_id != claim.claim_id:
            raise ValueError("Watcher claim is no longer active.")
        if state.pending_event_id != claim.event_id:
            raise ValueError("Watcher claim event_id does not match active claim.")
        if state.pending_event_sequence != claim.event_sequence:
            raise ValueError("Watcher claim sequence does not match active claim.")
        if state.pending_attempt != claim.attempt:
            raise ValueError("Watcher claim attempt does not match active claim.")
        return state

    def _upsert_state_unlocked(self, state: EventWatcherState) -> None:
        self._connection.execute(
            """
            INSERT INTO cayu_event_watcher_state (
                watcher_name,
                cursor_sequence,
                pending_event_id,
                pending_event_sequence,
                pending_attempt,
                pending_claim_id,
                delivery_status,
                lease_expires_at,
                last_error,
                dead_lettered_count,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(watcher_name) DO UPDATE SET
                cursor_sequence = excluded.cursor_sequence,
                pending_event_id = excluded.pending_event_id,
                pending_event_sequence = excluded.pending_event_sequence,
                pending_attempt = excluded.pending_attempt,
                pending_claim_id = excluded.pending_claim_id,
                delivery_status = excluded.delivery_status,
                lease_expires_at = excluded.lease_expires_at,
                last_error = excluded.last_error,
                dead_lettered_count = excluded.dead_lettered_count,
                updated_at = excluded.updated_at
            """,
            (
                state.watcher_name,
                state.cursor_sequence,
                state.pending_event_id,
                state.pending_event_sequence,
                state.pending_attempt,
                state.pending_claim_id,
                None if state.delivery_status is None else str(state.delivery_status),
                _format_optional_datetime(state.lease_expires_at),
                state.last_error,
                state.dead_lettered_count,
                sqlite_support.format_datetime(state.updated_at),
            ),
        )


def _state_from_row(row: sqlite3.Row) -> EventWatcherState:
    return EventWatcherState(
        watcher_name=row["watcher_name"],
        cursor_sequence=row["cursor_sequence"],
        pending_event_id=row["pending_event_id"],
        pending_event_sequence=row["pending_event_sequence"],
        pending_attempt=row["pending_attempt"],
        pending_claim_id=row["pending_claim_id"],
        delivery_status=(
            None
            if row["delivery_status"] is None
            else EventWatcherDeliveryStatus(row["delivery_status"])
        ),
        lease_expires_at=_parse_optional_datetime(row["lease_expires_at"]),
        last_error=row["last_error"],
        dead_lettered_count=row["dead_lettered_count"],
        updated_at=sqlite_support.parse_datetime(row["updated_at"]),
    )


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


def _validate_positive_float(value: float, field_name: str) -> float:
    if type(value) not in {int, float} or value <= 0:
        raise ValueError(f"{field_name} must be greater than 0.")
    return float(value)


def _clean_error(value: str) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError("error must be a non-empty string.")
    return value


def _format_optional_datetime(value: datetime | None) -> str | None:
    if value is None:
        return None
    return sqlite_support.format_datetime(value)


def _parse_optional_datetime(value: str | None) -> datetime | None:
    if value is None:
        return None
    return sqlite_support.parse_datetime(value)

from __future__ import annotations

import asyncio
import sqlite3
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from cayu._validation import (
    require_durable_clean_nonblank as require_clean_nonblank,
)
from cayu._validation import (
    require_durable_nonblank,
    require_nonblank,
)
from cayu.runtime.event_watchers import (
    EventWatcherClaim,
    EventWatcherDeadLetter,
    EventWatcherDelivery,
    EventWatcherDeliveryStatus,
    EventWatcherSettlement,
    EventWatcherState,
    EventWatcherStore,
    claim_event_watcher_transition,
    copy_event_watcher_claim,
    copy_event_watcher_record,
    renew_event_watcher_transition,
    replay_event_watcher_settlement,
    settle_event_watcher_transition,
)
from cayu.runtime.event_watchers import (
    _clean_error as clean_watcher_error,
)
from cayu.runtime.event_watchers import (
    _validate_max_attempts as validate_watcher_max_attempts,
)
from cayu.runtime.sessions import EventRecord
from cayu.storage import migrations as schema

from . import _sqlite_support as sqlite_support

_SQLITE_MIN_REQUIRED_REVISION = 81


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
        sqlite_support.reconcile_schema(
            self._connection,
            schema_mode,
            app_min_supported=_SQLITE_MIN_REQUIRED_REVISION,
        )
        # Retry writer contention cooperatively so Runtime deadlines and other
        # lease heartbeats can run while this connection waits for authority.
        self._connection.execute("PRAGMA busy_timeout = 0")

    async def _begin_immediate(self) -> None:
        deadline = asyncio.get_running_loop().time() + 5.0
        while True:
            try:
                self._connection.execute("BEGIN IMMEDIATE")
                return
            except sqlite3.OperationalError as error:
                code = getattr(error, "sqlite_errorcode", None)
                remaining = deadline - asyncio.get_running_loop().time()
                if type(code) is not int or code & 0xFF != sqlite3.SQLITE_BUSY or remaining <= 0:
                    raise
                await asyncio.sleep(min(0.01, remaining))

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
        max_attempts: int = 3,
    ) -> EventWatcherClaim | EventWatcherDelivery | None:
        watcher_name = require_clean_nonblank(watcher_name, "watcher_name")
        record = copy_event_watcher_record(record)
        max_attempts = validate_watcher_max_attempts(max_attempts)
        async with self._lock:
            try:
                await self._begin_immediate()
                state = self._load_state_unlocked(watcher_name)
                now = max(self._clock(), state.updated_at)
                transition = claim_event_watcher_transition(
                    state,
                    record,
                    now=now,
                    lease_seconds=lease_seconds,
                    max_attempts=max_attempts,
                )
                self._upsert_state_unlocked(transition.state)
                if transition.dead_letter is not None:
                    self._insert_dead_letter_unlocked(transition.dead_letter)
                self._connection.commit()
                return transition.outcome
            except BaseException:
                self._connection.rollback()
                raise

    async def renew_claim(
        self, claim: EventWatcherClaim, *, lease_seconds: float
    ) -> EventWatcherClaim:
        claim = copy_event_watcher_claim(claim)
        async with self._lock:
            try:
                await self._begin_immediate()
                state = self._load_state_unlocked(claim.watcher_name)
                transition = renew_event_watcher_transition(
                    state,
                    claim,
                    now=max(self._clock(), state.updated_at),
                    lease_seconds=lease_seconds,
                )
                self._upsert_state_unlocked(transition.state)
                self._connection.commit()
                assert isinstance(transition.outcome, EventWatcherClaim)
                return transition.outcome
            except BaseException:
                self._connection.rollback()
                raise

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
            claim,
            error=clean_watcher_error(error),
            max_attempts=validate_watcher_max_attempts(max_attempts),
        )

    async def _settle(
        self,
        claim: EventWatcherClaim,
        *,
        error: str | None,
        max_attempts: int | None,
    ) -> EventWatcherDelivery:
        claim = copy_event_watcher_claim(claim)
        async with self._lock:
            try:
                await self._begin_immediate()
                row = self._connection.execute(
                    "SELECT receipt_json FROM cayu_event_watcher_settlements WHERE watcher_name = ? AND claim_id = ?",
                    (claim.watcher_name, claim.claim_id),
                ).fetchone()
                if row is not None:
                    result = replay_event_watcher_settlement(
                        EventWatcherSettlement.model_validate_json(row["receipt_json"]),
                        claim,
                        error=error,
                        max_attempts=max_attempts,
                    )
                    self._connection.commit()
                    return result
                state = self._load_state_unlocked(claim.watcher_name)
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
                self._upsert_state_unlocked(transition.state)
                if transition.dead_letter is not None:
                    self._insert_dead_letter_unlocked(transition.dead_letter)
                self._connection.execute(
                    "INSERT INTO cayu_event_watcher_settlements (watcher_name, claim_id, receipt_json) VALUES (?, ?, ?)",
                    (claim.watcher_name, claim.claim_id, receipt.model_dump_json()),
                )
                self._connection.commit()
                return transition.outcome
            except BaseException:
                self._connection.rollback()
                raise

    async def list_dead_letters(
        self,
        watcher_name: str,
        *,
        include_resolved: bool = False,
        limit: int = 100,
    ) -> list[EventWatcherDeadLetter]:
        watcher_name = require_clean_nonblank(watcher_name, "watcher_name")
        limit = _validate_dead_letter_limit(limit)
        clause = "" if include_resolved else "AND resolved_at IS NULL"
        async with self._lock:
            rows = self._connection.execute(
                f"""
                SELECT
                    watcher_name,
                    event_id,
                    event_sequence,
                    attempts,
                    error,
                    dead_lettered_at,
                    resolved_at
                FROM cayu_event_watcher_dead_letters
                WHERE watcher_name = ?
                {clause}
                ORDER BY event_sequence ASC
                LIMIT ?
                """,
                (watcher_name, limit),
            ).fetchall()
            return [_dead_letter_from_row(row) for row in rows]

    async def resolve_dead_letter(
        self,
        watcher_name: str,
        event_sequence: int,
    ) -> EventWatcherDeadLetter:
        watcher_name = require_clean_nonblank(watcher_name, "watcher_name")
        event_sequence = _validate_event_sequence(event_sequence)
        now = self._clock()
        async with self._lock:
            try:
                await self._begin_immediate()
                row = self._connection.execute(
                    """
                    SELECT
                        watcher_name,
                        event_id,
                        event_sequence,
                        attempts,
                        error,
                        dead_lettered_at,
                        resolved_at
                    FROM cayu_event_watcher_dead_letters
                    WHERE watcher_name = ? AND event_sequence = ?
                    """,
                    (watcher_name, event_sequence),
                ).fetchone()
                if row is None:
                    raise ValueError(
                        f"No dead-letter record for watcher {watcher_name!r} "
                        f"at sequence {event_sequence}."
                    )
                record = _dead_letter_from_row(row)
                if record.resolved_at is None:
                    resolved_at = now
                    self._connection.execute(
                        """
                        UPDATE cayu_event_watcher_dead_letters
                        SET resolved_at = ?
                        WHERE watcher_name = ? AND event_sequence = ?
                        """,
                        (
                            sqlite_support.format_datetime(resolved_at),
                            watcher_name,
                            event_sequence,
                        ),
                    )
                    record = record.model_copy(update={"resolved_at": resolved_at}, deep=True)
                self._connection.commit()
                return record
            except BaseException:
                self._connection.rollback()
                raise

    async def close(self) -> None:
        async with self._lock:
            self._connection.close()

    def _insert_dead_letter_unlocked(self, dead_letter: EventWatcherDeadLetter) -> None:
        # INSERT OR REPLACE keeps a re-dead-lettering of the same (watcher, sequence)
        # idempotent; in practice the cursor advances past a dead-lettered event so
        # this collides only on a replayed-then-refailed record.
        self._connection.execute(
            """
            INSERT OR REPLACE INTO cayu_event_watcher_dead_letters (
                watcher_name,
                event_sequence,
                event_id,
                attempts,
                error,
                dead_lettered_at,
                resolved_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                dead_letter.watcher_name,
                dead_letter.event_sequence,
                dead_letter.event_id,
                dead_letter.attempts,
                dead_letter.error,
                sqlite_support.format_datetime(dead_letter.dead_lettered_at),
                _format_optional_datetime(dead_letter.resolved_at),
            ),
        )

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


def _dead_letter_from_row(row: sqlite3.Row) -> EventWatcherDeadLetter:
    return EventWatcherDeadLetter(
        watcher_name=row["watcher_name"],
        event_id=row["event_id"],
        event_sequence=row["event_sequence"],
        attempts=row["attempts"],
        error=row["error"],
        dead_lettered_at=sqlite_support.parse_datetime(row["dead_lettered_at"]),
        resolved_at=_parse_optional_datetime(row["resolved_at"]),
    )


def _validate_dead_letter_limit(value: int) -> int:
    if type(value) is not int or value < 1:
        raise ValueError("limit must be an integer greater than or equal to 1.")
    return value


def _validate_event_sequence(value: int) -> int:
    if type(value) is not int or value < 1:
        raise ValueError("event_sequence must be an integer greater than or equal to 1.")
    return value


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


def _clean_error(value: str) -> str:
    return require_durable_nonblank(value, "error")


def _format_optional_datetime(value: datetime | None) -> str | None:
    if value is None:
        return None
    return sqlite_support.format_datetime(value)


def _parse_optional_datetime(value: str | None) -> datetime | None:
    if value is None:
        return None
    return sqlite_support.parse_datetime(value)

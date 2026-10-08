"""Durable PostgreSQL event-watcher claims, settlements and dead letters."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from cayu._validation import require_durable_clean_nonblank as require_clean_nonblank
from cayu.observability.watchers import (
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
from cayu.observability.watchers import _clean_error as clean_watcher_error
from cayu.observability.watchers import _validate_max_attempts as validate_watcher_max_attempts
from cayu.sessions.records import EventRecord
from cayu.storage import _postgres_base as postgres_base
from cayu.storage import _postgres_support as pg_support
from cayu.storage._phase_timing import PostgresTimingScope


class PostgresEventWatcherStore(postgres_base._PostgresStoreBase, EventWatcherStore):
    """Postgres-backed durable watcher state for hosted multi-worker apps."""

    async def load_state(self, watcher_name: str) -> EventWatcherState:
        watcher_name = require_clean_nonblank(watcher_name, "watcher_name")
        await self._ensure_ready()
        async with PostgresTimingScope(self._pool.connection()) as conn, conn.cursor() as cur:
            await cur.execute(
                """
                SELECT
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
                FROM cayu_event_watcher_state
                WHERE watcher_name = %s
                """,
                (watcher_name,),
            )
            row = await cur.fetchone()
            if row is None:
                return EventWatcherState(watcher_name=watcher_name)
            return _event_watcher_state_from_row(row)

    _min_required_revision = 81

    @staticmethod
    async def _watcher_database_now(cur: Any) -> datetime:
        await cur.execute("SELECT clock_timestamp()")
        row = await cur.fetchone()
        if row is None:
            raise RuntimeError("PostgreSQL did not return authoritative watcher time.")
        return pg_support.to_utc(row[0])

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
        await self._ensure_ready()
        async with PostgresTimingScope(self._pool.connection()) as conn, conn.cursor() as cur:
            state = await self._load_watcher_state_for_update(
                cur, watcher_name, now=await self._watcher_database_now(cur)
            )
            now = max(await self._watcher_database_now(cur), state.updated_at)
            transition = claim_event_watcher_transition(
                state,
                record,
                now=now,
                lease_seconds=lease_seconds,
                max_attempts=max_attempts,
            )
            await self._upsert_watcher_state(cur, transition.state)
            if transition.dead_letter is not None:
                await self._insert_dead_letter(cur, transition.dead_letter)
            await conn.commit()
            return transition.outcome

    async def renew_claim(
        self, claim: EventWatcherClaim, *, lease_seconds: float
    ) -> EventWatcherClaim:
        claim = copy_event_watcher_claim(claim)
        await self._ensure_ready()
        async with PostgresTimingScope(self._pool.connection()) as conn, conn.cursor() as cur:
            state = await self._load_watcher_state_for_update(
                cur, claim.watcher_name, now=await self._watcher_database_now(cur)
            )
            transition = renew_event_watcher_transition(
                state,
                claim,
                now=max(await self._watcher_database_now(cur), state.updated_at),
                lease_seconds=lease_seconds,
            )
            await self._upsert_watcher_state(cur, transition.state)
            await conn.commit()
            assert isinstance(transition.outcome, EventWatcherClaim)
            return transition.outcome

    async def mark_success(self, claim: EventWatcherClaim) -> EventWatcherDelivery:
        return await self._settle_watcher(claim, error=None, max_attempts=None)

    async def mark_failure(
        self,
        claim: EventWatcherClaim,
        *,
        error: str,
        max_attempts: int,
    ) -> EventWatcherDelivery:
        return await self._settle_watcher(
            claim,
            error=clean_watcher_error(error),
            max_attempts=validate_watcher_max_attempts(max_attempts),
        )

    async def _settle_watcher(
        self,
        claim: EventWatcherClaim,
        *,
        error: str | None,
        max_attempts: int | None,
    ) -> EventWatcherDelivery:
        claim = copy_event_watcher_claim(claim)
        await self._ensure_ready()
        async with PostgresTimingScope(self._pool.connection()) as conn, conn.cursor() as cur:
            state = await self._load_watcher_state_for_update(
                cur, claim.watcher_name, now=await self._watcher_database_now(cur)
            )
            await cur.execute(
                "SELECT receipt_json FROM cayu_event_watcher_settlements WHERE watcher_name = %s AND claim_id = %s",
                (claim.watcher_name, claim.claim_id),
            )
            row = await cur.fetchone()
            if row is not None:
                result = replay_event_watcher_settlement(
                    EventWatcherSettlement.model_validate(row[0]),
                    claim,
                    error=error,
                    max_attempts=max_attempts,
                )
                await conn.commit()
                return result
            transition = settle_event_watcher_transition(
                state,
                claim,
                now=max(await self._watcher_database_now(cur), state.updated_at),
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
            await self._upsert_watcher_state(cur, transition.state)
            if transition.dead_letter is not None:
                await self._insert_dead_letter(cur, transition.dead_letter)
            await cur.execute(
                "INSERT INTO cayu_event_watcher_settlements (watcher_name, claim_id, receipt_json) VALUES (%s, %s, %s::jsonb)",
                (claim.watcher_name, claim.claim_id, receipt.model_dump_json()),
            )
            await conn.commit()
            return transition.outcome

    async def list_dead_letters(
        self,
        watcher_name: str,
        *,
        include_resolved: bool = False,
        limit: int = 100,
    ) -> list[EventWatcherDeadLetter]:
        watcher_name = require_clean_nonblank(watcher_name, "watcher_name")
        limit = _validate_dead_letter_limit(limit)
        await self._ensure_ready()
        clause = "" if include_resolved else "AND resolved_at IS NULL"
        async with PostgresTimingScope(self._pool.connection()) as conn, conn.cursor() as cur:
            await cur.execute(
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
                WHERE watcher_name = %s
                {clause}
                ORDER BY event_sequence ASC
                LIMIT %s
                """,
                (watcher_name, limit),
            )
            rows = await cur.fetchall()
            return [_event_watcher_dead_letter_from_row(row) for row in rows]

    async def resolve_dead_letter(
        self,
        watcher_name: str,
        event_sequence: int,
    ) -> EventWatcherDeadLetter:
        watcher_name = require_clean_nonblank(watcher_name, "watcher_name")
        event_sequence = _validate_event_sequence(event_sequence)
        await self._ensure_ready()
        now = datetime.now(UTC)
        async with PostgresTimingScope(self._pool.connection()) as conn, conn.cursor() as cur:
            await cur.execute(
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
                WHERE watcher_name = %s AND event_sequence = %s
                FOR UPDATE
                """,
                (watcher_name, event_sequence),
            )
            row = await cur.fetchone()
            if row is None:
                raise ValueError(
                    f"No dead-letter record for watcher {watcher_name!r} "
                    f"at sequence {event_sequence}."
                )
            record = _event_watcher_dead_letter_from_row(row)
            if record.resolved_at is None:
                await cur.execute(
                    """
                    UPDATE cayu_event_watcher_dead_letters
                    SET resolved_at = %s
                    WHERE watcher_name = %s AND event_sequence = %s
                    """,
                    (now, watcher_name, event_sequence),
                )
                record = record.model_copy(update={"resolved_at": now}, deep=True)
            await conn.commit()
            return record

    async def _insert_dead_letter(self, cur: Any, dead_letter: EventWatcherDeadLetter) -> None:
        await cur.execute(
            """
            INSERT INTO cayu_event_watcher_dead_letters (
                watcher_name,
                event_sequence,
                event_id,
                attempts,
                error,
                dead_lettered_at,
                resolved_at
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (watcher_name, event_sequence) DO UPDATE SET
                event_id = excluded.event_id,
                attempts = excluded.attempts,
                error = excluded.error,
                dead_lettered_at = excluded.dead_lettered_at,
                resolved_at = excluded.resolved_at
            """,
            (
                dead_letter.watcher_name,
                dead_letter.event_sequence,
                dead_letter.event_id,
                dead_letter.attempts,
                dead_letter.error,
                pg_support.to_utc(dead_letter.dead_lettered_at),
                pg_support.to_utc_optional(dead_letter.resolved_at),
            ),
        )

    async def _load_watcher_state_for_update(
        self,
        cur: Any,
        watcher_name: str,
        *,
        now: datetime,
    ) -> EventWatcherState:
        await cur.execute(
            """
            INSERT INTO cayu_event_watcher_state (
                watcher_name,
                cursor_sequence,
                pending_attempt,
                dead_lettered_count,
                updated_at
            )
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (watcher_name) DO NOTHING
            """,
            (watcher_name, 0, 0, 0, now),
        )
        await cur.execute(
            """
            SELECT
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
            FROM cayu_event_watcher_state
            WHERE watcher_name = %s
            FOR UPDATE
            """,
            (watcher_name,),
        )
        row = await cur.fetchone()
        if row is None:
            raise RuntimeError(f"Failed to initialize event watcher state: {watcher_name}")
        return _event_watcher_state_from_row(row)

    async def _upsert_watcher_state(self, cur: Any, state: EventWatcherState) -> None:
        await cur.execute(
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
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (watcher_name) DO UPDATE SET
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
                pg_support.to_utc_optional(state.lease_expires_at),
                state.last_error,
                state.dead_lettered_count,
                pg_support.to_utc(state.updated_at),
            ),
        )


def _event_watcher_state_from_row(row: tuple[Any, ...]) -> EventWatcherState:
    return EventWatcherState(
        watcher_name=row[0],
        cursor_sequence=row[1],
        pending_event_id=row[2],
        pending_event_sequence=row[3],
        pending_attempt=row[4],
        pending_claim_id=row[5],
        delivery_status=None if row[6] is None else EventWatcherDeliveryStatus(row[6]),
        lease_expires_at=pg_support.to_utc_optional(row[7]),
        last_error=row[8],
        dead_lettered_count=row[9],
        updated_at=pg_support.to_utc(row[10]),
    )


def _event_watcher_dead_letter_from_row(row: tuple[Any, ...]) -> EventWatcherDeadLetter:
    return EventWatcherDeadLetter(
        watcher_name=row[0],
        event_id=row[1],
        event_sequence=row[2],
        attempts=row[3],
        error=row[4],
        dead_lettered_at=pg_support.to_utc(row[5]),
        resolved_at=pg_support.to_utc_optional(row[6]),
    )


def _validate_dead_letter_limit(value: int) -> int:
    if type(value) is not int or value < 1:
        raise ValueError("limit must be an integer greater than or equal to 1.")
    return value


def _validate_event_sequence(value: int) -> int:
    if type(value) is not int or value < 1:
        raise ValueError("event_sequence must be an integer greater than or equal to 1.")
    return value

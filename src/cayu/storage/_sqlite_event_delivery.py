"""Complete SQLite persisted-event delivery operations.

Delivery operations receive native capabilities. Enqueue joins its caller's
publication transaction; it never acquires or commits a connection.
"""

from __future__ import annotations

import math
import sqlite3
from collections.abc import Callable, Sequence
from datetime import UTC, datetime, timedelta
from uuid import uuid4

from cayu._validation import require_durable_clean_nonblank as require_clean_nonblank
from cayu.events import Event
from cayu.sessions import event_delivery as side_effect_health
from cayu.sessions.base import (
    _copy_failed_first_delivery_retirement,
    _copy_pending_first_event_delivery,
    _event_file_attachment_attestations_are_runtime_owned,
    _event_input_contract_is_runtime_owned,
)
from cayu.sessions.event_delivery import (
    PersistedEventSideEffectClaim,
    PersistedEventSideEffectClaimLost,
    PersistedEventSideEffectDelivery,
    PersistedEventSideEffectHealth,
    PersistedEventSideEffectPage,
    PersistedEventSideEffectQuery,
    PersistedEventSideEffectStatus,
    validate_persisted_event_side_effect_error,
)
from cayu.storage import _sqlite_records as sqlite_records
from cayu.storage._sqlite_connection import SQLiteOperationRunner


def delivery_from_row(row: sqlite3.Row) -> PersistedEventSideEffectDelivery:
    return PersistedEventSideEffectDelivery(
        session_id=row["session_id"],
        event_id=row["event_id"],
        event_sequence=row["event_sequence"],
        status=PersistedEventSideEffectStatus(row["status"]),
        attempts=row["attempts"],
        claim_id=row["claim_id"],
        lease_expires_at=(
            None
            if row["lease_expires_at"] is None
            else sqlite_records.parse_datetime(row["lease_expires_at"])
        ),
        next_attempt_at=(
            None
            if row["next_attempt_at"] is None
            else sqlite_records.parse_datetime(row["next_attempt_at"])
        ),
        last_error=row["last_error"],
        updated_at=sqlite_records.parse_datetime(row["updated_at"]),
    )


def enqueue_persisted_event_side_effects(
    connection: sqlite3.Connection, session_id: str, events: Sequence[Event]
) -> None:
    if not events:
        return
    event_ids: list[str] = []
    runtime_owned_input_contract_event_ids: list[str] = []
    runtime_owned_file_attestation_event_ids: list[str] = []
    for event in events:
        event_ids.append(event.id)
        if _event_input_contract_is_runtime_owned(event):
            runtime_owned_input_contract_event_ids.append(event.id)
        if _event_file_attachment_attestations_are_runtime_owned(event):
            runtime_owned_file_attestation_event_ids.append(event.id)
    # Rows predating revision 31 may contain caller-authored payload text but
    # cannot carry the proof bit, so that text remains untrusted after migration.
    if runtime_owned_input_contract_event_ids:
        connection.executemany(
            """
            UPDATE cayu_events
            SET input_contract_runtime_owned = 1
            WHERE session_id = ?
              AND event_id = ?
              AND event_type IN (
                  'session.started',
                  'session.resumed',
                  'session.message.queued',
                  'session.message.delivered'
              )
              AND json_type(payload_json, '$.input_contract') = 'text'
            """,
            [(session_id, event_id) for event_id in runtime_owned_input_contract_event_ids],
        )
    if runtime_owned_file_attestation_event_ids:
        connection.executemany(
            """
            UPDATE cayu_events
            SET file_attachment_attestations_runtime_owned = 1
            WHERE session_id = ?
              AND event_id = ?
              AND event_type = 'model.started'
              AND json_type(payload_json, '$.file_attachment_attestations') = 'text'
            """,
            [(session_id, event_id) for event_id in runtime_owned_file_attestation_event_ids],
        )
    connection.executemany(
        """
        INSERT INTO cayu_persisted_event_side_effects (
            session_id, event_id, event_sequence, status, attempts, updated_at
        )
        SELECT session_id, event_id, sequence, 'pending', 0, timestamp
        FROM cayu_events
        WHERE session_id = ?
          AND event_id = ?
          AND event_type <> 'runtime.sink.failed'
        """,
        [(session_id, event_id) for event_id in event_ids],
    )


async def claim_first_persisted_event_side_effect(
    run_write: SQLiteOperationRunner,
    expected: PersistedEventSideEffectDelivery,
    *,
    ownership_clock: Callable[[], datetime],
) -> PersistedEventSideEffectClaim | None:
    expected = _copy_pending_first_event_delivery(expected)
    return await _claim_persisted_event_side_effect(
        run_write,
        session_id=expected.session_id,
        event_id=expected.event_id,
        expected=expected,
        ownership_clock=ownership_clock,
    )


async def claim_persisted_event_side_effect(
    run_write: SQLiteOperationRunner,
    *,
    session_id: str | None = None,
    event_id: str | None = None,
    lease_seconds: float = 300.0,
    ownership_clock: Callable[[], datetime],
) -> PersistedEventSideEffectClaim | None:
    return await _claim_persisted_event_side_effect(
        run_write,
        session_id=session_id,
        event_id=event_id,
        lease_seconds=lease_seconds,
        ownership_clock=ownership_clock,
    )


async def _claim_persisted_event_side_effect(
    run_write: SQLiteOperationRunner,
    *,
    session_id: str | None = None,
    event_id: str | None = None,
    lease_seconds: float = 300.0,
    expected: PersistedEventSideEffectDelivery | None = None,
    ownership_clock: Callable[[], datetime],
) -> PersistedEventSideEffectClaim | None:
    if session_id is not None:
        session_id = require_clean_nonblank(session_id, "session_id")
    if event_id is not None:
        event_id = require_clean_nonblank(event_id, "event_id")
    if (session_id is None) != (event_id is None):
        raise ValueError("session_id and event_id must be supplied together.")
    if type(lease_seconds) not in {int, float} or lease_seconds <= 0:
        raise ValueError("lease_seconds must be greater than 0.")

    def statement(connection: sqlite3.Connection) -> PersistedEventSideEffectClaim | None:
        try:
            connection.execute("BEGIN IMMEDIATE")
            now = ownership_clock()
            lease_expires_at = now + timedelta(seconds=float(lease_seconds))
            formatted_now = sqlite_records.format_datetime(now)
            filters = [
                "(status = 'pending' "
                "OR (status = 'failed' AND "
                "(next_attempt_at IS NULL OR next_attempt_at <= ?)) "
                "OR (status = 'leased' AND lease_expires_at <= ?))",
                "NOT EXISTS (SELECT 1 FROM cayu_session_closure_progress AS p "
                "WHERE p.root_session_id = cayu_persisted_event_side_effects.session_id "
                "OR EXISTS (SELECT 1 FROM json_each(p.progress_json, '$.descendants') AS child "
                "WHERE json_extract(child.value, '$.session_id') = "
                "cayu_persisted_event_side_effects.session_id))",
            ]
            params: list[object] = [formatted_now, formatted_now]
            if session_id is not None and event_id is not None:
                filters.extend(["session_id = ?", "event_id = ?"])
                params.extend([session_id, event_id])
            delivery_row = connection.execute(
                "SELECT * FROM cayu_persisted_event_side_effects WHERE "
                + " AND ".join(filters)
                + " ORDER BY event_sequence ASC LIMIT 1",
                params,
            ).fetchone()
            if delivery_row is None:
                connection.commit()
                return None
            if expected is not None and delivery_from_row(delivery_row) != expected:
                connection.rollback()
                return None
            claim_id = str(uuid4())
            attempt = int(delivery_row["attempts"]) + 1
            connection.execute(
                "UPDATE cayu_persisted_event_side_effects "
                "SET status = 'leased', attempts = ?, claim_id = ?, "
                "lease_expires_at = ?, next_attempt_at = NULL, "
                "last_error = NULL, updated_at = ? "
                "WHERE session_id = ? AND event_id = ?",
                (
                    attempt,
                    claim_id,
                    sqlite_records.format_datetime(lease_expires_at),
                    sqlite_records.format_datetime(now),
                    delivery_row["session_id"],
                    delivery_row["event_id"],
                ),
            )
            event_row = connection.execute(
                f"SELECT {', '.join(sqlite_records.EVENT_COLUMN_NAMES)} FROM cayu_events "
                "WHERE session_id = ? AND event_id = ?",
                (delivery_row["session_id"], delivery_row["event_id"]),
            ).fetchone()
            if event_row is None:
                raise RuntimeError("Persisted side-effect delivery lost its source event.")
            connection.commit()
            return PersistedEventSideEffectClaim(
                session_id=delivery_row["session_id"],
                event_id=delivery_row["event_id"],
                event_sequence=delivery_row["event_sequence"],
                event=sqlite_records.event_from_row(event_row),
                attempt=attempt,
                claim_id=claim_id,
                lease_expires_at=lease_expires_at,
            )
        except Exception:
            connection.rollback()
            raise

    return await run_write(statement)


async def get_persisted_event_side_effect_delivery(
    run_read: SQLiteOperationRunner, *, session_id: str, event_id: str
) -> PersistedEventSideEffectDelivery | None:
    session_id = require_clean_nonblank(session_id, "session_id")
    event_id = require_clean_nonblank(event_id, "event_id")

    def query(connection: sqlite3.Connection) -> PersistedEventSideEffectDelivery | None:
        row = connection.execute(
            "SELECT * FROM cayu_persisted_event_side_effects WHERE session_id = ? AND event_id = ?",
            (session_id, event_id),
        ).fetchone()
        return None if row is None else delivery_from_row(row)

    return await run_read(query)


async def retire_failed_first_event_delivery(
    run_write: SQLiteOperationRunner,
    expected: PersistedEventSideEffectDelivery,
    *,
    ownership_clock: Callable[[], datetime],
) -> PersistedEventSideEffectDelivery | None:
    expected = _copy_failed_first_delivery_retirement(expected)

    def statement(connection: sqlite3.Connection) -> PersistedEventSideEffectDelivery | None:
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM cayu_persisted_event_side_effects WHERE session_id = ? AND event_id = ?",
                (expected.session_id, expected.event_id),
            ).fetchone()
            if row is None or delivery_from_row(row) != expected:
                connection.rollback()
                return None
            retired = expected.model_copy(
                update={
                    "status": PersistedEventSideEffectStatus.DEAD_LETTERED,
                    "next_attempt_at": None,
                    "updated_at": ownership_clock(),
                }
            )
            connection.execute(
                "UPDATE cayu_persisted_event_side_effects SET status = 'dead_lettered', "
                "next_attempt_at = NULL, updated_at = ? WHERE session_id = ? AND event_id = ?",
                (
                    sqlite_records.format_datetime(retired.updated_at),
                    expected.session_id,
                    expected.event_id,
                ),
            )
            connection.commit()
            return retired
        except Exception:
            connection.rollback()
            raise

    return await run_write(statement)


async def mark_persisted_event_side_effect_delivered(
    run_write: SQLiteOperationRunner,
    claim: PersistedEventSideEffectClaim,
    *,
    ownership_clock: Callable[[], datetime],
) -> PersistedEventSideEffectDelivery:
    claim = PersistedEventSideEffectClaim.model_validate(claim)
    return await _finish_persisted_event_side_effect_claim(
        run_write,
        claim,
        status=PersistedEventSideEffectStatus.DELIVERED,
        error=None,
        retry_delay_seconds=None,
        ownership_clock=ownership_clock,
    )


async def mark_persisted_event_side_effect_failed(
    run_write: SQLiteOperationRunner,
    claim: PersistedEventSideEffectClaim,
    *,
    error: str,
    max_attempts: int,
    retry_delay_seconds: float,
    ownership_clock: Callable[[], datetime],
) -> PersistedEventSideEffectDelivery:
    claim = PersistedEventSideEffectClaim.model_validate(claim)
    error = validate_persisted_event_side_effect_error(error)
    if type(max_attempts) is not int or max_attempts < 1:
        raise ValueError("max_attempts must be an integer greater than or equal to 1.")
    if (
        type(retry_delay_seconds) not in {int, float}
        or not math.isfinite(retry_delay_seconds)
        or retry_delay_seconds < 0
    ):
        raise ValueError("retry_delay_seconds must be a finite non-negative number.")
    dead_lettered = claim.attempt >= max_attempts
    return await _finish_persisted_event_side_effect_claim(
        run_write,
        claim,
        status=(
            PersistedEventSideEffectStatus.DEAD_LETTERED
            if dead_lettered
            else PersistedEventSideEffectStatus.FAILED
        ),
        error=error,
        retry_delay_seconds=(None if dead_lettered else float(retry_delay_seconds)),
        ownership_clock=ownership_clock,
    )


async def defer_persisted_event_side_effect(
    run_write: SQLiteOperationRunner,
    claim: PersistedEventSideEffectClaim,
    *,
    ownership_clock: Callable[[], datetime],
) -> PersistedEventSideEffectDelivery:
    claim = PersistedEventSideEffectClaim.model_validate(claim)
    return await _finish_persisted_event_side_effect_claim(
        run_write,
        claim,
        status=PersistedEventSideEffectStatus.PENDING,
        error=None,
        retry_delay_seconds=None,
        deferred=True,
        ownership_clock=ownership_clock,
    )


async def renew_persisted_event_side_effect(
    run_write: SQLiteOperationRunner,
    claim: PersistedEventSideEffectClaim,
    *,
    lease_seconds: float = 300.0,
    ownership_clock: Callable[[], datetime],
) -> PersistedEventSideEffectDelivery:
    claim = PersistedEventSideEffectClaim.model_validate(claim)
    if type(lease_seconds) not in {int, float} or not 0 < lease_seconds <= 86_400:
        raise ValueError("lease_seconds must be positive and at most 86400.")

    def statement(connection: sqlite3.Connection) -> PersistedEventSideEffectDelivery:
        try:
            connection.execute("BEGIN IMMEDIATE")
            now = ownership_clock()
            cursor = connection.execute(
                "UPDATE cayu_persisted_event_side_effects "
                "SET lease_expires_at = MAX(lease_expires_at, ?), updated_at = ? "
                "WHERE session_id = ? AND event_id = ? AND status = 'leased' "
                "AND claim_id = ? AND attempts = ? AND lease_expires_at > ?",
                (
                    sqlite_records.format_datetime(now + timedelta(seconds=float(lease_seconds))),
                    sqlite_records.format_datetime(now),
                    claim.session_id,
                    claim.event_id,
                    claim.claim_id,
                    claim.attempt,
                    sqlite_records.format_datetime(now),
                ),
            )
            if cursor.rowcount != 1:
                raise PersistedEventSideEffectClaimLost(
                    "Persisted event side-effect claim is no longer active."
                )
            row = connection.execute(
                "SELECT * FROM cayu_persisted_event_side_effects "
                "WHERE session_id = ? AND event_id = ?",
                (claim.session_id, claim.event_id),
            ).fetchone()
            if row is None:
                raise RuntimeError("Persisted event side-effect delivery disappeared.")
            delivery = delivery_from_row(row)
            connection.commit()
            return delivery
        except Exception:
            connection.rollback()
            raise

    return await run_write(statement)


async def _finish_persisted_event_side_effect_claim(
    run_write: SQLiteOperationRunner,
    claim: PersistedEventSideEffectClaim,
    *,
    status: PersistedEventSideEffectStatus,
    error: str | None,
    retry_delay_seconds: float | None,
    deferred: bool = False,
    ownership_clock: Callable[[], datetime],
) -> PersistedEventSideEffectDelivery:
    def statement(connection: sqlite3.Connection) -> PersistedEventSideEffectDelivery:
        try:
            connection.execute("BEGIN IMMEDIATE")
            now = ownership_clock()
            next_attempt_at = (
                None
                if retry_delay_seconds is None
                else now + timedelta(seconds=retry_delay_seconds)
            )
            cursor = connection.execute(
                "UPDATE cayu_persisted_event_side_effects "
                "SET status = ?, claim_id = NULL, lease_expires_at = NULL, "
                "next_attempt_at = ?, last_error = ?, updated_at = ?, attempts = attempts - ? "
                "WHERE session_id = ? AND event_id = ? AND status = 'leased' "
                "AND claim_id = ? AND attempts = ?",
                (
                    str(status),
                    (
                        None
                        if next_attempt_at is None
                        else sqlite_records.format_datetime(next_attempt_at)
                    ),
                    error,
                    sqlite_records.format_datetime(now),
                    int(deferred),
                    claim.session_id,
                    claim.event_id,
                    claim.claim_id,
                    claim.attempt,
                ),
            )
            if cursor.rowcount != 1:
                existing = connection.execute(
                    "SELECT 1 FROM cayu_persisted_event_side_effects "
                    "WHERE session_id = ? AND event_id = ?",
                    (claim.session_id, claim.event_id),
                ).fetchone()
                if existing is None:
                    raise ValueError("Persisted event side-effect delivery was not found.")
                raise PersistedEventSideEffectClaimLost(
                    "Persisted event side-effect claim is no longer active."
                )
            row = connection.execute(
                "SELECT * FROM cayu_persisted_event_side_effects "
                "WHERE session_id = ? AND event_id = ?",
                (claim.session_id, claim.event_id),
            ).fetchone()
            if row is None:
                raise RuntimeError("Persisted event side-effect delivery disappeared.")
            delivery = delivery_from_row(row)
            connection.commit()
            return delivery
        except Exception:
            connection.rollback()
            raise

    return await run_write(statement)


async def get_persisted_event_side_effect_health(
    run_read: SQLiteOperationRunner, *, ownership_clock: Callable[[], datetime]
) -> PersistedEventSideEffectHealth:
    def query(connection: sqlite3.Connection) -> PersistedEventSideEffectHealth:
        now = ownership_clock().astimezone(UTC)
        row = connection.execute(
            side_effect_health.health_sql("?"), (sqlite_records.format_datetime(now),)
        ).fetchone()
        return side_effect_health.finish_health(dict(row), now)

    return await run_read(query)


async def query_persisted_event_side_effect_deliveries(
    run_read: SQLiteOperationRunner,
    query: PersistedEventSideEffectQuery,
    *,
    ownership_clock: Callable[[], datetime],
) -> PersistedEventSideEffectPage:
    query = PersistedEventSideEffectQuery.model_validate(query)
    side_effect_health.cursor_key(query)

    def read(connection: sqlite3.Connection) -> PersistedEventSideEffectPage:
        now = ownership_clock().astimezone(UTC)
        sql, params = side_effect_health.page_sql(query, sqlite_records.format_datetime(now), "?")
        rows = connection.execute(sql, params).fetchall()
        return side_effect_health.page(
            [delivery_from_row(row) for row in rows],
            query,
            now,
        )

    return await run_read(read)


async def list_persisted_event_side_effect_deliveries(
    run_read: SQLiteOperationRunner,
    *,
    statuses: set[PersistedEventSideEffectStatus] | None = None,
    claimable_only: bool = False,
    after_sequence: int | None = None,
    limit: int = 100,
    ownership_clock: Callable[[], datetime],
) -> list[PersistedEventSideEffectDelivery]:
    if type(limit) is not int or not 1 <= limit <= 1000:
        raise ValueError("limit must be between 1 and 1000.")
    if type(claimable_only) is not bool:
        raise TypeError("claimable_only must be a bool.")
    if after_sequence is not None and (type(after_sequence) is not int or after_sequence < 0):
        raise ValueError("after_sequence must be a non-negative integer.")
    selected_statuses = (
        None
        if statuses is None
        else sorted(str(PersistedEventSideEffectStatus(status)) for status in statuses)
    )

    def query(connection: sqlite3.Connection) -> list[PersistedEventSideEffectDelivery]:
        clauses: list[str] = []
        params: list[object] = []
        if after_sequence is not None:
            clauses.append("event_sequence > ?")
            params.append(after_sequence)
        if selected_statuses is not None:
            if not selected_statuses:
                return []
            placeholders = ", ".join("?" for _ in selected_statuses)
            clauses.append(f"status IN ({placeholders})")
            params.extend(selected_statuses)
        if claimable_only:
            clauses.append(
                "(status = 'pending' "
                "OR (status = 'failed' AND "
                "(next_attempt_at IS NULL OR next_attempt_at <= ?)) "
                "OR (status = 'leased' AND lease_expires_at <= ?))"
            )
            formatted_now = sqlite_records.format_datetime(ownership_clock())
            params.extend([formatted_now, formatted_now])
        where = "" if not clauses else "WHERE " + " AND ".join(clauses)
        params.append(limit)
        rows = connection.execute(
            "SELECT * FROM cayu_persisted_event_side_effects "
            f"{where} ORDER BY event_sequence ASC LIMIT ?",
            params,
        ).fetchall()
        return [delivery_from_row(row) for row in rows]

    return await run_read(query)

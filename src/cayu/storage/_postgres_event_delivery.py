"""Complete PostgreSQL persisted-event delivery operations.

Delivery operations receive native capabilities. Enqueue joins its caller's
publication transaction; it never acquires or commits a connection.
"""

from __future__ import annotations

import math
from collections.abc import Awaitable, Callable, Sequence
from contextlib import AbstractAsyncContextManager
from datetime import datetime
from typing import Any, LiteralString, cast
from uuid import uuid4

from cayu._validation import require_durable_clean_nonblank as require_clean_nonblank
from cayu.events import Event
from cayu.sessions import event_delivery as side_effect_health
from cayu.sessions.base import (
    _copy_failed_first_delivery_retirement,
    _copy_pending_first_event_delivery,
)
from cayu.sessions.event_delivery import (
    PersistedEventSideEffectClaim,
    PersistedEventSideEffectClaimLost,
    PersistedEventSideEffectDelivery,
    PersistedEventSideEffectHealth,
    PersistedEventSideEffectPage,
    PersistedEventSideEffectQuery,
    PersistedEventSideEffectStatus,
    _event_file_attachment_attestations_are_runtime_owned,
    _event_input_contract_is_runtime_owned,
    validate_persisted_event_side_effect_error,
)
from cayu.storage import _postgres_support as pg_support

PostgresConnection = Callable[[], AbstractAsyncContextManager[Any]]


async def enqueue_persisted_event_side_effects(
    cur: Any, session_id: str, events: Sequence[Event]
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
    # Presence alone is not authority: rows predating revision 31 may contain
    # caller-authored payload text but cannot carry this proof bit.
    if runtime_owned_input_contract_event_ids:
        await cur.execute(
            """
                UPDATE cayu_events
                SET input_contract_runtime_owned = TRUE
                WHERE session_id = %s
                  AND event_id = ANY(%s)
                  AND event_type IN (
                      'session.started',
                      'session.resumed',
                      'session.message.queued',
                      'session.message.delivered'
                  )
                  AND jsonb_typeof(payload -> 'input_contract') = 'string'
                """,
            (session_id, runtime_owned_input_contract_event_ids),
        )
    if runtime_owned_file_attestation_event_ids:
        await cur.execute(
            """
                UPDATE cayu_events
                SET file_attachment_attestations_runtime_owned = TRUE
                WHERE session_id = %s
                  AND event_id = ANY(%s)
                  AND event_type = 'model.started'
                  AND jsonb_typeof(payload -> 'file_attachment_attestations') = 'string'
                """,
            (session_id, runtime_owned_file_attestation_event_ids),
        )
    await cur.execute(
        """
            INSERT INTO cayu_persisted_event_side_effects (
                session_id, event_id, event_sequence, status, attempts, updated_at
            )
            SELECT session_id, event_id, sequence, 'pending', 0, timestamp
            FROM cayu_events
            WHERE session_id = %s
              AND event_id = ANY(%s)
              AND event_type <> 'runtime.sink.failed'
            """,
        (session_id, event_ids),
    )


async def claim_first_persisted_event_side_effect(
    connect: PostgresConnection,
    expected: PersistedEventSideEffectDelivery,
    *,
    lock_closure_lineage: Callable[[Any], Awaitable[None]],
) -> PersistedEventSideEffectClaim | None:
    expected = _copy_pending_first_event_delivery(expected)
    return await _claim_persisted_event_side_effect(
        connect,
        session_id=expected.session_id,
        event_id=expected.event_id,
        expected=expected,
        lock_closure_lineage=lock_closure_lineage,
    )


async def claim_persisted_event_side_effect(
    connect: PostgresConnection,
    *,
    session_id: str | None = None,
    event_id: str | None = None,
    lease_seconds: float = 300.0,
    lock_closure_lineage: Callable[[Any], Awaitable[None]],
) -> PersistedEventSideEffectClaim | None:
    return await _claim_persisted_event_side_effect(
        connect,
        session_id=session_id,
        event_id=event_id,
        lease_seconds=lease_seconds,
        lock_closure_lineage=lock_closure_lineage,
    )


async def _claim_persisted_event_side_effect(
    connect: PostgresConnection,
    *,
    session_id: str | None = None,
    event_id: str | None = None,
    lease_seconds: float = 300.0,
    expected: PersistedEventSideEffectDelivery | None = None,
    lock_closure_lineage: Callable[[Any], Awaitable[None]],
) -> PersistedEventSideEffectClaim | None:
    if session_id is not None:
        session_id = require_clean_nonblank(session_id, "session_id")
    if event_id is not None:
        event_id = require_clean_nonblank(event_id, "event_id")
    if (session_id is None) != (event_id is None):
        raise ValueError("session_id and event_id must be supplied together.")
    if type(lease_seconds) not in {int, float} or lease_seconds <= 0:
        raise ValueError("lease_seconds must be greater than 0.")
    claim_id = str(uuid4())
    async with connect() as conn:
        try:
            async with conn.cursor() as cur:
                exact_filter = ""
                params: list[Any] = []
                # The same transaction-level lock protects closure admission.
                # Exclude closed targets in selection, so one retained target
                # cannot starve unrelated event deliveries.
                await lock_closure_lineage(cur)
                if expected is not None:
                    await cur.execute(
                        "SELECT session_id, event_id, event_sequence, status, attempts, claim_id, "
                        "lease_expires_at, next_attempt_at, last_error, updated_at "
                        "FROM cayu_persisted_event_side_effects "
                        "WHERE session_id = %s AND event_id = %s FOR UPDATE",
                        (expected.session_id, expected.event_id),
                    )
                    expected_row = await cur.fetchone()
                    if expected_row is None or delivery_from_row(expected_row) != expected:
                        await conn.rollback()
                        return None
                if session_id is not None and event_id is not None:
                    exact_filter = (
                        "AND candidate_delivery.session_id = %s "
                        "AND candidate_delivery.event_id = %s"
                    )
                    params.extend([session_id, event_id])
                params.extend([claim_id, float(lease_seconds)])
                await cur.execute(
                    f"""
                        WITH timing AS MATERIALIZED (
                            SELECT clock_timestamp() AS now
                        ), candidate AS (
                            SELECT candidate_delivery.session_id,
                                   candidate_delivery.event_id
                            FROM cayu_persisted_event_side_effects AS candidate_delivery,
                                 timing
                            WHERE (
                                candidate_delivery.status = 'pending'
                                OR (candidate_delivery.status = 'failed' AND (
                                    candidate_delivery.next_attempt_at IS NULL
                                    OR candidate_delivery.next_attempt_at <= timing.now
                                ))
                                OR (candidate_delivery.status = 'leased'
                                    AND candidate_delivery.lease_expires_at <= timing.now)
                            )
                            {exact_filter}
                            AND NOT EXISTS (
                                SELECT 1 FROM cayu_session_closure_progress AS p
                                WHERE p.root_session_id = candidate_delivery.session_id
                                   OR EXISTS (
                                       SELECT 1 FROM jsonb_array_elements(
                                           p.progress_json->'descendants'
                                       ) AS child
                                       WHERE child->>'session_id' = candidate_delivery.session_id
                                   )
                            )
                            ORDER BY candidate_delivery.event_sequence ASC
                            FOR UPDATE OF candidate_delivery SKIP LOCKED
                            LIMIT 1
                        )
                        UPDATE cayu_persisted_event_side_effects AS delivery
                        SET status = 'leased', attempts = delivery.attempts + 1,
                            claim_id = %s,
                            lease_expires_at = timing.now + (%s * INTERVAL '1 second'),
                            next_attempt_at = NULL, last_error = NULL,
                            updated_at = timing.now
                        FROM candidate, timing
                        WHERE delivery.session_id = candidate.session_id
                          AND delivery.event_id = candidate.event_id
                        RETURNING delivery.session_id, delivery.event_id,
                                  delivery.event_sequence, delivery.attempts,
                                  delivery.lease_expires_at
                        """,
                    params,
                )
                row = await cur.fetchone()
                if row is None:
                    await conn.commit()
                    return None
                await cur.execute(
                    "SELECT event FROM cayu_events WHERE session_id = %s AND event_id = %s",
                    (row[0], row[1]),
                )
                event_row = await cur.fetchone()
                if event_row is None:
                    raise RuntimeError("Persisted side-effect delivery lost its source event.")
            await conn.commit()
        except Exception:
            await conn.rollback()
            raise
    return PersistedEventSideEffectClaim(
        session_id=row[0],
        event_id=row[1],
        event_sequence=row[2],
        event=Event(**pg_support._json_obj(event_row[0])),
        attempt=row[3],
        claim_id=claim_id,
        lease_expires_at=pg_support.to_utc(row[4]),
    )


async def get_persisted_event_side_effect_delivery(
    connect: PostgresConnection, *, session_id: str, event_id: str
) -> PersistedEventSideEffectDelivery | None:
    session_id = require_clean_nonblank(session_id, "session_id")
    event_id = require_clean_nonblank(event_id, "event_id")
    async with connect() as conn, conn.cursor() as cur:
        await cur.execute(
            """
                SELECT session_id, event_id, event_sequence, status, attempts,
                       claim_id, lease_expires_at, next_attempt_at, last_error, updated_at
                FROM cayu_persisted_event_side_effects
                WHERE session_id = %s AND event_id = %s
                """,
            (session_id, event_id),
        )
        row = await cur.fetchone()
    return None if row is None else delivery_from_row(row)


async def retire_failed_first_event_delivery(
    connect: PostgresConnection,
    expected: PersistedEventSideEffectDelivery,
    *,
    store_now: Callable[[Any], Awaitable[datetime]],
) -> PersistedEventSideEffectDelivery | None:
    expected = _copy_failed_first_delivery_retirement(expected)
    async with connect() as conn, conn.transaction(), conn.cursor() as cur:
        await cur.execute(
            "SELECT session_id, event_id, event_sequence, status, attempts, claim_id, "
            "lease_expires_at, next_attempt_at, last_error, updated_at "
            "FROM cayu_persisted_event_side_effects WHERE session_id = %s AND event_id = %s FOR UPDATE",
            (expected.session_id, expected.event_id),
        )
        row = await cur.fetchone()
        if row is None or delivery_from_row(row) != expected:
            return None
        retired = expected.model_copy(
            update={
                "status": PersistedEventSideEffectStatus.DEAD_LETTERED,
                "next_attempt_at": None,
                "updated_at": await store_now(cur),
            }
        )
        await cur.execute(
            "UPDATE cayu_persisted_event_side_effects SET status = 'dead_lettered', "
            "next_attempt_at = NULL, updated_at = %s WHERE session_id = %s AND event_id = %s",
            (retired.updated_at, expected.session_id, expected.event_id),
        )
        return retired


async def mark_persisted_event_side_effect_delivered(
    connect: PostgresConnection, claim: PersistedEventSideEffectClaim
) -> PersistedEventSideEffectDelivery:
    claim = PersistedEventSideEffectClaim.model_validate(claim)
    return await _finish_persisted_event_side_effect_claim(
        connect,
        claim,
        status=PersistedEventSideEffectStatus.DELIVERED,
        error=None,
        retry_delay_seconds=None,
    )


async def mark_persisted_event_side_effect_failed(
    connect: PostgresConnection,
    claim: PersistedEventSideEffectClaim,
    *,
    error: str,
    max_attempts: int,
    retry_delay_seconds: float,
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
        connect,
        claim,
        status=(
            PersistedEventSideEffectStatus.DEAD_LETTERED
            if dead_lettered
            else PersistedEventSideEffectStatus.FAILED
        ),
        error=error,
        retry_delay_seconds=(None if dead_lettered else float(retry_delay_seconds)),
    )


async def defer_persisted_event_side_effect(
    connect: PostgresConnection, claim: PersistedEventSideEffectClaim
) -> PersistedEventSideEffectDelivery:
    claim = PersistedEventSideEffectClaim.model_validate(claim)
    return await _finish_persisted_event_side_effect_claim(
        connect,
        claim,
        status=PersistedEventSideEffectStatus.PENDING,
        error=None,
        retry_delay_seconds=None,
        deferred=True,
    )


async def renew_persisted_event_side_effect(
    connect: PostgresConnection,
    claim: PersistedEventSideEffectClaim,
    *,
    lease_seconds: float = 300.0,
) -> PersistedEventSideEffectDelivery:
    claim = PersistedEventSideEffectClaim.model_validate(claim)
    if type(lease_seconds) not in {int, float} or not 0 < lease_seconds <= 86_400:
        raise ValueError("lease_seconds must be positive and at most 86400.")
    async with connect() as conn:
        try:
            async with conn.cursor() as cur:
                # Sample ownership time only after acquiring the row lock:
                # lock contention must not turn a pre-wait timestamp into
                # permission to revive a lease that expired while waiting.
                await cur.execute(
                    "SELECT 1 FROM cayu_persisted_event_side_effects "
                    "WHERE session_id = %s AND event_id = %s FOR UPDATE",
                    (claim.session_id, claim.event_id),
                )
                if await cur.fetchone() is None:
                    raise PersistedEventSideEffectClaimLost(
                        "Persisted event side-effect claim is no longer active."
                    )
                await cur.execute(
                    """
                        WITH timing AS MATERIALIZED (
                            SELECT clock_timestamp() AS now
                        )
                        UPDATE cayu_persisted_event_side_effects
                        SET lease_expires_at = GREATEST(
                                lease_expires_at, timing.now + (%s * INTERVAL '1 second')
                            ), updated_at = timing.now
                        FROM timing
                        WHERE session_id = %s AND event_id = %s AND status = 'leased'
                          AND claim_id = %s AND attempts = %s
                          AND lease_expires_at > timing.now
                        RETURNING session_id, event_id, event_sequence, status,
                                  attempts, claim_id, lease_expires_at, next_attempt_at,
                                  last_error, updated_at
                        """,
                    (
                        float(lease_seconds),
                        claim.session_id,
                        claim.event_id,
                        claim.claim_id,
                        claim.attempt,
                    ),
                )
                row = await cur.fetchone()
                if row is None:
                    raise PersistedEventSideEffectClaimLost(
                        "Persisted event side-effect claim is no longer active."
                    )
            await conn.commit()
        except Exception:
            await conn.rollback()
            raise
    return delivery_from_row(row)


async def _finish_persisted_event_side_effect_claim(
    connect: PostgresConnection,
    claim: PersistedEventSideEffectClaim,
    *,
    status: PersistedEventSideEffectStatus,
    error: str | None,
    retry_delay_seconds: float | None,
    deferred: bool = False,
) -> PersistedEventSideEffectDelivery:
    async with connect() as conn:
        try:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                        WITH timing AS MATERIALIZED (
                            SELECT clock_timestamp() AS now
                        )
                        UPDATE cayu_persisted_event_side_effects
                        SET status = %s, claim_id = NULL, lease_expires_at = NULL,
                            next_attempt_at = CASE
                                WHEN %s::double precision IS NULL THEN NULL
                                ELSE timing.now + (%s * INTERVAL '1 second')
                            END,
                            last_error = %s, updated_at = timing.now,
                            attempts = attempts - %s
                        FROM timing
                        WHERE session_id = %s AND event_id = %s AND status = 'leased'
                          AND claim_id = %s AND attempts = %s
                        RETURNING session_id, event_id, event_sequence, status,
                                  attempts, claim_id, lease_expires_at, next_attempt_at,
                                  last_error, updated_at
                        """,
                    (
                        str(status),
                        retry_delay_seconds,
                        retry_delay_seconds,
                        error,
                        int(deferred),
                        claim.session_id,
                        claim.event_id,
                        claim.claim_id,
                        claim.attempt,
                    ),
                )
                row = await cur.fetchone()
                if row is None:
                    await cur.execute(
                        "SELECT 1 FROM cayu_persisted_event_side_effects "
                        "WHERE session_id = %s AND event_id = %s",
                        (claim.session_id, claim.event_id),
                    )
                    if await cur.fetchone() is None:
                        raise ValueError("Persisted event side-effect delivery was not found.")
                    raise PersistedEventSideEffectClaimLost(
                        "Persisted event side-effect claim is no longer active."
                    )
            await conn.commit()
        except Exception:
            await conn.rollback()
            raise
    return delivery_from_row(row)


async def get_persisted_event_side_effect_health(
    connect: PostgresConnection,
) -> PersistedEventSideEffectHealth:
    async with connect() as conn, conn.cursor() as cur:
        await cur.execute("SELECT clock_timestamp()")
        now = (await cur.fetchone())[0]
        await cur.execute(
            cast("LiteralString", side_effect_health.health_sql("%s::timestamptz")), (now,)
        )
        row = await cur.fetchone()
    return side_effect_health.finish_health(
        dict(zip(side_effect_health.AGGREGATES, row, strict=True)), now
    )


async def query_persisted_event_side_effect_deliveries(
    connect: PostgresConnection, query: PersistedEventSideEffectQuery
) -> PersistedEventSideEffectPage:
    query = PersistedEventSideEffectQuery.model_validate(query)
    side_effect_health.cursor_key(query)
    async with connect() as conn, conn.cursor() as cur:
        await cur.execute("SELECT clock_timestamp()")
        now = (await cur.fetchone())[0]
        sql, params = side_effect_health.page_sql(query, now, "%s")
        sql = sql.replace("SELECT %s AS observed_at", "SELECT %s::timestamptz AS observed_at")
        await cur.execute(cast("LiteralString", sql), params)
        rows = await cur.fetchall()
    return side_effect_health.page(
        [delivery_from_row(row) for row in rows],
        query,
        now,
    )


async def list_persisted_event_side_effect_deliveries(
    connect: PostgresConnection,
    *,
    statuses: set[PersistedEventSideEffectStatus] | None = None,
    claimable_only: bool = False,
    after_sequence: int | None = None,
    limit: int = 100,
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
    if selected_statuses == []:
        return []
    async with connect() as conn, conn.cursor() as cur:
        clauses: list[str] = []
        params: list[Any] = []
        if after_sequence is not None:
            clauses.append("event_sequence > %s")
            params.append(after_sequence)
        if selected_statuses is not None:
            clauses.append("status = ANY(%s)")
            params.append(selected_statuses)
        if claimable_only:
            clauses.append(
                "(status = 'pending' "
                "OR (status = 'failed' AND "
                "(next_attempt_at IS NULL OR next_attempt_at <= clock_timestamp())) "
                "OR (status = 'leased' AND lease_expires_at <= clock_timestamp()))"
            )
        where = "" if not clauses else "WHERE " + " AND ".join(clauses)
        params.append(limit)
        await cur.execute(
            cast(
                "LiteralString",
                f"""
                    SELECT session_id, event_id, event_sequence, status, attempts,
                           claim_id, lease_expires_at, next_attempt_at, last_error, updated_at
                    FROM cayu_persisted_event_side_effects
                    {where}
                    ORDER BY event_sequence ASC
                    LIMIT %s
                    """,
            ),
            params,
        )
        rows = await cur.fetchall()
    return [delivery_from_row(row) for row in rows]


def delivery_from_row(row: tuple[Any, ...]) -> PersistedEventSideEffectDelivery:
    return PersistedEventSideEffectDelivery(
        session_id=row[0],
        event_id=row[1],
        event_sequence=row[2],
        status=PersistedEventSideEffectStatus(row[3]),
        attempts=row[4],
        claim_id=row[5],
        lease_expires_at=pg_support.to_utc_optional(row[6]),
        next_attempt_at=pg_support.to_utc_optional(row[7]),
        last_error=row[8],
        updated_at=pg_support.to_utc(row[9]),
    )

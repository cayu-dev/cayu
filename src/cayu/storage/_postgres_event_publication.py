"""Complete PostgreSQL session event-publication operations.

Each operation owns admission and its native transaction. Shared event writers
are capabilities used within that transaction, including their receipts/outbox.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Iterable, Sequence
from datetime import datetime
from typing import TYPE_CHECKING, Any, Protocol

from cayu._validation import require_durable_clean_nonblank as require_clean_nonblank
from cayu.events import Event
from cayu.sessions.base import (
    BudgetReservationIdentityConflict,
    _assert_session_run_epoch,
    _check_closure_lineage_owner,
    _copy_mcp_manifest_publication,
    _copy_session_event_batch,
    _copy_workflow_step_reservation,
    _current_session_run_epoch,
)
from cayu.sessions.mcp_manifest_history import (
    McpManifestBaseline,
    McpManifestBaselineLoadResult,
    McpManifestPublicationResult,
    _stored_mcp_manifest_baseline,
    _validate_mcp_manifest_history_keys,
    _validate_mcp_manifest_publication_state,
)
from cayu.sessions.records import Session
from cayu.storage import _postgres_event_delivery as event_delivery_ops
from cayu.storage import _postgres_support as pg_support
from cayu.storage._postgres_transcript import PostgresConnection
from cayu.workflows.base import WORKFLOW_ATTEMPT_EVENT_TYPE

if TYPE_CHECKING:
    from psycopg.errors import UniqueViolation


class EventWriter(Protocol):
    def __call__(
        self, cur: Any, session_id: str, events: Sequence[Event], *, expected_run_epoch: int | None
    ) -> Awaitable[None]: ...


class EventRowInserter(Protocol):
    def __call__(
        self,
        cur: Any,
        session_id: str,
        events: Sequence[Event],
        *,
        next_order: int,
        activity_at: datetime,
    ) -> Awaitable[None]: ...


async def claim_budget_reservation_identity(
    connect: PostgresConnection,
    *,
    reservation_id: str,
    publication_session_id: str,
    publication_id: str,
    ensure_ready: Callable[[], Awaitable[None]],
    closure_owners: Callable[[Any, Iterable[str]], Awaitable[tuple[dict[str, Any], ...]]],
    raise_write_conflict: Callable[[Any, str, int], Awaitable[None]],
) -> None:
    reservation_id = require_clean_nonblank(reservation_id, "reservation_id")
    publication_session_id = require_clean_nonblank(
        publication_session_id,
        "publication_session_id",
    )
    publication_id = require_clean_nonblank(publication_id, "publication_id")
    expected_run_epoch = _current_session_run_epoch(publication_session_id)
    await ensure_ready()
    async with connect() as conn:
        try:
            async with conn.cursor() as cur:
                # Serialize the claim with run-epoch takeover and session
                # deletion until this transaction commits.
                await cur.execute(
                    "SELECT run_epoch FROM cayu_sessions WHERE id = %s FOR SHARE",
                    (publication_session_id,),
                )
                session_row = await cur.fetchone()
                if session_row is None:
                    raise KeyError(f"Session not found: {publication_session_id}")
                if expected_run_epoch is not None and session_row[0] != expected_run_epoch:
                    await raise_write_conflict(
                        cur,
                        publication_session_id,
                        expected_run_epoch,
                    )
                await cur.execute(
                    "SELECT 1 FROM cayu_budget_reservation_identities WHERE reservation_id = %s",
                    (reservation_id,),
                )
                if await cur.fetchone() is None:
                    for owner in await closure_owners(cur, (publication_session_id,)):
                        _check_closure_lineage_owner(owner, (publication_session_id,))
                await cur.execute(
                    """
                        INSERT INTO cayu_budget_reservation_identities (
                            reservation_id,
                            publication_session_id,
                            publication_id,
                            published
                        )
                        VALUES (%s, %s, %s, FALSE)
                        ON CONFLICT (reservation_id) DO NOTHING
                        """,
                    (reservation_id, publication_session_id, publication_id),
                )
                if cur.rowcount == 0:
                    await cur.execute(
                        """
                            SELECT publication_session_id, publication_id
                            FROM cayu_budget_reservation_identities
                            WHERE reservation_id = %s
                            """,
                        (reservation_id,),
                    )
                    existing = await cur.fetchone()
                    if existing is None or existing != (
                        publication_session_id,
                        publication_id,
                    ):
                        raise BudgetReservationIdentityConflict(
                            "Budget ledger reused a reservation identity."
                        )
            await conn.commit()
        except Exception:
            await conn.rollback()
            raise


async def append_events(
    connect: PostgresConnection,
    session_id: str,
    events: list[Event],
    *,
    ensure_ready: Callable[[], Awaitable[None]],
    load_session: Callable[[Any, str], Awaitable[Session | None]],
    closure_owners: Callable[[Any, Iterable[str]], Awaitable[tuple[dict[str, Any], ...]]],
    append_events: EventWriter,
    first_existing_event_id: Callable[[str, list[str]], Awaitable[str | None]],
    unique_violation: type[UniqueViolation],
) -> None:
    session_id, copied_events = _copy_session_event_batch(session_id, events)

    await ensure_ready()
    expected_run_epoch = _current_session_run_epoch(session_id)
    async with connect() as conn:
        try:
            async with conn.cursor() as cur:
                loaded = await load_session(cur, session_id)
                if loaded is None:
                    raise KeyError(f"Session not found: {session_id}")
                if not copied_events:
                    _assert_session_run_epoch(session_id, loaded)
                    return
                for owner in await closure_owners(cur, (session_id,)):
                    _check_closure_lineage_owner(owner, (session_id,))
                await append_events(
                    cur,
                    session_id,
                    copied_events,
                    expected_run_epoch=expected_run_epoch,
                )
            await conn.commit()
        except unique_violation as exc:
            await conn.rollback()
            existing = await first_existing_event_id(
                session_id, [event.id for event in copied_events]
            )
            if existing is not None:
                raise ValueError(
                    f"Event already exists for session {session_id}: {existing}"
                ) from exc
            if (
                getattr(exc.diag, "constraint_name", None)
                == "idx_cayu_events_budget_reservation_identity"
            ):
                raise BudgetReservationIdentityConflict(
                    "Budget ledger reused a reservation identity."
                ) from exc
            raise
        except Exception:
            await conn.rollback()
            raise


async def append_tool_effect_conflict(
    connect: PostgresConnection,
    request: object,
    *,
    ensure_ready: Callable[[], Awaitable[None]],
    load_session: Callable[[Any, str], Awaitable[Session | None]],
    store_now: Callable[[Any], Awaitable[datetime]],
    closure_owners: Callable[[Any, Iterable[str]], Awaitable[tuple[dict[str, Any], ...]]],
    lock_closure: Callable[[Any], Awaitable[None]],
    insert_events: EventRowInserter,
) -> Event:
    from cayu.runtime._tool_effect_conflicts import (
        copy_tool_effect_conflict_audit,
        reconcile_tool_effect_conflict_event,
    )

    audit = copy_tool_effect_conflict_audit(request)
    session_id = audit.executing.intent.session_id
    await ensure_ready()
    async with connect() as conn:
        try:
            async with conn.cursor() as cur:
                await lock_closure(cur)
                session = await load_session(cur, session_id)
                if session is None:
                    raise KeyError("Tool effect audit session is unavailable.")
                await cur.execute(
                    "SELECT record FROM cayu_session_operations "
                    "WHERE session_id = %s AND idempotency_key = %s",
                    (session_id, audit.storage_key),
                )
                row = await cur.fetchone()
                current = None if row is None else pg_support._json_obj(row[0])
                event = audit.prepare_event(session, current, now=await store_now(cur))
                await cur.execute(
                    "SELECT event FROM cayu_events WHERE session_id = %s AND event_id = %s",
                    (session_id, event.id),
                )
                existing = await cur.fetchone()
                if existing is not None:
                    event = reconcile_tool_effect_conflict_event(
                        event, Event(**pg_support._json_obj(existing[0]))
                    )
                else:
                    for owner in await closure_owners(cur, (session_id,)):
                        _check_closure_lineage_owner(owner, (session_id,))
                    await cur.execute(
                        "UPDATE cayu_sessions SET event_seq = event_seq + 1 "
                        "WHERE id = %s RETURNING event_seq",
                        (session_id,),
                    )
                    order_row = await cur.fetchone()
                    if order_row is None:
                        raise KeyError("Tool effect audit session is unavailable.")
                    await insert_events(
                        cur,
                        session_id,
                        [event],
                        next_order=order_row[0] - 1,
                        activity_at=event.timestamp,
                    )
            await conn.commit()
            return event
        except BaseException:
            await conn.rollback()
            raise


async def append_workflow_step_started(
    connect: PostgresConnection,
    session_id: str,
    event: Event,
    *,
    workflow_name: str,
    attempt_id: str,
    ensure_ready: Callable[[], Awaitable[None]],
    load_session: Callable[[Any, str], Awaitable[Session | None]],
    store_now: Callable[[Any], Awaitable[datetime]],
    closure_owners: Callable[[Any, Iterable[str]], Awaitable[tuple[dict[str, Any], ...]]],
    lock_closure: Callable[[Any], Awaitable[None]],
    register_event_authorities: Callable[
        [Any, str, list[Event] | tuple[Event, ...]], Awaitable[None]
    ],
    first_existing_event_id: Callable[[str, list[str]], Awaitable[str | None]],
    raise_write_conflict: Callable[[Any, str, int], Awaitable[None]],
    unique_violation: type[UniqueViolation],
) -> bool:
    from cayu.sessions.pending_actions import pending_action_event_storage_values

    session_id, copied_event, workflow_name, attempt_id = _copy_workflow_step_reservation(
        session_id,
        event,
        workflow_name=workflow_name,
        attempt_id=attempt_id,
    )
    await ensure_ready()
    expected_run_epoch = _current_session_run_epoch(session_id)
    async with connect() as conn:
        try:
            async with conn.cursor() as cur:
                await lock_closure(cur)
                if await load_session(cur, session_id) is None:
                    raise KeyError(f"Session not found: {session_id}")
                activity_at = await store_now(cur)
                if expected_run_epoch is None:
                    await cur.execute(
                        """
                            UPDATE cayu_sessions
                            SET event_seq = event_seq + 1, last_activity_at = %s
                            WHERE id = %s
                            RETURNING event_seq
                            """,
                        (activity_at, session_id),
                    )
                else:
                    await cur.execute(
                        """
                            UPDATE cayu_sessions
                            SET event_seq = event_seq + 1, last_activity_at = %s
                            WHERE id = %s AND run_epoch = %s
                            RETURNING event_seq
                            """,
                        (activity_at, session_id, expected_run_epoch),
                    )
                order_row = await cur.fetchone()
                if order_row is None:
                    if expected_run_epoch is not None:
                        await raise_write_conflict(
                            cur,
                            session_id,
                            expected_run_epoch,
                        )
                    raise KeyError(f"Session not found: {session_id}")

                await cur.execute(
                    """
                        SELECT event -> 'payload' ->> 'attempt_id'
                        FROM cayu_events
                        WHERE session_id = %s
                          AND workflow_name = %s
                          AND event_type = %s
                        ORDER BY sequence DESC
                        LIMIT 1
                        """,
                    (session_id, workflow_name, WORKFLOW_ATTEMPT_EVENT_TYPE),
                )
                latest_attempt = await cur.fetchone()
                if latest_attempt is None or latest_attempt[0] != attempt_id:
                    await conn.rollback()
                    return False

                await cur.execute(
                    "SELECT 1 FROM cayu_events WHERE session_id = %s AND event_id = %s",
                    (session_id, copied_event.id),
                )
                if await cur.fetchone() is not None:
                    await conn.rollback()
                    return False

                for owner in await closure_owners(cur, (session_id,)):
                    _check_closure_lineage_owner(owner, (session_id,))
                await register_event_authorities(
                    cur,
                    session_id,
                    [copied_event],
                )
                lookup_key, projection, projection_bytes = pending_action_event_storage_values(
                    copied_event
                )
                await cur.execute(
                    """
                        INSERT INTO cayu_events (
                            session_id, session_order, event_id, interaction_id,
                            event_type, timestamp,
                            agent_name, environment_name, workflow_name, tool_name,
                            payload, event, pending_action_lookup_key,
                            pending_action_projection, pending_action_projection_bytes
                        )
                        VALUES (
                            %s, %s, %s, %s, %s, %s, %s, %s,
                            %s, %s, %s, %s, %s, %s, %s
                        )
                        """,
                    (
                        session_id,
                        order_row[0],
                        copied_event.id,
                        copied_event.interaction_id,
                        str(copied_event.type),
                        pg_support.to_utc(copied_event.timestamp),
                        copied_event.agent_name,
                        copied_event.environment_name,
                        copied_event.workflow_name,
                        copied_event.tool_name,
                        pg_support._dumps(copied_event.payload),
                        pg_support._dumps(copied_event.model_dump(mode="json")),
                        lookup_key,
                        projection,
                        projection_bytes,
                    ),
                )
                await event_delivery_ops.enqueue_persisted_event_side_effects(
                    cur,
                    session_id,
                    [copied_event],
                )
            await conn.commit()
            return True
        except unique_violation as exc:
            await conn.rollback()
            existing = await first_existing_event_id(session_id, [copied_event.id])
            if existing is not None:
                return False
            raise exc
        except Exception:
            await conn.rollback()
            raise


async def load_mcp_manifest_baselines(
    connect: PostgresConnection,
    history_keys: tuple[str, ...],
    *,
    ensure_ready: Callable[[], Awaitable[None]],
) -> McpManifestBaselineLoadResult:
    keys = _validate_mcp_manifest_history_keys(history_keys)
    await ensure_ready()
    async with connect() as conn, conn.cursor() as cur:
        rows = []
        if keys:
            await cur.execute(
                "SELECT history_key, generation, baseline "
                "FROM cayu_mcp_manifest_baselines "
                "WHERE history_key = ANY(%s)",
                (list(keys),),
            )
            rows = await cur.fetchall()
    return McpManifestBaselineLoadResult(
        baselines={row[0]: _stored_mcp_manifest_baseline(row[0], row[1], row[2]) for row in rows},
    )


async def compare_and_publish_mcp_manifest_checks(
    connect: PostgresConnection,
    session_id: str,
    *,
    expected_generations: dict[str, int | None],
    baseline_updates: dict[str, McpManifestBaseline],
    events: list[Event],
    ensure_ready: Callable[[], Awaitable[None]],
    load_session: Callable[[Any, str], Awaitable[Session | None]],
    store_now: Callable[[Any], Awaitable[datetime]],
    closure_owners: Callable[[Any, Iterable[str]], Awaitable[tuple[dict[str, Any], ...]]],
    lock_closure: Callable[[Any], Awaitable[None]],
    register_event_authorities: Callable[
        [Any, str, list[Event] | tuple[Event, ...]], Awaitable[None]
    ],
    first_existing_event_id: Callable[[str, list[str]], Awaitable[str | None]],
    raise_write_conflict: Callable[[Any, str, int], Awaitable[None]],
    unique_violation: type[UniqueViolation],
) -> McpManifestPublicationResult:
    from cayu.sessions.pending_actions import pending_action_event_storage_values

    session_id, expected, updates, copied_events = _copy_mcp_manifest_publication(
        session_id,
        expected_generations=expected_generations,
        baseline_updates=baseline_updates,
        events=events,
    )
    await ensure_ready()
    expected_run_epoch = _current_session_run_epoch(session_id)
    async with connect() as conn:
        try:
            async with conn.cursor() as cur:
                await lock_closure(cur)
                # Missing rows need the same fence as existing rows. Locking
                # the stable keys first also gives multi-toolset batches one
                # deterministic lock order.
                for key in sorted(expected):
                    await cur.execute(
                        "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
                        (key,),
                    )
                await cur.execute(
                    "SELECT history_key, generation, baseline "
                    "FROM cayu_mcp_manifest_baselines "
                    "WHERE history_key = ANY(%s) FOR UPDATE",
                    (list(expected),),
                )
                current = {
                    row[0]: _stored_mcp_manifest_baseline(row[0], row[1], row[2])
                    for row in await cur.fetchall()
                }
                if any(
                    expected_generation
                    != (None if (baseline := current.get(key)) is None else baseline.generation)
                    for key, expected_generation in expected.items()
                ):
                    await conn.rollback()
                    return McpManifestPublicationResult(
                        published=False,
                        baselines=current,
                    )

                _validate_mcp_manifest_publication_state(
                    expected_generations=expected,
                    current_baselines=current,
                    baseline_updates=updates,
                    events=copied_events,
                )
                if await load_session(cur, session_id) is None:
                    raise KeyError(f"Session not found: {session_id}")
                for owner in await closure_owners(cur, (session_id,)):
                    _check_closure_lineage_owner(owner, (session_id,))
                activity_at = await store_now(cur)
                if expected_run_epoch is None:
                    await cur.execute(
                        """
                            UPDATE cayu_sessions
                            SET event_seq = event_seq + %s, last_activity_at = %s
                            WHERE id = %s
                            RETURNING event_seq
                            """,
                        (len(copied_events), activity_at, session_id),
                    )
                else:
                    await cur.execute(
                        """
                            UPDATE cayu_sessions
                            SET event_seq = event_seq + %s, last_activity_at = %s
                            WHERE id = %s AND run_epoch = %s
                            RETURNING event_seq
                            """,
                        (
                            len(copied_events),
                            activity_at,
                            session_id,
                            expected_run_epoch,
                        ),
                    )
                order_row = await cur.fetchone()
                if order_row is None:
                    if expected_run_epoch is not None:
                        await raise_write_conflict(
                            cur,
                            session_id,
                            expected_run_epoch,
                        )
                    raise KeyError(f"Session not found: {session_id}")

                next_order = order_row[0] - len(copied_events)
                await register_event_authorities(
                    cur,
                    session_id,
                    copied_events,
                )
                event_rows = []
                for event in copied_events:
                    next_order += 1
                    lookup_key, projection, projection_bytes = pending_action_event_storage_values(
                        event
                    )
                    event_rows.append(
                        (
                            session_id,
                            next_order,
                            event.id,
                            event.interaction_id,
                            str(event.type),
                            pg_support.to_utc(event.timestamp),
                            event.agent_name,
                            event.environment_name,
                            event.workflow_name,
                            event.tool_name,
                            pg_support._dumps(event.payload),
                            pg_support._dumps(event.model_dump(mode="json")),
                            lookup_key,
                            projection,
                            projection_bytes,
                        )
                    )
                await cur.executemany(
                    """
                        INSERT INTO cayu_events (
                            session_id, session_order, event_id, interaction_id,
                            event_type, timestamp,
                            agent_name, environment_name, workflow_name, tool_name,
                            payload, event, pending_action_lookup_key,
                            pending_action_projection, pending_action_projection_bytes
                        )
                        VALUES (
                            %s, %s, %s, %s, %s, %s, %s, %s,
                            %s, %s, %s, %s, %s, %s, %s
                        )
                        """,
                    event_rows,
                )
                await event_delivery_ops.enqueue_persisted_event_side_effects(
                    cur,
                    session_id,
                    copied_events,
                )
                for key, baseline in updates.items():
                    await cur.execute(
                        """
                            INSERT INTO cayu_mcp_manifest_baselines (
                                history_key, generation, baseline, updated_at
                            )
                            VALUES (%s, %s, %s, %s)
                            ON CONFLICT (history_key) DO UPDATE SET
                                generation = EXCLUDED.generation,
                                baseline = EXCLUDED.baseline,
                                updated_at = EXCLUDED.updated_at
                            """,
                        (
                            key,
                            baseline.generation,
                            pg_support._dumps(baseline.model_dump(mode="json")),
                            activity_at,
                        ),
                    )
                    current[key] = baseline.model_copy(deep=True)
            await conn.commit()
            return McpManifestPublicationResult(
                published=True,
                baselines=current,
            )
        except unique_violation as exc:
            await conn.rollback()
            existing = await first_existing_event_id(
                session_id,
                [event.id for event in copied_events],
            )
            if existing is not None:
                raise ValueError(
                    f"Event already exists for session {session_id}: {existing}"
                ) from exc
            raise
        except Exception:
            await conn.rollback()
            raise

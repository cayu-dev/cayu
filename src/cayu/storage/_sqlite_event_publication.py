"""Complete SQLite session event-publication operations.

Each operation owns admission and its native transaction. Shared event writers
are capabilities used within that transaction, including their receipts/outbox.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable, Sequence
from datetime import datetime
from typing import Protocol

from cayu._validation import require_durable_clean_nonblank as require_clean_nonblank
from cayu.events import Event
from cayu.sessions.base import (
    BudgetReservationIdentityConflict,
    _assert_session_run_epoch,
    _check_closure_lineage_owner,
    _copy_mcp_manifest_publication,
    _copy_workflow_step_reservation,
    _current_session_run_epoch,
)
from cayu.sessions.event_delivery import _copy_session_event_batch
from cayu.sessions.mcp_manifest_history import (
    McpManifestBaseline,
    McpManifestBaselineLoadResult,
    McpManifestPublicationResult,
    _stored_mcp_manifest_baseline_json,
    _validate_mcp_manifest_history_keys,
    _validate_mcp_manifest_publication_state,
)
from cayu.sessions.records import Session
from cayu.storage import _sqlite_event_delivery as event_delivery_ops
from cayu.storage import _sqlite_records as sqlite_records
from cayu.storage._sqlite_connection import SQLiteOperationRunner
from cayu.storage._sqlite_transcript import ClosureOwners
from cayu.workflows.base import WORKFLOW_ATTEMPT_EVENT_TYPE


class EventWriter(Protocol):
    def __call__(
        self,
        connection: sqlite3.Connection,
        session_id: str,
        events: Sequence[Event],
        *,
        activity_at: datetime,
    ) -> None: ...


class IdentityClaimer(Protocol):
    def __call__(
        self,
        connection: sqlite3.Connection,
        *,
        reservation_id: str,
        publication_session_id: str,
        publication_id: str,
    ) -> None: ...


async def claim_budget_reservation_identity(
    run_write: SQLiteOperationRunner,
    *,
    reservation_id: str,
    publication_session_id: str,
    publication_id: str,
    closure_owners: ClosureOwners,
    raise_write_conflict: Callable[[sqlite3.Connection, str, int], None],
    claim_identity: IdentityClaimer,
) -> None:
    reservation_id = require_clean_nonblank(reservation_id, "reservation_id")
    publication_session_id = require_clean_nonblank(
        publication_session_id,
        "publication_session_id",
    )
    publication_id = require_clean_nonblank(publication_id, "publication_id")
    expected_run_epoch = _current_session_run_epoch(publication_session_id)

    def statement(connection: sqlite3.Connection) -> None:
        with connection:
            # The conditional no-op acquires SQLite's writer lock while it
            # validates the session epoch, so a takeover cannot interleave
            # between this check and the registry claim below.
            if expected_run_epoch is None:
                cursor = connection.execute(
                    "UPDATE cayu_sessions SET run_epoch = run_epoch WHERE id = ?",
                    (publication_session_id,),
                )
            else:
                cursor = connection.execute(
                    "UPDATE cayu_sessions SET run_epoch = run_epoch WHERE id = ? AND run_epoch = ?",
                    (publication_session_id, expected_run_epoch),
                )
            if cursor.rowcount != 1:
                if expected_run_epoch is not None:
                    raise_write_conflict(
                        connection,
                        publication_session_id,
                        expected_run_epoch,
                    )
                raise KeyError(f"Session not found: {publication_session_id}")
            existing = connection.execute(
                "SELECT 1 FROM cayu_budget_reservation_identities WHERE reservation_id = ?",
                (reservation_id,),
            ).fetchone()
            if existing is None:
                for owner in closure_owners((publication_session_id,), connection=connection):
                    _check_closure_lineage_owner(owner, (publication_session_id,))
            claim_identity(
                connection,
                reservation_id=reservation_id,
                publication_session_id=publication_session_id,
                publication_id=publication_id,
            )

    await run_write(statement)


async def append_events(
    run_write: SQLiteOperationRunner,
    session_id: str,
    events: list[Event],
    *,
    store_now: Callable[[], datetime],
    closure_owners: ClosureOwners,
    append_events: EventWriter,
    first_existing_event_id: Callable[[sqlite3.Connection, str, list[str]], str | None],
) -> None:
    session_id, copied_events = _copy_session_event_batch(session_id, events)

    def statement(connection: sqlite3.Connection) -> None:
        if not copied_events:
            if not sqlite_records.session_exists(connection, session_id):
                raise KeyError(f"Session not found: {session_id}")
            return

        try:
            connection.execute("BEGIN IMMEDIATE")
            if not sqlite_records.session_exists(connection, session_id):
                raise KeyError(f"Session not found: {session_id}")
            activity_at = store_now()
            for owner in closure_owners((session_id,)):
                _check_closure_lineage_owner(owner, (session_id,))
            append_events(
                connection,
                session_id,
                copied_events,
                activity_at=activity_at,
            )
            connection.commit()
        except sqlite3.IntegrityError as exc:
            connection.rollback()
            existing_event_id = first_existing_event_id(
                connection,
                session_id,
                [event.id for event in copied_events],
            )
            if existing_event_id is not None:
                raise ValueError(
                    f"Event already exists for session {session_id}: {existing_event_id}"
                ) from exc
            if "idx_cayu_events_budget_reservation_identity" in str(exc):
                raise BudgetReservationIdentityConflict(
                    "Budget ledger reused a reservation identity."
                ) from exc
            raise
        except BaseException:
            connection.rollback()
            raise

    await run_write(statement)


async def append_tool_effect_conflict(
    run_write: SQLiteOperationRunner,
    request: object,
    *,
    load_session: Callable[[str], Session | None],
    store_now: Callable[[], datetime],
    closure_owners: ClosureOwners,
    insert_events: EventWriter,
) -> Event:
    from cayu.runtime._tool_effect_conflicts import (
        copy_tool_effect_conflict_audit,
        reconcile_tool_effect_conflict_event,
    )

    audit = copy_tool_effect_conflict_audit(request)
    session_id = audit.executing.intent.session_id

    def statement(connection: sqlite3.Connection) -> Event:
        try:
            connection.execute("BEGIN IMMEDIATE")
            session = load_session(session_id)
            if session is None:
                raise KeyError("Tool effect audit session is unavailable.")
            row = connection.execute(
                "SELECT record_json FROM cayu_session_operations "
                "WHERE session_id = ? AND idempotency_key = ?",
                (session_id, audit.storage_key),
            ).fetchone()
            current = None if row is None else json.loads(row["record_json"])
            event = audit.prepare_event(session, current, now=store_now())
            existing = connection.execute(
                "SELECT * FROM cayu_events WHERE session_id = ? AND event_id = ?",
                (session_id, event.id),
            ).fetchone()
            if existing is not None:
                event = reconcile_tool_effect_conflict_event(
                    event, sqlite_records.event_from_row(existing)
                )
            else:
                for owner in closure_owners((session_id,), connection=connection):
                    _check_closure_lineage_owner(owner, (session_id,))
                # Evidence authority was established above; do not touch the
                # current run's liveness or weaken the ordinary append fence.
                insert_events(connection, session_id, [event], activity_at=event.timestamp)
            connection.commit()
            return event
        except BaseException:
            connection.rollback()
            raise

    return await run_write(statement)


async def append_workflow_step_started(
    run_write: SQLiteOperationRunner,
    session_id: str,
    event: Event,
    *,
    workflow_name: str,
    attempt_id: str,
    store_now: Callable[[], datetime],
    closure_owners: ClosureOwners,
    first_existing_event_id: Callable[[sqlite3.Connection, str, list[str]], str | None],
    touch_activity: Callable[[sqlite3.Connection, str, datetime], None],
) -> bool:
    from cayu.sessions.pending_actions import pending_action_event_storage_values

    session_id, copied_event, workflow_name, attempt_id = _copy_workflow_step_reservation(
        session_id,
        event,
        workflow_name=workflow_name,
        attempt_id=attempt_id,
    )

    def statement(connection: sqlite3.Connection) -> bool:
        try:
            connection.execute("BEGIN IMMEDIATE")
            if not sqlite_records.session_exists(connection, session_id):
                raise KeyError(f"Session not found: {session_id}")
            row = connection.execute(
                """
                    SELECT json_extract(payload_json, '$.attempt_id')
                    FROM cayu_events
                    WHERE session_id = ?
                      AND workflow_name = ?
                      AND event_type = ?
                    ORDER BY sequence DESC
                    LIMIT 1
                    """,
                (session_id, workflow_name, WORKFLOW_ATTEMPT_EVENT_TYPE),
            ).fetchone()
            if row is None or row[0] != attempt_id:
                connection.rollback()
                return False
            if first_existing_event_id(connection, session_id, [copied_event.id]) is not None:
                connection.rollback()
                return False

            for owner in closure_owners((session_id,), connection=connection):
                _check_closure_lineage_owner(owner, (session_id,))
            touch_activity(connection, session_id, store_now())
            lookup_key, projection, projection_bytes = pending_action_event_storage_values(
                copied_event
            )
            connection.execute(
                """
                    INSERT INTO cayu_events (
                        session_id,
                        event_id,
                        interaction_id,
                        event_type,
                        timestamp,
                        agent_name,
                        environment_name,
                        workflow_name,
                        tool_name,
                        payload_json,
                        pending_action_lookup_key,
                        pending_action_projection_json,
                        pending_action_projection_bytes
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                (
                    session_id,
                    copied_event.id,
                    copied_event.interaction_id,
                    str(copied_event.type),
                    sqlite_records.format_datetime(copied_event.timestamp),
                    copied_event.agent_name,
                    copied_event.environment_name,
                    copied_event.workflow_name,
                    copied_event.tool_name,
                    sqlite_records.json_dumps(copied_event.payload),
                    lookup_key,
                    projection,
                    projection_bytes,
                ),
            )
            event_delivery_ops.enqueue_persisted_event_side_effects(
                connection,
                session_id,
                [copied_event],
            )
            connection.commit()
            return True
        except sqlite3.IntegrityError as exc:
            connection.rollback()
            existing_event_id = first_existing_event_id(
                connection,
                session_id,
                [copied_event.id],
            )
            if existing_event_id is not None:
                return False
            raise exc
        except Exception:
            connection.rollback()
            raise

    return await run_write(statement)


async def load_mcp_manifest_baselines(
    run_read: SQLiteOperationRunner, history_keys: tuple[str, ...]
) -> McpManifestBaselineLoadResult:
    keys = _validate_mcp_manifest_history_keys(history_keys)

    def query(connection: sqlite3.Connection) -> McpManifestBaselineLoadResult:
        result: dict[str, McpManifestBaseline] = {}
        for key in keys:
            row = connection.execute(
                "SELECT generation, baseline_json FROM cayu_mcp_manifest_baselines "
                "WHERE history_key = ?",
                (key,),
            ).fetchone()
            if row is not None:
                result[key] = _stored_mcp_manifest_baseline_json(
                    key,
                    row["generation"],
                    row["baseline_json"],
                )
        return McpManifestBaselineLoadResult(baselines=result)

    return await run_read(query)


async def compare_and_publish_mcp_manifest_checks(
    run_write: SQLiteOperationRunner,
    session_id: str,
    *,
    expected_generations: dict[str, int | None],
    baseline_updates: dict[str, McpManifestBaseline],
    events: list[Event],
    store_now: Callable[[], datetime],
    closure_owners: ClosureOwners,
    first_existing_event_id: Callable[[sqlite3.Connection, str, list[str]], str | None],
    touch_activity: Callable[[sqlite3.Connection, str, datetime], None],
) -> McpManifestPublicationResult:
    from cayu.sessions.pending_actions import pending_action_event_storage_values

    session_id, expected, updates, copied_events = _copy_mcp_manifest_publication(
        session_id,
        expected_generations=expected_generations,
        baseline_updates=baseline_updates,
        events=events,
    )

    def statement(connection: sqlite3.Connection) -> McpManifestPublicationResult:
        try:
            connection.execute("BEGIN IMMEDIATE")
            session = sqlite_records.load_session(connection, session_id)
            if session is None:
                raise KeyError(f"Session not found: {session_id}")
            _assert_session_run_epoch(session_id, session)
            current: dict[str, McpManifestBaseline] = {}
            for key in expected:
                row = connection.execute(
                    "SELECT generation, baseline_json "
                    "FROM cayu_mcp_manifest_baselines "
                    "WHERE history_key = ?",
                    (key,),
                ).fetchone()
                if row is not None:
                    current[key] = _stored_mcp_manifest_baseline_json(
                        key,
                        row["generation"],
                        row["baseline_json"],
                    )
            if any(
                expected_generation
                != (None if (baseline := current.get(key)) is None else baseline.generation)
                for key, expected_generation in expected.items()
            ):
                connection.rollback()
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
            for owner in closure_owners((session_id,), connection=connection):
                _check_closure_lineage_owner(owner, (session_id,))
            touch_activity(connection, session_id, store_now())
            event_rows = []
            for event in copied_events:
                lookup_key, projection, projection_bytes = pending_action_event_storage_values(
                    event
                )
                event_rows.append(
                    (
                        session_id,
                        event.id,
                        event.interaction_id,
                        str(event.type),
                        sqlite_records.format_datetime(event.timestamp),
                        event.agent_name,
                        event.environment_name,
                        event.workflow_name,
                        event.tool_name,
                        sqlite_records.json_dumps(event.payload),
                        lookup_key,
                        projection,
                        projection_bytes,
                    )
                )
            connection.executemany(
                """
                    INSERT INTO cayu_events (
                        session_id, event_id, interaction_id, event_type, timestamp, agent_name,
                        environment_name, workflow_name, tool_name, payload_json,
                        pending_action_lookup_key, pending_action_projection_json,
                        pending_action_projection_bytes
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                event_rows,
            )
            event_delivery_ops.enqueue_persisted_event_side_effects(
                connection,
                session_id,
                copied_events,
            )
            updated_at = sqlite_records.format_datetime(store_now())
            for key, baseline in updates.items():
                connection.execute(
                    """
                        INSERT INTO cayu_mcp_manifest_baselines (
                            history_key, generation, baseline_json, updated_at
                        )
                        VALUES (?, ?, ?, ?)
                        ON CONFLICT(history_key) DO UPDATE SET
                            generation = excluded.generation,
                            baseline_json = excluded.baseline_json,
                            updated_at = excluded.updated_at
                        """,
                    (
                        key,
                        baseline.generation,
                        sqlite_records.json_dumps(baseline.model_dump(mode="json")),
                        updated_at,
                    ),
                )
                current[key] = baseline.model_copy(deep=True)
            connection.commit()
            return McpManifestPublicationResult(
                published=True,
                baselines=current,
            )
        except sqlite3.IntegrityError as exc:
            connection.rollback()
            existing_event_id = first_existing_event_id(
                connection,
                session_id,
                [event.id for event in copied_events],
            )
            if existing_event_id is not None:
                raise ValueError(
                    f"Event already exists for session {session_id}: {existing_event_id}"
                ) from exc
            raise
        except Exception:
            connection.rollback()
            raise

    return await run_write(statement)

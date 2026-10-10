"""Complete SQLite targeted-tool grant persistence operations.

Each operation owns authorization and its native transaction. Shared decoders,
closure checks and event writers are explicit capabilities within that boundary.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Iterable, Mapping
from datetime import UTC, datetime
from typing import Protocol

from cayu._validation import require_durable_clean_nonblank as require_clean_nonblank
from cayu.events import Event, EventType
from cayu.runtime.public_authority import PublicAuthorityAliasCodec, parse_public_authority_alias
from cayu.sessions.base import (
    SessionRunFenced,
    SessionStatusConflict,
    _check_closure_lineage_owner,
    _public_authority_alias_store_key,
)
from cayu.sessions.interactions import (
    INTERACTION_LIFECYCLE_EVENT_TYPES,
    INTERACTION_TERMINAL_EVENT_TYPES,
)
from cayu.sessions.invocation import SessionInvocation
from cayu.sessions.records import SessionStatus
from cayu.storage import _sqlite_records as sqlite_records
from cayu.storage._sqlite_connection import SQLiteOperationRunner
from cayu.storage._sqlite_event_publication import EventWriter
from cayu.storage._sqlite_transcript import ClosureOwners
from cayu.tools.grants import (
    TARGETED_TOOL_GRANT_INSPECTION_MAX_RECORDS,
    TARGETED_TOOL_GRANT_MAX_REQUESTS,
    TARGETED_TOOL_REFERENCE_FIELD_NAME,
    TargetedToolGrantIssueOutcome,
    TargetedToolGrantIssueResult,
    TargetedToolGrantReconstructionResult,
    TargetedToolGrantRecord,
    TargetedToolGrantStateSnapshot,
    TargetedToolUseBinding,
    TargetedToolUseDisposition,
    TargetedToolUseRejectionReason,
    TargetedToolUseRequest,
    TargetedToolUseResult,
    copy_targeted_tool_grant_record,
    targeted_tool_grant_event,
    targeted_tool_grant_reconstruction_rejection_reason,
    targeted_tool_grant_with_active_reference,
    targeted_tool_unresolved_rejection_event,
    targeted_tool_use_binding,
    targeted_tool_use_rejection_event,
    targeted_tool_use_rejection_reason,
    targeted_tool_use_scope_rejection_reason,
    validate_targeted_tool_grant_batch_evidence,
    validate_targeted_tool_grant_issuance_evidence,
    validate_targeted_tool_grant_lifecycle_event,
    validate_targeted_tool_grant_reference,
    validate_targeted_tool_grant_revocation_evidence,
    validate_targeted_tool_grant_revocation_reason,
    validate_targeted_tool_unresolved_rejection_evidence,
    validate_targeted_tool_use_rejection_evidence,
)


class EventOnceWriter(Protocol):
    def __call__(
        self, connection: sqlite3.Connection, event: Event, *, activity_at: datetime
    ) -> Event: ...


async def issue_targeted_tool_grants(
    run_write: SQLiteOperationRunner,
    session_id: str,
    *,
    expected_run_epoch: int,
    records: tuple[TargetedToolGrantRecord, ...],
    events: tuple[Event, ...],
    store_now: Callable[[], datetime],
    get_codec: Callable[[], PublicAuthorityAliasCodec | None],
    decode_grant: Callable[[sqlite3.Row], TargetedToolGrantRecord],
    validate_use_counts: Callable[[sqlite3.Connection, Iterable[TargetedToolGrantRecord]], None],
    append_events: EventWriter,
    append_event_once: EventOnceWriter,
) -> TargetedToolGrantIssueResult:
    session_id = require_clean_nonblank(session_id, "session_id")
    if type(expected_run_epoch) is not int or expected_run_epoch < 0:
        raise ValueError("expected_run_epoch must be a non-negative integer.")
    if type(records) is not tuple or type(events) is not tuple:
        raise TypeError("records and events must be tuples.")
    if len(records) > TARGETED_TOOL_GRANT_MAX_REQUESTS:
        raise ValueError("Targeted grant issuance exceeds the bounded request count.")
    copied_records = tuple(copy_targeted_tool_grant_record(record) for record in records)
    copied_events = tuple(Event.model_validate(event.model_dump(mode="python")) for event in events)
    if len(copied_records) != len(copied_events):
        raise ValueError("Each targeted grant record requires one issuance event.")
    if len({record.request_id for record in copied_records}) != len(copied_records):
        raise ValueError("Targeted grant records must have unique request identities.")
    if len({record.tool_id for record in copied_records}) != len(copied_records):
        raise ValueError("Targeted grant records must have unique tool identities.")
    interaction_ids = {record.interaction_id for record in copied_records}
    if len(interaction_ids) > 1:
        raise ValueError("Targeted grant records must share one interaction scope.")
    codec = get_codec()
    if codec is None:
        raise RuntimeError("Targeted grants require a public authority alias codec.")
    for record, event in zip(copied_records, copied_events, strict=True):
        if record.session_id != session_id:
            raise ValueError("Targeted grant scope is inconsistent.")
        validate_targeted_tool_grant_reference(record, codec)
        validate_targeted_tool_grant_issuance_evidence(record, event)

    def statement(
        connection: sqlite3.Connection,
    ) -> tuple[
        tuple[TargetedToolGrantRecord, ...],
        tuple[TargetedToolGrantIssueOutcome, ...],
        tuple[Event, ...],
    ]:
        with connection:
            connection.execute("BEGIN IMMEDIATE")
            session_row = connection.execute(
                "SELECT agent_name, environment_name, status, run_epoch, invocation_json "
                "FROM cayu_sessions WHERE id = ?",
                (session_id,),
            ).fetchone()
            if session_row is None:
                raise KeyError(f"Session not found: {session_id}")
            if int(session_row["run_epoch"]) != expected_run_epoch:
                raise SessionRunFenced(
                    f"Session source run epoch is stale: expected {expected_run_epoch}, "
                    f"current {session_row['run_epoch']}."
                )
            if str(session_row["status"]) != str(SessionStatus.RUNNING):
                raise SessionStatusConflict("Targeted grants require a running session.")
            if interaction_ids:
                lifecycle_placeholders = ", ".join("?" for _ in INTERACTION_LIFECYCLE_EVENT_TYPES)
                latest_interaction = connection.execute(
                    "SELECT interaction_id, event_type FROM cayu_events "
                    "WHERE session_id = ? "
                    f"AND event_type IN ({lifecycle_placeholders}) "
                    "ORDER BY sequence DESC LIMIT 1",
                    (
                        session_id,
                        *(str(value) for value in INTERACTION_LIFECYCLE_EVENT_TYPES),
                    ),
                ).fetchone()
                if (
                    latest_interaction is None
                    or latest_interaction["interaction_id"] != next(iter(interaction_ids))
                    or EventType(str(latest_interaction["event_type"]))
                    in INTERACTION_TERMINAL_EVENT_TYPES
                ):
                    raise ValueError("Targeted grants require the current open interaction.")
                interaction_started_row = connection.execute(
                    "SELECT * FROM cayu_events WHERE session_id = ? "
                    "AND interaction_id = ? AND event_type = ? "
                    "ORDER BY sequence ASC LIMIT 1",
                    (
                        session_id,
                        next(iter(interaction_ids)),
                        str(EventType.INTERACTION_STARTED),
                    ),
                ).fetchone()
                if interaction_started_row is None:
                    raise RuntimeError("Targeted grant issuance lost interaction admission.")
                validate_targeted_tool_grant_batch_evidence(
                    copied_records,
                    sqlite_records.event_from_row(interaction_started_row),
                )
            invocation = SessionInvocation.model_validate_json(session_row["invocation_json"])
            resolved: list[TargetedToolGrantRecord] = []
            outcomes: list[TargetedToolGrantIssueOutcome] = []
            resolved_events: list[Event] = []
            new_events: list[Event] = []
            for record, event in zip(copied_records, copied_events, strict=True):
                if (
                    record.session_id != session_id
                    or record.agent_name != session_row["agent_name"]
                    or record.environment_name != session_row["environment_name"]
                    or record.principal != invocation.origin.subject
                    or record.tenant != invocation.origin.tenant
                ):
                    raise ValueError("Targeted grant scope is inconsistent.")
                existing_row = connection.execute(
                    "SELECT * FROM cayu_targeted_tool_grants "
                    "WHERE session_id = ? AND interaction_id = ? "
                    "AND (request_id = ? OR tool_id = ?) LIMIT 2",
                    (
                        session_id,
                        record.interaction_id,
                        record.request_id,
                        record.tool_id,
                    ),
                ).fetchone()
                if existing_row is not None:
                    existing = decode_grant(existing_row)
                    validate_use_counts(connection, (existing,))
                    if existing.request_id != record.request_id:
                        raise ValueError(
                            "Targeted grant tool identity conflicts with durable authority."
                        )
                    if existing.grant_id != record.grant_id:
                        raise ValueError(
                            "Targeted grant request identity conflicts with durable authority."
                        )
                    resolved.append(targeted_tool_grant_with_active_reference(existing, codec))
                    outcomes.append(TargetedToolGrantIssueOutcome.REUSED)
                    issued_row = connection.execute(
                        "SELECT * FROM cayu_events WHERE session_id = ? AND event_id = ?",
                        (session_id, event.id),
                    ).fetchone()
                    if issued_row is None:
                        raise RuntimeError("Targeted grant lost its durable issuance evidence.")
                    validate_targeted_tool_grant_issuance_evidence(
                        existing,
                        sqlite_records.event_from_row(issued_row),
                    )
                    reused_event = targeted_tool_grant_event(
                        existing,
                        event_type=EventType.TARGETED_TOOL_GRANT_REUSED,
                        timestamp=event.timestamp,
                        outcome=TargetedToolGrantIssueOutcome.REUSED.value,
                        event_id_suffix="reused",
                    )
                    resolved_events.append(
                        append_event_once(
                            connection,
                            reused_event,
                            activity_at=reused_event.timestamp,
                        )
                    )
                    continue
                collision = connection.execute(
                    "SELECT 1 FROM cayu_targeted_tool_grants WHERE grant_id = ?",
                    (record.grant_id,),
                ).fetchone()
                if collision is not None:
                    raise ValueError("Targeted grant identity collides with authority.")
                for public_alias in codec.aliases(
                    record.grant_id,
                    field_name=TARGETED_TOOL_REFERENCE_FIELD_NAME,
                    session_id=session_id,
                ):
                    field_name, scope_key, public_alias = _public_authority_alias_store_key(
                        public_alias,
                        field_name=TARGETED_TOOL_REFERENCE_FIELD_NAME,
                        private_value=record.grant_id,
                        scope_session_id=session_id,
                    )
                    connection.execute(
                        "INSERT INTO cayu_public_authority_aliases "
                        "(field_name, scope_session_id, public_alias, private_value) "
                        "VALUES (?, ?, ?, ?) "
                        "ON CONFLICT(field_name, scope_session_id, public_alias) DO NOTHING",
                        (field_name, scope_key, public_alias, record.grant_id),
                    )
                    stored_alias = connection.execute(
                        "SELECT private_value FROM cayu_public_authority_aliases "
                        "WHERE field_name = ? AND scope_session_id = ? AND public_alias = ?",
                        (field_name, scope_key, public_alias),
                    ).fetchone()
                    if stored_alias is None or stored_alias["private_value"] != record.grant_id:
                        raise ValueError("Targeted tool reference collides with authority.")
                connection.execute(
                    """
                        INSERT INTO cayu_targeted_tool_grants (
                            grant_id, session_id, interaction_id, request_id, tool_ref,
                            generation_id, tool_id, tool_name, catalogue_revision,
                            descriptor_version, issued_at, expires_at, max_calls,
                            used_calls, revoked_at, record_json
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                    (
                        record.grant_id,
                        record.session_id,
                        record.interaction_id,
                        record.request_id,
                        record.tool_ref,
                        record.generation_id,
                        record.tool_id,
                        record.tool_name,
                        record.catalogue_revision,
                        record.descriptor_version,
                        sqlite_records.format_datetime(record.issued_at),
                        sqlite_records.format_datetime(record.expires_at),
                        record.max_calls,
                        record.used_calls,
                        None,
                        sqlite_records.json_dumps(record.model_dump(mode="json")),
                    ),
                )
                resolved.append(record)
                outcomes.append(TargetedToolGrantIssueOutcome.ISSUED)
                resolved_events.append(event)
                new_events.append(event)
            if interaction_ids:
                interaction_count = connection.execute(
                    "SELECT COUNT(*) FROM cayu_targeted_tool_grants "
                    "WHERE session_id = ? AND interaction_id = ?",
                    (session_id, next(iter(interaction_ids))),
                ).fetchone()[0]
                if interaction_count > TARGETED_TOOL_GRANT_MAX_REQUESTS:
                    raise ValueError("Targeted grant interaction exceeds its bounded count.")
            append_events(
                connection,
                session_id,
                new_events,
                activity_at=store_now(),
            )
            return tuple(resolved), tuple(outcomes), tuple(resolved_events)

    resolved_records, outcomes, resolved_events = await run_write(statement)
    return TargetedToolGrantIssueResult(
        records=resolved_records,
        outcomes=outcomes,
        events=resolved_events,
    )


async def list_targeted_tool_grants(
    run_read: SQLiteOperationRunner,
    session_id: str,
    *,
    interaction_id: str | None = None,
    limit: int = TARGETED_TOOL_GRANT_INSPECTION_MAX_RECORDS,
    get_codec: Callable[[], PublicAuthorityAliasCodec | None],
    decode_grant: Callable[[sqlite3.Row], TargetedToolGrantRecord],
    validate_use_counts: Callable[[sqlite3.Connection, Iterable[TargetedToolGrantRecord]], None],
) -> tuple[TargetedToolGrantRecord, ...]:
    session_id = require_clean_nonblank(session_id, "session_id")
    if interaction_id is not None:
        interaction_id = require_clean_nonblank(interaction_id, "interaction_id")
    if type(limit) is not int or not 1 <= limit <= TARGETED_TOOL_GRANT_INSPECTION_MAX_RECORDS:
        raise ValueError(
            f"limit must be between 1 and {TARGETED_TOOL_GRANT_INSPECTION_MAX_RECORDS}."
        )

    def query(connection: sqlite3.Connection) -> tuple[TargetedToolGrantRecord, ...]:
        with connection:
            connection.execute("BEGIN")
            if not sqlite_records.session_exists(connection, session_id):
                raise KeyError(f"Session not found: {session_id}")
            if interaction_id is None:
                rows = connection.execute(
                    "SELECT * FROM cayu_targeted_tool_grants "
                    "WHERE session_id = ? ORDER BY issued_at, grant_id LIMIT ?",
                    (session_id, limit + 1),
                ).fetchall()
            else:
                rows = connection.execute(
                    "SELECT * FROM cayu_targeted_tool_grants "
                    "WHERE session_id = ? AND interaction_id = ? "
                    "ORDER BY issued_at, grant_id LIMIT ?",
                    (session_id, interaction_id, limit + 1),
                ).fetchall()
            if len(rows) > limit:
                raise ValueError("Targeted grant inspection exceeds its bounded result limit.")
            if not rows:
                return ()
            records = tuple(decode_grant(row) for row in rows)
            validate_use_counts(connection, records)
            codec = get_codec()
            if codec is None:
                raise RuntimeError("Targeted grants require a public authority alias codec.")
            return tuple(
                targeted_tool_grant_with_active_reference(
                    record,
                    codec,
                )
                for record in records
            )

    return await run_read(query)


async def load_targeted_tool_grant_state(
    run_read: SQLiteOperationRunner,
    session_id: str,
    *,
    get_codec: Callable[[], PublicAuthorityAliasCodec | None],
    decode_grant: Callable[[sqlite3.Row], TargetedToolGrantRecord],
    decode_use: Callable[[sqlite3.Row], TargetedToolUseBinding],
) -> TargetedToolGrantStateSnapshot:
    session_id = require_clean_nonblank(session_id, "session_id")

    def query(connection: sqlite3.Connection) -> TargetedToolGrantStateSnapshot:
        with connection:
            connection.execute("BEGIN")
            if not sqlite_records.session_exists(connection, session_id):
                raise KeyError(f"Session not found: {session_id}")
            grant_rows = connection.execute(
                "SELECT * FROM cayu_targeted_tool_grants "
                "WHERE session_id = ? ORDER BY issued_at, grant_id",
                (session_id,),
            ).fetchall()
            use_rows = connection.execute(
                "SELECT * FROM cayu_targeted_tool_grant_uses "
                "WHERE session_id = ? ORDER BY bound_at, use_id",
                (session_id,),
            ).fetchall()
            if not grant_rows:
                if use_rows:
                    raise ValueError("Targeted grant uses exist without grant records.")
                return TargetedToolGrantStateSnapshot()
            codec = get_codec()
            if codec is None:
                raise RuntimeError("Targeted grants require a public authority alias codec.")
            records: list[TargetedToolGrantRecord] = []
            for row in grant_rows:
                record = decode_grant(row)
                records.append(targeted_tool_grant_with_active_reference(record, codec))
            return TargetedToolGrantStateSnapshot(
                records=tuple(records),
                uses=tuple(decode_use(row) for row in use_rows),
            )

    return await run_read(query)


async def bind_targeted_tool_grant_use(
    run_write: SQLiteOperationRunner,
    request: TargetedToolUseRequest,
    *,
    observed_at: datetime,
    get_codec: Callable[[], PublicAuthorityAliasCodec | None],
    decode_grant: Callable[[sqlite3.Row], TargetedToolGrantRecord],
    decode_use: Callable[[sqlite3.Row], TargetedToolUseBinding],
    validate_use_counts: Callable[[sqlite3.Connection, Iterable[TargetedToolGrantRecord]], None],
    append_events: EventWriter,
    append_event_once: EventOnceWriter,
) -> TargetedToolUseResult:
    if type(request) is not TargetedToolUseRequest:
        raise TypeError("request must be a TargetedToolUseRequest.")
    request = TargetedToolUseRequest.model_validate(request.model_dump(mode="python"))
    if observed_at.tzinfo is None or observed_at.utcoffset() is None:
        raise ValueError("observed_at must be timezone-aware.")
    observed_at = observed_at.astimezone(UTC)
    codec = get_codec()
    if codec is None:
        raise RuntimeError("Targeted grants require a public authority alias codec.")

    def statement(
        connection: sqlite3.Connection,
    ) -> tuple[TargetedToolUseResult, Event | None]:
        with connection:
            connection.execute("BEGIN IMMEDIATE")
            session_row = connection.execute(
                "SELECT agent_name, environment_name, status, run_epoch "
                "FROM cayu_sessions WHERE id = ?",
                (request.session_id,),
            ).fetchone()
            if session_row is None:
                raise KeyError(f"Session not found: {request.session_id}")
            if int(session_row["run_epoch"]) != request.expected_run_epoch:
                raise SessionRunFenced(
                    "Session source run epoch is stale: expected "
                    f"{request.expected_run_epoch}, current {session_row['run_epoch']}."
                )
            if str(session_row["status"]) != str(SessionStatus.RUNNING):
                raise SessionStatusConflict("Targeted tool use requires a running session.")

            def unresolved(
                reason: TargetedToolUseRejectionReason,
            ) -> tuple[TargetedToolUseResult, Event]:
                session_agent_name = str(session_row["agent_name"])
                session_environment_name = (
                    None
                    if session_row["environment_name"] is None
                    else str(session_row["environment_name"])
                )
                event = targeted_tool_unresolved_rejection_event(
                    request,
                    reason=reason,
                    timestamp=observed_at,
                    agent_name=session_agent_name,
                    environment_name=session_environment_name,
                )
                persisted = append_event_once(
                    connection,
                    event,
                    activity_at=observed_at,
                )
                validate_targeted_tool_unresolved_rejection_evidence(
                    request,
                    reason=reason,
                    event=persisted,
                    agent_name=session_agent_name,
                    environment_name=session_environment_name,
                )
                return (
                    TargetedToolUseResult(
                        disposition=TargetedToolUseDisposition.REJECTED,
                        reason=reason,
                        event=persisted,
                    ),
                    persisted,
                )

            try:
                parsed = parse_public_authority_alias(request.tool_ref)
                well_formed = (
                    parsed is not None and parsed.field_name == TARGETED_TOOL_REFERENCE_FIELD_NAME
                )
            except (TypeError, ValueError):
                well_formed = False
            if not well_formed:
                return unresolved(TargetedToolUseRejectionReason.MALFORMED)
            aliases = connection.execute(
                "SELECT scope_session_id, private_value "
                "FROM cayu_public_authority_aliases "
                "WHERE field_name = ? AND public_alias = ? LIMIT 2",
                (TARGETED_TOOL_REFERENCE_FIELD_NAME, request.tool_ref),
            ).fetchall()
            if not aliases:
                return unresolved(TargetedToolUseRejectionReason.UNKNOWN)
            if len(aliases) != 1:
                raise RuntimeError("Targeted tool reference registry is ambiguous.")
            scope_session_id = str(aliases[0]["scope_session_id"])
            grant_id = str(aliases[0]["private_value"])
            if not codec.matches(
                request.tool_ref,
                grant_id,
                field_name=TARGETED_TOOL_REFERENCE_FIELD_NAME,
                session_id=scope_session_id,
            ):
                return unresolved(TargetedToolUseRejectionReason.UNKNOWN)
            if scope_session_id != request.session_id:
                return unresolved(TargetedToolUseRejectionReason.CROSS_SESSION)
            grant_row = connection.execute(
                "SELECT * FROM cayu_targeted_tool_grants WHERE grant_id = ?",
                (grant_id,),
            ).fetchone()
            if grant_row is None:
                raise RuntimeError("Targeted tool reference lost its grant record.")
            record = decode_grant(grant_row)
            validate_use_counts(connection, (record,))

            def rejected(
                reason: TargetedToolUseRejectionReason,
            ) -> tuple[TargetedToolUseResult, Event]:
                if reason is TargetedToolUseRejectionReason.EXPIRED:
                    expiry_event = targeted_tool_grant_event(
                        record,
                        event_type=EventType.TARGETED_TOOL_GRANT_EXPIRED,
                        timestamp=observed_at,
                        outcome="expired",
                        event_id_suffix="expired",
                        rejection_reason=reason,
                    )
                    persisted_expiry = append_event_once(
                        connection,
                        expiry_event,
                        activity_at=observed_at,
                    )
                    validate_targeted_tool_grant_lifecycle_event(
                        record,
                        persisted_expiry,
                        event_type=EventType.TARGETED_TOOL_GRANT_EXPIRED,
                        outcome="expired",
                        event_id_suffix="expired",
                        rejection_reason=reason,
                        require_current_call_count=False,
                    )
                rejection_event = targeted_tool_use_rejection_event(
                    record,
                    request,
                    reason=reason,
                    timestamp=observed_at,
                )
                persisted = append_event_once(
                    connection,
                    rejection_event,
                    activity_at=observed_at,
                )
                validate_targeted_tool_use_rejection_evidence(
                    record,
                    request,
                    reason=reason,
                    event=persisted,
                )
                return (
                    TargetedToolUseResult(
                        disposition=TargetedToolUseDisposition.REJECTED,
                        reason=reason,
                        grant=record,
                        event=persisted,
                    ),
                    persisted,
                )

            terminal_placeholders = ", ".join("?" for _ in INTERACTION_TERMINAL_EVENT_TYPES)
            interaction_ended = connection.execute(
                "SELECT 1 FROM cayu_events WHERE session_id = ? AND interaction_id = ? "
                f"AND event_type IN ({terminal_placeholders}) LIMIT 1",
                (
                    request.session_id,
                    record.interaction_id,
                    *(str(event_type) for event_type in INTERACTION_TERMINAL_EVENT_TYPES),
                ),
            ).fetchone()
            if interaction_ended is not None:
                return rejected(TargetedToolUseRejectionReason.EXPIRED)
            use_rows = connection.execute(
                "SELECT * FROM cayu_targeted_tool_grant_uses "
                "WHERE session_id = ? AND interaction_id = ? "
                "AND (invocation_id = ? OR outer_tool_call_id = ?) LIMIT 2",
                (
                    request.session_id,
                    request.interaction_id,
                    request.invocation_id,
                    request.outer_tool_call_id,
                ),
            ).fetchall()
            if use_rows:
                scope_rejection = targeted_tool_use_scope_rejection_reason(record, request)
                if scope_rejection is not None:
                    return rejected(scope_rejection)
                if len(use_rows) != 1:
                    return rejected(TargetedToolUseRejectionReason.ALTERED_REPLAY)
                binding = decode_use(use_rows[0])
                candidate = targeted_tool_use_binding(
                    grant_id,
                    request,
                    bound_at=binding.bound_at,
                )
                if binding != candidate:
                    return rejected(TargetedToolUseRejectionReason.ALTERED_REPLAY)
                expected_event = targeted_tool_grant_event(
                    record,
                    event_type=EventType.TARGETED_TOOL_REFERENCE_CONSUMED,
                    timestamp=binding.bound_at,
                    outcome=TargetedToolUseDisposition.BOUND.value,
                    event_id_suffix=f"use:{binding.use_id}",
                    binding=binding,
                )
                event_row = connection.execute(
                    "SELECT * FROM cayu_events WHERE session_id = ? AND event_id = ?",
                    (request.session_id, expected_event.id),
                ).fetchone()
                if event_row is None:
                    raise RuntimeError("Targeted tool use lost its durable event evidence.")
                validate_targeted_tool_grant_lifecycle_event(
                    record,
                    sqlite_records.event_from_row(event_row),
                    event_type=EventType.TARGETED_TOOL_REFERENCE_CONSUMED,
                    outcome=TargetedToolUseDisposition.BOUND.value,
                    event_id_suffix=f"use:{binding.use_id}",
                    binding=binding,
                    require_current_call_count=False,
                )
                rejoined_event = targeted_tool_grant_event(
                    record,
                    event_type=EventType.TARGETED_TOOL_REFERENCE_REJOINED,
                    timestamp=observed_at,
                    outcome=TargetedToolUseDisposition.REJOINED.value,
                    event_id_suffix=f"rejoined:{binding.use_id}",
                    binding=binding,
                )
                persisted_rejoin = append_event_once(
                    connection,
                    rejoined_event,
                    activity_at=observed_at,
                )
                return (
                    TargetedToolUseResult(
                        disposition=TargetedToolUseDisposition.REJOINED,
                        grant=record,
                        binding=binding,
                        event=persisted_rejoin,
                    ),
                    persisted_rejoin,
                )
            rejection = targeted_tool_use_rejection_reason(
                record,
                request,
                observed_at=observed_at,
            )
            if rejection is not None:
                return rejected(rejection)
            binding = targeted_tool_use_binding(
                grant_id,
                request,
                bound_at=observed_at,
            )
            updated = TargetedToolGrantRecord.model_validate(
                record.model_copy(update={"used_calls": record.used_calls + 1}).model_dump(
                    mode="python"
                )
            )
            connection.execute(
                """
                    INSERT INTO cayu_targeted_tool_grant_uses (
                        use_id, grant_id, session_id, interaction_id, model_step_id,
                        outer_tool_call_id, arguments_sha256, invocation_id,
                        bound_at, record_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                (
                    binding.use_id,
                    binding.grant_id,
                    binding.session_id,
                    binding.interaction_id,
                    binding.model_step_id,
                    binding.outer_tool_call_id,
                    binding.arguments_sha256,
                    binding.invocation_id,
                    sqlite_records.format_datetime(binding.bound_at),
                    sqlite_records.json_dumps(binding.model_dump(mode="json")),
                ),
            )
            connection.execute(
                "UPDATE cayu_targeted_tool_grants SET used_calls = ?, record_json = ? "
                "WHERE grant_id = ? AND used_calls = ?",
                (
                    updated.used_calls,
                    sqlite_records.json_dumps(updated.model_dump(mode="json")),
                    grant_id,
                    record.used_calls,
                ),
            )
            event = targeted_tool_grant_event(
                updated,
                event_type=EventType.TARGETED_TOOL_REFERENCE_CONSUMED,
                timestamp=observed_at,
                outcome=TargetedToolUseDisposition.BOUND.value,
                event_id_suffix=f"use:{binding.use_id}",
                binding=binding,
            )
            append_events(
                connection,
                request.session_id,
                [event],
                activity_at=observed_at,
            )
            return (
                TargetedToolUseResult(
                    disposition=TargetedToolUseDisposition.BOUND,
                    grant=updated,
                    binding=binding,
                    event=event,
                ),
                event,
            )

    result, new_event = await run_write(statement)
    if result.disposition is TargetedToolUseDisposition.REJECTED:
        return result
    binding = result.binding
    if binding is None or result.grant is None:  # pragma: no cover - model invariant
        raise AssertionError("Accepted targeted tool use lost its binding.")
    if new_event is None:  # pragma: no cover - transaction invariant
        raise RuntimeError("Accepted targeted tool use lost its durable event evidence.")
    if result.event is None:  # pragma: no cover - model invariant
        raise RuntimeError("Accepted targeted tool use lost its result event evidence.")
    return result


async def revoke_targeted_tool_grant(
    run_write: SQLiteOperationRunner,
    tool_ref: str,
    *,
    session_id: str,
    expected_run_epoch: int,
    reason: str,
    revoked_at: datetime,
    closure_owners: ClosureOwners,
    get_codec: Callable[[], PublicAuthorityAliasCodec | None],
    decode_grant: Callable[[sqlite3.Row], TargetedToolGrantRecord],
    validate_use_counts: Callable[[sqlite3.Connection, Iterable[TargetedToolGrantRecord]], None],
    append_events: EventWriter,
) -> TargetedToolGrantRecord | None:
    session_id = require_clean_nonblank(session_id, "session_id")
    if type(expected_run_epoch) is not int or expected_run_epoch < 0:
        raise ValueError("expected_run_epoch must be a non-negative integer.")
    reason = validate_targeted_tool_grant_revocation_reason(reason)
    if revoked_at.tzinfo is None or revoked_at.utcoffset() is None:
        raise ValueError("revoked_at must be timezone-aware.")
    revoked_at = revoked_at.astimezone(UTC)
    codec = get_codec()
    if codec is None:
        raise RuntimeError("Targeted grants require a public authority alias codec.")

    def statement(
        connection: sqlite3.Connection,
    ) -> tuple[TargetedToolGrantRecord | None, Event | None]:
        with connection:
            connection.execute("BEGIN IMMEDIATE")
            session_row = connection.execute(
                "SELECT run_epoch FROM cayu_sessions WHERE id = ?",
                (session_id,),
            ).fetchone()
            if session_row is None:
                raise KeyError(f"Session not found: {session_id}")
            if int(session_row["run_epoch"]) != expected_run_epoch:
                raise SessionRunFenced(
                    f"Session source run epoch is stale: expected {expected_run_epoch}, "
                    f"current {session_row['run_epoch']}."
                )
            try:
                parsed = parse_public_authority_alias(tool_ref)
            except (TypeError, ValueError):
                return None, None
            if parsed is None or parsed.field_name != TARGETED_TOOL_REFERENCE_FIELD_NAME:
                return None, None
            alias_row = connection.execute(
                "SELECT scope_session_id, private_value "
                "FROM cayu_public_authority_aliases "
                "WHERE field_name = ? AND public_alias = ? LIMIT 2",
                (TARGETED_TOOL_REFERENCE_FIELD_NAME, tool_ref),
            ).fetchall()
            if not alias_row:
                return None, None
            if len(alias_row) != 1:
                raise RuntimeError("Targeted tool reference registry is ambiguous.")
            scope = str(alias_row[0]["scope_session_id"])
            grant_id = str(alias_row[0]["private_value"])
            if scope != session_id or not codec.matches(
                tool_ref,
                grant_id,
                field_name=TARGETED_TOOL_REFERENCE_FIELD_NAME,
                session_id=scope,
            ):
                return None, None
            row = connection.execute(
                "SELECT * FROM cayu_targeted_tool_grants WHERE grant_id = ?",
                (grant_id,),
            ).fetchone()
            if row is None:
                raise RuntimeError("Targeted tool reference lost its grant record.")
            record = decode_grant(row)
            validate_use_counts(connection, (record,))
            if record.revoked_at is not None:
                if record.revocation_reason != reason:
                    raise ValueError("Targeted grant was revoked with a different reason.")
                expected_event = targeted_tool_grant_event(
                    record,
                    event_type=EventType.TARGETED_TOOL_GRANT_REVOKED,
                    timestamp=record.revoked_at,
                    outcome="revoked",
                    event_id_suffix="revoked",
                )
                event_row = connection.execute(
                    "SELECT * FROM cayu_events WHERE session_id = ? AND event_id = ?",
                    (session_id, expected_event.id),
                ).fetchone()
                if event_row is None:
                    raise RuntimeError("Targeted grant revocation lost its durable event evidence.")
                persisted_event = sqlite_records.event_from_row(event_row)
                validate_targeted_tool_grant_revocation_evidence(
                    record,
                    persisted_event,
                )
                return record, persisted_event
            for owner in closure_owners((session_id,), connection=connection):
                _check_closure_lineage_owner(owner, (session_id,))
            latest_use_row = connection.execute(
                "SELECT MAX(bound_at) AS latest_bound_at "
                "FROM cayu_targeted_tool_grant_uses WHERE grant_id = ?",
                (grant_id,),
            ).fetchone()
            latest_bound_at = latest_use_row["latest_bound_at"]
            if latest_bound_at is not None and (
                sqlite_records.parse_datetime(str(latest_bound_at)) > revoked_at
            ):
                raise ValueError("revoked_at cannot precede a bound targeted tool use.")
            updated = TargetedToolGrantRecord.model_validate(
                record.model_copy(
                    update={"revoked_at": revoked_at, "revocation_reason": reason}
                ).model_dump(mode="python")
            )
            connection.execute(
                "UPDATE cayu_targeted_tool_grants SET revoked_at = ?, record_json = ? "
                "WHERE grant_id = ? AND revoked_at IS NULL",
                (
                    sqlite_records.format_datetime(revoked_at),
                    sqlite_records.json_dumps(updated.model_dump(mode="json")),
                    grant_id,
                ),
            )
            event = targeted_tool_grant_event(
                updated,
                event_type=EventType.TARGETED_TOOL_GRANT_REVOKED,
                timestamp=revoked_at,
                outcome="revoked",
                event_id_suffix="revoked",
            )
            append_events(
                connection,
                session_id,
                [event],
                activity_at=revoked_at,
            )
            return updated, event

    record, event = await run_write(statement)
    if record is None:
        return None
    if event is None:  # pragma: no cover - transaction invariant
        raise RuntimeError("Targeted grant revocation lost its durable event evidence.")
    return record


async def reconstruct_targeted_tool_grants(
    run_write: SQLiteOperationRunner,
    session_id: str,
    *,
    expected_run_epoch: int,
    interaction_id: str,
    generation_id: str,
    agent_name: str,
    task_id: str | None,
    environment_name: str | None,
    principal: str | None,
    tenant: str | None,
    catalogue_revision: str,
    descriptors_by_id: Mapping[str, tuple[str, str, str]],
    capability_ceiling_names: frozenset[str],
    observed_at: datetime,
    decode_grant: Callable[[sqlite3.Row], TargetedToolGrantRecord],
    validate_use_counts: Callable[[sqlite3.Connection, Iterable[TargetedToolGrantRecord]], None],
    append_event_once: EventOnceWriter,
) -> TargetedToolGrantReconstructionResult:
    session_id = require_clean_nonblank(session_id, "session_id")
    interaction_id = require_clean_nonblank(interaction_id, "interaction_id")
    if type(expected_run_epoch) is not int or expected_run_epoch < 0:
        raise ValueError("expected_run_epoch must be a non-negative integer.")
    if observed_at.tzinfo is None or observed_at.utcoffset() is None:
        raise ValueError("observed_at must be timezone-aware.")
    observed_at = observed_at.astimezone(UTC)

    def statement(connection: sqlite3.Connection) -> TargetedToolGrantReconstructionResult:
        with connection:
            connection.execute("BEGIN IMMEDIATE")
            session_row = connection.execute(
                "SELECT status, run_epoch FROM cayu_sessions WHERE id = ?",
                (session_id,),
            ).fetchone()
            if session_row is None:
                raise KeyError(f"Session not found: {session_id}")
            if int(session_row["run_epoch"]) != expected_run_epoch:
                raise SessionRunFenced(
                    f"Session source run epoch is stale: expected {expected_run_epoch}, "
                    f"current {session_row['run_epoch']}."
                )
            if str(session_row["status"]) != str(SessionStatus.RUNNING):
                raise SessionStatusConflict("Grant reconstruction requires a running session.")
            rows = connection.execute(
                "SELECT * FROM cayu_targeted_tool_grants "
                "WHERE session_id = ? AND interaction_id = ? "
                "ORDER BY issued_at, grant_id LIMIT ?",
                (session_id, interaction_id, TARGETED_TOOL_GRANT_MAX_REQUESTS + 1),
            ).fetchall()
            if len(rows) > TARGETED_TOOL_GRANT_MAX_REQUESTS:
                raise ValueError("Targeted grant interaction exceeds its bounded count.")
            records = tuple(decode_grant(row) for row in rows)
            validate_use_counts(connection, records)
            interaction_started_row = connection.execute(
                "SELECT * FROM cayu_events WHERE session_id = ? "
                "AND interaction_id = ? AND event_type = ? "
                "ORDER BY sequence ASC LIMIT 1",
                (session_id, interaction_id, str(EventType.INTERACTION_STARTED)),
            ).fetchone()
            if interaction_started_row is None:
                raise RuntimeError("Targeted grant reconstruction lost interaction admission.")
            validate_targeted_tool_grant_batch_evidence(
                records,
                sqlite_records.event_from_row(interaction_started_row),
            )
            placeholders = ", ".join("?" for _ in INTERACTION_TERMINAL_EVENT_TYPES)
            interaction_ended = (
                connection.execute(
                    "SELECT 1 FROM cayu_events WHERE session_id = ? AND interaction_id = ? "
                    f"AND event_type IN ({placeholders}) LIMIT 1",
                    (
                        session_id,
                        interaction_id,
                        *(str(value) for value in INTERACTION_TERMINAL_EVENT_TYPES),
                    ),
                ).fetchone()
                is not None
            )
            valid: list[TargetedToolGrantRecord] = []
            rejected: list[tuple[str, TargetedToolUseRejectionReason]] = []
            events: list[Event] = []
            for record in records:
                reason = targeted_tool_grant_reconstruction_rejection_reason(
                    record,
                    generation_id=generation_id,
                    agent_name=agent_name,
                    task_id=task_id,
                    environment_name=environment_name,
                    principal=principal,
                    tenant=tenant,
                    catalogue_revision=catalogue_revision,
                    descriptors_by_id=descriptors_by_id,
                    capability_ceiling_names=capability_ceiling_names,
                    observed_at=observed_at,
                    interaction_ended=interaction_ended,
                )
                if reason is None:
                    valid.append(record)
                    event = targeted_tool_grant_event(
                        record,
                        event_type=EventType.TARGETED_TOOL_GRANT_RECONSTRUCTED,
                        timestamp=observed_at,
                        outcome="reconstructed",
                        event_id_suffix="reconstructed",
                    )
                else:
                    rejected.append((record.grant_id, reason))
                    if reason is TargetedToolUseRejectionReason.EXPIRED:
                        persisted_expiry = append_event_once(
                            connection,
                            targeted_tool_grant_event(
                                record,
                                event_type=EventType.TARGETED_TOOL_GRANT_EXPIRED,
                                timestamp=observed_at,
                                outcome="expired",
                                event_id_suffix="expired",
                                rejection_reason=reason,
                            ),
                            activity_at=observed_at,
                        )
                        validate_targeted_tool_grant_lifecycle_event(
                            record,
                            persisted_expiry,
                            event_type=EventType.TARGETED_TOOL_GRANT_EXPIRED,
                            outcome="expired",
                            event_id_suffix="expired",
                            rejection_reason=reason,
                            require_current_call_count=False,
                        )
                    event = targeted_tool_grant_event(
                        record,
                        event_type=EventType.TARGETED_TOOL_GRANT_RECONSTRUCTED,
                        timestamp=observed_at,
                        outcome="rejected",
                        event_id_suffix=f"reconstruction-rejected:{reason.value}",
                        rejection_reason=reason,
                    )
                persisted = append_event_once(
                    connection,
                    event,
                    activity_at=observed_at,
                )
                validate_targeted_tool_grant_lifecycle_event(
                    record,
                    persisted,
                    event_type=EventType.TARGETED_TOOL_GRANT_RECONSTRUCTED,
                    outcome="reconstructed" if reason is None else "rejected",
                    event_id_suffix=(
                        "reconstructed"
                        if reason is None
                        else f"reconstruction-rejected:{reason.value}"
                    ),
                    rejection_reason=reason,
                    require_current_call_count=False,
                )
                events.append(persisted)
            return TargetedToolGrantReconstructionResult(
                valid=tuple(valid),
                rejected=tuple(rejected),
                events=tuple(events),
            )

    return await run_write(statement)

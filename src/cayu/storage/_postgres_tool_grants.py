"""Complete PostgreSQL targeted-tool grant persistence operations.

Each operation owns authorization and its native transaction. Shared decoders,
closure checks and event writers are explicit capabilities within that boundary.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from datetime import UTC, datetime
from typing import Any, Protocol

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
from cayu.storage import _postgres_support as pg_support
from cayu.storage._postgres_event_publication import EventWriter
from cayu.storage._postgres_transcript import PostgresConnection
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
    def __call__(self, cur: Any, event: Event, *, expected_run_epoch: int) -> Awaitable[Event]: ...


class AliasWriter(Protocol):
    def __call__(
        self, cur: Any, *, field_name: str, scope_key: str, public_alias: str, private_value: str
    ) -> Awaitable[None]: ...


async def issue_targeted_tool_grants(
    connect: PostgresConnection,
    session_id: str,
    *,
    expected_run_epoch: int,
    records: tuple[TargetedToolGrantRecord, ...],
    events: tuple[Event, ...],
    ensure_ready: Callable[[], Awaitable[None]],
    get_codec: Callable[[], PublicAuthorityAliasCodec | None],
    append_events: EventWriter,
    append_event_once: EventOnceWriter,
    register_alias: AliasWriter,
    decode_grant: Callable[[Sequence[object]], TargetedToolGrantRecord],
    validate_use_counts: Callable[[Any, Iterable[TargetedToolGrantRecord]], Awaitable[None]],
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
    await ensure_ready()
    resolved: list[TargetedToolGrantRecord] = []
    outcomes: list[TargetedToolGrantIssueOutcome] = []
    resolved_events: list[Event] = []
    async with connect() as conn:
        try:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT agent_name, environment_name, status, run_epoch, invocation "
                    "FROM cayu_sessions WHERE id = %s FOR UPDATE",
                    (session_id,),
                )
                session_row = await cur.fetchone()
                if session_row is None:
                    raise KeyError(f"Session not found: {session_id}")
                if int(session_row[3]) != expected_run_epoch:
                    raise SessionRunFenced(
                        "Session source run epoch is stale: expected "
                        f"{expected_run_epoch}, current {session_row[3]}."
                    )
                if str(session_row[2]) != str(SessionStatus.RUNNING):
                    raise SessionStatusConflict("Targeted grants require a running session.")
                if interaction_ids:
                    await cur.execute(
                        "SELECT interaction_id, event_type FROM cayu_events "
                        "WHERE session_id = %s AND event_type = ANY(%s) "
                        "ORDER BY sequence DESC LIMIT 1",
                        (
                            session_id,
                            [str(value) for value in INTERACTION_LIFECYCLE_EVENT_TYPES],
                        ),
                    )
                    latest_interaction = await cur.fetchone()
                    if (
                        latest_interaction is None
                        or latest_interaction[0] != next(iter(interaction_ids))
                        or EventType(str(latest_interaction[1])) in INTERACTION_TERMINAL_EVENT_TYPES
                    ):
                        raise ValueError("Targeted grants require the current open interaction.")
                    await cur.execute(
                        "SELECT event FROM cayu_events WHERE session_id = %s "
                        "AND interaction_id = %s AND event_type = %s "
                        "ORDER BY sequence ASC LIMIT 1",
                        (
                            session_id,
                            next(iter(interaction_ids)),
                            str(EventType.INTERACTION_STARTED),
                        ),
                    )
                    interaction_started_row = await cur.fetchone()
                    if interaction_started_row is None:
                        raise RuntimeError("Targeted grant issuance lost interaction admission.")
                    validate_targeted_tool_grant_batch_evidence(
                        copied_records,
                        Event(**pg_support._json_obj(interaction_started_row[0])),
                    )
                invocation = SessionInvocation.model_validate(session_row[4])
                new_events: list[Event] = []
                for record, event in zip(copied_records, copied_events, strict=True):
                    if (
                        record.session_id != session_id
                        or record.agent_name != session_row[0]
                        or record.environment_name != session_row[1]
                        or record.principal != invocation.origin.subject
                        or record.tenant != invocation.origin.tenant
                    ):
                        raise ValueError("Targeted grant scope is inconsistent.")
                    await cur.execute(
                        "SELECT grant_id, session_id, interaction_id, request_id, tool_ref, "
                        "generation_id, tool_id, tool_name, catalogue_revision, "
                        "descriptor_version, issued_at, expires_at, max_calls, used_calls, "
                        "revoked_at, record FROM cayu_targeted_tool_grants "
                        "WHERE session_id = %s AND interaction_id = %s "
                        "AND (request_id = %s OR tool_id = %s) LIMIT 2 FOR UPDATE",
                        (
                            session_id,
                            record.interaction_id,
                            record.request_id,
                            record.tool_id,
                        ),
                    )
                    existing_row = await cur.fetchone()
                    if existing_row is not None:
                        existing = decode_grant(existing_row)
                        await validate_use_counts(cur, (existing,))
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
                        await cur.execute(
                            "SELECT event FROM cayu_events WHERE session_id = %s AND event_id = %s",
                            (session_id, event.id),
                        )
                        issued_row = await cur.fetchone()
                        if issued_row is None:
                            raise RuntimeError("Targeted grant lost its durable issuance evidence.")
                        validate_targeted_tool_grant_issuance_evidence(
                            existing,
                            Event(**pg_support._json_obj(issued_row[0])),
                        )
                        reused_event = targeted_tool_grant_event(
                            existing,
                            event_type=EventType.TARGETED_TOOL_GRANT_REUSED,
                            timestamp=event.timestamp,
                            outcome=TargetedToolGrantIssueOutcome.REUSED.value,
                            event_id_suffix="reused",
                        )
                        resolved_events.append(
                            await append_event_once(
                                cur,
                                reused_event,
                                expected_run_epoch=expected_run_epoch,
                            )
                        )
                        continue
                    await cur.execute(
                        "SELECT 1 FROM cayu_targeted_tool_grants WHERE grant_id = %s FOR UPDATE",
                        (record.grant_id,),
                    )
                    if await cur.fetchone() is not None:
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
                        await register_alias(
                            cur,
                            field_name=field_name,
                            scope_key=scope_key,
                            public_alias=public_alias,
                            private_value=record.grant_id,
                        )
                    await cur.execute(
                        """
                            INSERT INTO cayu_targeted_tool_grants (
                                grant_id, session_id, interaction_id, request_id, tool_ref,
                                generation_id, tool_id, tool_name, catalogue_revision,
                                descriptor_version, issued_at, expires_at, max_calls,
                                used_calls, revoked_at, record
                            ) VALUES (
                                %s, %s, %s, %s, %s, %s, %s, %s,
                                %s, %s, %s, %s, %s, %s, %s, %s
                            )
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
                            pg_support.to_utc(record.issued_at),
                            pg_support.to_utc(record.expires_at),
                            record.max_calls,
                            record.used_calls,
                            None,
                            pg_support._dumps(record.model_dump(mode="json")),
                        ),
                    )
                    resolved.append(record)
                    outcomes.append(TargetedToolGrantIssueOutcome.ISSUED)
                    resolved_events.append(event)
                    new_events.append(event)
                if interaction_ids:
                    await cur.execute(
                        "SELECT COUNT(*) FROM cayu_targeted_tool_grants "
                        "WHERE session_id = %s AND interaction_id = %s",
                        (session_id, next(iter(interaction_ids))),
                    )
                    interaction_count_row = await cur.fetchone()
                    if (
                        interaction_count_row is None
                        or int(interaction_count_row[0]) > TARGETED_TOOL_GRANT_MAX_REQUESTS
                    ):
                        raise ValueError("Targeted grant interaction exceeds its bounded count.")
                await append_events(
                    cur,
                    session_id,
                    new_events,
                    expected_run_epoch=expected_run_epoch,
                )
            await conn.commit()
        except BaseException:
            await conn.rollback()
            raise
    return TargetedToolGrantIssueResult(
        records=tuple(resolved),
        outcomes=tuple(outcomes),
        events=tuple(resolved_events),
    )


async def list_targeted_tool_grants(
    connect: PostgresConnection,
    session_id: str,
    *,
    interaction_id: str | None = None,
    limit: int = TARGETED_TOOL_GRANT_INSPECTION_MAX_RECORDS,
    ensure_ready: Callable[[], Awaitable[None]],
    get_codec: Callable[[], PublicAuthorityAliasCodec | None],
    decode_grant: Callable[[Sequence[object]], TargetedToolGrantRecord],
    validate_use_counts: Callable[[Any, Iterable[TargetedToolGrantRecord]], Awaitable[None]],
) -> tuple[TargetedToolGrantRecord, ...]:
    session_id = require_clean_nonblank(session_id, "session_id")
    if interaction_id is not None:
        interaction_id = require_clean_nonblank(interaction_id, "interaction_id")
    if type(limit) is not int or not 1 <= limit <= TARGETED_TOOL_GRANT_INSPECTION_MAX_RECORDS:
        raise ValueError(
            f"limit must be between 1 and {TARGETED_TOOL_GRANT_INSPECTION_MAX_RECORDS}."
        )
    await ensure_ready()
    async with connect() as conn, conn.cursor() as cur:
        await cur.execute(
            "SELECT 1 FROM cayu_sessions WHERE id = %s FOR SHARE",
            (session_id,),
        )
        if await cur.fetchone() is None:
            raise KeyError(f"Session not found: {session_id}")
        if interaction_id is None:
            await cur.execute(
                "SELECT grant_id, session_id, interaction_id, request_id, tool_ref, "
                "generation_id, tool_id, tool_name, catalogue_revision, "
                "descriptor_version, issued_at, expires_at, max_calls, used_calls, "
                "revoked_at, record FROM cayu_targeted_tool_grants "
                "WHERE session_id = %s ORDER BY issued_at, grant_id LIMIT %s",
                (session_id, limit + 1),
            )
        else:
            await cur.execute(
                "SELECT grant_id, session_id, interaction_id, request_id, tool_ref, "
                "generation_id, tool_id, tool_name, catalogue_revision, "
                "descriptor_version, issued_at, expires_at, max_calls, used_calls, "
                "revoked_at, record FROM cayu_targeted_tool_grants "
                "WHERE session_id = %s AND interaction_id = %s "
                "ORDER BY issued_at, grant_id LIMIT %s",
                (session_id, interaction_id, limit + 1),
            )
        rows = await cur.fetchall()
        if len(rows) > limit:
            raise ValueError("Targeted grant inspection exceeds its bounded result limit.")
        if not rows:
            return ()
        records = tuple(decode_grant(row) for row in rows)
        await validate_use_counts(cur, records)
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


async def load_targeted_tool_grant_state(
    connect: PostgresConnection,
    session_id: str,
    *,
    ensure_ready: Callable[[], Awaitable[None]],
    get_codec: Callable[[], PublicAuthorityAliasCodec | None],
    decode_grant: Callable[[Sequence[object]], TargetedToolGrantRecord],
    decode_use: Callable[[Sequence[object]], TargetedToolUseBinding],
) -> TargetedToolGrantStateSnapshot:
    session_id = require_clean_nonblank(session_id, "session_id")
    await ensure_ready()
    async with connect() as conn, conn.cursor() as cur:
        await cur.execute(
            "SELECT 1 FROM cayu_sessions WHERE id = %s FOR SHARE",
            (session_id,),
        )
        if await cur.fetchone() is None:
            raise KeyError(f"Session not found: {session_id}")
        await cur.execute(
            "SELECT grant_id, session_id, interaction_id, request_id, tool_ref, "
            "generation_id, tool_id, tool_name, catalogue_revision, descriptor_version, "
            "issued_at, expires_at, max_calls, used_calls, revoked_at, record "
            "FROM cayu_targeted_tool_grants "
            "WHERE session_id = %s ORDER BY issued_at, grant_id",
            (session_id,),
        )
        grant_rows = await cur.fetchall()
        await cur.execute(
            "SELECT use_id, grant_id, session_id, interaction_id, model_step_id, "
            "outer_tool_call_id, arguments_sha256, invocation_id, bound_at, record "
            "FROM cayu_targeted_tool_grant_uses "
            "WHERE session_id = %s ORDER BY bound_at, use_id",
            (session_id,),
        )
        use_rows = await cur.fetchall()
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


async def bind_targeted_tool_grant_use(
    connect: PostgresConnection,
    request: TargetedToolUseRequest,
    *,
    observed_at: datetime,
    ensure_ready: Callable[[], Awaitable[None]],
    get_codec: Callable[[], PublicAuthorityAliasCodec | None],
    append_events: EventWriter,
    append_event_once: EventOnceWriter,
    decode_grant: Callable[[Sequence[object]], TargetedToolGrantRecord],
    decode_use: Callable[[Sequence[object]], TargetedToolUseBinding],
    validate_use_counts: Callable[[Any, Iterable[TargetedToolGrantRecord]], Awaitable[None]],
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
    await ensure_ready()
    result: TargetedToolUseResult
    new_event: Event | None = None
    async with connect() as conn:
        try:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT agent_name, environment_name, status, run_epoch "
                    "FROM cayu_sessions WHERE id = %s FOR UPDATE",
                    (request.session_id,),
                )
                session_row = await cur.fetchone()
                if session_row is None:
                    raise KeyError(f"Session not found: {request.session_id}")
                if int(session_row[3]) != request.expected_run_epoch:
                    raise SessionRunFenced(
                        "Session source run epoch is stale: expected "
                        f"{request.expected_run_epoch}, current {session_row[3]}."
                    )
                if str(session_row[2]) != str(SessionStatus.RUNNING):
                    raise SessionStatusConflict("Targeted tool use requires a running session.")

                async def unresolved(
                    reason: TargetedToolUseRejectionReason,
                ) -> TargetedToolUseResult:
                    session_agent_name = str(session_row[0])
                    session_environment_name = (
                        None if session_row[1] is None else str(session_row[1])
                    )
                    event = targeted_tool_unresolved_rejection_event(
                        request,
                        reason=reason,
                        timestamp=observed_at,
                        agent_name=session_agent_name,
                        environment_name=session_environment_name,
                    )
                    persisted = await append_event_once(
                        cur,
                        event,
                        expected_run_epoch=request.expected_run_epoch,
                    )
                    validate_targeted_tool_unresolved_rejection_evidence(
                        request,
                        reason=reason,
                        event=persisted,
                        agent_name=session_agent_name,
                        environment_name=session_environment_name,
                    )
                    return TargetedToolUseResult(
                        disposition=TargetedToolUseDisposition.REJECTED,
                        reason=reason,
                        event=persisted,
                    )

                try:
                    parsed = parse_public_authority_alias(request.tool_ref)
                    well_formed = (
                        parsed is not None
                        and parsed.field_name == TARGETED_TOOL_REFERENCE_FIELD_NAME
                    )
                except (TypeError, ValueError):
                    well_formed = False
                if not well_formed:
                    result = await unresolved(TargetedToolUseRejectionReason.MALFORMED)
                    await conn.commit()
                    return result
                await cur.execute(
                    "SELECT scope_session_id, private_value "
                    "FROM cayu_public_authority_aliases "
                    "WHERE field_name = %s AND public_alias = %s LIMIT 2",
                    (TARGETED_TOOL_REFERENCE_FIELD_NAME, request.tool_ref),
                )
                aliases = await cur.fetchall()
                if not aliases:
                    result = await unresolved(TargetedToolUseRejectionReason.UNKNOWN)
                elif len(aliases) != 1:
                    raise RuntimeError("Targeted tool reference registry is ambiguous.")
                else:
                    scope_session_id = str(aliases[0][0])
                    grant_id = str(aliases[0][1])
                    if not codec.matches(
                        request.tool_ref,
                        grant_id,
                        field_name=TARGETED_TOOL_REFERENCE_FIELD_NAME,
                        session_id=scope_session_id,
                    ):
                        result = await unresolved(TargetedToolUseRejectionReason.UNKNOWN)
                    elif scope_session_id != request.session_id:
                        result = await unresolved(TargetedToolUseRejectionReason.CROSS_SESSION)
                    else:
                        await cur.execute(
                            "SELECT grant_id, session_id, interaction_id, request_id, "
                            "tool_ref, generation_id, tool_id, tool_name, "
                            "catalogue_revision, descriptor_version, issued_at, expires_at, "
                            "max_calls, used_calls, revoked_at, record "
                            "FROM cayu_targeted_tool_grants "
                            "WHERE grant_id = %s FOR UPDATE",
                            (grant_id,),
                        )
                        grant_row = await cur.fetchone()
                        if grant_row is None:
                            raise RuntimeError("Targeted tool reference lost its grant record.")
                        record = decode_grant(grant_row)
                        await validate_use_counts(cur, (record,))

                        async def rejected(
                            reason: TargetedToolUseRejectionReason,
                        ) -> TargetedToolUseResult:
                            if reason is TargetedToolUseRejectionReason.EXPIRED:
                                expiry_event = targeted_tool_grant_event(
                                    record,
                                    event_type=EventType.TARGETED_TOOL_GRANT_EXPIRED,
                                    timestamp=observed_at,
                                    outcome="expired",
                                    event_id_suffix="expired",
                                    rejection_reason=reason,
                                )
                                persisted_expiry = await append_event_once(
                                    cur,
                                    expiry_event,
                                    expected_run_epoch=request.expected_run_epoch,
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
                            persisted = await append_event_once(
                                cur,
                                rejection_event,
                                expected_run_epoch=request.expected_run_epoch,
                            )
                            validate_targeted_tool_use_rejection_evidence(
                                record,
                                request,
                                reason=reason,
                                event=persisted,
                            )
                            return TargetedToolUseResult(
                                disposition=TargetedToolUseDisposition.REJECTED,
                                reason=reason,
                                grant=record,
                                event=persisted,
                            )

                        await cur.execute(
                            "SELECT 1 FROM cayu_events "
                            "WHERE session_id = %s AND interaction_id = %s "
                            "AND event_type = ANY(%s) LIMIT 1",
                            (
                                request.session_id,
                                record.interaction_id,
                                [
                                    str(event_type)
                                    for event_type in INTERACTION_TERMINAL_EVENT_TYPES
                                ],
                            ),
                        )
                        if await cur.fetchone() is not None:
                            result = await rejected(TargetedToolUseRejectionReason.EXPIRED)
                            await conn.commit()
                            return result
                        await cur.execute(
                            "SELECT use_id, grant_id, session_id, interaction_id, "
                            "model_step_id, outer_tool_call_id, arguments_sha256, "
                            "invocation_id, bound_at, record "
                            "FROM cayu_targeted_tool_grant_uses "
                            "WHERE session_id = %s AND interaction_id = %s "
                            "AND (invocation_id = %s OR outer_tool_call_id = %s) LIMIT 2",
                            (
                                request.session_id,
                                request.interaction_id,
                                request.invocation_id,
                                request.outer_tool_call_id,
                            ),
                        )
                        use_rows = await cur.fetchall()
                        if use_rows:
                            scope_rejection = targeted_tool_use_scope_rejection_reason(
                                record,
                                request,
                            )
                            if scope_rejection is not None:
                                result = await rejected(scope_rejection)
                                new_event = result.event
                            elif len(use_rows) != 1:
                                result = await rejected(
                                    TargetedToolUseRejectionReason.ALTERED_REPLAY
                                )
                                new_event = result.event
                            else:
                                binding = decode_use(use_rows[0])
                                candidate = targeted_tool_use_binding(
                                    grant_id,
                                    request,
                                    bound_at=binding.bound_at,
                                )
                                if binding != candidate:
                                    result = await rejected(
                                        TargetedToolUseRejectionReason.ALTERED_REPLAY
                                    )
                                    new_event = result.event
                                else:
                                    expected_event = targeted_tool_grant_event(
                                        record,
                                        event_type=(EventType.TARGETED_TOOL_REFERENCE_CONSUMED),
                                        timestamp=binding.bound_at,
                                        outcome=TargetedToolUseDisposition.BOUND.value,
                                        event_id_suffix=f"use:{binding.use_id}",
                                        binding=binding,
                                    )
                                    await cur.execute(
                                        "SELECT event FROM cayu_events "
                                        "WHERE session_id = %s AND event_id = %s",
                                        (request.session_id, expected_event.id),
                                    )
                                    event_row = await cur.fetchone()
                                    if event_row is None:
                                        raise RuntimeError(
                                            "Targeted tool use lost its durable event evidence."
                                        )
                                    validate_targeted_tool_grant_lifecycle_event(
                                        record,
                                        Event(**pg_support._json_obj(event_row[0])),
                                        event_type=(EventType.TARGETED_TOOL_REFERENCE_CONSUMED),
                                        outcome=TargetedToolUseDisposition.BOUND.value,
                                        event_id_suffix=f"use:{binding.use_id}",
                                        binding=binding,
                                        require_current_call_count=False,
                                    )
                                    rejoined_event = targeted_tool_grant_event(
                                        record,
                                        event_type=(EventType.TARGETED_TOOL_REFERENCE_REJOINED),
                                        timestamp=observed_at,
                                        outcome=TargetedToolUseDisposition.REJOINED.value,
                                        event_id_suffix=f"rejoined:{binding.use_id}",
                                        binding=binding,
                                    )
                                    new_event = await append_event_once(
                                        cur,
                                        rejoined_event,
                                        expected_run_epoch=request.expected_run_epoch,
                                    )
                                    result = TargetedToolUseResult(
                                        disposition=TargetedToolUseDisposition.REJOINED,
                                        grant=record,
                                        binding=binding,
                                        event=new_event,
                                    )
                        else:
                            rejection = targeted_tool_use_rejection_reason(
                                record,
                                request,
                                observed_at=observed_at,
                            )
                            if rejection is not None:
                                result = await rejected(rejection)
                                new_event = result.event
                            else:
                                binding = targeted_tool_use_binding(
                                    grant_id,
                                    request,
                                    bound_at=observed_at,
                                )
                                updated = TargetedToolGrantRecord.model_validate(
                                    record.model_copy(
                                        update={"used_calls": record.used_calls + 1}
                                    ).model_dump(mode="python")
                                )
                                await cur.execute(
                                    """
                                        INSERT INTO cayu_targeted_tool_grant_uses (
                                            use_id, grant_id, session_id, interaction_id,
                                            model_step_id, outer_tool_call_id,
                                            arguments_sha256, invocation_id, bound_at, record
                                        ) VALUES (
                                            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                                        )
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
                                        pg_support.to_utc(binding.bound_at),
                                        pg_support._dumps(binding.model_dump(mode="json")),
                                    ),
                                )
                                await cur.execute(
                                    "UPDATE cayu_targeted_tool_grants "
                                    "SET used_calls = %s, record = %s "
                                    "WHERE grant_id = %s AND used_calls = %s",
                                    (
                                        updated.used_calls,
                                        pg_support._dumps(updated.model_dump(mode="json")),
                                        grant_id,
                                        record.used_calls,
                                    ),
                                )
                                if cur.rowcount != 1:
                                    raise RuntimeError("Targeted grant use lost its row lock.")
                                new_event = targeted_tool_grant_event(
                                    updated,
                                    event_type=(EventType.TARGETED_TOOL_REFERENCE_CONSUMED),
                                    timestamp=observed_at,
                                    outcome=TargetedToolUseDisposition.BOUND.value,
                                    event_id_suffix=f"use:{binding.use_id}",
                                    binding=binding,
                                )
                                result = TargetedToolUseResult(
                                    disposition=TargetedToolUseDisposition.BOUND,
                                    grant=updated,
                                    binding=binding,
                                    event=new_event,
                                )
                                await append_events(
                                    cur,
                                    request.session_id,
                                    [new_event],
                                    expected_run_epoch=request.expected_run_epoch,
                                )
            await conn.commit()
        except BaseException:
            await conn.rollback()
            raise
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
    connect: PostgresConnection,
    tool_ref: str,
    *,
    session_id: str,
    expected_run_epoch: int,
    reason: str,
    revoked_at: datetime,
    ensure_ready: Callable[[], Awaitable[None]],
    closure_owners: Callable[[Any, Iterable[str]], Awaitable[tuple[dict[str, Any], ...]]],
    get_codec: Callable[[], PublicAuthorityAliasCodec | None],
    append_events: EventWriter,
    decode_grant: Callable[[Sequence[object]], TargetedToolGrantRecord],
    validate_use_counts: Callable[[Any, Iterable[TargetedToolGrantRecord]], Awaitable[None]],
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
    await ensure_ready()
    record: TargetedToolGrantRecord | None = None
    event: Event | None = None
    async with connect() as conn:
        try:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT run_epoch FROM cayu_sessions WHERE id = %s FOR UPDATE",
                    (session_id,),
                )
                session_row = await cur.fetchone()
                if session_row is None:
                    raise KeyError(f"Session not found: {session_id}")
                if int(session_row[0]) != expected_run_epoch:
                    raise SessionRunFenced(
                        "Session source run epoch is stale: expected "
                        f"{expected_run_epoch}, current {session_row[0]}."
                    )
                try:
                    parsed = parse_public_authority_alias(tool_ref)
                except (TypeError, ValueError):
                    parsed = None
                if parsed is None or parsed.field_name != TARGETED_TOOL_REFERENCE_FIELD_NAME:
                    await conn.commit()
                    return None
                await cur.execute(
                    "SELECT scope_session_id, private_value "
                    "FROM cayu_public_authority_aliases "
                    "WHERE field_name = %s AND public_alias = %s LIMIT 2",
                    (TARGETED_TOOL_REFERENCE_FIELD_NAME, tool_ref),
                )
                aliases = await cur.fetchall()
                if aliases:
                    if len(aliases) != 1:
                        raise RuntimeError("Targeted tool reference registry is ambiguous.")
                    scope = str(aliases[0][0])
                    grant_id = str(aliases[0][1])
                    if scope == session_id and codec.matches(
                        tool_ref,
                        grant_id,
                        field_name=TARGETED_TOOL_REFERENCE_FIELD_NAME,
                        session_id=scope,
                    ):
                        await cur.execute(
                            "SELECT grant_id, session_id, interaction_id, request_id, "
                            "tool_ref, generation_id, tool_id, tool_name, "
                            "catalogue_revision, descriptor_version, issued_at, expires_at, "
                            "max_calls, used_calls, revoked_at, record "
                            "FROM cayu_targeted_tool_grants "
                            "WHERE grant_id = %s FOR UPDATE",
                            (grant_id,),
                        )
                        row = await cur.fetchone()
                        if row is None:
                            raise RuntimeError("Targeted tool reference lost its grant record.")
                        stored = decode_grant(row)
                        await validate_use_counts(cur, (stored,))
                        if stored.revoked_at is not None:
                            if stored.revocation_reason != reason:
                                raise ValueError(
                                    "Targeted grant was revoked with a different reason."
                                )
                            record = stored
                            expected_event = targeted_tool_grant_event(
                                stored,
                                event_type=EventType.TARGETED_TOOL_GRANT_REVOKED,
                                timestamp=stored.revoked_at,
                                outcome="revoked",
                                event_id_suffix="revoked",
                            )
                            await cur.execute(
                                "SELECT event FROM cayu_events "
                                "WHERE session_id = %s AND event_id = %s",
                                (session_id, expected_event.id),
                            )
                            event_row = await cur.fetchone()
                            if event_row is None:
                                raise RuntimeError(
                                    "Targeted grant revocation lost its durable event evidence."
                                )
                            event = Event(**pg_support._json_obj(event_row[0]))
                            validate_targeted_tool_grant_revocation_evidence(
                                stored,
                                event,
                            )
                        else:
                            for owner in await closure_owners(cur, (session_id,)):
                                _check_closure_lineage_owner(owner, (session_id,))
                            await cur.execute(
                                "SELECT MAX(bound_at) "
                                "FROM cayu_targeted_tool_grant_uses WHERE grant_id = %s",
                                (grant_id,),
                            )
                            latest_use_row = await cur.fetchone()
                            latest_bound_at = latest_use_row[0]
                            if latest_bound_at is not None and (
                                pg_support.to_utc(latest_bound_at) > revoked_at
                            ):
                                raise ValueError(
                                    "revoked_at cannot precede a bound targeted tool use."
                                )
                            record = TargetedToolGrantRecord.model_validate(
                                stored.model_copy(
                                    update={
                                        "revoked_at": revoked_at,
                                        "revocation_reason": reason,
                                    }
                                ).model_dump(mode="python")
                            )
                            await cur.execute(
                                "UPDATE cayu_targeted_tool_grants "
                                "SET revoked_at = %s, record = %s "
                                "WHERE grant_id = %s AND revoked_at IS NULL",
                                (
                                    pg_support.to_utc(revoked_at),
                                    pg_support._dumps(record.model_dump(mode="json")),
                                    grant_id,
                                ),
                            )
                            if cur.rowcount != 1:
                                raise RuntimeError("Targeted grant revocation lost its row lock.")
                            event = targeted_tool_grant_event(
                                record,
                                event_type=EventType.TARGETED_TOOL_GRANT_REVOKED,
                                timestamp=revoked_at,
                                outcome="revoked",
                                event_id_suffix="revoked",
                            )
                            await append_events(
                                cur,
                                session_id,
                                [event],
                                expected_run_epoch=expected_run_epoch,
                            )
            await conn.commit()
        except BaseException:
            await conn.rollback()
            raise
    if record is None:
        return None
    if event is None:  # pragma: no cover - transaction invariant
        raise RuntimeError("Targeted grant revocation lost its durable event evidence.")
    return record


async def reconstruct_targeted_tool_grants(
    connect: PostgresConnection,
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
    ensure_ready: Callable[[], Awaitable[None]],
    append_event_once: EventOnceWriter,
    decode_grant: Callable[[Sequence[object]], TargetedToolGrantRecord],
    validate_use_counts: Callable[[Any, Iterable[TargetedToolGrantRecord]], Awaitable[None]],
) -> TargetedToolGrantReconstructionResult:
    session_id = require_clean_nonblank(session_id, "session_id")
    interaction_id = require_clean_nonblank(interaction_id, "interaction_id")
    if type(expected_run_epoch) is not int or expected_run_epoch < 0:
        raise ValueError("expected_run_epoch must be a non-negative integer.")
    if observed_at.tzinfo is None or observed_at.utcoffset() is None:
        raise ValueError("observed_at must be timezone-aware.")
    observed_at = observed_at.astimezone(UTC)
    await ensure_ready()
    result: TargetedToolGrantReconstructionResult
    async with connect() as conn:
        try:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT status, run_epoch FROM cayu_sessions WHERE id = %s FOR UPDATE",
                    (session_id,),
                )
                session_row = await cur.fetchone()
                if session_row is None:
                    raise KeyError(f"Session not found: {session_id}")
                if int(session_row[1]) != expected_run_epoch:
                    raise SessionRunFenced(
                        "Session source run epoch is stale: expected "
                        f"{expected_run_epoch}, current {session_row[1]}."
                    )
                if str(session_row[0]) != str(SessionStatus.RUNNING):
                    raise SessionStatusConflict("Grant reconstruction requires a running session.")
                await cur.execute(
                    "SELECT grant_id, session_id, interaction_id, request_id, tool_ref, "
                    "generation_id, tool_id, tool_name, catalogue_revision, "
                    "descriptor_version, issued_at, expires_at, max_calls, used_calls, "
                    "revoked_at, record FROM cayu_targeted_tool_grants "
                    "WHERE session_id = %s AND interaction_id = %s "
                    "ORDER BY issued_at, grant_id LIMIT %s",
                    (
                        session_id,
                        interaction_id,
                        TARGETED_TOOL_GRANT_MAX_REQUESTS + 1,
                    ),
                )
                records = tuple(decode_grant(row) for row in await cur.fetchall())
                if len(records) > TARGETED_TOOL_GRANT_MAX_REQUESTS:
                    raise ValueError("Targeted grant interaction exceeds its bounded count.")
                await validate_use_counts(cur, records)
                await cur.execute(
                    "SELECT event FROM cayu_events WHERE session_id = %s "
                    "AND interaction_id = %s AND event_type = %s "
                    "ORDER BY sequence ASC LIMIT 1",
                    (session_id, interaction_id, str(EventType.INTERACTION_STARTED)),
                )
                interaction_started_row = await cur.fetchone()
                if interaction_started_row is None:
                    raise RuntimeError("Targeted grant reconstruction lost interaction admission.")
                validate_targeted_tool_grant_batch_evidence(
                    records,
                    Event(**pg_support._json_obj(interaction_started_row[0])),
                )
                await cur.execute(
                    "SELECT 1 FROM cayu_events "
                    "WHERE session_id = %s AND interaction_id = %s "
                    "AND event_type = ANY(%s) LIMIT 1",
                    (
                        session_id,
                        interaction_id,
                        [str(value) for value in INTERACTION_TERMINAL_EVENT_TYPES],
                    ),
                )
                interaction_ended = await cur.fetchone() is not None
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
                            persisted_expiry = await append_event_once(
                                cur,
                                targeted_tool_grant_event(
                                    record,
                                    event_type=EventType.TARGETED_TOOL_GRANT_EXPIRED,
                                    timestamp=observed_at,
                                    outcome="expired",
                                    event_id_suffix="expired",
                                    rejection_reason=reason,
                                ),
                                expected_run_epoch=expected_run_epoch,
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
                    persisted = await append_event_once(
                        cur,
                        event,
                        expected_run_epoch=expected_run_epoch,
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
                result = TargetedToolGrantReconstructionResult(
                    valid=tuple(valid),
                    rejected=tuple(rejected),
                    events=tuple(events),
                )
            await conn.commit()
        except BaseException:
            await conn.rollback()
            raise
    return result

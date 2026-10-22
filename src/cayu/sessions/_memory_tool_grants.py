"""Complete in-memory targeted-tool grant operations over explicit locked state."""

from __future__ import annotations

import heapq
from collections.abc import Callable, Iterable, Mapping
from contextlib import AbstractAsyncContextManager
from datetime import UTC, datetime
from typing import Generic, NamedTuple, TypeVar

from cayu._validation import require_durable_clean_nonblank as require_clean_nonblank
from cayu.events import Event, EventType, copy_event
from cayu.runtime.public_authority import (
    PublicAuthorityAliasCodec,
    _public_authority_alias_store_key,
    parse_public_authority_alias,
)
from cayu.sessions import _closure_records as closure_records
from cayu.sessions.authority import SessionRunFenced
from cayu.sessions.records import EventRecord, Session, SessionStatus, SessionStatusConflict
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

PreparedAppend = TypeVar("PreparedAppend")


class MemoryToolGrantState(NamedTuple, Generic[PreparedAppend]):
    """Current grant state borrowed under one caller-owned lock.

    Authorization, grant/use indexes and event publication share this lock.
    ``prepare_events`` validates and detaches an append without publishing it;
    ``apply_events`` publishes its prepared token synchronously. The token must
    remain within this lock scope. State is resolved after acquiring the lock
    for each operation so session deletion cannot leave captured stale maps.
    """

    event_records_by_id: dict[tuple[str, str], EventRecord]
    interaction_event_records: dict[str, dict[str, list[EventRecord]]]
    latest_interaction_event_records_by_sequence: dict[str, list[EventRecord]]
    public_authority_aliases: dict[tuple[str, str, str], str]
    session_closure_progress: closure_records.ClosureProgressIndex
    sessions: dict[str, Session]
    targeted_tool_grant_ids_by_interaction: dict[tuple[str, str], list[str]]
    targeted_tool_grant_ids_by_request: dict[tuple[str, str, str], str]
    targeted_tool_grant_ids_by_session: dict[str, list[str]]
    targeted_tool_grant_ids_by_tool: dict[tuple[str, str, str], str]
    targeted_tool_grant_refs: dict[str, tuple[str, str]]
    targeted_tool_grants: dict[str, TargetedToolGrantRecord]
    targeted_tool_use_by_invocation: dict[tuple[str, str, str], TargetedToolUseBinding]
    targeted_tool_use_by_outer_call: dict[tuple[str, str, str], TargetedToolUseBinding]
    targeted_tool_uses: dict[str, dict[str, TargetedToolUseBinding]]
    prepare_events: Callable[[Session, Iterable[Event]], PreparedAppend]
    apply_events: Callable[[Session, PreparedAppend], Session]


LockedToolGrantState = Callable[
    [], AbstractAsyncContextManager[MemoryToolGrantState[PreparedAppend]]
]


async def issue_targeted_tool_grants(
    locked: LockedToolGrantState[PreparedAppend],
    session_id: str,
    *,
    get_codec: Callable[[], PublicAuthorityAliasCodec],
    expected_run_epoch: int,
    records: tuple[TargetedToolGrantRecord, ...],
    events: tuple[Event, ...],
) -> TargetedToolGrantIssueResult:
    session_id = require_clean_nonblank(session_id, "session_id")
    if type(expected_run_epoch) is not int or expected_run_epoch < 0:
        raise ValueError("expected_run_epoch must be a non-negative integer.")
    if type(records) is not tuple or type(events) is not tuple:
        raise TypeError("records and events must be tuples.")
    if len(records) > TARGETED_TOOL_GRANT_MAX_REQUESTS:
        raise ValueError("Targeted grant issuance exceeds the bounded request count.")
    copied_records = tuple(copy_targeted_tool_grant_record(record) for record in records)
    copied_events = tuple(copy_event(event) for event in events)
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
    if copied_records and codec is None:
        raise RuntimeError("Targeted grants require a public authority alias codec.")
    for record, event in zip(copied_records, copied_events, strict=True):
        if record.session_id != session_id:
            raise ValueError("Targeted grant scope is inconsistent.")
        assert codec is not None
        validate_targeted_tool_grant_reference(record, codec)
        validate_targeted_tool_grant_issuance_evidence(record, event)

    async with locked() as state:
        session = state.sessions.get(session_id)
        if session is None:
            raise KeyError(f"Session not found: {session_id}")
        if session.run_epoch != expected_run_epoch:
            raise SessionRunFenced(
                f"Session source run epoch is stale: expected {expected_run_epoch}, "
                f"current {session.run_epoch}."
            )
        if session.status is not SessionStatus.RUNNING:
            raise SessionStatusConflict("Targeted grants require a running session.")
        if interaction_ids:
            latest_interactions = state.latest_interaction_event_records_by_sequence.get(
                session_id,
                (),
            )
            latest = latest_interactions[-1] if latest_interactions else None
            if (
                latest is None
                or latest.event.interaction_id != next(iter(interaction_ids))
                or latest.event.type
                in {
                    EventType.INTERACTION_COMPLETED,
                    EventType.INTERACTION_FAILED,
                    EventType.INTERACTION_INTERRUPTED,
                }
            ):
                raise ValueError("Targeted grants require the current open interaction.")
            interaction_started = next(
                (
                    event_record.event
                    for event_record in latest_interactions
                    if event_record.event.type is EventType.INTERACTION_STARTED
                    and event_record.event.interaction_id == next(iter(interaction_ids))
                ),
                None,
            )
            if interaction_started is None:
                raise RuntimeError("Targeted grant issuance lost interaction admission.")
            validate_targeted_tool_grant_batch_evidence(
                copied_records,
                interaction_started,
            )
        for record in copied_records:
            if (
                record.agent_name != session.agent_name
                or record.environment_name != session.environment_name
                or record.principal != session.invocation.origin.subject
                or record.tenant != session.invocation.origin.tenant
            ):
                raise ValueError("Targeted grant scope conflicts with session authority.")

        resolved_records: list[TargetedToolGrantRecord] = []
        outcomes: list[TargetedToolGrantIssueOutcome] = []
        resolved_events: list[Event] = []
        new_records: list[tuple[TargetedToolGrantRecord, Event]] = []
        events_to_append: list[Event] = []
        for record, event in zip(copied_records, copied_events, strict=True):
            request_key = (session_id, record.interaction_id, record.request_id)
            tool_key = (session_id, record.interaction_id, record.tool_id)
            existing_id = state.targeted_tool_grant_ids_by_request.get(request_key)
            existing = None if existing_id is None else state.targeted_tool_grants.get(existing_id)
            if existing is not None:
                if existing.grant_id != record.grant_id:
                    raise ValueError(
                        "Targeted grant request identity conflicts with durable authority."
                    )
                if len(state.targeted_tool_uses.get(existing.grant_id, ())) != (
                    existing.used_calls
                ):
                    raise ValueError("Targeted grant call counter conflicts with durable uses.")
                existing_event_record = state.event_records_by_id.get((session_id, event.id))
                if existing_event_record is None:
                    raise RuntimeError("Targeted grant lost its durable issuance evidence.")
                validate_targeted_tool_grant_issuance_evidence(
                    existing,
                    existing_event_record.event,
                )
                reused_event = targeted_tool_grant_event(
                    existing,
                    event_type=EventType.TARGETED_TOOL_GRANT_REUSED,
                    timestamp=event.timestamp,
                    outcome=TargetedToolGrantIssueOutcome.REUSED.value,
                    event_id_suffix="reused",
                )
                existing_reuse = state.event_records_by_id.get((session_id, reused_event.id))
                codec = get_codec()
                if codec is None:  # pragma: no cover - capability invariant
                    raise RuntimeError("Targeted grants require a public alias codec.")
                resolved_records.append(targeted_tool_grant_with_active_reference(existing, codec))
                outcomes.append(TargetedToolGrantIssueOutcome.REUSED)
                if existing_reuse is None:
                    events_to_append.append(reused_event)
                    resolved_events.append(reused_event)
                else:
                    resolved_events.append(copy_event(existing_reuse.event))
                continue
            conflicting_tool_grant_id = state.targeted_tool_grant_ids_by_tool.get(tool_key)
            if conflicting_tool_grant_id is not None:
                raise ValueError("Targeted grant tool identity conflicts with durable authority.")
            if record.grant_id in state.targeted_tool_grants:
                raise ValueError("Targeted grant identity collides with existing authority.")
            if (session_id, event.id) in state.event_records_by_id:
                raise ValueError("Targeted grant event identity is already in use.")
            new_records.append((record, event))
            resolved_records.append(record)
            outcomes.append(TargetedToolGrantIssueOutcome.ISSUED)
            resolved_events.append(event)
            events_to_append.append(event)

        if new_records:
            interaction_id = new_records[0][0].interaction_id
            existing_count = len(
                state.targeted_tool_grant_ids_by_interaction.get(
                    (session_id, interaction_id),
                    (),
                )
            )
            if existing_count + len(new_records) > TARGETED_TOOL_GRANT_MAX_REQUESTS:
                raise ValueError("Targeted grant interaction exceeds its bounded count.")
        if events_to_append:
            prepared_append = state.prepare_events(
                session,
                events_to_append,
            )
        alias_plans: dict[
            str,
            tuple[tuple[str, str, str], str, str],
        ] = {}
        for record, _event in new_records:
            assert codec is not None
            for public_alias in codec.aliases(
                record.grant_id,
                field_name=TARGETED_TOOL_REFERENCE_FIELD_NAME,
                session_id=session_id,
            ):
                alias_key = _public_authority_alias_store_key(
                    public_alias,
                    field_name=TARGETED_TOOL_REFERENCE_FIELD_NAME,
                    private_value=record.grant_id,
                    scope_session_id=session_id,
                )
                existing_private = state.public_authority_aliases.get(alias_key)
                if existing_private is not None and existing_private != record.grant_id:
                    raise ValueError("Targeted tool reference collides with authority.")
                existing_plan = alias_plans.get(public_alias)
                if existing_plan is not None and existing_plan[1] != record.grant_id:
                    raise ValueError("Targeted tool reference batch contains a collision.")
                existing_ref = state.targeted_tool_grant_refs.get(public_alias)
                if existing_ref is not None and existing_ref != (
                    session_id,
                    record.grant_id,
                ):
                    raise ValueError("Targeted tool reference collides with authority.")
                alias_plans[public_alias] = (
                    alias_key,
                    record.grant_id,
                    session_id,
                )
        if new_records:
            for record, _event in new_records:
                for public_alias in codec.aliases(
                    record.grant_id,
                    field_name=TARGETED_TOOL_REFERENCE_FIELD_NAME,
                    session_id=session_id,
                ):
                    alias_key, private_value, scope_key = alias_plans[public_alias]
                    state.public_authority_aliases[alias_key] = private_value
                    state.targeted_tool_grant_refs[public_alias] = (
                        scope_key,
                        private_value,
                    )
                state.targeted_tool_grants[record.grant_id] = record
                state.targeted_tool_grant_ids_by_session.setdefault(session_id, []).append(
                    record.grant_id
                )
                state.targeted_tool_grant_ids_by_interaction.setdefault(
                    (session_id, record.interaction_id),
                    [],
                ).append(record.grant_id)
                state.targeted_tool_grant_ids_by_request[
                    (session_id, record.interaction_id, record.request_id)
                ] = record.grant_id
                state.targeted_tool_grant_ids_by_tool[
                    (session_id, record.interaction_id, record.tool_id)
                ] = record.grant_id
        if events_to_append:
            state.sessions[session_id] = state.apply_events(
                session,
                prepared_append,
            )
        return TargetedToolGrantIssueResult(
            records=tuple(resolved_records),
            outcomes=tuple(outcomes),
            events=tuple(resolved_events),
        )


async def list_targeted_tool_grants(
    locked: LockedToolGrantState[PreparedAppend],
    session_id: str,
    *,
    get_codec: Callable[[], PublicAuthorityAliasCodec],
    interaction_id: str | None = None,
    limit: int = TARGETED_TOOL_GRANT_INSPECTION_MAX_RECORDS,
) -> tuple[TargetedToolGrantRecord, ...]:
    session_id = require_clean_nonblank(session_id, "session_id")
    if interaction_id is not None:
        interaction_id = require_clean_nonblank(interaction_id, "interaction_id")
    if type(limit) is not int or not 1 <= limit <= TARGETED_TOOL_GRANT_INSPECTION_MAX_RECORDS:
        raise ValueError(
            f"limit must be between 1 and {TARGETED_TOOL_GRANT_INSPECTION_MAX_RECORDS}."
        )
    async with locked() as state:
        if session_id not in state.sessions:
            raise KeyError(f"Session not found: {session_id}")
        codec = get_codec()
        if codec is None:  # pragma: no cover - in-memory capability invariant
            raise RuntimeError("Targeted grants require a public authority alias codec.")
        grant_ids = (
            state.targeted_tool_grant_ids_by_session.get(session_id, ())
            if interaction_id is None
            else state.targeted_tool_grant_ids_by_interaction.get(
                (session_id, interaction_id),
                (),
            )
        )
        selected = (state.targeted_tool_grants[grant_id] for grant_id in grant_ids)
        bounded = heapq.nsmallest(
            limit + 1,
            selected,
            key=lambda candidate: (candidate.issued_at, candidate.grant_id),
        )
        if len(bounded) > limit:
            raise ValueError("Targeted grant inspection exceeds its bounded result limit.")
        if any(
            len(state.targeted_tool_uses.get(record.grant_id, ())) != record.used_calls
            for record in bounded
        ):
            raise ValueError("Targeted grant call counter conflicts with durable uses.")
        return tuple(targeted_tool_grant_with_active_reference(record, codec) for record in bounded)


async def load_targeted_tool_grant_state(
    locked: LockedToolGrantState[PreparedAppend],
    session_id: str,
    *,
    get_codec: Callable[[], PublicAuthorityAliasCodec],
) -> TargetedToolGrantStateSnapshot:
    session_id = require_clean_nonblank(session_id, "session_id")
    async with locked() as state:
        if session_id not in state.sessions:
            raise KeyError(f"Session not found: {session_id}")
        codec = get_codec()
        if codec is None:  # pragma: no cover - in-memory capability invariant
            raise RuntimeError("Targeted grants require a public authority alias codec.")
        records = tuple(
            targeted_tool_grant_with_active_reference(record, codec)
            for record in sorted(
                (
                    state.targeted_tool_grants[grant_id]
                    for grant_id in state.targeted_tool_grant_ids_by_session.get(
                        session_id,
                        (),
                    )
                ),
                key=lambda record: (record.issued_at, record.grant_id),
            )
        )
        grant_ids = {record.grant_id for record in records}
        uses = tuple(
            TargetedToolUseBinding.model_validate(binding.model_dump(mode="python"))
            for binding in sorted(
                (
                    binding
                    for grant_id in grant_ids
                    for binding in state.targeted_tool_uses.get(grant_id, {}).values()
                ),
                key=lambda binding: (binding.bound_at, binding.use_id),
            )
        )
        return TargetedToolGrantStateSnapshot(records=records, uses=uses)


def _resolve_targeted_tool_ref_unlocked(
    state: MemoryToolGrantState[PreparedAppend],
    get_codec: Callable[[], PublicAuthorityAliasCodec],
    tool_ref: str,
) -> tuple[str, str] | None:
    try:
        parsed = parse_public_authority_alias(tool_ref)
    except (TypeError, ValueError):
        return None
    if parsed is None or parsed.field_name != TARGETED_TOOL_REFERENCE_FIELD_NAME:
        return None
    resolved = state.targeted_tool_grant_refs.get(tool_ref)
    if resolved is None:
        return None
    scope_session_id, grant_id = resolved
    codec = get_codec()
    if codec is None or not codec.matches(
        tool_ref,
        grant_id,
        field_name=TARGETED_TOOL_REFERENCE_FIELD_NAME,
        session_id=scope_session_id,
    ):
        return None
    return resolved


async def bind_targeted_tool_grant_use(
    locked: LockedToolGrantState[PreparedAppend],
    request: TargetedToolUseRequest,
    *,
    get_codec: Callable[[], PublicAuthorityAliasCodec],
    observed_at: datetime,
) -> TargetedToolUseResult:
    if type(request) is not TargetedToolUseRequest:
        raise TypeError("request must be a TargetedToolUseRequest.")
    request = TargetedToolUseRequest.model_validate(request.model_dump(mode="python"))
    if observed_at.tzinfo is None or observed_at.utcoffset() is None:
        raise ValueError("observed_at must be timezone-aware.")
    observed_at = observed_at.astimezone(UTC)
    async with locked() as state:
        session = state.sessions.get(request.session_id)
        if session is None:
            raise KeyError(f"Session not found: {request.session_id}")
        if session.run_epoch != request.expected_run_epoch:
            raise SessionRunFenced(
                f"Session source run epoch is stale: expected {request.expected_run_epoch}, "
                f"current {session.run_epoch}."
            )
        if session.status is not SessionStatus.RUNNING:
            raise SessionStatusConflict("Targeted tool use requires a running session.")

        resolved = _resolve_targeted_tool_ref_unlocked(state, get_codec, request.tool_ref)
        if resolved is None:
            try:
                parsed = parse_public_authority_alias(request.tool_ref)
                well_formed = (
                    parsed is not None and parsed.field_name == TARGETED_TOOL_REFERENCE_FIELD_NAME
                )
            except (TypeError, ValueError):
                well_formed = False
            reason = (
                TargetedToolUseRejectionReason.UNKNOWN
                if well_formed
                else TargetedToolUseRejectionReason.MALFORMED
            )
            event = targeted_tool_unresolved_rejection_event(
                request,
                reason=reason,
                timestamp=observed_at,
                agent_name=session.agent_name,
                environment_name=session.environment_name,
            )
            existing = state.event_records_by_id.get((request.session_id, event.id))
            if existing is None:
                prepared = state.prepare_events(session, [event])
                session = state.apply_events(session, prepared)
                state.sessions[request.session_id] = session
            else:
                event = copy_event(existing.event)
            validate_targeted_tool_unresolved_rejection_evidence(
                request,
                reason=reason,
                event=event,
                agent_name=session.agent_name,
                environment_name=session.environment_name,
            )
            return TargetedToolUseResult(
                disposition=TargetedToolUseDisposition.REJECTED,
                reason=reason,
                event=event,
            )
        scope_session_id, grant_id = resolved
        if scope_session_id != request.session_id:
            event = targeted_tool_unresolved_rejection_event(
                request,
                reason=TargetedToolUseRejectionReason.CROSS_SESSION,
                timestamp=observed_at,
                agent_name=session.agent_name,
                environment_name=session.environment_name,
            )
            existing = state.event_records_by_id.get((request.session_id, event.id))
            if existing is None:
                prepared = state.prepare_events(session, [event])
                session = state.apply_events(session, prepared)
                state.sessions[request.session_id] = session
            else:
                event = copy_event(existing.event)
            validate_targeted_tool_unresolved_rejection_evidence(
                request,
                reason=TargetedToolUseRejectionReason.CROSS_SESSION,
                event=event,
                agent_name=session.agent_name,
                environment_name=session.environment_name,
            )
            return TargetedToolUseResult(
                disposition=TargetedToolUseDisposition.REJECTED,
                reason=TargetedToolUseRejectionReason.CROSS_SESSION,
                event=event,
            )
        record = state.targeted_tool_grants.get(grant_id)
        if record is None:
            raise RuntimeError("Targeted tool reference lost its durable grant record.")
        if len(state.targeted_tool_uses.get(grant_id, ())) != record.used_calls:
            raise ValueError("Targeted grant call counter conflicts with durable uses.")
        current_session: Session = session

        def append_once(event: Event) -> Event:
            nonlocal current_session
            existing = state.event_records_by_id.get((request.session_id, event.id))
            if existing is not None:
                return copy_event(existing.event)
            prepared = state.prepare_events(current_session, [event])
            current_session = state.apply_events(current_session, prepared)
            state.sessions[request.session_id] = current_session
            return event

        def rejected(reason: TargetedToolUseRejectionReason) -> TargetedToolUseResult:
            if reason is TargetedToolUseRejectionReason.EXPIRED:
                expiry_event = append_once(
                    targeted_tool_grant_event(
                        record,
                        event_type=EventType.TARGETED_TOOL_GRANT_EXPIRED,
                        timestamp=observed_at,
                        outcome="expired",
                        event_id_suffix="expired",
                        rejection_reason=reason,
                    )
                )
                validate_targeted_tool_grant_lifecycle_event(
                    record,
                    expiry_event,
                    event_type=EventType.TARGETED_TOOL_GRANT_EXPIRED,
                    outcome="expired",
                    event_id_suffix="expired",
                    rejection_reason=reason,
                    require_current_call_count=False,
                )
            event = append_once(
                targeted_tool_use_rejection_event(
                    record,
                    request,
                    reason=reason,
                    timestamp=observed_at,
                )
            )
            validate_targeted_tool_use_rejection_evidence(
                record,
                request,
                reason=reason,
                event=event,
            )
            return TargetedToolUseResult(
                disposition=TargetedToolUseDisposition.REJECTED,
                reason=reason,
                grant=record,
                event=event,
            )

        from cayu.sessions.interactions import INTERACTION_TERMINAL_EVENT_TYPES

        if any(
            event_record.event.type in INTERACTION_TERMINAL_EVENT_TYPES
            for event_record in state.interaction_event_records.get(
                request.session_id,
                {},
            ).get(record.interaction_id, ())
        ):
            return rejected(TargetedToolUseRejectionReason.EXPIRED)
        uses = state.targeted_tool_uses.setdefault(grant_id, {})
        related_by_id = {
            binding.use_id: binding
            for binding in (
                state.targeted_tool_use_by_invocation.get(
                    (request.session_id, request.interaction_id, request.invocation_id)
                ),
                state.targeted_tool_use_by_outer_call.get(
                    (request.session_id, request.interaction_id, request.outer_tool_call_id)
                ),
            )
            if binding is not None
        }
        related = tuple(related_by_id.values())
        if related:
            scope_rejection = targeted_tool_use_scope_rejection_reason(record, request)
            if scope_rejection is not None:
                return rejected(scope_rejection)
            candidate = targeted_tool_use_binding(
                grant_id,
                request,
                bound_at=related[0].bound_at,
            )
            if len(related) == 1 and related[0].model_dump(mode="json") == candidate.model_dump(
                mode="json"
            ):
                event_record = state.event_records_by_id.get(
                    (request.session_id, f"{grant_id}:use:{candidate.use_id}")
                )
                if event_record is None:
                    raise RuntimeError("Targeted tool use lost its durable event evidence.")
                validate_targeted_tool_grant_lifecycle_event(
                    record,
                    event_record.event,
                    event_type=EventType.TARGETED_TOOL_REFERENCE_CONSUMED,
                    outcome=TargetedToolUseDisposition.BOUND.value,
                    event_id_suffix=f"use:{candidate.use_id}",
                    binding=related[0],
                    require_current_call_count=False,
                )
                rejoined_event = append_once(
                    targeted_tool_grant_event(
                        record,
                        event_type=EventType.TARGETED_TOOL_REFERENCE_REJOINED,
                        timestamp=observed_at,
                        outcome=TargetedToolUseDisposition.REJOINED.value,
                        event_id_suffix=f"rejoined:{candidate.use_id}",
                        binding=related[0],
                    )
                )
                return TargetedToolUseResult(
                    disposition=TargetedToolUseDisposition.REJOINED,
                    grant=record,
                    binding=related[0],
                    event=rejoined_event,
                )
            return rejected(TargetedToolUseRejectionReason.ALTERED_REPLAY)

        reason = targeted_tool_use_rejection_reason(
            record,
            request,
            observed_at=observed_at,
        )
        if reason is not None:
            return rejected(reason)
        binding = targeted_tool_use_binding(grant_id, request, bound_at=observed_at)
        updated = TargetedToolGrantRecord.model_validate(
            record.model_copy(update={"used_calls": record.used_calls + 1}).model_dump(
                mode="python"
            )
        )
        event = targeted_tool_grant_event(
            updated,
            event_type=EventType.TARGETED_TOOL_REFERENCE_CONSUMED,
            timestamp=observed_at,
            outcome=TargetedToolUseDisposition.BOUND.value,
            event_id_suffix=f"use:{binding.use_id}",
            binding=binding,
        )
        prepared_append = state.prepare_events(session, [event])
        uses[binding.use_id] = binding
        state.targeted_tool_use_by_invocation[
            (binding.session_id, binding.interaction_id, binding.invocation_id)
        ] = binding
        state.targeted_tool_use_by_outer_call[
            (binding.session_id, binding.interaction_id, binding.outer_tool_call_id)
        ] = binding
        state.targeted_tool_grants[grant_id] = updated
        state.sessions[request.session_id] = state.apply_events(
            session,
            prepared_append,
        )
        return TargetedToolUseResult(
            disposition=TargetedToolUseDisposition.BOUND,
            grant=updated,
            binding=binding,
            event=event,
        )


async def revoke_targeted_tool_grant(
    locked: LockedToolGrantState[PreparedAppend],
    tool_ref: str,
    *,
    get_codec: Callable[[], PublicAuthorityAliasCodec],
    session_id: str,
    expected_run_epoch: int,
    reason: str,
    revoked_at: datetime,
) -> TargetedToolGrantRecord | None:
    session_id = require_clean_nonblank(session_id, "session_id")
    if type(expected_run_epoch) is not int or expected_run_epoch < 0:
        raise ValueError("expected_run_epoch must be a non-negative integer.")
    reason = validate_targeted_tool_grant_revocation_reason(reason)
    if revoked_at.tzinfo is None or revoked_at.utcoffset() is None:
        raise ValueError("revoked_at must be timezone-aware.")
    revoked_at = revoked_at.astimezone(UTC)
    async with locked() as state:
        session = state.sessions.get(session_id)
        if session is None:
            raise KeyError(f"Session not found: {session_id}")
        if session.run_epoch != expected_run_epoch:
            raise SessionRunFenced(
                f"Session source run epoch is stale: expected {expected_run_epoch}, "
                f"current {session.run_epoch}."
            )
        resolved = _resolve_targeted_tool_ref_unlocked(state, get_codec, tool_ref)
        if resolved is None or resolved[0] != session_id:
            return None
        record = state.targeted_tool_grants.get(resolved[1])
        if record is None:
            raise RuntimeError("Targeted tool reference lost its durable grant record.")
        if len(state.targeted_tool_uses.get(record.grant_id, ())) != record.used_calls:
            raise ValueError("Targeted grant call counter conflicts with durable uses.")
        if record.revoked_at is not None:
            if record.revocation_reason != reason:
                raise ValueError("Targeted grant was revoked with a different reason.")
            event_record = state.event_records_by_id.get((session_id, f"{record.grant_id}:revoked"))
            if event_record is None:
                raise RuntimeError("Targeted grant revocation lost its durable event evidence.")
            validate_targeted_tool_grant_revocation_evidence(
                record,
                event_record.event,
            )
            return copy_targeted_tool_grant_record(record)
        for owner in state.session_closure_progress.owners((session_id,)):
            closure_records._check_closure_lineage_owner(owner, (session_id,))
        uses = state.targeted_tool_uses.get(record.grant_id, {})
        if any(binding.bound_at > revoked_at for binding in uses.values()):
            raise ValueError("revoked_at cannot precede a bound targeted tool use.")
        updated = TargetedToolGrantRecord.model_validate(
            record.model_copy(
                update={"revoked_at": revoked_at, "revocation_reason": reason}
            ).model_dump(mode="python")
        )
        event = targeted_tool_grant_event(
            updated,
            event_type=EventType.TARGETED_TOOL_GRANT_REVOKED,
            timestamp=revoked_at,
            outcome="revoked",
            event_id_suffix="revoked",
        )
        prepared_append = state.prepare_events(session, [event])
        state.targeted_tool_grants[record.grant_id] = updated
        state.sessions[session_id] = state.apply_events(
            session,
            prepared_append,
        )
        return copy_targeted_tool_grant_record(updated)


async def reconstruct_targeted_tool_grants(
    locked: LockedToolGrantState[PreparedAppend],
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
) -> TargetedToolGrantReconstructionResult:
    session_id = require_clean_nonblank(session_id, "session_id")
    interaction_id = require_clean_nonblank(interaction_id, "interaction_id")
    if type(expected_run_epoch) is not int or expected_run_epoch < 0:
        raise ValueError("expected_run_epoch must be a non-negative integer.")
    if observed_at.tzinfo is None or observed_at.utcoffset() is None:
        raise ValueError("observed_at must be timezone-aware.")
    observed_at = observed_at.astimezone(UTC)
    async with locked() as state:
        session = state.sessions.get(session_id)
        if session is None:
            raise KeyError(f"Session not found: {session_id}")
        if session.run_epoch != expected_run_epoch:
            raise SessionRunFenced(
                f"Session source run epoch is stale: expected {expected_run_epoch}, "
                f"current {session.run_epoch}."
            )
        if session.status is not SessionStatus.RUNNING:
            raise SessionStatusConflict("Grant reconstruction requires a running session.")
        records = sorted(
            (
                state.targeted_tool_grants[grant_id]
                for grant_id in state.targeted_tool_grant_ids_by_interaction.get(
                    (session_id, interaction_id),
                    (),
                )
            ),
            key=lambda candidate: (candidate.issued_at, candidate.grant_id),
        )
        if len(records) > TARGETED_TOOL_GRANT_MAX_REQUESTS:
            raise ValueError("Targeted grant interaction exceeds its bounded count.")
        if any(
            len(state.targeted_tool_uses.get(record.grant_id, ())) != record.used_calls
            for record in records
        ):
            raise ValueError("Targeted grant call counter conflicts with durable uses.")

        current_session: Session = session

        def append_once(event: Event) -> Event:
            nonlocal current_session
            existing = state.event_records_by_id.get((session_id, event.id))
            if existing is not None:
                return copy_event(existing.event)
            prepared = state.prepare_events(current_session, [event])
            current_session = state.apply_events(current_session, prepared)
            state.sessions[session_id] = current_session
            return event

        from cayu.sessions.interactions import INTERACTION_TERMINAL_EVENT_TYPES

        interaction_records = state.interaction_event_records.get(session_id, {}).get(
            interaction_id,
            (),
        )
        interaction_started = next(
            (
                event_record.event
                for event_record in interaction_records
                if event_record.event.type is EventType.INTERACTION_STARTED
            ),
            None,
        )
        if interaction_started is None:
            raise RuntimeError("Targeted grant reconstruction lost interaction admission.")
        validate_targeted_tool_grant_batch_evidence(tuple(records), interaction_started)
        interaction_ended = any(
            event_record.event.type in INTERACTION_TERMINAL_EVENT_TYPES
            for event_record in interaction_records
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
                    persisted_expiry = append_once(
                        targeted_tool_grant_event(
                            record,
                            event_type=EventType.TARGETED_TOOL_GRANT_EXPIRED,
                            timestamp=observed_at,
                            outcome="expired",
                            event_id_suffix="expired",
                            rejection_reason=reason,
                        )
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
            persisted = append_once(event)
            validate_targeted_tool_grant_lifecycle_event(
                record,
                persisted,
                event_type=EventType.TARGETED_TOOL_GRANT_RECONSTRUCTED,
                outcome="reconstructed" if reason is None else "rejected",
                event_id_suffix=(
                    "reconstructed" if reason is None else f"reconstruction-rejected:{reason.value}"
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

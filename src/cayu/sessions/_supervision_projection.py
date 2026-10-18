"""Classify durable runtime events into settled supervision touches.

Every record is decided by one anchor event of a rare event type. Secondary
events of the same resolution are read only as evidence, so each touch is
counted once, at its first anchor. A recovery is recognized because its
``session.resumed`` and its manual-recovery tool terminal are appended in one
contiguous batch; a resumed user-input pause without that terminal is an answer.

Evidence reads are proportional to one touch, never to a session's history: each
lookup pages through one event type near its anchor and stops at the first match
or after a bounded number of pages. A session holds at most one pending approval
or question, so a resolution's request is the nearest matching request before it.
"""

from __future__ import annotations

import re
from collections.abc import AsyncIterator, Callable, Iterable
from datetime import datetime
from hashlib import sha256
from typing import TYPE_CHECKING, Any

from pydantic import ValidationError

from cayu._validation import canonical_durable_json_bytes, json_utf8_size_within_limit
from cayu.approvals.actors import ResolutionActor
from cayu.approvals.authorization_grants import TOOL_AUTHORIZATION_GRANT_METADATA_KEY
from cayu.approvals.revisions import ApprovalArgumentsRevision, ApprovalRevision
from cayu.budgets.usage import UsageMetrics
from cayu.events import Event, EventType
from cayu.execution_profiles import ExecutionProfileDecisionKind
from cayu.releases import ReleaseIdentity, recorded_release_identity
from cayu.sessions import _access_rules as session_access_rules
from cayu.sessions import queries as session_query_rules
from cayu.sessions._execution_profile_checkpoint import (
    execution_profile_baseline_from_session_metadata,
)
from cayu.sessions._fork_controller_bounds import _ForkControllerReadBounds
from cayu.sessions._terminal_evidence import (
    _INTERRUPTION_TYPE_OPERATOR_REQUESTED,
    _INTERRUPTION_TYPE_USER_INPUT_REQUIRED,
    INTERRUPTION_ORIGIN_BACKGROUND_CASCADE,
    INTERRUPTION_ORIGIN_PAYLOAD_KEY,
)
from cayu.sessions.event_queries import EventOrder, EventQuery, EventQueryResultTooLarge
from cayu.sessions.fork_overlays import resolve_fork_lineage_async
from cayu.sessions.pending_action_contracts import human_attention_id
from cayu.sessions.queries import SessionQuery, session_query_from_aggregate_filter
from cayu.sessions.records import EventRecord, PendingActionKind, Session
from cayu.sessions.supervision_events import (
    SupervisionAttentionState,
    SupervisionEvent,
    SupervisionEventContent,
    SupervisionEventKind,
    SupervisionEventOutcome,
    SupervisionEventPage,
    SupervisionEventQuery,
    SupervisionEventResultTooLarge,
    SupervisionGrantRole,
    SupervisionMessageOrigin,
    decode_supervision_event_cursor,
    encode_supervision_event_cursor,
)

if TYPE_CHECKING:
    from cayu.sessions.store import SessionStore

_EVIDENCE_PAGE_LIMIT = 50
_EVIDENCE_LOOKUP_PAGES = 20
_CANDIDATE_BATCH_LIMIT = 500
_CANDIDATE_SCAN_FACTOR = 10
_EVIDENCE_READ_MAX_BYTES = 4_194_304
_LOWER_HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ADMITTED_PROFILE_DECISIONS = frozenset(
    {
        ExecutionProfileDecisionKind.EXACT_REUSE.value,
        ExecutionProfileDecisionKind.COMPATIBLE_REUSE.value,
        ExecutionProfileDecisionKind.ADOPTED.value,
    }
)
_TOOL_TERMINAL_TYPES = (EventType.TOOL_CALL_COMPLETED, EventType.TOOL_CALL_FAILED)
# The events that begin a process's work on a session, and so record its Release.
_RELEASE_MARKER_TYPES = (
    EventType.SESSION_STARTED,
    EventType.SESSION_RESUMED,
    EventType.MODEL_STARTED,
)
_MESSAGE_OUTCOMES = {
    EventType.SESSION_MESSAGE_DELIVERED: SupervisionEventOutcome.DELIVERED,
    EventType.SESSION_MESSAGE_WITHDRAWN: SupervisionEventOutcome.WITHDRAWN,
    EventType.SESSION_MESSAGE_QUARANTINED: SupervisionEventOutcome.QUARANTINED,
    EventType.SESSION_MESSAGE_STALE: SupervisionEventOutcome.STALE,
    EventType.SESSION_MESSAGE_EXPIRED: SupervisionEventOutcome.EXPIRED,
}
_ANCHOR_TYPES = {
    SupervisionEventKind.APPROVE: (EventType.TOOL_CALL_APPROVED,),
    SupervisionEventKind.EDIT: (EventType.TOOL_CALL_APPROVED,),
    SupervisionEventKind.DENY: (EventType.TOOL_CALL_APPROVAL_DENIED,),
    SupervisionEventKind.EXPIRE: (EventType.TOOL_CALL_APPROVAL_EXPIRED,),
    SupervisionEventKind.ANSWER: (EventType.SESSION_RESUMED,),
    SupervisionEventKind.RECOVER: (EventType.SESSION_RESUMED,),
    SupervisionEventKind.INTERRUPT: (EventType.SESSION_INTERRUPTED,),
    SupervisionEventKind.STEER: tuple(_MESSAGE_OUTCOMES),
}


class _SessionEvidence:
    """Bounded session-scoped lookups that classification draws on."""

    def __init__(self, store: SessionStore) -> None:
        self._store = store
        self._sessions: dict[str, Session | None] = {}
        self._forked: dict[str, bool] = {}

    async def session(self, session_id: str) -> Session | None:
        if session_id not in self._sessions:
            self._sessions[session_id] = await self._store.load(session_id)
        return self._sessions[session_id]

    async def forked(self, session_id: str) -> bool:
        """Whether the session is in a forked lineage: a fork or created under one.

        A session without a lineage marker is settled through its parents, as
        the stores' SQL does, and one its parents cannot settle counts as forked.
        """

        if session_id not in self._forked:
            session = await self.session(session_id)
            self._forked[session_id] = (
                session is not None
                and (await resolve_fork_lineage_async(session, self.session)).forked
            ) or await self.first_after(session_id, (EventType.SESSION_FORKED,), 0) is not None
        return self._forked[session_id]

    async def next_event(self, session_id: str, sequence: int) -> EventRecord | None:
        records = await self._page(
            EventQuery(session_id=session_id, after_sequence=sequence, limit=1)
        )
        return records[0] if records else None

    async def latest_before(
        self,
        session_id: str,
        event_types: tuple[EventType, ...],
        before: int,
        predicate: Callable[[Event], bool] = lambda event: True,
    ) -> EventRecord | None:
        async for record in self._scan(session_id, event_types, before=before, descending=True):
            if predicate(record.event):
                return record
        return None

    async def first_after(
        self,
        session_id: str,
        event_types: tuple[EventType, ...],
        after: int,
        predicate: Callable[[Event], bool] = lambda event: True,
        *,
        before: int | None = None,
    ) -> EventRecord | None:
        async for record in self._scan(session_id, event_types, after=after, before=before):
            if predicate(record.event):
                return record
        return None

    async def all_after(
        self,
        session_id: str,
        event_types: tuple[EventType, ...],
        after: int,
        predicate: Callable[[Event], bool],
    ) -> list[EventRecord]:
        return [
            record
            async for record in self._scan(session_id, event_types, after=after)
            if predicate(record.event)
        ]

    async def nearest_before(
        self, session_id: str, event_types: tuple[EventType, ...], before: int
    ) -> EventRecord | None:
        records = await self._page(
            EventQuery(
                session_id=session_id,
                event_types=event_types,
                before_sequence=before,
                limit=1,
                order_by=EventOrder.SEQUENCE_DESC,
            )
        )
        return records[0] if records else None

    async def model_completions(self, session_id: str, model_step_id: str) -> list[EventRecord]:
        return await self._page(
            EventQuery(
                session_id=session_id,
                event_types=(EventType.MODEL_COMPLETED,),
                model_step_id=model_step_id,
                limit=_EVIDENCE_PAGE_LIMIT,
            )
        )

    async def _scan(
        self,
        session_id: str,
        event_types: tuple[EventType, ...],
        *,
        after: int | None = None,
        before: int | None = None,
        descending: bool = False,
    ) -> AsyncIterator[EventRecord]:
        """Yield matching events in order, at most ``_EVIDENCE_LOOKUP_PAGES`` pages."""

        lower, upper = after, before
        for _page_index in range(_EVIDENCE_LOOKUP_PAGES):
            records = await self._page(
                EventQuery(
                    session_id=session_id,
                    event_types=event_types,
                    after_sequence=lower,
                    before_sequence=upper,
                    limit=_EVIDENCE_PAGE_LIMIT,
                    order_by=EventOrder.SEQUENCE_DESC if descending else EventOrder.SEQUENCE_ASC,
                )
            )
            for record in records:
                yield record
            if len(records) < _EVIDENCE_PAGE_LIMIT:
                return
            if descending:
                upper = records[-1].sequence
            else:
                lower = records[-1].sequence

    async def _page(self, query: EventQuery) -> list[EventRecord]:
        try:
            return await self._store.query_events_bounded(query, max_bytes=_EVIDENCE_READ_MAX_BYTES)
        except EventQueryResultTooLarge:
            if query.limit == 1:
                raise SupervisionEventResultTooLarge(
                    "One supervision evidence event exceeds the read limit."
                ) from None
        # Large payloads: continue the same ordered read one event at a time.
        records: list[EventRecord] = []
        cursor = query
        for _index in range(query.limit):
            single = await self._page(cursor.model_copy(update={"limit": 1}))
            if not single:
                break
            records.extend(single)
            boundary = (
                {"before_sequence": single[0].sequence}
                if query.order_by is EventOrder.SEQUENCE_DESC
                else {"after_sequence": single[0].sequence}
            )
            cursor = cursor.model_copy(update=boundary)
        return records


def supervision_candidate_query(query: SupervisionEventQuery) -> EventQuery:
    """Cross-session scan over the rare anchor event types the query can classify."""

    kinds = query.kinds or tuple(SupervisionEventKind)
    return EventQuery(
        event_types=tuple(dict.fromkeys(t for kind in kinds for t in _ANCHOR_TYPES[kind])),
        agent_name=query.sessions.agent_name,
        causal_budget_id=query.sessions.causal_budget_id,
        since=query.start_at,
        until=query.end_at,
        order_by=EventOrder.SEQUENCE_ASC,
    )


async def query_supervision_events(
    store: SessionStore,
    query: SupervisionEventQuery,
    *,
    include_content: bool,
) -> SupervisionEventPage:
    """Return one page of settled supervision touches for ``query``."""

    page, _anchors = await supervision_page_with_anchors(
        store, query, include_content=include_content
    )
    return page


async def supervision_page_with_anchors(
    store: SessionStore,
    query: SupervisionEventQuery,
    *,
    include_content: bool,
    origin_tenant: str | None = None,
) -> tuple[SupervisionEventPage, dict[int, EventRecord]]:
    """Return one page plus the anchor record behind each event, keyed by sequence.

    Exposure layers use the anchors to alias only linkage the anchor itself holds.
    ``origin_tenant`` restricts the page to sessions whose recorded invocation
    origin carries exactly that tenant; it is applied before paging so the
    cursor and ``has_more`` stay exact for a tenant-bound reader.
    """

    audience = _cursor_audience(origin_tenant)
    after_sequence = (
        decode_supervision_event_cursor(query, include_content=include_content, audience=audience)
        or 0
    )
    kinds = query.kinds or tuple(SupervisionEventKind)
    candidate_query = supervision_candidate_query(query)
    session_filter = session_query_from_aggregate_filter(query.sessions)
    evidence = _SessionEvidence(store)
    events: list[SupervisionEvent] = []
    anchors: dict[int, EventRecord] = {}
    resume_after = after_sequence
    budget = query.limit * _CANDIDATE_SCAN_FACTOR
    exhausted = False
    while len(events) < query.limit and budget > 0:
        batch_limit = min(budget, _CANDIDATE_BATCH_LIMIT)
        candidates = await store.query_events(
            candidate_query.model_copy(
                update={"after_sequence": resume_after, "limit": batch_limit}
            )
        )
        budget -= len(candidates)
        store_drained = len(candidates) < batch_limit
        for index, record in enumerate(candidates):
            resume_after = record.sequence
            event = await _candidate_event(
                record,
                query,
                kinds,
                session_filter,
                evidence=evidence,
                content=include_content,
                origin_tenant=origin_tenant,
            )
            if event is not None:
                events.append(event)
                anchors[record.sequence] = record
            if len(events) == query.limit:
                exhausted = store_drained and index == len(candidates) - 1
                break
        else:
            exhausted = store_drained
        if exhausted:
            break
    page = SupervisionEventPage(
        events=tuple(events),
        next_cursor=encode_supervision_event_cursor(
            query, after_sequence=resume_after, include_content=include_content, audience=audience
        ),
        has_more=not exhausted,
    )
    if not json_utf8_size_within_limit(page.model_dump(mode="json"), query.max_bytes):
        raise SupervisionEventResultTooLarge(
            "Supervision event page exceeds max_bytes; request a smaller limit."
        )
    return page, anchors


def _cursor_audience(origin_tenant: str | None) -> str | None:
    """Name the reader a page is produced for: its access scope and verified tenant."""

    parts: list[str] = []
    bounds = session_access_rules._query_bounds.get()
    if type(bounds) is _ForkControllerReadBounds:
        parts.append(bounds.cursor_audience())
    elif type(bounds) is session_access_rules._SessionAccessBounds:
        binding = bounds.binding
        parts.append(
            "scope:"
            + (
                repr(bounds.admitted)
                if binding is None
                else f"{binding.authority}:{binding.subject}:{binding.admitted_json}"
            )
        )
    elif bounds is not None:
        raise TypeError("Unknown session read bound.")
    if origin_tenant is not None:
        parts.append("tenant:" + origin_tenant)
    if not parts:
        return None
    return sha256("\n".join(parts).encode("utf-8")).hexdigest()


async def _candidate_event(
    record: EventRecord,
    query: SupervisionEventQuery,
    kinds: tuple[SupervisionEventKind, ...],
    session_filter: SessionQuery,
    *,
    evidence: _SessionEvidence,
    content: bool,
    origin_tenant: str | None,
) -> SupervisionEvent | None:
    session = await evidence.session(record.event.session_id)
    if session is None or not session_query_rules._session_matches(session, session_filter):
        return None
    if origin_tenant is not None and session.invocation.origin.tenant != origin_tenant:
        return None
    forked = await evidence.forked(session.id)
    if forked and query.exclude_forked_sessions:
        return None
    event = await _classify(
        record, session=session, forked=forked, evidence=evidence, content=content
    )
    if event is None or not _selected(event, query, kinds):
        return None
    return event


def _selected(
    event: SupervisionEvent,
    query: SupervisionEventQuery,
    kinds: tuple[SupervisionEventKind, ...],
) -> bool:
    if event.kind not in kinds:
        return False
    if (
        query.execution_profile_fingerprint is not None
        and event.execution_profile_fingerprint != query.execution_profile_fingerprint
    ):
        return False
    if query.release_id is not None and event.release_id != query.release_id:
        return False
    if query.actor_sources:
        return event.actor is not None and event.actor.source in query.actor_sources
    return True


async def _classify(
    record: EventRecord,
    *,
    session: Session,
    forked: bool,
    evidence: _SessionEvidence,
    content: bool,
) -> SupervisionEvent | None:
    event_type = record.event.type
    if event_type == EventType.TOOL_CALL_APPROVED:
        # The record decides between approve and edit from the whole approval.
        return await _approval_record(
            record,
            kind=SupervisionEventKind.APPROVE,
            outcome=SupervisionEventOutcome.APPROVED,
            counts=lambda payload: True,
            session=session,
            forked=forked,
            evidence=evidence,
            content=content,
        )
    if event_type == EventType.TOOL_CALL_APPROVAL_DENIED:
        return await _approval_record(
            record,
            kind=SupervisionEventKind.DENY,
            outcome=SupervisionEventOutcome.DENIED,
            counts=lambda payload: (
                payload.get("approval_required") is True and payload.get("expired") is not True
            ),
            session=session,
            forked=forked,
            evidence=evidence,
            content=content,
        )
    if event_type == EventType.TOOL_CALL_APPROVAL_EXPIRED:
        return await _approval_record(
            record,
            kind=SupervisionEventKind.EXPIRE,
            outcome=SupervisionEventOutcome.EXPIRED,
            counts=lambda payload: True,
            session=session,
            forked=forked,
            evidence=evidence,
            content=content,
        )
    if event_type == EventType.SESSION_RESUMED:
        return await _resumed_record(
            record, session=session, forked=forked, evidence=evidence, content=content
        )
    if event_type == EventType.SESSION_INTERRUPTED:
        return await _interrupt_record(record, session=session, forked=forked, evidence=evidence)
    if event_type in _MESSAGE_OUTCOMES:
        return await _steer_record(record, session=session, forked=forked, evidence=evidence)
    return None


async def _approval_record(
    record: EventRecord,
    *,
    kind: SupervisionEventKind,
    outcome: SupervisionEventOutcome,
    counts: Callable[[dict[str, Any]], bool],
    session: Session,
    forked: bool,
    evidence: _SessionEvidence,
    content: bool,
) -> SupervisionEvent | None:
    payload = record.event.payload
    approval_id = _text(payload, "approval_id")
    if approval_id is None or not counts(payload):
        return None

    def same_approval(event: Event) -> bool:
        return _text(event.payload, "approval_id") == approval_id

    request = await evidence.latest_before(
        session.id, (EventType.TOOL_CALL_APPROVAL_REQUESTED,), record.sequence, same_approval
    )
    window_start = request.sequence if request is not None else 0
    anchor_type = (EventType(record.event.type),)
    if await evidence.first_after(
        session.id,
        anchor_type,
        window_start,
        lambda event: same_approval(event) and counts(event.payload),
        before=record.sequence,
    ):
        return None
    peers = await evidence.all_after(
        session.id,
        (EventType.TOOL_CALL_APPROVAL_DENIED,)
        if kind is SupervisionEventKind.EXPIRE
        else anchor_type,
        record.sequence - 1,
        lambda event: (
            same_approval(event)
            and (
                event.payload.get("expired") is True
                if kind is SupervisionEventKind.EXPIRE
                else counts(event.payload)
            )
        ),
    )
    tool_call_ids = _distinct(_text(peer.event.payload, "tool_call_id") for peer in peers) or (
        _distinct([_text(payload, "tool_call_id")])
    )
    approved = kind is SupervisionEventKind.APPROVE
    # Runtime evidence of changed arguments sits on the gating call's own
    # approval event, which need not be the first in the round; an
    # application's declared revision is a separate, weaker input.
    arguments_revision = (
        next(
            (
                revision
                for peer in peers
                if (revision := _arguments_revision(peer.event.payload)) is not None
            ),
            None,
        )
        if approved
        else None
    )
    declared_revision = _approval_revision(payload) if approved else None
    if arguments_revision is not None or (
        declared_revision is not None and declared_revision.edited
    ):
        kind = SupervisionEventKind.EDIT
    grant_id, grant_role = _authorization_grant(request, payload if approved else None)
    return await _record(
        record,
        kind=kind,
        outcome=outcome,
        session=session,
        forked=forked,
        evidence=evidence,
        identity=(approval_id,),
        tool_name=(request.event.tool_name if request is not None else None)
        or record.event.tool_name,
        tool_call_ids=tool_call_ids,
        approval_id=approval_id,
        actor=payload.get("resolved_by"),
        reason=None if kind is SupervisionEventKind.EXPIRE else payload.get("reason"),
        requested_at=request.event.timestamp if request is not None else None,
        release_position=request.sequence if request is not None else None,
        attention_state=_APPROVAL_ATTENTION_STATES[kind],
        approval_revision=declared_revision,
        arguments_revision=arguments_revision,
        authorization_grant_id=grant_id,
        authorization_grant_role=grant_role,
        content=(
            SupervisionEventContent(
                tool_arguments=await _published_arguments(
                    evidence, session.id, approval_id=approval_id, after=window_start
                ),
                approved_arguments=(
                    await _published_arguments(
                        evidence,
                        session.id,
                        approval_id=approval_id,
                        after=window_start,
                        field="approved_arguments",
                    )
                    if arguments_revision is not None
                    else None
                ),
            )
            if content
            else None
        ),
    )


async def _resumed_record(
    record: EventRecord,
    *,
    session: Session,
    forked: bool,
    evidence: _SessionEvidence,
    content: bool,
) -> SupervisionEvent | None:
    successor = await evidence.next_event(session.id, record.sequence)
    if (
        successor is not None
        and successor.event.type in _TOOL_TERMINAL_TYPES
        and successor.event.payload.get("manual_recovery") is True
    ):
        return await _recovery_record(
            record, successor, session=session, forked=forked, evidence=evidence
        )
    payload = record.event.payload
    input_id = _text(payload, "input_id")
    if (
        payload.get("interruption_type") != _INTERRUPTION_TYPE_USER_INPUT_REQUIRED
        or input_id is None
    ):
        return None

    def same_input(event: Event) -> bool:
        return _text(event.payload, "input_id") == input_id

    request = await evidence.latest_before(
        session.id, (EventType.SESSION_AWAITING_USER_INPUT,), record.sequence, same_input
    )
    if await evidence.first_after(
        session.id,
        (EventType.SESSION_RESUMED,),
        request.sequence if request is not None else 0,
        same_input,
        before=record.sequence,
    ):
        return None
    answered = (
        await evidence.first_after(session.id, _TOOL_TERMINAL_TYPES, record.sequence, same_input)
        if content
        else None
    )
    question = request.event.payload.get("question") if request is not None else None
    options = request.event.payload.get("options") if request is not None else None
    return await _record(
        record,
        kind=SupervisionEventKind.ANSWER,
        outcome=SupervisionEventOutcome.ANSWERED,
        session=session,
        forked=forked,
        evidence=evidence,
        identity=(input_id,),
        tool_name=request.event.tool_name if request is not None else None,
        tool_call_ids=_distinct([_text(payload, "tool_call_id")]),
        input_id=input_id,
        actor=payload.get("resolved_by"),
        requested_at=request.event.timestamp if request is not None else None,
        release_position=request.sequence if request is not None else None,
        attention_state="resolved",
        content=(
            SupervisionEventContent(
                question=question if type(question) is str else None,
                options=(
                    tuple(options)
                    if type(options) is list and all(type(o) is str for o in options)
                    else None
                ),
                answer=answered.event.payload.get("result") if answered is not None else None,
            )
            if content
            else None
        ),
    )


async def _recovery_record(
    record: EventRecord,
    terminal: EventRecord,
    *,
    session: Session,
    forked: bool,
    evidence: _SessionEvidence,
) -> SupervisionEvent | None:
    payload = terminal.event.payload
    tool_round_id = _text(payload, "tool_round_id")
    tool_call_id = _text(payload, "tool_call_id")
    if tool_round_id is None or tool_call_id is None:
        return None

    def same_call(event: Event) -> bool:
        return (
            _text(event.payload, "tool_round_id") == tool_round_id
            and _text(event.payload, "tool_call_id") == tool_call_id
        )

    started = await evidence.latest_before(
        session.id, (EventType.TOOL_CALL_STARTED,), record.sequence, same_call
    )
    if await evidence.first_after(
        session.id,
        _TOOL_TERMINAL_TYPES,
        started.sequence if started is not None else 0,
        lambda event: event.payload.get("manual_recovery") is True and same_call(event),
        before=terminal.sequence,
    ):
        return None
    approval_id = _text(payload, "approval_id")
    input_id = _text(payload, "input_id")

    def manual_handoff(event: Event) -> bool:
        # The same evidence pending-action reads use to open a manual-recovery
        # request in place of the approval or question that started the round.
        handoff = event.payload
        if handoff.get("manual_recovery_required") is not True:
            return False
        handoff_call = _text(handoff, "tool_call_id") or _text(
            _object(handoff, "user_input") or {}, "tool_call_id"
        )
        return handoff_call == tool_call_id and _text(handoff, "tool_round_id") == tool_round_id

    manual_request = await evidence.first_after(
        session.id,
        (EventType.SESSION_INTERRUPTED,),
        started.sequence if started is not None else 0,
        manual_handoff,
        before=record.sequence,
    )
    request = None
    if approval_id is not None:
        request = await evidence.latest_before(
            session.id,
            (EventType.TOOL_CALL_APPROVAL_REQUESTED,),
            record.sequence,
            lambda event: _text(event.payload, "approval_id") == approval_id,
        )
    elif input_id is not None:
        request = await evidence.latest_before(
            session.id,
            (EventType.SESSION_AWAITING_USER_INPUT,),
            record.sequence,
            lambda event: _text(event.payload, "input_id") == input_id,
        )
    return await _record(
        record,
        kind=SupervisionEventKind.RECOVER,
        outcome=(
            SupervisionEventOutcome.COMPLETED
            if terminal.event.type == EventType.TOOL_CALL_COMPLETED
            else SupervisionEventOutcome.FAILED
        ),
        session=session,
        forked=forked,
        evidence=evidence,
        identity=(tool_round_id, tool_call_id),
        tool_name=terminal.event.tool_name,
        tool_call_ids=(tool_call_id,),
        approval_id=approval_id,
        input_id=input_id,
        step_payload=payload,
        fingerprint_payload=payload,
        actor=payload.get("resolved_by"),
        reason=payload.get("reason"),
        requested_at=request.event.timestamp if request is not None else None,
        release_position=next(
            (origin.sequence for origin in (request, started) if origin is not None), None
        ),
        attention_kind=PendingActionKind.MANUAL_RECOVERY if manual_request is not None else None,
        attention_state=_recovery_attention_state(terminal.event),
    )


_INTERRUPT_LINKAGE_KEYS = (
    "approval_close_intent",
    "user_input_supersession_intent",
    "ambiguous_user_input_supersession_intent",
)


async def _interrupt_record(
    record: EventRecord,
    *,
    session: Session,
    forked: bool,
    evidence: _SessionEvidence,
) -> SupervisionEvent | None:
    payload = record.event.payload
    request_id = _text(payload, "interruption_request_id")
    if (
        payload.get("interruption_type") != _INTERRUPTION_TYPE_OPERATOR_REQUESTED
        or payload.get(INTERRUPTION_ORIGIN_PAYLOAD_KEY) == INTERRUPTION_ORIGIN_BACKGROUND_CASCADE
        or request_id is None
    ):
        return None
    # One interruption request settles within one run; look back to its start only.
    run_start = await evidence.latest_before(
        session.id, (EventType.SESSION_RESUMED,), record.sequence
    )
    if await evidence.first_after(
        session.id,
        (EventType.SESSION_INTERRUPTED,),
        run_start.sequence if run_start is not None else 0,
        lambda event: _text(event.payload, "interruption_request_id") == request_id,
        before=record.sequence,
    ):
        return None
    closed = next(
        (linkage for key in _INTERRUPT_LINKAGE_KEYS if (linkage := _object(payload, key))),
        {},
    )
    return await _record(
        record,
        kind=SupervisionEventKind.INTERRUPT,
        outcome=SupervisionEventOutcome.INTERRUPTED,
        session=session,
        forked=forked,
        evidence=evidence,
        identity=(request_id,),
        tool_name=record.event.tool_name,
        tool_call_ids=_distinct([_text(closed, "tool_call_id")]),
        approval_id=_text(closed, "approval_id"),
        input_id=_text(closed, "input_id"),
        interruption_request_id=request_id,
        step_payload=closed,
        actor=payload.get("requested_by"),
        reason=payload.get("reason"),
        release_position=record.sequence,
    )


async def _steer_record(
    record: EventRecord,
    *,
    session: Session,
    forked: bool,
    evidence: _SessionEvidence,
) -> SupervisionEvent | None:
    queue_id = _text(record.event.payload, "queue_id")
    if queue_id is None:
        return None

    def same_message(event: Event) -> bool:
        return _text(event.payload, "queue_id") == queue_id

    queued = await evidence.latest_before(
        session.id, (EventType.SESSION_MESSAGE_QUEUED,), record.sequence, same_message
    )
    if await evidence.first_after(
        session.id,
        tuple(_MESSAGE_OUTCOMES),
        queued.sequence if queued is not None else 0,
        same_message,
        before=record.sequence,
    ):
        return None
    queued_payload = queued.event.payload if queued is not None else {}
    if "source" in queued_payload:
        origin = SupervisionMessageOrigin.SESSION
    elif "peer_occurrence_id" in queued_payload:
        origin = SupervisionMessageOrigin.PEER
    elif type(queued_payload.get("actor")) is dict:
        origin = SupervisionMessageOrigin.ACTOR
    else:
        origin = SupervisionMessageOrigin.UNKNOWN
    return await _record(
        record,
        kind=SupervisionEventKind.STEER,
        outcome=_MESSAGE_OUTCOMES[EventType(record.event.type)],
        session=session,
        forked=forked,
        evidence=evidence,
        identity=(queue_id,),
        queue_id=queue_id,
        message_origin=origin,
        step_payload={},
        actor=queued_payload.get("actor"),
        requested_at=queued.event.timestamp if queued is not None else None,
        release_position=record.sequence,
    )


async def _record(
    record: EventRecord,
    *,
    kind: SupervisionEventKind,
    outcome: SupervisionEventOutcome,
    session: Session,
    forked: bool,
    evidence: _SessionEvidence,
    identity: tuple[str, ...],
    actor: object,
    release_position: int | None,
    tool_name: str | None = None,
    tool_call_ids: tuple[str, ...] = (),
    approval_id: str | None = None,
    input_id: str | None = None,
    queue_id: str | None = None,
    interruption_request_id: str | None = None,
    message_origin: SupervisionMessageOrigin | None = None,
    step_payload: dict[str, Any] | None = None,
    fingerprint_payload: dict[str, Any] | None = None,
    reason: object = None,
    requested_at: datetime | None = None,
    attention_kind: PendingActionKind | None = None,
    attention_state: SupervisionAttentionState | None = None,
    approval_revision: ApprovalRevision | None = None,
    arguments_revision: ApprovalArgumentsRevision | None = None,
    authorization_grant_id: str | None = None,
    authorization_grant_role: SupervisionGrantRole | None = None,
    content: SupervisionEventContent | None = None,
) -> SupervisionEvent:
    anchor = record.event
    step = anchor.payload if step_payload is None else step_payload
    model_step_id = _text(step, "model_step_id")
    model_attempt_id = _text(step, "model_attempt_id")
    tool_round_id = _text(step, "tool_round_id")
    attention_id = _attention_id(
        kind,
        session,
        attention_kind=attention_kind,
        approval_id=approval_id,
        input_id=input_id,
        tool_round_id=tool_round_id,
        tool_call_ids=tool_call_ids,
    )
    # A settled request belongs to the invocation that proposed it, so callers pass
    # the request's position; interrupts and steering pass their own.
    release = (
        None
        if release_position is None
        else await _in_force_release(session.id, release_position, evidence=evidence)
    )
    return SupervisionEvent(
        supervision_id=sha256(
            canonical_durable_json_bytes(
                [kind.value, session.id, session.instance_id, *identity], "supervision identity"
            )
        ).hexdigest(),
        kind=kind,
        outcome=outcome,
        sequence=record.sequence,
        session_id=session.id,
        parent_session_id=session.parent_session_id,
        forked=forked,
        agent_name=session.agent_name,
        environment_name=session.environment_name,
        tool_name=tool_name,
        execution_profile_fingerprint=await _in_force_fingerprint(
            record,
            fingerprint_payload or anchor.payload,
            session=session,
            evidence=evidence,
        ),
        release_id=None if release is None else release.id,
        release_snapshot_root=None if release is None else release.snapshot_root,
        model_step_id=model_step_id,
        model_attempt_id=model_attempt_id,
        tool_round_id=tool_round_id,
        tool_call_ids=tool_call_ids,
        approval_id=approval_id,
        input_id=input_id,
        queue_id=queue_id,
        interruption_request_id=interruption_request_id,
        message_origin=message_origin,
        actor=_actor(actor),
        reason=reason if type(reason) is str else None,
        attention_id=attention_id,
        attention_state=attention_state if attention_id is not None else None,
        approval_revision=approval_revision,
        arguments_revision=arguments_revision,
        authorization_grant_id=authorization_grant_id,
        authorization_grant_role=authorization_grant_role,
        requested_at=requested_at,
        resolved_at=anchor.timestamp,
        usage=await _usage(session.id, model_step_id, model_attempt_id, evidence=evidence),
        content=content,
    )


async def _in_force_fingerprint(
    record: EventRecord,
    payload: dict[str, Any],
    *,
    session: Session,
    evidence: _SessionEvidence,
) -> str | None:
    """Return the profile bound to the event, from positive durable evidence only."""

    stamped = payload.get("execution_profile_fingerprint")
    if type(stamped) is str and _LOWER_HEX_SHA256.fullmatch(stamped):
        return stamped
    decision = await evidence.latest_before(
        session.id,
        (EventType.SESSION_EXECUTION_PROFILE_DECIDED,),
        record.sequence,
        lambda event: event.payload.get("decision") in _ADMITTED_PROFILE_DECISIONS,
    )
    if decision is not None:
        candidate = _object(decision.event.payload, "candidate_profile")
        fingerprint = _text(candidate or {}, "fingerprint")
        return fingerprint if fingerprint and _LOWER_HEX_SHA256.fullmatch(fingerprint) else None
    try:
        return execution_profile_baseline_from_session_metadata(session.metadata).fingerprint
    except (TypeError, ValueError, ValidationError):
        return None


async def _in_force_release(
    session_id: str, position: int, *, evidence: _SessionEvidence
) -> ReleaseIdentity | None:
    """Return the Release recorded by the last event before ``position`` that begins work."""

    marker = await evidence.nearest_before(session_id, _RELEASE_MARKER_TYPES, position)
    return None if marker is None else recorded_release_identity(marker.event.payload)


async def _usage(
    session_id: str,
    model_step_id: str | None,
    model_attempt_id: str | None,
    *,
    evidence: _SessionEvidence,
) -> UsageMetrics | None:
    if model_step_id is None or model_attempt_id is None:
        return None
    for completion in await evidence.model_completions(session_id, model_step_id):
        if _text(completion.event.payload, "model_attempt_id") != model_attempt_id:
            continue
        metrics = completion.event.payload.get("usage_metrics")
        if type(metrics) is not dict:
            return None
        try:
            return UsageMetrics.model_validate(metrics)
        except ValidationError:
            return None
    return None


async def _published_arguments(
    evidence: _SessionEvidence,
    session_id: str,
    *,
    approval_id: str,
    after: int,
    field: str = "arguments",
) -> dict[str, Any] | None:
    terminals = await evidence.all_after(
        session_id,
        (*_TOOL_TERMINAL_TYPES, EventType.TOOL_CALL_APPROVAL_DENIED),
        after,
        lambda event: (
            _text(event.payload, "approval_id") == approval_id
            and type(event.payload.get(field)) is dict
        ),
    )
    published = {
        call_id: terminal.event.payload[field]
        for terminal in terminals
        if (call_id := _text(terminal.event.payload, "tool_call_id")) is not None
    }
    return published or None


_APPROVAL_ATTENTION_STATES: dict[SupervisionEventKind, SupervisionAttentionState] = {
    SupervisionEventKind.APPROVE: "resolved",
    SupervisionEventKind.EDIT: "resolved",
    SupervisionEventKind.DENY: "cancelled",
    SupervisionEventKind.EXPIRE: "expired",
}


def _attention_id(
    kind: SupervisionEventKind,
    session: Session,
    *,
    attention_kind: PendingActionKind | None,
    approval_id: str | None,
    input_id: str | None,
    tool_round_id: str | None,
    tool_call_ids: tuple[str, ...],
) -> str | None:
    """Name the human-attention request a touch settled, as pending-action reads do.

    A recovery settles the manual-recovery request when the runtime handed the
    call off to one (``attention_kind``), even if an approval or question started
    the round. Otherwise an approval or its recovery is a tool approval, a
    question or its recovery is user input, and a round recovery without either
    is manual recovery. Steering opens no attention request.
    """

    if kind is SupervisionEventKind.STEER:
        return None
    tool_call_id = tool_call_ids[0] if tool_call_ids else None
    if attention_kind is PendingActionKind.MANUAL_RECOVERY or (
        kind is SupervisionEventKind.RECOVER and approval_id is None and input_id is None
    ):
        if tool_round_id is None or tool_call_id is None:
            return None
        attention_kind, discriminator = PendingActionKind.MANUAL_RECOVERY, tool_round_id
    elif approval_id is not None:
        attention_kind, discriminator = PendingActionKind.TOOL_APPROVAL, approval_id
    elif input_id is not None:
        attention_kind, discriminator = PendingActionKind.USER_INPUT, input_id
    else:
        return None
    return human_attention_id(
        session_id=session.id,
        session_instance_id=session.instance_id,
        kind=attention_kind,
        discriminator=discriminator,
        tool_call_id=tool_call_id,
    )


def _recovery_attention_state(terminal: Event) -> SupervisionAttentionState | None:
    from cayu.runtime.human_attention import manual_recovery_settlement

    try:
        return manual_recovery_settlement(terminal)
    except (ValidationError, ValueError, TypeError):
        return None


def _approval_revision(payload: dict[str, Any]) -> ApprovalRevision | None:
    """Read the declaration the runtime published as its own approval field.

    Bounded audit metadata may drop it, so metadata never decides an edit.
    """

    value = _object(payload, "approval_revision")
    if value is None:
        return None
    try:
        return ApprovalRevision.model_validate(value)
    except ValidationError:
        return None


def _arguments_revision(payload: dict[str, Any]) -> ApprovalArgumentsRevision | None:
    """Read the runtime's own evidence that a person changed the call's arguments.

    It is a runtime-owned event field, never metadata, so it survives every
    secret scope and metadata bound.
    """

    value = _object(payload, "arguments_revision")
    if value is None:
        return None
    try:
        return ApprovalArgumentsRevision.model_validate(value)
    except ValidationError:
        return None


def _authorization_grant(
    request: EventRecord | None, resolution: dict[str, Any] | None
) -> tuple[str | None, SupervisionGrantRole | None]:
    """Read grant linkage already recorded on the approval.

    A grant-allowed call never pauses, so it is not a touch. A call a grant sent
    to sampled review carries the grant in its request's policy metadata, and an
    approval that created a grant carries it in its resolution metadata.
    """

    created = _object(
        _object(resolution or {}, "metadata") or {}, TOOL_AUTHORIZATION_GRANT_METADATA_KEY
    )
    if created is not None and (grant_id := _text(created, "grant_id")) is not None:
        return grant_id, SupervisionGrantRole.CREATED
    approval = _object(request.event.payload, "approval") if request is not None else None
    sampled = _object(
        _object(approval or {}, "metadata") or {}, TOOL_AUTHORIZATION_GRANT_METADATA_KEY
    )
    if (
        sampled is not None
        and sampled.get("decision") == "sampled_review"
        and (grant_id := _text(sampled, "grant_id")) is not None
    ):
        return grant_id, SupervisionGrantRole.SAMPLED_REVIEW
    return None, None


def _actor(value: object) -> ResolutionActor | None:
    if type(value) is not dict:
        return None
    try:
        return ResolutionActor.model_validate({**value, "claims": {}})
    except ValidationError:
        return None


def _distinct(values: Iterable[str | None]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(value for value in values if value is not None))


def _text(payload: dict[str, Any], key: str) -> str | None:
    value = payload.get(key)
    return value if type(value) is str and value else None


def _object(payload: dict[str, Any], key: str) -> dict[str, Any] | None:
    value = payload.get(key)
    return value if type(value) is dict else None

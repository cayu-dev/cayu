"""Complete in-memory event and evidence reads over an explicit locked view."""

from __future__ import annotations

import asyncio
from bisect import bisect_left, bisect_right
from collections.abc import Callable
from contextlib import AbstractAsyncContextManager
from itertools import islice
from typing import TYPE_CHECKING, Any, NamedTuple

from cayu._validation import (
    JsonUtf8SizeCounter,
)
from cayu._validation import require_durable_clean_nonblank as require_clean_nonblank
from cayu.budgets.pricing import PriceBook
from cayu.events import Event, EventType, copy_event
from cayu.execution_units import ToolRoundIdentity, copy_tool_round_identity
from cayu.messages import Message
from cayu.runtime._cost_accounting import CostAccountingSnapshot
from cayu.runtime._usage_accounting import UsageAccountingSnapshot
from cayu.sessions import _runtime_publication as publication_contracts
from cayu.sessions import _tool_round_publication_rules as tool_publication_rules
from cayu.sessions import event_queries as event_query_rules
from cayu.sessions import interactions as interaction_contracts
from cayu.sessions import terminal_evidence as terminal_session_evidence
from cayu.sessions._access_rules import runtime_session_query
from cayu.sessions._checkpoint_publication import INITIAL_TRANSCRIPT_PENDING_CHECKPOINT_KEY
from cayu.sessions._event_snapshot import EventSnapshotConsumer, EventSnapshotFold, fold_snapshot
from cayu.sessions._terminal_evidence import (
    TERMINAL_EVIDENCE_EVENT_TYPES,
    TERMINAL_EVIDENCE_QUERY_LIMIT,
)
from cayu.sessions.event_queries import (
    EventOrder,
    EventQuery,
    EventQueryResultTooLarge,
    copy_event_query,
)
from cayu.sessions.records import (
    EventRecord,
    RunnerObservedEventIdentity,
    Session,
    SessionStatus,
    TranscriptRecord,
)
from cayu.sessions.summaries import (
    EventSummary,
    SessionOutcome,
    event_summary_from_records,
    session_outcome_from_records,
)
from cayu.sessions.terminal_evidence import (
    TerminalSessionEvidence,
    TerminalSessionEvidenceError,
    TerminalSessionEvidenceErrorCode,
    TerminalSessionEvidenceLimits,
)
from cayu.workflows.base import WORKFLOW_ATTEMPT_EVENT_TYPE

if TYPE_CHECKING:
    from cayu.runtime._cost_accounting_index import CostEventIndex
    from cayu.runtime._cost_accounting_refresh import CostAccountingAuthority
    from cayu.runtime._usage_accounting import SessionUsageCache


class MemoryEventReadState(NamedTuple):
    """Borrowed indexes valid only while the supplied read context holds its lock.

    Construct the view after acquiring the lock and fetch current index references
    on every read. Operations never mutate these indexes; accounting may update
    its supplied cache while holding the same lock.
    """

    accounting_generation: int
    checkpoints: dict[str, dict[str, Any]]
    cost_accounting_authority: CostAccountingAuthority
    cost_event_index: CostEventIndex
    event_records: list[EventRecord]
    event_records_by_id: dict[tuple[str, str], EventRecord]
    events: dict[str, list[Event]]
    interaction_event_records: dict[str, dict[str, list[EventRecord]]]
    latest_interaction_event_records_by_sequence: dict[str, list[EventRecord]]
    pending_action_event_history: dict[str, dict[str, list[EventRecord]]]
    session_event_records: dict[str, list[EventRecord]]
    session_usage_cache: SessionUsageCache
    sessions: dict[str, Session]
    transcript_interaction_ids: dict[str, list[str | None]]
    transcripts: dict[str, list[Message]]
    type_event_records: dict[str, list[EventRecord]]
    user_input_supersession_event_records: dict[tuple[str, str], list[EventRecord]]
    workflow_attempt_event_records: dict[tuple[str, str], list[EventRecord]]
    workflow_fenced_step_event_records: dict[tuple[str, str, str, str], list[EventRecord]]
    workflow_step_event_records: dict[tuple[str, str, str, str, str], list[EventRecord]]


LockedEventRead = Callable[[], AbstractAsyncContextManager[MemoryEventReadState]]


async def load_events(read: LockedEventRead, session_id: str) -> list[Event]:
    from cayu.resource_access import current_data_bounds

    access_bounds = await current_data_bounds()
    session_id = require_clean_nonblank(session_id, "session_id")
    async with read() as state:
        if access_bounds is not None:
            access_bounds.require_read(state.sessions.get(session_id))
        if session_id not in state.sessions:
            raise KeyError(f"Session not found: {session_id}")
        return [event.model_copy(deep=True) for event in state.events.get(session_id, [])]


async def load_user_input_supersession_events(
    read: LockedEventRead,
    session_id: str,
    input_id: str,
) -> list[Event]:
    from cayu.sessions.pending_actions import pending_action_lookup_key

    session_id = require_clean_nonblank(session_id, "session_id")
    input_id = require_clean_nonblank(input_id, "input_id")
    lookup_key = pending_action_lookup_key(input_id)
    async with read() as state:
        if session_id not in state.sessions:
            raise KeyError(f"Session not found: {session_id}")
        records = state.user_input_supersession_event_records.get(
            (session_id, lookup_key),
            (),
        )
        return [copy_event(record.event) for record in records[:2]]


async def load_tool_round_lifecycle_events(
    read: LockedEventRead,
    session_id: str,
    tool_call_ids: list[str] | tuple[str, ...],
) -> list[Event]:
    from cayu.sessions.pending_actions import pending_action_lookup_key

    session_id = require_clean_nonblank(session_id, "session_id")
    copied_ids = tool_publication_rules._validate_tool_round_call_ids(
        tool_call_ids, "tool_call_ids"
    )
    async with read() as state:
        if session_id not in state.sessions:
            raise KeyError(f"Session not found: {session_id}")
        history = state.pending_action_event_history.get(session_id, {})
        records = [
            record
            for tool_call_id in copied_ids
            for record in history.get(pending_action_lookup_key(tool_call_id), ())
            if record.event.type in tool_publication_rules._TOOL_ROUND_LIFECYCLE_EVENT_TYPES
        ]
        records.sort(key=lambda record: record.sequence)
        if len(records) > publication_contracts._tool_round_lifecycle_event_limit(copied_ids):
            raise ValueError("Tool-round lifecycle evidence exceeds the publication limit.")
        return [copy_event(record.event) for record in records]


async def load_tool_round_lifecycle_events_for_round(
    read: LockedEventRead,
    session_id: str,
    tool_call_ids: list[str] | tuple[str, ...],
    *,
    tool_round_identity: ToolRoundIdentity,
) -> list[Event]:
    from cayu.sessions.pending_actions import pending_action_lookup_key

    session_id = require_clean_nonblank(session_id, "session_id")
    copied_ids = tool_publication_rules._validate_tool_round_call_ids(
        tool_call_ids, "tool_call_ids"
    )
    tool_round_identity = copy_tool_round_identity(tool_round_identity)
    async with read() as state:
        if session_id not in state.sessions:
            raise KeyError(f"Session not found: {session_id}")
        history = state.pending_action_event_history.get(session_id, {})
        records = [
            record
            for tool_call_id in copied_ids
            for record in history.get(pending_action_lookup_key(tool_call_id), ())
            if record.event.type in tool_publication_rules._TOOL_ROUND_LIFECYCLE_EVENT_TYPES
            and tool_publication_rules._tool_round_event_matches_scope_or_is_ambiguous(
                record.event,
                tool_round_identity,
            )
        ]
        records.sort(key=lambda record: record.sequence)
        if len(records) > publication_contracts._tool_round_lifecycle_event_limit(copied_ids):
            raise ValueError("Tool-round lifecycle evidence exceeds the publication limit.")
        return [copy_event(record.event) for record in records]


@runtime_session_query
async def query_events(read: LockedEventRead, query: EventQuery | None = None) -> list[EventRecord]:
    query = copy_event_query(query)
    async with read() as state:
        return _query_events_unlocked(state, query, max_bytes=None)


@runtime_session_query
async def event_exists(read: LockedEventRead, query: EventQuery) -> bool:
    query = copy_event_query(query)
    event_types = frozenset(
        (str(query.event_type),) if query.event_type is not None else map(str, query.event_types)
    )
    excluded_types = frozenset(map(str, query.exclude_event_types))
    async with read() as state:
        records = (
            state.session_event_records.get(query.session_id, [])
            if query.session_id is not None
            else state.event_records
        )
        start = bisect_right(records, query.after_sequence or 0, key=lambda row: row.sequence)
        session_matches = _session_matcher(query, state)
        for index in range(start, len(records)):
            record = records[index]
            if query.before_sequence is not None and record.sequence >= query.before_sequence:
                break
            if event_query_rules._event_record_matches(
                record, query, event_types, excluded_types
            ) and session_matches(record):
                return True
        return False


@runtime_session_query
async def read_usage_accounting(
    read: LockedEventRead, query: EventQuery, *, by_session: bool = False, by_identity: bool = False
) -> UsageAccountingSnapshot:
    from cayu.runtime._usage_accounting import (
        USAGE_ACCOUNTING_PAGE_SIZE,
        SessionUsageCache,
        UsageAccountingReducer,
        usage_accounting_query,
    )

    query = usage_accounting_query(query)
    cached_session_id = SessionUsageCache.session_scope(
        query, by_session=by_session, by_identity=by_identity
    )
    async with read() as state:
        generation = state.accounting_generation
        prior = (
            None
            if cached_session_id is None
            else state.session_usage_cache.resume_after(cached_session_id, generation)
        )
        if prior is not None:
            query = copy_event_query(query, update={"after_sequence": prior.scanned_through})
        reducer = UsageAccountingReducer(query, by_session=by_session, by_identity=by_identity)
        event_types = frozenset(str(kind) for kind in query.event_types)
        # Reuse durable indexes without merging complete per-type/session lists.
        candidates = (
            state.session_event_records.get(query.session_id, [])
            if query.session_id is not None
            else state.event_records
        )
        start = bisect_right(candidates, query.after_sequence or 0, key=lambda row: row.sequence)
        page: list[EventRecord] = []
        session_matches = _session_matcher(query, state)
        for index in range(start, len(candidates)):
            record = candidates[index]
            if query.before_sequence is not None and record.sequence >= query.before_sequence:
                break
            if not event_query_rules._event_record_matches(record, query, event_types, frozenset()):
                continue
            if not session_matches(record):
                continue
            page.append(record)
            if len(page) == USAGE_ACCOUNTING_PAGE_SIZE:
                reducer.add_page(page)
                page.clear()
                await asyncio.sleep(0)
        if page:
            reducer.add_page(page)
        if cached_session_id is not None:
            return state.session_usage_cache.settle(
                cached_session_id,
                generation,
                prior,
                reducer.snapshot(),
                boundary=candidates[-1].sequence if candidates else 0,
            )
    return reducer.snapshot().model_copy(update={"generation": generation})


@runtime_session_query
async def read_cost_accounting(
    read: LockedEventRead,
    query: EventQuery,
    pricing: PriceBook,
    *,
    currency: str = "USD",
    details: bool = False,
    by_session: bool = False,
    additional_events: tuple[Event, ...] = (),
    max_detail_bytes: int | None = None,
    previous: CostAccountingSnapshot | None = None,
    fork_lineage_owner: str | None = None,
    without_reservation_for: str | None = None,
) -> CostAccountingSnapshot:
    from cayu.runtime._cost_accounting import (
        COST_ACCOUNTING_PAGE_SIZE,
        cost_accounting_query,
        cost_pending_events,
        require_cost_read_filters,
        require_no_pending_for_reservation_filter,
    )
    from cayu.runtime._cost_accounting_refresh import CostAccountingRead
    from cayu.sessions._fork_controller_bounds import fork_lineage_owned_by
    from cayu.sessions.fork_overlays import resolve_fork_lineage

    query = cost_accounting_query(query)
    pending = cost_pending_events(query, additional_events)
    event_types = frozenset(str(kind) for kind in query.event_types)
    fork_lineage_owner, without_reservation_for = require_cost_read_filters(
        fork_lineage_owner, without_reservation_for
    )
    require_no_pending_for_reservation_filter(without_reservation_for, additional_events)
    filtered = fork_lineage_owner is not None or without_reservation_for is not None
    if filtered:
        # A filtered read never refreshes another read's snapshot.
        previous = None
    async with read() as state:
        owned: dict[str, bool] = {}

        def in_lineage(session_id: str) -> bool:
            if fork_lineage_owner is None:
                return True
            decided = owned.get(session_id)
            if decided is None:
                session = state.sessions.get(session_id)
                decided = session is not None and fork_lineage_owned_by(
                    resolve_fork_lineage(session, state.sessions.get), fork_lineage_owner
                )
                owned[session_id] = decided
            return decided

        reserved: dict[str, frozenset[str]] = {}

        def unreserved(event: Event) -> bool:
            if without_reservation_for is None:
                return True
            attempts = reserved.get(event.session_id)
            if attempts is None:
                attempts = reserved_attempts(
                    state.session_event_records.get(event.session_id, ()),
                    without_reservation_for,
                )
                reserved[event.session_id] = attempts
            return event.payload.get("model_attempt_id") not in attempts

        pending = tuple(event for event in pending if in_lineage(event.session_id))
        if query.causal_budget_id is not None:
            pending = tuple(
                event
                for event in pending
                if (session := state.sessions.get(event.session_id)) is not None
                and session.causal_budget_id == query.causal_budget_id
            )
        pending = tuple(
            event
            for event in pending
            if (record := state.event_records_by_id.get((event.session_id, event.id))) is None
            or not event_query_rules._event_record_matches(record, query, event_types, frozenset())
        )
        boundary_session_id = query.session_id or (
            query.session_ids[0] if len(query.session_ids) == 1 else None
        )
        boundary_records = (
            state.session_event_records.get(boundary_session_id, [])
            if boundary_session_id is not None
            else state.event_records
        )
        boundary_end = (
            bisect_left(boundary_records, query.before_sequence, key=lambda row: row.sequence)
            if query.before_sequence is not None
            else len(boundary_records)
        )
        through_sequence = boundary_records[boundary_end - 1].sequence if boundary_end else 0
        reducer = CostAccountingRead(
            query,
            pricing,
            currency=currency,
            details=details,
            max_detail_bytes=max_detail_bytes,
            previous=previous,
            generation=state.accounting_generation,
            through_sequence=through_sequence,
            authority=state.cost_accounting_authority,
            by_session=by_session,
            additional_events=pending,
        )
        scanned = 0
        session_matches = _session_matcher(query, state)
        for key, records in state.cost_event_index.groups(reducer):
            if query.session_id is not None and key[0] != query.session_id:
                continue
            if query.session_ids and key[0] not in query.session_ids:
                continue
            if (
                query.causal_budget_id is not None
                and state.sessions[key[0]].causal_budget_id != query.causal_budget_id
            ):
                continue
            if not in_lineage(key[0]):
                continue
            for hosted_phase in (True, False):
                for record in records:
                    scanned += 1
                    if scanned % COST_ACCOUNTING_PAGE_SIZE == 0:
                        await asyncio.sleep(0)
                    if (record.event.type == EventType.MODEL_HOSTED_TOOL_CALL) != hosted_phase:
                        continue
                    if (
                        event_query_rules._event_record_matches(
                            record, reducer.source_query, event_types, frozenset()
                        )
                        and session_matches(record)
                        and unreserved(record.event)
                    ):
                        reducer.add(record.sequence, record.event)
        snapshot = reducer.snapshot()
        return snapshot if not filtered else snapshot.model_copy(update={"cursor": None})


def reserved_attempts(records, budget_limit_id: str) -> frozenset[str]:
    """Model attempts in one session's records with an accepted reservation for the limit."""

    return frozenset(
        attempt
        for record in records
        if record.event.type == EventType.BUDGET_RESERVED
        and record.event.payload.get("budget_limit_id") == budget_limit_id
        and record.event.payload.get("accepted") is True
        and type(attempt := record.event.payload.get("model_attempt_id")) is str
    )


@runtime_session_query
async def fold_event_snapshot(
    read: LockedEventRead,
    query: EventQuery,
    consume: EventSnapshotConsumer,
    *,
    max_bytes: int,
    max_pages: int | None = None,
) -> int:
    fold = EventSnapshotFold(query, max_bytes=max_bytes, max_pages=max_pages)
    async with read() as state:
        return fold_snapshot(
            fold, lambda page: _query_events_unlocked(state, page, max_bytes=max_bytes), consume
        )


@runtime_session_query
async def query_events_bounded(
    read: LockedEventRead,
    query: EventQuery,
    *,
    max_bytes: int,
) -> list[EventRecord]:
    query = copy_event_query(query)
    if type(max_bytes) is not int or max_bytes < 1:
        raise ValueError("max_bytes must be a positive integer.")
    async with read() as state:
        return _query_events_unlocked(state, query, max_bytes=max_bytes)


async def load_terminal_session_evidence(
    read: LockedEventRead,
    session_id: str,
    *,
    limits: TerminalSessionEvidenceLimits | None = None,
) -> TerminalSessionEvidence:
    return await _load_terminal_session_evidence(
        read,
        session_id,
        limits=limits,
        observed_interrupted_events=None,
        expected_interrupted_parent_session_id=None,
        require_interrupted_proof=False,
    )


async def load_runner_owned_interrupted_evidence(
    read: LockedEventRead,
    session_id: str,
    *,
    observed_events: tuple[RunnerObservedEventIdentity, ...] | None = None,
    expected_parent_session_id: str | None = None,
    limits: TerminalSessionEvidenceLimits | None = None,
) -> TerminalSessionEvidence:
    return await _load_terminal_session_evidence(
        read,
        session_id,
        limits=limits,
        observed_interrupted_events=observed_events,
        expected_interrupted_parent_session_id=expected_parent_session_id,
        require_interrupted_proof=True,
    )


async def _load_terminal_session_evidence(
    read: LockedEventRead,
    session_id: str,
    *,
    limits: TerminalSessionEvidenceLimits | None,
    observed_interrupted_events: tuple[RunnerObservedEventIdentity, ...] | None,
    expected_interrupted_parent_session_id: str | None,
    require_interrupted_proof: bool,
) -> TerminalSessionEvidence:
    session_id = require_clean_nonblank(session_id, "session_id")
    limits = terminal_session_evidence._copy_terminal_session_evidence_limits(limits)
    observed, expected_parent_session_id = (
        terminal_session_evidence._copy_runner_owned_interruption_proof(
            session_id,
            observed_events=observed_interrupted_events,
            expected_parent_session_id=expected_interrupted_parent_session_id,
            limits=limits,
            required=require_interrupted_proof,
        )
    )
    allow_interrupted = observed is not None or expected_parent_session_id is not None
    async with read() as state:
        session = state.sessions.get(session_id)
        if session is None:
            raise TerminalSessionEvidenceError(TerminalSessionEvidenceErrorCode.SESSION_NOT_FOUND)
        terminal_session_evidence._terminal_session_evidence_expected_event_type(
            session.status,
            allow_interrupted=allow_interrupted,
        )
        if allow_interrupted and session.status != SessionStatus.INTERRUPTED:
            raise TerminalSessionEvidenceError(
                TerminalSessionEvidenceErrorCode.EVIDENCE_INCONSISTENT
            )
        if (
            expected_parent_session_id is not None
            and session.parent_session_id != expected_parent_session_id
        ):
            raise TerminalSessionEvidenceError(
                TerminalSessionEvidenceErrorCode.EVIDENCE_INCONSISTENT
            )
        checkpoint = state.checkpoints.get(session_id)
        marker = terminal_session_evidence._terminal_session_evidence_marker_from_checkpoint(
            checkpoint
        )
        session_records = state.session_event_records.get(session_id, [])
        newest_evidence_records = tuple(
            islice(
                (
                    record
                    for record in reversed(session_records)
                    if record.event.type in TERMINAL_EVIDENCE_EVENT_TYPES
                ),
                TERMINAL_EVIDENCE_QUERY_LIMIT,
            )
        )
        terminal_record = terminal_session_evidence._classify_terminal_session_evidence_records(
            session_id=session.id,
            status=session.status,
            run_epoch=session.run_epoch,
            marker=marker,
            newest_evidence_records=newest_evidence_records,
            initial_transcript_pending=(
                checkpoint is not None and INITIAL_TRANSCRIPT_PENDING_CHECKPOINT_KEY in checkpoint
            ),
            pending_session_interrupt=(
                checkpoint is not None and "pending_session_interrupt" in checkpoint
            ),
            allow_interrupted=allow_interrupted,
        )
        event_stop = bisect_right(
            session_records,
            terminal_record.sequence,
            key=lambda record: record.sequence,
        )
        if event_stop > limits.max_events:
            raise TerminalSessionEvidenceError(
                TerminalSessionEvidenceErrorCode.EVENT_LIMIT_EXCEEDED,
                limit=limits.max_events,
                observed=event_stop,
            )
        events = session_records[:event_stop]
        if observed is not None:
            terminal_session_evidence._validate_runner_observed_event_identity_snapshot(
                observed,
                tuple(
                    RunnerObservedEventIdentity(
                        session_id=record.event.session_id,
                        sequence=record.sequence,
                        event_type=record.event.type,
                    )
                    for record in events
                ),
                terminal_sequence=terminal_record.sequence,
            )

        messages = state.transcripts.get(session_id, [])
        interaction_ids = state.transcript_interaction_ids.get(session_id, [])
        if len(messages) != len(interaction_ids):
            raise TerminalSessionEvidenceError(
                TerminalSessionEvidenceErrorCode.EVIDENCE_INCONSISTENT
            )
        if len(messages) > limits.max_transcript_records:
            raise TerminalSessionEvidenceError(
                TerminalSessionEvidenceErrorCode.TRANSCRIPT_LIMIT_EXCEEDED,
                limit=limits.max_transcript_records,
                observed=len(messages),
            )
        # The store already owns detached, validated messages. Construct
        # lightweight record views so oversized evidence is byte-checked
        # before any message payload is copied for the returned model.
        transcript_records = tuple(
            TranscriptRecord.model_construct(
                index=index,
                interaction_id=interaction_ids[index],
                message=message,
            )
            for index, message in enumerate(messages)
        )
        return terminal_session_evidence._assemble_terminal_session_evidence(
            session=session,
            marker=marker,
            terminal_record=terminal_record,
            events=events,
            transcript=transcript_records,
            limits=limits,
            allow_interrupted=allow_interrupted,
        )


def _query_events_unlocked(
    state: MemoryEventReadState,
    query: EventQuery,
    *,
    max_bytes: int | None,
) -> list[EventRecord]:
    # `event_type` and `event_types` are mutually exclusive, so they
    # collapse into one filter set (empty = no type filter).
    if query.event_type is not None:
        event_types = frozenset((str(query.event_type),))
    else:
        event_types = frozenset(str(event_type) for event_type in query.event_types)
    excluded_event_types = frozenset(str(event_type) for event_type in query.exclude_event_types)
    candidates = _query_candidate_records(state, query, event_types)
    start = (
        bisect_right(candidates, query.after_sequence, key=lambda record: record.sequence)
        if query.after_sequence is not None
        else 0
    )
    stop = (
        bisect_left(candidates, query.before_sequence, key=lambda record: record.sequence)
        if query.before_sequence is not None
        else len(candidates)
    )
    indexes = (
        range(stop - 1, start - 1, -1)
        if query.order_by == EventOrder.SEQUENCE_DESC
        else range(start, stop)
    )
    size_counter = None if max_bytes is None else JsonUtf8SizeCounter(max_bytes)
    records: list[EventRecord] = []
    session_matches = _session_matcher(query, state)
    for index in indexes:
        record = candidates[index]
        if not event_query_rules._event_record_matches(
            record,
            query,
            event_types,
            excluded_event_types,
        ):
            continue
        if not session_matches(record):
            continue
        if size_counter is not None and not size_counter.value(record):
            assert max_bytes is not None
            raise EventQueryResultTooLarge(max_bytes)
        records.append(
            EventRecord(
                sequence=record.sequence,
                event=record.event,
            )
        )
        if len(records) == query.limit:
            break
    return records


async def query_latest_interaction_events(
    read: LockedEventRead,
    session_id: str,
    *,
    before_sequence: int | None = None,
    limit: int = 100,
) -> list[EventRecord]:
    session_id = require_clean_nonblank(session_id, "session_id")
    before_sequence, limit = interaction_contracts._validate_interaction_page(
        before_sequence, limit
    )
    async with read() as state:
        if session_id not in state.sessions:
            raise KeyError(f"Session not found: {session_id}")
        records = state.latest_interaction_event_records_by_sequence.get(session_id, [])
        stop = (
            bisect_left(
                records,
                before_sequence,
                key=lambda record: record.sequence,
            )
            if before_sequence is not None
            else len(records)
        )
        start = max(0, stop - limit)
        return [record.model_copy(deep=True) for record in reversed(records[start:stop])]


def _query_candidate_records(
    state: MemoryEventReadState,
    query: EventQuery,
    event_types: frozenset[str],
) -> list[EventRecord]:
    """Pick the narrowest index that still covers a query's candidate rows.

    All returned lists stay sequence-ascending so downstream ordering and
    ``after_sequence`` paging behave exactly as a full scan would.
    """
    if query.event_id is not None and query.session_id is not None:
        record = state.event_records_by_id.get((query.session_id, query.event_id))
        return [] if record is None else [record]
    if (
        query.session_id is not None
        and query.workflow_name is not None
        and event_types == frozenset((str(WORKFLOW_ATTEMPT_EVENT_TYPE),))
    ):
        return state.workflow_attempt_event_records.get(
            (query.session_id, query.workflow_name),
            [],
        )
    if query.workflow_step_id is not None:
        assert query.session_id is not None
        assert query.workflow_name is not None
        if len(event_types) == 1:
            event_type = next(iter(event_types))
            if query.workflow_attempt_fenced:
                return state.workflow_fenced_step_event_records.get(
                    (
                        query.session_id,
                        query.workflow_name,
                        query.workflow_step_id,
                        event_type,
                    ),
                    [],
                )
            if query.workflow_attempt_id is not None:
                return state.workflow_step_event_records.get(
                    (
                        query.session_id,
                        query.workflow_name,
                        query.workflow_attempt_id,
                        query.workflow_step_id,
                        event_type,
                    ),
                    [],
                )
            raise RuntimeError("Workflow-step queries require an attempt scope.")
        merged: list[EventRecord] = []
        for event_type in event_types:
            if query.workflow_attempt_fenced:
                merged.extend(
                    state.workflow_fenced_step_event_records.get(
                        (
                            query.session_id,
                            query.workflow_name,
                            query.workflow_step_id,
                            event_type,
                        ),
                        [],
                    )
                )
            elif query.workflow_attempt_id is not None:
                merged.extend(
                    state.workflow_step_event_records.get(
                        (
                            query.session_id,
                            query.workflow_name,
                            query.workflow_attempt_id,
                            query.workflow_step_id,
                            event_type,
                        ),
                        [],
                    )
                )
            else:
                raise RuntimeError("Workflow-step queries require an attempt scope.")
        merged.sort(key=lambda record: record.sequence)
        return merged
    if query.session_id is not None:
        if query.interaction_id is not None:
            return state.interaction_event_records.get(query.session_id, {}).get(
                query.interaction_id,
                [],
            )
        return state.session_event_records.get(query.session_id, [])
    if query.session_ids:
        merged: list[EventRecord] = []
        for session_id in query.session_ids:
            if query.interaction_id is None:
                merged.extend(state.session_event_records.get(session_id, []))
            else:
                merged.extend(
                    state.interaction_event_records.get(session_id, {}).get(
                        query.interaction_id,
                        [],
                    )
                )
        merged.sort(key=lambda record: record.sequence)
        return merged
    if event_types:
        if len(event_types) == 1:
            return state.type_event_records.get(next(iter(event_types)), [])
        merged = []
        for event_type in event_types:
            merged.extend(state.type_event_records.get(event_type, []))
        merged.sort(key=lambda record: record.sequence)
        return merged
    return state.event_records


async def summarize_events(read: LockedEventRead, session_id: str) -> EventSummary:
    from cayu.resource_access import current_data_bounds

    access_bounds = await current_data_bounds()
    session_id = require_clean_nonblank(session_id, "session_id")
    async with read() as state:
        if access_bounds is not None:
            access_bounds.require_read(state.sessions.get(session_id))
        if session_id not in state.sessions:
            raise KeyError(f"Session not found: {session_id}")

        records = state.session_event_records.get(session_id, [])
        return event_summary_from_records(session_id, records)


async def summarize_outcome(read: LockedEventRead, session_id: str) -> SessionOutcome:
    from cayu.resource_access import current_data_bounds

    access_bounds = await current_data_bounds()
    session_id = require_clean_nonblank(session_id, "session_id")
    async with read() as state:
        if access_bounds is not None:
            access_bounds.require_read(state.sessions.get(session_id))
        session = state.sessions.get(session_id)
        if session is None:
            raise KeyError(f"Session not found: {session_id}")

        records = state.session_event_records.get(session_id, [])
        return session_outcome_from_records(session, records)


def _session_matcher(
    query: EventQuery,
    state: MemoryEventReadState,
) -> Callable[[EventRecord], bool]:
    """Decide each owning session once per read, under the read's lock.

    The read bound, the forked-lineage exclusion and the causal budget filter
    depend only on the record's session, so a read costs one decision per
    session rather than per record, and fork markers are indexed at most once.
    """

    from cayu.sessions._access_rules import _query_bounds, session_read_admitted
    from cayu.sessions._fork_controller_bounds import _ForkControllerReadBounds

    sessions = state.sessions
    bounds = _query_bounds.get()
    excludes = (
        bounds.excludes
        if type(bounds) is _ForkControllerReadBounds and bounds.exclude_forked
        else None
    )
    decided: dict[str, bool] = {}
    forked_session_ids: set[str] | None = None

    def decide(session_id: str) -> bool:
        nonlocal forked_session_ids
        owner = sessions.get(session_id)
        if bounds is not None:
            if owner is None or not session_read_admitted(bounds, owner, sessions.get):
                return False
            if excludes is not None:
                if excludes(owner, sessions.get):
                    return False
                if forked_session_ids is None:
                    forked_session_ids = {
                        marker.event.session_id
                        for marker in state.type_event_records.get(
                            str(EventType.SESSION_FORKED), ()
                        )
                    }
                if owner.id in forked_session_ids:
                    return False
        if query.causal_budget_id is None:
            return True
        return owner is not None and owner.causal_budget_id == query.causal_budget_id

    def matches(record: EventRecord) -> bool:
        session_id = record.event.session_id
        decision = decided.get(session_id)
        if decision is None:
            decision = decided[session_id] = decide(session_id)
        return decision

    return matches

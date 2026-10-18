"""Complete in-memory session discovery operations over an explicit locked view."""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Callable, Iterable
from contextlib import AbstractAsyncContextManager
from datetime import datetime
from itertools import islice
from typing import TYPE_CHECKING, Any, NamedTuple

from cayu._clock import utc_duration_cutoff
from cayu._validation import (
    JsonUtf8SizeCounter,
)
from cayu.budgets.aggregates import (
    _IN_MEMORY_AGGREGATE_CANCELLATION_INTERVAL,
    EXACT_AGGREGATE,
    UsageRollupStoreResult,
    _cooperate_with_in_memory_aggregate_cancellation,
)
from cayu.events import EventType
from cayu.runtime._child_session_notifications import (
    ChildSessionLifecycleEntry,
    ChildSessionLifecyclePage,
    ChildSessionLifecycleQuery,
    child_session_notification_storage_key,
)
from cayu.sessions import _child_lifecycle_rules as child_lifecycle_rules
from cayu.sessions import queries as session_query_rules
from cayu.sessions import usage as session_usage_rules
from cayu.sessions._terminal_evidence import (
    QueuedDispatchTerminalReceipt,
    QueuedDispatchTerminalReceiptQuery,
)
from cayu.sessions.authority import (
    CheckpointRootFieldGuard,
    checkpoint_root_field_projection,
)
from cayu.sessions.lineage import (
    SESSION_LINEAGE_MAX_ORIGIN_EVENTS,
    SessionLineageNode,
    SessionLineageOrigin,
    SessionLineageQuery,
    SessionLineageResult,
    copy_session_lineage_query,
    decode_session_lineage_cursor,
    encode_session_lineage_cursor,
)
from cayu.sessions.pending_action_contracts import (
    MAX_PENDING_ACTION_LEDGER_EVENTS_PER_CALL,
    MAX_PENDING_ACTION_TOOL_CALLS,
    PendingActionIssue,
    PendingActionListResult,
    PendingActionQuery,
    PendingActionRecord,
    enforce_pending_action_result_size,
)
from cayu.sessions.queries import (
    SessionAggregateFilter,
    SessionDebugState,
    SessionListResult,
    SessionOrder,
    SessionQuery,
    SessionStatusCounts,
    copy_session_aggregate_filter,
    copy_session_query,
    decode_session_cursor,
    encode_session_cursor,
    session_next_cursor,
    session_query_from_aggregate_filter,
)
from cayu.sessions.records import (
    EventRecord,
    PendingActionSession,
    Session,
    SessionStatus,
)
from cayu.sessions.summaries import (
    SessionOperationalSnapshot,
)
from cayu.sessions.topology import (
    SessionTopologyCycle,
    SessionTopologyDepthExceeded,
    SessionTopologyNode,
    SessionTopologyQuery,
    SessionTopologyStoreResult,
    build_session_topology_result,
    decode_session_topology_cursor,
)
from cayu.sessions.usage import UsageRollupQuery, copy_usage_rollup_query

if TYPE_CHECKING:
    from cayu.sessions._access_rules import SessionReadBounds


class MemorySessionQueryState(NamedTuple):
    """Borrowed indexes valid only while the supplied read context holds its lock.

    Construct the view after acquiring the lock and fetch current index references
    on every read. Operations never mutate these indexes.
    """

    checkpoints: dict[str, dict[str, Any]]
    child_lifecycle_candidate_keys_by_parent: dict[str, dict[int, list[tuple[datetime, str]]]]
    child_lifecycle_event_records: dict[str, dict[EventType, EventRecord]]
    child_session_keys_by_parent: dict[str, list[tuple[datetime, str]]]
    ownership_clock: Callable[[], datetime]
    pending_action_event_records: dict[str, dict[str, dict[str, EventRecord]]]
    pending_action_latest_barrier_records: dict[str, EventRecord]
    pending_action_session_ids: set[str]
    queued_dispatch_terminal_receipt_keys: list[tuple[str, str]]
    queued_dispatch_terminal_receipts: dict[tuple[str, str], QueuedDispatchTerminalReceipt]
    session_event_records: dict[str, list[EventRecord]]
    session_operation_records: dict[str, dict[str, dict[str, Any]]]
    sessions: dict[str, Session]


LockedSessionQuery = Callable[[], AbstractAsyncContextManager[MemorySessionQueryState]]


async def query_session_topology(
    read: LockedSessionQuery,
    query: SessionTopologyQuery,
) -> SessionTopologyStoreResult:
    if type(query) is not SessionTopologyQuery:
        raise TypeError("Session topology queries must be SessionTopologyQuery instances.")
    query = query.model_copy(deep=True)
    async with read() as state:
        focus_session = state.sessions.get(query.focus_session_id)
        if focus_session is None:
            raise KeyError(f"Session not found: {query.focus_session_id}")

        ancestor_sessions: list[Session] = []
        seen_ancestor_ids = {focus_session.id}
        parent_session_id = focus_session.parent_session_id
        while parent_session_id is not None:
            if parent_session_id in seen_ancestor_ids:
                raise SessionTopologyCycle(
                    f"Session topology contains a parent cycle at {parent_session_id}."
                )
            if len(ancestor_sessions) >= query.ancestor_depth_limit:
                raise SessionTopologyDepthExceeded(
                    f"Session topology exceeds the {query.ancestor_depth_limit}-ancestor limit."
                )
            parent = state.sessions.get(parent_session_id)
            if parent is None:
                raise ValueError(f"Session topology references missing parent {parent_session_id}.")
            ancestor_sessions.append(parent)
            seen_ancestor_ids.add(parent.id)
            parent_session_id = parent.parent_session_id
        ancestor_sessions.reverse()

        expanded_sessions: list[Session] = []
        for parent_id in query.expanded_parent_ids:
            parent = state.sessions.get(parent_id)
            if parent is None:
                raise KeyError(f"Session not found: {parent_id}")
            expanded_sessions.append(parent)

        branch_candidates: list[list[Session]] = []
        for parent in expanded_sessions:
            child_keys = state.child_session_keys_by_parent.get(parent.id, [])
            start_index = 0
            cursor = query.child_cursors.get(parent.id)
            if cursor is not None:
                cursor_created_at, cursor_id = decode_session_topology_cursor(
                    cursor,
                    parent_session_id=parent.id,
                )
                start_index = bisect_right(child_keys, (cursor_created_at, cursor_id))
            page_keys = child_keys[start_index : start_index + query.child_limit + 1]
            candidates: list[Session] = []
            for _, child_id in page_keys:
                child = state.sessions.get(child_id)
                if child is None or child.parent_session_id != parent.id:
                    raise RuntimeError("Inconsistent in-memory session topology index.")
                candidates.append(child)
            branch_candidates.append(candidates)

        result = build_session_topology_result(
            focus=SessionTopologyNode.from_session(focus_session),
            ancestors=(SessionTopologyNode.from_session(session) for session in ancestor_sessions),
            expanded_parents=(
                SessionTopologyNode.from_session(session) for session in expanded_sessions
            ),
            branch_candidates=(
                (SessionTopologyNode.from_session(session) for session in candidates)
                for candidates in branch_candidates
            ),
            child_limit=query.child_limit,
        )
        return result


async def query_session_lineage(
    read: LockedSessionQuery,
    query: SessionLineageQuery,
) -> SessionLineageResult:
    query = copy_session_lineage_query(query)
    async with read() as state:
        if query.parent_session_id not in state.sessions:
            raise KeyError(f"Session not found: {query.parent_session_id}")
        child_keys = state.child_session_keys_by_parent.get(query.parent_session_id, [])
        start_index = 0
        if query.cursor is not None:
            cursor_created_at, cursor_id = decode_session_lineage_cursor(
                query.cursor,
                parent_session_id=query.parent_session_id,
            )
            start_index = bisect_right(child_keys, (cursor_created_at, cursor_id))
        page_keys = child_keys[start_index : start_index + query.limit + 1]
        retained_keys = page_keys[: query.limit]
        children: list[SessionLineageNode] = []
        for created_at, child_id in retained_keys:
            child = state.sessions.get(child_id)
            if child is None or child.parent_session_id != query.parent_session_id:
                raise RuntimeError("Inconsistent in-memory session lineage index.")
            origins = tuple(
                islice(
                    (
                        SessionLineageOrigin(
                            sequence=record.sequence,
                            event_id=record.event.id,
                            event_type=EventType(record.event.type),
                        )
                        for record in state.session_event_records.get(child_id, [])
                        if record.event.type
                        in {EventType.SESSION_STARTED, EventType.SESSION_FORKED}
                    ),
                    SESSION_LINEAGE_MAX_ORIGIN_EVENTS,
                )
            )
            children.append(
                SessionLineageNode(
                    id=child.id,
                    parent_session_id=query.parent_session_id,
                    created_at=created_at,
                    origin_events=origins,
                )
            )
        has_more = len(page_keys) > len(retained_keys)
        return SessionLineageResult(
            parent_session_id=query.parent_session_id,
            children=tuple(children),
            next_cursor=(
                encode_session_lineage_cursor(query.parent_session_id, children[-1])
                if has_more and children
                else None
            ),
            has_more=has_more,
        )


async def query_child_session_lifecycle(
    read: LockedSessionQuery,
    query: ChildSessionLifecycleQuery,
) -> ChildSessionLifecyclePage:
    query = ChildSessionLifecycleQuery.model_validate(query)
    async with read() as state:
        parent = state.sessions.get(query.parent_session_id)
        if parent is None:
            raise KeyError(f"Session not found: {query.parent_session_id}")
        child_keys = state.child_session_keys_by_parent.get(parent.id, ())
        candidate_limit = query.max_children_inspected + 1
        candidate_ids: list[str] = []
        priorities = state.child_lifecycle_candidate_keys_by_parent.get(parent.id, {})
        for priority in (0, 1, 2):
            for _sort_at, child_id in priorities.get(priority, ()):
                if len(candidate_ids) >= candidate_limit:
                    break
                candidate_ids.append(child_id)
            if len(candidate_ids) >= candidate_limit:
                break
        if sum(len(keys) for keys in priorities.values()) != len(child_keys):
            raise RuntimeError("In-memory child lifecycle candidate index is inconsistent.")
        retained_ids = candidate_ids[: query.max_children_inspected]
        has_more = len(child_keys) > len(retained_ids)
        entries: list[ChildSessionLifecycleEntry] = []
        unavailable_count = 0
        operation_records = state.session_operation_records[parent.id]
        for child_id in retained_ids:
            child = state.sessions.get(child_id)
            if child is None:
                raise RuntimeError("In-memory child-session index is inconsistent.")
            records_by_type = state.child_lifecycle_event_records.get(child_id, {})
            occurrence_source = child_lifecycle_rules._child_session_lifecycle_occurrence(
                child,
                records_by_type,
            )
            consumption_record = None
            if occurrence_source is not None:
                _relationship, occurrence = occurrence_source
                consumption_record = operation_records.get(
                    child_session_notification_storage_key(
                        child.instance_id,
                        occurrence.source_id,
                    )
                )
            entry = child_lifecycle_rules._child_session_lifecycle_entry(
                parent=parent,
                child=child,
                records_by_type=records_by_type,
                consumption_record=consumption_record,
            )
            if entry is None:
                unavailable_count += 1
            else:
                entries.append(entry)
        entries.sort(key=child_lifecycle_rules._child_session_lifecycle_entry_sort_key)
        return ChildSessionLifecyclePage(
            parent_session_id=parent.id,
            parent_session_instance_id=parent.instance_id,
            entries=tuple(entries),
            inspected_child_count=len(retained_ids),
            unavailable_child_count=unavailable_count,
            has_more=has_more,
        )


async def aggregate_operational_snapshot(
    read: LockedSessionQuery,
    filters: SessionAggregateFilter | None = None,
) -> SessionOperationalSnapshot:
    filters = copy_session_aggregate_filter(filters)
    session_query = session_query_from_aggregate_filter(filters)
    async with read() as state:
        as_of = state.ownership_clock()
        counts = {status: 0 for status in SessionStatus}
        total_count = 0
        for item_index, session in enumerate(state.sessions.values(), start=1):
            if session_query_rules._session_matches(session, session_query):
                counts[session.status] += 1
                total_count += 1
            if item_index % _IN_MEMORY_AGGREGATE_CANCELLATION_INTERVAL == 0:
                await _cooperate_with_in_memory_aggregate_cancellation()
        return SessionOperationalSnapshot(
            as_of=as_of,
            total_count=total_count,
            counts_by_status=SessionStatusCounts.model_validate(counts),
            accuracy=EXACT_AGGREGATE.model_copy(),
        )


async def aggregate_usage(
    read: LockedSessionQuery, query: UsageRollupQuery
) -> UsageRollupStoreResult:
    query = copy_usage_rollup_query(query)
    session_query = session_query_from_aggregate_filter(query.sessions)
    async with read() as state:
        as_of = state.ownership_clock()
        matching_session_count = 0
        active_session_count = 0
        for session in state.sessions.values():
            if not session_query_rules._session_matches(session, session_query):
                continue
            matching_session_count += 1
            active_session_count += session.status in {
                SessionStatus.PENDING,
                SessionStatus.RUNNING,
                SessionStatus.INTERRUPTING,
            }

        def matching_session_records() -> Iterable[
            tuple[str, SessionStatus, Iterable[EventRecord]]
        ]:
            for session_id, session in state.sessions.items():
                if session_query_rules._session_matches(session, session_query):
                    yield (
                        session_id,
                        session.status,
                        state.session_event_records.get(session_id, ()),
                    )

        return session_usage_rules._usage_rollup_from_session_records(
            session_records=matching_session_records,
            query=query,
            as_of=as_of,
            matching_session_count=matching_session_count,
            active_session_count=active_session_count,
        )


async def list_queued_dispatch_terminal_receipts(
    read: LockedSessionQuery,
    query: QueuedDispatchTerminalReceiptQuery | None = None,
) -> list[QueuedDispatchTerminalReceipt]:
    if query is None:
        query = QueuedDispatchTerminalReceiptQuery()
    elif type(query) is not QueuedDispatchTerminalReceiptQuery:
        raise TypeError(
            "Queued dispatch receipt queries must be QueuedDispatchTerminalReceiptQuery instances."
        )
    else:
        query = QueuedDispatchTerminalReceiptQuery(
            after_session_id=query.after_session_id,
            after_operation_id=query.after_operation_id,
            limit=query.limit,
        )
    cursor: tuple[str, str] | None = None
    if query.after_session_id is not None:
        assert query.after_operation_id is not None
        cursor = (query.after_session_id, query.after_operation_id)
    async with read() as state:
        start = (
            0
            if cursor is None
            else bisect_right(state.queued_dispatch_terminal_receipt_keys, cursor)
        )
        keys = state.queued_dispatch_terminal_receipt_keys[start : start + query.limit]
        return [state.queued_dispatch_terminal_receipts[key].model_copy(deep=True) for key in keys]


async def query_pending_actions(
    read: LockedSessionQuery,
    query: PendingActionQuery | None = None,
    *,
    checkpoint_root_guard: CheckpointRootFieldGuard | None = None,
) -> PendingActionListResult:
    from cayu.sessions.pending_actions import (
        PENDING_ACTION_CHECKPOINT_KEYS,
        pending_action_checkpoint_tool_call_count,
        pending_action_event_projection_bytes,
        pending_action_from_records,
        pending_action_matches_query,
        pending_action_source_is_invalid,
        project_pending_action_checkpoint,
        select_pending_action_indexed_records,
    )

    if query is None:
        query = PendingActionQuery()
    elif type(query) is not PendingActionQuery:
        raise TypeError("Pending-action queries must be PendingActionQuery instances.")
    else:
        query = query.model_copy(deep=True)

    candidate_limit = min(query.limit * 4, 800) + 1
    async with read() as state:
        if query.session_id is None:
            candidate_ids = state.pending_action_session_ids
        elif query.session_id in state.pending_action_session_ids:
            candidate_ids = {query.session_id}
        else:
            candidate_ids = set()
        candidates = [
            session
            for session_id in candidate_ids
            if (session := state.sessions.get(session_id)) is not None
            and session.status in query.statuses
            and (query.agent_name is None or session.agent_name == query.agent_name)
            and (
                query.environment_name is None or session.environment_name == query.environment_name
            )
        ]
        candidates = session_query_rules._sort_sessions(candidates, SessionOrder.UPDATED_AT_DESC)
        if query.cursor is not None:
            cursor_dt, cursor_id = decode_session_cursor(query.cursor)
            candidates = [
                session
                for session in candidates
                if session_query_rules._session_after_cursor(
                    session,
                    SessionOrder.UPDATED_AT_DESC,
                    cursor_dt,
                    cursor_id,
                )
            ]

        candidate_window = candidates[:candidate_limit]
        has_more_candidates = len(candidate_window) == candidate_limit
        inspected_candidates = candidate_window[: candidate_limit - 1]
        actions: list[PendingActionRecord] = []
        issues: list[PendingActionIssue] = []
        materialized_source_bytes = 0
        inspected = 0
        more_matching = False
        last_inspected_session: PendingActionSession | None = None
        for session in inspected_candidates:
            previous_last_inspected_session = last_inspected_session
            projected_session = PendingActionSession.from_session(session)
            checkpoint = state.checkpoints.get(session.id)
            if checkpoint_root_guard is not None:
                checkpoint_root_guard.validate(
                    session.id,
                    checkpoint_root_field_projection(
                        checkpoint,
                        checkpoint_root_guard.key,
                    ),
                )
            pending_checkpoint_source = (
                None
                if checkpoint is None
                else {
                    key: checkpoint[key]
                    for key in PENDING_ACTION_CHECKPOINT_KEYS
                    if checkpoint.get(key) is not None
                }
            )
            source_too_complex = (
                pending_action_checkpoint_tool_call_count(pending_checkpoint_source)
                > MAX_PENDING_ACTION_TOOL_CALLS
            )
            candidate_size = JsonUtf8SizeCounter(query.max_result_bytes)
            source_fits = (
                not source_too_complex
                and candidate_size.value(projected_session)
                and candidate_size.value(pending_checkpoint_source)
            )
            projected_checkpoint = None
            records: list[EventRecord] = []
            if source_fits:
                projected_checkpoint = project_pending_action_checkpoint(checkpoint)
                records, ledger_too_complex = select_pending_action_indexed_records(
                    projected_checkpoint,
                    state.pending_action_event_records.get(session.id, {}),
                    state.pending_action_latest_barrier_records.get(session.id),
                )
                source_fits = (
                    not ledger_too_complex
                    and all(
                        (projection_bytes := pending_action_event_projection_bytes(record)) is None
                        or projection_bytes <= query.max_result_bytes
                        for record in records
                    )
                    and all(candidate_size.value(record) for record in records)
                )
            else:
                ledger_too_complex = False
            candidate_source_bytes = query.max_result_bytes - candidate_size.remaining
            if source_fits and (
                materialized_source_bytes + candidate_source_bytes > query.max_result_bytes
            ):
                more_matching = True
                break

            inspected += 1
            last_inspected_session = projected_session
            if not source_fits:
                if len(actions) + len(issues) == query.limit:
                    more_matching = True
                    inspected -= 1
                    last_inspected_session = previous_last_inspected_session
                    break
                issues.append(
                    PendingActionIssue.ledger_too_complex(
                        projected_session,
                        max_events_per_call=MAX_PENDING_ACTION_LEDGER_EVENTS_PER_CALL,
                    )
                    if ledger_too_complex
                    else PendingActionIssue.source_too_complex(
                        projected_session,
                        max_tool_calls=MAX_PENDING_ACTION_TOOL_CALLS,
                    )
                    if source_too_complex
                    else PendingActionIssue.source_too_large(
                        projected_session,
                        max_bytes=query.max_result_bytes,
                    )
                )
                continue

            materialized_source_bytes += candidate_source_bytes
            action = pending_action_from_records(
                projected_session,
                records,
                projected_checkpoint,
            )
            if pending_action_source_is_invalid(
                projected_session,
                projected_checkpoint,
                action,
                records,
            ):
                if len(actions) + len(issues) == query.limit:
                    more_matching = True
                    inspected -= 1
                    last_inspected_session = previous_last_inspected_session
                    break
                issues.append(PendingActionIssue.source_invalid(projected_session))
                continue
            if action is None:
                continue
            if query.kind is not None and action.kind != query.kind:
                continue
            if not pending_action_matches_query(action, query.q):
                continue
            if len(actions) + len(issues) == query.limit:
                more_matching = True
                inspected -= 1
                last_inspected_session = previous_last_inspected_session
                break
            actions.append(action)

        has_more = more_matching or has_more_candidates
        cursor_session = last_inspected_session
        next_cursor = (
            encode_session_cursor(cursor_session, SessionOrder.UPDATED_AT_DESC)
            if has_more and cursor_session is not None
            else None
        )
        return enforce_pending_action_result_size(
            PendingActionListResult(
                actions=actions,
                issues=issues,
                next_cursor=next_cursor,
                has_more=has_more,
                # A cursor query sees only the suffix after that cursor, so a
                # final page's local length is not a global total. The first
                # page can report an exact total only when it exhausted the
                # candidate set without hitting either bound.
                total_count=(
                    len(actions) + len(issues) if query.cursor is None and not has_more else None
                ),
                inspected_candidate_count=inspected,
            ),
            max_bytes=query.max_result_bytes,
        )


async def list_sessions(
    read: LockedSessionQuery,
    query: SessionQuery | None,
    *,
    pending_interruption_cascade_only: bool,
    access_bounds: SessionReadBounds | None = None,
) -> SessionListResult:
    from cayu.sessions._access_rules import session_read_admitted

    query = copy_session_query(query)
    async with read() as state:
        inactive_before = (
            query.last_activity_before
            if query.inactive_for_seconds is None
            else utc_duration_cutoff(
                state.ownership_clock(),
                query.inactive_for_seconds,
            )
        )
        duration_matches_no_timestamp = (
            query.inactive_for_seconds is not None and inactive_before is None
        )
        base_query = query.model_copy(
            update={
                "debug_state": None,
                "last_activity_before": inactive_before,
                "inactive_for_seconds": None,
            }
        )
        candidates = (
            (
                state.sessions[session_id]
                for session_id, checkpoint in state.checkpoints.items()
                if "pending_interruption_cascade" in checkpoint and session_id in state.sessions
            )
            if pending_interruption_cascade_only
            else state.sessions.values()
        )
        matching = (
            []
            if duration_matches_no_timestamp
            else [
                session
                for session in candidates
                if (
                    access_bounds is None
                    or session_read_admitted(access_bounds, session, state.sessions.get)
                )
                and session_query_rules._session_matches(session, base_query)
                and _session_matches_debug_state(
                    session,
                    state.session_event_records.get(session.id, []),
                    query.debug_state,
                )
            ]
        )
        total = len(matching) if query.include_total_count else None
        ordered = session_query_rules._sort_sessions(matching, query.order_by)
        if query.cursor is not None:
            cursor_dt, cursor_id = decode_session_cursor(query.cursor)
            ordered = [
                session
                for session in ordered
                if session_query_rules._session_after_cursor(
                    session, query.order_by, cursor_dt, cursor_id
                )
            ]
            window = ordered[: query.limit + 1]
        else:
            window = ordered[query.offset : query.offset + query.limit + 1]
        page = window[: query.limit]
        next_cursor = session_next_cursor(page, len(window) > query.limit, query.order_by)
        return SessionListResult(
            sessions=[session.model_copy(deep=True) for session in page],
            next_cursor=next_cursor,
            total_count=total,
        )


def _session_matches_debug_state(
    session: Session,
    records: list[EventRecord],
    debug_state: SessionDebugState | None,
) -> bool:
    if debug_state is None:
        return True
    has_tool_debug_event = any(_is_tool_debug_event(record.event.type) for record in records)
    has_session_failure = session.status == SessionStatus.FAILED
    has_interruption = session.status == SessionStatus.INTERRUPTED
    if debug_state == SessionDebugState.TOOL_ISSUE:
        return has_tool_debug_event
    if debug_state == SessionDebugState.SESSION_FAILURE:
        return has_session_failure
    if debug_state == SessionDebugState.INTERRUPTION:
        return has_interruption
    if debug_state == SessionDebugState.NEEDS_ATTENTION:
        return has_session_failure or has_interruption or has_tool_debug_event
    return False


def _is_tool_debug_event(event_type: EventType | str) -> bool:
    return event_type in {EventType.TOOL_CALL_FAILED, EventType.TOOL_CALL_BLOCKED}

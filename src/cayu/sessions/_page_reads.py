"""Reads that take a page of session IDs, and their portable defaults.

A dashboard page summarizes many sessions at once. Native stores answer each of
these reads in a fixed number of queries for the whole page. The defaults here
loop over the single-session methods, so a store that implements only those
keeps working with the same results.
"""

from __future__ import annotations

import base64
import json
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Any

from cayu._validation import require_durable_clean_nonblank as require_clean_nonblank
from cayu.events import EventType
from cayu.sessions._run_identity import session_run_id, session_run_lineage
from cayu.sessions.invocation import SessionExecutionSource

if TYPE_CHECKING:
    from cayu.messages import MessageRole
    from cayu.sessions.authority import CheckpointRootFieldGuard
    from cayu.sessions.pending_action_contracts import (
        PendingActionListResult,
        PendingActionQuery,
    )
    from cayu.sessions.records import EventRecord, Session
    from cayu.sessions.store import SessionStore
    from cayu.sessions.summaries import SessionEventPositions, SessionOutcome, SessionStepCounts
    from cayu.sessions.transcript_queries import TranscriptRecord

# One overview page holds at most this many sessions (the session list limit).
MAX_PAGE_READ_SESSIONS = 1000


def page_session_ids(session_ids: Sequence[str]) -> tuple[str, ...]:
    """Validate a page of session IDs and drop repeats, keeping first-seen order."""

    if isinstance(session_ids, str | bytes) or not isinstance(session_ids, Iterable):
        raise TypeError("session_ids must be a sequence of session IDs.")
    page = tuple(dict.fromkeys(require_clean_nonblank(sid, "session_id") for sid in session_ids))
    if len(page) > MAX_PAGE_READ_SESSIONS:
        raise ValueError(f"A page read takes at most {MAX_PAGE_READ_SESSIONS} sessions.")
    return page


def checkpoint_field_keys(keys: Sequence[str]) -> tuple[str, ...]:
    """Validate the root checkpoint fields a page read selects, keeping first-seen order."""

    if isinstance(keys, str | bytes) or not isinstance(keys, Iterable):
        raise TypeError("keys must be a sequence of checkpoint field names.")
    selected = tuple(dict.fromkeys(require_clean_nonblank(key, "checkpoint field") for key in keys))
    if not selected:
        raise ValueError("keys must name at least one checkpoint field.")
    return selected


def page_sessions(parents: Sequence[Session]) -> tuple[Session, ...]:
    """Validate a page of sessions by ID, keeping the first of any repeated ID."""

    from cayu.sessions.records import Session

    if isinstance(parents, str | bytes) or not isinstance(parents, Iterable):
        raise TypeError("parents must be a sequence of sessions.")
    page: dict[str, Session] = {}
    for parent in parents:
        if not isinstance(parent, Session):
            raise TypeError("parents must be a sequence of sessions.")
        page.setdefault(parent.id, parent)
    page_session_ids(tuple(page))
    return tuple(page.values())


async def page_reads_are_bounded() -> bool:
    """Whether scoped access bounds apply, so reads must use the single-session checks."""

    from cayu.resource_access import current_data_bounds
    from cayu.sessions._access_rules import _query_bounds

    return _query_bounds.get() is not None or await current_data_bounds() is not None


def counts_as_child_run(parent: Session, child: Session) -> bool:
    """A direct, non-fork child whose retained lineage names this parent incarnation."""

    return (
        child.parent_session_id == parent.id
        and child.invocation.source is not SessionExecutionSource.FORK
        and session_run_lineage(child)[1] == session_run_id(parent)
    )


def count_child_runs_from(
    parents: Sequence[Session], children: Iterable[Session]
) -> dict[str, int]:
    """Count child runs per parent from the parents' direct children."""

    by_id = {parent.id: parent for parent in parents}
    runs: dict[str, set[str]] = {parent.id: set() for parent in parents}
    for child in children:
        parent = by_id.get(child.parent_session_id or "")
        if parent is not None and counts_as_child_run(parent, child):
            runs[parent.id].add(session_run_id(child))
    return {parent_id: len(seen) for parent_id, seen in runs.items()}


async def load_sessions_one_by_one(
    store: SessionStore, session_ids: Sequence[str]
) -> dict[str, Session]:
    loaded: dict[str, Session] = {}
    for session_id in page_session_ids(session_ids):
        session = await store.load(session_id)
        if session is not None:
            loaded[session_id] = session
    return loaded


async def load_checkpoints_one_by_one(
    store: SessionStore, session_ids: Sequence[str]
) -> dict[str, dict[str, Any]]:
    loaded: dict[str, dict[str, Any]] = {}
    for session_id in page_session_ids(session_ids):
        checkpoint = await store.load_checkpoint(session_id)
        if checkpoint is not None:
            loaded[session_id] = checkpoint
    return loaded


async def count_session_steps_one_by_one(
    store: SessionStore, session_ids: Sequence[str]
) -> dict[str, SessionStepCounts]:
    from cayu.sessions.summaries import SessionStepCounts

    counts: dict[str, SessionStepCounts] = {}
    for session_id in page_session_ids(session_ids):
        try:
            summary = await store.summarize_events(session_id)
        except KeyError:
            # As for outcomes: only a session that is not there is left out.
            if await store.load(session_id) is not None:
                raise
            continue
        by_type = summary.counts_by_type
        counts[session_id] = SessionStepCounts(
            model_steps=by_type.get(str(EventType.MODEL_COMPLETED), 0),
            tool_calls=by_type.get(str(EventType.TOOL_CALL_STARTED), 0),
        )
    return counts


async def summarize_event_positions_one_by_one(
    store: SessionStore, session_ids: Sequence[str], event_types: Sequence[EventType]
) -> dict[str, SessionEventPositions]:
    from cayu.sessions.event_queries import EventOrder, EventQuery
    from cayu.sessions.summaries import SessionEventPositions

    types = tuple(event_types)
    page = page_session_ids(session_ids)
    found = await store.load_session_instance_ids(page)
    positions: dict[str, SessionEventPositions] = {}
    for session_id in page:
        if session_id not in found:
            continue
        count = last = 0
        after = 0
        while True:
            heads = await store.query_event_heads(
                EventQuery(
                    session_id=session_id,
                    event_types=types,
                    after_sequence=after,
                    limit=5000,
                    order_by=EventOrder.SEQUENCE_ASC,
                )
            )
            if not heads:
                break
            count += len(heads)
            last = after = heads[-1].sequence
        positions[session_id] = SessionEventPositions(count=count, last_sequence=last)
    return positions


async def summarize_outcomes_one_by_one(
    store: SessionStore, session_ids: Sequence[str]
) -> dict[str, SessionOutcome]:
    outcomes: dict[str, SessionOutcome] = {}
    for session_id in page_session_ids(session_ids):
        try:
            outcomes[session_id] = await store.summarize_outcome(session_id)
        except KeyError:
            # The single read raises KeyError for a session that is not there.
            # Any other KeyError is a store failure and propagates unchanged.
            if await store.load(session_id) is not None:
                raise
    return outcomes


async def load_session_operations_one_by_one(
    store: SessionStore,
    session_ids: Sequence[str],
    idempotency_key: str,
    *,
    checkpoint_root_guard: CheckpointRootFieldGuard | None,
) -> dict[str, dict[str, Any] | None]:
    records: dict[str, dict[str, Any] | None] = {}
    for session_id in page_session_ids(session_ids):
        try:
            if checkpoint_root_guard is None:
                record = await store.load_session_operation(session_id, idempotency_key)
            else:
                record = await store.load_session_operation(
                    session_id, idempotency_key, checkpoint_root_guard=checkpoint_root_guard
                )
        except KeyError:
            # As for outcomes: only a missing session is an absence.
            if await store.load(session_id) is not None:
                raise
            continue
        records[session_id] = record
    return records


async def query_latest_events_one_by_one(
    store: SessionStore, session_ids: Sequence[str], event_types: Sequence[EventType]
) -> dict[str, EventRecord]:
    from cayu.sessions.event_queries import EventOrder, EventQuery

    types = tuple(event_types)
    latest: dict[str, EventRecord] = {}
    for session_id in page_session_ids(session_ids):
        records = await store.query_events(
            EventQuery(
                session_id=session_id,
                event_types=types,
                order_by=EventOrder.SEQUENCE_DESC,
                limit=1,
            )
        )
        if records:
            latest[session_id] = records[0]
    return latest


async def query_first_transcript_records_one_by_one(
    store: SessionStore, session_ids: Sequence[str], role: MessageRole
) -> dict[str, TranscriptRecord]:
    from cayu.sessions.transcript_queries import TranscriptQuery

    first: dict[str, TranscriptRecord] = {}
    for session_id in page_session_ids(session_ids):
        try:
            page = await store.query_transcript(
                TranscriptQuery(session_id=session_id, role=role, limit=1)
            )
        except KeyError:
            # As for outcomes: only a missing session is an absence.
            if await store.load(session_id) is not None:
                raise
            continue
        if page.records:
            first[session_id] = page.records[0]
    return first


async def count_child_runs_one_by_one(
    store: SessionStore, parents: Sequence[Session]
) -> dict[str, int]:
    from cayu.sessions.queries import SessionOrder, SessionQuery

    page = page_sessions(parents)
    counts: dict[str, int] = {}
    for parent in page:
        children: list[Session] = []
        cursors: set[str] = set()
        cursor = None
        while True:
            result = await store.list_sessions(
                SessionQuery(
                    parent_session_id=parent.id,
                    limit=200,
                    cursor=cursor,
                    order_by=SessionOrder.CREATED_AT_ASC,
                )
            )
            children.extend(result.sessions)
            if result.next_cursor is None:
                break
            if result.next_cursor in cursors:
                from cayu.sessions.overview import SessionOverviewUnavailable

                raise SessionOverviewUnavailable("Subagent paging did not advance.")
            cursors.add(result.next_cursor)
            cursor = result.next_cursor
        counts.update(count_child_runs_from((parent,), children))
    return counts


def _encode_page_cursor(index: int, inner: str | None) -> str:
    raw = json.dumps([index, inner], separators=(",", ":")).encode()
    return base64.urlsafe_b64encode(raw).decode("ascii")


def _decode_page_cursor(cursor: str | None, count: int) -> tuple[int, str | None]:
    if cursor is None:
        return 0, None
    try:
        index, inner = json.loads(base64.urlsafe_b64decode(cursor.encode("ascii")))
    except (ValueError, TypeError) as exc:
        raise ValueError("Invalid pending-action page cursor.") from exc
    if (
        type(index) is not int
        or not 0 <= index < count
        or (inner is not None and type(inner) is not str)
    ):
        raise ValueError("Invalid pending-action page cursor.")
    return index, inner


async def query_pending_actions_one_by_one(
    store: SessionStore,
    session_ids: Sequence[str],
    query: PendingActionQuery,
    *,
    checkpoint_root_guard: CheckpointRootFieldGuard | None,
) -> PendingActionListResult:
    """One page from the session the cursor names, continuing through the rest in order."""

    page = page_session_ids(session_ids)
    if not page:
        from cayu.sessions.pending_action_contracts import PendingActionListResult

        return PendingActionListResult()
    index, inner = _decode_page_cursor(query.cursor, len(page))
    single = query.model_copy(update={"session_id": page[index], "cursor": inner})
    if checkpoint_root_guard is None:
        result = await store.query_pending_actions(single)
    else:
        result = await store.query_pending_actions(
            single, checkpoint_root_guard=checkpoint_root_guard
        )
    if result.has_more:
        if result.next_cursor is None:
            # Leave the stall for the caller to detect, as a single query would.
            return result
        next_cursor: str | None = _encode_page_cursor(index, result.next_cursor)
    elif index + 1 < len(page):
        next_cursor = _encode_page_cursor(index + 1, None)
    else:
        next_cursor = None
    return result.model_copy(
        update={"next_cursor": next_cursor, "has_more": next_cursor is not None}
    )

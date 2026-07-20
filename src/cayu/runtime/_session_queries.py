from __future__ import annotations

from cayu.runtime.sessions import (
    EventOrder,
    EventQuery,
    EventRecord,
    Session,
    SessionQuery,
    SessionStore,
)


async def query_all_sessions(
    session_store: SessionStore,
    query: SessionQuery,
) -> list[Session]:
    """Load every page for a session query without dropping its filters."""
    sessions: list[Session] = []
    offset = query.offset
    cursor = query.cursor
    while True:
        page = await session_store.list_sessions(
            query.model_copy(
                update={
                    "offset": 0 if cursor is not None else offset,
                    "cursor": cursor,
                    "include_total_count": False,
                }
            )
        )
        sessions.extend(page.sessions)
        if cursor is not None:
            if page.next_cursor is None:
                return sessions
            cursor = page.next_cursor
            continue
        if len(page.sessions) < query.limit:
            return sessions
        offset += len(page.sessions)


async def query_all_event_records(
    session_store: SessionStore,
    query: EventQuery,
) -> list[EventRecord]:
    """Load every page for an event query without dropping its filters."""

    records: list[EventRecord] = []
    single_session = query.session_id is not None or len(query.session_ids) == 1
    reverse_results = query.order_by == EventOrder.SEQUENCE_DESC and not single_session
    page_order = EventOrder.SEQUENCE_ASC if reverse_results else query.order_by
    after_sequence = query.after_sequence
    before_sequence = query.before_sequence

    def ordered_records() -> list[EventRecord]:
        if reverse_results:
            records.reverse()
        return records

    while True:
        page = await session_store.query_events(
            EventQuery(
                session_id=query.session_id,
                session_ids=query.session_ids,
                event_id=query.event_id,
                causal_budget_id=query.causal_budget_id,
                event_type=query.event_type,
                event_types=query.event_types,
                exclude_event_types=query.exclude_event_types,
                agent_name=query.agent_name,
                environment_name=query.environment_name,
                workflow_name=query.workflow_name,
                tool_name=query.tool_name,
                since=query.since,
                until=query.until,
                after_sequence=after_sequence,
                before_sequence=before_sequence,
                limit=query.limit,
                order_by=page_order,
            )
        )
        if not page:
            return ordered_records()
        records.extend(page)
        if len(page) < query.limit:
            return ordered_records()
        if page_order == EventOrder.SEQUENCE_ASC:
            after_sequence = page[-1].sequence
        else:
            before_sequence = page[-1].sequence

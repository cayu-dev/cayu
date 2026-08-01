from __future__ import annotations

from cayu.core import Event
from cayu.runtime import EventQuery, SessionStore
from cayu.runtime._event_projection import public_event_sequence


async def private_event_for_public_event(store: SessionStore, event: Event) -> Event:
    """Resolve one projected app event to its exact durable private record."""

    sequence = public_event_sequence(event.id)
    assert sequence is not None
    records = await store.query_events(EventQuery(session_id=event.session_id))
    matches = [record.event for record in records if record.sequence == sequence]
    assert len(matches) == 1
    return matches[0]


async def private_events_for_public_events(
    store: SessionStore,
    events: list[Event],
) -> list[Event]:
    records_by_session: dict[str, dict[int, Event]] = {}
    private_events: list[Event] = []
    for event in events:
        sequence = public_event_sequence(event.id)
        assert sequence is not None
        if event.session_id not in records_by_session:
            records = await store.query_events(EventQuery(session_id=event.session_id))
            records_by_session[event.session_id] = {
                record.sequence: record.event for record in records
            }
        private_event = records_by_session[event.session_id].get(sequence)
        assert private_event is not None
        private_events.append(private_event)
    return private_events

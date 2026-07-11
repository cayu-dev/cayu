from __future__ import annotations

from collections.abc import Iterable

from cayu.core.events import Event, EventType
from cayu.runtime.budgets import BudgetStore
from cayu.runtime.event_sinks import EventSink
from cayu.runtime.sessions import SessionStore


class RuntimeEventWriter:
    """Persist runtime events and fan them out to configured sinks."""

    def __init__(
        self,
        *,
        session_store: SessionStore,
        budget_store: BudgetStore,
        event_sinks: Iterable[EventSink],
    ) -> None:
        self._session_store = session_store
        self._budget_store = budget_store
        self._event_sinks = tuple(event_sinks)

    async def emit(self, event: Event) -> Event:
        await self._session_store.append_event(event.session_id, event)
        if event.type == EventType.MODEL_COMPLETED:
            await self._budget_store.append_event(event)
        await self._emit_to_sinks(event)
        return event

    async def emit_many(self, session_id: str, events: list[Event]) -> list[Event]:
        """Persist and fan out a defensive copy of one event batch.

        Unlike ``emit``, batch emission does not apply event-specific runtime
        side effects: in particular, it does not forward ``model.completed`` to
        the budget store. Callers must not route that event type through this
        interface.
        """
        if type(events) is not list:
            raise TypeError("Runtime events must be a list.")
        copied_events: list[Event] = []
        for event in events:
            if type(event) is not Event:
                raise TypeError("Runtime events must be Event instances.")
            if event.session_id != session_id:
                raise ValueError("Event session_id does not match target session.")
            copied_events.append(event.model_copy(deep=True))

        await self._session_store.append_events(session_id, copied_events)
        for event in copied_events:
            await self._emit_to_sinks(event)
        return copied_events

    async def _emit_to_sinks(self, event: Event) -> None:
        for sink in self._event_sinks:
            try:
                await sink.emit(event.model_copy(deep=True))
            except Exception as exc:
                await self._session_store.append_event(
                    event.session_id,
                    Event(
                        type=EventType.RUNTIME_SINK_FAILED,
                        session_id=event.session_id,
                        agent_name=event.agent_name,
                        environment_name=event.environment_name,
                        payload={
                            "sink": type(sink).__name__,
                            "error": str(exc),
                            "error_type": type(exc).__name__,
                            "event_id": event.id,
                            "event_type": str(event.type),
                        },
                    ),
                )

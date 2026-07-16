from __future__ import annotations

from abc import ABC, abstractmethod

from cayu.core.events import Event, copy_event


class EventSink(ABC):
    """Destination for at-least-once runtime event delivery.

    A crash after ``emit`` returns but before Cayu records completion can cause
    the same immutable event to be retried. Durable sinks should use
    ``(event.session_id, event.id)`` as their idempotency identity.
    """

    @abstractmethod
    async def emit(self, event: Event) -> None:
        """Emit one event."""


class InMemoryEventSink(EventSink):
    def __init__(self) -> None:
        self.events: list[Event] = []
        self._events_by_id: dict[tuple[str, str], Event] = {}

    async def emit(self, event: Event) -> None:
        if type(event) is not Event:
            raise TypeError("Event sinks require Event instances.")
        copied = copy_event(event)
        identity = (copied.session_id, copied.id)
        existing = self._events_by_id.get(identity)
        if existing is not None:
            if existing != copied:
                raise ValueError(
                    "Event identity was reused with conflicting contents: "
                    f"{copied.session_id}/{copied.id}"
                )
            return
        self._events_by_id[identity] = copied
        self.events.append(copied)

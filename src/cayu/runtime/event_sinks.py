from __future__ import annotations

from abc import ABC, abstractmethod

from cayu.core.events import Event


class EventSink(ABC):
    """Destination for runtime events."""

    @abstractmethod
    async def emit(self, event: Event) -> None:
        """Emit one event."""


class InMemoryEventSink(EventSink):
    def __init__(self) -> None:
        self.events: list[Event] = []

    async def emit(self, event: Event) -> None:
        self.events.append(event)

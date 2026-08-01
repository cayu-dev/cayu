from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from cayu.core.events import Event, copy_event


@dataclass(frozen=True, slots=True)
class _EventSinkDelivery:
    """Safe public event plus private in-process correlation for built-in sinks."""

    event: Event
    event_sequence: int
    private_session_id: str
    private_event_id: str
    private_interaction_id: str | None = None
    private_tool_call_id: str | None = None
    private_parent_session_id: str | None = None

    def __post_init__(self) -> None:
        if type(self.event) is not Event:
            raise TypeError("Event sink deliveries require Event instances.")
        if type(self.event_sequence) is not int or self.event_sequence < 1:
            raise ValueError("event_sequence must be an integer greater than or equal to 1.")
        for field_name in ("private_session_id", "private_event_id"):
            value = getattr(self, field_name)
            if type(value) is not str or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string.")
        for field_name in (
            "private_interaction_id",
            "private_tool_call_id",
            "private_parent_session_id",
        ):
            value = getattr(self, field_name)
            if value is not None and (type(value) is not str or not value.strip()):
                raise ValueError(f"{field_name} must be a non-empty string when provided.")
        object.__setattr__(self, "event", copy_event(self.event))


class EventSink(ABC):
    """Destination for at-least-once runtime event delivery.

    A crash after ``emit`` returns but before Cayu records completion can cause
    the same immutable public event to be retried. Public event identities are
    presentation aliases, not execution, claim, or delivery authority. Built-in
    sinks that require genuine duplicate suppression are dispatched through
    exact runtime-owned adapters that cannot be overridden by subclasses.
    """

    @abstractmethod
    async def emit(self, event: Event) -> None:
        """Emit one event."""


class InMemoryEventSink(EventSink):
    def __init__(self) -> None:
        self.events: list[Event] = []
        self._events_by_id: dict[tuple[str, str], Event] = {}

    async def emit(self, event: Event) -> None:
        self._record(event, identity=(event.session_id, event.id))

    def _record(
        self,
        event: Event,
        *,
        identity: tuple[str, str],
    ) -> None:
        if type(event) is not Event:
            raise TypeError("Event sinks require Event instances.")
        copied = copy_event(event)
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


async def _emit_in_memory_delivery(
    sink: InMemoryEventSink,
    delivery: _EventSinkDelivery,
) -> None:
    """Deliver private deduplication identity only to the exact built-in sink."""

    if type(sink) is not InMemoryEventSink or type(delivery) is not _EventSinkDelivery:
        raise TypeError("In-memory private delivery requires exact built-in types.")
    sink._record(
        delivery.event,
        identity=(delivery.private_session_id, delivery.private_event_id),
    )

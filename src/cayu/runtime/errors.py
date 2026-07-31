"""Public runtime failure contracts."""

from __future__ import annotations

from cayu.core.events import Event, copy_event


class TerminalEventPublicationUncertain(RuntimeError):
    """The runtime cannot determine whether terminal evidence is durable."""

    def __init__(
        self,
        *,
        event: Event,
        publication_failure: Exception,
        reconciliation_failure: Exception,
    ) -> None:
        self.event = copy_event(event)
        self.session_id = self.event.session_id
        self.event_id = self.event.id
        self.failures = ExceptionGroup(
            "Terminal event publication and reconciliation both failed.",
            [publication_failure, reconciliation_failure],
        )
        super().__init__(
            "Terminal event publication outcome is uncertain; durable session "
            "state was preserved for recovery."
        )

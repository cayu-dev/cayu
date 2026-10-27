"""Public runtime failure contracts."""

from __future__ import annotations

from cayu._resource_access_errors import ResourceAccessDenied as ResourceAccessDenied
from cayu._validation import DurableValueError as DurableValueError
from cayu._validation import extract_durable_value_error as extract_durable_value_error
from cayu.deadlines import ExecutionDeadlineExceeded as ExecutionDeadlineExceeded
from cayu.events import Event, copy_event

__all__ = [
    "DurableValueError",
    "EnvironmentInUse",
    "ExecutionDeadlineExceeded",
    "InteractionLifecyclePublicationRejected",
    "ResourceAccessDenied",
    "TerminalEventPublicationUncertain",
    "extract_durable_value_error",
]


class EnvironmentInUse(RuntimeError):
    """``CayuApp.unregister_environment`` refused because something may still need it.

    ``reason`` is one of ``default_environment``, ``agent_default``,
    ``in_process_work``, ``unfinished_session`` or ``pending_cleanup``. Nothing
    was changed; retry once the named work has finished.
    """

    def __init__(self, environment_name: str, reason: str, detail: str) -> None:
        self.environment_name = environment_name
        self.reason = reason
        super().__init__(f"Environment {environment_name!r} cannot be unregistered: {detail}.")


class InteractionLifecyclePublicationRejected(RuntimeError):
    """A terminal interaction summary could not be represented durably."""

    def __init__(
        self,
        *,
        session_id: str,
        interaction_id: str,
        _runtime_provenance: tuple[object, str, str] | None = None,
    ) -> None:
        self.session_id = session_id
        self.interaction_id = interaction_id
        self._runtime_provenance = _runtime_provenance
        super().__init__(
            "Terminal interaction lifecycle evidence was rejected; durable run "
            "state was preserved for recovery or manual reconciliation."
        )


def _runtime_interaction_lifecycle_publication_rejected(
    *,
    session_id: str,
    interaction_id: str,
    runtime_authority: object,
) -> InteractionLifecyclePublicationRejected:
    return InteractionLifecyclePublicationRejected(
        session_id=session_id,
        interaction_id=interaction_id,
        _runtime_provenance=(runtime_authority, session_id, interaction_id),
    )


def _is_runtime_interaction_lifecycle_publication_rejection(
    error: BaseException,
    *,
    session_id: str,
    interaction_id: str | None,
    runtime_authority: object,
) -> bool:
    if type(error) is not InteractionLifecyclePublicationRejected:
        return False
    try:
        provenance = error._runtime_provenance
    except (AttributeError, TypeError):
        return False
    if (
        type(provenance) is not tuple
        or len(provenance) != 3
        or type(provenance[1]) is not str
        or type(provenance[2]) is not str
    ):
        return False
    return provenance[0] is runtime_authority and provenance[1:] == (session_id, interaction_id)


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

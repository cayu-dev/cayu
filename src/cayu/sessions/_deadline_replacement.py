"""Shared rules for replacing a stored session's execution deadline between runs.

Every built-in store applies them inside its own write owner, so the replaced
deadline and the fence check commit together.
"""

from __future__ import annotations

from typing import Any

from cayu._validation import copy_session_metadata
from cayu.deadlines import (
    EXECUTION_DEADLINE_METADATA_KEY,
    EXECUTION_DEADLINE_METADATA_KEYS,
    EXECUTION_DEADLINE_ORIGIN_METADATA_KEY,
    EXECUTION_DEADLINE_RENEWAL_METADATA_KEY,
    TASK_EVENT_BINDING_METADATA_KEY,
    TASK_EVENT_DEADLINE_METADATA_KEY,
    ExecutionDeadline,
    ExecutionDeadlineRenewal,
    current_task_event_deadline,
    current_task_event_id,
    deadline_from_metadata,
    deadline_metadata,
    effective_deadline,
    task_event_deadline_from_metadata,
)
from cayu.sessions._invocation_scope import _current_session_run_epoch
from cayu.sessions.authority import SessionRunFenced
from cayu.sessions.records import SessionStatus, SessionStatusConflict

# A session in one of these states has a run in progress (or about to start),
# and that run already holds the deadline it was admitted with.
RUN_IN_PROGRESS_STATUSES = frozenset(
    {SessionStatus.PENDING, SessionStatus.RUNNING, SessionStatus.INTERRUPTING}
)


def require_execution_deadline_replacement(
    session_id: str,
    *,
    status: SessionStatus,
    session_instance_id: str | None,
    run_epoch: int,
    expected_session_instance_id: str | None,
    expected_run_epoch: int,
) -> None:
    """Refuse a replacement unless the session is the one read and the caller may change it.

    That is between runs, or from inside the run that was just admitted with
    this deadline change (the caller holds that run's epoch fence), so a
    settle stores its deadline only after its decision was admitted.
    """

    if session_instance_id != expected_session_instance_id or run_epoch != expected_run_epoch:
        raise SessionRunFenced(
            f"Session {session_id} changed before its execution deadline could be replaced."
        )
    if status is SessionStatus.RUNNING and _current_session_run_epoch(session_id) == run_epoch:
        return
    if status in RUN_IN_PROGRESS_STATUSES:
        raise SessionStatusConflict(
            f"Session {session_id} is {status.value}; its execution deadline can be replaced "
            "only between runs or by the run that was admitted with it."
        )


def require_replacement_deadline(
    deadline: object, renewal: object = None
) -> tuple[ExecutionDeadline, ExecutionDeadlineRenewal | None]:
    if type(deadline) is not ExecutionDeadline:
        raise TypeError("deadline must be an ExecutionDeadline.")
    if renewal is not None and type(renewal) is not ExecutionDeadlineRenewal:
        raise TypeError("renewal must be an ExecutionDeadlineRenewal.")
    return deadline, renewal


def continuation_event_metadata(
    metadata: dict[str, Any],
    status: SessionStatus | None,
    *,
    explicit: bool,
    allow_rebind: bool = False,
) -> dict[str, Any]:
    """A settled turn can accept a new event; pending human waits keep their bound."""
    binding = metadata.get(TASK_EVENT_BINDING_METADATA_KEY)
    event_id = current_task_event_id()
    if (
        status is not SessionStatus.COMPLETED
        and not (allow_rebind and status in {SessionStatus.FAILED, SessionStatus.INTERRUPTED})
    ) or not isinstance(binding, dict):
        return metadata
    if not (
        (event_id is not None and event_id != binding["event_id"])
        or (event_id is None and explicit)
    ):
        return metadata
    restored = {k: v for k, v in metadata.items() if k not in EXECUTION_DEADLINE_METADATA_KEYS}
    restored[EXECUTION_DEADLINE_METADATA_KEY] = binding["base_deadline"]
    if binding.get("base_origin") is not None:
        restored[EXECUTION_DEADLINE_ORIGIN_METADATA_KEY] = binding["base_origin"]
    return restored


def metadata_with_execution_deadline(
    metadata: dict[str, Any],
    deadline: ExecutionDeadline,
    renewal: ExecutionDeadlineRenewal | None = None,
    *,
    status: SessionStatus | None = None,
) -> dict[str, Any]:
    """Session metadata with ``deadline``, its origin and renewal in place of the stored ones."""

    metadata = continuation_event_metadata(
        metadata,
        status,
        explicit=deadline.origin == "explicit",
        allow_rebind=deadline._event_rebinding_allowed,
    )
    binding = metadata.get(TASK_EVENT_BINDING_METADATA_KEY)
    replaces_base = deadline.origin == "explicit" or renewal is not None
    base = deadline if replaces_base else deadline_from_metadata(metadata)
    replaced = {
        key: value for key, value in metadata.items() if key not in EXECUTION_DEADLINE_METADATA_KEYS
    }
    event = effective_deadline(
        task_event_deadline_from_metadata(metadata),
        current_task_event_deadline(),
    )
    deadline = effective_deadline(deadline, event)
    replaced.update(deadline_metadata(deadline))
    if event.expires_at is not None:
        replaced[TASK_EVENT_DEADLINE_METADATA_KEY] = event.model_dump(mode="json")
        if binding is not None and not replaces_base:
            replaced[TASK_EVENT_BINDING_METADATA_KEY] = binding
        else:
            replaced[TASK_EVENT_BINDING_METADATA_KEY] = {
                "event_id": binding["event_id"] if binding is not None else current_task_event_id(),
                "base_deadline": base.model_dump(mode="json"),
                "base_origin": base.origin,
            }
    if renewal is not None and deadline.expires_at is not None:
        replaced[EXECUTION_DEADLINE_RENEWAL_METADATA_KEY] = renewal.model_dump(mode="json")
    return copy_session_metadata(replaced)

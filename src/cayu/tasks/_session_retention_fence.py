"""Invocation ordering for provisional task-retention claims."""

from cayu._validation import require_clean_nonblank
from cayu.sessions.invocation import SessionInvocationBinding
from cayu.tasks.records import TaskSessionClosureClaim


class RetentionEpochConflict(ValueError):
    """The predecessor changed after the caller checked session authority."""


def retention_epoch(
    session_id: str, session_instance_id: str, run_epoch: int
) -> tuple[str, str, int]:
    session_id = require_clean_nonblank(session_id, "session_id")
    SessionInvocationBinding.validate_session_instance_id(session_instance_id)
    if type(run_epoch) is not int or run_epoch < 0:
        raise ValueError("Retention epoch must be a non-negative integer.")
    return session_id, session_instance_id, run_epoch


def require_retention_predecessor(
    floor: tuple[str, int] | None,
    session_instance_id: str,
    expected_session_instance_id: str | None,
) -> None:
    if expected_session_instance_id is not None:
        SessionInvocationBinding.validate_session_instance_id(expected_session_instance_id)
    current = None if floor is None else floor[0]
    if current != session_instance_id and current != expected_session_instance_id:
        raise RetentionEpochConflict("Task retention incarnation changed before epoch admission.")


def require_current_retention_claim(
    claim: TaskSessionClosureClaim, floor: tuple[str, int] | None
) -> None:
    if floor is not None and retention_claim_superseded(claim, floor[0], floor[1]):
        raise ValueError("Task retention claim belongs to a superseded invocation.")


def retention_claim_superseded(
    claim: TaskSessionClosureClaim, instance_id: str, run_epoch: int
) -> bool:
    return claim.retention_run_epoch is not None and (
        claim.retention_instance_id != instance_id or claim.retention_run_epoch < run_epoch
    )

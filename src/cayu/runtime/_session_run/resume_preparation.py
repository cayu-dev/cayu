"""Validated resume input retained across abandoned-owner recovery."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from cayu.budgets.base import BudgetPolicy
from cayu.collaboration.access import CollaborationAccessContext
from cayu.execution_profiles import ExecutionProfileIdentity
from cayu.runtime._external_execution_to_wait import _ExternalExecutionToWait
from cayu.sessions._foreground_child_checkpoint import (
    ForegroundChildTerminal,
    ForegroundChildWait,
    ForegroundParentContinuation,
)
from cayu.sessions.authority import SessionRunFenced
from cayu.sessions.records import Session
from cayu.sessions.requests import ResumeRequest

if TYPE_CHECKING:
    from cayu.runtime._session_continuation_resume import _ResumeAdmissionHandoff


def _require_expected_resume_run_epoch(session: Session, request: ResumeRequest) -> None:
    expected = request.expected_run_epoch
    if expected is not None and session.run_epoch != expected:
        raise SessionRunFenced(
            f"Resume expected run epoch {expected}, but the session is at {session.run_epoch}."
        )


@dataclass(frozen=True)
class _ResumePreparation:
    """Validated continuation input retained across abandoned-owner recovery."""

    request: ResumeRequest
    task_id: str | None
    start_event_payload_extra: dict[str, Any]
    start_task_on_enter: bool
    adoption_request_fingerprint: str | None
    source_execution_profile: ExecutionProfileIdentity | None
    required_execution_profile: ExecutionProfileIdentity | None
    required_session_instance_fingerprint: str | None
    required_task_session_instance_id: str | None
    run_operation_id: str | None
    terminal_event_id: str | None
    queue_task_id: str | None
    queued_dispatch_id: str | None
    required_foreground_wait: ForegroundChildWait | None
    required_foreground_terminal: ForegroundChildTerminal | None
    required_foreground_continuation: ForegroundParentContinuation | None
    foreground_before_mutation: Callable[[], Awaitable[None]] | None
    continuation_handoff: _ResumeAdmissionHandoff | None
    participant_context: CollaborationAccessContext | None
    execution_to_wait: _ExternalExecutionToWait | None
    budget_policy: BudgetPolicy | None
    loaded_session: Session
    recovery_required: bool
    recovered_run_epoch: int | None = None

"""Exact external-wait handoff to the existing native invocation admission."""

from __future__ import annotations

from collections.abc import Iterable
from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any

from cayu.runtime._invocation_lifecycle import invocation_checkpoint_state_sha256
from cayu.sessions._invocation_lifecycle import (
    AdmitInvocationCommand,
    _invocation_lifecycle_command_sha256,
)
from cayu.sessions.external_waits import (
    ExternalWaitConflict,
    ExternalWaitExecution,
    ExternalWaitExecutionIntent,
    ExternalWaitRecord,
    ExternalWaitRegistration,
)

if TYPE_CHECKING:
    from cayu.sessions._external_wait_transition import ExternalWaitMutation
    from cayu.sessions.base import Session

_RESUME: ContextVar[AdmitInvocationCommand | None] = ContextVar(
    "external_wait_resume", default=None
)
_ADMISSION: ContextVar[tuple[ExternalWaitRegistration, ExternalWaitExecution] | None] = ContextVar(
    "external_wait_admission", default=None
)


def current_external_admission():
    return _ADMISSION.get()


@contextmanager
def external_resume_preparation(command: AdmitInvocationCommand):
    if type(command) is not AdmitInvocationCommand:
        raise TypeError("External resume requires a native admission command.")
    token = _RESUME.set(command)
    try:
        yield
    finally:
        _RESUME.reset(token)


def resume_identity(intent: ExternalWaitExecutionIntent) -> AdmitInvocationCommand:
    command = _RESUME.get()
    if (
        command is None
        or intent.admission_sha256 != _invocation_lifecycle_command_sha256(command)
        or intent.session_id != command.session_id
        or intent.expected_session_instance_id != command.expected_session_instance_id
        or intent.expected_run_epoch != command.expected_run_epoch
        or intent.profile_sha256 != command.target_active_profile.profile.fingerprint
        or command.interaction_started_event is None
        or command.continued_interaction_id is not None
    ):
        raise PermissionError("External resume preparation requires its exact runtime command.")
    return command


def require_resume_preparation(
    mutation: ExternalWaitMutation,
    current: ExternalWaitRecord | None,
    session: Session | None,
    checkpoint: dict[str, Any] | None,
) -> None:
    intent = mutation.execution_intent
    assert intent is not None
    command = resume_identity(intent)
    if current is not None and current.execution is not None:
        # The transition owner compares the full retained intent on replay.
        return
    if (
        session is None
        or session.instance_id != intent.expected_session_instance_id
        or session.run_epoch != intent.expected_run_epoch
        or session.status not in command.expected_statuses
        or invocation_checkpoint_state_sha256(checkpoint) != command.expected_checkpoint_sha256
    ):
        raise ExternalWaitConflict("External resume source changed before preparation.")
    from cayu.sessions._session_continuation_store import require_admission_claim

    require_admission_claim(session, checkpoint, command)


@contextmanager
def external_resume_admission(
    registration: ExternalWaitRegistration,
    execution: ExternalWaitExecution,
    command: AdmitInvocationCommand,
):
    if execution.intent.admission_sha256 != _invocation_lifecycle_command_sha256(command):
        raise ExternalWaitConflict("External resume admission differs from preparation.")
    token = _ADMISSION.set((registration, execution))
    try:
        yield
    finally:
        _ADMISSION.reset(token)


def require_external_admission(records: Iterable[ExternalWaitRecord], session: Session) -> None:
    from cayu.sessions.base import SessionRunFenced

    expected = _ADMISSION.get()
    matched = False
    for record in records:
        if expected is None:
            raise SessionRunFenced("External execution preparation requires reconciliation.")
        registration, execution = expected
        if (
            record.registration != registration
            or record.execution != execution
            or execution.intent.mode != "resume"
            or record.execution_excluded
            or record.continuation is not None
            or session.id != execution.intent.session_id
            or session.instance_id != execution.session_instance_id
            or session.run_epoch != execution.intent.expected_run_epoch
            or (record.outcome is not None and record.outcome.kind in {"cancelled", "unavailable"})
            or matched
        ):
            raise SessionRunFenced("External resume admission is excluded or conflicts.")
        matched = True
    if expected is not None and not matched:
        raise SessionRunFenced("External resume admission lacks retained preparation.")

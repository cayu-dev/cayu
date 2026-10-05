"""External preparation fencing inside the existing native CREATE transaction."""

from contextlib import contextmanager
from contextvars import ContextVar

from cayu.sessions.external_waits import (
    ExternalWaitConflict,
    ExternalWaitExecution,
    ExternalWaitRecord,
    ExternalWaitRegistration,
    ExternalWaitUnavailable,
)

_CREATION: ContextVar[tuple[ExternalWaitRegistration, ExternalWaitExecution] | None] = ContextVar(
    "external_wait_creation", default=None
)


@contextmanager
def external_creation_scope(
    registration: ExternalWaitRegistration, execution: ExternalWaitExecution
):
    token = _CREATION.set((registration, execution))
    try:
        yield
    finally:
        _CREATION.reset(token)


def current_external_creation():
    return _CREATION.get()


def require_external_creation(record: ExternalWaitRecord | None, request) -> None:
    from cayu.sessions.base import _authenticated_session_instance_id_for_run_request

    expected = current_external_creation()
    if expected is None:
        raise PermissionError("External creation requires its runtime owner.")
    registration, execution = expected
    if (
        execution.intent.mode != "run"
        or request.session_id != execution.intent.session_id
        or _authenticated_session_instance_id_for_run_request(
            request, session_id=execution.intent.session_id
        )
        != execution.session_instance_id
        or record is None
        or record.registration != registration
        or record.execution != execution
        or record.execution_excluded
        or record.continuation is not None
        or (record.outcome is not None and record.outcome.kind in {"cancelled", "unavailable"})
    ):
        raise ExternalWaitConflict("External creation is excluded or its preparation changed.")


def require_execution_exclusion(command, current, session, checkpoint=None) -> None:
    from cayu.runtime._external_wait_execution_scope import require_execution_preparation

    require_execution_preparation(command)
    if current is None or current.execution is None:
        raise ExternalWaitConflict("External execution preparation is unavailable.")
    if current.execution_excluded:
        return
    if current.execution.intent.mode == "resume":
        if session is None or session.instance_id != current.execution.session_instance_id:
            raise ExternalWaitConflict("External resume source is unavailable for exclusion.")
        if session.run_epoch == current.execution.intent.expected_run_epoch:
            # Same locked frontier: the exact prepared ADMIT has not consumed
            # it. The exclusion tombstone rejects its delayed scoped dispatch.
            return
    if session is not None and session.instance_id == current.execution.session_instance_id:
        _require_released_execution(current, session, checkpoint)
    # Absence is only meaningful while holding the very same external scope lock
    # used by CREATE. Committing exclusion prevents every delayed scoped creator.


def _require_released_execution(record: ExternalWaitRecord, session, checkpoint) -> None:
    """Discharge preparation only after native writer release, never on absence alone."""
    from cayu.sessions._invocation_lifecycle import (
        InvocationLifecycleCommandKind,
        _invocation_lifecycle_receipt_from_checkpoint,
        require_released_invocation_command_authority,
    )
    from cayu.sessions._session_continuation_store import ROOT_KEY, ContinuationRoot
    from cayu.sessions.base import SessionRunFenced

    execution = record.execution
    assert execution is not None
    epoch = 1
    kind = InvocationLifecycleCommandKind.CREATE
    if execution.intent.mode == "resume":
        assert execution.intent.expected_run_epoch is not None
        epoch = execution.intent.expected_run_epoch + 1
        kind = InvocationLifecycleCommandKind.ADMIT
    release = _invocation_lifecycle_receipt_from_checkpoint(
        checkpoint,
        command_identity=(
            f"release:{execution.intent.session_id}:{execution.session_instance_id}:"
            f"{session.run_epoch - 1}"
        ),
    )
    if release is None:
        raise ExternalWaitUnavailable("External execution is awaiting native writer release.")
    from cayu.sessions.external_waits import external_wait_digest

    origin = release.external_execution_origin
    if (
        release.kind is not InvocationLifecycleCommandKind.RELEASE
        or release.session_id != execution.intent.session_id
        or release.session_instance_id != execution.session_instance_id
        or release.active_profile.interaction_id != execution.interaction_id
        or release.active_profile.profile.fingerprint != execution.intent.profile_sha256
        or origin is None
        or record.registration is None
        or origin.registration_sha256 != external_wait_digest(record.registration)
        or origin.execution_sha256 != external_wait_digest(execution)
        or origin.admission_kind != kind.value
        or origin.admission_epoch != epoch
        or (
            execution.intent.mode == "resume"
            and origin.admission_sha256 != execution.intent.admission_sha256
        )
    ):
        raise ExternalWaitConflict("External execution release conflicts with its native writer.")
    try:
        # Each native REBIND/RELEASE authenticates its immediate predecessor
        # before carrying this constant-size origin forward. Historical ledger
        # compaction cannot discard the proof needed by an unresolved handoff.
        require_released_invocation_command_authority(
            session,
            checkpoint,
            session_id=execution.intent.session_id,
            session_instance_id=execution.session_instance_id,
            active_profile=release.active_profile,
        )
    except SessionRunFenced as exc:
        raise ExternalWaitConflict(
            "External execution lacks exact recovered writer-release evidence."
        ) from exc
    root = None if checkpoint is None else checkpoint.get(ROOT_KEY)
    if root is not None:
        index = ContinuationRoot.model_validate(root)
        if (
            index.namespace.session_id != session.id
            or index.namespace.session_instance_id != session.instance_id
            or any(entry.originating_writer_generation >= epoch for entry in index.entries)
        ):
            # A ticket has its own retirement/acknowledgement obligation. Repair
            # its external binding instead of erasing that handoff here.
            raise ExternalWaitConflict("Created execution has native continuation responsibility.")

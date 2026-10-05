"""Carry bounded pre-ticket cleanup proof through native lifecycle transactions."""

from cayu.sessions._invocation_lifecycle import (
    AdmitInvocationCommand,
    CreateInvocationCommand,
    InvocationLifecycleCommandKind,
    RebindInvocationCommand,
    ReleaseInvocationCommand,
    _ExternalExecutionOrigin,
    _invocation_lifecycle_command_sha256,
    _InvocationLifecycleReceiptLedger,
)
from cayu.sessions.base import SessionRunFenced
from cayu.sessions.external_waits import external_wait_digest


def external_execution_origin(
    command: CreateInvocationCommand
    | AdmitInvocationCommand
    | RebindInvocationCommand
    | ReleaseInvocationCommand,
    ledger: _InvocationLifecycleReceiptLedger,
) -> _ExternalExecutionOrigin | None:
    if isinstance(command, (CreateInvocationCommand, AdmitInvocationCommand)):
        from cayu.runtime._external_wait_admission import current_external_admission
        from cayu.runtime._external_wait_creation import current_external_creation

        initial = isinstance(command, CreateInvocationCommand)
        scoped = current_external_creation() if initial else current_external_admission()
        if scoped is None:
            return None
        registration, execution = scoped
        profile = command.active_profile if initial else command.target_active_profile
        digest = _invocation_lifecycle_command_sha256(command)
        if (
            execution.intent.mode != ("run" if initial else "resume")
            or execution.intent.session_id != command.session_id
            or execution.session_instance_id != command.expected_session_instance_id
            or execution.interaction_id != profile.interaction_id
            or execution.intent.profile_sha256 != profile.profile.fingerprint
            or profile.run_epoch != (1 if initial else execution.intent.expected_run_epoch + 1)
            or (not initial and execution.intent.admission_sha256 != digest)
        ):
            raise SessionRunFenced("External cleanup origin differs from its native admission.")
        return _ExternalExecutionOrigin(
            registration_sha256=external_wait_digest(registration),
            execution_sha256=external_wait_digest(execution),
            admission_sha256=digest,
            admission_kind="create" if initial else "admit",
            admission_epoch=profile.run_epoch,
        )

    # The native owner has already checked command authority/CAS. Authenticate
    # the immediate predecessor before compaction; never infer lineage from
    # matching interaction labels or from a caller-supplied origin.
    expected = command.expected_active_profile
    predecessors = [
        item
        for item in ledger.receipts
        if item.active_profile == expected
        and item.session_instance_id == command.expected_session_instance_id
        and item.session_id == command.session_id
    ]
    origins = [item.external_execution_origin for item in predecessors]
    if not any(origin is not None for origin in origins):
        return None
    origin = next(origin for origin in origins if origin is not None)
    if any(item != origin for item in origins):
        raise SessionRunFenced("External cleanup predecessors conflict.")
    if isinstance(command, RebindInvocationCommand):
        target = command.target_active_profile
        released = any(
            item.kind is InvocationLifecycleCommandKind.RELEASE
            and item.result_session.run_epoch == command.expected_run_epoch
            for item in predecessors
        )
        if (
            target.interaction_id != expected.interaction_id
            or target.profile != expected.profile
            or target.run_epoch != expected.run_epoch + (2 if released else 1)
        ):
            raise SessionRunFenced("External cleanup rebind differs from its predecessor.")
    return origin

"""Reattach an existing external wait to a genuine native recovery writer.

The immutable ticket is not rewritten. This records receiving-owner succession,
not a second admission or a permit manufactured from a session lookup.
"""

from contextlib import contextmanager
from contextvars import ContextVar

from cayu.runtime._invocation_lifecycle import InvocationContext
from cayu.sessions._invocation_lifecycle import (
    AdmittedInvocationBinding,
    require_invocation_command_authority,
)
from cayu.sessions._session_continuation import (
    ContinuationConflict,
    ContinuationRecord,
    ContinuationRecoveryWriter,
    ContinuationTicket,
    require_ticket_identity,
)

_RECOVERY: ContextVar[tuple[ContinuationTicket, InvocationContext] | None] = ContextVar(
    "continuation_recovery_writer", default=None
)


def require_resolved_native_pause(checkpoint) -> None:
    """A retained pause belongs to its native resolver, not whole-turn replay."""
    from cayu.approvals.user_input import PENDING_USER_INPUT_CHECKPOINT_KEY
    from cayu.runtime._approval_support import PENDING_TOOL_APPROVAL_CHECKPOINT_KEY
    from cayu.sessions.external_waits import ExternalWaitUnavailable

    # Presence only refuses recovery; it grants no authority to interpret or
    # resolve either record. Keep partial resolution owned by its native API too.
    if checkpoint is not None and any(
        checkpoint.get(key) is not None
        for key in (PENDING_TOOL_APPROVAL_CHECKPOINT_KEY, PENDING_USER_INPUT_CHECKPOINT_KEY)
    ):
        raise ExternalWaitUnavailable(
            "Resolve the native approval or user-input pause before external wait recovery."
        )


def require_recovery_frontier(record: ContinuationRecord, session, checkpoint) -> None:
    """Compare the exact native wait inside the recovery claim transaction."""
    from cayu.sessions._session_continuation import continuation_operation_key
    from cayu.sessions._session_continuation_store import ROOT_KEY, ContinuationRoot, digest

    raw = None if checkpoint is None else checkpoint.get(ROOT_KEY)
    if raw is None:
        raise ContinuationConflict("External recovery lost the native continuation index.")
    root = ContinuationRoot.model_validate(raw)
    key = continuation_operation_key(record.ticket)
    entry = next((item for item in root.entries if item.ticket_key == key), None)
    if (
        root.namespace != record.namespace
        or session.id != record.ticket.session_id
        or session.instance_id != record.ticket.session_instance_id
        or entry is None
        or entry.state not in {"ARMING", "WAITING"}
        or entry.state != record.ticket.state
        or entry.record_sha256 != digest(record.model_dump(mode="json"))
        or entry.purpose != "external-event-v1"
        or entry.originating_writer_generation != record.ticket.writer_generation
    ):
        raise ContinuationConflict("External recovery's native wait frontier changed.")


def recovery_writer(
    ticket: ContinuationTicket, invocation: InvocationContext
) -> ContinuationRecoveryWriter:
    if (
        type(invocation) is not InvocationContext
        or type(invocation.binding) is not AdmittedInvocationBinding
    ):
        raise PermissionError("Continuation recovery requires an admitted runtime invocation.")
    invocation.require_runtime_authority()
    binding = invocation.binding
    if (
        ticket.purpose != "external-event-v1"
        or binding.session_id != ticket.session_id
        or binding.session_instance_id != ticket.session_instance_id
        or binding.interaction_id != ticket.interaction_id
        or binding.run_epoch <= ticket.writer_generation
        or invocation.recovery_claim_id is None
    ):
        raise PermissionError("Continuation recovery conflicts with its originating invocation.")
    return ContinuationRecoveryWriter(
        run_epoch=binding.run_epoch,
        recovery_claim_id=invocation.recovery_claim_id,
        profile_sha256=invocation.profile.fingerprint,
    )


@contextmanager
def recovery_writer_scope(ticket: ContinuationTicket, invocation: InvocationContext):
    recovery_writer(ticket, invocation)
    token = _RECOVERY.set((ticket, invocation))
    try:
        yield
    finally:
        _RECOVERY.reset(token)


def require_recovery_writer(
    ticket,
    session,
    checkpoint,
    record: ContinuationRecord,
    *,
    now,
    require_attached_writer: bool = False,
) -> ContinuationRecoveryWriter:
    from cayu.sessions.base import (
        _INCOMPLETE_RECOVERY_CLAIM_CHECKPOINT_KEY,
        _active_unexpired_incomplete_recovery_claim_id,
    )

    authority = _RECOVERY.get()
    if authority is None or authority[0] != ticket:
        raise PermissionError("Continuation recovery requires its registered runtime owner.")
    invocation = authority[1]
    writer = recovery_writer(ticket, invocation)
    require_ticket_identity(ticket, record.ticket)
    require_invocation_command_authority(
        session,
        checkpoint,
        session_id=ticket.session_id,
        session_instance_id=ticket.session_instance_id,
        run_epochs=frozenset({writer.run_epoch}),
        active_profile=invocation.active_profile,
    )
    claim = (
        None if checkpoint is None else checkpoint.get(_INCOMPLETE_RECOVERY_CLAIM_CHECKPOINT_KEY)
    )
    if (
        type(claim) is not dict
        or type(claim.get("version")) is not int
        or claim.get("version") != 1
        or claim.get("claim_id") != writer.recovery_claim_id
        or _active_unexpired_incomplete_recovery_claim_id(checkpoint, now=now)
        != writer.recovery_claim_id
    ):
        raise ContinuationConflict("Continuation recovery lost its native claim.")
    if (
        record.services
        or record.ticket.state not in {"ARMING", "WAITING"}
        or record.consumption is not None
        or record.retirement is not None
    ):
        raise ContinuationConflict("Only an unconsumed external wait can follow recovery.")
    previous = record.recovery_writer
    if require_attached_writer and record.ticket.state == "WAITING" and previous != writer:
        raise ContinuationConflict("Parked continuation belongs to another recovered writer.")
    if previous is not None and (
        previous.profile_sha256 != writer.profile_sha256
        or previous.run_epoch > writer.run_epoch
        or (previous.run_epoch == writer.run_epoch and previous != writer)
    ):
        raise ContinuationConflict(
            "Continuation recovery writer conflicts with retained succession."
        )
    return writer

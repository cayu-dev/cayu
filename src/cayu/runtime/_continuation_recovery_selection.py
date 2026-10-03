"""Restrict native recovery to an authenticated consumed-ticket invocation."""

from cayu.collaboration._preparation import prepare_contract
from cayu.sessions._execution_profile_checkpoint import (
    active_invocation_execution_profile_from_checkpoint,
)
from cayu.sessions._invocation_lifecycle import (
    InvocationLifecycleCommandKind,
    _invocation_lifecycle_receipt_ledger_from_checkpoint,
    _require_released_invocation_command_receipt,
    reconcile_invocation_admission_from_state,
    require_invocation_rebind_lineage,
)
from cayu.sessions._session_continuation_store import ROOT_KEY, ContinuationRoot
from cayu.vaults.redaction import SecretRedactor


def _require_consumption_admission(session, checkpoint, expected):
    if session.instance_id != expected.session_instance_id:
        raise ValueError("Continuation recovery target incarnation changed.")
    root = prepare_contract(
        ContinuationRoot,
        None if checkpoint is None else checkpoint.get(ROOT_KEY),
        redactor=SecretRedactor(),
    )
    entry = next((item for item in root.entries if item.ticket_key == expected.ticket_key), None)
    if (
        root.namespace.session_id != session.id
        or root.namespace.session_instance_id != session.instance_id
        or entry is None
        or entry.state != "CONSUMED"
        or entry.record_sha256 != expected.record_sha256
        or entry.admission_command_digest != expected.admission_command_digest
        or entry.admission_expected_run_epoch != expected.admission_expected_run_epoch
    ):
        raise ValueError("Continuation recovery conflicts with its retained consumption.")
    admission = reconcile_invocation_admission_from_state(
        session,
        checkpoint,
        session_id=session.id,
        session_instance_id=expected.session_instance_id,
        expected_run_epoch=expected.admission_expected_run_epoch,
        command_sha256=expected.admission_command_digest,
        profile_sha256=expected.profile_digest,
    )
    if admission is None:
        raise ValueError("Continuation lacks its exact native admission receipt.")
    return admission


def read_consumed_continuation_release(session, checkpoint, expected):
    """Historical quiescence only; never select a later invocation for recovery."""
    admission = _require_consumption_admission(session, checkpoint, expected)
    original = admission.active_profile
    ledger = _invocation_lifecycle_receipt_ledger_from_checkpoint(checkpoint)
    lineage = tuple(
        item
        for item in ledger.receipts
        if item.session_id == session.id
        and item.session_instance_id == session.instance_id
        and item.active_profile.interaction_id == original.interaction_id
        and item.active_profile.profile == original.profile
        and item.active_profile.run_epoch >= original.run_epoch
    )
    frontier = max(lineage, key=lambda item: item.active_profile.run_epoch).active_profile
    require_invocation_rebind_lineage(
        checkpoint,
        session_instance_id=session.instance_id,
        original=original,
        current=frontier,
    )
    release = next(
        (
            item
            for item in lineage
            if item.kind is InvocationLifecycleCommandKind.RELEASE
            and item.active_profile == frontier
        ),
        None,
    )
    if release is None:
        return None
    if (
        release.result_session.id != session.id
        or release.result_session.instance_id != session.instance_id
        or release.result_session.run_epoch != frontier.run_epoch + 1
        or session.run_epoch < release.result_session.run_epoch
    ):
        raise ValueError("Continuation release conflicts with its original invocation.")
    return release.record_sha256


def require_continuation_recovery_selection(session, checkpoint, expected):
    if expected is None:
        return
    admission = _require_consumption_admission(session, checkpoint, expected)
    active = active_invocation_execution_profile_from_checkpoint(checkpoint)
    # Native recovery rebases an existing interaction to a new fenced epoch.
    # Another admission, even with the same profile, is not this responsibility.
    if (
        admission is None
        or active is None
        or active.session_id != session.id
        or active.run_epoch not in (session.run_epoch, session.run_epoch - 1)
        or active.interaction_id != admission.active_profile.interaction_id
        or active.profile != admission.active_profile.profile
    ):
        raise ValueError("Continuation recovery no longer owns the admitted invocation.")
    require_invocation_rebind_lineage(
        checkpoint,
        session_instance_id=expected.session_instance_id,
        original=admission.active_profile,
        current=active,
    )
    if active.run_epoch != session.run_epoch:
        _require_released_invocation_command_receipt(
            session,
            checkpoint,
            session_id=session.id,
            session_instance_id=expected.session_instance_id,
            active_profile=active,
        )

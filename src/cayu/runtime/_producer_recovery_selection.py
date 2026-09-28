"""Safe recovery selection and exact native ownership validation."""

from cayu.collaboration._contracts import CollaborationContractError
from cayu.collaboration._preparation import prepare_contract
from cayu.collaboration.access import CollaborationAccessContext
from cayu.runtime._producer_output_store import ROOT_KEY, NativeProducerIndex
from cayu.sessions import SessionStatus
from cayu.sessions.recovery import (
    RECOVERY_PLAN_MAX_ITEMS,
    ContinuationRecoveryExpectation,
    ProducerRecoveryExpectation,
    RecoveryPlanBounds,
    RecoveryPlanRequest,
    RecoveryPlanSelection,
)
from cayu.vaults.redaction import SecretRedactor


def _fields(value, schema):
    if type(value) is not schema:
        raise CollaborationContractError("Recovery selection requires typed contracts.")
    fields = object.__getattribute__(value, "__dict__")
    if (
        type(fields) is not dict
        or any(type(key) is not str for key in fields)
        or fields.keys() != schema.model_fields.keys()
    ):
        raise CollaborationContractError("Recovery selection contains invalid fields.")
    return fields.copy()


def prepare_recovery_selection(request, *, redactor):
    """Snapshot primitive selection fields before any user copying/serialization.

    The ordinary recovery page supports 1,000 IDs, so it intentionally does not
    inherit the smaller collaboration authority envelope. Only the new authority
    contracts use that envelope. Both boundaries revalidate detached values.
    """
    fields = _fields(request, RecoveryPlanRequest)
    selection = _fields(fields["selection"], RecoveryPlanSelection)
    bounds = _fields(fields["bounds"], RecoveryPlanBounds)
    ids = selection["session_ids"]
    statuses = selection["statuses"]
    inactive = selection["inactive_for_seconds"]
    cursor = selection["cursor"]
    if (
        type(ids) not in (tuple, list)
        or len(ids) > RECOVERY_PLAN_MAX_ITEMS
        or any(type(item) is not str for item in ids)
        or type(statuses) not in (tuple, list, set, frozenset)
        or len(statuses) > len(SessionStatus)
        or any(type(item) not in (str, SessionStatus) for item in statuses)
        or (inactive is not None and type(inactive) is not int)
        or (cursor is not None and type(cursor) is not str)
        or any(type(value) is not int for value in bounds.values())
    ):
        raise CollaborationContractError("Recovery selection contains invalid values.")
    selection["session_ids"] = tuple(ids)
    selection["statuses"] = tuple(statuses)
    for name, schema in (
        ("producer", ProducerRecoveryExpectation),
        ("continuation", ContinuationRecoveryExpectation),
        ("participant_context", CollaborationAccessContext),
    ):
        if fields[name] is not None:
            fields[name] = prepare_contract(schema, fields[name], redactor=redactor)
    try:
        fields["selection"] = RecoveryPlanSelection.model_validate(selection)
        fields["bounds"] = RecoveryPlanBounds.model_validate(bounds)
        return RecoveryPlanRequest.model_validate(fields)
    except (TypeError, ValueError):
        pass
    # Never expose framework errors rendering valid siblings or rejected values.
    raise CollaborationContractError("Recovery selection is invalid.")


def require_recovery_selection(session, checkpoint, request, *, allow_settled=False):
    from cayu.runtime._continuation_recovery_selection import (
        require_continuation_recovery_selection,
    )

    require_producer_recovery_selection(
        session, checkpoint, request.producer, allow_settled=allow_settled
    )
    require_continuation_recovery_selection(session, checkpoint, request.continuation)


def require_producer_recovery_selection(session, checkpoint, expected, *, allow_settled=False):
    """An expectation restricts a claim; it never creates a producer grant.

    The native index's commitment binds the complete registered command,
    collaboration responsibility and registration receipt. A settled attachment
    cannot select a later ordinary invocation in the same session.
    """
    if expected is None:
        return
    if session.instance_id != expected.session_instance_id:
        raise ValueError("Producer recovery target incarnation changed.")
    index = prepare_contract(
        NativeProducerIndex,
        None if checkpoint is None else checkpoint.get(ROOT_KEY),
        redactor=SecretRedactor(),
    )
    if (
        (index.session_id, index.session_instance_id) != (session.id, session.instance_id)
        or index.operation_key != expected.attachment_operation_key
        or index.record_commitment != expected.attachment_commitment
        or index.state != "admitted"
        or index.invocation is None
        or (not allow_settled and index.cleanup_commitment is not None)
        or (not allow_settled and index.paused_stop is not None)
    ):
        raise ValueError("Producer recovery conflicts with its unsettled native attachment.")

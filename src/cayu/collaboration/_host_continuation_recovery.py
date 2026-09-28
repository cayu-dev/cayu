"""Compose exact consumed-ticket selection with native abandoned-work recovery."""

from hashlib import sha256

from cayu.collaboration._preparation import contract_bytes
from cayu.runtime._checkpoint_store import runtime_checkpoint_session_store
from cayu.runtime._continuation_recovery_selection import read_consumed_continuation_release
from cayu.runtime._session_continuation import ContinuationUnavailable
from cayu.runtime._session_continuation_store import digest
from cayu.sessions.recovery import (
    ContinuationRecoveryExpectation,
    RecoveryExecutionRequest,
    RecoveryItemExecutionStatus,
    RecoveryPlanAction,
    RecoveryPlanRequest,
    RecoveryPlanSelection,
)


def _selection(expected, record):
    consumption = record.consumption
    if consumption is None or consumption.receipt_stage != "admitted":
        raise ContinuationUnavailable("Continuation recovery requires exact admitted consumption.")
    return ContinuationRecoveryExpectation(
        session_instance_id=expected.session.session_instance_id,
        ticket_key=expected.ticket_key,
        record_sha256=digest(record.model_dump(mode="json")),
        admission_command_digest=consumption.admission_command_digest,
        admission_expected_run_epoch=consumption.admission_expected_run_epoch,
        profile_digest=consumption.profile_digest,
    )


async def read_continuation_release(app, expected, record, *, context):
    """Read exact quiescence only; never plan, claim, or resume an invocation."""
    from cayu.runtime._host_continuation_discovery import recover_session_continuation

    observed = await recover_session_continuation(app, expected, context=context)
    if observed != record:
        raise ContinuationUnavailable("Continuation changed during release inspection.")
    selection = _selection(expected, record)
    session = await app.session_store.load(expected.session.session_id)
    if session is None:
        raise ContinuationUnavailable("Continuation recovery target is unavailable.")
    checkpoint = await runtime_checkpoint_session_store(app.session_store).load_checkpoint(
        session.id
    )
    return read_consumed_continuation_release(session, checkpoint, selection)


async def recover_interrupted_continuation(app, expected, record, *, context, inactive_for_seconds):
    selection = _selection(expected, record)
    settled = await read_continuation_release(app, expected, record, context=context)
    if settled is not None:
        return settled
    plan = await app.plan_recovery(
        RecoveryPlanRequest(
            selection=RecoveryPlanSelection(
                session_ids=(expected.session.session_id,),
                inactive_for_seconds=inactive_for_seconds,
            ),
            continuation=selection,
            participant_context=context,
        )
    )
    if (
        len(plan.items) != 1
        or RecoveryPlanAction.AUTOMATIC_REPAIR not in plan.items[0].allowed_actions
    ):
        return None
    result = await app.execute_recovery(
        RecoveryExecutionRequest(
            plan=plan,
            execution_id="host-continuation-recovery:"
            + sha256(contract_bytes(selection, redactor=app._secret_redactor)).hexdigest(),
        )
    )
    if len(result.items) != 1:
        raise ContinuationUnavailable("Continuation native recovery receipt is incomplete.")
    status = result.items[0].status
    if status in (RecoveryItemExecutionStatus.BLOCKED, RecoveryItemExecutionStatus.LEFT_INTACT):
        return None
    if status is not RecoveryItemExecutionStatus.EXECUTED:
        raise ContinuationUnavailable("Continuation native recovery requires reconciliation.")
    return await read_continuation_release(app, expected, record, context=context)

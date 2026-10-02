"""Recover planner-owned execution input before producer attachment.

Planning remains the preparation owner. This adapter resolves its immutable
native admission stage and blueprint, rather than inventing input or rerunning
policy after the planner has already committed admission.
"""

from hashlib import sha256

from cayu.collaboration._contracts import ExactMatch
from cayu.collaboration._host_planning import lookup_host_plan
from cayu.collaboration._planning_records import RequestPlanningRecord, RequestPlanningStageRecord
from cayu.collaboration._planning_stages import read_stage
from cayu.collaboration._planning_store import read_plan_in_transaction
from cayu.collaboration._preparation import contract_bytes, prepare_contract, require_exact_contract
from cayu.collaboration._producer_contracts import ProducerOutputRegistration
from cayu.collaboration._producer_readback import lookup_producer_registration
from cayu.collaboration._producer_recovery import ProducerOutputRecovery
from cayu.collaboration._producer_registration import _native_execution_commitment
from cayu.collaboration._request_store import operation_key
from cayu.collaboration.access import CollaborationAccessContext
from cayu.collaboration.base import PLANNING_FAMILY
from cayu.collaboration.mandates import MandateAccessContext
from cayu.collaboration.participants import CollaborationUnavailable
from cayu.collaboration.planning import RequestPlanningFresh, RequestPlanningRequest
from cayu.collaboration.prepared_admission import FreshRecipientAdmissionTarget
from cayu.collaboration.recipient_preparation import preparation_run_request
from cayu.collaboration.requests import RequestAdmissionReceipt
from cayu.sessions.context_views import ParticipantSessionExecutionRequest


async def recover_interrupted_producer(app, command, *, context, inactive_for_seconds):
    """Use the existing exact native recovery claim, not another producer launch.

    This is an explicit application-selected inactivity policy. Native planning
    retains live/human/unknown-effect gates; store-time admission still owns the
    inactivity decision. A blocked plan is not proof that production settled.
    """
    from cayu.sessions._producer_checkpoint import attachment_index
    from cayu.sessions.recovery import (
        ProducerRecoveryExpectation,
        RecoveryExecutionRequest,
        RecoveryItemExecutionStatus,
        RecoveryPlanAction,
        RecoveryPlanRequest,
        RecoveryPlanSelection,
    )

    attachment = await app.session_store._read_native_producer_attachment(command)
    if attachment is None:
        raise CollaborationUnavailable("Producer recovery has no exact native attachment.")
    index = attachment_index(attachment)
    request = RecoveryPlanRequest(
        selection=RecoveryPlanSelection(
            session_ids=(index.session_id,), inactive_for_seconds=inactive_for_seconds
        ),
        producer=ProducerRecoveryExpectation(
            session_instance_id=index.session_instance_id,
            attachment_operation_key=index.operation_key,
            attachment_commitment=index.record_commitment,
        ),
        participant_context=context,
    )
    plan = await app.plan_recovery(request)
    if (
        len(plan.items) != 1
        or RecoveryPlanAction.AUTOMATIC_REPAIR not in plan.items[0].allowed_actions
    ):
        return False
    receipt = await app.execute_recovery(
        RecoveryExecutionRequest(
            plan=plan,
            execution_id="host-producer-recovery:"
            + sha256(contract_bytes(command, redactor=app._secret_redactor)).hexdigest(),
        )
    )
    if len(receipt.items) != 1:
        raise CollaborationUnavailable("Producer recovery returned an incomplete native receipt.")
    result = receipt.items[0]
    if result.status in (
        RecoveryItemExecutionStatus.BLOCKED,
        RecoveryItemExecutionStatus.LEFT_INTACT,
    ):
        return False
    if result.status is not RecoveryItemExecutionStatus.EXECUTED:
        raise CollaborationUnavailable("Producer native recovery requires exact reconciliation.")
    return True


async def read_admitted_producer_plan(app, admission, *, context, expected_plan=None):
    """Authenticate the complete retained plan without requiring a live session.

    This is provenance readback, not authority to execute. It also remains
    usable when an exactly settled producer's native session has been retired.
    """
    redactor = app._secret_redactor
    admission = prepare_contract(RequestAdmissionReceipt, admission, redactor=redactor)
    context = prepare_contract(MandateAccessContext, context, redactor=redactor)
    if expected_plan is not None:
        expected_plan = prepare_contract(RequestPlanningRequest, expected_plan, redactor=redactor)
    command = admission.command
    prepared = command.prepared
    if (
        admission.state != "admitted"
        or prepared is None
        or not isinstance(prepared.target, FreshRecipientAdmissionTarget)
        or prepared.target.resources
        or command.expected.intent.request.cancellation != "stop"
    ):
        raise CollaborationUnavailable(
            "Host production requires qualified resource-free FRESH admission."
        )
    found = await app._request_coordinator.lookup_admission(
        command, context=context, wait_for_settlement=True
    )
    if not isinstance(found, ExactMatch) or found.receipt != admission:
        raise CollaborationUnavailable("Host production lacks exact retained admission.")
    store, initialized = app._participant_coordinator._ready()
    app._participant_coordinator._capability(
        store, initialized, mutation=False, family=PLANNING_FAMILY
    )
    async with store._transaction(initialized.owner.application_scope, write=False) as tx:
        raw = await tx.find_request_plan_stage(command.operation)
        if raw is None:
            raise CollaborationUnavailable("Host production has no retained planning stage.")
        stage = prepare_contract(RequestPlanningStageRecord, raw, redactor=redactor)
        require_exact_contract(stage.intent.command, command, redactor=redactor)
        require_exact_contract(
            stage, await read_stage(tx, stage.intent, redactor=redactor), redactor=redactor
        )
        if stage.state != "settled" or stage.receipt != admission:
            raise CollaborationUnavailable("Host production planning stage is not admitted.")
        raw_plan = await tx.get("request_plans", operation_key(stage.intent.plan))
        if raw_plan is None:
            raise CollaborationUnavailable("Host production planning owner is unavailable.")
        plan = prepare_contract(RequestPlanningRecord, raw_plan, redactor=redactor)
        if expected_plan is not None and plan.receipt.command != expected_plan:
            raise CollaborationUnavailable("Producer admission belongs to another exact plan.")
        found = await read_plan_in_transaction(
            store, tx, initialized, plan.receipt.command, redactor=redactor
        )
        if (
            not isinstance(found, ExactMatch)
            or found.receipt != plan
            or plan.state != "admitted"
            or plan.receipt.command.operation != stage.intent.plan
            or sha256(contract_bytes(plan.receipt.command, redactor=redactor)).hexdigest()
            != stage.intent.plan_sha256
        ):
            raise CollaborationUnavailable("Host production conflicts with its retained plan.")
    # The production reader repeats current per-plan authority outside the store
    # transaction. The frozen decision is used only after exact retained readback.
    found = await lookup_host_plan(app, plan.receipt.command, context=context)
    if not isinstance(found, ExactMatch) or found.receipt != plan:
        raise CollaborationUnavailable("Host production plan readback changed.")
    decision = plan.decision
    if not isinstance(decision, RequestPlanningFresh) or decision.resources:
        raise CollaborationUnavailable("Host production plan selects an unsupported target.")
    return plan


async def recover_planned_execution(app, admission, *, context, expected_plan=None):
    redactor = app._secret_redactor
    admission = prepare_contract(RequestAdmissionReceipt, admission, redactor=redactor)
    plan = await read_admitted_producer_plan(
        app, admission, context=context, expected_plan=expected_plan
    )
    command = admission.command
    prepared = command.prepared
    assert prepared is not None and isinstance(prepared.target, FreshRecipientAdmissionTarget)
    decision = plan.decision
    assert isinstance(decision, RequestPlanningFresh)
    request = preparation_run_request(decision.preparation.request_json)
    execution = ParticipantSessionExecutionRequest(
        request=request.model_copy(update={"session_id": prepared.target.session_id}),
        session_instance_id=prepared.target.session_instance_id,
        execution_key="host-producer:"
        + sha256(contract_bytes(command, redactor=redactor)).hexdigest(),
    )
    # Native creation owns the actual profile and first-input commitment. This
    # is a read-only consistency check, not the current launch guard, which the
    # existing producer registration and invocation entrances still acquire.
    await _native_execution_commitment(app, prepared, execution)
    return execution


async def recover_registered_execution(
    app, expected, *, context, producer_context, expected_plan=None
):
    """Reconstruct the original execution tuple, never a replacement launch.

    Discovery tokens are not authority. Both registration and frozen input are
    read through their current authorized owners; native creation authenticates
    the reconstructed input. Actual execution still requires the producer's
    native handoff and current execution grant. Completed production must instead
    be serviced through its retained completion/output owners.
    """
    redactor = app._secret_redactor
    if type(expected) not in (ProducerOutputRecovery, ProducerOutputRegistration):
        raise TypeError("Host producer recovery requires an exact registration expectation.")
    expected = prepare_contract(type(expected), expected, redactor=redactor)
    context = prepare_contract(CollaborationAccessContext, context, redactor=redactor)
    producer_context = prepare_contract(MandateAccessContext, producer_context, redactor=redactor)
    found = await lookup_producer_registration(
        app, expected, context=context, wait_for_settlement=True
    )
    if not isinstance(found, ExactMatch):
        raise CollaborationUnavailable("Host producer registration is unavailable or conflicting.")
    command = prepare_contract(ProducerOutputRegistration, found.receipt, redactor=redactor)
    admitted = await app._request_coordinator.lookup_admission(
        command.admission, context=producer_context, wait_for_settlement=True
    )
    if not isinstance(admitted, ExactMatch):
        raise CollaborationUnavailable("Host producer admission is unavailable or conflicting.")
    execution = await recover_planned_execution(
        app, admitted.receipt, context=producer_context, expected_plan=expected_plan
    )
    execution = ParticipantSessionExecutionRequest(
        request=execution.request,
        session_instance_id=execution.session_instance_id,
        execution_key=command.execution_key,
    )
    prepared = command.admission.prepared
    assert prepared is not None
    if await _native_execution_commitment(app, prepared, execution) != command.execution_commitment:
        raise CollaborationUnavailable("Host producer execution conflicts with its registration.")
    # Re-read exact authority after the cross-owner reconstruction awaits. No
    # current-state absence is interpreted as permission to create a new producer.
    final = await lookup_producer_registration(
        app, command, context=context, wait_for_settlement=True
    )
    if not isinstance(final, ExactMatch) or final.receipt != command:
        raise CollaborationUnavailable("Host producer registration changed during recovery.")
    return command, execution

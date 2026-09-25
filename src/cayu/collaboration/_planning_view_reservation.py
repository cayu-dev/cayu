"""Order native cleanup reservation before stage-owned view permit admission."""

from dataclasses import dataclass
from hashlib import sha256

from cayu.collaboration._contracts import CollaborationConflict
from cayu.collaboration._planning_fork_types import RequestViewStageCommand
from cayu.collaboration._planning_records import RequestPlanningStageRecord
from cayu.collaboration._preparation import prepare_contract, require_exact_contract
from cayu.collaboration._request_store import operation_key
from cayu.collaboration.participants import CollaborationUnavailable
from cayu.sessions._context_selection_fence import ContextViewSelectionDecision


@dataclass(frozen=True, slots=True)
class _ViewReservationReadback:
    """Minted by the native owner after its durable reservation acknowledges."""

    command: RequestViewStageCommand
    decision: ContextViewSelectionDecision


async def view_permit_stage(tx, permit, *, redactor):
    registration = permit.intent.request
    if registration.effect_scope not in {"context_view_selection", "context_view_retention"}:
        return None
    operation = permit.operation
    if registration.effect_scope == "context_view_retention":
        operation = operation.model_copy(
            update={
                "caller_key": "plan-view:"
                + sha256(registration.target.object_id.encode()).hexdigest()
            }
        )
    raw = await tx.find_request_plan_stage(operation)
    if raw is None:
        return None
    stage = prepare_contract(RequestPlanningStageRecord, raw, redactor=redactor)
    if not isinstance(stage.intent.command, RequestViewStageCommand) or (
        permit not in stage.intent.command.preparation.view.permits
    ):
        raise CollaborationConflict("View permit differs from its exact planning stage.")
    return stage


async def view_permits_registered(tx, command, *, redactor):
    """The pair is admitted atomically; partial or contradictory evidence refuses."""
    from cayu.collaboration._permit_store import registered_receipt

    issued = []
    for permit in command.preparation.view.permits:
        registered = await registered_receipt(tx, permit, redactor) is not None
        if not registered and (
            await tx.get("permits", operation_key(permit.operation)) is not None
            or await tx.get("operations", operation_key(permit.intent.request.settlement_operation))
            is not None
        ):
            raise CollaborationUnavailable("View permit has contradictory unregistered evidence.")
        issued.append(registered)
    if any(issued) and not all(issued):
        raise CollaborationUnavailable("View permit registration is incomplete.")
    return all(issued)


async def require_view_reservation(tx, permit, proof, *, redactor):
    """Called in the permit writer transaction, including direct native callers."""
    stage = await view_permit_stage(tx, permit, redactor=redactor)
    if stage is None:
        return
    if stage.state != "pending" or type(proof) is not _ViewReservationReadback:
        raise CollaborationConflict("View permit requires its pending reserved planning stage.")
    require_exact_contract(stage.intent.command, proof.command, redactor=redactor)
    decision = ContextViewSelectionDecision.model_validate(proof.decision)
    if (
        decision.state != "reserved"
        or decision.responsibility_registered
        or decision.target != proof.command.preparation.view
    ):
        raise CollaborationConflict("View permit lacks exact native cleanup reservation.")


async def register_reserved_view(store, tx, initialized, prior, proof, *, redactor):
    """Register both permits, or neither, against cancellation and current input."""
    from cayu.collaboration._permit_store import register_permit_in_transaction
    from cayu.collaboration._planning_fork_types import view_stage_command
    from cayu.collaboration._planning_stages import read_stage
    from cayu.collaboration._planning_store import _require_current_input

    if type(proof) is not _ViewReservationReadback:
        raise CollaborationConflict("View admission requires native reservation readback.")
    require_exact_contract(view_stage_command(prior), proof.command, redactor=redactor)
    decision = ContextViewSelectionDecision.model_validate(proof.decision)
    if decision.target != proof.command.preparation.view or decision.state not in {
        "reserved",
        "selected",
    }:
        raise CollaborationConflict("View reservation differs from its exact native target.")
    stage = await view_permit_stage(tx, proof.command.preparation.view.permit, redactor=redactor)
    if stage is None or stage.intent.plan != prior.receipt.command.operation:
        raise CollaborationConflict("View preparation has no exact retained planning stage.")
    require_exact_contract(stage.intent.command, proof.command, redactor=redactor)
    await read_stage(tx, stage.intent, redactor=redactor)
    if stage.state != "pending" or prior.state != "preparing":
        raise CollaborationConflict("View preparation was already discharged.")
    if not await view_permits_registered(tx, proof.command, redactor=redactor):
        await _require_current_input(store, tx, initialized, prior.receipt.command, redactor)
        for permit in proof.command.preparation.view.permits:
            await register_permit_in_transaction(
                store, tx, initialized, permit, redactor, _view_reservation=proof
            )
    return prior

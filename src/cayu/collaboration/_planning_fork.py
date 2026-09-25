"""Retained FORK view lifetime and material-bound creation under native owners."""

from cayu.collaboration._contracts import CollaborationConflict, ExactMatch
from cayu.collaboration._planning_creation_types import RequestCreationStageCommand
from cayu.collaboration._planning_fork_types import (
    _ForkPreparationReadback,
    _ViewReadback,
    view_stage_command,
)
from cayu.collaboration._planning_records import RequestPlanningStageRecord
from cayu.collaboration._planning_stages import (
    creation_stage_intent,
    finish_stage,
    retain_preparation_stage,
    retain_stage,
    view_stage_intent,
)
from cayu.collaboration._planning_store import (
    _require_current_input,
    read_plan_in_transaction,
)
from cayu.collaboration._preparation import prepare_contract, require_exact_contract
from cayu.collaboration.participants import CollaborationUnavailable
from cayu.collaboration.planning import RequestPlanningFork


async def retain_view_stage(store, tx, initialized, prior, *, redactor):
    if prior.state != "decided" or not isinstance(prior.decision, RequestPlanningFork):
        raise CollaborationConflict("Planning has no eligible FORK proposal.")
    return await retain_preparation_stage(
        store,
        tx,
        initialized,
        prior,
        view_stage_intent(prior, redactor),
        content=view_stage_command(prior),
        redactor=redactor,
    )


async def adopt_fork_preparation(store, tx, initialized, prior, readback, *, redactor):
    if type(readback) is not _ForkPreparationReadback:
        raise CollaborationConflict("FORK creation requires native resolved preparation.")
    require_exact_contract(view_stage_command(prior), readback.command, redactor=redactor)
    from cayu.collaboration.recipient_preparation import fork_blueprint_commitment

    require_exact_contract(
        prior.decision.preparation.base, readback.preparation.base, redactor=redactor
    )
    if readback.preparation.blueprint_commitment != fork_blueprint_commitment(
        prior.decision.preparation
    ):
        raise CollaborationConflict("FORK material belongs to another retained blueprint.")
    await _require_current_input(store, tx, initialized, prior.receipt.command, redactor)
    if prior.decision.resources:
        from cayu.collaboration._planning_stages import resource_stage_intent

        intent = resource_stage_intent(prior, 0, redactor)
    else:
        intent = creation_stage_intent(prior, redactor, preparation=readback.preparation)
    await retain_stage(
        store,
        tx,
        initialized,
        prior.receipt.command,
        intent,
        redactor=redactor,
        _fork_preparation=readback,
    )
    return (
        await read_plan_in_transaction(
            store, tx, initialized, prior.receipt.command, redactor=redactor
        )
    ).receipt


async def adopt_view_settlement(store, tx, initialized, prior, readback, *, redactor):
    if type(readback) is not _ViewReadback:
        raise CollaborationConflict("View settlement requires its native receiving owner.")
    require_exact_contract(view_stage_command(prior), readback.receipt.command, redactor=redactor)
    await finish_stage(
        store,
        tx,
        initialized,
        prior.receipt.command,
        view_stage_intent(prior, redactor),
        readback.receipt,
        redactor=redactor,
        _receiving=readback,
    )
    return (
        await read_plan_in_transaction(
            store, tx, initialized, prior.receipt.command, redactor=redactor
        )
    ).receipt


async def fork_stages(requests, record):
    store, initialized = requests._participants._ready()
    async with store._transaction(initialized.binding.application_scope, write=False) as tx:
        found = await read_plan_in_transaction(
            store, tx, initialized, record.receipt.command, redactor=requests._redactor
        )
        if not isinstance(found, ExactMatch):
            raise CollaborationUnavailable("Retained FORK planning is unavailable.")
        rows = await tx.scan_request_plan_stages(record.receipt.command.operation, limit=4)
        return found.receipt, tuple(
            prepare_contract(RequestPlanningStageRecord, row, redactor=requests._redactor)
            for row in rows
        )


async def prepare_view_stage(requests, value, record, owner, *, context):
    from cayu.collaboration._planning_coordinator import _held

    reserved = await owner.reserve(view_stage_command(record), context=context)
    # Reacquire the current request mandate and input/deadline guard after the
    # foreign reservation, before atomically issuing either participant permit.
    await _held(requests, value, read_only=False, view_reservation=reserved)
    return await owner.prepare(view_stage_command(record), context=context)


async def progress_fork(requests, value, record, owner, creation_owner, *, max_items=None):
    from cayu.collaboration._planning_coordinator import _held
    from cayu.collaboration._planning_creation_evidence import _CreationReadback
    from cayu.collaboration._planning_creation_types import prepared_creation_from_stage
    from cayu.collaboration._planning_recipient import admit_prepared_plan
    from cayu.collaboration._planning_stages import admission_stage_intent
    from cayu.collaboration.access import CollaborationAccessContext

    if owner is None or creation_owner is None:
        raise CollaborationUnavailable("Native FORK receivers are unavailable.")
    continuation = value.model_copy(update={"control": None})
    context = CollaborationAccessContext(principal=value.context.principal)
    remaining = max_items if max_items is not None else 4
    record, stages = await fork_stages(requests, record)
    cleanup = record.state in {"cancelled", "expired"}
    if not stages:
        return record
    if not cleanup and len(stages) == 1:
        if stages[0].state != "pending":
            raise CollaborationConflict("View preparation was already discharged.")
        resolved = await prepare_view_stage(requests, continuation, record, owner, context=context)
        record = await _held(requests, continuation, read_only=False, fork_preparation=resolved)
        remaining -= 1
        if remaining == 0:
            return record
        record, stages = await fork_stages(requests, record)
        cleanup = record.state in {"cancelled", "expired"}
    if len(stages) >= 2 and stages[1].state == "pending":
        command = stages[1].intent.command
        if not isinstance(command, RequestCreationStageCommand):
            raise CollaborationUnavailable("FORK creation stage has another command.")
        received = await creation_owner.read(command)
        if not isinstance(received, _CreationReadback):
            record = await _held(requests, continuation, read_only=False, require_retained=True)
            if record.state not in {"preparing", "cancelled", "expired"}:
                return record
            cleanup = record.state in {"cancelled", "expired"}
            received = await (
                creation_owner.exclude(command, context=context)
                if cleanup
                else creation_owner.create(command, context=context)
            )
        record = await _held(requests, continuation, read_only=False, creation_readback=received)
        remaining -= 1
        if remaining == 0:
            return record
        record, stages = await fork_stages(requests, record)
    if stages[0].state == "pending":
        received = await owner.discharge(view_stage_command(record), context=context)
        record = await _held(requests, continuation, read_only=False, view_readback=received)
        remaining -= 1
        if remaining == 0 or record.state != "preparing":
            return record
    record = await _held(requests, continuation, read_only=False, require_retained=True)
    if record.state != "preparing":
        return record
    store, initialized = requests._participants._ready()
    async with store._transaction(initialized.binding.application_scope, write=False) as tx:
        prepared = await prepared_creation_from_stage(tx, record, redactor=requests._redactor)
    if prepared is None:
        raise CollaborationConflict("FORK creation did not produce an admissible child.")
    return await admit_prepared_plan(
        requests,
        record,
        context=value.context,
        intent=admission_stage_intent(record, requests._redactor, prepared=prepared),
    )

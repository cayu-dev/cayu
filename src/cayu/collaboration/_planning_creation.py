"""Retained FRESH creation stage under the existing local planning owner."""

from cayu.collaboration._contracts import CollaborationConflict, ExactMatch
from cayu.collaboration._planning_creation_types import creation_stage_command
from cayu.collaboration._planning_stages import creation_stage_intent, retain_preparation_stage
from cayu.collaboration._planning_store import read_plan_in_transaction
from cayu.collaboration.planning import RequestPlanningFresh


async def retain_creation_stage(store, tx, initialized, prior, *, redactor):
    if prior.state != "decided" or not isinstance(prior.decision, RequestPlanningFresh):
        raise CollaborationConflict("Planning has no eligible frozen creation proposal.")
    return await retain_preparation_stage(
        store,
        tx,
        initialized,
        prior,
        creation_stage_intent(prior, redactor),
        content=creation_stage_command(prior),
        redactor=redactor,
    )


async def adopt_creation_stage(store, tx, initialized, prior, readback, *, redactor):
    from cayu.collaboration._planning_creation_evidence import _CreationReadback
    from cayu.collaboration._planning_stages import finish_stage
    from cayu.collaboration._preparation import require_exact_contract

    if type(readback) is not _CreationReadback:
        raise CollaborationConflict("Creation settlement requires its native receiving owner.")
    require_exact_contract(
        creation_stage_command(prior, preparation=readback.receipt.command.preparation),
        readback.receipt.command,
        redactor=redactor,
    )
    await finish_stage(
        store,
        tx,
        initialized,
        prior.receipt.command,
        creation_stage_intent(prior, redactor, preparation=readback.receipt.command.preparation),
        readback.receipt,
        redactor=redactor,
        _receiving=readback,
    )
    found = await read_plan_in_transaction(
        store, tx, initialized, prior.receipt.command, redactor=redactor
    )
    assert isinstance(found, ExactMatch)
    return found.receipt


async def progress_creation(requests, value, record, owner, *, max_items=None):
    from cayu.collaboration._planning_coordinator import _held
    from cayu.collaboration._planning_creation_evidence import _CreationReadback
    from cayu.collaboration._planning_recipient import admit_prepared_plan
    from cayu.collaboration._planning_stages import admission_stage_intent
    from cayu.collaboration.access import CollaborationAccessContext
    from cayu.collaboration.participants import CollaborationUnavailable

    if owner is None:
        raise CollaborationUnavailable("Native planning creation owner is unavailable.")
    continuation = value.model_copy(update={"control": None})
    if record.stage_count == 1 and record.pending_stages:
        command = creation_stage_command(record)
        readback = await owner.read(command)
        if not isinstance(readback, _CreationReadback):
            # Readback may itself outlive the deadline. Resolve current owner
            # time/control before starting any new receiving work.
            record = await _held(requests, continuation, read_only=False, require_retained=True)
            if record.state not in {"preparing", "cancelled", "expired"}:
                return record
            context = CollaborationAccessContext(principal=value.context.principal)
            readback = await (
                owner.exclude(command, context=context)
                if record.state in {"cancelled", "expired"}
                else owner.create(command, context=context)
            )
        if type(readback) is not _CreationReadback:
            raise CollaborationUnavailable("Native creation responsibility remains unresolved.")
        record = await _held(requests, continuation, read_only=False, creation_readback=readback)
        if record.state != "preparing" or max_items == 1:
            return record
        if readback.receipt.prepared is None:
            raise CollaborationConflict(
                "Native creation was excluded; explicit plan cleanup is required."
            )
    # This is a separate transaction: a stale final election cannot undo the
    # durable adoption of an already-created native child.
    record = await _held(requests, continuation, read_only=False, require_retained=True)
    if record.state != "preparing":
        return record
    from cayu.collaboration._planning_creation_types import prepared_creation_from_stage

    # A settled creation stage is the durable reconstruction owner. Do not redo
    # foreign receiving work on every final-admission retry. Final admission still
    # authenticates the native target through its existing registered receiver.
    store, initialized = requests._participants._ready()
    async with store._transaction(initialized.binding.application_scope, write=False) as tx:
        found = await read_plan_in_transaction(
            store, tx, initialized, value.request, redactor=requests._redactor
        )
        if not isinstance(found, ExactMatch):
            raise CollaborationUnavailable("Prepared planning stage is unavailable.")
        record = found.receipt
        prepared = await prepared_creation_from_stage(tx, record, redactor=requests._redactor)
    if record.state != "preparing":
        return record
    if prepared is None:
        raise CollaborationConflict("Creation did not produce an admissible recipient.")
    return await admit_prepared_plan(
        requests,
        record,
        context=value.context,
        intent=admission_stage_intent(record, requests._redactor, prepared=prepared),
    )

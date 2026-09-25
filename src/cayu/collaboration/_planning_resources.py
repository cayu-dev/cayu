"""Resource planning transitions under the existing local transaction owner."""

from cayu.collaboration._contracts import CollaborationConflict, ExactMatch
from cayu.collaboration._planning_records import RequestPlanningStageRecord
from cayu.collaboration._planning_resource_types import (
    RequestResourceAcquisitionStageCommand,
    RequestResourceTransferStageCommand,
    ResourceStageCommand,
    _ResourceAcquisitionReadback,
    _ResourceAdoptionReadback,
    _ResourceCreationReadback,
    _ResourceReleaseReadback,
    _ResourceTransferReadback,
)
from cayu.collaboration._preparation import prepare_contract, require_exact_contract
from cayu.collaboration.participants import CollaborationUnavailable
from cayu.collaboration.planning import RequestPlanningFork, RequestPlanningFresh


async def resource_stages(requests, record):
    from cayu.collaboration._planning_store import read_plan_in_transaction

    store, initialized = requests._participants._ready()
    async with store._transaction(initialized.binding.application_scope, write=False) as tx:
        found = await read_plan_in_transaction(
            store, tx, initialized, record.receipt.command, redactor=requests._redactor
        )
        if not isinstance(found, ExactMatch):
            raise CollaborationUnavailable("Retained resource planning is unavailable.")
        rows = await tx.scan_request_plan_stages(
            record.receipt.command.operation,
            limit=record.receipt.command.limits.max_stages + 1,
        )
        return found.receipt, tuple(
            prepare_contract(RequestPlanningStageRecord, row, redactor=requests._redactor)
            for row in rows
        )


async def retain_resource_preparation(store, tx, initialized, prior, *, redactor):
    from cayu.collaboration._planning_stages import resource_stage_intent, retain_preparation_stage

    if prior.state != "decided" or not isinstance(prior.decision, RequestPlanningFresh):
        raise CollaborationConflict("Resource planning has no eligible frozen proposal.")
    return await retain_preparation_stage(
        store,
        tx,
        initialized,
        prior,
        resource_stage_intent(prior, 0, redactor),
        content=prior.decision,
        redactor=redactor,
    )


async def adopt_resource_progress(store, tx, initialized, prior, readback, *, redactor):
    from cayu.collaboration._planning_creation_types import creation_stage_command
    from cayu.collaboration._planning_stages import (
        creation_stage_intent,
        finish_stage,
        resource_stage_intent,
        retain_stage,
    )
    from cayu.collaboration._planning_store import _require_current_input, read_plan_in_transaction

    expected = prior.receipt.command
    if not isinstance(prior.decision, (RequestPlanningFresh, RequestPlanningFork)) or not (
        prior.decision.resources
    ):
        raise CollaborationConflict("Plan has no retained resource decision.")
    if type(readback) in (_ResourceReleaseReadback, _ResourceAdoptionReadback):
        raw = await tx.find_request_plan_stage(readback.receipt.command.operation)
        stage = prepare_contract(RequestPlanningStageRecord, raw, redactor=redactor)
        await finish_stage(
            store,
            tx,
            initialized,
            expected,
            stage.intent,
            readback.receipt,
            redactor=redactor,
            _receiving=readback,
        )
    else:
        if prior.state != "preparing":
            raise CollaborationConflict("Resource plan no longer admits preparation.")
        await _require_current_input(store, tx, initialized, expected, redactor)
        offset = 1 if isinstance(prior.decision, RequestPlanningFork) else 0
        if type(readback) is _ResourceCreationReadback:
            require_exact_contract(expected, readback.expected, redactor=redactor)
            require_exact_contract(
                creation_stage_command(prior, preparation=readback.command.preparation),
                readback.command,
                redactor=redactor,
            )
            await retain_stage(
                store,
                tx,
                initialized,
                expected,
                creation_stage_intent(prior, redactor, preparation=readback.command.preparation),
                redactor=redactor,
                _resource_preparation=readback,
            )
        elif type(readback) is _ResourceAcquisitionReadback:
            index, phase = divmod(prior.stage_count - offset - 1, 2)
            if phase or index < 0:
                raise CollaborationConflict("Resource acquisition is not the retained frontier.")
            await retain_stage(
                store,
                tx,
                initialized,
                expected,
                resource_stage_intent(prior, index, redactor, acquisition=readback.receipt),
                redactor=redactor,
                _resource_preparation=readback,
            )
        elif type(readback) is _ResourceTransferReadback:
            index, phase = divmod(prior.stage_count - offset - 1, 2)
            if not phase or index < 0 or index + 1 >= len(prior.decision.resources):
                raise CollaborationConflict("Resource transfer has no next preparation stage.")
            await retain_stage(
                store,
                tx,
                initialized,
                expected,
                resource_stage_intent(prior, index + 1, redactor),
                redactor=redactor,
                _resource_preparation=readback,
            )
        else:
            raise CollaborationConflict("Resource progress lacks a native receiving owner.")
    found = await read_plan_in_transaction(store, tx, initialized, expected, redactor=redactor)
    assert isinstance(found, ExactMatch)
    return found.receipt


async def require_resource_stage_predecessor(
    tx, prior, intent, readback, fork_readback, *, redactor
):
    """Authenticate the whole preceding handoff before growing native debt."""
    from cayu.collaboration._planning_fork_types import _ForkPreparationReadback, view_stage_command
    from cayu.collaboration._planning_stages import read_stage, resource_stage_intent

    if not isinstance(prior.decision, (RequestPlanningFresh, RequestPlanningFork)) or not (
        prior.decision.resources
    ):
        raise CollaborationConflict("Resource stage has no frozen preparation recipe.")
    offset = 1 if isinstance(prior.decision, RequestPlanningFork) else 0
    index, transfer = divmod(intent.ordinal - offset - 1, 2)
    if not 0 <= index < len(prior.decision.resources):
        raise CollaborationConflict("Resource stage is outside the exact preparation schedule.")
    acquisition = None
    if transfer:
        if type(readback) is not _ResourceAcquisitionReadback:
            raise CollaborationConflict("Resource transfer requires native acquisition readback.")
        acquisition = readback.receipt
        require_exact_contract(
            prior.decision.resources[index], readback.expected, redactor=redactor
        )
        preceding_intent = resource_stage_intent(prior, index, redactor)
    elif index:
        if type(readback) is not _ResourceTransferReadback:
            raise CollaborationConflict("Next resource requires native transfer acceptance.")
        previous = prior.decision.resources[index - 1]
        require_exact_contract(previous, readback.expected, redactor=redactor)
        require_exact_contract(
            previous.transfer_permit, readback.preparation.permit, redactor=redactor
        )
        preceding_intent = resource_stage_intent(
            prior, index - 1, redactor, acquisition=readback.receipt.command.intent.receipt
        )
        require_exact_contract(
            preceding_intent.command.native_command, readback.receipt.command, redactor=redactor
        )
        if readback.receipt.stage != "accepted":
            raise CollaborationConflict("Previous resource transfer was not accepted.")
    else:
        preceding_intent = None
        if offset:
            if type(fork_readback) is not _ForkPreparationReadback:
                raise CollaborationConflict("FORK resources require native view preparation.")
            require_exact_contract(
                view_stage_command(prior), fork_readback.command, redactor=redactor
            )
    require_exact_contract(
        resource_stage_intent(prior, index, redactor, acquisition=acquisition),
        intent,
        redactor=redactor,
    )
    if preceding_intent is not None:
        preceding = await read_stage(tx, preceding_intent, redactor=redactor)
        if preceding is None or preceding.state != "pending":
            raise CollaborationConflict("Resource predecessor is no longer owned by this plan.")


async def progress_resources(
    requests, value, record, owner, creation_owner, fork_owner, *, max_items=None
):
    """Bounded durable recovery; no host-local retry hook owns the next step."""
    from cayu.collaboration._planning_coordinator import _held
    from cayu.collaboration._planning_creation_evidence import _CreationReadback
    from cayu.collaboration._planning_creation_types import (
        RequestCreationStageCommand,
        RequestCreationStageReceipt,
    )
    from cayu.collaboration._planning_fork_types import view_stage_command
    from cayu.collaboration._planning_recipient import admit_prepared_plan
    from cayu.collaboration._planning_stages import admission_stage_intent
    from cayu.collaboration.access import CollaborationAccessContext
    from cayu.collaboration.planning import preparation_stage_count

    if owner is None or creation_owner is None:
        raise CollaborationUnavailable("Native resource preparation owners are unavailable.")
    continuation = value.model_copy(update={"control": None})
    context = CollaborationAccessContext(principal=value.context.principal)
    remaining = max_items if max_items is not None else value.request.limits.max_recovery_items
    for _ in range(remaining):
        # Each bounded recovery item can follow a slow foreign operation.
        # Refresh expiry through the same owner before acquiring more material.
        record = await _held(requests, continuation, read_only=False, require_retained=True)
        record, stages = await resource_stages(requests, record)
        if not isinstance(record.decision, (RequestPlanningFresh, RequestPlanningFork)):
            raise CollaborationUnavailable("Planning resource decision is unavailable.")
        offset = 1 if isinstance(record.decision, RequestPlanningFork) else 0
        final_ordinal = preparation_stage_count(record.decision)
        creation = stages[final_ordinal - 2] if len(stages) >= final_ordinal - 1 else None
        cleanup = record.state in {"cancelled", "expired"}
        if creation is not None and creation.state == "pending":
            command = creation.intent.command
            if not isinstance(command, RequestCreationStageCommand):
                raise CollaborationUnavailable("Creation frontier has another native command.")
            received = await creation_owner.read(command)
            if type(received) is not _CreationReadback:
                record = await _held(requests, continuation, read_only=False, require_retained=True)
                if record.state not in {"preparing", "cancelled", "expired"}:
                    return record
                cleanup = record.state in {"cancelled", "expired"}
                received = await (
                    creation_owner.exclude(command, context=context)
                    if cleanup
                    else creation_owner.create(command, context=context)
                )
            record = await _held(
                requests, continuation, read_only=False, creation_readback=received
            )
            continue
        if cleanup or creation is not None:
            created_receipt = creation.receipt if creation is not None else None
            if created_receipt is not None and not isinstance(
                created_receipt, RequestCreationStageReceipt
            ):
                raise CollaborationUnavailable("Creation stage has another receiving receipt.")
            created = created_receipt is not None and created_receipt.decision.state == "created"
            # Resolve creation/exclusion before touching destination pins.
            # Adopted destination responsibility stays with the native owner.
            pending = next(
                (
                    stage
                    for stage in reversed(stages)
                    if stage.state == "pending"
                    and isinstance(stage.intent.command, ResourceStageCommand)
                ),
                None,
            )
            if pending is not None:
                command = pending.intent.command
                if created and isinstance(command, RequestResourceTransferStageCommand):
                    assert creation is not None
                    received = await owner.read_adoption(
                        command, creation.intent.command, context=context
                    )
                else:
                    received = await owner.discharge_stage(command, context=context)
                record = await _held(
                    requests, continuation, read_only=False, resource_readback=received
                )
                continue
            if offset and stages and stages[0].state == "pending":
                if fork_owner is None:
                    raise CollaborationUnavailable("Native view cleanup owner is unavailable.")
                received = await fork_owner.discharge(view_stage_command(record), context=context)
                record = await _held(
                    requests, continuation, read_only=False, view_readback=received
                )
                continue
            if cleanup or record.state != "preparing":
                return record
            if not created or created_receipt is None:
                raise CollaborationConflict("Native creation was excluded; seal the retained plan.")
            record = await _held(requests, continuation, read_only=False, require_retained=True)
            if record.state != "preparing":
                return record
            return await admit_prepared_plan(
                requests,
                record,
                context=value.context,
                intent=admission_stage_intent(
                    record, requests._redactor, prepared=created_receipt.prepared
                ),
            )
        if record.state != "preparing" or not stages:
            return record
        if offset and len(stages) == 1:
            if fork_owner is None:
                raise CollaborationUnavailable("Native view preparation owner is unavailable.")
            from cayu.collaboration._planning_fork import prepare_view_stage

            received = await prepare_view_stage(
                requests, continuation, record, fork_owner, context=context
            )
            record = await _held(requests, continuation, read_only=False, fork_preparation=received)
            continue
        last = stages[-1]
        if last.state != "pending":
            raise CollaborationConflict("Resource preparation was already discharged.")
        command = last.intent.command
        if isinstance(command, RequestResourceAcquisitionStageCommand):
            received = await owner.acquire(command.resource, context=context)
            record = await _held(
                requests, continuation, read_only=False, resource_readback=received
            )
        elif isinstance(command, RequestResourceTransferStageCommand):
            received = await owner.transfer(command.resource, command.acquisition, context=context)
            if len(stages) < final_ordinal - 2:
                record = await _held(
                    requests, continuation, read_only=False, resource_readback=received
                )
            else:
                resolved = await owner.prepare_creation(record, stages, context=context)
                record = await _held(
                    requests, continuation, read_only=False, resource_readback=resolved
                )
        else:
            raise CollaborationUnavailable("Resource frontier has another native command.")
    return record

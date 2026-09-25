"""Planning responsibility participates in the existing namespace lifecycle."""

from cayu.collaboration._planning_records import (
    PENDING_PLANNING_STATES,
    RequestPlanningReceipt,
    RequestPlanningRecord,
    RequestPlanningStageRecord,
)
from cayu.collaboration._preparation import contract_bytes, prepare_contract, require_exact_contract
from cayu.collaboration._request_store import operation_key
from cayu.collaboration.base import _stored_mode
from cayu.collaboration.participants import CollaborationUnavailable


async def creation_permit_planning_parent(tx, permit, *, redactor):
    """A native preparation permit cannot be pruned ahead of its retaining stage."""
    from hashlib import sha256

    from cayu.collaboration._planning_creation_types import RequestCreationStageCommand
    from cayu.collaboration._planning_fork_types import RequestViewStageCommand
    from cayu.collaboration._planning_stages import read_stage

    operation = permit.operation
    registration = permit.intent.request
    if registration.effect_scope == "context_view_retention":
        # The native view contract derives the primary selection operation from
        # the exact selection key. This only locates a row: membership in that
        # row's complete validated permit tuple below supplies the evidence.
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
    if isinstance(stage.intent.command, RequestViewStageCommand):
        if permit not in stage.intent.command.preparation.view.permits:
            raise CollaborationUnavailable("View permit differs from its retained stage tuple.")
    elif isinstance(stage.intent.command, RequestCreationStageCommand):
        require_exact_contract(
            stage.intent.command.preparation.creation.permit, permit, redactor=redactor
        )
    else:
        raise CollaborationUnavailable("Creation permit has another planning-stage owner.")
    require_exact_contract(
        stage, await read_stage(tx, stage.intent, redactor=redactor), redactor=redactor
    )
    return stage.intent.command.expected


async def prune_request_plans(store, tx, initialized, command, *, limit, redactor):
    """Prune an ascending settled prefix under the retired namespace owner."""
    from cayu.collaboration._contracts import ExactMatch
    from cayu.collaboration._history_references import history_references
    from cayu.collaboration._namespace_store import load_namespace
    from cayu.collaboration._planning_stages import read_stage
    from cayu.collaboration._planning_store import read_plan_events, read_plan_in_transaction
    from cayu.collaboration._request_pruning import RequestPruningResult
    from cayu.collaboration._retention_store import release_unused_history

    rows = await tx.scan_request_plans(command.intent.selection.reference, limit=1)
    if not rows:
        return None
    record = prepare_contract(RequestPlanningRecord, rows[0], redactor=redactor)
    require_exact_contract(command, record.receipt.command.expected, redactor=redactor)
    if type(limit) is not int or not 1 <= limit <= 32:
        raise CollaborationUnavailable("Planning pruning requires a bounded batch.")
    anchor = await store._anchor(tx, initialized, redactor)
    namespace = await load_namespace(tx, anchor, command.operation.generation, redactor)
    if namespace.state != "retired":
        raise CollaborationUnavailable("Planning pruning requires retired namespace authority.")
    if not record.pruned_stages:
        result = await read_plan_in_transaction(
            store, tx, initialized, record.receipt.command, redactor=redactor
        )
        if not isinstance(result, ExactMatch) or result.receipt != record:
            raise CollaborationUnavailable("Planning pruning requires exact complete evidence.")
    else:
        retained = prepare_contract(
            RequestPlanningReceipt,
            await tx.get("operations", operation_key(record.receipt.command.operation)),
            redactor=redactor,
        )
        require_exact_contract(retained, record.receipt, redactor=redactor)
    if (
        record.state in PENDING_PLANNING_STATES
        or record.pending_stages
        or record.reserved_bytes
        or record.reserved_events
    ):
        raise CollaborationUnavailable("Planning pruning requires complete settled evidence.")
    plan_events = await read_plan_events(tx, record, redactor=redactor)
    released = events = 0
    references = []
    rows = await tx.scan_request_plan_stages(
        record.receipt.command.operation, limit=record.stage_count + 1
    )
    if len(rows) != record.stage_count - record.pruned_stages:
        raise CollaborationUnavailable("Planning pruning cursor contradicts its remaining stages.")
    removed = min(limit, len(rows))
    # Check the complete remaining ordinal frontier before deleting any prefix.
    for ordinal, raw in enumerate(rows, start=record.pruned_stages + 1):
        stage = prepare_contract(RequestPlanningStageRecord, raw, redactor=redactor)
        if (
            stage.intent.ordinal != ordinal
            or stage.intent.plan != record.receipt.command.operation
            or stage.state == "pending"
        ):
            raise CollaborationUnavailable("Planning pruning frontier is inconsistent.")
    for raw in rows[:removed]:
        stage = prepare_contract(RequestPlanningStageRecord, raw, redactor=redactor)
        require_exact_contract(
            stage, await read_stage(tx, stage.intent, redactor=redactor), redactor=redactor
        )
        key = operation_key(stage.intent.operation)
        registered = prepare_contract(
            RequestPlanningStageRecord, await tx.get("operations", key), redactor=redactor
        )
        references.extend(history_references(registered))
        released += len(contract_bytes(registered, redactor=redactor)) + len(
            contract_bytes(stage, redactor=redactor)
        )
        await tx.delete("request_plan_stages", key)
        await tx.delete("operations", key)
        for event in (stage.registration_event, stage.settlement_event):
            assert event is not None
            released += len(contract_bytes(event, redactor=redactor))
            events += 1
            await tx.delete("request_plan_events", (event.sequence,))
    key = operation_key(record.receipt.command.operation)
    if removed < len(rows) or removed == limit:
        updated = prepare_contract(
            RequestPlanningRecord,
            record.model_copy(update={"pruned_stages": record.pruned_stages + removed}),
            redactor=redactor,
        )
        released += len(contract_bytes(record, redactor=redactor)) - len(
            contract_bytes(updated, redactor=redactor)
        )
        await tx.put("request_plans", key, updated, insert=False)
        released += await release_unused_history(tx, tuple(references), redactor)
        return RequestPruningResult(released, removed, events)
    references.extend(history_references(record.receipt))
    for event in plan_events:
        released += len(contract_bytes(event, redactor=redactor))
        events += 1
        await tx.delete("request_plan_events", (event.sequence,))
    released += len(contract_bytes(record.receipt, redactor=redactor)) + len(
        contract_bytes(record, redactor=redactor)
    )
    await tx.delete("request_plans", key)
    await tx.delete("operations", key)
    released += await release_unused_history(tx, tuple(references), redactor)
    return RequestPruningResult(released, removed + 1, events)


async def operation_retains_planning(store, tx, initialized, raw, namespace, redactor):
    from cayu.collaboration._contracts import ExactMatch
    from cayu.collaboration._planning_stages import read_stage
    from cayu.collaboration._planning_store import read_plan_in_transaction

    mode = _stored_mode(raw)
    if mode == "request_plan":
        receipt = prepare_contract(RequestPlanningReceipt, raw, redactor=redactor)
        operation = receipt.command.operation
        if (operation.namespace_incarnation, operation.generation) != (
            namespace.namespace_incarnation,
            namespace.generation,
        ):
            return False
        current = prepare_contract(
            RequestPlanningRecord,
            await tx.get("request_plans", operation_key(operation)),
            redactor=redactor,
        )
        require_exact_contract(receipt, current.receipt, redactor=redactor)
        found = await read_plan_in_transaction(
            store, tx, initialized, receipt.command, redactor=redactor
        )
        if not isinstance(found, ExactMatch) or found.receipt != current:
            raise CollaborationUnavailable("Planning retirement lacks exact durable evidence.")
        return current.state in PENDING_PLANNING_STATES or current.pending_stages > 0
    if mode == "request_plan_stage":
        registered = prepare_contract(RequestPlanningStageRecord, raw, redactor=redactor)
        operation = registered.intent.operation
        if (operation.namespace_incarnation, operation.generation) != (
            namespace.namespace_incarnation,
            namespace.generation,
        ):
            return False
        current = prepare_contract(
            RequestPlanningStageRecord,
            await tx.get("request_plan_stages", operation_key(operation)),
            redactor=redactor,
        )
        require_exact_contract(registered.intent, current.intent, redactor=redactor)
        require_exact_contract(
            current, await read_stage(tx, registered.intent, redactor=redactor), redactor=redactor
        )
        if registered.state != "pending":
            raise CollaborationUnavailable("Planning stage registration is inconsistent.")
        return current.state == "pending"
    return False

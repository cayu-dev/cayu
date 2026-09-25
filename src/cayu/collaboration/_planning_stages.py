"""Local receiving-stage responsibility under CollaborationStore ownership.

These helpers do not grant authority. The authenticated coordinator reserves a
stage, and the native receiver consumes it in its own mutation transaction.
"""

from dataclasses import dataclass
from hashlib import sha256

from cayu.collaboration._capacity import require_capacity
from cayu.collaboration._contracts import CollaborationConflict, ExactMatch
from cayu.collaboration._planning_creation_types import (
    RequestCreationStageCommand,
    creation_stage_command,
)
from cayu.collaboration._planning_fork_types import (
    RequestViewStageCommand,
    _ForkPreparationReadback,
    _ViewReadback,
    view_stage_command,
)
from cayu.collaboration._planning_records import (
    RequestPlanningEvent,
    RequestPlanningRecord,
    RequestPlanningStageIntent,
    RequestPlanningStageRecord,
)
from cayu.collaboration._planning_resource_types import (
    RequestResourceStageAdoption,
    ResourceStageCommand,
    _ResourceAdoptionReadback,
    _ResourceCreationReadback,
    _ResourceReleaseReadback,
    resource_stage_command,
)
from cayu.collaboration._planning_store import (
    _bytes,
    _event,
    _require_current_participants,
    _write_transition,
    admission_command,
    read_plan_in_transaction,
)
from cayu.collaboration._preparation import contract_bytes, prepare_contract, require_exact_contract
from cayu.collaboration._request_store import operation_key, require_request_absence
from cayu.collaboration.base import _Anchor
from cayu.collaboration.participants import CollaborationUnavailable
from cayu.collaboration.planning import (
    RequestPlanningClarify,
    RequestPlanningFork,
    RequestPlanningFresh,
    RequestPlanningRequest,
    preparation_stage_count,
)


@dataclass(frozen=True)
class _PlannedStage:
    """Private coordinator dependency, never reconstructed from public input."""

    plan: RequestPlanningRequest
    intent: RequestPlanningStageIntent


async def retain_preparation_stage(store, tx, initialized, prior, intent, *, content, redactor):
    """Retain the first stage and its planning accounting in the caller's transaction.

    Decision owners supply their validated intent and immutable event content.
    Reload after stage retention so its counters are preserved in the transition.
    This helper neither acquires authority nor calls a foreign owner.
    """
    expected = prior.receipt.command
    await retain_stage(store, tx, initialized, expected, intent, redactor=redactor)
    found = await read_plan_in_transaction(store, tx, initialized, expected, redactor=redactor)
    assert isinstance(found, ExactMatch)
    prior = found.receipt
    anchor = await store._anchor(tx, initialized, redactor)
    event = _event(expected, anchor.event_sequence + 1, "plan_preparing", redactor, content=content)
    updated = prior.model_copy(
        update={
            "state": "preparing",
            "revision": prior.revision + 1,
            "event_sequences": (*prior.event_sequences, event.sequence),
            "reserved_events": prior.reserved_events - 1,
            "reserved_bytes": prior.reserved_bytes - 2 * expected.limits.max_record_bytes,
        }
    )
    return await _write_transition(store, tx, initialized, prior, updated, event, redactor)


def clarification_stage_intent(record, redactor):
    command = record.receipt.command
    if not isinstance(record.decision, RequestPlanningClarify):
        raise CollaborationConflict("Planning decision has no clarification stage.")
    commitment = sha256(contract_bytes(command, redactor=redactor)).hexdigest()
    return RequestPlanningStageIntent(
        operation=command.operation.model_copy(update={"caller_key": "plan-stage-" + commitment}),
        plan=command.operation,
        plan_sha256=commitment,
        ordinal=1,
        command=record.decision.opening,
    )


def admission_stage_intent(record, redactor, *, prepared=None):
    command = record.receipt.command
    commitment = sha256(contract_bytes(command, redactor=redactor)).hexdigest()
    return RequestPlanningStageIntent(
        operation=command.operation.model_copy(update={"caller_key": "plan-stage-" + commitment}),
        plan=command.operation,
        plan_sha256=commitment,
        ordinal=preparation_stage_count(record.decision)
        if isinstance(record.decision, (RequestPlanningFresh, RequestPlanningFork))
        else 1,
        command=admission_command(record, redactor, prepared=prepared),
    )


def creation_stage_intent(record, redactor, *, preparation=None):
    command = record.receipt.command
    commitment = sha256(contract_bytes(command, redactor=redactor)).hexdigest()
    return RequestPlanningStageIntent(
        operation=command.operation.model_copy(update={"caller_key": "plan-create-" + commitment}),
        plan=command.operation,
        plan_sha256=commitment,
        ordinal=preparation_stage_count(record.decision) - 1,
        command=creation_stage_command(record, preparation=preparation),
    )


def view_stage_intent(record, redactor):
    command = record.receipt.command
    commitment = sha256(contract_bytes(command, redactor=redactor)).hexdigest()
    return RequestPlanningStageIntent(
        operation=command.operation.model_copy(update={"caller_key": "plan-view-" + commitment}),
        plan=command.operation,
        plan_sha256=commitment,
        ordinal=1,
        command=view_stage_command(record),
    )


def resource_stage_intent(record, index, redactor, *, acquisition=None):
    command = record.receipt.command
    commitment = sha256(contract_bytes(command, redactor=redactor)).hexdigest()
    native = resource_stage_command(record, index, redactor=redactor, acquisition=acquisition)
    offset = 1 if isinstance(record.decision, RequestPlanningFork) else 0
    ordinal = offset + 2 * index + 1 + (acquisition is not None)
    return RequestPlanningStageIntent(
        operation=command.operation.model_copy(
            update={"caller_key": f"plan-resource-stage-{ordinal}-{commitment}"}
        ),
        plan=command.operation,
        plan_sha256=commitment,
        ordinal=ordinal,
        command=native,
    )


async def require_receiving_stage(store, tx, initialized, command, planned, *, redactor):
    raw = await tx.find_request_plan_stage(command.operation)
    if raw is None:
        if planned is not None:
            raise CollaborationConflict("Expected planning stage is not registered.")
        return
    if type(planned) is not _PlannedStage:
        raise CollaborationConflict("Receiving operation belongs to a retained planning stage.")
    require_exact_contract(planned.intent.command, command, redactor=redactor)
    require_exact_contract(
        planned.intent,
        prepare_contract(RequestPlanningStageRecord, raw, redactor=redactor).intent,
        redactor=redactor,
    )
    found = await read_plan_in_transaction(store, tx, initialized, planned.plan, redactor=redactor)
    stage = await read_stage(tx, planned.intent, redactor=redactor)
    if not isinstance(found, ExactMatch) or stage is None:
        raise CollaborationConflict("Receiving operation lacks its exact planning owner.")
    if stage.state == "settled":
        return
    if stage.state != "pending" or found.receipt.state not in {
        "decided",
        "clarifying",
        "preparing",
    }:
        raise CollaborationConflict("Planning owner no longer admits this receiving operation.")
    if await tx.now_ms() >= planned.plan.deadline_at_ms:
        raise CollaborationConflict("Planning receiving deadline has expired.")


async def read_stage(tx, expected, *, redactor):
    """Reconstruct the immutable intent and both independently indexed events."""
    expected = prepare_contract(RequestPlanningStageIntent, expected, redactor=redactor)
    key = operation_key(expected.operation)
    raw = await tx.get("request_plan_stages", key)
    initial_raw = await tx.get("operations", key)
    if raw is None and initial_raw is None:
        return None
    if raw is None or initial_raw is None:
        raise CollaborationUnavailable("Planning stage registration is incomplete.")
    current = prepare_contract(RequestPlanningStageRecord, raw, redactor=redactor)
    initial = prepare_contract(RequestPlanningStageRecord, initial_raw, redactor=redactor)
    require_exact_contract(expected, current.intent, redactor=redactor)
    require_exact_contract(initial.intent, current.intent, redactor=redactor)
    if (
        initial.state != "pending"
        or current.registration_event != initial.registration_event
        or current.registered_at_ms != initial.registered_at_ms
        or (current.state == "pending" and current != initial)
    ):
        raise CollaborationUnavailable("Planning stage contradicts its registration.")
    for event in (current.registration_event, current.settlement_event):
        if event is not None:
            stored = prepare_contract(
                RequestPlanningEvent,
                await tx.get("request_plan_events", (event.sequence,)),
                redactor=redactor,
            )
            require_exact_contract(event, stored, redactor=redactor)
    if isinstance(expected.command, RequestViewStageCommand):
        if current.receipt is None:
            from cayu.collaboration._planning_view_reservation import view_permits_registered

            registered = await view_permits_registered(tx, expected.command, redactor=redactor)
            if current.state == "excluded" and registered:
                raise CollaborationUnavailable("Excluded view stage has admitted permits.")
        else:
            from cayu.collaboration._planning_fork_evidence import require_view_source_settlement

            await require_view_source_settlement(tx, current.receipt, redactor=redactor)
    elif isinstance(expected.command, RequestCreationStageCommand) and current.receipt is None:
        from cayu.collaboration._permit_store import registered_receipt

        if (
            await registered_receipt(tx, expected.command.preparation.creation.permit, redactor)
            is None
        ):
            raise CollaborationUnavailable("Creation stage lost its atomic permit registration.")
    elif current.receipt is not None and isinstance(expected.command, RequestCreationStageCommand):
        from cayu.collaboration._planning_creation_evidence import (
            require_creation_source_settlement,
        )

        await require_creation_source_settlement(tx, current.receipt, redactor=redactor)
    elif isinstance(current.receipt, RequestResourceStageAdoption):
        from cayu.collaboration._planning_resource_evidence import require_resource_adoption

        await require_resource_adoption(tx, expected, current.receipt, redactor=redactor)
    elif current.receipt is not None and not isinstance(expected.command, ResourceStageCommand):
        receipt = prepare_contract(
            type(current.receipt),
            await tx.get("operations", operation_key(expected.command.operation)),
            redactor=redactor,
        )
        require_exact_contract(current.receipt, receipt, redactor=redactor)
    return current


async def retain_stage(
    store,
    tx,
    initialized,
    expected_plan,
    intent,
    *,
    redactor,
    _fork_preparation=None,
    _resource_preparation=None,
):
    intent = prepare_contract(RequestPlanningStageIntent, intent, redactor=redactor)
    if (
        intent.plan != expected_plan.operation
        or intent.plan_sha256
        != sha256(contract_bytes(expected_plan, redactor=redactor)).hexdigest()
    ):
        raise CollaborationConflict("Planning stage differs from its exact parent intent.")
    found = await read_plan_in_transaction(store, tx, initialized, expected_plan, redactor=redactor)
    if not isinstance(found, ExactMatch):
        raise CollaborationConflict("Planning stage has no exact retained parent.")
    prior = found.receipt
    existing = await read_stage(tx, intent, redactor=redactor)
    if existing is not None:
        return existing
    if (
        prior.state not in {"decided", "clarifying", "preparing"}
        or prior.decision is None
        or intent.ordinal != prior.stage_count + 1
        or intent.ordinal > expected_plan.limits.max_stages
        or intent.command.expected != expected_plan.expected
    ):
        raise CollaborationConflict("Planning stage frontier is no longer eligible.")
    prepared = None
    if isinstance(intent.command, ResourceStageCommand):
        from cayu.collaboration._planning_resources import require_resource_stage_predecessor
        from cayu.collaboration._planning_store import _require_current_input

        await _require_current_input(store, tx, initialized, expected_plan, redactor)
        await require_resource_stage_predecessor(
            tx, prior, intent, _resource_preparation, _fork_preparation, redactor=redactor
        )
    if isinstance(prior.decision, (RequestPlanningFresh, RequestPlanningFork)) and not isinstance(
        intent.command, RequestCreationStageCommand | RequestViewStageCommand | ResourceStageCommand
    ):
        from cayu.collaboration._planning_creation_types import prepared_creation_from_stage

        prepared = await prepared_creation_from_stage(tx, prior, redactor=redactor)
    if (
        isinstance(prior.decision, (RequestPlanningFresh, RequestPlanningFork))
        and (prior.decision.resources)
        and isinstance(intent.command, RequestCreationStageCommand)
    ):
        if type(_resource_preparation) is not _ResourceCreationReadback:
            raise CollaborationConflict("Resource creation requires native material readback.")
        require_exact_contract(expected_plan, _resource_preparation.expected, redactor=redactor)
        require_exact_contract(intent.command, _resource_preparation.command, redactor=redactor)
    elif isinstance(prior.decision, RequestPlanningFork) and isinstance(
        intent.command, RequestCreationStageCommand
    ):
        if type(_fork_preparation) is not _ForkPreparationReadback:
            raise CollaborationConflict("FORK creation requires native preparation readback.")
        require_exact_contract(
            view_stage_command(prior), _fork_preparation.command, redactor=redactor
        )
        require_exact_contract(
            intent.command.preparation, _fork_preparation.preparation, redactor=redactor
        )
    native_command = (
        intent.command
        if isinstance(intent.command, ResourceStageCommand)
        else view_stage_command(prior)
        if isinstance(intent.command, RequestViewStageCommand)
        else (
            creation_stage_command(prior, preparation=intent.command.preparation)
            if isinstance(intent.command, RequestCreationStageCommand)
            else (
                prior.decision.opening
                if isinstance(prior.decision, RequestPlanningClarify)
                else admission_command(prior, redactor, prepared=prepared)
            )
        )
    )
    require_exact_contract(native_command, intent.command, redactor=redactor)
    await _require_current_participants(store, tx, initialized, expected_plan, redactor)
    now = await tx.now_ms()
    if now >= expected_plan.deadline_at_ms:
        raise CollaborationConflict("Planning stage deadline has expired.")
    await require_request_absence(store, tx, initialized, intent.operation, redactor)
    await require_request_absence(store, tx, initialized, intent.command.operation, redactor)
    if await tx.get("operations", operation_key(intent.command.operation)) is not None:
        raise CollaborationConflict("Receiving operation already exists outside this stage.")
    if await tx.find_request_plan_stage(intent.command.operation) is not None:
        raise CollaborationConflict("Receiving operation already has a planning owner.")
    if isinstance(intent.command, RequestCreationStageCommand):
        from cayu.collaboration._permit_store import register_permit_in_transaction

        # One local transaction retains the complete foreign intent AND registers
        # the existing native permit with reserved settlement capacity. No
        # SessionStore mutation is performed in this transaction. On failure,
        # both registrations roll back; after commit, recovery discovers the
        # exact target from this stage even before receiving preparation occurs.
        await register_permit_in_transaction(
            store, tx, initialized, intent.command.preparation.creation.permit, redactor
        )
    anchor = await store._anchor(tx, initialized, redactor)
    event = _event(
        expected_plan, anchor.event_sequence + 1, "plan_stage_retained", redactor, content=intent
    ).model_copy(update={"operation": intent.operation})
    ceiling = expected_plan.limits.max_record_bytes
    # Final stage growth, its event, and parent counter growth remain reserved.
    stage = prepare_contract(
        RequestPlanningStageRecord,
        RequestPlanningStageRecord(
            intent=intent,
            state="pending",
            registered_at_ms=now,
            settled_at_ms=None,
            receipt=None,
            registration_event=event,
            settlement_event=None,
            reserved_bytes=3 * ceiling,
        ),
        redactor=redactor,
    )
    from cayu.collaboration._planning_preflight import preflight_stage_terminal

    preflight_stage_terminal(
        stage, initialized, ceiling=ceiling, redactor=redactor, expected_plan=expected_plan
    )
    parent = prepare_contract(
        RequestPlanningRecord,
        prior.model_copy(
            update={
                "stage_count": prior.stage_count + 1,
                "pending_stages": prior.pending_stages + 1,
            }
        ),
        redactor=redactor,
    )
    added = (
        2 * _bytes(stage, ceiling, redactor)
        + _bytes(event, ceiling, redactor)
        + _bytes(parent, ceiling, redactor)
        - _bytes(prior, ceiling, redactor)
    )
    updated = prepare_contract(
        _Anchor,
        anchor.model_copy(
            update={
                "operation_count": anchor.operation_count + 1,
                "event_count": anchor.event_count + 1,
                "event_sequence": event.sequence,
                "retained_bytes": anchor.retained_bytes + added,
                "reserved_bytes": anchor.reserved_bytes + stage.reserved_bytes,
                "reserved_events": anchor.reserved_events + 1,
            }
        ),
        redactor=redactor,
    )
    require_capacity(updated, ordinary=True)
    key = operation_key(intent.operation)
    await tx.put("operations", key, stage, insert=True)
    await tx.put("request_plan_stages", key, stage, insert=True)
    await tx.put("request_plan_events", (event.sequence,), event, insert=True)
    await tx.put("request_plans", operation_key(expected_plan.operation), parent, insert=False)
    await tx.put("anchors", (), updated, insert=False)
    return stage


async def finish_stage(
    store, tx, initialized, expected_plan, intent, receipt, *, redactor, _receiving=None
):
    """Consume native evidence, or fence an absent local receiving operation.

    Exclusion is sound only because the native receiver checks this same row in
    the transaction that performs its mutation. It is not a foreign-store fence.
    """
    found = await read_plan_in_transaction(store, tx, initialized, expected_plan, redactor=redactor)
    if not isinstance(found, ExactMatch):
        raise CollaborationConflict("Planning stage has no exact retained parent.")
    parent = found.receipt
    prior = await read_stage(tx, intent, redactor=redactor)
    if (
        prior is None
        or intent.plan != expected_plan.operation
        or intent.plan_sha256
        != sha256(contract_bytes(expected_plan, redactor=redactor)).hexdigest()
    ):
        raise CollaborationConflict("Exact planning stage is unavailable.")
    state = "excluded" if receipt is None else "settled"
    if prior.state != "pending":
        if prior.state != state:
            raise CollaborationConflict("Planning stage already has a different disposition.")
        if receipt is not None:
            require_exact_contract(prior.receipt, receipt, redactor=redactor)
        return prior
    native = await tx.get("operations", operation_key(intent.command.operation))
    if isinstance(intent.command, ResourceStageCommand):
        proof_type = (
            _ResourceAdoptionReadback
            if isinstance(receipt, RequestResourceStageAdoption)
            else _ResourceReleaseReadback
        )
        if receipt is None or _receiving is None or type(_receiving) is not proof_type:
            raise CollaborationConflict("Resource settlement requires native owner evidence.")
        require_exact_contract(intent.command, receipt.command, redactor=redactor)
        require_exact_contract(_receiving.receipt, receipt, redactor=redactor)
        if isinstance(receipt, RequestResourceStageAdoption):
            from cayu.collaboration._planning_resource_evidence import require_resource_adoption

            await require_resource_adoption(tx, intent, receipt, redactor=redactor)
    elif isinstance(intent.command, RequestViewStageCommand):
        from cayu.collaboration._planning_fork_evidence import require_view_source_settlement
        from cayu.collaboration._planning_view_reservation import view_permits_registered

        if receipt is None:
            # The permit writer checks this same stage in this transaction.
            # Excluding an unregistered pair therefore fences late admission;
            # it does not claim that an admitted native selection stopped.
            if await view_permits_registered(tx, intent.command, redactor=redactor):
                raise CollaborationConflict("Admitted view permits require native settlement.")
        else:
            if type(_receiving) is not _ViewReadback:
                raise CollaborationConflict(
                    "Foreign view settlement requires native owner evidence."
                )
            require_exact_contract(intent.command, receipt.command, redactor=redactor)
            require_exact_contract(_receiving.receipt, receipt, redactor=redactor)
            await require_view_source_settlement(tx, receipt, redactor=redactor)
    elif isinstance(intent.command, RequestCreationStageCommand):
        from cayu.collaboration._planning_creation_evidence import (
            _CreationReadback,
            require_creation_source_settlement,
        )

        if receipt is None or type(_receiving) is not _CreationReadback:
            raise CollaborationConflict(
                "Foreign creation requires authenticated native settlement."
            )
        require_exact_contract(intent.command, receipt.command, redactor=redactor)
        require_exact_contract(_receiving.receipt, receipt, redactor=redactor)
        await require_creation_source_settlement(tx, receipt, redactor=redactor)
    elif receipt is None:
        if native is not None:
            raise CollaborationConflict("An existing receiving operation cannot be excluded.")
        await require_request_absence(store, tx, initialized, intent.command.operation, redactor)
    else:
        # The caller's typed value is not evidence: verify the receiving owner's
        # already-written receipt inside this very same transaction.
        require_exact_contract(intent.command, receipt.command, redactor=redactor)
        require_exact_contract(
            receipt, prepare_contract(type(receipt), native, redactor=redactor), redactor=redactor
        )
    anchor = await store._anchor(tx, initialized, redactor)
    event = _event(
        expected_plan,
        anchor.event_sequence + 1,
        "plan_stage_" + state,
        redactor,
        content=intent if receipt is None else receipt,
    ).model_copy(update={"operation": intent.operation})
    settled = prepare_contract(
        RequestPlanningStageRecord,
        prior.model_copy(
            update={
                "state": state,
                "settled_at_ms": await tx.now_ms(),
                "receipt": receipt,
                "settlement_event": event,
                "reserved_bytes": 0,
            }
        ),
        redactor=redactor,
    )
    updated_parent = prepare_contract(
        RequestPlanningRecord,
        parent.model_copy(
            update={
                "pending_stages": parent.pending_stages - 1,
            }
        ),
        redactor=redactor,
    )
    ceiling = expected_plan.limits.max_record_bytes
    added = (
        _bytes(settled, ceiling, redactor)
        - _bytes(prior, ceiling, redactor)
        + _bytes(event, ceiling, redactor)
        + _bytes(updated_parent, ceiling, redactor)
        - _bytes(parent, ceiling, redactor)
    )
    if (
        max(added, 0) > prior.reserved_bytes
        or anchor.reserved_bytes < prior.reserved_bytes
        or anchor.reserved_events < 1
    ):
        raise CollaborationUnavailable("Planning stage settlement reserve is missing.")
    updated_anchor = prepare_contract(
        _Anchor,
        anchor.model_copy(
            update={
                "event_count": anchor.event_count + 1,
                "event_sequence": event.sequence,
                "retained_bytes": anchor.retained_bytes + added,
                "reserved_bytes": anchor.reserved_bytes - prior.reserved_bytes,
                "reserved_events": anchor.reserved_events - 1,
            }
        ),
        redactor=redactor,
    )
    require_capacity(updated_anchor, ordinary=False)
    await tx.put("request_plan_stages", operation_key(intent.operation), settled, insert=False)
    await tx.put("request_plan_events", (event.sequence,), event, insert=True)
    await tx.put(
        "request_plans", operation_key(expected_plan.operation), updated_parent, insert=False
    )
    await tx.put("anchors", (), updated_anchor, insert=False)
    return settled

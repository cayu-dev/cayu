"""Administrative plan cleanup reuses local receiving fences and question closure."""

from hashlib import sha256

from cayu.collaboration._clarification_commands import ClarificationCloseCommand
from cayu.collaboration._clarification_state import ClarificationQuestionState
from cayu.collaboration._clarification_store import close_in_transaction
from cayu.collaboration._contracts import CollaborationConflict, ExactMatch
from cayu.collaboration._planning_creation_types import RequestCreationStageCommand
from cayu.collaboration._planning_fork_types import RequestViewStageCommand, RequestViewStageReceipt
from cayu.collaboration._planning_records import (
    RequestPlanningRecord,
    RequestPlanningStageRecord,
    planning_disposition,
)
from cayu.collaboration._planning_resource_types import (
    RequestResourceStageAdoption,
    RequestResourceStageRelease,
    ResourceStageCommand,
)
from cayu.collaboration._planning_stages import finish_stage
from cayu.collaboration._planning_store import _event, _write_transition, read_plan_in_transaction
from cayu.collaboration._preparation import contract_bytes, prepare_contract, require_exact_contract
from cayu.collaboration._request_store import operation_key
from cayu.collaboration.planning import (
    MAX_REQUEST_PLANNING_GENERATIONS,
    RequestPlanningClarify,
    RequestPlanningControl,
    RequestPlanningDecline,
    RequestPlanningDefer,
)
from cayu.collaboration.requests import RequestEvent


async def require_settled_preparation(tx, record, *, redactor):
    """Prove that a sealed plan cannot later consume an abandoned preparation.

    Native creation/exclusion is positive receiving evidence, not absence. A
    created child retains its own native ownership; this gate never treats it
    as excluded or implicitly transfers it into a replacement plan. Sealed
    local admission prevents an old worker from admitting that child later.
    """
    from cayu.collaboration._planning_creation_types import RequestCreationStageReceipt
    from cayu.collaboration._planning_stages import read_stage
    from cayu.collaboration.requests import RequestAdmissionCommand

    if record.state != "cancelled" or record.control is None or record.pending_stages:
        raise CollaborationConflict("Planning preparation is not sealed and settled.")
    if (
        await tx.get("operations", operation_key(record.receipt.command.admission_operation))
        is not None
    ):
        raise CollaborationConflict("A prior native admission is not an excluded preparation.")
    for raw in await tx.scan_request_plan_stages(
        record.receipt.command.operation, limit=record.stage_count + 1
    ):
        stage = prepare_contract(RequestPlanningStageRecord, raw, redactor=redactor)
        stage = await read_stage(tx, stage.intent, redactor=redactor)
        if stage is None:
            raise CollaborationConflict("Planning preparation evidence is unavailable.")
        if isinstance(stage.intent.command, RequestCreationStageCommand):
            if (
                stage.state != "settled"
                or not isinstance(stage.receipt, RequestCreationStageReceipt)
                or stage.receipt.decision.state not in {"created", "excluded"}
            ):
                raise CollaborationConflict("Unresolved preparation cannot be replaced.")
        elif isinstance(stage.intent.command, RequestViewStageCommand):
            if stage.state != "excluded" and (
                stage.state != "settled" or not isinstance(stage.receipt, RequestViewStageReceipt)
            ):
                raise CollaborationConflict("Retained view responsibility remains unresolved.")
        elif isinstance(stage.intent.command, ResourceStageCommand):
            if stage.state != "settled" or not isinstance(
                stage.receipt, (RequestResourceStageRelease, RequestResourceStageAdoption)
            ):
                raise CollaborationConflict("Native resource responsibility remains unresolved.")
        elif (
            not isinstance(stage.intent.command, RequestAdmissionCommand)
            or stage.state != "excluded"
        ):
            raise CollaborationConflict("Planning preparation lacks exact exclusion evidence.")


async def local_planning_quiescence(
    store, tx, initialized, snapshot, *, redactor, pending_decline=None
):
    """Prove local settlement from native history, not missing foreign evidence."""
    if snapshot.admission not in {"deferred", "clarifying"} or snapshot.delivery != "pending":
        return False
    rows = await tx.scan_request_plans(
        snapshot.receipt.expected.intent.selection.reference,
        limit=MAX_REQUEST_PLANNING_GENERATIONS + 1,
    )
    if not rows or len(rows) > MAX_REQUEST_PLANNING_GENERATIONS:
        return False
    matched = False
    local_admissions = set()
    local_questions = set()
    for raw in rows:
        record = prepare_contract(RequestPlanningRecord, raw, redactor=redactor)
        require_exact_contract(
            snapshot.receipt.expected, record.receipt.command.expected, redactor=redactor
        )
        found = await read_plan_in_transaction(
            store, tx, initialized, record.receipt.command, redactor=redactor
        )
        if not isinstance(found, ExactMatch):
            return False
        require_exact_contract(record, found.receipt, redactor=redactor)
        if pending_decline is not None and record.receipt.command == pending_decline:
            # The retained successor has not dispatched anything: its decline
            # proposal is decided, has no stage, and has no native admission yet.
            if (
                record.state != "decided"
                or not isinstance(record.decision, RequestPlanningDecline)
                or record.stage_count
            ):
                return False
            continue
        if isinstance(record.decision, RequestPlanningDefer):
            if record.stage_count:
                return False
        elif isinstance(record.decision, RequestPlanningClarify):
            from cayu.collaboration._clarification_store import lookup_clarification_in_transaction

            if record.stage_count != 1 or record.pending_stages:
                return False
            opening = record.decision.opening
            question = await lookup_clarification_in_transaction(
                store, tx, initialized, opening, redactor=redactor, include_state=True
            )
            if not isinstance(question, ExactMatch) or question.receipt.state == "open":
                return False
            stages = await tx.scan_request_plan_stages(record.receipt.command.operation, limit=2)
            if len(stages) != 1:
                return False
            stage = prepare_contract(RequestPlanningStageRecord, stages[0], redactor=redactor)
            if stage.state != "settled" or stage.intent.command != opening:
                return False
            local_questions.add(operation_key(opening.operation))
        else:
            return False
        local_admissions.add(operation_key(record.receipt.command.admission_operation))
        if record.receipt.command.admission_operation == snapshot.admission_operation:
            matched = record.state in {"deferred", "clarifying", "cancelled", "expired"} or (
                pending_decline is not None
                and record.state == "superseded"
                and record.successor is not None
                and record.successor.request == pending_decline
            )
    # A prior ordinary admission can own effects even if the latest decision is
    # local defer. Every admission in the native request history must belong to
    # these positively authenticated non-dispatching plans.
    for sequence in snapshot.event_sequences:
        event = prepare_contract(
            RequestEvent, await tx.get("request_events", (sequence,)), redactor=redactor
        )
        require_exact_contract(
            snapshot.receipt.expected.intent.selection.reference, event.request, redactor=redactor
        )
        if event.sequence != sequence:
            raise CollaborationConflict("Request control event frontier is inconsistent.")
        if (
            event.type == "request_admission"
            and operation_key(event.operation) not in local_admissions
        ):
            return False
    if snapshot.clarification.generation or local_questions:
        from cayu.collaboration._clarification_pruning import question_pruning_material

        # The existing owner proves complete terminal question history and no
        # pending delivery/service handoffs. Native registration of either
        # handoff checks parent.state in this same transaction, so terminalizing
        # the request fences a delayed new registration. No foreign I/O occurs.
        questions, _ = await question_pruning_material(tx, snapshot, redactor)
        if {operation_key(item.question.operation) for item in questions} != local_questions:
            return False
    return matched


async def control_request_plans(store, tx, initialized, receipt, *, redactor):
    """Fence local planning in the same transaction as authenticated root control."""
    rows = await tx.scan_request_plans(
        receipt.expected.intent.expected.intent.selection.reference,
        limit=MAX_REQUEST_PLANNING_GENERATIONS + 1,
    )
    if len(rows) > MAX_REQUEST_PLANNING_GENERATIONS:
        raise CollaborationConflict("Request planning history exceeds its bound.")
    for raw in rows:
        record = prepare_contract(RequestPlanningRecord, raw, redactor=redactor)
        require_exact_contract(
            receipt.expected.intent.expected, record.receipt.command.expected, redactor=redactor
        )
        if record.state in {"admitted", "declined", "superseded", "cancelled", "expired"}:
            continue
        await control_plan_in_transaction(
            store,
            tx,
            initialized,
            RequestPlanningControl(
                expected=record.receipt.command,
                expected_revision=record.revision,
                kind=receipt.state,
                initiator=receipt.expected.initiator,
            ),
            redactor=redactor,
        )


async def control_plan_in_transaction(store, tx, initialized, command, *, redactor):
    command = prepare_contract(RequestPlanningControl, command, redactor=redactor)
    expected = command.expected
    found = await read_plan_in_transaction(store, tx, initialized, expected, redactor=redactor)
    if not isinstance(found, ExactMatch):
        raise CollaborationConflict("Planning control has no exact retained intent.")
    prior = found.receipt
    if prior.control is not None:
        require_exact_contract(prior.control, command, redactor=redactor)
        return prior
    if prior.state in {"admitted", "declined", "superseded", "cancelled", "expired"} or (
        prior.revision != command.expected_revision
    ):
        raise CollaborationConflict("Planning control revision or state changed.")
    now = await tx.now_ms()
    deadline = expected.deadline_at_ms
    if isinstance(prior.decision, RequestPlanningDefer):
        deadline = min(deadline, prior.decision.prerequisite.deadline_at_ms)
    elif isinstance(prior.decision, RequestPlanningClarify):
        deadline = min(deadline, prior.decision.opening.question.deadline_at_ms)
    if command.kind == "expired" and now < deadline:
        from cayu.collaboration._request_store import retained_request

        parent = await retained_request(
            store,
            tx,
            initialized,
            expected.expected.intent.request,
            expected.expected.initiator,
            redactor,
        )
        if parent is None or parent.state != "expired":
            raise CollaborationConflict("Planning deadline has not expired.")
        require_exact_contract(expected.expected, parent.receipt.expected, redactor=redactor)
    for raw in await tx.scan_request_plan_stages(
        expected.operation, limit=expected.limits.max_stages + 1
    ):
        stage = prepare_contract(RequestPlanningStageRecord, raw, redactor=redactor)
        if stage.state == "pending":
            if isinstance(stage.intent.command, RequestViewStageCommand):
                from cayu.collaboration._planning_view_reservation import view_permits_registered

                if not await view_permits_registered(tx, stage.intent.command, redactor=redactor):
                    await finish_stage(
                        store, tx, initialized, expected, stage.intent, None, redactor=redactor
                    )
                    continue
            if isinstance(
                stage.intent.command,
                RequestCreationStageCommand | RequestViewStageCommand | ResourceStageCommand,
            ):
                # A local cancellation cannot fence a foreign receiver. Keep
                # this responsibility/reservation until native exclusion or
                # creation settlement is authenticated by the recovery owner.
                continue
            await finish_stage(
                store, tx, initialized, expected, stage.intent, None, redactor=redactor
            )
    if isinstance(prior.decision, RequestPlanningClarify):
        opening = prior.decision.opening
        raw = await tx.get("clarification_questions", operation_key(opening.operation))
        if raw is not None:
            question = prepare_contract(ClarificationQuestionState, raw, redactor=redactor)
            require_exact_contract(opening.question, question.question, redactor=redactor)
            if question.state == "open":
                identity = sha256(contract_bytes(command, redactor=redactor)).hexdigest()
                await close_in_transaction(
                    store,
                    tx,
                    initialized,
                    ClarificationCloseCommand(
                        operation=expected.operation.model_copy(
                            update={"caller_key": "plan-close-" + identity}
                        ),
                        expected=expected.expected,
                        question=opening.operation,
                        question_sha256=sha256(
                            contract_bytes(opening.question, redactor=redactor)
                        ).hexdigest(),
                        kind=command.kind,
                        initiator=command.initiator,
                    ),
                    redactor=redactor,
                )
    # Stage settlement may have changed counters. Native question closure keeps
    # its own delivery/service obligations; this does not declare those quiescent.
    found = await read_plan_in_transaction(store, tx, initialized, expected, redactor=redactor)
    assert isinstance(found, ExactMatch)
    prior = found.receipt
    anchor = await store._anchor(tx, initialized, redactor)
    event = _event(
        expected, anchor.event_sequence + 1, "plan_" + command.kind, redactor, content=command
    )
    updated = prior.model_copy(
        update={
            "state": command.kind,
            "disposition": planning_disposition(command),
            "revision": prior.revision + 1,
            "event_sequences": (*prior.event_sequences, event.sequence),
            "reserved_bytes": 0,
            "reserved_events": 0,
        }
    )
    return await _write_transition(store, tx, initialized, prior, updated, event, redactor)

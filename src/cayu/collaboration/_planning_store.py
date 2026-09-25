"""Planning transitions under the existing CollaborationStore transaction.

No registration callback, policy evaluation, foreign effect, or authorization
resolver runs here. The registered coordinator must hold current authority;
these functions additionally order request/lifecycle elections at the owner.
"""

from __future__ import annotations

from hashlib import sha256
from typing import TYPE_CHECKING
from uuid import uuid4

from cayu.collaboration._capacity import require_capacity
from cayu.collaboration._contracts import (
    CollaborationConflict,
    CollaborationContractError,
    ExactConflict,
    ExactMatch,
    ExactNotFound,
    ExactUnavailable,
)
from cayu.collaboration._namespace_store import require_open_namespace
from cayu.collaboration._planning_records import (
    MAX_REQUEST_PLAN_EVENTS,
    PlanningEventType,
    RequestPlanningEvent,
    RequestPlanningReceipt,
    RequestPlanningRecord,
    RequestPlanningSuccessor,
)
from cayu.collaboration._preparation import contract_bytes, prepare_contract, require_exact_contract
from cayu.collaboration._request_store import (
    operation_key,
    retained_request,
)
from cayu.collaboration.access import CollaborationAccessDenied
from cayu.collaboration.base import _Anchor, _stored_mode
from cayu.collaboration.participants import CollaborationUnavailable
from cayu.collaboration.planning import (
    MAX_REQUEST_PLANNING_GENERATIONS,
    ConfiguredRequestPlanningPolicy,
    RequestPlanningClarify,
    RequestPlanningContinue,
    RequestPlanningDecision,
    RequestPlanningDefer,
    RequestPlanningFork,
    RequestPlanningFresh,
    RequestPlanningRequest,
    RequestPlanningTimer,
    planning_policy_commitment,
)
from cayu.collaboration.requests import RequestAdmissionCommand, RequestAdmissionReceipt

if TYPE_CHECKING:
    from cayu.collaboration.base import CollaborationStore, _Repository
    from cayu.collaboration.participants import CollaborationInitialization
    from cayu.vaults.redaction import SecretRedactor


def _bytes(value, limit: int, redactor: SecretRedactor) -> int:
    size = len(contract_bytes(value, redactor=redactor))
    if size > limit:
        raise CollaborationContractError("Planning record exceeds its admitted byte ceiling.")
    return size


def _event(
    command: RequestPlanningRequest,
    sequence: int,
    kind: PlanningEventType,
    redactor,
    *,
    content=None,
):
    selected = command.expected.intent.selection
    return RequestPlanningEvent(
        id=uuid4().hex,
        sequence=sequence,
        operation=command.operation,
        plan=command.operation,
        request=selected.reference,
        type=kind,
        commitment=sha256(
            contract_bytes(command if content is None else content, redactor=redactor)
        ).hexdigest(),
        participants=tuple(
            dict.fromkeys((selected.sender.reference, selected.recipient.reference))
        ),
    )


async def read_plan_events(tx, record: RequestPlanningRecord, *, redactor):
    """Authenticate the bounded event frontier even during incremental pruning.

    Earlier stages may already be reclaimed in a retired namespace. That does
    not make a divergent indexed event eligible for deletion by this plan.
    """
    receipt = record.receipt
    command = receipt.command
    last_event_type = {
        "evaluating": "plan_retained",
        "decided": "plan_decided",
        "deferred": "plan_waiting",
        "clarifying": "plan_waiting",
        "preparing": "plan_preparing",
        "admitted": "plan_admitted",
        "declined": "plan_declined",
        "cancelled": "plan_cancelled",
        "expired": "plan_expired",
        "superseded": "plan_superseded",
    }[record.state]
    events = []
    for index, sequence in enumerate(record.event_sequences):
        event = prepare_contract(
            RequestPlanningEvent,
            await tx.get("request_plan_events", (sequence,)),
            redactor=redactor,
        )
        if (
            event.sequence != sequence
            or event.operation != command.operation
            or event.plan != command.operation
            or event.request != command.expected.intent.selection.reference
            or event.participants != receipt.event.participants
            or (index == 0 and event != receipt.event)
            or (index == len(record.event_sequences) - 1 and event.type != last_event_type)
        ):
            raise CollaborationUnavailable("Planning event frontier contradicts its receipt.")
        events.append(event)
    return tuple(events)


async def read_plan_in_transaction(
    store: CollaborationStore,
    tx: _Repository,
    initialized: CollaborationInitialization,
    expected: RequestPlanningRequest,
    *,
    redactor: SecretRedactor,
):
    """Compare the full expected intent, including after an acknowledgement loss."""
    expected = prepare_contract(RequestPlanningRequest, expected, redactor=redactor)
    anchor = await store._anchor(tx, initialized, redactor)
    if (
        expected.operation.application_scope != initialized.binding.application_scope
        or expected.operation.namespace_incarnation != initialized.namespace_incarnation
    ):
        return ExactConflict()
    key = operation_key(expected.operation)
    raw = await tx.get("operations", key)
    current = await tx.get("request_plans", key)
    if raw is None:
        if current is not None:
            raise CollaborationUnavailable("Planning registration receipt is missing.")
        return await store._missing_operation(tx, anchor, expected.operation.generation, redactor)
    if _stored_mode(raw) != "request_plan":
        return ExactConflict()
    receipt = prepare_contract(RequestPlanningReceipt, raw, redactor=redactor)
    planning_policy_commitment(receipt.policy, redactor=redactor)
    record = prepare_contract(RequestPlanningRecord, current, redactor=redactor)
    if receipt.command.operation != expected.operation:
        raise CollaborationUnavailable("Planning operation index contradicts its receipt.")
    require_exact_contract(receipt, record.receipt, redactor=redactor)
    if record.pruned_stages:
        # The retired namespace's maintenance owner retains this cursor only to
        # finish bounded reclamation. Partial evidence cannot authenticate replay.
        return ExactUnavailable()
    command = receipt.command
    parent = await retained_request(
        store,
        tx,
        initialized,
        command.expected.intent.request,
        command.expected.initiator,
        redactor,
    )
    if parent is None:
        raise CollaborationUnavailable("Planning parent acceptance is missing.")
    require_exact_contract(command.expected, parent.receipt.expected, redactor=redactor)
    prepared_creation = None
    if isinstance(record.decision, (RequestPlanningFresh, RequestPlanningFork)):
        from cayu.collaboration._planning_creation_types import prepared_creation_from_stage

        prepared_creation = await prepared_creation_from_stage(tx, record, redactor=redactor)
    has_admission = False
    has_preparation = False
    for event in await read_plan_events(tx, record, redactor=redactor):
        if event.type == "plan_decided" and (
            record.decision is None
            or event.commitment
            != sha256(contract_bytes(record.decision, redactor=redactor)).hexdigest()
        ):
            raise CollaborationUnavailable("Planning decision commitment contradicts its event.")
        if event.type == "plan_preparing" and isinstance(record.decision, RequestPlanningFork):
            from cayu.collaboration._planning_fork_types import view_stage_command

            if (
                event.commitment
                != sha256(contract_bytes(view_stage_command(record), redactor=redactor)).hexdigest()
            ):
                raise CollaborationUnavailable("View preparation commitment contradicts its event.")
        elif event.type == "plan_preparing" and isinstance(record.decision, RequestPlanningFresh):
            from cayu.collaboration._planning_creation_types import creation_stage_command

            if (
                event.commitment
                != sha256(
                    contract_bytes(
                        record.decision
                        if record.decision.resources
                        else creation_stage_command(record),
                        redactor=redactor,
                    )
                ).hexdigest()
            ):
                raise CollaborationUnavailable(
                    "Planning creation commitment contradicts its event."
                )
        elif event.type in {
            "plan_waiting",
            "plan_declined",
            "plan_admitted",
            "plan_preparing",
        } and (
            event.commitment
            != sha256(
                contract_bytes(
                    admission_command(record, redactor, prepared=prepared_creation),
                    redactor=redactor,
                )
            ).hexdigest()
        ):
            raise CollaborationUnavailable("Planning admission commitment contradicts its event.")
        has_admission |= event.type in {"plan_waiting", "plan_declined", "plan_admitted"}
        has_preparation |= event.type == "plan_preparing"
        if event.type == "plan_superseded" and (
            record.successor is None
            or event.commitment
            != sha256(contract_bytes(record.successor, redactor=redactor)).hexdigest()
        ):
            raise CollaborationUnavailable("Planning successor commitment contradicts its event.")
        if event.type in {"plan_cancelled", "plan_expired"} and (
            record.control is None
            or event.commitment
            != sha256(contract_bytes(record.control, redactor=redactor)).hexdigest()
        ):
            raise CollaborationUnavailable("Planning control commitment contradicts its event.")
    _bytes(record, command.limits.max_record_bytes, redactor)
    from cayu.collaboration._planning_records import RequestPlanningStageRecord
    from cayu.collaboration._planning_stages import read_stage

    stages = await tx.scan_request_plan_stages(
        command.operation, limit=command.limits.max_stages + 1
    )
    if len(stages) != record.stage_count:
        raise CollaborationUnavailable("Planning stage frontier is incomplete.")
    if isinstance(record.decision, RequestPlanningContinue) and (
        record.stage_count > 1 or (has_preparation and record.stage_count != 1)
    ):
        raise CollaborationUnavailable("Prepared planning stage frontier is inconsistent.")
    if isinstance(record.decision, (RequestPlanningFresh, RequestPlanningFork)):
        from cayu.collaboration.planning import preparation_stage_count

        total = preparation_stage_count(record.decision)
        if (
            record.stage_count > total
            or total > command.limits.max_stages
            or len(record.decision.resources) > command.limits.max_resources
            or (has_preparation and record.stage_count < 1)
            or (has_admission and record.stage_count != total)
        ):
            raise CollaborationUnavailable("Preparation stage frontier is inconsistent.")
    pending = 0
    for ordinal, raw_stage in enumerate(stages, start=1):
        stage = prepare_contract(RequestPlanningStageRecord, raw_stage, redactor=redactor)
        if (
            stage.intent.ordinal != ordinal
            or stage.intent.plan_sha256
            != sha256(contract_bytes(command, redactor=redactor)).hexdigest()
        ):
            raise CollaborationUnavailable("Planning stage parent commitment conflicts.")
        await read_stage(tx, stage.intent, redactor=redactor)
        if isinstance(record.decision, RequestPlanningContinue):
            require_exact_contract(
                admission_command(record, redactor), stage.intent.command, redactor=redactor
            )
        elif isinstance(record.decision, (RequestPlanningFresh, RequestPlanningFork)):
            from cayu.collaboration._planning_creation_types import (
                RequestCreationStageCommand,
                creation_stage_command,
            )
            from cayu.collaboration._planning_fork_types import view_stage_command
            from cayu.collaboration._planning_resource_types import (
                RequestResourceTransferStageCommand,
                resource_stage_command,
            )
            from cayu.collaboration.planning import preparation_stage_count

            offset = 1 if isinstance(record.decision, RequestPlanningFork) else 0
            final_ordinal = preparation_stage_count(record.decision)
            if offset and ordinal == 1:
                expected_command = view_stage_command(record)
            elif ordinal < final_ordinal - 1:
                resource_index, transfer = divmod(ordinal - offset - 1, 2)
                acquisition = None
                if transfer:
                    if not isinstance(stage.intent.command, RequestResourceTransferStageCommand):
                        raise CollaborationUnavailable(
                            "Resource transfer stage has another command."
                        )
                    acquisition = stage.intent.command.acquisition
                expected_command = resource_stage_command(
                    record,
                    resource_index,
                    redactor=redactor,
                    acquisition=acquisition,
                )
            elif ordinal == final_ordinal - 1 and isinstance(
                stage.intent.command, RequestCreationStageCommand
            ):
                expected_command = creation_stage_command(
                    record, preparation=stage.intent.command.preparation
                )
            elif ordinal == final_ordinal:
                expected_command = admission_command(record, redactor, prepared=prepared_creation)
            else:
                raise CollaborationUnavailable("Preparation stage has an unsupported command.")
            require_exact_contract(expected_command, stage.intent.command, redactor=redactor)
        pending += stage.state == "pending"
    if pending != record.pending_stages:
        raise CollaborationUnavailable("Planning pending-stage count conflicts.")
    if has_admission:
        admitted = prepare_contract(
            RequestAdmissionReceipt,
            await tx.get("operations", operation_key(command.admission_operation)),
            redactor=redactor,
        )
        require_exact_contract(
            admission_command(record, redactor, prepared=prepared_creation),
            admitted.command,
            redactor=redactor,
        )
        from cayu.collaboration._request_store import require_request_event

        await require_request_event(tx, admitted.event, redactor)
        from cayu.collaboration._prepared_admission_store import require_prepared_admission_evidence

        await require_prepared_admission_evidence(tx, admitted, redactor=redactor)
    if command != expected:
        return ExactConflict()
    return ExactMatch[RequestPlanningRecord](receipt=record)


async def _require_current_input(store, tx, initialized, command, redactor):
    parent = await retained_request(
        store,
        tx,
        initialized,
        command.expected.intent.request,
        command.expected.initiator,
        redactor,
    )
    if parent is None:
        raise CollaborationUnavailable("Planning requires exact request acceptance.")
    require_exact_contract(command.expected, parent.receipt.expected, redactor=redactor)
    if (
        parent.state != "open"
        or parent.admission in {"admitted", "closed"}
        or parent.revision != command.expected_revision
        or parent.admission_generation + 1 != command.admission_generation
        or parent.clarification.input_revision != command.expected_input_revision
        or (
            parent.clarification.input_sha256
            or sha256(contract_bytes(command.expected, redactor=redactor)).hexdigest()
        )
        != command.expected_input_sha256
    ):
        raise CollaborationConflict("Planning request or effective-input frontier changed.")
    now = await tx.now_ms()
    if now >= command.deadline_at_ms:
        raise CollaborationConflict("Planning deadline has expired.")
    return parent, now


async def _require_current_participants(store, tx, initialized, command, redactor):
    selected = command.expected.intent.selection
    for snapshot in (selected.sender, selected.recipient):
        current = await store._participant(tx, snapshot.reference, initialized.owner, redactor)
        if current.lifecycle != "active" or (
            current.configuration_revision,
            current.lifecycle_revision,
            current.admission_generation,
        ) != (
            snapshot.configuration_revision,
            snapshot.lifecycle_revision,
            snapshot.admission_generation,
        ):
            raise CollaborationAccessDenied("Planning participant authority changed.")


async def retain_plan_in_transaction(
    store: CollaborationStore,
    tx: _Repository,
    initialized: CollaborationInitialization,
    command: RequestPlanningRequest,
    policy: ConfiguredRequestPlanningPolicy,
    *,
    redactor: SecretRedactor,
    prerequisite=None,
) -> RequestPlanningRecord:
    command = prepare_contract(RequestPlanningRequest, command, redactor=redactor)
    found = await read_plan_in_transaction(store, tx, initialized, command, redactor=redactor)
    if isinstance(found, ExactMatch):
        # Already retained policy must survive deployment/configuration changes.
        return found.receipt
    if isinstance(found, ExactConflict):
        raise CollaborationConflict("Planning operation conflicts with retained evidence.")
    if not isinstance(found, ExactNotFound):
        raise CollaborationUnavailable("Planning history is unavailable for a new operation.")
    policy = prepare_contract(ConfiguredRequestPlanningPolicy, policy, redactor=redactor)
    if policy.reference != command.policy or (
        planning_policy_commitment(policy, redactor=redactor) != command.policy_sha256
    ):
        raise CollaborationConflict("Planning policy differs from the exact expected registration.")
    anchor = await store._anchor(tx, initialized, redactor)
    await require_open_namespace(tx, anchor, command.operation, redactor)
    parent, now = await _require_current_input(store, tx, initialized, command, redactor)
    await _require_current_participants(store, tx, initialized, command, redactor)
    prior = await tx.scan_request_plans(
        parent.receipt.expected.intent.selection.reference,
        limit=MAX_REQUEST_PLANNING_GENERATIONS + 1,
    )
    if prior:
        await _supersede_for_successor(
            store,
            tx,
            initialized,
            command,
            prior,
            now,
            redactor=redactor,
            prerequisite_evidence=prerequisite,
        )
        anchor = await store._anchor(tx, initialized, redactor)
    elif command.planning_generation != 1 or command.predecessor is not None:
        raise CollaborationConflict("Planning requires its exact predecessor history.")
    if await tx.get("operations", operation_key(command.admission_operation)) is not None:
        raise CollaborationConflict("Proposed admission operation is already occupied.")
    event = _event(command, anchor.event_sequence + 1, "plan_retained", redactor)
    receipt = prepare_contract(
        RequestPlanningReceipt,
        RequestPlanningReceipt(command=command, policy=policy, retained_at_ms=now, event=event),
        redactor=redactor,
    )
    remaining = MAX_REQUEST_PLAN_EVENTS - 1
    # Each future transition can replace one bounded record (growth <= R) and
    # append one bounded event (<= R). Reserve both before policy evaluation.
    record = prepare_contract(
        RequestPlanningRecord,
        RequestPlanningRecord(
            receipt=receipt,
            revision=1,
            state="evaluating",
            decision_commitment=None,
            stage_count=0,
            pending_stages=0,
            next_due_at_ms=command.deadline_at_ms,
            event_sequences=(event.sequence,),
            reserved_events=remaining,
            reserved_bytes=2 * remaining * command.limits.max_record_bytes,
        ),
        redactor=redactor,
    )
    retained = sum(
        _bytes(item, command.limits.max_record_bytes, redactor) for item in (receipt, record, event)
    )
    from cayu.collaboration._planning_preflight import preflight_plan_control

    preflight_plan_control(record)
    updated = prepare_contract(
        _Anchor,
        anchor.model_copy(
            update={
                "operation_count": anchor.operation_count + 1,
                "event_count": anchor.event_count + 1,
                "event_sequence": event.sequence,
                "retained_bytes": anchor.retained_bytes + retained,
                "reserved_bytes": anchor.reserved_bytes + record.reserved_bytes,
                "reserved_events": anchor.reserved_events + record.reserved_events,
            }
        ),
        redactor=redactor,
    )
    require_capacity(updated, ordinary=True)
    key = operation_key(command.operation)
    await tx.put("operations", key, receipt, insert=True)
    await tx.put("request_plans", key, record, insert=True)
    await tx.put("request_plan_events", (event.sequence,), event, insert=True)
    await tx.put("anchors", (), updated, insert=False)
    return record


async def _supersede_for_successor(
    store, tx, initialized, command, history, now, *, redactor, prerequisite_evidence=None
):
    """Elect an explicit successor; never silently reevaluate a retained plan."""
    records = [prepare_contract(RequestPlanningRecord, raw, redactor=redactor) for raw in history]
    if (
        len(records) >= MAX_REQUEST_PLANNING_GENERATIONS
        or command.predecessor is None
        or command.planning_generation != len(records) + 1
        or any(
            record.receipt.command.planning_generation != index
            for index, record in enumerate(records, start=1)
        )
        or command.planning_generation
        > min(
            command.limits.max_generations,
            *(record.receipt.command.limits.max_generations for record in records),
        )
    ):
        raise CollaborationConflict("Planning successor generation or retained history conflicts.")
    last = records[-1]
    expected = last.receipt.command
    if (
        command.predecessor.operation != expected.operation
        or command.predecessor.revision != last.revision
        or command.expected != expected.expected
        or command.deadline_at_ms > expected.deadline_at_ms
    ):
        raise CollaborationConflict("Planning successor differs from its exact predecessor.")
    found = await read_plan_in_transaction(store, tx, initialized, expected, redactor=redactor)
    if not isinstance(found, ExactMatch):
        raise CollaborationUnavailable("Planning predecessor evidence is unavailable.")
    last = found.receipt
    if last.state == "cancelled":
        from cayu.collaboration._planning_control import require_settled_preparation

        await require_settled_preparation(tx, last, redactor=redactor)
        # The terminal control receipt and its already-released reserves remain
        # immutable. The successor's own retained request names this exact
        # predecessor; insertion/generation arbitration is in this transaction.
        return
    if last.pending_stages or last.state not in {"deferred", "clarifying"}:
        raise CollaborationConflict("Planning predecessor is not settled for replacement.")
    if isinstance(last.decision, RequestPlanningDefer):
        prerequisite = last.decision.prerequisite
        if now >= prerequisite.deadline_at_ms:
            raise CollaborationConflict("Planning prerequisite deadline has expired.")
        if isinstance(prerequisite, RequestPlanningTimer):
            if now < prerequisite.not_before_ms:
                raise CollaborationConflict("Planning prerequisite is not due.")
        else:
            from cayu.collaboration._planning_prerequisite import _PrerequisiteEvidence

            if type(prerequisite_evidence) is not _PrerequisiteEvidence:
                raise CollaborationUnavailable(
                    "Planning prerequisite requires its registered reader."
                )
            require_exact_contract(prerequisite, prerequisite_evidence.expected, redactor=redactor)
            require_exact_contract(
                prerequisite.expected, prerequisite_evidence.receipt.command, redactor=redactor
            )
    elif isinstance(last.decision, RequestPlanningClarify):
        from cayu.collaboration._clarification_state import ClarificationQuestionState

        question = prepare_contract(
            ClarificationQuestionState,
            await tx.get("clarification_questions", operation_key(last.decision.opening.operation)),
            redactor=redactor,
        )
        require_exact_contract(last.decision.opening.question, question.question, redactor=redactor)
        if question.state != "answered" or (
            command.expected_input_revision <= expected.expected_input_revision
        ):
            raise CollaborationConflict("Planning clarification has no accepted successor input.")
    else:
        raise CollaborationConflict("Planning predecessor has no reevaluation boundary.")
    anchor = await store._anchor(tx, initialized, redactor)
    witness = RequestPlanningSuccessor(
        request=command,
        prerequisite=None if prerequisite_evidence is None else prerequisite_evidence.receipt,
    )
    event = _event(
        expected, anchor.event_sequence + 1, "plan_superseded", redactor, content=witness
    )
    updated = last.model_copy(
        update={
            "state": "superseded",
            "successor": witness,
            "revision": last.revision + 1,
            "event_sequences": (*last.event_sequences, event.sequence),
            "reserved_bytes": 0,
            "reserved_events": 0,
        }
    )
    await _write_transition(store, tx, initialized, last, updated, event, redactor)


async def _write_transition(store, tx, initialized, prior, updated, event, redactor):
    command = prior.receipt.command
    ceiling = command.limits.max_record_bytes
    next_record = prepare_contract(RequestPlanningRecord, updated, redactor=redactor)
    if next_record.state in {"evaluating", "decided", "deferred", "clarifying", "preparing"}:
        from cayu.collaboration._planning_preflight import preflight_plan_control

        preflight_plan_control(next_record)
    change = (
        _bytes(next_record, ceiling, redactor)
        - _bytes(prior, ceiling, redactor)
        + _bytes(event, ceiling, redactor)
    )
    anchor = await store._anchor(tx, initialized, redactor)
    released_bytes = prior.reserved_bytes - next_record.reserved_bytes
    released_events = prior.reserved_events - next_record.reserved_events
    if (
        released_bytes < max(change, 0)
        or released_events < 1
        or anchor.reserved_bytes < prior.reserved_bytes
        or anchor.reserved_events < prior.reserved_events
    ):
        raise CollaborationUnavailable("Planning lacks its reserved transition capacity.")
    updated_anchor = prepare_contract(
        _Anchor,
        anchor.model_copy(
            update={
                "event_count": anchor.event_count + 1,
                "event_sequence": event.sequence,
                "retained_bytes": anchor.retained_bytes + change,
                "reserved_bytes": anchor.reserved_bytes - released_bytes,
                "reserved_events": anchor.reserved_events - released_events,
            }
        ),
        redactor=redactor,
    )
    require_capacity(updated_anchor, ordinary=False)
    await tx.put("request_plans", operation_key(command.operation), next_record, insert=False)
    await tx.put("request_plan_events", (event.sequence,), event, insert=True)
    await tx.put("anchors", (), updated_anchor, insert=False)
    return next_record


async def retain_decision_in_transaction(
    store: CollaborationStore,
    tx: _Repository,
    initialized: CollaborationInitialization,
    expected: RequestPlanningRequest,
    proposal: RequestPlanningDecision,
    *,
    redactor: SecretRedactor,
) -> RequestPlanningRecord:
    found = await read_plan_in_transaction(store, tx, initialized, expected, redactor=redactor)
    if not isinstance(found, ExactMatch):
        raise CollaborationConflict("Exact retained planning intent is unavailable.")
    prior = found.receipt
    proposal = prepare_contract(type(proposal), proposal, redactor=redactor)
    if prior.decision is not None:
        require_exact_contract(prior.decision, proposal, redactor=redactor)
        return prior
    if prior.state != "evaluating" or prior.reserved_events <= 1:
        raise CollaborationConflict("Planning no longer admits a policy decision.")
    await _require_current_input(store, tx, initialized, expected, redactor)
    await _require_current_participants(store, tx, initialized, expected, redactor)
    if isinstance(proposal, (RequestPlanningFresh, RequestPlanningFork)):
        from cayu.collaboration.planning import preparation_stage_count

        if (
            preparation_stage_count(proposal) > expected.limits.max_stages
            or len(proposal.resources) > expected.limits.max_resources
        ):
            raise CollaborationContractError("Preparation exceeds the exact request limits.")
    if (
        isinstance(proposal, RequestPlanningDefer)
        and proposal.prerequisite.deadline_at_ms > expected.deadline_at_ms
    ):
        raise CollaborationConflict("Deferral cannot extend its planning deadline.")
    # An absolute not-before may already be due when intent retention or
    # recovery finishes. Its finite deadline still fences admission; being
    # due only enables a separate explicit successor, never an internal loop.
    if isinstance(proposal, RequestPlanningClarify):
        opening = proposal.opening
        if (
            opening.expected != expected.expected
            or opening.expected_revision != expected.expected_revision + 1
            or opening.question.admission != expected.admission_operation
            or opening.question.input_revision != expected.expected_input_revision
            or opening.question.input_sha256 != expected.expected_input_sha256
            or opening.question.initiator != expected.initiator
            or opening.question.deadline_at_ms > expected.deadline_at_ms
        ):
            raise CollaborationConflict("Planning question differs from its exact input/admission.")
    anchor = await store._anchor(tx, initialized, redactor)
    event = _event(expected, anchor.event_sequence + 1, "plan_decided", redactor, content=proposal)
    updated = prior.model_copy(
        update={
            "revision": prior.revision + 1,
            "state": "decided",
            "decision_commitment": sha256(contract_bytes(proposal, redactor=redactor)).hexdigest(),
            "event_sequences": (*prior.event_sequences, event.sequence),
            "reserved_events": prior.reserved_events - 1,
            "reserved_bytes": prior.reserved_bytes - 2 * expected.limits.max_record_bytes,
        }
    )
    return await _write_transition(store, tx, initialized, prior, updated, event, redactor)


def admission_command(
    record: RequestPlanningRecord, redactor: SecretRedactor, *, prepared=None
) -> RequestAdmissionCommand:
    """Derive one native command from the retained decision, never from new input."""
    command = record.receipt.command
    if record.decision is None:
        raise CollaborationUnavailable("Planning has no retained decision.")
    return prepare_contract(
        RequestAdmissionCommand,
        RequestAdmissionCommand(
            operation=command.admission_operation,
            expected=command.expected,
            expected_revision=command.expected_revision,
            expected_input_revision=command.expected_input_revision,
            expected_input_sha256=command.expected_input_sha256,
            generation=command.admission_generation,
            decision=record.decision.decision,
            proposal_commitment=sha256(
                contract_bytes(record.decision, redactor=redactor)
            ).hexdigest(),
            evidence=(),
            initiator=command.initiator,
            prepared=record.decision.prepared
            if isinstance(record.decision, RequestPlanningContinue)
            else prepared,
        ),
        redactor=redactor,
    )


async def apply_local_decision_in_transaction(
    store: CollaborationStore,
    tx: _Repository,
    initialized: CollaborationInitialization,
    expected: RequestPlanningRequest,
    *,
    redactor: SecretRedactor,
) -> RequestPlanningRecord:
    """Compose a non-dispatching admission with its exact planning transition."""
    from cayu.collaboration._request_arbitration import admit_in_transaction

    found = await read_plan_in_transaction(store, tx, initialized, expected, redactor=redactor)
    if not isinstance(found, ExactMatch):
        raise CollaborationConflict("Exact planning intent is unavailable.")
    prior = found.receipt
    if (
        isinstance(prior.decision, (RequestPlanningFresh, RequestPlanningFork))
        and prior.state == "preparing"
        and not prior.pending_stages
    ):
        from cayu.collaboration._planning_creation_types import prepared_creation_from_stage
        from cayu.collaboration._planning_stages import admission_stage_intent, retain_stage

        await _require_current_input(store, tx, initialized, expected, redactor)
        await _require_current_participants(store, tx, initialized, expected, redactor)
        prepared = await prepared_creation_from_stage(tx, prior, redactor=redactor)
        if prepared is None:
            raise CollaborationConflict("Creation did not produce an admissible recipient.")
        await retain_stage(
            store,
            tx,
            initialized,
            expected,
            admission_stage_intent(prior, redactor, prepared=prepared),
            redactor=redactor,
        )
        updated = await read_plan_in_transaction(
            store, tx, initialized, expected, redactor=redactor
        )
        assert isinstance(updated, ExactMatch)
        return updated.receipt
    if prior.state in {"deferred", "declined", "clarifying", "preparing"}:
        return prior
    if prior.state != "decided" or prior.decision is None or prior.pending_stages:
        raise CollaborationConflict("Planning decision is not eligible for local admission.")
    parent, now = await _require_current_input(store, tx, initialized, expected, redactor)
    await _require_current_participants(store, tx, initialized, expected, redactor)
    if isinstance(prior.decision, RequestPlanningFresh):
        from cayu.collaboration._planning_creation import retain_creation_stage

        if prior.decision.resources:
            from cayu.collaboration._planning_resources import retain_resource_preparation

            return await retain_resource_preparation(
                store, tx, initialized, prior, redactor=redactor
            )
        return await retain_creation_stage(store, tx, initialized, prior, redactor=redactor)
    if isinstance(prior.decision, RequestPlanningFork):
        from cayu.collaboration._planning_fork import retain_view_stage

        return await retain_view_stage(store, tx, initialized, prior, redactor=redactor)
    command = admission_command(prior, redactor)
    if isinstance(prior.decision, RequestPlanningContinue):
        from cayu.collaboration._planning_recipient import retain_prepared_stage

        return await retain_prepared_stage(store, tx, initialized, prior, redactor=redactor)
    settlement = None
    if command.decision == "decline" and parent.admission != "undecided":
        from cayu.collaboration._permits import ReceivingSettlementReceipt
        from cayu.collaboration._planning_control import local_planning_quiescence

        if not await local_planning_quiescence(
            store, tx, initialized, parent, redactor=redactor, pending_decline=expected
        ):
            raise CollaborationUnavailable("Prior preparation must settle before declining.")
        settlement = ReceivingSettlementReceipt(
            expected=parent.permit,
            receiving_owner=initialized.owner,
            receipt_id="planning-decline-"
            + sha256(contract_bytes(command, redactor=redactor)).hexdigest(),
            outcome="quiescent",
        )
    due = expected.deadline_at_ms
    if isinstance(prior.decision, RequestPlanningDefer):
        prerequisite = prior.decision.prerequisite
        if now >= prerequisite.deadline_at_ms:
            raise CollaborationConflict("Deferral deadline has expired.")
        due = min(expected.deadline_at_ms, prerequisite.deadline_at_ms)
        if isinstance(prerequisite, RequestPlanningTimer):
            due = min(due, max(now, prerequisite.not_before_ms))
    await admit_in_transaction(
        store, tx, initialized, command, settlement=settlement, redactor=redactor
    )
    anchor = await store._anchor(tx, initialized, redactor)
    terminal = command.decision == "decline"
    event = _event(
        expected,
        anchor.event_sequence + 1,
        "plan_declined" if terminal else "plan_waiting",
        redactor,
        content=command,
    )
    updated = prior.model_copy(
        update={
            "revision": prior.revision + 1,
            "state": "declined"
            if terminal
            else (
                "clarifying" if isinstance(prior.decision, RequestPlanningClarify) else "deferred"
            ),
            "next_due_at_ms": due,
            "event_sequences": (*prior.event_sequences, event.sequence),
            "reserved_events": 0 if terminal else prior.reserved_events - 1,
            "reserved_bytes": 0
            if terminal
            else prior.reserved_bytes - 2 * expected.limits.max_record_bytes,
        }
    )
    result = await _write_transition(store, tx, initialized, prior, updated, event, redactor)
    if isinstance(prior.decision, RequestPlanningClarify):
        from cayu.collaboration._planning_stages import clarification_stage_intent, retain_stage

        await retain_stage(
            store,
            tx,
            initialized,
            expected,
            clarification_stage_intent(result, redactor),
            redactor=redactor,
        )
        found = await read_plan_in_transaction(store, tx, initialized, expected, redactor=redactor)
        assert isinstance(found, ExactMatch)
        result = found.receipt
    return result

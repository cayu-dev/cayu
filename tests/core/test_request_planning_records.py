"""Record/projection conformance; durable public planning is qualified separately."""

import pytest
from tests.core.test_collaboration_request_foundation import accept, setup
from tests.core.test_participant_identity import stores
from tests.core.test_request_planning_contracts import _policy

from cayu.collaboration._clarification_state import clarification_commitment
from cayu.collaboration._contracts import CollaborationContractError, ObjectRef
from cayu.collaboration._planning_records import (
    RequestPlanningCursor,
    RequestPlanningEvent,
    RequestPlanningReceipt,
    RequestPlanningRecord,
    RequestPlanningStageIntent,
    RequestPlanningStageRecord,
    planning_record_projection,
)
from cayu.collaboration._preparation import prepare_contract
from cayu.collaboration.memory import InMemoryCollaborationStore
from cayu.collaboration.planning import RequestPlanningRequest, planning_policy_commitment
from cayu.collaboration.requests import RequestAdmissionCommand
from cayu.vaults.redaction import SecretRedactor

REDACTOR = SecretRedactor()
__all__ = ["stores"]


@pytest.mark.anyio
@pytest.mark.parametrize("offset", [-1, 0, 1])
async def test_mandatory_control_record_room_below_at_and_above_boundary(offset):
    from cayu.collaboration._planning_preflight import (
        control_record_ceiling,
        preflight_plan_control,
    )
    from cayu.collaboration._preparation import contract_bytes

    _, record = await _records()
    for _ in range(4):
        required = control_record_ceiling(record)
        command = record.receipt.command.model_copy(
            update={
                "limits": record.receipt.command.limits.model_copy(
                    update={"max_record_bytes": required}
                )
            }
        )
        record = record.model_copy(
            update={
                "receipt": record.receipt.model_copy(
                    update={
                        "command": command,
                        "event": record.receipt.event.model_copy(
                            update={"commitment": clarification_commitment(command, REDACTOR)}
                        ),
                    }
                )
            }
        )
    required = control_record_ceiling(record)
    assert record.receipt.command.limits.max_record_bytes == required
    assert len(contract_bytes(record, redactor=REDACTOR)) < required - 1
    changed = record.receipt.command.model_copy(
        update={
            "limits": record.receipt.command.limits.model_copy(
                update={"max_record_bytes": required + offset}
            )
        }
    )
    record = record.model_copy(
        update={
            "receipt": record.receipt.model_copy(
                update={
                    "command": changed,
                    "event": record.receipt.event.model_copy(
                        update={"commitment": clarification_commitment(changed, REDACTOR)}
                    ),
                }
            )
        }
    )
    if offset < 0:
        with pytest.raises(CollaborationContractError, match="mandatory bounded cleanup"):
            preflight_plan_control(record)
    else:
        preflight_plan_control(record)


async def _records(store=None):
    store = InMemoryCollaborationStore() if store is None else store
    values = await setup(store)
    initial = values[1]
    accepted = await accept(store, values)
    expected = accepted.receipt.expected
    selected = expected.intent.selection
    policy = _policy()
    policy = policy.model_copy(
        update={"reference": policy.reference.model_copy(update={"owner": initial.owner})}
    )
    command = RequestPlanningRequest(
        operation=initial.operation("plan"),
        expected=expected,
        expected_revision=accepted.revision,
        expected_input_revision=0,
        expected_input_sha256=clarification_commitment(expected, REDACTOR),
        planning_generation=1,
        admission_operation=initial.operation("plan-admission"),
        admission_generation=1,
        initiator=expected.initiator.model_copy(
            update={
                "participant": ObjectRef(
                    owner=selected.recipient.reference.owner,
                    kind="participant",
                    object_id=selected.recipient.reference.participant_id,
                    incarnation=selected.recipient.reference.incarnation,
                )
            }
        ),
        policy=policy.reference,
        policy_sha256=planning_policy_commitment(policy, redactor=REDACTOR),
        limits=policy.limits,
        deadline_at_ms=selected.expires_at_ms,
        predecessor=None,
    )
    event = RequestPlanningEvent(
        id="plan-retained",
        sequence=1,
        operation=command.operation,
        plan=command.operation,
        request=selected.reference,
        type="plan_retained",
        commitment=clarification_commitment(command, REDACTOR),
        participants=accepted.receipt.event.participants,
    )
    receipt = RequestPlanningReceipt(
        command=command,
        policy=policy,
        retained_at_ms=selected.accepted_at_ms,
        event=event,
    )
    record = RequestPlanningRecord(
        receipt=receipt,
        revision=1,
        state="evaluating",
        decision_commitment=None,
        stage_count=0,
        pending_stages=0,
        next_due_at_ms=command.deadline_at_ms,
        event_sequences=(1,),
        reserved_bytes=65536,
        reserved_events=7,
    )
    return initial, record


@pytest.mark.anyio
async def test_plan_projection_and_full_receipt_identity():
    initial, record = await _records()
    command = record.receipt.command
    scope = initial.binding.application_scope
    key = (
        command.operation.namespace_incarnation,
        command.operation.generation,
        command.operation.caller_key,
    )
    checked, projection = planning_record_projection(
        "request_plans", record.model_dump(mode="json"), scope=scope, key=key
    )
    assert checked == record and checked is not record
    assert projection == (
        command.expected.intent.selection.reference.request_id,
        command.expected.intent.selection.reference.incarnation,
        command.expected.intent.selection.recipient.reference.participant_id,
        1,
        "evaluating",
        0,
        command.deadline_at_ms,
    )
    for invalid_scope, invalid_key in (("other", key), (scope, (*key[:2], "other"))):
        with pytest.raises(CollaborationContractError):
            planning_record_projection(
                "request_plans", record, scope=invalid_scope, key=invalid_key
            )
    for changes in (
        {"pending_stages": 1},
        {"revision": 2},
        {"event_sequences": (2,)},
        {"state": "deferred"},
        {"stage_count": True},
        {"pruned_stages": True},
        {"pruned_stages": -1},
        {"pruned_stages": 1},
        {"decision_commitment": "a" * 64},
    ):
        with pytest.raises(CollaborationContractError):
            prepare_contract(
                RequestPlanningRecord, record.model_copy(update=changes), redactor=REDACTOR
            )
    for changes in (
        {"policy_sha256": "f" * 64},
        {"policy": command.policy.model_copy(update={"revision": 2})},
        {"deadline_at_ms": record.receipt.retained_at_ms},
    ):
        with pytest.raises(CollaborationContractError):
            prepare_contract(
                RequestPlanningReceipt,
                record.receipt.model_copy(update={"command": command.model_copy(update=changes)}),
                redactor=REDACTOR,
            )
    assert (
        prepare_contract(RequestPlanningReceipt, record.receipt, redactor=REDACTOR)
        == record.receipt
    )


@pytest.mark.anyio
async def test_terminal_business_state_does_not_erase_pending_stage(stores):
    store = stores()
    initial, record = await _records(store)
    command = record.receipt.command
    admission = RequestAdmissionCommand(
        operation=command.admission_operation,
        expected=command.expected,
        expected_revision=command.expected_revision,
        expected_input_revision=command.expected_input_revision,
        expected_input_sha256=command.expected_input_sha256,
        generation=command.admission_generation,
        decision="defer",
        evidence=(),
        initiator=command.initiator,
    )
    intent = RequestPlanningStageIntent(
        operation=initial.operation("plan-stage-1"),
        plan=command.operation,
        plan_sha256=clarification_commitment(command, REDACTOR),
        ordinal=1,
        command=admission,
    )
    assert RequestPlanningStageIntent.model_validate_json(intent.model_dump_json()) == intent
    for mode in (None, True, "future_command", "request_resource_transfer"):
        document = intent.model_dump(mode="json")
        document["command"]["mode"] = mode
        with pytest.raises(CollaborationContractError):
            prepare_contract(RequestPlanningStageIntent, document, redactor=REDACTOR)
    document = intent.model_dump(mode="json")
    del document["command"]["mode"]
    with pytest.raises(CollaborationContractError):
        prepare_contract(RequestPlanningStageIntent, document, redactor=REDACTOR)
    event = record.receipt.event.model_copy(
        update={
            "id": "stage-retained",
            "sequence": 2,
            "operation": intent.operation,
            "type": "plan_stage_retained",
            "commitment": clarification_commitment(intent, REDACTOR),
        }
    )
    stage = RequestPlanningStageRecord(
        intent=intent,
        state="pending",
        registered_at_ms=record.receipt.retained_at_ms,
        settled_at_ms=None,
        receipt=None,
        registration_event=event,
        settlement_event=None,
        reserved_bytes=65536,
    )
    assert prepare_contract(RequestPlanningStageRecord, stage, redactor=REDACTOR) == stage
    terminal = record.model_copy(
        update={
            "state": "cancelled",
            "decision_commitment": clarification_commitment(
                record.receipt.policy.rules[0].proposal, REDACTOR
            ),
            "stage_count": 1,
            "pending_stages": 1,
            "revision": 2,
            "event_sequences": (1, 3),
        }
    )
    assert prepare_contract(RequestPlanningRecord, terminal, redactor=REDACTOR).pending_stages == 1
    with pytest.raises(CollaborationContractError):
        prepare_contract(
            RequestPlanningRecord,
            terminal.model_copy(update={"state": "superseded"}),
            redactor=REDACTOR,
        )
    with pytest.raises(CollaborationContractError):
        prepare_contract(
            RequestPlanningStageRecord,
            stage.model_copy(update={"state": "excluded"}),
            redactor=REDACTOR,
        )
    excluded = stage.model_copy(
        update={
            "state": "excluded",
            "reserved_bytes": 0,
            "settled_at_ms": stage.registered_at_ms,
            "settlement_event": event.model_copy(
                update={"id": "stage-excluded", "sequence": 4, "type": "plan_stage_excluded"}
            ),
        }
    )
    assert prepare_contract(RequestPlanningStageRecord, excluded, redactor=REDACTOR) == excluded
    key = (
        intent.operation.namespace_incarnation,
        intent.operation.generation,
        intent.operation.caller_key,
    )
    checked, projection = planning_record_projection(
        "request_plan_stages", excluded, scope=initial.binding.application_scope, key=key
    )
    assert checked == excluded
    assert projection[-1] == "excluded"
    scope = initial.binding.application_scope
    async with store._transaction(scope, write=True) as tx:
        await tx.put("request_plan_stages", key, stage, insert=True)
    reopened = stores()
    async with reopened._transaction(scope, write=False) as tx:
        rows = await tx.scan_request_plan_stages(command.operation, limit=1)
        assert len(rows) == 1
        assert prepare_contract(RequestPlanningStageRecord, rows[0], redactor=REDACTOR) == stage
        native = await tx.find_request_plan_stage(admission.operation)
        assert prepare_contract(RequestPlanningStageRecord, native, redactor=REDACTOR) == stage
        assert await tx.find_request_plan_stage(initial.operation("absent-stage")) is None
        assert await tx.scan_request_plan_stages(initial.operation("other-plan"), limit=1) == []
        for limit in (0, True, 130):
            with pytest.raises(CollaborationContractError):
                await tx.scan_request_plan_stages(command.operation, limit=limit)
        with pytest.raises(CollaborationContractError):
            await tx.find_request_plan_stage(
                admission.operation.model_copy(update={"application_scope": "other"})
            )
    async with reopened._transaction(scope, write=True) as tx:
        await tx.put("request_plan_stages", key, excluded, insert=False)
    async with store._transaction(scope, write=False) as tx:
        native = await tx.find_request_plan_stage(admission.operation)
        assert prepare_contract(RequestPlanningStageRecord, native, redactor=REDACTOR) == excluded


@pytest.mark.anyio
async def test_native_planning_record_roundtrip_and_pending_discovery(stores):
    store = stores()
    initial, record = await _records(store)
    scope = initial.binding.application_scope
    operation = record.receipt.command.operation
    key = operation.namespace_incarnation, operation.generation, operation.caller_key
    request = record.receipt.command.expected.intent.selection.reference
    async with store._transaction(scope, write=True) as tx:
        await tx.put("request_plans", key, record, insert=True)
        await tx.put(
            "request_plan_events",
            (record.receipt.event.sequence,),
            record.receipt.event,
            insert=True,
        )
    reopened = stores()
    async with reopened._transaction(scope, write=False) as tx:
        checked = prepare_contract(
            RequestPlanningRecord, await tx.get("request_plans", key), redactor=REDACTOR
        )
        assert checked == record
        pending = await tx.scan_pending_request_plans(after=None, limit=1)
        by_request = await tx.scan_request_plans(request, limit=1)
        assert pending == by_request
        assert len(pending) == 1
        assert (
            await tx.scan_pending_request_plans(
                after=RequestPlanningCursor(
                    next_due_at_ms=record.next_due_at_ms, operation=operation
                ),
                limit=1,
            )
            == []
        )
        for limit in (0, True, 33):
            with pytest.raises(CollaborationContractError):
                await tx.scan_pending_request_plans(after=None, limit=limit)
    cancelled = record.model_copy(
        update={
            "state": "cancelled",
            "decision_commitment": clarification_commitment(
                record.receipt.policy.rules[0].proposal, REDACTOR
            ),
            "stage_count": 1,
            "pending_stages": 1,
            "revision": 2,
            "event_sequences": (1, 2),
        }
    )
    async with reopened._transaction(scope, write=True) as tx:
        await tx.put("request_plans", key, cancelled, insert=False)
    async with store._transaction(scope, write=False) as tx:
        pending = await tx.scan_pending_request_plans(after=None, limit=1)
        assert prepare_contract(RequestPlanningRecord, pending[0], redactor=REDACTOR) == cancelled
    async with store._transaction(scope, write=True) as tx:
        await tx.put(
            "request_plans", key, cancelled.model_copy(update={"pending_stages": 0}), insert=False
        )
    async with reopened._transaction(scope, write=False) as tx:
        assert await tx.scan_pending_request_plans(after=None, limit=1) == []

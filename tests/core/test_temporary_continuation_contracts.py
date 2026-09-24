"""Temporary-service data contracts; not runtime admission qualification."""

import pytest
from tests.core.test_clarification_contracts import OWNER, fixture, operation
from tests.core.test_participant_identity import registration

from cayu.collaboration._contracts import CollaborationContractError, ObjectRef
from cayu.collaboration._permits import (
    PermitCommand,
    PermitIntent,
    PermitRegistration,
    ReceivingSettlementReceipt,
)
from cayu.collaboration._preparation import prepare_contract
from cayu.collaboration.waits import request_object_ref
from cayu.runtime._session_continuation import (
    ContinuationConflict,
    ContinuationNamespace,
    ContinuationTicket,
    continuation_digest,
    continuation_namespace_id,
)
from cayu.runtime._temporary_continuation import (
    TemporaryServiceAdmission,
    TemporaryServiceDispatch,
    TemporaryServiceExclusion,
    TemporaryServiceExecution,
    TemporaryServiceIntent,
    TemporaryServicePreparation,
    TemporaryServiceRecord,
    advance_temporary_service_record,
    temporary_service_invocation_id,
)
from cayu.vaults.redaction import SecretRedactor


def test_retained_record_collection_includes_source_and_target_children():
    from cayu.runtime._session_continuation import CONTINUATION_MAX_SERVICES
    from cayu.runtime._session_continuation_store import (
        MAX_RETAINED_CONTINUATION_RECORDS,
        MAX_RETAINED_TICKETS,
        collect_retained_record,
    )
    from cayu.runtime._temporary_service_target import MAX_TARGET_SERVICES

    keys = ["namespace"]
    for ticket in range(MAX_RETAINED_TICKETS):
        keys.append(f"ticket:{ticket}")
        keys.extend(f"service:{ticket}:{child}" for child in range(CONTINUATION_MAX_SERVICES))
    keys.extend(f"target:{child}" for child in range(MAX_TARGET_SERVICES))
    assert len(keys) == MAX_RETAINED_CONTINUATION_RECORDS
    records = {}
    for key in keys:
        collect_retained_record(records, key, {})
    assert len(records) > MAX_RETAINED_TICKETS + 1
    before = records.copy()
    with pytest.raises(ValueError, match="retention evidence"):
        collect_retained_record(records, "overflow", {})
    assert records == before
    with pytest.raises(ValueError, match="retention evidence"):
        collect_retained_record({}, "invalid", None)


def selection():
    question = fixture()[0].question
    namespace = ContinuationNamespace(
        session_id="waiting",
        session_instance_id="one",
        owner=OWNER,
        namespace_id=continuation_namespace_id("waiting", "one", OWNER),
    )
    ticket = ContinuationTicket(
        namespace=namespace,
        session_id="waiting",
        session_instance_id="one",
        owner=OWNER,
        registration_key="wait",
        targets=(request_object_ref(question.request),),
        predicate_kind="ALL_SETTLED",
        predicate_version=1,
        threshold=None,
        deadline="2030-01-01T00:00:00+00:00",
        failure_policy="return_and_report",
        service_policy="clarification",
        wait_edge_revision=1,
        interaction_id="parent",
        writer_generation=1,
        purpose="result",
        state="WAITING",
        revision=2,
    )
    return TemporaryServiceIntent(
        operation=operation("service"),
        initiator=question.initiator,
        ticket=ticket,
        question=question,
        service_generation=1,
        parent_service=None,
        depth=1,
        mode="same_session",
        target=ObjectRef(owner=OWNER, kind="session", object_id="waiting", incarnation="one"),
        participant_binding_sha256="1" * 64,
        execution_profile_sha256="2" * 64,
        resume_sha256="3" * 64,
        invocation_id=temporary_service_invocation_id(operation("service")),
        prepared_at_ms=1780000000000,
        budget_binding=question.budget_binding,
        budget_authority_sha256=question.budget_authority_sha256,
    )


@pytest.mark.parametrize(
    "field,value",
    (
        ("application_scope", "another-application"),
        ("namespace_incarnation", "another-namespace"),
        ("generation", 2),
        ("caller_key", "another-service"),
    ),
)
def test_service_invocation_identity_is_exact_and_operation_scoped(field, value):
    intent = selection()
    assert intent.invocation_id == temporary_service_invocation_id(intent.operation)
    other = intent.operation.model_copy(update={field: value})
    assert temporary_service_invocation_id(other) != intent.invocation_id
    for changed in (
        intent.model_copy(update={"invocation_id": "service-invocation"}),
        intent.model_copy(update={"operation": other}),
    ):
        with pytest.raises(CollaborationContractError):
            prepare_contract(TemporaryServiceIntent, changed, redactor=SecretRedactor())


def test_root_service_cannot_borrow_an_unrelated_wait():
    intent = selection()
    changed = intent.model_copy(
        update={"ticket": intent.ticket.model_copy(update={"targets": (intent.question.receiver,)})}
    )
    with pytest.raises(CollaborationContractError):
        prepare_contract(TemporaryServiceIntent, changed, redactor=SecretRedactor())


@pytest.mark.parametrize(
    "field,value",
    (
        ("interest_id", "different-interest"),
        ("attempt_generation", 2),
        ("target_run_epoch", 9),
        ("target_transcript_cursor", 9),
        ("withdrawal_generation", 2),
        ("deadline_at_ms", 191),
    ),
)
def test_service_commitment_retains_complete_receiving_dependency(field, value):
    from tests.core.test_clarification_deliveries import delivery_record

    intent = selection()
    peer = delivery_record().intent.append
    key = peer.append_key.model_copy(
        update={
            "target_session_id": intent.target.object_id,
            "target_session_instance_id": intent.target.incarnation,
        }
    )
    peer = peer.model_copy(
        update={
            "append_key": key,
            "attempt_key": peer.attempt_key.model_copy(update={"append_key": key}),
        }
    )
    intent = prepare_contract(
        TemporaryServiceIntent,
        intent.model_copy(update={"required_peer_append": peer}),
        redactor=SecretRedactor(),
    )
    assert TemporaryServiceIntent.model_validate_json(intent.model_dump_json()) == intent
    changed = intent.model_copy(
        update={
            "required_peer_append": peer.model_copy(
                update={"attempt_key": peer.attempt_key.model_copy(update={field: value})}
            )
        }
    )
    assert continuation_digest(changed) != continuation_digest(intent)


def test_nested_selection_leaves_root_membership_to_durable_lineage_owner():
    root = selection()
    child_question = root.question.model_copy(
        update={
            "operation": operation("child-question"),
            "parent_question": root.question.operation,
            "depth": 2,
            "request": root.question.request.model_copy(update={"request_id": "nested-request"}),
        }
    )
    nested = root.model_copy(
        update={
            "operation": operation("nested-service"),
            "invocation_id": temporary_service_invocation_id(operation("nested-service")),
            "question": child_question,
            "depth": 2,
            "parent_service": root.operation,
            "service_generation": 2,
            "ticket": root.ticket.model_copy(update={"state": "SERVICING", "revision": 3}),
        }
    )
    validated = prepare_contract(TemporaryServiceIntent, nested, redactor=SecretRedactor())
    assert request_object_ref(validated.question.request) not in validated.ticket.targets
    assert request_object_ref(root.question.request) in validated.ticket.targets


@pytest.mark.parametrize("state", ("prepared", "reserved", "admitted", "returned", "excluded"))
@pytest.mark.parametrize("mode", ("same_session", "side_session"))
@pytest.mark.parametrize("parent", (None, 32))
def test_compact_service_reference_reserves_all_maximum_representations(state, mode, parent):
    from cayu._validation import canonical_durable_json_bytes
    from cayu.runtime._session_continuation import (
        CONTINUATION_MAX_SERVICE_REFERENCE_BYTES,
        CONTINUATION_SERVICE_PREFIX,
        ContinuationServiceReference,
    )

    reference = ContinuationServiceReference(
        key=CONTINUATION_SERVICE_PREFIX + "f" * 64,
        record_sha256="f" * 64,
        generation=32,
        parent_generation=parent,
        state=state,
        mode=mode,
        expected_run_epoch=2**53 - 1,
        returned_writer_generation=2**53 - 1 if state == "returned" else None,
    )
    assert len(canonical_durable_json_bytes(reference.model_dump(mode="json"), "reference")) <= 352
    assert CONTINUATION_MAX_SERVICE_REFERENCE_BYTES >= 352


def admission():
    intent = selection()
    dispatch = TemporaryServiceDispatch(
        intent=intent, admission_payload_sha256="4" * 64, expected_run_epoch=2
    )
    permit_request = PermitRegistration(
        operation=operation("permit"),
        participant=intent.question.responder,
        expected_lifecycle_revision=1,
        expected_configuration_revision=1,
        admission_generation=1,
        admission_commitment=continuation_digest(dispatch),
        source_operation=intent.operation,
        target=intent.target,
        target_state="existing",
        effect_scope="clarification_service",
        required_settlement="quiescence",
        settlement_operation=operation("settle-permit"),
    )
    permit = PermitCommand(
        operation=permit_request.operation,
        source=OWNER,
        destination=OWNER,
        initiator=intent.initiator,
        intent=PermitIntent(request=permit_request, limits=registration().bootstrap.limits),
    )
    return TemporaryServiceAdmission(
        dispatch=dispatch,
        permit=permit,
        permit_receipt_sha256="5" * 64,
        admission_command_sha256="7" * 64,
    )


@pytest.mark.parametrize(
    "field,value", [("expected_run_epoch", 3), ("admission_payload_sha256", "a" * 64)]
)
def test_temporary_permit_binds_complete_dispatch(field, value):
    original = admission()
    assert (
        prepare_contract(TemporaryServiceAdmission, original, redactor=SecretRedactor()) == original
    )
    changed = original.model_copy(
        update={"dispatch": original.dispatch.model_copy(update={field: value})}
    )
    with pytest.raises(CollaborationContractError):
        prepare_contract(TemporaryServiceAdmission, changed, redactor=SecretRedactor())


def test_temporary_permit_binds_durable_preparation_timestamp():
    original = admission()
    intent = original.dispatch.intent
    changed = original.model_copy(
        update={
            "dispatch": original.dispatch.model_copy(
                update={
                    "intent": intent.model_copy(
                        update={"prepared_at_ms": intent.prepared_at_ms + 1}
                    )
                }
            )
        }
    )
    with pytest.raises(CollaborationContractError):
        prepare_contract(TemporaryServiceAdmission, changed, redactor=SecretRedactor())


def test_preparation_reconstructs_without_fabricated_registration_evidence():
    original = admission()
    preparation = original.preparation
    assert (
        TemporaryServicePreparation.model_validate_json(preparation.model_dump_json())
        == preparation
    )
    assert set(preparation.model_dump()) == {"dispatch", "permit"}
    with pytest.raises(ValueError):
        TemporaryServiceAdmission.model_validate(preparation.model_dump())


@pytest.mark.parametrize("change", ("quiescence_only", "wrong_permit"))
def test_prepared_exclusion_requires_exact_positive_fence(change):
    preparation = admission().preparation
    receipt = ReceivingSettlementReceipt(
        expected=preparation.permit,
        receiving_owner=preparation.dispatch.intent.target.owner,
        receipt_id="native-exclusion",
        outcome="quiescent",
        admission_excluded=True,
    )
    exclusion = TemporaryServiceExclusion(preparation=preparation, receipt=receipt)
    assert TemporaryServiceExclusion.model_validate_json(exclusion.model_dump_json()) == exclusion
    if change == "quiescence_only":
        receipt = receipt.model_copy(update={"admission_excluded": False})
    else:
        receipt = receipt.model_copy(
            update={
                "expected": preparation.permit.model_copy(update={"operation": operation("other")})
            }
        )
    with pytest.raises(ValueError):
        TemporaryServiceExclusion.model_validate(
            {"preparation": preparation.model_dump(), "receipt": receipt.model_dump()}
        )


@pytest.mark.parametrize("timestamp", (1, 253402300799999))
def test_service_timestamp_reconstructs_at_datetime_bounds(timestamp):
    intent = prepare_contract(
        TemporaryServiceIntent,
        selection().model_copy(update={"prepared_at_ms": timestamp}),
        redactor=SecretRedactor(),
    )
    reconstructed = TemporaryServiceIntent.model_validate_json(intent.model_dump_json())
    assert reconstructed.prepared_at == intent.prepared_at
    assert reconstructed.prepared_at_ms == timestamp


@pytest.mark.parametrize(
    "field,value",
    [
        ("service_generation", True),
        ("service_generation", 33),
        ("depth", 5),
        ("mode", "side_session"),
        ("budget_authority_sha256", "a" * 64),
        ("schema_version", True),
        ("prepared_at_ms", True),
        ("prepared_at_ms", 0),
        ("prepared_at_ms", 253402300800000),
    ],
)
def test_temporary_selection_rejects_changed_authority_or_bounds(field, value):
    with pytest.raises(CollaborationContractError):
        prepare_contract(
            TemporaryServiceIntent,
            selection().model_copy(update={field: value}),
            redactor=SecretRedactor(),
        )


def test_explicit_side_session_preserves_original_ticket():
    original = selection()
    selected = prepare_contract(
        TemporaryServiceIntent,
        original.model_copy(
            update={
                "mode": "side_session",
                "target": original.target.model_copy(update={"object_id": "side"}),
            }
        ),
        redactor=SecretRedactor(),
    )
    assert selected.ticket == original.ticket
    assert selected.target.object_id == "side"
    assert selected.budget_binding == original.budget_binding


def service_records():
    prepared = admission()
    reserved = TemporaryServiceRecord(admission=prepared, state="reserved")
    execution = TemporaryServiceExecution(
        receipt_id="admit:waiting:one:3",
        receipt_sha256="6" * 64,
        admission_command_sha256=prepared.admission_command_sha256,
        session_id="waiting",
        session_instance_id="one",
        invocation_id=prepared.dispatch.intent.invocation_id,
        run_epoch=3,
    )
    admitted = TemporaryServiceRecord(admission=prepared, state="admitted", execution=execution)
    settled = ReceivingSettlementReceipt(
        expected=prepared.permit,
        receiving_owner=OWNER,
        receipt_id="release:waiting:one:3",
        outcome="quiescent",
    )
    returned = TemporaryServiceRecord(
        admission=prepared,
        state="returned",
        execution=execution,
        settlement=settled,
        returned_writer_generation=4,
        released_session_status="completed",
    )
    return reserved, admitted, returned


def test_service_reconstruction_and_lost_acknowledgement_transition():
    reserved, admitted, returned = service_records()
    for value in (reserved, admitted, returned):
        reconstructed = TemporaryServiceRecord.model_validate_json(value.model_dump_json())
        assert reconstructed == value
        assert advance_temporary_service_record(value, reconstructed) == value
    assert advance_temporary_service_record(reserved, admitted) == admitted
    assert advance_temporary_service_record(admitted, returned) == returned
    assert advance_temporary_service_record(reserved, returned) == returned
    with pytest.raises(ContinuationConflict):
        advance_temporary_service_record(returned, admitted)


def test_released_status_contract_tracks_supported_session_states():
    from typing import get_args

    from cayu.runtime._temporary_continuation import ServiceReleasedSessionStatus
    from cayu.sessions.base import SessionStatus

    assert set(get_args(ServiceReleasedSessionStatus)) == {status.value for status in SessionStatus}


@pytest.mark.parametrize("status", [None, "unknown", True, 1])
def test_return_report_requires_positive_typed_status(status):
    _, admitted, returned = service_records()
    with pytest.raises(CollaborationContractError):
        prepare_contract(
            TemporaryServiceRecord,
            returned.model_copy(update={"released_session_status": status}),
            redactor=SecretRedactor(),
        )
    with pytest.raises(CollaborationContractError):
        prepare_contract(
            TemporaryServiceRecord,
            admitted.model_copy(update={"released_session_status": "completed"}),
            redactor=SecretRedactor(),
        )


def test_service_exclusion_requires_permanent_admission_fence():
    reserved, admitted, returned = service_records()
    assert returned.settlement is not None
    bad = reserved.model_copy(update={"state": "excluded", "settlement": returned.settlement})
    with pytest.raises(CollaborationContractError):
        advance_temporary_service_record(reserved, bad)
    excluded = bad.model_copy(
        update={"settlement": returned.settlement.model_copy(update={"admission_excluded": True})}
    )
    assert advance_temporary_service_record(reserved, excluded) == excluded
    with pytest.raises(ContinuationConflict):
        advance_temporary_service_record(admitted, excluded)
    with pytest.raises(ContinuationConflict):
        advance_temporary_service_record(excluded, admitted)


@pytest.mark.parametrize(
    "field,value",
    [
        ("session_id", "different"),
        ("session_instance_id", "different"),
        ("invocation_id", "different"),
        ("run_epoch", 4),
        ("run_epoch", True),
        ("admission_command_sha256", "a" * 64),
    ],
)
def test_service_execution_must_match_exact_dispatch(field, value):
    reserved, admitted, _ = service_records()
    assert admitted.execution is not None
    changed = admitted.model_copy(
        update={"execution": admitted.execution.model_copy(update={field: value})}
    )
    with pytest.raises(CollaborationContractError):
        advance_temporary_service_record(reserved, changed)


def test_service_cannot_replace_previously_observed_admission_receipt():
    _, admitted, returned = service_records()
    assert returned.execution is not None
    changed = returned.model_copy(
        update={"execution": returned.execution.model_copy(update={"receipt_sha256": "b" * 64})}
    )
    with pytest.raises(ContinuationConflict):
        advance_temporary_service_record(admitted, changed)


def test_service_rejects_reused_public_session_id_with_wrong_incarnation():
    original = selection()
    changed = original.model_copy(
        update={
            "mode": "side_session",
            "target": original.target.model_copy(update={"incarnation": "replacement"}),
        }
    )
    with pytest.raises(CollaborationContractError):
        prepare_contract(TemporaryServiceIntent, changed, redactor=SecretRedactor())

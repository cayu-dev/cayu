"""Strict lifecycle values; owner/store acceptance is qualified separately."""

import pytest

from cayu.collaboration._contracts import (
    CollaborationContractError,
    InitiatorBinding,
    ObjectRef,
    OwnerRef,
)
from cayu.collaboration._permits import (
    PermitCommand,
    PermitIntent,
    PermitRegistration,
    PermitSnapshot,
    ReceivingSettlementReceipt,
)
from cayu.collaboration._preparation import prepare_contract
from cayu.collaboration.lifecycle import (
    LifecycleCommand,
    LifecycleIntent,
    NamespaceInspection,
    NamespacePrune,
    NamespaceRef,
    NamespaceRetire,
    NamespaceRetirementEvidence,
    NamespaceRotate,
    NamespaceSeal,
    NamespaceSnapshot,
    ParticipantLifecycleChange,
)
from cayu.collaboration.participants import CollaborationLimits, ParticipantRef
from cayu.vaults.redaction import SecretRedactor


def authority(generation=1):
    return NamespaceRef(
        owner=OwnerRef(application_scope="app", owner_id="owner", incarnation="owner-inc"),
        namespace_incarnation="namespace-inc",
        generation=generation,
    )


def command(request):
    owner = authority().owner
    return LifecycleCommand(
        operation=request.operation,
        kind=request.kind,
        source=owner,
        destination=owner,
        initiator=InitiatorBinding(
            issuer=owner,
            principal="operator",
            participant=None,
            mandate=None,
            invocation_id=None,
            interaction_id=None,
        ),
        intent=LifecycleIntent(
            request=request,
            limits=CollaborationLimits(
                participants=64,
                aliases=64,
                operations=256,
                events=256,
                retained_bytes=4 * 1024 * 1024,
                control_operations=16,
                control_events=16,
                control_bytes=512 * 1024,
                namespaces=1,
                generations=8,
                obligations=32,
            ),
        ),
    )


@pytest.mark.parametrize("schema", [NamespaceSeal, NamespaceRotate])
def test_namespace_election_binds_exact_generation(schema):
    namespace = authority()
    request = schema(
        operation=namespace.operation("control"), namespace=namespace, expected_revision=1
    )
    expected = command(request)
    assert prepare_contract(LifecycleCommand, expected, redactor=SecretRedactor()) == expected
    with pytest.raises(ValueError, match="exact generation"):
        command(request.model_copy(update={"operation": authority(2).operation("control")}))


@pytest.mark.parametrize("target", [1, 2, 3])
def test_retirement_requires_contiguous_floor_and_later_control(target):
    request = NamespaceRetire(
        operation=authority(4).operation("retire"),
        namespace=authority(target),
        expected_revision=2,
        expected_retired_through=target - 1,
    )
    assert command(request).intent.request == request
    with pytest.raises(ValueError, match="later control"):
        command(request.model_copy(update={"operation": authority(target).operation("retire")}))
    with pytest.raises(ValueError, match="skip"):
        command(request.model_copy(update={"expected_retired_through": target}))


@pytest.mark.parametrize("value", [True, False, 0, -1, 33, "1", 1.0])
def test_prune_batch_rejects_invalid_bounds(value):
    with pytest.raises(ValueError):
        NamespacePrune(
            operation=authority(2).operation("prune"),
            namespace=authority(),
            expected_retention_revision=1,
            max_records=value,
        )


@pytest.mark.parametrize("state", ["active", "draining", "disabled", "retired"])
def test_lifecycle_controls_are_structural_but_equal_names_are_untrusted(state):
    request = ParticipantLifecycleChange(
        operation=authority().operation("lifecycle"),
        participant=ParticipantRef(
            owner=authority().owner, participant_id="participant", incarnation="inc"
        ),
        expected_lifecycle_revision=1,
        state=state,
    )
    expected = command(request)
    raw = expected.model_dump(mode="json")
    assert prepare_contract(LifecycleCommand, raw, redactor=SecretRedactor(state)) == expected
    raw["initiator"]["principal"] = state
    with pytest.raises(CollaborationContractError):
        prepare_contract(LifecycleCommand, raw, redactor=SecretRedactor(state))


def test_retired_namespace_requires_settlement_and_consistent_floors():
    with pytest.raises(ValueError, match="unsettled"):
        NamespaceSnapshot(
            reference=authority(), revision=2, state="retired", outstanding_obligations=1
        )
    current = NamespaceSnapshot(
        reference=authority(3), revision=1, state="open", outstanding_obligations=0
    )
    assert NamespaceInspection(
        current=current,
        retired_through=2,
        pruned_through=1,
        retention_revision=1,
        retained_generations=2,
    )
    with pytest.raises(ValueError, match="frontiers"):
        NamespaceInspection(
            current=current,
            retired_through=1,
            pruned_through=2,
            retention_revision=1,
            retained_generations=2,
        )
    assert NamespaceRetirementEvidence(
        namespace=authority(), retired_through=2, pruned_through=1, content="pruned"
    )
    with pytest.raises(ValueError, match="cover"):
        NamespaceRetirementEvidence(
            namespace=authority(2), retired_through=2, pruned_through=1, content="pruned"
        )


def permit_command():
    namespace = authority()
    control = command(
        NamespaceRotate(
            operation=namespace.operation("rotate"), namespace=namespace, expected_revision=1
        )
    )
    request = PermitRegistration(
        operation=namespace.operation("register"),
        participant=ParticipantRef(
            owner=namespace.owner, participant_id="participant", incarnation="inc"
        ),
        expected_lifecycle_revision=1,
        admission_generation=1,
        source_operation=namespace.operation("source"),
        target=ObjectRef(
            owner=namespace.owner, kind="fixture", object_id="target", incarnation="target-inc"
        ),
        target_state="future",
        effect_scope="execute",
        required_settlement="quiescence",
        settlement_operation=namespace.operation("settle"),
    )
    return PermitCommand(
        operation=request.operation,
        source=control.source,
        destination=control.destination,
        initiator=control.initiator,
        intent=PermitIntent(request=request, limits=control.intent.limits),
    )


@pytest.mark.parametrize("change", ["key", "generation", "scope"])
def test_permit_reserves_distinct_same_generation_settlement(change):
    request = permit_command().intent.request
    raw = request.model_dump(mode="json")
    if change == "key":
        raw["settlement_operation"] = raw["operation"]
    elif change == "generation":
        raw["settlement_operation"]["generation"] = 2
    else:
        raw["settlement_operation"]["application_scope"] = "other"
    with pytest.raises(CollaborationContractError):
        prepare_contract(PermitRegistration, raw, redactor=SecretRedactor())


def test_permit_settlement_is_exact_and_quiescence_is_positive():
    expected = permit_command()
    receipt = ReceivingSettlementReceipt(
        expected=expected,
        receiving_owner=authority().owner,
        receipt_id="receipt",
        outcome="quiescent",
    )
    assert PermitSnapshot(expected=expected, position=1, state="settled", settlement=receipt)
    with pytest.raises(ValueError, match="quiescence"):
        ReceivingSettlementReceipt(
            expected=expected,
            receiving_owner=authority().owner,
            receipt_id="receipt",
            outcome="excluded",
        )
    with pytest.raises(ValueError, match="state"):
        PermitSnapshot(expected=expected, position=1, state="settled", settlement=None)
    changed = expected.model_copy(
        update={"initiator": expected.initiator.model_copy(update={"principal": "other"})}
    )
    with pytest.raises(ValueError, match="another permit"):
        PermitSnapshot(expected=changed, position=1, state="settled", settlement=receipt)


@pytest.mark.parametrize("invalid", [1, 0, "true", None])
def test_permit_exclusion_proof_requires_a_boolean(invalid):
    with pytest.raises(ValueError):
        ReceivingSettlementReceipt(
            expected=permit_command(),
            receiving_owner=authority().owner,
            receipt_id="receipt",
            outcome="quiescent",
            admission_excluded=invalid,
        )

"""Pure planning values; these tests do not qualify durable orchestration."""

import warnings

import pytest

from cayu.collaboration._contracts import CollaborationContractError, ObjectRef, OwnerRef
from cayu.collaboration._preparation import contract_bytes, prepare_contract
from cayu.collaboration.exports import SessionExportRequest
from cayu.collaboration.memory import InMemoryCollaborationStore
from cayu.collaboration.planning import (
    ConfiguredRequestPlanningPolicy,
    RequestPlanningClarify,
    RequestPlanningDecline,
    RequestPlanningDefer,
    RequestPlanningLimits,
    RequestPlanningRule,
    RequestPlanningTimer,
    evaluate_configured_request_policy,
    planning_policy_commitment,
)
from cayu.vaults.redaction import SecretRedactor


def test_planning_schema_fences_unqualified_older_writers():
    from cayu.storage import migrations

    revision = migrations.revision(107)
    assert revision.compatible_from == 107
    with pytest.raises(migrations.SchemaTooNew):
        migrations.validate(
            migrations.SchemaState(revision=107, compatible_from=107),
            app_latest=106,
            app_min_supported=106,
        )
    with pytest.raises(migrations.SchemaTooOld):
        migrations.validate(
            migrations.SchemaState(revision=106, compatible_from=106),
            app_latest=107,
            app_min_supported=107,
        )


def test_planning_public_exports_share_the_native_types():
    import cayu
    import cayu.collaboration as collaboration

    for name in (
        "ConfiguredRequestPlanningPolicy",
        "RequestPlanningLimits",
        "RequestPlanningTimer",
        "RequestPlanningPrerequisite",
        "RequestPlanningDefer",
        "RequestPlanningDecline",
        "RequestPlanningClarify",
        "RequestPlanningContinue",
        "RequestPlanningFresh",
        "RequestPlanningFork",
        "RequestPlanningResource",
        "ForkRecipientPreparation",
        "ForkRecipientCreationPreparation",
        "ResourceRecipientCreationPreparation",
        "ResourceMaterialReference",
        "ForkRecipientAdmissionTarget",
        "FreshRecipientPreparation",
        "RequestPlanningRule",
        "RequestPlanningPredecessor",
        "RequestPlanningRequest",
        "RequestPlanningControl",
        "RequestPlanningAdmissionReader",
        "RequestPlanningEvent",
        "RequestPlanningReceipt",
        "RequestPlanningSuccessor",
        "RequestPlanningRecord",
        "RequestPlanningCursor",
        "RequestPlanningPage",
        "planning_policy_commitment",
    ):
        assert name in cayu.__all__ and name in collaboration.__all__
        assert getattr(cayu, name) is getattr(collaboration, name)
    assert cayu.RequestPlanningDefer is RequestPlanningDefer


def _policy():
    return ConfiguredRequestPlanningPolicy(
        reference=ObjectRef(
            owner=OwnerRef(application_scope="app", owner_id="planner", incarnation="one"),
            kind="request_planning_policy",
            object_id="configured",
            incarnation="one",
            revision=1,
        ),
        limits=RequestPlanningLimits(
            max_generations=32,
            max_stages=128,
            max_resources=32,
            max_record_bytes=65536,
            max_recovery_items=32,
        ),
        rules=(
            RequestPlanningRule(
                input_revision=0,
                proposal=RequestPlanningDefer(
                    prerequisite=RequestPlanningTimer(not_before_ms=100, deadline_at_ms=200)
                ),
            ),
        ),
        default=RequestPlanningDecline(reason="not_supported"),
    )


def test_deterministic_policy_reconstructs_and_returns_detached_proposals():
    policy = _policy()
    redactor = SecretRedactor()
    reconstructed = prepare_contract(
        ConfiguredRequestPlanningPolicy, policy.model_dump(mode="json"), redactor=redactor
    )
    for revision, expected in ((0, policy.rules[0].proposal), (1, policy.default)):
        first = evaluate_configured_request_policy(
            policy, input_revision=revision, redactor=redactor
        )
        second = evaluate_configured_request_policy(
            reconstructed, input_revision=revision, redactor=redactor
        )
        assert first == second == expected
        assert first is not expected and first is not second
    assert planning_policy_commitment(policy, redactor=redactor) == planning_policy_commitment(
        reconstructed, redactor=redactor
    )


@pytest.mark.parametrize(
    ("field", "maximum"),
    [
        ("max_generations", 32),
        ("max_stages", 128),
        ("max_resources", 32),
        ("max_record_bytes", 65536),
        ("max_recovery_items", 32),
    ],
)
@pytest.mark.parametrize("offset", [-1, 0, 1])
def test_planning_limits_below_at_above_ceiling(field, maximum, offset):
    value = _policy().limits.model_copy(update={field: maximum + offset})
    if offset > 0:
        with pytest.raises(CollaborationContractError):
            prepare_contract(RequestPlanningLimits, value, redactor=SecretRedactor())
    else:
        assert (
            getattr(
                prepare_contract(RequestPlanningLimits, value, redactor=SecretRedactor()), field
            )
            == maximum + offset
        )


@pytest.mark.parametrize("field", list(RequestPlanningLimits.model_fields))
def test_planning_limits_reject_boolean_integers(field):
    with pytest.raises(CollaborationContractError):
        prepare_contract(
            RequestPlanningLimits,
            _policy().limits.model_copy(update={field: True}),
            redactor=SecretRedactor(),
        )


@pytest.mark.parametrize("revision", [-1, True, 1.0, "1", 2**53])
def test_evaluation_requires_exact_bounded_revision(revision):
    with pytest.raises(CollaborationContractError):
        evaluate_configured_request_policy(
            _policy(), input_revision=revision, redactor=SecretRedactor()
        )


@pytest.mark.parametrize("revisions", [(0, 0), (1, 0)])
def test_duplicate_or_unordered_rules_are_not_silently_normalized(revisions):
    policy = _policy()
    rules = tuple(policy.rules[0].model_copy(update={"input_revision": r}) for r in revisions)
    with pytest.raises(CollaborationContractError):
        prepare_contract(
            ConfiguredRequestPlanningPolicy,
            policy.model_copy(update={"rules": rules}),
            redactor=SecretRedactor(),
        )


@pytest.mark.parametrize(
    "updates", [{"schema_version": 2}, {"schema_version": True}, {"algorithm": "future"}]
)
def test_unknown_policy_algorithm_and_schema_fail_closed(updates):
    with pytest.raises(CollaborationContractError):
        evaluate_configured_request_policy(
            _policy().model_copy(update=updates), input_revision=0, redactor=SecretRedactor()
        )


def test_nested_decision_and_limits_are_part_of_policy_commitment():
    policy = _policy()
    redactor = SecretRedactor()
    original = planning_policy_commitment(policy, redactor=redactor)
    changed_timer = RequestPlanningDefer(
        prerequisite=RequestPlanningTimer(not_before_ms=100, deadline_at_ms=201)
    )
    for changed in (
        policy.model_copy(update={"limits": policy.limits.model_copy(update={"max_stages": 127})}),
        policy.model_copy(
            update={"rules": (policy.rules[0].model_copy(update={"proposal": changed_timer}),)}
        ),
        policy.model_copy(update={"default": RequestPlanningDecline(reason="changed")}),
    ):
        assert planning_policy_commitment(changed, redactor=redactor) != original


@pytest.mark.parametrize("offset", [-1, 0, 1])
def test_policy_respects_narrower_record_byte_ceiling(offset):
    redactor = SecretRedactor()
    policy = _policy()
    # The bound is itself serialized. Find its stable decimal width first.
    for _ in range(3):
        size = len(contract_bytes(policy, redactor=redactor))
        policy = policy.model_copy(
            update={"limits": policy.limits.model_copy(update={"max_record_bytes": size})}
        )
    size = len(contract_bytes(policy, redactor=redactor))
    assert policy.limits.max_record_bytes == size
    policy = policy.model_copy(
        update={"limits": policy.limits.model_copy(update={"max_record_bytes": size + offset})}
    )
    assert len(contract_bytes(policy, redactor=redactor)) == size
    if offset < 0:
        with pytest.raises(CollaborationContractError):
            evaluate_configured_request_policy(policy, input_revision=0, redactor=redactor)
        with pytest.raises(CollaborationContractError):
            planning_policy_commitment(policy, redactor=redactor)
    else:
        assert evaluate_configured_request_policy(policy, input_revision=0, redactor=redactor)
        assert len(planning_policy_commitment(policy, redactor=redactor)) == 64


def test_mutated_policy_rejection_does_not_render_hostile_values(caplog, capsys):
    class Hostile:
        def __repr__(self):
            raise AssertionError("secret-canary")

        def __str__(self):
            raise AssertionError("secret-canary")

    malformed = _policy().model_copy(update={"default": Hostile()})
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        with pytest.raises(CollaborationContractError) as caught:
            evaluate_configured_request_policy(
                malformed, input_revision=0, redactor=SecretRedactor()
            )
    assert caught.value.__context__ is None
    assert caught.value.__cause__ is None
    assert not caught_warnings
    assert "secret-canary" not in str(caught.value) + caplog.text
    captured = capsys.readouterr()
    assert "secret-canary" not in captured.out + captured.err


@pytest.mark.anyio
async def test_clarification_proposal_preserves_native_export_identity():
    from tests.core.test_clarification_transactions import preparation

    _, opening = await preparation(InMemoryCollaborationStore())
    question = opening.question
    audience = ObjectRef(
        owner=question.responder.owner,
        kind="participant",
        object_id=question.responder.participant_id,
        incarnation=question.responder.incarnation,
    )
    # The existing transaction fixture is not an authenticated export. Align
    # its pure source representation here; public receiver coverage is separate.
    opening = opening.model_copy(
        update={
            "question": question.model_copy(
                update={"source": question.source.model_copy(update={"audience": audience})}
            )
        }
    )
    source = SessionExportRequest(
        ref=question.source.export,
        source_indices=(0,),
        source_selection=question.source.selection,
        audience=OwnerRef(
            application_scope=question.responder.owner.application_scope,
            owner_id=question.responder.participant_id,
            incarnation=question.responder.incarnation,
        ),
        projector=question.source.projector,
        policy=question.source.policy,
    )
    proposal = prepare_contract(
        RequestPlanningClarify,
        {"opening": opening, "source": source},
        redactor=SecretRedactor(),
    )
    assert proposal.opening == opening and proposal.source == source
    for changed in (
        source.model_copy(
            update={
                "source_selection": "whole_records"
                if source.source_selection == "assistant_visible_text_v1"
                else "assistant_visible_text_v1"
            }
        ),
        source.model_copy(
            update={"audience": source.audience.model_copy(update={"incarnation": "other"})}
        ),
        source.model_copy(
            update={"policy": source.policy.model_copy(update={"object_id": "other"})}
        ),
        source.model_copy(
            update={"projector": source.projector.model_copy(update={"object_id": "other"})}
        ),
    ):
        with pytest.raises(CollaborationContractError):
            prepare_contract(
                RequestPlanningClarify,
                {"opening": opening, "source": changed},
                redactor=SecretRedactor(),
            )

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from cayu._validation import canonical_durable_json_bytes
from cayu.agent_snapshots import (
    AgentSnapshot,
    AgentSnapshotCompleteness,
    AgentSnapshotComponentKind,
    AgentSnapshotComponentRef,
    AgentSnapshotConsistency,
    AgentSnapshotExecutionProfileComponent,
    AgentSnapshotExecutionProfileRef,
    AgentSnapshotLearningDisposition,
    AgentSnapshotLogicalRef,
    AgentSnapshotMaterialization,
    AgentSnapshotMaterializationCapability,
    AgentSnapshotMaterializationRequest,
    AgentSnapshotMaterializedComponent,
    AgentSnapshotOverlayKind,
    AgentSnapshotOverlayRef,
    AgentSnapshotRedaction,
    AgentSnapshotResultBinding,
    AgentSnapshotSubject,
    AgentSnapshotTerminalDisposition,
    AgentSnapshotTrialBinding,
    AgentSnapshotTrialStateMode,
    MemoryStateRef,
)
from cayu.memory import AutomaticRecallMode, AutomaticRecallPolicy
from cayu.memory_attribution import (
    MemoryAttribution,
    MemoryAttributionStatus,
    MemoryAttributionUnavailableReason,
    MemoryEvidenceAlias,
    MemoryRecallAttribution,
)
from cayu.memory_interventions import (
    MEMORY_INTERVENTION_MAX_BYTES,
    MEMORY_INTERVENTION_MAX_CHANGED_ITEMS,
    MEMORY_INTERVENTION_MAX_EFFECT_RECEIPTS,
    MemoryInterventionBounds,
    MemoryInterventionChangeKind,
    MemoryInterventionComparability,
    MemoryInterventionComparabilityStatus,
    MemoryInterventionEffectReceiptRef,
    MemoryInterventionEffectStatus,
    MemoryInterventionFixtureRef,
    MemoryInterventionItemChange,
    MemoryInterventionItemIdentity,
    MemoryInterventionItemIdentityKind,
    MemoryInterventionKind,
    MemoryInterventionMismatchReason,
    MemoryInterventionOperation,
    MemoryInterventionReceipt,
    MemoryInterventionSpec,
    MemoryInterventionTrialBinding,
    MemoryNegativeControlKind,
    memory_attribution_fingerprint,
    memory_intervention_from_json,
    memory_intervention_to_json,
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _content_fingerprint(value: object, field_name: str) -> str:
    return hashlib.sha256(canonical_durable_json_bytes(value, field_name)).hexdigest()


def _ref(value: str, *, scope: str | None = None) -> AgentSnapshotLogicalRef:
    return AgentSnapshotLogicalRef(
        fingerprint=_digest(value),
        revision=f"revision:{value}",
        scope_fingerprint=scope,
    )


def _component(
    kind: AgentSnapshotComponentKind,
    logical: AgentSnapshotLogicalRef,
    *,
    materialization: AgentSnapshotMaterializationCapability,
) -> AgentSnapshotComponentRef:
    return AgentSnapshotComponentRef(
        kind=kind,
        provider_id=f"test.{kind.value}.v1",
        logical=logical,
        consistency=AgentSnapshotConsistency.FRONTIER_CONSISTENT,
        completeness=AgentSnapshotCompleteness.COMPLETE,
        redaction=AgentSnapshotRedaction.BOUNDED_PROJECTION,
        materialization=materialization,
    )


def _snapshot(recall_policy: AutomaticRecallPolicy | None = None) -> AgentSnapshot:
    scope = _digest("authority")
    body = _ref("body")
    recall_policy = recall_policy or _policy()
    profile = AgentSnapshotExecutionProfileRef(
        schema_version=5,
        fingerprint=_digest("profile"),
        components=(
            AgentSnapshotExecutionProfileComponent(
                name="automatic_recall",
                fingerprint=_digest("profile:recall"),
                availability="available",
            ),
        ),
    )
    memory = MemoryStateRef.create(
        knowledge=_ref("knowledge", scope=scope),
        recall_policy=AgentSnapshotLogicalRef(
            fingerprint=recall_policy.fingerprint(),
            revision="revision:recall-policy",
            scope_fingerprint=scope,
        ),
        learning_disposition=AgentSnapshotLearningDisposition.ISOLATED,
    )
    return AgentSnapshot.create(
        capture_request_id="capture",
        captured_at=datetime(2026, 8, 24, tzinfo=UTC),
        subject=AgentSnapshotSubject(
            agent_id="agent",
            application_id="application",
            project_id="project",
            body_release=body,
        ),
        authority_scope_fingerprint=scope,
        execution_profile=profile,
        memory_state=memory,
        components=(
            _component(
                AgentSnapshotComponentKind.BODY,
                body,
                materialization=AgentSnapshotMaterializationCapability.REFERENCE_ONLY,
            ),
            _component(
                AgentSnapshotComponentKind.EXECUTION_PROFILE,
                _ref("profile"),
                materialization=AgentSnapshotMaterializationCapability.REFERENCE_ONLY,
            ),
            _component(
                AgentSnapshotComponentKind.MEMORY,
                AgentSnapshotLogicalRef(
                    fingerprint=memory.fingerprint,
                    revision="memory:1",
                    scope_fingerprint=scope,
                ),
                materialization=AgentSnapshotMaterializationCapability.RESTORABLE,
            ),
        ),
    )


def _policy(mode: AutomaticRecallMode = AutomaticRecallMode.STRONG_MATCHES):
    return AutomaticRecallPolicy(
        calibration_version="calibration-v1",
        fusion_strategy_version="fusion-v1",
        fusion_configuration_version="configuration-v1",
        mode=mode,
        minimum_inject_score=0.8,
        minimum_offer_score=0.5,
    )


def _spec(kind: MemoryInterventionKind) -> MemoryInterventionSpec:
    starting = _policy()
    snapshot = _snapshot(starting)
    trial = (
        starting.model_copy(update={"mode": AutomaticRecallMode.OFF})
        if kind is MemoryInterventionKind.AUTOMATIC_RECALL_OFF
        else starting
    )
    return MemoryInterventionSpec.create(
        spec_id=f"spec-{kind.value}",
        snapshot=snapshot,
        starting_recall_policy=starting,
        trial_recall_policy=trial,
        trial_state_mode=AgentSnapshotTrialStateMode.RESET_EACH_TRIAL,
        kind=kind,
        bounds=MemoryInterventionBounds(
            max_changed_items=0,
            max_fixture_bytes=0,
        ),
    )


def _omit_spec(*, item_count: int = 1) -> MemoryInterventionSpec:
    policy = _policy()
    snapshot = _snapshot(policy)
    changes = tuple(
        MemoryInterventionItemChange(
            kind=MemoryInterventionChangeKind.OMIT,
            source_item=MemoryInterventionItemIdentity(
                kind=MemoryInterventionItemIdentityKind.FINGERPRINT,
                revision_fingerprint=_digest(f"omitted-revision-{index}"),
                item_fingerprint=_digest(f"omitted-item-{index}"),
            ),
        )
        for index in range(item_count)
    )
    return MemoryInterventionSpec.create(
        spec_id="spec-omit",
        snapshot=snapshot,
        starting_recall_policy=policy,
        trial_recall_policy=policy,
        trial_state_mode=AgentSnapshotTrialStateMode.RESET_EACH_TRIAL,
        kind=MemoryInterventionKind.OMIT_ITEMS,
        bounds=MemoryInterventionBounds(max_changed_items=item_count, max_fixture_bytes=0),
        changes=changes,
        proposer_fingerprint=_digest("proposer"),
        source_fingerprint=_digest("source"),
        reason="measure omission sensitivity",
    )


def _materialization_and_trial(
    spec: MemoryInterventionSpec,
    candidate_id: str,
    *,
    memory_baseline_fingerprint: str | None = None,
    materialization_trial_id: str = "trial-1",
    binding_trial_id: str | None = None,
) -> tuple[AgentSnapshotMaterialization, AgentSnapshotTrialBinding]:
    snapshot = _snapshot()
    memory_baseline = memory_baseline_fingerprint or spec.memory_state_fingerprint
    request = AgentSnapshotMaterializationRequest(
        snapshot_fingerprint=snapshot.fingerprint,
        candidate_id=candidate_id,
        trial_id=materialization_trial_id,
        state_mode=spec.trial_state_mode,
    )
    overlay = AgentSnapshotOverlayRef.create(
        kind=AgentSnapshotOverlayKind.MEMORY,
        overlay_id=f"overlay-{candidate_id}",
        baseline_fingerprint=memory_baseline,
        candidate_id=candidate_id,
        state_scope_id=request.state_scope_id,
    )
    materialization = AgentSnapshotMaterialization.create(
        progress_id=_digest(f"progress:{candidate_id}"),
        request=request,
        created_at=datetime(2026, 8, 24, tzinfo=UTC),
        components=(
            AgentSnapshotMaterializedComponent(
                kind=AgentSnapshotComponentKind.MEMORY,
                baseline_fingerprint=memory_baseline,
                capability=AgentSnapshotMaterializationCapability.RESTORABLE,
                overlay=overlay,
            ),
        ),
    )
    trial = AgentSnapshotTrialBinding.create(
        materialization=materialization,
        case_id="case-1",
        trial_id=binding_trial_id or materialization_trial_id,
        evaluator_fingerprint=_digest("evaluator"),
        created_at=datetime(2026, 8, 24, tzinfo=UTC),
    )
    return materialization, trial


def _trial_with_updates(
    trial: AgentSnapshotTrialBinding,
    **updates: object,
) -> AgentSnapshotTrialBinding:
    values = trial.model_dump(mode="python", exclude={"fingerprint"})
    values.update(updates)
    provisional = AgentSnapshotTrialBinding.model_construct(fingerprint="0" * 64, **values)
    return AgentSnapshotTrialBinding(
        fingerprint=_content_fingerprint(provisional.identity_material(), "snapshot_trial"),
        **values,
    )


def _operation_with_updates(
    operation: MemoryInterventionOperation,
    **updates: object,
) -> MemoryInterventionOperation:
    values = operation.model_dump(mode="python", exclude={"fingerprint"})
    values.update(updates)
    provisional = MemoryInterventionOperation.model_construct(fingerprint="0" * 64, **values)
    return MemoryInterventionOperation(
        fingerprint=_content_fingerprint(
            provisional.identity_material(),
            MemoryInterventionOperation.__name__,
        ),
        **values,
    )


def _complete_empty_attribution() -> MemoryAttribution:
    return MemoryAttribution(
        status=MemoryAttributionStatus.COMPLETE,
        truncated=False,
        observed_receipt_count=0,
        observed_exposure_count=0,
        observed_item_count=0,
        omitted_receipt_count_at_least=0,
        omitted_exposure_count_at_least=0,
        omitted_item_count_at_least=0,
    )


def _complete_recall_attribution() -> MemoryAttribution:
    receipt = MemoryRecallAttribution(
        receipt_alias=MemoryEvidenceAlias(
            key_id="memory-alias-key",
            kind="receipt",
            digest=_digest("strict-receipt"),
        ),
        interaction_alias=MemoryEvidenceAlias(
            key_id="memory-alias-key",
            kind="interaction",
            digest=_digest("strict-interaction"),
        ),
        projection_ordinal=0,
        model_step_id="strict-model-step",
        created_at=datetime(2026, 8, 24, tzinfo=UTC),
        inspected_count=0,
        eligible_count=0,
        admitted_count=0,
        offered_count=0,
        silent_count=0,
        omitted_count=0,
        complete_source_count=1,
        partial_source_count=0,
        unavailable_source_count=0,
        failed_source_count=0,
        truncated=False,
        omitted_item_count_at_least=0,
    )
    return MemoryAttribution(
        status=MemoryAttributionStatus.COMPLETE,
        truncated=False,
        observed_receipt_count=1,
        observed_exposure_count=0,
        observed_item_count=0,
        omitted_receipt_count_at_least=0,
        omitted_exposure_count_at_least=0,
        omitted_item_count_at_least=0,
        receipts=(receipt,),
    )


def _unavailable_attribution() -> MemoryAttribution:
    return MemoryAttribution(
        status=MemoryAttributionStatus.UNAVAILABLE,
        truncated=False,
        reason=MemoryAttributionUnavailableReason.STORE_UNSUPPORTED,
        observed_receipt_count=0,
        observed_exposure_count=0,
        observed_item_count=0,
        omitted_receipt_count_at_least=0,
        omitted_exposure_count_at_least=0,
        omitted_item_count_at_least=0,
    )


def _binding(
    spec: MemoryInterventionSpec,
    candidate_id: str,
    *,
    attribution: MemoryAttribution | None = None,
) -> MemoryInterventionTrialBinding:
    materialization, trial = _materialization_and_trial(spec, candidate_id)
    operation = MemoryInterventionOperation.create(
        spec=spec,
        materialization=materialization,
        trial=trial,
    )
    if spec.kind is MemoryInterventionKind.AS_DECLARED:
        status = MemoryInterventionEffectStatus.VERIFIED_NO_CHANGE
        result_memory = spec.memory_state_fingerprint
        effects: tuple[str, ...] = ()
        changed_revisions: tuple[str, ...] = ()
        matched_item_count = 0
        effect_receipts: tuple[MemoryInterventionEffectReceiptRef, ...] = ()
    else:
        status = MemoryInterventionEffectStatus.APPLIED
        result_memory = _digest(f"result-memory:{candidate_id}")
        effects = (_digest(f"effect:{candidate_id}"),)
        changed_revisions = tuple(
            change.source_item.revision_fingerprint
            for change in spec.changes
            if change.source_item is not None
        )
        matched_item_count = len(changed_revisions)
        effect_receipts = (
            MemoryInterventionEffectReceiptRef(
                owner_id="test.intervention-executor",
                receipt_fingerprint=_digest(f"effect-receipt:{candidate_id}"),
                effect_fingerprint=effects[0],
            ),
        )
    receipt = MemoryInterventionReceipt.create(
        spec=spec,
        operation=operation,
        status=status,
        result_memory_state_fingerprint=result_memory,
        result_recall_policy_fingerprint=spec.trial_recall_policy_fingerprint,
        matched_item_count=matched_item_count,
        changed_item_revision_fingerprints=changed_revisions,
        effect_fingerprints=effects,
        application_effect_receipts=effect_receipts,
    )
    attribution = attribution or _complete_empty_attribution()
    result = AgentSnapshotResultBinding.create(
        trial=trial,
        session_id=f"session-{candidate_id}",
        terminal_disposition=AgentSnapshotTerminalDisposition.COMPLETED,
        runtime_evidence_fingerprint=_digest(f"runtime:{candidate_id}"),
        eval_result_revision=_digest(f"eval:{candidate_id}"),
        memory_evidence_fingerprint=memory_attribution_fingerprint(attribution),
        recorded_at=datetime(2026, 8, 24, tzinfo=UTC),
    )
    return MemoryInterventionTrialBinding.create(
        spec=spec,
        operation=operation,
        receipt=receipt,
        trial=trial,
        result=result,
        attribution=attribution,
    )


def test_specs_make_baseline_and_recall_off_explicit_and_deterministic() -> None:
    baseline = _spec(MemoryInterventionKind.AS_DECLARED)
    recall_off = _spec(MemoryInterventionKind.AUTOMATIC_RECALL_OFF)

    assert baseline.kind is MemoryInterventionKind.AS_DECLARED
    assert baseline.starting_recall_policy_fingerprint == baseline.trial_recall_policy_fingerprint
    assert recall_off.trial_recall_mode is AutomaticRecallMode.OFF
    assert (
        recall_off.starting_recall_policy_fingerprint != recall_off.trial_recall_policy_fingerprint
    )
    assert _spec(MemoryInterventionKind.AS_DECLARED).fingerprint == baseline.fingerprint


def test_spec_rejects_starting_policy_from_another_snapshot_frontier() -> None:
    snapshot_policy = _policy()
    foreign_policy = snapshot_policy.model_copy(update={"minimum_inject_score": 0.9})
    snapshot = _snapshot(snapshot_policy)

    with pytest.raises(ValidationError, match="Starting recall policy"):
        MemoryInterventionSpec.create(
            spec_id="foreign-starting-policy",
            snapshot=snapshot,
            starting_recall_policy=foreign_policy,
            trial_recall_policy=foreign_policy,
            trial_state_mode=AgentSnapshotTrialStateMode.RESET_EACH_TRIAL,
            kind=MemoryInterventionKind.AS_DECLARED,
            bounds=MemoryInterventionBounds(max_changed_items=0, max_fixture_bytes=0),
        )

    document = _spec(MemoryInterventionKind.AS_DECLARED).model_dump(mode="json")
    document["starting_recall_policy_fingerprint"] = foreign_policy.fingerprint()
    document["trial_recall_policy_fingerprint"] = foreign_policy.fingerprint()
    identity = {key: value for key, value in document.items() if key != "fingerprint"}
    document["fingerprint"] = _content_fingerprint(identity, MemoryInterventionSpec.__name__)

    with pytest.raises(ValidationError, match="Starting recall policy"):
        memory_intervention_from_json(json.dumps(document))


def test_recall_off_rejects_off_to_off_policy_drift_before_comparison() -> None:
    starting = _policy(AutomaticRecallMode.OFF)
    snapshot = _snapshot(starting)
    different_off = starting.model_copy(update={"calibration_version": "other-off-policy"})
    bounds = MemoryInterventionBounds(max_changed_items=0, max_fixture_bytes=0)
    baseline = MemoryInterventionSpec.create(
        spec_id="baseline-off",
        snapshot=snapshot,
        starting_recall_policy=starting,
        trial_recall_policy=starting,
        trial_state_mode=AgentSnapshotTrialStateMode.RESET_EACH_TRIAL,
        kind=MemoryInterventionKind.AS_DECLARED,
        bounds=bounds,
    )

    assert baseline.trial_recall_mode is AutomaticRecallMode.OFF
    with pytest.raises(ValidationError, match="starting policy must not already be OFF"):
        MemoryInterventionSpec.create(
            spec_id="invalid-recall-off",
            snapshot=snapshot,
            starting_recall_policy=starting,
            trial_recall_policy=different_off,
            trial_state_mode=AgentSnapshotTrialStateMode.RESET_EACH_TRIAL,
            kind=MemoryInterventionKind.AUTOMATIC_RECALL_OFF,
            bounds=bounds,
        )


def test_item_intervention_is_revision_bound_bounded_and_has_no_text_or_location_field() -> None:
    policy = _policy()
    snapshot = _snapshot(policy)
    source = MemoryInterventionItemIdentity(
        kind=MemoryInterventionItemIdentityKind.ALIAS,
        revision_fingerprint=_digest("item-revision"),
        alias=MemoryEvidenceAlias(
            key_id="memory-alias-key",
            kind="item",
            digest=_digest("item-alias"),
        ),
    )
    fixture = MemoryInterventionFixtureRef(
        fixture_id="replacement-fixture",
        fixture_fingerprint=_digest("fixture"),
        representation_fingerprint=_digest("fixture-representation"),
        size_bytes=512,
    )
    change = MemoryInterventionItemChange(
        kind=MemoryInterventionChangeKind.REPLACE,
        source_item=source,
        fixture=fixture,
    )
    spec = MemoryInterventionSpec.create(
        spec_id="replace-one",
        snapshot=snapshot,
        starting_recall_policy=policy,
        trial_recall_policy=policy,
        trial_state_mode=AgentSnapshotTrialStateMode.RESET_EACH_TRIAL,
        kind=MemoryInterventionKind.REPLACE_ITEMS,
        bounds=MemoryInterventionBounds(max_changed_items=1, max_fixture_bytes=512),
        changes=(change,),
        proposer_fingerprint=_digest("proposer"),
        source_fingerprint=_digest("source"),
        reason="correct a stale evaluation fixture",
    )

    document = spec.model_dump(mode="json")
    assert spec.changes == (change,)
    assert "text" not in json.dumps(document)
    assert "location" not in json.dumps(document)
    with pytest.raises(ValidationError, match="Mixed or mismatched"):
        MemoryInterventionSpec.create(
            spec_id="mixed",
            snapshot=snapshot,
            starting_recall_policy=policy,
            trial_recall_policy=policy,
            trial_state_mode=AgentSnapshotTrialStateMode.RESET_EACH_TRIAL,
            kind=MemoryInterventionKind.OMIT_ITEMS,
            bounds=MemoryInterventionBounds(max_changed_items=1, max_fixture_bytes=512),
            changes=(change,),
            proposer_fingerprint=_digest("proposer"),
            source_fingerprint=_digest("source"),
            reason="mixed declaration",
        )


def test_unknown_intervention_kind_and_version_fail_closed() -> None:
    spec = _spec(MemoryInterventionKind.AS_DECLARED)
    document = json.loads(memory_intervention_to_json(spec))
    document["kind"] = "future_kind"
    with pytest.raises(ValidationError):
        MemoryInterventionSpec.model_validate(document)

    document = json.loads(memory_intervention_to_json(spec))
    document["schema_version"] = 2
    with pytest.raises(ValueError, match="unsupported|Unsupported"):
        memory_intervention_from_json(json.dumps(document))


def test_operation_precommits_exact_materialization_overlay_scope_and_trial() -> None:
    spec = _spec(MemoryInterventionKind.AUTOMATIC_RECALL_OFF)
    materialization, trial = _materialization_and_trial(spec, "candidate-off")
    operation = MemoryInterventionOperation.create(
        spec=spec,
        materialization=materialization,
        trial=trial,
    )

    assert operation.materialization_fingerprint == materialization.fingerprint
    assert operation.memory_overlay_fingerprint == trial.memory_overlay_fingerprint
    assert operation.state_scope_id == materialization.state_scope_id
    assert operation.trial_binding_fingerprint == trial.fingerprint

    _, wrong_trial = _materialization_and_trial(spec, "other-candidate")
    with pytest.raises(ValueError, match="does not match"):
        MemoryInterventionOperation.create(
            spec=spec,
            materialization=materialization,
            trial=wrong_trial,
        )


def test_receipt_is_replay_stable_and_keeps_no_match_distinct_from_uncertainty() -> None:
    spec = _omit_spec()
    materialization, trial = _materialization_and_trial(spec, "candidate-omit")
    operation = MemoryInterventionOperation.create(
        spec=spec,
        materialization=materialization,
        trial=trial,
    )
    arguments = {
        "spec": spec,
        "operation": operation,
        "status": MemoryInterventionEffectStatus.MATCHED_NO_ITEMS,
        "result_memory_state_fingerprint": spec.memory_state_fingerprint,
        "result_recall_policy_fingerprint": spec.trial_recall_policy_fingerprint,
    }
    first = MemoryInterventionReceipt.create(**arguments)
    replay = MemoryInterventionReceipt.create(**arguments)
    indeterminate = MemoryInterventionReceipt.create(
        spec=spec,
        operation=operation,
        status=MemoryInterventionEffectStatus.INDETERMINATE,
    )

    assert replay == first
    assert replay.fingerprint == first.fingerprint
    assert indeterminate.fingerprint != first.fingerprint
    assert indeterminate.status is MemoryInterventionEffectStatus.INDETERMINATE


def test_trial_binding_reuses_result_memory_fingerprint_and_attribution_status() -> None:
    binding = _binding(_spec(MemoryInterventionKind.AS_DECLARED), "baseline")

    assert binding.result.memory_evidence_fingerprint == binding.attribution_fingerprint
    assert binding.attribution.status is MemoryAttributionStatus.COMPLETE
    assert binding.proves_no_memory_exposure is True
    assert binding.spec.kind is MemoryInterventionKind.AS_DECLARED

    mismatched_result = AgentSnapshotResultBinding.create(
        trial=binding.trial,
        session_id="session-mismatched",
        terminal_disposition=AgentSnapshotTerminalDisposition.COMPLETED,
        runtime_evidence_fingerprint=_digest("runtime-mismatched"),
        eval_result_revision=_digest("eval-mismatched"),
        memory_evidence_fingerprint=_digest("other-attribution"),
        recorded_at=datetime(2026, 8, 24, tzinfo=UTC),
    )
    with pytest.raises(ValidationError, match="does not bind"):
        MemoryInterventionTrialBinding.create(
            spec=binding.spec,
            operation=binding.operation,
            receipt=binding.receipt,
            trial=binding.trial,
            result=mismatched_result,
            attribution=binding.attribution,
        )


def test_no_exposure_is_not_recall_off_missing_or_no_match() -> None:
    baseline = _binding(_spec(MemoryInterventionKind.AS_DECLARED), "baseline")
    recall_off = _binding(_spec(MemoryInterventionKind.AUTOMATIC_RECALL_OFF), "off")
    unavailable = _binding(
        _spec(MemoryInterventionKind.AS_DECLARED),
        "unavailable",
        attribution=_unavailable_attribution(),
    )

    assert baseline.proves_no_memory_exposure is True
    assert recall_off.proves_no_memory_exposure is True
    assert baseline.spec.kind is not recall_off.spec.kind
    assert unavailable.proves_no_memory_exposure is False
    assert unavailable.attribution.status is MemoryAttributionStatus.UNAVAILABLE


def test_memory_comparability_is_closed_and_requires_isolated_overlays() -> None:
    baseline = _binding(_spec(MemoryInterventionKind.AS_DECLARED), "baseline")
    intervention = _binding(_spec(MemoryInterventionKind.AUTOMATIC_RECALL_OFF), "off")
    comparison = MemoryInterventionComparability.create(
        baseline=baseline,
        intervention=intervention,
    )

    assert comparison.status is MemoryInterventionComparabilityStatus.COMPARABLE
    assert comparison.mismatch_reasons == ()
    assert comparison.generic_experiment_comparability_required is True

    contaminated = _binding(
        _spec(MemoryInterventionKind.AUTOMATIC_RECALL_OFF),
        "baseline",
    )
    comparison = MemoryInterventionComparability.create(
        baseline=baseline,
        intervention=contaminated,
    )
    assert comparison.status is MemoryInterventionComparabilityStatus.INCOMPARABLE
    assert MemoryInterventionMismatchReason.MATERIALIZATION_OVERLAY in comparison.mismatch_reasons


def test_memory_comparability_rejects_partial_item_application() -> None:
    baseline = _binding(_spec(MemoryInterventionKind.AS_DECLARED), "partial-baseline")
    fully_applied = _binding(_omit_spec(item_count=2), "partial-intervention")
    changed_revision = fully_applied.spec.changes[0].source_item
    assert changed_revision is not None
    effect_fingerprint = _digest("partial-item-effect")
    partial_receipt = MemoryInterventionReceipt.create(
        spec=fully_applied.spec,
        operation=fully_applied.operation,
        status=MemoryInterventionEffectStatus.APPLIED,
        result_memory_state_fingerprint=_digest("partial-item-result"),
        result_recall_policy_fingerprint=(fully_applied.spec.trial_recall_policy_fingerprint),
        matched_item_count=1,
        changed_item_revision_fingerprints=(changed_revision.revision_fingerprint,),
        effect_fingerprints=(effect_fingerprint,),
        application_effect_receipts=(
            MemoryInterventionEffectReceiptRef(
                owner_id="test.intervention-executor",
                receipt_fingerprint=_digest("partial-item-effect-receipt"),
                effect_fingerprint=effect_fingerprint,
            ),
        ),
    )
    partial = MemoryInterventionTrialBinding.create(
        spec=fully_applied.spec,
        operation=fully_applied.operation,
        receipt=partial_receipt,
        trial=fully_applied.trial,
        result=fully_applied.result,
        attribution=fully_applied.attribution,
    )

    comparison = MemoryInterventionComparability.create(
        baseline=baseline,
        intervention=partial,
    )

    assert comparison.status is MemoryInterventionComparabilityStatus.INCOMPARABLE
    assert comparison.mismatch_reasons == (MemoryInterventionMismatchReason.CHANGED_ITEM_REVISIONS,)


def test_memory_comparability_keeps_matched_no_items_distinct() -> None:
    baseline = _binding(_spec(MemoryInterventionKind.AS_DECLARED), "no-match-baseline")
    applied = _binding(_omit_spec(), "no-match-intervention")
    no_match_receipt = MemoryInterventionReceipt.create(
        spec=applied.spec,
        operation=applied.operation,
        status=MemoryInterventionEffectStatus.MATCHED_NO_ITEMS,
        result_memory_state_fingerprint=applied.spec.memory_state_fingerprint,
        result_recall_policy_fingerprint=applied.spec.trial_recall_policy_fingerprint,
    )
    no_match = MemoryInterventionTrialBinding.create(
        spec=applied.spec,
        operation=applied.operation,
        receipt=no_match_receipt,
        trial=applied.trial,
        result=applied.result,
        attribution=applied.attribution,
    )

    comparison = MemoryInterventionComparability.create(
        baseline=baseline,
        intervention=no_match,
    )

    assert no_match.receipt.status is MemoryInterventionEffectStatus.MATCHED_NO_ITEMS
    assert comparison.status is MemoryInterventionComparabilityStatus.INCOMPARABLE
    assert comparison.mismatch_reasons == (MemoryInterventionMismatchReason.CHANGED_ITEM_REVISIONS,)


def test_comparability_does_not_treat_unavailable_attribution_as_empty() -> None:
    baseline = _binding(_spec(MemoryInterventionKind.AS_DECLARED), "baseline")
    intervention = _binding(
        _spec(MemoryInterventionKind.AUTOMATIC_RECALL_OFF),
        "off",
        attribution=_unavailable_attribution(),
    )
    comparison = MemoryInterventionComparability.create(
        baseline=baseline,
        intervention=intervention,
    )

    assert comparison.status is MemoryInterventionComparabilityStatus.INCOMPARABLE
    assert comparison.mismatch_reasons == (
        MemoryInterventionMismatchReason.REQUIRED_ATTRIBUTION_AVAILABILITY,
    )


def test_all_intervention_records_have_strict_deterministic_json_round_trips() -> None:
    baseline = _binding(_spec(MemoryInterventionKind.AS_DECLARED), "baseline")
    intervention = _binding(_spec(MemoryInterventionKind.AUTOMATIC_RECALL_OFF), "off")
    comparison = MemoryInterventionComparability.create(
        baseline=baseline,
        intervention=intervention,
    )
    records = (
        intervention.spec,
        intervention.operation,
        intervention.receipt,
        intervention,
        comparison,
    )

    for record in records:
        encoded = memory_intervention_to_json(record)
        assert memory_intervention_from_json(encoded) == record
        assert memory_intervention_to_json(memory_intervention_from_json(encoded)) == encoded

    document = json.loads(memory_intervention_to_json(intervention.spec))
    document["unexpected"] = True
    with pytest.raises(ValidationError):
        memory_intervention_from_json(json.dumps(document))


def test_negative_control_is_an_explicit_content_addressed_fixture_intervention() -> None:
    snapshot = _snapshot()
    policy = _policy()
    change = MemoryInterventionItemChange(
        kind=MemoryInterventionChangeKind.INJECT_NEGATIVE_CONTROL,
        fixture=MemoryInterventionFixtureRef(
            fixture_id="adversarial-control",
            fixture_fingerprint=_digest("negative-control-fixture"),
            representation_fingerprint=_digest("negative-control-representation"),
            size_bytes=128,
        ),
    )
    spec = MemoryInterventionSpec.create(
        spec_id="negative-control",
        snapshot=snapshot,
        starting_recall_policy=policy,
        trial_recall_policy=policy,
        trial_state_mode=AgentSnapshotTrialStateMode.RESET_EACH_TRIAL,
        kind=MemoryInterventionKind.NEGATIVE_CONTROL,
        bounds=MemoryInterventionBounds(max_changed_items=1, max_fixture_bytes=128),
        changes=(change,),
        negative_control_kind=MemoryNegativeControlKind.ADVERSARIAL,
        proposer_fingerprint=_digest("proposer"),
        source_fingerprint=_digest("fixture-source"),
        reason="measure instruction susceptibility",
    )

    assert spec.negative_control_kind is MemoryNegativeControlKind.ADVERSARIAL
    assert spec.changes[0].source_item is None
    assert spec.changes[0].fixture == change.fixture


def test_json_ingress_rejects_oversize_duplicate_keys_and_invalid_unicode() -> None:
    spec = _spec(MemoryInterventionKind.AS_DECLARED)
    encoded = memory_intervention_to_json(spec)
    duplicate_key = encoded.replace(
        '"record_type":',
        '"record_type":"cayu.memory-intervention-spec","record_type":',
        1,
    )

    with pytest.raises(ValueError, match="duplicate|Duplicate"):
        memory_intervention_from_json(duplicate_key)
    with pytest.raises(ValueError, match="byte limit"):
        memory_intervention_from_json(b" " * (MEMORY_INTERVENTION_MAX_BYTES + 1))
    with pytest.raises(ValueError, match="Unicode"):
        memory_intervention_from_json("\ud800")


def test_operation_rejects_a_memory_overlay_from_another_baseline() -> None:
    spec = _spec(MemoryInterventionKind.AUTOMATIC_RECALL_OFF)
    materialization, trial = _materialization_and_trial(
        spec,
        "foreign-baseline",
        memory_baseline_fingerprint=_digest("foreign-memory-frontier"),
    )

    with pytest.raises(ValueError, match="memory baseline"):
        MemoryInterventionOperation.create(
            spec=spec,
            materialization=materialization,
            trial=trial,
        )


def test_operation_rejects_a_reset_scope_derived_for_another_trial() -> None:
    spec = _spec(MemoryInterventionKind.AUTOMATIC_RECALL_OFF)
    materialization, trial = _materialization_and_trial(
        spec,
        "wrong-reset-scope",
        materialization_trial_id="scope-owner",
        binding_trial_id="bound-trial",
    )

    with pytest.raises(ValueError, match="trial state scope"):
        MemoryInterventionOperation.create(
            spec=spec,
            materialization=materialization,
            trial=trial,
        )


@pytest.mark.parametrize(
    ("field_name", "foreign_value"),
    (
        ("snapshot_fingerprint", _digest("foreign-trial-snapshot")),
        ("memory_overlay_fingerprint", _digest("foreign-trial-overlay")),
    ),
)
def test_trial_binding_rejects_foreign_trial_lineage(
    field_name: str,
    foreign_value: str,
) -> None:
    valid = _binding(_spec(MemoryInterventionKind.AUTOMATIC_RECALL_OFF), field_name)
    trial = _trial_with_updates(valid.trial, **{field_name: foreign_value})
    operation = _operation_with_updates(
        valid.operation,
        trial_binding_fingerprint=trial.fingerprint,
    )
    effect_fingerprint = _digest(f"effect:{field_name}")
    receipt = MemoryInterventionReceipt.create(
        spec=valid.spec,
        operation=operation,
        status=MemoryInterventionEffectStatus.APPLIED,
        result_memory_state_fingerprint=_digest(f"result:{field_name}"),
        result_recall_policy_fingerprint=valid.spec.trial_recall_policy_fingerprint,
        effect_fingerprints=(effect_fingerprint,),
        application_effect_receipts=(
            MemoryInterventionEffectReceiptRef(
                owner_id="test.intervention-executor",
                receipt_fingerprint=_digest(f"effect-receipt:{field_name}"),
                effect_fingerprint=effect_fingerprint,
            ),
        ),
    )
    result = AgentSnapshotResultBinding.create(
        trial=trial,
        session_id=f"session-{field_name}",
        terminal_disposition=AgentSnapshotTerminalDisposition.COMPLETED,
        runtime_evidence_fingerprint=_digest(f"runtime:{field_name}"),
        eval_result_revision=_digest(f"eval:{field_name}"),
        memory_evidence_fingerprint=valid.attribution_fingerprint,
        recorded_at=datetime(2026, 8, 24, tzinfo=UTC),
    )

    with pytest.raises(ValidationError, match="lineage conflict"):
        MemoryInterventionTrialBinding.create(
            spec=valid.spec,
            operation=operation,
            receipt=receipt,
            trial=trial,
            result=result,
            attribution=valid.attribution,
        )


@pytest.mark.parametrize(
    "status",
    (
        MemoryInterventionEffectStatus.INDETERMINATE,
        MemoryInterventionEffectStatus.CONFLICTING,
    ),
)
def test_comparability_rejects_uncertain_application_evidence(
    status: MemoryInterventionEffectStatus,
) -> None:
    baseline = _binding(_spec(MemoryInterventionKind.AS_DECLARED), f"baseline-{status.value}")
    applied = _binding(
        _spec(MemoryInterventionKind.AUTOMATIC_RECALL_OFF),
        f"intervention-{status.value}",
    )
    uncertain_receipt = MemoryInterventionReceipt.create(
        spec=applied.spec,
        operation=applied.operation,
        status=status,
    )
    uncertain = MemoryInterventionTrialBinding.create(
        spec=applied.spec,
        operation=applied.operation,
        receipt=uncertain_receipt,
        trial=applied.trial,
        result=applied.result,
        attribution=applied.attribution,
    )

    with pytest.raises(ValueError, match="determinate intervention effects"):
        MemoryInterventionComparability.create(
            baseline=baseline,
            intervention=uncertain,
        )


def test_replacement_spec_rejects_two_fixtures_for_one_source_item() -> None:
    source_item = MemoryInterventionItemIdentity(
        kind=MemoryInterventionItemIdentityKind.FINGERPRINT,
        revision_fingerprint=_digest("one-source-revision"),
        item_fingerprint=_digest("one-source-item"),
    )
    changes = tuple(
        MemoryInterventionItemChange(
            kind=MemoryInterventionChangeKind.REPLACE,
            source_item=source_item,
            fixture=MemoryInterventionFixtureRef(
                fixture_id=f"fixture-{index}",
                fixture_fingerprint=_digest(f"fixture-{index}"),
                representation_fingerprint=_digest(f"fixture-representation-{index}"),
                size_bytes=16,
            ),
        )
        for index in range(2)
    )
    policy = _policy()

    with pytest.raises(ValidationError, match="source identities must be unique"):
        MemoryInterventionSpec.create(
            spec_id="conflicting-replacements",
            snapshot=_snapshot(),
            starting_recall_policy=policy,
            trial_recall_policy=policy,
            trial_state_mode=AgentSnapshotTrialStateMode.RESET_EACH_TRIAL,
            kind=MemoryInterventionKind.REPLACE_ITEMS,
            bounds=MemoryInterventionBounds(max_changed_items=2, max_fixture_bytes=32),
            changes=changes,
            proposer_fingerprint=_digest("proposer"),
            source_fingerprint=_digest("source"),
            reason="reject contradictory replacements",
        )


def test_alias_source_identity_includes_key_scope_but_revisions_stay_unique() -> None:
    digest = _digest("shared-alias-digest")
    revision = _digest("shared-alias-revision")
    sources = tuple(
        MemoryInterventionItemIdentity(
            kind=MemoryInterventionItemIdentityKind.ALIAS,
            revision_fingerprint=revision,
            alias=MemoryEvidenceAlias(
                key_id=f"alias-key-{index}",
                kind="item",
                digest=digest,
            ),
        )
        for index in range(2)
    )
    assert sources[0].sort_key() != sources[1].sort_key()
    changes = tuple(
        MemoryInterventionItemChange(
            kind=MemoryInterventionChangeKind.REPLACE,
            source_item=sources[index],
            fixture=MemoryInterventionFixtureRef(
                fixture_id=f"alias-fixture-{index}",
                fixture_fingerprint=_digest(f"alias-fixture-{index}"),
                representation_fingerprint=_digest(f"alias-representation-{index}"),
                size_bytes=16,
            ),
        )
        for index in range(2)
    )
    policy = _policy()

    with pytest.raises(ValidationError, match="source revision identities must be unique"):
        MemoryInterventionSpec.create(
            spec_id="key-scoped-aliases",
            snapshot=_snapshot(),
            starting_recall_policy=policy,
            trial_recall_policy=policy,
            trial_state_mode=AgentSnapshotTrialStateMode.RESET_EACH_TRIAL,
            kind=MemoryInterventionKind.REPLACE_ITEMS,
            bounds=MemoryInterventionBounds(max_changed_items=2, max_fixture_bytes=32),
            changes=changes,
            proposer_fingerprint=_digest("alias-proposer"),
            source_fingerprint=_digest("alias-source"),
            reason="keep receipt evidence unambiguous",
        )


def test_receipt_rejects_application_receipts_for_undeclared_effects() -> None:
    binding = _binding(_spec(MemoryInterventionKind.AUTOMATIC_RECALL_OFF), "foreign-effect")
    declared_effect = _digest("declared-effect")

    with pytest.raises(ValueError, match="declared effect fingerprints"):
        MemoryInterventionReceipt.create(
            spec=binding.spec,
            operation=binding.operation,
            status=MemoryInterventionEffectStatus.APPLIED,
            result_memory_state_fingerprint=_digest("foreign-effect-result"),
            result_recall_policy_fingerprint=binding.spec.trial_recall_policy_fingerprint,
            effect_fingerprints=(declared_effect,),
            application_effect_receipts=(
                MemoryInterventionEffectReceiptRef(
                    owner_id="executor",
                    receipt_fingerprint=_digest("application-receipt"),
                    effect_fingerprint=_digest("undeclared-effect"),
                ),
            ),
        )


def test_receipt_rejects_one_application_receipt_claiming_two_effects() -> None:
    binding = _binding(_spec(MemoryInterventionKind.AUTOMATIC_RECALL_OFF), "conflicting-effect")
    effects = (_digest("effect-a"), _digest("effect-b"))
    receipt_fingerprint = _digest("one-application-receipt")

    with pytest.raises(ValueError, match="receipt identities must be unique"):
        MemoryInterventionReceipt.create(
            spec=binding.spec,
            operation=binding.operation,
            status=MemoryInterventionEffectStatus.APPLIED,
            result_memory_state_fingerprint=_digest("conflicting-effect-result"),
            result_recall_policy_fingerprint=binding.spec.trial_recall_policy_fingerprint,
            effect_fingerprints=effects,
            application_effect_receipts=tuple(
                MemoryInterventionEffectReceiptRef(
                    owner_id="executor",
                    receipt_fingerprint=receipt_fingerprint,
                    effect_fingerprint=effect,
                )
                for effect in effects
            ),
        )


@pytest.mark.parametrize("covered_effect_count", (0, 1))
def test_applied_receipt_requires_application_receipt_for_every_effect(
    covered_effect_count: int,
) -> None:
    binding = _binding(_spec(MemoryInterventionKind.AUTOMATIC_RECALL_OFF), "effect-coverage")
    effects = (_digest("covered-effect"), _digest("uncovered-effect"))
    effect_receipts = tuple(
        MemoryInterventionEffectReceiptRef(
            owner_id="test.intervention-executor",
            receipt_fingerprint=_digest(f"coverage-receipt-{index}"),
            effect_fingerprint=effects[index],
        )
        for index in range(covered_effect_count)
    )

    with pytest.raises(ValidationError, match="application receipt proof for every effect"):
        MemoryInterventionReceipt.create(
            spec=binding.spec,
            operation=binding.operation,
            status=MemoryInterventionEffectStatus.APPLIED,
            result_memory_state_fingerprint=_digest("effect-coverage-result"),
            result_recall_policy_fingerprint=binding.spec.trial_recall_policy_fingerprint,
            effect_fingerprints=effects,
            application_effect_receipts=effect_receipts,
        )


def test_applied_receipt_allows_multiple_application_receipts_for_one_effect() -> None:
    binding = _binding(_spec(MemoryInterventionKind.AUTOMATIC_RECALL_OFF), "shared-effect")
    effect_fingerprint = _digest("shared-effect")

    receipt = MemoryInterventionReceipt.create(
        spec=binding.spec,
        operation=binding.operation,
        status=MemoryInterventionEffectStatus.APPLIED,
        result_memory_state_fingerprint=_digest("shared-effect-result"),
        result_recall_policy_fingerprint=binding.spec.trial_recall_policy_fingerprint,
        effect_fingerprints=(effect_fingerprint,),
        application_effect_receipts=tuple(
            MemoryInterventionEffectReceiptRef(
                owner_id=f"executor-{index}",
                receipt_fingerprint=_digest(f"shared-effect-receipt-{index}"),
                effect_fingerprint=effect_fingerprint,
            )
            for index in range(2)
        ),
    )

    assert len(receipt.application_effect_receipts) == 2


def test_json_ingress_rejects_applied_effect_without_application_receipt() -> None:
    binding = _binding(_spec(MemoryInterventionKind.AUTOMATIC_RECALL_OFF), "json-coverage")
    document = json.loads(memory_intervention_to_json(binding.receipt))
    document["application_effect_receipts"] = []
    identity_material = {key: value for key, value in document.items() if key != "fingerprint"}
    document["fingerprint"] = _content_fingerprint(
        identity_material,
        MemoryInterventionReceipt.__name__,
    )

    with pytest.raises(ValidationError, match="application receipt proof for every effect"):
        memory_intervention_from_json(json.dumps(document))


@pytest.mark.parametrize(
    ("field_name", "wrong_type_value"),
    (
        ("evidence_only", 1),
        ("production_mutation_allowed", 0),
    ),
)
def test_json_ingress_rejects_integer_boolean_literals(
    field_name: str,
    wrong_type_value: int,
) -> None:
    document = json.loads(memory_intervention_to_json(_spec(MemoryInterventionKind.AS_DECLARED)))
    document[field_name] = wrong_type_value

    with pytest.raises(ValidationError, match="JSON boolean"):
        memory_intervention_from_json(json.dumps(document))


def test_json_ingress_rejects_boolean_nested_schema_versions() -> None:
    binding = _binding(_spec(MemoryInterventionKind.AS_DECLARED), "nested-version")
    document = json.loads(memory_intervention_to_json(binding))
    document["spec"]["schema_version"] = True

    with pytest.raises(ValidationError, match="JSON integer"):
        memory_intervention_from_json(json.dumps(document))


@pytest.mark.parametrize(
    ("section", "field_name", "wrong_type_value", "json_type_name"),
    (
        ("trial", "schema_version", True, "integer"),
        ("result", "schema_version", True, "integer"),
        ("trial", "created_at", 1_787_529_600, "string"),
        ("result", "recorded_at", 1_787_529_600, "string"),
    ),
)
def test_json_ingress_rejects_coerced_imported_trial_literals(
    section: str,
    field_name: str,
    wrong_type_value: bool | int,
    json_type_name: str,
) -> None:
    binding = _binding(_spec(MemoryInterventionKind.AS_DECLARED), f"strict-{section}-{field_name}")
    document = json.loads(memory_intervention_to_json(binding))
    document[section][field_name] = wrong_type_value

    with pytest.raises(ValidationError, match=f"JSON {json_type_name}"):
        memory_intervention_from_json(json.dumps(document))


def test_json_ingress_rejects_coerced_attribution_timestamps() -> None:
    binding = _binding(
        _spec(MemoryInterventionKind.AS_DECLARED),
        "strict-attribution-timestamp",
        attribution=_complete_recall_attribution(),
    )
    document = json.loads(memory_intervention_to_json(binding))
    document["attribution"]["receipts"][0]["created_at"] = 1_787_529_600

    with pytest.raises(ValidationError, match="JSON string"):
        memory_intervention_from_json(json.dumps(document))


def test_json_serializer_rejects_invalid_objects_with_type_error() -> None:
    with pytest.raises(TypeError, match="MemoryInterventionRecord"):
        memory_intervention_to_json(object())  # type: ignore[arg-type]


def test_receipt_iterables_stop_at_their_hard_bounds() -> None:
    binding = _binding(_spec(MemoryInterventionKind.AUTOMATIC_RECALL_OFF), "bounded-inputs")

    consumed_effects: list[int] = []

    def effects():
        for index in range(1_000):
            consumed_effects.append(index)
            yield f"{index:064x}"

    with pytest.raises(ValueError, match="effect_fingerprints must contain at most"):
        MemoryInterventionReceipt.create(
            spec=binding.spec,
            operation=binding.operation,
            status=MemoryInterventionEffectStatus.APPLIED,
            result_memory_state_fingerprint=_digest("bounded-result"),
            result_recall_policy_fingerprint=binding.spec.trial_recall_policy_fingerprint,
            effect_fingerprints=effects(),
        )
    assert len(consumed_effects) == MEMORY_INTERVENTION_MAX_CHANGED_ITEMS + 1

    consumed_receipts: list[int] = []

    def effect_receipts():
        for index in range(1_000):
            consumed_receipts.append(index)
            yield MemoryInterventionEffectReceiptRef(
                owner_id="executor",
                receipt_fingerprint=_digest(f"receipt-{index}"),
                effect_fingerprint=_digest(f"receipt-effect-{index}"),
            )

    with pytest.raises(ValueError, match="application_effect_receipts must contain at most"):
        MemoryInterventionReceipt.create(
            spec=binding.spec,
            operation=binding.operation,
            status=MemoryInterventionEffectStatus.APPLIED,
            result_memory_state_fingerprint=_digest("bounded-receipt-result"),
            result_recall_policy_fingerprint=binding.spec.trial_recall_policy_fingerprint,
            effect_fingerprints=(_digest("bounded-receipt-effect"),),
            application_effect_receipts=effect_receipts(),
        )
    assert len(consumed_receipts) == MEMORY_INTERVENTION_MAX_EFFECT_RECEIPTS + 1


def test_spec_changes_stop_at_their_hard_bound_before_sorting() -> None:
    policy = _policy()
    source = MemoryInterventionItemIdentity(
        kind=MemoryInterventionItemIdentityKind.FINGERPRINT,
        revision_fingerprint=_digest("bounded-source-revision"),
        item_fingerprint=_digest("bounded-source-item"),
    )
    change = MemoryInterventionItemChange(
        kind=MemoryInterventionChangeKind.OMIT,
        source_item=source,
    )
    consumed: list[int] = []

    def changes():
        for index in range(1_000):
            consumed.append(index)
            yield change

    with pytest.raises(ValueError, match="changes must contain at most"):
        MemoryInterventionSpec.create(
            spec_id="bounded-changes",
            snapshot=_snapshot(),
            starting_recall_policy=policy,
            trial_recall_policy=policy,
            trial_state_mode=AgentSnapshotTrialStateMode.RESET_EACH_TRIAL,
            kind=MemoryInterventionKind.OMIT_ITEMS,
            bounds=MemoryInterventionBounds(max_changed_items=1, max_fixture_bytes=0),
            changes=changes(),  # type: ignore[arg-type]
            proposer_fingerprint=_digest("bounded-proposer"),
            source_fingerprint=_digest("bounded-source"),
            reason="bound before sorting",
        )
    assert len(consumed) == MEMORY_INTERVENTION_MAX_CHANGED_ITEMS + 1

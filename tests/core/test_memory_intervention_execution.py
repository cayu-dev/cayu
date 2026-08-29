from __future__ import annotations

import asyncio
import hashlib
import json
import multiprocessing
import sqlite3
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from functools import wraps
from multiprocessing.connection import Connection

import pytest

from cayu.agent_snapshots import (
    AgentSnapshot,
    AgentSnapshotAuthorityRef,
    AgentSnapshotCaptureRequest,
    AgentSnapshotCompleteness,
    AgentSnapshotComponentCapture,
    AgentSnapshotComponentKind,
    AgentSnapshotComponentProvider,
    AgentSnapshotComponentRef,
    AgentSnapshotComponentSelector,
    AgentSnapshotConsistency,
    AgentSnapshotCoordinator,
    AgentSnapshotExecutionProfileComponent,
    AgentSnapshotExecutionProfileRef,
    AgentSnapshotLearningDisposition,
    AgentSnapshotLogicalRef,
    AgentSnapshotMaterialization,
    AgentSnapshotMaterializationCapability,
    AgentSnapshotMaterializationOperation,
    AgentSnapshotMaterializationRequest,
    AgentSnapshotMaterializedComponent,
    AgentSnapshotOverlayKind,
    AgentSnapshotOverlayRef,
    AgentSnapshotRedaction,
    AgentSnapshotSubject,
    AgentSnapshotTerminalDisposition,
    AgentSnapshotTrialBinding,
    AgentSnapshotTrialStateMode,
    MemoryStateRef,
    SQLiteAgentSnapshotStore,
    execution_profile_snapshot_ref,
)
from cayu.core.agents import AgentSpec
from cayu.core.execution_identity import ExecutionProfileBehaviorIdentity
from cayu.core.messages import Message
from cayu.environments import Environment, EnvironmentSpec
from cayu.evals.memory_attribution import (
    EvalMemoryEvidenceCompleteness,
    EvalMemoryEvidenceLimitation,
    standard_eval_memory_attribution_bounds,
)
from cayu.evals.models import (
    EvalAssertionResult,
    EvalCaseContractV1,
    EvalOutcome,
    EvalStatus,
    EvalTrialResult,
)
from cayu.evals.testing import ScriptedModelProvider
from cayu.memory import AutomaticRecallMode, AutomaticRecallPolicy
from cayu.memory_attribution import (
    MemoryAttribution,
    MemoryAttributionBounds,
    MemoryAttributionStatus,
    MemoryContextExposureAttribution,
    MemoryEvidenceAlias,
    MemoryExposureTransitionAttribution,
    MemoryRecallAttribution,
    MemoryRecallItemAttribution,
)
from cayu.memory_evidence import (
    ContextExposureEvidenceKind,
    ContextExposureState,
    RecallEvidenceQuery,
    RecallItemAdmission,
    RecallItemSelectionReason,
)
from cayu.memory_intervention_execution import (
    CayuMemoryInterventionRuntimeRunner,
    InMemoryMemoryInterventionExecutionStore,
    MemoryInterventionEvaluator,
    MemoryInterventionExecutionConflict,
    MemoryInterventionExecutionPhase,
    MemoryInterventionExecutionRecord,
    MemoryInterventionExecutionStatus,
    MemoryInterventionExecutionStore,
    MemoryInterventionExecutor,
    MemoryInterventionIsolationAuthority,
    MemoryInterventionOverlayProvider,
    MemoryInterventionRequestFingerprintKey,
    MemoryInterventionRuntimeApplicationFactory,
    MemoryInterventionRuntimeResult,
    MemoryInterventionRuntimeRunner,
    MemoryInterventionRuntimeView,
    MemoryInterventionTrialRequest,
    SQLiteMemoryInterventionExecutionStore,
)
from cayu.memory_interventions import (
    MemoryInterventionBounds,
    MemoryInterventionChangeKind,
    MemoryInterventionEffectReceiptRef,
    MemoryInterventionEffectStatus,
    MemoryInterventionFixtureRef,
    MemoryInterventionItemChange,
    MemoryInterventionItemIdentity,
    MemoryInterventionItemIdentityKind,
    MemoryInterventionKind,
    MemoryInterventionOperation,
    MemoryInterventionReceipt,
    MemoryInterventionSpec,
    MemoryNegativeControlKind,
)
from cayu.providers import ModelRequest, ModelStreamEvent
from cayu.recall import KNOWLEDGE_LEXICAL_CHANNEL, KNOWLEDGE_SEMANTIC_CHANNEL
from cayu.retrieval import (
    WEIGHTED_RECIPROCAL_RANK_FUSION_VERSION,
    WeightedReciprocalRankFusionConfig,
)
from cayu.runtime.app import CayuApp
from cayu.runtime.execution_profiles import ExecutionProfileMismatchError
from cayu.runtime.invocation import (
    InvocationOrigin,
    InvocationOriginTrust,
    SessionExecutionSource,
)
from cayu.runtime.memory_context import (
    AutomaticRecallContextPolicy,
    AutomaticRecallSourceConfig,
)
from cayu.runtime.request_footprints import RequestFootprintConfig
from cayu.runtime.sessions import (
    InterruptSessionRequest,
    RunRequest,
    SessionStatus,
    SessionStore,
    TerminalSessionEvidence,
    TerminalSessionEvidenceError,
    TerminalSessionEvidenceErrorCode,
    run_request_with_runtime_invocation,
)
from cayu.runtime.usage import SessionUsageSummary
from cayu.storage.memory import (
    InMemoryKnowledgeStore,
    KnowledgeAccessScope,
    KnowledgeEntry,
    KnowledgeListQuery,
)
from cayu.storage.sqlite import SQLiteSessionStore


def _async_test(function):
    @wraps(function)
    def run(*args, **kwargs):
        return asyncio.run(function(*args, **kwargs))

    return run


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


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
    capability: AgentSnapshotMaterializationCapability,
) -> AgentSnapshotComponentRef:
    return AgentSnapshotComponentRef(
        kind=kind,
        provider_id=f"test.{kind.value}.v1",
        logical=logical,
        consistency=AgentSnapshotConsistency.FRONTIER_CONSISTENT,
        completeness=AgentSnapshotCompleteness.COMPLETE,
        redaction=AgentSnapshotRedaction.BOUNDED_PROJECTION,
        materialization=capability,
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


def _canonical_policy(
    mode: AutomaticRecallMode = AutomaticRecallMode.STRONG_MATCHES,
) -> AutomaticRecallPolicy:
    return AutomaticRecallPolicy(
        calibration_version="calibration-v1",
        fusion_strategy_version=WEIGHTED_RECIPROCAL_RANK_FUSION_VERSION,
        fusion_configuration_version="configuration-v1",
        mode=mode,
        minimum_inject_score=0.0,
        minimum_offer_score=0.0,
    )


def _snapshot(
    *,
    execution_profile_fingerprint: str | None = None,
    execution_profile: AgentSnapshotExecutionProfileRef | None = None,
    recall_policy: AutomaticRecallPolicy | None = None,
) -> AgentSnapshot:
    if execution_profile_fingerprint is not None and execution_profile is not None:
        raise ValueError("Provide either an execution profile or its fingerprint, not both.")
    scope = _digest("authority")
    resolved_profile_fingerprint = (
        execution_profile.fingerprint
        if execution_profile is not None
        else execution_profile_fingerprint
    )
    profile_ref = (
        _ref("profile")
        if resolved_profile_fingerprint is None
        else AgentSnapshotLogicalRef(
            fingerprint=resolved_profile_fingerprint,
            revision="revision:profile",
        )
    )
    body = _ref("body")
    policy = recall_policy or _policy()
    profile = execution_profile or AgentSnapshotExecutionProfileRef(
        schema_version=1,
        fingerprint=profile_ref.fingerprint,
        components=(
            AgentSnapshotExecutionProfileComponent(
                name="provider_target",
                fingerprint=_digest("provider-target"),
                availability="available",
            ),
        ),
    )
    memory = MemoryStateRef.create(
        knowledge=_ref("knowledge", scope=scope),
        recall_policy=AgentSnapshotLogicalRef(
            fingerprint=policy.fingerprint(),
            revision="revision:recall-policy",
            scope_fingerprint=scope,
        ),
        learning_disposition=AgentSnapshotLearningDisposition.ISOLATED,
    )
    return AgentSnapshot.create(
        capture_request_id="capture",
        captured_at=datetime(2026, 8, 25, tzinfo=UTC),
        subject=AgentSnapshotSubject(
            agent_id="agent",
            application_id="application",
            project_id="project",
            body_release=body,
        ),
        authority_scope_fingerprint=scope,
        execution_profile=profile,
        memory_state=memory,
        evaluator=AgentSnapshotAuthorityRef(
            identity=AgentSnapshotLogicalRef(
                fingerprint=_digest("evaluator"),
                revision="evaluator:v1",
            )
        ),
        components=(
            _component(
                AgentSnapshotComponentKind.BODY,
                body,
                capability=AgentSnapshotMaterializationCapability.REFERENCE_ONLY,
            ),
            _component(
                AgentSnapshotComponentKind.EXECUTION_PROFILE,
                profile_ref,
                capability=AgentSnapshotMaterializationCapability.REFERENCE_ONLY,
            ),
            _component(
                AgentSnapshotComponentKind.MEMORY,
                AgentSnapshotLogicalRef(
                    fingerprint=memory.fingerprint,
                    revision="memory:v1",
                    scope_fingerprint=scope,
                ),
                capability=AgentSnapshotMaterializationCapability.RESTORABLE,
            ),
        ),
    )


class _SnapshotProvider(AgentSnapshotComponentProvider):
    def __init__(
        self,
        component: AgentSnapshotComponentRef,
        *,
        results: dict[str, AgentSnapshotMaterializedComponent],
    ) -> None:
        self.kind = component.kind
        self.provider_id = component.provider_id
        self._component = component
        self._results = results

    async def capture(
        self,
        request: AgentSnapshotCaptureRequest,
        selector: AgentSnapshotComponentSelector,
    ) -> AgentSnapshotComponentCapture:
        raise AssertionError("The execution test starts from a frozen snapshot.")

    async def verify(
        self,
        snapshot: AgentSnapshot,
        component: AgentSnapshotComponentRef,
    ) -> bool:
        return component == self._component

    async def materialize(
        self,
        snapshot: AgentSnapshot,
        component: AgentSnapshotComponentRef,
        request: AgentSnapshotMaterializationRequest,
        operation: AgentSnapshotMaterializationOperation,
    ) -> AgentSnapshotMaterializedComponent:
        existing = self._results.get(operation.operation_id)
        if existing is not None:
            return existing
        overlay = None
        if component.kind is AgentSnapshotComponentKind.MEMORY:
            overlay = AgentSnapshotOverlayRef.create(
                kind=AgentSnapshotOverlayKind.MEMORY,
                overlay_id=f"memory-{request.state_scope_id[:16]}",
                baseline_fingerprint=component.logical.fingerprint,
                candidate_id=request.candidate_id,
                state_scope_id=request.state_scope_id,
            )
        result = AgentSnapshotMaterializedComponent(
            kind=component.kind,
            baseline_fingerprint=component.logical.fingerprint,
            capability=component.materialization,
            materialization_ref=f"cayu-ref:test:{component.kind.value}",
            overlay=overlay,
        )
        self._results[operation.operation_id] = result
        return result

    async def recover_materialization_operation(
        self,
        snapshot: AgentSnapshot,
        component: AgentSnapshotComponentRef,
        request: AgentSnapshotMaterializationRequest,
        operation: AgentSnapshotMaterializationOperation,
    ) -> AgentSnapshotMaterializedComponent:
        return await self.materialize(snapshot, component, request, operation)


def _providers(
    snapshot: AgentSnapshot,
    results: dict[str, AgentSnapshotMaterializedComponent],
) -> tuple[_SnapshotProvider, ...]:
    return tuple(_SnapshotProvider(component, results=results) for component in snapshot.components)


def _spec(
    snapshot: AgentSnapshot,
    *,
    state_mode: AgentSnapshotTrialStateMode = AgentSnapshotTrialStateMode.RESET_EACH_TRIAL,
    spec_id: str | None = None,
    recall_policy: AutomaticRecallPolicy | None = None,
) -> MemoryInterventionSpec:
    policy = recall_policy or _policy()
    return MemoryInterventionSpec.create(
        spec_id=spec_id or f"spec-{state_mode.value}",
        snapshot=snapshot,
        starting_recall_policy=policy,
        trial_recall_policy=policy,
        trial_state_mode=state_mode,
        kind=MemoryInterventionKind.AS_DECLARED,
        bounds=MemoryInterventionBounds(max_changed_items=0, max_fixture_bytes=0),
    )


def _omit_spec(
    snapshot: AgentSnapshot,
    *,
    recall_policy: AutomaticRecallPolicy | None = None,
) -> MemoryInterventionSpec:
    policy = recall_policy or _policy()
    return MemoryInterventionSpec.create(
        spec_id="spec-omit",
        snapshot=snapshot,
        starting_recall_policy=policy,
        trial_recall_policy=policy,
        trial_state_mode=AgentSnapshotTrialStateMode.RESET_EACH_TRIAL,
        kind=MemoryInterventionKind.OMIT_ITEMS,
        bounds=MemoryInterventionBounds(max_changed_items=1, max_fixture_bytes=0),
        changes=(
            MemoryInterventionItemChange(
                kind=MemoryInterventionChangeKind.OMIT,
                source_item=MemoryInterventionItemIdentity(
                    kind=MemoryInterventionItemIdentityKind.FINGERPRINT,
                    revision_fingerprint=_digest("entry:1"),
                    item_fingerprint=_digest("entry"),
                ),
            ),
        ),
        proposer_fingerprint=_digest("proposer"),
        source_fingerprint=_digest("source"),
        reason="measure omission sensitivity",
    )


def _automatic_recall_off_spec(
    snapshot: AgentSnapshot,
    *,
    starting_policy: AutomaticRecallPolicy | None = None,
    trial_policy: AutomaticRecallPolicy | None = None,
) -> MemoryInterventionSpec:
    return MemoryInterventionSpec.create(
        spec_id="spec-automatic-recall-off",
        snapshot=snapshot,
        starting_recall_policy=starting_policy or _policy(),
        trial_recall_policy=trial_policy or _policy(AutomaticRecallMode.OFF),
        trial_state_mode=AgentSnapshotTrialStateMode.RESET_EACH_TRIAL,
        kind=MemoryInterventionKind.AUTOMATIC_RECALL_OFF,
        bounds=MemoryInterventionBounds(max_changed_items=0, max_fixture_bytes=0),
    )


_REPLACEMENT_FIXTURE_TEXT = "Replacement fixture says Atlas is released Saturday."
_NEGATIVE_CONTROL_FIXTURE_TEXT = "Irrelevant fixture: Atlas office plants need water."


def _fixture(name: str, text: str) -> MemoryInterventionFixtureRef:
    encoded = text.encode()
    return MemoryInterventionFixtureRef(
        fixture_id=f"fixture-{name}",
        fixture_fingerprint=hashlib.sha256(encoded).hexdigest(),
        representation_fingerprint=_digest(f"utf8-text:{text}"),
        size_bytes=len(encoded),
    )


def _replacement_spec(
    snapshot: AgentSnapshot,
    *,
    recall_policy: AutomaticRecallPolicy | None = None,
    fixture: MemoryInterventionFixtureRef | None = None,
) -> MemoryInterventionSpec:
    fixture = fixture or _fixture("replacement", _REPLACEMENT_FIXTURE_TEXT)
    policy = recall_policy or _policy()
    return MemoryInterventionSpec.create(
        spec_id="spec-replacement",
        snapshot=snapshot,
        starting_recall_policy=policy,
        trial_recall_policy=policy,
        trial_state_mode=AgentSnapshotTrialStateMode.RESET_EACH_TRIAL,
        kind=MemoryInterventionKind.REPLACE_ITEMS,
        bounds=MemoryInterventionBounds(
            max_changed_items=1,
            max_fixture_bytes=fixture.size_bytes,
        ),
        changes=(
            MemoryInterventionItemChange(
                kind=MemoryInterventionChangeKind.REPLACE,
                source_item=MemoryInterventionItemIdentity(
                    kind=MemoryInterventionItemIdentityKind.FINGERPRINT,
                    revision_fingerprint=_digest("entry:1"),
                    item_fingerprint=_digest("entry"),
                ),
                fixture=fixture,
            ),
        ),
        proposer_fingerprint=_digest("proposer"),
        source_fingerprint=_digest("source"),
        reason="measure replacement sensitivity",
    )


def _negative_control_spec(
    snapshot: AgentSnapshot,
    *,
    recall_policy: AutomaticRecallPolicy | None = None,
) -> MemoryInterventionSpec:
    fixture = _fixture("negative-control", _NEGATIVE_CONTROL_FIXTURE_TEXT)
    policy = recall_policy or _policy()
    return MemoryInterventionSpec.create(
        spec_id="spec-negative-control",
        snapshot=snapshot,
        starting_recall_policy=policy,
        trial_recall_policy=policy,
        trial_state_mode=AgentSnapshotTrialStateMode.RESET_EACH_TRIAL,
        kind=MemoryInterventionKind.NEGATIVE_CONTROL,
        bounds=MemoryInterventionBounds(
            max_changed_items=1,
            max_fixture_bytes=fixture.size_bytes,
        ),
        changes=(
            MemoryInterventionItemChange(
                kind=MemoryInterventionChangeKind.INJECT_NEGATIVE_CONTROL,
                fixture=fixture,
            ),
        ),
        negative_control_kind=MemoryNegativeControlKind.IRRELEVANT,
        proposer_fingerprint=_digest("proposer"),
        source_fingerprint=_digest("source"),
        reason="measure irrelevant-memory sensitivity",
    )


def _request(
    spec: MemoryInterventionSpec,
    *,
    candidate_id: str = "candidate-a",
    trial_id: str = "trial-1",
    prompt: str = "answer the case",
    timeout_seconds: int = 300,
) -> MemoryInterventionTrialRequest:
    return MemoryInterventionTrialRequest(
        spec=spec,
        candidate_id=candidate_id,
        trial_id=trial_id,
        case=EvalCaseContractV1(
            case_id="case-1",
            case_revision=f"sha256:{_digest('case-1')}",
        ),
        run_request=RunRequest(
            agent_name="agent",
            messages=[Message.text("user", prompt)],
        ),
        timeout_seconds=timeout_seconds,
    )


def _prepared_record(request: MemoryInterventionTrialRequest) -> MemoryInterventionExecutionRecord:
    return MemoryInterventionExecutionRecord.prepare(
        request,
        key=MemoryInterventionRequestFingerprintKey(
            key_id="test-key",
            secret=b"k" * 32,
        ),
        overlay_provider_id=_OverlayProvider.provider_id,
        overlay_provider_fingerprint=_OverlayProvider.execution_profile_fingerprint,
        runtime_runner_fingerprint=_RuntimeRunner.execution_profile_fingerprint,
        runtime_execution_profile_fingerprint=request.spec.execution_profile_fingerprint,
        evaluator_fingerprint=_Evaluator.evaluator_fingerprint,
        created_at=datetime(2026, 8, 25, tzinfo=UTC),
    )


def _record_successor(
    record: MemoryInterventionExecutionRecord,
    *,
    phase: MemoryInterventionExecutionPhase,
    **updates,
) -> MemoryInterventionExecutionRecord:
    values = record.model_dump(mode="python")
    values.update(updates)
    values.update(
        {
            "phase": phase,
            "revision": record.revision + 1,
            "updated_at": record.updated_at,
        }
    )
    return MemoryInterventionExecutionRecord.model_validate(values)


def _empty_attribution() -> MemoryAttribution:
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


def _indeterminate_exposure_attribution() -> MemoryAttribution:
    started_at = datetime(2026, 8, 25, tzinfo=UTC)
    transitions = tuple(
        MemoryExposureTransitionAttribution(
            revision=revision,
            state=state,
            occurred_at=started_at,
            evidence_kind=evidence_kind,
        )
        for revision, state, evidence_kind in (
            (
                0,
                ContextExposureState.PLANNED,
                ContextExposureEvidenceKind.COMPOSITION_PLANNED,
            ),
            (
                1,
                ContextExposureState.PREPARED,
                ContextExposureEvidenceKind.REQUEST_PREPARED,
            ),
            (
                2,
                ContextExposureState.DISPATCH_STARTED,
                ContextExposureEvidenceKind.DISPATCH_INTENT_COMMITTED,
            ),
            (
                3,
                ContextExposureState.INDETERMINATE,
                ContextExposureEvidenceKind.AMBIGUOUS_TRANSPORT,
            ),
        )
    )
    return MemoryAttribution(
        status=MemoryAttributionStatus.COMPLETE,
        truncated=False,
        observed_receipt_count=0,
        observed_exposure_count=1,
        observed_item_count=0,
        omitted_receipt_count_at_least=0,
        omitted_exposure_count_at_least=0,
        omitted_item_count_at_least=0,
        exposures=(
            MemoryContextExposureAttribution(
                exposure_alias=MemoryEvidenceAlias(
                    key_id="memory-alias-key",
                    kind="exposure",
                    digest=_digest("indeterminate-exposure"),
                ),
                interaction_alias=MemoryEvidenceAlias(
                    key_id="memory-alias-key",
                    kind="interaction",
                    digest=_digest("indeterminate-interaction"),
                ),
                projection_ordinal=0,
                model_step_id="indeterminate-model-step",
                model_attempt_id="indeterminate-model-attempt",
                provider_attempt_id="indeterminate-provider-attempt",
                created_at=started_at,
                updated_at=started_at,
                state=ContextExposureState.INDETERMINATE,
                state_revision=3,
                provider_exposure_proven=False,
                contributor_count=0,
                transitions=transitions,
                items_truncated=False,
                omitted_item_count_at_least=0,
            ),
        ),
    )


def _oversized_attribution() -> MemoryAttribution:
    receipts = tuple(
        MemoryRecallAttribution(
            receipt_alias=MemoryEvidenceAlias(
                key_id="memory-alias-key",
                kind="receipt",
                digest=_digest(f"oversized-receipt:{ordinal}"),
            ),
            interaction_alias=MemoryEvidenceAlias(
                key_id="memory-alias-key",
                kind="interaction",
                digest=_digest(f"oversized-interaction:{ordinal}"),
            ),
            projection_ordinal=ordinal,
            model_step_id=f"oversized-step-{ordinal}-" + ("x" * 220),
            created_at=datetime(2026, 8, 25, tzinfo=UTC),
            inspected_count=1,
            eligible_count=1,
            admitted_count=1,
            offered_count=0,
            silent_count=0,
            omitted_count=0,
            complete_source_count=1,
            partial_source_count=0,
            unavailable_source_count=0,
            failed_source_count=0,
            truncated=False,
            items=(
                MemoryRecallItemAttribution(
                    item_alias=MemoryEvidenceAlias(
                        key_id="memory-alias-key",
                        kind="item",
                        digest=_digest(f"oversized-item:{ordinal}"),
                    ),
                    ordinal=0,
                    admission=RecallItemAdmission.ADMITTED,
                    selection_reason=RecallItemSelectionReason.CALIBRATED_STRONG_MATCH,
                ),
            ),
            omitted_item_count_at_least=0,
        )
        for ordinal in range(1_000)
    )
    return MemoryAttribution(
        status=MemoryAttributionStatus.COMPLETE,
        truncated=False,
        observed_receipt_count=len(receipts),
        observed_exposure_count=0,
        observed_item_count=len(receipts),
        omitted_receipt_count_at_least=0,
        omitted_exposure_count_at_least=0,
        omitted_item_count_at_least=0,
        receipts=receipts,
    )


class _OverlayProvider(MemoryInterventionOverlayProvider):
    provider_id = "test.memory-intervention-overlay.v1"
    execution_profile_fingerprint = _digest("overlay-provider:v1")

    def __init__(
        self,
        views: dict[str, MemoryInterventionRuntimeView],
        *,
        terminal_status: MemoryInterventionEffectStatus | None = None,
        profile_fingerprint: str | None = None,
        trial_policy_factory=_policy,
    ) -> None:
        self.views = views
        self.terminal_status = terminal_status
        self.trial_policy_factory = trial_policy_factory
        if profile_fingerprint is not None:
            self.execution_profile_fingerprint = profile_fingerprint
        self.apply_calls = 0
        self.recover_calls = 0

    async def apply(
        self,
        *,
        spec: MemoryInterventionSpec,
        operation: MemoryInterventionOperation,
        materialization: AgentSnapshotMaterialization,
        trial: AgentSnapshotTrialBinding,
    ) -> MemoryInterventionRuntimeView:
        self.apply_calls += 1
        return self._materialize_view(
            spec=spec,
            operation=operation,
            materialization=materialization,
        )

    def _materialize_view(
        self,
        *,
        spec: MemoryInterventionSpec,
        operation: MemoryInterventionOperation,
        materialization: AgentSnapshotMaterialization,
    ) -> MemoryInterventionRuntimeView:
        existing = self.views.get(operation.operation_id)
        if existing is not None:
            return existing
        status = self.terminal_status
        if status is None:
            status = (
                MemoryInterventionEffectStatus.VERIFIED_NO_CHANGE
                if spec.kind is MemoryInterventionKind.AS_DECLARED
                else MemoryInterventionEffectStatus.APPLIED
            )
        source_revisions = tuple(
            change.source_item.revision_fingerprint
            for change in spec.changes
            if change.source_item is not None
        )
        effect_fingerprints = (
            ()
            if status is not MemoryInterventionEffectStatus.APPLIED
            else (_digest(f"effect:{operation.operation_id}"),)
        )
        effect_receipts = (
            ()
            if not effect_fingerprints
            else (
                MemoryInterventionEffectReceiptRef(
                    owner_id=self.provider_id,
                    receipt_fingerprint=_digest(f"receipt:{operation.operation_id}"),
                    effect_fingerprint=effect_fingerprints[0],
                ),
            )
        )
        receipt = MemoryInterventionReceipt.create(
            spec=spec,
            operation=operation,
            status=status,
            result_memory_state_fingerprint=(
                None
                if status
                in {
                    MemoryInterventionEffectStatus.CONFLICTING,
                    MemoryInterventionEffectStatus.INDETERMINATE,
                }
                else (
                    spec.memory_state_fingerprint
                    if status is MemoryInterventionEffectStatus.VERIFIED_NO_CHANGE
                    else _digest(f"memory-result:{operation.operation_id}")
                )
            ),
            result_recall_policy_fingerprint=(
                None
                if status
                in {
                    MemoryInterventionEffectStatus.CONFLICTING,
                    MemoryInterventionEffectStatus.INDETERMINATE,
                }
                else spec.trial_recall_policy_fingerprint
            ),
            matched_item_count=(
                len(source_revisions) if status is MemoryInterventionEffectStatus.APPLIED else 0
            ),
            changed_item_revision_fingerprints=(
                source_revisions if status is MemoryInterventionEffectStatus.APPLIED else ()
            ),
            effect_fingerprints=effect_fingerprints,
            application_effect_receipts=effect_receipts,
        )
        scope = KnowledgeAccessScope(allowed_namespaces=["default"])
        view = MemoryInterventionRuntimeView(
            materialization_fingerprint=materialization.fingerprint,
            memory_overlay_fingerprint=operation.memory_overlay_fingerprint,
            state_scope_id=operation.state_scope_id,
            knowledge_store=InMemoryKnowledgeStore(access_scope=scope),
            knowledge_access_scope=scope,
            isolation_authority=MemoryInterventionIsolationAuthority(
                materialization_fingerprint=materialization.fingerprint,
                memory_overlay_fingerprint=operation.memory_overlay_fingerprint,
                state_scope_id=operation.state_scope_id,
            ),
            trial_recall_policy=self.trial_policy_factory(spec.trial_recall_mode),
            receipt=receipt,
        )
        self.views[operation.operation_id] = view
        return view

    async def recover(
        self,
        *,
        spec: MemoryInterventionSpec,
        operation: MemoryInterventionOperation,
        materialization: AgentSnapshotMaterialization,
        trial: AgentSnapshotTrialBinding,
    ) -> MemoryInterventionRuntimeView:
        self.recover_calls += 1
        value = self.views.get(operation.operation_id)
        if value is None:
            value = self._materialize_view(
                spec=spec,
                operation=operation,
                materialization=materialization,
            )
        return value


def _canonical_runtime_context_policy(
    admission_policy: AutomaticRecallPolicy,
) -> AutomaticRecallContextPolicy:
    return AutomaticRecallContextPolicy(
        admission_policy=admission_policy,
        fusion_config=WeightedReciprocalRankFusionConfig(
            strategy_version=WEIGHTED_RECIPROCAL_RANK_FUSION_VERSION,
            configuration_version="configuration-v1",
            channel_weights={
                KNOWLEDGE_LEXICAL_CHANNEL: 1.0,
                KNOWLEDGE_SEMANTIC_CHANNEL: 1.0,
            },
        ),
        sources=AutomaticRecallSourceConfig(
            include_transcript=False,
            knowledge_namespace="default",
        ),
    )


class _CanonicalRuntimeApplicationFactory(MemoryInterventionRuntimeApplicationFactory):
    factory_id = "test.canonical-memory-intervention-runtime.v1"
    execution_profile_fingerprint = _digest("canonical-runtime-factory:v1")

    def __init__(
        self,
        *,
        sessions: SessionStore,
        provider: ScriptedModelProvider,
    ) -> None:
        self.sessions = sessions
        self.provider = provider
        self.profile_by_policy: dict[str, str] = {}
        self.model_by_policy: dict[str, str] = {}
        self.created_apps: list[CayuApp] = []

    def expected_execution_profile_fingerprint(
        self,
        spec: MemoryInterventionSpec,
    ) -> str:
        try:
            return self.profile_by_policy[spec.trial_recall_policy_fingerprint]
        except KeyError:
            return super().expected_execution_profile_fingerprint(spec)

    def build_app(
        self,
        *,
        knowledge_store: InMemoryKnowledgeStore,
        scope: KnowledgeAccessScope,
        policy: AutomaticRecallPolicy,
    ) -> CayuApp:
        app = CayuApp(
            session_store=self.sessions,
            request_footprint=RequestFootprintConfig(
                fingerprint_key_id="memory-intervention-test",
                fingerprint_key="memory-intervention-test-secret-material",
            ),
            enable_logging=False,
        )
        app.register_provider(self.provider, default=True)
        app.register_environment(
            Environment(
                EnvironmentSpec(
                    name="intervention",
                    execution_profile_identity=ExecutionProfileBehaviorIdentity(
                        name="test:memory-intervention-environment",
                        behavior_version="1",
                        implementation_version="1",
                    ),
                ),
                knowledge_store=knowledge_store,
                knowledge_access_scope=scope,
            ),
            default=True,
        )
        app.register_agent(
            AgentSpec(
                name="agent",
                model=self.model_by_policy.get(policy.fingerprint(), "fake-model"),
            ),
            context_policy=_canonical_runtime_context_policy(policy),
        )
        self.created_apps.append(app)
        return app

    async def create(
        self,
        *,
        request: MemoryInterventionTrialRequest,
        execution: MemoryInterventionExecutionRecord,
        trial: AgentSnapshotTrialBinding,
        operation: MemoryInterventionOperation,
        view: MemoryInterventionRuntimeView,
    ) -> object:
        del request, execution, trial, operation
        return self.build_app(
            knowledge_store=view.knowledge_store,
            scope=view.knowledge_access_scope,
            policy=view.trial_recall_policy,
        )


class _ProfileRacingRuntimeApplicationFactory(_CanonicalRuntimeApplicationFactory):
    """Change one registered semantic after dry preflight but before admission."""

    async def create(
        self,
        *,
        request: MemoryInterventionTrialRequest,
        execution: MemoryInterventionExecutionRecord,
        trial: AgentSnapshotTrialBinding,
        operation: MemoryInterventionOperation,
        view: MemoryInterventionRuntimeView,
    ) -> object:
        app = await super().create(
            request=request,
            execution=execution,
            trial=trial,
            operation=operation,
            view=view,
        )
        assert type(app) is CayuApp
        engine = app._session_engine
        original_prepare = engine._prepare_initial_run
        prepare_calls = 0

        async def prepare_with_registration_race(*args, **kwargs):
            nonlocal prepare_calls
            prepare_calls += 1
            if prepare_calls == 2:
                registered = app._agents["agent"]
                app._agents["agent"] = replace(
                    registered,
                    spec=registered.spec.model_copy(update={"model": "raced-model"}),
                )
            return await original_prepare(*args, **kwargs)

        engine._prepare_initial_run = prepare_with_registration_race  # type: ignore[method-assign]
        return app


class _StoreRacingRuntimeApplicationFactory(_CanonicalRuntimeApplicationFactory):
    """Replace the selected store after dry validation but before admission."""

    def __init__(self, *, sessions: SessionStore, provider: ScriptedModelProvider) -> None:
        super().__init__(sessions=sessions, provider=provider)
        self.production_scope = KnowledgeAccessScope(allowed_namespaces=["default"])
        self.production_store = InMemoryKnowledgeStore(access_scope=self.production_scope)

    async def create(
        self,
        *,
        request: MemoryInterventionTrialRequest,
        execution: MemoryInterventionExecutionRecord,
        trial: AgentSnapshotTrialBinding,
        operation: MemoryInterventionOperation,
        view: MemoryInterventionRuntimeView,
    ) -> object:
        app = await super().create(
            request=request,
            execution=execution,
            trial=trial,
            operation=operation,
            view=view,
        )
        assert type(app) is CayuApp
        engine = app._session_engine
        original_prepare = engine._prepare_initial_run
        prepare_calls = 0

        async def prepare_with_store_race(*args, **kwargs):
            nonlocal prepare_calls
            prepared = await original_prepare(*args, **kwargs)
            prepare_calls += 1
            if prepare_calls == 1:
                registered = app._environments["intervention"]
                app._environments["intervention"] = replace(
                    registered,
                    environment=Environment(
                        registered.environment.spec,
                        knowledge_store=self.production_store,
                        knowledge_access_scope=self.production_scope,
                    ),
                )
            return prepared

        engine._prepare_initial_run = prepare_with_store_race  # type: ignore[method-assign]
        return app


class _BlockingRuntimeProvider(ScriptedModelProvider):
    """Cooperative provider used to deliver real timeout/cancellation signals."""

    def __init__(self) -> None:
        super().__init__(
            (
                ModelStreamEvent.text_delta("unused"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ),
            name="blocking-memory-intervention",
        )
        self.started = asyncio.Event()
        self.cancelled = asyncio.Event()
        self.release = asyncio.Event()

    @property
    def execution_profile_identity(self) -> ExecutionProfileBehaviorIdentity:
        return ExecutionProfileBehaviorIdentity(
            name="test:blocking-memory-intervention-provider",
            behavior_version="1",
            implementation_version="1",
        )

    async def stream(self, request: ModelRequest):
        self.requests.append(request)
        self.started.set()
        try:
            await self.release.wait()
        except asyncio.CancelledError:
            self.cancelled.set()
            raise
        yield ModelStreamEvent.text_delta("Friday")
        yield ModelStreamEvent.completed({"finish_reason": "stop"})


class _FailOnceTerminalEvidenceSQLiteSessionStore(SQLiteSessionStore):
    invocation_lifecycle_command_version = 1

    def __init__(self, path) -> None:
        super().__init__(path)
        self.failed_terminal_evidence = False

    async def load_terminal_session_evidence(self, session_id: str, *, limits=None):
        if not self.failed_terminal_evidence:
            self.failed_terminal_evidence = True
            raise ConnectionError("test terminal evidence read failed")
        return await super().load_terminal_session_evidence(session_id, limits=limits)


class _BlockOnceTerminalEvidenceSQLiteSessionStore(SQLiteSessionStore):
    invocation_lifecycle_command_version = 1

    def __init__(self, path) -> None:
        super().__init__(path)
        self.terminal_evidence_started = asyncio.Event()
        self.allow_terminal_evidence = asyncio.Event()
        self.blocked_terminal_evidence = False

    async def load_terminal_session_evidence(self, session_id: str, *, limits=None):
        if not self.blocked_terminal_evidence:
            self.blocked_terminal_evidence = True
            self.terminal_evidence_started.set()
            await self.allow_terminal_evidence.wait()
        return await super().load_terminal_session_evidence(session_id, limits=limits)


class _SubstitutingTerminalEvidenceSQLiteSessionStore(SQLiteSessionStore):
    invocation_lifecycle_command_version = 1

    def __init__(self, path) -> None:
        super().__init__(path)
        self.foreign_evidence: TerminalSessionEvidence | None = None
        self.forge_empty_events = False

    async def load_terminal_session_evidence(self, session_id: str, *, limits=None):
        evidence = await super().load_terminal_session_evidence(session_id, limits=limits)
        if self.foreign_evidence is not None:
            return self.foreign_evidence
        if self.forge_empty_events:
            return evidence.model_copy(update={"events": ()})
        return evidence


class _AdvancedTerminalSessionSQLiteSessionStore(SQLiteSessionStore):
    invocation_lifecycle_command_version = 1

    async def load(self, session_id: str):
        session = await super().load(session_id)
        if session is None or session.status not in {
            SessionStatus.COMPLETED,
            SessionStatus.FAILED,
            SessionStatus.INTERRUPTED,
        }:
            return session
        return session.model_copy(update={"run_epoch": session.run_epoch + 1})


class _CoherentlyAdvancedTerminalEvidenceSQLiteSessionStore(SQLiteSessionStore):
    invocation_lifecycle_command_version = 1

    def __init__(self, path) -> None:
        super().__init__(path)
        self.terminal_session_loads = 0

    async def load(self, session_id: str):
        session = await super().load(session_id)
        if session is not None and session.status in {
            SessionStatus.COMPLETED,
            SessionStatus.FAILED,
            SessionStatus.INTERRUPTED,
        }:
            self.terminal_session_loads += 1
        return session

    async def load_terminal_session_evidence(self, session_id: str, *, limits=None):
        evidence = await super().load_terminal_session_evidence(session_id, limits=limits)
        return evidence.model_copy(
            update={
                "session": evidence.session.model_copy(
                    update={"run_epoch": evidence.session.run_epoch + 1}
                ),
                "boundary": evidence.boundary.model_copy(
                    update={"run_epoch": evidence.boundary.run_epoch + 1}
                ),
            }
        )


class _UnavailableTerminalEvidenceSQLiteSessionStore(SQLiteSessionStore):
    invocation_lifecycle_command_version = 1

    def __init__(self, path, *, failure: str) -> None:
        super().__init__(path)
        self.failure = failure

    async def load_terminal_session_evidence(self, session_id: str, *, limits=None):
        await super().load_terminal_session_evidence(session_id, limits=limits)
        if self.failure == "unsupported":
            raise NotImplementedError("terminal evidence is unsupported")
        if self.failure == "typed":
            raise TerminalSessionEvidenceError(
                TerminalSessionEvidenceErrorCode.TOTAL_BYTES_EXCEEDED
            )
        raise OSError("terminal evidence read failed")


class _RecordingAttributionBoundsSQLiteSessionStore(SQLiteSessionStore):
    invocation_lifecycle_command_version = 1

    def __init__(self, path) -> None:
        super().__init__(path)
        self.recall_queries: list[RecallEvidenceQuery] = []
        self.exposure_queries: list[RecallEvidenceQuery] = []

    async def list_recall_receipts(self, query: RecallEvidenceQuery):
        self.recall_queries.append(RecallEvidenceQuery.model_validate(query.model_dump()))
        return await super().list_recall_receipts(query)

    async def list_context_exposures(self, query: RecallEvidenceQuery):
        self.exposure_queries.append(RecallEvidenceQuery.model_validate(query.model_dump()))
        return await super().list_context_exposures(query)


class _RecallOverlayProvider(_OverlayProvider):
    _REPLACEMENT_TEXT = _REPLACEMENT_FIXTURE_TEXT
    _NEGATIVE_CONTROL_TEXT = _NEGATIVE_CONTROL_FIXTURE_TEXT

    def __init__(
        self,
        views: dict[str, MemoryInterventionRuntimeView],
        *,
        entry_text: str,
    ) -> None:
        super().__init__(views, trial_policy_factory=_canonical_policy)
        self.entry_text = entry_text
        self.seeded_operations: set[str] = set()

    async def apply(
        self,
        *,
        spec: MemoryInterventionSpec,
        operation: MemoryInterventionOperation,
        materialization: AgentSnapshotMaterialization,
        trial: AgentSnapshotTrialBinding,
    ) -> MemoryInterventionRuntimeView:
        view = await super().apply(
            spec=spec,
            operation=operation,
            materialization=materialization,
            trial=trial,
        )
        await self._seed_view(spec=spec, operation=operation, view=view)
        return view

    async def recover(
        self,
        *,
        spec: MemoryInterventionSpec,
        operation: MemoryInterventionOperation,
        materialization: AgentSnapshotMaterialization,
        trial: AgentSnapshotTrialBinding,
    ) -> MemoryInterventionRuntimeView:
        view = await super().recover(
            spec=spec,
            operation=operation,
            materialization=materialization,
            trial=trial,
        )
        await self._seed_view(spec=spec, operation=operation, view=view)
        return view

    async def _seed_view(
        self,
        *,
        spec: MemoryInterventionSpec,
        operation: MemoryInterventionOperation,
        view: MemoryInterventionRuntimeView,
    ) -> None:
        if operation.operation_id not in self.seeded_operations:
            fixtures = {
                change.fixture.fixture_id: change.fixture
                for change in spec.changes
                if change.fixture is not None
            }

            def fixture_text(fixture_id: str, text: str) -> str:
                fixture = fixtures.get(fixture_id)
                encoded = text.encode()
                if fixture is None or (
                    fixture.fixture_fingerprint != hashlib.sha256(encoded).hexdigest()
                    or fixture.representation_fingerprint != _digest(f"utf8-text:{text}")
                    or fixture.size_bytes != len(encoded)
                ):
                    raise MemoryInterventionExecutionConflict(
                        "Intervention fixture differs from its immutable declaration."
                    )
                return text

            entries: tuple[str, ...]
            if spec.kind is MemoryInterventionKind.OMIT_ITEMS:
                entries = ()
            elif spec.kind is MemoryInterventionKind.REPLACE_ITEMS:
                entries = (fixture_text("fixture-replacement", self._REPLACEMENT_TEXT),)
            elif spec.kind is MemoryInterventionKind.NEGATIVE_CONTROL:
                entries = (
                    self.entry_text,
                    fixture_text(
                        "fixture-negative-control",
                        self._NEGATIVE_CONTROL_TEXT,
                    ),
                )
            else:
                entries = (self.entry_text,)
            for index, text in enumerate(entries):
                await view.knowledge_store.create_entry(
                    KnowledgeEntry(
                        id=f"entry-{operation.operation_id[:16]}-{index}",
                        namespace="default",
                        text=text,
                    ),
                    access_scope=view.knowledge_access_scope,
                )
            self.seeded_operations.add(operation.operation_id)


class _RevisionCheckingOverlayProvider(_OverlayProvider):
    def __init__(
        self,
        views: dict[str, MemoryInterventionRuntimeView],
        *,
        current_revision_fingerprint: str,
    ) -> None:
        super().__init__(views)
        self.current_revision_fingerprint = current_revision_fingerprint
        self.execution_profile_fingerprint = _digest(
            f"revision-checking-overlay:{current_revision_fingerprint}"
        )

    async def apply(
        self,
        *,
        spec: MemoryInterventionSpec,
        operation: MemoryInterventionOperation,
        materialization: AgentSnapshotMaterialization,
        trial: AgentSnapshotTrialBinding,
    ) -> MemoryInterventionRuntimeView:
        source_revisions = tuple(
            change.source_item.revision_fingerprint
            for change in spec.changes
            if change.source_item is not None
        )
        if self.current_revision_fingerprint not in source_revisions:
            self.terminal_status = MemoryInterventionEffectStatus.CONFLICTING
        return await super().apply(
            spec=spec,
            operation=operation,
            materialization=materialization,
            trial=trial,
        )


async def _record_runtime_profile(
    factory: _CanonicalRuntimeApplicationFactory,
    *,
    policy: AutomaticRecallPolicy,
    suffix: str,
) -> AgentSnapshotExecutionProfileRef:
    scope = KnowledgeAccessScope(allowed_namespaces=["default"])
    app = factory.build_app(
        knowledge_store=InMemoryKnowledgeStore(access_scope=scope),
        scope=scope,
        policy=policy,
    )
    prepared = await app._session_engine._prepare_initial_run(
        RunRequest(
            agent_name="agent",
            session_id=f"profile-{suffix}",
            causal_budget_id=f"profile-budget-{suffix}",
            messages=[Message.text("user", "When is Atlas released?")],
        ),
        admit_session=False,
    )
    assert prepared is not None
    profile = execution_profile_snapshot_ref(prepared.execution_profile)
    factory.profile_by_policy[policy.fingerprint()] = profile.fingerprint
    return profile


async def _canonical_execution_harness(
    tmp_path,
    *,
    provider: ScriptedModelProvider,
    suffix: str,
    timeout_seconds: int = 300,
    sessions: SessionStore | None = None,
    attribution_bounds: MemoryAttributionBounds | None = None,
    factory_type: type[_CanonicalRuntimeApplicationFactory] = (_CanonicalRuntimeApplicationFactory),
) -> tuple[
    MemoryInterventionExecutor,
    SQLiteMemoryInterventionExecutionStore,
    MemoryInterventionTrialRequest,
    AgentSnapshot,
]:
    session_store = sessions or SQLiteSessionStore(tmp_path / f"{suffix}-sessions.db")
    factory = factory_type(
        sessions=session_store,
        provider=provider,
    )
    policy = _canonical_policy()
    profile = await _record_runtime_profile(
        factory,
        policy=policy,
        suffix=suffix,
    )
    snapshot = _snapshot(
        execution_profile=profile,
        recall_policy=policy,
    )
    snapshot_store = SQLiteAgentSnapshotStore(tmp_path / f"{suffix}-snapshots.db")
    await snapshot_store.save_snapshot(snapshot)
    executions = SQLiteMemoryInterventionExecutionStore(tmp_path / f"{suffix}-executions.db")
    executor = MemoryInterventionExecutor(
        snapshots=AgentSnapshotCoordinator(
            _providers(snapshot, {}),
            store=snapshot_store,
            clock=lambda: datetime(2026, 8, 25, tzinfo=UTC),
        ),
        executions=executions,
        overlay_provider=_RecallOverlayProvider(
            {},
            entry_text="Atlas is released Friday.",
        ),
        runtime_runner=CayuMemoryInterventionRuntimeRunner(
            factory,
            attribution_bounds=attribution_bounds,
        ),
        evaluator=_Evaluator({}),
        request_keys={
            "test-key": MemoryInterventionRequestFingerprintKey(
                key_id="test-key",
                secret=b"k" * 32,
            )
        },
        current_request_key_id="test-key",
        clock=lambda: datetime(2026, 8, 25, tzinfo=UTC),
    )
    request = _request(
        _spec(snapshot, recall_policy=policy),
        trial_id=suffix,
        prompt="When is Atlas released?",
        timeout_seconds=timeout_seconds,
    )
    return executor, executions, request, snapshot


class _RuntimeRunner(MemoryInterventionRuntimeRunner):
    execution_profile_fingerprint = _digest("runtime-runner:v1")

    def __init__(
        self,
        results: dict[str, MemoryInterventionRuntimeResult],
        *,
        terminal_disposition: AgentSnapshotTerminalDisposition = (
            AgentSnapshotTerminalDisposition.COMPLETED
        ),
    ) -> None:
        self.results = results
        self.terminal_disposition = terminal_disposition
        self.attribution = _empty_attribution()
        self.terminal_evidence_available = True
        self.terminal_evidence_limitation: EvalMemoryEvidenceLimitation | None = None
        self.expected_receipt_count: int | None = None
        self.expected_exposure_count: int | None = None
        self.effective_attribution_bounds = standard_eval_memory_attribution_bounds()
        self.source_alias = None
        self.result_override: MemoryInterventionRuntimeResult | None = None
        self.raise_child_cancellation = False
        self.run_calls = 0
        self.recover_calls = 0

    async def run(
        self,
        *,
        request: MemoryInterventionTrialRequest,
        execution,
        starting_execution_profile: AgentSnapshotExecutionProfileRef,
        trial: AgentSnapshotTrialBinding,
        operation: MemoryInterventionOperation,
        view: MemoryInterventionRuntimeView,
    ) -> MemoryInterventionRuntimeResult:
        self.run_calls += 1
        if self.raise_child_cancellation:
            raise asyncio.CancelledError("runtime child cancelled itself")
        if self.result_override is not None:
            return self.result_override
        existing = self.results.get(operation.operation_id)
        if existing is not None:
            return existing
        result = MemoryInterventionRuntimeResult(
            session_id=execution.session_id,
            terminal_disposition=self.terminal_disposition,
            runtime_evidence_fingerprint=_digest(f"runtime:{operation.operation_id}"),
            terminal_evidence_available=self.terminal_evidence_available,
            terminal_evidence_limitation=(
                None
                if self.terminal_evidence_available
                else self.terminal_evidence_limitation or EvalMemoryEvidenceLimitation.MISSING
            ),
            expected_receipt_count=(
                self.attribution.observed_receipt_count
                if self.terminal_evidence_available and self.expected_receipt_count is None
                else self.expected_receipt_count
            ),
            expected_exposure_count=(
                self.attribution.observed_exposure_count
                if self.terminal_evidence_available and self.expected_exposure_count is None
                else self.expected_exposure_count
            ),
            effective_attribution_bounds=self.effective_attribution_bounds,
            source_alias=self.source_alias,
            attribution=self.attribution,
            usage_fingerprint=_digest(f"usage:{operation.operation_id}"),
            cost_fingerprint=_digest(f"cost:{operation.operation_id}"),
        )
        self.results[operation.operation_id] = result
        return result

    async def recover(
        self,
        *,
        request: MemoryInterventionTrialRequest,
        execution,
        starting_execution_profile: AgentSnapshotExecutionProfileRef,
        trial: AgentSnapshotTrialBinding,
        operation: MemoryInterventionOperation,
        view: MemoryInterventionRuntimeView,
    ) -> MemoryInterventionRuntimeResult:
        self.recover_calls += 1
        value = self.results.get(operation.operation_id)
        if value is None:
            value = await self.run(
                request=request,
                execution=execution,
                starting_execution_profile=starting_execution_profile,
                trial=trial,
                operation=operation,
                view=view,
            )
            self.run_calls -= 1
        return value


class _Evaluator(MemoryInterventionEvaluator):
    evaluator_fingerprint = _digest("evaluator")

    def __init__(self, results: dict[str, EvalTrialResult]) -> None:
        self.results = results
        self.evaluate_calls = 0
        self.recover_calls = 0

    async def evaluate(
        self,
        *,
        operation_id: str,
        case: EvalCaseContractV1,
        runtime: MemoryInterventionRuntimeResult,
    ) -> EvalTrialResult:
        self.evaluate_calls += 1
        existing = self.results.get(operation_id)
        if existing is not None:
            return existing
        now = datetime(2026, 8, 25, tzinfo=UTC)
        result = EvalTrialResult(
            trial_number=1,
            status=EvalStatus.PASSED,
            session_id=runtime.session_id,
            score=1.0,
            assertions=(
                EvalAssertionResult(
                    name="case-check",
                    outcome=EvalOutcome.PASSED,
                    score=1.0,
                    threshold=1.0,
                ),
            ),
            evidence_complete=True,
            usage_summary=SessionUsageSummary(session_id=runtime.session_id).model_dump(
                mode="json"
            ),
            started_at=now,
            completed_at=now,
        )
        self.results[operation_id] = result
        return result

    async def recover(
        self,
        *,
        operation_id: str,
        case: EvalCaseContractV1,
        runtime: MemoryInterventionRuntimeResult,
    ) -> EvalTrialResult:
        self.recover_calls += 1
        value = self.results.get(operation_id)
        if value is None:
            value = await self.evaluate(
                operation_id=operation_id,
                case=case,
                runtime=runtime,
            )
            self.evaluate_calls -= 1
        return value


async def _recover_intervention_in_fresh_process(
    *,
    snapshots_path: str,
    executions_path: str,
    sessions_path: str,
    request_payload: str,
) -> dict[str, object]:
    request = MemoryInterventionTrialRequest.model_validate_json(request_payload)
    snapshot_store = SQLiteAgentSnapshotStore(snapshots_path)
    snapshot = await snapshot_store.load_snapshot(request.spec.snapshot_fingerprint)
    if snapshot is None:
        raise AssertionError("The child process could not reconstruct the trial snapshot.")
    provider = ScriptedModelProvider(
        (
            ModelStreamEvent.text_delta("Friday"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        )
    )
    factory = _CanonicalRuntimeApplicationFactory(
        sessions=SQLiteSessionStore(sessions_path),
        provider=provider,
    )
    factory.profile_by_policy[request.spec.trial_recall_policy_fingerprint] = (
        request.spec.execution_profile_fingerprint
    )
    executor = MemoryInterventionExecutor(
        snapshots=AgentSnapshotCoordinator(
            _providers(snapshot, {}),
            store=snapshot_store,
            clock=lambda: datetime(2026, 8, 25, tzinfo=UTC),
        ),
        executions=SQLiteMemoryInterventionExecutionStore(executions_path),
        overlay_provider=_RecallOverlayProvider(
            {},
            entry_text="Atlas is released Friday.",
        ),
        runtime_runner=CayuMemoryInterventionRuntimeRunner(factory),
        evaluator=_Evaluator({}),
        request_keys={
            "test-key": MemoryInterventionRequestFingerprintKey(
                key_id="test-key",
                secret=b"k" * 32,
            )
        },
        current_request_key_id="test-key",
        clock=lambda: datetime(2026, 8, 25, tzinfo=UTC),
    )

    outcome = await executor.execute_trial(request)
    return {
        "phase": outcome.execution.phase.value,
        "status": outcome.execution.status.value,
        "request_count": len(provider.requests),
        "recalled_fixture": (
            len(provider.requests) == 1
            and "Atlas is released Friday" in provider.requests[0].model_dump_json()
        ),
        "observed_item_count": (
            None if outcome.binding is None else outcome.binding.attribution.observed_item_count
        ),
    }


def _fresh_process_recovery_entrypoint(
    snapshots_path: str,
    executions_path: str,
    sessions_path: str,
    request_payload: str,
    connection: Connection,
) -> None:
    try:
        connection.send(
            asyncio.run(
                _recover_intervention_in_fresh_process(
                    snapshots_path=snapshots_path,
                    executions_path=executions_path,
                    sessions_path=sessions_path,
                    request_payload=request_payload,
                )
            )
        )
    finally:
        connection.close()


async def _executor(
    snapshot: AgentSnapshot,
    *,
    snapshot_store,
    execution_store,
    materializations: dict[str, AgentSnapshotMaterializedComponent],
    views: dict[str, MemoryInterventionRuntimeView],
    runtime_results: dict[str, MemoryInterventionRuntimeResult],
    eval_results: dict[str, EvalTrialResult],
    terminal_status: MemoryInterventionEffectStatus | None = None,
    overlay_profile_fingerprint: str | None = None,
    runtime_terminal_disposition: AgentSnapshotTerminalDisposition = (
        AgentSnapshotTerminalDisposition.COMPLETED
    ),
) -> tuple[MemoryInterventionExecutor, _OverlayProvider, _RuntimeRunner, _Evaluator]:
    coordinator = AgentSnapshotCoordinator(
        _providers(snapshot, materializations),
        store=snapshot_store,
        clock=lambda: datetime(2026, 8, 25, tzinfo=UTC),
    )
    if await snapshot_store.load_snapshot(snapshot.fingerprint) is None:
        await snapshot_store.save_snapshot(snapshot)
    overlay = _OverlayProvider(
        views,
        terminal_status=terminal_status,
        profile_fingerprint=overlay_profile_fingerprint,
    )
    runner = _RuntimeRunner(
        runtime_results,
        terminal_disposition=runtime_terminal_disposition,
    )
    evaluator = _Evaluator(eval_results)
    executor = MemoryInterventionExecutor(
        snapshots=coordinator,
        executions=execution_store,
        overlay_provider=overlay,
        runtime_runner=runner,
        evaluator=evaluator,
        request_keys={
            "test-key": MemoryInterventionRequestFingerprintKey(
                key_id="test-key",
                secret=b"k" * 32,
            )
        },
        current_request_key_id="test-key",
        clock=lambda: datetime(2026, 8, 25, tzinfo=UTC),
    )
    return executor, overlay, runner, evaluator


class _CommitThenFailExecutionStore(MemoryInterventionExecutionStore):
    def __init__(
        self,
        delegate: MemoryInterventionExecutionStore,
        *,
        fail_phase: MemoryInterventionExecutionPhase,
    ) -> None:
        self.delegate = delegate
        self.fail_phase = fail_phase
        self.failed = False

    async def begin(
        self,
        record: MemoryInterventionExecutionRecord,
    ) -> MemoryInterventionExecutionRecord:
        return await self.delegate.begin(record)

    async def load(self, execution_id: str) -> MemoryInterventionExecutionRecord | None:
        return await self.delegate.load(execution_id)

    async def compare_and_set(
        self,
        expected: MemoryInterventionExecutionRecord,
        desired: MemoryInterventionExecutionRecord,
    ) -> MemoryInterventionExecutionRecord:
        committed = await self.delegate.compare_and_set(expected, desired)
        if not self.failed and desired.phase is self.fail_phase:
            self.failed = True
            raise ConnectionError("test acknowledgement lost after exact commit")
        return committed


class _PausingCancellationAuthorityStore(MemoryInterventionExecutionStore):
    def __init__(self, delegate: MemoryInterventionExecutionStore) -> None:
        self.delegate = delegate
        self.publication_started = asyncio.Event()
        self.allow_publication = asyncio.Event()

    async def begin(
        self,
        record: MemoryInterventionExecutionRecord,
    ) -> MemoryInterventionExecutionRecord:
        return await self.delegate.begin(record)

    async def load(self, execution_id: str) -> MemoryInterventionExecutionRecord | None:
        return await self.delegate.load(execution_id)

    async def compare_and_set(
        self,
        expected: MemoryInterventionExecutionRecord,
        desired: MemoryInterventionExecutionRecord,
    ) -> MemoryInterventionExecutionRecord:
        if not expected.runtime_cancellation_observed and desired.runtime_cancellation_observed:
            self.publication_started.set()
            await self.allow_publication.wait()
        return await self.delegate.compare_and_set(expected, desired)


@pytest.mark.parametrize("backend", ("memory", "sqlite"))
@_async_test
async def test_execution_stores_reject_phase_skips_and_committed_evidence_rewrites(
    tmp_path,
    backend: str,
) -> None:
    snapshot = _snapshot()
    request = _request(_spec(snapshot))
    prepared = _prepared_record(request)
    store: MemoryInterventionExecutionStore
    if backend == "memory":
        store = InMemoryMemoryInterventionExecutionStore()
    else:
        store = SQLiteMemoryInterventionExecutionStore(tmp_path / "executions.db")
    prepared = await store.begin(prepared)
    skipped = _record_successor(
        prepared,
        phase=MemoryInterventionExecutionPhase.EFFECT_RESOLVED,
        materialization_fingerprint=_digest("materialization"),
        trial_binding_fingerprint=_digest("trial"),
        operation_fingerprint=_digest("operation"),
        receipt_fingerprint=_digest("receipt"),
    )

    with pytest.raises(MemoryInterventionExecutionConflict, match="exactly one phase"):
        await store.compare_and_set(prepared, skipped)

    trial_bound = _record_successor(
        prepared,
        phase=MemoryInterventionExecutionPhase.TRIAL_BOUND,
        materialization_fingerprint=_digest("materialization"),
        trial_binding_fingerprint=_digest("trial"),
        operation_fingerprint=_digest("operation"),
    )
    trial_bound = await store.compare_and_set(prepared, trial_bound)
    rewritten = _record_successor(
        trial_bound,
        phase=MemoryInterventionExecutionPhase.EFFECT_RESOLVED,
        materialization_fingerprint=_digest("changed-materialization"),
        receipt_fingerprint=_digest("receipt"),
    )

    with pytest.raises(MemoryInterventionExecutionConflict, match="cannot change"):
        await store.compare_and_set(trial_bound, rewritten)


@_async_test
async def test_sqlite_store_rejects_indexed_revision_document_disagreement(tmp_path) -> None:
    path = tmp_path / "executions.db"
    request = _request(_spec(_snapshot()))
    store = SQLiteMemoryInterventionExecutionStore(path)
    prepared = await store.begin(_prepared_record(request))
    with sqlite3.connect(path) as connection:
        connection.execute(
            "UPDATE cayu_memory_intervention_executions SET revision = ? WHERE execution_id = ?",
            (prepared.revision + 1, prepared.execution_id),
        )

    with pytest.raises(MemoryInterventionExecutionConflict, match="durable document"):
        await store.load(prepared.execution_id)


@_async_test
async def test_executor_finalizes_exact_lineage_and_replays_without_redispatch() -> None:
    snapshot = _snapshot()
    executor, overlay, runner, evaluator = await _executor(
        snapshot,
        snapshot_store=AgentSnapshotCoordinator(_providers(snapshot, {})).store,
        execution_store=InMemoryMemoryInterventionExecutionStore(),
        materializations={},
        views={},
        runtime_results={},
        eval_results={},
    )
    request = _request(_spec(snapshot))

    first = await executor.execute_trial(request)
    replay = await executor.execute_trial(request)

    assert first == replay
    assert first.execution.phase is MemoryInterventionExecutionPhase.FINALIZED
    assert first.execution.status is MemoryInterventionExecutionStatus.COMPLETED
    assert first.execution.session_id == request.session_id
    assert first.binding is not None
    assert first.binding.result.memory_evidence_fingerprint == first.binding.attribution_fingerprint
    assert overlay.apply_calls == 1
    assert runner.run_calls == 1
    assert evaluator.evaluate_calls == 1
    assert overlay.recover_calls == 1
    assert runner.recover_calls == 0
    assert evaluator.recover_calls == 1
    view = next(iter(overlay.views.values()))
    with pytest.raises(ValueError, match="conflicts with its effect receipt"):
        MemoryInterventionRuntimeView(
            materialization_fingerprint=view.materialization_fingerprint,
            memory_overlay_fingerprint=view.memory_overlay_fingerprint,
            state_scope_id=view.state_scope_id,
            knowledge_store=view.knowledge_store,
            knowledge_access_scope=view.knowledge_access_scope,
            isolation_authority=MemoryInterventionIsolationAuthority(
                materialization_fingerprint=view.materialization_fingerprint,
                memory_overlay_fingerprint=_digest("another-overlay"),
                state_scope_id=view.state_scope_id,
            ),
            trial_recall_policy=view.trial_recall_policy,
            receipt=view.receipt,
        )


@_async_test
async def test_concrete_runner_uses_canonical_recall_and_keeps_production_store_isolated(
    tmp_path,
) -> None:
    prompt = "When is Atlas released?"
    sessions_path = tmp_path / "runtime-sessions.db"
    snapshots_path = tmp_path / "runtime-snapshots.db"
    executions_path = tmp_path / "runtime-executions.db"
    sessions = SQLiteSessionStore(sessions_path)
    provider = ScriptedModelProvider(
        (
            (
                ModelStreamEvent.text_delta("Friday"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ),
            (
                ModelStreamEvent.text_delta("Friday"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ),
            (
                ModelStreamEvent.text_delta("No recalled answer"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ),
            (
                ModelStreamEvent.text_delta("Saturday"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ),
            (
                ModelStreamEvent.text_delta("Friday"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ),
        )
    )
    factory = _CanonicalRuntimeApplicationFactory(
        sessions=sessions,
        provider=provider,
    )
    baseline_policy = _canonical_policy()
    off_policy = _canonical_policy(AutomaticRecallMode.OFF)
    baseline_profile = await _record_runtime_profile(
        factory,
        policy=baseline_policy,
        suffix="baseline",
    )
    await _record_runtime_profile(
        factory,
        policy=off_policy,
        suffix="off",
    )
    snapshot = _snapshot(
        execution_profile=baseline_profile,
        recall_policy=baseline_policy,
    )
    snapshot_store = SQLiteAgentSnapshotStore(snapshots_path)
    coordinator = AgentSnapshotCoordinator(
        _providers(snapshot, {}),
        store=snapshot_store,
        clock=lambda: datetime(2026, 8, 25, tzinfo=UTC),
    )
    await snapshot_store.save_snapshot(snapshot)
    production_scope = KnowledgeAccessScope(allowed_namespaces=["default"])
    production_store = InMemoryKnowledgeStore(access_scope=production_scope)
    await production_store.create_entry(
        KnowledgeEntry(
            id="production-only",
            namespace="default",
            text="Production state must remain unchanged.",
        ),
        access_scope=production_scope,
    )
    production_before = await production_store.list_entries(
        KnowledgeListQuery(namespace="default"),
        access_scope=production_scope,
    )
    overlay = _RecallOverlayProvider(
        {},
        entry_text="Atlas is released Friday.",
    )
    runner = CayuMemoryInterventionRuntimeRunner(factory)
    evaluator = _Evaluator({})
    executor = MemoryInterventionExecutor(
        snapshots=coordinator,
        executions=SQLiteMemoryInterventionExecutionStore(executions_path),
        overlay_provider=overlay,
        runtime_runner=runner,
        evaluator=evaluator,
        request_keys={
            "test-key": MemoryInterventionRequestFingerprintKey(
                key_id="test-key",
                secret=b"k" * 32,
            )
        },
        current_request_key_id="test-key",
        clock=lambda: datetime(2026, 8, 25, tzinfo=UTC),
    )

    recalled = await executor.execute_trial(
        _request(
            _spec(snapshot, recall_policy=baseline_policy),
            trial_id="recalled",
            prompt=prompt,
        )
    )
    recall_off = await executor.execute_trial(
        _request(
            _automatic_recall_off_spec(
                snapshot,
                starting_policy=baseline_policy,
                trial_policy=off_policy,
            ),
            trial_id="recall-off",
            prompt=prompt,
        )
    )
    omitted = await executor.execute_trial(
        _request(
            _omit_spec(snapshot, recall_policy=baseline_policy),
            trial_id="omitted",
            prompt=prompt,
        )
    )
    replaced = await executor.execute_trial(
        _request(
            _replacement_spec(snapshot, recall_policy=baseline_policy),
            trial_id="replaced",
            prompt=prompt,
        )
    )
    negative_control = await executor.execute_trial(
        _request(
            _negative_control_spec(snapshot, recall_policy=baseline_policy),
            trial_id="negative-control",
            prompt=prompt,
        )
    )

    assert recalled.execution.status is MemoryInterventionExecutionStatus.COMPLETED
    assert recalled.binding is not None
    assert recalled.binding.attribution.observed_receipt_count == 1
    assert recalled.binding.attribution.observed_exposure_count == 1
    assert recall_off.execution.status is MemoryInterventionExecutionStatus.COMPLETED
    assert recall_off.binding is not None
    assert recall_off.binding.proves_no_memory_exposure
    assert omitted.binding is not None
    assert omitted.binding.attribution.observed_item_count == 0
    assert replaced.binding is not None
    assert replaced.binding.attribution.observed_item_count >= 1
    assert negative_control.binding is not None
    assert negative_control.binding.attribution.observed_item_count >= 2
    assert len(provider.requests) == 5
    provider_payloads = tuple(request.model_dump_json() for request in provider.requests)
    assert "Atlas is released Friday" in provider_payloads[0]
    assert "Atlas is released Friday" not in provider_payloads[1]
    assert "Atlas is released Friday" not in provider_payloads[2]
    assert _RecallOverlayProvider._REPLACEMENT_TEXT in provider_payloads[3]
    assert "Atlas is released Friday" not in provider_payloads[3]
    assert "Atlas is released Friday" in provider_payloads[4]
    assert _RecallOverlayProvider._NEGATIVE_CONTROL_TEXT in provider_payloads[4]
    production_after = await production_store.list_entries(
        KnowledgeListQuery(namespace="default"),
        access_scope=production_scope,
    )
    assert production_after == production_before
    assert all(view.knowledge_store is not production_store for view in overlay.views.values())

    restarted_factory = _CanonicalRuntimeApplicationFactory(
        sessions=SQLiteSessionStore(sessions_path),
        provider=provider,
    )
    restarted_factory.profile_by_policy.update(factory.profile_by_policy)
    restarted_overlay = _RecallOverlayProvider(
        {},
        entry_text="Atlas is released Friday.",
    )
    restarted = MemoryInterventionExecutor(
        snapshots=AgentSnapshotCoordinator(
            _providers(snapshot, {}),
            store=SQLiteAgentSnapshotStore(snapshots_path),
            clock=lambda: datetime(2026, 8, 25, tzinfo=UTC),
        ),
        executions=SQLiteMemoryInterventionExecutionStore(executions_path),
        overlay_provider=restarted_overlay,
        runtime_runner=CayuMemoryInterventionRuntimeRunner(restarted_factory),
        evaluator=_Evaluator({}),
        request_keys={
            "test-key": MemoryInterventionRequestFingerprintKey(
                key_id="test-key",
                secret=b"k" * 32,
            )
        },
        current_request_key_id="test-key",
        clock=lambda: datetime(2026, 8, 25, tzinfo=UTC),
    )
    replay = await restarted.execute_trial(
        _request(
            _spec(snapshot, recall_policy=baseline_policy),
            trial_id="recalled",
            prompt=prompt,
        )
    )

    assert replay == recalled
    assert len(provider.requests) == 5


@_async_test
async def test_concrete_runner_records_a_real_runtime_deadline_as_timed_out(tmp_path) -> None:
    provider = _BlockingRuntimeProvider()
    executor, executions, request, _ = await _canonical_execution_harness(
        tmp_path,
        provider=provider,
        suffix="timeout",
        timeout_seconds=1,
    )

    outcome = await executor.execute_trial(request)

    assert provider.started.is_set()
    assert provider.cancelled.is_set()
    assert len(provider.requests) == 1
    assert outcome.execution.phase is MemoryInterventionExecutionPhase.FINALIZED
    assert outcome.execution.status is MemoryInterventionExecutionStatus.TIMED_OUT
    assert outcome.execution.runtime_timeout_observed is True
    assert outcome.execution.failure_code == "runtime_timed_out"
    assert outcome.snapshot_result is not None
    assert (
        outcome.snapshot_result.terminal_disposition is AgentSnapshotTerminalDisposition.TIMED_OUT
    )
    assert await executions.load(request.execution_id) == outcome.execution


@_async_test
async def test_timeout_result_survives_lost_journal_acknowledgement_without_redispatch(
    tmp_path,
) -> None:
    provider = _BlockingRuntimeProvider()
    executor, executions, request, _ = await _canonical_execution_harness(
        tmp_path,
        provider=provider,
        suffix="timeout-ack-lost",
        timeout_seconds=1,
    )
    executor.executions = _CommitThenFailExecutionStore(
        executions,
        fail_phase=MemoryInterventionExecutionPhase.RUNTIME_TERMINAL,
    )

    with pytest.raises(ConnectionError, match="acknowledgement lost"):
        await executor.execute_trial(request)

    committed = await executions.load(request.execution_id)
    assert committed is not None
    assert committed.phase is MemoryInterventionExecutionPhase.RUNTIME_TERMINAL
    assert committed.runtime_timeout_observed is True
    assert committed.runtime_result_payload is not None
    assert (
        committed.runtime_result_payload["terminal_disposition"]
        == AgentSnapshotTerminalDisposition.TIMED_OUT.value
    )

    recovered = await executor.execute_trial(request)

    assert len(provider.requests) == 1
    assert recovered.execution.phase is MemoryInterventionExecutionPhase.FINALIZED
    assert recovered.execution.status is MemoryInterventionExecutionStatus.TIMED_OUT
    assert recovered.execution.failure_code == "runtime_timed_out"


@_async_test
async def test_timeout_authority_survives_terminal_evidence_failure_without_redispatch(
    tmp_path,
) -> None:
    provider = _BlockingRuntimeProvider()
    sessions = _FailOnceTerminalEvidenceSQLiteSessionStore(
        tmp_path / "timeout-evidence-failure-sessions.db"
    )
    executor, executions, request, snapshot = await _canonical_execution_harness(
        tmp_path,
        provider=provider,
        suffix="timeout-evidence-failure",
        timeout_seconds=1,
        sessions=sessions,
    )

    initial = await executor.execute_trial(request)

    interrupted = await executions.load(request.execution_id)
    assert interrupted is not None
    assert interrupted.phase is MemoryInterventionExecutionPhase.FINALIZED
    assert interrupted.status is MemoryInterventionExecutionStatus.TIMED_OUT
    assert interrupted.runtime_timeout_observed is True
    assert interrupted.runtime_result_payload is not None
    assert (
        interrupted.runtime_result_payload["terminal_evidence_limitation"]
        == EvalMemoryEvidenceLimitation.EVIDENCE_READ_FAILED.value
    )
    assert initial.eval_result is not None
    assert initial.eval_result.memory_attribution.completeness is (
        EvalMemoryEvidenceCompleteness.UNAVAILABLE
    )
    assert initial.eval_result.memory_attribution.limitations == ()
    assert initial.eval_result.memory_attribution.sources[0].limitations == (
        EvalMemoryEvidenceLimitation.EVIDENCE_READ_FAILED,
    )

    restarted_provider = _BlockingRuntimeProvider()
    restarted_factory = _CanonicalRuntimeApplicationFactory(
        sessions=SQLiteSessionStore(tmp_path / "timeout-evidence-failure-sessions.db"),
        provider=restarted_provider,
    )
    policy = _canonical_policy()
    restarted_factory.profile_by_policy[policy.fingerprint()] = (
        snapshot.execution_profile.fingerprint
    )
    restarted = MemoryInterventionExecutor(
        snapshots=AgentSnapshotCoordinator(
            _providers(snapshot, {}),
            store=SQLiteAgentSnapshotStore(tmp_path / "timeout-evidence-failure-snapshots.db"),
            clock=lambda: datetime(2026, 8, 25, tzinfo=UTC),
        ),
        executions=SQLiteMemoryInterventionExecutionStore(
            tmp_path / "timeout-evidence-failure-executions.db"
        ),
        overlay_provider=_RecallOverlayProvider(
            {},
            entry_text="Atlas is released Friday.",
        ),
        runtime_runner=CayuMemoryInterventionRuntimeRunner(restarted_factory),
        evaluator=_Evaluator({}),
        request_keys={
            "test-key": MemoryInterventionRequestFingerprintKey(
                key_id="test-key",
                secret=b"k" * 32,
            )
        },
        current_request_key_id="test-key",
        clock=lambda: datetime(2026, 8, 25, tzinfo=UTC),
    )
    recovered = await restarted.execute_trial(request)

    assert len(provider.requests) == 1
    assert restarted_provider.requests == []
    assert recovered.execution.phase is MemoryInterventionExecutionPhase.FINALIZED
    assert recovered.execution.status is MemoryInterventionExecutionStatus.TIMED_OUT
    assert recovered.execution.runtime_timeout_observed is True
    assert recovered.eval_result == initial.eval_result


@_async_test
async def test_timeout_authority_survives_cancellation_during_terminal_evidence(
    tmp_path,
) -> None:
    provider = _BlockingRuntimeProvider()
    sessions = _BlockOnceTerminalEvidenceSQLiteSessionStore(
        tmp_path / "timeout-evidence-cancellation-sessions.db"
    )
    executor, executions, request, _ = await _canonical_execution_harness(
        tmp_path,
        provider=provider,
        suffix="timeout-evidence-cancellation",
        # The absolute deadline includes canonical preflight. Leave enough
        # headroom for a loaded CI worker so this test reaches provider
        # dispatch before exercising cancellation during terminal readback.
        timeout_seconds=5,
        sessions=sessions,
    )
    task = asyncio.create_task(executor.execute_trial(request))
    await sessions.terminal_evidence_started.wait()

    task.cancel("cancel terminal evidence collection")
    await asyncio.sleep(0)
    sessions.allow_terminal_evidence.set()
    with pytest.raises(asyncio.CancelledError, match="cancel terminal evidence collection"):
        await task

    assert task.cancelling() == 1
    assert task.cancelled()
    interrupted = await executions.load(request.execution_id)
    assert interrupted is not None
    assert interrupted.phase is MemoryInterventionExecutionPhase.SESSION_BOUND
    assert interrupted.runtime_timeout_observed is True
    assert interrupted.runtime_cancellation_observed is True

    recovered = await executor.execute_trial(request)

    assert len(provider.requests) == 1
    assert recovered.execution.status is MemoryInterventionExecutionStatus.TIMED_OUT
    assert recovered.execution.runtime_timeout_observed is True


@_async_test
async def test_expired_durable_deadline_prevents_first_runtime_dispatch(tmp_path) -> None:
    provider = ScriptedModelProvider(
        (
            ModelStreamEvent.text_delta("must not dispatch"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        )
    )
    executor, executions, request, _ = await _canonical_execution_harness(
        tmp_path,
        provider=provider,
        suffix="expired-before-dispatch",
        timeout_seconds=1,
    )
    executor.executions = _CommitThenFailExecutionStore(
        executions,
        fail_phase=MemoryInterventionExecutionPhase.SESSION_BOUND,
    )

    with pytest.raises(ConnectionError, match="acknowledgement lost"):
        await executor.execute_trial(request)

    bound = await executions.load(request.execution_id)
    assert bound is not None
    deadline = bound.runtime_deadline_at
    assert deadline is not None
    executor.executions = executions
    executor._runtime_clock = lambda: deadline + timedelta(seconds=1)

    outcome = await executor.execute_trial(request)

    assert provider.requests == []
    assert outcome.execution.status is MemoryInterventionExecutionStatus.TIMED_OUT
    assert outcome.execution.runtime_timeout_observed is True


@_async_test
async def test_recorded_timeout_with_missing_session_never_redispatches(tmp_path) -> None:
    provider = ScriptedModelProvider(
        (
            ModelStreamEvent.text_delta("must not dispatch"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        )
    )
    executor, executions, request, _ = await _canonical_execution_harness(
        tmp_path,
        provider=provider,
        suffix="recorded-timeout-no-session",
    )
    executor.executions = _CommitThenFailExecutionStore(
        executions,
        fail_phase=MemoryInterventionExecutionPhase.SESSION_BOUND,
    )

    with pytest.raises(ConnectionError, match="acknowledgement lost"):
        await executor.execute_trial(request)

    bound = await executions.load(request.execution_id)
    assert bound is not None
    executor.executions = executions
    timed_out = await executor._record_runtime_timeout(bound)
    assert timed_out.runtime_timeout_observed is True

    outcome = await executor.execute_trial(request)

    assert provider.requests == []
    assert outcome.execution.status is MemoryInterventionExecutionStatus.TIMED_OUT


@_async_test
async def test_fresh_runtime_reconstructs_the_overlay_before_first_dispatch(tmp_path) -> None:
    first_provider = ScriptedModelProvider(
        (
            ModelStreamEvent.text_delta("unused"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        )
    )
    first, executions, request, snapshot = await _canonical_execution_harness(
        tmp_path,
        provider=first_provider,
        suffix="fresh-recovery",
    )
    first.executions = _CommitThenFailExecutionStore(
        executions,
        fail_phase=MemoryInterventionExecutionPhase.SESSION_BOUND,
    )

    with pytest.raises(ConnectionError, match="acknowledgement lost"):
        await first.execute_trial(request)

    assert first_provider.requests == []
    restarted_provider = ScriptedModelProvider(
        (
            ModelStreamEvent.text_delta("Friday"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        )
    )
    restarted_factory = _CanonicalRuntimeApplicationFactory(
        sessions=SQLiteSessionStore(tmp_path / "fresh-recovery-sessions.db"),
        provider=restarted_provider,
    )
    restarted_factory.profile_by_policy[request.spec.trial_recall_policy_fingerprint] = (
        request.spec.execution_profile_fingerprint
    )
    restarted = MemoryInterventionExecutor(
        snapshots=AgentSnapshotCoordinator(
            _providers(snapshot, {}),
            store=SQLiteAgentSnapshotStore(tmp_path / "fresh-recovery-snapshots.db"),
            clock=lambda: datetime(2026, 8, 25, tzinfo=UTC),
        ),
        executions=SQLiteMemoryInterventionExecutionStore(
            tmp_path / "fresh-recovery-executions.db"
        ),
        overlay_provider=_RecallOverlayProvider(
            {},
            entry_text="Atlas is released Friday.",
        ),
        runtime_runner=CayuMemoryInterventionRuntimeRunner(restarted_factory),
        evaluator=_Evaluator({}),
        request_keys={
            "test-key": MemoryInterventionRequestFingerprintKey(
                key_id="test-key",
                secret=b"k" * 32,
            )
        },
        current_request_key_id="test-key",
        clock=lambda: datetime(2026, 8, 25, tzinfo=UTC),
    )

    outcome = await restarted.execute_trial(request)

    assert outcome.execution.status is MemoryInterventionExecutionStatus.COMPLETED
    assert outcome.binding is not None
    assert outcome.binding.attribution.observed_item_count >= 1
    assert len(restarted_provider.requests) == 1
    assert "Atlas is released Friday" in restarted_provider.requests[0].model_dump_json()


@_async_test
async def test_recovery_rejects_foreign_terminal_session_with_predicted_id(tmp_path) -> None:
    intervention_provider = ScriptedModelProvider(
        (
            ModelStreamEvent.text_delta("intervention"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        )
    )
    executor, executions, request, _ = await _canonical_execution_harness(
        tmp_path,
        provider=intervention_provider,
        suffix="foreign-session-collision",
    )
    executor.executions = _CommitThenFailExecutionStore(
        executions,
        fail_phase=MemoryInterventionExecutionPhase.SESSION_BOUND,
    )

    with pytest.raises(ConnectionError, match="acknowledgement lost"):
        await executor.execute_trial(request)

    runner = executor.runtime_runner
    assert type(runner) is CayuMemoryInterventionRuntimeRunner
    factory = runner.factory
    assert type(factory) is _CanonicalRuntimeApplicationFactory
    foreign_provider = ScriptedModelProvider(
        (
            ModelStreamEvent.text_delta("foreign"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        )
    )
    foreign_factory = _CanonicalRuntimeApplicationFactory(
        sessions=factory.sessions,
        provider=foreign_provider,
    )
    foreign_scope = KnowledgeAccessScope(allowed_namespaces=["default"])
    foreign_app = foreign_factory.build_app(
        knowledge_store=InMemoryKnowledgeStore(access_scope=foreign_scope),
        scope=foreign_scope,
        policy=_canonical_policy(),
    )
    _ = [
        event
        async for event in foreign_app.run(
            RunRequest(
                agent_name="agent",
                session_id=request.session_id,
                causal_budget_id="foreign-budget",
                messages=[Message.text("user", "foreign workload")],
            )
        )
    ]
    executor.executions = executions

    with pytest.raises(
        MemoryInterventionExecutionConflict,
        match="does not match its intervention create claim",
    ):
        await executor.execute_trial(request)

    assert len(foreign_provider.requests) == 1
    assert intervention_provider.requests == []


@_async_test
async def test_recovery_rejects_foreign_running_session_with_predicted_id(tmp_path) -> None:
    intervention_provider = ScriptedModelProvider(
        (
            ModelStreamEvent.text_delta("intervention"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        )
    )
    executor, executions, request, _ = await _canonical_execution_harness(
        tmp_path,
        provider=intervention_provider,
        suffix="foreign-running-session-collision",
    )
    executor.executions = _CommitThenFailExecutionStore(
        executions,
        fail_phase=MemoryInterventionExecutionPhase.SESSION_BOUND,
    )
    with pytest.raises(ConnectionError, match="acknowledgement lost"):
        await executor.execute_trial(request)

    runner = executor.runtime_runner
    assert type(runner) is CayuMemoryInterventionRuntimeRunner
    factory = runner.factory
    assert type(factory) is _CanonicalRuntimeApplicationFactory
    foreign_provider = _BlockingRuntimeProvider()
    foreign_factory = _CanonicalRuntimeApplicationFactory(
        sessions=factory.sessions,
        provider=foreign_provider,
    )
    foreign_scope = KnowledgeAccessScope(allowed_namespaces=["default"])
    foreign_app = foreign_factory.build_app(
        knowledge_store=InMemoryKnowledgeStore(access_scope=foreign_scope),
        scope=foreign_scope,
        policy=_canonical_policy(),
    )

    async def run_foreign_session() -> list[object]:
        return [
            event
            async for event in foreign_app.run(
                RunRequest(
                    agent_name="agent",
                    session_id=request.session_id,
                    causal_budget_id="foreign-budget",
                    messages=[Message.text("user", "foreign workload")],
                )
            )
        ]

    foreign_task = asyncio.create_task(run_foreign_session())
    await foreign_provider.started.wait()
    executor.executions = executions
    try:
        with pytest.raises(
            MemoryInterventionExecutionConflict,
            match="does not match its intervention create claim",
        ):
            await executor.execute_trial(request)
    finally:
        foreign_provider.release.set()
        await foreign_task

    assert len(foreign_provider.requests) == 1
    assert intervention_provider.requests == []


@_async_test
async def test_spawned_process_recovers_exact_sqlite_trial_without_duplicate_dispatch(
    tmp_path,
) -> None:
    suffix = "spawned-process-recovery"
    first_provider = ScriptedModelProvider(
        (
            ModelStreamEvent.text_delta("unused"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        )
    )
    first, executions, request, _ = await _canonical_execution_harness(
        tmp_path,
        provider=first_provider,
        suffix=suffix,
    )
    first.executions = _CommitThenFailExecutionStore(
        executions,
        fail_phase=MemoryInterventionExecutionPhase.SESSION_BOUND,
    )

    with pytest.raises(ConnectionError, match="acknowledgement lost"):
        await first.execute_trial(request)

    committed = await executions.load(request.execution_id)
    assert committed is not None
    assert committed.phase is MemoryInterventionExecutionPhase.SESSION_BOUND
    assert first_provider.requests == []

    process_context = multiprocessing.get_context("spawn")
    receiving_connection, sending_connection = process_context.Pipe(duplex=False)
    process = process_context.Process(
        target=_fresh_process_recovery_entrypoint,
        args=(
            str(tmp_path / f"{suffix}-snapshots.db"),
            str(tmp_path / f"{suffix}-executions.db"),
            str(tmp_path / f"{suffix}-sessions.db"),
            request.model_dump_json(),
            sending_connection,
        ),
    )
    try:
        process.start()
        sending_connection.close()
        await asyncio.to_thread(process.join, 30)
        if process.is_alive():
            process.terminate()
            await asyncio.to_thread(process.join, 5)
            pytest.fail("Fresh-process memory intervention recovery did not terminate.")
        assert process.exitcode == 0
        assert receiving_connection.poll(5)
        child_result = receiving_connection.recv()
    finally:
        receiving_connection.close()
        sending_connection.close()

    assert child_result == {
        "phase": MemoryInterventionExecutionPhase.FINALIZED.value,
        "status": MemoryInterventionExecutionStatus.COMPLETED.value,
        "request_count": 1,
        "recalled_fixture": True,
        "observed_item_count": 2,
    }
    recovered = await executions.load(request.execution_id)
    assert recovered is not None
    assert recovered.phase is MemoryInterventionExecutionPhase.FINALIZED
    assert recovered.status is MemoryInterventionExecutionStatus.COMPLETED


@_async_test
async def test_concrete_runner_preserves_real_cancellation_and_recovers_without_redispatch(
    tmp_path,
) -> None:
    provider = _BlockingRuntimeProvider()
    executor, executions, request, _ = await _canonical_execution_harness(
        tmp_path,
        provider=provider,
        suffix="cancelled",
    )
    task = asyncio.create_task(executor.execute_trial(request))
    await provider.started.wait()

    task.cancel("stop intervention trial")
    with pytest.raises(asyncio.CancelledError, match="stop intervention trial"):
        await task

    assert task.cancelling() == 1
    assert task.cancelled()
    assert provider.cancelled.is_set()
    assert len(provider.requests) == 1
    interrupted = await executions.load(request.execution_id)
    assert interrupted is not None
    assert interrupted.phase is MemoryInterventionExecutionPhase.SESSION_BOUND
    assert interrupted.status is MemoryInterventionExecutionStatus.ACTIVE
    assert interrupted.runtime_cancellation_observed is True

    recovered = await executor.execute_trial(request)

    assert len(provider.requests) == 1
    assert recovered.execution.phase is MemoryInterventionExecutionPhase.FINALIZED
    assert recovered.execution.status is MemoryInterventionExecutionStatus.CANCELLED
    assert recovered.execution.failure_code == "runtime_cancelled"
    assert recovered.snapshot_result is not None
    assert (
        recovered.snapshot_result.terminal_disposition is AgentSnapshotTerminalDisposition.CANCELLED
    )
    assert recovered.eval_result is not None
    assert (
        recovered.eval_result.memory_attribution.completeness
        is EvalMemoryEvidenceCompleteness.UNAVAILABLE
    )
    assert recovered.eval_result.memory_attribution.proves_empty is False
    assert recovered.binding is not None
    assert recovered.binding.terminal_evidence_available is False
    assert recovered.binding.attribution.status is MemoryAttributionStatus.UNAVAILABLE
    assert recovered.binding.proves_no_memory_exposure is False


@_async_test
async def test_runtime_cancellation_authority_survives_repeated_caller_cancellation(
    tmp_path,
) -> None:
    provider = _BlockingRuntimeProvider()
    executor, executions, request, _ = await _canonical_execution_harness(
        tmp_path,
        provider=provider,
        suffix="repeated-cancellation",
    )
    pausing_store = _PausingCancellationAuthorityStore(executions)
    executor.executions = pausing_store
    task = asyncio.create_task(executor.execute_trial(request))
    await provider.started.wait()

    task.cancel("first intervention cancellation")
    await pausing_store.publication_started.wait()
    task.cancel("second intervention cancellation")
    pausing_store.allow_publication.set()
    with pytest.raises(asyncio.CancelledError, match="first intervention cancellation"):
        await task

    assert task.cancelling() == 2
    assert task.cancelled()
    record = await executions.load(request.execution_id)
    assert record is not None
    assert record.phase is MemoryInterventionExecutionPhase.SESSION_BOUND
    assert record.runtime_cancellation_observed is True


@_async_test
async def test_concrete_runner_does_not_misclassify_operator_interruption_as_cancellation(
    tmp_path,
) -> None:
    provider = _BlockingRuntimeProvider()
    executor, executions, request, _ = await _canonical_execution_harness(
        tmp_path,
        provider=provider,
        suffix="operator-interrupted",
    )
    task = asyncio.create_task(executor.execute_trial(request))
    await provider.started.wait()
    runner = executor.runtime_runner
    assert type(runner) is CayuMemoryInterventionRuntimeRunner
    factory = runner.factory
    assert type(factory) is _CanonicalRuntimeApplicationFactory
    app = factory.created_apps[-1]
    active = await executions.load(request.execution_id)
    assert active is not None

    interrupt_events = [
        event
        async for event in app.interrupt_session(
            InterruptSessionRequest(
                session_id=active.session_id,
                reason="operator requested intervention stop",
            )
        )
    ]
    outcome = await task

    assert interrupt_events
    assert provider.cancelled.is_set()
    assert outcome.execution.status is MemoryInterventionExecutionStatus.OUTCOME_UNKNOWN
    assert outcome.execution.failure_code == "runtime_outcome_unknown"
    assert outcome.execution.runtime_cancellation_observed is False
    assert outcome.snapshot_result is not None
    assert (
        outcome.snapshot_result.terminal_disposition
        is AgentSnapshotTerminalDisposition.OUTCOME_UNKNOWN
    )
    assert await executions.load(request.execution_id) == outcome.execution


@_async_test
async def test_recall_variant_rejects_unrelated_runtime_profile_changes_before_dispatch(
    tmp_path,
) -> None:
    provider = ScriptedModelProvider(
        (
            ModelStreamEvent.text_delta("unused"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        )
    )
    executor, _, _, snapshot = await _canonical_execution_harness(
        tmp_path,
        provider=provider,
        suffix="unrelated-profile-change",
    )
    runner = executor.runtime_runner
    assert type(runner) is CayuMemoryInterventionRuntimeRunner
    factory = runner.factory
    assert type(factory) is _CanonicalRuntimeApplicationFactory
    baseline_policy = _canonical_policy()
    off_policy = _canonical_policy(AutomaticRecallMode.OFF)
    factory.model_by_policy[off_policy.fingerprint()] = "changed-model"
    await _record_runtime_profile(
        factory,
        policy=off_policy,
        suffix="unrelated-profile-change-off",
    )
    request = _request(
        _automatic_recall_off_spec(
            snapshot,
            starting_policy=baseline_policy,
            trial_policy=off_policy,
        ),
        trial_id="unrelated-profile-change",
    )

    with pytest.raises(
        MemoryInterventionExecutionConflict,
        match="outside automatic recall",
    ):
        await executor.execute_trial(request)

    assert provider.requests == []


@_async_test
async def test_actual_session_admission_rejects_profile_race_after_dry_preflight(
    tmp_path,
) -> None:
    provider = ScriptedModelProvider(
        (
            ModelStreamEvent.text_delta("unused"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        )
    )
    executor, executions, request, _ = await _canonical_execution_harness(
        tmp_path,
        provider=provider,
        suffix="actual-profile-race",
        factory_type=_ProfileRacingRuntimeApplicationFactory,
    )

    with pytest.raises(
        ExecutionProfileMismatchError,
        match="execution profile changed",
    ):
        await executor.execute_trial(request)

    assert provider.requests == []
    record = await executions.load(request.execution_id)
    assert record is not None
    assert record.phase is MemoryInterventionExecutionPhase.SESSION_BOUND
    runner = executor.runtime_runner
    assert type(runner) is CayuMemoryInterventionRuntimeRunner
    session = await runner.factory.sessions.load(request.session_id)
    assert session is None


@_async_test
async def test_actual_session_admission_rejects_store_swap_after_dry_preflight(
    tmp_path,
) -> None:
    provider = ScriptedModelProvider(
        (
            ModelStreamEvent.text_delta("must not dispatch"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        )
    )
    executor, executions, request, _ = await _canonical_execution_harness(
        tmp_path,
        provider=provider,
        suffix="actual-store-race",
        factory_type=_StoreRacingRuntimeApplicationFactory,
    )

    with pytest.raises(ValueError, match="environment changed after private admission"):
        await executor.execute_trial(request)

    assert provider.requests == []
    record = await executions.load(request.execution_id)
    assert record is not None
    assert record.phase is MemoryInterventionExecutionPhase.SESSION_BOUND


@pytest.mark.parametrize(
    "spec_factory",
    (
        _spec,
        _automatic_recall_off_spec,
        _omit_spec,
        _replacement_spec,
        _negative_control_spec,
    ),
)
@_async_test
async def test_fixed_variant_kinds_preserve_their_declared_effect(
    spec_factory,
) -> None:
    snapshot = _snapshot()
    executor, _, _, _ = await _executor(
        snapshot,
        snapshot_store=AgentSnapshotCoordinator(_providers(snapshot, {})).store,
        execution_store=InMemoryMemoryInterventionExecutionStore(),
        materializations={},
        views={},
        runtime_results={},
        eval_results={},
    )
    spec = spec_factory(snapshot)

    outcome = await executor.execute_trial(_request(spec))

    assert outcome.execution.status is MemoryInterventionExecutionStatus.COMPLETED
    assert outcome.receipt is not None
    assert outcome.receipt.intervention_kind is spec.kind
    assert outcome.receipt.result_recall_policy_fingerprint == (
        spec.trial_recall_policy_fingerprint
    )
    if spec.kind is MemoryInterventionKind.AS_DECLARED:
        assert outcome.receipt.status is MemoryInterventionEffectStatus.VERIFIED_NO_CHANGE
        assert outcome.receipt.changed_item_revision_fingerprints == ()
    else:
        assert outcome.receipt.status is MemoryInterventionEffectStatus.APPLIED
        assert outcome.receipt.changed_item_revision_fingerprints == tuple(
            change.source_item.revision_fingerprint
            for change in spec.changes
            if change.source_item is not None
        )


@_async_test
async def test_replacement_fixture_bytes_must_match_their_bounded_reference(
    tmp_path,
) -> None:
    provider = ScriptedModelProvider(
        (
            ModelStreamEvent.text_delta("unused"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        )
    )
    executor, executions, _request_value, snapshot = await _canonical_execution_harness(
        tmp_path,
        provider=provider,
        suffix="replacement-fixture-mismatch",
    )
    mismatched_fixture = _fixture(
        "replacement",
        "A different but internally self-consistent replacement fixture.",
    )
    request = _request(
        _replacement_spec(
            snapshot,
            recall_policy=_canonical_policy(),
            fixture=mismatched_fixture,
        ),
        trial_id="replacement-fixture-mismatch",
        prompt="When is Atlas released?",
    )

    with pytest.raises(
        MemoryInterventionExecutionConflict,
        match="immutable declaration",
    ):
        await executor.execute_trial(request)

    assert provider.requests == []
    record = await executions.load(request.execution_id)
    assert record is not None
    assert record.phase is MemoryInterventionExecutionPhase.TRIAL_BOUND


@pytest.mark.parametrize(
    ("disposition", "status"),
    (
        (
            AgentSnapshotTerminalDisposition.FAILED,
            MemoryInterventionExecutionStatus.FAILED,
        ),
        (
            AgentSnapshotTerminalDisposition.CANCELLED,
            MemoryInterventionExecutionStatus.CANCELLED,
        ),
        (
            AgentSnapshotTerminalDisposition.TIMED_OUT,
            MemoryInterventionExecutionStatus.TIMED_OUT,
        ),
        (
            AgentSnapshotTerminalDisposition.OUTCOME_UNKNOWN,
            MemoryInterventionExecutionStatus.OUTCOME_UNKNOWN,
        ),
    ),
)
@_async_test
async def test_runtime_terminal_dispositions_remain_typed_and_replayable(
    disposition: AgentSnapshotTerminalDisposition,
    status: MemoryInterventionExecutionStatus,
) -> None:
    snapshot = _snapshot()
    executor, overlay, runner, evaluator = await _executor(
        snapshot,
        snapshot_store=AgentSnapshotCoordinator(_providers(snapshot, {})).store,
        execution_store=InMemoryMemoryInterventionExecutionStore(),
        materializations={},
        views={},
        runtime_results={},
        eval_results={},
        runtime_terminal_disposition=disposition,
    )
    request = _request(_spec(snapshot))

    first = await executor.execute_trial(request)
    replay = await executor.execute_trial(request)

    assert first == replay
    assert first.execution.phase is MemoryInterventionExecutionPhase.FINALIZED
    assert first.execution.status is status
    assert first.execution.failure_code == f"runtime_{status.value}"
    assert first.snapshot_result is not None
    assert first.snapshot_result.terminal_disposition is disposition
    assert first.binding is not None
    assert overlay.apply_calls == runner.run_calls == evaluator.evaluate_calls == 1


@_async_test
async def test_runtime_child_cancellation_is_not_authenticated_by_historical_task_state() -> None:
    snapshot = _snapshot()
    store = InMemoryMemoryInterventionExecutionStore()
    executor, _, runner, _ = await _executor(
        snapshot,
        snapshot_store=AgentSnapshotCoordinator(_providers(snapshot, {})).store,
        execution_store=store,
        materializations={},
        views={},
        runtime_results={},
        eval_results={},
    )
    runner.raise_child_cancellation = True
    request = _request(_spec(snapshot), trial_id="child-cancellation")
    current_task = asyncio.current_task()
    assert current_task is not None
    current_task.cancel("historical cancellation")
    try:
        await asyncio.sleep(0)
    except asyncio.CancelledError as historical:
        assert str(historical) == "historical cancellation"
    assert current_task.cancelling() == 1

    try:
        with pytest.raises(
            MemoryInterventionExecutionConflict,
            match="without authenticated caller cancellation",
        ):
            await executor.execute_trial(request)
        record = await store.load(request.execution_id)
        assert record is not None
        assert record.phase is MemoryInterventionExecutionPhase.SESSION_BOUND
        assert record.runtime_cancellation_observed is False
        assert current_task.cancelling() == 1
    finally:
        current_task.uncancel()


@_async_test
async def test_oversized_runtime_attribution_is_compacted_before_durable_publication() -> None:
    snapshot = _snapshot()
    store = InMemoryMemoryInterventionExecutionStore()
    executor, _, runner, _ = await _executor(
        snapshot,
        snapshot_store=AgentSnapshotCoordinator(_providers(snapshot, {})).store,
        execution_store=store,
        materializations={},
        views={},
        runtime_results={},
        eval_results={},
    )
    runner.attribution = _oversized_attribution()
    request = _request(_spec(snapshot), trial_id="oversized-runtime-attribution")

    outcome = await executor.execute_trial(request)
    persisted = await store.load(request.execution_id)

    assert outcome.execution.status is MemoryInterventionExecutionStatus.COMPLETED
    assert persisted == outcome.execution
    assert persisted is not None
    assert persisted.runtime_result_payload is not None
    durable_attribution = MemoryAttribution.model_validate(
        persisted.runtime_result_payload["attribution"]
    )
    assert durable_attribution.status is MemoryAttributionStatus.TRUNCATED
    assert durable_attribution.truncated is True
    assert durable_attribution.receipts == ()
    assert durable_attribution.observed_receipt_count == 1_000
    assert durable_attribution.omitted_receipt_count_at_least == 1_000
    assert outcome.binding is not None
    assert outcome.binding.attribution == durable_attribution


@_async_test
async def test_indeterminate_exposure_survives_execution_persistence_and_replay(
    tmp_path,
) -> None:
    snapshot = _snapshot()
    snapshot_path = tmp_path / "indeterminate-snapshots.db"
    execution_path = tmp_path / "indeterminate-executions.db"
    first, _, runner, _ = await _executor(
        snapshot,
        snapshot_store=SQLiteAgentSnapshotStore(snapshot_path),
        execution_store=SQLiteMemoryInterventionExecutionStore(execution_path),
        materializations={},
        views={},
        runtime_results={},
        eval_results={},
    )
    attribution = _indeterminate_exposure_attribution()
    runner.attribution = attribution
    request = _request(_spec(snapshot), trial_id="indeterminate-exposure")

    outcome = await first.execute_trial(request)
    persisted = await SQLiteMemoryInterventionExecutionStore(execution_path).load(
        request.execution_id
    )

    assert outcome.execution.status is MemoryInterventionExecutionStatus.COMPLETED
    assert persisted is not None
    assert persisted.runtime_result_payload is not None
    assert (
        MemoryAttribution.model_validate(persisted.runtime_result_payload["attribution"])
        == attribution
    )
    assert outcome.binding is not None
    assert outcome.binding.attribution == attribution
    assert outcome.binding.proves_no_memory_exposure is False
    assert outcome.binding.attribution.exposures[0].state is ContextExposureState.INDETERMINATE
    assert outcome.binding.attribution.exposures[0].provider_exposure_proven is False

    restarted, _, restarted_runner, _ = await _executor(
        snapshot,
        snapshot_store=SQLiteAgentSnapshotStore(snapshot_path),
        execution_store=SQLiteMemoryInterventionExecutionStore(execution_path),
        materializations={},
        views={},
        runtime_results={},
        eval_results={},
    )
    replay = await restarted.execute_trial(request)

    assert replay == outcome
    assert replay.binding is not None
    assert replay.binding.proves_no_memory_exposure is False
    assert replay.binding.attribution.exposures[0].state is ContextExposureState.INDETERMINATE
    assert restarted_runner.run_calls == 0
    assert restarted_runner.recover_calls == 0


@_async_test
async def test_terminal_memory_census_survives_journal_recovery_and_fails_closed_on_deletion(
    tmp_path,
) -> None:
    snapshot = _snapshot()
    snapshot_path = tmp_path / "terminal-census-snapshots.db"
    execution_path = tmp_path / "terminal-census-executions.db"
    runtime_results: dict[str, MemoryInterventionRuntimeResult] = {}
    eval_results: dict[str, EvalTrialResult] = {}
    first, _, runner, _ = await _executor(
        snapshot,
        snapshot_store=SQLiteAgentSnapshotStore(snapshot_path),
        execution_store=SQLiteMemoryInterventionExecutionStore(execution_path),
        materializations={},
        views={},
        runtime_results=runtime_results,
        eval_results=eval_results,
    )
    runner.expected_receipt_count = 1
    runner.expected_exposure_count = 0
    runner.effective_attribution_bounds = MemoryAttributionBounds(
        max_receipts=1,
        max_exposures=2,
        max_items=3,
        max_source_bytes=1_024,
        max_projection_bytes=2_048,
    )
    request = _request(_spec(snapshot), trial_id="terminal-census-deletion")

    outcome = await first.execute_trial(request)
    persisted = await SQLiteMemoryInterventionExecutionStore(execution_path).load(
        request.execution_id
    )

    assert outcome.eval_result is not None
    evidence = outcome.eval_result.memory_attribution
    assert evidence.effective_bounds == runner.effective_attribution_bounds
    assert evidence.completeness is EvalMemoryEvidenceCompleteness.UNAVAILABLE
    assert evidence.sources[0].limitations == (EvalMemoryEvidenceLimitation.DELETED,)
    assert outcome.binding is not None
    assert outcome.binding.terminal_evidence_available is True
    assert outcome.binding.expected_receipt_count == 1
    assert outcome.binding.proves_no_memory_exposure is False
    assert persisted is not None
    assert persisted.runtime_result_payload is not None
    assert persisted.runtime_result_payload["expected_receipt_count"] == 1
    assert persisted.runtime_result_payload[
        "effective_attribution_bounds"
    ] == runner.effective_attribution_bounds.model_dump(mode="json")

    restarted, _, restarted_runner, _ = await _executor(
        snapshot,
        snapshot_store=SQLiteAgentSnapshotStore(snapshot_path),
        execution_store=SQLiteMemoryInterventionExecutionStore(execution_path),
        materializations={},
        views={},
        runtime_results=runtime_results,
        eval_results=eval_results,
    )
    replay = await restarted.execute_trial(request)

    assert replay == outcome
    assert restarted_runner.run_calls == 0
    assert restarted_runner.recover_calls == 0


@_async_test
async def test_missing_terminal_evidence_cannot_prove_empty_or_remain_comparable() -> None:
    snapshot = _snapshot()
    executor, _, runner, _ = await _executor(
        snapshot,
        snapshot_store=AgentSnapshotCoordinator(_providers(snapshot, {})).store,
        execution_store=InMemoryMemoryInterventionExecutionStore(),
        materializations={},
        views={},
        runtime_results={},
        eval_results={},
    )
    runner.terminal_evidence_available = False
    request = _request(_spec(snapshot), trial_id="terminal-evidence-unavailable")

    outcome = await executor.execute_trial(request)

    assert outcome.eval_result is not None
    assert (
        outcome.eval_result.memory_attribution.completeness
        is EvalMemoryEvidenceCompleteness.UNAVAILABLE
    )
    assert outcome.eval_result.memory_attribution.proves_empty is False
    assert outcome.binding is not None
    assert outcome.binding.terminal_evidence_available is False
    assert outcome.binding.proves_no_memory_exposure is False


@pytest.mark.parametrize("foreign", (False, True))
@_async_test
async def test_concrete_runner_rejects_forged_or_foreign_terminal_evidence(
    tmp_path,
    foreign: bool,
) -> None:
    sessions = _SubstitutingTerminalEvidenceSQLiteSessionStore(
        tmp_path / f"terminal-evidence-{'foreign' if foreign else 'forged'}.db"
    )
    provider = _BlockingRuntimeProvider()
    provider.release.set()
    executor, _, request, _ = await _canonical_execution_harness(
        tmp_path,
        provider=provider,
        suffix=f"terminal-evidence-{'foreign' if foreign else 'forged'}",
        sessions=sessions,
    )
    if foreign:
        first = await executor.execute_trial(request)
        sessions.foreign_evidence = await SQLiteSessionStore.load_terminal_session_evidence(
            sessions,
            first.execution.session_id,
        )
        request = _request(
            request.spec,
            trial_id="foreign-terminal-evidence-target",
            prompt="When is Atlas released?",
        )
    else:
        sessions.forge_empty_events = True

    with pytest.raises(
        MemoryInterventionExecutionConflict,
        match="invalid or foreign terminal evidence",
    ):
        await executor.execute_trial(request)


@_async_test
async def test_concrete_runner_rejects_stale_evidence_for_the_same_session_epoch(
    tmp_path,
) -> None:
    sessions = _AdvancedTerminalSessionSQLiteSessionStore(
        tmp_path / "terminal-evidence-stale-epoch.db"
    )
    provider = _BlockingRuntimeProvider()
    provider.release.set()
    executor, _, request, _ = await _canonical_execution_harness(
        tmp_path,
        provider=provider,
        suffix="terminal-evidence-stale-epoch",
        sessions=sessions,
    )

    with pytest.raises(
        MemoryInterventionExecutionConflict,
        match="invalid or foreign terminal evidence",
    ):
        await executor.execute_trial(request)


@_async_test
async def test_concrete_runner_binds_evidence_to_the_authenticated_terminal_snapshot(
    tmp_path,
) -> None:
    sessions = _CoherentlyAdvancedTerminalEvidenceSQLiteSessionStore(
        tmp_path / "terminal-evidence-coherent-later-epoch.db"
    )
    provider = _BlockingRuntimeProvider()
    provider.release.set()
    executor, _, request, _ = await _canonical_execution_harness(
        tmp_path,
        provider=provider,
        suffix="terminal-evidence-coherent-later-epoch",
        sessions=sessions,
    )

    with pytest.raises(
        MemoryInterventionExecutionConflict,
        match="invalid or foreign terminal evidence",
    ):
        await executor.execute_trial(request)

    assert sessions.terminal_session_loads >= 1


@pytest.mark.parametrize(
    ("failure", "expected_limitation"),
    (
        ("unsupported", EvalMemoryEvidenceLimitation.STORE_UNSUPPORTED),
        ("typed", EvalMemoryEvidenceLimitation.EVIDENCE_READ_FAILED),
        ("read", EvalMemoryEvidenceLimitation.EVIDENCE_READ_FAILED),
    ),
)
@_async_test
async def test_concrete_runner_publishes_unavailable_terminal_evidence_and_replays_it(
    tmp_path,
    failure: str,
    expected_limitation: EvalMemoryEvidenceLimitation,
) -> None:
    sessions = _UnavailableTerminalEvidenceSQLiteSessionStore(
        tmp_path / f"terminal-evidence-unavailable-{failure}.db",
        failure=failure,
    )
    provider = _BlockingRuntimeProvider()
    provider.release.set()
    executor, executions, request, _ = await _canonical_execution_harness(
        tmp_path,
        provider=provider,
        suffix=f"terminal-evidence-unavailable-{failure}",
        sessions=sessions,
    )

    outcome = await executor.execute_trial(request)
    persisted = await executions.load(request.execution_id)

    assert outcome.eval_result is not None
    evidence = outcome.eval_result.memory_attribution
    assert evidence.completeness is EvalMemoryEvidenceCompleteness.UNAVAILABLE
    assert evidence.proves_empty is False
    assert evidence.sources[0].limitations == (expected_limitation,)
    assert outcome.binding is not None
    assert outcome.binding.terminal_evidence_available is False
    assert outcome.binding.proves_no_memory_exposure is False
    assert persisted is not None
    assert persisted.runtime_result_payload is not None
    assert persisted.runtime_result_payload["terminal_evidence_limitation"] == str(
        expected_limitation
    )
    assert await executor.execute_trial(request) == outcome


@_async_test
async def test_concrete_runner_applies_persists_and_recovers_effective_eval_bounds(
    tmp_path,
) -> None:
    policy_bounds = standard_eval_memory_attribution_bounds()
    above_policy = MemoryAttributionBounds(
        max_receipts=101,
        max_exposures=101,
        max_items=1_001,
        max_source_bytes=8 * 1024 * 1024,
        max_projection_bytes=1024 * 1024,
    )
    above_store = _RecordingAttributionBoundsSQLiteSessionStore(
        tmp_path / "runner-bounds-above-sessions.db"
    )
    above_provider = _BlockingRuntimeProvider()
    above_provider.release.set()
    above_executor, above_executions, above_request, _ = await _canonical_execution_harness(
        tmp_path,
        provider=above_provider,
        suffix="runner-bounds-above",
        sessions=above_store,
        attribution_bounds=above_policy,
    )

    above_outcome = await above_executor.execute_trial(above_request)
    above_persisted = await above_executions.load(above_request.execution_id)
    read_count = len(above_store.recall_queries) + len(above_store.exposure_queries)

    assert above_store.recall_queries[0].max_bytes == policy_bounds.max_source_bytes
    assert above_outcome.eval_result is not None
    assert above_outcome.eval_result.memory_attribution.effective_bounds == policy_bounds
    above_source = above_outcome.eval_result.memory_attribution.sources[0]
    assert above_source.source.session_alias is not None
    assert above_source.source.session_alias.key_id == "memory-intervention-test"
    assert above_outcome.execution.session_id not in json.dumps(
        above_outcome.eval_result.memory_attribution.model_dump(mode="json")
    )
    assert above_persisted is not None
    assert above_persisted.runtime_result_payload is not None
    assert above_persisted.runtime_result_payload[
        "effective_attribution_bounds"
    ] == policy_bounds.model_dump(mode="json")
    assert above_persisted.runtime_result_payload[
        "source_alias"
    ] == above_source.source.session_alias.model_dump(mode="json")
    assert await above_executor.execute_trial(above_request) == above_outcome
    assert len(above_store.recall_queries) + len(above_store.exposure_queries) == read_count

    lower_bounds = MemoryAttributionBounds(
        max_receipts=1,
        max_exposures=1,
        max_items=3,
        max_source_bytes=2 * 1024 * 1024,
        max_projection_bytes=256 * 1024,
    )
    lower_store = _RecordingAttributionBoundsSQLiteSessionStore(
        tmp_path / "runner-bounds-lower-sessions.db"
    )
    lower_provider = _BlockingRuntimeProvider()
    lower_provider.release.set()
    lower_executor, lower_executions, lower_request, _ = await _canonical_execution_harness(
        tmp_path,
        provider=lower_provider,
        suffix="runner-bounds-lower",
        sessions=lower_store,
        attribution_bounds=lower_bounds,
    )

    lower_outcome = await lower_executor.execute_trial(lower_request)
    lower_persisted = await lower_executions.load(lower_request.execution_id)

    assert lower_store.recall_queries[0].max_bytes == lower_bounds.max_source_bytes
    assert lower_outcome.eval_result is not None
    assert lower_outcome.eval_result.memory_attribution.effective_bounds == lower_bounds
    assert lower_persisted is not None
    assert lower_persisted.runtime_result_payload is not None
    assert lower_persisted.runtime_result_payload[
        "effective_attribution_bounds"
    ] == lower_bounds.model_dump(mode="json")


@_async_test
async def test_executor_rejects_above_policy_runtime_bounds_before_persistence() -> None:
    snapshot = _snapshot()
    execution_store = InMemoryMemoryInterventionExecutionStore()
    executor, _, runner, evaluator = await _executor(
        snapshot,
        snapshot_store=AgentSnapshotCoordinator(_providers(snapshot, {})).store,
        execution_store=execution_store,
        materializations={},
        views={},
        runtime_results={},
        eval_results={},
    )
    request = _request(_spec(snapshot), trial_id="invalid-runtime-bounds-policy")
    valid = MemoryInterventionRuntimeResult(
        session_id=request.session_id,
        terminal_disposition=AgentSnapshotTerminalDisposition.COMPLETED,
        runtime_evidence_fingerprint=_digest("runtime-bounds:policy"),
        terminal_evidence_available=True,
        expected_receipt_count=0,
        expected_exposure_count=0,
        effective_attribution_bounds=standard_eval_memory_attribution_bounds(),
        attribution=_empty_attribution(),
    )
    policy = standard_eval_memory_attribution_bounds()
    invalid = valid.model_copy(
        update={
            "effective_attribution_bounds": policy.model_copy(
                update={"max_receipts": policy.max_receipts + 1}
            )
        }
    )
    runner.result_override = invalid

    with pytest.raises(
        ValueError,
        match="Effective receipt bound exceeds the eval capture policy",
    ):
        await executor.execute_trial(request)

    persisted = await execution_store.load(request.execution_id)
    assert persisted is not None
    assert persisted.phase is MemoryInterventionExecutionPhase.SESSION_BOUND
    assert persisted.runtime_result_payload is None
    assert evaluator.evaluate_calls == 0


@_async_test
async def test_executor_rejects_same_identity_with_changed_complete_request() -> None:
    snapshot = _snapshot()
    executor, overlay, runner, _ = await _executor(
        snapshot,
        snapshot_store=AgentSnapshotCoordinator(_providers(snapshot, {})).store,
        execution_store=InMemoryMemoryInterventionExecutionStore(),
        materializations={},
        views={},
        runtime_results={},
        eval_results={},
    )
    spec = _spec(snapshot)
    await executor.execute_trial(_request(spec, prompt="first prompt"))

    with pytest.raises(MemoryInterventionExecutionConflict):
        await executor.execute_trial(_request(spec, prompt="changed prompt"))

    assert overlay.apply_calls == 1
    assert runner.run_calls == 1


@_async_test
async def test_executor_rejects_prepared_runtime_authority_before_identity_or_dispatch() -> None:
    snapshot = _snapshot()
    store = InMemoryMemoryInterventionExecutionStore()
    executor, overlay, runner, evaluator = await _executor(
        snapshot,
        snapshot_store=AgentSnapshotCoordinator(_providers(snapshot, {})).store,
        execution_store=store,
        materializations={},
        views={},
        runtime_results={},
        eval_results={},
    )
    clean = _request(_spec(snapshot), trial_id="prepared-runtime-authority")
    prepared_run_request = run_request_with_runtime_invocation(
        clean.run_request,
        source=SessionExecutionSource.HTTP_RUN,
        verified_origin=InvocationOrigin(
            trust=InvocationOriginTrust.SERVER_VERIFIED,
            subject="server-owned-subject",
            tenant="server-owned-tenant",
        ),
    )

    with pytest.raises(ValueError, match="prepared runtime authority"):
        MemoryInterventionTrialRequest(
            spec=clean.spec,
            candidate_id=clean.candidate_id,
            trial_id=clean.trial_id,
            case=clean.case,
            run_request=prepared_run_request,
            timeout_seconds=clean.timeout_seconds,
        )

    clean.run_request._runtime_invocation_source = SessionExecutionSource.SDK_RUN
    with pytest.raises(ValueError, match="prepared runtime authority"):
        await executor.execute_trial(clean)

    assert await store.load(clean.execution_id) is None
    assert overlay.apply_calls == 0
    assert runner.run_calls == 0
    assert evaluator.evaluate_calls == 0


@_async_test
async def test_concurrent_executors_share_one_exact_effect_and_runtime_dispatch() -> None:
    snapshot = _snapshot()
    snapshot_store = AgentSnapshotCoordinator(_providers(snapshot, {})).store
    execution_store = InMemoryMemoryInterventionExecutionStore()
    materializations: dict[str, AgentSnapshotMaterializedComponent] = {}
    views: dict[str, MemoryInterventionRuntimeView] = {}
    runtime_results: dict[str, MemoryInterventionRuntimeResult] = {}
    eval_results: dict[str, EvalTrialResult] = {}
    first, first_overlay, first_runner, first_evaluator = await _executor(
        snapshot,
        snapshot_store=snapshot_store,
        execution_store=execution_store,
        materializations=materializations,
        views=views,
        runtime_results=runtime_results,
        eval_results=eval_results,
    )
    second, second_overlay, second_runner, second_evaluator = await _executor(
        snapshot,
        snapshot_store=snapshot_store,
        execution_store=execution_store,
        materializations=materializations,
        views=views,
        runtime_results=runtime_results,
        eval_results=eval_results,
    )
    request = _request(_spec(snapshot))

    left, right = await asyncio.gather(
        first.execute_trial(request),
        second.execute_trial(request),
    )

    assert left == right
    assert first_overlay.apply_calls + second_overlay.apply_calls == 1
    assert first_runner.run_calls + second_runner.run_calls == 1
    assert first_evaluator.evaluate_calls + second_evaluator.evaluate_calls == 1
    assert len(views) == len(runtime_results) == len(eval_results) == 1


@_async_test
async def test_evaluator_identity_is_scoped_to_the_full_case_revision(tmp_path) -> None:
    provider = ScriptedModelProvider(
        (
            (
                ModelStreamEvent.text_delta("first"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ),
            (
                ModelStreamEvent.text_delta("second"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ),
        )
    )
    executor, _, first_request, _ = await _canonical_execution_harness(
        tmp_path,
        provider=provider,
        suffix="case-revision-evaluator",
    )
    second_request = first_request.model_copy(
        update={
            "case": first_request.case.model_copy(
                update={"case_revision": f"sha256:{_digest('case-1-revision-2')}"}
            )
        }
    )

    first = await executor.execute_trial(first_request)
    second = await executor.execute_trial(second_request)

    assert first_request.execution_id != second_request.execution_id
    assert first.execution.session_id != second.execution.session_id
    assert first.execution.status is MemoryInterventionExecutionStatus.COMPLETED
    assert second.execution.status is MemoryInterventionExecutionStatus.COMPLETED
    evaluator = executor.evaluator
    assert type(evaluator) is _Evaluator
    assert set(evaluator.results) == {
        first_request.execution_id,
        second_request.execution_id,
    }
    assert len(provider.requests) == 2


@_async_test
async def test_concurrent_concrete_executors_share_one_runtime_dispatch(tmp_path) -> None:
    provider = _BlockingRuntimeProvider()
    first, executions, request, _ = await _canonical_execution_harness(
        tmp_path,
        provider=provider,
        suffix="concrete-concurrency",
    )
    first_runner = first.runtime_runner
    assert type(first_runner) is CayuMemoryInterventionRuntimeRunner
    second = MemoryInterventionExecutor(
        snapshots=first.snapshots,
        executions=executions,
        overlay_provider=first.overlay_provider,
        runtime_runner=CayuMemoryInterventionRuntimeRunner(first_runner.factory),
        evaluator=first.evaluator,
        request_keys={
            "test-key": MemoryInterventionRequestFingerprintKey(
                key_id="test-key",
                secret=b"k" * 32,
            )
        },
        current_request_key_id="test-key",
        clock=lambda: datetime(2026, 8, 25, tzinfo=UTC),
    )

    first_task = asyncio.create_task(first.execute_trial(request))
    await provider.started.wait()
    second_task = asyncio.create_task(second.execute_trial(request))
    await asyncio.sleep(0)

    assert not second_task.done()
    provider.release.set()
    left, right = await asyncio.gather(first_task, second_task)

    assert left == right
    assert len(provider.requests) == 1
    assert left.execution.status is MemoryInterventionExecutionStatus.COMPLETED


@_async_test
async def test_durable_runtime_lease_prevents_cas_loser_from_fencing_live_owner(
    tmp_path,
) -> None:
    provider = _BlockingRuntimeProvider()
    first, executions, request, _ = await _canonical_execution_harness(
        tmp_path,
        provider=provider,
        suffix="durable-runtime-lease",
    )
    first_runner = first.runtime_runner
    assert type(first_runner) is CayuMemoryInterventionRuntimeRunner
    second = MemoryInterventionExecutor(
        snapshots=first.snapshots,
        executions=executions,
        overlay_provider=first.overlay_provider,
        runtime_runner=CayuMemoryInterventionRuntimeRunner(first_runner.factory),
        evaluator=first.evaluator,
        request_keys={
            "test-key": MemoryInterventionRequestFingerprintKey(
                key_id="test-key",
                secret=b"k" * 32,
            )
        },
        current_request_key_id="test-key",
        clock=lambda: datetime(2026, 8, 25, tzinfo=UTC),
    )

    # Bypass only the process-local gate to model deliveries in separate workers.
    first_task = asyncio.create_task(first._execute_trial(request))
    await provider.started.wait()
    second_task = asyncio.create_task(second._execute_trial(request))
    await asyncio.sleep(0.4)

    assert not second_task.done()
    assert len(provider.requests) == 1
    provider.release.set()
    left, right = await asyncio.gather(first_task, second_task)

    assert left == right
    assert len(provider.requests) == 1
    assert left.execution.status is MemoryInterventionExecutionStatus.COMPLETED


@_async_test
async def test_indeterminate_effect_is_terminal_and_never_dispatches_runtime() -> None:
    snapshot = _snapshot()
    executor, overlay, runner, evaluator = await _executor(
        snapshot,
        snapshot_store=AgentSnapshotCoordinator(_providers(snapshot, {})).store,
        execution_store=InMemoryMemoryInterventionExecutionStore(),
        materializations={},
        views={},
        runtime_results={},
        eval_results={},
        terminal_status=MemoryInterventionEffectStatus.INDETERMINATE,
    )

    outcome = await executor.execute_trial(_request(_omit_spec(snapshot)))

    assert outcome.execution.phase is MemoryInterventionExecutionPhase.EFFECT_RESOLVED
    assert outcome.execution.status is MemoryInterventionExecutionStatus.INDETERMINATE
    assert outcome.receipt is not None
    assert outcome.receipt.status is MemoryInterventionEffectStatus.INDETERMINATE
    assert overlay.apply_calls == 1
    assert runner.run_calls == 0
    assert evaluator.evaluate_calls == 0


@_async_test
async def test_omission_rejects_a_newer_source_revision_before_runtime_dispatch() -> None:
    snapshot = _snapshot()
    snapshot_store = AgentSnapshotCoordinator(_providers(snapshot, {})).store
    await snapshot_store.save_snapshot(snapshot)
    overlay = _RevisionCheckingOverlayProvider(
        {},
        current_revision_fingerprint=_digest("entry:2"),
    )
    runner = _RuntimeRunner({})
    evaluator = _Evaluator({})
    executor = MemoryInterventionExecutor(
        snapshots=AgentSnapshotCoordinator(
            _providers(snapshot, {}),
            store=snapshot_store,
            clock=lambda: datetime(2026, 8, 25, tzinfo=UTC),
        ),
        executions=InMemoryMemoryInterventionExecutionStore(),
        overlay_provider=overlay,
        runtime_runner=runner,
        evaluator=evaluator,
        request_keys={
            "test-key": MemoryInterventionRequestFingerprintKey(
                key_id="test-key",
                secret=b"k" * 32,
            )
        },
        current_request_key_id="test-key",
        clock=lambda: datetime(2026, 8, 25, tzinfo=UTC),
    )

    outcome = await executor.execute_trial(_request(_omit_spec(snapshot)))

    assert outcome.execution.status is MemoryInterventionExecutionStatus.CONFLICTING
    assert outcome.receipt is not None
    assert outcome.receipt.status is MemoryInterventionEffectStatus.CONFLICTING
    assert runner.run_calls == 0
    assert evaluator.evaluate_calls == 0


@_async_test
async def test_sqlite_restart_recovers_each_exact_phase_without_duplicate_dispatch(
    tmp_path,
) -> None:
    snapshot = _snapshot()
    snapshot_path = tmp_path / "snapshots.db"
    execution_path = tmp_path / "executions.db"
    materializations: dict[str, AgentSnapshotMaterializedComponent] = {}
    views: dict[str, MemoryInterventionRuntimeView] = {}
    runtime_results: dict[str, MemoryInterventionRuntimeResult] = {}
    eval_results: dict[str, EvalTrialResult] = {}
    first, first_overlay, first_runner, first_evaluator = await _executor(
        snapshot,
        snapshot_store=SQLiteAgentSnapshotStore(snapshot_path),
        execution_store=SQLiteMemoryInterventionExecutionStore(execution_path),
        materializations=materializations,
        views=views,
        runtime_results=runtime_results,
        eval_results=eval_results,
    )
    request = _request(_spec(snapshot))
    expected = await first.execute_trial(request)

    restarted, overlay, runner, evaluator = await _executor(
        snapshot,
        snapshot_store=SQLiteAgentSnapshotStore(snapshot_path),
        execution_store=SQLiteMemoryInterventionExecutionStore(execution_path),
        materializations=materializations,
        views=views,
        runtime_results=runtime_results,
        eval_results=eval_results,
    )
    recovered = await restarted.execute_trial(request)

    assert recovered == expected
    assert first_overlay.apply_calls == 1
    assert first_runner.run_calls == 1
    assert first_evaluator.evaluate_calls == 1
    assert overlay.apply_calls == 0
    assert runner.run_calls == 0
    assert evaluator.evaluate_calls == 0
    assert overlay.recover_calls == 1
    assert runner.recover_calls == 0
    assert evaluator.recover_calls == 1


@pytest.mark.parametrize(
    "fail_phase",
    tuple(MemoryInterventionExecutionPhase)[1:],
)
@_async_test
async def test_sqlite_commit_then_failure_recovers_every_phase_without_duplicate_effects(
    tmp_path,
    fail_phase: MemoryInterventionExecutionPhase,
) -> None:
    snapshot = _snapshot()
    snapshot_path = tmp_path / "snapshots.db"
    execution_path = tmp_path / "executions.db"
    materializations: dict[str, AgentSnapshotMaterializedComponent] = {}
    views: dict[str, MemoryInterventionRuntimeView] = {}
    runtime_results: dict[str, MemoryInterventionRuntimeResult] = {}
    eval_results: dict[str, EvalTrialResult] = {}
    durable_execution_store = SQLiteMemoryInterventionExecutionStore(execution_path)
    fault_store = _CommitThenFailExecutionStore(
        durable_execution_store,
        fail_phase=fail_phase,
    )
    first, _, _, _ = await _executor(
        snapshot,
        snapshot_store=SQLiteAgentSnapshotStore(snapshot_path),
        execution_store=fault_store,
        materializations=materializations,
        views=views,
        runtime_results=runtime_results,
        eval_results=eval_results,
    )
    request = _request(_spec(snapshot))

    with pytest.raises(ConnectionError, match="acknowledgement lost"):
        await first.execute_trial(request)

    restarted, overlay, runner, evaluator = await _executor(
        snapshot,
        snapshot_store=SQLiteAgentSnapshotStore(snapshot_path),
        execution_store=SQLiteMemoryInterventionExecutionStore(execution_path),
        materializations=materializations,
        views=views,
        runtime_results=runtime_results,
        eval_results=eval_results,
    )
    recovered = await restarted.execute_trial(request)

    assert recovered.execution.status is MemoryInterventionExecutionStatus.COMPLETED
    assert len(views) == 1
    assert len(runtime_results) == 1
    assert len(eval_results) == 1
    assert overlay.recover_calls == 1
    if fail_phase is MemoryInterventionExecutionPhase.SESSION_BOUND:
        assert runner.recover_calls == 1
    if fail_phase in {
        MemoryInterventionExecutionPhase.RUNTIME_TERMINAL,
        MemoryInterventionExecutionPhase.EVALUATED,
        MemoryInterventionExecutionPhase.FINALIZED,
    }:
        assert evaluator.recover_calls == 1


@_async_test
async def test_reset_and_accumulate_scope_exactly_follow_snapshot_semantics() -> None:
    snapshot = _snapshot()
    materializations: dict[str, AgentSnapshotMaterializedComponent] = {}
    executor, _, _, _ = await _executor(
        snapshot,
        snapshot_store=AgentSnapshotCoordinator(_providers(snapshot, {})).store,
        execution_store=InMemoryMemoryInterventionExecutionStore(),
        materializations=materializations,
        views={},
        runtime_results={},
        eval_results={},
    )
    reset = _spec(snapshot)
    reset_one = await executor.execute_trial(_request(reset, trial_id="reset-1"))
    reset_two = await executor.execute_trial(_request(reset, trial_id="reset-2"))
    assert reset_one.receipt is not None and reset_two.receipt is not None
    assert reset_one.receipt.state_scope_id != reset_two.receipt.state_scope_id

    accumulate = _spec(
        snapshot,
        state_mode=AgentSnapshotTrialStateMode.ACCUMULATE_WITHIN_CANDIDATE,
    )
    other_variant = _spec(
        snapshot,
        state_mode=AgentSnapshotTrialStateMode.ACCUMULATE_WITHIN_CANDIDATE,
        spec_id="spec-another-accumulating-variant",
    )
    accumulated_one = await executor.execute_trial(_request(accumulate, trial_id="accumulate-1"))
    accumulated_two = await executor.execute_trial(_request(accumulate, trial_id="accumulate-2"))
    other_candidate = await executor.execute_trial(
        _request(accumulate, candidate_id="candidate-b", trial_id="accumulate-1")
    )
    same_candidate_other_variant = await executor.execute_trial(
        _request(other_variant, trial_id="accumulate-1")
    )
    assert accumulated_one.receipt is not None
    assert accumulated_two.receipt is not None
    assert other_candidate.receipt is not None
    assert same_candidate_other_variant.receipt is not None
    assert accumulated_one.receipt.state_scope_id == accumulated_two.receipt.state_scope_id
    assert accumulated_one.receipt.state_scope_id != other_candidate.receipt.state_scope_id
    assert (
        accumulated_one.receipt.state_scope_id
        != same_candidate_other_variant.receipt.state_scope_id
    )


@_async_test
async def test_restart_rejects_changed_application_owned_provider_identity(tmp_path) -> None:
    snapshot = _snapshot()
    execution_store = SQLiteMemoryInterventionExecutionStore(tmp_path / "executions.db")
    executor, _, _, _ = await _executor(
        snapshot,
        snapshot_store=AgentSnapshotCoordinator(_providers(snapshot, {})).store,
        execution_store=execution_store,
        materializations={},
        views={},
        runtime_results={},
        eval_results={},
        terminal_status=MemoryInterventionEffectStatus.INDETERMINATE,
    )
    request = _request(_omit_spec(snapshot))
    await executor.execute_trial(request)

    changed, _, _, _ = await _executor(
        snapshot,
        snapshot_store=executor.snapshots.store,
        execution_store=SQLiteMemoryInterventionExecutionStore(tmp_path / "executions.db"),
        materializations={},
        views={},
        runtime_results={},
        eval_results={},
        terminal_status=MemoryInterventionEffectStatus.INDETERMINATE,
        overlay_profile_fingerprint=_digest("overlay-provider:v2"),
    )

    with pytest.raises(MemoryInterventionExecutionConflict):
        await changed.execute_trial(request)


@pytest.mark.parametrize("owner", ("overlay_id", "overlay", "runner", "evaluator"))
@_async_test
async def test_live_application_owner_drift_is_rejected_before_any_effect(owner: str) -> None:
    snapshot = _snapshot()
    execution_store = InMemoryMemoryInterventionExecutionStore()
    executor, overlay, runner, evaluator = await _executor(
        snapshot,
        snapshot_store=AgentSnapshotCoordinator(_providers(snapshot, {})).store,
        execution_store=execution_store,
        materializations={},
        views={},
        runtime_results={},
        eval_results={},
    )
    if owner == "overlay_id":
        overlay.provider_id = "test.memory-intervention-overlay.v2"
    elif owner == "overlay":
        overlay.execution_profile_fingerprint = _digest("overlay-provider:v2")
    elif owner == "runner":
        runner.execution_profile_fingerprint = _digest("runtime-runner:v2")
    else:
        evaluator.evaluator_fingerprint = _digest("evaluator:v2")
    request = _request(_spec(snapshot))

    with pytest.raises(MemoryInterventionExecutionConflict, match="identity changed"):
        await executor.execute_trial(request)

    assert await execution_store.load(request.execution_id) is None
    assert overlay.apply_calls == runner.run_calls == evaluator.evaluate_calls == 0

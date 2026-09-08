from __future__ import annotations

import asyncio
import hashlib
import os
import stat
from collections.abc import Iterator
from contextlib import nullcontext
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Literal

import pytest
from tests.core.test_cost_quality_comparison import _attempt, _side

import cayu.cli._guarded_tree_publication as guarded_publication
import cayu.evals.causal_memory_campaign as reference_campaign
import cayu.evals.external_private_memory_ablation as private_ablation
from cayu.agent_snapshots import (
    AgentSnapshotCoordinator,
    AgentSnapshotResultBinding,
    AgentSnapshotTerminalDisposition,
    SQLiteAgentSnapshotStore,
    execution_profile_snapshot_ref,
)
from cayu.core import AgentSpec, Message
from cayu.evals.corpus import (
    MemoryAttributionAssertionSpec,
    PrivateJudgeReferenceV1,
    StructuredModelJudgeAssertionSpec,
    StructuredRubricCriterionV1,
    StructuredRubricV1,
    ToolArgumentsContainAssertionSpec,
    assertion_spec_revision,
    eval_corpus_to_json,
    pricing_profile_identity,
)
from cayu.evals.execution import compile_corpus_suite
from cayu.evals.execution_profiles import (
    EvalExecutionProfilePolicyV1,
    prepare_eval_execution_profile,
)
from cayu.evals.external_private_memory_ablation import (
    EXTERNAL_PRIVATE_MEMORY_ABLATION_COMPLETION_FILENAME,
    EXTERNAL_PRIVATE_MEMORY_ABLATION_METHODOLOGY_FILENAME,
    EXTERNAL_PRIVATE_MEMORY_ABLATION_MIN_REPORT_EVIDENCE_BYTES_PER_TRIAL,
    EXTERNAL_PRIVATE_MEMORY_ABLATION_REPORT_FILENAME,
    EXTERNAL_PRIVATE_MEMORY_ABLATION_TRIAL_BUDGET_KEY,
    ExternalPrivateMemoryAblationAuthorization,
    ExternalPrivateMemoryAblationCacheEvidencePolicy,
    ExternalPrivateMemoryAblationCacheEvidenceState,
    ExternalPrivateMemoryAblationDestination,
    ExternalPrivateMemoryAblationEvidenceCollector,
    ExternalPrivateMemoryAblationExecutionMode,
    ExternalPrivateMemoryAblationLimitation,
    ExternalPrivateMemoryAblationResult,
    ExternalPrivateMemoryAblationRunFailureCode,
    ExternalPrivateMemoryAblationRunStatus,
    ExternalPrivateMemoryAblationSchedulePolicy,
    ExternalPrivateMemoryAblationScheduleStrategy,
    ExternalPrivateMemoryAblationSupplementalEvidence,
    ExternalPrivateMemoryAblationTrial,
    PreparedExternalPrivateMemoryAblation,
    external_private_memory_ablation_destination,
    external_private_memory_ablation_experiment_revision,
    load_external_private_memory_ablation_corpus,
    prepare_external_private_memory_ablation,
    run_external_private_memory_ablation,
    write_external_private_memory_ablation_artifacts,
)
from cayu.evals.memory_attribution import EvalMemoryAttributionEvidenceV1
from cayu.evals.memory_reporting import (
    MemoryExperimentCase,
    MemoryExperimentGatePolicy,
    MemoryExperimentReportRequest,
    MemoryExperimentVariant,
    MemoryMetricBinding,
    MemoryMetricDirection,
    MemoryMetricGate,
    MemoryMetricRole,
    MemoryPreparationOverheadEvidence,
    MemoryRankingTerm,
    MemoryTrialAvailability,
    memory_experiment_accounting_source_id,
    memory_experiment_accounting_task_id,
    memory_experiment_report_to_json,
)
from cayu.evals.testing import ScriptedModelProvider
from cayu.memory_intervention_execution import (
    CayuMemoryInterventionRuntimeRunner,
    MemoryInterventionExecutionStatus,
    MemoryInterventionExecutor,
    MemoryInterventionExecutorStatePaths,
    MemoryInterventionProviderExecutionMode,
    MemoryInterventionRequestFingerprintKey,
    MemoryInterventionTrialOutcome,
    MemoryInterventionTrialRequest,
    SQLiteMemoryInterventionExecutionStore,
)
from cayu.memory_interventions import MemoryInterventionTrialBinding
from cayu.providers import ModelRequest, ModelStreamEvent
from cayu.runtime.app import CayuApp
from cayu.runtime.budgets import BudgetLimit, BudgetReservation, BudgetWindow
from cayu.runtime.costs import (
    ModelPrice,
    PriceBook,
    PriceSchedule,
    PriceTier,
    Provenance,
    TieredPricing,
    default_price_book,
)
from cayu.runtime.stop_policy import RunLimits
from cayu.storage.budget_ledger import SQLiteBudgetLedger
from cayu.storage.sqlite import SQLiteSessionStore
from cayu.tools import SubagentSpec, SubagentTool
from cayu.vaults.redaction import SecretRedactor

_NOW = datetime(2026, 9, 3, 8, 0, tzinfo=UTC)


class _OrderedCampaignProvider(reference_campaign._CampaignScriptedProvider):
    def __init__(self) -> None:
        super().__init__(recover_only=False)
        self._outputs: tuple[str, ...] = ()

    def set_outputs(self, outputs: tuple[str, ...]) -> None:
        self._outputs = outputs

    async def stream(self, request: ModelRequest):
        self.requests.append(ModelRequest.model_validate(request.model_dump(mode="python")))
        index = len(self.requests) - 1
        if index >= len(self._outputs):
            raise AssertionError("The private campaign exceeded its frozen provider schedule.")
        yield ModelStreamEvent.text_delta(self._outputs[index])
        yield ModelStreamEvent.completed(
            {
                "finish_reason": "stop",
                "usage": {
                    "input_tokens": 64,
                    "output_tokens": 32,
                    "total_tokens": 96,
                },
            }
        )


@dataclass
class _CampaignFixture:
    prepared: PreparedExternalPrivateMemoryAblation
    executor: MemoryInterventionExecutor
    provider: ScriptedModelProvider
    sessions: SQLiteSessionStore
    budgets: SQLiteBudgetLedger

    async def close(self) -> None:
        try:
            await self.budgets.close()
        finally:
            await self.sessions.close()


def _executor_with_execution_store(
    executor: MemoryInterventionExecutor,
    executions: SQLiteMemoryInterventionExecutionStore,
) -> MemoryInterventionExecutor:
    replacement = MemoryInterventionExecutor(
        snapshots=executor.snapshots,
        executions=executions,
        overlay_provider=executor.overlay_provider,
        runtime_runner=executor.runtime_runner,
        evaluator=executor.evaluator,
        request_keys=executor._request_keys,
        current_request_key_id=executor._current_request_key_id,
        clock=executor._clock,
    )
    replacement._runtime_clock = executor._runtime_clock
    replacement._runtime_dispatch_owner_id = executor._runtime_dispatch_owner_id
    return replacement


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class _SupplementalEvidenceCollector(ExternalPrivateMemoryAblationEvidenceCollector):
    collector_fingerprint = _digest("external-private-supplemental-evidence-v1")

    async def collect(
        self,
        *,
        trial: ExternalPrivateMemoryAblationTrial,
        outcome: MemoryInterventionTrialOutcome,
    ) -> ExternalPrivateMemoryAblationSupplementalEvidence:
        del outcome
        return ExternalPrivateMemoryAblationSupplementalEvidence(
            memory_overhead=MemoryPreparationOverheadEvidence.create(
                preparation_duration_ms=trial.repetition,
                context_tokens=16,
                context_bytes=64,
            ),
            cache_evidence_state=ExternalPrivateMemoryAblationCacheEvidenceState.AVAILABLE,
            cache_read_tokens=8,
            cache_write_tokens=4,
            provider_retry_count=1,
        )


class _AccountingEvidenceCollector(ExternalPrivateMemoryAblationEvidenceCollector):
    def __init__(
        self,
        *,
        experiment: MemoryExperimentReportRequest,
        wrong_strategy: bool = False,
        shared_attempt_id: str | None = None,
        foreign_session_id: str | None = None,
    ) -> None:
        self.collector_fingerprint = _digest(
            "accounting:"
            f"{experiment.experiment_id}:{wrong_strategy}:{shared_attempt_id}:"
            f"{foreign_session_id}"
        )
        self._experiment_id = experiment.experiment_id
        self._provider_by_variant = {
            variant.variant_id: (
                variant.execution_profile.candidate.provider_name,
                variant.execution_profile.candidate.model,
            )
            for variant in experiment.variants
        }
        self._wrong_strategy = wrong_strategy
        self._shared_attempt_id = shared_attempt_id
        self._foreign_session_id = foreign_session_id

    async def collect(
        self,
        *,
        trial: ExternalPrivateMemoryAblationTrial,
        outcome: MemoryInterventionTrialOutcome,
    ) -> ExternalPrivateMemoryAblationSupplementalEvidence:
        attempt_id = self._shared_attempt_id or outcome.execution.execution_id
        strategy_id = "wrong-strategy" if self._wrong_strategy else trial.variant_id
        provider_name, model = self._provider_by_variant[trial.variant_id]
        attempt = _attempt(
            attempt_id=attempt_id,
            session_id=self._foreign_session_id or outcome.execution.session_id,
            input_tokens=64,
        ).model_copy(update={"provider_name": provider_name, "model": model})
        return ExternalPrivateMemoryAblationSupplementalEvidence(
            accounting_side=_side(
                strategy_id=strategy_id,
                workload_id=self._experiment_id,
                task_id=memory_experiment_accounting_task_id(self._experiment_id),
                source_id=memory_experiment_accounting_source_id(
                    trial.request.case.case_revision,
                    trial.repetition,
                ),
                attempts=(attempt,),
            ),
            cache_evidence_state=ExternalPrivateMemoryAblationCacheEvidenceState.AVAILABLE,
            cache_read_tokens=8,
            cache_write_tokens=4,
        )


async def _campaign_fixture(
    private_root: Path,
    *,
    artifact_name: str,
    strategy: ExternalPrivateMemoryAblationScheduleStrategy = (
        ExternalPrivateMemoryAblationScheduleStrategy.FIXED
    ),
    budgeted_live: bool = False,
    prepare_context: bool = False,
    memory_attribution: bool = False,
) -> _CampaignFixture:
    variant_ids = reference_campaign.CAUSAL_MEMORY_CAMPAIGN_VARIANTS
    if memory_attribution:
        variant_ids = variant_ids[:2]
    private_root.mkdir(parents=True, exist_ok=True)
    corpus_path = private_root / "private-corpus.json"
    state_directory = private_root / "state"
    state_directory.mkdir(exist_ok=True)
    provider = _OrderedCampaignProvider()
    sessions = SQLiteSessionStore(state_directory / "sessions.sqlite")
    budgets = SQLiteBudgetLedger(state_directory / "budgets.sqlite")
    factory = reference_campaign._CampaignApplicationFactory(
        sessions=sessions,
        budgets=budgets,
        provider=provider,
    )
    starting_policy = reference_campaign._recall_policy()
    off_policy = reference_campaign._recall_policy(reference_campaign.AutomaticRecallMode.OFF)
    target = reference_campaign._target(factory, policy=starting_policy)
    off_target = reference_campaign._target(factory, policy=off_policy)
    if prepare_context:
        if not budgeted_live:
            raise ValueError("Preparation fixture requires budgeted live admission.")
        history = (
            Message.text("user", "Retain this authored background."),
            Message.text("assistant", "Background retained."),
            Message.text("user", "Now change to another subject."),
            Message.text("assistant", "Subject changed."),
        )
        target = target.model_copy(update={"bootstrap_messages": history})
        off_target = off_target.model_copy(update={"bootstrap_messages": history})
    private_document = reference_campaign.build_causal_memory_reference_corpus(
        app_manifest=target.app.describe()
    )
    if memory_attribution:
        private_document = type(private_document).create(
            target_key=private_document.target_key,
            evidence_policy=private_document.evidence_policy,
            suites=private_document.suites,
            cases=tuple(
                type(case).create(
                    id=case.id,
                    suite_id=case.suite_id,
                    name=case.name,
                    source=case.source,
                    input=case.input,
                    assertions=(
                        *case.assertions,
                        MemoryAttributionAssertionSpec(
                            id="memory-attribution",
                            min_admitted_items=0,
                            min_provider_exposures=0,
                        ),
                    ),
                )
                for case in private_document.cases
            ),
        )
    pricing = None
    if budgeted_live:
        # Live-classified authority with a credential-free scripted transport.
        factory.provider_execution_mode = MemoryInterventionProviderExecutionMode.LIVE
        pricing = PriceBook(
            price_book_version="private-live-fixture-v1",
            generated_at="2026-09-03",
            prices=(
                ModelPrice(
                    provider_name=provider.name,
                    model=reference_campaign._MODEL,
                    schedules=(
                        PriceSchedule(
                            pricing=TieredPricing(
                                standard=(
                                    PriceTier(
                                        input_per_million=Decimal("1"),
                                        output_per_million=Decimal("1"),
                                    ),
                                )
                            ),
                            provenance=Provenance(
                                source="fixture",
                                url="https://example.invalid/pricing",
                                as_of="2026-09-03",
                            ),
                        ),
                    ),
                ),
            ),
        )
        request_base = target.request_base.model_copy(
            update={
                "limits": RunLimits(scope="session", max_total_tokens=30_000),
                "budget_limits": (
                    BudgetLimit(
                        scope="causal",
                        key=EXTERNAL_PRIVATE_MEMORY_ABLATION_TRIAL_BUDGET_KEY,
                        max_estimated_cost=Decimal("0.08"),
                        pricing=pricing,
                        reservation=BudgetReservation(
                            max_input_tokens=20_000, max_output_tokens=1_000
                        ),
                    ),
                ),
            }
        )
        target = target.model_copy(update={"request_base": request_base, "price_book": pricing})
        off_target = off_target.model_copy(
            update={"request_base": request_base, "price_book": pricing}
        )
        private_document = type(private_document).create(
            target_key=private_document.target_key,
            evidence_policy=private_document.evidence_policy,
            suites=private_document.suites,
            cases=private_document.cases,
            pricing_profile=pricing_profile_identity(pricing),
        )
    if prepare_context:
        private_document = type(private_document).create(
            target_key=private_document.target_key,
            evidence_policy=private_document.evidence_policy,
            cases=private_document.cases,
            pricing_profile=private_document.pricing_profile,
            suites=tuple(
                type(suite).create(
                    id=suite.id,
                    name=suite.name,
                    description=suite.description,
                    trial_request=suite.trial_request.model_copy(update={"timeout_seconds": 120}),
                )
                for suite in private_document.suites
            ),
        )
    if not corpus_path.exists():
        corpus_path.write_text(eval_corpus_to_json(private_document), encoding="utf-8")
    corpus = load_external_private_memory_ablation_corpus(
        corpus_path,
        approved_private_root=private_root,
    )
    ordered_coordinates = tuple(
        (case.id, repetition, variant_id)
        for case in corpus.document.cases
        for repetition in range(1, reference_campaign.CAUSAL_MEMORY_CAMPAIGN_REPETITIONS + 1)
        for variant_id in variant_ids
    )
    provider.set_outputs(
        tuple(
            reference_campaign._output(case_id, variant_id)
            for case_id, _, variant_id in ordered_coordinates
        )
    )
    profile_policy = EvalExecutionProfilePolicyV1(
        fixture_strategy="application_managed",
        reset_strategy="application_managed",
        effect_posture="isolated_application_authority",
        isolation_revision=reference_campaign._ISOLATION_REVISION,
        max_trials=reference_campaign.CAUSAL_MEMORY_CAMPAIGN_REPETITIONS,
        max_concurrency=1,
    )
    prepared_profile = await prepare_eval_execution_profile(
        target,
        profile_id="external-private-reference",
        label="External private reference",
        source="explicit",
        app_manifest_fingerprint=target.app.describe().fingerprint,
        policy=profile_policy,
    )
    prepared_off = await prepare_eval_execution_profile(
        off_target,
        profile_id="external-private-reference",
        label="External private reference",
        source="explicit",
        app_manifest_fingerprint=off_target.app.describe().fingerprint,
        policy=profile_policy,
    )
    factory.profile_by_policy[starting_policy.fingerprint()] = (
        prepared_profile.binding.runtime_execution_profile.fingerprint
    )
    factory.profile_by_policy[off_policy.fingerprint()] = (
        prepared_off.binding.runtime_execution_profile.fingerprint
    )
    snapshot = reference_campaign._snapshot(
        execution_profile_snapshot_ref(prepared_profile.binding.runtime_execution_profile),
        starting_policy,
    )
    snapshot_store = SQLiteAgentSnapshotStore(state_directory / "snapshots.sqlite")
    existing_snapshot = await snapshot_store.load_snapshot(snapshot.fingerprint)
    if existing_snapshot is None:
        await snapshot_store.save_snapshot(snapshot)
    else:
        assert existing_snapshot == snapshot
    specs = reference_campaign._intervention_specs(snapshot, starting_policy)
    profiles = {
        variant_id: (
            (prepared_off.snapshot, prepared_off.binding)
            if variant_id == "automatic-recall-off"
            else (prepared_profile.snapshot, prepared_profile.binding)
        )
        for variant_id in variant_ids
    }
    variants = tuple(
        MemoryExperimentVariant(
            variant_id=variant_id,
            candidate_id=variant_id,
            spec=specs[variant_id],
            execution_profile=profiles[variant_id][0],
            execution_profile_binding=profiles[variant_id][1],
            evaluator_fingerprint=reference_campaign._EVALUATOR_FINGERPRINT,
        )
        for variant_id in variant_ids
    )
    metric_bindings = tuple(
        MemoryMetricBinding(
            role=role,
            assertion_id=role.value,
            assertion_revision=assertion_spec_revision(
                next(
                    assertion
                    for assertion in corpus.document.cases[0].assertions
                    if assertion.id == role.value
                )
            ),
        )
        for role in sorted(reference_campaign._METRIC_MARKERS, key=str)
    )
    experiment = MemoryExperimentReportRequest(
        experiment_id="external-private-reference-v1",
        cases=tuple(
            MemoryExperimentCase(case_id=case.id, case_revision=case.revision)
            for case in corpus.document.cases
        ),
        repetitions=reference_campaign.CAUSAL_MEMORY_CAMPAIGN_REPETITIONS,
        baseline_variant_id="as-declared",
        variants=variants,
        metric_bindings=metric_bindings,
        ranking=(
            MemoryRankingTerm(
                role=MemoryMetricRole.TASK_QUALITY,
                direction=MemoryMetricDirection.HIGHER_IS_BETTER,
            ),
        ),
        gates=MemoryExperimentGatePolicy(
            required_metric_roles=tuple(sorted(reference_campaign._METRIC_MARKERS, key=str)),
            metric_gates=(
                MemoryMetricGate(role=MemoryMetricRole.SAFETY, minimum=1.0),
                MemoryMetricGate(
                    role=MemoryMetricRole.UNAUTHORIZED_EXPOSURE_AVOIDANCE,
                    minimum=1.0,
                ),
            ),
            minimum_comparable_pairs=1,
            require_priced_cost=budgeted_live,
            maximum_candidate_cost=Decimal("20") if budgeted_live else None,
            cost_currency="USD" if budgeted_live else None,
        ),
    )
    trials = tuple(
        ExternalPrivateMemoryAblationTrial(
            case_id=case.id,
            repetition=repetition,
            variant_id=variant_id,
            request=reference_campaign._trial_request(
                case=case,
                spec=specs[variant_id],
                variant_id=variant_id,
                repetition=repetition,
            ),
        )
        for case in corpus.document.cases
        for repetition in range(1, reference_campaign.CAUSAL_MEMORY_CAMPAIGN_REPETITIONS + 1)
        for variant_id in variant_ids
    )
    if budgeted_live:
        compiled = compile_corpus_suite(
            corpus.document, target, reference_campaign.CAUSAL_MEMORY_CAMPAIGN_SUITE_ID
        )
        requests = {case.id: case.request for case in compiled.suite.cases}
        trials = tuple(
            replace(
                trial,
                request=trial.request.model_copy(
                    update={
                        "run_request": requests[trial.case_id],
                        "context_preparation": "compact_then_resume" if prepare_context else "none",
                        "timeout_seconds": 120
                        if prepare_context
                        else trial.request.timeout_seconds,
                    }
                ),
            )
            for trial in trials
        )
    schedule_policy = ExternalPrivateMemoryAblationSchedulePolicy(
        strategy=strategy,
        seed_fingerprint=(
            None
            if strategy is ExternalPrivateMemoryAblationScheduleStrategy.FIXED
            else _digest("external-private-schedule")
        ),
    )
    destination = external_private_memory_ablation_destination(
        state_storage_id="approved-private-state",
        destination_id="approved-private-report",
        approved_private_root=private_root,
        state_directory=state_directory,
        artifact_directory=artifact_name,
    )
    manifest = target.app.describe()
    authorization = ExternalPrivateMemoryAblationAuthorization(
        authorization_id="approved-private-campaign",
        context_preparation="compact_then_resume" if prepare_context else "none",
        valid_from=datetime(2026, 9, 1, tzinfo=UTC),
        valid_through=datetime(2026, 9, 10, tzinfo=UTC),
        corpus_revision=corpus.document.revision,
        suite_id=reference_campaign.CAUSAL_MEMORY_CAMPAIGN_SUITE_ID,
        experiment_revision=external_private_memory_ablation_experiment_revision(experiment),
        schedule_policy_revision=schedule_policy.revision,
        target_key=target.key,
        application_release_id=target.application_release_id,
        app_manifest_fingerprint=manifest.fingerprint,
        snapshot_fingerprint=snapshot.fingerprint,
        evaluator_fingerprint=reference_campaign._EVALUATOR_FINGERPRINT,
        provider_name=provider.name,
        model=reference_campaign._MODEL,
        provider_configuration_fingerprint=factory.provider_configuration_fingerprint,
        evidence_policy_revision=target.evidence_policy.revision,
        pricing_profile_fingerprint=None
        if pricing is None
        else pricing_profile_identity(pricing).fingerprint,
        evidence_collector_fingerprint=(
            _AccountingEvidenceCollector(experiment=experiment).collector_fingerprint
            if budgeted_live
            else None
        ),
        redaction_policy_revision="sha256:" + _digest("private-redaction-policy"),
        retention_policy_revision="sha256:" + _digest("private-retention-policy"),
        state_storage_id="approved-private-state",
        report_destination_id="approved-private-report",
        report_destination_fingerprint=destination.fingerprint,
        execution_mode=(
            ExternalPrivateMemoryAblationExecutionMode.LIVE
            if budgeted_live
            else ExternalPrivateMemoryAblationExecutionMode.HERMETIC
        ),
        live_execution_authorization_id="approved-fixture-live" if budgeted_live else None,
        allowed_variant_ids=tuple(sorted(variant_ids)),
        allowed_variant_kinds=tuple(sorted({variant.spec.kind for variant in variants}, key=str)),
        minimum_cases=len(corpus.document.cases),
        maximum_cases=len(corpus.document.cases),
        minimum_repetitions=reference_campaign.CAUSAL_MEMORY_CAMPAIGN_REPETITIONS,
        maximum_repetitions=reference_campaign.CAUSAL_MEMORY_CAMPAIGN_REPETITIONS,
        maximum_total_trials=len(trials),
        maximum_concurrency=1,
        maximum_timeout_seconds=120 if prepare_context else 30,
        maximum_model_steps=1,
        maximum_report_evidence_bytes_per_trial=(
            EXTERNAL_PRIVATE_MEMORY_ABLATION_MIN_REPORT_EVIDENCE_BYTES_PER_TRIAL
        ),
        maximum_supplemental_evidence_bytes_per_trial=(4 << 10) if budgeted_live else None,
        maximum_total_tokens_per_trial=30_000 if budgeted_live else None,
        maximum_estimated_cost_total=Decimal("20") if budgeted_live else None,
        cost_currency="USD" if budgeted_live else None,
        cache_evidence_policy=(
            ExternalPrivateMemoryAblationCacheEvidencePolicy.BEST_EFFORT
            if budgeted_live
            else ExternalPrivateMemoryAblationCacheEvidencePolicy.UNAVAILABLE
        ),
    )
    evaluator = reference_campaign._CampaignEvaluator(
        corpus=corpus.document,
        factory=factory,
    )
    for trial in trials:
        evaluator.register_trial(trial.request.execution_id, trial.repetition)
    overlay = reference_campaign._CampaignOverlayProvider()
    executor = MemoryInterventionExecutor(
        snapshots=AgentSnapshotCoordinator(
            tuple(
                reference_campaign._SnapshotProvider(component) for component in snapshot.components
            ),
            store=snapshot_store,
            clock=lambda: reference_campaign._CAMPAIGN_TIME,
        ),
        executions=SQLiteMemoryInterventionExecutionStore(state_directory / "interventions.sqlite"),
        overlay_provider=overlay,
        runtime_runner=CayuMemoryInterventionRuntimeRunner(factory),
        evaluator=evaluator,
        request_keys={
            "private-key": MemoryInterventionRequestFingerprintKey(
                key_id="private-key",
                secret=b"external-private-reference-key-32",
            )
        },
        current_request_key_id="private-key",
        clock=lambda: reference_campaign._CAMPAIGN_TIME,
    )
    prepared = prepare_external_private_memory_ablation(
        corpus=corpus,
        target=target,
        snapshot=snapshot,
        experiment=experiment,
        trials=trials,
        executor=executor,
        authorization=authorization,
        schedule_policy=schedule_policy,
        destination=destination,
        now=_NOW,
    )
    return _CampaignFixture(
        prepared=prepared,
        executor=executor,
        provider=provider,
        sessions=sessions,
        budgets=budgets,
    )


def _exercise_posix_private_artifact_hardening(
    result: ExternalPrivateMemoryAblationResult,
    destination: ExternalPrivateMemoryAblationDestination,
    monkeypatch: pytest.MonkeyPatch,
    private_filenames: set[str],
) -> None:
    original_open = private_ablation.os.open
    held_stage = destination.artifact_directory.parent / ".held-stage"
    redirected_stage = destination.artifact_directory.parent / ".redirected-stage"
    staging_path: Path | None = None

    def open_after_stage_substitution(path, flags, mode=0o777, *, dir_fd=None):
        nonlocal staging_path
        if (
            staging_path is None
            and dir_fd is not None
            and isinstance(path, str)
            and path.startswith(".cayu-private-stage-")
            and flags & private_ablation.os.O_DIRECTORY
        ):
            staging_path = destination.artifact_directory.parent / path
            staging_path.rename(held_stage)
            staging_path.mkdir(mode=0o700)
            descriptor = original_open(path, flags, mode, dir_fd=dir_fd)
            staging_path.rename(redirected_stage)
            held_stage.rename(staging_path)
            return descriptor
        return original_open(path, flags, mode, dir_fd=dir_fd)

    with monkeypatch.context() as raced:
        raced.setattr(private_ablation.os, "open", open_after_stage_substitution)
        with pytest.raises(ValueError, match="staging directory changed during creation"):
            write_external_private_memory_ablation_artifacts(result, destination)
    assert staging_path is not None
    assert not any(path.name in private_filenames for path in staging_path.rglob("*"))
    assert not any(path.name in private_filenames for path in redirected_stage.rglob("*"))
    staging_path.rmdir()
    redirected_stage.rmdir()

    original_write_all = private_ablation._write_all_private_artifact_bytes
    write_failed = False

    def fail_first_staged_write(descriptor: int, content: bytes) -> None:
        nonlocal write_failed
        if not write_failed:
            write_failed = True
            private_ablation.os.write(descriptor, content[:1])
            raise OSError("injected private artifact write failure")
        original_write_all(descriptor, content)

    with monkeypatch.context() as failed_write:
        failed_write.setattr(
            private_ablation,
            "_write_all_private_artifact_bytes",
            fail_first_staged_write,
        )
        with pytest.raises(OSError, match="injected private artifact write failure"):
            write_external_private_memory_ablation_artifacts(result, destination)
    assert write_failed
    assert not destination.artifact_directory.exists()
    assert not tuple(destination.artifact_directory.parent.glob(".cayu-private-stage-*"))

    def inject_unexpected_stage_entry(phase: str) -> None:
        if phase != "payload_entries_synced":
            return
        stages = tuple(destination.artifact_directory.parent.glob(".cayu-private-stage-*"))
        assert len(stages) == 1
        (stages[0] / "unexpected").write_bytes(b"unexpected")

    with monkeypatch.context() as raced:
        raced.setattr(
            private_ablation,
            "_private_artifact_publication_fault",
            inject_unexpected_stage_entry,
        )
        with pytest.raises(ValueError, match="contains unexpected entries"):
            write_external_private_memory_ablation_artifacts(result, destination)
    assert not destination.artifact_directory.exists()
    stages = tuple(destination.artifact_directory.parent.glob(".cayu-private-stage-*"))
    assert len(stages) == 1
    assert {path.name for path in stages[0].iterdir()} == {"unexpected"}
    with pytest.raises(ValueError, match="contains unexpected entries"):
        write_external_private_memory_ablation_artifacts(result, destination)
    assert tuple(destination.artifact_directory.parent.glob(".cayu-private-stage-*")) == stages
    (stages[0] / "unexpected").unlink()
    stages[0].rmdir()

    def replace_staged_payload(phase: str) -> None:
        if phase != "payload_entries_synced":
            return
        stages = tuple(destination.artifact_directory.parent.glob(".cayu-private-stage-*"))
        assert len(stages) == 1
        report_path = stages[0] / EXTERNAL_PRIVATE_MEMORY_ABLATION_REPORT_FILENAME
        report_path.unlink()
        report_path.write_bytes(b"replaced")

    with monkeypatch.context() as raced:
        raced.setattr(
            private_ablation,
            "_private_artifact_publication_fault",
            replace_staged_payload,
        )
        with pytest.raises(ValueError, match="payload changed before publication"):
            write_external_private_memory_ablation_artifacts(result, destination)
    assert not destination.artifact_directory.exists()
    stages = tuple(destination.artifact_directory.parent.glob(".cayu-private-stage-*"))
    assert len(stages) == 1
    assert (
        stages[0] / EXTERNAL_PRIVATE_MEMORY_ABLATION_REPORT_FILENAME
    ).read_bytes() == b"replaced"
    for path in stages[0].iterdir():
        path.unlink()
    stages[0].rmdir()

    held_published = destination.artifact_directory.parent / ".held-published-artifacts"

    def replace_published_tree(phase: str) -> None:
        if phase != "tree_renamed":
            return
        destination.artifact_directory.rename(held_published)
        destination.artifact_directory.mkdir(mode=0o700)
        (destination.artifact_directory / "unowned").write_bytes(b"unowned")

    with monkeypatch.context() as raced:
        raced.setattr(
            private_ablation,
            "_private_artifact_publication_fault",
            replace_published_tree,
        )
        with pytest.raises(ValueError, match="changed during publication"):
            write_external_private_memory_ablation_artifacts(result, destination)
    assert (destination.artifact_directory / "unowned").read_bytes() == b"unowned"
    assert {path.name for path in held_published.iterdir()} == private_filenames
    (destination.artifact_directory / "unowned").unlink()
    destination.artifact_directory.rmdir()
    for path in held_published.iterdir():
        path.unlink()
    held_published.rmdir()

    report_content = (memory_experiment_report_to_json(result.report) + "\n").encode("utf-8")
    interrupted_stage = destination.artifact_directory.parent / (
        private_ablation._posix_private_artifact_stage_name(destination.artifact_directory.name)
    )
    interrupted_stage.mkdir(mode=0o700)
    interrupted_pending = interrupted_stage / (
        private_ablation._posix_private_artifact_pending_name(
            EXTERNAL_PRIVATE_MEMORY_ABLATION_REPORT_FILENAME,
            report_content,
        )
    )
    interrupted_pending.write_bytes(report_content[:1])
    interrupted_pending.chmod(0o600)

    recovered_partial = write_external_private_memory_ablation_artifacts(result, destination)
    assert {
        path.name for path in recovered_partial.artifact_directory.iterdir()
    } == private_filenames
    for path in recovered_partial.artifact_directory.iterdir():
        path.unlink()
    recovered_partial.artifact_directory.rmdir()

    crash_cases = (
        (
            "payload_entries_synced",
            {
                EXTERNAL_PRIVATE_MEMORY_ABLATION_REPORT_FILENAME,
                EXTERNAL_PRIVATE_MEMORY_ABLATION_METHODOLOGY_FILENAME,
            },
        ),
        (
            "completion_entry_synced",
            private_filenames,
        ),
    )
    for crash_phase, expected_staged_names in crash_cases:

        def interrupt_publication(phase: str, *, expected: str = crash_phase) -> None:
            if phase == expected:
                raise RuntimeError("simulated process interruption")

        with monkeypatch.context() as interrupted:
            interrupted.setattr(
                private_ablation,
                "_private_artifact_publication_fault",
                interrupt_publication,
            )
            interrupted.setattr(
                private_ablation,
                "_remove_incomplete_posix_private_stage",
                lambda *_args, **_kwargs: None,
            )
            with pytest.raises(RuntimeError, match="simulated process interruption"):
                write_external_private_memory_ablation_artifacts(result, destination)
        assert not destination.artifact_directory.exists()
        stages = tuple(destination.artifact_directory.parent.glob(".cayu-private-stage-*"))
        assert len(stages) == 1
        assert {path.name for path in stages[0].iterdir()} == expected_staged_names

        recovered = write_external_private_memory_ablation_artifacts(result, destination)
        assert {path.name for path in recovered.artifact_directory.iterdir()} == private_filenames
        for path in recovered.artifact_directory.iterdir():
            path.unlink()
        recovered.artifact_directory.rmdir()

    def interrupt_after_publish(phase: str) -> None:
        if phase == "tree_renamed":
            raise RuntimeError("simulated ambiguous publication acknowledgement")

    with monkeypatch.context() as interrupted:
        interrupted.setattr(
            private_ablation,
            "_private_artifact_publication_fault",
            interrupt_after_publish,
        )
        with pytest.raises(RuntimeError, match="ambiguous publication acknowledgement"):
            write_external_private_memory_ablation_artifacts(result, destination)
    assert {path.name for path in destination.artifact_directory.iterdir()} == private_filenames
    recovered_publish = write_external_private_memory_ablation_artifacts(result, destination)
    assert {
        path.name for path in recovered_publish.artifact_directory.iterdir()
    } == private_filenames
    for path in recovered_publish.artifact_directory.iterdir():
        path.unlink()
    recovered_publish.artifact_directory.rmdir()


def test_windows_private_artifacts_sync_payloads_before_protected_completion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "artifacts"
    stage_path = tmp_path / "stage"
    stage_path.mkdir()
    contents = {
        EXTERNAL_PRIVATE_MEMORY_ABLATION_REPORT_FILENAME: b"report",
        EXTERNAL_PRIVATE_MEMORY_ABLATION_METHODOLOGY_FILENAME: b"methodology",
        EXTERNAL_PRIVATE_MEMORY_ABLATION_COMPLETION_FILENAME: b"complete",
    }
    synced: list[tuple[str, bool]] = []

    class _Stage:
        def write_tree(self, values, **_kwargs) -> None:
            for name, content in values.items():
                (stage_path / name).write_bytes(content)

        def _specialized_path(self) -> Path:
            return stage_path

        def capture_owned_identity(self) -> os.stat_result:
            return stage_path.stat()

    def publish(_destination: Path, **kwargs) -> None:
        assert kwargs["preserve_windows_destination_dacl"] is True
        assert kwargs["policy"] is guarded_publication.DestinationPolicy.ABSENT_OR_EMPTY
        kwargs["populate"](_Stage())

    monkeypatch.setattr(private_ablation, "publish_guarded_tree", publish)
    monkeypatch.setattr(
        private_ablation,
        "_windows_directory_namespace_fence",
        lambda _path: nullcontext(),
    )
    monkeypatch.setattr(
        private_ablation,
        "_assert_windows_directory_dacl_is_protected",
        lambda _path: None,
    )
    monkeypatch.setattr(
        private_ablation,
        "_sync_windows_path",
        lambda path, *, directory: synced.append((path.name, directory)),
    )
    parent_identity = private_ablation._PrivateFilesystemIdentity.from_guarded_identity(
        guarded_publication._capture_parent(tmp_path)
    )

    private_ablation._publish_windows_private_artifact_tree(
        destination,
        contents=contents,
        expected_parent_identity=parent_identity,
        request_digest="sha256:" + _digest("windows-publication"),
    )

    assert synced == [
        (EXTERNAL_PRIVATE_MEMORY_ABLATION_REPORT_FILENAME, False),
        (EXTERNAL_PRIVATE_MEMORY_ABLATION_METHODOLOGY_FILENAME, False),
        (stage_path.name, True),
        (EXTERNAL_PRIVATE_MEMORY_ABLATION_COMPLETION_FILENAME, False),
        (stage_path.name, True),
    ]


def test_private_campaign_runs_and_publishes_only_bounded_private_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def exercise():
        fixture = await _campaign_fixture(tmp_path, artifact_name="artifacts")
        try:
            result = await run_external_private_memory_ablation(
                fixture.prepared,
                fixture.executor,
                clock=lambda: _NOW,
            )
        finally:
            await fixture.close()
        return fixture, result

    fixture, result = asyncio.run(exercise())

    assert result.methodology.status is ExternalPrivateMemoryAblationRunStatus.COMPLETE
    assert result.methodology.run_failure_code is None
    assert len(result.report.rows) == len(fixture.prepared.schedule)
    assert all(row.availability is MemoryTrialAvailability.AVAILABLE for row in result.report.rows)
    assert len(fixture.provider.requests) == len(result.report.rows)
    runner = fixture.executor.runtime_runner
    assert type(runner) is CayuMemoryInterventionRuntimeRunner
    factory = runner.factory
    assert type(factory) is reference_campaign._CampaignApplicationFactory
    assert factory.app_by_session
    assert all(app.budget_ledger is fixture.budgets for app in factory.app_by_session.values())
    assert fixture.executor.durable_state_paths().runtime_budget_ledger == (
        fixture.budgets.path.resolve(),
    )
    cleanup = tuple(
        app.provider_operation_cancellation_status() for app in factory.app_by_session.values()
    )
    assert all(status.admissions_sealed for status in cleanup)
    assert all(status.active_owners == 0 for status in cleanup)
    assert result.report.selected_variant_id == "as-declared"
    assert result.methodology.limitations == (
        ExternalPrivateMemoryAblationLimitation.ACCOUNTING_EVIDENCE_UNAVAILABLE,
        ExternalPrivateMemoryAblationLimitation.CACHE_EVIDENCE_UNAVAILABLE,
        ExternalPrivateMemoryAblationLimitation.MEMORY_OVERHEAD_UNAVAILABLE,
    )
    serialized_methodology = result.methodology.model_dump_json()
    assert "What day does the current Atlas release record specify?" not in serialized_methodology
    assert reference_campaign._PRIMARY_CURRENT_TEXT not in serialized_methodology
    serialized_report = memory_experiment_report_to_json(result.report)
    assert "What day does the current Atlas release record specify?" not in serialized_report
    assert reference_campaign._PRIMARY_CURRENT_TEXT not in serialized_report
    assert "ATLAS-WEEKDAY=" not in serialized_report

    other_root = tmp_path / "other-private-root"
    other_state = other_root / "state"
    other_state.mkdir(parents=True)
    substituted_destination = external_private_memory_ablation_destination(
        state_storage_id=fixture.prepared.destination.state_storage_id,
        destination_id=fixture.prepared.destination.destination_id,
        approved_private_root=other_root,
        state_directory=other_state,
        artifact_directory="substituted-artifacts",
    )
    with pytest.raises(ValueError, match="does not authorize these private destination paths"):
        write_external_private_memory_ablation_artifacts(result, substituted_destination)
    assert not substituted_destination.artifact_directory.exists()

    with monkeypatch.context() as bounded:
        bounded.setattr(
            private_ablation,
            "EXTERNAL_PRIVATE_MEMORY_ABLATION_MAX_ARTIFACT_BYTES",
            1,
        )
        with pytest.raises(ValueError, match="artifacts exceed their byte bound"):
            write_external_private_memory_ablation_artifacts(
                result,
                fixture.prepared.destination,
            )
    assert not fixture.prepared.destination.artifact_directory.exists()

    publisher_name = (
        "_publish_windows_private_artifact_tree"
        if os.name == "nt"
        else "_publish_posix_private_artifact_tree"
    )
    original_publish = getattr(private_ablation, publisher_name)

    def publish_after_empty_destination_appears(destination: Path, **kwargs):
        destination.mkdir()
        return original_publish(destination, **kwargs)

    with monkeypatch.context() as raced:
        raced.setattr(
            private_ablation,
            publisher_name,
            publish_after_empty_destination_appears,
        )
        with pytest.raises(FileExistsError, match="already exists"):
            write_external_private_memory_ablation_artifacts(
                result,
                fixture.prepared.destination,
            )
    assert not any(fixture.prepared.destination.artifact_directory.iterdir())
    fixture.prepared.destination.artifact_directory.rmdir()

    held_root = tmp_path.parent / f"{tmp_path.name}-authorized"
    redirected_root = tmp_path.parent / f"{tmp_path.name}-redirected"

    def publish_after_root_substitution(*args, **kwargs):
        tmp_path.rename(held_root)
        tmp_path.mkdir()
        try:
            return original_publish(*args, **kwargs)
        finally:
            tmp_path.rename(redirected_root)
            held_root.rename(tmp_path)

    with monkeypatch.context() as raced:
        raced.setattr(
            private_ablation,
            publisher_name,
            publish_after_root_substitution,
        )
        with pytest.raises(ValueError, match="artifact parent changed"):
            write_external_private_memory_ablation_artifacts(
                result,
                fixture.prepared.destination,
            )
    private_filenames = {
        EXTERNAL_PRIVATE_MEMORY_ABLATION_REPORT_FILENAME,
        EXTERNAL_PRIVATE_MEMORY_ABLATION_METHODOLOGY_FILENAME,
        EXTERNAL_PRIVATE_MEMORY_ABLATION_COMPLETION_FILENAME,
    }
    assert not any(path.name in private_filenames for path in redirected_root.rglob("*"))

    if os.name != "nt":
        _exercise_posix_private_artifact_hardening(
            result,
            fixture.prepared.destination,
            monkeypatch,
            private_filenames,
        )

    publication_observations: list[tuple[str, frozenset[str]]] = []

    def observe_publication(phase: str) -> None:
        names = frozenset()
        if fixture.prepared.destination.artifact_directory.exists():
            names = frozenset(
                path.name for path in fixture.prepared.destination.artifact_directory.iterdir()
            )
        publication_observations.append((phase, names))

    with monkeypatch.context() as publication_probe:
        if os.name == "nt":
            publication_probe.setattr(
                guarded_publication,
                "_publication_fault",
                observe_publication,
            )
        else:
            publication_probe.setattr(
                private_ablation,
                "_private_artifact_publication_fault",
                observe_publication,
            )
        paths = write_external_private_memory_ablation_artifacts(
            result,
            fixture.prepared.destination,
        )
    completed_filenames = frozenset(private_filenames)
    if os.name == "nt":
        assert ("stage_synced", frozenset()) in publication_observations
        assert ("tree_renamed", completed_filenames) in publication_observations
    else:
        assert ("payload_entries_synced", frozenset()) in publication_observations
        assert ("completion_entry_synced", frozenset()) in publication_observations
        assert ("stage_synced", frozenset()) in publication_observations
        assert ("tree_renamed", completed_filenames) in publication_observations
    assert paths.report_path.name == EXTERNAL_PRIVATE_MEMORY_ABLATION_REPORT_FILENAME
    assert paths.methodology_path.name == EXTERNAL_PRIVATE_MEMORY_ABLATION_METHODOLOGY_FILENAME
    assert paths.completion_path.name == EXTERNAL_PRIVATE_MEMORY_ABLATION_COMPLETION_FILENAME
    assert memory_experiment_report_to_json(result.report) in paths.report_path.read_text()
    assert stat.S_IMODE(paths.artifact_directory.stat().st_mode) == 0o700
    assert stat.S_IMODE(paths.report_path.stat().st_mode) == 0o600
    assert stat.S_IMODE(paths.methodology_path.stat().st_mode) == 0o600
    replayed_paths = write_external_private_memory_ablation_artifacts(
        result,
        fixture.prepared.destination,
    )
    assert replayed_paths == paths


def test_private_corpus_read_rejects_ancestor_substitution_without_redirection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def make_fixture():
        return await _campaign_fixture(tmp_path / "fixture", artifact_name="artifacts")

    fixture = asyncio.run(make_fixture())
    try:
        corpus_bytes = fixture.prepared.corpus.source_path.read_bytes()
    finally:
        asyncio.run(fixture.close())

    private_root = tmp_path / "race-private"
    held_root = tmp_path / "race-private-held"
    outside_root = tmp_path / "race-private-outside"
    private_root.mkdir()
    outside_root.mkdir()
    source = private_root / "corpus.json"
    source.write_bytes(corpus_bytes)
    (outside_root / "corpus.json").write_text("{}", encoding="utf-8")
    original_open = private_ablation.os.open
    original_read = private_ablation.os.read
    substituted = False
    root_open_count = 0
    observed_content: list[bytes] = []

    def substituting_open(path, flags, mode=0o777, *, dir_fd=None):
        nonlocal root_open_count, substituted
        descriptor = original_open(path, flags, mode, dir_fd=dir_fd)
        if (
            not substituted
            and dir_fd is None
            and Path(path) == private_root
            and flags & private_ablation.os.O_DIRECTORY
        ):
            root_open_count += 1
            if root_open_count == 2:
                private_root.rename(held_root)
                private_root.symlink_to(outside_root, target_is_directory=True)
                substituted = True
        return descriptor

    def recording_read(descriptor: int, count: int) -> bytes:
        content = original_read(descriptor, count)
        observed_content.append(content)
        return content

    with monkeypatch.context() as raced:
        raced.setattr(private_ablation.os, "open", substituting_open)
        raced.setattr(private_ablation.os, "read", recording_read)
        with pytest.raises(ValueError, match="changed while private data was accessed"):
            load_external_private_memory_ablation_corpus(
                source,
                approved_private_root=private_root,
            )

    assert substituted
    assert b"".join(observed_content) == corpus_bytes


def test_private_corpus_loader_rejects_root_substitution_before_pinning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def make_fixture():
        return await _campaign_fixture(tmp_path / "fixture", artifact_name="artifacts")

    fixture = asyncio.run(make_fixture())
    try:
        corpus_bytes = fixture.prepared.corpus.source_path.read_bytes()
    finally:
        asyncio.run(fixture.close())

    private_root = tmp_path / "private"
    held_root = tmp_path / "private-held"
    private_root.mkdir()
    source = private_root / "corpus.json"
    source.write_bytes(corpus_bytes)
    original_open = private_ablation.os.open
    root_open_count = 0

    def substituting_open(path, flags, mode=0o777, *, dir_fd=None):
        nonlocal root_open_count
        if (
            dir_fd is None
            and Path(path) == private_root
            and flags & private_ablation.os.O_DIRECTORY
        ):
            root_open_count += 1
            if root_open_count == 2:
                private_root.rename(held_root)
                private_root.mkdir()
        return original_open(path, flags, mode, dir_fd=dir_fd)

    try:
        with monkeypatch.context() as raced:
            raced.setattr(private_ablation.os, "open", substituting_open)
            with pytest.raises(ValueError, match="approved_private_root changed"):
                load_external_private_memory_ablation_corpus(
                    source,
                    approved_private_root=private_root,
                )
    finally:
        if held_root.exists():
            private_root.rmdir()
            held_root.rename(private_root)
    assert root_open_count == 2


def test_private_destination_rejects_root_substitution_before_pinning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    private_root = tmp_path / "private"
    held_root = tmp_path / "private-held"
    state = private_root / "state"
    state.mkdir(parents=True)
    original_open = private_ablation.os.open
    root_open_count = 0

    def substituting_open(path, flags, mode=0o777, *, dir_fd=None):
        nonlocal root_open_count
        if (
            dir_fd is None
            and Path(path) == private_root
            and flags & private_ablation.os.O_DIRECTORY
        ):
            root_open_count += 1
            if root_open_count == 2:
                private_root.rename(held_root)
                private_root.mkdir()
        return original_open(path, flags, mode, dir_fd=dir_fd)

    try:
        with monkeypatch.context() as raced:
            raced.setattr(private_ablation.os, "open", substituting_open)
            with pytest.raises(ValueError, match="approved_private_root changed"):
                external_private_memory_ablation_destination(
                    state_storage_id="state",
                    destination_id="artifacts",
                    approved_private_root=private_root,
                    state_directory=state,
                    artifact_directory=private_root / "artifacts",
                )
    finally:
        if held_root.exists():
            private_root.rmdir()
            held_root.rename(private_root)
    assert root_open_count == 2


def test_private_filesystem_authority_leases_close_independently(tmp_path: Path) -> None:
    async def exercise():
        fixture = await _campaign_fixture(tmp_path, artifact_name="artifacts")
        try:
            corpus_lease = private_ablation._validate_external_private_corpus(
                fixture.prepared.corpus
            )
            destination_lease = private_ablation._validate_private_destination(
                fixture.prepared.destination
            )
            corpus_lease.close()
            destination_lease.close()
            with pytest.raises(ValueError, match="authority is closed"):
                private_ablation._validate_external_private_corpus(corpus_lease)
            with pytest.raises(ValueError, match="authority is closed"):
                private_ablation._validate_private_destination(destination_lease)
            assert fixture.prepared.corpus.__enter__() is fixture.prepared.corpus
            assert fixture.prepared.destination.__enter__() is fixture.prepared.destination
            await asyncio.gather(*(asyncio.to_thread(fixture.prepared.close) for _ in range(8)))
            fixture.prepared.close()
            with pytest.raises(ValueError, match="authority is closed"):
                fixture.prepared.__enter__()
            with pytest.raises(ValueError, match="authority is closed"):
                await run_external_private_memory_ablation(
                    fixture.prepared,
                    fixture.executor,
                    clock=lambda: _NOW,
                )
            assert fixture.provider.requests == []
        finally:
            fixture.prepared.close()
            await fixture.close()

    asyncio.run(exercise())


@pytest.mark.parametrize("changed_field", ["messages", "metadata", "max_steps"])
def test_private_campaign_rejects_changed_prepared_trial_input(
    tmp_path: Path,
    changed_field: str,
) -> None:
    async def exercise() -> None:
        fixture = await _campaign_fixture(tmp_path, artifact_name="artifacts")
        trial = fixture.prepared.scheduled_trials[0]
        request = trial.request.run_request
        if changed_field == "messages":
            request.messages.append(Message.text("user", "Changed after campaign preflight."))
        elif changed_field == "metadata":
            request.metadata["changed_after_preflight"] = True
        else:
            request.max_steps += 1
        try:
            with pytest.raises(ValueError, match="Trial input differs from the campaign preflight"):
                await run_external_private_memory_ablation(
                    fixture.prepared,
                    fixture.executor,
                    clock=lambda: _NOW,
                )
            assert fixture.provider.requests == []
            assert await fixture.executor.executions.load(trial.request.execution_id) is None
        finally:
            fixture.prepared.close()
            await fixture.close()

    asyncio.run(exercise())


def test_private_campaign_detaches_trial_inputs_before_waiting_for_admission(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def exercise() -> None:
        fixture = await _campaign_fixture(tmp_path, artifact_name="artifacts")
        prepared = fixture.prepared
        admission_started = asyncio.Event()
        release_admission = asyncio.Event()
        original_execute = fixture.executor.execute_trial
        clock_calls = 0

        def clock() -> datetime:
            nonlocal clock_calls
            clock_calls += 1
            # Admit two trials, then stop the rest of the matrix.
            if clock_calls <= 3:
                return _NOW
            return prepared.authorization.valid_through + timedelta(seconds=1)

        async def execute_after_admission_wait(request, **kwargs):
            admission_started.set()
            await release_admission.wait()
            return await original_execute(request, **kwargs)

        monkeypatch.setattr(fixture.executor, "execute_trial", execute_after_admission_wait)
        task = asyncio.create_task(
            run_external_private_memory_ablation(prepared, fixture.executor, clock=clock)
        )
        try:
            await asyncio.wait_for(admission_started.wait(), timeout=10)
            for trial in prepared.scheduled_trials[:2]:
                trial.request.run_request.messages.append(
                    Message.text("user", "POST_PREFLIGHT_CHANGED_INPUT")
                )
            release_admission.set()
            result = await task
            assert len(fixture.provider.requests) == 2
            assert all(
                "POST_PREFLIGHT_CHANGED_INPUT" not in request.model_dump_json()
                for request in fixture.provider.requests
            )
            assert all(
                trial.execution_status is MemoryInterventionExecutionStatus.COMPLETED
                for trial in result.methodology.trials[:2]
            )
            assert result.methodology.run_failure_code is (
                ExternalPrivateMemoryAblationRunFailureCode.AUTHORIZATION_EXPIRED
            )
            assert result.methodology.failure_ordinal == 3
        finally:
            release_admission.set()
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
            prepared.close()
            await fixture.close()

    asyncio.run(exercise())


def test_private_campaign_rejects_replaced_state_root_before_provider_dispatch(
    tmp_path: Path,
) -> None:
    async def exercise():
        fixture = await _campaign_fixture(tmp_path, artifact_name="artifacts")
        state = fixture.prepared.destination.state_directory
        held_state = state.with_name("state-authorized")
        state.rename(held_state)
        state.mkdir()
        try:
            with pytest.raises(
                ValueError,
                match="Executor identity or private state differs",
            ):
                await run_external_private_memory_ablation(
                    fixture.prepared,
                    fixture.executor,
                    clock=lambda: _NOW,
                )
            assert fixture.provider.requests == []
        finally:
            await fixture.close()

    asyncio.run(exercise())


def test_private_campaign_rechecks_state_authority_inside_executor_admission(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def exercise():
        fixture = await _campaign_fixture(tmp_path, artifact_name="artifacts")
        state = fixture.prepared.destination.state_directory
        held_state = state.with_name("state-authorized")
        original_execute = fixture.executor.execute_trial
        substituted = False

        async def execute_after_state_substitution(request, **kwargs):
            nonlocal substituted
            if not substituted:
                state.rename(held_state)
                state.mkdir()
                substituted = True
            return await original_execute(request, **kwargs)

        try:
            monkeypatch.setattr(
                fixture.executor,
                "execute_trial",
                execute_after_state_substitution,
            )
            result = await run_external_private_memory_ablation(
                fixture.prepared,
                fixture.executor,
                clock=lambda: _NOW,
            )
            assert substituted
            assert fixture.provider.requests == []
            assert result.methodology.run_failure_code is (
                ExternalPrivateMemoryAblationRunFailureCode.EXECUTION_AUTHORITY_CHANGED
            )
            assert result.methodology.failure_ordinal == 1
        finally:
            if held_state.exists():
                state.rmdir()
                held_state.rename(state)
            await fixture.close()

    asyncio.run(exercise())


def test_private_campaign_requires_the_exact_preflighted_executor(tmp_path: Path) -> None:
    async def exercise() -> None:
        fixture = await _campaign_fixture(tmp_path, artifact_name="artifacts")
        replacement = _executor_with_execution_store(
            fixture.executor,
            fixture.executor.executions,
        )
        try:
            assert replacement is not fixture.executor
            assert replacement.execution_authority == fixture.executor.execution_authority
            assert replacement.durable_state_paths() == fixture.executor.durable_state_paths()
            with pytest.raises(
                ValueError,
                match="Executor identity or private state differs",
            ):
                await run_external_private_memory_ablation(
                    fixture.prepared,
                    replacement,
                    clock=lambda: _NOW,
                )
            assert fixture.provider.requests == []
        finally:
            fixture.prepared.close()
            await fixture.close()

    asyncio.run(exercise())


def test_private_state_revision_binds_nested_parent_identity(tmp_path: Path) -> None:
    private_root = tmp_path / "private"
    state = private_root / "state"
    nested = state / "nested"
    nested.mkdir(parents=True)
    destination = external_private_memory_ablation_destination(
        state_storage_id="nested-state",
        destination_id="nested-artifacts",
        approved_private_root=private_root,
        state_directory=state,
        artifact_directory=private_root / "artifacts",
    )
    state_paths = MemoryInterventionExecutorStatePaths(
        snapshot_store=(nested / "snapshots.sqlite",),
        execution_store=(nested / "executions.sqlite",),
        runtime_session_store=(nested / "sessions.sqlite",),
        runtime_budget_ledger=(nested / "budgets.sqlite",),
    )
    for path in (
        *state_paths.snapshot_store,
        *state_paths.execution_store,
        *state_paths.runtime_session_store,
        *state_paths.runtime_budget_ledger,
    ):
        path.touch()
    _, original_revision = private_ablation._canonical_executor_state_files(
        state_paths,
        destination=destination,
    )

    nested.rename(state / "nested-authorized")
    nested.mkdir()
    for path in (
        *state_paths.snapshot_store,
        *state_paths.execution_store,
        *state_paths.runtime_session_store,
        *state_paths.runtime_budget_ledger,
    ):
        path.touch()
    _, replacement_revision = private_ablation._canonical_executor_state_files(
        state_paths,
        destination=destination,
    )

    assert replacement_revision != original_revision


def test_private_campaign_rejects_same_path_state_file_replacement(tmp_path: Path) -> None:
    async def exercise() -> None:
        fixture = await _campaign_fixture(tmp_path, artifact_name="artifacts")
        execution_store = fixture.executor.executions
        assert type(execution_store) is SQLiteMemoryInterventionExecutionStore
        path = execution_store.path
        held = path.with_name(f"{path.name}.authorized")
        original_bytes = path.read_bytes()
        path.rename(held)
        path.write_bytes(original_bytes)
        try:
            with pytest.raises(
                ValueError,
                match="Executor identity or private state differs",
            ):
                await run_external_private_memory_ablation(
                    fixture.prepared,
                    fixture.executor,
                    clock=lambda: _NOW,
                )
            assert fixture.provider.requests == []
        finally:
            await fixture.close()

    asyncio.run(exercise())


def test_private_preflight_rejects_oversized_trial_sequence_before_iteration(
    tmp_path: Path,
) -> None:
    class _OversizedTrials(list[ExternalPrivateMemoryAblationTrial]):
        def __len__(self) -> int:
            return fixture.prepared.authorization.maximum_total_trials + 1

        def __iter__(self) -> Iterator[ExternalPrivateMemoryAblationTrial]:
            raise AssertionError("oversized trial sequence was iterated")

    async def prepare_fixture() -> _CampaignFixture:
        return await _campaign_fixture(tmp_path, artifact_name="artifacts")

    fixture = asyncio.run(prepare_fixture())
    try:
        with pytest.raises(ValueError, match="authorized trial ceiling"):
            prepare_external_private_memory_ablation(
                corpus=fixture.prepared.corpus,
                target=fixture.prepared.target,
                snapshot=fixture.prepared.snapshot,
                experiment=fixture.prepared.experiment,
                trials=_OversizedTrials(),
                executor=fixture.executor,
                authorization=fixture.prepared.authorization,
                schedule_policy=fixture.prepared.schedule_policy,
                destination=fixture.prepared.destination,
                now=_NOW,
            )
        assert fixture.provider.requests == []
    finally:
        asyncio.run(fixture.close())


def test_private_preflight_bounds_a_sequence_that_underreports_its_length(
    tmp_path: Path,
) -> None:
    class _UnderreportedTrials(list[ExternalPrivateMemoryAblationTrial]):
        def __len__(self) -> int:
            return fixture.prepared.authorization.maximum_total_trials

    async def prepare_fixture() -> _CampaignFixture:
        return await _campaign_fixture(tmp_path, artifact_name="underreported-artifacts")

    fixture = asyncio.run(prepare_fixture())
    trials = _UnderreportedTrials(fixture.prepared.scheduled_trials)
    trials.append(fixture.prepared.scheduled_trials[0])
    try:
        with pytest.raises(ValueError, match="authorized trial ceiling"):
            prepare_external_private_memory_ablation(
                corpus=fixture.prepared.corpus,
                target=fixture.prepared.target,
                snapshot=fixture.prepared.snapshot,
                experiment=fixture.prepared.experiment,
                trials=trials,
                executor=fixture.executor,
                authorization=fixture.prepared.authorization,
                schedule_policy=fixture.prepared.schedule_policy,
                destination=fixture.prepared.destination,
                now=_NOW,
            )
        assert fixture.provider.requests == []
    finally:
        fixture.prepared.close()
        asyncio.run(fixture.close())


def test_private_campaign_records_authorized_supplemental_evidence(tmp_path: Path) -> None:
    async def exercise():
        fixture = await _campaign_fixture(tmp_path, artifact_name="evidence-artifacts")
        collector = _SupplementalEvidenceCollector()
        authorization_values = fixture.prepared.authorization.model_dump(mode="json")
        authorization_values.update(
            {
                "evidence_collector_fingerprint": collector.collector_fingerprint,
                "maximum_supplemental_evidence_bytes_per_trial": 4 << 10,
                "cache_evidence_policy": "required",
            }
        )
        authorization = ExternalPrivateMemoryAblationAuthorization.model_validate(
            authorization_values
        )
        prepared = prepare_external_private_memory_ablation(
            corpus=fixture.prepared.corpus,
            target=fixture.prepared.target,
            snapshot=fixture.prepared.snapshot,
            experiment=fixture.prepared.experiment,
            trials=fixture.prepared.scheduled_trials,
            executor=fixture.executor,
            authorization=authorization,
            schedule_policy=fixture.prepared.schedule_policy,
            destination=fixture.prepared.destination,
            now=_NOW,
        )
        try:
            result = await run_external_private_memory_ablation(
                prepared,
                fixture.executor,
                evidence_collector=collector,
                clock=lambda: _NOW,
            )
        finally:
            await fixture.close()
        return fixture, result

    fixture, result = asyncio.run(exercise())

    assert result.methodology.status is ExternalPrivateMemoryAblationRunStatus.COMPLETE
    assert all(
        trial.cache_evidence_state is ExternalPrivateMemoryAblationCacheEvidenceState.AVAILABLE
        for trial in result.methodology.trials
    )
    assert all(trial.provider_retry_count == 1 for trial in result.methodology.trials)
    assert all(trial.memory_overhead_available for trial in result.methodology.trials)
    assert all(row.total_tokens == 96 for row in result.report.rows)
    assert all(row.memory_overhead is not None for row in result.report.rows)
    assert result.methodology.limitations == (
        ExternalPrivateMemoryAblationLimitation.ACCOUNTING_EVIDENCE_UNAVAILABLE,
    )
    assert len(fixture.provider.requests) == len(result.report.rows)


async def _run_accounting_evidence_failure_campaign(
    tmp_path: Path,
    *,
    artifact_name: str,
    wrong_strategy: bool = False,
    shared_attempt_id: str | None = None,
    foreign_session_id: str | None = None,
) -> tuple[_CampaignFixture, ExternalPrivateMemoryAblationResult]:
    fixture = await _campaign_fixture(tmp_path, artifact_name=artifact_name)
    collector = _AccountingEvidenceCollector(
        experiment=fixture.prepared.experiment,
        wrong_strategy=wrong_strategy,
        shared_attempt_id=shared_attempt_id,
        foreign_session_id=foreign_session_id,
    )
    authorization = fixture.prepared.authorization.model_copy(
        update={
            "evidence_collector_fingerprint": collector.collector_fingerprint,
            "maximum_supplemental_evidence_bytes_per_trial": 4 << 10,
        }
    )
    prepared = prepare_external_private_memory_ablation(
        corpus=fixture.prepared.corpus,
        target=fixture.prepared.target,
        snapshot=fixture.prepared.snapshot,
        experiment=fixture.prepared.experiment,
        trials=fixture.prepared.scheduled_trials,
        executor=fixture.executor,
        authorization=authorization,
        schedule_policy=fixture.prepared.schedule_policy,
        destination=fixture.prepared.destination,
        now=_NOW,
    )
    try:
        result = await run_external_private_memory_ablation(
            prepared,
            fixture.executor,
            evidence_collector=collector,
            clock=lambda: _NOW,
        )
    finally:
        prepared.close()
        await fixture.close()
    return fixture, result


def test_private_campaign_rejects_wrong_accounting_authority_before_next_trial(
    tmp_path: Path,
) -> None:
    fixture, result = asyncio.run(
        _run_accounting_evidence_failure_campaign(
            tmp_path,
            artifact_name="wrong-accounting-authority-artifacts",
            wrong_strategy=True,
        )
    )

    assert len(fixture.provider.requests) == 1
    assert result.methodology.status is ExternalPrivateMemoryAblationRunStatus.INCOMPLETE
    assert result.methodology.run_failure_code is (
        ExternalPrivateMemoryAblationRunFailureCode.EVIDENCE_COLLECTION_FAILED
    )
    assert result.methodology.failure_ordinal == 1
    assert result.methodology.trials[0].execution_status is (
        MemoryInterventionExecutionStatus.COMPLETED
    )
    assert not result.methodology.trials[0].accounting_evidence_available
    assert (
        sum(row.availability is not MemoryTrialAvailability.MISSING for row in result.report.rows)
        == 1
    )


def test_private_campaign_rejects_accounting_from_a_foreign_session(
    tmp_path: Path,
) -> None:
    fixture, result = asyncio.run(
        _run_accounting_evidence_failure_campaign(
            tmp_path,
            artifact_name="foreign-accounting-session-artifacts",
            foreign_session_id="foreign-session",
        )
    )

    assert len(fixture.provider.requests) == 1
    assert result.methodology.status is ExternalPrivateMemoryAblationRunStatus.INCOMPLETE
    assert result.methodology.run_failure_code is (
        ExternalPrivateMemoryAblationRunFailureCode.EVIDENCE_COLLECTION_FAILED
    )
    assert result.methodology.failure_ordinal == 1
    assert result.methodology.trials[0].execution_status is (
        MemoryInterventionExecutionStatus.COMPLETED
    )
    assert not result.methodology.trials[0].accounting_evidence_available


def test_private_campaign_rejects_reused_accounting_attempt_before_next_trial(
    tmp_path: Path,
) -> None:
    fixture, result = asyncio.run(
        _run_accounting_evidence_failure_campaign(
            tmp_path,
            artifact_name="reused-accounting-attempt-artifacts",
            shared_attempt_id="shared-accounting-attempt",
        )
    )

    assert len(fixture.provider.requests) == 2
    assert result.methodology.status is ExternalPrivateMemoryAblationRunStatus.INCOMPLETE
    assert result.methodology.run_failure_code is (
        ExternalPrivateMemoryAblationRunFailureCode.EVIDENCE_COLLECTION_FAILED
    )
    assert result.methodology.failure_ordinal == 2
    assert result.methodology.trials[0].accounting_evidence_available
    assert not result.methodology.trials[1].accounting_evidence_available
    assert (
        sum(row.availability is not MemoryTrialAvailability.MISSING for row in result.report.rows)
        == 2
    )


def test_private_preflight_rejects_secret_executor_identity_before_provider_work(
    tmp_path: Path,
) -> None:
    fixture = asyncio.run(
        _campaign_fixture(tmp_path, artifact_name="secret-runtime-identity-artifacts")
    )
    fixture.prepared.target.app._secret_redactor = SecretRedactor(
        fixture.executor.execution_authority.overlay_provider_id
    )
    try:
        with pytest.raises(ValueError, match="Runtime report identities contain"):
            prepare_external_private_memory_ablation(
                corpus=fixture.prepared.corpus,
                target=fixture.prepared.target,
                snapshot=fixture.prepared.snapshot,
                experiment=fixture.prepared.experiment,
                trials=fixture.prepared.scheduled_trials,
                executor=fixture.executor,
                authorization=fixture.prepared.authorization,
                schedule_policy=fixture.prepared.schedule_policy,
                destination=fixture.prepared.destination,
                now=_NOW,
            )
        assert fixture.provider.requests == []
    finally:
        fixture.prepared.close()
        asyncio.run(fixture.close())


def test_private_campaign_returns_incomplete_when_runtime_binding_fails_redaction(
    tmp_path: Path,
) -> None:
    async def exercise():
        fixture = await _campaign_fixture(
            tmp_path,
            artifact_name="runtime-redaction-failure-artifacts",
        )
        fixture.prepared.target.app._secret_redactor = SecretRedactor(
            "cayu.memory-intervention-trial"
        )
        first_schedule_entry = fixture.prepared.schedule[0]
        try:
            result = await run_external_private_memory_ablation(
                fixture.prepared,
                fixture.executor,
                clock=lambda: _NOW,
            )
        finally:
            fixture.prepared.close()
            await fixture.close()
        return fixture, first_schedule_entry, result

    fixture, first_schedule_entry, result = asyncio.run(exercise())

    assert len(fixture.provider.requests) == 1
    assert result.methodology.status is ExternalPrivateMemoryAblationRunStatus.INCOMPLETE
    assert result.methodology.run_failure_code is (
        ExternalPrivateMemoryAblationRunFailureCode.REPORT_EVIDENCE_FAILED
    )
    assert result.methodology.failure_ordinal == 1
    assert result.methodology.limitations == (
        ExternalPrivateMemoryAblationLimitation.ACCOUNTING_EVIDENCE_UNAVAILABLE,
        ExternalPrivateMemoryAblationLimitation.CACHE_EVIDENCE_UNAVAILABLE,
        ExternalPrivateMemoryAblationLimitation.INCOMPLETE_TRIAL_MATRIX,
        ExternalPrivateMemoryAblationLimitation.MEMORY_OVERHEAD_UNAVAILABLE,
        ExternalPrivateMemoryAblationLimitation.REPORT_EVIDENCE_OMITTED,
        ExternalPrivateMemoryAblationLimitation.RUNNER_STOPPED,
    )
    first_row = next(
        row
        for row in result.report.rows
        if (row.case_id, row.repetition, row.variant_id)
        == (
            first_schedule_entry.case_id,
            first_schedule_entry.repetition,
            first_schedule_entry.variant_id,
        )
    )
    assert first_row.availability is MemoryTrialAvailability.UNMATCHED
    assert first_row.intervention_binding is None
    assert first_row.intervention_binding_omitted
    assert first_row.final_binding_fingerprint is not None
    assert first_row.execution_binding_lineage_revision is not None


def test_private_campaign_rechecks_content_free_boundary_before_provider_work(
    tmp_path: Path,
) -> None:
    async def exercise() -> None:
        fixture = await _campaign_fixture(
            tmp_path,
            artifact_name="changed-publication-boundary-artifacts",
        )
        fixture.prepared.target.app._secret_redactor = SecretRedactor(
            fixture.prepared.scheduled_trials[0].request.candidate_id
        )
        try:
            with pytest.raises(
                ValueError,
                match="content-free publication boundary differs",
            ):
                await run_external_private_memory_ablation(
                    fixture.prepared,
                    fixture.executor,
                    clock=lambda: _NOW,
                )
            assert fixture.provider.requests == []
        finally:
            fixture.prepared.close()
            await fixture.close()

    asyncio.run(exercise())


def test_private_campaign_rejects_unsafe_fallback_when_policy_changes_after_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def exercise() -> _CampaignFixture:
        fixture = await _campaign_fixture(
            tmp_path,
            artifact_name="late-publication-boundary-artifacts",
        )
        original_execute = fixture.executor.execute_trial

        async def execute_then_change_redaction_policy(request, **kwargs):
            outcome = await original_execute(request, **kwargs)
            fixture.prepared.target.app._secret_redactor = SecretRedactor(request.candidate_id)
            return outcome

        monkeypatch.setattr(
            fixture.executor,
            "execute_trial",
            execute_then_change_redaction_policy,
        )
        try:
            with pytest.raises(
                ValueError,
                match="publication boundary changed after provider dispatch",
            ):
                await run_external_private_memory_ablation(
                    fixture.prepared,
                    fixture.executor,
                    clock=lambda: _NOW,
                )
        finally:
            fixture.prepared.close()
            await fixture.close()
        return fixture

    fixture = asyncio.run(exercise())

    assert len(fixture.provider.requests) == 1


def test_private_campaign_returns_content_free_result_after_repeated_report_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def exercise():
        fixture = await _campaign_fixture(
            tmp_path,
            artifact_name="repeated-report-failure-artifacts",
        )
        original_report = private_ablation._report_from_outcomes
        report_calls = 0
        clock_calls = 0

        def advancing_clock() -> datetime:
            nonlocal clock_calls
            observed_at = _NOW + timedelta(seconds=clock_calls)
            clock_calls += 1
            return observed_at

        def fail_both_runtime_report_attempts(*args, **kwargs):
            nonlocal report_calls
            report_calls += 1
            if report_calls in {2, 3}:
                raise private_ablation._ExternalPrivateReportRedactionFailed(
                    "synthetic report redaction failure"
                )
            return original_report(*args, **kwargs)

        monkeypatch.setattr(
            private_ablation,
            "_report_from_outcomes",
            fail_both_runtime_report_attempts,
        )
        try:
            result = await run_external_private_memory_ablation(
                fixture.prepared,
                fixture.executor,
                clock=advancing_clock,
            )
        finally:
            fixture.prepared.close()
            await fixture.close()
        return fixture, report_calls, clock_calls, result

    fixture, report_calls, clock_calls, result = asyncio.run(exercise())

    assert len(fixture.provider.requests) == len(result.report.rows)
    assert report_calls == 4
    assert clock_calls == len(result.report.rows) + 2
    assert result.methodology.status is ExternalPrivateMemoryAblationRunStatus.INCOMPLETE
    assert result.methodology.run_failure_code is (
        ExternalPrivateMemoryAblationRunFailureCode.REPORT_EVIDENCE_FAILED
    )
    assert result.methodology.failure_ordinal == len(result.report.rows)
    assert result.methodology.started_at == _NOW
    assert result.methodology.completed_at == _NOW + timedelta(seconds=len(result.report.rows) + 1)
    assert all(row.availability is MemoryTrialAvailability.MISSING for row in result.report.rows)


@pytest.mark.parametrize("fallback_stage", ("report", "methodology"))
def test_private_fallback_preserves_an_existing_run_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fallback_stage: str,
) -> None:
    async def exercise():
        fixture = await _campaign_fixture(
            tmp_path,
            artifact_name="failed-run-report-fallback-artifacts",
        )
        original_execute = fixture.executor.execute_trial
        original_report = private_ablation._report_from_outcomes
        original_methodology = private_ablation._methodology
        execution_calls = 0
        report_calls = 0
        methodology_calls = 0

        async def fail_third_execution(request, **kwargs):
            nonlocal execution_calls
            execution_calls += 1
            if execution_calls == 3:
                raise RuntimeError("synthetic execution failure")
            return await original_execute(request, **kwargs)

        def fail_runtime_report(*args, **kwargs):
            nonlocal report_calls
            report_calls += 1
            if fallback_stage == "report" and report_calls == 2:
                raise private_ablation._ExternalPrivateReportRedactionFailed(
                    "synthetic report redaction failure"
                )
            return original_report(*args, **kwargs)

        def fail_runtime_methodology(*args, **kwargs):
            nonlocal methodology_calls
            methodology_calls += 1
            if fallback_stage == "methodology" and methodology_calls == 2:
                raise private_ablation._ExternalPrivateReportRedactionFailed(
                    "synthetic methodology redaction failure"
                )
            return original_methodology(*args, **kwargs)

        monkeypatch.setattr(fixture.executor, "execute_trial", fail_third_execution)
        monkeypatch.setattr(private_ablation, "_report_from_outcomes", fail_runtime_report)
        monkeypatch.setattr(private_ablation, "_methodology", fail_runtime_methodology)
        try:
            result = await run_external_private_memory_ablation(
                fixture.prepared,
                fixture.executor,
                clock=lambda: _NOW,
            )
        finally:
            fixture.prepared.close()
            await fixture.close()
        return fixture, execution_calls, report_calls, methodology_calls, result

    fixture, execution_calls, report_calls, methodology_calls, result = asyncio.run(exercise())

    assert execution_calls == 3
    assert len(fixture.provider.requests) == 2
    assert report_calls == 3
    assert methodology_calls == (2 if fallback_stage == "report" else 3)
    assert result.methodology.status is ExternalPrivateMemoryAblationRunStatus.INCOMPLETE
    assert result.methodology.run_failure_code is (
        ExternalPrivateMemoryAblationRunFailureCode.EXECUTION_FAILED
    )
    assert result.methodology.failure_ordinal == 3
    assert ExternalPrivateMemoryAblationLimitation.REPORT_EVIDENCE_FAILED in (
        result.methodology.limitations
    )


def test_private_campaign_content_free_methodology_fallback_preserves_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def exercise():
        fixture = await _campaign_fixture(
            tmp_path,
            artifact_name="methodology-failure-artifacts",
        )
        original_methodology = private_ablation._methodology
        methodology_calls = 0
        clock_calls = 0

        def advancing_clock() -> datetime:
            nonlocal clock_calls
            observed_at = _NOW + timedelta(seconds=clock_calls)
            clock_calls += 1
            return observed_at

        def fail_runtime_methodology(*args, **kwargs):
            nonlocal methodology_calls
            methodology_calls += 1
            if methodology_calls == 2:
                raise private_ablation._ExternalPrivateReportRedactionFailed(
                    "synthetic methodology redaction failure"
                )
            return original_methodology(*args, **kwargs)

        monkeypatch.setattr(
            private_ablation,
            "_methodology",
            fail_runtime_methodology,
        )
        try:
            result = await run_external_private_memory_ablation(
                fixture.prepared,
                fixture.executor,
                clock=advancing_clock,
            )
        finally:
            fixture.prepared.close()
            await fixture.close()
        return fixture, methodology_calls, clock_calls, result

    fixture, methodology_calls, clock_calls, result = asyncio.run(exercise())

    assert len(fixture.provider.requests) == len(result.report.rows)
    assert methodology_calls == 3
    assert clock_calls == len(result.report.rows) + 2
    assert result.methodology.status is ExternalPrivateMemoryAblationRunStatus.INCOMPLETE
    assert result.methodology.run_failure_code is (
        ExternalPrivateMemoryAblationRunFailureCode.REPORT_EVIDENCE_FAILED
    )
    assert result.methodology.failure_ordinal == len(result.report.rows)
    assert result.methodology.started_at == _NOW
    assert result.methodology.completed_at == _NOW + timedelta(seconds=len(result.report.rows) + 1)
    assert all(row.availability is MemoryTrialAvailability.MISSING for row in result.report.rows)


def test_private_report_fallback_omits_only_completed_bindings(tmp_path: Path) -> None:
    async def exercise():
        fixture = await _campaign_fixture(
            tmp_path,
            artifact_name="mixed-terminal-redaction-artifacts",
        )
        first_trial, second_trial = fixture.prepared.scheduled_trials[:2]
        try:
            completed = await fixture.executor.execute_trial(first_trial.request)
            second = await fixture.executor.execute_trial(second_trial.request)
            assert second.binding is not None
            assert second.snapshot_result is not None
            failed_snapshot_result = AgentSnapshotResultBinding.create(
                trial=second.binding.trial,
                session_id=second.snapshot_result.session_id,
                terminal_disposition=AgentSnapshotTerminalDisposition.FAILED,
                runtime_evidence_fingerprint=(second.snapshot_result.runtime_evidence_fingerprint),
                eval_result_revision=second.snapshot_result.eval_result_revision,
                memory_evidence_fingerprint=(second.snapshot_result.memory_evidence_fingerprint),
                usage_fingerprint=second.snapshot_result.usage_fingerprint,
                cost_fingerprint=second.snapshot_result.cost_fingerprint,
                safe_frontier_fingerprint=(second.snapshot_result.safe_frontier_fingerprint),
                open_operation_ids=second.snapshot_result.open_operation_ids,
                pending_approval_ids=second.snapshot_result.pending_approval_ids,
                provider_continuation_ids=(second.snapshot_result.provider_continuation_ids),
                recorded_at=second.snapshot_result.recorded_at,
            )
            failed_binding = MemoryInterventionTrialBinding.create(
                spec=second.binding.spec,
                operation=second.binding.operation,
                receipt=second.binding.receipt,
                trial=second.binding.trial,
                result=failed_snapshot_result,
                attribution=second.binding.attribution,
                terminal_evidence_available=(second.binding.terminal_evidence_available),
                expected_receipt_count=second.binding.expected_receipt_count,
                expected_exposure_count=second.binding.expected_exposure_count,
            )
            failed_execution = type(second.execution).model_validate(
                {
                    **second.execution.model_dump(mode="python"),
                    "status": MemoryInterventionExecutionStatus.FAILED,
                    "failure_code": "runtime_failed",
                    "snapshot_result_fingerprint": failed_snapshot_result.fingerprint,
                    "final_binding_fingerprint": failed_binding.fingerprint,
                }
            )
            failed = MemoryInterventionTrialOutcome(
                execution=failed_execution,
                receipt=second.receipt,
                snapshot_result=failed_snapshot_result,
                binding=failed_binding,
            )
            retained, omitted = private_ablation._portable_terminal_report_outcomes(
                fixture.prepared,
                {
                    first_trial.coordinate: completed,
                    second_trial.coordinate: failed,
                },
            )
            report = private_ablation._report_from_outcomes(
                fixture.prepared,
                retained,
                {},
                omitted_binding_coordinates=omitted,
            )
        finally:
            fixture.prepared.close()
            await fixture.close()
        return first_trial, second_trial, retained, omitted, report

    first_trial, second_trial, retained, omitted, report = asyncio.run(exercise())

    assert set(retained) == {first_trial.coordinate, second_trial.coordinate}
    assert omitted == frozenset({first_trial.coordinate})
    rows = {(row.case_id, row.repetition, row.variant_id): row for row in report.rows}
    assert rows[first_trial.coordinate].intervention_binding_omitted
    assert not rows[second_trial.coordinate].intervention_binding_omitted


def test_private_campaign_retains_terminal_outcome_when_evaluator_evidence_exceeds_ceiling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def exercise():
        fixture = await _campaign_fixture(
            tmp_path,
            artifact_name="oversized-evaluator-evidence-artifacts",
        )
        provider = fixture.provider
        assert isinstance(provider, _OrderedCampaignProvider)
        provider.set_outputs(("oversized-evaluator-evidence",))
        original_validate = private_ablation._validate_report_evidence_size

        def reject_evaluator_evidence(outcome, *, maximum_bytes, omit_binding=False):
            if outcome.eval_result is not None:
                raise private_ablation._ExternalPrivateReportEvidenceTooLarge(
                    "synthetic evaluator evidence overflow"
                )
            return original_validate(
                outcome,
                maximum_bytes=maximum_bytes,
                omit_binding=omit_binding,
            )

        monkeypatch.setattr(
            private_ablation,
            "_validate_report_evidence_size",
            reject_evaluator_evidence,
        )
        authorization = fixture.prepared.authorization.model_copy(
            update={
                "maximum_report_evidence_bytes_per_trial": (
                    EXTERNAL_PRIVATE_MEMORY_ABLATION_MIN_REPORT_EVIDENCE_BYTES_PER_TRIAL
                )
            }
        )
        prepared = prepare_external_private_memory_ablation(
            corpus=fixture.prepared.corpus,
            target=fixture.prepared.target,
            snapshot=fixture.prepared.snapshot,
            experiment=fixture.prepared.experiment,
            trials=fixture.prepared.scheduled_trials,
            executor=fixture.executor,
            authorization=authorization,
            schedule_policy=fixture.prepared.schedule_policy,
            destination=fixture.prepared.destination,
            now=_NOW,
        )
        first_schedule_entry = prepared.schedule[0]
        try:
            result = await run_external_private_memory_ablation(
                prepared,
                fixture.executor,
                clock=lambda: _NOW,
            )
        finally:
            prepared.close()
            await fixture.close()
        return fixture, first_schedule_entry, result

    fixture, first_schedule_entry, result = asyncio.run(exercise())

    assert len(fixture.provider.requests) == 1
    assert result.methodology.status is ExternalPrivateMemoryAblationRunStatus.INCOMPLETE
    assert result.methodology.run_failure_code is (
        ExternalPrivateMemoryAblationRunFailureCode.REPORT_EVIDENCE_FAILED
    )
    assert result.methodology.failure_ordinal == 1
    assert result.methodology.trials[0].execution_status is (
        MemoryInterventionExecutionStatus.COMPLETED
    )
    first_row = next(
        row
        for row in result.report.rows
        if (row.case_id, row.repetition, row.variant_id)
        == (
            first_schedule_entry.case_id,
            first_schedule_entry.repetition,
            first_schedule_entry.variant_id,
        )
    )
    assert first_row.availability is MemoryTrialAvailability.UNMATCHED


def test_private_campaign_retains_partial_matrix_and_exactly_recovers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def exercise():
        fixture = await _campaign_fixture(tmp_path, artifact_name="partial-artifacts")
        original_execute = MemoryInterventionExecutor.execute_trial
        durable_trials = 0

        async def stop_after_three(
            executor: MemoryInterventionExecutor,
            request: MemoryInterventionTrialRequest,
            **kwargs,
        ) -> MemoryInterventionTrialOutcome:
            nonlocal durable_trials
            outcome = await original_execute(executor, request, **kwargs)
            durable_trials += 1
            if durable_trials == 3:
                raise RuntimeError("simulated process boundary after durable completion")
            return outcome

        monkeypatch.setattr(MemoryInterventionExecutor, "execute_trial", stop_after_three)
        partial = await run_external_private_memory_ablation(
            fixture.prepared,
            fixture.executor,
            clock=lambda: _NOW,
        )
        monkeypatch.setattr(MemoryInterventionExecutor, "execute_trial", original_execute)
        recovered = await run_external_private_memory_ablation(
            fixture.prepared,
            fixture.executor,
            clock=lambda: _NOW,
        )
        target_checks = 0

        def target_drifts_after_dispatch(
            prepared: PreparedExternalPrivateMemoryAblation,
        ) -> bool:
            nonlocal target_checks
            del prepared
            target_checks += 1
            return target_checks != 2

        with monkeypatch.context() as target_drift:
            target_drift.setattr(
                private_ablation,
                "_target_matches_authorization",
                target_drifts_after_dispatch,
            )
            drifted = await run_external_private_memory_ablation(
                fixture.prepared,
                fixture.executor,
                clock=lambda: _NOW,
            )
        await fixture.close()
        return fixture, partial, recovered, drifted

    fixture, partial, recovered, drifted = asyncio.run(exercise())

    assert partial.methodology.status is ExternalPrivateMemoryAblationRunStatus.INCOMPLETE
    assert partial.methodology.run_failure_code is (
        ExternalPrivateMemoryAblationRunFailureCode.EXECUTION_FAILED
    )
    assert partial.methodology.failure_ordinal == 3
    assert (
        sum(row.availability is MemoryTrialAvailability.MISSING for row in partial.report.rows)
        == len(partial.report.rows) - 2
    )
    assert recovered.methodology.status is ExternalPrivateMemoryAblationRunStatus.COMPLETE
    assert all(
        row.availability is MemoryTrialAvailability.AVAILABLE for row in recovered.report.rows
    )
    assert len(fixture.provider.requests) == len(recovered.report.rows)
    assert drifted.methodology.status is ExternalPrivateMemoryAblationRunStatus.INCOMPLETE
    assert drifted.methodology.run_failure_code is (
        ExternalPrivateMemoryAblationRunFailureCode.TARGET_IDENTITY_CHANGED
    )
    assert drifted.methodology.failure_ordinal == 1
    assert (
        sum(row.availability is MemoryTrialAvailability.MISSING for row in drifted.report.rows)
        == len(drifted.report.rows) - 1
    )

    async def recover_with_fresh_runtime():
        fresh = await _campaign_fixture(tmp_path, artifact_name="recovered-artifacts")
        try:
            result = await run_external_private_memory_ablation(
                fresh.prepared,
                fresh.executor,
                clock=lambda: _NOW,
            )
        finally:
            await fresh.close()
        return fresh, result

    fresh, replayed = asyncio.run(recover_with_fresh_runtime())
    assert fresh.provider.requests == []
    assert memory_experiment_report_to_json(replayed.report) == memory_experiment_report_to_json(
        recovered.report
    )


def test_private_campaign_bounds_invalid_evaluator_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def exercise():
        fixture = await _campaign_fixture(tmp_path, artifact_name="invalid-evaluator-artifacts")
        evaluator_type = type(fixture.executor.evaluator)
        original_evaluate = evaluator_type.evaluate

        async def evaluate_with_wrong_assertion_id(
            evaluator,
            *,
            operation_id,
            case,
            runtime,
        ):
            result = await original_evaluate(
                evaluator,
                operation_id=operation_id,
                case=case,
                runtime=runtime,
            )
            assertions = list(result.assertions)
            assertions[0] = assertions[0].model_copy(update={"name": "wrong-assertion"})
            return result.model_copy(update={"assertions": tuple(assertions)})

        monkeypatch.setattr(evaluator_type, "evaluate", evaluate_with_wrong_assertion_id)
        try:
            result = await run_external_private_memory_ablation(
                fixture.prepared,
                fixture.executor,
                clock=lambda: _NOW,
            )
            return fixture, result
        finally:
            fixture.prepared.close()
            await fixture.close()

    fixture, result = asyncio.run(exercise())

    assert len(fixture.provider.requests) == 1
    assert result.methodology.status is ExternalPrivateMemoryAblationRunStatus.INCOMPLETE
    assert result.methodology.run_failure_code is (
        ExternalPrivateMemoryAblationRunFailureCode.REPORT_EVIDENCE_FAILED
    )
    assert result.methodology.failure_ordinal == 1
    first = result.report.rows[0]
    assert first.execution_status is MemoryInterventionExecutionStatus.COMPLETED
    assert first.availability is MemoryTrialAvailability.UNMATCHED
    assert first.published_result_revision is None
    assert all(
        row.availability is MemoryTrialAvailability.MISSING for row in result.report.rows[1:]
    )


def test_private_preflight_binds_schedule_gates_and_private_destinations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def prepare_fixture():
        return await _campaign_fixture(
            tmp_path / "private",
            artifact_name="counterbalanced-artifacts",
            strategy=ExternalPrivateMemoryAblationScheduleStrategy.COUNTERBALANCED,
        )

    fixture = asyncio.run(prepare_fixture())
    try:
        schedule = fixture.prepared.schedule
        group_size = len(fixture.prepared.experiment.variants)
        first_positions = tuple(
            schedule[index].variant_id for index in range(0, len(schedule), group_size)
        )
        assert len(set(first_positions[:group_size])) == group_size
        assert fixture.provider.requests == []

        wrong_evaluator = _digest("wrong-evaluator")
        evaluator_mismatch_experiment = fixture.prepared.experiment.model_copy(
            update={
                "variants": tuple(
                    variant.model_copy(update={"evaluator_fingerprint": wrong_evaluator})
                    for variant in fixture.prepared.experiment.variants
                )
            }
        )
        evaluator_mismatch_authority = fixture.prepared.authorization.model_copy(
            update={
                "evaluator_fingerprint": wrong_evaluator,
                "experiment_revision": external_private_memory_ablation_experiment_revision(
                    evaluator_mismatch_experiment
                ),
            }
        )
        with pytest.raises(ValueError, match="AgentSnapshot evaluator differs"):
            prepare_external_private_memory_ablation(
                corpus=fixture.prepared.corpus,
                target=fixture.prepared.target,
                snapshot=fixture.prepared.snapshot,
                experiment=evaluator_mismatch_experiment,
                trials=fixture.prepared.scheduled_trials,
                executor=fixture.executor,
                authorization=evaluator_mismatch_authority,
                schedule_policy=fixture.prepared.schedule_policy,
                destination=fixture.prepared.destination,
                now=_NOW,
            )
        assert fixture.provider.requests == []

        provider_mismatch_authority = fixture.prepared.authorization.model_copy(
            update={"provider_configuration_fingerprint": _digest("wrong-provider-config")}
        )
        with pytest.raises(ValueError, match="provider configuration differs"):
            prepare_external_private_memory_ablation(
                corpus=fixture.prepared.corpus,
                target=fixture.prepared.target,
                snapshot=fixture.prepared.snapshot,
                experiment=fixture.prepared.experiment,
                trials=fixture.prepared.scheduled_trials,
                executor=fixture.executor,
                authorization=provider_mismatch_authority,
                schedule_policy=fixture.prepared.schedule_policy,
                destination=fixture.prepared.destination,
                now=_NOW,
            )
        assert fixture.provider.requests == []

        runner = fixture.executor.runtime_runner
        assert type(runner) is CayuMemoryInterventionRuntimeRunner
        factory = runner.factory
        factory.provider_execution_mode = MemoryInterventionProviderExecutionMode.LIVE
        try:
            live_executor = MemoryInterventionExecutor(
                snapshots=fixture.executor.snapshots,
                executions=fixture.executor.executions,
                overlay_provider=fixture.executor.overlay_provider,
                runtime_runner=CayuMemoryInterventionRuntimeRunner(factory),
                evaluator=fixture.executor.evaluator,
                request_keys=fixture.executor._request_keys,
                current_request_key_id=fixture.executor._current_request_key_id,
                clock=fixture.executor._clock,
            )
            with pytest.raises(ValueError, match="provider execution mode differs"):
                prepare_external_private_memory_ablation(
                    corpus=fixture.prepared.corpus,
                    target=fixture.prepared.target,
                    snapshot=fixture.prepared.snapshot,
                    experiment=fixture.prepared.experiment,
                    trials=fixture.prepared.scheduled_trials,
                    executor=live_executor,
                    authorization=fixture.prepared.authorization,
                    schedule_policy=fixture.prepared.schedule_policy,
                    destination=fixture.prepared.destination,
                    now=_NOW,
                )
        finally:
            factory.provider_execution_mode = MemoryInterventionProviderExecutionMode.HERMETIC
        assert fixture.provider.requests == []

        unsafe_case = fixture.prepared.corpus.document.cases[0].model_copy(
            update={
                "assertions": (
                    ToolArgumentsContainAssertionSpec(
                        id="private-tool-arguments",
                        tool_name="private-tool",
                        expected_subset={"value": "case-owned"},
                    ),
                )
            }
        )
        with pytest.raises(ValueError, match="not approved for private reports"):
            private_ablation._validate_private_assertion_projections(
                unsafe_case,
                execution_mode=ExternalPrivateMemoryAblationExecutionMode.HERMETIC,
            )

        rubric = StructuredRubricV1.create(
            id="private-quality",
            criteria=(
                StructuredRubricCriterionV1(
                    id="correctness",
                    name="Correctness",
                    description="Assess correctness.",
                    weight="1",
                ),
            ),
        )
        public_explanation_case = fixture.prepared.corpus.document.cases[0].model_copy(
            update={
                "assertions": (
                    StructuredModelJudgeAssertionSpec(
                        id="private-quality",
                        judge_profile_key="private-judge",
                        judge_profile_revision="sha256:" + _digest("judge-profile"),
                        rubric=rubric,
                    ),
                )
            }
        )
        with pytest.raises(ValueError, match="require a private reference"):
            private_ablation._validate_private_assertion_projections(
                public_explanation_case,
                execution_mode=ExternalPrivateMemoryAblationExecutionMode.HERMETIC,
            )
        private_reference_case = public_explanation_case.model_copy(
            update={
                "assertions": (
                    public_explanation_case.assertions[0].model_copy(
                        update={
                            "reference": PrivateJudgeReferenceV1(
                                key="private-reference",
                                revision="sha256:" + _digest("private-reference"),
                                privacy_policy_key="private-policy",
                                privacy_policy_revision=("sha256:" + _digest("private-policy")),
                            )
                        }
                    ),
                )
            }
        )
        private_ablation._validate_private_assertion_projections(
            private_reference_case,
            execution_mode=ExternalPrivateMemoryAblationExecutionMode.HERMETIC,
        )
        with pytest.raises(ValueError, match="cannot use structured model judges"):
            private_ablation._validate_private_assertion_projections(
                private_reference_case,
                execution_mode=ExternalPrivateMemoryAblationExecutionMode.LIVE,
            )

        with pytest.raises(ValueError, match="authorization is not currently valid"):
            asyncio.run(
                run_external_private_memory_ablation(
                    fixture.prepared,
                    fixture.executor,
                    clock=lambda: datetime(2026, 9, 11, tzinfo=UTC),
                )
            )
        with pytest.raises(ValueError, match="Evidence collector differs"):
            asyncio.run(
                run_external_private_memory_ablation(
                    fixture.prepared,
                    fixture.executor,
                    evidence_collector=_SupplementalEvidenceCollector(),
                    clock=lambda: _NOW,
                )
            )
        assert fixture.provider.requests == []

        runner = fixture.executor.runtime_runner
        assert type(runner) is CayuMemoryInterventionRuntimeRunner
        with monkeypatch.context() as changed_executor:
            changed_executor.setattr(
                runner.factory,
                "provider_configuration_fingerprint",
                _digest("changed-provider-configuration"),
            )
            with pytest.raises(ValueError, match="Executor identity or private state differs"):
                asyncio.run(
                    run_external_private_memory_ablation(
                        fixture.prepared,
                        fixture.executor,
                        clock=lambda: _NOW,
                    )
                )
        assert fixture.provider.requests == []

        substituted_destination = external_private_memory_ablation_destination(
            state_storage_id=fixture.prepared.destination.state_storage_id,
            destination_id=fixture.prepared.destination.destination_id,
            approved_private_root=fixture.prepared.destination.approved_private_root,
            state_directory=fixture.prepared.destination.state_directory,
            artifact_directory="different-artifact-path",
        )
        with pytest.raises(ValueError, match="destination paths differ"):
            prepare_external_private_memory_ablation(
                corpus=fixture.prepared.corpus,
                target=fixture.prepared.target,
                snapshot=fixture.prepared.snapshot,
                experiment=fixture.prepared.experiment,
                trials=fixture.prepared.scheduled_trials,
                executor=fixture.executor,
                authorization=fixture.prepared.authorization,
                schedule_policy=fixture.prepared.schedule_policy,
                destination=substituted_destination,
                now=_NOW,
            )
        assert fixture.provider.requests == []

        changed_target = fixture.prepared.target.model_copy(
            update={
                "request_base": fixture.prepared.target.request_base.model_copy(
                    update={"max_steps": 2}
                )
            }
        )
        with pytest.raises(ValueError, match="execution profile differs"):
            prepare_external_private_memory_ablation(
                corpus=fixture.prepared.corpus,
                target=changed_target,
                snapshot=fixture.prepared.snapshot,
                experiment=fixture.prepared.experiment,
                trials=fixture.prepared.scheduled_trials,
                executor=fixture.executor,
                authorization=fixture.prepared.authorization,
                schedule_policy=fixture.prepared.schedule_policy,
                destination=fixture.prepared.destination,
                now=_NOW,
            )
        assert fixture.provider.requests == []

        changed_experiment = fixture.prepared.experiment.model_copy(
            update={
                "gates": MemoryExperimentGatePolicy(
                    required_metric_roles=fixture.prepared.experiment.gates.required_metric_roles,
                    metric_gates=fixture.prepared.experiment.gates.metric_gates,
                    minimum_comparable_pairs=2,
                )
            }
        )
        with pytest.raises(ValueError, match="gates or selection contract"):
            prepare_external_private_memory_ablation(
                corpus=fixture.prepared.corpus,
                target=fixture.prepared.target,
                snapshot=fixture.prepared.snapshot,
                experiment=changed_experiment,
                trials=fixture.prepared.scheduled_trials,
                executor=fixture.executor,
                authorization=fixture.prepared.authorization,
                schedule_policy=fixture.prepared.schedule_policy,
                destination=fixture.prepared.destination,
                now=_NOW,
            )

        outside_state = tmp_path / "outside-state"
        outside_state.mkdir()
        with pytest.raises(ValueError, match="below the approved private root"):
            external_private_memory_ablation_destination(
                state_storage_id="approved-private-state",
                destination_id="approved-private-report",
                approved_private_root=tmp_path / "private",
                state_directory=outside_state,
                artifact_directory="unauthorized-artifacts",
            )

        with monkeypatch.context() as changed_store:
            changed_store.setattr(
                fixture.executor.executions,
                "path",
                outside_state / "interventions.sqlite",
            )
            with pytest.raises(ValueError, match="below the approved private root"):
                prepare_external_private_memory_ablation(
                    corpus=fixture.prepared.corpus,
                    target=fixture.prepared.target,
                    snapshot=fixture.prepared.snapshot,
                    experiment=fixture.prepared.experiment,
                    trials=fixture.prepared.scheduled_trials,
                    executor=fixture.executor,
                    authorization=fixture.prepared.authorization,
                    schedule_policy=fixture.prepared.schedule_policy,
                    destination=fixture.prepared.destination,
                    now=_NOW,
                )
        assert fixture.provider.requests == []

        with monkeypatch.context() as changed_ledger:
            changed_ledger.setattr(
                fixture.budgets,
                "path",
                outside_state / "budgets.sqlite",
            )
            with pytest.raises(ValueError, match="below the approved private root"):
                prepare_external_private_memory_ablation(
                    corpus=fixture.prepared.corpus,
                    target=fixture.prepared.target,
                    snapshot=fixture.prepared.snapshot,
                    experiment=fixture.prepared.experiment,
                    trials=fixture.prepared.scheduled_trials,
                    executor=fixture.executor,
                    authorization=fixture.prepared.authorization,
                    schedule_policy=fixture.prepared.schedule_policy,
                    destination=fixture.prepared.destination,
                    now=_NOW,
                )
        assert fixture.provider.requests == []

        forged_destination = private_ablation.ExternalPrivateMemoryAblationDestination(
            state_storage_id=fixture.prepared.destination.state_storage_id,
            destination_id=fixture.prepared.destination.destination_id,
            approved_private_root=fixture.prepared.destination.approved_private_root,
            state_directory=outside_state,
            artifact_directory=(
                fixture.prepared.destination.approved_private_root / "forged-artifacts"
            ),
        )
        with pytest.raises(ValueError, match="lacks trusted filesystem authority"):
            prepare_external_private_memory_ablation(
                corpus=fixture.prepared.corpus,
                target=fixture.prepared.target,
                snapshot=fixture.prepared.snapshot,
                experiment=fixture.prepared.experiment,
                trials=fixture.prepared.scheduled_trials,
                executor=fixture.executor,
                authorization=fixture.prepared.authorization,
                schedule_policy=fixture.prepared.schedule_policy,
                destination=forged_destination,
                now=_NOW,
            )

        fixture.prepared.corpus.source_path.write_text(
            fixture.prepared.corpus.source_path.read_text(encoding="utf-8") + " ",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="corpus bytes changed"):
            prepare_external_private_memory_ablation(
                corpus=fixture.prepared.corpus,
                target=fixture.prepared.target,
                snapshot=fixture.prepared.snapshot,
                experiment=fixture.prepared.experiment,
                trials=fixture.prepared.scheduled_trials,
                executor=fixture.executor,
                authorization=fixture.prepared.authorization,
                schedule_policy=fixture.prepared.schedule_policy,
                destination=fixture.prepared.destination,
                now=_NOW,
            )
        assert fixture.provider.requests == []
    finally:
        asyncio.run(fixture.close())


@pytest.mark.parametrize("store_kind", ("snapshot", "execution"))
def test_private_preflight_rejects_process_local_sqlite_state(
    tmp_path: Path,
    store_kind: str,
) -> None:
    async def prepare_fixture():
        return await _campaign_fixture(
            tmp_path / store_kind,
            artifact_name="memory-state-artifacts",
        )

    fixture = asyncio.run(prepare_fixture())
    try:
        if store_kind == "snapshot":
            fixture.executor.snapshots.store = SQLiteAgentSnapshotStore(":memory:")
            expected_group = "snapshot_store"
        else:
            fixture.executor = _executor_with_execution_store(
                fixture.executor,
                SQLiteMemoryInterventionExecutionStore(":memory:"),
            )
            expected_group = "execution_store"

        with pytest.raises(
            ValueError,
            match=f"requires local durable {expected_group} files",
        ):
            prepare_external_private_memory_ablation(
                corpus=fixture.prepared.corpus,
                target=fixture.prepared.target,
                snapshot=fixture.prepared.snapshot,
                experiment=fixture.prepared.experiment,
                trials=fixture.prepared.scheduled_trials,
                executor=fixture.executor,
                authorization=fixture.prepared.authorization,
                schedule_policy=fixture.prepared.schedule_policy,
                destination=fixture.prepared.destination,
                now=_NOW,
            )
        assert fixture.provider.requests == []
    finally:
        asyncio.run(fixture.close())


def test_live_authorization_requires_explicit_budgets_pricing_and_collector() -> None:
    base = {
        "authorization_id": "live",
        "valid_from": datetime(2026, 9, 1, tzinfo=UTC),
        "valid_through": datetime(2026, 9, 10, tzinfo=UTC),
        "corpus_revision": "sha256:" + _digest("corpus"),
        "suite_id": "suite",
        "experiment_revision": "sha256:" + _digest("experiment"),
        "schedule_policy_revision": "sha256:" + _digest("schedule"),
        "target_key": "target",
        "application_release_id": "release",
        "app_manifest_fingerprint": _digest("manifest"),
        "snapshot_fingerprint": _digest("snapshot"),
        "evaluator_fingerprint": _digest("evaluator"),
        "provider_name": "provider",
        "model": "model",
        "provider_configuration_fingerprint": _digest("provider-config"),
        "evidence_policy_revision": "sha256:" + _digest("evidence"),
        "redaction_policy_revision": "sha256:" + _digest("redaction"),
        "retention_policy_revision": "sha256:" + _digest("retention"),
        "state_storage_id": "state",
        "report_destination_id": "report",
        "report_destination_fingerprint": _digest("report-destination"),
        "execution_mode": "live",
        "live_execution_authorization_id": "approved-live-run",
        "allowed_variant_ids": ("as-declared", "omit-items"),
        "allowed_variant_kinds": ("as_declared", "omit_items"),
        "minimum_cases": 1,
        "maximum_cases": 1,
        "minimum_repetitions": 1,
        "maximum_repetitions": 1,
        "maximum_total_trials": 2,
        "maximum_concurrency": 1,
        "maximum_timeout_seconds": 30,
        "maximum_model_steps": 1,
        "maximum_report_evidence_bytes_per_trial": (
            EXTERNAL_PRIVATE_MEMORY_ABLATION_MIN_REPORT_EVIDENCE_BYTES_PER_TRIAL
        ),
        "cache_evidence_policy": "unavailable",
    }

    with pytest.raises(ValueError, match="Live execution requires token and cost ceilings"):
        ExternalPrivateMemoryAblationAuthorization.model_validate(base)

    hermetic_cost_cap: dict[str, object] = dict(base)
    hermetic_cost_cap.update(
        {
            "execution_mode": "hermetic",
            "live_execution_authorization_id": None,
            "maximum_estimated_cost_total": Decimal("1"),
            "cost_currency": "USD",
        }
    )
    with pytest.raises(ValueError, match="requires an exact pricing identity"):
        ExternalPrivateMemoryAblationAuthorization.model_validate(hermetic_cost_cap)

    required_cache: dict[str, object] = dict(base)
    required_cache.update(
        {
            "execution_mode": "hermetic",
            "live_execution_authorization_id": None,
            "cache_evidence_policy": "required",
        }
    )
    with pytest.raises(ValueError, match="needs an authorized evidence collector"):
        ExternalPrivateMemoryAblationAuthorization.model_validate(required_cache)

    unbounded_collector: dict[str, object] = dict(base)
    unbounded_collector.update(
        {
            "execution_mode": "hermetic",
            "live_execution_authorization_id": None,
            "evidence_collector_fingerprint": _digest("collector"),
        }
    )
    with pytest.raises(ValueError, match="per-trial supplemental byte ceiling"):
        ExternalPrivateMemoryAblationAuthorization.model_validate(unbounded_collector)

    huge_cost: dict[str, object] = dict(base)
    huge_cost.update(
        {
            "execution_mode": "hermetic",
            "live_execution_authorization_id": None,
            "maximum_estimated_cost_total": Decimal("1e1000000"),
            "cost_currency": "USD",
            "pricing_profile_fingerprint": "sha256:" + _digest("pricing"),
        }
    )
    with pytest.raises(ValueError, match="64-digit"):
        ExternalPrivateMemoryAblationAuthorization.model_validate(huge_cost)


def test_private_live_token_preflight_detects_child_session_tools() -> None:
    app = CayuApp(enable_logging=False)
    app.register_agent(
        AgentSpec(name="parent", model="model"),
        tools=[
            SubagentTool(
                app,
                agents={"helper": SubagentSpec(agent_name="helper")},
            )
        ],
    )
    app.register_agent(AgentSpec(name="helper", model="model"))

    assert private_ablation._app_agent_has_child_session_tools(app, "parent")
    assert not private_ablation._app_agent_has_child_session_tools(app, "helper")


def test_private_preflight_accepts_only_restart_safe_trial_budgets(tmp_path: Path) -> None:
    async def prepare_fixture():
        return await _campaign_fixture(tmp_path, artifact_name="budget-artifacts")

    fixture = asyncio.run(prepare_fixture())
    try:
        trial = fixture.prepared.scheduled_trials[0]
        pricing = default_price_book()
        authorization = fixture.prepared.authorization.model_copy(
            update={
                "maximum_total_tokens_per_trial": 256,
                "maximum_estimated_cost_total": Decimal("1"),
                "cost_currency": "USD",
                "pricing_profile_fingerprint": pricing_profile_identity(pricing).fingerprint,
            }
        )

        def with_limits(
            *,
            token_scope: Literal["session", "run"] = "session",
            token_ceiling: int | None = 256,
            cost_scope: Literal["causal", "session"] = "causal",
            cost_key: str | None = trial.request.causal_budget_id,
            reserve: bool = True,
            rolling: bool = False,
        ) -> MemoryInterventionTrialRequest:
            cost_limit = BudgetLimit(
                scope=cost_scope,
                key=cost_key,
                max_estimated_cost=Decimal("0.25"),
                pricing=pricing,
                window=(BudgetWindow.rolling(seconds=60) if rolling else BudgetWindow.all_time()),
                reservation=(
                    BudgetReservation(max_input_tokens=128, max_output_tokens=64)
                    if reserve
                    else None
                ),
            )
            return trial.request.model_copy(
                update={
                    "run_request": trial.request.run_request.model_copy(
                        update={
                            "limits": RunLimits(
                                max_total_tokens=token_ceiling,
                                scope=token_scope,
                            ),
                            "budget_limits": (cost_limit,),
                        }
                    )
                }
            )

        bounded = with_limits()
        assert private_ablation._matching_token_ceiling(bounded, authorization) == 256
        assert private_ablation._matching_cost_ceiling(bounded, authorization) == Decimal("0.25")

        assert (
            private_ablation._matching_token_ceiling(with_limits(token_scope="run"), authorization)
            is None
        )
        assert (
            private_ablation._matching_token_ceiling(with_limits(token_ceiling=None), authorization)
            is None
        )
        assert (
            private_ablation._matching_token_ceiling(with_limits(token_ceiling=257), authorization)
            is None
        )
        assert (
            private_ablation._matching_cost_ceiling(
                with_limits(cost_scope="session", cost_key=None), authorization
            )
            is None
        )
        assert (
            private_ablation._matching_cost_ceiling(
                with_limits(cost_key="another-causal-budget"), authorization
            )
            is None
        )
        assert (
            private_ablation._matching_cost_ceiling(with_limits(reserve=False), authorization)
            is None
        )
        assert (
            private_ablation._matching_cost_ceiling(with_limits(rolling=True), authorization)
            is None
        )
    finally:
        asyncio.run(fixture.close())


def test_private_preflight_reserves_complete_report_capacity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = asyncio.run(
        _campaign_fixture(tmp_path, artifact_name="capacity-reservation-artifacts")
    )
    try:
        experiment = fixture.prepared.experiment
        authorization = fixture.prepared.authorization
        suite_cases = tuple(
            case
            for case in fixture.prepared.corpus.document.cases
            if case.suite_id == authorization.suite_id
        )
        reserved = private_ablation._report_capacity_reservation_bytes(
            missing_report=private_ablation.build_memory_experiment_report(experiment),
            cases=suite_cases,
            repetitions=experiment.repetitions,
            variant_count=len(experiment.variants),
            trial_count=len(fixture.prepared.scheduled_trials),
            target_identity=fixture.prepared.target_identity,
            report_evidence_bytes_per_trial=(authorization.maximum_report_evidence_bytes_per_trial),
            supplemental_bytes_per_trial=None,
        )
        monkeypatch.setattr(
            private_ablation,
            "MEMORY_EXPERIMENT_REPORT_MAX_BYTES",
            reserved - 1,
        )

        with pytest.raises(ValueError, match="campaign evidence cannot fit"):
            prepare_external_private_memory_ablation(
                corpus=fixture.prepared.corpus,
                target=fixture.prepared.target,
                snapshot=fixture.prepared.snapshot,
                experiment=experiment,
                trials=fixture.prepared.scheduled_trials,
                executor=fixture.executor,
                authorization=authorization,
                schedule_policy=fixture.prepared.schedule_policy,
                destination=fixture.prepared.destination,
                now=_NOW,
            )
        assert fixture.provider.requests == []
    finally:
        fixture.prepared.close()
        asyncio.run(fixture.close())


def test_private_preflight_rejects_unrepresentable_report_evidence_ceiling(
    tmp_path: Path,
) -> None:
    fixture = asyncio.run(
        _campaign_fixture(tmp_path, artifact_name="unrepresentable-report-evidence-artifacts")
    )
    authorization = fixture.prepared.authorization.model_copy(
        update={
            "maximum_report_evidence_bytes_per_trial": (
                EXTERNAL_PRIVATE_MEMORY_ABLATION_MIN_REPORT_EVIDENCE_BYTES_PER_TRIAL - 1
            )
        }
    )
    try:
        with pytest.raises(ValueError, match="maximum_report_evidence_bytes_per_trial"):
            prepare_external_private_memory_ablation(
                corpus=fixture.prepared.corpus,
                target=fixture.prepared.target,
                snapshot=fixture.prepared.snapshot,
                experiment=fixture.prepared.experiment,
                trials=fixture.prepared.scheduled_trials,
                executor=fixture.executor,
                authorization=authorization,
                schedule_policy=fixture.prepared.schedule_policy,
                destination=fixture.prepared.destination,
                now=_NOW,
            )
        assert fixture.provider.requests == []
    finally:
        fixture.prepared.close()
        asyncio.run(fixture.close())


def test_private_cost_ceiling_aggregation_is_exact() -> None:
    authorized = Decimal("10000000000000000000000000000")
    ceilings = (authorized, Decimal("0.1"))

    assert private_ablation._cost_ceilings_within_authorization((authorized,), authorized)
    assert not private_ablation._cost_ceilings_within_authorization(ceilings, authorized)
    with pytest.raises(ValueError, match="64-digit"):
        private_ablation._cost_ceilings_within_authorization(
            (Decimal("1e-1000000"),),
            Decimal("1"),
        )


def test_live_campaign_binds_distinct_trial_budgets_and_recovers_exactly(tmp_path: Path) -> None:
    async def run() -> None:
        fixture = await _campaign_fixture(
            tmp_path / "private",
            artifact_name="report",
            budgeted_live=True,
        )
        try:
            prepared = fixture.prepared
            trials = prepared.scheduled_trials
            assert len({trial.variant_id for trial in trials}) > 1
            assert len({trial.repetition for trial in trials}) > 1
            assert len({trial.request.causal_budget_id for trial in trials}) == len(trials)
            for trial in trials:
                limits = trial.request.run_request.budget_limits
                assert len(limits) == 1
                assert limits[0].key == trial.request.causal_budget_id
                assert limits[0].max_estimated_cost == Decimal("0.08")
                assert private_ablation._matching_cost_ceiling(
                    trial.request,
                    prepared.authorization,
                ) == Decimal("0.08")
            assert prepared.target.request_base.budget_limits[0].key == (
                EXTERNAL_PRIVATE_MEMORY_ABLATION_TRIAL_BUDGET_KEY
            )
            # Re-preparing resolved requests is exact, and never requires a caller
            # to replace the trusted target template or weaken input comparison.
            again = prepare_external_private_memory_ablation(
                corpus=prepared.corpus,
                target=prepared.target,
                snapshot=prepared.snapshot,
                experiment=prepared.experiment,
                trials=trials,
                executor=fixture.executor,
                authorization=prepared.authorization,
                schedule_policy=prepared.schedule_policy,
                destination=prepared.destination,
                now=_NOW,
            )
            assert again.preflight_revision == prepared.preflight_revision
            assert again._trial_request_revisions == prepared._trial_request_revisions
            collector = _AccountingEvidenceCollector(experiment=prepared.experiment)
            result = await run_external_private_memory_ablation(
                prepared,
                fixture.executor,
                evidence_collector=collector,
                clock=lambda: _NOW,
            )
            assert result.methodology.status is ExternalPrivateMemoryAblationRunStatus.COMPLETE
            assert len(fixture.provider.requests) == len(trials)
            for trial in trials:
                session = await fixture.sessions.load(trial.request.session_id)
                assert session is not None
                assert session.causal_budget_id == trial.request.causal_budget_id
            recovered = await run_external_private_memory_ablation(
                again,
                fixture.executor,
                evidence_collector=collector,
                clock=lambda: _NOW,
            )
            assert recovered.report == result.report
            assert len(fixture.provider.requests) == len(trials)
        finally:
            await fixture.close()
        restarted = await _campaign_fixture(
            tmp_path / "private",
            artifact_name="report",
            budgeted_live=True,
        )
        try:
            recovered_after_restart = await run_external_private_memory_ablation(
                restarted.prepared,
                restarted.executor,
                evidence_collector=_AccountingEvidenceCollector(
                    experiment=restarted.prepared.experiment,
                ),
                clock=lambda: _NOW,
            )
            assert recovered_after_restart.report == result.report
            assert restarted.provider.requests == []
        finally:
            await restarted.close()

    asyncio.run(run())


@pytest.mark.parametrize("mutation", ["prompt", "foreign_key", "ceiling", "reservation", "pricing"])
def test_trial_budget_binding_rejects_unapproved_request_changes(
    tmp_path: Path,
    mutation: str,
) -> None:
    async def run() -> None:
        fixture = await _campaign_fixture(
            tmp_path / "private",
            artifact_name="report",
            budgeted_live=True,
        )
        try:
            prepared = fixture.prepared
            trial = prepared.scheduled_trials[0]
            request = trial.request.run_request
            limit = request.budget_limits[0]
            if mutation == "prompt":
                request = request.model_copy(
                    update={
                        "messages": [*request.messages, Message.text("user", "unapproved")],
                    }
                )
            else:
                updates = {
                    "foreign_key": {"key": prepared.scheduled_trials[1].request.causal_budget_id},
                    "ceiling": {"max_estimated_cost": Decimal("0.09")},
                    "reservation": {
                        "reservation": BudgetReservation(max_input_tokens=1, max_output_tokens=1)
                    },
                    "pricing": {"pricing": default_price_book()},
                }
                request = request.model_copy(
                    update={
                        "budget_limits": (limit.model_copy(update=updates[mutation]),),
                    }
                )
            changed = replace(
                trial, request=trial.request.model_copy(update={"run_request": request})
            )
            with pytest.raises(ValueError, match="Trial request differs from the compiled"):
                prepare_external_private_memory_ablation(
                    corpus=prepared.corpus,
                    target=prepared.target,
                    snapshot=prepared.snapshot,
                    experiment=prepared.experiment,
                    trials=(changed, *prepared.scheduled_trials[1:]),
                    executor=fixture.executor,
                    authorization=prepared.authorization,
                    schedule_policy=prepared.schedule_policy,
                    destination=prepared.destination,
                    now=_NOW,
                )
            assert fixture.provider.requests == []
        finally:
            await fixture.close()

    asyncio.run(run())


@pytest.mark.parametrize("crash_before_resume", [False, True])
@pytest.mark.qualification
def test_native_compaction_then_resume_precedes_provider_work_and_recovers(
    tmp_path: Path, crash_before_resume: bool
):
    import subprocess
    import sys

    from cayu.core.events import EventType

    if crash_before_resume:
        child = subprocess.run(
            [
                sys.executable,
                "-c",
                """
import asyncio, os, sys
from pathlib import Path
import importlib
sys.path.insert(0, str(Path("tests/evals").resolve()))
fixture_module = importlib.import_module(sys.argv[2])
_campaign_fixture = fixture_module._campaign_fixture
_AccountingEvidenceCollector = fixture_module._AccountingEvidenceCollector
_NOW = fixture_module._NOW
from cayu.evals.external_private_memory_ablation import run_external_private_memory_ablation
from cayu.runtime.app import CayuApp
async def crash(self, *args, **kwargs):
    os._exit(75)
    yield
CayuApp._resume_private = crash
async def main():
    fixture = await _campaign_fixture(
        Path(sys.argv[1]), artifact_name="report", budgeted_live=True, prepare_context=True,
    )
    await run_external_private_memory_ablation(
        fixture.prepared, fixture.executor,
        evidence_collector=_AccountingEvidenceCollector(experiment=fixture.prepared.experiment),
        clock=lambda: _NOW,
    )
asyncio.run(main())
""",
                str(tmp_path / "private"),
                __name__,
            ],
            capture_output=True,
            text=True,
            timeout=90,
        )
        assert child.returncode == 75, child.stderr

    async def run():
        fixture = await _campaign_fixture(
            tmp_path / "private",
            artifact_name="report",
            budgeted_live=True,
            prepare_context=True,
        )
        try:
            result = await run_external_private_memory_ablation(
                fixture.prepared,
                fixture.executor,
                evidence_collector=_AccountingEvidenceCollector(
                    experiment=fixture.prepared.experiment
                ),
                clock=lambda: _NOW,
            )
            assert result.methodology.status is ExternalPrivateMemoryAblationRunStatus.COMPLETE, (
                result.methodology.model_dump(mode="json")
            )
            assert len(fixture.provider.requests) == len(fixture.prepared.scheduled_trials)
            for trial in fixture.prepared.scheduled_trials:
                events = await fixture.sessions.load_events(trial.request.session_id)
                kinds = [e.type for e in events]
                assert kinds.count(EventType.INTERACTION_STARTED) == 2
                assert kinds.count(EventType.SESSION_STARTED) == 1
                assert kinds.index(EventType.SESSION_STARTED) < kinds.index(
                    EventType.SESSION_INTERRUPTED
                )
                assert kinds.index(EventType.CONTEXT_COMPACTION_COMPLETED) < kinds.index(
                    EventType.MODEL_STARTED
                )
                from cayu.evals.trajectory import trajectory_from_session

                trajectory = await trajectory_from_session(
                    fixture.prepared.target.app, trial.request.session_id
                )
                assert trajectory.final_output
        finally:
            await fixture.close()
        restarted = await _campaign_fixture(
            tmp_path / "private",
            artifact_name="report",
            budgeted_live=True,
            prepare_context=True,
        )
        try:
            replay = await run_external_private_memory_ablation(
                restarted.prepared,
                restarted.executor,
                evidence_collector=_AccountingEvidenceCollector(
                    experiment=restarted.prepared.experiment
                ),
                clock=lambda: _NOW,
            )
            assert replay.report == result.report
            assert restarted.provider.requests == []
        finally:
            await restarted.close()

    asyncio.run(run())


@pytest.mark.parametrize(
    "tamper",
    [
        "missing_compaction",
        "early_provider",
        "foreign_query",
        "extra_interaction",
        "foreign_compaction",
        "missing_checkpoint",
        "zero_checkpoint",
    ],
)
def test_prepared_trial_recovery_rejects_undeclared_continuation(tamper: str):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from cayu.core.events import Event, EventType
    from cayu.memory_intervention_execution import (
        CayuMemoryInterventionRuntimeRunner,
        MemoryInterventionExecutionConflict,
    )

    async def run():
        first = Event(
            type=EventType.INTERACTION_STARTED, session_id="trial", interaction_id="history"
        )
        compacted = Event(
            type=EventType.CONTEXT_COMPACTION_COMPLETED,
            session_id="trial",
            interaction_id="history",
            payload={
                "request_id": "memory-trial-compact:execution",
                "previous_compacted_transcript_cursor": 0,
                "compacted_transcript_cursor": 1,
                "newly_compacted_message_count": 1,
            },
        )
        second = Event(
            type=EventType.INTERACTION_STARTED, session_id="trial", interaction_id="query"
        )
        events = [first, compacted, second]
        query = Message.text("user", "Declared query")
        messages = [query]
        checkpoint = {"context_compaction": {"compacted_transcript_cursor": 1}}
        if tamper == "foreign_compaction":
            compacted.payload["request_id"] = "foreign"
        elif tamper == "missing_checkpoint":
            checkpoint = None
        elif tamper == "zero_checkpoint":
            checkpoint["context_compaction"]["compacted_transcript_cursor"] = 0
        elif tamper == "missing_compaction":
            events.remove(compacted)
        elif tamper == "early_provider":
            events.insert(
                1, Event(type=EventType.MODEL_STARTED, session_id="trial", interaction_id="history")
            )
        elif tamper == "foreign_query":
            messages = [Message.text("user", "Unrelated continuation")]
        else:
            events.append(
                Event(
                    type=EventType.INTERACTION_STARTED, session_id="trial", interaction_id="third"
                )
            )
        store = SimpleNamespace(
            load_events=AsyncMock(return_value=events),
            load_checkpoint=AsyncMock(return_value=checkpoint),
            load_transcript_snapshot=AsyncMock(
                return_value=SimpleNamespace(
                    records=[
                        SimpleNamespace(interaction_id="query", message=message)
                        for message in messages
                    ]
                )
            ),
        )
        with pytest.raises(MemoryInterventionExecutionConflict):
            await CayuMemoryInterventionRuntimeRunner._continue_compacted_trial(
                app=SimpleNamespace(session_store=store),
                request=SimpleNamespace(run_request=SimpleNamespace(messages=[query])),
                execution=SimpleNamespace(execution_id="execution"),
                session=SimpleNamespace(id="trial"),
            )

    asyncio.run(run())


@pytest.mark.parametrize("already_resumed", [False, True])
def test_native_preparation_rejects_zero_progress_compaction(already_resumed: bool):
    from types import SimpleNamespace

    from cayu.core.events import EventType
    from cayu.memory_intervention_execution import MemoryInterventionExecutionConflict
    from cayu.runtime import InMemorySessionStore, ResumeRequest, RunRequest

    async def run():
        store = InMemorySessionStore()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(ScriptedModelProvider([ModelStreamEvent.completed({})]), default=True)
        policy = reference_campaign._context_policy(
            reference_campaign._recall_policy(reference_campaign.AutomaticRecallMode.OFF)
        )
        policy.base_policy.compactor.max_summary_chars = 200
        app.register_agent(AgentSpec(name="assistant", model="fake-model"), context_policy=policy)
        base = RunRequest(
            agent_name="assistant",
            messages=[
                Message.text("user", "x" * 1000),
                Message.text("assistant", "Acknowledged"),
                Message.text("user", "Another subject"),
                Message.text("assistant", "OK"),
                Message.text("user", "Final query"),
            ],
            limits=RunLimits(scope="session"),
        )
        initial = base.model_copy(update={"messages": base.messages[:-1]})
        events = [
            event async for event in app._run_private(initial, pause_after_initial_transcript=True)
        ]
        session = await store.load(events[0].session_id)
        request = SimpleNamespace(run_request=base, timeout_seconds=120)
        execution = SimpleNamespace(
            execution_id="zero-progress",
            runtime_deadline_at=datetime.now(UTC) + timedelta(seconds=120),
        )
        with pytest.raises(MemoryInterventionExecutionConflict, match="no verified progress"):
            await CayuMemoryInterventionRuntimeRunner._continue_compacted_trial(
                app=app, request=request, execution=execution, session=session
            )
        events = await store.load_events(session.id)
        completed = next(
            event for event in events if event.type is EventType.CONTEXT_COMPACTION_COMPLETED
        )
        assert completed.payload["coverage_mode"] == "no_progress"
        assert completed.payload["newly_compacted_message_count"] == 0
        assert not any(event.type is EventType.MODEL_STARTED for event in events)
        assert (await store.load_checkpoint(session.id))["context_compaction"][
            "compacted_transcript_cursor"
        ] == 0

        if already_resumed:
            # Recreate a continuation admitted by the pre-fix runner so recovery
            # must reject its zero-progress preparation as well.
            async for _event in app._resume_private(
                ResumeRequest(
                    session_id=session.id,
                    messages=[base.messages[-1]],
                    limits=base.limits,
                ),
                store_resolved_session_id=session.id,
            ):
                pass
        session = await store.load(session.id)
        before = await store.load_events(session.id)
        with pytest.raises(MemoryInterventionExecutionConflict, match="no verified progress"):
            await CayuMemoryInterventionRuntimeRunner._continue_compacted_trial(
                app=app, request=request, execution=execution, session=session
            )
        assert await store.load_events(session.id) == before

    asyncio.run(run())


@pytest.mark.parametrize("terminal_without_evaluation", [False, True])
def test_private_incomplete_memory_attribution_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    terminal_without_evaluation: bool,
) -> None:
    async def exercise():
        fixture = await _campaign_fixture(
            tmp_path,
            artifact_name="incomplete-memory-attribution",
            memory_attribution=True,
        )
        original_execute = MemoryInterventionExecutor.execute_trial
        original_publish = private_ablation._publish_variant
        dispatches = []
        attempts = []
        publications = []
        retained_count = 3 if terminal_without_evaluation else 2

        async def bounded_execute(executor, request, **kwargs):
            attempts.append(request.trial_id)
            if len(dispatches) == retained_count:
                raise RuntimeError("bounded stop before the remaining schedule")
            dispatches.append(request.trial_id)
            outcome = await original_execute(executor, request, **kwargs)
            if terminal_without_evaluation and len(dispatches) == 3:
                return outcome.model_copy(update={"eval_result": None})
            return outcome

        def capture_publication(**kwargs):
            published = original_publish(**kwargs)
            publications.append(published)
            return published

        monkeypatch.setattr(MemoryInterventionExecutor, "execute_trial", bounded_execute)
        monkeypatch.setattr(private_ablation, "_publish_variant", capture_publication)
        try:
            result = await run_external_private_memory_ablation(
                fixture.prepared,
                fixture.executor,
                clock=lambda: _NOW,
            )
            schedule = fixture.prepared.schedule
            assert len(attempts) == retained_count + 1
            assert len(set(attempts)) == len(attempts)
            assert len(dispatches) == retained_count
            assert len(set(dispatches)) == retained_count
            assert len(fixture.provider.requests) == retained_count
            # Both constructors validate the serialized report and methodology.
            assert (
                ExternalPrivateMemoryAblationResult(
                    report=result.report,
                    methodology=result.methodology,
                )
                == result
            )
        finally:
            fixture.prepared.close()
            await fixture.close()
        return result, schedule, publications, retained_count

    result, schedule, publications, retained_count = asyncio.run(exercise())
    assert result.methodology.status is ExternalPrivateMemoryAblationRunStatus.INCOMPLETE
    assert (
        result.methodology.run_failure_code
        is ExternalPrivateMemoryAblationRunFailureCode.EXECUTION_FAILED
    )
    assert result.methodology.failure_ordinal == retained_count + 1
    assert [(row.case_id, row.repetition, row.variant_id) for row in result.methodology.trials] == [
        (entry.case_id, entry.repetition, entry.variant_id) for entry in schedule
    ]
    assert all(
        row.execution_status is MemoryInterventionExecutionStatus.COMPLETED
        for row in result.methodology.trials[:retained_count]
    )
    assert all(
        row.availability is MemoryTrialAvailability.MISSING
        for row in result.methodology.trials[retained_count:]
    )
    if terminal_without_evaluation:
        assert result.methodology.trials[2].availability is MemoryTrialAvailability.UNMATCHED
    assert len(publications) == 2
    unavailable = []
    observed = []
    for published in publications:
        assert type(published).model_validate_json(published.model_dump_json()) == published
        for case in published.run.cases:
            for trial in case.trials:
                assertion = next(
                    item for item in trial.assertions if item.assertion_id == "memory-attribution"
                )
                if assertion.outcome == "unavailable":
                    unavailable.append(assertion.detail)
                else:
                    observed.append(assertion.detail)
    assert len(observed) == 2
    assert len(unavailable) == len(schedule) - 2
    assert all(detail.observation_state == "complete" for detail in observed)
    for detail in unavailable:
        assert detail.observation_state == "unavailable"
        assert detail.evidence_revision == EvalMemoryAttributionEvidenceV1.unavailable().revision
        assert tuple(item.value for item in detail.limitations) == ("missing",)
        assert detail.admitted_item_count is None
        assert detail.provider_exposure_count is None

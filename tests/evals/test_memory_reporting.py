from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import warnings
from datetime import UTC, datetime, timedelta
from decimal import Decimal, localcontext

import pytest
from pydantic import ValidationError
from tests.core.test_cost_quality_comparison import (
    _attempt,
    _cost,
    _price_book,
    _quality,
    _side,
)
from tests.core.test_memory_intervention_execution import (
    _automatic_recall_off_spec,
    _executor,
    _indeterminate_exposure_attribution,
    _omit_spec,
    _providers,
    _request,
    _snapshot,
    _spec,
)
from tests.evals.test_corpus_execution import _provider, _target
from tests.evals.test_structured_model_judge import (
    _corpus as _structured_judge_corpus,
)
from tests.evals.test_structured_model_judge import (
    _judge as _structured_judge,
)
from tests.evals.test_structured_model_judge import (
    _judgment as _structured_judgment,
)
from tests.evals.test_structured_model_judge import (
    _target as _structured_judge_target,
)

import cayu.evals.memory_reporting as memory_reporting
from cayu.agent_snapshots import (
    AgentSnapshotCoordinator,
    AgentSnapshotResultBinding,
    execution_profile_snapshot_ref,
)
from cayu.cli import main
from cayu.evals._memory_attribution import (
    eval_memory_attribution_evidence_from_runtime_source,
)
from cayu.evals.corpus import (
    _MODEL_JUDGE_RESULT_METADATA_KEY,
    CorpusUserMessageSpec,
    EvalCaseSpec,
    EvalCorpusDocument,
    EvalSuiteSpec,
    EvaluationEvidencePolicySpec,
    EvaluationSourceIdentityV1,
    JudgeProfileIdentityV1,
    ModelJudgeAssertionSpec,
    RunInputSpec,
    TrialRequestSpec,
    assertion_spec_revision,
    eval_run_contract_for_corpus,
)
from cayu.evals.corpus import (
    _content_revision as _eval_content_revision,
)
from cayu.evals.execution import (
    CorpusExecutionResult,
    CorpusTarget,
    evaluation_target_identity,
    run_corpus_suite,
)
from cayu.evals.execution_profiles import EvalExecutionProfileBindingV1, EvalExecutionProfileV1
from cayu.evals.memory_attribution import (
    EvalMemoryAttributionEvidenceV1,
    EvalMemoryEvidenceCompleteness,
    EvalMemoryEvidenceLimitation,
    eval_memory_attribution_fingerprint,
    standard_eval_memory_attribution_bounds,
)
from cayu.evals.memory_reporting import (
    MemoryCaseComparison,
    MemoryExperimentCase,
    MemoryExperimentGatePolicy,
    MemoryExperimentReport,
    MemoryExperimentReportRequest,
    MemoryExperimentTrialEvidence,
    MemoryExperimentVariant,
    MemoryMetricAvailability,
    MemoryMetricBinding,
    MemoryMetricDirection,
    MemoryMetricGate,
    MemoryMetricRole,
    MemoryOperationalDimension,
    MemoryPairStatus,
    MemoryPreparationOverheadEvidence,
    MemoryPublishedResultEvidence,
    MemoryRankingTerm,
    MemoryTrialAvailability,
    MemoryVariantDispositionStatus,
    build_memory_experiment_report,
    memory_experiment_accounting_source_id,
    memory_experiment_accounting_task_id,
    memory_experiment_report_from_json,
    memory_experiment_report_to_json,
    memory_experiment_request_from_json,
    render_memory_experiment_report_html,
)
from cayu.evals.memory_reporting import (
    _content_revision as _memory_report_content_revision,
)
from cayu.evals.models import (
    EvalAssertionResult,
    EvalCaseResult,
    EvalOutcome,
    EvalRun,
    EvalStatus,
    EvalTrialResult,
    aggregate_eval_score,
)
from cayu.evals.published import (
    PublishedEvalRun,
    PublishedStructuredModelJudgeDetail,
    _publish_eval_run_with_trial_public_data,
)
from cayu.evals.result_contract import (
    EVAL_TRIAL_OUTPUT_MAX_PREVIEW_BYTES,
    EvalTrialDiagnosticCode,
    EvalTrialOutputPreviewV1,
    _EvalTrialPublicData,
)
from cayu.evals.revisions import eval_trial_result_revision
from cayu.evals.store import EvalRunInvocation
from cayu.memory_intervention_execution import (
    InMemoryMemoryInterventionExecutionStore,
    MemoryInterventionExecutionPhase,
    MemoryInterventionExecutionRecord,
    MemoryInterventionExecutionStatus,
)
from cayu.memory_interventions import MemoryInterventionTrialBinding
from cayu.runtime.cost_quality import (
    CostQualityAttemptOperation,
    CostQualityComparisonStatus,
    QualityEvidenceStatus,
)
from cayu.runtime.execution_profiles import (
    ExecutionProfileComponentClass,
    ExecutionProfileComponentIdentity,
    ExecutionProfileIdentityAvailability,
    ExecutionProfileIdentityStrength,
    execution_profile_with_component,
)
from cayu.runtime.usage import (
    SessionUsageSummary,
    build_aggregate_usage_metrics,
    session_usage_summary_payload,
)
from cayu.server.evals_registry import explicit_eval_target_registry

_CASE_ID = "memory-case"
_EXPERIMENT_ID = "memory-experiment"
_ROLES = tuple(sorted(MemoryMetricRole, key=str))
_EVALUATOR_FINGERPRINT = hashlib.sha256(b"evaluator").hexdigest()


class _SecretRepr:
    def __init__(self, value: str) -> None:
        self._value = value

    def __repr__(self) -> str:
        return self._value

    def __str__(self) -> str:
        return self._value


async def _report_fixture(
    *,
    candidate_evaluator_revision: str = "sha256:" + "b" * 64,
    candidate_count: int = 1,
    candidate_unavailable: bool = False,
    candidate_attribution_completeness: EvalMemoryEvidenceCompleteness | None = None,
    candidate_scores: tuple[float, ...] | None = None,
    automatic_recall_off: bool = False,
    total_tokens_base: int = 10,
) -> tuple[MemoryExperimentReportRequest, CorpusTarget, EvalCorpusDocument]:
    target = _target(_provider(trials=1))
    prepared = await explicit_eval_target_registry(target).prepare_execution_profile(target.key)
    snapshot = _snapshot(
        execution_profile=execution_profile_snapshot_ref(prepared.binding.runtime_execution_profile)
    )
    suite = EvalSuiteSpec.create(
        id="memory-suite",
        name="Memory suite",
        trial_request=TrialRequestSpec(trials=2, timeout_seconds=60),
    )
    assertion_specs = tuple(
        ModelJudgeAssertionSpec(
            id=role.value,
            evaluator_key="memory-evaluator",
            rubric=f"Evaluate {role.value}.",
            rubric_version="memory-v1",
            threshold=0.5,
        )
        for role in _ROLES
    )

    def model_judge_record(implementation_revision: str, *, recorded: bool) -> dict[str, object]:
        profile_document = {
            "schema_version": 1,
            "key": "memory-evaluator",
            "label": "Memory evaluator",
            "provider_name": "scripted",
            "model": "memory-judge",
            "implementation_revision": implementation_revision,
            "allowed_evidence": ["final_output"],
            "timeout_seconds": 60,
            "max_input_tokens": 1_000,
            "max_output_tokens": 1_000,
            "max_total_tokens": 2_000,
            "max_estimated_cost": None,
            "cost_currency": None,
            "pricing_profile_fingerprint": None,
            "privacy_policy_key": "public-only",
            "privacy_policy_revision": "sha256:" + "d" * 64,
            "same_model_use": "forbidden",
        }
        profile = JudgeProfileIdentityV1(
            revision=_eval_content_revision(profile_document, "judge profile identity"),
            **profile_document,
        )
        accounting = (
            {
                "usage": {
                    "model_steps": 1,
                    "input_tokens": 2,
                    "output_tokens": 1,
                    "total_tokens": 3,
                },
                "cost": {"availability": "unavailable"},
            }
            if recorded
            else {}
        )
        return {
            "judge_profile": profile.model_dump(mode="json"),
            "candidate_route_relation": "independent_model",
            **accounting,
        }

    case = EvalCaseSpec.create(
        id=_CASE_ID,
        suite_id=suite.id,
        name="Memory case",
        source=EvaluationSourceIdentityV1(
            application_release_id=target.application_release_id,
            app_manifest_schema_version="7",
            app_manifest_fingerprint=target.app.describe().fingerprint,
            evidence_revision="sha256:" + "a" * 64,
        ),
        input=RunInputSpec(messages=(CorpusUserMessageSpec(text="Evaluate memory."),)),
        assertions=assertion_specs,
    )
    corpus = EvalCorpusDocument.create(
        target_key=target.key,
        evidence_policy=EvaluationEvidencePolicySpec.standard(),
        suites=(suite,),
        cases=(case,),
    )
    baseline_spec = _spec(snapshot, spec_id="baseline-spec")
    candidate_spec = (
        _automatic_recall_off_spec(snapshot) if automatic_recall_off else _omit_spec(snapshot)
    )
    candidate_profile = prepared.snapshot
    candidate_profile_binding = prepared.binding
    if automatic_recall_off:
        effective_runtime_profile = execution_profile_with_component(
            prepared.binding.runtime_execution_profile,
            ExecutionProfileComponentIdentity(
                component_class=ExecutionProfileComponentClass.AUTOMATIC_RECALL,
                strength=ExecutionProfileIdentityStrength.STRUCTURAL,
                availability=ExecutionProfileIdentityAvailability.AVAILABLE,
                fingerprint=hashlib.sha256(b"automatic-recall-off").hexdigest(),
            ),
        )
        candidate_identity = prepared.snapshot.candidate.model_copy(
            update={"runtime_execution_profile_fingerprint": effective_runtime_profile.fingerprint}
        )
        candidate_profile = EvalExecutionProfileV1.create(
            profile_id=prepared.snapshot.profile_id,
            label=prepared.snapshot.label,
            source=prepared.snapshot.source,
            target_key=prepared.snapshot.target_key,
            application_release_id=prepared.snapshot.application_release_id,
            app_manifest_fingerprint=prepared.snapshot.app_manifest_fingerprint,
            candidate=candidate_identity,
            target_material=prepared.snapshot.target_material,
            fixture_strategy=prepared.snapshot.fixture_strategy,
            reset_strategy=prepared.snapshot.reset_strategy,
            effect_posture=prepared.snapshot.effect_posture,
            isolation_revision=prepared.snapshot.isolation_revision,
            evidence_policy=prepared.snapshot.evidence_policy,
            ceilings=prepared.snapshot.ceilings,
        )
        candidate_profile_binding = EvalExecutionProfileBindingV1(
            profile_revision=candidate_profile.revision,
            runtime_execution_profile=effective_runtime_profile,
        )
    executor, _, _, _ = await _executor(
        snapshot,
        snapshot_store=AgentSnapshotCoordinator(_providers(snapshot, {})).store,
        execution_store=InMemoryMemoryInterventionExecutionStore(),
        materializations={},
        views={},
        runtime_results={},
        eval_results={},
    )
    variants = (
        MemoryExperimentVariant(
            variant_id="baseline",
            candidate_id="baseline-candidate",
            spec=baseline_spec,
            execution_profile=prepared.snapshot,
            execution_profile_binding=prepared.binding,
            evaluator_fingerprint=_EVALUATOR_FINGERPRINT,
        ),
        *(
            MemoryExperimentVariant(
                variant_id="candidate" if index == 0 else f"candidate-{index + 1}",
                candidate_id=f"candidate-{index + 1}",
                spec=candidate_spec,
                execution_profile=candidate_profile,
                execution_profile_binding=candidate_profile_binding,
                evaluator_fingerprint=_EVALUATOR_FINGERPRINT,
            )
            for index in range(candidate_count)
        ),
    )
    executions: dict[
        tuple[str, int],
        tuple[MemoryInterventionExecutionRecord, MemoryInterventionTrialBinding],
    ] = {}
    for variant in variants:
        spec = baseline_spec if variant.variant_id == "baseline" else candidate_spec
        for repetition in (1, 2):
            trial_request = _request(
                spec,
                candidate_id=variant.candidate_id,
                trial_id=f"{variant.variant_id}-trial-{repetition}",
            )
            request_document = trial_request.model_dump(
                mode="python", round_trip=True, warnings="none"
            )
            request_document["case"] = {
                "case_id": _CASE_ID,
                "case_revision": case.revision,
            }
            outcome = await executor.execute_trial(
                type(trial_request).model_validate(request_document)
            )
            assert outcome.binding is not None
            execution = outcome.execution
            if automatic_recall_off and variant.variant_id != "baseline":
                execution = MemoryInterventionExecutionRecord.model_validate(
                    {
                        **execution.model_dump(
                            mode="python",
                            round_trip=True,
                            warnings="none",
                        ),
                        "runtime_execution_profile_fingerprint": (
                            candidate_profile_binding.runtime_execution_profile.fingerprint
                        ),
                    }
                )
            executions[(variant.variant_id, repetition)] = (
                execution,
                outcome.binding,
            )

    metric_bindings = tuple(
        MemoryMetricBinding(
            role=role,
            assertion_id=role.value,
            assertion_revision=assertion_spec_revision(assertion),
        )
        for role, assertion in zip(_ROLES, assertion_specs, strict=True)
    )
    published_results: list[MemoryPublishedResultEvidence] = []
    trial_evidence: list[MemoryExperimentTrialEvidence] = []
    for variant in variants:
        trials: list[EvalTrialResult] = []
        revised_executions: dict[int, MemoryInterventionExecutionRecord] = {}
        revised_bindings: dict[int, MemoryInterventionTrialBinding] = {}
        unavailable = candidate_unavailable and variant.variant_id == "candidate"
        if variant.variant_id == "baseline":
            score = 0.7
        else:
            candidate_index = tuple(
                item.variant_id for item in variants if item.variant_id != "baseline"
            ).index(variant.variant_id)
            score = (
                0.9 - candidate_index * 0.1
                if candidate_scores is None
                else candidate_scores[candidate_index]
            )
        for repetition in (1, 2):
            execution, binding = executions[(variant.variant_id, repetition)]
            attribution = eval_memory_attribution_evidence_from_runtime_source(
                terminal_status="completed",
                attribution=binding.attribution,
                terminal_evidence_available=binding.terminal_evidence_available,
                terminal_evidence_limitation=None,
                expected_receipt_count=binding.expected_receipt_count,
                expected_exposure_count=binding.expected_exposure_count,
                effective_bounds=standard_eval_memory_attribution_bounds(),
                source_alias=None,
            )
            if variant.variant_id != "baseline" and candidate_attribution_completeness is not None:
                limitation = (
                    EvalMemoryEvidenceLimitation.SOURCE_BYTES_LIMIT
                    if candidate_attribution_completeness
                    is EvalMemoryEvidenceCompleteness.TRUNCATED
                    else EvalMemoryEvidenceLimitation.EVIDENCE_READ_FAILED
                )
                attribution = EvalMemoryAttributionEvidenceV1.create(
                    effective_bounds=attribution.effective_bounds,
                    effective_source_limit=attribution.effective_source_limit,
                    effective_max_bytes=attribution.effective_max_bytes,
                    completeness=candidate_attribution_completeness,
                    limitations=(limitation,),
                    total_source_count=attribution.total_source_count,
                    sources=attribution.sources,
                    omitted_source_count_at_least=attribution.omitted_source_count_at_least,
                )
            started_at = datetime(2026, 8, 27, tzinfo=UTC) + timedelta(seconds=repetition)
            trial = EvalTrialResult(
                trial_number=repetition,
                status=EvalStatus.UNAVAILABLE if unavailable else EvalStatus.PASSED,
                session_id=f"{variant.variant_id}-session-{repetition}",
                score=(
                    None if unavailable else aggregate_eval_score(score for _ in assertion_specs)
                ),
                final_output="Memory answer",
                assertions=tuple(
                    EvalAssertionResult(
                        name=item.id,
                        assertion_revision=assertion_spec_revision(item),
                        outcome=(EvalOutcome.UNAVAILABLE if unavailable else EvalOutcome.PASSED),
                        score=None if unavailable else score,
                        threshold=0.5,
                        metadata={
                            _MODEL_JUDGE_RESULT_METADATA_KEY: model_judge_record(
                                (
                                    candidate_evaluator_revision
                                    if variant.variant_id == "candidate"
                                    else "sha256:" + "b" * 64
                                ),
                                recorded=not unavailable,
                            )
                        },
                    )
                    for item in assertion_specs
                ),
                unavailable_reason=("Evaluator evidence was unavailable." if unavailable else None),
                evidence_complete=True,
                usage_summary=session_usage_summary_payload(
                    SessionUsageSummary(
                        session_id=f"{variant.variant_id}-session-{repetition}",
                        model_steps=1,
                        usage=build_aggregate_usage_metrics(
                            total_tokens=total_tokens_base + repetition
                        ),
                    )
                ),
                memory_attribution=attribution,
                started_at=started_at,
                completed_at=started_at + timedelta(milliseconds=100 + repetition),
                duration_ms=100 + repetition,
            )
            trials.append(trial)
            revised_binding = _binding_with_eval_result_revision(
                binding,
                eval_trial_result_revision(trial),
            )
            document = execution.model_dump(mode="python", round_trip=True, warnings="none")
            document["eval_result_revision"] = eval_trial_result_revision(trial)
            document["snapshot_result_fingerprint"] = revised_binding.result.fingerprint
            document["final_binding_fingerprint"] = revised_binding.fingerprint
            revised_executions[repetition] = MemoryInterventionExecutionRecord.model_validate(
                document
            )
            revised_bindings[repetition] = revised_binding
        case_result = EvalCaseResult.from_trials(
            case_id=_CASE_ID,
            trials=tuple(trials),
            started_at=trials[0].started_at,
            completed_at=trials[-1].completed_at,
        )
        run = EvalRun(
            run_id=f"{variant.variant_id}-run",
            suite_id=suite.id,
            status=case_result.status,
            score=case_result.score,
            cases=(case_result,),
            started_at=trials[0].started_at,
            completed_at=trials[-1].completed_at,
            duration_ms=case_result.duration_ms,
            run_contract=eval_run_contract_for_corpus(corpus, suite.id),
        )
        result = CorpusExecutionResult.create(
            target=evaluation_target_identity(target),
            run=_publish_eval_run_with_trial_public_data(
                corpus,
                run,
                trial_public_data_by_case={
                    _CASE_ID: tuple(
                        _EvalTrialPublicData(
                            diagnostic_code=(
                                EvalTrialDiagnosticCode.ASSERTION_EVIDENCE_UNAVAILABLE
                                if unavailable
                                else EvalTrialDiagnosticCode.PASSED
                            ),
                            output=(
                                EvalTrialOutputPreviewV1.unavailable()
                                if unavailable
                                else EvalTrialOutputPreviewV1.from_retained_evidence(
                                    "Memory answer",
                                    "complete",
                                    max_preview_bytes=EVAL_TRIAL_OUTPUT_MAX_PREVIEW_BYTES,
                                )
                            ),
                        )
                        for _ in trials
                    )
                },
            ),
        )
        published_results.append(MemoryPublishedResultEvidence(run_id=run.run_id, result=result))
        for repetition in (1, 2):
            binding = revised_bindings[repetition]
            attempts = (
                _attempt(
                    attempt_id=f"{variant.variant_id}-{repetition}-attempt-1",
                    session_id=f"{variant.variant_id}-{repetition}-session-1",
                    input_tokens=10,
                    operation=CostQualityAttemptOperation.EVALUATION,
                    attempt_ordinal=1,
                ),
                _attempt(
                    attempt_id=f"{variant.variant_id}-{repetition}-attempt-2",
                    session_id=f"{variant.variant_id}-{repetition}-session-2",
                    input_tokens=5,
                    operation=CostQualityAttemptOperation.EVALUATION,
                    attempt_ordinal=2,
                ),
            )
            trial_evidence.append(
                MemoryExperimentTrialEvidence(
                    case_id=_CASE_ID,
                    case_revision=case.revision,
                    repetition=repetition,
                    variant_id=variant.variant_id,
                    execution=revised_executions[repetition],
                    intervention_binding=binding,
                    published_result_revision=result.revision,
                    accounting_side=_side(
                        strategy_id=variant.variant_id,
                        attempts=attempts,
                        workload_id=_EXPERIMENT_ID,
                        task_id=memory_experiment_accounting_task_id(_EXPERIMENT_ID),
                        source_id=memory_experiment_accounting_source_id(
                            case.revision,
                            repetition,
                        ),
                    ),
                    memory_overhead=MemoryPreparationOverheadEvidence.create(
                        preparation_duration_ms=20 + repetition,
                        context_tokens=30 + repetition,
                        context_bytes=100 + repetition,
                    ),
                )
            )
    request = MemoryExperimentReportRequest(
        experiment_id=_EXPERIMENT_ID,
        cases=(MemoryExperimentCase(case_id=_CASE_ID, case_revision=case.revision),),
        repetitions=2,
        baseline_variant_id="baseline",
        variants=variants,
        metric_bindings=metric_bindings,
        ranking=(
            MemoryRankingTerm(
                role=MemoryMetricRole.TASK_QUALITY,
                direction=MemoryMetricDirection.HIGHER_IS_BETTER,
            ),
        ),
        gates=MemoryExperimentGatePolicy(
            required_metric_roles=(
                MemoryMetricRole.PRIVACY,
                MemoryMetricRole.SAFETY,
            ),
            metric_gates=(
                MemoryMetricGate(role=MemoryMetricRole.PRIVACY, minimum=0.8),
                MemoryMetricGate(role=MemoryMetricRole.SAFETY, minimum=0.8),
            ),
            minimum_comparable_pairs=2,
            require_priced_cost=True,
        ),
        published_results=tuple(sorted(published_results, key=lambda item: item.result.revision)),
        trials=tuple(
            sorted(
                trial_evidence,
                key=lambda item: (item.case_id, item.repetition, item.variant_id),
            )
        ),
    )
    return request, target, corpus


async def _report_request(
    *,
    candidate_evaluator_revision: str = "sha256:" + "b" * 64,
    candidate_count: int = 1,
    candidate_unavailable: bool = False,
    candidate_attribution_completeness: EvalMemoryEvidenceCompleteness | None = None,
    candidate_scores: tuple[float, ...] | None = None,
    automatic_recall_off: bool = False,
    total_tokens_base: int = 10,
) -> MemoryExperimentReportRequest:
    request, _, _ = await _report_fixture(
        candidate_evaluator_revision=candidate_evaluator_revision,
        candidate_count=candidate_count,
        candidate_unavailable=candidate_unavailable,
        candidate_attribution_completeness=candidate_attribution_completeness,
        candidate_scores=candidate_scores,
        automatic_recall_off=automatic_recall_off,
        total_tokens_base=total_tokens_base,
    )
    return request


def _published_result_with_run_updates(
    result: CorpusExecutionResult,
    **updates: object,
) -> CorpusExecutionResult:
    document = result.run.model_dump(mode="json", round_trip=True, warnings="none")
    document.update(updates)
    document["revision"] = _eval_content_revision(document, "published eval run")
    return CorpusExecutionResult.create(
        target=result.target,
        run=PublishedEvalRun.model_validate_json(json.dumps(document)),
    )


def _published_result_with_trial_attribution(
    result: CorpusExecutionResult,
    *,
    repetition: int,
    attribution: EvalMemoryAttributionEvidenceV1,
) -> CorpusExecutionResult:
    selected_case = result.run.cases[0]
    selected_trial = selected_case.trials[repetition - 1]
    trials = tuple(
        selected_trial.model_copy(update={"memory_attribution": attribution})
        if item.trial_number == repetition
        else item
        for item in selected_case.trials
    )
    cases = (
        selected_case.model_copy(update={"trials": trials}),
        *result.run.cases[1:],
    )
    return _published_result_with_run_updates(
        result,
        cases=[item.model_dump(mode="json") for item in cases],
    )


def _request_with_candidate_result(
    request: MemoryExperimentReportRequest,
    result: CorpusExecutionResult,
) -> MemoryExperimentReportRequest:
    candidate_trials = tuple(item for item in request.trials if item.variant_id == "candidate")
    source_revision = candidate_trials[0].published_result_revision
    assert source_revision is not None
    assert all(item.published_result_revision == source_revision for item in candidate_trials)
    published_results = tuple(
        MemoryPublishedResultEvidence(run_id=item.run_id, result=result)
        if item.result.revision == source_revision
        else item
        for item in request.published_results
    )
    trials = tuple(
        item.model_copy(update={"published_result_revision": result.revision})
        if item.variant_id == "candidate"
        else item
        for item in request.trials
    )
    return MemoryExperimentReportRequest.model_validate(
        request.model_copy(
            update={
                "published_results": tuple(
                    sorted(published_results, key=lambda item: item.result.revision)
                ),
                "trials": trials,
            }
        ).model_dump(mode="python")
    )


def test_report_retains_exact_repeated_matrix_and_canonical_accounting() -> None:
    request = asyncio.run(_report_request())

    report = build_memory_experiment_report(request)

    assert report.repetitions == 2
    assert report.selected_variant_id == "candidate"
    assert len(report.rows) == 4
    assert all(row.availability is MemoryTrialAvailability.AVAILABLE for row in report.rows)
    assert all(row.source_trial_revision is not None for row in report.rows)
    assert all(row.attribution_evidence_revision is not None for row in report.rows)
    assert all(row.memory_overhead is not None for row in report.rows)
    assert len(report.cost_quality) == 1
    assert report.cost_quality[0].candidate_variant_id == "candidate"
    assert report.cost_quality[0].report is not None
    assert report.cost_quality[0].report.aggregate.status is CostQualityComparisonStatus.VERIFIED
    assert report.cost_quality[0].report.aggregate.pair_count == 2
    assert all(
        side.whole_harness.retry_attempt_count == 1
        for pair in report.cost_quality[0].report.pairs
        for side in (pair.baseline, pair.candidate)
        if side is not None
    )
    assert all(
        pair.status is MemoryPairStatus.COMPARABLE
        for comparison in report.comparisons
        for pair in comparison.pairs
    )
    assert all(
        delta.candidate == 0.9 and delta.baseline == 0.7 and delta.delta == pytest.approx(0.2)
        for comparison in report.comparisons
        for pair in comparison.pairs
        for delta in pair.metric_deltas
    )
    assert all(
        {item.dimension for item in pair.operational_deltas} == set(MemoryOperationalDimension)
        for comparison in report.comparisons
        for pair in comparison.pairs
    )
    assert all(
        distribution.available_count == 2
        for comparison in report.comparisons
        for distribution in comparison.operational_distributions
    )
    assert all(
        distribution.available_count == 2
        for summary in report.operational_summary
        for distribution in summary.distributions
    )
    candidate = next(item for item in report.dispositions if item.variant_id == "candidate")
    assert candidate.status is MemoryVariantDispositionStatus.SELECTED


def test_trial_binding_matches_only_the_published_root_attribution() -> None:
    request = asyncio.run(_report_request())
    selected = next(
        item for item in request.trials if item.variant_id == "candidate" and item.repetition == 1
    )
    assert selected.intervention_binding is not None
    foreign_attribution = _indeterminate_exposure_attribution()
    evidence = next(
        item
        for item in request.published_results
        if item.result.revision == selected.published_result_revision
    )
    published_trial = evidence.result.run.cases[0].trials[0]
    original_attribution = published_trial.memory_attribution
    original_root = original_attribution.sources[0]
    assert original_root.source.tree_path == ()
    foreign_root = original_root.model_copy(
        update={
            "attribution": foreign_attribution,
            "attribution_fingerprint": eval_memory_attribution_fingerprint(foreign_attribution),
        }
    )
    matching_descendant = original_root.model_copy(
        update={
            "source": original_root.source.model_copy(
                update={"role": "descendant", "tree_path": (0,)}
            )
        }
    )
    mismatched = EvalMemoryAttributionEvidenceV1.create(
        effective_bounds=original_attribution.effective_bounds,
        effective_source_limit=original_attribution.effective_source_limit,
        effective_max_bytes=original_attribution.effective_max_bytes,
        completeness=EvalMemoryEvidenceCompleteness.COMPLETE,
        limitations=(),
        total_source_count=2,
        sources=(foreign_root, matching_descendant),
    )
    changed_result = _published_result_with_trial_attribution(
        evidence.result,
        repetition=1,
        attribution=mismatched,
    )

    with pytest.raises(ValidationError, match="memory attribution conflicts"):
        _request_with_candidate_result(request, changed_result)


def test_trial_binding_ignores_a_foreign_descendant_when_the_root_matches() -> None:
    request = asyncio.run(_report_request())
    selected = next(
        item for item in request.trials if item.variant_id == "candidate" and item.repetition == 1
    )
    assert selected.intervention_binding is not None
    foreign_attribution = _indeterminate_exposure_attribution()
    evidence = next(
        item
        for item in request.published_results
        if item.result.revision == selected.published_result_revision
    )
    published_trial = evidence.result.run.cases[0].trials[0]
    original_attribution = published_trial.memory_attribution
    original_root = original_attribution.sources[0]
    foreign_descendant = original_root.model_copy(
        update={
            "source": original_root.source.model_copy(
                update={"role": "descendant", "tree_path": (0,)}
            ),
            "attribution": foreign_attribution,
            "attribution_fingerprint": eval_memory_attribution_fingerprint(foreign_attribution),
        }
    )
    retained = EvalMemoryAttributionEvidenceV1.create(
        effective_bounds=original_attribution.effective_bounds,
        effective_source_limit=original_attribution.effective_source_limit,
        effective_max_bytes=original_attribution.effective_max_bytes,
        completeness=EvalMemoryEvidenceCompleteness.COMPLETE,
        limitations=(),
        total_source_count=2,
        sources=(original_root, foreign_descendant),
    )
    changed_result = _published_result_with_trial_attribution(
        evidence.result,
        repetition=1,
        attribution=retained,
    )

    rebuilt = _request_with_candidate_result(request, changed_result)

    assert build_memory_experiment_report(rebuilt).rows[1].availability is (
        MemoryTrialAvailability.AVAILABLE
    )


def test_automatic_recall_off_is_reported_under_exact_recall_only_authority() -> None:
    request = asyncio.run(_report_request(automatic_recall_off=True))

    report = build_memory_experiment_report(request)

    candidate = report.variants[1]
    baseline_runtime = report.variants[0].execution_profile_binding.runtime_execution_profile
    candidate_runtime = candidate.execution_profile_binding.runtime_execution_profile
    changed = tuple(
        component.component_class
        for component in candidate_runtime.components
        if component != baseline_runtime.component(component.component_class)
    )
    assert candidate.spec.execution_profile_fingerprint == baseline_runtime.fingerprint
    assert changed == (ExecutionProfileComponentClass.AUTOMATIC_RECALL,)
    assert all(
        pair.status is MemoryPairStatus.COMPARABLE
        for comparison in report.comparisons
        for pair in comparison.pairs
    )
    assert report.selected_variant_id == "candidate"


def test_automatic_recall_off_manifest_change_cannot_retain_comparable_pairs() -> None:
    report = build_memory_experiment_report(asyncio.run(_report_request(automatic_recall_off=True)))
    candidate = report.variants[1]
    changed_manifest_fingerprint = hashlib.sha256(b"different-app-manifest").hexdigest()
    profile = EvalExecutionProfileV1.create(
        profile_id=candidate.execution_profile.profile_id,
        label=candidate.execution_profile.label,
        source=candidate.execution_profile.source,
        target_key=candidate.execution_profile.target_key,
        application_release_id=candidate.execution_profile.application_release_id,
        app_manifest_fingerprint=changed_manifest_fingerprint,
        candidate=candidate.execution_profile.candidate,
        target_material=candidate.execution_profile.target_material,
        fixture_strategy=candidate.execution_profile.fixture_strategy,
        reset_strategy=candidate.execution_profile.reset_strategy,
        effect_posture=candidate.execution_profile.effect_posture,
        isolation_revision=candidate.execution_profile.isolation_revision,
        evidence_policy=candidate.execution_profile.evidence_policy,
        ceilings=candidate.execution_profile.ceilings,
    )
    replacement = candidate.model_copy(
        update={
            "execution_profile": profile,
            "execution_profile_binding": EvalExecutionProfileBindingV1(
                profile_revision=profile.revision,
                runtime_execution_profile=(
                    candidate.execution_profile_binding.runtime_execution_profile
                ),
            ),
        }
    )
    variants = tuple(
        replacement if item.variant_id == candidate.variant_id else item for item in report.variants
    )
    rows = tuple(
        item.model_copy(update={"execution_profile_revision": profile.revision})
        if item.variant_id == candidate.variant_id
        else item
        for item in report.rows
    )

    with pytest.raises(
        ValidationError,
        match="pair classification conflicts with its exact evidence",
    ):
        MemoryExperimentReport.model_validate(
            report.model_copy(update={"variants": variants, "rows": rows}).model_dump(mode="python")
        )


def test_non_recall_runtime_profile_change_is_rejected_before_report_build() -> None:
    request = asyncio.run(_report_request(automatic_recall_off=True))
    candidate = request.variants[1]
    runtime = candidate.execution_profile_binding.runtime_execution_profile
    altered = execution_profile_with_component(
        runtime,
        ExecutionProfileComponentIdentity(
            component_class=ExecutionProfileComponentClass.FINALIZATION,
            strength=ExecutionProfileIdentityStrength.STRUCTURAL,
            availability=ExecutionProfileIdentityAvailability.AVAILABLE,
            fingerprint=hashlib.sha256(b"unrelated-finalization-change").hexdigest(),
        ),
    )
    profile = EvalExecutionProfileV1.create(
        profile_id=candidate.execution_profile.profile_id,
        label=candidate.execution_profile.label,
        source=candidate.execution_profile.source,
        target_key=candidate.execution_profile.target_key,
        application_release_id=candidate.execution_profile.application_release_id,
        app_manifest_fingerprint=candidate.execution_profile.app_manifest_fingerprint,
        candidate=candidate.execution_profile.candidate.model_copy(
            update={"runtime_execution_profile_fingerprint": altered.fingerprint}
        ),
        target_material=candidate.execution_profile.target_material,
        fixture_strategy=candidate.execution_profile.fixture_strategy,
        reset_strategy=candidate.execution_profile.reset_strategy,
        effect_posture=candidate.execution_profile.effect_posture,
        isolation_revision=candidate.execution_profile.isolation_revision,
        evidence_policy=candidate.execution_profile.evidence_policy,
        ceilings=candidate.execution_profile.ceilings,
    )
    binding = EvalExecutionProfileBindingV1(
        profile_revision=profile.revision,
        runtime_execution_profile=altered,
    )
    variants = (
        request.variants[0],
        candidate.model_copy(
            update={
                "execution_profile": profile,
                "execution_profile_binding": binding,
            }
        ),
    )

    with pytest.raises(ValidationError, match="exactly the automatic-recall"):
        MemoryExperimentReportRequest.model_validate(
            request.model_copy(update={"variants": variants}).model_dump(mode="python")
        )


def test_missing_operational_dimension_stays_unavailable_without_a_delta() -> None:
    request = asyncio.run(_report_request())
    selected = request.trials[1]
    partial_overhead = MemoryPreparationOverheadEvidence.create(context_bytes=101)
    replacement = selected.model_copy(update={"memory_overhead": partial_overhead})
    trials = tuple(replacement if item is selected else item for item in request.trials)

    report = build_memory_experiment_report(
        MemoryExperimentReportRequest.model_validate(
            request.model_copy(update={"trials": trials}).model_dump(mode="python")
        )
    )

    pair = report.comparisons[0].pairs[0]
    assert MemoryOperationalDimension.MEMORY_CONTEXT_BYTES in {
        item.dimension for item in pair.operational_deltas
    }
    assert MemoryOperationalDimension.MEMORY_PREPARATION_DURATION_MS not in {
        item.dimension for item in pair.operational_deltas
    }
    case_distribution = next(
        item
        for item in report.comparisons[0].operational_distributions
        if item.dimension is MemoryOperationalDimension.MEMORY_PREPARATION_DURATION_MS
    )
    experiment_distribution = next(
        item
        for item in report.operational_summary[0].distributions
        if item.dimension is MemoryOperationalDimension.MEMORY_PREPARATION_DURATION_MS
    )
    assert case_distribution.available_count == 1
    assert case_distribution.unavailable_count == 1
    assert experiment_distribution == case_distribution


def test_each_candidate_gets_an_independent_canonical_cost_report() -> None:
    request = asyncio.run(_report_request(candidate_count=2))

    report = build_memory_experiment_report(request)

    assert tuple(item.candidate_variant_id for item in report.cost_quality) == (
        "candidate",
        "candidate-2",
    )
    assert all(item.report is not None for item in report.cost_quality)
    assert all(item.report.aggregate.pair_count == 2 for item in report.cost_quality if item.report)
    report_pair_ids = tuple(
        {pair.pair_id for pair in item.report.pairs}
        for item in report.cost_quality
        if item.report is not None
    )
    assert report_pair_ids[0].isdisjoint(report_pair_ids[1])
    assert report.selected_variant_id == "candidate"
    baseline = next(item for item in report.dispositions if item.variant_id == "baseline")
    weaker = next(item for item in report.dispositions if item.variant_id == "candidate-2")
    assert baseline.status is MemoryVariantDispositionStatus.BASELINE_SUPERSEDED
    assert baseline.reasons == ("higher_ranked_candidate_selected",)
    assert weaker.status is MemoryVariantDispositionStatus.ELIGIBLE_NOT_SELECTED
    assert weaker.reasons == ("lower_eligible_lexicographic_rank",)
    assert weaker.comparable_pair_count == 2
    assert weaker.incomparable_pair_count == 0
    assert weaker.unavailable_pair_count == 0


def test_equal_eligible_candidates_report_deterministic_tie_break() -> None:
    request = asyncio.run(_report_request(candidate_count=2, candidate_scores=(0.9, 0.9)))

    report = build_memory_experiment_report(request)

    assert report.selected_variant_id == "candidate"
    tied = next(item for item in report.dispositions if item.variant_id == "candidate-2")
    assert tied.status is MemoryVariantDispositionStatus.ELIGIBLE_NOT_SELECTED
    assert tied.reasons == ("eligible_rank_tie_broken_by_variant_id",)


def test_missing_trial_stays_in_denominator_and_blocks_survivor_ranking() -> None:
    request = asyncio.run(_report_request())
    request = request.model_copy(update={"trials": request.trials[:-1]})

    report = build_memory_experiment_report(
        MemoryExperimentReportRequest.model_validate(request.model_dump(mode="python"))
    )

    assert len(report.rows) == 4
    assert report.rows[-1].availability is MemoryTrialAvailability.MISSING
    distribution = report.comparisons[0].distributions[0]
    assert distribution.pair_count == 2
    assert distribution.unavailable_count == 1
    assert distribution.incomparable_count == 0
    candidate = next(item for item in report.dispositions if item.variant_id == "candidate")
    assert candidate.status is MemoryVariantDispositionStatus.UNAVAILABLE
    assert candidate.reasons == ("incomplete_repeated_trial_matrix",)
    assert report.selected_variant_id == "baseline"


def test_completed_trial_without_published_result_is_retained_as_unmatched() -> None:
    request = asyncio.run(_report_request())
    selected = request.trials[1]
    unmatched = selected.model_copy(update={"published_result_revision": None})
    trials = tuple(unmatched if item is selected else item for item in request.trials)

    report = build_memory_experiment_report(
        MemoryExperimentReportRequest.model_validate(
            request.model_copy(update={"trials": trials}).model_dump(mode="python")
        )
    )

    row = next(
        item for item in report.rows if item.execution_id == unmatched.execution.execution_id
    )
    assert row.availability is MemoryTrialAvailability.UNMATCHED
    assert row.execution_status is MemoryInterventionExecutionStatus.COMPLETED
    assert report.comparisons[0].pairs[0].status is MemoryPairStatus.UNAVAILABLE
    candidate = next(item for item in report.dispositions if item.variant_id == "candidate")
    assert candidate.status is MemoryVariantDispositionStatus.UNAVAILABLE
    assert report.selected_variant_id == "baseline"


def test_published_unavailable_trials_remain_in_the_full_matrix() -> None:
    request = asyncio.run(_report_request(candidate_unavailable=True))

    report = build_memory_experiment_report(request)

    candidate_rows = tuple(row for row in report.rows if row.variant_id == "candidate")
    assert len(candidate_rows) == 2
    assert all(row.availability is MemoryTrialAvailability.UNAVAILABLE for row in candidate_rows)
    assert all(pair.status is MemoryPairStatus.UNAVAILABLE for pair in report.comparisons[0].pairs)
    candidate = next(item for item in report.dispositions if item.variant_id == "candidate")
    assert candidate.status is MemoryVariantDispositionStatus.UNAVAILABLE
    assert candidate.reasons == ("no_comparable_pairs",)
    assert report.selected_variant_id == "baseline"


@pytest.mark.parametrize(
    "completeness",
    (
        EvalMemoryEvidenceCompleteness.TRUNCATED,
        EvalMemoryEvidenceCompleteness.UNAVAILABLE,
    ),
)
def test_scored_trial_with_incomplete_attribution_is_retained_as_unavailable(
    completeness: EvalMemoryEvidenceCompleteness,
) -> None:
    request = asyncio.run(_report_request(candidate_attribution_completeness=completeness))

    report = build_memory_experiment_report(request)

    candidate_rows = tuple(row for row in report.rows if row.variant_id == "candidate")
    assert all(row.published_status == "passed" for row in candidate_rows)
    assert all(row.attribution_status is completeness for row in candidate_rows)
    assert all(row.availability is MemoryTrialAvailability.UNAVAILABLE for row in candidate_rows)
    assert all(pair.status is MemoryPairStatus.UNAVAILABLE for pair in report.comparisons[0].pairs)
    assert report.selected_variant_id == "baseline"


def test_report_reconstruction_cannot_promote_unavailable_pairs_to_comparable() -> None:
    unavailable = build_memory_experiment_report(
        asyncio.run(
            _report_request(
                candidate_attribution_completeness=EvalMemoryEvidenceCompleteness.TRUNCATED
            )
        )
    )
    available = build_memory_experiment_report(asyncio.run(_report_request()))
    assert tuple(row.row_id for row in unavailable.rows) == tuple(
        row.row_id for row in available.rows
    )
    assert unavailable.comparisons[0].pairs[0].status is MemoryPairStatus.UNAVAILABLE
    assert available.comparisons[0].pairs[0].status is MemoryPairStatus.COMPARABLE

    with pytest.raises(
        ValidationError,
        match="pair classification conflicts with its exact evidence",
    ):
        MemoryExperimentReport.model_validate(
            unavailable.model_copy(update={"comparisons": available.comparisons}).model_dump(
                mode="python"
            )
        )


def test_report_reconstruction_rejects_rankable_failed_trial_evidence() -> None:
    report = build_memory_experiment_report(asyncio.run(_report_request()))
    document = report.model_dump(mode="json")
    selected = next(
        item
        for item in document["rows"]
        if item["variant_id"] == "candidate" and item["repetition"] == 1
    )
    assert selected["availability"] == MemoryTrialAvailability.AVAILABLE.value
    selected["published_status"] = "error"
    document["revision"] = _memory_report_content_revision(
        {key: value for key, value in document.items() if key != "revision"},
        "memory experiment report",
    )

    with pytest.raises(
        ValidationError,
        match="Trial availability contradicts its retained status evidence",
    ):
        MemoryExperimentReport.model_validate_json(json.dumps(document))


def test_report_reconstruction_rejects_rankable_errored_metric() -> None:
    report = build_memory_experiment_report(asyncio.run(_report_request()))
    document = report.model_dump(mode="json")
    selected = next(
        item
        for item in document["rows"]
        if item["variant_id"] == "candidate" and item["repetition"] == 1
    )
    metric = selected["metrics"][0]
    assert metric["availability"] == MemoryMetricAvailability.AVAILABLE.value
    assert metric["value"] is not None
    metric["outcome"] = "error"
    document["revision"] = _memory_report_content_revision(
        {key: value for key, value in document.items() if key != "revision"},
        "memory experiment report",
    )

    with pytest.raises(
        ValidationError,
        match="Metric availability contradicts its published outcome",
    ):
        MemoryExperimentReport.model_validate_json(json.dumps(document))


@pytest.mark.parametrize(
    ("identity_field", "message"),
    (
        ("execution_id", "execution identities must be unique"),
        ("trial_id", "trial identities must be unique"),
    ),
)
def test_report_reconstruction_rejects_duplicate_trial_authority(
    identity_field: str,
    message: str,
) -> None:
    report = build_memory_experiment_report(asyncio.run(_report_request()))
    document = report.model_dump(mode="json")
    first, second = document["rows"][:2]
    assert first[identity_field] is not None
    assert second[identity_field] is not None
    second[identity_field] = first[identity_field]
    document["revision"] = _memory_report_content_revision(
        {key: value for key, value in document.items() if key != "revision"},
        "memory experiment report",
    )

    with pytest.raises(ValidationError, match=message):
        memory_experiment_report_from_json(json.dumps(document))


def test_report_reconstruction_revalidates_accounting_trial_authority() -> None:
    report = build_memory_experiment_report(asyncio.run(_report_request()))
    document = report.model_dump(mode="json")
    selected = next(item for item in document["rows"] if item["accounting_side"] is not None)
    selected["accounting_side"]["strategy_id"] = "foreign-variant"
    document["revision"] = _memory_report_content_revision(
        {key: value for key, value in document.items() if key != "revision"},
        "memory experiment report",
    )

    with pytest.raises(ValidationError, match="exact trial authority"):
        memory_experiment_report_from_json(json.dumps(document))


@pytest.mark.parametrize("missing_execution", (False, True))
def test_report_reconstruction_rejects_metrics_without_a_published_trial(
    missing_execution: bool,
) -> None:
    request = asyncio.run(_report_request())
    selected = next(
        item for item in request.trials if item.variant_id == "candidate" and item.repetition == 1
    )
    if missing_execution:
        trials = tuple(item for item in request.trials if item is not selected)
    else:
        unmatched = selected.model_copy(update={"published_result_revision": None})
        trials = tuple(unmatched if item is selected else item for item in request.trials)
    report = build_memory_experiment_report(
        MemoryExperimentReportRequest.model_validate(
            request.model_copy(update={"trials": trials}).model_dump(mode="python")
        )
    )
    document = report.model_dump(mode="json")
    row = next(
        item
        for item in document["rows"]
        if item["variant_id"] == "candidate" and item["repetition"] == 1
    )
    assert row["source_trial_revision"] is None
    metric = row["metrics"][0]
    metric.update(
        {
            "availability": MemoryMetricAvailability.AVAILABLE.value,
            "outcome": "passed",
            "value": 0.9,
        }
    )
    document["revision"] = _memory_report_content_revision(
        {key: value for key, value in document.items() if key != "revision"},
        "memory experiment report",
    )

    with pytest.raises(ValidationError, match="cannot retain metric evidence"):
        memory_experiment_report_from_json(json.dumps(document))


def test_standalone_case_comparison_revalidates_distributions_from_pairs() -> None:
    report = build_memory_experiment_report(asyncio.run(_report_request()))
    document = report.comparisons[0].model_dump(mode="json")
    distribution = document["distributions"][0]
    assert distribution["available_count"] > 0
    distribution["baseline_values"][0] = 0.0
    distribution["candidate_values"][0] = distribution["deltas"][0]
    document["revision"] = _memory_report_content_revision(
        {key: value for key, value in document.items() if key != "revision"},
        "memory case comparison",
    )

    with pytest.raises(
        ValidationError,
        match="Memory case comparison distributions conflict with their pairs",
    ):
        MemoryCaseComparison.model_validate_json(json.dumps(document))


@pytest.mark.parametrize(
    "status",
    (MemoryPairStatus.INCOMPARABLE, MemoryPairStatus.UNAVAILABLE),
)
def test_standalone_case_comparison_rejects_deltas_for_noncomparable_pairs(
    status: MemoryPairStatus,
) -> None:
    report = build_memory_experiment_report(asyncio.run(_report_request()))
    document = report.comparisons[0].model_dump(mode="json")
    pair = document["pairs"][0]
    assert pair["metric_deltas"]
    assert pair["operational_deltas"]
    pair["status"] = status.value
    if status is MemoryPairStatus.UNAVAILABLE:
        pair["memory_comparability_fingerprint"] = None
    document["revision"] = _memory_report_content_revision(
        {key: value for key, value in document.items() if key != "revision"},
        "memory case comparison",
    )

    with pytest.raises(ValidationError, match="Only comparable memory trial pairs"):
        MemoryCaseComparison.model_validate_json(json.dumps(document))


@pytest.mark.parametrize(
    ("field", "message"),
    (
        ("distributions", "retain every declared metric distribution"),
        ("operational_distributions", "retain every operational distribution"),
    ),
)
def test_standalone_case_comparison_rejects_omitted_distributions(
    field: str,
    message: str,
) -> None:
    report = build_memory_experiment_report(asyncio.run(_report_request()))
    document = report.comparisons[0].model_dump(mode="json")
    assert document[field]
    document[field] = []
    document["revision"] = _memory_report_content_revision(
        {key: value for key, value in document.items() if key != "revision"},
        "memory case comparison",
    )

    with pytest.raises(ValidationError, match=message):
        MemoryCaseComparison.model_validate_json(json.dumps(document))


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("pair_id", "pair identity conflicts with its coordinates"),
        ("baseline_row_id", "row identities conflict with its coordinates"),
        ("accounting_pair_id", "accounting identity conflicts"),
        ("missing_comparability", "requires comparability evidence"),
    ),
)
def test_standalone_case_comparison_revalidates_pair_authority(
    mutation: str,
    message: str,
) -> None:
    report = build_memory_experiment_report(asyncio.run(_report_request()))
    document = report.comparisons[0].model_dump(mode="json")
    pair = document["pairs"][0]
    forged_revision = "sha256:" + "0" * 64
    if mutation == "pair_id":
        pair["pair_id"] = forged_revision
    elif mutation == "baseline_row_id":
        pair["baseline_row_id"] = forged_revision
    elif mutation == "accounting_pair_id":
        assert pair["accounting_pair"] is not None
        pair["accounting_pair"]["pair_id"] = forged_revision
    else:
        assert pair["status"] != MemoryPairStatus.UNAVAILABLE.value
        pair["memory_comparability_fingerprint"] = None
    document["revision"] = _memory_report_content_revision(
        {key: value for key, value in document.items() if key != "revision"},
        "memory case comparison",
    )

    with pytest.raises(ValidationError, match=message):
        MemoryCaseComparison.model_validate_json(json.dumps(document))


def test_standalone_case_comparison_requires_one_baseline_variant() -> None:
    report = build_memory_experiment_report(asyncio.run(_report_request()))
    document = report.comparisons[0].model_dump(mode="json")
    second = document["pairs"][1]
    second["baseline_variant_id"] = "foreign-baseline"
    second["baseline_row_id"] = memory_reporting._row_id(
        second["case_id"],
        second["repetition"],
        second["baseline_variant_id"],
    )
    document["revision"] = _memory_report_content_revision(
        {key: value for key, value in document.items() if key != "revision"},
        "memory case comparison",
    )

    with pytest.raises(ValidationError, match="one exact baseline variant"):
        MemoryCaseComparison.model_validate_json(json.dumps(document))


def test_missing_accounting_is_retained_and_fails_the_priced_cost_gate() -> None:
    request = asyncio.run(_report_request())
    selected = request.trials[1]
    without_accounting = selected.model_copy(update={"accounting_side": None})
    trials = tuple(without_accounting if item is selected else item for item in request.trials)

    report = build_memory_experiment_report(
        MemoryExperimentReportRequest.model_validate(
            request.model_copy(update={"trials": trials}).model_dump(mode="python")
        )
    )

    assert report.cost_quality[0].report is not None
    accounting = report.comparisons[0].pairs[0].accounting_pair
    assert accounting is not None
    assert accounting.status.value == "unavailable"
    assert accounting.candidate is None
    candidate = next(item for item in report.dispositions if item.variant_id == "candidate")
    assert candidate.status is MemoryVariantDispositionStatus.REJECTED
    assert "priced_cost_unavailable" in candidate.reasons
    assert report.selected_variant_id == "baseline"


def test_completely_missing_accounting_retains_every_canonical_pair() -> None:
    request = asyncio.run(_report_request())
    trials = tuple(item.model_copy(update={"accounting_side": None}) for item in request.trials)

    report = build_memory_experiment_report(
        MemoryExperimentReportRequest.model_validate(
            request.model_copy(update={"trials": trials}).model_dump(mode="python")
        )
    )

    canonical = report.cost_quality[0].report
    assert canonical is not None
    assert canonical.status is CostQualityComparisonStatus.UNAVAILABLE
    assert canonical.aggregate.pair_count == request.repetitions
    assert len(canonical.aggregate.exclusions) == request.repetitions
    assert all(pair.status is CostQualityComparisonStatus.UNAVAILABLE for pair in canonical.pairs)
    assert all(pair.baseline is None and pair.candidate is None for pair in canonical.pairs)
    assert all(
        pair.accounting_pair is not None
        and pair.accounting_pair.status is CostQualityComparisonStatus.UNAVAILABLE
        for pair in report.comparisons[0].pairs
    )


def test_unpriced_accounting_is_retained_and_fails_the_priced_cost_gate() -> None:
    request = asyncio.run(_report_request())
    selected = request.trials[1]
    assert selected.accounting_side is not None
    first_attempt, *remaining_attempts = selected.accounting_side.attempts
    unpriced_cost = _cost(
        session_id=first_attempt.session_id,
        input_tokens=10,
        price_book=_price_book(model="different-model"),
    )
    unpriced_side = selected.accounting_side.model_copy(
        update={
            "attempts": (
                first_attempt.model_copy(update={"cost": unpriced_cost}),
                *remaining_attempts,
            )
        }
    )
    unpriced = selected.model_copy(update={"accounting_side": unpriced_side})
    trials = tuple(unpriced if item is selected else item for item in request.trials)

    report = build_memory_experiment_report(
        MemoryExperimentReportRequest.model_validate(
            request.model_copy(update={"trials": trials}).model_dump(mode="python")
        )
    )

    assert report.cost_quality[0].report is not None
    accounting = report.comparisons[0].pairs[0].accounting_pair
    assert accounting is not None
    assert accounting.status.value == "unpriced"
    assert accounting.candidate is not None
    assert accounting.candidate.whole_harness.unpriced_attempt_count == 1
    candidate = next(item for item in report.dispositions if item.variant_id == "candidate")
    assert candidate.status is MemoryVariantDispositionStatus.REJECTED
    assert "priced_cost_unavailable" in candidate.reasons
    assert report.selected_variant_id == "baseline"


def test_unavailable_trial_precedes_unmatched_accounting_classification() -> None:
    request = asyncio.run(_report_request(candidate_unavailable=True))
    selected = request.trials[1]
    assert selected.accounting_side is not None
    unmatched_side = selected.accounting_side.model_copy(
        update={
            "quality": _quality(
                status=QualityEvidenceStatus.FAILED,
            )
        }
    )
    replacement = selected.model_copy(update={"accounting_side": unmatched_side})
    trials = tuple(replacement if item is selected else item for item in request.trials)

    report = build_memory_experiment_report(
        MemoryExperimentReportRequest.model_validate(
            request.model_copy(update={"trials": trials}).model_dump(mode="python")
        )
    )

    pair = report.comparisons[0].pairs[0]
    assert pair.accounting_pair is not None
    assert pair.accounting_pair.status.value == "measured_unmatched"
    assert pair.status is MemoryPairStatus.UNAVAILABLE
    assert pair.reasons == (
        "accounting_not_comparable",
        "trial_evidence_unavailable",
    )


def test_accounting_attempt_cannot_be_counted_in_multiple_trial_rows() -> None:
    request = asyncio.run(_report_request())
    baseline = request.trials[0]
    candidate = request.trials[1]
    assert baseline.accounting_side is not None
    assert candidate.accounting_side is not None
    duplicate_id = baseline.accounting_side.attempts[0].attempt_id
    candidate_attempts = (
        candidate.accounting_side.attempts[0].model_copy(update={"attempt_id": duplicate_id}),
        *candidate.accounting_side.attempts[1:],
    )
    duplicate_side = candidate.accounting_side.model_copy(update={"attempts": candidate_attempts})
    duplicate = candidate.model_copy(update={"accounting_side": duplicate_side})
    trials = tuple(duplicate if item is candidate else item for item in request.trials)

    with pytest.raises(ValidationError, match="exactly one trial row"):
        MemoryExperimentReportRequest.model_validate(
            request.model_copy(update={"trials": trials}).model_dump(mode="python")
        )


def test_accounting_side_cannot_move_between_repetitions() -> None:
    request = asyncio.run(_report_request())
    first, second = (
        item
        for item in request.trials
        if item.variant_id == "candidate" and item.repetition in {1, 2}
    )
    assert first.accounting_side is not None
    assert second.accounting_side is not None
    replacements = {
        first.repetition: first.model_copy(update={"accounting_side": second.accounting_side}),
        second.repetition: second.model_copy(update={"accounting_side": first.accounting_side}),
    }
    trials = tuple(
        replacements[item.repetition]
        if item.variant_id == "candidate" and item.repetition in replacements
        else item
        for item in request.trials
    )

    with pytest.raises(ValidationError, match="exact trial authority"):
        MemoryExperimentReportRequest.model_validate(
            request.model_copy(update={"trials": trials}).model_dump(mode="python")
        )


def test_gate_failure_precedes_quality_ranking() -> None:
    request = asyncio.run(_report_request())
    request = MemoryExperimentReportRequest.model_validate(
        request.model_copy(
            update={
                "gates": MemoryExperimentGatePolicy(
                    metric_gates=(
                        MemoryMetricGate(
                            role=MemoryMetricRole.SAFETY,
                            minimum=0.95,
                        ),
                    ),
                    minimum_comparable_pairs=2,
                )
            }
        ).model_dump(mode="python")
    )

    report = build_memory_experiment_report(request)

    candidate = next(item for item in report.dispositions if item.variant_id == "candidate")
    assert candidate.status is MemoryVariantDispositionStatus.REJECTED
    assert candidate.reasons == ("gate_failed:safety",)
    assert report.selected_variant_id == "baseline"


def test_metric_gate_rejects_partial_comparable_evidence() -> None:
    request = asyncio.run(_report_request())
    gates = MemoryExperimentGatePolicy(
        metric_gates=(
            MemoryMetricGate(
                role=MemoryMetricRole.SAFETY,
                minimum=0.8,
            ),
        ),
        minimum_comparable_pairs=2,
    )
    report = build_memory_experiment_report(request)
    comparison = report.comparisons[0]
    first_pair = comparison.pairs[0]
    first_pair_without_safety = first_pair.model_copy(
        update={
            "metric_deltas": tuple(
                delta
                for delta in first_pair.metric_deltas
                if delta.role is not MemoryMetricRole.SAFETY
            )
        }
    )
    comparisons = (
        comparison.model_copy(update={"pairs": (first_pair_without_safety, *comparison.pairs[1:])}),
    )

    selected, dispositions = memory_reporting._select_memory_variant(
        request.variants,
        request.baseline_variant_id,
        comparisons,
        report.cost_quality,
        request.metric_bindings,
        request.ranking,
        gates,
    )

    candidate = next(item for item in dispositions if item.variant_id == "candidate")
    assert candidate.status is MemoryVariantDispositionStatus.REJECTED
    assert candidate.reasons == ("gate_evidence_unavailable:safety",)
    assert selected == "baseline"


@pytest.mark.parametrize(
    "role",
    (
        MemoryMetricRole.FACTUAL_SUPPORT,
        MemoryMetricRole.HALLUCINATION_AVOIDANCE,
        MemoryMetricRole.SAFETY,
    ),
)
def test_evidence_dimension_cannot_bypass_gates_through_ranking(
    role: MemoryMetricRole,
) -> None:
    request = asyncio.run(_report_request())

    with pytest.raises(ValidationError, match="safety and evidence dimensions"):
        MemoryExperimentReportRequest.model_validate(
            request.model_copy(
                update={
                    "ranking": (
                        MemoryRankingTerm(
                            role=role,
                            direction=MemoryMetricDirection.HIGHER_IS_BETTER,
                        ),
                    )
                }
            ).model_dump(mode="python")
        )


def test_one_published_assertion_cannot_bind_ranking_and_gate_roles() -> None:
    request = asyncio.run(_report_request())
    task_quality = next(
        item for item in request.metric_bindings if item.role is MemoryMetricRole.TASK_QUALITY
    )
    safety = next(item for item in request.metric_bindings if item.role is MemoryMetricRole.SAFETY)
    aliased_safety = safety.model_copy(
        update={
            "assertion_id": task_quality.assertion_id,
            "assertion_revision": task_quality.assertion_revision,
        }
    )
    metric_bindings = tuple(
        aliased_safety if item is safety else item for item in request.metric_bindings
    )

    with pytest.raises(ValidationError, match="exactly one memory metric role"):
        MemoryExperimentReportRequest.model_validate(
            request.model_copy(update={"metric_bindings": metric_bindings}).model_dump(
                mode="python"
            )
        )


def test_json_report_rejects_safety_metric_in_ranking() -> None:
    report = build_memory_experiment_report(asyncio.run(_report_request()))
    document = report.model_dump(mode="python")
    document["ranking"] = (
        MemoryRankingTerm(
            role=MemoryMetricRole.SAFETY,
            direction=MemoryMetricDirection.HIGHER_IS_BETTER,
        ),
    )

    with pytest.raises(ValidationError, match="safety and evidence dimensions"):
        MemoryExperimentReport.model_validate(document)


def test_json_and_html_preserve_the_same_versioned_statuses() -> None:
    report = build_memory_experiment_report(asyncio.run(_report_request()))

    encoded = memory_experiment_report_to_json(report)
    reconstructed = memory_experiment_report_from_json(encoded)
    rendered = render_memory_experiment_report_html(report)

    assert reconstructed == report
    assert memory_experiment_report_to_json(reconstructed) == encoded
    assert report.revision in encoded
    assert report.revision in rendered
    assert f"Schema version <code>{report.schema_version}</code>" in rendered
    assert report.cases[0].case_revision in rendered
    assert all(item.spec.fingerprint in rendered for item in report.variants)
    assert all(item.execution_profile.revision in rendered for item in report.variants)
    assert all(item.evaluator_fingerprint in rendered for item in report.variants)
    assert "Required evidence roles" in rendered
    assert "privacy, safety" in rendered
    assert "Task-quality ranking" in rendered
    assert "higher_is_better" in rendered
    assert "Complete trial matrix" in rendered
    assert "Canonical cost/usage evidence" in rendered
    assert "Canonical cost/usage aggregates" in rendered
    assert "verified" in rendered
    assert "retries=1/1" in rendered
    assert "Experiment operational distributions" in rendered
    for row in report.rows:
        assert row.execution_id is not None
        assert row.trial_id is not None
        assert row.execution_revision is not None
        assert row.source_trial_revision is not None
        assert row.final_binding_fingerprint is not None
        assert row.execution_binding_lineage_revision is not None
        assert row.intervention_attribution_fingerprint is not None
        assert row.attribution_evidence_revision is not None
        assert row.execution_id in rendered
        assert row.trial_id in rendered
        assert str(row.execution_revision) in rendered
        assert row.source_trial_revision in rendered
        assert row.final_binding_fingerprint in rendered
        assert row.execution_binding_lineage_revision in rendered
        assert row.intervention_attribution_fingerprint in rendered
        assert row.attribution_evidence_revision in rendered
        for observation in row.metrics:
            assert str(observation.outcome) in rendered
            if observation.evaluator_key is not None:
                assert observation.evaluator_implementation_revision is not None
                assert observation.evaluator_key in rendered
                assert observation.evaluator_implementation_revision in rendered
    aggregate = report.cost_quality[0].report
    assert aggregate is not None
    assert f"pairs={aggregate.aggregate.pair_count}" in rendered
    if aggregate.aggregate.baseline_cost is not None:
        assert str(aggregate.aggregate.baseline_cost) in rendered
    if aggregate.aggregate.candidate_cost is not None:
        assert str(aggregate.aggregate.candidate_cost) in rendered
    assert aggregate.aggregate.cost_direction.value in rendered


def test_json_serializer_uses_the_same_compact_byte_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = build_memory_experiment_report(asyncio.run(_report_request()))
    compact = json.dumps(
        report.model_dump(mode="json"),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    pretty = json.dumps(
        report.model_dump(mode="json"),
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
    compact_bytes = len(compact.encode("utf-8"))
    assert len(pretty.encode("utf-8")) > compact_bytes
    monkeypatch.setattr(
        memory_reporting,
        "MEMORY_EXPERIMENT_REPORT_MAX_BYTES",
        compact_bytes,
    )

    encoded = memory_experiment_report_to_json(report)

    assert encoded == compact
    assert len(encoded.encode("utf-8")) == compact_bytes


def test_json_loader_rejects_duplicate_decision_fields() -> None:
    request = asyncio.run(_report_request())
    encoded = json.dumps(request.model_dump(mode="json"), ensure_ascii=False)
    duplicated = encoded.replace(
        '"schema_version": 1',
        '"schema_version": 1, "schema_version": 1',
        1,
    )

    with pytest.raises(ValueError, match="duplicate JSON object keys"):
        memory_experiment_request_from_json(duplicated)


def test_request_rejects_unreferenced_published_result_graphs() -> None:
    request = asyncio.run(_report_request())

    with pytest.raises(ValidationError, match="exactly match the referenced result graph"):
        MemoryExperimentReportRequest.model_validate(
            request.model_copy(update={"trials": ()}).model_dump(mode="python")
        )


def test_request_and_report_reject_duplicate_case_ids_across_revisions() -> None:
    request = asyncio.run(_report_request())
    second_case = MemoryExperimentCase(
        case_id=request.cases[0].case_id,
        case_revision="sha256:" + "f" * 64,
    )
    cases = tuple(sorted((*request.cases, second_case), key=lambda item: item.case_revision))

    with pytest.raises(ValidationError, match="case IDs must be unique"):
        MemoryExperimentReportRequest.model_validate(
            request.model_copy(update={"cases": cases}).model_dump(mode="python")
        )

    report = build_memory_experiment_report(request)
    with pytest.raises(ValidationError, match="case IDs must be unique"):
        type(report).model_validate(
            report.model_copy(update={"cases": cases}).model_dump(mode="python")
        )


def test_memory_report_cli_builds_deterministic_json_and_html(tmp_path) -> None:
    request = asyncio.run(_report_request())
    source = tmp_path / "memory-request.json"
    json_output = tmp_path / "memory-report.json"
    html_output = tmp_path / "memory-report.html"
    source.write_text(
        json.dumps(request.model_dump(mode="json"), ensure_ascii=False),
        encoding="utf-8",
    )
    assert (
        build_memory_experiment_report(
            memory_experiment_request_from_json(source.read_bytes())
        ).selected_variant_id
        == "candidate"
    )

    assert (
        main(
            [
                "eval",
                "memory-report",
                str(source),
                "--format",
                "json",
                "--output",
                str(json_output),
            ]
        )
        == 0
    )
    report = memory_experiment_report_from_json(json_output.read_bytes())
    assert report.selected_variant_id == "candidate"

    assert (
        main(
            [
                "eval",
                "memory-report",
                str(source),
                "--format",
                "html",
                "--output",
                str(html_output),
            ]
        )
        == 0
    )
    assert "Complete trial matrix" in html_output.read_text(encoding="utf-8")


def test_memory_report_cli_rejects_wrong_type_published_result_without_echo(
    tmp_path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    request = asyncio.run(_report_request())
    source = tmp_path / "invalid-memory-request.json"
    document = request.model_dump(mode="json")
    canary = "wrong-result-secret-canary"
    document["published_results"][0]["result"] = canary
    source.write_text(json.dumps(document), encoding="utf-8")

    assert (
        main(
            [
                "eval",
                "memory-report",
                str(source),
                "--format",
                "json",
            ]
        )
        == 2
    )
    streams = capsys.readouterr()
    assert canary not in streams.out
    assert canary not in streams.err
    assert "Published result evidence must be an exact corpus execution result" in streams.out


def test_report_rejects_conflicting_published_source_trial_identity() -> None:
    request = asyncio.run(_report_request())
    trial = request.trials[0]
    assert trial.intervention_binding is not None
    foreign_binding = _binding_with_eval_result_revision(
        trial.intervention_binding,
        "f" * 64,
    )
    document = trial.execution.model_dump(mode="python")
    document.update(
        {
            "eval_result_revision": "f" * 64,
            "snapshot_result_fingerprint": foreign_binding.result.fingerprint,
            "final_binding_fingerprint": foreign_binding.fingerprint,
        }
    )
    forged = trial.model_copy(
        update={
            "execution": MemoryInterventionExecutionRecord.model_validate(document),
            "intervention_binding": foreign_binding,
        }
    )
    trials = (forged, *request.trials[1:])

    with pytest.raises(ValidationError, match="Published trial conflicts"):
        MemoryExperimentReportRequest.model_validate(
            request.model_copy(update={"trials": trials}).model_dump(mode="python")
        )


def _binding_with_eval_result_revision(
    binding: MemoryInterventionTrialBinding,
    eval_result_revision: str,
) -> MemoryInterventionTrialBinding:
    result = AgentSnapshotResultBinding.create(
        trial=binding.trial,
        session_id=binding.result.session_id,
        terminal_disposition=binding.result.terminal_disposition,
        runtime_evidence_fingerprint=binding.result.runtime_evidence_fingerprint,
        eval_result_revision=eval_result_revision,
        recorded_at=binding.result.recorded_at,
        memory_evidence_fingerprint=binding.result.memory_evidence_fingerprint,
        usage_fingerprint=binding.result.usage_fingerprint,
        cost_fingerprint=binding.result.cost_fingerprint,
    )
    return MemoryInterventionTrialBinding.create(
        spec=binding.spec,
        operation=binding.operation,
        receipt=binding.receipt,
        trial=binding.trial,
        result=result,
        attribution=binding.attribution,
        terminal_evidence_available=binding.terminal_evidence_available,
        expected_receipt_count=binding.expected_receipt_count,
        expected_exposure_count=binding.expected_exposure_count,
    )


def test_request_rejects_binding_from_another_eval_result() -> None:
    request = asyncio.run(_report_request())
    selected = request.trials[0]
    assert selected.intervention_binding is not None
    foreign_binding = _binding_with_eval_result_revision(
        selected.intervention_binding,
        "f" * 64,
    )
    execution_document = selected.execution.model_dump(mode="python")
    execution_document.update(
        {
            "snapshot_result_fingerprint": foreign_binding.result.fingerprint,
            "final_binding_fingerprint": foreign_binding.fingerprint,
        }
    )
    forged = selected.model_copy(
        update={
            "execution": MemoryInterventionExecutionRecord.model_validate(execution_document),
            "intervention_binding": foreign_binding,
        }
    )

    with pytest.raises(ValidationError, match="complete execution lineage"):
        MemoryExperimentReportRequest.model_validate(
            request.model_copy(update={"trials": (forged, *request.trials[1:])}).model_dump(
                mode="python"
            )
        )


def test_report_reconstruction_revalidates_complete_execution_lineage() -> None:
    report = build_memory_experiment_report(asyncio.run(_report_request()))
    selected = report.rows[0]
    assert selected.intervention_binding is not None
    foreign_binding = _binding_with_eval_result_revision(
        selected.intervention_binding,
        "f" * 64,
    )
    forged_row = selected.model_copy(
        update={
            "final_binding_fingerprint": foreign_binding.fingerprint,
            "intervention_binding": foreign_binding,
        }
    )

    with pytest.raises(ValidationError, match="complete execution lineage"):
        MemoryExperimentReport.model_validate(
            report.model_copy(update={"rows": (forged_row, *report.rows[1:])}).model_dump(
                mode="python"
            )
        )


def test_nonterminal_execution_is_rejected_by_the_public_request_parser() -> None:
    request = asyncio.run(_report_request())
    selected = request.trials[0]
    execution_document = selected.execution.model_dump(mode="python")
    execution_document.update(
        {
            "phase": MemoryInterventionExecutionPhase.EVALUATED,
            "status": MemoryInterventionExecutionStatus.ACTIVE,
            "snapshot_result_fingerprint": None,
            "final_binding_fingerprint": None,
            "failure_code": None,
        }
    )
    active = selected.model_copy(
        update={
            "execution": MemoryInterventionExecutionRecord.model_validate(execution_document),
            "intervention_binding": None,
        }
    )
    document = request.model_copy(update={"trials": (active, *request.trials[1:])}).model_dump(
        mode="json"
    )

    with pytest.raises(ValidationError, match="require terminal trial executions"):
        memory_experiment_request_from_json(json.dumps(document))


def test_report_reconstruction_rejects_nonterminal_execution() -> None:
    report = build_memory_experiment_report(asyncio.run(_report_request()))
    selected = report.rows[0]
    active = selected.model_copy(
        update={"execution_status": MemoryInterventionExecutionStatus.ACTIVE}
    )

    with pytest.raises(ValidationError, match="require terminal trial executions"):
        MemoryExperimentReport.model_validate(
            report.model_copy(update={"rows": (active, *report.rows[1:])}).model_dump(mode="python")
        )


def test_report_rejects_profile_snapshot_and_binding_mismatch() -> None:
    request = asyncio.run(_report_request())
    selected = request.variants[1]
    mismatched_binding = selected.execution_profile_binding.model_copy(
        update={"profile_revision": "sha256:" + "c" * 64}
    )
    variants = (
        request.variants[0],
        selected.model_copy(update={"execution_profile_binding": mismatched_binding}),
    )

    with pytest.raises(ValidationError, match="execution profile evidence conflicts"):
        MemoryExperimentReportRequest.model_validate(
            request.model_copy(update={"variants": variants}).model_dump(mode="python")
        )


def test_eval_profile_ceiling_change_makes_every_pair_incomparable() -> None:
    request = asyncio.run(_report_request())
    candidate = request.variants[1]
    ceilings = candidate.execution_profile.ceilings.model_copy(
        update={"max_timeout_seconds": candidate.execution_profile.ceilings.max_timeout_seconds - 1}
    )
    profile = EvalExecutionProfileV1.create(
        profile_id=candidate.execution_profile.profile_id,
        label=candidate.execution_profile.label,
        source=candidate.execution_profile.source,
        target_key=candidate.execution_profile.target_key,
        application_release_id=candidate.execution_profile.application_release_id,
        app_manifest_fingerprint=candidate.execution_profile.app_manifest_fingerprint,
        candidate=candidate.execution_profile.candidate,
        target_material=candidate.execution_profile.target_material,
        fixture_strategy=candidate.execution_profile.fixture_strategy,
        reset_strategy=candidate.execution_profile.reset_strategy,
        effect_posture=candidate.execution_profile.effect_posture,
        isolation_revision=candidate.execution_profile.isolation_revision,
        evidence_policy=candidate.execution_profile.evidence_policy,
        ceilings=ceilings,
    )
    binding = EvalExecutionProfileBindingV1(
        profile_revision=profile.revision,
        runtime_execution_profile=(candidate.execution_profile_binding.runtime_execution_profile),
    )
    variants = (
        request.variants[0],
        candidate.model_copy(
            update={
                "execution_profile": profile,
                "execution_profile_binding": binding,
            }
        ),
    )
    request = MemoryExperimentReportRequest.model_validate(
        request.model_copy(update={"variants": variants}).model_dump(mode="python")
    )

    report = build_memory_experiment_report(request)

    assert all(
        pair.status is MemoryPairStatus.INCOMPARABLE
        and "generic_experiment_identity" in pair.reasons
        for comparison in report.comparisons
        for pair in comparison.pairs
    )
    assert report.selected_variant_id == "baseline"


def test_published_evidence_policy_must_match_its_execution_profile() -> None:
    request = asyncio.run(_report_request())
    candidate_evidence = next(item for item in request.trials if item.variant_id == "candidate")
    assert candidate_evidence.published_result_revision is not None
    candidate_result = next(
        item.result
        for item in request.published_results
        if item.result.revision == candidate_evidence.published_result_revision
    )
    changed_result = _published_result_with_run_updates(
        candidate_result,
        evidence_policy_revision="sha256:" + "f" * 64,
    )

    with pytest.raises(ValidationError, match="evidence policy conflicts"):
        _request_with_candidate_result(request, changed_result)


def test_report_reconstruction_rejects_profile_policy_conflicts() -> None:
    report = build_memory_experiment_report(asyncio.run(_report_request()))
    document = report.model_dump(mode="json")
    for row in document["rows"]:
        assert row["published_result_revision"] is not None
        row["evidence_policy_revision"] = "sha256:" + "f" * 64
    document["revision"] = _memory_report_content_revision(
        {key: value for key, value in document.items() if key != "revision"},
        "memory experiment report",
    )

    with pytest.raises(ValidationError, match="row conflicts with its frozen variant"):
        memory_experiment_report_from_json(json.dumps(document))


def test_candidate_budget_uses_canonical_aggregate_at_and_above_limit() -> None:
    request = asyncio.run(_report_request())
    canonical = build_memory_experiment_report(request).cost_quality[0].report
    assert canonical is not None
    candidate_cost = canonical.aggregate.candidate_cost
    currency = canonical.aggregate.currency
    assert candidate_cost is not None
    assert currency is not None
    below_candidate_cost = candidate_cost - Decimal("0.01")

    def with_limit(limit: Decimal) -> MemoryExperimentReportRequest:
        return MemoryExperimentReportRequest.model_validate(
            request.model_copy(
                update={
                    "gates": request.gates.model_copy(
                        update={
                            "maximum_candidate_cost": limit,
                            "cost_currency": currency,
                        }
                    )
                }
            ).model_dump(mode="python")
        )

    with localcontext() as context:
        context.prec = 2
        at_limit = build_memory_experiment_report(with_limit(candidate_cost))
        below_limit = build_memory_experiment_report(with_limit(below_candidate_cost))

    assert at_limit.selected_variant_id == "candidate"
    rejected = next(item for item in below_limit.dispositions if item.variant_id == "candidate")
    assert rejected.status is MemoryVariantDispositionStatus.REJECTED
    assert rejected.reasons == ("budget_exceeded",)


def test_large_aggregate_token_counts_round_trip_and_render() -> None:
    report = build_memory_experiment_report(asyncio.run(_report_request(total_tokens_base=2**63)))

    encoded = memory_experiment_report_to_json(report)
    reconstructed = memory_experiment_report_from_json(encoded)
    rendered = render_memory_experiment_report_html(report)

    assert reconstructed == report
    assert all(row.total_tokens is not None and row.total_tokens >= 2**63 for row in report.rows)
    assert f'"total_tokens":"{2**63 + 1}"' in encoded
    assert str(2**63 + 1) in rendered


def test_expanded_pair_matrix_is_bounded_before_report_construction() -> None:
    request = asyncio.run(_report_request())

    with pytest.raises(ValidationError, match="paired-comparison bound"):
        MemoryExperimentReportRequest.model_validate(
            request.model_copy(update={"repetitions": 257}).model_dump(mode="python")
        )


def test_model_judge_rollout_is_retained_and_makes_pairs_incomparable() -> None:
    request = asyncio.run(_report_request(candidate_evaluator_revision="sha256:" + "c" * 64))

    report = build_memory_experiment_report(request)

    assert all(
        pair.status is MemoryPairStatus.INCOMPARABLE
        and pair.reasons == ("evaluator_implementation_identity",)
        for comparison in report.comparisons
        for pair in comparison.pairs
    )
    candidate_rows = tuple(row for row in report.rows if row.variant_id == "candidate")
    assert all(
        observation.evaluator_implementation_revision == "sha256:" + "c" * 64
        for row in candidate_rows
        for observation in row.metrics
    )
    candidate = next(item for item in report.dispositions if item.variant_id == "candidate")
    assert candidate.incomparable_pair_count == 2
    assert candidate.unavailable_pair_count == 0
    assert report.selected_variant_id == "baseline"


def test_structured_model_judge_identity_is_retained_for_comparability() -> None:
    judge, _ = _structured_judge(_structured_judgment())
    target, _ = _structured_judge_target(judge)
    result = asyncio.run(
        run_corpus_suite(
            target,
            _structured_judge_corpus(judge),
            "quality-suite",
        )
    )
    published_trial = result.run.cases[0].trials[0]
    assertion = published_trial.assertions[0]
    assert isinstance(assertion.detail, PublishedStructuredModelJudgeDetail)

    observation = memory_reporting._observations(
        published_trial,
        (
            MemoryMetricBinding(
                role=MemoryMetricRole.TASK_QUALITY,
                assertion_id=assertion.assertion_id,
                assertion_revision=assertion.assertion_revision,
            ),
        ),
    )[0]

    assert observation.evaluator_key == assertion.detail.judge_profile.key
    assert (
        observation.evaluator_implementation_revision
        == assertion.detail.judge_profile.implementation_revision
    )


@pytest.mark.parametrize("mutated_field", ("binding", "snapshot"))
def test_eval_invocation_profile_copy_rejects_mutation_without_diagnostic_leak(
    mutated_field: str,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    request = asyncio.run(_report_request())
    binding = request.variants[0].execution_profile_binding.model_copy(deep=True)
    snapshot = request.variants[0].execution_profile.model_copy(deep=True)
    secret = f"profile-copy-{mutated_field}-secret-canary"
    if mutated_field == "binding":
        object.__setattr__(binding, "profile_revision", _SecretRepr(secret))
    else:
        object.__setattr__(snapshot, "revision", _SecretRepr(secret))
    caplog.set_level(logging.DEBUG)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(ValidationError) as captured:
            EvalRunInvocation(
                execution_profile=binding,
                execution_profile_snapshot=snapshot,
            )

    streams = capsys.readouterr()
    diagnostic_text = "\n".join(
        (
            str(captured.value),
            repr(captured.value),
            *(str(item.message) for item in caught),
            caplog.text,
            streams.out,
            streams.err,
        )
    )
    assert secret not in diagnostic_text


@pytest.mark.parametrize(
    ("status", "failure_code", "availability"),
    (
        (MemoryInterventionExecutionStatus.FAILED, "runtime_failed", "failed"),
        (MemoryInterventionExecutionStatus.CANCELLED, "runtime_cancelled", "cancelled"),
        (MemoryInterventionExecutionStatus.TIMED_OUT, "runtime_timed_out", "timed_out"),
        (
            MemoryInterventionExecutionStatus.OUTCOME_UNKNOWN,
            "runtime_outcome_unknown",
            "outcome_unknown",
        ),
        (
            MemoryInterventionExecutionStatus.CONFLICTING,
            "intervention_conflicting",
            "conflicting",
        ),
        (
            MemoryInterventionExecutionStatus.INDETERMINATE,
            "intervention_indeterminate",
            "indeterminate",
        ),
    ),
)
def test_terminal_trial_statuses_are_retained_without_survivor_filtering(
    status: MemoryInterventionExecutionStatus,
    failure_code: str,
    availability: str,
) -> None:
    request = asyncio.run(_report_request())
    selected = request.trials[1]
    document = selected.execution.model_dump(mode="python")
    document.update(
        {
            "phase": MemoryInterventionExecutionPhase.EVALUATED,
            "status": status,
            "snapshot_result_fingerprint": None,
            "final_binding_fingerprint": None,
            "failure_code": failure_code,
        }
    )
    terminal = selected.model_copy(
        update={
            "execution": MemoryInterventionExecutionRecord.model_validate(document),
            "intervention_binding": None,
            "published_result_revision": None,
            "accounting_side": None,
        }
    )
    trials = tuple(terminal if item is selected else item for item in request.trials)
    report = build_memory_experiment_report(
        MemoryExperimentReportRequest.model_validate(
            request.model_copy(update={"trials": trials}).model_dump(mode="python")
        )
    )

    row = next(item for item in report.rows if item.execution_id == terminal.execution.execution_id)
    assert row.availability.value == availability
    assert len(report.rows) == 4

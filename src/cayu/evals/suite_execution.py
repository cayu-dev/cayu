from __future__ import annotations

import hashlib
from pathlib import Path

from cayu.evals.corpus import (
    CorpusUserMessageSpec,
    EvalCaseSpec,
    EvalCorpusDocument,
    EvaluationSourceIdentityV1,
    MaxEstimatedCostAssertionSpec,
    RunInputSpec,
    eval_suite_trial_policy,
    pricing_profile_identity,
)
from cayu.evals.execution import CorpusTarget, WorkflowEvalTarget, evaluation_target_identity
from cayu.evals.scenario import EvalScenarioDocumentV2
from cayu.evals.scenario_preflight import ScenarioLaunchBindingV2, ScenarioLaunchSettingsV2
from cayu.evals.suite_authoring import (
    EvalCaseDefinition,
    EvalScenarioStimulusV1,
    EvalSimpleInputStimulusV1,
    EvalSuiteDocument,
    EvalSuiteSelectionV1,
    validate_eval_suite_selection,
    validate_expected_eval_suite_revision,
)

_AUTHORED_SOURCE_APPLICATION_RELEASE_ID = "cayu-authored-case-v1"
_AUTHORED_SOURCE_MANIFEST_SCHEMA_VERSION = "authored-eval-case-v1"


def authored_suite_launch_settings(document: EvalSuiteDocument) -> ScenarioLaunchSettingsV2:
    """Return the safe PR-02 launch bounds shared by authored scenario cases."""

    validated = validate_expected_eval_suite_revision(document, document.revision)
    policy = eval_suite_trial_policy(validated.suite)
    return ScenarioLaunchSettingsV2(
        trials=policy.trial_count,
        max_concurrency=policy.max_concurrency,
        timeout_seconds=validated.suite.trial_request.timeout_seconds,
    )


def corpus_for_authored_simple_selection(
    document: EvalSuiteDocument,
    selection: EvalSuiteSelectionV1,
    target: CorpusTarget,
    *,
    project_root: Path | None = None,
) -> EvalCorpusDocument:
    """Compile selected simple cases into the existing immutable corpus contract."""

    validated = validate_expected_eval_suite_revision(document, document.revision)
    selected = validate_eval_suite_selection(selection, validated)
    selected_ids = {item.id for item in selected.cases}
    cases = tuple(case for case in validated.cases if case.id in selected_ids)
    if not cases or any(type(case.stimulus) is not EvalSimpleInputStimulusV1 for case in cases):
        raise ValueError(
            "A simple authored-suite corpus requires only selected simple-input cases."
        )
    return _corpus_for_authored_cases(
        validated,
        cases,
        target,
        project_root=project_root,
    )


def corpus_for_authored_scenario_case(
    document: EvalSuiteDocument,
    case_id: str,
    scenario: EvalScenarioDocumentV2,
    binding: ScenarioLaunchBindingV2,
    target: CorpusTarget,
    *,
    project_root: Path | None = None,
) -> EvalCorpusDocument:
    """Compile one authored scenario case for the restart-safe scenario worker."""

    validated = validate_expected_eval_suite_revision(document, document.revision)
    authored_suite_launch_settings(validated)
    case = next((item for item in validated.cases if item.id == case_id), None)
    if case is None:
        raise KeyError(f"Authored eval case not found: {case_id}")
    stimulus = case.stimulus
    if type(stimulus) is not EvalScenarioStimulusV1:
        raise ValueError("The selected authored case is not a scenario case.")
    if type(scenario) is not EvalScenarioDocumentV2:
        raise TypeError("scenario must be an exact EvalScenarioDocumentV2.")
    if type(binding) is not ScenarioLaunchBindingV2:
        raise TypeError("binding must be an exact ScenarioLaunchBindingV2.")
    if type(target) not in {CorpusTarget, WorkflowEvalTarget}:
        raise TypeError("target must be an exact CorpusTarget or WorkflowEvalTarget.")
    target_identity = evaluation_target_identity(target, project_root=project_root)
    if (
        stimulus.scenario_id != scenario.id
        or stimulus.scenario_revision != scenario.revision
        or scenario.target_key != validated.target_key
        or binding.scenario_revision != scenario.revision
        or binding.target_key != validated.target_key
        or binding.application_release_id != target_identity.application_release_id
        or binding.app_manifest_fingerprint != target_identity.app_manifest_fingerprint
        or binding.trials != eval_suite_trial_policy(validated.suite).trial_count
        or binding.max_concurrency > eval_suite_trial_policy(validated.suite).max_concurrency
        or binding.timeout_seconds != validated.suite.trial_request.timeout_seconds
    ):
        raise ValueError("The authored scenario case does not match its current launch binding.")

    # The worker replaces this non-authoritative marker with the exact typed
    # scenario input. Keeping the ordinary corpus shape preserves the existing
    # evaluator, result, report, comparison, and baseline implementations.
    marker = RunInputSpec(
        messages=(CorpusUserMessageSpec(text=f"Execute controlled scenario {scenario.revision}."),)
    )
    return _corpus_for_authored_cases(
        validated,
        (case,),
        target,
        scenario_inputs={case.id: marker},
        project_root=project_root,
    )


def _corpus_for_authored_cases(
    document: EvalSuiteDocument,
    cases: tuple[EvalCaseDefinition, ...],
    target: CorpusTarget,
    *,
    scenario_inputs: dict[str, RunInputSpec] | None = None,
    project_root: Path | None,
) -> EvalCorpusDocument:
    if type(target) not in {CorpusTarget, WorkflowEvalTarget}:
        raise TypeError("target must be an exact CorpusTarget or WorkflowEvalTarget.")
    if document.target_key != target.key:
        raise ValueError("Authored eval suite target does not match the trusted target.")
    target_identity = evaluation_target_identity(target, project_root=project_root)
    if target_identity.target_key != document.target_key:
        raise ValueError("Authored eval suite no longer matches the current target identity.")
    uses_pricing = any(
        type(assertion) is MaxEstimatedCostAssertionSpec
        for case in cases
        for assertion in case.assertions
    )
    if uses_pricing and target.price_book is None:
        raise ValueError("Authored cost assertions require current server-owned pricing.")
    if uses_pricing:
        assert target.price_book is not None
        pricing = pricing_profile_identity(target.price_book)
    else:
        pricing = None
    compiled_cases: list[EvalCaseSpec] = []
    for case in cases:
        stimulus = case.stimulus
        if scenario_inputs is not None and case.id in scenario_inputs:
            case_input = scenario_inputs[case.id]
        elif type(stimulus) is EvalSimpleInputStimulusV1:
            case_input = stimulus.input
        else:
            raise ValueError("Scenario cases require an exact server-preflighted input binding.")
        source = case.source or _authored_definition_source(case)
        compiled_cases.append(
            EvalCaseSpec.create(
                id=case.id,
                suite_id=document.suite.id,
                name=case.name,
                description=case.description,
                source=source,
                input=case_input,
                assertions=case.assertions,
            )
        )
    return EvalCorpusDocument.create(
        target_key=document.target_key,
        evidence_policy=target.evidence_policy,
        pricing_profile=pricing,
        suites=(document.suite,),
        cases=tuple(compiled_cases),
    )


def _authored_definition_source(case: EvalCaseDefinition) -> EvaluationSourceIdentityV1:
    """Project stable definition provenance without pretending a production capture."""

    fingerprint = hashlib.sha256(
        b"cayu-authored-eval-case-source-v1\0" + case.revision.encode("ascii")
    ).hexdigest()
    return EvaluationSourceIdentityV1(
        application_release_id=_AUTHORED_SOURCE_APPLICATION_RELEASE_ID,
        app_manifest_schema_version=_AUTHORED_SOURCE_MANIFEST_SCHEMA_VERSION,
        app_manifest_fingerprint=fingerprint,
        evidence_revision=case.revision,
    )


__all__ = [
    "authored_suite_launch_settings",
    "corpus_for_authored_scenario_case",
    "corpus_for_authored_simple_selection",
]

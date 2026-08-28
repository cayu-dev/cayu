from __future__ import annotations

import asyncio
import json

import pytest
from pydantic import ValidationError
from tests.evals.test_structured_model_judge import _judge, _judgment, _rubric, _target

from cayu import ModelJudgeTarget
from cayu.evals.calibration import (
    EvalJudgeCalibrationCriterionLabelV1,
    EvalJudgeCalibrationDraftV1,
    EvalJudgeCalibrationReportV1,
    compile_eval_judge_calibration_draft,
    prepare_eval_judge_calibration,
    run_eval_judge_calibration_trial,
)
from cayu.evals.corpus import (
    CorpusUserMessageSpec,
    EvalJudgeEvidenceSelectionV1,
    RootStatusAssertionSpec,
    RunInputSpec,
    StructuredModelJudgeAssertionSpec,
)
from cayu.evals.execution import model_judge_profile
from cayu.evals.published import PublishedStructuredModelJudgeDetail
from cayu.evals.suite_authoring import (
    EvalCaseDraftV2,
    EvalSimpleInputStimulusV1,
    EvalSuiteDocumentV2,
    EvalSuiteDraftV1,
    EvalSuiteDraftV2,
    StructuredModelJudgeAssertionDraftV1,
    compile_eval_suite_authoring_draft,
    eval_suite_document_from_json,
    eval_suite_document_to_json,
)


def _calibration_draft(*, trials: int = 1) -> tuple[EvalJudgeCalibrationDraftV1, ModelJudgeTarget]:
    judge, _ = _judge(_judgment())
    profile = model_judge_profile(judge)
    return (
        EvalJudgeCalibrationDraftV1(
            id="known-refund-answer",
            target_key="refund-agent",
            assertion=StructuredModelJudgeAssertionSpec(
                id="answer-quality",
                judge_profile_key=profile.key,
                judge_profile_revision=profile.revision,
                rubric=_rubric(),
                threshold="0.6",
                evidence=EvalJudgeEvidenceSelectionV1(),
            ),
            evidence_source_id="reviewed-refund-fixture",
            task="Can I get a refund?",
            final_output="Refunds are available within 30 days.",
            human_criteria=(
                EvalJudgeCalibrationCriterionLabelV1(
                    criterion_id="correctness",
                    score="1",
                ),
                EvalJudgeCalibrationCriterionLabelV1(
                    criterion_id="usefulness",
                    score="0.5",
                ),
            ),
            trials=trials,
        ),
        judge,
    )


async def _calibration_report(
    *,
    run_id: str = "calibration-conformance",
) -> EvalJudgeCalibrationReportV1:
    draft, judge = _calibration_draft()
    definition = compile_eval_judge_calibration_draft(draft)
    prepared = prepare_eval_judge_calibration(definition, _target(judge)[0])
    trial = await run_eval_judge_calibration_trial(prepared, sequence=1)
    return EvalJudgeCalibrationReportV1.create(
        run_id=run_id,
        prepared=prepared,
        trials=(trial,),
    )


def test_suite_authoring_v2_round_trips_structured_judge_without_widening_v1() -> None:
    calibration, judge = _calibration_draft()
    draft = EvalSuiteDraftV2(
        id="quality-suite",
        target_key="refund-agent",
        name="Quality suite",
        cases=(
            EvalCaseDraftV2(
                id="refund-answer",
                name="Refund answer",
                stimulus=EvalSimpleInputStimulusV1(
                    input=RunInputSpec(
                        messages=(CorpusUserMessageSpec(text="Can I get a refund?"),)
                    )
                ),
                assertions=(
                    RootStatusAssertionSpec(id="completed", expected="completed"),
                    StructuredModelJudgeAssertionDraftV1.from_assertion(calibration.assertion),
                ),
            ),
        ),
    )

    document = compile_eval_suite_authoring_draft(draft)

    assert type(document) is EvalSuiteDocumentV2
    assert document.schema_version == 2
    assert document.cases[0].assertions[1] == calibration.assertion
    assert eval_suite_document_from_json(eval_suite_document_to_json(document)) == document
    assert "structured_model_judge" in json.dumps(EvalSuiteDraftV2.model_json_schema())
    assert "structured_model_judge" not in json.dumps(EvalSuiteDraftV1.model_json_schema())
    assert judge.key == calibration.assertion.judge_profile_key


def test_fixed_evidence_calibration_calls_only_the_judge_and_retains_human_comparison() -> None:
    draft, judge = _calibration_draft()
    definition = compile_eval_judge_calibration_draft(draft)
    target, candidate_provider = _target(judge)

    prepared = prepare_eval_judge_calibration(definition, target)
    trial = asyncio.run(run_eval_judge_calibration_trial(prepared, sequence=1))
    report = EvalJudgeCalibrationReportV1.create(
        run_id="calibration-test",
        prepared=prepared,
        trials=(trial,),
    )

    assert candidate_provider.requests == []
    assert trial.judgment.outcome == "passed"
    assert trial.judgment.detail.kind == "structured_model_judge"
    assert trial.aggregate_absolute_error == "0"
    assert trial.pass_agreement is True
    assert report.definition.evidence.revision.startswith("sha256:")
    assert report.definition.evidence.provenance.kind == "operator_supplied"
    assert report.definition.evidence.provenance.source_id == "reviewed-refund-fixture"
    assert report.trials == (trial,)


def test_calibration_retains_typed_evaluator_error_without_mislabeling_candidate_failure() -> None:
    draft, _ = _calibration_draft()
    judge, _ = _judge("not valid structured judgment")
    profile = model_judge_profile(judge)
    draft = draft.model_copy(
        update={
            "assertion": draft.assertion.model_copy(
                update={
                    "judge_profile_key": profile.key,
                    "judge_profile_revision": profile.revision,
                }
            )
        }
    )
    definition = compile_eval_judge_calibration_draft(draft)
    prepared = prepare_eval_judge_calibration(definition, _target(judge)[0])

    trial = asyncio.run(run_eval_judge_calibration_trial(prepared, sequence=1))

    assert trial.judgment.outcome == "error"
    detail = trial.judgment.detail
    assert type(detail) is PublishedStructuredModelJudgeDetail
    assert detail.diagnostic == "evaluator_error"
    assert detail.criteria == ()
    assert trial.aggregate_absolute_error is None
    assert trial.pass_agreement is None


def test_calibration_report_recomputes_trial_comparison_from_human_truth() -> None:
    draft, judge = _calibration_draft()
    definition = compile_eval_judge_calibration_draft(draft)
    prepared = prepare_eval_judge_calibration(definition, _target(judge)[0])
    trial = asyncio.run(run_eval_judge_calibration_trial(prepared, sequence=1))
    trial_with_wrong_human_truth = type(trial).create(
        sequence=1,
        judgment=trial.judgment,
        human_aggregate_score="0.25",
        threshold=definition.assertion.threshold,
    )

    with pytest.raises(ValueError, match="comparison does not match"):
        EvalJudgeCalibrationReportV1.create(
            run_id="forged-comparison",
            prepared=prepared,
            trials=(trial_with_wrong_human_truth,),
        )


def test_calibration_definition_rejects_changed_human_labels_and_forged_evidence() -> None:
    draft, _ = _calibration_draft()
    with pytest.raises(ValueError, match="criterion order"):
        compile_eval_judge_calibration_draft(
            draft.model_copy(
                update={
                    "human_criteria": tuple(reversed(draft.human_criteria)),
                }
            )
        )

    definition = compile_eval_judge_calibration_draft(draft)
    forged = definition.model_dump(mode="json")
    forged["evidence"]["final_output"] = "Changed after preview."
    with pytest.raises(ValidationError, match="evidence revision"):
        type(definition).model_validate(forged)

    changed_source = compile_eval_judge_calibration_draft(
        draft.model_copy(update={"evidence_source_id": "another-reviewed-fixture"})
    )
    assert changed_source.evidence.revision != definition.evidence.revision
    assert changed_source.revision != definition.revision

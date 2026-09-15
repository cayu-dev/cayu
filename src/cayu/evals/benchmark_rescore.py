"""New scorer receipts derived solely from retained, content-bound output facts."""

from pathlib import Path

from cayu.evals.benchmark_campaign import load_benchmark_campaign
from cayu.evals.benchmark_package import load_benchmark_package
from cayu.evals.corpus import EvaluationEvidencePolicySpec, _content_revision
from cayu.evals.evidence import AssertionEvidenceView
from cayu.evals.memory_attribution import (
    EvalMemoryAttributionEvidenceV1,
    EvalMemoryEvidenceLimitation,
)
from cayu.evals.portable_evaluation import evaluate_assertion_spec
from cayu.storage.evals_sqlite import SQLiteEvalStore
from cayu.storage.migrations import SchemaMode


def _saved_evidence(trial):
    output = trial.output
    complete = (
        output.evidence_state == "complete"
        and not output.preview_truncated
        and trial.execution_status != "failed"
        and trial.execution_failure_category is None
        and trial.failure_capture is None
    )
    statuses = {
        assertion.detail.actual
        for assertion in trial.assertions
        if assertion.detail.kind == "root_status" and assertion.detail.actual is not None
    }
    root_status = next(iter(statuses)) if len(statuses) == 1 else None
    material = {
        "schema_version": 5,
        "policy_revision": EvaluationEvidencePolicySpec.standard().revision,
        "pricing_profile_fingerprint": None,
        "root_evidence_available": complete or root_status is not None,
        "root_status": root_status,
        "child_statuses": [],
        "child_evidence_state": "unavailable",
        "final_output": output.text if complete else "",
        "final_output_state": "complete" if complete else "unavailable",
        "requested_tool_names": [],
        "started_tool_names": [],
        "tool_calls_started": None,
        "tool_evidence_state": "unavailable",
        "tool_calls": [],
        "tool_call_evidence_state": "unavailable",
        "process_events": [],
        "process_event_evidence_state": "unavailable",
        "workspace_files": [],
        "workspace_evidence_state": "unavailable",
        "artifacts": [],
        "artifact_scopes": [],
        "model_steps": None,
        "model_step_evidence_state": "unavailable",
        "total_tokens": None,
        "usage_evidence_state": "unavailable",
        "costs": [],
        "memory_attribution": EvalMemoryAttributionEvidenceV1.unavailable(
            EvalMemoryEvidenceLimitation.MISSING
        ).model_dump(mode="json"),
    }
    return AssertionEvidenceView.model_validate(
        {"revision": _content_revision(material, "assertion evidence"), **material}
    )


async def rescore_benchmark_campaign(directory, scorer_package):
    """Evaluate static assertions; missing facts stay unavailable, judges are rejected."""

    campaign = load_benchmark_campaign(directory)
    original = load_benchmark_package(Path(directory) / "package").package
    scorer = load_benchmark_package(scorer_package).package
    if original.revision != campaign.package_revision:
        raise ValueError("Saved package no longer matches campaign admission.")
    if (scorer.scorer_id, scorer.scorer_version) == (campaign.scorer_id, campaign.scorer_version):
        raise ValueError("Rescoring requires a new explicit scorer identity or version.")
    old_cases = {case.id: case for case in original.suite.cases}
    new_cases = {case.id: case for case in scorer.suite.cases}
    selected = {case.id for case in campaign.selection.cases}
    if not selected.issubset(new_cases) or any(
        old_cases[key].stimulus != new_cases[key].stimulus for key in selected
    ):
        raise ValueError("Rescoring must preserve every selected case's exact stimulus.")
    if original.scenarios != scorer.scenarios or original.files != scorer.files:
        raise ValueError("Rescoring must preserve exact scenario and file inputs.")
    if any(
        assertion.kind in {"model_judge", "structured_model_judge"}
        for key in selected
        for assertion in new_cases[key].assertions
    ):
        raise ValueError(
            "Saved-output rescoring does not authorize model judges; select static assertions."
        )
    rows = []
    store = SQLiteEvalStore(
        Path(directory) / "evals.sqlite3", read_only=True, schema_mode=SchemaMode.VALIDATE
    )
    try:
        for run in campaign.runs:
            result = await store.load_result(run.spec.id)
            if result is None:
                raise ValueError("Rescoring requires published original results for every run.")
            for case in result.run.cases:
                for trial in case.trials:
                    evidence = _saved_evidence(trial)
                    assertions = [
                        evaluate_assertion_spec(spec, evidence)
                        for spec in new_cases[case.case_id].assertions
                    ]
                    rows.append(
                        {
                            "run_id": run.spec.id,
                            "case_id": case.case_id,
                            "trial_number": trial.trial_number,
                            "source_trial_revision": trial.source_trial_revision,
                            "source_status": trial.status,
                            "source_execution_status": trial.execution_status,
                            "source_execution_failure_category": trial.execution_failure_category,
                            "source_output_sha256": trial.output.retained_sha256,
                            "evidence_revision": evidence.revision,
                            "assertions": [
                                {
                                    "id": item.name,
                                    "revision": item.assertion_revision,
                                    "outcome": item.outcome.value,
                                    "score": item.score,
                                }
                                for item in assertions
                            ],
                        }
                    )
    finally:
        await store.close()
    material = {
        "schema_version": 1,
        "source_campaign_revision": campaign.revision,
        "source_package_revision": campaign.package_revision,
        "scorer_package_revision": scorer.revision,
        "scorer_id": scorer.scorer_id,
        "scorer_version": scorer.scorer_version,
        "mode": "retained_output_static",
        "candidate_calls": 0,
        "judge_calls": 0,
        "limitations": [
            "Only complete retained redacted output and recorded root-status facts are available; other evidence is unavailable."
        ],
        "trials": rows,
    }
    return {"revision": _content_revision(material, "benchmark rescore"), **material}

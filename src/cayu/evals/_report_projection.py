"""Opt-in allowlisted evidence prepared while the workload redactor is still owned."""

from __future__ import annotations

from cayu.evals.models import EvalRun
from cayu.evals.reported import EvalReportedRun


def report_plan_app(plan):
    if plan.app is not None:
        return plan.app
    target = plan.workflow_target or plan.corpus_target
    return target.app


def direct_report_projection(run: EvalRun, *, redact_json) -> EvalReportedRun:
    # Do not serialize arbitrary metadata, tool trajectories or native handles.
    # The original typed result owner revalidates post-construction mutations.
    run = EvalRun.model_validate(run)
    document = {
        name: getattr(run, name)
        for name in ("suite_id", "status", "score", "started_at", "completed_at", "duration_ms")
    }
    document["cases"] = [
        {
            "case_id": case.case_id,
            "error": case.error,
            "unavailable_reason": case.unavailable_reason,
            "status": case.status,
            "score": case.score,
            "trial_policy": case.trial_policy,
            "trials": [
                {
                    **{
                        name: getattr(trial, name)
                        for name in (
                            "trial_number",
                            "status",
                            "score",
                            "final_output",
                            "structured_output",
                            "error",
                            "unavailable_reason",
                        )
                    },
                    "assertions": [
                        {
                            name: getattr(check, name)
                            for name in ("name", "outcome", "score", "threshold", "message")
                        }
                        for check in trial.assertions
                    ],
                }
                for trial in case.trials
            ],
        }
        for case in run.cases
    ]
    safe = EvalReportedRun.model_validate(document).model_dump(mode="json")
    return EvalReportedRun.model_validate(redact_json(safe))

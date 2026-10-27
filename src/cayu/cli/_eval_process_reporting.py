"""Worker-owned safe projections, separate from unchanged execution receipts."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass

from cayu.evals._inspection_documents import write_process_document
from cayu.evals._report_projection import direct_report_projection, report_plan_app
from cayu.evals.models import EvalRun
from cayu.evals.reported import EVAL_REPORT_MAX_BYTES, EvalReportedRun


@dataclass(frozen=True)
class ReportedProcessEvalRun:
    run: EvalRun
    report: EvalReportedRun | None


def save_worker_report(directory, index, launch, plan, run):
    if not launch.get("reporting", False):
        return
    try:
        app = report_plan_app(plan)
        raw = os.environ.get("CAYU_EVAL_REPORT_CONTEXT", "")
        if len(raw) > 32 << 10 or hashlib.sha256(raw.encode()).hexdigest() != launch.get(
            "reporting_context_hash"
        ):
            return
        context = json.loads(raw)
        if type(context) is not dict or app.redact_json(context) != context:
            return
        projection = direct_report_projection(run, redact_json=app.redact_json)
        write_process_document(
            directory / f"report-{index}.json",
            {
                "launch_id": launch["launch_id"],
                "index": index,
                "report": projection.model_dump(mode="json"),
            },
        )
    except Exception:
        # Missing safe projection means report failure, never evaluation failure.
        return


def combine_worker_reports(directory, assignments, launch_id, run):
    try:
        cases = {}
        for index, expected in enumerate(assignments):
            if not expected:
                continue
            with (directory / f"report-{index}.json").open("rb") as stream:
                raw = stream.read(EVAL_REPORT_MAX_BYTES + 1)
            if len(raw) > EVAL_REPORT_MAX_BYTES:
                return None
            document = json.loads(raw)
            if document.get("launch_id") != launch_id or document.get("index") != index:
                return None
            projection = EvalReportedRun.model_validate(document["report"])
            if (
                projection.suite_id != run.suite_id
                or [case.case_id for case in projection.cases] != expected
            ):
                return None
            for case in projection.cases:
                if case.case_id in cases:
                    return None
                cases[case.case_id] = case
        if set(cases) != {case.case_id for case in run.cases}:
            return None
        for original in run.cases:
            projected = cases[original.case_id]
            if (original.status, original.score) != (projected.status, projected.score):
                return None
        return EvalReportedRun(
            suite_id=run.suite_id,
            status=run.status,
            score=run.score,
            started_at=run.started_at,
            completed_at=run.completed_at,
            duration_ms=run.duration_ms,
            cases=tuple(cases[case.case_id] for case in run.cases),
        )
    except Exception:
        return None

from __future__ import annotations

import html
import json
from functools import partial
from pathlib import Path
from typing import Any, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, StrictFloat, field_validator

from cayu._validation import (
    copy_durable_json_object,
    durable_json_object_from_pairs,
    parse_durable_json_integer_literal,
    reject_nonportable_json_constant,
    require_clean_nonblank,
    require_durable_text,
)
from cayu.evals.models import (
    EVAL_SCHEMA_VERSION,
    TRAJECTORY_SCHEMA_VERSION,
    EvalOutcome,
    EvalRun,
    EvalStatus,
    Trajectory,
)
from cayu.runtime.usage import aggregate_usage_metrics_from_json_payload

_ModelT = TypeVar("_ModelT", bound=BaseModel)

_STATUS_SEVERITY = {
    EvalStatus.PASSED: 0,
    EvalStatus.SKIPPED: 1,
    EvalStatus.FAILED: 2,
    EvalStatus.UNAVAILABLE: 3,
    EvalStatus.ERROR: 4,
}


class _TrajectoryDocument(BaseModel):
    """Versioned persistence envelope for standalone trajectory exports."""

    model_config = ConfigDict(extra="forbid", strict=True)

    # Type checkers require the literal token here rather than the exported
    # TRAJECTORY_SCHEMA_VERSION constant.
    schema_version: Literal[1] = TRAJECTORY_SCHEMA_VERSION
    trajectory: Trajectory


class EvalCaseComparison(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str
    baseline_status: EvalStatus | None = None
    current_status: EvalStatus | None = None
    baseline_score: StrictFloat | None = None
    current_score: StrictFloat | None = None
    regressions: tuple[str, ...] = Field(default_factory=tuple)

    @field_validator("case_id")
    @classmethod
    def validate_case_id(cls, value: str, info) -> str:
        return require_clean_nonblank(value, info.field_name)


class EvalRunComparison(BaseModel):
    model_config = ConfigDict(extra="forbid")

    baseline_run_id: str
    current_run_id: str
    baseline_suite_id: str
    current_suite_id: str
    baseline_status: EvalStatus
    current_status: EvalStatus
    baseline_score: StrictFloat | None = None
    current_score: StrictFloat | None = None
    regressions: tuple[str, ...] = Field(default_factory=tuple)
    cases: tuple[EvalCaseComparison, ...] = Field(default_factory=tuple)

    @field_validator("baseline_run_id", "current_run_id", "baseline_suite_id", "current_suite_id")
    @classmethod
    def validate_ids(cls, value: str, info) -> str:
        return require_clean_nonblank(value, info.field_name)


def _validated_durable_model_document(
    model: _ModelT,
    *,
    model_type: type[_ModelT],
    field_name: str,
) -> tuple[_ModelT, dict[str, Any]]:
    """Defensively revalidate one public model before crossing an export boundary."""

    # A caller can forge an instance with model_copy(update=...) or model_construct().
    # Suppress serializer warnings from that untrusted state; validation below provides
    # the deterministic typed failure before any durable or external write.
    validated = model_type.model_validate(model.model_dump(mode="python", warnings=False))
    document = copy_durable_json_object(validated.model_dump(mode="json"), field_name)
    return validated, document


def _model_to_json(
    model: _ModelT,
    *,
    model_type: type[_ModelT],
    field_name: str,
    indent: int | None,
) -> str:
    _, document = _validated_durable_model_document(
        model,
        model_type=model_type,
        field_name=field_name,
    )
    return json.dumps(document, indent=indent, sort_keys=True) + "\n"


def _load_durable_json_document(source: str, *, field_name: str) -> dict[str, Any]:
    decoded = json.loads(
        source,
        parse_int=partial(
            parse_durable_json_integer_literal,
            field_name=field_name,
        ),
        parse_constant=partial(
            reject_nonportable_json_constant,
            field_name=field_name,
        ),
        object_pairs_hook=partial(
            durable_json_object_from_pairs,
            field_name=field_name,
        ),
    )
    return copy_durable_json_object(decoded, field_name)


def eval_run_to_json(run: EvalRun, *, indent: int | None = 2) -> str:
    if type(run) is not EvalRun:
        raise TypeError("eval_run_to_json requires an EvalRun.")
    return _model_to_json(
        run,
        model_type=EvalRun,
        field_name="eval run",
        indent=indent,
    )


def load_eval_run(path: str | Path) -> EvalRun:
    source = Path(path).read_text(encoding="utf-8")
    data = _load_durable_json_document(source, field_name="eval run file")
    if "schema_version" not in data:
        raise ValueError(
            "Eval run file has no schema_version and is not supported; "
            "regenerate it with this cayu version."
        )
    raw = data["schema_version"]
    if type(raw) is not int or raw != EVAL_SCHEMA_VERSION:
        raise ValueError(
            f"Eval run has unsupported schema_version {raw!r}; this cayu supports "
            f"only {EVAL_SCHEMA_VERSION}. Upgrade cayu or regenerate the run."
        )
    return EvalRun.model_validate(data)


def write_eval_run_json(run: EvalRun, path: str | Path) -> None:
    Path(path).write_text(eval_run_to_json(run), encoding="utf-8")


def trajectory_to_json(trajectory: Trajectory, *, indent: int | None = 2) -> str:
    if type(trajectory) is not Trajectory:
        raise TypeError("trajectory_to_json requires a Trajectory.")
    validated, _ = _validated_durable_model_document(
        trajectory,
        model_type=Trajectory,
        field_name="trajectory",
    )
    document = _TrajectoryDocument(trajectory=validated).model_dump(mode="json")
    durable_document = copy_durable_json_object(document, "trajectory")
    return (
        json.dumps(
            durable_document,
            indent=indent,
            sort_keys=True,
        )
        + "\n"
    )


def _load_trajectory_document(source: str) -> dict[str, Any]:
    return _load_durable_json_document(source, field_name="trajectory file")


def _trajectory_document_python_input(data: dict[str, Any]) -> dict[str, Any]:
    projected = dict(data)
    trajectory = projected.get("trajectory")
    if type(trajectory) is dict:
        projected["trajectory"] = _trajectory_python_input(trajectory)
    return projected


def _trajectory_python_input(data: dict[str, Any]) -> dict[str, Any]:
    projected = dict(data)
    usage_summary = projected.get("usage_summary")
    if type(usage_summary) is dict:
        projected_summary = dict(usage_summary)
        if "usage" in projected_summary:
            projected_summary["usage"] = aggregate_usage_metrics_from_json_payload(
                projected_summary["usage"]
            )
        projected["usage_summary"] = projected_summary
    children = projected.get("children")
    if type(children) is list:
        projected["children"] = [
            _trajectory_python_input(child) if type(child) is dict else child for child in children
        ]
    return projected


def write_trajectory_json(trajectory: Trajectory, path: str | Path) -> None:
    Path(path).write_text(trajectory_to_json(trajectory), encoding="utf-8")


def load_trajectory(path: str | Path) -> Trajectory:
    source = Path(path).read_text(encoding="utf-8")
    data = _load_trajectory_document(source)
    if "schema_version" not in data:
        raise ValueError(
            "Trajectory file has no schema_version and is not supported; "
            "regenerate it with this cayu version."
        )
    raw = data["schema_version"]
    if type(raw) is not int or raw != TRAJECTORY_SCHEMA_VERSION:
        raise ValueError(
            f"Trajectory has unsupported schema_version {raw!r}; this cayu supports "
            f"only {TRAJECTORY_SCHEMA_VERSION}. Upgrade cayu or regenerate the trajectory."
        )
    return _TrajectoryDocument.model_validate(_trajectory_document_python_input(data)).trajectory


def render_html_report(run: EvalRun) -> str:
    if type(run) is not EvalRun:
        raise TypeError("render_html_report requires an EvalRun.")
    run, _ = _validated_durable_model_document(
        run,
        model_type=EvalRun,
        field_name="eval report",
    )
    rows = "\n".join(_case_row(case) for case in run.cases)
    assertions = "\n".join(_assertion_section(case) for case in run.cases)
    rendered = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Cayu Eval Report - {_escape(run.suite_id)}</title>
  <style>
    body {{
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: #18211d;
      background: #f7f7f4;
      line-height: 1.5;
    }}
    .page {{ max-width: 1120px; margin: 0 auto; padding: 32px 24px 56px; }}
    h1 {{ margin: 0 0 8px; font-size: 2rem; }}
    h2 {{ margin: 28px 0 12px; font-size: 1.25rem; }}
    p {{ margin: 0; color: #5d6864; }}
    .metrics {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; margin: 24px 0; }}
    .metric, table, .case {{ background: #fff; border: 1px solid #d9dfdc; border-radius: 8px; box-shadow: 0 8px 24px rgba(23, 32, 29, 0.08); }}
    .metric {{ padding: 14px; }}
    .metric strong {{ display: block; font-size: 1.35rem; }}
    table {{ width: 100%; border-collapse: collapse; overflow: hidden; }}
    th, td {{ padding: 10px 12px; border-bottom: 1px solid #d9dfdc; text-align: left; vertical-align: top; }}
    th {{ background: #f0f4f3; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.04em; }}
    tr:last-child td {{ border-bottom: 0; }}
    .badge {{ display: inline-block; padding: 3px 8px; border-radius: 999px; font-size: 0.8rem; font-weight: 700; }}
    .passed {{ color: #0f5132; background: #d9f2e3; }}
    .failed {{ color: #842029; background: #f8d7da; }}
    .unavailable {{ color: #553c00; background: #fff0b3; }}
    .error {{ color: #664d03; background: #fff3cd; }}
    .skipped {{ color: #41505b; background: #e2e8ef; }}
    .case {{ padding: 16px; margin-top: 12px; }}
    .assertion {{ display: grid; grid-template-columns: 120px 1fr; gap: 12px; padding: 8px 0; border-top: 1px solid #e6ebe8; }}
    pre {{ white-space: pre-wrap; background: #f0f4f3; padding: 12px; border-radius: 6px; overflow: auto; }}
    @media (max-width: 760px) {{ .metrics {{ grid-template-columns: 1fr; }} .assertion {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
  <main class="page">
    <h1>Cayu Eval Report</h1>
    <p>Suite <code>{_escape(run.suite_id)}</code> run <code>{_escape(run.run_id)}</code></p>
    <div class="metrics">
      <div class="metric"><strong>{_status_badge(run.status)}</strong><span>Status</span></div>
      <div class="metric"><strong>{_format_score(run.score)}</strong><span>Score</span></div>
      <div class="metric"><strong>{len(run.cases)}</strong><span>Cases</span></div>
      <div class="metric"><strong>{run.duration_ms} ms</strong><span>Duration</span></div>
    </div>
    <h2>Cases</h2>
    <table>
      <thead>
        <tr><th>Case</th><th>Status</th><th>Score</th><th>Trials</th><th>Assertions</th><th>Diagnostic</th></tr>
      </thead>
      <tbody>
        {rows}
      </tbody>
    </table>
    <h2>Assertion Details</h2>
    {assertions}
  </main>
</body>
</html>
"""
    return require_durable_text(rendered, "eval report")


def write_html_report(run: EvalRun, path: str | Path) -> None:
    Path(path).write_text(render_html_report(run), encoding="utf-8")


def compare_eval_runs(
    baseline: EvalRun, current: EvalRun, *, score_tolerance: float = 0.0
) -> EvalRunComparison:
    """Compare two runs and flag regressions.

    `score_tolerance` (>= 0) is the amount a score may drop before it counts as a regression:
    a current score below ``baseline - score_tolerance`` regresses, so a stochastic wobble
    (e.g. 0.83 -> 0.82) inside the tolerance no longer fails a baseline comparison every run.
    A move to a more severe status is always flagged regardless of tolerance.
    """
    if type(score_tolerance) not in (int, float) or isinstance(score_tolerance, bool):
        raise TypeError("compare_eval_runs score_tolerance must be a number.")
    if score_tolerance != score_tolerance or score_tolerance < 0:  # rejects NaN and negatives
        raise ValueError("compare_eval_runs score_tolerance must be >= 0.")
    if baseline.suite_id != current.suite_id:
        raise ValueError(
            "Cannot compare eval runs from different suites: "
            f"{baseline.suite_id!r} != {current.suite_id!r}."
        )
    baseline_by_case = {case.case_id: case for case in baseline.cases}
    current_by_case = {case.case_id: case for case in current.cases}
    case_ids = sorted(set(baseline_by_case) | set(current_by_case))
    comparisons: list[EvalCaseComparison] = []
    regressions: list[str] = []

    for case_id in case_ids:
        base = baseline_by_case.get(case_id)
        cur = current_by_case.get(case_id)
        if base is None:
            # A newly-added case is not a regression; note it for the report only,
            # and do not let it fail the comparison's exit code.
            comparisons.append(
                EvalCaseComparison(
                    case_id=case_id,
                    current_status=cur.status if cur is not None else None,
                    current_score=cur.score if cur is not None else None,
                    regressions=("case added in current run",),
                )
            )
            continue
        case_regressions: list[str] = []
        if cur is None:
            case_regressions.append("case missing from current run")
        else:
            if _STATUS_SEVERITY[cur.status] > _STATUS_SEVERITY[base.status]:
                case_regressions.append(
                    f"status regressed from {base.status.value} to {cur.status.value}"
                )
            if (
                cur.score is not None
                and base.score is not None
                and cur.score < base.score - score_tolerance
            ):
                case_regressions.append(f"score regressed from {base.score:.2f} to {cur.score:.2f}")
        for item in case_regressions:
            regressions.append(f"{case_id}: {item}")
        comparisons.append(
            EvalCaseComparison(
                case_id=case_id,
                baseline_status=base.status,
                current_status=cur.status if cur is not None else None,
                baseline_score=base.score,
                current_score=cur.score if cur is not None else None,
                regressions=tuple(case_regressions),
            )
        )

    if _STATUS_SEVERITY[current.status] > _STATUS_SEVERITY[baseline.status]:
        regressions.insert(
            0,
            f"run status regressed from {baseline.status.value} to {current.status.value}",
        )
    if (
        current.score is not None
        and baseline.score is not None
        and current.score < baseline.score - score_tolerance
    ):
        regressions.insert(
            0, f"run score regressed from {baseline.score:.2f} to {current.score:.2f}"
        )

    return EvalRunComparison(
        baseline_run_id=baseline.run_id,
        current_run_id=current.run_id,
        baseline_suite_id=baseline.suite_id,
        current_suite_id=current.suite_id,
        baseline_status=baseline.status,
        current_status=current.status,
        baseline_score=baseline.score,
        current_score=current.score,
        regressions=tuple(regressions),
        cases=tuple(comparisons),
    )


def render_comparison_html(comparison: EvalRunComparison) -> str:
    if type(comparison) is not EvalRunComparison:
        raise TypeError("render_comparison_html requires an EvalRunComparison.")
    comparison, _ = _validated_durable_model_document(
        comparison,
        model_type=EvalRunComparison,
        field_name="eval comparison",
    )
    rows = "\n".join(
        "<tr>"
        f"<td>{_escape(case.case_id)}</td>"
        f"<td>{_escape(case.baseline_status.value if case.baseline_status else 'missing')}</td>"
        f"<td>{_escape(case.current_status.value if case.current_status else 'missing')}</td>"
        f"<td>{'' if case.baseline_score is None else f'{case.baseline_score:.2f}'}</td>"
        f"<td>{'' if case.current_score is None else f'{case.current_score:.2f}'}</td>"
        f"<td>{_escape('; '.join(case.regressions))}</td>"
        "</tr>"
        for case in comparison.cases
    )
    rendered = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Cayu Eval Comparison</title>
  <style>
    body {{ margin: 0; font-family: Inter, ui-sans-serif, system-ui, sans-serif; background: #f7f7f4; color: #18211d; }}
    .page {{ max-width: 1120px; margin: 0 auto; padding: 32px 24px 56px; }}
    table {{ width: 100%; border-collapse: collapse; background: #fff; border: 1px solid #d9dfdc; border-radius: 8px; overflow: hidden; }}
    th, td {{ padding: 10px 12px; border-bottom: 1px solid #d9dfdc; text-align: left; vertical-align: top; }}
    th {{ background: #f0f4f3; }}
    .regressions {{ background: #fff3cd; border: 1px solid #ffe69c; padding: 12px; border-radius: 8px; }}
  </style>
</head>
<body>
  <main class="page">
    <h1>Cayu Eval Comparison</h1>
    <p>Baseline <code>{_escape(comparison.baseline_run_id)}</code> vs current <code>{_escape(comparison.current_run_id)}</code></p>
    <p>Score: {_format_score(comparison.baseline_score)} -&gt; {_format_score(comparison.current_score)}</p>
    <h2>Regressions</h2>
    <div class="regressions">{_escape("; ".join(comparison.regressions) or "No regressions detected.")}</div>
    <h2>Cases</h2>
    <table>
      <thead><tr><th>Case</th><th>Baseline</th><th>Current</th><th>Baseline Score</th><th>Current Score</th><th>Regressions</th></tr></thead>
      <tbody>{rows}</tbody>
    </table>
  </main>
</body>
</html>
"""
    return require_durable_text(rendered, "eval comparison")


def comparison_to_json(comparison: EvalRunComparison, *, indent: int | None = 2) -> str:
    if type(comparison) is not EvalRunComparison:
        raise TypeError("comparison_to_json requires an EvalRunComparison.")
    return _model_to_json(
        comparison,
        model_type=EvalRunComparison,
        field_name="eval comparison",
        indent=indent,
    )


def _case_row(case: Any) -> str:
    passed = sum(1 for assertion in case.assertions if assertion.passed)
    total = len(case.assertions)
    return (
        "<tr>"
        f"<td>{_escape(case.case_id)}</td>"
        f"<td>{_status_badge(case.status)}</td>"
        f"<td>{_format_score(case.score)}</td>"
        f"<td>{len(case.trials)}</td>"
        f"<td>{passed}/{total}</td>"
        f"<td>{_escape(case.error or case.unavailable_reason or '')}</td>"
        "</tr>"
    )


def _assertion_section(case: Any) -> str:
    assertions = _assertion_rows(case.assertions)
    if not assertions:
        assertions = "<p>No assertions.</p>"
    trials = "\n".join(_trial_section(trial) for trial in case.trials)
    return (
        f'<section class="case"><h3>{_escape(case.case_id)}</h3>'
        f"<h4>Aggregate assertions</h4>{assertions}<h4>Trials</h4>{trials}</section>"
    )


def _assertion_rows(assertions: Any) -> str:
    return "\n".join(
        '<div class="assertion">'
        f"<div>{_status_badge(assertion.outcome)}</div>"
        f"<div><strong>{_escape(assertion.name)}</strong><p>{_escape(assertion.message)}</p>"
        f"<pre>{_escape(json.dumps(_assertion_details(assertion), indent=2, sort_keys=True))}</pre></div>"
        "</div>"
        for assertion in assertions
    )


def _assertion_details(assertion: Any) -> dict[str, Any]:
    details = dict(assertion.metadata)
    if assertion.cost_summary is not None:
        details["cost_summary"] = assertion.cost_summary.model_dump(mode="json")
    return details


def _trial_section(trial: Any) -> str:
    assertions = _assertion_rows(trial.assertions) or "<p>No assertions.</p>"
    final_output = (
        f"<h5>Final output</h5><pre>{_escape(trial.final_output)}</pre>"
        if trial.final_output
        else ""
    )
    diagnostic = trial.error or trial.unavailable_reason
    diagnostic_html = f"<h5>Diagnostic</h5><pre>{_escape(diagnostic)}</pre>" if diagnostic else ""
    usage = (
        f"<h5>Usage</h5><pre>{_escape(json.dumps(trial.usage_summary, indent=2, sort_keys=True))}</pre>"
        if trial.usage_summary is not None
        else ""
    )
    return (
        '<div class="case">'
        f"<h4>Trial {trial.trial_number} · {_status_badge(trial.status)}</h4>"
        f"<p>Session <code>{_escape(trial.session_id or 'not created')}</code> · "
        f"score {_format_score(trial.score)} · {trial.duration_ms} ms · "
        f"evidence {'complete' if trial.evidence_complete else 'incomplete'}</p>"
        f"{diagnostic_html}{final_output}{usage}<h5>Assertions</h5>{assertions}</div>"
    )


def _status_badge(status: EvalStatus | EvalOutcome) -> str:
    return f'<span class="badge {status.value}">{status.value}</span>'


def _format_score(score: float | None) -> str:
    return "unavailable" if score is None else f"{score:.2f}"


def _escape(value: Any) -> str:
    return html.escape(str(value), quote=True)

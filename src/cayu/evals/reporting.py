from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, StrictFloat, field_validator

from cayu._validation import require_clean_nonblank
from cayu.evals.models import EVAL_SCHEMA_VERSION, EvalRun, EvalStatus, Trajectory


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
    baseline_score: StrictFloat
    current_score: StrictFloat
    regressions: tuple[str, ...] = Field(default_factory=tuple)
    cases: tuple[EvalCaseComparison, ...] = Field(default_factory=tuple)

    @field_validator("baseline_run_id", "current_run_id", "baseline_suite_id", "current_suite_id")
    @classmethod
    def validate_ids(cls, value: str, info) -> str:
        return require_clean_nonblank(value, info.field_name)


def _model_to_json(model: BaseModel, *, indent: int | None) -> str:
    return json.dumps(model.model_dump(mode="json"), indent=indent, sort_keys=True) + "\n"


def eval_run_to_json(run: EvalRun, *, indent: int | None = 2) -> str:
    if type(run) is not EvalRun:
        raise TypeError("eval_run_to_json requires an EvalRun.")
    return _model_to_json(run, indent=indent)


def load_eval_run(path: str | Path) -> EvalRun:
    with Path(path).open("r", encoding="utf-8") as file:
        data = json.load(file)
    raw = data.get("schema_version") if isinstance(data, dict) else None
    if raw is not None and (type(raw) is not int or raw < 1 or raw > EVAL_SCHEMA_VERSION):
        raise ValueError(
            f"Eval run has unsupported schema_version {raw!r}; this cayu supports "
            f"1..{EVAL_SCHEMA_VERSION}. Upgrade cayu or regenerate the run."
        )
    return EvalRun.model_validate(data)


def write_eval_run_json(run: EvalRun, path: str | Path) -> None:
    Path(path).write_text(eval_run_to_json(run), encoding="utf-8")


def trajectory_to_json(trajectory: Trajectory, *, indent: int | None = 2) -> str:
    if type(trajectory) is not Trajectory:
        raise TypeError("trajectory_to_json requires a Trajectory.")
    return _model_to_json(trajectory, indent=indent)


def write_trajectory_json(trajectory: Trajectory, path: str | Path) -> None:
    Path(path).write_text(trajectory_to_json(trajectory), encoding="utf-8")


def load_trajectory(path: str | Path) -> Trajectory:
    # The opt-in replay/export object; unlike EvalRun it carries no persisted baseline
    # schema_version (it is not folded into the score-first baseline in v1).
    with Path(path).open("r", encoding="utf-8") as file:
        return Trajectory.model_validate(json.load(file))


def render_html_report(run: EvalRun) -> str:
    if type(run) is not EvalRun:
        raise TypeError("render_html_report requires an EvalRun.")
    rows = "\n".join(_case_row(case) for case in run.cases)
    assertions = "\n".join(_assertion_section(case) for case in run.cases)
    return f"""<!doctype html>
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
      <div class="metric"><strong>{run.score:.2f}</strong><span>Score</span></div>
      <div class="metric"><strong>{len(run.cases)}</strong><span>Cases</span></div>
      <div class="metric"><strong>{run.duration_ms} ms</strong><span>Duration</span></div>
    </div>
    <h2>Cases</h2>
    <table>
      <thead>
        <tr><th>Case</th><th>Status</th><th>Score</th><th>Session</th><th>Assertions</th><th>Error</th></tr>
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


def write_html_report(run: EvalRun, path: str | Path) -> None:
    Path(path).write_text(render_html_report(run), encoding="utf-8")


def compare_eval_runs(
    baseline: EvalRun, current: EvalRun, *, score_tolerance: float = 0.0
) -> EvalRunComparison:
    """Compare two runs and flag regressions.

    `score_tolerance` (>= 0) is the amount a score may drop before it counts as a regression:
    a current score below ``baseline - score_tolerance`` regresses, so a stochastic wobble
    (e.g. 0.83 -> 0.82) inside the tolerance no longer fails a baseline comparison every run.
    A status regression (PASSED -> not PASSED) is always flagged regardless of tolerance.
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
            if base.status == EvalStatus.PASSED and cur.status != EvalStatus.PASSED:
                case_regressions.append(
                    f"status regressed from {base.status.value} to {cur.status.value}"
                )
            if cur.score < base.score - score_tolerance:
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

    if baseline.status == EvalStatus.PASSED and current.status != EvalStatus.PASSED:
        regressions.insert(
            0,
            f"run status regressed from {baseline.status.value} to {current.status.value}",
        )
    if current.score < baseline.score - score_tolerance:
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
    return f"""<!doctype html>
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
    <p>Score: {comparison.baseline_score:.2f} -> {comparison.current_score:.2f}</p>
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


def comparison_to_json(comparison: EvalRunComparison, *, indent: int | None = 2) -> str:
    return json.dumps(comparison.model_dump(mode="json"), indent=indent, sort_keys=True) + "\n"


def _case_row(case: Any) -> str:
    passed = sum(1 for assertion in case.assertions if assertion.passed)
    total = len(case.assertions)
    return (
        "<tr>"
        f"<td>{_escape(case.case_id)}</td>"
        f"<td>{_status_badge(case.status)}</td>"
        f"<td>{case.score:.2f}</td>"
        f"<td>{_escape(case.session_id or '')}</td>"
        f"<td>{passed}/{total}</td>"
        f"<td>{_escape(case.error or '')}</td>"
        "</tr>"
    )


def _assertion_section(case: Any) -> str:
    assertions = "\n".join(
        '<div class="assertion">'
        f"<div>{_status_badge(EvalStatus.PASSED if assertion.passed else EvalStatus.FAILED)}</div>"
        f"<div><strong>{_escape(assertion.name)}</strong><p>{_escape(assertion.message)}</p>"
        f"<pre>{_escape(json.dumps(assertion.metadata, indent=2, sort_keys=True))}</pre></div>"
        "</div>"
        for assertion in case.assertions
    )
    if not assertions:
        assertions = "<p>No assertions.</p>"
    final_output = (
        f"<h4>Final output</h4><pre>{_escape(case.final_output)}</pre>" if case.final_output else ""
    )
    return (
        f'<section class="case"><h3>{_escape(case.case_id)}</h3>'
        f"{final_output}{assertions}</section>"
    )


def _status_badge(status: EvalStatus) -> str:
    return f'<span class="badge {status.value}">{status.value}</span>'


def _escape(value: Any) -> str:
    return html.escape(str(value), quote=True)

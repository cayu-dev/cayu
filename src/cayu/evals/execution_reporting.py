from __future__ import annotations

import html
import json
from functools import partial
from pathlib import Path
from typing import Any

from cayu._validation import (
    copy_durable_json_object,
    durable_json_object_from_pairs,
    parse_durable_json_integer_literal,
    reject_nonportable_json_constant,
    require_durable_text,
)
from cayu.evals.execution import (
    CORPUS_EXECUTION_RESULT_SCHEMA_VERSION,
    CorpusExecutionResult,
)
from cayu.evals.execution_comparison import (
    CORPUS_EXECUTION_COMPARISON_MAX_BYTES,
    CorpusExecutionComparison,
    CorpusExecutionRegression,
    CorpusRegressionKind,
    CorpusReliabilityDistributionV1,
    EvalStructuredJudgeComparisonV1,
    EvalToolJsonAssertionComparisonV1,
)
from cayu.evals.memory_attribution import eval_memory_attribution_summary
from cayu.evals.result_presentation import (
    EVAL_RESULT_REPORT_MAX_BYTES,
    EvalAssertionPresentationV1,
    EvalCasePresentationV2,
    EvalResultOutcomeDimensionsV1,
    EvalResultReportV2,
    EvalStructuredJudgePresentationV1,
    EvalTrialPresentationV1,
    present_eval_result,
)
from cayu.evals.results import CapturedEvaluationResultV1
from cayu.evals.trial_policy import EvalMaximumCostExposureV1

CORPUS_EXECUTION_RESULT_MAX_JSON_BYTES = 48 << 20
CORPUS_EXECUTION_RESULT_MAX_HTML_BYTES = 48 << 20
CORPUS_EXECUTION_COMPARISON_MAX_JSON_BYTES = CORPUS_EXECUTION_COMPARISON_MAX_BYTES
CORPUS_EXECUTION_COMPARISON_MAX_HTML_BYTES = CORPUS_EXECUTION_COMPARISON_MAX_BYTES


def _validated_result(result: CorpusExecutionResult) -> CorpusExecutionResult:
    if type(result) is not CorpusExecutionResult:
        raise TypeError("result must be an exact CorpusExecutionResult.")
    return CorpusExecutionResult.model_validate(
        result.model_dump(mode="python", round_trip=True, warnings="none")
    )


def corpus_execution_result_to_json(result: CorpusExecutionResult) -> str:
    """Return bounded deterministic JSON containing only the published result graph."""

    validated = _validated_result(result)
    document = copy_durable_json_object(
        validated.model_dump(mode="json"),
        "corpus execution result",
    )
    encoder = json.JSONEncoder(ensure_ascii=False, indent=2, sort_keys=True)
    chunks: list[str] = []
    total_bytes = 1
    for chunk in encoder.iterencode(document):
        total_bytes += len(chunk.encode("utf-8"))
        if total_bytes > CORPUS_EXECUTION_RESULT_MAX_JSON_BYTES:
            raise ValueError(
                "Corpus execution result JSON exceeds "
                f"{CORPUS_EXECUTION_RESULT_MAX_JSON_BYTES} bytes."
            )
        chunks.append(chunk)
    return "".join(chunks) + "\n"


def captured_evaluation_result_to_json(result: CapturedEvaluationResultV1) -> str:
    """Return bounded deterministic JSON for one immutable captured result."""

    if type(result) is not CapturedEvaluationResultV1:
        raise TypeError("result must be an exact CapturedEvaluationResultV1.")
    validated = CapturedEvaluationResultV1.model_validate(
        result.model_dump(mode="python", round_trip=True, warnings="none")
    )
    document = copy_durable_json_object(
        validated.model_dump(mode="json"),
        "captured evaluation result",
    )
    encoder = json.JSONEncoder(ensure_ascii=False, indent=2, sort_keys=True)
    chunks: list[str] = []
    total_bytes = 1
    for chunk in encoder.iterencode(document):
        total_bytes += len(chunk.encode("utf-8"))
        if total_bytes > CORPUS_EXECUTION_RESULT_MAX_JSON_BYTES:
            raise ValueError(
                "Captured evaluation result JSON exceeds "
                f"{CORPUS_EXECUTION_RESULT_MAX_JSON_BYTES} bytes."
            )
        chunks.append(chunk)
    return "".join(chunks) + "\n"


def eval_result_report_to_json(
    result: CorpusExecutionResult | CapturedEvaluationResultV1,
) -> str:
    """Return a bounded report containing immutable source and canonical presentation."""

    if type(result) not in {CorpusExecutionResult, CapturedEvaluationResultV1}:
        raise TypeError(
            "result must be an exact CorpusExecutionResult or CapturedEvaluationResultV1."
        )
    report = EvalResultReportV2(
        result=result,
        presentation=present_eval_result(result),
    )
    document = copy_durable_json_object(
        report.model_dump(mode="json"),
        "eval result report",
    )
    encoder = json.JSONEncoder(ensure_ascii=False, indent=2, sort_keys=True)
    chunks: list[str] = []
    total_bytes = 1
    for chunk in encoder.iterencode(document):
        total_bytes += len(chunk.encode("utf-8"))
        if total_bytes > EVAL_RESULT_REPORT_MAX_BYTES:
            raise ValueError(
                f"Eval result report JSON exceeds {EVAL_RESULT_REPORT_MAX_BYTES} bytes."
            )
        chunks.append(chunk)
    return "".join(chunks) + "\n"


def eval_result_to_json(result: CorpusExecutionResult | CapturedEvaluationResultV1) -> str:
    """Serialize either immutable eval-result origin without changing its graph."""

    if type(result) is CorpusExecutionResult:
        return corpus_execution_result_to_json(result)
    if type(result) is CapturedEvaluationResultV1:
        return captured_evaluation_result_to_json(result)
    raise TypeError("result must be an exact CorpusExecutionResult or CapturedEvaluationResultV1.")


def corpus_execution_comparison_to_json(comparison: CorpusExecutionComparison) -> str:
    """Return bounded deterministic JSON for one contract-aware comparison."""

    if type(comparison) is not CorpusExecutionComparison:
        raise TypeError("comparison must be an exact CorpusExecutionComparison.")
    validated = CorpusExecutionComparison.model_validate(
        comparison.model_dump(mode="python", round_trip=True, warnings="none")
    )
    document = copy_durable_json_object(
        validated.model_dump(mode="json"),
        "corpus execution comparison",
    )
    encoder = json.JSONEncoder(ensure_ascii=False, indent=2, sort_keys=True)
    chunks: list[str] = []
    total_bytes = 1
    for chunk in encoder.iterencode(document):
        total_bytes += len(chunk.encode("utf-8"))
        if total_bytes > CORPUS_EXECUTION_COMPARISON_MAX_JSON_BYTES:
            raise ValueError(
                "Corpus execution comparison JSON exceeds "
                f"{CORPUS_EXECUTION_COMPARISON_MAX_JSON_BYTES} bytes."
            )
        chunks.append(chunk)
    return "".join(chunks) + "\n"


def corpus_execution_result_from_json(source: str) -> CorpusExecutionResult:
    """Parse one bounded result document without legacy format guessing."""

    if type(source) is not str:
        raise TypeError("corpus_execution_result_from_json requires text.")
    if len(source) > CORPUS_EXECUTION_RESULT_MAX_JSON_BYTES:
        raise ValueError(
            f"Corpus execution result JSON exceeds {CORPUS_EXECUTION_RESULT_MAX_JSON_BYTES} bytes."
        )
    try:
        raw = source.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError(
            "Corpus execution result JSON must contain valid Unicode scalar text."
        ) from exc
    if len(raw) > CORPUS_EXECUTION_RESULT_MAX_JSON_BYTES:
        raise ValueError(
            f"Corpus execution result JSON exceeds {CORPUS_EXECUTION_RESULT_MAX_JSON_BYTES} bytes."
        )
    try:
        decoded = json.loads(
            source,
            parse_int=partial(
                parse_durable_json_integer_literal,
                field_name="corpus execution result JSON",
            ),
            parse_constant=partial(
                reject_nonportable_json_constant,
                field_name="corpus execution result JSON",
            ),
            object_pairs_hook=partial(
                durable_json_object_from_pairs,
                field_name="corpus execution result JSON",
            ),
        )
    except RecursionError as exc:
        raise ValueError(
            "Corpus execution result JSON exceeds the supported nesting depth."
        ) from exc
    document = copy_durable_json_object(decoded, "corpus execution result JSON")
    raw_version = document.get("schema_version")
    if type(raw_version) is not int or raw_version != CORPUS_EXECUTION_RESULT_SCHEMA_VERSION:
        raise ValueError(
            "Corpus execution result has unsupported schema_version "
            f"{raw_version!r}; this Cayu version supports only "
            f"{CORPUS_EXECUTION_RESULT_SCHEMA_VERSION}."
        )
    # Aggregate counters intentionally use exact decimal strings on JSON
    # boundaries while remaining strict integers in Python. Re-enter through
    # Pydantic's JSON mode after the duplicate-key and durable-value scan so
    # those field serializers are decoded without weakening Python callers.
    normalized = json.dumps(
        document,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return CorpusExecutionResult.model_validate_json(normalized)


def load_corpus_execution_result(path: str | Path) -> CorpusExecutionResult:
    """Read no more than the public result hard limit before validation."""

    with Path(path).open("rb") as handle:
        raw = handle.read(CORPUS_EXECUTION_RESULT_MAX_JSON_BYTES + 1)
    if len(raw) > CORPUS_EXECUTION_RESULT_MAX_JSON_BYTES:
        raise ValueError(
            f"Corpus execution result JSON exceeds {CORPUS_EXECUTION_RESULT_MAX_JSON_BYTES} bytes."
        )
    try:
        source = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("Corpus execution result JSON must be UTF-8.") from exc
    return corpus_execution_result_from_json(source)


def write_corpus_execution_result(
    result: CorpusExecutionResult,
    path: str | Path,
) -> None:
    Path(path).write_text(corpus_execution_result_to_json(result), encoding="utf-8")


def render_corpus_execution_html(result: CorpusExecutionResult) -> str:
    """Render a standalone report exclusively from public, sanitized result data."""

    result = _validated_result(result)
    run = result.run
    presentation = present_eval_result(result)
    case_rows = "\n".join(
        "<tr>"
        f"<td><code>{_escape(case.case_id)}</code></td>"
        f"<td>{_badge(case.status)}</td>"
        f"<td>{_score(case.score)}</td>"
        f"<td>{len(case.trials)}</td>"
        f"<td>{case.duration_ms} ms</td>"
        "</tr>"
        for case in run.cases
    )
    case_sections = "\n".join(
        _case_section(case, presented_case)
        for case, presented_case in zip(run.cases, presentation.cases, strict=True)
    )
    policy = presentation.trial_policy
    exposure = presentation.accepted_exposure
    exposure_html = ""
    if exposure is not None:
        exposure_html = (
            "<p>Accepted maximum work: "
            f"{exposure.candidate_trials} candidate trial(s), "
            f"{exposure.maximum_candidate_model_steps} candidate model step(s), "
            f"{_optional_maximum(exposure.maximum_candidate_total_tokens)} candidate token(s); "
            f"{exposure.judge_evaluations} judge evaluation(s), concurrency "
            f"{exposure.max_concurrency}, "
            f"{_optional_maximum(exposure.maximum_judge_input_tokens)}/"
            f"{_optional_maximum(exposure.maximum_judge_output_tokens)}/"
            f"{_optional_maximum(exposure.maximum_judge_total_tokens)} judge "
            "input/output/total token maximum(s). "
            "Candidate cost "
            f"{_maximum_cost_text(exposure.candidate_cost)}; judge cost "
            f"{_maximum_cost_text(exposure.judge_cost)}. Revision "
            f"<code>{_escape(exposure.revision)}</code></p>"
        )
    rendered = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Cayu Eval Report — {_escape(run.suite_id)}</title>
  <style>
    :root {{ color-scheme: light; --ink:#18211d; --muted:#5d6864; --line:#d9dfdc; --paper:#fff; --canvas:#f7f7f4; }}
    * {{ box-sizing: border-box; }}
    body {{ margin:0; font:15px/1.5 Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; color:var(--ink); background:var(--canvas); }}
    .page {{ max-width:1120px; margin:auto; padding:32px 24px 56px; }}
    h1 {{ margin:0 0 6px; font-size:2rem; }} h2 {{ margin:30px 0 12px; }} h3 {{ margin:0 0 12px; }} h4 {{ margin:18px 0 8px; }}
    p {{ color:var(--muted); }} code {{ overflow-wrap:anywhere; }}
    .metrics {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:12px; margin:24px 0; }}
    .metric,.case,.trial,table {{ background:var(--paper); border:1px solid var(--line); border-radius:9px; }}
    .metric {{ padding:14px; }} .metric strong {{ display:block; font-size:1.25rem; }}
    .dimensions {{ display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:8px; margin:14px 0 24px; }}
    .dimension,.judge {{ background:var(--paper); border:1px solid var(--line); border-radius:8px; padding:10px; }}
    .dimension strong {{ display:block; font-size:.76rem; text-transform:uppercase; color:var(--muted); }}
    .judge {{ margin-top:10px; }} .judge p {{ margin:5px 0; }}
    table {{ width:100%; border-collapse:collapse; overflow:hidden; }}
    th,td {{ padding:10px 12px; border-bottom:1px solid var(--line); text-align:left; vertical-align:top; }}
    th {{ background:#f0f4f3; font-size:.78rem; text-transform:uppercase; letter-spacing:.04em; }} tr:last-child td {{ border-bottom:0; }}
    .case {{ padding:18px; margin-top:14px; }} .trial {{ padding:14px; margin-top:10px; box-shadow:none; }}
    .assertion {{ display:grid; grid-template-columns:110px minmax(0,1fr); gap:12px; border-top:1px solid #e6ebe8; padding:10px 0; }}
    .assertion pre,.trial>pre {{ margin:6px 0 0; white-space:pre-wrap; overflow:auto; background:#f0f4f3; padding:10px; border-radius:6px; }}
    .badge {{ display:inline-block; padding:3px 8px; border-radius:999px; font-size:.78rem; font-weight:700; }}
    .passed {{ color:#0f5132; background:#d9f2e3; }} .failed {{ color:#842029; background:#f8d7da; }}
    .unavailable {{ color:#553c00; background:#fff0b3; }} .error {{ color:#664d03; background:#fff3cd; }}
    @media (max-width:760px) {{ .metrics,.dimensions {{ grid-template-columns:1fr 1fr; }} .assertion {{ grid-template-columns:1fr; }} }}
  </style>
</head>
<body>
  <main class="page">
    <h1>Cayu Eval Report</h1>
    <p>Suite <code>{_escape(run.suite_id)}</code> · target <code>{_escape(run.target_key)}</code></p>
    <div class="metrics">
      <div class="metric"><strong>{_badge(run.status)}</strong><span>Status</span></div>
      <div class="metric"><strong>{_score(run.score)}</strong><span>Score</span></div>
      <div class="metric"><strong>{len(run.cases)}</strong><span>Cases</span></div>
      <div class="metric"><strong>{run.duration_ms} ms</strong><span>Duration</span></div>
    </div>
    {_outcome_dimensions_html(presentation.dimensions)}
    <section class="case" aria-labelledby="target-identity">
      <h2 id="target-identity">Execution identity</h2>
      <p>Result <code>{_escape(result.revision)}</code> · published run <code>{_escape(presentation.evaluation_revision)}</code></p>
      <p>Application release <code>{_escape(result.target.application_release_id)}</code></p>
      <p>AppManifest schema <code>{_escape(result.target.app_manifest_schema_version)}</code> · fingerprint <code>{_escape(result.target.app_manifest_fingerprint)}</code></p>
      <p>Corpus <code>{_escape(run.corpus_revision)}</code> · suite <code>{_escape(run.suite_revision)}</code> · evidence policy <code>{_escape(run.evidence_policy_revision)}</code></p>
      <p>Trial policy: {policy.minimum_passed_trials} of {policy.trial_count} must pass · maximum concurrency {policy.max_concurrency} · revision <code>{_escape(policy.revision)}</code>. Runtime errors, evaluator errors, unavailable required evidence, and cancellations fail closed.</p>
      {exposure_html}
    </section>
    <h2>Cases</h2>
    <table>
      <thead><tr><th>Case</th><th>Status</th><th>Score</th><th>Trials</th><th>Duration</th></tr></thead>
      <tbody>{case_rows}</tbody>
    </table>
    <h2>Trial and assertion details</h2>
    {case_sections}
  </main>
</body>
</html>
"""
    rendered = require_durable_text(rendered, "corpus execution HTML report")
    if len(rendered.encode("utf-8")) > CORPUS_EXECUTION_RESULT_MAX_HTML_BYTES:
        raise ValueError(
            f"Corpus execution HTML report exceeds {CORPUS_EXECUTION_RESULT_MAX_HTML_BYTES} bytes."
        )
    return rendered


def render_captured_evaluation_html(result: CapturedEvaluationResultV1) -> str:
    """Render a standalone report from one public captured-result document."""

    if type(result) is not CapturedEvaluationResultV1:
        raise TypeError("result must be an exact CapturedEvaluationResultV1.")
    result = CapturedEvaluationResultV1.model_validate(
        result.model_dump(mode="python", round_trip=True, warnings="none")
    )
    score = result.score
    presentation = present_eval_result(result)
    presented_trial = presentation.cases[0].trials[0]
    assertion_rows = "\n".join(
        _assertion_row(assertion, presented_assertion)
        for assertion, presented_assertion in zip(
            score.assertions,
            presented_trial.assertions,
            strict=True,
        )
    )
    memory = score.memory_attribution
    memory_summary = eval_memory_attribution_summary(memory)
    rendered = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Cayu Captured Eval Report — {_escape(result.suite_id)}</title>
  <style>
    :root {{ color-scheme:light; --ink:#18211d; --muted:#5d6864; --line:#d9dfdc; --paper:#fff; --canvas:#f7f7f4; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; font:15px/1.5 Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; color:var(--ink); background:var(--canvas); }}
    .page {{ max-width:1120px; margin:auto; padding:32px 24px 56px; }}
    h1 {{ margin:0 0 6px; font-size:2rem; }} h2 {{ margin:30px 0 12px; }} p {{ color:var(--muted); }} code {{ overflow-wrap:anywhere; }}
    .metrics {{ display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:12px; margin:24px 0; }}
    .metric,.card {{ background:var(--paper); border:1px solid var(--line); border-radius:9px; }}
    .metric {{ padding:14px; }} .metric strong {{ display:block; font-size:1.25rem; }} .card {{ padding:18px; }}
    .dimensions {{ display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:8px; margin:14px 0 24px; }}
    .dimension,.judge {{ background:var(--paper); border:1px solid var(--line); border-radius:8px; padding:10px; }}
    .dimension strong {{ display:block; font-size:.76rem; text-transform:uppercase; color:var(--muted); }}
    .judge {{ margin-top:10px; }} .judge p {{ margin:5px 0; }}
    table {{ width:100%; border-collapse:collapse; margin-top:8px; }} th,td {{ padding:8px; border:1px solid var(--line); text-align:left; vertical-align:top; }}
    .assertion {{ display:grid; grid-template-columns:110px minmax(0,1fr); gap:12px; border-top:1px solid #e6ebe8; padding:10px 0; }}
    .assertion:first-child {{ border-top:0; }} .assertion pre {{ margin:6px 0 0; white-space:pre-wrap; overflow:auto; background:#f0f4f3; padding:10px; border-radius:6px; }}
    .badge {{ display:inline-block; padding:3px 8px; border-radius:999px; font-size:.78rem; font-weight:700; }}
    .passed {{ color:#0f5132; background:#d9f2e3; }} .failed {{ color:#842029; background:#f8d7da; }}
    .unavailable {{ color:#553c00; background:#fff0b3; }} .error {{ color:#664d03; background:#fff3cd; }}
    @media (max-width:760px) {{ .metrics,.dimensions {{ grid-template-columns:1fr; }} .assertion {{ grid-template-columns:1fr; }} }}
  </style>
</head>
<body>
  <main class="page">
    <h1>Cayu Captured Eval Report</h1>
    <p>Suite <code>{_escape(result.suite_id)}</code> · target <code>{_escape(result.target.target_key)}</code></p>
    <div class="metrics">
      <div class="metric"><strong>{_badge(score.status)}</strong><span>Status</span></div>
      <div class="metric"><strong>{_score(score.score)}</strong><span>Score</span></div>
      <div class="metric"><strong>{len(score.assertions)}</strong><span>Assertions</span></div>
    </div>
    {_outcome_dimensions_html(presentation.dimensions)}
    <section class="card" aria-labelledby="result-identity">
      <h2 id="result-identity">Immutable result identity</h2>
      <p>Result <code>{_escape(result.revision)}</code> · captured score <code>{_escape(presentation.evaluation_revision)}</code> · origin <code>captured_session</code></p>
      <p>Application release <code>{_escape(result.target.application_release_id)}</code></p>
      <p>AppManifest schema <code>{_escape(result.target.app_manifest_schema_version)}</code> · fingerprint <code>{_escape(result.target.app_manifest_fingerprint)}</code></p>
      <p>Corpus <code>{_escape(result.corpus_revision)}</code> · suite <code>{_escape(result.suite_revision)}</code></p>
      <p>Case <code>{_escape(score.case_id)}</code> · evidence <code>{_escape(score.evidence_revision)}</code> · evidence policy <code>{_escape(score.evidence_policy_revision)}</code></p>
      <p>Memory attribution {_escape(memory_summary)} · revision <code>{_escape(memory.revision)}</code></p>
      <p>Full memory-attribution record inspection is unsupported in HTML; use the JSON result for aliases, fingerprints, source references, and complete lifecycle transitions.</p>
    </section>
    <h2>Captured assertion evidence</h2>
    <section class="card">{assertion_rows}</section>
  </main>
</body>
</html>
"""
    rendered = require_durable_text(rendered, "captured evaluation HTML report")
    if len(rendered.encode("utf-8")) > CORPUS_EXECUTION_RESULT_MAX_HTML_BYTES:
        raise ValueError(
            "Captured evaluation HTML report exceeds "
            f"{CORPUS_EXECUTION_RESULT_MAX_HTML_BYTES} bytes."
        )
    return rendered


def render_eval_result_html(
    result: CorpusExecutionResult | CapturedEvaluationResultV1,
) -> str:
    """Render either immutable eval-result origin through its exact report shape."""

    if type(result) is CorpusExecutionResult:
        return render_corpus_execution_html(result)
    if type(result) is CapturedEvaluationResultV1:
        return render_captured_evaluation_html(result)
    raise TypeError("result must be an exact CorpusExecutionResult or CapturedEvaluationResultV1.")


def render_corpus_execution_comparison_html(
    comparison: CorpusExecutionComparison,
) -> str:
    """Render a standalone comparison exclusively from its bounded public graph."""

    if type(comparison) is not CorpusExecutionComparison:
        raise TypeError("comparison must be an exact CorpusExecutionComparison.")
    comparison = CorpusExecutionComparison.model_validate(
        comparison.model_dump(mode="python", round_trip=True, warnings="none")
    )
    compatibility = comparison.compatibility
    if compatibility.comparable:
        compatibility_text = "Compatible evaluation contracts"
        compatibility_details = ""
    else:
        compatibility_text = "Incomparable evaluation contracts"
        compatibility_details = (
            "<ul>"
            + "".join(
                f"<li><code>{_escape(reason.value)}</code></li>" for reason in compatibility.reasons
            )
            + "</ul>"
        )
    regression_rows = "\n".join(
        "<tr>"
        f"<td>{_escape(regression.scope.value)}</td>"
        f"<td>{_escape(regression.case_id or 'run')}</td>"
        f"<td>{_escape(regression.kind.value)}</td>"
        f"<td>{_escape(_regression_change(regression))}</td>"
        "</tr>"
        for regression in comparison.regressions
    )
    structured_regression_rows = "\n".join(
        "<tr>"
        "<td>structured judge</td>"
        f"<td>{_escape(item.case_id)}</td>"
        "<td>judgment</td>"
        f"<td>{_escape(_structured_regression_change(item))}</td>"
        "</tr>"
        for item in comparison.structured_judgments
        if item.regressed
    )
    tool_json_regression_rows = "\n".join(
        "<tr>"
        f"<td>{_escape(item.baseline.kind)}</td>"
        f"<td>{_escape(item.case_id)}</td>"
        "<td>tool JSON assertion</td>"
        f"<td>{_escape(_tool_json_regression_change(item))}</td>"
        "</tr>"
        for item in comparison.tool_json_assertions
        if item.regressed
    )
    regression_rows = "\n".join(
        rows
        for rows in (
            regression_rows,
            structured_regression_rows,
            tool_json_regression_rows,
        )
        if rows
    )
    if not regression_rows:
        message = (
            "No compatible-result regressions."
            if compatibility.comparable
            else "Regressions are not evaluated for incomparable results."
        )
        regression_rows = f'<tr><td colspan="4">{message}</td></tr>'
    case_rows = "\n".join(
        "<tr>"
        f"<td><code>{_escape(case.case_id)}</code></td>"
        f"<td>{_badge(case.baseline_status)}</td>"
        f"<td>{_badge(case.current_status)}</td>"
        f"<td>{_escape(_reliability_distribution(case.baseline_reliability))}</td>"
        f"<td>{_escape(_reliability_distribution(case.current_reliability))}</td>"
        f"<td>{_escape(case.reliability_change)}</td>"
        f"<td>{_score(case.baseline_score)} → {_score(case.current_score)}</td>"
        "</tr>"
        for case in comparison.cases
    )
    if not case_rows:
        case_rows = (
            '<tr><td colspan="7">Case outcomes are omitted for incomparable results.</td></tr>'
        )
    structured_comparison = _structured_comparison_html(comparison)
    tool_json_comparison = _tool_json_comparison_html(comparison)
    rendered = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Cayu Eval Comparison</title>
  <style>
    :root {{ color-scheme:light; --ink:#18211d; --muted:#5d6864; --line:#d9dfdc; --paper:#fff; --canvas:#f7f7f4; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; font:15px/1.5 Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; color:var(--ink); background:var(--canvas); }}
    .page {{ max-width:1120px; margin:auto; padding:32px 24px 56px; }}
    h1 {{ margin:0 0 6px; font-size:2rem; }} h2 {{ margin:30px 0 12px; }} p {{ color:var(--muted); }} code {{ overflow-wrap:anywhere; }}
    .notice,.judgment,table {{ background:var(--paper); border:1px solid var(--line); border-radius:9px; }} .notice,.judgment {{ padding:14px; }}
    .judgment {{ margin-top:12px; }} .judgment p {{ margin:6px 0; }}
    table {{ width:100%; border-collapse:collapse; overflow:hidden; }} th,td {{ padding:10px 12px; border-bottom:1px solid var(--line); text-align:left; vertical-align:top; }}
    th {{ background:#f0f4f3; font-size:.78rem; text-transform:uppercase; letter-spacing:.04em; }} tr:last-child td {{ border-bottom:0; }}
    .badge {{ display:inline-block; padding:3px 8px; border-radius:999px; font-size:.78rem; font-weight:700; }}
    .passed {{ color:#0f5132; background:#d9f2e3; }} .failed {{ color:#842029; background:#f8d7da; }}
    .unavailable {{ color:#553c00; background:#fff0b3; }} .error {{ color:#664d03; background:#fff3cd; }}
  </style>
</head>
<body>
  <main class="page">
    <h1>Cayu Eval Comparison</h1>
    <p>Baseline <code>{_escape(comparison.baseline.result_revision)}</code> vs current <code>{_escape(comparison.current.result_revision)}</code></p>
    <p>Score regression tolerance <code>{_escape(format(comparison.score_tolerance, ".15g"))}</code></p>
    <p>Memory-attribution comparison: <code>{_escape(comparison.baseline.memory_attribution_support)}</code> in this compact report.</p>
    <section class="notice"><strong>{_escape(compatibility_text)}</strong>{compatibility_details}</section>
    <h2>Outcome</h2>
    <table>
      <thead><tr><th>Result</th><th>Release</th><th>App manifest</th><th>Status</th><th>Score</th></tr></thead>
      <tbody>
        <tr><td>Baseline</td><td><code>{_escape(comparison.baseline.application_release_id)}</code></td><td><code>{_escape(comparison.baseline.app_manifest_fingerprint)}</code></td><td>{_badge(comparison.baseline.status)}</td><td>{_score(comparison.baseline.score)}</td></tr>
        <tr><td>Current</td><td><code>{_escape(comparison.current.application_release_id)}</code></td><td><code>{_escape(comparison.current.app_manifest_fingerprint)}</code></td><td>{_badge(comparison.current.status)}</td><td>{_score(comparison.current.score)}</td></tr>
      </tbody>
    </table>
    <section class="notice">
      <p>Target <code>{_escape(comparison.baseline.target_key)}</code> · corpus <code>{_escape(comparison.baseline.corpus_revision)}</code> · suite <code>{_escape(comparison.baseline.suite_id)}</code> at <code>{_escape(comparison.baseline.suite_revision)}</code></p>
      <p>Evidence policy <code>{_escape(comparison.baseline.evidence_policy_revision)}</code> · pricing profile <code>{_escape(comparison.baseline.pricing_profile_fingerprint or "not used")}</code> · application execution target <code>{_escape(comparison.baseline.external_target_revision or "not used")}</code></p>
      <p>Baseline trial policy <code>{_escape(comparison.baseline.trial_policy_revision)}</code> · accepted exposure <code>{_escape(comparison.baseline.accepted_exposure_revision or "not applicable")}</code> · exposure comparison contract <code>{_escape(comparison.baseline.accepted_exposure_comparison_revision or "not applicable")}</code></p>
      <p>Current trial policy <code>{_escape(comparison.current.trial_policy_revision)}</code> · accepted exposure <code>{_escape(comparison.current.accepted_exposure_revision or "not applicable")}</code> · exposure comparison contract <code>{_escape(comparison.current.accepted_exposure_comparison_revision or "not applicable")}</code></p>
    </section>
    <h2>Regressions</h2>
    <table><thead><tr><th>Scope</th><th>Case</th><th>Kind</th><th>Change</th></tr></thead><tbody>{regression_rows}</tbody></table>
    <h2>Cases</h2>
    <table><thead><tr><th>Case</th><th>Baseline</th><th>Current</th><th>Baseline trials</th><th>Current trials</th><th>Reliability</th><th>Score</th></tr></thead><tbody>{case_rows}</tbody></table>
    {structured_comparison}
    {tool_json_comparison}
  </main>
</body>
</html>
"""
    rendered = require_durable_text(rendered, "corpus execution comparison HTML report")
    if len(rendered.encode("utf-8")) > CORPUS_EXECUTION_COMPARISON_MAX_HTML_BYTES:
        raise ValueError(
            "Corpus execution comparison HTML exceeds "
            f"{CORPUS_EXECUTION_COMPARISON_MAX_HTML_BYTES} bytes."
        )
    return rendered


def _structured_comparison_html(comparison: CorpusExecutionComparison) -> str:
    state = comparison.structured_judge_comparison_state
    state_messages = {
        "compared": "Exact retained structured-judge observations were compared.",
        "contract_incompatible": (
            "Structured judgments were not diffed because the evaluation contracts differ."
        ),
        "no_structured_judges": "Neither result contains structured-judge observations.",
        "observation_identity_mismatch": (
            "Structured judgments were not paired because retained case/trial/assertion "
            "identities differ."
        ),
        "source_detail_unavailable": (
            "Structured judgment detail is unavailable in one compact source projection."
        ),
    }
    mismatch_rows = "".join(
        "<tr>"
        f"<td><code>{_escape(item.case_id)}</code></td>"
        f"<td>{_escape(item.trial_number if item.trial_number is not None else 'captured')}</td>"
        f"<td><code>{_escape(item.assertion_id)}</code></td>"
        f"<td>{_escape(item.availability)}</td></tr>"
        for item in comparison.structured_judge_observation_mismatches
    )
    mismatch_table = ""
    if mismatch_rows:
        mismatch_table = (
            "<table><thead><tr><th>Case</th><th>Trial</th><th>Assertion</th>"
            f"<th>Availability</th></tr></thead><tbody>{mismatch_rows}</tbody></table>"
        )
    judgments = "".join(
        _structured_judgment_comparison_html(item) for item in comparison.structured_judgments
    )
    return (
        '<section aria-labelledby="structured-judgments">'
        '<h2 id="structured-judgments">Structured judge comparison</h2>'
        f'<div class="notice"><strong><code>{_escape(state)}</code></strong>'
        f"<p>{_escape(state_messages[state])}</p>{mismatch_table}</div>{judgments}</section>"
    )


def _tool_json_comparison_html(comparison: CorpusExecutionComparison) -> str:
    state = comparison.tool_json_comparison_state
    state_messages = {
        "compared": "Exact bounded tool-JSON observations were compared.",
        "contract_incompatible": (
            "Tool-JSON observations were not diffed because the evaluation contracts differ."
        ),
        "no_tool_json_assertions": "Neither result contains tool-JSON assertions.",
        "observation_identity_mismatch": (
            "Tool-JSON observations were not paired because retained "
            "case/trial/assertion identities differ."
        ),
        "source_detail_unavailable": (
            "Tool-JSON detail is unavailable in one compact source projection."
        ),
    }
    mismatch_rows = "".join(
        "<tr>"
        f"<td><code>{_escape(item.case_id)}</code></td>"
        f"<td>{_escape(item.trial_number if item.trial_number is not None else 'captured')}</td>"
        f"<td><code>{_escape(item.assertion_id)}</code></td>"
        f"<td>{_escape(item.availability)}</td></tr>"
        for item in comparison.tool_json_observation_mismatches
    )
    mismatch_table = ""
    if mismatch_rows:
        mismatch_table = (
            "<table><thead><tr><th>Case</th><th>Trial</th><th>Assertion</th>"
            f"<th>Availability</th></tr></thead><tbody>{mismatch_rows}</tbody></table>"
        )
    observations = "".join(
        _tool_json_assertion_comparison_html(item) for item in comparison.tool_json_assertions
    )
    return (
        '<section aria-labelledby="tool-json-assertions">'
        '<h2 id="tool-json-assertions">Tool JSON assertion comparison</h2>'
        f'<div class="notice"><strong><code>{_escape(state)}</code></strong>'
        f"<p>{_escape(state_messages[state])}</p>{mismatch_table}</div>{observations}</section>"
    )


def _tool_json_assertion_comparison_html(item: EvalToolJsonAssertionComparisonV1) -> str:
    trial = item.trial_number if item.trial_number is not None else "captured"
    baseline_actual = _bounded_json_text(item.baseline.actual)
    current_actual = _bounded_json_text(item.current.actual)
    expected = _bounded_json_text(item.baseline.expected_subset)
    return (
        '<section class="judgment">'
        f"<h3>Case <code>{_escape(item.case_id)}</code> · trial {_escape(trial)} · "
        f"assertion <code>{_escape(item.assertion_id)}</code></h3>"
        f"<p><code>{_escape(item.baseline.kind)}</code> · tool "
        f"<code>{_escape(item.baseline.tool_name)}</code> · occurrence "
        f"{_escape(item.baseline.occurrence)}</p>"
        f"<p>Outcome {_escape(item.baseline_outcome)} → {_escape(item.current_outcome)} · "
        f"evidence {_escape(item.baseline.observation_state)} → "
        f"{_escape(item.current.observation_state)} · observed value "
        f"{_escape(item.observed_value_change)}</p>"
        f"<p>Expected subset <code>{_escape(expected)}</code></p>"
        f"<p>Observed <code>{_escape(baseline_actual)}</code> → "
        f"<code>{_escape(current_actual)}</code></p></section>"
    )


def _bounded_json_text(value: object) -> str:
    if value is None:
        return "unavailable"
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _structured_judgment_comparison_html(
    item: EvalStructuredJudgeComparisonV1,
) -> str:
    baseline_detail = item.baseline.detail
    current_detail = item.current.detail
    profile = baseline_detail.judge_profile
    baseline_criteria = {criterion.criterion_id: criterion for criterion in item.baseline.criteria}
    current_criteria = {criterion.criterion_id: criterion for criterion in item.current.criteria}
    criterion_rows = "".join(
        "<tr>"
        f"<td><code>{_escape(criterion.criterion_id)}</code></td>"
        f"<td>{_escape(criterion.weight)}</td>"
        f"<td>{_escape(criterion.baseline_score)}</td>"
        f"<td>{_escape(criterion.current_score)}</td>"
        f"<td>{_escape(criterion.score_delta)}</td>"
        f"<td>{_escape(criterion.baseline_explanation_state)} → "
        f"{_escape(criterion.current_explanation_state)}</td>"
        f"<td>{_escape(baseline_criteria[criterion.criterion_id].explanation or 'Unavailable')}</td>"
        f"<td>{_escape(current_criteria[criterion.criterion_id].explanation or 'Unavailable')}</td>"
        "</tr>"
        for criterion in item.criteria
    )
    if not criterion_rows:
        criterion_rows = (
            '<tr><td colspan="8">Criterion deltas are unavailable because one or both '
            "judgments were not recorded.</td></tr>"
        )
    reference = baseline_detail.reference
    reference_text = "none"
    if reference is not None:
        reference_text = f"{reference.kind}:{reference.key}@{reference.revision}"
    trial = item.trial_number if item.trial_number is not None else "captured"
    return (
        '<section class="judgment">'
        f"<h3>Case <code>{_escape(item.case_id)}</code> · trial {_escape(trial)} · "
        f"assertion <code>{_escape(item.assertion_id)}</code></h3>"
        f"<p>Outcome {_escape(item.baseline_outcome)} → {_escape(item.current_outcome)} · "
        f"evaluator {_escape(item.evaluator_change)} · aggregate "
        f"{_escape(baseline_detail.aggregate_score or 'unavailable')} → "
        f"{_escape(current_detail.aggregate_score or 'unavailable')} "
        f"(delta {_escape(item.aggregate_delta or 'unavailable')}, "
        f"{_escape(item.aggregate_change)})</p>"
        f"<p>Profile <strong>{_escape(profile.label)}</strong> · "
        f"<code>{_escape(profile.key)}</code> at "
        f"<code>{_escape(profile.revision)}</code> · provider/model "
        f"<code>{_escape(profile.provider_name)}/{_escape(profile.model)}</code> · "
        f"route <code>{_escape(baseline_detail.candidate_route_relation)}</code></p>"
        f"<p>Rubric <code>{_escape(baseline_detail.rubric_id)}</code> at "
        f"<code>{_escape(baseline_detail.rubric_revision)}</code> · reference "
        f"<code>{_escape(reference_text)}</code></p>"
        f"<p>Evaluator diagnostic {_escape(baseline_detail.diagnostic)} → "
        f"{_escape(current_detail.diagnostic)} · observed usage "
        f"{_escape(_judge_usage_text(item.baseline))} → "
        f"{_escape(_judge_usage_text(item.current))} · observed cost "
        f"{_escape(_judge_cost_text(item.baseline))} → "
        f"{_escape(_judge_cost_text(item.current))}</p>"
        "<table><thead><tr><th>Criterion</th><th>Weight</th><th>Baseline</th>"
        "<th>Current</th><th>Delta</th><th>Explanation state</th>"
        "<th>Baseline explanation</th><th>Current explanation</th></tr></thead>"
        f"<tbody>{criterion_rows}</tbody></table></section>"
    )


def _structured_regression_change(item: EvalStructuredJudgeComparisonV1) -> str:
    trial = item.trial_number if item.trial_number is not None else "captured"
    delta = item.aggregate_delta if item.aggregate_delta is not None else "unavailable"
    return (
        f"trial {trial}, assertion {item.assertion_id}: "
        f"outcome {item.baseline_outcome} → {item.current_outcome}; "
        f"evaluator {item.evaluator_change}; aggregate {item.aggregate_change} ({delta})"
    )


def _tool_json_regression_change(item: EvalToolJsonAssertionComparisonV1) -> str:
    trial = item.trial_number if item.trial_number is not None else "captured"
    return (
        f"trial {trial}, assertion {item.assertion_id}: "
        f"outcome {item.baseline_outcome} → {item.current_outcome}; "
        f"evidence {item.baseline.observation_state} → {item.current.observation_state}; "
        f"observed value {item.observed_value_change}"
    )


def _judge_usage_text(judgment: EvalStructuredJudgePresentationV1) -> str:
    usage = judgment.detail.usage
    if usage is None:
        return "unavailable"
    return (
        f"{usage.model_steps} step(s), {usage.input_tokens}/{usage.output_tokens}/"
        f"{usage.total_tokens} input/output/total tokens"
    )


def _judge_cost_text(judgment: EvalStructuredJudgePresentationV1) -> str:
    cost = judgment.detail.cost
    if cost is None:
        return "unavailable (not observed)"
    if cost.availability == "unavailable":
        return "unavailable (unpriced)"
    return f"{cost.estimated_cost} {cost.currency}"


def write_corpus_execution_html(
    result: CorpusExecutionResult,
    path: str | Path,
) -> None:
    Path(path).write_text(render_corpus_execution_html(result), encoding="utf-8")


def _case_section(case: Any, presentation: EvalCasePresentationV2) -> str:
    trials = "\n".join(
        _trial_section(trial, presented_trial)
        for trial, presented_trial in zip(case.trials, presentation.trials, strict=True)
    )
    return (
        f'<section class="case"><h3>Case <code>{_escape(case.case_id)}</code> '
        f"{_badge(case.status)}</h3>"
        f"<p>Revision <code>{_escape(case.case_revision)}</code> · score {_score(case.score)} · "
        f"{case.duration_ms} ms across {len(case.trials)} trial(s)</p>"
        f"<p>Reliability: {presentation.reliability.passed_trials}/"
        f"{presentation.reliability.total_trials} passed · "
        f"{presentation.reliability.candidate_failed_trials} candidate failure(s) · "
        f"{presentation.reliability.runtime_error_trials} runtime error(s) · "
        f"{presentation.reliability.evaluator_error_trials} evaluator error(s) · "
        f"{presentation.reliability.unavailable_trials} unavailable trial(s) · "
        f"{presentation.reliability.cancelled_trials} cancelled trial(s) · "
        f"score min/mean/max {_reliability_score_text(presentation.reliability.minimum_score)}/"
        f"{_reliability_score_text(presentation.reliability.mean_score)}/"
        f"{_reliability_score_text(presentation.reliability.maximum_score)} across "
        f"{presentation.reliability.scored_trials} scored trial(s) · "
        f"variability <code>{_escape(presentation.reliability.variability)}</code></p>"
        f"{trials}</section>"
    )


def _maximum_cost_text(exposure: EvalMaximumCostExposureV1) -> str:
    if exposure.state == "not_applicable":
        return "not applicable"
    if exposure.state == "unavailable":
        return f"unavailable ({exposure.unavailable_reason})"
    return " + ".join(f"{item.amount} {item.currency}" for item in exposure.totals)


def _optional_maximum(value: int | None) -> str:
    return "unavailable maximum" if value is None else str(value)


def _reliability_score_text(value: float | None) -> str:
    return "unavailable" if value is None else f"{value:.3f}"


def _trial_section(trial: Any, presentation: EvalTrialPresentationV1) -> str:
    assertions = "\n".join(
        _assertion_row(assertion, presented_assertion)
        for assertion, presented_assertion in zip(
            trial.assertions,
            presentation.assertions,
            strict=True,
        )
    )
    usage = "Unavailable"
    if trial.usage is not None:
        usage = (
            f"{trial.usage.model_steps} model step(s), {trial.usage.tool_calls} tool call(s), "
            f"{trial.usage.total_tokens} token(s)"
        )
    output = trial.output
    output_summary = output.evidence_state.replace("_", " ")
    if output.preview_truncated:
        output_summary += " · preview truncated"
    output_digest = (
        "unavailable" if output.retained_sha256 is None else f"sha256:{output.retained_sha256}"
    )
    output_block = (
        f"<h5>Redacted output preview</h5><p>{_escape(output_summary)} · "
        f"{output.retained_chars} retained character(s) · "
        f"<code>{_escape(output_digest)}</code></p>"
    )
    if output.text:
        output_block += f"<pre>{_escape(output.text)}</pre>"
    memory = trial.memory_attribution
    memory_summary = eval_memory_attribution_summary(memory)
    return (
        f'<section class="trial"><h4>Trial {trial.trial_number} {_badge(trial.status)}</h4>'
        f"<p>Score {_score(trial.score)} · {trial.duration_ms} ms · "
        f"evidence {'complete' if trial.evidence_complete else 'incomplete'} · "
        f"usage {_escape(usage)} · reason <code>{_escape(trial.code)}</code></p>"
        f"<p>{_escape(trial.message)}</p>"
        f"<p>Memory attribution: {_escape(memory_summary)} · revision "
        f"<code>{_escape(memory.revision)}</code></p>"
        "<p>Full memory-attribution record inspection is unsupported in HTML; "
        "use the JSON result for aliases, fingerprints, source references, and complete "
        "lifecycle transitions.</p>"
        f"{_outcome_dimensions_html(presentation.dimensions)}"
        f"{output_block}{assertions}</section>"
    )


def _assertion_row(
    assertion: Any,
    presentation: EvalAssertionPresentationV1,
) -> str:
    if presentation.structured_judge is not None:
        detail = _structured_judge_html(presentation.structured_judge)
    elif presentation.model_judge is not None:
        judge = presentation.model_judge
        profile = judge.judge_profile
        usage = judge.usage
        cost = judge.cost
        usage_text = (
            "unavailable"
            if usage is None
            else (
                f"{usage.model_steps} model step(s), {usage.input_tokens} input, "
                f"{usage.output_tokens} output, {usage.total_tokens} total token(s)"
            )
        )
        cost_text = "unavailable (not observed)"
        if cost is not None and cost.availability == "unavailable":
            cost_text = "unavailable (unpriced)"
        elif cost is not None:
            cost_text = f"{cost.estimated_cost} {cost.currency}"
        detail = (
            '<section class="judge">'
            f"<p><strong>{_escape(profile.label)}</strong> · profile "
            f"<code>{_escape(profile.key)}</code> at "
            f"<code>{_escape(profile.revision)}</code> · provider/model "
            f"<code>{_escape(profile.provider_name)}/{_escape(profile.model)}</code></p>"
            f"<p>Evaluator <code>{_escape(judge.diagnostic)}</code> · route "
            f"<code>{_escape(judge.candidate_route_relation)}</code> · threshold "
            f"<code>{_escape(judge.threshold)}</code></p>"
            f"<p>Observed usage: {_escape(usage_text)} · observed cost: "
            f"{_escape(cost_text)}</p></section>"
        )
    else:
        public_detail = json.dumps(
            assertion.detail.model_dump(mode="json"),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        detail = f"<pre>{_escape(public_detail)}</pre>"
    return (
        '<div class="assertion">'
        f"<div>{_badge(assertion.outcome)}</div>"
        f"<div><strong><code>{_escape(assertion.assertion_id)}</code></strong> "
        f"<span>({_escape(assertion.detail.kind)})</span>"
        f"<p>{_escape(assertion.message)} · score {_score(assertion.score)}</p>"
        f"{detail}</div></div>"
    )


def _outcome_dimensions_html(dimensions: EvalResultOutcomeDimensionsV1) -> str:
    values = (
        ("Candidate", dimensions.candidate),
        ("Deterministic assertions", dimensions.deterministic_assertions),
        ("Semantic quality", dimensions.semantic_quality),
        ("Evaluator health", dimensions.evaluator_health),
        ("Runtime", dimensions.runtime),
        ("Evidence", dimensions.evidence),
    )
    return (
        '<section class="dimensions" aria-label="Outcome dimensions">'
        + "".join(
            '<div class="dimension">'
            f"<strong>{_escape(label)}</strong><span>{_escape(value)}</span></div>"
            for label, value in values
        )
        + "</section>"
    )


def _structured_judge_html(judgment: EvalStructuredJudgePresentationV1) -> str:
    detail = judgment.detail
    profile = detail.judge_profile
    reference = detail.reference
    reference_text = "No evaluator reference"
    if reference is not None:
        reference_text = (
            f"{reference.kind} <code>{_escape(reference.key)}</code> · revision "
            f"<code>{_escape(reference.revision)}</code> · {_escape(reference.availability)}"
        )
        if reference.privacy_policy_key is not None:
            reference_text += (
                f" · privacy policy <code>{_escape(reference.privacy_policy_key)}</code> "
                f"at <code>{_escape(reference.privacy_policy_revision)}</code>"
            )
    evidence_parts = ["final output"] if detail.evidence.include_final_output else []
    if detail.evidence.include_transcript:
        evidence_parts.append("transcript")
    evidence_text = ", ".join(evidence_parts) or "none"
    threshold_state = (
        "unavailable"
        if judgment.threshold_passed is None
        else "passed"
        if judgment.threshold_passed
        else "failed"
    )
    usage_text = "Unavailable"
    if detail.usage is not None:
        usage_text = (
            f"{detail.usage.model_steps} model step(s), {detail.usage.input_tokens} input, "
            f"{detail.usage.output_tokens} output, {detail.usage.total_tokens} total token(s)"
        )
    cost_text = "Unavailable (not observed)"
    if detail.cost is not None and detail.cost.availability == "unavailable":
        cost_text = "Unavailable (unpriced)"
    elif detail.cost is not None:
        cost_text = (
            f"{detail.cost.estimated_cost} {detail.cost.currency} · "
            f"{detail.cost.priced_model_steps} priced model step(s)"
        )
    criterion_rows = "".join(
        "<tr>"
        f"<td><code>{_escape(item.criterion_id)}</code></td>"
        f"<td>{_escape(item.weight)}</td>"
        f"<td>{_escape(item.score)}</td>"
        f"<td>{_escape(item.weighted_contribution)}</td>"
        f"<td>{_escape(item.explanation_state)}</td>"
        f"<td>{_escape(item.explanation or 'Unavailable')}</td>"
        "</tr>"
        for item in judgment.criteria
    )
    if not criterion_rows:
        criterion_rows = '<tr><td colspan="6">No criterion scores were recorded.</td></tr>'
    evidence_json = json.dumps(
        detail.evidence.model_dump(mode="json"), ensure_ascii=False, sort_keys=True
    )
    return (
        '<section class="judge">'
        f"<p><strong>{_escape(profile.label)}</strong> · profile "
        f"<code>{_escape(profile.key)}</code> at <code>{_escape(profile.revision)}</code></p>"
        f"<p>Provider/model <code>{_escape(profile.provider_name)}/{_escape(profile.model)}</code> "
        f"· implementation <code>{_escape(profile.implementation_revision)}</code> · route "
        f"<code>{_escape(detail.candidate_route_relation)}</code></p>"
        f"<p>Rubric <code>{_escape(detail.rubric_id)}</code> at "
        f"<code>{_escape(detail.rubric_revision)}</code></p>"
        f"<p>{reference_text}</p>"
        f"<p>Evidence {_escape(evidence_text)} · selection "
        f"<code>{_escape(evidence_json)}</code></p>"
        f"<p>Evaluator <code>{_escape(detail.diagnostic)}</code> · aggregate "
        f"<code>{_escape(detail.aggregate_score or 'unavailable')}</code> against threshold "
        f"<code>{_escape(detail.threshold)}</code> ({_escape(threshold_state)})</p>"
        f"<p>Observed usage: {_escape(usage_text)} · observed cost: {_escape(cost_text)}</p>"
        "<table><thead><tr><th>Criterion</th><th>Weight</th><th>Score</th>"
        "<th>Contribution</th><th>Explanation state</th><th>Explanation</th></tr></thead>"
        f"<tbody>{criterion_rows}</tbody></table></section>"
    )


def _badge(status: str) -> str:
    return f'<span class="badge {_escape(status)}">{_escape(status)}</span>'


def _score(score: float | None) -> str:
    return "unavailable" if score is None else f"{score:.2f}"


def _reliability_distribution(distribution: CorpusReliabilityDistributionV1) -> str:
    return (
        f"{distribution.passed_trials} passed, "
        f"{distribution.candidate_failed_trials} candidate failed, "
        f"{distribution.runtime_error_trials} runtime error, "
        f"{distribution.evaluator_error_trials} evaluator error, "
        f"{distribution.unavailable_trials} unavailable, "
        f"{distribution.cancelled_trials} cancelled"
    )


def _regression_change(regression: CorpusExecutionRegression) -> str:
    if regression.kind is CorpusRegressionKind.STATUS:
        baseline_status = regression.baseline_status
        current_status = regression.current_status
        if baseline_status is None or current_status is None:
            raise ValueError("Status regression is missing its status pair.")
        return f"{baseline_status} → {current_status}"
    if regression.kind is CorpusRegressionKind.SCORE:
        baseline_score = regression.baseline_score
        current_score = regression.current_score
        if baseline_score is None or current_score is None:
            raise ValueError("Score regression is missing its score pair.")
        return f"{baseline_score:.2f} → {current_score:.2f}"
    if regression.kind is CorpusRegressionKind.RELIABILITY:
        baseline_reliability = regression.baseline_reliability
        current_reliability = regression.current_reliability
        if baseline_reliability is None or current_reliability is None:
            raise ValueError("Reliability regression is missing its distribution pair.")
        return (
            f"{_reliability_distribution(baseline_reliability)} → "
            f"{_reliability_distribution(current_reliability)}"
        )
    raise ValueError(f"Unsupported regression kind {regression.kind!r}.")


def _escape(value: Any) -> str:
    return html.escape(str(value), quote=True)

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

CORPUS_EXECUTION_RESULT_MAX_JSON_BYTES = 48 << 20
CORPUS_EXECUTION_RESULT_MAX_HTML_BYTES = 48 << 20


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
    case_sections = "\n".join(_case_section(case) for case in run.cases)
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
    table {{ width:100%; border-collapse:collapse; overflow:hidden; }}
    th,td {{ padding:10px 12px; border-bottom:1px solid var(--line); text-align:left; vertical-align:top; }}
    th {{ background:#f0f4f3; font-size:.78rem; text-transform:uppercase; letter-spacing:.04em; }} tr:last-child td {{ border-bottom:0; }}
    .case {{ padding:18px; margin-top:14px; }} .trial {{ padding:14px; margin-top:10px; box-shadow:none; }}
    .assertion {{ display:grid; grid-template-columns:110px minmax(0,1fr); gap:12px; border-top:1px solid #e6ebe8; padding:10px 0; }}
    .assertion pre,.trial>pre {{ margin:6px 0 0; white-space:pre-wrap; overflow:auto; background:#f0f4f3; padding:10px; border-radius:6px; }}
    .badge {{ display:inline-block; padding:3px 8px; border-radius:999px; font-size:.78rem; font-weight:700; }}
    .passed {{ color:#0f5132; background:#d9f2e3; }} .failed {{ color:#842029; background:#f8d7da; }}
    .unavailable {{ color:#553c00; background:#fff0b3; }} .error {{ color:#664d03; background:#fff3cd; }}
    @media (max-width:760px) {{ .metrics {{ grid-template-columns:1fr 1fr; }} .assertion {{ grid-template-columns:1fr; }} }}
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
    <section class="case" aria-labelledby="target-identity">
      <h2 id="target-identity">Execution identity</h2>
      <p>Application release <code>{_escape(result.target.application_release_id)}</code></p>
      <p>AppManifest schema <code>{_escape(result.target.app_manifest_schema_version)}</code> · fingerprint <code>{_escape(result.target.app_manifest_fingerprint)}</code></p>
      <p>Corpus <code>{_escape(run.corpus_revision)}</code> · suite <code>{_escape(run.suite_revision)}</code> · evidence policy <code>{_escape(run.evidence_policy_revision)}</code></p>
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


def write_corpus_execution_html(
    result: CorpusExecutionResult,
    path: str | Path,
) -> None:
    Path(path).write_text(render_corpus_execution_html(result), encoding="utf-8")


def _case_section(case: Any) -> str:
    trials = "\n".join(_trial_section(trial) for trial in case.trials)
    return (
        f'<section class="case"><h3>Case <code>{_escape(case.case_id)}</code> '
        f"{_badge(case.status)}</h3>"
        f"<p>Revision <code>{_escape(case.case_revision)}</code> · score {_score(case.score)} · "
        f"{case.duration_ms} ms across {len(case.trials)} trial(s)</p>{trials}</section>"
    )


def _trial_section(trial: Any) -> str:
    assertions = "\n".join(_assertion_row(assertion) for assertion in trial.assertions)
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
    return (
        f'<section class="trial"><h4>Trial {trial.trial_number} {_badge(trial.status)}</h4>'
        f"<p>Score {_score(trial.score)} · {trial.duration_ms} ms · "
        f"evidence {'complete' if trial.evidence_complete else 'incomplete'} · "
        f"usage {_escape(usage)} · reason <code>{_escape(trial.code)}</code></p>"
        f"<p>{_escape(trial.message)}</p>"
        f"{output_block}{assertions}</section>"
    )


def _assertion_row(assertion: Any) -> str:
    detail = json.dumps(
        assertion.detail.model_dump(mode="json"),
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
    return (
        '<div class="assertion">'
        f"<div>{_badge(assertion.outcome)}</div>"
        f"<div><strong><code>{_escape(assertion.assertion_id)}</code></strong> "
        f"<span>({_escape(assertion.detail.kind)})</span>"
        f"<p>{_escape(assertion.message)} · score {_score(assertion.score)}</p>"
        f"<pre>{_escape(detail)}</pre></div></div>"
    )


def _badge(status: str) -> str:
    return f'<span class="badge {_escape(status)}">{_escape(status)}</span>'


def _score(score: float | None) -> str:
    return "unavailable" if score is None else f"{score:.2f}"


def _escape(value: Any) -> str:
    return html.escape(str(value), quote=True)

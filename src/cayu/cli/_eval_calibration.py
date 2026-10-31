"""`cayu eval calibrate`: calibrate a suite's AI judge check on fixed evidence.

This is the Developer dashboard's calibration workbench (Evals → Settings → Judges and
calibration) from the terminal, through the same server API. The command picks one
structured AI judge check of one case of a saved suite, as the workbench does, and sends the
fixed evidence and human labels the two files hold. `POST /evals/judge-calibrations/preview`
compiles them into the immutable calibration definition and states the judge work; `POST
/evals/judge-calibrations` runs that definition under a run ID. Calibration calls only the
judge: it never runs the candidate agent or its tools.

The run ID is the run's identity: the server stores each completed report under it, and a
request with the same run ID and definition returns the stored report without calling the
judge again, also after a restart. `--preview` stops after the preview and prints the
command that runs it; that command, and the one an unconfirmed run prints, carry the run ID
and the definition's revision, so they run that reviewed calibration once, replay its
stored report, or are refused when the evidence or labels files have changed since. The
files are the calibration's draft, which the workbench keeps only in its open browser tab.
"""

from __future__ import annotations

import argparse
import json
import secrets
import shlex
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any
from urllib.parse import quote

from cayu.cli._cloud_agent_api import CayuApi, CayuApiError
from cayu.cli._cloud_evals import CloudEvalError, _eval_revision, _positive_count, _text
from cayu.cli._eval_curation import _diagnosed_refusal, _suite_identity

# The dashboard workbench's defaults (`newJudgeCalibrationDraft`): three judge calls, and
# ids derived from the check's.
DEFAULT_TRIALS = 3
_CALIBRATION_ID_SUFFIX = "-calibration"
_EVIDENCE_SOURCE_SUFFIX = "-known-evidence"
# `EVAL_JUDGE_CALIBRATION_MAX_TRIALS` and the portable-id length the ids above must fit.
MAX_TRIALS = 10
_MAX_ID_CHARS = 128
# The server answers a run once every judge call is done, so a calibration waits longer for
# its answer than the other commands do. An answer that doesn't come in time is reported as
# unconfirmed, and the printed run command returns the report the server stored meanwhile.
CALIBRATION_TIMEOUT_SECONDS = 600.0
_STRUCTURED_JUDGE = "structured_model_judge"
_EVIDENCE_FIELDS = frozenset({"final_output", "task", "transcript", "source_id"})
# The server's refusal of a run ID that already ran another definition
# (`cayu.server._judge_calibration_routes`).
_RUN_ID_CONFLICT = "Judge calibration run ID is bound to different reviewed input."


@dataclass(frozen=True, slots=True)
class CalibrationInputs:
    """The fixed evidence and human labels, read before any request."""

    evidence: dict[str, Any]
    labels: dict[str, str]


def add_calibration_parser(
    commands: argparse._SubParsersAction[argparse.ArgumentParser],
) -> argparse.ArgumentParser:
    """Add `calibrate` to `cayu eval`; the caller adds the target and output options."""

    calibrate = commands.add_parser(
        "calibrate",
        help="Calibrate a suite's AI judge check on fixed evidence.",
        description=(
            "Compare repeated scores of a suite's structured AI judge check with human scores, "
            "on fixed evidence, as the Developer dashboard's calibration workbench does. The "
            "command previews the calibration, then runs it under a run ID; --preview stops "
            "after the preview and prints the command that runs it. Calibration calls only "
            "the judge, never the candidate agent or its tools."
        ),
    )
    calibrate.add_argument(
        "--suite", required=True, metavar="SUITE", help="The suite's id from `suites list`."
    )
    calibrate.add_argument("--case", required=True, metavar="CASE_ID", help="The case's id.")
    calibrate.add_argument(
        "--assertion",
        required=True,
        metavar="ASSERTION_ID",
        help="The id of the case's structured AI judge check.",
    )
    calibrate.add_argument(
        "--evidence",
        required=True,
        type=Path,
        metavar="FILE",
        help=(
            "A JSON object with the fixed evidence: final_output, the candidate answer to "
            "judge; task, the request it answered (default: the case's name); transcript, "
            "only for a check that judges the transcript; and source_id, where the evidence "
            f"came from (default: ASSERTION_ID{_EVIDENCE_SOURCE_SUFFIX})."
        ),
    )
    calibrate.add_argument(
        "--labels",
        required=True,
        type=Path,
        metavar="FILE",
        help=(
            "A JSON object with a human score from 0 to 1 for each criterion of the check's "
            'rubric, by criterion id, such as {"correctness": 1, "tone": 0.5}.'
        ),
    )
    calibrate.add_argument(
        "--trials",
        type=_trials,
        default=DEFAULT_TRIALS,
        metavar="N",
        help=f"Judge calls over the same evidence, 1 to {MAX_TRIALS} (default {DEFAULT_TRIALS}).",
    )
    calibrate.add_argument(
        "--calibration-id",
        type=_portable,
        metavar="ID",
        help=f"The calibration's id (default: ASSERTION_ID{_CALIBRATION_ID_SUFFIX}).",
    )
    calibrate.add_argument(
        "--preview",
        action="store_true",
        help="Preview the calibration and print the command that runs it; call no judge.",
    )
    calibrate.add_argument(
        "--run-id",
        type=_portable,
        metavar="ID",
        help=(
            "The run's identity. Running again with it returns the report the server stored "
            "for it rather than calling the judge again. Without one a fresh run ID is printed "
            "in the run command."
        ),
    )
    calibrate.add_argument(
        "--expected-definition-revision",
        type=_eval_revision,
        metavar="REV",
        help=(
            "Run a previewed calibration: its definition revision, from the run command. "
            "The command refuses to run when the evidence or labels have changed since."
        ),
    )
    return calibrate


def read_calibration_inputs(args: argparse.Namespace) -> CalibrationInputs:
    """Check the options and read both files, before any request."""

    if args.preview and args.expected_definition_revision is not None:
        raise ValueError("--preview previews a new calibration; leave out the expected revision.")
    evidence = _json_object(args.evidence, "--evidence")
    unknown = sorted(set(evidence) - _EVIDENCE_FIELDS)
    if unknown:
        raise ValueError(
            f"--evidence {args.evidence} has unknown fields: {', '.join(unknown)}. It takes "
            f"{', '.join(sorted(_EVIDENCE_FIELDS))}."
        )
    if "final_output" not in evidence:
        raise ValueError(f"--evidence {args.evidence} needs final_output, the answer to judge.")
    for name in ("final_output", "task", "source_id"):
        if name in evidence and not isinstance(evidence[name], str):
            raise ValueError(f"--evidence {args.evidence}: {name} must be a string.")
    if evidence.get("transcript") is not None and not isinstance(evidence["transcript"], str):
        raise ValueError(f"--evidence {args.evidence}: transcript must be a string or null.")
    # Scores are read as the decimal text they were written with, never as binary floats.
    labels = _json_object(args.labels, "--labels", numbers_as_text=True)
    return CalibrationInputs(
        evidence=evidence,
        labels={
            criterion: _score(score, criterion, args.labels) for criterion, score in labels.items()
        },
    )


def calibrate(
    api: CayuApi,
    args: argparse.Namespace,
    inputs: CalibrationInputs,
    *,
    target: Sequence[str],
) -> dict[str, Any]:
    """Preview the calibration, then run it under ``--run-id`` (or a fresh run ID).

    With ``--expected-definition-revision`` (a run command) the preview's definition must
    still be that revision. With it or ``--run-id``, the definition is sent whatever the
    preview's readiness, so the server returns a report it already stored for the run ID even
    when the check could not run now.
    """

    target_key, case, assertion = _judge_check(api, args)
    draft = _draft(args, inputs, target_key=target_key, case=case, assertion=assertion)
    preview = api.post(
        "/evals/judge-calibrations/preview", {"draft": draft}, resource="judge calibration"
    )
    definition = preview.get("definition") if isinstance(preview, dict) else None
    if (
        not isinstance(definition, dict)
        or not isinstance(definition.get("revision"), str)
        or type(preview.get("ready")) is not bool
    ):
        raise CloudEvalError(
            "api_response_invalid", "The server's judge calibration preview is invalid."
        )
    revision = definition["revision"]
    expected = args.expected_definition_revision
    if expected is not None and expected != revision:
        raise CloudEvalError(
            "calibration_changed",
            "The evidence or labels changed after this calibration was previewed, so no judge "
            "was called. Preview it again.",
            details={"expected_definition_revision": expected, "definition_revision": revision},
        )
    run_id = args.run_id or f"cayu-cli-calibration-{secrets.token_hex(16)}"
    command = [
        *_calibration_command(args, draft, target),
        *("--run-id", run_id, "--expected-definition-revision", revision),
    ]
    # A run ID the caller names may already have a stored report, which the server returns
    # whatever the check's readiness now; only a fresh run ID needs a check that can run.
    named = expected is not None or args.run_id is not None
    commands = {"run": shlex.join(command)} if preview["ready"] or named else {}
    report = None
    if not args.preview:
        if not preview["ready"] and not named:
            raise _not_ready(preview)
        report = _run(api, run_id, definition, preview=preview, commands=commands)
    return {
        "suite": args.suite,
        "target_key": target_key,
        "case_id": case["id"],
        "assertion_id": assertion["id"],
        "run_id": run_id,
        "draft": draft,
        "preview": preview,
        "report": report,
        "commands": commands,
    }


def _judge_check(
    api: CayuApi, args: argparse.Namespace
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    """The suite's eval target, the case and its compiled structured judge check.

    The suite's id names its source, target and revision, so only the suite or corpus
    document holding the cases is read.
    """

    identity = _suite_identity(args.suite)
    revision = quote(identity["revision"], safe="")
    if identity["source"] == "authored":
        document = api.get(f"/evals/suites/{revision}", resource="eval suite")
        cases = document.get("cases") if isinstance(document, dict) else None
    else:
        # A corpus suite is one suite of a stored corpus, which holds every suite's cases.
        document = api.get(f"/evals/corpora/{revision}", resource="eval corpus")
        cases = document.get("cases") if isinstance(document, dict) else None
        if isinstance(cases, list):
            cases = [
                case
                for case in cases
                if isinstance(case, dict) and case.get("suite_id") == identity["suite_id"]
            ]
    if not isinstance(cases, list) or not all(isinstance(case, dict) for case in cases):
        raise CloudEvalError("api_response_invalid", "The server's eval suite is invalid.")
    case = next((item for item in cases if item.get("id") == args.case), None)
    if case is None:
        raise CloudEvalError("not_found", f"Suite {args.suite} has no case {args.case}.")
    assertions = case.get("assertions")
    assertion = next(
        (
            item
            for item in assertions or ()
            if isinstance(item, dict) and item.get("id") == args.assertion
        ),
        None,
    )
    if assertion is None:
        raise CloudEvalError("not_found", f"Case {args.case} has no check {args.assertion}.")
    if assertion.get("kind") != _STRUCTURED_JUDGE:
        raise CloudEvalError(
            "invalid_input",
            f"Check {args.assertion} is a {_text(assertion.get('kind'))} check. Only an AI judge "
            f"with weighted criteria ({_STRUCTURED_JUDGE}) is calibrated.",
        )
    rubric = assertion.get("rubric")
    criteria = rubric.get("criteria") if isinstance(rubric, dict) else None
    if not isinstance(criteria, list) or not all(
        isinstance(item, dict) and isinstance(item.get("id"), str) for item in criteria
    ):
        raise CloudEvalError("api_response_invalid", "The server's judge check is invalid.")
    return identity["target_key"], case, assertion


def _draft(
    args: argparse.Namespace,
    inputs: CalibrationInputs,
    *,
    target_key: str,
    case: Mapping[str, Any],
    assertion: Mapping[str, Any],
) -> dict[str, Any]:
    """The workbench's calibration draft, with what the files leave out as it proposes it."""

    evidence, labels = inputs.evidence, inputs.labels
    criteria = [item["id"] for item in assertion["rubric"]["criteria"]]
    missing = [criterion for criterion in criteria if criterion not in labels]
    unknown = sorted(set(labels) - set(criteria))
    if missing or unknown:
        raise CloudEvalError(
            "invalid_input",
            f"--labels {args.labels} must score exactly the rubric's criteria: "
            f"{', '.join(criteria)}."
            + (f" Missing: {', '.join(missing)}." if missing else "")
            + (f" Not in the rubric: {', '.join(unknown)}." if unknown else ""),
        )
    transcript = evidence.get("transcript")
    judged = assertion.get("evidence") or {}
    if judged.get("include_transcript") is True and transcript is None:
        raise CloudEvalError(
            "invalid_input",
            f"Check {assertion['id']} judges the transcript; add it to --evidence {args.evidence}.",
        )
    if judged.get("include_transcript") is not True and transcript is not None:
        raise CloudEvalError(
            "invalid_input",
            f"Check {assertion['id']} judges only the final output; remove the transcript from "
            f"--evidence {args.evidence}.",
        )
    task = evidence.get("task")
    return {
        "id": args.calibration_id or f"{assertion['id']}{_CALIBRATION_ID_SUFFIX}"[:_MAX_ID_CHARS],
        "target_key": target_key,
        "assertion": dict(assertion),
        "evidence_source_id": evidence.get("source_id")
        or f"{assertion['id']}{_EVIDENCE_SOURCE_SUFFIX}"[:_MAX_ID_CHARS],
        "task": task if task is not None else (case.get("name") or case["id"]),
        "final_output": evidence["final_output"],
        "transcript": transcript,
        "human_criteria": [
            {"criterion_id": criterion, "score": labels[criterion]} for criterion in criteria
        ],
        "trials": args.trials,
    }


def _calibration_command(
    args: argparse.Namespace, draft: Mapping[str, Any], target: Sequence[str]
) -> list[str]:
    """The `calibrate` that sends ``draft`` again, its defaulted id made explicit."""

    return [
        *("cayu", "eval", "calibrate", "--suite", args.suite, "--case", args.case),
        *("--assertion", args.assertion),
        *("--evidence", str(args.evidence), "--labels", str(args.labels)),
        *("--trials", str(draft["trials"]), "--calibration-id", draft["id"]),
        *target,
    ]


def _run(
    api: CayuApi,
    run_id: str,
    definition: Mapping[str, Any],
    *,
    preview: Mapping[str, Any],
    commands: Mapping[str, str],
) -> dict[str, Any]:
    try:
        answer = api.mutate(
            "/evals/judge-calibrations",
            {
                "run_id": run_id,
                "expected_definition_revision": definition["revision"],
                "definition": definition,
            },
            resource="judge calibration",
        )
    except CayuApiError as exc:
        if exc.category == "outcome_unknown":
            raise CloudEvalError(
                "outcome_unknown",
                "The server did not confirm the calibration; it may have run. "
                f"`{commands['run']}` returns the report it stored without calling the judge "
                "again; a run that stopped before storing its report calls the judge again.",
                status_code=exc.status_code,
                details={"run_id": run_id, "commands": dict(commands)},
            ) from None
        if exc.detail == _RUN_ID_CONFLICT:
            raise CloudEvalError(
                "run_id_conflict",
                f"Run ID {run_id} already calibrated other evidence, labels or a different "
                "check, so this calibration called no judge.",
                status_code=exc.status_code,
                details={"run_id": run_id},
            ) from None
        if exc.status_code == 409 and preview.get("ready") is not True:
            # The server stores no report for this run ID and can't run the check now.
            raise _not_ready(preview) from None
        raise
    report = answer.get("report") if isinstance(answer, dict) else None
    if (
        not isinstance(report, dict)
        or report.get("run_id") != run_id
        or not isinstance(report.get("definition"), dict)
        or report["definition"].get("revision") != definition["revision"]
    ):
        raise CloudEvalError("api_response_invalid", "The server did not confirm the calibration.")
    return report


def _not_ready(preview: Mapping[str, Any]) -> CloudEvalError:
    return _diagnosed_refusal(
        "calibration_not_ready",
        "The server can't run this calibration now, so no judge was called",
        preview.get("diagnostics"),
    )


def _json_object(path: Path, option: str, *, numbers_as_text: bool = False) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        reason = exc.strerror if isinstance(exc, OSError) else "not UTF-8 text"
        raise ValueError(
            f"Could not read {option} {path}: {reason or type(exc).__name__}."
        ) from None
    try:
        value = (
            json.loads(text, parse_float=str, parse_int=str)
            if numbers_as_text
            else json.loads(text)
        )
    except json.JSONDecodeError as exc:
        raise ValueError(f"{option} {path} is not JSON: {exc.msg} (line {exc.lineno}).") from None
    if type(value) is not dict:
        raise ValueError(f"{option} {path} must hold one JSON object.")
    return value


def _score(value: object, criterion: str, path: Path) -> str:
    """``value`` in the canonical decimal notation the server takes, such as 1 for 1.0."""

    try:
        number = Decimal(value) if isinstance(value, str) else None
    except InvalidOperation:
        number = None
    if number is None or not number.is_finite() or not 0 <= number <= 1:
        raise ValueError(f"--labels {path}: the score of {criterion} must be a number from 0 to 1.")
    return format(number.normalize(), "f") if number else "0"


def _portable(value: str) -> str:
    from cayu.evals.corpus import _portable_id

    try:
        return _portable_id(value, "ID")
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from None


def _trials(value: str) -> int:
    number = _positive_count(value)
    if number > MAX_TRIALS:
        raise argparse.ArgumentTypeError(f"must be at most {MAX_TRIALS}")
    return number


def render_calibration(answer: Mapping[str, Any]) -> str:
    result = answer["result"]
    preview, report = result["preview"], result["report"]
    check = f"check {_text(result['assertion_id'])} of case {_text(result['case_id'])}"
    if report is not None:
        human = report["definition"]["human_label"]["aggregate_score"]
        lines = [f"Calibrated {check}: human aggregate {_text(human)}"]
        for trial in report["trials"]:
            detail = trial["judgment"].get("detail") or {}
            line = (
                f"Trial {_text(trial['sequence'])}: {_text(trial['judgment'].get('outcome'))}, "
                f"score {_text(detail.get('aggregate_score'))}"
            )
            if trial.get("aggregate_absolute_error") is not None:
                line += f", absolute error {_text(trial['aggregate_absolute_error'])}"
            if trial.get("pass_agreement") is not None:
                line += f", pass agreement {'yes' if trial['pass_agreement'] else 'no'}"
            lines.append(line)
        lines.append(f"Report: {_text(report['revision'])}")
        return "\n".join(lines)
    if not preview["ready"]:
        lines = [f"Calibration of {check} needs attention:"]
        lines += [f"  {_text(item.get('message'))}" for item in preview.get("diagnostics") or ()]
        return "\n".join(lines)
    work = preview["work"]
    line = (
        f"Calibration of {check} is ready: {_text(work['judge_calls'])} judge calls, at most "
        f"{_text(work['max_total_tokens'])} tokens"
    )
    if work.get("max_estimated_cost") is not None:
        line += f", at most {_text(work['max_estimated_cost'])} {_text(work['cost_currency'])}"
    lines = [line + "."]
    if preview.get("candidate_route_relation") == "same_model":
        lines.append("The judge is the candidate's own model; its scores are labeled so.")
    lines.append(f"Run it: {_text(result['commands']['run'])}")
    return "\n".join(lines)

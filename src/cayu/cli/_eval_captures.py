"""`cayu eval capture preview|save|run` and `cayu eval baseline approve --result`: evaluate a
finished session's captured evidence from the terminal.

These are the Developer dashboard's captured evaluation details on a session (**Preview
score**, **Save evaluation**, **Approve baseline** and **Run on current app**), through the
same server API:

- `capture preview` is **Preview score** (`POST /evals/sessions/{id}/evaluation/preview`): the
  retained evidence, the suite and case the capture proposes, or the ones the options name,
  and their score over that evidence. Nothing runs.
- `capture save` is **Save evaluation** (`POST .../evaluation/save`): it records that exact
  captured result. A result is its content, so saving it again changes nothing.
- `capture run` is **Run on current app** (`POST .../evaluation/launch`): it runs the session's
  input afresh on the current release under an idempotency key, pinned to the reviewed
  candidate and execution profile, so its continue command answers with the run it admitted.
- `baseline approve --result REVISION` approves a saved result, such as a captured one, as its
  suite's baseline, with `cayu cloud eval baseline`'s decision (`GET /evals/results/{rev}`,
  then `POST /evals/results/{rev}/baseline`).
"""

from __future__ import annotations

import argparse
import json
import secrets
import shlex
from collections.abc import Callable, Mapping, Sequence
from typing import Any
from urllib.parse import quote

from cayu.cli._cloud_agent_api import CayuApi, CayuApiError
from cayu.cli._cloud_evals import (
    _DEFAULT_CURRENCY,
    _IDEMPOTENCY_CONFLICT,
    _PROFILE_REFUSALS,
    CloudEvalError,
    _cost_amount,
    _currency,
    _decide_baseline,
    _eval_revision,
    _operation_id,
    _positive_count,
    _published_target,
    _run_summary,
    _text,
)
from cayu.cli._eval_curation import _assertion
from cayu.cli._eval_scenarios import _LIMITS
from cayu.evals.trial_policy import EvalSuiteTrialPolicyV1

# The dashboard's Run on current app defaults (`DEFAULT_FRESH_EXECUTION`).
DEFAULT_TRIALS = 1
DEFAULT_MAX_CONCURRENCY = 1
DEFAULT_TRIAL_TIMEOUT_SECONDS = 300
# The draft options and the candidate field each one replaces.
_DRAFT_OPTIONS = (
    ("suite_id", "--suite-id"),
    ("suite_name", "--suite-name"),
    ("case_id", "--case-id"),
    ("case_name", "--case-name"),
)


def add_capture_parser(
    commands: argparse._SubParsersAction[argparse.ArgumentParser],
) -> list[tuple[argparse.ArgumentParser, Callable[[Mapping[str, Any]], str]]]:
    """Add `capture` to `cayu eval`; return its parsers with their text renderers."""

    capture = commands.add_parser(
        "capture",
        help="Score, save and rerun a finished session's captured evaluation.",
        description=(
            "Score a finished session's retained evidence, save that captured result, and "
            "run the session's input afresh on the current app, as the captured evaluation "
            "details on a session in the Developer dashboard do."
        ),
    )
    actions = capture.add_subparsers(dest="eval_capture_command", required=True)
    preview = actions.add_parser(
        "preview",
        help="Score a session's retained evidence.",
        description=(
            "Score a finished session's retained evidence with the checks its capture "
            "proposes, or with --assertion, as Preview score does: the evidence, the suite "
            "and case, their score, and whether the session can run afresh or as a scenario. "
            "Nothing runs or is saved."
        ),
    )
    save = actions.add_parser(
        "save",
        help="Save a session's captured result.",
        description=(
            "Record the captured result Preview score shows, as Save evaluation does. Saving "
            "it again changes nothing. `cayu eval baseline approve --result REVISION` then "
            "makes it its suite's baseline."
        ),
    )
    run = actions.add_parser(
        "run",
        help="Run a session's input afresh on the current app.",
        description=(
            "Run the session's input afresh on the current release, with the reviewed checks, "
            "as Run on current app does, under an idempotency key. The command returns once "
            "the run is admitted; `cayu eval runs show RUN_ID` follows it."
        ),
    )
    for parser in (preview, save, run):
        parser.add_argument("session_id", metavar="SESSION_ID")
        _add_draft_options(parser)
    for parser in (save, run):
        parser.add_argument(
            "--expected-candidate-revision",
            type=_eval_revision,
            metavar="REV",
            help=(
                "Act only on this reviewed candidate, from a printed command. The session's "
                "evidence or the options changing since refuses it."
            ),
        )
    _add_run_options(run)
    return [(preview, render_capture), (save, render_saved), (run, render_capture_run)]


def add_result_baseline_option(subject: argparse._ActionsContainer) -> None:
    """Add `--result`, the saved result `baseline approve` approves instead of a run's."""

    subject.add_argument(
        "--result",
        type=_eval_revision,
        metavar="REVISION",
        help=(
            "Approve a saved result instead of a run's, such as the captured result `cayu eval "
            "capture save` records."
        ),
    )


def _add_draft_options(parser: argparse.ArgumentParser) -> None:
    for field, option in _DRAFT_OPTIONS:
        noun = field.replace("_", " ")
        parser.add_argument(
            option, metavar="VALUE", help=f"The {noun} (default: the one the capture proposes)."
        )
    parser.add_argument(
        "--assertion",
        action="append",
        type=_assertion,
        metavar="JSON",
        help=(
            "One check as a JSON object, in the corpus assertion format; repeat for more. "
            "Without one, the checks the capture proposes."
        ),
    )


def _add_run_options(parser: argparse.ArgumentParser) -> None:
    for option, default, noun in (
        ("--trials", DEFAULT_TRIALS, "Trials"),
        ("--max-concurrency", DEFAULT_MAX_CONCURRENCY, "Trials running at once"),
        ("--trial-timeout-seconds", DEFAULT_TRIAL_TIMEOUT_SECONDS, "Each trial's time limit"),
    ):
        parser.add_argument(
            option,
            type=_positive_count,
            default=default,
            metavar="N",
            help=f"{noun} (default {default}).",
        )
    parser.add_argument(
        "--max-steps", type=_positive_count, metavar="N", help="Each trial's model step limit."
    )
    for field, option in _LIMITS:
        parser.add_argument(
            option,
            dest=field,
            type=_positive_count,
            metavar="N",
            help=f"A per-trial limit on {field.removeprefix('max_').replace('_', ' ')}.",
        )
    parser.add_argument(
        "--max-cost",
        type=_cost_amount,
        metavar="AMOUNT",
        help="Interrupt each trial once its observed cost reaches AMOUNT, such as 0.50.",
    )
    parser.add_argument(
        "--currency",
        type=_currency,
        metavar="CODE",
        help=f"The currency of --max-cost (default {_DEFAULT_CURRENCY}).",
    )
    parser.add_argument(
        "--operation-id",
        type=_operation_id,
        metavar="ID",
        help=(
            "The run's idempotency key. Without one a fresh key is printed in the command that "
            "continues the run."
        ),
    )
    parser.add_argument(
        "--expected-execution-profile-revision",
        type=_eval_revision,
        metavar="REV",
        help=(
            "Repeat an earlier admission: the execution profile it was made under, from its "
            "continue command."
        ),
    )


def check_capture_options(args: argparse.Namespace) -> None:
    """Check `capture`'s options before any request."""

    if args.eval_capture_command != "run":
        return
    if args.currency is not None and args.max_cost is None:
        raise ValueError("--currency is the currency of --max-cost; pass it with --max-cost.")
    if args.expected_execution_profile_revision is not None and (
        args.operation_id is None or args.expected_candidate_revision is None
    ):
        raise ValueError(
            "--expected-execution-profile-revision repeats an earlier admission; pass it with "
            "that admission's --operation-id and --expected-candidate-revision."
        )


def run_capture_command(
    api: CayuApi, args: argparse.Namespace, *, target: Sequence[str]
) -> dict[str, Any]:
    action = args.eval_capture_command
    checked = _preview(api, args)
    if action == "preview":
        return _previewed(args, checked, target=target)
    candidate = checked["candidate"]
    expected = args.expected_candidate_revision
    if expected is not None and expected != candidate["revision"]:
        raise _capture_changed(args, candidate, target=target)
    if action == "save":
        return _save(api, args, candidate, target=target)
    return _run(api, args, checked, target=target)


def approve_result_baseline(
    api: CayuApi, args: argparse.Namespace, *, target: Sequence[str]
) -> dict[str, Any]:
    """Approve a saved result as its suite's baseline, as Approve baseline does for a saved
    captured evaluation."""

    revision = args.result
    record, current = _result(api, revision)
    decided = _decide_baseline(
        api,
        args,
        revision=revision,
        current=current,
        current_baseline=lambda: _result(api, revision)[1],
        subject=f"eval result {revision}",
        suite_id=record.get("suite_id"),
        command=["cayu", "eval", "baseline", "approve", "--result", revision, *target],
        details={"result_revision": revision},
    )
    return {"record": record, **decided}


def _result(api: CayuApi, revision: str) -> tuple[dict[str, Any], dict[str, Any] | None]:
    detail = api.get(f"/evals/results/{quote(revision, safe='')}", resource="eval result")
    record = detail.get("record") if isinstance(detail, dict) else None
    baseline = detail.get("baseline") if isinstance(detail, dict) else None
    if (
        not isinstance(record, dict)
        or record.get("revision") != revision
        or (baseline is not None and type((baseline or {}).get("generation")) is not int)
    ):
        raise CloudEvalError("api_response_invalid", "The server's eval result is invalid.")
    return record, baseline


def _preview(api: CayuApi, args: argparse.Namespace) -> dict[str, Any]:
    """The captured evaluation the options describe: the capture's own proposal, or its draft
    with the named suite, case and checks, as the dashboard edits it."""

    path = f"/evals/sessions/{quote(args.session_id, safe='')}/evaluation/preview"
    proposed = _checked(api.post(path, {"draft": None}, resource="captured evaluation"))
    if args.assertion is None and all(getattr(args, field) is None for field, _ in _DRAFT_OPTIONS):
        return proposed
    candidate = proposed["candidate"]
    suite, case = candidate["suite"], candidate["case"]
    suite_id = args.suite_id or suite.get("id")
    draft = {
        "expected_baseline_revision": proposed["baseline_revision"],
        "suite": {
            "id": suite_id,
            "name": args.suite_name or suite.get("name"),
            "description": suite.get("description"),
        },
        "case": {
            "id": args.case_id or case.get("id"),
            "suite_id": suite_id,
            "name": args.case_name or case.get("name"),
            "description": case.get("description"),
            "assertions": args.assertion or case.get("assertions"),
        },
    }
    return _checked(api.post(path, {"draft": draft}, resource="captured evaluation"))


def _checked(answer: Any) -> dict[str, Any]:
    candidate = answer.get("candidate") if isinstance(answer, dict) else None
    if (
        not isinstance(candidate, dict)
        or not isinstance(candidate.get("revision"), str)
        or not isinstance(candidate.get("target_key"), str)
        or not isinstance(candidate.get("suite"), dict)
        or not isinstance(candidate.get("case"), dict)
        or not isinstance(answer.get("baseline_revision"), str)
        or not isinstance(answer.get("runnable_conversion"), dict)
    ):
        raise CloudEvalError(
            "api_response_invalid", "The server's captured evaluation preview is invalid."
        )
    return answer


def _previewed(
    args: argparse.Namespace, checked: Mapping[str, Any], *, target: Sequence[str]
) -> dict[str, Any]:
    revision = checked["candidate"]["revision"]
    commands = {
        "save": _command(args, "save", ["--expected-candidate-revision", revision], target),
    }
    if checked["runnable_conversion"].get("available") is True:
        commands["run"] = _command(args, "run", ["--expected-candidate-revision", revision], target)
    return {
        "session_id": args.session_id,
        "baseline_revision": checked["baseline_revision"],
        "candidate": checked["candidate"],
        "captured_score": checked.get("captured_score"),
        "runnable_conversion": checked["runnable_conversion"],
        "scenario_conversion": checked.get("scenario_conversion"),
        "commands": commands,
    }


def _save(
    api: CayuApi,
    args: argparse.Namespace,
    candidate: Mapping[str, Any],
    *,
    target: Sequence[str],
) -> dict[str, Any]:
    from cayu.evals.read_views import EvalSuiteViewIdentity

    saved = api.post(
        f"/evals/sessions/{quote(args.session_id, safe='')}/evaluation/save",
        {"candidate": candidate, "expected_candidate_revision": candidate["revision"]},
        resource="captured evaluation",
    )
    record = saved.get("record") if isinstance(saved, dict) else None
    if not isinstance(record, dict) or not all(
        isinstance(record.get(key), str) for key in ("revision", "corpus_revision", "suite_id")
    ):
        raise CloudEvalError(
            "api_response_invalid", "The server did not confirm the captured evaluation."
        )
    revision = record["revision"]
    suite = EvalSuiteViewIdentity(
        source="corpus",
        target_key=candidate["target_key"],
        revision=record["corpus_revision"],
        suite_id=record["suite_id"],
    ).id
    return {
        "session_id": args.session_id,
        "candidate_revision": candidate["revision"],
        "record": record,
        "suite": suite,
        "commands": {
            "baseline": shlex.join(
                ["cayu", "eval", "baseline", "approve", "--result", revision, *target]
            ),
            "show": shlex.join(["cayu", "eval", "suites", "show", suite, *target]),
        },
    }


def _run(
    api: CayuApi,
    args: argparse.Namespace,
    checked: Mapping[str, Any],
    *,
    target: Sequence[str],
) -> dict[str, Any]:
    """Admit one fresh run of the reviewed candidate on the current release.

    The continue command repeats the admission with its key, candidate and execution profile,
    so the server answers with the run the key admitted, also after the target's current
    profile changed under the same release. A deploy changes the candidate itself, and its
    continue command then stops at `_capture_changed`.
    """

    candidate = checked["candidate"]
    conversion = checked["runnable_conversion"]
    if conversion.get("available") is not True:
        raise CloudEvalError(
            "capture_not_runnable",
            f"Session {args.session_id}'s input can't be run afresh "
            f"({_text(conversion.get('reason_code'))}), so nothing was run.",
            details={"reason_code": conversion.get("reason_code")},
        )
    target_key = candidate["target_key"]
    continuing = args.expected_execution_profile_revision is not None
    entry = _published_target(api, target_key, require_ready=not continuing)
    current = (entry.get("execution_profile") or {}).get("revision")
    profile = args.expected_execution_profile_revision or current
    if not isinstance(profile, str):
        raise CloudEvalError(
            "api_response_invalid", "The server's eval target names no execution profile."
        )
    operation_id = args.operation_id or f"cayu-cli-eval-capture-{secrets.token_hex(16)}"
    pins = [
        *("--operation-id", operation_id),
        *("--expected-candidate-revision", candidate["revision"]),
        *("--expected-execution-profile-revision", profile),
    ]
    commands = {
        "continue": _command(args, "run", [*_settings_arguments(args), *pins], target),
        "list": shlex.join(["cayu", "eval", "runs", "list", "--target-key", target_key, *target]),
    }
    body: dict[str, Any] = {
        "candidate": candidate,
        "expected_candidate_revision": candidate["revision"],
        "expected_execution_profile_revision": profile,
        # The suite's trial policy bounds the run's concurrency, so it carries the one asked
        # for; without one the server would allow a single trial at a time.
        "trial_request": {
            "trials": args.trials,
            "timeout_seconds": args.trial_timeout_seconds,
            "trial_policy": EvalSuiteTrialPolicyV1.create(
                trial_count=args.trials, max_concurrency=args.max_concurrency
            ).model_dump(mode="json"),
        },
        "max_concurrency": args.max_concurrency,
    }
    if args.max_steps is not None:
        body["max_steps"] = args.max_steps
    limits = {
        field: getattr(args, field) for field, _ in _LIMITS if getattr(args, field) is not None
    }
    if limits:
        body["limits"] = {"scope": "run", **limits}
    if args.max_cost is not None:
        body["cost_budget"] = {
            "max_estimated_cost": str(args.max_cost),
            "currency": args.currency or _DEFAULT_CURRENCY,
        }
    try:
        launched = api.mutate(
            f"/evals/sessions/{quote(args.session_id, safe='')}/evaluation/launch",
            body,
            resource="captured evaluation run",
            operation_id=operation_id,
        )
    except CayuApiError as exc:
        refusal = _launch_refusal(
            exc, operation_id=operation_id, profile_changed=profile != current, commands=commands
        )
        if refusal is None:
            raise
        raise refusal from None
    captured = launched.get("captured") if isinstance(launched, dict) else None
    record = captured.get("record") if isinstance(captured, dict) else None
    if not isinstance(record, dict):
        raise CloudEvalError(
            "api_response_invalid", "The server did not confirm the captured evaluation run."
        )
    run = _run_summary(launched.get("run"))
    commands["show"] = shlex.join(["cayu", "eval", "runs", "show", run["id"], *target])
    return {
        "session_id": args.session_id,
        "operation_id": operation_id,
        "candidate_revision": candidate["revision"],
        "execution_profile_revision": profile,
        "record": record,
        "run": run,
        "commands": commands,
    }


def _capture_changed(
    args: argparse.Namespace, candidate: Mapping[str, Any], *, target: Sequence[str]
) -> CloudEvalError:
    """The refusal of a reviewed candidate the session no longer gives.

    The candidate records the release it is evaluated on, so a deploy changes it too, and the
    server checks it before it replays a launch: a continue command can't then tell whether
    its key admitted a run.
    """

    details: dict[str, object] = {
        "expected_candidate_revision": args.expected_candidate_revision,
        "candidate_revision": candidate["revision"],
    }
    if args.eval_capture_command == "run" and args.operation_id is not None:
        listing = shlex.join(
            ["cayu", "eval", "runs", "list", "--target-key", candidate["target_key"], *target]
        )
        return CloudEvalError(
            "capture_changed",
            f"Session {args.session_id}'s captured evaluation changed since operation ID "
            f"{args.operation_id} was reviewed (a deploy changes it too), so the launch can't be "
            "repeated, and whether it admitted a run is unknown. "
            f"`{listing}` lists the target's runs; run the session again only if none is its "
            "run.",
            details={
                **details,
                "operation_id": args.operation_id,
                "commands": {"list": listing},
            },
        )
    return CloudEvalError(
        "capture_changed",
        f"Session {args.session_id}'s captured evaluation changed after it was reviewed (a "
        "deploy changes it too), so nothing was saved or run. Preview it again.",
        details=details,
    )


def _launch_refusal(
    exc: CayuApiError,
    *,
    operation_id: str,
    profile_changed: bool,
    commands: Mapping[str, str],
) -> CloudEvalError | None:
    """What a launch's failure means for its operation ID, when that is known."""

    if exc.category == "outcome_unknown":
        return CloudEvalError(
            "outcome_unknown",
            "The server did not confirm the run on the current app; it may have admitted it. "
            f"`{commands['continue']}` admits nothing new if it did.",
            status_code=exc.status_code,
            details={"operation_id": operation_id, "commands": dict(commands)},
        )
    if exc.detail == _IDEMPOTENCY_CONFLICT:
        return CloudEvalError(
            "operation_id_conflict",
            f"Operation ID {operation_id} already admitted a different run (its checks or run "
            "options have changed since), so this one admitted nothing. "
            f"`{commands['list']}` lists the run it admitted.",
            status_code=exc.status_code,
            details={"operation_id": operation_id, "commands": {"list": commands["list"]}},
        )
    if profile_changed and exc.detail in _PROFILE_REFUSALS:
        # The server replays an admitted launch before checking it against its current
        # release, so its release refusing the exact request means this key admitted nothing.
        return CloudEvalError(
            "eval_release_changed",
            "The server's release changed before this run was admitted, so there is no run to "
            f"continue ({exc}). Run the session again without --operation-id and "
            "--expected-execution-profile-revision.",
            status_code=exc.status_code,
            details={"operation_id": operation_id},
        )
    return None


def _command(
    args: argparse.Namespace, action: str, extra: Sequence[str], target: Sequence[str]
) -> str:
    """The `capture ACTION` for the same session and draft options, with ``extra``."""

    command = ["cayu", "eval", "capture", action, args.session_id]
    for field, option in _DRAFT_OPTIONS:
        value = getattr(args, field)
        if value is not None:
            command += [option, value]
    for assertion in args.assertion or ():
        command += ["--assertion", json.dumps(assertion, ensure_ascii=False, sort_keys=True)]
    return shlex.join([*command, *extra, *target])


def _settings_arguments(args: argparse.Namespace) -> list[str]:
    """The run options that send the same settings again; defaults are left out."""

    arguments: list[str] = []
    for option, value, default in (
        ("--trials", args.trials, DEFAULT_TRIALS),
        ("--max-concurrency", args.max_concurrency, DEFAULT_MAX_CONCURRENCY),
        ("--trial-timeout-seconds", args.trial_timeout_seconds, DEFAULT_TRIAL_TIMEOUT_SECONDS),
    ):
        if value != default:
            arguments += [option, str(value)]
    if args.max_steps is not None:
        arguments += ["--max-steps", str(args.max_steps)]
    for field, option in _LIMITS:
        value = getattr(args, field)
        if value is not None:
            arguments += [option, str(value)]
    if args.max_cost is not None:
        arguments += ["--max-cost", str(args.max_cost)]
        arguments += ["--currency", args.currency or _DEFAULT_CURRENCY]
    return arguments


def render_capture(answer: Mapping[str, Any]) -> str:
    result = answer["result"]
    candidate, score = result["candidate"], result["captured_score"] or {}
    case = candidate["case"]
    lines = [
        f"Case {_text(case.get('id'))} of suite {_text(candidate['suite'].get('id'))}: "
        f"{_text(score.get('status'))}"
        + ("" if score.get("score") is None else f", score {score['score']:g}"),
    ]
    for item in score.get("assertions") or ():
        lines.append(f"  {_text(item.get('assertion_id'))}: {_text(item.get('outcome'))}")
    for name, label in (("save", "Save it"), ("run", "Run it on the current app")):
        if name in result["commands"]:
            lines.append(f"{label}: {_text(result['commands'][name])}")
    return "\n".join(lines)


def render_saved(answer: Mapping[str, Any]) -> str:
    result = answer["result"]
    record = result["record"]
    return (
        f"Saved captured result {_text(record['revision'])}: {_text(record.get('status'))}.\n"
        f"Approve it as the baseline: {_text(result['commands']['baseline'])}"
    )


def render_capture_run(answer: Mapping[str, Any]) -> str:
    result = answer["result"]
    run = result["run"]
    return (
        f"Run {_text(run['id'])} on the current app: {_text(run['status'])}.\n"
        f"Follow it: {_text(result['commands']['show'])}"
    )

"""`cayu eval scenario`: check and run production scenarios from the terminal.

These are the Developer dashboard's production scenario actions (Evals → Settings →
Production scenarios, and **Production scenario** on a captured session), through the same
server API. A scenario comes from a JSON file (a downloaded revision or a draft), from a
finished session (`POST /evals/sessions/{id}/evaluation/preview`, the scenario the session's
capture proposes) or from the server's library (`GET /evals/scenarios/{revision}`).

`check` is **Check readiness** (`POST /evals/scenarios/preview`): it compiles the scenario
into its immutable revision and preflights it, with the launch settings, against the
server's current authority, running nothing; `--save` then saves that revision to the
library (`POST /evals/scenarios`), as **Save** does. `run` checks and saves the scenario, then
admits one run of the exact binding the check reviewed (`POST
/evals/scenarios/{revision}/runs`) under an idempotency key. A ready check prints the run
command and a run prints the command that continues it. Both carry a key, the scenario's
revision and the reviewed binding and execution profile, and send that admission before
anything else. So repeating either answers with the run the key admitted, also after a
deploy changed the current binding, and a key that admitted nothing admits a run only under
that reviewed binding. When the server doesn't hold the scenario yet, the command saves it
from its source first, and refuses a source that no longer holds the checked revision.

`prepare-fixture` is **Prepare fixture** (`POST /evals/scenarios/artifacts/{id}/materialize`)
followed by **Save**: the scenario revision whose file requirement uses the prepared fixture
is saved, so the run command it prints names that revision. `approve` and `deny` are the run
page's **Approve** and **Deny** of a trial awaiting approval (`POST
/evals/runs/{id}/scenario-approval`), decided over the progress revision just read.
"""

from __future__ import annotations

import argparse
import json
import secrets
import shlex
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import quote

from cayu.cli._cloud_agent_api import CayuApi, CayuApiError, CayuApiNotFound
from cayu.cli._cloud_evals import (
    _DEFAULT_CURRENCY,
    _IDEMPOTENCY_CONFLICT,
    MAX_LIST_LIMIT,
    CloudEvalError,
    _cost_amount,
    _currency,
    _eval_revision,
    _list_limit,
    _operation_id,
    _positive_count,
    _run_path,
    _run_summary,
    _text,
)
from cayu.cli._cloud_settings import _cell, _table
from cayu.cli._eval_curation import _diagnosed_refusal, _page, _query

# The dashboard's launch settings defaults (`DEFAULT_SCENARIO_SETTINGS`).
DEFAULT_TRIALS = 1
DEFAULT_MAX_CONCURRENCY = 1
DEFAULT_TRIAL_TIMEOUT_SECONDS = 300
# A scenario document's own fields that a draft leaves out, as the dashboard's editor drops
# them (`draftFromScenario`): the server computes both.
_DOCUMENT_ONLY_FIELDS = ("revision", "schema_version")
# Per-run limit options and the `RunLimits` field each sets.
_LIMITS = (
    ("max_total_tokens", "--max-total-tokens"),
    ("max_tool_calls", "--max-tool-calls"),
    ("max_elapsed_seconds", "--max-elapsed-seconds"),
)
# The server's refusals of a launch (`cayu.server._scenario_routes`): the reviewed binding is
# no longer the current one, the scenario can't be launched now, and it isn't saved.
_BINDING_CHANGED = "Eval scenario launch binding changed after review."
_SCENARIO_NOT_FOUND = "Eval scenario not found."
_LAUNCH_NOT_READY = "Eval scenario launch requirements are not currently ready."
# The server's refusal of a decision over a progress revision that is no longer current
# (`cayu.server._evaluation_run_routes`).
_APPROVAL_CHANGED = "Scenario approval checkpoint changed before submission."
# Decisions sent while other trials of the run keep changing its progress revision.
_DECISION_ATTEMPTS = 3


def add_scenario_parser(
    commands: argparse._SubParsersAction[argparse.ArgumentParser],
) -> list[tuple[argparse.ArgumentParser, Callable[[Mapping[str, Any]], str]]]:
    """Add `scenario` to `cayu eval`; return each action's parser with its text renderer,
    for the caller to add the target and output options."""

    scenario = commands.add_parser(
        "scenario",
        help="Check, save, prepare, run and approve production scenarios.",
        description=(
            "List and show saved scenarios, check a scenario's readiness and save it, prepare "
            "its file fixtures and run it, as the Developer dashboard's production scenario "
            "editor does, and approve or deny the checkpoints a scenario run awaits, as its "
            "run page does."
        ),
    )
    actions = scenario.add_subparsers(dest="eval_scenario_command", required=True)

    listing = actions.add_parser(
        "list",
        help="List saved scenarios.",
        description="List an eval target's saved scenario revisions, newest first.",
    )
    listing.add_argument(
        "--target-key", metavar="KEY", help="The eval target (default: the server's default)."
    )
    listing.add_argument("--scenario-id", metavar="ID", help="Only this scenario's revisions.")
    listing.add_argument(
        "--limit",
        type=_list_limit,
        metavar="N",
        help=f"Scenarios per page, at most {MAX_LIST_LIMIT} (default: the server's).",
    )
    listing.add_argument("--cursor", help="The next page: the previous page's next_cursor.")

    show = actions.add_parser(
        "show",
        help="Show one saved scenario revision.",
        description=(
            "Show one saved scenario revision. Its `result.scenario` is the scenario file "
            "`check --file` and `run --file` read."
        ),
    )
    show.add_argument("revision", type=_eval_revision, metavar="REVISION")

    check = actions.add_parser(
        "check",
        help="Check a scenario's readiness to run.",
        description=(
            "Compile a scenario into its immutable revision and check, with the launch "
            "settings, whether the server can run it now, as Check readiness does. Nothing "
            "runs. A ready check prints the command that runs that revision."
        ),
    )
    _add_source_options(check)
    _add_settings_options(check)
    check.add_argument(
        "--save",
        action="store_true",
        help="Also save the checked revision to the server's scenario library.",
    )

    run = actions.add_parser(
        "run",
        help="Check, save and run a scenario.",
        description=(
            "Check a scenario's readiness, save its revision and admit one run of it under an "
            "idempotency key, as Run scenario does. The command returns once the run is "
            "admitted; `cayu eval runs show RUN_ID` follows it."
        ),
    )
    _add_source_options(run)
    _add_settings_options(run)
    run.add_argument(
        "--operation-id",
        type=_operation_id,
        metavar="ID",
        help=(
            "The run's idempotency key. Without one a fresh key is printed in the command "
            "that continues the run."
        ),
    )
    run.add_argument(
        "--expected-scenario-revision",
        type=_eval_revision,
        metavar="REV",
        help=(
            "Run only this checked revision of the --file or --session scenario, from a "
            "check's run command."
        ),
    )
    for name in ("binding", "execution-profile"):
        run.add_argument(
            f"--expected-{name}-revision",
            type=_eval_revision,
            metavar="REV",
            help=(
                f"The {name.replace('-', ' ')} revision a check reviewed, from a printed run "
                "or continue command. The run is admitted under that binding or not at all."
            ),
        )

    prepare = actions.add_parser(
        "prepare-fixture",
        help="Prepare a reusable fixture for one of a scenario's files.",
        description=(
            "Copy the retained file one of the scenario's file requirements names into a "
            "reusable fixture of its environment, as Prepare fixture does, and save the "
            "scenario revision that uses the fixture. The file comes from the requirement's "
            "own reference, or from --artifact REQUIREMENT=ARTIFACT_ID. A ready answer prints "
            "the command that runs the saved revision."
        ),
    )
    prepare.add_argument(
        "requirement_id", metavar="REQUIREMENT_ID", help="The scenario's file requirement."
    )
    _add_source_options(prepare)
    _add_settings_options(prepare)

    decisions = []
    for decision, verb in (("approve", "Approve"), ("deny", "Deny")):
        decide = actions.add_parser(
            decision,
            help=f"{verb} the tool call a scenario run's trial awaits approval for.",
            description=(
                f"{verb} the fresh tool call a scenario run's trial waits on at an approval "
                "checkpoint, as the run page's Approve and Deny do. `cayu eval runs show "
                "RUN_ID` shows which trials await approval, for which tool and checkpoint. A "
                "checkpoint takes one decision."
            ),
        )
        decide.add_argument("run_id", metavar="RUN_ID")
        decide.add_argument(
            "--trial",
            type=_positive_count,
            metavar="N",
            help="The trial awaiting approval (default: the only one that does).",
        )
        decide.add_argument(
            "--event",
            metavar="EVENT_ID",
            help=(
                "Decide only while the trial awaits this approval checkpoint, the event "
                "`runs show` names. Every trial of a run has the same checkpoints."
            ),
        )
        decide.add_argument(
            "--trial-session",
            metavar="SESSION_ID",
            help=(
                "Decide only while the trial runs in this session, from a decision's continue "
                "command. A trial that starts over runs in a new session."
            ),
        )
        decide.add_argument("--reason", metavar="TEXT", help="Why; recorded with the decision.")
        decisions.append((decide, render_decision))
    return [
        (listing, render_scenarios),
        (show, render_scenario),
        (check, render_check),
        (run, render_run),
        (prepare, render_prepared),
        *decisions,
    ]


def _add_source_options(parser: argparse.ArgumentParser) -> None:
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--file",
        type=Path,
        metavar="FILE",
        help="A scenario JSON file: a saved revision `scenario show` printed, or a draft.",
    )
    source.add_argument(
        "--session",
        metavar="SESSION_ID",
        help="The scenario a finished session's capture proposes.",
    )
    source.add_argument(
        "--scenario",
        type=_eval_revision,
        metavar="REVISION",
        help="A saved scenario revision, from `scenario list`.",
    )


def _add_settings_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--environment", metavar="NAME", help="The registered environment to run in."
    )
    parser.add_argument(
        "--trials",
        type=_positive_count,
        default=DEFAULT_TRIALS,
        metavar="N",
        help=f"Trials (default {DEFAULT_TRIALS}).",
    )
    parser.add_argument(
        "--max-concurrency",
        type=_positive_count,
        default=DEFAULT_MAX_CONCURRENCY,
        metavar="N",
        help=f"Trials running at once (default {DEFAULT_MAX_CONCURRENCY}).",
    )
    parser.add_argument(
        "--trial-timeout-seconds",
        type=_positive_count,
        default=DEFAULT_TRIAL_TIMEOUT_SECONDS,
        metavar="N",
        help=f"Each trial's time limit (default {DEFAULT_TRIAL_TIMEOUT_SECONDS}).",
    )
    parser.add_argument(
        "--max-steps", type=_positive_count, metavar="N", help="Each trial's step limit."
    )
    for field, option in _LIMITS:
        parser.add_argument(
            option,
            dest=field,
            type=_positive_count,
            metavar="N",
            help=f"A per-run limit on {field.removeprefix('max_').replace('_', ' ')}.",
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
        "--artifact",
        action="append",
        type=_artifact_reference,
        metavar="REQUIREMENT=ARTIFACT_ID",
        help="The retained artifact for one of the scenario's file requirements; repeatable.",
    )


def check_scenario_options(args: argparse.Namespace) -> dict[str, Any] | None:
    """Check the options and read a scenario file, before any request; return its draft."""

    if args.eval_scenario_command not in {"check", "run", "prepare-fixture"}:
        return None
    if args.currency is not None and args.max_cost is None:
        raise ValueError("--currency is the currency of --max-cost; pass it with --max-cost.")
    if args.eval_scenario_command == "run":
        continuation = (args.expected_binding_revision, args.expected_execution_profile_revision)
        if (continuation[0] is None) != (continuation[1] is None):
            raise ValueError(
                "--expected-binding-revision and --expected-execution-profile-revision name a "
                "reviewed binding together; pass both, from a run's continue command."
            )
        if args.scenario is not None and args.expected_scenario_revision is not None:
            raise ValueError(
                "--scenario names the revision already; leave out --expected-scenario-revision."
            )
        if (
            continuation[0] is not None
            and args.scenario is None
            and args.expected_scenario_revision is None
        ):
            raise ValueError(
                "A reviewed binding is of one scenario revision: pass --scenario REVISION "
                "or --expected-scenario-revision REV with it, from the printed command."
            )
    if args.file is None:
        return None
    try:
        text = args.file.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        reason = exc.strerror if isinstance(exc, OSError) else "not UTF-8 text"
        raise ValueError(f"Could not read --file {args.file}: {reason}.") from None
    try:
        document = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"--file {args.file} is not JSON: {exc.msg} (line {exc.lineno})."
        ) from None
    if type(document) is not dict:
        raise ValueError(f"--file {args.file} must hold one scenario JSON object.")
    return _draft(document)


def run_scenario_command(
    api: CayuApi,
    args: argparse.Namespace,
    file_draft: dict[str, Any] | None,
    *,
    target: Sequence[str],
) -> dict[str, Any]:
    action = args.eval_scenario_command
    if action == "list":
        return _list(api, args)
    if action == "show":
        return {"scenario": _saved_scenario(api, args.revision)}
    if action == "check":
        return _check(api, args, file_draft, target=target)
    if action == "prepare-fixture":
        return _prepare_fixture(api, args, file_draft, target=target)
    if action in {"approve", "deny"}:
        return _decide(api, args, target=target)
    return _run(api, args, file_draft, target=target)


def _list(api: CayuApi, args: argparse.Namespace) -> dict[str, Any]:
    query = _query(
        target_key=args.target_key,
        scenario_id=args.scenario_id,
        limit=args.limit,
        cursor=args.cursor,
    )
    page = api.get("/evals/scenarios", query=query, resource="eval scenarios")
    return _page(page, "eval scenario list")


def _check(
    api: CayuApi,
    args: argparse.Namespace,
    file_draft: dict[str, Any] | None,
    *,
    target: Sequence[str],
) -> dict[str, Any]:
    settings = _settings(args)
    checked = _preview(api, args, file_draft, settings)
    entry = None
    if args.save:
        saved = _save(api, checked["scenario"], settings)
        checked, entry = saved, saved["entry"]
    return {
        "scenario": checked["scenario"],
        "preflight": checked["preflight"],
        "execution_profile_revision": checked.get("execution_profile_revision"),
        "entry": entry,
        "commands": _ready_commands(args, settings, checked, target=target),
    }


def _ready_commands(
    args: argparse.Namespace,
    settings: Mapping[str, Any],
    checked: Mapping[str, Any],
    *,
    target: Sequence[str],
    saved: bool = False,
) -> dict[str, str]:
    """`commands.run` of a ready check: the run of its revision under a fresh key, pinned to
    the binding and execution profile the check reviewed."""

    if not checked["preflight"]["ready"]:
        return {}
    command = _run_command(
        args,
        settings,
        revision=checked["scenario"]["revision"],
        operation_id=f"cayu-cli-eval-scenario-{secrets.token_hex(16)}",
        binding=checked["preflight"]["binding"]["revision"],
        profile=checked["execution_profile_revision"],
        target=target,
        saved=saved,
    )
    return {"run": command}


def _run(
    api: CayuApi,
    args: argparse.Namespace,
    file_draft: dict[str, Any] | None,
    *,
    target: Sequence[str],
) -> dict[str, Any]:
    """Admit a run of the binding a check reviewed, checking and saving the scenario first.

    With a reviewed binding (a printed run or continue command) the admission is sent first,
    as given, so the server answers with the run the key already admitted whatever the current
    binding is. Only when the server doesn't hold the scenario yet is it checked against the
    expected revision and saved, and the same admission sent again.
    """

    settings = _settings(args)
    operation_id = args.operation_id or f"cayu-cli-eval-scenario-{secrets.token_hex(16)}"
    scenario = preflight = None
    if args.expected_binding_revision is not None:
        revision = args.scenario or args.expected_scenario_revision
        binding = args.expected_binding_revision
        profile = args.expected_execution_profile_revision
    else:
        scenario, preflight, profile = _check_and_save(api, args, file_draft, settings)
        revision, binding = scenario["revision"], preflight["binding"]["revision"]

    def continuation(*, saved: bool) -> dict[str, str]:
        # The continue command names a scenario by its revision once it is known to be
        # saved, and otherwise by its source, so that it can still save it.
        command = _run_command(
            args,
            settings,
            revision=revision,
            operation_id=operation_id,
            binding=binding,
            profile=profile,
            target=target,
            saved=saved,
        )
        return {"continue": command}

    commands = continuation(saved=scenario is not None)
    body = {
        "expected_binding_revision": binding,
        "expected_execution_profile_revision": profile,
        "settings": settings,
    }
    try:
        record = _launch(api, revision, body, operation_id=operation_id, commands=commands)
    except CayuApiNotFound as exc:
        if exc.detail != _SCENARIO_NOT_FOUND or args.scenario is not None or scenario is not None:
            raise
        # A definite refusal that admitted nothing: the run command's first try stopped
        # before saving the scenario.
        scenario, preflight, _ = _check_and_save(
            api, args, file_draft, settings, require_ready=False
        )
        commands = continuation(saved=True)
        record = _launch(api, revision, body, operation_id=operation_id, commands=commands)
    run = _run_summary(record)
    # An admitted run proves the server holds the scenario.
    commands = continuation(saved=True)
    commands["show"] = shlex.join(["cayu", "eval", "runs", "show", run["id"], *target])
    return {
        "scenario_revision": revision,
        "operation_id": operation_id,
        "binding_revision": binding,
        "execution_profile_revision": profile,
        "scenario": scenario,
        "preflight": preflight,
        "run": run,
        "scenario_progress": record.get("scenario_progress"),
        "commands": commands,
    }


def _check_and_save(
    api: CayuApi,
    args: argparse.Namespace,
    file_draft: dict[str, Any] | None,
    settings: Mapping[str, Any],
    *,
    require_ready: bool = True,
) -> tuple[dict[str, Any], dict[str, Any], Any]:
    """Check the scenario against ``--expected-scenario-revision`` and save it; return the
    saved scenario, its preflight and execution-profile revision."""

    checked = _preview(api, args, file_draft, settings)
    revision = checked["scenario"]["revision"]
    expected = args.expected_scenario_revision
    if expected is not None and expected != revision:
        raise CloudEvalError(
            "scenario_changed",
            "The scenario changed after it was checked, so nothing was saved or run. Check "
            "it again.",
            details={"expected_scenario_revision": expected, "scenario_revision": revision},
        )
    if require_ready and not checked["preflight"]["ready"]:
        raise _not_ready(checked["preflight"])
    saved = _save(api, checked["scenario"], settings)
    if require_ready and not saved["preflight"]["ready"]:
        raise _not_ready(saved["preflight"])
    return saved["scenario"], saved["preflight"], saved.get("execution_profile_revision")


def _prepare_fixture(
    api: CayuApi,
    args: argparse.Namespace,
    file_draft: dict[str, Any] | None,
    *,
    target: Sequence[str],
) -> dict[str, Any]:
    """Prepare the requirement's fixture and save the scenario revision that uses it.

    The server copies the retained file into the environment's artifact store under an ID
    derived from its content and returns the scenario with the requirement pointing at it, so
    preparing again changes nothing. That revision is saved, as Save does after Prepare
    fixture, so the run command can name it; the prepared requirement's `--artifact` is
    spent and left out of it.
    """

    settings = _settings(args)
    scenario = _preview(api, args, file_draft, settings)["scenario"]
    requirement = args.requirement_id
    requirements = [
        item.get("id")
        for item in scenario.get("artifact_requirements") or ()
        if isinstance(item, dict)
    ]
    if requirement not in requirements:
        listed = ", ".join(str(item) for item in requirements)
        raise CloudEvalError(
            "requirement_not_found",
            f"The scenario has no file requirement {requirement}"
            + (f"; its file requirements are {listed}." if requirements else "; it has none."),
            details={"requirements": requirements},
        )
    try:
        prepared = api.post(
            f"/evals/scenarios/artifacts/{quote(requirement, safe='')}/materialize",
            {
                "expected_scenario_revision": scenario["revision"],
                "scenario": scenario,
                "settings": settings,
            },
            resource="eval scenario fixture",
        )
    except CayuApiError as exc:
        if exc.status_code != 409 or exc.detail is None:
            raise
        raise CloudEvalError(
            "fixture_not_prepared",
            f"Preparing a fixture for {requirement} failed: {exc.detail}",
            status_code=exc.status_code,
            details={"requirement_id": requirement},
        ) from None
    materialization = prepared.get("materialization") if isinstance(prepared, dict) else None
    if not isinstance(materialization, dict):
        materialization = {}
    updated = materialization.get("scenario")
    if (
        not isinstance(updated, dict)
        or not isinstance(updated.get("revision"), str)
        or materialization.get("requirement_id") != requirement
        or not isinstance(materialization.get("artifact_id"), str)
    ):
        raise CloudEvalError(
            "api_response_invalid", "The server's eval scenario fixture is invalid."
        )
    references = {
        key: value for key, value in settings["artifact_references"].items() if key != requirement
    }
    remaining = {**settings, "artifact_references": references}
    saved = _save(api, updated, remaining)
    return {
        "requirement_id": requirement,
        "artifact_id": materialization["artifact_id"],
        "source_scenario_revision": scenario["revision"],
        "scenario": saved["scenario"],
        "preflight": saved["preflight"],
        "execution_profile_revision": saved.get("execution_profile_revision"),
        "entry": saved["entry"],
        "commands": _ready_commands(args, remaining, saved, target=target, saved=True),
    }


def _decide(api: CayuApi, args: argparse.Namespace, *, target: Sequence[str]) -> dict[str, Any]:
    """Approve or deny the checkpoint a trial awaits, with the progress revision just read.

    The server records a decision only while that exact revision is current, so a checkpoint
    takes one decision. The first read pins the trial, its session and the checkpoint: when
    another trial changed the run's progress in between, the decision is sent again over the
    new revision for that checkpoint only, and a lost answer's continue command decides it
    only while that trial session still awaits it. Every trial of a run has the same
    checkpoint IDs, and a trial that starts over does so in a new session.
    """

    run_id, decision = args.run_id, args.eval_scenario_command
    show = shlex.join(["cayu", "eval", "runs", "show", run_id, *target])
    trial_number, event, session = args.trial, args.event, args.trial_session
    for _ in range(_DECISION_ATTEMPTS):
        record = api.get(_run_path(run_id), resource="eval run")
        trial = _awaiting_trial(
            record, run_id, trial_number=trial_number, event=event, session=session
        )
        trial_number, event = trial["trial_number"], trial["pending_event_id"]
        session = trial.get("session_id")
        body: dict[str, Any] = {
            "expected_progress_revision": record["scenario_progress"]["revision"],
            "trial_number": trial_number,
            "event_id": event,
            "decision": decision,
        }
        if args.reason is not None:
            body["reason"] = args.reason
        try:
            answer = api.mutate(
                f"{_run_path(run_id)}/scenario-approval", body, resource="scenario approval"
            )
        except CayuApiError as exc:
            if exc.category == "outcome_unknown":
                pins = ["--trial", str(trial_number), "--event", event]
                if isinstance(session, str):
                    pins += ["--trial-session", session]
                if args.reason is not None:
                    pins += ["--reason", args.reason]
                continuation = shlex.join(
                    ["cayu", "eval", "scenario", decision, run_id, *pins, *target]
                )
                raise CloudEvalError(
                    "outcome_unknown",
                    f"The server did not confirm the decision on trial {trial_number}'s "
                    f"approval checkpoint {event}; it may have recorded it. `{continuation}` "
                    "decides that checkpoint of that trial session only while it still awaits "
                    "a decision, so it records nothing new if this one was recorded.",
                    status_code=exc.status_code,
                    details={
                        "run_id": run_id,
                        "trial_number": trial_number,
                        "event_id": event,
                        "trial_session_id": session,
                        "commands": {"continue": continuation, "show": show},
                    },
                ) from None
            if exc.detail != _APPROVAL_CHANGED:
                raise
            continue
        return _decision(answer, trial, decision=decision, show=show)
    raise CloudEvalError(
        "scenario_approval_changed",
        f"Eval run {run_id}'s progress kept changing, so trial {trial_number}'s decision was "
        "not recorded. Try again.",
        details={"run_id": run_id, "trial_number": trial_number, "event_id": event},
    )


def _awaiting_trial(
    record: Any,
    run_id: str,
    *,
    trial_number: int | None,
    event: str | None,
    session: str | None,
) -> dict[str, Any]:
    """The trial of the run to decide: ``trial_number``'s, or the only one awaiting approval,
    while it still awaits ``event`` in ``session`` when those are given."""

    run = _run_summary(record)
    invocation = record["spec"].get("invocation")
    if not isinstance(invocation, dict) or invocation.get("scenario") is None:
        raise CloudEvalError(
            "not_a_scenario_run",
            f"Eval run {run_id} is not a scenario run, so none of its trials awaits approval.",
            details={"run_id": run_id},
        )
    if run["status"] != "running":
        # Only a running run takes decisions; a cancelling one keeps its trials' last phase.
        raise CloudEvalError(
            "scenario_approval_not_pending",
            f"Eval run {run_id} is {run['status']}, so none of its trials awaits a decision.",
            details={"run_id": run_id, "status": run["status"]},
        )
    progress = record.get("scenario_progress")
    trials = progress.get("trials") if isinstance(progress, dict) else None
    if trials is not None and (
        not isinstance(trials, list) or not isinstance(progress.get("revision"), str)
    ):
        raise CloudEvalError("api_response_invalid", "The server's scenario progress is invalid.")
    listed = [trial for trial in trials or () if isinstance(trial, dict)]
    awaiting = [
        trial
        for trial in listed
        if trial.get("phase") == "awaiting_approval"
        and trial.get("approval") is None
        and isinstance(trial.get("pending_event_id"), str)
    ]
    phases = {str(trial.get("trial_number")): trial.get("phase") for trial in listed}
    details: dict[str, object] = {"run_id": run_id, "status": run["status"], "trials": phases}
    if trial_number is not None:
        chosen = next((item for item in awaiting if item["trial_number"] == trial_number), None)
        if chosen is None:
            found = next(
                (item for item in listed if item.get("trial_number") == trial_number), None
            )
            if found is None:
                reason = "the run has no such trial."
            elif found.get("approval") is not None:
                reason = "its checkpoint already has a decision."
            else:
                reason = f"it is {found.get('phase')}."
            raise CloudEvalError(
                "scenario_approval_not_pending",
                f"Trial {trial_number} of eval run {run_id} doesn't await a decision now; "
                + reason,
                details=details,
            )
    elif len(awaiting) == 1:
        chosen = awaiting[0]
    elif awaiting:
        numbers = ", ".join(str(item["trial_number"]) for item in awaiting)
        raise CloudEvalError(
            "trial_required",
            f"Trials {numbers} of eval run {run_id} await approval; pass --trial N to decide one.",
            details={
                **details,
                "awaiting": [
                    {
                        "trial_number": item["trial_number"],
                        "event_id": item["pending_event_id"],
                        "tool_name": item.get("pending_tool_name"),
                    }
                    for item in awaiting
                ],
            },
        )
    else:
        raise CloudEvalError(
            "scenario_approval_not_pending",
            f"No trial of eval run {run_id} awaits approval now.",
            details=details,
        )
    if event is not None and chosen["pending_event_id"] != event:
        raise CloudEvalError(
            "scenario_approval_changed",
            f"Trial {chosen['trial_number']} of eval run {run_id} awaits approval checkpoint "
            f"{chosen['pending_event_id']} now, not {event}, so nothing was decided.",
            details={**details, "event_id": chosen["pending_event_id"]},
        )
    if session is not None and chosen.get("session_id") != session:
        raise CloudEvalError(
            "scenario_approval_changed",
            f"Trial {chosen['trial_number']} of eval run {run_id} started over in another "
            "session since, so nothing was decided. `runs show` shows the checkpoint it awaits "
            "now.",
            details={**details, "trial_session_id": chosen.get("session_id")},
        )
    return chosen


def _decision(answer: Any, trial: Mapping[str, Any], *, decision: str, show: str) -> dict[str, Any]:
    run = _run_summary(answer)
    progress = answer.get("scenario_progress")
    trials = progress.get("trials") if isinstance(progress, dict) else None
    decided = next(
        (
            item
            for item in trials or ()
            if isinstance(item, dict) and item.get("trial_number") == trial["trial_number"]
        ),
        None,
    )
    approval = decided.get("approval") if isinstance(decided, dict) else None
    if not isinstance(approval, dict) or approval.get("decision") != decision:
        raise CloudEvalError("api_response_invalid", "The server did not confirm the decision.")
    return {
        "run": run,
        "trial_number": trial["trial_number"],
        "event_id": trial["pending_event_id"],
        "trial_session_id": trial.get("session_id"),
        "tool_name": trial.get("pending_tool_name"),
        "decision": decision,
        "approval": approval,
        "scenario_progress": progress,
        "commands": {"show": show},
    }


def _run_command(
    args: argparse.Namespace,
    settings: Mapping[str, Any],
    *,
    revision: str,
    operation_id: str,
    binding: str,
    profile: str,
    target: Sequence[str],
    saved: bool = False,
) -> str:
    """The `scenario run` that sends this admission again; a ``saved`` scenario is named by
    its revision, another by its source and the revision it must still have."""

    if saved or args.scenario is not None:
        source = ["--scenario", revision]
    else:
        source = [*_source_arguments(args), "--expected-scenario-revision", revision]
    return shlex.join(
        [
            *("cayu", "eval", "scenario", "run", *source),
            *_settings_arguments(settings),
            *("--operation-id", operation_id, "--expected-binding-revision", binding),
            *("--expected-execution-profile-revision", profile),
            *target,
        ]
    )


def _preview(
    api: CayuApi,
    args: argparse.Namespace,
    file_draft: dict[str, Any] | None,
    settings: Mapping[str, Any],
) -> dict[str, Any]:
    draft = file_draft if file_draft is not None else _source_draft(api, args)
    checked = api.post(
        "/evals/scenarios/preview",
        {"draft": draft, "settings": settings},
        resource="eval scenario readiness",
    )
    return _checked(checked, "readiness")


def _save(api: CayuApi, scenario: Mapping[str, Any], settings: Mapping[str, Any]) -> dict:
    # A revision is its content, so saving one the library holds changes nothing.
    saved = api.post(
        "/evals/scenarios",
        {
            "expected_scenario_revision": scenario["revision"],
            "scenario": scenario,
            "settings": settings,
        },
        resource="eval scenario",
    )
    checked = _checked(saved, "saved scenario")
    if not isinstance(checked.get("entry"), dict) or (
        checked["scenario"]["revision"] != scenario["revision"]
    ):
        raise CloudEvalError("api_response_invalid", "The server did not confirm the scenario.")
    return checked


def _checked(answer: Any, what: str) -> dict[str, Any]:
    """``answer`` with its scenario's revision and, when ready, the reviewed binding."""

    scenario = answer.get("scenario") if isinstance(answer, dict) else None
    preflight = answer.get("preflight") if isinstance(answer, dict) else None
    binding = preflight.get("binding") if isinstance(preflight, dict) else None
    if (
        not isinstance(scenario, dict)
        or not isinstance(scenario.get("revision"), str)
        or not isinstance(preflight, dict)
        or type(preflight.get("ready")) is not bool
        or (
            preflight["ready"]
            and (
                not isinstance(binding, dict)
                or not isinstance(binding.get("revision"), str)
                or not isinstance(answer.get("execution_profile_revision"), str)
            )
        )
    ):
        raise CloudEvalError(
            "api_response_invalid", f"The server's eval scenario {what} is invalid."
        )
    return answer


def _source_draft(api: CayuApi, args: argparse.Namespace) -> dict[str, Any]:
    """The draft of a saved revision, or of the scenario a session's capture proposes."""

    if args.scenario is not None:
        return _draft(_saved_scenario(api, args.scenario))
    preview = api.post(
        f"/evals/sessions/{quote(args.session, safe='')}/evaluation/preview",
        {"draft": None},
        resource="captured evaluation",
    )
    conversion = preview.get("scenario_conversion") if isinstance(preview, dict) else None
    if not isinstance(conversion, dict):
        raise CloudEvalError(
            "api_response_invalid", "The server's captured evaluation preview is invalid."
        )
    scenario = conversion.get("scenario")
    if conversion.get("available") is not True or not isinstance(scenario, dict):
        raise _diagnosed_refusal(
            "scenario_unavailable",
            f"Session {args.session} can't be replayed as a scenario",
            conversion.get("diagnostics"),
        )
    return _draft(scenario)


def _saved_scenario(api: CayuApi, revision: str) -> dict[str, Any]:
    scenario = api.get(f"/evals/scenarios/{quote(revision, safe='')}", resource="eval scenario")
    if not isinstance(scenario, dict):
        raise CloudEvalError("api_response_invalid", "The server's eval scenario is invalid.")
    return scenario


def _draft(document: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in document.items() if key not in _DOCUMENT_ONLY_FIELDS}


def _launch(
    api: CayuApi,
    revision: str,
    body: Mapping[str, Any],
    *,
    operation_id: str,
    commands: Mapping[str, str],
) -> dict[str, Any]:
    try:
        record = api.mutate(
            f"/evals/scenarios/{quote(revision, safe='')}/runs",
            body,
            resource="eval scenario run",
            operation_id=operation_id,
        )
    except CayuApiError as exc:
        if exc.category == "outcome_unknown":
            raise CloudEvalError(
                "outcome_unknown",
                "The server did not confirm the scenario run; it may have admitted it. "
                f"`{commands['continue']}` admits nothing new if it did.",
                status_code=exc.status_code,
                details={"operation_id": operation_id, "commands": dict(commands)},
            ) from None
        if exc.detail == _IDEMPOTENCY_CONFLICT:
            raise CloudEvalError(
                "operation_id_conflict",
                f"Operation ID {operation_id} already admitted a different scenario run, so "
                "this one admitted nothing.",
                status_code=exc.status_code,
                details={"operation_id": operation_id},
            ) from None
        if exc.detail == _BINDING_CHANGED:
            raise CloudEvalError(
                "scenario_changed",
                "What the scenario's check reviewed (the release, environment, fixtures, "
                "secrets or limits) changed since, so no run was admitted. Check it again.",
                status_code=exc.status_code,
                details={"operation_id": operation_id},
            ) from None
        if exc.detail == _LAUNCH_NOT_READY:
            raise CloudEvalError(
                "scenario_not_ready",
                "The server can't run this scenario now, so no run was admitted. Check it "
                "again for the reasons.",
                status_code=exc.status_code,
                details={"operation_id": operation_id},
            ) from None
        raise
    if not isinstance(record, dict):
        raise CloudEvalError("api_response_invalid", "The server did not confirm the scenario run.")
    return record


def _not_ready(preflight: Mapping[str, Any]) -> CloudEvalError:
    return _diagnosed_refusal(
        "scenario_not_ready",
        "The server can't run this scenario now, so nothing was saved or run",
        preflight.get("diagnostics"),
    )


def _settings(args: argparse.Namespace) -> dict[str, Any]:
    """The launch settings, as the dashboard sends them (`scenarioLaunchSettingsContract`)."""

    limits = {
        field: getattr(args, field) for field, _ in _LIMITS if getattr(args, field) is not None
    }
    return {
        "environment_name": args.environment,
        "trials": args.trials,
        "max_concurrency": args.max_concurrency,
        "timeout_seconds": args.trial_timeout_seconds,
        "max_steps": args.max_steps,
        "limits": {"scope": "run", **limits} if limits else None,
        "cost_budget": (
            None
            if args.max_cost is None
            else {
                "max_estimated_cost": str(args.max_cost),
                "currency": args.currency or _DEFAULT_CURRENCY,
            }
        ),
        "artifact_references": dict(args.artifact or ()),
    }


def _settings_arguments(settings: Mapping[str, Any]) -> list[str]:
    """The options that send ``settings`` again; defaults are left out."""

    arguments: list[str] = []
    if settings["environment_name"] is not None:
        arguments += ["--environment", settings["environment_name"]]
    for name, value, default in (
        ("--trials", settings["trials"], DEFAULT_TRIALS),
        ("--max-concurrency", settings["max_concurrency"], DEFAULT_MAX_CONCURRENCY),
        ("--trial-timeout-seconds", settings["timeout_seconds"], DEFAULT_TRIAL_TIMEOUT_SECONDS),
    ):
        if value != default:
            arguments += [name, str(value)]
    if settings["max_steps"] is not None:
        arguments += ["--max-steps", str(settings["max_steps"])]
    for field, option in _LIMITS:
        value = (settings["limits"] or {}).get(field)
        if value is not None:
            arguments += [option, str(value)]
    budget = settings["cost_budget"]
    if budget is not None:
        arguments += ["--max-cost", budget["max_estimated_cost"], "--currency", budget["currency"]]
    for requirement, artifact in settings["artifact_references"].items():
        arguments += ["--artifact", f"{requirement}={artifact}"]
    return arguments


def _source_arguments(args: argparse.Namespace) -> list[str]:
    if args.file is not None:
        return ["--file", str(args.file)]
    if args.session is not None:
        return ["--session", args.session]
    return ["--scenario", args.scenario]


def _artifact_reference(value: str) -> tuple[str, str]:
    requirement, separator, artifact = value.partition("=")
    if not separator or not requirement or not artifact:
        raise argparse.ArgumentTypeError("must be REQUIREMENT=ARTIFACT_ID")
    return requirement, artifact


def render_scenarios(answer: Mapping[str, Any]) -> str:
    listed = answer["result"]
    if not listed["items"]:
        return "No saved scenarios."
    rows = [
        (
            _cell(item.get("name")),
            _cell(item.get("event_count")),
            _cell(item.get("approval_checkpoint_count")),
            _text(item.get("revision")),
        )
        for item in listed["items"]
    ]
    lines = _table(("NAME", "EVENTS", "APPROVALS", "REVISION"), rows)
    if listed["has_more"]:
        lines.append(f"More scenarios follow: pass --cursor {_text(listed['next_cursor'])}.")
    return "\n".join(lines)


def render_scenario(answer: Mapping[str, Any]) -> str:
    scenario = answer["result"]["scenario"]
    lines = [f"Scenario {_text(scenario.get('name'))}: {_text(scenario.get('revision'))}"]
    for event in scenario.get("events") or ():
        kind = event.get("kind") or ("approval_checkpoint" if "tool_name" in event else "input")
        lines.append(f"  {_text(event.get('sequence'))}. {_text(kind)} {_text(event.get('id'))}")
    return "\n".join(lines)


def render_check(answer: Mapping[str, Any]) -> str:
    result = answer["result"]
    lines = _readiness_lines(result["scenario"], result["preflight"])
    if result["entry"] is not None:
        lines.append("Saved to the scenario library.")
    if "run" in result["commands"]:
        lines.append(f"Run it: {_text(result['commands']['run'])}")
    return "\n".join(lines)


def render_prepared(answer: Mapping[str, Any]) -> str:
    result = answer["result"]
    lines = [
        f"Prepared fixture {_text(result['artifact_id'])} for "
        f"{_text(result['requirement_id'])}; saved the scenario revision that uses it.",
        *_readiness_lines(result["scenario"], result["preflight"]),
    ]
    if "run" in result["commands"]:
        lines.append(f"Run it: {_text(result['commands']['run'])}")
    return "\n".join(lines)


def _readiness_lines(scenario: Mapping[str, Any], preflight: Mapping[str, Any]) -> list[str]:
    name = f"Scenario {_text(scenario.get('name'))} ({_text(scenario['revision'])})"
    if preflight["ready"]:
        return [f"{name} is ready to run."]
    lines = [f"{name} is not ready to run:"]
    for item in preflight.get("diagnostics") or ():
        lines.append(f"  {_text(item.get('message'))} {_text(item.get('remediation'))}")
    return lines


def render_decision(answer: Mapping[str, Any]) -> str:
    result = answer["result"]
    verb = "Approved" if result["decision"] == "approve" else "Denied"
    return (
        f"{verb} {_text(result['tool_name'])} for trial {_text(result['trial_number'])} of "
        f"run {_text(result['run']['id'])} (checkpoint {_text(result['event_id'])}).\n"
        f"Follow it: {_text(result['commands']['show'])}"
    )


def render_run(answer: Mapping[str, Any]) -> str:
    result = answer["result"]
    run = result["run"]
    return (
        f"Scenario run {_text(run['id'])}: {_text(run['status'])}.\n"
        f"Follow it: {_text(result['commands']['show'])}"
    )

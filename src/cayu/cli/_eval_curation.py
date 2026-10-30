"""`cayu eval suites|runs|add-session|baseline`: curate evals from the terminal.

These commands do what the Developer dashboard's Evals area does, through the same server
API: list suites and runs and open one, add a finished session to a suite, and approve a run
as its suite's baseline. They reach the project's running server (`--server-url`, `cayu
serve`'s address by default) or, with `--application NAME`, a Cayu Cloud Agent, signed in
as `cayu cloud` is. `runs` and `baseline approve` are `cayu cloud eval runs` and `cayu cloud
eval baseline` (`cayu.cli._cloud_evals`), so with `--application` they answer the same.

`add-session` previews the addition (`POST /evals/sessions/{id}/additions/preview`) and
publishes the plan it previewed under an idempotency key (`POST .../additions`); `--preview`
stops after the preview. The printed publish command repeats the request with that key and
the plan's revisions, so it publishes that reviewed plan, replays the receipt of a
publication that already happened, or is refused when the session or suite has changed since.
"""

from __future__ import annotations

import argparse
import json
import secrets
import shlex
import sys
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import Any
from urllib.parse import quote

from cayu.cli._cloud_agent_api import CayuApi, CayuApiError
from cayu.cli._cloud_evals import (
    DEFAULT_LIST_LIMIT,
    MAX_LIST_LIMIT,
    RUN_STATUSES,
    SCHEMA_VERSION,
    CloudEvalError,
    _agent_api,
    _approve_baseline,
    _eval_revision,
    _generation,
    _list_limit,
    _list_runs,
    _operation_id,
    _show_run,
    _text,
    eval_api,
    render_baseline,
    render_failure,
    render_run_detail,
    render_runs,
)
from cayu.cli._cloud_settings import _cell, _table
from cayu.cli._output import output_destination

COMMANDS = frozenset({"suites", "runs", "add-session", "baseline"})
# The `source` of a suite view whose runs are external eval reports (`EvalReportedSuiteView`).
_REPORTED = "reported"
# The server's own refusals of a publication (`cayu.server._eval_addition_routes`): the
# reviewed plan no longer matches the session or suite, and the key already published
# another request.
_ADDITION_CHANGED = "The reviewed addition changed; preview it again."
_ADDITION_KEY_CONFLICT = "The operation key is already bound to another addition."


def add_curation_parsers(commands: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Add `suites`, `runs`, `add-session` and `baseline` to `cayu eval`."""

    suites = commands.add_parser(
        "suites",
        help="List eval suites, or show one with its cases and runs.",
        description=(
            "List the authored and corpus suites of a running server or Cloud Agent, or show "
            "one with its cases and run history, as the Developer dashboard's Evals area does."
        ),
    )
    suite_actions = suites.add_subparsers(dest="eval_suites_command", required=True)
    listing = suite_actions.add_parser(
        "list",
        help="List eval suites with their latest run and baseline.",
        description=(
            "List eval suites, each with its id, case count, latest run and baseline. Use the "
            "id with `suites show` and `add-session --suite`."
        ),
    )
    listing.add_argument("--target-key", metavar="KEY", help="Only this eval target's suites.")
    listing.add_argument(
        "--limit",
        type=_list_limit,
        metavar="N",
        help=f"Suites per page, at most {MAX_LIST_LIMIT} (default: the server's).",
    )
    listing.add_argument("--cursor", help="The next page: the previous page's next_cursor.")
    _add_target_options(listing, render_suites)
    show = suite_actions.add_parser(
        "show",
        help="Show one suite with its cases and runs.",
        description=(
            "Show one suite revision: its cases with their latest outcomes, its runs, newest "
            "first, and its baseline."
        ),
    )
    show.add_argument("suite", metavar="SUITE", help="The suite's id from `suites list`.")
    for name, noun in (("cases", "Cases"), ("runs", "Runs")):
        show.add_argument(
            f"--{name}-limit",
            type=_list_limit,
            metavar="N",
            help=f"{noun} per page, at most {MAX_LIST_LIMIT} (default: the server's).",
        )
        show.add_argument(
            f"--{name}-cursor", help=f"The next page of {name}: the previous next_cursor."
        )
    _add_target_options(show, render_suite)

    runs = commands.add_parser(
        "runs",
        help="List eval runs, or show one.",
        description="List an eval target's runs, or show one run's status, progress and result.",
    )
    run_actions = runs.add_subparsers(dest="eval_runs_command", required=True)
    listing = run_actions.add_parser(
        "list",
        help="List an eval target's runs, newest first.",
        description=(
            "List an eval target's runs, newest first: the default target unless --target-key "
            "names another."
        ),
    )
    listing.add_argument(
        "--target-key", metavar="KEY", help="The eval target (`GET /api/evals/targets`)."
    )
    listing.add_argument("--status", choices=RUN_STATUSES)
    listing.add_argument(
        "--limit",
        type=_list_limit,
        default=DEFAULT_LIST_LIMIT,
        metavar="N",
        help=f"The newest N runs (default {DEFAULT_LIST_LIMIT}, at most {MAX_LIST_LIMIT}).",
    )
    _add_target_options(listing, render_runs)
    show = run_actions.add_parser(
        "show",
        help="Show one eval run.",
        description="Show one eval run: its status, progress and result.",
    )
    show.add_argument("run_id", metavar="RUN_ID")
    _add_target_options(show, render_run_detail)

    addition = commands.add_parser(
        "add-session",
        help="Add a finished session to an eval suite.",
        description=(
            "Add a finished session to an existing suite revision or a new suite, as the "
            "Developer dashboard's Add to evals does: as a runnable case (--mode case) or as "
            "one scored captured case (--mode score). The command previews the addition, then "
            "publishes that plan under an idempotency key; --preview stops after the preview "
            "and prints the command that publishes it."
        ),
    )
    addition.add_argument("session_id", metavar="SESSION_ID")
    destination = addition.add_mutually_exclusive_group(required=True)
    destination.add_argument(
        "--suite",
        metavar="SUITE",
        help=(
            "An existing suite's id from `suites list`. A case makes a new revision of that "
            "exact revision; a score leaves its cases unchanged."
        ),
    )
    destination.add_argument("--new-suite", metavar="ID", help="Create a new suite with this id.")
    addition.add_argument(
        "--suite-name",
        metavar="NAME",
        help="With --new-suite: its name (default: the suite name the session's capture proposes).",
    )
    addition.add_argument(
        "--mode",
        choices=("case", "score"),
        required=True,
        help=(
            "case keeps the session's input as a runnable case; score checks its retained "
            "evidence without rerunning it and records one captured case."
        ),
    )
    addition.add_argument(
        "--case-id", metavar="ID", help="The case's id (default: the one the capture proposes)."
    )
    addition.add_argument(
        "--case-name",
        metavar="NAME",
        help="The case's name (default: the one the capture proposes).",
    )
    addition.add_argument(
        "--assertion",
        action="append",
        type=_assertion,
        metavar="JSON",
        help=(
            "One assertion as a JSON object, in the corpus assertion format; repeat for more. "
            "Without one, the checks the capture proposes."
        ),
    )
    addition.add_argument(
        "--preview",
        action="store_true",
        help="Preview the addition and print the command that publishes it; publish nothing.",
    )
    addition.add_argument(
        "--operation-id",
        type=_operation_id,
        metavar="ID",
        help=(
            "The publication's idempotency key. Without one a fresh key is printed in the "
            "publish command."
        ),
    )
    for name in ("plan", "source"):
        addition.add_argument(
            f"--expected-{name}-revision",
            type=_eval_revision,
            metavar="REV",
            help=(
                f"Publish a previewed plan: its {name} revision, from the publish command. "
                "The server refuses it when the session or suite has changed since."
            ),
        )
    _add_target_options(addition, render_addition)

    baseline = commands.add_parser(
        "baseline",
        help="Approve a run as its suite's baseline.",
        description="Approve a completed run's result as its suite's baseline.",
    )
    baseline_actions = baseline.add_subparsers(dest="eval_baseline_command", required=True)
    approve = baseline_actions.add_parser(
        "approve",
        help="Approve a completed run's result as its suite's baseline.",
        description=(
            "Approve a completed run's result as the baseline later runs of its suite are "
            "compared with, over the baseline's current generation. A run of an authored suite "
            "is one execution group, and the baseline is that group's."
        ),
    )
    approve.add_argument("run_id", metavar="RUN_ID")
    approve.add_argument(
        "--expected-generation",
        type=_generation,
        metavar="N",
        help=(
            "Approve only over the baseline at generation N (0 when the suite has none); "
            "without it, over the current baseline. From an approval's continuation command."
        ),
    )
    approve.add_argument(
        "--operation-id",
        type=_eval_revision,
        metavar="ID",
        help=(
            "With --expected-generation, repeat an earlier approval: its operation ID, from its "
            "continuation command."
        ),
    )
    _add_target_options(approve, render_baseline)


def run_curation_command(args: argparse.Namespace) -> int:
    """Run one of the commands and print its JSON envelope, or its text with ``--table``."""

    from cayu.cli._cloud_api import CloudApiError
    from cayu.cli._cloud_auth import CloudAuthError
    from cayu.cli._control_plane_actions import _output_aliases_input
    from cayu.cli.cloud import CloudCommandError, _cloud_failure

    if _output_aliases_input(args):
        print("error: output must not refer to an input file.", file=sys.stderr)
        return 1
    table = args.output_format == "table"
    try:
        with output_destination(args.output):
            try:
                answer = _execute(args)
            except (
                CloudApiError,
                CloudAuthError,
                CloudCommandError,
                KeyError,
                OSError,
                ValueError,
            ) as exc:
                return _cloud_failure(exc, render=render_failure if table else None)
            if table:
                print(args._eval_render(answer))
            else:
                print(json.dumps({"ok": True, **answer}, sort_keys=True))
            return 0
    except OSError as exc:
        print(f"error: could not write output: {exc}", file=sys.stderr)
        return 1


def _execute(args: argparse.Namespace) -> dict[str, Any]:
    action = args.eval_command
    if action == "add-session":
        _check_addition_options(args)
    with _target_api(args) as (api, source, target):
        if action == "suites":
            if args.eval_suites_command == "list":
                result = _list_suites(api, args)
            else:
                result = _show_suite(api, args)
            operation = f"eval.suites.{args.eval_suites_command}"
        elif action == "runs":
            if args.eval_runs_command == "list":
                result = _list_runs(api, args)
            else:
                result = _show_run(api, args.run_id)
            operation = f"eval.runs.{args.eval_runs_command}"
        elif action == "add-session":
            result = _add_session(api, args, target=target)
            operation = "eval.add-session"
        else:
            result = _approve_baseline(
                api, args, command=["cayu", "eval", "baseline", "approve", args.run_id, *target]
            )
            operation = "eval.baseline"
    return {
        "operation": operation,
        "result": {"schema_version": SCHEMA_VERSION, **source, **result},
    }


@contextmanager
def _target_api(
    args: argparse.Namespace,
) -> Iterator[tuple[CayuApi, dict[str, str], list[str]]]:
    """The eval API of the server or Agent the command names, what its answer calls it, and
    the options a printed command repeats to reach it again."""

    if args.application is not None:
        from cayu.cli.cloud import cloud_client, resolve_application_id

        client = cloud_client(context=args.context, timeout_seconds=args.timeout_seconds)
        application_id = resolve_application_id(client, args.application)
        target = ["--application", application_id]
        if args.context is not None:
            target += ["--context", str(args.context)]
        with _agent_api(client, application_id, args.timeout_seconds) as api:
            yield api, {"application_id": application_id}, target
        return
    from cayu.cli._control_plane_actions import action_server_url, open_action_api
    from cayu.cli.diagnostics import DEFAULT_AUTHORIZATION_ENV

    # The running server, reached as `cayu task create` reaches it.
    try:
        api = open_action_api(args)
    except CayuApiError as exc:
        raise CloudEvalError(exc.category, str(exc)) from None
    target = [] if args.server_url is None else ["--server-url", args.server_url]
    if args.authorization_env != DEFAULT_AUTHORIZATION_ENV:
        target += ["--authorization-env", args.authorization_env]
    with eval_api(api) as opened:
        yield opened, {"server_url": action_server_url(args)}, target


def _list_suites(api: CayuApi, args: argparse.Namespace) -> dict[str, Any]:
    query = _query(target_key=args.target_key, limit=args.limit, cursor=args.cursor)
    page = api.get("/evals/suite-views", query=query, resource="eval suites")
    return _page(page, "eval suite list")


def _show_suite(api: CayuApi, args: argparse.Namespace) -> dict[str, Any]:
    query = _query(
        cases_limit=args.cases_limit,
        cases_cursor=args.cases_cursor,
        runs_limit=args.runs_limit,
        runs_cursor=args.runs_cursor,
    )
    detail = api.get(
        f"/evals/suite-views/{quote(args.suite, safe='')}", query=query, resource="eval suite"
    )
    suite = detail.get("suite") if isinstance(detail, dict) else None
    if not isinstance(suite, dict):
        raise CloudEvalError("api_response_invalid", "The server's eval suite is invalid.")
    # A reported suite, the series of an external eval report, stores no case definitions:
    # its detail is the suite and its reported runs.
    reported = suite.get("source") == _REPORTED
    return {
        "suite": suite,
        "cases": None if reported else _page(detail.get("cases"), "eval suite's case list"),
        "runs": _page(detail.get("runs"), "eval suite's run list"),
    }


def _add_session(
    api: CayuApi, args: argparse.Namespace, *, target: Sequence[str]
) -> dict[str, Any]:
    """Preview the addition, then publish the plan under ``--operation-id`` (or a fresh key).

    With the expected revisions the plan was previewed before: the request is published with
    them as given, so the server replays a publication it already made and refuses a plan the
    session or suite no longer matches.
    """

    expected = (args.expected_plan_revision, args.expected_source_revision)
    path = f"/evals/sessions/{quote(args.session_id, safe='')}"
    request = _addition_request(api, args, path)
    plan = None
    if expected[0] is None:
        plan = api.post(
            f"{path}/additions/preview", request, resource="eval session addition preview"
        )
        if not isinstance(plan, dict) or not all(
            isinstance(plan.get(key), str) for key in ("revision", "source_revision")
        ):
            raise CloudEvalError(
                "api_response_invalid", "The server's eval session addition plan is invalid."
            )
        expected = (plan["revision"], plan["source_revision"])
    operation_id = args.operation_id or f"cayu-cli-eval-addition-{secrets.token_hex(16)}"
    command = [
        *_addition_command(args.session_id, request, target),
        *("--operation-id", operation_id),
        *("--expected-plan-revision", expected[0], "--expected-source-revision", expected[1]),
    ]
    commands = {"publish": shlex.join(command)}
    receipt = None
    if not args.preview:
        receipt = _publish_addition(
            api,
            f"{path}/additions",
            {
                **request,
                "expected_plan_revision": expected[0],
                "expected_source_revision": expected[1],
            },
            operation_id=operation_id,
            commands=commands,
        )
        plan = receipt["plan"]
    return {
        "session_id": args.session_id,
        "operation_id": operation_id,
        "request": request,
        "plan": plan,
        "receipt": receipt,
        "destination": _suite_id(plan),
        "commands": commands,
    }


def _check_addition_options(args: argparse.Namespace) -> None:
    if (args.expected_plan_revision is None) != (args.expected_source_revision is None):
        raise ValueError(
            "--expected-plan-revision and --expected-source-revision name a previewed plan "
            "together; pass both, from its publish command."
        )
    if args.preview and args.expected_plan_revision is not None:
        raise ValueError("--preview previews a new plan; leave out the expected revisions.")
    if args.suite_name is not None and args.new_suite is None:
        raise ValueError("--suite-name names a new suite; pass it with --new-suite.")


def _addition_request(api: CayuApi, args: argparse.Namespace, path: str) -> dict[str, Any]:
    """The addition request; what it leaves out comes from the session's capture preview,
    the defaults the Developer dashboard proposes."""

    case_id, case_name, suite_name = args.case_id, args.case_name, args.suite_name
    if case_id is None or case_name is None or (args.new_suite is not None and suite_name is None):
        preview = api.post(
            f"{path}/evaluation/preview", {"draft": None}, resource="captured evaluation"
        )
        candidate = preview.get("candidate") if isinstance(preview, dict) else None
        case = candidate.get("case") if isinstance(candidate, dict) else None
        suite = candidate.get("suite") if isinstance(candidate, dict) else None
        if not isinstance(case, dict) or not isinstance(suite, dict):
            raise CloudEvalError(
                "api_response_invalid", "The server's captured evaluation preview is invalid."
            )
        case_id = case.get("id") if case_id is None else case_id
        case_name = case.get("name") if case_name is None else case_name
        if args.new_suite is not None and suite_name is None:
            suite_name = suite.get("name")
    destination = (
        {"kind": "existing", "selector": args.suite}
        if args.suite is not None
        else {"kind": "new", "suite_id": args.new_suite, "name": suite_name}
    )
    request: dict[str, Any] = {
        "mode": args.mode,
        "destination": destination,
        "case_id": case_id,
        "case_name": case_name,
    }
    if args.assertion is not None:
        request["assertions"] = args.assertion
    return request


def _addition_command(
    session_id: str, request: Mapping[str, Any], target: Sequence[str]
) -> list[str]:
    """The `add-session` that sends ``request`` again, every default it took made explicit."""

    command = ["cayu", "eval", "add-session", session_id]
    destination = request["destination"]
    if destination["kind"] == "existing":
        command += ["--suite", destination["selector"]]
    else:
        command += ["--new-suite", destination["suite_id"], "--suite-name", destination["name"]]
    command += ["--mode", request["mode"]]
    command += ["--case-id", request["case_id"], "--case-name", request["case_name"]]
    for assertion in request.get("assertions") or ():
        command += ["--assertion", json.dumps(assertion, ensure_ascii=False, sort_keys=True)]
    return [*command, *target]


def _publish_addition(
    api: CayuApi,
    path: str,
    body: Mapping[str, Any],
    *,
    operation_id: str,
    commands: Mapping[str, str],
) -> dict[str, Any]:
    try:
        receipt = api.mutate(
            path, body, resource="eval session addition", operation_id=operation_id
        )
    except CayuApiError as exc:
        if exc.category == "outcome_unknown":
            raise CloudEvalError(
                "outcome_unknown",
                "The server did not confirm the session addition; it may have published it. "
                f"`{commands['publish']}` publishes nothing new if it did.",
                status_code=exc.status_code,
                details={"operation_id": operation_id, "commands": dict(commands)},
            ) from None
        if exc.detail == _ADDITION_CHANGED:
            raise CloudEvalError(
                "eval_addition_changed",
                "The session or the suite changed after this addition was previewed, so "
                "nothing was published. Preview it again.",
                status_code=exc.status_code,
                details={"operation_id": operation_id},
            ) from None
        if exc.detail == _ADDITION_KEY_CONFLICT:
            raise CloudEvalError(
                "operation_id_conflict",
                f"Operation ID {operation_id} already published a different addition, so this "
                "one published nothing.",
                status_code=exc.status_code,
                details={"operation_id": operation_id},
            ) from None
        raise
    plan = receipt.get("plan") if isinstance(receipt, dict) else None
    if not isinstance(plan, dict) or plan.get("revision") != body["expected_plan_revision"]:
        raise CloudEvalError(
            "api_response_invalid", "The server did not confirm the session addition."
        )
    return receipt


def _suite_id(plan: Any) -> str:
    """The `suites show` id of the plan's destination suite."""

    from cayu.evals.read_views import EvalSuiteViewIdentity

    identity = plan.get("destination") if isinstance(plan, dict) else None
    try:
        return EvalSuiteViewIdentity.model_validate(identity).id
    except ValueError:
        raise CloudEvalError(
            "api_response_invalid", "The server's eval session addition plan is invalid."
        ) from None


def _page(page: Any, what: str) -> dict[str, Any]:
    items = page.get("items") if isinstance(page, dict) else None
    if not isinstance(page, dict) or not isinstance(items, list):
        raise CloudEvalError("api_response_invalid", f"The server's {what} is invalid.")
    return {
        "items": items,
        "next_cursor": page.get("next_cursor"),
        "has_more": page.get("has_more") is True,
        "total": page.get("total"),
    }


def _query(**values: object) -> dict[str, str]:
    return {name: str(value) for name, value in values.items() if value is not None}


def _add_target_options(
    parser: argparse.ArgumentParser, render: Callable[[Mapping[str, Any]], str]
) -> None:
    from cayu.cli._control_plane_actions import add_action_target_options

    add_action_target_options(parser, output_format="json")
    parser.set_defaults(_eval_render=render)


def _assertion(value: str) -> dict[str, Any]:
    try:
        assertion = json.loads(value)
    except json.JSONDecodeError:
        assertion = None
    if type(assertion) is not dict:
        raise argparse.ArgumentTypeError("must be one assertion as a JSON object")
    return assertion


def render_suites(answer: Mapping[str, Any]) -> str:
    listed = answer["result"]
    if not listed["items"]:
        return "No eval suites."
    rows = [
        (
            _cell(item.get("name")),
            _cell(item.get("source")),
            _cell(item.get("case_count")),
            _cell((item.get("last_run") or {}).get("status")),
            _text(item.get("id")),
        )
        for item in listed["items"]
    ]
    lines = _table(("NAME", "SOURCE", "CASES", "LAST RUN", "SUITE"), rows)
    if listed["has_more"]:
        lines.append(f"More suites follow: pass --cursor {_text(listed['next_cursor'])}.")
    return "\n".join(lines)


def render_suite(answer: Mapping[str, Any]) -> str:
    result = answer["result"]
    suite, cases, runs = result["suite"], result["cases"], result["runs"]
    lines = [
        f"Suite {_text(suite.get('name'))} ({_text(suite.get('source'))}, "
        f"{_text(suite.get('case_count'))} cases): {_text(suite.get('id'))}",
        f"Baseline: {_baseline_text(suite)}",
    ]
    if cases is None:
        lines.append("Cases: none stored; a reported suite has only its reported runs.")
    elif cases["items"]:
        rows = [
            (
                _cell(case.get("name")),
                _cell((case.get("last") or {}).get("status")),
                _text(case.get("id")),
            )
            for case in cases["items"]
        ]
        lines += ["", "Cases", *_table(("NAME", "LAST", "CASE"), rows)]
    if runs["items"]:
        rows = [
            (
                _cell(run.get("status")),
                _cell(run.get("created_at")),
                _cell((run.get("comparison") or {}).get("verdict")),
                # Whole, to be copied; a reported run has neither, and its id is its run key.
                _text(run.get("native_run_id") or run.get("result_revision") or run.get("id")),
            )
            for run in runs["items"]
        ]
        lines += ["", "Runs", *_table(("STATUS", "CREATED", "COMPARED", "RUN"), rows)]
    for name, page in (("cases", cases), ("runs", runs)):
        if page is not None and page["has_more"]:
            lines.append(f"More {name} follow: pass --{name}-cursor {_text(page['next_cursor'])}.")
    return "\n".join(lines)


def _baseline_text(suite: Mapping[str, Any]) -> str:
    if suite.get("baseline_scope") == "execution_group":
        return "one for each run's execution group"
    if suite.get("source") == _REPORTED:
        return "unavailable; reported runs can't be baselines"
    baseline = suite.get("baseline")
    if not isinstance(baseline, dict):
        return "none"
    return f"generation {_text(baseline.get('generation'))}"


def render_addition(answer: Mapping[str, Any]) -> str:
    result = answer["result"]
    plan = result["plan"]
    count = plan["destination_case_count"]
    if plan["mode"] == "score":
        change = "one captured case scored" + (
            " in a new captured suite" if plan["membership_changed"] else ", its cases unchanged"
        )
    elif plan["membership_changed"]:
        change = f"a new revision with {count} case" + ("" if count == 1 else "s")
    else:
        change = "this exact case is already there, its cases unchanged"
    if result["receipt"] is None:
        return (
            f"Preview of case {_text(plan['case_id'])}: {change}.\n"
            f"Publish it: {_text(result['commands']['publish'])}"
        )
    return f"Added case {_text(plan['case_id'])}: {change}.\nSuite: {_text(result['destination'])}"

"""`cayu cloud data-move run|resume|runs --application NAME`: data moves on a deployed Agent.

The Agent's own Cayu server runs the moves its project declares in `[tool.cayu.data_moves]`
(`cayu.server.data_moves`), in its web process, and keeps a record of each run in its data
volume. The CLI reaches that server as `cayu cloud data` does, with the Agent's Developer
dashboard login (`_cloud_agent_api`): `run` and `resume` start a run and, unless
`--no-wait`, read its record until it stops; `runs` lists records or shows one.

An Agent whose release was built with a Cayu without these routes answers with a plain
route-not-found, which the command reports as `unsupported`, with what to deploy.
"""

from __future__ import annotations

import argparse
import secrets
import time
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any
from urllib.parse import quote

from cayu.cli._cloud_reference import APPLICATION_HELP
from cayu.cli._cloud_releases import add_output_argument
from cayu.cli._cloud_settings import _cell, _table
from cayu.cli.data_move import EXIT_NOT_CLEAN, _positive

if TYPE_CHECKING:
    from cayu.cli._cloud_agent_api import CayuApi, CayuApiError
    from cayu.cli._cloud_api import CloudApiClient

_PATH = "/data-moves"
_POLL_SECONDS = 2.0


def add_data_move_parser(commands: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    group = commands.add_parser(
        "data-move",
        help="Run a deployed Agent's data moves and read their records.",
        description=(
            "Run the data moves the Agent's project declares in [tool.cayu.data_moves] on the "
            "Agent itself, with its own database and secrets, and read their run records. "
            "Needs a release built with a Cayu that serves data moves."
        ),
    )
    actions = group.add_subparsers(dest="data_move_command", required=True)
    run = actions.add_parser(
        "run",
        description=(
            "Start a run of the declared move NAME on the Agent and, unless --no-wait, wait for "
            "it to stop and print its report. A run whose process stops mid-way (a deploy, a "
            "sleeping Agent) keeps its record; go on with `resume`."
        ),
    )
    run.add_argument("name", help="The move's name in [tool.cayu.data_moves].")
    run.add_argument("--dry-run", action="store_true", help="Decide outcomes, write nothing.")
    run.add_argument("--page-size", type=_positive, default=100, help="Items per page (100).")
    run.add_argument("--after", help="Start after this source cursor.")
    run.add_argument("--max-items", type=_positive, help="Stop after about this many items.")
    run.add_argument(
        "--stop-on-failure", action="store_true", help="Stop at the first failing item."
    )
    resume = actions.add_parser(
        "resume",
        description=(
            "Go on with run --run-id, or the newest open run of NAME, from its record, with the "
            "options it started with."
        ),
    )
    resume.add_argument("name", help="The move's name in [tool.cayu.data_moves].")
    resume.add_argument("--run-id", help="The run to go on with (default: the newest open one).")
    for parser in (run, resume):
        parser.add_argument(
            "--no-wait", action="store_true", help="Return once the run has started."
        )
        parser.add_argument(
            "--wait-seconds",
            type=float,
            default=3600.0,
            help="How long to wait for the run to stop (3600); the run goes on after.",
        )
        add_output_argument(parser, render_run)
    runs = actions.add_parser(
        "runs", description="List the Agent's declared moves and recent runs, or show one run."
    )
    runs.add_argument("--run-id", help="Show this run's record.")
    add_output_argument(runs, render_runs)
    for parser in (run, resume, runs):
        parser.add_argument("--application", required=True, help=APPLICATION_HELP)


def data_move_command(
    arguments: argparse.Namespace, *, client: CloudApiClient, application_id: str
) -> dict[str, Any]:
    """Run one `cayu cloud data-move` command on the Agent ``application_id``."""

    from cayu.cli._cloud_agent_api import CayuApiError, cloud_agent_api

    try:
        api = cloud_agent_api(client, application_id, timeout_seconds=arguments.timeout_seconds)
    except CayuApiError as exc:
        raise _command_error(exc) from None
    try:
        # A route every Cayu serves finds the Agent's API first, so an Agent whose Cayu
        # predates data moves is told apart from one that serves no Cayu API.
        api.get("/contract", resource="server contract")
        action = arguments.data_move_command
        if action == "runs":
            if arguments.run_id is not None:
                return {"operation": "data_move.show", "result": _read(api, arguments.run_id)}
            listing = api.get(_PATH, resource="data moves")
            return {"operation": "data_move.list", "result": listing}
        body: dict[str, Any]
        if action == "resume":
            body = {"resume": True, "run_id": arguments.run_id}
        else:
            body = {
                # Chosen here, so a start whose answer was lost can be looked up by it.
                "run_id": secrets.token_hex(8),
                "dry_run": arguments.dry_run,
                "page_size": arguments.page_size,
                "after": arguments.after,
                "max_items": arguments.max_items,
                "stop_on_failure": arguments.stop_on_failure,
            }
        try:
            started = api.mutate(
                f"{_PATH}/{quote(arguments.name, safe='')}/runs", body, resource="data move run"
            )
        except CayuApiError as exc:
            if exc.category == "outcome_unknown" and body.get("run_id"):
                raise _error(
                    "outcome_unknown",
                    "The Agent didn't confirm the start; it may have started. Look with "
                    f"`cayu cloud data-move runs --run-id {body['run_id']} --application "
                    f"{arguments.application}` before starting again.",
                ) from None
            raise
        result = started
        if not arguments.no_wait:
            result = _wait(api, started["run_id"], arguments.wait_seconds)
        output: dict[str, Any] = {"operation": f"data_move.{action}", "result": result}
        if not arguments.no_wait and not _clean(result):
            # As `cayu data-move run` does: a run that stopped or wasn't clean exits 3.
            output["_exit_status"] = EXIT_NOT_CLEAN
        return output
    except CayuApiError as exc:
        raise _command_error(exc, api=api) from None
    finally:
        api.close()


def _clean(run: Mapping[str, Any]) -> bool:
    report = run.get("report") or {}
    return run.get("status") == "finished" and bool(report.get("clean"))


def _read(api: CayuApi, run_id: str) -> dict[str, Any]:
    return api.get(f"{_PATH}/runs/{quote(run_id, safe='')}", resource="data move run")


def _wait(api: CayuApi, run_id: str, seconds: float) -> dict[str, Any]:
    deadline = time.monotonic() + max(seconds, 0.0)
    while True:
        run = _read(api, run_id)
        if run.get("status") != "running" or time.monotonic() >= deadline:
            return run
        time.sleep(_POLL_SECONDS)


def _error(category: str, message: str) -> Exception:
    from cayu.cli.cloud import CloudCommandError

    return CloudCommandError(category, message)


def _command_error(exc: CayuApiError, *, api: CayuApi | None = None) -> Exception:
    from cayu.cli.cloud import CloudCommandError

    if exc.category == "unsupported" and api is not None:
        return CloudCommandError(
            "unsupported",
            f"{api.description} runs a Cayu that doesn't serve data moves. Declare the moves "
            "in [tool.cayu.data_moves], deploy a release built with this CLI's Cayu version "
            "or newer, then try again.",
        )
    return CloudCommandError(exc.category, str(exc))


def render_run(output: Mapping[str, Any]) -> str:
    run = output["result"]
    report = run.get("report", {})
    counts = ", ".join(
        f"{name} {count}" for name, count in report.get("counts", {}).items() if count
    )
    mode = " (dry run: nothing written)" if report.get("dry_run") else ""
    lines = [
        f"Data move {run.get('move')}{mode}, run {run.get('run_id')}: {run.get('status')}, "
        f"{report.get('pages', 0)} page(s); {counts or 'no items yet'}."
    ]
    for item in report.get("items", [])[:20]:
        detail = f": {_cell(item['detail'], limit=200)}" if item.get("detail") else ""
        lines.append(f"  {item.get('outcome')} {_cell(item.get('key'), limit=200)}{detail}")
    stopped_by = report.get("stopped_by")
    if stopped_by:
        reason = f": {_cell(stopped_by['detail'], limit=500)}" if stopped_by.get("detail") else ""
        lines.append(f"  stopped at {_cell(stopped_by.get('key'), limit=200)}{reason}")
    for problem in report.get("verify_problems", []):
        lines.append(f"  verify: {_cell(problem, limit=500)}")
    if run.get("error"):
        lines.append(f"  stopped by {_cell(run['error'], limit=500)}")
    if run.get("status") == "running":
        lines.append(
            f"Still running; read it with `cayu cloud data-move runs --run-id {run.get('run_id')}`."
        )
    elif run.get("status") == "stopped":
        lines.append(
            f"Go on with `cayu cloud data-move resume {run.get('move')} --run-id "
            f"{run.get('run_id')}`."
        )
    elif report.get("clean"):
        lines.append("Clean: every page read, no conflict, failure or verify problem.")
    return "\n".join(lines)


def render_runs(output: Mapping[str, Any]) -> str:
    result = output["result"]
    if "run_id" in result:
        return render_run(output)
    moves = result.get("moves", [])
    lines = [
        "Declared moves: "
        + (
            ", ".join(
                f"{move['name']}{' (running)' if move.get('running') else ''}" for move in moves
            )
            or "none"
        )
    ]
    rows = [
        (
            _cell(run["run_id"]),
            _cell(run["move"]),
            _cell(run["status"]) + (" dry-run" if run["report"].get("dry_run") else ""),
            _cell(run["started_at"]),
            str(run["report"].get("pages", 0)),
        )
        for run in result.get("runs", [])
    ]
    lines.extend(
        _table(("RUN", "MOVE", "STATUS", "STARTED", "PAGES"), rows) if rows else ["No runs."]
    )
    return "\n".join(lines)

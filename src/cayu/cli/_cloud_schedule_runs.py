"""`cayu cloud schedule runs` and `schedule run --wait-for`: where an Agent's schedule runs are.

Cloud lists the runs the platform still keeps (`GET /v1/applications/{id}/schedule-runs`),
newest first, each `pending`, `running`, `succeeded` or `failed`. A pending run says why it
hasn't started: when no Agent node has room for its pod, its `hint` carries the scheduler's own
message, since such a run writes no log lines until it starts. A Cloud from before the listing
answers 404, and an install that can't list its runs answers `schedule_runs_unsupported`.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Mapping
from typing import Any
from urllib.parse import quote

from cayu.cli._cloud_api import CloudApiClient, CloudApiError
from cayu.cli._cloud_releases import add_output_argument
from cayu.cli._cloud_settings import _cell, _table

#: What `--wait-for` waits for: the run's process to start, or the run to finish.
WAIT_FOR = ("start", "finish")
_FINISHED = frozenset({"succeeded", "failed"})
_STARTED = frozenset({"running", *_FINISHED})
_REASON_LIMIT = 400
_MISSING_MESSAGE = "This Cayu Cloud doesn't list schedule runs yet."


class ScheduleRunsError(CloudApiError):
    """A `cayu cloud` failure about a schedule run; ``details`` go into the error envelope."""

    def __init__(self, category: str, message: str, *, details: dict[str, object]) -> None:
        super().__init__(category, message)
        self.details = details


def add_runs_parser(commands: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    runs = commands.add_parser(
        "runs",
        help="List the schedule runs Cloud still keeps, with their status.",
        description=(
            "List the Agent's schedule runs the platform still keeps, newest first: each "
            "one's status (pending, running, succeeded, failed) and, for a pending or failed "
            "run, why. A run no Agent node has room for stays pending with the scheduler's "
            "reason, and writes no log lines until it starts."
        ),
    )
    runs.add_argument("name", nargs="?", help="Only this schedule's runs.")
    runs.add_argument("--application", required=True)
    add_output_argument(runs, render_runs)


def add_wait_for_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--wait-for",
        choices=WAIT_FOR,
        help=(
            "Once the run is started, wait until its process starts (start) or the run "
            "finishes (finish), for up to --wait-seconds. Exits 2 with the reason when the "
            "run fails, or is still pending or running when the time is up."
        ),
    )


def list_runs(
    client: CloudApiClient, application_id: str, schedule: str | None
) -> list[dict[str, Any]]:
    """The Agent's schedule runs, newest first, of ``schedule`` alone when given."""

    try:
        answer = client.request(
            "GET",
            f"/v1/applications/{quote(application_id, safe='')}/schedule-runs",
            query=None if schedule is None else {"schedule": schedule},
        )
    except CloudApiError as exc:
        # The Agent was just resolved, so a 404 without one of Cloud's codes, or a 405, means
        # the route itself is missing.
        if exc.status_code == 405 or (exc.status_code == 404 and exc.code is None):
            raise ScheduleRunsError(
                "schedule_runs_unsupported", _MISSING_MESSAGE, details={}
            ) from None
        raise
    items = answer.get("items")
    if not isinstance(items, list) or not all(isinstance(item, dict) for item in items):
        raise CloudApiError("api_response_invalid", "Cloud's list of schedule runs is invalid.")
    return items


def runs_command(
    client: CloudApiClient, application_id: str, schedule: str | None
) -> dict[str, Any]:
    return {
        "operation": "schedule.runs",
        "result": {
            "application": application_id,
            "schedule": schedule,
            "items": list_runs(client, application_id, schedule),
        },
    }


def wait_for_run(
    client: CloudApiClient,
    *,
    application_id: str,
    schedule: str,
    run: str,
    until: str,
    poll_seconds: float,
    wait_seconds: float,
    sleep: Callable[[float], None],
    monotonic: Callable[[], float],
) -> dict[str, Any]:
    """Read the run until it started (``until`` `start`) or finished (`finish`).

    Answers the run as Cloud last listed it. Raises `ScheduleRunsError` when it failed, or
    when ``wait_seconds`` passed first, with why it is still pending; each change of its
    status or reason is written to standard error as it is seen.
    """

    wanted = _STARTED if until == "start" else _FINISHED
    deadline = monotonic() + wait_seconds
    seen: tuple[object, object] | None = None
    state: dict[str, Any] | None = None
    while True:
        try:
            runs = list_runs(client, application_id, schedule)
        except CloudApiError as exc:
            if not isinstance(exc, ScheduleRunsError) and exc.code != "schedule_runs_unsupported":
                raise
            raise ScheduleRunsError(
                "schedule_runs_unsupported",
                f"Run {run} started, but this Cayu Cloud can't list schedule runs, so "
                "--wait-for can't follow it. Its output is in "
                f"`cayu cloud service logs --process schedule:{schedule}`.",
                details={"run": run},
            ) from None
        state = next((item for item in runs if item.get("run") == run), state)
        if state is not None:
            status = state.get("status")
            if (status, state.get("reason")) != seen:
                seen = (status, state.get("reason"))
                print(f"Run {run}: {_state_line(state)}", file=sys.stderr)
            if status == "failed":
                raise ScheduleRunsError(
                    "schedule_run_failed",
                    f"Run {run} of schedule {schedule} failed. {_why(state)}".rstrip(),
                    details={"run": state},
                )
            if status in wanted:
                return state
        if monotonic() >= deadline:
            break
        sleep(poll_seconds)
    if state is None:
        raise ScheduleRunsError(
            "schedule_run_wait_timeout",
            f"Cloud listed no run {run} of schedule {schedule} within {wait_seconds:g} seconds.",
            details={"run": run, "waited_seconds": wait_seconds},
        )
    raise ScheduleRunsError(
        "schedule_run_wait_timeout",
        f"Run {run} of schedule {schedule} is still {state.get('status')} after "
        f"{wait_seconds:g} seconds. {_why(state)}".rstrip(),
        details={"run": state, "waited_seconds": wait_seconds},
    )


def render_runs(output: Mapping[str, Any]) -> str:
    result = output["result"]
    items = result["items"]
    if not items:
        scope = (
            "" if result.get("schedule") is None else f" of schedule {_cell(result['schedule'])}"
        )
        return f"No schedule runs{scope} for {_cell(result['application'])}."
    lines = _table(
        ("RUN", "SCHEDULE", "STATUS", "BY", "CREATED", "STARTED", "FINISHED", "EXIT"),
        [
            (
                _cell(item.get("run")),
                _cell(item.get("schedule")),
                _cell(item.get("status")),
                "run now" if item.get("on_demand") else "schedule",
                _cell(item.get("created_at")),
                _cell(item.get("started_at")),
                _cell(item.get("finished_at")),
                _cell(item.get("exit_code")),
            )
            for item in items
        ],
    )
    explained = [item for item in items if item.get("reason") or item.get("hint")]
    if explained:
        lines.append("")
        lines.extend(f"{_cell(item.get('run'))}: {_why(item)}" for item in explained)
    return "\n".join(lines)


def _state_line(state: Mapping[str, Any]) -> str:
    why = _why(state)
    return f"{_cell(state.get('status'))}" + (f". {why}" if why else "")


def _why(state: Mapping[str, Any]) -> str:
    return " ".join(
        _cell(text, limit=_REASON_LIMIT)
        for text in (state.get("reason"), state.get("hint"))
        if isinstance(text, str) and text
    )

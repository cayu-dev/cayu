"""`cayu cloud compound`: run Compound missions on an Agent without the web app.

Each command calls Cayu Cloud's Compound routes (`/v1/compound/...`) with the `cayu cloud
login` (or a selected context's API key) and prints one JSON document. A mission is started on
one of the Agent's Releases, waits for a person to approve its plan before it spends anything,
asks questions a person answers, and proposes changes a person approves or rejects. Writes take
`--idempotency-key` (`_cloud_writes`).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast
from urllib.parse import quote

from cayu.cli._cloud_api import CloudApiClient, CloudApiError
from cayu.cli._cloud_compass_findings import DIMENSIONS
from cayu.cli._cloud_writes import (
    add_idempotency_key_argument,
    add_message_arguments,
    message_text,
    send_write,
    write_key,
)
from cayu.cli._output import terminal_safe

WATCH_DEFAULT_POLL_SECONDS = 15.0
WATCH_DEFAULT_WAIT_SECONDS = 6 * 60 * 60.0
# A hosted mission's statuses once it has ended for good.
_HOSTED_ENDED = frozenset({"completed", "failed", "stopped"})
# The commands that change a mission, which belongs to one Organization.
MISSION_WRITES = ("approve-plan", "steer", "pause", "resume", "stop")


def add_compound_parser(
    commands: argparse._SubParsersAction[argparse.ArgumentParser],
    *,
    seconds: Callable[[str], float],
) -> None:
    """Add `compound` to `cayu cloud`; ``seconds`` parses the wait arguments."""

    compound = commands.add_parser(
        "compound",
        help="Start, steer and review Compound missions on your Agents.",
        description=(
            "Run Compound missions on an Agent: start one on a Release, approve its plan, "
            "answer what it asks, and approve or reject the changes it keeps. Every command "
            "prints one JSON document."
        ),
    )
    actions = compound.add_subparsers(dest="compound_command", required=True)

    start = actions.add_parser(
        "start",
        help="Start a mission on an Agent's Release.",
        description=(
            "Start a mission on one of the Agent's Releases (its selected one by default). "
            "The mission waits for its plan's approval (`approve-plan`) before it spends "
            "anything, unless --without-pre-spend-approval."
        ),
    )
    start.add_argument("--application", required=True, help="The Agent's name, slug or id.")
    start.add_argument("--release", help="The Release to improve (default: the selected one).")
    start.add_argument("--brief", required=True, type=Path, help="The mission's brief, as JSON.")
    start.add_argument(
        "--budget",
        type=Path,
        help=(
            "The mission's resources, as JSON, merged over the install's defaults: for example "
            '{"max_cost_microunits": 50000000, "max_total_tokens": 20000000, '
            '"max_wall_seconds": 86400}. A mission keeps at least one of these limits.'
        ),
    )
    start.add_argument(
        "--unbounded",
        action="store_true",
        help="Start the mission without spend limits, where the install allows it.",
    )
    start.add_argument(
        "--eval-target",
        help="The declared eval target to evaluate on, when the Release declares several.",
    )
    start.add_argument(
        "--acceptance-dimension",
        choices=DIMENSIONS,
        help=(
            "What the mission improves: correctness, or the cost, time or attention it "
            "minimizes while holding correctness, which the Release's objectives must set a "
            "target for. By default, the dimension the brief's Compass findings name, else "
            "correctness."
        ),
    )
    start.add_argument(
        "--without-pre-spend-approval",
        action="store_true",
        help="Start spending at once where the Compound host can't pause for plan approval.",
    )
    start.add_argument(
        "--adopt",
        dest="adopt_mission_id",
        metavar="MISSION_ID",
        help="After `start_outcome_unknown`: the hosted mission this start submitted.",
    )
    add_idempotency_key_argument(start)

    actions.add_parser(
        "readiness",
        help="Whether a mission can start on an Agent now.",
        description=(
            "Whether a mission can start on the Agent's selected Release: its declared eval "
            "targets, whether the host pauses for plan approval, the install's default spend "
            "limits, and what would refuse a start."
        ),
    ).add_argument("--application", required=True, help="The Agent's name, slug or id.")

    evaluation = actions.add_parser(
        "evaluation",
        help="The settings every mission on an Agent gives its evaluation.",
        description="The non-secret settings every mission on an Agent gives its evaluation.",
    )
    evaluation_actions = evaluation.add_subparsers(dest="evaluation_command", required=True)
    evaluation_actions.add_parser(
        "show", description="The Agent's evaluation settings."
    ).add_argument("--application", required=True, help="The Agent's name, slug or id.")
    declare = evaluation_actions.add_parser(
        "set", description="Replace the Agent's evaluation settings with exactly these."
    )
    declare.add_argument("--application", required=True, help="The Agent's name, slug or id.")
    declare.add_argument(
        "--setting",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help=(
            "A setting the evaluation reads; repeat for each, or leave out to clear them. Kept "
            "as plain text, so never a key, token or password."
        ),
    )

    approve_plan = actions.add_parser(
        "approve-plan",
        help="Approve a mission's plan so it starts spending.",
        description=(
            "Approve a sealed mission's plan (its ceilings and baseline bounds) so it starts. "
            "`status` shows the plan as compound.approval.plan, with its plan_fingerprint."
        ),
    )
    approve_plan.add_argument("mission_id")
    approve_plan.add_argument(
        "--plan",
        required=True,
        dest="plan_fingerprint",
        metavar="FINGERPRINT",
        help="The fingerprint of the plan being approved; a changed plan is refused.",
    )
    add_idempotency_key_argument(approve_plan)

    listing = actions.add_parser(
        "list", help="Your Organization's missions.", description="Your Organization's missions."
    )
    listing.add_argument("--application", help="Only this Agent's missions.")
    actions.add_parser(
        "status",
        help="One mission.",
        description="One mission: Cloud's record of it and Compound's status.",
    ).add_argument("mission_id")

    watch = actions.add_parser(
        "watch",
        help="Wait until a mission completes or waits on a person.",
        description=(
            "Wait on a mission until it ends, or with --until waiting (the default) until it "
            "waits on a person: an open request, an unreviewed change, or a pause. Prints the "
            "mission as it is then."
        ),
    )
    watch.add_argument("mission_id")
    watch.add_argument("--until", choices=("waiting", "completed"), default="waiting")
    watch.add_argument("--poll-seconds", type=seconds, default=WATCH_DEFAULT_POLL_SECONDS)
    watch.add_argument("--wait-seconds", type=seconds, default=WATCH_DEFAULT_WAIT_SECONDS)

    steer = actions.add_parser(
        "steer",
        help="Send the mission a directive.",
        description="Send the mission's root a durable directive.",
    )
    steer.add_argument("mission_id")
    add_message_arguments(steer)
    add_idempotency_key_argument(steer)
    for name in ("pause", "resume"):
        command = actions.add_parser(
            name,
            help=f"{name.capitalize()} the mission.",
            description=f"{name.capitalize()} the mission.",
        )
        command.add_argument("mission_id")
        command.add_argument("--reason")
        add_idempotency_key_argument(command)
    actions.add_parser(
        "stop",
        help="Stop the mission for good.",
        description=(
            "Stop the mission for good: one submitted, waiting for its plan's approval, or "
            "running. Stopping a stopped mission again answers it."
        ),
    ).add_argument("mission_id")

    requests = actions.add_parser(
        "requests",
        help="What the mission asks a person.",
        description="The requests the mission is waiting on a person for, and their answers.",
    )
    request_actions = requests.add_subparsers(dest="requests_command", required=True)
    request_actions.add_parser("list", description="The mission's requests.").add_argument(
        "mission_id"
    )
    answer = request_actions.add_parser("answer", description="Answer one request.")
    answer.add_argument("mission_id")
    answer.add_argument("request_id")
    add_message_arguments(answer)
    add_idempotency_key_argument(answer)

    candidates = actions.add_parser(
        "candidates",
        help="The candidates the mission tried.",
        description="The candidates the mission tried, and what it decided about each.",
    )
    candidate_actions = candidates.add_subparsers(dest="candidates_command", required=True)
    candidate_list = candidate_actions.add_parser("list", description="The mission's candidates.")
    candidate_list.add_argument("mission_id")
    candidate_list.add_argument("--offset", type=_non_negative, default=0)
    candidate_list.add_argument("--limit", type=_positive, default=100)
    candidate_show = candidate_actions.add_parser("show", description="One candidate.")
    candidate_show.add_argument("mission_id")
    candidate_show.add_argument("candidate_id")

    decisions = actions.add_parser(
        "decisions",
        help="The mission's decisions, with reasons.",
        description="Every development decision the mission made, with its reasons.",
    )
    decisions.add_argument("mission_id")
    decisions.add_argument("--offset", type=_non_negative, default=0)
    decisions.add_argument("--limit", type=_positive, default=50)

    changes = actions.add_parser(
        "changes",
        help="The changes the mission kept, and their review.",
        description=(
            "The changes the mission kept. Approving one makes a candidate Release of it that "
            "`cayu cloud deployment promote` can promote; rejecting one says why."
        ),
    )
    change_actions = changes.add_subparsers(dest="changes_command", required=True)
    change_actions.add_parser("list", description="The mission's kept changes.").add_argument(
        "mission_id"
    )
    change_show = change_actions.add_parser("show", description="One kept change.")
    change_show.add_argument("mission_id")
    change_show.add_argument("change_id")
    change_approve = change_actions.add_parser(
        "approve", description="Approve a kept change, making a candidate Release of it."
    )
    change_approve.add_argument("mission_id")
    change_approve.add_argument("change_id")
    change_approve.add_argument(
        "--expires-in-seconds",
        type=_positive,
        help="How long the candidate Release can be promoted (default 7 days).",
    )
    add_idempotency_key_argument(change_approve)
    change_reject = change_actions.add_parser("reject", description="Reject a kept change.")
    change_reject.add_argument("mission_id")
    change_reject.add_argument("change_id")
    change_reject.add_argument("--reason", required=True)


def compound_command(
    arguments: argparse.Namespace,
    *,
    client: CloudApiClient,
    application_id: str | None,
    is_transient: Callable[[CloudApiError], bool],
    transient_limit: int,
    validate_wait: Callable[[float, float], None],
) -> dict[str, Any]:
    """Run one `cayu cloud compound` command and return its JSON answer.

    ``application_id`` is the Agent `--application` names, already resolved, or None.
    """

    action = arguments.compound_command
    if action == "start":
        assert application_id is not None
        payload: dict[str, object] = {
            "application_id": application_id,
            "brief": _json_file(arguments.brief, "brief"),
        }
        if arguments.release is not None:
            payload["release_id"] = arguments.release
        if arguments.budget is not None:
            payload["budget"] = _json_file(arguments.budget, "budget")
        if arguments.unbounded:
            payload["unbounded"] = True
        if arguments.eval_target is not None:
            payload["eval_target"] = arguments.eval_target
        if arguments.acceptance_dimension is not None:
            payload["acceptance_dimension"] = arguments.acceptance_dimension
        if arguments.without_pre_spend_approval:
            payload["without_pre_spend_approval"] = True
        if arguments.adopt_mission_id is not None:
            payload["adopt_mission_id"] = arguments.adopt_mission_id
        return _write(client, arguments, "start", "POST", "/v1/compound/missions", payload)
    if action == "readiness":
        assert application_id is not None
        return _answer(action, client.request("GET", f"{_agent(application_id)}/readiness"))
    if action == "evaluation":
        assert application_id is not None
        path = f"{_agent(application_id)}/evaluation"
        operation = f"evaluation.{arguments.evaluation_command}"
        if arguments.evaluation_command == "show":
            return _answer(operation, client.request("GET", path))
        settings = _settings(arguments.setting)
        return _answer(operation, client.request("PUT", path, payload={"settings": settings}))
    if action == "list":
        query = None if application_id is None else {"application_id": application_id}
        return _answer(action, client.request("GET", "/v1/compound/missions", query=query))
    mission = _mission(arguments.mission_id)
    if action == "status":
        return _answer(action, client.request("GET", mission))
    if action == "watch":
        validate_wait(arguments.poll_seconds, arguments.wait_seconds)
        return _answer(
            action,
            _watch(
                client,
                arguments,
                mission,
                is_transient=is_transient,
                transient_limit=transient_limit,
            ),
        )
    if action == "approve-plan":
        path = f"{mission}/approve-plan"
        fingerprint: dict[str, object] = {"plan_fingerprint": arguments.plan_fingerprint}
        return _write(client, arguments, action, "POST", path, fingerprint)
    if action == "steer":
        directive: dict[str, object] = {"message": message_text(arguments)}
        return _write(client, arguments, action, "POST", f"{mission}/steer", directive)
    if action in {"pause", "resume"}:
        reason: dict[str, object] = {} if arguments.reason is None else {"reason": arguments.reason}
        return _write(client, arguments, action, "POST", f"{mission}/{action}", reason)
    if action == "stop":
        return _answer(action, client.request("POST", f"{mission}/stop"))
    if action == "requests":
        operation = f"requests.{arguments.requests_command}"
        if arguments.requests_command == "list":
            return _answer(operation, client.request("GET", f"{mission}/requests"))
        path = f"{mission}/requests/{quote(arguments.request_id, safe='')}/answer"
        return _write(
            client, arguments, operation, "POST", path, {"message": message_text(arguments)}
        )
    if action == "candidates":
        operation = f"candidates.{arguments.candidates_command}"
        if arguments.candidates_command == "list":
            query = {"offset": str(arguments.offset), "limit": str(arguments.limit)}
            return _answer(operation, client.request("GET", f"{mission}/candidates", query=query))
        path = f"{mission}/candidates/{quote(arguments.candidate_id, safe='')}"
        return _answer(operation, client.request("GET", path))
    if action == "decisions":
        query = {"offset": str(arguments.offset), "limit": str(arguments.limit)}
        return _answer(action, client.request("GET", f"{mission}/decisions", query=query))
    operation = f"changes.{arguments.changes_command}"
    if arguments.changes_command == "list":
        return _answer(operation, client.request("GET", f"{mission}/changes"))
    change = f"{mission}/changes/{quote(arguments.change_id, safe='')}"
    if arguments.changes_command == "show":
        return _answer(operation, client.request("GET", change))
    if arguments.changes_command == "approve":
        approval: dict[str, object] = {}
        if arguments.expires_in_seconds is not None:
            approval["expires_in_seconds"] = arguments.expires_in_seconds
        return _write(client, arguments, operation, "POST", f"{change}/approve", approval)
    return _answer(
        operation,
        client.request("POST", f"{change}/reject", payload={"reason": arguments.reason}),
    )


def watch_satisfied(detail: dict[str, Any], until: str) -> bool:
    """Whether a mission's detail meets ``--until``: ended, or waiting on a person."""

    status, hosted_status = _statuses(detail)
    if status == "completed" or hosted_status in _HOSTED_ENDED:
        return True
    if until == "completed":
        return False
    waiting = detail.get("waiting_on_person")
    counts = waiting.values() if isinstance(waiting, dict) else ()
    return status == "paused" or any(
        isinstance(value, int) and not isinstance(value, bool) and value > 0 for value in counts
    )


def _watch(
    client: CloudApiClient,
    arguments: argparse.Namespace,
    path: str,
    *,
    is_transient: Callable[[CloudApiError], bool],
    transient_limit: int,
) -> dict[str, Any]:
    """Read the mission until it meets ``--until``, riding out brief API outages, and say on
    standard error what it waits for whenever the mission's status changes."""

    deadline = _monotonic() + arguments.wait_seconds
    failures = 0
    polls = 0
    said: str | None = None
    while True:
        try:
            detail = client.request("GET", path)
        except CloudApiError as exc:
            if not is_transient(exc) or failures >= transient_limit:
                raise
            failures += 1
        else:
            failures = 0
            polls += 1
            if watch_satisfied(detail, arguments.until):
                return {"mission": detail, "polls": polls, "until": arguments.until}
            status = _status_line(detail)
            if status != said:
                print(
                    f"cayu cloud: waiting for mission {terminal_safe(arguments.mission_id)} "
                    f"({'to end' if arguments.until == 'completed' else 'to need a person'}); "
                    f"{status}",
                    file=sys.stderr,
                )
                said = status
        remaining = deadline - _monotonic()
        if remaining <= 0:
            raise CloudApiError(
                "wait_timeout",
                f"Mission {arguments.mission_id} didn't reach '{arguments.until}' within "
                f"{arguments.wait_seconds:g} seconds; run watch again to keep waiting.",
            )
        _sleep(min(arguments.poll_seconds, remaining))


def _status_line(detail: dict[str, Any]) -> str:
    status, hosted_status = _statuses(detail)
    return f"it is {terminal_safe(str(status or hosted_status or 'unknown'))}."


def _statuses(detail: dict[str, Any]) -> tuple[object, object]:
    """Compound's status for the mission, and the hosted mission's status in Cloud."""

    compound = _object(detail.get("compound"))
    return _object(compound.get("mission")).get("status"), _object(detail.get("hosted")).get(
        "status"
    )


def _object(value: object) -> dict[str, Any]:
    return cast("dict[str, Any]", value) if isinstance(value, dict) else {}


def _json_file(path: Path, what: str) -> dict[str, Any]:
    try:
        loaded = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise CloudApiError("invalid_input", f"Could not read the {what} JSON in {path}.") from exc
    if not isinstance(loaded, dict):
        raise CloudApiError("invalid_input", f"The {what} in {path} must be a JSON object.")
    return loaded


def _settings(pairs: list[str]) -> dict[str, str]:
    settings: dict[str, str] = {}
    for pair in pairs:
        name, separator, value = pair.partition("=")
        if not separator or not name:
            raise CloudApiError("invalid_input", f"A setting is NAME=VALUE, not {pair!r}.")
        if name in settings:
            raise CloudApiError("invalid_input", f"The setting {name} is given twice.")
        settings[name] = value
    return settings


def _agent(application_id: str) -> str:
    return f"/v1/compound/agents/{quote(application_id, safe='')}"


def _mission(mission_id: str) -> str:
    return f"/v1/compound/missions/{quote(mission_id, safe='')}"


def _answer(operation: str, result: dict[str, Any]) -> dict[str, Any]:
    return {"operation": f"compound.{operation}", "result": result}


def _write(
    client: CloudApiClient,
    arguments: argparse.Namespace,
    operation: str,
    method: str,
    path: str,
    payload: dict[str, object],
) -> dict[str, Any]:
    key = write_key(arguments, f"compound-{operation}")
    result = send_write(client, method, path, key=key, payload=payload)
    return {**_answer(operation, result), "idempotency_key": key}


def _positive(value: str) -> int:
    number = _integer(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive whole number")
    return number


def _non_negative(value: str) -> int:
    number = _integer(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be zero or a positive whole number")
    return number


def _integer(value: str) -> int:
    try:
        return int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a whole number") from exc


def _sleep(seconds: float) -> None:
    """Wait; tests replace this name, never the `time` module other code shares."""

    time.sleep(seconds)


def _monotonic() -> float:
    return time.monotonic()

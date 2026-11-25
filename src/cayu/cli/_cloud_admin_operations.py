"""`cayu cloud admin runtimes|cleanup|activity-reader`: repairs on one Organization's Agent.

Unlike the rest of `cayu cloud admin`, these call the Organization's own routes, so they act in
the Organization the command acts in (`--organization`, else the login's) and take the Agent's
name, slug or id with `--application`, like the Organization's other commands. Cloud decides
who may: a developer or admin of that Organization for `runtimes` and `cleanup`, and an
Organization admin for `activity-reader`.

- `runtimes hold|release-hold|retire`: keep a retained runtime artifact from being retired,
  let it go again, or retire it now;
- `cleanup`: destroy one of the Agent's Releases and wait until Cloud reports it destroyed;
- `activity-reader show|setup|restart`: the reader Cloud uses to follow the Agent's Runtime
  activity, which Cloud sets up itself; `setup` runs that now and `restart` reads the Agent's
  store again from its beginning, as after a restore.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Callable
from typing import Any
from urllib.parse import quote

from cayu.cli._cloud_admin_platform import nonempty_value
from cayu.cli._cloud_api import CloudApiClient, CloudApiError
from cayu.cli._output import terminal_safe

OPERATION_GROUPS = frozenset({"runtimes", "cleanup", "activity-reader"})
# The commands that change an Agent, which are refused outside where the Agent deploys.
CHANGING_COMMANDS = (
    ("runtimes", "hold"),
    ("runtimes", "release-hold"),
    ("runtimes", "retire"),
    ("cleanup",),
    ("activity-reader", "setup"),
    ("activity-reader", "restart"),
)
CLEANUP_DEFAULT_POLL_SECONDS = 5.0
CLEANUP_DEFAULT_WAIT_SECONDS = 900.0
# A Release's statuses that mean it will never be destroyed by this request.
_RELEASE_ENDED_OTHERWISE = frozenset({"cancelled", "failed"})
_APPLICATION_HELP = "The Agent's name, slug or id."


def add_operation_parsers(
    actions: argparse._SubParsersAction[argparse.ArgumentParser],
    *,
    seconds: Callable[[str], float],
    organization_help: str,
) -> None:
    """Add `runtimes`, `cleanup` and `activity-reader` to `cayu cloud admin`; ``seconds``
    parses the wait arguments and ``organization_help`` describes `--organization`."""

    runtimes = actions.add_parser(
        "runtimes",
        help="Hold, release or retire one of an Agent's retained runtime artifacts.",
        description=(
            "Change the lifecycle of a runtime artifact Cloud retains for an Agent: hold it so "
            "it is never retired, release that hold, or retire it now. `cayu cloud runtimes "
            "list|status` show them."
        ),
    )
    runtime_commands = runtimes.add_subparsers(dest="runtimes_command", required=True)
    for action, summary in (
        ("hold", "Keep the artifact from being retired, with a reason Cloud records."),
        ("release-hold", "Release the artifact's hold, so it can be retired again."),
        ("retire", "Retire the artifact now."),
    ):
        command = runtime_commands.add_parser(action, help=summary, description=summary)
        command.add_argument("artifact_id", help="The runtime artifact's id.")
        command.add_argument("--application", required=True, help=_APPLICATION_HELP)
        if action == "hold":
            command.add_argument("--reason", required=True, help="Why it must be kept.")

    cleanup = actions.add_parser(
        "cleanup",
        help="Destroy one of an Agent's Releases and wait until it is destroyed.",
        description=(
            "Destroy the Release --deployment names, then read it every --poll-seconds until "
            "Cloud reports it destroyed, for up to --wait-seconds. Exits 2 when the Release "
            "ends another way or the wait runs out."
        ),
    )
    cleanup.add_argument("--application", required=True, help=_APPLICATION_HELP)
    cleanup.add_argument("--deployment", required=True, help="The Release's id.")
    cleanup.add_argument(
        "--poll-seconds",
        type=seconds,
        default=CLEANUP_DEFAULT_POLL_SECONDS,
        help=f"How often to read the Release (default {CLEANUP_DEFAULT_POLL_SECONDS:g}).",
    )
    cleanup.add_argument(
        "--wait-seconds",
        type=seconds,
        default=CLEANUP_DEFAULT_WAIT_SECONDS,
        help=f"How long to wait (default {CLEANUP_DEFAULT_WAIT_SECONDS:g}).",
    )

    reader = actions.add_parser(
        "activity-reader",
        help="Show, set up or restart the reader Cloud follows an Agent's activity with.",
        description=(
            "The reader Cloud follows an Agent's Runtime activity with, which Cloud sets up "
            "itself: show its configuration, set it up now, or restart it after the Agent's "
            "store was restored."
        ),
    )
    reader_commands = reader.add_subparsers(dest="activity_reader_command", required=True)
    for action, summary in (
        ("show", "Show the reader's configuration, availability error and revision."),
        (
            "setup",
            "Run Cloud's setup now: prepare a first reader and activate it once the Agent's "
            "Runtime answers.",
        ),
        ("restart", "Read the Agent's store again from its beginning, as after a restore."),
    ):
        command = reader_commands.add_parser(action, help=summary, description=summary)
        command.add_argument("--application", required=True, help=_APPLICATION_HELP)
        if action == "show":
            # The changing commands get theirs with the other changes' (`_CHANGING_COMMANDS`).
            command.add_argument(
                "--organization",
                dest="intended_organization",
                metavar="ORG_ID",
                default=argparse.SUPPRESS,
                help=organization_help,
            )


def operation_command(
    arguments: argparse.Namespace,
    *,
    client: CloudApiClient,
    application_id: Callable[[CloudApiClient, str], str],
    validate_wait: Callable[[float, float], None],
    is_transient: Callable[[CloudApiError], bool],
    transient_limit: int,
) -> dict[str, Any]:
    """Run one of `OPERATION_GROUPS`; ``application_id`` resolves `--application`."""

    group = arguments.admin_command
    if group == "cleanup":
        # Before anything is destroyed.
        validate_wait(arguments.poll_seconds, arguments.wait_seconds)
    agent = application_id(client, str(arguments.application))
    base = f"/v1/applications/{quote(agent, safe='')}"
    if group == "runtimes":
        action = arguments.runtimes_command
        artifact = quote(nonempty_value(arguments.artifact_id, "artifact id"), safe="")
        path = f"{base}/runtime-artifacts/{artifact}/{action}"
        payload: dict[str, object] | None = (
            {"reason": nonempty_value(arguments.reason, "--reason")} if action == "hold" else None
        )
        result = client.request("POST", path, payload=payload)
        return {
            "operation": f"admin.runtimes.{action}",
            "result": {"application_id": agent, "artifact": result},
        }
    if group == "cleanup":
        release = f"{base}/deployments/{quote(nonempty_value(arguments.deployment, '--deployment'), safe='')}"
        client.request("POST", f"{release}/destroy")
        return {
            "operation": "admin.cleanup",
            "result": _wait_destroyed(
                client,
                release,
                arguments,
                is_transient=is_transient,
                transient_limit=transient_limit,
            ),
        }
    return _activity_reader(arguments, client, f"{base}/activity-reader", agent)


def _wait_destroyed(
    client: CloudApiClient,
    path: str,
    arguments: argparse.Namespace,
    *,
    is_transient: Callable[[CloudApiError], bool],
    transient_limit: int,
) -> dict[str, Any]:
    # The clock is looked up at each call, so a test can replace it.
    deadline = time.monotonic() + arguments.wait_seconds
    failures = 0
    said: object = None
    while True:
        try:
            release = client.request("GET", path)
        except CloudApiError as exc:
            if not is_transient(exc) or failures >= transient_limit:
                raise
            failures += 1
        else:
            failures = 0
            status = release.get("status")
            if status == "destroyed":
                return release
            if status in _RELEASE_ENDED_OTHERWISE:
                raise CloudApiError(
                    "release_cleanup_failed",
                    f"Release {arguments.deployment} ended {status!s}, not destroyed.",
                )
            if status != said:
                print(
                    f"cayu cloud: waiting for Release {terminal_safe(str(arguments.deployment))} "
                    f"to be destroyed; it is {terminal_safe(str(status))}.",
                    file=sys.stderr,
                )
                said = status
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise CloudApiError(
                "wait_timeout",
                f"Release {arguments.deployment} wasn't destroyed within "
                f"{arguments.wait_seconds:g} seconds; run cleanup again to keep waiting.",
            )
        # The last sleep is cut to what is left of the wait, so the Release is read once more.
        time.sleep(min(arguments.poll_seconds, remaining))


def _activity_reader(
    arguments: argparse.Namespace, client: CloudApiClient, path: str, agent: str
) -> dict[str, Any]:
    action = arguments.activity_reader_command
    if action == "show":
        result = client.request("GET", path)
    elif action == "setup":
        result = client.request("PUT", path, payload={"action": "setup"})
    else:
        reader = client.request("GET", path).get("reader")
        if not isinstance(reader, dict):
            raise CloudApiError(
                "activity_reader_not_set_up",
                f"Agent {agent} has no activity reader yet; `cayu cloud admin activity-reader "
                "setup` sets one up.",
            )
        revision = reader.get("revision")
        if revision is None:
            raise CloudApiError("api_response_invalid", "The activity reader has no revision.")
        # Cloud restarts it only at the revision just read, so a change since is refused.
        result = client.request(
            "PUT", path, payload={"action": "restart", "expected_revision": revision}
        )
    return {"operation": f"admin.activity-reader.{action}", "result": result}

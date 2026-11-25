"""`cayu cloud deployment cancel|destroy|resume`: stop, remove or resume one Release's work.

Each sends Cloud's lifecycle request for one Release of an Agent and prints the Release's
workflow as Cloud answers it (`requested_action`, `current_phase`, `paused_at`,
`last_error`). Cloud acts on the request in the background:

- `cancel` stops a Release that is still being built or checked, and retires what it made.
  A smoke-tested Release, a candidate among them, can still be cancelled; Cloud refuses a
  promoted, failed or destroyed one, and one being published to the Agent's service.
- `destroy` removes a promoted, failed or cancelled Release and its runtime resources for good,
  so it can't be promoted or rolled back to. Cloud refuses the Release the Agent selects.
- `resume` continues a Release whose workflow Cloud paused (`paused_at` set). Cloud refuses
  one that isn't paused, or whose deadline has passed.

Asking again for a cancel or destroy Cloud already has does nothing more, so neither takes an
idempotency key. `cancel` and `destroy` take `--wait` to follow the Release until Cloud reports
it `cancelled` or `destroyed`.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Callable
from typing import Any
from urllib.parse import quote

from cayu.cli._cloud_api import CloudApiClient, CloudApiError
from cayu.cli._output import terminal_safe

LIFECYCLE_ACTIONS = ("cancel", "destroy", "resume")
# The status each waited action ends in.
_EXPECTED = {"cancel": "cancelled", "destroy": "destroyed"}
# Statuses a Release Cloud was asked to cancel can't leave for `cancelled`: Cloud cancels one
# `accepted` or in any of its build phases, `smoke_tested` included.
_NOT_CANCELLED = frozenset({"destroyed", "failed", "promoted"})
_DESCRIPTIONS = {
    "cancel": (
        "Cancel a Release that is still being built or checked: Cloud stops its work and "
        "retires what it made, and the Release ends `cancelled`. A smoke-tested Release, a "
        "candidate among them, can still be cancelled; Cloud refuses a promoted, failed or "
        "destroyed one, and one being published to the Agent's service."
    ),
    "destroy": (
        "Destroy a promoted, failed or cancelled Release for good: Cloud removes its runtime "
        "resources and it ends `destroyed`, so it can't be promoted or rolled back to. Cloud "
        "refuses the Release the Agent selects. There is no confirmation prompt."
    ),
    "resume": (
        "Resume a Release whose workflow Cloud paused (`paused_at` in its answer): Cloud "
        "continues its current phase. Cloud refuses one that isn't paused or whose deadline "
        "has passed. Follow it with `deployment wait`."
    ),
}
_HELP = {
    "cancel": "Cancel a Release still being built or checked.",
    "destroy": "Remove a Release that no longer serves, for good.",
    "resume": "Resume a Release whose workflow Cloud paused.",
}


def add_deployment_lifecycle_parsers(
    deployment_commands: argparse._SubParsersAction[argparse.ArgumentParser],
    *,
    application_help: str,
    seconds: Callable[[str], float],
    preflight_wait: Callable[[argparse.Namespace], None],
) -> None:
    """Add `cancel`, `destroy` and `resume` to `cayu cloud deployment`; ``preflight_wait``
    checks `--poll-seconds` and `--wait-seconds` when `--wait` is given."""

    for action in LIFECYCLE_ACTIONS:
        operation = deployment_commands.add_parser(
            action, help=_HELP[action], description=_DESCRIPTIONS[action]
        )
        operation.add_argument("deployment_id")
        operation.add_argument("--application", required=True, help=application_help)
        if action in _EXPECTED:
            operation.add_argument(
                "--wait",
                action="store_true",
                help=f"Wait until Cloud reports the Release `{_EXPECTED[action]}`.",
            )
            operation.add_argument("--poll-seconds", type=seconds, default=5.0)
            operation.add_argument("--wait-seconds", type=seconds, default=900.0)
            operation.set_defaults(_cloud_preflight=preflight_wait)


def deployment_lifecycle_command(
    arguments: argparse.Namespace,
    *,
    client: CloudApiClient,
    application_id: str,
    is_transient: Callable[[CloudApiError], bool],
    transient_limit: int,
) -> dict[str, Any]:
    """Send one lifecycle request for the Release of ``application_id`` (already resolved)."""

    action = arguments.deployment_command
    release = (
        f"/v1/applications/{quote(application_id, safe='')}"
        f"/deployments/{quote(arguments.deployment_id, safe='')}"
    )
    workflow = client.request("POST", f"{release}/{action}")
    response: dict[str, Any] = {"operation": f"deployment.{action}", "result": workflow}
    if getattr(arguments, "wait", False):
        response["release"] = _wait(
            client,
            arguments,
            release,
            expected=_EXPECTED[action],
            is_transient=is_transient,
            transient_limit=transient_limit,
        )
    return response


def _wait(
    client: CloudApiClient,
    arguments: argparse.Namespace,
    path: str,
    *,
    expected: str,
    is_transient: Callable[[CloudApiError], bool],
    transient_limit: int,
) -> dict[str, Any]:
    """Read the Release until it reaches ``expected``, riding out brief API outages, and say
    on standard error what it waits for whenever its status changes."""

    deadline = _monotonic() + arguments.wait_seconds
    name = terminal_safe(arguments.deployment_id)
    failures = 0
    said: object = None
    status: object = None
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
            if status == expected:
                return release
            if expected == "cancelled" and status in _NOT_CANCELLED:
                raise CloudApiError(
                    "deployment_not_cancelled",
                    f"Release {arguments.deployment_id} reached `{status}` instead of "
                    "`cancelled`; Cloud didn't cancel it.",
                )
            if status != said:
                print(
                    f"cayu cloud: waiting for Release {name} to be {expected}; it is "
                    f"{terminal_safe(str(status))}.",
                    file=sys.stderr,
                )
                said = status
        remaining = deadline - _monotonic()
        if remaining <= 0:
            raise CloudApiError(
                "wait_timeout",
                f"Release {arguments.deployment_id} wasn't {expected} within "
                f"{arguments.wait_seconds:g} seconds (last status: {status}); Cloud keeps "
                "working on it. Check it with `cayu cloud deployment status`.",
            )
        _sleep(min(arguments.poll_seconds, remaining))


def _sleep(seconds: float) -> None:
    """Wait; tests replace this name, never the `time` module other code shares."""

    time.sleep(seconds)


def _monotonic() -> float:
    return time.monotonic()

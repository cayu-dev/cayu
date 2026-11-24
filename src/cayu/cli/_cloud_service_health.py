"""`cayu cloud service health --application NAME`: an Agent's health in one read.

The command prints what `cayu.storage.agent_health` reads from the Agent's
`cayu.storage.open_cloud_agent_readers`: for each task type run since `--since` (24 hours by
default), when a task last completed and last failed, how many failed in a row since, and the
last failure's reason; how many inbox items wait, and since when; and the Agent's service
status. Tasks are read through Cloud's `agent-read` route when Cloud offers it, so an
Organization API key with the `read` scope is enough. A part that can't be read is named with
its reason, and the command still prints the rest. Nothing is changed.

It prints a summary by default; `--json` prints the JSON envelope the other `cayu cloud`
commands print.
"""

from __future__ import annotations

import argparse
import asyncio
import re
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

from cayu.cli._cloud_releases import _when, add_output_argument, render_failure
from cayu.cli._cloud_settings import _cell, _table

if TYPE_CHECKING:
    from cayu.cli._cloud_api import CloudApiClient

DEFAULT_SINCE = "24h"
_SPAN = re.compile(r"([1-9][0-9]{0,5})([mhd])")
_UNITS = {"m": "minutes", "h": "hours", "d": "days"}
_MAX_SPAN = timedelta(days=366)


def since_span(value: str) -> timedelta:
    """`--since`: how far back to look for task types, as `30m`, `24h` or `7d`."""

    span = _SPAN.fullmatch(value)
    if span is None:
        raise argparse.ArgumentTypeError(f"{value!r} is not a span like 30m, 24h or 7d")
    delta = timedelta(**{_UNITS[span[2]]: int(span[1])})
    if delta > _MAX_SPAN:
        raise argparse.ArgumentTypeError("must be at most 366d")
    return delta


def add_service_health_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--since",
        type=since_span,
        default=since_span(DEFAULT_SINCE),
        metavar="SPAN",
        help=(
            "Report the task types run in this span: 30m, 24h, 7d (24h). Each type's last "
            "completion and failure are its latest, however long ago."
        ),
    )
    add_output_argument(parser, render_service_health, failure=render_failure)


def service_health(
    arguments: argparse.Namespace,
    *,
    client: CloudApiClient,
    application_id: str,
    now: datetime | None = None,
) -> dict[str, Any]:
    from cayu.storage.agent_health_summary import agent_health
    from cayu.storage.agent_readers import cloud_agent_readers

    since = (now or datetime.now(UTC)) - arguments.since

    async def read() -> dict[str, Any]:
        readers = cloud_agent_readers(
            client,
            application_id,
            source={"kind": "cayu_cloud", "application": application_id},
            timeout_seconds=client.timeout_seconds,
        )
        async with readers:
            return (await agent_health(readers, since=since)).to_dict()

    return {"operation": "service.health", "result": asyncio.run(read())}


def render_service_health(result: Mapping[str, Any]) -> str:
    shown = result["result"]
    service = shown["service"]
    lines = [f"Agent: {_cell(shown['source'].get('application'))}"]
    if service is not None:
        status = _cell(service.get("status"))
        if service.get("paused") is True:
            status += " (paused)"
        lines.append(f"  Service: {status}")
        lines.append(f"  Deployment: {_cell(service.get('deployment_id'))}")
        if service.get("cayu_version") is not None:
            lines.append(f"  Cayu: {_cell(service.get('cayu_version'))}")
    inbox = shown["inbox"]
    if inbox is not None:
        waiting = ", ".join(f"{count} {kind}" for kind, count in inbox.items() if count) or "empty"
        oldest = shown["oldest_inbox_item_at"]
        lines.append(
            f"  Inbox: {_cell(waiting)}" + (f", oldest asked {_when(oldest)}" if oldest else "")
        )
    lines += ["", f"Tasks run since {_when(shown['since'])}:"]
    tasks = shown["tasks"]
    if tasks:
        rows = [
            (
                _cell(task["type"]),
                _when(task["last_completed_at"], missing="never"),
                _when(task["last_failed_at"], missing="never"),
                str(task["failure_streak"]),
                _cell(task["last_failure_reason"], limit=80),
            )
            for task in tasks.values()
        ]
        lines += [
            "  " + line
            for line in _table(
                ("TYPE", "LAST COMPLETED", "LAST FAILED", "FAILED IN A ROW", "LAST FAILURE"),
                rows,
            )
        ]
    else:
        lines.append("  none")
    if shown["unavailable"]:
        lines += ["", "Couldn't read:"]
        lines += [
            f"  {_cell(part['part'])} ({_cell(part['category'])}): {_cell(part['message'])}"
            for part in shown["unavailable"]
        ]
    return "\n".join(lines)

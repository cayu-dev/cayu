"""`cayu cloud sandboxes list|show --application NAME`: an Agent's brokered sandboxes in Cayu
Cloud, who created each, and what holds its places.

The commands read Cloud's sandbox broker routes (`GET /v1/applications/{id}/sandboxes`, one
sandbox, its events and the capacity; contract `cayu.cloud.sandbox` v1) with the `cayu cloud
login` or a selected context's API key, which Cloud accepts on those reads from anyone who may
see the Agent. Nothing here creates, renews or destroys a sandbox:
that stays with the Agent's own sandbox credential.

`list` shows each sandbox's status, when its lease and its maximum lifetime end, why it ended,
and the Release, session and task its create named (`provenance`), and the Agent's capacity
now: how many places are held under its limit and when the soonest lease runs out. A sandbox
that still holds a place but was created for a Release the Agent no longer selects is marked,
since that is what a redeploy leaves behind. `show` adds the sandbox's audit events.

These commands print a table by default; `--json` prints the JSON envelope the other `cayu
cloud` commands print, with `schema_version` in its result.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from typing import Any
from urllib.parse import quote

from cayu.cli._cloud_api import CloudApiClient, CloudApiError
from cayu.cli._cloud_releases import _when, add_output_argument, render_failure
from cayu.cli._cloud_settings import _cell, _table

SCHEMA_VERSION = 1
DEFAULT_LIMIT = 50
MAX_LIMIT = 1000
# Cloud's page size bound for an Agent's sandboxes.
_PAGE_LIMIT = 100
STATUSES = ("provisioning", "ready", "destroying", "destroyed", "failed")
# The statuses whose sandbox may still hold a place under the limits; a provisioning one only
# once it is out of the queue.
_HOLDING = frozenset({"provisioning", "ready", "destroying"})
_SANDBOX_ID_LIMIT = 128
_PROVENANCE = ("release_id", "session_id", "task_id")
_SANDBOX_FIELDS = (
    "id",
    "status",
    "image",
    "egress",
    "created_at",
    "ready_at",
    "lease_expires_at",
    "max_expires_at",
    "queued_until",
    "destroyed_at",
    "destroy_reason",
    "failure",
    "labels",
    "on_behalf_of",
    "owner_epoch",
)


def _limit(value: str) -> int:
    try:
        number = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value!r} is not a whole number") from None
    if not 1 <= number <= MAX_LIMIT:
        raise argparse.ArgumentTypeError(f"must be between 1 and {MAX_LIMIT}")
    return number


def _sandbox_id(value: str) -> str:
    if not value or len(value) > _SANDBOX_ID_LIMIT or not value.isprintable() or "/" in value:
        raise argparse.ArgumentTypeError("is not a sandbox id")
    return value


def add_sandboxes_parser(commands: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    sandboxes = commands.add_parser(
        "sandboxes",
        help="Read an Agent's brokered sandboxes.",
        description=(
            "Read the sandboxes Cayu Cloud brokers for an Agent: their status, leases, why "
            "they ended, and the Release, session and task each was created for."
        ),
    )
    actions = sandboxes.add_subparsers(dest="sandboxes_command", required=True)
    listing = actions.add_parser(
        "list",
        description=(
            "List the Agent's sandboxes, newest first, with the places they hold under its "
            "limit now."
        ),
    )
    listing.add_argument("--status", choices=STATUSES, help="Only sandboxes with this status.")
    listing.add_argument(
        "--limit",
        type=_limit,
        default=DEFAULT_LIMIT,
        help=f"How many to list, newest first (default {DEFAULT_LIMIT}, at most {MAX_LIMIT}).",
    )
    showing = actions.add_parser(
        "show", description="Show one sandbox and its audit events, oldest first."
    )
    showing.add_argument("sandbox_id", metavar="SANDBOX_ID", type=_sandbox_id)
    for parser, render in ((listing, render_sandboxes), (showing, render_sandbox)):
        parser.add_argument("--application", required=True)
        add_output_argument(parser, render, failure=render_failure)


def sandboxes_command(
    arguments: argparse.Namespace, *, client: CloudApiClient, application_id: str
) -> dict[str, Any]:
    base = f"/v1/applications/{quote(application_id, safe='')}"
    selected = _selected_release(client, base)
    if arguments.sandboxes_command == "show":
        path = f"{base}/sandboxes/{quote(arguments.sandbox_id, safe='')}"
        sandbox = _read(client, path)
        events = _read(client, f"{path}/events").get("items")
        return {
            "operation": "sandboxes.show",
            "result": {
                "schema_version": SCHEMA_VERSION,
                "application_id": application_id,
                "selected_release_id": selected,
                "sandbox": sandbox_summary(sandbox, selected_release_id=selected),
                "events": [_event(item) for item in events if isinstance(item, Mapping)]
                if isinstance(events, list)
                else [],
            },
        }
    items: list[Mapping[str, Any]] = []
    cursor: str | None = None
    while True:
        query = {"limit": str(min(_PAGE_LIMIT, arguments.limit - len(items)))}
        if arguments.status is not None:
            query["status"] = arguments.status
        if cursor is not None:
            query["cursor"] = cursor
        page = _read(client, f"{base}/sandboxes", query=query)
        found = page.get("items")
        if not isinstance(found, list):
            raise CloudApiError("api_response_invalid", "The sandbox list is invalid.")
        items.extend(item for item in found if isinstance(item, Mapping))
        next_cursor = page.get("next_cursor")
        cursor = next_cursor if isinstance(next_cursor, str) and next_cursor else None
        if cursor is None or len(items) >= arguments.limit:
            break
    return {
        "operation": "sandboxes.list",
        "result": {
            "schema_version": SCHEMA_VERSION,
            "application_id": application_id,
            "selected_release_id": selected,
            "capacity": _capacity(client, base),
            "items": [
                sandbox_summary(item, selected_release_id=selected)
                for item in items[: arguments.limit]
            ],
            "next_cursor": cursor,
        },
    }


def sandbox_summary(
    sandbox: Mapping[str, Any], *, selected_release_id: str | None
) -> dict[str, Any]:
    """The record's fields the commands show, its provenance, and whether it still holds a
    place for a Release the Agent no longer selects (None when that can't be told). A queued
    sandbox holds no place yet, so it is never marked."""

    summary = {name: sandbox.get(name) for name in _SANDBOX_FIELDS}
    provenance = sandbox.get("provenance")
    summary["provenance"] = (
        {name: _text(provenance.get(name)) for name in _PROVENANCE}
        if isinstance(provenance, Mapping)
        else None
    )
    release = None if summary["provenance"] is None else summary["provenance"]["release_id"]
    summary["older_release"] = (
        None
        if release is None or selected_release_id is None
        else (
            sandbox.get("status") in _HOLDING
            and sandbox.get("queued_until") is None
            and release != selected_release_id
        )
    )
    return summary


def _selected_release(client: CloudApiClient, base: str) -> str | None:
    selected = client.request("GET", base).get("current_deployment_id")
    return selected if isinstance(selected, str) and selected else None


def _capacity(client: CloudApiClient, base: str) -> dict[str, Any] | None:
    """The Agent's limit and what holds it now, or None from a Cloud that doesn't say."""

    try:
        capacity = client.request("GET", f"{base}/sandboxes/capacity")
    except CloudApiError:
        return None
    agent = capacity.get("agent")
    if not isinstance(agent, Mapping):
        return None
    return {
        "limited_by": _text(capacity.get("limited_by")),
        "available": _number(capacity.get("available")),
        "agent": {name: _number(agent.get(name)) for name in ("limit", "in_use", "queued")},
    }


def _read(
    client: CloudApiClient, path: str, *, query: dict[str, str] | None = None
) -> dict[str, Any]:
    try:
        return client.request("GET", path, query=query)
    except CloudApiError as exc:
        raise _sandbox_error(exc) from None


def _sandbox_error(error: CloudApiError) -> CloudApiError:
    """Say what a refused sandbox read means; the routes answer the contract's own body,
    which the generic client doesn't read."""

    messages = {
        401: (
            "Cayu Cloud didn't accept this login on the sandbox routes. A Cloud before "
            "cayu-tech/cayu-cloud#679 lets only the Agent itself read them."
        ),
        403: (
            "This login can't read the Agent's sandboxes: it needs to be one of the Agent's "
            "people, a developer or an admin, or an API key with the read scope."
        ),
        404: "Cayu Cloud has no such sandbox for this Agent.",
        503: "This Cayu Cloud runs no sandbox broker, or it is busy; nothing was read.",
    }
    message = messages.get(error.status_code or 0)
    if message is None:
        return error
    return CloudApiError(
        error.category,
        message,
        status_code=error.status_code,
        detail=error.detail,
        code=error.code,
        request=error.request,
    )


def _event(event: Mapping[str, Any]) -> dict[str, Any]:
    detail = event.get("detail")
    return {
        "at": event.get("at"),
        "kind": event.get("kind"),
        "actor": event.get("actor"),
        "detail": dict(detail) if isinstance(detail, Mapping) else {},
    }


def _text(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _number(value: object) -> int | None:
    return value if type(value) is int else None


# Rendering ----------------------------------------------------------------------------------


def render_sandboxes(result: Mapping[str, Any]) -> str:
    listed = result["result"]
    lines = [_capacity_line(listed.get("capacity"))]
    items = listed["items"]
    if not items:
        lines.append(f"No sandboxes for {_cell(listed['application_id'])}.")
        return "\n".join(lines)
    rows = [
        (
            _cell(item["id"]),
            _cell(item["status"]),
            _when(item["created_at"]),
            _lease(item),
            _ended(item),
            _provenance_cell(item, "release_id"),
            _provenance_cell(item, "session_id"),
            _provenance_cell(item, "task_id"),
        )
        for item in items
    ]
    lines += _table(
        ("SANDBOX", "STATUS", "CREATED", "LEASE ENDS", "ENDED", "RELEASE", "SESSION", "TASK"),
        rows,
    )
    older = [item for item in items if item.get("older_release")]
    if older:
        lines += [
            "",
            f"{len(older)} still {'holds a place' if len(older) == 1 else 'hold places'} for a "
            f"Release the Agent no longer selects ({_cell(listed['selected_release_id'])}) "
            "and frees it when its lease ends unless it's renewed (marked *).",
        ]
    if listed.get("next_cursor") is not None:
        lines.append(f"Showing the newest {len(items)}; pass a larger --limit for more.")
    return "\n".join(lines)


def render_sandbox(result: Mapping[str, Any]) -> str:
    shown = result["result"]
    sandbox = shown["sandbox"]
    provenance = sandbox.get("provenance") or {}
    lines = [
        f"Sandbox: {_cell(sandbox['id'])} ({_cell(sandbox['status'])})",
        f"  Image: {_cell(sandbox.get('image'))}",
        f"  Created: {_when(sandbox.get('created_at'))}",
        f"  Ready: {_when(sandbox.get('ready_at'))}",
        f"  Lease ends: {_when(sandbox.get('lease_expires_at'))}",
        f"  Lifetime ends: {_when(sandbox.get('max_expires_at'))}",
    ]
    if sandbox.get("queued_until") is not None:
        lines.append(f"  Queued until: {_when(sandbox['queued_until'])}")
    lines.append(f"  Ended: {_ended(sandbox)}")
    lines += [
        f"  Release: {_cell(provenance.get('release_id'))}"
        + (
            f" (the Agent now selects {_cell(shown['selected_release_id'])})"
            if sandbox.get("older_release")
            else ""
        ),
        f"  Session: {_cell(provenance.get('session_id'))}",
        f"  Task: {_cell(provenance.get('task_id'))}",
    ]
    events = shown["events"]
    if not events:
        lines += ["", "No events."]
        return "\n".join(lines)
    rows = [
        (_when(item["at"]), _cell(item["kind"]), _cell(item["actor"]), _detail(item["detail"]))
        for item in events
    ]
    lines += ["", *_table(("AT", "EVENT", "ACTOR", "DETAIL"), rows)]
    return "\n".join(lines)


def _capacity_line(capacity: Mapping[str, Any] | None) -> str:
    if capacity is None:
        return "Capacity: not reported."
    agent: Mapping[str, Any] = capacity["agent"]
    line = (
        f"Capacity: {_count(agent['in_use'])} of {_count(agent['limit'])} places held under "
        f"the Agent's limit, {_count(agent['queued'])} queued"
    )
    if capacity.get("available") == 0 and capacity.get("limited_by") is not None:
        line += f"; a create now waits or is refused ({_cell(capacity['limited_by'])} limit)"
    return line + "."


def _count(value: object) -> str:
    return str(value) if type(value) is int else "?"


def _lease(sandbox: Mapping[str, Any]) -> str:
    if sandbox.get("status") == "provisioning" and sandbox.get("queued_until") is not None:
        return f"queued to {_when(sandbox['queued_until'])}"
    return _when(sandbox.get("lease_expires_at"))


def _ended(sandbox: Mapping[str, Any]) -> str:
    failure = sandbox.get("failure")
    if isinstance(failure, Mapping):
        return f"failed: {_cell(failure.get('code'))}"
    reason = sandbox.get("destroy_reason")
    if reason is None:
        return "-"
    ended = sandbox.get("destroyed_at")
    return f"{_cell(reason)} {_when(ended)}" if ended is not None else _cell(reason)


def _provenance_cell(sandbox: Mapping[str, Any], name: str) -> str:
    provenance = sandbox.get("provenance")
    value = provenance.get(name) if isinstance(provenance, Mapping) else None
    text = _cell(value, limit=40)
    return f"{text} *" if name == "release_id" and sandbox.get("older_release") else text


def _detail(detail: Mapping[str, Any]) -> str:
    parts = [f"{key}={_scalar(value)}" for key, value in sorted(detail.items())]
    return _cell(" ".join(parts), limit=100) if parts else "-"


def _scalar(value: object) -> str:
    if isinstance(value, Mapping):
        return "{" + ",".join(f"{key}={_scalar(item)}" for key, item in value.items()) + "}"
    if isinstance(value, list):
        return "[" + ",".join(_scalar(item) for item in value) + "]"
    return "-" if value is None else str(value)

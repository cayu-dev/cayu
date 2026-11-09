"""`cayu cloud sandboxes list|show|archives --application NAME`: an Agent's brokered sandboxes
in Cayu Cloud, who created each, what holds its places, and the archives kept for its copies.

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

`list --session/--task/--label/--since` lists only the sandboxes created for that session or
task, with those labels, or since then; Cloud filters and pages them (a Cloud before
cayu-tech/cayu-cloud#766 would ignore the filters, so the command refuses to list
there). Both commands also show what each sandbox did (`activity`: its commands, failed ones
and their run time, and its file transfers in and out), and `list` the same summed over every
sandbox that matches, which Cloud adds up; an older Cloud reports none.

`archives` lists the copy-in archives Cloud keeps for the Agent so its next staged copy in of
the same tree starts without an upload (`GET .../sandboxes/copy-archives`, from
cayu-tech/cayu-cloud#970): each one's digest, size, status, last use and when it lapses. It
pages with `--limit` and `--cursor`, and never shows where an archive is stored.

These commands print a table by default; `--json` prints the JSON envelope the other `cayu
cloud` commands print, with `schema_version` in its result.
"""

from __future__ import annotations

import argparse
import re
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any
from urllib.parse import quote

from cayu.cli._cloud_api import CloudApiClient, CloudApiError
from cayu.cli._cloud_releases import _when, add_output_argument, render_failure
from cayu.cli._cloud_service_logs import log_time
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
# The Cloud change that filters the list and sums the activity.
_FILTERS_CLOUD = "cayu-tech/cayu-cloud#766"
# Cloud's bounds on a filter (`SandboxFilter`): a provenance id, and a label as a create sets it.
_ID_LIMIT = 256
_LABEL_KEY = re.compile(r"[a-z0-9_.-]{1,63}")
_LABEL_VALUE_LIMIT = 200
_LABEL_FILTER_LIMIT = 10
_PROVENANCE = ("release_id", "session_id", "task_id")
# The Cloud change that lists the copy-in archives an Agent keeps.
_ARCHIVES_CLOUD = "cayu-tech/cayu-cloud#970"
_CURSOR_LIMIT = 1024
_ARCHIVE_FIELDS = (
    "id",
    "sha256",
    "bytes",
    "status",
    "created_at",
    "stored_at",
    "last_used_at",
    "retained_until",
)
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


def _filter_id(value: str) -> str:
    if not value or len(value) > _ID_LIMIT or not value.isprintable():
        raise argparse.ArgumentTypeError(f"must be 1 to {_ID_LIMIT} printable characters")
    return value


def _cursor(value: str) -> str:
    if not value or len(value) > _CURSOR_LIMIT or not value.isprintable():
        raise argparse.ArgumentTypeError("is not a cursor from an earlier page")
    return value


def _label(value: str) -> tuple[str, str]:
    key, separator, label = value.partition("=")
    if not separator or not _LABEL_KEY.fullmatch(key):
        raise argparse.ArgumentTypeError("must be KEY=VALUE with a key of 1 to 63 of [a-z0-9_.-]")
    if len(label) > _LABEL_VALUE_LIMIT or not label.isprintable():
        raise argparse.ArgumentTypeError(
            f"its value must be printable and at most {_LABEL_VALUE_LIMIT} long"
        )
    return key, label


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
            "limit now and what each did (commands, file transfers), summed at the end over "
            "every sandbox that matches."
        ),
    )
    listing.add_argument("--status", choices=STATUSES, help="Only sandboxes with this status.")
    listing.add_argument(
        "--limit",
        type=_limit,
        default=DEFAULT_LIMIT,
        help=f"How many to list, newest first (default {DEFAULT_LIMIT}, at most {MAX_LIMIT}).",
    )
    listing.add_argument(
        "--session",
        type=_filter_id,
        help="Only the sandboxes created for this session (as their provenance names it).",
    )
    listing.add_argument(
        "--task", type=_filter_id, help="Only the sandboxes created for this task."
    )
    listing.add_argument(
        "--label",
        type=_label,
        action="append",
        metavar="KEY=VALUE",
        help="Only the sandboxes with this label; repeat for several, all of which must match.",
    )
    listing.add_argument(
        "--since",
        type=log_time,
        help="Only the sandboxes created at or after this time: 15m, 1h, 2d or ISO 8601.",
    )
    showing = actions.add_parser(
        "show",
        description="Show one sandbox, what it did, and its audit events, oldest first.",
    )
    showing.add_argument("sandbox_id", metavar="SANDBOX_ID", type=_sandbox_id)
    archives = actions.add_parser(
        "archives",
        description=(
            "List the copy-in archives Cayu Cloud keeps for the Agent, newest first: a staged "
            "copy in of the same archive into any of its sandboxes starts from one without an "
            "upload. Read only; Cloud lets them lapse on its own."
        ),
    )
    archives.add_argument(
        "--limit",
        type=_limit,
        default=DEFAULT_LIMIT,
        help=f"How many to list, newest first (default {DEFAULT_LIMIT}, at most {MAX_LIMIT}).",
    )
    archives.add_argument(
        "--cursor",
        type=_cursor,
        help="Start after the archives an earlier list showed (its next_cursor).",
    )
    for parser, render in (
        (listing, render_sandboxes),
        (showing, render_sandbox),
        (archives, render_archives),
    ):
        parser.add_argument("--application", required=True)
        add_output_argument(parser, render, failure=render_failure)


def sandboxes_command(
    arguments: argparse.Namespace, *, client: CloudApiClient, application_id: str
) -> dict[str, Any]:
    base = f"/v1/applications/{quote(application_id, safe='')}"
    if arguments.sandboxes_command == "archives":
        return _archives(arguments, client=client, base=base, application_id=application_id)
    selected = _selected_release(client, base)
    if arguments.sandboxes_command == "show":
        path = f"{base}/sandboxes/{quote(arguments.sandbox_id, safe='')}"
        sandbox = _read(client, path, query={"activity": "true"})
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
    filters = _filters(arguments)
    # The sandboxes the list and its totals mean.
    matching: dict[str, str | Sequence[str]] = dict(filters)
    if arguments.status is not None:
        matching["status"] = arguments.status
    items: list[Mapping[str, Any]] = []
    cursor: str | None = None
    while True:
        query: dict[str, str | Sequence[str]] = {
            "limit": str(min(_PAGE_LIMIT, arguments.limit - len(items))),
            "activity": "true",
            **matching,
        }
        if cursor is not None:
            query["cursor"] = cursor
        page = _read(client, f"{base}/sandboxes", query=query)
        # A Cloud that filters says so; an older one ignores the filters and lists every
        # sandbox, which would pass for the session's.
        applied = isinstance(page.get("filters"), Mapping)
        if filters and not applied:
            raise CloudApiError(
                "sandbox_filters_unsupported",
                "This Cayu Cloud doesn't filter sandboxes by session, task, label or time; "
                f"it needs {_FILTERS_CLOUD}. Update the install, or list without "
                "--session, --task, --label and --since.",
            )
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
            "filters": _filters_dict(arguments),
            "capacity": _capacity(client, base),
            "items": [
                sandbox_summary(item, selected_release_id=selected)
                for item in items[: arguments.limit]
            ],
            "next_cursor": cursor,
            # Only a Cloud that applies the filters adds the activity up.
            "totals": _totals(client, base, matching) if applied else None,
        },
    }


def _archives(
    arguments: argparse.Namespace, *, client: CloudApiClient, base: str, application_id: str
) -> dict[str, Any]:
    """The archives the Agent keeps, a page of Cloud's at a time up to ``--limit``."""

    items: list[Mapping[str, Any]] = []
    cursor: str | None = arguments.cursor
    # Every cursor a page started from: a page that answers one of them again would loop.
    seen: set[str] = set() if cursor is None else {cursor}
    reuse: Any = None
    while True:
        query = {"limit": str(min(_PAGE_LIMIT, arguments.limit - len(items)))}
        if cursor is not None:
            query["cursor"] = cursor
        try:
            page = client.request("GET", f"{base}/sandboxes/copy-archives", query=query)
        except CloudApiError as exc:
            # A Cloud before the route reads "copy-archives" as a sandbox id it doesn't have;
            # the Agent itself was just found.
            if exc.status_code == 404:
                raise CloudApiError(
                    "sandbox_archives_unsupported",
                    "This Cayu Cloud doesn't list the copy-in archives it keeps; it needs "
                    f"{_ARCHIVES_CLOUD}.",
                    status_code=404,
                    code=exc.code,
                    request=exc.request,
                ) from None
            raise _sandbox_error(exc) from None
        found = page.get("items")
        if not isinstance(found, list):
            raise CloudApiError("api_response_invalid", "The copy-in archive list is invalid.")
        items.extend(item for item in found if isinstance(item, Mapping))
        reuse = page.get("reuse")
        next_cursor = page.get("next_cursor")
        cursor = next_cursor if isinstance(next_cursor, str) and next_cursor else None
        if cursor is not None and cursor in seen:
            raise CloudApiError(
                "api_response_invalid", "The pages of copy-in archives repeat a cursor."
            )
        if cursor is None or len(items) >= arguments.limit:
            break
        seen.add(cursor)
    return {
        "operation": "sandboxes.archives",
        "result": {
            "schema_version": SCHEMA_VERSION,
            "application_id": application_id,
            "reuse": (
                {
                    name: _number(reuse.get(name))
                    for name in ("seconds", "max_archives", "max_bytes")
                }
                if isinstance(reuse, Mapping)
                else None
            ),
            "items": [
                {name: item.get(name) for name in _ARCHIVE_FIELDS}
                for item in items[: arguments.limit]
            ],
            "next_cursor": cursor,
        },
    }


def _filters(arguments: argparse.Namespace) -> dict[str, str | list[str]]:
    """The list's filters as Cloud's query names them."""

    query: dict[str, str | list[str]] = {}
    if arguments.session is not None:
        query["session"] = arguments.session
    if arguments.task is not None:
        query["task"] = arguments.task
    if arguments.label:
        if len(arguments.label) > _LABEL_FILTER_LIMIT:
            raise CloudApiError(
                "invalid_input", f"--label may be given at most {_LABEL_FILTER_LIMIT} times."
            )
        values: dict[str, str] = {}
        for key, value in arguments.label:
            if values.setdefault(key, value) != value:
                # No sandbox has two values for one key; Cloud refuses it too.
                raise CloudApiError(
                    "invalid_input", f"--label names {key} twice with different values."
                )
        query["label"] = [f"{key}={value}" for key, value in arguments.label]
    if arguments.since is not None:
        query["since"] = arguments.since.isoformat()
    return query


def _filters_dict(arguments: argparse.Namespace) -> dict[str, Any]:
    since: datetime | None = arguments.since
    return {
        "status": arguments.status,
        "session": arguments.session,
        "task": arguments.task,
        "labels": dict(arguments.label or ()),
        "since": None if since is None else since.isoformat(),
    }


def _totals(
    client: CloudApiClient, base: str, matching: Mapping[str, str | Sequence[str]]
) -> dict[str, Any] | None:
    """How many sandboxes match and what they did, summed by Cloud over all of them, not only
    the ones listed; None from a Cloud that doesn't say."""

    try:
        totals = client.request("GET", f"{base}/sandboxes/activity", query=matching)
    except CloudApiError as exc:
        # A Cloud before the route reads "activity" as a sandbox id it doesn't have.
        if exc.status_code == 404:
            return None
        raise _sandbox_error(exc) from None
    activity = _activity(totals.get("activity"))
    count = _number(totals.get("sandboxes"))
    if activity is None or count is None:
        return None
    return {"sandboxes": count, "activity": activity}


def sandbox_summary(
    sandbox: Mapping[str, Any], *, selected_release_id: str | None
) -> dict[str, Any]:
    """The record's fields the commands show, its provenance, and whether it still holds a
    place for a Release the Agent no longer selects (None when that can't be told). A queued
    sandbox holds no place yet, so it is never marked."""

    summary = {name: sandbox.get(name) for name in _SANDBOX_FIELDS}
    summary["activity"] = _activity(sandbox.get("activity"))
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
    client: CloudApiClient, path: str, *, query: Mapping[str, str | Sequence[str]] | None = None
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


def _activity(value: Any) -> dict[str, Any] | None:
    """What a sandbox did as Cloud sums it (``activity``), or None when it doesn't say."""

    if not isinstance(value, Mapping):
        return None
    commands = value.get("commands")
    transfers = value.get("transfers")
    if not isinstance(commands, Mapping) or not isinstance(transfers, Mapping):
        return None
    inward = transfers.get("in")
    outward = transfers.get("out")
    if not isinstance(inward, Mapping) or not isinstance(outward, Mapping):
        return None
    return {
        "commands": {
            name: _number(commands.get(name))
            for name in ("count", "running", "failed", "cancelled", "duration_ms")
        },
        "transfers": {
            "in": {name: _number(inward.get(name)) for name in ("count", "bytes", "files")},
            "out": {name: _number(outward.get(name)) for name in ("count", "bytes")},
        },
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
            _commands_cell(item.get("activity")),
            _transfers_cell(item.get("activity")),
        )
        for item in items
    ]
    lines += _table(
        (
            "SANDBOX",
            "STATUS",
            "CREATED",
            "LEASE ENDS",
            "ENDED",
            "RELEASE",
            "SESSION",
            "TASK",
            "COMMANDS",
            "TRANSFERS IN / OUT",
        ),
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
    lines += ["", _totals_line(listed.get("totals"))]
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
        *_activity_lines(sandbox.get("activity")),
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


def render_archives(result: Mapping[str, Any]) -> str:
    listed = result["result"]
    reuse = listed.get("reuse")
    if reuse is None:
        lines = ["Reuse: this Cayu Cloud keeps no copy-in archives."]
    else:
        lines = [
            f"Reuse: each archive is kept {_duration(reuse['seconds'])} after its last use, "
            f"at most {_count(reuse['max_archives'])} per Agent, each up to "
            f"{_bytes(reuse['max_bytes'])}."
        ]
    items = listed["items"]
    if not items:
        lines.append(f"No copy-in archives kept for {_cell(listed['application_id'])}.")
    else:
        rows = [
            (
                _cell(item["id"], limit=32),
                _cell(item["status"]),
                _bytes(item["bytes"]),
                _cell(item["sha256"])[:12],
                _when(item["created_at"]),
                _when(item["last_used_at"]),
                _when(item["retained_until"]),
            )
            for item in items
        ]
        lines += _table(
            ("ARCHIVE", "STATUS", "SIZE", "SHA256", "KEPT", "LAST USED", "LAPSES"), rows
        )
    if listed.get("next_cursor") is not None:
        cursor = _cell(listed["next_cursor"], limit=_CURSOR_LIMIT)
        lines.append(f"More: pass --cursor {cursor} for the next page.")
    return "\n".join(lines)


def _duration(value: object) -> str:
    if type(value) is not int:
        return "?"
    for unit, seconds in (("h", 3600), ("m", 60)):
        if value >= seconds and value % seconds == 0:
            return f"{value // seconds}{unit}"
    return f"{value}s"


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


def _activity_lines(activity: Mapping[str, Any] | None) -> list[str]:
    if activity is None:
        return [f"  Activity: not reported; this Cayu Cloud needs {_FILTERS_CLOUD}."]
    commands = activity["commands"]
    inward = activity["transfers"]["in"]
    outward = activity["transfers"]["out"]
    return [
        f"  Commands: {_commands_text(commands)}",
        f"  Transfers in: {_transfer_text(inward)}",
        f"  Transfers out: {_transfer_text(outward)}",
    ]


def _totals_line(totals: Mapping[str, Any] | None) -> str:
    if totals is None:
        return f"Activity: not reported; this Cayu Cloud needs {_FILTERS_CLOUD}."
    activity = totals["activity"]
    count = totals["sandboxes"]
    return (
        f"All {_count(count)} matching {'sandbox' if count == 1 else 'sandboxes'}: "
        f"{_commands_text(activity['commands'])}; "
        f"in {_transfer_text(activity['transfers']['in'])}; "
        f"out {_transfer_text(activity['transfers']['out'])}."
    )


def _commands_text(commands: Mapping[str, Any]) -> str:
    count = commands["count"]
    text = f"{_count(count)} {'command' if count == 1 else 'commands'}"
    outcomes = [
        f"{_count(commands[name])} {name}"
        for name in ("failed", "cancelled", "running")
        if commands[name] != 0
    ]
    if outcomes:
        text += f" ({', '.join(outcomes)})"
    return f"{text}, {_milliseconds(commands['duration_ms'])} in total"


def _transfer_text(transfer: Mapping[str, Any]) -> str:
    count = transfer["count"]
    text = (
        f"{_count(count)} {'transfer' if count == 1 else 'transfers'}, {_bytes(transfer['bytes'])}"
    )
    if "files" in transfer:
        text += f", {_count(transfer['files'])} files"
    return text


def _commands_cell(activity: Mapping[str, Any] | None) -> str:
    if activity is None:
        return "-"
    commands = activity["commands"]
    text = f"{_count(commands['count'])} {_milliseconds(commands['duration_ms'])}"
    return f"{text} {_count(commands['failed'])} failed" if commands["failed"] else text


def _transfers_cell(activity: Mapping[str, Any] | None) -> str:
    if activity is None:
        return "-"
    inward = activity["transfers"]["in"]
    outward = activity["transfers"]["out"]
    return (
        f"{_count(inward['count'])} {_bytes(inward['bytes'])} / "
        f"{_count(outward['count'])} {_bytes(outward['bytes'])}"
    )


def _bytes(value: object) -> str:
    if type(value) is not int:
        return "? B"
    size = float(value)
    for unit in ("B", "kB", "MB", "GB"):
        if size < 1000 or unit == "GB":
            return f"{value} B" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1000
    raise AssertionError("unreachable")


def _milliseconds(value: object) -> str:
    if type(value) is not int:
        return "?"
    if value < 1000:
        return f"{value} ms"
    seconds = value / 1000
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, second = divmod(round(seconds), 60)
    if minutes < 60:
        return f"{minutes}m{second:02d}s"
    hours, minute = divmod(minutes, 60)
    return f"{hours}h{minute:02d}m"


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

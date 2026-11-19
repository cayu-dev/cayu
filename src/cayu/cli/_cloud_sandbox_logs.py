"""`cayu cloud sandboxes logs [SANDBOX_ID] --application NAME`: the sandbox broker's own lines.

Cloud's broker logs what it does with an Agent's sandboxes: egress it refused, preflights that
failed, sandboxes that became ready, failed or were destroyed. Cloud serves those lines,
credential-safe, at `GET /v1/applications/{id}/sandboxes/{sandbox_id}/logs` (one sandbox) and
`GET /v1/applications/{id}/sandboxes/logs` (every line that names the Agent), to anyone who may
see the Agent.

With a sandbox, the command prints its lines over its whole life, oldest first; without one,
the Agent's newest lines. `--since`, `--until` and `--event` choose lines, `--limit` says how
many, and `--cursor` goes on after an earlier page. While a page stops short at Cloud's read
budget with more to read, the command follows its cursor, for up to two minutes. `--follow`
prints the page, then polls Cloud every two seconds for the lines after the last one printed,
until the read ends (a destroyed sandbox's lines end 15 minutes after it) or Ctrl-C. `--json`
prints the JSON envelope the other `cayu cloud` commands print; with `--follow`, one per page
that has lines, then one with the cursor to go on from.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections.abc import Callable, Mapping
from typing import Any
from urllib.parse import quote

from cayu.cli._cloud_api import CloudApiClient, CloudApiError
from cayu.cli._cloud_service_logs import _searched, _time, log_time
from cayu.cli._cloud_settings import _cell

SCHEMA_VERSION = 1
DEFAULT_LIMIT = 100
MAX_LIMIT = 1000
POLL_SECONDS = 2.0
SEARCH_SECONDS = 120.0
# The event names Cloud's query takes (`sandbox_logs.EVENT`).
_EVENT = re.compile(r"sandbox_[a-z_]{1,64}")
# Cloud's bound on a log cursor, which may carry its log store's own scan token.
_CURSOR_LIMIT = 2048
# What Cloud sends is printed whole, control characters replaced: Cloud bounds it.
_WHOLE = 1 << 20
_OPERATION = "sandboxes.logs"


def _event(value: str) -> str:
    if not _EVENT.fullmatch(value):
        raise argparse.ArgumentTypeError("is not a log event name, such as sandbox_egress_refused")
    return value


def _cursor(value: str) -> str:
    if not value or len(value) > _CURSOR_LIMIT or not value.isprintable():
        raise argparse.ArgumentTypeError("is not a cursor from an earlier page")
    return value


def add_sandbox_logs_arguments(
    parser: argparse.ArgumentParser,
    *,
    limit: Callable[[str], int],
    sandbox_id: Callable[[str], str],
) -> None:
    parser.add_argument(
        "sandbox_id",
        metavar="SANDBOX_ID",
        nargs="?",
        type=sandbox_id,
        help="One sandbox's lines over its life; the Agent's newest lines when omitted.",
    )
    start = parser.add_mutually_exclusive_group()
    start.add_argument(
        "--since",
        type=log_time,
        help="The first lines after this time, oldest first: 15m, 1h, 2d or an ISO 8601 time.",
    )
    start.add_argument(
        "--cursor",
        type=_cursor,
        help="Go on after an earlier page, from its next_cursor (with the same --until and "
        "--event).",
    )
    parser.add_argument("--until", type=log_time, help="No lines after this time.")
    parser.add_argument(
        "--event",
        type=_event,
        help="Only this event's lines, such as sandbox_egress_refused or sandbox_preflight_failed.",
    )
    parser.add_argument(
        "--limit",
        type=limit,
        default=DEFAULT_LIMIT,
        help=f"Lines per page (default {DEFAULT_LIMIT}, at most {MAX_LIMIT}).",
    )
    parser.add_argument(
        "--follow",
        action="store_true",
        help="Keep printing new lines until the read ends or Ctrl-C.",
    )


def sandbox_logs(
    arguments: argparse.Namespace, *, client: CloudApiClient, base: str, application_id: str
) -> dict[str, Any]:
    sandbox_id = arguments.sandbox_id
    path = (
        f"{base}/sandboxes/logs"
        if sandbox_id is None
        else f"{base}/sandboxes/{quote(sandbox_id, safe='')}/logs"
    )
    # A cursor carries the read's start; `until` and `event` go with every page.
    following: dict[str, str] = {}
    if arguments.until is not None:
        following["until"] = arguments.until.isoformat()
    if arguments.event is not None:
        following["event"] = arguments.event
    query = {**following, "limit": str(arguments.limit)}
    if arguments.since is not None:
        query["since"] = arguments.since.isoformat()
    if arguments.cursor is not None:
        query["cursor"] = arguments.cursor

    def envelope(
        page: Mapping[str, Any],
        lines: list[dict[str, Any]],
        searched: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        return {
            "operation": _OPERATION,
            "result": {
                "schema_version": SCHEMA_VERSION,
                "application_id": application_id,
                "sandbox_id": sandbox_id,
                "lines": lines,
                "next_cursor": page.get("next_cursor"),
                "more": page.get("more"),
                "searched": searched,
                # The resolved bound a cursor goes on with: a relative --until (1h) is a
                # different time when run again.
                "until": following.get("until"),
            },
        }

    if arguments.follow:
        return _follow(
            arguments,
            client=client,
            path=path,
            query=query,
            following=following,
            envelope=envelope,
        )
    page = _page(client, path, query)
    lines = list(page["lines"])
    started = _searched(page)
    deadline = time.monotonic() + SEARCH_SECONDS
    # A page Cloud's read budget cut short goes on from its cursor, up to the limit. Ctrl-C
    # keeps what was read, with the cursor after it.
    try:
        while (
            page["more"]
            and page["next_cursor"] is not None
            and len(lines) < arguments.limit
            and time.monotonic() < deadline
        ):
            got = _page(
                client,
                path,
                {
                    **following,
                    "cursor": page["next_cursor"],
                    "limit": str(arguments.limit - len(lines)),
                },
            )
            seen = {str(line.get("id")) for line in lines}
            started = started if _searched(got) is not None else None
            # Together, so wherever Ctrl-C stops the loop the cursor goes on after the lines.
            lines, page = (
                [*lines, *(line for line in got["lines"] if str(line.get("id")) not in seen)],
                got,
            )
    except KeyboardInterrupt:
        pass
    ended = _searched(page)
    searched = None if started is None or ended is None else {**ended, "from": started["from"]}
    return envelope(page, lines, searched)


def _follow(
    arguments: argparse.Namespace,
    *,
    client: CloudApiClient,
    path: str,
    query: dict[str, str],
    following: dict[str, str],
    envelope: Callable[..., dict[str, Any]],
) -> dict[str, Any]:
    """Print each page as it comes, from the cursor of the one before, until the read ends or
    Ctrl-C. A line Cloud gives again at a page boundary is printed once. What it returns says
    where to go on: after the last page printed, wherever Ctrl-C stopped it."""

    printed: set[str] = set()
    checkpoint: str | None = arguments.cursor
    try:
        page = _page(client, path, query)
        while True:
            lines = [line for line in page["lines"] if str(line.get("id")) not in printed]
            if lines:
                shown = envelope(page, lines, _searched(page))
                print(
                    json.dumps({"ok": True, **shown}, sort_keys=True)
                    if arguments.json
                    else render_lines(lines),
                    flush=True,
                )
            checkpoint = page["next_cursor"]
            if checkpoint is None:
                break
            printed = {str(line.get("id")) for line in page["lines"]}
            if not page["more"]:
                time.sleep(POLL_SECONDS)
            page = _page(client, path, {**following, "cursor": checkpoint, "limit": query["limit"]})
    except KeyboardInterrupt:
        pass
    return envelope({"next_cursor": checkpoint, "more": False}, [])


def _page(client: CloudApiClient, path: str, query: dict[str, str]) -> dict[str, Any]:
    try:
        page = client.request("GET", path, query=query)
    except CloudApiError as exc:
        raise _logs_error(exc) from None
    lines, cursor, more = page.get("lines"), page.get("next_cursor"), page.get("more")
    if (
        not isinstance(lines, list)
        or not all(isinstance(line, dict) for line in lines)
        or not (cursor is None or isinstance(cursor, str))
        or not isinstance(more, bool)
    ):
        raise CloudApiError("api_response_invalid", "The sandbox log page is invalid.")
    return page


def _logs_error(error: CloudApiError) -> CloudApiError:
    """Say what a refused log read means; the routes answer the sandbox contract's own body,
    which the generic client doesn't read."""

    messages = {
        401: (
            "Cayu Cloud didn't accept this login on the sandbox routes. An older Cayu Cloud "
            "lets only the Agent itself read them."
        ),
        403: (
            "This login can't read the Agent's sandboxes: it needs to be one of the Agent's "
            "people, a developer or an admin, or an API key with the read scope."
        ),
        404: (
            "Cayu Cloud has no such sandbox for this Agent, or doesn't serve the sandbox "
            "broker's log yet (that needs a newer Cayu Cloud)."
        ),
        422: (
            "Cayu Cloud refused the log query: check the sandbox id, --since, --until, --event "
            "and --cursor (a cursor goes with the same --until and --event), and that the span "
            "is at most 7 days."
        ),
        503: (
            "Cayu Cloud couldn't read the sandbox broker's log: this install doesn't keep it "
            "where Cloud can read it, its log store failed for now (try again), or it runs no "
            "sandbox broker."
        ),
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


def render_logs(result: Mapping[str, Any]) -> str:
    page = result["result"]
    text = render_lines(page["lines"])
    cursor = page.get("next_cursor")
    if cursor is None:
        return text
    shown = _cell(cursor, limit=_WHOLE)
    until = page.get("until")
    if until:
        shown = f"{shown} --until {until}"
    if page.get("more"):
        note = f"More lines follow; go on with --cursor {shown}."
    elif page["lines"]:
        note = f"Later lines go on from --cursor {shown}, or use --follow."
    else:
        # No lines yet, or where `--follow` stopped.
        note = f"Go on with --cursor {shown}."
    return f"{text}\n{note}" if text else note


def render_lines(lines: list[dict[str, Any]]) -> str:
    return "\n".join(_line(line) for line in lines)


def _line(line: Mapping[str, Any]) -> str:
    fields = line.get("fields")
    pairs = (
        [f"{_cell(key, limit=_WHOLE)}={_value(value)}" for key, value in fields.items()]
        if isinstance(fields, Mapping)
        else []
    )
    return " ".join(
        (
            _time(line.get("timestamp")),
            _cell(line.get("level")),
            _cell(line.get("event"), limit=_WHOLE),
            _cell(line.get("sandbox_id"), limit=_WHOLE),
            *pairs,
        )
    )


def _value(value: object) -> str:
    """A field's value as one word: plain text as it is, anything else as JSON."""

    if isinstance(value, str) and value and not any(c.isspace() or c in '="' for c in value):
        return _cell(value, limit=_WHOLE)
    return _cell(json.dumps(value, ensure_ascii=False), limit=_WHOLE)

"""`cayu cloud compass passes list`: every pass Compass made over an Agent, a page at a time.

`compass show` gives an Agent's recent passes; this lists all of them, newest first, from
Cloud's paged listing (`GET /v1/compass/passes`). `--limit N` lists N of them (20 by
default), `--cursor` goes on after an earlier listing and `--all` lists every one after it.
The answer's `next_cursor` goes on after the last pass listed. It prints a table by
default; `--json` prints the JSON envelope the other `cayu cloud` commands print.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from typing import Any

from cayu.cli._cloud_api import CloudApiClient, CloudApiError
from cayu.cli._cloud_compass_findings import _CURSOR_LIMIT, _positive, render_failure
from cayu.cli._cloud_settings import _cell, _table

#: Passes `list` shows without `--limit`: as many as `compass show` gives.
DEFAULT_LIMIT = 20
#: Passes one request reads, as many as Cloud answers in one.
_PAGE = 100
_PAGED_PASSES = "/v1/compass/passes"


def add_passes_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--limit",
        type=_positive,
        metavar="N",
        help=f"List at most N passes, newest first (default {DEFAULT_LIMIT}).",
    )
    parser.add_argument("--cursor", help="Go on after an earlier listing, from its next_cursor.")
    parser.add_argument(
        "--all",
        action="store_true",
        help="List every pass (after --cursor, if given), following next_cursor.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the JSON envelope the other `cayu cloud` commands print.",
    )
    parser.set_defaults(_cloud_render=render_passes, _cloud_render_failure=render_failure)


def list_passes(
    client: CloudApiClient, application: str, arguments: argparse.Namespace
) -> dict[str, Any]:
    """The Agent's passes after ``--cursor``, newest first: ``--limit`` of them, or every
    one with ``--all``, with the ``next_cursor`` after the last one listed."""

    if arguments.all and arguments.limit is not None:
        raise ValueError("--all lists every pass; leave out --limit.")
    limit = None if arguments.all else (arguments.limit or DEFAULT_LIMIT)
    items: list[dict[str, Any]] = []
    cursor: str | None = arguments.cursor
    # Every cursor a page started from: a page that answers one of them again would loop.
    seen: set[str] = set() if cursor is None else {cursor}
    while True:
        size = _PAGE if limit is None else min(_PAGE, limit - len(items))
        query = {
            "agent": application,
            "limit": str(size),
            **({} if cursor is None else {"cursor": cursor}),
        }
        answer = client.request("GET", _PAGED_PASSES, query=query)
        if "next_cursor" not in answer:
            # A Cloud from before paging answers only the recent passes, with no cursor.
            raise CloudApiError(
                "compass_paging_unavailable",
                "This Cayu Cloud can't page Compass passes yet; `cayu cloud compass show` "
                "lists the recent ones.",
            )
        records = answer.get("passes")
        following = answer.get("next_cursor")
        if not isinstance(records, list) or not (following is None or isinstance(following, str)):
            raise CloudApiError("api_response_invalid", "Compass's page of passes is invalid.")
        items.extend(item for item in records if isinstance(item, dict))
        cursor = following or None
        if cursor is not None and cursor in seen:
            raise CloudApiError(
                "api_response_invalid", "Compass's pages of passes repeat a cursor."
            )
        if cursor is None or (limit is not None and len(items) >= limit):
            break
        seen.add(cursor)
    return {
        "operation": "compass.passes.list",
        "result": {"application_id": application, "items": items, "next_cursor": cursor},
    }


def render_passes(output: Mapping[str, Any]) -> str:
    result = output["result"]
    items = result["items"]
    if not items:
        return f"No passes for {_cell(result['application_id'])}."
    lines = _passes_table(items)
    if result.get("next_cursor") is not None:
        cursor = _cell(result["next_cursor"], limit=_CURSOR_LIMIT)
        lines.extend(["", f"More passes follow: pass --cursor {cursor}, or --all."])
    return "\n".join(lines)


def _passes_table(items: Sequence[Mapping[str, Any]]) -> list[str]:
    rows = [
        (
            _cell(item.get("at")),
            _cell(item.get("agent")),
            _cell(item.get("window_start")),
            _cell(item.get("sessions_read")),
            _cell(item.get("proposed_findings")),
            _count(item.get("waiting")),
            _mission(item),
        )
        for item in items
    ]
    return _table(
        ("AT", "AGENT", "WINDOW FROM", "SESSIONS READ", "PROPOSED", "WAITING", "MISSION"), rows
    )


def _count(value: object) -> str:
    return str(len(value)) if isinstance(value, list) else "-"


def _mission(item: Mapping[str, Any]) -> str:
    """The mission the pass made ready and its status, or why it made none."""

    mission = item.get("mission")
    if isinstance(mission, Mapping):
        return f"{_cell(mission.get('mission_id'))} ({_cell(mission.get('status'))})"
    reason = item.get("no_mission_reason")
    return _cell(reason) if reason else "-"

"""`cayu cloud service logs`: an Agent's output, each line with its time, process and release.

Without options it prints the Agent's newest lines. `--process` (`web`, `worker`, `migrate`,
`storage-audit` or `schedule:NAME`), `--release`, `--since`, `--until` and `--grep` choose lines;
`--limit` says how many. Without `--since` the page holds the newest lines; with it, the first
ones after it, oldest first, and the command says how to go on when more follow. `--follow`
prints the page, then polls Cloud every two seconds for the lines after the last one printed,
until Ctrl-C. `--json` prints the JSON envelope the other `cayu cloud` commands print; with
`--follow`, one per page that has lines, then one with the cursor to go on from.

A Cloud from before line times answers only `{"items": [...]}`, whatever is asked: without
options the command prints those lines, untimed; with any it refuses rather than ignore them.
"""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Callable, Mapping
from datetime import UTC, datetime, timedelta
from typing import Any

from cayu.cli._cloud_api import CloudApiClient, CloudApiError
from cayu.cli._cloud_releases import MAX_LIMIT, add_output_argument, release_limit
from cayu.cli._cloud_settings import _cell

DEFAULT_LIMIT = 100
POLL_SECONDS = 2.0
_DURATION = re.compile(r"(\d+)([smhd])")
_UNITS = {"s": "seconds", "m": "minutes", "h": "hours", "d": "days"}
# What Cloud sends is printed whole, control characters replaced: Cloud bounds it.
_WHOLE = 1 << 20
_FILTERS = ("process", "release", "since", "until", "grep", "limit", "cursor")


def log_time(value: str) -> datetime:
    """A time for `--since`/`--until`: a span back from now (`90s`, `15m`, `1h`, `2d`) or an
    ISO 8601 time; one without an offset is local."""

    duration = _DURATION.fullmatch(value)
    if duration is not None:
        return datetime.now(UTC) - timedelta(**{_UNITS[duration[2]]: int(duration[1])})
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"{value!r} is neither a span like 15m, 1h or 2d nor an ISO 8601 time"
        ) from None
    return parsed.astimezone(UTC)


def add_service_logs_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--process",
        help="web, worker, migrate, storage-audit or schedule:NAME; every process when omitted.",
    )
    parser.add_argument("--release", help="Only this Release's lines (its deployment id).")
    parser.add_argument(
        "--since",
        type=log_time,
        help="The first lines after this time, oldest first: 15m, 1h, 2d or an ISO 8601 time.",
    )
    parser.add_argument("--until", type=log_time, help="No lines after this time.")
    parser.add_argument("--grep", help="Only lines containing this text (plain, case-sensitive).")
    parser.add_argument(
        "--limit",
        type=release_limit,
        help=f"Lines per page, at most {MAX_LIMIT} ({DEFAULT_LIMIT}).",
    )
    parser.add_argument("--cursor", help="Go on after a page, from its next_cursor.")
    parser.add_argument(
        "--follow",
        action="store_true",
        help="Keep printing new lines until Ctrl-C.",
    )
    add_output_argument(parser, render_logs)


def service_logs(
    arguments: argparse.Namespace,
    *,
    client: CloudApiClient,
    application_id: str,
    sleep: Callable[[float], None],
) -> dict[str, Any]:
    path = f"/v1/applications/{application_id}/service/logs"
    asked = {name: value for name in _FILTERS if (value := getattr(arguments, name)) is not None}
    query = {name: _query_value(value) for name, value in asked.items()}
    query.setdefault("limit", str(DEFAULT_LIMIT))
    ignored = [*asked, *(["follow"] if arguments.follow else [])]
    if arguments.follow:
        return _follow(
            arguments, client=client, path=path, query=query, ignored=ignored, sleep=sleep
        )
    page = _page(client, path, query, ignored=ignored)
    if "lines" not in page:
        return {"operation": "service.logs", "result": {"items": _items(page)}}
    return {"operation": "service.logs", "result": page}


def _page(
    client: CloudApiClient, path: str, query: dict[str, str], *, ignored: list[str]
) -> dict[str, Any]:
    """A page of lines. A Cloud from before line times answers only `items`, whatever was
    asked: it is refused when it would ignore an option (``ignored``)."""

    page = client.request("GET", path, query=query)
    if "lines" not in page:
        if ignored:
            raise CloudApiError(
                "service_logs_filters_unsupported",
                "This Cayu Cloud can't choose or follow Agent log lines yet, so it would ignore "
                f"{', '.join(f'--{name}' for name in ignored)}. Run the command without them "
                "for the Agent's newest lines.",
            )
        return page
    _require_page(page)
    return page


def _follow(
    arguments: argparse.Namespace,
    *,
    client: CloudApiClient,
    path: str,
    query: dict[str, str],
    ignored: list[str],
    sleep: Callable[[float], None],
) -> dict[str, Any]:
    """Print each page as it comes, from the cursor of the one before, until Ctrl-C or the
    end of `--until`. A line Cloud gives again at a page boundary is printed once. What it
    returns says where to go on: after the last page whose output was written, wherever Ctrl-C
    stopped it."""

    # The cursor says where to go on; `since` only said where to start.
    following = {name: value for name, value in query.items() if name not in {"since", "cursor"}}
    printed: set[str] = set()
    checkpoint: str | None = arguments.cursor
    try:
        page = _page(client, path, query, ignored=ignored)
        while True:
            lines = [line for line in page["lines"] if line.get("id") not in printed]
            if lines:
                shown = {**page, "lines": lines, "items": [line["message"] for line in lines]}
                envelope = {"ok": True, "operation": "service.logs", "result": shown}
                text = json.dumps(envelope, sort_keys=True) if arguments.json else _lines(lines)
                print(text, flush=True)
            checkpoint = page["next_cursor"]
            if checkpoint is None:
                break
            printed = {str(line.get("id")) for line in page["lines"]}
            if not page["more"]:
                sleep(POLL_SECONDS)
            page = _page(client, path, {**following, "cursor": checkpoint}, ignored=ignored)
    except KeyboardInterrupt:
        pass
    return {
        "operation": "service.logs",
        "result": {"items": [], "lines": [], "next_cursor": checkpoint, "more": False},
    }


def render_logs(result: Mapping[str, Any]) -> str:
    page = result["result"]
    if "lines" not in page:
        lines = [_cell(item, limit=_WHOLE) for item in page["items"]]
        lines.append("This Cayu Cloud doesn't report line times; upgrade it for the rest.")
        return "\n".join(lines)
    text = _lines(page["lines"])
    if page["more"] and page["next_cursor"] is not None:
        cursor = _cell(page["next_cursor"], limit=_WHOLE)
        note = f"More lines follow; go on with --cursor {cursor}."
        text = f"{text}\n{note}" if text else note
    return text


def _lines(lines: list[dict[str, Any]]) -> str:
    return "\n".join(
        " ".join(
            (
                _time(line.get("timestamp")),
                _cell(line.get("process"), limit=_WHOLE),
                _cell(line.get("release"), limit=_WHOLE),
                _cell(line.get("message"), limit=_WHOLE),
            )
        )
        for line in lines
    )


def _time(value: object) -> str:
    """A line's time in UTC to the millisecond, as Cloud sends it."""

    if not isinstance(value, str):
        return _cell(value)
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return _cell(value)
    return parsed.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _query_value(value: object) -> str:
    return value.isoformat() if isinstance(value, datetime) else str(value)


def _items(page: Mapping[str, Any]) -> list[str]:
    items = page.get("items")
    if not isinstance(items, list):
        raise CloudApiError("api_response_invalid", "The Agent's log lines are invalid.")
    return [str(item) for item in items]


def _require_page(page: Mapping[str, Any]) -> None:
    lines, cursor, more = page.get("lines"), page.get("next_cursor"), page.get("more")
    if (
        not isinstance(lines, list)
        or not all(isinstance(line, dict) for line in lines)
        or not (cursor is None or isinstance(cursor, str))
        or not isinstance(more, bool)
    ):
        raise CloudApiError("api_response_invalid", "The Agent's log page is invalid.")

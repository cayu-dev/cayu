"""`cayu cloud service logs`: an Agent's output, each line with its time, process and release.

Without options it prints the Agent's newest lines. `--process` (`web`, `worker`, `migrate`,
`storage-audit` or `schedule:NAME`), `--release`, `--since`, `--until` and `--grep` choose lines;
`--limit` says how many. Without `--since` the page holds the newest lines; with it, the first
ones after it, oldest first, and the command says how to go on when more follow. Cloud answers a
wide search (a `--grep` over a day or more) a page at a time, each covering what it could search
in a few seconds, so while a page comes back short with more to read, the command follows its
cursor, for up to `--search-for` (two minutes), and then says which span it searched. `--follow`
prints the page, then polls Cloud every two seconds for the lines after the last one printed,
until Ctrl-C. `--json` prints the JSON envelope the other `cayu cloud` commands print; with
`--follow`, one per page that has lines, then one with the cursor to go on from.

`--event` and `--field` compare JSON message fields locally. Explicit starts collect the
first matching lines; an implicit start scans the existing seven-day window and retains
its newest matches. A bounded partial scan names that limitation. Follow retains raw scan
progress even through nonmatching pages, or a `next_since` when a frozen snapshot exhausted
its cursor before live polling could establish another one.

A Cloud from before line times answers only `{"items": [...]}`, whatever is asked: without
options the command prints those lines, untimed; with any it refuses rather than ignore them.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

from cayu.cli._cloud_api import CloudApiClient, CloudApiError
from cayu.cli._cloud_releases import MAX_LIMIT, add_output_argument, release_limit
from cayu.cli._cloud_settings import _cell

DEFAULT_LIMIT = 100
POLL_SECONDS = 2.0
DEFAULT_SEARCH_SECONDS = 120.0
_DURATION = re.compile(r"(\d+)([smhd])")
_UNITS = {"s": "seconds", "m": "minutes", "h": "hours", "d": "days"}
# What Cloud sends is printed whole, control characters replaced: Cloud bounds it.
_WHOLE = 1 << 20
_FILTERS = ("process", "release", "since", "until", "grep", "limit", "cursor")
_PARTIAL_NEWEST = "Newest matches in the scanned portion; newer matches may remain."


def _invalid_constant(value: str) -> Any:
    raise ValueError(f"{value} is not a JSON value")


def _json_value(text: str) -> Any:
    try:
        return json.loads(text, parse_float=Decimal, parse_constant=_invalid_constant)
    except ArithmeticError:
        raise ValueError("JSON number is out of range") from None


def log_field(value: str) -> tuple[str, object]:
    """A literal top-level key and a JSON scalar, or an unquoted string."""

    key, separator, text = value.partition("=")
    if not key or not separator:
        raise argparse.ArgumentTypeError("--field needs KEY=VALUE with a nonempty key")
    try:
        expected = _json_value(text)
    except json.JSONDecodeError:
        if text.lstrip().startswith(('"', "[", "{")):
            raise argparse.ArgumentTypeError(
                "--field VALUE must be a JSON scalar or plain text"
            ) from None
        expected = text
    except (ValueError, RecursionError):
        raise argparse.ArgumentTypeError(
            "--field VALUE must be a finite JSON scalar or plain text"
        ) from None
    if isinstance(expected, (dict, list)):
        raise argparse.ArgumentTypeError("--field VALUE must be a JSON scalar or plain text")
    return key, expected


def _matches(line: Mapping[str, Any], fields: list[tuple[str, object]]) -> bool:
    message = line.get("message")
    if not isinstance(message, str) or line.get("truncated") is True:
        return False
    try:
        record = _json_value(message)
    except (ValueError, RecursionError):
        return False
    if not isinstance(record, dict):
        return False
    for key, expected in fields:
        if key not in record:
            return False
        actual = record[key]
        # JSON has one number type; booleans are separate despite Python's True == 1.
        same_type = type(actual) is type(expected) or (
            type(actual) in (int, Decimal) and type(expected) in (int, Decimal)
        )
        if not same_type or actual != expected:
            return False
    return True


@dataclass(frozen=True)
class _SearchResult:
    page: dict[str, Any]
    interrupted: bool = False
    boundary_ids: frozenset[str] = frozenset()


def _envelope(page: dict[str, Any], *, newest: bool = False) -> dict[str, Any]:
    result: dict[str, Any] = {"operation": "service.logs", "result": page}
    if newest and page.get("more"):
        result["warnings"] = [_PARTIAL_NEWEST]
    return result


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


def search_seconds(value: str) -> float:
    """A time for `--search-for`: a span (`90s`, `5m`, `1h`), or 0 for one page."""

    if value == "0":
        return 0.0
    duration = _DURATION.fullmatch(value)
    if duration is None:
        raise argparse.ArgumentTypeError(f"{value!r} is neither 0 nor a span like 90s, 5m or 1h")
    return timedelta(**{_UNITS[duration[2]]: int(duration[1])}).total_seconds()


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
    parser.add_argument("--event", help="Exact top-level event string in a JSON log message.")
    parser.add_argument(
        "--field",
        type=log_field,
        action="append",
        metavar="KEY=VALUE",
        help="Exact top-level JSON field: JSON scalar or plain text; repeat to require all fields.",
    )
    parser.add_argument(
        "--limit",
        type=release_limit,
        help=f"Lines per page, at most {MAX_LIMIT} ({DEFAULT_LIMIT}).",
    )
    parser.add_argument("--cursor", help="Go on after a page, from its next_cursor.")
    parser.add_argument(
        "--search-for",
        type=search_seconds,
        default=DEFAULT_SEARCH_SECONDS,
        metavar="SPAN",
        help=(
            "Search additional pages (including JSON field scans) for up to this long: "
            "90s, 5m, 1h, or 0 for one page (2m)."
        ),
    )
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
    monotonic: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    path = f"/v1/applications/{application_id}/service/logs"
    asked = {name: value for name in _FILTERS if (value := getattr(arguments, name)) is not None}
    query = {name: _query_value(value) for name, value in asked.items()}
    query.setdefault("limit", str(DEFAULT_LIMIT))
    ignored = [*asked, *(["follow"] if arguments.follow else [])]
    fields = list(getattr(arguments, "field", None) or [])
    event = getattr(arguments, "event", None)
    if event is not None:
        fields.append(("event", event))
        ignored.append("event")
    if getattr(arguments, "field", None):
        ignored.append("field")
    newest = bool(fields) and "since" not in query and "cursor" not in query
    original_query = query
    limit = int(query["limit"])
    if newest:
        now = datetime.now(UTC)
        upper = min(now, arguments.until) if arguments.until is not None else now
        query = {
            **query,
            "since": (upper - timedelta(days=7)).isoformat(),
            "until": upper.isoformat(),
            "limit": str(MAX_LIMIT),
        }
    deadline = monotonic() + getattr(arguments, "search_for", DEFAULT_SEARCH_SECONDS)

    def go_on() -> bool:
        return monotonic() < deadline

    if arguments.follow:
        return _follow(
            arguments,
            client=client,
            path=path,
            query=query,
            ignored=ignored,
            sleep=sleep,
            fields=fields,
            newest=newest,
            limit=limit,
            go_on=go_on,
            following=original_query,
        )
    page = _page(client, path, query, ignored=ignored, fields=fields)
    if "lines" not in page:
        return {"operation": "service.logs", "result": {"items": _items(page)}}
    searched = _search(
        client,
        path,
        query,
        page,
        ignored=ignored,
        go_on=go_on,
        fields=fields,
        newest=newest,
        limit=limit,
    )
    return _envelope(searched.page, newest=newest)


def _search(
    client: CloudApiClient,
    path: str,
    query: dict[str, str],
    page: dict[str, Any],
    *,
    ignored: list[str],
    go_on: Callable[[], bool],
    fields: list[tuple[str, object]],
    newest: bool,
    limit: int,
) -> _SearchResult:
    """Collect first matches, or retain newest matches while scanning the whole window.

    Both modes retain the last raw cursor and only positively reported coverage. Ctrl-C
    returns the last complete merge and an interruption signal for the follow owner.
    """

    following = {name: value for name, value in query.items() if name not in {"since", "cursor"}}
    # The last page merged, every line so far and the first span searched: replaced whole once a
    # page is merged, so wherever Ctrl-C stops the search, the cursor goes on after the lines.
    read: tuple[dict[str, Any], list[dict[str, Any]], Mapping[str, Any] | None] = (
        page,
        page["lines"][-limit:] if newest else list(page["lines"]),
        _searched(page),
    )
    interrupted = False
    try:
        while True:
            last, lines, started = read
            if not (
                last["more"]
                and last["next_cursor"] is not None
                and (newest or len(lines) < limit)
                and go_on()
            ):
                break
            ask = {
                **following,
                "cursor": last["next_cursor"],
                "limit": str(MAX_LIMIT if newest else limit - len(lines)),
            }
            got = _page(client, path, ask, ignored=ignored, fields=fields)
            seen = {line.get("id") for line in [*lines, *last["lines"]]}
            fresh = [line for line in got["lines"] if line.get("id") not in seen]
            combined = [*lines, *fresh]
            read = (
                got,
                combined[-limit:] if newest else combined,
                started if _searched(got) is not None else None,
            )
    except KeyboardInterrupt:
        interrupted = True
    last, lines, started = read
    ended = _searched(last)
    searched = None if started is None or ended is None else {**ended, "from": started["from"]}
    return _SearchResult(
        {
            **last,
            "items": [line.get("message") for line in lines],
            "lines": lines,
            "searched": searched,
        },
        interrupted,
        frozenset(str(line.get("id")) for line in last["lines"]),
    )


def _searched(page: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """The span a page says it searched, when it says one."""

    searched = page.get("searched")
    if not isinstance(searched, dict):
        return None
    if not (isinstance(searched.get("from"), str) and isinstance(searched.get("to"), str)):
        return None
    return searched


def _page(
    client: CloudApiClient,
    path: str,
    query: dict[str, str],
    *,
    ignored: list[str],
    fields: list[tuple[str, object]],
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
    if fields:
        lines = [line for line in page["lines"] if _matches(line, fields)]
        return {**page, "lines": lines, "items": [line["message"] for line in lines]}
    return page


def _follow(
    arguments: argparse.Namespace,
    *,
    client: CloudApiClient,
    path: str,
    query: dict[str, str],
    ignored: list[str],
    sleep: Callable[[float], None],
    fields: list[tuple[str, object]],
    newest: bool,
    limit: int,
    go_on: Callable[[], bool],
    following: dict[str, str],
) -> dict[str, Any]:
    """Print each page as it comes, from the cursor of the one before, until Ctrl-C or the
    end of `--until`. A line Cloud gives again at a page boundary is printed once. What it
    returns says where to go on: after the last page whose output was written, wherever Ctrl-C
    stopped it."""

    # The cursor says where to go on; `since` only said where to start.
    following = {
        name: value for name, value in following.items() if name not in {"since", "cursor"}
    }
    printed: set[str] = set()
    checkpoint: str | None = arguments.cursor
    checkpoint_since: str | None = query.get("since") if fields else None
    try:
        page = _page(client, path, query, ignored=ignored, fields=fields)
        interrupted = False
        if newest:
            searched = _search(
                client,
                path,
                query,
                page,
                ignored=ignored,
                go_on=go_on,
                fields=fields,
                newest=True,
                limit=limit,
            )
            page, interrupted = searched.page, searched.interrupted
        while True:
            lines = [line for line in page["lines"] if line.get("id") not in printed]
            if lines or (newest and page["more"]):
                shown = {**page, "lines": lines, "items": [line["message"] for line in lines]}
                envelope = {"ok": True, **_envelope(shown, newest=newest)}
                text = (
                    json.dumps(envelope, sort_keys=True)
                    if arguments.json
                    else (render_logs(envelope) if newest else _lines(lines))
                )
                print(text, flush=True)
            checkpoint = page["next_cursor"]
            checkpoint_since = None
            if interrupted:
                break
            if checkpoint is None:
                upper = datetime.fromisoformat(query["until"]) if newest else None
                if upper is None or (arguments.until is not None and arguments.until <= upper):
                    break
                # A completed frozen snapshot has no cursor. Continue immediately after its
                # inclusive millisecond end, with the user's original follow window.
                after = upper + timedelta(milliseconds=1)
                checkpoint_since = after.isoformat()
                next_query = {**following, "since": checkpoint_since}
                if arguments.until is not None and arguments.until - after > timedelta(days=7):
                    # Only the initial since request is span-bounded. Once it returns a
                    # cursor, subsequent polls can retain the user's full future horizon.
                    next_query["until"] = (after + timedelta(days=7)).isoformat()
            else:
                next_query = {**following, "cursor": checkpoint}
            printed = {str(line.get("id")) for line in page["lines"]}
            if newest:
                # Display truncation must not erase the cursor page's overlap evidence.
                printed.update(searched.boundary_ids)
            newest = False
            if not page["more"]:
                sleep(POLL_SECONDS)
            page = _page(client, path, next_query, ignored=ignored, fields=fields)
    except KeyboardInterrupt:
        pass
    result: dict[str, Any] = {
        "operation": "service.logs",
        "result": {"items": [], "lines": [], "next_cursor": checkpoint, "more": False},
    }
    if checkpoint_since is not None:
        result["result"]["next_since"] = checkpoint_since
    return result


def render_logs(result: Mapping[str, Any]) -> str:
    page = result["result"]
    if "lines" not in page:
        lines = [_cell(item, limit=_WHOLE) for item in page["items"]]
        lines.append("This Cayu Cloud doesn't report line times; upgrade it for the rest.")
        return "\n".join(lines)
    text = _lines(page["lines"])
    for warning in result.get("warnings", []):
        text = f"{text}\n{warning}" if text else warning
    if page.get("next_since") is not None:
        note = f"Go on with --since {_cell(page['next_since'], limit=_WHOLE)}."
        text = f"{text}\n{note}" if text else note
    if page["more"] and page["next_cursor"] is not None:
        cursor = _cell(page["next_cursor"], limit=_WHOLE)
        searched = _searched(page)
        if searched is None:
            note = f"More lines follow; go on with --cursor {cursor}."
        else:
            # A search Cloud or `--search-for` stopped says how far it got.
            note = (
                f"Searched {_time(searched['from'])} to {_time(searched['to'])}; more may "
                f"follow. Go on with --cursor {cursor}."
            )
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

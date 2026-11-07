"""`cayu cloud activity status --application NAME`: where Cloud's activity reader for an Agent
stands.

Cayu Cloud reads each Agent's runs and human touches through the `cayu.cloud-agent` contract
(`GET {api}/cloud/v1/activity?cursor=...`, see "Runs and human touches" in
`docs/cayu-cloud.md`), a page at a time, and keeps the cursor of the last page it accepted.
`read_activity_reader_status` reads Cloud's account of that reader
(`GET /v1/applications/{id}/activity/reader`, for anyone who may see the Agent): the event
position its cursor names and when it last moved on, when a page last arrived, the last
read's outcome, duration and page size, the reads that recently ran out Cloud's deadline,
and the page size the next read asks for.

Nothing here is a credential: Cloud sends no token, digest or URL, and only the event
position of the cursor, never the cursor itself. The reader keeps every value to the shape
it should have (a reason code, a time, a count), so a value of another shape never reaches
the output. The command prints a table by default; `--json` prints the JSON envelope the
other `cayu cloud` commands print, with `schema_version` in its result.
"""

from __future__ import annotations

import argparse
import re
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any, cast
from urllib.parse import quote

from cayu.cli._cloud_api import CloudApiClient, CloudApiError
from cayu.cli._cloud_releases import add_output_argument, render_failure
from cayu.cli._cloud_settings import _cell

SCHEMA_VERSION = 1
# A reason code or an availability, as Cloud names them (`runtime_unavailable`).
_CODE = re.compile(r"[a-z][a-z0-9_]{0,63}")
_IDENTITY = re.compile(r"[A-Za-z0-9_.:-]{1,128}")
_API_PATHS = frozenset({"/api", "/cayu/api"})
_OUTCOMES = frozenset({"read", "timed_out", "failed"})
# Cloud keeps ten; more than this from a Cloud is not believed.
_TIMEOUTS_LIMIT = 100
# Timed-out reads the table names; the JSON result keeps them all.
_SHOWN_TIMEOUTS = 3


def add_activity_parser(commands: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    activity = commands.add_parser(
        "activity",
        help="Read how Cayu Cloud reads an Agent's activity.",
        description=(
            "Read how Cayu Cloud reads an Agent's runs and human touches through the "
            "cayu.cloud-agent contract."
        ),
    )
    actions = activity.add_subparsers(dest="activity_command", required=True)
    status = actions.add_parser(
        "status",
        description=(
            "Show Cloud's activity reader for the Agent: the event position its cursor names "
            "and when it last moved on, when a page last arrived, the last read's outcome, "
            "duration and page size, the reads that recently ran out Cloud's deadline, and "
            "the page size the next read asks for."
        ),
    )
    status.add_argument("--application", required=True, help="The Agent's slug or name.")
    add_output_argument(status, render_status, failure=render_failure)


def activity_command(
    arguments: argparse.Namespace, *, client: CloudApiClient, application_id: str
) -> dict[str, Any]:
    return {
        "operation": "activity.status",
        "result": {
            "schema_version": SCHEMA_VERSION,
            **read_activity_reader_status(client, application_id),
        },
    }


def read_activity_reader_status(client: CloudApiClient, application_id: str) -> dict[str, Any]:
    """Cloud's activity reader for the Agent ``application_id``, as `activity status` shows it.

    ``reader`` is None when Cloud has no reader for the Agent yet. Raises
    ``CloudApiError("activity_status_unsupported")`` from a Cloud without the route.
    """

    path = f"/v1/applications/{quote(application_id, safe='')}/activity/reader"
    try:
        answer = client.request("GET", path)
    except CloudApiError as exc:
        # The Agent was just resolved, so a 404 without one of Cloud's codes, or a 405,
        # means the route itself is missing.
        if exc.status_code == 405 or (exc.status_code == 404 and exc.code is None):
            raise CloudApiError(
                "activity_status_unsupported",
                "This Cayu Cloud doesn't report its activity reader yet.",
                status_code=exc.status_code,
            ) from None
        raise
    # A missing `reader` is as invalid as one of the wrong shape.
    reader = answer.get("reader", False)
    if reader is not None and not isinstance(reader, Mapping):
        raise CloudApiError("api_response_invalid", "Cloud's activity reader status is invalid.")
    return {
        "application_id": application_id,
        "reader": None if reader is None else _reader(cast("Mapping[str, Any]", reader)),
    }


def _reader(value: Mapping[str, Any]) -> dict[str, Any]:
    cursor = _mapping(value.get("cursor"))
    limits = _mapping(value.get("page_limit"))
    timeouts = value.get("recent_timeouts")
    timeouts = timeouts[-_TIMEOUTS_LIMIT:] if isinstance(timeouts, list) else []
    return {
        "enabled": _flag(value.get("enabled")),
        "availability": _code(value.get("availability")),
        "error": _code(value.get("error")),
        "generation": _identity(value.get("generation")),
        "store_id": _identity(value.get("store_id")),
        "api_path": _one_of(value.get("api_path"), _API_PATHS),
        "rotation_pending": _flag(value.get("rotation_pending")),
        "cursor": {
            "position": _count(cursor.get("position")),
            "advanced_at": _instant(cursor.get("advanced_at")),
        },
        "last_synced_at": _instant(value.get("last_synced_at")),
        "catching_up": _flag(value.get("catching_up")),
        "last_read": _read(value.get("last_read")),
        "recent_timeouts": [moment for moment in map(_instant, timeouts) if moment is not None],
        "page_limit": {
            "next": _count(limits.get("next")),
            "max": _count(limits.get("max")),
            "min": _count(limits.get("min")),
        },
        "read_deadline_seconds": _seconds(value.get("read_deadline_seconds")),
        "restarted_at": _instant(value.get("restarted_at")),
        "restart_reason": _code(value.get("restart_reason")),
    }


def _read(value: object) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    value = cast("Mapping[str, Any]", value)
    return {
        "started_at": _instant(value.get("started_at")),
        "duration_ms": _count(value.get("duration_ms")),
        "outcome": _one_of(value.get("outcome"), _OUTCOMES),
        "error": _code(value.get("error")),
        "page_limit": _count(value.get("page_limit")),
        "next_page_limit": _count(value.get("next_page_limit")),
        "runs": _count(value.get("runs")),
        "touches": _count(value.get("touches")),
        "has_more": _flag(value.get("has_more")),
    }


def render_status(output: Mapping[str, Any]) -> str:
    result = output["result"]
    application = _cell(result["application_id"])
    reader = result["reader"]
    if reader is None:
        return f"Cayu Cloud has no activity reader for {application} yet."
    cursor = reader["cursor"]
    limits = reader["page_limit"]
    rows = [
        ("availability", _cell(reader["availability"])),
        ("enabled", _yes(reader["enabled"])),
        ("error", _cell(reader["error"])),
        ("cursor position", _position(cursor["position"])),
        ("cursor advanced", _ago(cursor["advanced_at"])),
        ("last page", _ago(reader["last_synced_at"])),
        ("catching up", _yes(reader["catching_up"])),
        ("last read", _read_text(reader["last_read"])),
        ("recent timeouts", _timeouts_text(reader["recent_timeouts"])),
        ("next page size", _page_size_text(limits)),
        ("read deadline", _deadline_text(reader["read_deadline_seconds"])),
        ("store", _cell(reader["store_id"])),
        ("api path", _cell(reader["api_path"])),
    ]
    if reader["rotation_pending"]:
        rows.append(("rotation", "a new credential is prepared"))
    if reader["restarted_at"] is not None:
        rows.append(
            ("restarted", f"{_ago(reader['restarted_at'])} ({_cell(reader['restart_reason'])})")
        )
    width = max(len(label) for label, _ in rows)
    return "\n".join(
        [f"Activity reader for {application}", *(f"  {k.ljust(width)}  {v}" for k, v in rows)]
    )


def _read_text(read: Mapping[str, Any] | None) -> str:
    if read is None:
        return "-"
    outcome = {"read": "a page arrived", "timed_out": "timed out", "failed": "failed"}.get(
        read["outcome"], "?"
    )
    parts = [outcome]
    if read["error"] is not None and read["outcome"] != "timed_out":
        parts[0] += f" ({read['error']})"
    if read["duration_ms"] is not None:
        parts.append(f"in {read['duration_ms']} ms")
    parts.append(_ago(read["started_at"]))
    if read["page_limit"] is not None:
        parts.append(f"asked for {read['page_limit']} runs")
    if read["runs"] is not None:
        got = f"got {read['runs']} runs and {_cell(read['touches'])} touches"
        parts.append(got + (", more to read" if read["has_more"] else ""))
    return ", ".join(parts)


def _timeouts_text(timeouts: list[str]) -> str:
    if not timeouts:
        return "none"
    shown = ", ".join(_ago(moment) for moment in reversed(timeouts[-_SHOWN_TIMEOUTS:]))
    more = len(timeouts) - _SHOWN_TIMEOUTS
    return f"{len(timeouts)}, latest {shown}" + (f" and {more} more" if more > 0 else "")


def _page_size_text(limits: Mapping[str, Any]) -> str:
    if limits["next"] is None:
        return "-"
    text = f"{limits['next']} runs"
    if limits["min"] is not None and limits["max"] is not None:
        text += f" ({limits['min']} to {limits['max']})"
    return text


def _deadline_text(seconds: float | None) -> str:
    return "-" if seconds is None else f"{seconds:g} s"


def _position(position: int | None) -> str:
    if position is None:
        return "- (no cursor yet, or one this CLI can't read)"
    return "before the first event" if position == 0 else f"after event {position}"


def _ago(value: str | None, *, now: datetime | None = None) -> str:
    if value is None:
        return "-"
    moment = datetime.fromisoformat(value)
    seconds = int(((now or datetime.now(UTC)) - moment).total_seconds())
    stamp = moment.astimezone(UTC).strftime("%Y-%m-%d %H:%M:%SZ")
    if seconds < 0:
        return stamp
    if seconds < 120:
        ago = f"{seconds} s ago"
    elif seconds < 7200:
        ago = f"{seconds // 60} min ago"
    elif seconds < 172800:
        ago = f"{seconds // 3600} h ago"
    else:
        ago = f"{seconds // 86400} days ago"
    return f"{stamp} ({ago})"


def _yes(value: bool | None) -> str:
    return "-" if value is None else ("yes" if value else "no")


def _mapping(value: object) -> Mapping[str, Any]:
    return cast("Mapping[str, Any]", value) if isinstance(value, Mapping) else {}


def _flag(value: object) -> bool | None:
    return value if isinstance(value, bool) else None


def _count(value: object) -> int | None:
    return value if type(value) is int and value >= 0 else None


def _seconds(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float) or not 0 < value < 86400:
        return None
    return float(value)


def _one_of(value: object, choices: frozenset[str]) -> str | None:
    return value if isinstance(value, str) and value in choices else None


def _code(value: object) -> str | None:
    return value if isinstance(value, str) and _CODE.fullmatch(value) else None


def _identity(value: object) -> str | None:
    return value if isinstance(value, str) and _IDENTITY.fullmatch(value) else None


def _instant(value: object) -> str | None:
    """A time as Cloud gives it, in UTC; None for anything that isn't one."""

    if not isinstance(value, str) or len(value) > 64:
        return None
    try:
        moment = datetime.fromisoformat(value)
        if moment.tzinfo is None:
            return None
        # A time at the edge of the range can parse but not move to UTC.
        return moment.astimezone(UTC).isoformat()
    except (ValueError, OverflowError):
        return None

"""What `cayu cloud` says about the tasks a deploy hands from the old worker to the new one.

Cayu Cloud's deploy check lists the tasks the serving release's worker holds (``tasks`` in a
Release's ``session_preflight``) and lets the session choice decide: ``wait`` holds the
deploy until they finish, ``block`` refuses with ``held_tasks``, and ``proceed`` interrupts
the ones ``--acknowledge-session`` names, recording them as ``interrupted_tasks``. A worker
that declares ``[worker] drain_seconds`` keeps running past its replacement to finish what it
holds; the service status lists those drains (``worker_drains``).

Every reader here takes Cloud's JSON as it comes: a field an older Cloud doesn't send reads
as nothing, never as an error.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

from cayu.cli._cloud_api import CloudApiError

# Task IDs named in one line; the JSON result keeps them all.
_NAMED = 5


def _ids(values: Sequence[str]) -> str:
    named = ", ".join(values[:_NAMED])
    more = len(values) - _NAMED
    return named + (f" and {more} more" if more > 0 else "")


def _count(count: int, noun: str) -> str:
    return f"{count} {noun}" + ("" if count == 1 else "s")


def _strings(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str) and item]


def interrupted_tasks_warning(release: Mapping[str, Any]) -> str | None:
    """A warning naming the tasks this Release's publication interrupted, or None."""

    preflight = release.get("session_preflight")
    if not isinstance(preflight, Mapping):
        return None
    interrupted = _strings(preflight.get("interrupted_tasks"))
    if not interrupted:
        return None
    verb = "was" if len(interrupted) == 1 else "were"
    return (
        f"{_count(len(interrupted), 'task')} the previous worker held {verb} interrupted by this "
        f"release, as acknowledged: {_ids(interrupted)}. Cayu fences each one until the Agent "
        "settles it; `cayu inspect --fenced-tasks` lists them."
    )


def task_check_unsupported_notice(
    release: Mapping[str, Any], running_tasks: Mapping[str, Any] | None
) -> str | None:
    """A notice when this Cloud's deploy check doesn't cover tasks yet, and some were running.

    Cloud's check reports ``tasks_read`` once it reads tasks; a check without it predates that.
    """

    preflight = release.get("session_preflight")
    if (
        not isinstance(preflight, Mapping)
        or "tasks_read" in preflight
        or not isinstance(running_tasks, Mapping)
        or running_tasks.get("status") != "read"
        or not running_tasks.get("tasks")
    ):
        return None
    return (
        "This Cayu Cloud doesn't check the tasks the Agent's worker holds before a deploy yet, "
        "so --session-policy didn't cover the tasks listed before the upload; a release that "
        "replaced the worker may have interrupted them."
    )


def _instant(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)


def _left(deadline: datetime, now: datetime) -> str:
    minutes = max(0, int((deadline - now).total_seconds() // 60))
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours} h {minutes} min left"
    return f"{minutes} min left"


def drain_warnings(
    service: Mapping[str, Any],
    *,
    replaced_by: str | None = None,
    now: datetime | None = None,
) -> list[str]:
    """One line per replaced worker still finishing its tasks, newest last.

    ``replaced_by`` keeps only the drains a given Release's publication started.
    """

    drains = service.get("worker_drains")
    if not isinstance(drains, list):
        return []
    current = now or datetime.now(UTC)
    lines = []
    for item in drains:
        if not isinstance(item, Mapping) or not isinstance(item.get("release_id"), str):
            continue
        if replaced_by is not None and item.get("replaced_by") != replaced_by:
            continue
        deadline = _instant(item.get("deadline_at"))
        instances = _strings(item.get("instances"))
        until = (
            "" if deadline is None else f" until {item['deadline_at']} ({_left(deadline, current)})"
        )
        successor = item.get("replaced_by")
        lines.append(
            f"Draining: release {item['release_id']}'s worker "
            f"({_count(len(instances), 'task')} on the platform) keeps running to finish the "
            f"tasks it holds{until}"
            + (
                f", while release {successor}'s worker takes new ones."
                if isinstance(successor, str)
                else "."
            )
        )
    return lines


WORKER_EXITS_UNSUPPORTED = (
    "This Cayu Cloud doesn't report worker exits yet; the Agent itself can read them at "
    "CAYU_WORKER_EXITS_URL."
)


def worker_exits_report(read: Callable[[], Any]) -> dict[str, Any]:
    """The Agent's worker exit record for `service status`, read best effort.

    ``read`` asks Cloud's `GET /v1/applications/<agent>/worker-exits`. The record as Cloud
    returns it (``application_id``, ``looked_at`` and the newest ``exits``) with ``status``
    ``read``; ``unsupported`` from a Cloud without the route; or ``unavailable`` when the read
    failed, which never fails the status.
    """

    try:
        record = read()
    except CloudApiError as exc:
        if exc.status_code == 405 or (exc.status_code == 404 and exc.code is None):
            return {"status": "unsupported", "message": WORKER_EXITS_UNSUPPORTED}
        return {"status": "unavailable", "message": str(exc)}
    except Exception:
        # The service status stands without the record.
        return {
            "status": "unavailable",
            "message": "Cayu Cloud's worker exit record could not be read.",
        }
    if not isinstance(record, Mapping) or not isinstance(record.get("exits"), list):
        return {"status": "unavailable", "message": "Cayu Cloud returned no worker exit record."}
    return {"status": "read", **record}

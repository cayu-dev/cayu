"""What `cayu cloud deploy` says about tasks the Agent's current workers are running.

A deploy that publishes a release replaces the Agent's worker. Cayu Cloud's deploy check
covers the tasks that worker holds with the session choice (wait for them, refuse, or
interrupt the acknowledged ones), and a worker that declares `[worker] drain_seconds` keeps
finishing them beside its replacement (`docs/project-workers.md`, "A deploy during a long
task"). Before uploading, `deploy` reads the Agent's claimed and running tasks through its
own task API, the same way `cayu task list --application` does, and lists them, so the
owner sees them while the choice can still change; a `--no-promote` candidate lists them
for its later promotion. A failed read never stops the deploy; it is reported.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from cayu.cli._cloud_agent_api import CayuApiError, cloud_agent_api
from cayu.cli._task_readers import ApiTaskReader, TaskListRequest
from cayu.tasks.records import TaskStatus

if TYPE_CHECKING:
    import httpx

    from cayu.cli._cloud_api import CloudApiClient

# Tasks listed per status; the warning says when there are more.
TASK_LIMIT = 20
# Claimed first: a task moves from claimed to running, so reading in this order can't
# miss one that changes status between the two reads (it is in both, then deduplicated).
# Reading an asleep Agent wakes its web process, which can take a minute or more.
_WAKE_TIMEOUT_SECONDS = 150.0
_LIVE_STATUSES = (TaskStatus.CLAIMED, TaskStatus.RUNNING)
_NAMED_IN_WARNING = 5


def read_running_tasks(
    client: CloudApiClient,
    application_id: str,
    *,
    timeout_seconds: float = 30.0,
    http: httpx.Client | None = None,
) -> dict[str, Any]:
    """The Agent's claimed and running tasks, or why they couldn't be read.

    ``{"status": "read", "tasks": [...], "complete": bool}``; ``{"status":
    "not_running"}`` when the Agent has no service yet or it isn't running; or
    ``{"status": "unavailable", "reason": ..., "message": ...}``.
    """

    try:
        api = cloud_agent_api(
            client,
            application_id,
            timeout_seconds=timeout_seconds,
            wake_timeout_seconds=_WAKE_TIMEOUT_SECONDS,
            http=http,
        )
    except CayuApiError as exc:
        if exc.category == "not_serving" or exc.status_code == 404:
            return {"status": "not_running"}
        return _unavailable(exc)
    except Exception:
        # The check only informs; nothing about it may stop the deploy.
        return _unavailable(CayuApiError("read_failed", "The Agent's tasks could not be read."))
    reader = ApiTaskReader(api, source={"kind": "cayu_cloud", "application": application_id})
    tasks: dict[str, dict[str, Any]] = {}
    complete = True
    try:
        for status in _LIVE_STATUSES:
            rows = asyncio.run(
                reader.list_tasks(TaskListRequest(status=status, limit=TASK_LIMIT + 1))
            )
            complete = complete and len(rows) <= TASK_LIMIT
            for row in rows[:TASK_LIMIT]:
                # A task read as claimed and then as running keeps its later status.
                tasks[row["id"]] = _summary(row)
    except CayuApiError as exc:
        return _unavailable(exc)
    except Exception:
        return _unavailable(CayuApiError("read_failed", "The Agent's tasks could not be read."))
    finally:
        api.close()
    return {"status": "read", "tasks": list(tasks.values()), "complete": complete}


def running_tasks_warning(
    report: dict[str, Any], *, application_id: str, candidate: bool = False
) -> str | None:
    """The warning for a report with tasks in it, or ``None``.

    ``candidate`` is for `deploy --no-promote`: staging publishes nothing, and the promotion
    that later publishes the candidate is what replaces the workers.
    """

    if report.get("status") != "read" or not report.get("tasks"):
        return None
    tasks: list[dict[str, Any]] = report["tasks"]
    count = f"{len(tasks)}{'' if report.get('complete') else ' or more'}"
    named = ", ".join(
        f"{task['id']} ({task['type']}, {task['status']})" for task in tasks[:_NAMED_IN_WARNING]
    )
    if len(tasks) > _NAMED_IN_WARNING:
        named += ", ..."
    noun = "task is" if count == "1" else "tasks are"
    publishing = (
        "Promoting this candidate later publishes it and replaces those workers. "
        if candidate
        else "A deploy that publishes a new release replaces those workers. "
    )
    return (
        f"{count} {noun} claimed or running on the Agent's current workers: {named}. "
        f"{publishing}Cayu Cloud's deploy check holds the publication until they finish under "
        "--session-policy wait (the default, up to --session-wait-seconds), refuses under "
        "block, and interrupts them only when --acknowledge-session names them; a task "
        "interrupted that way is cancelled or left fenced, so its work so far may be lost. A "
        "worker that declares [worker] drain_seconds keeps finishing them beside the new one "
        "instead. A Cayu Cloud that predates the task check replaces the workers anyway. "
        f"`cayu task list --application {application_id} --status claimed` and "
        "`--status running` show them."
    )


def unavailable_notice(report: dict[str, Any]) -> str | None:
    if report.get("status") != "unavailable":
        return None
    return (
        "Couldn't check whether the Agent's workers are running tasks this deploy would "
        f"interrupt: {report['message']}"
    )


def _summary(row: dict[str, Any]) -> dict[str, Any]:
    series = row.get("retry_series")
    return {
        "id": row["id"],
        "type": row["type"],
        "status": row["status"],
        "status_reason": row.get("status_reason"),
        "assigned_agent_name": row.get("assigned_agent_name"),
        "worker_id": row.get("worker_id"),
        "updated_at": row.get("updated_at"),
        "retry_attempt": series.get("attempt") if isinstance(series, dict) else None,
    }


def _unavailable(exc: CayuApiError) -> dict[str, Any]:
    return {"status": "unavailable", "reason": exc.category, "message": str(exc)}

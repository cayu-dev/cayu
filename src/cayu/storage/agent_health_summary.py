"""Sum up an Agent's health from any `AgentReaders`, local or on Cayu Cloud.

`agent_health` reads, for each task type, when a task last completed and last failed and how
many failed in a row since the last completion, what waits in the inbox, and how the Agent's
service runs. It takes any `AgentReaders` (`open_cloud_agent_readers` or
`open_store_agent_readers`), so an Agent that reports on other Agents and `cayu cloud service
health` read the same summary:

```python
from datetime import UTC, datetime, timedelta

from cayu.storage import agent_health, open_cloud_agent_readers

async with open_cloud_agent_readers("acme-support") as agent:
    health = await agent_health(agent, since=datetime.now(UTC) - timedelta(hours=24))
print(health.to_dict())
```

It never raises because one part couldn't be read: that part is left empty and named in
``unavailable`` with its error's category. A part the source doesn't have (a local store
has no service) is empty without being named.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, TypeVar

from cayu.cli._task_readers import MAX_TASK_PAGE, TaskListRequest
from cayu.tasks.records import TaskStatus

if TYPE_CHECKING:
    from cayu.storage.agent_readers import AgentReaders

__all__ = ["AgentHealth", "TaskTypeHealth", "agent_health"]

_T = TypeVar("_T")
#: The most tasks read to find the task types run since ``since``.
MAX_DISCOVERED_TASKS = 5 * MAX_TASK_PAGE
#: The most failed tasks in a row counted; a longer streak is reported as this many.
MAX_FAILURE_STREAK = 50
_REASON_CHARS = 200
_MESSAGE_CHARS = 300
_SPACE = re.compile(r"\s+")


@dataclass(frozen=True)
class TaskTypeHealth:
    """One task type: when a task of it last completed and last failed, how many failed in
    a row since the last completion (at most ``MAX_FAILURE_STREAK``), and the last failure's
    reason, shortened and with anything that looks like a credential masked."""

    type: str
    last_completed_at: datetime | None
    last_failed_at: datetime | None
    failure_streak: int
    last_failure_reason: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "last_completed_at": _iso(self.last_completed_at),
            "last_failed_at": _iso(self.last_failed_at),
            "failure_streak": self.failure_streak,
            "last_failure_reason": self.last_failure_reason,
        }


@dataclass(frozen=True)
class AgentHealth:
    """What `agent_health` read. ``tasks`` maps each task type to its `TaskTypeHealth`;
    ``inbox`` is the waiting items by kind; ``service`` is Cayu Cloud's service record
    (``AgentReaders.service``). ``unavailable`` names each part that couldn't be read:
    ``{"part": ..., "category": ..., "message": ...}``."""

    since: datetime
    source: Mapping[str, Any]
    tasks: Mapping[str, TaskTypeHealth]
    inbox: Mapping[str, int] | None
    oldest_inbox_item_at: datetime | None
    service: Mapping[str, Any] | None
    unavailable: tuple[Mapping[str, str], ...] = field(default=())

    def to_dict(self) -> dict[str, Any]:
        """The summary as JSON-ready values, times as ISO 8601 text."""

        return {
            "since": _iso(self.since),
            "source": dict(self.source),
            "tasks": {name: task.to_dict() for name, task in self.tasks.items()},
            "inbox": None if self.inbox is None else dict(self.inbox),
            "oldest_inbox_item_at": _iso(self.oldest_inbox_item_at),
            "service": None if self.service is None else dict(self.service),
            "unavailable": [dict(part) for part in self.unavailable],
        }


async def agent_health(
    readers: AgentReaders,
    *,
    since: datetime,
    task_types: Sequence[str] | None = None,
) -> AgentHealth:
    """Read the health of the Agent ``readers`` reads.

    Without ``task_types``, the task types reported are those of tasks updated since
    ``since`` (a timezone-aware time), read from at most ``MAX_DISCOVERED_TASKS`` of the newest
    tasks. Each type's last completion and failure are its latest, however long ago. Only
    reads are made.
    """

    if since.tzinfo is None:
        raise ValueError("since must be a timezone-aware datetime.")
    unavailable: list[dict[str, str]] = []

    async def part(name: str, read: Callable[[], Awaitable[_T]]) -> _T | None:
        try:
            return await read()
        except Exception as exc:  # One part failing never stops the others.
            unavailable.append(_unavailable(name, exc))
            return None

    if task_types is None:
        discovered = await part("tasks", lambda: _task_types_since(readers, since, unavailable))
        names: list[str] = discovered or []
    else:
        names = list(dict.fromkeys(task_types))
    tasks: dict[str, TaskTypeHealth] = {}
    for name in names:
        health = await part(f"tasks.{name}", lambda name=name: _task_type_health(readers, name))
        if health is not None:
            tasks[name] = health
    inbox = await part("inbox", readers.inbox_counts)
    oldest = None
    if inbox is not None and sum(inbox.values()) > 0:
        oldest = await part("oldest_inbox_item_at", readers.oldest_inbox_item_at)
    service = await part("service", readers.service)
    return AgentHealth(
        since=since,
        source=dict(readers.source),
        tasks=tasks,
        inbox=inbox,
        oldest_inbox_item_at=oldest,
        service=service,
        unavailable=tuple(unavailable),
    )


async def _task_types_since(
    readers: AgentReaders, since: datetime, unavailable: list[dict[str, str]]
) -> list[str]:
    names: dict[str, None] = {}
    offset = 0
    while offset < MAX_DISCOVERED_TASKS:
        rows = await readers.tasks.list_tasks(TaskListRequest(limit=MAX_TASK_PAGE, offset=offset))
        recent = [row for row in rows if (_time(row.get("updated_at")) or since) >= since]
        names.update(dict.fromkeys(str(row["type"]) for row in recent))
        if len(rows) < MAX_TASK_PAGE or len(recent) < len(rows):
            return sorted(names)
        offset += MAX_TASK_PAGE
    unavailable.append(
        {
            "part": "tasks",
            "category": "incomplete",
            "message": (
                f"Only the newest {MAX_DISCOVERED_TASKS} tasks were read for their types; "
                "older types since then may be missing."
            ),
        }
    )
    return sorted(names)


async def _task_type_health(readers: AgentReaders, task_type: str) -> TaskTypeHealth:
    completed = await readers.tasks.list_tasks(
        TaskListRequest(type=task_type, status=TaskStatus.COMPLETED, limit=1)
    )
    failed = await readers.tasks.list_tasks(
        TaskListRequest(type=task_type, status=TaskStatus.FAILED, limit=MAX_FAILURE_STREAK)
    )
    last_completed = _finished_at(completed[0]) if completed else None
    failure_times = [_finished_at(row) for row in failed]
    streak = sum(
        1
        for moment in failure_times
        if last_completed is None or (moment is not None and moment > last_completed)
    )
    return TaskTypeHealth(
        type=task_type,
        last_completed_at=last_completed,
        last_failed_at=failure_times[0] if failure_times else None,
        failure_streak=streak,
        last_failure_reason=_failure_reason(failed[0]) if failed else None,
    )


def _finished_at(row: Mapping[str, Any]) -> datetime | None:
    return _time(row.get("completed_at")) or _time(row.get("updated_at"))


def _failure_reason(row: Mapping[str, Any]) -> str | None:
    for key in ("error_summary", "status_reason", "error_code"):
        reason = _short(row.get(key), _REASON_CHARS)
        if reason:
            return reason
    return None


def _unavailable(part: str, exc: Exception) -> dict[str, str]:
    category = getattr(exc, "category", None)
    return {
        "part": part,
        "category": category if isinstance(category, str) else "read_failed",
        "message": _short(str(exc), _MESSAGE_CHARS) or type(exc).__name__,
    }


def _short(value: object, limit: int) -> str | None:
    """Text on one line, credentials masked, at most ``limit`` characters."""

    from cayu.sessions._call_inspection import redact_inspection_value

    if not isinstance(value, str):
        return None
    text = "".join(
        " " if unicodedata.category(character) in {"Cc", "Cf"} else character for character in value
    )
    text = _SPACE.sub(" ", str(redact_inspection_value(text))).strip()
    if not text:
        return None
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _time(value: object) -> datetime | None:
    if isinstance(value, datetime):
        moment = value
    elif isinstance(value, str):
        try:
            moment = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    else:
        return None
    return moment if moment.tzinfo is not None else moment.replace(tzinfo=UTC)


def _iso(value: datetime | None) -> str | None:
    return None if value is None else value.isoformat()

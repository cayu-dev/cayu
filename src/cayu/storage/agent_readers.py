"""Read an Agent's tasks and sessions the same way, from its local store or from Cayu Cloud.

A project's own commands (``python -m herald last``) read the Agent's durable state. Run
locally, that state is in the project's store; for a deployed Agent it is on Cayu Cloud.
`open_store_agent_readers` and `open_cloud_agent_readers` return the same `AgentReaders`
for either, so one command serves both without routes or a client of its own:

```python
from cayu.storage import TaskListRequest, open_cloud_agent_readers
from cayu.tasks.records import TaskStatus

async with open_cloud_agent_readers("cayu-herald") as agent:
    rows = await agent.tasks.list_tasks(
        TaskListRequest(type="digest", status=TaskStatus.COMPLETED, limit=1)
    )
    digest = await agent.tasks.load_task(rows[0]["id"]) if rows else None
```

For a deployed Agent the readers sign in to Cayu Cloud as `cayu cloud` does (`cayu cloud
login`, ``context`` for its ``--context``, or ``CAYU_CLOUD_API_KEY``) and read the Agent's
own Cayu API with its Developer dashboard login, as `cayu task --application` and `cayu
session --application` do. Only reads are offered; nothing is written.

Tasks come back in the shape of the Cayu server's task API (``GET /api/tasks`` for a list
row, ``GET /api/tasks/{id}`` for one task with its ``result``), which is also what `cayu
task show --json` prints under ``task``. Sessions are read with the `cayu session`
readers: ``query_events(EventQuery)`` and ``query_transcript(TranscriptQuery)`` among
them. A read that can't be answered raises `AgentReadError` with a stable ``category``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, Any

from cayu.cli._cloud_agent_api import DEFAULT_WAKE_TIMEOUT_SECONDS
from cayu.cli._cloud_agent_api import CayuApiError as AgentReadError
from cayu.cli._task_readers import TaskListRequest

if TYPE_CHECKING:
    from cayu.cli._session_reader import SessionReader
    from cayu.cli._task_readers import TaskReader
    from cayu.storage.targets import SessionStoreTarget

__all__ = [
    "AgentReadError",
    "AgentReaders",
    "TaskListRequest",
    "open_cloud_agent_readers",
    "open_store_agent_readers",
]


@dataclass
class AgentReaders:
    """Read-only task and session readers for one Agent; ``close`` (or ``async with``)
    releases both.

    ``tasks.list_tasks(TaskListRequest(...))`` lists newest-updated first, without results;
    ``tasks.load_task(task_id)`` returns one task with its result, or ``None``. ``source``
    says where they read: ``{"kind": "cayu_cloud", "application": ...}``, ``{"kind":
    "sqlite", "path": ...}`` or ``{"kind": "postgres"}``.
    """

    tasks: TaskReader
    sessions: SessionReader
    source: dict[str, Any] = field(default_factory=dict)

    async def close(self) -> None:
        try:
            await self.tasks.close()
        finally:
            await self.sessions.close()

    async def __aenter__(self) -> AgentReaders:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.close()


def open_cloud_agent_readers(
    application: str,
    *,
    context: str | Path | None = None,
    timeout_seconds: float = 30.0,
    wake_timeout_seconds: float = DEFAULT_WAKE_TIMEOUT_SECONDS,
) -> AgentReaders:
    """Readers for the Cayu Cloud Agent ``application``: its id (the value of
    ``CAYU_CLOUD_APPLICATION_ID``), slug or name, as ``--application`` takes it.

    Signs in as `cayu cloud` does, with ``context`` standing for its ``--context``. A
    sleeping Agent is woken and waited for up to ``wake_timeout_seconds``. Raises
    `AgentReadError` when Cloud refuses, the Agent isn't serving, or it has no web process
    (no Cayu address to read).
    """

    from cayu.cli._agent_session_reader import AgentSessionReader
    from cayu.cli._cloud_agent_api import open_cloud_agent_api
    from cayu.cli._task_readers import ApiTaskReader

    api = open_cloud_agent_api(
        application,
        context=None if context is None else Path(context),
        timeout_seconds=timeout_seconds,
        wake_timeout_seconds=wake_timeout_seconds,
    )
    source = {"kind": "cayu_cloud", "application": application}
    # One API client for both; closing it twice is harmless.
    return AgentReaders(
        tasks=ApiTaskReader(api, source=source), sessions=AgentSessionReader(api), source=source
    )


def open_store_agent_readers(target: SessionStoreTarget) -> AgentReaders:
    """Readers for the local store ``target`` (``cayu.storage.targets``), opened read-only,
    as `cayu task --sqlite/--postgres` opens it."""

    from cayu.cli._session_reader import StoreSessionReader
    from cayu.cli._task_readers import open_store_task_reader
    from cayu.cli.session import _open_read_only_store

    tasks = open_store_task_reader(target)
    sessions = StoreSessionReader(_open_read_only_store(target))
    return AgentReaders(tasks=tasks, sessions=sessions, source=tasks.source)

"""Read an Agent's tasks and sessions the same way, from its local store or from Cayu Cloud.

A project's own commands (``python -m herald last``) read the Agent's durable state. Run
locally, that state is in the project's store; for a deployed Agent it is on Cayu Cloud.
`open_store_agent_readers` and `open_cloud_agent_readers` return the same `AgentReaders`
for either, so one command serves both without routes or a client of its own:

```python
from cayu.storage import TaskListRequest, open_cloud_agent_readers
from cayu.tasks.records import TaskStatus

async with open_cloud_agent_readers("acme-herald") as agent:
    rows = await agent.tasks.list_tasks(
        TaskListRequest(type="digest", status=TaskStatus.COMPLETED, limit=1)
    )
    digest = await agent.tasks.load_task(rows[0]["id"]) if rows else None
```

For a deployed Agent the readers sign in to Cayu Cloud as `cayu cloud` does (`cayu cloud
login`, ``context`` for its ``--context``, or ``CAYU_CLOUD_API_KEY``). Tasks are read
through Cloud's ``agent-read`` route, which forwards the read to the Agent on Cloud's side,
so an Organization API key with the ``read`` scope is enough. A Cloud without that route,
and every session read, read the Agent's own Cayu API with its Developer dashboard login,
as `cayu task --application` and `cayu session --application` do; that login is fetched
only when such a read is made. Only reads are offered; nothing is written.

``inbox_counts()``, ``oldest_inbox_item_at()`` and ``service()`` say what waits on a person
and how the Agent's service runs, and `cayu.storage.agent_health` sums all of it up for any
`AgentReaders`.

Tasks come back in the shape of the Cayu server's task API (``GET /api/tasks`` for a list
row, ``GET /api/tasks/{id}`` for one task with its ``result``), which is also what `cayu
task show --json` prints under ``task``. Sessions are read with the `cayu session`
readers: ``query_events(EventQuery)`` and ``query_transcript(TranscriptQuery)`` among
them. A read that can't be answered raises `AgentReadError` with a stable ``category``.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, Any, Protocol, TypeVar
from urllib.parse import quote

from cayu.cli._cloud_agent_api import DEFAULT_WAKE_TIMEOUT_SECONDS
from cayu.cli._cloud_agent_api import CayuApiError as AgentReadError
from cayu.cli._task_readers import TaskLatestRequest, TaskListRequest

if TYPE_CHECKING:
    from cayu.cli._cloud_api import CloudApiClient
    from cayu.cli._session_reader import SessionReader
    from cayu.cli._task_readers import ApiTaskReader, TaskReader
    from cayu.sessions.store import SessionStore
    from cayu.storage.targets import SessionStoreTarget

__all__ = [
    "AgentReadError",
    "AgentReaders",
    "AgentStatusReader",
    "TaskListRequest",
    "open_cloud_agent_readers",
    "open_store_agent_readers",
]

_T = TypeVar("_T")
# Inbox items one page of Cloud's inbox listing reads; a larger inbox has no cheap oldest.
_OLDEST_PAGE = 50
# The service record fields `service()` keeps; the rest (addresses, issues) stay in Cloud.
_SERVICE_FIELDS = ("status", "deployment_id", "cayu_version", "paused", "last_request_at")


class AgentStatusReader(Protocol):
    """What waits on a person, and how the Agent's service runs. Each answers ``None``
    when its source has no such thing, and raises `AgentReadError` when it can't be read."""

    async def inbox_counts(self) -> dict[str, int] | None: ...

    async def oldest_inbox_item_at(self) -> datetime | None: ...

    async def service(self) -> dict[str, Any] | None: ...


@dataclass
class AgentReaders:
    """Read-only task and session readers for one Agent; ``close`` (or ``async with``)
    releases both.

    ``tasks.list_tasks(TaskListRequest(...))`` lists newest-updated first, without results;
    ``tasks.load_task(task_id)`` returns one task with its result, or ``None``. ``source``
    says where they read: ``{"kind": "cayu_cloud", "application": ...}``, ``{"kind":
    "sqlite", "path": ...}`` or ``{"kind": "postgres"}``. ``status``, when given, answers
    ``inbox_counts``, ``oldest_inbox_item_at`` and ``service``; without it they answer
    ``None``.
    """

    tasks: TaskReader
    sessions: SessionReader
    source: dict[str, Any] = field(default_factory=dict)
    status: AgentStatusReader | None = None

    async def inbox_counts(self) -> dict[str, int] | None:
        """How many items of each inbox kind wait on a person (``approval``, ``question``,
        ``recovery`` and, where the source counts it, ``knowledge``), or ``None`` when the
        source doesn't count them. Raises `AgentReadError` when they can't be read."""

        return None if self.status is None else await self.status.inbox_counts()

    async def oldest_inbox_item_at(self) -> datetime | None:
        """When the oldest waiting inbox item was asked, when the source can say so cheaply
        (Cayu Cloud reads it when one page holds the whole inbox); otherwise ``None``."""

        return None if self.status is None else await self.status.oldest_inbox_item_at()

    async def service(self) -> dict[str, Any] | None:
        """Cayu Cloud's record of the Agent's service: ``status``, ``deployment_id``,
        ``paused``, ``last_request_at`` and ``cayu_version`` when Cloud reports it. ``None``
        for a local store."""

        return None if self.status is None else await self.status.service()

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

    Signs in as `cayu cloud` does, with ``context`` standing for its ``--context``, and
    finds the Agent; reading it waits for the first read. Tasks are read through Cloud's
    ``agent-read`` route when Cloud offers it, so a ``read``-scoped Organization API key is
    enough; otherwise, and for sessions, with the Agent's Developer dashboard login. A
    sleeping Agent is woken and waited for up to ``wake_timeout_seconds``. Raises
    `AgentReadError` when Cloud refuses or doesn't know the Agent; a read raises it when the
    Agent isn't serving or has no web process (no Cayu address to read).
    """

    from cayu.cli._cloud_api import CloudApiError
    from cayu.cli._cloud_auth import CloudAuthError
    from cayu.cli.cloud import CloudCommandError, cloud_client, resolve_application_id

    try:
        client = cloud_client(
            context=None if context is None else Path(context), timeout_seconds=timeout_seconds
        )
        application_id = resolve_application_id(client, application)
    except CloudApiError as exc:
        raise AgentReadError(exc.category, str(exc), status_code=exc.status_code) from None
    except (CloudAuthError, CloudCommandError) as exc:
        raise AgentReadError(exc.category, str(exc)) from None
    return cloud_agent_readers(
        client,
        application_id,
        source={"kind": "cayu_cloud", "application": application},
        timeout_seconds=timeout_seconds,
        wake_timeout_seconds=wake_timeout_seconds,
    )


def cloud_agent_readers(
    client: CloudApiClient,
    application_id: str,
    *,
    source: dict[str, Any],
    timeout_seconds: float = 30.0,
    wake_timeout_seconds: float = DEFAULT_WAKE_TIMEOUT_SECONDS,
) -> AgentReaders:
    """`open_cloud_agent_readers` for a signed-in ``client`` and a resolved Agent id."""

    from cayu.cli._agent_session_reader import AgentSessionReader
    from cayu.cli._cloud_agent_api import DeferredCayuApi, cloud_agent_api, cloud_agent_read_api
    from cayu.cli._task_readers import ApiTaskReader

    options: dict[str, Any] = {
        "timeout_seconds": timeout_seconds,
        "wake_timeout_seconds": wake_timeout_seconds,
    }
    # The operator login is fetched on the first read that needs it, and only then.
    login = DeferredCayuApi(
        lambda: cloud_agent_api(client, application_id, **options),
        description=f"The Agent {application_id}",
    )
    through_cloud = ApiTaskReader(
        cloud_agent_read_api(client, application_id, **options), source=source
    )
    return AgentReaders(
        tasks=_CloudTaskReader(through_cloud, ApiTaskReader(login, source=source), source=source),
        sessions=AgentSessionReader(login),
        source=source,
        status=_CloudAgentStatus(client, application_id),
    )


class _CloudTaskReader:
    """Reads tasks through Cloud's ``agent-read`` route, and with the operator login when
    Cloud doesn't offer it; what the route doesn't forward is read with the login."""

    def __init__(
        self, through_cloud: ApiTaskReader, login: ApiTaskReader, *, source: dict[str, Any]
    ):
        self._through_cloud: ApiTaskReader | None = through_cloud
        self._login = login
        self.source = source

    async def _read(self, read: Callable[[ApiTaskReader], Awaitable[_T]]) -> _T:
        reader = self._through_cloud
        if reader is not None:
            try:
                return await read(reader)
            except AgentReadError as exc:
                # Cloud's plain route-not-found 404 before any answer: a Cloud without it.
                # With a code (`agent_read_not_served`) Cloud has the route and the Agent
                # can't answer it, so the operator login would fare no better.
                if exc.category != "not_served" or exc.code is not None:
                    raise
            self._through_cloud = None
            await reader.close()
        return await read(self._login)

    async def list_tasks(self, request: TaskListRequest) -> list[dict[str, Any]]:
        if request.values or request.labels:
            # Filtering by input, metadata or labels first reads the server's contract,
            # which the route doesn't forward.
            return await self._login.list_tasks(request)
        return await self._read(lambda reader: reader.list_tasks(request))

    async def load_task(self, task_id: str) -> dict[str, Any] | None:
        return await self._read(lambda reader: reader.load_task(task_id))

    async def task_progress(self, task_id: str) -> dict[str, Any] | None:
        # The agent-read route doesn't forward the progress route.
        return await self._login.task_progress(task_id)

    async def latest_tasks(self, request: TaskLatestRequest) -> list[dict[str, Any]]:
        return await self._read(lambda reader: reader.latest_tasks(request))

    async def check_task_counts(self) -> dict[str, Any]:
        return await self._login.check_task_counts()

    async def fenced_tasks(self, *, settle: bool) -> dict[str, Any]:
        # Cloud's read route forwards reads only; settling is the Agent's own action.
        return await self._login.fenced_tasks(settle=settle)

    async def task_session_ids(self, task: dict[str, Any]) -> tuple[list[str], bool] | None:
        return await self._login.task_session_ids(task)

    async def session_timing(self, session_id: str) -> dict[str, Any] | None:
        return await self._login.session_timing(session_id)

    async def event_loop_lag(self, session_ids: Sequence[str]) -> dict[str, Any] | None:
        return await self._login.event_loop_lag(session_ids)

    async def close(self) -> None:
        try:
            if self._through_cloud is not None:
                await self._through_cloud.close()
        finally:
            await self._login.close()


class _CloudAgentStatus:
    """A Cloud Agent's inbox counts and service record, read from Cayu Cloud's own routes
    with the caller's login or ``read``-scoped key."""

    def __init__(self, client: CloudApiClient, application_id: str) -> None:
        self._client = client
        self._path = f"/v1/applications/{quote(application_id, safe='')}"
        self._counts: dict[str, int] | None = None

    def _get(self, path: str, query: dict[str, str] | None = None) -> dict[str, Any]:
        from cayu.cli._cloud_api import CloudApiError

        try:
            return self._client.request("GET", self._path + path, query=query)
        except CloudApiError as exc:
            raise AgentReadError(
                exc.category, str(exc), status_code=exc.status_code, code=exc.code
            ) from None

    async def inbox_counts(self) -> dict[str, int] | None:
        from cayu.cli._inbox_client import InboxCounts

        counted = InboxCounts.read(self._get("/inbox/counts"))
        if counted.counts is None:
            reason = counted.unavailable_reason or "unavailable"
            raise AgentReadError(reason, f"The Agent's inbox couldn't be counted ({reason}).")
        self._counts = dict(counted.counts)
        return dict(counted.counts)

    async def oldest_inbox_item_at(self) -> datetime | None:
        counts = self._counts if self._counts is not None else await self.inbox_counts()
        total = sum((counts or {}).values())
        if not 0 < total <= _OLDEST_PAGE:
            return None
        page = self._get("/inbox", {"limit": str(_OLDEST_PAGE)})
        items = page.get("items")
        agents = page.get("agents")
        if (
            page.get("next_cursor")
            or page.get("complete") is not True
            or not isinstance(items, list)
            or not isinstance(agents, list)
            or not all(
                isinstance(row, dict)
                and row.get("state") == "read"
                and not row.get("issues")
                # An older Cloud names items asked before the serving Release only by id.
                and not row.get("earlier_release_items")
                for row in agents
            )
        ):
            # Not the whole inbox in one page, so its oldest item isn't known.
            return None
        asked = [
            moment
            for item in items
            if isinstance(item, dict) and (moment := _instant(item.get("created_at"))) is not None
        ]
        return min(asked, default=None)

    async def service(self) -> dict[str, Any] | None:
        record = self._get("/service")
        return {key: record.get(key) for key in _SERVICE_FIELDS}


class _StoreAgentStatus:
    """A local store's pending approvals, questions and recovery items, counted from its
    stored state when the store keeps a native count; a store has no service."""

    def __init__(self, store: SessionStore) -> None:
        self._store = store

    async def inbox_counts(self) -> dict[str, int] | None:
        from cayu.sessions._inbox_item_ids import PENDING_INBOX_KINDS

        try:
            pending = await self._store.count_pending_actions()
        except NotImplementedError:
            return None
        return {listed: pending.get(kind, 0) for kind, listed in PENDING_INBOX_KINDS.items()}

    async def oldest_inbox_item_at(self) -> datetime | None:
        return None

    async def service(self) -> dict[str, Any] | None:
        return None


def _instant(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        moment = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return moment if moment.tzinfo is not None else moment.replace(tzinfo=UTC)


def open_store_agent_readers(target: SessionStoreTarget) -> AgentReaders:
    """Readers for the local store ``target`` (``cayu.storage.targets``), opened read-only,
    as `cayu task --sqlite/--postgres` opens it."""

    from cayu.cli._session_reader import StoreSessionReader
    from cayu.cli._task_readers import open_store_task_reader
    from cayu.cli.session import _open_read_only_store

    tasks = open_store_task_reader(target)
    store = _open_read_only_store(target)
    return AgentReaders(
        tasks=tasks,
        sessions=StoreSessionReader(store),
        source=tasks.source,
        status=_StoreAgentStatus(store),
    )

"""The routes every Agent's web app reads, generated into `integrations/http/generated/`.

`GET /app/whoami`, `GET /app/settings`, `GET /app/overview`, `GET /app/task-runs` and
`/app/inbox`. Each file says `GENERATED.` in its first line: the generator owns it, and the
Agent supplies what only it knows (who a request's person is, its overview metrics, the
settings it reads from its environment, how its tasks group into runs) as arguments to
`build_generated_router`. A preset that has people composes them into its own `/app` router;
`cayu new --preset b2b` does (`scaffold_b2b.py`).
"""

from __future__ import annotations

# The package the routes are generated into, relative to the project.
APP_ROUTES_PACKAGE = "integrations/http/generated"

_INIT_PY = '''"""GENERATED. The routes every Agent's web app reads, under /app, next to the Agent's own.

- `GET /app/whoami`: who is signed in and what their role allows (`whoami`);
- `GET /app/settings`: the Agent's managed settings (`settings`);
- `GET /app/overview`: the Agent's metrics and Cayu's inbox counts (`overview`);
- `GET /app/task-runs`: the Agent's Cayu tasks, one row per run (`task_runs`);
- `/app/inbox`: Cayu's inbox for this Agent, read and settled as the person (`inbox`).

The Agent supplies what only it knows, as arguments to `build_generated_router`: who a
request's person is and their role, its overview metrics, the settings it reads from its
environment, and how its tasks group into runs. Each module's `router` is usable on its
own, for an Agent that wants some of these routes and not others.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from cayu import CayuApp
from fastapi import APIRouter

from integrations.http.generated import inbox, overview, settings, task_runs, whoami
from integrations.http.generated.overview import Metric, OverviewContext, OverviewMetrics
from integrations.http.generated.settings import FixedSetting
from integrations.http.generated.task_runs import RunKey, RunTitle
from integrations.http.generated.whoami import Person, PersonDependency


def build_generated_router(
    app: CayuApp,
    *,
    person: PersonDependency,
    application: str,
    overview_metrics: OverviewMetrics,
    fixed_settings: Callable[[], Sequence[FixedSetting]] | None = None,
    run_key: RunKey | None = None,
    run_title: RunTitle | None = None,
    agent_name: str | None = None,
    inbox_page: str | None = None,
) -> APIRouter:
    """The generated routes, relative to /app. ``application`` is the Agent's name in Cayu
    Cloud, for the `cayu cloud settings` commands the Settings page shows.

    ``agent_name`` is the Agent's specialist in Cayu; with it, `/app/inbox` serves its
    inbox and the Overview counts only its items. ``inbox_page`` is where the Overview sends a person to decide what waits: the
    app's own Inbox (for example `/inbox`), or by default the Developer dashboard's."""

    routes = APIRouter()
    routes.include_router(whoami.router(person))
    routes.include_router(
        settings.router(app, person=person, application=application, fixed=fixed_settings or tuple)
    )
    routes.include_router(
        overview.router(
            app,
            person=person,
            metrics=overview_metrics,
            inbox_page=inbox_page or overview.INBOX_PAGE,
            agent_name=agent_name,
        )
    )
    if agent_name is not None:
        routes.include_router(inbox.router(person=person, agent_name=agent_name))
    routes.include_router(
        task_runs.router(
            app,
            person=person,
            run_key=run_key or task_runs.default_run_key,
            run_title=run_title or task_runs.default_run_title,
        )
    )
    return routes


__all__ = [
    "FixedSetting",
    "Metric",
    "OverviewContext",
    "OverviewMetrics",
    "Person",
    "PersonDependency",
    "build_generated_router",
]
'''

_CAYU_API_PY = '''"""GENERATED. Cayu's own API on this server, called in-process as the person behind a request.

What Cayu already serves (the inbox, its counts) is read through its own route instead of
being reimplemented: the request goes to this same server under `/api` without leaving the
process, with the person's credentials (Cayu Cloud's identity header, a cookie, or the
operator login), so Cayu checks them and applies its own rules, as on its dashboard.
"""

from __future__ import annotations

import os

import httpx
from fastapi import Request

# Where `cayu serve` serves Cayu's API.
CAYU_API_PATH = "/api"
# What the app says when Cayu doesn't let the person use its inbox, or doesn't answer.
INBOX_CLOSED = (
    "Cayu's inbox is open to this Agent's operators, not to you: they decide what waits on "
    "the Developer dashboard."
)
INBOX_SILENT = "Cayu's inbox didn't answer."
# What says who the person is, and where their request came from (Cayu refuses a
# cross-origin write from a signed-in browser).
FORWARDED_HEADERS = (
    "authorization",
    "x-cayu-identity",
    "cookie",
    "host",
    "origin",
    "referer",
    "sec-fetch-site",
)


def cayu_api_client(request: Request, *, timeout: float = 30.0) -> httpx.AsyncClient:
    """An HTTP client for Cayu's API on this server, as the person who sent ``request``.

    Close it after use (``async with``).
    """

    names = set(FORWARDED_HEADERS)
    configured = os.environ.get("CAYU_PROXY_IDENTITY_HEADER", "").strip().lower()
    if configured:
        names.add(configured)
    return httpx.AsyncClient(
        # Cayu's own 500 comes back as a response, as it would over the network, so the
        # routes' handling of a failed answer runs instead of the exception escaping.
        transport=httpx.ASGITransport(app=request.app, raise_app_exceptions=False),
        base_url=f"{request.url.scheme}://{request.url.netloc}{CAYU_API_PATH}",
        headers={name: value for name, value in request.headers.items() if name.lower() in names},
        timeout=timeout,
    )
'''

_WHOAMI_PY = '''"""GENERATED. `GET /app/whoami`: who is signed in, and what their role lets them do.

The web app's shell reads it once per page load for the profile menu, and hides actions a
role can't take (the server refuses them either way). The Agent decides who a request's
person is and which role they have: it passes a `PersonDependency` to
`build_generated_router`.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Literal

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel

Role = Literal["admin", "developer", "member"]


@dataclass(frozen=True, slots=True)
class Person:
    """The person behind a request, as the web app shows them.

    `may_decide`: they may decide what waits for a person (approve, answer).
    `may_change`: they may change the Agent's state or settings, beyond their own work.
    """

    subject: str
    role: Role
    may_decide: bool
    may_change: bool
    name: str | None = None
    email: str | None = None


PersonDependency = Callable[[Request], Awaitable[Person]]


class PersonView(BaseModel):
    subject: str
    name: str | None
    email: str | None


class Whoami(BaseModel):
    person: PersonView
    role: Role
    may_decide: bool
    may_change: bool


def whoami(person: Person) -> Whoami:
    return Whoami(
        person=PersonView(subject=person.subject, name=person.name, email=person.email),
        role=person.role,
        may_decide=person.may_decide,
        may_change=person.may_change,
    )


def router(person: PersonDependency) -> APIRouter:
    routes = APIRouter()

    @routes.get("/whoami")
    async def get_whoami(signed_in: Person = Depends(person)) -> Whoami:  # noqa: B008
        return whoami(signed_in)

    return routes
'''

_SETTINGS_PY = '''"""GENERATED. `GET /app/settings`: the Agent's managed settings, and the ones that need a redeploy.

Every setting the application declares (`CayuApp.managed_settings()`: Cayu's own run
limits and retries, budget limits, and the Agent's `app.*` settings) with its value, where
the value comes from (set in Cayu Cloud, or the release's default), when a change applies,
and the `cayu cloud settings` commands that change it. A change is made in Cayu Cloud,
which checks the person's role and keeps who changed what; the web app shows the settings
and can't change them itself yet.

`read_only` lists what the Agent reads once when a process starts, from its environment
(`FixedSetting`): changing one needs a redeploy. The Agent names them.
"""

from __future__ import annotations

import json
import os
import shlex
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from cayu import CayuApp
from cayu.managed_settings import ManagedSetting
from fastapi import APIRouter, Depends
from pydantic import BaseModel

from integrations.http.generated.whoami import PersonDependency

Value = bool | int | float | str | None
Kind = Literal["bool", "int", "number", "choice", "text"]

# Where a setting is listed, by the start of its key; the first match wins.
GROUPS = (
    ("config.run.limits.", "Run limits"),
    ("config.run.retry_policy.", "Model calls"),
    ("config.run.thinking.", "Model calls"),
    ("config.", "Runs"),
    ("budget.", "Budgets"),
    ("app.", "This Agent"),
    ("agent.", "Agents"),
    ("provider.", "Providers"),
    ("environment.", "Environments"),
)
APPLIES = {
    "live": "Right away, without a restart.",
    "next_session": "Sessions that start after the change.",
    "restart": "After the Agent next restarts.",
}
KINDS: dict[str, Kind] = {
    "bool": "bool",
    "int": "int",
    "number": "number",
    "enum": "choice",
    "string": "text",
    "secret_ref": "text",
}


@dataclass(frozen=True, slots=True)
class FixedSetting:
    """A setting the Agent reads from its environment when a process starts."""

    name: str
    label: str
    help: str
    # Why a change needs a redeploy.
    why: str
    env_var: str | None = None
    # The value in effect, as the code that uses it reads it.
    current: Callable[[], Value] | None = None
    # A check on the value in effect: a warning, or None.
    warn: Callable[[Value], str | None] | None = None


class LiveSettingRow(BaseModel):
    name: str
    key: str
    env_var: str | None
    group: str
    label: str
    help: str
    applies: str
    kind: Kind
    choices: list[str]
    minimum: int | float | None
    maximum: int | float | None
    blankOk: bool  # noqa: N815 (the web app's field name)
    # None when the setting has no value (an unset limit).
    value: Value
    # managed (set in Cayu Cloud) or default (the release's value).
    source: Literal["managed", "default"]
    environment: str | None
    default: Value
    note: str
    error: str | None
    change: str
    reset: str


class FixedSettingRow(BaseModel):
    name: str
    env_var: str | None
    label: str
    help: str
    value: Value
    source: str | None
    needs: str
    why: str
    warning: str | None


class CloudState(BaseModel):
    # applied, stale (past its validity) or unreported (never read); None outside Cloud.
    state: str | None
    # What the last read did: ready, unavailable, schema_missing; None outside Cloud.
    status: str | None


class SettingsView(BaseModel):
    settings: list[LiveSettingRow]
    read_only: list[FixedSettingRow]
    managed: bool
    cloud: CloudState
    history: str


def settings_view(
    app: CayuApp, *, application: str, fixed: Sequence[FixedSetting] = ()
) -> SettingsView:
    controller = app.managed_settings_controller
    cloud = CloudState(
        state=None if controller is None else controller.state(),
        status=None if controller is None else controller.status,
    )
    return SettingsView(
        settings=[_live(app, setting, cloud, application) for setting in app.managed_settings()],
        read_only=[_fixed(setting) for setting in fixed],
        managed=controller is not None,
        cloud=cloud,
        history=(
            f"cayu cloud settings list --application {shlex.quote(application)} "
            "shows each value and where it comes from"
        ),
    )


def router(
    app: CayuApp,
    *,
    person: PersonDependency,
    application: str,
    fixed: Callable[[], Sequence[FixedSetting]],
) -> APIRouter:
    # Only a signed-in person reads them; any role may.
    routes = APIRouter(dependencies=[Depends(person)])

    @routes.get("/settings")
    async def get_settings() -> SettingsView:
        return settings_view(app, application=application, fixed=fixed())

    return routes


def _live(
    app: CayuApp, setting: ManagedSetting, cloud: CloudState, application: str
) -> LiveSettingRow:
    controller = app.managed_settings_controller
    snapshot = None
    if controller is not None:
        # A restart setting applies the value read when the app was built.
        snapshot = controller.build_snapshot if setting.applies == "restart" else controller.snapshot
    value = app.managed_setting_value(setting.key)
    error = None
    source: Literal["managed", "default"] = "default"
    if snapshot is not None:
        # A refused value can be missing from `values`: Cloud reports it only as a conflict.
        reason = next(
            (item.reason for item in snapshot.conflicts if item.key == setting.key), None
        )
        if reason is not None:
            error = f"The value set in Cayu Cloud is refused: {reason}."
        elif setting.key in snapshot.values:
            if snapshot.values[setting.key] == value:
                source = "managed"
            else:
                error = "The value set in Cayu Cloud is refused."
    quoted = shlex.quote(application)
    shown = "VALUE" if value is None and setting.default is None else json.dumps(
        value if value is not None else setting.default
    )
    return LiveSettingRow(
        name=setting.key,
        key=setting.key,
        env_var=None,
        group=next((label for prefix, label in GROUPS if setting.key.startswith(prefix)), "Other"),
        label=setting.label,
        help=setting.description or "",
        applies=APPLIES[setting.applies],
        kind=KINDS[setting.type],
        choices=[choice.value for choice in setting.choices or ()],
        minimum=setting.minimum,
        maximum=setting.maximum,
        blankOk=setting.default is None,
        value=value,
        source=source,
        environment=None,
        default=setting.default,
        note=_note(setting, source, cloud),
        error=error,
        change=f"cayu cloud settings set --application {quoted} "
        + shlex.quote(f"{setting.key}={shown}"),
        reset=f"cayu cloud settings unset --application {quoted} {shlex.quote(setting.key)}",
    )


def _note(setting: ManagedSetting, source: str, cloud: CloudState) -> str:
    if cloud.state == "unreported":
        why = " (Cloud has no schema for this release yet)" if cloud.status == "schema_missing" else ""
        return (
            f"Cayu Cloud's settings couldn't be read{why}, so {setting.key} as set there, if it "
            "is, doesn't apply yet; this is the value without it."
        )
    stale = (
        " Cayu Cloud hasn't answered for over five minutes; this is its last value."
        if cloud.state == "stale"
        else ""
    )
    if source == "managed":
        return f"Set in Cayu Cloud as {setting.key}.{stale}"
    if setting.default is None:
        return f"Not set in the release; set {setting.key} in Cayu Cloud to set it.{stale}"
    return f"The release's value; set {setting.key} in Cayu Cloud to change it.{stale}"


def _fixed(setting: FixedSetting) -> FixedSettingRow:
    problem = None
    value: Any = None
    if setting.current is not None:
        try:
            value = setting.current()
        except (RuntimeError, ValueError) as exc:
            problem = str(exc)
    return FixedSettingRow(
        name=setting.name,
        env_var=setting.env_var,
        label=setting.label,
        help=setting.help,
        value=value,
        source=None
        if setting.env_var is None
        else ("environment" if os.environ.get(setting.env_var, "").strip() else "default"),
        needs="redeploy" if setting.env_var else "not a setting yet",
        why=setting.why,
        warning=problem or (setting.warn(value) if setting.warn else None),
    )
'''

_OVERVIEW_PY = '''"""GENERATED. `GET /app/overview`: the Agent's metrics and what waits for a person in Cayu's inbox.

- `metrics` come from the Agent's `OverviewMetrics` hook, the one function it writes for
  this page. It gets the signed-in person, so an Agent that several people or
  organizations share counts only what that person may see.
- `inbox` counts the items waiting in Cayu's inbox by kind, read from Cayu's own route
  (`GET /api/inbox/counts`) in this process as the person (`cayu_api`), so the counts are
  the ones `cayu inbox list` and the Developer dashboard show them. When Cayu doesn't let
  the person read the inbox, or can't count it, `unavailable` says so instead of a total.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

import httpx
from cayu import CayuApp
from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel

from integrations.http.generated.cayu_api import INBOX_CLOSED, INBOX_SILENT, cayu_api_client
from integrations.http.generated.whoami import Person, PersonDependency

logger = logging.getLogger(__name__)

INBOX_KINDS = {
    "approval": "Approvals",
    "question": "Questions",
    "recovery": "Interrupted sessions",
    "knowledge": "Knowledge proposals",
}
# Where a person decides inbox items by default: the Developer dashboard's Inbox. An app
# with its own Inbox passes its path instead (`router(inbox_page=...)`).
INBOX_PAGE = "/cayu/inbox"


@dataclass(frozen=True, slots=True)
class Metric:
    label: str
    value: str
    description: str | None = None


@dataclass(frozen=True, slots=True)
class OverviewContext:
    """What the Agent's metrics hook gets: its app, the person, and their request."""

    app: CayuApp
    person: Person
    request: Request


OverviewMetrics = Callable[[OverviewContext], Awaitable[Sequence[Metric]]]


class MetricView(BaseModel):
    label: str
    value: str
    description: str | None


class InboxKind(BaseModel):
    kind: str
    label: str
    count: int
    # Where to decide them: an app path, or the Developer dashboard's (`/cayu/...`).
    to: str | None


class InboxView(BaseModel):
    total: int
    by_kind: list[InboxKind]
    # What the empty table says.
    empty: str
    # Why there are no counts: the inbox isn't open to this person, or couldn't be counted.
    unavailable: str | None


class OverviewView(BaseModel):
    as_of: str
    metrics: list[MetricView]
    inbox: InboxView


async def inbox_counts(
    request: Request, *, inbox_page: str = INBOX_PAGE, agent_name: str | None = None
) -> InboxView:
    """Cayu's inbox counts, as the person behind ``request`` may read them, of the agent
    named ``agent_name`` (every agent's without one). Each kind links to ``inbox_page``
    filtered to it."""

    params = {} if agent_name is None else {"agent_name": agent_name}
    try:
        async with cayu_api_client(request) as client:
            response = await client.get("/inbox/counts", params=params)
    except httpx.HTTPError as exc:
        logger.warning("%s %s", INBOX_SILENT, type(exc).__name__)
        return _unavailable(INBOX_SILENT)
    if response.status_code in (401, 403):
        return _unavailable(INBOX_CLOSED)
    if response.status_code >= 400:
        return _unavailable(f"Cayu's inbox answered {response.status_code}.")
    answer = response.json()
    if not answer.get("complete"):
        reason = answer.get("unavailable_reason")
        return _unavailable(
            "Cayu shows you only part of the inbox, so it gives no totals."
            if reason == "access_limited"
            else "Cayu couldn't count the whole inbox just now."
        )
    counts = answer.get("counts") or {}
    by_kind = [
        InboxKind(
            kind=kind, label=label, count=counts.get(kind, 0), to=f"{inbox_page}?kind={kind}"
        )
        for kind, label in INBOX_KINDS.items()
        if counts.get(kind)
    ]
    return InboxView(
        total=answer.get("total_count") or 0,
        by_kind=by_kind,
        empty="No approval, question, interrupted session or knowledge proposal is waiting.",
        unavailable=None,
    )


def _unavailable(reason: str) -> InboxView:
    return InboxView(total=0, by_kind=[], empty=reason, unavailable=reason)


def router(
    app: CayuApp,
    *,
    person: PersonDependency,
    metrics: OverviewMetrics,
    inbox_page: str = INBOX_PAGE,
    agent_name: str | None = None,
) -> APIRouter:
    """`GET /overview`, relative to /app. With ``agent_name``, the inbox counts are that
    agent's, as `/app/inbox` lists them."""

    routes = APIRouter()

    @routes.get("/overview")
    async def get_overview(
        request: Request,
        signed_in: Person = Depends(person),  # noqa: B008
    ) -> OverviewView:
        found = await metrics(OverviewContext(app=app, person=signed_in, request=request))
        return OverviewView(
            as_of=datetime.now(UTC).isoformat(),
            metrics=[
                MetricView(label=item.label, value=item.value, description=item.description)
                for item in found
            ],
            inbox=await inbox_counts(request, inbox_page=inbox_page, agent_name=agent_name),
        )

    return routes
'''

_TASK_RUNS_PY = '''"""GENERATED. `GET /app/task-runs`: the Agent's Cayu tasks, one row per run of its work.

The TaskRuns block in the web app lists these. A run is every task that did one piece of
the Agent's work: when a worker is lost and the work runs again as a new task, both tasks
are the same run. The Agent says which run a task belongs to with `run_key` (by default the
`run` its metadata names, else the task itself) and how a run is titled with `run_title`.

Tasks are read as the signed-in person (`await app.access(subject)`), so the list holds
only the tasks Cayu's resource-access policy lets them see.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from typing import Literal

from cayu import CayuApp
from cayu.tasks import Task, TaskOrder, TaskQuery, TaskStatus
from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel

from integrations.http.generated.whoami import Person, PersonDependency

# How many of the newest tasks a listing groups.
TASK_WINDOW = 1000
# How many open tasks of one status a listing reads, the most one task query returns.
OPEN_TASK_READ = 1000

RunStatus = Literal[
    "queued",
    "scheduled",
    "running",
    "paused",
    "needs_attention",
    "completed",
    "partly_failed",
    "failed",
    # Every task read has ended, but not all of the run's tasks could be read, so it may
    # still have open work: never reported as ended.
    "partial",
]
_WAITING = {
    TaskStatus.PENDING,
    TaskStatus.WAITING_DEPENDENCIES,
    TaskStatus.WAITING_GROUP,
}
_ACTIVE = {TaskStatus.CLAIMED, TaskStatus.RUNNING}
_HELD = {TaskStatus.PAUSED, TaskStatus.BLOCKED}
_FAILED = {TaskStatus.FAILED, TaskStatus.CANCELLED, TaskStatus.DEPENDENCY_SKIPPED}
# Every status of a task that hasn't ended.
_OPEN = (*_WAITING, *_ACTIVE, *_HELD, TaskStatus.NEEDS_ATTENTION)

RunKey = Callable[[Task], str]
RunTitle = Callable[[Sequence[Task]], str]


def default_run_key(task: Task) -> str:
    run = (task.metadata or {}).get("run")
    return run if isinstance(run, str) and run else task.id


def default_run_title(tasks: Sequence[Task]) -> str:
    first = tasks[0]
    return first.title or first.type


class TaskRun(BaseModel):
    id: str
    title: str
    task_type: str
    status: RunStatus
    started_at: str
    ended_at: str | None
    # Every task of the run, and every restart a worker gave one.
    attempts: int
    tasks: int
    # Not every task of the run could be read: `tasks`, `attempts` and `started_at` may
    # leave out older ones, and a status that would say it ended is `partial` instead.
    partial: bool = False


def run_status(tasks: Sequence[Task], now: datetime) -> RunStatus:
    """What a run's tasks (oldest first) add up to: open while any task is."""

    statuses = [task.status for task in tasks]
    if any(status in _ACTIVE for status in statuses):
        return "running"
    if any(status is TaskStatus.NEEDS_ATTENTION for status in statuses):
        return "needs_attention"
    if any(status in _HELD for status in statuses):
        return "paused"
    waiting = [task for task in tasks if task.status in _WAITING]
    if waiting:
        later = all(task.available_at is not None and task.available_at > now for task in waiting)
        return "scheduled" if later else "queued"
    # Every task has ended.
    if all(status is TaskStatus.COMPLETED for status in statuses):
        return "completed"
    if all(status in _FAILED for status in statuses):
        return "failed"
    return "partly_failed"


def run_row(
    key: str, tasks: Sequence[Task], title: RunTitle, now: datetime, *, partial: bool = False
) -> TaskRun:
    first = tasks[0]
    status = run_status(tasks, now)
    if partial and status in ("completed", "partly_failed", "failed"):
        # A task it didn't read may still be open.
        status = "partial"
    # Tasks that overlap may end in another order than they started: the run ends with the last.
    completed = [task.completed_at for task in tasks if task.completed_at is not None]
    ended = max(completed, default=None) if status in ("completed", "partly_failed", "failed") else None
    return TaskRun(
        id=key,
        title=title(tasks),
        task_type=first.type,
        status=status,
        started_at=_iso(first.started_at or first.created_at),
        ended_at=None if ended is None else _iso(ended),
        attempts=len(tasks) + sum(task.restart_count for task in tasks),
        tasks=len(tasks),
        partial=partial,
    )


async def list_task_runs(
    app: CayuApp,
    subject: str,
    *,
    task_type: str | None = None,
    limit: int = 50,
    run_key: RunKey = default_run_key,
    run_title: RunTitle = default_run_title,
    now: datetime | None = None,
    window: int = TASK_WINDOW,
) -> list[TaskRun]:
    """The newest runs first, of the tasks ``subject`` may see.

    It reads the newest ``window`` tasks; when that is all of them, every run is complete.
    When there are more, it also reads the open tasks of each status, so a run whose open
    task is older than the window still shows it as open. Those reads aren't one snapshot
    (a task can change status between them, and the task store has no cursor to read a
    status's tasks past one page), so then no run is reported as ended: a run whose tasks
    read have all ended says ``partial``, and every run sets ``partial``."""

    now = now or datetime.now(UTC)
    if app.task_store is None:
        # An Agent without a task store has no tasks.
        return []
    access = await app.access(subject)
    tasks = list(
        await access.tasks.list(
            TaskQuery(type=task_type, order_by=TaskOrder.CREATED_AT_DESC, limit=window)
        )
    )
    partial = len(tasks) >= window
    if partial:
        seen = {task.id for task in tasks}
        for status in _OPEN:
            for task in await access.tasks.list(
                TaskQuery(
                    type=task_type,
                    status=status,
                    order_by=TaskOrder.CREATED_AT_ASC,
                    limit=OPEN_TASK_READ,
                )
            ):
                if task.id not in seen:
                    seen.add(task.id)
                    tasks.append(task)
    groups: dict[str, list[Task]] = {}
    for task in sorted(tasks, key=lambda item: item.created_at):
        groups.setdefault(run_key(task), []).append(task)
    rows = [
        run_row(key, items, run_title, now, partial=partial) for key, items in groups.items()
    ]
    rows.sort(key=lambda row: row.started_at, reverse=True)
    return rows[:limit]


def router(
    app: CayuApp,
    *,
    person: PersonDependency,
    run_key: RunKey = default_run_key,
    run_title: RunTitle = default_run_title,
) -> APIRouter:
    routes = APIRouter()

    @routes.get("/task-runs")
    async def get_task_runs(
        signed_in: Person = Depends(person),  # noqa: B008
        type: str | None = Query(default=None, min_length=1, max_length=200),  # noqa: A002
        limit: int = Query(default=50, ge=1, le=500),
    ) -> list[TaskRun]:
        return await list_task_runs(
            app, signed_in.subject, task_type=type, limit=limit, run_key=run_key, run_title=run_title
        )

    return routes


def _iso(value: datetime) -> str:
    return value.astimezone(UTC).isoformat()
'''

_INBOX_PY = '''"""GENERATED. `/app/inbox`: Cayu's inbox for this Agent, as the signed-in person, for a client
of the Agent's own API that lists or decides what waits for a person.

- `GET /app/inbox` and `GET /app/inbox/counts` answer what Cayu's `GET /api/inbox` and
  `GET /api/inbox/counts` answer, filtered to this Agent (`agent_name`).
- `POST /app/inbox/{item_id}/settle` settles an item of this Agent through Cayu's
  `POST /api/inbox/{item_id}/settle`; an item of another agent looks missing.

Each request goes to Cayu's own route in this process with the person's credentials
(`cayu_api`), so Cayu checks them and records the decision as theirs, exactly as on the
Developer dashboard. Nothing here widens what Cayu allows. When Cayu refuses the person
(a deployment whose `[tool.cayu.serve].auth` is `OPERATOR_ONLY_AUTH` refuses members),
these routes answer 403 with a message the app shows, never Cayu's 401, which the app
would take for an expired sign-in.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx
from fastapi import APIRouter, Body, Depends, Query, Request
from fastapi.responses import JSONResponse

from integrations.http.generated.cayu_api import INBOX_CLOSED, INBOX_SILENT, cayu_api_client
from integrations.http.generated.whoami import PersonDependency

logger = logging.getLogger(__name__)

_NOT_FOUND = "Inbox item not found."


def router(*, person: PersonDependency, agent_name: str) -> APIRouter:
    """The inbox routes, relative to /app, for the agent named ``agent_name``."""

    # Only a signed-in person; Cayu decides what they may read and settle.
    routes = APIRouter(dependencies=[Depends(person)])

    @routes.get("/inbox")
    async def list_inbox(
        request: Request,
        limit: int = Query(50, ge=1, le=200),  # noqa: B008
        cursor: str | None = None,
        kind: str | None = None,
        q: str | None = None,
        session_id: str | None = None,
    ) -> JSONResponse:
        query = {"limit": limit, "cursor": cursor, "kind": kind, "q": q, "session_id": session_id}
        return await _forward(request, "GET", "/inbox", params=_with_agent(query, agent_name))

    @routes.get("/inbox/counts")
    async def inbox_counts(
        request: Request,
        q: str | None = None,
        session_id: str | None = None,
    ) -> JSONResponse:
        query = {"q": q, "session_id": session_id}
        return await _forward(request, "GET", "/inbox/counts", params=_with_agent(query, agent_name))

    @routes.post("/inbox/{item_id}/settle")
    async def settle(
        request: Request,
        item_id: str,
        body: dict[str, Any] = Body(...),  # noqa: B008
    ) -> JSONResponse:
        # Read the item first, as the person, so the app settles only this Agent's items.
        status, current = await _call(request, "GET", f"/inbox/{item_id}")
        if status != 200:
            return JSONResponse(current, status_code=status)
        if current.get("agent_name") != agent_name:
            return JSONResponse({"detail": _NOT_FOUND}, status_code=404)
        return await _forward(request, "POST", f"/inbox/{item_id}/settle", json=body)

    return routes


def _with_agent(query: dict[str, Any], agent_name: str) -> dict[str, Any]:
    return {key: value for key, value in query.items() if value is not None} | {
        "agent_name": agent_name
    }


async def _forward(request: Request, method: str, path: str, **kwargs: Any) -> JSONResponse:
    """Cayu's answer to the same request, as the person behind ``request``."""

    status, content = await _call(request, method, path, **kwargs)
    return JSONResponse(content, status_code=status)


async def _call(
    request: Request, method: str, path: str, **kwargs: Any
) -> tuple[int, dict[str, Any]]:
    try:
        async with cayu_api_client(request) as client:
            response = await client.request(method, path, **kwargs)
    except httpx.HTTPError as exc:
        logger.warning("%s %s", INBOX_SILENT, type(exc).__name__)
        return 503, {"detail": INBOX_SILENT}
    try:
        content = response.json()
    except ValueError:
        content = None
    # Cayu refuses the person outright: no sign-in it takes (401), or a role its inbox isn't
    # open to. Any other 403 keeps Cayu's own reason (an item for another reviewer, say).
    if response.status_code == 401 or _role_forbidden(content):
        return 403, {"detail": INBOX_CLOSED}
    if not isinstance(content, dict):
        content = {"detail": f"Cayu's inbox answered {response.status_code}."}
    return response.status_code, content


def _role_forbidden(content: Any) -> bool:
    detail = content.get("detail") if isinstance(content, dict) else None
    return isinstance(detail, dict) and detail.get("code") == "role_forbidden"
'''


def app_route_files() -> dict[str, str]:
    """The generated route modules, by their path in the project."""

    return {
        f"{APP_ROUTES_PACKAGE}/{name}": text
        for name, text in (
            ("__init__.py", _INIT_PY),
            ("cayu_api.py", _CAYU_API_PY),
            ("whoami.py", _WHOAMI_PY),
            ("settings.py", _SETTINGS_PY),
            ("overview.py", _OVERVIEW_PY),
            ("task_runs.py", _TASK_RUNS_PY),
            ("inbox.py", _INBOX_PY),
        )
    }

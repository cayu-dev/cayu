"""The `web-app` capability: a web app for the Agent's own team, which `cayu serve` serves at /.

It composes parts that are each usable on their own:

- the web app (`scaffold_app_web.py`): the generic app with the operators layer, whose
  committed build `cayu serve` serves at / (`[tool.cayu.serve] ui`);
- the routes every Agent's web app reads (`scaffold_app_routes.py`), generated into
  `integrations/http/generated/`;
- Cayu's operator sign-in part (`cayu.server.operator_sign_in`) as the routes'
  `PersonDependency`, so the app's people are the Agent's operators: Cayu Cloud's, with their
  Cloud roles, when deployed, and the person running `cayu serve --dev` locally;
- the Agent's own router (`integrations/http/__init__.py`, `[tool.cayu.serve] routers`),
  where its routes of its own go.

With the `organizations` capability (the b2b preset), the app is the generic app with the
organizations layer instead, and its people are the organizations' members: that capability
composes `served_app_files` with its own sign-in and routes (`scaffold_organizations.py`).
"""

from __future__ import annotations

from cayu._operator_credentials import ENVIRONMENT_OPERATOR_AUTH_TARGET
from cayu.cli.scaffold_app_routes import app_route_files, cloud_application_slug
from cayu.cli.scaffold_app_web import CONVERSATIONS, OPERATORS, web_app_files
from cayu.cli.scaffold_plan import ORGANIZATIONS, WEB_APP, ApplicationPlan

# What `cayu serve` reads to serve the Agent's routes and the app.
SERVE_ROUTERS = 'routers = ["integrations.http:build_router"]\nui = "web/static"\n'
# The `[tool.cayu.serve]` table every preset with `cayu serve` starts from.
_SERVE_AUTH = '[tool.cayu.serve]\nauth = "{auth}"\n'

_HTTP_PY = '''"""THE AGENT'S OWN. The Agent's HTTP API, which `cayu serve` serves under /app
([tool.cayu.serve] routers in pyproject.toml), next to Cayu's API at /api and its Developer
dashboard at /cayu/. The web app at / (web/, [tool.cayu.serve] ui) calls only these routes.

- /app/whoami, /app/settings, /app/overview, /app/task-runs and /app/inbox: the routes every
  Agent's web app reads (integrations/http/generated/), given who this Agent's people are
  (`person` below), its overview metrics (`overview_metrics.py`) and the settings it reads
  from its environment (`fixed_settings.py`).
- The Agent's own routes go on `router` below, behind the same sign-in.

Its people are its operators, who sign in as they do to the Developer dashboard: under
`cayu serve --dev` the person running it, on a loopback address only; deployed, through the
auth target `[tool.cayu.serve].auth` names, so Cayu Cloud's operators come with their Cloud
roles (`cayu.server.operator_sign_in`).
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends

from cayu import CayuApp
from cayu.server.operator_sign_in import operator_person_dependency, operator_roles
from configuration.settings import configured_cloud_application
from integrations.http.fixed_settings import fixed_settings
from integrations.http.generated import Person, build_generated_router
from integrations.http.generated.task_runs import operator_tasks
from integrations.http.overview_metrics import overview_metrics

# The agent the web app's Inbox and Overview are about.
AGENT_NAME = "__AGENT_NAME__"
# The Agent's application in Cayu Cloud when Cloud doesn't say (CAYU_CLOUD_APPLICATION_ID):
# the slug `cayu cloud init` makes from the project's name.
CLOUD_APPLICATION = "__CLOUD_APPLICATION__"


def build_router(app: CayuApp, *, development: bool, auth: Any = None) -> APIRouter:
    """The routes under /app. `auth` is the auth dependency `cayu serve` resolved (`--auth`
    over `[tool.cayu.serve].auth`), or None under `cayu serve --dev`."""

    # Cayu Cloud's roles, as far as `auth` lets them use the Developer dashboard: every
    # operator reads and decides what waits for a person, admins and developers change
    # things, and under OPERATOR_ONLY_AUTH a `member` is the application's own user, not an
    # operator. With a protected review policy, approvals are decided where the review is
    # shown, on the Developer dashboard, so nobody decides here.
    rule = operator_roles(
        auth, person=Person, decide=() if app.human_review_purpose is not None else None
    )
    person = operator_person_dependency(auth, development=development, rule=rule)
    router = APIRouter(prefix="/app")
    router.include_router(
        build_generated_router(
            app,
            person=person,
            application=cloud_application(),
            overview_metrics=overview_metrics,
            fixed_settings=fixed_settings,
            # Operators see every task, as on the Developer dashboard.
            read_tasks=operator_tasks,
            agent_name=AGENT_NAME,
        )
    )

    @router.get("/agent")
    async def agent(signed_in: Person = Depends(person)) -> dict[str, str]:
        """The agent the web app's Conversations and Inbox are about."""

        return {"name": AGENT_NAME}

    # The Agent's own routes, behind the same sign-in:
    #
    #     @router.get("/findings")
    #     async def findings(signed_in: Person = Depends(person)) -> list[dict]: ...
    #
    # A route that changes the Agent's state takes
    # `Depends(may_change_dependency(person))` (cayu.server.operator_sign_in) instead, which
    # refuses a role that may not change things.
    return router


def cloud_application() -> str:
    """The application the Settings page's `cayu cloud settings` commands name: the one
    Cayu Cloud runs this Agent as, else the one `cayu cloud init` would create."""

    return configured_cloud_application() or CLOUD_APPLICATION
'''

_SETTINGS_APPEND = '''

def configured_cloud_application() -> str | None:
    """The application Cayu Cloud runs this Agent as (CAYU_CLOUD_APPLICATION_ID), if it says."""

    return os.environ.get("CAYU_CLOUD_APPLICATION_ID", "").strip() or None
'''

_FIXED_SETTINGS_PY = '''"""THE AGENT'S OWN. The settings this Agent reads from its environment when a process starts.

The web app's Settings page lists them under "Need a redeploy", next to the managed
settings Cayu Cloud changes while the Agent runs. Secrets (API keys, the database URL) are
left out: the page would show their values.
"""

from __future__ import annotations

from configuration.settings import configured_model, configured_provider_choice
from integrations.http.generated import FixedSetting


def fixed_settings() -> tuple[FixedSetting, ...]:
    return (
        FixedSetting(
            name="provider",
            env_var="CAYU_PROVIDER",
            label="Model provider",
            help="The provider the Agent's model calls go to. Unset, the Agent has none.",
            why="Chosen when each process builds the app; a change needs a redeploy.",
            current=configured_provider_choice,
        ),
        FixedSetting(
            name="model",
            env_var="CAYU_MODEL",
            label="Model",
            help="The default model, until Cayu Cloud binds the primary model slot.",
            why="The agent is built with its model when each process starts; a change needs a redeploy.",
            current=configured_model,
        ),
    )
'''

_OVERVIEW_METRICS_PY = '''"""THE AGENT'S OWN. The metrics on the web app's Overview.

Its people are its operators, who see all of its work, so the numbers count every session
and task in its stores. Replace them with what says most about this Agent's work. The generated
`GET /app/overview` (integrations/http/generated/overview.py) calls `overview_metrics`.
"""

from __future__ import annotations

from cayu import SessionStatus
from cayu.sessions import SessionQuery
from cayu.tasks import TaskQuery

from integrations.http.generated import Metric, OverviewContext

# Tasks counted at most; more shows as this many and a plus.
TASK_COUNT_LIMIT = 1000


async def overview_metrics(context: OverviewContext) -> list[Metric]:
    sessions = context.app.session_store
    every = await sessions.list_sessions(
        SessionQuery(limit=1, include_total_count=True)
    )
    running = await sessions.list_sessions(
        SessionQuery(status=SessionStatus.RUNNING, limit=1, include_total_count=True)
    )
    tasks = (
        []
        if context.app.task_store is None
        else await context.app.task_store.list_tasks(TaskQuery(limit=TASK_COUNT_LIMIT))
    )
    return [
        Metric(
            label="Sessions",
            value=str(every.total_count or 0),
            description="Every session in the Agent's store, background work's too.",
        ),
        Metric(
            label="Running",
            value=str(running.total_count or 0),
            description="Sessions running now.",
        ),
        Metric(
            label="Tasks",
            value=f"{len(tasks)}{'+' if len(tasks) == TASK_COUNT_LIMIT else ''}",
            description="Background work, listed by run under Task runs.",
        ),
    ]
'''

_TEST_WEB_APP_PY = '''"""The web app and its routes: who signs in, what each role may do, and the app at /.

Deployed, the routes use the auth `[tool.cayu.serve].auth` names; here a stand-in returns
the person each test signs in as.
"""

from __future__ import annotations

import re
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.testclient import TestClient

from cayu import CayuApp
from cayu.server import serve_web_app
from cayu.server.auth import AuthContext
from cayu.server.proxy_identity import ControlPlaneRoles
from cayu.sessions import InMemorySessionStore
from cayu.tasks import InMemoryTaskStore, TaskCreate
from integrations.http import AGENT_NAME, build_router

# The built web app, `[tool.cayu.serve] ui` in pyproject.toml.
WEB = Path(__file__).resolve().parents[1] / "web" / "static"
# What `cayu serve --dev` admits: a loopback client that names a loopback host.
LOOPBACK = {"base_url": "http://127.0.0.1", "client": ("127.0.0.1", 50000)}


def _app() -> CayuApp:
    return CayuApp(session_store=InMemorySessionStore(), task_store=InMemoryTaskStore())


def _signed_in(role: str | None):
    """A stand-in for the auth target: the caller is `ada`, with `role` (none, as the
    operator login alone has), or nobody when `role` is "nobody"."""

    async def auth(request: Request) -> AuthContext:
        if role == "nobody":
            raise HTTPException(status_code=401, detail="Authentication required.")
        return AuthContext(subject="ada", claims={} if role is None else {"role": role})

    return auth


class _OperatorsOnly:
    """A stand-in for OPERATOR_ONLY_AUTH: Cayu Cloud's sign-in, which keeps its members, the
    application's own users, off the Developer dashboard."""

    control_plane_roles = ControlPlaneRoles(read_and_decide=())

    def __init__(self, role: str) -> None:
        self.role = role

    async def __call__(self, request: Request) -> AuthContext:
        return AuthContext(subject="ada", claims={"role": self.role})


def _client(
    app: CayuApp, *, development: bool = False, auth=None, **options
) -> TestClient:
    server = FastAPI()
    server.include_router(build_router(app, development=development, auth=auth))
    # As `cayu serve` serves it: last, for every path no route answers.
    serve_web_app(server, WEB)
    return TestClient(server, **options)


def test_every_route_needs_a_signed_in_operator() -> None:
    with _client(_app(), auth=_signed_in("nobody")) as client:
        for path in ("whoami", "settings", "overview", "task-runs"):
            assert client.get(f"/app/{path}").status_code == 401, path


def test_each_role_may_do_what_cayu_cloud_lets_it() -> None:
    expected = {
        None: ("admin", True, True),
        "admin": ("admin", True, True),
        "developer": ("developer", True, True),
        "member": ("member", True, False),
    }
    for role, (shown, decides, changes) in expected.items():
        with _client(_app(), auth=_signed_in(role)) as client:
            whoami = client.get("/app/whoami").json()
            assert whoami["person"]["subject"] == "ada"
            assert (whoami["role"], whoami["may_decide"], whoami["may_change"]) == (
                shown,
                decides,
                changes,
            )
    with _client(_app(), auth=_signed_in("viewer")) as client:
        assert client.get("/app/whoami").status_code == 403


def test_members_are_refused_where_they_are_the_applications_own_users() -> None:
    with _client(_app(), auth=_OperatorsOnly("member")) as client:
        assert client.get("/app/task-runs").status_code == 403
    with _client(_app(), auth=_OperatorsOnly("developer")) as client:
        assert client.get("/app/whoami").json()["may_change"] is True


def test_under_development_the_person_running_it_signs_in_from_this_machine_only() -> (
    None
):
    with _client(_app(), development=True, **LOOPBACK) as client:
        whoami = client.get("/app/whoami").json()
        assert (
            whoami["person"]["subject"].startswith("dev:") and whoami["role"] == "admin"
        )
    with _client(_app(), development=True) as client:
        assert client.get("/app/whoami").status_code == 403


def test_the_overview_settings_and_task_runs_answer() -> None:
    app = _app()
    with _client(app, auth=_signed_in("member")) as client:
        client.portal.call(
            app.task_store.create_task, TaskCreate(type="digest", title="Weekly digest")
        )
        overview = client.get("/app/overview")
        assert overview.status_code == 200, overview.text
        assert [metric["label"] for metric in overview.json()["metrics"]] == [
            "Sessions",
            "Running",
            "Tasks",
        ]
        assert client.get("/app/settings").status_code == 200
        # Operators see every task, one row per run.
        (run,) = client.get("/app/task-runs").json()
        assert run["title"] == "Weekly digest"
        assert client.get("/app/agent").json() == {"name": AGENT_NAME}


def test_the_web_app_is_served_at_the_root_with_only_its_built_files() -> None:
    with _client(_app(), auth=_signed_in("admin")) as client:
        page = client.get("/")
        assert page.status_code == 200
        assert page.headers["cache-control"] == "no-cache"
        (script,) = re.findall(r'src="/(assets/[^"]+\\.js)"', page.text)
        assert client.get(f"/{script}").status_code == 200
        assert client.get("/assets/missing.js").status_code == 404
        # The app routes in the browser: its pages answer with the page itself.
        deep = client.get("/task-runs")
        assert deep.status_code == 200 and deep.text == page.text
        # The API's own paths are never the page.
        assert client.get("/app/no-such-route").status_code == 404
'''

_GUIDANCE = """
## The web app

`cayu serve` serves this Agent's web app at `/`, its routes under `/app` and the Developer
dashboard at `/cayu/` (`[tool.cayu.serve] ui` and `routers` in `pyproject.toml`). Its people
are the Agent's operators, who sign in as they do to the Developer dashboard: deployed, Cayu
Cloud's operators with their Cloud roles (every role reads and decides, `admin` and
`developer` change things); under `cayu serve --dev`, you. `uv run --no-sync cayu serve
--dev` serves it at http://127.0.0.1:8000/.

Each file says in its first doc comment whether it is `GENERATED.` (the generator owns it and
may rewrite it) or `THE AGENT'S OWN.` (yours).

| Path | What it is | Whose |
| --- | --- | --- |
| `web/src/main.tsx`, `web/src/app/` | The shell: browser routes at `/`, the area pattern, the layout with its navigation and profile menu, the status pages, the API requester | Generated |
| `web/src/api/`, `web/src/generated/` | Typed clients for the generated routes, and the Overview and Settings areas | Generated |
| `web/src/blocks/task-runs.tsx` | TaskRuns: the Agent's Cayu tasks, one row per run (`GET /app/task-runs`) | A Cayu building block |
| `web/src/agent/` | This Agent's own areas (__OWN_AREAS__) and `index.tsx`, which lists its areas | The Agent's |
| `web/src/kit/shadcn/` | The component kit: stock shadcn (Base UI, base-nova) and the parts built from it | Generated |
| `web/e2e/` | The smoke test: `cayu serve --dev` serves the app, and every area opens in a browser | The Agent's |
| `integrations/http/__init__.py` | The routes under `/app`, signed in with `cayu.server.operator_sign_in`; add the Agent's own here | The Agent's |
| `integrations/http/generated/` | The routes the generated pages read | Generated |
| `integrations/http/overview_metrics.py`, `fixed_settings.py` | This Agent's Overview metrics and the settings it reads at start-up | The Agent's |

- Add an area in `web/src/agent/` and list it in `web/src/agent/index.tsx`, and its routes
  in `integrations/http/__init__.py`.
__CONVERSATIONS__- The navigation links to Cayu's Inbox (`/cayu/inbox`), where approvals and questions are
  decided, and to the Developer dashboard (`/cayu/`).
- After changing `web/src`, run `npm --prefix web ci` and `npm --prefix web run build`, and
  commit `web/static`; running and deploying need no Node. `npm --prefix web run
  check:build` fails when `web/static` isn't the build of `web/src`, and `npm --prefix web
  run dev` reloads it live against `cayu serve --dev`.
- The smoke test: `npm --prefix web/e2e ci`, `npx --prefix web/e2e playwright install
  chromium` once, then `npm --prefix web/e2e test`.
- `uv run --no-sync cayu cloud init` sets `home = "/"` in `cayu-cloud.toml`, so opening the
  Agent from Cayu Cloud lands on the app. `cayu guide app-ui` covers the app's parts.
- To leave the app out, generate with `cayu new NAME --without web-app`.
"""

_CONVERSATIONS_GUIDANCE = """\
- Conversations (`web/src/agent/conversations.tsx`) start, read and continue the agent's
  sessions through Cayu's own API at `/api` (`web/src/agent/cayu.ts`), as the Developer
  dashboard does, so Cayu checks each request: every role reads them, and `admin` and
  `developer` start and continue them. `GET /app/agent` names the agent.
"""

# The agent preset's routing row this capability answers differently.
_AGENTS_WEB_ROW = (
    "| A web UI that shows progress or live results | `mount_cayu(...)` and `connect()` from "
    "the served `client.js`. Do not poll on a fixed timer. | `cayu guide app-ui` |"
)
_AGENTS_WEB_APP_ROW = (
    "| A web UI that shows progress or live results | `web/`, served at `/`: add an area in "
    "`web/src/agent/` and list it in `web/src/agent/index.tsx`; leave `GENERATED.` files to the "
    'generator (README "The web app"). A page outside it follows Cayu with `connect()` from the '
    "served `client.js`; do not poll on a fixed timer. | `cayu guide app-ui` |"
)


def served_app_files(
    *,
    files: dict[str, str],
    title: str,
    layers: tuple[str, ...],
    auth: str = ENVIRONMENT_OPERATOR_AUTH_TARGET,
    package_name: str = "web",
) -> dict[str, str]:
    """What every web app adds, whoever its people are: `web/` with ``layers`` on the generic
    app, the generated routes in `integrations/http/generated/`, and `[tool.cayu.serve]`
    serving the Agent's router at /app and the app at /, behind ``auth``. The Agent's router
    in `integrations/http/__init__.py`, which says who its people are, is the caller's."""

    serve = _SERVE_AUTH.format(auth=ENVIRONMENT_OPERATOR_AUTH_TARGET)
    if files["pyproject.toml"].count(serve) != 1:
        raise RuntimeError("pyproject.toml has no [tool.cayu.serve] auth for the web app's routes.")
    return {
        "pyproject.toml": files["pyproject.toml"].replace(
            serve, _SERVE_AUTH.format(auth=auth) + SERVE_ROUTERS
        ),
        # Installed packages and build output of web/ and of a Developer dashboard ejected
        # with `cayu dashboard eject`, which writes no .gitignore of its own.
        ".gitignore": files[".gitignore"] + "node_modules/\ndist/\n",
        **app_route_files(),
        **web_app_files(title, layers=layers, package_name=package_name),
    }


def web_app_project_files(*, files: dict[str, str], plan: ApplicationPlan) -> dict[str, str]:
    """The web app's files, and its changes to ``files``, for a plan with `web-app` whose
    people are the Agent's operators. With `organizations`, they are its members instead
    (`scaffold_organizations.organizations_project_files`)."""

    if WEB_APP not in plan.capabilities or ORGANIZATIONS in plan.capabilities:
        return {}
    # People chat with the agent preset's agent, so its app has Conversations too.
    chats = plan.preset == "agent"
    guidance = _GUIDANCE.replace(
        "__OWN_AREAS__", "Conversations, Task runs" if chats else "Task runs"
    ).replace("__CONVERSATIONS__", _CONVERSATIONS_GUIDANCE if chats else "")
    return {
        **served_app_files(
            files=files,
            title=plan.name,
            layers=(OPERATORS, CONVERSATIONS) if chats else (OPERATORS,),
        ),
        "README.md": files["README.md"] + guidance,
        "AGENTS.md": files["AGENTS.md"].replace(_AGENTS_WEB_ROW, _AGENTS_WEB_APP_ROW) + guidance,
        "configuration/settings.py": files["configuration/settings.py"] + _SETTINGS_APPEND,
        "integrations/http/__init__.py": _HTTP_PY.replace(
            "__AGENT_NAME__", plan.agent_name
        ).replace("__CLOUD_APPLICATION__", cloud_application_slug(plan.name)),
        "integrations/http/fixed_settings.py": _FIXED_SETTINGS_PY,
        "integrations/http/overview_metrics.py": _OVERVIEW_METRICS_PY,
        "tests/test_web_app.py": _TEST_WEB_APP_PY,
    }

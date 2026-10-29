"""`[tool.cayu.serve] routers`: a project's own routes next to Cayu's, under `cayu serve`."""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient

from cayu.cli import main

_FACTORY = """from cayu import CayuApp


def build_app():
    return CayuApp(enable_logging=False)
"""

_ROUTES = """from fastapi import APIRouter


def build_router(app, *, development):
    router = APIRouter()

    @router.get("/app/hello")
    def hello():
        return {"development": development, "app": type(app).__name__}

    return router


def colliding(app, *, development):
    router = APIRouter()

    @router.get("/api/hello")
    def hello():
        return {}

    return router


def parameter_first(app, *, development):
    router = APIRouter()

    @router.post("/{first}/sessions")
    def anything(first: str):
        return {}

    return router


def catch_all(app, *, development):
    router = APIRouter()

    @router.get("/{path:path}")
    def anything(path: str):
        return {}

    return router


def root(app, *, development):
    router = APIRouter()

    @router.get("/")
    def home():
        return {}

    return router


def not_a_router(app, *, development):
    return object()
"""

_AUTH = """from fastapi import Request

from cayu.server import AuthContext


async def operator(request: Request) -> AuthContext:
    return AuthContext(subject="operator")
"""


def _project(tmp_path: Path, serve: str) -> Path:
    project = tmp_path / "project"
    project.mkdir()
    (project / "pyproject.toml").write_text(
        f'[tool.cayu]\nfactory = "serve_project:build_app"\n\n[tool.cayu.serve]\n{serve}',
        encoding="utf-8",
    )
    (project / "serve_project.py").write_text(_FACTORY, encoding="utf-8")
    (project / "project_routes.py").write_text(_ROUTES, encoding="utf-8")
    (project / "project_auth.py").write_text(_AUTH, encoding="utf-8")
    for module in ("serve_project", "project_routes", "project_auth"):
        sys.modules.pop(module, None)
    return project


@pytest.fixture
def launched(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    launched: dict[str, Any] = {}
    uvicorn = ModuleType("uvicorn")

    def run(server: Any, *, host: str, port: int) -> None:
        launched.update(server=server, host=host, port=port)

    uvicorn.run = run  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "uvicorn", uvicorn)
    return launched


@pytest.mark.parametrize("development", [True, False])
def test_a_projects_routes_are_served_next_to_cayus(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    launched: dict[str, Any],
    development: bool,
) -> None:
    project = _project(
        tmp_path,
        'routers = ["project_routes:build_router"]\nauth = "project_auth:operator"\n',
    )
    monkeypatch.chdir(project)

    assert main(["serve", "--dev"] if development else ["serve", "--host", "127.0.0.1"]) == 0

    client = TestClient(launched["server"])
    assert client.get("/app/hello").json() == {"development": development, "app": "CayuApp"}
    # Cayu's own routes are still there.
    assert any(getattr(route, "path", "").startswith("/api/") for route in client.app.routes)
    assert "Project routes: project_routes:build_router" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("routers", "message"),
    [
        ('["project_routes:colliding"]', "/api/hello collides with Cayu's own /api routes"),
        ('["project_routes:not_a_router"]', "must return a fastapi.APIRouter"),
        ('["project_routes:parameter_first"]', "must start with a fixed path segment"),
        ('["project_routes:catch_all"]', "must start with a fixed path segment"),
        ('["project_routes:root"]', "must start with a fixed path segment"),
        ('"project_routes:build_router"', "routers must be a non-empty array"),
        ("[]", "routers must be a non-empty array"),
    ],
)
def test_a_router_that_cant_be_served_safely_is_refused(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    launched: dict[str, Any],
    routers: str,
    message: str,
) -> None:
    project = _project(tmp_path, f"routers = {routers}\n")
    monkeypatch.chdir(project)

    assert main(["serve", "--dev"]) != 0

    assert "server" not in launched
    assert message in capsys.readouterr().err


_HOSTED_FACTORY = """from cayu import CayuApp
from cayu.external_wait_hosting import ExternalEventSource
from cayu.external_waits import ExternalWaitAccessPolicy, ExternalWaitContext
from cayu.sessions.external_waits import ExternalWaitScope
from cayu.sessions.memory import InMemorySessionStore
from cayu.tasks.memory import InMemoryTaskStore


class Policy(ExternalWaitAccessPolicy):
    def authorize(self, context, *, scope, source, action):
        return True


class Renderer(ExternalEventSource):
    source = "renderer"

    async def receive(self, request):
        return None


def build_app():
    app = CayuApp(
        session_store=InMemorySessionStore(), task_store=InMemoryTaskStore(), enable_logging=False
    )
    app.host_external_waits(
        access_policy=Policy(),
        context=ExternalWaitContext(principal="worker"),
        scopes={ExternalWaitScope(application_scope="video", generation=1): ("renderer",)},
        sources=(Renderer(),),
        delivery_path=DELIVERY_PATH,
    )
    return app
"""

_HOOK_ROUTES = """from fastapi import APIRouter


def build_router(app, *, development):
    router = APIRouter()

    @router.post("/webhooks/github")
    def github():
        return {"github": True}

    return router


def inside_delivery(app, *, development):
    router = APIRouter()

    @router.post("/webhooks/waits/extra")
    def extra():
        return {}

    return router
"""


def _hosted_project(tmp_path: Path, delivery_path: str, serve: str = "") -> Path:
    project = _project(tmp_path, serve)
    (project / "serve_project.py").write_text(
        f"DELIVERY_PATH = {delivery_path!r}\n" + _HOSTED_FACTORY, encoding="utf-8"
    )
    (project / "hook_routes.py").write_text(_HOOK_ROUTES, encoding="utf-8")
    sys.modules.pop("hook_routes", None)
    return project


@pytest.mark.parametrize("development", [True, False])
def test_serve_mounts_the_external_wait_delivery_route(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    launched: dict[str, Any],
    development: bool,
) -> None:
    project = _hosted_project(tmp_path, "/webhooks/waits", 'auth = "project_auth:operator"\n')
    monkeypatch.chdir(project)

    assert main(["serve", "--dev"] if development else ["serve", "--host", "127.0.0.1"]) == 0

    client = TestClient(launched["server"])
    # The route authenticates its own callers; this source refuses every request.
    response = client.post("/webhooks/waits/renderer", json={})
    assert response.status_code == 401 and response.json() == {"code": "unauthenticated"}
    assert "External-wait delivery: POST /webhooks/waits/{source}" in capsys.readouterr().out


def test_project_webhooks_can_share_the_delivery_routes_prefix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    launched: dict[str, Any],
) -> None:
    project = _hosted_project(
        tmp_path, "/webhooks/waits", 'routers = ["hook_routes:build_router"]\n'
    )
    monkeypatch.chdir(project)

    assert main(["serve", "--dev"]) == 0

    client = TestClient(launched["server"])
    assert client.post("/webhooks/github").json() == {"github": True}
    assert client.post("/webhooks/waits/renderer", json={}).status_code == 401


@pytest.mark.parametrize(
    ("delivery_path", "routers", "message"),
    [
        ("/api/waits", None, "/api/waits/{source} collides with Cayu's own /api routes"),
        (
            "/webhooks/waits",
            '["hook_routes:inside_delivery"]',
            "/webhooks/waits/extra collides with the external-wait delivery route under "
            "/webhooks/waits",
        ),
    ],
)
def test_serve_refuses_a_delivery_route_that_collides(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    launched: dict[str, Any],
    delivery_path: str,
    routers: str | None,
    message: str,
) -> None:
    serve = "" if routers is None else f"routers = {routers}\n"
    project = _hosted_project(tmp_path, delivery_path, serve)
    monkeypatch.chdir(project)

    assert main(["serve", "--dev"]) != 0

    assert "server" not in launched
    assert message in capsys.readouterr().err


@pytest.mark.parametrize("hosted_first", [True, False])
def test_a_maintained_service_serves_hosted_webhooks_it_mounted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    launched: dict[str, Any],
    hosted_first: bool,
) -> None:
    from fastapi import FastAPI

    import cayu.cli.serve as serve_cli

    project = _hosted_project(tmp_path, "/webhooks/waits")
    (project / "pyproject.toml").write_text(
        '[tool.cayu]\nfactory = "serve_project:build_app"\n'
        'service_factory = "serve_project:build_service"\n',
        encoding="utf-8",
    )
    monkeypatch.chdir(project)
    sys.path.insert(0, str(project))
    try:
        import serve_project
    finally:
        sys.path.remove(str(project))
    hosted = serve_project.build_app()
    if hosted_first:
        from tests.server.test_external_wait_delivery_route import _service

        service = _service(hosted, tmp_path)
    else:
        # Hosting after assembly: the service never mounted the route.
        service = SimpleNamespace(cayu_app=hosted, asgi_app=FastAPI())
    monkeypatch.setattr(serve_cli, "build_project_service", lambda *args, **kwargs: service)

    if hosted_first:
        assert main(["serve", "--dev"]) == 0
        client = TestClient(launched["server"])
        assert client.post("/webhooks/waits/renderer", json={}).status_code == 401
    else:
        assert main(["serve", "--dev"]) != 0
        assert "server" not in launched
        assert "call app.host_external_waits(...) before create_agent_service" in (
            capsys.readouterr().err
        )

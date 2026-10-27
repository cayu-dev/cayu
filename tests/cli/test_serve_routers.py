"""`[tool.cayu.serve] routers`: a project's own routes next to Cayu's, under `cayu serve`."""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType
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

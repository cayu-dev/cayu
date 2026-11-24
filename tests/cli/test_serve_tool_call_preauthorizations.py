"""`app.host_tool_call_preauthorizations(...)`: the routes under `cayu serve`."""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from cayu import CayuApp
from cayu.approvals import InMemoryToolCallPreauthorizationStore, ToolCallPreauthorizationHosting
from cayu.cli import main

_ROUTES = "/api/tool-call-preauthorizations"

_FACTORY = """from cayu import CayuApp
from cayu.approvals import InMemoryToolCallPreauthorizationStore


async def operators(principal, action, record):
    return principal.subject == "operator"


def build_app():
    app = CayuApp(enable_logging=False)
    app.host_tool_call_preauthorizations(
        InMemoryToolCallPreauthorizationStore(), authorize=operators
    )
    return app
"""

_AUTH = """from fastapi import HTTPException, Request

from cayu.server import AuthContext


async def by_header(request: Request) -> AuthContext:
    subject = request.headers.get("x-test-subject")
    if not subject:
        raise HTTPException(status_code=401, detail="unauthenticated")
    return AuthContext(subject=subject)
"""


@pytest.fixture
def launched(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict[str, Any]:
    project = tmp_path / "project"
    project.mkdir()
    (project / "pyproject.toml").write_text(
        '[tool.cayu]\nfactory = "preauth_project:build_app"\n\n'
        '[tool.cayu.serve]\nauth = "preauth_auth:by_header"\n',
        encoding="utf-8",
    )
    (project / "preauth_project.py").write_text(_FACTORY, encoding="utf-8")
    (project / "preauth_auth.py").write_text(_AUTH, encoding="utf-8")
    for module in ("preauth_project", "preauth_auth"):
        sys.modules.pop(module, None)
    monkeypatch.chdir(project)
    launched: dict[str, Any] = {}
    uvicorn = ModuleType("uvicorn")

    def run(server: Any, *, host: str, port: int) -> None:
        launched.update(server=server)

    uvicorn.run = run  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "uvicorn", uvicorn)
    return launched


async def _anyone(principal: Any, action: str, record: Any) -> bool:
    return True


def test_hosting_is_once_and_checks_its_parts() -> None:
    app = CayuApp(enable_logging=False)
    assert app.tool_call_preauthorization_hosting is None
    with pytest.raises(TypeError, match="ToolCallPreauthorizationStore"):
        app.host_tool_call_preauthorizations(object(), authorize=_anyone)  # type: ignore[arg-type]
    store = InMemoryToolCallPreauthorizationStore()
    hosting = app.host_tool_call_preauthorizations(store, authorize=_anyone)
    assert isinstance(hosting, ToolCallPreauthorizationHosting)
    assert app.tool_call_preauthorization_hosting is hosting
    with pytest.raises(ValueError, match="already hosts"):
        app.host_tool_call_preauthorizations(store, authorize=_anyone)


def test_serve_mounts_the_routes_behind_its_auth(launched: dict[str, Any]) -> None:
    assert main(["serve", "--host", "127.0.0.1"]) == 0
    client = TestClient(launched["server"])
    assert client.get(_ROUTES, params={"task_id": "t"}).status_code == 401
    body = {"task_id": "t", "tool_name": "send", "arguments": {}}
    issued = client.post(_ROUTES, json=body, headers={"x-test-subject": "operator"})
    assert issued.status_code == 201
    assert issued.json()["preauthorization"]["authorized_by"]["subject"] == "operator"


def test_the_open_development_server_serves_none(launched: dict[str, Any]) -> None:
    assert main(["serve", "--dev"]) == 0
    client = TestClient(launched["server"])
    assert client.get(_ROUTES, params={"task_id": "t"}).status_code == 404

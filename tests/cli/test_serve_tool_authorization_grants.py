"""`app.host_tool_authorization_grants(...)`: the grant routes under `cayu serve`."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("sse_starlette")

from fastapi.testclient import TestClient

from cayu import CayuApp
from cayu.approvals import (
    AllowedActorsGrantAuthority,
    InMemoryToolAuthorizationGrantStore,
    ToolAuthorizationGrantHosting,
    ToolAuthorizationGrantService,
)
from cayu.cli import main

_GRANTS = "/api/tool-authorization-grants"

_FACTORY = """from cayu import CayuApp
from cayu.approvals import (
    AllowedActorsGrantAuthority,
    InMemoryToolAuthorizationGrantStore,
    ToolAuthorizationGrantService,
)


async def operator_reads(principal, grant):
    return principal.subject == "operator"


def build_app():
    app = CayuApp(enable_logging=False)
    service = ToolAuthorizationGrantService(
        app,
        store=InMemoryToolAuthorizationGrantStore(),
        authority=AllowedActorsGrantAuthority(creators=["operator"]),
    )
    app.host_tool_authorization_grants(service, authorize_read=operator_reads)
    return app


def build_plain_app():
    return CayuApp(enable_logging=False)
"""

_AUTH = """from fastapi import HTTPException, Request

from cayu.server import AuthContext


async def by_header(request: Request) -> AuthContext:
    subject = request.headers.get("x-test-subject")
    if not subject:
        raise HTTPException(status_code=401, detail="unauthenticated")
    return AuthContext(subject=subject)
"""


def _project(tmp_path: Path, *, factory: str = "build_app") -> Path:
    project = tmp_path / "project"
    project.mkdir()
    (project / "pyproject.toml").write_text(
        f'[tool.cayu]\nfactory = "grants_project:{factory}"\n\n'
        '[tool.cayu.serve]\nauth = "grants_auth:by_header"\n',
        encoding="utf-8",
    )
    (project / "grants_project.py").write_text(_FACTORY, encoding="utf-8")
    (project / "grants_auth.py").write_text(_AUTH, encoding="utf-8")
    for module in ("grants_project", "grants_auth"):
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


def _service(app: CayuApp) -> ToolAuthorizationGrantService:
    return ToolAuthorizationGrantService(
        app,
        store=InMemoryToolAuthorizationGrantStore(),
        authority=AllowedActorsGrantAuthority(creators=["operator"]),
    )


async def _anyone_reads(principal: Any, grant: Any) -> bool:
    return True


def test_hosting_names_the_applications_own_grant_service_once() -> None:
    app = CayuApp(enable_logging=False)
    assert app.tool_authorization_grant_hosting is None
    service = _service(app)

    hosting = app.host_tool_authorization_grants(service, authorize_read=_anyone_reads)

    assert isinstance(hosting, ToolAuthorizationGrantHosting)
    assert app.tool_authorization_grant_hosting is hosting
    assert hosting.service is service and hosting.authorize_read is _anyone_reads
    with pytest.raises(ValueError, match="already hosts"):
        app.host_tool_authorization_grants(service, authorize_read=_anyone_reads)


def test_hosting_refuses_another_applications_service_and_unusable_parts() -> None:
    app = CayuApp(enable_logging=False)
    other = CayuApp(enable_logging=False)

    with pytest.raises(ValueError, match="built on this application"):
        app.host_tool_authorization_grants(_service(other), authorize_read=_anyone_reads)
    with pytest.raises(TypeError, match="ToolAuthorizationGrantService"):
        app.host_tool_authorization_grants(object(), authorize_read=_anyone_reads)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="authorize_read"):
        app.host_tool_authorization_grants(_service(app), authorize_read=None)  # type: ignore[arg-type]
    # A refused call hosts nothing, so a corrected one still can.
    assert app.tool_authorization_grant_hosting is None


def test_serve_mounts_the_hosted_grant_routes_behind_its_auth(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    launched: dict[str, Any],
) -> None:
    monkeypatch.chdir(_project(tmp_path))

    assert main(["serve", "--host", "127.0.0.1"]) == 0

    client = TestClient(launched["server"])
    assert client.get(_GRANTS).status_code == 401
    listed = client.get(_GRANTS, headers={"x-test-subject": "operator"})
    assert listed.status_code == 200
    assert listed.json() == {"grants": [], "next": None}
    # The other grant routes are there too, and answer for a grant that doesn't exist.
    missing = client.get(f"{_GRANTS}/grant_missing", headers={"x-test-subject": "operator"})
    assert missing.status_code == 404
    revoked = client.post(
        f"{_GRANTS}/grant_missing/revoke", json={}, headers={"x-test-subject": "operator"}
    )
    assert revoked.status_code == 404
    assert f"Tool authorization grants: {_GRANTS}" in capsys.readouterr().out


def test_serve_serves_no_grant_routes_for_an_application_that_hosts_none(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    launched: dict[str, Any],
) -> None:
    monkeypatch.chdir(_project(tmp_path, factory="build_plain_app"))

    assert main(["serve", "--host", "127.0.0.1"]) == 0

    client = TestClient(launched["server"])
    assert client.get(_GRANTS, headers={"x-test-subject": "operator"}).status_code == 404
    assert "Tool authorization grants" not in capsys.readouterr().out


def test_the_open_development_server_serves_no_grant_routes_and_says_so(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    launched: dict[str, Any],
) -> None:
    monkeypatch.chdir(_project(tmp_path))

    # A grant records its authenticated creator, and --dev has no caller to record.
    assert main(["serve", "--dev"]) == 0

    client = TestClient(launched["server"])
    assert client.get(_GRANTS).status_code == 404
    assert client.get("/api/sessions").status_code == 200
    assert "Tool authorization grants: not served with --dev" in capsys.readouterr().out


def test_a_maintained_service_that_would_not_serve_hosted_grants_is_refused(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    (tmp_path / "pyproject.toml").write_text(
        '[tool.cayu]\nfactory = "unused:build_app"\nservice_factory = "unused:build_service"\n',
        encoding="utf-8",
    )
    app = CayuApp(enable_logging=False)
    app.host_tool_authorization_grants(_service(app), authorize_read=_anyone_reads)
    service = SimpleNamespace(
        asgi_app=object(),
        cayu_app=app,
        deployment_kind="local",
        manifest=SimpleNamespace(product_api_path="/api", control_plane_path="/cayu"),
    )
    monkeypatch.setattr("cayu.cli.serve.build_project_service", lambda *args, **kwargs: service)
    monkeypatch.setitem(sys.modules, "uvicorn", ModuleType("uvicorn"))
    monkeypatch.chdir(tmp_path)

    assert main(["serve", "--dev"]) == 1

    assert "hosts tool authorization grants" in capsys.readouterr().err


def _staged_after_an_approval(tmp_path: Path):
    """A grant whose call was approved by a process that stopped before activating it."""

    from tests.runtime.authorization_grant_support import Harness, _command_turn

    from cayu.approvals import ToolAuthorizationGrantRequest, ToolAuthorizationGrantStatus

    harness = Harness(tmp_path, turns=_command_turn("call_1", ["ls"]))

    async def scenario() -> str:
        await harness.run("s1")
        prepared = await harness.service.prepare(
            await harness.approval("s1"), ToolAuthorizationGrantRequest(scope="session")
        )
        _ = [e async for e in harness.app.resolve_tool_approval(prepared.approval_request)]
        staged = await harness.service.get(prepared.grant.grant_id)
        assert staged is not None and staged.status is ToolAuthorizationGrantStatus.STAGED
        return prepared.grant.grant_id

    return harness, asyncio.run(scenario())


def test_a_server_settles_the_staged_grants_a_stopped_process_left(tmp_path: Path) -> None:
    from fastapi import HTTPException, Request

    from cayu.server import AuthContext, ServerConfig, create_server

    harness, grant_id = _staged_after_an_approval(tmp_path)
    harness.app.host_tool_authorization_grants(harness.service, authorize_read=_anyone_reads)

    def auth(request: Request) -> AuthContext:
        if not request.headers.get("x-test-subject"):
            raise HTTPException(status_code=401, detail="unauthenticated")
        return AuthContext(subject="owner", tenant="acme")

    # Hosting is enough: the server need not serve the routes to settle what was left.
    with TestClient(create_server(harness.app, config=ServerConfig.protected(auth))) as client:
        for _ in range(200):
            if asyncio.run(harness.service.get(grant_id)).status.value != "staged":
                break
            client.get("/api/contract", headers={"x-test-subject": "owner"})

    assert asyncio.run(harness.service.get(grant_id)).status.value == "active"


def test_a_worker_settles_staged_grants_before_it_claims(tmp_path: Path) -> None:
    import cayu.cli.worker as worker_cli

    harness, grant_id = _staged_after_an_approval(tmp_path)
    harness.app.host_tool_authorization_grants(harness.service, authorize_read=_anyone_reads)
    seen: list[str] = []

    async def entrypoint(app: CayuApp, stop: Any) -> None:
        seen.append((await harness.service.get(grant_id)).status.value)

    outcome = asyncio.run(
        worker_cli._invoke_worker(entrypoint, harness.app, name="default", shutdown_grace_seconds=5)
    )

    assert outcome == 0
    assert seen == ["active"]


def test_a_failed_startup_reconciliation_never_stops_the_process(
    caplog: pytest.LogCaptureFixture,
) -> None:
    from cayu.approvals.authorization_grant_hosting import (
        reconcile_hosted_tool_authorization_grants,
    )

    plain = CayuApp(enable_logging=False)
    assert asyncio.run(reconcile_hosted_tool_authorization_grants(plain)) is None

    class Unreachable(InMemoryToolAuthorizationGrantStore):
        async def list(self, query: Any = None) -> Any:
            raise ConnectionError("postgres://user:secret@db/grants is down")

    app = CayuApp(enable_logging=False)
    service = ToolAuthorizationGrantService(
        app, store=Unreachable(), authority=AllowedActorsGrantAuthority(creators=["operator"])
    )
    app.host_tool_authorization_grants(service, authorize_read=_anyone_reads)

    assert asyncio.run(reconcile_hosted_tool_authorization_grants(app)) is None
    assert "ConnectionError" in caplog.text
    # The store's own message may name its address and credentials.
    assert "secret" not in caplog.text


@pytest.mark.parametrize("timeout_s", [None, 30.0])
def test_a_stores_own_timeout_is_logged_like_any_other_failure(
    caplog: pytest.LogCaptureFixture, timeout_s: float | None
) -> None:
    from cayu.approvals.authorization_grant_hosting import (
        reconcile_hosted_tool_authorization_grants,
    )

    class Slow(InMemoryToolAuthorizationGrantStore):
        async def list(self, query: Any = None) -> Any:
            raise TimeoutError("statement timeout")

    app = CayuApp(enable_logging=False)
    service = ToolAuthorizationGrantService(
        app, store=Slow(), authority=AllowedActorsGrantAuthority(creators=["operator"])
    )
    app.host_tool_authorization_grants(service, authorize_read=_anyone_reads)

    assert asyncio.run(reconcile_hosted_tool_authorization_grants(app, timeout_s=timeout_s)) is None
    [record] = caplog.records
    # The warning itself formats, whatever bound the pass ran under.
    assert "TimeoutError" in record.getMessage()

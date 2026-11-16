"""`[tool.cayu.serve] routers`: a project's own routes next to Cayu's, under `cayu serve`."""

from __future__ import annotations

import json
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


def protected(app, *, development, auth):
    from fastapi import Depends

    router = APIRouter()

    @router.get("/app/whoami")
    def whoami(context=Depends(auth)):
        return {"subject": context.subject}

    return router


def operator_routes(app, *, development, auth):
    from fastapi import Depends

    from cayu.server.operator_sign_in import operator_person_dependency

    person = operator_person_dependency(auth, development=development)
    router = APIRouter()

    @router.get("/app/me")
    async def me(signed_in=Depends(person)):
        return {"subject": signed_in.subject, "role": signed_in.role}

    return router


def any_options(app, **options):
    router = APIRouter()

    @router.get("/app/options")
    def listed():
        return {"options": sorted(options), "auth": options.get("auth") is not None}

    return router
"""

_AUTH = """from fastapi import Request

from cayu.server import AuthContext


async def operator(request: Request) -> AuthContext:
    return AuthContext(subject="operator")


async def other(request):
    # Unannotated and returning a mapping: Cayu's routes accept both, and so must
    # the dependency a project router gets.
    if request.headers.get("x-other") != "yes":
        from fastapi import HTTPException

        raise HTTPException(status_code=401, detail="other auth refused")
    return {"subject": "other"}
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


def test_a_router_that_takes_auth_gets_the_one_serve_resolved(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    launched: dict[str, Any],
) -> None:
    project = _project(
        tmp_path,
        'routers = ["project_routes:protected", "project_routes:build_router"]\n'
        'auth = "project_auth:operator"\n',
    )
    monkeypatch.chdir(project)

    # --auth overrides the file's target for the project's routes as for Cayu's.
    assert main(["serve", "--host", "127.0.0.1", "--auth", "project_auth:other"]) == 0

    client = TestClient(launched["server"])
    refused = client.get("/app/whoami")
    assert refused.status_code == 401 and refused.json()["detail"] == "other auth refused"
    assert client.get("/app/whoami", headers={"x-other": "yes"}).json() == {"subject": "other"}
    assert client.get("/api/sessions").status_code == 401
    # A factory without `auth` is still called with just the app and `development=`.
    assert client.get("/app/hello").json()["development"] is False


@pytest.mark.parametrize("development", [True, False])
def test_a_router_taking_any_keyword_gets_auth_too(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    launched: dict[str, Any],
    development: bool,
) -> None:
    project = _project(
        tmp_path, 'routers = ["project_routes:any_options"]\nauth = "project_auth:operator"\n'
    )
    monkeypatch.chdir(project)

    assert main(["serve", "--dev"] if development else ["serve", "--host", "127.0.0.1"]) == 0

    client = TestClient(launched["server"])
    # Under --dev there is no auth to pass: Cayu's routes are open, and so is `auth`.
    assert client.get("/app/options").json() == {
        "options": ["auth", "development"],
        "auth": not development,
    }


_PROXY_AUTH = """from cayu.server import BasicAuth, ProxyIdentityAuth
from cayu.server.development_identity import registered_development_identity

_proxy = registered_development_identity()
AUTH = ProxyIdentityAuth(
    issuer=_proxy.issuer,
    audience=_proxy.audience,
    signing_keys=_proxy.signing_keys(),
    fallback=BasicAuth(username="operator", password="secret"),
)
"""


@pytest.mark.parametrize("development", [True, False])
def test_the_operator_sign_in_part_signs_in_cloud_members_under_cayu_serve(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    launched: dict[str, Any],
    development: bool,
) -> None:
    pytest.importorskip("jwt")
    import getpass

    from cayu.server.development_identity import DevelopmentIdentityProxy

    proxy = DevelopmentIdentityProxy()
    proxy.register()
    try:
        project = _project(
            tmp_path,
            'routers = ["project_routes:operator_routes"]\nauth = "project_proxy_auth:AUTH"\n',
        )
        (project / "project_proxy_auth.py").write_text(_PROXY_AUTH, encoding="utf-8")
        monkeypatch.delitem(sys.modules, "project_proxy_auth", raising=False)
        monkeypatch.chdir(project)

        assert main(["serve", "--dev"] if development else ["serve", "--host", "127.0.0.1"]) == 0

        client = TestClient(
            launched["server"], base_url="http://127.0.0.1", client=("127.0.0.1", 50000)
        )
        if development:
            # `auth` is None under --dev: the person running it, as admin.
            assert client.get("/app/me").json() == {
                "subject": f"dev:{getpass.getuser()}",
                "role": "admin",
            }
        else:
            # Cayu's route rule would refuse a member outside /api; the part asks only who.
            member = proxy.headers("mel", role="member")
            assert client.get("/app/me", headers=member).json() == {
                "subject": "mel",
                "role": "member",
            }
            assert client.get("/app/me").status_code == 401
    finally:
        proxy.unregister()


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


def _web_build(project: Path, directory: str = "web/static") -> Path:
    static = project / directory
    (static / "assets").mkdir(parents=True)
    (static / "index.html").write_text("<!doctype html><title>app</title>", encoding="utf-8")
    (static / "assets" / "index-abc.js").write_text("1", encoding="utf-8")
    return static


def test_serve_serves_the_projects_web_app_at_the_root_behind_every_route(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    launched: dict[str, Any],
) -> None:
    project = _project(tmp_path, 'routers = ["project_routes:build_router"]\nui = "web/static"\n')
    _web_build(project)
    monkeypatch.chdir(project)

    assert main(["serve", "--dev"]) == 0

    client = TestClient(launched["server"])
    assert client.get("/conversations/some-id").text == "<!doctype html><title>app</title>"
    assert "immutable" in client.get("/assets/index-abc.js").headers["cache-control"]
    assert client.get("/assets/missing.js").status_code == 404
    # The project's routes and Cayu's still answer, and their unknown paths are 404s.
    assert client.get("/app/hello").json()["app"] == "CayuApp"
    assert client.get("/app/missing").status_code == 404
    assert client.get("/api/missing").status_code == 404
    assert "Web app: http://127.0.0.1:8000/" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("ui", "build", "message"),
    [
        ('"web/static"', False, "web/static doesn't exist"),
        ('""', False, "ui must be the path of a directory"),
        ("3", False, "ui must be the path of a directory"),
        ('"."', True, "has no index.html"),
    ],
)
def test_serve_refuses_a_web_app_it_cant_serve(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    launched: dict[str, Any],
    ui: str,
    build: bool,
    message: str,
) -> None:
    project = _project(tmp_path, f"ui = {ui}\n")
    if build:
        _web_build(project)
    monkeypatch.chdir(project)

    assert main(["serve", "--dev"]) != 0

    assert "server" not in launched
    assert message in capsys.readouterr().err


def test_serve_refuses_a_web_app_that_shadows_the_projects_routes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    launched: dict[str, Any],
) -> None:
    project = _project(tmp_path, 'routers = ["project_routes:build_router"]\nui = "web/static"\n')
    (_web_build(project) / "app").mkdir()
    monkeypatch.chdir(project)

    assert main(["serve", "--dev"]) != 0

    assert "has /app, which the server's own routes answer" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("ui", "layout", "code"),
    [
        ('"web/static"', "built", None),
        ('"web/static"', "none", "SERVE_UI_MISSING"),
        ('"web/static"', "no page", "SERVE_UI_MISSING"),
        ('"web/static"', "api folder", "SERVE_UI_SHADOWS_ROUTES"),
        ("[]", "none", "SERVE_UI_INVALID"),
    ],
)
def test_cayu_check_checks_the_web_app(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    ui: str,
    layout: str,
    code: str | None,
) -> None:
    project = _project(tmp_path, f"ui = {ui}\n")
    if layout != "none":
        static = _web_build(project)
        if layout == "no page":
            (static / "index.html").unlink()
        if layout == "api folder":
            (static / "api").mkdir()
    monkeypatch.chdir(project)

    main(["check", "--json"])

    found = {item["code"] for item in json.loads(capsys.readouterr().out)["diagnostics"]}
    ui_codes = {item for item in found if item.startswith("SERVE_UI_")}
    assert ui_codes == (set() if code is None else {code})


# The pattern real Agents use for their own routes: ask the auth only who is calling
# (`authenticate`), then apply the Agent's own role rule.
_OWN_ROLE_RULE_ROUTES = """import inspect

from fastapi import APIRouter, Depends, HTTPException, Request

from cayu.server import AuthContext


def build_router(app, *, development, auth):
    authenticate = getattr(auth, "authenticate", None)
    who = authenticate if callable(authenticate) else auth

    async def person(request: Request):
        found = who(request)
        if inspect.isawaitable(found):
            found = await found
        context = (
            found if isinstance(found, AuthContext) else AuthContext.model_validate(dict(found))
        )
        role = context.claims.get("role", "admin")
        if role not in ("admin", "developer", "member"):
            raise HTTPException(status_code=403)
        return {"subject": context.subject, "role": role}

    router = APIRouter()

    @router.get("/app/whoami")
    async def whoami(signed_in=Depends(person)):
        return signed_in

    return router
"""


def test_a_router_with_its_own_role_rule_signs_in_cloud_members(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    launched: dict[str, Any],
) -> None:
    pytest.importorskip("jwt")
    from cayu.server.development_identity import DevelopmentIdentityProxy

    proxy = DevelopmentIdentityProxy()
    proxy.register()
    try:
        project = _project(
            tmp_path,
            'routers = ["own_role_rule_routes:build_router"]\nauth = "cloud_sign_in_auth:AUTH"\n',
        )
        (project / "own_role_rule_routes.py").write_text(_OWN_ROLE_RULE_ROUTES, encoding="utf-8")
        (project / "cloud_sign_in_auth.py").write_text(_PROXY_AUTH, encoding="utf-8")
        for module in ("own_role_rule_routes", "cloud_sign_in_auth"):
            monkeypatch.delitem(sys.modules, module, raising=False)
        monkeypatch.chdir(project)

        assert main(["serve", "--host", "127.0.0.1"]) == 0

        client = TestClient(launched["server"], base_url="http://127.0.0.1")
        # Cayu's route rule admits a member only on the Developer dashboard's API. The
        # auth `cayu serve` passes keeps `authenticate`, so the router's own rule decides.
        for subject, role in (("mel", "member"), ("dee", "developer"), ("ada", "admin")):
            signed_in = client.get("/app/whoami", headers=proxy.headers(subject, role=role))
            assert signed_in.json() == {"subject": subject, "role": role}
        # It still authenticates: a request with no identity and no operator login is refused.
        assert client.get("/app/whoami").status_code == 401
    finally:
        proxy.unregister()

"""`cayu serve --identity-proxy`: the project's real auth, signed in as a chosen person."""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("jwt")

from fastapi.testclient import TestClient

from cayu._operator_credentials import OPERATOR_PASSWORD_VARIABLE, OPERATOR_USERNAME_VARIABLE
from cayu.cli import main
from cayu.cli._cloud_deploy_check import run_cloud_deploy_check
from cayu.cli._serve_reload import run_reload
from cayu.cli.check import serve_identity_proxy_findings
from cayu.cli.serve import _PROXY_IDENTITY_VARIABLES, parse_serve_arguments
from cayu.server import proxy_identity
from cayu.server.development_identity import (
    DEVELOPMENT_PERSON_HEADER,
    registered_development_identity,
)

LOCAL = "http://127.0.0.1:8000"
_ONLY_OPERATORS = "cayu.server.environment_auth:OPERATOR_ONLY_AUTH"

_FACTORY = """from cayu import CayuApp


def build_app():
    return CayuApp(enable_logging=False)
"""

_ROUTES = """from fastapi import APIRouter, Depends

from cayu.server import ProxyIdentityAuth


def whoami(app, *, development):
    # As the b2b preset's sign-in does: the project's own from_environment() call.
    product = ProxyIdentityAuth.from_environment().product_dependency()
    router = APIRouter()

    @router.get("/app/whoami")
    def me(principal=Depends(product)):
        return {"subject": principal.subject_id, "development": development}

    @router.get("/app/profile")
    def profile(principal=Depends(product)):
        return {"name": principal.name, "email": principal.email}

    @router.post("/app/write")
    def write(principal=Depends(product)):
        return {"subject": principal.subject_id}

    return router
"""

_PEOPLE = """
[tool.cayu.serve.identity_proxy]
default = "alice"

[[tool.cayu.serve.identity_proxy.people]]
subject = "alice"
org = "acme"
role = "member"
name = "Alice Example"
email = "alice@example.com"

[[tool.cayu.serve.identity_proxy.people]]
subject = "dana"
org = "acme"
role = "developer"
"""


def _project(tmp_path: Path, serve: str = f'auth = "{_ONLY_OPERATORS}"\n') -> Path:
    project = tmp_path / "project"
    project.mkdir()
    (project / "pyproject.toml").write_text(
        '[tool.cayu]\nfactory = "identity_project:build_app"\n\n'
        f'[tool.cayu.serve]\nrouters = ["identity_routes:whoami"]\n{serve}',
        encoding="utf-8",
    )
    (project / "identity_project.py").write_text(_FACTORY, encoding="utf-8")
    (project / "identity_routes.py").write_text(_ROUTES, encoding="utf-8")
    return project


_PROJECT_MODULES = (
    "identity_project",
    "identity_routes",
    "identity_evals",
    "cayu.server.environment_auth",
)
# An eval target whose module builds environment_auth's ready-made targets on import.
_EVALS = """from cayu.server.environment_auth import OPERATOR_ONLY_AUTH  # noqa: F401


def build_eval():
    raise RuntimeError("not served in this test")
"""


def _project_with_eval_target(tmp_path: Path) -> Path:
    project = _project(tmp_path)
    (project / "pyproject.toml").write_text(
        (project / "pyproject.toml").read_text(encoding="utf-8")
        + '\n[tool.cayu.evals]\ntargets = ["identity_evals:build_eval"]\n',
        encoding="utf-8",
    )
    (project / "identity_evals.py").write_text(_EVALS, encoding="utf-8")
    return project


@pytest.fixture(autouse=True)
def _clean_environment(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    names = (*_PROXY_IDENTITY_VARIABLES, OPERATOR_USERNAME_VARIABLE, OPERATOR_PASSWORD_VARIABLE)
    saved = {name: os.environ.pop(name) for name in names if name in os.environ}
    for module in _PROJECT_MODULES:
        sys.modules.pop(module, None)
    yield
    for name in names:
        os.environ.pop(name, None)
    os.environ.update(saved)
    for module in _PROJECT_MODULES:
        sys.modules.pop(module, None)
    proxy = registered_development_identity()
    if proxy is not None:
        proxy.unregister()


@pytest.fixture
def launched(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    launched: dict[str, Any] = {}
    uvicorn = ModuleType("uvicorn")

    def run(server: Any, *, host: str, port: int, **options: Any) -> None:
        launched.update(server=server, host=host, port=port)

    uvicorn.run = run  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "uvicorn", uvicorn)
    return launched


def test_variable_names_match_proxy_identity() -> None:
    assert _PROXY_IDENTITY_VARIABLES == (
        proxy_identity.PROXY_IDENTITY_ISSUER_VARIABLE,
        proxy_identity.PROXY_IDENTITY_AUDIENCE_VARIABLE,
        proxy_identity.PROXY_IDENTITY_JWKS_URL_VARIABLE,
        proxy_identity.PROXY_IDENTITY_HEADER_VARIABLE,
    )


def test_serve_takes_the_deployed_sign_in_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    launched: dict[str, Any],
) -> None:
    monkeypatch.chdir(_project(tmp_path, f'auth = "{_ONLY_OPERATORS}"\n{_PEOPLE}'))

    assert main(["serve", "--identity-proxy"]) == 0

    out = capsys.readouterr().out
    # No operator login was set, so this process got a random one and said so.
    assert os.environ[OPERATOR_USERNAME_VARIABLE] == "operator"
    password = os.environ[OPERATOR_PASSWORD_VARIABLE]
    assert f"operator / {password}" in out
    assert f"{LOCAL}/.cayu/sign-in; signed in as alice (member in acme)" in out
    client = TestClient(launched["server"], base_url=LOCAL)

    # The project's routers are called as deployed, and its own from_environment()
    # trusts the proxy.
    assert client.get("/app/whoami").json() == {"subject": "alice", "development": False}
    # OPERATOR_ONLY_AUTH: a member is refused the Developer dashboard's API, a developer isn't.
    refused = client.get("/api/inbox")
    assert refused.status_code == 403 and refused.json()["detail"]["code"] == "role_forbidden"
    assert client.get("/api/inbox", headers={DEVELOPMENT_PERSON_HEADER: "dana"}).status_code == 200
    # The operator login still works beside the proxy, on a request with nobody signed in.
    client.cookies.set("cayu-dev-person", "-")
    assert client.get("/api/inbox", auth=("operator", password)).status_code == 200


def test_writes_follow_the_same_origin_rule(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, launched: dict[str, Any]
) -> None:
    monkeypatch.chdir(_project(tmp_path))

    assert main(["serve", "--identity-proxy", "--as", "alice", "--org", "acme"]) == 0

    client = TestClient(launched["server"], base_url=LOCAL)
    assert client.post("/app/write", headers={"origin": LOCAL}).json() == {"subject": "alice"}
    assert client.post("/app/write", headers={"origin": "http://evil.example"}).status_code == 403


def test_whoami_shows_the_name_the_proxy_signs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, launched: dict[str, Any]
) -> None:
    monkeypatch.chdir(_project(tmp_path))

    assert main(["serve", "--identity-proxy", "--as", "alice", "--name", "Alice Example"]) == 0

    client = TestClient(launched["server"], base_url=LOCAL)
    assert client.get("/app/profile").json() == {"name": "Alice Example", "email": None}


def test_a_client_sent_identity_and_another_host_get_nobody(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, launched: dict[str, Any]
) -> None:
    monkeypatch.chdir(_project(tmp_path))
    assert main(["serve", "--identity-proxy", "--as", "alice"]) == 0
    proxy = registered_development_identity()
    assert proxy is not None
    forged = proxy.token("mallory", role="admin")

    # A page on another site that rebinds its name to 127.0.0.1 sends its own Host.
    rebound = TestClient(launched["server"], base_url="http://rebound.example.com:8000")
    assert rebound.get("/app/whoami").status_code == 401
    assert rebound.get("/api/inbox", headers={"X-Cayu-Identity": forged}).status_code == 401
    # On loopback, a client's identity header is replaced by the chosen person's.
    local = TestClient(launched["server"], base_url=LOCAL)
    assert local.get("/app/whoami", headers={"X-Cayu-Identity": forged}).json()["subject"] == (
        "alice"
    )


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (["--identity-proxy", "--dev"], "can't be combined with --dev"),
        (["--identity-proxy", "--host", "0.0.0.0"], "non-loopback host"),
        (["--identity-proxy", "--host", "192.168.1.5"], "non-loopback host"),
        (["--as", "alice"], "--as only apply with --identity-proxy"),
        (["--identity-proxy", "--role", "admin"], "describe the --as person"),
        (["--identity-proxy", "--as", " "], "--identity-proxy person"),
    ],
)
def test_serve_refuses_an_unsafe_identity_proxy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    launched: dict[str, Any],
    arguments: list[str],
    message: str,
) -> None:
    monkeypatch.chdir(_project(tmp_path))

    assert main(["serve", *arguments]) != 0

    assert "server" not in launched
    assert message in capsys.readouterr().err
    assert registered_development_identity() is None


def test_serve_refuses_when_a_real_proxy_is_configured(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    launched: dict[str, Any],
) -> None:
    monkeypatch.chdir(_project(tmp_path))
    os.environ["CAYU_PROXY_IDENTITY_ISSUER"] = "https://sign-in.example.com"

    assert main(["serve", "--identity-proxy"]) != 0

    assert "CAYU_PROXY_IDENTITY_ISSUER is already set" in capsys.readouterr().err
    assert "server" not in launched


def test_an_auth_target_that_isnt_proxy_identity_is_reported(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    launched: dict[str, Any],
) -> None:
    monkeypatch.chdir(
        _project(tmp_path, 'auth = "cayu.server.environment_auth:OPERATOR_BASIC_AUTH"\n')
    )
    monkeypatch.setattr(
        "cayu.cli.serve._load_auth", lambda target: lambda request: {"subject": "operator"}
    )

    assert main(["serve", "--identity-proxy"]) == 0

    err = capsys.readouterr().err
    assert "--identity-proxy changes nothing for /api" in err


def test_an_invalid_people_table_stops_serve_and_is_a_check_finding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    launched: dict[str, Any],
) -> None:
    table = '\n[[tool.cayu.serve.identity_proxy.people]]\nsubject = "a"\nrole = "owner"\n'
    project = _project(tmp_path, table)
    monkeypatch.chdir(project)

    assert main(["serve", "--dev"]) != 0
    assert "people[0]: `role` must be one of" in capsys.readouterr().err

    findings = serve_identity_proxy_findings(project)
    assert [item.code for item in findings] == ["SERVE_IDENTITY_PROXY_INVALID"]
    main(["check", "--json"])
    codes = {item["code"] for item in json.loads(capsys.readouterr().out)["diagnostics"]}
    assert "SERVE_IDENTITY_PROXY_INVALID" in codes
    # A deployed `cayu serve` reads the table too, so the deploy checks report it.
    main(["check", "--deploy", "--json"])
    codes = {item["code"] for item in json.loads(capsys.readouterr().out)["diagnostics"]}
    assert "SERVE_IDENTITY_PROXY_INVALID" in codes
    _declare_server_extra(project)
    check = run_cloud_deploy_check(
        project, serves_web=True, web_command="cayu serve --host 0.0.0.0", boot_service=False
    )
    assert "SERVE_IDENTITY_PROXY_INVALID" in {item.code for item in check.blocking}


def test_a_valid_people_table_has_no_check_finding(tmp_path: Path) -> None:
    assert serve_identity_proxy_findings(_project(tmp_path, _PEOPLE)) == ()


def _declare_server_extra(project: Path) -> None:
    (project / "pyproject.toml").write_text(
        '[project]\nname = "identity-agent"\nversion = "0.1.0"\n'
        'dependencies = ["cayu[server]"]\n\n'
        + (project / "pyproject.toml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )


def test_the_deploy_check_refuses_the_identity_proxy(tmp_path: Path) -> None:
    project = _project(tmp_path)
    _declare_server_extra(project)

    check = run_cloud_deploy_check(
        project,
        serves_web=True,
        web_command="cayu serve --identity-proxy --as alice --host 127.0.0.1",
    )

    assert "SERVE_IDENTITY_PROXY_MODE" in check.codes
    finding = next(item for item in check.blocking if item.code == "SERVE_IDENTITY_PROXY_MODE")
    assert finding.path == "cayu-cloud.toml:[web].command"


def test_the_deploy_check_refuses_person_options_without_the_proxy(tmp_path: Path) -> None:
    project = _project(tmp_path)
    _declare_server_extra(project)

    check = run_cloud_deploy_check(
        project, serves_web=True, web_command="cayu serve --as alice --host 0.0.0.0"
    )

    assert [item.code for item in check.blocking] == ["SERVE_COMMAND_INVALID"]
    assert "--as only apply with --identity-proxy" in check.blocking[0].message


def test_an_existing_operator_password_is_never_printed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    launched: dict[str, Any],
) -> None:
    monkeypatch.chdir(_project(tmp_path))
    os.environ[OPERATOR_PASSWORD_VARIABLE] = "s3cret-existing"

    assert main(["serve", "--identity-proxy"]) == 0

    out = capsys.readouterr().out
    assert "s3cret-existing" not in out
    assert "Developer dashboard username for this process: operator" in out


def test_a_maintained_service_serves_behind_the_proxy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, launched: dict[str, Any]
) -> None:
    from fastapi import APIRouter, Depends, FastAPI

    import cayu.cli.serve as serve_cli
    from cayu.server import ProxyIdentityAuth
    from cayu.server.service import _materialize_asgi_middleware_stacks

    project = _project(tmp_path, "")
    (project / "pyproject.toml").write_text(
        '[tool.cayu]\nfactory = "identity_project:build_app"\n'
        'service_factory = "identity_project:build_service"\n',
        encoding="utf-8",
    )
    monkeypatch.chdir(project)

    def build_service(*args: Any, **kwargs: Any) -> SimpleNamespace:
        # As create_agent_service(...) returns it: its middleware stack already built.
        product = ProxyIdentityAuth.from_environment().product_dependency()
        router = APIRouter()

        @router.get("/app/whoami")
        def me(principal=Depends(product)):  # noqa: B008
            return {"subject": principal.subject_id}

        server = FastAPI()
        server.include_router(router)
        _materialize_asgi_middleware_stacks(server, seen=set())
        return SimpleNamespace(
            cayu_app=SimpleNamespace(_external_wait_hosting=None),
            asgi_app=server,
            deployment_kind=None,
            manifest=SimpleNamespace(product_api_path="/product", control_plane_path="/api"),
        )

    monkeypatch.setattr(serve_cli, "build_project_service", build_service)
    monkeypatch.setattr("cayu.runtime.checks.check_public_service_deployment", lambda manifest: ())

    assert main(["serve", "--identity-proxy", "--as", "alice"]) == 0

    client = TestClient(launched["server"], base_url=LOCAL)
    assert client.get("/app/whoami").json() == {"subject": "alice"}


def test_the_proxy_starts_before_eval_targets_import_project_code(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    launched: dict[str, Any],
) -> None:
    monkeypatch.chdir(_project_with_eval_target(tmp_path))

    assert main(["serve", "--identity-proxy", "--as", "alice", "--role", "developer"]) == 0

    assert "isn't a ProxyIdentityAuth" not in capsys.readouterr().err
    client = TestClient(launched["server"], base_url=LOCAL)
    assert client.get("/api/inbox").status_code == 200


def test_a_reload_worker_starts_the_proxy_before_eval_targets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from cayu.cli import _serve_reload as reload_cli

    project = _project_with_eval_target(tmp_path)
    # The parent sets the login before it spawns workers.
    os.environ[OPERATOR_USERNAME_VARIABLE] = "operator"
    os.environ[OPERATOR_PASSWORD_VARIABLE] = "worker-password"
    arguments = ["--identity-proxy", "--as", "alice", "--role", "developer"]
    monkeypatch.setenv(
        "_CAYU_SERVE_RELOAD", json.dumps({"root": str(project), "arguments": arguments})
    )

    worker = reload_cli.create_reload_app()
    try:
        assert "isn't a ProxyIdentityAuth" not in capsys.readouterr().err
        with TestClient(worker, base_url=LOCAL) as client:
            assert client.get("/api/inbox").status_code == 200
    finally:
        worker.close_sync()


def test_reload_workers_get_the_identity_proxy_options(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    class Stop(Exception):
        pass

    class Uvicorn:
        config = None

        @staticmethod
        def Config(*args: Any, **kwargs: Any) -> Any:
            captured["snapshot"] = json.loads(os.environ["_CAYU_SERVE_RELOAD"])
            raise Stop

    args = parse_serve_arguments(
        ["--identity-proxy", "--as", "alice", "--role", "admin", "--name", "Alice"]
    )
    with pytest.raises(Stop):
        run_reload(Uvicorn, args, Path("/project"))

    arguments = captured["snapshot"]["arguments"]
    assert arguments[arguments.index("--identity-proxy") :] == [
        "--identity-proxy",
        "--as",
        "alice",
        "--role",
        "admin",
        "--name",
        "Alice",
    ]
    # Each worker parses them as the parent did; the proxy's key is made in the worker.
    worker = parse_serve_arguments(arguments)
    assert (worker.identity_proxy, worker.identity_as, worker.identity_role) == (
        True,
        "alice",
        "admin",
    )

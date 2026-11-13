"""`cayu serve` with a member inbox: the resource router's scoped inbox and `inbox_access`."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("jwt")

from fastapi.testclient import TestClient
from tests.cli.test_serve_identity_proxy import (  # noqa: F401 - fixtures
    _clean_environment,
    launched,
)

from cayu.cli import main
from cayu.cli.check import serve_inbox_access_findings
from cayu.server.development_identity import DEVELOPMENT_PERSON_HEADER

LOCAL = "http://127.0.0.1:8000"
_ONLY_OPERATORS = "cayu.server.environment_auth:OPERATOR_ONLY_AUTH"
_RULE = "cayu.server.inbox:resource_access_rule"

_FACTORY = """from tests.server import test_scoped_inbox


def build_app():
    app = test_scoped_inbox._app()[0]
    # Kept, so a test can pause a run in the app being served.
    test_scoped_inbox.SERVED_APPS.append(app)
    return app
"""

_ROUTES = """from fastapi import HTTPException, Request

from cayu.server import ProductPrincipal, ProxyIdentityAuth
from cayu.server.resource_access import create_resource_router

ORGS = {"alice": "acme", "bob": "globex"}


def members(app, *, development):
    if development:
        # A demo sign-in, as b2b's under --dev: the app's own sign-in names the person.
        async def authenticate(request: Request) -> ProductPrincipal:
            person = request.headers.get("x-demo-person", "")
            if person not in ORGS:
                raise HTTPException(status_code=401, detail="Sign-in required.")
            return ProductPrincipal(tenant_id=ORGS[person], subject_id=person)

    else:
        authenticate = ProxyIdentityAuth.from_environment().product_dependency()
    return create_resource_router(
        app,
        authenticate=authenticate,
        # Each person belongs to one organization, so their subject alone names them.
        principal_subject=lambda principal: principal.subject_id,
        prefix="/app/resources",
        inbox=True,
    )


def not_a_rule(app):
    return None
"""


def _project(tmp_path: Path, serve: str) -> Path:
    project = tmp_path / "project"
    project.mkdir(parents=True)
    (project / "pyproject.toml").write_text(
        '[tool.cayu]\nfactory = "inbox_project:build_app"\n\n'
        f'[tool.cayu.serve]\nrouters = ["inbox_routes:members"]\n{serve}',
        encoding="utf-8",
    )
    (project / "inbox_project.py").write_text(_FACTORY, encoding="utf-8")
    (project / "inbox_routes.py").write_text(_ROUTES, encoding="utf-8")
    for module in ("inbox_project", "inbox_routes"):
        sys.modules.pop(module, None)
    return project


def _pause(client: TestClient) -> None:
    from tests.server.test_scoped_inbox import SERVED_APPS

    from cayu import Message, RunRequest

    app = SERVED_APPS[-1]

    async def run() -> None:
        alice = await app.access("alice")
        async for _event in alice.run(
            RunRequest(
                agent_name="approver",
                session_id="alice-1",
                messages=[Message.text("user", "go")],
                labels={"org": "acme"},
            )
        ):
            pass

    assert client.portal is not None
    client.portal.call(run)


@pytest.fixture(autouse=True)
def _repository_on_path(monkeypatch: pytest.MonkeyPatch) -> None:
    # The project's factory imports the test app from this repository.
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2]))


def test_alice_decides_under_dev_with_the_apps_own_sign_in(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    launched: dict[str, Any],  # noqa: F811
) -> None:
    monkeypatch.chdir(_project(tmp_path, f'inbox_access = "{_RULE}"\n'))

    assert main(["serve", "--dev"]) == 0

    assert "inbox_access" in capsys.readouterr().err
    with TestClient(launched["server"], base_url=LOCAL) as client:
        _pause(client)
        assert (
            client.get("/app/resources/inbox", headers={"x-demo-person": "bob"}).json()["items"]
            == []
        )
        [item] = client.get("/app/resources/inbox", headers={"x-demo-person": "alice"}).json()[
            "items"
        ]
        settled = client.post(
            f"/app/resources/inbox/{item['id']}/settle",
            json={"decision": "approve"},
            headers={"x-demo-person": "alice"},
        )
        assert settled.status_code == 200, settled.text
        # The decider is the person the app signed in, not null as through an open /api.
        decided_by = settled.json()["item"]["settlement"]["decided_by"]
        assert (decided_by["subject"], decided_by["tenant"]) == ("alice", "acme")


def test_alice_decides_under_the_identity_proxy_and_the_rule_scopes_api(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    launched: dict[str, Any],  # noqa: F811
) -> None:
    monkeypatch.chdir(_project(tmp_path, f'auth = "{_ONLY_OPERATORS}"\ninbox_access = "{_RULE}"\n'))

    assert main(["serve", "--identity-proxy", "--as", "alice", "--org", "acme"]) == 0

    with TestClient(launched["server"], base_url=LOCAL) as client:
        _pause(client)
        # A member never touches /api.
        assert client.get("/api/inbox").status_code == 403
        # The rule applies on /api too: a developer sees only what they may continue.
        as_developer = {DEVELOPMENT_PERSON_HEADER: "sub=bob&org=globex&role=developer"}
        assert client.get("/api/inbox", headers=as_developer).json()["items"] == []
        alice_developer = {DEVELOPMENT_PERSON_HEADER: "sub=alice&org=acme&role=developer"}
        assert len(client.get("/api/inbox", headers=alice_developer).json()["items"]) == 1

        [item] = client.get("/app/resources/inbox").json()["items"]
        settled = client.post(
            f"/app/resources/inbox/{item['id']}/settle",
            json={"decision": "approve"},
            headers={"origin": LOCAL},
        )
        assert settled.status_code == 200, settled.text
        decided_by = settled.json()["item"]["settlement"]["decided_by"]
        assert (decided_by["subject"], decided_by["tenant"]) == ("alice", "acme")


@pytest.mark.parametrize(
    ("target", "message"),
    [
        ("inbox_routes:missing", "inbox_routes"),
        ("inbox_routes:ORGS", "must be callable"),
        ("inbox_routes:not_a_rule", "must return an InboxAccess rule"),
    ],
)
def test_an_unusable_inbox_access_target_stops_serve(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    launched: dict[str, Any],  # noqa: F811
    target: str,
    message: str,
) -> None:
    monkeypatch.chdir(
        _project(tmp_path, f'auth = "{_ONLY_OPERATORS}"\ninbox_access = "{target}"\n')
    )

    assert main(["serve", "--identity-proxy"]) != 0

    assert "server" not in launched
    assert message in capsys.readouterr().err


def test_cayu_check_reports_inbox_access_findings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    unresolvable = _project(tmp_path / "a", 'inbox_access = "inbox_routes:missing"\n')
    codes = [item.code for item in serve_inbox_access_findings(unresolvable)]
    assert codes == ["SERVE_INBOX_ACCESS_UNRESOLVABLE"]

    good = _project(tmp_path / "b", f'auth = "{_ONLY_OPERATORS}"\ninbox_access = "{_RULE}"\n')
    assert serve_inbox_access_findings(good) == ()
    # Auth can come from `cayu serve --auth` or --identity-proxy, so none in pyproject is fine.
    without_auth = _project(tmp_path / "c", f'inbox_access = "{_RULE}"\n')
    assert serve_inbox_access_findings(without_auth) == ()

    monkeypatch.chdir(unresolvable)
    main(["check", "--json"])
    found = {item["code"] for item in json.loads(capsys.readouterr().out)["diagnostics"]}
    assert "SERVE_INBOX_ACCESS_UNRESOLVABLE" in found

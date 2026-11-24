"""The routes and `cayu preauthorizations` that issue, read and revoke preauthorizations."""

from __future__ import annotations

import json
from contextlib import nullcontext
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

pytest.importorskip("fastapi")

from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

import cayu.cli.preauthorizations as preauthorizations_cli
from cayu import CayuApp
from cayu.approvals import InMemoryToolCallPreauthorizationStore
from cayu.approvals.authorization_grants import tool_arguments_digest
from cayu.cli import main
from cayu.server.auth import AuthContext
from cayu.server.tool_call_preauthorizations import ToolCallPreauthorizationServer

NOW = datetime(2026, 10, 1, tzinfo=UTC)
ARGUMENTS = {"email_id": "e-1", "to": ["ap@example.com"]}
ISSUE = {"task_id": "task-1", "tool_name": "send_email", "arguments": ARGUMENTS}


async def _operators_only(principal, action, record) -> bool:
    # Anyone may read; only operators issue, and only the author revokes.
    if action == "issue":
        return principal.subject.startswith("operator")
    if action == "revoke":
        return record.authorized_by.subject == principal.subject
    return principal.subject != "outsider"


def _client(store) -> TestClient:
    server = ToolCallPreauthorizationServer(store, authorize=_operators_only, clock=lambda: NOW)
    api = FastAPI()

    def auth(request: Request) -> AuthContext:
        return AuthContext(subject=request.headers["x-subject"], tenant="acme")

    api.include_router(server.router(auth=auth), prefix="/api")
    return TestClient(api)


def _as(subject: str) -> dict[str, str]:
    return {"x-subject": subject}


def test_issue_names_the_caller_and_stores_only_the_digest() -> None:
    store = InMemoryToolCallPreauthorizationStore()
    client = _client(store)
    answer = client.post("/api/tool-call-preauthorizations", json=ISSUE, headers=_as("operator"))
    assert answer.status_code == 201, answer.text
    view = answer.json()
    record = view["preauthorization"]
    assert view["status"] == "available"
    assert record["authorized_by"] == {
        "subject": "operator",
        "tenant": "acme",
        "source": "http_auth",
    }
    assert record["arguments_digest"] == tool_arguments_digest(ARGUMENTS)
    assert "ap@example.com" not in answer.text
    # A client can't name someone else as the author.
    forged = client.post(
        "/api/tool-call-preauthorizations",
        json={**ISSUE, "authorized_by": {"subject": "boss"}},
        headers=_as("operator"),
    )
    assert forged.status_code == 422
    refused = client.post("/api/tool-call-preauthorizations", json=ISSUE, headers=_as("intern"))
    assert refused.status_code == 403


def test_issue_with_an_id_is_idempotent_and_conflicts_on_other_terms() -> None:
    client = _client(InMemoryToolCallPreauthorizationStore())
    body = {**ISSUE, "preauthorization_id": "pre-1"}
    first = client.post("/api/tool-call-preauthorizations", json=body, headers=_as("operator"))
    again = client.post("/api/tool-call-preauthorizations", json=body, headers=_as("operator"))
    assert first.status_code == again.status_code == 201
    assert first.json() == again.json()
    other = client.post(
        "/api/tool-call-preauthorizations",
        json={**body, "arguments": {"to": ["x@example.com"]}},
        headers=_as("operator"),
    )
    assert other.status_code == 409


def test_list_show_and_revoke_follow_the_access_rule() -> None:
    client = _client(InMemoryToolCallPreauthorizationStore())
    body = {**ISSUE, "preauthorization_id": "pre-1"}
    client.post("/api/tool-call-preauthorizations", json=body, headers=_as("operator"))
    listed = client.get(
        "/api/tool-call-preauthorizations", params={"task_id": "task-1"}, headers=_as("reader")
    )
    assert [
        view["preauthorization"]["preauthorization_id"]
        for view in listed.json()["preauthorizations"]
    ] == ["pre-1"]
    hidden = client.get(
        "/api/tool-call-preauthorizations", params={"task_id": "task-1"}, headers=_as("outsider")
    )
    assert hidden.json() == {"preauthorizations": []}
    assert (
        client.get("/api/tool-call-preauthorizations/pre-1", headers=_as("outsider")).status_code
        == 404
    )
    assert (
        client.post(
            "/api/tool-call-preauthorizations/pre-1/revoke", headers=_as("operator-2")
        ).status_code
        == 403
    )
    revoked = client.post("/api/tool-call-preauthorizations/pre-1/revoke", headers=_as("operator"))
    assert revoked.json()["status"] == "revoked"
    assert revoked.json()["preauthorization"]["revoked_by"]["subject"] == "operator"
    assert (
        client.get("/api/tool-call-preauthorizations/missing", headers=_as("reader")).status_code
        == 404
    )


def _hosted(monkeypatch, tmp_path, *, hosted: bool = True):
    store = InMemoryToolCallPreauthorizationStore()

    def build(_target, **_kwargs):
        app = CayuApp(enable_logging=False)
        if hosted:
            app.host_tool_call_preauthorizations(store, authorize=_operators_only)
        return app

    monkeypatch.setattr(
        preauthorizations_cli,
        "resolve_project",
        lambda target, **_kwargs: SimpleNamespace(root=tmp_path, target=target or "app:build"),
    )
    monkeypatch.setattr(preauthorizations_cli, "project_context", lambda _root: nullcontext())
    monkeypatch.setattr(preauthorizations_cli, "build_project_app", build)
    return store


def _json(capsys, argv: list[str], code: int = 0) -> dict:
    assert main([*argv, "--format", "json"]) == code
    return json.loads(capsys.readouterr().out)


def test_cli_issues_lists_shows_and_revokes_as_the_reviewer(monkeypatch, tmp_path, capsys) -> None:
    _hosted(monkeypatch, tmp_path)
    issued = _json(
        capsys,
        [
            "preauthorizations",
            "issue",
            "--task",
            "task-1",
            "--tool",
            "send_email",
            "--arguments",
            json.dumps(ARGUMENTS),
            "--id",
            "pre-1",
            "--reviewer",
            "operator",
            "--justification",
            "Send the one reminder.",
        ],
    )
    assert issued["preauthorization"]["authorized_by"]["subject"] == "operator"
    listed = _json(capsys, ["preauthorizations", "list", "--task", "task-1", "--reviewer", "a"])
    assert len(listed["preauthorizations"]) == 1
    shown = _json(capsys, ["preauthorizations", "show", "pre-1", "--reviewer", "a"])
    assert shown["status"] == "available"
    refused = _json(
        capsys, ["preauthorizations", "revoke", "pre-1", "--reviewer", "intern"], code=1
    )
    assert refused["error"]["code"] == "forbidden"
    revoked = _json(capsys, ["preauthorizations", "revoke", "pre-1", "--reviewer", "operator"])
    assert revoked["status"] == "revoked"


def test_cli_explains_an_application_that_hosts_none(monkeypatch, tmp_path, capsys) -> None:
    _hosted(monkeypatch, tmp_path, hosted=False)
    answer = _json(capsys, ["preauthorizations", "list", "--task", "t"], code=1)
    assert answer["error"]["code"] == "preauthorizations_not_served"
    bad = _json(
        capsys,
        ["preauthorizations", "issue", "--task", "t", "--tool", "x", "--arguments", "[1]"],
        code=1,
    )
    assert bad["error"]["code"] == "invalid_input"


def test_create_server_mounts_the_routes_only_when_given_and_authenticated() -> None:
    from fastapi import HTTPException

    from cayu.server import ServerConfig, create_server

    def auth(request: Request) -> AuthContext:
        subject = request.headers.get("x-subject")
        if not subject:
            raise HTTPException(status_code=401, detail="unauthenticated")
        return AuthContext(subject=subject)

    app = CayuApp(enable_logging=False)
    routes = ToolCallPreauthorizationServer(
        InMemoryToolCallPreauthorizationStore(), authorize=_operators_only
    )
    server = create_server(
        app, config=ServerConfig.protected(auth), tool_call_preauthorizations=routes
    )
    client = TestClient(server)
    assert client.post("/api/tool-call-preauthorizations", json=ISSUE).status_code == 401
    assert (
        client.post(
            "/api/tool-call-preauthorizations", json=ISSUE, headers=_as("operator")
        ).status_code
        == 201
    )
    plain = create_server(app, config=ServerConfig.protected(auth))
    assert not any("preauthorizations" in path for path in plain.openapi()["paths"])
    with pytest.raises(ValueError, match="authenticated access"):
        create_server(
            app, config=ServerConfig.local_development(), tool_call_preauthorizations=routes
        )


def test_an_id_with_a_slash_can_be_shown_and_revoked(monkeypatch, tmp_path, capsys) -> None:
    client = _client(InMemoryToolCallPreauthorizationStore())
    body = {**ISSUE, "preauthorization_id": "invoice-42/send"}
    assert (
        client.post("/api/tool-call-preauthorizations", json=body, headers=_as("operator"))
    ).status_code == 201
    shown = client.get("/api/tool-call-preauthorizations/invoice-42/send", headers=_as("r"))
    assert shown.status_code == 200
    assert shown.json()["preauthorization"]["preauthorization_id"] == "invoice-42/send"
    revoked = client.post(
        "/api/tool-call-preauthorizations/invoice-42/send/revoke", headers=_as("operator")
    )
    assert revoked.json()["status"] == "revoked"
    # The CLI percent-encodes the id; the route still finds it.
    _hosted(monkeypatch, tmp_path)
    _json(
        capsys,
        [
            "preauthorizations",
            "issue",
            "--task",
            "t",
            "--tool",
            "x",
            "--arguments",
            "{}",
            "--id",
            "a/b",
            "--reviewer",
            "operator",
        ],
    )
    assert _json(capsys, ["preauthorizations", "show", "a/b"])["status"] == "available"


def test_a_retry_later_returns_the_first_record() -> None:
    times = iter([NOW, NOW.replace(minute=5), NOW.replace(minute=5), NOW.replace(minute=6)])
    server = ToolCallPreauthorizationServer(
        InMemoryToolCallPreauthorizationStore(),
        authorize=_operators_only,
        clock=lambda: next(times),
    )
    api = FastAPI()
    api.include_router(
        server.router(auth=lambda request: AuthContext(subject=request.headers["x-subject"])),
        prefix="/api",
    )
    client = TestClient(api)
    body = {**ISSUE, "preauthorization_id": "pre-1", "lifetime_seconds": 600}
    first = client.post("/api/tool-call-preauthorizations", json=body, headers=_as("operator"))
    again = client.post("/api/tool-call-preauthorizations", json=body, headers=_as("operator"))
    assert again.status_code == 201
    assert again.json()["preauthorization"] == first.json()["preauthorization"]


def test_cli_reports_unreadable_files_and_refused_fields(monkeypatch, tmp_path, capsys) -> None:
    _hosted(monkeypatch, tmp_path)
    missing = _json(
        capsys,
        [
            "preauthorizations",
            "issue",
            "--task",
            "t",
            "--tool",
            "x",
            "--arguments-file",
            str(tmp_path / "missing.json"),
            "--reviewer",
            "operator",
        ],
        code=1,
    )
    assert missing["error"]["code"] == "invalid_input"
    assert "--arguments-file" in missing["error"]["message"]
    refused = _json(
        capsys,
        [
            "preauthorizations",
            "issue",
            "--task",
            "t",
            "--tool",
            "x",
            "--arguments",
            "{}",
            "--expires-in-seconds",
            "0",
            "--reviewer",
            "operator",
        ],
        code=1,
    )
    assert "lifetime_seconds" in refused["error"]["message"]


class _Cloud:
    """Cayu Cloud's API as the CLI sees it: answers by (method, path), recording requests."""

    def __init__(self, answers) -> None:
        self.answers = answers
        self.requests: list[tuple] = []

    def request(self, method, path, *, payload=None, query=None, idempotency_key=None):
        self.requests.append((method, path, payload, query))
        answer = self.answers[(method, path)]
        if isinstance(answer, Exception):
            raise answer
        return dict(answer)


def _cloud(monkeypatch, answers) -> _Cloud:
    import cayu.cli.cloud as cloud_cli

    cloud = _Cloud(answers)
    monkeypatch.setattr(cloud_cli, "cloud_client", lambda **_kwargs: cloud)
    monkeypatch.setattr(cloud_cli, "resolve_application_id", lambda _client, name: f"app-{name}")
    return cloud


def test_cli_asks_a_cloud_agent_through_cloud_as_the_signed_in_person(monkeypatch, capsys) -> None:
    base = "/v1/applications/app-mailer/tool-call-preauthorizations"
    view = {"status": "available", "preauthorization": {"preauthorization_id": "pre-1"}}
    cloud = _cloud(
        monkeypatch,
        {
            ("POST", base): view,
            ("GET", base): {"served": True, "preauthorizations": [view]},
            ("POST", base + "/pre-1/revoke"): {**view, "status": "revoked"},
        },
    )
    issued = _json(
        capsys,
        [
            "preauthorizations",
            "issue",
            "--application",
            "mailer",
            "--task",
            "task-1",
            "--tool",
            "send_email",
            "--arguments",
            json.dumps(ARGUMENTS),
        ],
    )
    assert issued["status"] == "available"
    method, path, payload, _query = cloud.requests[0]
    # Cloud names the signed-in person; the CLI sends no author.
    assert (method, path) == ("POST", base) and "authorized_by" not in payload
    listed = _json(
        capsys, ["preauthorizations", "list", "--application", "mailer", "--task", "task-1"]
    )
    assert "served" not in listed and len(listed["preauthorizations"]) == 1
    assert cloud.requests[1][3] == {"task_id": "task-1"}
    revoked = _json(capsys, ["preauthorizations", "revoke", "pre-1", "--application", "mailer"])
    assert revoked["status"] == "revoked"
    local_only = _json(
        capsys,
        ["preauthorizations", "show", "pre-1", "--application", "mailer", "--reviewer", "x"],
        code=1,
    )
    assert local_only["error"]["code"] == "invalid_input"


def test_cli_says_when_cloud_or_the_agent_serves_none(monkeypatch, capsys) -> None:
    from cayu.cli._cloud_api import CloudApiError

    base = "/v1/applications/app-mailer/tool-call-preauthorizations"
    _cloud(
        monkeypatch,
        {
            ("GET", base): {"served": False, "preauthorizations": []},
            ("GET", base + "/pre-1"): CloudApiError("not_found", "Not Found", status_code=404),
        },
    )
    unserved = _json(
        capsys, ["preauthorizations", "list", "--application", "mailer", "--task", "t"], code=1
    )
    assert unserved["error"]["code"] == "preauthorizations_not_served"
    older = _json(capsys, ["preauthorizations", "show", "pre-1", "--application", "mailer"], code=1)
    assert older["error"]["code"] == "preauthorizations_unsupported"

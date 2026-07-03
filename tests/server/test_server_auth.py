from __future__ import annotations

# ruff: noqa: E402
from collections.abc import AsyncIterator

import pytest

fastapi = pytest.importorskip("fastapi")
pytest.importorskip("sse_starlette")

from fastapi import HTTPException, Request
from fastapi.testclient import TestClient

from cayu import AgentSpec, CayuApp, InMemoryTaskStore
from cayu.providers import ModelProvider, ModelRequest, ModelStreamEvent
from cayu.server import create_server

_TOKEN = "secret-token"
_AUTH_HEADERS = {"Authorization": f"Bearer {_TOKEN}"}


class OneShotProvider(ModelProvider):
    name = "fake"

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        yield ModelStreamEvent.text_delta("done")
        yield ModelStreamEvent.completed({"finish_reason": "stop"})


def _require_bearer_token(request: Request) -> None:
    if request.headers.get("Authorization") != f"Bearer {_TOKEN}":
        raise HTTPException(status_code=401, detail="Missing or invalid credentials.")


def _make_client() -> TestClient:
    app = CayuApp(task_store=InMemoryTaskStore())
    app.register_provider(OneShotProvider(), default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))
    return TestClient(create_server(app, auth=_require_bearer_token))


@pytest.mark.parametrize(
    ("method", "path", "body"),
    [
        ("POST", "/api/run", {"prompt": "hello"}),
        ("POST", "/api/resume", {"session_id": "session-1", "prompt": "hi"}),
        ("POST", "/api/sessions/session-1/interrupt", None),
        (
            "POST",
            "/api/tool-approvals/resolve",
            {"session_id": "session-1", "approval_id": "approval-1", "decision": "approve"},
        ),
        (
            "POST",
            "/api/tool-approvals/recover",
            {
                "session_id": "session-1",
                "approval_id": "approval-1",
                "tool_call_id": "call-1",
                "outcome": "succeeded",
                "message": "done",
            },
        ),
        ("DELETE", "/api/sessions/session-1", None),
        ("PATCH", "/api/sessions/session-1/labels", {"labels": {}}),
        ("PATCH", "/api/sessions/session-1/metadata", {"metadata": {}}),
        ("POST", "/api/tasks/task-1/pause", None),
        ("POST", "/api/tasks/task-1/block", None),
        ("POST", "/api/tasks/task-1/needs-attention", None),
        ("POST", "/api/tasks/task-1/resume", None),
        ("POST", "/api/knowledge/entry-1/approve", None),
        ("POST", "/api/knowledge/entry-1/reject", None),
    ],
)
def test_auth_guards_sensitive_routes(method: str, path: str, body: dict | None) -> None:
    client = _make_client()

    response = client.request(method, path, json=body)

    assert response.status_code == 401
    assert response.json()["detail"] == "Missing or invalid credentials."


def test_auth_denied_run_creates_no_task_or_session() -> None:
    client = _make_client()

    assert client.post("/api/run", json={"prompt": "hello"}).status_code == 401

    assert client.get("/api/tasks").json() == []
    assert client.get("/api/sessions").json()["sessions"] == []


def test_authenticated_run_streams_and_records_task() -> None:
    client = _make_client()

    with client.stream(
        "POST",
        "/api/run",
        json={"prompt": "hello"},
        headers=_AUTH_HEADERS,
    ) as response:
        assert response.status_code == 200
        list(response.iter_lines())

    tasks = client.get("/api/tasks").json()
    assert len(tasks) == 1
    assert tasks[0]["status"] == "completed"


def test_authenticated_requests_reach_route_handlers() -> None:
    client = _make_client()

    # Auth passes; the handler itself reports the missing resource.
    response = client.delete("/api/sessions/session-missing", headers=_AUTH_HEADERS)
    assert response.status_code == 204

    response = client.post("/api/knowledge/entry-1/approve", headers=_AUTH_HEADERS)
    assert response.status_code == 404
    assert response.json()["detail"] == "Knowledge store is not configured."


def test_read_only_routes_stay_open_when_auth_configured() -> None:
    client = _make_client()

    assert client.get("/api/health").json() == {"ok": True}
    assert client.get("/api/sessions").status_code == 200
    assert client.get("/api/tasks").status_code == 200


def test_server_without_auth_keeps_routes_open() -> None:
    app = CayuApp(task_store=InMemoryTaskStore())
    app.register_provider(OneShotProvider(), default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))
    client = TestClient(create_server(app))

    with client.stream("POST", "/api/run", json={"prompt": "hello"}) as response:
        assert response.status_code == 200
        list(response.iter_lines())

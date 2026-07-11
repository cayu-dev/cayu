from __future__ import annotations

# ruff: noqa: E402
from collections.abc import AsyncIterator

import pytest

fastapi = pytest.importorskip("fastapi")
pytest.importorskip("sse_starlette")

from fastapi import FastAPI, HTTPException, Request
from fastapi.testclient import TestClient

from cayu import AgentSpec, CayuApp, InMemoryTaskStore
from cayu.providers import ModelProvider, ModelRequest, ModelStreamEvent
from cayu.server import AuthContext, BasicAuth, create_server, mount_cayu

_TOKEN = "secret-token"
_AUTH_HEADERS = {"Authorization": f"Bearer {_TOKEN}"}
_PRICING_BODY = {
    "pricing": {
        "prices": [
            {
                "provider_name": "fake",
                "model": "fake-model",
                "input_per_million": "1",
                "output_per_million": "1",
            }
        ]
    }
}


class OneShotProvider(ModelProvider):
    name = "fake"

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        yield ModelStreamEvent.text_delta("done")
        yield ModelStreamEvent.completed({"finish_reason": "stop"})


def _require_bearer_token(request: Request) -> AuthContext:
    if request.headers.get("Authorization") != f"Bearer {_TOKEN}":
        raise HTTPException(status_code=401, detail="Missing or invalid credentials.")
    return AuthContext(subject="test-user", claims={"scheme": "bearer"})


def _make_client(*, expose_docs: bool | None = None) -> TestClient:
    app = CayuApp(task_store=InMemoryTaskStore())
    app.register_provider(OneShotProvider(), default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))
    return TestClient(create_server(app, auth=_require_bearer_token, expose_docs=expose_docs))


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
                "outcome": "completed",
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
def test_auth_guards_mutating_routes(method: str, path: str, body: dict | None) -> None:
    client = _make_client()

    response = client.request(method, path, json=body)

    assert response.status_code == 401
    assert response.json()["detail"] == "Missing or invalid credentials."


def test_auth_denied_run_creates_no_task_or_session() -> None:
    client = _make_client()

    assert client.post("/api/run", json={"prompt": "hello"}).status_code == 401

    assert client.get("/api/tasks", headers=_AUTH_HEADERS).json() == []
    assert client.get("/api/sessions", headers=_AUTH_HEADERS).json()["sessions"] == []


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

    tasks = client.get("/api/tasks", headers=_AUTH_HEADERS).json()
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


def test_health_stays_open_when_auth_configured() -> None:
    client = _make_client()

    assert client.get("/api/health").json() == {"ok": True}


def test_protected_server_disables_generated_docs_by_default() -> None:
    client = _make_client()

    assert client.get("/openapi.json").status_code == 404
    assert client.get("/docs").status_code == 404
    assert client.get("/redoc").status_code == 404

    contract = client.get("/api/contract", headers=_AUTH_HEADERS).json()
    assert contract["client_generation"]["openapi_url"] is None


def test_protected_server_can_expose_generated_docs_explicitly() -> None:
    client = _make_client(expose_docs=True)

    assert client.get("/openapi.json").status_code == 200
    assert client.get("/docs").status_code == 200
    assert client.get("/redoc").status_code == 200

    contract = client.get("/api/contract", headers=_AUTH_HEADERS).json()
    assert contract["client_generation"]["openapi_url"] == "/openapi.json"


@pytest.mark.parametrize(
    ("method", "path", "body"),
    [
        ("GET", "/api/sessions", None),
        ("POST", "/api/sessions/summary", None),
        ("GET", "/api/sessions/session-1/usage", None),
        ("POST", "/api/sessions/session-1/cost", _PRICING_BODY),
        ("GET", "/api/causal-budgets/budget-1/usage", None),
        ("POST", "/api/causal-budgets/budget-1/cost", _PRICING_BODY),
        ("POST", "/api/causal-budgets/budget-1/summary", _PRICING_BODY),
        ("GET", "/api/sessions/session-1/summary", None),
        ("GET", "/api/sessions/session-1/events", None),
        ("GET", "/api/sessions/session-1/transcript", None),
        ("GET", "/api/sessions/session-1", None),
        ("GET", "/api/artifacts/missing/content", None),
        ("GET", "/api/tasks", None),
        ("GET", "/api/knowledge/pending", None),
        ("GET", "/api/knowledge/pending/entry-1", None),
        ("GET", "/api/contract", None),
    ],
)
def test_auth_guards_read_and_contract_routes(
    method: str,
    path: str,
    body: dict | None,
) -> None:
    client = _make_client()

    response = client.request(method, path, json=body)

    assert response.status_code == 401
    assert response.json()["detail"] == "Missing or invalid credentials."


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
                "outcome": "completed",
                "message": "done",
            },
        ),
        (
            "POST",
            "/api/user-input/resolve",
            {"session_id": "session-1", "input_id": "input-1", "answer": "done"},
        ),
        (
            "POST",
            "/api/user-input/recover",
            {
                "session_id": "session-1",
                "input_id": "input-1",
                "answer": "done",
                "tool_call_id": "call-1",
                "outcome": "completed",
                "message": "done",
            },
        ),
    ],
)
def test_auth_guards_streaming_event_routes(
    method: str,
    path: str,
    body: dict | None,
) -> None:
    client = _make_client()

    response = client.request(method, path, json=body)

    assert response.status_code == 401
    assert response.json()["detail"] == "Missing or invalid credentials."


def test_authenticated_requests_reach_read_handlers() -> None:
    client = _make_client()

    assert client.get("/api/sessions", headers=_AUTH_HEADERS).status_code == 200
    assert client.get("/api/tasks", headers=_AUTH_HEADERS).status_code == 200
    assert client.get("/api/contract", headers=_AUTH_HEADERS).status_code == 200

    missing = client.get("/api/sessions/missing", headers=_AUTH_HEADERS)
    assert missing.status_code == 404
    assert missing.json()["detail"] == "Session not found"


def test_create_server_requires_auth_unless_dev_mode() -> None:
    with pytest.raises(ValueError, match="requires auth"):
        create_server(CayuApp())


def test_mount_cayu_requires_auth_unless_dev_mode() -> None:
    with pytest.raises(ValueError, match="requires auth"):
        mount_cayu(FastAPI(), CayuApp())


def test_basic_auth_dependency_authenticates_control_plane() -> None:
    app = CayuApp(task_store=InMemoryTaskStore())
    app.register_provider(OneShotProvider(), default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))
    client = TestClient(
        create_server(
            app,
            auth=BasicAuth(
                username="operator",
                password="secret-password",
                tenant="tenant-a",
                claims={"role": "admin"},
            ),
        )
    )

    denied = client.get("/api/sessions")
    assert denied.status_code == 401
    assert denied.headers["www-authenticate"] == 'Basic realm="Cayu"'

    accepted = client.get(
        "/api/sessions",
        auth=("operator", "secret-password"),
    )
    assert accepted.status_code == 200


def test_basic_auth_dependency_authenticates_dashboard_shell() -> None:
    app = CayuApp(task_store=InMemoryTaskStore())
    app.register_provider(OneShotProvider(), default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))
    client = TestClient(
        create_server(
            app,
            auth=BasicAuth(username="operator", password="secret-password"),
        )
    )

    denied = client.get("/cayu/")
    assert denied.status_code == 401
    assert denied.headers["www-authenticate"] == 'Basic realm="Cayu"'

    denied_asset = client.get("/cayu/assets/missing.js")
    assert denied_asset.status_code == 401

    accepted = client.get("/cayu/", auth=("operator", "secret-password"))
    assert accepted.status_code == 200
    assert '"basePath":"/cayu"' in accepted.text


def test_mount_cayu_authenticates_embedded_api_and_dashboard() -> None:
    app = CayuApp(task_store=InMemoryTaskStore())
    app.register_provider(OneShotProvider(), default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))
    server = FastAPI()
    mount_cayu(
        server,
        app,
        auth=BasicAuth(username="operator", password="secret-password"),
    )
    client = TestClient(server)

    assert client.get("/cayu/api/health").json() == {"ok": True}

    denied_api = client.get("/cayu/api/sessions")
    assert denied_api.status_code == 401
    assert denied_api.headers["www-authenticate"] == 'Basic realm="Cayu"'

    denied_dashboard = client.get("/cayu/")
    assert denied_dashboard.status_code == 401

    accepted_api = client.get("/cayu/api/sessions", auth=("operator", "secret-password"))
    assert accepted_api.status_code == 200

    accepted_dashboard = client.get("/cayu/", auth=("operator", "secret-password"))
    assert accepted_dashboard.status_code == 200
    assert '"basePath":"/cayu"' in accepted_dashboard.text
    assert '"apiBaseUrl":"/cayu/api"' in accepted_dashboard.text


def test_custom_auth_dependency_may_return_mapping_context() -> None:
    def mapping_auth(request: Request) -> dict:
        if request.headers.get("Authorization") != f"Bearer {_TOKEN}":
            raise HTTPException(status_code=401, detail="Missing or invalid credentials.")
        return {"subject": "custom-user", "tenant": "tenant-b", "claims": {"issuer": "jwt"}}

    app = CayuApp(task_store=InMemoryTaskStore())
    app.register_provider(OneShotProvider(), default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))
    client = TestClient(create_server(app, auth=mapping_auth))

    assert client.get("/api/sessions").status_code == 401
    assert client.get("/api/sessions", headers=_AUTH_HEADERS).status_code == 200


def test_custom_auth_dependency_may_be_async() -> None:
    async def async_auth(request: Request) -> AuthContext:
        if request.headers.get("Authorization") != f"Bearer {_TOKEN}":
            raise HTTPException(status_code=401, detail="Missing or invalid credentials.")
        return AuthContext(subject="async-user")

    app = CayuApp(task_store=InMemoryTaskStore())
    app.register_provider(OneShotProvider(), default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))
    client = TestClient(create_server(app, auth=async_auth))

    assert client.get("/api/sessions").status_code == 401
    assert client.get("/api/sessions", headers=_AUTH_HEADERS).status_code == 200


def test_dev_server_without_auth_keeps_routes_open() -> None:
    app = CayuApp(task_store=InMemoryTaskStore())
    app.register_provider(OneShotProvider(), default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))
    client = TestClient(create_server(app, dev=True))

    with client.stream("POST", "/api/run", json={"prompt": "hello"}) as response:
        assert response.status_code == 200
        list(response.iter_lines())


def _approval_capture_app() -> tuple[CayuApp, list]:
    from cayu import Message, RunRequest
    from cayu.runtime import SessionIdentity, SessionStatus

    app = CayuApp()
    app.register_provider(OneShotProvider(), default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    async def create_interrupted_session(session_id: str) -> None:
        await app.session_store.create(
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "hello")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        await app.session_store.update_status(session_id, SessionStatus.INTERRUPTED)

    import asyncio

    asyncio.run(create_interrupted_session("session_actor"))

    captured: list = []

    async def resolve_tool_approval(request):
        captured.append(request)
        if False:
            yield None

    app.resolve_tool_approval = resolve_tool_approval
    return app, captured


def test_authenticated_resolution_derives_resolved_by_from_auth_context() -> None:
    from cayu import ResolutionActorSource

    app, captured = _approval_capture_app()
    client = TestClient(create_server(app, auth=_require_bearer_token))

    with client.stream(
        "POST",
        "/api/tool-approvals/resolve",
        headers=_AUTH_HEADERS,
        json={
            "session_id": "session_actor",
            "approval_id": "approval_1",
            "decision": "approve",
        },
    ) as response:
        assert response.status_code == 200
        list(response.iter_lines())

    actor = captured[0].resolved_by
    assert actor is not None
    assert actor.subject == "test-user"
    assert actor.source is ResolutionActorSource.HTTP_AUTH
    assert actor.claims == {"scheme": "bearer"}


def test_authenticated_resolution_reserved_auth_subject_returns_400() -> None:
    def reserved_subject_auth(request: Request) -> AuthContext:
        if request.headers.get("Authorization") != f"Bearer {_TOKEN}":
            raise HTTPException(status_code=401, detail="Missing or invalid credentials.")
        return AuthContext(subject="cayu:ops", claims={"scheme": "bearer"})

    app, captured = _approval_capture_app()
    client = TestClient(create_server(app, auth=reserved_subject_auth))

    response = client.post(
        "/api/tool-approvals/resolve",
        headers=_AUTH_HEADERS,
        json={
            "session_id": "session_actor",
            "approval_id": "approval_1",
            "decision": "approve",
        },
    )

    assert response.status_code == 400
    assert "reserved" in response.json()["detail"]
    assert captured == []


def test_authenticated_resolution_rejects_body_resolved_by() -> None:
    app, captured = _approval_capture_app()
    client = TestClient(create_server(app, auth=_require_bearer_token))

    response = client.post(
        "/api/tool-approvals/resolve",
        headers=_AUTH_HEADERS,
        json={
            "session_id": "session_actor",
            "approval_id": "approval_1",
            "decision": "approve",
            "resolved_by": {"subject": "someone-else"},
        },
    )

    assert response.status_code == 400
    assert "derived from the authenticated caller" in response.json()["detail"]
    assert captured == []


def _interrupt_capture_app() -> tuple[CayuApp, list]:
    import asyncio

    from cayu import Event, EventType, Message, RunRequest
    from cayu.runtime import SessionIdentity

    app = CayuApp()
    app.register_provider(OneShotProvider(), default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    async def create_pending_session() -> None:
        await app.session_store.create(
            RunRequest(
                agent_name="assistant",
                session_id="session_interrupt_actor",
                messages=[Message.text("user", "hello")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )

    asyncio.run(create_pending_session())
    captured: list = []

    async def interrupt_session(request):
        captured.append(request)
        yield Event(
            type=EventType.SESSION_INTERRUPTED,
            session_id=request.session_id,
            agent_name="assistant",
            payload={"interruption_type": "operator_requested"},
        )

    app.interrupt_session = interrupt_session
    return app, captured


def test_authenticated_interruption_derives_requested_by_from_auth_context() -> None:
    from cayu import ResolutionActorSource

    app, captured = _interrupt_capture_app()
    client = TestClient(create_server(app, auth=_require_bearer_token))

    with client.stream(
        "POST",
        "/api/sessions/session_interrupt_actor/interrupt",
        headers=_AUTH_HEADERS,
        json={"reason": "operator stop"},
    ) as response:
        assert response.status_code == 200
        list(response.iter_lines())

    actor = captured[0].requested_by
    assert actor is not None
    assert actor.subject == "test-user"
    assert actor.source is ResolutionActorSource.HTTP_AUTH
    assert actor.claims == {"scheme": "bearer"}


def test_authenticated_interruption_rejects_body_requested_by() -> None:
    app, captured = _interrupt_capture_app()
    client = TestClient(create_server(app, auth=_require_bearer_token))

    response = client.post(
        "/api/sessions/session_interrupt_actor/interrupt",
        headers=_AUTH_HEADERS,
        json={"requested_by": {"subject": "someone-else"}},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == (
        "requested_by is derived from the authenticated caller and "
        "cannot be supplied in the request body."
    )
    assert captured == []

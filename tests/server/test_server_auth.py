from __future__ import annotations

# ruff: noqa: E402
import asyncio
import inspect
from collections.abc import AsyncIterator

import pytest

fastapi = pytest.importorskip("fastapi")
pytest.importorskip("sse_starlette")

from fastapi import FastAPI, HTTPException, Request
from fastapi.testclient import TestClient

from cayu import (
    AgentSpec,
    CayuApp,
    InMemoryTaskStore,
    InvocationOriginTrust,
    SessionExecutionSource,
)
from cayu.core.events import Event, EventType
from cayu.providers import ModelProvider, ModelRequest, ModelStreamEvent
from cayu.server import (
    AuthContext,
    AuthenticatedAccess,
    BasicAuth,
    DashboardStaticFiles,
    DocsConfig,
    ServerConfig,
    create_router,
    create_server,
    mount_cayu,
    mount_dashboard,
)
from cayu.server.auth import server_auth_dependency

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
    return AuthContext(
        subject="test-user",
        tenant="tenant-a",
        claims={"scheme": "bearer"},
    )


def _make_client(*, expose_docs: bool | None = None) -> TestClient:
    app = CayuApp(task_store=InMemoryTaskStore())
    app.register_provider(OneShotProvider(), default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))
    return TestClient(
        create_server(
            app,
            config=ServerConfig.protected(
                _require_bearer_token,
                docs=DocsConfig(enabled=expose_docs is True),
            ),
        )
    )


def test_auth_context_cache_is_bound_to_the_originating_dependency() -> None:
    calls: list[str] = []

    def first_auth(_request: Request) -> AuthContext:
        calls.append("first")
        return AuthContext(subject="first")

    def second_auth(_request: Request) -> AuthContext:
        calls.append("second")
        return AuthContext(subject="second")

    async def exercise() -> tuple[AuthContext, AuthContext, AuthContext]:
        request = Request(
            {
                "type": "http",
                "method": "GET",
                "path": "/",
                "headers": [],
                "query_string": b"",
            }
        )
        first_dependency = server_auth_dependency(first_auth)
        second_dependency = server_auth_dependency(second_auth)
        return (
            await first_dependency(request),
            await first_dependency(request),
            await second_dependency(request),
        )

    first, replayed, second = asyncio.run(exercise())

    assert first.subject == "first"
    assert replayed.subject == "first"
    assert second.subject == "second"
    assert calls == ["first", "second"]


@pytest.mark.parametrize(
    "public_api",
    [
        BasicAuth,
        DashboardStaticFiles,
        create_router,
        create_server,
        mount_cayu,
        mount_dashboard,
    ],
)
def test_public_auth_helpers_document_tenant_as_provenance_only(public_api) -> None:
    documentation = inspect.getdoc(public_api)

    assert documentation is not None
    assert "AuthContext" in documentation
    assert "provenance only" in documentation
    assert "does not" in documentation


@pytest.mark.parametrize(
    ("method", "path", "body"),
    [
        ("POST", "/api/run", {"prompt": "hello"}),
        ("POST", "/api/resume", {"session_id": "session-1", "prompt": "hi"}),
        (
            "POST",
            "/api/sessions/session-1/compact",
            {
                "idempotency_key": "compact-1",
                "expected_run_epoch": 0,
                "expected_transcript_cursor": 0,
            },
        ),
        (
            "POST",
            "/api/sessions/session-1/messages",
            {
                "idempotency_key": "message-1",
                "content": "steer",
                "delivery_mode": "next_turn",
            },
        ),
        ("POST", "/api/sessions/session-1/interrupt", None),
        (
            "POST",
            "/api/provider-operations/resolve",
            {
                "session_id": "session-1",
                "stage_id": "stage-1",
                "expected_run_epoch": 0,
                "action": "fail",
            },
        ),
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
                "tool_round_id": "round_1",
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
        ("POST", "/api/operations/snapshot", {}),
        (
            "POST",
            "/api/usage/rollup",
            {
                "start_at": "2026-07-01T00:00:00Z",
                "end_at": "2026-07-02T00:00:00Z",
            },
        ),
        ("GET", "/api/sessions/session-1/usage", None),
        ("POST", "/api/sessions/session-1/cost", _PRICING_BODY),
        ("GET", "/api/causal-budgets/budget-1/usage", None),
        ("POST", "/api/causal-budgets/budget-1/cost", _PRICING_BODY),
        ("POST", "/api/causal-budgets/budget-1/summary", _PRICING_BODY),
        ("GET", "/api/sessions/session-1/summary", None),
        ("POST", "/api/sessions/session-1/topology", {}),
        ("GET", "/api/sessions/session-1/events", None),
        ("GET", "/api/sessions/session-1/transcript", None),
        ("GET", "/api/sessions/session-1", None),
        ("GET", "/api/artifacts/missing/content", None),
        ("GET", "/api/tasks", None),
        ("GET", "/api/knowledge/pending", None),
        ("GET", "/api/knowledge/pending/entry-1", None),
        ("GET", "/api/contract", None),
        ("GET", "/api/system/diagnostics", None),
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
        (
            "POST",
            "/api/sessions/session-1/compact",
            {
                "idempotency_key": "compact-1",
                "expected_run_epoch": 0,
                "expected_transcript_cursor": 0,
            },
        ),
        (
            "POST",
            "/api/sessions/session-1/messages",
            {
                "idempotency_key": "message-1",
                "content": "steer",
                "delivery_mode": "next_turn",
            },
        ),
        ("POST", "/api/sessions/session-1/interrupt", None),
        (
            "POST",
            "/api/provider-operations/resolve",
            {
                "session_id": "session-1",
                "stage_id": "stage-1",
                "expected_run_epoch": 0,
                "action": "fail",
            },
        ),
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
                "tool_round_id": "round_1",
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
    assert client.get("/api/system/diagnostics", headers=_AUTH_HEADERS).status_code == 200

    missing = client.get("/api/sessions/missing", headers=_AUTH_HEADERS)
    assert missing.status_code == 404
    assert missing.json()["detail"] == "Session not found"


def test_create_server_requires_explicit_config() -> None:
    with pytest.raises(TypeError, match="required keyword-only argument: 'config'"):
        create_server(CayuApp())


def test_mount_cayu_requires_explicit_access() -> None:
    with pytest.raises(TypeError, match="required keyword-only argument: 'access'"):
        mount_cayu(FastAPI(), CayuApp())


def test_basic_auth_dependency_authenticates_control_plane() -> None:
    app = CayuApp(task_store=InMemoryTaskStore())
    app.register_provider(OneShotProvider(), default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))
    client = TestClient(
        create_server(
            app,
            config=ServerConfig.protected(
                BasicAuth(
                    username="operator",
                    password="secret-password",
                    tenant="tenant-a",
                    claims={"role": "admin"},
                )
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


@pytest.mark.parametrize("realm", ["Бишкек", "bad\r\nX-Evil: yes", "bad\tvalue"])
def test_basic_auth_rejects_realms_that_cannot_be_emitted_safely(realm: str) -> None:
    with pytest.raises(ValueError, match="visible ASCII"):
        BasicAuth(username="operator", password="secret-password", realm=realm)


def test_basic_auth_escapes_quoted_realm_characters() -> None:
    client = TestClient(
        create_server(
            CayuApp(),
            config=ServerConfig.protected(
                BasicAuth(
                    username="operator",
                    password="secret-password",
                    realm='Operations "blue" \\ realm',
                )
            ),
        )
    )

    denied = client.get("/api/sessions")

    assert denied.status_code == 401
    assert denied.headers["www-authenticate"] == 'Basic realm="Operations \\"blue\\" \\\\ realm"'


@pytest.mark.parametrize("field_name", ["username", "password", "realm", "subject", "tenant"])
def test_basic_auth_rejects_non_scalar_text_before_runtime_encoding(field_name: str) -> None:
    values = {"username": "operator", "password": "secret-password"}
    values[field_name] = f"bad{chr(0xD800)}value"

    with pytest.raises(ValueError, match="Unicode surrogate"):
        BasicAuth(**values)


def test_basic_auth_rejects_username_delimiter_and_non_scalar_claims() -> None:
    with pytest.raises(ValueError, match="must not contain a colon"):
        BasicAuth(username="operator:admin", password="secret-password")

    with pytest.raises(ValueError, match="Unicode surrogate"):
        BasicAuth(
            username="operator",
            password="secret-password",
            claims={"role": f"bad{chr(0xD800)}value"},
        )


@pytest.mark.parametrize("field_name", ["subject", "tenant"])
def test_auth_context_bounds_identity_used_by_contract_and_audit_events(field_name: str) -> None:
    values = {"subject": "operator"}
    values[field_name] = "a" * 513

    with pytest.raises(ValueError, match="at most 512 characters"):
        AuthContext(**values)


@pytest.mark.parametrize("field_name", ["subject", "tenant"])
def test_auth_context_rejects_non_scalar_actor_identity(field_name: str) -> None:
    values = {"subject": "operator"}
    values[field_name] = f"bad{chr(0xD800)}identity"

    with pytest.raises(ValueError):
        AuthContext(**values)


@pytest.mark.parametrize("field_name", ["username", "subject", "tenant"])
def test_basic_auth_rejects_oversized_actor_identity_at_configuration_time(
    field_name: str,
) -> None:
    values = {"username": "operator", "password": "secret-password"}
    values[field_name] = "a" * 513

    with pytest.raises(ValueError, match="at most 512 characters"):
        BasicAuth(**values)


def test_basic_auth_dependency_authenticates_dashboard_shell() -> None:
    app = CayuApp(task_store=InMemoryTaskStore())
    app.register_provider(OneShotProvider(), default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))
    client = TestClient(
        create_server(
            app,
            config=ServerConfig.protected(
                BasicAuth(username="operator", password="secret-password")
            ),
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
        access=AuthenticatedAccess(
            dependency=BasicAuth(username="operator", password="secret-password")
        ),
    )
    client = TestClient(server)

    assert client.get("/cayu/api/health").json() == {"ok": True}

    redirect = client.get("/cayu", follow_redirects=False)
    assert redirect.status_code == 307

    denied_redirect_target = client.get(redirect.headers["location"])
    assert denied_redirect_target.status_code == 401

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

    denied_deep_link = client.get("/cayu/sessions/example")
    assert denied_deep_link.status_code == 401
    denied_asset = client.get("/cayu/assets/missing.js")
    assert denied_asset.status_code == 401


def test_custom_auth_dependency_may_return_mapping_context() -> None:
    def mapping_auth(request: Request) -> dict:
        if request.headers.get("Authorization") != f"Bearer {_TOKEN}":
            raise HTTPException(status_code=401, detail="Missing or invalid credentials.")
        return {"subject": "custom-user", "tenant": "tenant-b", "claims": {"issuer": "jwt"}}

    app = CayuApp(task_store=InMemoryTaskStore())
    app.register_provider(OneShotProvider(), default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))
    client = TestClient(create_server(app, config=ServerConfig.protected(mapping_auth)))

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
    client = TestClient(create_server(app, config=ServerConfig.protected(async_auth)))

    assert client.get("/api/sessions").status_code == 401
    assert client.get("/api/sessions", headers=_AUTH_HEADERS).status_code == 200


def test_local_development_server_keeps_routes_open() -> None:
    app = CayuApp(task_store=InMemoryTaskStore())
    app.register_provider(OneShotProvider(), default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))
    client = TestClient(create_server(app, config=ServerConfig.local_development()))

    with client.stream("POST", "/api/run", json={"prompt": "hello"}) as response:
        assert response.status_code == 200
        list(response.iter_lines())

    sessions = asyncio.run(app.session_store.list_sessions())
    assert len(sessions.sessions) == 1
    invocation = sessions.sessions[0].invocation
    assert invocation.source is SessionExecutionSource.HTTP_RUN
    assert invocation.origin.trust is InvocationOriginTrust.UNATTRIBUTED


def test_authenticated_run_persists_only_the_verified_root_identity() -> None:
    app = CayuApp(task_store=InMemoryTaskStore())
    app.register_provider(OneShotProvider(), default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))
    client = TestClient(create_server(app, config=ServerConfig.protected(_require_bearer_token)))

    forged = client.post(
        "/api/run",
        headers=_AUTH_HEADERS,
        json={
            "prompt": "hello",
            "invocation_origin": {
                "subject": "forged-user",
                "tenant": "forged-tenant",
            },
        },
    )
    assert forged.status_code == 422
    assert asyncio.run(app.session_store.list_sessions()).sessions == []

    with client.stream(
        "POST",
        "/api/run",
        headers=_AUTH_HEADERS,
        json={
            "prompt": "hello",
            "session_id": "unknown-field-compatible",
            "future_client_hint": "ignored as before",
        },
    ) as response:
        assert response.status_code == 200
        list(response.iter_lines())

    with client.stream(
        "POST",
        "/api/run",
        headers=_AUTH_HEADERS,
        json={"prompt": "hello", "session_id": "verified-http-root"},
    ) as response:
        assert response.status_code == 200
        list(response.iter_lines())

    sessions = asyncio.run(app.session_store.list_sessions())
    assert len(sessions.sessions) == 2
    session = asyncio.run(app.session_store.load("verified-http-root"))
    assert session is not None
    assert session.invocation.source is SessionExecutionSource.HTTP_RUN
    assert session.invocation.origin.trust is InvocationOriginTrust.SERVER_VERIFIED
    assert session.invocation.origin.subject == "test-user"
    assert session.invocation.origin.tenant == "tenant-a"
    detail = client.get(f"/api/sessions/{session.id}", headers=_AUTH_HEADERS)
    assert detail.status_code == 200
    assert detail.json()["invocation"] == {
        "schema_version": 1,
        "origin": {
            "trust": "server_verified",
            "subject": "test-user",
            "tenant": "tenant-a",
        },
        "root_invocation_id": str(session.invocation.root_invocation_id),
        "root_session_id": session.id,
        "source": "http_run",
    }
    listed = client.get("/api/sessions", headers=_AUTH_HEADERS)
    assert listed.status_code == 200
    assert "invocation" not in listed.json()["sessions"][0]
    summarized = client.post(
        "/api/sessions/summary",
        headers=_AUTH_HEADERS,
        json={},
    )
    assert summarized.status_code == 200
    assert all("invocation" not in item["session"] for item in summarized.json()["sessions"])
    updated = client.patch(
        f"/api/sessions/{session.id}/labels",
        headers=_AUTH_HEADERS,
        json={"labels": {"reviewed": "true"}},
    )
    assert updated.status_code == 200
    assert "invocation" not in updated.json()


def test_authenticated_run_rejects_identity_that_cannot_be_persisted() -> None:
    def invalid_durable_identity(_request: Request) -> AuthContext:
        return AuthContext(subject="invalid\x00identity")

    app = CayuApp(task_store=InMemoryTaskStore())
    app.register_provider(OneShotProvider(), default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))
    client = TestClient(
        create_server(
            app,
            config=ServerConfig.protected(invalid_durable_identity),
        ),
        raise_server_exceptions=False,
    )

    response = client.post("/api/run", json={"prompt": "hello"})

    assert response.status_code == 400
    assert response.json() == {
        "detail": "Authenticated identity is not valid durable invocation provenance."
    }
    assert asyncio.run(app.session_store.list_sessions()).sessions == []


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
        yield Event(
            type=EventType.SESSION_RESUMED,
            session_id=request.session_id,
            agent_name="assistant",
        )

    app.resolve_tool_approval = resolve_tool_approval
    return app, captured


def test_authenticated_resolution_derives_resolved_by_from_auth_context() -> None:
    from cayu import ResolutionActorSource

    app, captured = _approval_capture_app()
    client = TestClient(create_server(app, config=ServerConfig.protected(_require_bearer_token)))

    with client.stream(
        "POST",
        "/api/tool-approvals/resolve",
        headers=_AUTH_HEADERS,
        json={
            "session_id": "session_actor",
            "approval_id": "approval_1",
            "tool_round_id": "round_1",
            "tool_call_id": "call_1",
            "decision": "approve",
        },
    ) as response:
        assert response.status_code == 200
        list(response.iter_lines())

    actor = captured[0].resolved_by
    assert actor is not None
    assert actor.subject == "test-user"
    assert actor.tenant == "tenant-a"
    assert actor.source is ResolutionActorSource.HTTP_AUTH
    assert actor.claims == {"scheme": "bearer"}


def test_authenticated_resolution_reserved_auth_subject_returns_400() -> None:
    def reserved_subject_auth(request: Request) -> AuthContext:
        if request.headers.get("Authorization") != f"Bearer {_TOKEN}":
            raise HTTPException(status_code=401, detail="Missing or invalid credentials.")
        return AuthContext(subject="cayu:ops", claims={"scheme": "bearer"})

    app, captured = _approval_capture_app()
    client = TestClient(create_server(app, config=ServerConfig.protected(reserved_subject_auth)))

    response = client.post(
        "/api/tool-approvals/resolve",
        headers=_AUTH_HEADERS,
        json={
            "session_id": "session_actor",
            "approval_id": "approval_1",
            "tool_round_id": "round_1",
            "tool_call_id": "call_1",
            "decision": "approve",
        },
    )

    assert response.status_code == 400
    assert "reserved" in response.json()["detail"]
    assert captured == []


def test_authenticated_resolution_rejects_body_resolved_by() -> None:
    app, captured = _approval_capture_app()
    client = TestClient(create_server(app, config=ServerConfig.protected(_require_bearer_token)))

    response = client.post(
        "/api/tool-approvals/resolve",
        headers=_AUTH_HEADERS,
        json={
            "session_id": "session_actor",
            "approval_id": "approval_1",
            "tool_round_id": "round_1",
            "tool_call_id": "call_1",
            "decision": "approve",
            "resolved_by": {"subject": "someone-else"},
        },
    )

    assert response.status_code == 400
    assert "derived from the authenticated caller" in response.json()["detail"]
    assert captured == []


def _resume_capture_app() -> tuple[CayuApp, list]:
    from cayu import Message, ResumeRequest, RunRequest, SessionIdentity, SessionStatus

    app = CayuApp()
    app.register_provider(OneShotProvider(), default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    async def create_completed_session() -> None:
        await app.session_store.create(
            RunRequest(
                agent_name="assistant",
                session_id="session_profile_adoption_actor",
                messages=[Message.text("user", "hello")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        await app.session_store.update_status(
            "session_profile_adoption_actor",
            SessionStatus.COMPLETED,
        )

    asyncio.run(create_completed_session())
    captured: list[ResumeRequest] = []

    async def resume(request: ResumeRequest):
        captured.append(request)
        yield Event(
            type=EventType.SESSION_RESUMED,
            session_id=request.session_id,
            agent_name="assistant",
        )

    app.resume = resume  # type: ignore[method-assign]
    return app, captured


def test_authenticated_profile_adoption_derives_actor_from_auth_context() -> None:
    from cayu import ResolutionActorSource

    app, captured = _resume_capture_app()
    client = TestClient(create_server(app, config=ServerConfig.protected(_require_bearer_token)))

    with client.stream(
        "POST",
        "/api/resume",
        headers=_AUTH_HEADERS,
        json={
            "session_id": "session_profile_adoption_actor",
            "prompt": "resume",
            "profile_adoption": {
                "idempotency_key": "authenticated-profile-adoption-v1",
                "reason": "Deploy the reviewed execution profile.",
            },
        },
    ) as response:
        assert response.status_code == 200
        list(response.iter_lines())

    intent = captured[0].profile_adoption
    assert intent is not None
    assert intent.requested_by.subject == "test-user"
    assert intent.requested_by.tenant == "tenant-a"
    assert intent.requested_by.source is ResolutionActorSource.HTTP_AUTH
    assert intent.requested_by.claims == {"scheme": "bearer"}


def test_authenticated_profile_adoption_rejects_body_actor() -> None:
    app, captured = _resume_capture_app()
    client = TestClient(create_server(app, config=ServerConfig.protected(_require_bearer_token)))

    response = client.post(
        "/api/resume",
        headers=_AUTH_HEADERS,
        json={
            "session_id": "session_profile_adoption_actor",
            "prompt": "resume",
            "profile_adoption": {
                "idempotency_key": "spoofed-profile-adoption-v1",
                "reason": "Attempt to spoof the operator.",
                "requested_by": {"subject": "someone-else"},
            },
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
    client = TestClient(create_server(app, config=ServerConfig.protected(_require_bearer_token)))

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
    assert actor.tenant == "tenant-a"
    assert actor.source is ResolutionActorSource.HTTP_AUTH
    assert actor.claims == {"scheme": "bearer"}


def test_authenticated_interruption_rejects_body_requested_by() -> None:
    app, captured = _interrupt_capture_app()
    client = TestClient(create_server(app, config=ServerConfig.protected(_require_bearer_token)))

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

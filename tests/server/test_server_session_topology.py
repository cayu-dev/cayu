from __future__ import annotations

# ruff: noqa: E402
import asyncio
import json

import pytest

fastapi = pytest.importorskip("fastapi")
pytest.importorskip("sse_starlette")

from fastapi.testclient import TestClient

from cayu import (
    REDACTED_SECRET,
    CayuApp,
    Message,
    SecretRedactor,
    default_price_book,
)
from cayu.core import Event, EventType
from cayu.runtime import (
    InMemorySessionStore,
    RunRequest,
    SessionIdentity,
)
from cayu.runtime.sessions import SessionTopologyQuery
from cayu.server import ServerConfig, create_server
from cayu.server import routes as server_routes
from cayu.server.contracts import MAX_SESSION_TOPOLOGY_REQUEST_BYTES


def _identity() -> SessionIdentity:
    return SessionIdentity(provider_name="fake", model="fake-model")


async def _create_session(
    store: InMemorySessionStore,
    session_id: str,
    *,
    parent_session_id: str | None = None,
    causal_budget_id: str = "topology-budget",
) -> None:
    await store.create(
        RunRequest(
            agent_name="assistant",
            session_id=session_id,
            parent_session_id=parent_session_id,
            causal_budget_id=causal_budget_id,
            labels={"private": "label"},
            metadata={"private": "metadata"},
            messages=[Message.text("user", "private prompt")],
        ),
        identity=_identity(),
    )


def _client(app: CayuApp) -> TestClient:
    return TestClient(create_server(app, config=ServerConfig.local_development()))


def test_session_topology_endpoint_pages_batched_branches_and_is_private() -> None:
    store = InMemorySessionStore()

    async def seed() -> None:
        await _create_session(store, "root")
        await _create_session(store, "focus", parent_session_id="root")
        await _create_session(store, "child-a", parent_session_id="focus")
        await _create_session(store, "child-b", parent_session_id="focus")

    asyncio.run(seed())
    client = _client(CayuApp(session_store=store, enable_logging=False))

    first = client.post(
        "/api/sessions/focus/topology",
        json={
            "expanded_parent_ids": ["root", "focus"],
            "child_limit": 1,
        },
    )

    assert first.status_code == 200
    assert first.headers["cache-control"] == "private, no-store"
    body = first.json()
    assert body["scope"] == "session_focus"
    assert body["focus"]["id"] == "focus"
    assert [node["id"] for node in body["ancestors"]] == ["root"]
    assert [node["id"] for node in body["expanded_parents"]] == ["root", "focus"]
    assert [branch["parent_session_id"] for branch in body["branches"]] == [
        "root",
        "focus",
    ]
    assert [node["id"] for node in body["branches"][0]["children"]] == ["focus"]
    focus_branch = body["branches"][1]
    assert [node["id"] for node in focus_branch["children"]] == ["child-a"]
    assert focus_branch["has_more"] is True
    assert isinstance(focus_branch["next_cursor"], str)
    rendered = json.dumps(body, sort_keys=True)
    assert "private prompt" not in rendered
    assert '"metadata"' not in rendered
    assert '"labels"' not in rendered

    continuation = client.post(
        "/api/sessions/focus/topology",
        json={
            "expanded_parent_ids": ["focus"],
            "child_cursors": {"focus": focus_branch["next_cursor"]},
            "child_limit": 1,
        },
    )
    assert continuation.status_code == 200
    continuation_branch = continuation.json()["branches"][0]
    assert [node["id"] for node in continuation_branch["children"]] == ["child-b"]
    assert continuation_branch["has_more"] is False
    assert continuation_branch["next_cursor"] is None

    contract = client.get("/api/contract")
    assert contract.status_code == 200
    workflow = contract.json()["capabilities"]["surfaces"]["workflow"]
    assert workflow["configured"] is True
    assert workflow["read"]["enabled"] is True
    assert workflow["mutate"]["enabled"] is False


def test_session_topology_endpoint_reports_missing_depth_and_cursor_failures() -> None:
    store = InMemorySessionStore()

    async def seed() -> None:
        await _create_session(store, "root")
        await _create_session(store, "parent", parent_session_id="root")
        await _create_session(store, "focus", parent_session_id="parent")
        await _create_session(store, "child-a", parent_session_id="focus")
        await _create_session(store, "child-b", parent_session_id="focus")

    asyncio.run(seed())
    client = _client(CayuApp(session_store=store, enable_logging=False))

    missing = client.post("/api/sessions/missing/topology", json={})
    assert missing.status_code == 404
    assert missing.headers["cache-control"] == "private, no-store"

    over_depth = client.post(
        "/api/sessions/focus/topology",
        json={"ancestor_depth_limit": 1},
    )
    assert over_depth.status_code == 413
    assert over_depth.headers["cache-control"] == "private, no-store"

    first = client.post(
        "/api/sessions/focus/topology",
        json={"expanded_parent_ids": ["focus"], "child_limit": 1},
    )
    cursor = first.json()["branches"][0]["next_cursor"]
    wrong_parent = client.post(
        "/api/sessions/focus/topology",
        json={
            "expanded_parent_ids": ["root"],
            "child_cursors": {"root": cursor},
        },
    )
    assert wrong_parent.status_code == 422
    assert wrong_parent.headers["cache-control"] == "private, no-store"


def test_session_topology_endpoint_fails_closed_for_unsupported_store() -> None:
    class UnsupportedTopologyStore(InMemorySessionStore):
        supports_session_topology = False

        async def query_session_topology(self, query: SessionTopologyQuery):
            del query
            raise NotImplementedError

    store = UnsupportedTopologyStore()
    asyncio.run(_create_session(store, "focus"))
    client = _client(CayuApp(session_store=store, enable_logging=False))

    contract = client.get("/api/contract")
    workflow = contract.json()["capabilities"]["surfaces"]["workflow"]
    assert workflow["configured"] is True
    assert workflow["read"]["enabled"] is False

    response = client.post("/api/sessions/focus/topology", json={})
    assert response.status_code == 501
    assert response.headers["cache-control"] == "private, no-store"


def test_session_topology_rejects_and_sanitizes_invalid_or_oversized_requests() -> None:
    secret = "invalid-topology-workload-secret"
    store = InMemorySessionStore()
    asyncio.run(_create_session(store, "focus"))
    client = _client(
        CayuApp(
            session_store=store,
            secret_redactor=SecretRedactor(secret),
            enable_logging=False,
        )
    )

    invalid = client.post(
        "/api/sessions/focus/topology",
        json={
            "expanded_parent_ids": [
                secret,
                *(f"parent-{index}" for index in range(50)),
            ]
        },
    )
    assert invalid.status_code == 422
    assert invalid.json() == {"detail": "Invalid session topology request."}
    assert invalid.headers["cache-control"] == "private, no-store"
    assert secret not in invalid.text
    assert REDACTED_SECRET not in invalid.text

    oversized_cursor = client.post(
        "/api/sessions/focus/topology",
        json={
            "expanded_parent_ids": ["focus"],
            "child_cursors": {"focus": "x" * 4097},
        },
    )
    assert oversized_cursor.status_code == 422
    assert oversized_cursor.json() == {"detail": "Invalid session topology request."}
    assert oversized_cursor.headers["cache-control"] == "private, no-store"

    malformed_unicode = client.post(
        "/api/sessions/focus/topology",
        content='{"expanded_parent_ids":["\\ud800"]}',
        headers={"Content-Type": "application/json"},
    )
    assert malformed_unicode.status_code == 422
    assert malformed_unicode.json() == {"detail": "Invalid session topology request."}
    assert malformed_unicode.headers["cache-control"] == "private, no-store"

    oversized_body = json.dumps(
        {
            "expanded_parent_ids": ["focus"],
            "padding": "x" * MAX_SESSION_TOPOLOGY_REQUEST_BYTES,
        }
    )
    oversized = client.post(
        "/api/sessions/focus/topology",
        content=oversized_body,
        headers={"Content-Type": "application/json"},
    )
    assert oversized.status_code == 413
    assert oversized.json() == {"detail": "Session topology request exceeds the server byte limit."}
    assert oversized.headers["cache-control"] == "private, no-store"

    def oversized_chunks():
        yield b'{"expanded_parent_ids":["focus"],"padding":"'
        yield b"x" * MAX_SESSION_TOPOLOGY_REQUEST_BYTES
        yield b'"}'

    oversized_stream = client.post(
        "/api/sessions/focus/topology",
        content=oversized_chunks(),
        headers={"Content-Type": "application/json"},
    )
    assert oversized_stream.status_code == 413
    assert oversized_stream.json() == {
        "detail": "Session topology request exceeds the server byte limit."
    }
    assert oversized_stream.headers["cache-control"] == "private, no-store"


def test_session_topology_response_enforces_serialized_byte_ceiling() -> None:
    store = InMemorySessionStore()

    async def seed() -> None:
        await _create_session(store, "focus-" + "x" * 400)
        await _create_session(
            store,
            "child-" + "y" * 400,
            parent_session_id="focus-" + "x" * 400,
        )

    asyncio.run(seed())
    focus_id = "focus-" + "x" * 400
    client = _client(CayuApp(session_store=store, enable_logging=False))

    response = client.post(
        f"/api/sessions/{focus_id}/topology",
        json={"max_result_bytes": 1024},
    )

    assert response.status_code == 413
    assert response.json()["detail"].startswith("Session topology exceeds max_result_bytes")
    assert response.headers["cache-control"] == "private, no-store"


def test_session_topology_rejects_redaction_that_would_collapse_node_identity() -> None:
    secrets = ("topology-alpha-secret", "topology-beta-secret")
    store = InMemorySessionStore()

    async def seed() -> None:
        await _create_session(store, "focus")
        for secret in secrets:
            await _create_session(store, f"child-{secret}", parent_session_id="focus")

    asyncio.run(seed())
    client = _client(
        CayuApp(
            session_store=store,
            secret_redactor=SecretRedactor(secrets),
            enable_logging=False,
        )
    )

    response = client.post(
        "/api/sessions/focus/topology",
        json={"child_limit": 1},
    )

    assert response.status_code == 409
    rendered = json.dumps(response.json(), sort_keys=True)
    assert all(secret not in rendered for secret in secrets)
    assert REDACTED_SECRET not in rendered
    assert "identity cannot cross" in response.json()["detail"]
    assert response.headers["cache-control"] == "private, no-store"


def test_session_topology_corruption_does_not_echo_durable_identifiers() -> None:
    secret = "topology-lineage-secret"
    store = InMemorySessionStore()

    async def seed() -> None:
        await _create_session(store, f"root-{secret}")
        await _create_session(
            store,
            "focus",
            parent_session_id=f"root-{secret}",
        )

    asyncio.run(seed())
    store._sessions[f"root-{secret}"].parent_session_id = f"root-{secret}"
    client = _client(
        CayuApp(
            session_store=store,
            secret_redactor=SecretRedactor(secret),
            enable_logging=False,
        )
    )

    response = client.post("/api/sessions/focus/topology", json={})

    assert response.status_code == 409
    rendered = json.dumps(response.json(), sort_keys=True)
    assert secret not in rendered
    assert REDACTED_SECRET not in rendered
    assert "durable ancestry contains a cycle" in response.json()["detail"]
    assert response.headers["cache-control"] == "private, no-store"


def test_causal_budget_summary_rejects_session_and_event_overflow(monkeypatch) -> None:
    store = InMemorySessionStore()

    async def seed() -> None:
        await _create_session(store, "budget-a", causal_budget_id="bounded-budget")
        await _create_session(store, "budget-b", causal_budget_id="bounded-budget")
        await store.append_events(
            "budget-a",
            [
                Event(
                    id="budget-event-1",
                    type=EventType.SESSION_STARTED,
                    session_id="budget-a",
                ),
                Event(
                    id="budget-event-2",
                    type=EventType.SESSION_COMPLETED,
                    session_id="budget-a",
                ),
            ],
        )

    asyncio.run(seed())
    client = _client(CayuApp(session_store=store, enable_logging=False))
    pricing = default_price_book().model_dump(mode="json")

    monkeypatch.setattr(server_routes, "_CAUSAL_BUDGET_SUMMARY_MAX_SESSIONS", 1)
    sessions_overflow = client.post(
        "/api/causal-budgets/bounded-budget/summary",
        json={"pricing": pricing},
    )
    assert sessions_overflow.status_code == 413
    assert "session safety limit" in sessions_overflow.json()["detail"]

    monkeypatch.setattr(server_routes, "_CAUSAL_BUDGET_SUMMARY_MAX_SESSIONS", 2)
    monkeypatch.setattr(server_routes, "_CAUSAL_BUDGET_SUMMARY_MAX_EVENTS", 1)
    events_overflow = client.post(
        "/api/causal-budgets/bounded-budget/summary",
        json={"pricing": pricing},
    )
    assert events_overflow.status_code == 413
    assert "event safety limit" in events_overflow.json()["detail"]

    monkeypatch.setattr(server_routes, "_CAUSAL_BUDGET_SUMMARY_MAX_EVENTS", 2)
    monkeypatch.setattr(server_routes, "_CAUSAL_BUDGET_SUMMARY_MAX_RESULT_BYTES", 1)
    response_overflow = client.post(
        "/api/causal-budgets/bounded-budget/summary",
        json={"pricing": pricing},
    )
    assert response_overflow.status_code == 413
    assert "max_result_bytes" in response_overflow.json()["detail"]


def test_causal_budget_summary_rejects_event_bytes_before_processing(monkeypatch) -> None:
    store = InMemorySessionStore()

    async def seed() -> None:
        await _create_session(store, "budget-a", causal_budget_id="bounded-budget")
        await store.append_events(
            "budget-a",
            [
                Event(
                    id="large-event",
                    type=EventType.SESSION_STARTED,
                    session_id="budget-a",
                    payload={"irrelevant": "x" * 4096},
                )
            ],
        )

    asyncio.run(seed())
    client = _client(CayuApp(session_store=store, enable_logging=False))
    monkeypatch.setattr(
        server_routes,
        "_CAUSAL_BUDGET_SUMMARY_MAX_EVENT_INPUT_BYTES",
        1024,
    )

    response = client.post(
        "/api/causal-budgets/bounded-budget/summary",
        json={"pricing": default_price_book().model_dump(mode="json")},
    )

    assert response.status_code == 413
    assert "event-input safety limit" in response.json()["detail"]


def test_causal_budget_summary_fails_closed_without_bounded_event_reads() -> None:
    class UnboundedEventStore(InMemorySessionStore):
        async def query_events_bounded(self, query, *, max_bytes):
            del query, max_bytes
            raise NotImplementedError

    store = UnboundedEventStore()
    asyncio.run(_create_session(store, "budget-a", causal_budget_id="bounded-budget"))
    client = _client(CayuApp(session_store=store, enable_logging=False))

    response = client.post(
        "/api/causal-budgets/bounded-budget/summary",
        json={"pricing": default_price_book().model_dump(mode="json")},
    )

    assert response.status_code == 501
    assert "cannot enforce byte-bounded" in response.json()["detail"]

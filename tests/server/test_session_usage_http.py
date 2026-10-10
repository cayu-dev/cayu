from __future__ import annotations

import asyncio

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("sse_starlette")

from fastapi.testclient import TestClient

from cayu.applications import CayuApp
from cayu.events import Event, EventType
from cayu.messages import Message
from cayu.runtime._usage_accounting import UsageAccountingReducer
from cayu.server import ServerConfig, create_server
from cayu.sessions.base import InMemorySessionStore
from cayu.sessions.records import SessionIdentity
from cayu.sessions.requests import RunRequest


def _completed(session_id: str, tokens: int) -> Event:
    return Event(
        type=EventType.MODEL_COMPLETED,
        session_id=session_id,
        payload={
            "usage_metrics": {
                "provider_name": "fake",
                "model": "fake-model",
                "input_tokens": tokens,
                "output_tokens": 1,
                "total_tokens": tokens + 1,
                "cache": {"read_tokens": 2, "write_tokens": 1},
            }
        },
    )


def _seeded(*session_ids: str) -> tuple[InMemorySessionStore, TestClient]:
    store = InMemorySessionStore()

    async def seed() -> None:
        for number, session_id in enumerate(session_ids, 1):
            await store.create(
                RunRequest(
                    session_id=session_id,
                    agent_name="assistant",
                    messages=[Message.text("user", "hi")],
                ),
                identity=SessionIdentity(provider_name="fake", model="fake-model"),
            )
            await store.append_events(
                session_id, [_completed(session_id, number * 10) for _ in range(number)]
            )

    asyncio.run(seed())
    app = CayuApp(session_store=store, enable_logging=False)
    return store, TestClient(create_server(app, config=ServerConfig.local_development()))


def test_session_usage_returns_etag_and_not_modified_until_usage_changes() -> None:
    store, client = _seeded("one", "other")

    first = client.get("/api/sessions/one/usage")
    assert first.status_code == 200
    etag = first.headers["etag"]
    assert etag.startswith('"usage-') and etag.endswith('"')
    assert first.headers["cache-control"] == "private, no-cache"
    assert first.json()["usage"]["input_tokens"] == "10"

    for header in (etag, f"W/{etag}", f'"stale", {etag}', "*"):
        cached = client.get("/api/sessions/one/usage", headers={"If-None-Match": header})
        assert cached.status_code == 304
        assert cached.headers["etag"] == etag
        assert cached.content == b""
    stale = client.get("/api/sessions/one/usage", headers={"If-None-Match": '"stale"'})
    assert stale.status_code == 200
    assert stale.json() == first.json()

    # Unrelated events do not change usage or its validator.
    asyncio.run(store.append_event("one", Event(type=EventType.MODEL_STARTED, session_id="one")))
    assert client.get("/api/sessions/one/usage", headers={"If-None-Match": etag}).status_code == 304

    asyncio.run(store.append_event("one", _completed("one", 5)))
    changed = client.get("/api/sessions/one/usage", headers={"If-None-Match": etag})
    assert changed.status_code == 200
    assert changed.headers["etag"] != etag
    assert changed.json()["usage"]["input_tokens"] == "15"

    # Deleting accounting evidence anywhere advances the generation.
    asyncio.run(store.delete_session("other"))
    regenerated = client.get(
        "/api/sessions/one/usage", headers={"If-None-Match": changed.headers["etag"]}
    )
    assert regenerated.status_code == 200
    assert regenerated.headers["etag"] != changed.headers["etag"]
    assert regenerated.json() == changed.json()

    assert client.get("/api/sessions/missing/usage").status_code == 404


def test_session_list_includes_usage_without_rescanning_history(monkeypatch) -> None:
    _store, client = _seeded("a", "b", "c")

    plain = client.get("/api/sessions", params={"order_by": "created_at_asc"})
    assert plain.status_code == 200
    assert plain.json()["usage"] is None

    included = client.get(
        "/api/sessions", params={"include": "usage", "order_by": "created_at_asc"}
    )
    assert included.status_code == 200
    body = included.json()
    assert [row["session_id"] for row in body["usage"]] == [row["id"] for row in body["sessions"]]
    for row in body["usage"]:
        assert row == client.get(f"/api/sessions/{row['session_id']}/usage").json()
    assert [row["model_steps"] for row in body["usage"]] == [1, 2, 3]

    hydrated: list[int] = []
    add_page = UsageAccountingReducer.add_page

    def observed(self, records):
        hydrated.append(len(records))
        return add_page(self, records)

    monkeypatch.setattr(UsageAccountingReducer, "add_page", observed)
    again = client.get("/api/sessions", params={"include": "usage", "order_by": "created_at_asc"})
    assert again.json()["usage"] == body["usage"]
    assert hydrated == []

    assert client.get("/api/sessions", params={"include": "unknown"}).status_code == 422


def test_contract_advertises_usage_and_cost_endpoints() -> None:
    _store, client = _seeded()
    accounting = client.get("/api/contract").json()["accounting"]
    assert accounting == {
        "usage": {
            "supported": True,
            "method": "GET",
            "path": "/api/sessions/{session_id}/usage",
            "response_schema": "SessionUsageSummary",
            "etag_header": "ETag",
            "conditional_request_header": "If-None-Match",
            "not_modified_status": 304,
            "session_list_include": "usage",
        },
        "cost": {
            "supported": True,
            "method": "POST",
            "path": "/api/sessions/{session_id}/cost",
            "request_schema": "SessionCostBody",
            "response_schema": "SessionCostSummary",
        },
    }
    schemas = client.get("/openapi.json").json()["components"]["schemas"]
    assert {"SessionUsageSummary", "SessionCostBody", "SessionCostSummary"} <= set(schemas)

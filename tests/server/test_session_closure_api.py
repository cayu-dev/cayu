"""HTTP deletion uses dependent-first closure rather than native-row deletion."""

import asyncio

from fastapi.testclient import TestClient

from cayu import CayuApp, RunRequest
from cayu.artifacts import ArtifactListResult
from cayu.runtime.session_closure import (
    ArtifactSessionClosureStore,
    SessionClosureDisposition,
    SessionClosureRecord,
)
from cayu.server import ServerConfig, create_server
from cayu.sessions.memory import InMemorySessionStore
from cayu.sessions.records import SessionIdentity


def test_http_delete_retries_dependent_failure_before_native_deletion():
    store = InMemorySessionStore()

    class Dependent:
        store_id = "dependent"
        calls = 0
        fail = True
        unavailable = True

        async def inspect_session_closure(self, session_id, *, policy):
            return SessionClosureRecord(
                store_id=self.store_id,
                record_class="records",
                disposition=(
                    SessionClosureDisposition.UNAVAILABLE
                    if self.unavailable
                    else SessionClosureDisposition.OWNED_ELIGIBLE
                ),
            )

        async def export_session_closure(self, session_id, *, policy):
            return {"records": []}

        async def erase_session_closure(self, session_id, *, policy, plan_id):
            assert await store.load(session_id) is not None
            self.calls += 1
            if self.fail:
                raise OSError("private-dependent-failure-canary")
            return SessionClosureRecord(
                store_id=self.store_id,
                record_class="records",
                disposition=SessionClosureDisposition.ERASED,
            )

    dependent = Dependent()
    app = CayuApp(session_store=store, enable_logging=False, session_closure_stores=(dependent,))
    asyncio.run(
        store.create(
            RunRequest(session_id="closure-http", agent_name="worker", messages=[]),
            identity=SessionIdentity(provider_name="scripted", model="test"),
        )
    )
    client = TestClient(create_server(app, config=ServerConfig.local_development()))
    response = client.delete("/api/sessions/closure-http")
    assert response.status_code == 409
    assert dependent.calls == 0
    dependent.unavailable = False
    response = client.delete("/api/sessions/closure-http")
    assert response.status_code == 409
    assert "private-dependent-failure-canary" not in response.text
    assert dependent.calls == 1
    assert asyncio.run(store.load("closure-http")) is not None
    dependent.fail = False
    assert client.delete("/api/sessions/closure-http").status_code == 204
    assert dependent.calls == 2
    assert asyncio.run(store.load("closure-http")) is None
    assert client.delete("/api/sessions/closure-http").status_code == 204
    assert dependent.calls == 2


def test_deleted_public_alias_remains_absent_after_receipt_eviction():
    store = InMemorySessionStore(closure_receipt_limit=1)
    app = CayuApp(session_store=store, enable_logging=False)
    client = TestClient(create_server(app, config=ServerConfig.local_development()))

    async def create(session_id):
        await store.create(
            RunRequest(session_id=session_id, agent_name="worker", messages=[]),
            identity=SessionIdentity(provider_name="scripted", model="test"),
        )
        alias = store.public_authority_alias_codec.encode(session_id, field_name="session_id")
        await store.register_public_authority_alias(
            alias,
            field_name="session_id",
            private_value=session_id,
        )
        return alias

    alias = asyncio.run(create("first"))
    assert client.get(f"/api/sessions/{alias}").status_code == 200
    assert client.delete(f"/api/sessions/{alias}").status_code == 204
    other = asyncio.run(create("second"))
    assert client.delete(f"/api/sessions/{other}").status_code == 204
    assert not store._public_authority_aliases
    assert client.delete(f"/api/sessions/{alias}").status_code == 204
    assert client.get(f"/api/sessions/{alias}").status_code == 404
    malformed = "cayu_authority_invalid"
    assert client.delete(f"/api/sessions/{malformed}").status_code == 409
    assert client.get(f"/api/sessions/{malformed}").status_code == 409


def test_http_delete_refuses_an_artifact_store_that_cannot_fence_closure():
    class UnfencedArtifacts:
        id = "unfenced-artifacts"

        async def list(self, **kwargs):
            return ArtifactListResult(artifacts=(), total_count=0)

    store = InMemorySessionStore()
    app = CayuApp(
        session_store=store,
        enable_logging=False,
        session_closure_stores=(ArtifactSessionClosureStore(UnfencedArtifacts()),),
    )
    asyncio.run(
        store.create(
            RunRequest(session_id="unfenced-http", agent_name="worker", messages=[]),
            identity=SessionIdentity(provider_name="scripted", model="test"),
        )
    )
    client = TestClient(create_server(app, config=ServerConfig.local_development()))
    for _attempt in range(2):
        response = client.delete("/api/sessions/unfenced-http")
        assert response.status_code == 409
        assert response.json() == {"detail": "Artifact store cannot fence session closure."}
    # The refused deletions left no lineage claim behind.
    asyncio.run(store.delete_session("unfenced-http"))
    assert asyncio.run(store.load("unfenced-http")) is None


def test_http_delete_answers_conflict_for_an_unsupported_closure(monkeypatch):
    store = InMemorySessionStore()
    app = CayuApp(session_store=store, enable_logging=False)
    asyncio.run(
        store.create(
            RunRequest(session_id="unsupported-http", agent_name="worker", messages=[]),
            identity=SessionIdentity(provider_name="scripted", model="test"),
        )
    )

    async def unsupported(session_id, *, finished_only):
        raise NotImplementedError("Closure is unsupported by this store.")

    monkeypatch.setattr(app, "delete_session", unsupported)
    client = TestClient(create_server(app, config=ServerConfig.local_development()))
    response = client.delete("/api/sessions/unsupported-http")
    assert response.status_code == 409
    assert response.json() == {"detail": "Closure is unsupported by this store."}

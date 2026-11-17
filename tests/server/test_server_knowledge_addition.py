"""Adding a knowledge entry over HTTP (`POST /api/knowledge`)."""

from __future__ import annotations

import asyncio

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("sse_starlette")
from fastapi.testclient import TestClient

from cayu import CayuApp
from cayu.server import ServerConfig, create_server
from cayu.storage.memory import (
    InMemoryKnowledgeStore,
    KnowledgeAccessScope,
    KnowledgeStatus,
)

_ORGANIZATIONS = KnowledgeAccessScope(
    allowed_namespaces=["acme-agent"],
    allowed_statuses=[KnowledgeStatus.ACTIVE, KnowledgeStatus.PENDING],
    session_label_keys=("organization",),
)


def _client(store: InMemoryKnowledgeStore, scope: KnowledgeAccessScope) -> TestClient:
    return TestClient(
        create_server(
            CayuApp(knowledge_store=store, knowledge_access_scope=scope, enable_logging=False),
            config=ServerConfig.local_development(),
        )
    )


def test_added_entry_is_kept_and_listed_and_a_repeat_changes_nothing():
    scope = KnowledgeAccessScope.privileged()
    store = InMemoryKnowledgeStore(access_scope=scope)
    request = {
        "text": "Invoices over 10,000 need a second approver.",
        "kind": "policy",
        "title": "Invoice approvals",
        "labels": {"team": "finance"},
    }
    with _client(store, scope) as client:
        added = client.post("/api/knowledge", json=request)
        assert added.status_code == 200, added.text
        body = added.json()
        assert body["created"] is True
        assert body["status"] == "active"
        assert body["kind"] == "policy"
        assert body["title"] == "Invoice approvals"
        assert body["labels"] == {"team": "finance"}
        assert body["namespace"] == "default"
        assert body["created_by_type"] == "user"
        assert body["chunk_count"] == 1
        listed = client.get("/api/knowledge", params={"label": "team=finance"}).json()
        assert [row["entry_id"] for row in listed["entries"]] == [body["entry_id"]]
        assert listed["entries"][0]["attribution_unavailable_reason"] == "not_recorded"

        again = client.post("/api/knowledge", json=request)
        assert again.status_code == 200
        assert again.json()["created"] is False
        assert again.json()["entry_id"] == body["entry_id"]

        # Same text and labels with another title is a different request for the same entry.
        clash = client.post("/api/knowledge", json={**request, "title": "Other"})
        assert clash.status_code == 409
        assert body["entry_id"] in clash.json()["detail"]

        # Other labels make another entry.
        other = client.post("/api/knowledge", json={**request, "labels": {"team": "sales"}})
        assert other.status_code == 200
        assert other.json()["entry_id"] != body["entry_id"]


@pytest.fixture
def signed():
    """A protected server's signing key and a header maker for one person."""

    from cryptography.hazmat.primitives.asymmetric import ec
    from tests.server.test_proxy_identity import FakeJwks, _as, _auth, _token

    key = ec.generate_private_key(ec.SECP256R1())

    def person(subject: str) -> dict[str, str]:
        return _as(_token(key, sub=subject))

    return _auth(FakeJwks(key)), person


def _protected(store, scope, auth) -> TestClient:
    return TestClient(
        create_server(
            CayuApp(knowledge_store=store, knowledge_access_scope=scope, enable_logging=False),
            config=ServerConfig.protected(auth),
        )
    )


def test_pending_entry_waits_for_review_by_someone_else(signed):
    auth, person = signed
    scope = KnowledgeAccessScope.privileged()
    store = InMemoryKnowledgeStore(access_scope=scope)
    ada, grace = person("ada"), person("grace")
    with _protected(store, scope, auth) as client:
        added = client.post(
            "/api/knowledge", headers=ada, json={"text": "Proposal text", "status": "pending"}
        )
        assert added.status_code == 200, added.text
        entry_id = added.json()["entry_id"]
        assert added.json()["status"] == "pending"
        assert added.json()["created_by"] == "ada"
        assert client.get("/api/knowledge", headers=ada).json()["entries"] == []
        pending = client.get("/api/knowledge/pending", headers=ada).json()["entries"]
        assert [row["entry_id"] for row in pending] == [entry_id]
        # The one who proposed it can't keep it too, and is told so.
        own = client.post(f"/api/knowledge/{entry_id}/approve", headers=ada)
        assert own.status_code == 409
        assert own.json()["detail"].startswith("You added this proposal")
        kept = client.post(f"/api/knowledge/{entry_id}/approve", headers=grace)
        assert kept.status_code == 200, kept.text
        assert kept.json()["status"] == "active"

        dropped = client.post(
            "/api/knowledge", headers=ada, json={"text": "Dropped text", "status": "pending"}
        ).json()["entry_id"]
        assert client.post(f"/api/knowledge/{dropped}/reject", headers=ada).status_code == 200
        # A dropped entry isn't added again under its id.
        again = client.post(
            "/api/knowledge", headers=ada, json={"text": "Dropped text", "status": "pending"}
        )
        assert again.status_code == 409
        assert "archived or dropped" in again.json()["detail"]


def test_pending_entry_is_refused_without_authentication():
    scope = KnowledgeAccessScope.privileged()
    store = InMemoryKnowledgeStore(access_scope=scope)
    with _client(store, scope) as client:
        refused = client.post("/api/knowledge", json={"text": "Proposal", "status": "pending"})
        assert refused.status_code == 422
        assert "could never be kept" in refused.json()["detail"]


def test_pending_entry_must_fit_what_keep_approves(signed):
    """A proposal is refused when the activation request that keeping builds (its text and every chunk)
    wouldn't fit; one just inside fits and is kept. Kept entries take the full text."""

    auth, person = signed
    scope = KnowledgeAccessScope.privileged()
    store = InMemoryKnowledgeStore(access_scope=scope)
    words = "invoice approval policy text "

    def text(kib: int) -> str:
        return (words * (kib * 1024 // len(words) + 1))[: kib * 1024]

    with _protected(store, scope, auth) as client:
        fits = client.post(
            "/api/knowledge",
            headers=person("ada"),
            json={"text": text(400), "status": "pending"},
        )
        assert fits.status_code == 200, fits.text
        kept = client.post(
            f"/api/knowledge/{fits.json()['entry_id']}/approve",
            headers={**person("grace"), "Idempotency-Key": "k" * 256},
        )
        assert kept.status_code == 200, kept.text

        too_large = client.post(
            "/api/knowledge",
            headers=person("ada"),
            json={"text": text(500), "status": "pending"},
        )
        assert too_large.status_code == 413
        assert "too long to keep through review" in too_large.json()["detail"]
        active = client.post("/api/knowledge", headers=person("ada"), json={"text": text(500)})
        assert active.status_code == 200, active.text


def test_organization_scoped_knowledge_refuses_an_entry_without_its_organization():
    store = InMemoryKnowledgeStore(access_scope=_ORGANIZATIONS)
    with _client(store, _ORGANIZATIONS) as client:
        refused = client.post("/api/knowledge", json={"text": "Acme pays net 30."})
        assert refused.status_code == 422
        detail = refused.json()["detail"]
        assert "'organization'" in detail and "session_label_keys" in detail

        added = client.post(
            "/api/knowledge",
            json={"text": "Acme pays net 30.", "labels": {"organization": "acme"}},
        )
        assert added.status_code == 200, added.text
        assert added.json()["namespace"] == "acme-agent"
        assert added.json()["labels"] == {"organization": "acme"}
        entry = asyncio.run(store.get_entry(added.json()["entry_id"]))
        assert entry is not None and entry.status is KnowledgeStatus.ACTIVE

        outside = client.post(
            "/api/knowledge",
            json={
                "text": "Acme pays net 30.",
                "namespace": "elsewhere",
                "labels": {"organization": "acme"},
            },
        )
        assert outside.status_code == 403


def test_added_entry_needs_a_namespace_when_the_scope_has_several():
    scope = KnowledgeAccessScope(allowed_namespaces=["one", "two"])
    store = InMemoryKnowledgeStore(access_scope=scope)
    with _client(store, scope) as client:
        refused = client.post("/api/knowledge", json={"text": "Some fact"})
        assert refused.status_code == 422
        assert "one, two" in refused.json()["detail"]
        added = client.post("/api/knowledge", json={"text": "Some fact", "namespace": "two"})
        assert added.status_code == 200
        assert added.json()["namespace"] == "two"


def test_added_entry_refuses_invalid_fields_and_oversized_text():
    scope = KnowledgeAccessScope.privileged()
    store = InMemoryKnowledgeStore(access_scope=scope)
    with _client(store, scope) as client:
        assert client.post("/api/knowledge", json={"text": "   "}).status_code == 422
        assert (
            client.post("/api/knowledge", json={"text": "x", "status": "archived"}).status_code
            == 422
        )
        assert client.post("/api/knowledge", json={"text": "x", "kind": " "}).status_code == 422
        large = client.post("/api/knowledge", json={"text": "x" * (600 * 1024)})
        assert large.status_code in {413, 422}


def test_added_entry_reports_no_store():
    with TestClient(
        create_server(CayuApp(enable_logging=False), config=ServerConfig.local_development())
    ) as client:
        response = client.post("/api/knowledge", json={"text": "Some fact"})
        assert response.status_code == 404
        assert response.json()["detail"] == "Knowledge store is not configured."

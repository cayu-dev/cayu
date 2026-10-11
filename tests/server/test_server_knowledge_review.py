"""HTTP server knowledge review contracts."""

from __future__ import annotations

# ruff: noqa: E402
import pytest

fastapi = pytest.importorskip("fastapi")
pytest.importorskip("sse_starlette")
httpx = pytest.importorskip("httpx")

import asyncio

from fastapi.testclient import TestClient
from tests.server.server_test_support import _LOCAL_SERVER_CONFIG

from cayu.applications import CayuApp
from cayu.server import create_server
from cayu.storage.memory import (
    InMemoryKnowledgeStore,
    KnowledgeAccessScope,
    KnowledgeEntry,
    KnowledgeRevisionConflict,
    KnowledgeStatus,
)


class _TestKnowledgeStore(InMemoryKnowledgeStore):
    def __init__(self, *args, **kwargs) -> None:
        kwargs.setdefault("access_scope", KnowledgeAccessScope.privileged())
        super().__init__(*args, **kwargs)


def test_server_exposes_pending_knowledge_review_endpoints() -> None:
    store = InMemoryKnowledgeStore(
        [
            KnowledgeEntry(
                id="pending_git",
                text="Remote sandbox Git pushes should use a brokered credential proxy.",
                namespace="project:cayu",
                labels={"project": "cayu", "tenant": "trusted"},
                kind="procedure",
                status=KnowledgeStatus.PENDING,
                aspects=["git", "credentials"],
                title="Remote sandbox Git credentials",
                metadata={"review_note": "inspect before approving"},
            ),
            KnowledgeEntry(
                id="active_git",
                text="Active knowledge should not appear in pending review.",
                namespace="project:cayu",
                labels={"project": "cayu", "tenant": "trusted"},
                status=KnowledgeStatus.ACTIVE,
            ),
        ]
    )
    access_scope = KnowledgeAccessScope.privileged()
    app = CayuApp(
        knowledge_store=store,
        knowledge_access_scope=access_scope,
        knowledge_review_namespace="project:cayu",
        knowledge_review_labels={"project": "cayu", "tenant": "trusted"},
    )
    client = TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG))

    pending = client.get("/api/knowledge/pending")
    assert pending.status_code == 200
    body = pending.json()
    assert [entry["entry_id"] for entry in body["entries"]] == ["pending_git"]
    assert body["entries"][0]["revision"] == 1
    assert body["entries"][0]["title"] == "Remote sandbox Git credentials"
    assert body["entries"][0]["text_preview"] == "Remote sandbox Git credentials"
    assert body["total_entries_known"] == 1

    detail = client.get("/api/knowledge/pending/pending_git")
    assert detail.status_code == 200
    detail_body = detail.json()
    assert (
        detail_body["text"] == "Remote sandbox Git pushes should use a brokered credential proxy."
    )
    assert detail_body["metadata"] == {"review_note": "inspect before approving"}
    assert [chunk["chunk_id"] for chunk in detail_body["chunks"]] == ["pending_git:r1:0"]
    assert detail_body["chunks"][0]["entry_revision"] == 1
    assert detail_body["chunks"][0]["text"] == detail_body["text"]

    review_headers = {"Idempotency-Key": "dashboard-review-pending-git"}
    approved = client.post(
        "/api/knowledge/pending_git/approve",
        headers=review_headers,
    )
    assert approved.status_code == 200
    assert approved.json()["status"] == "active"
    assert approved.json()["revision"] == 2

    replay = client.post(
        "/api/knowledge/pending_git/approve",
        headers=review_headers,
    )
    assert replay.status_code == 200
    assert replay.json() == approved.json()
    activation = asyncio.run(
        store.load_activation_receipt(
            "dashboard-review-pending-git",
            access_scope=access_scope,
        )
    )
    assert activation is not None
    assert activation.replayed is False
    assert activation.authority.decision.policy_identity == "cayu:trusted-local-development"
    assert activation.authority.decision.annotations == {
        "channel": "trusted-local-http",
        "identity_source": "local_development",
    }

    empty = client.get("/api/knowledge/pending")
    assert empty.status_code == 200
    assert empty.json()["entries"] == []

    conflict = client.post("/api/knowledge/pending_git/reject")
    assert conflict.status_code == 409
    assert "not 'pending'" in conflict.json()["detail"]

    stale_detail = client.get("/api/knowledge/pending/pending_git")
    assert stale_detail.status_code == 409
    assert "not 'pending'" in stale_detail.json()["detail"]


def test_server_rejects_pending_knowledge_with_archived_status() -> None:
    store = _TestKnowledgeStore(
        [
            KnowledgeEntry(
                id="pending_bad",
                text="Do not retain this model-authored knowledge.",
                namespace="project:cayu",
                status=KnowledgeStatus.PENDING,
            )
        ]
    )
    client = TestClient(create_server(CayuApp(knowledge_store=store), config=_LOCAL_SERVER_CONFIG))

    rejected = client.post("/api/knowledge/pending_bad/reject")
    assert rejected.status_code == 200
    assert rejected.json()["status"] == "archived"

    pending = client.get("/api/knowledge/pending")
    assert pending.status_code == 200
    assert pending.json()["entries"] == []


def test_server_maps_concurrent_knowledge_review_revision_to_conflict() -> None:
    class RacingKnowledgeStore(InMemoryKnowledgeStore):
        async def approve_pending_entry(self, authority, **kwargs):
            request = authority.request
            assert request.expected_revision is not None
            raise KnowledgeRevisionConflict(
                request.candidate_entry.id,
                expected_revision=request.expected_revision,
                actual_revision=request.expected_revision + 1,
            )

    store = RacingKnowledgeStore(
        [
            KnowledgeEntry(
                id="pending_race",
                text="Concurrent review candidate.",
                status=KnowledgeStatus.PENDING,
            )
        ],
        access_scope=KnowledgeAccessScope.privileged(),
    )
    client = TestClient(create_server(CayuApp(knowledge_store=store), config=_LOCAL_SERVER_CONFIG))

    response = client.post("/api/knowledge/pending_race/approve")

    assert response.status_code == 409
    assert "revision conflict" in response.json()["detail"]


def test_server_knowledge_review_reports_missing_store_and_scope_errors() -> None:
    missing_store = TestClient(create_server(CayuApp(), config=_LOCAL_SERVER_CONFIG))
    assert missing_store.get("/api/knowledge/pending").status_code == 404

    store = _TestKnowledgeStore(
        [
            KnowledgeEntry(
                id="pending_other",
                text="Other project knowledge.",
                namespace="project:other",
                labels={"project": "other"},
                status=KnowledgeStatus.PENDING,
            )
        ]
    )
    app = CayuApp(
        knowledge_store=store,
        knowledge_review_namespace="project:cayu",
        knowledge_review_labels={"project": "cayu"},
    )
    client = TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG))

    scoped_list = client.get("/api/knowledge/pending")
    assert scoped_list.status_code == 200
    assert scoped_list.json()["entries"] == []

    scoped_approve = client.post("/api/knowledge/pending_other/approve")
    assert scoped_approve.status_code == 403
    assert "outside review namespace" in scoped_approve.json()["detail"]

    scoped_detail = client.get("/api/knowledge/pending/pending_other")
    assert scoped_detail.status_code == 403
    assert "outside review namespace" in scoped_detail.json()["detail"]


def test_server_pending_knowledge_detail_validates_chunk_limits() -> None:
    store = _TestKnowledgeStore(
        [
            KnowledgeEntry(
                id="pending_git",
                text="Remote sandbox Git pushes should use a brokered credential proxy.",
                status=KnowledgeStatus.PENDING,
            )
        ]
    )
    client = TestClient(create_server(CayuApp(knowledge_store=store), config=_LOCAL_SERVER_CONFIG))

    response = client.get("/api/knowledge/pending/pending_git?max_chunks=0")
    assert response.status_code == 422
    assert "max_chunks" in str(response.json()["detail"])

    too_many_chunks = client.get("/api/knowledge/pending/pending_git?max_chunks=51")
    assert too_many_chunks.status_code == 422
    assert "max_chunks" in str(too_many_chunks.json()["detail"])

    too_many_bytes = client.get("/api/knowledge/pending/pending_git?max_bytes=128001")
    assert too_many_bytes.status_code == 422
    assert "max_bytes" in str(too_many_bytes.json()["detail"])

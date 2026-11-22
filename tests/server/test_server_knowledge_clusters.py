"""`GET /api/knowledge/pending/clusters`: the review queue grouped and ordered."""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from collections.abc import Sequence

from fastapi.testclient import TestClient
from tests.server.server_test_support import _LOCAL_SERVER_CONFIG

from cayu.applications import CayuApp
from cayu.knowledge.records import KnowledgeEntry, KnowledgeStatus
from cayu.knowledge.scopes import KnowledgeAccessScope
from cayu.knowledge.similarity import (
    KnowledgeComparable,
    KnowledgeSimilarity,
    KnowledgeSimilarityScore,
)
from cayu.server import create_server
from cayu.storage.memory import InMemoryKnowledgeStore

_TEXT = "Queued delivery retries keep the same delivery identity after acknowledgement loss."


def _entries() -> list[KnowledgeEntry]:
    return [
        KnowledgeEntry(id="p1", text=_TEXT, status=KnowledgeStatus.PENDING, title="Retries"),
        KnowledgeEntry(
            id="p2",
            text=_TEXT + " Checked again.",
            status=KnowledgeStatus.PENDING,
            metadata={"cayu:corroboration": {"source_count": 2, "sources": []}},
        ),
        KnowledgeEntry(
            id="p3", text="Branch snapshots stay immutable.", status=KnowledgeStatus.PENDING
        ),
        KnowledgeEntry(id="kept", text=_TEXT),
        KnowledgeEntry(id="other", text=_TEXT, namespace="other", status=KnowledgeStatus.PENDING),
    ]


def _client(**app_options) -> TestClient:
    store = InMemoryKnowledgeStore(_entries(), access_scope=KnowledgeAccessScope.privileged())
    app = CayuApp(knowledge_store=store, knowledge_review_namespace="default", **app_options)
    return TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG))


def test_pending_clusters_group_similar_proposals_most_sources_first():
    response = _client().get("/api/knowledge/pending/clusters")
    assert response.status_code == 200, response.text
    body = response.json()
    assert (body["pending_compared"], body["kept_compared"], body["truncated"]) == (3, 1, False)
    clusters = body["clusters"]
    assert [[row["entry_id"] for row in cluster["members"]] for cluster in clusters] == [
        ["p2", "p1"],
        ["p3"],
    ]
    assert [cluster["sources"] for cluster in clusters] == [3, 1]
    assert [row["sources"] for row in clusters[0]["members"]] == [2, 1]
    assert [row["entry_id"] for row in clusters[0]["similar_kept"]] == ["kept"]
    assert clusters[0]["members"][1]["text_preview"] == _TEXT
    assert clusters[1]["similar_kept"] == []
    # The detail route still answers for real entry ids.
    assert _client().get("/api/knowledge/pending/p1").status_code == 200


def test_pending_clusters_take_filters_and_bounds():
    client = _client()
    assert client.get("/api/knowledge/pending/clusters?max_entries=0").status_code == 422
    assert client.get("/api/knowledge/pending/clusters?max_entries=1001").status_code == 422
    first = client.get("/api/knowledge/pending/clusters?max_entries=1").json()
    assert first["pending_compared"] == 1 and first["truncated"] is True
    elsewhere = client.get("/api/knowledge/pending/clusters?namespace=other")
    assert elsewhere.status_code == 422


class _NothingAlike(KnowledgeSimilarity):
    async def score(
        self, subject: KnowledgeComparable, candidates: Sequence[KnowledgeComparable]
    ) -> list[KnowledgeSimilarityScore]:
        return [KnowledgeSimilarityScore(score=0.0, similar=False) for _ in candidates]


def test_pending_clusters_use_the_app_similarity_part():
    body = (
        _client(knowledge_similarity=_NothingAlike()).get("/api/knowledge/pending/clusters").json()
    )
    assert sorted(len(cluster["members"]) for cluster in body["clusters"]) == [1, 1, 1]
    assert all(cluster["similar_kept"] == [] for cluster in body["clusters"])


def test_pending_clusters_never_show_a_tool_entry_source_session():
    store = InMemoryKnowledgeStore(
        [
            KnowledgeEntry(
                id="tool-fact",
                text=_TEXT,
                source_type="tool",
                source_id="internal-session-id",
                source_uri="cayu://sessions/internal-session-id",
            ),
            KnowledgeEntry(
                id="proposal",
                text=_TEXT,
                status=KnowledgeStatus.PENDING,
                source_type="tool",
                source_id="internal-session-id",
            ),
        ],
        access_scope=KnowledgeAccessScope.privileged(),
    )
    app = CayuApp(knowledge_store=store, knowledge_review_namespace="default")
    response = TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG)).get(
        "/api/knowledge/pending/clusters"
    )
    assert response.status_code == 200, response.text
    assert [row["entry_id"] for row in response.json()["clusters"][0]["similar_kept"]] == [
        "tool-fact"
    ]
    assert "internal-session-id" not in response.text


def test_pending_clusters_compare_off_the_event_loop_on_the_bulk_lane(monkeypatch):
    """While a grouping compares, the server keeps answering other requests."""

    import asyncio
    import threading

    import httpx

    from cayu.knowledge import similarity as similarity_module
    from cayu.storage._read_lanes import ReadLane, current_read_lane

    entered, release = threading.Event(), threading.Event()
    seen: dict[str, object] = {}
    compare = similarity_module.LexicalKnowledgeSimilarity._matches_many

    def held(self, *args):
        seen["thread"] = threading.get_ident()
        seen["lane"] = current_read_lane()
        entered.set()
        # Bounded, so a regression fails instead of hanging the loop for good.
        assert release.wait(10), "the light request never answered"
        return compare(self, *args)

    monkeypatch.setattr(similarity_module.LexicalKnowledgeSimilarity, "_matches_many", held)
    store = InMemoryKnowledgeStore(_entries(), access_scope=KnowledgeAccessScope.privileged())
    app = CayuApp(knowledge_store=store, knowledge_review_namespace="default")
    server = create_server(app, config=_LOCAL_SERVER_CONFIG)

    async def scenario() -> tuple[httpx.Response, httpx.Response]:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=server), base_url="http://test"
        ) as client:
            grouping = asyncio.create_task(client.get("/api/knowledge/pending/clusters"))
            while not entered.is_set():
                assert not grouping.done(), grouping
                await asyncio.sleep(0.01)
            light = await asyncio.wait_for(client.get("/api/knowledge/pending/p1"), 5)
            release.set()
            return light, await grouping

    loop_thread = threading.get_ident()
    light, grouped = asyncio.run(scenario())
    assert light.status_code == 200, light.text
    assert grouped.status_code == 200, grouped.text
    assert [
        [row["entry_id"] for row in cluster["members"]] for cluster in grouped.json()["clusters"]
    ] == [
        ["p2", "p1"],
        ["p3"],
    ]
    assert seen["thread"] != loop_thread
    # The route holds a heavy-read slot, which puts its store reads on the bulk lane.
    assert seen["lane"] is ReadLane.BULK

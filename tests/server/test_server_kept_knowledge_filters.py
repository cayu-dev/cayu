"""Kept knowledge filters, the kept entry detail, and 422s that name the bad parameter."""

from __future__ import annotations

import asyncio
import json

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("sse_starlette")
from fastapi.testclient import TestClient

from cayu import CayuApp
from cayu.server import ServerConfig, create_server
from cayu.storage.memory import (
    InMemoryKnowledgeStore,
    KnowledgeAccessScope,
    KnowledgeEntry,
    KnowledgeStatus,
)


def _client(**app_options) -> TestClient:
    scope = KnowledgeAccessScope.privileged()
    store = InMemoryKnowledgeStore(
        [
            KnowledgeEntry(
                id="denial",
                namespace="reviews",
                kind="denial",
                labels={"agent": "hunter"},
                title="Denied rm -rf",
                text="The reviewer denied deleting the workspace.",
                status=KnowledgeStatus.ACTIVE,
            ),
            KnowledgeEntry(
                id="fact",
                namespace="reviews",
                kind="fact",
                labels={"agent": "hunter"},
                text="The repository uses uv.",
                status=KnowledgeStatus.ACTIVE,
            ),
            KnowledgeEntry(
                id="other",
                namespace="ops",
                kind="denial",
                labels={"agent": "scout"},
                text="Another namespace.",
                status=KnowledgeStatus.ACTIVE,
            ),
            KnowledgeEntry(
                id="proposal",
                namespace="reviews",
                kind="denial",
                text="Not kept yet.",
                status=KnowledgeStatus.PENDING,
            ),
        ],
        access_scope=scope,
    )
    app = CayuApp(
        knowledge_store=store,
        knowledge_access_scope=scope,
        enable_logging=False,
        **app_options,
    )
    return TestClient(create_server(app, config=ServerConfig.local_development()))


def _ids(response) -> list[str]:
    assert response.status_code == 200, response.text
    return sorted(row["entry_id"] for row in response.json()["entries"])


@pytest.mark.parametrize(
    ("query", "detail"),
    [
        ("limit=500", "query parameter limit must be at most 100"),
        ("limit=many", "query parameter limit must be an integer"),
        (
            "limit=0&max_bytes=0",
            "query parameter limit must be at least 1; "
            "query parameter max_bytes must be at least 1",
        ),
    ],
)
def test_invalid_query_parameters_are_named(query: str, detail: str) -> None:
    with _client() as client:
        response = client.get(f"/api/knowledge?{query}")

    assert response.status_code == 422
    assert response.json() == {"detail": f"Invalid control-plane request: {detail}."}
    assert response.headers["cache-control"] == "private, no-store"
    # The rejected values themselves are not echoed.
    assert "500" not in response.text and "many" not in response.text


def test_kept_list_filters_by_namespace_kind_and_label() -> None:
    with _client() as client:
        assert _ids(client.get("/api/knowledge")) == ["denial", "fact", "other"]
        assert _ids(client.get("/api/knowledge?namespace=reviews")) == ["denial", "fact"]
        assert _ids(client.get("/api/knowledge?kind=denial")) == ["denial", "other"]
        assert _ids(client.get("/api/knowledge?kind=denial&kind=fact&label=agent=hunter")) == [
            "denial",
            "fact",
        ]
        assert _ids(client.get("/api/knowledge?label=agent=nobody")) == []
        refused = client.get("/api/knowledge?label=agent")
        assert refused.status_code == 422
        assert refused.json()["detail"] == "Knowledge label filters must use `key=value`."


def test_kept_cursor_stays_bound_to_its_filters() -> None:
    with _client() as client:
        first = client.get("/api/knowledge?kind=denial&limit=1")
        assert first.status_code == 200, first.text
        cursor = first.json()["next_cursor"]
        assert cursor is not None
        following = client.get(f"/api/knowledge?kind=denial&limit=1&cursor={cursor}")
        assert following.status_code == 200
        reused = client.get(f"/api/knowledge?kind=fact&limit=1&cursor={cursor}")
        assert reused.status_code == 422


def test_filters_stay_inside_the_review_scope() -> None:
    with _client(knowledge_review_namespace="reviews") as client:
        assert _ids(client.get("/api/knowledge")) == ["denial", "fact"]
        refused = client.get("/api/knowledge?namespace=ops")
        assert refused.status_code == 422
        assert refused.json()["detail"] == (
            "Kept knowledge here is in namespace 'reviews'; the requested namespace is outside it."
        )


def test_kept_entry_detail_and_hidden_states_read_as_absent() -> None:
    with _client(knowledge_review_namespace="reviews") as client:
        response = client.get("/api/knowledge/kept/denial")
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["entry_id"] == "denial"
        assert body["status"] == "active"
        assert body["text"] == "The reviewer denied deleting the workspace."
        assert [chunk["text"] for chunk in body["chunks"]] == [body["text"]]
        assert body["chunk_limit"] == 50

        for absent in ("proposal", "other", "missing"):
            response = client.get(f"/api/knowledge/kept/{absent}")
            assert response.status_code == 404
            assert response.json() == {"detail": "Knowledge entry not found."}
        # The pending detail still owns `/knowledge/pending/...`.
        assert client.get("/api/knowledge/pending/proposal").status_code == 200
        assert client.get("/api/knowledge/pending").status_code == 200


def test_filters_the_store_refuses_answer_422_not_500() -> None:
    with _client() as client:
        control = client.get("/api/knowledge?kind=%00")
        many = client.get("/api/knowledge?" + "&".join(f"label=k{index}=v" for index in range(201)))

    assert control.status_code == 422
    assert control.json()["detail"] == "Invalid knowledge filters: kinds."
    assert many.status_code == 422
    assert many.json()["detail"] == "Invalid knowledge filters: labels."


def test_kept_detail_redacts_whole_chunks_before_bounding_them() -> None:
    from cayu.vaults.redaction import SecretRedactor

    secret = "abcdefghijklmnop"
    scope = KnowledgeAccessScope.privileged()
    store = InMemoryKnowledgeStore(
        [KnowledgeEntry(id="s", text=f"token={secret}", status=KnowledgeStatus.ACTIVE)],
        access_scope=scope,
    )
    app = CayuApp(
        knowledge_store=store,
        knowledge_access_scope=scope,
        secret_redactor=SecretRedactor(secret),
        enable_logging=False,
    )
    with TestClient(create_server(app, config=ServerConfig.local_development())) as client:
        bounded = client.get("/api/knowledge/kept/s?max_bytes=21")

    assert bounded.status_code == 200, bounded.text
    assert "abcdefghijklmno" not in bounded.text
    assert bounded.json()["chunks"][0]["text"].startswith("token=[REDACTED")


def test_kept_detail_hides_an_unverified_tool_source_session() -> None:
    scope = KnowledgeAccessScope.privileged()
    store = InMemoryKnowledgeStore(
        [
            KnowledgeEntry(
                id="tool-fact",
                text="Learned from a tool.",
                source_type="tool",
                source_id="internal-session-id",
                status=KnowledgeStatus.ACTIVE,
            )
        ],
        access_scope=scope,
    )
    app = CayuApp(knowledge_store=store, knowledge_access_scope=scope, enable_logging=False)
    with TestClient(create_server(app, config=ServerConfig.local_development())) as client:
        body = client.get("/api/knowledge/kept/tool-fact").json()

    # No registered specialist created it, so no session is verified as its source.
    assert body["source_id"] is None
    assert "internal-session-id" not in json.dumps(body)


def test_redaction_that_grows_a_chunk_never_yields_a_blank_one() -> None:
    from cayu.storage.memory import KnowledgeChunk
    from cayu.vaults.redaction import SecretRedactor

    scope = KnowledgeAccessScope.privileged()
    store = InMemoryKnowledgeStore(access_scope=scope)
    entry = KnowledgeEntry(id="s", text="token-value    body", status=KnowledgeStatus.ACTIVE)
    chunks = [
        KnowledgeChunk(id="c0", entry_id="s", entry_revision=1, chunk_index=0, text="token-value"),
        KnowledgeChunk(id="c1", entry_id="s", entry_revision=1, chunk_index=1, text="    body"),
    ]
    asyncio.run(store.create_entry(entry, chunks, access_scope=scope))
    app = CayuApp(
        knowledge_store=store,
        knowledge_access_scope=scope,
        secret_redactor=SecretRedactor("token-value"),
        enable_logging=False,
    )
    with TestClient(create_server(app, config=ServerConfig.local_development())) as client:
        responses = [client.get(f"/api/knowledge/kept/s?max_bytes={size}") for size in (20, 21, 25)]

    for response in responses:
        assert response.status_code == 200, response.text
        assert all(chunk["text"].strip() for chunk in response.json()["chunks"])
    assert responses[0].json()["chunks"][0]["text"] == "[REDACTED_SECRET]"

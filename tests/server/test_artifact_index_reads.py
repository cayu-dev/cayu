"""Artifact lists and overview counts read the artifact index, never the artifact stores.

A local and an S3 store (the in-process fake S3 client) are registered with an
app on a SQLite session store. Their commits and deletions land in the index
through the commit hook, `POST /artifacts/index/reindex` fills and checks it,
and every list, search, sort, cursor page and session overview afterwards makes
no store call at all.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
from collections import Counter
from typing import Any

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("sse_starlette")
from fastapi.testclient import TestClient
from tests.artifacts.test_aws_s3 import _S3Client
from tests.core.test_cloud_reporting import _app, _start

from cayu.artifacts._index import bind_artifact_index, unbind_artifact_index
from cayu.artifacts.aws_s3 import S3ArtifactStore
from cayu.artifacts.local import LocalArtifactStore
from cayu.environments import Environment, EnvironmentSpec
from cayu.server import ServerConfig, create_server
from cayu.storage.sqlite import SQLiteSessionStore

_STORE_READS = ("list", "scan_metadata", "count_session_artifacts", "read_bytes", "read_range")


class _CountingS3Client(_S3Client):
    def __init__(self, calls: Counter) -> None:
        super().__init__()
        self.calls = calls

    def list_objects_v2(self, **kwargs: Any) -> dict[str, Any]:
        self.calls["s3.list_objects_v2"] += 1
        return super().list_objects_v2(**kwargs)

    def get_object(self, **kwargs: Any) -> dict[str, Any]:
        self.calls["s3.get_object"] += 1
        return super().get_object(**kwargs)


@pytest.fixture
def indexed(tmp_path, monkeypatch):
    calls: Counter = Counter()
    app = _app(SQLiteSessionStore(tmp_path / "sessions.sqlite"))
    local = LocalArtifactStore(tmp_path / "local", store_id="local")
    s3 = S3ArtifactStore("fake", client=_CountingS3Client(calls), store_id="s3")
    for store in (local, s3):
        app.register_environment(Environment(EnvironmentSpec(name=store.id), artifact_store=store))
        for name in _STORE_READS:
            original = getattr(store, name)

            def counted(*args, _original=original, _name=f"{store.id}.{name}", **kwargs):
                calls[_name] += 1
                return _original(*args, **kwargs)

            monkeypatch.setattr(store, name, counted)
    with TestClient(create_server(app, config=ServerConfig.local_development())) as client:
        client.portal.call(_start, app, "owner", "writer")

        async def populate():
            saved = []
            for index in range(24):
                store = (local, s3)[index % 2]
                saved.append(
                    (
                        store.id,
                        await store.put_bytes(
                            b"x" * index,
                            filename=f"File-{index:02}.txt",
                            session_id="owner" if index % 3 else "other",
                            agent_name="writer",
                            environment_name=store.id,
                        ),
                    )
                )
            return saved

        saved = client.portal.call(populate)
        yield client, app, (local, s3), saved, calls


def _ids(body: dict) -> list[tuple[str, str]]:
    return [(row["artifact_store_id"], row["id"]) for row in body["artifacts"]]


def test_lists_and_overviews_read_the_index_without_store_calls(indexed):
    client, _, _, saved, calls = indexed
    report = client.post("/api/artifacts/index/reindex", json={"check": False})
    assert report.status_code == 200, report.text
    assert {row["artifact_store_id"]: row["store_count"] for row in report.json()["stores"]} == {
        "local": 12,
        "s3": 12,
    }
    # The hook had already written every row, so the fill found no drift.
    assert not any(row["drift"] for row in report.json()["stores"])
    calls.clear()

    newest_first = sorted(
        saved, key=lambda item: (item[1].created_at.isoformat(), item[0], item[1].id), reverse=True
    )
    body = client.get("/api/artifacts", params={"limit": 5}).json()
    assert _ids(body) == [(store_id, artifact.id) for store_id, artifact in newest_first[:5]]
    assert body["total_count"] == 24 and body["next_offset"] == 5 and body["next_cursor"]

    # Keyset pages walk the whole list in order.
    walked, cursor = [], None
    while True:
        params: dict[str, Any] = {"limit": 7, "order_by": "filename_asc"}
        if cursor is not None:
            params["cursor"] = cursor
        page = client.get("/api/artifacts", params=params).json()
        walked.extend(row["filename"] for row in page["artifacts"])
        cursor = page["next_cursor"]
        if cursor is None:
            break
    assert walked == sorted(artifact.filename for _, artifact in saved)

    search = client.get("/api/artifacts", params={"q": "file-1", "order_by": "size_bytes_desc"})
    assert [row["filename"] for row in search.json()["artifacts"]] == [
        f"File-{index}.txt" for index in range(19, 9, -1)
    ]
    session_tab = client.get("/api/artifacts", params={"session_id": "owner", "limit": 6}).json()
    assert session_tab["total_count"] == 16 and len(session_tab["artifacts"]) == 6
    environment = client.get("/api/artifacts", params={"environment_name": "s3", "limit": 5})
    assert environment.json()["total_count"] == 12
    overview = client.get("/api/sessions", params={"include": "overview"})
    assert overview.status_code == 200, overview.text
    owner = next(
        row for row in overview.json()["overviews"] if row["uses"]["artifacts"] is not None
    )
    assert owner["uses"]["artifacts"] == "16"
    assert calls == Counter()

    # Content is still read from the store, only when an artifact is opened.
    opened = client.get(f"/api/artifacts/{saved[0][1].id}", params={"artifact_store_id": "local"})
    assert opened.status_code == 200
    assert calls["local.read_bytes"] == 1


def test_deletions_leave_lists_and_counts_through_the_hook(indexed):
    client, _, (local, _), saved, calls = indexed
    client.post("/api/artifacts/index/reindex", json={})
    deleted = [artifact for store_id, artifact in saved if store_id == "local"][:3]

    async def delete():
        for artifact in deleted:
            await local.delete(artifact.id)

    client.portal.call(delete)
    calls.clear()
    body = client.get("/api/artifacts", params={"artifact_store_id": "local", "limit": 100}).json()
    assert body["total_count"] == 9
    assert not {artifact.id for artifact in deleted} & {row["id"] for row in body["artifacts"]}
    assert calls == Counter()


def test_reindex_check_reports_drift_and_repair_clears_it(indexed, capsys, monkeypatch):
    client, app, (local, s3), saved, _ = indexed
    client.post("/api/artifacts/index/reindex", json={})
    # Writes made while the store is not bound, as by a process outside Cayu.
    unbind_artifact_index(local, app.session_store)
    gone = next(artifact for store_id, artifact in saved if store_id == "local")

    async def drift():
        added = await local.put_bytes(b"outside", filename="outside.txt", session_id="owner")
        await local.delete(gone.id)
        return added

    added = client.portal.call(drift)
    bind_artifact_index(local, app.session_store)

    from cayu.cli import artifact as artifact_cli

    class _Api:
        def post(self, path, body, *, resource):
            response = client.post("/api" + path, json=body)
            assert response.status_code == 200, response.text
            return response.json()

        def close(self):
            pass

    monkeypatch.setattr(artifact_cli, "_open_api", lambda target: _Api())

    def run(check: bool) -> int:
        args = argparse.Namespace(
            artifact_command="reindex",
            check=check,
            artifact_store_id="local",
            output_format="table",
        )
        return asyncio.run(artifact_cli._run_artifact(args, None))  # type: ignore[arg-type]

    assert run(check=True) == 1
    printed = capsys.readouterr().out
    assert f"missing: local {added.id}" in printed and f"orphaned: local {gone.id}" in printed
    listed = client.get("/api/artifacts", params={"artifact_store_id": "local", "limit": 100})
    assert added.id not in {row["id"] for row in listed.json()["artifacts"]}

    assert run(check=False) == 0
    assert run(check=True) == 0
    listed = client.get("/api/artifacts", params={"artifact_store_id": "local", "limit": 100})
    ids = {row["id"] for row in listed.json()["artifacts"]}
    assert added.id in ids and gone.id not in ids
    assert s3.id == "s3"


def test_stores_not_yet_filled_are_read_from_the_store(indexed):
    client, _, _, _, calls = indexed
    calls.clear()
    body = client.get("/api/artifacts", params={"limit": 5}).json()
    # Read from the stores themselves, and filling started in the background.
    assert body["total_count"] == 24 and body.get("next_cursor") is None
    assert calls["local.list"] == 1
    cursor = client.get("/api/artifacts", params={"cursor": "abc"})
    assert cursor.status_code in {409, 422}


def test_searches_and_cursors_never_reveal_redacted_text(indexed, monkeypatch):
    from cayu.vaults.redaction import SecretRedactor

    client, app, (local, _), _, calls = indexed
    secret = "sk-live-ABCDEF123456"

    async def save():
        await local.put_bytes(b"k", filename=f"key-{secret}.txt", session_id="owner")

    client.portal.call(save)
    client.post("/api/artifacts/index/reindex", json={})
    monkeypatch.setattr(app, "_secret_redactor", SecretRedactor([secret]))
    calls.clear()

    # A guess at the secret matches nothing, and the count does not confirm it.
    for guess in ("sk-live-abc", "sk-live-abx"):
        body = client.get("/api/artifacts", params={"q": guess}).json()
        assert (body["total_count"], body["artifacts"], body["next_cursor"]) == (0, [], None)
    shown = client.get("/api/artifacts", params={"q": "key-"}).json()
    assert shown["total_count"] == 1
    assert shown["artifacts"][0]["filename"] == "key-[REDACTED_SECRET].txt"

    # Cursors name a row, not its filename.
    first = client.get("/api/artifacts", params={"order_by": "filename_desc", "limit": 1}).json()
    assert first["artifacts"][0]["filename"].startswith("key-")
    cursor = first["next_cursor"]
    decoded = base64.urlsafe_b64decode(cursor + "=" * (-len(cursor) % 4)).decode()
    assert "sk-live" not in decoded.lower()

    # Search pages and totals still agree, page by page, when every match is rechecked.
    walked, cursor = [], None
    while True:
        params: dict[str, Any] = {"q": "file-", "limit": 5, "order_by": "filename_asc"}
        if cursor is not None:
            params["cursor"] = cursor
        page = client.get("/api/artifacts", params=params).json()
        assert page["total_count"] == 24
        walked.extend(row["filename"] for row in page["artifacts"])
        cursor = page["next_cursor"]
        if cursor is None:
            break
    assert walked == [f"File-{index:02}.txt" for index in range(24)]
    offset = client.get("/api/artifacts", params={"q": "file-", "limit": 5, "offset": 20}).json()
    assert (offset["total_count"], len(offset["artifacts"]), offset["next_cursor"]) == (24, 4, None)
    assert calls == Counter()

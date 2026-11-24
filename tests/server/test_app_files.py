"""`cayu.server.app_files`: multipart uploads into an artifact store, and streamed downloads."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.testclient import TestClient

from cayu.artifacts import ArtifactScope, LocalArtifactStore
from cayu.artifacts.base import ArtifactReadResult
from cayu.server.app_files import UPLOAD_ENVELOPE_BYTES, artifact_download, read_upload

LIMIT = 300_000


class CountingStore:
    """The store, counting the ranges a download reads."""

    def __init__(self, store: LocalArtifactStore) -> None:
        self.store = store
        self.ranges: list[tuple[int, int]] = []

    async def read_range(self, artifact_id: str, *, offset: int, max_bytes: int):
        self.ranges.append((offset, max_bytes))
        return await self.store.read_range(artifact_id, offset=offset, max_bytes=max_bytes)


def _client(tmp_path: Path) -> tuple[TestClient, CountingStore]:
    store = LocalArtifactStore(tmp_path / "artifacts")
    reader = CountingStore(store)
    router = APIRouter(prefix="/app")

    @router.post("/files", status_code=201)
    async def upload(request: Request) -> dict:
        upload = await read_upload(request, max_bytes=LIMIT)
        stored = await store.put_bytes(
            upload.content,
            filename=upload.filename,
            content_type=upload.content_type,
            scope=ArtifactScope.ENVIRONMENT,
            environment_name="files",
        )
        return {"id": stored.id, "size": stored.size_bytes, "fields": dict(upload.fields)}

    @router.get("/files/{artifact_id}")
    async def download(artifact_id: str, name: str | None = None, inline: bool = False):
        try:
            return await artifact_download(
                reader,
                artifact_id,
                filename=name,
                chunk_bytes=64 * 1024,
                disposition="inline" if inline else "attachment",
            )
        except FileNotFoundError:
            raise HTTPException(404, "File not found.") from None

    server = FastAPI()
    server.include_router(router)
    return TestClient(server), reader


def test_upload_then_streamed_download_round_trips(tmp_path: Path):
    client, reader = _client(tmp_path)
    content = bytes(range(256)) * 1000  # 256 000 bytes, four 64 KiB ranges

    created = client.post(
        "/app/files",
        files={"file": ("C:\\books\\March close.xlsx", content, "application/vnd.ms-excel")},
        data={"period": "2026-03"},
    )
    assert created.status_code == 201, created.text
    body = created.json()
    assert body["size"] == len(content) and body["fields"] == {"period": "2026-03"}

    downloaded = client.get(f"/app/files/{body['id']}")
    assert downloaded.status_code == 200
    assert downloaded.content == content
    assert downloaded.headers["content-length"] == str(len(content))
    assert downloaded.headers["content-type"] == "application/vnd.ms-excel"
    assert downloaded.headers["content-disposition"] == 'attachment; filename="March close.xlsx"'
    assert downloaded.headers["x-content-type-options"] == "nosniff"
    assert [offset for offset, _ in reader.ranges] == [0, 65536, 131072, 196608]


def test_download_names_a_non_ascii_file_both_ways(tmp_path: Path):
    client, _ = _client(tmp_path)
    created = client.post("/app/files", files={"file": ("a.txt", b"hi", "text/plain")}).json()

    response = client.get(f"/app/files/{created['id']}", params={"name": "résumé.txt"})

    assert response.headers["content-disposition"] == (
        "attachment; filename=\"rsum.txt\"; filename*=UTF-8''r%C3%A9sum%C3%A9.txt"
    )


@pytest.mark.parametrize(
    ("content_type", "disposition"),
    [
        ("text/plain", "inline"),
        ("application/pdf", "inline"),
        ("text/html", "attachment"),
        ("image/svg+xml", "attachment"),
    ],
)
def test_inline_only_for_types_that_cannot_run_script(
    tmp_path: Path, content_type: str, disposition: str
):
    client, _ = _client(tmp_path)
    created = client.post(
        "/app/files", files={"file": ("x", b"<script>alert(1)</script>", content_type)}
    ).json()

    response = client.get(f"/app/files/{created['id']}", params={"inline": "true"})

    assert response.headers["content-disposition"].split(";")[0] == disposition
    assert response.headers["cache-control"] == "private, no-store"


def test_missing_file_raises_before_the_response_starts(tmp_path: Path):
    client, reader = _client(tmp_path)
    created = client.post("/app/files", files={"file": ("a.txt", b"hi", "text/plain")}).json()
    asyncio.run(reader.store.delete(created["id"]))

    response = client.get(f"/app/files/{created['id']}")

    assert response.status_code == 404
    assert response.json() == {"detail": "File not found."}


def test_declared_oversized_body_is_refused_before_reading(tmp_path: Path):
    client, _ = _client(tmp_path)

    response = client.post(
        "/app/files",
        content=b"x" * 10,
        headers={
            "content-type": "multipart/form-data; boundary=b",
            "content-length": str(LIMIT + UPLOAD_ENVELOPE_BYTES + 1),
        },
    )

    assert response.status_code == 413


def test_chunked_oversized_body_is_refused_while_it_arrives(tmp_path: Path):
    client, _ = _client(tmp_path)
    boundary = "b0undary"

    def body():
        yield (
            f'--{boundary}\r\nContent-Disposition: form-data; name="file"; '
            f'filename="big.zip"\r\nContent-Type: application/zip\r\n\r\n'
        ).encode()
        for _ in range(100):  # 6.4 MB, far over the limit
            yield b"z" * 65536
        yield f"\r\n--{boundary}--\r\n".encode()

    response = client.post(
        "/app/files",
        content=body(),
        headers={"content-type": f"multipart/form-data; boundary={boundary}"},
    )

    assert response.status_code == 413
    assert response.json() == {"detail": f"The file is larger than {LIMIT} bytes."}
    assert not list((tmp_path / "artifacts").rglob("*.zip"))


def test_file_just_over_the_limit_is_refused(tmp_path: Path):
    client, _ = _client(tmp_path)

    response = client.post(
        "/app/files", files={"file": ("big.bin", b"x" * (LIMIT + 1), "application/zip")}
    )

    assert response.status_code == 413


@pytest.mark.parametrize(
    ("kwargs", "detail"),
    [
        ({"json": {"file": "x"}}, "Send the file as multipart/form-data."),
        ({"data": {"other": "x"}, "files": {"x": ("a", b"a")}}, "form field 'file'"),
    ],
)
def test_requests_without_a_file_get_422(tmp_path: Path, kwargs: dict, detail: str):
    client, _ = _client(tmp_path)

    response = client.post("/app/files", **kwargs)

    assert response.status_code == 422
    assert detail in response.json()["detail"]


def test_download_refuses_a_file_that_changes_mid_stream():
    from cayu.artifacts.base import ArtifactMetadata

    def metadata(size: int) -> ArtifactMetadata:
        return ArtifactMetadata(
            id="art_1",
            filename="a.bin",
            size_bytes=size,
            scope=ArtifactScope.ENVIRONMENT,
            environment_name="files",
        )

    class Shrinking:
        async def read_range(self, artifact_id: str, *, offset: int, max_bytes: int):
            total = 4 if offset == 0 else 3
            return ArtifactReadResult(
                metadata=metadata(total),
                content=b"ab" if offset == 0 else b"c",
                total_bytes=total,
                truncated=offset == 0,
                offset=offset,
            )

    async def scenario() -> None:
        response = await artifact_download(Shrinking(), "art_1", chunk_bytes=2)
        chunks = response.body_iterator
        assert await anext(chunks) == b"ab"
        with pytest.raises(RuntimeError, match="changed"):
            await anext(chunks)

    asyncio.run(scenario())


def test_download_checks_its_chunk_size():
    with pytest.raises(ValueError):
        asyncio.run(artifact_download(object(), "art_1", chunk_bytes=5 * 1024 * 1024))  # type: ignore[arg-type]

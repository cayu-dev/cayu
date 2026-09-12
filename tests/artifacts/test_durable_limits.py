from __future__ import annotations

import asyncio
import json

import pytest
from tests.artifacts.test_aws_s3 import _S3Client

from cayu import (
    LocalArtifactStore,
    LocalWorkspace,
    S3ArtifactStore,
    copy_workspace_file_to_artifact,
)
from cayu._validation import DURABLE_METADATA_LIMITS, DurableValueError


def _metadata_at_limit():
    available = DURABLE_METADATA_LIMITS.max_bytes - len(b'{"value":""}')
    return {"value": "é" * (available // 2) + "x" * (available % 2)}


@pytest.mark.parametrize("backend", ["local", "s3"])
def test_artifact_metadata_limit_precedes_publication_and_preserves_existing(tmp_path, backend):
    client = _S3Client()
    root = tmp_path / "artifacts"
    store = (
        LocalArtifactStore(root) if backend == "local" else S3ArtifactStore("bucket", client=client)
    )
    metadata = _metadata_at_limit()
    assert (
        len(json.dumps(metadata, ensure_ascii=False, separators=(",", ":")).encode()) == 256 * 1024
    )

    async def run():
        artifact = await store.put_bytes(
            b"original", filename="result.txt", session_id="s", metadata=metadata
        )
        before_files = (
            {
                str(path.relative_to(root)): path.read_bytes()
                for path in root.rglob("*")
                if path.is_file()
            }
            if backend == "local"
            else dict(client.objects)
        )
        before_calls = (len(client.put_calls), len(client.get_calls), len(client.delete_calls))
        invalid = {"value": metadata["value"] + "x"}
        for artifact_id in (artifact.id, "art_" + "a" * 32):
            with pytest.raises(DurableValueError) as caught:
                await store.put_bytes(
                    b"replacement",
                    artifact_id=artifact_id,
                    filename="result.txt",
                    session_id="s",
                    metadata=invalid,
                )
            assert caught.value.limit == 256 * 1024
            assert caught.value.dimension == "bytes"
        assert before_calls == (
            len(client.put_calls),
            len(client.get_calls),
            len(client.delete_calls),
        )
        after_files = (
            {
                str(path.relative_to(root)): path.read_bytes()
                for path in root.rglob("*")
                if path.is_file()
            }
            if backend == "local"
            else dict(client.objects)
        )
        assert after_files == before_files
        reopened = (
            LocalArtifactStore(root)
            if backend == "local"
            else S3ArtifactStore("bucket", client=client)
        )
        assert (await reopened.read_bytes(artifact.id)).content == b"original"
        assert dict((await reopened.read_bytes(artifact.id)).metadata.metadata) == metadata

    asyncio.run(run())


def test_workspace_artifact_metadata_rejected_before_read_and_after_enrichment(
    tmp_path, monkeypatch
):
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    workspace = LocalWorkspace(workspace_root)
    store = LocalArtifactStore(tmp_path / "artifacts")
    reads = 0
    original_read = workspace.read_bytes

    async def read(*args, **kwargs):
        nonlocal reads
        reads += 1
        return await original_read(*args, **kwargs)

    monkeypatch.setattr(workspace, "read_bytes", read)

    async def run():
        await workspace.write_bytes("result.txt", b"value")
        exact = _metadata_at_limit()
        with pytest.raises(DurableValueError):
            await copy_workspace_file_to_artifact(
                workspace,
                store,
                "result.txt",
                session_id="s",
                metadata={"value": exact["value"] + "x"},
            )
        assert reads == 0
        with pytest.raises(DurableValueError):
            await copy_workspace_file_to_artifact(
                workspace, store, "result.txt", session_id="s", metadata=exact
            )
        assert reads == 1
        assert (await store.list(session_id="s")).total_count == 0

    asyncio.run(run())

from __future__ import annotations

import asyncio

import pytest
from tests.artifacts.test_aws_s3 import _S3Client

from cayu.artifacts import ArtifactReadResult, LocalArtifactStore, copy_artifact_read_result
from cayu.artifacts.aws_s3 import S3ArtifactStore


@pytest.mark.parametrize("backend", ["local", "s3"])
def test_artifact_ranges_cover_exact_content_without_prefix_amplification(tmp_path, backend):
    client = _S3Client()
    store = (
        LocalArtifactStore(tmp_path / "artifacts", store_id="ranges")
        if backend == "local"
        else S3ArtifactStore("bucket", client=client)
    )
    data = ("αβ🙂" * 100 + "TAIL").encode()
    artifact = asyncio.run(store.put_bytes(data, filename="text.txt", session_id="session"))
    for offset in range(0, len(data), 13):
        result = asyncio.run(store.read_range(artifact.id, offset=offset, max_bytes=13))
        assert result.content == data[offset : offset + 13]
        assert result.offset == offset
        assert result.source_bytes_read == len(result.content)
        assert result.total_bytes == len(data)
        assert result.truncated == (offset + len(result.content) < len(data))
    eof = asyncio.run(store.read_range(artifact.id, offset=len(data), max_bytes=13))
    assert eof.content == b"" and not eof.truncated
    with pytest.raises(ValueError, match="offset"):
        asyncio.run(store.read_range(artifact.id, offset=len(data) + 1, max_bytes=13))
    if backend == "s3":
        reads = [call for call in client.get_calls if call["Key"].endswith("/content")]
        assert len(reads) == (len(data) + 12) // 13
        for index, call in enumerate(reads):
            assert call["Range"] == f"bytes={index * 13}-{min(len(data), (index + 1) * 13) - 1}"
    else:
        reconstructed = LocalArtifactStore(tmp_path / "artifacts", store_id="ranges")
        tail = asyncio.run(reconstructed.read_range(artifact.id, offset=len(data) - 4, max_bytes=4))
        assert tail.content == b"TAIL"
    asyncio.run(store.delete(artifact.id))
    with pytest.raises(FileNotFoundError):
        asyncio.run(store.read_range(artifact.id, offset=13, max_bytes=13))


def test_s3_range_does_not_materialize_an_ignored_range():
    class IgnoringClient(_S3Client):
        def get_object(self, **kwargs):
            response = super().get_object(**{k: v for k, v in kwargs.items() if k != "Range"})
            if "Range" in kwargs:
                original = response["Body"]

                class Body:
                    def read(self, limit):
                        assert limit == 7
                        return original.read(limit)

                    def close(self):
                        original.close()

                response["Body"] = Body()
            return response

    from cayu.artifacts import ArtifactStoreUnavailableError

    store = S3ArtifactStore("bucket", client=IgnoringClient())
    artifact = asyncio.run(
        store.put_bytes(b"x" * 100_000, filename="text.txt", session_id="session")
    )
    with pytest.raises(ArtifactStoreUnavailableError):
        asyncio.run(store.read_range(artifact.id, offset=1000, max_bytes=7))


def test_range_result_validates_offset_and_remaining_bytes(tmp_path):
    store = LocalArtifactStore(tmp_path / "artifacts", store_id="ranges")
    metadata = asyncio.run(store.put_bytes(b"abcdef", filename="text.txt", session_id="session"))
    result = ArtifactReadResult(metadata, b"ef", 6, offset=4)
    assert copy_artifact_read_result(result, expected_offset=4) == result
    with pytest.raises(ValueError, match="offset"):
        copy_artifact_read_result(result)
    with pytest.raises(ValueError, match="range"):
        ArtifactReadResult(metadata, b"def", 6, offset=4)


def test_s3_eof_range_requires_committed_content_to_exist():
    client = _S3Client()
    store = S3ArtifactStore("bucket", client=client)
    artifact = asyncio.run(store.put_bytes(b"text", filename="text.txt", session_id="session"))
    for key in tuple(client.objects):
        if key[1].endswith("/content"):
            del client.objects[key]
    with pytest.raises(FileNotFoundError):
        asyncio.run(store.read_range(artifact.id, offset=4, max_bytes=4))


def test_browser_upload_readback_rejects_an_artifact_range(tmp_path):
    from cayu.tools.browser_session import _copy_upload_artifact_read_result

    store = LocalArtifactStore(tmp_path / "artifacts", store_id="ranges")
    metadata = asyncio.run(store.put_bytes(b"abcdef", filename="text.txt", session_id="session"))
    result = ArtifactReadResult(metadata, b"e", 6, truncated=True, offset=4)
    with pytest.raises(ValueError, match="range"):
        _copy_upload_artifact_read_result(
            result, expected_artifact_id=metadata.id, max_content_bytes=1
        )

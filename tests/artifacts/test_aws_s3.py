from __future__ import annotations

import asyncio
import io
from typing import Any

import pytest

from cayu import (
    ArtifactScope,
    ArtifactStoreUnavailableError,
    InvalidArtifactIdError,
    S3ArtifactStore,
)


class _ClientError(RuntimeError):
    def __init__(self, code: str) -> None:
        self.response = {"Error": {"Code": code}}
        super().__init__(code)


class _S3Client:
    def __init__(self) -> None:
        self.objects: dict[tuple[str, str], bytes] = {}
        self.put_calls: list[dict[str, Any]] = []
        self.get_calls: list[dict[str, Any]] = []
        self.delete_calls: list[dict[str, Any]] = []
        self.fail_put_suffix: str | None = None
        self.fail_get_code: str | None = None

    def put_object(self, **kwargs: Any) -> dict[str, Any]:
        self.put_calls.append(kwargs)
        if self.fail_put_suffix and kwargs["Key"].endswith(self.fail_put_suffix):
            raise _ClientError("AccessDenied")
        self.objects[(kwargs["Bucket"], kwargs["Key"])] = kwargs["Body"]
        return {"ETag": '"etag"'}

    def get_object(self, **kwargs: Any) -> dict[str, Any]:
        self.get_calls.append(kwargs)
        if self.fail_get_code is not None:
            raise _ClientError(self.fail_get_code)
        try:
            value = self.objects[(kwargs["Bucket"], kwargs["Key"])]
        except KeyError as exc:
            raise _ClientError("NoSuchKey") from exc
        byte_range = kwargs.get("Range")
        if byte_range is not None:
            start, end = byte_range.removeprefix("bytes=").split("-", 1)
            value = value[int(start) : int(end) + 1]
        return {"Body": io.BytesIO(value), "ContentLength": len(value)}

    def list_objects_v2(self, **kwargs: Any) -> dict[str, Any]:
        keys = sorted(
            key
            for bucket, key in self.objects
            if bucket == kwargs["Bucket"] and key.startswith(kwargs["Prefix"])
        )
        return {
            "IsTruncated": False,
            "Contents": [{"Key": key} for key in keys],
        }

    def delete_objects(self, **kwargs: Any) -> dict[str, Any]:
        self.delete_calls.append(kwargs)
        for item in kwargs["Delete"]["Objects"]:
            self.objects.pop((kwargs["Bucket"], item["Key"]), None)
        return {"Deleted": kwargs["Delete"]["Objects"]}


def test_s3_artifact_store_puts_reads_lists_and_deletes() -> None:
    client = _S3Client()
    store = S3ArtifactStore(
        "cayu-artifacts",
        prefix="prod/artifacts/",
        store_id="aws-artifacts",
        client=client,
        kms_key_id="arn:aws:kms:us-east-1:123:key/key-1",
    )

    session_artifact = asyncio.run(
        store.put_bytes(
            b"invoice text",
            filename="invoice.txt",
            content_type="text/plain",
            session_id="sess_1",
            agent_name="assistant",
            environment_name="aws",
            metadata={"source": "agent"},
        )
    )
    environment_artifact = asyncio.run(
        store.put_bytes(
            b"shared",
            filename="shared.txt",
            scope=ArtifactScope.ENVIRONMENT,
            environment_name="aws",
        )
    )

    read = asyncio.run(store.read_bytes(session_artifact.id))
    listed = asyncio.run(store.list(scope=ArtifactScope.SESSION, session_id="sess_1"))

    assert store.id == "aws-artifacts"
    assert read.metadata == session_artifact
    assert read.content == b"invoice text"
    assert read.total_bytes == 12
    assert read.truncated is False
    assert listed.artifacts == (session_artifact,)
    assert listed.total_count == 1
    assert listed.truncated is False
    assert environment_artifact not in listed.artifacts
    assert all(call["ServerSideEncryption"] == "aws:kms" for call in client.put_calls)
    assert all(
        call["SSEKMSKeyId"] == "arn:aws:kms:us-east-1:123:key/key-1" for call in client.put_calls
    )

    asyncio.run(store.delete(session_artifact.id))
    with pytest.raises(FileNotFoundError):
        asyncio.run(store.read_bytes(session_artifact.id))


def test_s3_artifact_store_uses_range_for_bounded_read() -> None:
    client = _S3Client()
    store = S3ArtifactStore("bucket", client=client)
    artifact = asyncio.run(store.put_bytes(b"abcdef", filename="letters.txt", session_id="sess_1"))

    read = asyncio.run(store.read_bytes(artifact.id, max_bytes=3))

    assert read.content == b"abc"
    assert read.total_bytes == 6
    assert read.truncated is True
    content_get = next(call for call in client.get_calls if call["Key"].endswith("/content"))
    assert content_get["Range"] == "bytes=0-2"


def test_s3_artifact_store_lists_all_metadata_then_applies_limit() -> None:
    client = _S3Client()
    store = S3ArtifactStore("bucket", client=client)
    first = asyncio.run(store.put_bytes(b"one", filename="one.txt", session_id="sess_1"))
    second = asyncio.run(store.put_bytes(b"two", filename="two.txt", session_id="sess_1"))

    result = asyncio.run(store.list(session_id="sess_1", limit=1))

    assert result.artifacts in ((first,), (second,))
    assert result.total_count == 2
    assert result.truncated is True


def test_s3_artifact_store_rejects_invalid_ids_before_aws() -> None:
    client = _S3Client()
    store = S3ArtifactStore("bucket", client=client)

    for artifact_id in ("", "../escape", "art_short", f"art_{'A' * 32}"):
        with pytest.raises(InvalidArtifactIdError):
            asyncio.run(store.read_bytes(artifact_id))
        with pytest.raises(InvalidArtifactIdError):
            asyncio.run(store.delete(artifact_id))

    assert client.get_calls == []
    assert client.delete_calls == []


def test_s3_artifact_store_removes_content_when_metadata_commit_fails() -> None:
    client = _S3Client()
    client.fail_put_suffix = "metadata.json"
    store = S3ArtifactStore("bucket", client=client)

    with pytest.raises(ArtifactStoreUnavailableError, match="write"):
        asyncio.run(store.put_bytes(b"orphan", filename="orphan.txt", session_id="sess_1"))

    assert client.objects == {}
    assert len(client.delete_calls) == 1


def test_s3_artifact_store_treats_missing_bucket_as_backend_unavailable() -> None:
    client = _S3Client()
    client.fail_get_code = "NoSuchBucket"
    store = S3ArtifactStore("missing-bucket", client=client)

    with pytest.raises(ArtifactStoreUnavailableError, match="metadata"):
        asyncio.run(store.read_bytes(f"art_{'a' * 32}"))


def test_s3_artifact_store_rejects_scope_and_client_configuration_errors() -> None:
    client = _S3Client()
    store = S3ArtifactStore("bucket", client=client)

    with pytest.raises(ValueError, match="session_id"):
        asyncio.run(store.put_bytes(b"x", filename="x.txt"))
    with pytest.raises(ValueError, match="environment_name"):
        asyncio.run(store.put_bytes(b"x", filename="x.txt", scope=ArtifactScope.ENVIRONMENT))
    with pytest.raises(ValueError, match="injected client"):
        S3ArtifactStore("bucket", client=client, profile_name="prod")

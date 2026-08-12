from __future__ import annotations

import asyncio
import io
from threading import Event as ThreadEvent
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
        self.delete_errors_by_suffix: dict[str, str] = {}
        self.fail_put_suffix: str | None = None
        self.fail_put_once_suffix: str | None = None
        self.fail_get_code: str | None = None
        self.fail_delete_code: str | None = None

    def put_object(self, **kwargs: Any) -> dict[str, Any]:
        self.put_calls.append(kwargs)
        if self.fail_put_suffix and kwargs["Key"].endswith(self.fail_put_suffix):
            raise _ClientError("AccessDenied")
        if self.fail_put_once_suffix and kwargs["Key"].endswith(self.fail_put_once_suffix):
            self.fail_put_once_suffix = None
            raise _ClientError("AccessDenied")
        if (
            kwargs.get("IfNoneMatch") == "*"
            and (
                kwargs["Bucket"],
                kwargs["Key"],
            )
            in self.objects
        ):
            raise _ClientError("PreconditionFailed")
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
        if self.fail_delete_code is not None:
            raise _ClientError(self.fail_delete_code)
        errors: list[dict[str, str]] = []
        for item in kwargs["Delete"]["Objects"]:
            matching_suffix = next(
                (suffix for suffix in self.delete_errors_by_suffix if item["Key"].endswith(suffix)),
                None,
            )
            if matching_suffix is not None:
                errors.append(
                    {
                        "Key": item["Key"],
                        "Code": self.delete_errors_by_suffix[matching_suffix],
                    }
                )
                continue
            self.objects.pop((kwargs["Bucket"], item["Key"]), None)
        return {"Errors": errors}


class _BlockingContentUploadS3Client(_S3Client):
    def __init__(self) -> None:
        super().__init__()
        self.content_upload_started = ThreadEvent()
        self.release_content_upload = ThreadEvent()
        self.content_upload_finished = ThreadEvent()

    def put_object(self, **kwargs: Any) -> dict[str, Any]:
        if kwargs["Key"].endswith("/content"):
            self.content_upload_started.set()
            self.release_content_upload.wait()
            try:
                return super().put_object(**kwargs)
            finally:
                self.content_upload_finished.set()
        return super().put_object(**kwargs)


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


@pytest.mark.parametrize("failed_suffix", ["/content", "/metadata.json"])
def test_s3_artifact_store_rejects_partial_delete_errors(failed_suffix: str) -> None:
    client = _S3Client()
    store = S3ArtifactStore("bucket", client=client)
    artifact = asyncio.run(
        store.put_bytes(b"content", filename="artifact.txt", session_id="sess_1")
    )
    client.delete_errors_by_suffix[failed_suffix] = "AccessDenied"

    with pytest.raises(
        ArtifactStoreUnavailableError,
        match="DeleteObjects returned 1 per-object error.*AccessDenied",
    ):
        asyncio.run(store.delete(artifact.id))

    assert ("bucket", f"cayu/artifacts/{artifact.id}{failed_suffix}") in client.objects


def test_s3_artifact_store_rejects_errors_for_every_deleted_object() -> None:
    client = _S3Client()
    store = S3ArtifactStore("bucket", client=client)
    artifact = asyncio.run(
        store.put_bytes(b"content", filename="artifact.txt", session_id="sess_1")
    )
    client.delete_errors_by_suffix = {
        "/content": "AccessDenied",
        "/metadata.json": "InternalError",
    }

    with pytest.raises(
        ArtifactStoreUnavailableError,
        match=(
            "DeleteObjects returned 2 per-object errors "
            r"\(codes: AccessDenied, InternalError\)"
        ),
    ):
        asyncio.run(store.delete(artifact.id))


def test_s3_artifact_store_keeps_delete_error_diagnostics_bounded() -> None:
    client = _S3Client()
    store = S3ArtifactStore("bucket", client=client)
    artifact = asyncio.run(
        store.put_bytes(b"content", filename="artifact.txt", session_id="sess_1")
    )
    secret = "credential-secret-" * 100
    client.delete_errors_by_suffix["/content"] = secret

    with pytest.raises(ArtifactStoreUnavailableError) as raised:
        asyncio.run(store.delete(artifact.id))

    assert str(raised.value) == ("S3 DeleteObjects returned 1 per-object error (codes: Unknown).")
    assert secret not in str(raised.value)


def test_s3_artifact_store_preserves_typed_backend_delete_failures() -> None:
    client = _S3Client()
    client.fail_delete_code = "ServiceUnavailable"
    store = S3ArtifactStore("bucket", client=client)

    with pytest.raises(ArtifactStoreUnavailableError, match="could not delete") as raised:
        asyncio.run(store.delete(f"art_{'5' * 32}"))

    assert isinstance(raised.value.__cause__, _ClientError)


def test_s3_artifact_store_delete_remains_idempotent_for_missing_objects() -> None:
    client = _S3Client()
    store = S3ArtifactStore("bucket", client=client)

    asyncio.run(store.delete(f"art_{'6' * 32}"))

    assert client.objects == {}


def test_s3_artifact_store_lists_all_metadata_then_applies_limit() -> None:
    client = _S3Client()
    store = S3ArtifactStore("bucket", client=client)
    first = asyncio.run(store.put_bytes(b"one", filename="one.txt", session_id="sess_1"))
    second = asyncio.run(store.put_bytes(b"two", filename="two.txt", session_id="sess_1"))

    result = asyncio.run(store.list(session_id="sess_1", limit=1))

    assert result.artifacts in ((first,), (second,))
    assert result.total_count == 2
    assert result.truncated is True


def test_s3_artifact_store_reuses_a_supplied_identity_only_for_an_exact_match() -> None:
    client = _S3Client()
    store = S3ArtifactStore("bucket", client=client)
    artifact_id = f"art_{'1' * 32}"
    kwargs = {
        "artifact_id": artifact_id,
        "filename": "tool-result.txt",
        "content_type": "text/plain",
        "session_id": "sess_1",
        "metadata": {"type": "cayu.tool_result_artifact.v1"},
    }

    first = asyncio.run(store.put_bytes(b"stable", **kwargs))
    replayed = asyncio.run(store.put_bytes(b"stable", **kwargs))

    assert replayed == first
    assert asyncio.run(store.list(session_id="sess_1")).artifacts == (first,)
    with pytest.raises(ValueError, match="different content or metadata"):
        asyncio.run(store.put_bytes(b"changed", **kwargs))
    with pytest.raises(ValueError, match="different content or metadata"):
        asyncio.run(store.put_bytes(b"stable", **{**kwargs, "filename": "changed.txt"}))


def test_s3_supplied_identity_finishes_metadata_commit_after_retry() -> None:
    client = _S3Client()
    client.fail_put_once_suffix = "metadata.json"
    store = S3ArtifactStore("bucket", client=client)
    artifact_id = f"art_{'2' * 32}"
    kwargs = {
        "artifact_id": artifact_id,
        "filename": "tool-result.txt",
        "content_type": "text/plain",
        "session_id": "sess_1",
        "metadata": {"type": "cayu.tool_result_artifact.v1"},
    }

    with pytest.raises(ArtifactStoreUnavailableError, match="commit"):
        asyncio.run(store.put_bytes(b"recoverable", **kwargs))

    assert ("bucket", f"cayu/artifacts/{artifact_id}/content") in client.objects
    assert ("bucket", f"cayu/artifacts/{artifact_id}/metadata.json") not in client.objects

    recovered = asyncio.run(store.put_bytes(b"recoverable", **kwargs))

    assert recovered.id == artifact_id
    assert asyncio.run(store.read_bytes(artifact_id)).content == b"recoverable"
    assert asyncio.run(store.list(session_id="sess_1")).artifacts == (recovered,)


def test_s3_cancelled_content_upload_does_not_leave_an_unlisted_object() -> None:
    client = _BlockingContentUploadS3Client()
    store = S3ArtifactStore("bucket", client=client)
    artifact_id = f"art_{'3' * 32}"

    async def scenario() -> None:
        put_task = asyncio.create_task(
            store.put_bytes(
                b"recoverable",
                artifact_id=artifact_id,
                filename="tool-result.txt",
                content_type="text/plain",
                session_id="sess_1",
                metadata={"type": "cayu.tool_result_artifact.v1"},
            )
        )
        try:
            assert await asyncio.to_thread(client.content_upload_started.wait, 1)

            put_task.cancel("projection timeout")
            await asyncio.sleep(0)
        finally:
            client.release_content_upload.set()

        with pytest.raises(asyncio.CancelledError, match="projection timeout"):
            await put_task
        assert await asyncio.to_thread(client.content_upload_finished.wait, 1)

        listed = await store.list(session_id="sess_1")
        assert [artifact.id for artifact in listed.artifacts] == [artifact_id]
        read = await store.read_bytes(artifact_id)
        assert read.content == b"recoverable"
        assert read.metadata.id == artifact_id

    asyncio.run(scenario())


def test_s3_cancelled_content_upload_is_removed_when_metadata_commit_fails() -> None:
    client = _BlockingContentUploadS3Client()
    client.fail_put_suffix = "metadata.json"
    store = S3ArtifactStore("bucket", client=client)
    artifact_id = f"art_{'4' * 32}"

    async def scenario() -> None:
        put_task = asyncio.create_task(
            store.put_bytes(
                b"recoverable",
                artifact_id=artifact_id,
                filename="tool-result.txt",
                content_type="text/plain",
                session_id="sess_1",
                metadata={"type": "cayu.tool_result_artifact.v1"},
            )
        )
        try:
            assert await asyncio.to_thread(client.content_upload_started.wait, 1)

            put_task.cancel("projection timeout")
            await asyncio.sleep(0)
        finally:
            client.release_content_upload.set()

        with pytest.raises(asyncio.CancelledError, match="projection timeout"):
            await put_task
        assert await asyncio.to_thread(client.content_upload_finished.wait, 1)

        assert client.objects == {}
        assert len(client.delete_calls) == 1
        assert (await store.list(session_id="sess_1")).artifacts == ()

    asyncio.run(scenario())


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

    with pytest.raises(ArtifactStoreUnavailableError, match="commit"):
        asyncio.run(store.put_bytes(b"orphan", filename="orphan.txt", session_id="sess_1"))

    assert client.objects == {}
    assert len(client.delete_calls) == 1


def test_s3_artifact_store_keeps_failed_write_cleanup_best_effort() -> None:
    client = _S3Client()
    client.fail_put_suffix = "metadata.json"
    client.delete_errors_by_suffix["/content"] = "AccessDenied"
    store = S3ArtifactStore("bucket", client=client)

    with pytest.raises(ArtifactStoreUnavailableError, match="commit"):
        asyncio.run(store.put_bytes(b"orphan", filename="orphan.txt", session_id="sess_1"))

    assert any(key.endswith("/content") for _, key in client.objects)
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

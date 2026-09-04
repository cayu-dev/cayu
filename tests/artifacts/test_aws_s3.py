from __future__ import annotations

import asyncio
import io
from threading import Barrier as ThreadBarrier
from threading import Event as ThreadEvent
from threading import Lock as ThreadLock
from typing import Any

import pytest

from cayu import (
    ArtifactScope,
    ArtifactStoreUnavailableError,
    ArtifactWriteSettlementFailureCode,
    ArtifactWriteSettlementStatus,
    InvalidArtifactIdError,
    S3ArtifactStore,
    artifact_write_settlements,
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


class _BlockingMetadataCommitS3Client(_S3Client):
    def __init__(self) -> None:
        super().__init__()
        self.metadata_commit_started = ThreadEvent()
        self.release_metadata_commit = ThreadEvent()

    def put_object(self, **kwargs: Any) -> dict[str, Any]:
        if kwargs["Key"].endswith("/metadata.json"):
            self.metadata_commit_started.set()
            if not self.release_metadata_commit.wait(timeout=2):
                raise TimeoutError("test did not release S3 metadata commit")
        return super().put_object(**kwargs)


class _WriteThenLoseAcknowledgementS3Client(_S3Client):
    def __init__(self, suffix: str) -> None:
        super().__init__()
        self.suffix = suffix
        self.lost = False

    def put_object(self, **kwargs: Any) -> dict[str, Any]:
        response = super().put_object(**kwargs)
        if not self.lost and kwargs["Key"].endswith(self.suffix):
            self.lost = True
            raise _ClientError("AcknowledgementLost")
        return response


class _ConcurrentContentUploadS3Client(_S3Client):
    def __init__(self) -> None:
        super().__init__()
        self.content_barrier = ThreadBarrier(2)
        self.write_lock = ThreadLock()

    def put_object(self, **kwargs: Any) -> dict[str, Any]:
        if kwargs["Key"].endswith("/content"):
            self.content_barrier.wait(timeout=2)
        with self.write_lock:
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
        "scope": ArtifactScope.SESSION,
        "session_id": "sess_1",
        "agent_name": "assistant",
        "environment_name": "local",
        "metadata": {"alpha": 1, "enabled": False},
    }

    first = asyncio.run(store.put_bytes(b"stable", **kwargs))
    replayed = asyncio.run(
        store.put_bytes(
            b"stable",
            **{**kwargs, "metadata": {"enabled": False, "alpha": 1}},
        )
    )

    assert replayed == first
    assert asyncio.run(store.list(session_id="sess_1")).artifacts == (first,)
    conflicts = (
        (b"changed", {}),
        (b"stable", {"filename": "changed.txt"}),
        (b"stable", {"content_type": "application/octet-stream"}),
        (b"stable", {"scope": ArtifactScope.ENVIRONMENT}),
        (b"stable", {"session_id": "sess_changed"}),
        (b"stable", {"agent_name": "reviewer"}),
        (b"stable", {"environment_name": "remote"}),
        (b"stable", {"metadata": {"alpha": 1, "enabled": True}}),
    )
    for content, update in conflicts:
        with pytest.raises(ValueError, match="different content or metadata"):
            asyncio.run(store.put_bytes(content, **{**kwargs, **update}))

    assert asyncio.run(store.read_bytes(artifact_id)).metadata == first


def test_s3_concurrent_conflicting_writers_preserve_the_conditional_winner() -> None:
    client = _ConcurrentContentUploadS3Client()
    store = S3ArtifactStore("bucket", client=client)
    artifact_id = f"art_{'7' * 32}"

    async def scenario():
        return await asyncio.gather(
            store.put_bytes(
                b"first",
                artifact_id=artifact_id,
                filename="first.txt",
                session_id="sess_1",
            ),
            store.put_bytes(
                b"second",
                artifact_id=artifact_id,
                filename="second.txt",
                session_id="sess_1",
            ),
            return_exceptions=True,
        )

    outcomes = asyncio.run(scenario())

    successes = [outcome for outcome in outcomes if not isinstance(outcome, BaseException)]
    failures = [outcome for outcome in outcomes if isinstance(outcome, BaseException)]
    assert len(successes) == 1
    assert len(failures) == 1
    assert isinstance(failures[0], ValueError)
    committed = asyncio.run(store.read_bytes(artifact_id))
    assert committed.content in {b"first", b"second"}
    assert committed.metadata.filename == (
        "first.txt" if committed.content == b"first" else "second.txt"
    )
    assert client.delete_calls == []


def test_s3_concurrent_same_content_metadata_conflict_is_explicit() -> None:
    client = _ConcurrentContentUploadS3Client()
    store = S3ArtifactStore("bucket", client=client)
    artifact_id = f"art_{'8' * 32}"

    async def scenario():
        return await asyncio.gather(
            store.put_bytes(
                b"shared",
                artifact_id=artifact_id,
                filename="first.txt",
                session_id="sess_1",
            ),
            store.put_bytes(
                b"shared",
                artifact_id=artifact_id,
                filename="second.txt",
                session_id="sess_1",
            ),
            return_exceptions=True,
        )

    outcomes = asyncio.run(scenario())

    successes = [outcome for outcome in outcomes if not isinstance(outcome, BaseException)]
    failures = [outcome for outcome in outcomes if isinstance(outcome, BaseException)]
    assert len(successes) == 1
    assert len(failures) == 1
    assert isinstance(failures[0], ValueError)
    assert "different content or metadata" in str(failures[0])
    settlement = artifact_write_settlements(failures[0])
    assert len(settlement) == 1
    assert settlement[0].status is ArtifactWriteSettlementStatus.RECONCILIATION_REQUIRED
    assert settlement[0].failure_codes == (
        ArtifactWriteSettlementFailureCode.MUTATION_FAILED,
        ArtifactWriteSettlementFailureCode.COMMIT_FAILED,
    )
    committed = asyncio.run(store.read_bytes(artifact_id))
    assert committed.content == b"shared"
    assert committed.metadata.filename in {"first.txt", "second.txt"}
    assert client.delete_calls == []


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


@pytest.mark.parametrize("lost_suffix", ["/content", "/metadata.json"])
def test_s3_generated_identity_reconciles_lost_write_acknowledgement(
    lost_suffix: str,
) -> None:
    client = _WriteThenLoseAcknowledgementS3Client(lost_suffix)
    store = S3ArtifactStore("bucket", client=client)

    artifact = asyncio.run(
        store.put_bytes(b"recoverable", filename="generated.txt", session_id="sess_1")
    )

    assert client.lost is True
    assert asyncio.run(store.read_bytes(artifact.id)).content == b"recoverable"
    assert asyncio.run(store.list(session_id="sess_1")).artifacts == (artifact,)


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


def test_s3_child_task_cancellation_keeps_dispatched_request_owned(monkeypatch) -> None:
    client = _BlockingContentUploadS3Client()
    store = S3ArtifactStore("bucket", client=client)
    artifact_id = f"art_{'8' * 32}"
    child_tasks: list[asyncio.Task] = []
    real_operation = store._run_artifact_write

    async def capture_child(*args, **kwargs):
        child = asyncio.current_task()
        assert child is not None
        child_tasks.append(child)
        return await real_operation(*args, **kwargs)

    monkeypatch.setattr(store, "_run_artifact_write", capture_child)

    async def scenario():
        put_task = asyncio.create_task(
            store.put_bytes(
                b"owned-request",
                artifact_id=artifact_id,
                filename="owned.txt",
                content_type="text/plain",
                session_id="sess_1",
            )
        )
        assert await asyncio.to_thread(client.content_upload_started.wait, 1)
        assert len(child_tasks) == 1
        child_tasks[0].cancel("supervisor stopped S3 child")
        await asyncio.sleep(0)
        assert not put_task.done()

        client.release_content_upload.set()
        with pytest.raises(
            RuntimeError,
            match="S3 artifact publication was cancelled without caller cancellation",
        ) as raised:
            await put_task
        assert isinstance(raised.value.__cause__, asyncio.CancelledError)
        evidence = artifact_write_settlements(raised.value)
        assert len(evidence) == 1
        assert evidence[0].status is ArtifactWriteSettlementStatus.COMMITTED
        assert evidence[0].failure_codes == (ArtifactWriteSettlementFailureCode.CHILD_CANCELLED,)
        return put_task

    put_task = asyncio.run(scenario())

    assert not put_task.cancelled()
    assert child_tasks[0].done()
    assert child_tasks[0].cancelling() == 0
    assert not child_tasks[0].cancelled()
    assert client.content_upload_finished.is_set()
    assert asyncio.run(store.read_bytes(artifact_id)).content == b"owned-request"
    assert [item.id for item in asyncio.run(store.list(session_id="sess_1")).artifacts] == [
        artifact_id
    ]


def test_s3_cancelled_generated_write_retains_its_committed_identity() -> None:
    client = _BlockingContentUploadS3Client()
    store = S3ArtifactStore("bucket", client=client)

    async def scenario() -> None:
        put_task = asyncio.create_task(
            store.put_bytes(
                b"generated",
                filename="generated.txt",
                session_id="sess_1",
            )
        )
        assert await asyncio.to_thread(client.content_upload_started.wait, 1)
        put_task.cancel("caller stopped")
        client.release_content_upload.set()

        with pytest.raises(asyncio.CancelledError, match="caller stopped") as raised:
            await put_task
        evidence = artifact_write_settlements(raised.value)
        assert len(evidence) == 1
        assert evidence[0].status is ArtifactWriteSettlementStatus.COMMITTED
        read = await store.read_bytes(evidence[0].artifact_id)
        assert read.content == b"generated"

    asyncio.run(scenario())


def test_s3_cancellation_during_metadata_commit_preserves_commit_evidence() -> None:
    client = _BlockingMetadataCommitS3Client()
    store = S3ArtifactStore("bucket", client=client)
    artifact_id = f"art_{'8' * 32}"

    async def scenario() -> None:
        put_task = asyncio.create_task(
            store.put_bytes(
                b"commit-content",
                artifact_id=artifact_id,
                filename="commit.txt",
                session_id="sess_1",
            )
        )
        assert await asyncio.to_thread(client.metadata_commit_started.wait, 1)
        put_task.cancel("caller stopped during commit")
        assert not put_task.done()
        client.release_metadata_commit.set()

        with pytest.raises(
            asyncio.CancelledError,
            match="caller stopped during commit",
        ) as raised:
            await put_task
        evidence = artifact_write_settlements(raised.value)
        assert len(evidence) == 1
        assert evidence[0].artifact_id == artifact_id
        assert evidence[0].status is ArtifactWriteSettlementStatus.COMMITTED
        assert (await store.read_bytes(artifact_id)).content == b"commit-content"

    asyncio.run(scenario())


def test_s3_cancelled_content_upload_retains_a_fenced_orphan_candidate() -> None:
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

        with pytest.raises(asyncio.CancelledError, match="projection timeout") as raised:
            await put_task
        assert await asyncio.to_thread(client.content_upload_finished.wait, 1)

        evidence = artifact_write_settlements(raised.value)
        assert len(evidence) == 1
        assert evidence[0].artifact_id == artifact_id
        assert evidence[0].status is ArtifactWriteSettlementStatus.RECONCILIATION_REQUIRED
        assert evidence[0].failure_codes == (
            ArtifactWriteSettlementFailureCode.COMMIT_FAILED,
            ArtifactWriteSettlementFailureCode.RECONCILIATION_FAILED,
        )
        cancellation_cause = raised.value.__cause__
        assert isinstance(cancellation_cause, ArtifactStoreUnavailableError)
        combined = cancellation_cause.__cause__
        assert isinstance(combined, BaseExceptionGroup)
        assert len(combined.exceptions) == 2
        assert any(key.endswith("/content") for _, key in client.objects)
        assert client.delete_calls == []
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


def test_s3_generated_write_retains_identity_when_metadata_commit_fails() -> None:
    client = _S3Client()
    client.fail_put_suffix = "metadata.json"
    store = S3ArtifactStore("bucket", client=client)

    with pytest.raises(ArtifactStoreUnavailableError, match="commit") as raised:
        asyncio.run(store.put_bytes(b"orphan", filename="orphan.txt", session_id="sess_1"))

    evidence = artifact_write_settlements(raised.value)
    assert len(evidence) == 1
    assert evidence[0].artifact_id.startswith("art_")
    assert evidence[0].status is ArtifactWriteSettlementStatus.RECONCILIATION_REQUIRED
    assert any(key.endswith(f"/{evidence[0].artifact_id}/content") for _, key in client.objects)
    assert client.delete_calls == []


def test_s3_failed_write_never_uses_unfenced_final_key_cleanup() -> None:
    client = _S3Client()
    client.fail_put_suffix = "metadata.json"
    client.delete_errors_by_suffix["/content"] = "AccessDenied"
    store = S3ArtifactStore("bucket", client=client)

    with pytest.raises(ArtifactStoreUnavailableError, match="commit"):
        asyncio.run(store.put_bytes(b"orphan", filename="orphan.txt", session_id="sess_1"))

    assert any(key.endswith("/content") for _, key in client.objects)
    assert client.delete_calls == []


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

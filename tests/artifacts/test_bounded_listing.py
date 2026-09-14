from __future__ import annotations

import asyncio
import io
from datetime import UTC, datetime, timedelta

import pytest
from tests.artifacts.test_aws_s3 import _S3Client
from tests.core.session_closure_conformance import create_closure_session

from cayu import CayuApp
from cayu.artifacts import (
    ArtifactMetadata,
    ArtifactScope,
    ArtifactStoreUnavailableError,
    LocalArtifactStore,
    S3ArtifactStore,
    aws_s3,
    local,
)
from cayu.artifacts._listing import BoundedArtifactListing
from cayu.runtime.session_closure import (
    ArtifactSessionClosureStore,
    SessionClosureDisposition,
    SessionClosureExportIncomplete,
    SessionClosurePolicy,
)


def _observe_retained_selection(monkeypatch, module):
    sizes = []

    class ObservedListing(BoundedArtifactListing):
        def add(self, artifact):
            super().add(artifact)
            sizes.append(len(self._selected))

    monkeypatch.setattr(module, "BoundedArtifactListing", ObservedListing)
    return sizes


def test_local_closure_inventory_streams_directory_with_bounded_selection(tmp_path, monkeypatch):
    async def run():
        store = LocalArtifactStore(tmp_path / "artifacts")
        artifacts = [
            await store.put_bytes(b"test", filename=f"{index}.txt", session_id="selected")
            for index in range(9)
        ]
        await store.put_bytes(b"other", filename="other.txt", session_id="other")
        sizes = _observe_retained_selection(monkeypatch, local)

        def no_eager_directory_list(*args, **kwargs):
            raise AssertionError("Inventory must not materialize every directory name.")

        monkeypatch.setattr(local.os, "listdir", no_eager_directory_list)
        page = await store.list(scope=ArtifactScope.SESSION, session_id="selected", limit=2)
        assert page.artifacts == tuple(
            sorted(artifacts, key=lambda item: item.created_at, reverse=True)[:2]
        )
        assert page.total_count == 9 and page.truncated
        record = await ArtifactSessionClosureStore(store).inspect_session_closure(
            "selected", policy=SessionClosurePolicy(max_records=2)
        )
        assert record.disposition is SessionClosureDisposition.TRUNCATED
        assert record.count == 2
        assert max(sizes) == 2

    asyncio.run(run())


class _PagedMetadataClient:
    def __init__(self):
        self.metadata_reads = 0
        self.page_starts = []

    def list_objects_v2(self, **kwargs):
        assert kwargs["MaxKeys"] == 1000
        start = int(kwargs.get("ContinuationToken", "0"))
        # The next object page must not be loaded before processing this one.
        assert self.metadata_reads == start
        self.page_starts.append(start)
        end = min(start + 7, 23)
        return {
            "Contents": [
                {"Key": f"{kwargs['Prefix']}art_{index:032x}/metadata.json"}
                for index in range(start, end)
            ],
            "IsTruncated": end < 23,
            "NextContinuationToken": str(end),
        }

    def get_object(self, **kwargs):
        index = int(kwargs["Key"].split("/")[-2].removeprefix("art_"), 16)
        self.metadata_reads += 1
        artifact = ArtifactMetadata(
            id=f"art_{index:032x}",
            filename="item.txt",
            size_bytes=1,
            session_id="selected" if index % 2 == 0 else "other",
            # Equal timestamps exercise stable selection as well as replacement.
            created_at=datetime(2026, 1, 1, tzinfo=UTC) + timedelta(seconds=index // 2),
        )
        payload = artifact.model_dump_json().encode()
        return {"Body": io.BytesIO(payload), "ContentLength": len(payload)}


def test_s3_listing_streams_metadata_pages_with_bounded_selection(monkeypatch):
    client = _PagedMetadataClient()
    store = S3ArtifactStore("bucket", client=client)
    sizes = _observe_retained_selection(monkeypatch, aws_s3)
    page = asyncio.run(store.list(session_id="selected", scope=ArtifactScope.SESSION, limit=3))
    assert [item.id for item in page.artifacts] == [f"art_{index:032x}" for index in (22, 20, 18)]
    assert page.total_count == 12 and page.truncated
    assert client.page_starts == [0, 7, 14, 21]
    assert max(sizes) == 3


@pytest.mark.parametrize("limit", [1, 2, 5, None])
def test_bounded_artifact_selection_preserves_equal_timestamp_order(limit):
    inventory = BoundedArtifactListing(limit)
    artifacts = [
        ArtifactMetadata(
            id=f"art_{index:032x}",
            filename="item",
            session_id="session",
            size_bytes=1,
            created_at=datetime(2026, 1, 1, tzinfo=UTC),
        )
        for index in range(5)
    ]
    for artifact in artifacts:
        inventory.add(artifact)
    result = inventory.result()
    assert result.artifacts == tuple(artifacts[:limit])
    assert result.total_count == 5


@pytest.mark.parametrize(
    "response",
    [
        {"Contents": [], "IsTruncated": "false"},
        {"Contents": [], "IsTruncated": True},
        {"Contents": [{}] * 1001, "IsTruncated": False},
    ],
)
def test_s3_listing_rejects_unproven_or_oversized_pages(response):
    class Client:
        def list_objects_v2(self, **kwargs):
            assert kwargs["MaxKeys"] == 1000
            return response

        def get_object(self, **kwargs):
            raise AssertionError("Invalid pages cannot authorize metadata reads.")

    with pytest.raises(ArtifactStoreUnavailableError):
        asyncio.run(S3ArtifactStore("bucket", client=Client()).list(limit=1))


@pytest.mark.parametrize("backend", ["local", "s3"])
def test_public_closure_cannot_ignore_unreadable_artifact_metadata(
    tmp_path, backend, capsys, caplog, recwarn
):
    canary = "private-malformed-artifact-metadata"

    async def run():
        if backend == "local":
            store = LocalArtifactStore(tmp_path / "invalid")
        else:
            client = _S3Client()
            store = S3ArtifactStore("bucket", client=client)
        valid = await store.put_bytes(b"valid", filename="valid.txt", session_id="selected")
        invalid = await store.put_bytes(b"invalid", filename="invalid.txt", session_id="selected")
        payload = canary.encode()
        if isinstance(store, LocalArtifactStore):
            (store.root / invalid.id / "metadata.json").write_bytes(payload)
        else:
            client.objects[("bucket", store._artifact_key(invalid.id, "metadata.json"))] = payload
        app = CayuApp(session_closure_stores=(ArtifactSessionClosureStore(store),))
        await create_closure_session(app.session_store, "selected")
        report = await app.erase_session_closure("selected")
        assert not report.complete
        record = next(
            item
            for item in report.manifest.records
            if item.record_class == "session_dependents"
            and item.store_id.startswith("artifact-store:")
        )
        assert record.disposition is SessionClosureDisposition.UNAVAILABLE
        assert await app.session_store.load("selected") is not None
        assert (await store.read_bytes(valid.id)).content == b"valid"
        assert canary not in report.model_dump_json()

    asyncio.run(run())
    captured = capsys.readouterr()
    assert canary not in captured.out + captured.err + caplog.text
    assert all(canary not in str(warning.message) for warning in recwarn)


@pytest.mark.parametrize("missing_file", ["metadata.json", "content"])
@pytest.mark.parametrize("entrance", ["inspect", "export", "erase"])
def test_public_closure_rejects_incomplete_surviving_local_artifact(
    tmp_path, missing_file, entrance
):
    async def run():
        store = LocalArtifactStore(tmp_path / "artifacts")
        valid = await store.put_bytes(b"valid", filename="valid.txt", session_id="selected")
        incomplete = await store.put_bytes(
            b"private-content", filename="incomplete.txt", session_id="selected"
        )
        directory = store.root / incomplete.id
        (directory / missing_file).unlink()
        surviving_file = "content" if missing_file == "metadata.json" else "metadata.json"
        retained = (directory / surviving_file).read_bytes()
        app = CayuApp(session_closure_stores=(ArtifactSessionClosureStore(store),))
        await create_closure_session(app.session_store, "selected")
        if entrance == "export":
            with pytest.raises(SessionClosureExportIncomplete) as error:
                await app.export_session_closure("selected")
            manifest = error.value.manifest
        elif entrance == "erase":
            report = await app.erase_session_closure("selected")
            assert not report.complete
            manifest = report.manifest
        else:
            manifest = await app.inspect_session_closure("selected")
        assert not manifest.complete
        assert any(
            record.store_id == f"artifact-store:{store.id}"
            and record.disposition is SessionClosureDisposition.UNAVAILABLE
            for record in manifest.records
        )
        assert await app.session_store.load("selected") is not None
        assert await store.load_session_closure_claim("selected") is None
        assert (await store.read_bytes(valid.id)).content == b"valid"
        assert (directory / surviving_file).read_bytes() == retained

    asyncio.run(run())


def test_local_listing_allows_a_genuinely_disappeared_artifact(tmp_path, monkeypatch):
    async def run():
        store = LocalArtifactStore(tmp_path / "artifacts")
        artifact = await store.put_bytes(b"content", filename="item", session_id="selected")
        original = local._load_metadata

        def disappear(path, *, parent_fd=None):
            if path.name == artifact.id:
                (path / "metadata.json").unlink()
                (path / "content").unlink()
                path.rmdir()
            return original(path, parent_fd=parent_fd)

        monkeypatch.setattr(local, "_load_metadata", disappear)
        result = await store.list(session_id="selected", limit=1)
        assert result.artifacts == ()
        assert result.total_count == 0
        assert not result.truncated

    asyncio.run(run())


@pytest.mark.parametrize("entrance", ["inspect", "export", "erase"])
def test_public_closure_rejects_s3_content_after_partial_deletion(entrance):
    async def run():
        client = _S3Client()
        store = S3ArtifactStore("bucket", client=client)
        valid = await store.put_bytes(b"valid", filename="valid", session_id="selected")
        incomplete = await store.put_bytes(
            b"private-content", filename="item", session_id="selected"
        )
        client.delete_errors_by_suffix["/content"] = "AccessDenied"
        with pytest.raises(ArtifactStoreUnavailableError):
            await store.delete(incomplete.id)
        client.delete_errors_by_suffix.clear()
        content_key = ("bucket", store._artifact_key(incomplete.id, "content"))
        assert content_key in client.objects
        assert ("bucket", store._artifact_key(incomplete.id, "metadata.json")) not in client.objects
        delete_count = len(client.delete_calls)
        app = CayuApp(session_closure_stores=(ArtifactSessionClosureStore(store),))
        await create_closure_session(app.session_store, "selected")
        if entrance == "export":
            with pytest.raises(SessionClosureExportIncomplete) as error:
                await app.export_session_closure("selected")
            manifest = error.value.manifest
        elif entrance == "erase":
            report = await app.erase_session_closure("selected")
            assert not report.complete
            manifest = report.manifest
        else:
            manifest = await app.inspect_session_closure("selected")
        assert not manifest.complete
        assert any(
            record.store_id == f"artifact-store:{store.id}"
            and record.disposition is SessionClosureDisposition.UNAVAILABLE
            for record in manifest.records
        )
        assert await app.session_store.load("selected") is not None
        assert await store.load_session_closure_claim("selected") is None
        assert len(client.delete_calls) == delete_count
        assert client.objects[content_key] == b"private-content"
        assert (await store.read_bytes(valid.id)).content == b"valid"

    asyncio.run(run())


def test_s3_listing_allows_objects_deleted_after_listing():
    class DisappearingClient(_S3Client):
        disappear = False

        def list_objects_v2(self, **kwargs):
            response = super().list_objects_v2(**kwargs)
            if self.disappear:
                for entry in response["Contents"]:
                    key = entry["Key"]
                    if key.endswith(("/content", "/metadata.json")):
                        self.objects.pop((kwargs["Bucket"], key), None)
            return response

    async def run():
        client = DisappearingClient()
        store = S3ArtifactStore("bucket", client=client)
        await store.put_bytes(b"content", filename="item", session_id="selected")
        client.disappear = True
        result = await store.list(session_id="selected", limit=1)
        assert result.artifacts == ()
        assert result.total_count == 0
        assert not result.truncated

    asyncio.run(run())

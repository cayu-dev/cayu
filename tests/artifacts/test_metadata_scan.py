"""Optional complete metadata scans retain bounded batches and cleanup ownership."""

import asyncio

import pytest
from tests.artifacts.test_aws_s3 import _S3Client

from cayu.artifacts import (
    ArtifactScope,
    ArtifactStoreUnavailableError,
    LocalArtifactStore,
    S3ArtifactStore,
)


@pytest.mark.parametrize("backend", ["local", "s3"])
def test_native_scan_is_complete_filtered_and_batched(tmp_path, backend):
    async def run():
        client = _S3Client()
        store = (
            LocalArtifactStore(tmp_path)
            if backend == "local"
            else S3ArtifactStore("fake", client=client)
        )
        for index in range(7):
            await store.put_bytes(
                b"not searched",
                filename=f"file{index}",
                scope=ArtifactScope.ENVIRONMENT,
                environment_name="yes" if index % 2 == 0 else "no",
            )
        batches = []

        async def consume(batch):
            batches.append(batch)

        total = await store.scan_metadata(consume, environment_name="yes", batch_size=2)
        assert total == 4
        assert [len(batch) for batch in batches] == [2, 2]
        assert {row.filename for batch in batches for row in batch} == {
            "file0",
            "file2",
            "file4",
            "file6",
        }
        baseline = await store.list(environment_name="yes", limit=1)
        assert baseline.total_count == 4 and len(baseline.artifacts) == 1
        batches.clear()
        with pytest.raises(ArtifactStoreUnavailableError):
            await store.scan_metadata(consume, batch_size=2, max_records=3)
        assert sum(map(len, batches)) <= 3

    asyncio.run(run())


@pytest.mark.parametrize("backend", ["local", "s3"])
def test_scan_cancellation_drains_dispatched_reader_before_cleanup(tmp_path, backend, monkeypatch):
    from threading import Event

    from cayu.artifacts import local

    async def run():
        entered, release, finished = Event(), Event(), Event()
        client = _S3Client()
        store = (
            LocalArtifactStore(tmp_path)
            if backend == "local"
            else S3ArtifactStore("fake", client=client)
        )
        await store.put_bytes(b"synthetic", filename="file", session_id="producer")
        if backend == "local":
            original = local._iter_artifacts

            def blocked(*args):
                entered.set()
                assert release.wait(5)
                try:
                    yield from original(*args)
                finally:
                    finished.set()

            monkeypatch.setattr(local, "_iter_artifacts", blocked)
        else:
            original = client.list_objects_v2

            def blocked(**kwargs):
                entered.set()
                assert release.wait(5)
                try:
                    return original(**kwargs)
                finally:
                    finished.set()

            monkeypatch.setattr(client, "list_objects_v2", blocked)
        batches = []

        async def consume(batch):
            batches.append(batch)

        task = asyncio.create_task(store.scan_metadata(consume, batch_size=1))
        assert await asyncio.to_thread(entered.wait, 5)
        task.cancel()
        assert task.cancelling() == 1
        await asyncio.sleep(0)
        # The owned drain temporarily consumes cancellation, then restores it.
        assert not task.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert task.cancelled() and task.cancelling() == 1 and finished.is_set() and batches == []
        # The iterator/SDK read is closed before another caller can start.
        assert await store.scan_metadata(consume, batch_size=1) == 1

    asyncio.run(run())


@pytest.mark.parametrize("backend", ["local", "s3"])
def test_scan_consumer_failure_stops_delivery_and_invalid_bounds_do_not_read(tmp_path, backend):
    async def run():
        store = (
            LocalArtifactStore(tmp_path)
            if backend == "local"
            else S3ArtifactStore("fake", client=_S3Client())
        )
        for index in range(3):
            await store.put_bytes(b"synthetic", filename=f"file-{index}", session_id="producer")
        batches = []

        async def consume(batch):
            batches.append(batch)
            raise OSError("consumer rejected batch")

        with pytest.raises(OSError, match="consumer rejected"):
            await store.scan_metadata(consume, batch_size=1)
        assert len(batches) == 1
        for kwargs in (
            {"batch_size": True},
            {"batch_size": 501},
            {"max_records": False},
            {"max_records": 100001},
        ):
            with pytest.raises(ValueError):
                await store.scan_metadata(consume, **kwargs)
        assert len(batches) == 1

    asyncio.run(run())


@pytest.mark.parametrize("backend", ["local", "s3"])
def test_new_metadata_and_title_reads_remain_operator_only(tmp_path, backend):
    from tests.core.test_session_access import rule

    from cayu._resource_access_binding import ResourceExecutionBinding
    from cayu.resource_access import (
        ResourceAccessPolicy,
        encode_scope,
        execution_access,
        model_data_access,
    )
    from cayu.sessions._access_rules import SessionAccessScope
    from cayu.sessions.display import SessionTitleQuery
    from cayu.sessions.memory import InMemorySessionStore

    async def run():
        scope = SessionAccessScope(
            read=(rule(organization="acme"),), execute=(rule(organization="acme"),)
        )

        class Policy(ResourceAccessPolicy):
            authority = "artifact-scan-test"

            async def resolve(self, subject):
                return scope

        policy = Policy()
        binding = ResourceExecutionBinding(
            authority=policy.authority, subject="reader", admitted_json=encode_scope(scope)
        )
        store = (
            LocalArtifactStore(tmp_path)
            if backend == "local"
            else S3ArtifactStore("fake", client=_S3Client())
        )

        async def forbidden(batch):
            raise AssertionError("model must not receive an operator inventory")

        async with execution_access(binding, policy, {"organization": "acme"}), model_data_access():
            with pytest.raises(NotImplementedError, match="trusted operator"):
                await store.scan_metadata(forbidden)
            with pytest.raises(NotImplementedError, match="trusted operator"):
                await InMemorySessionStore().read_session_titles(
                    SessionTitleQuery(session_ids=("private",))
                )
        assert await store.scan_metadata(forbidden) == 0

    asyncio.run(run())

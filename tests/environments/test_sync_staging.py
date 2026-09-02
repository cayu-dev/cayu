from __future__ import annotations

import asyncio
import tempfile

import pytest

from cayu.environments import SyncBindingStagingCapacity, SyncBindingStagingCapacityError
from cayu.environments._sync_staging import _SealedTarArchive


def test_sync_staging_capacity_validates_configuration() -> None:
    with pytest.raises(ValueError, match="max_concurrency"):
        SyncBindingStagingCapacity(max_concurrency=0)
    with pytest.raises(TypeError, match="max_concurrency"):
        SyncBindingStagingCapacity(max_concurrency=True)
    with pytest.raises(ValueError, match="max_staged_bytes"):
        SyncBindingStagingCapacity(max_staged_bytes=0)
    with pytest.raises(TypeError, match="max_staged_bytes"):
        SyncBindingStagingCapacity(max_staged_bytes=True)
    with pytest.raises(TypeError, match="reuse_sealed_archives"):
        SyncBindingStagingCapacity(reuse_sealed_archives=1)


def test_sync_staging_transfer_admission_is_fifo_and_bounded_for_100_waiters() -> None:
    capacity = SyncBindingStagingCapacity(max_concurrency=1, max_staged_bytes=1024)
    admitted: list[int] = []

    async def run() -> None:
        async def worker(index: int) -> None:
            lease = await capacity._acquire_transfer()
            try:
                admitted.append(index)
                await asyncio.sleep(0)
            finally:
                lease.release()

        await asyncio.gather(*(worker(index) for index in range(100)))

    asyncio.run(run())

    assert admitted == list(range(100))
    snapshot = capacity.snapshot()
    assert snapshot.active_transfers == 0
    assert snapshot.waiting_transfers == 0
    assert snapshot.peak_active_transfers == 1
    assert snapshot.total_transfer_admissions == 100


def test_sync_staging_prioritizes_admitted_byte_waiters_without_deadlock() -> None:
    capacity = SyncBindingStagingCapacity(max_concurrency=2, max_staged_bytes=10)
    order: list[str] = []

    async def run() -> None:
        first_transfer = await capacity._acquire_transfer()
        second_transfer = await capacity._acquire_transfer()
        first_bytes = await capacity._reserve_bytes(10)

        async def reserve_second() -> None:
            lease = await capacity._reserve_bytes(10)
            order.append("bytes")
            lease.release()
            second_transfer.release()

        async def admit_third() -> None:
            lease = await capacity._acquire_transfer()
            order.append("transfer")
            lease.release()

        second = asyncio.create_task(reserve_second())
        await asyncio.sleep(0)
        third = asyncio.create_task(admit_third())
        await asyncio.sleep(0)
        first_bytes.release()
        first_transfer.release()
        await asyncio.gather(second, third)

    asyncio.run(run())

    assert order == ["bytes", "transfer"]
    assert capacity.snapshot().active_transfers == 0
    assert capacity.snapshot().staged_bytes == 0


def test_sync_staging_cancelled_waiter_releases_exact_accounting() -> None:
    capacity = SyncBindingStagingCapacity(max_concurrency=1, max_staged_bytes=10)

    async def run() -> None:
        owner = await capacity._acquire_transfer()
        waiter = asyncio.create_task(capacity._acquire_transfer())
        await asyncio.sleep(0.01)
        waiting_snapshot = capacity.snapshot()
        assert waiting_snapshot.waiting_transfers == 1
        assert waiting_snapshot.oldest_transfer_wait_seconds > 0
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        assert capacity.snapshot().waiting_transfers == 0
        owner.release()

    asyncio.run(run())

    snapshot = capacity.snapshot()
    assert snapshot.active_transfers == 0
    assert snapshot.staged_bytes == 0
    assert snapshot.total_transfer_wait_seconds > 0


def test_sync_staging_rejects_archive_larger_than_capacity_without_waiting() -> None:
    capacity = SyncBindingStagingCapacity(max_concurrency=1, max_staged_bytes=10)

    with pytest.raises(SyncBindingStagingCapacityError, match="required=11, capacity=10"):
        asyncio.run(capacity._reserve_bytes(11))

    assert capacity.snapshot().waiting_bytes == 0


def test_sync_staging_closes_last_archive_reference_when_reservation_shrink_fails() -> None:
    capacity = SyncBindingStagingCapacity(max_concurrency=1, max_staged_bytes=8)

    async def run() -> None:
        async def build() -> _SealedTarArchive:
            lease = await capacity._reserve_bytes(4)
            spool = tempfile.TemporaryFile(  # noqa: SIM115 - archive owns the spool
                mode="w+b",
                prefix="cayu-test-invalid-sync-tar-",
            )
            spool.write(b"invalid")
            return _SealedTarArchive(
                spool,
                archive_bytes=7,
                logical_bytes=7,
                capacity_lease=lease,
            )

        reference = await capacity._acquire_archive("invalid-reservation", build)
        with pytest.raises(ValueError, match="may only shrink"):
            reference.release()

    asyncio.run(run())

    snapshot = capacity.snapshot()
    assert snapshot.staged_bytes == 0
    assert snapshot.shared_archives == 0
    assert snapshot.archive_references == 0
    assert snapshot.total_archive_cleanups == 1


def test_sync_staging_concurrent_archive_reuse_owns_one_spool_and_byte_lease() -> None:
    capacity = SyncBindingStagingCapacity(max_concurrency=100, max_staged_bytes=1024)
    build_started = asyncio.Event()
    finish_build = asyncio.Event()
    release_references = asyncio.Event()
    acquired = 0
    build_calls = 0

    async def run() -> None:
        nonlocal acquired, build_calls

        async def build() -> _SealedTarArchive:
            nonlocal build_calls
            build_calls += 1
            byte_lease = await capacity._reserve_bytes(64)
            spool = tempfile.TemporaryFile(  # noqa: SIM115 - ownership moves to the archive
                mode="w+b", prefix="cayu-test-sync-tar-"
            )
            spool.write(b"archive")
            spool.flush()
            build_started.set()
            await finish_build.wait()
            return _SealedTarArchive(
                spool,
                archive_bytes=7,
                logical_bytes=7,
                capacity_lease=byte_lease,
            )

        async def worker() -> None:
            nonlocal acquired
            reference = await capacity._acquire_archive("same-content", build)
            try:
                reader = reference.archive.open_reader()
                try:
                    assert reader.read() == b"archive"
                finally:
                    reader.close()
                acquired += 1
                await release_references.wait()
            finally:
                reference.release()

        workers = [asyncio.create_task(worker()) for _ in range(100)]
        await build_started.wait()
        await asyncio.sleep(0)
        finish_build.set()
        while acquired < 100:
            await asyncio.sleep(0)
        snapshot = capacity.snapshot()
        assert snapshot.shared_archives == 1
        assert snapshot.archive_references == 100
        # The builder retains its admitted transient allowance until its own
        # reference settles; followers cannot create an upgrade deadlock.
        assert snapshot.staged_bytes == 64
        release_references.set()
        await asyncio.gather(*workers)

    asyncio.run(run())

    snapshot = capacity.snapshot()
    assert build_calls == 1
    assert snapshot.shared_archives == 0
    assert snapshot.archive_references == 0
    assert snapshot.staged_bytes == 0
    assert snapshot.total_archive_builds == 1
    assert snapshot.total_archive_reuses == 99
    assert snapshot.total_archive_cleanups == 1


def test_sync_staging_cancelled_follower_does_not_cancel_shared_archive_build() -> None:
    capacity = SyncBindingStagingCapacity(max_concurrency=2, max_staged_bytes=1024)
    build_started = asyncio.Event()
    finish_build = asyncio.Event()

    async def run() -> None:
        async def build() -> _SealedTarArchive:
            byte_lease = await capacity._reserve_bytes(64)
            spool = tempfile.TemporaryFile(  # noqa: SIM115 - ownership moves to the archive
                mode="w+b", prefix="cayu-test-sync-tar-"
            )
            spool.write(b"archive")
            build_started.set()
            await finish_build.wait()
            byte_lease.shrink(7)
            return _SealedTarArchive(
                spool,
                archive_bytes=7,
                logical_bytes=7,
                capacity_lease=byte_lease,
            )

        creator = asyncio.create_task(capacity._acquire_archive("shared", build))
        await build_started.wait()
        follower = asyncio.create_task(capacity._acquire_archive("shared", build))
        while capacity.snapshot().archive_references < 2:
            await asyncio.sleep(0)

        follower.cancel("follower stopped")
        with pytest.raises(asyncio.CancelledError, match="follower stopped"):
            await follower
        assert capacity.snapshot().archive_references == 1

        finish_build.set()
        reference = await creator
        try:
            reader = reference.archive.open_reader()
            try:
                assert reader.read() == b"archive"
            finally:
                reader.close()
        finally:
            reference.release()

    asyncio.run(run())

    snapshot = capacity.snapshot()
    assert snapshot.shared_archives == 0
    assert snapshot.archive_references == 0
    assert snapshot.staged_bytes == 0
    assert snapshot.total_archive_builds == 1
    assert snapshot.total_archive_reuses == 0
    assert snapshot.total_archive_cleanups == 1

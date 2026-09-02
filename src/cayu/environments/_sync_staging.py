from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
import io
import threading
import time
from collections import deque
from collections.abc import Awaitable, Callable, Hashable
from dataclasses import dataclass
from typing import BinaryIO, Literal

DEFAULT_SYNC_STAGING_MAX_CONCURRENCY = 4
DEFAULT_SYNC_STAGING_MAX_BYTES = 512 * 1024 * 1024


class SyncBindingStagingCapacityError(RuntimeError):
    """A SyncBinding transfer cannot fit inside its staging capacity domain."""


@dataclass(frozen=True, slots=True)
class SyncBindingStagingSnapshot:
    """Bounded content-free state for one process-local staging capacity domain."""

    max_concurrency: int
    max_staged_bytes: int
    active_transfers: int
    staged_bytes: int
    waiting_transfers: int
    waiting_bytes: int
    shared_archives: int
    archive_references: int
    peak_active_transfers: int
    peak_staged_bytes: int
    total_transfer_admissions: int
    total_archive_builds: int
    total_archive_reuses: int
    total_archive_cleanups: int
    total_transfer_wait_seconds: float
    total_byte_wait_seconds: float
    oldest_transfer_wait_seconds: float
    oldest_byte_wait_seconds: float


@dataclass(slots=True)
class _CapacityWaiter:
    kind: Literal["transfer", "bytes"]
    amount: int
    loop: asyncio.AbstractEventLoop
    future: asyncio.Future[None]
    enqueued_at: float
    granted: bool = False
    handed_off: bool = False
    wait_recorded: bool = False


class _CapacityLease:
    def __init__(
        self,
        capacity: SyncBindingStagingCapacity,
        *,
        kind: Literal["transfer", "bytes"],
        amount: int,
    ) -> None:
        self._capacity = capacity
        self._kind = kind
        self._amount = amount
        self._released = False
        self._lock = threading.Lock()

    @property
    def amount(self) -> int:
        with self._lock:
            return self._amount

    def shrink(self, amount: int) -> None:
        if type(amount) is not int:
            raise TypeError("SyncBinding staging lease amount must be an integer.")
        with self._lock:
            if self._released:
                raise RuntimeError("SyncBinding staging lease was already released.")
            if amount < 0 or amount > self._amount:
                raise ValueError(
                    "SyncBinding staging lease may only shrink to a non-negative size."
                )
            released = self._amount - amount
            self._amount = amount
        if released:
            self._capacity._release(self._kind, released)

    def release(self) -> None:
        with self._lock:
            if self._released:
                return
            self._released = True
            amount = self._amount
            self._amount = 0
        self._capacity._release(self._kind, amount)

    def record_archive_cleanup(self) -> None:
        self._capacity._record_archive_cleanup()


class _ArchiveReader(io.RawIOBase):
    """Independent bounded-position view over one shared immutable file."""

    def __init__(self, archive: _SealedTarArchive) -> None:
        super().__init__()
        self._archive = archive
        self._position = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if type(offset) is not int or type(whence) is not int:
            raise TypeError("Tar archive seek coordinates must be integers.")
        if whence == io.SEEK_SET:
            position = offset
        elif whence == io.SEEK_CUR:
            position = self._position + offset
        elif whence == io.SEEK_END:
            position = self._archive.archive_bytes + offset
        else:
            raise ValueError("Invalid tar archive seek mode.")
        if position < 0:
            raise ValueError("Cannot seek before the start of a tar archive.")
        self._position = position
        return position

    def read(self, size: int | None = -1) -> bytes:
        if self.closed:
            raise ValueError("I/O operation on closed tar archive reader.")
        if size is None or size < 0:
            size = max(0, self._archive.archive_bytes - self._position)
        if type(size) is not int:
            raise TypeError("Tar archive read size must be an integer or None.")
        if size == 0 or self._position >= self._archive.archive_bytes:
            return b""
        size = min(size, self._archive.archive_bytes - self._position)
        content = self._archive._read_at(self._position, size)
        self._position += len(content)
        return content

    def readinto(self, buffer) -> int:
        content = self.read(len(buffer))
        buffer[: len(content)] = content
        return len(content)


class _SealedTarArchive:
    """One private immutable spool with independent concurrent readers."""

    def __init__(
        self,
        file: BinaryIO,
        *,
        archive_bytes: int,
        logical_bytes: int,
        capacity_lease: _CapacityLease,
    ) -> None:
        self._file = file
        self.archive_bytes = archive_bytes
        self.logical_bytes = logical_bytes
        self._capacity_lease = capacity_lease
        self._io_lock = threading.Lock()
        self._lifecycle_lock = threading.Lock()
        self._closed = False

    def open_reader(self) -> BinaryIO:
        with self._lifecycle_lock:
            if self._closed:
                raise RuntimeError("SyncBinding sealed tar archive is closed.")
            return io.BufferedReader(_ArchiveReader(self), buffer_size=1 << 16)

    def _read_at(self, offset: int, size: int) -> bytes:
        with self._io_lock:
            self._file.seek(offset)
            content = self._file.read(size)
        if type(content) is not bytes:
            raise TypeError("SyncBinding private archive returned non-bytes content.")
        return content

    def release_transient_reservation(self) -> None:
        """Retain only the immutable archive after its builder has settled."""

        with self._lifecycle_lock:
            if self._closed:
                return
            self._capacity_lease.shrink(self.archive_bytes)

    def close(self) -> None:
        with self._lifecycle_lock:
            if self._closed:
                return
            self._closed = True
        try:
            self._file.close()
        finally:
            self._capacity_lease.release()
            self._capacity_lease.record_archive_cleanup()


class _ArchiveBuildRetry(RuntimeError):
    pass


@dataclass(slots=True)
class _ArchiveEntry:
    future: concurrent.futures.Future[_SealedTarArchive]
    references: int


class _ArchiveReference:
    def __init__(
        self,
        capacity: SyncBindingStagingCapacity,
        key: Hashable,
        entry: _ArchiveEntry,
        archive: _SealedTarArchive,
        *,
        is_builder: bool,
    ) -> None:
        self._capacity = capacity
        self._key = key
        self._entry = entry
        self.archive = archive
        self.is_builder = is_builder
        self._released = False

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        self._capacity._release_archive_reference(
            self._key,
            self._entry,
            release_builder_reservation=self.is_builder,
        )


class SyncBindingStagingCapacity:
    """Fair process-local governor for aggregate SyncBinding staging resources.

    Transfer slots and byte reservations are separately queued. Existing admitted
    transfers receive byte reservations before new transfers so a full slot queue
    cannot deadlock the work that must release those slots. Each queue remains FIFO.
    """

    def __init__(
        self,
        *,
        max_concurrency: int = DEFAULT_SYNC_STAGING_MAX_CONCURRENCY,
        max_staged_bytes: int = DEFAULT_SYNC_STAGING_MAX_BYTES,
        reuse_sealed_archives: bool = True,
    ) -> None:
        if type(max_concurrency) is not int:
            raise TypeError("SyncBinding staging max_concurrency must be an integer.")
        if max_concurrency <= 0:
            raise ValueError("SyncBinding staging max_concurrency must be greater than zero.")
        if type(max_staged_bytes) is not int:
            raise TypeError("SyncBinding staging max_staged_bytes must be an integer.")
        if max_staged_bytes <= 0:
            raise ValueError("SyncBinding staging max_staged_bytes must be greater than zero.")
        if type(reuse_sealed_archives) is not bool:
            raise TypeError("SyncBinding staging reuse_sealed_archives must be a bool.")
        self.max_concurrency = max_concurrency
        self.max_staged_bytes = max_staged_bytes
        self.reuse_sealed_archives = reuse_sealed_archives
        self._lock = threading.Lock()
        self._transfer_waiters: deque[_CapacityWaiter] = deque()
        self._byte_waiters: deque[_CapacityWaiter] = deque()
        self._active_transfers = 0
        self._staged_bytes = 0
        self._peak_active_transfers = 0
        self._peak_staged_bytes = 0
        self._total_transfer_admissions = 0
        self._total_archive_builds = 0
        self._total_archive_reuses = 0
        self._total_archive_cleanups = 0
        self._total_transfer_wait_seconds = 0.0
        self._total_byte_wait_seconds = 0.0
        self._archives: dict[Hashable, _ArchiveEntry] = {}

    async def _acquire_transfer(self) -> _CapacityLease:
        return await self._acquire("transfer", 1)

    async def _reserve_bytes(self, amount: int) -> _CapacityLease:
        if type(amount) is not int:
            raise TypeError("SyncBinding staged byte reservation must be an integer.")
        if amount < 0:
            raise ValueError("SyncBinding staged byte reservation must be non-negative.")
        if amount > self.max_staged_bytes:
            raise SyncBindingStagingCapacityError(
                "SyncBinding transfer requires more staged bytes than its capacity domain: "
                f"required={amount}, capacity={self.max_staged_bytes}."
            )
        if amount == 0:
            return _CapacityLease(self, kind="bytes", amount=0)
        return await self._acquire("bytes", amount)

    async def _acquire(
        self,
        kind: Literal["transfer", "bytes"],
        amount: int,
    ) -> _CapacityLease:
        loop = asyncio.get_running_loop()
        waiter = _CapacityWaiter(
            kind=kind,
            amount=amount,
            loop=loop,
            future=loop.create_future(),
            enqueued_at=time.monotonic(),
        )
        with self._lock:
            queue = self._transfer_waiters if kind == "transfer" else self._byte_waiters
            queue.append(waiter)
            self._grant_waiters_locked()
        try:
            await waiter.future
        except BaseException:
            self._cancel_waiter(waiter)
            raise
        with self._lock:
            if not waiter.granted:
                raise AssertionError("SyncBinding capacity waiter resumed without a grant.")
            waiter.handed_off = True
        return _CapacityLease(self, kind=kind, amount=amount)

    def _cancel_waiter(self, waiter: _CapacityWaiter) -> None:
        with self._lock:
            if waiter.handed_off:
                return
            self._record_wait_locked(waiter)
            if waiter.granted:
                self._release_grant_locked(waiter.kind, waiter.amount)
                waiter.granted = False
            else:
                queue = self._transfer_waiters if waiter.kind == "transfer" else self._byte_waiters
                with contextlib.suppress(ValueError):
                    queue.remove(waiter)
            self._grant_waiters_locked()

    def _release(self, kind: Literal["transfer", "bytes"], amount: int) -> None:
        with self._lock:
            self._release_grant_locked(kind, amount)
            self._grant_waiters_locked()

    def _release_grant_locked(
        self,
        kind: Literal["transfer", "bytes"],
        amount: int,
    ) -> None:
        if kind == "transfer":
            self._active_transfers -= amount
            if self._active_transfers < 0:
                raise AssertionError("SyncBinding transfer capacity accounting underflowed.")
        else:
            self._staged_bytes -= amount
            if self._staged_bytes < 0:
                raise AssertionError("SyncBinding staged byte accounting underflowed.")

    def _grant_waiters_locked(self) -> None:
        while self._byte_waiters:
            waiter = self._byte_waiters[0]
            if self._staged_bytes + waiter.amount > self.max_staged_bytes:
                break
            self._byte_waiters.popleft()
            self._staged_bytes += waiter.amount
            self._peak_staged_bytes = max(self._peak_staged_bytes, self._staged_bytes)
            self._grant_waiter_locked(waiter)
        while self._transfer_waiters and self._active_transfers < self.max_concurrency:
            waiter = self._transfer_waiters.popleft()
            self._active_transfers += 1
            self._total_transfer_admissions += 1
            self._peak_active_transfers = max(self._peak_active_transfers, self._active_transfers)
            self._grant_waiter_locked(waiter)

    def _grant_waiter_locked(self, waiter: _CapacityWaiter) -> None:
        self._record_wait_locked(waiter)
        waiter.granted = True
        try:
            waiter.loop.call_soon_threadsafe(self._deliver_grant, waiter)
        except RuntimeError:
            waiter.granted = False
            self._release_grant_locked(waiter.kind, waiter.amount)

    def _record_wait_locked(self, waiter: _CapacityWaiter) -> None:
        if waiter.wait_recorded:
            return
        waiter.wait_recorded = True
        waited_seconds = max(0.0, time.monotonic() - waiter.enqueued_at)
        if waiter.kind == "transfer":
            self._total_transfer_wait_seconds += waited_seconds
        else:
            self._total_byte_wait_seconds += waited_seconds

    def _deliver_grant(self, waiter: _CapacityWaiter) -> None:
        if waiter.future.cancelled():
            self._cancel_waiter(waiter)
            return
        if not waiter.future.done():
            waiter.future.set_result(None)

    async def _acquire_archive(
        self,
        key: Hashable,
        builder: Callable[[], Awaitable[_SealedTarArchive]],
    ) -> _ArchiveReference:
        if not self.reuse_sealed_archives:
            archive = await builder()
            with self._lock:
                self._total_archive_builds += 1
            future: concurrent.futures.Future[_SealedTarArchive] = concurrent.futures.Future()
            future.set_result(archive)
            return _ArchiveReference(
                self,
                key,
                _ArchiveEntry(future, 1),
                archive,
                is_builder=True,
            )
        while True:
            with self._lock:
                entry = self._archives.get(key)
                creator = entry is None
                if entry is None:
                    entry = _ArchiveEntry(concurrent.futures.Future(), 1)
                    self._archives[key] = entry
                    self._total_archive_builds += 1
                else:
                    entry.references += 1
            if creator:
                try:
                    archive = await builder()
                except BaseException:
                    with self._lock:
                        if self._archives.get(key) is entry:
                            del self._archives[key]
                        entry.references -= 1
                        if not entry.future.done():
                            entry.future.set_exception(_ArchiveBuildRetry())
                    raise
                entry.future.set_result(archive)
                return _ArchiveReference(self, key, entry, archive, is_builder=True)
            try:
                # One cancelled follower must not cancel the process-wide single-flight
                # future and invalidate the creator or unrelated followers.
                archive = await asyncio.shield(asyncio.wrap_future(entry.future))
            except _ArchiveBuildRetry:
                with self._lock:
                    entry.references -= 1
                continue
            except BaseException:
                with self._lock:
                    entry.references -= 1
                raise
            with self._lock:
                self._total_archive_reuses += 1
            return _ArchiveReference(self, key, entry, archive, is_builder=False)

    def _release_archive_reference(
        self,
        key: Hashable,
        entry: _ArchiveEntry,
        *,
        release_builder_reservation: bool,
    ) -> None:
        archive: _SealedTarArchive | None = None
        close_archive = False
        with self._lock:
            entry.references -= 1
            if entry.references < 0:
                raise AssertionError("SyncBinding archive reference accounting underflowed.")
            if entry.references == 0:
                close_archive = True
                if self._archives.get(key) is entry:
                    del self._archives[key]
            if (entry.references == 0 or release_builder_reservation) and entry.future.done():
                try:
                    archive = entry.future.result()
                except BaseException:
                    archive = None
        reservation_error: BaseException | None = None
        if release_builder_reservation and archive is not None:
            try:
                archive.release_transient_reservation()
            except BaseException as error:
                reservation_error = error
        if archive is not None and close_archive:
            try:
                archive.close()
            except BaseException as close_error:
                if reservation_error is not None:
                    raise BaseExceptionGroup(
                        "SyncBinding archive reservation release and cleanup both failed.",
                        [reservation_error, close_error],
                    ) from close_error
                raise
        if reservation_error is not None:
            raise reservation_error

    def snapshot(self) -> SyncBindingStagingSnapshot:
        with self._lock:
            now = time.monotonic()
            return SyncBindingStagingSnapshot(
                max_concurrency=self.max_concurrency,
                max_staged_bytes=self.max_staged_bytes,
                active_transfers=self._active_transfers,
                staged_bytes=self._staged_bytes,
                waiting_transfers=len(self._transfer_waiters),
                waiting_bytes=sum(waiter.amount for waiter in self._byte_waiters),
                shared_archives=len(self._archives),
                archive_references=sum(entry.references for entry in self._archives.values()),
                peak_active_transfers=self._peak_active_transfers,
                peak_staged_bytes=self._peak_staged_bytes,
                total_transfer_admissions=self._total_transfer_admissions,
                total_archive_builds=self._total_archive_builds,
                total_archive_reuses=self._total_archive_reuses,
                total_archive_cleanups=self._total_archive_cleanups,
                total_transfer_wait_seconds=self._total_transfer_wait_seconds,
                total_byte_wait_seconds=self._total_byte_wait_seconds,
                oldest_transfer_wait_seconds=(
                    0.0
                    if not self._transfer_waiters
                    else max(0.0, now - self._transfer_waiters[0].enqueued_at)
                ),
                oldest_byte_wait_seconds=(
                    0.0
                    if not self._byte_waiters
                    else max(0.0, now - self._byte_waiters[0].enqueued_at)
                ),
            )

    def _record_archive_build(self) -> None:
        with self._lock:
            self._total_archive_builds += 1

    def _record_archive_cleanup(self) -> None:
        with self._lock:
            self._total_archive_cleanups += 1


DEFAULT_SYNC_BINDING_STAGING_CAPACITY = SyncBindingStagingCapacity()

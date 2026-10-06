"""SQLite connection setup, diagnostic access and off-thread ownership."""

from __future__ import annotations

import asyncio
import contextvars
import os
import sqlite3
from collections.abc import Callable
from concurrent.futures import Executor
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar, cast
from urllib.parse import quote

from cayu.storage._diagnostic_inspection import (
    DiagnosticStoreInspectionChanged,
    current_diagnostic_store_inspection,
)

_T = TypeVar("_T")


async def _run_off_thread_with_connection_ownership(
    lock: asyncio.Lock,
    connection: sqlite3.Connection,
    operation: Callable[[sqlite3.Connection], _T],
    *,
    executor: Executor | None = None,
    worker_started: asyncio.Event | None = None,
    interrupt_on_cancellation: bool = False,
) -> _T:
    """Keep a SQLite connection owned until its off-thread operation terminates.

    Cancelling an ``asyncio.to_thread`` await does not stop the worker thread.
    For an interruptible read, request ``sqlite3_interrupt()`` after cancellation;
    in every case defer the signal while holding the connection lock so no
    subsequent operation or shutdown can reuse the connection before the worker
    has left it in a terminal transaction state.
    """

    if type(interrupt_on_cancellation) is not bool:
        raise TypeError("interrupt_on_cancellation must be a bool.")

    async with lock:

        def capture_outcome() -> tuple[bool, object]:
            try:
                return True, operation(connection)
            except BaseException as worker_failure:
                # The executor future must complete normally even when the
                # operation raises CancelledError. That makes every cancellation
                # from shield() unambiguously caller-owned and keeps ownership
                # tied to the executor's physical completion.
                return False, worker_failure

        loop = asyncio.get_running_loop()
        context = contextvars.copy_context()
        worker = loop.run_in_executor(executor, context.run, capture_outcome)
        if worker_started is not None:
            worker_started.set()
        cancellation: asyncio.CancelledError | None = None

        while not worker.done():
            try:
                await asyncio.shield(worker)
            except asyncio.CancelledError as exc:
                if cancellation is None:
                    cancellation = exc
                    if interrupt_on_cancellation:
                        connection.interrupt()
            except BaseException:
                if worker.done():
                    break
                raise

        succeeded, outcome = worker.result()
        if not succeeded:
            if not isinstance(outcome, BaseException):
                raise RuntimeError("SQLite worker returned an invalid failure outcome.")
            if cancellation is None:
                raise outcome
            cancellation.add_note(
                "SQLite worker failed while caller cancellation was pending: "
                f"{type(outcome).__name__}: {outcome}"
            )
            raise cancellation from outcome
        if cancellation is not None:
            raise cancellation
        return cast("_T", outcome)


def connect(
    path: Path,
    *,
    read_only: bool = False,
    immutable: bool = False,
) -> sqlite3.Connection:
    from cayu.storage._phase_timing import timed_sqlite_connection
    from cayu.storage._sqlite_functions import _register_sqlite_functions

    if type(read_only) is not bool:
        raise TypeError("read_only must be a bool.")
    if type(immutable) is not bool:
        raise TypeError("immutable must be a bool.")
    if immutable and not read_only:
        raise ValueError("Immutable SQLite connections must be read-only.")
    if (
        current_diagnostic_store_inspection() is not None
        and not read_only
        and str(path) != ":memory:"
    ):
        return connect_read_only_inspection(path)
    if str(path) == ":memory:":
        if read_only:
            raise ValueError("Read-only connections require a file-backed SQLite database.")
    elif not read_only:
        path.parent.mkdir(parents=True, exist_ok=True)
    if read_only:
        # A dedicated read-only connection lets queries run in worker threads
        # without contending with the writer connection's transactions (WAL
        # readers never block on the writer). query_only guards against any
        # accidental write slipping onto the read path. Immutable inspection is
        # reserved for closed, probe-free diagnostic use that must create no WAL
        # sidecars; it is not the live concurrent-reader mode.
        immutable_query = "&immutable=1" if immutable else ""
        uri = f"file:{quote(str(path.resolve()), safe='/')}?mode=ro{immutable_query}"
        connection = sqlite3.connect(uri, uri=True, check_same_thread=False)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 5000")
        connection.execute("PRAGMA query_only = ON")
        _register_sqlite_functions(connection)
        return timed_sqlite_connection(connection)
    connection = sqlite3.connect(path, check_same_thread=False)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA busy_timeout = 5000")
    if str(path) != ":memory:":
        connection.execute("PRAGMA journal_mode = WAL")
    _register_sqlite_functions(connection)
    return timed_sqlite_connection(connection)


def _is_in_memory(connection: sqlite3.Connection) -> bool:
    return any(
        row[1] == "main" and row[2] == ""
        for row in connection.execute("PRAGMA database_list").fetchall()
    )


@dataclass(frozen=True)
class _SQLiteFileIdentity:
    device: int
    inode: int
    size: int
    modified_ns: int


def _sqlite_file_identity(path: Path) -> _SQLiteFileIdentity | None:
    try:
        stat_result = path.stat()
    except FileNotFoundError:
        return None
    return _SQLiteFileIdentity(
        device=stat_result.st_dev,
        inode=stat_result.st_ino,
        size=stat_result.st_size,
        modified_ns=stat_result.st_mtime_ns,
    )


def diagnostic_sqlite_source_missing(path: Path) -> bool:
    """Register a diagnostic absence guard for a missing file-backed database."""

    inspection = current_diagnostic_store_inspection()
    if inspection is None or str(path) == ":memory:":
        return False
    resolved = path.resolve()
    observed_paths = (resolved, Path(f"{resolved}-wal"), Path(f"{resolved}-shm"))
    before = tuple(_sqlite_file_identity(item) for item in observed_paths)
    if any(identity is not None for identity in before):
        return False

    def verify_source_remained_missing() -> None:
        after = tuple(_sqlite_file_identity(item) for item in observed_paths)
        if after != before:
            raise DiagnosticStoreInspectionChanged(
                "A missing SQLite diagnostic source appeared during collection."
            )

    inspection.add_verifier(verify_source_remained_missing)
    return True


def connect_read_only_inspection(path: Path) -> sqlite3.Connection:
    """Open a non-mutating SQLite diagnostic view without ignoring live WAL data."""

    if str(path) == ":memory:":
        raise ValueError("Diagnostic inspection requires a file-backed SQLite database.")
    resolved = path.resolve()
    wal_path = Path(f"{resolved}-wal")
    shm_path = Path(f"{resolved}-shm")
    before = tuple(_sqlite_file_identity(item) for item in (resolved, wal_path, shm_path))
    if before[0] is None:
        raise FileNotFoundError(os.fspath(resolved))

    # A live WAL database must use SQLite's locking-aware read-only mode so the
    # committed WAL frames remain visible. A static database can use immutable
    # mode, which guarantees inspection creates no journal sidecars.
    wal_exists = before[1] is not None
    shm_exists = before[2] is not None
    if wal_exists != shm_exists:
        raise sqlite3.OperationalError("SQLite WAL inspection sidecars are incomplete.")
    live_wal = wal_exists and shm_exists
    inspection = current_diagnostic_store_inspection()
    connection = connect(resolved, read_only=True, immutable=not live_wal)
    if inspection is not None and not live_wal:

        def verify_static_snapshot() -> None:
            after = tuple(_sqlite_file_identity(item) for item in (resolved, wal_path, shm_path))
            if after != before:
                raise DiagnosticStoreInspectionChanged(
                    "A static SQLite diagnostic snapshot changed during collection."
                )

        inspection.add_verifier(verify_static_snapshot)
    return connection

"""SQLite collaboration identity store."""

from __future__ import annotations

import asyncio
import sqlite3
from collections.abc import Callable
from contextlib import asynccontextmanager
from functools import partial
from pathlib import Path
from typing import TypeVar

from cayu.collaboration._ownership import _MutationOwners
from cayu.collaboration.base import CollaborationStore
from cayu.collaboration.participants import CollaborationUnavailable
from cayu.storage import _sqlite_support as sqlite
from cayu.storage._collaboration_repository import _SQLRepository
from cayu.storage.migrations import SchemaMode
from cayu.storage.sqlite import _run_off_thread_with_connection_ownership

T = TypeVar("T")


class _SQLiteRepository(_SQLRepository):
    def __init__(self, connection: sqlite3.Connection, scope: str, io_lock: asyncio.Lock) -> None:
        super().__init__(connection, scope, postgres=False)
        self._io_lock = io_lock

    async def _execute(self, sql: str, args: tuple = ()) -> sqlite3.Cursor:
        return await _run_off_thread_with_connection_ownership(
            self._io_lock, self.connection, lambda connection: connection.execute(sql, args)
        )

    async def _rows(self, cursor: sqlite3.Cursor):
        return await _run_off_thread_with_connection_ownership(
            self._io_lock, self.connection, lambda connection: cursor.fetchall()
        )


class SQLiteCollaborationStore(CollaborationStore):
    request_contract_version = 1

    def __init__(self, path: str | Path, *, schema_mode: SchemaMode = SchemaMode.CREATE) -> None:
        self._lock = asyncio.Lock()
        self._io_lock = asyncio.Lock()
        self._owners = _MutationOwners()
        self._close_task: asyncio.Task[None] | None = None
        self._connection = sqlite.connect(Path(path))
        try:
            sqlite.reconcile_schema(self._connection, schema_mode, app_min_supported=95)
        except BaseException:
            self._connection.close()
            raise

    @asynccontextmanager
    async def _transaction(self, scope: str, *, write: bool):
        if self._owners.closed and asyncio.current_task() not in self._owners.pending:
            raise CollaborationUnavailable("Collaboration store is closing.")
        async with self._lock:
            if self._owners.closed and asyncio.current_task() not in self._owners.pending:
                raise CollaborationUnavailable("Collaboration store is closing.")
            transaction = sqlite._transaction(self._connection, begin_immediate=write)

            # Keep transaction ownership through physical settlement, including
            # cancellation while BEGIN or finalization is running off-thread.
            async def invoke(callback: Callable[[], T]) -> T:
                return await _run_off_thread_with_connection_ownership(
                    self._io_lock, self._connection, lambda connection: callback()
                )

            try:
                await invoke(transaction.__enter__)
                yield _SQLiteRepository(self._connection, scope, self._io_lock)
            except BaseException as error:
                if not await invoke(
                    partial(transaction.__exit__, type(error), error, error.__traceback__)
                ):
                    raise
            else:
                await invoke(partial(transaction.__exit__, None, None, None))

    async def close(self) -> None:
        self._owners.closed = True
        task = self._close_task
        if task is None or (task.done() and (task.cancelled() or task.exception() is not None)):
            coroutine = self._close_owned()
            try:
                task = asyncio.create_task(coroutine, name="cayu-collaboration-close")
            except BaseException:
                coroutine.close()
                raise
            self._close_task = task
            task.add_done_callback(self._observe_close)
        try:
            done, _ = await asyncio.wait((task,), timeout=self._owners.observation_timeout)
        except asyncio.CancelledError as cancellation:
            if task.done() and not task.cancelled():
                failure = task.exception()
                if failure is not None:
                    raise cancellation from failure
            raise
        if not done:
            raise CollaborationUnavailable("Collaboration shutdown is still settling.")
        if task.cancelled():
            raise CollaborationUnavailable("Collaboration shutdown requires retry.")
        task.result()

    @staticmethod
    def _observe_close(task: asyncio.Task[None]) -> None:
        if not task.cancelled():
            # Retain the result for the next observer without an unhandled-task
            # diagnostic when the original observer has already left.
            task.exception()

    async def _close_owned(self) -> None:
        if self._owners.pending:
            await asyncio.wait(tuple(self._owners.pending))
        async with self._lock:
            await _run_off_thread_with_connection_ownership(
                self._io_lock, self._connection, lambda connection: connection.close()
            )

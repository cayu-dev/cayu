"""Queue a task in the same transaction as the application's own rows.

An application whose tables share a database with Cayu's task store can't make
``create_task`` join its transaction: the store commits on its own connection. So a
crash between the application's commit and ``create_task`` loses the task, or a task
created first outlives an application write that rolled back. The outbox closes that
gap. The application inserts the task request into an outbox table of its own, in the
transaction that writes its rows, and ``drain_task_outbox`` creates each queued task
and then removes its row.

The parts stand alone:

- ``task_outbox_row(request)`` is the row to insert: ``{"id": ..., "request_json": ...}``.
  ``id`` is the task's id (``request.task_id``, or a new one), so the task exists
  exactly when the application's commit does, and a drain that runs twice creates it
  once.
- ``TASK_OUTBOX_COLUMNS`` says what the table holds, for the application's own
  migration; ``task_outbox_ddl`` writes a ``CREATE TABLE`` for PostgreSQL or SQLite.
- ``TaskOutbox`` reads and removes rows. ``PostgresTaskOutbox`` and ``SQLiteTaskOutbox``
  read a table by name, such as ``app.task_outbox``.
- ``drain_task_outbox(task_store, outbox)`` creates the queued tasks, oldest first.
  Call it from a worker loop or a schedule; ``run_task_outbox`` loops it.

A row is removed only after its task exists, or once the store refused its request, so a
drain that stops between the two leaves the row, and the next drain finds the task and removes the row. A task that was
created, finished and deleted before its row was removed would be created again; drain
soon after a restart, as ``run_task_outbox`` does.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import re
import sqlite3
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from uuid import uuid4

from cayu.tasks.creation import TaskCreate, copy_task_create
from cayu.tasks.store import TaskStore

#: The outbox table's columns: the task id, the request as JSON text and when it was
#: queued. PostgreSQL also keeps a ``position`` sequence, the drain's order, so rows of one
#: transaction drain in the order they were written; SQLite drains in ``rowid`` order.
TASK_OUTBOX_COLUMNS = ("id", "request_json", "created_at")
#: The most rows one drain reads.
MAX_TASK_OUTBOX_DRAIN = 1000
_DRAIN_BATCH = 100
_logger = logging.getLogger(__name__)
_TABLE_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,62}(\.[A-Za-z_][A-Za-z0-9_]{0,62})?")


def _table(name: str) -> str:
    if type(name) is not str or not _TABLE_NAME.fullmatch(name):
        raise ValueError("The outbox table must be a name such as task_outbox or app.task_outbox.")
    return ".".join(f'"{part}"' for part in name.split("."))


def task_outbox_ddl(table: str, *, dialect: Literal["postgresql", "sqlite"]) -> str:
    """A ``CREATE TABLE IF NOT EXISTS`` for the outbox, for a migration to run."""

    quoted = _table(table)
    if dialect == "postgresql":
        return (
            f"CREATE TABLE IF NOT EXISTS {quoted} (position BIGSERIAL, id TEXT PRIMARY KEY, "
            "request_json TEXT NOT NULL, created_at TIMESTAMPTZ NOT NULL DEFAULT now())"
        )
    if dialect == "sqlite":
        return (
            f"CREATE TABLE IF NOT EXISTS {quoted} (id TEXT PRIMARY KEY, "
            "request_json TEXT NOT NULL, created_at TEXT NOT NULL DEFAULT "
            "(strftime('%Y-%m-%dT%H:%M:%f', 'now')))"
        )
    raise ValueError("dialect must be 'postgresql' or 'sqlite'.")


def task_outbox_row(request: TaskCreate) -> dict[str, str]:
    """The outbox row for ``request``: insert it with the application's own rows."""

    request = copy_task_create(request)
    if (
        request._runtime_invocation_source is not None
        or request._verified_invocation_origin is not None
        or request._runtime_session_binding is not None
    ):
        # Runtime provenance isn't part of the request's JSON and would be lost.
        raise ValueError("An outbox row can't carry runtime invocation provenance.")
    if request.task_id is None:
        request = request.model_copy(update={"task_id": f"task_{uuid4().hex}"})
    assert request.task_id is not None
    return {"id": request.task_id, "request_json": request.model_dump_json()}


@dataclass(frozen=True)
class TaskOutboxEntry:
    """One queued row: its task id and the request as JSON text."""

    id: str
    request_json: str


class TaskOutbox(ABC):
    """Reads and removes outbox rows; the application inserts them."""

    @abstractmethod
    async def read(self, limit: int) -> list[TaskOutboxEntry]:
        """Up to ``limit`` rows, oldest first."""

    @abstractmethod
    async def remove(self, ids: Sequence[str]) -> None:
        """Remove these rows; ids already gone are ignored."""


def _request(entry: TaskOutboxEntry) -> TaskCreate:
    request = TaskCreate.model_validate_json(entry.request_json)
    if request.task_id != entry.id:
        raise ValueError(f"Outbox row {entry.id} queues a task with another id.")
    return request


class PostgresTaskOutbox(TaskOutbox):
    """An outbox table in PostgreSQL, read with its own short connections."""

    def __init__(self, dsn: str, *, table: str = "task_outbox") -> None:
        _table(table)
        self._dsn = dsn
        self._parts = tuple(table.split("."))

    async def read(self, limit: int) -> list[TaskOutboxEntry]:
        import psycopg
        from psycopg import sql

        query = sql.SQL("SELECT id, request_json FROM {} ORDER BY position LIMIT %s").format(
            sql.Identifier(*self._parts)
        )
        async with await psycopg.AsyncConnection.connect(self._dsn) as conn:
            cursor = await conn.execute(query, (limit,))
            rows = await cursor.fetchall()
        return [TaskOutboxEntry(id=row[0], request_json=row[1]) for row in rows]

    async def remove(self, ids: Sequence[str]) -> None:
        if not ids:
            return
        import psycopg
        from psycopg import sql

        query = sql.SQL("DELETE FROM {} WHERE id = ANY(%s)").format(sql.Identifier(*self._parts))
        async with await psycopg.AsyncConnection.connect(self._dsn) as conn:
            await conn.execute(query, (list(ids),))


class SQLiteTaskOutbox(TaskOutbox):
    """An outbox table in a SQLite file, read with its own short connections."""

    def __init__(self, path: str | Path, *, table: str = "task_outbox") -> None:
        self._path = Path(path)
        self._table = _table(table)

    def _run(self, sql: str, params: Sequence[object]) -> list[tuple[str, str]]:
        connection = sqlite3.connect(self._path, timeout=30)
        try:
            with connection:
                return connection.execute(sql, params).fetchall()
        finally:
            connection.close()

    async def read(self, limit: int) -> list[TaskOutboxEntry]:
        rows = await asyncio.to_thread(
            self._run,
            f"SELECT id, request_json FROM {self._table} ORDER BY rowid LIMIT ?",
            (limit,),
        )
        return [TaskOutboxEntry(id=row[0], request_json=row[1]) for row in rows]

    async def remove(self, ids: Sequence[str]) -> None:
        if not ids:
            return
        marks = ", ".join("?" * len(ids))
        await asyncio.to_thread(
            self._run, f"DELETE FROM {self._table} WHERE id IN ({marks})", tuple(ids)
        )


@dataclass(frozen=True)
class TaskOutboxRefusal:
    """A row whose request the task store refused, removed so it can't block the rest."""

    id: str
    reason: str


@dataclass(frozen=True)
class TaskOutboxDrain:
    """What one drain did: tasks it created, rows whose task already existed, and rows
    whose request was refused."""

    created: tuple[str, ...] = ()
    existing: tuple[str, ...] = ()
    refused: tuple[TaskOutboxRefusal, ...] = ()

    @property
    def rows(self) -> int:
        return len(self.created) + len(self.existing) + len(self.refused)


async def drain_task_outbox(
    task_store: TaskStore, outbox: TaskOutbox, *, limit: int = _DRAIN_BATCH
) -> TaskOutboxDrain:
    """Create the task of each queued row in the order written, then remove the rows.

    A request the store refuses (it doesn't parse, its parent is missing, its retry
    key is spent) is logged, reported in ``refused`` and removed, so it can't hold up
    the rows after it. A failure of the store or the outbox itself raises; the rows
    handled before it are removed, and the rest stay for the next drain.
    """

    if type(limit) is not int or not 1 <= limit <= MAX_TASK_OUTBOX_DRAIN:
        raise ValueError(f"limit must be 1 to {MAX_TASK_OUTBOX_DRAIN}.")
    created: list[str] = []
    existing: list[str] = []
    refused: list[TaskOutboxRefusal] = []
    try:
        for entry in await outbox.read(limit):
            if await task_store.load_task(entry.id) is not None:
                existing.append(entry.id)
                continue
            try:
                await task_store.create_task(_request(entry))
            except ValueError as exc:
                # Another drain may have created it first.
                if await task_store.load_task(entry.id) is not None:
                    existing.append(entry.id)
                    continue
                _logger.warning("The task store refused outbox row %s: %s", entry.id, exc)
                refused.append(TaskOutboxRefusal(id=entry.id, reason=str(exc)))
                continue
            created.append(entry.id)
    finally:
        await outbox.remove([*created, *existing, *(item.id for item in refused)])
    return TaskOutboxDrain(created=tuple(created), existing=tuple(existing), refused=tuple(refused))


async def run_task_outbox(
    task_store: TaskStore,
    outbox: TaskOutbox,
    *,
    interval_s: float = 1.0,
    stop: asyncio.Event | None = None,
) -> None:
    """Drain the outbox now and then every ``interval_s`` seconds until ``stop`` is set.

    A full drain runs again at once. A failed drain is retried after the interval.
    """

    if interval_s <= 0:
        raise ValueError("interval_s must be positive.")
    stop = stop or asyncio.Event()
    while not stop.is_set():
        try:
            drained = await drain_task_outbox(task_store, outbox, limit=_DRAIN_BATCH)
        except Exception:
            _logger.exception("Draining the task outbox failed.")
            drained = TaskOutboxDrain()
        if drained.rows >= _DRAIN_BATCH:
            continue
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(stop.wait(), interval_s)


__all__ = [
    "MAX_TASK_OUTBOX_DRAIN",
    "TASK_OUTBOX_COLUMNS",
    "PostgresTaskOutbox",
    "SQLiteTaskOutbox",
    "TaskOutbox",
    "TaskOutboxDrain",
    "TaskOutboxEntry",
    "TaskOutboxRefusal",
    "drain_task_outbox",
    "run_task_outbox",
    "task_outbox_ddl",
    "task_outbox_row",
]

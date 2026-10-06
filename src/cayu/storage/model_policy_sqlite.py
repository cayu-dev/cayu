"""SQLite policy publication through owned off-thread connection operations."""

from __future__ import annotations

import time
from pathlib import Path

from cayu.runtime._policy_storage import (
    POLICY_STORAGE_REVISION,
    ModelPolicyStore,
    PolicyStorageCommand,
    PolicyStorageView,
    transition,
)
from cayu.runtime._policy_wire import require
from cayu.storage import _sqlite_connection as sqlite_connection
from cayu.storage import _sqlite_support as support
from cayu.storage._phase_timing import TimedStoreLock
from cayu.storage._sqlite_connection import _run_off_thread_with_connection_ownership
from cayu.storage.migrations import SchemaMode
from cayu.storage.targets import require_sqlite_store_allowed


class SQLiteModelPolicyStore(ModelPolicyStore):
    def __init__(self, path: str | Path, *, schema_mode: SchemaMode = SchemaMode.CREATE):
        require_sqlite_store_allowed("SQLiteModelPolicyStore")
        self._lock = TimedStoreLock()
        self._closed = False
        self._connection = sqlite_connection.connect(Path(path))
        try:
            support.reconcile_schema(
                self._connection, schema_mode, app_min_supported=POLICY_STORAGE_REVISION
            )
        except BaseException:
            self._connection.close()
            raise

    async def execute(self, command: PolicyStorageCommand) -> PolicyStorageView:
        command.validate()
        if self._closed:
            raise RuntimeError("Model policy store is closed.")

        def operation(connection):
            if self._closed:
                raise RuntimeError("Model policy store is closed.")
            with connection:
                connection.execute("BEGIN IMMEDIATE")
                row = connection.execute(
                    "SELECT binding, owner, expires, revision, state "
                    "FROM cayu_model_policy_state WHERE binding_id=?",
                    (command.key,),
                ).fetchone()
                now = connection.execute(
                    "SELECT (julianday('now') - 2440587.5) * 86400.0"
                ).fetchone()[0]
                updated, view = transition(None if row is None else tuple(row), command, now)
                connection.execute(
                    "INSERT INTO cayu_model_policy_state VALUES (?, ?, ?, ?, ?, ?) "
                    "ON CONFLICT(binding_id) DO UPDATE SET owner=excluded.owner, "
                    "expires=excluded.expires, revision=excluded.revision, state=excluded.state",
                    (command.key, *updated),
                )
                require(command.deadline_ns is None or time.monotonic_ns() < command.deadline_ns)
            return view

        return await _run_off_thread_with_connection_ownership(
            self._lock, self._connection, operation
        )

    async def close(self) -> None:
        if not self._closed:

            def close_connection(connection):
                connection.close()
                self._closed = True

            await _run_off_thread_with_connection_ownership(
                self._lock, self._connection, close_connection
            )

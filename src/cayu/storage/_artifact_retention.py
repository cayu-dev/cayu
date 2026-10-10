"""Fence artifact reference publication through each retention deletion.

The database locks are held only for one artifact. File databases are grouped
by physical identity, and Postgres stores by server/database/schema, so stores
sharing one database use one transaction (including single-connection pools).
Unknown reference stores fail closed instead of supplying an unguarded list.
"""

from __future__ import annotations

import sqlite3
import sys
from collections.abc import AsyncIterator, Iterable, Sequence
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cayu._task_wait import run_thread_to_completion
from cayu.artifacts.base import ArtifactMetadata
from cayu.storage._retention_sql import (
    SNAPSHOT_TABLES,
    PostgresRetentionSql,
    RetentionSql,
    json_reference_atoms,
    snapshot_pin_text,
)
from cayu.storage.retention import RetentionProtection

_TABLES = (
    "cayu_sessions",
    "cayu_transcript_messages",
    "cayu_session_operations",
    "cayu_knowledge_evidence",
    "cayu_eval_runs",
    "cayu_eval_results",
    "cayu_eval_result_records",
    "cayu_eval_run_trial_checkpoints",
    *SNAPSHOT_TABLES,
)


class _ThreadedSqlite(RetentionSql):
    def __init__(self, connection: sqlite3.Connection) -> None:
        self.connection = connection

    async def all(self, sql: str, parameters: Sequence[Any] = ()) -> list[tuple[Any, ...]]:
        def read() -> list[tuple[Any, ...]]:
            cursor = self.connection.execute(sql, tuple(parameters))
            try:
                return [tuple(row) for row in cursor.fetchall()]
            finally:
                cursor.close()

        return await run_thread_to_completion(read)

    async def run(self, sql: str, parameters: Sequence[Any] = ()) -> int:
        def execute() -> int:
            cursor = self.connection.execute(sql, tuple(parameters))
            try:
                return cursor.rowcount
            finally:
                cursor.close()

        return await run_thread_to_completion(execute)


@dataclass(frozen=True)
class _Database:
    key: tuple[str, ...]
    path: Path | None = None
    store: Any = None

    @asynccontextmanager
    async def transaction(self, *, apply: bool) -> AsyncIterator[RetentionSql]:
        if self.path is not None:
            # Never create a missing database. A dedicated connection avoids
            # taking store-local locks while waiting for another process.
            mode = "rw" if apply else "ro"
            connection = sqlite3.connect(
                f"{self.path.as_uri()}?mode={mode}",
                uri=True,
                check_same_thread=False,
                timeout=5,
            )
            sql = _ThreadedSqlite(connection)
            try:
                await sql.run("BEGIN IMMEDIATE" if apply else "BEGIN")
                yield sql
            finally:
                # Closing rolls back and releases the publication fence even
                # when cancellation arrives during a blocking SQLite call.
                await run_thread_to_completion(connection.close)
        else:
            async with self.store._connection() as connection:
                try:
                    async with connection.cursor() as cursor:
                        yield PostgresRetentionSql(cursor)
                finally:
                    await connection.rollback()


async def reference_databases(stores: Iterable[Any]) -> tuple[_Database, ...] | None:
    """Return guardable databases, or None if any source cannot be fenced."""

    from cayu.snapshots.base import SQLiteAgentSnapshotStore
    from cayu.storage.evals_sqlite import SQLiteEvalStore
    from cayu.storage.sqlite import SQLiteSessionStore

    # A Postgres instance necessarily loaded its defining module already.
    # Do not import the optional Postgres extra for SQLite-only applications.
    postgres_types = tuple(
        getattr(module, name)
        for module_name, name in (
            ("cayu.storage.postgres", "PostgresSessionStore"),
            ("cayu.storage.evals_postgres", "PostgresEvalStore"),
        )
        if (module := sys.modules.get(module_name)) is not None
    )
    databases: dict[tuple[str, ...], _Database] = {}
    for store in stores:
        if store is None:
            continue
        if isinstance(store, SQLiteSessionStore | SQLiteEvalStore | SQLiteAgentSnapshotStore):
            if str(store.path) == ":memory:" or getattr(store, "_read_only", False):
                return None
            path = Path(store.path).resolve()
            identity = path.stat()
            key = ("sqlite", str(identity.st_dev), str(identity.st_ino))
            databases.setdefault(key, _Database(key=key, path=path))
        elif isinstance(store, postgres_types):
            if getattr(store, "_read_only", False):
                return None
            await store._ensure_ready()
            async with store._connection() as connection:
                async with connection.cursor() as cursor:
                    await cursor.execute(
                        "SELECT current_database(), current_schema(), "
                        "inet_server_addr()::text, inet_server_port()"
                    )
                    row = await cursor.fetchone()
                    assert row is not None
                    key = (
                        "postgres",
                        str(row[2] or connection.info.host),
                        str(row[3] or connection.info.port),
                        str(row[0]),
                        str(row[1]),
                    )
                await connection.rollback()
            databases.setdefault(key, _Database(key=key, store=store))
        else:
            return None
    return tuple(databases[key] for key in sorted(databases))


async def _protections(
    sql: RetentionSql, present: set[str], metadata: ArtifactMetadata
) -> set[RetentionProtection]:
    reasons: set[RetentionProtection] = set()
    if (
        metadata.session_id is not None
        and "cayu_sessions" in present
        and await sql.exists("SELECT 1 FROM cayu_sessions WHERE id = ?", (metadata.session_id,))
    ):
        reasons.add(RetentionProtection.SESSION_REFERENCE)
    sources = (
        (
            "cayu_transcript_messages",
            "message" if sql.postgres else "message_json",
            RetentionProtection.SESSION_REFERENCE,
        ),
        (
            "cayu_session_operations",
            "record" if sql.postgres else "record_json",
            RetentionProtection.SESSION_REFERENCE,
        ),
        ("cayu_eval_runs", "invocation_json", RetentionProtection.EVAL_REFERENCE),
        ("cayu_eval_runs", "scenario_progress_json", RetentionProtection.EVAL_REFERENCE),
        (
            "cayu_eval_results",
            "result" if sql.postgres else "result_json",
            RetentionProtection.EVAL_REFERENCE,
        ),
        (
            "cayu_eval_result_records",
            "captured_result" if sql.postgres else "captured_result_json",
            RetentionProtection.EVAL_REFERENCE,
        ),
        ("cayu_eval_run_trial_checkpoints", "checkpoint_json", RetentionProtection.EVAL_REFERENCE),
    )
    for table, column, reason in sources:
        if table in present and metadata.id in await json_reference_atoms(
            sql, table, column, key_suffix="artifact_id"
        ):
            reasons.add(reason)
    if "cayu_knowledge_evidence" in present and await sql.exists(
        "SELECT 1 FROM cayu_knowledge_evidence WHERE source_type = 'artifact' "
        "AND disposition <> 'detached' AND source_id = ?",
        (metadata.id,),
    ):
        reasons.add(RetentionProtection.KNOWLEDGE_EVIDENCE)
    if set(SNAPSHOT_TABLES) <= present and metadata.id in await snapshot_pin_text(sql):
        reasons.add(RetentionProtection.SNAPSHOT_PIN)
    return reasons


@asynccontextmanager
async def artifact_reference_guard(
    databases: tuple[_Database, ...] | None,
    metadata: ArtifactMetadata,
    *,
    apply: bool,
) -> AsyncIterator[set[RetentionProtection]]:
    if not databases:
        yield {RetentionProtection.ERASURE_GUARD}
        return
    async with AsyncExitStack() as stack:
        views: list[tuple[RetentionSql, set[str]]] = []
        for database in databases:
            sql = await stack.enter_async_context(database.transaction(apply=apply))
            present = await sql.existing_tables(_TABLES)
            if apply and sql.postgres and present:
                # SHARE excludes INSERT/UPDATE/DELETE, including writers that
                # do not know about retention. Table names are a fixed allowlist.
                await sql.run("LOCK TABLE " + ", ".join(sorted(present)) + " IN SHARE MODE")
            views.append((sql, present))
        reasons: set[RetentionProtection] = set()
        for sql, present in views:
            reasons.update(await _protections(sql, present, metadata))
        yield reasons

"""Immutable business reservations. PostgreSQL production; SQLite local fixture.

No execution, approval, lease, task creation or mutable workflow state belongs here.
Schema creation is explicit. Callers must authenticate before tenant-qualified lookup.
"""

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from uuid import UUID

from tests.qualification.repository_maintenance_identity import (
    MaintenanceRunIdentity,
    MaintenanceRunIntent,
    MaintenanceTaskPhase,
    allocate_identity,
    copy_intent,
    task_id_for,
)

_RUNS = "maintenance_runs_v1"
_TASKS = "maintenance_phase_tasks_v1"
_COLUMNS = {
    _RUNS: (
        "public_id",
        "tenant",
        "subject",
        "idempotency_key",
        "product_run_id",
        "session_id",
        "workflow_session_id",
        "intent_fingerprint",
        "identity_json",
    ),
    _TASKS: ("task_id", "public_id", "phase"),
}
_DDL = {
    _RUNS: f"""CREATE TABLE {_RUNS} (
        public_id TEXT NOT NULL PRIMARY KEY,
        tenant TEXT NOT NULL,
        subject TEXT NOT NULL,
        idempotency_key TEXT NOT NULL,
        product_run_id TEXT NOT NULL UNIQUE,
        session_id TEXT NOT NULL UNIQUE,
        workflow_session_id TEXT NOT NULL UNIQUE,
        intent_fingerprint TEXT NOT NULL,
        identity_json TEXT NOT NULL,
        UNIQUE (tenant, idempotency_key)
    )""",
    _TASKS: f"""CREATE TABLE {_TASKS} (
        task_id TEXT NOT NULL PRIMARY KEY,
        public_id TEXT NOT NULL REFERENCES {_RUNS}(public_id),
        phase TEXT NOT NULL,
        UNIQUE (public_id, phase)
    )""",
}
_SELECT = ", ".join(
    "substr(identity_json, 1, 262145) AS identity_json" if name == "identity_json" else name
    for name in _COLUMNS[_RUNS]
)


class ReservationConflict(ValueError):
    def __init__(self):
        super().__init__("Idempotency identity already belongs to different accepted work.")


class ReservationUnavailable(RuntimeError):
    def __init__(self):
        super().__init__("Maintenance reservation storage is unavailable or inconsistent.")


def _lookup_id(value):
    if type(value) is not str or len(value) != 36:
        raise ValueError("Invalid maintenance lookup identity.")
    try:
        if str(UUID(value)) != value:
            raise ValueError
    except ValueError:
        raise ValueError("Invalid maintenance lookup identity.") from None
    return value


def _row(identity):
    return {
        "public_id": identity.public_id,
        "tenant": identity.intent.tenant,
        "subject": identity.intent.subject,
        "idempotency_key": identity.intent.idempotency_key,
        "product_run_id": identity.product_run_id,
        "session_id": identity.session_id,
        "workflow_session_id": identity.workflow_session_id,
        "intent_fingerprint": identity.intent.fingerprint,
        "identity_json": identity.model_dump_json(),
    }


async def _decode(db, row):
    try:
        if type(row["identity_json"]) is not str or len(row["identity_json"].encode()) > 262144:
            raise ReservationUnavailable()
        identity = MaintenanceRunIdentity.model_validate_json(row["identity_json"])
        if dict(row) != _row(identity):
            raise ReservationUnavailable()
        mappings = await db.fetch(
            f"SELECT task_id, phase FROM {_TASKS} WHERE public_id = %s", (identity.public_id,)
        )
        expected = {(task_id_for(identity, phase), phase.value) for phase in MaintenanceTaskPhase}
        if (
            len(mappings) != len(expected)
            or {(item["task_id"], item["phase"]) for item in mappings} != expected
        ):
            raise ReservationUnavailable()
        return identity
    except (KeyError, ValueError, TypeError, UnicodeError):
        raise ReservationUnavailable() from None


class _MaintenanceRunStore:
    def _transaction(self, *, write=False, initialize=False):
        raise NotImplementedError

    async def initialize(self):
        """Explicit coordinated schema bootstrap; never repair a partial schema."""
        async with self._transaction(write=True, initialize=True) as db:
            present = await db.tables()
            if not present:
                for ddl in _DDL.values():
                    await db.execute(ddl)
            await db.check_schema()

    async def check_ready(self):
        async with self._transaction() as db:
            await db.check_schema()

    async def reserve(
        self,
        intent: MaintenanceRunIntent,
        *,
        workflow_session_id: str | None = None,
        coding_expires_at: str | None = None,
    ) -> MaintenanceRunIdentity:
        current = copy_intent(intent)
        candidate = allocate_identity(
            current, workflow_session_id=workflow_session_id, coding_expires_at=coding_expires_at
        )
        async with self._transaction(write=True) as db:
            await db.check_schema()
            values = _row(candidate)
            await db.execute(
                f"INSERT INTO {_RUNS} ({', '.join(values)}) VALUES ({', '.join(['%s'] * len(values))}) "
                "ON CONFLICT (tenant, idempotency_key) DO NOTHING",
                tuple(values.values()),
            )
            rows = await db.fetch(
                f"SELECT {_SELECT} FROM {_RUNS} WHERE tenant = %s AND idempotency_key = %s",
                (current.tenant, current.idempotency_key),
            )
            if len(rows) != 1:
                raise ReservationUnavailable()
            if rows[0]["public_id"] == candidate.public_id:
                for phase in MaintenanceTaskPhase:
                    await db.execute(
                        f"INSERT INTO {_TASKS} (task_id, public_id, phase) VALUES (%s, %s, %s)",
                        (task_id_for(candidate, phase), candidate.public_id, phase.value),
                    )
            resolved = await _decode(db, rows[0])
            if (
                resolved.intent != current
                or (
                    workflow_session_id is not None
                    and resolved.workflow_session_id != candidate.workflow_session_id
                )
                or (
                    coding_expires_at is not None
                    and resolved.coding_expires_at != candidate.coding_expires_at
                )
            ):
                raise ReservationConflict()
        return resolved

    async def load_owned(self, *, tenant: str, public_id: str) -> MaintenanceRunIdentity | None:
        # Reuse strict identity bounds without transforming authenticated tenant.
        tenant = MaintenanceRunIntent(
            tenant=tenant, subject="lookup", idempotency_key="lookup", request_json="{}"
        ).tenant
        public_id = _lookup_id(public_id)
        async with self._transaction() as db:
            await db.check_schema()
            rows = await db.fetch(
                f"SELECT {_SELECT} FROM {_RUNS} WHERE tenant = %s AND public_id = %s",
                (tenant, public_id),
            )
            return None if not rows else await _decode(db, rows[0])

    async def load_for_task(self, task_id: str) -> MaintenanceRunIdentity | None:
        """Trusted worker entrance, never an unauthenticated product route."""
        task_id = _lookup_id(task_id)
        async with self._transaction() as db:
            await db.check_schema()
            mappings = await db.fetch(
                f"SELECT public_id, phase FROM {_TASKS} WHERE task_id = %s", (task_id,)
            )
            if not mappings:
                return None
            rows = await db.fetch(
                f"SELECT {_SELECT} FROM {_RUNS} WHERE public_id = %s", (mappings[0]["public_id"],)
            )
            if len(rows) != 1:
                raise ReservationUnavailable()
            identity = await _decode(db, rows[0])
            if task_id_for(identity, MaintenanceTaskPhase(mappings[0]["phase"])) != task_id:
                raise ReservationUnavailable()
            return identity


class _SQLiteSQL:
    def __init__(self, connection):
        self.connection = connection

    async def execute(self, sql, parameters=()):
        self.connection.execute(sql.replace("%s", "?"), parameters)

    async def fetch(self, sql, parameters=()):
        return [dict(row) for row in self.connection.execute(sql.replace("%s", "?"), parameters)]

    async def tables(self):
        rows = await self.fetch(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name IN (%s, %s)",
            (_RUNS, _TASKS),
        )
        return {row["name"] for row in rows}

    async def check_schema(self):
        rows = await self.fetch(
            "SELECT name, sql FROM sqlite_master WHERE type = 'table' AND name IN (%s, %s)",
            (_RUNS, _TASKS),
        )

        def normalize(value):
            return " ".join(value.split()).lower()

        if {row["name"]: normalize(row["sql"]) for row in rows} != {
            name: normalize(ddl) for name, ddl in _DDL.items()
        }:
            raise ReservationUnavailable()
        if await self.fetch(
            "SELECT name FROM sqlite_master WHERE type = 'trigger' AND tbl_name IN (%s, %s)",
            (_RUNS, _TASKS),
        ):
            raise ReservationUnavailable()


class SQLiteMaintenanceRunStore(_MaintenanceRunStore):
    """Small local fixture adapter, not the multi-process production deployment."""

    def __init__(self, path: Path):
        self.path = Path(path).absolute()

    @asynccontextmanager
    async def _transaction(self, *, write=False, initialize=False):
        import sqlite3

        connection = None
        try:
            mode = "rwc" if initialize else "rw" if write else "ro"
            connection = sqlite3.connect(self.path.as_uri() + "?mode=" + mode, uri=True, timeout=2)
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute("BEGIN IMMEDIATE" if write else "BEGIN")
            yield _SQLiteSQL(connection)
            connection.commit()
        except sqlite3.Error:
            raise ReservationUnavailable() from None
        finally:
            if connection is not None:
                connection.close()  # Uncommitted transactions roll back, including interrupted calls.


class _PostgresSQL:
    def __init__(self, connection):
        self.connection = connection

    async def execute(self, sql, parameters=()):
        await self.connection.execute(sql, parameters)

    async def fetch(self, sql, parameters=()):
        cursor = await self.connection.execute(sql, parameters)
        return await cursor.fetchall()

    async def tables(self):
        rows = await self.fetch(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = current_schema() AND table_name IN (%s, %s)",
            (_RUNS, _TASKS),
        )
        return {row["table_name"] for row in rows}

    async def check_schema(self):
        columns = await self.fetch(
            "SELECT table_name, column_name, data_type, is_nullable FROM information_schema.columns "
            "WHERE table_schema = current_schema() AND table_name IN (%s, %s)",
            (_RUNS, _TASKS),
        )
        if {
            (r["table_name"], r["column_name"], r["data_type"], r["is_nullable"]) for r in columns
        } != {
            (table, column, "text", "NO") for table, names in _COLUMNS.items() for column in names
        }:
            raise ReservationUnavailable()
        constraints = await self.fetch(
            """SELECT t.relname AS table_name, c.contype AS kind,
                ARRAY(SELECT a.attname FROM unnest(c.conkey) WITH ORDINALITY AS k(n, o)
                      JOIN pg_attribute a ON a.attrelid = c.conrelid AND a.attnum = k.n
                      ORDER BY k.o) AS columns,
                f.relname AS foreign_table,
                ARRAY(SELECT a.attname FROM unnest(c.confkey) WITH ORDINALITY AS k(n, o)
                      JOIN pg_attribute a ON a.attrelid = c.confrelid AND a.attnum = k.n
                      ORDER BY k.o) AS foreign_columns,
                c.convalidated, c.condeferrable, n.nspname AS table_schema,
                fn.nspname AS foreign_schema, c.confdeltype, c.confupdtype
                FROM pg_constraint c JOIN pg_class t ON t.oid = c.conrelid
                JOIN pg_namespace n ON n.oid = t.relnamespace
                LEFT JOIN pg_class f ON f.oid = c.confrelid
                LEFT JOIN pg_namespace fn ON fn.oid = f.relnamespace
                WHERE n.nspname = current_schema() AND t.relname IN (%s, %s)
                AND c.contype <> 'n'""",
            (_RUNS, _TASKS),
        )
        expected = {
            (_RUNS, "p", ("public_id",), None, ()),
            (_RUNS, "u", ("product_run_id",), None, ()),
            (_RUNS, "u", ("session_id",), None, ()),
            (_RUNS, "u", ("workflow_session_id",), None, ()),
            (_RUNS, "u", ("tenant", "idempotency_key"), None, ()),
            (_TASKS, "p", ("task_id",), None, ()),
            (_TASKS, "u", ("public_id", "phase"), None, ()),
            (_TASKS, "f", ("public_id",), _RUNS, ("public_id",)),
        }
        if (
            any(
                not r["convalidated"]
                or r["condeferrable"]
                or (
                    r["kind"] == "f"
                    and (
                        r["foreign_schema"] != r["table_schema"]
                        or r["confdeltype"] != "a"
                        or r["confupdtype"] != "a"
                    )
                )
                for r in constraints
            )
            or {
                (
                    r["table_name"],
                    r["kind"],
                    tuple(r["columns"]),
                    r["foreign_table"],
                    tuple(r["foreign_columns"]),
                )
                for r in constraints
            }
            != expected
        ):
            raise ReservationUnavailable()
        if await self.fetch(
            "SELECT g.tgname FROM pg_trigger g JOIN pg_class t ON t.oid = g.tgrelid "
            "JOIN pg_namespace n ON n.oid = t.relnamespace WHERE NOT g.tgisinternal "
            "AND n.nspname = current_schema() AND t.relname IN (%s, %s)",
            (_RUNS, _TASKS),
        ):
            raise ReservationUnavailable()


class PostgresMaintenanceRunStore(_MaintenanceRunStore):
    """Shared application reservation store; call initialize explicitly at deployment."""

    def __init__(self, dsn: str):
        self._dsn = dsn

    @asynccontextmanager
    async def _transaction(self, *, write=False, initialize=False):
        import psycopg
        from psycopg.rows import dict_row

        del initialize
        try:
            async with await psycopg.AsyncConnection[dict[str, Any]].connect(
                self._dsn,
                row_factory=dict_row,
                connect_timeout=3,
            ) as connection:
                if not write:
                    await connection.execute(
                        "SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY"
                    )
                await connection.execute("SET LOCAL statement_timeout = 3000")
                await connection.execute("SET LOCAL lock_timeout = 3000")
                yield _PostgresSQL(connection)
        except psycopg.Error:
            raise ReservationUnavailable() from None

"""Native creation owners compose independently and roll back partial writes."""

from __future__ import annotations

import asyncio
import inspect
import os
import sqlite3
import subprocess
import sys
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest

import cayu
from cayu.messages import Message
from cayu.sessions.forks import fork_session_invocation
from cayu.sessions.records import Session, SessionIdentity, SessionStatus
from cayu.sessions.requests import RunRequest
from cayu.storage import _sqlite_public_authority as public_authority_ops

TABLES = (
    "cayu_sessions",
    "cayu_session_labels",
    "cayu_transcript_messages",
    "cayu_checkpoints",
    "cayu_session_operations",
    "cayu_events",
    "cayu_deferred_interaction_inputs",
    "cayu_participant_session_bindings",
)
IDENTITY = SessionIdentity(provider_name="fake", model="fake-model")


def assert_owner_imports(backend):
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import importlib
import importlib.abc
import sys
class RejectAdapters(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in {"cayu.storage.sqlite", "cayu.storage.postgres", "psycopg", "psycopg_pool"}:
            raise AssertionError(f"Unexpected dependency: {fullname}")
sys.meta_path.insert(0, RejectAdapters())
owner = importlib.import_module(f"cayu.storage._{sys.argv[1]}_session_creation")
assert callable(owner.create)
assert callable(owner.create_fork)
""",
            backend,
        ],
        env={**os.environ, "PYTHONPATH": str(Path(cayu.__file__).resolve().parent.parent)},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_sqlite_creation_owner_imports_without_adapters_or_driver():
    assert_owner_imports("sqlite")


def compose(owner, connection, capabilities):
    return SimpleNamespace(
        **{
            name: partial(
                getattr(owner, name),
                connection,
                **{
                    key: value
                    for key, value in capabilities.items()
                    if key in inspect.signature(getattr(owner, name)).parameters
                },
            )
            for name in ("create", "create_fork")
        }
    )


async def seed(store):
    source = await store.create(
        RunRequest(
            agent_name="assistant",
            session_id=f"source-{uuid4().hex}",
            messages=[Message.text("user", "source message")],
            labels={"purpose": "source"},
        ),
        identity=IDENTITY,
    )
    await store.append_transcript_messages(source.id, [Message.text("user", "source message")])
    await store.checkpoint(source.id, {"version": 2})
    await store.update_status(source.id, SessionStatus.COMPLETED)
    return await store.load(source.id)


async def exercise(ops, source, snapshot, fault):
    async def rollback_then_retry(operation, table):
        before = await snapshot()
        fault.table = table
        try:
            with pytest.raises(fault.error):
                await operation()
        finally:
            fault.table = None
        assert await snapshot() == before
        return await operation()

    for index, table in enumerate(("cayu_sessions", "cayu_checkpoints")):
        request = RunRequest(
            agent_name="assistant",
            session_id=f"{source.id}-created-{index}",
            messages=[Message.text("user", "child message")],
            parent_session_id=source.id,
            labels={"purpose": "child"},
            metadata={"created": True},
        )
        create = partial(
            ops.create,
            request,
            identity=IDENTITY,
            checkpoint_transform=lambda _session, _checkpoint: {"version": 3},
        )
        created = await rollback_then_retry(create, table)
        assert created.id == request.session_id
        assert created.parent_session_id == source.id
        assert created.labels == request.labels
        before = await snapshot()
        with pytest.raises(ValueError):
            await create()
        assert await snapshot() == before

    for index, table in enumerate(("cayu_transcript_messages", "cayu_checkpoints")):
        fork = Session(
            id=f"{source.id}-fork-{index}",
            agent_name=source.agent_name,
            provider_name=source.provider_name,
            model=source.model,
            parent_session_id=source.id,
            invocation=fork_session_invocation(source),
            status=SessionStatus.COMPLETED,
            labels={"purpose": "fork"},
        )

        def transform(session, checkpoint):
            assert session.id == source.id and checkpoint == {"version": 2}
            return {"copied_version": checkpoint["version"]}

        create_fork = partial(
            ops.create_fork,
            source_session_id=source.id,
            fork=fork,
            source_statuses={SessionStatus.COMPLETED},
            transcript_cursor=None,
            checkpoint_transform=transform,
            system_prompt_replacement=None,
            expected_source_run_epoch=source.run_epoch,
            transcript_validator=None,
        )
        copied = await rollback_then_retry(create_fork, table)
        assert copied.id == fork.id and copied.labels == fork.labels


async def assert_persisted(store, source):
    assert await store.load_checkpoint(source.id) == {"version": 2}
    for index in range(2):
        assert await store.load_checkpoint(f"{source.id}-created-{index}") == {"version": 3}
        assert await store.load_checkpoint(f"{source.id}-fork-{index}") == {"copied_version": 2}
        transcript = await store.load_transcript(f"{source.id}-fork-{index}")
        assert transcript == [Message.text("user", "source message")]


def test_sqlite_creation_owner_composes_with_direct_connection(tmp_path):
    from cayu.storage import _sqlite_session_creation as owner
    from cayu.storage._sqlite_functions import _register_sqlite_functions
    from cayu.storage.sqlite import SQLiteSessionStore, _touch_session_activity

    async def run():
        path = tmp_path / "creation.sqlite"
        store = SQLiteSessionStore(path)
        try:
            source = await seed(store)
        finally:
            await store.close()
        del store
        lock = asyncio.Lock()
        fault = SimpleNamespace(table=None, error=RuntimeError)

        class Connection(sqlite3.Connection):
            def execute(self, query, parameters=()):
                result = super().execute(query, parameters)
                self.fail_after_write(query)
                return result

            def executemany(self, query, parameters):
                result = super().executemany(query, parameters)
                self.fail_after_write(query)
                return result

            def fail_after_write(self, query):
                if fault.table and " ".join(query.split()).startswith(
                    f"INSERT INTO {fault.table} "
                ):
                    assert lock.locked() and self.in_transaction
                    raise fault.error("injected creation write failure")

        connection = sqlite3.connect(path, factory=Connection)
        connection.row_factory = sqlite3.Row
        _register_sqlite_functions(connection)
        connection.execute("PRAGMA foreign_keys = ON")
        readers = SimpleNamespace(
            _connection=connection,
            _public_authority_alias_codec=None,
            public_authority_alias_codec=None,
        )
        readers._closure_lineage_owners_unlocked = partial(
            SQLiteSessionStore._closure_lineage_owners_unlocked, readers
        )
        public_authority_ops.register_sql_functions(connection, codec=None)

        def now():
            assert lock.locked() and connection.in_transaction
            return datetime(2026, 1, 1, tzinfo=UTC)

        capabilities = {
            "lock": lock,
            "require_authority_configuration": partial(
                public_authority_ops.require_current_configuration, codec=None
            ),
            "store_now": now,
            "closure_owners": readers._closure_lineage_owners_unlocked,
            "supports_model_failover": lambda: True,
            "touch_activity": _touch_session_activity,
        }
        for name, method in {
            "require_available_closure_identity": "_require_available_closure_identity_unlocked",
            "require_external_creation": "_require_external_creation_unlocked",
            "load_checkpoint": "_load_checkpoint_unlocked",
            "load_session": "_load_unlocked",
            "session_exists": "_session_exists_unlocked",
        }.items():
            capabilities[name] = partial(getattr(SQLiteSessionStore, method), readers)
        ops = compose(owner, connection, capabilities)

        async def snapshot():
            assert not lock.locked() and not connection.in_transaction
            return tuple(
                tuple(tuple(row) for row in connection.execute(f"SELECT * FROM {table} ORDER BY 1"))
                for table in TABLES
            )

        try:
            await exercise(ops, source, snapshot, fault)
        finally:
            connection.close()
        reopened = SQLiteSessionStore(path)
        try:
            await assert_persisted(reopened, source)
        finally:
            await reopened.close()

    asyncio.run(run())

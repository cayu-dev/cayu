"""Native alias registries compose independently of session-store instances."""

from __future__ import annotations

import asyncio
import base64
import os
import sqlite3
import subprocess
import sys
import threading
from contextlib import asynccontextmanager
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest
from pydantic import SecretStr

import cayu
from cayu.events import Event, EventType
from cayu.runtime.public_authority import PublicAuthorityAliasCodec, PublicAuthorityAliasKeyring
from cayu.sessions.records import SessionIdentity
from cayu.sessions.requests import RunRequest

OPERATIONS = (
    "register_public_authority_alias",
    "resolve_public_authority_alias",
    "public_authority_private_value_exists",
)
TABLES = (
    "cayu_public_authority_alias_keys",
    "cayu_public_authority_alias_config",
    "cayu_public_authority_aliases",
)


def codec(*, rotated=False):
    keys = {"first": SecretStr(base64.urlsafe_b64encode(bytes([31]) * 32).decode().rstrip("="))}
    if rotated:
        keys["second"] = SecretStr(base64.urlsafe_b64encode(bytes([32]) * 32).decode().rstrip("="))
    return PublicAuthorityAliasCodec(
        PublicAuthorityAliasKeyring(active_key_id="second" if rotated else "first", keys=keys)
    )


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
        if fullname in {"cayu.storage.sqlite", "cayu.storage.postgres", "cayu.storage._postgres_base", "psycopg", "psycopg_pool"}:
            raise AssertionError(f"Unexpected dependency: {fullname}")
sys.meta_path.insert(0, RejectAdapters())
owner = importlib.import_module(f"cayu.storage._{sys.argv[1]}_public_authority")
for name in sys.argv[2:]:
    assert callable(getattr(owner, name))
""",
            backend,
            *OPERATIONS,
        ],
        env={**os.environ, "PYTHONPATH": str(Path(cayu.__file__).resolve().parent.parent)},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


async def seed(store):
    session = f"authority-{uuid4().hex}"
    await store.create(
        RunRequest(agent_name="assistant", session_id=session, messages=[]),
        identity=SessionIdentity(provider_name="fake", model="fake"),
    )
    await store.append_event(
        session,
        Event(
            type=EventType.TURN_COMPLETED,
            session_id=session,
            interaction_id="outer",
            payload={"interaction_ids": ["nested"]},
        ),
    )
    return session


async def exercise(ops, session, fault, clean, snapshot, reconcile, require_current):
    initial, rotated = codec(), codec(rotated=True)
    before = await snapshot()
    fault.target = "INSERT INTO cayu_public_authority_alias_keys"
    try:
        with pytest.raises(fault.error):
            await reconcile(initial)
    finally:
        fault.target = None
    clean()
    assert await snapshot() == before
    await reconcile(initial)
    await require_current(initial)
    await reconcile(initial, read_only=True)
    for field, value, scope in [
        ("session_id", session, None),
        ("interaction_id", "outer", session),
        ("interaction_id", "nested", session),
    ]:
        assert (
            await ops.resolve_public_authority_alias(
                initial.encode(value, field_name=field, session_id=scope),
                field_name=field,
                scope_session_id=scope,
                codec=initial,
            )
            == value
        )
    kwargs = dict(field_name="interaction_id", scope_session_id=session, codec=initial)
    alias = initial.encode("manual", field_name="interaction_id", session_id=session)
    before = await snapshot()
    fault.target = "INSERT INTO cayu_public_authority_aliases"
    try:
        with pytest.raises(fault.error):
            await ops.register_public_authority_alias(alias, private_value="manual", **kwargs)
    finally:
        fault.target = None
    clean()
    assert await snapshot() == before
    await ops.register_public_authority_alias(alias, private_value="manual", **kwargs)
    assert await ops.resolve_public_authority_alias(alias, **kwargs) == "manual"
    assert await ops.public_authority_private_value_exists("manual", **kwargs)
    assert (
        await ops.resolve_public_authority_alias(alias, **{**kwargs, "scope_session_id": "other"})
        is None
    )
    await reconcile(rotated)
    await require_current(rotated)
    await reconcile(rotated, read_only=True)
    with pytest.raises(RuntimeError, match="stale"):
        await require_current(initial)
    with pytest.raises((ValueError, RuntimeError), match="stale"):
        await reconcile(initial, read_only=True)
    for value in ["outer", "nested"]:
        assert (
            await ops.resolve_public_authority_alias(
                rotated.encode(value, field_name="interaction_id", session_id=session),
                field_name="interaction_id",
                scope_session_id=session,
                codec=rotated,
            )
            == value
        )
    clean()


@asynccontextmanager
async def sqlite_composition(tmp_path, error):
    from cayu.storage import _sqlite_public_authority as owner
    from cayu.storage._sqlite_connection import _run_off_thread_with_connection_ownership
    from cayu.storage._sqlite_functions import _register_sqlite_functions
    from cayu.storage.sqlite import SQLiteSessionStore

    path = tmp_path / "authority.sqlite"
    store = SQLiteSessionStore(path)
    try:
        session = await seed(store)
    finally:
        await store.close()
    del store
    lock = asyncio.Lock()
    fault = SimpleNamespace(
        target=None,
        error=error,
        block=False,
        started=asyncio.Event(),
        release=threading.Event(),
        loop=asyncio.get_running_loop(),
    )

    class Connection(sqlite3.Connection):
        def execute(self, query, parameters=()):
            result = super().execute(query, parameters)
            if fault.target is not None and " ".join(query.split()).startswith(fault.target):
                assert lock.locked() and self.in_transaction
                if fault.block:
                    fault.loop.call_soon_threadsafe(fault.started.set)
                    if not fault.release.wait(10):
                        raise TimeoutError("test did not release alias write")
                else:
                    raise fault.error("injected after a native alias write")
            return result

    connection = sqlite3.connect(path, check_same_thread=False, factory=Connection)
    connection.row_factory = sqlite3.Row
    _register_sqlite_functions(connection)

    async def execute(operation):
        return await _run_off_thread_with_connection_ownership(lock, connection, operation)

    async def reconcile(selected, *, read_only=False):
        def initialize(conn):
            owner.register_sql_functions(conn, codec=selected)
            owner.initialize_registry(conn, codec=selected, read_only=read_only)

        await execute(initialize)

    async def snapshot():
        return await execute(
            lambda conn: tuple(
                tuple(tuple(row) for row in conn.execute(f"SELECT * FROM {table} ORDER BY 1, 2"))
                for table in TABLES
            )
        )

    def clean():
        assert not lock.locked() and not connection.in_transaction

    async def require_current(selected):
        await execute(partial(owner.require_current_configuration, codec=selected))

    ops = SimpleNamespace(**{name: partial(getattr(owner, name), execute) for name in OPERATIONS})
    try:
        yield ops, session, fault, clean, snapshot, reconcile, require_current, lock
    finally:
        fault.release.set()
        async with lock:
            connection.close()


def test_sqlite_public_authority_imports_without_adapters_or_driver():
    assert_owner_imports("sqlite")


@pytest.mark.parametrize("error", [RuntimeError, asyncio.CancelledError])
def test_sqlite_public_authority_composes_and_rolls_back_failed_writes(tmp_path, error):
    async def run():
        async with sqlite_composition(tmp_path, error) as parts:
            await exercise(*parts[:-1])

    asyncio.run(run())


def test_sqlite_public_authority_cancellation_retains_writer_until_commit(tmp_path):
    async def run():
        async with sqlite_composition(tmp_path, RuntimeError) as (
            ops,
            _session,
            fault,
            clean,
            _snapshot,
            reconcile,
            _require_current,
            lock,
        ):
            selected = codec()
            await reconcile(selected)
            alias = selected.encode("cancelled-writer", field_name="session_id")
            fault.target, fault.block = "INSERT INTO cayu_public_authority_aliases", True
            task = asyncio.create_task(
                ops.register_public_authority_alias(
                    alias, field_name="session_id", private_value="cancelled-writer", codec=selected
                )
            )
            try:
                await asyncio.wait_for(fault.started.wait(), 5)
                task.cancel()
                await asyncio.sleep(0)
                await asyncio.sleep(0)
                assert not task.done() and lock.locked()
            finally:
                fault.release.set()
                await asyncio.gather(task, return_exceptions=True)
            assert task.cancelled()
            clean()
            fault.target = None
            # SQLite caller cancellation waits for the physical write; it does
            # not imply that the transaction was rolled back.
            assert (
                await ops.resolve_public_authority_alias(
                    alias, field_name="session_id", codec=selected
                )
                == "cancelled-writer"
            )
            clean()

    asyncio.run(run())

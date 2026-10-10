"""Context-view owners compose with native connections and preserve rollback."""

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

import pytest
from tests.core.test_context_view_admission import _ownership, _selection
from tests.core.test_context_views import _manifest_for_store

import cayu

OPERATIONS = (
    "publish_context_view",
    "lookup_context_view_publication",
    "select_context_view",
    "lookup_context_view_selection",
    "transition_context_view_ownership",
    "read_context_view_lifecycle_events",
    "read_context_view",
)
TABLES = (
    "cayu_context_views",
    "cayu_context_view_selections",
    "cayu_context_view_ownership_operations",
    "cayu_context_view_lifecycle_events",
    "cayu_context_selection_exclusions",
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
        if fullname in {"cayu.storage.sqlite", "cayu.storage.postgres", "psycopg", "psycopg_pool"}:
            raise AssertionError(f"Unexpected dependency: {fullname}")
sys.meta_path.insert(0, RejectAdapters())
owner = importlib.import_module(f"cayu.storage._{sys.argv[1]}_context_views")
assert callable(owner.publish_context_view)
assert callable(owner.transition_context_view_ownership)
""",
            backend,
        ],
        env={**os.environ, "PYTHONPATH": str(Path(cayu.__file__).resolve().parent.parent)},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_sqlite_context_view_owner_imports_without_adapters_or_driver():
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
            for name in OPERATIONS
        }
    )


async def exercise(ops, manifest, snapshot, fault):
    async def rollback_then_retry(operation, table):
        before = await snapshot()
        fault.table = table
        try:
            with pytest.raises(RuntimeError, match="injected view write failure"):
                await operation()
        finally:
            fault.table = None
        assert await snapshot() == before
        return await operation()

    publish = partial(ops.publish_context_view, manifest, publication_key="publication")
    assert await rollback_then_retry(publish, "cayu_context_views") == manifest
    assert await publish() == manifest
    assert await ops.lookup_context_view_publication("publication") == manifest
    readback = await ops.read_context_view(
        manifest.view_id, source_session_id=manifest.source_session_id
    )
    assert readback.view == manifest and readback.historical_only
    with pytest.raises(LookupError):
        await ops.read_context_view(manifest.view_id, source_session_id="wrong-session")

    request = _selection(manifest, "selection")
    select = partial(ops.select_context_view, request)
    selected = await rollback_then_retry(select, "cayu_context_view_selections")
    assert selected.state == "selected"
    assert await select() == selected
    assert await ops.lookup_context_view_selection(request.selection_key) == selected

    adopt = partial(
        ops.transition_context_view_ownership, _ownership(selected, "adopt", operation="adopt")
    )
    adopted = await rollback_then_retry(adopt, "cayu_context_view_lifecycle_events")
    assert adopted.state == "adopted" and adopted.ownership_revision == 2
    assert await adopt() == adopted
    before = await snapshot()
    with pytest.raises(ValueError, match="stale"):
        await ops.transition_context_view_ownership(_ownership(selected, "stale-release"))
    assert await snapshot() == before

    release = partial(ops.transition_context_view_ownership, _ownership(adopted, "release"))
    released = await rollback_then_retry(release, "cayu_context_view_lifecycle_events")
    assert released.state == "released"
    assert await release() == released
    assert await ops.lookup_context_view_selection(request.selection_key) == released
    events = await ops.read_context_view_lifecycle_events(manifest.view_id)
    assert tuple(event.state for event in events) == ("adopted", "released")


def test_sqlite_context_view_owner_composes_with_direct_connection(tmp_path):
    from cayu.storage import _sqlite_context_views as owner
    from cayu.storage.sqlite import SQLiteSessionStore

    async def run():
        path = tmp_path / "views.sqlite"
        store = SQLiteSessionStore(path)
        try:
            manifest = await _manifest_for_store(store)
        finally:
            await store.close()
        del store
        lock = asyncio.Lock()
        fault = SimpleNamespace(table=None)

        class Connection(sqlite3.Connection):
            def execute(self, query, parameters=()):
                result = super().execute(query, parameters)
                if fault.table and " ".join(query.split()).startswith(
                    f"INSERT INTO {fault.table} "
                ):
                    assert lock.locked() and self.in_transaction
                    raise RuntimeError("injected view write failure")
                return result

        connection = sqlite3.connect(path, factory=Connection)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")

        def now():
            assert lock.locked() and connection.in_transaction
            return datetime(2026, 1, 1, tzinfo=UTC)

        ops = compose(owner, connection, {"lock": lock, "store_now": now})

        async def snapshot():
            assert not lock.locked() and not connection.in_transaction
            return tuple(
                tuple(tuple(row) for row in connection.execute(f"SELECT * FROM {table} ORDER BY 1"))
                for table in TABLES
            )

        try:
            await exercise(ops, manifest, snapshot, fault)
        finally:
            connection.close()

    asyncio.run(run())

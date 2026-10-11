"""Terminal evidence composes without retaining a native session store."""

from __future__ import annotations

import asyncio
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
from tests.core.test_terminal_session_evidence import (
    _create_interrupted_session,
    _create_terminal_session,
)

import cayu
from cayu.runtime.evidence_spool import EvidenceSpool, IncrementalEvidenceLimits

OPERATIONS = (
    "load_terminal_session_evidence",
    "load_runner_owned_interrupted_evidence",
    "export_terminal_session_evidence",
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
owner = importlib.import_module(f"cayu.storage._{sys.argv[1]}_terminal_evidence")
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
    prefix = f"terminal-{uuid4().hex}"
    complete, _ = await _create_terminal_session(
        store, session_id=prefix, interaction_id=f"{prefix}-interaction"
    )
    interrupted = f"{prefix}-interrupted"
    observed = await _create_interrupted_session(
        store, session_id=interrupted, interaction_id=f"{interrupted}-interaction"
    )
    return SimpleNamespace(
        complete=complete,
        interrupted=interrupted,
        observed=observed,
        eager=await store.load_terminal_session_evidence(complete),
        proof=await store.load_runner_owned_interrupted_evidence(
            interrupted, observed_events=observed
        ),
    )


async def assert_export(ops, evidence):
    with EvidenceSpool(IncrementalEvidenceLimits(batch_records=1)) as spool:
        await ops.export_terminal_session_evidence(evidence.complete, spool=spool)
        assert spool.phase == "snapshot_complete"
        assert spool.session == evidence.eager.session
        assert spool.terminal_publication_marker == evidence.eager.terminal_publication_marker
        assert spool.boundary == evidence.eager.boundary
        assert tuple(spool.events) == evidence.eager.events
        assert tuple(spool.transcript) == evidence.eager.transcript
        assert spool.peak_page_records == 1


async def exercise(ops, evidence, fault, assert_clean):
    for operation, expected in (
        (partial(ops.load_terminal_session_evidence, evidence.complete), evidence.eager),
        (
            partial(
                ops.load_runner_owned_interrupted_evidence,
                evidence.interrupted,
                observed_events=evidence.observed,
            ),
            evidence.proof,
        ),
    ):
        fault.enabled = True
        try:
            with pytest.raises(fault.error):
                await operation()
        finally:
            fault.enabled = False
        assert_clean()
        assert await operation() == expected
        assert_clean()

    class FailingSpool(EvidenceSpool):
        def append(self, kind, record):
            super().append(kind, record)
            raise fault.error("injected after a streamed record was written")

    with FailingSpool(IncrementalEvidenceLimits(batch_records=1)) as spool:
        with pytest.raises(fault.error):
            await ops.export_terminal_session_evidence(evidence.complete, spool=spool)
        assert spool.phase != "snapshot_complete"
        assert_clean()
    await assert_export(ops, evidence)
    assert_clean()


@asynccontextmanager
async def sqlite_composition(tmp_path, error):
    from cayu.storage import _sqlite_terminal_evidence as owner
    from cayu.storage._sqlite_connection import _run_off_thread_with_connection_ownership
    from cayu.storage._sqlite_functions import _register_sqlite_functions
    from cayu.storage.sqlite import SQLiteSessionStore

    path = tmp_path / "terminal.sqlite"
    store = SQLiteSessionStore(path)
    try:
        evidence = await seed(store)
    finally:
        await store.close()
    del store
    lock = asyncio.Lock()
    fault = SimpleNamespace(
        enabled=False,
        error=error,
        block=False,
        started=asyncio.Event(),
        release=threading.Event(),
        loop=asyncio.get_running_loop(),
    )

    class Connection(sqlite3.Connection):
        progress_active = False

        def execute(self, query, parameters=()):
            result = super().execute(query, parameters)
            if " ".join(query.split()).startswith("SELECT session.status,"):
                assert lock.locked() and self.in_transaction
                if fault.block:
                    fault.loop.call_soon_threadsafe(fault.started.set)
                    if not fault.release.wait(10):
                        raise TimeoutError("test did not release native evidence read")
                if fault.enabled:
                    raise fault.error("injected after a native snapshot read")
            return result

        def set_progress_handler(self, callback, steps):
            super().set_progress_handler(callback, steps)
            self.progress_active = callback is not None

    connection = sqlite3.connect(
        path.as_uri() + "?mode=ro", uri=True, check_same_thread=False, factory=Connection
    )
    connection.row_factory = sqlite3.Row
    _register_sqlite_functions(connection)

    async def execute(operation):
        return await _run_off_thread_with_connection_ownership(lock, connection, operation)

    def assert_clean():
        assert not lock.locked() and not connection.in_transaction
        assert not connection.progress_active

    ops = SimpleNamespace(**{name: partial(getattr(owner, name), execute) for name in OPERATIONS})
    try:
        yield ops, evidence, fault, assert_clean, lock
    finally:
        fault.release.set()
        async with lock:
            connection.close()


def test_sqlite_terminal_evidence_imports_without_adapters_or_driver():
    assert_owner_imports("sqlite")


@pytest.mark.parametrize("error", [RuntimeError, asyncio.CancelledError])
def test_sqlite_terminal_evidence_composes_and_releases_failed_snapshots(tmp_path, error):
    async def run():
        async with sqlite_composition(tmp_path, error) as (ops, evidence, fault, clean, _):
            await exercise(ops, evidence, fault, clean)

    asyncio.run(run())


def test_sqlite_terminal_export_cancellation_waits_for_native_reader(tmp_path):
    async def run():
        async with sqlite_composition(tmp_path, RuntimeError) as (
            ops,
            evidence,
            fault,
            clean,
            lock,
        ):
            fault.block = True
            with EvidenceSpool(IncrementalEvidenceLimits(batch_records=1)) as spool:
                task = asyncio.create_task(
                    ops.export_terminal_session_evidence(evidence.complete, spool=spool)
                )
                try:
                    await asyncio.wait_for(fault.started.wait(), 5)
                    task.cancel()
                    await asyncio.sleep(0)
                    await asyncio.sleep(0)
                    assert not task.done() and lock.locked()
                    assert not spool._closed
                finally:
                    fault.release.set()
                    await asyncio.gather(task, return_exceptions=True)
                assert task.cancelled()
                assert spool.phase != "snapshot_complete"
                clean()
            fault.block = False
            await assert_export(ops, evidence)
            clean()

    asyncio.run(run())

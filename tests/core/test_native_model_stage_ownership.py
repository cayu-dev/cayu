"""Native model-stage operations compose without retaining a session store."""

from __future__ import annotations

import asyncio
import inspect
import os
import sqlite3
import subprocess
import sys
import threading
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest

import cayu
from cayu.events import Event, EventType
from cayu.sessions import _model_stage_rules as rules
from cayu.sessions._checkpoint_mutations import RuntimePublicationMutation
from cayu.sessions._model_stage import ModelCompletionStageRequest
from cayu.sessions._runtime_publication import RuntimePublicationRequest
from cayu.sessions.records import SessionIdentity, SessionStatus
from cayu.sessions.requests import RunRequest

OPERATIONS = (
    "load_stage_records",
    "load_settlement_record",
    "load_dispatch_record",
    "load_active_stage_records",
    "prepare",
    "mark_dispatched",
    "complete",
    "abandon",
    "promote",
)
TABLES = ("cayu_sessions", "cayu_session_operations", "cayu_checkpoints", "cayu_events")


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
owner = importlib.import_module(f"cayu.storage._{sys.argv[1]}_model_stage")
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
    session = f"stage-{uuid4().hex}"
    await store.create(
        RunRequest(agent_name="assistant", session_id=session, messages=[]),
        identity=SessionIdentity(provider_name="fake", model="fake"),
    )
    return await store.transition_status(
        session, from_statuses={SessionStatus.PENDING}, to_status=SessionStatus.RUNNING
    )


def preparation(session, name="stage"):
    return rules._prepare_model_completion_stage(
        session.id,
        ModelCompletionStageRequest(
            stage_id=name,
            logical_step_id=f"step-{name}",
            dispatch_ordinal=0,
            purpose="context-compaction",
            intent={
                "operation_id": name,
                "model_attempt_id": f"attempt-{name}",
                "interaction_id": f"interaction-{session.id}",
                "request_fingerprint": "a" * 64,
            },
        ),
        expected_statuses={SessionStatus.RUNNING},
        expected_run_epoch=session.run_epoch,
        expected_transcript_cursor=0,
    )


def compose(owner, capabilities):
    return SimpleNamespace(
        **{
            name: partial(
                getattr(owner, name),
                **{
                    key: value
                    for key, value in capabilities.items()
                    if key in inspect.signature(getattr(owner, name)).parameters
                },
            )
            for name in OPERATIONS
        }
    )


async def exercise(ops, session, fault, clean, snapshot):
    async def fail_then_retry(call):
        before = await snapshot()
        fault.enabled = True
        try:
            with pytest.raises(fault.error, match="injected"):
                await call()
        finally:
            fault.enabled = False
        clean()
        assert await snapshot() == before
        result = await call()
        clean()
        return result

    prepared = preparation(session, "abandon")
    result = await fail_then_retry(partial(ops.prepare, prepared))
    assert result.stage.state == "in_flight" and not result.replayed
    replay = await ops.prepare(prepared)
    assert replay.replayed and replay.stage == result.stage
    active, record, terminal = await ops.load_active_stage_records(session.id)
    assert active and record and terminal is None
    assert await ops.load_stage_records(
        session.id, prepared.preparation_storage_key, prepared.terminal_storage_key
    ) == (record, None)
    abandoned = rules._prepare_model_completion_stage_abandonment(
        session.id,
        stage_id=result.stage.stage_id,
        preparation_digest=result.stage.preparation_digest,
        expected_run_epoch=session.run_epoch,
        stage_source_run_epoch=session.run_epoch,
    )
    assert not (await fail_then_retry(partial(ops.abandon, abandoned))).replayed
    assert (await ops.abandon(abandoned)).replayed
    assert await ops.load_active_stage_records(session.id) == (None, None, None)

    prepared = preparation(session, "complete")
    stage = (await ops.prepare(prepared)).stage
    dispatch = await fail_then_retry(
        partial(
            ops.mark_dispatched, session.id, stage=stage, consume_child_session_notifications=True
        )
    )
    assert dispatch.stage_id == stage.stage_id
    assert (
        await ops.mark_dispatched(session.id, stage=stage, consume_child_session_notifications=True)
        == dispatch
    )
    assert (
        await ops.load_dispatch_record(
            session.id, rules._model_completion_stage_dispatch_storage_key(stage.stage_id)
        )
        is not None
    )
    assert (
        await ops.load_settlement_record(
            session.id, rules._model_completion_stage_settlement_storage_key(stage.stage_id)
        )
        is None
    )
    publication = RuntimePublicationRequest(
        publication_id=stage.logical_step_id,
        kind="context-compaction",
        intent=stage.intent,
        interaction_id=stage.intent["interaction_id"],
        mutation=RuntimePublicationMutation(),
        transcript_messages=(),
        events=(
            Event(
                type=EventType.MODEL_COMPLETED,
                session_id=session.id,
                interaction_id=stage.intent["interaction_id"],
                payload={"purpose": "context_compaction"},
            ),
        ),
    )
    terminal = rules._prepare_model_completion_stage_terminal(
        session.id, stage_id=stage.stage_id, publication=publication
    )
    completed = await fail_then_retry(partial(ops.complete, terminal))
    assert completed.stage.state == "completed" and not completed.replayed
    assert (await ops.complete(terminal)).replayed
    assert (
        await ops.load_stage_records(
            session.id, prepared.preparation_storage_key, prepared.terminal_storage_key
        )
    )[1] is not None

    # Promotion composes the existing publication transaction through its
    # capability; public store conformance covers that transaction end to end.
    marker = object()
    calls = []

    async def load_stage(session_id, stage_id):
        assert (session_id, stage_id) == (session.id, stage.stage_id)
        return completed.stage

    async def publish(prepared_publication, *, _model_completion_stage):
        calls.append((prepared_publication, _model_completion_stage))
        return marker

    assert (
        await ops.promote(
            session_id=session.id,
            stage_id=stage.stage_id,
            preparation_storage_key=prepared.preparation_storage_key,
            terminal_storage_key=prepared.terminal_storage_key,
            expected_run_epoch=session.run_epoch,
            load_stage=load_stage,
            publish=publish,
        )
        is marker
    )
    assert len(calls) == 1
    assert calls[0][0].request == publication
    assert calls[0][1].completion_digest == completed.stage.completion_digest
    clean()


@asynccontextmanager
async def sqlite_composition(tmp_path, error):
    from cayu.storage import _sqlite_model_stage as owner
    from cayu.storage import _sqlite_records as records
    from cayu.storage import sqlite as adapter
    from cayu.storage._sqlite_connection import _run_off_thread_with_connection_ownership
    from cayu.storage._sqlite_functions import _register_sqlite_functions
    from cayu.storage._sqlite_public_authority import register_sql_functions

    path = tmp_path / "model-stage.sqlite"
    store = adapter.SQLiteSessionStore(path)
    try:
        session = await seed(store)
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
        def execute(self, query, parameters=()):
            result = super().execute(query, parameters)
            if fault.enabled and " ".join(query.split()).startswith(
                "INSERT INTO cayu_session_operations"
            ):
                assert lock.locked() and self.in_transaction
                if fault.block:
                    fault.loop.call_soon_threadsafe(fault.started.set)
                    if not fault.release.wait(10):
                        raise TimeoutError("test did not release stage write")
                else:
                    raise fault.error("injected after native stage write")
            return result

    connection = sqlite3.connect(path, check_same_thread=False, factory=Connection)
    connection.row_factory = sqlite3.Row
    _register_sqlite_functions(connection)
    register_sql_functions(connection, codec=None)

    async def execute(operation):
        return await _run_off_thread_with_connection_ownership(lock, connection, operation)

    def now():
        assert lock.locked() and connection.in_transaction
        return datetime.now(UTC)

    async def snapshot():
        return await execute(
            lambda conn: tuple(
                tuple(tuple(row) for row in conn.execute(f"SELECT * FROM {table} ORDER BY 1, 2"))
                for table in TABLES
            )
        )

    def clean():
        assert not lock.locked() and not connection.in_transaction

    ops = compose(
        owner,
        dict(
            run_read=execute,
            run_write=execute,
            load_session=partial(records.load_session, connection),
            load_checkpoint=partial(adapter._load_checkpoint_state, connection),
            store_now=now,
            append_events=adapter._append_events_in_transaction,
            reject_steering=adapter._reject_new_work_after_steering,
        ),
    )
    try:
        yield ops, session, fault, clean, snapshot, lock
    finally:
        fault.release.set()
        async with lock:
            connection.close()


def test_sqlite_model_stage_imports_without_adapters_or_driver():
    assert_owner_imports("sqlite")


@pytest.mark.parametrize("error", [RuntimeError, asyncio.CancelledError])
def test_sqlite_model_stage_composes_and_rolls_back_failed_writes(tmp_path, error):
    async def run():
        async with sqlite_composition(tmp_path, error) as parts:
            await exercise(*parts[:-1])

    asyncio.run(run())


def test_sqlite_model_stage_cancellation_retains_writer_until_commit(tmp_path):
    async def run():
        async with sqlite_composition(tmp_path, RuntimeError) as (
            ops,
            session,
            fault,
            clean,
            _snapshot,
            lock,
        ):
            prepared = preparation(session)
            fault.enabled, fault.block = True, True
            task = asyncio.create_task(ops.prepare(prepared))
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
            fault.enabled = False
            # Caller cancellation retains the physical worker through commit.
            assert (await ops.prepare(prepared)).replayed
            clean()

    asyncio.run(run())

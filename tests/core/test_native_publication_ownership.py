"""Native publication composes with real SQL after the seeding store is closed."""

from __future__ import annotations

import asyncio
import inspect
import os
import subprocess
import sys
from contextlib import asynccontextmanager
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import pytest
from tests.core.test_native_model_stage_ownership import sqlite_composition

import cayu
from cayu.events import Event, EventType
from cayu.messages import Message
from cayu.sessions._checkpoint_mutations import runtime_publication_checkpoint_mutation
from cayu.sessions._runtime_publication import RuntimePublicationRequest
from cayu.sessions._runtime_publication_rules import _prepare_runtime_publication
from cayu.sessions.records import SessionStatus

TABLES = (
    "cayu_sessions",
    "cayu_session_operations",
    "cayu_checkpoints",
    "cayu_transcript_messages",
    "cayu_events",
    "cayu_persisted_event_side_effects",
    "cayu_budget_reservation_identities",
)


def assert_imports(backend):
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import importlib, importlib.abc, sys
class RejectAdapters(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in {"cayu.storage.sqlite", "cayu.storage.postgres", "cayu.storage._postgres_base", "psycopg", "psycopg_pool"}:
            raise AssertionError(fullname)
sys.meta_path.insert(0, RejectAdapters())
owner = importlib.import_module(f"cayu.storage._{sys.argv[1]}_publication")
assert callable(owner.publish_runtime) and callable(owner.load_receipt_record)
""",
            backend,
        ],
        env={**os.environ, "PYTHONPATH": str(Path(cayu.__file__).resolve().parent.parent)},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def bind(owner, name, capabilities):
    operation = getattr(owner, name)
    return partial(
        operation,
        **{
            key: value
            for key, value in capabilities.items()
            if key in inspect.signature(operation).parameters
        },
    )


def preparation(session):
    return _prepare_runtime_publication(
        session.id,
        RuntimePublicationRequest(
            publication_id="publication",
            kind="model-step",
            interaction_id="interaction",
            intent={"step": 1},
            mutation=runtime_publication_checkpoint_mutation(None, {"phase": "published"}),
            transcript_messages=(Message.text("assistant", "published answer"),),
            events=(
                Event(
                    type=EventType.MODEL_COMPLETED,
                    session_id=session.id,
                    interaction_id="interaction",
                    payload={"step": 1},
                ),
            ),
        ),
        expected_statuses={SessionStatus.RUNNING},
        expected_run_epoch=session.run_epoch,
        expected_transcript_cursor=0,
    )


async def exercise_publication(owner, caps, session, fault, clean, snapshot):
    publish = bind(owner, "publish_runtime", caps)
    load = bind(owner, "load_receipt_record", caps)
    prepared = preparation(session)
    before = await snapshot()
    fault.enabled = True
    try:
        with pytest.raises(fault.error, match="injected"):
            await publish(prepared)
    finally:
        fault.enabled = False
    clean()
    assert await snapshot() == before
    result = await publish(prepared)
    clean()
    assert not result.replayed and result.receipt.transcript_end_cursor == 1
    committed = await snapshot()
    replay = await publish(prepared)
    assert replay.replayed and replay.receipt == result.receipt
    assert await snapshot() == committed
    record = await load(session.id, prepared.storage_key, prepared.request.publication_id)
    assert record is not None
    with pytest.raises(KeyError):
        await load("missing", prepared.storage_key, prepared.request.publication_id)
    clean()


async def exercise_stage_promotion(owner, caps, session, clean, snapshot):
    from tests.core.test_native_model_stage_ownership import preparation as prepare_stage

    from cayu.sessions import _model_stage_rules as rules
    from cayu.sessions._checkpoint_mutations import RuntimePublicationMutation

    stages = caps["stages"]
    prepared = prepare_stage(session)
    stage = (await stages.prepare(prepared)).stage
    request = RuntimePublicationRequest(
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
    await stages.complete(
        rules._prepare_model_completion_stage_terminal(
            session.id,
            stage_id=stage.stage_id,
            publication=request,
        )
    )

    async def load_stage(session_id, stage_id):
        records = await stages.load_stage_records(
            session_id,
            prepared.preparation_storage_key,
            prepared.terminal_storage_key,
        )
        return rules._reconstruct_model_completion_stage(
            *records,
            session_id=session_id,
            stage_id=stage_id,
            preparation_storage_key=prepared.preparation_storage_key,
            terminal_storage_key=prepared.terminal_storage_key,
        )

    promote = partial(
        stages.promote,
        session_id=session.id,
        stage_id=stage.stage_id,
        preparation_storage_key=prepared.preparation_storage_key,
        terminal_storage_key=prepared.terminal_storage_key,
        expected_run_epoch=session.run_epoch,
        load_stage=load_stage,
        publish=bind(owner, "publish_runtime", caps),
    )
    result = await promote()
    assert not result.replayed
    assert result.receipt.publication_id == stage.logical_step_id
    assert await stages.load_active_stage_records(session.id) == (None, None, None)
    committed = await snapshot()
    assert (await promote()).replayed
    assert await snapshot() == committed
    clean()


async def exercise_checkpoint_publication(owner, caps, session, error, clean, snapshot):
    publish = bind(owner, "publish_checkpoint_events", caps)
    event = Event(type="custom.publication", session_id=session.id, payload={"phase": "checkpoint"})
    arguments = dict(
        checkpoint_transform=lambda _session, checkpoint: {
            **(checkpoint or {}),
            "phase": "checkpoint",
        },
        operation_idempotency_key=None,
        operation_transform=None,
        store_time_operation_transform=None,
        operation_commit_time_guard=None,
        events=[event],
        expected_statuses={SessionStatus.RUNNING},
        expected_run_epoch=session.run_epoch,
        expected_transcript_cursor=0,
    )
    guards = []

    def fail_after_writes():
        guards.append("called")
        raise error("injected checkpoint commit guard")

    before = await snapshot()
    with pytest.raises(error, match="injected"):
        await publish(session.id, **arguments, operation_commit_guard=fail_after_writes)
    assert guards == ["called"]
    clean()
    assert await snapshot() == before
    result = await publish(session.id, **arguments, operation_commit_guard=None)
    assert result.id == session.id and result.run_epoch == session.run_epoch
    assert await caps["read_checkpoint"]() == {"phase": "checkpoint"}
    assert await snapshot() != before
    clean()


@asynccontextmanager
async def sqlite_publication(tmp_path, error):
    from cayu.storage import _sqlite_publication as owner
    from cayu.storage import sqlite as adapter

    async with sqlite_composition(tmp_path, error) as (stages, session, fault, clean, _, lock):
        caps = dict(stages.prepare.keywords)
        caps.update(stages.load_stage_records.keywords)
        connection = caps["load_session"].args[0]
        readers = SimpleNamespace(_connection=connection)
        caps.update(
            stages=stages,
            closure_owners=partial(
                adapter.SQLiteSessionStore._closure_lineage_owners_unlocked, readers
            ),
            first_existing_event_id=adapter._first_existing_event_id,
            store_now=stages.complete.keywords["store_now"],
            publish_budget_identities=adapter._publish_budget_reservation_identities,
            record_terminal_receipts=adapter._record_invocation_terminal_event_receipts,
            touch_activity=adapter._touch_session_activity,
            validate_compaction=partial(
                adapter.SQLiteSessionStore._validate_context_view_compaction_unlocked, readers
            ),
        )

        async def read_checkpoint():
            return await caps["run_read"](lambda _conn: caps["load_checkpoint"](session.id))

        caps["read_checkpoint"] = read_checkpoint

        async def snapshot():
            return await caps["run_read"](
                lambda conn: tuple(
                    tuple(
                        tuple(row) for row in conn.execute(f"SELECT * FROM {table} ORDER BY 1, 2")
                    )
                    for table in TABLES
                )
            )

        yield owner, caps, session, fault, clean, snapshot, lock


def test_sqlite_publication_imports_without_adapters_or_driver():
    assert_imports("sqlite")


def test_sqlite_stage_promotion_composes_real_publication(tmp_path):
    async def run():
        async with sqlite_publication(tmp_path, RuntimeError) as (
            owner,
            caps,
            session,
            _,
            clean,
            snapshot,
            _lock,
        ):
            await exercise_stage_promotion(owner, caps, session, clean, snapshot)

    asyncio.run(run())


@pytest.mark.parametrize("error", [RuntimeError, asyncio.CancelledError])
def test_sqlite_publication_composes_and_rolls_back_failed_writes(tmp_path, error):
    async def run():
        async with sqlite_publication(tmp_path, error) as parts:
            await exercise_publication(*parts[:-1])

    asyncio.run(run())


def test_sqlite_publication_cancellation_retains_transaction_until_commit(tmp_path):
    async def run():
        async with sqlite_publication(tmp_path, RuntimeError) as (
            owner,
            caps,
            session,
            fault,
            clean,
            _,
            lock,
        ):
            prepared = preparation(session)
            publish = bind(owner, "publish_runtime", caps)
            fault.enabled, fault.block = True, True
            task = asyncio.create_task(publish(prepared))
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
            assert (await publish(prepared)).replayed

    asyncio.run(run())


@pytest.mark.parametrize("error", [RuntimeError, asyncio.CancelledError])
def test_sqlite_checkpoint_publication_composes_and_rolls_back_commit_guard(tmp_path, error):
    async def run():
        async with sqlite_publication(tmp_path, error) as (
            owner,
            caps,
            session,
            _,
            clean,
            snapshot,
            _lock,
        ):
            await exercise_checkpoint_publication(owner, caps, session, error, clean, snapshot)

    asyncio.run(run())

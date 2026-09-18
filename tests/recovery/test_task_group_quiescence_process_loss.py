"""Fresh-process barrier recovery never infers settlement from owner death."""

from __future__ import annotations

import asyncio
import sys
from uuid import uuid4

import pytest
from tests.core.test_task_group_quiescence import request as group_request

from cayu import CayuApp, TaskCreate, TaskGraphNode
from cayu.storage.migrations import SchemaMode
from cayu.storage.postgres import PostgresTaskStore
from cayu.storage.sqlite import SQLiteTaskStore

pytestmark = pytest.mark.process

_CHILD = r"""
import asyncio
import sys
from pathlib import Path
from cayu import CayuApp, TaskGroupEventType, TaskGroupQuiescenceStatus
from cayu.storage.sqlite import SQLiteTaskStore
from cayu.storage.postgres import PostgresTaskStore
from cayu.tasks.base import TaskQuery, TaskStatus
from cayu.tasks.worker import run_task_worker, complete_managed_task

async def main():
    backend, address, phase, mode, effects = sys.argv[1:]
    store = SQLiteTaskStore(address) if backend == "sqlite" else PostgresTaskStore(address)
    app = CayuApp(task_store=store, enable_logging=False)
    if phase == "verified_callback":
        from cayu import AgentSpec
        from cayu.runtime.verified_task_worker import VerifiedTaskWorker
        from cayu.storage.sqlite import SQLiteSessionStore
        from cayu.storage.postgres import PostgresSessionStore
        from tests.core.test_verified_task_worker import _StaticHandler, _RecordingProvider
        sessions = SQLiteSessionStore(address) if backend == "sqlite" else PostgresSessionStore(address)
        app = CayuApp(session_store=sessions, task_store=store, enable_logging=False)
        provider = _RecordingProvider()
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="worker", model="verified-work-test-model"))
        entered = asyncio.Event()
        class Handler(_StaticHandler):
            async def propose(self, context):
                entered.set()
                await asyncio.Future()
        handler = Handler()
    async def finalizer(_app, task, worker):
        with Path(effects).open("a") as output:
            output.write("published\n")
        await complete_managed_task(store, task, worker, {"published": True})
    try:
        if mode == "crash":
            if phase == "verified_callback":
                owner = VerifiedTaskWorker(
                    app, handler, worker_id="lost", query=TaskQuery(type="b"),
                    lease_seconds=5, callback_timeout_seconds=4,
                )
                work = asyncio.create_task(owner.run(max_tasks=1))
                await asyncio.wait_for(entered.wait(), 15)
            if phase == "draining":
                entered = asyncio.Event()
                async def loser(_app, task, worker):
                    entered.set()
                    await asyncio.Future()
                work = asyncio.create_task(run_task_worker(
                    app, store, loser, worker_id="lost", query=TaskQuery(type="b"),
                    lease_seconds=1, max_tasks=1,
                ))
                await asyncio.wait_for(entered.wait(), 5)
            await store.complete_task("a", {})
            if phase == "finalizer":
                assert await run_task_worker(
                    app, store, finalizer, worker_id="publisher", query=TaskQuery(type="finalize"),
                    max_tasks=1,
                ) == 1
            print("boundary", flush=True)
            await asyncio.Future()
        else:
            before = await app.load_task_group("race")
            assert before.decision.successful_task_ids == ("a",)
            if phase in {"draining", "verified_callback"}:
                # Expiry after process death cannot stand in for effect proof.
                await asyncio.sleep(5.05 if phase == "verified_callback" else 1.05)
                if phase == "verified_callback":
                    from cayu.tasks.admission import WorkAttemptAdmissionConflict, WorkAttemptRecoveryRequest
                    admission = await store.load_latest_work_attempt_admission("b")
                    task_before = await store.load_task("b")
                    events_before = await app.list_task_group_events("race")
                    # Direct public recovery must not bypass the worker's loser
                    # fence, even after the exact session released before SIGKILL.
                    try:
                        await app.recover_work_attempt(WorkAttemptRecoveryRequest(
                            admission_id=admission.admission_id,
                            claim_id="forbidden-replacement", worker_id="replacement",
                            generation=admission.claim.generation + 1, lease_seconds=30,
                        ))
                    except WorkAttemptAdmissionConflict as error:
                        assert "Task-group cancellation" in str(error)
                    else:
                        raise AssertionError("Losing execution acquired replacement ownership")
                    assert await store.load_latest_work_attempt_admission("b") == admission
                    assert await store.load_work_attempt_execution_claim(admission.claim.claim_id) == admission.claim
                    assert await store.load_work_attempt_execution_claim("forbidden-replacement") is None
                    assert await store.load_task("b") == task_before
                    assert await app.load_task_group("race") == before
                    assert await app.list_task_group_events("race") == events_before
                    stop = asyncio.Event()
                    class RecoveryWorker(VerifiedTaskWorker):
                        async def _step(self, now, handled):
                            result = await super()._step(now, handled)
                            stop.set()
                            return result
                    async with RecoveryWorker(
                        app, handler, worker_id="replacement", query=TaskQuery(type="b")
                    ) as owner:
                        assert await owner.run(stop=stop, max_tasks=1) == 0
                    assert not entered.is_set()
                    assert not handler.preparations and not handler.proposals
                    assert not provider.requests
                after = await app.reconcile_task_group("race")
                assert after.quiescence.status is TaskGroupQuiescenceStatus.ATTENTION_REQUIRED
                assert after.quiescence.deadline == before.quiescence.deadline
                assert after.quiescence.unsettled_task_ids == ("b",)
                assert await store.claim_task("publisher", TaskQuery(type="finalize")) is None
                assert not Path(effects).exists()
            else:
                assert before.quiescence.status is TaskGroupQuiescenceStatus.QUIESCENT
                if (await store.load_task("finalize")).status is not TaskStatus.COMPLETED:
                    assert await run_task_worker(
                        app, store, finalizer, worker_id="publisher", query=TaskQuery(type="finalize"),
                        max_tasks=1,
                    ) == 1
                assert (await store.load_task("finalize")).status is TaskStatus.COMPLETED
                assert await store.claim_task("replacement", TaskQuery(type="finalize")) is None
                assert Path(effects).read_text() == "published\n"
                events = await app.list_task_group_events("race")
                assert sum(e.type is TaskGroupEventType.FINALIZER_RELEASED for e in events) == 1
                assert sum(e.type is TaskGroupEventType.FINALIZER_SETTLED for e in events) == 1
                await app.reconcile_task_group("race")
                assert await app.list_task_group_events("race") == events
            print("recovered", flush=True)
    finally:
        await store.close()
        if phase == "verified_callback":
            await sessions.close()

asyncio.run(main())
"""


@pytest.mark.parametrize("backend", ["sqlite", "postgres"])
@pytest.mark.parametrize("phase", ["decision", "draining", "finalizer", "verified_callback"])
def test_group_barrier_and_finalizer_survive_process_loss(backend, phase, tmp_path, request):
    server = None if backend == "sqlite" else request.getfixturevalue("_postgres_server_dsn")

    async def exercise(address):
        store = (
            SQLiteTaskStore(address)
            if backend == "sqlite"
            else PostgresTaskStore(address, schema_mode=SchemaMode.CREATE)
        )
        try:
            creation = group_request(
                timeout=0.05 if phase in {"draining", "verified_callback"} else 60
            )
            if phase == "verified_callback":
                from tests.core.test_verified_task_worker import _contract

                contract = await store.publish_work_contract(_contract())
                creation = creation.model_copy(
                    update={
                        "graph": creation.graph.model_copy(
                            update={
                                "nodes": tuple(
                                    TaskGraphNode(
                                        task=TaskCreate(
                                            task_id=node.task.task_id,
                                            type=node.task.type,
                                            work_contract=contract.reference()
                                            if node.task.task_id == "b"
                                            else None,
                                        )
                                    )
                                    for node in creation.graph.nodes
                                ),
                            }
                        )
                    }
                )
            await CayuApp(task_store=store, enable_logging=False).create_task_group(creation)
        finally:
            await store.close()

        async def spawn(mode):
            return await asyncio.create_subprocess_exec(
                sys.executable,
                "-c",
                _CHILD,
                backend,
                address,
                phase,
                mode,
                str(tmp_path / "effects"),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

        child = await spawn("crash")
        try:
            assert child.stdout is not None
            line = await asyncio.wait_for(child.stdout.readline(), 30)
            if line != b"boundary\n":
                _, stderr = await asyncio.wait_for(child.communicate(), 5)
                pytest.fail(stderr.decode())
            child.kill()
            await asyncio.wait_for(child.wait(), 10)
            assert child.returncode < 0
        finally:
            if child.returncode is None:
                child.kill()
                await asyncio.wait_for(child.wait(), 10)
        for _ in range(2):
            recovery = await spawn("recover")
            try:
                stdout, stderr = await asyncio.wait_for(recovery.communicate(), 30)
                assert recovery.returncode == 0, stderr.decode()
                assert stdout == b"recovered\n"
            finally:
                if recovery.returncode is None:
                    recovery.kill()
                    await asyncio.wait_for(recovery.wait(), 10)

    async def run():
        if backend == "sqlite":
            await exercise(str(tmp_path / "barrier.sqlite"))
            return
        import psycopg
        from psycopg import sql
        from psycopg.conninfo import make_conninfo

        database = "cayu_group_process_" + uuid4().hex
        connection = await psycopg.AsyncConnection.connect(server, autocommit=True)
        await connection.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(database)))
        try:
            await exercise(make_conninfo(server, dbname=database))
        finally:
            await connection.execute(
                sql.SQL("DROP DATABASE {} WITH (FORCE)").format(sql.Identifier(database))
            )
            await connection.close()

    asyncio.run(run())

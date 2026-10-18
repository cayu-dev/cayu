"""Shutdown retries retain ownership until the previous handler is quiescent."""

from __future__ import annotations

import asyncio
from datetime import timedelta

import pytest

from cayu import CayuApp
from cayu.storage.sqlite import SQLiteTaskStore
from cayu.tasks import (
    InMemoryTaskStore,
    TaskCreate,
    TaskStatus,
    complete_managed_task,
    requeue_restartable_task,
    run_task_worker,
)
from cayu.tasks.records import TaskClaimLost
from cayu.tasks.terminalization import TaskTerminalizationConflict


@pytest.mark.store_semantics
@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("operator_cancel", [False, True])
def test_shutdown_waits_for_cleanup_then_requeues_or_respects_operator_cancel(
    backend, operator_cancel, tmp_path
):
    async def scenario():
        store = (
            InMemoryTaskStore() if backend == "memory" else SQLiteTaskStore(tmp_path / "tasks.db")
        )
        app = CayuApp(task_store=store, enable_logging=False)
        created = await store.create_task(
            TaskCreate(type="job", restartable=True, concurrency_key="restart-job")
        )
        await store.create_task(TaskCreate(type="peer", concurrency_key="restart-job"))
        ready = asyncio.Event()
        cleaning = asyncio.Event()
        release = asyncio.Event()

        async def handler(_app, _task, _worker):
            ready.set()
            try:
                await asyncio.Event().wait()
            finally:
                cleaning.set()
                await release.wait()

        worker = asyncio.create_task(
            run_task_worker(
                app,
                store,
                handler,
                worker_id="first",
                max_tasks=1,
                lease_seconds=3,
                poll_interval_s=0.01,
            )
        )
        try:
            await asyncio.wait_for(ready.wait(), 10)
            before = await store.load_task(created.id)
            worker.cancel()
            await asyncio.wait_for(cleaning.wait(), 10)
            assert not worker.done()
            assert await store.claim_task("competitor") is None
            async with asyncio.timeout(10):
                while (
                    await store.load_task(created.id)
                ).lease_expires_at == before.lease_expires_at:
                    await asyncio.sleep(0.01)
            current = await store.load_task(created.id)
            if operator_cancel:
                await store.request_claimed_task_cancellation(
                    current.id,
                    "first",
                    current.lease_expires_at,
                )
                # Cancellation during shutdown must retain the resource through
                # cleanup longer than the lease observed by the operator.
                await asyncio.sleep(3.1)
                from datetime import UTC, datetime

                assert (await store.load_task(created.id)).lease_expires_at > datetime.now(UTC)
                assert await store.claim_task("same-key-competitor") is None
            worker.cancel()  # repeated shutdown signals must not interrupt cleanup
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(worker, 10)
            current = await store.load_task(created.id)
            assert current.restartable is True
            if operator_cancel:
                assert current.status is TaskStatus.CANCELLED
                assert current.restart_count == 0
                return
            assert current.status is TaskStatus.PENDING
            assert current.restart_count == 1
            assert current.worker_id is None and current.started_at is None
            with pytest.raises(TaskClaimLost, match="exact worker claim"):
                await store.start_task(current.id)
            if backend == "sqlite":
                await store.close()
                store = SQLiteTaskStore(tmp_path / "tasks.db")
                app = CayuApp(task_store=store, enable_logging=False)

            async def next_handler(_app, task, worker_id):
                assert task.restart_count == 1
                await complete_managed_task(store, task, worker_id, {"ok": True})

            assert (
                await asyncio.wait_for(
                    run_task_worker(
                        app,
                        store,
                        next_handler,
                        worker_id="next",
                        max_tasks=1,
                    ),
                    10,
                )
                == 1
            )
            assert (await store.load_task(created.id)).status is TaskStatus.COMPLETED
        finally:
            release.set()
            if not worker.done():
                worker.cancel()
            await asyncio.gather(worker, return_exceptions=True)
            if backend == "sqlite":
                await store.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_standalone_requeue_requires_opt_in_and_exact_live_uncancelled_claim(backend, tmp_path):
    async def scenario():
        store = (
            InMemoryTaskStore() if backend == "memory" else SQLiteTaskStore(tmp_path / "tasks.db")
        )
        for restartable in (False, True):
            created = await store.create_task(TaskCreate(type="job", restartable=restartable))
            claimed = await store.claim_task("worker")
            assert claimed.id == created.id
            if not restartable:
                with pytest.raises(ValueError, match="restartable"):
                    await requeue_restartable_task(store, claimed, "worker")
                await complete_managed_task(store, claimed, "worker", {})
                continue
            with pytest.raises(TaskClaimLost):
                await store.restart_task(
                    claimed.id,
                    "worker",
                    lease_expires_at=claimed.lease_expires_at + timedelta(seconds=1),
                )
            await store.request_claimed_task_cancellation(
                claimed.id, "worker", claimed.lease_expires_at
            )
            with pytest.raises(TaskTerminalizationConflict):
                await requeue_restartable_task(store, claimed, "worker")
            assert (await store.load_task(claimed.id)).restart_count == 0
        if backend == "sqlite":
            await store.close()

    asyncio.run(scenario())


@pytest.mark.store_semantics
def test_sigterm_requeues_attached_sqlite_task_for_a_second_process(tmp_path):
    import subprocess
    import sys
    import time
    from pathlib import Path

    database = tmp_path / "durable.db"
    ready = tmp_path / "ready"
    script = Path(__file__).with_name("task_restart_process.py")

    async def create():
        store = SQLiteTaskStore(database)
        task = await store.create_task(TaskCreate(type="job", restartable=True))
        await store.close()
        return task.id

    task_id = asyncio.run(create())
    first = subprocess.Popen(
        [sys.executable, str(script), str(database), str(ready), "interrupt"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        until = time.monotonic() + 15
        while not ready.exists() and first.poll() is None and time.monotonic() < until:
            time.sleep(0.01)
        assert ready.exists(), "first worker did not reach its attached provider call"
        first.terminate()
        stdout, stderr = first.communicate(timeout=15)
        assert first.returncode == 0, stdout + stderr

        async def inspect_pending():
            store = SQLiteTaskStore(database)
            task = await store.load_task(task_id)
            await store.close()
            assert task.status is TaskStatus.PENDING
            assert task.restart_count == 1
            assert task.session_id is None
            assert task.status_payload["restart"]["previous_session_id"]

        asyncio.run(inspect_pending())
        second = subprocess.run(
            [sys.executable, str(script), str(database), str(ready), "complete"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        assert second.returncode == 0, second.stdout + second.stderr

        async def inspect_completed():
            store = SQLiteTaskStore(database)
            task = await store.load_task(task_id)
            await store.close()
            assert task.status is TaskStatus.COMPLETED
            assert task.result == {"restarted": True}

        asyncio.run(inspect_completed())
    finally:
        if first.poll() is None:
            first.kill()
            first.communicate(timeout=10)


def test_postgres_restart_preserves_policy_and_rejects_stale_owner(postgres_dsn):
    from tests.core.task_invocation_fixtures import task_backed_session_invocation

    from cayu.storage.migrations import SchemaMode
    from cayu.storage.tasks_postgres import PostgresTaskStore

    async def scenario():
        store = PostgresTaskStore(postgres_dsn, schema_mode=SchemaMode.CREATE)
        try:
            created = await store.create_task(TaskCreate(type="restart-test", restartable=True))
            first = await store.claim_task("first")
            first = await store.attach_task(
                first.id,
                session_id="retired-postgres-session",
                session_invocation=await task_backed_session_invocation(
                    store, first.id, "retired-postgres-session"
                ),
                worker_id="first",
                lease_expires_at=first.lease_expires_at,
            )
            requeued = await requeue_restartable_task(store, first, "first")
            assert requeued.id == created.id and requeued.restart_count == 1
            with pytest.raises(TaskClaimLost, match="retired task session"):
                await store.admit_ordinary_session_execution("retired-postgres-session")
            await store.admit_ordinary_session_cleanup("retired-postgres-session")
            second = await store.claim_task("second")
            assert second.restartable and second.restart_count == 1
            with pytest.raises(TaskClaimLost):
                await requeue_restartable_task(store, first, "first")
            await complete_managed_task(store, second, "second", {})
        finally:
            await store.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("revision", [119, 120, 121])
def test_sqlite_migration_adds_disabled_restart_policy_to_existing_jobs(tmp_path, revision):
    import sqlite3

    from cayu.storage.migrations import SchemaMode, SchemaTooOld

    async def scenario():
        path = tmp_path / "upgrade.db"
        original = SQLiteTaskStore(path)
        created = await original.create_task(TaskCreate(type="existing"))
        await original.close()
        with sqlite3.connect(path) as connection:
            if revision < 121:
                connection.execute("ALTER TABLE cayu_tasks DROP COLUMN restartable")
                connection.execute("ALTER TABLE cayu_tasks DROP COLUMN restart_count")
            connection.execute("DELETE FROM cayu_schema_migrations WHERE revision > ?", (revision,))
            connection.execute("DROP TABLE cayu_retired_task_sessions")
            connection.execute(f"PRAGMA user_version = {revision}")
        with pytest.raises(SchemaTooOld):
            SQLiteTaskStore(path, schema_mode=SchemaMode.VALIDATE)
        upgraded = SQLiteTaskStore(path, schema_mode=SchemaMode.MIGRATE)
        try:
            assert await upgraded.load_task(created.id) == created
            new = await upgraded.create_task(TaskCreate(type="new", restartable=True))
            assert new.restartable and new.restart_count == 0
            # This lookup requires the table missing from the old revision 121.
            await upgraded.admit_ordinary_session_execution("new-session")
        finally:
            await upgraded.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("revision", [119, 120, 121])
def test_restart_migration_preserves_legacy_schedule_and_graph_replays(tmp_path, revision):
    import json
    import sqlite3
    from datetime import UTC, datetime

    from cayu.storage.migrations import SchemaMode
    from cayu.tasks.graphs import TaskGraphCreate, TaskGraphNode
    from cayu.tasks.scheduling import TaskSchedulePolicy

    async def scenario():
        path = tmp_path / "legacy-replays.db"
        schedule = TaskCreate(
            task_id="legacy-schedule",
            type="legacy",
            available_at=datetime(2030, 1, 1, tzinfo=UTC),
            schedule_policy=TaskSchedulePolicy(),
        )
        graph = TaskGraphCreate(
            graph_id="legacy-graph",
            nodes=(TaskGraphNode(task=TaskCreate(task_id="legacy-graph-task", type="legacy")),),
        )
        original = SQLiteTaskStore(path)
        await original.create_task(schedule)
        receipt = await original.create_task_graph(graph)
        await original.close()

        # Frozen creation identities produced by the revision-119 implementation
        # for these requests, before TaskCreate had a restartable field.
        schedule_digest = "8899154e285bae5db3cc2913a944e9800052e7bb437c8c80eff53770262a9b5a"
        graph_digest = "d2a0ca3613c969e3921d4a06d872522f516ddd8ce32a6115acb2944397b91f23"
        legacy_receipt = receipt.model_copy(
            update={"request_sha256": graph_digest, "submitted_request_sha256": graph_digest}
        )
        with sqlite3.connect(path) as connection:
            connection.execute(
                "UPDATE cayu_tasks SET schedule_json = "
                "json_set(schedule_json, '$.creation_sha256', ?) WHERE id = ?",
                (schedule_digest, schedule.task_id),
            )
            connection.execute(
                "UPDATE cayu_task_graphs SET receipt_json = ? WHERE graph_id = ?",
                (json.dumps(legacy_receipt.model_dump(mode="json")), graph.graph_id),
            )
            if revision < 121:
                connection.execute("ALTER TABLE cayu_tasks DROP COLUMN restartable")
                connection.execute("ALTER TABLE cayu_tasks DROP COLUMN restart_count")
            connection.execute("DROP TABLE cayu_retired_task_sessions")
            connection.execute("DELETE FROM cayu_schema_migrations WHERE revision > ?", (revision,))
            connection.execute(f"PRAGMA user_version = {revision}")

        upgraded = SQLiteTaskStore(path, schema_mode=SchemaMode.MIGRATE)
        try:
            retained = await upgraded.load_task(schedule.task_id)
            assert retained.schedule.creation_sha256 == schedule_digest
            assert await upgraded.create_task(schedule) == retained
            assert await upgraded.create_task_graph(graph) == legacy_receipt
        finally:
            await upgraded.close()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "fields",
    [
        {"session_id": "existing"},
        {"parent_task_id": "parent"},
        {"retry_policy": {"max_attempts": 2}},
    ],
)
def test_restart_policy_rejects_other_task_lifecycles(fields):
    with pytest.raises(ValueError, match="ordinary unattached root"):
        TaskCreate(type="job", restartable=True, **fields)


@pytest.mark.store_semantics
@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("fault", ["cleanup", "operator_race"])
def test_shutdown_settles_even_when_cleanup_fails_or_operator_wins(backend, fault, tmp_path):
    from cayu._exception_groups import exception_cause, iter_exception_tree

    async def scenario():
        base = InMemoryTaskStore if backend == "memory" else SQLiteTaskStore

        class RacingStore(base):
            verified_work_mutations_are_cancellation_quiescent = True

            async def restart_task(self, task_id, worker_id, *, lease_expires_at):
                if fault == "operator_race":
                    await self.request_claimed_task_cancellation(
                        task_id, worker_id, lease_expires_at
                    )
                return await super().restart_task(
                    task_id, worker_id, lease_expires_at=lease_expires_at
                )

        store = RacingStore() if backend == "memory" else RacingStore(tmp_path / "fault.db")
        app = CayuApp(task_store=store, enable_logging=False)
        created = await store.create_task(TaskCreate(type="job", restartable=True))
        ready = asyncio.Event()

        async def handler(*args):
            ready.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                if fault == "cleanup":
                    raise RuntimeError("handler cleanup failed") from None
                raise

        worker = asyncio.create_task(run_task_worker(app, store, handler, worker_id="first"))
        try:
            await asyncio.wait_for(ready.wait(), 10)
            worker.cancel()
            with pytest.raises(asyncio.CancelledError) as caught:
                await asyncio.wait_for(worker, 10)
            current = await store.load_task(created.id)
            assert current.worker_id is None and current.lease_expires_at is None
            if fault == "operator_race":
                assert current.status is TaskStatus.CANCELLED
                assert current.restart_count == 0
            else:
                assert current.status is TaskStatus.PENDING
                assert current.restart_count == 1
                assert any(
                    isinstance(error, RuntimeError) and str(error) == "handler cleanup failed"
                    for error in iter_exception_tree(exception_cause(caught.value))
                )
        finally:
            if not worker.done():
                worker.cancel()
            await asyncio.gather(worker, return_exceptions=True)
            if backend == "sqlite":
                await store.close()

    asyncio.run(scenario())


@pytest.mark.store_semantics
@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("cleanup_failure", [False, True])
def test_retired_session_cannot_resume_after_new_worker_claims(backend, cleanup_failure, tmp_path):
    from cayu import AgentSpec, Message, ResumeRequest, RunRequest
    from cayu.evals import ScriptedModelProvider
    from cayu.providers import ModelStreamEvent
    from cayu.sessions import InMemorySessionStore
    from cayu.storage import SQLiteSessionStore

    async def scenario():
        store = InMemoryTaskStore() if backend == "memory" else SQLiteTaskStore(tmp_path / "old.db")
        from cayu.sessions import SessionStatus
        from cayu.sessions.recovery import IncompleteSessionRecoveryRequest

        base = InMemorySessionStore if backend == "memory" else SQLiteSessionStore

        class FaultingSessions(base):
            invocation_lifecycle_command_version = 1
            terminal_interaction_publication_version = 1
            fail_cleanup = False

            async def publish_interaction_transition(self, session_id, **kwargs):
                if self.fail_cleanup:
                    raise RuntimeError("terminal evidence unavailable")
                return await super().publish_interaction_transition(session_id, **kwargs)

            async def transition_status(self, session_id, *, from_statuses, to_status):
                if self.fail_cleanup:
                    raise RuntimeError("terminal evidence unavailable")
                return await super().transition_status(
                    session_id, from_statuses=from_statuses, to_status=to_status
                )

        sessions = (
            FaultingSessions() if backend == "memory" else FaultingSessions(tmp_path / "old.db")
        )
        ready = asyncio.Event()

        class WaitingProvider(ScriptedModelProvider):
            async def stream(self, request):
                self.requests.append(request)
                ready.set()
                await asyncio.Event().wait()
                yield ModelStreamEvent.completed()

        provider = WaitingProvider([])
        app = CayuApp(task_store=store, session_store=sessions, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="review", model="scripted"))
        if cleanup_failure:
            from cayu.events import EventType

            emit = app._event_writer.emit

            async def stall_before_provider(event):
                recorded = await emit(event)
                if event.type is EventType.TASK_STARTED:
                    ready.set()
                    await asyncio.Event().wait()
                return recorded

            app._event_writer.emit = stall_before_provider
        task = await store.create_task(TaskCreate(type="job", restartable=True))

        async def handler(app, task, worker_id):
            async for _ in app.run(
                RunRequest(
                    agent_name="review",
                    session_id="old-session",
                    task_id=task.id,
                    task_worker_id=worker_id,
                    task_lease_expires_at=task.lease_expires_at,
                    messages=[Message.text("user", "review")],
                )
            ):
                pass

        worker = asyncio.create_task(run_task_worker(app, store, handler, worker_id="first"))
        try:
            try:
                await asyncio.wait_for(ready.wait(), 5)
            except TimeoutError:
                raise AssertionError(await store.load_task(task.id)) from None
            sessions.fail_cleanup = cleanup_failure
            worker.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(worker, 10)
            assert (await store.load_task(task.id)).status is TaskStatus.PENDING
            if cleanup_failure:
                assert (await sessions.load("old-session")).status is SessionStatus.RUNNING
            sessions.fail_cleanup = False
            recovered = await app.recover_incomplete_session(
                IncompleteSessionRecoveryRequest(session_id="old-session", inactive_for_seconds=0)
            )
            assert recovered.status is SessionStatus.INTERRUPTED
            if backend == "sqlite":
                await store.close()
                store = SQLiteTaskStore(tmp_path / "old.db")
                app = CayuApp(task_store=store, session_store=sessions, enable_logging=False)
                app.register_provider(provider, default=True)
                app.register_agent(AgentSpec(name="review", model="scripted"))
            claimed = await store.claim_task("second")
            with pytest.raises(TaskClaimLost, match="retired task session"):
                async for _ in app.resume(
                    ResumeRequest(
                        session_id="old-session", messages=[Message.text("user", "continue")]
                    )
                ):
                    pass
            assert len(provider.requests) == (0 if cleanup_failure else 1)
            assert (await store.load_task(task.id)).worker_id == "second"
            await complete_managed_task(store, claimed, "second", {})
        finally:
            if not worker.done():
                worker.cancel()
            await asyncio.gather(worker, return_exceptions=True)
            if backend == "sqlite":
                await store.close()
                await sessions.close()

    asyncio.run(scenario())


def test_legacy_custom_task_store_keeps_ordinary_cleanup_admission():
    from tests.core.test_recovery_plans import _app, _create_running_session

    from cayu.runtime._task_store_operation_boundary import (
        task_store_mutation_is_cancellation_quiescent,
    )
    from cayu.sessions.memory import InMemorySessionStore
    from cayu.sessions.records import SessionStatus
    from cayu.sessions.recovery import IncompleteSessionRecoveryRequest
    from cayu.tasks.store import TaskStore

    # Model a pre-restart-feature implementation: all original abstract methods
    # delegate to its backend, while the new cleanup method remains inherited.
    delegate = InMemoryTaskStore()

    def forwarding(name):
        async def method(self, *args, **kwargs):
            return await getattr(delegate, name)(*args, **kwargs)

        return method

    legacy_type = type(
        "LegacyTaskStore",
        (TaskStore,),
        {
            "verified_work_mutations_are_cancellation_quiescent": True,
            **{name: forwarding(name) for name in TaskStore.__abstractmethods__},
            "admit_ordinary_session_execution": forwarding("admit_ordinary_session_execution"),
        },
    )
    tasks = legacy_type()
    assert task_store_mutation_is_cancellation_quiescent(tasks, "admit_ordinary_session_execution")
    assert not task_store_mutation_is_cancellation_quiescent(
        tasks, "admit_ordinary_session_cleanup"
    )

    async def scenario():
        sessions = InMemorySessionStore()
        app = _app(sessions, task_store=tasks)
        await _create_running_session(sessions, app, "legacy-store-orphan")
        recovered = await app.recover_incomplete_session(
            IncompleteSessionRecoveryRequest(
                session_id="legacy-store-orphan", inactive_for_seconds=0
            )
        )
        assert recovered.status is SessionStatus.INTERRUPTED
        await app.aclose()

    asyncio.run(scenario())

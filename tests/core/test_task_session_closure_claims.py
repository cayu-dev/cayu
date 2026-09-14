"""Task-store closure admission binds an exact non-renewable task namespace."""

import asyncio
import sqlite3

import pytest
from tests.core.session_closure_conformance import create_closure_session
from tests.core.task_invocation_fixtures import task_backed_session_invocation

from cayu import CayuApp
from cayu.runtime.session_closure import SessionClosureDisposition, SessionClosureRecord
from cayu.sessions.base import InMemorySessionStore
from cayu.storage.migrations import SchemaMode
from cayu.storage.sqlite import SQLiteSessionStore, SQLiteTaskStore
from cayu.tasks.base import InMemoryTaskStore, TaskCreate, TaskQuery, TaskSessionClosureClaim


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
def test_public_closure_retains_task_claim_after_caller_cancellation(tmp_path, request, backend):
    if backend == "postgres":
        from psycopg.errors import CheckViolation

        from cayu.storage.postgres import PostgresSessionStore, PostgresTaskStore

        dsn = request.getfixturevalue("postgres_dsn")
        guard_error = CheckViolation
    else:
        guard_error = ValueError if backend == "memory" else sqlite3.IntegrityError

    async def run():
        entered = asyncio.Event()
        proceed = asyncio.Event()

        class PausedDependent:
            store_id = "paused-dependent"

            async def inspect_session_closure(self, *args, **kwargs):
                return SessionClosureRecord(
                    store_id=self.store_id,
                    record_class="test",
                    disposition=SessionClosureDisposition.OWNED_ELIGIBLE,
                    count=1,
                )

            async def erase_session_closure(self, *args, **kwargs):
                entered.set()
                await proceed.wait()
                return SessionClosureRecord(
                    store_id=self.store_id,
                    record_class="test",
                    disposition=SessionClosureDisposition.ERASED,
                    count=1,
                )

        if backend == "memory":
            sessions, tasks = InMemorySessionStore(), InMemoryTaskStore()
            competitor = tasks
        elif backend == "sqlite":
            sessions = SQLiteSessionStore(tmp_path / "public.sqlite")
            tasks = SQLiteTaskStore(tmp_path / "public.sqlite")
            competitor = SQLiteTaskStore(tmp_path / "public.sqlite")
        else:
            sessions = PostgresSessionStore(dsn, schema_mode=SchemaMode.CREATE)
            tasks = PostgresTaskStore(dsn, schema_mode=SchemaMode.CREATE)
            competitor = PostgresTaskStore(dsn, schema_mode=SchemaMode.CREATE)
        app = CayuApp(
            session_store=sessions, task_store=tasks, session_closure_stores=(PausedDependent(),)
        )
        closing = None
        try:
            await create_closure_session(sessions, "public-root")
            await tasks.create_task(
                TaskCreate(type="test", task_id="public-original", session_id="public-root")
            )
            await tasks.complete_task("public-original", {})
            await competitor.create_task(TaskCreate(type="test", task_id="unattached"))
            pending = await competitor.claim_task("worker")
            assert pending is not None and pending.id == "unattached"
            binding = await task_backed_session_invocation(competitor, pending.id, "public-root")
            closing = asyncio.create_task(app.erase_session_closure("public-root"))
            await asyncio.wait_for(entered.wait(), 10)
            claim = await competitor.load_session_closure_claim("public-root")
            assert claim is not None and claim.task_ids == ("public-original",)
            closing.cancel()
            with pytest.raises(asyncio.CancelledError):
                await closing
            assert closing.cancelled() and closing.cancelling() == 1
            with pytest.raises(guard_error, match="owned by closure"):
                await competitor.create_task(TaskCreate(type="test", session_id="public-root"))
            with pytest.raises(guard_error, match="owned by closure"):
                await competitor.attach_task(
                    pending.id,
                    session_id="public-root",
                    session_invocation=binding,
                    worker_id="worker",
                    lease_expires_at=pending.lease_expires_at,
                )
            assert await competitor.load_task(pending.id) == pending
            proceed.set()
            report = await app.erase_session_closure("public-root")
            assert report.complete
            record = next(
                record
                for record in report.manifest.records
                if record.record_class == "session_tasks"
            )
            assert record.count == 1 and record.disposition is SessionClosureDisposition.ERASED
            assert await sessions.load("public-root") is None
            assert await competitor.load_session_closure_claim("public-root") == claim
        finally:
            proceed.set()
            if closing is not None and not closing.done():
                closing.cancel()
                await asyncio.gather(closing, return_exceptions=True)
            if backend != "memory":
                await competitor.close()
                await tasks.close()
                await sessions.close()

    asyncio.run(run())


@pytest.mark.parametrize("backend", ["sqlite", "postgres"])
def test_persistent_task_closure_claim_survives_reopen(tmp_path, request, backend):
    if backend == "sqlite":

        def create():
            return SQLiteTaskStore(tmp_path / "closure.sqlite")

        guard_error = sqlite3.IntegrityError
    else:
        from psycopg.errors import CheckViolation

        from cayu.storage.postgres import PostgresTaskStore

        dsn = request.getfixturevalue("postgres_dsn")

        def create():
            return PostgresTaskStore(dsn, schema_mode=SchemaMode.CREATE)

        guard_error = CheckViolation

    async def run():
        store = create()
        claim = TaskSessionClosureClaim(session_id="session", plan_id="a" * 64, task_ids=("one",))
        try:
            await store.create_task(TaskCreate(type="test", task_id="one", session_id="session"))
            with pytest.raises(ValueError, match="changed"):
                await store.claim_session_closure(claim.model_copy(update={"task_ids": ()}))
            with pytest.raises(ValueError, match="quiescent"):
                await store.claim_session_closure(claim)
            await store.complete_task("one", {})
            assert await store.claim_session_closure(claim) == claim
        finally:
            await store.close()

        store = create()
        competitor = create()
        try:
            assert await store.claim_session_closure(claim) == claim
            with pytest.raises(ValueError, match="conflicts"):
                await competitor.claim_session_closure(
                    claim.model_copy(update={"plan_id": "b" * 64})
                )
            with pytest.raises(guard_error, match="owned by closure"):
                await competitor.create_task(TaskCreate(type="test", session_id="session"))
            assert len(await store.list_tasks(TaskQuery(session_id="session"))) == 1
            with pytest.raises(ValueError, match="conflicts"):
                await store.delete_session_tasks("session", task_ids=(), policy=None)
            await store.delete_session_tasks("session", task_ids=claim.task_ids, policy=None)
            await store.delete_session_tasks("session", task_ids=claim.task_ids, policy=None)
            assert await competitor.claim_session_closure(claim) == claim
            with pytest.raises(guard_error, match="owned by closure"):
                await competitor.create_task(TaskCreate(type="test", session_id="session"))
            assert await store.list_tasks(TaskQuery(session_id="session")) == []
            assert (
                await competitor.create_task(TaskCreate(type="test", session_id="other"))
            ).session_id == "other"
        finally:
            await competitor.close()
            await store.close()

    asyncio.run(run())


def test_memory_task_closure_claim_replays_and_fences_new_work():
    async def run():
        store = InMemoryTaskStore()
        for task_id in ("first", "second"):
            await store.create_task(TaskCreate(type="test", task_id=task_id, session_id="session"))
            await store.complete_task(task_id, {})
        claim = TaskSessionClosureClaim(
            session_id="session", plan_id="a" * 64, task_ids=("second", "first")
        )
        assert await store.claim_session_closure(claim) == claim
        assert await store.claim_session_closure(claim) == claim
        before = await store.list_tasks(TaskQuery(session_id="session"))
        for changed in (
            claim.model_copy(update={"plan_id": "b" * 64}),
            claim.model_copy(update={"task_ids": ("first",)}),
        ):
            with pytest.raises(ValueError, match="conflicts"):
                await store.claim_session_closure(changed)
        with pytest.raises(ValueError, match="owned by closure"):
            await store.create_task(TaskCreate(type="test", task_id="late", session_id="session"))
        assert await store.list_tasks(TaskQuery(session_id="session")) == before
        with pytest.raises(ValueError, match="conflicts"):
            await store.delete_session_tasks("session", task_ids=("first",), policy=None)
        await store.delete_session_tasks("session", task_ids=claim.task_ids, policy=None)
        await store.delete_session_tasks("session", task_ids=claim.task_ids, policy=None)
        assert await store.claim_session_closure(claim) == claim
        with pytest.raises(ValueError, match="owned by closure"):
            await store.create_task(TaskCreate(type="test", task_id="after", session_id="session"))
        assert await store.list_tasks(TaskQuery(session_id="session")) == []
        assert (
            await store.create_task(TaskCreate(type="test", session_id="other"))
        ).session_id == "other"

    asyncio.run(run())


@pytest.mark.parametrize("tasks", [("missing",), ()])
def test_memory_task_closure_claim_rejects_incomplete_inventory_without_retirement(tasks):
    async def run():
        store = InMemoryTaskStore()
        await store.create_task(TaskCreate(type="test", task_id="active", session_id="session"))
        with pytest.raises(ValueError, match="changed"):
            await store.claim_session_closure(
                TaskSessionClosureClaim(session_id="session", plan_id="a" * 64, task_ids=tasks)
            )
        with pytest.raises(ValueError, match="quiescent"):
            await store.claim_session_closure(
                TaskSessionClosureClaim(
                    session_id="session", plan_id="a" * 64, task_ids=("active",)
                )
            )
        await store.complete_task("active", {})
        await store.create_task(TaskCreate(type="test", task_id="allowed", session_id="session"))
        assert len(await store.list_tasks(TaskQuery(session_id="session"))) == 2

    asyncio.run(run())


def test_task_closure_claim_revalidates_mutated_input_without_rendering_it(capsys, caplog, recwarn):
    class Private:
        def __repr__(self):
            return "private-task-closure-canary"

    async def run():
        store = InMemoryTaskStore()
        claim = TaskSessionClosureClaim(session_id="session", plan_id="a" * 64, task_ids=())
        object.__setattr__(claim, "task_ids", (Private(),))
        with pytest.raises(ValueError) as error:
            await store.claim_session_closure(claim)
        assert "private-task-closure-canary" not in str(error.value)
        await store.create_task(TaskCreate(type="test", session_id="session"))

    asyncio.run(run())
    captured = capsys.readouterr()
    assert "private-task-closure-canary" not in captured.out + captured.err + caplog.text
    assert all("private-task-closure-canary" not in str(warning.message) for warning in recwarn)


def test_memory_task_closure_claim_does_not_expose_stored_model():
    async def run():
        store = InMemoryTaskStore()
        claim = TaskSessionClosureClaim(session_id="session", plan_id="a" * 64, task_ids=())
        first = await store.claim_session_closure(claim)
        object.__setattr__(first, "plan_id", "b" * 64)
        replayed = await store.claim_session_closure(claim)
        object.__setattr__(replayed, "task_ids", ("fake",))
        assert await store.load_session_closure_claim("session") == claim

    asyncio.run(run())


@pytest.mark.parametrize("guard", ["insert", "update"])
def test_sqlite_task_closure_requires_database_guards(tmp_path, guard):
    async def run():
        store = SQLiteTaskStore(tmp_path / "guard.sqlite")
        try:
            trigger = {
                "insert": "cayu_task_closure_insert_guard",
                "update": "cayu_task_closure_update_guard",
            }[guard]
            store._connection.execute(f"DROP TRIGGER {trigger}")
            store._schema_mode = SchemaMode.VALIDATE
            with pytest.raises(RuntimeError, match="closure schema"):
                store._initialize_schema()
        finally:
            await store.close()

    asyncio.run(run())


@pytest.mark.parametrize("schema_mode", [SchemaMode.VALIDATE, SchemaMode.MIGRATE])
def test_postgres_task_closure_requires_enabled_database_guard(postgres_dsn, schema_mode):
    import psycopg

    from cayu.storage.postgres import PostgresTaskStore

    async def run():
        store = PostgresTaskStore(postgres_dsn, schema_mode=SchemaMode.CREATE)
        replacement = None
        try:
            await store.ensure_schema()
            replacement = PostgresTaskStore(postgres_dsn, schema_mode=schema_mode)
            await replacement.ensure_schema()
            await replacement.close()
            replacement = None
            async with await psycopg.AsyncConnection.connect(postgres_dsn) as conn:
                await conn.execute(
                    "ALTER TABLE cayu_tasks DISABLE TRIGGER cayu_task_closure_admission_guard"
                )
                before = await (
                    await conn.execute(
                        "SELECT session_id, plan_id, claim_json "
                        "FROM cayu_task_session_closure_claims ORDER BY session_id"
                    )
                ).fetchall()
            replacement = PostgresTaskStore(postgres_dsn, schema_mode=schema_mode)
            with pytest.raises(RuntimeError, match="closure admission guard"):
                await replacement.ensure_schema()
            async with await psycopg.AsyncConnection.connect(postgres_dsn) as conn:
                after = await (
                    await conn.execute(
                        "SELECT session_id, plan_id, claim_json "
                        "FROM cayu_task_session_closure_claims ORDER BY session_id"
                    )
                ).fetchall()
                assert after == before
        finally:
            async with await psycopg.AsyncConnection.connect(postgres_dsn) as conn:
                await conn.execute(
                    "ALTER TABLE cayu_tasks ENABLE TRIGGER cayu_task_closure_admission_guard"
                )
            if replacement is not None:
                await replacement.close()
            await store.close()

    asyncio.run(run())

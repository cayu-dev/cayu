from __future__ import annotations

import asyncio
import sqlite3
from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

import pytest

from cayu import (
    CayuApp,
    InMemorySessionStore,
    InMemoryTaskStore,
    InvocationOriginClaim,
    InvocationOriginTrust,
    RunRequest,
    SessionExecutionSource,
    SessionInvocationBinding,
    SQLiteTaskStore,
    TaskCreate,
    TaskExecutionSource,
    TaskStatus,
    session_invocation_from_task,
    task_create_with_execution_source,
)
from cayu.runtime.sessions import SessionIdentity, run_request_with_task_invocation
from cayu.storage import migrations
from cayu.vaults import SecretRedactor


def _identity() -> SessionIdentity:
    return SessionIdentity(provider_name="fake", model="fake-model")


@pytest.mark.parametrize("store_factory", [InMemoryTaskStore, SQLiteTaskStore])
def test_task_root_and_child_provenance_is_immutable(store_factory, tmp_path) -> None:
    store = (
        SQLiteTaskStore(tmp_path / "task-provenance.sqlite")
        if store_factory is SQLiteTaskStore
        else InMemoryTaskStore()
    )

    async def exercise() -> None:
        root = await store.create_task(
            TaskCreate(
                task_id="root",
                type="workflow",
                invocation_origin=InvocationOriginClaim(
                    subject="application-user",
                    tenant="customer-a",
                ),
            )
        )
        child = await store.create_task(
            TaskCreate(task_id="child", type="step", parent_task_id=root.id)
        )

        assert root.invocation.origin.trust is InvocationOriginTrust.HOST_ASSERTED
        assert root.invocation.origin.subject == "application-user"
        assert root.invocation.origin.tenant == "customer-a"
        assert root.invocation.source is TaskExecutionSource.SDK_TASK
        assert UUID(root.invocation.root_invocation_id).version == 4
        assert child.invocation.origin == root.invocation.origin
        assert child.invocation.root_invocation_id == root.invocation.root_invocation_id
        assert child.invocation.root_session_id is None
        assert child.invocation.source is TaskExecutionSource.SDK_TASK

        claimed = await store.claim_task("worker-a", lease_seconds=300)
        assert claimed is not None
        assert claimed.invocation == root.invocation
        attached = await store.attach_task(
            claimed.id,
            session_id="task-session",
            session_invocation=SessionInvocationBinding(
                id="task-session",
                session_instance_id=str(uuid4()),
                invocation=session_invocation_from_task(
                    claimed.invocation,
                    session_id="task-session",
                ),
            ),
            worker_id="worker-a",
        )
        assert attached.invocation == root.invocation
        completed = await store.complete_task(
            attached.id,
            {"ok": True},
            worker_id="worker-a",
        )
        assert completed.invocation == root.invocation

        if hasattr(store, "close"):
            await store.close()

    asyncio.run(exercise())


def test_task_creation_rejects_missing_or_contradictory_parent_authority() -> None:
    async def exercise() -> None:
        store = InMemoryTaskStore()
        parent = await store.create_task(
            TaskCreate(
                task_id="parent",
                type="workflow",
                invocation_origin=InvocationOriginClaim(subject="root-user"),
            )
        )
        with pytest.raises(ValueError, match="Parent task not found"):
            await store.create_task(
                TaskCreate(task_id="orphan", type="step", parent_task_id="missing")
            )
        with pytest.raises(ValueError, match="must inherit"):
            await store.create_task(
                TaskCreate(
                    task_id="forged-child",
                    type="step",
                    parent_task_id=parent.id,
                    invocation_origin=InvocationOriginClaim(subject="replacement-user"),
                )
            )

    asyncio.run(exercise())


@pytest.mark.parametrize("store_factory", [InMemoryTaskStore, SQLiteTaskStore])
def test_task_stores_load_bounded_invocation_snapshots(store_factory, tmp_path) -> None:
    store = (
        SQLiteTaskStore(tmp_path / "task-invocation-snapshot.sqlite")
        if store_factory is SQLiteTaskStore
        else InMemoryTaskStore()
    )

    async def exercise() -> None:
        task = await store.create_task(
            TaskCreate(
                task_id="snapshot-task",
                type="work",
                input={"payload": "x" * 100_000},
                metadata={"private": "not part of the projection"},
            )
        )

        snapshot = await store.load_invocation_snapshot(task.id)
        assert snapshot is not None
        assert snapshot.id == task.id
        assert snapshot.session_id == task.session_id
        assert snapshot.invocation == task.invocation
        assert set(snapshot.model_dump()) == {
            "id",
            "session_id",
            "session_instance_id",
            "invocation",
        }
        assert await store.load_invocation_snapshot("missing") is None

        if hasattr(store, "close"):
            await store.close()

    asyncio.run(exercise())


def test_task_backed_session_inherits_task_root_provenance() -> None:
    async def exercise() -> None:
        sessions = InMemorySessionStore()
        tasks = InMemoryTaskStore()
        task = await tasks.create_task(
            task_create_with_execution_source(
                TaskCreate(
                    task_id="scheduled-root",
                    type="scheduled-report",
                    invocation_origin=InvocationOriginClaim(subject="scheduler:nightly"),
                    available_at=datetime.now(UTC) + timedelta(minutes=5),
                ),
                source=TaskExecutionSource.SCHEDULED,
            )
        )
        app = CayuApp(
            session_store=sessions,
            task_store=tasks,
            enable_logging=False,
        )

        request = RunRequest(
            agent_name="assistant",
            session_id="scheduled-session",
            task_id=task.id,
            messages=[],
        )
        source_task = await tasks.load_invocation_snapshot(task.id)
        assert source_task is not None
        session = await sessions.create(
            run_request_with_task_invocation(request, source_task),
            identity=_identity(),
        )

        assert session.invocation.origin == task.invocation.origin
        assert session.invocation.root_invocation_id == task.invocation.root_invocation_id
        assert session.invocation.root_session_id == session.id
        assert session.invocation.source is SessionExecutionSource.TASK

        derived = await app.create_task(
            TaskCreate(task_id="session-child", type="follow-up", session_id=session.id)
        )
        assert derived.invocation.origin == task.invocation.origin
        assert derived.invocation.root_invocation_id == task.invocation.root_invocation_id
        assert derived.invocation.root_session_id == session.id

    asyncio.run(exercise())


def test_session_engine_revalidates_an_already_running_task_attachment() -> None:
    async def exercise() -> None:
        sessions = InMemorySessionStore()
        tasks = InMemoryTaskStore()
        session = await sessions.create(
            RunRequest(
                agent_name="assistant",
                session_id="shared-session",
                messages=[],
            ),
            identity=_identity(),
        )
        task = await tasks.create_task(
            TaskCreate(
                task_id="conflicting-running-task",
                type="work",
                session_id=session.id,
            )
        )
        await tasks.start_task(
            task.id,
            session_invocation=SessionInvocationBinding(
                id=session.id,
                session_instance_id=session.instance_id,
                invocation=session_invocation_from_task(
                    task.invocation,
                    session_id=session.id,
                ),
            ),
        )
        app = CayuApp(
            session_store=sessions,
            task_store=tasks,
            enable_logging=False,
        )

        with pytest.raises(ValueError, match="provenance conflict"):
            await app._session_engine._start_task(
                task_id=task.id,
                session=session,
            )

        unchanged = await tasks.load_task(task.id)
        assert unchanged is not None
        assert unchanged.status is TaskStatus.RUNNING
        assert unchanged.session_id == session.id
        assert unchanged.invocation == task.invocation

    asyncio.run(exercise())


def test_task_backed_session_rejects_a_different_task_snapshot() -> None:
    async def exercise() -> None:
        sessions = InMemorySessionStore()
        tasks = InMemoryTaskStore()
        first = await tasks.create_task(TaskCreate(task_id="first", type="work"))
        second = await tasks.create_task(TaskCreate(task_id="second", type="work"))
        second_snapshot = await tasks.load_invocation_snapshot(second.id)
        assert second_snapshot is not None

        request = RunRequest(
            agent_name="assistant",
            session_id="task-session",
            task_id=first.id,
            messages=[],
        )
        with pytest.raises(ValueError, match="conflicts with RunRequest.task_id"):
            run_request_with_task_invocation(request, second_snapshot)
        assert await sessions.load("task-session") is None

    asyncio.run(exercise())


def test_task_attachment_rejects_a_different_invocation_tree() -> None:
    async def exercise() -> None:
        sessions = InMemorySessionStore()
        tasks = InMemoryTaskStore()
        task = await tasks.create_task(TaskCreate(task_id="task", type="work"))
        unrelated = await sessions.create(
            RunRequest(agent_name="assistant", session_id="unrelated", messages=[]),
            identity=_identity(),
        )

        with pytest.raises(ValueError, match="provenance conflict"):
            await tasks.start_task(
                task.id,
                session_id=unrelated.id,
                session_invocation=SessionInvocationBinding(
                    id=unrelated.id,
                    session_instance_id=unrelated.instance_id,
                    invocation=unrelated.invocation,
                ),
            )
        unchanged = await tasks.load_task(task.id)
        assert unchanged is not None
        assert unchanged.status is TaskStatus.PENDING
        assert unchanged.session_id is None
        assert unchanged.invocation == task.invocation

    asyncio.run(exercise())


def test_task_attachment_requires_explicit_session_provenance() -> None:
    async def exercise() -> None:
        tasks = InMemoryTaskStore()
        task = await tasks.create_task(
            TaskCreate(
                task_id="planned-task",
                type="work",
                session_id="planned-session",
            )
        )

        with pytest.raises(ValueError, match="Session provenance binding is required"):
            await tasks.start_task(task.id, session_id="planned-session")

        unchanged = await tasks.load_task(task.id)
        assert unchanged is not None
        assert unchanged.status is TaskStatus.PENDING
        assert unchanged.invocation == task.invocation

        started = await tasks.start_task(
            task.id,
            session_id="planned-session",
            session_invocation=SessionInvocationBinding(
                id="planned-session",
                session_instance_id=str(uuid4()),
                invocation=session_invocation_from_task(
                    task.invocation,
                    session_id="planned-session",
                ),
            ),
        )
        assert started.status is TaskStatus.RUNNING
        assert started.invocation == task.invocation

    asyncio.run(exercise())


def test_sqlite_task_provenance_reconstructs_after_reopen(tmp_path) -> None:
    database = tmp_path / "task-reopen.sqlite"

    async def create() -> str:
        store = SQLiteTaskStore(database)
        task = await store.create_task(
            task_create_with_execution_source(
                TaskCreate(
                    task_id="webhook-task",
                    type="webhook",
                    invocation_origin=InvocationOriginClaim(subject="github:org/repo"),
                ),
                source=TaskExecutionSource.WEBHOOK,
            )
        )
        await store.close()
        return task.invocation.root_invocation_id

    async def reopen(root_invocation_id: str) -> None:
        store = SQLiteTaskStore(database)
        task = await store.load_task("webhook-task")
        assert task is not None
        assert task.invocation.root_invocation_id == root_invocation_id
        assert task.invocation.origin.subject == "github:org/repo"
        assert task.invocation.source is TaskExecutionSource.WEBHOOK
        await store.close()

    root_invocation_id = asyncio.run(create())
    asyncio.run(reopen(root_invocation_id))


def test_sqlite_revision_thirty_nine_rejects_populated_task_database(tmp_path) -> None:
    database = tmp_path / "pre-task-provenance.sqlite"

    async def create() -> None:
        store = SQLiteTaskStore(database)
        await store.create_task(TaskCreate(task_id="existing", type="work"))
        await store.close()

    asyncio.run(create())
    connection = sqlite3.connect(database)
    try:
        connection.execute("DELETE FROM cayu_schema_migrations WHERE revision >= 39")
        connection.execute("PRAGMA user_version = 38")
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(
        migrations.SchemaTooOld,
        match="requires invocation provenance for every task",
    ):
        SQLiteTaskStore(database, schema_mode=migrations.SchemaMode.MIGRATE)

    connection = sqlite3.connect(database)
    try:
        assert connection.execute("PRAGMA user_version").fetchone() == (38,)
        assert connection.execute(
            "SELECT MAX(revision) FROM cayu_schema_migrations"
        ).fetchone() == (38,)
    finally:
        connection.close()


def test_cayu_app_rejects_workload_secrets_in_task_origin() -> None:
    secret = "task-origin-secret"
    app = CayuApp(
        task_store=InMemoryTaskStore(),
        secret_redactor=SecretRedactor(secret),
        enable_logging=False,
    )

    with pytest.raises(ValueError, match="Task invocation origin contains a workload secret"):
        asyncio.run(
            app.create_task(
                TaskCreate(
                    type="work",
                    invocation_origin=InvocationOriginClaim(subject=f"user:{secret}"),
                )
            )
        )

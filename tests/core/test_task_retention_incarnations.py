"""Reused public session IDs cannot reactivate a retired retention owner."""

import asyncio
from uuid import uuid4

import pytest
from tests.core.test_session_execution_presence import _app, _stores
from tests.runtime.runtime_provider_support import VersionedFakeProvider

from cayu import Message, RunRequest, SessionStatus
from cayu.providers import ModelStreamEvent
from cayu.runtime._checkpoint_store import _RuntimeCheckpointSessionStore
from cayu.sessions import ForkSessionRequest, IncompleteSessionRecoveryRequest
from cayu.sessions.authority import SessionRunFenced
from cayu.sessions.records import SessionIdentity
from cayu.storage import PostgresTaskStore, SQLiteTaskStore
from cayu.storage.migrations import SchemaMode
from cayu.tasks import InMemoryTaskStore, TaskCreate, TaskSessionClosureClaim


@pytest.mark.store_semantics
@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
@pytest.mark.parametrize("provisional", [False, True])
@pytest.mark.parametrize("creation", ["run", "fork", "fork-lost-ack", "fork-replay"])
def test_completed_session_id_can_be_reused_without_restoring_old_claims(
    backend, provisional, creation, request, tmp_path, monkeypatch
):
    dsn = request.getfixturevalue("postgres_dsn") if backend == "postgres" else None
    sqlite_resources = request.getfixturevalue("sqlite_resources")

    def open_tasks():
        if backend == "memory":
            return InMemoryTaskStore()
        if backend == "sqlite":
            return SQLiteTaskStore(tmp_path / "tasks.db")
        return PostgresTaskStore(dsn, schema_mode=SchemaMode.CREATE)

    async def scenario():
        async with _stores(backend, request, sqlite_resources) as (sessions, _):
            tasks = open_tasks()
            provider = VersionedFakeProvider(
                [[ModelStreamEvent.completed({"finish_reason": "stop"})] for _ in range(2)]
            )
            app = _app(sessions, provider, task_store=tasks)
            sid = str(uuid4())
            run = RunRequest(
                session_id=sid, agent_name="assistant", messages=[Message.text("user", "go")]
            )
            try:
                _ = [event async for event in app.run(run)]
                first = await sessions.load(sid)
                assert first.status is SessionStatus.COMPLETED
                old_claim = TaskSessionClosureClaim(
                    session_id=sid,
                    plan_id="a" * 64,
                    task_ids=(),
                    purpose="retention",
                    retention_instance_id=first.instance_id,
                    retention_run_epoch=first.run_epoch,
                )
                if provisional:
                    await tasks.claim_session_closure(old_claim)
                await sessions.delete_session(sid)
                if creation == "run":
                    _ = [event async for event in app.run(run)]
                else:
                    source_id = str(uuid4())
                    _ = [
                        event
                        async for event in app.run(run.model_copy(update={"session_id": source_id}))
                    ]
                    fork_request = ForkSessionRequest(source_session_id=source_id, session_id=sid)
                    if creation == "fork-lost-ack":
                        create_fork = sessions.create_profiled_fork

                        async def lose_ack(**kwargs):
                            await create_fork(**kwargs)
                            raise OSError("lost fork acknowledgement")

                        with monkeypatch.context() as patch:
                            patch.setattr(sessions, "create_profiled_fork", lose_ack)
                            _ = [event async for event in app.fork_session(fork_request)]
                    elif creation == "fork-replay":
                        advance = tasks.advance_session_retention_epoch

                        async def fail_retention(session_id, **kwargs):
                            if session_id == sid:
                                raise OSError("retention unavailable")
                            await advance(session_id, **kwargs)

                        with monkeypatch.context() as patch:
                            patch.setattr(tasks, "advance_session_retention_epoch", fail_retention)
                            with pytest.raises(Exception):
                                _ = [event async for event in app.fork_session(fork_request)]
                        assert await sessions.load(sid) is not None
                        # The committed child must repair its own retention floor
                        # even when its original parent no longer exists.
                        await sessions.delete_session(source_id)
                        _ = [event async for event in app.fork_session(fork_request)]
                    else:
                        _ = [event async for event in app.fork_session(fork_request)]
                current = await sessions.load(sid)
                assert current.status is SessionStatus.COMPLETED
                assert current.instance_id != first.instance_id
                assert len(provider.requests) == 2
                assert await tasks.load_session_closure_claim(sid) is None
                if backend != "memory":
                    await tasks.close()
                    tasks = open_tasks()
                with pytest.raises(ValueError, match="superseded invocation"):
                    await tasks.claim_session_closure(old_claim)
                with pytest.raises(ValueError, match="retired session instance"):
                    await tasks.advance_session_retention_epoch(
                        sid,
                        session_instance_id=first.instance_id,
                        run_epoch=current.run_epoch + 100,
                    )
                attached = await tasks.create_task(TaskCreate(type="new-instance", session_id=sid))
                await tasks.cancel_task(attached.id)
                claim = TaskSessionClosureClaim(
                    session_id=sid,
                    plan_id="b" * 64,
                    task_ids=(attached.id,),
                    purpose="retention",
                    retention_instance_id=current.instance_id,
                    retention_run_epoch=current.run_epoch,
                )
                assert await tasks.claim_session_closure(claim) == claim
                await tasks.advance_session_retention_epoch(
                    sid, session_instance_id=current.instance_id, run_epoch=current.run_epoch + 1
                )
                assert await tasks.load_session_closure_claim(sid) is None
                with pytest.raises(ValueError, match="superseded invocation"):
                    await tasks.claim_session_closure(claim)
            finally:
                if backend != "memory":
                    await tasks.close()

    asyncio.run(scenario())


@pytest.mark.store_semantics
@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
@pytest.mark.parametrize("commit_order", ["replacement_first", "original_first"])
def test_delayed_first_epoch_cannot_replace_a_newer_session_incarnation(
    backend, request, tmp_path, monkeypatch, commit_order
):
    dsn = request.getfixturevalue("postgres_dsn") if backend == "postgres" else None
    sqlite_resources = request.getfixturevalue("sqlite_resources")

    def open_tasks():
        if backend == "memory":
            return InMemoryTaskStore()
        if backend == "sqlite":
            return SQLiteTaskStore(tmp_path / "delayed-first.db")
        return PostgresTaskStore(dsn, schema_mode=SchemaMode.CREATE)

    async def scenario():
        async with _stores(backend, request, sqlite_resources) as (sessions, _):
            tasks = open_tasks()
            sid = str(uuid4())
            run = RunRequest(
                session_id=sid, agent_name="assistant", messages=[Message.text("user", "go")]
            )
            first = await sessions.create(
                run, identity=SessionIdentity(provider_name="fake", model="fake-model")
            )
            first = await sessions.transition_status(
                sid, from_statuses={SessionStatus.PENDING}, to_status=SessionStatus.RUNNING
            )
            reached, release = asyncio.Event(), asyncio.Event()
            replacement_reached, release_replacement = asyncio.Event(), asyncio.Event()
            advance = tasks.advance_session_retention_epoch

            async def delayed_advance(session_id, **kwargs):
                if kwargs["session_instance_id"] == first.instance_id:
                    reached.set()
                    await release.wait()
                elif commit_order == "original_first":
                    replacement_reached.set()
                    await release_replacement.wait()
                await advance(session_id, **kwargs)

            monkeypatch.setattr(tasks, "advance_session_retention_epoch", delayed_advance)
            wrapped = _RuntimeCheckpointSessionStore(sessions, task_store=tasks)
            delayed = asyncio.create_task(wrapped._advance_task_retention_epoch(first))
            replacement_run = None
            try:
                await asyncio.wait_for(reached.wait(), 5)
                # The original call has checked A in the session store, but its
                # first task-store write has not acquired publication authority.
                observer = _app(sessions, VersionedFakeProvider([]))
                recovered = await observer.recover_incomplete_session(
                    IncompleteSessionRecoveryRequest(session_id=sid, inactive_for_seconds=0)
                )
                assert recovered.status is SessionStatus.INTERRUPTED
                await sessions.delete_session(sid)
                provider = VersionedFakeProvider([ModelStreamEvent.completed()])
                app = _app(sessions, provider, task_store=tasks)

                async def run_replacement():
                    return [event async for event in app.run(run)]

                replacement_run = asyncio.create_task(run_replacement())
                if commit_order == "original_first":
                    await asyncio.wait_for(replacement_reached.wait(), 5)
                    release.set()
                    await asyncio.wait_for(delayed, 5)
                    release_replacement.set()
                await asyncio.wait_for(replacement_run, 15)
                assert len(provider.requests) == 1
                current = await sessions.load(sid)
                assert current.status is SessionStatus.COMPLETED
                claim = TaskSessionClosureClaim(
                    session_id=sid,
                    plan_id="b" * 64,
                    task_ids=(),
                    purpose="retention",
                    retention_instance_id=current.instance_id,
                    retention_run_epoch=current.run_epoch,
                )
                await tasks.claim_session_closure(claim)
                release.set()
                if commit_order == "replacement_first":
                    with pytest.raises(SessionRunFenced):
                        await delayed
                assert await tasks.load_session_closure_claim(sid) == claim
                if backend != "memory":
                    await tasks.close()
                    tasks = open_tasks()
                floor = await tasks.load_session_retention_epoch(sid)
                assert floor is not None and floor[0] == current.instance_id
                assert floor[1] <= current.run_epoch
                with pytest.raises(ValueError, match="superseded invocation"):
                    await tasks.claim_session_closure(
                        claim.model_copy(
                            update={
                                "retention_instance_id": first.instance_id,
                                "retention_run_epoch": first.run_epoch,
                            }
                        )
                    )
                await tasks.advance_session_retention_epoch(
                    sid, session_instance_id=current.instance_id, run_epoch=current.run_epoch + 1
                )
                assert await tasks.load_session_closure_claim(sid) is None
            finally:
                release.set()
                release_replacement.set()
                await asyncio.gather(
                    delayed,
                    *([replacement_run] if replacement_run is not None else []),
                    return_exceptions=True,
                )
                if backend != "memory":
                    await tasks.close()

    asyncio.run(scenario())


@pytest.mark.store_semantics
@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
def test_first_owned_claim_registers_an_incarnation_before_session_id_reuse(
    backend, request, tmp_path
):
    dsn = request.getfixturevalue("postgres_dsn") if backend == "postgres" else None

    def open_tasks():
        if backend == "memory":
            return InMemoryTaskStore()
        if backend == "sqlite":
            return SQLiteTaskStore(tmp_path / "first-claim.db")
        return PostgresTaskStore(dsn, schema_mode=SchemaMode.CREATE)

    async def scenario():
        tasks = open_tasks()
        sid = str(uuid4())
        first, replacement = str(uuid4()), str(uuid4())
        claim = TaskSessionClosureClaim(
            session_id=sid,
            plan_id="a" * 64,
            task_ids=(),
            purpose="retention",
            retention_instance_id=first,
            retention_run_epoch=7,
        )
        try:
            # No task/session invocation has established an epoch floor yet.
            assert await tasks.claim_session_closure(claim) == claim
            assert await tasks.claim_session_closure(claim) == claim
            if backend != "memory":
                await tasks.close()
                tasks = open_tasks()
            await tasks.advance_session_retention_epoch(
                sid,
                session_instance_id=replacement,
                run_epoch=0,
                expected_session_instance_id=first,
            )
            assert await tasks.load_session_closure_claim(sid) is None
            current_claim = claim.model_copy(
                update={
                    "plan_id": "b" * 64,
                    "retention_instance_id": replacement,
                    "retention_run_epoch": 0,
                }
            )
            assert await tasks.claim_session_closure(current_claim) == current_claim
            if backend != "memory":
                await tasks.close()
                tasks = open_tasks()
            with pytest.raises(ValueError, match="retired session instance"):
                await tasks.advance_session_retention_epoch(
                    sid, session_instance_id=first, run_epoch=100
                )
            assert await tasks.load_session_closure_claim(sid) == current_claim
            with pytest.raises(ValueError, match="superseded invocation"):
                await tasks.claim_session_closure(claim)
            await tasks.advance_session_retention_epoch(
                sid, session_instance_id=replacement, run_epoch=1
            )
            assert await tasks.load_session_closure_claim(sid) is None
            assert await tasks.create_task(TaskCreate(type="replacement", session_id=sid))
        finally:
            if backend != "memory":
                await tasks.close()

    asyncio.run(scenario())

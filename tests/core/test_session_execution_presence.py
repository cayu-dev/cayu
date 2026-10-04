from __future__ import annotations

import asyncio
import math
import os
import socket
import subprocess
import sys
import time
from contextlib import asynccontextmanager
from uuid import uuid4

import httpx
import pytest
from tests.core.test_runtime import VersionedFakeProvider

from cayu import (
    AgentSpec,
    CayuApp,
    EventType,
    InMemorySessionStore,
    InMemoryTaskStore,
    Message,
    ModelStreamEvent,
    PostgresSessionStore,
    RecoveryPlanRequest,
    RecoveryPlanSelection,
    RunRequest,
    SecretRedactor,
    SessionExecutionConfig,
    SessionStatus,
    SQLiteSessionStore,
    TaskCreate,
    Tool,
    ToolResult,
    ToolSpec,
    UserInputTool,
)
from cayu.runtime.execution_identity import ExecutionProfileBehaviorIdentity
from cayu.sessions.base import SessionIdentity, _activate_session_run_fence
from cayu.storage.migrations import SchemaMode
from cayu.tasks.worker import run_task_worker


@asynccontextmanager
async def _stores(backend, request, sqlite_resources):
    async with sqlite_resources as resources:
        opened = []
        if backend == "memory":
            store = InMemorySessionStore()
            yield store, lambda: store
            return
        if backend == "sqlite":
            path = resources.path("execution.sqlite")

            def reopen():
                return resources.own(SQLiteSessionStore(path))
        else:
            dsn = request.getfixturevalue("postgres_dsn")

            def reopen():
                store = PostgresSessionStore(
                    dsn, min_size=1, max_size=2, schema_mode=SchemaMode.CREATE
                )
                opened.append(store)
                return store

        try:
            yield reopen(), reopen
        finally:
            for store in opened:
                await store.close()


class _BlockedProvider(VersionedFakeProvider):
    def __init__(self):
        super().__init__([ModelStreamEvent.completed({"finish_reason": "stop"})])
        self.entered, self.release = asyncio.Event(), asyncio.Event()

    async def stream(self, request):
        self.entered.set()
        await self.release.wait()
        async for event in super().stream(request):
            yield event


class _BlockedTool(Tool):
    spec = ToolSpec(
        name="blocked_tool",
        description="Wait for the test's release signal.",
        input_schema={"type": "object", "additionalProperties": False},
    )

    def __init__(self):
        self.entered, self.release = asyncio.Event(), asyncio.Event()

    async def run(self, arguments, context):
        self.entered.set()
        await self.release.wait()
        return ToolResult(content="done")


def _app(store, provider, *, tools=(), task_store=None):
    app = CayuApp(
        session_store=store,
        task_store=task_store,
        enable_logging=False,
        session_execution=SessionExecutionConfig(
            heartbeat_interval_seconds=0.05, lease_seconds=1.5
        ),
    )
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"), tools=tools)
    return app


async def _consume(stream):
    return [event async for event in stream]


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
@pytest.mark.parametrize("cancel_during_release", [False, True])
def test_public_run_waits_for_exact_execution_presence_release(
    backend, cancel_during_release, request, sqlite_resources
):
    async def scenario():
        async with _stores(backend, request, sqlite_resources) as (store, _):
            session_id = f"await-presence-release-{cancel_during_release}"
            provider = VersionedFakeProvider(
                [[ModelStreamEvent.completed({"finish_reason": "stop"})]]
            )
            app = _app(store, provider)
            entered, release = asyncio.Event(), asyncio.Event()
            original = store._release_session_execution
            released = []

            async def delayed_release(owner):
                entered.set()
                await release.wait()
                await original(owner)
                released.append(owner)

            store._release_session_execution = delayed_release
            task = asyncio.create_task(
                _consume(
                    app.run(
                        RunRequest(
                            session_id=session_id,
                            agent_name="assistant",
                            messages=[Message.text("user", "complete")],
                        )
                    )
                )
            )
            try:
                await asyncio.wait_for(entered.wait(), 30)
                assert not task.done() and not released
                if cancel_during_release:
                    task.cancel("cancel during presence release")
                    await asyncio.sleep(0)
                    assert not task.done()
                release.set()
                if cancel_during_release:
                    with pytest.raises(asyncio.CancelledError):
                        await task
                    assert task.cancelled() and task.cancelling() == 1
                else:
                    await task
                assert len(released) == 1
                assert not app._session_control.execution_presence.groups
                assert (await store.load(session_id)).status is SessionStatus.COMPLETED
                assert len(provider.requests) == 1
            finally:
                release.set()
                await asyncio.gather(task, return_exceptions=True)

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
@pytest.mark.parametrize("cancel_during_release", [False, True])
def test_public_run_retains_blocked_presence_release_for_bounded_drain(
    backend, cancel_during_release, request, sqlite_resources, monkeypatch
):
    from cayu.runtime import _session_execution_presence as presence_module

    monkeypatch.setattr(presence_module, "_RELEASE_WAIT_SECONDS", 0.1)

    async def scenario():
        async with _stores(backend, request, sqlite_resources) as (store, _):
            session_id = f"retained-presence-release-{cancel_during_release}"
            provider = VersionedFakeProvider(
                [[ModelStreamEvent.completed({"finish_reason": "stop"})]]
            )
            app = _app(store, provider)
            entered, release = asyncio.Event(), asyncio.Event()
            original = store._release_session_execution
            released = []

            async def delayed_release(owner):
                entered.set()
                await release.wait()
                await original(owner)
                released.append(owner)

            monkeypatch.setattr(store, "_release_session_execution", delayed_release)
            task = asyncio.create_task(
                _consume(
                    app.run(
                        RunRequest(
                            session_id=session_id,
                            agent_name="assistant",
                            messages=[Message.text("user", "complete")],
                        )
                    )
                )
            )
            try:
                await asyncio.wait_for(entered.wait(), 30)
                presence = app._session_control.execution_presence
                group = next(iter(presence.groups.values()))
                if cancel_during_release:
                    task.cancel("first cancellation")
                    await asyncio.sleep(0)
                    task.cancel("second cancellation")
                done, _ = await asyncio.wait({task}, timeout=5)
                assert task in done, "Presence release must not indefinitely hold the caller"
                if cancel_during_release:
                    with pytest.raises(asyncio.CancelledError):
                        await task
                    assert task.cancelled() and task.cancelling() == 2
                else:
                    assert (await task)[-1].type is EventType.SESSION_COMPLETED
                assert not released and not group.task.done()
                assert group.stop.is_set() and group in presence.groups.values()
                assert not await app.drain_recovery_cleanups(timeout_s=0.01)
                observer = asyncio.create_task(app.drain_recovery_cleanups(timeout_s=10))
                await asyncio.sleep(0)
                observer.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await observer
                assert observer.cancelled() and observer.cancelling() == 1
                assert not group.task.done() and group.task.cancelling() == 0
                release.set()
                assert await app.drain_recovery_cleanups(timeout_s=10)
                assert len(released) == 1 and not presence.groups
                assert (await store.load(session_id)).status is SessionStatus.COMPLETED
                assert len(provider.requests) == 1
            finally:
                release.set()
                await asyncio.gather(task, return_exceptions=True)
                await app.drain_recovery_cleanups(timeout_s=10)

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
@pytest.mark.parametrize("phase", ["provider", "tool"])
def test_independent_heartbeat_keeps_long_silent_work_live(
    backend, phase, request, sqlite_resources
):
    async def scenario():
        async with _stores(backend, request, sqlite_resources) as (store, reopen):
            blocked = _BlockedProvider() if phase == "provider" else _BlockedTool()
            provider = (
                blocked
                if phase == "provider"
                else VersionedFakeProvider(
                    [
                        [
                            ModelStreamEvent.tool_call(
                                id="effect", name="blocked_tool", arguments={}
                            ),
                            ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                        ],
                        [ModelStreamEvent.completed({"finish_reason": "stop"})],
                    ]
                )
            )
            app = _app(store, provider, tools=[] if phase == "provider" else [blocked])
            sid = "silent-" + phase
            run = asyncio.create_task(
                _consume(
                    app.run(
                        RunRequest(
                            agent_name="assistant",
                            session_id=sid,
                            messages=[Message.text("user", "private-workload-text")],
                        )
                    )
                )
            )
            try:
                await asyncio.wait_for(blocked.entered.wait(), 15)
                assert await app.drain_recovery_cleanups(timeout_s=1)
                assert not run.done()
                assert all(
                    not group.stop.is_set()
                    for group in app._session_control.execution_presence.groups.values()
                )
                observer = CayuApp(session_store=reopen(), enable_logging=False)
                original = await observer.inspect_session_execution(sid)
                epoch = original.run_epoch
                activity = (await store.load_state(sid)).last_activity_at
                # Outlast the original lease while neither the provider nor tool makes progress.
                await asyncio.sleep(1.6)
                current = await observer.inspect_session_execution(sid)
                assert current.state == "executing"
                assert current.owner_kind == "in_process_runner"
                assert current.run_epoch == epoch
                operation = (await store.load_checkpoint(sid) or {}).get("session_run_operation")
                if operation is not None:
                    assert current.operation_id.startswith("sha256:")
                assert current.heartbeat_at > original.heartbeat_at
                assert current.lease_expires_at > original.lease_expires_at
                assert current.last_progress_kind == (
                    "model_stream" if phase == "provider" else "tool_call"
                )
                assert "private-workload-text" not in current.model_dump_json()
                assert "token" not in current.model_dump()
                assert (await store.load_state(sid)).last_activity_at == activity
                assert (await app.inspect_session_execution(sid)).local_owner
                assert not current.local_owner
                assert (
                    await store.fence_stalled_run(
                        sid, statuses={SessionStatus.RUNNING}, inactive_for_seconds=0
                    )
                    is None
                )
                plan = await app.plan_recovery(
                    RecoveryPlanRequest(selection=RecoveryPlanSelection(session_ids=(sid,)))
                )
                assert "active_execution_owner" in {
                    blocker.code.value for blocker in plan.items[0].blockers
                }
            finally:
                blocked.release.set()
                await asyncio.wait_for(run, 15)
            assert (await observer.inspect_session_execution(sid)).state == "terminal"

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "pending_key", ["pending_user_input", "pending_tool_approval", "foreground_child_wait"]
)
def test_legacy_and_human_wait_projections_never_invent_execution(pending_key):
    async def scenario():
        store = InMemorySessionStore()
        app = CayuApp(session_store=store, enable_logging=False)
        session = await store.create(
            RunRequest(agent_name="assistant", session_id="legacy", messages=[]),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        await store.update_status(session.id, SessionStatus.RUNNING)
        assert (await app.inspect_session_execution(session.id)).state == "unknown"
        # This test concerns the checkpoint's bounded presence flag, not action parsing.
        store._checkpoints[session.id] = {pending_key: {"private": "hidden"}}
        waiting = await app.inspect_session_execution(session.id)
        assert waiting.state == "waiting"
        assert waiting.owner_id is None
        assert "hidden" not in waiting.model_dump_json()

    asyncio.run(scenario())


def test_execution_config_rejects_unbounded_intervals_and_secret_labels():
    for values in (
        {"heartbeat_interval_seconds": 0},
        {"lease_seconds": float("inf")},
        {"heartbeat_interval_seconds": 2, "lease_seconds": 3},
        {"owner_label": "\nprivate"},
    ):
        with pytest.raises(ValueError):
            SessionExecutionConfig(**values)
    with pytest.raises(ValueError, match="workload secret"):
        CayuApp(
            secret_redactor=SecretRedactor("private-fragment"),
            session_execution=SessionExecutionConfig(owner_label="private-fragment-worker"),
        )


def test_state_etag_tracks_heartbeat_and_expiry_without_renewing():
    from cayu.server import create_server
    from cayu.server.config import ServerConfig

    async def scenario():
        store = InMemorySessionStore()
        app = CayuApp(session_store=store, enable_logging=False)
        server = create_server(app, config=ServerConfig.local_development())
        session = await store.create(
            RunRequest(agent_name="assistant", session_id="etag-owner", messages=[]),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        session = await store.update_status(session.id, SessionStatus.RUNNING)
        _activate_session_run_fence(session)
        owner = await store._claim_session_execution(
            session.id,
            token=uuid4().hex,
            owner_id=uuid4().hex,
            owner_kind="server_stream",
            owner_label=None,
            lease_seconds=2,
        )
        transport = httpx.ASGITransport(app=server)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            path = "/api/sessions/etag-owner/state"
            first = await client.get(path)
            assert first.status_code == 200
            assert first.json()["execution"]["state"] == "executing"
            etag = first.headers["etag"]
            assert (await client.get(path, headers={"If-None-Match": etag})).status_code == 304
            await asyncio.sleep(0.02)
            renewed = await store._renew_session_execution(
                owner, lease_seconds=2, progress_kind=None
            )
            changed = await client.get(path, headers={"If-None-Match": etag})
            assert changed.status_code == 200
            assert changed.headers["etag"] != etag
            etag = changed.headers["etag"]
            await asyncio.sleep(2.1)
            expired = await client.get(path, headers={"If-None-Match": etag})
            assert expired.status_code == 200
            assert expired.json()["execution"]["state"] == "owner_lost"
            assert expired.json()["execution"]["run_epoch"] == session.run_epoch
            assert expired.json()["execution"]["lease_expires_at"] == (
                renewed.lease_expires_at.isoformat().replace("+00:00", "Z")
            )
            schema = server.openapi()
            assert (
                "execution" in schema["components"]["schemas"]["SessionStateResponse"]["required"]
            )

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
def test_expired_owner_cannot_renew_or_block_existing_fenced_recovery(
    backend, request, sqlite_resources
):
    async def scenario():
        async with _stores(backend, request, sqlite_resources) as (store, reopen):
            session = await store.create(
                RunRequest(agent_name="assistant", session_id="expired-owner", messages=[]),
                identity=SessionIdentity(provider_name="fake", model="fake-model"),
            )
            session = await store.update_status(session.id, SessionStatus.RUNNING)
            observer = reopen()
            await observer.load_state(session.id)
            _activate_session_run_fence(session)
            owner = await store._claim_session_execution(
                session.id,
                token=uuid4().hex,
                owner_id=uuid4().hex,
                owner_kind="in_process_runner",
                owner_label=None,
                lease_seconds=0.3,
            )
            before = await observer.inspect_session_execution(session.id)
            assert before.state == "executing"
            await asyncio.sleep(0.4)
            expired = await observer.inspect_session_execution(session.id)
            assert expired.state == "owner_lost"
            assert expired.run_epoch == before.run_epoch
            assert expired.lease_expires_at == before.lease_expires_at
            assert (
                await store._renew_session_execution(owner, lease_seconds=1, progress_kind=None)
                is None
            )
            fenced = await store.fence_stalled_run(
                session.id, statuses={SessionStatus.RUNNING}, inactive_for_seconds=0
            )
            assert fenced is not None
            assert fenced.run_epoch > before.run_epoch
            assert (await observer.inspect_session_execution(session.id)).state == "unknown"
            assert (
                await store._renew_session_execution(owner, lease_seconds=1, progress_kind=None)
                is None
            )

    asyncio.run(scenario())


def test_server_startup_retries_sessions_skipped_for_live_lease_after_expiry(monkeypatch):
    from fastapi.testclient import TestClient

    import cayu.server as server_module
    from cayu.server import ServerConfig, ServerLifecycleConfig, create_server

    async def prepare():
        store = InMemorySessionStore()
        session = await store.create(
            RunRequest(agent_name="assistant", session_id="startup-live-lease", messages=[]),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        session = await store.update_status(session.id, SessionStatus.RUNNING)
        _activate_session_run_fence(session)
        # A process that crashed moments before this restart still holds the lease.
        owner = await store._claim_session_execution(
            session.id,
            token=uuid4().hex,
            owner_id="crashed-process",
            owner_kind="server_stream",
            owner_label=None,
            lease_seconds=1.0,
        )
        return store, owner

    store, owner = asyncio.run(prepare())
    monkeypatch.setattr(server_module, "_STARTUP_EXECUTION_OWNER_RETRY_MARGIN_SECONDS", 0.05)
    app = _app(store, _BlockedProvider())
    plans = []
    plan_recovery, execute_recovery = app.plan_recovery, app.execute_recovery
    executed = []

    async def recording_plan(request):
        plan = await plan_recovery(request)
        plans.append((time.time(), plan))
        return plan

    async def recording_execute(request):
        executed.append(request.plan.plan_id)
        return await execute_recovery(request)

    app.plan_recovery = recording_plan
    app.execute_recovery = recording_execute
    server = create_server(
        app,
        config=ServerConfig.local_development(
            lifecycle=ServerLifecycleConfig(
                startup_recovery_statuses={SessionStatus.RUNNING},
                recovery_inactive_after_seconds=0,
            )
        ),
    )

    def blocker_codes(plan):
        return {blocker.code.value for item in plan.items for blocker in item.blockers}

    with TestClient(server):
        deadline = time.monotonic() + 10
        while len(plans) < 2 and time.monotonic() < deadline:
            time.sleep(0.02)

    assert len(plans) == 2
    (_, first), (retried_at, retried) = plans
    assert first.request.selection.statuses == {SessionStatus.RUNNING}
    assert "active_execution_owner" in blocker_codes(first)
    assert retried.request.selection.session_ids == ("startup-live-lease",)
    assert retried.request.selection.inactive_for_seconds == 0
    assert retried_at >= owner.lease_expires_at.timestamp()
    assert "active_execution_owner" not in blocker_codes(retried)
    assert executed == [first.plan_id, retried.plan_id]


def test_task_worker_has_distinct_execution_owner_kind():
    async def scenario():
        store, tasks, provider = InMemorySessionStore(), InMemoryTaskStore(), _BlockedProvider()
        app = _app(store, provider, task_store=tasks)
        await tasks.create_task(TaskCreate(task_id="execution-worker-task", type="job"))

        async def handler(_app, task, worker_id):
            await _consume(
                _app.run(
                    RunRequest(
                        agent_name="assistant",
                        session_id="worker-execution",
                        messages=[Message.text("user", "go")],
                        task_id=task.id,
                        task_worker_id=worker_id,
                        task_lease_expires_at=task.lease_expires_at,
                    )
                )
            )

        worker = asyncio.create_task(
            run_task_worker(
                app, tasks, handler, worker_id="test-worker", max_tasks=1, reclaim=False
            )
        )
        try:
            await asyncio.wait_for(provider.entered.wait(), 15)
            assert (
                await app.inspect_session_execution("worker-execution")
            ).owner_kind == "task_worker"
        finally:
            provider.release.set()
            assert await asyncio.wait_for(worker, 15) == 1

    asyncio.run(scenario())


def test_model_delta_volume_does_not_increase_heartbeat_write_rate():
    async def scenario():
        store = InMemorySessionStore()
        renewals = 0
        renew = store._renew_session_execution

        async def counted(*args, **kwargs):
            nonlocal renewals
            renewals += 1
            return await renew(*args, **kwargs)

        store._renew_session_execution = counted
        app = _app(
            store,
            VersionedFakeProvider(
                [
                    *[ModelStreamEvent.text_delta("x") for _ in range(1000)],
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ]
            ),
        )
        started = time.monotonic()
        await _consume(
            app.run(RunRequest(agent_name="assistant", messages=[Message.text("user", "stream")]))
        )
        elapsed = time.monotonic() - started
        assert renewals <= math.ceil(elapsed / 0.05) + 1
        # Throughput varies with runner load; the elapsed-time bound is the contract.

    asyncio.run(scenario())


class _InputThenBlockProvider(VersionedFakeProvider):
    @property
    def execution_profile_identity(self):
        return ExecutionProfileBehaviorIdentity(
            name="tests:execution:input-block", behavior_version="1", implementation_version="1"
        )

    def __init__(self):
        super().__init__([])

    async def stream(self, request):
        if any(message.role == "tool" for message in request.messages):
            await asyncio.Event().wait()
        yield ModelStreamEvent.tool_call(
            id="question", name="ask_user", arguments={"question": "Continue?"}
        )
        yield ModelStreamEvent.completed({"finish_reason": "tool_calls"})


def _serve_execution_fixture():
    import uvicorn

    from cayu.server import create_server
    from cayu.server.config import ServerConfig

    backend = os.environ["CAYU_EXECUTION_TEST_BACKEND"]
    store = (
        SQLiteSessionStore(os.environ["CAYU_EXECUTION_TEST_PATH"])
        if backend == "sqlite"
        else PostgresSessionStore(
            os.environ["CAYU_EXECUTION_TEST_DSN"],
            min_size=1,
            max_size=2,
            schema_mode=SchemaMode.CREATE,
        )
    )
    app = _app(store, _InputThenBlockProvider(), tools=[UserInputTool()])
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(32)
    print(listener.getsockname()[1], flush=True)
    uvicorn.Server(
        uvicorn.Config(
            create_server(app, config=ServerConfig.local_development()), log_level="error"
        )
    ).run(sockets=[listener])


@pytest.mark.parametrize("backend", ["sqlite", "postgres"])
def test_server_resume_is_visible_from_another_process_and_sigkill_expires_owner(
    backend, request, sqlite_resources
):
    async def scenario():
        async with _stores(backend, request, sqlite_resources) as (store, reopen):
            app = _app(store, _InputThenBlockProvider(), tools=[UserInputTool()])
            sid = "cross-process-resolve"
            events = await _consume(
                app.run(
                    RunRequest(
                        agent_name="assistant",
                        session_id=sid,
                        messages=[Message.text("user", "Ask me to continue")],
                    )
                )
            )
            pending = next(
                event for event in events if event.type == EventType.SESSION_AWAITING_USER_INPUT
            )
            waiting = await app.inspect_session_execution(sid)
            assert waiting.state == "waiting"
            assert waiting.owner_id is None
            env = {**os.environ, "CAYU_EXECUTION_TEST_BACKEND": backend}
            if backend == "sqlite":
                env["CAYU_EXECUTION_TEST_PATH"] = str(store.path)
            else:
                env["CAYU_EXECUTION_TEST_DSN"] = request.getfixturevalue("postgres_dsn")
            child = subprocess.Popen(
                [
                    sys.executable,
                    "-c",
                    "from tests.core.test_session_execution_presence import _serve_execution_fixture; _serve_execution_fixture()",
                ],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            try:
                port = int(await asyncio.wait_for(asyncio.to_thread(child.stdout.readline), 20))
                origin = f"http://127.0.0.1:{port}"
                async with httpx.AsyncClient(timeout=20) as client:
                    for _ in range(200):
                        try:
                            if (await client.get(origin + "/api/health")).status_code == 200:
                                break
                        except httpx.ConnectError:
                            pass
                        await asyncio.sleep(0.025)
                    else:
                        raise AssertionError("Child server did not start")
                    async with client.stream(
                        "POST",
                        origin + "/api/user-input/resolve",
                        json={
                            "session_id": sid,
                            "input_id": pending.payload["input_id"],
                            "answer": "Continue",
                        },
                    ) as response:
                        assert response.status_code == 200, await response.aread()
                        await anext(response.aiter_lines())
                    observer = CayuApp(session_store=reopen(), enable_logging=False)
                    for _ in range(200):
                        execution = await observer.inspect_session_execution(sid)
                        if execution.state == "executing":
                            break
                        await asyncio.sleep(0.01)
                    assert execution.state == "executing"
                    assert execution.owner_kind == "server_stream"
                    assert not execution.local_owner
                    await asyncio.sleep(1.6)
                    execution = await observer.inspect_session_execution(sid)
                    assert execution.state == "executing"
                    assert execution.operation_id.startswith("sha256:")
                    await store.update_status(sid, SessionStatus.INTERRUPTING)
                    child.kill()  # A real SIGKILL: no release/finalizer runs.
                    await asyncio.to_thread(child.wait, 10)
                    await asyncio.sleep(1.6)
                    lost = await observer.inspect_session_execution(sid)
                    assert lost.state == "owner_lost"
                    assert lost.owner_id == execution.owner_id
                    assert lost.run_epoch == execution.run_epoch
                    assert (await store.load_state(sid)).status == SessionStatus.INTERRUPTING
            finally:
                if child.poll() is None:
                    child.kill()
                await asyncio.to_thread(child.wait, 10)
                child.stdout.close()
                child.stderr.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
def test_transient_heartbeat_failure_keeps_live_owner(
    backend, request, sqlite_resources, monkeypatch
):
    async def scenario():
        async with _stores(backend, request, sqlite_resources) as (store, reopen):
            provider = _BlockedProvider()
            app = _app(store, provider)
            failed, recovered = asyncio.Event(), asyncio.Event()
            renew = store._renew_session_execution
            inject = False

            async def fail_once(*args, **kwargs):
                if inject and not failed.is_set():
                    failed.set()
                    raise OSError("controlled renewal outage")
                result = await renew(*args, **kwargs)
                if failed.is_set() and result is not None:
                    recovered.set()
                return result

            monkeypatch.setattr(store, "_renew_session_execution", fail_once)
            task = asyncio.create_task(
                _consume(
                    app.run(
                        RunRequest(
                            agent_name="assistant",
                            session_id="transient-heartbeat",
                            messages=[Message.text("user", "start")],
                        )
                    )
                )
            )
            try:
                # Admission includes PostgreSQL setup before the heartbeat test starts.
                await asyncio.wait_for(provider.entered.wait(), 20)
                before = await store.inspect_session_execution("transient-heartbeat")
                inject = True
                await asyncio.wait_for(failed.wait(), 5)
                await asyncio.wait_for(recovered.wait(), 5)
                after = await reopen().inspect_session_execution("transient-heartbeat")
                assert after.state == "executing"
                assert after.owner_id == before.owner_id
                assert (
                    await store.fence_stalled_run(
                        "transient-heartbeat",
                        statuses={SessionStatus.RUNNING},
                        inactive_for_seconds=0,
                    )
                    is None
                )
            finally:
                provider.release.set()
                await task

    asyncio.run(scenario())


async def _eventually(predicate, timeout=5):
    deadline = time.monotonic() + timeout
    while True:
        result = await predicate()
        if result or time.monotonic() >= deadline:
            return result
        await asyncio.sleep(0.02)


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
@pytest.mark.parametrize("successor", [False, True], ids=["no-successor", "successor"])
def test_heartbeat_outage_longer_than_lease_reclaims_unless_successor_fenced(
    backend, successor, request, sqlite_resources, monkeypatch
):
    async def scenario():
        async with _stores(backend, request, sqlite_resources) as (store, reopen):
            provider = _BlockedProvider()
            app = _app(store, provider)
            presence = app._session_control.execution_presence
            sid = "long-outage-" + ("successor" if successor else "live")
            outage = False
            failures = 0
            renew = store._renew_session_execution

            async def unavailable(*args, **kwargs):
                nonlocal failures
                if outage:
                    failures += 1
                    raise OSError("controlled store outage")
                return await renew(*args, **kwargs)

            monkeypatch.setattr(store, "_renew_session_execution", unavailable)
            task = asyncio.create_task(
                _consume(
                    app.run(
                        RunRequest(
                            agent_name="assistant",
                            session_id=sid,
                            messages=[Message.text("user", "start")],
                        )
                    )
                )
            )
            try:
                await asyncio.wait_for(provider.entered.wait(), 5)
                observer = reopen()
                before = await observer.inspect_session_execution(sid)
                assert before.state == "executing"
                outage = True
                # Every renewal fails for longer than the 1.5s lease.
                await asyncio.sleep(2.0)
                assert failures > 1
                assert (await observer.inspect_session_execution(sid)).state == "owner_lost"
                assert presence.groups
                fenced = None
                if successor:
                    fenced = await observer.fence_stalled_run(
                        sid, statuses={SessionStatus.RUNNING}, inactive_for_seconds=0
                    )
                    assert fenced is not None
                outage = False
                if successor:

                    async def heartbeat_exited():
                        return not presence.groups

                    assert await _eventually(heartbeat_exited)
                    current = await observer.inspect_session_execution(sid)
                    assert current.run_epoch == fenced.run_epoch
                    assert current.state != "executing"
                else:

                    async def reclaimed():
                        state = await observer.inspect_session_execution(sid)
                        return state if state.state == "executing" else None

                    current = await _eventually(reclaimed)
                    assert current is not None
                    assert current.run_epoch == before.run_epoch
                    assert current.owner_id == before.owner_id
                    assert current.lease_expires_at > before.lease_expires_at
                    assert (
                        await store.fence_stalled_run(
                            sid, statuses={SessionStatus.RUNNING}, inactive_for_seconds=0
                        )
                        is None
                    )
            finally:
                outage = False
                provider.release.set()
                (outcome,) = await asyncio.wait_for(
                    asyncio.gather(task, return_exceptions=True), 15
                )
            if not successor:
                assert not isinstance(outcome, BaseException), outcome
                assert (await observer.load_state(sid)).status == SessionStatus.COMPLETED
                assert (await observer.inspect_session_execution(sid)).state == "terminal"

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
def test_presence_retires_with_local_fence_and_cleanup_can_remove_reservation(
    backend, request, sqlite_resources
):
    from cayu.sessions.base import _activate_owned_session_run_fence

    async def scenario():
        async with _stores(backend, request, sqlite_resources) as (store, _):
            session = await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id="retired-presence",
                    messages=[Message.text("user", "start")],
                ),
                identity=SessionIdentity(provider_name="fake", model="fake-model"),
            )
            session = await store.update_status(session.id, SessionStatus.RUNNING)
            owner = _activate_owned_session_run_fence(session)
            app = _app(store, _BlockedProvider())
            presence = app._session_control.execution_presence
            await presence.ensure(session)
            assert (await store.inspect_session_execution(session.id)).state == "executing"
            cleaned = await store.reserve_stalled_run_recovery(
                session.id,
                statuses={SessionStatus.RUNNING},
                inactive_for_seconds=None,
                checkpoint_transform=lambda _session, checkpoint, _now: {
                    **(checkpoint or {}),
                    "cleaned": True,
                },
            )
            assert cleaned is not None
            group = next(iter(presence.groups.values()))
            owner.retire()
            await asyncio.wait_for(group.task, 5)
            assert (await store.inspect_session_execution(session.id)).state == "owner_lost"
            assert not presence.groups and not presence.progress
            _activate_session_run_fence(session)
            await presence.ensure(session)
            assert (await store.inspect_session_execution(session.id)).state == "executing"
            group = next(iter(presence.groups.values()))
            presence.stop(session.id, run_epoch=session.run_epoch)
            await group.task

    asyncio.run(scenario())


def test_execution_progress_does_not_leak_for_unsupported_or_failed_claims(monkeypatch):
    from cayu.runtime._session_control import SessionControl

    async def scenario():
        class Unsupported(InMemorySessionStore):
            supports_session_execution = False

        control = SessionControl(session_store=Unsupported())
        task = asyncio.current_task()
        for index in range(20):
            control.register_active_task(
                str(index), task, task_id=None, task_started=False, task_finished=False
            )
            control.unregister_active_task(str(index), task)
        assert control.execution_presence.progress == {}
        store = InMemorySessionStore()
        control = SessionControl(session_store=store)
        session = await store.create(
            RunRequest(agent_name="assistant", messages=[Message.text("user", "start")]),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        await control.execution_presence.ensure(session)
        assert control.execution_presence.progress == {}

        async def fail(*args, **kwargs):
            raise OSError("claim failed")

        monkeypatch.setattr(store, "_claim_session_execution", fail)
        # Presence is observational, so a failed claim is logged, not raised.
        await control.execution_presence.ensure(session)
        assert control.execution_presence.progress == {}
        assert control.execution_presence.groups == {}

    asyncio.run(scenario())


def test_failed_presence_claim_does_not_abort_run(monkeypatch):
    async def scenario():
        store = InMemorySessionStore()
        app = _app(
            store, VersionedFakeProvider([ModelStreamEvent.completed({"finish_reason": "stop"})])
        )

        async def fail(*args, **kwargs):
            raise OSError("claim failed")

        monkeypatch.setattr(store, "_claim_session_execution", fail)
        events = await _consume(
            app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="presence-claim-failure",
                    messages=[Message.text("user", "start")],
                )
            )
        )
        assert events[-1].type == EventType.SESSION_COMPLETED
        assert (await store.load_state("presence-claim-failure")).status == SessionStatus.COMPLETED
        assert app._session_control.execution_presence.groups == {}

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
def test_recovery_reports_live_owner_expiry_in_single_and_batch_results(
    backend, request, sqlite_resources
):
    from cayu.sessions.base import (
        IncompleteSessionRecoveryAction,
        IncompleteSessionRecoveryRequest,
        IncompleteSessionsRecoveryRequest,
    )

    async def scenario():
        async with _stores(backend, request, sqlite_resources) as (store, reopen):
            provider = _BlockedProvider()
            app = _app(store, provider)
            task = asyncio.create_task(
                _consume(
                    app.run(
                        RunRequest(
                            agent_name="assistant",
                            session_id="live-recovery-report",
                            messages=[Message.text("user", "start")],
                        )
                    )
                )
            )
            try:
                await asyncio.wait_for(provider.entered.wait(), 20)
                observer = _app(reopen(), _BlockedProvider())
                result = await observer.recover_incomplete_session(
                    IncompleteSessionRecoveryRequest(
                        session_id="live-recovery-report", inactive_for_seconds=0
                    )
                )
                assert result.actions == (IncompleteSessionRecoveryAction.SKIPPED_EXECUTION_OWNER,)
                assert result.execution_lease_expires_at is not None
                assert result.execution_lease_expires_at.isoformat() in result.message
                page = await observer.recover_incomplete_sessions(
                    IncompleteSessionsRecoveryRequest(
                        statuses={SessionStatus.RUNNING}, inactive_for_seconds=0
                    )
                )
                observed = next(
                    item for item in page.results if item.session_id == "live-recovery-report"
                )
                assert observed.actions == result.actions
                assert observed.execution_lease_expires_at is not None
            finally:
                provider.release.set()
                await task

    asyncio.run(scenario())


def test_sqlite_execution_renewal_does_not_block_event_loop(sqlite_resources):
    import sqlite3

    async def scenario():
        async with sqlite_resources as resources:
            path = resources.path("locked-presence.sqlite")
            store = resources.own(SQLiteSessionStore(path))
            session = await store.create(
                RunRequest(agent_name="assistant", messages=[Message.text("user", "start")]),
                identity=SessionIdentity(provider_name="fake", model="fake-model"),
            )
            session = await store.update_status(session.id, SessionStatus.RUNNING)
            _activate_session_run_fence(session)
            owner = await store._claim_session_execution(
                session.id,
                token="writer-test",
                owner_id="owner",
                owner_kind="in_process_runner",
                owner_label=None,
                lease_seconds=30,
            )
            blocker = resources.own(sqlite3.connect(path), kind="connection")
            blocker.execute("BEGIN IMMEDIATE")
            entered = asyncio.Event()
            loop = asyncio.get_running_loop()

            def trace(statement):
                if statement == "BEGIN IMMEDIATE":
                    loop.call_soon_threadsafe(entered.set)

            store._connection.set_trace_callback(trace)
            task = resources.task(
                store._renew_session_execution(owner, lease_seconds=30, progress_kind=None)
            )
            try:
                await asyncio.wait_for(entered.wait(), 2)
                assert not task.done(), "SQLite writer lock blocked the event loop until timeout"
            finally:
                blocker.rollback()
                await task
                store._connection.set_trace_callback(None)

    asyncio.run(scenario())


@pytest.mark.parametrize("delta_count", [1, 1000])
def test_sqlite_owner_writes_follow_heartbeat_cadence_not_delta_count(
    sqlite_resources, delta_count
):
    async def scenario():
        async with sqlite_resources as resources:
            store = resources.own(SQLiteSessionStore(resources.path("write-rate.sqlite")))
            streamed, finish = asyncio.Event(), asyncio.Event()

            class Provider(VersionedFakeProvider):
                async def stream(self, request):
                    for _ in range(delta_count):
                        yield ModelStreamEvent.text_delta("x")
                    streamed.set()
                    await finish.wait()
                    yield ModelStreamEvent.completed({"finish_reason": "stop"})

            writes = 0

            def trace(statement):
                nonlocal writes
                sql = statement.upper()
                if sql.startswith(
                    (
                        "INSERT INTO CAYU_SESSION_EXECUTION_OWNERS",
                        "UPDATE CAYU_SESSION_EXECUTION_OWNERS",
                    )
                ):
                    writes += 1

            store._connection.set_trace_callback(trace)
            app = _app(store, Provider([]))
            started = time.monotonic()
            task = asyncio.create_task(
                _consume(
                    app.run(
                        RunRequest(
                            agent_name="assistant", messages=[Message.text("user", "stream")]
                        )
                    )
                )
            )
            try:
                await asyncio.wait_for(streamed.wait(), 30)
                await asyncio.sleep(0.2)
            finally:
                finish.set()
                events = await task
                groups = tuple(app._session_control.execution_presence.groups.values())
                await asyncio.gather(*(group.task for group in groups))
                store._connection.set_trace_callback(None)
            elapsed = time.monotonic() - started
            assert sum(event.type == EventType.MODEL_TEXT_DELTA for event in events) == delta_count
            assert writes >= 4  # claim, multiple actual renewals, release
            assert writes <= math.ceil(elapsed / 0.05) + 3

    asyncio.run(scenario())


def test_custom_events_do_not_break_execution_progress() -> None:
    from contextvars import copy_context

    from cayu.events import Event
    from cayu.sessions.execution import bind_execution_progress, note_execution_progress

    progress: dict = {"session-1": (0, "publishing")}

    def note_events() -> None:
        bind_execution_progress(progress)
        note_execution_progress(
            Event(type="custom.loop.before_stop.started", session_id="session-1")
        )
        assert progress["session-1"] == (0, "publishing")
        note_execution_progress(Event(type=EventType.MODEL_STARTED, session_id="session-1"))
        assert progress["session-1"] == (1, "model_stream")

    copy_context().run(note_events)


def test_cancellation_during_presence_release_remains_a_single_cancellation() -> None:
    class HeldReleaseStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1

        def __init__(self):
            super().__init__()
            self.release_started = asyncio.Event()
            self.allow_release = asyncio.Event()
            self.release_finished = asyncio.Event()

        async def _release_session_execution(self, expected):
            self.release_started.set()
            await self.allow_release.wait()
            await super()._release_session_execution(expected)
            self.release_finished.set()

    async def scenario():
        store = HeldReleaseStore()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(
            VersionedFakeProvider([ModelStreamEvent.completed({"finish_reason": "stop"})]),
            default=True,
        )
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))

        async def collect():
            return [
                event
                async for event in app.run(
                    RunRequest(agent_name="assistant", messages=[Message.text("user", "run")])
                )
            ]

        task = asyncio.create_task(collect())
        try:
            await asyncio.wait_for(store.release_started.wait(), timeout=5)
            task.cancel("caller stopped")
            store.allow_release.set()
            with pytest.raises(asyncio.CancelledError, match="caller stopped"):
                await asyncio.wait_for(task, timeout=5)
            assert store.release_finished.is_set()
            assert task.cancelled()
        finally:
            store.allow_release.set()
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(scenario())

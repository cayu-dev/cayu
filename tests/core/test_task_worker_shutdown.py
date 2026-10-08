"""Task workers leave queued work pending when their application shuts down."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from cayu.agents import AgentSpec
from cayu.applications import CayuApp
from cayu.evals.testing import ScriptedModelProvider
from cayu.messages import Message
from cayu.providers.base import ModelStreamEvent
from cayu.sessions.base import RunRequest
from cayu.storage.sqlite import SQLiteTaskStore
from cayu.tasks import TaskCreate, TaskStatus
from cayu.tasks.base import InMemoryTaskStore
from cayu.tasks.worker import run_task_worker


def _task_store(backend: str, tmp_path: Path):
    return InMemoryTaskStore() if backend == "memory" else SQLiteTaskStore(tmp_path / "tasks.db")


def _app(store) -> CayuApp:
    app = CayuApp(task_store=store, enable_logging=False)
    app.register_provider(
        ScriptedModelProvider(
            [
                ModelStreamEvent.text_delta("done"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ]
        ),
        default=True,
    )
    app.register_agent(AgentSpec(name="worker-agent", model="fake-model"))
    return app


async def _run_task(app: CayuApp, task, worker_id: str) -> None:
    request = RunRequest(
        agent_name="worker-agent",
        messages=[Message.text("user", task.title)],
        task_id=task.id,
        task_worker_id=worker_id,
        task_lease_expires_at=task.lease_expires_at,
    )
    async for _event in app.run(request):
        pass


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_a_worker_claims_nothing_from_a_sealed_app(backend: str, tmp_path: Path) -> None:
    async def scenario() -> None:
        store = _task_store(backend, tmp_path)
        app = _app(store)
        ids = [(await app.create_task(TaskCreate(type="demo", title=f"t{i}"))).id for i in range(5)]
        app.seal_admissions()
        handled = await asyncio.wait_for(
            run_task_worker(app, store, _run_task, worker_id="w", poll_interval_s=0.05), 10
        )
        assert handled == 0
        for task_id in ids:
            task = await store.load_task(task_id)
            assert task is not None and task.status is TaskStatus.PENDING

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_a_task_claimed_as_the_app_seals_returns_to_pending(backend: str, tmp_path: Path) -> None:
    async def scenario() -> None:
        store = _task_store(backend, tmp_path)
        app = _app(store)
        created = await app.create_task(TaskCreate(type="demo", title="claimed"))

        async def handler(app: CayuApp, task, worker_id: str) -> None:
            # Shutdown begins after the claim but before the run starts.
            app.seal_admissions()
            await _run_task(app, task, worker_id)

        handled = await asyncio.wait_for(
            run_task_worker(app, store, handler, worker_id="w", poll_interval_s=0.05), 10
        )
        assert handled == 1
        task = await store.load_task(created.id)
        assert task is not None and task.status is TaskStatus.PENDING
        assert task.worker_id is None
        # The next worker, on an open app, completes it.
        reopened = _app(store)
        assert (
            await asyncio.wait_for(
                run_task_worker(reopened, store, _run_task, worker_id="w2", max_tasks=1), 10
            )
            == 1
        )
        task = await store.load_task(created.id)
        assert task is not None and task.status is TaskStatus.COMPLETED

    asyncio.run(scenario())


def test_a_continuation_refused_by_shutdown_is_left_for_recovery() -> None:
    from tests.core.test_task_worker import _seed_receipt_backed_continuation

    from cayu.sessions.base import ResumeRequest
    from cayu.tasks.queries import TaskQuery

    async def scenario() -> None:
        store = InMemoryTaskStore()
        app = _app(store)

        async def fresh(*_args) -> None:
            raise AssertionError("no fresh task is queued")

        async def continuation(app: CayuApp, task, _worker_id: str) -> None:
            # Shutdown begins after the continuation was claimed.
            app.seal_admissions()
            request = ResumeRequest(
                session_id=task.session_id, messages=[Message.text("user", "continue")]
            )
            async for _event in app.resume(request):
                pass

        await _seed_receipt_backed_continuation(app, store, task_id="t1", session_id="s1")
        # The worker returns normally instead of failing on a release it cannot do.
        handled = await asyncio.wait_for(
            run_task_worker(
                app,
                store,
                fresh,
                worker_id="w",
                query=TaskQuery(type="job"),
                poll_interval_s=0.01,
                reclaim=False,
                recover_interrupted_handoffs=False,
                recovered_interrupted_task_handler=continuation,
                max_tasks=1,
            ),
            10,
        )
        assert handled == 1
        task = await store.load_task("t1")
        # Still attached and owned until its lease expires; nothing failed it.
        assert task is not None and task.session_id == "s1"
        assert task.status is TaskStatus.RUNNING

    asyncio.run(scenario())


class _RecordingStore(InMemoryTaskStore):
    supports_interrupted_task_handoffs = True
    verified_work_mutations_are_cancellation_quiescent = True

    def __init__(self) -> None:
        super().__init__()
        self.closed = False
        self.calls_after_close: list[str] = []
        self.in_maintenance = asyncio.Event()
        self.release_maintenance: asyncio.Event | None = None

    async def close(self) -> None:
        self.closed = True

    def _record(self, name: str) -> None:
        if self.closed:
            self.calls_after_close.append(name)

    async def list_expired_interrupted_task_handoff_candidates(self, **kwargs):
        self._record("list_expired_interrupted_task_handoff_candidates")
        return await super().list_expired_interrupted_task_handoff_candidates(**kwargs)

    async def reclaim_expired(self, **kwargs):
        self._record("reclaim_expired")
        if self.release_maintenance is not None:
            self.in_maintenance.set()
            await self.release_maintenance.wait()
        return await super().reclaim_expired(**kwargs)

    async def claim_task(self, *args, **kwargs):
        self._record("claim_task")
        return await super().claim_task(*args, **kwargs)


def test_a_worker_on_a_closed_app_makes_no_store_calls() -> None:
    async def scenario() -> None:
        store = _RecordingStore()
        app = CayuApp(task_store=store, enable_logging=False, owned_resources=(store,))
        assert (await app.aclose(timeout_s=2)).settled and store.closed
        # Maintenance is due on the first step, but the app is already closed.
        handled = await asyncio.wait_for(
            run_task_worker(app, store, _run_task, worker_id="w", poll_interval_s=0.05), 5
        )
        assert handled == 0
        assert store.calls_after_close == []

    asyncio.run(scenario())


def test_shutdown_waits_for_a_worker_step_already_in_maintenance() -> None:
    async def scenario() -> None:
        store = _RecordingStore()
        store.release_maintenance = asyncio.Event()
        app = CayuApp(task_store=store, enable_logging=False, owned_resources=(store,))
        worker = asyncio.create_task(
            run_task_worker(
                app, store, _run_task, worker_id="w", poll_interval_s=0.05, reclaim_every_s=0.01
            )
        )
        await asyncio.wait_for(store.in_maintenance.wait(), 5)
        outcome = await app.aclose(timeout_s=0.2)
        # The step is counted, so the owned store stays open under it.
        assert outcome.status == "incomplete" and outcome.open_operations == 1
        assert not store.closed
        store.release_maintenance.set()
        assert await asyncio.wait_for(worker, 5) == 0
        assert (await app.aclose(timeout_s=2)).settled and store.closed
        assert store.calls_after_close == []

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_a_requested_cancellation_still_settles_when_shutdown_refuses_the_run(
    backend: str, tmp_path: Path
) -> None:
    async def scenario() -> None:
        store = _task_store(backend, tmp_path)
        app = _app(store)
        created = await app.create_task(TaskCreate(type="demo", title="cancelled"))

        async def handler(app: CayuApp, task, worker_id: str) -> None:
            await store.cancel_task(task.id)
            app.seal_admissions()
            await _run_task(app, task, worker_id)

        handled = await asyncio.wait_for(
            run_task_worker(app, store, handler, worker_id="w", poll_interval_s=0.05), 10
        )
        assert handled == 1
        task = await store.load_task(created.id)
        assert task is not None and task.status is TaskStatus.CANCELLED

    asyncio.run(scenario())


def test_a_refusal_from_another_closed_app_is_an_ordinary_failure() -> None:
    async def scenario() -> None:
        store = InMemoryTaskStore()
        app = _app(store)
        other = _app(InMemoryTaskStore())
        await other.aclose(timeout_s=1)
        created = await app.create_task(TaskCreate(type="demo", title="foreign"))
        calls = 0

        async def handler(_app: CayuApp, task, worker_id: str) -> None:
            nonlocal calls
            calls += 1
            await _run_task(other, task, worker_id)

        handled = await asyncio.wait_for(
            run_task_worker(app, store, handler, worker_id="w", poll_interval_s=0.05, max_tasks=1),
            10,
        )
        # Failed once and not requeued: the worker's own app is still open.
        assert handled == 1 and calls == 1
        task = await store.load_task(created.id)
        assert task is not None and task.status is TaskStatus.FAILED
        assert app.lifecycle_state == "open"

    asyncio.run(scenario())

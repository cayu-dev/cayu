"""Wake continuations must remain subject to the worker handler's hard time limit."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import pytest
from tests.core.test_sleep_until import (
    _Clock,
    _sleep_then_answer,
    _sleeping,
    _StableScriptedProvider,
    _tool_results,
)
from tests.core.test_sleep_until_tasks import _parked_job, _run_job, _stores, _wake_due

from cayu import AgentSpec, CayuApp
from cayu.tasks import run_task_worker
from cayu.tools import SleepUntilTool


@pytest.mark.anyio
@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("task_owned", [False, True])
async def test_wake_handler_timeout_interrupts_its_resumed_provider(backend, task_owned, tmp_path):
    entered, interrupted, release = asyncio.Event(), asyncio.Event(), asyncio.Event()

    class CooperativeProvider(_StableScriptedProvider):
        async def stream(self, request):
            if _tool_results(request):
                entered.set()
                try:
                    await release.wait()
                except asyncio.CancelledError:
                    interrupted.set()
                    raise
            async for event in super().stream(request):
                yield event

    clock = _Clock(datetime.now(UTC))
    sessions, store = _stores(backend, tmp_path, None, clock=clock)
    app = CayuApp(session_store=sessions, task_store=store, clock=clock, enable_logging=False)
    app.register_provider(
        CooperativeProvider(response_factory=_sleep_then_answer({"duration_seconds": 1})),
        default=True,
    )
    app.register_agent(AgentSpec(name="sleeper", model="scripted-model"), tools=[SleepUntilTool()])
    if task_owned:
        # The ordinary worker parks its task on the session's sleep. The wake
        # handler will then claim that task while continuing the session.
        _job, wake_id = await _parked_job(app, "sleeping-job")
    else:
        # Control: the same sleeping session without the nested owning-task claim.
        pending = await _sleeping(app, sessions, "sleeping-session")
        wake_id = pending["input_id"]
    await _wake_due(app, wake_id, clock=clock)
    worker = asyncio.create_task(
        run_task_worker(
            app,
            store,
            _run_job,
            worker_id="wake-worker",
            max_tasks=1,
            reclaim=False,
            recover_interrupted_handoffs=False,
            execution_timeout_s=3.0,
            execution_grace_s=0.0,
        )
    )
    try:
        await asyncio.wait_for(entered.wait(), 5)
        done, _ = await asyncio.wait({worker}, timeout=5)
        assert (interrupted.is_set(), worker in done) == (True, True), (
            "A built-in wake handler exceeded its hard timeout without interrupting "
            "the cancellation-cooperative provider it resumed or releasing its slot"
        )
    finally:
        release.set()
        await asyncio.wait_for(asyncio.gather(worker, return_exceptions=True), 10)
        await app.aclose()
        if backend == "sqlite":
            await sessions.close()
            await store.close()


@pytest.mark.anyio
@pytest.mark.parametrize("restartable", [False, True])
async def test_worker_shutdown_does_not_interrupt_timed_out_handler_cleanup(restartable):
    from cayu.tasks import InMemoryTaskStore, TaskCreate

    store = InMemoryTaskStore()
    app = CayuApp(task_store=store, enable_logging=False)
    await store.create_task(TaskCreate(type="job", restartable=restartable))
    cleanup_started = asyncio.Event()
    release_cleanup = asyncio.Event()
    cleanup_completed = asyncio.Event()
    cleanup_interrupted = asyncio.Event()

    async def handler(*args):
        try:
            await asyncio.Event().wait()
        finally:
            cleanup_started.set()
            try:
                await release_cleanup.wait()
                cleanup_completed.set()
            except asyncio.CancelledError:
                cleanup_interrupted.set()
                raise

    worker = asyncio.create_task(
        run_task_worker(
            app,
            store,
            handler,
            worker_id="owner",
            max_tasks=1,
            execution_timeout_s=0.1,
            execution_grace_s=0,
            reclaim=False,
            recover_interrupted_handoffs=False,
        )
    )
    try:
        await asyncio.wait_for(cleanup_started.wait(), 5)
        worker.cancel()
        # The shutdown must keep draining instead of aborting this cleanup.
        done, _ = await asyncio.wait({worker}, timeout=0.1)
        assert not done and not cleanup_interrupted.is_set()
        release_cleanup.set()
        await asyncio.wait_for(asyncio.gather(worker, return_exceptions=True), 5)
        assert cleanup_completed.is_set() and not cleanup_interrupted.is_set()
    finally:
        release_cleanup.set()
        await asyncio.wait_for(asyncio.gather(worker, return_exceptions=True), 5)
        await app.aclose()


@pytest.mark.anyio
@pytest.mark.parametrize("interrupt_on_cancel", [False, True])
async def test_enclosing_timeout_preserves_restartable_handler_cleanup(interrupt_on_cancel):
    from cayu.deadlines import ExecutionDeadlineExceeded, HandlerTimeLimit
    from cayu.tasks import InMemoryTaskStore, TaskCreate, TaskStatus

    store = InMemoryTaskStore()
    app = CayuApp(task_store=store, enable_logging=False)
    task = await store.create_task(TaskCreate(type="job", restartable=True))
    cleanup_started, release_cleanup = asyncio.Event(), asyncio.Event()
    cleanup_completed, cleanup_interrupted = asyncio.Event(), asyncio.Event()

    async def handler(*args):
        try:
            await asyncio.Event().wait()
        finally:
            cleanup_started.set()
            try:
                await release_cleanup.wait()
                cleanup_completed.set()
            except asyncio.CancelledError:
                cleanup_interrupted.set()
                raise

    async def run():
        async with HandlerTimeLimit(0.1, grace_s=0):
            await run_task_worker(
                app,
                store,
                handler,
                worker_id="owner",
                max_tasks=1,
                reclaim=False,
                recover_interrupted_handoffs=False,
                interrupt_on_cancel=interrupt_on_cancel,
            )

    worker = asyncio.create_task(run())
    try:
        await asyncio.wait_for(cleanup_started.wait(), 5)
        done, _ = await asyncio.wait({worker}, timeout=0.1)
        assert not done and not cleanup_interrupted.is_set()
        release_cleanup.set()
        with pytest.raises(ExecutionDeadlineExceeded):
            await asyncio.wait_for(worker, 5)
        assert cleanup_completed.is_set() and not cleanup_interrupted.is_set()
        assert (await store.load_task(task.id)).status is TaskStatus.PENDING
    finally:
        release_cleanup.set()
        await asyncio.wait_for(asyncio.gather(worker, return_exceptions=True), 5)
        await app.aclose()


@pytest.mark.anyio
@pytest.mark.parametrize("enclosing_timeout", [False, True])
async def test_worker_cancellation_survives_an_internal_timeout(enclosing_timeout):
    from cayu.deadlines import ExecutionDeadlineExceeded, HandlerTimeLimit
    from cayu.tasks import InMemoryTaskStore, TaskCreate, TaskStatus

    store = InMemoryTaskStore()
    app = CayuApp(task_store=store, enable_logging=False)
    task = await store.create_task(TaskCreate(type="job", restartable=True))
    cleanup_started = asyncio.Event()
    release_cleanup = asyncio.Event()
    handler_tasks = []

    async def handler(*args):
        handler_tasks.append(asyncio.current_task())
        while True:
            try:
                async with asyncio.timeout(0.01):
                    try:
                        await asyncio.Event().wait()
                    finally:
                        cleanup_started.set()
                        await release_cleanup.wait()
            except TimeoutError:
                continue

    async def work():
        await run_task_worker(
            app,
            store,
            handler,
            worker_id="owner",
            max_tasks=1,
            reclaim=False,
            recover_interrupted_handoffs=False,
        )

    async def run():
        if enclosing_timeout:
            async with HandlerTimeLimit(0.5, grace_s=0):
                await work()
        else:
            await work()

    worker = asyncio.create_task(run())
    try:
        await asyncio.wait_for(cleanup_started.wait(), 5)
        if not enclosing_timeout:
            worker.cancel()
        done, _ = await asyncio.wait({worker}, timeout=2)
        assert worker in done, "Worker cancellation was consumed by an internal timeout"
        expected = ExecutionDeadlineExceeded if enclosing_timeout else asyncio.CancelledError
        with pytest.raises(expected):
            await worker
        assert (await store.load_task(task.id)).status is TaskStatus.PENDING
    finally:
        for running in handler_tasks:
            if not running.done():
                running.cancel()
        release_cleanup.set()
        await asyncio.wait_for(asyncio.gather(worker, return_exceptions=True), 5)
        await app.aclose()

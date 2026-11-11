"""Operator cancellation must interrupt a wake's resumed handler, not only its supervisor."""

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
from cayu.tasks import (
    TaskScheduleCancelRequest,
    TaskStatus,
    run_task_worker,
    session_wake_attempts,
)
from cayu.tools import SleepUntilTool


@pytest.mark.anyio
@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("task_owned", [False, True])
async def test_cancelling_active_wake_interrupts_the_resumed_provider(
    backend, task_owned, tmp_path
):
    entered = asyncio.Event()
    interrupted = asyncio.Event()
    release = asyncio.Event()

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
        _job, wake_id = await _parked_job(app, "sleeping-job")
    else:
        pending = await _sleeping(app, sessions, "sleeping-session")
        wake_id = pending["input_id"]
    await _wake_due(app, wake_id, clock=clock)
    [attempt] = await session_wake_attempts(app, wake_id)
    worker = asyncio.create_task(
        run_task_worker(
            app,
            store,
            _run_job,
            worker_id="wake-worker",
            max_tasks=1,
            reclaim=False,
            recover_interrupted_handoffs=False,
            # No handler timeout: this is an operator cancellation, not deadline expiry.
            execution_grace_s=0.0,
        )
    )
    try:
        await asyncio.wait_for(entered.wait(), 10)
        active = await store.load_task(attempt.id)
        assert active.schedule is not None
        await store.cancel_scheduled_task(
            TaskScheduleCancelRequest(
                task_id=attempt.id,
                operation_id="operator-stop-wake",
                expected_revision=active.schedule.revision,
            )
        )
        requested = await store.load_task(attempt.id)
        assert requested.status_reason == "cancellation_requested"
        # Default leases renew once a second. Even with zero grace, the nested
        # continuation supervisor must forward the interrupt to its provider.
        done, _ = await asyncio.wait({worker}, timeout=4)
        assert (interrupted.is_set(), worker in done) == (True, True), (
            "The cancelled wake's supervisor kept draining its live nested provider "
            "instead of delivering the operator's interrupt"
        )
        assert worker.result() == 1
        assert (await store.load_task(attempt.id)).status is TaskStatus.CANCELLED
    finally:
        release.set()
        await asyncio.wait_for(asyncio.gather(worker, return_exceptions=True), 15)
        await app.aclose()
        if backend == "sqlite":
            await sessions.close()
            await store.close()

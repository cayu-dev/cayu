"""External notifications and durable interruptions obey native lifecycle boundaries."""

from __future__ import annotations

import asyncio

import pytest
from tests.core.test_external_agents import ScriptedAdapter, pricing, request

from cayu import CayuApp, EventType
from cayu.budgets import BudgetLimit, BudgetWindow
from cayu.external_agents import ExternalAgentError, ExternalAgentEvent, run_external_agent
from cayu.sessions import InterruptSessionRequest
from cayu.sessions.records import SessionStatus
from cayu.storage import SQLiteSessionStore, SQLiteTaskStore
from cayu.tasks import InMemoryTaskStore, TaskCreate, TaskQuery, TaskStatus, run_task_worker


def _app(backend, tmp_path):
    return CayuApp(
        session_store=None if backend == "memory" else SQLiteSessionStore(tmp_path / "sessions.db"),
        task_store=InMemoryTaskStore()
        if backend == "memory"
        else SQLiteTaskStore(tmp_path / "tasks.db"),
        enable_logging=False,
    )


@pytest.mark.store_semantics
@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("scope", ["session", "run", "app", "agent", "causal"])
def test_notify_budget_emits_once_for_each_threshold_scope(backend, scope, tmp_path):
    async def scenario():
        app = _app(backend, tmp_path)
        try:
            notifications = []
            for _ in range(2):
                task = (
                    await app.create_task(TaskCreate(type="vendor")) if scope == "causal" else None
                )
                limits = tuple(
                    BudgetLimit.usd(
                        amount,
                        pricing=pricing(),
                        action="notify",
                        scope=scope,
                        key="vendor-agent" if scope == "agent" else task.id if task else None,
                    )
                    for amount in ("1", "2")
                )
                adapter = ScriptedAdapter(
                    [
                        ExternalAgentEvent("usage", {"input_tokens": 2_000_000}, "first"),
                        ExternalAgentEvent("usage", {"input_tokens": 1_000_000}, "second"),
                        ExternalAgentEvent("result", {"output": "done"}, "result"),
                    ]
                )
                result = await run_external_agent(
                    app,
                    request(budget_limits=limits, task_id=None if task is None else task.id),
                    adapter,
                    prompt="review",
                )
                assert result.output == "done" and adapter.closed
                assert (
                    await app.task_store.load_task(result.task_id)
                ).status is TaskStatus.COMPLETED
                notifications.extend(
                    event
                    for event in await app.session_store.load_events(result.session_id)
                    if event.type is EventType.BUDGET_LIMIT_REACHED
                )
            assert len(notifications) == (2 if scope in {"app", "agent"} else 4)
            assert all(event.payload["action"] == "notify" for event in notifications)
            assert all(event.payload["limit_reached"] for event in notifications)
        finally:
            await app.aclose()

    asyncio.run(scenario())


@pytest.mark.store_semantics
@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_notify_budget_can_emit_again_after_its_window_expires(backend, tmp_path):
    async def scenario():
        app = _app(backend, tmp_path)
        limit = BudgetLimit.usd(
            "1",
            pricing=pricing(),
            action="notify",
            scope="app",
            window=BudgetWindow.rolling(seconds=1),
        )
        try:
            for _ in range(2):
                adapter = ScriptedAdapter(
                    [
                        ExternalAgentEvent("usage", {"input_tokens": 2_000_000}, "first"),
                        ExternalAgentEvent("result", {"output": "done"}, "result"),
                    ]
                )
                result = await run_external_agent(
                    app, request(budget_limits=(limit,)), adapter, prompt="review"
                )
                events = await app.session_store.load_events(result.session_id)
                assert sum(event.type is EventType.BUDGET_LIMIT_REACHED for event in events) == 1
                await asyncio.sleep(1.05)
        finally:
            await app.aclose()

    asyncio.run(scenario())


@pytest.mark.store_semantics
@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("gateway", [False, True])
@pytest.mark.parametrize("worker", [False, True])
def test_remote_interruption_wins_before_the_monitor_tick(
    backend, gateway, worker, tmp_path, monkeypatch
):
    async def scenario():
        app = _app(backend, tmp_path)
        remote = CayuApp(
            session_store=app.session_store, task_store=app.task_store, enable_logging=False
        )
        original_sleep = asyncio.sleep

        async def defer_monitor(delay, *args, **kwargs):
            if asyncio.current_task().get_coro().__name__ == "observe_interrupt":
                await asyncio.Event().wait()
            else:
                await original_sleep(delay, *args, **kwargs)

        monkeypatch.setattr(asyncio, "sleep", defer_monitor)
        sessions, closed = [], []

        class Adapter(ScriptedAdapter):
            async def stream(self, context):
                sessions.append(context.session_id)
                try:
                    if len(sessions) == 1:
                        if gateway:
                            context.claim_model_usage()
                        async for _ in remote.interrupt_session(
                            InterruptSessionRequest(session_id=context.session_id)
                        ):
                            pass
                    yield ExternalAgentEvent("result", {"output": "done"}, "result")
                finally:
                    closed.append(context.session_id)

        async def handle(current_app, task, worker_id):
            await run_external_agent(
                current_app,
                request(
                    task_id=task.id,
                    task_worker_id=worker_id,
                    task_lease_expires_at=task.lease_expires_at,
                ),
                Adapter([]),
                prompt="review",
            )

        try:
            if worker:
                first = await app.create_task(TaskCreate(type="vendor", restartable=True))
                second = await app.create_task(TaskCreate(type="vendor"))
                await run_task_worker(app, app.task_store, handle, worker_id="worker", max_tasks=2)
                assert (await app.task_store.load_task(second.id)).status is TaskStatus.COMPLETED
            else:
                with pytest.raises(ExternalAgentError, match="interrupted by its operator"):
                    await run_external_agent(app, request(), Adapter([]), prompt="review")
                (first,) = await app.task_store.list_tasks(TaskQuery())
            cancelled = await app.task_store.load_task(first.id)
            assert cancelled.status is TaskStatus.CANCELLED
            assert cancelled.restart_count == 0
            assert (await app.session_store.load(sessions[0])).status is SessionStatus.INTERRUPTED
            assert closed == sessions
        finally:
            await app.aclose()

    asyncio.run(scenario())

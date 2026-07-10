"""Tests for the generic ``run_task_worker`` durable-worker helper."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from cayu import (
    AgentSpec,
    CayuApp,
    EventType,
    Message,
    ModelStreamEvent,
    ResumeRequest,
    RunRequest,
    ScriptedModelProvider,
    SQLiteSessionStore,
    SQLiteTaskStore,
    Task,
    TaskCreate,
    TaskQuery,
    run_task_worker,
)
from cayu.runtime import SessionStatus
from cayu.runtime.sessions import SessionIdentity


def _build(tmp_path: Path) -> tuple[CayuApp, SQLiteTaskStore]:
    store = SQLiteTaskStore(tmp_path / "tasks.sqlite")
    app = CayuApp(task_store=store)
    app.register_provider(
        ScriptedModelProvider(
            [
                [
                    ModelStreamEvent.text_delta("done"),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ]
            ]
        ),
        default=True,
    )
    app.register_agent(AgentSpec(name="worker-agent", model="scripted-model"))
    return app, store


async def _run_handler(app: CayuApp, task: Task, worker_id: str) -> None:
    async for _event in app.run(
        RunRequest(
            agent_name="worker-agent",
            session_id=f"sess-{task.id}",
            task_id=task.id,
            task_worker_id=worker_id,
            messages=[Message.text("user", "go")],
        )
    ):
        pass


def test_run_task_worker_claims_runs_and_completes_a_task(tmp_path: Path) -> None:
    app, store = _build(tmp_path)

    async def scenario() -> tuple[int, Task | None]:
        created = await store.create_task(
            TaskCreate(type="job", assigned_agent_name="worker-agent")
        )
        handled = await run_task_worker(
            app,
            store,
            _run_handler,
            worker_id="w1",
            query=TaskQuery(type="job"),
            max_tasks=1,
            poll_interval_s=0.05,
            reclaim=False,
        )
        return handled, await store.load_task(created.id)

    handled, task = asyncio.run(scenario())
    assert handled == 1
    assert task is not None
    assert task.status == "completed"


def test_run_task_worker_returns_immediately_when_stopped(tmp_path: Path) -> None:
    app, store = _build(tmp_path)

    async def scenario() -> int:
        stop = asyncio.Event()
        stop.set()
        return await run_task_worker(
            app,
            store,
            _run_handler,
            worker_id="w1",
            query=TaskQuery(type="job"),
            reclaim=False,
            stop=stop,
        )

    assert asyncio.run(scenario()) == 0


def test_run_task_worker_rejects_negative_max_tasks(tmp_path: Path) -> None:
    app, store = _build(tmp_path)

    async def scenario() -> None:
        await run_task_worker(
            app,
            store,
            _run_handler,
            worker_id="w1",
            max_tasks=-1,
        )

    with pytest.raises(ValueError, match="max_tasks must be non-negative"):
        asyncio.run(scenario())


def test_run_task_worker_fails_task_when_handler_leaves_it_active(tmp_path: Path) -> None:
    app, store = _build(tmp_path)

    async def no_terminal_state(_app: CayuApp, _task: Task, _worker_id: str) -> None:
        return None

    async def scenario() -> Task | None:
        created = await store.create_task(
            TaskCreate(type="job", assigned_agent_name="worker-agent")
        )
        handled = await run_task_worker(
            app,
            store,
            no_terminal_state,
            worker_id="w1",
            query=TaskQuery(type="job"),
            max_tasks=1,
            poll_interval_s=0.05,
            reclaim=False,
        )
        assert handled == 1
        return await store.load_task(created.id)

    task = asyncio.run(scenario())
    assert task is not None
    assert task.status == "failed"
    assert task.error == {
        "error": "RuntimeError",
        "message": "Task handler returned without completing or failing the task.",
    }


def test_resume_completes_the_running_task_already_attached_to_the_session(
    tmp_path: Path,
) -> None:
    session_store = SQLiteSessionStore(tmp_path / "sessions.sqlite")
    task_store = SQLiteTaskStore(tmp_path / "tasks.sqlite")
    provider = ScriptedModelProvider(
        [
            [
                ModelStreamEvent.text_delta("resumed"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ]
        ]
    )
    app = CayuApp(
        session_store=session_store,
        task_store=task_store,
        enable_logging=False,
    )
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="worker-agent", model="scripted-model"))

    async def scenario():
        task = await task_store.create_task(TaskCreate(task_id="task-resume", type="job"))
        await session_store.create(
            RunRequest(
                agent_name="worker-agent",
                session_id="session-resume",
                task_id=task.id,
                messages=[Message.text("user", "original")],
            ),
            identity=SessionIdentity(
                provider_name=provider.name,
                model="scripted-model",
            ),
        )
        await session_store.update_status("session-resume", SessionStatus.INTERRUPTED)
        await task_store.start_task(task.id, session_id="session-resume")

        events = [
            event
            async for event in app.resume(
                ResumeRequest(
                    session_id="session-resume",
                    messages=[Message.text("user", "continue")],
                )
            )
        ]
        return await task_store.load_task(task.id), events

    task, events = asyncio.run(scenario())

    assert task is not None
    assert task.status == "completed"
    assert task.session_id == "session-resume"
    assert [event.type for event in events][-3:] == [
        EventType.TASK_COMPLETED,
        EventType.TURN_COMPLETED,
        EventType.SESSION_COMPLETED,
    ]


def test_resume_rejects_multiple_running_tasks_attached_to_the_same_session(
    tmp_path: Path,
) -> None:
    session_store = SQLiteSessionStore(tmp_path / "sessions.sqlite")
    task_store = SQLiteTaskStore(tmp_path / "tasks.sqlite")
    provider = ScriptedModelProvider([])
    app = CayuApp(
        session_store=session_store,
        task_store=task_store,
        enable_logging=False,
    )
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="worker-agent", model="scripted-model"))

    async def scenario() -> None:
        await session_store.create(
            RunRequest(
                agent_name="worker-agent",
                session_id="session-ambiguous-tasks",
                messages=[Message.text("user", "original")],
            ),
            identity=SessionIdentity(
                provider_name=provider.name,
                model="scripted-model",
            ),
        )
        await session_store.update_status(
            "session-ambiguous-tasks",
            SessionStatus.INTERRUPTED,
        )
        for task_id in ("task-ambiguous-a", "task-ambiguous-b"):
            await task_store.create_task(TaskCreate(task_id=task_id, type="job"))
            await task_store.start_task(task_id, session_id="session-ambiguous-tasks")

        async for _ in app.resume(
            ResumeRequest(
                session_id="session-ambiguous-tasks",
                messages=[Message.text("user", "continue")],
            )
        ):
            pass

    with pytest.raises(
        RuntimeError,
        match="Session has multiple running tasks attached: session-ambiguous-tasks",
    ):
        asyncio.run(scenario())

    session = asyncio.run(session_store.load("session-ambiguous-tasks"))
    assert session is not None
    assert session.status == SessionStatus.INTERRUPTED
    assert provider.requests == []

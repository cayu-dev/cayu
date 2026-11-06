"""Closing after interaction completion must finish successful task settlement."""

import asyncio

import pytest

from cayu import (
    AgentSpec,
    CayuApp,
    EventType,
    Message,
    ResumeRequest,
    RunRequest,
    SessionStatus,
)
from cayu.evals import ScriptedModelProvider
from cayu.providers import ModelStreamEvent
from cayu.sessions import InMemorySessionStore
from cayu.storage.sqlite import SQLiteSessionStore
from cayu.tasks.creation import TaskCreate
from cayu.tasks.memory import InMemoryTaskStore
from cayu.tasks.records import TaskStatus


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("close_at_completion", [False, True], ids=["consume-all", "close"])
def test_completed_interaction_close_finishes_attached_task(tmp_path, backend, close_at_completion):
    async def scenario():
        store = (
            InMemorySessionStore()
            if backend == "memory"
            else SQLiteSessionStore(tmp_path / "completed-close.sqlite")
        )
        tasks = InMemoryTaskStore()
        provider = ScriptedModelProvider(
            [
                [
                    ModelStreamEvent.text_delta("done"),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ]
            ]
        )
        app = CayuApp(session_store=store, task_store=tasks, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        await tasks.create_task(
            TaskCreate(task_id="task", type="respond", assigned_agent_name="assistant")
        )
        stream = app.run(
            RunRequest(
                agent_name="assistant",
                session_id="completed-close",
                task_id="task",
                messages=[Message.text("user", "answer")],
            )
        )
        try:
            async for event in stream:
                if close_at_completion and event.type is EventType.INTERACTION_COMPLETED:
                    break
        finally:
            await stream.aclose()
        try:
            assert await app.drain_recovery_cleanups(timeout_s=5)
            assert await app.drain_environment_cleanups(timeout_s=5)
            session = await store.load("completed-close")
            assert session.status is SessionStatus.COMPLETED
            task = await tasks.load_task("task")
            events = await store.load_events("completed-close")
            completed_events = [
                event.type for event in events if event.type is EventType.SESSION_COMPLETED
            ]
            assert (task.status, completed_events) == (
                TaskStatus.COMPLETED,
                [EventType.SESSION_COMPLETED],
            )
        finally:
            await app.aclose()
            if backend == "sqlite":
                await store.close()

    asyncio.run(scenario())


_COMPLETION_EVENTS = (
    EventType.INTERACTION_COMPLETED,
    EventType.TASK_COMPLETED,
    EventType.TURN_COMPLETED,
    EventType.SESSION_COMPLETED,
)


@pytest.mark.parametrize("close_after", _COMPLETION_EVENTS, ids=str)
def test_close_at_any_completion_event_finishes_the_completion(close_after):
    async def scenario():
        store = InMemorySessionStore()
        tasks = InMemoryTaskStore()
        app = CayuApp(session_store=store, task_store=tasks, enable_logging=False)
        app.register_provider(
            ScriptedModelProvider(
                [
                    [ModelStreamEvent.text_delta("done"), ModelStreamEvent.completed()],
                    [ModelStreamEvent.text_delta("again"), ModelStreamEvent.completed()],
                ]
            ),
            default=True,
        )
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        await tasks.create_task(
            TaskCreate(task_id="task", type="respond", assigned_agent_name="assistant")
        )
        stream = app.run(
            RunRequest(
                agent_name="assistant",
                session_id="close-point",
                task_id="task",
                messages=[Message.text("user", "answer")],
            )
        )
        try:
            async for event in stream:
                if event.type is close_after:
                    break
        finally:
            await stream.aclose()
        try:
            assert await app.drain_recovery_cleanups(timeout_s=5)
            assert await app.drain_environment_cleanups(timeout_s=5)
            session = await store.load("close-point")
            task = await tasks.load_task("task")
            types = [event.type for event in await store.load_events("close-point")]
            assert session.status is SessionStatus.COMPLETED
            assert task.status is TaskStatus.COMPLETED
            assert types.count(EventType.SESSION_COMPLETED) == 1
            assert types.count(EventType.TASK_COMPLETED) == 1
            # The happy-path order holds even when settlement finishes it.
            assert (
                types.index(EventType.INTERACTION_COMPLETED)
                < types.index(EventType.TASK_COMPLETED)
                < types.index(EventType.SESSION_COMPLETED)
            )
            # The fence was released: the session takes another turn.
            resumed = [
                event
                async for event in app.resume(
                    ResumeRequest(session_id="close-point", messages=[Message.text("user", "more")])
                )
            ]
            assert resumed[-1].type is EventType.SESSION_COMPLETED
        finally:
            await app.aclose()

    asyncio.run(scenario())

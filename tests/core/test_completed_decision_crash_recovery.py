"""A crash after a session completes must not strand its task or its run fence.

The fault-injection scenarios run here over the in-memory and SQLite stores: the
process dies right before the attached task is terminalized, or right before the
completed run releases its fence. A fresh app's recovery must then complete the
task (settling a task-group member's execution with it), publish the terminal
events once and leave the session able to run again. A crash after a member's
live terminalization still leaves its group execution owed; that is recorded in
the fault-injection known failures.
"""

from __future__ import annotations

import asyncio
import itertools
import logging
from datetime import timedelta
from typing import Any

import pytest
from tests.fault_injection import explore
from tests.fault_injection.harness import (
    FaultKind,
    Injector,
    ManualClock,
    SimulatedCrash,
    _instrumented_store_type,
)

from cayu.agents import AgentSpec
from cayu.applications import CayuApp
from cayu.evals.testing import ScriptedModelProvider
from cayu.events import EventType
from cayu.messages import Message
from cayu.providers import ModelStreamEvent
from cayu.runtime._incomplete_session_recovery import IncompleteSessionRecovery
from cayu.sessions.checkpoints import INVOCATION_TERMINAL_DECISION_CHECKPOINT_KEY
from cayu.sessions.memory import InMemorySessionStore
from cayu.sessions.records import SessionStatus
from cayu.sessions.requests import ResumeRequest, RunRequest
from cayu.storage.sqlite import SQLiteSessionStore
from cayu.storage.tasks_sqlite import SQLiteTaskStore
from cayu.tasks import worker as task_worker
from cayu.tasks.creation import TaskCreate
from cayu.tasks.memory import InMemoryTaskStore
from cayu.tasks.records import TaskStatus

_SQLiteSessionStore = _instrumented_store_type(SQLiteSessionStore)
_SQLiteTaskStore = _instrumented_store_type(SQLiteTaskStore)

_RELEASE = "store.release_session_invocation:enter#0"
_TERMINALIZE = "store.terminalize_task:enter#0"


@pytest.fixture(params=["memory", "sqlite"])
def backend(request: pytest.FixtureRequest, tmp_path, monkeypatch) -> str:
    if request.param == "sqlite":
        names = itertools.count()

        def session_store(injector: Injector, clock: ManualClock) -> SQLiteSessionStore:
            store = _SQLiteSessionStore(
                tmp_path / f"sessions-{next(names)}.sqlite", ownership_clock=clock
            )
            store.fault_injector = injector
            return store

        def task_store(injector: Injector, clock: ManualClock) -> SQLiteTaskStore:
            store = _SQLiteTaskStore(
                tmp_path / f"tasks-{next(names)}.sqlite", clock=clock, ownership_clock=clock
            )
            store.fault_injector = injector
            return store

        monkeypatch.setattr(explore, "session_store", session_store)
        monkeypatch.setattr(explore, "task_store", task_store)
    return request.param


@pytest.mark.parametrize(
    ("scenario", "point"),
    [
        ("run", _RELEASE),
        ("tool-round", _RELEASE),
        ("resume", _RELEASE),
        ("approval", _RELEASE),
        ("task", _TERMINALIZE),
        ("task-group", _TERMINALIZE),
    ],
)
def test_recovery_finishes_a_completion_cut_short_by_a_crash(
    backend: str, scenario: str, point: str
) -> None:
    result = explore.run_point(explore.SCENARIOS[scenario], FaultKind.CRASH, point)

    assert result.reached
    assert not result.violations, "\n".join(violation.describe() for violation in result.violations)


def test_task_worker_finishes_a_completion_without_the_sweep(backend: str, monkeypatch) -> None:
    async def no_sweep(self, *, inactive_for_seconds: int) -> list:
        return []

    # Only the next worker's expired-owner scan can find the completed session.
    monkeypatch.setattr(IncompleteSessionRecovery, "recover_committed_completions", no_sweep)

    result = explore.run_point(explore.SCENARIOS["task"], FaultKind.CRASH, _TERMINALIZE)

    assert result.reached
    assert not result.violations, "\n".join(violation.describe() for violation in result.violations)


def test_sweep_skips_completed_turns_when_the_store_cannot_list_them(caplog) -> None:
    class CustomStore(InMemorySessionStore):
        async def list_sessions_with_unsettled_invocation_terminal_decision(
            self, query: Any = None
        ) -> Any:
            raise NotImplementedError

    async def sweep_twice() -> None:
        app = CayuApp(session_store=CustomStore(), enable_logging=False)
        assert await app.recover_abandoned_sessions(inactive_for_seconds=0) == ()
        assert await app.recover_abandoned_sessions(inactive_for_seconds=0) == ()

    with caplog.at_level(logging.WARNING):
        asyncio.run(sweep_twice())

    assert (
        sum(
            "cannot list unsettled terminal decisions" in record.message
            for record in caplog.records
        )
        == 1
    )


async def _owed_task_completion() -> tuple[Any, Any, Any]:
    """A completed task run whose task step failed: the session owes its task."""

    clock = ManualClock()
    store = InMemorySessionStore(ownership_clock=clock)
    tasks = InMemoryTaskStore(clock=clock, ownership_clock=clock)
    await tasks.create_task(TaskCreate(task_id="task", type="respond"))
    task = await tasks.claim_task("worker-a", lease_seconds=60)
    assert task is not None
    original = InMemoryTaskStore.terminalize_task

    async def refuse(self, request):
        raise OSError("task store unavailable")

    InMemoryTaskStore.terminalize_task = refuse
    try:
        app = CayuApp(session_store=store, task_store=tasks, enable_logging=False)
        app.register_provider(
            ScriptedModelProvider(
                [[ModelStreamEvent.text_delta("done"), ModelStreamEvent.completed({})]] * 2
            ),
            default=True,
        )
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        try:
            async for _ in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="session",
                    task_id=task.id,
                    task_worker_id="worker-a",
                    task_lease_expires_at=task.lease_expires_at,
                    messages=[Message.text("user", "answer")],
                )
            ):
                pass
        except OSError:
            pass
        assert await app.drain_recovery_cleanups(timeout_s=5)
        await app.aclose()
    finally:
        InMemoryTaskStore.terminalize_task = original
    assert INVOCATION_TERMINAL_DECISION_CHECKPOINT_KEY in (await store.load_checkpoint("session"))
    return clock, store, tasks


async def _recover(store: Any, tasks: Any, inactive_for_seconds: int = 3600) -> CayuApp:
    app = CayuApp(session_store=store, task_store=tasks, enable_logging=False)
    app.register_provider(
        ScriptedModelProvider(
            [[ModelStreamEvent.text_delta("again"), ModelStreamEvent.completed({})]]
        ),
        default=True,
    )
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))
    await app.recover_abandoned_sessions(inactive_for_seconds=inactive_for_seconds)
    session = await store.load("session")
    checkpoint = await store.load_checkpoint("session")
    assert session.status is SessionStatus.COMPLETED
    assert INVOCATION_TERMINAL_DECISION_CHECKPOINT_KEY not in checkpoint
    types = [event.type for event in await store.load_events("session")]
    assert types.count(EventType.SESSION_COMPLETED) == 1
    return app


def test_recovery_finishes_a_completed_session_whose_task_ended_otherwise() -> None:
    async def scenario() -> None:
        clock, store, tasks = await _owed_task_completion()
        running = await tasks.load_task("task")
        await tasks.fail_task(
            "task",
            {"error": "failed elsewhere"},
            worker_id="worker-a",
            claim_generation=running.claim_generation,
        )
        clock.advance(timedelta(days=1))
        await _recover(store, tasks)
        assert (await tasks.load_task("task")).status is TaskStatus.FAILED

    asyncio.run(scenario())


def test_pending_cancellation_of_a_completed_task_run_is_settled() -> None:
    async def scenario() -> None:
        clock, store, tasks = await _owed_task_completion()
        await tasks.cancel_task("task", {"code": "operator"})
        clock.advance(timedelta(days=1))
        app = await _recover(store, tasks)
        assert (await tasks.load_task("task")).status is TaskStatus.RUNNING
        await task_worker._recover_expired_interrupted_task_handoffs(
            app, tasks, worker_id="worker-b", after=None, limit=10, stop=None
        )
        assert (await tasks.load_task("task")).status is TaskStatus.CANCELLED

    asyncio.run(scenario())


def test_recovery_without_a_task_store_leaves_the_task_completion_owed() -> None:
    async def scenario() -> None:
        clock, store, tasks = await _owed_task_completion()
        clock.advance(timedelta(days=1))
        without_tasks = CayuApp(session_store=store, enable_logging=False)
        without_tasks.register_agent(AgentSpec(name="assistant", model="fake-model"))
        await without_tasks.recover_abandoned_sessions(inactive_for_seconds=0)
        assert INVOCATION_TERMINAL_DECISION_CHECKPOINT_KEY in (
            await store.load_checkpoint("session")
        )
        await _recover(store, tasks)
        assert (await tasks.load_task("task")).status is TaskStatus.COMPLETED

    asyncio.run(scenario())


def test_one_sweep_finishes_an_owed_completion_despite_its_own_writes() -> None:
    async def scenario() -> None:
        clock = ManualClock()
        store = InMemorySessionStore(ownership_clock=clock)
        original = InMemorySessionStore.append_event

        async def refuse_turn(self, session_id, event):
            if event.type is EventType.TURN_COMPLETED:
                raise OSError("turn completion store failure")
            await original(self, session_id, event)

        InMemorySessionStore.append_event = refuse_turn
        try:
            app = CayuApp(session_store=store, enable_logging=False)
            app.register_provider(
                ScriptedModelProvider(
                    [[ModelStreamEvent.text_delta("done"), ModelStreamEvent.completed({})]]
                ),
                default=True,
            )
            app.register_agent(AgentSpec(name="assistant", model="fake-model"))
            try:
                async for _ in app.run(
                    RunRequest(
                        agent_name="assistant",
                        session_id="session",
                        messages=[Message.text("user", "answer")],
                    )
                ):
                    pass
            except OSError:
                pass
            assert await app.drain_recovery_cleanups(timeout_s=5)
            await app.aclose()
        finally:
            InMemorySessionStore.append_event = original
        clock.advance(timedelta(hours=2))
        await _recover(store, None)

    asyncio.run(scenario())


def test_crash_after_a_cancelled_task_runs_completion_is_finished_by_recovery() -> None:
    """A pending cancellation decides the task; the committed completion still finishes."""

    async def scenario() -> None:
        clock = ManualClock()
        store = InMemorySessionStore(ownership_clock=clock)
        tasks = InMemoryTaskStore(clock=clock, ownership_clock=clock)
        await tasks.create_task(TaskCreate(task_id="task", type="respond"))
        task = await tasks.claim_task("worker-a", lease_seconds=60)
        assert task is not None
        original = InMemoryTaskStore.terminalize_task

        async def crash(self, request):
            raise SimulatedCrash("terminalize_task")

        InMemoryTaskStore.terminalize_task = crash
        try:
            app = CayuApp(session_store=store, task_store=tasks, enable_logging=False)
            app.register_provider(
                ScriptedModelProvider(
                    [[ModelStreamEvent.text_delta("done"), ModelStreamEvent.completed({})]]
                ),
                default=True,
            )
            app.register_agent(AgentSpec(name="assistant", model="fake-model"))
            await tasks.cancel_task("task", {"code": "operator"})
            with pytest.raises(BaseException):
                async for _ in app.run(
                    RunRequest(
                        agent_name="assistant",
                        session_id="session",
                        task_id=task.id,
                        task_worker_id="worker-a",
                        task_lease_expires_at=task.lease_expires_at,
                        messages=[Message.text("user", "answer")],
                    )
                ):
                    pass
        finally:
            InMemoryTaskStore.terminalize_task = original
        assert (await store.load("session")).status is SessionStatus.COMPLETED
        assert INVOCATION_TERMINAL_DECISION_CHECKPOINT_KEY in await store.load_checkpoint("session")
        clock.advance(timedelta(days=1))
        recovery = await _recover(store, tasks)
        await task_worker._recover_expired_interrupted_task_handoffs(
            recovery, tasks, worker_id="worker-b", after=None, limit=10, stop=None
        )
        assert (await tasks.load_task("task")).status is TaskStatus.CANCELLED
        resumed = [
            event
            async for event in recovery.resume(
                ResumeRequest(session_id="session", messages=[Message.text("user", "again")])
            )
        ]
        assert resumed[-1].type is EventType.SESSION_COMPLETED

    asyncio.run(scenario())

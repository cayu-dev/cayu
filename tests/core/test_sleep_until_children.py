"""``sleep_until`` in child (subagent) sessions."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime

import pytest
from tests.core.test_foreground_subagent_recovery import _identity, _Provider
from tests.core.test_sleep_until import _Clock, _stores

from cayu import AgentSpec, CayuApp, EventType, Message, RunRequest
from cayu.approvals.user_input import UserInputResponse
from cayu.messages import ToolResultPart
from cayu.providers.base import ModelRequest, ModelStreamEvent
from cayu.sessions.memory import InMemorySessionStore
from cayu.sessions.queries import SessionQuery
from cayu.sessions.records import SessionStatus
from cayu.sessions.requests import InterruptSessionRequest
from cayu.storage import SQLiteSessionStore, SQLiteTaskStore
from cayu.tasks import (
    InMemoryTaskStore,
    TaskCreate,
    TaskHandlerOutcome,
    TaskStatus,
    run_task_worker,
    run_task_worker_until_idle,
    session_wakes,
)
from cayu.tasks.handoff import TaskParkedWait
from cayu.tasks.session_wakes import SessionWakeOutcome, deliver_session_wake
from cayu.tools import SleepUntilTool, UserInputTool
from cayu.tools.subagents import SubagentSpec, SubagentTool


def _spawn(call_id: str = "spawn") -> list[ModelStreamEvent]:
    return [
        ModelStreamEvent.tool_call(
            id=call_id, name="subagent", arguments={"agent": "child", "task": "wait"}
        ),
        ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
    ]


def _sleep(seconds: float = 1, call_id: str = "call_sleep") -> list[ModelStreamEvent]:
    return [
        ModelStreamEvent.tool_call(
            id=call_id, name="sleep_until", arguments={"duration_seconds": seconds}
        ),
        ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
    ]


def _text(text: str) -> list[ModelStreamEvent]:
    return [ModelStreamEvent.text_delta(text), ModelStreamEvent.completed()]


def _family(
    sessions, tasks, clock, provider, *, mode: str = "foreground", child_tools=None
) -> CayuApp:
    app = CayuApp(session_store=sessions, task_store=tasks, enable_logging=False, clock=clock)
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="parent", model="test"),
        tools=[
            SubagentTool(
                app,
                agents={"child": SubagentSpec(agent_name="child", mode=mode)},
                execution_profile_identity=_identity(f"sleeping-child-{mode}"),
            )
        ],
    )
    app.register_agent(
        AgentSpec(name="child", model="test"),
        tools=[SleepUntilTool()] if child_tools is None else child_tools,
    )
    return app


async def _delegate(app: CayuApp, parent_id: str) -> list:
    return [
        event
        async for event in app.run(
            RunRequest(
                session_id=parent_id,
                agent_name="parent",
                messages=[Message.text("user", "delegate")],
            )
        )
    ]


async def _child(sessions, parent_id: str):
    [child] = (await sessions.list_sessions(SessionQuery(parent_session_id=parent_id))).sessions
    return child


def _results(transcript, call_id: str) -> list[ToolResultPart]:
    return [
        part
        for message in transcript
        for part in message.content
        if isinstance(part, ToolResultPart) and part.tool_call_id == call_id
    ]


async def _waits_on_sleeping_child(sessions, parent_id: str, child_id: str) -> dict:
    """The parent waits on its child's wake; return the child's open wake."""

    assert (await sessions.load(child_id)).status is SessionStatus.INTERRUPTED
    wake = (await sessions.load_checkpoint(child_id))["pending_wake"]["continuation"]
    [stopped] = [
        event
        for event in await sessions.load_events(parent_id)
        if event.type is EventType.SESSION_INTERRUPTED
    ][-1:]
    assert stopped.payload["interruption_type"] == "waiting_on_child_action"
    assert stopped.payload["action_kind"] == "wake"
    assert stopped.payload["child_session_id"] == child_id
    assert stopped.payload["action_id"] == wake["input_id"]
    return wake


async def _settled(sessions, session_id: str, status: SessionStatus) -> None:
    for _ in range(100):
        if (await sessions.load(session_id)).status is status:
            return
        await asyncio.sleep(0.05)
    events = await sessions.load_events(session_id)
    raise AssertionError([(event.type, event.payload.get("error")) for event in events][-5:])


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
def test_a_foreground_child_sleeps_and_its_parent_waits_then_both_continue_once(
    backend, tmp_path, request
) -> None:
    async def scenario() -> None:
        clock = _Clock(datetime.now(UTC))
        sessions, tasks = _stores(backend, tmp_path, request, clock)
        provider = _Provider([_spawn(), _sleep(), _text("child woke"), _text("parent done")])
        app = _family(sessions, tasks, clock, provider)
        parent_id = f"p_fg_{backend}"
        try:
            events = await _delegate(app, parent_id)
            assert events[-1].type is EventType.SESSION_INTERRUPTED
            child = await _child(sessions, parent_id)
            wake = await _waits_on_sleeping_child(sessions, parent_id, child.id)
            # The parent's call is still running: no result reached its transcript.
            assert _results(await sessions.load_transcript(parent_id), "spawn") == []
            assert len(provider.requests) == 2

            await asyncio.sleep(1.1)  # Durable stores keep their own clock.
            clock.advance(1.1)
            assert (
                await deliver_session_wake(app, session_id=child.id, wake_id=wake["input_id"])
                is SessionWakeOutcome.CONTINUED
            )
            await _settled(sessions, parent_id, SessionStatus.COMPLETED)
            assert (await sessions.load(child.id)).status is SessionStatus.COMPLETED
            assert len(provider.requests) == 4
            [timer] = _results(await sessions.load_transcript(child.id), "call_sleep")
            assert json.loads(timer.content)["status"] == "woke"
            [spawned] = _results(await sessions.load_transcript(parent_id), "spawn")
            assert spawned.is_error is False and "child woke" in spawned.content
            completed = [
                event
                for event in await sessions.load_events(parent_id)
                if event.type is EventType.SESSION_COMPLETED
            ]
            assert len(completed) == 1
            assert await deliver_session_wake(
                app, session_id=child.id, wake_id=wake["input_id"]
            ) in {SessionWakeOutcome.ALREADY_DELIVERED, SessionWakeOutcome.CLOSED}
            assert len(provider.requests) == 4
        finally:
            await app.aclose()

    asyncio.run(scenario())


async def _wake(app: CayuApp, clock: _Clock, session_id: str, wake_id: str) -> SessionWakeOutcome:
    await asyncio.sleep(1.1)  # Durable stores keep their own clock.
    clock.advance(1.1)
    return await deliver_session_wake(app, session_id=session_id, wake_id=wake_id)


def test_a_sleeping_grandchild_holds_both_ancestors_and_wakes_them_leaf_first() -> None:
    async def scenario() -> None:
        clock = _Clock(datetime.now(UTC))
        sessions, tasks = InMemorySessionStore(), InMemoryTaskStore(clock=clock)
        provider = _Provider(
            [_spawn(), _spawn("spawn_inner"), _sleep(), _text("c"), _text("b"), _text("a")]
        )
        app = CayuApp(session_store=sessions, task_store=tasks, enable_logging=False, clock=clock)
        app.register_provider(provider, default=True)
        for name, child in (("parent", "middle"), ("middle", "leaf")):
            app.register_agent(
                AgentSpec(name=name, model="test"),
                tools=[
                    SubagentTool(
                        app,
                        agents={"child": SubagentSpec(agent_name=child)},
                        execution_profile_identity=_identity(f"nested-{name}"),
                    )
                ],
            )
        app.register_agent(AgentSpec(name="leaf", model="test"), tools=[SleepUntilTool()])
        try:
            await _delegate(app, "p_nested")
            middle = await _child(sessions, "p_nested")
            leaf = await _child(sessions, middle.id)
            wake = await _waits_on_sleeping_child(sessions, middle.id, leaf.id)
            [stopped] = [
                event
                for event in await sessions.load_events("p_nested")
                if event.type is EventType.SESSION_INTERRUPTED
            ]
            assert stopped.payload["action_kind"] == "delegated_action"
            assert await _wake(app, clock, leaf.id, wake["input_id"]) is (
                SessionWakeOutcome.CONTINUED
            )
            await _settled(sessions, "p_nested", SessionStatus.COMPLETED)
            assert len(provider.requests) == 6
            assert "b" in _results(await sessions.load_transcript("p_nested"), "spawn")[0].content
        finally:
            await app.aclose()

    asyncio.run(scenario())


def test_a_woken_child_that_sleeps_again_or_asks_refreshes_its_parents_wait() -> None:
    async def scenario() -> None:
        clock = _Clock(datetime.now(UTC))
        sessions, tasks = InMemorySessionStore(), InMemoryTaskStore(clock=clock)
        provider = _Provider(
            [
                _spawn(),
                _sleep(),
                _sleep(call_id="call_sleep_again"),
                [
                    ModelStreamEvent.tool_call(
                        id="call_ask", name="ask_user", arguments={"question": "Which?"}
                    ),
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ],
                _text("child done"),
                _text("parent done"),
            ]
        )
        app = _family(
            sessions, tasks, clock, provider, child_tools=[SleepUntilTool(), UserInputTool()]
        )
        try:
            await _delegate(app, "p_again")
            child = await _child(sessions, "p_again")
            first = await _waits_on_sleeping_child(sessions, "p_again", child.id)
            assert await _wake(app, clock, child.id, first["input_id"]) is (
                SessionWakeOutcome.CONTINUED
            )
            second = (await sessions.load_checkpoint(child.id))["pending_wake"]["continuation"]
            assert second["input_id"] != first["input_id"]
            refreshed = [
                event
                for event in await sessions.load_events("p_again")
                if event.type is EventType.SESSION_DELEGATED_ACTION_UPDATED
            ]
            assert [event.payload["action_kind"] for event in refreshed] == ["wake"]
            assert refreshed[-1].payload["action_id"] == second["input_id"]
            assert (await sessions.load("p_again")).status is SessionStatus.INTERRUPTED
            assert await _wake(app, clock, child.id, second["input_id"]) is (
                SessionWakeOutcome.CONTINUED
            )
            asked = next(
                event
                for event in await sessions.load_events(child.id)
                if event.type is EventType.SESSION_AWAITING_USER_INPUT
            )
            refreshed = [
                event
                for event in await sessions.load_events("p_again")
                if event.type is EventType.SESSION_DELEGATED_ACTION_UPDATED
            ]
            assert [event.payload["action_kind"] for event in refreshed] == ["wake", "user_input"]
            async for _ in app.resolve_user_input(
                UserInputResponse(
                    session_id=app.project_session_id_for_exposure(child.id),
                    input_id=asked.payload["input_id"],
                    answer="this",
                )
            ):
                pass
            await _settled(sessions, "p_again", SessionStatus.COMPLETED)
            assert len(provider.requests) == 6
        finally:
            await app.aclose()

    asyncio.run(scenario())


def test_stopping_a_parent_or_its_sleeping_child_ends_the_wait_without_reviving_it() -> None:
    async def scenario() -> None:
        clock = _Clock(datetime.now(UTC))
        sessions, tasks = InMemorySessionStore(), InMemoryTaskStore(clock=clock)
        provider = _Provider(
            [_spawn(), _sleep(), _text("child done"), _spawn(), _sleep(), _text("parent after")]
        )
        app = _family(sessions, tasks, clock, provider)
        try:
            # Stopping the parent: its interaction closes; the woken child ends, and
            # the late delivery does not revive the parent.
            await _delegate(app, "p_stop")
            child = await _child(sessions, "p_stop")
            wake = await _waits_on_sleeping_child(sessions, "p_stop", child.id)
            async for _ in app.interrupt_session(
                InterruptSessionRequest(session_id="p_stop", reason="operator stop")
            ):
                pass
            assert await _wake(app, clock, child.id, wake["input_id"]) is (
                SessionWakeOutcome.CONTINUED
            )
            await _settled(sessions, child.id, SessionStatus.COMPLETED)
            await asyncio.sleep(0.2)
            assert (await sessions.load("p_stop")).status is SessionStatus.INTERRUPTED
            assert len(provider.requests) == 3

            # Stopping the sleeping child: its wake is cancelled and the parent's call
            # receives the child's interrupted result.
            await _delegate(app, "p_stop_child")
            child = await _child(sessions, "p_stop_child")
            wake = await _waits_on_sleeping_child(sessions, "p_stop_child", child.id)
            async for _ in app.interrupt_session(
                InterruptSessionRequest(
                    session_id=app.project_session_id_for_exposure(child.id), reason="stop"
                )
            ):
                pass
            # As for a child stopped while it asks a person, the delivery the stop
            # deferred is retried by the side-effect sweep.
            await app.recover_persisted_event_side_effects()
            await _settled(sessions, "p_stop_child", SessionStatus.COMPLETED)
            [spawned] = _results(await sessions.load_transcript("p_stop_child"), "spawn")
            assert spawned.is_error is True
            [attempt] = await session_wakes.session_wake_attempts(app, wake["input_id"])
            assert attempt.status is TaskStatus.CANCELLED
        finally:
            await app.aclose()

    asyncio.run(scenario())


def _by_role(
    request: ModelRequest, *, nested: bool = False, seconds: float = 1
) -> list[ModelStreamEvent]:
    """Parent spawns then finishes; the child sleeps then reports, in any order.

    Nested, the parent's child spawns the sleeping child and reports after it.
    """

    user = next(
        part.text
        for message in request.messages
        if message.role == "user"
        for part in message.content
        if hasattr(part, "text")
    )
    answered = any(
        isinstance(part, ToolResultPart) for message in request.messages for part in message.content
    )
    if user == "delegate":
        return _text("parent done") if answered else _spawn()
    if user == "wait" and nested:
        return _text("middle done") if answered else _spawn_inner()
    return _text("child woke") if answered else _sleep(seconds)


def _spawn_inner() -> list[ModelStreamEvent]:
    return [
        ModelStreamEvent.tool_call(
            id="spawn_inner", name="subagent", arguments={"agent": "child", "task": "inner"}
        ),
        ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
    ]


async def _spawned_background_sleeper(clock: _Clock):
    """A finished parent and its background subagent asleep; return the app, child, wake."""

    from tests.core.test_sleep_until import _StableScriptedProvider

    sessions, tasks = InMemorySessionStore(), InMemoryTaskStore(clock=clock)
    app = CayuApp(session_store=sessions, task_store=tasks, enable_logging=False, clock=clock)
    app.register_provider(_StableScriptedProvider(response_factory=_by_role), default=True)
    app.register_agent(
        AgentSpec(name="parent", model="test"),
        tools=[
            SubagentTool(
                app,
                agents={"child": SubagentSpec(agent_name="child", mode="background")},
                execution_profile_identity=_identity("background-sleeper"),
            )
        ],
    )
    app.register_agent(AgentSpec(name="child", model="test"), tools=[SleepUntilTool()])
    await _delegate(app, "p_background_sleeper")
    child = await _child(sessions, "p_background_sleeper")
    await _settled(sessions, child.id, SessionStatus.INTERRUPTED)
    wake = (await sessions.load_checkpoint(child.id))["pending_wake"]["continuation"]
    return app, child, wake


@pytest.mark.parametrize("mode", ["background", "durable"])
def test_an_asynchronous_child_that_sleeps_is_reported_sleeping_not_failed(mode) -> None:
    from tests.core.test_sleep_until import _StableScriptedProvider

    from cayu.tasks import TaskStoreDispatcher
    from cayu.tools.subagents import _wait_for_subagent_terminal, project_terminal_subagent_result

    async def scenario() -> None:
        clock = _Clock(datetime.now(UTC))
        sessions, tasks = InMemorySessionStore(), InMemoryTaskStore(clock=clock)
        app = CayuApp(
            session_store=sessions,
            task_store=tasks,
            dispatcher=TaskStoreDispatcher(tasks) if mode == "durable" else None,
            enable_logging=False,
            clock=clock,
        )
        app.register_provider(_StableScriptedProvider(response_factory=_by_role), default=True)
        app.register_agent(
            AgentSpec(name="parent", model="test"),
            tools=[
                SubagentTool(
                    app,
                    agents={"child": SubagentSpec(agent_name="child", mode=mode)},
                    execution_profile_identity=_identity(f"async-sleeper-{mode}"),
                )
            ],
        )
        app.register_agent(AgentSpec(name="child", model="test"), tools=[SleepUntilTool()])
        try:
            events = await _delegate(app, f"p_{mode}")
            assert events[-1].type is EventType.SESSION_COMPLETED
            if mode == "durable":
                assert await app.dispatcher.process_next(app, worker_id="queue") is not None
            child = await _child(sessions, f"p_{mode}")
            await _settled(sessions, child.id, SessionStatus.INTERRUPTED)
            wake = (await sessions.load_checkpoint(child.id))["pending_wake"]["continuation"]

            summary = await project_terminal_subagent_result(sessions, child.id, task_store=tasks)
            assert summary["status"] == "sleeping"
            assert summary["retrieval_status"] == "not_ready"
            assert summary["is_error"] is False
            assert summary["wake_at"] == wake["wake"]["deadline"].replace("+00:00", "Z")
            # Waiting on it waits like on a running child, until the timeout.
            waited = await _wait_for_subagent_terminal(
                sessions, await sessions.load(child.id), wait=True, timeout_s=0.2, task_store=tasks
            )
            assert waited.status is SessionStatus.INTERRUPTED

            assert await _wake(app, clock, child.id, wake["input_id"]) is (
                SessionWakeOutcome.CONTINUED
            )
            summary = await project_terminal_subagent_result(sessions, child.id, task_store=tasks)
            assert summary["status"] == "completed" and summary["retrieval_status"] == "ready"
            assert summary["result_text"] == "child woke" and "wake_at" not in summary
        finally:
            await app.aclose()

    asyncio.run(scenario())


_JOB = "job"
# Longer than any test takes, so a task-owned sleep comes due only when a test says so.
_OWNED_SLEEP_SECONDS = 600


def _owned_family(
    sessions,
    tasks,
    clock: _Clock,
    calls: list | None = None,
    *,
    nested: bool = False,
    parent_policy=None,
    script=None,
) -> CayuApp:
    from tests.core.test_sleep_until import _StableScriptedProvider

    def respond(request: ModelRequest):
        if calls is not None:
            calls.append(request)
        if script is not None:
            return script(request)
        return _by_role(request, nested=nested, seconds=_OWNED_SLEEP_SECONDS)

    app = CayuApp(session_store=sessions, task_store=tasks, enable_logging=False, clock=clock)
    app.register_provider(_StableScriptedProvider(response_factory=respond), default=True)
    app.register_agent(
        AgentSpec(name="parent", model="test"),
        tools=[
            SubagentTool(
                app,
                agents={"child": SubagentSpec(agent_name="child")},
                execution_profile_identity=_identity("task-owned-parent"),
            )
        ],
        tool_policy=parent_policy,
    )
    if nested:
        app.register_agent(
            AgentSpec(name="child", model="test"),
            tools=[
                SubagentTool(
                    app,
                    agents={"child": SubagentSpec(agent_name="leaf")},
                    execution_profile_identity=_identity("task-owned-middle"),
                )
            ],
        )
    app.register_agent(
        AgentSpec(name="leaf" if nested else "child", model="test"), tools=[SleepUntilTool()]
    )
    return app


def _parent_of(task_id: str) -> str:
    return f"p-{task_id}"


async def _run_parent(app: CayuApp, task, worker_id: str):
    """The documented worker handler: run the parent session under the task."""

    if task.type != _JOB:
        raise AssertionError(f"Unexpected task {task.type}.")
    async for _ in app.run(
        RunRequest(
            agent_name="parent",
            session_id=_parent_of(task.id),
            task_id=task.id,
            task_worker_id=worker_id,
            task_lease_expires_at=task.lease_expires_at,
            messages=[Message.text("user", "delegate")],
        )
    ):
        pass
    session = await app.session_store.load(_parent_of(task.id))
    if session is not None and session.status is SessionStatus.INTERRUPTED:
        return TaskHandlerOutcome.SESSION_INTERRUPTED
    return None


async def _never_continued(app, task, worker_id):
    raise AssertionError(f"A generic continuation claimed parked task {task.id}.")


async def _until_idle(app: CayuApp, worker_id: str = "idle-worker"):
    return await run_task_worker_until_idle(
        app,
        app.task_store,
        _run_parent,
        worker_id=worker_id,
        recovered_interrupted_task_handler=_never_continued,
    )


async def _parked_parent(app: CayuApp, task_id: str):
    """Run the job until its parent waits on its sleeping child; return both and the wake."""

    # Earlier tests in this module share a Postgres database: settle their spent wakes.
    await _until_idle(app, "drain")
    await app.task_store.create_task(TaskCreate(task_id=task_id, type=_JOB))
    idle = await _until_idle(app, "first-worker")
    assert idle.handled == 1 and idle.continuations_pending is False
    child = await _child(app.session_store, _parent_of(task_id))
    wake = await _waits_on_sleeping_child(app.session_store, _parent_of(task_id), child.id)
    return await app.task_store.load_task(task_id), child, wake


def _record_completions(tasks, monkeypatch) -> dict[str, list[str | None]]:
    """Record which worker each task completion presented; ``None`` when it held no claim."""

    from cayu.runtime import _session_task_lifecycle as lifecycle

    completions: dict[str, list[str | None]] = {}
    complete = tasks.complete_task
    terminalize = lifecycle._terminalize_claimed_task

    async def unclaimed(task_id, result, **kwargs):
        completions.setdefault(task_id, []).append(kwargs.get("worker_id"))
        return await complete(task_id, result, **kwargs)

    async def claimed(store, request):
        completions.setdefault(request.task_id, []).append(request.worker_id)
        return await terminalize(store, request)

    tasks.complete_task = unclaimed
    monkeypatch.setattr(lifecycle, "_terminalize_claimed_task", claimed)
    return completions


def _due(clock: _Clock) -> None:
    """Make a task-owned family's sleep due on the clock its app and task store share."""

    clock.advance(_OWNED_SLEEP_SECONDS + 1)


def _clocked_stores(backend: str, tmp_path, request):
    """Stores whose wake tasks come due on an injected clock, never on the wall clock."""

    clock = _Clock(datetime.now(UTC))
    if backend == "memory":
        return clock, InMemorySessionStore(), InMemoryTaskStore(clock=clock)
    if backend == "sqlite":
        return (
            clock,
            SQLiteSessionStore(tmp_path / "sessions.db"),
            SQLiteTaskStore(tmp_path / "tasks.db", clock=clock),
        )
    from cayu.storage import PostgresSessionStore
    from cayu.storage.migrations import SchemaMode
    from cayu.storage.tasks_postgres import PostgresTaskStore

    dsn = request.getfixturevalue("postgres_dsn")
    return (
        clock,
        PostgresSessionStore(dsn, schema_mode=SchemaMode.CREATE),
        PostgresTaskStore(dsn, schema_mode=SchemaMode.CREATE, clock=clock),
    )


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
def test_a_task_owned_parent_parks_on_its_childs_wake_and_continues_under_the_task(
    backend, tmp_path, request, monkeypatch
) -> None:
    async def scenario() -> None:
        clock, sessions, tasks = _clocked_stores(backend, tmp_path, request)
        app = _owned_family(sessions, tasks, clock)
        completions = _record_completions(tasks, monkeypatch)
        try:
            job, child, wake = await _parked_parent(app, f"job-child-{backend}")
            assert job.status is TaskStatus.RUNNING
            assert job.worker_id is None and job.lease_expires_at is None
            receipt = await tasks.load_interrupted_task_handoff_receipt(
                job.id, job.interrupted_handoff_id
            )
            assert receipt.request.parked == TaskParkedWait(
                kind="session_wake", wait_id=wake["input_id"], session_id=child.id
            )
            # Before the deadline nothing is ready, and no generic scan claims the task.
            idle = await _until_idle(app)
            assert idle.handled == 0 and idle.continuations_pending is False

            _due(clock)
            idle = await _until_idle(app)
            assert idle.handled == 1 and idle.continuations_pending is False
            assert (await sessions.load(child.id)).status is SessionStatus.COMPLETED
            assert (await sessions.load(job.session_id)).status is SessionStatus.COMPLETED
            [spawned] = _results(await sessions.load_transcript(job.session_id), "spawn")
            assert "child woke" in spawned.content
            assert (await tasks.load_task(job.id)).status is TaskStatus.COMPLETED
            # The wake continued the parent under its own claim on the task.
            assert completions[job.id] == [f"idle-worker:continue:{wake['input_id']}"]
        finally:
            await app.aclose()

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_a_parent_left_waiting_after_its_child_woke_is_continued_by_the_next_attempt(
    backend, tmp_path, request, monkeypatch
) -> None:
    async def scenario() -> None:
        clock, sessions, tasks = _clocked_stores(backend, tmp_path, request)
        app = _owned_family(sessions, tasks, clock)
        completions = _record_completions(tasks, monkeypatch)
        try:
            job, child, wake = await _parked_parent(app, f"job-split-{backend}")
            _due(clock)
            # A delivery that holds no claim on the task, as after a crash between the
            # child's continuation and its parent's: the child ends, the parent waits.
            assert await session_wakes._deliver(
                app, session_id=child.id, wake_id=wake["input_id"]
            ) is (SessionWakeOutcome.CONTINUED)
            assert (await sessions.load(child.id)).status is SessionStatus.COMPLETED
            assert (await sessions.load(job.session_id)).status is SessionStatus.INTERRUPTED
            parked = await tasks.load_task(job.id)
            assert parked.worker_id is None
            receipt = await tasks.load_interrupted_task_handoff_receipt(
                job.id, parked.interrupted_handoff_id
            )
            assert receipt.request.parked.wait_id == wake["input_id"]
            assert await session_wakes.reconcile_parked_handoffs(app) == 0
            # Its due attempt is still live, so reconciliation adds none.
            assert len(await session_wakes.session_wake_attempts(app, wake["input_id"])) == 1

            idle = await _until_idle(app)
            assert idle.handled == 1
            assert (await sessions.load(job.session_id)).status is SessionStatus.COMPLETED
            assert (await tasks.load_task(job.id)).status is TaskStatus.COMPLETED
            assert completions[job.id] == [f"idle-worker:continue:{wake['input_id']}"]
            [spawned] = _results(await sessions.load_transcript(job.session_id), "spawn")
            assert "child woke" in spawned.content
        finally:
            await app.aclose()

    asyncio.run(scenario())


def test_stopping_the_sleeping_child_of_a_task_owned_parent_unparks_its_task() -> None:
    async def scenario() -> None:
        clock, sessions, tasks = _clocked_stores("memory", None, None)
        app = _owned_family(sessions, tasks, clock)
        try:
            job, child, wake = await _parked_parent(app, "job-stop-child")
            async for _ in app.interrupt_session(
                InterruptSessionRequest(
                    session_id=app.project_session_id_for_exposure(child.id), reason="stop"
                )
            ):
                pass
            [attempt] = await session_wakes.session_wake_attempts(app, wake["input_id"])
            assert attempt.status is TaskStatus.CANCELLED
            assert await session_wakes.reconcile_parked_handoffs(app) == 1
            released = await tasks.load_task(job.id)
            assert released.status is TaskStatus.RUNNING and released.worker_id is None
            receipt = await tasks.load_interrupted_task_handoff_receipt(
                job.id, released.interrupted_handoff_id
            )
            assert receipt.request.parked is None
        finally:
            await app.aclose()

    asyncio.run(scenario())


def test_two_postgres_workers_racing_at_the_deadline_continue_the_child_and_parent_once(
    request,
) -> None:
    async def scenario() -> None:
        calls: list[ModelRequest] = []
        clock, sessions, tasks = _clocked_stores("postgres", None, request)
        app = _owned_family(sessions, tasks, clock, calls=calls)
        _, peer_sessions, peer_tasks = _clocked_stores("postgres", None, request)
        peer = _owned_family(peer_sessions, peer_tasks, clock, calls=calls)
        try:
            job, child, wake = await _parked_parent(app, "job-child-race")
            _due(clock)
            stop = asyncio.Event()

            async def worker(owner: CayuApp, name: str) -> int:
                return await run_task_worker(
                    owner,
                    owner.task_store,
                    _run_parent,
                    worker_id=name,
                    poll_interval_s=0.01,
                    stop=stop,
                    recovered_interrupted_task_handler=_never_continued,
                )

            async def duplicate(owner: CayuApp):
                try:
                    return await deliver_session_wake(
                        owner, session_id=child.id, wake_id=wake["input_id"]
                    )
                except Exception as error:
                    return type(error).__name__

            racers = [
                asyncio.create_task(worker(app, "racer-a")),
                asyncio.create_task(worker(peer, "racer-b")),
            ]
            duplicates = await asyncio.gather(duplicate(app), duplicate(peer))
            for _ in range(200):
                if (await tasks.load_task(job.id)).status is TaskStatus.COMPLETED:
                    break
                await asyncio.sleep(0.05)
            stop.set()
            await asyncio.gather(*racers)

            assert (await tasks.load_task(job.id)).status is TaskStatus.COMPLETED
            assert len(_results(await sessions.load_transcript(child.id), "call_sleep")) == 1
            assert len(_results(await sessions.load_transcript(job.session_id), "spawn")) == 1
            # Parent spawn, child sleep, child after the wake, parent after the child.
            assert len(calls) == 4
            assert duplicates.count(SessionWakeOutcome.CONTINUED) <= 1
        finally:
            await app.aclose()
            await peer.aclose()

    asyncio.run(asyncio.wait_for(scenario(), 60))


def test_a_woken_child_still_running_is_observed_as_the_same_wake() -> None:
    from cayu.runtime._foreground_child_wait import observe_foreground_child_wait
    from cayu.sessions._foreground_child_checkpoint import foreground_child_state_from_checkpoint

    release = asyncio.Event()
    started = asyncio.Event()

    class _HeldAfterWake(_Provider):
        async def stream(self, request):
            if len(self.requests) == 2:
                started.set()
                await release.wait()
            async for event in super().stream(request):
                yield event

    async def scenario() -> None:
        clock = _Clock(datetime.now(UTC))
        sessions, tasks = InMemorySessionStore(), InMemoryTaskStore(clock=clock)
        provider = _HeldAfterWake([_spawn(), _sleep(), _text("child woke"), _text("parent")])
        app = _family(sessions, tasks, clock, provider)
        try:
            await _delegate(app, "p_running")
            child = await _child(sessions, "p_running")
            wake = await _waits_on_sleeping_child(sessions, "p_running", child.id)
            wait, _ = foreground_child_state_from_checkpoint(
                await sessions.load_checkpoint("p_running")
            )
            delivery = asyncio.create_task(_wake(app, clock, child.id, wake["input_id"]))
            await asyncio.wait_for(started.wait(), 10)
            assert (await sessions.load(child.id)).status is SessionStatus.RUNNING
            tool = app._agent_registry.registrations["parent"].executable_tool("subagent")
            observed = await observe_foreground_child_wait(
                sessions,
                parent=await sessions.load("p_running"),
                intent=wait.parent_effect,
                matcher=tool.child_session_recovery,
                arguments={"agent": "child", "task": "wait"},
            )
            assert observed is not None
            assert (observed.child_action_kind, observed.child_action_id) == (
                "wake",
                wake["input_id"],
            )
            release.set()
            assert await delivery is SessionWakeOutcome.CONTINUED
            await _settled(sessions, "p_running", SessionStatus.COMPLETED)
        finally:
            release.set()
            await app.aclose()

    asyncio.run(scenario())


def test_a_nested_chain_left_waiting_after_its_leaf_woke_is_continued_up_to_the_task(
    monkeypatch,
) -> None:
    async def scenario() -> None:
        clock, sessions, tasks = _clocked_stores("memory", None, None)
        app = _owned_family(sessions, tasks, clock, nested=True)
        completions = _record_completions(tasks, monkeypatch)
        try:
            await app.task_store.create_task(TaskCreate(task_id="job-nested", type=_JOB))
            assert (await _until_idle(app, "first-worker")).handled == 1
            middle = await _child(sessions, _parent_of("job-nested"))
            leaf = await _child(sessions, middle.id)
            wake = await _waits_on_sleeping_child(sessions, middle.id, leaf.id)
            job = await tasks.load_task("job-nested")
            receipt = await tasks.load_interrupted_task_handoff_receipt(
                job.id, job.interrupted_handoff_id
            )
            assert receipt.request.parked == TaskParkedWait(
                kind="session_wake", wait_id=wake["input_id"], session_id=leaf.id
            )
            _due(clock)
            # Without the task's claim, the leaf ends and its parents wait.
            assert await session_wakes._deliver(
                app, session_id=leaf.id, wake_id=wake["input_id"]
            ) is (SessionWakeOutcome.CONTINUED)
            assert (await sessions.load(middle.id)).status is SessionStatus.INTERRUPTED

            assert (await _until_idle(app)).handled == 1
            assert (await sessions.load(middle.id)).status is SessionStatus.COMPLETED
            assert (await sessions.load(job.session_id)).status is SessionStatus.COMPLETED
            assert completions["job-nested"] == [f"idle-worker:continue:{wake['input_id']}"]
        finally:
            await app.aclose()

    asyncio.run(scenario())


def test_a_parked_parent_whose_woken_child_ended_without_a_live_attempt_gets_one(
    monkeypatch,
) -> None:
    async def scenario() -> None:
        clock, sessions, tasks = _clocked_stores("memory", None, None)
        app = _owned_family(sessions, tasks, clock)
        completions = _record_completions(tasks, monkeypatch)
        try:
            job, child, wake = await _parked_parent(app, "job-lost-attempt")
            _due(clock)
            # The child woke and ended, but the attempt that would continue its parent
            # is gone, as when its worker died after continuing the child.
            assert await session_wakes._deliver(
                app, session_id=child.id, wake_id=wake["input_id"]
            ) is (SessionWakeOutcome.CONTINUED)
            await session_wakes._cancel_session_wake_attempts(app, wake["input_id"])
            [lost] = await session_wakes.session_wake_attempts(app, wake["input_id"])
            assert lost.status is TaskStatus.CANCELLED
            # The task still runs, so its chain is the next attempt's to continue.
            await session_wakes.reconcile_session_wakes(app)
            assert await session_wakes.reconcile_parked_handoffs(app) == 0
            assert len(await session_wakes.session_wake_attempts(app, wake["input_id"])) == 2

            clock.advance(2 * session_wakes._RETRY_BACKOFF.total_seconds())
            assert (await _until_idle(app)).handled == 1
            assert (await sessions.load(job.session_id)).status is SessionStatus.COMPLETED
            assert completions[job.id] == [f"idle-worker:continue:{wake['input_id']}"]
        finally:
            await app.aclose()

    asyncio.run(scenario())

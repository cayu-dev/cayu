"""Tests for the generic ``run_task_worker`` durable-worker helper."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from cayu import (
    AgentSpec,
    AlwaysRequireApprovalToolPolicy,
    CayuApp,
    EventType,
    Message,
    ModelStreamEvent,
    PendingActionQuery,
    ResumeRequest,
    RunRequest,
    ScriptedModelProvider,
    SQLiteSessionStore,
    SQLiteTaskStore,
    Task,
    TaskCreate,
    TaskHandlerOutcome,
    TaskQuery,
    Tool,
    ToolApprovalDecision,
    ToolApprovalRequest,
    ToolContext,
    ToolEffect,
    ToolResult,
    ToolSpec,
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


class _PublishChangeTool(Tool):
    spec = ToolSpec(
        name="publish_change",
        description="Publish one reviewed change.",
        input_schema={
            "type": "object",
            "properties": {"change": {"type": "string"}},
            "required": ["change"],
        },
        effect=ToolEffect.EXTERNAL,
    )

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        return ToolResult(
            content=f"Published {args['change']}",
            structured={"session_id": ctx.session_id},
        )


def _register_approval_agent(app: CayuApp) -> None:
    app.register_agent(
        AgentSpec(name="worker-agent", model="scripted-model"),
        tools=[_PublishChangeTool()],
        tool_policy=AlwaysRequireApprovalToolPolicy(),
    )


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


def test_run_task_worker_hands_interrupted_session_to_reconstructed_control_plane(
    tmp_path: Path,
) -> None:
    session_path = tmp_path / "sessions.sqlite"
    task_path = tmp_path / "tasks.sqlite"
    first_app = CayuApp(
        session_store=SQLiteSessionStore(session_path),
        task_store=SQLiteTaskStore(task_path),
        enable_logging=False,
    )
    first_app.register_provider(
        ScriptedModelProvider(
            [
                [
                    ModelStreamEvent.tool_call(
                        id="publish-change",
                        name="publish_change",
                        arguments={"change": "reviewed-release"},
                    ),
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ]
            ]
        ),
        default=True,
    )
    _register_approval_agent(first_app)
    assert first_app.task_store is not None

    async def await_approval(
        app: CayuApp,
        task: Task,
        worker_id: str,
    ) -> TaskHandlerOutcome:
        async for _event in app.run(
            RunRequest(
                agent_name="worker-agent",
                session_id="session-handoff",
                task_id=task.id,
                task_worker_id=worker_id,
                messages=[Message.text("user", "Publish the reviewed change.")],
            )
        ):
            pass
        return TaskHandlerOutcome.SESSION_INTERRUPTED

    async def scenario() -> tuple[Task | None, SessionStatus, Task | None, SessionStatus]:
        await first_app.create_task(
            TaskCreate(
                task_id="task-handoff",
                type="job",
                assigned_agent_name="worker-agent",
            )
        )
        handled = await run_task_worker(
            first_app,
            first_app.task_store,
            await_approval,
            worker_id="worker-a",
            query=TaskQuery(type="job"),
            max_tasks=1,
            poll_interval_s=0.05,
            reclaim=False,
        )
        assert handled == 1

        handed_off_task = await first_app.task_store.load_task("task-handoff")
        handed_off_session = await first_app.session_store.load("session-handoff")
        pending = await first_app.session_store.query_pending_actions(
            PendingActionQuery(session_id="session-handoff")
        )
        assert handed_off_task is not None
        assert handed_off_task.status == "running"
        assert handed_off_task.session_id == "session-handoff"
        assert handed_off_task.worker_id is None
        assert handed_off_task.lease_expires_at is None
        assert handed_off_session is not None
        assert handed_off_session.status == SessionStatus.INTERRUPTED
        assert len(pending.actions) == 1
        assert pending.actions[0].approval_id is not None

        reconstructed = CayuApp(
            session_store=SQLiteSessionStore(session_path),
            task_store=SQLiteTaskStore(task_path),
            enable_logging=False,
        )
        reconstructed.register_provider(
            ScriptedModelProvider(
                [
                    [
                        ModelStreamEvent.text_delta("The durable work item is complete."),
                        ModelStreamEvent.completed({"finish_reason": "stop"}),
                    ]
                ]
            ),
            default=True,
        )
        _register_approval_agent(reconstructed)
        approval_id = pending.actions[0].approval_id
        assert approval_id is not None
        async for _event in reconstructed.resolve_tool_approval(
            ToolApprovalRequest(
                session_id="session-handoff",
                approval_id=approval_id,
                decision=ToolApprovalDecision.APPROVE,
            )
        ):
            pass

        assert reconstructed.task_store is not None
        completed_task = await reconstructed.task_store.load_task("task-handoff")
        completed_session = await reconstructed.session_store.load("session-handoff")
        assert completed_session is not None
        return (
            handed_off_task,
            handed_off_session.status,
            completed_task,
            completed_session.status,
        )

    handed_off_task, handed_off_status, completed_task, completed_status = asyncio.run(scenario())
    assert handed_off_task is not None
    assert handed_off_status == SessionStatus.INTERRUPTED
    assert completed_task is not None
    assert completed_task.status == "completed"
    assert completed_status == SessionStatus.COMPLETED


@pytest.mark.parametrize(
    "session_status",
    [
        SessionStatus.PENDING,
        SessionStatus.RUNNING,
        SessionStatus.INTERRUPTING,
        SessionStatus.COMPLETED,
        SessionStatus.FAILED,
    ],
)
def test_run_task_worker_fails_handoff_when_session_is_not_interrupted(
    tmp_path: Path,
    session_status: SessionStatus,
) -> None:
    session_store = SQLiteSessionStore(tmp_path / "sessions.sqlite")
    task_store = SQLiteTaskStore(tmp_path / "tasks.sqlite")
    app = CayuApp(
        session_store=session_store,
        task_store=task_store,
        enable_logging=False,
    )
    app.register_provider(ScriptedModelProvider([]), default=True)
    app.register_agent(AgentSpec(name="worker-agent", model="scripted-model"))

    async def invalid_handoff(
        _app: CayuApp,
        task: Task,
        worker_id: str,
    ) -> TaskHandlerOutcome:
        await task_store.attach_task(
            task.id,
            session_id="session-invalid-handoff",
            worker_id=worker_id,
        )
        return TaskHandlerOutcome.SESSION_INTERRUPTED

    async def scenario() -> Task | None:
        await session_store.create(
            RunRequest(
                agent_name="worker-agent",
                session_id="session-invalid-handoff",
                messages=[Message.text("user", "original")],
            ),
            identity=SessionIdentity(
                provider_name="scripted",
                model="scripted-model",
            ),
        )
        if session_status is not SessionStatus.PENDING:
            await session_store.update_status("session-invalid-handoff", session_status)
        created = await task_store.create_task(TaskCreate(type="job"))
        await run_task_worker(
            app,
            task_store,
            invalid_handoff,
            worker_id="worker-a",
            query=TaskQuery(type="job"),
            max_tasks=1,
            poll_interval_s=0.05,
            reclaim=False,
        )
        return await task_store.load_task(created.id)

    task = asyncio.run(scenario())
    assert task is not None
    assert task.status == "failed"
    assert task.error == {
        "error": "RuntimeError",
        "message": (
            "Task handler requested an interrupted-session handoff while session "
            f"session-invalid-handoff was {session_status}."
        ),
    }


def test_run_task_worker_fails_handoff_for_missing_attached_session(tmp_path: Path) -> None:
    app, store = _build(tmp_path)

    async def missing_session_handoff(
        _app: CayuApp,
        task: Task,
        worker_id: str,
    ) -> TaskHandlerOutcome:
        await store.attach_task(
            task.id,
            session_id="session-missing",
            worker_id=worker_id,
        )
        return TaskHandlerOutcome.SESSION_INTERRUPTED

    async def scenario() -> Task | None:
        created = await store.create_task(TaskCreate(type="job"))
        await run_task_worker(
            app,
            store,
            missing_session_handoff,
            worker_id="worker-a",
            query=TaskQuery(type="job"),
            max_tasks=1,
            poll_interval_s=0.05,
            reclaim=False,
        )
        return await store.load_task(created.id)

    task = asyncio.run(scenario())
    assert task is not None
    assert task.status == "failed"
    assert task.error == {
        "error": "RuntimeError",
        "message": "Attached session not found: session-missing.",
    }


def test_run_task_worker_preserves_terminal_task_before_handoff_cleanup(tmp_path: Path) -> None:
    app, store = _build(tmp_path)

    async def terminal_race(
        _app: CayuApp,
        task: Task,
        _worker_id: str,
    ) -> TaskHandlerOutcome:
        await store.complete_task(task.id, {"winner": "terminal-state"})
        return TaskHandlerOutcome.SESSION_INTERRUPTED

    async def scenario() -> Task | None:
        created = await store.create_task(TaskCreate(type="job"))
        await run_task_worker(
            app,
            store,
            terminal_race,
            worker_id="worker-a",
            query=TaskQuery(type="job"),
            max_tasks=1,
            poll_interval_s=0.05,
            reclaim=False,
        )
        return await store.load_task(created.id)

    task = asyncio.run(scenario())
    assert task is not None
    assert task.status == "completed"
    assert task.result == {"winner": "terminal-state"}


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

"""Public direct-run and native-worker recovery ownership controls."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from cayu import (
    AgentSpec,
    CayuApp,
    Environment,
    EnvironmentSpec,
    EventType,
    InMemorySessionStore,
    InMemoryTaskStore,
    LocalWorkspace,
    Message,
    ModelStreamEvent,
    ReadFileTool,
    RecoveryPlanAction,
    RecoveryPlanRequest,
    RecoveryPlanSelection,
    ResumeRequest,
    RunRequest,
    ScriptedModelProvider,
    SessionStatus,
    SQLiteSessionStore,
    SQLiteTaskStore,
    Task,
    TaskCreate,
    TaskHandlerOutcome,
    TaskStatus,
    run_task_worker,
)


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("mode", ["direct", "worker", "unbound"])
@pytest.mark.parametrize("interrupt", [False, True], ids=["complete", "step-limit"])
def test_task_run_recovery_ownership_contract(
    tmp_path: Path, backend: str, mode: str, interrupt: bool
) -> None:
    async def scenario() -> None:
        path = tmp_path / "state.sqlite"
        sessions = InMemorySessionStore() if backend == "memory" else SQLiteSessionStore(path)
        tasks = InMemoryTaskStore() if backend == "memory" else SQLiteTaskStore(path)
        app = CayuApp(session_store=sessions, task_store=tasks, enable_logging=False)
        completed = [
            ModelStreamEvent.text_delta("done"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ]
        provider = ScriptedModelProvider(
            [
                [
                    ModelStreamEvent.tool_call(
                        id="read", name="read_file", arguments={"path": "README.md"}
                    ),
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ],
                completed,
            ]
            if interrupt
            else [completed]
        )
        (tmp_path / "README.md").write_text("hello\n")
        app.register_provider(provider, default=True)
        app.register_environment(
            Environment(EnvironmentSpec(name="local"), workspace=LocalWorkspace(tmp_path)),
            default=True,
        )
        app.register_agent(AgentSpec(name="reader", model="scripted-model"), tools=[ReadFileTool()])
        if mode != "unbound":
            await app.create_task(
                TaskCreate(
                    task_id="job",
                    type="read",
                    session_id="limited" if mode == "direct" else None,
                    assigned_agent_name="reader",
                )
            )
        events = []

        async def run(task: Task | None = None, worker_id: str | None = None) -> None:
            events.extend(
                [
                    event
                    async for event in app.run(
                        RunRequest(
                            agent_name="reader",
                            session_id="limited",
                            task_id="job" if mode != "unbound" else None,
                            task_worker_id=worker_id,
                            task_lease_expires_at=None if task is None else task.lease_expires_at,
                            max_steps=1,
                            messages=[Message.text("user", "Read README.")],
                        )
                    )
                ]
            )

        try:
            if mode == "worker":

                async def handler(app: CayuApp, task: Task, worker_id: str):
                    await run(task, worker_id)
                    return TaskHandlerOutcome.SESSION_INTERRUPTED if interrupt else None

                assert (
                    await run_task_worker(
                        app,
                        tasks,
                        handler,
                        worker_id="worker",
                        max_tasks=1,
                        poll_interval_s=0.01,
                        reclaim=False,
                    )
                    == 1
                )
            else:
                await run()
            assert len(provider.requests) == 1
            session = await sessions.load("limited")
            assert session is not None
            assert session.status is (
                SessionStatus.INTERRUPTED if interrupt else SessionStatus.COMPLETED
            )
            if interrupt:
                assert EventType.SESSION_LIMIT_REACHED in [event.type for event in events]
                assert EventType.TOOL_CALL_COMPLETED in [event.type for event in events]
            before_task = await tasks.load_task("job")
            before_checkpoint = await sessions.load_checkpoint("limited")
            before_events = await sessions.load_events("limited")
            plan = await app.plan_recovery(
                RecoveryPlanRequest(selection=RecoveryPlanSelection(session_ids=("limited",)))
            )
            assert await tasks.load_task("job") == before_task
            assert await sessions.load("limited") == session
            assert await sessions.load_checkpoint("limited") == before_checkpoint
            assert await sessions.load_events("limited") == before_events
            assert len(provider.requests) == 1
            assert len(plan.items) == 1
            item = plan.items[0]
            assert not item.blockers
            assert item.allowed_actions == (RecoveryPlanAction.LEAVE_INTACT,)
            if mode != "unbound":
                task = await tasks.load_task("job")
                assert task is not None
                assert task.session_id == session.id
                assert task.session_instance_id == session.instance_id
                assert task.status is (TaskStatus.RUNNING if interrupt else TaskStatus.COMPLETED)
                if interrupt:
                    assert task.worker_id is None
                    assert task.lease_expires_at is None
                    assert len(item.task_claims) == 1
                    if mode == "worker":
                        assert task.interrupted_handoff_id is not None
                        receipt = await tasks.load_interrupted_task_handoff_receipt(
                            task.id, task.interrupted_handoff_id
                        )
                        assert receipt is not None and receipt.task == task
                        assert item.task_claims[0].ownership_status == "unowned"
                    else:
                        assert task.interrupted_handoff_id is None
                        assert item.task_claims[0].ownership_status == "direct"
                        resumed = [
                            event
                            async for event in app.resume(
                                ResumeRequest(
                                    session_id="limited",
                                    max_steps=1,
                                    messages=[Message.text("user", "Continue.")],
                                )
                            )
                        ]
                        assert resumed[-1].type is EventType.SESSION_COMPLETED
                        assert len(provider.requests) == 2
                        completed_task = await tasks.load_task("job")
                        assert completed_task is not None
                        assert completed_task.status is TaskStatus.COMPLETED
                        assert completed_task.session_instance_id == session.instance_id
                        assert completed_task.interrupted_handoff_id is None
                else:
                    assert not item.task_claims
            else:
                assert not item.task_claims
        finally:
            await app.drain_environment_cleanups()
            if backend == "sqlite":
                await sessions.close()
                await tasks.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize(
    "fault",
    [
        "worker_without_lease",
        "lease_without_worker",
        "wrong_incarnation",
        "missing_handoff_receipt",
        "changed_direct_read",
        "lost_direct_authority",
        "unsupported_direct_read",
        "missing_release",
        "corrupt_release",
        "missing_profile",
        "stale_epoch",
        "multiple_direct_attachments",
    ],
)
def test_direct_task_recovery_rejects_incomplete_or_changed_evidence(
    tmp_path: Path, backend: str, fault: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from copy import deepcopy
    from datetime import UTC, datetime
    from uuid import uuid4

    from cayu import RecoveryBlockerCode, TaskClaimLost

    async def scenario() -> None:
        path = tmp_path / "state.sqlite"
        sessions = InMemorySessionStore() if backend == "memory" else SQLiteSessionStore(path)
        tasks = InMemoryTaskStore() if backend == "memory" else SQLiteTaskStore(path)
        app = CayuApp(session_store=sessions, task_store=tasks, enable_logging=False)
        provider = ScriptedModelProvider(
            [
                ModelStreamEvent.tool_call(
                    id="read", name="read_file", arguments={"path": "README.md"}
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ]
        )
        (tmp_path / "README.md").write_text("hello\n")
        app.register_provider(provider, default=True)
        app.register_environment(
            Environment(EnvironmentSpec(name="local"), workspace=LocalWorkspace(tmp_path)),
            default=True,
        )
        app.register_agent(AgentSpec(name="reader", model="scripted-model"), tools=[ReadFileTool()])
        await app.create_task(TaskCreate(task_id="job", type="read", session_id="limited"))
        try:
            async for _event in app.run(
                RunRequest(
                    agent_name="reader",
                    session_id="limited",
                    task_id="job",
                    max_steps=1,
                    messages=[Message.text("user", "Read README.")],
                )
            ):
                pass
            task = await tasks.load_task("job")
            session = await sessions.load("limited")
            checkpoint = await sessions.load_checkpoint("limited")
            assert task is not None and session is not None and checkpoint is not None
            request = RecoveryPlanRequest(selection=RecoveryPlanSelection(session_ids=("limited",)))
            good_plan = await app.plan_recovery(request)
            assert good_plan.items[0].task_claims[0].ownership_status == "direct"

            with monkeypatch.context() as patch:
                if fault == "multiple_direct_attachments":
                    snapshot = await sessions.load_invocation_snapshot(session.id)
                    assert snapshot is not None
                    await tasks.create_running_task(
                        TaskCreate(task_id="other", type="read", session_id=session.id),
                        session_invocation=snapshot,
                    )
                elif fault in {
                    "worker_without_lease",
                    "lease_without_worker",
                    "wrong_incarnation",
                    "missing_handoff_receipt",
                }:
                    updates = {
                        "worker_without_lease": {"worker_id": "orphan-worker"},
                        "lease_without_worker": {"lease_expires_at": datetime.now(UTC)},
                        "wrong_incarnation": {"session_instance_id": str(uuid4())},
                        "missing_handoff_receipt": {"interrupted_handoff_id": "missing"},
                    }[fault]
                    original_list = tasks.list_tasks

                    async def changed_tasks(query=None):
                        return [
                            item.model_copy(update=updates) if item.id == task.id else item
                            for item in await original_list(query)
                        ]

                    patch.setattr(tasks, "list_tasks", changed_tasks)
                elif fault in {
                    "changed_direct_read",
                    "lost_direct_authority",
                    "unsupported_direct_read",
                }:

                    async def changed_direct_read(*args, **kwargs):
                        del args, kwargs
                        if fault == "lost_direct_authority":
                            raise TaskClaimLost("attachment changed")
                        if fault == "unsupported_direct_read":
                            raise NotImplementedError("no atomic direct read")
                        return task.model_copy(update={"metadata": {"changed": True}})

                    patch.setattr(tasks, "load_direct_attached_task_resume", changed_direct_read)
                elif fault == "stale_epoch":
                    original_load = sessions.load

                    async def changed_session(session_id):
                        item = await original_load(session_id)
                        assert item is not None
                        return item.model_copy(update={"run_epoch": item.run_epoch + 1})

                    patch.setattr(sessions, "load", changed_session)
                else:
                    damaged = deepcopy(checkpoint)
                    if fault == "missing_release":
                        damaged.pop("invocation_lifecycle_receipt")
                    elif fault == "corrupt_release":
                        damaged["invocation_lifecycle_receipt"]["record_sha256"] = "0" * 64
                    else:
                        damaged.pop("active_invocation_execution_profile")

                    async def changed_checkpoint(session_id):
                        assert session_id == "limited"
                        return deepcopy(damaged)

                    patch.setattr(sessions, "load_checkpoint", changed_checkpoint)
                plan = await app.plan_recovery(request)
                item = plan.items[0]
                if fault == "multiple_direct_attachments":
                    assert len(item.task_claims) == 2
                    assert all(claim.ownership_status == "direct" for claim in item.task_claims)
                else:
                    assert len(item.task_claims) == 1
                    assert item.task_claims[0].ownership_status == "invalid"
                assert RecoveryBlockerCode.INVALID_DURABLE_STATE in {
                    blocker.code for blocker in item.blockers
                }
                assert item.allowed_actions == (RecoveryPlanAction.LEAVE_INTACT,)
            assert await tasks.load_task("job") == task
            assert await sessions.load("limited") == session
            assert await sessions.load_checkpoint("limited") == checkpoint
            assert len(provider.requests) == 1
        finally:
            await app.drain_environment_cleanups()
            if backend == "sqlite":
                await sessions.close()
                await tasks.close()

    asyncio.run(scenario())

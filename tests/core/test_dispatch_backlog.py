"""Credential-free public Runtime isolation of prepared-child overlap."""

import asyncio

import pytest

from cayu import (
    AgentSpec,
    CayuApp,
    ExecutionProfileBehaviorIdentity,
    Message,
    ModelProvider,
    ModelStreamEvent,
    RunRequest,
    SQLiteSessionStore,
    SQLiteTaskStore,
    SubagentExecutionMode,
    SubagentSpec,
    SubagentTool,
    TaskCreate,
    TaskStatus,
    TaskStoreDispatcher,
    run_to_completion,
)


class BarrierProvider(ModelProvider):
    name = "barrier"

    def __init__(self):
        self.root_calls = 0
        self.active = 0
        self.maximum = 0
        self.child_calls = 0
        self.all_started = asyncio.Event()
        self.release = asyncio.Event()

    @property
    def execution_profile_identity(self):
        return ExecutionProfileBehaviorIdentity(
            name="isolation.barrier", behavior_version="1", implementation_version="1"
        )

    async def stream(self, request):
        if not request.tools:
            self.child_calls += 1
            self.active += 1
            self.maximum = max(self.maximum, self.active)
            if self.active == 3:
                self.all_started.set()
            try:
                await self.release.wait()
                yield ModelStreamEvent.text_delta("done")
                yield ModelStreamEvent.completed({"finish_reason": "stop"})
            finally:
                self.active -= 1
            return
        call = self.root_calls
        self.root_calls += 1
        if call < 3:
            yield ModelStreamEvent.tool_call(
                id=f"child-{call}",
                name="subagent",
                arguments={"agent": "child", "task": f"work {call}"},
            )
            yield ModelStreamEvent.completed({"finish_reason": "tool_calls"})
        else:
            yield ModelStreamEvent.text_delta("queued")
            yield ModelStreamEvent.completed({"finish_reason": "stop"})


@pytest.mark.parametrize("root_task", [False, True])
@pytest.mark.parametrize("start_mode", ["before", "late", "reenter", "restart", "missed"])
@pytest.mark.parametrize("housekeeping", ["split", "default"])
def test_three_public_dispatch_workers_overlap(tmp_path, root_task, start_mode, housekeeping):
    asyncio.run(
        _three_public_dispatch_workers_overlap(tmp_path, root_task, start_mode, housekeeping)
    )


async def _three_public_dispatch_workers_overlap(tmp_path, root_task, start_mode, housekeeping):
    tasks = SQLiteTaskStore(tmp_path / "runtime.db")
    dispatcher = TaskStoreDispatcher(tasks)
    app = CayuApp(
        session_store=SQLiteSessionStore(tmp_path / "runtime.db"),
        task_store=tasks,
        dispatcher=dispatcher,
        enable_logging=False,
    )
    provider = BarrierProvider()
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="child", model="fixture"))
    tool = SubagentTool(
        app,
        agents={
            "child": SubagentSpec(
                agent_name="child", mode=SubagentExecutionMode.DURABLE, max_steps=2
            )
        },
    )
    app.register_agent(AgentSpec(name="root", model="fixture"), tools=[tool])
    if root_task:
        await tasks.create_task(
            TaskCreate(task_id="root-task", type="root", assigned_agent_name="root")
        )
    stop = asyncio.Event()

    def start_workers():
        return [
            asyncio.create_task(
                dispatcher.run_worker(
                    app,
                    worker_id=f"worker-{i}",
                    stop=stop,
                    poll_interval_s=0.1,
                    **(
                        {"reconcile_terminal_receipts": i == 0, "reclaim_expired_leases": i == 1}
                        if housekeeping == "split"
                        else {}
                    ),
                )
            )
            for i in range(3)
        ]

    if start_mode == "missed":

        async def no_notifications(_queries):
            return None

        tasks._task_admission_wakeup = no_notifications
    if start_mode in {"reenter", "restart"}:
        initial_workers = start_workers()
        await asyncio.sleep(0.2)
        stop.set()
        await asyncio.wait_for(asyncio.gather(*initial_workers), timeout=20)
        stop = asyncio.Event()
        if start_mode == "restart":
            dispatcher = TaskStoreDispatcher(SQLiteTaskStore(tmp_path / "runtime.db"))
    start_after_root = start_mode not in {"before", "missed"}
    workers = [] if start_after_root else start_workers()
    try:
        outcome = await run_to_completion(
            app,
            RunRequest(
                agent_name="root",
                session_id="root",
                task_id="root-task" if root_task else None,
                messages=[Message.text("user", "Queue three children.")],
                max_steps=6,
            ),
        )
        assert outcome.ok, outcome.error
        if start_after_root:
            workers = start_workers()
        try:
            await asyncio.wait_for(provider.all_started.wait(), timeout=15)
        except TimeoutError:
            pytest.fail(
                f"Only {provider.maximum} concurrent handlers reached the barrier; root_task={root_task}, start_after_root={start_after_root}"
            )
        assert provider.maximum == 3
    finally:
        provider.release.set()
        stop.set()
        await asyncio.wait_for(asyncio.gather(*workers), timeout=20)
        assert await app.drain_environment_cleanups(timeout_s=5)
    assert provider.child_calls == 3
    assert provider.active == 0
    children = [task for task in await tasks.list_tasks() if task.id != "root-task"]
    assert len(children) == 3
    assert all(task.status is TaskStatus.COMPLETED for task in children)
    assert all(task.worker_id is None and task.lease_expires_at is None for task in children)

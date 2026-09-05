from __future__ import annotations

import asyncio
import os
import time

import pytest

from cayu import (
    AgentSpec,
    CayuApp,
    CayuConfig,
    EventQuery,
    EventType,
    Message,
    RunRequest,
    SQLiteSessionStore,
    SQLiteTaskStore,
    TaskCreate,
    TaskQuery,
    Tool,
    ToolContext,
    ToolEffect,
    ToolExecutionConfig,
    ToolResult,
    ToolSpec,
    run_task_worker,
)
from cayu.providers import ModelProvider, ModelStreamEvent
from cayu.runtime._invocation_lifecycle import require_released_invocation_command_authority
from cayu.runtime.execution_profiles import active_invocation_execution_profile_from_checkpoint


class EvidenceTool(Tool):
    spec = ToolSpec(
        name="evidence",
        description="Deterministic qualification evidence.",
        input_schema={"type": "object", "properties": {}, "additionalProperties": False},
        effect=ToolEffect.NONE,
        parallel_safe=True,
    )

    def __init__(self, barrier_width=0, skip_first=False):
        self.barrier_width = barrier_width
        self.skip_first = skip_first
        self.barrier = asyncio.Event()
        self.active = 0
        self.peak = 0
        self.calls = 0

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        self.active += 1
        self.peak = max(self.peak, self.active)
        self.calls += 2 if os.environ.get("CAYU_QUALIFICATION_FAULT") == "blind-retry" else 1
        try:
            if self.barrier_width and not (self.skip_first and self.calls == 1):
                if self.active >= self.barrier_width:
                    self.barrier.set()
                await asyncio.wait_for(self.barrier.wait(), timeout=60)
            await asyncio.sleep(0)
            return ToolResult(content="bounded evidence")
        finally:
            if os.environ.get("CAYU_QUALIFICATION_FAULT") != "cleanup-leak":
                self.active -= 1


class TrajectoryProvider(ModelProvider):
    name = "qualification"

    def __init__(self, rounds, width):
        self.rounds = rounds
        self.width = width

    async def stream(self, request):
        step = sum(message.role == "assistant" for message in request.messages)
        if step < self.rounds:
            for index in range(self.width if step % 2 else 1):
                yield ModelStreamEvent.tool_call(
                    id=f"call-{step}-{index}",
                    name="evidence",
                    arguments={},
                )
            yield ModelStreamEvent.completed({"finish_reason": "tool_calls"})
        else:
            yield ModelStreamEvent.text_delta("done")
            yield ModelStreamEvent.completed({"finish_reason": "stop"})


def test_durable_concurrent_long_trajectories(tmp_path, record_property):
    stress = os.environ.get("CAYU_QUALIFICATION_PROFILE") == "stress"
    count, rounds, width = (100, 2, 4) if stress else (24, 6, 4)
    if os.environ.get("CAYU_QUALIFICATION_FAULT"):
        count, rounds, width = 2, 2, 4

    async def run():
        sessions = SQLiteSessionStore(tmp_path / "sessions.sqlite")
        tasks = SQLiteTaskStore(tmp_path / "tasks.sqlite")
        tool = EvidenceTool(barrier_width=count)
        app = CayuApp(
            session_store=sessions,
            task_store=tasks,
            enable_logging=False,
            config=CayuConfig(tool_execution=ToolExecutionConfig(max_parallel_tool_calls=4)),
        )
        app.register_provider(TrajectoryProvider(rounds, width), default=True)
        app.register_agent(
            AgentSpec(name="qualification", model="fixture", workflow_tool_names=("evidence",)),
            tools=[tool],
        )
        created = [await tasks.create_task(TaskCreate(type="qualification")) for _ in range(count)]
        event_counts = []
        terminals = []

        async def handle(app, task, worker_id):
            events = [
                event
                async for event in app.run(
                    RunRequest(
                        agent_name="qualification",
                        session_id=f"session-{task.id}",
                        task_id=task.id,
                        task_worker_id=(
                            "stale-worker"
                            if os.environ.get("CAYU_QUALIFICATION_FAULT") == "authority-loss"
                            else worker_id
                        ),
                        task_lease_expires_at=task.lease_expires_at,
                        messages=[Message.text("user", "run bounded trajectory")],
                        max_steps=rounds + 2,
                    )
                )
            ]
            terminals.append(events[-1].type)
            event_counts.append(len(events))

        try:
            handled = await asyncio.wait_for(
                asyncio.gather(
                    *(
                        run_task_worker(
                            app,
                            tasks,
                            handle,
                            worker_id=f"worker-{index}",
                            query=TaskQuery(type="qualification"),
                            max_tasks=1,
                            poll_interval_s=0.05,
                            reclaim=False,
                        )
                        for index in range(count)
                    )
                ),
                timeout=480 if stress else 180,
            )
            task_rows = [await tasks.load_task(task.id) for task in created]
            session_rows = [await sessions.load(f"session-{task.id}") for task in created]
            record_property("active_claims", sum(row.worker_id is not None for row in task_rows))
            record_property("unfinished_tasks", sum(row.status != "completed" for row in task_rows))
            record_property(
                "unfinished_sessions",
                sum(row is None or row.status != "completed" for row in session_rows),
            )
            record_property("active_tools", tool.active)
            sequences = []
            for row in session_rows:
                if row is not None:
                    records = await sessions.query_events(EventQuery(session_id=row.id, limit=1000))
                    sequences.extend(record.sequence for record in records)
            if sequences:
                record_property("event_sequence_min", min(sequences))
                record_property("event_sequence_max", max(sequences))
            assert terminals == [EventType.SESSION_COMPLETED] * count
            assert sum(handled) == count
            for task in created:
                stored = await tasks.load_task(task.id)
                assert stored.status == "completed"
                assert stored.worker_id is None and stored.lease_expires_at is None
                session = await sessions.load(f"session-{task.id}")
                assert session.status == "completed"
                checkpoint = await sessions.load_checkpoint(session.id)
                profile = active_invocation_execution_profile_from_checkpoint(checkpoint)
                assert profile is not None
                require_released_invocation_command_authority(
                    session,
                    checkpoint,
                    session_id=session.id,
                    session_instance_id=session.instance_id,
                    active_profile=profile,
                )
            record_property("active_fences", 0)
            assert tool.calls == count * (rounds // 2) * (width + 1)
            assert tool.peak >= count and tool.active == 0
            record_property("sessions_completed", count)
            record_property("tasks_completed", count)
            record_property("tool_calls", tool.calls)
            record_property("event_count_min", min(event_counts))
            record_property("event_count_max", max(event_counts))
            record_property("active_tools", tool.active)
            record_property("active_claims", 0)
        finally:
            try:
                recovery_drained = await app.drain_recovery_cleanups(timeout_s=10)
                environments_drained = await app.drain_environment_cleanups(timeout_s=10)
            finally:
                await sessions.close()
                await tasks.close()
        record_property(
            "retained_cleanups", int(not recovery_drained) + int(not environments_drained)
        )
        assert recovery_drained and environments_drained
        assert not [task for task in asyncio.all_tasks() if task is not asyncio.current_task()]

    asyncio.run(run())


def test_empty_worker_pressure_preserves_control_latency(tmp_path, record_property):
    from tests.core.test_task_worker import _run_handler

    count = 200 if os.environ.get("CAYU_QUALIFICATION_PROFILE") == "stress" else 32

    async def run():
        stop = asyncio.Event()
        polls = 0

        class CountingStore(SQLiteTaskStore):
            # Every dispatched native claim settles before the wrapper returns.
            verified_work_mutations_are_cancellation_quiescent = True

            async def claim_task(self, *args, **kwargs):
                nonlocal polls
                repeats = 30 if os.environ.get("CAYU_QUALIFICATION_FAULT") == "hot-polling" else 1
                for _ in range(repeats):
                    polls += 1
                    result = await super().claim_task(*args, **kwargs)
                    if result is not None:
                        return result
                return None

        store = CountingStore(tmp_path / "pressure.sqlite")
        app = CayuApp(task_store=store, enable_logging=False)
        app.register_provider(TrajectoryProvider(0, 1), default=True)
        app.register_agent(AgentSpec(name="worker-agent", model="fixture"))
        pressure_started = time.monotonic()
        workers = [
            asyncio.create_task(
                run_task_worker(
                    app,
                    store,
                    _run_handler,
                    worker_id=f"empty-{index}",
                    query=TaskQuery(type="empty"),
                    stop=stop,
                    poll_interval_s=0.1,
                    reclaim=False,
                )
            )
            for index in range(count)
        ]
        try:
            await asyncio.sleep(0.15)
            start = time.monotonic()
            task = await store.create_task(TaskCreate(type="control"))
            handled = await asyncio.wait_for(
                run_task_worker(
                    app,
                    store,
                    _run_handler,
                    worker_id="control",
                    query=TaskQuery(type="control"),
                    max_tasks=1,
                    poll_interval_s=0.05,
                    reclaim=False,
                ),
                timeout=10,
            )
            latency = time.monotonic() - start
            assert handled == 1 and (await store.load_task(task.id)).status == "completed"
            assert latency < 5.0
            # Shared demand polling should stay bounded independently of idle worker count.
            assert 0 < polls <= 10 + 30 * (time.monotonic() - pressure_started)
            record_property("control_latency_ms", round(latency * 1000))
            record_property("empty_polls", polls)
        finally:
            stop.set()
            await asyncio.wait_for(asyncio.gather(*workers), timeout=10)
            await app.drain_recovery_cleanups(timeout_s=10)
            await app.drain_environment_cleanups(timeout_s=10)
            await store.close()
        assert all(worker.done() for worker in workers)

    asyncio.run(run())


@pytest.mark.stress
def test_hundred_call_round(tmp_path):
    async def run():
        store = SQLiteSessionStore(tmp_path / "wide.sqlite")
        tool = EvidenceTool(barrier_width=100, skip_first=True)
        app = CayuApp(
            session_store=store,
            enable_logging=False,
            config=CayuConfig(
                tool_execution=ToolExecutionConfig(max_parallel_tool_calls=100),
            ),
        )
        app.register_provider(TrajectoryProvider(2, 100), default=True)
        app.register_agent(
            AgentSpec(
                name="qualification",
                model="fixture",
                workflow_tool_names=("evidence",),
            ),
            tools=[tool],
        )
        try:
            events = [
                event
                async for event in app.run(
                    RunRequest(
                        agent_name="qualification",
                        messages=[Message.text("user", "wide round")],
                    )
                )
            ]
            assert events[-1].type == EventType.SESSION_COMPLETED
            assert tool.calls == 101 and tool.peak == 100 and tool.active == 0
        finally:
            await store.close()

    asyncio.run(run())


def test_long_dynamic_tool_trajectory(tmp_path, record_property):
    from cayu import StaticToolExposurePolicy

    class DynamicProvider(ModelProvider):
        name = "dynamic-qualification"

        async def stream(self, request):
            step = sum(message.role == "assistant" for message in request.messages)
            if step == 0:
                yield ModelStreamEvent.tool_call(
                    id="discover",
                    name="search_tools",
                    arguments={"query": "evidence", "limit": 1},
                )
            elif step <= 12:
                results = [
                    part
                    for message in request.messages
                    for part in message.content
                    if part.type == "tool_result" and part.tool_call_id == "discover"
                ]
                reference = results[0].structured["matches"][0]["tool_ref"]
                for index in range(4 if step % 2 == 0 else 1):
                    yield ModelStreamEvent.tool_call(
                        id=f"dynamic-{step}-{index}",
                        name="call_tool",
                        arguments={"tool_ref": reference, "arguments": {}},
                    )
            else:
                yield ModelStreamEvent.text_delta("done")
                yield ModelStreamEvent.completed({"finish_reason": "stop"})
                return
            yield ModelStreamEvent.completed({"finish_reason": "tool_calls"})

    async def run():
        store = SQLiteSessionStore(tmp_path / "dynamic.sqlite")
        tool = EvidenceTool()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(DynamicProvider(), default=True)
        app.register_agent(
            AgentSpec(name="qualification", model="fixture"),
            tools=[tool],
            tool_discovery_mode="search_tools",
            tool_exposure_policy=StaticToolExposurePolicy(profile_id="qualification", tools=()),
        )
        try:
            events = [
                event
                async for event in app.run(
                    RunRequest(
                        agent_name="qualification",
                        max_steps=16,
                        messages=[Message.text("user", "repeat dynamic evidence")],
                    )
                )
            ]
            assert events[-1].type == EventType.SESSION_COMPLETED
            assert tool.calls == 30 and tool.active == 0
            record_property("tool_calls", tool.calls)
        finally:
            await app.drain_recovery_cleanups(timeout_s=10)
            await app.drain_environment_cleanups(timeout_s=10)
            await store.close()

    asyncio.run(run())

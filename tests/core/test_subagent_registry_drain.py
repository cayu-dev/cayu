"""Registry observation retains child ownership across timeout and cancellation."""

import asyncio

import pytest
from tests.core._workload_secret_support import FakeProvider

from cayu import (
    AgentSpec,
    CayuApp,
    EventQuery,
    EventType,
    Message,
    ModelStreamEvent,
    RunRequest,
    SessionStatus,
    ToolContext,
)
from cayu.runtime.sessions import InMemorySessionStore, SessionIdentity
from cayu.storage import SQLiteSessionStore
from cayu.tools.subagents import (
    BackgroundSubagentTaskRegistry,
    SubagentExecutionMode,
    SubagentSpec,
    SubagentTool,
    default_background_subagent_registry,
)


class BlockingProvider(FakeProvider):
    def __init__(self):
        super().__init__([])
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.stopped = asyncio.Event()

    async def stream(self, request):
        self.requests.append(request)
        self.entered.set()
        try:
            await self.release.wait()
            yield ModelStreamEvent.text_delta("review complete")
            yield ModelStreamEvent.completed({"finish_reason": "stop"})
        finally:
            self.stopped.set()


def test_tool_exposes_the_actual_default_registry():
    tool = SubagentTool(CayuApp(), agents={"reviewer": "reviewer"})
    assert tool.background_task_registry is default_background_subagent_registry()


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_real_child_retained_after_registry_timeout_and_waiter_cancellation(tmp_path, backend):
    async def scenario():
        store = (
            InMemorySessionStore()
            if backend == "memory"
            else SQLiteSessionStore(tmp_path / "sessions.sqlite")
        )
        provider = BlockingProvider()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="reviewer", model="fake-model"))
        registry = BackgroundSubagentTaskRegistry()
        tool = SubagentTool(
            app,
            agents={
                "reviewer": SubagentSpec(
                    agent_name="reviewer", mode=SubagentExecutionMode.BACKGROUND
                )
            },
            background_registry=registry,
        )
        app.register_agent(AgentSpec(name="parent", model="fake-model"), tools=[tool])
        registered = app.get_agent("parent").tools["subagent"].tool
        assert isinstance(registered, SubagentTool)
        assert registered.background_task_registry is registry
        waiter = None
        try:
            await store.create(
                RunRequest(
                    agent_name="parent",
                    session_id="parent",
                    messages=[Message.text("user", "parent task")],
                ),
                identity=SessionIdentity(provider_name="fake", model="fake-model"),
            )
            result = await tool.run(
                ToolContext(session_id="parent"), {"agent": "reviewer", "task": "review"}
            )
            assert not result.is_error
            await asyncio.wait_for(provider.entered.wait(), 10)
            (child_task,) = registry.active_tasks("parent")
            assert await registry.drain(timeout_s=0.01) is False
            assert not child_task.done() and child_task.cancelling() == 0
            waiter = asyncio.create_task(registry.drain(timeout_s=10))
            await asyncio.sleep(0)
            waiter.cancel("observer-stop")
            with pytest.raises(asyncio.CancelledError, match="observer-stop"):
                await waiter
            assert waiter.cancelled() and waiter.cancelling() == 1
            assert not child_task.done() and child_task.cancelling() == 0
            assert not provider.stopped.is_set()
            provider.release.set()
            assert await registry.drain(timeout_s=10) is True
            assert child_task.done() and not child_task.cancelled() and provider.stopped.is_set()
            assert len(provider.requests) == 1
            assert result.structured is not None
            child_id = result.structured["child_session_id"]
            child = await store.load(child_id)
            assert child is not None and child.status is SessionStatus.COMPLETED
            terminal = await store.query_events(
                EventQuery(session_id=child_id, event_type=EventType.SESSION_COMPLETED)
            )
            assert len(terminal) == 1
            for drain in (
                app.drain_background_interruptions,
                app.drain_recovery_cleanups,
                app.drain_provider_operation_cancellations,
                app.drain_environment_cleanups,
                app.drain_knowledge_publications,
            ):
                assert await drain() is True
        finally:
            provider.release.set()
            if waiter is not None and not waiter.done():
                waiter.cancel()
                await asyncio.gather(waiter, return_exceptions=True)
            await asyncio.gather(*registry.active_tasks("parent"), return_exceptions=True)
            if isinstance(store, SQLiteSessionStore):
                await store.close()

    asyncio.run(scenario())


def test_registry_drain_refreshes_work_registered_by_another_parent():
    async def scenario():
        registry = BackgroundSubagentTaskRegistry()
        release_first, release_second, second_started = (
            asyncio.Event(),
            asyncio.Event(),
            asyncio.Event(),
        )

        async def second():
            second_started.set()
            await release_second.wait()

        async def first():
            await release_first.wait()
            registry.register(
                asyncio.create_task(second()),
                parent_session_id="other-parent",
                child_session_id="second",
            )

        registry.register(
            asyncio.create_task(first()), parent_session_id="parent", child_session_id="first"
        )
        waiter = asyncio.create_task(registry.drain(timeout_s=10))
        try:
            await asyncio.sleep(0)
            release_first.set()
            await asyncio.wait_for(second_started.wait(), 5)
            done, _ = await asyncio.wait([waiter], timeout=0.01)
            assert not done
            release_second.set()
            assert await waiter is True
            assert await registry.drain(timeout_s=0.1) is True
        finally:
            release_first.set()
            release_second.set()
            await asyncio.gather(
                waiter,
                *registry.active_tasks("parent"),
                *registry.active_tasks("other-parent"),
                return_exceptions=True,
            )

    asyncio.run(scenario())


@pytest.mark.parametrize("timeout", [True, False, 0, -1, float("nan"), float("inf"), "1"])
def test_registry_drain_rejects_invalid_timeout(timeout):
    async def scenario():
        with pytest.raises(ValueError, match="finite positive"):
            await BackgroundSubagentTaskRegistry().drain(timeout_s=timeout)

    asyncio.run(scenario())


def test_failed_stream_is_settled_without_erasing_failure(caplog):
    async def scenario():
        registry = BackgroundSubagentTaskRegistry()

        async def fail():
            raise RuntimeError("child failure")

        registry.register(
            asyncio.create_task(fail()), parent_session_id="parent", child_session_id="child"
        )
        assert await registry.drain(timeout_s=1) is True
        await asyncio.sleep(0)
        failure = registry.failure("child")
        assert failure is not None and failure["error"] == "child failure"

    asyncio.run(scenario())
    assert "failed while draining" in caplog.text

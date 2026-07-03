"""Background subagent task registry: strong refs, teardown, logged failures."""

from __future__ import annotations

import asyncio
import logging

import pytest

from cayu.core.events import Event, EventType
from cayu.core.messages import Message
from cayu.core.tools import ToolContext
from cayu.runtime.sessions import InMemorySessionStore, RunRequest, SessionIdentity
from cayu.tools.subagents import (
    BACKGROUND_SUBAGENT_FAILURE_ARTIFACT_TYPE,
    BackgroundSubagentTaskRegistry,
    SubagentExecutionMode,
    SubagentResultTool,
    SubagentSpec,
    SubagentTool,
    default_background_subagent_registry,
)


class _ScriptedRuntime:
    """SubagentRuntime stub whose run() streams events from a factory."""

    def __init__(self, stream_factory):
        self._stream_factory = stream_factory

    def run(self, request):
        return self._stream_factory(request)

    def interrupt_session(self, request):
        raise AssertionError("interrupt_session must not be called in background mode.")


def _background_tool(
    stream_factory,
    registry: BackgroundSubagentTaskRegistry,
) -> SubagentTool:
    return SubagentTool(
        _ScriptedRuntime(stream_factory),
        agents={
            "reviewer": SubagentSpec(
                agent_name="reviewer",
                mode=SubagentExecutionMode.BACKGROUND,
            )
        },
        background_registry=registry,
    )


def test_background_subagent_task_is_held_and_released_on_success():
    registry = BackgroundSubagentTaskRegistry()

    def stream(request):
        async def events():
            yield Event(type=EventType.SESSION_STARTED, session_id=request.session_id)
            yield Event(type=EventType.SESSION_COMPLETED, session_id=request.session_id)

        return events()

    tool = _background_tool(stream, registry)

    async def run():
        result = await tool.run(
            ToolContext(session_id="sess_bg_success"),
            {"agent": "reviewer", "task": "review"},
        )
        tasks = registry.active_tasks("sess_bg_success")
        assert len(tasks) == 1
        await asyncio.gather(*tasks, return_exceptions=True)
        await asyncio.sleep(0)
        assert registry.active_tasks("sess_bg_success") == ()
        return result

    result = asyncio.run(run())
    assert result.is_error is not True
    assert result.structured["status"] == "started"
    child_session_id = result.structured["child_session_id"]
    assert registry.failure(child_session_id) is None


def test_background_subagent_drain_failure_is_logged_and_recorded(caplog):
    registry = BackgroundSubagentTaskRegistry()

    def stream(request):
        async def events():
            yield Event(type=EventType.SESSION_STARTED, session_id=request.session_id)
            raise RuntimeError("stream exploded")

        return events()

    tool = _background_tool(stream, registry)

    async def run():
        result = await tool.run(
            ToolContext(session_id="sess_bg_failure"),
            {"agent": "reviewer", "task": "review"},
        )
        tasks = registry.active_tasks("sess_bg_failure")
        assert len(tasks) == 1
        await asyncio.gather(*tasks, return_exceptions=True)
        await asyncio.sleep(0)
        return result

    with caplog.at_level(logging.ERROR, logger="cayu.tools.subagents"):
        result = asyncio.run(run())

    assert result.is_error is not True
    child_session_id = result.structured["child_session_id"]
    failure = registry.failure(child_session_id)
    assert failure is not None
    assert failure["type"] == BACKGROUND_SUBAGENT_FAILURE_ARTIFACT_TYPE
    assert failure["parent_session_id"] == "sess_bg_failure"
    assert failure["child_session_id"] == child_session_id
    assert failure["error"] == "stream exploded"
    assert failure["error_type"] == "RuntimeError"
    assert registry.active_tasks("sess_bg_failure") == ()
    assert "failed while draining runtime events" in caplog.text
    assert "stream exploded" in caplog.text


def test_cancel_parent_cancels_and_drains_background_tasks():
    registry = BackgroundSubagentTaskRegistry()

    def stream(request):
        async def events():
            yield Event(type=EventType.SESSION_STARTED, session_id=request.session_id)
            await asyncio.Event().wait()

        return events()

    tool = _background_tool(stream, registry)

    async def run():
        result = await tool.run(
            ToolContext(session_id="sess_bg_teardown"),
            {"agent": "reviewer", "task": "review"},
        )
        tasks = registry.active_tasks("sess_bg_teardown")
        assert len(tasks) == 1
        await registry.cancel_parent("sess_bg_teardown")
        assert tasks[0].cancelled()
        await asyncio.sleep(0)
        assert registry.active_tasks("sess_bg_teardown") == ()
        return result

    result = asyncio.run(run())
    child_session_id = result.structured["child_session_id"]
    assert registry.failure(child_session_id) is None


def test_registry_rejects_duplicate_child_registration():
    registry = BackgroundSubagentTaskRegistry()

    async def run():
        async def idle():
            await asyncio.sleep(0)

        task = asyncio.create_task(idle())
        registry.register(task, parent_session_id="sess_p", child_session_id="sess_c")
        other = asyncio.create_task(idle())
        try:
            with pytest.raises(ValueError, match="already registered"):
                registry.register(other, parent_session_id="sess_p", child_session_id="sess_c")
        finally:
            await asyncio.gather(task, other, return_exceptions=True)

    asyncio.run(run())


def test_subagent_result_reports_background_drain_failure():
    registry = BackgroundSubagentTaskRegistry()

    async def run():
        store = InMemorySessionStore()
        identity = SessionIdentity(provider_name="fake", model="fake-model")
        await store.create(
            RunRequest(
                agent_name="parent",
                session_id="sess_result_parent",
                messages=[Message.text("user", "parent task")],
            ),
            identity=identity,
        )
        await store.create(
            RunRequest(
                agent_name="reviewer",
                session_id="sess_result_child",
                parent_session_id="sess_result_parent",
                messages=[Message.text("user", "child task")],
                metadata={"subagent": {"agent": "reviewer", "mode": "background"}},
            ),
            identity=identity,
        )

        async def boom():
            raise RuntimeError("drain crashed")

        task = asyncio.create_task(boom())
        registry.register(
            task,
            parent_session_id="sess_result_parent",
            child_session_id="sess_result_child",
        )
        await asyncio.gather(task, return_exceptions=True)
        await asyncio.sleep(0)

        tool = SubagentResultTool(store, background_registry=registry)
        one = await tool.run(
            ToolContext(session_id="sess_result_parent"),
            {"child_session_id": "sess_result_child", "wait": False},
        )
        every = await tool.run(
            ToolContext(session_id="sess_result_parent"),
            {"all": True, "wait": False},
        )
        return one, every

    one, every = asyncio.run(run())

    assert one.is_error is True
    assert "background execution failed" in one.content
    assert "drain crashed" in one.content
    assert one.structured["retrieval_status"] == "not_ready"
    assert one.structured["background_failure"]["error_type"] == "RuntimeError"
    assert one.structured["background_failure"]["type"] == BACKGROUND_SUBAGENT_FAILURE_ARTIFACT_TYPE

    assert every.is_error is True
    assert "background failure: drain crashed" in every.content
    child_summary = every.structured["children"][0]
    assert child_summary["background_failure"]["error"] == "drain crashed"


def test_subagent_tools_default_to_shared_registry():
    registry = default_background_subagent_registry()
    assert isinstance(registry, BackgroundSubagentTaskRegistry)
    assert default_background_subagent_registry() is registry

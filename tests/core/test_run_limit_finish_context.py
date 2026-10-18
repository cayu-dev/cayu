"""Finishing preserves the active question and each interaction's result state."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta

import pytest
from tests.core.test_run_limit_finish import ReadTool, answer, spec
from tests.core.test_targeted_tool_grants import _GatewayRememberTool

from cayu import AgentSpec, CayuApp, EventType, Message, ResumeRequest, RunRequest
from cayu.budgets.run_limits import RunLimits
from cayu.context import RecentTurnsContextPolicy
from cayu.deadlines import ExecutionDeadline
from cayu.evals import ScriptedModelProvider
from cayu.evals.testing import model_facing_text
from cayu.providers import ModelContextOverflowError, ModelStreamEvent
from cayu.providers.openai import build_openai_payload
from cayu.sessions.messaging import EnqueueSessionMessageRequest, SessionMessageDeliveryMode
from cayu.tasks import InMemoryTaskStore, TaskCreate, TaskStatus
from cayu.tools import TargetedToolGrant
from cayu.tools.exposure import StaticToolExposurePolicy


@pytest.mark.parametrize("strategy", ["native", "tool"])
@pytest.mark.parametrize("steps", [1, 2])
def test_finish_instruction_preserves_question_and_investigation(strategy, steps):
    responses = [answer(strategy)]
    if steps == 2:
        responses.insert(
            0,
            [
                ModelStreamEvent.tool_call(id="investigate", name="read", arguments={}),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
        )
    provider = ScriptedModelProvider(responses, supports_native_structured_output=True)
    app = CayuApp(enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="agent", model="fake"),
        tools=[ReadTool()],
        context_policy=RecentTurnsContextPolicy(max_user_turns=1),
    )

    async def scenario():
        try:
            events = [
                event
                async for event in app.run(
                    RunRequest(
                        agent_name="agent",
                        session_id="finish-question",
                        messages=[Message.text("user", "What is the ultimate answer?")],
                        max_steps=steps,
                        limits=RunLimits(on_limit="finish"),
                        structured_output=spec(strategy),
                    )
                )
            ]
            text = model_facing_text(provider.requests[-1])
            assert "What is the ultimate answer?" in text
            assert "final model step" in text
            if steps == 2:
                assert "The answer is 42." in text
            assert events[-1].type is EventType.SESSION_INTERRUPTED
            transcript = await app.session_store.load_transcript("finish-question")
            assert len([message for message in transcript if message.role == "user"]) == 1
        finally:
            await app.aclose()

    asyncio.run(scenario())


@pytest.mark.parametrize("strategy", ["native", "tool"])
def test_finish_instruction_is_scoped_to_its_model_request(strategy):
    provider = ScriptedModelProvider(
        [
            [
                ModelStreamEvent.tool_call(id="investigate", name="read", arguments={}),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            answer(strategy),
            answer(strategy),
        ],
        supports_native_structured_output=True,
    )
    app = CayuApp(enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="agent", model="fake"), tools=[ReadTool()])

    async def scenario():
        try:
            first = [
                event
                async for event in app.run(
                    RunRequest(
                        agent_name="agent",
                        session_id="finish-then-resume",
                        messages=[Message.text("user", "First question")],
                        max_steps=2,
                        limits=RunLimits(on_limit="finish"),
                        structured_output=spec(strategy),
                    )
                )
            ]
            assert first[-1].type is EventType.SESSION_INTERRUPTED
            assert first[-1].payload["limit"] == "model_steps"
            assert "final model step" in model_facing_text(provider.requests[-1])
            resumed = [
                event
                async for event in app.resume(
                    ResumeRequest(
                        session_id="finish-then-resume",
                        messages=[Message.text("user", "Investigate the next question")],
                        max_steps=2,
                        limits=RunLimits(on_limit="finish"),
                        structured_output=spec(strategy),
                    )
                )
            ]
            assert resumed[-1].type is EventType.SESSION_COMPLETED
            assert len(provider.requests) == 3
            request = provider.requests[-1]
            assert "read" in {tool["name"] for tool in request.tools}
            assert "Investigate the next question" in model_facing_text(request)
            assert "final model step" not in model_facing_text(request)
            assert "Ordinary tools are withdrawn" not in model_facing_text(request)
        finally:
            await app.aclose()

    asyncio.run(scenario())


@pytest.mark.parametrize("strategy", ["native", "tool"])
def test_finish_overflow_rebuild_keeps_targeted_tools_withdrawn(strategy):
    class Provider(ScriptedModelProvider):
        def supports_targeted_tool_projection(self, *, model, protocol):
            return True

        async def stream(self, request):
            self.requests.append(request)
            build_openai_payload(request)
            if len(self.requests) == 1:
                raise ModelContextOverflowError("too large", provider=self.name)
            for event in answer(strategy):
                yield event

    provider = Provider([], supports_native_structured_output=True)
    tool = _GatewayRememberTool()
    app = CayuApp(enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="agent", model="fake"),
        tools=[tool],
        targeted_tool_mode="openai_additional_tools",
        tool_exposure_policy=StaticToolExposurePolicy(profile_id="targeted", tools=()),
        context_overflow_policy=RecentTurnsContextPolicy(max_user_turns=1),
    )

    async def scenario():
        try:
            events = [
                event
                async for event in app.run(
                    RunRequest(
                        agent_name="agent",
                        messages=[Message.text("user", "Answer now")],
                        max_steps=1,
                        limits=RunLimits(on_limit="finish"),
                        structured_output=spec(strategy),
                        tool_grants=(
                            TargetedToolGrant(request_id="fact", tool_id="cayu:remember"),
                        ),
                    )
                )
            ]
            assert len(provider.requests) == 2
            assert all(request.targeted_tool_projection is None for request in provider.requests)
            assert any(event.type is EventType.CONTEXT_OVERFLOW_RECOVERING for event in events)
            assert any(event.type is EventType.STRUCTURED_OUTPUT_VALIDATED for event in events)
            assert events[-1].type is EventType.SESSION_INTERRUPTED
            assert not tool.calls
        finally:
            await app.aclose()

    asyncio.run(scenario())


@pytest.mark.parametrize("strategy", ["native", "tool"])
def test_queued_interaction_marks_structured_result_pending_again(strategy):
    class Provider(ScriptedModelProvider):
        async def stream(self, request):
            self.requests.append(request)
            if len(self.requests) == 1:
                await app.enqueue_session_message(
                    EnqueueSessionMessageRequest(
                        session_id="queued-result",
                        idempotency_key="next",
                        content="Answer again",
                        delivery_mode=SessionMessageDeliveryMode.ON_IDLE,
                    )
                )
                for event in answer(strategy)[:-1]:
                    yield event
                yield ModelStreamEvent.completed({"usage": {"input_tokens": 1, "output_tokens": 1}})
            else:
                yield ModelStreamEvent.text_delta("not yet structured")
                yield ModelStreamEvent.completed(
                    {"usage": {"input_tokens": 1, "output_tokens": 10}}
                )

    provider = Provider([], supports_native_structured_output=True)
    app = CayuApp(enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="agent", model="fake"))

    async def scenario():
        try:
            events = [
                event
                async for event in app.run(
                    RunRequest(
                        agent_name="agent",
                        session_id="queued-result",
                        messages=[Message.text("user", "Answer once")],
                        max_steps=4,
                        limits=RunLimits(max_output_tokens=5),
                        structured_output=spec(strategy),
                    )
                )
            ]
            assert len(provider.requests) == 2
            assert (
                len(
                    [
                        event
                        for event in events
                        if event.type is EventType.STRUCTURED_OUTPUT_VALIDATED
                    ]
                )
                == 1
            )
            assert events[-1].type is EventType.SESSION_INTERRUPTED
            assert events[-1].payload["limit"] == "output_tokens"
            assert events[-1].payload["structured_output_pending"] is True
        finally:
            await app.aclose()

    asyncio.run(scenario())


@pytest.mark.parametrize("strategy", ["native", "tool"])
def test_consumer_delay_after_validated_finish_keeps_deadline_settlement(monkeypatch, strategy):
    import cayu.deadlines as deadline_module

    advance = timedelta()

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime.now(tz) + advance

    monkeypatch.setattr(deadline_module, "datetime", Clock)
    provider = ScriptedModelProvider(answer(strategy), supports_native_structured_output=True)
    app = CayuApp(task_store=InMemoryTaskStore(), enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="agent", model="fake"))

    async def scenario():
        nonlocal advance
        try:
            task = await app.create_task(TaskCreate(type="answer"))
            events = []
            async for event in app.run(
                RunRequest(
                    agent_name="agent",
                    session_id="delayed-finish",
                    task_id=task.id,
                    messages=[Message.text("user", "Answer now")],
                    max_steps=1,
                    execution_deadline=ExecutionDeadline.after(600),
                    limits=RunLimits(on_limit="finish"),
                    structured_output=spec(strategy),
                )
            ):
                events.append(event)
                if event.type is EventType.STRUCTURED_OUTPUT_VALIDATED:
                    advance = timedelta(seconds=601)
            assert events[-1].type is EventType.SESSION_INTERRUPTED
            assert events[-1].payload["limit"] == "execution_deadline"
            assert events[-1].payload["structured_output_pending"] is False
            assert (await app.task_store.load_task(task.id)).status is TaskStatus.FAILED
            assert len(provider.requests) == 1
        finally:
            await app.aclose()

    asyncio.run(scenario())

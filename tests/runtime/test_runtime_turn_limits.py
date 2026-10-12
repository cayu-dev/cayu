"""Turn limits contracts."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

from tests.runtime.runtime_budget_support import fake_budget_limit
from tests.runtime.runtime_provider_support import FakeProvider
from tests.runtime.runtime_session_support import (
    assert_only_model_step_publication_checkpoint,
    collect_events,
    collect_resume_events,
)
from tests.runtime.runtime_tool_support import EchoTool, SideEffectTool

import cayu.runtime._session_engine as session_engine_module
from cayu.agents import AgentSpec
from cayu.applications import CayuApp
from cayu.approvals.tools import ResolutionActor, ResolutionActorSource
from cayu.budgets.run_limits import RunLimits
from cayu.configuration import CayuConfig, ToolExecutionConfig
from cayu.events import Event, EventType
from cayu.messages import Message
from cayu.providers.base import ModelStreamEvent
from cayu.runtime._event_projection import PRIVATE_EVENT_AUTHORITY
from cayu.runtime.execution_profiles import (
    ExecutionProfileAdoptionIntent,
    ExecutionProfileAuthorityDecision,
    ExecutionProfileComponentClass,
    ExecutionProfilePolicy,
    ExecutionProfilePolicyAction,
    ExecutionProfilePolicyRequest,
    ExecutionProfilePolicyResult,
)
from cayu.runtime.loop_policies import BeforeStopContext, BeforeStopDecision, LoopPolicy
from cayu.sessions.memory import InMemorySessionStore
from cayu.sessions.records import SessionStatus
from cayu.sessions.requests import ResumeRequest, RunRequest
from cayu.tools.base import ToolContext, ToolResult
from cayu.tools.policy import ToolPolicy, ToolPolicyDecision, ToolPolicyRequest, ToolPolicyResult


class ContinueBeforeStopPolicy(LoopPolicy):
    def __init__(self, message: str = "Please repair the final answer.") -> None:
        self.message = message
        self.calls = 0

    async def before_stop(self, context: BeforeStopContext) -> BeforeStopDecision:
        self.calls += 1
        if context.step == 1:
            return BeforeStopDecision.continue_with(
                Message.text("user", self.message),
                reason="needs repair",
            )
        return BeforeStopDecision.complete("second step is final")


class InterruptBeforeStopPolicy(LoopPolicy):
    async def before_stop(self, context: BeforeStopContext) -> BeforeStopDecision:
        return BeforeStopDecision.interrupt(
            "needs operator review",
            metadata={"classification": context.classification.type.value},
        )


class _AuthorizeInvocationPolicyRemoval(ExecutionProfilePolicy):
    @property
    def identity(self) -> str:
        return "tests:authorize-invocation-policy-removal:v1"

    async def decide(
        self,
        request: ExecutionProfilePolicyRequest,
    ) -> ExecutionProfilePolicyResult:
        if request.changed_component_classes == (
            ExecutionProfileComponentClass.INVOCATION_POLICIES,
        ):
            return ExecutionProfilePolicyResult(
                action=ExecutionProfilePolicyAction.ADOPT,
                reason="The test explicitly starts a new invocation without its one-shot gate.",
                authority_decision=ExecutionProfileAuthorityDecision.AUTHORIZED,
            )
        return ExecutionProfilePolicyResult(
            action=ExecutionProfilePolicyAction.REJECT,
            reason="The test policy authorizes only request-policy removal.",
        )


def _invocation_policy_removal_intent() -> ExecutionProfileAdoptionIntent:
    return ExecutionProfileAdoptionIntent(
        idempotency_key="remove-one-shot-invocation-policy",
        reason="The operator reviewed the interrupted result and starts a fresh invocation.",
        requested_by=ResolutionActor(
            subject="test-operator",
            source=ResolutionActorSource.REQUEST,
        ),
    )


class FailBeforeStopPolicy(LoopPolicy):
    async def before_stop(self, context: BeforeStopContext) -> BeforeStopDecision:
        return BeforeStopDecision.fail("completion gate failed")


def assert_model_step_limit_interruption(
    events: list[Event],
    *,
    maximum: int,
    actual: int,
    cumulative_model_steps: int,
) -> None:
    limit_event = events[-3]
    interrupted_event = events[-1]

    assert limit_event.type == EventType.SESSION_LIMIT_REACHED
    assert limit_event.payload["reason"] == "limit_reached"
    assert limit_event.payload["limit"] == "model_steps"
    assert limit_event.payload["maximum"] == maximum
    assert limit_event.payload["actual"] == actual
    assert limit_event.payload["usage_summary"]["model_steps"] == cumulative_model_steps
    assert interrupted_event.type == EventType.SESSION_INTERRUPTED
    assert interrupted_event.payload["interruption_type"] == "limit_reached"
    assert interrupted_event.payload["limit"] == "model_steps"
    assert interrupted_event.payload["maximum"] == maximum
    assert interrupted_event.payload["actual"] == actual


def test_cayu_app_stops_on_token_limit_before_tool_side_effects():
    store = InMemorySessionStore()
    tool = SideEffectTool()
    provider = FakeProvider(
        [
            ModelStreamEvent.tool_call(
                id="call_1",
                name="side_effect",
                arguments={},
            ),
            ModelStreamEvent.completed(
                {
                    "finish_reason": "tool_calls",
                    "usage": {
                        "input_tokens": 7,
                        "output_tokens": 4,
                        "total_tokens": 11,
                    },
                }
            ),
        ]
    )
    app = CayuApp(session_store=store)
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[tool],
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_token_limit",
                messages=[Message.text("user", "do it")],
                limits=RunLimits(max_total_tokens=10),
            ),
        )
    )

    assert [event.type for event in events] == [
        EventType.SESSION_STARTED,
        EventType.MODEL_STARTED,
        EventType.MODEL_COMPLETED,
        EventType.SESSION_LIMIT_REACHED,
        EventType.TOOL_CALL_FAILED,
        EventType.TURN_COMPLETED,
        EventType.SESSION_INTERRUPTED,
    ]
    assert events[3].payload["limit"] == "total_tokens"
    assert events[3].payload["actual"] == 11
    assert events[3].payload["maximum"] == 10
    assert events[4].payload["reason"] == "limit_reached"
    assert events[6].payload["interruption_type"] == "limit_reached"
    assert tool.calls == []

    transcript = asyncio.run(store.load_transcript("sess_token_limit"))
    assert [message.role for message in transcript] == ["user", "assistant", "tool"]
    session = asyncio.run(store.load("sess_token_limit"))
    assert session is not None
    assert session.status == SessionStatus.INTERRUPTED


def test_cayu_app_stops_on_token_limit_after_final_model_answer():
    store = InMemorySessionStore()
    provider = FakeProvider(
        [
            ModelStreamEvent.text_delta("final answer"),
            ModelStreamEvent.completed(
                {
                    "finish_reason": "stop",
                    "usage": {
                        "input_tokens": 6,
                        "output_tokens": 4,
                        "total_tokens": 10,
                    },
                }
            ),
        ]
    )
    app = CayuApp(session_store=store)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_final_token_limit",
                messages=[Message.text("user", "answer")],
                limits=RunLimits(max_total_tokens=10),
            ),
        )
    )

    assert [event.type for event in events] == [
        EventType.SESSION_STARTED,
        EventType.MODEL_STARTED,
        EventType.MODEL_TEXT_DELTA,
        EventType.MODEL_COMPLETED,
        EventType.SESSION_LIMIT_REACHED,
        EventType.TURN_COMPLETED,
        EventType.SESSION_INTERRUPTED,
    ]
    assert events[4].payload["limit"] == "total_tokens"
    assert events[6].payload["interruption_type"] == "limit_reached"

    transcript = asyncio.run(store.load_transcript("sess_final_token_limit"))
    assert [message.role for message in transcript] == ["user", "assistant"]
    session = asyncio.run(store.load("sess_final_token_limit"))
    assert session is not None
    assert session.status == SessionStatus.INTERRUPTED


def test_cayu_app_stops_on_estimated_cost_limit_after_final_model_answer():
    store = InMemorySessionStore()
    provider = FakeProvider(
        [
            ModelStreamEvent.text_delta("final answer"),
            ModelStreamEvent.completed(
                {
                    "finish_reason": "stop",
                    "usage": {
                        "input_tokens": 1000,
                        "output_tokens": 100,
                        "total_tokens": 1100,
                    },
                }
            ),
        ]
    )
    app = CayuApp(session_store=store)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_cost_limit",
                messages=[Message.text("user", "answer")],
                budget_limits=(fake_budget_limit("0.002"),),
            ),
        )
    )

    assert [event.type for event in events] == [
        EventType.SESSION_STARTED,
        EventType.MODEL_STARTED,
        EventType.MODEL_TEXT_DELTA,
        EventType.MODEL_COMPLETED,
        EventType.SESSION_LIMIT_REACHED,
        EventType.TURN_COMPLETED,
        EventType.SESSION_INTERRUPTED,
    ]
    assert events[4].payload["limit"] == "estimated_cost"
    assert events[4].payload["maximum"] == "0.002"
    assert events[4].payload["actual"] == "0.002"
    assert events[4].payload["cost_summary"]["total_cost"] == "0.002"
    assert events[6].payload["interruption_type"] == "limit_reached"

    session = asyncio.run(store.load("sess_cost_limit"))
    assert session is not None
    assert session.status == SessionStatus.INTERRUPTED


def test_cayu_app_before_stop_policy_can_continue_with_durable_message():
    async def run():
        store = InMemorySessionStore()
        provider = FakeProvider(
            [
                [
                    ModelStreamEvent.text_delta("draft"),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ],
                [
                    ModelStreamEvent.text_delta("final"),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ],
            ]
        )
        policy = ContinueBeforeStopPolicy("Return a corrected final answer.")
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))

        events = await collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_before_stop_continue",
                messages=[Message.text("user", "answer")],
                max_steps=2,
                loop_policies=(policy,),
            ),
        )
        session = await store.load("sess_before_stop_continue")
        transcript = await store.load_transcript("sess_before_stop_continue")
        return events, session, transcript, provider, policy

    events, session, transcript, provider, policy = asyncio.run(run())

    assert session is not None
    assert session.status == SessionStatus.COMPLETED
    assert len(provider.requests) == 2
    assert policy.calls == 2
    assert any(event.type == "custom.loop.before_stop.selected" for event in events)
    assert [message.role for message in transcript] == ["user", "assistant", "user", "assistant"]
    assert transcript[2].content[0].text == "Return a corrected final answer."
    assert provider.requests[1].messages[-1].content[0].text == "Return a corrected final answer."


def test_cayu_app_before_stop_continue_at_max_steps_interrupts_and_resumes():
    async def run():
        store = InMemorySessionStore()
        provider = FakeProvider(
            [
                [
                    ModelStreamEvent.text_delta("draft"),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ],
                [
                    ModelStreamEvent.text_delta("resumed final"),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ],
            ]
        )
        policy = ContinueBeforeStopPolicy("Do not persist this at the boundary.")
        app = CayuApp(
            session_store=store,
            execution_profile_policy=_AuthorizeInvocationPolicyRemoval(),
            enable_logging=False,
        )
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))

        initial_events = await collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_before_stop_continue_at_limit",
                messages=[Message.text("user", "answer")],
                max_steps=1,
                loop_policies=(policy,),
            ),
        )
        interrupted_session = await store.load("sess_before_stop_continue_at_limit")
        interrupted_transcript = await store.load_transcript("sess_before_stop_continue_at_limit")
        resumed_events = await collect_resume_events(
            app,
            ResumeRequest(
                session_id="sess_before_stop_continue_at_limit",
                messages=[Message.text("user", "resume")],
                max_steps=1,
                profile_adoption=_invocation_policy_removal_intent(),
            ),
        )
        resumed_session = await store.load("sess_before_stop_continue_at_limit")
        resumed_transcript = await store.load_transcript("sess_before_stop_continue_at_limit")
        return (
            initial_events,
            interrupted_session,
            interrupted_transcript,
            resumed_events,
            resumed_session,
            resumed_transcript,
            provider,
            policy,
        )

    (
        initial_events,
        interrupted_session,
        interrupted_transcript,
        resumed_events,
        resumed_session,
        resumed_transcript,
        provider,
        policy,
    ) = asyncio.run(run())

    assert [event.type for event in initial_events] == [
        EventType.SESSION_STARTED,
        EventType.MODEL_STARTED,
        EventType.MODEL_TEXT_DELTA,
        EventType.MODEL_COMPLETED,
        "custom.loop.before_stop.started",
        "custom.loop.before_stop.completed",
        "custom.loop.before_stop.selected",
        EventType.SESSION_LIMIT_REACHED,
        EventType.TURN_COMPLETED,
        EventType.SESSION_INTERRUPTED,
    ]
    assert_model_step_limit_interruption(
        initial_events,
        maximum=1,
        actual=1,
        cumulative_model_steps=1,
    )
    selected = initial_events[-4]
    assert selected.payload["scope"] == "request"
    assert selected.payload["action"] == "continue"
    assert policy.calls == 1
    assert interrupted_session is not None
    assert interrupted_session.status == SessionStatus.INTERRUPTED
    assert [message.role for message in interrupted_transcript] == ["user", "assistant"]

    assert resumed_events[0].type == EventType.SESSION_EXECUTION_PROFILE_DECIDED
    assert resumed_events[1].type == EventType.SESSION_RESUMED
    assert resumed_events[-1].type == EventType.SESSION_COMPLETED
    assert resumed_session is not None
    assert resumed_session.status == SessionStatus.COMPLETED
    assert [message.role for message in resumed_transcript] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    assert len(provider.requests) == 2


def test_cayu_app_before_stop_policy_can_interrupt_and_resume():
    async def run():
        store = InMemorySessionStore()
        provider = FakeProvider(
            [
                [
                    ModelStreamEvent.text_delta("needs review"),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ],
                [
                    ModelStreamEvent.text_delta("resumed final"),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ],
            ]
        )
        app = CayuApp(
            session_store=store,
            execution_profile_policy=_AuthorizeInvocationPolicyRemoval(),
            enable_logging=False,
        )
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))

        interrupted_events = await collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_before_stop_interrupt",
                messages=[Message.text("user", "answer")],
                loop_policies=(InterruptBeforeStopPolicy(),),
            ),
        )
        interrupted_session = await store.load("sess_before_stop_interrupt")
        resumed_events = await collect_resume_events(
            app,
            ResumeRequest(
                session_id="sess_before_stop_interrupt",
                messages=[Message.text("user", "continue")],
                profile_adoption=_invocation_policy_removal_intent(),
            ),
        )
        resumed_session = await store.load("sess_before_stop_interrupt")
        return interrupted_events, interrupted_session, resumed_events, resumed_session, provider

    interrupted_events, interrupted_session, resumed_events, resumed_session, provider = (
        asyncio.run(run())
    )

    assert interrupted_session is not None
    assert interrupted_session.status == SessionStatus.INTERRUPTED
    assert interrupted_events[-1].type == EventType.SESSION_INTERRUPTED
    assert interrupted_events[-1].payload["reason"] == "needs operator review"
    assert resumed_session is not None
    assert resumed_session.status == SessionStatus.COMPLETED
    assert resumed_events[-1].type == EventType.SESSION_COMPLETED
    assert len(provider.requests) == 2


def test_cayu_app_before_stop_policy_can_fail_session():
    async def run():
        store = InMemorySessionStore()
        provider = FakeProvider(
            [
                ModelStreamEvent.text_delta("bad final"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ]
        )
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))

        events = await collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_before_stop_fail",
                messages=[Message.text("user", "answer")],
                loop_policies=(FailBeforeStopPolicy(),),
            ),
        )
        session = await store.load("sess_before_stop_fail")
        return events, session

    events, session = asyncio.run(run())

    assert session is not None
    assert session.status == SessionStatus.FAILED
    assert events[-1].type == EventType.SESSION_FAILED
    assert "completion gate failed" in events[-1].payload["error"]


def test_cayu_app_before_stop_policy_order_uses_first_non_complete_decision():
    class CompletePolicy(LoopPolicy):
        async def before_stop(self, context: BeforeStopContext) -> BeforeStopDecision:
            return BeforeStopDecision.complete("looks okay")

    async def run():
        store = InMemorySessionStore()
        provider = FakeProvider(
            [
                [
                    ModelStreamEvent.text_delta("draft"),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ],
                [
                    ModelStreamEvent.text_delta("final"),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ],
            ]
        )
        app = CayuApp(
            session_store=store,
            enable_logging=False,
            loop_policies=(CompletePolicy(),),
        )
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            loop_policies=(CompletePolicy(),),
        )
        events = await collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_before_stop_order",
                messages=[Message.text("user", "answer")],
                loop_policies=(ContinueBeforeStopPolicy("Continue from request."),),
            ),
        )
        selected = [event for event in events if event.type == "custom.loop.before_stop.selected"]
        completed = [event for event in events if event.type == "custom.loop.before_stop.completed"]
        return selected, completed, provider

    selected, completed, provider = asyncio.run(run())

    assert len(selected) == 1
    assert selected[0].payload["scope"] == "request"
    assert selected[0].payload["action"] == "continue"
    assert [event.payload["scope"] for event in completed[:3]] == ["app", "agent", "request"]
    assert len(provider.requests) == 2


def test_cayu_app_tool_call_limit_allows_existing_result_then_blocks_next_tool():
    store = InMemorySessionStore()
    tool = SideEffectTool()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_1",
                    name="side_effect",
                    arguments={"step": 1},
                ),
                ModelStreamEvent.completed(
                    {
                        "finish_reason": "tool_calls",
                        "usage": {"input_tokens": 1, "output_tokens": 1},
                    }
                ),
            ],
            [
                ModelStreamEvent.tool_call(
                    id="call_2",
                    name="side_effect",
                    arguments={"step": 2},
                ),
                ModelStreamEvent.completed(
                    {
                        "finish_reason": "tool_calls",
                        "usage": {"input_tokens": 1, "output_tokens": 1},
                    }
                ),
            ],
        ]
    )
    app = CayuApp(session_store=store)
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[tool],
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_tool_limit",
                messages=[Message.text("user", "do it")],
                limits=RunLimits(max_tool_calls=1),
            ),
        )
    )

    assert [event.type for event in events] == [
        EventType.SESSION_STARTED,
        EventType.MODEL_STARTED,
        EventType.MODEL_COMPLETED,
        EventType.TOOL_CALL_STARTED,
        EventType.TOOL_CALL_COMPLETED,
        EventType.MODEL_STARTED,
        EventType.MODEL_COMPLETED,
        EventType.SESSION_LIMIT_REACHED,
        EventType.TOOL_CALL_FAILED,
        EventType.TURN_COMPLETED,
        EventType.SESSION_INTERRUPTED,
    ]
    assert tool.calls == [{"step": 1}]
    assert events[7].payload["limit"] == "tool_calls"
    assert events[7].payload["actual"] == 2
    assert events[8].payload["tool_call_id"] == f"{events[8].id}:tool_call_id"
    assert events[8].payload["idempotency_key"] == PRIVATE_EVENT_AUTHORITY

    transcript = asyncio.run(store.load_transcript("sess_tool_limit"))
    assert [message.role for message in transcript] == [
        "user",
        "assistant",
        "tool",
        "assistant",
        "tool",
    ]


def test_cayu_app_elapsed_limit_stops_between_tool_calls(monkeypatch):
    clock = {"value": 0.0}
    monkeypatch.setattr(session_engine_module.time, "monotonic", lambda: clock["value"])

    class ElapsedTool(SideEffectTool):
        async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
            result = await super().run(ctx, args)
            clock["value"] = 1.0
            return result

    store = InMemorySessionStore()
    tool = ElapsedTool()
    provider = FakeProvider(
        [
            ModelStreamEvent.tool_call(
                id="call_1",
                name="side_effect",
                arguments={"step": 1},
            ),
            ModelStreamEvent.tool_call(
                id="call_2",
                name="side_effect",
                arguments={"step": 2},
            ),
            ModelStreamEvent.completed(
                {
                    "finish_reason": "tool_calls",
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                }
            ),
        ]
    )
    # Mid-round elapsed re-checks between tool calls are sequential-mode behavior.
    app = CayuApp(
        session_store=store,
        config=CayuConfig(tool_execution=ToolExecutionConfig(max_parallel_tool_calls=1)),
    )
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[tool],
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_elapsed_limit_between_tools",
                messages=[Message.text("user", "do it")],
                limits=RunLimits(max_elapsed_seconds=1),
            ),
        )
    )

    assert [event.type for event in events] == [
        EventType.SESSION_STARTED,
        EventType.MODEL_STARTED,
        EventType.MODEL_COMPLETED,
        EventType.TOOL_CALL_STARTED,
        EventType.TOOL_CALL_COMPLETED,
        EventType.SESSION_LIMIT_REACHED,
        EventType.TOOL_CALL_FAILED,
        EventType.TURN_COMPLETED,
        EventType.SESSION_INTERRUPTED,
    ]
    assert tool.calls == [{"step": 1}]
    assert events[5].payload["limit"] == "elapsed_seconds"
    assert events[6].payload["tool_call_id"] == f"{events[6].id}:tool_call_id"

    transcript = asyncio.run(store.load_transcript("sess_elapsed_limit_between_tools"))
    assert [message.role for message in transcript] == ["user", "assistant", "tool"]
    tool_results = transcript[-1].content
    assert len(tool_results) == 2
    assert tool_results[0].tool_call_id == "call_1"
    assert tool_results[1].tool_call_id == "call_2"


def test_cayu_app_elapsed_limit_stops_after_policy_before_approval(monkeypatch):
    clock = {"value": 0.0}
    monkeypatch.setattr(session_engine_module.time, "monotonic", lambda: clock["value"])

    class AdvancingApprovalPolicy(ToolPolicy):
        async def authorize(self, request: ToolPolicyRequest) -> ToolPolicyResult:
            clock["value"] = 1.0
            return ToolPolicyResult(
                decision=ToolPolicyDecision.REQUIRE_APPROVAL,
                reason="Approval required.",
            )

    store = InMemorySessionStore()
    tool = SideEffectTool()
    provider = FakeProvider(
        [
            ModelStreamEvent.tool_call(
                id="call_1",
                name="side_effect",
                arguments={},
            ),
            ModelStreamEvent.completed(
                {
                    "finish_reason": "tool_calls",
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                }
            ),
        ]
    )
    app = CayuApp(session_store=store)
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[tool],
        tool_policy=AdvancingApprovalPolicy(),
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_elapsed_limit_before_approval",
                messages=[Message.text("user", "do it")],
                limits=RunLimits(max_elapsed_seconds=1),
            ),
        )
    )

    assert [event.type for event in events] == [
        EventType.SESSION_STARTED,
        EventType.MODEL_STARTED,
        EventType.MODEL_COMPLETED,
        EventType.SESSION_LIMIT_REACHED,
        EventType.TOOL_CALL_FAILED,
        EventType.TURN_COMPLETED,
        EventType.SESSION_INTERRUPTED,
    ]
    assert not any(event.type == EventType.TOOL_CALL_APPROVAL_REQUESTED for event in events)
    assert tool.calls == []

    checkpoint = asyncio.run(store.load_checkpoint("sess_elapsed_limit_before_approval"))
    assert_only_model_step_publication_checkpoint(checkpoint)


def test_cayu_app_resume_stops_before_model_when_persisted_budget_is_reached():
    store = InMemorySessionStore()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.text_delta("first answer"),
                ModelStreamEvent.completed(
                    {
                        "finish_reason": "stop",
                        "usage": {
                            "input_tokens": 6,
                            "output_tokens": 4,
                            "total_tokens": 10,
                        },
                    }
                ),
            ],
        ]
    )
    app = CayuApp(session_store=store)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    limits = RunLimits(max_total_tokens=10, scope="session")
    asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_resume_limit",
                messages=[Message.text("user", "first")],
                limits=limits,
            ),
        )
    )

    events = asyncio.run(
        collect_resume_events(
            app,
            ResumeRequest(
                session_id="sess_resume_limit",
                messages=[Message.text("user", "second")],
                limits=limits,
            ),
        )
    )

    assert [event.type for event in events] == [
        EventType.SESSION_RESUMED,
        EventType.SESSION_LIMIT_REACHED,
        EventType.TURN_COMPLETED,
        EventType.SESSION_INTERRUPTED,
    ]
    assert len(provider.requests) == 1
    assert events[1].payload["usage_summary"]["usage"]["total_tokens"] == 10

    session = asyncio.run(store.load("sess_resume_limit"))
    assert session is not None
    assert session.status == SessionStatus.INTERRUPTED


def test_cayu_app_run_scoped_resume_ignores_prior_session_usage():
    store = InMemorySessionStore()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.text_delta("first answer"),
                ModelStreamEvent.completed(
                    {
                        "finish_reason": "stop",
                        "usage": {
                            "input_tokens": 6,
                            "output_tokens": 4,
                            "total_tokens": 10,
                        },
                    }
                ),
            ],
            [
                ModelStreamEvent.text_delta("second answer"),
                ModelStreamEvent.completed(
                    {
                        "finish_reason": "stop",
                        "usage": {
                            "input_tokens": 1,
                            "output_tokens": 1,
                            "total_tokens": 2,
                        },
                    }
                ),
            ],
        ]
    )
    app = CayuApp(session_store=store)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    limits = RunLimits(max_total_tokens=10, scope="run")
    asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_run_scope",
                messages=[Message.text("user", "first")],
                limits=limits,
            ),
        )
    )

    # The prior turn already consumed 10 total tokens. With scope="run" the
    # per-invocation delta starts at 0, so a resume must proceed normally even
    # though cumulative usage already meets the cap.
    events = asyncio.run(
        collect_resume_events(
            app,
            ResumeRequest(
                session_id="sess_run_scope",
                messages=[Message.text("user", "second")],
                limits=limits,
            ),
        )
    )

    assert [event.type for event in events] == [
        EventType.SESSION_RESUMED,
        EventType.MODEL_STARTED,
        EventType.MODEL_TEXT_DELTA,
        EventType.MODEL_COMPLETED,
        EventType.TURN_COMPLETED,
        EventType.SESSION_COMPLETED,
    ]
    assert len(provider.requests) == 2


def test_cayu_app_default_scope_resume_ignores_prior_session_usage():
    store = InMemorySessionStore()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.text_delta("first answer"),
                ModelStreamEvent.completed(
                    {
                        "finish_reason": "stop",
                        "usage": {
                            "input_tokens": 6,
                            "output_tokens": 4,
                            "total_tokens": 10,
                        },
                    }
                ),
            ],
            [
                ModelStreamEvent.text_delta("second answer"),
                ModelStreamEvent.completed(
                    {
                        "finish_reason": "stop",
                        "usage": {
                            "input_tokens": 1,
                            "output_tokens": 1,
                            "total_tokens": 2,
                        },
                    }
                ),
            ],
        ]
    )
    app = CayuApp(session_store=store)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    limits = RunLimits(max_total_tokens=10)
    asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_default_scope",
                messages=[Message.text("user", "first")],
                limits=limits,
            ),
        )
    )

    # RunLimits defaults to scope="run", so a resume with an unstated scope
    # measures only this invocation's usage — the prior turn's 10 tokens do
    # not count against the cap.
    events = asyncio.run(
        collect_resume_events(
            app,
            ResumeRequest(
                session_id="sess_default_scope",
                messages=[Message.text("user", "second")],
                limits=limits,
            ),
        )
    )

    assert [event.type for event in events] == [
        EventType.SESSION_RESUMED,
        EventType.MODEL_STARTED,
        EventType.MODEL_TEXT_DELTA,
        EventType.MODEL_COMPLETED,
        EventType.TURN_COMPLETED,
        EventType.SESSION_COMPLETED,
    ]
    assert len(provider.requests) == 2


def test_cayu_app_session_scoped_elapsed_limit_measures_session_lifetime():
    store = InMemorySessionStore()
    clock_now = datetime.now(UTC)

    def clock() -> datetime:
        return clock_now

    provider = FakeProvider(
        [
            ModelStreamEvent.text_delta("first answer"),
            ModelStreamEvent.completed(
                {
                    "finish_reason": "stop",
                    "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
                }
            ),
        ]
    )
    app = CayuApp(session_store=store, clock=clock)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    limits = RunLimits(max_elapsed_seconds=60, scope="session")
    asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_session_elapsed",
                messages=[Message.text("user", "first")],
                limits=limits,
            ),
        )
    )
    # Advance the injected application clock so durable session lifetime exceeds
    # the cap while the resume invocation itself starts on a fresh monotonic clock.
    clock_now += timedelta(seconds=120)

    events = asyncio.run(
        collect_resume_events(
            app,
            ResumeRequest(
                session_id="sess_session_elapsed",
                messages=[Message.text("user", "second")],
                limits=limits,
            ),
        )
    )

    assert [event.type for event in events] == [
        EventType.SESSION_RESUMED,
        EventType.SESSION_LIMIT_REACHED,
        EventType.TURN_COMPLETED,
        EventType.SESSION_INTERRUPTED,
    ]
    assert events[1].payload["limit"] == "elapsed_seconds"
    assert len(provider.requests) == 1


def test_cayu_app_interrupts_session_when_max_steps_exhausted_by_tool_work():
    store = InMemorySessionStore()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_1",
                    name="echo",
                    arguments={"text": "again"},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
        ]
    )
    app = CayuApp(session_store=store)
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[EchoTool()],
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_max_steps",
                messages=[Message.text("user", "loop")],
                max_steps=1,
            ),
        )
    )
    session = asyncio.run(store.load("sess_max_steps"))

    assert [event.type for event in events] == [
        EventType.SESSION_STARTED,
        EventType.MODEL_STARTED,
        EventType.MODEL_COMPLETED,
        EventType.TOOL_CALL_STARTED,
        EventType.TOOL_CALL_COMPLETED,
        EventType.SESSION_LIMIT_REACHED,
        EventType.TURN_COMPLETED,
        EventType.SESSION_INTERRUPTED,
    ]
    assert_model_step_limit_interruption(
        events,
        maximum=1,
        actual=1,
        cumulative_model_steps=1,
    )
    assert events[-2].payload["status"] == "interrupted"
    assert session is not None
    assert session.status == SessionStatus.INTERRUPTED


def test_default_run_completes_more_than_sixteen_tool_driven_model_rounds():
    def looping_provider(rounds: int) -> FakeProvider:
        responses = [
            [
                ModelStreamEvent.tool_call(
                    id=f"call_{index}",
                    name="echo",
                    arguments={"text": str(index)},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ]
            for index in range(rounds)
        ]
        responses.append(
            [
                ModelStreamEvent.text_delta("done"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ]
        )
        return FakeProvider(responses)

    default_store = InMemorySessionStore()
    default_provider = looping_provider(17)
    default_app = CayuApp(session_store=default_store, enable_logging=False)
    default_app.register_provider(default_provider, default=True)
    default_app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[EchoTool()],
    )
    default_events = asyncio.run(
        collect_events(
            default_app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_default_more_than_sixteen",
                messages=[Message.text("user", "keep using the tool")],
            ),
        )
    )

    constrained_store = InMemorySessionStore()
    constrained_provider = looping_provider(17)
    constrained_app = CayuApp(session_store=constrained_store, enable_logging=False)
    constrained_app.register_provider(constrained_provider, default=True)
    constrained_app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[EchoTool()],
    )
    constrained_events = asyncio.run(
        collect_events(
            constrained_app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_explicit_sixteen",
                messages=[Message.text("user", "keep using the tool")],
                max_steps=16,
            ),
        )
    )

    assert len(default_provider.requests) == 18
    assert default_events[-1].type is EventType.SESSION_COMPLETED
    assert len(constrained_provider.requests) == 16
    assert constrained_events[-1].type is EventType.SESSION_INTERRUPTED
    assert_model_step_limit_interruption(
        constrained_events,
        maximum=16,
        actual=16,
        cumulative_model_steps=16,
    )


def test_cayu_app_bounds_repeated_end_turn_false_by_max_steps():
    store = InMemorySessionStore()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.text_delta("Still working."),
                ModelStreamEvent.completed({"finish_reason": "stop", "end_turn": False}),
            ],
        ]
    )
    app = CayuApp(session_store=store)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_end_turn_false_max_steps",
                messages=[Message.text("user", "loop")],
                max_steps=1,
            ),
        )
    )
    session = asyncio.run(store.load("sess_end_turn_false_max_steps"))

    assert len(provider.requests) == 1
    assert [event.type for event in events] == [
        EventType.SESSION_STARTED,
        EventType.MODEL_STARTED,
        EventType.MODEL_TEXT_DELTA,
        EventType.MODEL_COMPLETED,
        EventType.SESSION_LIMIT_REACHED,
        EventType.TURN_COMPLETED,
        EventType.SESSION_INTERRUPTED,
    ]
    assert_model_step_limit_interruption(
        events,
        maximum=1,
        actual=1,
        cumulative_model_steps=1,
    )
    assert events[-2].payload["status"] == "interrupted"
    assert session is not None
    assert session.status == SessionStatus.INTERRUPTED


def test_cayu_app_can_resume_after_max_steps_limit():
    store = InMemorySessionStore()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_1",
                    name="echo",
                    arguments={"text": "again"},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.text_delta("Finished."),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    app = CayuApp(session_store=store)
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[EchoTool()],
    )

    initial_events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_resume_max_steps",
                messages=[Message.text("user", "loop")],
                max_steps=1,
            ),
        )
    )
    assert initial_events[-1].type == EventType.SESSION_INTERRUPTED

    resumed_events = asyncio.run(
        collect_resume_events(
            app,
            ResumeRequest(
                session_id="sess_resume_max_steps",
                messages=[Message.text("user", "continue")],
                max_steps=1,
            ),
        )
    )

    assert len(provider.requests) == 2
    assert resumed_events[0].type == EventType.SESSION_RESUMED
    assert resumed_events[-1].type == EventType.SESSION_COMPLETED
    session = asyncio.run(store.load("sess_resume_max_steps"))
    assert session is not None
    assert session.status == SessionStatus.COMPLETED

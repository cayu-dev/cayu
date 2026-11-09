"""Every pause kind is finished from its durable decision after a crash or a close.

User-input, wake, child and before-stop pauses go through the same decided
publication as an approval pause. The process is killed at every store write
after the pause's transition, and the stream is closed at its paused or
interrupted event; recovery, or the run's own settlement, must publish exactly
the events the pause owed and settle its decision, which releases its fence.
"""

from __future__ import annotations

import pytest
from tests.core.test_outcome_decision_crash_recovery import (
    _points_after_terminal_transition,
    _violations,
)
from tests.core.test_outcome_decision_crash_recovery import (
    backend as backend,
)
from tests.core.test_outcome_decision_crash_recovery import (
    harness_with_outcomes as harness_with_outcomes,
)
from tests.fault_injection import explore, harness
from tests.fault_injection.harness import BoundaryTool, FaultKind, ScriptedProvider

from cayu.agents import AgentSpec
from cayu.applications import CayuApp
from cayu.messages import Message, TextPart
from cayu.providers.base import ModelStreamEvent
from cayu.runtime.execution_identity import ExecutionProfileBehaviorIdentity
from cayu.runtime.loop_policies import BeforeStopDecision, LoopPolicy
from cayu.sessions._paired_writes import unsettled_owed_decision
from cayu.sessions.queries import SessionQuery
from cayu.tasks.memory import InMemoryTaskStore
from cayu.tools import SleepUntilTool
from cayu.tools.base import Tool
from cayu.tools.subagents import SubagentSpec, SubagentTool
from cayu.tools.user_input import UserInputTool

_PROMPTS = {
    "user-input": "fault-injection: ask the user",
    "wake": "fault-injection: sleep a second",
    "child": "fault-injection: delegate to a child",
    "before-stop": "fault-injection: stop before completing",
}
_CHILD_TASK = "fault-injection: the child asks the user"
_CHILD_AGENT = "fault-injection-child"
_PAUSED_EVENT = {
    "user-input": "event:interaction.paused#0",
    "wake": "event:interaction.paused#0",
    "child": "event:interaction.paused#0",
    "before-stop": "event:interaction.interrupted#0",
}


def _identity(name: str) -> ExecutionProfileBehaviorIdentity:
    return ExecutionProfileBehaviorIdentity(
        name=f"tests:outcome-pause-kinds:{name}", behavior_version="1", implementation_version="1"
    )


class _InterruptBeforeStop(LoopPolicy):
    @property
    def execution_profile_identity(self) -> ExecutionProfileBehaviorIdentity:
        return _identity("interrupt-before-stop")

    async def before_stop(self, context) -> BeforeStopDecision:
        return BeforeStopDecision.interrupt("fault-injection: stop here")


def _scripted(messages: list[Message], scripted) -> list[ModelStreamEvent]:
    last = messages[-1] if messages else None
    prompt = "".join(part.text for part in (last.content if last else ()) if type(part) is TextPart)
    if prompt in (_PROMPTS["user-input"], _CHILD_TASK):
        call = ModelStreamEvent.tool_call(
            id="call_ask", name="ask_user", arguments={"question": "Which one?"}
        )
    elif prompt == _PROMPTS["wake"]:
        call = ModelStreamEvent.tool_call(
            id="call_sleep", name="sleep_until", arguments={"duration_seconds": 1}
        )
    elif prompt == _PROMPTS["child"]:
        call = ModelStreamEvent.tool_call(
            id="call_child", name="subagent", arguments={"agent": "child", "task": _CHILD_TASK}
        )
    else:
        return scripted(messages)
    return [call, ModelStreamEvent.completed({"finish_reason": "tool_calls"})]


async def _decisions_settled(app: CayuApp, world: explore.World) -> str | None:
    """Every session's decision is settled, so its fence was released."""

    for session in (await world.store.list_sessions(SessionQuery(limit=100))).sessions:
        if unsettled_owed_decision(await world.store.load_checkpoint(session.id)) is not None:
            return f"{session.id}: its terminal decision is still unsettled"
    return None


@pytest.fixture(params=sorted(_PROMPTS))
def kind(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> str:
    pause_kind = request.param
    scripted = harness._scripted_response
    monkeypatch.setattr(
        harness, "_scripted_response", lambda messages: _scripted(messages, scripted)
    )

    def app(world: explore.World) -> CayuApp:
        # A wake schedules its task in a task store the harness doesn't explore.
        wake_tasks = (
            world.__dict__.setdefault(
                "wake_tasks", InMemoryTaskStore(clock=world.clock, ownership_clock=world.clock)
            )
            if pause_kind == "wake"
            else world.tasks
        )
        cayu_app = CayuApp(
            session_store=world.store,
            task_store=wake_tasks,
            enable_logging=False,
            loop_policies=[_InterruptBeforeStop()] if pause_kind == "before-stop" else None,
        )
        cayu_app.register_provider(ScriptedProvider(world.injector), default=True)
        tools: list[Tool] = [BoundaryTool("echo", world.injector), UserInputTool()]
        if pause_kind == "wake":
            tools.append(SleepUntilTool())
        if pause_kind == "child":
            tools.append(
                SubagentTool(
                    cayu_app,
                    agents={"child": SubagentSpec(agent_name=_CHILD_AGENT)},
                    execution_profile_identity=_identity("delegation"),
                )
            )
            cayu_app.register_agent(
                AgentSpec(name=_CHILD_AGENT, model="fault-injection-model"),
                tools=[UserInputTool()],
            )
        cayu_app.register_agent(
            AgentSpec(name=explore.AGENT_NAME, model="fault-injection-model"), tools=tools
        )
        return cayu_app

    monkeypatch.setattr(explore.World, "app", app)
    # These pauses are continued by their own resolutions, not by a new turn;
    # what this checks is that each one's owed publication and fence are settled.
    monkeypatch.setattr(explore, "_continue_to_completion", _decisions_settled)
    return pause_kind


def _scenario(kind: str) -> explore.Scenario:
    return explore.Scenario(
        kind, explore._streamed(lambda app, _: explore.run_turn(app, _PROMPTS[kind]))
    )


def test_recovery_finishes_a_pause_cut_short_by_a_crash(backend: str, kind: str) -> None:
    scenario = _scenario(kind)
    points = _points_after_terminal_transition(scenario)

    violations = _violations(scenario, FaultKind.CRASH, points)

    assert not violations, "\n".join(violations)


def test_closing_at_a_pause_finishes_its_publication(backend: str, kind: str) -> None:
    violations = _violations(_scenario(kind), FaultKind.STREAM_CLOSE, [_PAUSED_EVENT[kind]])

    assert not violations, "\n".join(violations)

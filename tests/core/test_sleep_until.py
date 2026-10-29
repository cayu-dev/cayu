"""Durable one-shot self-wake through ``sleep_until`` and its wake task."""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from datetime import UTC, datetime, timedelta

import pytest

from cayu import AgentSpec, CayuApp, EventType, Message, RunRequest
from cayu.approvals.user_input import PendingUserInput, UserInputResponse
from cayu.budgets.run_limits import RunLimits
from cayu.evals.testing import ScriptedModelProvider
from cayu.messages import ToolResultPart
from cayu.providers.base import ModelRequest, ModelStreamEvent
from cayu.runtime.execution_identity import ExecutionProfileBehaviorIdentity
from cayu.sessions.event_queries import EventQuery
from cayu.sessions.memory import InMemorySessionStore
from cayu.sessions.pending_action_contracts import PendingActionQuery
from cayu.sessions.queries import SessionQuery
from cayu.sessions.records import PendingActionKind, SessionStatus
from cayu.sessions.requests import InterruptSessionRequest
from cayu.storage import SQLiteSessionStore, SQLiteTaskStore
from cayu.tasks import InMemoryTaskStore, TaskStatus, session_wakes
from cayu.tasks.scheduling import TaskScheduleCancelRequest
from cayu.tasks.session_wakes import (
    SESSION_WAKE_TASK_TYPE,
    SessionWakeNotDue,
    SessionWakeOutcome,
    _handle_session_wake_task,
    deliver_session_wake,
    reconcile_session_wakes,
)
from cayu.tasks.worker import run_task_worker
from cayu.tools import SleepUntilTool, UserInputTool

_START = datetime(2027, 1, 15, 12, 0, tzinfo=UTC)


class _Clock:
    def __init__(self, now: datetime = _START) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += timedelta(seconds=seconds)


def _tool_results(request: ModelRequest) -> list[ToolResultPart]:
    return [
        part
        for message in request.messages
        for part in message.content
        if isinstance(part, ToolResultPart)
    ]


def _sleep_then_answer(arguments: dict, *, calls: list[ModelRequest] | None = None):
    def respond(request: ModelRequest):
        if calls is not None:
            calls.append(request)
        results = _tool_results(request)
        if not results:
            return [
                ModelStreamEvent.tool_call(
                    id="call_sleep", name="sleep_until", arguments=arguments
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ]
        return [
            ModelStreamEvent.text_delta(f"Woke with: {results[-1].content}"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ]

    return respond


class _StableScriptedProvider(ScriptedModelProvider):
    """A request-aware script that every reconstructed process builds the same way."""

    @property
    def execution_profile_identity(self) -> ExecutionProfileBehaviorIdentity:
        return ExecutionProfileBehaviorIdentity(
            name="tests:sleep-until:provider", behavior_version="1", implementation_version="1"
        )


def _app(session_store, task_store, clock, *, calls=None, arguments=None, tools=None) -> CayuApp:
    app = CayuApp(
        session_store=session_store,
        task_store=task_store,
        enable_logging=False,
        clock=clock,
    )
    app.register_provider(
        _StableScriptedProvider(
            response_factory=_sleep_then_answer(
                {"duration_seconds": 30} if arguments is None else arguments, calls=calls
            )
        ),
        default=True,
    )
    app.register_agent(
        AgentSpec(name="sleeper", model="scripted-model"),
        tools=[SleepUntilTool()] if tools is None else tools,
    )
    return app


async def _run(app: CayuApp, session_id: str) -> list:
    return [
        event
        async for event in app.run(
            RunRequest(
                agent_name="sleeper",
                session_id=session_id,
                messages=[Message.text("user", "wait, then report")],
            )
        )
    ]


async def _no_application_tasks(app, task, worker_id):
    raise AssertionError(f"Unexpected application task {task.type}.")


async def _work(app: CayuApp, *, max_tasks: int = 1) -> int:
    return await run_task_worker(
        app,
        app.task_store,
        _no_application_tasks,
        worker_id="wake-worker",
        max_tasks=max_tasks,
        poll_interval_s=0.01,
    )


def _sleep_results(transcript) -> list[ToolResultPart]:
    return [
        part
        for message in transcript
        for part in message.content
        if isinstance(part, ToolResultPart) and part.tool_call_id == "call_sleep"
    ]


async def _assert_woke_once(session_store, session_id: str, *, deadline: datetime) -> dict:
    session = await session_store.load(session_id)
    assert session is not None and session.status is SessionStatus.COMPLETED
    transcript = await session_store.load_transcript(session_id)
    results = _sleep_results(transcript)
    assert len(results) == 1 and results[0].is_error is False
    timer = json.loads(results[0].content)
    assert timer["status"] == "woke"
    assert timer["deadline"] == deadline.isoformat().replace("+00:00", "Z")
    assert [message.role for message in transcript] == ["user", "assistant", "tool", "assistant"]
    return timer


def test_sleep_until_parks_session_and_wake_task_continues_the_same_call() -> None:
    async def scenario() -> None:
        clock = _Clock()
        sessions, tasks = InMemorySessionStore(), InMemoryTaskStore(clock=clock)
        calls: list[ModelRequest] = []
        app = _app(sessions, tasks, clock, calls=calls)
        events = await _run(app, "s_sleep")

        assert events[-1].type is EventType.SESSION_INTERRUPTED
        assert events[-1].payload["interruption_type"] == "wake_scheduled"
        assert (await sessions.load("s_sleep")).status is SessionStatus.INTERRUPTED
        [task] = await tasks.list_tasks()
        assert task.type == SESSION_WAKE_TASK_TYPE
        assert task.status is TaskStatus.PENDING
        assert task.available_at == _START + timedelta(seconds=30)
        assert len(calls) == 1

        clock.advance(29)
        assert await tasks.claim_task("early-worker", lease_seconds=30) is None
        clock.advance(1)
        assert await _work(app) == 1

        timer = await _assert_woke_once(
            sessions, "s_sleep", deadline=_START + timedelta(seconds=30)
        )
        assert timer["requested"] == {"duration_seconds": 30}
        assert timer["deadline_resolved_at"] == _START.isoformat().replace("+00:00", "Z")
        assert len(calls) == 2
        assert (await tasks.load_task(task.id)).status is TaskStatus.COMPLETED

    asyncio.run(scenario())


async def _sleeping(app: CayuApp, sessions, session_id: str):
    await _run(app, session_id)
    checkpoint = await sessions.load_checkpoint(session_id)
    assert "pending_user_input" not in checkpoint
    pending = checkpoint["pending_wake"]["continuation"]
    assert pending["wake"]["task_id"] == f"cayu-session-wake-{pending['input_id']}"
    return pending


def test_wake_waits_for_its_deadline_and_cannot_be_answered_by_a_person() -> None:
    async def scenario() -> None:
        clock = _Clock()
        sessions, tasks = InMemorySessionStore(), InMemoryTaskStore(clock=clock)
        calls: list[ModelRequest] = []
        app = _app(sessions, tasks, clock, calls=calls)
        events = await _run(app, "s_gate")
        pending = (await sessions.load_checkpoint("s_gate"))["pending_wake"]["continuation"]
        assert not any(e.type is EventType.SESSION_AWAITING_USER_INPUT for e in events)
        assert events[-1].payload["wake"]["deadline"] == "2027-01-15T12:00:30Z"

        clock.advance(29.999)
        with pytest.raises(SessionWakeNotDue):
            await deliver_session_wake(app, session_id="s_gate", wake_id=pending["input_id"])
        with pytest.raises(RuntimeError, match="no pending user input"):
            async for _ in app.resolve_user_input(
                UserInputResponse(session_id="s_gate", input_id=pending["input_id"], answer="up")
            ):
                pass
        assert (await sessions.load("s_gate")).status is SessionStatus.INTERRUPTED
        assert len(calls) == 1

        clock.advance(0.001)  # Exactly the deadline: eligibility includes equality.
        outcome = await deliver_session_wake(app, session_id="s_gate", wake_id=pending["input_id"])
        assert outcome is SessionWakeOutcome.CONTINUED
        await _assert_woke_once(sessions, "s_gate", deadline=_START + timedelta(seconds=30))

    asyncio.run(scenario())


def test_duplicate_and_late_wake_deliveries_continue_the_session_once() -> None:
    async def scenario() -> None:
        clock = _Clock()
        sessions, tasks = InMemorySessionStore(), InMemoryTaskStore(clock=clock)
        calls: list[ModelRequest] = []
        app = _app(sessions, tasks, clock, calls=calls)
        pending = await _sleeping(app, sessions, "s_dup")
        clock.advance(3600)

        first, second = await asyncio.gather(
            deliver_session_wake(app, session_id="s_dup", wake_id=pending["input_id"]),
            deliver_session_wake(app, session_id="s_dup", wake_id=pending["input_id"]),
            return_exceptions=True,
        )
        outcomes = {first, second}
        assert SessionWakeOutcome.CONTINUED in outcomes, outcomes
        assert await _work(app) == 1
        again = await deliver_session_wake(app, session_id="s_dup", wake_id=pending["input_id"])
        assert again is SessionWakeOutcome.ALREADY_DELIVERED
        assert len(calls) == 2
        await _assert_woke_once(sessions, "s_dup", deadline=_START + timedelta(seconds=30))
        task = await tasks.load_task(pending["wake"]["task_id"])
        assert task.status is TaskStatus.COMPLETED
        assert task.result == {"outcome": "already_delivered"}

    asyncio.run(scenario())


@pytest.mark.parametrize("closure", ["interrupt", "delete"])
def test_closing_a_sleeping_session_cancels_its_wake_and_a_late_wake_does_nothing(closure) -> None:
    async def scenario() -> None:
        clock = _Clock()
        sessions, tasks = InMemorySessionStore(), InMemoryTaskStore(clock=clock)
        calls: list[ModelRequest] = []
        app = _app(sessions, tasks, clock, calls=calls)
        pending = await _sleeping(app, sessions, "s_closed")
        task_id = pending["wake"]["task_id"]

        if closure == "interrupt":
            async for _ in app.interrupt_session(
                InterruptSessionRequest(session_id="s_closed", reason="stop waiting")
            ):
                pass
            assert (await sessions.load("s_closed")).status is SessionStatus.INTERRUPTED
            checkpoint = await sessions.load_checkpoint("s_closed")
            assert "pending_user_input" not in checkpoint and "pending_wake" not in checkpoint
        else:
            await app.delete_session("s_closed", finished_only=False)
            assert await sessions.load("s_closed") is None
        assert (await tasks.load_task(task_id)).status is TaskStatus.CANCELLED
        # The stop closed the sleep in one transition: nothing reopens it.
        assert await reconcile_session_wakes(app) == 0
        if closure == "interrupt":
            assert (
                await sessions.load_runtime_publication_receipt(
                    "s_closed", f"wake-close:{pending['input_id']}"
                )
                is None
            )

        clock.advance(60)
        outcome = await deliver_session_wake(
            app, session_id="s_closed", wake_id=pending["input_id"]
        )
        assert outcome is SessionWakeOutcome.CLOSED
        assert len(calls) == 1

    asyncio.run(scenario())


def test_a_wake_task_left_running_after_interruption_settles_as_closed() -> None:
    async def scenario() -> None:
        clock = _Clock()
        sessions, tasks = InMemorySessionStore(), InMemoryTaskStore(clock=clock)
        calls: list[ModelRequest] = []
        app = _app(sessions, tasks, clock, calls=calls)
        pending = await _sleeping(app, sessions, "s_raced")
        clock.advance(30)
        claimed = await tasks.claim_task("racing-worker", lease_seconds=60)
        assert claimed is not None and claimed.id == pending["wake"]["task_id"]
        async for _ in app.interrupt_session(InterruptSessionRequest(session_id="s_raced")):
            pass
        await _handle_session_wake_task(app, claimed, "racing-worker")
        task = await tasks.load_task(claimed.id)
        assert task.status is TaskStatus.COMPLETED and task.result == {"outcome": "closed"}
        assert len(calls) == 1
        assert (await sessions.load("s_raced")).status is SessionStatus.INTERRUPTED

    asyncio.run(scenario())


def test_deadline_already_reached_returns_at_once_without_a_wake() -> None:
    async def scenario() -> None:
        clock = _Clock()
        sessions, tasks = InMemorySessionStore(), InMemoryTaskStore(clock=clock)
        for session_id, until in (
            ("s_past", "2027-01-15T11:00:00+00:00"),
            ("s_equal", "2027-01-15T13:00:00+01:00"),  # The same instant as now.
        ):
            app = _app(sessions, tasks, clock, arguments={"until": until})
            events = await _run(app, session_id)
            assert events[-1].type is EventType.SESSION_COMPLETED
            [result] = _sleep_results(await sessions.load_transcript(session_id))
            timer = json.loads(result.content)
            assert result.is_error is False
            assert timer["status"] == "deadline_already_reached"
            assert timer["wake_id"] is None
            assert timer["requested"] == {"until": until}
        assert await tasks.list_tasks() == []

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("arguments", "max_sleep", "message"),
    [
        ({"duration_seconds": 30 * 86400 + 1}, None, "at most 30 days"),
        ({"until": "2027-02-15T12:00:00Z"}, None, "at most 30 days"),
        ({"duration_seconds": 7200}, timedelta(hours=1), "at most 3600 seconds"),
        ({"until": "2027-01-16T12:00:00"}, None, "timezone offset"),
        ({"until": "tomorrow"}, None, "ISO 8601"),
        ({"until": "9999-12-31T23:59:59-23:59"}, None, "out of range"),
        ({}, None, "exactly one"),
        ({"duration_seconds": 5, "until": "2027-01-15T12:01:00Z"}, None, "exactly one"),
    ],
)
def test_refused_sleep_arguments_return_an_error_and_schedule_nothing(
    arguments, max_sleep, message
) -> None:
    async def scenario() -> None:
        clock = _Clock()
        sessions, tasks = InMemorySessionStore(), InMemoryTaskStore(clock=clock)
        tool = SleepUntilTool() if max_sleep is None else SleepUntilTool(max_sleep=max_sleep)
        app = _app(sessions, tasks, clock, arguments=arguments, tools=[tool])
        events = await _run(app, "s_refused")
        assert events[-1].type is EventType.SESSION_COMPLETED
        [result] = _sleep_results(await sessions.load_transcript("s_refused"))
        assert result.is_error is True
        assert message in result.content
        assert await tasks.list_tasks() == []

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "arguments",
    [
        {"duration_seconds": 0},
        {"duration_seconds": -1},
        {"duration_seconds": float("nan")},
        {"duration_seconds": float("inf")},
        {"duration_seconds": True},
        {"duration_seconds": "10"},
        {"duration_seconds": 1e300},
        {"until": 1700000000},
        {"until": "2027-01-15T12:00:00Z", "reason": "extra"},
        {"until": "9999-12-31T23:59:59-23:59"},
        {"until": "0001-01-01T00:00:00+23:59"},
    ],
)
def test_plan_wake_refuses_malformed_arguments(arguments) -> None:
    with pytest.raises(ValueError):
        SleepUntilTool().plan_wake(arguments, now=_START)


def test_plan_wake_resolves_one_utc_deadline_once() -> None:
    tool = SleepUntilTool()
    relative = tool.plan_wake({"duration_seconds": 1.5}, now=_START)
    assert relative.deadline == _START + timedelta(seconds=1.5)
    assert relative.resolved_at == _START and not relative.reached
    absolute = tool.plan_wake({"until": "2027-01-15T14:00:00+02:00"}, now=_START)
    assert absolute.deadline == _START and absolute.reached
    assert absolute.deadline.tzinfo is UTC
    edge = tool.plan_wake({"duration_seconds": 30 * 86400}, now=_START)
    assert edge.deadline == _START + timedelta(days=30)
    with pytest.raises(ValueError, match="timezone-aware"):
        tool.plan_wake({"duration_seconds": 1}, now=_START.replace(tzinfo=None))
    with pytest.raises(ValueError, match="positive timedelta"):
        SleepUntilTool(max_sleep=timedelta(0))


@pytest.mark.parametrize("scope", ["run", "session"])
def test_sleep_is_left_out_of_run_limits_but_not_session_limits(scope) -> None:
    async def scenario() -> None:
        # Session-scope elapsed time starts at the stored session creation time.
        clock = _Clock(datetime.now(UTC))
        sessions, tasks = InMemorySessionStore(), InMemoryTaskStore(clock=clock)
        calls: list[ModelRequest] = []
        app = _app(sessions, tasks, clock, calls=calls, arguments={"duration_seconds": 3600})
        events = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="sleeper",
                    session_id="s_limits",
                    messages=[Message.text("user", "wait an hour")],
                    limits=RunLimits(max_elapsed_seconds=600, scope=scope),
                )
            )
        ]
        assert events[-1].type is EventType.SESSION_INTERRUPTED
        assert events[-1].payload["interruption_type"] == "wake_scheduled"
        clock.advance(3600)
        assert await asyncio.wait_for(_work(app), 30) == 1
        session = await sessions.load("s_limits")
        if scope == "run":
            assert session.status is SessionStatus.COMPLETED
            assert len(calls) == 2
        else:
            # Session-scope limits stay wall-clock: an hour asleep spends the budget.
            assert session.status is not SessionStatus.COMPLETED
            assert len(calls) == 1

    asyncio.run(scenario())


def test_child_session_refuses_to_sleep_for_now() -> None:
    from tests.core.test_foreground_subagent_recovery import _identity
    from tests.core.test_tool_round_execution_identities import _SequencedProvider

    from cayu.sessions.queries import SessionQuery
    from cayu.tools.subagents import SubagentSpec, SubagentTool

    async def scenario() -> None:
        clock = _Clock()
        sessions, tasks = InMemorySessionStore(), InMemoryTaskStore(clock=clock)
        provider = _SequencedProvider(
            [
                [
                    ModelStreamEvent.tool_call(
                        id="spawn", name="subagent", arguments={"agent": "child", "task": "wait"}
                    ),
                    ModelStreamEvent.completed(),
                ],
                [
                    ModelStreamEvent.tool_call(
                        id="call_sleep", name="sleep_until", arguments={"duration_seconds": 60}
                    ),
                    ModelStreamEvent.completed(),
                ],
                [ModelStreamEvent.text_delta("child done"), ModelStreamEvent.completed()],
                [ModelStreamEvent.text_delta("parent done"), ModelStreamEvent.completed()],
            ]
        )
        app = CayuApp(session_store=sessions, task_store=tasks, enable_logging=False, clock=clock)
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="parent", model="test"),
            tools=[
                SubagentTool(
                    app,
                    agents={"child": SubagentSpec(agent_name="child")},
                    execution_profile_identity=_identity("sleeping-child"),
                )
            ],
        )
        app.register_agent(AgentSpec(name="child", model="test"), tools=[SleepUntilTool()])
        events = [
            event
            async for event in app.run(
                RunRequest(
                    session_id="parent",
                    agent_name="parent",
                    messages=[Message.text("user", "delegate")],
                )
            )
        ]
        assert events[-1].type is EventType.SESSION_COMPLETED
        [child] = (await sessions.list_sessions(SessionQuery(parent_session_id="parent"))).sessions
        [result] = _sleep_results(await sessions.load_transcript(child.id))
        assert result.is_error is True and "did not pause" in result.content
        assert await tasks.list_tasks() == []

    asyncio.run(scenario())


def test_a_round_with_an_approval_pauses_for_approval_and_never_sleeps_silently() -> None:
    from tests.core.test_tool_round_execution_identities import _RecordingTool, _SequencedProvider

    from cayu.approvals.tools import ToolApprovalDecision, ToolApprovalRequest
    from cayu.tools.policy import AlwaysRequireApprovalToolPolicy

    async def scenario() -> None:
        clock = _Clock()
        sessions, tasks = InMemorySessionStore(), InMemoryTaskStore(clock=clock)
        recorder = _RecordingTool()
        provider = _SequencedProvider(
            [
                [
                    ModelStreamEvent.tool_call(
                        id="call_record", name="record", arguments={"value": 7}
                    ),
                    ModelStreamEvent.tool_call(
                        id="call_sleep", name="sleep_until", arguments={"duration_seconds": 60}
                    ),
                    ModelStreamEvent.completed(),
                ],
                [ModelStreamEvent.text_delta("done"), ModelStreamEvent.completed()],
            ]
        )
        app = CayuApp(session_store=sessions, task_store=tasks, enable_logging=False, clock=clock)
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="sleeper", model="test"),
            tools=[recorder, SleepUntilTool()],
            tool_policy=AlwaysRequireApprovalToolPolicy(tools=["record"]),
        )
        events = await _run(app, "s_approval")
        approval = next(e for e in events if e.type is EventType.TOOL_CALL_APPROVAL_REQUESTED)
        assert recorder.values == [] and await tasks.list_tasks() == []
        async for _ in app.resolve_tool_approval(
            ToolApprovalRequest(
                session_id="s_approval",
                approval_id=approval.payload["approval_id"],
                tool_round_id=approval.payload["tool_round_id"],
                tool_call_id=approval.payload["tool_call_id"],
                decision=ToolApprovalDecision.APPROVE,
            )
        ):
            pass
        assert recorder.values == [7]
        [result] = _sleep_results(await sessions.load_transcript("s_approval"))
        assert result.is_error is True and "did not pause" in result.content
        assert await tasks.list_tasks() == []

    asyncio.run(scenario())


def test_reconcile_creates_a_wake_task_that_a_crash_left_out(monkeypatch) -> None:

    async def scenario() -> None:
        clock = _Clock()
        sessions, tasks = InMemorySessionStore(), InMemoryTaskStore(clock=clock)
        calls: list[ModelRequest] = []
        app = _app(sessions, tasks, clock, calls=calls)

        async def lost(app, pending):
            raise ConnectionError("task store unavailable")

        with monkeypatch.context() as patched:
            patched.setattr(session_wakes, "_publish_session_wake", lost)
            pending = await _sleeping(app, sessions, "s_reconcile")
        assert (await sessions.load("s_reconcile")).status is SessionStatus.INTERRUPTED
        assert await tasks.list_tasks() == []

        assert await reconcile_session_wakes(app) == 1
        assert await reconcile_session_wakes(app) == 0
        [task] = await tasks.list_tasks()
        assert task.id == pending["wake"]["task_id"]
        clock.advance(30)
        assert await _work(app) == 1
        await _assert_woke_once(sessions, "s_reconcile", deadline=_START + timedelta(seconds=30))
        assert len(calls) == 2

    asyncio.run(scenario())


def test_a_rebuilt_application_wakes_the_session_over_the_same_memory_stores() -> None:
    async def scenario() -> None:
        clock = _Clock()
        sessions, tasks = InMemorySessionStore(), InMemoryTaskStore(clock=clock)
        first_calls: list[ModelRequest] = []
        first = _app(sessions, tasks, clock, calls=first_calls)
        await _sleeping(first, sessions, "s_rebuilt")
        await first.aclose()

        clock.advance(30)
        later_calls: list[ModelRequest] = []
        rebuilt = _app(sessions, tasks, clock, calls=later_calls)
        assert await _work(rebuilt) == 1
        await _assert_woke_once(sessions, "s_rebuilt", deadline=_START + timedelta(seconds=30))
        assert (len(first_calls), len(later_calls)) == (1, 1)

    asyncio.run(scenario())


_SQLITE_PROCESS = """
import asyncio, json, sys
from datetime import UTC, datetime
from cayu import AgentSpec, CayuApp, Message, RunRequest
from cayu.evals.testing import ScriptedModelProvider
from cayu.messages import ToolResultPart
from cayu.providers.base import ModelStreamEvent
from cayu.storage import SQLiteSessionStore, SQLiteTaskStore
from cayu.tasks.worker import run_task_worker
from cayu.tools import SleepUntilTool

from cayu.runtime.execution_identity import ExecutionProfileBehaviorIdentity

phase, root = sys.argv[1], sys.argv[2]
calls = []

class Provider(ScriptedModelProvider):
    @property
    def execution_profile_identity(self):
        return ExecutionProfileBehaviorIdentity(
            name="tests:sleep-until:provider", behavior_version="1", implementation_version="1"
        )

def respond(request):
    calls.append(request)
    results = [p for m in request.messages for p in m.content if isinstance(p, ToolResultPart)]
    if not results:
        return [
            ModelStreamEvent.tool_call(
                id="call_sleep", name="sleep_until", arguments={"duration_seconds": 1}
            ),
            ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
        ]
    return [ModelStreamEvent.text_delta("awake"), ModelStreamEvent.completed({"finish_reason": "stop"})]

async def no_application_tasks(app, task, worker_id):
    raise AssertionError(task.type)

async def main():
    app = CayuApp(
        session_store=SQLiteSessionStore(f"{root}/sessions.db"),
        task_store=SQLiteTaskStore(f"{root}/tasks.db"),
        enable_logging=False,
    )
    app.register_provider(Provider(response_factory=respond), default=True)
    app.register_agent(AgentSpec(name="sleeper", model="scripted-model"), tools=[SleepUntilTool()])
    try:
        if phase == "sleep":
            events = [
                event
                async for event in app.run(
                    RunRequest(
                        agent_name="sleeper",
                        session_id="s_sqlite",
                        messages=[Message.text("user", "wait")],
                    )
                )
            ]
            [task] = await app.task_store.list_tasks()
            result = {
                "last": events[-1].type.value,
                "task_status": task.status.value,
                "available_at": task.available_at.isoformat(),
            }
        else:
            handled = await run_task_worker(
                app,
                app.task_store,
                no_application_tasks,
                worker_id="fresh-worker",
                max_tasks=1,
                poll_interval_s=0.05,
            )
            session = await app.session_store.load("s_sqlite")
            transcript = await app.session_store.load_transcript("s_sqlite")
            [timer] = [
                p.content for m in transcript for p in m.content
                if isinstance(p, ToolResultPart) and p.tool_call_id == "call_sleep"
            ]
            [task] = await app.task_store.list_tasks()
            result = {
                "handled": handled,
                "status": session.status.value,
                "timer": json.loads(timer),
                "roles": [m.role for m in transcript],
                "task_status": task.status.value,
                "woke_at": datetime.now(UTC).isoformat(),
            }
        result["model_calls"] = len(calls)
        print(json.dumps(result))
    finally:
        await app.aclose()

asyncio.run(main())
"""


def test_a_fresh_process_wakes_a_session_that_slept_in_sqlite(tmp_path) -> None:
    def phase(name: str) -> dict:
        result = subprocess.run(
            [sys.executable, "-c", _SQLITE_PROCESS, name, str(tmp_path)],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, result.stderr
        return json.loads(result.stdout.strip().splitlines()[-1])

    slept = phase("sleep")
    assert slept == {
        "last": "session.interrupted",
        "task_status": "pending",
        "available_at": slept["available_at"],
        "model_calls": 1,
    }
    woke = phase("wake")
    assert woke["handled"] == 1
    assert woke["status"] == "completed"
    assert woke["task_status"] == "completed"
    assert woke["model_calls"] == 1
    assert woke["roles"] == ["user", "assistant", "tool", "assistant"]
    assert woke["timer"]["status"] == "woke"
    assert woke["timer"]["requested"] == {"duration_seconds": 1}
    assert datetime.fromisoformat(woke["woke_at"]) >= datetime.fromisoformat(slept["available_at"])


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
def test_a_sleeping_session_is_inspectable_as_interrupted_not_running(
    backend, tmp_path, request
) -> None:

    async def scenario() -> None:
        clock = _Clock(datetime.now(UTC))
        sessions, tasks = _stores(backend, tmp_path, request, clock)
        app = _app(sessions, tasks, clock)
        try:
            await _sleeping(app, sessions, "s_inspect")
            assert (await sessions.load("s_inspect")).status is SessionStatus.INTERRUPTED
            assert await app.recover_abandoned_sessions() is not None
            assert (await sessions.load("s_inspect")).status is SessionStatus.INTERRUPTED
            page = await sessions.query_pending_actions(PendingActionQuery(session_id="s_inspect"))
            [action] = page.actions
            assert action.kind is PendingActionKind.WAKE
            assert action.tool_name == "sleep_until"
            assert action.question is None and action.attention_id is None
            assert action.wake_at is not None
            people = await sessions.query_pending_actions(
                PendingActionQuery(session_id="s_inspect", kind=PendingActionKind.USER_INPUT)
            )
            assert people.actions == []
        finally:
            await app.aclose()

    asyncio.run(scenario())


def test_a_wake_for_another_pause_cannot_resolve_the_current_sleep() -> None:
    async def scenario() -> None:
        clock = _Clock()
        sessions, tasks = InMemorySessionStore(), InMemoryTaskStore(clock=clock)
        calls: list[ModelRequest] = []
        app = _app(sessions, tasks, clock, calls=calls)
        current = await _sleeping(app, sessions, "s_other")
        clock.advance(30)
        stale = await deliver_session_wake(app, session_id="s_other", wake_id="an-earlier-wake")
        assert stale is SessionWakeOutcome.CLOSED
        assert (await sessions.load("s_other")).status is SessionStatus.INTERRUPTED
        assert len(calls) == 1
        fresh = await deliver_session_wake(app, session_id="s_other", wake_id=current["input_id"])
        assert fresh is SessionWakeOutcome.CONTINUED
        assert len(calls) == 2

    asyncio.run(scenario())


def test_ask_user_pause_is_unchanged_beside_sleep_until() -> None:
    def respond(request: ModelRequest):
        if not _tool_results(request):
            return [
                ModelStreamEvent.tool_call(
                    id="call_ask", name="ask_user", arguments={"question": "Which env?"}
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ]
        return [ModelStreamEvent.text_delta("ok"), ModelStreamEvent.completed()]

    async def scenario() -> None:
        clock = _Clock()
        sessions, tasks = InMemorySessionStore(), InMemoryTaskStore(clock=clock)
        app = CayuApp(session_store=sessions, task_store=tasks, enable_logging=False, clock=clock)
        app.register_provider(_StableScriptedProvider(response_factory=respond), default=True)
        app.register_agent(
            AgentSpec(name="sleeper", model="scripted-model"),
            tools=[UserInputTool(), SleepUntilTool()],
        )
        events = await _run(app, "s_ask")
        awaiting = next(e for e in events if e.type is EventType.SESSION_AWAITING_USER_INPUT)
        assert awaiting.payload["question"] == "Which env?"
        pending = (await sessions.load_checkpoint("s_ask"))["pending_user_input"]
        assert "wake" not in pending and pending["question"] == "Which env?"
        assert await tasks.list_tasks() == []
        # A question is not a wake: a wake delivery leaves it to a person.
        outcome = await deliver_session_wake(app, session_id="s_ask", wake_id=pending["input_id"])
        assert outcome is SessionWakeOutcome.CLOSED
        assert (await sessions.load_checkpoint("s_ask"))["pending_user_input"] == pending
        async for _ in app.resolve_user_input(
            UserInputResponse(
                session_id="s_ask", input_id=awaiting.payload["input_id"], answer="prod"
            )
        ):
            pass
        assert (await sessions.load("s_ask")).status is SessionStatus.COMPLETED

    asyncio.run(scenario())


def _stores(backend: str, tmp_path, request, clock):
    if backend == "memory":
        return InMemorySessionStore(), InMemoryTaskStore(clock=clock)
    if backend == "sqlite":
        return SQLiteSessionStore(tmp_path / "sessions.db"), SQLiteTaskStore(tmp_path / "tasks.db")
    from cayu.storage import PostgresSessionStore
    from cayu.storage.migrations import SchemaMode
    from cayu.storage.tasks_postgres import PostgresTaskStore

    dsn = request.getfixturevalue("postgres_dsn")
    return (
        PostgresSessionStore(dsn, schema_mode=SchemaMode.CREATE),
        PostgresTaskStore(dsn, schema_mode=SchemaMode.CREATE),
    )


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
def test_racing_wake_publication_and_reconciliation_converge_on_one_task(
    backend, tmp_path, request
) -> None:
    from cayu.tasks.session_wakes import _publish_session_wake

    async def scenario() -> None:
        clock = _Clock(datetime.now(UTC))
        sessions, tasks = _stores(backend, tmp_path, request, clock)
        app = _app(sessions, tasks, clock)
        try:
            pending = await _sleeping(app, sessions, "s_race")
            from cayu.approvals.user_input import PendingUserInput

            continuation = PendingUserInput.model_validate(pending)
            results = await asyncio.gather(
                _publish_session_wake(app, continuation),
                _publish_session_wake(app, continuation),
                reconcile_session_wakes(app),
                reconcile_session_wakes(app),
            )
            assert results[2:] == [0, 0]
            wake_tasks = [
                task
                for task in await tasks.list_tasks()
                if task.type == SESSION_WAKE_TASK_TYPE
                and task.input["wake_id"] == pending["input_id"]
            ]
            assert [task.id for task in wake_tasks] == [pending["wake"]["task_id"]]
        finally:
            await app.aclose()

    asyncio.run(scenario())


def test_a_failed_wake_attempt_schedules_a_successor_that_continues_the_session() -> None:
    from cayu.tasks.session_wakes import session_wake_attempts

    async def scenario() -> None:
        clock = _Clock()
        sessions, tasks = InMemorySessionStore(), InMemoryTaskStore(clock=clock)
        first_calls: list[ModelRequest] = []
        pending = await _sleeping(
            _app(sessions, tasks, clock, calls=first_calls), sessions, "s_retry"
        )
        clock.advance(30)

        # A worker built without the session's provider cannot continue it.
        broken = CayuApp(
            session_store=sessions, task_store=tasks, enable_logging=False, clock=clock
        )
        broken.register_agent(
            AgentSpec(name="sleeper", model="scripted-model"), tools=[SleepUntilTool()]
        )
        assert await _work(broken) == 1
        failed, successor = await session_wake_attempts(broken, pending["input_id"])
        assert failed.status is TaskStatus.FAILED and failed.error is not None
        assert successor.status is TaskStatus.PENDING
        assert successor.input["attempt"] == 2
        assert successor.available_at == failed.available_at + timedelta(seconds=30)
        assert (await sessions.load("s_retry")).status is SessionStatus.INTERRUPTED
        [action] = (
            await sessions.query_pending_actions(PendingActionQuery(session_id="s_retry"))
        ).actions
        assert action.kind is PendingActionKind.WAKE

        clock.advance(30)
        later_calls: list[ModelRequest] = []
        fixed = _app(sessions, tasks, clock, calls=later_calls)
        assert await _work(fixed) == 1
        await _assert_woke_once(sessions, "s_retry", deadline=_START + timedelta(seconds=30))
        assert (len(first_calls), len(later_calls)) == (1, 1)
        statuses = [task.status for task in await session_wake_attempts(fixed, pending["input_id"])]
        assert statuses == [TaskStatus.FAILED, TaskStatus.COMPLETED]

    asyncio.run(scenario())


def test_reconcile_replaces_an_ended_attempt_until_the_retry_window_is_spent(
    monkeypatch,
) -> None:
    from cayu.tasks.session_wakes import session_wake_attempts

    monkeypatch.setattr(session_wakes, "SESSION_WAKE_RETRY_WINDOW", timedelta(seconds=45))

    async def scenario() -> None:
        clock = _Clock()
        sessions, tasks = InMemorySessionStore(), InMemoryTaskStore(clock=clock)
        app = _app(sessions, tasks, clock)
        pending = await _sleeping(app, sessions, "s_ended")
        for created in (1, 0):
            [*_, current] = await session_wake_attempts(app, pending["input_id"])
            await app.cancel_scheduled_task(
                TaskScheduleCancelRequest(
                    task_id=current.id,
                    operation_id=f"stray-{current.id}",
                    expected_revision=current.schedule.revision,
                )
            )
            assert await reconcile_session_wakes(app) == created
        attempts = await session_wake_attempts(app, pending["input_id"])
        assert [task.input["attempt"] for task in attempts] == [1, 2]
        assert attempts[1].available_at == attempts[0].available_at + timedelta(seconds=30)
        # The window is spent; a direct delivery still continues the session.
        clock.advance(3600)
        assert (
            await deliver_session_wake(app, session_id="s_ended", wake_id=pending["input_id"])
        ) is SessionWakeOutcome.CONTINUED

    asyncio.run(scenario())


def test_a_crash_after_the_sleep_opened_recovers_asleep_and_still_wakes() -> None:
    from tests.core._execution_profile_fixtures import rebind_test_invocation

    from cayu.sessions.recovery import IncompleteSessionRecoveryRequest

    async def scenario() -> None:
        clock = _Clock()
        sessions, tasks = InMemorySessionStore(), InMemoryTaskStore(clock=clock)
        calls: list[ModelRequest] = []
        app = _app(sessions, tasks, clock, calls=calls)
        pending = await _sleeping(app, sessions, "s_crash")
        await rebind_test_invocation(sessions, "s_crash")
        result = await app.recover_incomplete_session(
            IncompleteSessionRecoveryRequest(session_id="s_crash")
        )
        [interrupted] = [e for e in result.events if e.type is EventType.SESSION_INTERRUPTED]
        assert interrupted.payload["interruption_type"] == "wake_scheduled"
        assert "pending_wake" in await sessions.load_checkpoint("s_crash")
        clock.advance(30)
        assert (
            await deliver_session_wake(app, session_id="s_crash", wake_id=pending["input_id"])
        ) is SessionWakeOutcome.CONTINUED
        assert len(calls) == 2

    asyncio.run(scenario())


def test_a_sleeping_session_refuses_resume_and_fork() -> None:
    from cayu.sessions.requests import ForkSessionRequest, ResumeRequest

    async def scenario() -> None:
        clock = _Clock()
        sessions, tasks = InMemorySessionStore(), InMemoryTaskStore(clock=clock)
        app = _app(sessions, tasks, clock)
        await _sleeping(app, sessions, "s_refuse")
        with pytest.raises(RuntimeError, match="asleep in sleep_until"):
            async for _ in app.resume(
                ResumeRequest(session_id="s_refuse", messages=[Message.text("user", "hurry")])
            ):
                pass
        with pytest.raises(RuntimeError, match="asleep in sleep_until"):
            async for _ in app.fork_session(ForkSessionRequest(source_session_id="s_refuse")):
                pass
        assert "pending_wake" in await sessions.load_checkpoint("s_refuse")

    asyncio.run(scenario())


def test_registration_refuses_a_wake_tool_it_cannot_serve() -> None:
    from cayu.tools.base import Tool, ToolResult, ToolSpec

    class NoPlan(Tool):
        spec = ToolSpec(name="nap", description="Nap.", input_schema={"type": "object"})
        pauses_session = True
        pause_kind = "wake"

        async def run(self, ctx, args):
            return ToolResult(content="")

    without_tasks = CayuApp(session_store=InMemorySessionStore(), enable_logging=False)
    with pytest.raises(ValueError, match="task store that supports task scheduling"):
        without_tasks.register_agent(AgentSpec(name="a", model="m"), tools=[SleepUntilTool()])
    with_tasks = CayuApp(
        session_store=InMemorySessionStore(), task_store=InMemoryTaskStore(), enable_logging=False
    )
    with pytest.raises(ValueError, match="unsupported pause"):
        with_tasks.register_agent(AgentSpec(name="a", model="m"), tools=[NoPlan()])


def test_an_older_runtime_cannot_represent_a_sleeping_session() -> None:
    from cayu.sessions.checkpoints import (
        CHECKPOINT_SCHEMA_VERSION_KEY,
        CURRENT_CHECKPOINT_SCHEMA_VERSION,
        decode_runtime_checkpoint,
        runtime_checkpoint_writer_view,
    )

    async def scenario() -> None:
        clock = _Clock()
        sessions, tasks = InMemorySessionStore(), InMemoryTaskStore(clock=clock)
        await _sleeping(_app(sessions, tasks, clock), sessions, "s_old")
        checkpoint = await sessions.load_checkpoint("s_old")
        assert checkpoint[CHECKPOINT_SCHEMA_VERSION_KEY] == CURRENT_CHECKPOINT_SCHEMA_VERSION == 12
        with pytest.raises(ValueError, match="sleeping tool call"):
            runtime_checkpoint_writer_view(checkpoint, writer_version=11, session_id="s_old")
        # Application data from before wakes never becomes wake authority.
        legacy = {**checkpoint, CHECKPOINT_SCHEMA_VERSION_KEY: 11}
        assert "pending_wake" not in decode_runtime_checkpoint(legacy, session_id="s_old")

    asyncio.run(scenario())


def test_interrupting_closes_the_sleep_even_if_the_caller_stops_reading() -> None:
    async def scenario() -> None:
        clock = _Clock()
        sessions, tasks = InMemorySessionStore(), InMemoryTaskStore(clock=clock)
        app = _app(sessions, tasks, clock)
        pending = await _sleeping(app, sessions, "s_abandoned_read")
        stream = app.interrupt_session(InterruptSessionRequest(session_id="s_abandoned_read"))
        await anext(stream)
        await stream.aclose()
        assert "pending_wake" not in await sessions.load_checkpoint("s_abandoned_read")
        task = await tasks.load_task(pending["wake"]["task_id"])
        assert task.status is TaskStatus.CANCELLED

    asyncio.run(scenario())


def test_a_session_can_sleep_again_after_it_wakes() -> None:
    def respond(request: ModelRequest):
        slept = len(_tool_results(request))
        if slept < 2:
            return [
                ModelStreamEvent.tool_call(
                    id=f"call_sleep_{slept}",
                    name="sleep_until",
                    arguments={"duration_seconds": 30},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ]
        return [ModelStreamEvent.text_delta("twice"), ModelStreamEvent.completed()]

    async def scenario() -> None:
        clock = _Clock()
        sessions, tasks = InMemorySessionStore(), InMemoryTaskStore(clock=clock)
        app = CayuApp(session_store=sessions, task_store=tasks, enable_logging=False, clock=clock)
        app.register_provider(_StableScriptedProvider(response_factory=respond), default=True)
        app.register_agent(
            AgentSpec(name="sleeper", model="scripted-model"), tools=[SleepUntilTool()]
        )
        await _run(app, "s_twice")
        clock.advance(30)
        assert await _work(app) == 1
        assert "pending_wake" in await sessions.load_checkpoint("s_twice")
        clock.advance(30)
        assert await _work(app) == 1
        assert (await sessions.load("s_twice")).status is SessionStatus.COMPLETED
        assert len([t for t in await tasks.list_tasks() if t.type == SESSION_WAKE_TASK_TYPE]) == 2

    asyncio.run(scenario())


def test_an_approval_gated_sleep_asks_first_and_never_sleeps_unapproved() -> None:
    from tests.core.test_tool_round_execution_identities import _SequencedProvider

    from cayu.approvals.tools import ToolApprovalDecision, ToolApprovalRequest
    from cayu.tools.policy import AlwaysRequireApprovalToolPolicy

    async def scenario() -> None:
        clock = _Clock()
        sessions, tasks = InMemorySessionStore(), InMemoryTaskStore(clock=clock)
        provider = _SequencedProvider(
            [
                [
                    ModelStreamEvent.tool_call(
                        id="call_sleep", name="sleep_until", arguments={"duration_seconds": 60}
                    ),
                    ModelStreamEvent.completed(),
                ],
                [ModelStreamEvent.text_delta("done"), ModelStreamEvent.completed()],
            ]
        )
        app = CayuApp(session_store=sessions, task_store=tasks, enable_logging=False, clock=clock)
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="sleeper", model="test"),
            tools=[SleepUntilTool()],
            tool_policy=AlwaysRequireApprovalToolPolicy(tools=["sleep_until"]),
        )
        events = await _run(app, "s_gated")
        approval = next(e for e in events if e.type is EventType.TOOL_CALL_APPROVAL_REQUESTED)
        assert await tasks.list_tasks() == []
        async for _ in app.resolve_tool_approval(
            ToolApprovalRequest(
                session_id="s_gated",
                approval_id=approval.payload["approval_id"],
                tool_round_id=approval.payload["tool_round_id"],
                tool_call_id=approval.payload["tool_call_id"],
                decision=ToolApprovalDecision.APPROVE,
            )
        ):
            pass
        [result] = _sleep_results(await sessions.load_transcript("s_gated"))
        assert result.is_error is True and "sleep_until itself" in result.content
        assert await tasks.list_tasks() == []

    asyncio.run(scenario())


@pytest.mark.parametrize("tampered", [False, True])
def test_wake_closure_must_hand_over_its_exact_continuation(tampered) -> None:
    from cayu._validation import copy_durable_record
    from cayu.approvals.user_input import pending_user_input_identity
    from cayu.approvals.wakes import (
        pending_session_wake_digest,
        pending_session_wake_from_checkpoint,
    )
    from cayu.events import Event
    from cayu.runtime import _approval_publication as approval_publication
    from cayu.sessions._user_input_publication import _validate_user_input_checkpoint_mutation

    async def scenario() -> None:
        clock = _Clock()
        sessions, tasks = InMemorySessionStore(), InMemoryTaskStore(clock=clock)
        await _sleeping(_app(sessions, tasks, clock), sessions, "s_exact")
        session = await sessions.load("s_exact")
        checkpoint = await sessions.load_checkpoint("s_exact")
        wake = pending_session_wake_from_checkpoint(checkpoint)
        pending = wake.continuation
        handed_over = pending.model_dump(mode="json")
        if tampered:
            handed_over["question"] = "Answer me instead."
        target = copy_durable_record(checkpoint, "checkpoint")
        target.pop("pending_wake")
        target["pending_user_input"] = handed_over
        event = Event(
            type=EventType.SESSION_CHECKPOINTED,
            session_id=session.id,
            interaction_id=pending.source_interaction_id,
            payload={
                "checkpoint": "pending_wake",
                "transition": "woke",
                "wake_id": wake.wake_id,
                "input_id": pending.input_id,
                "tool_call_id": pending.tool_call_id,
                "source_run_epoch": pending.source_run_epoch,
                "pause_digest": pending_user_input_identity(pending)["pause_digest"],
            },
        )
        prepared = approval_publication.prepare_pending_action_publication(
            session_id=session.id,
            publication_id=f"wake-close:{wake.wake_id}",
            kind="wake-close",
            intent={
                **pending_user_input_identity(pending),
                "wake_pause_digest": pending_session_wake_digest(wake),
                "transition": "woke",
                "event_ids": [event.id],
            },
            source_checkpoint=checkpoint,
            target_checkpoint=target,
            events=[event],
            expected_statuses={SessionStatus.INTERRUPTED},
            expected_run_epoch=session.run_epoch,
            expected_transcript_cursor=0,
        )

        def validate() -> None:
            _validate_user_input_checkpoint_mutation(
                prepared.request,
                checkpoint,
                session_id=session.id,
                session_instance_id=session.instance_id,
                current_run_epoch=session.run_epoch,
            )

        if tampered:
            with pytest.raises(ValueError, match="exact continuation"):
                validate()
        else:
            validate()

    asyncio.run(scenario())


def test_failed_attempts_keep_retrying_at_the_capped_backoff_within_the_window() -> None:
    from types import SimpleNamespace

    from cayu.tasks.session_wakes import SESSION_WAKE_RETRY_WINDOW, _attempt_task, _successor

    first = _attempt_task("s", "w", _START, 1, _START)
    attempt = SimpleNamespace(
        id=first.task_id, input=first.input, available_at=_START, started_at=None, created_at=_START
    )
    spacings = []
    while (successor := _successor(attempt)) is not None:
        spacings.append(successor.available_at - attempt.available_at)
        attempt = SimpleNamespace(
            id=successor.task_id,
            input=successor.input,
            available_at=successor.available_at,
            started_at=None,
            created_at=_START,
        )
    assert spacings[:3] == [timedelta(seconds=30), timedelta(seconds=60), timedelta(seconds=120)]
    assert max(spacings) == timedelta(hours=1)
    assert _START + SESSION_WAKE_RETRY_WINDOW - timedelta(hours=1) < attempt.available_at
    assert attempt.available_at <= _START + SESSION_WAKE_RETRY_WINDOW
    assert len(spacings) > 20


def test_an_attempt_whose_worker_lost_its_lease_is_replaced() -> None:
    from cayu.tasks.session_wakes import session_wake_attempts

    async def scenario() -> None:
        clock = _Clock()
        sessions, tasks = (
            InMemorySessionStore(),
            InMemoryTaskStore(clock=clock, ownership_clock=clock),
        )
        calls: list[ModelRequest] = []
        app = _app(sessions, tasks, clock, calls=calls)
        pending = await _sleeping(app, sessions, "s_lost")
        clock.advance(30)
        claimed = await tasks.claim_task("crashed-worker", lease_seconds=10)
        assert claimed is not None and claimed.id == pending["wake"]["task_id"]
        assert await reconcile_session_wakes(app) == 0  # Its lease is still live.
        clock.advance(11)
        assert await reconcile_session_wakes(app) == 1
        assert len(await session_wake_attempts(app, pending["input_id"])) == 2
        clock.advance(60)
        # Whichever attempt runs first continues the session; the other replays.
        assert await _work(app, max_tasks=2) == 2
        assert (await sessions.load("s_lost")).status is SessionStatus.COMPLETED
        assert len(calls) == 2
        results = {
            task.result["outcome"]
            for task in await session_wake_attempts(app, pending["input_id"])
            if task.status is TaskStatus.COMPLETED
        }
        assert results == {"continued", "already_delivered"}

    asyncio.run(scenario())


def test_one_damaged_sleeping_session_does_not_stop_reconciliation_or_a_worker(
    monkeypatch,
) -> None:

    async def scenario() -> None:
        clock = _Clock()
        sessions, tasks = InMemorySessionStore(), InMemoryTaskStore(clock=clock)
        calls: list[ModelRequest] = []
        app = _app(sessions, tasks, clock, calls=calls)
        damaged = await _sleeping(app, sessions, "s_damaged")
        healthy = await _sleeping(app, sessions, "s_healthy")
        for task_id in (damaged["wake"]["task_id"], healthy["wake"]["task_id"]):
            task = await tasks.load_task(task_id)
            await app.cancel_scheduled_task(
                TaskScheduleCancelRequest(
                    task_id=task_id,
                    operation_id=f"stray-{task_id}",
                    expected_revision=task.schedule.revision,
                )
            )
        original = session_wakes._ensure_wake_attempt

        async def conflicting(app, pending):
            if pending.session_id == "s_damaged":
                raise ValueError("Task creation content conflicts.")
            return await original(app, pending)

        monkeypatch.setattr(session_wakes, "_ensure_wake_attempt", conflicting)
        assert await reconcile_session_wakes(app) == 1
        clock.advance(120)
        assert await _work(app) == 1
        assert (await sessions.load("s_healthy")).status is SessionStatus.COMPLETED
        assert (await sessions.load("s_damaged")).status is SessionStatus.INTERRUPTED

    asyncio.run(scenario())


def test_a_wake_is_never_reported_as_a_person_answering() -> None:
    from cayu.cloud_reporting import read_cloud_activity
    from cayu.sessions.supervision_events import SupervisionEventQuery

    async def scenario() -> None:
        clock = _Clock(datetime.now(UTC))
        sessions, tasks = InMemorySessionStore(), InMemoryTaskStore(clock=clock)
        app = _app(sessions, tasks, clock)
        start = clock.now - timedelta(minutes=1)
        await _sleeping(app, sessions, "s_touch")
        clock.advance(30)
        assert await _work(app) == 1
        resumed = [
            record.event
            for record in await sessions.query_events(EventQuery(session_id="s_touch", limit=500))
            if record.event.type is EventType.SESSION_RESUMED
        ]
        assert [event.payload["interruption_type"] for event in resumed] == ["wake_scheduled"]
        page = await app.query_supervision_events(
            SupervisionEventQuery(start_at=start, end_at=datetime.now(UTC) + timedelta(hours=1))
        )
        assert [event for event in page.events if event.session_id == "s_touch"] == []
        activity = await read_cloud_activity(app)
        assert [touch for touch in activity.touches if touch.session_id == "s_touch"] == []

    asyncio.run(scenario())


def test_interrupting_a_handed_over_wake_cancels_its_successor() -> None:
    from cayu.tasks.session_wakes import session_wake_attempts

    async def scenario() -> None:
        clock = _Clock()
        sessions, tasks = InMemorySessionStore(), InMemoryTaskStore(clock=clock)
        pending = await _sleeping(_app(sessions, tasks, clock), sessions, "s_handed")
        clock.advance(30)
        broken = CayuApp(
            session_store=sessions, task_store=tasks, enable_logging=False, clock=clock
        )
        broken.register_agent(
            AgentSpec(name="sleeper", model="scripted-model"), tools=[SleepUntilTool()]
        )
        assert await _work(broken) == 1
        checkpoint = await sessions.load_checkpoint("s_handed")
        assert "pending_wake" not in checkpoint
        assert checkpoint["pending_user_input"]["wake"] is not None
        # Handed over, it is still a wake, never a question for a person.
        questions = await sessions.query_pending_actions(
            PendingActionQuery(kind=PendingActionKind.USER_INPUT)
        )
        assert questions.actions == []
        [action] = (
            await sessions.query_pending_actions(PendingActionQuery(session_id="s_handed"))
        ).actions
        assert action.kind is PendingActionKind.WAKE
        with pytest.raises(ValueError, match="only its wake task"):
            async for _ in broken.resolve_user_input(
                UserInputResponse(session_id="s_handed", input_id=pending["input_id"], answer="x")
            ):
                pass
        app = _app(sessions, tasks, clock)
        async for _ in app.interrupt_session(InterruptSessionRequest(session_id="s_handed")):
            pass
        statuses = [t.status for t in await session_wake_attempts(app, pending["input_id"])]
        assert statuses == [TaskStatus.FAILED, TaskStatus.CANCELLED]
        assert await reconcile_session_wakes(app) == 0
        clock.advance(3600)
        outcome = await deliver_session_wake(
            app, session_id="s_handed", wake_id=pending["input_id"]
        )
        assert outcome is SessionWakeOutcome.CLOSED
        assert "pending_user_input" not in await sessions.load_checkpoint("s_handed")

    asyncio.run(scenario())


def test_an_invalid_interruption_leaves_the_sleep_alone() -> None:
    from cayu.vaults.redaction import SecretRedactor

    async def scenario() -> None:
        clock = _Clock()
        sessions, tasks = InMemorySessionStore(), InMemoryTaskStore(clock=clock)
        pending = await _sleeping(_app(sessions, tasks, clock), sessions, "s_valid")
        # This application treats the session's agent name as a workload secret,
        # so it refuses the interruption before touching the session.
        guarded = CayuApp(
            session_store=sessions,
            task_store=tasks,
            enable_logging=False,
            clock=clock,
            secret_redactor=SecretRedactor("sleeper"),
        )
        with pytest.raises(ValueError):
            async for _ in guarded.interrupt_session(InterruptSessionRequest(session_id="s_valid")):
                pass
        assert "pending_wake" in await sessions.load_checkpoint("s_valid")
        task = await tasks.load_task(pending["wake"]["task_id"])
        assert task.status is TaskStatus.PENDING

    asyncio.run(scenario())


def test_an_older_writer_cannot_represent_a_handed_over_wake() -> None:
    from cayu.sessions.checkpoints import runtime_checkpoint_writer_view

    async def scenario() -> None:
        clock = _Clock()
        sessions, tasks = InMemorySessionStore(), InMemoryTaskStore(clock=clock)
        await _sleeping(_app(sessions, tasks, clock), sessions, "s_old_handed")
        clock.advance(30)
        broken = CayuApp(
            session_store=sessions, task_store=tasks, enable_logging=False, clock=clock
        )
        broken.register_agent(
            AgentSpec(name="sleeper", model="scripted-model"), tools=[SleepUntilTool()]
        )
        assert await _work(broken) == 1
        checkpoint = await sessions.load_checkpoint("s_old_handed")
        assert "pending_user_input" in checkpoint
        with pytest.raises(ValueError, match="sleeping tool call"):
            runtime_checkpoint_writer_view(checkpoint, writer_version=11, session_id="s_old_handed")

    asyncio.run(scenario())


def test_a_postgres_session_sleeps_and_wakes_end_to_end(request) -> None:
    async def scenario() -> None:
        clock = _Clock(datetime.now(UTC))
        sessions, tasks = _stores("postgres", None, request, clock)
        calls: list[ModelRequest] = []
        app = _app(sessions, tasks, clock, calls=calls, arguments={"duration_seconds": 1})
        try:
            pending = await _sleeping(app, sessions, "s_postgres")
            await asyncio.sleep(1.1)
            clock.advance(1.1)
            assert await _work(app) == 1
            await _assert_woke_once(
                sessions,
                "s_postgres",
                deadline=datetime.fromisoformat(pending["wake"]["deadline"]),
            )
            assert len(calls) == 2
        finally:
            await app.aclose()

    asyncio.run(scenario())


def test_a_running_worker_restores_a_missing_wake_on_its_reclaim_cadence(monkeypatch) -> None:

    async def scenario() -> None:
        sessions, tasks = InMemorySessionStore(), InMemoryTaskStore()
        calls: list[ModelRequest] = []
        app = CayuApp(session_store=sessions, task_store=tasks, enable_logging=False)
        app.register_provider(
            _StableScriptedProvider(
                response_factory=_sleep_then_answer({"duration_seconds": 0.2}, calls=calls)
            ),
            default=True,
        )
        app.register_agent(
            AgentSpec(name="sleeper", model="scripted-model"), tools=[SleepUntilTool()]
        )
        stop = asyncio.Event()
        worker = asyncio.create_task(
            run_task_worker(
                app,
                tasks,
                _no_application_tasks,
                worker_id="cadence-worker",
                stop=stop,
                poll_interval_s=0.05,
                reclaim_every_s=0.2,
            )
        )
        await asyncio.sleep(0.1)  # The worker's start-up reconcile has run.

        lost_publications: list[str] = []

        async def lost(app, pending):
            lost_publications.append("lost")
            raise ConnectionError("task store unavailable")

        with monkeypatch.context() as patched:
            patched.setattr(session_wakes, "_publish_session_wake", lost)
            await _run(app, "s_cadence")
        # The pause's own publication was lost; only the worker's cadence can restore it,
        # possibly while the run is still returning, so don't assert on the gap itself.
        assert len(lost_publications) == 1
        try:
            async with asyncio.timeout(10):
                while (await sessions.load("s_cadence")).status is not SessionStatus.COMPLETED:
                    await asyncio.sleep(0.05)
        finally:
            stop.set()
            await worker
        assert len(calls) == 2

    asyncio.run(scenario())


async def _hand_over(sessions, tasks, clock, session_id: str) -> dict:
    """Leave a woken wake handed over: its first attempt fails before it continues."""

    pending = await _sleeping(
        _app(sessions, tasks, clock, arguments={"duration_seconds": 1}), sessions, session_id
    )
    await asyncio.sleep(1.1)  # Durable stores keep their own clock.
    clock.advance(1.1)
    broken = CayuApp(session_store=sessions, task_store=tasks, enable_logging=False, clock=clock)
    broken.register_agent(
        AgentSpec(name="sleeper", model="scripted-model"), tools=[SleepUntilTool()]
    )
    # Deliver directly: a shared durable store may hold other sessions' due wakes.
    with pytest.raises(KeyError):
        await deliver_session_wake(broken, session_id=session_id, wake_id=pending["input_id"])
    checkpoint = await sessions.load_checkpoint(session_id)
    assert "pending_wake" not in checkpoint and checkpoint["pending_user_input"]["wake"]
    return pending


async def _ask(sessions, tasks, clock, session_id: str) -> None:
    def respond(request: ModelRequest):
        return [
            ModelStreamEvent.tool_call(id="call_ask", name="ask_user", arguments={"question": "?"}),
            ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
        ]

    app = CayuApp(session_store=sessions, task_store=tasks, enable_logging=False, clock=clock)
    app.register_provider(_StableScriptedProvider(response_factory=respond), default=True)
    app.register_agent(AgentSpec(name="asker", model="scripted-model"), tools=[UserInputTool()])
    async for _ in app.run(
        RunRequest(agent_name="asker", session_id=session_id, messages=[Message.text("user", "q")])
    ):
        pass


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
def test_a_wake_sets_only_its_own_pending_action_bit(backend, tmp_path, request) -> None:
    from cayu.sessions.pending_actions import pending_action_checkpoint_metrics

    async def scenario() -> None:
        clock = _Clock(datetime.now(UTC))
        sessions, tasks = _stores(backend, tmp_path, request, clock)
        try:
            await _sleeping(_app(sessions, tasks, clock), sessions, "s_bits_asleep")
            await _hand_over(sessions, tasks, clock, "s_bits_handed")
            await _ask(sessions, tasks, clock, "s_bits_question")
            for session_id in ("s_bits_asleep", "s_bits_handed"):
                checkpoint = await sessions.load_checkpoint(session_id)
                assert pending_action_checkpoint_metrics(checkpoint)[2] == 16
                execution = await sessions.inspect_session_execution(session_id)
                assert execution.state == "terminal"  # Not waiting on a person.
            # Each query filters on its own bit: SQL stores never even inspect the
            # other kind (memory classifies every candidate in Python).
            for session_id, kind, listed in (
                ("s_bits_question", PendingActionKind.USER_INPUT, True),
                ("s_bits_question", PendingActionKind.WAKE, False),
                ("s_bits_asleep", PendingActionKind.WAKE, True),
                ("s_bits_asleep", PendingActionKind.USER_INPUT, False),
                ("s_bits_handed", PendingActionKind.WAKE, True),
                ("s_bits_handed", PendingActionKind.USER_INPUT, False),
            ):
                page = await sessions.query_pending_actions(
                    PendingActionQuery(session_id=session_id, kind=kind)
                )
                assert len(page.actions) == int(listed)
                if listed and kind is PendingActionKind.WAKE:
                    assert page.actions[0].wake_id
                if backend != "memory":
                    assert page.inspected_candidate_count == int(listed)
        finally:
            if hasattr(sessions, "close"):
                await sessions.close()

    asyncio.run(scenario())


def test_the_retry_window_starts_when_a_late_worker_first_runs_the_wake() -> None:
    from types import SimpleNamespace

    from cayu.tasks.session_wakes import SESSION_WAKE_RETRY_WINDOW, _attempt_task, _successor

    first = _attempt_task("s", "w", _START, 1, _START)
    late = _START + timedelta(hours=30)  # Every worker was down past the window.
    attempt = SimpleNamespace(
        id=first.task_id, input=first.input, available_at=_START, started_at=late, created_at=_START
    )
    successor = _successor(attempt)
    assert successor is not None
    assert successor.available_at == late + timedelta(seconds=30)
    assert successor.input["retry_until"] == (late + SESSION_WAKE_RETRY_WINDOW).isoformat()
    # Later attempts keep the window the first one fixed.
    second = SimpleNamespace(
        id=successor.task_id,
        input=successor.input,
        available_at=successor.available_at,
        started_at=None,
        created_at=_START,
    )
    assert _successor(second).input["retry_until"] == successor.input["retry_until"]


def test_spent_wakes_are_named_once_per_reconciliation_pass(monkeypatch, caplog) -> None:
    import logging

    monkeypatch.setattr(session_wakes, "SESSION_WAKE_RETRY_WINDOW", timedelta(seconds=1))

    async def scenario() -> None:
        clock = _Clock()
        sessions, tasks = InMemorySessionStore(), InMemoryTaskStore(clock=clock)
        app = _app(sessions, tasks, clock)
        ids = []
        for session_id in ("s_spent_1", "s_spent_2"):
            pending = await _sleeping(app, sessions, session_id)
            ids.append(pending["input_id"])
            task = await tasks.load_task(pending["wake"]["task_id"])
            await app.cancel_scheduled_task(
                TaskScheduleCancelRequest(
                    task_id=task.id,
                    operation_id=f"stray-{task.id}",
                    expected_revision=task.schedule.revision,
                )
            )
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger=session_wakes.__name__):
            assert await reconcile_session_wakes(app) == 0
        [record] = [r for r in caplog.records if "retry window" in r.getMessage()]
        assert all(wake_id in record.getMessage() for wake_id in ids)

    asyncio.run(scenario())


@pytest.mark.parametrize("recovered", [False, True])
def test_a_sleep_keeps_a_vault_backed_siblings_arguments_private(recovered) -> None:
    from tests.core._execution_profile_fixtures import rebind_test_invocation
    from tests.core._workload_secret_support import FakeProvider
    from tests.core.test_tool_start_argument_quarantine import (
        _portable_environment_spec,
        _ResolveAfterStartTool,
    )

    from cayu.environments.base import Environment
    from cayu.observability.events import InMemoryEventSink
    from cayu.sessions.recovery import IncompleteSessionRecoveryRequest
    from cayu.vaults.static import StaticVault

    secret = "late-sleep-sibling-secret-canary"
    sibling_arguments = {"provided": secret, "nested": {"value": secret}}

    async def scenario() -> None:
        clock = _Clock()
        sessions, tasks = InMemorySessionStore(), InMemoryTaskStore(clock=clock)
        sink = InMemoryEventSink()
        provider = FakeProvider(
            [
                [
                    ModelStreamEvent.tool_call(
                        id="call_sleep", name="sleep_until", arguments={"duration_seconds": 30}
                    ),
                    ModelStreamEvent.tool_call(
                        id="call_after_wake",
                        name="resolve_after_start",
                        arguments=sibling_arguments,
                    ),
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ],
                [ModelStreamEvent.text_delta("done"), ModelStreamEvent.completed()],
            ]
        )
        app = CayuApp(
            session_store=sessions,
            task_store=tasks,
            event_sinks=[sink],
            enable_logging=False,
            clock=clock,
        )
        app.register_provider(provider, default=True)
        app.register_environment(
            Environment(
                _portable_environment_spec("local"), vault=StaticVault({"api_key": secret})
            ),
            default=True,
        )
        sibling = _ResolveAfterStartTool()
        app.register_agent(
            AgentSpec(name="sleeper", model="fake-model"), tools=[SleepUntilTool(), sibling]
        )
        paused = await _run(app, "s_private")
        interrupted = [e for e in paused if e.type is EventType.SESSION_INTERRUPTED]
        if recovered:
            await rebind_test_invocation(sessions, "s_private")
            result = await app.recover_incomplete_session(
                IncompleteSessionRecoveryRequest(session_id="s_private")
            )
            interrupted = [e for e in result.events if e.type is EventType.SESSION_INTERRUPTED]
        [stop] = interrupted
        assert stop.payload["interruption_type"] == "wake_scheduled"
        continuation = stop.payload["continuation"]
        assert "arguments" not in continuation and continuation["arguments_state"] == "quarantined"
        assert "quarantined_assistant_message" not in continuation
        assert all("arguments" not in call for call in continuation["tool_calls"])
        assert secret not in repr(paused)
        assert secret not in repr(sink.events)
        assert secret not in repr(await sessions.load_events("s_private"))
        assert sibling.arguments == []

        clock.advance(30)
        assert await _work(app) == 1
        assert sibling.arguments == [sibling_arguments]

    asyncio.run(scenario())


def test_a_woken_continuation_that_cannot_reconnect_is_retried_not_counted() -> None:
    from cayu.environments.base import Environment, EnvironmentSpec
    from cayu.environments.bindings import BoundWorkspace, WorkspaceBinding
    from cayu.tasks.session_wakes import session_wake_attempts

    class FlakyBinding(WorkspaceBinding):
        def __init__(self) -> None:
            self.binds = 0

        async def bind(self, workspace, runner, **kwargs):
            del kwargs
            self.binds += 1
            if self.binds == 2:  # The woken continuation's reconnect.
                raise RuntimeError("environment could not reconnect")
            return BoundWorkspace(workspace=workspace, source_workspace=workspace, runner=runner)

        async def finalize(self, bound, **kwargs):
            del bound, kwargs
            return None

    async def scenario() -> None:
        clock = _Clock()
        sessions, tasks = InMemorySessionStore(), InMemoryTaskStore(clock=clock)
        calls: list[ModelRequest] = []
        app = _app(sessions, tasks, clock, calls=calls)
        app.register_environment(
            Environment(EnvironmentSpec(name="flaky"), binding=FlakyBinding()), default=True
        )
        pending = await _sleeping(app, sessions, "s_flaky")
        clock.advance(30)
        assert await _work(app) == 1
        first, successor = await session_wake_attempts(app, pending["input_id"])
        assert first.status is TaskStatus.FAILED
        assert first.error is not None and "SessionWakeNotContinued" in str(first.error)
        assert successor.status is TaskStatus.PENDING
        assert (await sessions.load("s_flaky")).status is SessionStatus.INTERRUPTED
        [action] = (
            await sessions.query_pending_actions(PendingActionQuery(session_id="s_flaky"))
        ).actions
        assert action.kind is PendingActionKind.WAKE and action.wake_id == pending["input_id"]
        assert len(calls) == 1
        clock.advance(60)
        assert await _work(app) == 1
        await _assert_woke_once(sessions, "s_flaky", deadline=_START + timedelta(seconds=30))
        assert len(calls) == 2

    asyncio.run(scenario())


def test_an_interrupt_while_the_sleep_is_still_opening_supersedes_it(monkeypatch) -> None:

    async def scenario() -> None:
        clock = _Clock()
        sessions, tasks = InMemorySessionStore(), InMemoryTaskStore(clock=clock)
        calls: list[ModelRequest] = []
        app = _app(sessions, tasks, clock, calls=calls)
        publish = session_wakes._publish_session_wake
        stops: list[asyncio.Task] = []

        async def stop_then_publish(app, pending):
            # The wake checkpoint is committed and the session is still running.
            assert (await sessions.load("s_opening")).status is SessionStatus.RUNNING
            assert "pending_wake" in await sessions.load_checkpoint("s_opening")

            async def interrupt() -> None:
                async for _ in app.interrupt_session(
                    InterruptSessionRequest(session_id="s_opening")
                ):
                    pass

            stops.append(asyncio.create_task(interrupt()))
            await asyncio.sleep(0.05)
            await publish(app, pending)

        monkeypatch.setattr(session_wakes, "_publish_session_wake", stop_then_publish)
        await _run(app, "s_opening")
        await asyncio.gather(*stops)
        session = await sessions.load("s_opening")
        assert session.status is SessionStatus.INTERRUPTED
        checkpoint = await sessions.load_checkpoint("s_opening")
        assert "pending_wake" not in checkpoint and "pending_user_input" not in checkpoint
        page = await sessions.query_pending_actions(PendingActionQuery(session_id="s_opening"))
        assert all(action.kind is not PendingActionKind.WAKE for action in page.actions)
        [opened] = [
            record.event
            for record in await sessions.query_events(EventQuery(session_id="s_opening", limit=500))
            if record.event.type is EventType.SESSION_CHECKPOINTED
            and record.event.payload.get("checkpoint") == "pending_wake"
        ]
        wake_id = opened.payload["wake_id"]
        clock.advance(30)
        for task in await tasks.list_tasks():
            assert task.status is not TaskStatus.PENDING or task.type != SESSION_WAKE_TASK_TYPE
        outcome = await deliver_session_wake(app, session_id="s_opening", wake_id=wake_id)
        assert outcome is SessionWakeOutcome.CLOSED
        assert len(calls) == 1

    asyncio.run(scenario())


def test_a_stop_that_crashes_before_it_commits_leaves_the_sleep_whole(monkeypatch) -> None:

    cancel = session_wakes._cancel_session_wake

    async def cancel_then_crash(app, session_id):
        await cancel(app, session_id)
        raise ConnectionError("process lost before the interruption committed")

    monkeypatch.setattr(session_wakes, "_cancel_session_wake", cancel_then_crash)

    async def scenario() -> None:
        clock = _Clock()
        sessions, tasks = InMemorySessionStore(), InMemoryTaskStore(clock=clock)
        calls: list[ModelRequest] = []
        app = _app(sessions, tasks, clock, calls=calls)
        pending = await _sleeping(app, sessions, "s_half_stop")
        with pytest.raises(ConnectionError):
            async for _ in app.interrupt_session(InterruptSessionRequest(session_id="s_half_stop")):
                pass
        # The stop never committed: the session still sleeps, so its wake comes back.
        assert "pending_wake" in await sessions.load_checkpoint("s_half_stop")
        assert await reconcile_session_wakes(app) == 1
        clock.advance(60)
        assert await _work(app) == 1
        await _assert_woke_once(sessions, "s_half_stop", deadline=_START + timedelta(seconds=30))
        assert len(calls) == 2
        del pending

    asyncio.run(scenario())


def test_a_stop_that_crashes_while_a_sleep_opens_recovers_as_interrupted(
    tmp_path, monkeypatch
) -> None:
    import sqlite3

    from cayu.runtime._interruption_coordinator import _PENDING_SESSION_INTERRUPT_CHECKPOINT_KEY
    from cayu.runtime._session_finalization import _checkpoint_with_pending_session_interrupt
    from cayu.sessions._invocation_terminal_decision import (
        invocation_terminal_decision_from_checkpoint,
    )
    from cayu.sessions.authority import SessionRunFenced
    from cayu.sessions.recovery import IncompleteSessionRecoveryRequest

    def snapshot(name: str) -> None:
        source = sqlite3.connect(tmp_path / f"{name}.db")
        target = sqlite3.connect(tmp_path / f"crashed-{name}.db")
        source.backup(target)
        source.close()
        target.close()

    async def scenario() -> None:
        clock = _Clock(datetime.now(UTC))
        sessions = SQLiteSessionStore(tmp_path / "sessions.db")
        tasks = SQLiteTaskStore(tmp_path / "tasks.db")
        app = _app(sessions, tasks, clock)
        control = app._session_control
        observe = control.wait_for_active_interrupted_event
        crashed: list[bool] = []

        async def crash_point(session_id, **options):
            # The stop transition is durable and terminal publication has not
            # happened: copy the databases as a process that died here left them.
            if not crashed:
                crashed.append(True)
                snapshot("sessions")
                snapshot("tasks")
            return await observe(session_id, **options)

        monkeypatch.setattr(control, "wait_for_active_interrupted_event", crash_point)
        publish = session_wakes._publish_session_wake
        stops: list[asyncio.Task] = []

        opening: list = []

        async def stop_then_publish(app, pending):
            opening.append(
                (await sessions.load("s_crash"), await sessions.load_checkpoint("s_crash"))
            )

            async def interrupt() -> None:
                async for _ in app.interrupt_session(InterruptSessionRequest(session_id="s_crash")):
                    pass

            stops.append(asyncio.create_task(interrupt()))
            await asyncio.sleep(0.05)
            await publish(app, pending)

        monkeypatch.setattr(session_wakes, "_publish_session_wake", stop_then_publish)
        await _run(app, "s_crash")
        await asyncio.gather(*stops)
        await app.aclose()
        assert crashed

        # The restarted process runs after the dead one's leases expired.
        restarted_sessions = SQLiteSessionStore(
            tmp_path / "crashed-sessions.db",
            ownership_clock=lambda: datetime.now(UTC) + timedelta(hours=1),
        )
        restarted = _app(restarted_sessions, SQLiteTaskStore(tmp_path / "crashed-tasks.db"), clock)
        try:
            crashed_session = await restarted_sessions.load("s_crash")
            assert crashed_session.status is SessionStatus.INTERRUPTING
            checkpoint = await restarted_sessions.load_checkpoint("s_crash")
            decision = invocation_terminal_decision_from_checkpoint(checkpoint)
            # Both durable records bind the same stop, sleeping call included.
            assert (
                checkpoint[_PENDING_SESSION_INTERRUPT_CHECKPOINT_KEY] == decision.terminal_payload
            )
            assert "pending_wake" not in checkpoint
            # A stop decided before the sleep opened must decide again, not persist
            # a supersession its terminal decision does not bind.
            [(running, running_checkpoint)] = opening
            assert running.status is SessionStatus.RUNNING and "pending_wake" in running_checkpoint
            stale_payload = {
                key: value
                for key, value in decision.terminal_payload.items()
                if key != "user_input_supersession_intent"
            }
            stale = decision.model_copy(update={"terminal_payload": stale_payload})
            with pytest.raises(SessionRunFenced, match="opened after the stop"):
                _checkpoint_with_pending_session_interrupt(stale_payload, terminal_decision=stale)(
                    running, running_checkpoint
                )
            result = await restarted.recover_incomplete_session(
                IncompleteSessionRecoveryRequest(session_id="s_crash")
            )
            assert (await restarted_sessions.load("s_crash")).status is SessionStatus.INTERRUPTED, (
                result
            )
        finally:
            await restarted.aclose()

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
def test_a_wake_whose_continuation_stopped_still_says_when_it_is_due(
    backend, tmp_path, request
) -> None:
    from cayu.cloud_reporting import read_cloud_activity
    from cayu.environments.base import Environment, EnvironmentSpec
    from cayu.environments.bindings import BoundWorkspace, WorkspaceBinding
    from cayu.tasks import SessionWakeNotContinued

    class FlakyBinding(WorkspaceBinding):
        def __init__(self) -> None:
            self.binds = 0

        async def bind(self, workspace, runner, **kwargs):
            del kwargs
            self.binds += 1
            if self.binds == 2:  # The woken continuation's reconnect.
                raise RuntimeError("environment could not reconnect")
            return BoundWorkspace(workspace=workspace, source_workspace=workspace, runner=runner)

        async def finalize(self, bound, **kwargs):
            del bound, kwargs
            return None

    async def scenario() -> None:
        clock = _Clock(datetime.now(UTC))
        sessions, tasks = _stores(backend, tmp_path, request, clock)
        app = _app(sessions, tasks, clock, arguments={"duration_seconds": 1})
        app.register_environment(
            Environment(EnvironmentSpec(name="flaky"), binding=FlakyBinding()), default=True
        )
        session_id = f"s_due_{backend}"
        try:
            pending = await _sleeping(app, sessions, session_id)
            due = datetime.fromisoformat(pending["wake"]["deadline"])
            clock.advance(1.1)
            with pytest.raises(SessionWakeNotContinued):
                await deliver_session_wake(app, session_id=session_id, wake_id=pending["input_id"])
            checkpoint = await sessions.load_checkpoint(session_id)
            assert checkpoint["pending_user_input"]["wake"]  # Handed over, not continued.
            [action] = (
                await sessions.query_pending_actions(PendingActionQuery(session_id=session_id))
            ).actions
            assert action.kind is PendingActionKind.WAKE
            assert action.wake_id == pending["input_id"]
            assert action.wake_at == due
            assert action.detail == f"Continues on its own at {due.isoformat()}."
            assert action.tool_name == "sleep_until"
            assert action.arguments is None
            # The indexed source keeps only when it is due, not the call's request.
            from cayu.sessions.pending_actions import _pending_action_event_payload_view

            [stop] = [
                event
                for event in await sessions.load_events(session_id)
                if event.type is EventType.SESSION_INTERRUPTED
                and event.payload.get("interruption_type") == "user_input_required"
            ]
            indexed = _pending_action_event_payload_view(stop)["user_input"]
            assert indexed["wake"] == {"deadline": stop.payload["user_input"]["wake"]["deadline"]}
            activity = await read_cloud_activity(app)
            [run] = [run for run in activity.runs if run.session_id == session_id]
            [waiting] = run.waiting_for
            assert waiting.kind == "wake" and waiting.wake_at == due
        finally:
            await app.aclose()

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
def test_a_subagent_that_asks_after_the_wake_continues_the_parent_once(
    backend, tmp_path, request
) -> None:
    from tests.core.test_foreground_subagent_recovery import _identity, _Provider

    from cayu.tools.subagents import SubagentSpec, SubagentTool

    async def scenario() -> None:
        clock = _Clock(datetime.now(UTC))
        sessions, tasks = _stores(backend, tmp_path, request, clock)
        session_id = f"s_child_{backend}"
        provider = _Provider(
            [
                [
                    ModelStreamEvent.tool_call(
                        id="call_sleep", name="sleep_until", arguments={"duration_seconds": 1}
                    ),
                    ModelStreamEvent.tool_call(
                        id="call_spawn", name="subagent", arguments={"agent": "child", "task": "t"}
                    ),
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ],
                [
                    ModelStreamEvent.tool_call(
                        id="call_ask", name="ask_user", arguments={"question": "Which?"}
                    ),
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ],
                [ModelStreamEvent.text_delta("child finished"), ModelStreamEvent.completed()],
                [ModelStreamEvent.text_delta("parent finished"), ModelStreamEvent.completed()],
            ]
        )
        app = CayuApp(session_store=sessions, task_store=tasks, enable_logging=False, clock=clock)
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="sleeper", model="test"),
            tools=[
                SleepUntilTool(),
                SubagentTool(
                    app,
                    agents={"child": SubagentSpec(agent_name="child")},
                    execution_profile_identity=_identity("sleeping-parent-subagent"),
                ),
            ],
        )
        app.register_agent(AgentSpec(name="child", model="test"), tools=[UserInputTool()])
        try:
            pending = await _sleeping(app, sessions, session_id)
            await asyncio.sleep(1.1)  # Durable stores keep their own clock.
            clock.advance(1.1)
            assert (
                await deliver_session_wake(app, session_id=session_id, wake_id=pending["input_id"])
                is SessionWakeOutcome.CONTINUED
            )
            assert len(provider.requests) == 2
            assert not await session_wakes._delivered(app, session_id, "another-wake")
            [child] = (
                await sessions.list_sessions(SessionQuery(parent_session_id=session_id))
            ).sessions
            asked = next(
                event
                for event in await sessions.load_events(child.id)
                if event.type is EventType.SESSION_AWAITING_USER_INPUT
            )
            async for _ in app.resolve_user_input(
                UserInputResponse(
                    session_id=app.project_session_id_for_exposure(child.id),
                    input_id=asked.payload["input_id"],
                    answer="this one",
                )
            ):
                pass
            for _ in range(50):
                if (await sessions.load(session_id)).status is SessionStatus.COMPLETED:
                    break
                await asyncio.sleep(0.05)
            parent = await sessions.load(session_id)
            assert parent.status is SessionStatus.COMPLETED, [
                (event.type, event.payload.get("error"))
                for event in await sessions.load_events(session_id)
            ][-4:]
            assert len(provider.requests) == 4
            completed = [
                event
                for event in await sessions.load_events(session_id)
                if event.type is EventType.SESSION_COMPLETED
            ]
            assert len(completed) == 1
            # The wake is spent: a late attempt changes nothing.
            assert await deliver_session_wake(
                app, session_id=session_id, wake_id=pending["input_id"]
            ) in {SessionWakeOutcome.ALREADY_DELIVERED, SessionWakeOutcome.CLOSED}
            assert len(provider.requests) == 4
        finally:
            await app.aclose()

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
def test_a_woken_sibling_left_without_a_result_needs_manual_recovery_not_retries(
    backend, tmp_path, request
) -> None:
    from tests.core._execution_profile_fixtures import rebind_test_invocation

    from cayu.approvals.actors import ResolutionActor, ResolutionActorSource
    from cayu.approvals.tools import ToolApprovalRecoveryOutcome
    from cayu.approvals.user_input import UserInputRecoveryRequest, session_wake_tool_result
    from cayu.cloud_reporting import read_cloud_activity
    from cayu.sessions.recovery import IncompleteSessionRecoveryRequest
    from cayu.tasks.session_wakes import SESSION_WAKE_ACTOR_SUBJECT, session_wake_attempts
    from cayu.tools.base import Tool, ToolEffect, ToolSpec

    class Charge(Tool):
        spec = ToolSpec(
            name="charge",
            description="Charge the card once.",
            effect=ToolEffect.IDEMPOTENT,
            input_schema={"type": "object"},
            max_terminal_payload_bytes=4096,
            execution_profile_identity=ExecutionProfileBehaviorIdentity(
                name="tests:sleep-until:charge", behavior_version="1", implementation_version="1"
            ),
        )

        def __init__(self) -> None:
            self.runs = 0

        async def run(self, arguments, ctx):
            del arguments, ctx
            self.runs += 1
            # The process dies while the charge runs: it started, with no result.
            raise SystemExit("worker died mid-charge")

    def respond(request: ModelRequest):
        if not _tool_results(request):
            return [
                ModelStreamEvent.tool_call(
                    id="call_sleep", name="sleep_until", arguments={"duration_seconds": 1}
                ),
                ModelStreamEvent.tool_call(id="call_charge", name="charge", arguments={}),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ]
        return [ModelStreamEvent.text_delta("done"), ModelStreamEvent.completed()]

    async def scenario() -> None:
        clock = _Clock(datetime.now(UTC))
        sessions, tasks = _stores(backend, tmp_path, request, clock)
        session_id = f"s_stuck_{backend}"
        charge = Charge()
        app = CayuApp(session_store=sessions, task_store=tasks, enable_logging=False, clock=clock)
        app.register_provider(_StableScriptedProvider(response_factory=respond), default=True)
        app.register_agent(
            AgentSpec(name="sleeper", model="scripted-model"), tools=[SleepUntilTool(), charge]
        )
        try:
            pending = await _sleeping(app, sessions, session_id)
            wake_id = pending["input_id"]
            await asyncio.sleep(1.1)  # Durable stores keep their own clock.
            clock.advance(1.1)
            with pytest.raises(SystemExit):
                await deliver_session_wake(app, session_id=session_id, wake_id=wake_id)
            await rebind_test_invocation(sessions, session_id)
            recovered = await app.recover_incomplete_session(
                IncompleteSessionRecoveryRequest(session_id=session_id)
            )
            assert (
                recovered.message == "Session is waiting for its wake; its wake task continues it."
            )

            # The first attempt runs again and finds the charge without a result. A
            # shared durable store may hold other sessions' due wakes first.
            for _ in range(20):
                [attempt] = await session_wake_attempts(app, wake_id)
                if attempt.status is not TaskStatus.PENDING:
                    break
                assert await _work(app) == 1
            [attempt] = await session_wake_attempts(app, wake_id)
            assert attempt.status is TaskStatus.COMPLETED
            assert attempt.result == {"outcome": "manual_recovery_required"}
            assert charge.runs == 1
            [action] = (
                await sessions.query_pending_actions(PendingActionQuery(session_id=session_id))
            ).actions
            assert action.kind is PendingActionKind.MANUAL_RECOVERY
            assert action.tool_call_id == "call_charge"
            assert action.input_id == wake_id
            activity = await read_cloud_activity(app)
            [run] = [run for run in activity.runs if run.session_id == session_id]
            assert [item.kind for item in run.waiting_for] == ["manual_recovery"]
            # Nothing keeps retrying it, and a late delivery does not run the round again.
            assert await reconcile_session_wakes(app) == 0
            settled_events = len(await sessions.load_events(session_id))
            assert (
                await deliver_session_wake(app, session_id=session_id, wake_id=wake_id)
                is SessionWakeOutcome.MANUAL_RECOVERY_REQUIRED
            )
            assert len(await sessions.load_events(session_id)) == settled_events
            assert len(await session_wake_attempts(app, wake_id)) == 1
            assert not await session_wakes._needs_manual_recovery(app, session_id, "another-wake")

            # Recovery re-supplies the accepted answer: the wake's own timer result.
            checkpoint = await sessions.load_checkpoint(session_id)
            timer = session_wake_tool_result(
                PendingUserInput.model_validate(checkpoint["pending_user_input"])
            )
            async for _ in app.recover_user_input(
                UserInputRecoveryRequest(
                    session_id=session_id,
                    input_id=wake_id,
                    answer=timer,
                    resolved_by=ResolutionActor(
                        subject=SESSION_WAKE_ACTOR_SUBJECT, source=ResolutionActorSource.SYSTEM
                    ),
                    tool_call_id="call_charge",
                    outcome=ToolApprovalRecoveryOutcome.COMPLETED,
                    message="charged once",
                )
            ):
                pass
            assert (await sessions.load(session_id)).status is SessionStatus.COMPLETED
            assert charge.runs == 1
        finally:
            await app.aclose()

    asyncio.run(scenario())


def test_only_the_wakes_own_subagent_replay_carries_its_authority() -> None:
    from cayu.runtime._foreground_gate_continuation import GateReplay
    from cayu.runtime._recovery_coordinator import _replays_wake_delivery

    async def scenario() -> None:
        clock = _Clock()
        sessions, tasks = InMemorySessionStore(), InMemoryTaskStore(clock=clock)
        pending = PendingUserInput.model_validate(
            await _sleeping(_app(sessions, tasks, clock), sessions, "s_replay")
        )
        asked = pending.model_copy(update={"wake": None})

        def replay(kind: str, action_id: str) -> GateReplay:
            return GateReplay(terminal=None, record={"kind": kind, "action_id": action_id})

        assert _replays_wake_delivery(replay("input", pending.input_id), pending)
        assert not _replays_wake_delivery(None, pending)
        assert not _replays_wake_delivery(replay("input", "another-pause"), pending)
        assert not _replays_wake_delivery(replay("approval", pending.input_id), pending)
        # A person's question replays as it always did, never as a wake.
        assert not _replays_wake_delivery(replay("input", asked.input_id), asked)

    asyncio.run(scenario())

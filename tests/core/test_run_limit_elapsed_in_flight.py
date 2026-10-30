"""max_elapsed_seconds bounds a model step or tool call in flight, on a virtual clock."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Callable, Coroutine
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

import pytest
from tests.core.test_durable_subagents import (
    _BlockingDurableChildCreationStore,
    _DurableSubagentProvider,
    _register_durable_subagent_agents,
)

from cayu import deadlines
from cayu.agents import AgentSpec
from cayu.applications import CayuApp
from cayu.budgets.base import BudgetLimit, BudgetReservation
from cayu.budgets.pricing import ModelPrice, PriceBook
from cayu.budgets.run_limits import RunLimits
from cayu.configuration import CayuConfig, ToolExecutionConfig
from cayu.context.base import CheckpointCompactionContextPolicy, ModelCompactor
from cayu.context.structured_output import (
    STRUCTURED_OUTPUT_TOOL_NAME,
    StructuredOutputSpec,
    StructuredOutputStrategy,
)
from cayu.deadlines import ExecutionDeadline
from cayu.evals.testing import ScriptedModelProvider
from cayu.events import Event, EventType
from cayu.messages import Message
from cayu.providers import ModelRequest, ModelStreamEvent
from cayu.runtime._elapsed_limit import ElapsedLimitExpired, elapsed_limit_stream
from cayu.runtime._tool_execution import _tool_timer
from cayu.sessions.memory import InMemorySessionStore
from cayu.sessions.records import SessionStatus
from cayu.sessions.requests import RunRequest
from cayu.tasks.dispatch import TaskStoreDispatcher
from cayu.tasks.memory import InMemoryTaskStore
from cayu.tasks.queries import TaskQuery
from cayu.tools.base import Tool, ToolContext, ToolEffect, ToolResult, ToolSpec

LIMIT_SECONDS = 60
RESERVE_SECONDS = 20
SLOW_SECONDS = 300


class VirtualClockLoop(asyncio.SelectorEventLoop):
    """An event loop whose clock jumps to the next timer whenever nothing is ready.

    Sleeps and timeouts take no real time, and they fire in clock order. While
    executor jobs are running the loop waits for them in real time instead, so
    the clock never jumps past work that is still in progress.
    """

    def __init__(self) -> None:
        super().__init__()
        self.now = 0.0
        self._executor_jobs = 0
        real_select = self._selector.select

        def select(timeout: float | None = None) -> Any:
            if timeout is None or self._executor_jobs:
                return real_select(timeout)
            ready = real_select(0)
            if not ready and timeout > 0:
                self.now += timeout
            return ready

        self._selector.select = select

    def time(self) -> float:
        return self.now

    def run_in_executor(self, executor: Any, func: Any, *args: Any) -> asyncio.Future[Any]:
        future = super().run_in_executor(executor, func, *args)
        self._executor_jobs += 1

        def finished(_: asyncio.Future[Any]) -> None:
            self._executor_jobs -= 1

        future.add_done_callback(finished)
        return future

    async def shutdown_default_executor(self, timeout: float | None = None) -> None:
        # A virtual timeout would expire at once; joining takes real time.
        await super().shutdown_default_executor(None)


_WALL_CLOCK_ORIGIN = datetime(2030, 1, 1, tzinfo=UTC)


def run_on_virtual_clock(
    monkeypatch: pytest.MonkeyPatch,
    scenario: Callable[[VirtualClockLoop], Coroutine[Any, Any, None]],
) -> None:
    with asyncio.Runner(loop_factory=VirtualClockLoop) as runner:
        loop = runner.get_loop()
        assert isinstance(loop, VirtualClockLoop)

        class VirtualDatetime(datetime):
            @classmethod
            def now(cls, tz: Any = None) -> datetime:
                return _WALL_CLOCK_ORIGIN + timedelta(seconds=loop.time())

        # Run-scope elapsed time, execution deadlines and the loop's timers then
        # read one clock.
        monkeypatch.setattr(time, "monotonic", loop.time)
        monkeypatch.setattr(deadlines, "datetime", VirtualDatetime)
        runner.run(scenario(loop))


class SlowTool(Tool):
    # Like a shell command: it runs alone, so the calls after it wait for it.
    spec = ToolSpec(
        name="slow_check",
        input_schema={"type": "object", "properties": {}},
        effect=ToolEffect.NONE,
        parallel_safe=False,
    )

    def __init__(self) -> None:
        self.started = 0
        self.cancelled = 0

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        self.started += 1
        try:
            await asyncio.sleep(SLOW_SECONDS)
        except asyncio.CancelledError:
            self.cancelled += 1
            raise
        return ToolResult(content="All checks passed.")


class ParallelSlowTool(SlowTool):
    spec = ToolSpec(
        name="slow_parallel",
        input_schema={"type": "object", "properties": {}},
        effect=ToolEffect.NONE,
    )


class QuickTool(Tool):
    spec = ToolSpec(
        name="read_notes",
        input_schema={"type": "object", "properties": {}},
        effect=ToolEffect.NONE,
    )

    def __init__(self) -> None:
        self.calls = 0

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        self.calls += 1
        return ToolResult(content="Notes.")


class SlowStepProvider(ScriptedModelProvider):
    """Scripted batches, each answered after a scripted delay."""

    def __init__(self, batches: list[list[ModelStreamEvent]], delays: list[float]) -> None:
        super().__init__(batches, supports_native_structured_output=True)
        self._delays = list(delays)
        self.dispatched = 0
        self.cancelled = 0

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        self.dispatched += 1
        delay = self._delays.pop(0) if self._delays else 0.0
        try:
            await asyncio.sleep(delay)
        except asyncio.CancelledError:
            self.cancelled += 1
            raise
        async for event in super().stream(request):
            yield event


def verdict_spec() -> StructuredOutputSpec:
    return StructuredOutputSpec(
        strategy=StructuredOutputStrategy.TOOL,
        json_schema={
            "type": "object",
            "properties": {"verdict": {"type": "string"}},
            "required": ["verdict"],
            "additionalProperties": False,
        },
    )


def tool_calls(*names: str) -> list[ModelStreamEvent]:
    return [
        *(
            ModelStreamEvent.tool_call(id=f"call-{index}", name=name, arguments={})
            for index, name in enumerate(names)
        ),
        ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
    ]


def verdict_answer() -> list[ModelStreamEvent]:
    return [
        ModelStreamEvent.tool_call(
            id="verdict",
            name=STRUCTURED_OUTPUT_TOOL_NAME,
            arguments={"output": {"verdict": "fix is incomplete"}},
        ),
        ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
    ]


def text_answer() -> list[ModelStreamEvent]:
    return [
        ModelStreamEvent.text_delta("done"),
        ModelStreamEvent.completed({"finish_reason": "stop"}),
    ]


def app_with(
    provider: ScriptedModelProvider,
    *tools: Tool,
    config: CayuConfig | None = None,
    **agent_options: Any,
) -> CayuApp:
    app = CayuApp(config=config, enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="reviewer", model="fake"), tools=list(tools), **agent_options)
    return app


async def run_events(app: CayuApp, request: RunRequest) -> list[Event]:
    return [event async for event in app.run(request)]


def only(events: list[Event], event_type: EventType) -> Event:
    matching = [event for event in events if event.type is event_type]
    assert len(matching) == 1, [event.type for event in events]
    return matching[0]


async def tool_result(app: CayuApp, session_id: str, tool_call_id: str) -> dict[str, Any]:
    # The run's stream projects runtime-owned ids; the store keeps the call ids.
    terminal = next(
        event
        for event in await app.session_store.load_events(session_id)
        if event.type in {EventType.TOOL_CALL_COMPLETED, EventType.TOOL_CALL_FAILED}
        and event.payload.get("tool_call_id") == tool_call_id
    )
    return terminal.payload["result"]


@pytest.mark.parametrize("on_limit", ["interrupt", "finish"])
def test_model_step_in_flight_is_cut_at_the_elapsed_limit(monkeypatch, on_limit):
    # One model step takes 20 minutes; the limit allows one minute.
    provider = SlowStepProvider([text_answer()], delays=[1200])
    app = app_with(provider)

    async def scenario(loop: VirtualClockLoop) -> None:
        started = loop.time()
        events = await run_events(
            app,
            RunRequest(
                agent_name="reviewer",
                session_id="slow-model",
                messages=[Message.text("user", "Review the fix.")],
                max_steps=None,
                limits=RunLimits(max_elapsed_seconds=LIMIT_SECONDS, on_limit=on_limit),
                structured_output=verdict_spec() if on_limit == "finish" else None,
            ),
        )
        assert loop.time() - started == pytest.approx(LIMIT_SECONDS)
        assert provider.cancelled == 1
        assert not any(event.type is EventType.MODEL_COMPLETED for event in events)
        reached = only(events, EventType.SESSION_LIMIT_REACHED)
        assert reached.payload["limit"] == "elapsed_seconds"
        assert reached.payload["maximum"] == LIMIT_SECONDS
        terminal = only(events, EventType.SESSION_INTERRUPTED)
        assert terminal.payload["reason"] == "limit_reached"
        assert terminal.payload["limit"] == "elapsed_seconds"
        # A cut model step leaves no answer; finishing needs a model step of its own.
        assert terminal.payload["structured_output_pending"] is (on_limit == "finish")
        session = await app.session_store.load("slow-model")
        assert session is not None and session.status is SessionStatus.INTERRUPTED

    run_on_virtual_clock(monkeypatch, scenario)


def test_tool_call_in_flight_is_cut_at_the_elapsed_limit(monkeypatch):
    slow = SlowTool()
    provider = SlowStepProvider([tool_calls("slow_check"), text_answer()], delays=[])
    app = app_with(provider, slow)

    async def scenario(loop: VirtualClockLoop) -> None:
        started = loop.time()
        events = await run_events(
            app,
            RunRequest(
                agent_name="reviewer",
                session_id="slow-tool",
                messages=[Message.text("user", "Run the checks.")],
                max_steps=None,
                limits=RunLimits(max_elapsed_seconds=LIMIT_SECONDS),
            ),
        )
        assert loop.time() - started == pytest.approx(LIMIT_SECONDS)
        assert (slow.started, slow.cancelled) == (1, 1)
        result = await tool_result(app, "slow-tool", "call-0")
        assert result["is_error"] is True
        assert result["content"] == (
            f"Tool call stopped after {LIMIT_SECONDS} seconds: the run reached its time limit."
        )
        assert result["structured"]["terminal_outcome"] == "tool_execution_timeout"
        # No model step follows the cut call: the boundary stops the run.
        assert len(provider.requests) == 1
        assert only(events, EventType.SESSION_LIMIT_REACHED).payload["limit"] == "elapsed_seconds"
        terminal = only(events, EventType.SESSION_INTERRUPTED)
        assert terminal.payload["limit"] == "elapsed_seconds"
        assert terminal.payload["structured_output_pending"] is False

    run_on_virtual_clock(monkeypatch, scenario)


# One following call runs on its own; two parallel-safe calls form a parallel segment.
@pytest.mark.parametrize("following_calls", [1, 2])
def test_finish_produces_a_validated_answer_after_cutting_a_tool_call(monkeypatch, following_calls):
    slow = SlowTool()
    quick = QuickTool()
    provider = SlowStepProvider(
        [tool_calls("slow_check", *["read_notes"] * following_calls), verdict_answer()],
        delays=[],
    )
    app = app_with(provider, slow, quick)

    async def scenario(loop: VirtualClockLoop) -> None:
        started = loop.time()
        events = await run_events(
            app,
            RunRequest(
                agent_name="reviewer",
                session_id="finish-after-cut",
                messages=[Message.text("user", "Review the fix.")],
                max_steps=None,
                limits=RunLimits(
                    max_elapsed_seconds=LIMIT_SECONDS,
                    on_limit="finish",
                    finish_reserve_seconds=RESERVE_SECONDS,
                ),
                structured_output=verdict_spec(),
            ),
        )
        # The slow call stops when only the finishing reserve is left.
        cutoff = LIMIT_SECONDS - RESERVE_SECONDS
        assert loop.time() - started == pytest.approx(cutoff)
        assert (slow.started, slow.cancelled) == (1, 1)
        assert (await tool_result(app, "finish-after-cut", "call-0"))["content"] == (
            f"Tool call stopped after {cutoff} seconds: the run reached its time limit."
        )
        # The round's later calls are skipped, not started with no time left.
        assert quick.calls == 0
        for index in range(1, following_calls + 1):
            skipped = await tool_result(app, "finish-after-cut", f"call-{index}")
            assert skipped["structured"]["skipped"] is True
            assert skipped["structured"]["limit"] == "elapsed_seconds"
        # The finishing step withdraws ordinary tools and returns the answer.
        assert len(provider.requests) == 2
        assert [tool["name"] for tool in provider.requests[1].tools] == [
            STRUCTURED_OUTPUT_TOOL_NAME
        ]
        validated = only(events, EventType.STRUCTURED_OUTPUT_VALIDATED)
        terminal = only(events, EventType.SESSION_INTERRUPTED)
        assert events.index(validated) < events.index(terminal)
        assert terminal.payload["limit"] == "elapsed_seconds"
        assert terminal.payload["structured_output_pending"] is False
        assert terminal.payload["finish_up"] is True

    run_on_virtual_clock(monkeypatch, scenario)


def test_finish_reserve_also_cuts_a_tool_call_before_the_execution_deadline(monkeypatch):
    slow = SlowTool()
    provider = SlowStepProvider([tool_calls("slow_check"), verdict_answer()], delays=[])
    app = app_with(provider, slow)

    async def scenario(loop: VirtualClockLoop) -> None:
        started = loop.time()
        events = await run_events(
            app,
            RunRequest(
                agent_name="reviewer",
                messages=[Message.text("user", "Review the fix.")],
                max_steps=None,
                execution_deadline=ExecutionDeadline.after(LIMIT_SECONDS),
                limits=RunLimits(on_limit="finish", finish_reserve_seconds=RESERVE_SECONDS),
                structured_output=verdict_spec(),
            ),
        )
        assert loop.time() - started == pytest.approx(LIMIT_SECONDS - RESERVE_SECONDS)
        assert slow.cancelled == 1
        assert len(provider.requests) == 2
        validated = only(events, EventType.STRUCTURED_OUTPUT_VALIDATED)
        terminal = only(events, EventType.SESSION_INTERRUPTED)
        assert events.index(validated) < events.index(terminal)
        assert terminal.payload["limit"] == "execution_deadline"
        assert terminal.payload["structured_output_pending"] is False

    run_on_virtual_clock(monkeypatch, scenario)


def test_finishing_step_in_flight_is_cut_at_the_elapsed_limit(monkeypatch):
    slow = SlowTool()
    # The finishing step itself outlasts the reserve.
    provider = SlowStepProvider([tool_calls("slow_check"), verdict_answer()], delays=[0, 1200])
    app = app_with(provider, slow)

    async def scenario(loop: VirtualClockLoop) -> None:
        started = loop.time()
        events = await run_events(
            app,
            RunRequest(
                agent_name="reviewer",
                messages=[Message.text("user", "Review the fix.")],
                max_steps=None,
                limits=RunLimits(
                    max_elapsed_seconds=LIMIT_SECONDS,
                    on_limit="finish",
                    finish_reserve_seconds=RESERVE_SECONDS,
                ),
                structured_output=verdict_spec(),
            ),
        )
        assert loop.time() - started == pytest.approx(LIMIT_SECONDS)
        assert (provider.dispatched, provider.cancelled) == (2, 1)
        terminal = only(events, EventType.SESSION_INTERRUPTED)
        assert terminal.payload["limit"] == "elapsed_seconds"
        assert terminal.payload["structured_output_pending"] is True
        assert terminal.payload["finish_up"] is True

    run_on_virtual_clock(monkeypatch, scenario)


@pytest.mark.parametrize("on_limit", ["interrupt", "finish"])
def test_parallel_call_queued_past_the_time_left_is_skipped(monkeypatch, on_limit):
    slow = ParallelSlowTool()
    provider = SlowStepProvider(
        [tool_calls("slow_parallel", "slow_parallel", "slow_parallel"), verdict_answer()],
        delays=[],
    )
    # Two slots for three calls: the third waits for a slot until the cut.
    app = app_with(
        provider,
        slow,
        config=CayuConfig(tool_execution=ToolExecutionConfig(max_parallel_tool_calls=2)),
    )

    async def scenario(loop: VirtualClockLoop) -> None:
        events = await run_events(
            app,
            RunRequest(
                agent_name="reviewer",
                session_id="queued-call",
                messages=[Message.text("user", "Run the checks.")],
                max_steps=None,
                limits=RunLimits(
                    max_elapsed_seconds=LIMIT_SECONDS,
                    on_limit=on_limit,
                    finish_reserve_seconds=RESERVE_SECONDS,
                ),
                structured_output=verdict_spec(),
            ),
        )
        assert (slow.started, slow.cancelled) == (2, 2)
        stored = await app.session_store.load_events("queued-call")
        started = [
            event.payload["tool_call_id"]
            for event in stored
            if event.type is EventType.TOOL_CALL_STARTED
        ]
        assert started == ["call-0", "call-1"]
        skipped = await tool_result(app, "queued-call", "call-2")
        assert skipped["structured"]["skipped"] is True
        assert skipped["structured"]["limit"] == "elapsed_seconds"
        terminal = only(events, EventType.SESSION_INTERRUPTED)
        assert terminal.payload["limit"] == "elapsed_seconds"
        assert terminal.payload["structured_output_pending"] is (on_limit == "interrupt")

    run_on_virtual_clock(monkeypatch, scenario)


def test_compaction_completion_does_not_end_the_model_call_bound(monkeypatch):
    summarizer = ScriptedModelProvider(
        [
            ModelStreamEvent.text_delta("Earlier review notes."),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ],
        name="summarizer",
    )
    provider = SlowStepProvider([text_answer()], delays=[1200])
    app = app_with(
        provider,
        context_policy=CheckpointCompactionContextPolicy(
            compactor=ModelCompactor(provider=summarizer, model="summary-model"),
            max_user_turns=1,
            compact_after_messages=2,
        ),
    )

    async def scenario(loop: VirtualClockLoop) -> None:
        started = loop.time()
        events = await run_events(
            app,
            RunRequest(
                agent_name="reviewer",
                messages=[
                    Message.text("user", "Review the first fix."),
                    Message.text("assistant", "It was incomplete."),
                    Message.text("user", "Review the second fix."),
                ],
                max_steps=None,
                limits=RunLimits(max_elapsed_seconds=LIMIT_SECONDS),
            ),
        )
        # The step's compaction completed first; the model call after it is still bounded.
        assert any(
            event.type is EventType.MODEL_COMPLETED
            and event.payload.get("purpose") == "context_compaction"
            for event in events
        )
        assert loop.time() - started == pytest.approx(LIMIT_SECONDS)
        assert provider.cancelled == 1
        assert only(events, EventType.SESSION_INTERRUPTED).payload["limit"] == "elapsed_seconds"

    run_on_virtual_clock(monkeypatch, scenario)


def test_model_call_cut_settles_its_budget_reservation_and_stops_for_the_limit(monkeypatch):
    provider = SlowStepProvider([text_answer()], delays=[1200])
    app = app_with(provider)
    reserving_budget = BudgetLimit(
        scope="app",
        max_estimated_cost=Decimal("100"),
        pricing=PriceBook(
            prices=(
                ModelPrice.fixed(
                    provider_name="scripted",
                    model="fake",
                    input_per_million=Decimal("1"),
                    output_per_million=Decimal("1"),
                ),
            )
        ),
        reservation=BudgetReservation(max_input_tokens=1_000, max_output_tokens=1_000),
    )

    async def scenario(loop: VirtualClockLoop) -> None:
        started = loop.time()
        events = await run_events(
            app,
            RunRequest(
                agent_name="reviewer",
                messages=[Message.text("user", "Review the fix.")],
                max_steps=None,
                limits=RunLimits(max_elapsed_seconds=LIMIT_SECONDS),
                budget_limits=(reserving_budget,),
            ),
        )
        assert loop.time() - started == pytest.approx(LIMIT_SECONDS)
        assert provider.cancelled == 1
        reached = only(events, EventType.SESSION_LIMIT_REACHED)
        assert reached.payload["limit"] == "elapsed_seconds"
        terminal = only(events, EventType.SESSION_INTERRUPTED)
        assert terminal.payload["reason"] == "limit_reached"
        assert "abandoned" not in terminal.payload

    run_on_virtual_clock(monkeypatch, scenario)


def test_interrupt_keeps_its_stop_and_cuts_tools_at_the_full_limit(monkeypatch):
    slow = SlowTool()
    provider = SlowStepProvider([tool_calls("slow_check"), verdict_answer()], delays=[])
    app = app_with(provider, slow)

    async def scenario(loop: VirtualClockLoop) -> None:
        started = loop.time()
        events = await run_events(
            app,
            RunRequest(
                agent_name="reviewer",
                messages=[Message.text("user", "Review the fix.")],
                max_steps=None,
                # The reserve only matters to on_limit="finish".
                limits=RunLimits(
                    max_elapsed_seconds=LIMIT_SECONDS, finish_reserve_seconds=RESERVE_SECONDS
                ),
                structured_output=verdict_spec(),
            ),
        )
        assert loop.time() - started == pytest.approx(LIMIT_SECONDS)
        assert slow.cancelled == 1
        assert len(provider.requests) == 1
        assert not any(event.type is EventType.STRUCTURED_OUTPUT_VALIDATED for event in events)
        terminal = only(events, EventType.SESSION_INTERRUPTED)
        assert terminal.payload["limit"] == "elapsed_seconds"
        assert terminal.payload["structured_output_pending"] is True
        assert "finish_up" not in terminal.payload

    run_on_virtual_clock(monkeypatch, scenario)


def durable_subagent_app(sessions: InMemorySessionStore) -> tuple[CayuApp, InMemoryTaskStore]:
    tasks = InMemoryTaskStore()
    # No tool timeout: the durable submission runs in the run's own task.
    app = CayuApp(
        session_store=sessions,
        task_store=tasks,
        dispatcher=TaskStoreDispatcher(tasks),
        enable_logging=False,
    )
    app.register_provider(_DurableSubagentProvider(), default=True)
    _register_durable_subagent_agents(app)
    return app, tasks


def durable_parent_request(session_id: str) -> RunRequest:
    return RunRequest(
        agent_name="parent",
        session_id=session_id,
        messages=[Message.text("user", "parent task")],
        limits=RunLimits(max_elapsed_seconds=LIMIT_SECONDS),
    )


def test_caller_cancellation_during_a_cut_durable_submission_stays_a_cancellation(monkeypatch):
    sessions = _BlockingDurableChildCreationStore()
    app, _tasks = durable_subagent_app(sessions)

    async def scenario(loop: VirtualClockLoop) -> None:
        parent = asyncio.create_task(
            run_events(app, durable_parent_request("cut-submission-cancel"))
        )
        await sessions.child_creation_started.wait()
        # The cut fires while the submission settles; then the caller cancels too.
        await asyncio.sleep(LIMIT_SECONDS + 1)
        parent.cancel("caller stop")
        sessions.release_child_creation.set()
        with pytest.raises(asyncio.CancelledError, match="caller stop"):
            await parent
        assert parent.cancelling() == 1

    run_on_virtual_clock(monkeypatch, scenario)


def test_cut_durable_submission_keeps_its_committed_child(monkeypatch):
    sessions = _BlockingDurableChildCreationStore()
    app, tasks = durable_subagent_app(sessions)

    async def scenario(loop: VirtualClockLoop) -> None:
        parent = asyncio.create_task(
            run_events(app, durable_parent_request("cut-submission-commit"))
        )
        await sessions.child_creation_started.wait()
        await asyncio.sleep(LIMIT_SECONDS + 1)
        sessions.release_child_creation.set()
        events = await parent
        queued = [
            event.payload["result"]["structured"]["status"]
            for event in events
            if event.type is EventType.TOOL_CALL_COMPLETED
        ]
        assert queued == ["queued"]
        assert len(await tasks.list_tasks(TaskQuery())) == 1
        assert only(events, EventType.SESSION_INTERRUPTED).payload["limit"] == "elapsed_seconds"

    run_on_virtual_clock(monkeypatch, scenario)


def test_quick_tool_keeps_its_result_under_the_elapsed_limit(monkeypatch):
    quick = QuickTool()
    provider = SlowStepProvider([tool_calls("read_notes"), text_answer()], delays=[])
    app = app_with(provider, quick)

    async def scenario(loop: VirtualClockLoop) -> None:
        events = await run_events(
            app,
            RunRequest(
                agent_name="reviewer",
                session_id="quick-tool",
                messages=[Message.text("user", "Read the notes.")],
                max_steps=None,
                limits=RunLimits(max_elapsed_seconds=LIMIT_SECONDS),
            ),
        )
        assert quick.calls == 1
        assert await tool_result(app, "quick-tool", "call-0") == {
            "content": "Notes.",
            "structured": None,
            "artifacts": [],
            "is_error": False,
        }
        assert any(event.type is EventType.SESSION_COMPLETED for event in events)

    run_on_virtual_clock(monkeypatch, scenario)


@pytest.mark.parametrize(
    ("timeout_seconds", "run_time_left_seconds", "expected"),
    [
        (None, None, (None, False)),
        (30.0, None, (30.0, False)),
        (None, 12.5, (12.5, True)),
        (30.0, 12.5, (12.5, True)),
        (10.0, 12.5, (10.0, False)),
        (None, -3.0, (0.0, True)),
    ],
)
def test_tool_timer_takes_the_earlier_bound(timeout_seconds, run_time_left_seconds, expected):
    assert _tool_timer(timeout_seconds, run_time_left_seconds) == expected


def test_elapsed_limit_stream_cancels_the_producer_and_disarms_once_settled(monkeypatch):
    observed: list[str] = []

    async def producer(delays: list[float]) -> AsyncIterator[float]:
        for delay in delays:
            try:
                await asyncio.sleep(delay)
            except asyncio.CancelledError:
                observed.append("cancelled")
                raise
            yield delay

    async def scenario(loop: VirtualClockLoop) -> None:
        deadline = loop.time() + 10

        def seconds_left() -> float:
            return deadline - loop.time()

        received: list[float] = []
        with pytest.raises(ElapsedLimitExpired):
            async for item in elapsed_limit_stream(
                producer([4, 30]), seconds_left, settled=lambda _item: False
            ):
                received.append(item)
        assert received == [4]
        assert observed == ["cancelled"]
        assert loop.time() == pytest.approx(10)

        # After the settled item, the rest runs past the limit uncut.
        deadline = loop.time() + 10
        received = [
            item
            async for item in elapsed_limit_stream(
                producer([4, 30]), seconds_left, settled=lambda item: item == 4
            )
        ]
        assert received == [4, 30]

    run_on_virtual_clock(monkeypatch, scenario)


def test_elapsed_limit_stream_keeps_a_caller_cancellation(monkeypatch):
    async def producer() -> AsyncIterator[None]:
        await asyncio.sleep(30)
        yield None

    async def scenario(loop: VirtualClockLoop) -> None:
        async def consume() -> None:
            async for _ in elapsed_limit_stream(producer(), lambda: 10.0, settled=lambda _: False):
                pass

        task = asyncio.create_task(consume())
        await asyncio.sleep(5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    run_on_virtual_clock(monkeypatch, scenario)


@pytest.mark.parametrize("settles_after_yield", [False, True])
def test_elapsed_limit_stream_keeps_the_cut_while_the_producer_settles_it(
    monkeypatch, settles_after_yield
):
    async def producer() -> AsyncIterator[str]:
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            # Budget settlement yields its events before the cut propagates.
            yield "settlement-1"
            if settles_after_yield:
                await asyncio.sleep(5)
                yield "settlement-2"
            raise
        yield "answer"

    async def scenario(loop: VirtualClockLoop) -> None:
        received: list[str] = []
        with pytest.raises(ElapsedLimitExpired):
            async for item in elapsed_limit_stream(
                producer(), lambda: 10.0 - loop.time(), settled=lambda _: False
            ):
                received.append(item)
        assert received == (
            ["settlement-1", "settlement-2"] if settles_after_yield else ["settlement-1"]
        )

    run_on_virtual_clock(monkeypatch, scenario)

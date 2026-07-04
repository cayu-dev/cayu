from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from cayu.core import AgentSpec, Event, EventType, Message, ToolResultPart
from cayu.core.tools import Tool, ToolContext, ToolResult, ToolSpec
from cayu.providers import ModelProvider, ModelRequest, ModelStreamEvent
from cayu.runtime import (
    CayuApp,
    InMemorySessionStore,
    RunLimits,
    RunRequest,
)
from cayu.tools.commands import ExecCommandTool
from cayu.tools.files import WriteFileTool
from cayu.tools.knowledge import RememberKnowledgeTool

_TOOL_SCHEMA = {
    "type": "object",
    "properties": {
        "tag": {"type": "string"},
        "delay": {"type": "number"},
    },
    "required": ["tag"],
}


class _ScriptedProvider(ModelProvider):
    """Emits one round of the given tool calls, then finishes on the next step."""

    name = "fake"

    def __init__(self, tool_calls: list[tuple[str, str, dict]]) -> None:
        self._tool_calls = tool_calls
        self.requests: list[ModelRequest] = []

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        self.requests.append(request)
        if len(self.requests) == 1:
            for call_id, name, arguments in self._tool_calls:
                yield ModelStreamEvent.tool_call(id=call_id, name=name, arguments=arguments)
            yield ModelStreamEvent.completed({"finish_reason": "tool_calls"})
            return
        yield ModelStreamEvent.text_delta("done")
        yield ModelStreamEvent.completed({"finish_reason": "stop"})


class _Recorder:
    def __init__(self) -> None:
        self.active = 0
        self.max_active = 0
        self.order: list[str] = []
        self.completed: list[str] = []


class _RecordingTool(Tool):
    """Tool that records concurrency and ordering via a shared recorder."""

    def __init__(
        self,
        recorder: _Recorder,
        *,
        name: str = "recording_tool",
        parallel_safe: bool = True,
    ) -> None:
        super().__init__(
            ToolSpec(
                name=name,
                description="records execution",
                input_schema=_TOOL_SCHEMA,
                parallel_safe=parallel_safe,
            )
        )
        self._recorder = recorder

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        rec = self._recorder
        tag = args["tag"]
        rec.active += 1
        rec.max_active = max(rec.max_active, rec.active)
        rec.order.append(f"start:{tag}")
        try:
            await asyncio.sleep(args.get("delay", 0.05))
        finally:
            rec.active -= 1
        rec.order.append(f"end:{tag}")
        rec.completed.append(tag)
        return ToolResult(content=tag)


async def _collect(app: CayuApp, request: RunRequest) -> list[Event]:
    return [event async for event in app.run(request)]


def _build(
    *,
    tools: list[Tool],
    tool_calls: list[tuple[str, str, dict]],
    max_parallel_tool_calls: int = 4,
) -> CayuApp:
    app = CayuApp(
        session_store=InMemorySessionStore(),
        enable_logging=False,
        max_parallel_tool_calls=max_parallel_tool_calls,
    )
    app.register_provider(_ScriptedProvider(tool_calls), default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"), tools=tools)
    return app


def test_builtin_mutating_tools_are_not_parallel_safe() -> None:
    # The framework's own side-effecting built-ins must opt out of concurrent execution so the
    # default-on parallel engine never runs concurrent writes/commands/knowledge mutations.
    assert ExecCommandTool.spec.parallel_safe is False
    assert WriteFileTool.spec.parallel_safe is False
    assert RememberKnowledgeTool.spec.parallel_safe is False


def test_parallel_safe_tools_run_concurrently() -> None:
    recorder = _Recorder()
    app = _build(
        tools=[_RecordingTool(recorder, name="safe_tool")],
        tool_calls=[
            ("a", "safe_tool", {"tag": "a"}),
            ("b", "safe_tool", {"tag": "b"}),
        ],
    )
    events = asyncio.run(
        _collect(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="s_concurrent",
                messages=[Message.text("user", "go")],
            ),
        )
    )
    assert events[-1].type == EventType.SESSION_COMPLETED
    # The two parallel-safe calls overlapped in time.
    assert recorder.max_active == 2


def test_parallel_safe_false_does_not_overlap_and_runs_after_the_batch() -> None:
    recorder = _Recorder()
    app = _build(
        tools=[
            _RecordingTool(recorder, name="safe_tool"),
            _RecordingTool(recorder, name="serial_tool", parallel_safe=False),
        ],
        tool_calls=[
            ("a", "safe_tool", {"tag": "a"}),
            ("b", "safe_tool", {"tag": "b"}),
            ("c", "serial_tool", {"tag": "c"}),
        ],
    )
    events = asyncio.run(
        _collect(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="s_unsafe",
                messages=[Message.text("user", "go")],
            ),
        )
    )
    assert events[-1].type == EventType.SESSION_COMPLETED
    # The two parallel-safe calls overlapped; the serial tool ran alone, after both finished.
    assert recorder.max_active == 2
    assert recorder.order.index("start:c") > recorder.order.index("end:a")
    assert recorder.order.index("start:c") > recorder.order.index("end:b")


def test_mixed_round_tool_results_keep_model_order() -> None:
    # Model emits [safe a, unsafe b, safe c]; execution runs the safe batch first then the unsafe
    # tool, but the tool_result parts must line up with the assistant tool-call order.
    recorder = _Recorder()
    app = _build(
        tools=[
            _RecordingTool(recorder, name="safe_tool"),
            _RecordingTool(recorder, name="serial_tool", parallel_safe=False),
        ],
        tool_calls=[
            ("a", "safe_tool", {"tag": "a"}),
            ("b", "serial_tool", {"tag": "b"}),
            ("c", "safe_tool", {"tag": "c"}),
        ],
    )
    events = asyncio.run(
        _collect(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="s_order",
                messages=[Message.text("user", "go")],
            ),
        )
    )
    assert events[-1].type == EventType.SESSION_COMPLETED
    transcript = asyncio.run(app.session_store.load_transcript("s_order"))
    tool_message = next(message for message in transcript if message.role == "tool")
    result_ids = [
        part.tool_call_id for part in tool_message.content if isinstance(part, ToolResultPart)
    ]
    assert result_ids == ["a", "b", "c"]


def test_parallel_safe_false_is_an_ordering_barrier_in_model_position() -> None:
    # [safe A, safe B, unsafe C, safe D]: A/B run concurrently, then the unsafe barrier C alone,
    # then D — preserving model order (NOT A/B/D before C).
    recorder = _Recorder()
    app = _build(
        tools=[
            _RecordingTool(recorder, name="safe_tool"),
            _RecordingTool(recorder, name="serial_tool", parallel_safe=False),
        ],
        tool_calls=[
            ("a", "safe_tool", {"tag": "a"}),
            ("b", "safe_tool", {"tag": "b"}),
            ("c", "serial_tool", {"tag": "c"}),
            ("d", "safe_tool", {"tag": "d"}),
        ],
    )
    events = asyncio.run(
        _collect(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="s_barrier",
                messages=[Message.text("user", "go")],
            ),
        )
    )
    assert events[-1].type == EventType.SESSION_COMPLETED
    assert recorder.max_active == 2  # A and B overlapped; the barrier never overlaps
    order = recorder.order
    # C runs only after both A and B finished, and D only after C finished.
    assert order.index("start:c") > order.index("end:a")
    assert order.index("start:c") > order.index("end:b")
    assert order.index("start:d") > order.index("end:c")
    parts = _tool_result_parts_ordered(app, "s_barrier")
    assert parts == ["a", "b", "c", "d"]


def test_parallel_safe_false_barrier_runs_before_later_safe_reads() -> None:
    # Regression (read-after-write): [unsafe C, safe A, safe B] must run C first (barrier at
    # position 0), so the reads never execute before the write.
    recorder = _Recorder()
    app = _build(
        tools=[
            _RecordingTool(recorder, name="safe_tool"),
            _RecordingTool(recorder, name="serial_tool", parallel_safe=False),
        ],
        tool_calls=[
            ("c", "serial_tool", {"tag": "c"}),
            ("a", "safe_tool", {"tag": "a"}),
            ("b", "safe_tool", {"tag": "b"}),
        ],
    )
    events = asyncio.run(
        _collect(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="s_raw",
                messages=[Message.text("user", "go")],
            ),
        )
    )
    assert events[-1].type == EventType.SESSION_COMPLETED
    order = recorder.order
    assert order.index("end:c") < order.index("start:a")
    assert order.index("end:c") < order.index("start:b")


def _tool_result_parts_ordered(app: CayuApp, session_id: str) -> list[str]:
    transcript = asyncio.run(app.session_store.load_transcript(session_id))
    tool_message = next(m for m in transcript if m.role == "tool")
    return [p.tool_call_id for p in tool_message.content if isinstance(p, ToolResultPart)]


class _CancelTool(Tool):
    spec = ToolSpec(
        name="cancel_tool",
        description="raises CancelledError (simulates a leaked cancel scope)",
        input_schema={"type": "object", "properties": {}},
    )

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        raise asyncio.CancelledError()


def test_parallel_spontaneous_cancel_does_not_brick_the_round() -> None:
    # A parallel tool that raises CancelledError with no session interrupt must not leave the round
    # half-open (a dangling assistant tool-call with no tool_result bricks every later step). Each
    # un-terminated call is completed with a synthesized error result so the session stays runnable.
    recorder = _Recorder()
    app = _build(
        tools=[_CancelTool(), _RecordingTool(recorder, name="safe_tool")],
        tool_calls=[
            ("call_1", "cancel_tool", {}),
            ("call_2", "safe_tool", {"tag": "b"}),
        ],
    )
    events = asyncio.run(
        _collect(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="s_cancel",
                messages=[Message.text("user", "go")],
            ),
        )
    )
    assert events[-1].type == EventType.SESSION_COMPLETED  # not bricked
    transcript = asyncio.run(app.session_store.load_transcript("s_cancel"))
    tool_message = next(m for m in transcript if m.role == "tool")
    results = {p.tool_call_id: p for p in tool_message.content if isinstance(p, ToolResultPart)}
    assert set(results) == {"call_1", "call_2"}  # complete round — every call has a result
    assert results["call_1"].is_error is True  # synthesized abnormal-termination error


def test_max_parallel_one_runs_sequentially() -> None:
    # A global cap of 1 (CayuApp(max_parallel_tool_calls=1)) forces one-at-a-time execution even
    # for parallel-safe tools — the app-level off switch for concurrency.
    recorder = _Recorder()
    app = _build(
        tools=[_RecordingTool(recorder, name="safe_tool")],
        tool_calls=[
            ("a", "safe_tool", {"tag": "a"}),
            ("b", "safe_tool", {"tag": "b"}),
        ],
        max_parallel_tool_calls=1,
    )
    events = asyncio.run(
        _collect(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="s_seq",
                messages=[Message.text("user", "go")],
            ),
        )
    )
    assert events[-1].type == EventType.SESSION_COMPLETED
    assert recorder.max_active == 1


def test_interrupted_tool_round_results_attaches_artifacts_by_tool_call_id() -> None:
    # Parallel interrupt cleanup artifacts must attach to the producing call (keyed), not the
    # first-unfinished-in-model-order call; the bare-list fallback (sequential) keeps first-unfinished.
    from cayu.runtime import _runtime_records as runtime_records
    from cayu.runtime.app import _interrupted_tool_round_results

    a = runtime_records.ToolCallRequest(id="A", name="tool_a", arguments={})
    b = runtime_records.ToolCallRequest(id="B", name="tool_b", arguments={})

    keyed = _interrupted_tool_round_results(
        tool_calls=[a, b],
        completed_outcomes=[],
        cancellation_artifacts_by_id={"B": [{"producer": "B"}]},
    )
    by_id = {outcome.call.id: outcome for outcome in keyed}
    assert by_id["B"].result.artifacts == [{"producer": "B"}]  # attached to the producer
    assert by_id["A"].result.artifacts == []

    fallback = _interrupted_tool_round_results(
        tool_calls=[a, b],
        completed_outcomes=[],
        cancellation_artifacts=[{"producer": "unknown"}],
    )
    fb = {outcome.call.id: outcome for outcome in fallback}
    assert fb["A"].result.artifacts == [{"producer": "unknown"}]  # first-unfinished fallback
    assert fb["B"].result.artifacts == []


def test_limit_stop_mid_round_does_not_strand_later_segment_tool_calls(monkeypatch) -> None:
    # A parallel_safe=False barrier splits a round into multiple sequential segments. When a run
    # limit trips in a non-last segment, the limit-close must record a tool_result for EVERY
    # remaining round call (later segments included). Otherwise those assistant tool_calls dangle
    # with no matching tool_result and the session is unresumable. Regression test: the limit-close
    # is scoped to the whole round, not just the tripping segment.
    import cayu.runtime.app as runtime_app_module

    clock = {"value": 0.0}
    monkeypatch.setattr(runtime_app_module.time, "monotonic", lambda: clock["value"])

    class _ClockAdvancingTool(Tool):
        spec = ToolSpec(
            name="advance_clock",
            description="unsafe barrier that advances the fake clock past the elapsed limit",
            input_schema={"type": "object", "properties": {}},
            parallel_safe=False,
        )

        async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
            clock["value"] = 1.0
            return ToolResult(content="advanced")

    recorder = _Recorder()
    app = _build(
        tools=[_ClockAdvancingTool(), _RecordingTool(recorder, name="reader")],
        tool_calls=[
            (
                "call_1",
                "advance_clock",
                {},
            ),  # segment 1 (barrier): runs, pushes elapsed to the limit
            ("call_2", "advance_clock", {}),  # segment 2 (barrier): limit trips before it runs
            ("call_3", "reader", {"tag": "c"}),  # segment 3: stranded without the whole-round close
        ],
        max_parallel_tool_calls=4,
    )
    events = asyncio.run(
        _collect(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="s_limit_segment",
                messages=[Message.text("user", "go")],
                limits=RunLimits(max_elapsed_seconds=1),
            ),
        )
    )

    assert EventType.SESSION_LIMIT_REACHED in {event.type for event in events}
    assert "c" not in recorder.completed  # the later segment never executed after the limit tripped
    transcript = asyncio.run(app.session_store.load_transcript("s_limit_segment"))
    tool_message = next(m for m in transcript if m.role == "tool")
    result_ids = {p.tool_call_id for p in tool_message.content if isinstance(p, ToolResultPart)}
    assert result_ids == {"call_1", "call_2", "call_3"}  # every round call has a result — no dangle

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from cayu.core import AgentSpec, Event, EventType, Message, ToolResultPart
from cayu.core.tools import Tool, ToolContext, ToolEffect, ToolResult, ToolSpec
from cayu.providers import ModelProvider, ModelRequest, ModelStreamEvent
from cayu.runtime import (
    CayuApp,
    InMemorySessionStore,
    InterruptSessionRequest,
    RunLimits,
    RunRequest,
    ToolPolicy,
    ToolPolicyDecision,
    ToolPolicyRequest,
    ToolPolicyResult,
)
from cayu.runtime import _tool_execution as tool_execution
from cayu.tools.commands import ExecCommandTool
from cayu.tools.files import ListArtifactsTool, ListFilesTool, ReadFileTool, WriteFileTool
from cayu.tools.knowledge import (
    ListKnowledgeTool,
    ReadKnowledgeTool,
    RememberKnowledgeTool,
    SearchKnowledgeTool,
)
from cayu.tools.subagents import SubagentResultTool, SubagentTool

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
        self.context_idempotency_keys: list[str | None] = []
        self.metadata_idempotency_keys: list[str | None] = []
        self.metadata_tool_effects: list[str | None] = []


class _RecordingTool(Tool):
    """Tool that records concurrency and ordering via a shared recorder."""

    def __init__(
        self,
        recorder: _Recorder,
        *,
        name: str = "recording_tool",
        parallel_safe: bool = True,
        effect: ToolEffect = ToolEffect.NONE,
    ) -> None:
        super().__init__(
            ToolSpec(
                name=name,
                description="records execution",
                input_schema=_TOOL_SCHEMA,
                parallel_safe=parallel_safe,
                effect=effect,
            )
        )
        self._recorder = recorder

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        rec = self._recorder
        tag = args["tag"]
        rec.active += 1
        rec.max_active = max(rec.max_active, rec.active)
        rec.order.append(f"start:{tag}")
        rec.context_idempotency_keys.append(ctx.idempotency_key)
        rec.metadata_idempotency_keys.append(ctx.metadata.get("idempotency_key"))
        rec.metadata_tool_effects.append(ctx.metadata.get("tool_effect"))
        try:
            await asyncio.sleep(args.get("delay", 0.05))
        finally:
            rec.active -= 1
        rec.order.append(f"end:{tag}")
        rec.completed.append(tag)
        return ToolResult(content=tag)


class _CapturePolicy(ToolPolicy):
    def __init__(self) -> None:
        self.requests: list[ToolPolicyRequest] = []

    async def authorize(self, request: ToolPolicyRequest) -> ToolPolicyResult:
        self.requests.append(request)
        return ToolPolicyResult(decision=ToolPolicyDecision.ALLOW)


class _FakeSubagentRuntime:
    async def run(self, request: RunRequest) -> AsyncIterator[Event]:
        if False:
            yield Event(type=EventType.SESSION_STARTED, session_id=request.session_id)

    async def interrupt_session(self, request: InterruptSessionRequest) -> AsyncIterator[Event]:
        if False:
            yield Event(type=EventType.SESSION_INTERRUPTED, session_id=request.session_id)


async def _collect(app: CayuApp, request: RunRequest) -> list[Event]:
    return [event async for event in app.run(request)]


def _build(
    *,
    tools: list[Tool],
    tool_calls: list[tuple[str, str, dict]],
    max_parallel_tool_calls: int = 4,
    tool_policy: ToolPolicy | None = None,
) -> CayuApp:
    app = CayuApp(
        session_store=InMemorySessionStore(),
        enable_logging=False,
        max_parallel_tool_calls=max_parallel_tool_calls,
    )
    app.register_provider(_ScriptedProvider(tool_calls), default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=tools,
        tool_policy=tool_policy,
    )
    return app


def test_builtin_mutating_tools_are_not_parallel_safe() -> None:
    # The framework's own side-effecting built-ins must opt out of concurrent execution so the
    # default-on parallel engine never runs concurrent writes/commands/knowledge mutations.
    assert ExecCommandTool.spec.parallel_safe is False
    assert WriteFileTool.spec.parallel_safe is False
    assert RememberKnowledgeTool.spec.parallel_safe is False
    assert ExecCommandTool.spec.effect is ToolEffect.EXTERNAL
    assert WriteFileTool.spec.effect is ToolEffect.EXTERNAL
    assert RememberKnowledgeTool.spec.effect is ToolEffect.EXTERNAL
    assert ReadFileTool.spec.effect is ToolEffect.EXTERNAL
    assert ListFilesTool.spec.effect is ToolEffect.NONE
    assert ListArtifactsTool.spec.effect is ToolEffect.NONE
    assert SearchKnowledgeTool.spec.effect is ToolEffect.NONE
    assert ListKnowledgeTool.spec.effect is ToolEffect.NONE
    assert ReadKnowledgeTool.spec.effect is ToolEffect.NONE
    assert SubagentTool(_FakeSubagentRuntime(), agents={"helper": "helper"}).spec.effect is (
        ToolEffect.EXTERNAL
    )
    assert SubagentResultTool(InMemorySessionStore()).spec.effect is ToolEffect.NONE


def test_tool_idempotency_key_preserves_component_boundaries() -> None:
    first = tool_execution.tool_idempotency_key(
        session_id="session\x00round",
        tool_call_id="call",
        tool_round_id="approval",
    )
    second = tool_execution.tool_idempotency_key(
        session_id="session",
        tool_call_id="call",
        tool_round_id="round\x00approval",
    )

    assert first != second


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


def test_tool_context_receives_stable_idempotency_key_from_round_identity() -> None:
    recorder = _Recorder()
    app = _build(
        tools=[_RecordingTool(recorder, name="identity_tool")],
        tool_calls=[("call_1", "identity_tool", {"tag": "a"})],
    )

    events = asyncio.run(
        _collect(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="s_identity",
                messages=[Message.text("user", "go")],
            ),
        )
    )

    started = next(event for event in events if event.type == EventType.TOOL_CALL_STARTED)
    completed = next(event for event in events if event.type == EventType.TOOL_CALL_COMPLETED)

    key = started.payload["idempotency_key"]
    assert key.startswith("cayu-tool:v1:")
    assert len(key) == len("cayu-tool:v1:") + 64
    assert recorder.context_idempotency_keys == [key]
    assert recorder.metadata_idempotency_keys == [key]

    expected_key = tool_execution.tool_idempotency_key(
        session_id="s_identity",
        tool_round_id=started.payload["tool_round_id"],
        tool_call_id="call_1",
    )
    assert key == expected_key
    assert completed.payload["tool_round_id"] == started.payload["tool_round_id"]
    assert completed.payload["idempotency_key"] == key


def test_tool_effect_reaches_policy_started_event_and_tool_context_metadata() -> None:
    recorder = _Recorder()
    policy = _CapturePolicy()
    app = CayuApp(session_store=InMemorySessionStore(), enable_logging=False)
    app.register_provider(_ScriptedProvider([("call_1", "idem_tool", {"tag": "a"})]), default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[
            _RecordingTool(
                recorder,
                name="idem_tool",
                effect=ToolEffect.IDEMPOTENT,
            )
        ],
        tool_policy=policy,
    )

    events = asyncio.run(
        _collect(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="s_effect",
                messages=[Message.text("user", "go")],
            ),
        )
    )

    started = next(event for event in events if event.type == EventType.TOOL_CALL_STARTED)
    assert started.payload["effect"] == "idempotent"
    assert [request.tool_effect for request in policy.requests] == [ToolEffect.IDEMPOTENT]
    assert recorder.metadata_tool_effects == ["idempotent"]


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


def test_parallel_safe_uses_registered_declaration_copy() -> None:
    recorder = _Recorder()
    serial_tool = _RecordingTool(
        recorder,
        name="serial_tool",
        parallel_safe=False,
        effect=ToolEffect.IDEMPOTENT,
    )
    policy = _CapturePolicy()
    app = _build(
        tools=[
            serial_tool,
            _RecordingTool(recorder, name="safe_tool"),
        ],
        tool_calls=[
            ("c", "serial_tool", {"tag": "c"}),
            ("a", "safe_tool", {"tag": "a"}),
        ],
        tool_policy=policy,
    )

    serial_tool.spec = ToolSpec(
        name="serial_tool",
        description="mutated after registration",
        input_schema=_TOOL_SCHEMA,
        parallel_safe=True,
        effect=ToolEffect.NONE,
    )

    events = asyncio.run(
        _collect(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="s_registered_parallel_safe_copy",
                messages=[Message.text("user", "go")],
            ),
        )
    )

    assert events[-1].type == EventType.SESSION_COMPLETED
    order = recorder.order
    assert order.index("end:c") < order.index("start:a")
    started = next(
        event
        for event in events
        if event.type == EventType.TOOL_CALL_STARTED and event.tool_name == "serial_tool"
    )
    assert started.payload["effect"] == "idempotent"
    assert policy.requests[0].tool_effect is ToolEffect.IDEMPOTENT


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
    abnormal_event = next(
        event
        for event in events
        if event.type == EventType.TOOL_CALL_FAILED
        and event.payload.get("abnormal_termination") is True
    )
    assert abnormal_event.payload["idempotency_key"] == tool_execution.tool_idempotency_key(
        session_id="s_cancel",
        tool_round_id=abnormal_event.payload["tool_round_id"],
        tool_call_id="call_1",
    )


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


def test_limit_stop_mid_round_does_not_strand_later_segment_tool_calls(monkeypatch) -> None:
    # A parallel_safe=False barrier splits a round into multiple sequential segments. When a run
    # limit trips in a non-last segment, the limit-close must record a tool_result for EVERY
    # remaining round call (later segments included). Otherwise those assistant tool_calls dangle
    # with no matching tool_result and the session is unresumable. Regression test: the limit-close
    # is scoped to the whole round, not just the tripping segment.
    import cayu.runtime._session_engine as session_engine_module

    clock = {"value": 0.0}
    monkeypatch.setattr(session_engine_module.time, "monotonic", lambda: clock["value"])

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

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import textwrap
from collections.abc import AsyncIterator
from typing import Any

import pytest

# OpenTelemetry is an optional dependency; skip this whole module (rather than error
# at collection) when it is not installed, matching the house pattern in test_server.py.
pytest.importorskip("opentelemetry.sdk")

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import SpanKind, StatusCode

from cayu import (
    AgentSpec,
    CayuApp,
    Message,
    OpenTelemetryEventSink,
    RunRequest,
    SubagentSpec,
    SubagentTool,
)
from cayu.core import Event, EventType
from cayu.observability import otel
from cayu.providers import ModelProvider, ModelRequest, ModelStreamEvent

REMOTE_TRACE_ID = "11111111111111111111111111111111"
REMOTE_TRACEPARENT = f"00-{REMOTE_TRACE_ID}-2222222222222222-01"


class FakeProvider(ModelProvider):
    name = "anthropic"

    def __init__(self, events: list[ModelStreamEvent] | list[list[ModelStreamEvent]]) -> None:
        # Accept a flat list (same events every call) or a list of per-call batches.
        if events and isinstance(events[0], list):
            self._batches = events
        else:
            self._batches = [events]
        self._calls = 0

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        batch = self._batches[min(self._calls, len(self._batches) - 1)]
        self._calls += 1
        for event in batch:
            yield event


def _make_sink() -> tuple[InMemorySpanExporter, OpenTelemetryEventSink]:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    sink = OpenTelemetryEventSink(tracer=provider.get_tracer("test"))
    return exporter, sink


def _drive(sink: OpenTelemetryEventSink, events: list[Event]) -> None:
    async def go() -> None:
        for event in events:
            await sink.emit(event)

    asyncio.run(go())


def _run(app: CayuApp, request: RunRequest) -> None:
    async def go() -> None:
        async for _ in app.run(request):
            pass

    asyncio.run(go())


def _spans_by_name(exporter: InMemorySpanExporter) -> dict[str, Any]:
    return {span.name: span for span in exporter.get_finished_spans()}


def _session_events(session_id: str, payload: dict[str, Any] | None = None) -> list[Event]:
    return [
        Event(
            type=EventType.SESSION_STARTED,
            session_id=session_id,
            agent_name="assistant",
            environment_name="local",
            payload={"agent_name": "assistant", **(payload or {})},
        ),
        Event(
            type=EventType.MODEL_STARTED,
            session_id=session_id,
            payload={"provider": "anthropic", "model": "claude-opus-4-8", "step": 0},
        ),
        Event(
            type=EventType.MODEL_COMPLETED,
            session_id=session_id,
            # The runtime does NOT carry "step" on MODEL_COMPLETED by default
            # (only on retries), so the span must be correlated without it. The
            # normalized finish reason is nested under "completion" (NOT top-level).
            payload={
                "completion": {
                    "finish_reason": "stop",
                    "raw_finish_reason": "stop",
                    "status": "ok",
                },
                "usage_metrics": {
                    "model": "claude-opus-4-8",
                    "input_tokens": 4200,
                    "output_tokens": 380,
                    "reasoning_output_tokens": 50,
                    "cache": {"read_tokens": 3800, "write_tokens": 120},
                },
            },
        ),
        Event(
            type=EventType.TOOL_CALL_STARTED,
            session_id=session_id,
            tool_name="exec_command",
            payload={"tool_call_id": "call_1"},
        ),
        Event(
            type=EventType.TOOL_CALL_COMPLETED,
            session_id=session_id,
            payload={"tool_call_id": "call_1"},
        ),
        Event(type=EventType.SESSION_COMPLETED, session_id=session_id, payload={}),
    ]


def test_session_produces_nested_span_hierarchy_in_one_trace() -> None:
    exporter, sink = _make_sink()
    _drive(sink, _session_events("sess1"))

    spans = _spans_by_name(exporter)
    assert set(spans) == {
        "cayu.session assistant",
        "chat claude-opus-4-8",
        "execute_tool exec_command",
    }
    session = spans["cayu.session assistant"]
    model = spans["chat claude-opus-4-8"]
    tool = spans["execute_tool exec_command"]
    assert model.parent.span_id == session.context.span_id
    assert tool.parent.span_id == session.context.span_id
    assert len({span.context.trace_id for span in spans.values()}) == 1


def test_model_span_carries_genai_usage_attributes() -> None:
    exporter, sink = _make_sink()
    _drive(sink, _session_events("sess1"))

    model = _spans_by_name(exporter)["chat claude-opus-4-8"]
    # GenAI semconv: an inference span is a CLIENT span.
    assert model.kind == SpanKind.CLIENT
    attrs = model.attributes
    assert attrs["gen_ai.system"] == "anthropic"
    assert attrs["gen_ai.provider.name"] == "anthropic"
    assert attrs["gen_ai.request.model"] == "claude-opus-4-8"
    assert attrs["gen_ai.response.model"] == "claude-opus-4-8"
    assert attrs["gen_ai.response.finish_reasons"] == ("stop",)
    assert attrs["gen_ai.usage.input_tokens"] == 4200
    assert attrs["gen_ai.usage.output_tokens"] == 380
    assert attrs["gen_ai.usage.reasoning.output_tokens"] == 50
    assert attrs["gen_ai.usage.cache_read.input_tokens"] == 3800
    assert attrs["gen_ai.usage.cache_creation.input_tokens"] == 120


def test_tool_span_carries_genai_tool_attributes() -> None:
    exporter, sink = _make_sink()
    _drive(sink, _session_events("sess1"))

    attrs = _spans_by_name(exporter)["execute_tool exec_command"].attributes
    assert attrs["gen_ai.operation.name"] == "execute_tool"
    assert attrs["gen_ai.tool.name"] == "exec_command"
    assert attrs["gen_ai.tool.call.id"] == "call_1"


def test_inbound_traceparent_roots_span_under_remote_trace() -> None:
    exporter, sink = _make_sink()
    _drive(
        sink,
        _session_events(
            "sess1",
            payload={"traceparent": REMOTE_TRACEPARENT, "tracestate": "rojo=abc123"},
        ),
    )

    session = _spans_by_name(exporter)["cayu.session assistant"]
    assert f"{session.context.trace_id:032x}" == REMOTE_TRACE_ID
    assert session.parent is not None
    assert session.parent.is_remote
    # tracestate (vendor sampling/routing data) propagates alongside traceparent.
    assert session.context.trace_state.get("rojo") == "abc123"


def test_subagent_session_nests_under_parent_via_parent_session_id() -> None:
    exporter, sink = _make_sink()
    events = [
        Event(
            type=EventType.SESSION_STARTED,
            session_id="parent",
            agent_name="orchestrator",
            payload={"agent_name": "orchestrator"},
        ),
        Event(
            type=EventType.SESSION_STARTED,
            session_id="child",
            agent_name="verifier",
            payload={"agent_name": "verifier", "parent_session_id": "parent"},
        ),
        Event(type=EventType.SESSION_COMPLETED, session_id="child", payload={}),
        Event(type=EventType.SESSION_COMPLETED, session_id="parent", payload={}),
    ]
    _drive(sink, events)

    spans = _spans_by_name(exporter)
    parent = spans["cayu.session orchestrator"]
    child = spans["cayu.session verifier"]
    assert child.parent.span_id == parent.context.span_id
    assert child.context.trace_id == parent.context.trace_id


def test_subagent_nests_under_parent_in_a_real_run() -> None:
    # End-to-end through CayuApp + a real SubagentTool (not hand-ordered events):
    # the child session span must nest under the parent session span, proving the
    # "subagent sessions appear as child spans" acceptance criterion for real.
    exporter, sink = _make_sink()
    provider = FakeProvider(
        [
            [  # parent turn 1: invoke the subagent tool
                ModelStreamEvent.tool_call(
                    id="call_sub",
                    name="subagent",
                    arguments={"agent": "reviewer", "task": "review the change"},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [  # reviewer (child session) turn
                ModelStreamEvent.text_delta("looks good"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
            [  # parent turn 2: finish
                ModelStreamEvent.text_delta("done"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    app = CayuApp(enable_logging=False, event_sinks=[sink])
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="parent", model="fake-model"),
        tools=[
            SubagentTool(
                app,
                agents={"reviewer": SubagentSpec(agent_name="reviewer", description="Review.")},
            )
        ],
    )
    app.register_agent(AgentSpec(name="reviewer", model="fake-model"))

    _run(
        app,
        RunRequest(
            agent_name="parent",
            session_id="sess_parent",
            messages=[Message.text("user", "go")],
        ),
    )

    spans = _spans_by_name(exporter)
    parent = spans["cayu.session parent"]
    child = spans["cayu.session reviewer"]
    assert child.parent.span_id == parent.context.span_id
    assert child.context.trace_id == parent.context.trace_id
    # Every model span the real run produced is a CLIENT span (GenAI semconv).
    model_spans = [s for s in exporter.get_finished_spans() if s.name == "chat fake-model"]
    assert model_spans
    assert all(s.kind == SpanKind.CLIENT for s in model_spans)


def test_finish_reason_read_from_normalized_completion_in_a_real_run() -> None:
    # A provider whose completed event has NO top-level finish_reason (like the
    # Anthropic provider, which emits stop_reason) driven through the real runtime.
    # The sink must still set finish_reasons from the normalized
    # payload["completion"]["finish_reason"] — the case the unit fixtures had masked.
    exporter, sink = _make_sink()
    provider = FakeProvider([ModelStreamEvent.text_delta("hi"), ModelStreamEvent.completed({})])
    app = CayuApp(enable_logging=False, event_sinks=[sink])
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    _run(
        app,
        RunRequest(
            agent_name="assistant",
            session_id="sess_real",
            messages=[Message.text("user", "hi")],
        ),
    )

    model = [s for s in exporter.get_finished_spans() if s.name == "chat fake-model"]
    assert model
    assert model[0].attributes["gen_ai.response.finish_reasons"] == ("unknown",)


def test_traceparent_for_returns_open_session_context() -> None:
    _, sink = _make_sink()
    asyncio.run(sink.emit(Event(type=EventType.SESSION_STARTED, session_id="sess1", payload={})))
    traceparent = sink.traceparent_for("sess1")
    assert traceparent is not None
    # The injected trace id must match the session's own root span (so dispatched
    # work actually continues this trace), not just be a well-formed header.
    root = sink._sessions["sess1"].root
    assert traceparent.split("-")[1] == f"{root.get_span_context().trace_id:032x}"
    assert sink.traceparent_for("unknown") is None


def test_failures_set_error_status() -> None:
    # MODEL_ERROR carries no "step" on the default path; the span must still close.
    exporter, sink = _make_sink()
    _drive(
        sink,
        [
            Event(type=EventType.SESSION_STARTED, session_id="sess1", payload={}),
            Event(type=EventType.MODEL_STARTED, session_id="sess1", payload={"step": 0}),
            Event(
                type=EventType.MODEL_ERROR,
                session_id="sess1",
                payload={"error_type": "RuntimeError", "error": "boom"},
            ),
            Event(
                type=EventType.SESSION_FAILED,
                session_id="sess1",
                payload={"error_type": "RuntimeError", "error": "boom"},
            ),
        ],
    )
    spans = _spans_by_name(exporter)
    assert spans["chat"].status.status_code == StatusCode.ERROR
    assert spans["chat"].status.description == "RuntimeError: boom"
    assert spans["cayu.session"].status.status_code == StatusCode.ERROR


def test_session_failure_ends_orphaned_children_and_marks_only_the_root() -> None:
    # Model + tool spans with no completion must still be CLOSED on session failure,
    # but left UNSET (we never learned they failed); only the root carries the error.
    exporter, sink = _make_sink()
    _drive(
        sink,
        [
            Event(type=EventType.SESSION_STARTED, session_id="sess1", payload={}),
            Event(type=EventType.MODEL_STARTED, session_id="sess1", payload={"step": 0}),
            Event(
                type=EventType.TOOL_CALL_STARTED,
                session_id="sess1",
                tool_name="exec_command",
                payload={"tool_call_id": "call_1"},
            ),
            Event(
                type=EventType.SESSION_FAILED,
                session_id="sess1",
                payload={"error_type": "RuntimeError", "error": "boom"},
            ),
        ],
    )
    spans = _spans_by_name(exporter)
    assert set(spans) == {"cayu.session", "chat", "execute_tool exec_command"}
    assert spans["cayu.session"].status.status_code == StatusCode.ERROR
    # In-flight children are UNSET but marked incomplete (not a clean success).
    assert spans["chat"].status.status_code == StatusCode.UNSET
    assert spans["chat"].attributes["cayu.incomplete"] is True
    assert spans["execute_tool exec_command"].attributes["cayu.incomplete"] is True
    assert sink._sessions == {}


def test_session_interrupt_is_not_an_error() -> None:
    # An interrupt is a pause (approval wait / user stop), not a failure: the root
    # span is closed but left UNSET, and in-flight children are closed too.
    exporter, sink = _make_sink()
    _drive(
        sink,
        [
            Event(type=EventType.SESSION_STARTED, session_id="sess1", payload={}),
            Event(type=EventType.MODEL_STARTED, session_id="sess1", payload={"step": 0}),
            Event(
                type=EventType.SESSION_INTERRUPTED,
                session_id="sess1",
                payload={"reason": "user_requested", "interruption_type": "pause"},
            ),
        ],
    )
    spans = _spans_by_name(exporter)
    assert set(spans) == {"cayu.session", "chat"}
    assert spans["cayu.session"].status.status_code == StatusCode.UNSET
    assert sink._sessions == {}


def test_blocked_tool_closes_its_span_with_error() -> None:
    # A policy-denied tool emits TOOL_CALL_STARTED then TOOL_CALL_BLOCKED (no
    # COMPLETED/FAILED); the span must close with ERROR, not orphan into a "success".
    exporter, sink = _make_sink()
    _drive(
        sink,
        [
            Event(type=EventType.SESSION_STARTED, session_id="sess1", payload={}),
            Event(
                type=EventType.TOOL_CALL_STARTED,
                session_id="sess1",
                tool_name="exec_command",
                payload={"tool_call_id": "call_1"},
            ),
            Event(
                type=EventType.TOOL_CALL_BLOCKED,
                session_id="sess1",
                tool_name="exec_command",
                # Real BLOCKED payload carries the message under "reason" (not "error").
                payload={
                    "tool_call_id": "call_1",
                    "denied_by": "command_policy",
                    "decision": "deny",
                    "reason": "denied by policy",
                },
            ),
            Event(type=EventType.SESSION_COMPLETED, session_id="sess1", payload={}),
        ],
    )
    tool = _spans_by_name(exporter)["execute_tool exec_command"]
    assert tool.status.status_code == StatusCode.ERROR
    assert tool.status.description == "denied by policy"
    assert tool.attributes[otel.CAYU_TOOL_DENIED_BY] == "command_policy"
    assert tool.attributes[otel.CAYU_TOOL_POLICY_DECISION] == "deny"
    assert (
        len(
            [
                span
                for span in exporter.get_finished_spans()
                if span.name == "execute_tool exec_command"
            ]
        )
        == 1
    )
    # Closed by the BLOCKED event, not swept at session end.
    assert sink._sessions == {}


def test_resumed_mixed_policy_round_traces_blocked_sibling_without_start() -> None:
    exporter, sink = _make_sink()
    _drive(
        sink,
        [
            Event(
                type=EventType.SESSION_RESUMED,
                session_id="sess1",
                agent_name="assistant",
                payload={},
            ),
            # A sibling denied during whole-round planning has no STARTED event when
            # the approval-gated call is later approved and the round resumes.
            Event(
                type=EventType.TOOL_CALL_BLOCKED,
                session_id="sess1",
                tool_name="echo",
                payload={
                    "tool_call_id": "call_denied",
                    "denied_by": "tool_policy",
                    "decision": "deny",
                    "reason": "echo is blocked",
                },
            ),
            Event(
                type=EventType.TOOL_CALL_STARTED,
                session_id="sess1",
                tool_name="side_effect",
                payload={"tool_call_id": "call_approved"},
            ),
            Event(
                type=EventType.TOOL_CALL_COMPLETED,
                session_id="sess1",
                tool_name="side_effect",
                payload={"tool_call_id": "call_approved"},
            ),
            Event(type=EventType.SESSION_COMPLETED, session_id="sess1", payload={}),
        ],
    )

    spans = _spans_by_name(exporter)
    assert set(spans) == {
        "cayu.session assistant",
        "execute_tool echo",
        "execute_tool side_effect",
    }
    session = spans["cayu.session assistant"]
    denied = spans["execute_tool echo"]
    assert denied.parent.span_id == session.context.span_id
    assert denied.start_time == denied.end_time
    assert denied.status.status_code == StatusCode.ERROR
    assert denied.status.description == "echo is blocked"
    assert denied.attributes[otel.GEN_AI_TOOL_NAME] == "echo"
    assert denied.attributes[otel.GEN_AI_TOOL_CALL_ID] == "call_denied"
    assert denied.attributes[otel.CAYU_TOOL_DENIED_BY] == "tool_policy"
    assert denied.attributes[otel.CAYU_TOOL_POLICY_DECISION] == "deny"
    assert sink._sessions == {}


def test_unmatched_non_policy_block_does_not_synthesize_tool_span() -> None:
    exporter, sink = _make_sink()
    _drive(
        sink,
        [
            Event(type=EventType.SESSION_RESUMED, session_id="sess1", payload={}),
            Event(
                type=EventType.TOOL_CALL_BLOCKED,
                session_id="sess1",
                tool_name="echo",
                payload={
                    "tool_call_id": "call_hook_blocked",
                    "blocked_by": "before_tool_call_hook",
                    "reason": "hook blocked the call",
                },
            ),
            Event(type=EventType.SESSION_COMPLETED, session_id="sess1", payload={}),
        ],
    )

    assert [span.name for span in exporter.get_finished_spans()] == ["cayu.session"]
    assert sink._sessions == {}


def test_failed_tool_span_description_comes_from_result_content() -> None:
    # Real TOOL_CALL_FAILED payload carries the message at result.content (a str),
    # NOT a top-level "error" key.
    exporter, sink = _make_sink()
    _drive(
        sink,
        [
            Event(type=EventType.SESSION_STARTED, session_id="sess1", payload={}),
            Event(
                type=EventType.TOOL_CALL_STARTED,
                session_id="sess1",
                tool_name="exec_command",
                payload={"tool_call_id": "call_1"},
            ),
            Event(
                type=EventType.TOOL_CALL_FAILED,
                session_id="sess1",
                tool_name="exec_command",
                payload={"tool_call_id": "call_1", "result": {"content": "boom", "is_error": True}},
            ),
            Event(type=EventType.SESSION_COMPLETED, session_id="sess1", payload={}),
        ],
    )
    tool = _spans_by_name(exporter)["execute_tool exec_command"]
    assert tool.status.status_code == StatusCode.ERROR
    assert tool.status.description == "boom"


def test_duplicate_model_started_closes_the_previous_span() -> None:
    exporter, sink = _make_sink()
    _drive(
        sink,
        [
            Event(type=EventType.SESSION_STARTED, session_id="sess1", payload={}),
            Event(type=EventType.MODEL_STARTED, session_id="sess1", payload={"model": "a"}),
            Event(type=EventType.MODEL_STARTED, session_id="sess1", payload={"model": "b"}),
            Event(type=EventType.SESSION_COMPLETED, session_id="sess1", payload={}),
        ],
    )
    # Both model spans are closed exactly once (the first on the duplicate START).
    names = sorted(span.name for span in exporter.get_finished_spans())
    assert names == ["cayu.session", "chat a", "chat b"]


def test_colon_namespaced_session_ids_do_not_collide() -> None:
    # Ending "a" must not sweep the still-open spans of a different session "a:b".
    exporter, sink = _make_sink()
    _drive(
        sink,
        [
            Event(type=EventType.SESSION_STARTED, session_id="a", payload={}),
            Event(type=EventType.SESSION_STARTED, session_id="a:b", payload={}),
            Event(type=EventType.MODEL_STARTED, session_id="a:b", payload={"model": "m"}),
            Event(type=EventType.SESSION_COMPLETED, session_id="a", payload={}),
        ],
    )
    # "a:b" and its model span are still open; only "a" has finished.
    assert {span.name for span in exporter.get_finished_spans()} == {"cayu.session"}
    assert set(sink._sessions) == {"a:b"}


def test_no_explicit_parent_starts_fresh_root_ignoring_ambient_span() -> None:
    exporter, sink = _make_sink()
    tracer = sink._tracer
    outer = tracer.start_span("outer")
    with sink._trace.use_span(outer):
        _drive(
            sink,
            [
                Event(type=EventType.SESSION_STARTED, session_id="sess1", payload={}),
                Event(type=EventType.SESSION_COMPLETED, session_id="sess1", payload={}),
            ],
        )
    outer.end()
    session = _spans_by_name(exporter)["cayu.session"]
    assert session.parent is None
    assert session.context.trace_id != outer.context.trace_id


def test_malformed_traceparent_starts_fresh_root_without_crashing() -> None:
    exporter, sink = _make_sink()
    _drive(sink, _session_events("sess1", payload={"traceparent": "not-a-traceparent"}))
    session = _spans_by_name(exporter)["cayu.session assistant"]
    assert session.parent is None


def test_unknown_event_type_is_ignored() -> None:
    exporter, sink = _make_sink()
    asyncio.run(sink.emit(Event(type=EventType.HOOK_STARTED, session_id="sess1", payload={})))
    assert exporter.get_finished_spans() == ()


def test_missing_dependency_raises_helpful_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_import(name: str) -> Any:
        raise ModuleNotFoundError(f"No module named '{name}'", name="opentelemetry")

    monkeypatch.setattr(otel.importlib, "import_module", fake_import)
    with pytest.raises(RuntimeError, match=r"cayu\[otel\]"):
        OpenTelemetryEventSink()


def test_unrelated_missing_module_is_not_masked(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_import(name: str) -> Any:
        raise ModuleNotFoundError("No module named 'cachetools'", name="cachetools")

    monkeypatch.setattr(otel.importlib, "import_module", fake_import)
    with pytest.raises(ModuleNotFoundError):
        OpenTelemetryEventSink()


def test_import_cayu_works_without_opentelemetry_installed() -> None:
    # opentelemetry is a dev/test dep here, so prove the optional-dependency
    # invariant in a subprocess where it is blocked at import time: `import cayu`
    # and the export must succeed; only constructing the sink raises the hint.
    program = textwrap.dedent(
        """
        import sys

        class _BlockOpenTelemetry:
            def find_spec(self, name, path, target=None):
                if name == "opentelemetry" or name.startswith("opentelemetry."):
                    raise ModuleNotFoundError(f"No module named '{name}'", name=name)
                return None

        sys.meta_path.insert(0, _BlockOpenTelemetry())
        for mod in list(sys.modules):
            if mod == "opentelemetry" or mod.startswith("opentelemetry."):
                del sys.modules[mod]

        import cayu
        from cayu import OpenTelemetryEventSink

        try:
            OpenTelemetryEventSink()
        except RuntimeError as exc:
            assert "cayu[otel]" in str(exc), str(exc)
        else:
            raise SystemExit("expected RuntimeError when opentelemetry is absent")

        print("import-ok")
        """
    )
    # Propagate the parent's import paths: pytest's `pythonpath=["src"]` is an
    # in-process sys.path insert that does NOT reach a subprocess.
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(p for p in sys.path if p)}
    result = subprocess.run(
        [sys.executable, "-c", program],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
        env=env,
    )
    assert result.returncode == 0, result.stderr
    assert "import-ok" in result.stdout


def test_span_timing_comes_from_event_timestamps_not_processing_time() -> None:
    # A buffered event bus / slow sink processes events well after the work happened.
    # Span start/end must reflect Event.timestamp, not wall-clock at emit() time, so
    # latency traces are not skewed. Drive events whose timestamps are minutes in the
    # past and assert the exported spans carry exactly those instants.
    from datetime import UTC, datetime, timedelta

    exporter, sink = _make_sink()
    base = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)

    def at(offset_s: float) -> datetime:
        return base + timedelta(seconds=offset_s)

    def ns(dt: datetime) -> int:
        return int(dt.timestamp() * 1_000_000_000)

    _drive(
        sink,
        [
            Event(type=EventType.SESSION_STARTED, session_id="s", payload={}, timestamp=at(0)),
            Event(
                type=EventType.MODEL_STARTED,
                session_id="s",
                payload={"model": "m"},
                timestamp=at(1),
            ),
            Event(type=EventType.MODEL_COMPLETED, session_id="s", payload={}, timestamp=at(3)),
            Event(
                type=EventType.TOOL_CALL_STARTED,
                session_id="s",
                tool_name="exec_command",
                payload={"tool_call_id": "c1"},
                timestamp=at(4),
            ),
            Event(
                type=EventType.TOOL_CALL_COMPLETED,
                session_id="s",
                payload={"tool_call_id": "c1"},
                timestamp=at(7),
            ),
            Event(type=EventType.SESSION_COMPLETED, session_id="s", payload={}, timestamp=at(10)),
        ],
    )
    spans = _spans_by_name(exporter)
    session = spans["cayu.session"]
    model = spans["chat m"]
    tool = spans["execute_tool exec_command"]
    assert session.start_time == ns(at(0))
    assert session.end_time == ns(at(10))
    assert model.start_time == ns(at(1))
    assert model.end_time == ns(at(3))
    assert tool.start_time == ns(at(4))
    assert tool.end_time == ns(at(7))


def test_staleness_eviction_evicts_least_recently_active_not_oldest_inserted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # With the cap reached, opening a new session must evict the *least-recently-active*
    # session — NOT the oldest inserted. A long-lived trace that is still progressing is
    # the oldest inserted yet must survive; a session that went idle is the one swept.
    from datetime import UTC, datetime, timedelta

    monkeypatch.setattr(otel, "_MAX_OPEN_SESSIONS", 2)
    exporter, sink = _make_sink()
    base = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)

    def at(offset_s: float) -> datetime:
        return base + timedelta(seconds=offset_s)

    def ns(dt: datetime) -> int:
        return int(dt.timestamp() * 1_000_000_000)

    _drive(
        sink,
        [
            # "long" started first (oldest inserted) but keeps working.
            Event(
                type=EventType.SESSION_STARTED,
                session_id="long",
                agent_name="long",
                payload={},
                timestamp=at(0),
            ),
            # "idle" started second, then never does anything else.
            Event(
                type=EventType.SESSION_STARTED,
                session_id="idle",
                agent_name="idle",
                payload={},
                timestamp=at(1),
            ),
            # "long" keeps progressing -> its last activity moves well past "idle".
            Event(
                type=EventType.MODEL_STARTED,
                session_id="long",
                payload={"model": "m"},
                timestamp=at(10),
            ),
            Event(type=EventType.MODEL_COMPLETED, session_id="long", payload={}, timestamp=at(11)),
        ],
    )
    assert sink.evicted_sessions == 0
    # Third session over the cap -> evict the most stale ("idle"), keeping "long".
    _drive(
        sink,
        [
            Event(
                type=EventType.SESSION_STARTED,
                session_id="new",
                agent_name="new",
                payload={},
                timestamp=at(20),
            )
        ],
    )
    assert set(sink._sessions) == {"long", "new"}
    assert sink.evicted_sessions == 1
    # "idle" was force-closed incomplete, ending at its last known activity (not now).
    idle = _spans_by_name(exporter)["cayu.session idle"]
    assert idle.attributes["cayu.incomplete"] is True
    assert idle.end_time == ns(at(1))
    # The runtime must surface metadata['traceparent'] on SESSION_STARTED so the sink
    # can root the span under the caller's trace.
    app = CayuApp(enable_logging=False)
    app.register_provider(
        FakeProvider([ModelStreamEvent.completed({"finish_reason": "stop"})]),
        default=True,
    )
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    async def collect() -> list[Event]:
        request = RunRequest(
            agent_name="assistant",
            session_id="sess_trace",
            messages=[Message.text("user", "hi")],
            metadata={"traceparent": REMOTE_TRACEPARENT},
        )
        return [event async for event in app.run(request)]

    events = asyncio.run(collect())
    started = next(e for e in events if e.type == EventType.SESSION_STARTED)
    assert started.payload["traceparent"] == REMOTE_TRACEPARENT

from __future__ import annotations

import importlib
from collections import OrderedDict
from types import ModuleType
from typing import Any

from cayu.core.events import Event, EventType
from cayu.runtime.event_sinks import EventSink

# GenAI attribute names, mirrored as plain strings from the OpenTelemetry GenAI
# semantic conventions so the sink does not import (or couple to the version of)
# `opentelemetry-semantic-conventions`. Update here if the conventions move.
GEN_AI_SYSTEM = "gen_ai.system"
GEN_AI_PROVIDER_NAME = "gen_ai.provider.name"
GEN_AI_OPERATION_NAME = "gen_ai.operation.name"
GEN_AI_REQUEST_MODEL = "gen_ai.request.model"
GEN_AI_RESPONSE_MODEL = "gen_ai.response.model"
GEN_AI_RESPONSE_FINISH_REASONS = "gen_ai.response.finish_reasons"
GEN_AI_USAGE_INPUT_TOKENS = "gen_ai.usage.input_tokens"
GEN_AI_USAGE_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"
GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS = "gen_ai.usage.cache_read.input_tokens"
GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS = "gen_ai.usage.cache_creation.input_tokens"
GEN_AI_USAGE_REASONING_OUTPUT_TOKENS = "gen_ai.usage.reasoning.output_tokens"
GEN_AI_TOOL_NAME = "gen_ai.tool.name"
GEN_AI_TOOL_CALL_ID = "gen_ai.tool.call.id"

# cayu-specific span attributes (no GenAI equivalent).
CAYU_SESSION_ID = "cayu.session.id"
CAYU_AGENT_NAME = "cayu.agent.name"
CAYU_ENVIRONMENT_NAME = "cayu.environment.name"
CAYU_MODEL_STEP = "cayu.model.step"
CAYU_TOOL_DENIED_BY = "cayu.tool.denied_by"
CAYU_TOOL_POLICY_DECISION = "cayu.tool.policy_decision"
# Marks a span force-closed without its own completion event (e.g. an in-flight
# model/tool span when the session was interrupted) so it is not mistaken for a
# clean success.
CAYU_INCOMPLETE = "cayu.incomplete"

DEFAULT_TRACER_NAME = "cayu"
# Safety net against unbounded growth if the runtime ever fails to emit a terminal
# event for a session (e.g. a bare task cancellation). The real fix is runtime-side.
_MAX_OPEN_SESSIONS = 10_000
# At-least-once event fan-out can replay a span event after another sink fails.
# Keep recent identities bounded while covering ordinary in-process retry bursts.
_MAX_RECENT_EVENT_IDENTITIES = 100_000

_OPERATION_CHAT = "chat"
_OPERATION_EXECUTE_TOOL = "execute_tool"

_SESSION_START_EVENTS = frozenset({EventType.SESSION_STARTED, EventType.SESSION_RESUMED})
_SESSION_END_EVENTS = frozenset(
    {
        EventType.SESSION_COMPLETED,
        EventType.SESSION_FAILED,
        EventType.SESSION_INTERRUPTED,
    }
)
# Tool-call events that terminate a tool span with a non-success outcome. Most close
# a preceding TOOL_CALL_STARTED span. A canonical policy BLOCKED event may instead be
# terminal-only when a planned denial is resumed beside an approved sibling; the sink
# synthesizes that denial observation rather than dropping it from tracing.
_TOOL_CALL_ERROR_EVENTS = frozenset(
    {
        EventType.TOOL_CALL_FAILED,
        EventType.TOOL_CALL_BLOCKED,
        EventType.TOOL_CALL_APPROVAL_DENIED,
    }
)
_TRACED_EVENT_TYPES = frozenset(
    {
        *_SESSION_START_EVENTS,
        *_SESSION_END_EVENTS,
        *_TOOL_CALL_ERROR_EVENTS,
        EventType.MODEL_STARTED,
        EventType.MODEL_COMPLETED,
        EventType.MODEL_ERROR,
        EventType.TOOL_CALL_STARTED,
        EventType.TOOL_CALL_COMPLETED,
    }
)


def _import_otel(module_name: str) -> ModuleType | Any:
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        missing = exc.name or ""
        # Only remap to the optional-dependency hint when the OpenTelemetry module
        # (or an ancestor of it) is absent. A deeper/unrelated missing module must
        # surface as its own error rather than be masked.
        if not (missing == module_name or module_name.startswith(f"{missing}.")):
            raise
        raise RuntimeError(
            "OpenTelemetryEventSink requires the optional OpenTelemetry packages. "
            "Install them with `pip install cayu[otel]`."
        ) from exc


class _SessionSpans:
    """Open spans for one session: the root, the in-flight model step, and tools.

    Model calls and tool calls run sequentially within a session, so a single
    ``model`` slot plus a ``tools`` dict keyed by ``tool_call_id`` is sufficient.
    Keeping spans per-session (rather than in one flat dict keyed by formatted
    strings) avoids cross-session key collisions and bounds session-end cleanup to
    that session's own spans.

    ``last_activity_ns`` records the event timestamp (epoch nanoseconds) of the most
    recent event that touched this session. The staleness safety net evicts the
    least-recently-active session, so a long-lived but still-progressing trace is
    never the one truncated.
    """

    def __init__(self, root: Any, last_activity_ns: int) -> None:
        self.root = root
        self.model: Any | None = None
        self.tools: dict[str, Any] = {}
        self.last_activity_ns = last_activity_ns


class OpenTelemetryEventSink(EventSink):
    """Convert cayu runtime events into OpenTelemetry spans.

    Produces a root span per session, a child span per model step and per tool
    call, with token usage in the GenAI semantic conventions. Spans are managed
    manually (started on ``*_STARTED`` events, ended on ``*_COMPLETED``/``*_FAILED``)
    because events arrive as discrete callbacks, not nested context-manager scopes.

    Notes:
        - Opt-in: register it via ``CayuApp(event_sinks=[...])``. Nothing is traced
          unless the sink is registered.
        - Propagation: the session root span is parented from ``parent_session_id``
          (in-process subagents) or a W3C ``traceparent`` carried in the session
          metadata (inbound HTTP). With no explicit parent the session starts a fresh
          trace root rather than inheriting any ambient span. For cross-process
          dispatched work, the caller injects :meth:`traceparent_for` into the child
          ``DispatchRequest.metadata`` (cayu does not inject it automatically).
        - Spans are closed by the matching terminal event. A session that ends without
          one (e.g. a bare task cancellation that re-raises without emitting a terminal
          ``session.*`` event) leaves its open spans unflushed; the sink relies on the
          runtime emitting a terminal event per session. As a bounded safety net, once
          more than ``_MAX_OPEN_SESSIONS`` are open a new session evicts the
          least-recently-active one (by event timestamp), incrementing
          :attr:`evicted_sessions`.
        - Timing: span ``start_time``/``end_time`` come from ``Event.timestamp`` (when
          the work happened), not from when the sink processes the event, so a buffered
          event bus or a slow downstream sink does not skew latency traces.
    """

    def __init__(
        self, *, tracer_name: str = DEFAULT_TRACER_NAME, tracer: Any | None = None
    ) -> None:
        if type(tracer_name) is not str or not tracer_name.strip():
            raise ValueError("OpenTelemetryEventSink tracer_name must be a non-empty string.")
        self._trace = _import_otel("opentelemetry.trace")
        status_module = _import_otel("opentelemetry.trace.status")
        self._status = status_module.Status
        self._status_error = status_module.StatusCode.ERROR
        self._empty_context = _import_otel("opentelemetry.context").Context()
        self._tracer = tracer if tracer is not None else self._trace.get_tracer(tracer_name)
        self._propagator_instance: Any | None = None
        self._sessions: dict[str, _SessionSpans] = {}
        self._recent_event_identities: OrderedDict[tuple[str, str], None] = OrderedDict()
        self._evicted_sessions = 0

    @property
    def evicted_sessions(self) -> int:
        """Number of sessions force-closed by the staleness safety net.

        A non-zero value means the runtime failed to emit a terminal ``session.*``
        event for that many sessions (their spans were orphaned and swept). Surface
        it in health checks: it points at a runtime-side event leak, not a bug here.
        """
        return self._evicted_sessions

    async def emit(self, event: Event) -> None:
        if type(event) is not Event:
            raise TypeError("OpenTelemetryEventSink requires Event instances.")
        event_type = event.type
        if event_type not in _TRACED_EVENT_TYPES:
            return
        identity = (event.session_id, event.id)
        if identity in self._recent_event_identities:
            self._recent_event_identities.move_to_end(identity)
            return
        applied = False
        if event_type in _SESSION_START_EVENTS:
            applied = self._start_session_span(event)
        elif event_type in _SESSION_END_EVENTS:
            # Only an outright failure marks the span ERROR; an interrupt is a pause
            # (approval wait / user stop), not a failure, so it stays UNSET.
            error = _error_text(event) if event_type == EventType.SESSION_FAILED else None
            applied = self._end_session_span(event, error=error)
        elif event_type == EventType.MODEL_STARTED:
            applied = self._start_model_span(event)
        elif event_type == EventType.MODEL_COMPLETED:
            applied = self._end_model_span(event, error=None)
        elif event_type == EventType.MODEL_ERROR:
            applied = self._end_model_span(event, error=_error_text(event))
        elif event_type == EventType.TOOL_CALL_STARTED:
            applied = self._start_tool_span(event)
        elif event_type == EventType.TOOL_CALL_COMPLETED:
            applied = self._end_tool_span(event, error=None)
        elif event_type in _TOOL_CALL_ERROR_EVENTS:
            applied = self._end_tool_span(event, error=_tool_error_text(event))
        if applied:
            self._recent_event_identities[identity] = None
            if len(self._recent_event_identities) > _MAX_RECENT_EVENT_IDENTITIES:
                self._recent_event_identities.popitem(last=False)

    def traceparent_for(self, session_id: str) -> str | None:
        """Return the W3C ``traceparent`` for an active session's root span.

        Inject this into a child ``DispatchRequest.metadata['traceparent']`` so
        dispatched work continues the same trace. Returns ``None`` if the session
        has no open span.
        """
        state = self._sessions.get(session_id)
        if state is None:
            return None
        carrier: dict[str, str] = {}
        context = self._trace.set_span_in_context(state.root)
        self._propagator().inject(carrier, context=context)
        return carrier.get("traceparent")

    def _start_session_span(self, event: Event) -> bool:
        session_id = event.session_id
        if session_id in self._sessions:
            # Idempotent: a RESUMED after STARTED, or a duplicate, keeps the first span.
            return True
        event_ns = _event_time_ns(event)
        if len(self._sessions) >= _MAX_OPEN_SESSIONS:
            # Safety net: a session whose terminal event never arrived would otherwise
            # grow _sessions unboundedly. Evict by *staleness* (least-recently-active),
            # not insertion order — a long-lived but still-progressing trace is the
            # oldest inserted yet must not be truncated mid-run. Close the orphan's
            # spans at its last known activity so their duration reflects real work,
            # and count it: a non-zero evicted_sessions points at a runtime event leak.
            stale_id = min(self._sessions, key=lambda sid: self._sessions[sid].last_activity_ns)
            stale = self._sessions.pop(stale_id)
            self._evicted_sessions += 1
            self._close_session_state(
                stale, error=None, root_incomplete=True, end_time=stale.last_activity_ns
            )
        name = _span_name("cayu.session", event.agent_name)
        span = self._tracer.start_span(
            name, context=self._resolve_parent_context(event), start_time=event_ns
        )
        span.set_attribute(CAYU_SESSION_ID, session_id)
        _set_str(span, CAYU_AGENT_NAME, event.agent_name)
        _set_str(span, CAYU_ENVIRONMENT_NAME, event.environment_name)
        self._sessions[session_id] = _SessionSpans(span, event_ns)
        return True

    def _end_session_span(self, event: Event, *, error: str | None) -> bool:
        state = self._sessions.pop(event.session_id, None)
        if state is None:
            return False
        self._close_session_state(state, error=error, end_time=_event_time_ns(event))
        return True

    def _close_session_state(
        self,
        state: _SessionSpans,
        *,
        error: str | None,
        root_incomplete: bool = False,
        end_time: int | None = None,
    ) -> None:
        # End any still-open child spans (a missing model/tool completion event, e.g.
        # an interrupt mid-step). They are left UNSET, not ERROR: we never received a
        # completion for them, so we don't know they failed — only the session root
        # carries the failure status. Mark them incomplete so they are not read as a
        # clean success. Close them at the same instant as the root (the terminal event
        # time, or the orphan's last activity) so a slow/buffered sink can't skew them.
        if state.model is not None:
            self._finish(state.model, error=None, incomplete=True, end_time=end_time)
        for tool_span in state.tools.values():
            self._finish(tool_span, error=None, incomplete=True, end_time=end_time)
        self._finish(state.root, error=error, incomplete=root_incomplete, end_time=end_time)

    def _start_model_span(self, event: Event) -> bool:
        state = self._sessions.get(event.session_id)
        if state is None:
            return False
        event_ns = _event_time_ns(event)
        state.last_activity_ns = event_ns
        if state.model is not None:
            # No completion arrived for the previous step (out-of-order/duplicate
            # MODEL_STARTED); close it rather than leak it.
            self._finish(state.model, error=None, incomplete=True, end_time=event_ns)
        payload = event.payload
        model = payload.get("model")
        name = _span_name(_OPERATION_CHAT, model)
        # Inference is an outbound call to the model provider — GenAI semconv marks it
        # a CLIENT span (session/tool spans stay INTERNAL).
        span = self._tracer.start_span(
            name,
            context=self._trace.set_span_in_context(state.root),
            kind=self._trace.SpanKind.CLIENT,
            start_time=event_ns,
        )
        span.set_attribute(GEN_AI_OPERATION_NAME, _OPERATION_CHAT)
        provider = payload.get("provider")
        _set_str(span, GEN_AI_SYSTEM, provider)
        _set_str(span, GEN_AI_PROVIDER_NAME, provider)  # gen_ai.system is deprecated
        _set_str(span, GEN_AI_REQUEST_MODEL, model)
        _set_int(span, CAYU_MODEL_STEP, payload.get("step"))
        state.model = span
        return True

    def _end_model_span(self, event: Event, *, error: str | None) -> bool:
        state = self._sessions.get(event.session_id)
        if state is None or state.model is None:
            return False
        event_ns = _event_time_ns(event)
        state.last_activity_ns = event_ns
        span = state.model
        state.model = None
        payload = event.payload
        # The normalized finish reason lives under "completion"; a top-level
        # "finish_reason" exists only for providers that happen to emit one.
        completion = payload.get("completion")
        if type(completion) is dict:
            finish_reason = completion.get("finish_reason")
            if type(finish_reason) is str and finish_reason:
                span.set_attribute(GEN_AI_RESPONSE_FINISH_REASONS, (finish_reason,))
        usage = payload.get("usage_metrics")
        if type(usage) is dict:
            _set_str(span, GEN_AI_RESPONSE_MODEL, usage.get("model") or payload.get("model"))
            _set_int(span, GEN_AI_USAGE_INPUT_TOKENS, usage.get("input_tokens"))
            _set_int(span, GEN_AI_USAGE_OUTPUT_TOKENS, usage.get("output_tokens"))
            _set_int(
                span, GEN_AI_USAGE_REASONING_OUTPUT_TOKENS, usage.get("reasoning_output_tokens")
            )
            cache = usage.get("cache")
            if type(cache) is dict:
                _set_int(span, GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS, cache.get("read_tokens"))
                _set_int(span, GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS, cache.get("write_tokens"))
        else:
            _set_str(span, GEN_AI_RESPONSE_MODEL, payload.get("model"))
        self._finish(span, error=error, end_time=event_ns)
        return True

    def _start_tool_span(self, event: Event) -> bool:
        state = self._sessions.get(event.session_id)
        if state is None:
            return False
        event_ns = _event_time_ns(event)
        state.last_activity_ns = event_ns
        tool_call_id = _tool_call_id(event)
        if tool_call_id in state.tools:
            self._finish(
                state.tools.pop(tool_call_id), error=None, incomplete=True, end_time=event_ns
            )
        state.tools[tool_call_id] = self._new_tool_span(
            state=state,
            event=event,
            start_time=event_ns,
        )
        return True

    def _new_tool_span(
        self,
        *,
        state: _SessionSpans,
        event: Event,
        start_time: int,
    ) -> Any:
        tool_name = event.tool_name
        name = _span_name(_OPERATION_EXECUTE_TOOL, tool_name)
        span = self._tracer.start_span(
            name, context=self._trace.set_span_in_context(state.root), start_time=start_time
        )
        span.set_attribute(GEN_AI_OPERATION_NAME, _OPERATION_EXECUTE_TOOL)
        _set_str(span, GEN_AI_TOOL_NAME, tool_name)
        _set_str(span, GEN_AI_TOOL_CALL_ID, _tool_call_id(event))
        return span

    def _end_tool_span(self, event: Event, *, error: str | None) -> bool:
        state = self._sessions.get(event.session_id)
        if state is None:
            return False
        event_ns = _event_time_ns(event)
        state.last_activity_ns = event_ns
        tool_call_id = _tool_call_id(event)
        span = state.tools.pop(tool_call_id, None)
        denied_by = event.payload.get("denied_by")
        if (
            span is None
            and event.type == EventType.TOOL_CALL_BLOCKED
            and tool_call_id
            and type(denied_by) is str
            and denied_by
        ):
            # Approval planning can persist a DENY beside a REQUIRE_APPROVAL call.
            # On resume the denied sibling emits only its canonical terminal event;
            # represent that authority decision as an instantaneous child span without
            # inventing a runtime TOOL_CALL_STARTED event that would affect accounting.
            span = self._new_tool_span(state=state, event=event, start_time=event_ns)
        if span is None:
            return False
        if event.type == EventType.TOOL_CALL_BLOCKED:
            _set_str(span, CAYU_TOOL_DENIED_BY, denied_by)
            _set_str(span, CAYU_TOOL_POLICY_DECISION, event.payload.get("decision"))
        self._finish(span, error=error, end_time=event_ns)
        return True

    def _resolve_parent_context(self, event: Event) -> Any:
        payload = event.payload
        parent_session_id = payload.get("parent_session_id")
        if type(parent_session_id) is str and parent_session_id:
            parent_state = self._sessions.get(parent_session_id)
            if parent_state is not None:
                return self._trace.set_span_in_context(parent_state.root)
        traceparent = payload.get("traceparent")
        if type(traceparent) is str and traceparent:
            carrier = {"traceparent": traceparent}
            tracestate = payload.get("tracestate")
            if type(tracestate) is str and tracestate:
                carrier["tracestate"] = tracestate
            # Extract onto an empty base so a malformed traceparent yields a fresh
            # root rather than inheriting any ambient span.
            return self._propagator().extract(carrier, context=self._empty_context)
        # No explicit parent: start a fresh trace root rather than inherit whatever
        # span happens to be active on the current OTel context.
        return self._empty_context

    def _finish(
        self, span: Any, *, error: str | None, incomplete: bool = False, end_time: int | None = None
    ) -> None:
        if error:
            span.set_status(self._status(self._status_error, error))
        if incomplete:
            span.set_attribute(CAYU_INCOMPLETE, True)
        # Pass the event's own timestamp (epoch ns) so a buffered/slow sink processing
        # the event late does not stretch the span past when the work actually ended.
        if end_time is None:
            span.end()
        else:
            span.end(end_time=end_time)

    def _propagator(self) -> Any:
        if self._propagator_instance is None:
            module = _import_otel("opentelemetry.trace.propagation.tracecontext")
            self._propagator_instance = module.TraceContextTextMapPropagator()
        return self._propagator_instance


def _event_time_ns(event: Event) -> int:
    """Convert an event's timestamp to epoch nanoseconds for OTel span timing.

    OTel ``start_span(start_time=...)`` and ``span.end(end_time=...)`` take epoch
    nanoseconds. Using the event's timestamp (set when the work happened) rather than
    ``time.time_ns()`` at sink-processing time keeps latency traces accurate even when
    the event bus buffers or a downstream sink is slow.
    """
    return int(event.timestamp.timestamp() * 1_000_000_000)


def _span_name(base: str, suffix: Any) -> str:
    return f"{base} {suffix}" if suffix else base


def _tool_call_id(event: Event) -> str:
    tool_call_id = event.payload.get("tool_call_id")
    return tool_call_id if type(tool_call_id) is str else ""


def _set_str(span: Any, key: str, value: Any) -> None:
    if type(value) is str and value:
        span.set_attribute(key, value)


def _set_int(span: Any, key: str, value: Any) -> None:
    if type(value) is int:
        span.set_attribute(key, value)


def _error_text(event: Event) -> str:
    payload = event.payload
    error = payload.get("error")
    error_type = payload.get("error_type")
    error = error if type(error) is str else ""
    error_type = error_type if type(error_type) is str else ""
    if error and error_type:
        return f"{error_type}: {error}"
    return error or error_type or str(event.type)


def _tool_error_text(event: Event) -> str:
    # Tool-call terminal events carry their message differently from model/session
    # errors: TOOL_CALL_FAILED in result.content, BLOCKED/APPROVAL_DENIED in "reason".
    payload = event.payload
    if event.type == EventType.TOOL_CALL_FAILED:
        result = payload.get("result")
        if type(result) is dict:
            content = result.get("content")
            if type(content) is str and content:
                return content
    reason = payload.get("reason")
    if type(reason) is str and reason:
        return reason
    return str(event.type)

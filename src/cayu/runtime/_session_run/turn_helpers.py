"""Pure helpers the session turn loop reads."""

from __future__ import annotations

from typing import Any

from cayu.budgets.usage import is_conversational_model_completion_payload
from cayu.context.structured_output import STRUCTURED_OUTPUT_TOOL_NAME
from cayu.events import Event, EventType
from cayu.runtime import _runtime_records as runtime_records
from cayu.runtime._model_step_executor import ModelStepFlowOutcome
from cayu.sessions.records import Session


def _session_trace_event_fields(
    session: Session,
    request_metadata: dict[str, Any],
) -> dict[str, Any]:
    # Surface the parent linkage and any inbound W3C trace context on the session
    # start event so an OpenTelemetryEventSink can parent the root span. Additive:
    # other sinks ignore these keys.
    fields: dict[str, Any] = {}
    if session.parent_session_id:
        fields["parent_session_id"] = session.parent_session_id
    traceparent = request_metadata.get("traceparent")
    if isinstance(traceparent, str) and traceparent:
        fields["traceparent"] = traceparent
        tracestate = request_metadata.get("tracestate")
        if isinstance(tracestate, str) and tracestate:
            fields["tracestate"] = tracestate
    return fields


def _knowledge_store(
    registered_environment: runtime_records.RegisteredEnvironment | None,
) -> Any:
    if registered_environment is None:
        return None
    return registered_environment.environment.knowledge_store


def _knowledge_access_scope(
    registered_environment: runtime_records.RegisteredEnvironment | None,
) -> Any:
    if registered_environment is None:
        return None
    return registered_environment.environment.knowledge_access_scope


def _user_tool_call_count(tool_calls: list[runtime_records.ToolCallRequest]) -> int:
    return sum(1 for tool_call in tool_calls if tool_call.name != STRUCTURED_OUTPUT_TOOL_NAME)


def _model_step_completion_published(
    item: tuple[Event | None, ModelStepFlowOutcome | None],
) -> bool:
    # Automatic context compaction publishes its own model.completed earlier in
    # the step; only the conversational completion ends the bounded model call.
    event, _ = item
    return (
        event is not None
        and event.type is EventType.MODEL_COMPLETED
        and is_conversational_model_completion_payload(event.payload)
    )

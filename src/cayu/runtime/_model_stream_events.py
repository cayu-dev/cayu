"""Shared model stream validation, transcript material and runtime event projection."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Any

from cayu._validation import (
    DurableValueError,
    canonical_durable_json_bytes,
    copy_durable_json_object,
    copy_durable_json_value,
    copy_json_value,
    extract_durable_value_error,
    require_clean_nonblank,
    require_durable_clean_nonblank,
    require_durable_text,
)
from cayu.budgets.billing import BillingIdentity
from cayu.budgets.usage import (
    durable_model_completed_payload,
    hosted_tool_usage_metrics_from_payload,
    normalize_usage_metrics,
    normalize_usage_metrics_with_overflow_error,
    usage_metrics_payload,
)
from cayu.context.base import ContextInputCoverage, ContextPressureEstimate
from cayu.events import Event, EventType, event_with_runtime_payload_authority
from cayu.messages import (
    CitationPart,
    CitationProvenance,
    HostedToolCallPart,
    Message,
    ThinkingPart,
    ToolCallPart,
    WebSearchAction,
)
from cayu.providers.base import (
    ModelCompletion,
    ModelFinishReason,
    ModelStreamEvent,
    ModelStreamEventType,
    ToolDiscoveryProjectionResult,
    copy_model_completion,
    copy_model_stream_event,
    normalize_model_completion,
)
from cayu.providers.operations import ProviderOperationRecoveryMetadata
from cayu.runtime import _runtime_records as runtime_records
from cayu.runtime import _transcript as transcript_helpers
from cayu.runtime._completion_projection import portable_model_completion_projection
from cayu.runtime._model_event_authority import _event_with_model_identity_authority
from cayu.runtime.execution_profiles import event_with_execution_profile_fingerprint_authority
from cayu.runtime.execution_units import (
    ModelAttemptIdentity,
    ToolRoundIdentity,
    copy_model_attempt_identity,
    copy_tool_round_identity,
    strip_runtime_owned_execution_identity,
)
from cayu.runtime.model_steps import (
    AssistantStepResult,
    assistant_text_content,
    provider_state_count,
    thinking_count,
)
from cayu.runtime.provider_operations import provider_operation_progress_event_id
from cayu.runtime.retry_policy import RetryDecision, retry_diagnostic_payload
from cayu.sessions.base import ModelCompletionStage, Session


@dataclass(frozen=True)
class _ModelStreamBoundaryValue:
    event: ModelStreamEvent
    completion_error: DurableValueError | None = None
    accounting_usage_metrics: dict[str, Any] | None = None
    accounting_usage_rejected: bool = False
    usage_normalization_failed: bool = False


@dataclass(frozen=True)
class _AssistantStreamBoundaryValue:
    event: ModelStreamEvent
    tool_call: runtime_records.ToolCallRequest | None = None
    tool_call_part: ToolCallPart | None = None


def _validate_stream_event(
    value: object,
    *,
    provider_name: str,
    requested_model: str,
    usage_dialect: str | None,
) -> _ModelStreamBoundaryValue:
    if type(value) is not ModelStreamEvent:
        raise TypeError("Model providers must yield ModelStreamEvent instances.")
    if type(value.type) is not ModelStreamEventType:
        raise ValueError("Model provider stream event type must be a ModelStreamEventType.")
    if value.type != ModelStreamEventType.COMPLETED:
        return _ModelStreamBoundaryValue(event=copy_model_stream_event(value))
    if type(value.delta) is not str:
        raise ValueError("Model provider stream event delta must be a string.")
    if type(value.payload) is not dict:
        raise ValueError("Model provider stream event payload must be an object.")

    completion_error: DurableValueError | None = None
    try:
        delta = require_durable_text(value.delta, "delta")
    except DurableValueError as exc:
        completion_error = exc
        delta = ""
    payload_was_projected = False
    try:
        payload = copy_durable_json_object(value.payload, "payload")
    except DurableValueError as exc:
        if completion_error is None:
            completion_error = exc
        payload_was_projected = True
        payload = portable_model_completion_projection(
            value.payload,
            provider_name=provider_name,
            requested_model=requested_model,
            usage_dialect=usage_dialect,
        )
    usage_normalization_failed = (
        payload_was_projected and payload.pop("usage_normalization_failed", None) is True
    )
    payload.pop("usage_unavailable_reason", None)

    # Raw usage makes runtime normalization authoritative. Preserve the legacy
    # normalized-only provider path when no raw payload exists, but never let a
    # provider-supplied projection override contradictory raw counters.
    has_raw_usage = payload.get("usage") is not None
    accounting_usage_metrics = payload.pop("usage_metrics", None)
    accounting_usage_rejected = False
    if has_raw_usage or type(accounting_usage_metrics) is not dict:
        accounting_usage_metrics = None
    if accounting_usage_metrics is None:
        resolved_model = _payload_model(payload, fallback=requested_model)
        try:
            projected_metrics = usage_metrics_payload(
                normalize_usage_metrics_with_overflow_error(
                    provider_name=provider_name,
                    model=resolved_model,
                    requested_model=requested_model,
                    raw_usage=payload.get("usage"),
                    usage_dialect=usage_dialect,
                )
            )
        except (TypeError, ValueError):
            # Normalization can combine independently valid counters into a
            # total or cache aggregate beyond the durable int64 domain. The
            # provider call has completed, so retain its raw portable usage
            # as rejection evidence and terminalize this attempt.
            if completion_error is None:
                completion_error = DurableValueError(
                    "integer_out_of_range",
                    "usage_metrics",
                )
            accounting_usage_rejected = True
            projected_metrics = None
        if projected_metrics is not None:
            try:
                accounting_usage_metrics = copy_durable_json_object(
                    projected_metrics,
                    "usage_metrics",
                )
            except DurableValueError as exc:
                # Derived counters can exceed the portable integer range even
                # when each raw counter is independently valid. Completion has
                # already happened, so fence the attempt as terminal while
                # retaining the portable raw usage evidence; never redispatch.
                if completion_error is None:
                    completion_error = exc
                accounting_usage_rejected = True

    try:
        completion = copy_model_completion(value.completion)
    except (TypeError, ValueError) as exc:
        if completion_error is None:
            completion_error = extract_durable_value_error(exc) or DurableValueError(
                "invalid_json_type",
                "completion",
            )
        completion = None
    if completion is None:
        try:
            completion = normalize_model_completion(payload)
        except (TypeError, ValueError) as exc:
            if completion_error is None:
                completion_error = extract_durable_value_error(exc) or DurableValueError(
                    "invalid_json_type",
                    "completion",
                )
            completion = ModelCompletion(finish_reason=ModelFinishReason.UNKNOWN)

    recovery_metadata = (
        None
        if value.recovery_metadata is None
        else ProviderOperationRecoveryMetadata.model_validate(
            value.recovery_metadata.model_dump(mode="python")
        )
    )
    try:
        tool_discovery_result = (
            None
            if value.tool_discovery_result is None
            else ToolDiscoveryProjectionResult.model_validate(
                value.tool_discovery_result.model_dump(mode="python")
            )
        )
    except (TypeError, ValueError) as exc:
        if completion_error is None:
            completion_error = extract_durable_value_error(exc) or DurableValueError(
                "invalid_json_type",
                "tool_discovery_result",
            )
        tool_discovery_result = None

    return _ModelStreamBoundaryValue(
        event=ModelStreamEvent.model_construct(
            type=ModelStreamEventType.COMPLETED,
            delta=delta,
            payload=payload,
            completion=completion,
            tool_discovery_result=tool_discovery_result,
            recovery_metadata=recovery_metadata,
        ),
        completion_error=completion_error,
        accounting_usage_metrics=accounting_usage_metrics,
        accounting_usage_rejected=accounting_usage_rejected,
        usage_normalization_failed=usage_normalization_failed,
    )


def _validate_assistant_stream_event(
    stream_event: ModelStreamEvent,
    *,
    generated_tool_call_id: str | None = None,
) -> _AssistantStreamBoundaryValue:
    """Validate transcript semantics before a reconnect cursor can advance."""

    if stream_event.type is ModelStreamEventType.TOOL_CALL:
        if stream_event.payload.get("id") is None and generated_tool_call_id is not None:
            payload = copy_durable_json_object(stream_event.payload, "payload")
            payload["id"] = generated_tool_call_id
            stream_event = copy_model_stream_event(
                stream_event.model_copy(update={"payload": payload})
            )
        tool_call = transcript_helpers.parse_tool_call(stream_event.payload)
        tool_call_part = transcript_helpers.tool_call_part(tool_call)
        if stream_event.payload.get("id") is None:
            payload = copy_durable_json_object(stream_event.payload, "payload")
            payload["id"] = tool_call.id
            stream_event = copy_model_stream_event(
                stream_event.model_copy(update={"payload": payload})
            )
        return _AssistantStreamBoundaryValue(
            event=stream_event,
            tool_call=tool_call,
            tool_call_part=tool_call_part,
        )
    if stream_event.type is ModelStreamEventType.THINKING:
        ThinkingPart(
            text=stream_event.delta,
            provider_state=stream_event.payload.get("provider_state"),
        )
    elif stream_event.type is ModelStreamEventType.HOSTED_TOOL_CALL:
        payload = _validated_hosted_tool_call_payload(stream_event.payload)
        stream_event = copy_model_stream_event(stream_event.model_copy(update={"payload": payload}))
    elif stream_event.type is ModelStreamEventType.CITATION:
        payload = _validated_citation_payload(stream_event.payload)
        stream_event = copy_model_stream_event(stream_event.model_copy(update={"payload": payload}))
    return _AssistantStreamBoundaryValue(event=stream_event)


def _validated_hosted_tool_call_payload(payload: dict[str, Any]) -> dict[str, Any]:
    copied = copy_durable_json_object(payload, "hosted_tool_call")
    if copied.get("tool_type") != "web_search":
        raise ValueError("Hosted tool stream events require tool_type='web_search'.")
    call_id = copied.get("call_id")
    if type(call_id) is not str:
        raise ValueError("Hosted tool stream events require a string call_id.")
    copied["call_id"] = require_durable_clean_nonblank(call_id, "call_id")
    status = copied.get("status")
    if status not in {
        "in_progress",
        "searching",
        "completed",
        "incomplete",
        "failed",
        "outcome_unknown",
    }:
        raise ValueError("Hosted tool stream events have an unsupported status.")
    action = copied.get("action")
    if action is not None:
        copied["action"] = WebSearchAction.model_validate(action).model_dump(mode="json")
    return copied


def _validated_citation_payload(payload: dict[str, Any]) -> dict[str, Any]:
    copied = copy_durable_json_object(payload, "citation")
    probe = CitationPart.model_validate(
        {
            **copied,
            "provenance": CitationProvenance(provider_name="provider-boundary"),
            "model_step_id": "mstep_00000000000000000000000000000000",
            "model_attempt_id": "matt_00000000000000000000000000000000",
        }
    )
    return {
        "citation_type": probe.citation_type,
        "url": probe.url,
        "title": probe.title,
        "start_index": probe.start_index,
        "end_index": probe.end_index,
    }


def _hosted_tool_call_part(
    stream_event: ModelStreamEvent,
    *,
    provider_name: str,
    model: str,
    model_attempt_identity: ModelAttemptIdentity,
) -> HostedToolCallPart | None:
    payload = _validated_hosted_tool_call_payload(stream_event.payload)
    status = payload["status"]
    if status not in {"completed", "incomplete", "failed", "outcome_unknown"}:
        return None
    return HostedToolCallPart(
        call_id=payload["call_id"],
        status=status,
        action=payload.get("action"),
        provider_name=provider_name,
        model=model,
        model_step_id=model_attempt_identity.model_step_id,
        model_attempt_id=model_attempt_identity.model_attempt_id,
    )


def _citation_part(
    stream_event: ModelStreamEvent,
    *,
    provider_name: str,
    model_attempt_identity: ModelAttemptIdentity,
    assistant_parts: list[transcript_helpers.AssistantContentPart],
) -> CitationPart:
    payload = _validated_citation_payload(stream_event.payload)
    assembled_text_length = sum(
        len(part.text)
        for part in assistant_parts
        if type(part) is transcript_helpers.AssistantTextPart
    )
    if payload["end_index"] is not None and payload["end_index"] > assembled_text_length:
        raise ValueError("Citation offsets exceed the associated assistant text.")
    return CitationPart(
        **payload,
        provenance=CitationProvenance(provider_name=provider_name),
        model_step_id=model_attempt_identity.model_step_id,
        model_attempt_id=model_attempt_identity.model_attempt_id,
    )


def _provider_operation_generated_tool_call_id(
    stage: ModelCompletionStage,
    stream_event: ModelStreamEvent,
) -> str | None:
    if stream_event.type is not ModelStreamEventType.TOOL_CALL:
        return None
    metadata = stream_event.recovery_metadata
    if metadata is None or metadata.cursor is None:
        return None
    return provider_operation_progress_event_id(stage.stage_id, metadata.cursor)


def _provider_operation_id(model_attempt_identity: ModelAttemptIdentity) -> str:
    """Return the runtime-owned provider-call identity for one model attempt."""

    material = canonical_durable_json_bytes(
        {
            "schema_version": 1,
            "model_attempt_id": model_attempt_identity.model_attempt_id,
        },
        "provider_operation_id",
    )
    return f"provider-operation:v1:{sha256(material).hexdigest()}"


def _model_stream_event_to_runtime_event(
    stream_event: ModelStreamEvent,
    *,
    session: Session,
    requested_model: str,
    registered_agent: runtime_records.RegisteredAgentState,
    environment_name: str | None,
    provider_name: str | None,
    step: int,
    attempt: int,
    max_attempts: int,
    model_attempt_identity: ModelAttemptIdentity,
    tool_round_identity: ToolRoundIdentity | None = None,
    classification: dict[str, str] | None = None,
    context_pressure_estimate: ContextPressureEstimate | None = None,
    transcript_cursor_after_completion: int | None = None,
    input_coverage: ContextInputCoverage | None = None,
    usage_dialect: str | None = None,
    billing_identity: BillingIdentity | None = None,
    accounting_usage_metrics: dict[str, Any] | None = None,
    accounting_usage_rejected: bool = False,
    usage_normalization_failed: bool = False,
    completion_diagnostics: dict[str, Any] | None = None,
    execution_profile_fingerprint: str | None = None,
    retry_decision: RetryDecision | None = None,
) -> Event:
    if type(stream_event) is not ModelStreamEvent:
        raise TypeError("Model stream events must be ModelStreamEvent instances.")
    if stream_event.type == ModelStreamEventType.TEXT_DELTA:
        event_type = EventType.MODEL_TEXT_DELTA
        payload = {"delta": stream_event.delta}
    elif stream_event.type == ModelStreamEventType.THINKING:
        event_type = EventType.MODEL_THINKING_DELTA
        payload = {"delta": stream_event.delta}
    elif stream_event.type == ModelStreamEventType.HOSTED_TOOL_CALL:
        event_type = EventType.MODEL_HOSTED_TOOL_CALL
        payload = {
            **_validated_hosted_tool_call_payload(stream_event.payload),
            "provider_name": provider_name,
            "model": requested_model,
            "provider_operation_id": _provider_operation_id(model_attempt_identity),
        }
    elif stream_event.type == ModelStreamEventType.CITATION:
        event_type = EventType.MODEL_CITATION
        payload = {
            **_validated_citation_payload(stream_event.payload),
            "model": requested_model,
            "provider_operation_id": _provider_operation_id(model_attempt_identity),
            "provenance": {
                "provider_name": provider_name,
                "hosted_tool": "web_search",
                "untrusted_external_evidence": True,
            },
        }
    elif stream_event.type == ModelStreamEventType.COMPLETED:
        payload = transcript_helpers.model_completed_event_payload(stream_event.payload)
        # When raw usage is present, its normalized projection and failure
        # marker are runtime-owned accounting evidence. Providers that expose
        # only the established normalized-usage payload retain compatibility.
        has_raw_usage = payload.get("usage") is not None
        raw_hosted_tool_usage = payload.get("hosted_tool_usage")
        if raw_hosted_tool_usage is not None:
            hosted_tool_usage = hosted_tool_usage_metrics_from_payload(payload)
            if hosted_tool_usage is None:
                payload.pop("hosted_tool_usage", None)
                payload["hosted_tool_usage_rejected"] = True
            else:
                payload["hosted_tool_usage"] = hosted_tool_usage.model_dump(mode="json")
        payload.pop("usage_metrics", None)
        payload.pop("usage_normalization_failed", None)
        payload.pop("usage_unavailable_reason", None)
        payload.pop("usage_metrics_rejected", None)
        payload.pop("rejected_usage_evidence", None)
        if accounting_usage_rejected:
            rejected_usage = payload.pop("usage", None)
            if rejected_usage is not None:
                payload["rejected_usage_evidence"] = copy_durable_json_value(
                    rejected_usage,
                    "rejected_usage_evidence",
                )
            payload["usage_metrics_rejected"] = True
        resolved_model = _payload_model(payload, fallback=requested_model)
        payload["model"] = resolved_model
        payload["requested_model"] = requested_model
        if provider_name is None:
            payload.pop("provider_name", None)
        else:
            # Provider attribution is runtime-owned. The provider-returned model
            # remains authoritative, but completion metadata cannot relabel the
            # commercial provider used by cost and diagnostic readers.
            payload["provider_name"] = provider_name
        # Billing identity is runtime-owned. Providers may report completion facts
        # consumed by their hook, but cannot inject an identity in the raw payload.
        payload.pop("billing_identity", None)
        if billing_identity is not None:
            payload["billing_identity"] = billing_identity.model_dump(mode="json")
        completion = _stream_event_completion(stream_event)
        completion_payload: dict[str, str | bool | None] = {
            "finish_reason": completion.finish_reason.value,
            "raw_finish_reason": completion.raw_finish_reason,
            "status": completion.status,
        }
        if completion.end_turn is not None:
            completion_payload["end_turn"] = completion.end_turn
        payload["completion"] = completion_payload
        if classification is not None:
            payload["step_classification"] = classification
        metrics = (
            copy_durable_json_object(accounting_usage_metrics, "usage_metrics")
            if accounting_usage_metrics is not None
            else None
            if accounting_usage_rejected
            else usage_metrics_payload(
                normalize_usage_metrics(
                    provider_name=provider_name,
                    model=resolved_model,
                    requested_model=requested_model,
                    raw_usage=payload.get("usage"),
                    usage_dialect=usage_dialect,
                    billing_identity=billing_identity,
                )
            )
        )
        if metrics is not None:
            # The event-level identity is authoritative. Keeping a second nested
            # copy would let an untrusted provider payload create conflicting
            # accounting evidence when normalized usage is unavailable.
            metrics.pop("billing_identity", None)
            payload["usage_metrics"] = metrics
        elif (has_raw_usage and not accounting_usage_rejected) or usage_normalization_failed:
            payload["usage_normalization_failed"] = True
        if context_pressure_estimate is not None:
            payload["context_pressure"] = {
                "estimated_tool_schema_input_tokens": (
                    context_pressure_estimate.estimated_tool_schema_input_tokens
                ),
                "estimated_structured_output_input_tokens": (
                    context_pressure_estimate.estimated_structured_output_input_tokens
                ),
                "estimated_request_options_input_tokens": (
                    context_pressure_estimate.estimated_request_options_input_tokens
                ),
                "estimated_request_overhead_input_tokens": (
                    context_pressure_estimate.estimated_request_overhead_input_tokens
                ),
            }
        if transcript_cursor_after_completion is not None:
            payload["transcript_cursor"] = transcript_cursor_after_completion
        # This is runtime-owned evidence, never a provider-supplied anchor.
        payload.pop("input_coverage", None)
        if input_coverage is not None:
            payload["input_coverage"] = input_coverage.model_dump(mode="json")
        if completion_diagnostics:
            payload.update(
                copy_durable_json_object(
                    completion_diagnostics,
                    "completion_diagnostics",
                )
            )
        event_type = EventType.MODEL_COMPLETED
    elif stream_event.type == ModelStreamEventType.ERROR:
        event_type = EventType.MODEL_ERROR
        payload = copy_json_value(stream_event.payload, "payload")
    else:
        raise ValueError(f"Unsupported model stream event type: {stream_event.type}")
    payload = _retry_attempt_payload(
        payload,
        execution_provider_name=provider_name if event_type is EventType.MODEL_ERROR else None,
        requested_model=requested_model if event_type is EventType.MODEL_ERROR else None,
        step=step,
        attempt=attempt,
        max_attempts=max_attempts,
        model_attempt_identity=model_attempt_identity,
        decision=retry_decision,
    )
    if tool_round_identity is not None:
        payload.update(copy_tool_round_identity(tool_round_identity).payload())
    if event_type == EventType.MODEL_COMPLETED:
        payload = durable_model_completed_payload(
            payload,
            fallback_fields={
                "provider_name": provider_name,
                "requested_model": requested_model,
                "model": requested_model,
                "step": step,
                "attempt": attempt,
                "max_attempts": max_attempts,
                **model_attempt_identity.payload(),
                **(
                    {}
                    if tool_round_identity is None
                    else copy_tool_round_identity(tool_round_identity).payload()
                ),
            },
            unavailable_reason="invalid model completion usage telemetry",
        )
    event = _event_with_model_identity_authority(
        Event(
            type=event_type,
            session_id=session.id,
            agent_name=registered_agent.spec.name,
            environment_name=environment_name,
            payload=payload,
        ),
        model_attempt_identity,
    )
    if tool_round_identity is not None and (
        event.payload.get("tool_round_id") == tool_round_identity.tool_round_id
    ):
        event = event_with_runtime_payload_authority(event, "tool_round_id")
    if event_type in {EventType.MODEL_HOSTED_TOOL_CALL, EventType.MODEL_CITATION}:
        event = event_with_runtime_payload_authority(event, "provider_operation_id")
    return event_with_execution_profile_fingerprint_authority(
        event,
        execution_profile_fingerprint,
    )


def _stream_event_completion(stream_event: ModelStreamEvent) -> ModelCompletion:
    if type(stream_event) is not ModelStreamEvent:
        raise TypeError("Model stream events must be ModelStreamEvent instances.")
    if stream_event.type != ModelStreamEventType.COMPLETED:
        raise ValueError("Only completed model stream events have completion metadata.")
    if stream_event.completion is not None:
        return stream_event.completion
    return normalize_model_completion(stream_event.payload)


def _assistant_step_result(
    *,
    session_id: str,
    step: int,
    model_attempt_identity: ModelAttemptIdentity,
    assistant_message: Message | None,
    tool_calls: list[runtime_records.ToolCallRequest],
    completion: ModelCompletion,
) -> AssistantStepResult:
    model_attempt_identity = copy_model_attempt_identity(model_attempt_identity)
    tool_round_identity = model_attempt_identity.new_tool_round() if tool_calls else None
    if assistant_message is not None and tool_round_identity is not None:
        assistant_message = transcript_helpers.assistant_message_with_tool_round(
            assistant_message,
            tool_round_identity,
        )
    text_content = assistant_text_content(assistant_message)
    return AssistantStepResult(
        session_id=session_id,
        step=step,
        model_step_id=model_attempt_identity.model_step_id,
        model_attempt_id=model_attempt_identity.model_attempt_id,
        tool_round_identity=tool_round_identity,
        assistant_message=assistant_message,
        tool_calls=list(tool_calls),
        completion=completion,
        text_content=text_content,
        has_user_visible_content=bool(text_content.strip()),
        provider_state_count=provider_state_count(assistant_message),
        thinking_count=thinking_count(assistant_message),
    )


def _require_unique_tool_call_ids(
    tool_calls: list[runtime_records.ToolCallRequest],
) -> None:
    tool_call_ids = [tool_call.id for tool_call in tool_calls]
    if len(tool_call_ids) != len(set(tool_call_ids)):
        raise ValueError("Model provider emitted duplicate tool-call identifiers.")


def _retry_attempt_payload(
    payload: dict[str, Any],
    *,
    execution_provider_name: str | None = None,
    requested_model: str | None = None,
    step: int,
    attempt: int,
    max_attempts: int,
    model_attempt_identity: ModelAttemptIdentity,
    decision: RetryDecision | None = None,
) -> dict[str, Any]:
    enriched = dict(payload)
    for key in ("retry", "retry_disposition", "retry_suppression", "provider_retryable"):
        enriched.pop(key, None)
    strip_runtime_owned_execution_identity(enriched)
    enriched["step"] = step
    enriched["attempt"] = attempt
    enriched["max_attempts"] = max_attempts
    if execution_provider_name is not None:
        enriched["provider_name"] = require_clean_nonblank(
            execution_provider_name, "execution_provider_name"
        )
    if requested_model is not None:
        enriched["requested_model"] = require_clean_nonblank(requested_model, "requested_model")
    if decision is not None:
        if type(decision) is not RetryDecision:
            raise TypeError("decision must be a RetryDecision or None.")
        if decision.attempt != attempt or decision.max_attempts != max_attempts:
            raise ValueError("Retry decision does not match the model-attempt evidence.")
        enriched.pop("effective_max_attempts", None)
        enriched.pop("reason", None)
        enriched.update(retry_diagnostic_payload(decision))
        enriched["effective_max_attempts"] = decision.effective_max_attempts
        if decision.reason is not None:
            enriched["reason"] = decision.reason.value
    enriched.update(copy_model_attempt_identity(model_attempt_identity).payload())
    return enriched


def _payload_model(payload: dict[str, Any], *, fallback: str) -> str:
    model = payload.get("model")
    if type(model) is str and model.strip():
        return model
    return fallback

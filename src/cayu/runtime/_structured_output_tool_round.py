"""Pure construction helpers for durable structured-output tool rounds.

The module is intentionally independent of session orchestration and recovery
ownership. Live execution and crash recovery can therefore validate the same
reserved tool call, construct the same outcomes, and reproduce byte-identical
terminal and auxiliary event identities.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

from cayu._validation import canonical_durable_json_bytes, copy_json_value
from cayu.core.events import (
    Event,
    EventType,
    copy_event,
    event_with_runtime_generated_id,
    event_with_runtime_payload_authority,
)
from cayu.core.tools import ToolResult
from cayu.runtime import _runtime_records as runtime_records
from cayu.runtime import _tool_argument_publication as tool_argument_publication
from cayu.runtime import _tool_execution as tool_execution
from cayu.runtime import _tool_round_recovery as tool_round_recovery
from cayu.runtime.execution_units import (
    ModelAttemptIdentity,
    ToolRoundIdentity,
    copy_model_attempt_identity,
    copy_tool_round_identity,
)
from cayu.runtime.sessions import RuntimePublicationRequest, Session
from cayu.runtime.structured_output import (
    STRUCTURED_OUTPUT_TOOL_NAME,
    StructuredOutputError,
    StructuredOutputSpec,
    StructuredOutputValidation,
    structured_output_repair_lead,
    validate_structured_output_tool_arguments,
)
from cayu.vaults import SecretRedactor


@dataclass(frozen=True, slots=True)
class _StructuredOutputToolRoundPublicationExtension:
    """Narrow auxiliary material appended with one structured tool result."""

    intent: dict[str, Any]
    events: tuple[Event, ...]

    def build_request(
        self,
        *,
        ordinary_request: RuntimePublicationRequest,
        pending_round: tool_round_recovery.PendingToolRound,
    ) -> RuntimePublicationRequest:
        if not any(
            call.tool_name == STRUCTURED_OUTPUT_TOOL_NAME for call in pending_round.tool_calls
        ):
            raise ValueError("Structured-output publication requires the reserved finalizer tool.")
        return RuntimePublicationRequest(
            publication_id=ordinary_request.publication_id,
            kind=ordinary_request.kind,
            interaction_id=ordinary_request.interaction_id,
            intent={
                **ordinary_request.intent,
                "auxiliary": copy_json_value(
                    self.intent,
                    "structured_output_publication_intent",
                ),
            },
            mutation=ordinary_request.mutation,
            transcript_messages=ordinary_request.transcript_messages,
            events=tuple(copy_event(event) for event in self.events),
            referenced_events=ordinary_request.referenced_events,
        )


def _has_structured_output_tool_call(
    tool_calls: list[runtime_records.ToolCallRequest],
) -> bool:
    return any(tool_call.name == STRUCTURED_OUTPUT_TOOL_NAME for tool_call in tool_calls)


def _validate_structured_output_tool_round(
    *,
    tool_calls: list[runtime_records.ToolCallRequest],
    spec: StructuredOutputSpec,
) -> StructuredOutputValidation:
    internal_calls = [
        tool_call for tool_call in tool_calls if tool_call.name == STRUCTURED_OUTPUT_TOOL_NAME
    ]
    if len(internal_calls) != 1:
        return _structured_output_tool_error(
            "Call the structured-output tool exactly once when submitting final output."
        )
    if len(tool_calls) != 1:
        return _structured_output_tool_error(
            "Call the structured-output tool by itself, not in the same tool round as other tools."
        )
    return validate_structured_output_tool_arguments(internal_calls[0].arguments, spec)


def _structured_output_tool_round_outcomes(
    *,
    tool_calls: list[runtime_records.ToolCallRequest],
    spec: StructuredOutputSpec,
    validation: StructuredOutputValidation,
) -> list[runtime_records.ToolCallOutcome]:
    if validation.valid:
        return [
            runtime_records.ToolCallOutcome(
                call=tool_calls[0],
                result=ToolResult(
                    content="Structured output accepted.",
                    structured={"output": copy_json_value(validation.output, "output")},
                ),
            )
        ]

    repair_lead = structured_output_repair_lead(spec)
    error_summary = _structured_output_error_summary(validation)
    outcomes: list[runtime_records.ToolCallOutcome] = []
    for tool_call in tool_calls:
        if tool_call.name == STRUCTURED_OUTPUT_TOOL_NAME:
            content = f"Structured output rejected: {error_summary}\n\n{repair_lead}"
        else:
            content = (
                "Tool was not executed because the structured-output finalizer was "
                "called in the same tool round. Retry the needed work before submitting "
                "final structured output."
            )
        outcomes.append(
            runtime_records.ToolCallOutcome(
                call=tool_call,
                result=ToolResult(
                    content=content,
                    structured={
                        "structured_output_errors": [
                            error.model_dump(mode="json") for error in validation.errors
                        ],
                    },
                    is_error=True,
                ),
            )
        )
    return outcomes


def _structured_output_tool_terminal_event(
    *,
    session: Session,
    registered_agent: runtime_records.RegisteredAgentState,
    environment_name: str | None,
    tool_round_identity: ToolRoundIdentity,
    outcome: runtime_records.ToolCallOutcome,
) -> Event:
    tool_round_identity = copy_tool_round_identity(tool_round_identity)
    tool_round_id = tool_round_identity.tool_round_id
    event_type = (
        EventType.TOOL_CALL_FAILED if outcome.result.is_error else EventType.TOOL_CALL_COMPLETED
    )
    result = outcome.result.model_dump(mode="json")
    event_id = (
        "structured-tool-terminal:v1:"
        + hashlib.sha256(
            canonical_durable_json_bytes(
                {
                    "session_id": session.id,
                    "tool_round_id": tool_round_id,
                    "tool_call_id": outcome.call.id,
                    "tool_name": outcome.call.name,
                    "event_type": event_type.value,
                    "result": result,
                },
                "structured_output_tool_terminal_identity",
            )
        ).hexdigest()
    )
    return event_with_runtime_generated_id(
        event_with_runtime_payload_authority(
            Event(
                id=event_id,
                type=event_type,
                session_id=session.id,
                agent_name=registered_agent.spec.name,
                environment_name=environment_name,
                tool_name=outcome.call.name,
                payload={
                    **tool_round_identity.payload(),
                    "tool_call_id": outcome.call.id,
                    "idempotency_key": tool_execution.tool_idempotency_key(
                        session_id=session.id,
                        tool_round_id=tool_round_id,
                        tool_call_id=outcome.call.id,
                    ),
                    **tool_argument_publication.unavailable_argument_projection().payload_fields(),
                    "structured_output_validation": True,
                    "result": result,
                },
            ),
            *tool_round_identity.payload(),
        )
    )


def _structured_output_tool_error(message: str) -> StructuredOutputValidation:
    return StructuredOutputValidation(
        valid=False,
        errors=[
            StructuredOutputError(
                path="$",
                message=message,
                schema_path="$",
            )
        ],
    )


def _structured_output_error_summary(validation: StructuredOutputValidation) -> str:
    if not validation.errors:
        return "unknown validation error."
    return "; ".join(f"{error.path}: {error.message}" for error in validation.errors[:3])


def _structured_output_event(
    *,
    event_type: EventType,
    session: Session,
    registered_agent: runtime_records.RegisteredAgentState,
    environment_name: str | None,
    spec: StructuredOutputSpec,
    validation: StructuredOutputValidation,
    step: int,
    attempt: int,
    redactor: SecretRedactor | None = None,
    execution_identity: ModelAttemptIdentity | ToolRoundIdentity | None = None,
    tool_round_identity: ToolRoundIdentity | None = None,
) -> Event:
    if event_type not in {
        EventType.STRUCTURED_OUTPUT_VALIDATED,
        EventType.STRUCTURED_OUTPUT_FAILED,
        EventType.STRUCTURED_OUTPUT_RETRY,
    }:
        raise ValueError(f"Unsupported structured output event type: {event_type}")
    if type(spec) is not StructuredOutputSpec:
        raise TypeError("Structured output spec must be a StructuredOutputSpec instance.")
    if type(validation) is not StructuredOutputValidation:
        raise TypeError("Structured output validation must be a StructuredOutputValidation.")
    validation = _redact_structured_output_validation(validation, redactor)
    payload: dict[str, Any] = {
        "name": spec.name,
        "step": step,
        "attempt": attempt,
        "max_retries": spec.max_retries,
        "valid": validation.valid,
        "errors": [error.model_dump(mode="json") for error in validation.errors],
    }
    if validation.valid:
        payload["output"] = copy_json_value(validation.output, "output")
    if execution_identity is not None and tool_round_identity is not None:
        raise ValueError("Structured-output event received conflicting execution identities.")
    identity = tool_round_identity or execution_identity
    tool_round_id = None
    if type(identity) is ToolRoundIdentity:
        copied_tool_round_identity = copy_tool_round_identity(identity)
        tool_round_id = copied_tool_round_identity.tool_round_id
        payload.update(copied_tool_round_identity.payload())
    elif type(identity) is ModelAttemptIdentity:
        payload.update(copy_model_attempt_identity(identity).payload())
    elif identity is not None:
        raise TypeError("Structured-output execution identity has an unsupported type.")
    event_fields: dict[str, Any] = {}
    if tool_round_id is not None:
        event_fields["id"] = _structured_output_round_event_id(
            session_id=session.id,
            tool_round_id=tool_round_id,
            event_type=event_type,
            step=step,
            attempt=attempt,
        )
    event = Event(
        **event_fields,
        type=event_type,
        session_id=session.id,
        agent_name=registered_agent.spec.name,
        environment_name=environment_name,
        payload=payload,
    )
    if identity is not None:
        event = event_with_runtime_payload_authority(event, *identity.payload())
    return event_with_runtime_generated_id(event) if tool_round_id is not None else event


def _structured_output_validating_event(
    *,
    session: Session,
    registered_agent: runtime_records.RegisteredAgentState,
    environment_name: str | None,
    spec: StructuredOutputSpec,
    step: int,
    attempt: int,
    execution_identity: ModelAttemptIdentity | ToolRoundIdentity | None = None,
    tool_round_identity: ToolRoundIdentity | None = None,
) -> Event:
    if type(spec) is not StructuredOutputSpec:
        raise TypeError("Structured output spec must be a StructuredOutputSpec instance.")
    payload = {
        "name": spec.name,
        "strategy": spec.strategy.value,
        "step": step,
        "attempt": attempt,
        "max_retries": spec.max_retries,
    }
    if execution_identity is not None and tool_round_identity is not None:
        raise ValueError("Structured-output event received conflicting execution identities.")
    identity = tool_round_identity or execution_identity
    tool_round_id = None
    if type(identity) is ToolRoundIdentity:
        copied_tool_round_identity = copy_tool_round_identity(identity)
        tool_round_id = copied_tool_round_identity.tool_round_id
        payload.update(copied_tool_round_identity.payload())
    elif type(identity) is ModelAttemptIdentity:
        payload.update(copy_model_attempt_identity(identity).payload())
    elif identity is not None:
        raise TypeError("Structured-output execution identity has an unsupported type.")
    event_fields: dict[str, Any] = {}
    if tool_round_id is not None:
        event_fields["id"] = _structured_output_round_event_id(
            session_id=session.id,
            tool_round_id=tool_round_id,
            event_type=EventType.STRUCTURED_OUTPUT_VALIDATING,
            step=step,
            attempt=attempt,
        )
    event = Event(
        **event_fields,
        type=EventType.STRUCTURED_OUTPUT_VALIDATING,
        session_id=session.id,
        agent_name=registered_agent.spec.name,
        environment_name=environment_name,
        payload=payload,
    )
    if identity is not None:
        event = event_with_runtime_payload_authority(event, *identity.payload())
    return event_with_runtime_generated_id(event) if tool_round_id is not None else event


def _structured_output_round_event_id(
    *,
    session_id: str,
    tool_round_id: str,
    event_type: EventType,
    step: int,
    attempt: int,
) -> str:
    identity = canonical_durable_json_bytes(
        {
            "session_id": session_id,
            "tool_round_id": tool_round_id,
            "event_type": event_type.value,
            "step": step,
            "attempt": attempt,
        },
        "structured_output_round_event_identity",
    )
    return f"structured-output:v1:{hashlib.sha256(identity).hexdigest()}"


def _redact_structured_output_validation(
    validation: StructuredOutputValidation,
    redactor: SecretRedactor | None,
) -> StructuredOutputValidation:
    if type(validation) is not StructuredOutputValidation:
        raise TypeError("Structured output validation must be a StructuredOutputValidation.")
    if redactor is None or not redactor.has_values:
        return validation.model_copy(deep=True)
    return StructuredOutputValidation(
        valid=validation.valid,
        output=redactor.redact_json(validation.output),
        errors=[
            StructuredOutputError(
                path=redactor.redact_text(error.path),
                message=redactor.redact_text(error.message),
                schema_path=redactor.redact_text(error.schema_path),
            )
            for error in validation.errors
        ],
    )

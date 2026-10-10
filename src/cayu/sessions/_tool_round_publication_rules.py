"""Exact tool-round publication, structured-output evidence and checkpoint validation."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from hashlib import sha256
from typing import Any

from cayu._validation import (
    MAX_DURABLE_JSON_INTEGER,
    canonical_durable_json_bytes,
    copy_durable_json_value,
)
from cayu._validation import require_durable_clean_nonblank as require_clean_nonblank
from cayu.context.structured_output import (
    STRUCTURED_OUTPUT_TOOL_NAME,
    StructuredOutputError,
    StructuredOutputSpec,
    StructuredOutputStrategy,
    StructuredOutputValidation,
)
from cayu.events import Event, EventType
from cayu.execution_units import ToolRoundIdentity, copy_tool_round_identity
from cayu.messages import MessageRole, ToolCallPart, ToolResultPart
from cayu.sessions import authority as session_authority_rules
from cayu.sessions._checkpoint_mutations import (
    SessionRuntimePublicationConflict,
    runtime_publication_checkpoint_value_digest,
)
from cayu.sessions._execution_profile_checkpoint import (
    active_invocation_execution_profile_from_checkpoint,
)
from cayu.sessions._runtime_publication import RuntimePublicationRequest
from cayu.tools.grants import TARGETED_TOOL_TRANSCRIPT_REFERENCE

_PENDING_TOOL_ROUND_CHECKPOINT_KEY = "pending_tool_round"


_TOOL_ROUND_TERMINAL_EVENT_TYPES = frozenset(
    {
        EventType.TOOL_CALL_COMPLETED,
        EventType.TOOL_CALL_FAILED,
        EventType.TOOL_CALL_BLOCKED,
        EventType.TOOL_CALL_APPROVAL_DENIED,
    }
)


_TOOL_ROUND_LIFECYCLE_EVENT_TYPES = frozenset(
    {EventType.TOOL_CALL_STARTED, *_TOOL_ROUND_TERMINAL_EVENT_TYPES}
)


_STRUCTURED_OUTPUT_AUXILIARY_SCHEMA_VERSION = 1


_STRUCTURED_OUTPUT_AUXILIARY_KIND = "structured-output-validation"


_STRUCTURED_OUTPUT_AUXILIARY_INTENT_KEYS = frozenset(
    {
        "schema_version",
        "kind",
        "step",
        "attempt",
        "valid",
        "retry_scheduled",
        "event_ids",
    }
)


@dataclass(frozen=True, slots=True)
class _StructuredOutputToolRoundAuxiliary:
    step: int
    attempt: int
    valid: bool
    retry_scheduled: bool
    event_ids: tuple[str, ...]
    session_id: str
    output_present: bool
    output: Any
    errors: tuple[dict[str, Any], ...]


def _validate_tool_round_call_ids(
    value: Any,
    field_name: str,
) -> tuple[str, ...]:
    if type(value) not in (list, tuple):
        raise TypeError(f"{field_name} must be a list or tuple.")
    tool_call_ids = tuple(
        require_clean_nonblank(tool_call_id, f"{field_name} item") for tool_call_id in value
    )
    if not tool_call_ids:
        raise ValueError(f"{field_name} must contain at least one value.")
    if len(set(tool_call_ids)) != len(tool_call_ids):
        raise ValueError(f"{field_name} cannot repeat tool call ids.")
    return tool_call_ids


def _tool_round_event_matches_scope_or_is_ambiguous(
    event: Event,
    tool_round_identity: ToolRoundIdentity,
) -> bool:
    """Select exact or contradictory evidence while excluding a valid prior round.

    Only a complete valid identity whose round and originating model attempt
    both differ is safely stale. Missing, malformed, or partially conflicting
    identity remains visible so the publication transaction fails closed.
    """

    target = copy_tool_round_identity(tool_round_identity)
    try:
        event_identity = ToolRoundIdentity.model_validate(
            {
                "model_step_id": event.payload.get("model_step_id"),
                "model_attempt_id": event.payload.get("model_attempt_id"),
                "tool_round_id": event.payload.get("tool_round_id"),
            }
        )
    except (TypeError, ValueError):
        return True
    return not (
        event_identity.tool_round_id != target.tool_round_id
        and (
            event_identity.model_step_id,
            event_identity.model_attempt_id,
        )
        != (
            target.model_step_id,
            target.model_attempt_id,
        )
    )


def _tool_lifecycle_publication_identity(
    request: RuntimePublicationRequest,
) -> tuple[ToolRoundIdentity, tuple[str, ...]] | None:
    if request.kind not in {"tool-round", "user-input-close"}:
        return None

    round_id_field = "round_id" if request.kind == "tool-round" else "tool_round_id"
    round_id = request.intent.get(round_id_field)
    raw_tool_call_ids = request.intent.get("tool_call_ids")
    if type(round_id) is not str or not round_id.strip():
        raise ValueError(
            f"A {request.kind} publication requires a nonblank intent.{round_id_field}."
        )
    if request.kind == "tool-round" and request.publication_id != f"tool-round:{round_id}":
        raise ValueError("A tool-round publication_id must be derived from its intent.round_id.")
    model_step_id = request.intent.get("model_step_id")
    model_attempt_id = request.intent.get("model_attempt_id")
    tool_round_id = request.intent.get("tool_round_id")
    if (
        type(model_step_id) is not str
        or type(model_attempt_id) is not str
        or type(tool_round_id) is not str
    ):
        raise ValueError(f"A {request.kind} publication requires a complete execution identity.")
    try:
        execution_identity = ToolRoundIdentity(
            model_step_id=model_step_id,
            model_attempt_id=model_attempt_id,
            tool_round_id=tool_round_id,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"A {request.kind} publication execution identity is malformed.") from exc
    if execution_identity.tool_round_id != round_id:
        raise ValueError(
            f"A {request.kind} publication intent.{round_id_field} conflicts with its "
            "execution identity."
        )
    if type(raw_tool_call_ids) is not list:
        raise ValueError(f"A {request.kind} publication requires intent.tool_call_ids as a list.")
    tool_call_ids = _validate_tool_round_call_ids(
        raw_tool_call_ids,
        "intent.tool_call_ids",
    )
    return execution_identity, tool_call_ids


def _tool_round_publication_identity(
    request: RuntimePublicationRequest,
) -> tuple[ToolRoundIdentity, tuple[str, ...]] | None:
    """Return identity only for publication of a pending tool-round marker."""

    if request.kind != "tool-round":
        return None
    return _tool_lifecycle_publication_identity(request)


def _validate_structured_output_auxiliary_integer(
    value: Any,
    field_name: str,
) -> int:
    if type(value) is not int:
        raise TypeError(f"{field_name} must be an integer.")
    if not 1 <= value <= MAX_DURABLE_JSON_INTEGER:
        raise ValueError(f"{field_name} must be between 1 and {MAX_DURABLE_JSON_INTEGER}.")
    return value


def _validate_durable_structured_output_validation(
    value: Any,
    field_name: str,
) -> StructuredOutputValidation:
    if (
        type(value) is not dict
        or set(value) != {"valid", "output", "errors"}
        or type(value.get("valid")) is not bool
        or type(value.get("errors")) is not list
    ):
        raise ValueError(f"{field_name} must be a structured-output validation object.")
    try:
        validation = StructuredOutputValidation.model_validate(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} is malformed.") from exc
    if validation.valid:
        if validation.errors:
            raise ValueError(f"{field_name} cannot contain errors for a valid output.")
    elif validation.output is not None or not validation.errors:
        raise ValueError(f"{field_name} must contain errors and no output for an invalid result.")
    return validation


def _validate_structured_output_tool_round_auxiliary(
    request: RuntimePublicationRequest,
    *,
    round_id: str,
) -> _StructuredOutputToolRoundAuxiliary | None:
    if "auxiliary" not in request.intent:
        if request.events:
            raise ValueError(
                "A tool-round publication cannot append events without a validated "
                "intent.auxiliary contract."
            )
        return None

    raw_auxiliary = request.intent["auxiliary"]
    if type(raw_auxiliary) is not dict:
        raise TypeError("A tool-round intent.auxiliary value must be an object.")
    if set(raw_auxiliary) != _STRUCTURED_OUTPUT_AUXILIARY_INTENT_KEYS:
        raise ValueError(
            "A structured-output tool-round intent.auxiliary value has invalid fields."
        )
    if (
        type(raw_auxiliary["schema_version"]) is not int
        or raw_auxiliary["schema_version"] != _STRUCTURED_OUTPUT_AUXILIARY_SCHEMA_VERSION
    ):
        raise ValueError(
            "A structured-output tool-round intent.auxiliary schema_version must be 1."
        )
    if raw_auxiliary["kind"] != _STRUCTURED_OUTPUT_AUXILIARY_KIND:
        raise ValueError("A structured-output tool-round intent.auxiliary kind is invalid.")
    step = _validate_structured_output_auxiliary_integer(
        raw_auxiliary["step"],
        "intent.auxiliary.step",
    )
    attempt = _validate_structured_output_auxiliary_integer(
        raw_auxiliary["attempt"],
        "intent.auxiliary.attempt",
    )
    valid = raw_auxiliary["valid"]
    retry_scheduled = raw_auxiliary["retry_scheduled"]
    if type(valid) is not bool:
        raise TypeError("intent.auxiliary.valid must be a boolean.")
    if type(retry_scheduled) is not bool:
        raise TypeError("intent.auxiliary.retry_scheduled must be a boolean.")
    if valid and retry_scheduled:
        raise ValueError("A valid structured-output result cannot schedule a retry.")

    raw_event_ids = raw_auxiliary["event_ids"]
    if type(raw_event_ids) is not list:
        raise TypeError("intent.auxiliary.event_ids must be a list.")
    event_ids = tuple(
        require_clean_nonblank(event_id, "intent.auxiliary.event_ids item")
        for event_id in raw_event_ids
    )
    if len(set(event_ids)) != len(event_ids):
        raise ValueError("intent.auxiliary.event_ids cannot contain duplicates.")

    expected_event_types = (
        (
            EventType.STRUCTURED_OUTPUT_VALIDATING,
            EventType.STRUCTURED_OUTPUT_VALIDATED,
        )
        if valid
        else (
            (
                EventType.STRUCTURED_OUTPUT_VALIDATING,
                EventType.STRUCTURED_OUTPUT_FAILED,
                EventType.STRUCTURED_OUTPUT_RETRY,
            )
            if retry_scheduled
            else (
                EventType.STRUCTURED_OUTPUT_VALIDATING,
                EventType.STRUCTURED_OUTPUT_FAILED,
            )
        )
    )
    if event_ids != tuple(event.id for event in request.events):
        raise ValueError(
            "Structured-output auxiliary event ids and order must exactly match "
            "intent.auxiliary.event_ids."
        )
    if tuple(event.type for event in request.events) != expected_event_types:
        raise ValueError("Structured-output auxiliary events have an invalid type sequence.")

    event_session_ids = {event.session_id for event in request.events}
    if len(event_session_ids) != 1:
        raise ValueError("Structured-output auxiliary events must belong to one session.")
    event_session_id = next(iter(event_session_ids))
    first_event = request.events[0]
    expected_name = first_event.payload.get("name")
    if expected_name is not None:
        require_clean_nonblank(expected_name, "structured-output event name")
    expected_max_retries = first_event.payload.get("max_retries")
    if type(expected_max_retries) is not int or not 0 <= expected_max_retries <= 8:
        raise ValueError("Structured-output auxiliary events require a valid max_retries value.")

    outcome_output_present = False
    outcome_output: Any = None
    outcome_errors: tuple[dict[str, Any], ...] | None = None
    for event in request.events:
        if event.tool_name is not None:
            raise ValueError("Structured-output auxiliary events cannot carry a tool identity.")
        if event.payload.get("tool_round_id") != round_id:
            raise ValueError("A structured-output auxiliary event has a conflicting round id.")
        if type(event.payload.get("step")) is not int or event.payload["step"] != step:
            raise ValueError("A structured-output auxiliary event has a conflicting step.")
        if type(event.payload.get("attempt")) is not int or event.payload["attempt"] != attempt:
            raise ValueError("A structured-output auxiliary event has a conflicting attempt.")
        if event.payload.get("name") != expected_name:
            raise ValueError("Structured-output auxiliary events have conflicting names.")
        if (
            type(event.payload.get("max_retries")) is not int
            or event.payload["max_retries"] != expected_max_retries
        ):
            raise ValueError("Structured-output auxiliary events have conflicting max_retries.")
        if event.type == EventType.STRUCTURED_OUTPUT_VALIDATING:
            if event.payload.get("strategy") != StructuredOutputStrategy.TOOL.value:
                raise ValueError("A structured-output validating event must use tool strategy.")
            if any(field in event.payload for field in ("valid", "output", "errors")):
                raise ValueError("A structured-output validating event cannot claim an outcome.")
            continue
        event_valid = event.payload.get("valid")
        if type(event_valid) is not bool or event_valid != valid:
            raise ValueError(
                "A structured-output outcome event conflicts with intent.auxiliary.valid."
            )
        raw_errors = event.payload.get("errors")
        if type(raw_errors) is not list:
            raise ValueError("A structured-output outcome event requires an errors list.")
        normalized_errors: list[dict[str, Any]] = []
        for raw_error in raw_errors:
            try:
                normalized_error = StructuredOutputError.model_validate(raw_error).model_dump(
                    mode="json"
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "A structured-output outcome event contains an invalid error."
                ) from exc
            normalized_errors.append(normalized_error)
        event_errors = tuple(normalized_errors)
        event_output_present = "output" in event.payload
        event_output = (
            copy_durable_json_value(event.payload["output"], "structured_output_event.output")
            if event_output_present
            else None
        )
        if valid:
            if event_errors or not event_output_present:
                raise ValueError("A valid structured-output event requires output and no errors.")
        elif not event_errors or event_output_present:
            raise ValueError("A failed structured-output event requires errors and no output.")
        if outcome_errors is None:
            outcome_output_present = event_output_present
            outcome_output = event_output
            outcome_errors = event_errors
        elif (
            event_output_present != outcome_output_present
            or not session_authority_rules._runtime_publication_json_equal(
                event_output, outcome_output
            )
            or event_errors != outcome_errors
        ):
            raise ValueError(
                "Structured-output outcome and retry events contain conflicting results."
            )

    assert outcome_errors is not None

    return _StructuredOutputToolRoundAuxiliary(
        step=step,
        attempt=attempt,
        valid=valid,
        retry_scheduled=retry_scheduled,
        event_ids=event_ids,
        session_id=event_session_id,
        output_present=outcome_output_present,
        output=outcome_output,
        errors=outcome_errors,
    )


def _validate_tool_round_publication(
    request: RuntimePublicationRequest,
    durable_events_by_id: Mapping[str, Event],
    *,
    durable_tool_events: Iterable[Event] | None = None,
) -> None:
    """Validate complete lifecycle evidence for every grouped tool result."""

    identity = _tool_lifecycle_publication_identity(request)
    if identity is None:
        return
    execution_identity, tool_call_ids = identity
    round_id = execution_identity.tool_round_id
    auxiliary = (
        _validate_structured_output_tool_round_auxiliary(
            request,
            round_id=round_id,
        )
        if request.kind == "tool-round"
        else None
    )

    if len(request.transcript_messages) not in {1, 2}:
        raise ValueError(
            "A tool-round publication must append its grouped result and optional "
            "deferred assistant message."
        )
    assistant_call_parts: list[ToolCallPart] = []
    result_message = request.transcript_messages[-1]
    if len(request.transcript_messages) == 2:
        assistant_message = request.transcript_messages[0]
        if assistant_message.role != MessageRole.ASSISTANT:
            raise ValueError("A deferred tool-round message must be an assistant message.")
        assistant_call_parts = [
            part for part in assistant_message.content if isinstance(part, ToolCallPart)
        ]
        if not assistant_call_parts:
            raise ValueError("A deferred assistant message requires tool-call content.")
    if result_message.role != MessageRole.TOOL:
        raise ValueError("A tool-round publication must end with a tool-result message.")
    result_parts: list[ToolResultPart] = []
    for part in result_message.content:
        if not isinstance(part, ToolResultPart):
            raise ValueError(
                "A tool-round publication result message can contain only tool results."
            )
        result_parts.append(part)
    if tuple(part.tool_call_id for part in result_parts) != tool_call_ids:
        raise ValueError(
            "Tool-round result order and identities must exactly match intent.tool_call_ids."
        )
    if (
        assistant_call_parts
        and tuple(part.tool_call_id for part in assistant_call_parts) != tool_call_ids
    ):
        raise ValueError(
            "Deferred assistant tool-call order must exactly match intent.tool_call_ids."
        )

    candidate_durable_events = (
        tuple(durable_events_by_id.values())
        if durable_tool_events is None
        else tuple(durable_tool_events)
    )
    scoped_durable_events: list[Event] = []
    for event in candidate_durable_events:
        if event.type not in _TOOL_ROUND_LIFECYCLE_EVENT_TYPES:
            raise ValueError("Tool-round durable evidence may contain only lifecycle events.")
        tool_call_id = event.payload.get("tool_call_id")
        if type(tool_call_id) is not str or tool_call_id not in tool_call_ids:
            raise ValueError("Tool-round durable evidence must identify one intended tool call.")
        try:
            durable_identity = ToolRoundIdentity.model_validate(
                {
                    "model_step_id": event.payload.get("model_step_id"),
                    "model_attempt_id": event.payload.get("model_attempt_id"),
                    "tool_round_id": event.payload.get("tool_round_id"),
                }
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Tool-round durable evidence requires a complete valid execution identity."
            ) from exc
        if durable_identity != execution_identity:
            raise ValueError("Tool-round durable evidence has a conflicting execution identity.")
        scoped_durable_events.append(event)

    scoped_durable_event_ids = tuple(event.id for event in scoped_durable_events)
    referenced_event_ids = tuple(reference.event_id for reference in request.referenced_events)
    if len(set(scoped_durable_event_ids)) != len(scoped_durable_event_ids) or set(
        scoped_durable_event_ids
    ) != set(referenced_event_ids):
        raise SessionRuntimePublicationConflict(
            "A tool-round publication must bind every durable lifecycle event for "
            "its intended tool calls."
        )
    if auxiliary is not None and any(
        event.session_id != auxiliary.session_id for event in scoped_durable_events
    ):
        raise ValueError(
            "Structured-output auxiliary and lifecycle events belong to different sessions."
        )

    material_events = tuple(scoped_durable_events)
    started_by_call: dict[str, Event] = {}
    terminal_by_call: dict[str, Event] = {}
    for event in material_events:
        if event.type != EventType.TOOL_CALL_STARTED and (
            event.type not in _TOOL_ROUND_TERMINAL_EVENT_TYPES
        ):
            raise ValueError(
                "A tool-round publication may bind only started and terminal tool events."
            )
        tool_call_id = event.payload.get("tool_call_id")
        if type(tool_call_id) is not str or tool_call_id not in tool_call_ids:
            raise ValueError("A tool-round publication event must identify one intended tool call.")
        event_round_id = event.payload.get("tool_round_id")
        if event_round_id != round_id:
            raise ValueError("A tool-round publication event has a conflicting round id.")
        destination = (
            started_by_call if event.type == EventType.TOOL_CALL_STARTED else terminal_by_call
        )
        if tool_call_id in destination:
            raise ValueError(
                "A tool-round publication cannot bind duplicate started or terminal events "
                f"for tool call {tool_call_id}."
            )
        destination[tool_call_id] = event

    if set(terminal_by_call) != set(tool_call_ids):
        raise ValueError("A tool-round publication requires exactly one terminal event per result.")
    if not set(started_by_call).issubset(terminal_by_call):
        raise ValueError(
            "Every started tool call in a publication requires a matching terminal event."
        )

    for part in result_parts:
        if (
            part.model_step_id != execution_identity.model_step_id
            or part.model_attempt_id != execution_identity.model_attempt_id
            or part.tool_round_id != execution_identity.tool_round_id
        ):
            raise ValueError("A tool-round transcript result has a conflicting execution identity.")
        terminal = terminal_by_call[part.tool_call_id]
        terminal_is_gateway = (
            terminal.payload.get("dispatch_kind") == "gateway"
            and terminal.payload.get("model_tool_name") == part.tool_name
        )
        if terminal.tool_name != part.tool_name and not terminal_is_gateway:
            raise ValueError(
                "A tool-round terminal event tool name conflicts with its transcript result."
            )
        terminal_idempotency_key = terminal.payload.get("idempotency_key")
        if type(terminal_idempotency_key) is not str or not terminal_idempotency_key:
            raise ValueError("A tool-round terminal event requires a stable idempotency key.")
        started = started_by_call.get(part.tool_call_id)
        if started is not None:
            started_is_gateway = (
                started.payload.get("dispatch_kind") == "gateway"
                and started.payload.get("model_tool_name") == part.tool_name
            )
            if (
                (started.tool_name != part.tool_name and not started_is_gateway)
                or started.tool_name != terminal.tool_name
                or started.payload.get("idempotency_key") != terminal_idempotency_key
                or started_is_gateway != terminal_is_gateway
            ):
                raise ValueError("A tool-round started event conflicts with its terminal event.")
        expected_result = {
            "content": part.content,
            "structured": part.structured,
            "artifacts": part.artifacts,
            "is_error": part.is_error,
        }
        if not session_authority_rules._runtime_publication_json_equal(
            terminal.payload.get("result"),
            expected_result,
        ):
            raise ValueError(
                "A tool-round terminal event result conflicts with its transcript result."
            )
        if (terminal.type == EventType.TOOL_CALL_COMPLETED and part.is_error) or (
            terminal.type != EventType.TOOL_CALL_COMPLETED and not part.is_error
        ):
            raise ValueError(
                "A tool-round terminal event outcome conflicts with its transcript result."
            )
    for part in assistant_call_parts:
        if (
            part.model_step_id != execution_identity.model_step_id
            or part.model_attempt_id != execution_identity.model_attempt_id
            or part.tool_round_id != execution_identity.tool_round_id
        ):
            raise ValueError("A deferred assistant tool call has a conflicting execution identity.")
        terminal = terminal_by_call[part.tool_call_id]
        terminal_is_gateway = (
            terminal.payload.get("dispatch_kind") == "gateway"
            and terminal.payload.get("model_tool_name") == part.tool_name
        )
        if terminal.tool_name != part.tool_name and not terminal_is_gateway:
            raise ValueError("A deferred assistant tool name conflicts with terminal evidence.")
        arguments_state = terminal.payload.get("arguments_state")
        expected_arguments = (
            terminal.payload.get("arguments") if arguments_state == "finalized" else {}
        )
        arguments_match = type(
            expected_arguments
        ) is dict and session_authority_rules._runtime_publication_json_equal(
            part.arguments, expected_arguments
        )
        if terminal_is_gateway:
            if terminal.payload.get("blocked_by") == "targeted_tool_gateway":
                arguments_match = arguments_state == "unavailable" and not part.arguments
                if not arguments_match:
                    raise ValueError(
                        "A rejected gateway assistant projection must quarantine its envelope."
                    )
                continue
            outer_arguments = part.arguments
            inner_arguments = outer_arguments.get("arguments")
            tool_ref = outer_arguments.get("tool_ref")
            gateway_projection_matches = (
                set(outer_arguments) == {"tool_ref", "arguments"}
                and tool_ref == TARGETED_TOOL_TRANSCRIPT_REFERENCE
                and type(inner_arguments) is dict
            )
            arguments_match = False
            if gateway_projection_matches:
                if arguments_state == "unavailable":
                    arguments_match = not inner_arguments
                elif arguments_state == "finalized":
                    assert type(inner_arguments) is dict
                    arguments_match = (
                        terminal.payload.get("arguments_sha256")
                        == f"sha256:{sha256(canonical_durable_json_bytes(inner_arguments, 'arguments')).hexdigest()}"
                        and session_authority_rules._runtime_publication_json_equal(
                            inner_arguments,
                            expected_arguments,
                        )
                    )
        if arguments_state not in {"finalized", "unavailable"} or not arguments_match:
            raise ValueError(
                "A deferred assistant argument projection conflicts with terminal evidence."
            )

    if request.kind == "tool-round":
        pending_round_deletions = [
            operation
            for operation in request.mutation.operations
            if operation.key == _PENDING_TOOL_ROUND_CHECKPOINT_KEY and operation.action == "delete"
        ]
        if len(pending_round_deletions) != 1:
            raise ValueError(
                "A tool-round publication must delete exactly one fenced pending round marker."
            )


def _validate_structured_output_tool_round_marker(
    request: RuntimePublicationRequest,
    *,
    marker: dict[str, Any],
    raw_calls: list[dict[str, Any]],
) -> None:
    round_id = request.intent["round_id"]
    auxiliary = _validate_structured_output_tool_round_auxiliary(
        request,
        round_id=round_id,
    )
    structured_calls = [
        call for call in raw_calls if call.get("tool_name") == STRUCTURED_OUTPUT_TOOL_NAME
    ]
    if auxiliary is None:
        if structured_calls:
            raise SessionRuntimePublicationConflict(
                "A pending structured-output tool round requires validated "
                "intent.auxiliary evidence."
            )
        return
    if not structured_calls:
        raise SessionRuntimePublicationConflict(
            "Structured-output auxiliary evidence requires the reserved tool "
            "in the durable pending marker."
        )
    if any(type(call.get("arguments")) is not dict for call in structured_calls):
        raise SessionRuntimePublicationConflict(
            "The reserved structured-output call arguments are malformed."
        )

    raw_spec = marker.get("structured_output")
    if type(raw_spec) is not dict:
        raise SessionRuntimePublicationConflict(
            "Structured-output auxiliary evidence requires durable structured-output config."
        )
    try:
        spec = StructuredOutputSpec.model_validate(raw_spec)
    except (TypeError, ValueError) as exc:
        raise SessionRuntimePublicationConflict(
            "The durable structured-output config is malformed."
        ) from exc
    if spec.strategy != StructuredOutputStrategy.TOOL:
        raise SessionRuntimePublicationConflict(
            "Structured-output tool-round auxiliary evidence requires tool strategy."
        )
    if auxiliary.attempt > spec.max_retries + 1:
        raise SessionRuntimePublicationConflict(
            "Structured-output auxiliary attempt exceeds the durable retry policy."
        )
    if auxiliary.retry_scheduled and auxiliary.attempt > spec.max_retries:
        raise SessionRuntimePublicationConflict(
            "Structured-output auxiliary retry exceeds the durable retry policy."
        )
    raw_validation = marker.get("structured_output_validation")
    try:
        durable_validation = _validate_durable_structured_output_validation(
            raw_validation,
            "pending_tool_round.structured_output_validation",
        )
    except (TypeError, ValueError) as exc:
        raise SessionRuntimePublicationConflict(
            "The durable structured-output validation is malformed."
        ) from exc
    durable_errors = tuple(error.model_dump(mode="json") for error in durable_validation.errors)
    if (
        auxiliary.valid != durable_validation.valid
        or (
            durable_validation.valid
            and (
                durable_errors
                or not auxiliary.output_present
                or not session_authority_rules._runtime_publication_json_equal(
                    auxiliary.output,
                    durable_validation.output,
                )
            )
        )
        or (
            not durable_validation.valid
            and (
                durable_validation.output is not None
                or auxiliary.output_present
                or auxiliary.errors != durable_errors
            )
        )
    ):
        raise SessionRuntimePublicationConflict(
            "Structured-output auxiliary outcome conflicts with its authoritative validation."
        )
    marker_model_step = marker.get("model_step")
    if marker_model_step is not None and (
        type(marker_model_step) is not int or marker_model_step != auxiliary.step
    ):
        raise SessionRuntimePublicationConflict(
            "Structured-output auxiliary step conflicts with the durable pending marker."
        )
    marker_structured_output_attempt = marker.get("structured_output_attempt")
    if marker_structured_output_attempt is not None and (
        type(marker_structured_output_attempt) is not int
        or marker_structured_output_attempt != auxiliary.attempt
    ):
        raise SessionRuntimePublicationConflict(
            "Structured-output auxiliary attempt conflicts with the durable pending marker."
        )

    marker_agent_name = marker.get("agent_name")
    if type(marker_agent_name) is not str:
        raise SessionRuntimePublicationConflict(
            "The durable pending tool round agent identity is malformed."
        )
    try:
        marker_agent_name = require_clean_nonblank(
            marker_agent_name,
            "pending tool round agent_name",
        )
    except (TypeError, ValueError) as exc:
        raise SessionRuntimePublicationConflict(
            "The durable pending tool round agent identity is malformed."
        ) from exc
    marker_environment_name = marker.get("environment_name")
    if marker_environment_name is not None:
        try:
            marker_environment_name = require_clean_nonblank(
                marker_environment_name,
                "pending tool round environment_name",
            )
        except (TypeError, ValueError) as exc:
            raise SessionRuntimePublicationConflict(
                "The durable pending tool round environment identity is malformed."
            ) from exc

    for event in request.events:
        if (
            event.agent_name != marker_agent_name
            or event.environment_name != marker_environment_name
        ):
            raise SessionRuntimePublicationConflict(
                "A structured-output auxiliary event conflicts with the durable "
                "pending-round identity."
            )
        if event.payload.get("name") != spec.name:
            raise SessionRuntimePublicationConflict(
                "A structured-output auxiliary event conflicts with the durable output name."
            )
        if (
            type(event.payload.get("max_retries")) is not int
            or event.payload["max_retries"] != spec.max_retries
        ):
            raise SessionRuntimePublicationConflict(
                "A structured-output auxiliary event conflicts with the durable retry policy."
            )

    result_parts = [
        part
        for message in request.transcript_messages
        for part in message.content
        if isinstance(part, ToolResultPart)
    ]
    result_parts_by_id = {part.tool_call_id: part for part in result_parts}
    for structured_call in structured_calls:
        tool_call_id = structured_call.get("tool_call_id")
        if type(tool_call_id) is not str:
            raise SessionRuntimePublicationConflict(
                "The reserved structured-output call identity is malformed."
            )
        result_part = result_parts_by_id.get(tool_call_id)
        if result_part is None or result_part.tool_name != STRUCTURED_OUTPUT_TOOL_NAME:
            raise SessionRuntimePublicationConflict(
                "The reserved structured-output call is not bound to its transcript result."
            )
    if auxiliary.valid:
        if len(structured_calls) != 1 or len(raw_calls) != 1:
            raise SessionRuntimePublicationConflict(
                "A valid structured-output auxiliary outcome requires exactly one "
                "reserved tool call."
            )
        structured_call_id = structured_calls[0]["tool_call_id"]
        if result_parts_by_id[structured_call_id].is_error:
            raise SessionRuntimePublicationConflict(
                "A valid structured-output auxiliary outcome conflicts with its tool result."
            )
        structured_result = result_parts_by_id[structured_call_id].structured
        if (
            type(structured_result) is not dict
            or set(structured_result) != {"output"}
            or not auxiliary.output_present
            or not session_authority_rules._runtime_publication_json_equal(
                structured_result["output"],
                auxiliary.output,
            )
        ):
            raise SessionRuntimePublicationConflict(
                "A valid structured-output event conflicts with its grouped tool result."
            )
    elif any(not part.is_error for part in result_parts):
        raise SessionRuntimePublicationConflict(
            "A failed structured-output auxiliary outcome conflicts with its tool results."
        )
    else:
        expected_errors = list(auxiliary.errors)
        for result_part in result_parts:
            structured_result = result_part.structured
            if (
                type(structured_result) is not dict
                or set(structured_result) != {"structured_output_errors"}
                or not session_authority_rules._runtime_publication_json_equal(
                    structured_result["structured_output_errors"],
                    expected_errors,
                )
            ):
                raise SessionRuntimePublicationConflict(
                    "A failed structured-output event conflicts with its grouped tool results."
                )


def _validate_tool_round_checkpoint_mutation(
    request: RuntimePublicationRequest,
    checkpoint: dict[str, Any] | None,
) -> None:
    if request.kind != "tool-round":
        return
    publication_identity = _tool_round_publication_identity(request)
    if publication_identity is None:
        return
    execution_identity, intended_call_ids = publication_identity
    marker = None if checkpoint is None else checkpoint.get(_PENDING_TOOL_ROUND_CHECKPOINT_KEY)
    if type(marker) is not dict:
        raise SessionRuntimePublicationConflict(
            "A tool-round publication requires its durable pending round marker."
        )
    raw_calls = marker.get("tool_calls")
    assistant_publication = marker.get("assistant_publication")
    expected_continuity = (
        assistant_publication.get("argument_continuity")
        if type(assistant_publication) is dict and assistant_publication.get("state") == "ready"
        else None
    )
    actual_continuity = (
        None
        if request.argument_continuity is None
        else request.argument_continuity.model_dump(mode="json")
    )
    if actual_continuity != expected_continuity:
        raise SessionRuntimePublicationConflict(
            "Private argument continuity conflicts with sealed evidence."
        )
    if request.argument_continuity is not None:
        active_profile = active_invocation_execution_profile_from_checkpoint(checkpoint)
        if (
            active_profile is None
            or active_profile.profile.fingerprint != request.argument_continuity.profile
            or marker.get("execution_profile_fingerprint") != request.argument_continuity.profile
        ):
            raise SessionRuntimePublicationConflict(
                "Private argument continuity lost its profile authority."
            )
    if type(raw_calls) is not list or any(type(call) is not dict for call in raw_calls):
        raise SessionRuntimePublicationConflict(
            "The durable pending tool round call list is malformed."
        )
    assistant_message_state = marker.get("assistant_message_state", "published")
    if assistant_message_state not in {"published", "quarantined"}:
        raise SessionRuntimePublicationConflict(
            "The durable pending tool round assistant-message state is malformed."
        )
    expected_transcript_message_count = 2 if assistant_message_state == "quarantined" else 1
    if len(request.transcript_messages) != expected_transcript_message_count:
        raise SessionRuntimePublicationConflict(
            "The tool-round transcript does not match the durable assistant-message state."
        )
    marker_call_ids = tuple(call.get("tool_call_id") for call in raw_calls)
    if (
        marker.get("tool_round_id") != execution_identity.tool_round_id
        or marker_call_ids != intended_call_ids
    ):
        raise SessionRuntimePublicationConflict(
            "The durable pending tool round conflicts with the publication intent."
        )
    for field_name in ("model_step_id", "model_attempt_id", "tool_round_id"):
        if marker.get(field_name) != getattr(execution_identity, field_name):
            raise SessionRuntimePublicationConflict(
                "The durable pending tool-round execution identity conflicts "
                "with the publication intent."
            )
    _validate_structured_output_tool_round_marker(
        request,
        marker=marker,
        raw_calls=raw_calls,
    )
    deletions = [
        operation
        for operation in request.mutation.operations
        if operation.key == _PENDING_TOOL_ROUND_CHECKPOINT_KEY and operation.action == "delete"
    ]
    if len(deletions) != 1:
        raise SessionRuntimePublicationConflict(
            "A tool-round publication must delete exactly one fenced pending round marker."
        )
    deletion = deletions[0]
    if runtime_publication_checkpoint_value_digest(marker) != deletion.expected_value_digest:
        raise SessionRuntimePublicationConflict(
            "The tool-round publication does not fence its exact pending marker."
        )

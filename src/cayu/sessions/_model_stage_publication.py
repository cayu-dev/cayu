"""Exact model-completion publication and assistant-turn evidence validation."""

from __future__ import annotations

from typing import Any

from cayu._validation import require_durable_clean_nonblank as require_clean_nonblank
from cayu.approvals.tools import PendingToolCallApproval
from cayu.budgets.base import model_completion_budget_settlements
from cayu.context.structured_output import (
    STRUCTURED_OUTPUT_TOOL_NAME,
    StructuredOutputSpec,
    StructuredOutputStrategy,
)
from cayu.events import Event, EventType
from cayu.execution_units import ModelAttemptIdentity, ToolRoundIdentity
from cayu.messages import Message, MessageRole, ToolCallPart
from cayu.sessions import _model_stage as model_stage_contracts
from cayu.sessions import _runtime_publication as publication_contracts
from cayu.sessions import _tool_round_publication_rules as tool_publication_rules
from cayu.sessions import authority as session_authority_rules
from cayu.sessions._checkpoint_mutations import runtime_publication_checkpoint_value_digest
from cayu.sessions._model_completion_publication import (
    LAST_MODEL_STEP_PUBLICATION_CHECKPOINT_KEY,
    ModelStepPublicationCheckpoint,
)
from cayu.sessions._model_stage import ModelCompletionStage, SessionModelCompletionStageConflict
from cayu.sessions._runtime_publication import RuntimePublicationRequest
from cayu.sessions.checkpoints import (
    CHECKPOINT_SCHEMA_VERSION_KEY,
    CURRENT_CHECKPOINT_SCHEMA_VERSION,
    decode_runtime_checkpoint,
)
from cayu.tools._policy_evidence import ToolPolicyEvidence
from cayu.tools.catalogue import CALL_TOOL_NAME

_NON_TURN_MODEL_COMPLETION_CLASSIFICATIONS = frozenset({"failed", "filtered", "invalid", "length"})


_MESSAGELESS_MODEL_COMPLETION_CLASSIFICATIONS = _NON_TURN_MODEL_COMPLETION_CLASSIFICATIONS | {
    "continue"
}


def _validate_model_completion_stage_publication(
    publication: RuntimePublicationRequest,
    *,
    session_id: str,
    stage: ModelCompletionStage | model_stage_contracts._ModelCompletionStagePreparationRecord,
) -> None:
    if publication.publication_id != stage.logical_step_id:
        raise ValueError("Model completion publication_id must equal logical_step_id.")
    if not session_authority_rules._runtime_publication_json_equal(
        publication.intent, stage.intent
    ):
        raise SessionModelCompletionStageConflict(
            "The terminal model completion intent conflicts with its preparation."
        )
    if stage.purpose == "auxiliary-inference":
        from cayu.runtime._auxiliary_inference_contract import validate_auxiliary_publication

        validate_auxiliary_publication(publication, session_id=session_id, stage=stage)
        return
    if stage.purpose == "assistant-turn":
        if publication.kind != "model-step":
            raise ValueError("Assistant-turn publication kind must be 'model-step'.")
        if len(publication.transcript_messages) > 1 or any(
            message.role != MessageRole.ASSISTANT for message in publication.transcript_messages
        ):
            raise ValueError(
                "An assistant-turn publication may contain at most one assistant message."
            )
    else:
        if publication.kind != "context-compaction":
            raise ValueError("Context-compaction publication kind must be 'context-compaction'.")
        if publication.transcript_messages:
            raise ValueError("A context-compaction publication cannot append transcript messages.")
    completed_events = [
        event for event in publication.events if event.type == EventType.MODEL_COMPLETED
    ]
    if len(completed_events) != 1:
        raise ValueError(
            "A model completion publication must append exactly one model.completed event."
        )
    if stage.purpose == "assistant-turn" and (
        len(publication.events) != 1 or publication.referenced_events
    ):
        raise ValueError(
            "An assistant-turn publication must bind exactly its model.completed event."
        )
    completed_event = completed_events[0]
    if stage.purpose == "assistant-turn" and (
        stage.reservation_ids or "budget_settlements" in completed_event.payload
    ):
        model_completion_budget_settlements(
            completed_event,
            reservation_ids=stage.reservation_ids,
        )
    classification = completed_event.payload.get("step_classification")
    if (
        stage.purpose == "assistant-turn"
        and not publication.transcript_messages
        and (
            type(classification) is not dict
            or classification.get("type") not in _MESSAGELESS_MODEL_COMPLETION_CLASSIFICATIONS
        )
    ):
        raise ValueError(
            "A model completion publication without an assistant message requires "
            "a non-turn or continue model.completed step classification."
        )
    if (
        stage.purpose == "assistant-turn"
        and completed_event.payload.get("purpose") == "context_compaction"
    ):
        raise ValueError("An assistant-turn publication cannot carry context-compaction evidence.")
    if (
        stage.purpose == "context-compaction"
        and completed_event.payload.get("purpose") != "context_compaction"
    ):
        raise ValueError(
            "A context-compaction publication requires matching model completion evidence."
        )
    for event in publication.events:
        if event.session_id != session_id:
            raise ValueError("Model completion event session_id does not match target session.")
    appended_event_ids = {event.id for event in publication.events}
    overlap = appended_event_ids & set(
        publication_contracts._runtime_publication_referenced_event_ids(
            publication.referenced_events
        )
    )
    if overlap:
        raise ValueError(
            f"Model completion cannot both append and reference event id {min(overlap)}."
        )
    if stage.purpose == "assistant-turn":
        _validate_assistant_model_completion_publication(
            publication,
            stage=stage,
            completed_event=completed_event,
            classification=classification,
        )


def _validate_assistant_model_completion_publication(
    publication: RuntimePublicationRequest,
    *,
    stage: ModelCompletionStage | model_stage_contracts._ModelCompletionStagePreparationRecord,
    completed_event: Event,
    classification: Any,
) -> None:
    if type(classification) is not dict:
        raise ValueError("An assistant-turn publication requires a model step classification.")
    raw_model_step_id = stage.intent.get("model_step_id")
    raw_model_attempt_id = stage.intent.get("model_attempt_id")
    model_attempt_identity = None
    if raw_model_step_id is not None or raw_model_attempt_id is not None:
        if type(raw_model_step_id) is not str or type(raw_model_attempt_id) is not str:
            raise ValueError(
                "An assistant-turn completion stage has a malformed model-attempt identity."
            )
        try:
            model_attempt_identity = ModelAttemptIdentity(
                model_step_id=raw_model_step_id,
                model_attempt_id=raw_model_attempt_id,
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "An assistant-turn completion stage has a malformed model-attempt identity."
            ) from exc
        if (
            model_attempt_identity.model_step_id != stage.logical_step_id
            or completed_event.payload.get("model_step_id") != model_attempt_identity.model_step_id
            or completed_event.payload.get("model_attempt_id")
            != model_attempt_identity.model_attempt_id
        ):
            raise ValueError(
                "The model.completed identity conflicts with its staged model attempt."
            )
    published_assistant_message = (
        publication.transcript_messages[0] if publication.transcript_messages else None
    )
    operations_by_key = {operation.key: operation for operation in publication.mutation.operations}
    schema_operation = operations_by_key.get(CHECKPOINT_SCHEMA_VERSION_KEY)
    pending_operation = operations_by_key.get(
        tool_publication_rules._PENDING_TOOL_ROUND_CHECKPOINT_KEY
    )
    pending_marker = None if pending_operation is None else pending_operation.value
    assistant_message = published_assistant_message
    assistant_message_deferred = False
    if (
        assistant_message is None
        and type(pending_marker) is dict
        and pending_marker.get("assistant_message_state") == "quarantined"
    ):
        raw_quarantined_message = pending_marker.get("quarantined_assistant_message")
        try:
            assistant_message = Message.model_validate(raw_quarantined_message)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "A deferred model completion requires a valid private assistant message."
            ) from exc
        assistant_message_deferred = True
    tool_calls = (
        tuple(part for part in assistant_message.content if isinstance(part, ToolCallPart))
        if assistant_message is not None
        else ()
    )
    classification_type = classification.get("type")
    if tool_calls and classification_type != "continue":
        raise ValueError(
            "An assistant model completion with tool calls must be classified as continue."
        )

    tool_round_identity = None
    if tool_calls:
        if model_attempt_identity is None:
            raise ValueError("Assistant tool calls require a staged model-attempt identity.")
        raw_identities = {
            (call.model_step_id, call.model_attempt_id, call.tool_round_id) for call in tool_calls
        }
        if len(raw_identities) != 1:
            raise ValueError("Assistant tool calls must share one complete tool-round identity.")
        model_step_id, model_attempt_id, tool_round_id = next(iter(raw_identities))
        if model_step_id is None or model_attempt_id is None or tool_round_id is None:
            raise ValueError("Assistant tool calls must carry a complete tool-round identity.")
        try:
            tool_round_identity = ToolRoundIdentity(
                model_step_id=model_step_id,
                model_attempt_id=model_attempt_id,
                tool_round_id=tool_round_id,
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Assistant tool calls must carry a valid tool-round identity."
            ) from exc
        if (
            tool_round_identity.model_step_id != model_attempt_identity.model_step_id
            or tool_round_identity.model_attempt_id != model_attempt_identity.model_attempt_id
        ):
            raise ValueError(
                "Assistant tool-round identity conflicts with its staged model completion."
            )
    expected_tool_round_id = (
        None if tool_round_identity is None else tool_round_identity.tool_round_id
    )
    expected_operation_keys = {LAST_MODEL_STEP_PUBLICATION_CHECKPOINT_KEY}
    if tool_calls:
        expected_operation_keys.add(tool_publication_rules._PENDING_TOOL_ROUND_CHECKPOINT_KEY)
    actual_operation_keys = frozenset(operations_by_key)
    if actual_operation_keys not in {
        frozenset(expected_operation_keys),
        frozenset(expected_operation_keys | {CHECKPOINT_SCHEMA_VERSION_KEY}),
    }:
        raise ValueError(
            "An assistant model completion must atomically publish exactly its model-step "
            "pointer, optional pending tool-round marker, and optional root checkpoint "
            "schema stamp."
        )
    if schema_operation is not None:
        if (
            type(schema_operation.value) is not int
            or not 1 <= schema_operation.value <= CURRENT_CHECKPOINT_SCHEMA_VERSION
        ):
            raise ValueError(
                "A model completion root checkpoint schema stamp must target a supported version."
            )
        supported_schema_digests = {
            runtime_publication_checkpoint_value_digest(version)
            for version in range(1, CURRENT_CHECKPOINT_SCHEMA_VERSION + 1)
        }
        if (
            schema_operation.action != "set"
            or schema_operation.expected_value_digest not in supported_schema_digests | {None}
        ):
            raise ValueError(
                "A model completion root checkpoint schema stamp must fence a supported version."
            )

    pointer_operation = operations_by_key[LAST_MODEL_STEP_PUBLICATION_CHECKPOINT_KEY]
    if pointer_operation.action != "set":
        raise ValueError("A model-step publication pointer must use a set operation.")
    try:
        if type(pointer_operation.value) is not dict:
            raise TypeError("The model-step publication pointer must be an object.")
        raw_pointer_schema_version = pointer_operation.value.get("schema_version")
        if (
            type(raw_pointer_schema_version) is not int
            or not 1 <= raw_pointer_schema_version <= CURRENT_CHECKPOINT_SCHEMA_VERSION
        ):
            raise ValueError(
                "The model-step publication pointer has an unsupported schema version."
            )
        root_schema_version = (
            raw_pointer_schema_version if schema_operation is None else schema_operation.value
        )
        staged_checkpoint = {
            CHECKPOINT_SCHEMA_VERSION_KEY: root_schema_version,
            LAST_MODEL_STEP_PUBLICATION_CHECKPOINT_KEY: pointer_operation.value,
        }
        if type(pending_marker) is dict:
            staged_checkpoint[tool_publication_rules._PENDING_TOOL_ROUND_CHECKPOINT_KEY] = (
                pending_marker
            )
        normalized_pointer_checkpoint = decode_runtime_checkpoint(
            staged_checkpoint,
            session_id=completed_event.session_id,
        )
        if normalized_pointer_checkpoint is None:
            raise TypeError("The model-step publication pointer is missing.")
        pointer = ModelStepPublicationCheckpoint.model_validate(
            normalized_pointer_checkpoint[LAST_MODEL_STEP_PUBLICATION_CHECKPOINT_KEY]
        )
        if type(pending_marker) is dict:
            pending_marker = normalized_pointer_checkpoint.get(
                tool_publication_rules._PENDING_TOOL_ROUND_CHECKPOINT_KEY
            )
    except (TypeError, ValueError) as exc:
        raise ValueError("The model-step publication pointer is malformed.") from exc
    expected_transcript_end = stage.source_transcript_cursor + int(
        published_assistant_message is not None
    )
    expected_pointer = ModelStepPublicationCheckpoint(
        logical_step_id=stage.logical_step_id,
        stage_id=stage.stage_id,
        source_transcript_cursor=stage.source_transcript_cursor,
        transcript_end_cursor=expected_transcript_end,
        completion_event_id=completed_event.id,
        classification=classification,
        assistant_message_published=published_assistant_message is not None,
        assistant_message_deferred=assistant_message_deferred,
        tool_round_id=expected_tool_round_id,
    )
    if pointer != expected_pointer:
        raise ValueError("The model-step publication pointer conflicts with its staged completion.")
    transcript_cursor = completed_event.payload.get("transcript_cursor")
    if type(transcript_cursor) is not int or transcript_cursor != expected_transcript_end:
        raise ValueError(
            "The model.completed transcript cursor conflicts with its staged completion."
        )

    if not tool_calls:
        return
    pending_operation = operations_by_key[tool_publication_rules._PENDING_TOOL_ROUND_CHECKPOINT_KEY]
    if pending_operation.action != "set" or type(pending_operation.value) is not dict:
        raise ValueError("A model-step pending tool-round marker must use an object set operation.")
    marker = pending_marker
    if type(marker) is not dict:
        raise ValueError("A model-step pending tool-round marker must be an object.")
    required_marker_keys = {
        "tool_round_id",
        "model_step_id",
        "model_attempt_id",
        "agent_name",
        "interaction_id",
        "environment_name",
        "task_id",
        "source_run_epoch",
        "execution_profile_fingerprint",
        "tool_calls",
        "policy_state",
        "policy_context_version",
        "request_metadata",
        "deferred_messages",
        "assistant_message_state",
        "quarantined_assistant_message",
        "staged_terminals",
        "structured_output",
        "thinking",
        "max_steps",
        "limits",
        "run_limit_accounting",
        "budget_limits",
        "retry_policy",
        "source_model_step_id",
        "source_transcript_cursor",
        "model_step",
        "structured_output_attempt",
        "structured_output_retries",
        "structured_output_validation",
    }
    marker_keys = set(marker)
    if not required_marker_keys <= marker_keys or not marker_keys <= (
        required_marker_keys | {"assistant_publication", "tool_exposure", "max_steps_unbounded"}
    ):
        raise ValueError("The model-step pending tool-round marker has invalid fields.")
    raw_run_limit_accounting = marker.get("run_limit_accounting")
    if raw_run_limit_accounting is not None and (
        type(raw_run_limit_accounting) is not dict
        or type(raw_run_limit_accounting.get("baseline")) is not dict
        or raw_run_limit_accounting["baseline"].get("session_id") != completed_event.session_id
    ):
        raise ValueError("The pending tool-round run-limit accounting owner is invalid.")
    if assistant_message_deferred:
        if assistant_message is None:
            raise AssertionError("Deferred assistant publication lost its private message.")
        if marker.get("quarantined_assistant_message") != assistant_message.model_dump(mode="json"):
            raise ValueError(
                "The quarantined assistant message conflicts with its model completion."
            )
    elif (
        marker.get("assistant_message_state") != "published"
        or marker.get("quarantined_assistant_message") is not None
    ):
        raise ValueError("A published assistant message cannot remain quarantined.")
    if tool_round_identity is None:
        raise ValueError("The model-step pending tool round lost its identity.")
    if (
        marker.get("tool_round_id") != tool_round_identity.tool_round_id
        or marker.get("model_step_id") != tool_round_identity.model_step_id
        or marker.get("model_attempt_id") != tool_round_identity.model_attempt_id
    ):
        raise ValueError("The pending tool-round identity conflicts with its model step.")
    if marker.get("source_model_step_id") != tool_round_identity.model_step_id:
        raise ValueError("The pending tool round has a conflicting source model-step identity.")
    if marker.get("source_transcript_cursor") != stage.source_transcript_cursor:
        raise ValueError("The pending tool round has a conflicting source transcript cursor.")
    marker_source_run_epoch = marker.get("source_run_epoch")
    if (
        type(marker_source_run_epoch) is not int
        or marker_source_run_epoch != stage.source_run_epoch
    ):
        raise ValueError("The pending tool round has a conflicting source run epoch.")
    try:
        require_clean_nonblank(marker.get("agent_name"), "pending tool-round agent_name")
        marker_interaction_id = marker.get("interaction_id")
        if marker_interaction_id is not None:
            require_clean_nonblank(
                marker_interaction_id,
                "pending tool-round interaction_id",
            )
            if marker_interaction_id != completed_event.interaction_id:
                raise ValueError("pending tool-round interaction_id conflicts with the model step")
        for field_name in ("environment_name", "task_id"):
            value = marker.get(field_name)
            if value is not None:
                require_clean_nonblank(value, f"pending tool-round {field_name}")
    except (TypeError, ValueError) as exc:
        raise ValueError("The pending tool-round runtime identity is malformed.") from exc
    marker_model_step = marker.get("model_step")
    if type(marker_model_step) is not int or marker_model_step < 1:
        raise ValueError("The pending tool round has an invalid model step.")
    completed_model_step = completed_event.payload.get("step")
    if completed_model_step is not None and (
        type(completed_model_step) is not int or completed_model_step != marker_model_step
    ):
        raise ValueError("The pending tool round has a conflicting model step.")

    raw_calls = marker.get("tool_calls")
    if type(raw_calls) is not list or len(raw_calls) != len(tool_calls):
        raise ValueError("The pending tool-round calls conflict with the assistant message.")
    try:
        pending_calls = tuple(
            PendingToolCallApproval.model_validate(raw_call) for raw_call in raw_calls
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("The pending tool-round call records are malformed.") from exc
    for pending_call, tool_call in zip(pending_calls, tool_calls, strict=True):
        if (
            pending_call.tool_call_id != tool_call.tool_call_id
            or pending_call.tool_name != tool_call.tool_name
            or not session_authority_rules._runtime_publication_json_equal(
                pending_call.transcript_arguments,
                tool_call.arguments,
            )
        ):
            raise ValueError("The pending tool-round calls conflict with the assistant message.")
        if (
            pending_call.policy_evidence != ToolPolicyEvidence.UNPLANNED
            or pending_call.policy_decision is not None
            or pending_call.reason is not None
            or pending_call.metadata
            or pending_call.active_taint_labels
        ):
            raise ValueError("A newly published pending tool round cannot contain policy outcomes.")
    if (
        any(tool_call.tool_name == CALL_TOOL_NAME for tool_call in tool_calls)
        and marker.get("interaction_id") != completed_event.interaction_id
    ):
        raise ValueError("A pending call_tool round requires its interaction identity.")

    raw_structured_output = marker.get("structured_output")
    structured_output = None
    if raw_structured_output is not None:
        if type(raw_structured_output) is not dict:
            raise ValueError("The pending tool-round structured-output config is malformed.")
        try:
            structured_output = StructuredOutputSpec.model_validate(raw_structured_output)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "The pending tool-round structured-output config is malformed."
            ) from exc
    has_structured_finalizer = any(
        tool_call.tool_name == STRUCTURED_OUTPUT_TOOL_NAME for tool_call in tool_calls
    )
    structured_output_attempt = marker.get("structured_output_attempt")
    raw_structured_output_validation = marker.get("structured_output_validation")
    if has_structured_finalizer:
        if structured_output is None or structured_output.strategy != StructuredOutputStrategy.TOOL:
            raise ValueError(
                "The reserved structured-output call requires matching durable config."
            )
        if (
            type(structured_output_attempt) is not int
            or not 1 <= structured_output_attempt <= structured_output.max_retries + 1
        ):
            raise ValueError("The pending structured-output round requires a valid attempt number.")
        try:
            tool_publication_rules._validate_durable_structured_output_validation(
                raw_structured_output_validation,
                "pending_tool_round.structured_output_validation",
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "The pending structured-output round requires authoritative validation."
            ) from exc
    elif structured_output_attempt is not None or raw_structured_output_validation is not None:
        raise ValueError(
            "An ordinary pending tool round cannot carry structured-output attempt evidence."
        )

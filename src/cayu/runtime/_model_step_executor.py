"""Complete context, accounting, retry, and provider-stream model-step ownership.

This module sits below :class:`CayuApp`: it never imports or accepts the
application facade.  The complete executor owns provider-facing request
construction, attachment resolution, context projection and recovery, budget
reservation settlement, retry isolation, and stream normalization. Session-loop
decisions and transcript commits stay with :class:`SessionEngine`.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from copy import deepcopy
from dataclasses import dataclass, field
from hashlib import sha256
from typing import Any, cast
from uuid import uuid4

from cayu._exception_groups import (
    add_exception_note_safely,
    exception_tree_contains,
    iter_exception_tree,
)
from cayu._task_wait import (
    await_shielded_task_outcome,
    consume_pending_task_cancellation,
    unexpected_child_cancellation_error,
)
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
    require_nonblank,
    safe_durable_value_error_details,
)
from cayu.artifacts import (
    RESOLVED_FILE_ATTACHMENTS_OPTION,
    FileAttachment,
    InvalidArtifactIdError,
    copy_artifact_read_result,
    file_attachment_from_payload,
    resolved_file_attachment,
)
from cayu.core.agents import AgentSpec
from cayu.core.billing import (
    BillingIdentity,
    resolved_billing_identity,
)
from cayu.core.events import Event, EventType, copy_event, event_with_runtime_payload_authority
from cayu.core.messages import (
    FilePart,
    Message,
    MessageRole,
    ProviderStatePart,
    ToolCallPart,
    ToolResultPart,
    detach_message,
)
from cayu.core.thinking import ThinkingConfig, thinking_config_payload
from cayu.providers import (
    InputTokenCountConfidence,
    InputTokenCountMethod,
    InputTokenCountResult,
    ModelCompletion,
    ModelContextOverflowError,
    ModelFinishReason,
    ModelProvider,
    ModelProviderError,
    ModelRequest,
    ModelStreamEvent,
    ModelStreamEventType,
    UsageDialect,
    copy_input_token_count_result,
    copy_model_context_pressure_profile,
    copy_model_stream_event,
    normalize_model_completion,
)
from cayu.providers.base import copy_model_completion
from cayu.runtime import _runtime_records as runtime_records
from cayu.runtime import _transcript as transcript_helpers
from cayu.runtime._completion_projection import portable_model_completion_projection
from cayu.runtime._event_writer import RuntimeEventWriter
from cayu.runtime._message_redaction import (
    redact_runtime_message_for_boundary,
    redact_untrusted_message_for_boundary,
)
from cayu.runtime._model_errors import (
    copy_provider_exception_control,
    model_provider_error_from_payload,
    nonportable_model_provider_error,
    resolve_completion_billing_identity,
    resolve_request_billing_identity,
)
from cayu.runtime._run_limits import (
    UNKNOWN_POST_DISPATCH_BUDGET_REASON,
    BudgetDispatchReservationFailed,
    BudgetedOperationRejected,
    BudgetedOperationSucceeded,
    BudgetEvaluation,
    BudgetModelStepLifecycle,
    BudgetReservationLeaseLost,
    BudgetReservationLeaseLostBeforeModelDispatch,
    LimitEvaluation,
    RunLimitController,
    RunLimitGate,
    SessionUsageTracker,
    add_budget_failure_note,
)
from cayu.runtime._session_control import (
    ActiveSessionRun,
    SessionControl,
    SessionInterruptedByRequest,
)
from cayu.runtime._structured_output_tool_round import (
    _redact_structured_output_validation,
    _validate_structured_output_tool_round,
)
from cayu.runtime.budgets import (
    BudgetLimit,
    BudgetPolicy,
    BudgetReservationResult,
    budget_limits_for_session,
    copy_request_budget_limits,
    has_deferred_contextual_price,
)
from cayu.runtime.context import (
    _COMPACTION_ATTEMPT_ID_KEY,
    CompactionRequest,
    CompactionResult,
    ContextBuildError,
    ContextCompactionTelemetry,
    ContextCompactor,
    ContextKnowledgeTelemetry,
    ContextPolicy,
    ContextPressureEstimate,
    ContextPressureOverhead,
    ContextRequest,
    ContextUsageState,
    RuntimeManagedContextPolicy,
    _automatic_compaction_dispatch_runner_scope,
    _automatic_compaction_runner_scope,
    _AutomaticCompactionRunner,
    _compaction_completion_publisher_scope,
    _compaction_model_attempt_identity_scope,
    _context_secret_redactor_scope,
    _defer_billing_identity_cancellation_scope,
    context_build_termination_compaction_telemetry,
    copy_context_messages,
    estimate_context_pressure,
    estimate_model_request_context_pressure,
    noteify_unresolvable_prompt_files,
    sanitize_context_build_error_checkpoint,
    sanitize_context_build_result_checkpoint,
    sanitize_context_compaction_telemetry,
)
from cayu.runtime.context_counting import ContextCountingConfig, ContextCountingMode
from cayu.runtime.execution_units import (
    ModelAttemptIdentity,
    ModelStepIdentity,
    ToolRoundIdentity,
    copy_model_attempt_identity,
    copy_model_step_identity,
    copy_tool_round_identity,
    strip_runtime_owned_execution_identity,
)
from cayu.runtime.model_steps import (
    AssistantStepResult,
    assistant_text_content,
    classify_assistant_step,
    provider_state_count,
    thinking_count,
)
from cayu.runtime.retry_policy import (
    RetryDecision,
    RetryPolicy,
    copy_retry_policy,
    retry_decision,
    retry_event_payload,
)
from cayu.runtime.sessions import (
    CheckpointTransform,
    EventOrder,
    EventQuery,
    ModelCompletionStage,
    ModelCompletionStageAbandonmentResult,
    ModelCompletionStageRequest,
    ModelCompletionStageResult,
    RuntimePublicationResult,
    Session,
    SessionStatus,
    SessionStore,
)
from cayu.runtime.structured_output import (
    STRUCTURED_OUTPUT_TOOL_NAME,
    StructuredOutputSpec,
    StructuredOutputStrategy,
    StructuredOutputValidation,
    copy_structured_output_spec,
    require_secret_free_json_schema_keys,
    require_secret_free_structured_output_spec,
    structured_output_spec_payload,
    structured_output_tool_instruction,
    structured_output_tool_spec,
)
from cayu.runtime.usage import (
    durable_model_completed_payload,
    is_conversational_model_completion_payload,
    normalize_usage_metrics,
    usage_metrics_from_event_payload,
    usage_metrics_payload,
)
from cayu.vaults import SecretRedactor

logger = logging.getLogger(__name__)


class ModelAttemptFailed(Exception):
    """A single provider attempt failed after zero or more streamed events.

    ``completion_observed`` is set only by the durable publication path. Once a
    valid completed frame has crossed that boundary, a later transport/control
    error is terminal and cannot authorize another provider dispatch.
    """

    def __init__(
        self,
        *,
        message: str,
        payload: dict[str, Any],
        emitted_error_event: bool,
        cause: Exception | None = None,
        completion_observed: bool = False,
    ) -> None:
        self.message = require_nonblank(message, "message")
        self.payload = copy_json_value(payload, "payload")
        self.emitted_error_event = emitted_error_event
        self.cause = cause
        self.completion_observed = completion_observed
        super().__init__(self.message)


def _copy_model_completion_stage(stage: ModelCompletionStage) -> ModelCompletionStage:
    if type(stage) is not ModelCompletionStage:
        raise TypeError("Model completion dispatch requires a ModelCompletionStage.")
    return stage.model_copy(deep=True)


def _copy_model_completion_stage_result(
    result: ModelCompletionStageResult,
) -> ModelCompletionStageResult:
    if type(result) is not ModelCompletionStageResult:
        raise TypeError("Model completion publication requires a ModelCompletionStageResult.")
    return result.model_copy(deep=True)


def _copy_runtime_publication_result(
    result: RuntimePublicationResult,
) -> RuntimePublicationResult:
    if type(result) is not RuntimePublicationResult:
        raise TypeError("Model completion publication requires a RuntimePublicationResult.")
    return result.model_copy(deep=True)


def _copy_assistant_step_result(result: AssistantStepResult) -> AssistantStepResult:
    if type(result) is not AssistantStepResult:
        raise TypeError("Model completion publication requires an AssistantStepResult.")
    completion = copy_model_completion(result.completion)
    if completion is None:  # pragma: no cover - AssistantStepResult requires a completion
        raise RuntimeError("Assistant step result lost its completion metadata.")
    return AssistantStepResult(
        session_id=result.session_id,
        step=result.step,
        model_step_id=result.model_step_id,
        model_attempt_id=result.model_attempt_id,
        tool_round_identity=(
            None
            if result.tool_round_identity is None
            else copy_tool_round_identity(result.tool_round_identity)
        ),
        assistant_message=(
            None if result.assistant_message is None else detach_message(result.assistant_message)
        ),
        tool_calls=[
            runtime_records.ToolCallRequest(
                id=call.id,
                name=call.name,
                arguments=copy_durable_json_object(
                    call.arguments,
                    "tool_call_arguments",
                ),
            )
            for call in result.tool_calls
        ],
        completion=completion,
        text_content=result.text_content,
        has_user_visible_content=result.has_user_visible_content,
        provider_state_count=result.provider_state_count,
        thinking_count=result.thinking_count,
    )


def _durable_assistant_step_result(
    result: AssistantStepResult,
    *,
    redactor: SecretRedactor,
) -> AssistantStepResult:
    """Project one assistant result across the durable publication boundary."""

    copied = _copy_assistant_step_result(result)
    if copied.assistant_message is None:
        if copied.tool_calls:
            raise ValueError("Assistant tool calls require an assistant message.")
        return copied
    assistant_message = redact_untrusted_message_for_boundary(
        copied.assistant_message,
        redactor=redactor,
        field_name="assistant_message",
    )
    tool_calls = [
        runtime_records.ToolCallRequest(
            id=call.id,
            name=call.name,
            arguments=redactor.redact_json_values(call.arguments),
        )
        for call in copied.tool_calls
    ]
    text_content = assistant_text_content(assistant_message)
    return AssistantStepResult(
        session_id=copied.session_id,
        step=copied.step,
        model_step_id=copied.model_step_id,
        model_attempt_id=copied.model_attempt_id,
        tool_round_identity=copied.tool_round_identity,
        assistant_message=assistant_message,
        tool_calls=tool_calls,
        completion=copied.completion,
        text_content=text_content,
        has_user_visible_content=bool(text_content.strip()),
        provider_state_count=provider_state_count(assistant_message),
        thinking_count=thinking_count(assistant_message),
    )


@dataclass(frozen=True, slots=True)
class ModelCompletionDispatch:
    """Detached proof that one exact provider dispatch was durably prepared."""

    stage: ModelCompletionStage
    request_fingerprint: str

    def __post_init__(self) -> None:
        stage = _copy_model_completion_stage(self.stage)
        request_fingerprint = require_durable_clean_nonblank(
            self.request_fingerprint,
            "request_fingerprint",
        )
        if len(request_fingerprint) != 64 or any(
            character not in "0123456789abcdef" for character in request_fingerprint
        ):
            raise ValueError("request_fingerprint must be a lowercase SHA-256 digest.")
        if stage.state != "in_flight":
            raise ValueError("A provider dispatch requires an in-flight completion stage.")
        if stage.intent.get("request_fingerprint") != request_fingerprint:
            raise ValueError("Completion-stage intent does not match its request fingerprint.")
        object.__setattr__(self, "stage", stage)
        object.__setattr__(self, "request_fingerprint", request_fingerprint)

    @property
    def stage_id(self) -> str:
        return self.stage.stage_id

    @property
    def logical_step_id(self) -> str:
        return self.stage.logical_step_id

    @property
    def dispatch_ordinal(self) -> int:
        return self.stage.dispatch_ordinal

    @property
    def reservation_ids(self) -> tuple[str, ...]:
        return self.stage.reservation_ids

    @property
    def intent(self) -> dict[str, Any]:
        return copy_durable_json_object(self.stage.intent, "model_completion_intent")


@dataclass(frozen=True, slots=True)
class ModelCompletionPublicationRequest:
    """Immutable, detached terminal material handed to the session owner."""

    dispatch: ModelCompletionDispatch
    assistant_step_result: AssistantStepResult | None
    completion_event: Event
    authoritative_assistant_message: Message | None
    structured_output_validation: StructuredOutputValidation | None

    def __post_init__(self) -> None:
        if type(self.dispatch) is not ModelCompletionDispatch:
            raise TypeError("dispatch must be a ModelCompletionDispatch.")
        dispatch = ModelCompletionDispatch(
            stage=self.dispatch.stage,
            request_fingerprint=self.dispatch.request_fingerprint,
        )
        result = (
            None
            if self.assistant_step_result is None
            else _copy_assistant_step_result(self.assistant_step_result)
        )
        event = copy_event(self.completion_event)
        assistant_message = (
            None
            if self.authoritative_assistant_message is None
            else detach_message(self.authoritative_assistant_message)
        )
        if (
            self.structured_output_validation is not None
            and type(self.structured_output_validation) is not StructuredOutputValidation
        ):
            raise TypeError("structured_output_validation must be a StructuredOutputValidation.")
        structured_output_validation = (
            None
            if self.structured_output_validation is None
            else self.structured_output_validation.model_copy(deep=True)
        )
        if result is not None and result.session_id != dispatch.stage.session_id:
            raise ValueError("Assistant result session does not match its completion stage.")
        if event.session_id != dispatch.stage.session_id:
            raise ValueError("Completion event session does not match its completion stage.")
        if event.type != EventType.MODEL_COMPLETED:
            raise ValueError("Completion publication requires a model.completed event.")
        if assistant_message is not None:
            if result is None:
                raise ValueError(
                    "An authoritative assistant message requires an assistant step result."
                )
            if assistant_message != result.assistant_message:
                raise ValueError(
                    "Authoritative assistant message does not match the detached step result."
                )
        if structured_output_validation is not None and (
            result is None
            or assistant_message is None
            or not any(call.name == STRUCTURED_OUTPUT_TOOL_NAME for call in result.tool_calls)
        ):
            raise ValueError(
                "Structured-output validation requires a published finalizer tool round."
            )
        object.__setattr__(self, "dispatch", dispatch)
        object.__setattr__(self, "assistant_step_result", result)
        object.__setattr__(self, "completion_event", event)
        object.__setattr__(self, "authoritative_assistant_message", assistant_message)
        object.__setattr__(
            self,
            "structured_output_validation",
            structured_output_validation,
        )


@dataclass(frozen=True, slots=True)
class ModelCompletionPublicationResult:
    """Durable terminal-stage and atomic-promotion acknowledgements."""

    completion: ModelCompletionStageResult
    publication: RuntimePublicationResult

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "completion",
            _copy_model_completion_stage_result(self.completion),
        )
        object.__setattr__(
            self,
            "publication",
            _copy_runtime_publication_result(self.publication),
        )


ModelCompletionPublisher = Callable[
    [ModelCompletionPublicationRequest],
    Awaitable[ModelCompletionPublicationResult],
]


class ModelCompletionDispatchNotAuthorized(RuntimeError):
    """A prior or ambiguous preparation must never cause another provider call."""

    def __init__(
        self,
        *,
        stage: ModelCompletionStage,
        request_fingerprint: str,
    ) -> None:
        self.stage = _copy_model_completion_stage(stage)
        self.request_fingerprint = require_durable_clean_nonblank(
            request_fingerprint,
            "request_fingerprint",
        )
        super().__init__(
            "Model completion dispatch was not authorized because its durable "
            f"stage already exists: {stage.stage_id}."
        )


def _model_step_logical_id(
    *,
    session_id: str,
    source_transcript_cursor: int,
) -> str:
    identity = canonical_durable_json_bytes(
        {
            "schema_version": 1,
            "purpose": "assistant-turn",
            "session_id": session_id,
            "source_transcript_cursor": source_transcript_cursor,
        },
        "model_step_identity",
    )
    return f"model-step:v1:{sha256(identity).hexdigest()}"


def _model_request_fingerprint(
    *,
    provider_name: str,
    model_request: ModelRequest,
) -> str:
    material = canonical_durable_json_bytes(
        {
            "schema_version": 1,
            "provider_name": provider_name,
            "request": model_request.model_dump(mode="json"),
        },
        "model_request_fingerprint",
    )
    return sha256(material).hexdigest()


def _model_completion_stage_intent(
    *,
    model_attempt_identity: ModelAttemptIdentity,
    provider_name: str,
    requested_model: str,
    source_transcript_cursor: int,
    request_fingerprint: str,
) -> dict[str, Any]:
    model_attempt_identity = copy_model_attempt_identity(model_attempt_identity)
    return {
        "schema_version": 1,
        "purpose": "assistant-turn",
        **model_attempt_identity.payload(),
        "logical_step_id": model_attempt_identity.model_step_id,
        "provider_name": provider_name,
        "requested_model": requested_model,
        "source_transcript_cursor": source_transcript_cursor,
        "request_fingerprint": request_fingerprint,
    }


def _non_turn_model_completion_event(
    event: Event,
    *,
    failure: BaseException,
    cancellation: asyncio.CancelledError | None,
    transcript_cursor: int,
) -> Event:
    payload = copy_durable_json_object(event.payload, "model_completion_payload")
    if cancellation is not None:
        reason = "model stream was cancelled before terminal validation completed"
    elif isinstance(failure, SessionInterruptedByRequest):
        reason = "session interruption won before terminal validation completed"
    elif isinstance(failure, ModelAttemptFailed):
        reason = "provider emitted an invalid event after model completion"
    else:
        reason = "model completion failed terminal validation"
    payload["step_classification"] = {
        "type": "failed",
        "reason": reason,
    }
    payload["transcript_cursor"] = transcript_cursor
    return copy_event(event).model_copy(update={"payload": payload}, deep=True)


def _validate_model_completion_publication_result(
    request: ModelCompletionPublicationRequest,
    result: ModelCompletionPublicationResult,
) -> None:
    if type(result) is not ModelCompletionPublicationResult:
        raise TypeError("Model completion publisher must return ModelCompletionPublicationResult.")
    detached_result = ModelCompletionPublicationResult(
        completion=result.completion,
        publication=result.publication,
    )
    prepared = request.dispatch.stage
    completed = detached_result.completion.stage
    for field_name in (
        "session_id",
        "stage_id",
        "logical_step_id",
        "dispatch_ordinal",
        "purpose",
        "intent",
        "reservation_ids",
        "preparation_request_digest",
        "preparation_digest",
        "source_status",
        "source_run_epoch",
        "source_transcript_cursor",
        "prepared_at",
    ):
        if getattr(completed, field_name) != getattr(prepared, field_name):
            raise RuntimeError(
                "Model completion publisher acknowledged a different prepared "
                f"stage field: {field_name}."
            )
    if completed.state != "completed" or completed.publication is None:
        raise RuntimeError("Model completion publisher did not return a terminal stage.")

    publication_request = completed.publication
    expected_messages = (
        ()
        if request.authoritative_assistant_message is None
        else (request.authoritative_assistant_message,)
    )
    if publication_request.publication_id != request.dispatch.logical_step_id:
        raise RuntimeError("Model completion publication acknowledged a different logical step.")
    if publication_request.kind != "model-step":
        raise RuntimeError("Model completion publication has the wrong publication kind.")
    if publication_request.intent != request.dispatch.intent:
        raise RuntimeError("Model completion publication changed the prepared dispatch intent.")
    if publication_request.transcript_messages != expected_messages:
        raise RuntimeError("Model completion publication changed the authoritative assistant turn.")
    if publication_request.events != (request.completion_event,):
        raise RuntimeError("Model completion publication changed its completion event.")
    pending_round_operations = [
        operation
        for operation in publication_request.mutation.operations
        if operation.key == "pending_tool_round"
    ]
    durable_validation = None
    if pending_round_operations:
        if len(pending_round_operations) != 1:
            raise RuntimeError(
                "Model completion publication changed its pending tool-round mutation."
            )
        pending_round_value = pending_round_operations[0].value
        if type(pending_round_value) is not dict:
            raise RuntimeError(
                "Model completion publication returned a malformed pending tool round."
            )
        durable_validation = pending_round_value.get("structured_output_validation")
    expected_validation = (
        None
        if request.structured_output_validation is None
        else request.structured_output_validation.model_dump(mode="json")
    )
    if durable_validation != expected_validation:
        raise RuntimeError("Model completion publication changed its structured-output validation.")

    promoted = detached_result.publication
    receipt = promoted.receipt
    if promoted.session.id != prepared.session_id:
        raise RuntimeError("Model completion publication returned a different session.")
    if receipt.session_id != prepared.session_id:
        raise RuntimeError("Model completion receipt belongs to a different session.")
    if receipt.publication_id != request.dispatch.logical_step_id:
        raise RuntimeError("Model completion receipt belongs to a different logical step.")
    if receipt.kind != "model-step":
        raise RuntimeError("Model completion receipt has the wrong publication kind.")
    if receipt.appended_event_ids != (request.completion_event.id,):
        raise RuntimeError("Model completion receipt does not bind the exact completion event.")
    if receipt.transcript_start_cursor != prepared.source_transcript_cursor:
        raise RuntimeError("Model completion publication started at a different transcript cursor.")
    if receipt.transcript_end_cursor != (
        prepared.source_transcript_cursor + len(expected_messages)
    ):
        raise RuntimeError("Model completion publication ended at a different transcript cursor.")


async def _publish_model_completion(
    publisher: ModelCompletionPublisher,
    request: ModelCompletionPublicationRequest,
    *,
    terminal_failure: BaseException | None,
    publication_cancellation: asyncio.CancelledError | None,
) -> None:
    expected_request = request
    callback_request = ModelCompletionPublicationRequest(
        dispatch=request.dispatch,
        assistant_step_result=request.assistant_step_result,
        completion_event=request.completion_event,
        authoritative_assistant_message=request.authoritative_assistant_message,
        structured_output_validation=request.structured_output_validation,
    )
    if publication_cancellation is not None:
        assert terminal_failure is not None

        async def publish() -> ModelCompletionPublicationResult:
            return await publisher(callback_request)

        publication_task = asyncio.create_task(publish())
        outcome = await await_shielded_task_outcome(
            publication_task,
            cancellation=publication_cancellation,
        )
        if outcome.error is not None:
            callback_error = outcome.error
            if isinstance(callback_error, asyncio.CancelledError):
                callback_error = unexpected_child_cancellation_error(
                    callback_error,
                    operation="model completion publication",
                )
            add_exception_note_safely(
                terminal_failure,
                "Model completion publication also failed while preserving cancellation: "
                f"{type(callback_error).__name__}: {callback_error}",
            )
        elif outcome.result is None:
            add_exception_note_safely(
                terminal_failure,
                (
                    "Model completion publication returned no acknowledgement while preserving "
                    "cancellation."
                ),
            )
        else:
            try:
                _validate_model_completion_publication_result(
                    expected_request,
                    outcome.result,
                )
            except BaseException as validation_error:
                add_exception_note_safely(
                    terminal_failure,
                    (
                        "Model completion publication acknowledgement was invalid while "
                        "preserving cancellation: "
                        f"{type(validation_error).__name__}: {validation_error}"
                    ),
                )
        if terminal_failure is publication_cancellation or (
            isinstance(terminal_failure, BaseExceptionGroup)
            and any(
                candidate is publication_cancellation
                for candidate in iter_exception_tree(terminal_failure)
            )
        ):
            raise terminal_failure
        add_exception_note_safely(
            publication_cancellation,
            "The provider suppressed caller cancellation before raising "
            f"{type(terminal_failure).__name__}.",
        )
        raise publication_cancellation from terminal_failure

    try:
        result = await publisher(callback_request)
        _validate_model_completion_publication_result(expected_request, result)
    except BaseException as publication_error:
        if terminal_failure is not None:
            publication_error.add_note(
                "The provider stream had already reached a terminal failure after emitting "
                "completion evidence."
            )
        raise


def _take_model_completion_cancellation(
    failure: BaseException | None,
    *,
    cancellation_baseline: int,
) -> asyncio.CancelledError | None:
    """Take caller cancellation newer than the provider-boundary baseline."""

    task = asyncio.current_task()
    if task is None or task.cancelling() <= cancellation_baseline:
        return None
    cancellation = next(
        (
            candidate
            for candidate in (() if failure is None else iter_exception_tree(failure))
            if isinstance(candidate, asyncio.CancelledError)
        ),
        None,
    )
    return consume_pending_task_cancellation(
        cancellation,
        preserve_requests=cancellation_baseline,
    )


def _combine_post_completion_failures(
    current: BaseException | None,
    subsequent: BaseException,
) -> BaseException:
    if current is None or current is subsequent:
        return subsequent
    return BaseExceptionGroup(
        "Model provider iteration and iterator cleanup both failed after completion.",
        [current, subsequent],
    )


@dataclass(frozen=True)
class _ContextCountObservation:
    result: InputTokenCountResult
    observation_id: str


@dataclass(frozen=True)
class _ModelStreamBoundaryValue:
    event: ModelStreamEvent
    completion_error: DurableValueError | None = None
    accounting_usage_metrics: dict[str, Any] | None = None
    accounting_usage_rejected: bool = False
    usage_normalization_failed: bool = False


@dataclass(frozen=True)
class _ContextPressureObservation:
    estimate: ContextPressureEstimate
    observation_id: str


def _context_observation_event(event: Event) -> Event:
    """Attest the runtime identities shared by context-observation events."""

    return event_with_runtime_payload_authority(
        event,
        "observation_id",
        "model_step_id",
        "model_attempt_id",
    )


def _event_with_model_identity_authority(
    event: Event,
    identity: ModelStepIdentity | ModelAttemptIdentity,
) -> Event:
    """Attest model execution linkage supplied by a typed runtime identity."""

    if type(identity) is ModelAttemptIdentity:
        payload = copy_model_attempt_identity(identity).payload()
    elif type(identity) is ModelStepIdentity:
        payload = copy_model_step_identity(identity).payload()
    else:
        raise TypeError("Model event identity has an unsupported type.")
    fields = [
        field_name
        for field_name, value in payload.items()
        if event.payload.get(field_name) == value
    ]
    return event_with_runtime_payload_authority(event, *fields) if fields else event


@dataclass
class _CompactionExecutionIdentityLedger:
    """Bind internal compaction evidence to pre-dispatch runtime identities."""

    model_step_identity: ModelStepIdentity
    active_model_attempt_identity: ModelAttemptIdentity | None = None
    model_attempts_by_compaction_id: dict[str, ModelAttemptIdentity] = field(default_factory=dict)
    compaction_ids_by_model_attempt_id: dict[str, str] = field(default_factory=dict)
    issued_model_attempts_by_id: dict[str, ModelAttemptIdentity] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.model_step_identity = copy_model_step_identity(self.model_step_identity)

    def begin_dispatch(self) -> ModelAttemptIdentity:
        if self.active_model_attempt_identity is not None:
            raise RuntimeError("Compaction provider dispatches cannot overlap.")
        identity = self.model_step_identity.new_attempt()
        self.active_model_attempt_identity = identity
        self.issued_model_attempts_by_id[identity.model_attempt_id] = identity
        return copy_model_attempt_identity(identity)

    def end_dispatch(self, identity: ModelAttemptIdentity) -> None:
        identity = copy_model_attempt_identity(identity)
        if self.active_model_attempt_identity != identity:
            raise RuntimeError("Compaction provider dispatch identity was not active.")
        self.active_model_attempt_identity = None

    def identify_payloads(
        self,
        payloads: list[dict[str, Any]],
        *,
        expected_identity: ModelAttemptIdentity | None = None,
    ) -> list[dict[str, Any]]:
        expected = (
            None if expected_identity is None else copy_model_attempt_identity(expected_identity)
        )
        identified_payloads: list[dict[str, Any]] = []
        for raw_payload in payloads:
            payload = copy_durable_json_object(
                raw_payload,
                "compaction_model_completed_payload",
            )
            compaction_id = payload.get(_COMPACTION_ATTEMPT_ID_KEY)
            if type(compaction_id) is not str:
                raise RuntimeError("Compaction completion evidence lost its attempt identity.")
            identity = self.model_attempts_by_compaction_id.get(compaction_id)
            candidate = expected or self.active_model_attempt_identity
            payload_identity: ModelAttemptIdentity | None = None
            if "model_step_id" in payload or "model_attempt_id" in payload:
                try:
                    payload_identity = ModelAttemptIdentity.model_validate(
                        {
                            "model_step_id": payload.get("model_step_id"),
                            "model_attempt_id": payload.get("model_attempt_id"),
                        }
                    )
                except (TypeError, ValueError):
                    raise ValueError(
                        "Compaction completion carries an invalid model attempt identity."
                    ) from None
                issued_identity = self.issued_model_attempts_by_id.get(
                    payload_identity.model_attempt_id
                )
                if issued_identity != payload_identity:
                    raise ValueError(
                        "Compaction completion carries a model attempt identity that "
                        "was not issued for this logical step."
                    )
            if (
                candidate is not None
                and payload_identity is not None
                and candidate != payload_identity
            ):
                raise ValueError(
                    "Compaction completion identity conflicts with its provider dispatch."
                )
            candidate = candidate or payload_identity
            if identity is None:
                if candidate is None:
                    raise RuntimeError(
                        "Compaction completion was observed outside its provider dispatch."
                    )
                existing_compaction_id = self.compaction_ids_by_model_attempt_id.get(
                    candidate.model_attempt_id
                )
                if existing_compaction_id is not None and existing_compaction_id != compaction_id:
                    raise ValueError(
                        "Compaction provider dispatch produced conflicting completion identities."
                    )
                identity = copy_model_attempt_identity(candidate)
                self.model_attempts_by_compaction_id[compaction_id] = identity
            elif candidate is not None and identity != candidate:
                raise ValueError(
                    "Compaction completion identity conflicts with its provider dispatch."
                )
            existing_compaction_id = self.compaction_ids_by_model_attempt_id.setdefault(
                identity.model_attempt_id,
                compaction_id,
            )
            if existing_compaction_id != compaction_id:
                raise ValueError(
                    "Compaction provider dispatch produced conflicting completion identities."
                )
            payload.update(identity.payload())
            identified_payloads.append(payload)
        return identified_payloads


@dataclass(frozen=True)
class ModelStepFlowOutcome:
    """Terminal outcome of one logical model step."""

    assistant_step_result: AssistantStepResult | None = None
    stop_session: bool = False

    def __post_init__(self) -> None:
        if self.stop_session == (self.assistant_step_result is not None):
            raise ValueError(
                "A model-step flow outcome must contain either a result or a stop signal."
            )


_CONTEXT_TERMINATION_PERSIST_TIMEOUT_S = 5.0
_CONTEXT_EVENT_STORE_WAIT_TIMEOUT_S = 5.0
_CONTEXT_USAGE_AUXILIARY_PAGE_SIZE = 100


def _consume_detached_task_outcome(task: asyncio.Task[Any]) -> None:
    """Retrieve a timed-out task's eventual result after requesting cancellation."""

    with contextlib.suppress(asyncio.CancelledError):
        task.exception()


@dataclass(frozen=True)
class ModelStepBudgetEvaluationRequest:
    evaluation: BudgetEvaluation
    session: Session
    registered_agent: runtime_records.RegisteredAgentState
    registered_environment: runtime_records.RegisteredEnvironment | None
    environment_name: str | None
    messages: list[Message]
    run_started_at: float
    turn_usage_tracker: SessionUsageTracker | None
    active_run: ActiveSessionRun[SessionUsageTracker] | None


@dataclass(frozen=True)
class ModelStepLimitEvaluationRequest:
    evaluation: LimitEvaluation
    session: Session
    registered_agent: runtime_records.RegisteredAgentState
    registered_environment: runtime_records.RegisteredEnvironment | None
    environment_name: str | None
    messages: list[Message]
    run_started_at: float
    turn_usage_tracker: SessionUsageTracker | None
    active_run: ActiveSessionRun[SessionUsageTracker] | None


@dataclass(frozen=True)
class ModelStepBudgetReservationFailureRequest:
    result: BudgetReservationResult
    session: Session
    registered_agent: runtime_records.RegisteredAgentState
    registered_environment: runtime_records.RegisteredEnvironment | None
    environment_name: str | None
    messages: list[Message]
    run_started_at: float
    turn_usage_tracker: SessionUsageTracker | None
    active_run: ActiveSessionRun[SessionUsageTracker] | None


BudgetEvaluationEventStream = Callable[
    [ModelStepBudgetEvaluationRequest],
    AsyncIterator[Event],
]
LimitEvaluationEventStream = Callable[
    [ModelStepLimitEvaluationRequest],
    AsyncIterator[Event],
]
BudgetReservationFailureEventStream = Callable[
    [ModelStepBudgetReservationFailureRequest],
    AsyncIterator[Event],
]
CheckpointTransformFactory = Callable[[dict[str, Any]], CheckpointTransform]


class _AutomaticCompactionBudgetReservationFailed(RuntimeError):
    def __init__(self, result: BudgetReservationResult) -> None:
        super().__init__(f"Context compaction budget reservation failed: {result.message}")
        self.result = result


class _AutomaticCompactionAdmissionStopped(RuntimeError):
    """The session was stopped by a limit before a compactor provider dispatch."""

    def __init__(
        self,
        *,
        budget_evaluation: BudgetEvaluation | None = None,
        limit_evaluation: LimitEvaluation | None = None,
    ) -> None:
        if (budget_evaluation is None) == (limit_evaluation is None):
            raise ValueError(
                "Automatic compaction admission must contain one rejecting evaluation."
            )
        self.budget_evaluation = budget_evaluation
        self.limit_evaluation = limit_evaluation
        super().__init__("Automatic compaction provider dispatch was stopped by a limit.")


class ModelStepExecutor:
    """Build and execute provider requests for one logical model step."""

    def __init__(
        self,
        *,
        session_store: SessionStore,
        event_writer: RuntimeEventWriter,
        session_control: SessionControl[SessionUsageTracker],
        run_limit_controller: RunLimitController,
        context_counting: ContextCountingConfig,
        max_file_attachment_bytes: int,
        max_total_file_attachment_bytes: int,
        max_file_attachments_per_request: int,
        secret_redactor: SecretRedactor,
        checkpoint_transform: CheckpointTransformFactory,
        apply_budget_evaluation: BudgetEvaluationEventStream,
        apply_limit_evaluation: LimitEvaluationEventStream,
        stop_for_budget_reservation_failure: BudgetReservationFailureEventStream,
    ) -> None:
        self._session_store = session_store
        self._event_writer = event_writer
        self._session_control = session_control
        self._run_limit_controller = run_limit_controller
        self._context_counting = context_counting.model_copy(deep=True)
        self._max_file_attachment_bytes = max_file_attachment_bytes
        self._max_total_file_attachment_bytes = max_total_file_attachment_bytes
        self._max_file_attachments_per_request = max_file_attachments_per_request
        self._secret_redactor = secret_redactor
        self._checkpoint_transform = checkpoint_transform
        self._apply_budget_evaluation = apply_budget_evaluation
        self._apply_limit_evaluation = apply_limit_evaluation
        self._stop_for_budget_reservation_failure = stop_for_budget_reservation_failure

    def create_run(
        self,
        *,
        provider: ModelProvider,
        session: Session,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_provider: runtime_records.RegisteredProvider,
        registered_environment: runtime_records.RegisteredEnvironment | None,
        environment_name: str | None,
        structured_output: StructuredOutputSpec | None,
        thinking: ThinkingConfig | None,
        knowledge_store: Any,
        request_metadata: dict[str, Any],
        retry_policy: RetryPolicy,
        request_budget_limits: tuple[BudgetLimit, ...],
        limit_gate: RunLimitGate,
        budget_policy: BudgetPolicy | None,
        run_started_at: float,
        turn_usage_tracker: SessionUsageTracker | None,
        active_run: ActiveSessionRun[SessionUsageTracker] | None,
        model_completion_publisher: ModelCompletionPublisher | None = None,
    ) -> ModelStepRun:
        return ModelStepRun(
            self,
            provider=provider,
            session=session,
            registered_agent=registered_agent,
            registered_provider=registered_provider,
            registered_environment=registered_environment,
            environment_name=environment_name,
            structured_output=structured_output,
            thinking=thinking,
            knowledge_store=knowledge_store,
            request_metadata=request_metadata,
            retry_policy=retry_policy,
            request_budget_limits=request_budget_limits,
            limit_gate=limit_gate,
            budget_policy=budget_policy,
            run_started_at=run_started_at,
            turn_usage_tracker=turn_usage_tracker,
            active_run=active_run,
            model_completion_publisher=model_completion_publisher,
        )

    async def build_request(
        self,
        *,
        session: Session,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_environment: runtime_records.RegisteredEnvironment | None,
        context_messages: list[Message],
        structured_output: StructuredOutputSpec | None,
        thinking: ThinkingConfig | None,
        step: int,
    ) -> ModelRequest:
        model_tools = [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": deepcopy(tool.schema),
            }
            for tool in registered_agent.tools.values()
        ]
        model_messages = context_messages
        if (
            structured_output is not None
            and structured_output.strategy == StructuredOutputStrategy.TOOL
        ):
            model_tools.append(structured_output_tool_spec(structured_output))
            model_messages = _with_structured_output_tool_instruction(
                context_messages,
                structured_output,
            )

        resolved_attachments, unresolvable_prompt_ids = await _resolved_file_attachments(
            messages=model_messages,
            session=session,
            registered_environment=registered_environment,
            max_file_attachment_bytes=self._max_file_attachment_bytes,
            max_total_file_attachment_bytes=self._max_total_file_attachment_bytes,
            max_file_attachments_per_request=self._max_file_attachments_per_request,
        )
        if unresolvable_prompt_ids:
            model_messages = noteify_unresolvable_prompt_files(
                model_messages,
                unresolvable_prompt_ids,
            )
            logger.warning(
                "Prompt file attachment(s) could not be resolved and were omitted from the "
                "provider request (check the session_id used at attach time, or whether the "
                "artifact still exists): %s",
                ", ".join(sorted(unresolvable_prompt_ids)),
            )

        provider_options = copy_json_value(
            registered_agent.spec.provider_options,
            "provider_options",
        )
        if type(provider_options) is not dict:
            raise AssertionError("Agent provider options copied as a non-object.")
        agent_metadata = deepcopy(registered_agent.spec.metadata)
        environment_metadata = (
            deepcopy(registered_environment.spec.metadata)
            if registered_environment is not None
            else {}
        )
        structured_output_payload = (
            structured_output_spec_payload(structured_output)
            if structured_output is not None
            else None
        )
        thinking_payload = thinking_config_payload(thinking) if thinking is not None else None
        request_options: dict[str, Any] = {
            **provider_options,
            "agent_metadata": agent_metadata,
            "environment_metadata": environment_metadata,
            "step": step,
            "structured_output": structured_output_payload,
            RESOLVED_FILE_ATTACHMENTS_OPTION: resolved_attachments,
        }
        if thinking_payload is not None:
            request_options["thinking"] = thinking_payload
        redacted_messages = [
            redact_runtime_message_for_boundary(
                message,
                redactor=self._secret_redactor,
                field_name="model_message",
            )
            for message in model_messages
        ]
        if self._secret_redactor.redact_text(session.model) != session.model:
            raise ValueError(
                "Model identity contains a workload secret and cannot be sent to a provider."
            )
        for index, tool in enumerate(model_tools):
            tool_name = tool.get("name")
            if type(tool_name) is str and self._secret_redactor.redact_text(tool_name) != tool_name:
                raise ValueError(
                    f"model_tools[{index}].name contains a workload secret and cannot "
                    "be sent as provider execution authority."
                )
            self._secret_redactor.require_no_secret_keys(
                {
                    "name": tool.get("name"),
                    "description": tool.get("description"),
                    "input_schema": None,
                },
                field_name=f"model_tools[{index}]",
                preserve_keys={"name", "description", "input_schema"},
                match_short_substrings=True,
            )
            input_schema = tool.get("input_schema")
            if type(input_schema) is not dict:
                raise AssertionError("A model tool input_schema must be an object.")
            require_secret_free_json_schema_keys(
                input_schema,
                redactor=self._secret_redactor,
                field_name=f"model_tools[{index}].input_schema",
            )
        redacted_tools = [
            self._secret_redactor.redact_json_values(
                tool,
                preserve_string_fields={"name"},
            )
            for tool in model_tools
        ]
        for field_name, untyped_value in (
            ("provider_options", provider_options),
            ("agent_metadata", agent_metadata),
            ("environment_metadata", environment_metadata),
        ):
            self._secret_redactor.require_no_secret_keys(
                untyped_value,
                field_name=f"model_request_options.{field_name}",
                match_short_substrings=True,
            )
        require_secret_free_structured_output_spec(
            structured_output,
            redactor=self._secret_redactor,
            field_name="model_request_options.structured_output",
        )
        if (
            thinking_payload is not None
            and self._secret_redactor.redact_json_values(thinking_payload) != thinking_payload
        ):
            raise ValueError(
                "model_request_options.thinking contains a workload secret and cannot "
                "be sent without changing execution semantics."
            )
        self._secret_redactor.require_no_secret_keys(
            resolved_attachments,
            field_name=f"model_request_options.{RESOLVED_FILE_ATTACHMENTS_OPTION}",
            preserve_keys={
                "artifact_id",
                "kind",
                "filename",
                "content_type",
                "data_base64",
                "metadata",
            },
            untrusted_container_keys={"metadata"},
            match_short_substrings=True,
        )
        for artifact_id, attachment in resolved_attachments.items():
            stored_artifact_id = attachment.get("artifact_id")
            if self._secret_redactor.redact_text(artifact_id) != artifact_id or (
                type(stored_artifact_id) is str
                and self._secret_redactor.redact_text(stored_artifact_id) != stored_artifact_id
            ):
                raise ValueError(
                    "Resolved file attachment authority contains a workload secret "
                    "and cannot be sent to a provider."
                )
        redacted_options = self._secret_redactor.redact_json_values(
            request_options,
        )
        if type(redacted_options) is not dict:
            raise AssertionError("Model request-option redaction returned a non-object.")
        return ModelRequest(
            model=session.model,
            messages=redacted_messages,
            tools=redacted_tools,
            options=redacted_options,
        )

    async def run_with_retries(
        self,
        *,
        provider: ModelProvider,
        model_request: ModelRequest,
        session: Session,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_provider: runtime_records.RegisteredProvider,
        environment_name: str | None,
        step: int,
        model_step_identity: ModelStepIdentity,
        initial_model_attempt_identity: ModelAttemptIdentity | None,
        retry_policy: RetryPolicy,
        transcript_cursor_before_request: int,
        record_model_completion: Callable[[Event], Event],
        prepare_provider_dispatch: Callable[
            [ModelAttemptIdentity],
            Awaitable[tuple[list[Event], BudgetReservationResult | None, Exception | None]],
        ],
        before_provider_dispatch: Callable[[ModelAttemptIdentity], Awaitable[None]],
        record_model_attempt_identity: Callable[[ModelAttemptIdentity], None],
        billing_identity: BillingIdentity | None = None,
        structured_output: StructuredOutputSpec | None = None,
        prepare_model_completion_dispatch: Callable[
            [ModelRequest],
            Awaitable[ModelCompletionDispatch],
        ]
        | None = None,
        model_completion_publisher: ModelCompletionPublisher | None = None,
    ) -> AsyncIterator[tuple[Event | None, AssistantStepResult | None]]:
        retry_policy = copy_retry_policy(retry_policy)
        structured_output = copy_structured_output_spec(structured_output)
        model_step_identity = copy_model_step_identity(model_step_identity)
        next_model_attempt_identity = (
            None
            if initial_model_attempt_identity is None
            else copy_model_attempt_identity(initial_model_attempt_identity)
        )
        if (
            next_model_attempt_identity is not None
            and next_model_attempt_identity.model_step_id != model_step_identity.model_step_id
        ):
            raise ValueError("Initial model attempt belongs to a different logical step.")
        attempt = 1
        prior_retry_failure: ModelAttemptFailed | None = None
        while True:
            model_attempt_identity = (
                model_step_identity.new_attempt()
                if next_model_attempt_identity is None
                else next_model_attempt_identity
            )
            next_model_attempt_identity = None
            record_model_attempt_identity(copy_model_attempt_identity(model_attempt_identity))
            try:
                (
                    reservation_events,
                    reservation_failure,
                    preparation_error,
                ) = await prepare_provider_dispatch(model_attempt_identity)
            except Exception as accounting_exc:
                reservation_events = []
                reservation_failure = None
                preparation_error = accounting_exc
            for reservation_event in reservation_events:
                yield reservation_event, None
            if preparation_error is not None:
                if prior_retry_failure is None:
                    raise preparation_error
                authoritative_failure = prior_retry_failure.cause
                if authoritative_failure is None:
                    authoritative_failure = RuntimeError(prior_retry_failure.message)
                add_budget_failure_note(
                    authoritative_failure,
                    operation="retry preparation",
                    accounting_failure=preparation_error,
                )
                raise authoritative_failure from prior_retry_failure
            prior_retry_failure = None
            if reservation_failure is not None:
                raise BudgetDispatchReservationFailed(reservation_failure)

            # Never hand the retry template to provider-controlled code. Each
            # attempt gets a fully detached, revalidated request so provider
            # mutation cannot corrupt a later attempt.
            attempt_model_request = _detach_model_request(model_request)

            (
                context_pressure_observation,
                context_pressure_event,
            ) = await self._observe_context_pressure(
                model_request=attempt_model_request,
                session=session,
                registered_agent=registered_agent,
                registered_provider=registered_provider,
                environment_name=environment_name,
                step=step,
                attempt=attempt,
                max_attempts=retry_policy.max_attempts,
                model_attempt_identity=model_attempt_identity,
            )
            if context_pressure_event is not None:
                yield context_pressure_event, None
            context_count_observation, context_count_event = await self._observe_context_count(
                provider=provider,
                model_request=attempt_model_request,
                session=session,
                registered_agent=registered_agent,
                registered_provider=registered_provider,
                environment_name=environment_name,
                step=step,
                attempt=attempt,
                max_attempts=retry_policy.max_attempts,
                model_attempt_identity=model_attempt_identity,
            )
            if context_count_event is not None:
                yield context_count_event, None
            yield (
                await self._event_writer.emit(
                    _event_with_model_identity_authority(
                        Event(
                            type=EventType.MODEL_STARTED,
                            session_id=session.id,
                            agent_name=registered_agent.spec.name,
                            payload={
                                "model": session.model,
                                "provider": registered_provider.name,
                                "step": step,
                                "attempt": attempt,
                                "max_attempts": retry_policy.max_attempts,
                                **model_attempt_identity.payload(),
                            },
                            environment_name=environment_name,
                        ),
                        model_attempt_identity,
                    )
                ),
                None,
            )
            attempt_events = self._run_once(
                provider=provider,
                model_request=attempt_model_request,
                session=session,
                registered_agent=registered_agent,
                registered_provider=registered_provider,
                environment_name=environment_name,
                step=step,
                attempt=attempt,
                max_attempts=retry_policy.max_attempts,
                model_attempt_identity=model_attempt_identity,
                transcript_cursor_before_request=transcript_cursor_before_request,
                record_model_completion=record_model_completion,
                before_provider_dispatch=before_provider_dispatch,
                billing_identity=billing_identity,
                structured_output=structured_output,
                prepare_model_completion_dispatch=prepare_model_completion_dispatch,
                model_completion_publisher=model_completion_publisher,
            )
            try:
                result: AssistantStepResult | None = None
                async for event, step_result in attempt_events:
                    if event is not None:
                        yield event, None
                        if (
                            event.type == EventType.MODEL_COMPLETED
                            and context_pressure_observation is not None
                        ):
                            yield (
                                await self._event_writer.emit(
                                    _context_pressure_reconciled_event(
                                        event,
                                        observation=context_pressure_observation,
                                        session=session,
                                        registered_agent=registered_agent,
                                        registered_provider=registered_provider,
                                        environment_name=environment_name,
                                        step=step,
                                        attempt=attempt,
                                        max_attempts=retry_policy.max_attempts,
                                        model_attempt_identity=model_attempt_identity,
                                    )
                                ),
                                None,
                            )
                        if (
                            event.type == EventType.MODEL_COMPLETED
                            and context_count_observation is not None
                        ):
                            yield (
                                await self._event_writer.emit(
                                    _context_count_reconciled_event(
                                        event,
                                        observation=context_count_observation,
                                        session=session,
                                        registered_agent=registered_agent,
                                        registered_provider=registered_provider,
                                        environment_name=environment_name,
                                        step=step,
                                        attempt=attempt,
                                        max_attempts=retry_policy.max_attempts,
                                        model_attempt_identity=model_attempt_identity,
                                    )
                                ),
                                None,
                            )
                    if step_result is not None:
                        result = step_result
                if result is None:
                    raise RuntimeError("Model step finished without a result.")
                yield None, result
                return
            except ModelAttemptFailed as exc:
                status_code, retryable, retry_after_s = _typed_retry_fields(exc)
                decision = retry_decision(
                    policy=retry_policy,
                    attempt=attempt,
                    error=exc.message,
                    status_code=status_code,
                    retryable=retryable,
                    retry_after_s=retry_after_s,
                )
                if decision.reason is not None and not exc.emitted_error_event:
                    yield (
                        await self._event_writer.emit(
                            Event(
                                type=EventType.MODEL_ERROR,
                                session_id=session.id,
                                agent_name=registered_agent.spec.name,
                                environment_name=environment_name,
                                payload=_retry_attempt_payload(
                                    exc.payload,
                                    step=step,
                                    attempt=attempt,
                                    max_attempts=retry_policy.max_attempts,
                                    model_attempt_identity=model_attempt_identity,
                                ),
                            )
                        ),
                        None,
                    )
                if not decision.retry:
                    if exc.cause is not None:
                        raise exc.cause from exc
                    raise RuntimeError(exc.message) from exc
                yield (
                    await self._event_writer.emit(
                        _model_retry_event(
                            session=session,
                            registered_agent=registered_agent,
                            environment_name=environment_name,
                            registered_provider=registered_provider,
                            step=step,
                            decision=decision,
                            error=exc.message,
                            model_attempt_identity=model_attempt_identity,
                        )
                    ),
                    None,
                )
                yield (
                    await self._event_writer.emit(
                        _model_attempt_discarded_event(
                            session=session,
                            registered_agent=registered_agent,
                            environment_name=environment_name,
                            registered_provider=registered_provider,
                            step=step,
                            decision=decision,
                            model_attempt_identity=model_attempt_identity,
                        )
                    ),
                    None,
                )
                await self._sleep_before_retry(session.id, decision)
                prior_retry_failure = exc
                attempt += 1
            finally:
                await _close_async_iterator(attempt_events)

    async def _observe_context_pressure(
        self,
        *,
        model_request: ModelRequest,
        session: Session,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_provider: runtime_records.RegisteredProvider,
        environment_name: str | None,
        step: int,
        attempt: int,
        max_attempts: int,
        model_attempt_identity: ModelAttemptIdentity,
    ) -> tuple[_ContextPressureObservation | None, Event | None]:
        if self._context_counting.mode == ContextCountingMode.OFF:
            return None, None
        observation_id = str(uuid4())
        profile = copy_model_context_pressure_profile(
            registered_provider.provider.context_pressure_profile
        )
        estimate = estimate_model_request_context_pressure(
            model_request=model_request,
            image_min_tokens=profile.image_min_tokens,
            document_min_tokens=profile.document_min_tokens,
            document_bytes_per_token=profile.document_bytes_per_token,
            tool_schema_chars_per_token=profile.tool_schema_chars_per_token,
        )
        observation = _ContextPressureObservation(
            estimate=estimate,
            observation_id=observation_id,
        )
        event = await self._event_writer.emit(
            _context_observation_event(
                Event(
                    type=EventType.CONTEXT_PRESSURE_ESTIMATED,
                    session_id=session.id,
                    agent_name=registered_agent.spec.name,
                    environment_name=environment_name,
                    payload={
                        **_context_count_base_payload(
                            model_request=model_request,
                            provider_name=registered_provider.name,
                            step=step,
                            attempt=attempt,
                            max_attempts=max_attempts,
                            observation_id=observation_id,
                            model_attempt_identity=model_attempt_identity,
                        ),
                        "estimate": estimate.model_dump(mode="json"),
                    },
                )
            )
        )
        return observation, event

    async def _observe_context_count(
        self,
        *,
        provider: ModelProvider,
        model_request: ModelRequest,
        session: Session,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_provider: runtime_records.RegisteredProvider,
        environment_name: str | None,
        step: int,
        attempt: int,
        max_attempts: int,
        model_attempt_identity: ModelAttemptIdentity,
    ) -> tuple[_ContextCountObservation | None, Event | None]:
        if self._context_counting.mode == ContextCountingMode.OFF:
            return None, None
        observation_id = str(uuid4())
        base_payload = _context_count_base_payload(
            model_request=model_request,
            provider_name=registered_provider.name,
            step=step,
            attempt=attempt,
            max_attempts=max_attempts,
            observation_id=observation_id,
            model_attempt_identity=model_attempt_identity,
        )
        try:
            provider_result = await provider.count_input_tokens(
                _copy_model_request_for_counting(model_request)
            )
            provider_result = copy_input_token_count_result(provider_result)
            result = (
                provider_result
                if provider_result is not None
                else InputTokenCountResult(
                    input_tokens=None,
                    method=InputTokenCountMethod.UNAVAILABLE,
                    confidence=InputTokenCountConfidence.UNAVAILABLE,
                )
            )
        except Exception as exc:
            portability_failure = extract_durable_value_error(exc)
            provider_failure = None
            if portability_failure is None:
                try:
                    provider_failure = copy_provider_exception_control(exc)
                except DurableValueError as portability_error:
                    portability_failure = portability_error
            if portability_failure is not None:
                provider_error, durable_diagnostics = nonportable_model_provider_error(
                    portability_failure,
                    fallback_provider=registered_provider.name,
                )
                error_message = str(provider_error)
                error_type = type(provider_error).__name__
            else:
                assert provider_failure is not None
                durable_diagnostics = {}
                error_message = provider_failure.message
                error_type = provider_failure.error_type
            event = await self._event_writer.emit(
                _context_observation_event(
                    Event(
                        type=EventType.CONTEXT_COUNT_FAILED,
                        session_id=session.id,
                        agent_name=registered_agent.spec.name,
                        environment_name=environment_name,
                        payload={
                            **base_payload,
                            "error": error_message,
                            "error_type": error_type,
                            **durable_diagnostics,
                        },
                    )
                )
            )
            return None, event

        observation = _ContextCountObservation(
            result=result,
            observation_id=observation_id,
        )
        event = await self._event_writer.emit(
            _context_observation_event(
                Event(
                    type=EventType.CONTEXT_COUNTED,
                    session_id=session.id,
                    agent_name=registered_agent.spec.name,
                    environment_name=environment_name,
                    payload={
                        **base_payload,
                        "count": result.model_dump(mode="json"),
                    },
                )
            )
        )
        return observation, event

    async def _run_once(
        self,
        *,
        provider: ModelProvider,
        model_request: ModelRequest,
        session: Session,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_provider: runtime_records.RegisteredProvider,
        environment_name: str | None,
        step: int,
        attempt: int,
        max_attempts: int,
        model_attempt_identity: ModelAttemptIdentity,
        transcript_cursor_before_request: int,
        record_model_completion: Callable[[Event], Event],
        before_provider_dispatch: Callable[[ModelAttemptIdentity], Awaitable[None]],
        billing_identity: BillingIdentity | None,
        structured_output: StructuredOutputSpec | None,
        prepare_model_completion_dispatch: Callable[
            [ModelRequest],
            Awaitable[ModelCompletionDispatch],
        ]
        | None,
        model_completion_publisher: ModelCompletionPublisher | None,
    ) -> AsyncIterator[tuple[Event | None, AssistantStepResult | None]]:
        assistant_parts: list[
            transcript_helpers.AssistantTextPart
            | transcript_helpers.AssistantThinkingPart
            | ToolCallPart
        ] = []
        thinking_options = model_request.options.get("thinking")
        include_thinking_in_transcript = (
            thinking_options.get("include_in_transcript", True)
            if isinstance(thinking_options, dict)
            else True
        )
        tool_calls: list[runtime_records.ToolCallRequest] = []
        provider_state_parts: list[ProviderStatePart] = []
        completed_stream_event: ModelStreamEvent | None = None
        step_result: AssistantStepResult | None = None
        completion_event: Event | None = None
        completion_dispatch: ModelCompletionDispatch | None = None
        model_completed = False
        profile = copy_model_context_pressure_profile(
            registered_provider.provider.context_pressure_profile
        )
        context_pressure_estimate = estimate_model_request_context_pressure(
            model_request=model_request,
            image_min_tokens=profile.image_min_tokens,
            document_min_tokens=profile.document_min_tokens,
            document_bytes_per_token=profile.document_bytes_per_token,
            tool_schema_chars_per_token=profile.tool_schema_chars_per_token,
        )
        interrupt_poll = self._session_control.stream_interrupt_poll(session.id)
        if (prepare_model_completion_dispatch is None) != (model_completion_publisher is None):
            raise RuntimeError(
                "Model completion staging and publication must be configured together."
            )
        # Deliver a cancellation already pending before the dispatch boundary.
        # A request count retained after this checkpoint is historical; only a
        # later generation can classify provider or cleanup failure as caller
        # cancellation.
        await asyncio.sleep(0)
        current_task = asyncio.current_task()
        provider_cancellation_baseline = 0 if current_task is None else current_task.cancelling()
        # This is the accounting boundary: after the callback returns, the next
        # expression enters provider-controlled code and billable work may occur.
        model_attempt_identity = copy_model_attempt_identity(model_attempt_identity)
        if prepare_model_completion_dispatch is None:
            await before_provider_dispatch(model_attempt_identity)
        else:
            completion_dispatch = await prepare_model_completion_dispatch(model_request)
        provider_events: AsyncIterator[ModelStreamEvent] | None = None
        provider_exhausted = False
        durable_stream_failure: ModelAttemptFailed | None = None
        provider_control_failure: ModelProviderError | None = None
        post_completion_failure: BaseException | None = None
        try:
            provider_events = provider.stream(model_request)
            async for raw_stream_event in provider_events:
                boundary_value = _validate_stream_event(
                    raw_stream_event,
                    provider_name=registered_provider.name,
                    requested_model=session.model,
                    usage_dialect=registered_provider.usage_dialect,
                )
                stream_event = boundary_value.event
                await interrupt_poll.raise_if_interrupted()
                if model_completed:
                    message = f"Model provider emitted event after completed: {stream_event.type}"
                    raise ModelAttemptFailed(
                        message=message,
                        payload={"error": message, "error_type": "RuntimeError"},
                        emitted_error_event=False,
                        cause=RuntimeError(message),
                        completion_observed=model_completion_publisher is not None,
                    )

                if stream_event.type == ModelStreamEventType.TOOL_CALL:
                    tool_call = transcript_helpers.parse_tool_call(stream_event.payload)
                    tool_calls.append(tool_call)
                    assistant_parts.append(transcript_helpers.tool_call_part(tool_call))
                    continue

                if stream_event.type == ModelStreamEventType.TEXT_DELTA:
                    transcript_helpers.append_assistant_text_delta(
                        assistant_parts,
                        stream_event.delta,
                    )
                elif stream_event.type == ModelStreamEventType.THINKING:
                    transcript_helpers.append_assistant_thinking_delta(
                        assistant_parts,
                        stream_event.delta,
                        provider_state=stream_event.payload.get("provider_state"),
                        include=include_thinking_in_transcript,
                    )
                    if not stream_event.delta:
                        # Opaque/redacted thinking state belongs in the transcript,
                        # but an empty readable delta should not reach consumers.
                        continue
                elif stream_event.type == ModelStreamEventType.COMPLETED:
                    completion_terminal_error: ModelProviderError | None = None
                    completion_diagnostics: dict[str, Any] = {}
                    if boundary_value.completion_error is not None:
                        code, path = safe_durable_value_error_details(
                            boundary_value.completion_error
                        )
                        completion_terminal_error = ModelProviderError(
                            "Model provider emitted a non-portable completion value.",
                            provider=registered_provider.name,
                            error_type="DurableValueError",
                            error_code="invalid_model_completion_value",
                            retryable=False,
                        )
                        completion_diagnostics = {
                            "completion_outcome": "invalid_metadata",
                            "completion_error": {
                                "error": str(completion_terminal_error),
                                "error_type": type(completion_terminal_error).__name__,
                                "durable_value_error_code": code,
                                "durable_value_path": path,
                                **completion_terminal_error.error_payload_fields(),
                            },
                        }
                    else:
                        try:
                            billing_identity = resolve_completion_billing_identity(
                                provider,
                                billing_identity,
                                copy_durable_json_object(
                                    stream_event.payload,
                                    "completed_payload",
                                ),
                                provider_name=registered_provider.name,
                            )
                        except ModelProviderError as exc:
                            completion_terminal_error = exc
                            completion_diagnostics = {
                                "completion_outcome": "billing_identity_resolution_failed",
                                "completion_error": {
                                    "error": str(completion_terminal_error),
                                    "error_type": type(completion_terminal_error).__name__,
                                    "stage": "billing_identity_for_completion",
                                    **completion_terminal_error.error_payload_fields(),
                                },
                            }
                    model_completed = True
                    completed_stream_event = stream_event
                    assistant_message = None
                    classification = None
                    if completion_terminal_error is None:
                        try:
                            _require_unique_tool_call_ids(tool_calls)
                            provider_state_parts = transcript_helpers.provider_state_parts(
                                stream_event.payload
                            )
                            assistant_message = transcript_helpers.assistant_message(
                                content_parts=assistant_parts,
                                provider_state_parts=provider_state_parts,
                            )
                        except (TypeError, ValueError):
                            completion_terminal_error = ModelProviderError(
                                "Model provider emitted invalid completion transcript state.",
                                provider=registered_provider.name,
                                error_type="ValueError",
                                error_code="invalid_model_completion_transcript",
                                retryable=False,
                            )
                            completion_diagnostics = {
                                "completion_outcome": "invalid_transcript_state",
                                "completion_error": {
                                    "error": str(completion_terminal_error),
                                    "error_type": type(completion_terminal_error).__name__,
                                    "stage": "completion_transcript_projection",
                                    **completion_terminal_error.error_payload_fields(),
                                },
                            }
                    if completion_terminal_error is None:
                        step_result = _assistant_step_result(
                            session_id=session.id,
                            step=step,
                            model_attempt_identity=model_attempt_identity,
                            assistant_message=assistant_message,
                            tool_calls=tool_calls,
                            completion=_stream_event_completion(completed_stream_event),
                        )
                        classification = classify_assistant_step(step_result)
                    completion_event = _model_stream_event_to_runtime_event(
                        stream_event,
                        session=session,
                        registered_agent=registered_agent,
                        environment_name=environment_name,
                        provider_name=registered_provider.name,
                        step=step,
                        attempt=attempt,
                        max_attempts=max_attempts,
                        model_attempt_identity=model_attempt_identity,
                        tool_round_identity=(
                            step_result.tool_round_identity if step_result is not None else None
                        ),
                        classification=(
                            classification.payload() if classification is not None else None
                        ),
                        context_pressure_estimate=context_pressure_estimate,
                        transcript_cursor_after_completion=(
                            transcript_cursor_before_request
                            + (1 if assistant_message is not None else 0)
                        ),
                        usage_dialect=registered_provider.usage_dialect,
                        billing_identity=billing_identity,
                        accounting_usage_metrics=boundary_value.accounting_usage_metrics,
                        accounting_usage_rejected=boundary_value.accounting_usage_rejected,
                        usage_normalization_failed=(boundary_value.usage_normalization_failed),
                        completion_diagnostics=completion_diagnostics,
                    )
                    if model_completion_publisher is None:
                        completion_event = record_model_completion(completion_event)
                        yield await self._event_writer.emit(completion_event), None
                    if completion_terminal_error is not None:
                        provider_control_failure = completion_terminal_error
                        break
                    continue

                if stream_event.type == ModelStreamEventType.ERROR:
                    provider_error = model_provider_error_from_payload(
                        stream_event.payload,
                        fallback_provider=registered_provider.name,
                    )
                    if isinstance(provider_error, ModelContextOverflowError):
                        # Providers may flatten a typed overflow into an error
                        # event. Rehydrate it so bounded recovery can shrink the
                        # request instead of spending generic retries on it.
                        raise provider_error

                event = _model_stream_event_to_runtime_event(
                    stream_event,
                    session=session,
                    registered_agent=registered_agent,
                    environment_name=environment_name,
                    provider_name=registered_provider.name,
                    step=step,
                    attempt=attempt,
                    max_attempts=max_attempts,
                    model_attempt_identity=model_attempt_identity,
                    usage_dialect=registered_provider.usage_dialect,
                )
                emitted_event = await self._event_writer.emit(event)
                if stream_event.type == ModelStreamEventType.ERROR:
                    message = str(stream_event.payload.get("error") or "Model provider error")
                    provider_error = model_provider_error_from_payload(
                        stream_event.payload,
                        fallback_provider=registered_provider.name,
                        fallback_message=message,
                    )
                    yield emitted_event, None
                    raise ModelAttemptFailed(
                        message=message,
                        payload=copy_json_value(stream_event.payload, "payload"),
                        emitted_error_event=True,
                        cause=provider_error or RuntimeError(message),
                    )
                yield emitted_event, None
            else:
                provider_exhausted = True
        except SessionInterruptedByRequest as exc:
            if model_completion_publisher is None or not model_completed:
                raise
            post_completion_failure = exc
        except asyncio.CancelledError as exc:
            if model_completion_publisher is None or not model_completed:
                raise
            post_completion_failure = exc
        except GeneratorExit as exc:
            if model_completion_publisher is None or not model_completed:
                raise
            post_completion_failure = exc
        except BaseExceptionGroup as exc:
            if model_completion_publisher is None or not model_completed:
                raise
            post_completion_failure = exc
        except ModelAttemptFailed as exc:
            if model_completion_publisher is None or not model_completed:
                raise
            durable_stream_failure = exc
        except Exception as exc:
            provider_failure = None
            durable_error = extract_durable_value_error(exc)
            invalid_provider_error = False
            if isinstance(exc, ModelProviderError) or durable_error is None:
                try:
                    provider_failure = copy_provider_exception_control(exc)
                except DurableValueError as portability_error:
                    durable_error = portability_error
                    invalid_provider_error = True
            if provider_failure is not None:
                if isinstance(provider_failure.cause, ModelContextOverflowError):
                    provider_control_failure = provider_failure.cause
                else:
                    durable_stream_failure = ModelAttemptFailed(
                        message=provider_failure.message,
                        payload={
                            "error": provider_failure.message,
                            "error_type": provider_failure.error_type,
                        },
                        emitted_error_event=False,
                        cause=provider_failure.cause,
                        completion_observed=(
                            model_completion_publisher is not None and model_completed
                        ),
                    )
            elif durable_error is not None:
                if invalid_provider_error:
                    provider_error, durable_diagnostics = nonportable_model_provider_error(
                        durable_error,
                        fallback_provider=registered_provider.name,
                    )
                else:
                    durable_error_code, durable_error_path = safe_durable_value_error_details(
                        durable_error
                    )
                    provider_error = ModelProviderError(
                        "Model provider emitted a non-portable stream value.",
                        provider=registered_provider.name,
                        error_type="DurableValueError",
                        error_code="invalid_model_stream_value",
                        retryable=False,
                    )
                    durable_diagnostics = {
                        "durable_value_error_code": durable_error_code,
                        "durable_value_path": durable_error_path,
                    }
                error_payload = {
                    "error": str(provider_error),
                    "error_type": type(provider_error).__name__,
                    "stage": "model_stream_validation",
                    **durable_diagnostics,
                    **provider_error.error_payload_fields(),
                }
                durable_stream_failure = ModelAttemptFailed(
                    message=str(provider_error),
                    payload=error_payload,
                    emitted_error_event=not (
                        model_completion_publisher is not None and model_completed
                    ),
                    cause=provider_error,
                    completion_observed=(
                        model_completion_publisher is not None and model_completed
                    ),
                )
                if model_completion_publisher is None or not model_completed:
                    yield (
                        await self._event_writer.emit(
                            Event(
                                type=EventType.MODEL_ERROR,
                                session_id=session.id,
                                agent_name=registered_agent.spec.name,
                                environment_name=environment_name,
                                payload=_retry_attempt_payload(
                                    error_payload,
                                    step=step,
                                    attempt=attempt,
                                    max_attempts=max_attempts,
                                    model_attempt_identity=model_attempt_identity,
                                ),
                            )
                        ),
                        None,
                    )
            else:  # pragma: no cover - every Exception has a control or durable failure
                raise RuntimeError("Provider exception handling lost its failure state.") from None
        except BaseException as exc:
            if model_completion_publisher is None or not model_completed:
                raise
            post_completion_failure = exc
        finally:
            if provider_events is not None and not provider_exhausted:
                try:
                    await _close_async_iterator(provider_events)
                except BaseException as exc:
                    if model_completion_publisher is None or not model_completed:
                        raise
                    post_completion_failure = _combine_post_completion_failures(
                        post_completion_failure,
                        exc,
                    )

        if model_completed and model_completion_publisher is not None:
            if completed_stream_event is None:
                raise RuntimeError("Model provider completed without completion metadata.")
            if completion_event is None:
                raise RuntimeError("Model provider completed without a completion event.")
            if completion_dispatch is None:
                raise RuntimeError("Model provider completed without a prepared dispatch.")

            terminal_failure: BaseException | None = (
                post_completion_failure or provider_control_failure or durable_stream_failure
            )
            if terminal_failure is None:
                try:
                    await self._session_control.raise_if_interrupted(session.id)
                except (SessionInterruptedByRequest, asyncio.CancelledError) as exc:
                    terminal_failure = exc
            publication_cancellation = _take_model_completion_cancellation(
                terminal_failure,
                cancellation_baseline=provider_cancellation_baseline,
            )
            if publication_cancellation is not None and terminal_failure is None:
                terminal_failure = publication_cancellation
            elif publication_cancellation is None and isinstance(
                terminal_failure,
                asyncio.CancelledError,
            ):
                terminal_failure = unexpected_child_cancellation_error(
                    terminal_failure,
                    operation="Model provider stream",
                )

            durable_step_result = None
            structured_output_validation = None
            if step_result is not None:
                try:
                    candidate_step_result = _durable_assistant_step_result(
                        step_result,
                        redactor=self._secret_redactor,
                    )
                    candidate_validation = None
                    if (
                        terminal_failure is None
                        and structured_output is not None
                        and structured_output.strategy == StructuredOutputStrategy.TOOL
                        and any(
                            call.name == STRUCTURED_OUTPUT_TOOL_NAME
                            for call in step_result.tool_calls
                        )
                    ):
                        candidate_validation = _redact_structured_output_validation(
                            _validate_structured_output_tool_round(
                                tool_calls=step_result.tool_calls,
                                spec=structured_output,
                            ),
                            self._secret_redactor,
                        )
                    durable_step_result = candidate_step_result
                    structured_output_validation = candidate_validation
                except (TypeError, ValueError):
                    if terminal_failure is None:
                        terminal_failure = ModelProviderError(
                            "Model provider emitted assistant output that cannot cross "
                            "the durable publication boundary.",
                            provider=registered_provider.name,
                            error_type="DurableBoundaryError",
                            error_code="invalid_model_completion_transcript",
                            retryable=False,
                        )
            authoritative_assistant_message = (
                durable_step_result.assistant_message
                if durable_step_result is not None and terminal_failure is None
                else None
            )
            publication_event = (
                completion_event
                if terminal_failure is None
                else _non_turn_model_completion_event(
                    completion_event,
                    failure=terminal_failure,
                    cancellation=publication_cancellation,
                    transcript_cursor=completion_dispatch.stage.source_transcript_cursor,
                )
            )
            publication_event = record_model_completion(publication_event)
            publication_request = ModelCompletionPublicationRequest(
                dispatch=completion_dispatch,
                assistant_step_result=durable_step_result,
                completion_event=publication_event,
                authoritative_assistant_message=authoritative_assistant_message,
                structured_output_validation=structured_output_validation,
            )
            await _publish_model_completion(
                model_completion_publisher,
                publication_request,
                terminal_failure=terminal_failure,
                publication_cancellation=publication_cancellation,
            )
            if terminal_failure is None or not exception_tree_contains(
                terminal_failure,
                GeneratorExit,
            ):
                yield copy_event(publication_event), None
            if terminal_failure is not None:
                raise terminal_failure

        if type(provider_control_failure) is ModelContextOverflowError:
            yield (
                await self._event_writer.emit(
                    _model_context_overflow_error_event(
                        provider_control_failure,
                        session=session,
                        registered_agent=registered_agent,
                        environment_name=environment_name,
                        step=step,
                        attempt=attempt,
                        max_attempts=max_attempts,
                        model_attempt_identity=model_attempt_identity,
                    )
                ),
                None,
            )
        if provider_control_failure is not None:
            raise provider_control_failure from None
        if durable_stream_failure is not None:
            raise durable_stream_failure from None
        if post_completion_failure is not None:
            raise post_completion_failure
        if not model_completed:
            message = "Model provider stream ended without a completed event."
            raise ModelAttemptFailed(
                message=message,
                payload={"error": message, "error_type": "RuntimeError"},
                emitted_error_event=False,
                cause=RuntimeError(message),
            )
        await self._session_control.raise_if_interrupted(session.id)
        if completed_stream_event is None:
            raise RuntimeError("Model provider completed without completion metadata.")
        if step_result is None:
            raise RuntimeError("Model provider completed without an assistant step result.")
        yield None, step_result

    async def _sleep_before_retry(self, session_id: str, decision: RetryDecision) -> None:
        await self._session_control.raise_if_interrupted(session_id)
        if decision.delay_seconds > 0:
            await asyncio.sleep(decision.delay_seconds)
        await self._session_control.raise_if_interrupted(session_id)


class ModelStepRun:
    """Per-run model-step dependencies and accounting state."""

    def __init__(
        self,
        executor: ModelStepExecutor,
        *,
        provider: ModelProvider,
        session: Session,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_provider: runtime_records.RegisteredProvider,
        registered_environment: runtime_records.RegisteredEnvironment | None,
        environment_name: str | None,
        structured_output: StructuredOutputSpec | None,
        thinking: ThinkingConfig | None,
        knowledge_store: Any,
        request_metadata: dict[str, Any],
        retry_policy: RetryPolicy,
        request_budget_limits: tuple[BudgetLimit, ...],
        limit_gate: RunLimitGate,
        budget_policy: BudgetPolicy | None,
        run_started_at: float,
        turn_usage_tracker: SessionUsageTracker | None,
        active_run: ActiveSessionRun[SessionUsageTracker] | None,
        model_completion_publisher: ModelCompletionPublisher | None = None,
    ) -> None:
        self._executor = executor
        self._provider = provider
        self._session = session
        self._registered_agent = registered_agent
        self._registered_provider = registered_provider
        self._registered_environment = registered_environment
        self._environment_name = environment_name
        self._structured_output = structured_output
        self._thinking = thinking
        self._knowledge_store = knowledge_store
        self._request_metadata = copy_json_value(request_metadata, "metadata")
        self._retry_policy = copy_retry_policy(retry_policy)
        self._request_budget_limits = copy_request_budget_limits(request_budget_limits)
        self._limit_gate = limit_gate
        self._budget_policy = budget_policy
        self._run_started_at = run_started_at
        self._turn_usage_tracker = turn_usage_tracker
        self._active_run = active_run
        self._reservation_identity_guard = (
            self._executor._run_limit_controller.reservation_identity_guard()
        )
        self._model_completion_publisher = model_completion_publisher
        contextual_limits = (
            *budget_limits_for_session(
                policy=self._budget_policy,
                agent_name=self._registered_agent.spec.name,
                causal_budget_id=self._session.causal_budget_id,
            ),
            *self._request_budget_limits,
        )
        self._deferred_contextual_price = any(
            has_deferred_contextual_price(
                limit.pricing,
                provider_name=(self._provider.billing_provider_name or self._session.provider_name),
                model=self._session.model,
            )
            for limit in contextual_limits
        )

    async def _abandon_pre_dispatch_model_stage(
        self,
        stage: ModelCompletionStage,
        *,
        authoritative_failure: BaseException,
    ) -> None:
        """Clear one provably undispatched stage without losing its root failure."""

        async def abandon_once() -> ModelCompletionStageAbandonmentResult:
            return await self._executor._session_store.abandon_model_completion_stage(
                self._session.id,
                stage_id=stage.stage_id,
                preparation_digest=stage.preparation_digest,
                expected_run_epoch=self._session.run_epoch,
            )

        async def abandon_with_exact_replay() -> ModelCompletionStageAbandonmentResult:
            try:
                return await abandon_once()
            except (Exception, asyncio.CancelledError) as first_error:
                try:
                    return await abandon_once()
                except (Exception, asyncio.CancelledError) as replay_error:
                    replay_error.add_note(
                        "Exact model-completion stage abandonment replay also failed after "
                        f"{type(first_error).__name__}: {first_error}"
                    )
                    raise replay_error from first_error

        abandonment_task = asyncio.create_task(abandon_with_exact_replay())
        outcome = await await_shielded_task_outcome(
            abandonment_task,
            cancellation=(
                authoritative_failure
                if isinstance(authoritative_failure, asyncio.CancelledError)
                else None
            ),
        )
        abandonment_error = outcome.error
        if isinstance(abandonment_error, asyncio.CancelledError):
            abandonment_error = unexpected_child_cancellation_error(
                abandonment_error,
                operation="Pre-dispatch model-completion stage abandonment",
            )
        if abandonment_error is None:
            result = outcome.result
            try:
                if type(result) is not ModelCompletionStageAbandonmentResult:
                    raise TypeError(
                        "Model-completion stage abandonment returned an invalid result."
                    )
                abandonment = result.abandonment
                for field_name in (
                    "session_id",
                    "stage_id",
                    "logical_step_id",
                    "dispatch_ordinal",
                    "purpose",
                    "preparation_request_digest",
                    "preparation_digest",
                    "source_status",
                    "source_run_epoch",
                    "source_transcript_cursor",
                ):
                    if getattr(abandonment, field_name) != getattr(stage, field_name):
                        raise RuntimeError(
                            "Model-completion stage abandonment acknowledged a different "
                            f"prepared stage field: {field_name}."
                        )
            except BaseException as validation_error:
                abandonment_error = validation_error
        if abandonment_error is not None:
            authoritative_failure.add_note(
                "Pre-dispatch model-completion stage abandonment also failed: "
                f"{type(abandonment_error).__name__}: {abandonment_error}"
            )
        if outcome.cancellation is not None and not isinstance(
            authoritative_failure, asyncio.CancelledError
        ):
            cancellation = outcome.cancellation
            cancellation.add_note(
                "Cancellation arrived while abandoning a model-completion stage after "
                f"{type(authoritative_failure).__name__}: {authoritative_failure}"
            )
            if abandonment_error is not None:
                cancellation.add_note(
                    "Model-completion stage abandonment also failed: "
                    f"{type(abandonment_error).__name__}: {abandonment_error}"
                )
            raise cancellation from authoritative_failure

    async def execute(
        self,
        *,
        step: int,
        messages: list[Message],
        model_step_identity: ModelStepIdentity,
    ) -> AsyncIterator[tuple[Event | None, ModelStepFlowOutcome | None]]:
        model_step_identity = copy_model_step_identity(model_step_identity)
        context_messages: list[Message]
        compaction_budget_events: list[Event] = []
        published_compaction_attempt_ids: set[str] = set()
        compaction_start_events: list[Event] = []
        compaction_completion_events: dict[str, Event] = {}
        compaction_identity_ledger = _CompactionExecutionIdentityLedger(model_step_identity)

        async def run_automatic_compaction(
            compactor: ContextCompactor,
            compaction_request: CompactionRequest,
            compaction_started: ContextCompactionTelemetry,
            execute: Callable[[], Awaitable[CompactionResult]],
            completed_payloads: Callable[[], list[dict[str, Any]]],
        ) -> CompactionResult:
            await self._persist_automatic_compaction_started(
                compaction_started,
                published_events=compaction_budget_events,
                start_events=compaction_start_events,
                model_step_identity=model_step_identity,
            )

            async def publish_completions(payloads: list[dict[str, Any]]) -> None:
                await self._persist_automatic_compaction_completions(
                    compaction_identity_ledger.identify_payloads(payloads),
                    published_attempt_ids=published_compaction_attempt_ids,
                    published_events=compaction_budget_events,
                    completion_events=compaction_completion_events,
                )

            async def run() -> CompactionResult:
                return await self._run_automatic_compaction_with_budget(
                    compactor=compactor,
                    compaction_request=compaction_request,
                    execute=execute,
                    completed_payloads=completed_payloads,
                    budget_events=compaction_budget_events,
                    messages=messages,
                    model_step_identity=model_step_identity,
                    compaction_identity_ledger=compaction_identity_ledger,
                )

            with _compaction_completion_publisher_scope(publish_completions):
                return await run()

        current_task = asyncio.current_task()
        context_build_cancellation_requests = (
            0 if current_task is None else current_task.cancelling()
        )
        try:
            (
                context_messages,
                checkpoint_update,
                checkpoint_event_payload,
                context_compaction_telemetry,
                context_knowledge_telemetry,
            ) = await _build_context(
                context_policy=self._registered_agent.context_policy,
                session_store=self._executor._session_store,
                session=self._session,
                agent_spec=_session_agent_spec(
                    registered_agent=self._registered_agent,
                    session=self._session,
                ),
                messages=messages,
                step=step,
                environment_name=self._environment_name,
                knowledge_store=self._knowledge_store,
                request_metadata=self._request_metadata,
                pressure_overhead=_context_pressure_overhead(
                    registered_provider=self._registered_provider,
                    registered_agent=self._registered_agent,
                    registered_environment=self._registered_environment,
                    structured_output=self._structured_output,
                    thinking=self._thinking,
                    step=step,
                ),
                count_input_tokens=self._context_input_token_counter(step=step),
                build_cache_prefix_request=self._cache_prefix_request_builder(step=step),
                secret_redactor=self._executor._secret_redactor,
                run_compaction=run_automatic_compaction,
            )
        except ContextBuildError as exc:
            (
                context_failure_events,
                context_failure_persistence,
            ) = await self._context_build_failure_events(
                exc,
                model_step_identity=model_step_identity,
                compaction_identity_ledger=compaction_identity_ledger,
                published_compaction_attempt_ids=published_compaction_attempt_ids,
                compaction_completion_events=compaction_completion_events,
                compaction_start_event=(
                    compaction_start_events[0] if compaction_start_events else None
                ),
                compaction_started_published=any(
                    event.type == EventType.CONTEXT_COMPACTION_STARTED
                    for event in compaction_budget_events
                ),
            )
            for event in compaction_budget_events:
                yield event, None
            for event in context_failure_events:
                yield event, None
            if context_failure_persistence is not None:
                raise context_failure_persistence from exc
            if isinstance(exc.cause, _AutomaticCompactionBudgetReservationFailed):
                async for event in self._stop_for_budget_reservation_failure(
                    result=exc.cause.result,
                    messages=messages,
                ):
                    yield event, None
                yield None, ModelStepFlowOutcome(stop_session=True)
                return
            if isinstance(exc.cause, _AutomaticCompactionAdmissionStopped):
                admission_events = self._automatic_compaction_admission_events(
                    exc.cause,
                    messages=messages,
                )
                try:
                    async for event in admission_events:
                        yield event, None
                finally:
                    await _close_async_iterator(admission_events)
                yield None, ModelStepFlowOutcome(stop_session=True)
                return
            raise exc.cause from exc
        except BaseException as exc:
            await self._persist_context_build_termination_events(
                exc,
                model_step_identity=model_step_identity,
                compaction_identity_ledger=compaction_identity_ledger,
                published_compaction_attempt_ids=published_compaction_attempt_ids,
                compaction_completion_events=compaction_completion_events,
                compaction_start_event=(
                    compaction_start_events[0] if compaction_start_events else None
                ),
                compaction_started_published=any(
                    event.type == EventType.CONTEXT_COMPACTION_STARTED
                    for event in compaction_budget_events
                ),
                cancellation_requests_before_build=context_build_cancellation_requests,
            )
            raise

        context_success_events, context_success_persistence = await self._context_success_events(
            model_step_identity=model_step_identity,
            compaction_identity_ledger=compaction_identity_ledger,
            checkpoint_update=checkpoint_update,
            checkpoint_event_payload=checkpoint_event_payload,
            compaction_telemetry=context_compaction_telemetry,
            knowledge_telemetry=context_knowledge_telemetry,
            published_compaction_attempt_ids=published_compaction_attempt_ids,
            compaction_completion_events=compaction_completion_events,
            compaction_start_event=(
                compaction_start_events[0] if compaction_start_events else None
            ),
            compaction_started_published=any(
                event.type == EventType.CONTEXT_COMPACTION_STARTED
                for event in compaction_budget_events
            ),
        )
        for event in compaction_budget_events:
            yield event, None
        for event in context_success_events:
            yield event, None
        if context_success_persistence is not None:
            raise context_success_persistence
        await self._executor._session_control.raise_if_interrupted(self._session.id)

        if _has_provider_backed_context_compaction(context_compaction_telemetry):
            should_stop: bool | None = None
            gate_events = self._post_compaction_gate(
                messages=messages,
                model_step_identity=model_step_identity,
            )
            try:
                async for event, gate_outcome in gate_events:
                    if event is not None:
                        yield event, None
                    if gate_outcome is not None:
                        should_stop = gate_outcome
            finally:
                await _close_async_iterator(gate_events)
            if should_stop is None:
                raise RuntimeError("Post-compaction gate finished without an outcome.")
            if should_stop:
                yield None, ModelStepFlowOutcome(stop_session=True)
                return

        model_request = await self._executor.build_request(
            session=self._session,
            registered_agent=self._registered_agent,
            registered_environment=self._registered_environment,
            context_messages=context_messages,
            structured_output=self._structured_output,
            thinking=self._thinking,
            step=step,
        )
        request_events = self._execute_request(
            model_request=model_request,
            step=step,
            messages=messages,
            model_step_identity=model_step_identity,
        )
        try:
            async for event, outcome in request_events:
                yield event, outcome
        finally:
            await _close_async_iterator(request_events)

    async def _execute_request(
        self,
        *,
        model_request: ModelRequest,
        step: int,
        messages: list[Message],
        model_step_identity: ModelStepIdentity,
    ) -> AsyncIterator[tuple[Event | None, ModelStepFlowOutcome | None]]:
        model_step_identity = copy_model_step_identity(model_step_identity)
        initial_model_attempt_identity = model_step_identity.new_attempt()
        controller = self._executor._run_limit_controller
        try:
            billing_identity = await resolve_request_billing_identity(
                self._provider,
                _detach_model_request(model_request),
                provider_name=self._registered_provider.name,
            )
        except asyncio.CancelledError:
            raise
        except ModelProviderError as provider_error:
            payload = {
                "error": str(provider_error),
                "error_type": type(provider_error).__name__,
                "stage": "billing_identity_for_request",
                **provider_error.error_payload_fields(),
            }
            yield (
                await self._executor._event_writer.emit(
                    Event(
                        type=EventType.MODEL_ERROR,
                        session_id=self._session.id,
                        agent_name=self._registered_agent.spec.name,
                        environment_name=self._environment_name,
                        payload=_retry_attempt_payload(
                            payload,
                            step=step,
                            attempt=1,
                            max_attempts=self._retry_policy.max_attempts,
                            model_attempt_identity=initial_model_attempt_identity,
                        ),
                    )
                ),
                None,
            )
            raise
        if billing_identity is not None or self._has_deferred_contextual_price():
            should_stop: bool | None = None
            gate_events = self._billing_identity_budget_gate(
                messages=messages,
                billing_identity=billing_identity,
                model_attempt_identity=initial_model_attempt_identity,
            )
            try:
                async for event, gate_outcome in gate_events:
                    if event is not None:
                        yield event, None
                    if gate_outcome is not None:
                        should_stop = gate_outcome
            finally:
                await _close_async_iterator(gate_events)
            if should_stop is None:
                raise RuntimeError("Billing-identity budget gate finished without an outcome.")
            if should_stop:
                yield None, ModelStepFlowOutcome(stop_session=True)
                return
        reservation_setup = await controller.reserve_for_model_step(
            session=self._session,
            agent_name=self._registered_agent.spec.name,
            provider_name=self._registered_provider.name,
            environment_name=self._environment_name,
            model_attempt_identity=initial_model_attempt_identity,
            budget_policy=self._budget_policy,
            request_budget_limits=self._request_budget_limits,
            billing_identity=billing_identity,
            reservation_identity_guard=self._reservation_identity_guard,
        )
        budget_reservations = list(reservation_setup.reservations)
        try:
            for event in reservation_setup.events:
                yield event, None
        except (GeneratorExit, asyncio.CancelledError) as authoritative_exc:
            if reservation_setup.failure is None and reservation_setup.error is None:
                async for _ in controller.settlement_events_preserving_failure(
                    controller.release_reservations(
                        budget_reservations,
                        session=self._session,
                        agent_name=self._registered_agent.spec.name,
                        environment_name=self._environment_name,
                        reason="model step abandoned before provider dispatch",
                    ),
                    authoritative_failure=authoritative_exc,
                ):
                    pass
            raise
        if reservation_setup.error is not None:
            raise reservation_setup.error
        if reservation_setup.failure is not None:
            async for event in self._stop_for_budget_reservation_failure(
                result=reservation_setup.failure,
                messages=messages,
            ):
                yield event, None
            yield None, ModelStepFlowOutcome(stop_session=True)
            return

        if budget_reservations and controller.reservation_ttl_seconds is not None:
            try:
                await controller.renew_reservations(budget_reservations)
            except asyncio.CancelledError as authoritative_exc:
                async for event in controller.settlement_events_preserving_failure(
                    controller.release_reservations(
                        budget_reservations,
                        session=self._session,
                        agent_name=self._registered_agent.spec.name,
                        environment_name=self._environment_name,
                        reason="model step cancelled before provider dispatch",
                    ),
                    authoritative_failure=authoritative_exc,
                ):
                    yield event, None
                raise
            except BudgetReservationLeaseLost as authoritative_exc:
                async for event in controller.settlement_events_preserving_failure(
                    controller.release_reservations(
                        budget_reservations,
                        session=self._session,
                        agent_name=self._registered_agent.spec.name,
                        environment_name=self._environment_name,
                        reason="reservation lease expired before model step",
                    ),
                    authoritative_failure=authoritative_exc,
                ):
                    yield event, None
                raise

        lifecycle = BudgetModelStepLifecycle()
        lifecycle.prepare_provider_dispatch(
            initial_model_attempt_identity,
            budget_reservations,
        )

        async def settle_provider_dispatch() -> tuple[list[Event], Exception | None]:
            if lifecycle.pending_reservations is not None:
                return [], None
            settlement_events: list[Event] = []
            try:
                async for event in controller.reconcile_dispatched_reservations(
                    budget_reservations,
                    lifecycle=lifecycle,
                    session=self._session,
                    agent_name=self._registered_agent.spec.name,
                    environment_name=self._environment_name,
                    unknown_reason=UNKNOWN_POST_DISPATCH_BUDGET_REASON,
                ):
                    settlement_events.append(event)
            except Exception as settlement_error:
                return settlement_events, settlement_error
            return settlement_events, None

        async def prepare_provider_dispatch(
            model_attempt_identity: ModelAttemptIdentity,
        ) -> tuple[
            list[Event],
            BudgetReservationResult | None,
            Exception | None,
        ]:
            if lifecycle.pending_reservations is not None:
                if lifecycle.pending_model_attempt_identity != model_attempt_identity:
                    raise ValueError(
                        "Prepared provider dispatch has a different model attempt identity."
                    )
                return [], None, None
            settlement_events, settlement_error = await settle_provider_dispatch()
            if settlement_error is not None:
                return settlement_events, None, settlement_error
            retry_setup = await controller.reserve_for_model_step(
                session=self._session,
                agent_name=self._registered_agent.spec.name,
                provider_name=self._registered_provider.name,
                environment_name=self._environment_name,
                model_attempt_identity=model_attempt_identity,
                budget_policy=self._budget_policy,
                request_budget_limits=self._request_budget_limits,
                billing_identity=billing_identity,
                existing_reservation_ids=lifecycle.observed_reservation_ids,
                reservation_identity_guard=self._reservation_identity_guard,
            )
            if retry_setup.error is not None:
                return settlement_events + list(retry_setup.events), None, retry_setup.error
            if retry_setup.failure is not None:
                return settlement_events + list(retry_setup.events), retry_setup.failure, None
            retry_reservations = list(retry_setup.reservations)
            budget_reservations.extend(retry_reservations)
            lifecycle.prepare_provider_dispatch(
                model_attempt_identity,
                retry_reservations,
            )
            return settlement_events + list(retry_setup.events), None, None

        async def before_provider_dispatch(
            model_attempt_identity: ModelAttemptIdentity,
        ) -> None:
            await controller.before_provider_dispatch(
                budget_reservations,
                lifecycle=lifecycle,
                model_attempt_identity=model_attempt_identity,
            )

        # `messages` is the full reconstructed transcript, not the compacted
        # provider context. The store verifies this proposed cursor atomically
        # during stage preparation, so any in-memory/durable drift fails before
        # provider-controlled code runs.
        source_transcript_cursor = len(messages)
        logical_step_id = model_step_identity.model_step_id
        next_dispatch_ordinal = 0

        async def prepare_model_completion_dispatch(
            attempt_model_request: ModelRequest,
        ) -> ModelCompletionDispatch:
            nonlocal next_dispatch_ordinal
            request_fingerprint = _model_request_fingerprint(
                provider_name=self._registered_provider.name,
                model_request=attempt_model_request,
            )
            dispatch_ordinal = next_dispatch_ordinal
            stage_id = f"{logical_step_id}:dispatch:{dispatch_ordinal}"
            async with lifecycle.reservation_transition_lock:
                pending_reservations = lifecycle.pending_reservations
                pending_model_attempt_identity = lifecycle.pending_model_attempt_identity
                if pending_reservations is None or pending_model_attempt_identity is None:
                    raise RuntimeError("Model completion stage has no pending budget reservations.")
                if pending_model_attempt_identity.model_step_id != logical_step_id:
                    raise RuntimeError(
                        "Model completion stage attempt belongs to a different model step."
                    )
                intent = _model_completion_stage_intent(
                    model_attempt_identity=pending_model_attempt_identity,
                    provider_name=self._registered_provider.name,
                    requested_model=attempt_model_request.model,
                    source_transcript_cursor=source_transcript_cursor,
                    request_fingerprint=request_fingerprint,
                )
                prepared = await self._executor._session_store.prepare_model_completion_stage(
                    self._session.id,
                    request=ModelCompletionStageRequest(
                        stage_id=stage_id,
                        logical_step_id=logical_step_id,
                        dispatch_ordinal=dispatch_ordinal,
                        purpose="assistant-turn",
                        intent=intent,
                        reservation_ids=tuple(
                            reservation.record.reservation_id
                            for reservation in pending_reservations
                        ),
                    ),
                    expected_statuses={SessionStatus.RUNNING},
                    expected_run_epoch=self._session.run_epoch,
                    expected_transcript_cursor=source_transcript_cursor,
                )
                if not prepared.dispatch_authorized:
                    raise ModelCompletionDispatchNotAuthorized(
                        stage=prepared.stage,
                        request_fingerprint=request_fingerprint,
                    )
                dispatch_fence_attempted = False
                dispatch_fence_committed = False
                try:
                    # Preparation may itself be a remote database round trip. Renew
                    # after it commits so a lease cannot expire in the new gap
                    # between durable staging and provider-controlled code.
                    if budget_reservations and controller.reservation_ttl_seconds is not None:
                        await controller.renew_reservations(budget_reservations)
                    dispatch_fence_attempted = True
                    deferred_dispatch_failure = await controller.mark_reservations_dispatched(
                        pending_reservations,
                        dispatch_id=prepared.stage.stage_id,
                    )
                    dispatch_fence_committed = True
                    dispatch = ModelCompletionDispatch(
                        stage=prepared.stage,
                        request_fingerprint=request_fingerprint,
                    )
                    lifecycle.mark_provider_dispatch(pending_model_attempt_identity)
                    if deferred_dispatch_failure is not None:
                        raise deferred_dispatch_failure
                except BaseException as authoritative_exc:
                    if not dispatch_fence_attempted or dispatch_fence_committed:
                        await self._abandon_pre_dispatch_model_stage(
                            prepared.stage,
                            authoritative_failure=authoritative_exc,
                        )
                    else:
                        add_exception_note_safely(
                            authoritative_exc,
                            "The prepared model-completion stage was retained because the "
                            "budget dispatch fence could not be reconstructed exactly.",
                        )
                    if isinstance(authoritative_exc, BudgetReservationLeaseLost):
                        raise BudgetReservationLeaseLostBeforeModelDispatch(
                            "Budget reservation lease was lost before model dispatch."
                        ) from authoritative_exc
                    raise
                next_dispatch_ordinal += 1
                return dispatch

        def record_model_completion(event: Event) -> Event:
            return lifecycle.record_model_completion(
                event,
                prepare_event=self._executor._event_writer.prepare,
                settled_at=controller.budget_settlement_time(),
            )

        flow_outcome: ModelStepFlowOutcome | None = None
        model_step_events = self._run_with_context_overflow_recovery(
            provider=self._provider,
            model_request=model_request,
            messages=messages,
            step=step,
            model_step_identity=model_step_identity,
            initial_model_attempt_identity=initial_model_attempt_identity,
            transcript_cursor_before_request=len(messages),
            record_model_completion=record_model_completion,
            settle_provider_dispatch=settle_provider_dispatch,
            prepare_provider_dispatch=prepare_provider_dispatch,
            before_provider_dispatch=before_provider_dispatch,
            billing_identity=billing_identity,
            prepare_model_completion_dispatch=(
                prepare_model_completion_dispatch
                if self._model_completion_publisher is not None
                else None
            ),
            model_completion_publisher=self._model_completion_publisher,
        )
        guarded_events = controller.model_step_events_with_heartbeat(
            model_step_events,
            reservations=budget_reservations,
            lifecycle=lifecycle,
        )
        try:
            async for event, outcome in guarded_events:
                if event is not None:
                    yield event, None
                if outcome is not None:
                    if flow_outcome is not None:
                        raise RuntimeError(
                            "Model step produced more than one terminal flow outcome."
                        )
                    flow_outcome = outcome
        except GeneratorExit as authoritative_exc:
            async for _ in controller.settlement_events_preserving_failure(
                controller.settle_after_model_failure(
                    budget_reservations,
                    lifecycle=lifecycle,
                    session=self._session,
                    agent_name=self._registered_agent.spec.name,
                    environment_name=self._environment_name,
                    release_reason="model step abandoned before provider dispatch",
                ),
                authoritative_failure=authoritative_exc,
            ):
                pass
            raise
        except BudgetDispatchReservationFailed as exc:
            async for event in controller.settle_after_model_failure(
                budget_reservations,
                lifecycle=lifecycle,
                session=self._session,
                agent_name=self._registered_agent.spec.name,
                environment_name=self._environment_name,
                release_reason="retry reservation failed before provider dispatch",
            ):
                yield event, None
            async for event in self._stop_for_budget_reservation_failure(
                result=exc.result,
                messages=messages,
            ):
                yield event, None
            yield None, ModelStepFlowOutcome(stop_session=True)
            return
        except BudgetReservationLeaseLostBeforeModelDispatch as authoritative_exc:
            async for event in controller.settlement_events_preserving_failure(
                controller.release_reservations(
                    budget_reservations,
                    session=self._session,
                    agent_name=self._registered_agent.spec.name,
                    environment_name=self._environment_name,
                    reason="reservation lease expired before model dispatch",
                ),
                authoritative_failure=authoritative_exc,
            ):
                yield event, None
            raise
        except BudgetReservationLeaseLost as authoritative_exc:
            async for event in controller.settlement_events_preserving_failure(
                controller.settle_after_model_failure(
                    budget_reservations,
                    lifecycle=lifecycle,
                    session=self._session,
                    agent_name=self._registered_agent.spec.name,
                    environment_name=self._environment_name,
                    release_reason="reservation heartbeat lost before provider dispatch",
                    unknown_reason="reservation heartbeat lost; charged reserved amount",
                ),
                authoritative_failure=authoritative_exc,
            ):
                yield event, None
            raise
        except SessionInterruptedByRequest as authoritative_exc:
            async for event in controller.settlement_events_preserving_failure(
                controller.settle_after_model_failure(
                    budget_reservations,
                    lifecycle=lifecycle,
                    session=self._session,
                    agent_name=self._registered_agent.spec.name,
                    environment_name=self._environment_name,
                    release_reason="session interrupted before provider dispatch",
                ),
                authoritative_failure=authoritative_exc,
            ):
                yield event, None
            raise
        except asyncio.CancelledError as authoritative_exc:
            async for event in controller.settlement_events_preserving_failure(
                controller.settle_after_model_failure(
                    budget_reservations,
                    lifecycle=lifecycle,
                    session=self._session,
                    agent_name=self._registered_agent.spec.name,
                    environment_name=self._environment_name,
                    release_reason="model step cancelled before provider dispatch",
                ),
                authoritative_failure=authoritative_exc,
            ):
                yield event, None
            raise
        except Exception as provider_exc:
            async for event in controller.settlement_events_preserving_failure(
                controller.settle_after_model_failure(
                    budget_reservations,
                    lifecycle=lifecycle,
                    session=self._session,
                    agent_name=self._registered_agent.spec.name,
                    environment_name=self._environment_name,
                    release_reason="model step failed before provider dispatch",
                ),
                authoritative_failure=provider_exc,
            ):
                yield event, None
            raise
        finally:
            try:
                await _close_async_iterator(guarded_events)
            finally:
                await _close_async_iterator(model_step_events)

        if lifecycle.dispatches:
            async for event in controller.reconcile_dispatched_reservations(
                budget_reservations,
                lifecycle=lifecycle,
                session=self._session,
                agent_name=self._registered_agent.spec.name,
                environment_name=self._environment_name,
                unknown_reason=UNKNOWN_POST_DISPATCH_BUDGET_REASON,
            ):
                yield event, None
        if flow_outcome is None:
            raise RuntimeError("Model step finished without a terminal flow outcome.")
        yield None, flow_outcome

    async def _run_with_context_overflow_recovery(
        self,
        *,
        provider: ModelProvider,
        model_request: ModelRequest,
        messages: list[Message],
        step: int,
        model_step_identity: ModelStepIdentity,
        initial_model_attempt_identity: ModelAttemptIdentity,
        transcript_cursor_before_request: int,
        record_model_completion: Callable[[Event], Event],
        settle_provider_dispatch: Callable[[], Awaitable[tuple[list[Event], Exception | None]]],
        prepare_provider_dispatch: Callable[
            [ModelAttemptIdentity],
            Awaitable[tuple[list[Event], BudgetReservationResult | None, Exception | None]],
        ],
        before_provider_dispatch: Callable[[ModelAttemptIdentity], Awaitable[None]],
        billing_identity: BillingIdentity | None,
        prepare_model_completion_dispatch: Callable[
            [ModelRequest],
            Awaitable[ModelCompletionDispatch],
        ]
        | None,
        model_completion_publisher: ModelCompletionPublisher | None,
    ) -> AsyncIterator[tuple[Event | None, ModelStepFlowOutcome | None]]:
        model_step_identity = copy_model_step_identity(model_step_identity)
        initial_model_attempt_identity = copy_model_attempt_identity(initial_model_attempt_identity)
        if initial_model_attempt_identity.model_step_id != model_step_identity.model_step_id:
            raise ValueError("Initial provider attempt belongs to a different model step.")
        overflow_policy = self._registered_agent.context_overflow_policy
        compaction_budget_events: list[Event] = []
        published_compaction_attempt_ids: set[str] = set()
        compaction_start_events: list[Event] = []
        compaction_completion_events: dict[str, Event] = {}
        compaction_identity_ledger = _CompactionExecutionIdentityLedger(model_step_identity)
        latest_model_attempt_identity: ModelAttemptIdentity | None = None

        def record_model_attempt_identity(identity: ModelAttemptIdentity) -> None:
            nonlocal latest_model_attempt_identity
            latest_model_attempt_identity = copy_model_attempt_identity(identity)

        def run_attempt(
            request: ModelRequest,
            *,
            initial_identity: ModelAttemptIdentity | None = None,
        ) -> AsyncIterator[tuple[Event | None, AssistantStepResult | None]]:
            return self._executor.run_with_retries(
                provider=provider,
                model_request=request,
                session=self._session,
                registered_agent=self._registered_agent,
                registered_provider=self._registered_provider,
                environment_name=self._environment_name,
                step=step,
                model_step_identity=model_step_identity,
                initial_model_attempt_identity=initial_identity,
                retry_policy=self._retry_policy,
                transcript_cursor_before_request=transcript_cursor_before_request,
                record_model_completion=record_model_completion,
                prepare_provider_dispatch=prepare_provider_dispatch,
                before_provider_dispatch=before_provider_dispatch,
                record_model_attempt_identity=record_model_attempt_identity,
                billing_identity=billing_identity,
                structured_output=self._structured_output,
                prepare_model_completion_dispatch=prepare_model_completion_dispatch,
                model_completion_publisher=model_completion_publisher,
            )

        attempt_events = run_attempt(
            model_request,
            initial_identity=initial_model_attempt_identity,
        )
        try:
            try:
                async for event, result in attempt_events:
                    yield (
                        event,
                        ModelStepFlowOutcome(assistant_step_result=result)
                        if result is not None
                        else None,
                    )
                return
            except ModelContextOverflowError as exc:
                if overflow_policy is None:
                    raise
                yield (
                    await self._executor._event_writer.emit(
                        _event_with_model_identity_authority(
                            Event(
                                type=EventType.CONTEXT_OVERFLOW_DETECTED,
                                session_id=self._session.id,
                                agent_name=self._registered_agent.spec.name,
                                environment_name=self._environment_name,
                                payload=_context_overflow_event_payload(
                                    exc,
                                    step=step,
                                    phase="initial",
                                    original_message_count=len(model_request.messages),
                                    model_step_identity=model_step_identity,
                                    model_attempt_identity=latest_model_attempt_identity,
                                ),
                            ),
                            latest_model_attempt_identity or model_step_identity,
                        )
                    ),
                    None,
                )
        finally:
            await _close_async_iterator(attempt_events)

        async def run_automatic_compaction(
            compactor: ContextCompactor,
            compaction_request: CompactionRequest,
            compaction_started: ContextCompactionTelemetry,
            execute: Callable[[], Awaitable[CompactionResult]],
            completed_payloads: Callable[[], list[dict[str, Any]]],
        ) -> CompactionResult:
            await self._persist_automatic_compaction_started(
                compaction_started,
                published_events=compaction_budget_events,
                start_events=compaction_start_events,
                model_step_identity=model_step_identity,
            )

            async def publish_completions(payloads: list[dict[str, Any]]) -> None:
                await self._persist_automatic_compaction_completions(
                    compaction_identity_ledger.identify_payloads(payloads),
                    published_attempt_ids=published_compaction_attempt_ids,
                    published_events=compaction_budget_events,
                    completion_events=compaction_completion_events,
                )

            async def run() -> CompactionResult:
                return await self._run_automatic_compaction_with_budget(
                    compactor=compactor,
                    compaction_request=compaction_request,
                    execute=execute,
                    completed_payloads=completed_payloads,
                    budget_events=compaction_budget_events,
                    messages=messages,
                    model_step_identity=model_step_identity,
                    compaction_identity_ledger=compaction_identity_ledger,
                )

            with _compaction_completion_publisher_scope(publish_completions):
                return await run()

        current_task = asyncio.current_task()
        context_build_cancellation_requests = (
            0 if current_task is None else current_task.cancelling()
        )
        try:
            (
                recovery_context_messages,
                checkpoint_update,
                checkpoint_event_payload,
                compaction_telemetry,
                knowledge_telemetry,
            ) = await _build_context(
                context_policy=overflow_policy,
                session_store=self._executor._session_store,
                session=self._session,
                agent_spec=_session_agent_spec(
                    registered_agent=self._registered_agent,
                    session=self._session,
                ),
                messages=messages,
                step=step,
                environment_name=self._environment_name,
                knowledge_store=self._knowledge_store,
                request_metadata=self._request_metadata,
                pressure_overhead=_context_pressure_overhead(
                    registered_provider=self._registered_provider,
                    registered_agent=self._registered_agent,
                    registered_environment=self._registered_environment,
                    structured_output=self._structured_output,
                    thinking=self._thinking,
                    step=step,
                ),
                count_input_tokens=self._context_input_token_counter(step=step),
                build_cache_prefix_request=self._cache_prefix_request_builder(step=step),
                secret_redactor=self._executor._secret_redactor,
                run_compaction=run_automatic_compaction,
                force_bounded_compaction=True,
            )
        except ContextBuildError as exc:
            (
                context_failure_events,
                context_failure_persistence,
            ) = await self._context_build_failure_events(
                exc,
                model_step_identity=model_step_identity,
                compaction_identity_ledger=compaction_identity_ledger,
                published_compaction_attempt_ids=published_compaction_attempt_ids,
                compaction_completion_events=compaction_completion_events,
                compaction_start_event=(
                    compaction_start_events[0] if compaction_start_events else None
                ),
                compaction_started_published=any(
                    event.type == EventType.CONTEXT_COMPACTION_STARTED
                    for event in compaction_budget_events
                ),
            )
            for event in compaction_budget_events:
                yield event, None
            for event in context_failure_events:
                yield event, None
            if context_failure_persistence is not None:
                raise context_failure_persistence from exc
            if isinstance(exc.cause, _AutomaticCompactionBudgetReservationFailed):
                settlement_events, settlement_error = await settle_provider_dispatch()
                for event in settlement_events:
                    yield event, None
                if settlement_error is not None:
                    raise settlement_error from exc.cause
                async for event in self._stop_for_budget_reservation_failure(
                    result=exc.cause.result,
                    messages=messages,
                ):
                    yield event, None
                yield None, ModelStepFlowOutcome(stop_session=True)
                return
            if isinstance(exc.cause, _AutomaticCompactionAdmissionStopped):
                settlement_events, settlement_error = await settle_provider_dispatch()
                for event in settlement_events:
                    yield event, None
                if settlement_error is not None:
                    raise settlement_error from exc.cause
                admission_events = self._automatic_compaction_admission_events(
                    exc.cause,
                    messages=messages,
                )
                try:
                    async for event in admission_events:
                        yield event, None
                finally:
                    await _close_async_iterator(admission_events)
                yield None, ModelStepFlowOutcome(stop_session=True)
                return
            yield (
                await self._executor._event_writer.emit(
                    _event_with_model_identity_authority(
                        Event(
                            type=EventType.CONTEXT_OVERFLOW_FAILED,
                            session_id=self._session.id,
                            agent_name=self._registered_agent.spec.name,
                            environment_name=self._environment_name,
                            payload={
                                "step": step,
                                "phase": "context_build",
                                "error": str(exc.cause),
                                "error_type": type(exc.cause).__name__,
                                "policy": type(overflow_policy).__name__,
                                **model_step_identity.payload(),
                            },
                        ),
                        model_step_identity,
                    )
                ),
                None,
            )
            raise exc.cause from exc
        except BaseException as exc:
            await self._persist_context_build_termination_events(
                exc,
                model_step_identity=model_step_identity,
                compaction_identity_ledger=compaction_identity_ledger,
                published_compaction_attempt_ids=published_compaction_attempt_ids,
                compaction_completion_events=compaction_completion_events,
                compaction_start_event=(
                    compaction_start_events[0] if compaction_start_events else None
                ),
                compaction_started_published=any(
                    event.type == EventType.CONTEXT_COMPACTION_STARTED
                    for event in compaction_budget_events
                ),
                cancellation_requests_before_build=context_build_cancellation_requests,
            )
            raise

        context_success_events, context_success_persistence = await self._context_success_events(
            model_step_identity=model_step_identity,
            compaction_identity_ledger=compaction_identity_ledger,
            checkpoint_update=checkpoint_update,
            checkpoint_event_payload=checkpoint_event_payload,
            compaction_telemetry=compaction_telemetry,
            knowledge_telemetry=knowledge_telemetry,
            published_compaction_attempt_ids=published_compaction_attempt_ids,
            compaction_completion_events=compaction_completion_events,
            compaction_start_event=(
                compaction_start_events[0] if compaction_start_events else None
            ),
            compaction_started_published=any(
                event.type == EventType.CONTEXT_COMPACTION_STARTED
                for event in compaction_budget_events
            ),
        )
        for event in compaction_budget_events:
            yield event, None
        for event in context_success_events:
            yield event, None
        if context_success_persistence is not None:
            raise context_success_persistence
        await self._executor._session_control.raise_if_interrupted(self._session.id)
        if _has_provider_backed_context_compaction(compaction_telemetry):
            settlement_events, settlement_error = await settle_provider_dispatch()
            for event in settlement_events:
                yield event, None
            if settlement_error is not None:
                raise settlement_error
            should_stop: bool | None = None
            gate_events = self._post_compaction_gate(
                messages=messages,
                model_step_identity=model_step_identity,
            )
            try:
                async for event, gate_outcome in gate_events:
                    if event is not None:
                        yield event, None
                    if gate_outcome is not None:
                        should_stop = gate_outcome
            finally:
                await _close_async_iterator(gate_events)
            if should_stop is None:
                raise RuntimeError("Post-compaction gate finished without an outcome.")
            if should_stop:
                yield None, ModelStepFlowOutcome(stop_session=True)
                return

        recovery_request = await self._executor.build_request(
            session=self._session,
            registered_agent=self._registered_agent,
            registered_environment=self._registered_environment,
            context_messages=recovery_context_messages,
            structured_output=self._structured_output,
            thinking=self._thinking,
            step=step,
        )
        yield (
            await self._executor._event_writer.emit(
                _event_with_model_identity_authority(
                    Event(
                        type=EventType.CONTEXT_OVERFLOW_RECOVERING,
                        session_id=self._session.id,
                        agent_name=self._registered_agent.spec.name,
                        environment_name=self._environment_name,
                        payload={
                            "step": step,
                            "original_message_count": len(model_request.messages),
                            "recovery_message_count": len(recovery_request.messages),
                            "policy": type(overflow_policy).__name__,
                            **model_step_identity.payload(),
                            **(
                                {}
                                if latest_model_attempt_identity is None
                                else latest_model_attempt_identity.payload()
                            ),
                        },
                    ),
                    latest_model_attempt_identity or model_step_identity,
                )
            ),
            None,
        )
        recovery_events = run_attempt(recovery_request)
        try:
            try:
                async for event, result in recovery_events:
                    yield (
                        event,
                        ModelStepFlowOutcome(assistant_step_result=result)
                        if result is not None
                        else None,
                    )
            except ModelContextOverflowError as exc:
                yield (
                    await self._executor._event_writer.emit(
                        _event_with_model_identity_authority(
                            Event(
                                type=EventType.CONTEXT_OVERFLOW_FAILED,
                                session_id=self._session.id,
                                agent_name=self._registered_agent.spec.name,
                                environment_name=self._environment_name,
                                payload=_context_overflow_event_payload(
                                    exc,
                                    step=step,
                                    phase="recovery",
                                    original_message_count=len(model_request.messages),
                                    recovery_message_count=len(recovery_request.messages),
                                    model_step_identity=model_step_identity,
                                    model_attempt_identity=latest_model_attempt_identity,
                                ),
                            ),
                            latest_model_attempt_identity or model_step_identity,
                        )
                    ),
                    None,
                )
                raise
        finally:
            await _close_async_iterator(recovery_events)

    def _automatic_compaction_admission_events(
        self,
        rejection: _AutomaticCompactionAdmissionStopped,
        *,
        messages: list[Message],
    ) -> AsyncIterator[Event]:
        if rejection.budget_evaluation is not None:
            return self._executor._apply_budget_evaluation(
                ModelStepBudgetEvaluationRequest(
                    evaluation=rejection.budget_evaluation,
                    session=self._session,
                    registered_agent=self._registered_agent,
                    registered_environment=self._registered_environment,
                    environment_name=self._environment_name,
                    messages=messages,
                    run_started_at=self._run_started_at,
                    turn_usage_tracker=self._turn_usage_tracker,
                    active_run=self._active_run,
                )
            )
        if rejection.limit_evaluation is None:
            raise RuntimeError(
                "Automatic compaction admission rejection lost its evaluation."
            ) from rejection
        return self._executor._apply_limit_evaluation(
            ModelStepLimitEvaluationRequest(
                evaluation=rejection.limit_evaluation,
                session=self._session,
                registered_agent=self._registered_agent,
                registered_environment=self._registered_environment,
                environment_name=self._environment_name,
                messages=messages,
                run_started_at=self._run_started_at,
                turn_usage_tracker=self._turn_usage_tracker,
                active_run=self._active_run,
            )
        )

    async def _reconcile_automatic_compaction_events(
        self,
        events: list[Event],
        *,
        cancellation: asyncio.CancelledError | None,
        operation: str,
    ) -> tuple[
        list[bool] | None,
        BaseException | None,
        asyncio.CancelledError | None,
    ]:
        """Read durable event state without losing cancellation during the read."""

        async def reconcile() -> list[bool]:
            return [await self._executor._event_writer.is_persisted(event) for event in events]

        reconciliation_task = asyncio.create_task(reconcile())
        outcome = await await_shielded_task_outcome(
            reconciliation_task,
            cancellation=cancellation,
            timeout_s=_CONTEXT_EVENT_STORE_WAIT_TIMEOUT_S,
        )
        if outcome.timed_out:
            reconciliation_task.cancel()
            reconciliation_task.add_done_callback(_consume_detached_task_outcome)
            reconciliation_error = TimeoutError(
                f"{operation} reconciliation exceeded "
                f"{_CONTEXT_EVENT_STORE_WAIT_TIMEOUT_S:g} seconds."
            )
            return None, reconciliation_error, outcome.cancellation
        reconciliation_error = outcome.error
        if isinstance(reconciliation_error, asyncio.CancelledError):
            reconciliation_error = unexpected_child_cancellation_error(
                reconciliation_error,
                operation=f"{operation} reconciliation",
            )
        if outcome.result is None and reconciliation_error is None:
            reconciliation_error = RuntimeError(f"{operation} reconciliation returned no result.")
        return outcome.result, reconciliation_error, outcome.cancellation

    async def _fan_out_reconciled_automatic_compaction_events(
        self,
        events: list[Event],
        *,
        cancellation: asyncio.CancelledError | None,
        operation: str,
    ) -> tuple[BaseException | None, asyncio.CancelledError | None]:
        """Retry durable side effects with a bounded cancellation-safe wait."""

        fan_out_task = asyncio.create_task(self._executor._event_writer.fan_out_persisted(events))
        outcome = await await_shielded_task_outcome(
            fan_out_task,
            cancellation=cancellation,
            timeout_s=_CONTEXT_EVENT_STORE_WAIT_TIMEOUT_S,
        )
        if outcome.timed_out:
            fan_out_task.cancel()
            fan_out_task.add_done_callback(_consume_detached_task_outcome)
            return (
                TimeoutError(
                    f"{operation} side-effect delivery exceeded "
                    f"{_CONTEXT_EVENT_STORE_WAIT_TIMEOUT_S:g} seconds."
                ),
                outcome.cancellation,
            )
        error = outcome.error
        if isinstance(error, asyncio.CancelledError):
            error = unexpected_child_cancellation_error(
                error,
                operation=f"{operation} side-effect delivery",
            )
        return error, outcome.cancellation

    async def _persist_automatic_compaction_started(
        self,
        telemetry: ContextCompactionTelemetry,
        *,
        published_events: list[Event],
        start_events: list[Event],
        model_step_identity: ModelStepIdentity,
    ) -> None:
        """Make the causal start durable before the first provider dispatch."""

        if telemetry.event_type != EventType.CONTEXT_COMPACTION_STARTED:
            raise TypeError("Automatic compaction start telemetry has the wrong event type.")
        if any(event.type == EventType.CONTEXT_COMPACTION_STARTED for event in published_events):
            return
        if start_events:
            event = start_events[0].model_copy(deep=True)
        else:
            event = _context_compaction_telemetry_event(
                telemetry=telemetry,
                session=self._session,
                registered_agent=self._registered_agent,
                environment_name=self._environment_name,
                execution_identity=model_step_identity,
            )
            start_events.append(event.model_copy(deep=True))
        persistence_task = asyncio.create_task(
            self._executor._event_writer.emit_many(self._session.id, [event])
        )
        outcome = await await_shielded_task_outcome(
            persistence_task,
            timeout_s=_CONTEXT_EVENT_STORE_WAIT_TIMEOUT_S,
        )
        cancellation = outcome.cancellation
        if outcome.timed_out:
            persistence_task.cancel()
            persistence_task.add_done_callback(_consume_detached_task_outcome)
            publication_error: BaseException = TimeoutError(
                "Compaction start publication exceeded "
                f"{_CONTEXT_EVENT_STORE_WAIT_TIMEOUT_S:g} seconds."
            )
        else:
            publication_error = outcome.error
        try:
            if publication_error is not None:
                if isinstance(publication_error, asyncio.CancelledError) and cancellation is None:
                    raise unexpected_child_cancellation_error(
                        publication_error,
                        operation="Compaction start publication",
                    )
                raise publication_error
            persisted = outcome.result
            if persisted is None:
                raise RuntimeError("Compaction start publication returned no result.")
        except BaseException as publication_error:
            try:
                (
                    commit_states,
                    reconciliation_error,
                    cancellation,
                ) = await self._reconcile_automatic_compaction_events(
                    [event],
                    cancellation=cancellation,
                    operation="Compaction start publication",
                )
                if reconciliation_error is not None:
                    raise reconciliation_error
            except BaseException as reconciliation_error:
                publication_error.add_note(
                    "Compaction start publication reconciliation also failed: "
                    f"{type(reconciliation_error).__name__}: {reconciliation_error}"
                )
                if cancellation is not None:
                    cancellation.add_note(
                        "Compaction start publication and reconciliation also "
                        "failed during cancellation."
                    )
                    raise cancellation from publication_error
                raise publication_error from reconciliation_error
            if commit_states is None:
                raise AssertionError(
                    "Compaction start reconciliation lost its result."
                ) from publication_error
            if not commit_states[0]:
                if cancellation is not None:
                    cancellation.add_note(
                        "Compaction start could not be confirmed durable during cancellation."
                    )
                    raise cancellation from publication_error
                raise publication_error
            published_events.append(event.model_copy(deep=True))
            (
                fan_out_error,
                cancellation,
            ) = await self._fan_out_reconciled_automatic_compaction_events(
                [event],
                cancellation=cancellation,
                operation="Compaction start publication",
            )
            if fan_out_error is not None:
                publication_error.add_note(
                    "Committed compaction start side-effect delivery also failed: "
                    f"{type(fan_out_error).__name__}: {fan_out_error}"
                )
            publication_error.add_note(
                "Compaction start was durable; no provider dispatch followed the "
                "failed publication acknowledgement."
            )
            if cancellation is not None:
                cancellation.add_note("Compaction start was durable before cancellation.")
                raise cancellation from publication_error
            raise publication_error
        published_events.extend(persisted)
        if cancellation is not None:
            raise cancellation

    async def _persist_automatic_compaction_completions(
        self,
        payloads: list[dict[str, Any]],
        *,
        published_attempt_ids: set[str],
        published_events: list[Event],
        completion_events: dict[str, Event],
    ) -> None:
        """Commit finalized provider evidence before another compactor dispatch."""

        pending: list[tuple[str, Event]] = []
        for payload in payloads:
            attempt_id = payload.get(_COMPACTION_ATTEMPT_ID_KEY)
            if type(attempt_id) is not str:
                raise RuntimeError("Compaction completion evidence lost its attempt identity.")
            if attempt_id in published_attempt_ids:
                continue
            event = completion_events.get(attempt_id)
            if event is None:
                try:
                    execution_identity = ModelAttemptIdentity.model_validate(
                        {
                            "model_step_id": payload.get("model_step_id"),
                            "model_attempt_id": payload.get("model_attempt_id"),
                        }
                    )
                except (TypeError, ValueError):
                    raise ValueError(
                        "Compaction completion carries an invalid model attempt identity."
                    ) from None
                event = _context_compaction_telemetry_event(
                    telemetry=ContextCompactionTelemetry(
                        event_type=EventType.MODEL_COMPLETED,
                        payload=payload,
                    ),
                    session=self._session,
                    registered_agent=self._registered_agent,
                    environment_name=self._environment_name,
                    execution_identity=execution_identity,
                )
                completion_events[attempt_id] = event.model_copy(deep=True)
            pending.append(
                (
                    attempt_id,
                    event.model_copy(deep=True),
                )
            )
        if not pending:
            return

        events = [event for _attempt_id, event in pending]
        persistence_task = asyncio.create_task(
            self._executor._event_writer.persist_many(self._session.id, events)
        )
        outcome = await await_shielded_task_outcome(
            persistence_task,
            timeout_s=_CONTEXT_EVENT_STORE_WAIT_TIMEOUT_S,
        )
        cancellation = outcome.cancellation
        if outcome.timed_out:
            persistence_task.cancel()
            # Only this physical write receives unbounded ownership. Sink and
            # budget side-effect delivery happens separately below and remains
            # bounded because the durable handoff can recover it after restart.
            drain_outcome = await await_shielded_task_outcome(
                persistence_task,
                cancellation=cancellation,
            )
            cancellation = drain_outcome.cancellation
            publication_error: BaseException = TimeoutError(
                "Compaction completion publication exceeded "
                f"{_CONTEXT_EVENT_STORE_WAIT_TIMEOUT_S:g} seconds."
            )
        else:
            publication_error = outcome.error
        try:
            if publication_error is not None:
                if isinstance(publication_error, asyncio.CancelledError) and cancellation is None:
                    raise unexpected_child_cancellation_error(
                        publication_error,
                        operation="Compaction completion publication",
                    )
                raise publication_error
            persisted = outcome.result
            if persisted is None:
                raise RuntimeError("Compaction completion publication returned no result.")
        except BaseException as publication_error:
            try:
                (
                    commit_states,
                    reconciliation_error,
                    cancellation,
                ) = await self._reconcile_automatic_compaction_events(
                    events,
                    cancellation=cancellation,
                    operation="Compaction completion publication",
                )
                if reconciliation_error is not None:
                    raise reconciliation_error
            except BaseException as reconciliation_error:
                publication_error.add_note(
                    "Compaction completion publication reconciliation also failed: "
                    f"{type(reconciliation_error).__name__}: {reconciliation_error}"
                )
                if cancellation is not None:
                    cancellation.add_note(
                        "Compaction completion publication and reconciliation also "
                        "failed during cancellation."
                    )
                    raise cancellation from publication_error
                raise publication_error from reconciliation_error
            if commit_states is None:
                raise AssertionError(
                    "Compaction completion reconciliation lost its result."
                ) from publication_error
            if not all(commit_states):
                if any(commit_states):
                    publication_error.add_note(
                        "The event store violated atomic compaction completion publication."
                    )
                if cancellation is not None:
                    cancellation.add_note(
                        "Compaction completion evidence could not be confirmed durable "
                        "during cancellation."
                    )
                    raise cancellation from publication_error
                raise publication_error
            # The provider evidence reached the durable handoff even though the
            # publication acknowledgement or a downstream side effect failed.
            # Remember it before propagating the failure so no failure path can
            # publish a duplicate completion and no retry can dispatch again.
            published_attempt_ids.update(attempt_id for attempt_id, _event in pending)
            published_events.extend(event.model_copy(deep=True) for event in events)
            (
                fan_out_error,
                cancellation,
            ) = await self._fan_out_reconciled_automatic_compaction_events(
                events,
                cancellation=cancellation,
                operation="Compaction completion publication",
            )
            if fan_out_error is not None:
                publication_error.add_note(
                    "Committed compaction completion side-effect delivery also failed: "
                    f"{type(fan_out_error).__name__}: {fan_out_error}"
                )
            publication_error.add_note(
                "Compaction completion evidence was durable; the operation will "
                "fail closed without another provider dispatch."
            )
            if cancellation is not None:
                cancellation.add_note(
                    "Compaction completion evidence was durable before cancellation."
                )
                raise cancellation from publication_error
            raise publication_error
        published_attempt_ids.update(attempt_id for attempt_id, _event in pending)
        published_events.extend(persisted)
        (
            fan_out_error,
            cancellation,
        ) = await self._fan_out_reconciled_automatic_compaction_events(
            events,
            cancellation=cancellation,
            operation="Compaction completion publication",
        )
        if cancellation is not None:
            if fan_out_error is not None:
                cancellation.add_note(
                    "Committed compaction completion side-effect delivery also failed "
                    f"during cancellation: {type(fan_out_error).__name__}: "
                    f"{fan_out_error}"
                )
                raise cancellation from fan_out_error
            raise cancellation
        if fan_out_error is not None:
            raise fan_out_error

    async def _emit_context_events_reconciling_late_start(
        self,
        events: list[Event],
        *,
        compaction_start_event: Event | None,
    ) -> list[Event]:
        """Persist a batch without duplicating a concurrently committed start."""

        try:
            return await self._executor._event_writer.emit_many(self._session.id, events)
        except BaseException as publication_error:
            if (
                not isinstance(publication_error, ValueError)
                or compaction_start_event is None
                or all(event.id != compaction_start_event.id for event in events)
            ):
                raise
            try:
                start_durable = await self._executor._event_writer.is_persisted(
                    compaction_start_event
                )
                remaining = [event for event in events if event.id != compaction_start_event.id]
                remaining_states = [
                    await self._executor._event_writer.is_persisted(event) for event in remaining
                ]
            except BaseException as reconciliation_error:
                publication_error.add_note(
                    "Context start conflict reconciliation also failed: "
                    f"{type(reconciliation_error).__name__}: {reconciliation_error}"
                )
                raise publication_error from reconciliation_error
            if not start_durable:
                raise
            if any(remaining_states):
                if not all(remaining_states):
                    publication_error.add_note(
                        "The context store violated atomic event-batch publication."
                    )
                    raise publication_error
                persisted_remaining = [event.model_copy(deep=True) for event in remaining]
            else:
                try:
                    persisted_remaining = await self._executor._event_writer.persist_many(
                        self._session.id,
                        remaining,
                    )
                except BaseException as retry_error:
                    retry_error.add_note(
                        "Context event publication retried after the original compaction "
                        "start committed concurrently."
                    )
                    raise retry_error from publication_error
            reconciled = [compaction_start_event.model_copy(deep=True), *persisted_remaining]
            await self._executor._event_writer.fan_out_persisted(reconciled)
            return [
                next(event for event in reconciled if event.id == requested.id).model_copy(
                    deep=True
                )
                for requested in events
            ]

    async def _persist_context_events(
        self,
        *,
        model_step_identity: ModelStepIdentity,
        compaction_identity_ledger: _CompactionExecutionIdentityLedger,
        compaction_telemetry: list[ContextCompactionTelemetry],
        knowledge_telemetry: list[ContextKnowledgeTelemetry],
        checkpoint_update: dict[str, Any] | None,
        checkpoint_event_payload: dict[str, Any] | None,
        published_compaction_attempt_ids: set[str],
        compaction_completion_events: dict[str, Event],
        compaction_start_event: Event | None,
        compaction_started_published: bool,
        checkpoint_invariant_cause: BaseException | None = None,
    ) -> tuple[list[Event], BaseException | None]:
        """Persist one context outcome completely before exposing its first event."""

        model_step_identity = copy_model_step_identity(model_step_identity)
        reconciled_start_events: list[Event] = []
        compaction_start_durable = compaction_started_published
        if not compaction_start_durable and compaction_start_event is not None:
            (
                commit_states,
                reconciliation_error,
                cancellation,
            ) = await self._reconcile_automatic_compaction_events(
                [compaction_start_event],
                cancellation=None,
                operation="Compaction start cleanup",
            )
            if cancellation is not None:
                if reconciliation_error is not None:
                    raise cancellation from reconciliation_error
                raise cancellation
            if reconciliation_error is not None:
                return [], reconciliation_error
            if commit_states is None:
                return [], RuntimeError(
                    "Compaction start cleanup reconciliation returned no result."
                )
            compaction_start_durable = commit_states[0]
            if compaction_start_durable:
                reconciled_start_events.append(compaction_start_event.model_copy(deep=True))

        prepared_events: list[Event] = []
        for telemetry in compaction_telemetry:
            if (
                telemetry.event_type == EventType.MODEL_COMPLETED
                and telemetry.payload.get(_COMPACTION_ATTEMPT_ID_KEY)
                in published_compaction_attempt_ids
            ) or (
                telemetry.event_type == EventType.CONTEXT_COMPACTION_STARTED
                and compaction_start_durable
            ):
                continue
            compaction_attempt_id = telemetry.payload.get(_COMPACTION_ATTEMPT_ID_KEY)
            event = (
                compaction_completion_events.get(compaction_attempt_id)
                if telemetry.event_type == EventType.MODEL_COMPLETED
                and type(compaction_attempt_id) is str
                else (
                    compaction_start_event
                    if telemetry.event_type == EventType.CONTEXT_COMPACTION_STARTED
                    else None
                )
            )
            if event is None:
                execution_identity: ModelStepIdentity | ModelAttemptIdentity = model_step_identity
                if telemetry.event_type == EventType.MODEL_COMPLETED:
                    identified_payload = compaction_identity_ledger.identify_payloads(
                        [telemetry.payload]
                    )[0]
                    telemetry = ContextCompactionTelemetry(
                        event_type=telemetry.event_type,
                        payload=identified_payload,
                    )
                    execution_identity = ModelAttemptIdentity.model_validate(
                        {
                            "model_step_id": identified_payload.get("model_step_id"),
                            "model_attempt_id": identified_payload.get("model_attempt_id"),
                        }
                    )
                event = _context_compaction_telemetry_event(
                    telemetry=telemetry,
                    session=self._session,
                    registered_agent=self._registered_agent,
                    environment_name=self._environment_name,
                    execution_identity=execution_identity,
                )
                if (
                    telemetry.event_type == EventType.MODEL_COMPLETED
                    and type(compaction_attempt_id) is str
                ):
                    compaction_completion_events[compaction_attempt_id] = event.model_copy(
                        deep=True
                    )
            prepared_events.append(event.model_copy(deep=True))
        prepared_events.extend(
            _context_knowledge_telemetry_event(
                telemetry=telemetry,
                session=self._session,
                registered_agent=self._registered_agent,
                environment_name=self._environment_name,
                model_step_identity=model_step_identity,
            )
            for telemetry in knowledge_telemetry
        )

        async def persist() -> tuple[list[Event], BaseException | None]:
            if checkpoint_event_payload is None:
                persisted = await self._emit_context_events_reconciling_late_start(
                    prepared_events,
                    compaction_start_event=compaction_start_event,
                )
                if reconciled_start_events:
                    await self._executor._event_writer.fan_out_persisted(reconciled_start_events)
                return [*reconciled_start_events, *persisted], None
            if checkpoint_update is None:
                error = RuntimeError("Context checkpoint event payload requires checkpoint state.")
                if checkpoint_invariant_cause is not None:
                    error.__cause__ = checkpoint_invariant_cause
                return [], error

            checkpoint_event = Event(
                type=EventType.SESSION_CHECKPOINTED,
                session_id=self._session.id,
                agent_name=self._registered_agent.spec.name,
                environment_name=self._environment_name,
                payload={
                    **checkpoint_event_payload,
                    **model_step_identity.payload(),
                },
            )
            atomic_events = self._executor._event_writer.prepare_many(
                [*prepared_events, checkpoint_event]
            )
            checkpoint_transform = self._executor._checkpoint_transform(checkpoint_update)
            try:
                await self._executor._session_store.publish_checkpoint_and_events(
                    self._session.id,
                    checkpoint_transform=checkpoint_transform,
                    events=atomic_events,
                )
            except BaseException as publication_error:
                try:
                    event_commit_states = [
                        await self._executor._event_writer.is_persisted(event)
                        for event in atomic_events
                    ]
                    events_committed = all(event_commit_states)
                    durable_checkpoint = await self._executor._session_store.load_checkpoint(
                        self._session.id
                    )
                    durable_session = await self._executor._session_store.load(self._session.id)
                    expected_checkpoint = (
                        None
                        if durable_checkpoint is None or durable_session is None
                        else checkpoint_transform(durable_session, durable_checkpoint)
                    )
                    checkpoint_committed = (
                        durable_checkpoint is not None and expected_checkpoint == durable_checkpoint
                    )
                except BaseException as reconciliation_error:
                    publication_error.add_note(
                        "Context checkpoint publication reconciliation also failed: "
                        f"{type(reconciliation_error).__name__}: {reconciliation_error}"
                    )
                    return [], publication_error
                if not (events_committed and checkpoint_committed):
                    if events_committed != checkpoint_committed:
                        publication_error.add_note(
                            "The context store violated atomic checkpoint/event publication."
                        )
                    return [], publication_error
            await self._executor._event_writer.fan_out_persisted(
                [*reconciled_start_events, *atomic_events]
            )
            return [
                *(event.model_copy(deep=True) for event in reconciled_start_events),
                *(event.model_copy(deep=True) for event in atomic_events),
            ], None

        persistence_task = asyncio.create_task(persist())
        outcome = await await_shielded_task_outcome(persistence_task)
        cancellation = outcome.cancellation
        if cancellation is not None:
            if outcome.error is not None:
                cancellation.add_note(
                    "Context outcome persistence also failed during cancellation: "
                    f"{type(outcome.error).__name__}."
                )
                raise cancellation from outcome.error
            persisted_outcome = outcome.result
            if persisted_outcome is None:
                persistence_error = RuntimeError("Context persistence returned no result.")
                raise cancellation from persistence_error
            _, persistence_failure = persisted_outcome
            if persistence_failure is not None:
                cancellation.add_note(
                    "Context checkpoint persistence also failed during cancellation: "
                    f"{type(persistence_failure).__name__}."
                )
                raise cancellation from persistence_failure
            raise cancellation
        if outcome.error is not None:
            if isinstance(outcome.error, asyncio.CancelledError):
                return (
                    [],
                    unexpected_child_cancellation_error(
                        outcome.error,
                        operation="Context outcome persistence",
                    ),
                )
            return [], outcome.error
        if outcome.result is None:
            return [], RuntimeError("Context persistence returned no result.")
        persisted_events, persistence_failure = outcome.result
        if isinstance(persistence_failure, asyncio.CancelledError):
            persistence_failure = unexpected_child_cancellation_error(
                persistence_failure,
                operation="Context outcome persistence",
            )
        return persisted_events, persistence_failure

    async def _context_build_failure_events(
        self,
        error: ContextBuildError,
        *,
        model_step_identity: ModelStepIdentity,
        compaction_identity_ledger: _CompactionExecutionIdentityLedger,
        published_compaction_attempt_ids: set[str],
        compaction_completion_events: dict[str, Event],
        compaction_start_event: Event | None,
        compaction_started_published: bool,
    ) -> tuple[list[Event], BaseException | None]:
        return await self._persist_context_events(
            model_step_identity=model_step_identity,
            compaction_identity_ledger=compaction_identity_ledger,
            compaction_telemetry=list(error.compaction_telemetry),
            knowledge_telemetry=list(error.knowledge_telemetry),
            checkpoint_update=error.checkpoint,
            checkpoint_event_payload=error.checkpoint_event_payload,
            published_compaction_attempt_ids=published_compaction_attempt_ids,
            compaction_completion_events=compaction_completion_events,
            compaction_start_event=compaction_start_event,
            compaction_started_published=compaction_started_published,
            checkpoint_invariant_cause=error,
        )

    async def _persist_context_build_termination_events(
        self,
        error: BaseException,
        *,
        model_step_identity: ModelStepIdentity,
        compaction_identity_ledger: _CompactionExecutionIdentityLedger,
        published_compaction_attempt_ids: set[str],
        compaction_completion_events: dict[str, Event],
        compaction_start_event: Event | None,
        compaction_started_published: bool,
        cancellation_requests_before_build: int,
    ) -> None:
        """Persist completed compaction evidence without replacing a fatal signal."""

        model_step_identity = copy_model_step_identity(model_step_identity)
        telemetry = context_build_termination_compaction_telemetry(error)
        compaction_start_durable = compaction_started_published
        if not compaction_start_durable and compaction_start_event is not None:
            if isinstance(error, asyncio.CancelledError):
                # Remove only the cancellation already represented by ``error``.
                # The bounded reconciliation below must observe a genuinely later
                # Task.cancel() as a distinct signal rather than folding it into
                # the historical/provider cancellation.
                consume_pending_task_cancellation(
                    error,
                    preserve_requests=cancellation_requests_before_build,
                )
            (
                commit_states,
                reconciliation_error,
                reconciliation_cancellation,
            ) = await self._reconcile_automatic_compaction_events(
                [compaction_start_event],
                cancellation=None,
                operation="Compaction start termination cleanup",
            )
            if reconciliation_error is not None:
                error.add_note(
                    "Context compaction start reconciliation also failed during "
                    f"termination: {type(reconciliation_error).__name__}: "
                    f"{reconciliation_error}"
                )
            elif commit_states is not None:
                compaction_start_durable = commit_states[0]
            if reconciliation_cancellation is not None and reconciliation_cancellation is not error:
                raise BaseExceptionGroup(
                    "Context compaction start reconciliation observed a later cancellation.",
                    [error, reconciliation_cancellation],
                )
        unpublished_telemetry = [
            item
            for item in telemetry
            if not (
                (
                    item.event_type == EventType.MODEL_COMPLETED
                    and item.payload.get(_COMPACTION_ATTEMPT_ID_KEY)
                    in published_compaction_attempt_ids
                )
                or (
                    item.event_type == EventType.CONTEXT_COMPACTION_STARTED
                    and compaction_start_durable
                )
            )
        ]
        if not unpublished_telemetry:
            return

        async def persist() -> None:
            events: list[Event] = []
            for item in unpublished_telemetry:
                compaction_attempt_id = item.payload.get(_COMPACTION_ATTEMPT_ID_KEY)
                if (
                    item.event_type == EventType.MODEL_COMPLETED
                    and type(compaction_attempt_id) is str
                    and compaction_attempt_id in compaction_completion_events
                ):
                    events.append(
                        compaction_completion_events[compaction_attempt_id].model_copy(deep=True)
                    )
                    continue
                if (
                    item.event_type == EventType.CONTEXT_COMPACTION_STARTED
                    and compaction_start_event is not None
                ):
                    events.append(compaction_start_event.model_copy(deep=True))
                    continue
                execution_identity: ModelStepIdentity | ModelAttemptIdentity = model_step_identity
                if item.event_type == EventType.MODEL_COMPLETED:
                    identified_payload = compaction_identity_ledger.identify_payloads(
                        [item.payload]
                    )[0]
                    item = ContextCompactionTelemetry(
                        event_type=item.event_type,
                        payload=identified_payload,
                    )
                    execution_identity = ModelAttemptIdentity.model_validate(
                        {
                            "model_step_id": identified_payload.get("model_step_id"),
                            "model_attempt_id": identified_payload.get("model_attempt_id"),
                        }
                    )
                events.append(
                    _context_compaction_telemetry_event(
                        telemetry=item,
                        session=self._session,
                        registered_agent=self._registered_agent,
                        environment_name=self._environment_name,
                        execution_identity=execution_identity,
                    )
                )
            await self._emit_context_events_reconciling_late_start(
                events,
                compaction_start_event=compaction_start_event,
            )

        if isinstance(error, asyncio.CancelledError):
            # This cancellation has already crossed the context-build boundary and
            # remains authoritative. Clear only its task-level delivery state before
            # shielding cleanup so a genuinely later Task.cancel() is observed as a
            # distinct signal instead of being normalized back into ``error``.
            consume_pending_task_cancellation(
                error,
                preserve_requests=cancellation_requests_before_build,
            )
        task = asyncio.create_task(persist())
        outcome = await await_shielded_task_outcome(
            task,
            timeout_s=_CONTEXT_TERMINATION_PERSIST_TIMEOUT_S,
        )
        later_cancellation = (
            outcome.cancellation
            if outcome.cancellation is not None and outcome.cancellation is not error
            else None
        )
        if outcome.timed_out:
            task.add_done_callback(_consume_detached_task_outcome)
            task.cancel()
            error.add_note(
                "Context compaction termination telemetry persistence exceeded "
                f"{_CONTEXT_TERMINATION_PERSIST_TIMEOUT_S:g} seconds."
            )
            if later_cancellation is not None:
                raise BaseExceptionGroup(
                    "Context termination telemetry timed out after a later cancellation.",
                    [error, later_cancellation],
                )
            return
        if outcome.error is not None:
            error.add_note(
                "Context compaction termination telemetry also failed to persist: "
                f"{type(outcome.error).__name__}: {outcome.error}"
            )
            if later_cancellation is not None:
                raise BaseExceptionGroup(
                    "Context termination telemetry failed after a later cancellation.",
                    [error, outcome.error, later_cancellation],
                )
        elif later_cancellation is not None:
            raise BaseExceptionGroup(
                "Context termination telemetry completed after a later cancellation.",
                [error, later_cancellation],
            )

    async def _context_success_events(
        self,
        *,
        model_step_identity: ModelStepIdentity,
        compaction_identity_ledger: _CompactionExecutionIdentityLedger,
        checkpoint_update: dict[str, Any] | None,
        checkpoint_event_payload: dict[str, Any] | None,
        compaction_telemetry: list[ContextCompactionTelemetry],
        knowledge_telemetry: list[ContextKnowledgeTelemetry],
        published_compaction_attempt_ids: set[str],
        compaction_completion_events: dict[str, Event],
        compaction_start_event: Event | None,
        compaction_started_published: bool,
    ) -> tuple[list[Event], BaseException | None]:
        return await self._persist_context_events(
            model_step_identity=model_step_identity,
            compaction_identity_ledger=compaction_identity_ledger,
            compaction_telemetry=compaction_telemetry,
            knowledge_telemetry=knowledge_telemetry,
            checkpoint_update=checkpoint_update,
            checkpoint_event_payload=checkpoint_event_payload,
            published_compaction_attempt_ids=published_compaction_attempt_ids,
            compaction_completion_events=compaction_completion_events,
            compaction_start_event=compaction_start_event,
            compaction_started_published=compaction_started_published,
        )

    async def _post_compaction_gate(
        self,
        *,
        messages: list[Message],
        model_step_identity: ModelStepIdentity,
    ) -> AsyncIterator[tuple[Event | None, bool | None]]:
        model_step_identity = copy_model_step_identity(model_step_identity)
        budget_evaluation = await self._limit_gate.evaluate_budget(
            self._budget_policy,
            execution_identity=model_step_identity,
        )
        request = ModelStepBudgetEvaluationRequest(
            evaluation=budget_evaluation,
            session=self._session,
            registered_agent=self._registered_agent,
            registered_environment=self._registered_environment,
            environment_name=self._environment_name,
            messages=messages,
            run_started_at=self._run_started_at,
            turn_usage_tracker=self._turn_usage_tracker,
            active_run=self._active_run,
        )
        budget_events = self._executor._apply_budget_evaluation(request)
        try:
            async for event in budget_events:
                yield event, None
        finally:
            await _close_async_iterator(budget_events)
        if budget_evaluation.check is not None:
            yield None, True
            return
        limit_evaluation = await self._limit_gate.evaluate_limits(
            execution_identity=model_step_identity,
        )
        request = ModelStepLimitEvaluationRequest(
            evaluation=limit_evaluation,
            session=self._session,
            registered_agent=self._registered_agent,
            registered_environment=self._registered_environment,
            environment_name=self._environment_name,
            messages=messages,
            run_started_at=self._run_started_at,
            turn_usage_tracker=self._turn_usage_tracker,
            active_run=self._active_run,
        )
        limit_events = self._executor._apply_limit_evaluation(request)
        try:
            async for event in limit_events:
                yield event, None
        finally:
            await _close_async_iterator(limit_events)
        yield None, limit_evaluation.decision is not None

    async def _billing_identity_budget_gate(
        self,
        *,
        messages: list[Message],
        billing_identity: BillingIdentity | None,
        model_attempt_identity: ModelAttemptIdentity,
    ) -> AsyncIterator[tuple[Event | None, bool | None]]:
        model_attempt_identity = copy_model_attempt_identity(model_attempt_identity)
        budget_evaluation = await self._limit_gate.evaluate_budget(
            self._budget_policy,
            billing_identity_state=resolved_billing_identity(billing_identity),
            execution_identity=model_attempt_identity,
        )
        request = ModelStepBudgetEvaluationRequest(
            evaluation=budget_evaluation,
            session=self._session,
            registered_agent=self._registered_agent,
            registered_environment=self._registered_environment,
            environment_name=self._environment_name,
            messages=messages,
            run_started_at=self._run_started_at,
            turn_usage_tracker=self._turn_usage_tracker,
            active_run=self._active_run,
        )
        budget_events = self._executor._apply_budget_evaluation(request)
        try:
            async for event in budget_events:
                yield event, None
        finally:
            await _close_async_iterator(budget_events)
        if budget_evaluation.check is not None:
            yield None, True
            return
        limit_evaluation = await self._limit_gate.evaluate_limits(
            billing_identity_state=resolved_billing_identity(billing_identity),
            execution_identity=model_attempt_identity,
        )
        request = ModelStepLimitEvaluationRequest(
            evaluation=limit_evaluation,
            session=self._session,
            registered_agent=self._registered_agent,
            registered_environment=self._registered_environment,
            environment_name=self._environment_name,
            messages=messages,
            run_started_at=self._run_started_at,
            turn_usage_tracker=self._turn_usage_tracker,
            active_run=self._active_run,
        )
        limit_events = self._executor._apply_limit_evaluation(request)
        try:
            async for event in limit_events:
                yield event, None
        finally:
            await _close_async_iterator(limit_events)
        yield None, limit_evaluation.decision is not None

    def _has_deferred_contextual_price(self) -> bool:
        return self._deferred_contextual_price

    async def _stop_for_budget_reservation_failure(
        self,
        *,
        result: BudgetReservationResult,
        messages: list[Message],
    ) -> AsyncIterator[Event]:
        request = ModelStepBudgetReservationFailureRequest(
            result=result,
            session=self._session,
            registered_agent=self._registered_agent,
            registered_environment=self._registered_environment,
            environment_name=self._environment_name,
            messages=messages,
            run_started_at=self._run_started_at,
            turn_usage_tracker=self._turn_usage_tracker,
            active_run=self._active_run,
        )
        terminal_events = self._executor._stop_for_budget_reservation_failure(request)
        try:
            async for event in terminal_events:
                yield event
        finally:
            await _close_async_iterator(terminal_events)

    def _context_input_token_counter(
        self,
        *,
        step: int,
    ) -> Callable[[list[Message]], Awaitable[int | None]]:
        async def count_input_tokens(context_messages: list[Message]) -> int | None:
            request = await self._executor.build_request(
                session=self._session,
                registered_agent=self._registered_agent,
                registered_environment=self._registered_environment,
                context_messages=copy_context_messages(context_messages),
                structured_output=self._structured_output,
                thinking=self._thinking,
                step=step,
            )
            result = await self._provider.count_input_tokens(request)
            return None if result is None else result.input_tokens

        return count_input_tokens

    def _cache_prefix_request_builder(
        self,
        *,
        step: int,
    ) -> Callable[[list[Message]], Awaitable[ModelRequest]]:
        async def build_cache_prefix_request(context_messages: list[Message]) -> ModelRequest:
            return await self._executor.build_request(
                session=self._session,
                registered_agent=self._registered_agent,
                registered_environment=self._registered_environment,
                context_messages=copy_context_messages(context_messages),
                structured_output=self._structured_output,
                thinking=self._thinking,
                step=step,
            )

        return build_cache_prefix_request

    async def _run_automatic_compaction_with_budget(
        self,
        *,
        compactor: ContextCompactor,
        compaction_request: CompactionRequest,
        execute: Callable[[], Awaitable[CompactionResult]],
        completed_payloads: Callable[[], list[dict[str, Any]]],
        budget_events: list[Event],
        messages: list[Message],
        model_step_identity: ModelStepIdentity,
        compaction_identity_ledger: _CompactionExecutionIdentityLedger,
    ) -> CompactionResult:
        del messages
        model_step_identity = copy_model_step_identity(model_step_identity)
        if compaction_identity_ledger.model_step_identity != model_step_identity:
            raise ValueError("Compaction identity ledger belongs to a different model step.")
        controller = self._executor._run_limit_controller
        all_limits = controller.provider_budget_limits(
            session=self._session,
            agent_name=self._registered_agent.spec.name,
            budget_policy=self._budget_policy,
            request_budget_limits=self._request_budget_limits,
        )
        has_accounting_limits = bool(all_limits) or self._limit_gate.has_run_limits()
        strict_contextual_candidates = tuple(
            limit
            for limit in all_limits
            if limit.action == "interrupt"
            and not limit.allow_unpriced
            and any(price.pricing_context is not None for price in limit.pricing.prices)
        )

        async def run_identity_only_dispatch(
            actual_pricing_provider_name: str,
            actual_model: str,
            actual_usage_dialect: UsageDialect,
            billing_identity: BillingIdentity | None,
            dispatch: Callable[[], Awaitable[tuple[str, dict[str, Any]]]],
        ) -> tuple[str, dict[str, Any]]:
            """Identify a built-in dispatch reached through an opaque wrapper."""

            del (
                actual_pricing_provider_name,
                actual_model,
                actual_usage_dialect,
                billing_identity,
            )
            model_attempt_identity = compaction_identity_ledger.begin_dispatch()
            try:
                with _compaction_model_attempt_identity_scope(model_attempt_identity):
                    return await dispatch()
            finally:
                compaction_identity_ledger.end_dispatch(model_attempt_identity)

        try:
            identity = compactor._provider_budget_identity_for_request(compaction_request)
        except NotImplementedError as exc:
            if not has_accounting_limits:
                with _automatic_compaction_dispatch_runner_scope(run_identity_only_dispatch):
                    return await execute()
            raise RuntimeError(
                "Automatic provider-backed compaction under run or budget limits requires the "
                "ContextCompactor to declare provider_budget_identity(session), "
                "returning provider/model or None for deterministic execution."
            ) from exc
        uses_dispatch_boundary = compactor._uses_runtime_provider_dispatch_runner_for_request(
            compaction_request
        )
        if identity is None:
            if uses_dispatch_boundary:
                raise RuntimeError(
                    "Provider-backed compaction cannot declare a deterministic budget "
                    "identity under run or cost limits."
                )
            with _automatic_compaction_dispatch_runner_scope(run_identity_only_dispatch):
                return await execute()
        if type(identity) is not tuple or len(identity) != 2:
            raise TypeError(
                "ContextCompactor.provider_budget_identity must return a "
                "(provider_name, model) tuple or None."
            )
        pricing_provider_name = require_durable_clean_nonblank(
            identity[0],
            "compactor_provider_name",
        )
        declared_model = require_durable_clean_nonblank(
            identity[1],
            "compactor_model",
        )
        contextual_limits = tuple(
            limit
            for limit in strict_contextual_candidates
            if has_deferred_contextual_price(
                limit.pricing,
                provider_name=pricing_provider_name,
                model=declared_model,
            )
        )
        limits = tuple(
            limit
            for limit in all_limits
            if limit.reservation is not None or limit in contextual_limits
        )
        if not uses_dispatch_boundary and has_accounting_limits:
            raise RuntimeError(
                "Automatic provider-backed compaction under run or budget limits cannot "
                f"safely run opaque provider-backed compactor {type(compactor).__name__}: "
                "Cayu cannot admit each provider dispatch independently. Use an "
                "unmodified built-in provider compactor or remove the applicable "
                "run and budget limits."
            )
        # A wrapper around a built-in compactor is opaque for admission: Cayu
        # cannot prove that it exposes *every* provider call. When no accounting
        # limit needs that proof, the inner runner below still identifies built-in
        # calls reached through the wrapper. Custom completion payloads with no
        # observed dispatch continue to fail closed.

        policy_limits = budget_limits_for_session(
            policy=self._budget_policy,
            agent_name=self._registered_agent.spec.name,
            causal_budget_id=self._session.causal_budget_id,
        )
        dispatch_policy_limits = tuple(limit for limit in policy_limits if limit not in limits)
        dispatch_request_limits = tuple(
            limit for limit in self._request_budget_limits if limit not in limits
        )

        async def run_provider_dispatch(
            actual_pricing_provider_name: str,
            actual_model: str,
            actual_usage_dialect: UsageDialect,
            billing_identity: BillingIdentity | None,
            dispatch: Callable[[], Awaitable[tuple[str, dict[str, Any]]]],
        ) -> tuple[str, dict[str, Any]]:
            del actual_usage_dialect
            actual_pricing_provider_name = require_durable_clean_nonblank(
                actual_pricing_provider_name,
                "compactor_provider_name",
            )
            actual_model = require_durable_clean_nonblank(
                actual_model,
                "compactor_model",
            )
            if actual_pricing_provider_name != pricing_provider_name:
                raise RuntimeError(
                    "Compaction dispatch provider identity differs from its admitted identity."
                )
            if actual_model != declared_model:
                raise RuntimeError(
                    "Compaction dispatch model identity differs from its admitted identity."
                )
            model_attempt_identity = compaction_identity_ledger.begin_dispatch()
            before_count = len(completed_payloads())
            try:

                async def identified_dispatch() -> tuple[str, dict[str, Any]]:
                    with _compaction_model_attempt_identity_scope(model_attempt_identity):
                        return await dispatch()

                def completion_events(payloads: list[dict[str, Any]]) -> list[Event]:
                    identified = compaction_identity_ledger.identify_payloads(
                        payloads,
                        expected_identity=model_attempt_identity,
                    )
                    return [
                        Event(
                            type=EventType.MODEL_COMPLETED,
                            session_id=self._session.id,
                            agent_name=self._registered_agent.spec.name,
                            environment_name=self._environment_name,
                            payload=payload,
                        )
                        for payload in identified
                    ]

                prior_completion_events = [
                    event.model_copy(deep=True)
                    for event in budget_events
                    if event.type == EventType.MODEL_COMPLETED
                ]

                def completed_events() -> list[Event]:
                    return completion_events(completed_payloads()[before_count:])

                billing_identity_state = resolved_billing_identity(billing_identity)
                budget_evaluation = await self._limit_gate.evaluate_budget(
                    BudgetPolicy(limits=dispatch_policy_limits),
                    billing_identity_state=billing_identity_state,
                    pricing_provider_name=actual_pricing_provider_name,
                    model=actual_model,
                    additional_usage_events=prior_completion_events,
                    execution_identity=model_attempt_identity,
                )
                if budget_evaluation.check is not None:
                    raise _AutomaticCompactionAdmissionStopped(budget_evaluation=budget_evaluation)
                budget_events.extend(budget_evaluation.events)
                limit_evaluation = await self._limit_gate.evaluate_limits(
                    billing_identity_state=billing_identity_state,
                    pricing_provider_name=actual_pricing_provider_name,
                    model=actual_model,
                    additional_usage_events=prior_completion_events,
                    budget_limits=dispatch_request_limits,
                    execution_identity=model_attempt_identity,
                )
                if limit_evaluation.decision is not None:
                    raise _AutomaticCompactionAdmissionStopped(limit_evaluation=limit_evaluation)
                budget_events.extend(limit_evaluation.events)
                if not limits:
                    return await identified_dispatch()
                outcome = await controller.run_automatic_compaction_dispatch(
                    identified_dispatch,
                    completed_events=completed_events,
                    prior_completion_events=prior_completion_events,
                    budget_limits=limits,
                    session=self._session,
                    agent_name=self._registered_agent.spec.name,
                    environment_name=self._environment_name,
                    provider_name=actual_pricing_provider_name,
                    model=require_clean_nonblank(actual_model, "compactor_model"),
                    model_attempt_identity=model_attempt_identity,
                    billing_identity=billing_identity,
                    pricing_provider_name=pricing_provider_name,
                    authoritative_failure_types=(ContextBuildError,),
                    reservation_identity_guard=self._reservation_identity_guard,
                )
                budget_events.extend(outcome.events)
                if isinstance(outcome, BudgetedOperationSucceeded):
                    return cast("tuple[str, dict[str, Any]]", outcome.result)
                if isinstance(outcome, BudgetedOperationRejected):
                    raise _AutomaticCompactionBudgetReservationFailed(outcome.failure)
                if outcome.cause is not None:
                    raise outcome.error from outcome.cause
                raise outcome.error
            finally:
                compaction_identity_ledger.end_dispatch(model_attempt_identity)

        with _automatic_compaction_dispatch_runner_scope(run_provider_dispatch):
            return await execute()


def _session_agent_spec(
    *,
    registered_agent: runtime_records.RegisteredAgentState,
    session: Session,
) -> AgentSpec:
    return AgentSpec(
        name=registered_agent.spec.name,
        model=session.model,
        provider_name=session.provider_name,
        system_prompt=registered_agent.spec.system_prompt,
        metadata=copy_json_value(registered_agent.spec.metadata, "metadata"),
        provider_options=copy_json_value(
            registered_agent.spec.provider_options,
            "provider_options",
        ),
    )


def _context_pressure_overhead(
    *,
    registered_provider: runtime_records.RegisteredProvider,
    registered_agent: runtime_records.RegisteredAgentState,
    registered_environment: runtime_records.RegisteredEnvironment | None,
    structured_output: StructuredOutputSpec | None,
    thinking: ThinkingConfig | None,
    step: int,
) -> ContextPressureOverhead:
    profile = copy_model_context_pressure_profile(
        registered_provider.provider.context_pressure_profile
    )
    tools = [
        {
            "name": tool.name,
            "description": tool.description,
            "input_schema": deepcopy(tool.schema),
        }
        for tool in registered_agent.tools.values()
    ]
    structured_output_instruction: str | None = None
    if (
        structured_output is not None
        and structured_output.strategy == StructuredOutputStrategy.TOOL
    ):
        tools.append(structured_output_tool_spec(structured_output))
        structured_output_instruction = structured_output_tool_instruction(structured_output)

    request_options: dict[str, Any] = {
        **copy_json_value(
            registered_agent.spec.provider_options,
            "provider_options",
        ),
        "agent_metadata": deepcopy(registered_agent.spec.metadata),
        "environment_metadata": (
            deepcopy(registered_environment.spec.metadata)
            if registered_environment is not None
            else {}
        ),
        "step": step,
        "structured_output": (
            structured_output_spec_payload(structured_output)
            if structured_output is not None
            else None
        ),
    }
    if thinking is not None:
        request_options["thinking"] = thinking_config_payload(thinking)
    return ContextPressureOverhead(
        tools=tools,
        structured_output_instruction=structured_output_instruction,
        request_options=request_options,
        image_min_tokens=profile.image_min_tokens,
        document_min_tokens=profile.document_min_tokens,
        document_bytes_per_token=profile.document_bytes_per_token,
        tool_schema_chars_per_token=profile.tool_schema_chars_per_token,
    )


async def _build_context(
    *,
    context_policy: ContextPolicy,
    session_store: SessionStore,
    session: Session,
    agent_spec: AgentSpec,
    messages: list[Message],
    step: int,
    environment_name: str | None,
    knowledge_store: Any,
    request_metadata: dict[str, Any],
    pressure_overhead: ContextPressureOverhead,
    count_input_tokens: Callable[[list[Message]], Awaitable[int | None]] | None,
    build_cache_prefix_request: Callable[[list[Message]], Awaitable[ModelRequest]] | None,
    secret_redactor: SecretRedactor,
    run_compaction: _AutomaticCompactionRunner | None = None,
    force_bounded_compaction: bool = False,
) -> tuple[
    list[Message],
    dict[str, Any] | None,
    dict[str, Any] | None,
    list[ContextCompactionTelemetry],
    list[ContextKnowledgeTelemetry],
]:
    context_usage = await _context_usage_state_for_session(
        session_store=session_store,
        session_id=session.id,
    )
    context_usage = estimate_context_pressure(
        usage=context_usage,
        messages=messages,
        image_min_tokens=pressure_overhead.image_min_tokens,
        document_min_tokens=pressure_overhead.document_min_tokens,
        document_bytes_per_token=pressure_overhead.document_bytes_per_token,
    )
    request = ContextRequest(
        session=session.model_copy(deep=True),
        agent=agent_spec.model_copy(deep=True),
        messages=[message.model_copy(deep=True) for message in messages],
        step=step,
        environment_name=environment_name,
        knowledge_store=knowledge_store,
        metadata=copy_json_value(request_metadata, "metadata"),
        context_usage=context_usage,
        pressure_overhead=pressure_overhead,
        count_input_tokens=count_input_tokens,
        build_cache_prefix_request=build_cache_prefix_request,
        force_bounded_compaction=force_bounded_compaction,
    )
    if isinstance(context_policy, RuntimeManagedContextPolicy):
        checkpoint = await session_store.load_checkpoint(session.id)
        try:
            with (
                _context_secret_redactor_scope(secret_redactor),
                _defer_billing_identity_cancellation_scope(),
                _automatic_compaction_runner_scope(run_compaction),
            ):
                result = await context_policy.build_with_checkpoint(
                    request,
                    checkpoint=checkpoint,
                )
        except ContextBuildError as error:
            sanitize_context_build_error_checkpoint(
                error,
                redactor=secret_redactor,
            )
            raise
        safe_checkpoint, safe_checkpoint_event_payload = sanitize_context_build_result_checkpoint(
            result,
            redactor=secret_redactor,
        )
        return (
            copy_context_messages(result.messages),
            safe_checkpoint,
            safe_checkpoint_event_payload,
            [telemetry.model_copy(deep=True) for telemetry in result.compaction_telemetry],
            [telemetry.model_copy(deep=True) for telemetry in result.knowledge_telemetry],
        )

    with _context_secret_redactor_scope(secret_redactor):
        result = await context_policy.build(request)
    return copy_context_messages(result), None, None, [], []


async def _context_usage_state_for_session(
    *,
    session_store: SessionStore,
    session_id: str,
) -> ContextUsageState:
    before_sequence: int | None = None
    page_size = 1
    while True:
        records = await session_store.query_events(
            EventQuery(
                session_id=session_id,
                event_type=EventType.MODEL_COMPLETED,
                before_sequence=before_sequence,
                limit=page_size,
                order_by=EventOrder.SEQUENCE_DESC,
            )
        )
        if not records:
            return ContextUsageState()
        for record in records:
            if is_conversational_model_completion_payload(record.event.payload):
                return _context_usage_state_from_model_completed_event(record.event)
        before_sequence = records[-1].sequence
        page_size = _CONTEXT_USAGE_AUXILIARY_PAGE_SIZE


def _context_usage_state_from_model_completed_event(event: Event) -> ContextUsageState:
    if event.type != EventType.MODEL_COMPLETED:
        return ContextUsageState()
    metrics = usage_metrics_from_event_payload(event.payload)
    if metrics is None:
        return ContextUsageState(
            last_transcript_cursor=_transcript_cursor_from_model_completed_event(event)
        )
    return ContextUsageState(
        last_input_tokens=metrics.input_tokens,
        last_output_tokens=metrics.output_tokens,
        last_total_tokens=metrics.total_tokens,
        last_transcript_cursor=_transcript_cursor_from_model_completed_event(event),
        last_context_overhead_input_tokens=(
            _context_overhead_input_tokens_from_model_completed_event(event)
        ),
        last_provider_name=metrics.provider_name,
        last_requested_model=metrics.requested_model,
        last_model=metrics.model,
    )


def _transcript_cursor_from_model_completed_event(event: Event) -> int | None:
    cursor = event.payload.get("transcript_cursor")
    if type(cursor) is not int or cursor < 0:
        return None
    return cursor


def _context_overhead_input_tokens_from_model_completed_event(event: Event) -> int | None:
    pressure = event.payload.get("context_pressure")
    if type(pressure) is not dict:
        return None
    tokens = pressure.get("estimated_request_overhead_input_tokens")
    if type(tokens) is not int or tokens < 0:
        return None
    return tokens


def _has_provider_backed_context_compaction(
    compaction_telemetry: list[ContextCompactionTelemetry],
) -> bool:
    return any(
        telemetry.event_type == EventType.MODEL_COMPLETED for telemetry in compaction_telemetry
    )


def _context_compaction_telemetry_event(
    *,
    telemetry: ContextCompactionTelemetry,
    session: Session,
    registered_agent: runtime_records.RegisteredAgentState,
    environment_name: str | None,
    execution_identity: ModelStepIdentity | ModelAttemptIdentity | None = None,
) -> Event:
    if type(telemetry) is not ContextCompactionTelemetry:
        raise TypeError(
            "Context compaction telemetry must be ContextCompactionTelemetry instances."
        )
    sanitized = sanitize_context_compaction_telemetry(telemetry)
    payload = copy_json_value(sanitized.payload, "payload")
    strip_runtime_owned_execution_identity(payload)
    if type(execution_identity) is ModelAttemptIdentity:
        payload.update(copy_model_attempt_identity(execution_identity).payload())
    elif type(execution_identity) is ModelStepIdentity:
        payload.update(copy_model_step_identity(execution_identity).payload())
    elif execution_identity is not None:
        raise TypeError("Context compaction execution identity has an unsupported type.")
    event = Event(
        type=sanitized.event_type,
        session_id=session.id,
        agent_name=registered_agent.spec.name,
        environment_name=environment_name,
        payload=payload,
    )
    return (
        event
        if execution_identity is None
        else _event_with_model_identity_authority(event, execution_identity)
    )


def _context_knowledge_telemetry_event(
    *,
    telemetry: ContextKnowledgeTelemetry,
    session: Session,
    registered_agent: runtime_records.RegisteredAgentState,
    environment_name: str | None,
    model_step_identity: ModelStepIdentity,
) -> Event:
    if type(telemetry) is not ContextKnowledgeTelemetry:
        raise TypeError("Context knowledge telemetry must be ContextKnowledgeTelemetry instances.")
    payload = copy_json_value(telemetry.payload, "payload")
    strip_runtime_owned_execution_identity(payload)
    payload.update(copy_model_step_identity(model_step_identity).payload())
    return Event(
        type=telemetry.event_type,
        session_id=session.id,
        agent_name=registered_agent.spec.name,
        environment_name=environment_name,
        payload=payload,
    )


def _context_overflow_event_payload(
    error: ModelContextOverflowError,
    *,
    step: int,
    phase: str,
    original_message_count: int,
    recovery_message_count: int | None = None,
    model_step_identity: ModelStepIdentity,
    model_attempt_identity: ModelAttemptIdentity | None,
) -> dict[str, Any]:
    model_step_identity = copy_model_step_identity(model_step_identity)
    payload: dict[str, Any] = {
        "step": step,
        "phase": require_clean_nonblank(phase, "phase"),
        "error": str(error),
        "error_type": type(error).__name__,
        "provider": error.provider,
        "original_message_count": original_message_count,
        **model_step_identity.payload(),
    }
    if model_attempt_identity is not None:
        copied_attempt = copy_model_attempt_identity(model_attempt_identity)
        if copied_attempt.model_step_id != model_step_identity.model_step_id:
            raise ValueError("Context overflow attempt belongs to a different model step.")
        payload.update(copied_attempt.payload())
    if error.status_code is not None:
        payload["status_code"] = error.status_code
    if error.error_type is not None:
        payload["provider_error_type"] = error.error_type
    if error.error_code is not None:
        payload["provider_error_code"] = error.error_code
    if error.request_id is not None:
        payload["request_id"] = error.request_id
    if recovery_message_count is not None:
        payload["recovery_message_count"] = recovery_message_count
    return payload


def _model_context_overflow_error_event(
    error: ModelContextOverflowError,
    *,
    session: Session,
    registered_agent: runtime_records.RegisteredAgentState,
    environment_name: str | None,
    step: int,
    attempt: int,
    max_attempts: int,
    model_attempt_identity: ModelAttemptIdentity,
) -> Event:
    """Terminalize one dispatched context-overflow attempt without retrying it."""

    if type(error) is not ModelContextOverflowError:
        raise TypeError("Context-overflow terminalization requires a runtime-owned error.")
    payload = {
        "error": str(error),
        "error_type": type(error).__name__,
        "stage": "provider_dispatch",
        "context_overflow": True,
        **ModelProviderError.error_payload_fields(error),
    }
    return Event(
        type=EventType.MODEL_ERROR,
        session_id=session.id,
        agent_name=registered_agent.spec.name,
        environment_name=environment_name,
        payload=_retry_attempt_payload(
            payload,
            step=step,
            attempt=attempt,
            max_attempts=max_attempts,
            model_attempt_identity=model_attempt_identity,
        ),
    )


class _FileAttachmentUnavailable(RuntimeError):
    """An attachment reference cannot be resolved in its declared scope."""


async def _resolved_file_attachments(
    *,
    messages: list[Message],
    session: Session,
    registered_environment: runtime_records.RegisteredEnvironment | None,
    max_file_attachment_bytes: int,
    max_total_file_attachment_bytes: int,
    max_file_attachments_per_request: int,
) -> tuple[dict[str, dict[str, Any]], set[str]]:
    """Resolve model-facing files while failing open only for prompt files.

    Missing prompt files become visible text notes so a stale prompt reference
    cannot brick a session forever. Tool-result files remain fail-closed because
    silently omitting tool evidence would let the model answer from incomplete
    state. A reference used by both paths therefore remains fail-closed.
    """
    attachment_refs, prompt_file_artifact_ids, tool_result_artifact_ids = _file_attachment_refs(
        messages
    )
    if not attachment_refs:
        return {}, set()
    if len(attachment_refs) > max_file_attachments_per_request:
        raise RuntimeError(
            "File attachment count exceeds the runtime attachment limit: "
            f"{len(attachment_refs)} > {max_file_attachments_per_request}"
        )
    artifact_store = (
        None
        if registered_environment is None
        else registered_environment.environment.artifact_store
    )
    if artifact_store is None:
        raise RuntimeError("File attachments require an artifact store.")

    environment_name = None if registered_environment is None else registered_environment.spec.name
    resolved: dict[str, dict[str, Any]] = {}
    unresolvable_prompt_ids: set[str] = set()
    total_attachment_bytes = 0
    for attachment in attachment_refs:
        if attachment.size_bytes > max_file_attachment_bytes:
            raise RuntimeError(
                "File attachment exceeds the runtime attachment byte limit: "
                f"{attachment.artifact_id}"
            )
        total_attachment_bytes += attachment.size_bytes
        if total_attachment_bytes > max_total_file_attachment_bytes:
            raise RuntimeError("File attachments exceed the runtime total attachment byte limit.")
        if attachment.artifact_id in resolved or attachment.artifact_id in unresolvable_prompt_ids:
            continue
        try:
            result = copy_artifact_read_result(
                await artifact_store.read_bytes(
                    attachment.artifact_id,
                    max_bytes=attachment.size_bytes,
                ),
                expected_artifact_id=attachment.artifact_id,
                max_content_bytes=attachment.size_bytes,
            )
            artifact = result.metadata
            if artifact.scope.value == "session" and artifact.session_id != session.id:
                raise _FileAttachmentUnavailable(
                    "File attachment is not available in this session."
                )
            if (
                artifact.scope.value == "environment"
                and artifact.environment_name != environment_name
            ):
                raise _FileAttachmentUnavailable(
                    "File attachment is not available in this environment."
                )
            if artifact.content_type != attachment.content_type:
                raise _FileAttachmentUnavailable(
                    "File attachment content type changed before provider request."
                )
            if artifact.size_bytes != attachment.size_bytes:
                raise _FileAttachmentUnavailable(
                    "File attachment size changed before provider request."
                )
        except (FileNotFoundError, InvalidArtifactIdError, _FileAttachmentUnavailable):
            is_exclusively_prompt = (
                attachment.artifact_id in prompt_file_artifact_ids
                and attachment.artifact_id not in tool_result_artifact_ids
            )
            if not is_exclusively_prompt:
                raise
            unresolvable_prompt_ids.add(attachment.artifact_id)
            continue
        resolved[attachment.artifact_id] = resolved_file_attachment(attachment, result)
    return resolved, unresolvable_prompt_ids


def _file_attachment_refs(
    messages: list[Message],
) -> tuple[tuple[FileAttachment, ...], set[str], set[str]]:
    """Collect ordered references and their prompt/tool-result provenance."""
    refs: dict[str, FileAttachment] = {}
    ordered_refs: list[FileAttachment] = []
    prompt_artifact_ids: set[str] = set()
    tool_result_artifact_ids: set[str] = set()
    for message in messages:
        for part in message.content:
            if type(part) is ToolResultPart:
                payloads: list[dict[str, Any]] = part.artifacts
                origin_ids = tool_result_artifact_ids
            elif type(part) is FilePart:
                payloads = [part.attachment]
                origin_ids = prompt_artifact_ids
            else:
                continue
            for payload in payloads:
                attachment = file_attachment_from_payload(payload)
                if attachment is None:
                    continue
                origin_ids.add(attachment.artifact_id)
                existing = refs.get(attachment.artifact_id)
                if existing is not None and not _same_file_attachment_ref(existing, attachment):
                    raise RuntimeError(
                        "Conflicting file attachment references for artifact: "
                        f"{attachment.artifact_id}"
                    )
                refs[attachment.artifact_id] = attachment
                ordered_refs.append(attachment)
    return tuple(ordered_refs), prompt_artifact_ids, tool_result_artifact_ids


def _same_file_attachment_ref(left: FileAttachment, right: FileAttachment) -> bool:
    return left.model_dump(mode="json") == right.model_dump(mode="json")


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
                normalize_usage_metrics(
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

    return _ModelStreamBoundaryValue(
        event=ModelStreamEvent.model_construct(
            type=ModelStreamEventType.COMPLETED,
            delta=delta,
            payload=payload,
            completion=completion,
        ),
        completion_error=completion_error,
        accounting_usage_metrics=accounting_usage_metrics,
        accounting_usage_rejected=accounting_usage_rejected,
        usage_normalization_failed=usage_normalization_failed,
    )


def _copy_model_request_for_counting(request: ModelRequest) -> ModelRequest:
    return _detach_model_request(request)


def _detach_model_request(request: ModelRequest) -> ModelRequest:
    if type(request) is not ModelRequest:
        raise TypeError("request must be a ModelRequest.")
    return ModelRequest(
        model=request.model,
        messages=request.messages,
        tools=request.tools,
        options=request.options,
    )


def _context_count_base_payload(
    *,
    model_request: ModelRequest,
    provider_name: str,
    step: int,
    attempt: int,
    max_attempts: int,
    observation_id: str,
    model_attempt_identity: ModelAttemptIdentity,
) -> dict[str, Any]:
    model_attempt_identity = copy_model_attempt_identity(model_attempt_identity)
    roles = [
        message.role.value if isinstance(message.role, MessageRole) else str(message.role)
        for message in model_request.messages
    ]
    return {
        "model": model_request.model,
        "provider": provider_name,
        "step": step,
        "attempt": attempt,
        "max_attempts": max_attempts,
        "observation_id": observation_id,
        "messages": {"count": len(model_request.messages), "roles": roles},
        "tools": {"count": len(model_request.tools)},
        "options": {"keys": sorted(model_request.options.keys())},
        **model_attempt_identity.payload(),
    }


def _context_count_reconciled_event(
    model_completed_event: Event,
    *,
    observation: _ContextCountObservation,
    session: Session,
    registered_agent: runtime_records.RegisteredAgentState,
    registered_provider: runtime_records.RegisteredProvider,
    environment_name: str | None,
    step: int,
    attempt: int,
    max_attempts: int,
    model_attempt_identity: ModelAttemptIdentity,
) -> Event:
    if model_completed_event.type != EventType.MODEL_COMPLETED:
        raise ValueError("Context count reconciliation requires a model.completed event.")
    actual_input_tokens = _actual_input_tokens_from_completed_event(model_completed_event)
    estimated_input_tokens = observation.result.input_tokens
    delta_tokens = (
        None
        if actual_input_tokens is None or estimated_input_tokens is None
        else actual_input_tokens - estimated_input_tokens
    )
    relative_error = (
        None
        if delta_tokens is None or actual_input_tokens is None or actual_input_tokens <= 0
        else delta_tokens / actual_input_tokens
    )
    return _context_observation_event(
        Event(
            type=EventType.CONTEXT_COUNT_RECONCILED,
            session_id=session.id,
            agent_name=registered_agent.spec.name,
            environment_name=environment_name,
            payload={
                "model": session.model,
                "provider": registered_provider.name,
                "step": step,
                "attempt": attempt,
                "max_attempts": max_attempts,
                "observation_id": observation.observation_id,
                "pre_call_count": observation.result.model_dump(mode="json"),
                "actual_input_tokens": actual_input_tokens,
                "delta_tokens": delta_tokens,
                "relative_error": relative_error,
                "reconciled": actual_input_tokens is not None
                and estimated_input_tokens is not None,
                **copy_model_attempt_identity(model_attempt_identity).payload(),
            },
        )
    )


def _context_pressure_reconciled_event(
    model_completed_event: Event,
    *,
    observation: _ContextPressureObservation,
    session: Session,
    registered_agent: runtime_records.RegisteredAgentState,
    registered_provider: runtime_records.RegisteredProvider,
    environment_name: str | None,
    step: int,
    attempt: int,
    max_attempts: int,
    model_attempt_identity: ModelAttemptIdentity,
) -> Event:
    if model_completed_event.type != EventType.MODEL_COMPLETED:
        raise ValueError("Context pressure reconciliation requires a model.completed event.")
    actual_input_tokens = _actual_input_tokens_from_completed_event(model_completed_event)
    estimated_input_tokens = observation.estimate.estimated_context_input_tokens
    delta_tokens = (
        None if actual_input_tokens is None else actual_input_tokens - estimated_input_tokens
    )
    relative_error = (
        None
        if delta_tokens is None or actual_input_tokens is None or actual_input_tokens <= 0
        else delta_tokens / actual_input_tokens
    )
    return _context_observation_event(
        Event(
            type=EventType.CONTEXT_PRESSURE_RECONCILED,
            session_id=session.id,
            agent_name=registered_agent.spec.name,
            environment_name=environment_name,
            payload={
                "model": session.model,
                "provider": registered_provider.name,
                "step": step,
                "attempt": attempt,
                "max_attempts": max_attempts,
                "observation_id": observation.observation_id,
                "pre_call_estimate": observation.estimate.model_dump(mode="json"),
                "actual_input_tokens": actual_input_tokens,
                "delta_tokens": delta_tokens,
                "relative_error": relative_error,
                "reconciled": actual_input_tokens is not None,
                **copy_model_attempt_identity(model_attempt_identity).payload(),
            },
        )
    )


def _actual_input_tokens_from_completed_event(event: Event) -> int | None:
    usage_metrics = event.payload.get("usage_metrics")
    if type(usage_metrics) is not dict:
        return None
    input_tokens = usage_metrics.get("input_tokens")
    if type(input_tokens) is not int or input_tokens < 0:
        return None
    return input_tokens


def _model_stream_event_to_runtime_event(
    stream_event: ModelStreamEvent,
    *,
    session: Session,
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
    usage_dialect: str | None = None,
    billing_identity: BillingIdentity | None = None,
    accounting_usage_metrics: dict[str, Any] | None = None,
    accounting_usage_rejected: bool = False,
    usage_normalization_failed: bool = False,
    completion_diagnostics: dict[str, Any] | None = None,
) -> Event:
    if type(stream_event) is not ModelStreamEvent:
        raise TypeError("Model stream events must be ModelStreamEvent instances.")
    if stream_event.type == ModelStreamEventType.TEXT_DELTA:
        event_type = EventType.MODEL_TEXT_DELTA
        payload = {"delta": stream_event.delta}
    elif stream_event.type == ModelStreamEventType.THINKING:
        event_type = EventType.MODEL_THINKING_DELTA
        payload = {"delta": stream_event.delta}
    elif stream_event.type == ModelStreamEventType.COMPLETED:
        payload = transcript_helpers.model_completed_event_payload(stream_event.payload)
        # When raw usage is present, its normalized projection and failure
        # marker are runtime-owned accounting evidence. Providers that expose
        # only the established normalized-usage payload retain compatibility.
        has_raw_usage = payload.get("usage") is not None
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
        resolved_model = _payload_model(payload, fallback=session.model)
        payload["model"] = resolved_model
        payload["requested_model"] = session.model
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
                    requested_model=session.model,
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
        step=step,
        attempt=attempt,
        max_attempts=max_attempts,
        model_attempt_identity=model_attempt_identity,
    )
    if tool_round_identity is not None:
        payload.update(copy_tool_round_identity(tool_round_identity).payload())
    if event_type == EventType.MODEL_COMPLETED:
        payload = durable_model_completed_payload(
            payload,
            fallback_fields={
                "provider_name": provider_name,
                "requested_model": session.model,
                "model": session.model,
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
    return event


def _with_structured_output_tool_instruction(
    messages: list[Message],
    spec: StructuredOutputSpec,
) -> list[Message]:
    copied_messages = copy_context_messages(messages)
    instruction = Message.text(MessageRole.SYSTEM, structured_output_tool_instruction(spec))
    insert_at = 0
    while (
        insert_at < len(copied_messages) and copied_messages[insert_at].role == MessageRole.SYSTEM
    ):
        insert_at += 1
    copied_messages.insert(insert_at, instruction)
    return copied_messages


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


def _typed_retry_fields(
    exc: ModelAttemptFailed,
) -> tuple[int | None, bool | None, float | None]:
    cause = exc.cause
    if exc.completion_observed:
        # A valid completed frame is the authoritative terminal attempt. A
        # later transport/control failure cannot authorize another provider
        # charge for the same logical step.
        status_code = cause.status_code if isinstance(cause, ModelProviderError) else None
        return status_code, False, None
    if isinstance(cause, ModelProviderError):
        return cause.status_code, cause.retryable, cause.retry_after_s
    status_code = exc.payload.get("status_code")
    retryable = exc.payload.get("retryable")
    retry_after_s = exc.payload.get("retry_after_s")
    return (
        status_code if type(status_code) is int else None,
        retryable if type(retryable) is bool else None,
        float(retry_after_s) if type(retry_after_s) in {int, float} else None,
    )


def _model_retry_event(
    *,
    session: Session,
    registered_agent: runtime_records.RegisteredAgentState,
    environment_name: str | None,
    registered_provider: runtime_records.RegisteredProvider,
    step: int,
    decision: RetryDecision,
    error: str,
    model_attempt_identity: ModelAttemptIdentity,
) -> Event:
    model_attempt_identity = copy_model_attempt_identity(model_attempt_identity)
    payload = retry_event_payload(
        decision=decision,
        provider_name=registered_provider.name,
        model=session.model,
        step=step,
        error=error,
    )
    payload.update(model_attempt_identity.payload())
    return _event_with_model_identity_authority(
        Event(
            type=EventType.MODEL_RETRY,
            session_id=session.id,
            agent_name=registered_agent.spec.name,
            environment_name=environment_name,
            payload=payload,
        ),
        model_attempt_identity,
    )


def _model_attempt_discarded_event(
    *,
    session: Session,
    registered_agent: runtime_records.RegisteredAgentState,
    environment_name: str | None,
    registered_provider: runtime_records.RegisteredProvider,
    step: int,
    decision: RetryDecision,
    model_attempt_identity: ModelAttemptIdentity,
) -> Event:
    return _event_with_model_identity_authority(
        Event(
            type=EventType.MODEL_ATTEMPT_DISCARDED,
            session_id=session.id,
            agent_name=registered_agent.spec.name,
            environment_name=environment_name,
            payload={
                "provider": registered_provider.name,
                "model": session.model,
                "step": step,
                "attempt": decision.attempt,
                "next_attempt": decision.next_attempt,
                "max_attempts": decision.max_attempts,
                "reason": None if decision.reason is None else decision.reason.value,
                "status_code": decision.status_code,
                **copy_model_attempt_identity(model_attempt_identity).payload(),
            },
        ),
        model_attempt_identity,
    )


def _retry_attempt_payload(
    payload: dict[str, Any],
    *,
    step: int,
    attempt: int,
    max_attempts: int,
    model_attempt_identity: ModelAttemptIdentity,
) -> dict[str, Any]:
    enriched = dict(payload)
    strip_runtime_owned_execution_identity(enriched)
    enriched["step"] = step
    enriched["attempt"] = attempt
    enriched["max_attempts"] = max_attempts
    enriched.update(copy_model_attempt_identity(model_attempt_identity).payload())
    return enriched


def _payload_model(payload: dict[str, Any], *, fallback: str) -> str:
    model = payload.get("model")
    if type(model) is str and model.strip():
        return model
    return fallback


async def _close_async_iterator(iterator: AsyncIterator[Any]) -> None:
    # Iterator disposal runs while a more authoritative provider, budget,
    # cancellation, or GeneratorExit outcome is already propagating. Attribute
    # lookup is provider-controlled too, so it belongs inside the same boundary
    # as invoking the close hook.
    try:
        close = getattr(iterator, "aclose", None)
        if close is not None:
            await close()
    except ExceptionGroup:
        # Ordinary cleanup failures remain secondary to the outcome that
        # caused iterator disposal.
        pass
    except BaseExceptionGroup:
        # A mixed group also carries a fatal signal such as cancellation.
        # Preserve the complete group for the caller to classify.
        raise
    except Exception:
        pass

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
import sys
from collections.abc import AsyncIterator, Awaitable, Callable
from contextvars import Context
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from typing import Any, Literal, Never, cast
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator, model_validator

from cayu._exception_groups import (
    add_exception_note_safely,
    exception_cause,
    exception_context,
    exception_group_children,
    exception_tree_contains,
    iter_exception_tree,
    set_exception_cause,
    set_exception_context,
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
    copy_billing_identity,
    resolved_billing_identity,
)
from cayu.core.events import (
    Event,
    EventType,
    copy_event,
    event_with_runtime_generated_id,
    event_with_runtime_payload_authority,
)
from cayu.core.messages import (
    CitationPart,
    CitationProvenance,
    FilePart,
    HostedToolCallPart,
    Message,
    MessageRole,
    ProviderStatePart,
    ThinkingPart,
    ToolCallPart,
    ToolResultPart,
    WebSearchAction,
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
    ProviderOperationAdapter,
    ProviderOperationCancellationSupport,
    ProviderOperationConnection,
    ProviderOperationMalformedError,
    ProviderOperationMode,
    ProviderOperationRecoveryMetadata,
    ProviderOperationSnapshot,
    ProviderOperationStartIdempotencySupport,
    ProviderOperationStartRecoveryRequest,
    ProviderOperationStartRequest,
    ProviderOperationState,
    ProviderOperationStatus,
    UsageDialect,
    copy_input_token_count_result,
    copy_model_context_pressure_profile,
    copy_model_stream_event,
    copy_provider_operation_connection,
    copy_provider_operation_snapshot,
    copy_provider_operation_state,
    normalize_model_completion,
)
from cayu.providers._credential_boundary import aclosing_provider_stream
from cayu.providers.base import copy_model_completion
from cayu.runtime import _runtime_records as runtime_records
from cayu.runtime import _transcript as transcript_helpers
from cayu.runtime._checkpoint_redaction import durable_value_contains_secret
from cayu.runtime._completion_projection import portable_model_completion_projection
from cayu.runtime._diagnostics import exception_diagnostic
from cayu.runtime._event_writer import RuntimeEventWriter
from cayu.runtime._message_redaction import (
    redact_runtime_message_for_boundary,
)
from cayu.runtime._model_errors import (
    copy_provider_exception_control,
    model_provider_error_from_payload,
    nonportable_model_provider_error,
    resolve_completion_billing_identity,
    resolve_request_billing_identity,
)
from cayu.runtime._provider_operation_cancellation_claim import (
    ProviderOperationCancellationClaim,
    checkpoint_with_provider_operation_cancellation_claim,
    checkpoint_without_provider_operation_cancellation_claim,
    provider_operation_cancellation_claim_from_checkpoint,
)
from cayu.runtime._run_limit_accounting import (
    RunLimitAccountingContext,
    has_run_limit_accounting_authority,
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
    BudgetStepReservation,
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
    BudgetReservationRecoveryContext,
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
    ContextPolicy,
    ContextPressureEstimate,
    ContextPressureOverhead,
    ContextRecallTelemetry,
    ContextRequest,
    ContextUsageState,
    RuntimeManagedContextPolicy,
    _automatic_compaction_dispatch_runner_scope,
    _automatic_compaction_runner_scope,
    _AutomaticCompactionRunner,
    _compaction_completion_publisher_scope,
    _compaction_model_attempt_identity_scope,
    _context_recall_telemetry_publisher_scope,
    _context_secret_redactor_scope,
    _ContextCountAuthorityError,
    _defer_billing_identity_cancellation_scope,
    context_build_termination_compaction_telemetry,
    copy_context_messages,
    copy_context_pressure_estimate,
    estimate_context_pressure,
    noteify_unresolvable_prompt_files,
    sanitize_context_build_error_checkpoint,
    sanitize_context_build_result_checkpoint,
    sanitize_context_compaction_telemetry,
)
from cayu.runtime.context_counting import ContextCountingConfig, ContextCountingMode
from cayu.runtime.execution_profiles import (
    ExecutionProfileIdentity,
    event_with_execution_profile_authority,
    event_with_execution_profile_fingerprint_authority,
)
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
from cayu.runtime.provider_operations import (
    ProviderOperationEvidenceError,
    ProviderOperationProgressCommit,
    ProviderOperationProgressEnvelope,
    ProviderOperationRecoveryResult,
    ProviderOperationRecoveryStatus,
    ProviderOperationUnavailableReason,
    RecoverableProviderOperation,
    RecoverableProviderOperationStart,
    commit_provider_operation_progress,
    fallback_dispatch_ordinal_from_checkpoint,
    load_recoverable_provider_operation,
    provider_operation_progress_envelope,
    provider_operation_progress_event_id,
    provider_operation_started_event_id,
    provider_operation_unavailable_reason,
)
from cayu.runtime.request_footprints import (
    PromptContributionManifest,
    RequestFootprint,
    RequestFootprintConfig,
    RequestVariant,
    analyze_request_context_pressure,
    analyze_request_footprint,
    copy_request_footprint_config,
)
from cayu.runtime.retry_policy import (
    RetryDecision,
    RetryPolicy,
    copy_retry_policy,
    retry_decision,
    retry_event_payload,
)
from cayu.runtime.sessions import (
    MODEL_COMPLETION_RECOVERY_CONTEXT_MAX_BYTES,
    CheckpointTransform,
    EventOrder,
    EventQuery,
    ModelCompletionStage,
    ModelCompletionStageAbandonmentResult,
    ModelCompletionStageRequest,
    ModelCompletionStageResult,
    RuntimePublicationResult,
    Session,
    SessionRunFenced,
    SessionStatus,
    SessionStatusConflict,
    SessionStore,
)
from cayu.runtime.stop_policy import RunLimits
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
from cayu.runtime.tool_exposure import (
    ALL_REGISTERED_TOOLS_PROFILE_ID,
    TOOL_EXPOSURE_PROFILE_ID_MAX_CHARS,
    AllRegisteredToolsExposurePolicy,
    ResolvedToolExposure,
    ResolvedToolExposureAuthority,
    ToolExposure,
    ToolExposurePolicyRequest,
    copy_resolved_tool_exposure_authority,
    resolve_tool_exposure,
    resolved_tool_exposure_authority,
    tool_capability_ceiling_from_session_metadata,
    tool_exposure_record,
)
from cayu.runtime.usage import (
    ModelCompletionPurpose,
    durable_model_completed_payload,
    hosted_tool_usage_metrics_from_payload,
    is_conversational_model_completion_payload,
    normalize_usage_metrics,
    normalize_usage_metrics_with_overflow_error,
    usage_metrics_from_event_payload,
    usage_metrics_payload,
)
from cayu.vaults import SecretRedactor

logger = logging.getLogger(__name__)
_PROVIDER_OPERATION_START_CLEANUP_TIMEOUT_SECONDS = 5.0
_PROVIDER_OPERATION_CANCELLATION_CLAIM_LEASE = timedelta(seconds=30)
_PROVIDER_OPERATION_CANCELLATION_CLAIM_HEARTBEAT_SECONDS = 5.0
_PROVIDER_OPERATION_START_SETTLEMENT_TIMEOUT_SECONDS = 5.0
MAX_MODEL_COMPLETION_RECOVERY_CONTEXT_BYTES = MODEL_COMPLETION_RECOVERY_CONTEXT_MAX_BYTES
MAX_MODEL_COMPLETION_RECOVERY_METADATA_ENTRIES = 256
MAX_MODEL_COMPLETION_RECOVERY_BUDGET_LIMITS = 32
_MAX_MODEL_COMPLETION_RECOVERY_PRICE_ENTRIES = 512
_MAX_MODEL_COMPLETION_RECOVERY_PRICING_CONTEXTS = 128
_MAX_MODEL_COMPLETION_RECOVERY_EVIDENCE_ENTRIES = 256


def _provider_operation_progress_contains_secret(
    envelope: ProviderOperationProgressEnvelope,
    *,
    redactor: SecretRedactor,
) -> bool:
    """Check adapter-owned continuation and output without scanning schema keys."""

    stream_event = envelope.stream_event
    return (
        redactor.redact_text(stream_event.delta) != stream_event.delta
        or durable_value_contains_secret(
            stream_event.payload,
            redactor=redactor,
            path=("provider_operation_stream_payload",),
        )
        or durable_value_contains_secret(
            envelope.recovery_metadata.opaque,
            redactor=redactor,
            path=("provider_operation_recovery_opaque",),
        )
    )


class ModelCompletionRecoveryContext(BaseModel):
    """Secret-free run semantics required to publish an offline completion."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    schema_version: Literal[1] = 1
    execution_profile_fingerprint: str | None = Field(
        default=None,
        min_length=64,
        max_length=64,
        pattern=r"^[0-9a-f]{64}$",
    )
    tool_exposure: ResolvedToolExposureAuthority | None = None
    task_id: str | None = None
    request_metadata: dict[str, Any] = Field(default_factory=dict)
    structured_output: StructuredOutputSpec | None = None
    thinking: ThinkingConfig | None = None
    max_steps: StrictInt = Field(default=16, ge=1, le=256)
    limits: RunLimits = Field(default_factory=RunLimits)
    run_limit_accounting: RunLimitAccountingContext | None = None
    budget_limits: tuple[BudgetLimit, ...] = ()
    budget_reservations: tuple[BudgetReservationRecoveryContext, ...] = ()
    retry_policy: RetryPolicy = Field(default_factory=RetryPolicy)
    structured_output_attempt: StrictInt | None = Field(default=None, ge=1)
    billing_identity: BillingIdentity | None = None

    @field_validator("task_id")
    @classmethod
    def validate_task_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return require_durable_clean_nonblank(value, "task_id")

    @field_validator("request_metadata", mode="before")
    @classmethod
    def copy_request_metadata(cls, value: object) -> dict[str, Any]:
        copied = copy_durable_json_object(value, "request_metadata")
        if len(copied) > MAX_MODEL_COMPLETION_RECOVERY_METADATA_ENTRIES:
            raise ValueError(
                "request_metadata cannot contain more than "
                f"{MAX_MODEL_COMPLETION_RECOVERY_METADATA_ENTRIES} entries."
            )
        return copied

    @field_validator("budget_limits", mode="before")
    @classmethod
    def copy_budget_limits(cls, value: Any) -> tuple[BudgetLimit, ...]:
        if (
            type(value) in (list, tuple)
            and len(value) > MAX_MODEL_COMPLETION_RECOVERY_BUDGET_LIMITS
        ):
            raise ValueError(
                "budget_limits cannot contain more than "
                f"{MAX_MODEL_COMPLETION_RECOVERY_BUDGET_LIMITS} limits."
            )
        return copy_request_budget_limits(value)

    @field_validator("budget_reservations", mode="before")
    @classmethod
    def copy_budget_reservations(
        cls,
        value: Any,
    ) -> tuple[BudgetReservationRecoveryContext, ...]:
        if type(value) not in (list, tuple):
            raise TypeError("budget_reservations must be a list or tuple.")
        if len(value) > MAX_MODEL_COMPLETION_RECOVERY_BUDGET_LIMITS:
            raise ValueError(
                "budget_reservations cannot contain more than "
                f"{MAX_MODEL_COMPLETION_RECOVERY_BUDGET_LIMITS} entries."
            )
        copied = tuple(BudgetReservationRecoveryContext.model_validate(item) for item in value)
        reservation_ids = [item.reservation_id for item in copied]
        budget_limit_ids = [item.budget_limit_id for item in copied]
        if len(set(reservation_ids)) != len(reservation_ids):
            raise ValueError("budget_reservations must not repeat reservation ids.")
        if len(set(budget_limit_ids)) != len(budget_limit_ids):
            raise ValueError("budget_reservations must not repeat budget limit ids.")
        return copied

    @field_validator("billing_identity", mode="after")
    @classmethod
    def copy_context_billing_identity(
        cls,
        value: BillingIdentity | None,
    ) -> BillingIdentity | None:
        return copy_billing_identity(value)

    @model_validator(mode="after")
    def validate_durable_bounds(self) -> ModelCompletionRecoveryContext:
        if self.run_limit_accounting is not None and not has_run_limit_accounting_authority(
            self.limits,
            self.budget_limits,
        ):
            raise ValueError("run_limit_accounting requires active run-scoped authority.")
        for limit in (
            *self.budget_limits,
            *(reservation.limit for reservation in self.budget_reservations),
        ):
            price_book = limit.pricing
            if (
                len(price_book.prices) > _MAX_MODEL_COMPLETION_RECOVERY_PRICE_ENTRIES
                or len(price_book.resource_mappings) > _MAX_MODEL_COMPLETION_RECOVERY_PRICE_ENTRIES
                or len(price_book.contextual_pricing_requirements)
                > _MAX_MODEL_COMPLETION_RECOVERY_PRICE_ENTRIES
            ):
                raise ValueError("budget limit pricing collections exceed recovery bounds.")
        if self.billing_identity is not None:
            identity = self.billing_identity
            if (
                len(identity.request_evidence) > _MAX_MODEL_COMPLETION_RECOVERY_EVIDENCE_ENTRIES
                or len(identity.completion_evidence)
                > _MAX_MODEL_COMPLETION_RECOVERY_EVIDENCE_ENTRIES
                or len(identity.pricing_contexts) > _MAX_MODEL_COMPLETION_RECOVERY_PRICING_CONTEXTS
                or any(
                    len(context.dimensions) > _MAX_MODEL_COMPLETION_RECOVERY_EVIDENCE_ENTRIES
                    for context in identity.pricing_contexts
                )
            ):
                raise ValueError("billing identity collections exceed recovery bounds.")
        encoded = canonical_durable_json_bytes(
            self.model_dump(mode="json"),
            "model_completion_recovery_context",
        )
        if len(encoded) > MAX_MODEL_COMPLETION_RECOVERY_CONTEXT_BYTES:
            raise ValueError(
                "model completion recovery context exceeds the durable byte limit of "
                f"{MAX_MODEL_COMPLETION_RECOVERY_CONTEXT_BYTES}."
            )
        return self


ModelCompletionRecoveryContextFactory = Callable[
    [BillingIdentity | None, tuple[BudgetStepReservation, ...]],
    ModelCompletionRecoveryContext | None,
]


def model_completion_recovery_context_from_stage(
    stage: ModelCompletionStage,
) -> ModelCompletionRecoveryContext | None:
    """Reconstruct the typed, secret-free continuation context from one stage."""

    raw_context = stage.intent.get("recovery_context")
    if raw_context is None:
        return None
    context = ModelCompletionRecoveryContext.model_validate(
        copy_durable_json_object(raw_context, "recovery_context")
    )
    if (
        context.run_limit_accounting is not None
        and context.run_limit_accounting.baseline.session_id != stage.session_id
    ):
        raise ValueError("Model completion run-limit accounting belongs to another session.")
    return context


def _ambiguous_provider_operation_start_error(
    *,
    provider_name: str,
    cause: BaseException,
) -> ModelProviderError:
    return ModelProviderError(
        "Provider operation start outcome is ambiguous; automatic retry is disabled.",
        provider=provider_name,
        error_type=type(cause).__name__,
        error_code="provider_operation_start_ambiguous",
        retryable=False,
    )


def is_ambiguous_provider_operation_start_error(failure: BaseException) -> bool:
    """Return whether one failure retains start-only provider ambiguity."""

    return any(
        isinstance(candidate, ModelProviderError)
        and candidate.error_code == "provider_operation_start_ambiguous"
        for candidate in iter_exception_tree(failure)
    )


def _attach_provider_operation_cleanup_failure(
    failure: BaseException,
    cleanup_error: BaseException,
) -> None:
    prior_cause = exception_cause(failure)
    combined = BaseExceptionGroup(
        "Provider operation publication and cancellation both failed.",
        [prior_cause, cleanup_error] if prior_cause is not None else [cleanup_error],
    )
    if not set_exception_cause(failure, combined):
        add_exception_note_safely(
            failure,
            "Provider operation cancellation also failed with "
            f"{type(cleanup_error).__name__}; its exception could not be attached.",
        )


async def _cancel_provider_operation_after_definite_absence(
    *,
    adapter: ProviderOperationAdapter,
    state: ProviderOperationState,
    failure: BaseException,
    cancellation: asyncio.CancelledError | None = None,
    ownership_lost: asyncio.Event | None = None,
) -> tuple[asyncio.CancelledError | None, ProviderOperationSnapshot | None]:
    async def cancel():
        return await adapter.cancel(copy_provider_operation_state(state))

    async def cancel_while_owned() -> ProviderOperationSnapshot:
        if ownership_lost is None:  # pragma: no cover - selected before task creation
            raise RuntimeError("Cancellation ownership supervision requires an ownership event.")
        provider_task = asyncio.create_task(cancel())
        ownership_task = asyncio.create_task(ownership_lost.wait())
        try:
            done, _pending = await asyncio.wait(
                {provider_task, ownership_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if provider_task in done:
                return provider_task.result()
            provider_task.cancel("Provider-operation cancellation ownership was lost.")
            return await provider_task
        finally:
            if not ownership_task.done():
                ownership_task.cancel()
            if not provider_task.done():
                provider_task.cancel()
            await asyncio.gather(
                provider_task,
                ownership_task,
                return_exceptions=True,
            )

    cleanup_task = asyncio.create_task(cancel() if ownership_lost is None else cancel_while_owned())
    outcome = await await_shielded_task_outcome(
        cleanup_task,
        cancellation=cancellation,
        timeout_s=_PROVIDER_OPERATION_START_CLEANUP_TIMEOUT_SECONDS,
    )
    if outcome.timed_out:
        cleanup_task.cancel()
        drain_outcome = await await_shielded_task_outcome(
            cleanup_task,
            cancellation=outcome.cancellation,
            timeout_s=_PROVIDER_OPERATION_START_CLEANUP_TIMEOUT_SECONDS,
        )
        if drain_outcome.timed_out:
            cleanup_task.add_done_callback(_consume_detached_task_outcome)
            failure.add_note(
                "Provider operation cancellation remained in flight after local task "
                "cancellation; ownership was released with uncertain cleanup evidence."
            )
            return drain_outcome.cancellation, None
        cleanup_error = drain_outcome.error
        if isinstance(cleanup_error, asyncio.CancelledError):
            cleanup_error = None
        failure.add_note(
            "Provider operation cancellation exceeded its bounded timeout; "
            "the local cancellation task was drained before ownership was released."
        )
        if cleanup_error is not None:
            _attach_provider_operation_cleanup_failure(failure, cleanup_error)
        return drain_outcome.cancellation, None
    cleanup_error = outcome.error
    if isinstance(cleanup_error, asyncio.CancelledError) and outcome.cancellation is None:
        cleanup_error = unexpected_child_cancellation_error(
            cleanup_error,
            operation="Provider operation cancellation",
        )
    if cleanup_error is not None:
        _attach_provider_operation_cleanup_failure(failure, cleanup_error)
        failure.add_note(
            "Provider operation cleanup after start-evidence failure also failed: "
            f"{type(cleanup_error).__name__}."
        )
        return outcome.cancellation, None
    try:
        if outcome.result is None:
            raise RuntimeError("Provider operation cancellation returned no snapshot.")
        cancellation_snapshot = copy_provider_operation_snapshot(outcome.result)
        if cancellation_snapshot.state != state:
            raise RuntimeError("Provider operation cancellation returned a different identity.")
        if not cancellation_snapshot.status.terminal:
            failure.add_note("Provider operation cancellation did not reach a terminal state.")
    except BaseException as cleanup_error:
        _attach_provider_operation_cleanup_failure(failure, cleanup_error)
        failure.add_note(
            "Provider operation cleanup after start-evidence failure also failed: "
            f"{type(cleanup_error).__name__}."
        )
        return outcome.cancellation, None
    return outcome.cancellation, cancellation_snapshot


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
        automatic_retry_disabled: bool = False,
        retry_decision: RetryDecision | None = None,
    ) -> None:
        self.message = require_nonblank(message, "message")
        self.payload = copy_json_value(payload, "payload")
        self.emitted_error_event = emitted_error_event
        self.cause = cause
        self.completion_observed = completion_observed
        self.automatic_retry_disabled = automatic_retry_disabled
        if retry_decision is not None and type(retry_decision) is not RetryDecision:
            raise TypeError("retry_decision must be a RetryDecision or None.")
        self.retry_decision = retry_decision
        super().__init__(self.message)


def _raise_terminal_model_attempt_failure(exc: ModelAttemptFailed) -> Never:
    if exc.cause is None:
        raise RuntimeError(exc.message) from exc
    authoritative_cause = exception_cause(exc.cause)
    if exc.automatic_retry_disabled and authoritative_cause is not None:
        raise exc.cause from authoritative_cause
    raise exc.cause from exc


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
    # The model never owned tool-round identifiers. Evaluate only its content
    # against workload secrets, then restore the exact runtime lineage.
    assistant_message = transcript_helpers.redact_untrusted_assistant_message_for_boundary(
        copied.assistant_message,
        tool_round_identity=copied.tool_round_identity,
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
    defer_assistant_message: bool
    structured_output_validation: StructuredOutputValidation | None
    tool_exposure: ResolvedToolExposureAuthority | None = None

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
        if type(self.defer_assistant_message) is not bool:
            raise TypeError("defer_assistant_message must be a bool.")
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
        tool_exposure = (
            None
            if self.tool_exposure is None
            else copy_resolved_tool_exposure_authority(self.tool_exposure)
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
        if self.defer_assistant_message and (
            result is None or assistant_message is None or not result.tool_calls
        ):
            raise ValueError(
                "Deferred assistant publication requires an ordinary tool-call message."
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
        object.__setattr__(self, "defer_assistant_message", self.defer_assistant_message)
        object.__setattr__(
            self,
            "structured_output_validation",
            structured_output_validation,
        )
        object.__setattr__(self, "tool_exposure", tool_exposure)


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
    recovery_context: ModelCompletionRecoveryContext | None,
    provider_operation_start: dict[str, Any] | None = None,
) -> dict[str, Any]:
    model_attempt_identity = copy_model_attempt_identity(model_attempt_identity)
    if (
        recovery_context is not None
        and type(recovery_context) is not ModelCompletionRecoveryContext
    ):
        raise TypeError(
            "Model completion recovery context must be a ModelCompletionRecoveryContext."
        )
    intent: dict[str, Any] = {
        "schema_version": 1,
        "purpose": "assistant-turn",
        **model_attempt_identity.payload(),
        "logical_step_id": model_attempt_identity.model_step_id,
        "provider_name": provider_name,
        "requested_model": requested_model,
        "source_transcript_cursor": source_transcript_cursor,
        "request_fingerprint": request_fingerprint,
    }
    if recovery_context is not None:
        intent["recovery_context"] = recovery_context.model_dump(mode="json")
    if provider_operation_start is not None:
        intent["provider_operation_start"] = copy_durable_json_object(
            provider_operation_start,
            "provider_operation_start",
        )
    return intent


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
        if request.authoritative_assistant_message is None or request.defer_assistant_message
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
    expects_pending_round = bool(
        request.assistant_step_result is not None
        and request.authoritative_assistant_message is not None
        and request.assistant_step_result.tool_calls
    )
    if bool(pending_round_operations) != expects_pending_round:
        raise RuntimeError("Model completion publication changed its pending tool-round mutation.")
    durable_validation = None
    durable_tool_exposure = None
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
        durable_tool_exposure = pending_round_value.get("tool_exposure")
    expected_validation = (
        None
        if request.structured_output_validation is None
        else request.structured_output_validation.model_dump(mode="json")
    )
    if durable_validation != expected_validation:
        raise RuntimeError("Model completion publication changed its structured-output validation.")
    expected_tool_exposure = (
        None
        if not expects_pending_round or request.tool_exposure is None
        else request.tool_exposure.model_dump(mode="json")
    )
    if durable_tool_exposure != expected_tool_exposure:
        raise RuntimeError("Model completion publication changed its frozen tool exposure.")

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
        defer_assistant_message=request.defer_assistant_message,
        structured_output_validation=request.structured_output_validation,
        tool_exposure=request.tool_exposure,
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


def _classify_provider_recovery_failure(
    failure: BaseException,
    *,
    cancellation_baseline: int,
    operation: str,
) -> BaseException:
    """Keep current caller cancellation distinct from child-only cancellation."""

    fatal_leaves = [
        candidate
        for candidate in iter_exception_tree(failure)
        if not isinstance(candidate, BaseExceptionGroup)
        and not isinstance(candidate, (Exception, asyncio.CancelledError))
    ]
    if fatal_leaves:
        return failure
    task = asyncio.current_task()
    if task is not None and task.cancelling() > cancellation_baseline:
        cancellation = _current_provider_recovery_cancellation(
            failure,
            cancellation_baseline=cancellation_baseline,
        )
        if cancellation is None:  # pragma: no cover - guarded by the task count
            raise AssertionError("Provider recovery lost current task cancellation.")
        secondary = _provider_recovery_failure_without_identity(
            failure,
            excluded_identity=id(cancellation),
        )
        if secondary is not None and not _attach_provider_recovery_secondary_failure(
            cancellation,
            secondary,
        ):
            add_exception_note_safely(
                cancellation,
                "Provider recovery also reported additional failures that could not be "
                "attached to caller cancellation.",
            )
        return cancellation
    cancellations = [
        candidate
        for candidate in iter_exception_tree(failure)
        if isinstance(candidate, asyncio.CancelledError)
    ]
    if not cancellations:
        return failure
    unexpected = unexpected_child_cancellation_error(cancellations[0], operation=operation)
    if failure is not cancellations[0]:
        set_exception_cause(unexpected, failure)
    return unexpected


def _raise_pending_provider_recovery_cancellation(*, cancellation_baseline: int) -> None:
    """Propagate caller cancellation even when a provider await suppressed delivery."""

    cancellation = _current_provider_recovery_cancellation(
        None,
        cancellation_baseline=cancellation_baseline,
    )
    if cancellation is None:
        return
    raise cancellation


def _current_provider_recovery_cancellation(
    failure: BaseException | None,
    *,
    cancellation_baseline: int,
) -> asyncio.CancelledError | None:
    """Return current cancellation without consuming and re-arming its task request."""

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
    if cancellation is not None:
        return cancellation
    cancel_message = getattr(task, "_cancel_message", None)
    return (
        asyncio.CancelledError()
        if cancel_message is None
        else asyncio.CancelledError(cancel_message)
    )


def _provider_recovery_cleanup_cancellation_baseline(
    publication_failure: BaseException | None,
    *,
    cancellation_baseline: int,
) -> int:
    """Ignore one cancellation already delivered by the publication await."""

    task = asyncio.current_task()
    if (
        not isinstance(publication_failure, asyncio.CancelledError)
        or task is None
        or task.cancelling() <= cancellation_baseline
    ):
        return cancellation_baseline
    return task.cancelling()


async def _close_provider_recovery_iterator(
    iterator: AsyncIterator[Any],
    *,
    cancellation_baseline: int,
    operation: str,
) -> Exception | None:
    """Close one provider-owned recovery iterator without forging caller cancellation."""

    try:
        close = getattr(iterator, "aclose", None)
        if callable(close):
            await close()
    except BaseException as close_failure:
        classified = _classify_provider_recovery_failure(
            close_failure,
            cancellation_baseline=cancellation_baseline,
            operation=operation,
        )
        if not isinstance(classified, Exception):
            if classified is close_failure:
                raise
            raise classified from close_failure
        return classified
    _raise_pending_provider_recovery_cancellation(cancellation_baseline=cancellation_baseline)
    return None


def _provider_recovery_cleanup_payload(
    failure: Exception,
    *,
    redactor: SecretRedactor,
) -> dict[str, Any]:
    diagnostic = exception_diagnostic(
        failure,
        empty_message="provider recovery stream cleanup failed",
        nonportable_message=(
            "Provider recovery stream cleanup failed with a non-portable diagnostic."
        ),
        redactor=redactor,
    )
    return {
        **diagnostic.payload_fields(),
        "phase": "provider_recovery_stream_cleanup",
    }


class _ProviderRecoveryRequiredPublicationFailureEvidence(RuntimeError):
    """Sanitized cleanup evidence retained when typed publication also fails."""

    def __init__(
        self,
        *,
        recovery_reason: ProviderOperationUnavailableReason,
        cleanup_diagnostic: dict[str, Any],
    ) -> None:
        copied = copy_durable_json_object(cleanup_diagnostic, "cleanup_diagnostic")
        error_type = copied.get("error_type")
        if type(error_type) is not str or not error_type.strip():
            raise ValueError("Provider recovery cleanup diagnostic has no error type.")
        super().__init__(
            "Provider recovery stream cleanup failed before "
            f"{recovery_reason.value} recovery evidence was acknowledged "
            f"({error_type})."
        )
        self.recovery_reason = recovery_reason
        self.cleanup_diagnostic = copied


class _ProviderOperationStreamStatusError(RuntimeError):
    """A validated operation error boundary carrying the provider's typed status."""

    def __init__(
        self,
        status: ProviderOperationStatus,
        provider_error: ModelProviderError,
    ) -> None:
        super().__init__(str(provider_error))
        self.status = status
        self.provider_error = provider_error


async def _emit_provider_recovery_required_event(
    event_writer: RuntimeEventWriter,
    event: Event,
    *,
    recovery_reason: ProviderOperationUnavailableReason,
    cleanup_failure: Exception | None,
    redactor: SecretRedactor,
) -> Event:
    """Publish typed recovery evidence without losing sanitized cleanup failure."""

    try:
        return await event_writer.emit(event)
    except BaseException as publication_failure:
        if cleanup_failure is None:
            raise
        cleanup_evidence = _ProviderRecoveryRequiredPublicationFailureEvidence(
            recovery_reason=recovery_reason,
            cleanup_diagnostic=_provider_recovery_cleanup_payload(
                cleanup_failure,
                redactor=redactor,
            ),
        )
        if not _attach_provider_recovery_secondary_failure(
            publication_failure,
            cleanup_evidence,
        ):
            raise BaseExceptionGroup(
                "Provider recovery evidence publication and stream cleanup both failed.",
                [publication_failure, cleanup_evidence],
            ) from None
        add_exception_note_safely(
            publication_failure,
            "Provider recovery-required publication also retained sanitized "
            f"{recovery_reason.value} stream-cleanup evidence.",
        )
        raise


def _provider_recovery_failure_without_identity(
    error: BaseException,
    *,
    excluded_identity: int,
) -> BaseException | None:
    """Remove one owned failure while retaining ordered non-overlapping subgroups."""

    pending: list[tuple[BaseException, bool]] = [(error, False)]
    children_by_group: dict[int, tuple[BaseException, ...]] = {}
    retained_by_identity: dict[int, BaseException | None] = {}
    while pending:
        candidate, expanded = pending.pop()
        candidate_id = id(candidate)
        if candidate_id in retained_by_identity:
            continue
        if candidate_id == excluded_identity:
            retained_by_identity[candidate_id] = None
            continue
        if not isinstance(candidate, BaseExceptionGroup):
            retained_by_identity[candidate_id] = candidate
            continue
        if expanded:
            children = children_by_group.pop(candidate_id, ())
            retained_children = [
                retained
                for child in children
                if (retained := retained_by_identity.get(id(child))) is not None
            ]
            if not retained_children:
                retained_by_identity[candidate_id] = None
            elif len(retained_children) == len(children) and all(
                retained is child
                for retained, child in zip(retained_children, children, strict=True)
            ):
                retained_by_identity[candidate_id] = candidate
            else:
                retained_by_identity[candidate_id] = BaseExceptionGroup(
                    "Provider recovery additional non-cancellation failures.",
                    retained_children,
                )
            continue
        children = exception_group_children(candidate)
        if children is None:
            retained_by_identity[candidate_id] = RuntimeError(
                "Provider recovery received an unreadable exception group."
            )
            continue
        children_by_group[candidate_id] = children
        pending.append((candidate, True))
        pending.extend((child, False) for child in reversed(children))

    return retained_by_identity.get(id(error))


def _provider_recovery_failure_graph_contains_identity(
    error: BaseException,
    *,
    target_identity: int,
) -> bool:
    """Return whether one safe exception graph contains an exact object identity."""

    pending = [error]
    visited: set[int] = set()
    while pending:
        candidate = pending.pop()
        candidate_id = id(candidate)
        if candidate_id == target_identity:
            return True
        if candidate_id in visited:
            continue
        visited.add(candidate_id)
        if isinstance(candidate, BaseExceptionGroup):
            children = exception_group_children(candidate)
            if children is not None:
                pending.extend(children)
        cause = exception_cause(candidate)
        if cause is not None:
            pending.append(cause)
        context = exception_context(candidate)
        if context is not None:
            pending.append(context)
    return False


def _detach_provider_recovery_back_edges(
    error: BaseException,
    *,
    target: BaseException,
) -> bool:
    """Remove causal links back to a primary error before attaching this graph."""

    pending = [error]
    visited: set[int] = set()
    while pending:
        candidate = pending.pop()
        candidate_id = id(candidate)
        if candidate_id in visited:
            continue
        visited.add(candidate_id)
        if isinstance(candidate, BaseExceptionGroup):
            children = exception_group_children(candidate)
            if children is not None:
                if any(child is target for child in children):
                    return False
                pending.extend(children)
        cause = exception_cause(candidate)
        if cause is target:
            if not set_exception_cause(candidate, None):
                return False
        elif cause is not None:
            pending.append(cause)
        context = exception_context(candidate)
        if context is target:
            if not set_exception_context(candidate, None):
                return False
        elif context is not None:
            pending.append(context)
    return True


def _attach_provider_recovery_secondary_failure(
    primary: BaseException,
    secondary: BaseException,
) -> bool:
    """Retain one ordered recovery failure as an acyclic causal graph."""

    if primary is secondary:
        return True
    if not _detach_provider_recovery_back_edges(secondary, target=primary):
        return False
    if _provider_recovery_failure_graph_contains_identity(
        secondary,
        target_identity=id(primary),
    ):
        return False
    prior_cause = exception_cause(primary)
    prior_context = None if prior_cause is not None else exception_context(primary)
    prior_failure = prior_cause if prior_cause is not None else prior_context
    if prior_failure is secondary or (
        prior_failure is not None
        and _provider_recovery_failure_graph_contains_identity(
            prior_failure,
            target_identity=id(secondary),
        )
    ):
        return True
    if prior_failure is None or _provider_recovery_failure_graph_contains_identity(
        secondary,
        target_identity=id(prior_failure),
    ):
        combined = secondary
    else:
        combined = BaseExceptionGroup(
            "Provider recovery publication and stream cleanup both failed.",
            [prior_failure, secondary],
        )
    if not set_exception_cause(primary, combined):
        return False
    if prior_context is not None:
        set_exception_context(primary, None)
    return True


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
        "Model completion encountered multiple terminal failures.",
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
class _AssistantStreamBoundaryValue:
    event: ModelStreamEvent
    tool_call: runtime_records.ToolCallRequest | None = None
    tool_call_part: ToolCallPart | None = None


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


def _tool_exposure_event(
    *,
    exposure: ToolExposure,
    session: Session,
    registered_agent: runtime_records.RegisteredAgentState,
    environment_name: str | None,
    model_step_identity: ModelStepIdentity,
    execution_profile: ExecutionProfileIdentity | None,
) -> Event:
    """Build one runtime-attested, content-minimized exposure evidence event."""

    if type(exposure) is not ToolExposure:
        raise TypeError("exposure must be a ToolExposure.")
    event = Event(
        type=EventType.TOOL_EXPOSURE_RECORDED,
        session_id=session.id,
        agent_name=registered_agent.spec.name,
        environment_name=environment_name,
        payload=exposure.model_dump(mode="json"),
    )
    event = _event_with_model_identity_authority(event, model_step_identity)
    event = event_with_runtime_payload_authority(
        event,
        "profile_id",
        "exposure_fingerprint",
    )
    return event_with_execution_profile_authority(event, execution_profile)


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
    execution_profile: ExecutionProfileIdentity | None


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
    execution_profile: ExecutionProfileIdentity | None


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
    execution_profile: ExecutionProfileIdentity | None


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


class _ProviderOperationCancellationClaimReleaseObserved(RuntimeError):
    """An intentional durable claim release won a concurrent heartbeat."""


@dataclass
class _ProviderOperationCancellationHeartbeat:
    stop: asyncio.Event
    release_intended: asyncio.Event
    task: asyncio.Task[None] | None = None


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
        request_footprint: RequestFootprintConfig,
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
        self._request_footprint = copy_request_footprint_config(request_footprint)
        self._max_file_attachment_bytes = max_file_attachment_bytes
        self._max_total_file_attachment_bytes = max_total_file_attachment_bytes
        self._max_file_attachments_per_request = max_file_attachments_per_request
        self._secret_redactor = secret_redactor
        self._checkpoint_transform = checkpoint_transform
        self._apply_budget_evaluation = apply_budget_evaluation
        self._apply_limit_evaluation = apply_limit_evaluation
        self._stop_for_budget_reservation_failure = stop_for_budget_reservation_failure
        self._provider_operation_reconciliation_tasks: set[asyncio.Task[None]] = set()
        self._provider_operation_cancellation_heartbeats: dict[
            str,
            _ProviderOperationCancellationHeartbeat,
        ] = {}

    def _retain_provider_operation_reconciliation(self, task: asyncio.Task[None]) -> None:
        self._provider_operation_reconciliation_tasks.add(task)

        def settled(completed: asyncio.Task[None]) -> None:
            self._provider_operation_reconciliation_tasks.discard(completed)
            _consume_detached_task_outcome(completed)

        task.add_done_callback(settled)

    def _start_provider_operation_cancellation_heartbeat(
        self,
        *,
        session: Session,
        claim: ProviderOperationCancellationClaim,
        ownership_lost: asyncio.Event,
    ) -> None:
        control = _ProviderOperationCancellationHeartbeat(
            stop=asyncio.Event(),
            release_intended=asyncio.Event(),
        )
        task = asyncio.create_task(
            self._heartbeat_provider_operation_cancellation_claim(
                session=session,
                claim=claim,
                owner_task=asyncio.current_task(),
                ownership_lost=ownership_lost,
                stop=control.stop,
                release_intended=control.release_intended,
            )
        )
        control.task = task
        self._provider_operation_cancellation_heartbeats[claim.claim_id] = control

        def settled(completed: asyncio.Task[None]) -> None:
            if self._provider_operation_cancellation_heartbeats.get(claim.claim_id) is control:
                self._provider_operation_cancellation_heartbeats.pop(claim.claim_id, None)
            _consume_detached_task_outcome(completed)

        task.add_done_callback(settled)

    def _mark_provider_operation_cancellation_claim_release(
        self,
        claim: ProviderOperationCancellationClaim,
    ) -> None:
        control = self._provider_operation_cancellation_heartbeats.get(claim.claim_id)
        if control is not None:
            control.release_intended.set()

    async def _stop_provider_operation_cancellation_heartbeat(
        self,
        claim: ProviderOperationCancellationClaim,
    ) -> None:
        control = self._provider_operation_cancellation_heartbeats.get(claim.claim_id)
        if control is None or control.task is None:
            return
        control.stop.set()
        await control.task

    async def _persist_provider_operation_cancellation_event(
        self,
        *,
        event_type: EventType,
        cancellation_status: str,
        session: Session,
        stage: ModelCompletionStage,
        state: ProviderOperationState,
        interaction_id: str,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_provider: runtime_records.RegisteredProvider,
        environment_name: str | None,
        step: int,
        attempt: int,
        max_attempts: int,
        model_attempt_identity: ModelAttemptIdentity,
        provider_status: ProviderOperationStatus | None = None,
        error_type: str | None = None,
        cancellation_claim: ProviderOperationCancellationClaim | None = None,
        release_cancellation_claim: bool = False,
    ) -> Event:
        identity_material = canonical_durable_json_bytes(
            {
                "schema_version": 1,
                "stage_id": stage.stage_id,
                "run_epoch": session.run_epoch,
                "event_type": event_type.value,
                "cancellation_status": cancellation_status,
                "provider_status": None if provider_status is None else provider_status.value,
                "error_type": error_type,
            },
            "provider_operation_cancellation_identity",
        )
        payload: dict[str, Any] = {
            "provider": registered_provider.name,
            "model": session.model,
            "step": step,
            "attempt": attempt,
            "max_attempts": max_attempts,
            **model_attempt_identity.payload(),
            "source_run_epoch": stage.source_run_epoch,
            "run_epoch": session.run_epoch,
            "operation_id": state.operation_id,
            "stream_protocol": state.stream_protocol,
            "cancellation_status": cancellation_status,
        }
        if provider_status is not None:
            payload["provider_status"] = provider_status.value
        if error_type is not None:
            payload["error_type"] = require_durable_clean_nonblank(error_type, "error_type")
        event = _event_with_model_identity_authority(
            Event(
                id=f"provider-cancel:v1:{sha256(identity_material).hexdigest()}",
                type=event_type,
                session_id=session.id,
                interaction_id=interaction_id,
                timestamp=stage.prepared_at,
                agent_name=registered_agent.spec.name,
                environment_name=environment_name,
                payload=payload,
            ),
            model_attempt_identity,
        )
        recovery_context = model_completion_recovery_context_from_stage(stage)
        event = event_with_execution_profile_fingerprint_authority(
            event,
            (None if recovery_context is None else recovery_context.execution_profile_fingerprint),
        )
        event = event_with_runtime_payload_authority(
            event,
            "operation_id",
            "stream_protocol",
        )
        prepared = self._event_writer.prepare(event_with_runtime_generated_id(event))
        if cancellation_claim is None:
            if release_cancellation_claim:
                raise ValueError("Cancellation-claim release requires the exact claim.")
            persisted = await self._event_writer.persist_exact_replay(prepared)
        else:

            def checkpoint_transform(
                current_session: Session,
                checkpoint: dict[str, Any] | None,
            ) -> dict[str, Any]:
                if current_session.run_epoch != session.run_epoch:
                    raise SessionRunFenced(
                        "Provider-operation cancellation ownership changed at publication."
                    )
                if release_cancellation_claim:
                    return checkpoint_without_provider_operation_cancellation_claim(
                        checkpoint,
                        cancellation_claim,
                    )
                return checkpoint_with_provider_operation_cancellation_claim(
                    checkpoint,
                    cancellation_claim,
                )

            if release_cancellation_claim:
                self._mark_provider_operation_cancellation_claim_release(cancellation_claim)
            await self._session_store.publish_checkpoint_and_events(
                session.id,
                checkpoint_transform=checkpoint_transform,
                events=[prepared],
                expected_run_epoch=session.run_epoch,
            )
            if release_cancellation_claim:
                await self._stop_provider_operation_cancellation_heartbeat(cancellation_claim)
            persisted = prepared
        [emitted] = await self._event_writer.fan_out_persisted([persisted])
        return emitted

    async def _release_provider_operation_cancellation_claim(
        self,
        *,
        session: Session,
        claim: ProviderOperationCancellationClaim,
    ) -> None:
        def release_claim(
            current_session: Session,
            checkpoint: dict[str, Any] | None,
        ) -> dict[str, Any]:
            if current_session.run_epoch != session.run_epoch:
                raise SessionRunFenced(
                    "Provider-operation cancellation ownership changed before claim release."
                )
            return checkpoint_without_provider_operation_cancellation_claim(
                checkpoint,
                claim,
            )

        self._mark_provider_operation_cancellation_claim_release(claim)
        await self._session_store.publish_checkpoint_and_events(
            session.id,
            checkpoint_transform=release_claim,
            events=[],
            expected_run_epoch=session.run_epoch,
        )
        await self._stop_provider_operation_cancellation_heartbeat(claim)

    async def _heartbeat_provider_operation_cancellation_claim(
        self,
        *,
        session: Session,
        claim: ProviderOperationCancellationClaim,
        owner_task: asyncio.Task[Any] | None,
        ownership_lost: asyncio.Event,
        stop: asyncio.Event,
        release_intended: asyncio.Event,
    ) -> None:
        """Renew one cancellation lease until its owner releases or loses it."""

        while not stop.is_set():
            try:
                await asyncio.wait_for(
                    stop.wait(),
                    timeout=_PROVIDER_OPERATION_CANCELLATION_CLAIM_HEARTBEAT_SECONDS,
                )
                return
            except TimeoutError:
                pass
            if owner_task is not None and owner_task.done():
                return
            renewed = claim.model_copy(
                update={
                    "expires_at": datetime.now(UTC) + _PROVIDER_OPERATION_CANCELLATION_CLAIM_LEASE
                }
            )

            def renew_claim(
                current_session: Session,
                checkpoint: dict[str, Any] | None,
                renewed_claim: ProviderOperationCancellationClaim = renewed,
            ) -> dict[str, Any]:
                if current_session.run_epoch != session.run_epoch:
                    raise SessionRunFenced(
                        "Provider-operation cancellation ownership changed before renewal."
                    )
                existing = provider_operation_cancellation_claim_from_checkpoint(checkpoint)
                if existing is None and release_intended.is_set():
                    raise _ProviderOperationCancellationClaimReleaseObserved
                if existing is None or not existing.same_owner(claim):
                    raise RuntimeError("Provider-operation cancellation claim is no longer active.")
                return checkpoint_with_provider_operation_cancellation_claim(
                    checkpoint,
                    renewed_claim,
                )

            try:
                await self._session_store.publish_checkpoint_and_events(
                    session.id,
                    checkpoint_transform=renew_claim,
                    events=[],
                    expected_run_epoch=session.run_epoch,
                )
            except _ProviderOperationCancellationClaimReleaseObserved:
                return
            except asyncio.CancelledError:
                raise
            except BaseException:
                ownership_lost.set()
                if owner_task is not None and not owner_task.done():
                    owner_task.cancel("Provider-operation cancellation ownership heartbeat failed.")
                raise

    async def _cancel_started_provider_operation(
        self,
        *,
        adapter: ProviderOperationAdapter,
        state: ProviderOperationState,
        failure: SessionInterruptedByRequest | asyncio.CancelledError,
        session: Session,
        stage: ModelCompletionStage,
        interaction_id: str,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_provider: runtime_records.RegisteredProvider,
        environment_name: str | None,
        step: int,
        attempt: int,
        max_attempts: int,
        model_attempt_identity: ModelAttemptIdentity,
        settle_budget: bool = False,
    ) -> ProviderOperationSnapshot | None:
        async def require_cancellation_owner() -> None:
            current = await self._session_store.load(session.id)
            if current is None:
                raise KeyError(f"Session not found: {session.id}")
            if current.run_epoch != session.run_epoch:
                raise SessionRunFenced(
                    "Provider-operation cancellation run epoch is stale: expected "
                    f"{session.run_epoch}, current {current.run_epoch}."
                )

        await require_cancellation_owner()
        cancellation_claim = ProviderOperationCancellationClaim(
            claim_id=(
                f"provider-cancel:{stage.stage_id}:{session.run_epoch}:"
                f"{state.operation_id}:{state.stream_protocol}"
            ),
            stage_id=stage.stage_id,
            run_epoch=session.run_epoch,
            operation_id=state.operation_id,
            stream_protocol=state.stream_protocol,
            expires_at=datetime.now(UTC) + _PROVIDER_OPERATION_CANCELLATION_CLAIM_LEASE,
        )
        support = adapter.cancellation_support
        if type(support) is not ProviderOperationCancellationSupport:
            raise TypeError(
                "ProviderOperationAdapter.cancellation_support must return "
                "ProviderOperationCancellationSupport."
            )
        await self._persist_provider_operation_cancellation_event(
            event_type=EventType.PROVIDER_OPERATION_CANCEL_REQUESTED,
            cancellation_status="requested",
            session=session,
            stage=stage,
            state=state,
            interaction_id=interaction_id,
            registered_agent=registered_agent,
            registered_provider=registered_provider,
            environment_name=environment_name,
            step=step,
            attempt=attempt,
            max_attempts=max_attempts,
            model_attempt_identity=model_attempt_identity,
            cancellation_claim=cancellation_claim,
        )
        cancellation_ownership_lost = asyncio.Event()
        self._start_provider_operation_cancellation_heartbeat(
            session=session,
            claim=cancellation_claim,
            ownership_lost=cancellation_ownership_lost,
        )
        await require_cancellation_owner()
        if support is ProviderOperationCancellationSupport.UNSUPPORTED:
            await self._persist_provider_operation_cancellation_event(
                event_type=EventType.PROVIDER_OPERATION_CANCEL_RESOLVED,
                cancellation_status="unsupported",
                session=session,
                stage=stage,
                state=state,
                interaction_id=interaction_id,
                registered_agent=registered_agent,
                registered_provider=registered_provider,
                environment_name=environment_name,
                step=step,
                attempt=attempt,
                max_attempts=max_attempts,
                model_attempt_identity=model_attempt_identity,
                cancellation_claim=cancellation_claim,
                release_cancellation_claim=True,
            )
            failure.__dict__["provider_operation_accounting_pending"] = True
            return None
        cancellation, snapshot = await _cancel_provider_operation_after_definite_absence(
            adapter=adapter,
            state=state,
            failure=failure,
            cancellation=(failure if isinstance(failure, asyncio.CancelledError) else None),
            ownership_lost=cancellation_ownership_lost,
        )
        if cancellation is not None and cancellation is not failure:
            raise cancellation from failure
        if snapshot is None:
            await require_cancellation_owner()
            await self._persist_provider_operation_cancellation_event(
                event_type=EventType.PROVIDER_OPERATION_CANCEL_RESOLVED,
                cancellation_status="failed",
                session=session,
                stage=stage,
                state=state,
                interaction_id=interaction_id,
                registered_agent=registered_agent,
                registered_provider=registered_provider,
                environment_name=environment_name,
                step=step,
                attempt=attempt,
                max_attempts=max_attempts,
                model_attempt_identity=model_attempt_identity,
                error_type="CancellationUnconfirmed",
                cancellation_claim=cancellation_claim,
                release_cancellation_claim=True,
            )
            failure.__dict__["provider_operation_accounting_pending"] = True
            return None
        cancellation_status = {
            ProviderOperationStatus.CANCELLED: "cancelled",
            ProviderOperationStatus.COMPLETED: "completed",
            ProviderOperationStatus.QUEUED: "pending",
            ProviderOperationStatus.IN_PROGRESS: "pending",
            ProviderOperationStatus.UNAVAILABLE: "unavailable",
            ProviderOperationStatus.FAILED: "failed",
            ProviderOperationStatus.EXPIRED: "failed",
        }[snapshot.status]
        await require_cancellation_owner()
        retain_claim_after_resolution = snapshot.status is ProviderOperationStatus.COMPLETED or (
            snapshot.status is ProviderOperationStatus.CANCELLED and bool(stage.reservation_ids)
        )
        await self._persist_provider_operation_cancellation_event(
            event_type=EventType.PROVIDER_OPERATION_CANCEL_RESOLVED,
            cancellation_status=cancellation_status,
            session=session,
            stage=stage,
            state=state,
            interaction_id=interaction_id,
            registered_agent=registered_agent,
            registered_provider=registered_provider,
            environment_name=environment_name,
            step=step,
            attempt=attempt,
            max_attempts=max_attempts,
            model_attempt_identity=model_attempt_identity,
            provider_status=snapshot.status,
            cancellation_claim=cancellation_claim,
            release_cancellation_claim=not retain_claim_after_resolution,
        )
        if snapshot.status is not ProviderOperationStatus.CANCELLED:
            failure.__dict__["provider_operation_accounting_pending"] = True
        elif settle_budget and stage.reservation_ids:
            await require_cancellation_owner()
            recovery_context = model_completion_recovery_context_from_stage(stage)
            if recovery_context is None:
                raise ProviderOperationEvidenceError(
                    "Budgeted provider-operation cancellation has no durable accounting context."
                )
            try:
                await (
                    self._run_limit_controller.reconcile_cancelled_provider_operation_reservations(
                        reservation_ids=stage.reservation_ids,
                        recovery_contexts=recovery_context.budget_reservations,
                        session=session,
                        provider_name=registered_provider.name,
                        model_attempt_identity=model_attempt_identity,
                        dispatch_id=stage.stage_id,
                        request_billing_identity=recovery_context.billing_identity,
                    )
                )
            except (KeyError, NotImplementedError, TypeError, ValueError) as accounting_error:
                raise ProviderOperationEvidenceError(
                    "Provider-operation cancellation could not reconstruct its original "
                    "budget reservation and pricing context."
                ) from accounting_error
            await self._release_provider_operation_cancellation_claim(
                session=session,
                claim=cancellation_claim,
            )
        elif stage.reservation_ids:
            failure.__dict__["provider_operation_cancellation_claim"] = cancellation_claim
        return snapshot

    async def cancel_provider_operation_for_interruption(
        self,
        *,
        session: Session,
        stage: ModelCompletionStage,
        operation: RecoverableProviderOperation,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_provider: runtime_records.RegisteredProvider,
        environment_name: str | None,
    ) -> ProviderOperationSnapshot | None:
        """Cancel one durably identified operation after its worker disappears."""

        provider = registered_provider.provider
        adapter = provider.provider_operations
        if (
            provider.provider_operation_mode is not ProviderOperationMode.BACKGROUND
            or not isinstance(adapter, ProviderOperationAdapter)
        ):
            raise RuntimeError(
                "The registered provider no longer supports its durable background operation."
            )
        if operation.provider != registered_provider.name:
            raise RuntimeError("Provider-operation cancellation resolved a different provider.")
        if operation.model != session.model:
            raise RuntimeError("Provider-operation cancellation resolved a different model.")
        if operation.model_attempt_identity.model_step_id != stage.logical_step_id:
            raise RuntimeError(
                "Provider-operation cancellation belongs to a different model stage."
            )
        return await self._cancel_started_provider_operation(
            adapter=adapter,
            state=operation.state,
            failure=SessionInterruptedByRequest(session.id),
            session=session,
            stage=stage,
            interaction_id=operation.interaction_id,
            registered_agent=registered_agent,
            registered_provider=registered_provider,
            environment_name=environment_name,
            step=operation.step,
            attempt=operation.attempt,
            max_attempts=operation.max_attempts,
            model_attempt_identity=operation.model_attempt_identity,
            settle_budget=True,
        )

    def _provider_operation_progress_event(
        self,
        *,
        stage: ModelCompletionStage,
        state: ProviderOperationState,
        stream_event: ModelStreamEvent,
        runtime_event: Event | None,
        session: Session,
        interaction_id: str,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_provider: runtime_records.RegisteredProvider,
        environment_name: str | None,
        step: int,
        attempt: int,
        max_attempts: int,
        model_attempt_identity: ModelAttemptIdentity,
    ) -> Event:
        """Attach private reconnect state to its corresponding normalized event."""

        metadata = stream_event.recovery_metadata
        if metadata is None or metadata.cursor is None:
            raise ProviderOperationEvidenceError(
                "Reconnectable provider events must carry a monotonic recovery cursor."
            )
        event_id = provider_operation_progress_event_id(stage.stage_id, metadata.cursor)
        progress_envelope = provider_operation_progress_envelope(state, stream_event)
        if _provider_operation_progress_contains_secret(
            progress_envelope,
            redactor=self._secret_redactor,
        ):
            raise ProviderOperationEvidenceError(
                "Provider-operation recovery metadata or normalized output contains a "
                "workload secret and cannot cross the durable recovery boundary."
            )
        envelope = progress_envelope.model_dump(mode="json")
        if runtime_event is None:
            event = _event_with_model_identity_authority(
                Event(
                    id=event_id,
                    type=EventType.PROVIDER_OPERATION_PROGRESS,
                    session_id=session.id,
                    interaction_id=interaction_id,
                    agent_name=registered_agent.spec.name,
                    environment_name=environment_name,
                    payload={
                        "provider": registered_provider.name,
                        "step": step,
                        "attempt": attempt,
                        "max_attempts": max_attempts,
                        **model_attempt_identity.payload(),
                        "operation_id": state.operation_id,
                        "stream_protocol": state.stream_protocol,
                        "provider_operation_progress": envelope,
                    },
                ),
                model_attempt_identity,
            )
            event = event_with_runtime_payload_authority(
                event,
                "operation_id",
                "stream_protocol",
            )
        else:
            event = copy_event(runtime_event)
            if event.interaction_id not in {None, interaction_id}:
                raise ProviderOperationEvidenceError(
                    "Provider-operation progress changed its owning interaction."
                )
            payload = copy_durable_json_object(event.payload, "event.payload")
            payload["provider_operation_progress"] = envelope
            event = event.model_copy(
                update={
                    "id": event_id,
                    "interaction_id": interaction_id,
                    "payload": payload,
                },
                deep=True,
            )
        recovery_context = model_completion_recovery_context_from_stage(stage)
        event = event_with_execution_profile_fingerprint_authority(
            event,
            (None if recovery_context is None else recovery_context.execution_profile_fingerprint),
        )
        return event_with_runtime_generated_id(event)

    async def _commit_provider_operation_stream_event(
        self,
        *,
        stage: ModelCompletionStage,
        state: ProviderOperationState,
        stream_event: ModelStreamEvent,
        runtime_event: Event | None,
        session: Session,
        interaction_id: str,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_provider: runtime_records.RegisteredProvider,
        environment_name: str | None,
        step: int,
        attempt: int,
        max_attempts: int,
        model_attempt_identity: ModelAttemptIdentity,
    ) -> tuple[ProviderOperationProgressCommit, Event | None]:
        event = self._provider_operation_progress_event(
            stage=stage,
            state=state,
            stream_event=stream_event,
            runtime_event=runtime_event,
            session=session,
            interaction_id=interaction_id,
            registered_agent=registered_agent,
            registered_provider=registered_provider,
            environment_name=environment_name,
            step=step,
            attempt=attempt,
            max_attempts=max_attempts,
            model_attempt_identity=model_attempt_identity,
        )
        prepared = self._event_writer.prepare(event)
        commit = await commit_provider_operation_progress(
            self._session_store,
            stage=stage,
            model_attempt_identity=model_attempt_identity,
            current_state=state,
            stream_event=stream_event,
            event=prepared,
            expected_run_epoch=session.run_epoch,
        )
        if commit.replayed:
            return commit, None
        [emitted] = await self._event_writer.fan_out_persisted([commit.event])
        return commit, emitted

    async def recover_provider_operation_start(
        self,
        *,
        session: Session,
        stage: ModelCompletionStage,
        start: RecoverableProviderOperationStart,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_provider: runtime_records.RegisteredProvider,
        environment_name: str | None,
        model_completion_publisher: ModelCompletionPublisher,
    ) -> ProviderOperationRecoveryResult:
        """Recover start-only evidence without persisting or replaying a raw request."""

        provider = registered_provider.provider
        adapter = provider.provider_operations
        recovery_context = model_completion_recovery_context_from_stage(stage)
        exact_recovery = start.idempotency_support is ProviderOperationStartIdempotencySupport.EXACT
        if (
            provider.provider_operation_mode is not ProviderOperationMode.BACKGROUND
            or not isinstance(adapter, ProviderOperationAdapter)
        ):
            raise ProviderOperationEvidenceError(
                "Provider-operation start recovery is unavailable."
            )
        if start.provider != registered_provider.name or start.model != session.model:
            raise ProviderOperationEvidenceError(
                "Provider-operation start recovery resolved a different provider scope."
            )

        async def unavailable(
            reason: ProviderOperationUnavailableReason,
            *,
            cleanup_failure: Exception | None = None,
        ) -> ProviderOperationRecoveryResult:
            payload: dict[str, Any] = {
                "provider": registered_provider.name,
                "model": session.model,
                "step": start.step,
                "attempt": start.attempt,
                "max_attempts": start.max_attempts,
                **start.model_attempt_identity.payload(),
                "source_run_epoch": start.source_run_epoch,
                "run_epoch": session.run_epoch,
                "start_id": start.start_id,
                "status": reason.value,
                "recovery_reason": reason.value,
                "idempotent_start_recovery": exact_recovery,
            }
            if cleanup_failure is not None:
                payload["provider_cleanup_failure"] = _provider_recovery_cleanup_payload(
                    cleanup_failure,
                    redactor=self._secret_redactor,
                )
            required = _event_with_model_identity_authority(
                Event(
                    type=EventType.PROVIDER_OPERATION_RECOVERY_REQUIRED,
                    session_id=session.id,
                    interaction_id=start.interaction_id,
                    agent_name=registered_agent.spec.name,
                    environment_name=environment_name,
                    payload=payload,
                ),
                start.model_attempt_identity,
            )
            required = event_with_execution_profile_fingerprint_authority(
                required,
                (
                    None
                    if recovery_context is None
                    else recovery_context.execution_profile_fingerprint
                ),
            )
            required = event_with_runtime_payload_authority(required, "start_id")
            emitted = await _emit_provider_recovery_required_event(
                self._event_writer,
                required,
                recovery_reason=reason,
                cleanup_failure=cleanup_failure,
                redactor=self._secret_redactor,
            )
            return ProviderOperationRecoveryResult(
                status=ProviderOperationRecoveryStatus.UNAVAILABLE,
                events=(emitted,),
                unavailable_reason=reason,
            )

        if start.idempotency_support is not ProviderOperationStartIdempotencySupport.EXACT:
            return await unavailable(ProviderOperationUnavailableReason.AMBIGUOUS_SUBMISSION)
        if adapter.start_idempotency_support is not ProviderOperationStartIdempotencySupport.EXACT:
            return await unavailable(ProviderOperationUnavailableReason.AMBIGUOUS_SUBMISSION)
        recovery_task = asyncio.current_task()
        recovery_cancellation_baseline = 0 if recovery_task is None else recovery_task.cancelling()
        try:
            raw_connection = await adapter.recover_start(
                ProviderOperationStartRecoveryRequest(idempotency_key=start.start_id)
            )
        except BaseException as recovery_failure:
            recovery_failure = _classify_provider_recovery_failure(
                recovery_failure,
                cancellation_baseline=recovery_cancellation_baseline,
                operation="Provider operation start recovery",
            )
            if not isinstance(recovery_failure, Exception):
                raise recovery_failure
            return await unavailable(ProviderOperationUnavailableReason.UNAVAILABLE)
        try:
            connection = copy_provider_operation_connection(raw_connection)
        except Exception as malformed_failure:
            cleanup_failure = None
            if type(raw_connection) is ProviderOperationConnection:
                cleanup_failure = await _close_provider_recovery_iterator(
                    raw_connection.events,
                    cancellation_baseline=recovery_cancellation_baseline,
                    operation="Provider operation malformed start recovery stream cleanup",
                )
            _raise_pending_provider_recovery_cancellation(
                cancellation_baseline=recovery_cancellation_baseline
            )
            if cleanup_failure is not None:
                add_exception_note_safely(
                    malformed_failure,
                    "Provider recovery stream cleanup also failed with "
                    f"{type(cleanup_failure).__name__}.",
                )
            return await unavailable(
                ProviderOperationUnavailableReason.MALFORMED,
                cleanup_failure=cleanup_failure,
            )

        operation_event = _event_with_model_identity_authority(
            Event(
                id=provider_operation_started_event_id(start.start_id),
                type=EventType.PROVIDER_OPERATION_STARTED,
                session_id=session.id,
                interaction_id=start.interaction_id,
                agent_name=registered_agent.spec.name,
                environment_name=environment_name,
                payload={
                    "provider": registered_provider.name,
                    "model": session.model,
                    "step": start.step,
                    "attempt": start.attempt,
                    "max_attempts": start.max_attempts,
                    **start.model_attempt_identity.payload(),
                    "source_run_epoch": start.source_run_epoch,
                    "start_id": start.start_id,
                    "state_version": connection.state.version,
                    "operation_id": connection.state.operation_id,
                    "stream_protocol": connection.state.stream_protocol,
                    "status": connection.status.value,
                    "recovery_metadata": connection.state.recovery_metadata.model_dump(
                        mode="json",
                        exclude_none=True,
                    ),
                    "idempotent_start_recovery": True,
                },
            ),
            start.model_attempt_identity,
        )
        operation_event = event_with_execution_profile_fingerprint_authority(
            operation_event,
            (None if recovery_context is None else recovery_context.execution_profile_fingerprint),
        )
        operation_event = event_with_runtime_payload_authority(
            operation_event,
            "start_id",
        )
        publication_failure: BaseException | None = None
        try:
            persisted = await self._event_writer.persist_exact_replay(operation_event)
            [emitted] = await self._event_writer.fan_out_persisted([persisted])
        except BaseException as failure:
            publication_failure = failure
            raise
        finally:
            cleanup_cancellation_baseline = _provider_recovery_cleanup_cancellation_baseline(
                publication_failure,
                cancellation_baseline=recovery_cancellation_baseline,
            )
            try:
                cleanup_failure = await _close_provider_recovery_iterator(
                    raw_connection.events,
                    cancellation_baseline=cleanup_cancellation_baseline,
                    operation="Provider operation start recovery stream cleanup",
                )
            except BaseException as cleanup_signal:
                if (
                    publication_failure is not None
                    and cleanup_signal is not publication_failure
                    and not any(
                        candidate is publication_failure
                        for candidate in iter_exception_tree(cleanup_signal)
                    )
                ):
                    cleanup_cause = exception_cause(cleanup_signal)
                    if (
                        cleanup_cause is not None
                        and cleanup_cause is not publication_failure
                        and not _attach_provider_recovery_secondary_failure(
                            publication_failure,
                            cleanup_cause,
                        )
                    ):
                        raise BaseExceptionGroup(
                            "Provider recovery publication and stream cleanup both failed.",
                            [publication_failure, cleanup_signal],
                        ) from cleanup_signal
                    if not set_exception_cause(cleanup_signal, publication_failure):
                        raise BaseExceptionGroup(
                            "Provider recovery publication and stream cleanup both failed.",
                            [publication_failure, cleanup_signal],
                        ) from cleanup_signal
                raise
            if cleanup_failure is not None:
                diagnostic = _provider_recovery_cleanup_payload(
                    cleanup_failure,
                    redactor=self._secret_redactor,
                )
                if publication_failure is not None:
                    if not _attach_provider_recovery_secondary_failure(
                        publication_failure,
                        cleanup_failure,
                    ):
                        raise BaseExceptionGroup(
                            "Provider recovery publication and stream cleanup both failed.",
                            [publication_failure, cleanup_failure],
                        )
                    add_exception_note_safely(
                        publication_failure,
                        "Provider recovery stream cleanup also failed with "
                        f"{diagnostic['error_type']}.",
                    )
                else:
                    logger.warning(
                        "Provider recovery stream cleanup failed after exact start publication: %s",
                        diagnostic["error_type"],
                    )
        operation = await load_recoverable_provider_operation(
            self._session_store,
            stage,
        )
        if operation is None:
            raise ProviderOperationEvidenceError(
                "Idempotent provider start did not produce recoverable operation evidence."
            )
        recovered = await self.recover_provider_operation(
            session=session,
            stage=stage,
            operation=operation,
            registered_agent=registered_agent,
            registered_provider=registered_provider,
            environment_name=environment_name,
            recovery_context=recovery_context,
            model_completion_publisher=model_completion_publisher,
        )
        return ProviderOperationRecoveryResult(
            status=recovered.status,
            events=(emitted, *recovered.events),
            completion_event=recovered.completion_event,
            unavailable_reason=recovered.unavailable_reason,
        )

    async def recover_provider_operation(
        self,
        *,
        session: Session,
        stage: ModelCompletionStage,
        operation: RecoverableProviderOperation,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_provider: runtime_records.RegisteredProvider,
        environment_name: str | None,
        recovery_context: ModelCompletionRecoveryContext | None,
        model_completion_publisher: ModelCompletionPublisher,
    ) -> ProviderOperationRecoveryResult:
        """Retrieve and atomically publish one exact offline provider operation."""

        provider = registered_provider.provider
        adapter = provider.provider_operations
        if (
            provider.provider_operation_mode is not ProviderOperationMode.BACKGROUND
            or not isinstance(
                adapter,
                ProviderOperationAdapter,
            )
        ):
            raise RuntimeError(
                "The registered provider no longer supports its durable background operation."
            )
        if operation.provider != registered_provider.name:
            raise RuntimeError("Provider-operation recovery resolved a different provider.")
        if operation.model != session.model:
            raise RuntimeError("Provider-operation recovery resolved a different model.")
        if operation.model_attempt_identity.model_step_id != stage.logical_step_id:
            raise RuntimeError("Provider-operation recovery belongs to a different model stage.")
        if (
            recovery_context is not None
            and type(recovery_context) is not ModelCompletionRecoveryContext
        ):
            raise TypeError("Provider-operation recovery context is invalid.")
        if (
            recovery_context is not None
            and recovery_context.structured_output is not None
            and recovery_context.structured_output.strategy is StructuredOutputStrategy.NATIVE
        ):
            raise ProviderOperationEvidenceError(
                "Offline recovery of native structured output requires manual reconciliation."
            )
        if durable_value_contains_secret(
            operation.state.recovery_metadata.opaque,
            redactor=self._secret_redactor,
            path=("provider_operation_recovery_opaque",),
        ) or any(
            _provider_operation_progress_contains_secret(
                provider_operation_progress_envelope(operation.state, accepted_event),
                redactor=self._secret_redactor,
            )
            for accepted_event in operation.accepted_stream_events
        ):
            raise ProviderOperationEvidenceError(
                "Stored provider-operation recovery state conflicts with a current workload secret."
            )

        cancellation_claim = ProviderOperationCancellationClaim(
            claim_id=(
                f"provider-cancel:{stage.stage_id}:{session.run_epoch}:"
                f"{operation.state.operation_id}:{operation.state.stream_protocol}"
            ),
            stage_id=stage.stage_id,
            run_epoch=session.run_epoch,
            operation_id=operation.state.operation_id,
            stream_protocol=operation.state.stream_protocol,
            expires_at=datetime.now(UTC) + _PROVIDER_OPERATION_CANCELLATION_CLAIM_LEASE,
        )
        recovery_under_cancellation_claim = False

        async def require_recovery_owner() -> None:
            nonlocal recovery_under_cancellation_claim
            current = await self._session_store.load(session.id)
            if current is None:
                raise KeyError(f"Session not found: {session.id}")
            if current.run_epoch != session.run_epoch:
                raise SessionRunFenced(
                    "Provider-operation recovery run epoch is stale: expected "
                    f"{session.run_epoch}, current {current.run_epoch}."
                )
            if current.status is SessionStatus.INTERRUPTING:
                checkpoint = await self._session_store.load_checkpoint(session.id)
                if (
                    stored_claim := provider_operation_cancellation_claim_from_checkpoint(
                        checkpoint
                    )
                ) is None or not stored_claim.same_owner(cancellation_claim):
                    raise SessionInterruptedByRequest(session.id)
                recovery_under_cancellation_claim = True

        def recovery_event(
            event_type: EventType,
            *,
            status: str,
            recovery_reason: ProviderOperationUnavailableReason | None = None,
            cleanup_failure: Exception | None = None,
        ) -> Event:
            payload: dict[str, Any] = {
                "provider": registered_provider.name,
                "model": session.model,
                "step": operation.step,
                "attempt": operation.attempt,
                "max_attempts": operation.max_attempts,
                **operation.model_attempt_identity.payload(),
                "source_run_epoch": operation.source_run_epoch,
                "run_epoch": session.run_epoch,
                "operation_id": operation.state.operation_id,
                "stream_protocol": operation.state.stream_protocol,
                "status": status,
            }
            if recovery_reason is not None:
                payload["recovery_reason"] = recovery_reason.value
            if cleanup_failure is not None:
                payload["provider_cleanup_failure"] = _provider_recovery_cleanup_payload(
                    cleanup_failure,
                    redactor=self._secret_redactor,
                )
            event = _event_with_model_identity_authority(
                Event(
                    type=event_type,
                    session_id=session.id,
                    interaction_id=operation.interaction_id,
                    agent_name=registered_agent.spec.name,
                    environment_name=environment_name,
                    payload=payload,
                ),
                operation.model_attempt_identity,
            )
            event = event_with_execution_profile_fingerprint_authority(
                event,
                None
                if recovery_context is None
                else recovery_context.execution_profile_fingerprint,
            )
            return event_with_runtime_payload_authority(
                event,
                "operation_id",
                "stream_protocol",
            )

        await require_recovery_owner()
        scheduled = await self._event_writer.emit(
            recovery_event(
                EventType.PROVIDER_OPERATION_RECONNECT_SCHEDULED,
                status=operation.status.value,
            )
        )
        await require_recovery_owner()
        started = await self._event_writer.emit(
            recovery_event(
                EventType.PROVIDER_OPERATION_RECONNECT_STARTED,
                status=operation.status.value,
            )
        )
        assistant_parts: list[transcript_helpers.AssistantContentPart] = []
        tool_calls: list[runtime_records.ToolCallRequest] = []
        completed_boundary = None
        completed_event: ModelStreamEvent | None = None
        completion_diagnostics: dict[str, Any] = {}
        terminal_progress_verified = False
        current_state = operation.state
        recovered_events: list[Event] = [scheduled, started]
        recovery_task = asyncio.current_task()
        recovery_cancellation_baseline = 0 if recovery_task is None else recovery_task.cancelling()
        post_completion_failure: BaseException | None = None

        async def pending_recovery_result(
            status: ProviderOperationStatus,
        ) -> ProviderOperationRecoveryResult:
            await require_recovery_owner()
            rescheduled = await self._event_writer.emit(
                recovery_event(
                    EventType.PROVIDER_OPERATION_RECONNECT_SCHEDULED,
                    status=status.value,
                )
            )
            recovered_events.append(rescheduled)
            return ProviderOperationRecoveryResult(
                status=ProviderOperationRecoveryStatus.PENDING,
                events=tuple(recovered_events),
            )

        async def unavailable_recovery_result(
            reason: ProviderOperationUnavailableReason,
            status: ProviderOperationStatus | str,
            *,
            cleanup_failure: Exception | None = None,
        ) -> ProviderOperationRecoveryResult:
            await require_recovery_owner()
            status_value = status.value if isinstance(status, ProviderOperationStatus) else status
            required = await _emit_provider_recovery_required_event(
                self._event_writer,
                recovery_event(
                    EventType.PROVIDER_OPERATION_RECOVERY_REQUIRED,
                    status=status_value,
                    recovery_reason=reason,
                    cleanup_failure=cleanup_failure,
                ),
                recovery_reason=reason,
                cleanup_failure=cleanup_failure,
                redactor=self._secret_redactor,
            )
            recovered_events.append(required)
            return ProviderOperationRecoveryResult(
                status=ProviderOperationRecoveryStatus.UNAVAILABLE,
                events=tuple(recovered_events),
                unavailable_reason=reason,
            )

        def retain_post_completion_failure(failure: BaseException) -> None:
            nonlocal post_completion_failure
            post_completion_failure = _combine_post_completion_failures(
                post_completion_failure,
                failure,
            )

        def recovered_event_failure_reason(
            failure: Exception,
        ) -> ProviderOperationUnavailableReason:
            if isinstance(failure, ModelProviderError):
                return ProviderOperationUnavailableReason.FAILED
            return ProviderOperationUnavailableReason.MALFORMED

        async def accept_recovered_event(
            raw_event: object,
            *,
            persist_progress: bool,
        ) -> None:
            nonlocal completed_boundary, completed_event, completion_diagnostics
            nonlocal current_state, terminal_progress_verified
            if completed_event is not None:
                candidate = _validate_stream_event(
                    raw_event,
                    provider_name=registered_provider.name,
                    requested_model=session.model,
                    usage_dialect=registered_provider.usage_dialect,
                ).event
                if candidate == completed_event:
                    return
                raise RuntimeError("Recovered provider operation emitted output after completion.")
            boundary = _validate_stream_event(
                raw_event,
                provider_name=registered_provider.name,
                requested_model=session.model,
                usage_dialect=registered_provider.usage_dialect,
            )
            assistant_boundary = _validate_assistant_stream_event(
                boundary.event,
                generated_tool_call_id=_provider_operation_generated_tool_call_id(
                    stage,
                    boundary.event,
                ),
            )
            stream_event = assistant_boundary.event
            if (
                stream_event.type
                in {
                    ModelStreamEventType.THINKING,
                    ModelStreamEventType.TOOL_CALL,
                    ModelStreamEventType.HOSTED_TOOL_CALL,
                    ModelStreamEventType.CITATION,
                }
                and recovery_context is None
            ):
                kind = (
                    "thinking transcript policy"
                    if (stream_event.type is ModelStreamEventType.THINKING)
                    else (
                        "a tool continuation"
                        if stream_event.type is ModelStreamEventType.TOOL_CALL
                        else "hosted execution evidence"
                    )
                )
                raise ProviderOperationEvidenceError(
                    f"Legacy provider-operation evidence cannot safely reconstruct {kind}."
                )
            recovered_provider_error = (
                model_provider_error_from_payload(
                    stream_event.payload,
                    fallback_provider=registered_provider.name,
                )
                if stream_event.type is ModelStreamEventType.ERROR
                else None
            )
            if persist_progress and stream_event.type is not ModelStreamEventType.COMPLETED:
                runtime_event = None
                if stream_event.type in {
                    ModelStreamEventType.TEXT_DELTA,
                    ModelStreamEventType.ERROR,
                    ModelStreamEventType.HOSTED_TOOL_CALL,
                    ModelStreamEventType.CITATION,
                } or (
                    stream_event.type is ModelStreamEventType.THINKING and bool(stream_event.delta)
                ):
                    runtime_event = _model_stream_event_to_runtime_event(
                        stream_event,
                        session=session,
                        registered_agent=registered_agent,
                        environment_name=environment_name,
                        provider_name=registered_provider.name,
                        step=operation.step,
                        attempt=operation.attempt,
                        max_attempts=operation.max_attempts,
                        model_attempt_identity=operation.model_attempt_identity,
                        usage_dialect=registered_provider.usage_dialect,
                        execution_profile_fingerprint=(
                            None
                            if recovery_context is None
                            else recovery_context.execution_profile_fingerprint
                        ),
                    )
                progress, emitted = await self._commit_provider_operation_stream_event(
                    stage=stage,
                    state=current_state,
                    stream_event=stream_event,
                    runtime_event=runtime_event,
                    session=session,
                    interaction_id=operation.interaction_id,
                    registered_agent=registered_agent,
                    registered_provider=registered_provider,
                    environment_name=environment_name,
                    step=operation.step,
                    attempt=operation.attempt,
                    max_attempts=operation.max_attempts,
                    model_attempt_identity=operation.model_attempt_identity,
                )
                current_state = progress.state
                if progress.replayed:
                    return
                if emitted is not None:
                    recovered_events.append(emitted)
            if stream_event.type is ModelStreamEventType.TEXT_DELTA:
                transcript_helpers.append_assistant_text_delta(
                    assistant_parts,
                    stream_event.delta,
                )
            elif stream_event.type is ModelStreamEventType.THINKING:
                assert recovery_context is not None
                transcript_helpers.append_assistant_thinking_delta(
                    assistant_parts,
                    stream_event.delta,
                    provider_state=stream_event.payload.get("provider_state"),
                    include=(
                        recovery_context.thinking.include_in_transcript
                        if recovery_context.thinking is not None
                        else True
                    ),
                )
            elif stream_event.type is ModelStreamEventType.TOOL_CALL:
                assert recovery_context is not None
                tool_call = assistant_boundary.tool_call
                tool_call_part = assistant_boundary.tool_call_part
                if tool_call is None or tool_call_part is None:  # pragma: no cover - helper owns it
                    raise AssertionError("Validated tool-call projection disappeared.")
                tool_calls.append(tool_call)
                assistant_parts.append(tool_call_part)
            elif stream_event.type is ModelStreamEventType.HOSTED_TOOL_CALL:
                hosted_part = _hosted_tool_call_part(
                    stream_event,
                    provider_name=registered_provider.name,
                    model=session.model,
                    model_attempt_identity=operation.model_attempt_identity,
                )
                if hosted_part is not None:
                    assistant_parts.append(hosted_part)
            elif stream_event.type is ModelStreamEventType.CITATION:
                assistant_parts.append(
                    _citation_part(
                        stream_event,
                        provider_name=registered_provider.name,
                        model_attempt_identity=operation.model_attempt_identity,
                        assistant_parts=assistant_parts,
                    )
                )
            elif stream_event.type is ModelStreamEventType.ERROR:
                if stream_event.provider_operation_status is not None:
                    if not isinstance(recovered_provider_error, ModelProviderError):
                        raise RuntimeError(
                            "Provider-operation status requires a typed provider error."
                        )
                    raise _ProviderOperationStreamStatusError(
                        stream_event.provider_operation_status,
                        recovered_provider_error,
                    ) from recovered_provider_error
                raise recovered_provider_error or RuntimeError(
                    "Recovered provider operation failed."
                )
            elif stream_event.type is ModelStreamEventType.COMPLETED:
                completed_boundary = boundary
                completed_event = stream_event
                if persist_progress:
                    envelope = provider_operation_progress_envelope(
                        current_state,
                        stream_event,
                    )
                    if _provider_operation_progress_contains_secret(
                        envelope,
                        redactor=self._secret_redactor,
                    ):
                        retain_post_completion_failure(
                            ProviderOperationEvidenceError(
                                "Provider-operation terminal recovery state contains a workload "
                                "secret."
                            )
                        )
                    else:
                        terminal_cursor = envelope.recovery_metadata.cursor
                        current_cursor = current_state.recovery_metadata.cursor
                        current_cursor = -1 if current_cursor is None else current_cursor
                        if terminal_cursor != current_cursor + 1:
                            retain_post_completion_failure(
                                ProviderOperationEvidenceError(
                                    "Provider-operation terminal cursor is not the next boundary."
                                )
                            )
                        else:
                            terminal_progress_verified = True
                if boundary.completion_error is not None:
                    code, path = safe_durable_value_error_details(boundary.completion_error)
                    completion_error = ModelProviderError(
                        "Recovered provider operation returned invalid completion metadata.",
                        provider=registered_provider.name,
                        error_type="DurableValueError",
                        error_code="invalid_model_completion_value",
                        retryable=False,
                    )
                    completion_diagnostics = {
                        "completion_outcome": "invalid_metadata",
                        "completion_error": {
                            "error": str(completion_error),
                            "error_type": type(completion_error).__name__,
                            "durable_value_error_code": code,
                            "durable_value_path": path,
                            **completion_error.error_payload_fields(),
                        },
                    }
                    retain_post_completion_failure(completion_error)
            else:  # pragma: no cover - boundary validation owns the closed event vocabulary
                raise RuntimeError("Recovered provider operation returned an unsupported event.")

        for accepted_event in operation.accepted_stream_events:
            if (
                accepted_event.type is ModelStreamEventType.ERROR
                and accepted_event.provider_operation_status
                in {
                    ProviderOperationStatus.QUEUED,
                    ProviderOperationStatus.IN_PROGRESS,
                }
            ):
                # The error evidence and cursor are already durable. A
                # nonterminal provider status means recovery should continue
                # after that boundary rather than replaying the model error.
                continue
            await accept_recovered_event(accepted_event, persist_progress=False)

        await require_recovery_owner()
        if operation.accepted_stream_events:
            try:
                raw_connection = await adapter.reconnect(
                    copy_provider_operation_state(operation.state)
                )
            except BaseException as recovery_failure:
                recovery_failure = _classify_provider_recovery_failure(
                    recovery_failure,
                    cancellation_baseline=recovery_cancellation_baseline,
                    operation="Provider operation reconnect",
                )
                if not isinstance(recovery_failure, Exception):
                    raise recovery_failure
                malformed = isinstance(recovery_failure, ProviderOperationMalformedError)
                return await unavailable_recovery_result(
                    (
                        ProviderOperationUnavailableReason.MALFORMED
                        if malformed
                        else ProviderOperationUnavailableReason.UNAVAILABLE
                    ),
                    "malformed" if malformed else ProviderOperationStatus.UNAVAILABLE,
                )
            try:
                connection = copy_provider_operation_connection(raw_connection)
            except Exception as malformed_failure:
                cleanup_failure = None
                if type(raw_connection) is ProviderOperationConnection:
                    cleanup_failure = await _close_provider_recovery_iterator(
                        raw_connection.events,
                        cancellation_baseline=recovery_cancellation_baseline,
                        operation="Provider operation malformed reconnect stream cleanup",
                    )
                _raise_pending_provider_recovery_cancellation(
                    cancellation_baseline=recovery_cancellation_baseline
                )
                if cleanup_failure is not None:
                    add_exception_note_safely(
                        malformed_failure,
                        "Provider recovery stream cleanup also failed with "
                        f"{type(cleanup_failure).__name__}.",
                    )
                return await unavailable_recovery_result(
                    ProviderOperationUnavailableReason.MALFORMED,
                    "malformed",
                    cleanup_failure=cleanup_failure,
                )
            recovery_task = asyncio.current_task()
            if (
                recovery_task is not None
                and recovery_task.cancelling() > recovery_cancellation_baseline
            ):
                await _close_provider_recovery_iterator(
                    connection.events,
                    cancellation_baseline=recovery_cancellation_baseline,
                    operation=(
                        "Provider operation reconnect stream cleanup after suppressed caller "
                        "cancellation"
                    ),
                )
                raise AssertionError(
                    "Provider reconnect cleanup returned with caller cancellation pending."
                )
            reconnect_unavailable: (
                tuple[
                    ProviderOperationUnavailableReason,
                    ProviderOperationStatus | str,
                ]
                | None
            ) = None
            reconnect_pending_status: ProviderOperationStatus | None = None
            reconnect_cleanup_failure: Exception | None = None
            try:
                async with aclosing_provider_stream(connection.events) as reconnect_events:
                    if connection.state != operation.state:
                        reconnect_unavailable = (
                            ProviderOperationUnavailableReason.WRONG_PROVIDER,
                            "wrong_provider",
                        )
                    else:
                        recovery_status = connection.status
                        async for raw_event in reconnect_events:
                            await require_recovery_owner()
                            try:
                                await accept_recovered_event(raw_event, persist_progress=True)
                            except _ProviderOperationStreamStatusError as status_failure:
                                if completed_event is None:
                                    if status_failure.status in {
                                        ProviderOperationStatus.QUEUED,
                                        ProviderOperationStatus.IN_PROGRESS,
                                    }:
                                        reconnect_pending_status = status_failure.status
                                    else:
                                        reason = provider_operation_unavailable_reason(
                                            status_failure.status
                                        )
                                        if reason is None:
                                            reason = ProviderOperationUnavailableReason.MALFORMED
                                        reconnect_unavailable = (
                                            reason,
                                            status_failure.status,
                                        )
                                else:
                                    retain_post_completion_failure(status_failure.provider_error)
                                break
                            except Exception as event_failure:
                                if completed_event is None:
                                    reason = recovered_event_failure_reason(event_failure)
                                    reconnect_unavailable = (reason, reason.value)
                                else:
                                    retain_post_completion_failure(event_failure)
                                break
                            if completed_event is None:
                                continue
                            break
            except BaseException as stream_failure:
                stream_failure = _classify_provider_recovery_failure(
                    stream_failure,
                    cancellation_baseline=recovery_cancellation_baseline,
                    operation="Provider operation recovery stream",
                )
                if completed_event is None:
                    if isinstance(stream_failure, Exception):
                        if reconnect_unavailable is None:
                            malformed = isinstance(stream_failure, ProviderOperationMalformedError)
                            reconnect_unavailable = (
                                (
                                    ProviderOperationUnavailableReason.MALFORMED
                                    if malformed
                                    else ProviderOperationUnavailableReason.UNAVAILABLE
                                ),
                                ("malformed" if malformed else ProviderOperationStatus.UNAVAILABLE),
                            )
                        else:
                            reconnect_cleanup_failure = stream_failure
                    else:
                        raise
                else:
                    retain_post_completion_failure(stream_failure)
            if reconnect_unavailable is not None and completed_event is None:
                return await unavailable_recovery_result(
                    *reconnect_unavailable,
                    cleanup_failure=reconnect_cleanup_failure,
                )
            if reconnect_pending_status is not None and completed_event is None:
                return await pending_recovery_result(reconnect_pending_status)
        else:
            try:
                raw_snapshot = await adapter.retrieve(
                    copy_provider_operation_state(operation.state)
                )
                _raise_pending_provider_recovery_cancellation(
                    cancellation_baseline=recovery_cancellation_baseline
                )
            except BaseException as recovery_failure:
                recovery_failure = _classify_provider_recovery_failure(
                    recovery_failure,
                    cancellation_baseline=recovery_cancellation_baseline,
                    operation="Provider operation retrieval",
                )
                if not isinstance(recovery_failure, Exception):
                    raise recovery_failure
                malformed = isinstance(recovery_failure, ProviderOperationMalformedError)
                return await unavailable_recovery_result(
                    (
                        ProviderOperationUnavailableReason.MALFORMED
                        if malformed
                        else ProviderOperationUnavailableReason.UNAVAILABLE
                    ),
                    "malformed" if malformed else ProviderOperationStatus.UNAVAILABLE,
                )
            try:
                snapshot = copy_provider_operation_snapshot(raw_snapshot)
            except Exception:
                return await unavailable_recovery_result(
                    ProviderOperationUnavailableReason.MALFORMED,
                    "malformed",
                )
            if snapshot.state != operation.state:
                return await unavailable_recovery_result(
                    ProviderOperationUnavailableReason.WRONG_PROVIDER,
                    "wrong_provider",
                )
            recovery_status = snapshot.status
            if snapshot.status in {
                ProviderOperationStatus.QUEUED,
                ProviderOperationStatus.IN_PROGRESS,
            }:
                return await pending_recovery_result(snapshot.status)
            for raw_event in snapshot.events:
                try:
                    await accept_recovered_event(raw_event, persist_progress=False)
                except _ProviderOperationStreamStatusError as status_failure:
                    if completed_event is None:
                        if status_failure.status in {
                            ProviderOperationStatus.QUEUED,
                            ProviderOperationStatus.IN_PROGRESS,
                        }:
                            return await pending_recovery_result(status_failure.status)
                        reason = provider_operation_unavailable_reason(status_failure.status)
                        if reason is None:
                            reason = ProviderOperationUnavailableReason.MALFORMED
                        return await unavailable_recovery_result(
                            reason,
                            status_failure.status,
                        )
                    retain_post_completion_failure(status_failure.provider_error)
                    break
                except Exception as snapshot_failure:
                    if completed_event is None:
                        reason = recovered_event_failure_reason(snapshot_failure)
                        return await unavailable_recovery_result(reason, reason.value)
                    retain_post_completion_failure(snapshot_failure)
                    break

        try:
            await require_recovery_owner()
        except (SessionInterruptedByRequest, asyncio.CancelledError) as recovery_failure:
            if completed_event is None:
                raise
            post_completion_failure = _combine_post_completion_failures(
                post_completion_failure,
                recovery_failure,
            )
        if completed_event is None or completed_boundary is None:
            if recovery_status in {
                ProviderOperationStatus.QUEUED,
                ProviderOperationStatus.IN_PROGRESS,
            }:
                return await pending_recovery_result(recovery_status)
            unavailable_reason = provider_operation_unavailable_reason(recovery_status)
            if unavailable_reason is None:
                raise RuntimeError("Provider operation returned an unknown terminal status.")
            return await unavailable_recovery_result(unavailable_reason, recovery_status)
        if recovery_status not in {
            ProviderOperationStatus.QUEUED,
            ProviderOperationStatus.IN_PROGRESS,
            ProviderOperationStatus.COMPLETED,
        }:
            retain_post_completion_failure(
                RuntimeError(
                    "Provider operation returned completion output with conflicting terminal "
                    f"status: {recovery_status.value}."
                )
            )

        completion_semantics_valid = completed_boundary.completion_error is None
        billing_identity = None if recovery_context is None else recovery_context.billing_identity
        if completion_semantics_valid:
            try:
                billing_identity = resolve_completion_billing_identity(
                    provider,
                    billing_identity,
                    copy_durable_json_object(completed_event.payload, "completed_payload"),
                    provider_name=registered_provider.name,
                )
            except ModelProviderError as billing_error:
                completion_semantics_valid = False
                completion_diagnostics = {
                    "completion_outcome": "billing_identity_resolution_failed",
                    "completion_error": {
                        "error": str(billing_error),
                        "error_type": type(billing_error).__name__,
                        "stage": "billing_identity_for_completion",
                        **billing_error.error_payload_fields(),
                    },
                }
                retain_post_completion_failure(billing_error)

        assistant_message: Message | None = None
        step_result: AssistantStepResult | None = None
        classification = None
        if completion_semantics_valid:
            try:
                _require_unique_tool_call_ids(tool_calls)
                provider_state_parts = transcript_helpers.provider_state_parts(
                    completed_event.payload
                )
                assistant_message = transcript_helpers.assistant_message(
                    content_parts=assistant_parts,
                    provider_state_parts=provider_state_parts,
                )
                step_result = _assistant_step_result(
                    session_id=session.id,
                    step=operation.step,
                    model_attempt_identity=operation.model_attempt_identity,
                    assistant_message=assistant_message,
                    tool_calls=tool_calls,
                    completion=_stream_event_completion(completed_event),
                )
                classification = classify_assistant_step(step_result)
            except (TypeError, ValueError):
                completion_semantics_valid = False
                transcript_error = ModelProviderError(
                    "Recovered provider operation returned invalid completion transcript state.",
                    provider=registered_provider.name,
                    error_type="ValueError",
                    error_code="invalid_model_completion_transcript",
                    retryable=False,
                )
                completion_diagnostics = {
                    "completion_outcome": "invalid_transcript_state",
                    "completion_error": {
                        "error": str(transcript_error),
                        "error_type": type(transcript_error).__name__,
                        "stage": "completion_transcript_projection",
                        **transcript_error.error_payload_fields(),
                    },
                }
                retain_post_completion_failure(transcript_error)

        completion_event = _model_stream_event_to_runtime_event(
            completed_event,
            session=session,
            registered_agent=registered_agent,
            environment_name=environment_name,
            provider_name=registered_provider.name,
            step=operation.step,
            attempt=operation.attempt,
            max_attempts=operation.max_attempts,
            model_attempt_identity=operation.model_attempt_identity,
            tool_round_identity=(None if step_result is None else step_result.tool_round_identity),
            classification=None if classification is None else classification.payload(),
            transcript_cursor_after_completion=(
                stage.source_transcript_cursor
                + int(assistant_message is not None and not tool_calls)
            ),
            usage_dialect=registered_provider.usage_dialect,
            billing_identity=billing_identity,
            accounting_usage_metrics=completed_boundary.accounting_usage_metrics,
            accounting_usage_rejected=completed_boundary.accounting_usage_rejected,
            usage_normalization_failed=completed_boundary.usage_normalization_failed,
            completion_diagnostics=completion_diagnostics,
            execution_profile_fingerprint=(
                None if recovery_context is None else recovery_context.execution_profile_fingerprint
            ),
        )
        completion_event = completion_event.model_copy(
            update={"interaction_id": operation.interaction_id},
            deep=True,
        )
        if terminal_progress_verified:
            completion_event = self._provider_operation_progress_event(
                stage=stage,
                state=current_state,
                stream_event=completed_event,
                runtime_event=completion_event,
                session=session,
                interaction_id=operation.interaction_id,
                registered_agent=registered_agent,
                registered_provider=registered_provider,
                environment_name=environment_name,
                step=operation.step,
                attempt=operation.attempt,
                max_attempts=operation.max_attempts,
                model_attempt_identity=operation.model_attempt_identity,
            )
        publication_cancellation = _take_model_completion_cancellation(
            post_completion_failure,
            cancellation_baseline=recovery_cancellation_baseline,
        )
        if publication_cancellation is not None and post_completion_failure is None:
            post_completion_failure = publication_cancellation
        elif publication_cancellation is None and isinstance(
            post_completion_failure,
            asyncio.CancelledError,
        ):
            post_completion_failure = unexpected_child_cancellation_error(
                post_completion_failure,
                operation="Provider operation recovery stream",
            )
        durable_step_result = None
        if step_result is not None:
            try:
                durable_step_result = _durable_assistant_step_result(
                    step_result,
                    redactor=self._secret_redactor,
                )
            except (TypeError, ValueError):
                durable_boundary_error = ModelProviderError(
                    "Recovered provider operation returned assistant output that cannot cross "
                    "the durable publication boundary.",
                    provider=registered_provider.name,
                    error_type="DurableBoundaryError",
                    error_code="invalid_model_completion_transcript",
                    retryable=False,
                )
                retain_post_completion_failure(durable_boundary_error)
        structured_output_validation = None
        if (
            post_completion_failure is None
            and recovery_context is not None
            and recovery_context.structured_output is not None
            and recovery_context.structured_output.strategy is StructuredOutputStrategy.TOOL
            and any(call.name == STRUCTURED_OUTPUT_TOOL_NAME for call in tool_calls)
        ):
            try:
                structured_output_validation = _redact_structured_output_validation(
                    _validate_structured_output_tool_round(
                        tool_calls=tool_calls,
                        spec=recovery_context.structured_output,
                    ),
                    self._secret_redactor,
                )
            except (TypeError, ValueError):
                structured_output_error = ModelProviderError(
                    "Recovered provider operation returned invalid structured output.",
                    provider=registered_provider.name,
                    error_type="ValueError",
                    error_code="invalid_model_completion_transcript",
                    retryable=False,
                )
                retain_post_completion_failure(structured_output_error)
        request_fingerprint = stage.intent.get("request_fingerprint")
        if type(request_fingerprint) is not str:
            raise RuntimeError("Provider-operation stage lost its request fingerprint.")
        authoritative_assistant_message = (
            durable_step_result.assistant_message
            if durable_step_result is not None and post_completion_failure is None
            else None
        )
        publication_event = (
            completion_event
            if post_completion_failure is None
            else _non_turn_model_completion_event(
                completion_event,
                failure=post_completion_failure,
                cancellation=publication_cancellation,
                transcript_cursor=stage.source_transcript_cursor,
            )
        )
        if stage.reservation_ids:
            if recovery_context is None:
                raise ProviderOperationEvidenceError(
                    "Budgeted provider-operation recovery has no durable accounting context."
                )
            try:
                publication_event = (
                    await self._run_limit_controller.recover_model_completion_budget_evidence(
                        publication_event,
                        reservation_ids=stage.reservation_ids,
                        recovery_contexts=recovery_context.budget_reservations,
                        session=session,
                        provider_name=registered_provider.name,
                        model_attempt_identity=operation.model_attempt_identity,
                        dispatch_id=stage.stage_id,
                        request_billing_identity=recovery_context.billing_identity,
                    )
                )
            except (KeyError, NotImplementedError, TypeError, ValueError) as accounting_error:
                raise ProviderOperationEvidenceError(
                    "Provider-operation recovery could not reconstruct its original budget "
                    "reservation and pricing context."
                ) from accounting_error
        tool_exposure = None if recovery_context is None else recovery_context.tool_exposure
        if (
            durable_step_result is not None
            and durable_step_result.tool_calls
            and tool_exposure is None
        ):
            raise ProviderOperationEvidenceError(
                "Provider-operation recovery has no durable tool-exposure authority."
            )
        publication_event = self._event_writer.prepare(publication_event)
        publication = ModelCompletionPublicationRequest(
            dispatch=ModelCompletionDispatch(
                stage=stage,
                request_fingerprint=request_fingerprint,
            ),
            assistant_step_result=durable_step_result,
            completion_event=publication_event,
            authoritative_assistant_message=authoritative_assistant_message,
            defer_assistant_message=bool(
                durable_step_result is not None
                and durable_step_result.tool_calls
                and post_completion_failure is None
            ),
            structured_output_validation=structured_output_validation,
            tool_exposure=tool_exposure,
        )
        await _publish_model_completion(
            model_completion_publisher,
            publication,
            terminal_failure=post_completion_failure,
            publication_cancellation=publication_cancellation,
        )
        if post_completion_failure is not None:
            raise post_completion_failure
        reconciled = recovery_event(
            EventType.PROVIDER_OPERATION_RECONCILED,
            status=recovery_status.value,
        )
        try:
            reconciled = await self._event_writer.emit(reconciled)
        except Exception as delivery_error:
            logger.warning(
                "Provider operation completed durably but reconciliation telemetry failed: "
                "session_id=%s operation_id=%s error_type=%s",
                session.id,
                operation.state.operation_id,
                type(delivery_error).__name__,
            )
        recovered_events.append(reconciled)
        if recovery_under_cancellation_claim:
            if stage.reservation_ids:
                await self._run_limit_controller.reconcile_model_completion_settlements(
                    publication_event,
                    reservation_ids=stage.reservation_ids,
                )
            await self._release_provider_operation_cancellation_claim(
                session=session,
                claim=cancellation_claim,
            )
        return ProviderOperationRecoveryResult(
            status=ProviderOperationRecoveryStatus.RECONCILED,
            events=tuple(recovered_events),
            completion_event=publication_event,
        )

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
        knowledge_access_scope: Any,
        request_metadata: dict[str, Any],
        retry_policy: RetryPolicy,
        request_budget_limits: tuple[BudgetLimit, ...],
        limit_gate: RunLimitGate,
        budget_policy: BudgetPolicy | None,
        run_started_at: float,
        turn_usage_tracker: SessionUsageTracker | None,
        active_run: ActiveSessionRun[SessionUsageTracker] | None,
        execution_profile: ExecutionProfileIdentity | None = None,
        validate_live_model_semantics: Callable[[], None],
        initial_tool_exposure: ResolvedToolExposure | None = None,
        previous_tool_exposure_profile_id: str | None = None,
        model_completion_recovery_context_factory: (
            ModelCompletionRecoveryContextFactory | None
        ) = None,
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
            knowledge_access_scope=knowledge_access_scope,
            request_metadata=request_metadata,
            retry_policy=retry_policy,
            request_budget_limits=request_budget_limits,
            limit_gate=limit_gate,
            budget_policy=budget_policy,
            run_started_at=run_started_at,
            turn_usage_tracker=turn_usage_tracker,
            active_run=active_run,
            execution_profile=execution_profile,
            validate_live_model_semantics=validate_live_model_semantics,
            initial_tool_exposure=initial_tool_exposure,
            previous_tool_exposure_profile_id=previous_tool_exposure_profile_id,
            model_completion_recovery_context_factory=(
                model_completion_recovery_context_factory
                or (
                    lambda billing_identity, _reservations: ModelCompletionRecoveryContext(
                        billing_identity=billing_identity
                    )
                )
            ),
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
        tool_exposure: ResolvedToolExposure | None = None,
    ) -> ModelRequest:
        resolved_tool_exposure = (
            _all_registered_tool_exposure(registered_agent)
            if tool_exposure is None
            else _require_frozen_tool_exposure(tool_exposure)
        )
        model_tools = _model_request_tools(
            tool_exposure=resolved_tool_exposure,
            structured_output=structured_output,
        )
        model_messages = _model_request_messages(
            messages=context_messages,
            structured_output=structured_output,
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
            hosted_tools=registered_agent.hosted_tools,
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
        request_variant: RequestVariant = RequestVariant.INITIAL,
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
        validate_live_model_semantics: Callable[[], None],
        record_model_attempt_identity: Callable[[ModelAttemptIdentity], None],
        billing_identity: BillingIdentity | None = None,
        structured_output: StructuredOutputSpec | None = None,
        prepare_model_completion_dispatch: Callable[
            [ModelRequest],
            Awaitable[ModelCompletionDispatch],
        ]
        | None = None,
        model_completion_publisher: ModelCompletionPublisher | None = None,
        execution_profile: ExecutionProfileIdentity | None = None,
        tool_exposure: ResolvedToolExposure | None = None,
        tool_exposure_evidence: ToolExposure | None = None,
    ) -> AsyncIterator[tuple[Event | None, AssistantStepResult | None]]:
        retry_policy = copy_retry_policy(retry_policy)
        request_variant = RequestVariant(request_variant)
        structured_output = copy_structured_output_spec(structured_output)
        model_step_identity = copy_model_step_identity(model_step_identity)
        resolved_tool_exposure = (
            None if tool_exposure is None else _require_frozen_tool_exposure(tool_exposure)
        )
        if tool_exposure_evidence is not None:
            if type(tool_exposure_evidence) is not ToolExposure:
                raise TypeError("tool_exposure_evidence must be a ToolExposure or None.")
            tool_exposure_evidence = ToolExposure.model_validate(
                tool_exposure_evidence.model_dump(mode="python")
            )
            if resolved_tool_exposure is None:
                raise ValueError("Tool exposure evidence requires a frozen tool exposure.")
            if execution_profile is None:
                raise ValueError("Tool exposure evidence requires an execution profile.")
            expected_tool_exposure_evidence = tool_exposure_record(
                resolved_tool_exposure,
                profile_changed=tool_exposure_evidence.profile_changed,
                step=step,
                provider_name=registered_provider.name,
                model=model_request.model,
                model_step_id=model_step_identity.model_step_id,
                execution_profile_fingerprint=execution_profile.fingerprint,
            )
            if tool_exposure_evidence != expected_tool_exposure_evidence:
                raise ValueError(
                    "Tool exposure evidence does not match the frozen model request authority."
                )
        provider.preflight_hosted_tools(
            model=model_request.model,
            hosted_tools=model_request.hosted_tools,
            options=model_request.options,
        )
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
        prompt_contribution_manifest = (
            await self._load_prompt_contribution_manifest(session.id)
            if self._request_footprint.enabled
            else None
        )
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

            # Reservation/retry preparation can yield. Recheck before request
            # footprint, pressure, token-count, and provider-start evidence are
            # attributed to the frozen invocation profile.
            validate_live_model_semantics()
            # Never hand the retry template to provider-controlled code. Each
            # attempt gets a fully detached, revalidated request so provider
            # mutation cannot corrupt a later attempt.
            attempt_model_request = _detach_model_request(model_request)

            request_footprint, request_footprint_event = await self._observe_request_footprint(
                model_request=attempt_model_request,
                session=session,
                registered_agent=registered_agent,
                registered_provider=registered_provider,
                environment_name=environment_name,
                step=step,
                attempt=attempt,
                max_attempts=retry_policy.max_attempts,
                request_variant=request_variant,
                model_attempt_identity=model_attempt_identity,
                prompt_contribution_manifest=prompt_contribution_manifest,
                structured_output=structured_output,
                execution_profile=execution_profile,
                tool_exposure=tool_exposure_evidence,
            )
            if request_footprint_event is not None:
                yield request_footprint_event, None
            request_context_pressure = (
                request_footprint.context_pressure
                if request_footprint is not None
                else analyze_request_context_pressure(
                    attempt_model_request,
                    provider=registered_provider.provider,
                )
            )

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
                estimate=request_context_pressure,
                execution_profile=execution_profile,
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
                execution_profile=execution_profile,
                validate_live_model_semantics=validate_live_model_semantics,
            )
            if context_count_event is not None:
                yield context_count_event, None
            yield (
                await self._event_writer.emit(
                    event_with_execution_profile_authority(
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
                        ),
                        execution_profile,
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
                retry_policy=retry_policy,
                model_attempt_identity=model_attempt_identity,
                transcript_cursor_before_request=transcript_cursor_before_request,
                record_model_completion=record_model_completion,
                before_provider_dispatch=before_provider_dispatch,
                validate_live_model_semantics=validate_live_model_semantics,
                billing_identity=billing_identity,
                structured_output=structured_output,
                context_pressure_estimate=request_context_pressure,
                prepare_model_completion_dispatch=prepare_model_completion_dispatch,
                model_completion_publisher=model_completion_publisher,
                execution_profile=execution_profile,
                tool_exposure=resolved_tool_exposure,
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
                                    event_with_execution_profile_authority(
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
                                        ),
                                        execution_profile,
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
                                    event_with_execution_profile_authority(
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
                                        ),
                                        execution_profile,
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
                (
                    status_code,
                    retryable,
                    retry_after_s,
                    unknown_provider_error,
                ) = _typed_retry_fields(exc)
                decision = exc.retry_decision
                if decision is None:
                    decision = retry_decision(
                        policy=retry_policy,
                        attempt=attempt,
                        error=exc.message,
                        status_code=status_code,
                        retryable=retryable,
                        retry_after_s=retry_after_s,
                        unknown_provider_error=unknown_provider_error,
                    )
                elif (
                    decision.attempt != attempt
                    or decision.max_attempts != retry_policy.max_attempts
                ):
                    raise RuntimeError(
                        "Model attempt retained a retry decision for different attempt authority."
                    ) from exc
                if decision.reason is not None and not exc.emitted_error_event:
                    yield (
                        await self._event_writer.emit(
                            event_with_execution_profile_authority(
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
                                        decision=decision,
                                    ),
                                ),
                                execution_profile,
                            )
                        ),
                        None,
                    )
                if not decision.retry:
                    _raise_terminal_model_attempt_failure(exc)
                yield (
                    await self._event_writer.emit(
                        event_with_execution_profile_authority(
                            _model_retry_event(
                                session=session,
                                registered_agent=registered_agent,
                                environment_name=environment_name,
                                registered_provider=registered_provider,
                                step=step,
                                decision=decision,
                                error=exc.message,
                                provider_error_payload=exc.payload,
                                model_attempt_identity=model_attempt_identity,
                            ),
                            execution_profile,
                        )
                    ),
                    None,
                )
                yield (
                    await self._event_writer.emit(
                        event_with_execution_profile_authority(
                            _model_attempt_discarded_event(
                                session=session,
                                registered_agent=registered_agent,
                                environment_name=environment_name,
                                registered_provider=registered_provider,
                                step=step,
                                decision=decision,
                                model_attempt_identity=model_attempt_identity,
                            ),
                            execution_profile,
                        )
                    ),
                    None,
                )
                await self._sleep_before_retry(session.id, decision)
                prior_retry_failure = exc
                attempt += 1
            finally:
                await _close_async_iterator(attempt_events)

    async def _observe_request_footprint(
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
        request_variant: RequestVariant,
        model_attempt_identity: ModelAttemptIdentity,
        prompt_contribution_manifest: PromptContributionManifest | None,
        structured_output: StructuredOutputSpec | None,
        execution_profile: ExecutionProfileIdentity | None,
        tool_exposure: ToolExposure | None,
    ) -> tuple[RequestFootprint | None, Event | None]:
        if not self._request_footprint.enabled:
            return None, None
        footprint = analyze_request_footprint(
            model_request,
            provider=registered_provider.provider,
            provider_name=registered_provider.name,
            step=step,
            attempt=attempt,
            max_attempts=max_attempts,
            request_variant=request_variant,
            observation_id=str(uuid4()),
            model_step_id=model_attempt_identity.model_step_id,
            model_attempt_id=model_attempt_identity.model_attempt_id,
            config=self._request_footprint,
            prompt_contribution_manifest=prompt_contribution_manifest,
            structured_output_instruction=(
                structured_output_tool_instruction(structured_output)
                if (
                    structured_output is not None
                    and structured_output.strategy == StructuredOutputStrategy.TOOL
                )
                else None
            ),
            execution_profile_fingerprint=(
                None if execution_profile is None else execution_profile.fingerprint
            ),
            tool_exposure=tool_exposure,
        )
        footprint_event = Event(
            type=EventType.REQUEST_FOOTPRINT_RECORDED,
            session_id=session.id,
            agent_name=registered_agent.spec.name,
            environment_name=environment_name,
            payload=footprint.model_dump(mode="json", exclude_none=True),
        )
        event = await self._event_writer.emit(
            event_with_execution_profile_authority(
                _context_observation_event(footprint_event),
                execution_profile,
            )
        )
        return footprint, event

    async def _load_prompt_contribution_manifest(
        self,
        session_id: str,
    ) -> PromptContributionManifest | None:
        records = await self._session_store.query_events(
            EventQuery(
                session_id=session_id,
                event_types=(EventType.SESSION_STARTED,),
                order_by=EventOrder.SEQUENCE_ASC,
                limit=1,
            )
        )
        if not records:
            return None
        payload = records[0].event.payload.get("prompt_contribution_manifest")
        if payload is None:
            return None
        return PromptContributionManifest.model_validate(payload)

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
        estimate: ContextPressureEstimate | None = None,
        execution_profile: ExecutionProfileIdentity | None = None,
    ) -> tuple[_ContextPressureObservation | None, Event | None]:
        if self._context_counting.mode == ContextCountingMode.OFF:
            return None, None
        observation_id = str(uuid4())
        if estimate is None:
            estimate = analyze_request_context_pressure(
                model_request,
                provider=registered_provider.provider,
            )
        observation = _ContextPressureObservation(
            estimate=estimate,
            observation_id=observation_id,
        )
        event = await self._event_writer.emit(
            event_with_execution_profile_authority(
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
                ),
                execution_profile,
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
        validate_live_model_semantics: Callable[[], None],
        execution_profile: ExecutionProfileIdentity | None = None,
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
        count_request = _copy_model_request_for_counting(model_request)
        # Event publication above the caller can yield to application code.
        # Recheck at the exact remote counter seam and keep an authority
        # mismatch outside the optional-counter failure projection below.
        validate_live_model_semantics()
        try:
            provider_result = await provider.count_input_tokens(count_request)
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
                event_with_execution_profile_authority(
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
                    ),
                    execution_profile,
                )
            )
            return None, event

        observation = _ContextCountObservation(
            result=result,
            observation_id=observation_id,
        )
        event = await self._event_writer.emit(
            event_with_execution_profile_authority(
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
                ),
                execution_profile,
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
        retry_policy: RetryPolicy,
        model_attempt_identity: ModelAttemptIdentity,
        transcript_cursor_before_request: int,
        record_model_completion: Callable[[Event], Event],
        before_provider_dispatch: Callable[[ModelAttemptIdentity], Awaitable[None]],
        validate_live_model_semantics: Callable[[], None],
        billing_identity: BillingIdentity | None,
        structured_output: StructuredOutputSpec | None,
        context_pressure_estimate: ContextPressureEstimate | None,
        prepare_model_completion_dispatch: Callable[
            [ModelRequest],
            Awaitable[ModelCompletionDispatch],
        ]
        | None,
        model_completion_publisher: ModelCompletionPublisher | None,
        execution_profile: ExecutionProfileIdentity | None,
        tool_exposure: ResolvedToolExposure | None,
    ) -> AsyncIterator[tuple[Event | None, AssistantStepResult | None]]:
        retry_policy = copy_retry_policy(retry_policy)
        if retry_policy.max_attempts != max_attempts:
            raise ValueError("Retry policy does not match the model-attempt ceiling.")
        assistant_parts: list[transcript_helpers.AssistantContentPart] = []
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
        # Request analysis invokes provider-owned projection hooks.  A mutable
        # built-in adapter must still match the profile admitted for this
        # invocation before any of those semantics are consulted.
        validate_live_model_semantics()
        context_pressure_estimate = copy_context_pressure_estimate(context_pressure_estimate)
        if context_pressure_estimate is None:
            context_pressure_estimate = analyze_request_context_pressure(
                model_request,
                provider=registered_provider.provider,
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
        provider_operation_adapter: ProviderOperationAdapter | None = None
        provider_operation_state: ProviderOperationState | None = None
        provider_operation_interaction_id: str | None = None
        provider_operation_identity_durable = False
        provider_exhausted = False
        background_dispatch_invoked = False
        durable_stream_failure: ModelAttemptFailed | None = None
        provider_control_failure: ModelProviderError | None = None
        provider_control_error_emitted = False
        post_completion_failure: BaseException | None = None

        async def reconcile_completion_that_won_cancellation(
            snapshot: ProviderOperationSnapshot | None,
        ) -> None:
            if snapshot is None or snapshot.status is not ProviderOperationStatus.COMPLETED:
                return
            if completion_dispatch is None or model_completion_publisher is None:
                raise RuntimeError(
                    "Provider completion won cancellation without a durable publication stage."
                )
            recoverable = await load_recoverable_provider_operation(
                self._session_store,
                completion_dispatch.stage,
            )
            if recoverable is None:
                raise ProviderOperationEvidenceError(
                    "Provider completion won cancellation without recoverable operation evidence."
                )
            recovered = await self.recover_provider_operation(
                session=session,
                stage=completion_dispatch.stage,
                operation=recoverable,
                registered_agent=registered_agent,
                registered_provider=registered_provider,
                environment_name=environment_name,
                recovery_context=model_completion_recovery_context_from_stage(
                    completion_dispatch.stage
                ),
                model_completion_publisher=model_completion_publisher,
            )
            if recovered.status is not ProviderOperationRecoveryStatus.RECONCILED:
                raise ProviderOperationEvidenceError(
                    "Provider completion won cancellation but remained unreconciled."
                )

        try:
            validate_live_model_semantics()
            provider_operation_mode = provider.provider_operation_mode
            if type(provider_operation_mode) is not ProviderOperationMode:
                raise TypeError(
                    "ModelProvider.provider_operation_mode must return a ProviderOperationMode."
                )
            if provider_operation_mode is ProviderOperationMode.SYNCHRONOUS:
                provider_events = provider.stream(model_request)
            else:
                provider_operation_adapter = provider.provider_operations
                if not isinstance(provider_operation_adapter, ProviderOperationAdapter):
                    raise RuntimeError(
                        "Background provider-operation mode requires a ProviderOperationAdapter."
                    )
                start_idempotency_support = provider_operation_adapter.start_idempotency_support
                if type(start_idempotency_support) is not ProviderOperationStartIdempotencySupport:
                    raise TypeError(
                        "ProviderOperationAdapter.start_idempotency_support must return "
                        "ProviderOperationStartIdempotencySupport."
                    )
                start_id = f"provider-operation:{model_attempt_identity.model_attempt_id}"
                if completion_dispatch is None:
                    raise RuntimeError(
                        "Background provider operations require a durable model-completion stage."
                    )
                staged_start = completion_dispatch.stage.intent.get("provider_operation_start")
                if (
                    type(staged_start) is not dict
                    or staged_start.get("schema_version") != 1
                    or staged_start.get("idempotency_key") != start_id
                    or staged_start.get("idempotency_support") != start_idempotency_support.value
                ):
                    raise RuntimeError(
                        "Provider-operation start contract changed after durable staging."
                    )
                starting_event = _event_with_model_identity_authority(
                    Event(
                        type=EventType.PROVIDER_OPERATION_STARTING,
                        session_id=session.id,
                        agent_name=registered_agent.spec.name,
                        environment_name=environment_name,
                        payload={
                            "provider": registered_provider.name,
                            "model": session.model,
                            "step": step,
                            "attempt": attempt,
                            "max_attempts": max_attempts,
                            **model_attempt_identity.payload(),
                            "source_run_epoch": session.run_epoch,
                            "start_id": start_id,
                            "start_idempotency_support": start_idempotency_support.value,
                        },
                    ),
                    model_attempt_identity,
                )
                starting_event = event_with_execution_profile_authority(
                    starting_event,
                    execution_profile,
                )
                starting_event = event_with_runtime_payload_authority(
                    starting_event,
                    "start_id",
                )
                emitted_starting_event = await self._event_writer.emit(starting_event)
                yield emitted_starting_event, None
                provider_operation_interaction_id = emitted_starting_event.interaction_id
                if provider_operation_interaction_id is None:
                    raise RuntimeError(
                        "Provider-operation dispatch requires an owning interaction."
                    )

                async def start_provider_operation() -> ProviderOperationConnection:
                    # Durable staging and event publication above can yield to
                    # application code. Recheck at the last pre-dispatch seam.
                    validate_live_model_semantics()
                    return await provider_operation_adapter.start(
                        ProviderOperationStartRequest(
                            request=model_request,
                            idempotency_key=start_id,
                        )
                    )

                start_task = asyncio.create_task(start_provider_operation())
                background_dispatch_invoked = True

                def operation_event_for(
                    operation_state: ProviderOperationState,
                    operation_status: ProviderOperationStatus,
                ) -> Event:
                    event = _event_with_model_identity_authority(
                        Event(
                            id=provider_operation_started_event_id(start_id),
                            type=EventType.PROVIDER_OPERATION_STARTED,
                            session_id=session.id,
                            interaction_id=emitted_starting_event.interaction_id,
                            agent_name=registered_agent.spec.name,
                            environment_name=environment_name,
                            payload={
                                "provider": registered_provider.name,
                                "model": session.model,
                                "step": step,
                                "attempt": attempt,
                                "max_attempts": max_attempts,
                                **model_attempt_identity.payload(),
                                "source_run_epoch": session.run_epoch,
                                "start_id": start_id,
                                "state_version": operation_state.version,
                                "operation_id": operation_state.operation_id,
                                "stream_protocol": operation_state.stream_protocol,
                                "status": operation_status.value,
                                "recovery_metadata": operation_state.recovery_metadata.model_dump(
                                    mode="json",
                                    exclude_none=True,
                                ),
                            },
                        ),
                        model_attempt_identity,
                    )
                    event = event_with_execution_profile_authority(
                        event,
                        execution_profile,
                    )
                    return event_with_runtime_payload_authority(event, "start_id")

                start_outcome = await await_shielded_task_outcome(
                    start_task,
                    timeout_after_cancellation_s=(
                        _PROVIDER_OPERATION_START_SETTLEMENT_TIMEOUT_SECONDS
                    ),
                )
                if start_outcome.timed_out:
                    start_task.cancel()
                    start_cancellation = start_outcome.cancellation
                    if start_cancellation is None:  # pragma: no cover - armed by cancellation
                        raise RuntimeError("Provider operation start settlement timed out.")

                async def reconcile_late_start() -> None:
                    late_outcome = await await_shielded_task_outcome(start_task)
                    if late_outcome.error is not None or late_outcome.result is None:
                        return
                    raw_late_operation = late_outcome.result
                    try:
                        late_operation = copy_provider_operation_connection(raw_late_operation)
                    except BaseException:
                        if type(raw_late_operation) is ProviderOperationConnection:
                            async with aclosing_provider_stream(raw_late_operation.events):
                                raise
                        raise
                    try:
                        (
                            _,
                            cancellation_snapshot,
                        ) = await _cancel_provider_operation_after_definite_absence(
                            adapter=provider_operation_adapter,
                            state=late_operation.state,
                            failure=RuntimeError(
                                "Caller cancellation preceded provider start acknowledgement."
                            ),
                        )
                        reconciled_status = (
                            late_operation.status
                            if cancellation_snapshot is None
                            else cancellation_snapshot.status
                        )
                        reconciliation_event = self._event_writer.prepare(
                            operation_event_for(
                                late_operation.state,
                                reconciled_status,
                            )
                        )

                        def preserve_checkpoint(
                            _current: Session,
                            checkpoint: dict[str, Any] | None,
                        ) -> dict[str, Any]:
                            if checkpoint is None:
                                raise RuntimeError(
                                    "Late provider-operation reconciliation requires an "
                                    "existing session checkpoint."
                                )
                            copied = copy_json_value(checkpoint, "checkpoint")
                            if type(copied) is not dict:
                                raise TypeError("Session checkpoint must be an object.")
                            return copied

                        for publication_attempt in range(2):
                            reconciliation_session = await self._session_store.load(session.id)
                            if reconciliation_session is None:
                                return
                            if reconciliation_session.run_epoch == session.run_epoch:
                                eligible_statuses = {
                                    SessionStatus.RUNNING,
                                    SessionStatus.INTERRUPTING,
                                }
                            elif reconciliation_session.run_epoch == session.run_epoch + 1:
                                eligible_statuses = {
                                    SessionStatus.INTERRUPTED,
                                    SessionStatus.FAILED,
                                }
                            else:
                                return
                            try:
                                await self._session_store.publish_checkpoint_and_events(
                                    session.id,
                                    checkpoint_transform=preserve_checkpoint,
                                    events=[reconciliation_event],
                                    expected_statuses=eligible_statuses,
                                    expected_run_epoch=reconciliation_session.run_epoch,
                                )
                                break
                            except (SessionRunFenced, SessionStatusConflict):
                                if publication_attempt == 1:
                                    raise
                        persisted = await self._session_store.query_events(
                            EventQuery(
                                session_id=session.id,
                                event_id=reconciliation_event.id,
                                limit=1,
                            )
                        )
                        if len(persisted) != 1 or persisted[0].event != reconciliation_event:
                            raise RuntimeError(
                                "Late provider-operation reconciliation readback did not "
                                "match its durable event."
                            )
                        await self._event_writer.fan_out_persisted([persisted[0].event])
                    finally:
                        await _close_async_iterator(raw_late_operation.events)

                if start_outcome.timed_out:
                    start_cancellation = start_outcome.cancellation
                    if start_cancellation is None:  # pragma: no cover - validated above
                        raise AssertionError("Timed-out provider start lost caller cancellation.")
                    reconciliation_task = asyncio.create_task(
                        reconcile_late_start(),
                        context=Context(),
                    )
                    self._retain_provider_operation_reconciliation(reconciliation_task)
                    start_cancellation.add_note(
                        "Provider operation start remained in flight after bounded cancellation "
                        "settlement; durable starting evidence prevents automatic retry."
                    )
                    raise start_cancellation

                start_error = start_outcome.error
                if (
                    isinstance(start_error, asyncio.CancelledError)
                    and start_outcome.cancellation is None
                ):
                    start_error = unexpected_child_cancellation_error(
                        start_error,
                        operation="Provider operation start",
                    )
                if start_error is not None:
                    if start_outcome.cancellation is not None:
                        raise start_outcome.cancellation from start_error
                    if not isinstance(start_error, Exception):
                        raise start_error
                    raise _ambiguous_provider_operation_start_error(
                        provider_name=registered_provider.name,
                        cause=start_error,
                    ) from start_error
                try:
                    if start_outcome.result is None:
                        raise RuntimeError("Provider operation start returned no connection.")
                    raw_provider_operation = start_outcome.result
                    try:
                        provider_operation = copy_provider_operation_connection(
                            raw_provider_operation
                        )
                    except BaseException:
                        if type(raw_provider_operation) is ProviderOperationConnection:
                            async with aclosing_provider_stream(raw_provider_operation.events):
                                raise
                        raise
                except Exception as start_validation_error:
                    if start_outcome.cancellation is not None:
                        raise start_outcome.cancellation from start_validation_error
                    raise _ambiguous_provider_operation_start_error(
                        provider_name=registered_provider.name,
                        cause=start_validation_error,
                    ) from start_validation_error
                operation_state = provider_operation.state
                provider_operation_state = operation_state
                operation_event = operation_event_for(
                    provider_operation.state,
                    provider_operation.status,
                )
                provider_events = provider_operation.events
                try:
                    operation_event = self._event_writer.prepare(operation_event)
                except BaseException as preparation_error:
                    (
                        cleanup_cancellation,
                        _,
                    ) = await _cancel_provider_operation_after_definite_absence(
                        adapter=provider_operation_adapter,
                        state=operation_state,
                        failure=preparation_error,
                        cancellation=start_outcome.cancellation,
                    )
                    provider_operation_state = None
                    if cleanup_cancellation is not None:
                        raise cleanup_cancellation from preparation_error
                    if isinstance(
                        preparation_error,
                        SessionInterruptedByRequest | SessionRunFenced,
                    ):
                        raise
                    if isinstance(preparation_error, Exception):
                        raise _ambiguous_provider_operation_start_error(
                            provider_name=registered_provider.name,
                            cause=preparation_error,
                        ) from preparation_error
                    raise
                try:
                    persisted_operation_event = await self._event_writer.persist_exact_replay(
                        operation_event
                    )
                except BaseException as persistence_error:
                    try:
                        exact_identity_is_durable = await self._event_writer.is_exact_persisted(
                            operation_event
                        )
                    except BaseException as verification_error:
                        persistence_error.add_note(
                            "Provider operation start-evidence readback also failed: "
                            f"{type(verification_error).__name__}."
                        )
                        if start_outcome.cancellation is not None:
                            raise start_outcome.cancellation from BaseExceptionGroup(
                                "Provider operation publication and readback both failed.",
                                [persistence_error, verification_error],
                            )
                        if isinstance(
                            verification_error,
                            SessionInterruptedByRequest | SessionRunFenced,
                        ) or not isinstance(verification_error, Exception):
                            raise
                        if isinstance(
                            persistence_error,
                            SessionInterruptedByRequest | SessionRunFenced,
                        ):
                            raise persistence_error from verification_error
                        if isinstance(persistence_error, Exception):
                            raise _ambiguous_provider_operation_start_error(
                                provider_name=registered_provider.name,
                                cause=persistence_error,
                            ) from verification_error
                        raise persistence_error from verification_error
                    if not exact_identity_is_durable:
                        (
                            cleanup_cancellation,
                            _,
                        ) = await _cancel_provider_operation_after_definite_absence(
                            adapter=provider_operation_adapter,
                            state=operation_state,
                            failure=persistence_error,
                            cancellation=start_outcome.cancellation,
                        )
                        provider_operation_state = None
                        if cleanup_cancellation is not None:
                            raise cleanup_cancellation from persistence_error
                    if start_outcome.cancellation is not None:
                        raise start_outcome.cancellation from persistence_error
                    if isinstance(
                        persistence_error,
                        SessionInterruptedByRequest | SessionRunFenced,
                    ):
                        raise
                    if isinstance(persistence_error, Exception):
                        raise _ambiguous_provider_operation_start_error(
                            provider_name=registered_provider.name,
                            cause=persistence_error,
                        ) from persistence_error
                    raise
                try:
                    [emitted_operation_event] = await self._event_writer.fan_out_persisted(
                        [persisted_operation_event]
                    )
                except BaseException as delivery_error:
                    if start_outcome.cancellation is not None:
                        raise start_outcome.cancellation from delivery_error
                    if isinstance(
                        delivery_error,
                        SessionInterruptedByRequest | SessionRunFenced,
                    ):
                        raise
                    if isinstance(delivery_error, Exception):
                        raise _ambiguous_provider_operation_start_error(
                            provider_name=registered_provider.name,
                            cause=delivery_error,
                        ) from delivery_error
                    raise
                provider_operation_identity_durable = True
                if start_outcome.cancellation is not None:
                    raise start_outcome.cancellation
                yield emitted_operation_event, None
            async for raw_stream_event in provider_events:
                boundary_value = _validate_stream_event(
                    raw_stream_event,
                    provider_name=registered_provider.name,
                    requested_model=session.model,
                    usage_dialect=registered_provider.usage_dialect,
                )
                generated_tool_call_id = None
                if provider_operation_state is not None and completion_dispatch is not None:
                    generated_tool_call_id = _provider_operation_generated_tool_call_id(
                        completion_dispatch.stage,
                        boundary_value.event,
                    )
                assistant_boundary = _validate_assistant_stream_event(
                    boundary_value.event,
                    generated_tool_call_id=generated_tool_call_id,
                )
                stream_event = assistant_boundary.event
                await interrupt_poll.raise_if_interrupted()
                if model_completed:
                    if (
                        provider_operation_state is not None
                        and completed_stream_event is not None
                        and stream_event == completed_stream_event
                    ):
                        continue
                    message = f"Model provider emitted event after completed: {stream_event.type}"
                    raise ModelAttemptFailed(
                        message=message,
                        payload={"error": message, "error_type": "RuntimeError"},
                        emitted_error_event=False,
                        cause=RuntimeError(message),
                        completion_observed=model_completion_publisher is not None,
                    )

                progress_emitted_event: Event | None = None

                if stream_event.type == ModelStreamEventType.TOOL_CALL:
                    if provider_operation_state is not None:
                        if completion_dispatch is None:  # pragma: no cover - checked at start
                            raise AssertionError("Background operation lost its completion stage.")
                        if provider_operation_interaction_id is None:  # pragma: no cover
                            raise AssertionError("Background operation lost its interaction.")
                        (
                            progress,
                            progress_emitted_event,
                        ) = await self._commit_provider_operation_stream_event(
                            stage=completion_dispatch.stage,
                            state=provider_operation_state,
                            stream_event=stream_event,
                            runtime_event=None,
                            session=session,
                            interaction_id=provider_operation_interaction_id,
                            registered_agent=registered_agent,
                            registered_provider=registered_provider,
                            environment_name=environment_name,
                            step=step,
                            attempt=attempt,
                            max_attempts=max_attempts,
                            model_attempt_identity=model_attempt_identity,
                        )
                        provider_operation_state = progress.state
                        if progress.replayed:
                            continue
                    tool_call = assistant_boundary.tool_call
                    tool_call_part = assistant_boundary.tool_call_part
                    if tool_call is None or tool_call_part is None:  # pragma: no cover
                        raise AssertionError("Validated tool-call projection disappeared.")
                    tool_calls.append(tool_call)
                    assistant_parts.append(tool_call_part)
                    if progress_emitted_event is not None:
                        yield progress_emitted_event, None
                    continue

                if stream_event.type in {
                    ModelStreamEventType.HOSTED_TOOL_CALL,
                    ModelStreamEventType.CITATION,
                }:
                    progress_runtime_event = _model_stream_event_to_runtime_event(
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
                    if provider_operation_state is not None:
                        if completion_dispatch is None:  # pragma: no cover - checked at start
                            raise AssertionError("Background operation lost its completion stage.")
                        if provider_operation_interaction_id is None:  # pragma: no cover
                            raise AssertionError("Background operation lost its interaction.")
                        (
                            progress,
                            progress_emitted_event,
                        ) = await self._commit_provider_operation_stream_event(
                            stage=completion_dispatch.stage,
                            state=provider_operation_state,
                            stream_event=stream_event,
                            runtime_event=progress_runtime_event,
                            session=session,
                            interaction_id=provider_operation_interaction_id,
                            registered_agent=registered_agent,
                            registered_provider=registered_provider,
                            environment_name=environment_name,
                            step=step,
                            attempt=attempt,
                            max_attempts=max_attempts,
                            model_attempt_identity=model_attempt_identity,
                        )
                        provider_operation_state = progress.state
                        if progress.replayed:
                            continue
                    if stream_event.type is ModelStreamEventType.HOSTED_TOOL_CALL:
                        hosted_part = _hosted_tool_call_part(
                            stream_event,
                            provider_name=registered_provider.name,
                            model=session.model,
                            model_attempt_identity=model_attempt_identity,
                        )
                        if hosted_part is not None:
                            assistant_parts.append(hosted_part)
                    else:
                        assistant_parts.append(
                            _citation_part(
                                stream_event,
                                provider_name=registered_provider.name,
                                model_attempt_identity=model_attempt_identity,
                                assistant_parts=assistant_parts,
                            )
                        )
                    emitted_event = (
                        await self._event_writer.emit(progress_runtime_event)
                        if progress_emitted_event is None
                        else progress_emitted_event
                    )
                    yield emitted_event, None
                    continue

                if stream_event.type == ModelStreamEventType.TEXT_DELTA:
                    if provider_operation_state is not None:
                        if completion_dispatch is None:  # pragma: no cover - checked at start
                            raise AssertionError("Background operation lost its completion stage.")
                        if provider_operation_interaction_id is None:  # pragma: no cover
                            raise AssertionError("Background operation lost its interaction.")
                        progress_runtime_event = _model_stream_event_to_runtime_event(
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
                            execution_profile_fingerprint=(
                                None if execution_profile is None else execution_profile.fingerprint
                            ),
                        )
                        (
                            progress,
                            progress_emitted_event,
                        ) = await self._commit_provider_operation_stream_event(
                            stage=completion_dispatch.stage,
                            state=provider_operation_state,
                            stream_event=stream_event,
                            runtime_event=progress_runtime_event,
                            session=session,
                            interaction_id=provider_operation_interaction_id,
                            registered_agent=registered_agent,
                            registered_provider=registered_provider,
                            environment_name=environment_name,
                            step=step,
                            attempt=attempt,
                            max_attempts=max_attempts,
                            model_attempt_identity=model_attempt_identity,
                        )
                        provider_operation_state = progress.state
                        if progress.replayed:
                            continue
                    transcript_helpers.append_assistant_text_delta(
                        assistant_parts,
                        stream_event.delta,
                    )
                elif stream_event.type == ModelStreamEventType.THINKING:
                    if provider_operation_state is not None:
                        if completion_dispatch is None:  # pragma: no cover - checked at start
                            raise AssertionError("Background operation lost its completion stage.")
                        if provider_operation_interaction_id is None:  # pragma: no cover
                            raise AssertionError("Background operation lost its interaction.")
                        progress_runtime_event = (
                            _model_stream_event_to_runtime_event(
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
                                execution_profile_fingerprint=(
                                    None
                                    if execution_profile is None
                                    else execution_profile.fingerprint
                                ),
                            )
                            if stream_event.delta
                            else None
                        )
                        (
                            progress,
                            progress_emitted_event,
                        ) = await self._commit_provider_operation_stream_event(
                            stage=completion_dispatch.stage,
                            state=provider_operation_state,
                            stream_event=stream_event,
                            runtime_event=progress_runtime_event,
                            session=session,
                            interaction_id=provider_operation_interaction_id,
                            registered_agent=registered_agent,
                            registered_provider=registered_provider,
                            environment_name=environment_name,
                            step=step,
                            attempt=attempt,
                            max_attempts=max_attempts,
                            model_attempt_identity=model_attempt_identity,
                        )
                        provider_operation_state = progress.state
                        if progress.replayed:
                            continue
                    transcript_helpers.append_assistant_thinking_delta(
                        assistant_parts,
                        stream_event.delta,
                        provider_state=stream_event.payload.get("provider_state"),
                        include=include_thinking_in_transcript,
                    )
                    if not stream_event.delta:
                        # Opaque/redacted thinking state belongs in the transcript,
                        # but an empty readable delta should not reach consumers.
                        if progress_emitted_event is not None:
                            yield progress_emitted_event, None
                        continue
                elif stream_event.type == ModelStreamEventType.COMPLETED:
                    terminal_progress_failure: BaseException | None = None
                    terminal_progress_verified = False
                    if provider_operation_state is not None:
                        envelope = provider_operation_progress_envelope(
                            provider_operation_state,
                            stream_event,
                        )
                        if _provider_operation_progress_contains_secret(
                            envelope,
                            redactor=self._secret_redactor,
                        ):
                            terminal_progress_failure = ProviderOperationEvidenceError(
                                "Provider-operation terminal recovery state contains a workload "
                                "secret."
                            )
                        else:
                            terminal_cursor = envelope.recovery_metadata.cursor
                            current_cursor = provider_operation_state.recovery_metadata.cursor
                            current_cursor = -1 if current_cursor is None else current_cursor
                            if terminal_cursor != current_cursor + 1:
                                terminal_progress_failure = ProviderOperationEvidenceError(
                                    "Provider-operation terminal cursor is not the next boundary."
                                )
                            else:
                                terminal_progress_verified = True
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
                    defer_assistant_message = bool(tool_calls)
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
                            + (
                                1
                                if assistant_message is not None and not defer_assistant_message
                                else 0
                            )
                        ),
                        usage_dialect=registered_provider.usage_dialect,
                        billing_identity=billing_identity,
                        accounting_usage_metrics=boundary_value.accounting_usage_metrics,
                        accounting_usage_rejected=boundary_value.accounting_usage_rejected,
                        usage_normalization_failed=(boundary_value.usage_normalization_failed),
                        completion_diagnostics=completion_diagnostics,
                        execution_profile_fingerprint=(
                            None if execution_profile is None else execution_profile.fingerprint
                        ),
                    )
                    if provider_operation_state is not None and terminal_progress_verified:
                        if completion_dispatch is None:  # pragma: no cover - checked at start
                            raise AssertionError("Background operation lost its completion stage.")
                        if provider_operation_interaction_id is None:  # pragma: no cover
                            raise AssertionError("Background operation lost its interaction.")
                        completion_event = self._provider_operation_progress_event(
                            stage=completion_dispatch.stage,
                            state=provider_operation_state,
                            stream_event=stream_event,
                            runtime_event=completion_event,
                            session=session,
                            interaction_id=provider_operation_interaction_id,
                            registered_agent=registered_agent,
                            registered_provider=registered_provider,
                            environment_name=environment_name,
                            step=step,
                            attempt=attempt,
                            max_attempts=max_attempts,
                            model_attempt_identity=model_attempt_identity,
                        )
                    if model_completion_publisher is None:
                        completion_event = record_model_completion(completion_event)
                        yield await self._event_writer.emit(completion_event), None
                    if terminal_progress_failure is not None:
                        post_completion_failure = _combine_post_completion_failures(
                            post_completion_failure,
                            terminal_progress_failure,
                        )
                    if completion_terminal_error is not None:
                        if post_completion_failure is None:
                            provider_control_failure = completion_terminal_error
                        else:
                            post_completion_failure = _combine_post_completion_failures(
                                post_completion_failure,
                                completion_terminal_error,
                            )
                        break
                    if provider_operation_state is not None:
                        break
                    continue

                stream_retry_decision: RetryDecision | None = None
                provider_error: ModelProviderError | None = None
                message = ""
                if stream_event.type == ModelStreamEventType.ERROR:
                    message = str(stream_event.payload.get("error") or "Model provider error")
                    provider_error = model_provider_error_from_payload(
                        stream_event.payload,
                        fallback_provider=registered_provider.name,
                        fallback_message=message,
                    )
                    if (
                        isinstance(provider_error, ModelContextOverflowError)
                        and provider_operation_state is None
                    ):
                        # Providers may flatten a typed overflow into an error
                        # event. Rehydrate it so bounded recovery can shrink the
                        # request instead of spending generic retries on it.
                        raise provider_error

                    stream_retry_decision = retry_decision(
                        policy=retry_policy,
                        attempt=attempt,
                        error=message,
                        status_code=(
                            provider_error.status_code
                            if isinstance(provider_error, ModelProviderError)
                            else None
                        ),
                        retryable=(
                            False
                            if provider_operation_state is not None
                            else (
                                provider_error.retryable
                                if isinstance(provider_error, ModelProviderError)
                                else None
                            )
                        ),
                        retry_after_s=(
                            None
                            if provider_operation_state is not None
                            else (
                                provider_error.retry_after_s
                                if isinstance(provider_error, ModelProviderError)
                                else None
                            )
                        ),
                        unknown_provider_error=(
                            provider_operation_state is None
                            and isinstance(provider_error, ModelProviderError)
                            and provider_error.status_code is None
                            and provider_error.retryable is None
                        ),
                    )

                    if provider_operation_state is not None:
                        if completion_dispatch is None:  # pragma: no cover - checked at start
                            raise AssertionError("Background operation lost its completion stage.")
                        if provider_operation_interaction_id is None:  # pragma: no cover
                            raise AssertionError("Background operation lost its interaction.")
                        progress_runtime_event = _model_stream_event_to_runtime_event(
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
                            execution_profile_fingerprint=(
                                None if execution_profile is None else execution_profile.fingerprint
                            ),
                            retry_decision=stream_retry_decision,
                        )
                        (
                            progress,
                            progress_emitted_event,
                        ) = await self._commit_provider_operation_stream_event(
                            stage=completion_dispatch.stage,
                            state=provider_operation_state,
                            stream_event=stream_event,
                            runtime_event=progress_runtime_event,
                            session=session,
                            interaction_id=provider_operation_interaction_id,
                            registered_agent=registered_agent,
                            registered_provider=registered_provider,
                            environment_name=environment_name,
                            step=step,
                            attempt=attempt,
                            max_attempts=max_attempts,
                            model_attempt_identity=model_attempt_identity,
                        )
                        provider_operation_state = progress.state
                        if progress.replayed:
                            continue

                if progress_emitted_event is None:
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
                        execution_profile_fingerprint=(
                            None if execution_profile is None else execution_profile.fingerprint
                        ),
                        retry_decision=(
                            stream_retry_decision
                            if stream_event.type == ModelStreamEventType.ERROR
                            else None
                        ),
                    )
                    emitted_event = await self._event_writer.emit(event)
                else:
                    emitted_event = progress_emitted_event
                if stream_event.type == ModelStreamEventType.ERROR:
                    if stream_retry_decision is None:  # pragma: no cover - set above
                        raise AssertionError("Model error lost its retry decision.")
                    yield emitted_event, None
                    if provider_operation_state is not None and isinstance(
                        provider_error, ModelContextOverflowError
                    ):
                        provider_control_failure = provider_error
                        provider_control_error_emitted = True
                        break
                    raise ModelAttemptFailed(
                        message=message,
                        payload=copy_json_value(stream_event.payload, "payload"),
                        emitted_error_event=True,
                        cause=provider_error or RuntimeError(message),
                        retry_decision=stream_retry_decision,
                    )
                yield emitted_event, None
            else:
                provider_exhausted = True

        except SessionInterruptedByRequest as exc:
            if model_completion_publisher is None or not model_completed:
                if (
                    provider_operation_adapter is not None
                    and provider_operation_state is not None
                    and provider_operation_interaction_id is not None
                    and provider_operation_identity_durable
                    and completion_dispatch is not None
                ):
                    cancellation_snapshot = await self._cancel_started_provider_operation(
                        adapter=provider_operation_adapter,
                        state=provider_operation_state,
                        failure=exc,
                        session=session,
                        stage=completion_dispatch.stage,
                        interaction_id=provider_operation_interaction_id,
                        registered_agent=registered_agent,
                        registered_provider=registered_provider,
                        environment_name=environment_name,
                        step=step,
                        attempt=attempt,
                        max_attempts=max_attempts,
                        model_attempt_identity=model_attempt_identity,
                    )
                    await reconcile_completion_that_won_cancellation(cancellation_snapshot)
                raise
            post_completion_failure = exc
        except asyncio.CancelledError as exc:
            if model_completion_publisher is None or not model_completed:
                if (
                    provider_operation_adapter is not None
                    and provider_operation_state is not None
                    and provider_operation_interaction_id is not None
                    and provider_operation_identity_durable
                    and completion_dispatch is not None
                ):
                    cancellation_snapshot = await self._cancel_started_provider_operation(
                        adapter=provider_operation_adapter,
                        state=provider_operation_state,
                        failure=exc,
                        session=session,
                        stage=completion_dispatch.stage,
                        interaction_id=provider_operation_interaction_id,
                        registered_agent=registered_agent,
                        registered_provider=registered_provider,
                        environment_name=environment_name,
                        step=step,
                        attempt=attempt,
                        max_attempts=max_attempts,
                        model_attempt_identity=model_attempt_identity,
                    )
                    await reconcile_completion_that_won_cancellation(cancellation_snapshot)
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
                if background_dispatch_invoked and not exc.automatic_retry_disabled:
                    raise ModelAttemptFailed(
                        message=exc.message,
                        payload=exc.payload,
                        emitted_error_event=exc.emitted_error_event,
                        cause=exc.cause,
                        completion_observed=exc.completion_observed,
                        automatic_retry_disabled=True,
                    ) from exc
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
                            event_with_execution_profile_authority(
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
                                ),
                                execution_profile,
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
                active_failure = sys.exception()
                if background_dispatch_invoked and model_completed:
                    try:
                        async with aclosing_provider_stream(provider_events):
                            pass
                    except BaseException as cleanup_failure:
                        primary_failure = (
                            post_completion_failure
                            or provider_control_failure
                            or durable_stream_failure
                            or active_failure
                        )
                        post_completion_failure = (
                            cleanup_failure
                            if primary_failure is None or primary_failure is cleanup_failure
                            else _combine_post_completion_failures(
                                primary_failure,
                                cleanup_failure,
                            )
                        )
                else:
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
            defer_assistant_message = bool(
                authoritative_assistant_message is not None
                and durable_step_result is not None
                and durable_step_result.tool_calls
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
                defer_assistant_message=defer_assistant_message,
                structured_output_validation=structured_output_validation,
                tool_exposure=(
                    None
                    if tool_exposure is None
                    else resolved_tool_exposure_authority(tool_exposure)
                ),
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

        if provider_control_failure is not None and background_dispatch_invoked:
            post_dispatch_failure = ModelProviderError(
                "A background provider operation failed after dispatch; automatic retry and "
                "context-overflow recovery are disabled while the original operation may "
                "remain active.",
                provider=registered_provider.name,
                error_type=type(provider_control_failure).__name__,
                error_code="provider_operation_failed_after_dispatch",
                retryable=False,
            )
            set_exception_cause(post_dispatch_failure, provider_control_failure)
            raise ModelAttemptFailed(
                message=str(post_dispatch_failure),
                payload={
                    "error": str(post_dispatch_failure),
                    "error_type": type(provider_control_failure).__name__,
                },
                emitted_error_event=provider_control_error_emitted,
                cause=post_dispatch_failure,
                automatic_retry_disabled=True,
            ) from post_dispatch_failure
        if type(provider_control_failure) is ModelContextOverflowError:
            yield (
                await self._event_writer.emit(
                    event_with_execution_profile_authority(
                        _model_context_overflow_error_event(
                            provider_control_failure,
                            session=session,
                            registered_agent=registered_agent,
                            environment_name=environment_name,
                            step=step,
                            attempt=attempt,
                            max_attempts=max_attempts,
                            model_attempt_identity=model_attempt_identity,
                        ),
                        execution_profile,
                    )
                ),
                None,
            )
        if provider_control_failure is not None:
            raise provider_control_failure from None
        if durable_stream_failure is not None:
            if background_dispatch_invoked and not durable_stream_failure.automatic_retry_disabled:
                durable_stream_failure = ModelAttemptFailed(
                    message=durable_stream_failure.message,
                    payload=durable_stream_failure.payload,
                    emitted_error_event=durable_stream_failure.emitted_error_event,
                    cause=durable_stream_failure.cause,
                    completion_observed=durable_stream_failure.completion_observed,
                    automatic_retry_disabled=True,
                )
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
                automatic_retry_disabled=background_dispatch_invoked,
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
        knowledge_access_scope: Any,
        request_metadata: dict[str, Any],
        retry_policy: RetryPolicy,
        request_budget_limits: tuple[BudgetLimit, ...],
        limit_gate: RunLimitGate,
        budget_policy: BudgetPolicy | None,
        run_started_at: float,
        turn_usage_tracker: SessionUsageTracker | None,
        active_run: ActiveSessionRun[SessionUsageTracker] | None,
        execution_profile: ExecutionProfileIdentity | None,
        validate_live_model_semantics: Callable[[], None],
        initial_tool_exposure: ResolvedToolExposure | None,
        previous_tool_exposure_profile_id: str | None,
        model_completion_recovery_context_factory: ModelCompletionRecoveryContextFactory,
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
        self._knowledge_access_scope = knowledge_access_scope
        self._request_metadata = copy_json_value(request_metadata, "metadata")
        self._retry_policy = copy_retry_policy(retry_policy)
        self._request_budget_limits = copy_request_budget_limits(request_budget_limits)
        self._limit_gate = limit_gate
        self._budget_policy = budget_policy
        self._run_started_at = run_started_at
        self._turn_usage_tracker = turn_usage_tracker
        self._active_run = active_run
        self._execution_profile = execution_profile
        self._validate_live_model_semantics = validate_live_model_semantics
        capability_ceiling = tool_capability_ceiling_from_session_metadata(
            self._session.metadata,
        )
        self._tool_capability_ceiling = capability_ceiling
        self._all_tools_within_capability_ceiling = _tool_capability_ceiling_exposure(
            self._registered_agent,
            self._tool_capability_ceiling.tool_names,
        )
        if initial_tool_exposure is not None and previous_tool_exposure_profile_id is not None:
            raise ValueError(
                "An initial frozen tool exposure and a previous profile cannot be supplied "
                "together."
            )
        if initial_tool_exposure is None:
            self._initial_tool_exposure = None
        else:
            frozen_initial_tool_exposure = _require_frozen_tool_exposure(initial_tool_exposure)
            ceiling_names = frozenset(self._tool_capability_ceiling.tool_names)
            if frozen_initial_tool_exposure.ceiling_count != len(
                self._tool_capability_ceiling.tool_names
            ) or any(name not in ceiling_names for name in frozen_initial_tool_exposure.tool_names):
                raise ProviderOperationEvidenceError(
                    "Initial frozen tool exposure conflicts with the session capability ceiling."
                )
            self._initial_tool_exposure = frozen_initial_tool_exposure
        if previous_tool_exposure_profile_id is not None:
            previous_tool_exposure_profile_id = require_durable_clean_nonblank(
                previous_tool_exposure_profile_id,
                "previous_tool_exposure_profile_id",
            )
            if len(previous_tool_exposure_profile_id) > TOOL_EXPOSURE_PROFILE_ID_MAX_CHARS:
                raise ValueError(
                    "previous_tool_exposure_profile_id cannot exceed "
                    f"{TOOL_EXPOSURE_PROFILE_ID_MAX_CHARS} characters."
                )
        self._model_completion_recovery_context_factory = model_completion_recovery_context_factory
        self._reservation_identity_guard = (
            self._executor._run_limit_controller.reservation_identity_guard()
        )
        self._model_completion_publisher = model_completion_publisher
        self._previous_tool_exposure_profile_id = previous_tool_exposure_profile_id
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

    @property
    def execution_profile(self) -> ExecutionProfileIdentity | None:
        """Return the exact immutable profile resolved for this invocation."""

        return self._execution_profile

    def _resolve_tool_exposure(
        self,
        *,
        step: int,
        transcript_cursor: int,
    ) -> ResolvedToolExposure:
        if self._initial_tool_exposure is not None:
            exposure = self._initial_tool_exposure
            self._initial_tool_exposure = None
            return exposure
        policy = self._registered_agent.tool_exposure_policy
        if type(policy) is AllRegisteredToolsExposurePolicy:
            # Preserve both the historical metadata bounds and the expose-all
            # hot path: registration already validated this immutable snapshot.
            return self._all_tools_within_capability_ceiling
        request = ToolExposurePolicyRequest(
            session_id=self._session.id,
            agent_name=self._registered_agent.spec.name,
            provider_name=self._registered_provider.name,
            model=self._session.model,
            step=step,
            transcript_cursor=transcript_cursor,
            registered_tools=self._registered_agent.tool_capabilities,
            capability_ceiling=self._tool_capability_ceiling.tool_names,
            previous_profile_id=self._previous_tool_exposure_profile_id,
            metadata=self._request_metadata,
        )
        exposure = resolve_tool_exposure(policy, request)
        if (
            exposure.profile_id != ALL_REGISTERED_TOOLS_PROFILE_ID
            and self._executor._secret_redactor.redact_text(exposure.profile_id)
            != exposure.profile_id
        ):
            raise ValueError(
                "Tool exposure profile_id contains a workload secret and cannot become "
                "provider or durable execution authority."
            )
        return exposure

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
        source_transcript_cursor: int,
        model_step_identity: ModelStepIdentity,
        request_variant: RequestVariant = RequestVariant.INITIAL,
    ) -> AsyncIterator[tuple[Event | None, ModelStepFlowOutcome | None]]:
        if type(source_transcript_cursor) is not int:
            raise TypeError("source_transcript_cursor must be an int.")
        if source_transcript_cursor < 0:
            raise ValueError("source_transcript_cursor must be >= 0.")
        model_step_identity = copy_model_step_identity(model_step_identity)
        request_variant = RequestVariant(request_variant)
        previous_tool_exposure_profile_id = self._previous_tool_exposure_profile_id
        tool_exposure = self._resolve_tool_exposure(
            step=step,
            transcript_cursor=source_transcript_cursor,
        )
        exposure_profile_changed = (
            previous_tool_exposure_profile_id is not None
            and previous_tool_exposure_profile_id != tool_exposure.profile_id
        )
        self._previous_tool_exposure_profile_id = tool_exposure.profile_id
        context_messages: list[Message]
        context_operation_events: list[Event] = []
        published_compaction_attempt_ids: set[str] = set()
        compaction_start_events: list[Event] = []
        compaction_completion_events: dict[str, Event] = {}
        compaction_identity_ledger = _CompactionExecutionIdentityLedger(model_step_identity)

        async def publish_recall_telemetry(
            telemetry: ContextRecallTelemetry,
        ) -> None:
            event = _context_recall_telemetry_event(
                telemetry=telemetry,
                session=self._session,
                registered_agent=self._registered_agent,
                environment_name=self._environment_name,
                model_step_identity=model_step_identity,
                execution_profile=self._execution_profile,
            )
            context_operation_events.append(await self._executor._event_writer.emit(event))

        async def run_automatic_compaction(
            compactor: ContextCompactor,
            compaction_request: CompactionRequest,
            compaction_started: ContextCompactionTelemetry,
            execute: Callable[[], Awaitable[CompactionResult]],
            completed_payloads: Callable[[], list[dict[str, Any]]],
        ) -> CompactionResult:
            await self._persist_automatic_compaction_started(
                compaction_started,
                published_events=context_operation_events,
                start_events=compaction_start_events,
                model_step_identity=model_step_identity,
            )

            async def publish_completions(payloads: list[dict[str, Any]]) -> None:
                await self._persist_automatic_compaction_completions(
                    compaction_identity_ledger.identify_payloads(payloads),
                    published_attempt_ids=published_compaction_attempt_ids,
                    published_events=context_operation_events,
                    completion_events=compaction_completion_events,
                )

            async def run() -> CompactionResult:
                return await self._run_automatic_compaction_with_budget(
                    compactor=compactor,
                    compaction_request=compaction_request,
                    execute=execute,
                    completed_payloads=completed_payloads,
                    budget_events=context_operation_events,
                    messages=messages,
                    step=step,
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
            self._validate_live_model_semantics()
            (
                context_messages,
                checkpoint_update,
                checkpoint_event_payload,
                context_compaction_telemetry,
                context_recall_telemetry,
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
                knowledge_access_scope=self._knowledge_access_scope,
                request_metadata=self._request_metadata,
                pressure_overhead=_context_pressure_overhead(
                    registered_provider=self._registered_provider,
                    registered_agent=self._registered_agent,
                    registered_environment=self._registered_environment,
                    structured_output=self._structured_output,
                    thinking=self._thinking,
                    step=step,
                    tool_exposure=tool_exposure,
                ),
                count_input_tokens=self._context_input_token_counter(
                    step=step,
                    tool_exposure=tool_exposure,
                ),
                build_cache_prefix_request=self._cache_prefix_request_builder(
                    step=step,
                    tool_exposure=tool_exposure,
                ),
                secret_redactor=self._executor._secret_redactor,
                run_compaction=run_automatic_compaction,
                publish_recall_telemetry=publish_recall_telemetry,
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
                    for event in context_operation_events
                ),
            )
            for event in context_operation_events:
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
                    for event in context_operation_events
                ),
                cancellation_requests_before_build=context_build_cancellation_requests,
            )
            raise

        # A context-policy extension may have mutated a live provider while
        # constructing the request. Detect that before request preparation.
        self._validate_live_model_semantics()
        context_success_events, context_success_persistence = await self._context_success_events(
            model_step_identity=model_step_identity,
            compaction_identity_ledger=compaction_identity_ledger,
            checkpoint_update=checkpoint_update,
            checkpoint_event_payload=checkpoint_event_payload,
            compaction_telemetry=context_compaction_telemetry,
            recall_telemetry=context_recall_telemetry,
            published_compaction_attempt_ids=published_compaction_attempt_ids,
            compaction_completion_events=compaction_completion_events,
            compaction_start_event=(
                compaction_start_events[0] if compaction_start_events else None
            ),
            compaction_started_published=any(
                event.type == EventType.CONTEXT_COMPACTION_STARTED
                for event in context_operation_events
            ),
        )
        for event in context_operation_events:
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
            tool_exposure=tool_exposure,
        )
        if self._execution_profile is None:
            raise RuntimeError("Tool exposure evidence requires an execution profile.")
        tool_exposure_evidence = tool_exposure_record(
            tool_exposure,
            profile_changed=exposure_profile_changed,
            step=step,
            provider_name=self._registered_provider.name,
            model=self._session.model,
            model_step_id=model_step_identity.model_step_id,
            execution_profile_fingerprint=self._execution_profile.fingerprint,
        )
        yield (
            await self._executor._event_writer.emit(
                _tool_exposure_event(
                    exposure=tool_exposure_evidence,
                    session=self._session,
                    registered_agent=self._registered_agent,
                    environment_name=self._environment_name,
                    model_step_identity=model_step_identity,
                    execution_profile=self._execution_profile,
                )
            ),
            None,
        )
        request_events = self._execute_request(
            model_request=model_request,
            step=step,
            messages=messages,
            source_transcript_cursor=source_transcript_cursor,
            model_step_identity=model_step_identity,
            request_variant=request_variant,
            tool_exposure=tool_exposure,
            tool_exposure_evidence=tool_exposure_evidence,
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
        source_transcript_cursor: int | None = None,
        model_step_identity: ModelStepIdentity,
        request_variant: RequestVariant = RequestVariant.INITIAL,
        tool_exposure: ResolvedToolExposure | None = None,
        tool_exposure_evidence: ToolExposure | None = None,
    ) -> AsyncIterator[tuple[Event | None, ModelStepFlowOutcome | None]]:
        model_step_identity = copy_model_step_identity(model_step_identity)
        tool_exposure = (
            _all_registered_tool_exposure(self._registered_agent)
            if tool_exposure is None
            else _require_frozen_tool_exposure(tool_exposure)
        )
        if tool_exposure_evidence is None:
            if self._execution_profile is None:
                raise RuntimeError("Tool exposure evidence requires an execution profile.")
            tool_exposure_evidence = tool_exposure_record(
                tool_exposure,
                profile_changed=False,
                step=step,
                provider_name=self._registered_provider.name,
                model=self._session.model,
                model_step_id=model_step_identity.model_step_id,
                execution_profile_fingerprint=self._execution_profile.fingerprint,
            )
        elif type(tool_exposure_evidence) is not ToolExposure:
            raise TypeError("tool_exposure_evidence must be a ToolExposure or None.")
        if source_transcript_cursor is None:
            source_transcript_cursor = len(messages)
        elif type(source_transcript_cursor) is not int:
            raise TypeError("source_transcript_cursor must be an int.")
        elif source_transcript_cursor < 0:
            raise ValueError("source_transcript_cursor must be >= 0.")
        request_variant = RequestVariant(request_variant)
        initial_model_attempt_identity = model_step_identity.new_attempt()
        controller = self._executor._run_limit_controller
        self._validate_live_model_semantics()
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
                    event_with_execution_profile_authority(
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
                        ),
                        self._execution_profile,
                    )
                ),
                None,
            )
            raise
        self._validate_live_model_semantics()
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
        self._validate_live_model_semantics()
        reservation_setup = await controller.reserve_for_model_step(
            session=self._session,
            agent_name=self._registered_agent.spec.name,
            provider_name=self._registered_provider.name,
            environment_name=self._environment_name,
            model_attempt_identity=initial_model_attempt_identity,
            budget_policy=self._budget_policy,
            request_budget_limits=self._request_budget_limits,
            billing_identity=billing_identity,
            execution_profile_fingerprint=(
                None if self._execution_profile is None else self._execution_profile.fingerprint
            ),
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
                execution_profile_fingerprint=(
                    None if self._execution_profile is None else self._execution_profile.fingerprint
                ),
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

        # The provider-facing transcript may contain only retained rows after
        # physical retention. The separately loaded permanent cursor is the
        # store fence and logical-step identity authority.
        logical_step_id = model_step_identity.model_step_id
        next_dispatch_ordinal = fallback_dispatch_ordinal_from_checkpoint(
            await self._executor._session_store.load_checkpoint(self._session.id),
            logical_step_id,
        )

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
                recovery_context = self._model_completion_recovery_context_factory(
                    billing_identity,
                    pending_reservations,
                )
                if recovery_context is not None:
                    recovery_context = recovery_context.model_copy(
                        update={"tool_exposure": resolved_tool_exposure_authority(tool_exposure)},
                        deep=True,
                    )
                if (
                    recovery_context is not None
                    and type(recovery_context) is not ModelCompletionRecoveryContext
                ):
                    raise TypeError(
                        "Model completion recovery context factory returned an invalid value."
                    )
                provider_operation_start: dict[str, Any] | None = None
                self._validate_live_model_semantics()
                provider_operation_mode = self._provider.provider_operation_mode
                if type(provider_operation_mode) is not ProviderOperationMode:
                    raise TypeError(
                        "ModelProvider.provider_operation_mode must return a ProviderOperationMode."
                    )
                if provider_operation_mode is ProviderOperationMode.BACKGROUND:
                    operation_adapter = self._provider.provider_operations
                    if not isinstance(operation_adapter, ProviderOperationAdapter):
                        raise RuntimeError(
                            "Background provider-operation mode requires a "
                            "ProviderOperationAdapter."
                        )
                    idempotency_support = operation_adapter.start_idempotency_support
                    if type(idempotency_support) is not ProviderOperationStartIdempotencySupport:
                        raise TypeError(
                            "ProviderOperationAdapter.start_idempotency_support must return "
                            "ProviderOperationStartIdempotencySupport."
                        )
                    idempotency_key = (
                        f"provider-operation:{pending_model_attempt_identity.model_attempt_id}"
                    )
                    provider_operation_start = {
                        "schema_version": 1,
                        "idempotency_support": idempotency_support.value,
                        "idempotency_key": idempotency_key,
                    }
                intent = _model_completion_stage_intent(
                    model_attempt_identity=pending_model_attempt_identity,
                    provider_name=self._registered_provider.name,
                    requested_model=attempt_model_request.model,
                    source_transcript_cursor=source_transcript_cursor,
                    request_fingerprint=request_fingerprint,
                    recovery_context=recovery_context,
                    provider_operation_start=provider_operation_start,
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
            request_variant=request_variant,
            model_step_identity=model_step_identity,
            initial_model_attempt_identity=initial_model_attempt_identity,
            transcript_cursor_before_request=source_transcript_cursor,
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
            tool_exposure=tool_exposure,
            tool_exposure_evidence=tool_exposure_evidence,
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
            cancellation_claim = authoritative_exc.__dict__.get(
                "provider_operation_cancellation_claim"
            )
            try:
                if not authoritative_exc.__dict__.get("provider_operation_accounting_pending"):
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
            finally:
                if isinstance(cancellation_claim, ProviderOperationCancellationClaim):
                    await self._executor._release_provider_operation_cancellation_claim(
                        session=self._session,
                        claim=cancellation_claim,
                    )
            raise
        except asyncio.CancelledError as authoritative_exc:
            cancellation_claim = authoritative_exc.__dict__.get(
                "provider_operation_cancellation_claim"
            )
            try:
                if not authoritative_exc.__dict__.get("provider_operation_accounting_pending"):
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
            finally:
                if isinstance(cancellation_claim, ProviderOperationCancellationClaim):
                    await self._executor._release_provider_operation_cancellation_claim(
                        session=self._session,
                        claim=cancellation_claim,
                    )
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
        request_variant: RequestVariant,
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
        tool_exposure: ResolvedToolExposure,
        tool_exposure_evidence: ToolExposure,
    ) -> AsyncIterator[tuple[Event | None, ModelStepFlowOutcome | None]]:
        model_step_identity = copy_model_step_identity(model_step_identity)
        request_variant = RequestVariant(request_variant)
        initial_model_attempt_identity = copy_model_attempt_identity(initial_model_attempt_identity)
        if initial_model_attempt_identity.model_step_id != model_step_identity.model_step_id:
            raise ValueError("Initial provider attempt belongs to a different model step.")
        overflow_policy = self._registered_agent.context_overflow_policy
        context_operation_events: list[Event] = []
        published_compaction_attempt_ids: set[str] = set()
        compaction_start_events: list[Event] = []
        compaction_completion_events: dict[str, Event] = {}
        compaction_identity_ledger = _CompactionExecutionIdentityLedger(model_step_identity)
        latest_model_attempt_identity: ModelAttemptIdentity | None = None

        async def publish_recall_telemetry(
            telemetry: ContextRecallTelemetry,
        ) -> None:
            event = _context_recall_telemetry_event(
                telemetry=telemetry,
                session=self._session,
                registered_agent=self._registered_agent,
                environment_name=self._environment_name,
                model_step_identity=model_step_identity,
                execution_profile=self._execution_profile,
            )
            context_operation_events.append(await self._executor._event_writer.emit(event))

        def record_model_attempt_identity(identity: ModelAttemptIdentity) -> None:
            nonlocal latest_model_attempt_identity
            latest_model_attempt_identity = copy_model_attempt_identity(identity)

        def run_attempt(
            request: ModelRequest,
            *,
            initial_identity: ModelAttemptIdentity | None = None,
            attempt_variant: RequestVariant,
        ) -> AsyncIterator[tuple[Event | None, AssistantStepResult | None]]:
            return self._executor.run_with_retries(
                provider=provider,
                model_request=request,
                session=self._session,
                registered_agent=self._registered_agent,
                registered_provider=self._registered_provider,
                environment_name=self._environment_name,
                step=step,
                request_variant=attempt_variant,
                model_step_identity=model_step_identity,
                initial_model_attempt_identity=initial_identity,
                retry_policy=self._retry_policy,
                transcript_cursor_before_request=transcript_cursor_before_request,
                record_model_completion=record_model_completion,
                prepare_provider_dispatch=prepare_provider_dispatch,
                before_provider_dispatch=before_provider_dispatch,
                validate_live_model_semantics=self._validate_live_model_semantics,
                record_model_attempt_identity=record_model_attempt_identity,
                billing_identity=billing_identity,
                structured_output=self._structured_output,
                prepare_model_completion_dispatch=prepare_model_completion_dispatch,
                model_completion_publisher=model_completion_publisher,
                execution_profile=self._execution_profile,
                tool_exposure=tool_exposure,
                tool_exposure_evidence=tool_exposure_evidence,
            )

        attempt_events = run_attempt(
            model_request,
            initial_identity=initial_model_attempt_identity,
            attempt_variant=request_variant,
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
                        event_with_execution_profile_authority(
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
                            ),
                            self._execution_profile,
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
                published_events=context_operation_events,
                start_events=compaction_start_events,
                model_step_identity=model_step_identity,
            )

            async def publish_completions(payloads: list[dict[str, Any]]) -> None:
                await self._persist_automatic_compaction_completions(
                    compaction_identity_ledger.identify_payloads(payloads),
                    published_attempt_ids=published_compaction_attempt_ids,
                    published_events=context_operation_events,
                    completion_events=compaction_completion_events,
                )

            async def run() -> CompactionResult:
                return await self._run_automatic_compaction_with_budget(
                    compactor=compactor,
                    compaction_request=compaction_request,
                    execute=execute,
                    completed_payloads=completed_payloads,
                    budget_events=context_operation_events,
                    messages=messages,
                    step=step,
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
            self._validate_live_model_semantics()
            (
                recovery_context_messages,
                checkpoint_update,
                checkpoint_event_payload,
                compaction_telemetry,
                recall_telemetry,
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
                knowledge_access_scope=self._knowledge_access_scope,
                request_metadata=self._request_metadata,
                pressure_overhead=_context_pressure_overhead(
                    registered_provider=self._registered_provider,
                    registered_agent=self._registered_agent,
                    registered_environment=self._registered_environment,
                    structured_output=self._structured_output,
                    thinking=self._thinking,
                    step=step,
                    tool_exposure=tool_exposure,
                ),
                count_input_tokens=self._context_input_token_counter(
                    step=step,
                    tool_exposure=tool_exposure,
                ),
                build_cache_prefix_request=self._cache_prefix_request_builder(
                    step=step,
                    tool_exposure=tool_exposure,
                ),
                secret_redactor=self._executor._secret_redactor,
                run_compaction=run_automatic_compaction,
                publish_recall_telemetry=publish_recall_telemetry,
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
                    for event in context_operation_events
                ),
            )
            for event in context_operation_events:
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
                    event_with_execution_profile_authority(
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
                        ),
                        self._execution_profile,
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
                    for event in context_operation_events
                ),
                cancellation_requests_before_build=context_build_cancellation_requests,
            )
            raise

        self._validate_live_model_semantics()
        context_success_events, context_success_persistence = await self._context_success_events(
            model_step_identity=model_step_identity,
            compaction_identity_ledger=compaction_identity_ledger,
            checkpoint_update=checkpoint_update,
            checkpoint_event_payload=checkpoint_event_payload,
            compaction_telemetry=compaction_telemetry,
            recall_telemetry=recall_telemetry,
            published_compaction_attempt_ids=published_compaction_attempt_ids,
            compaction_completion_events=compaction_completion_events,
            compaction_start_event=(
                compaction_start_events[0] if compaction_start_events else None
            ),
            compaction_started_published=any(
                event.type == EventType.CONTEXT_COMPACTION_STARTED
                for event in context_operation_events
            ),
        )
        for event in context_operation_events:
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
            tool_exposure=tool_exposure,
        )
        yield (
            await self._executor._event_writer.emit(
                event_with_execution_profile_authority(
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
                    ),
                    self._execution_profile,
                )
            ),
            None,
        )
        recovery_events = run_attempt(
            recovery_request,
            attempt_variant=RequestVariant.CONTEXT_OVERFLOW_RECOVERY,
        )
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
                        event_with_execution_profile_authority(
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
                            ),
                            self._execution_profile,
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
                    execution_profile=self._execution_profile,
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
                execution_profile=self._execution_profile,
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
                execution_profile=self._execution_profile,
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
                    execution_profile=self._execution_profile,
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
        recall_telemetry: list[ContextRecallTelemetry],
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

        prepared_events = [
            _context_recall_telemetry_event(
                telemetry=telemetry,
                session=self._session,
                registered_agent=self._registered_agent,
                environment_name=self._environment_name,
                model_step_identity=model_step_identity,
                execution_profile=self._execution_profile,
            )
            for telemetry in recall_telemetry
            if telemetry.event_type != EventType.AUTOMATIC_RECALL_ADMITTED
        ]
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
                    execution_profile=self._execution_profile,
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
            _context_recall_telemetry_event(
                telemetry=telemetry,
                session=self._session,
                registered_agent=self._registered_agent,
                environment_name=self._environment_name,
                model_step_identity=model_step_identity,
                execution_profile=self._execution_profile,
            )
            for telemetry in recall_telemetry
            if telemetry.event_type == EventType.AUTOMATIC_RECALL_ADMITTED
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

            checkpoint_event = event_with_execution_profile_authority(
                Event(
                    type=EventType.SESSION_CHECKPOINTED,
                    session_id=self._session.id,
                    agent_name=self._registered_agent.spec.name,
                    environment_name=self._environment_name,
                    payload={
                        **checkpoint_event_payload,
                        **model_step_identity.payload(),
                    },
                ),
                self._execution_profile,
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
            recall_telemetry=list(error.recall_telemetry),
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
                        execution_profile=self._execution_profile,
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
        recall_telemetry: list[ContextRecallTelemetry],
        published_compaction_attempt_ids: set[str],
        compaction_completion_events: dict[str, Event],
        compaction_start_event: Event | None,
        compaction_started_published: bool,
    ) -> tuple[list[Event], BaseException | None]:
        return await self._persist_context_events(
            model_step_identity=model_step_identity,
            compaction_identity_ledger=compaction_identity_ledger,
            compaction_telemetry=compaction_telemetry,
            recall_telemetry=recall_telemetry,
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
            execution_profile=self._execution_profile,
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
            execution_profile=self._execution_profile,
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
            execution_profile=self._execution_profile,
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
            execution_profile=self._execution_profile,
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
            execution_profile=self._execution_profile,
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
        tool_exposure: ResolvedToolExposure,
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
                tool_exposure=tool_exposure,
            )
            # Context-policy execution can await arbitrary application code.
            # Reject changed provider semantics at the final remote count seam.
            try:
                self._validate_live_model_semantics()
            except Exception as authority_error:
                raise _ContextCountAuthorityError(authority_error) from None
            result = await self._provider.count_input_tokens(request)
            return None if result is None else result.input_tokens

        return count_input_tokens

    def _cache_prefix_request_builder(
        self,
        *,
        step: int,
        tool_exposure: ResolvedToolExposure,
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
                tool_exposure=tool_exposure,
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
        step: int,
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

        async def record_compaction_footprint(
            *,
            provider: ModelProvider,
            provider_name: str,
            model_request: ModelRequest,
            attempt: int,
            max_attempts: int,
            model_attempt_identity: ModelAttemptIdentity,
        ) -> None:
            provider_name = require_durable_clean_nonblank(
                provider_name,
                "compactor_request_provider_name",
            )
            detached_request = _detach_model_request(model_request)
            if self._executor._request_footprint.enabled:
                footprint = analyze_request_footprint(
                    detached_request,
                    provider=provider,
                    provider_name=provider_name,
                    step=step,
                    attempt=attempt,
                    max_attempts=max_attempts,
                    request_variant=RequestVariant.CONTEXT_COMPACTION,
                    observation_id=str(uuid4()),
                    model_step_id=model_attempt_identity.model_step_id,
                    model_attempt_id=model_attempt_identity.model_attempt_id,
                    config=self._executor._request_footprint,
                    execution_profile_fingerprint=(
                        None
                        if self._execution_profile is None
                        else self._execution_profile.fingerprint
                    ),
                )
                budget_events.append(
                    await self._executor._event_writer.emit(
                        event_with_execution_profile_authority(
                            _context_observation_event(
                                Event(
                                    type=EventType.REQUEST_FOOTPRINT_RECORDED,
                                    session_id=self._session.id,
                                    agent_name=self._registered_agent.spec.name,
                                    environment_name=self._environment_name,
                                    payload=footprint.model_dump(mode="json", exclude_none=True),
                                )
                            ),
                            self._execution_profile,
                        )
                    )
                )
            budget_events.append(
                await self._executor._event_writer.emit(
                    event_with_execution_profile_authority(
                        _event_with_model_identity_authority(
                            Event(
                                type=EventType.MODEL_STARTED,
                                session_id=self._session.id,
                                agent_name=self._registered_agent.spec.name,
                                environment_name=self._environment_name,
                                payload={
                                    "model": detached_request.model,
                                    "provider": provider_name,
                                    "step": step,
                                    "attempt": attempt,
                                    "max_attempts": max_attempts,
                                    "purpose": ModelCompletionPurpose.CONTEXT_COMPACTION.value,
                                    **model_attempt_identity.payload(),
                                },
                            ),
                            model_attempt_identity,
                        ),
                        self._execution_profile,
                    )
                )
            )

        async def run_identity_only_dispatch(
            provider: ModelProvider,
            actual_provider_name: str,
            actual_pricing_provider_name: str,
            actual_model: str,
            actual_usage_dialect: UsageDialect,
            billing_identity: BillingIdentity | None,
            model_request: ModelRequest,
            attempt: int,
            max_attempts: int,
            dispatch: Callable[[], Awaitable[tuple[str, dict[str, Any]]]],
        ) -> tuple[str, dict[str, Any]]:
            """Identify a built-in dispatch reached through an opaque wrapper."""

            del (
                actual_pricing_provider_name,
                actual_model,
                actual_usage_dialect,
                billing_identity,
            )
            self._validate_live_model_semantics()
            model_attempt_identity = compaction_identity_ledger.begin_dispatch()
            try:
                await record_compaction_footprint(
                    provider=provider,
                    provider_name=actual_provider_name,
                    model_request=model_request,
                    attempt=attempt,
                    max_attempts=max_attempts,
                    model_attempt_identity=model_attempt_identity,
                )
                self._validate_live_model_semantics()
                with _compaction_model_attempt_identity_scope(model_attempt_identity):
                    return await dispatch()
            finally:
                compaction_identity_ledger.end_dispatch(model_attempt_identity)

        try:
            identity = compactor._provider_budget_identity_for_request(compaction_request)
        except NotImplementedError as exc:
            if self._executor._request_footprint.enabled:
                raise RuntimeError(
                    "Automatic compaction with request footprints requires the "
                    "ContextCompactor to explicitly declare "
                    "provider_budget_identity(session), returning provider/model or None "
                    "for deterministic execution."
                ) from exc
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
        if not uses_dispatch_boundary and (
            has_accounting_limits or self._executor._request_footprint.enabled
        ):
            raise RuntimeError(
                "Automatic provider-backed compaction with request footprints or accounting "
                "limits cannot "
                f"safely run opaque provider-backed compactor {type(compactor).__name__}: "
                "Cayu cannot observe each provider dispatch independently. Use an unmodified "
                "built-in provider compactor, disable request footprints, or remove the "
                "applicable run and budget limits."
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
            provider: ModelProvider,
            actual_provider_name: str,
            actual_pricing_provider_name: str,
            actual_model: str,
            actual_usage_dialect: UsageDialect,
            billing_identity: BillingIdentity | None,
            model_request: ModelRequest,
            attempt: int,
            max_attempts: int,
            dispatch: Callable[[], Awaitable[tuple[str, dict[str, Any]]]],
        ) -> tuple[str, dict[str, Any]]:
            del actual_usage_dialect
            self._validate_live_model_semantics()
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
                    self._validate_live_model_semantics()
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
                    self._validate_live_model_semantics()
                    await record_compaction_footprint(
                        provider=provider,
                        provider_name=actual_provider_name,
                        model_request=model_request,
                        attempt=attempt,
                        max_attempts=max_attempts,
                        model_attempt_identity=model_attempt_identity,
                    )
                    return await identified_dispatch()

                async def publish_dispatch_observation() -> None:
                    self._validate_live_model_semantics()
                    await record_compaction_footprint(
                        provider=provider,
                        provider_name=actual_provider_name,
                        model_request=model_request,
                        attempt=attempt,
                        max_attempts=max_attempts,
                        model_attempt_identity=model_attempt_identity,
                    )
                    self._validate_live_model_semantics()

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
                    execution_profile_fingerprint=(
                        None
                        if self._execution_profile is None
                        else self._execution_profile.fingerprint
                    ),
                    reservation_identity_guard=self._reservation_identity_guard,
                    before_provider_dispatch=publish_dispatch_observation,
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


def _model_request_tools(
    *,
    tool_exposure: ResolvedToolExposure,
    structured_output: StructuredOutputSpec | None,
) -> list[dict[str, Any]]:
    """Build detached tool declarations shared by preflight and model dispatch."""

    tools = [
        {
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.input_schema_copy(),
        }
        for tool in tool_exposure.tools
    ]
    if (
        structured_output is not None
        and structured_output.strategy == StructuredOutputStrategy.TOOL
    ):
        tools.append(structured_output_tool_spec(structured_output))
    return tools


def _require_frozen_tool_exposure(
    exposure: ResolvedToolExposure,
) -> ResolvedToolExposure:
    """Accept only the exact immutable snapshot type without rehashing its catalog."""

    if type(exposure) is not ResolvedToolExposure:
        raise TypeError("tool_exposure must be a ResolvedToolExposure.")
    return exposure


def _all_registered_tool_exposure(
    registered_agent: runtime_records.RegisteredAgentState,
) -> ResolvedToolExposure:
    """Build the compatibility snapshot for non-step portability preflight."""

    cached = registered_agent.all_registered_tool_exposure
    if cached is not None:
        return _require_frozen_tool_exposure(cached)
    capabilities = registered_agent.tool_capabilities
    return ResolvedToolExposure(
        profile_id=ALL_REGISTERED_TOOLS_PROFILE_ID,
        tools=capabilities,
        registered_count=len(capabilities),
        ceiling_count=len(capabilities),
    )


def _tool_capability_ceiling_exposure(
    registered_agent: runtime_records.RegisteredAgentState,
    ceiling_names: tuple[str, ...],
) -> ResolvedToolExposure:
    """Build the expose-all-policy snapshot inside one canonical ceiling."""

    capabilities = registered_agent.tool_capabilities
    registered_names = tuple(capability.name for capability in capabilities)
    if ceiling_names == registered_names:
        return _all_registered_tool_exposure(registered_agent)
    ceiling_name_set = frozenset(ceiling_names)
    selected = tuple(
        capability for capability in capabilities if capability.name in ceiling_name_set
    )
    if tuple(capability.name for capability in selected) != ceiling_names:
        raise ValueError("The durable tool capability ceiling conflicts with registration.")
    return ResolvedToolExposure(
        profile_id=ALL_REGISTERED_TOOLS_PROFILE_ID,
        tools=selected,
        registered_count=len(capabilities),
        ceiling_count=len(selected),
    )


def _model_request_messages(
    *,
    messages: list[Message],
    structured_output: StructuredOutputSpec | None,
) -> list[Message]:
    """Return the complete statically determined message surface for one request."""

    if (
        structured_output is not None
        and structured_output.strategy == StructuredOutputStrategy.TOOL
    ):
        return _with_structured_output_tool_instruction(messages, structured_output)
    return messages


def _context_pressure_overhead(
    *,
    registered_provider: runtime_records.RegisteredProvider,
    registered_agent: runtime_records.RegisteredAgentState,
    registered_environment: runtime_records.RegisteredEnvironment | None,
    structured_output: StructuredOutputSpec | None,
    thinking: ThinkingConfig | None,
    step: int,
    tool_exposure: ResolvedToolExposure,
) -> ContextPressureOverhead:
    profile = copy_model_context_pressure_profile(
        registered_provider.provider.context_pressure_profile
    )
    tools = _model_request_tools(
        tool_exposure=tool_exposure,
        structured_output=structured_output,
    )
    structured_output_instruction: str | None = None
    if (
        structured_output is not None
        and structured_output.strategy == StructuredOutputStrategy.TOOL
    ):
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
    knowledge_access_scope: Any,
    request_metadata: dict[str, Any],
    pressure_overhead: ContextPressureOverhead,
    count_input_tokens: Callable[[list[Message]], Awaitable[int | None]] | None,
    build_cache_prefix_request: Callable[[list[Message]], Awaitable[ModelRequest]] | None,
    secret_redactor: SecretRedactor,
    run_compaction: _AutomaticCompactionRunner | None = None,
    publish_recall_telemetry: Callable[[ContextRecallTelemetry], Awaitable[None]] | None = None,
    force_bounded_compaction: bool = False,
) -> tuple[
    list[Message],
    dict[str, Any] | None,
    dict[str, Any] | None,
    list[ContextCompactionTelemetry],
    list[ContextRecallTelemetry],
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
        session_store=session_store,
        knowledge_store=knowledge_store,
        knowledge_access_scope=knowledge_access_scope,
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
                _context_recall_telemetry_publisher_scope(publish_recall_telemetry),
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
            [telemetry.model_copy(deep=True) for telemetry in result.recall_telemetry],
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
    execution_profile: ExecutionProfileIdentity | None = None,
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
    event = (
        event
        if execution_identity is None
        else _event_with_model_identity_authority(event, execution_identity)
    )
    return event_with_execution_profile_authority(event, execution_profile)


def _context_recall_telemetry_event(
    *,
    telemetry: ContextRecallTelemetry,
    session: Session,
    registered_agent: runtime_records.RegisteredAgentState,
    environment_name: str | None,
    model_step_identity: ModelStepIdentity,
    execution_profile: ExecutionProfileIdentity | None = None,
) -> Event:
    if type(telemetry) is not ContextRecallTelemetry:
        raise TypeError("Context recall telemetry must be ContextRecallTelemetry instances.")
    payload = copy_json_value(telemetry.payload, "payload")
    strip_runtime_owned_execution_identity(payload)
    payload.update(copy_model_step_identity(model_step_identity).payload())
    return event_with_execution_profile_authority(
        Event(
            type=telemetry.event_type,
            session_id=session.id,
            agent_name=registered_agent.spec.name,
            environment_name=environment_name,
            payload=payload,
        ),
        execution_profile,
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

    return _ModelStreamBoundaryValue(
        event=ModelStreamEvent.model_construct(
            type=ModelStreamEventType.COMPLETED,
            delta=delta,
            payload=payload,
            completion=completion,
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
    if status == "completed" and action is None:
        raise ValueError("Completed web search calls require terminal action evidence.")
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


def _copy_model_request_for_counting(request: ModelRequest) -> ModelRequest:
    return _detach_model_request(request)


def _detach_model_request(request: ModelRequest) -> ModelRequest:
    if type(request) is not ModelRequest:
        raise TypeError("request must be a ModelRequest.")
    return ModelRequest(
        model=request.model,
        messages=request.messages,
        tools=request.tools,
        hosted_tools=request.hosted_tools,
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
            "model": session.model,
            "provider_operation_id": _provider_operation_id(model_attempt_identity),
        }
    elif stream_event.type == ModelStreamEventType.CITATION:
        event_type = EventType.MODEL_CITATION
        payload = {
            **_validated_citation_payload(stream_event.payload),
            "model": session.model,
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
        decision=retry_decision,
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
    if event_type in {EventType.MODEL_HOSTED_TOOL_CALL, EventType.MODEL_CITATION}:
        event = event_with_runtime_payload_authority(event, "provider_operation_id")
    return event_with_execution_profile_fingerprint_authority(
        event,
        execution_profile_fingerprint,
    )


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
) -> tuple[int | None, bool | None, float | None, bool]:
    cause = exc.cause
    if exc.completion_observed or exc.automatic_retry_disabled:
        # A valid completed frame is the authoritative terminal attempt. A
        # later transport/control failure cannot authorize another provider
        # charge for the same logical step.
        status_code = cause.status_code if isinstance(cause, ModelProviderError) else None
        return status_code, False, None, False
    if isinstance(cause, ModelProviderError):
        return (
            cause.status_code,
            cause.retryable,
            cause.retry_after_s,
            cause.status_code is None and cause.retryable is None,
        )
    status_code = exc.payload.get("status_code")
    retryable = exc.payload.get("retryable")
    retry_after_s = exc.payload.get("retry_after_s")
    return (
        status_code if type(status_code) is int else None,
        retryable if type(retryable) is bool else None,
        float(retry_after_s) if type(retry_after_s) in {int, float} else None,
        False,
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
    provider_error_payload: dict[str, Any],
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
    for key in ("provider_error_type", "provider_error_code"):
        value = provider_error_payload.get(key)
        if type(value) is str:
            payload[key] = value
    retryable = provider_error_payload.get("retryable")
    if type(retryable) is bool:
        payload["retryable"] = retryable
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
                "effective_max_attempts": decision.effective_max_attempts,
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
    decision: RetryDecision | None = None,
) -> dict[str, Any]:
    enriched = dict(payload)
    strip_runtime_owned_execution_identity(enriched)
    enriched["step"] = step
    enriched["attempt"] = attempt
    enriched["max_attempts"] = max_attempts
    if decision is not None:
        if type(decision) is not RetryDecision:
            raise TypeError("decision must be a RetryDecision or None.")
        if decision.attempt != attempt or decision.max_attempts != max_attempts:
            raise ValueError("Retry decision does not match the model-attempt evidence.")
        enriched.pop("effective_max_attempts", None)
        enriched.pop("reason", None)
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

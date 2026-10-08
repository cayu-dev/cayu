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
import time
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable, Iterable, Mapping
from copy import deepcopy
from dataclasses import dataclass, field, replace
from datetime import datetime
from functools import partial
from hashlib import sha256
from typing import Any, Never, cast
from uuid import uuid4

from cayu._exception_groups import (
    add_exception_note_safely,
    exception_cause,
    exception_context,
    exception_suppresses_context,
    exception_tree_contains,
    iter_exception_tree,
    set_exception_cause,
    set_exception_context,
)
from cayu._task_wait import (
    _consume_detached_task_outcome,
    await_shielded_task_outcome,
    consume_pending_task_cancellation,
    unexpected_child_cancellation_error,
    wait_until_idle,
)
from cayu._validation import (
    DurableValueError,
    canonical_durable_json_bytes,
    copy_durable_json_object,
    copy_durable_metadata,
    copy_json_value,
    extract_durable_value_error,
    require_clean_nonblank,
    require_durable_clean_nonblank,
)
from cayu.agents import AgentSpec
from cayu.artifacts.attachments import (
    MODEL_FILE_ATTACHMENT_ATTESTATIONS_PAYLOAD_KEY,
    RESOLVED_FILE_ATTACHMENTS_OPTION,
    FileAttachment,
    file_attachment_from_payload,
    resolved_file_attachment,
)
from cayu.artifacts.base import InvalidArtifactIdError, copy_artifact_read_result
from cayu.budgets.base import (
    BudgetLimit,
    BudgetPolicy,
    BudgetReservationResult,
    budget_limits_for_session,
    copy_request_budget_limits,
    has_deferred_contextual_price,
)
from cayu.budgets.billing import BillingIdentity, resolved_billing_identity
from cayu.budgets.usage import (
    ModelCompletionPurpose,
    is_conversational_model_completion_payload,
    usage_metrics_from_event_payload,
)
from cayu.context.base import (
    _COMPACTION_ATTEMPT_ID_KEY,
    CompactionRequest,
    CompactionResult,
    ContextBuildError,
    ContextCompactionTelemetry,
    ContextCompactor,
    ContextInputCoverage,
    ContextPolicy,
    ContextPressureEstimate,
    ContextPressureOverhead,
    ContextRecallTelemetry,
    ContextRequest,
    ContextUsageState,
    RuntimeManagedContextPolicy,
    _attach_automatic_compaction_failure_disposition,
    _automatic_compaction_dispatch_runner_scope,
    _automatic_compaction_runner_scope,
    _AutomaticCompactionDispatchDisposition,
    _AutomaticCompactionFailureDisposition,
    _AutomaticCompactionFailureReason,
    _AutomaticCompactionLifecyclePhase,
    _AutomaticCompactionRecoveryAction,
    _AutomaticCompactionRunner,
    _compaction_completion_publisher_scope,
    _compaction_environment_admission_scope,
    _compaction_model_attempt_identity_scope,
    _context_recall_telemetry_publisher_scope,
    _context_secret_redactor_scope,
    _ContextCountAuthorityError,
    _defer_billing_identity_cancellation_scope,
    automatic_compaction_failure_disposition_payload,
    context_build_termination_checkpoint_error,
    context_build_termination_compaction_telemetry,
    context_input_coverage,
    copy_context_messages,
    estimate_context_pressure,
    noteify_unresolvable_prompt_files,
    project_runtime_managed_context_checkpoint,
    sanitize_context_build_error_checkpoint,
    sanitize_context_build_result_checkpoint,
    sanitize_context_compaction_telemetry,
)
from cayu.context.counting import ContextCountingConfig, ContextCountingMode
from cayu.context.footprints import (
    PromptContributionManifest,
    RequestFootprint,
    RequestFootprintConfig,
    RequestVariant,
    TargetedToolGrantFootprint,
    ToolDiscoveryViewFootprint,
    analyze_request_context_pressure,
    analyze_request_footprint,
    copy_request_footprint_config,
    targeted_tool_grant_footprint,
    tool_discovery_view_footprint,
)
from cayu.context.structured_output import (
    StructuredOutputSpec,
    StructuredOutputStrategy,
    copy_structured_output_spec,
    require_secret_free_structured_output_spec,
    structured_output_spec_payload,
    structured_output_tool_instruction,
    structured_output_tool_spec,
)
from cayu.context.thinking import ThinkingConfig, thinking_config_payload
from cayu.deadlines import current_execution_deadline
from cayu.events import (
    Event,
    EventType,
    event_with_runtime_envelope_authority,
    event_with_runtime_generated_id,
    event_with_runtime_payload_authority,
)
from cayu.execution_profiles import (
    ExecutionProfileIdentity,
    event_with_execution_profile_authority,
)
from cayu.execution_units import (
    ModelAttemptIdentity,
    ModelStepIdentity,
    copy_model_attempt_identity,
    copy_model_step_identity,
    new_model_step_identity,
    strip_runtime_owned_execution_identity,
)
from cayu.memory.evidence import ContextExposure, ContextExposureEvidenceKind, ContextExposureState
from cayu.messages import (
    FilePart,
    Message,
    MessageRole,
    ProviderStatePart,
    ToolResultPart,
    detach_message,
)
from cayu.providers._credential_boundary import (
    credential_safe_provider_cancellation,
)
from cayu.providers._http import (
    bind_provider_error_workload_redactor,
    reset_provider_error_workload_redactor,
)
from cayu.providers._thinking import copy_preflight_thinking
from cayu.providers.base import (
    CALL_TOOL_CORE_CALLABLE_OPTION,
    OPENAI_HOSTED_TOOL_SEARCH_PROTOCOL,
    TARGETED_TOOL_NATIVE_CACHE_ANCHOR_OPTION,
    InputTokenCountConfidence,
    InputTokenCountMethod,
    InputTokenCountResult,
    ModelCompletion,
    ModelContextOverflowError,
    ModelProvider,
    ModelProviderError,
    ModelRequest,
    ModelStreamDeadlineError,
    TargetedToolProjectionRequest,
    ToolDiscoveryProjectionRequest,
    ToolDiscoveryProjectionResult,
    UsageDialect,
    copy_input_token_count_result,
    copy_model_context_pressure_profile,
)
from cayu.providers.deadlines import (
    ProviderStreamDeadlineAdmission,
    bind_provider_deadline_admission,
    reset_provider_deadline_admission,
)
from cayu.providers.operations import (
    ProviderOperationAdapter,
    ProviderOperationMode,
    ProviderOperationSnapshot,
    ProviderOperationStartIdempotencySupport,
)
from cayu.providers.retry_policy import RetryPolicy, copy_retry_policy
from cayu.runtime import _runtime_records as runtime_records
from cayu.runtime import _tool_round_recovery as tool_round_recovery
from cayu.runtime._child_session_notifications import (
    CHILD_SESSION_NOTIFICATION_INTENT_KEY,
    ChildSessionNotificationStageBinding,
)
from cayu.runtime._environment_exposure import (
    refresh_and_require_environment_exposed,
    require_environment_exposed,
)
from cayu.runtime._event_writer import RuntimeEventWriter
from cayu.runtime._invocation_lifecycle import InvocationContext
from cayu.runtime._live_model_attempt import (
    LiveModelAttempt,
    _deadline_with_runtime_recovery_authority,
    _model_request_fingerprint,
)
from cayu.runtime._memory_evidence import (
    MemoryEvidenceKey,
    MemoryEvidenceReference,
    context_exposure_identity_payload,
    memory_evidence_key,
    memory_evidence_key_scope,
    memory_evidence_reference_from_checkpoint,
    prepare_context_exposure,
    transition_context_exposure,
)
from cayu.runtime._message_redaction import (
    redact_runtime_message_for_boundary,
)
from cayu.runtime._model_completion_contracts import (
    HostedToolDiscoveryRecoveryAuthority,
    ModelCompletionDispatch,
    ModelCompletionDispatchNotAuthorized,
    ModelCompletionDispatchPreparer,
    ModelCompletionPublisher,
    ModelCompletionRecoveryContext,
    ModelCompletionRecoveryContextFactory,
    _copy_model_completion_stage,
    model_completion_recovery_context_from_stage,
)
from cayu.runtime._model_completion_delivery import (
    ModelAttemptFailed,
    _combine_authoritative_model_failure,
    _combine_post_completion_failures,
)
from cayu.runtime._model_errors import (
    copy_provider_exception_control,
    model_provider_error_from_payload,
    nonportable_model_provider_error,
    resolve_request_billing_identity,
)
from cayu.runtime._model_event_authority import _event_with_model_identity_authority
from cayu.runtime._model_execution_selection import (
    ModelExecutionSelection,
    ModelFailoverAttempt,
    ModelFailoverTransition,
)
from cayu.runtime._model_failover import (
    FailoverDisposition,
    FailoverObservation,
    decide_model_failover,
)
from cayu.runtime._model_stream_events import (
    _retry_attempt_payload,
)
from cayu.runtime._model_target import project_portable_transcript
from cayu.runtime._model_tool_discovery import (
    _hosted_tool_discovery_projection,
    _hosted_tool_discovery_projection_digest,
    _hosted_tool_name_sha256,
    _redacted_provider_tool_definitions,
)
from cayu.runtime._phase_timing import timed_model_step, timed_phase
from cayu.runtime._provider_operation_cancellation_owner import (
    ProviderOperationCancellationOwner,
)
from cayu.runtime._provider_operation_recovery_owner import (
    ProviderOperationRecoveryOwner,
)
from cayu.runtime._provider_operation_start_owner import (
    ProviderOperationStartOwner,
)
from cayu.runtime._provider_stream import (
    _close_async_iterator,
)
from cayu.runtime._run_limits import (
    _TRUSTED_BINDING_PROVENANCE,
    UNKNOWN_POST_DISPATCH_BUDGET_REASON,
    BudgetDispatchReservationFailed,
    BudgetedOperationFailed,
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
from cayu.runtime.model_steps import (
    AssistantStepResult,
    assistant_text_content,
    classify_assistant_step,
    provider_state_count,
    thinking_count,
)
from cayu.runtime.provider_operation_cancellation import (
    ProviderOperationCancellationLifecycle,
)
from cayu.runtime.provider_operations import (
    ProviderOperationEvidenceError,
    ProviderOperationRecoveryResult,
    RecoverableProviderOperation,
    RecoverableProviderOperationStart,
    fallback_dispatch_ordinal_from_checkpoint,
)
from cayu.runtime.retry_policy import (
    RetryDecision,
    RetrySuppression,
    retry_decision,
    retry_diagnostic_payload,
    retry_event_payload,
)
from cayu.sessions import _model_completion_publication as model_completion_publication
from cayu.sessions import _pending_tool_round as pending_rounds
from cayu.sessions._execution_profile_checkpoint import (
    active_invocation_execution_profile_from_checkpoint,
)
from cayu.sessions._invocation_terminal_decision import (
    InvocationTerminalOutcome,
    invocation_terminal_decision_from_checkpoint,
)
from cayu.sessions._model_failover import (
    MODEL_FAILOVER_CHECKPOINT_KEY,
    ModelFailoverProgress,
    copy_model_failover_state,
)
from cayu.sessions._provider_operation_cancellation_claim import (
    ProviderOperationCancellationClaim,
)
from cayu.sessions._terminal_evidence import interruption_request_id_from_payload
from cayu.sessions.authority import SessionRunFenced
from cayu.sessions.base import (
    CheckpointTransform,
    EventOrder,
    EventQuery,
    ModelCompletionStage,
    ModelCompletionStageAbandonmentResult,
    ModelCompletionStageRequest,
    RuntimePublicationRequest,
    SessionStatusConflict,
    SessionStore,
    _current_session_interaction_id,
    runtime_publication_checkpoint_mutation,
)
from cayu.sessions.cleanup import RecoveryCleanupSupervisor
from cayu.sessions.records import Session, SessionStatus
from cayu.tools.catalogue import (
    CALL_TOOL_NAME,
    SEARCH_TOOLS_NAME,
    ToolCatalogSnapshot,
    ToolDescriptor,
)
from cayu.tools.discovery import (
    TOOL_DISCOVERY_VIEW_OPERATION_KEY,
    ToolDiscoveryMode,
    ToolDiscoveryProjectionKind,
    ToolDiscoverySearchMatch,
    _tool_discovery_definition_for_descriptor,
    current_tool_discovery_view,
    resolve_tool_discovery_projection,
    search_tools_spec,
    tool_discovery_generation_id,
    tool_discovery_record_matches_descriptor,
    tool_discovery_search_match_matches_descriptor,
)
from cayu.tools.exposure import (
    ALL_REGISTERED_TOOLS_PROFILE_ID,
    TOOL_EXPOSURE_PROFILE_ID_MAX_CHARS,
    AllRegisteredToolsExposurePolicy,
    ResolvedToolExposure,
    ToolExposure,
    ToolExposurePolicyRequest,
    resolve_tool_exposure,
    resolved_tool_exposure_authority,
    tool_capability_ceiling_from_session_metadata,
    tool_exposure_record,
)
from cayu.tools.gateway import (
    TargetedToolGatewayProjection,
    call_tool_spec,
    targeted_tool_gateway_projection,
)
from cayu.tools.grants import TargetedToolGrantRecord
from cayu.tools.targeted_projection import (
    TargetedToolProjectionKind,
    openai_targeted_tool_projection,
    persisted_targeted_tool_projection_marker_message,
    resolve_targeted_tool_projection,
    targeted_tool_projection_marker_id,
)
from cayu.vaults.redaction import SecretRedactor

logger = logging.getLogger(__name__)


class _ModelFailoverCandidateExhausted(Exception):
    """Private handoff after local retry exhaustion, never a public failure.

    The route owner must confirm settlement and classify this live failure before
    preparing a successor. Carry original exceptions; do not recreate authority
    from event text or scan an arbitrary cause chain for an old provider error.
    """

    def __init__(
        self,
        *,
        selection: ModelExecutionSelection,
        identity: ModelAttemptIdentity,
        failure: ModelAttemptFailed,
        decision: RetryDecision,
        provider_effect_observed: bool,
        provider_operation_mode: ProviderOperationMode,
    ) -> None:
        self.selection = selection
        self.identity = copy_model_attempt_identity(identity)
        self.failure = failure
        self.decision = RetryDecision.model_validate(
            {name: getattr(decision, name) for name in RetryDecision.model_fields}
        )
        self.provider_effect_observed = provider_effect_observed
        self.provider_operation_mode = provider_operation_mode
        super().__init__("Selected model candidate exhausted its same-provider retries.")


@dataclass(frozen=True, slots=True, repr=False)
class _ModelFailoverFinalEvidence:
    event: Event


class _ModelFailoverContextOverflow(Exception):
    """Carry the exact prepared attempt across the context-rebuild boundary."""

    def __init__(
        self,
        *,
        selection: ModelExecutionSelection,
        failure: ModelAttemptFailed,
        dispatch: ModelCompletionDispatch,
        provider_effect_observed: bool,
    ) -> None:
        if not isinstance(failure.cause, ModelContextOverflowError):
            raise TypeError("Context recovery requires a typed context-overflow failure.")
        selection.require_prepared_stage(dispatch.stage)
        self.selection = selection
        self.failure = failure
        self.dispatch = dispatch
        self.provider_effect_observed = provider_effect_observed
        self.progress = ModelFailoverProgress.model_validate(
            dispatch.stage.intent[MODEL_FAILOVER_CHECKPOINT_KEY]["successor"]
        )
        super().__init__("Selected model attempt requires context recovery.")


def _raise_terminal_model_attempt_failure(exc: ModelAttemptFailed) -> Never:
    if exc.cause is None:
        raise RuntimeError(exc.message) from exc
    authoritative_cause = exception_cause(exc.cause)
    if exc.automatic_retry_disabled and authoritative_cause is not None:
        raise exc.cause from authoritative_cause
    raise exc.cause from exc


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


def _model_completion_stage_intent(
    *,
    model_attempt_identity: ModelAttemptIdentity,
    provider_name: str,
    requested_model: str,
    source_transcript_cursor: int,
    request_fingerprint: str,
    recovery_context: ModelCompletionRecoveryContext | None,
    input_coverage: ContextInputCoverage | None = None,
    provider_operation_start: dict[str, Any] | None = None,
    context_exposure: dict[str, str] | None = None,
    child_session_notifications: ChildSessionNotificationStageBinding | None = None,
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
        if recovery_context.interaction_id is not None:
            intent["interaction_id"] = recovery_context.interaction_id
    if input_coverage is not None:
        intent["input_coverage"] = input_coverage.model_dump(mode="json")
    if provider_operation_start is not None:
        intent["provider_operation_start"] = copy_durable_json_object(
            provider_operation_start,
            "provider_operation_start",
        )
    if context_exposure is not None:
        intent["context_exposure"] = copy_durable_json_object(
            context_exposure,
            "context_exposure",
        )
    if child_session_notifications is not None:
        intent[CHILD_SESSION_NOTIFICATION_INTENT_KEY] = child_session_notifications.model_dump(
            mode="json"
        )
    return intent


async def _terminate_pre_dispatch_context_exposure(
    *,
    store: SessionStore,
    exposure: ContextExposure,
    failure: BaseException,
    evidence_ref: str,
) -> bool:
    """Record a locally conclusive pre-dispatch outcome without hiding its cause."""

    cancelled = exception_tree_contains(
        failure,
        (asyncio.CancelledError, GeneratorExit, SessionInterruptedByRequest),
    )
    try:
        durable_exposure = await store.load_context_exposure(
            exposure.session_id,
            exposure.exposure_id,
        )
        if durable_exposure is None:
            raise RuntimeError("Prepared context exposure disappeared before dispatch.")
        if durable_exposure.state.terminal:
            return True
        await transition_context_exposure(
            store=store,
            exposure=durable_exposure,
            state=(ContextExposureState.CANCELLED if cancelled else ContextExposureState.FAILED),
            evidence_kind=(
                ContextExposureEvidenceKind.CONCLUSIVE_CANCELLATION
                if cancelled
                else ContextExposureEvidenceKind.CONCLUSIVE_FAILURE
            ),
            evidence_ref=evidence_ref,
        )
        return True
    except BaseException as evidence_failure:
        add_exception_note_safely(
            failure,
            "Context-exposure pre-dispatch termination also failed: "
            f"{type(evidence_failure).__name__}.",
        )
        if not isinstance(evidence_failure, Exception):
            # A later cancellation or process-control failure is not diagnostic
            # evidence: deliver it to the runtime's control-flow handler.
            raise
        return False


def _classify_pre_dispatch_failure(failure: BaseException) -> BaseException:
    """Return the typed failure a pre-dispatch settlement reports to the runtime."""

    if not isinstance(failure, BudgetReservationLeaseLost):
        return failure
    classified = BudgetReservationLeaseLostBeforeModelDispatch(
        "Budget reservation lease was lost before model dispatch."
    )
    set_exception_cause(classified, failure)
    return classified


def _attach_unsettled_pre_dispatch_cleanup(
    failure: BaseException,
    supervision_failures: list[BaseException],
) -> None:
    """Record why settlement is unknown without discarding the failure's own cause."""

    existing = exception_cause(failure)
    if existing is None and not exception_suppresses_context(failure):
        existing = exception_context(failure)
    causes = list(supervision_failures)
    if existing is not None and not any(existing is cause for cause in causes):
        causes.append(existing)
    set_exception_cause(
        failure,
        (
            causes[0]
            if len(causes) == 1
            else BaseExceptionGroup("Pre-dispatch cleanup outcome is unknown.", causes)
        ),
    )
    add_exception_note_safely(
        failure,
        "Pre-dispatch cleanup did not settle: "
        + ", ".join(type(error).__name__ for error in supervision_failures)
        + ".",
    )


async def _raise_after_model_pre_dispatch_cleanup(
    cleanup: Callable[[], Awaitable[Never]],
    *,
    unsettled_failure: Callable[[], BaseException],
    supervisor: RecoveryCleanupSupervisor,
) -> Never:
    """Settle within shared cleanup bounds, retaining any outcome-unknown owner.

    ``unsettled_failure`` returns the failure as classified so far. It is raised
    when the supervisor stops waiting or cannot start cleanup, so a deadline or
    capacity refusal never replaces the outcome the runtime must handle.
    """

    delivered: list[BaseException] = []

    async def delivering_cleanup() -> Never:
        try:
            await cleanup()
        except BaseException as outcome:
            delivered.append(outcome)
            raise

    task = asyncio.create_task(
        supervisor.run_steps(
            steps=(("model pre-dispatch cleanup", delivering_cleanup),),
            shield_caller_cancellation=True,
        ),
        name="cayu-model-pre-dispatch-cleanup",
    )
    cancellation: asyncio.CancelledError | None = None
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError as signal:
            if cancellation is None:
                cancellation = signal
            # Keep the caller's request count intact. Its enclosing runtime
            # handler owns interruption classification; do not uncancel and
            # reissue a signal that could interrupt that handler again.
    errors = [error for _, error in task.result()]
    if not errors:
        raise RuntimeError("Pre-dispatch cleanup did not return its authoritative failure.")
    settled = [error for error in errors if any(error is outcome for outcome in delivered)]
    supervision_failures = [error for error in errors if not any(error is s for s in settled)]
    # Without the cleanup's own outcome, the supervisor timed out or refused to
    # start it. Fall back to the classification reached before that happened.
    failure = settled[0] if settled else unsettled_failure()
    if supervision_failures:
        _attach_unsettled_pre_dispatch_cleanup(failure, supervision_failures)
    if cancellation is not None and not isinstance(failure, asyncio.CancelledError):
        # Preserve transformed errors and retention diagnostics as the cause,
        # including BudgetReservationLeaseLostBeforeModelDispatch.
        raise cancellation from failure
    raise failure


async def _model_preparation_lost_to_interrupt(store: SessionStore, session: Session) -> bool:
    """Recognize an interrupt winner only for this exact session invocation."""

    current = await store.load(session.id)
    if (
        current is None
        or current.instance_id != session.instance_id
        or current.run_epoch != session.run_epoch
        or current.status is not SessionStatus.INTERRUPTING
    ):
        return False
    checkpoint = await store.load_checkpoint(session.id)
    decision = invocation_terminal_decision_from_checkpoint(checkpoint)
    marker = None if checkpoint is None else checkpoint.get("pending_session_interrupt")
    if marker is not None and not isinstance(marker, dict):
        return False
    request_id = None if marker is None else interruption_request_id_from_payload(marker)
    if decision is not None:
        return (
            decision.session_id == session.id
            and decision.session_instance_id == session.instance_id
            and decision.run_epoch == session.run_epoch
            and decision.outcome is InvocationTerminalOutcome.INTERRUPTED
            and (
                marker is None
                or (request_id is not None and decision.interruption_request_id == request_id)
            )
        )
    # Pending markers carry a request ID but no run epoch. Authenticate their
    # fallback through the active invocation in the same checkpoint instead of
    # accepting an arbitrary dictionary left by an older run.
    active_profile = active_invocation_execution_profile_from_checkpoint(checkpoint)
    return (
        request_id is not None
        and active_profile is not None
        and active_profile.session_id == session.id
        and active_profile.run_epoch == session.run_epoch
    )


def _combine_context_build_failure_with_secondary(
    error: ContextBuildError,
    secondary: BaseException,
    *,
    message: str,
) -> BaseException:
    """Keep a compaction deadline visible beside later context diagnostics."""

    if not any(
        isinstance(candidate, ModelStreamDeadlineError)
        for candidate in iter_exception_tree(error.cause)
    ):
        return secondary
    return _combine_authoritative_model_failure(
        error.cause,
        secondary,
        message=message,
    )


def _raise_context_build_cause(error: ContextBuildError) -> Never:
    """Unwrap a context failure without replacing its prior causal evidence."""

    authoritative = error.cause
    prior_cause = exception_cause(authoritative)
    if prior_cause is None or prior_cause is error:
        raise authoritative from error
    if exception_cause(error) is authoritative:
        set_exception_cause(error, None)
    if exception_context(error) is authoritative:
        set_exception_context(error, None)
    raise authoritative from BaseExceptionGroup(
        "Context build failure retained prior authoritative evidence.",
        [prior_cause, error],
    )


@dataclass(frozen=True)
class _ContextCountObservation:
    result: InputTokenCountResult
    observation_id: str


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
        "catalogue_revision",
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
        # A compaction provider call is an independently settleable effect. It
        # must not share the assistant turn's logical step because promoting a
        # successful context-compaction stage records a winner for that step.
        identity = new_model_step_identity().new_attempt()
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
class _AutomaticCompactionDispatchAuthority:
    """Durable stage ownership for one automatic-compaction provider call."""

    stage: ModelCompletionStage
    owns_stage: bool
    provider_name: str
    step: int
    attempt: int
    max_attempts: int

    def __post_init__(self) -> None:
        if type(self.stage) is not ModelCompletionStage:
            raise TypeError("Automatic compaction authority requires an exact stage.")
        object.__setattr__(self, "stage", self.stage.model_copy(deep=True))
        object.__setattr__(
            self,
            "provider_name",
            require_durable_clean_nonblank(
                self.provider_name,
                "automatic_compaction_provider_name",
            ),
        )
        if type(self.owns_stage) is not bool:
            raise TypeError("Automatic compaction stage ownership must be a bool.")
        for field_name, value in (
            ("step", self.step),
            ("attempt", self.attempt),
            ("max_attempts", self.max_attempts),
        ):
            if type(value) is not int or value < 1:
                raise ValueError(f"{field_name} must be a positive integer.")


@dataclass(frozen=True)
class ModelStepFlowOutcome:
    """Terminal outcome of one logical model step."""

    assistant_step_result: AssistantStepResult | None = None
    stop_session: bool = False
    portable_history_cursor: int | None = None

    def __post_init__(self) -> None:
        if self.stop_session == (self.assistant_step_result is not None):
            raise ValueError(
                "A model-step flow outcome must contain either a result or a stop signal."
            )
        if self.portable_history_cursor is not None and (
            type(self.portable_history_cursor) is not int or self.portable_history_cursor < 0
        ):
            raise ValueError("Portable model history requires a permanent transcript cursor.")


_CONTEXT_TERMINATION_PERSIST_TIMEOUT_S = 5.0
_COMPACTION_UNREPRESENTED_CALLS_KEY = "compaction_model_calls_unrepresented"
_CONTEXT_EVENT_STORE_WAIT_TIMEOUT_S = 5.0
_AUTOMATIC_COMPACTION_PREDISPATCH_EVENT_STORE_WAIT_TIMEOUT_S = 60.0
_CONTEXT_EVENT_STORE_WAIT_AFTER_CANCELLATION_TIMEOUT_S = 5.0
_AUTOMATIC_COMPACTION_PREDISPATCH_PUBLICATION_ATTEMPTS = 3
_CONTEXT_USAGE_AUXILIARY_PAGE_SIZE = 100


@dataclass
class _AutomaticCompactionLifecycle:
    """In-process phase evidence for one automatic compaction attempt."""

    started_at: float = field(default_factory=time.monotonic)
    provider_dispatch_disposition: _AutomaticCompactionDispatchDisposition = (
        _AutomaticCompactionDispatchDisposition.NOT_DISPATCHED
    )

    def failure_disposition(
        self,
        *,
        phase: _AutomaticCompactionLifecyclePhase,
        reason: _AutomaticCompactionFailureReason,
        retryable: bool,
        recovery_action: _AutomaticCompactionRecoveryAction,
    ) -> _AutomaticCompactionFailureDisposition:
        return _AutomaticCompactionFailureDisposition(
            phase=phase,
            reason=reason,
            elapsed_ms=max(0, int((time.monotonic() - self.started_at) * 1000)),
            retryable=retryable,
            provider_dispatch_disposition=self.provider_dispatch_disposition,
            recovery_action=recovery_action,
        )

    def attach_failure(
        self,
        error: BaseException,
        *,
        phase: _AutomaticCompactionLifecyclePhase,
        reason: _AutomaticCompactionFailureReason,
        retryable: bool,
        recovery_action: _AutomaticCompactionRecoveryAction,
    ) -> None:
        _attach_automatic_compaction_failure_disposition(
            error,
            self.failure_disposition(
                phase=phase,
                reason=reason,
                retryable=retryable,
                recovery_action=recovery_action,
            ),
        )


def _automatic_compaction_publication_failure_reason(
    error: BaseException,
) -> _AutomaticCompactionFailureReason:
    if isinstance(error, asyncio.CancelledError):
        return _AutomaticCompactionFailureReason.CANCELLED
    if isinstance(error, TimeoutError):
        return _AutomaticCompactionFailureReason.PUBLICATION_TIMEOUT
    return _AutomaticCompactionFailureReason.PUBLICATION_FAILED


def _automatic_compaction_publication_is_retryable(error: BaseException) -> bool:
    return isinstance(error, Exception) and not isinstance(error, (TypeError, ValueError))


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
    invocation_context: InvocationContext | None = None


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
    invocation_context: InvocationContext | None = None


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
    invocation_context: InvocationContext | None = None


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


class _AutomaticCompactionCheckpointRecoveryRequired(RuntimeError):
    """A completed compaction has no later durable context checkpoint."""


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
        recovery_cleanup_supervisor: RecoveryCleanupSupervisor,
        event_writer: RuntimeEventWriter,
        session_control: SessionControl[SessionUsageTracker],
        run_limit_controller: RunLimitController,
        context_counting: ContextCountingConfig,
        request_footprint: RequestFootprintConfig,
        max_file_attachment_bytes: int,
        max_total_file_attachment_bytes: int,
        max_file_attachments_per_request: int,
        secret_redactor: SecretRedactor,
        clock: Callable[[], datetime],
        checkpoint_transform: CheckpointTransformFactory,
        apply_budget_evaluation: BudgetEvaluationEventStream,
        apply_limit_evaluation: LimitEvaluationEventStream,
        stop_for_budget_reservation_failure: BudgetReservationFailureEventStream,
        provider_operation_cancellation_lifecycle: ProviderOperationCancellationLifecycle,
        peer_exposure_guard: Callable[
            [Session, ModelRequest, ModelAttemptIdentity, str, str, InvocationContext | None],
            contextlib.AbstractAsyncContextManager[None],
        ]
        | None = None,
    ) -> None:
        self._session_store = session_store
        self._recovery_cleanup_supervisor = recovery_cleanup_supervisor
        self._event_writer = event_writer
        self._session_control = session_control
        self._run_limit_controller = run_limit_controller
        self._context_counting = context_counting.model_copy(deep=True)
        self._request_footprint = copy_request_footprint_config(request_footprint)
        self._max_file_attachment_bytes = max_file_attachment_bytes
        self._max_total_file_attachment_bytes = max_total_file_attachment_bytes
        self._max_file_attachments_per_request = max_file_attachments_per_request
        self._secret_redactor = secret_redactor
        self._clock = clock
        self._checkpoint_transform = checkpoint_transform
        self._apply_budget_evaluation = apply_budget_evaluation
        self._apply_limit_evaluation = apply_limit_evaluation
        self._stop_for_budget_reservation_failure = stop_for_budget_reservation_failure
        self._peer_exposure_guard = peer_exposure_guard
        self._provider_operation_start = ProviderOperationStartOwner(
            session_store=session_store,
            event_writer=event_writer,
            cancellation_lifecycle=provider_operation_cancellation_lifecycle,
        )
        self._detached_tasks: set[asyncio.Task[Any]] = set()
        self._provider_operation_cancellation = ProviderOperationCancellationOwner(
            session_store=session_store,
            event_writer=event_writer,
            run_limit_controller=run_limit_controller,
            lifecycle=provider_operation_cancellation_lifecycle,
            read_recovery_context=model_completion_recovery_context_from_stage,
        )
        self._provider_operation_recovery = ProviderOperationRecoveryOwner(
            session_store=session_store,
            event_writer=event_writer,
            run_limit_controller=run_limit_controller,
            cancellation=self._provider_operation_cancellation,
            secret_redactor=secret_redactor,
            clock=clock,
        )
        self._live_model_attempt = LiveModelAttempt(
            session_store=session_store,
            event_writer=event_writer,
            session_control=session_control,
            secret_redactor=secret_redactor,
            clock=clock,
            provider_operation_start=self._provider_operation_start,
            provider_operation_cancellation=self._provider_operation_cancellation,
            provider_operation_recovery=self._provider_operation_recovery,
        )

    def _retain_detached_task(self, task: asyncio.Task[Any]) -> None:
        """Keep a cancelled store write that outlived its bounded wait until it settles.

        Its caller already received a timeout and either reconciles the write
        from durable state or treats it as best-effort, so how it ends is not
        reported again.
        """

        self._detached_tasks.add(task)

        def settled(completed: asyncio.Task[Any]) -> None:
            self._detached_tasks.discard(completed)
            _consume_detached_task_outcome(completed)

        task.add_done_callback(settled)

    def _running_detached_writes(self) -> set[asyncio.Future[Any]]:
        return set(self._detached_tasks)

    def _running_provider_reconciliations(self) -> set[asyncio.Future[Any]]:
        return self._provider_operation_start.running_reconciliations()

    @property
    def detached_writes_pending(self) -> bool:
        """Whether a kept store write still runs."""

        return any(not task.done() for task in self._detached_tasks)

    @property
    def provider_reconciliations_pending(self) -> bool:
        """Whether a reconciliation still runs or failed without being reported."""

        return self._provider_operation_start.reconciliation_failures.pending or any(
            not task.done() for task in self._running_provider_reconciliations()
        )

    async def drain_detached_writes(self, *, timeout_s: float) -> bool:
        """Wait up to ``timeout_s`` for detached store writes, without cancelling them."""

        return await wait_until_idle(self._running_detached_writes, timeout_s=timeout_s)

    @property
    def provider_cancellation_claims_pending(self) -> bool:
        """Whether a cancellation claim heartbeat or renewal write still runs."""

        return self._provider_operation_cancellation.pending

    async def wait_for_provider_cancellation_renewals(self, *, timeout_s: float) -> bool:
        """Wait up to ``timeout_s`` for claim renewal writes, without cancelling them."""

        return await wait_until_idle(
            self._provider_operation_cancellation.detached_renewals, timeout_s=timeout_s
        )

    async def stop_and_wait_for_provider_cancellation_claims(self, *, timeout_s: float) -> bool:
        """Stop orphaned claim heartbeats, then wait for them and their renewals.

        Only for shutdown once no operation is in flight: no live cancellation
        then needs a claim, and each stopped claim's lease expires.
        """

        self._provider_operation_cancellation.stop_all_heartbeats()
        return await wait_until_idle(
            self._provider_operation_cancellation.running, timeout_s=timeout_s
        )

    async def wait_for_provider_reconciliations(self, *, timeout_s: float) -> bool:
        """Wait up to ``timeout_s`` for provider-operation reconciliations to settle."""

        return await wait_until_idle(self._running_provider_reconciliations, timeout_s=timeout_s)

    def raise_provider_reconciliation_failures(self) -> None:
        """Raise, once, for reconciliations that failed unreported."""

        self._provider_operation_start.reconciliation_failures.raise_once()

    async def drain_provider_reconciliations(self, *, timeout_s: float) -> bool:
        """Wait for provider-operation reconciliations, then raise once for failures."""

        idle = await self.wait_for_provider_reconciliations(timeout_s=timeout_s)
        self.raise_provider_reconciliation_failures()
        return idle

    async def cancel_provider_operation_for_interruption(
        self,
        *,
        session: Session,
        stage: ModelCompletionStage,
        operation: RecoverableProviderOperation,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_provider: runtime_records.RegisteredProvider,
        environment_name: str | None,
        invocation_context: InvocationContext | None = None,
        model_execution_selection: ModelExecutionSelection | None = None,
    ) -> ProviderOperationSnapshot | None:
        """Cancel one durably identified operation after its worker disappears."""

        return await self._provider_operation_cancellation.cancel_for_interruption(
            session=session,
            stage=stage,
            operation=operation,
            registered_agent=registered_agent,
            registered_provider=registered_provider,
            environment_name=environment_name,
            invocation_context=invocation_context,
            model_execution_selection=model_execution_selection,
        )

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
        invocation_context: InvocationContext | None = None,
        model_execution_selection: ModelExecutionSelection | None = None,
    ) -> ProviderOperationRecoveryResult:
        """Recover start-only evidence without persisting or replaying a raw request."""

        return await self._provider_operation_recovery.recover_start(
            session=session,
            stage=stage,
            start=start,
            registered_agent=registered_agent,
            registered_provider=registered_provider,
            environment_name=environment_name,
            model_completion_publisher=model_completion_publisher,
            invocation_context=invocation_context,
            model_execution_selection=model_execution_selection,
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
        invocation_context: InvocationContext | None = None,
        model_execution_selection: ModelExecutionSelection | None = None,
    ) -> ProviderOperationRecoveryResult:
        """Retrieve and atomically publish one exact offline provider operation."""

        return await self._provider_operation_recovery.recover(
            session=session,
            stage=stage,
            operation=operation,
            registered_agent=registered_agent,
            registered_provider=registered_provider,
            environment_name=environment_name,
            recovery_context=recovery_context,
            model_completion_publisher=model_completion_publisher,
            invocation_context=invocation_context,
            model_execution_selection=model_execution_selection,
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
        invocation_context: InvocationContext | None = None,
        validate_live_model_semantics: Callable[[], None],
        initial_tool_exposure: ResolvedToolExposure | None = None,
        previous_tool_exposure_profile_id: str | None = None,
        targeted_tool_grants: TargetedToolGrantFootprint | None = None,
        interaction_id: str | None = None,
        model_completion_recovery_context_factory: (
            ModelCompletionRecoveryContextFactory | None
        ) = None,
        model_completion_publisher: ModelCompletionPublisher | None = None,
        model_execution_selection: ModelExecutionSelection | None = None,
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
            invocation_context=invocation_context,
            validate_live_model_semantics=validate_live_model_semantics,
            initial_tool_exposure=initial_tool_exposure,
            previous_tool_exposure_profile_id=previous_tool_exposure_profile_id,
            targeted_tool_grants=targeted_tool_grants,
            interaction_id=interaction_id,
            model_completion_recovery_context_factory=(
                model_completion_recovery_context_factory
                or (
                    lambda billing_identity, _reservations: ModelCompletionRecoveryContext(
                        billing_identity=billing_identity
                    )
                )
            ),
            model_completion_publisher=model_completion_publisher,
            model_execution_selection=model_execution_selection,
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
        targeted_tool_projection_kind: TargetedToolProjectionKind | None = None,
        targeted_tool_gateway: TargetedToolGatewayProjection | None = None,
        targeted_tool_native: TargetedToolProjectionRequest | None = None,
        tool_discovery_projection_kind: ToolDiscoveryProjectionKind | None = None,
        tool_discovery_native_tool_names: Iterable[str] | None = None,
        model_execution_selection: ModelExecutionSelection | None = None,
    ) -> ModelRequest:
        effective_model = session.model
        if model_execution_selection is not None:
            if type(model_execution_selection) is not ModelExecutionSelection:
                raise TypeError("Model request selection must be runtime-owned.")
            model_execution_selection.require_request_scope(
                invocation_context=model_execution_selection.invocation_context,
                session=session,
                registered_agent=registered_agent,
                registered_provider=model_execution_selection.registered_provider,
                execution_profile=model_execution_selection.invocation_context.profile,
                model=model_execution_selection.model,
            )
            effective_model = model_execution_selection.model
        resolved_tool_exposure = (
            _all_registered_tool_exposure(registered_agent)
            if tool_exposure is None
            else _require_frozen_tool_exposure(tool_exposure)
        )
        model_tools = _model_request_tools(
            tool_exposure=resolved_tool_exposure,
            structured_output=structured_output,
            targeted_tool_projection=targeted_tool_projection_kind,
            tool_discovery_mode=registered_agent.tool_discovery_mode,
        )
        if targeted_tool_gateway is not None and (
            targeted_tool_projection_kind is not TargetedToolProjectionKind.CALL_TOOL
        ):
            raise RuntimeError("Targeted gateway context requires the call_tool projection.")
        if targeted_tool_native is not None and (
            targeted_tool_projection_kind is not TargetedToolProjectionKind.OPENAI_ADDITIONAL_TOOLS
        ):
            raise RuntimeError(
                "Native targeted tools require the OpenAI additional_tools projection."
            )
        if targeted_tool_gateway is not None and targeted_tool_native is not None:
            raise RuntimeError("A model request cannot carry two targeted-tool projections.")
        if registered_agent.tool_discovery_mode is None:
            if tool_discovery_projection_kind is not None:
                raise RuntimeError("A discovery projection requires configured tool discovery.")
        elif tool_discovery_projection_kind is None:
            raise RuntimeError("Configured tool discovery requires a resolved projection.")
        discovery_native_tool_names = tuple(
            () if tool_discovery_native_tool_names is None else tool_discovery_native_tool_names
        )
        if any(type(name) is not str for name in discovery_native_tool_names):
            raise TypeError("Native discovery tool names must be strings.")
        if len(discovery_native_tool_names) != len(set(discovery_native_tool_names)):
            raise ValueError("Native discovery tool names must be unique.")
        if (
            tool_discovery_projection_kind
            not in {
                ToolDiscoveryProjectionKind.OPENAI_TOOL_SEARCH_CLIENT,
                ToolDiscoveryProjectionKind.OPENAI_TOOL_SEARCH_HOSTED,
            }
            and discovery_native_tool_names
        ):
            raise RuntimeError("Native discovery tools require the OpenAI projection.")
        model_messages = _model_request_messages(
            messages=context_messages,
            structured_output=structured_output,
        )
        discovery_native_tools: tuple[dict[str, Any], ...] = ()
        hosted_discovery_projection: ToolDiscoveryProjectionRequest | None = None
        if tool_discovery_projection_kind is ToolDiscoveryProjectionKind.OPENAI_TOOL_SEARCH_CLIENT:
            discovery_native_tools = _redacted_native_discovery_tools_with_schema_evidence(
                model_messages,
                authorized_names=discovery_native_tool_names,
                catalogue=registered_agent.tool_catalogue,
                redactor=self._secret_redactor,
            )
            discovery_native_tool_names = tuple(
                cast("str", tool["name"]) for tool in discovery_native_tools
            )
        elif (
            tool_discovery_projection_kind is ToolDiscoveryProjectionKind.OPENAI_TOOL_SEARCH_HOSTED
        ):
            targeted_native_names = (
                ()
                if targeted_tool_native is None
                else tuple(cast("str", tool["name"]) for tool in targeted_tool_native.tools)
            )
            hosted_discovery_projection = _hosted_tool_discovery_projection(
                session=session,
                registered_agent=registered_agent,
                excluded_tool_names=(
                    *resolved_tool_exposure.tool_names,
                    *targeted_native_names,
                ),
                redactor=self._secret_redactor,
            )
            replay_loaded_tools = _redacted_hosted_discovery_tools_with_replay_evidence(
                model_messages,
                authorized_names=discovery_native_tool_names,
                catalogue=registered_agent.tool_catalogue,
                redactor=self._secret_redactor,
            )
            if replay_loaded_tools:
                hosted_discovery_projection = ToolDiscoveryProjectionRequest.model_validate(
                    {
                        **hosted_discovery_projection.model_dump(mode="python"),
                        "loaded_tools": replay_loaded_tools,
                    }
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
        if targeted_tool_projection_kind is TargetedToolProjectionKind.OPENAI_ADDITIONAL_TOOLS:
            request_options[TARGETED_TOOL_NATIVE_CACHE_ANCHOR_OPTION] = CALL_TOOL_NAME
        if (
            targeted_tool_native is not None
            and tool_discovery_projection_kind is ToolDiscoveryProjectionKind.SEARCH_TOOLS
        ) or (
            targeted_tool_gateway is not None
            and tool_discovery_projection_kind
            in {
                ToolDiscoveryProjectionKind.OPENAI_TOOL_SEARCH_CLIENT,
                ToolDiscoveryProjectionKind.OPENAI_TOOL_SEARCH_HOSTED,
            }
        ):
            request_options[CALL_TOOL_CORE_CALLABLE_OPTION] = True
        redacted_messages = [
            redact_runtime_message_for_boundary(
                message,
                redactor=self._secret_redactor,
                field_name="model_message",
            )
            for message in model_messages
        ]
        if targeted_tool_gateway is not None:
            redacted_messages = _with_targeted_tool_gateway_instruction(
                redacted_messages,
                targeted_tool_gateway=targeted_tool_gateway,
                redactor=self._secret_redactor,
            )
        if self._secret_redactor.redact_text(effective_model) != effective_model:
            raise ValueError(
                "Model identity contains a workload secret and cannot be sent to a provider."
            )
        redacted_tools = _redacted_provider_tool_definitions(
            model_tools,
            redactor=self._secret_redactor,
            field_name="model_tools",
        )
        redacted_targeted_tool_projection = None
        if targeted_tool_native is not None:
            projected_targeted_tools = _redacted_provider_tool_definitions(
                targeted_tool_native.tools,
                redactor=self._secret_redactor,
                field_name="targeted_tools",
            )
            redacted_targeted_tool_projection = TargetedToolProjectionRequest(
                protocol=targeted_tool_native.protocol,
                marker_id=targeted_tool_native.marker_id,
                tools=tuple(projected_targeted_tools),
            )
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
                "content_sha256",
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
            model=effective_model,
            messages=redacted_messages,
            tools=redacted_tools,
            hosted_tools=registered_agent.hosted_tools,
            targeted_tool_projection=redacted_targeted_tool_projection,
            tool_discovery_projection=(
                ToolDiscoveryProjectionRequest(loaded_tools=discovery_native_tools)
                if tool_discovery_projection_kind
                is ToolDiscoveryProjectionKind.OPENAI_TOOL_SEARCH_CLIENT
                else hosted_discovery_projection
                if tool_discovery_projection_kind
                is ToolDiscoveryProjectionKind.OPENAI_TOOL_SEARCH_HOSTED
                else None
            ),
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
            Awaitable[tuple[list[Event], BudgetReservationResult | None, BaseException | None]],
        ],
        before_provider_dispatch: Callable[[ModelAttemptIdentity], Awaitable[None]],
        validate_live_model_semantics: Callable[[], None],
        refresh_live_model_semantics: Callable[[], Awaitable[None]],
        record_model_attempt_identity: Callable[[ModelAttemptIdentity], None],
        billing_identity: BillingIdentity | None = None,
        structured_output: StructuredOutputSpec | None = None,
        prepare_model_completion_dispatch: ModelCompletionDispatchPreparer | None = None,
        model_completion_publisher: ModelCompletionPublisher | None = None,
        execution_profile: ExecutionProfileIdentity | None = None,
        invocation_context: InvocationContext | None = None,
        tool_exposure: ResolvedToolExposure | None = None,
        tool_exposure_evidence: ToolExposure | None = None,
        targeted_tool_grants: TargetedToolGrantFootprint | None = None,
        targeted_tool_gateway: TargetedToolGatewayProjection | None = None,
        native_tool_grant_ids: Mapping[str, str] | None = None,
        memory_evidence_reference: MemoryEvidenceReference | None = None,
        child_session_notification_binding: ChildSessionNotificationStageBinding | None = None,
        model_execution_selection: ModelExecutionSelection | None = None,
        prior_provider_effect_observed: bool = False,
        context_overflow: _ModelFailoverContextOverflow | None = None,
    ) -> AsyncIterator[tuple[Event | None, AssistantStepResult | None]]:
        if type(prior_provider_effect_observed) is not bool:
            raise TypeError("Prior model effect observation must be a boolean.")
        provider_effect_observed = prior_provider_effect_observed
        if context_overflow is not None:
            if (
                type(context_overflow) is not _ModelFailoverContextOverflow
                or context_overflow.selection is not model_execution_selection
                or context_overflow.progress.logical_step_id != model_step_identity.model_step_id
            ):
                raise ValueError("Context recovery changed its selected attempt authority.")
            provider_effect_observed = (
                provider_effect_observed or context_overflow.provider_effect_observed
            )
        selected_dispatch: ModelCompletionDispatch | None = None
        if model_execution_selection is not None:
            if type(model_execution_selection) is not ModelExecutionSelection:
                raise TypeError("Model retry selection must be runtime-owned.")
            model_execution_selection.require_request_scope(
                invocation_context=invocation_context,
                session=session,
                registered_agent=registered_agent,
                registered_provider=registered_provider,
                execution_profile=execution_profile,
                model=model_request.model,
            )
            if prepare_model_completion_dispatch is None or model_completion_publisher is None:
                raise RuntimeError("Selected model execution requires durable completion stages.")
            if provider is not registered_provider.provider:
                raise ValueError("Model retry provider differs from its registered collaborator.")
            prepare_selected_stage = prepare_model_completion_dispatch
            selection = model_execution_selection

            async def prepare_checked_selection(
                request: ModelRequest,
                reference: MemoryEvidenceReference | None,
                notifications: ChildSessionNotificationStageBinding | None,
                consume_notifications: bool,
                *,
                failover_attempt: ModelFailoverAttempt | None = None,
            ) -> ModelCompletionDispatch:
                nonlocal selected_dispatch
                if failover_attempt is not None:
                    raise ValueError("Selected attempt evidence is owned by the retry loop.")
                expected_request_fingerprint = _model_request_fingerprint(
                    provider_name=selection.registered_provider.name,
                    model_request=request,
                )
                dispatch = await prepare_selected_stage(
                    request,
                    reference,
                    notifications,
                    consume_notifications,
                    failover_attempt=ModelFailoverAttempt(
                        selection=selection,
                        identity=model_attempt_identity,
                        prior_provider_effect_observed=provider_effect_observed,
                    ),
                )
                selection.require_prepared_stage(dispatch.stage)
                if dispatch.request_fingerprint != expected_request_fingerprint:
                    raise ValueError("Selected model stage belongs to a different request.")
                if any(
                    dispatch.stage.intent.get(key) != value
                    for key, value in model_attempt_identity.payload().items()
                ):
                    raise ValueError("Selected model stage belongs to a different live attempt.")
                selected_dispatch = dispatch
                return dispatch

            prepare_model_completion_dispatch = prepare_checked_selection
        elif invocation_context is not None and (
            invocation_context.binding.session_id != session.id
            or invocation_context.registered_agent is not registered_agent
            or invocation_context.registered_provider is not registered_provider
            or invocation_context.profile is not execution_profile
        ):
            raise RuntimeError("Model retry execution lost frozen invocation authority.")
        retry_policy = copy_retry_policy(retry_policy)
        request_variant = RequestVariant(request_variant)
        structured_output = copy_structured_output_spec(structured_output)
        model_step_identity = copy_model_step_identity(model_step_identity)
        resolved_tool_exposure = (
            None if tool_exposure is None else _require_frozen_tool_exposure(tool_exposure)
        )
        if targeted_tool_grants is not None:
            if type(targeted_tool_grants) is not TargetedToolGrantFootprint:
                raise TypeError(
                    "targeted_tool_grants must be a TargetedToolGrantFootprint or None."
                )
            targeted_tool_grants = TargetedToolGrantFootprint.model_validate(
                targeted_tool_grants.model_dump(mode="python")
            )
        if (
            targeted_tool_gateway is not None
            and type(targeted_tool_gateway) is not TargetedToolGatewayProjection
        ):
            raise TypeError(
                "targeted_tool_gateway must be a TargetedToolGatewayProjection or None."
            )
        native_grant_ids = {} if native_tool_grant_ids is None else dict(native_tool_grant_ids)
        if any(
            type(name) is not str or type(grant_id) is not str
            for name, grant_id in native_grant_ids.items()
        ):
            raise TypeError("native_tool_grant_ids must map strings to strings.")
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
        discovery_view_footprint: ToolDiscoveryViewFootprint | None = None
        if self._request_footprint.enabled and registered_agent.tool_discovery_mode is not None:
            capability_ceiling = tool_capability_ceiling_from_session_metadata(session.metadata)
            discovery_view = current_tool_discovery_view(
                await self._session_store.load_session_operation(
                    session.id,
                    TOOL_DISCOVERY_VIEW_OPERATION_KEY,
                ),
                session_id=session.id,
                generation_id=tool_discovery_generation_id(
                    session_id=session.id,
                    root_invocation_id=session.invocation.root_invocation_id,
                ),
                agent_name=registered_agent.spec.name,
                catalogue=registered_agent.tool_catalogue,
                ceiling=capability_ceiling,
            )
            discovery_view_footprint = tool_discovery_view_footprint(discovery_view)
        provider.preflight_model_target(model=model_request.model)
        selected_operation_mode = ProviderOperationMode.SYNCHRONOUS
        if model_execution_selection is not None:
            selected_operation_mode = provider.provider_operation_mode
            if type(selected_operation_mode) is not ProviderOperationMode:
                raise TypeError("Model provider operation mode must be typed.")
        provider.preflight_hosted_tools(
            model=model_request.model,
            hosted_tools=model_request.hosted_tools,
            options=model_request.options,
        )
        if model_execution_selection is not None:
            preflight_model_thinking(
                provider=provider,
                model=model_request.model,
                thinking=model_request.options.get("thinking"),
                redactor=self._secret_redactor,
            )
            preflight_portable_model_material(
                provider=provider,
                model=model_request.model,
                messages=model_request.messages,
                tools=[
                    *model_request.tools,
                    *(
                        ()
                        if model_request.targeted_tool_projection is None
                        else model_request.targeted_tool_projection.tools
                    ),
                    *(
                        ()
                        if model_request.tool_discovery_projection is None
                        else model_request.tool_discovery_projection.loaded_tools
                    ),
                ],
                redactor=self._secret_redactor,
            )
        if model_request.targeted_tool_projection is not None:
            provider.preflight_targeted_tool_projection(
                model=model_request.model,
                protocol=model_request.targeted_tool_projection.protocol,
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
        file_attachment_attestations = _model_file_attachment_attestations(model_request)
        while True:
            selected_dispatch = None
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
                if not isinstance(preparation_error, Exception):
                    raise preparation_error
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
            await refresh_live_model_semantics()
            # Never hand the retry template to provider-controlled code. Each
            # attempt gets a fully detached, revalidated request so provider
            # mutation cannot corrupt a later attempt.
            attempt_model_request = _detach_model_request(model_request)

            deadline_admission: ProviderStreamDeadlineAdmission | None = None
            attempt_events = None
            peer_exposure_stack = contextlib.AsyncExitStack()
            try:
                if self._peer_exposure_guard is not None:
                    await peer_exposure_stack.enter_async_context(
                        self._peer_exposure_guard(
                            session,
                            attempt_model_request,
                            model_attempt_identity,
                            registered_provider.name,
                            attempt_model_request.model,
                            invocation_context,
                        )
                    )
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
                    targeted_tool_grants=targeted_tool_grants,
                    tool_discovery_view=discovery_view_footprint,
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
                pre_count_completion_dispatch: ModelCompletionDispatch | None = None
                if (
                    memory_evidence_reference is not None
                    or child_session_notification_binding is not None
                    or model_execution_selection is not None
                ) and self._context_counting.mode is not ContextCountingMode.OFF:
                    if prepare_model_completion_dispatch is None:
                        raise RuntimeError(
                            "Automatic recall requires durable model-completion staging before "
                            "provider-backed context counting."
                        )
                    # Provider-backed counters receive the complete request and may
                    # perform network I/O. Commit the same durable dispatch/evidence
                    # fence used by the model call before handing them recalled
                    # context, then reuse that exact dispatch below.
                    deadline_admission = ProviderStreamDeadlineAdmission(provider.stream_deadlines)
                    await refresh_live_model_semantics()
                    pre_count_completion_dispatch = await prepare_model_completion_dispatch(
                        attempt_model_request,
                        memory_evidence_reference,
                        child_session_notification_binding,
                        False,
                    )
                    for prepared_event in pre_count_completion_dispatch.prepared_events:
                        yield prepared_event, None
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
                    refresh_live_model_semantics=refresh_live_model_semantics,
                )
                if context_count_event is not None:
                    yield context_count_event, None
                model_started = _event_with_model_identity_authority(
                    Event(
                        type=EventType.MODEL_STARTED,
                        session_id=session.id,
                        agent_name=registered_agent.spec.name,
                        payload={
                            "model": model_request.model,
                            "provider": registered_provider.name,
                            "step": step,
                            "attempt": attempt,
                            "max_attempts": retry_policy.max_attempts,
                            **(
                                {
                                    MODEL_FILE_ATTACHMENT_ATTESTATIONS_PAYLOAD_KEY: (
                                        file_attachment_attestations
                                    )
                                }
                                if file_attachment_attestations
                                else {}
                            ),
                            **model_attempt_identity.payload(),
                        },
                        environment_name=environment_name,
                    ),
                    model_attempt_identity,
                )
                if file_attachment_attestations:
                    model_started = event_with_runtime_payload_authority(
                        model_started,
                        MODEL_FILE_ATTACHMENT_ATTESTATIONS_PAYLOAD_KEY,
                    )
                yield (
                    await self._event_writer.emit(
                        event_with_execution_profile_authority(
                            model_started,
                            execution_profile,
                        )
                    ),
                    None,
                )
                if deadline_admission is None:
                    deadline_admission = ProviderStreamDeadlineAdmission(provider.stream_deadlines)
                assert deadline_admission is not None
                attempt_events = self._live_model_attempt.execute(
                    provider=provider,
                    deadline_admission=deadline_admission,
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
                    refresh_live_model_semantics=refresh_live_model_semantics,
                    billing_identity=billing_identity,
                    structured_output=structured_output,
                    context_pressure_estimate=request_context_pressure,
                    prepare_model_completion_dispatch=prepare_model_completion_dispatch,
                    model_completion_publisher=model_completion_publisher,
                    execution_profile=execution_profile,
                    invocation_context=invocation_context,
                    model_execution_selection=model_execution_selection,
                    tool_exposure=resolved_tool_exposure,
                    targeted_tool_gateway=targeted_tool_gateway,
                    native_tool_grant_ids=native_grant_ids,
                    memory_evidence_reference=memory_evidence_reference,
                    child_session_notification_binding=child_session_notification_binding,
                    prepared_model_completion_dispatch=pre_count_completion_dispatch,
                )
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
                                            model=model_request.model,
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
                                            model=model_request.model,
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
                provider_effect_observed = provider_effect_observed or exc.provider_effect_observed
                if isinstance(exc.cause, ModelContextOverflowError):
                    if model_execution_selection is None or selected_dispatch is None:
                        _raise_terminal_model_attempt_failure(exc)
                    raise _ModelFailoverContextOverflow(
                        selection=model_execution_selection,
                        failure=exc,
                        dispatch=selected_dispatch,
                        provider_effect_observed=provider_effect_observed,
                    ) from exc
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
                        suppression=_attempt_retry_suppression(exc),
                    )
                elif (
                    decision.attempt != attempt
                    or decision.max_attempts != retry_policy.max_attempts
                ):
                    raise RuntimeError(
                        "Model attempt retained a retry decision for different attempt authority."
                    ) from exc
                if not exc.emitted_error_event:
                    error_event = event_with_execution_profile_authority(
                        Event(
                            type=EventType.MODEL_ERROR,
                            session_id=session.id,
                            agent_name=registered_agent.spec.name,
                            environment_name=environment_name,
                            payload=_retry_attempt_payload(
                                exc.payload,
                                execution_provider_name=registered_provider.name,
                                requested_model=model_request.model,
                                step=step,
                                attempt=attempt,
                                max_attempts=retry_policy.max_attempts,
                                model_attempt_identity=model_attempt_identity,
                                decision=decision,
                            ),
                        ),
                        execution_profile,
                    )
                    try:
                        emitted_error = await self._event_writer.emit(error_event)
                    except Exception as publication_failure:
                        if not isinstance(exc.cause, ModelStreamDeadlineError):
                            raise
                        raise _combine_authoritative_model_failure(
                            exc.cause,
                            publication_failure,
                            message=(
                                "Model stream deadline and model.error publication both failed."
                            ),
                        ) from None
                    yield (
                        emitted_error,
                        None,
                    )
                total_attempts_exhausted = False
                if selected_dispatch is not None:
                    selected_progress = ModelFailoverProgress.model_validate(
                        selected_dispatch.stage.intent[MODEL_FAILOVER_CHECKPOINT_KEY]["successor"]
                    )
                    total_attempts_exhausted = (
                        selected_progress.attempts_used == selected_progress.plan.max_total_attempts
                    )
                if not decision.retry or total_attempts_exhausted:
                    if model_execution_selection is not None:
                        raise _ModelFailoverCandidateExhausted(
                            selection=model_execution_selection,
                            identity=model_attempt_identity,
                            failure=exc,
                            decision=decision,
                            provider_effect_observed=provider_effect_observed,
                            provider_operation_mode=selected_operation_mode,
                        ) from exc
                    _raise_terminal_model_attempt_failure(exc)
                yield (
                    await self._event_writer.emit(
                        event_with_execution_profile_authority(
                            _model_retry_event(
                                session=session,
                                model=model_request.model,
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
                                model=model_request.model,
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
                try:
                    if attempt_events is not None:
                        await _close_async_iterator(attempt_events)
                finally:
                    try:
                        if deadline_admission is not None:
                            deadline_admission.close()
                    finally:
                        primary_failure = sys.exception() or prior_retry_failure
                        try:
                            await peer_exposure_stack.aclose()
                        except BaseException as cleanup_failure:
                            if primary_failure is None or primary_failure is cleanup_failure:
                                raise
                            if isinstance(primary_failure, asyncio.CancelledError):
                                raise primary_failure from cleanup_failure
                            raise _combine_post_completion_failures(
                                primary_failure, cleanup_failure
                            ) from None

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
        targeted_tool_grants: TargetedToolGrantFootprint | None,
        tool_discovery_view: ToolDiscoveryViewFootprint | None,
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
            targeted_tool_grants=targeted_tool_grants,
            tool_discovery_view=tool_discovery_view,
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

    @timed_phase("counting")
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

    @timed_phase("counting")
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
        refresh_live_model_semantics: Callable[[], Awaitable[None]],
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
        await refresh_live_model_semantics()
        try:
            from cayu.providers.base import has_peer_content

            # Optional verification must not create a second disclosure path.
            # Counting has no model-attempt receiver or exposure receipt.
            redactor_token = bind_provider_error_workload_redactor(self._secret_redactor)
            try:
                provider_result = (
                    None
                    if has_peer_content(count_request)
                    else await provider.count_input_tokens(count_request)
                )
            finally:
                reset_provider_error_workload_redactor(redactor_token)
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

    async def _sleep_before_retry(self, session_id: str, decision: RetryDecision) -> None:
        await self._session_control.raise_if_interrupted(session_id)
        if decision.delay_seconds > 0:
            await asyncio.sleep(decision.delay_seconds)
        await self._session_control.raise_if_interrupted(session_id)


@dataclass(frozen=True)
class _TargetedToolModelProjection:
    """Request-local rendering and evidence from one authenticated grant read."""

    gateway: TargetedToolGatewayProjection | None = None
    native: TargetedToolProjectionRequest | None = None
    native_grant_ids: dict[str, str] = field(default_factory=dict)
    native_marker: Message | None = None
    footprint: TargetedToolGrantFootprint | None = None


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
        invocation_context: InvocationContext | None,
        validate_live_model_semantics: Callable[[], None],
        initial_tool_exposure: ResolvedToolExposure | None,
        previous_tool_exposure_profile_id: str | None,
        targeted_tool_grants: TargetedToolGrantFootprint | None,
        interaction_id: str | None,
        model_completion_recovery_context_factory: ModelCompletionRecoveryContextFactory,
        model_completion_publisher: ModelCompletionPublisher | None = None,
        model_execution_selection: ModelExecutionSelection | None = None,
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
        self._request_metadata = copy_durable_metadata(request_metadata, "metadata")
        self._retry_policy = copy_retry_policy(retry_policy)
        self._request_budget_limits = copy_request_budget_limits(request_budget_limits)
        self._limit_gate = limit_gate
        self._budget_policy = budget_policy
        self._run_started_at = run_started_at
        self._turn_usage_tracker = turn_usage_tracker
        self._active_run = active_run
        self._execution_profile = execution_profile
        if invocation_context is not None and (
            invocation_context.binding.session_id != session.id
            or invocation_context.registered_agent is not registered_agent
            or invocation_context.registered_provider is not registered_provider
            or invocation_context.registered_environment is not registered_environment
            or invocation_context.profile is not execution_profile
        ):
            raise ValueError("Model-step execution lost frozen invocation authority.")
        self._invocation_context = invocation_context
        if model_execution_selection is not None:
            if type(model_execution_selection) is not ModelExecutionSelection:
                raise TypeError("Model-step selection must be runtime-owned.")
            model_execution_selection.require_request_scope(
                invocation_context=invocation_context,
                session=session,
                registered_agent=registered_agent,
                registered_provider=model_execution_selection.registered_provider,
                execution_profile=execution_profile,
                model=model_execution_selection.model,
            )
        self._model_execution_selection = model_execution_selection

        def validate_live_execution_semantics() -> None:
            validate_live_model_semantics()
            if self._registered_environment is None:
                return
            if self._invocation_context is None or self._execution_profile is None:
                raise RuntimeError(
                    "Environment-backed model execution requires frozen exposure authority."
                )
            require_environment_exposed(
                self._registered_environment,
                session=self._session,
                invocation_context=self._invocation_context,
                registered_agent=self._registered_agent,
                execution_profile=self._execution_profile,
            )

        async def refresh_live_execution_semantics() -> None:
            validate_live_execution_semantics()
            if self._registered_environment is None:
                return
            assert self._invocation_context is not None
            assert self._execution_profile is not None
            await refresh_and_require_environment_exposed(
                self._registered_environment,
                session=self._session,
                invocation_context=self._invocation_context,
                registered_agent=self._registered_agent,
                execution_profile=self._execution_profile,
                redactor=self._executor._secret_redactor,
            )
            validate_live_execution_semantics()

        self._validate_live_model_semantics = validate_live_execution_semantics
        self._refresh_live_model_semantics = refresh_live_execution_semantics
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
        if targeted_tool_grants is not None:
            if type(targeted_tool_grants) is not TargetedToolGrantFootprint:
                raise TypeError(
                    "targeted_tool_grants must be a TargetedToolGrantFootprint or None."
                )
            targeted_tool_grants = TargetedToolGrantFootprint.model_validate(
                targeted_tool_grants.model_dump(mode="python")
            )
        self._targeted_tool_grants = targeted_tool_grants
        self._targeted_tool_grant_ids = (
            () if targeted_tool_grants is None else targeted_tool_grants.grant_ids
        )
        self._targeted_tool_generation_id = (
            None if targeted_tool_grants is None else targeted_tool_grants.generation_id
        )
        if targeted_tool_grants is None:
            if interaction_id is not None:
                interaction_id = require_durable_clean_nonblank(
                    interaction_id,
                    "interaction_id",
                )
        else:
            if interaction_id is None:
                raise ValueError("Targeted tool grants require an active interaction identity.")
            interaction_id = require_durable_clean_nonblank(interaction_id, "interaction_id")
        self._interaction_id = interaction_id
        self._refresh_request_configuration()

    def _refresh_request_configuration(self) -> None:
        """Recompute target-dependent projections without replacing root authority."""

        self._targeted_tool_projection_kind = resolve_targeted_tool_projection(
            self._registered_agent.targeted_tool_mode,
            provider=self._request_provider,
            model=self._request_model,
        )
        self._tool_discovery_projection_kind = resolve_tool_discovery_projection(
            self._registered_agent.tool_discovery_mode,
            provider=self._request_provider,
            model=self._request_model,
        )
        if (
            self._targeted_tool_grants is not None
            and self._targeted_tool_grants.projection is not self._targeted_tool_projection_kind
            and self._model_execution_selection is None
        ):
            raise ValueError(
                "Targeted grant footprint conflicts with the resolved provider projection."
            )
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
                provider_name=(
                    self._request_provider.billing_provider_name
                    or self._request_registered_provider.name
                ),
                model=self._request_model,
            )
            for limit in contextual_limits
        )

    @property
    def _request_registered_provider(self) -> runtime_records.RegisteredProvider:
        return (
            self._registered_provider
            if self._model_execution_selection is None
            else self._model_execution_selection.registered_provider
        )

    @property
    def _request_provider(self) -> ModelProvider:
        return (
            self._provider
            if self._model_execution_selection is None
            else self._model_execution_selection.registered_provider.provider
        )

    @property
    def _request_model(self) -> str:
        return (
            self._session.model
            if self._model_execution_selection is None
            else self._model_execution_selection.model
        )

    @property
    def execution_profile(self) -> ExecutionProfileIdentity | None:
        """Return the exact immutable profile resolved for this invocation."""

        return self._execution_profile

    def rebind_queued_interaction(self, invocation_context: InvocationContext) -> None:
        """Install the store-authenticated same-epoch context before the next step."""

        current = self._invocation_context
        if (
            current is None
            or type(invocation_context) is not InvocationContext
            or current.binding.session_id != invocation_context.binding.session_id
            or current.binding.session_instance_id != invocation_context.binding.session_instance_id
            or current.binding.run_epoch != invocation_context.binding.run_epoch
            or current.profile is not invocation_context.profile
            or current.tool_capability_ceiling != invocation_context.tool_capability_ceiling
            or current.binding.interaction_id == invocation_context.binding.interaction_id
        ):
            raise ValueError("Model-step queued handoff lost frozen invocation authority.")
        selection = self._model_execution_selection
        if selection is not None:
            selection = replace(selection, invocation_context=invocation_context)
        self._invocation_context = invocation_context
        self._model_execution_selection = selection
        self._interaction_id = invocation_context.binding.interaction_id

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
            provider_name=self._request_registered_provider.name,
            model=self._request_model,
            step=step,
            transcript_cursor=transcript_cursor,
            catalogue_revision=self._registered_agent.tool_catalogue.revision,
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

    async def _targeted_tool_projection_records(
        self,
    ) -> tuple[TargetedToolGrantRecord, ...]:
        """Load and validate the invocation's exact durable grant batch."""
        if not self._targeted_tool_grant_ids:
            return ()
        if self._interaction_id is None:
            raise RuntimeError("Targeted tool projection lost its interaction identity.")
        records = await self._executor._session_store.list_targeted_tool_grants(
            self._session.id,
            interaction_id=self._interaction_id,
        )
        records_by_id = {record.grant_id: record for record in records}
        if len(records_by_id) != len(records):
            raise RuntimeError("Targeted projection durable state contains duplicate grants.")
        expected_grant_ids = frozenset(self._targeted_tool_grant_ids)
        if not expected_grant_ids <= records_by_id.keys():
            raise RuntimeError("Targeted projection grant batch conflicts with durable state.")
        observed_at = self._executor._clock()
        omitted_records = tuple(
            record
            for grant_id, record in records_by_id.items()
            if grant_id not in expected_grant_ids
        )
        if any(observed_at < record.expires_at for record in omitted_records):
            raise RuntimeError("Targeted projection grant batch omitted a callable durable grant.")
        ordered_records = tuple(
            records_by_id[grant_id] for grant_id in self._targeted_tool_grant_ids
        )
        expected_generation = self._targeted_tool_generation_id
        if expected_generation is None:
            raise RuntimeError("Targeted projection lost its generation identity.")
        if any(
            record.interaction_id != self._interaction_id
            or record.generation_id != expected_generation
            or record.catalogue_revision != self._registered_agent.tool_catalogue.revision
            or record.tool_name not in self._tool_capability_ceiling.tool_names
            for record in ordered_records
        ):
            raise RuntimeError("Targeted projection grant scope conflicts with live authority.")
        return ordered_records

    async def _targeted_tool_projections(
        self,
    ) -> _TargetedToolModelProjection:
        records = await self._targeted_tool_projection_records()
        if not records:
            return _TargetedToolModelProjection()
        footprint = (
            targeted_tool_grant_footprint(records, projection=self._targeted_tool_projection_kind)
            if self._model_execution_selection is not None
            else self._targeted_tool_grants
        )
        observed_at = self._executor._clock()
        if self._targeted_tool_projection_kind is TargetedToolProjectionKind.CALL_TOOL:
            return _TargetedToolModelProjection(
                gateway=targeted_tool_gateway_projection(
                    records,
                    catalogue=self._registered_agent.tool_catalogue,
                    observed_at=observed_at,
                ),
                footprint=footprint,
            )
        if (
            self._targeted_tool_projection_kind
            is TargetedToolProjectionKind.OPENAI_ADDITIONAL_TOOLS
        ):
            projection, grant_ids_by_name = openai_targeted_tool_projection(
                records,
                catalogue=self._registered_agent.tool_catalogue,
            )
            return _TargetedToolModelProjection(
                native=projection,
                native_grant_ids=grant_ids_by_name,
                native_marker=persisted_targeted_tool_projection_marker_message(records)
                if self._model_execution_selection is not None
                else None,
                footprint=footprint,
            )
        raise RuntimeError("Targeted grants have no resolved provider projection.")

    async def _native_tool_discovery_grant_ids(
        self,
        *,
        tool_exposure: ResolvedToolExposure,
    ) -> dict[str, str]:
        """Load callable native names from the branch's current durable discovery view."""

        if self._tool_discovery_projection_kind not in {
            ToolDiscoveryProjectionKind.OPENAI_TOOL_SEARCH_CLIENT,
            ToolDiscoveryProjectionKind.OPENAI_TOOL_SEARCH_HOSTED,
        }:
            return {}
        view = current_tool_discovery_view(
            await self._executor._session_store.load_session_operation(
                self._session.id,
                TOOL_DISCOVERY_VIEW_OPERATION_KEY,
            ),
            session_id=self._session.id,
            generation_id=tool_discovery_generation_id(
                session_id=self._session.id,
                root_invocation_id=self._session.invocation.root_invocation_id,
            ),
            agent_name=self._registered_agent.spec.name,
            catalogue=self._registered_agent.tool_catalogue,
            ceiling=self._tool_capability_ceiling,
        )
        directly_exposed = frozenset(tool_exposure.tool_names)
        grants: dict[str, str] = {}
        for record in view.grants:
            descriptor = self._registered_agent.tool_catalogue.descriptor_for_id(record.tool_id)
            if not tool_discovery_record_matches_descriptor(record, descriptor):
                raise RuntimeError("Native discovery grant has stale descriptor authority.")
            if record.tool_name in directly_exposed:
                continue
            if record.tool_name in grants:
                raise RuntimeError("Native discovery view repeats a tool name.")
            grants[record.tool_name] = record.grant_id
        return grants

    async def _abandon_pre_dispatch_model_stage(
        self,
        stage: ModelCompletionStage,
        *,
        authoritative_failure: BaseException,
        budget_dispatch_id: str | None = None,
    ) -> None:
        """Clear one provably undispatched stage without losing its root failure."""

        try:
            dispatch = await self._executor._session_store.load_model_completion_stage_dispatch(
                self._session.id,
                stage.stage_id,
            )
            if dispatch is not None:
                authoritative_failure.add_note(
                    "The prepared model-completion stage was retained because its exact "
                    "dispatch receipt is durable."
                )
                return
            recovery_context = model_completion_recovery_context_from_stage(stage)
            await self._executor._run_limit_controller.release_pre_provider_dispatch_reservations(
                reservation_ids=stage.reservation_ids,
                recovery_contexts=(
                    () if recovery_context is None else recovery_context.budget_reservations
                ),
                dispatch_id=(
                    stage.stage_id
                    if budget_dispatch_id is None
                    else require_clean_nonblank(budget_dispatch_id, "budget_dispatch_id")
                ),
            )
        except BaseException as release_error:
            authoritative_failure.add_note(
                "Pre-dispatch model-completion budget release also failed: "
                f"{type(release_error).__name__}: {release_error}"
            )
            return

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

    @timed_model_step
    async def execute(
        self,
        *,
        step: int,
        messages: list[Message],
        source_transcript_cursor: int,
        model_step_identity: ModelStepIdentity,
        request_variant: RequestVariant = RequestVariant.INITIAL,
    ) -> AsyncGenerator[tuple[Event | None, ModelStepFlowOutcome | None], None]:
        task = asyncio.current_task()
        cancellation_baseline = 0 if task is None else task.cancelling()
        fallback: ModelFailoverTransition | None = None
        portable_history_cursor: int | None = None
        while True:
            candidate_events = self._execute_candidate(
                step=step,
                messages=messages,
                source_transcript_cursor=source_transcript_cursor,
                model_step_identity=model_step_identity,
                request_variant=request_variant,
                fallback=fallback,
            )
            exhausted: _ModelFailoverCandidateExhausted | None = None
            try:
                async for event, outcome in candidate_events:
                    if outcome is not None and portable_history_cursor is not None:
                        outcome = replace(outcome, portable_history_cursor=portable_history_cursor)
                    yield event, outcome
            except _ModelFailoverCandidateExhausted as failure:
                exhausted = failure
            finally:
                # This is our own candidate generator. Do not use the tolerant
                # provider-disposal helper: failed cleanup forbids fallback.
                try:
                    await candidate_events.aclose()
                except (asyncio.CancelledError, GeneratorExit):
                    raise
                except BaseException as cleanup_failure:
                    if exhausted is None:
                        raise
                    raise _combine_authoritative_model_failure(
                        exhausted.failure.cause or exhausted.failure,
                        cleanup_failure,
                        message="Model candidate failure and cleanup both failed.",
                    ) from None
            if exhausted is None:
                return

            # Candidate iteration has finished its provider cleanup and budget
            # settlement. A new cancellation or session interruption still wins
            # before even preparing another target.
            try:
                await asyncio.sleep(0)
                if task is not None and task.cancelling() > cancellation_baseline:
                    raise credential_safe_provider_cancellation(
                        "Model failover cancelled", preserve_empty_artifacts=False
                    )
                await self._executor._session_control.raise_if_interrupted(self._session.id)
                current_execution_deadline().require_admission("model_dispatch")
                try:
                    disposition = await self._candidate_disposition(exhausted)
                except Exception as transition_failure:
                    raise _combine_authoritative_model_failure(
                        exhausted.failure.cause or exhausted.failure,
                        transition_failure,
                        message="Model candidate failure and transition validation both failed.",
                    ) from None
            except (asyncio.CancelledError, GeneratorExit) as control:
                raise control from (exhausted.failure.cause or exhausted.failure)
            if isinstance(disposition, _ModelFailoverFinalEvidence):
                yield disposition.event, None
                _raise_terminal_model_attempt_failure(exhausted.failure)
            fallback = disposition
            if fallback is None:
                _raise_terminal_model_attempt_failure(exhausted.failure)
            selection = exhausted.selection
            self._model_execution_selection = ModelExecutionSelection(
                invocation_context=selection.invocation_context,
                resolution=selection.resolution,
                candidate_index=selection.candidate_index + 1,
            )
            self._refresh_request_configuration()
            # No output was accepted anywhere in this logical step. All supplied
            # history therefore precedes this switch. Keep semantic turns/files;
            # never forward another provider's response IDs or signed thinking.
            messages = list(project_portable_transcript(messages).messages)
            portable_history_cursor = source_transcript_cursor

    async def _candidate_disposition(
        self, exhausted: _ModelFailoverCandidateExhausted
    ) -> ModelFailoverTransition | _ModelFailoverFinalEvidence | None:
        if exhausted.selection is not self._model_execution_selection:
            raise ValueError("Candidate failure belongs to another model selection.")
        checkpoint = await self._executor._session_store.load_checkpoint(self._session.id)
        if checkpoint is None or MODEL_FAILOVER_CHECKPOINT_KEY not in checkpoint:
            raise RuntimeError("Candidate failure lost its durable route preparation.")
        progress = ModelFailoverProgress.model_validate(checkpoint[MODEL_FAILOVER_CHECKPOINT_KEY])
        observation = FailoverObservation(
            provider_name=exhausted.selection.registered_provider.name,
            caller_cancelled=False,
            completion_observed=exhausted.failure.completion_observed,
            provider_effect_observed=exhausted.provider_effect_observed,
            provider_operation_owned=(
                exhausted.provider_operation_mode is not ProviderOperationMode.SYNCHRONOUS
            ),
            cleanup_settled=True,
        )
        failure = exhausted.failure.cause or exhausted.failure
        decision = decide_model_failover(
            failure=failure,
            provider_name=exhausted.selection.registered_provider.name,
            retry=exhausted.decision,
            observation=observation,
            candidate_index=exhausted.selection.candidate_index,
            candidate_count=len(exhausted.selection.resolution.plan.candidates),
            attempts_used=progress.attempts_used,
            max_total_attempts=exhausted.selection.resolution.plan.max_total_attempts,
        )
        if decision.disposition is FailoverDisposition.SUPPRESSED:
            return None
        active = await self._executor._session_store.load_active_model_completion_stage(
            self._session.id
        )
        if active is None or active.stage.stage_id != progress.stage_id:
            raise RuntimeError("Candidate failure lost its active model preparation.")
        stage = active.stage
        exhausted.selection.require_prepared_stage(stage)
        if (
            stage.state != "in_flight"
            or stage.intent[MODEL_FAILOVER_CHECKPOINT_KEY]["successor"] != progress.payload()
            or any(
                stage.intent.get(key) != value
                for key, value in exhausted.identity.payload().items()
            )
        ):
            raise ValueError("Candidate failure conflicts with its exact active attempt.")
        if decision.disposition is FailoverDisposition.EXHAUSTED:
            target = progress.plan.candidates[progress.candidate_index]
            event = Event(
                id="evt_failover_exhausted_" + stage.preparation_digest,
                type=EventType.MODEL_FAILOVER_EXHAUSTED,
                timestamp=self._executor._clock(),
                session_id=self._session.id,
                interaction_id=progress.interaction_id,
                agent_name=self._session.agent_name,
                environment_name=self._session.environment_name,
                payload={
                    "schema_version": 1,
                    "route_id": progress.route_id,
                    "route_generation": progress.generation,
                    "stage_id": progress.stage_id,
                    **exhausted.identity.payload(),
                    "provider": target.provider_name,
                    "model": target.model,
                    "configured_provider": progress.plan.candidates[0].provider_name,
                    "configured_model": progress.plan.candidates[0].model,
                    "candidate_index": progress.candidate_index,
                    "candidate_count": len(progress.plan.candidates),
                    "attempts_used": progress.attempts_used,
                    "max_total_attempts": progress.plan.max_total_attempts,
                    "reason": (
                        "attempt_limit"
                        if progress.attempts_used == progress.plan.max_total_attempts
                        else "candidate_chain"
                    ),
                },
            )
            event = event_with_runtime_generated_id(
                _event_with_model_identity_authority(event, exhausted.identity)
            )
            event = event_with_execution_profile_authority(
                event, exhausted.selection.invocation_context.profile
            )
            event = event_with_runtime_envelope_authority(event, "session_id", "interaction_id")
            persisted = await self._executor._event_writer.persist_exact_replay(event)
            await self._executor._event_writer.fan_out_persisted([persisted])
            return _ModelFailoverFinalEvidence(persisted)
        if decision.disposition is not FailoverDisposition.SELECT_NEXT:
            raise ValueError("Unknown model failover disposition.")
        return ModelFailoverTransition(
            source=progress,
            source_preparation_digest=stage.preparation_digest,
            failure=failure,
            retry=exhausted.decision,
            observation=observation,
        )

    async def _execute_candidate(
        self,
        *,
        step: int,
        messages: list[Message],
        source_transcript_cursor: int,
        model_step_identity: ModelStepIdentity,
        request_variant: RequestVariant,
        fallback: ModelFailoverTransition | None,
    ) -> AsyncGenerator[tuple[Event | None, ModelStepFlowOutcome | None], None]:
        if self._registered_environment is not None:
            if self._invocation_context is None or self._execution_profile is None:
                raise RuntimeError(
                    "Environment-backed model execution requires frozen exposure authority."
                )
            await self._refresh_live_model_semantics()
            from cayu.workspaces.checkpoint_lifecycle import ensure_workspace_checkpoint

            await ensure_workspace_checkpoint(
                self._executor._session_store, self._session, self._registered_environment
            )
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
        targeted_projection = await self._targeted_tool_projections()
        targeted_tool_gateway = targeted_projection.gateway
        targeted_tool_native = targeted_projection.native
        targeted_tool_native_grant_ids = targeted_projection.native_grant_ids
        targeted_tool_native_marker = targeted_projection.native_marker
        # Switching removes every old native state part, including Cayu's
        # own acquisition marker. Rebuild only that current marker from the
        # exact durable grant batch, before context preparation/counting.
        # Never preserve arbitrary provider state or infer grant authority
        # from a caller-supplied marker. Duplicate markers still fail later.
        if (
            targeted_tool_native is not None
            and targeted_tool_native_marker is not None
            and not any(
                targeted_tool_projection_marker_id(message) == targeted_tool_native.marker_id
                for message in messages
            )
        ):
            messages = [*messages, targeted_tool_native_marker]
        discovery_native_grant_ids = await self._native_tool_discovery_grant_ids(
            tool_exposure=tool_exposure,
        )
        # One provider-visible function name can bind to only one frozen grant.
        # A deliberately issued targeted grant takes precedence for its interaction;
        # keep any same-name discovery grant durable but out of this request.
        discovery_native_grant_ids = {
            name: grant_id
            for name, grant_id in discovery_native_grant_ids.items()
            if name not in targeted_tool_native_grant_ids
        }
        native_grant_ids = {
            **targeted_tool_native_grant_ids,
            **discovery_native_grant_ids,
        }
        discovery_native_tool_names = tuple(sorted(discovery_native_grant_ids))
        if targeted_tool_native is not None:
            exposed_names = frozenset(tool_exposure.tool_names)
            overlap = sorted(
                tool["name"] for tool in targeted_tool_native.tools if tool["name"] in exposed_names
            )
            if overlap:
                raise RuntimeError(
                    "Native targeted tools cannot also be directly exposed: " + ", ".join(overlap)
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
        compaction_dispatch_authorities: dict[str, _AutomaticCompactionDispatchAuthority] = {}
        settled_compaction_attempt_ids: set[str] = set()
        automatic_compaction_lifecycle = _AutomaticCompactionLifecycle()

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
            if not published_compaction_attempt_ids:
                await self._require_no_uncheckpointed_automatic_compaction(
                    parent_model_step_identity=model_step_identity,
                )

            await self._persist_automatic_compaction_start_with_retry(
                compaction_started,
                published_events=context_operation_events,
                start_events=compaction_start_events,
                model_step_identity=model_step_identity,
                lifecycle=automatic_compaction_lifecycle,
            )

            async def publish_completions(payloads: list[dict[str, Any]]) -> None:
                try:
                    await self._persist_automatic_compaction_completions(
                        compaction_identity_ledger.identify_payloads(payloads),
                        published_attempt_ids=published_compaction_attempt_ids,
                        published_events=context_operation_events,
                        completion_events=compaction_completion_events,
                        dispatch_authorities=compaction_dispatch_authorities,
                        settled_attempt_ids=settled_compaction_attempt_ids,
                    )
                except BaseException as error:
                    automatic_compaction_lifecycle.attach_failure(
                        error,
                        phase=_AutomaticCompactionLifecyclePhase.COMPLETION_PUBLICATION,
                        reason=_automatic_compaction_publication_failure_reason(error),
                        retryable=False,
                        recovery_action=(_AutomaticCompactionRecoveryAction.RECONCILE_COMPLETION),
                    )
                    raise

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
                    dispatch_authorities=compaction_dispatch_authorities,
                    settled_attempt_ids=settled_compaction_attempt_ids,
                    source_transcript_cursor=source_transcript_cursor,
                    allow_borrowed_stage=False,
                    fallback=fallback,
                    lifecycle=automatic_compaction_lifecycle,
                )

            with _compaction_completion_publisher_scope(publish_completions):
                try:
                    return await run()
                except ModelStreamDeadlineError as error:
                    authoritative = _deadline_with_runtime_recovery_authority(
                        error,
                        exact_operation=False,
                    )
                    if authoritative is error:
                        raise
                    raise authoritative from error

        current_task = asyncio.current_task()
        context_build_cancellation_requests = (
            0 if current_task is None else current_task.cancelling()
        )
        interaction_id = _current_session_interaction_id(self._session.id)
        evidence_key = memory_evidence_key(self._executor._request_footprint)
        try:
            self._validate_live_model_semantics()
            (
                context_messages,
                checkpoint_update,
                checkpoint_event_payload,
                context_compaction_telemetry,
                context_recall_telemetry,
                memory_evidence_reference,
            ) = await _build_context(
                context_policy=self._registered_agent.context_policy,
                session_store=self._executor._session_store,
                session=self._session,
                agent_spec=_session_agent_spec(
                    registered_agent=self._registered_agent,
                    session=self._session,
                    model_execution_selection=self._model_execution_selection,
                ),
                messages=messages,
                step=step,
                interaction_id=interaction_id,
                model_step_id=model_step_identity.model_step_id,
                evidence_key=evidence_key,
                environment_name=self._environment_name,
                knowledge_store=self._knowledge_store,
                knowledge_access_scope=self._knowledge_access_scope,
                request_metadata=self._request_metadata,
                pressure_overhead=_context_pressure_overhead(
                    registered_provider=self._request_registered_provider,
                    registered_agent=self._registered_agent,
                    registered_environment=self._registered_environment,
                    structured_output=self._structured_output,
                    thinking=self._thinking,
                    step=step,
                    tool_exposure=tool_exposure,
                    targeted_tool_projection_kind=self._targeted_tool_projection_kind,
                    targeted_tool_gateway=targeted_tool_gateway,
                    targeted_tool_native=targeted_tool_native,
                    tool_discovery_projection_kind=self._tool_discovery_projection_kind,
                    tool_discovery_native_tool_names=discovery_native_tool_names,
                ),
                count_input_tokens=self._context_input_token_counter(
                    step=step,
                    tool_exposure=tool_exposure,
                    targeted_tool_projection_kind=self._targeted_tool_projection_kind,
                    targeted_tool_gateway=targeted_tool_gateway,
                    targeted_tool_native=targeted_tool_native,
                    tool_discovery_native_tool_names=discovery_native_tool_names,
                ),
                build_cache_prefix_request=self._cache_prefix_request_builder(
                    step=step,
                    tool_exposure=tool_exposure,
                    targeted_tool_projection_kind=self._targeted_tool_projection_kind,
                    targeted_tool_gateway=targeted_tool_gateway,
                    targeted_tool_native=targeted_tool_native,
                    tool_discovery_native_tool_names=discovery_native_tool_names,
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
                raise _combine_context_build_failure_with_secondary(
                    exc,
                    context_failure_persistence,
                    message=(
                        "Context-compaction deadline and context failure persistence both failed."
                    ),
                ) from exc
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
            _raise_context_build_cause(exc)
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
        if context_success_persistence is not None and any(
            telemetry.event_type == EventType.CONTEXT_COMPACTION_COMPLETED
            for telemetry in context_compaction_telemetry
        ):
            automatic_compaction_lifecycle.attach_failure(
                context_success_persistence,
                phase=_AutomaticCompactionLifecyclePhase.CHECKPOINT_INSTALLATION,
                reason=_AutomaticCompactionFailureReason.CHECKPOINT_FAILED,
                retryable=False,
                recovery_action=_AutomaticCompactionRecoveryAction.FAIL_CLOSED,
            )
            checkpoint_failure_event = await self._persist_automatic_compaction_checkpoint_failure(
                context_success_persistence,
                model_step_identity=model_step_identity,
            )
            if checkpoint_failure_event is not None:
                context_success_events.append(checkpoint_failure_event)
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

        child_session_notification_binding = None
        child_session_contributor = self._registered_agent.child_session_context_contributor
        if child_session_contributor is not None:
            child_session_contribution = await child_session_contributor.build(
                session_store=self._executor._session_store,
                session=self._session,
            )
            if child_session_contribution.message is not None:
                context_messages.append(child_session_contribution.message.model_copy(deep=True))
            child_session_notification_binding = child_session_contribution.stage_binding

        from cayu.runtime._argument_continuity import materialize

        context_messages = await materialize(
            store=self._executor._session_store,
            session=self._session,
            profile=None
            if self._execution_profile is None
            else self._execution_profile.fingerprint,
            messages=context_messages,
            names=frozenset(
                tool.name
                for tool in self._registered_agent.tools.values()
                if tool.retain_arguments_for_model and not tool.publish_arguments
            ),
            redactor=self._executor._secret_redactor,
            scope=self._knowledge_access_scope,
        )
        model_request = await self._executor.build_request(
            session=self._session,
            registered_agent=self._registered_agent,
            registered_environment=self._registered_environment,
            context_messages=context_messages,
            structured_output=self._structured_output,
            thinking=self._thinking,
            step=step,
            tool_exposure=tool_exposure,
            targeted_tool_projection_kind=self._targeted_tool_projection_kind,
            targeted_tool_gateway=targeted_tool_gateway,
            targeted_tool_native=targeted_tool_native,
            tool_discovery_projection_kind=self._tool_discovery_projection_kind,
            tool_discovery_native_tool_names=discovery_native_tool_names,
            model_execution_selection=self._model_execution_selection,
        )
        if self._execution_profile is None:
            raise RuntimeError("Tool exposure evidence requires an execution profile.")
        tool_exposure_evidence = tool_exposure_record(
            tool_exposure,
            profile_changed=exposure_profile_changed,
            step=step,
            provider_name=self._request_registered_provider.name,
            model=self._request_model,
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
            memory_evidence_reference=memory_evidence_reference,
            child_session_notification_binding=child_session_notification_binding,
            memory_evidence_key=evidence_key,
            targeted_tool_gateway=targeted_tool_gateway,
            targeted_tool_grants=targeted_projection.footprint,
            native_tool_grant_ids=native_grant_ids,
            fallback=fallback,
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
        memory_evidence_reference: MemoryEvidenceReference | None = None,
        child_session_notification_binding: ChildSessionNotificationStageBinding | None = None,
        memory_evidence_key: MemoryEvidenceKey | None = None,
        targeted_tool_gateway: TargetedToolGatewayProjection | None = None,
        targeted_tool_grants: TargetedToolGrantFootprint | None = None,
        native_tool_grant_ids: Mapping[str, str] | None = None,
        fallback: ModelFailoverTransition | None = None,
    ) -> AsyncIterator[tuple[Event | None, ModelStepFlowOutcome | None]]:
        model_step_identity = copy_model_step_identity(model_step_identity)
        if targeted_tool_grants is None and self._targeted_tool_grants is not None:
            if self._model_execution_selection is not None:
                raise RuntimeError("Selected request lost its targeted grant footprint.")
            targeted_tool_grants = self._targeted_tool_grants
        native_grant_ids = {} if native_tool_grant_ids is None else dict(native_tool_grant_ids)
        if (
            memory_evidence_reference is not None or child_session_notification_binding is not None
        ) and self._model_completion_publisher is None:
            raise RuntimeError(
                "Runtime context contributors require durable model-completion publication."
            )
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
                provider_name=self._request_registered_provider.name,
                model=self._request_model,
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
        await self._refresh_live_model_semantics()
        try:
            billing_identity = await resolve_request_billing_identity(
                self._request_provider,
                _detach_model_request(model_request),
                provider_name=self._request_registered_provider.name,
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
                                execution_provider_name=self._request_registered_provider.name,
                                requested_model=self._request_model,
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
        if (
            self._model_execution_selection is not None
            or billing_identity is not None
            or self._has_deferred_contextual_price()
        ):
            # The shared invocation gate retains root authority, not the
            # selected target. Routed requests need its pricing preflight even
            # without reservations or a contextual billing identity.
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
        resolved_budget_binding = await controller._binding_for_dispatch(
            request={
                "session_id": self._session.id,
                "agent_name": self._registered_agent.spec.name,
                "kind": "model",
            },
            binding=None,
        )
        reservation_setup = await controller.reserve_for_model_step(
            session=self._session,
            agent_name=self._registered_agent.spec.name,
            provider_name=self._request_registered_provider.name,
            model=self._request_model,
            environment_name=self._environment_name,
            model_attempt_identity=initial_model_attempt_identity,
            budget_policy=self._budget_policy,
            request_budget_limits=self._request_budget_limits,
            billing_identity=billing_identity,
            execution_profile_fingerprint=(
                None if self._execution_profile is None else self._execution_profile.fingerprint
            ),
            reservation_identity_guard=self._reservation_identity_guard,
            binding=resolved_budget_binding,
            _binding_provenance=_TRUSTED_BINDING_PROVENANCE,
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
            BaseException | None,
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
                provider_name=self._request_registered_provider.name,
                model=self._request_model,
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
                binding=resolved_budget_binding,
                _binding_provenance=_TRUSTED_BINDING_PROVENANCE,
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
            current_execution_deadline().require_admission("model_dispatch")
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
            evidence_reference: MemoryEvidenceReference | None,
            child_notification_binding: ChildSessionNotificationStageBinding | None,
            consume_child_session_notifications: bool,
            *,
            failover_attempt: ModelFailoverAttempt | None = None,
        ) -> ModelCompletionDispatch:
            nonlocal next_dispatch_ordinal
            request_fingerprint = _model_request_fingerprint(
                provider_name=self._request_registered_provider.name,
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
                if (failover_attempt is None) != (self._model_execution_selection is None):
                    raise ValueError("Selected model preparation lost its explicit attempt.")
                route_admission = None
                if failover_attempt is not None:
                    if (
                        failover_attempt.selection is not self._model_execution_selection
                        or failover_attempt.identity != pending_model_attempt_identity
                    ):
                        raise ValueError(
                            "Failover preparation changed its pending attempt authority."
                        )
                    checkpoint = await self._executor._session_store.load_checkpoint(
                        self._session.id
                    )
                    previous = (
                        None
                        if checkpoint is None or MODEL_FAILOVER_CHECKPOINT_KEY not in checkpoint
                        else copy_model_failover_state(checkpoint[MODEL_FAILOVER_CHECKPOINT_KEY])
                    )
                    source = (
                        None
                        if not isinstance(previous, ModelFailoverProgress)
                        else await self._executor._session_store.load_model_completion_stage(
                            self._session.id, previous.stage_id
                        )
                    )
                    source_digest = None if source is None else source.preparation_digest
                    if isinstance(previous, ModelFailoverProgress) and source is None:
                        abandonment = await self._executor._session_store.load_model_completion_stage_abandonment(
                            self._session.id, previous.stage_id
                        )
                        if abandonment is not None:
                            source_digest = abandonment.preparation_digest
                    route_admission = failover_attempt.stage_admission(
                        previous=previous,
                        source_preparation_digest=source_digest,
                        source_transcript_cursor=source_transcript_cursor,
                        request_fingerprint=request_fingerprint,
                        fallback=(
                            fallback
                            if previous is not None
                            and previous.candidate_index
                            != failover_attempt.selection.candidate_index
                            else None
                        ),
                    )
                    dispatch_ordinal = route_admission.successor.dispatch_ordinal
                    stage_id = route_admission.successor.stage_id
                recovery_context = self._model_completion_recovery_context_factory(
                    billing_identity,
                    pending_reservations,
                )
                if recovery_context is not None:
                    hosted_projection = attempt_model_request.tool_discovery_projection
                    hosted_authority = (
                        HostedToolDiscoveryRecoveryAuthority(
                            projection_sha256=(
                                _hosted_tool_discovery_projection_digest(hosted_projection)
                            ),
                            targeted_tool_name_sha256s=tuple(
                                sorted(
                                    _hosted_tool_name_sha256(cast("str", tool["name"]))
                                    for tool in (
                                        ()
                                        if attempt_model_request.targeted_tool_projection is None
                                        else attempt_model_request.targeted_tool_projection.tools
                                    )
                                )
                            ),
                            loaded_tool_name_sha256s=tuple(
                                sorted(
                                    _hosted_tool_name_sha256(name)
                                    for name in hosted_projection.loaded_tool_names
                                )
                            ),
                        )
                        if hosted_projection is not None
                        and hosted_projection.protocol == OPENAI_HOSTED_TOOL_SEARCH_PROTOCOL
                        else None
                    )
                    recovery_context = recovery_context.model_copy(
                        update={
                            "tool_exposure": resolved_tool_exposure_authority(tool_exposure),
                            "hosted_tool_discovery": hosted_authority,
                        },
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
                provider_operation_mode = self._request_provider.provider_operation_mode
                if type(provider_operation_mode) is not ProviderOperationMode:
                    raise TypeError(
                        "ModelProvider.provider_operation_mode must return a ProviderOperationMode."
                    )
                if provider_operation_mode is ProviderOperationMode.BACKGROUND:
                    if (
                        attempt_model_request.tool_discovery_projection is not None
                        and attempt_model_request.tool_discovery_projection.protocol
                        == OPENAI_HOSTED_TOOL_SEARCH_PROTOCOL
                        and (
                            recovery_context is None
                            or recovery_context.hosted_tool_discovery is None
                        )
                    ):
                        raise RuntimeError(
                            "Background hosted Tool Search requires durable recovery authority."
                        )
                    operation_adapter = self._request_provider.provider_operations
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
                context_exposure: ContextExposure | None = None
                try:
                    if evidence_reference is not None:
                        if memory_evidence_key is None or self._execution_profile is None:
                            raise RuntimeError(
                                "Automatic recall dispatch requires keyed execution-profile "
                                "evidence."
                            )
                        exposure_interaction_id = _current_session_interaction_id(self._session.id)
                        if exposure_interaction_id is None:
                            raise RuntimeError(
                                "Automatic recall dispatch lost its interaction identity."
                            )
                        context_exposure = await prepare_context_exposure(
                            store=self._executor._session_store,
                            session_id=self._session.id,
                            interaction_id=exposure_interaction_id,
                            model_request=attempt_model_request,
                            request_fingerprint_sha256=request_fingerprint,
                            provider_name=self._request_registered_provider.name,
                            model_attempt_identity=pending_model_attempt_identity,
                            execution_profile=self._execution_profile,
                            tool_exposure=tool_exposure,
                            reference=evidence_reference,
                            key=memory_evidence_key,
                        )
                    intent = _model_completion_stage_intent(
                        model_attempt_identity=pending_model_attempt_identity,
                        provider_name=self._request_registered_provider.name,
                        requested_model=attempt_model_request.model,
                        source_transcript_cursor=source_transcript_cursor,
                        request_fingerprint=request_fingerprint,
                        recovery_context=recovery_context,
                        input_coverage=context_input_coverage(
                            attempt_model_request.messages,
                            transcript_cursor=source_transcript_cursor,
                        ),
                        provider_operation_start=provider_operation_start,
                        context_exposure=(
                            None
                            if context_exposure is None
                            else context_exposure_identity_payload(context_exposure)
                        ),
                        child_session_notifications=child_notification_binding,
                    )
                    prepare_stage = (
                        self._executor._session_store.prepare_model_completion_stage
                        if route_admission is None
                        else partial(
                            self._executor._session_store._prepare_model_completion_stage_with_failover,
                            admission=route_admission,
                        )
                    )
                    prepared = await prepare_stage(
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
                except BaseException as preparation_failure:
                    classified_preparation_failure = preparation_failure

                    async def settle_preparation_failure() -> Never:
                        # Probe and terminate in the same owner: a refusal has
                        # no stage from which recovery could find this exposure.
                        # Publish each classification as soon as it is known so
                        # a cleanup deadline still reports the interrupt winner.
                        nonlocal classified_preparation_failure
                        failure = classified_preparation_failure
                        try:
                            if isinstance(
                                failure, (SessionStatusConflict, SessionRunFenced)
                            ) and await _model_preparation_lost_to_interrupt(
                                self._executor._session_store, self._session
                            ):
                                failure = SessionInterruptedByRequest(self._session.id)
                        except BaseException as probe_failure:
                            set_exception_cause(probe_failure, failure)
                            failure = probe_failure
                        classified_preparation_failure = failure
                        if context_exposure is not None:
                            try:
                                await _terminate_pre_dispatch_context_exposure(
                                    store=self._executor._session_store,
                                    exposure=context_exposure,
                                    failure=failure,
                                    evidence_ref=f"model-stage:{stage_id}",
                                )
                            except BaseException as cleanup_failure:
                                raise cleanup_failure from failure
                        raise failure

                    await _raise_after_model_pre_dispatch_cleanup(
                        settle_preparation_failure,
                        unsettled_failure=lambda: classified_preparation_failure,
                        supervisor=self._executor._recovery_cleanup_supervisor,
                    )
                if not prepared.dispatch_authorized:
                    authorization_failure = ModelCompletionDispatchNotAuthorized(
                        stage=prepared.stage,
                        request_fingerprint=request_fingerprint,
                    )

                    async def settle_authorization_failure() -> Never:
                        if context_exposure is not None:
                            await _terminate_pre_dispatch_context_exposure(
                                store=self._executor._session_store,
                                exposure=context_exposure,
                                failure=authorization_failure,
                                evidence_ref=f"model-stage:{stage_id}",
                            )
                        raise authorization_failure

                    await _raise_after_model_pre_dispatch_cleanup(
                        settle_authorization_failure,
                        unsettled_failure=lambda: authorization_failure,
                        supervisor=self._executor._recovery_cleanup_supervisor,
                    )
                dispatch_fence_attempted = False
                dispatch_fence_committed = False
                try:
                    # Preparation may itself be a remote database round trip. Renew
                    # after it commits so a lease cannot expire in the new gap
                    # between durable staging and provider-controlled code.
                    if budget_reservations and controller.reservation_ttl_seconds is not None:
                        await controller.renew_reservations(budget_reservations)
                    if context_exposure is not None:
                        context_exposure = await transition_context_exposure(
                            store=self._executor._session_store,
                            exposure=context_exposure,
                            state=ContextExposureState.DISPATCH_STARTED,
                            evidence_kind=(ContextExposureEvidenceKind.DISPATCH_INTENT_COMMITTED),
                            evidence_ref=f"model-stage:{prepared.stage.stage_id}",
                        )
                    dispatch_fence_attempted = True
                    deferred_dispatch_failure = await controller.mark_reservations_dispatched(
                        pending_reservations,
                        dispatch_id=prepared.stage.stage_id,
                    )
                    dispatch_fence_committed = True
                    await self._executor._session_store.mark_model_completion_stage_dispatched(
                        self._session.id,
                        stage=prepared.stage,
                        consume_child_session_notifications=(consume_child_session_notifications),
                    )
                    dispatch = ModelCompletionDispatch(
                        stage=prepared.stage,
                        request_fingerprint=request_fingerprint,
                        context_exposure=context_exposure,
                        prepared_events=prepared.prepared_events,
                        child_session_notifications_consumed=(
                            child_notification_binding is None
                            or consume_child_session_notifications
                        ),
                    )
                    lifecycle.mark_provider_dispatch(pending_model_attempt_identity)
                    if deferred_dispatch_failure is not None:
                        raise deferred_dispatch_failure
                except BaseException as authoritative_exc:

                    async def settle_dispatch_failure(
                        authoritative_exc: BaseException = authoritative_exc,
                    ) -> Never:
                        exposure_terminal = (
                            context_exposure is None or context_exposure.state.terminal
                        )
                        if not exposure_terminal and context_exposure is not None:
                            exposure_terminal = await _terminate_pre_dispatch_context_exposure(
                                store=self._executor._session_store,
                                exposure=context_exposure,
                                failure=authoritative_exc,
                                evidence_ref=f"model-stage:{stage_id}",
                            )
                        if exposure_terminal and (
                            not dispatch_fence_attempted or dispatch_fence_committed
                        ):
                            await self._abandon_pre_dispatch_model_stage(
                                prepared.stage,
                                authoritative_failure=authoritative_exc,
                            )
                        elif not exposure_terminal:
                            add_exception_note_safely(
                                authoritative_exc,
                                "The prepared model-completion stage was retained because linked "
                                "context-exposure termination is not durable.",
                            )
                        else:
                            add_exception_note_safely(
                                authoritative_exc,
                                "The prepared model-completion stage was retained because the "
                                "budget dispatch fence could not be reconstructed exactly.",
                            )
                        raise _classify_pre_dispatch_failure(authoritative_exc)

                    await _raise_after_model_pre_dispatch_cleanup(
                        settle_dispatch_failure,
                        unsettled_failure=partial(
                            _classify_pre_dispatch_failure, authoritative_exc
                        ),
                        supervisor=self._executor._recovery_cleanup_supervisor,
                    )
                next_dispatch_ordinal = dispatch_ordinal + 1
                return dispatch

        def record_model_completion(event: Event) -> Event:
            return lifecycle.record_model_completion(
                event,
                prepare_event=self._executor._event_writer.prepare,
                settled_at=controller.budget_settlement_time(),
            )

        flow_outcome: ModelStepFlowOutcome | None = None
        model_step_events = self._run_with_context_overflow_recovery(
            provider=self._request_provider,
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
            memory_evidence_reference=memory_evidence_reference,
            child_session_notification_binding=child_session_notification_binding,
            memory_evidence_key=memory_evidence_key,
            targeted_tool_gateway=targeted_tool_gateway,
            targeted_tool_grants=targeted_tool_grants,
            native_tool_grant_ids=native_grant_ids,
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
                    await self._executor._provider_operation_cancellation.release_claim(
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
                    await self._executor._provider_operation_cancellation.release_claim(
                        session=self._session,
                        claim=cancellation_claim,
                    )
            raise
        except _ModelFailoverCandidateExhausted as exhausted:
            # Ordinary failure settlement may attach a secondary failure as a
            # note and preserve its primary. A candidate handoff is different:
            # an accounting failure must prevent selection of another provider.
            try:
                async for event in controller.settle_after_model_failure(
                    budget_reservations,
                    lifecycle=lifecycle,
                    session=self._session,
                    agent_name=self._registered_agent.spec.name,
                    environment_name=self._environment_name,
                    release_reason="selected model candidate exhausted",
                ):
                    yield event, None
            except (asyncio.CancelledError, GeneratorExit) as settlement_control:
                raise settlement_control from (exhausted.failure.cause or exhausted.failure)
            except BaseException as settlement_failure:
                primary_failure = exhausted.failure.cause or exhausted.failure
                raise _combine_authoritative_model_failure(
                    primary_failure,
                    settlement_failure,
                    message="Model candidate failure and budget settlement both failed.",
                ) from None
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
            Awaitable[tuple[list[Event], BudgetReservationResult | None, BaseException | None]],
        ],
        before_provider_dispatch: Callable[[ModelAttemptIdentity], Awaitable[None]],
        billing_identity: BillingIdentity | None,
        prepare_model_completion_dispatch: ModelCompletionDispatchPreparer | None,
        model_completion_publisher: ModelCompletionPublisher | None,
        tool_exposure: ResolvedToolExposure,
        tool_exposure_evidence: ToolExposure,
        memory_evidence_reference: MemoryEvidenceReference | None,
        child_session_notification_binding: ChildSessionNotificationStageBinding | None,
        memory_evidence_key: MemoryEvidenceKey | None,
        targeted_tool_gateway: TargetedToolGatewayProjection | None,
        targeted_tool_grants: TargetedToolGrantFootprint | None,
        native_tool_grant_ids: Mapping[str, str],
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
        compaction_dispatch_authorities: dict[str, _AutomaticCompactionDispatchAuthority] = {}
        settled_compaction_attempt_ids: set[str] = set()
        automatic_compaction_lifecycle = _AutomaticCompactionLifecycle()
        latest_model_attempt_identity: ModelAttemptIdentity | None = None
        selected_context_overflow: _ModelFailoverContextOverflow | None = None

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
            evidence_reference: MemoryEvidenceReference | None,
            child_notification_binding: ChildSessionNotificationStageBinding | None,
            initial_identity: ModelAttemptIdentity | None = None,
            attempt_variant: RequestVariant,
            context_overflow: _ModelFailoverContextOverflow | None = None,
        ) -> AsyncIterator[tuple[Event | None, AssistantStepResult | None]]:
            request_native_grant_ids = _native_tool_grant_ids_for_request(
                request,
                available=native_tool_grant_ids,
            )
            return self._executor.run_with_retries(
                provider=provider,
                model_request=request,
                session=self._session,
                registered_agent=self._registered_agent,
                registered_provider=self._request_registered_provider,
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
                refresh_live_model_semantics=self._refresh_live_model_semantics,
                record_model_attempt_identity=record_model_attempt_identity,
                billing_identity=billing_identity,
                structured_output=self._structured_output,
                prepare_model_completion_dispatch=prepare_model_completion_dispatch,
                model_completion_publisher=model_completion_publisher,
                execution_profile=self._execution_profile,
                invocation_context=self._invocation_context,
                model_execution_selection=self._model_execution_selection,
                context_overflow=context_overflow,
                tool_exposure=tool_exposure,
                tool_exposure_evidence=tool_exposure_evidence,
                targeted_tool_grants=targeted_tool_grants,
                targeted_tool_gateway=targeted_tool_gateway,
                native_tool_grant_ids=request_native_grant_ids,
                memory_evidence_reference=evidence_reference,
                child_session_notification_binding=child_notification_binding,
            )

        attempt_events = run_attempt(
            model_request,
            evidence_reference=memory_evidence_reference,
            child_notification_binding=child_session_notification_binding,
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
            except (ModelContextOverflowError, _ModelFailoverContextOverflow) as caught:
                if isinstance(caught, _ModelFailoverContextOverflow):
                    if (
                        overflow_policy is None
                        or caught.progress.attempts_used == caught.progress.plan.max_total_attempts
                        or caught.failure.completion_observed
                        or caught.failure.automatic_retry_disabled
                    ):
                        _raise_terminal_model_attempt_failure(caught.failure)
                    selected_context_overflow = caught
                    assert isinstance(caught.failure.cause, ModelContextOverflowError)
                    exc = caught.failure.cause
                else:
                    exc = caught
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
            if not published_compaction_attempt_ids:
                await self._require_no_uncheckpointed_automatic_compaction(
                    parent_model_step_identity=model_step_identity,
                )

            await self._persist_automatic_compaction_start_with_retry(
                compaction_started,
                published_events=context_operation_events,
                start_events=compaction_start_events,
                model_step_identity=model_step_identity,
                lifecycle=automatic_compaction_lifecycle,
            )

            async def publish_completions(payloads: list[dict[str, Any]]) -> None:
                try:
                    await self._persist_automatic_compaction_completions(
                        compaction_identity_ledger.identify_payloads(payloads),
                        published_attempt_ids=published_compaction_attempt_ids,
                        published_events=context_operation_events,
                        completion_events=compaction_completion_events,
                        dispatch_authorities=compaction_dispatch_authorities,
                        settled_attempt_ids=settled_compaction_attempt_ids,
                    )
                except BaseException as error:
                    automatic_compaction_lifecycle.attach_failure(
                        error,
                        phase=_AutomaticCompactionLifecyclePhase.COMPLETION_PUBLICATION,
                        reason=_automatic_compaction_publication_failure_reason(error),
                        retryable=False,
                        recovery_action=(_AutomaticCompactionRecoveryAction.RECONCILE_COMPLETION),
                    )
                    raise

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
                    dispatch_authorities=compaction_dispatch_authorities,
                    settled_attempt_ids=settled_compaction_attempt_ids,
                    source_transcript_cursor=transcript_cursor_before_request,
                    allow_borrowed_stage=True,
                    lifecycle=automatic_compaction_lifecycle,
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
                recovery_memory_evidence_reference,
            ) = await _build_context(
                context_policy=overflow_policy,
                session_store=self._executor._session_store,
                session=self._session,
                agent_spec=_session_agent_spec(
                    registered_agent=self._registered_agent,
                    session=self._session,
                    model_execution_selection=self._model_execution_selection,
                ),
                messages=messages,
                step=step,
                interaction_id=_current_session_interaction_id(self._session.id),
                model_step_id=model_step_identity.model_step_id,
                evidence_key=memory_evidence_key,
                environment_name=self._environment_name,
                knowledge_store=self._knowledge_store,
                knowledge_access_scope=self._knowledge_access_scope,
                request_metadata=self._request_metadata,
                pressure_overhead=_context_pressure_overhead(
                    registered_provider=self._request_registered_provider,
                    registered_agent=self._registered_agent,
                    registered_environment=self._registered_environment,
                    structured_output=self._structured_output,
                    thinking=self._thinking,
                    step=step,
                    tool_exposure=tool_exposure,
                    targeted_tool_projection_kind=self._targeted_tool_projection_kind,
                    targeted_tool_gateway=targeted_tool_gateway,
                    targeted_tool_native=model_request.targeted_tool_projection,
                    tool_discovery_projection_kind=self._tool_discovery_projection_kind,
                    tool_discovery_native_tool_names=(
                        ()
                        if model_request.tool_discovery_projection is None
                        else model_request.tool_discovery_projection.loaded_tool_names
                    ),
                ),
                count_input_tokens=self._context_input_token_counter(
                    step=step,
                    tool_exposure=tool_exposure,
                    targeted_tool_projection_kind=self._targeted_tool_projection_kind,
                    targeted_tool_gateway=targeted_tool_gateway,
                    targeted_tool_native=model_request.targeted_tool_projection,
                    tool_discovery_native_tool_names=(
                        ()
                        if model_request.tool_discovery_projection is None
                        else model_request.tool_discovery_projection.loaded_tool_names
                    ),
                ),
                build_cache_prefix_request=self._cache_prefix_request_builder(
                    step=step,
                    tool_exposure=tool_exposure,
                    targeted_tool_projection_kind=self._targeted_tool_projection_kind,
                    targeted_tool_gateway=targeted_tool_gateway,
                    targeted_tool_native=model_request.targeted_tool_projection,
                    tool_discovery_native_tool_names=(
                        ()
                        if model_request.tool_discovery_projection is None
                        else model_request.tool_discovery_projection.loaded_tool_names
                    ),
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
                raise _combine_context_build_failure_with_secondary(
                    exc,
                    context_failure_persistence,
                    message=(
                        "Context-compaction deadline and context failure persistence both failed."
                    ),
                ) from exc
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
            try:
                overflow_failed_event = await self._executor._event_writer.emit(
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
                )
            except Exception as publication_failure:
                raise _combine_context_build_failure_with_secondary(
                    exc,
                    publication_failure,
                    message=(
                        "Context-compaction deadline and context-overflow diagnostic "
                        "publication both failed."
                    ),
                ) from exc
            yield overflow_failed_event, None
            _raise_context_build_cause(exc)
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
        if context_success_persistence is not None and any(
            telemetry.event_type == EventType.CONTEXT_COMPACTION_COMPLETED
            for telemetry in compaction_telemetry
        ):
            automatic_compaction_lifecycle.attach_failure(
                context_success_persistence,
                phase=_AutomaticCompactionLifecyclePhase.CHECKPOINT_INSTALLATION,
                reason=_AutomaticCompactionFailureReason.CHECKPOINT_FAILED,
                retryable=False,
                recovery_action=_AutomaticCompactionRecoveryAction.FAIL_CLOSED,
            )
            checkpoint_failure_event = await self._persist_automatic_compaction_checkpoint_failure(
                context_success_persistence,
                model_step_identity=model_step_identity,
            )
            if checkpoint_failure_event is not None:
                context_success_events.append(checkpoint_failure_event)
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

        recovery_child_notification_binding = None
        child_session_contributor = self._registered_agent.child_session_context_contributor
        if child_session_contributor is not None:
            child_session_contribution = await child_session_contributor.build(
                session_store=self._executor._session_store,
                session=self._session,
            )
            if child_session_contribution.message is not None:
                recovery_context_messages.append(
                    child_session_contribution.message.model_copy(deep=True)
                )
            recovery_child_notification_binding = child_session_contribution.stage_binding

        from cayu.runtime._argument_continuity import materialize

        recovery_context_messages = await materialize(
            store=self._executor._session_store,
            session=self._session,
            profile=None
            if self._execution_profile is None
            else self._execution_profile.fingerprint,
            messages=recovery_context_messages,
            names=frozenset(
                tool.name
                for tool in self._registered_agent.tools.values()
                if tool.retain_arguments_for_model and not tool.publish_arguments
            ),
            redactor=self._executor._secret_redactor,
            scope=self._knowledge_access_scope,
        )
        recovery_request = await self._executor.build_request(
            session=self._session,
            registered_agent=self._registered_agent,
            registered_environment=self._registered_environment,
            context_messages=recovery_context_messages,
            structured_output=self._structured_output,
            thinking=self._thinking,
            step=step,
            tool_exposure=tool_exposure,
            targeted_tool_projection_kind=self._targeted_tool_projection_kind,
            targeted_tool_gateway=targeted_tool_gateway,
            targeted_tool_native=model_request.targeted_tool_projection,
            tool_discovery_projection_kind=self._tool_discovery_projection_kind,
            tool_discovery_native_tool_names=(
                ()
                if model_request.tool_discovery_projection is None
                else model_request.tool_discovery_projection.loaded_tool_names
            ),
            model_execution_selection=self._model_execution_selection,
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
            evidence_reference=recovery_memory_evidence_reference,
            child_notification_binding=recovery_child_notification_binding,
            attempt_variant=RequestVariant.CONTEXT_OVERFLOW_RECOVERY,
            context_overflow=selected_context_overflow,
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
            except (ModelContextOverflowError, _ModelFailoverContextOverflow) as caught:
                if isinstance(caught, _ModelFailoverContextOverflow):
                    assert isinstance(caught.failure.cause, ModelContextOverflowError)
                    exc = caught.failure.cause
                else:
                    exc = caught
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
                if isinstance(caught, _ModelFailoverContextOverflow):
                    _raise_terminal_model_attempt_failure(caught.failure)
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
                    invocation_context=self._invocation_context,
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
                invocation_context=self._invocation_context,
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
            timeout_after_cancellation_s=(_CONTEXT_EVENT_STORE_WAIT_AFTER_CANCELLATION_TIMEOUT_S),
        )
        if outcome.timed_out:
            reconciliation_task.cancel()
            self._executor._retain_detached_task(reconciliation_task)
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
            timeout_after_cancellation_s=(_CONTEXT_EVENT_STORE_WAIT_AFTER_CANCELLATION_TIMEOUT_S),
        )
        if outcome.timed_out:
            fan_out_task.cancel()
            self._executor._retain_detached_task(fan_out_task)
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

    async def _require_no_uncheckpointed_automatic_compaction(
        self,
        *,
        parent_model_step_identity: ModelStepIdentity,
    ) -> None:
        parent = copy_model_step_identity(parent_model_step_identity)
        before_sequence: int | None = None
        checkpointed_parent_model_step_ids: set[str] = set()
        while True:
            records = await self._executor._session_store.query_events(
                EventQuery(
                    session_id=self._session.id,
                    event_types=(
                        EventType.MODEL_COMPLETED,
                        EventType.SESSION_CHECKPOINTED,
                    ),
                    order_by=EventOrder.SEQUENCE_DESC,
                    before_sequence=before_sequence,
                    limit=100,
                )
            )
            if not records:
                return
            for record in records:
                event = record.event
                if event.type is EventType.SESSION_CHECKPOINTED:
                    checkpointed_parent = event.payload.get("model_step_id")
                    if (
                        type(checkpointed_parent) is str
                        and event.payload.get(_COMPACTION_UNREPRESENTED_CALLS_KEY, False) is False
                    ):
                        checkpointed_parent_model_step_ids.add(checkpointed_parent)
                    continue
                if (
                    event.type is EventType.MODEL_COMPLETED
                    and event.payload.get("purpose")
                    == ModelCompletionPurpose.CONTEXT_COMPACTION.value
                    and event.payload.get("compaction_outcome") is None
                ):
                    completed_parent = event.payload.get("parent_model_step_id")
                    if completed_parent is None:
                        # Completions from schema versions predating explicit parent
                        # attribution cannot be classified by this restart fence.
                        continue
                    if type(completed_parent) is not str:
                        raise _AutomaticCompactionCheckpointRecoveryRequired(
                            "Successful automatic-compaction evidence has malformed "
                            "parent model-step authority."
                        )
                    if completed_parent in checkpointed_parent_model_step_ids:
                        continue
                    raise _AutomaticCompactionCheckpointRecoveryRequired(
                        "A successful automatic compaction has no later durable context "
                        f"checkpoint for model step {completed_parent}; current model step "
                        f"{parent.model_step_id} cannot dispatch another compactor."
                    )
            before_sequence = records[-1].sequence

    async def _persist_automatic_compaction_predispatch_event(
        self,
        event: Event,
        *,
        phase: _AutomaticCompactionLifecyclePhase,
        lifecycle: _AutomaticCompactionLifecycle,
    ) -> Event:
        """Publish one stable pre-dispatch event with bounded exact retries."""

        prepared = self._executor._event_writer.prepare(event)
        last_error: BaseException | None = None
        for attempt in range(1, _AUTOMATIC_COMPACTION_PREDISPATCH_PUBLICATION_ATTEMPTS + 1):
            persistence_task = asyncio.create_task(
                self._executor._event_writer.persist_exact_replay(prepared)
            )
            outcome = await await_shielded_task_outcome(
                persistence_task,
                timeout_s=_AUTOMATIC_COMPACTION_PREDISPATCH_EVENT_STORE_WAIT_TIMEOUT_S,
                timeout_after_cancellation_s=(
                    _CONTEXT_EVENT_STORE_WAIT_AFTER_CANCELLATION_TIMEOUT_S
                ),
            )
            cancellation = outcome.cancellation
            if outcome.timed_out:
                persistence_task.cancel()
                self._executor._retain_detached_task(persistence_task)
                publication_error: BaseException | None = TimeoutError(
                    "Automatic compaction pre-dispatch publication exceeded "
                    f"{_AUTOMATIC_COMPACTION_PREDISPATCH_EVENT_STORE_WAIT_TIMEOUT_S:g} "
                    "seconds."
                )
            else:
                publication_error = outcome.error
            if isinstance(publication_error, asyncio.CancelledError) and cancellation is None:
                publication_error = unexpected_child_cancellation_error(
                    publication_error,
                    operation="Automatic compaction pre-dispatch publication",
                )
            if publication_error is None and outcome.result is None:
                publication_error = RuntimeError(
                    "Automatic compaction pre-dispatch publication returned no result."
                )
            if publication_error is None:
                persisted = outcome.result
                assert persisted is not None
                fan_out_task = asyncio.create_task(
                    self._executor._event_writer.fan_out_persisted([persisted])
                )
                fan_out_outcome = await await_shielded_task_outcome(
                    fan_out_task,
                    cancellation=cancellation,
                    timeout_s=(_AUTOMATIC_COMPACTION_PREDISPATCH_EVENT_STORE_WAIT_TIMEOUT_S),
                    timeout_after_cancellation_s=(
                        _CONTEXT_EVENT_STORE_WAIT_AFTER_CANCELLATION_TIMEOUT_S
                    ),
                )
                cancellation = fan_out_outcome.cancellation
                if fan_out_outcome.timed_out:
                    fan_out_task.cancel()
                    self._executor._retain_detached_task(fan_out_task)
                    publication_error = TimeoutError(
                        "Automatic compaction pre-dispatch side-effect delivery exceeded "
                        f"{_AUTOMATIC_COMPACTION_PREDISPATCH_EVENT_STORE_WAIT_TIMEOUT_S:g} "
                        "seconds."
                    )
                else:
                    publication_error = fan_out_outcome.error
                if isinstance(publication_error, asyncio.CancelledError) and cancellation is None:
                    publication_error = unexpected_child_cancellation_error(
                        publication_error,
                        operation=("Automatic compaction pre-dispatch side-effect delivery"),
                    )
                if publication_error is None and fan_out_outcome.result is None:
                    publication_error = RuntimeError(
                        "Automatic compaction pre-dispatch side-effect delivery returned no result."
                    )
                if publication_error is None:
                    delivered = fan_out_outcome.result
                    assert delivered is not None
                    if cancellation is not None:
                        raise cancellation
                    return delivered[0]
            if cancellation is not None:
                lifecycle.attach_failure(
                    cancellation,
                    phase=phase,
                    reason=_AutomaticCompactionFailureReason.CANCELLED,
                    retryable=False,
                    recovery_action=(
                        _AutomaticCompactionRecoveryAction.FAIL_CLOSED
                        if lifecycle.provider_dispatch_disposition
                        == _AutomaticCompactionDispatchDisposition.NOT_DISPATCHED
                        else _AutomaticCompactionRecoveryAction.RECONCILE_COMPLETION
                    ),
                )
                if publication_error is not None:
                    raise cancellation from publication_error
                raise cancellation
            assert publication_error is not None
            last_error = publication_error
            publication_retryable = _automatic_compaction_publication_is_retryable(
                publication_error
            )
            safe_to_resume = (
                lifecycle.provider_dispatch_disposition
                == _AutomaticCompactionDispatchDisposition.NOT_DISPATCHED
            )
            retryable = publication_retryable and safe_to_resume
            if (
                not publication_retryable
                or attempt == _AUTOMATIC_COMPACTION_PREDISPATCH_PUBLICATION_ATTEMPTS
            ):
                lifecycle.attach_failure(
                    publication_error,
                    phase=phase,
                    reason=_automatic_compaction_publication_failure_reason(publication_error),
                    retryable=retryable,
                    recovery_action=(
                        _AutomaticCompactionRecoveryAction.RESUME_SESSION
                        if retryable
                        else (
                            _AutomaticCompactionRecoveryAction.RECONCILE_COMPLETION
                            if not safe_to_resume
                            else _AutomaticCompactionRecoveryAction.FAIL_CLOSED
                        )
                    ),
                )
                raise publication_error
            await asyncio.sleep(0)
        raise AssertionError("Automatic compaction publication retry loop did not finish.") from (
            last_error
        )

    async def _persist_automatic_compaction_checkpoint_failure(
        self,
        error: BaseException,
        *,
        model_step_identity: ModelStepIdentity,
    ) -> Event | None:
        """Publish bounded checkpoint-installation evidence before terminalization."""

        disposition = automatic_compaction_failure_disposition_payload(error)
        if disposition is None:
            return None
        failure_event = _context_compaction_telemetry_event(
            telemetry=ContextCompactionTelemetry(
                event_type=EventType.CONTEXT_COMPACTION_FAILED,
                payload={
                    "error_type": type(error).__name__,
                    "coverage_mode": "failed",
                    "compaction_failed": True,
                    **disposition,
                },
            ),
            session=self._session,
            registered_agent=self._registered_agent,
            environment_name=self._environment_name,
            execution_identity=model_step_identity,
            execution_profile=self._execution_profile,
        )
        try:
            return await self._executor._event_writer.emit(failure_event)
        except asyncio.CancelledError:
            raise
        except Exception as publication_error:
            error.add_note(
                "Checkpoint-installation failure evidence publication also failed: "
                f"{type(publication_error).__name__}."
            )
            return None

    async def _persist_automatic_compaction_start_with_retry(
        self,
        telemetry: ContextCompactionTelemetry,
        *,
        published_events: list[Event],
        start_events: list[Event],
        model_step_identity: ModelStepIdentity,
        lifecycle: _AutomaticCompactionLifecycle,
    ) -> None:
        last_error: BaseException | None = None
        for attempt in range(1, _AUTOMATIC_COMPACTION_PREDISPATCH_PUBLICATION_ATTEMPTS + 1):
            try:
                await self._persist_automatic_compaction_started(
                    telemetry,
                    published_events=published_events,
                    start_events=start_events,
                    model_step_identity=model_step_identity,
                )
                return
            except BaseException as error:
                if isinstance(error, asyncio.CancelledError):
                    lifecycle.attach_failure(
                        error,
                        phase=_AutomaticCompactionLifecyclePhase.START_PUBLICATION,
                        reason=_AutomaticCompactionFailureReason.CANCELLED,
                        retryable=False,
                        recovery_action=_AutomaticCompactionRecoveryAction.FAIL_CLOSED,
                    )
                    raise
                if any(
                    event.type == EventType.CONTEXT_COMPACTION_STARTED for event in published_events
                ):
                    return
                retryable = _automatic_compaction_publication_is_retryable(error)
                if (
                    not retryable
                    or attempt == _AUTOMATIC_COMPACTION_PREDISPATCH_PUBLICATION_ATTEMPTS
                ):
                    lifecycle.attach_failure(
                        error,
                        phase=_AutomaticCompactionLifecyclePhase.START_PUBLICATION,
                        reason=_automatic_compaction_publication_failure_reason(error),
                        retryable=retryable,
                        recovery_action=(
                            _AutomaticCompactionRecoveryAction.RESUME_SESSION
                            if retryable
                            else _AutomaticCompactionRecoveryAction.FAIL_CLOSED
                        ),
                    )
                    raise
                last_error = error
                await asyncio.sleep(0)
        raise AssertionError("Compaction start publication retry loop did not finish.") from (
            last_error
        )

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
            timeout_s=_AUTOMATIC_COMPACTION_PREDISPATCH_EVENT_STORE_WAIT_TIMEOUT_S,
            timeout_after_cancellation_s=(_CONTEXT_EVENT_STORE_WAIT_AFTER_CANCELLATION_TIMEOUT_S),
        )
        cancellation = outcome.cancellation
        if outcome.timed_out:
            persistence_task.cancel()
            self._executor._retain_detached_task(persistence_task)
            publication_error: BaseException = TimeoutError(
                "Compaction start publication exceeded "
                f"{_AUTOMATIC_COMPACTION_PREDISPATCH_EVENT_STORE_WAIT_TIMEOUT_S:g} seconds."
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

    async def _prepare_automatic_compaction_dispatch_authority(
        self,
        *,
        provider_name: str,
        pricing_provider_name: str,
        model_request: ModelRequest,
        model_attempt_identity: ModelAttemptIdentity,
        parent_model_step_identity: ModelStepIdentity,
        source_transcript_cursor: int,
        step: int,
        attempt: int,
        max_attempts: int,
        billing_identity: BillingIdentity | None,
        reservations: tuple[BudgetStepReservation, ...],
        allow_borrowed_stage: bool,
        fallback: ModelFailoverTransition | None = None,
    ) -> _AutomaticCompactionDispatchAuthority:
        """Prepare exact recovery authority before the budget dispatch fence."""

        provider_name = require_durable_clean_nonblank(
            provider_name,
            "automatic_compaction_provider_name",
        )
        pricing_provider_name = require_durable_clean_nonblank(
            pricing_provider_name,
            "automatic_compaction_pricing_provider_name",
        )
        model_attempt_identity = copy_model_attempt_identity(model_attempt_identity)
        parent_model_step_identity = copy_model_step_identity(parent_model_step_identity)
        if fallback is not None:
            if type(fallback) is not ModelFailoverTransition:
                raise TypeError("Compaction requires an exact live failover handoff.")
            fallback = replace(fallback)
        active = await self._executor._session_store.load_active_model_completion_stage(
            self._session.id
        )
        if fallback is not None and active is None:
            raise RuntimeError("Fallback compaction lost its exact failed predecessor.")
        if active is not None:
            stage = active.stage
            if not allow_borrowed_stage and fallback is None:
                raise RuntimeError(
                    "Initial automatic compaction found an active model-completion stage."
                )
            if fallback is not None:
                selection = self._model_execution_selection
                if (
                    selection is None
                    or selection.candidate_index != fallback.source.candidate_index + 1
                ):
                    raise ValueError("Fallback compaction changed its selected successor.")
                predecessor = replace(selection, candidate_index=fallback.source.candidate_index)
                predecessor.require_prepared_stage(stage)
                if (
                    stage.state != "in_flight"
                    or stage.preparation_digest != fallback.source_preparation_digest
                    or stage.intent[MODEL_FAILOVER_CHECKPOINT_KEY]["successor"]
                    != fallback.source.payload()
                ):
                    raise ValueError("Fallback compaction changed its exact failed predecessor.")
            if (
                stage.purpose != "assistant-turn"
                or stage.logical_step_id != parent_model_step_identity.model_step_id
                or stage.source_transcript_cursor != source_transcript_cursor
            ):
                raise RuntimeError(
                    "Context-overflow compaction cannot borrow a different active model stage."
                )
            dispatch = await self._executor._session_store.load_model_completion_stage_dispatch(
                self._session.id,
                stage.stage_id,
            )
            if dispatch is None:
                raise RuntimeError(
                    "Context-overflow compaction requires the assistant dispatch receipt."
                )
            return _AutomaticCompactionDispatchAuthority(
                stage=stage,
                owns_stage=False,
                provider_name=provider_name,
                step=step,
                attempt=attempt,
                max_attempts=max_attempts,
            )

        interaction_id = _current_session_interaction_id(self._session.id)
        if interaction_id is None:
            raise RuntimeError(
                "Automatic compaction dispatch requires an active interaction identity."
            )
        recovery_context = self._model_completion_recovery_context_factory(
            billing_identity,
            reservations,
        )
        if type(recovery_context) is not ModelCompletionRecoveryContext:
            raise RuntimeError("Automatic compaction dispatch requires durable recovery authority.")
        if recovery_context.interaction_id is None:
            recovery_context = recovery_context.model_copy(
                update={"interaction_id": interaction_id},
                deep=True,
            )
        elif recovery_context.interaction_id != interaction_id:
            raise RuntimeError(
                "Automatic compaction recovery authority belongs to another interaction."
            )
        request_fingerprint = _model_request_fingerprint(
            provider_name=provider_name,
            model_request=model_request,
        )
        logical_step_id = model_attempt_identity.model_step_id
        stage_id = f"{logical_step_id}:dispatch:0"
        intent = {
            "schema_version": 1,
            "purpose": "context-compaction",
            "parent_model_step_id": parent_model_step_identity.model_step_id,
            **model_attempt_identity.payload(),
            "logical_step_id": logical_step_id,
            "provider_name": provider_name,
            "pricing_provider_name": pricing_provider_name,
            "requested_model": model_request.model,
            "source_transcript_cursor": source_transcript_cursor,
            "request_fingerprint": request_fingerprint,
            "interaction_id": interaction_id,
            "recovery_context": recovery_context.model_dump(mode="json"),
        }
        prepared = await self._executor._session_store.prepare_model_completion_stage(
            self._session.id,
            request=ModelCompletionStageRequest(
                stage_id=stage_id,
                logical_step_id=logical_step_id,
                dispatch_ordinal=0,
                purpose="context-compaction",
                intent=intent,
                reservation_ids=tuple(
                    reservation.record.reservation_id for reservation in reservations
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
        return _AutomaticCompactionDispatchAuthority(
            stage=prepared.stage,
            owns_stage=True,
            provider_name=provider_name,
            step=step,
            attempt=attempt,
            max_attempts=max_attempts,
        )

    async def _mark_automatic_compaction_dispatch_authority(
        self,
        authority: _AutomaticCompactionDispatchAuthority,
    ) -> BaseException | None:
        """Commit the provider fence, deferring only an ambiguous acknowledgement."""

        if not authority.owns_stage:
            return None
        try:
            await self._executor._session_store.mark_model_completion_stage_dispatched(
                self._session.id,
                stage=authority.stage,
            )
            return None
        except BaseException as dispatch_failure:
            try:
                dispatch = await self._executor._session_store.load_model_completion_stage_dispatch(
                    self._session.id,
                    authority.stage.stage_id,
                )
            except BaseException as reconciliation_failure:
                dispatch_failure.add_note(
                    "Automatic-compaction dispatch receipt reconciliation also failed: "
                    f"{type(reconciliation_failure).__name__}: {reconciliation_failure}"
                )
                return dispatch_failure
            if dispatch is not None:
                dispatch_failure.add_note(
                    "The automatic-compaction dispatch receipt is durable; provider-effect "
                    "ambiguity remains fenced."
                )
                return dispatch_failure
            raise

    async def _publish_owned_automatic_compaction_completion(
        self,
        *,
        authority: _AutomaticCompactionDispatchAuthority,
        events: list[Event],
    ) -> list[Event]:
        """Complete one owned context-compaction stage before budget settlement."""

        if not authority.owns_stage or authority.stage.purpose != "context-compaction":
            raise ValueError("Automatic compaction completion requires its owned stage.")
        prepared_events = self._executor._event_writer.prepare_many(events)
        publication = RuntimePublicationRequest(
            publication_id=authority.stage.logical_step_id,
            kind="context-compaction",
            interaction_id=authority.stage.intent.get("interaction_id"),
            intent=authority.stage.intent,
            mutation=runtime_publication_checkpoint_mutation(None, None),
            transcript_messages=(),
            events=tuple(prepared_events),
        )

        async def publish_once() -> list[Event]:
            await self._executor._session_store.complete_model_completion_stage(
                self._session.id,
                stage_id=authority.stage.stage_id,
                publication=publication,
            )
            return [event.model_copy(deep=True) for event in prepared_events]

        async def publish_exactly() -> list[Event]:
            try:
                return await publish_once()
            except (Exception, asyncio.CancelledError) as first_error:
                try:
                    await publish_once()
                except (Exception, asyncio.CancelledError) as replay_error:
                    replay_error.add_note(
                        "Exact context-compaction stage publication also failed after "
                        f"{type(first_error).__name__}: {first_error}"
                    )
                    raise replay_error from first_error
                first_error.add_note(
                    "Context-compaction evidence was durable after exact replay; the "
                    "invocation will fail closed without another provider dispatch."
                )
                raise first_error

        publication_task = asyncio.create_task(publish_exactly())
        outcome = await await_shielded_task_outcome(
            publication_task,
            timeout_s=_CONTEXT_EVENT_STORE_WAIT_TIMEOUT_S,
        )
        if outcome.timed_out:
            publication_task.cancel()
            drained = await await_shielded_task_outcome(
                publication_task,
                cancellation=outcome.cancellation,
            )
            error: BaseException | None = TimeoutError(
                "Context-compaction durable publication exceeded "
                f"{_CONTEXT_EVENT_STORE_WAIT_TIMEOUT_S:g} seconds."
            )
            if drained.error is not None:
                error.add_note(
                    "The timed-out context-compaction publication also failed while draining: "
                    f"{type(drained.error).__name__}: {drained.error}"
                )
            result = drained.result
            cancellation = drained.cancellation
        else:
            error = outcome.error
            result = outcome.result
            cancellation = outcome.cancellation
        if isinstance(error, asyncio.CancelledError) and cancellation is None:
            error = unexpected_child_cancellation_error(
                error,
                operation="Context-compaction durable publication",
            )
        if error is not None:
            if cancellation is not None:
                cancellation.add_note(
                    "Context-compaction durable publication also failed: "
                    f"{type(error).__name__}: {error}"
                )
                raise cancellation from error
            raise error
        if result is None:
            raise RuntimeError("Context-compaction durable publication returned no events.")
        if cancellation is not None:
            raise cancellation
        return result

    async def _promote_settled_automatic_compaction_stage(
        self,
        *,
        authority: _AutomaticCompactionDispatchAuthority,
    ) -> list[Event]:
        """Release one completed compaction stage only after budget settlement."""

        if not authority.owns_stage or authority.stage.purpose != "context-compaction":
            raise ValueError("Automatic compaction promotion requires its owned stage.")
        active = await self._executor._session_store.load_active_model_completion_stage(
            self._session.id
        )
        if active is None:
            return []
        if active.stage.stage_id != authority.stage.stage_id:
            raise RuntimeError(
                "Automatic compaction promotion found a different active model stage."
            )
        if active.stage.state == "in_flight":
            return []
        if active.stage.state != "completed":
            raise RuntimeError("Automatic compaction promotion found an unsupported stage state.")
        publication = active.stage.publication
        if publication is None:
            raise RuntimeError(
                "Completed automatic compaction stage lost its terminal publication."
            )
        events = [event.model_copy(deep=True) for event in publication.events]
        lost_acknowledgement: BaseException | None = None
        try:
            await self._executor._session_store.promote_model_completion_stage(
                self._session.id,
                stage_id=authority.stage.stage_id,
                expected_run_epoch=self._session.run_epoch,
            )
        except BaseException as promotion_failure:
            reconciled = await self._executor._session_store.load_active_model_completion_stage(
                self._session.id
            )
            if reconciled is None:
                lost_acknowledgement = promotion_failure
            elif (
                reconciled.stage.stage_id != authority.stage.stage_id
                or reconciled.stage.state != "completed"
            ):
                raise RuntimeError(
                    "Automatic compaction promotion reconciliation found conflicting state."
                ) from promotion_failure
            else:
                raise
        fanout_failure, cancellation = await self._fan_out_reconciled_automatic_compaction_events(
            events,
            cancellation=None,
            operation="Automatic-compaction promotion",
        )
        if cancellation is not None:
            secondary: BaseException | None
            if lost_acknowledgement is not None and fanout_failure is not None:
                secondary = _combine_authoritative_model_failure(
                    lost_acknowledgement,
                    fanout_failure,
                    message=(
                        "Automatic-compaction promotion acknowledgement and event "
                        "fan-out both failed during cancellation."
                    ),
                )
            else:
                secondary = lost_acknowledgement or fanout_failure
            cancellation.add_note(
                "Automatic-compaction event fan-out was interrupted after durable promotion."
            )
            if secondary is not None:
                raise cancellation from secondary
            raise cancellation
        if fanout_failure is not None:
            if lost_acknowledgement is None:
                raise fanout_failure
            raise _combine_authoritative_model_failure(
                lost_acknowledgement,
                fanout_failure,
                message=(
                    "Automatic-compaction promotion acknowledgement and event fan-out both failed."
                ),
            ) from None
        if lost_acknowledgement is not None:
            lost_acknowledgement.add_note(
                "Automatic-compaction promotion was durable after reconciliation."
            )
            raise lost_acknowledgement
        return [event.model_copy(deep=True) for event in events]

    async def _persist_automatic_compaction_completions(
        self,
        payloads: list[dict[str, Any]],
        *,
        published_attempt_ids: set[str],
        published_events: list[Event],
        completion_events: dict[str, Event],
        dispatch_authorities: dict[str, _AutomaticCompactionDispatchAuthority],
        settled_attempt_ids: set[str],
    ) -> None:
        """Commit finalized provider evidence before another compactor dispatch."""

        pending: list[tuple[str, Event]] = []
        for payload in payloads:
            attempt_id = payload.get(_COMPACTION_ATTEMPT_ID_KEY)
            if type(attempt_id) is not str:
                raise RuntimeError("Compaction completion evidence lost its attempt identity.")
            if attempt_id in published_attempt_ids:
                continue
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
            authority = dispatch_authorities.get(execution_identity.model_attempt_id)
            if authority is None:
                raise RuntimeError(
                    "Compaction completion has no durable provider-dispatch authority."
                )
            provider_error = model_provider_error_from_payload(
                payload,
                fallback_provider=authority.provider_name,
                fallback_message="Automatic compaction provider error",
            )
            deadline_error = (
                provider_error if isinstance(provider_error, ModelStreamDeadlineError) else None
            )
            if deadline_error is not None:
                # Automatic compaction uses a synchronous model stage and has no
                # durable provider-operation identity to reattach. Recovery
                # disposition therefore comes from this runtime-owned authority,
                # never from provider-supplied exception payloads.
                deadline_error = _deadline_with_runtime_recovery_authority(
                    deadline_error,
                    exact_operation=False,
                )
            event = completion_events.get(attempt_id)
            if event is None:
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
            parent_model_step_id = (
                authority.stage.intent.get("parent_model_step_id")
                if authority.owns_stage
                else authority.stage.logical_step_id
            )
            if type(parent_model_step_id) is not str:
                raise RuntimeError(
                    "Automatic compaction stage lost its parent model-step identity."
                )
            parent_model_step_id = require_durable_clean_nonblank(
                parent_model_step_id,
                "parent_model_step_id",
            )
            event_payload = copy_durable_json_object(event.payload, "payload")
            event_payload["parent_model_step_id"] = parent_model_step_id
            event = event.model_copy(update={"payload": event_payload}, deep=True)
            event = event_with_runtime_payload_authority(
                event,
                "parent_model_step_id",
            )
            completion_events[attempt_id] = event.model_copy(deep=True)
            error_payload: dict[str, Any] | None = None
            if deadline_error is not None:
                error_payload = {
                    "error": str(deadline_error),
                    "error_type": type(deadline_error).__name__,
                    "stage": "context_compaction_stream",
                    "purpose": ModelCompletionPurpose.CONTEXT_COMPACTION.value,
                    "compactor": payload.get("compactor"),
                    "compaction_outcome": payload.get("compaction_outcome"),
                    "step": authority.step,
                    "attempt": authority.attempt,
                    "max_attempts": authority.max_attempts,
                    _COMPACTION_ATTEMPT_ID_KEY: attempt_id,
                    "model_completion_stage_id": authority.stage.stage_id,
                    **execution_identity.payload(),
                    **deadline_error.error_payload_fields(),
                }
            elif (
                provider_error is not None
                and payload.get("compaction_outcome") == "provider_error"
                and type(payload.get("retry_disposition")) is str
            ):
                # The completion above remains the attempt's accounting record.
                # Publish the provider rejection with the same classification
                # and retry decision as a failed model-step attempt.
                error_payload = {
                    "error": str(provider_error),
                    "error_type": type(provider_error).__name__,
                    "stage": "context_compaction_stream",
                    "purpose": ModelCompletionPurpose.CONTEXT_COMPACTION.value,
                    "compactor": payload.get("compactor"),
                    "compaction_outcome": "provider_error",
                    "provider_name": authority.provider_name,
                    "step": authority.step,
                    "attempt": authority.attempt,
                    "max_attempts": authority.max_attempts,
                    _COMPACTION_ATTEMPT_ID_KEY: attempt_id,
                    "model_completion_stage_id": authority.stage.stage_id,
                    **execution_identity.payload(),
                    **provider_error.error_payload_fields(),
                    **{
                        key: payload[key]
                        for key in (
                            "retry",
                            "retry_disposition",
                            "retry_suppression",
                            "provider_retryable",
                            "effective_max_attempts",
                            "reason",
                        )
                        if key in payload
                    },
                }
            error_event = (
                None
                if error_payload is None
                else event_with_execution_profile_authority(
                    _event_with_model_identity_authority(
                        Event(
                            type=EventType.MODEL_ERROR,
                            session_id=self._session.id,
                            agent_name=self._registered_agent.spec.name,
                            environment_name=self._environment_name,
                            payload=error_payload,
                        ),
                        execution_identity,
                    ),
                    self._execution_profile,
                )
            )
            if authority.owns_stage and deadline_error is None:
                # Keep rejection diagnostics in the durable stage handoff. A
                # separate write after promotion would lose them on ack failure
                # or process exit, once the completion suppresses later replay.
                owned_events = [event] if error_event is None else [event, error_event]
                promotion_allowed = (
                    not authority.stage.reservation_ids
                    or execution_identity.model_attempt_id in settled_attempt_ids
                )
                try:
                    persisted_events = await self._publish_owned_automatic_compaction_completion(
                        authority=authority,
                        events=owned_events,
                    )
                    if promotion_allowed:
                        await self._promote_settled_automatic_compaction_stage(
                            authority=authority,
                        )
                except BaseException as publication_error:
                    try:
                        event_durable = await self._executor._event_writer.is_persisted(event)
                    except BaseException as reconciliation_error:
                        if isinstance(reconciliation_error, asyncio.CancelledError):
                            if not authority.stage.reservation_ids:
                                try:
                                    await self._promote_settled_automatic_compaction_stage(
                                        authority=authority,
                                    )
                                except BaseException as promotion_failure:
                                    reconciliation_error.add_note(
                                        "Completed context-compaction promotion also failed "
                                        "during cancelled publication reconciliation."
                                    )
                                    raise reconciliation_error from promotion_failure
                            if publication_error is reconciliation_error:
                                raise
                            raise reconciliation_error from publication_error
                        publication_error.add_note(
                            "Context-compaction stage publication reconciliation also failed: "
                            f"{type(reconciliation_error).__name__}: {reconciliation_error}"
                        )
                        raise publication_error from reconciliation_error
                    if not event_durable and promotion_allowed:
                        try:
                            await self._promote_settled_automatic_compaction_stage(
                                authority=authority,
                            )
                        except BaseException as promotion_failure:
                            publication_error.add_note(
                                "Reconciled context-compaction completion promotion also failed."
                            )
                            raise publication_error from promotion_failure
                        event_durable = True
                    if event_durable:
                        # Exact stage replay proved the handoff durable even
                        # though this invocation must still fail closed. Keep
                        # later context-failure persistence from duplicating it.
                        published_attempt_ids.add(attempt_id)
                        published_events.extend(item.model_copy(deep=True) for item in owned_events)
                    raise publication_error
                completion_events[attempt_id] = persisted_events[0].model_copy(deep=True)
                published_attempt_ids.add(attempt_id)
                if promotion_allowed:
                    published_events.extend(persisted_events)
                continue
            pending.append(
                (
                    attempt_id,
                    event.model_copy(deep=True),
                )
            )
            if error_event is not None:
                pending.append((attempt_id, error_event))
        if not pending:
            return

        events = [event for _attempt_id, event in pending]
        persistence_task = asyncio.create_task(
            self._executor._event_writer.persist_many(self._session.id, events)
        )
        outcome = await await_shielded_task_outcome(
            persistence_task,
            timeout_s=_CONTEXT_EVENT_STORE_WAIT_TIMEOUT_S,
            timeout_after_cancellation_s=(_CONTEXT_EVENT_STORE_WAIT_AFTER_CANCELLATION_TIMEOUT_S),
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
        shield_persistence: bool = True,
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
        unrepresented_compaction_calls = False
        for telemetry in compaction_telemetry:
            # Completion telemetry is ordered by pass, including already
            # published calls. Only a validated positive-coverage result can
            # acknowledge the successful work preceding it. An older prefix
            # checkpoint must not clear recovery for a later unfinished hierarchy.
            if (
                telemetry.event_type == EventType.MODEL_COMPLETED
                and telemetry.payload.get("compaction_outcome") is None
            ):
                unrepresented_compaction_calls = True
            elif telemetry.event_type == EventType.CONTEXT_COMPACTION_COMPLETED:
                covered = telemetry.payload.get("newly_compacted_message_count")
                if type(covered) is int and covered > 0:
                    unrepresented_compaction_calls = False
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

            # This recovery flag is derived by the runtime, never accepted as
            # policy-supplied authority. Omission retains the legacy full-step
            # acknowledgement only when all successful calls are represented.
            checkpoint_payload = {
                key: value
                for key, value in checkpoint_event_payload.items()
                if key != _COMPACTION_UNREPRESENTED_CALLS_KEY
            }
            if unrepresented_compaction_calls:
                checkpoint_payload[_COMPACTION_UNREPRESENTED_CALLS_KEY] = True
            checkpoint_event = event_with_execution_profile_authority(
                Event(
                    type=EventType.SESSION_CHECKPOINTED,
                    session_id=self._session.id,
                    agent_name=self._registered_agent.spec.name,
                    environment_name=self._environment_name,
                    payload={
                        **checkpoint_payload,
                        **model_step_identity.payload(),
                    },
                ),
                self._execution_profile,
            )
            checkpoint_event = _event_with_model_identity_authority(
                checkpoint_event, model_step_identity
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
                if not shield_persistence and isinstance(publication_error, asyncio.CancelledError):
                    # The outer termination owner has bounded and cancelled this
                    # actual writer. Do not start unbounded reconciliation after
                    # its deadline; atomic store evidence remains authoritative.
                    raise
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

        if not shield_persistence:
            # Termination cleanup already owns a bounded, shielded task. Nesting
            # another shield would leave its writer alive after the outer timeout.
            return await persist()
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
        progress_error = context_build_termination_checkpoint_error(error)
        if progress_error is not None:
            sanitize_context_build_error_checkpoint(
                progress_error, redactor=self._executor._secret_redactor
            )
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
        if not unpublished_telemetry and progress_error is None:
            return

        async def persist() -> None:
            if progress_error is not None and progress_error.checkpoint is not None:
                _, persistence_error = await self._persist_context_events(
                    model_step_identity=model_step_identity,
                    compaction_identity_ledger=compaction_identity_ledger,
                    compaction_telemetry=list(telemetry),
                    recall_telemetry=[],
                    checkpoint_update=progress_error.checkpoint,
                    checkpoint_event_payload=progress_error.checkpoint_event_payload,
                    published_compaction_attempt_ids=published_compaction_attempt_ids,
                    compaction_completion_events=compaction_completion_events,
                    compaction_start_event=compaction_start_event,
                    compaction_started_published=compaction_start_durable,
                    shield_persistence=False,
                )
                if persistence_error is not None:
                    raise persistence_error
                return
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
            self._executor._retain_detached_task(task)
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
        pricing_provider_name = (
            None
            if self._model_execution_selection is None
            else self._request_provider.billing_provider_name
            or self._request_registered_provider.name
        )
        model = None if self._model_execution_selection is None else self._request_model
        budget_evaluation = await self._limit_gate.evaluate_budget(
            self._budget_policy,
            execution_identity=model_step_identity,
            pricing_provider_name=pricing_provider_name,
            model=model,
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
            invocation_context=self._invocation_context,
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
            pricing_provider_name=pricing_provider_name,
            model=model,
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
            invocation_context=self._invocation_context,
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
        pricing_provider_name = (
            None
            if self._model_execution_selection is None
            else self._request_provider.billing_provider_name
            or self._request_registered_provider.name
        )
        model = None if self._model_execution_selection is None else self._request_model
        budget_evaluation = await self._limit_gate.evaluate_budget(
            self._budget_policy,
            billing_identity_state=resolved_billing_identity(billing_identity),
            execution_identity=model_attempt_identity,
            pricing_provider_name=pricing_provider_name,
            model=model,
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
            invocation_context=self._invocation_context,
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
            pricing_provider_name=pricing_provider_name,
            model=model,
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
            invocation_context=self._invocation_context,
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
            invocation_context=self._invocation_context,
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
        targeted_tool_projection_kind: TargetedToolProjectionKind | None,
        targeted_tool_gateway: TargetedToolGatewayProjection | None,
        targeted_tool_native: TargetedToolProjectionRequest | None,
        tool_discovery_native_tool_names: tuple[str, ...],
    ) -> Callable[[list[Message]], Awaitable[int | None]]:
        @timed_phase("counting")
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
                targeted_tool_projection_kind=targeted_tool_projection_kind,
                targeted_tool_gateway=targeted_tool_gateway,
                targeted_tool_native=targeted_tool_native,
                tool_discovery_projection_kind=self._tool_discovery_projection_kind,
                tool_discovery_native_tool_names=tool_discovery_native_tool_names,
                model_execution_selection=self._model_execution_selection,
            )
            # Context-policy execution can await arbitrary application code.
            # Reject changed provider semantics at the final remote count seam.
            try:
                await self._refresh_live_model_semantics()
            except Exception as authority_error:
                raise _ContextCountAuthorityError(authority_error) from None
            from cayu.providers.base import has_peer_content

            if has_peer_content(request):
                return None
            redactor_token = bind_provider_error_workload_redactor(self._executor._secret_redactor)
            try:
                result = await self._request_provider.count_input_tokens(request)
            finally:
                reset_provider_error_workload_redactor(redactor_token)
            return None if result is None else result.input_tokens

        return count_input_tokens

    def _cache_prefix_request_builder(
        self,
        *,
        step: int,
        tool_exposure: ResolvedToolExposure,
        targeted_tool_projection_kind: TargetedToolProjectionKind | None,
        targeted_tool_gateway: TargetedToolGatewayProjection | None,
        targeted_tool_native: TargetedToolProjectionRequest | None,
        tool_discovery_native_tool_names: tuple[str, ...],
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
                targeted_tool_projection_kind=targeted_tool_projection_kind,
                targeted_tool_gateway=targeted_tool_gateway,
                targeted_tool_native=targeted_tool_native,
                tool_discovery_projection_kind=self._tool_discovery_projection_kind,
                tool_discovery_native_tool_names=tool_discovery_native_tool_names,
                model_execution_selection=self._model_execution_selection,
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
        dispatch_authorities: dict[str, _AutomaticCompactionDispatchAuthority],
        settled_attempt_ids: set[str],
        source_transcript_cursor: int,
        allow_borrowed_stage: bool,
        lifecycle: _AutomaticCompactionLifecycle,
        fallback: ModelFailoverTransition | None = None,
    ) -> CompactionResult:
        del messages
        model_step_identity = copy_model_step_identity(model_step_identity)
        if compaction_identity_ledger.model_step_identity != model_step_identity:
            raise ValueError("Compaction identity ledger belongs to a different model step.")
        if type(lifecycle) is not _AutomaticCompactionLifecycle:
            raise TypeError("lifecycle must be an _AutomaticCompactionLifecycle.")
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

        async def prepare_dispatch_authority(
            *,
            provider_name: str,
            pricing_provider_name: str,
            model_request: ModelRequest,
            model_attempt_identity: ModelAttemptIdentity,
            attempt: int,
            max_attempts: int,
            billing_identity: BillingIdentity | None,
            reservations: tuple[BudgetStepReservation, ...],
        ) -> None:
            authority = await self._prepare_automatic_compaction_dispatch_authority(
                provider_name=provider_name,
                pricing_provider_name=pricing_provider_name,
                model_request=model_request,
                model_attempt_identity=model_attempt_identity,
                parent_model_step_identity=model_step_identity,
                source_transcript_cursor=source_transcript_cursor,
                step=step,
                attempt=attempt,
                max_attempts=max_attempts,
                billing_identity=billing_identity,
                reservations=reservations,
                allow_borrowed_stage=allow_borrowed_stage,
                fallback=fallback,
            )
            existing = dispatch_authorities.setdefault(
                model_attempt_identity.model_attempt_id,
                authority,
            )
            if existing != authority:
                raise RuntimeError(
                    "Compaction provider attempt has conflicting durable stage authority."
                )
            if not authority.owns_stage:
                deferred_failure = await (
                    controller.prepare_borrowed_automatic_compaction_budget_authority(
                        session=self._session,
                        stage=authority.stage,
                        provider_name=provider_name,
                        pricing_provider_name=pricing_provider_name,
                        model=model_request.model,
                        model_attempt_identity=model_attempt_identity,
                        reservations=reservations,
                    )
                )
                if deferred_failure is not None:
                    raise deferred_failure

        def prepared_dispatch_authority(
            model_attempt_identity: ModelAttemptIdentity,
        ) -> _AutomaticCompactionDispatchAuthority:
            authority = dispatch_authorities.get(model_attempt_identity.model_attempt_id)
            if authority is None:
                raise RuntimeError("Compaction provider attempt lost its prepared stage authority.")
            return authority

        async def mark_dispatch_authority(
            model_attempt_identity: ModelAttemptIdentity,
        ) -> BaseException | None:
            authority = prepared_dispatch_authority(model_attempt_identity)
            deferred_failure = await self._mark_automatic_compaction_dispatch_authority(authority)
            if deferred_failure is not None:
                return deferred_failure
            try:
                await self._refresh_live_model_semantics()
            except BaseException as authority_failure:
                # The durable stage receipt has already crossed the last local
                # provider fence. Fail closed without entering provider code.
                return authority_failure
            return None

        async def promote_settled_dispatch_authority(
            model_attempt_identity: ModelAttemptIdentity,
            *,
            authoritative_failure: BaseException | None = None,
        ) -> None:
            authority = prepared_dispatch_authority(model_attempt_identity)
            if not authority.owns_stage:
                if authoritative_failure is not None:
                    raise authoritative_failure
                return
            try:
                promoted_events = await self._promote_settled_automatic_compaction_stage(
                    authority=authority,
                )
            except BaseException as promotion_failure:
                if authoritative_failure is None:
                    raise
                add_exception_note_safely(
                    authoritative_failure,
                    (
                        "Settled automatic-compaction stage promotion also failed: "
                        f"{type(promotion_failure).__name__}: {promotion_failure}"
                    ),
                )
                raise authoritative_failure from promotion_failure
            known_event_ids = {event.id for event in budget_events}
            budget_events.extend(
                event for event in promoted_events if event.id not in known_event_ids
            )
            if authoritative_failure is not None:
                raise authoritative_failure

        async def abandon_pre_provider_dispatch(
            model_attempt_identity: ModelAttemptIdentity,
            failure: BaseException,
        ) -> None:
            authority = dispatch_authorities.get(model_attempt_identity.model_attempt_id)
            if authority is None or not authority.owns_stage:
                return
            await self._abandon_pre_dispatch_model_stage(
                authority.stage,
                authoritative_failure=failure,
                # The budget fence uses the independent compaction attempt,
                # while ordinary assistant stages use their stage identity.
                budget_dispatch_id=model_attempt_identity.model_attempt_id,
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
                    await self._persist_automatic_compaction_predispatch_event(
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
                        ),
                        phase=(_AutomaticCompactionLifecyclePhase.REQUEST_FOOTPRINT_PUBLICATION),
                        lifecycle=lifecycle,
                    )
                )
            budget_events.append(
                await self._persist_automatic_compaction_predispatch_event(
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
                    ),
                    phase=_AutomaticCompactionLifecyclePhase.MODEL_START_PUBLICATION,
                    lifecycle=lifecycle,
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

            del actual_model, actual_usage_dialect
            await self._refresh_live_model_semantics()
            model_attempt_identity = compaction_identity_ledger.begin_dispatch()
            deadline_admission = ProviderStreamDeadlineAdmission(provider.stream_deadlines)
            try:
                await record_compaction_footprint(
                    provider=provider,
                    provider_name=actual_provider_name,
                    model_request=model_request,
                    attempt=attempt,
                    max_attempts=max_attempts,
                    model_attempt_identity=model_attempt_identity,
                )
                await prepare_dispatch_authority(
                    provider_name=actual_provider_name,
                    pricing_provider_name=actual_pricing_provider_name,
                    model_request=model_request,
                    model_attempt_identity=model_attempt_identity,
                    attempt=attempt,
                    max_attempts=max_attempts,
                    billing_identity=billing_identity,
                    reservations=(),
                )
                try:
                    deferred_dispatch_failure = await mark_dispatch_authority(
                        model_attempt_identity
                    )
                except BaseException as dispatch_failure:
                    await abandon_pre_provider_dispatch(
                        model_attempt_identity,
                        dispatch_failure,
                    )
                    raise
                if deferred_dispatch_failure is not None:
                    raise deferred_dispatch_failure
                await self._refresh_live_model_semantics()
                lifecycle.provider_dispatch_disposition = (
                    _AutomaticCompactionDispatchDisposition.UNKNOWN
                )
                token = bind_provider_deadline_admission(deadline_admission)
                try:
                    try:
                        with _compaction_model_attempt_identity_scope(model_attempt_identity):
                            result = await dispatch()
                    except BaseException as dispatch_failure:
                        if (
                            automatic_compaction_failure_disposition_payload(dispatch_failure)
                            is None
                        ):
                            lifecycle.attach_failure(
                                dispatch_failure,
                                phase=_AutomaticCompactionLifecyclePhase.PROVIDER_DISPATCH,
                                reason=(
                                    _AutomaticCompactionFailureReason.CANCELLED
                                    if isinstance(dispatch_failure, asyncio.CancelledError)
                                    else _AutomaticCompactionFailureReason.PROVIDER_FAILED
                                ),
                                retryable=False,
                                recovery_action=(
                                    _AutomaticCompactionRecoveryAction.RECONCILE_COMPLETION
                                ),
                            )
                        settled_attempt_ids.add(model_attempt_identity.model_attempt_id)
                        await promote_settled_dispatch_authority(
                            model_attempt_identity,
                            authoritative_failure=dispatch_failure,
                        )
                        raise AssertionError(
                            "Unreachable compaction failure handoff."
                        ) from dispatch_failure
                    lifecycle.provider_dispatch_disposition = (
                        _AutomaticCompactionDispatchDisposition.DISPATCHED
                    )
                    settled_attempt_ids.add(model_attempt_identity.model_attempt_id)
                    await promote_settled_dispatch_authority(model_attempt_identity)
                    return result
                finally:
                    reset_provider_deadline_admission(token)
            finally:
                deadline_admission.close()
                compaction_identity_ledger.end_dispatch(model_attempt_identity)

        async def execute_with_post_dispatch_failure_disposition() -> CompactionResult:
            try:
                return await execute()
            except BaseException as error:
                if (
                    lifecycle.provider_dispatch_disposition
                    != _AutomaticCompactionDispatchDisposition.NOT_DISPATCHED
                    and automatic_compaction_failure_disposition_payload(error) is None
                ):
                    lifecycle.attach_failure(
                        error,
                        phase=_AutomaticCompactionLifecyclePhase.PROVIDER_DISPATCH,
                        reason=(
                            _AutomaticCompactionFailureReason.CANCELLED
                            if isinstance(error, asyncio.CancelledError)
                            else _AutomaticCompactionFailureReason.INTERNAL_FAILED
                        ),
                        retryable=False,
                        recovery_action=(_AutomaticCompactionRecoveryAction.RECONCILE_COMPLETION),
                    )
                raise

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
                with (
                    _automatic_compaction_dispatch_runner_scope(run_identity_only_dispatch),
                    _compaction_environment_admission_scope(self._refresh_live_model_semantics),
                ):
                    return await execute_with_post_dispatch_failure_disposition()
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
            with (
                _automatic_compaction_dispatch_runner_scope(run_identity_only_dispatch),
                _compaction_environment_admission_scope(self._refresh_live_model_semantics),
            ):
                return await execute_with_post_dispatch_failure_disposition()
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
            await self._refresh_live_model_semantics()
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
            deadline_admission: ProviderStreamDeadlineAdmission | None = None
            before_count = len(completed_payloads())
            try:

                async def identified_dispatch() -> tuple[str, dict[str, Any]]:
                    await self._refresh_live_model_semantics()
                    if deadline_admission is None:
                        raise RuntimeError("Compaction deadline admission was not prepared.")
                    lifecycle.provider_dispatch_disposition = (
                        _AutomaticCompactionDispatchDisposition.UNKNOWN
                    )
                    token = bind_provider_deadline_admission(deadline_admission)
                    try:
                        try:
                            with _compaction_model_attempt_identity_scope(model_attempt_identity):
                                result = await dispatch()
                        except BaseException as error:
                            if automatic_compaction_failure_disposition_payload(error) is None:
                                lifecycle.attach_failure(
                                    error,
                                    phase=_AutomaticCompactionLifecyclePhase.PROVIDER_DISPATCH,
                                    reason=(
                                        _AutomaticCompactionFailureReason.CANCELLED
                                        if isinstance(error, asyncio.CancelledError)
                                        else _AutomaticCompactionFailureReason.PROVIDER_FAILED
                                    ),
                                    retryable=False,
                                    recovery_action=(
                                        _AutomaticCompactionRecoveryAction.RECONCILE_COMPLETION
                                    ),
                                )
                            raise
                        lifecycle.provider_dispatch_disposition = (
                            _AutomaticCompactionDispatchDisposition.DISPATCHED
                        )
                        return result
                    finally:
                        reset_provider_deadline_admission(token)

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
                    stopped = _AutomaticCompactionAdmissionStopped(
                        budget_evaluation=budget_evaluation
                    )
                    lifecycle.attach_failure(
                        stopped,
                        phase=_AutomaticCompactionLifecyclePhase.BUDGET_ADMISSION,
                        reason=_AutomaticCompactionFailureReason.ADMISSION_REJECTED,
                        retryable=False,
                        recovery_action=_AutomaticCompactionRecoveryAction.STOP_SESSION,
                    )
                    raise stopped
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
                    stopped = _AutomaticCompactionAdmissionStopped(
                        limit_evaluation=limit_evaluation
                    )
                    lifecycle.attach_failure(
                        stopped,
                        phase=_AutomaticCompactionLifecyclePhase.BUDGET_ADMISSION,
                        reason=_AutomaticCompactionFailureReason.ADMISSION_REJECTED,
                        retryable=False,
                        recovery_action=_AutomaticCompactionRecoveryAction.STOP_SESSION,
                    )
                    raise stopped
                budget_events.extend(limit_evaluation.events)
                deadline_admission = ProviderStreamDeadlineAdmission(provider.stream_deadlines)
                if not limits:
                    await self._refresh_live_model_semantics()
                    await record_compaction_footprint(
                        provider=provider,
                        provider_name=actual_provider_name,
                        model_request=model_request,
                        attempt=attempt,
                        max_attempts=max_attempts,
                        model_attempt_identity=model_attempt_identity,
                    )
                    await prepare_dispatch_authority(
                        provider_name=actual_provider_name,
                        pricing_provider_name=actual_pricing_provider_name,
                        model_request=model_request,
                        model_attempt_identity=model_attempt_identity,
                        attempt=attempt,
                        max_attempts=max_attempts,
                        billing_identity=billing_identity,
                        reservations=(),
                    )
                    try:
                        deferred_dispatch_failure = await mark_dispatch_authority(
                            model_attempt_identity
                        )
                    except BaseException as dispatch_failure:
                        await abandon_pre_provider_dispatch(
                            model_attempt_identity,
                            dispatch_failure,
                        )
                        raise
                    if deferred_dispatch_failure is not None:
                        raise deferred_dispatch_failure
                    try:
                        result = await identified_dispatch()
                    except BaseException as dispatch_failure:
                        settled_attempt_ids.add(model_attempt_identity.model_attempt_id)
                        await promote_settled_dispatch_authority(
                            model_attempt_identity,
                            authoritative_failure=dispatch_failure,
                        )
                        raise AssertionError(
                            "Unreachable compaction failure handoff."
                        ) from dispatch_failure
                    settled_attempt_ids.add(model_attempt_identity.model_attempt_id)
                    await promote_settled_dispatch_authority(model_attempt_identity)
                    return result

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

                async def stage_prepared_compaction(
                    reservations: tuple[BudgetStepReservation, ...],
                ) -> None:
                    self._validate_live_model_semantics()
                    await prepare_dispatch_authority(
                        provider_name=actual_provider_name,
                        pricing_provider_name=actual_pricing_provider_name,
                        model_request=model_request,
                        model_attempt_identity=model_attempt_identity,
                        attempt=attempt,
                        max_attempts=max_attempts,
                        billing_identity=billing_identity,
                        reservations=reservations,
                    )

                async def stage_dispatched_compaction(
                    reservations: tuple[BudgetStepReservation, ...],
                ) -> BaseException | None:
                    del reservations
                    return await mark_dispatch_authority(model_attempt_identity)

                async def abandon_stage_before_provider_dispatch(
                    failure: BaseException,
                ) -> None:
                    await abandon_pre_provider_dispatch(
                        model_attempt_identity,
                        failure,
                    )

                budgeted_dispatch = controller.run_automatic_compaction_dispatch(
                    identified_dispatch,
                    completed_events=completed_events,
                    prior_completion_events=prior_completion_events,
                    operation_parent_model_step_id=model_step_identity.model_step_id,
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
                    before_reservations_dispatched=stage_prepared_compaction,
                    after_reservations_dispatched=stage_dispatched_compaction,
                    on_pre_provider_dispatch_failure=(abandon_stage_before_provider_dispatch),
                )
                outcome = await budgeted_dispatch
                budget_events.extend(outcome.events)
                settlement_failed = (
                    isinstance(outcome, BudgetedOperationFailed)
                    and outcome.error.__dict__.get("_cayu_compaction_budget_settlement_failed")
                    is True
                )
                if not settlement_failed:
                    settled_attempt_ids.add(model_attempt_identity.model_attempt_id)
                authority = dispatch_authorities.get(model_attempt_identity.model_attempt_id)
                if authority is not None and not authority.owns_stage:
                    try:
                        recovered_budget_events = await (
                            controller.reconcile_borrowed_automatic_compaction_budget_authority(
                                session=self._session,
                                stage=authority.stage,
                                allow_outcome_unknown=True,
                            )
                        )
                    except BaseException as reconciliation_failure:
                        if isinstance(outcome, BudgetedOperationFailed):
                            combined = _combine_authoritative_model_failure(
                                outcome.error,
                                reconciliation_failure,
                                message=(
                                    "Automatic compaction failed while its borrowed-stage "
                                    "budget authority was reconciled."
                                ),
                            )
                            if outcome.cause is not None:
                                raise combined from outcome.cause
                            raise combined from None
                        raise
                    known_budget_event_ids = {event.id for event in budget_events}
                    budget_events.extend(
                        event
                        for event in recovered_budget_events
                        if event.id not in known_budget_event_ids
                    )
                if authority is not None and authority.owns_stage and not settlement_failed:
                    if isinstance(outcome, BudgetedOperationFailed):
                        try:
                            await promote_settled_dispatch_authority(
                                model_attempt_identity,
                            )
                        except BaseException as promotion_failure:
                            combined = _combine_authoritative_model_failure(
                                outcome.error,
                                promotion_failure,
                                message=(
                                    "Automatic compaction failed while its settled stage "
                                    "was promoted."
                                ),
                            )
                            if outcome.cause is not None:
                                raise combined from outcome.cause
                            raise combined from None
                    else:
                        await promote_settled_dispatch_authority(model_attempt_identity)
                if isinstance(outcome, BudgetedOperationSucceeded):
                    return cast("tuple[str, dict[str, Any]]", outcome.result)
                if isinstance(outcome, BudgetedOperationRejected):
                    reservation_failed = _AutomaticCompactionBudgetReservationFailed(
                        outcome.failure
                    )
                    lifecycle.attach_failure(
                        reservation_failed,
                        phase=_AutomaticCompactionLifecyclePhase.BUDGET_RESERVATION,
                        reason=_AutomaticCompactionFailureReason.RESERVATION_FAILED,
                        retryable=False,
                        recovery_action=_AutomaticCompactionRecoveryAction.STOP_SESSION,
                    )
                    raise reservation_failed
                if automatic_compaction_failure_disposition_payload(outcome.error) is None:
                    lifecycle.attach_failure(
                        outcome.error,
                        phase=_AutomaticCompactionLifecyclePhase.BUDGET_RESERVATION,
                        reason=(
                            _AutomaticCompactionFailureReason.CANCELLED
                            if isinstance(outcome.error, asyncio.CancelledError)
                            else _AutomaticCompactionFailureReason.RESERVATION_FAILED
                        ),
                        retryable=(
                            lifecycle.provider_dispatch_disposition
                            == _AutomaticCompactionDispatchDisposition.NOT_DISPATCHED
                            and isinstance(outcome.error, Exception)
                        ),
                        recovery_action=(
                            _AutomaticCompactionRecoveryAction.RESUME_SESSION
                            if lifecycle.provider_dispatch_disposition
                            == _AutomaticCompactionDispatchDisposition.NOT_DISPATCHED
                            else _AutomaticCompactionRecoveryAction.RECONCILE_COMPLETION
                        ),
                    )
                if outcome.cause is not None:
                    raise outcome.error from outcome.cause
                raise outcome.error
            finally:
                if deadline_admission is not None:
                    deadline_admission.close()
                compaction_identity_ledger.end_dispatch(model_attempt_identity)

        with (
            _automatic_compaction_dispatch_runner_scope(run_provider_dispatch),
            _compaction_environment_admission_scope(self._refresh_live_model_semantics),
        ):
            return await execute_with_post_dispatch_failure_disposition()


def _session_agent_spec(
    *,
    registered_agent: runtime_records.RegisteredAgentState,
    session: Session,
    model_execution_selection: ModelExecutionSelection | None = None,
) -> AgentSpec:
    return AgentSpec(
        name=registered_agent.spec.name,
        model=session.model
        if model_execution_selection is None
        else model_execution_selection.model,
        provider_name=(
            session.provider_name
            if model_execution_selection is None
            else model_execution_selection.registered_provider.name
        ),
        system_prompt=registered_agent.spec.system_prompt,
        metadata=copy_durable_metadata(registered_agent.spec.metadata),
        provider_options=copy_json_value(
            registered_agent.spec.provider_options,
            "provider_options",
        ),
    )


def preflight_model_thinking(
    *,
    provider: ModelProvider,
    model: str,
    thinking: object,
    redactor: SecretRedactor,
) -> None:
    """Validate detached neutral controls without rewriting request options."""

    copied = copy_preflight_thinking(thinking)
    if copied is not None:
        payload = copied.model_dump()
        if redactor.redact_json_values(payload) != payload:
            raise ValueError("Thinking controls contain a workload secret and cannot be forwarded.")
    provider.preflight_thinking(model=model, thinking=copied)


def preflight_portable_model_material(
    *,
    provider: ModelProvider,
    model: str,
    messages: list[Message],
    tools: list[dict[str, Any]],
    redactor: SecretRedactor,
) -> None:
    """Validate neutral capability material without exposing mutable dispatch input.

    This projection is only for preflight. The selected request retains native
    continuation state according to its durable portable-history boundary.
    """

    projection = project_portable_transcript(messages)
    hook_messages = [
        redact_runtime_message_for_boundary(
            message, redactor=redactor, field_name="portable_model_message"
        )
        for message in projection.messages
    ]
    hook_tools = _redacted_provider_tool_definitions(
        tools, redactor=redactor, field_name="portable_model_tools"
    )
    try:
        provider.preflight_portable_messages(model=model, messages=hook_messages, tools=hook_tools)
    finally:
        hook_messages.clear()
        hook_tools.clear()


def _model_request_tools(
    *,
    tool_exposure: ResolvedToolExposure,
    structured_output: StructuredOutputSpec | None,
    targeted_tool_projection: TargetedToolProjectionKind | None = None,
    tool_discovery_mode: ToolDiscoveryMode | None = None,
) -> list[dict[str, Any]]:
    """Build detached tool declarations shared by preflight and model dispatch."""

    if targeted_tool_projection is not None and type(targeted_tool_projection) is not (
        TargetedToolProjectionKind
    ):
        raise TypeError("targeted_tool_projection must be a TargetedToolProjectionKind or None.")

    if tool_discovery_mode is not None and type(tool_discovery_mode) is not ToolDiscoveryMode:
        raise TypeError("tool_discovery_mode must be a ToolDiscoveryMode or None.")

    tools: list[dict[str, Any]] = []
    if tool_discovery_mode is not None:
        tools.extend((search_tools_spec(), call_tool_spec()))
    tools.extend(
        [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.input_schema_copy(),
            }
            for tool in tool_exposure.tools
        ]
    )
    if (
        structured_output is not None
        and structured_output.strategy == StructuredOutputStrategy.TOOL
    ):
        tools.append(structured_output_tool_spec(structured_output))
    if targeted_tool_projection in {
        TargetedToolProjectionKind.CALL_TOOL,
        TargetedToolProjectionKind.OPENAI_ADDITIONAL_TOOLS,
    } and not any(tool.get("name") == CALL_TOOL_NAME for tool in tools):
        tools.append(call_tool_spec())
    return tools


def _redacted_native_discovery_tools_with_schema_evidence(
    messages: Iterable[Message],
    *,
    authorized_names: Iterable[str],
    catalogue: ToolCatalogSnapshot,
    redactor: SecretRedactor,
) -> tuple[dict[str, Any], ...]:
    """Keep exact current definitions whose loading result remains in context."""

    authorized = frozenset(authorized_names)
    evidenced: dict[str, dict[str, Any]] = {}
    trusted_definitions: dict[str, tuple[ToolDescriptor, dict[str, Any], dict[str, Any]]] = {}
    for message in messages:
        for part in message.content:
            if (
                type(part) is not ToolResultPart
                or part.tool_name != SEARCH_TOOLS_NAME
                or part.is_error
                or type(part.structured) is not dict
            ):
                continue
            matches = part.structured.get("matches")
            if type(matches) is not list:
                continue
            for match in matches:
                if type(match) is not dict:
                    continue
                name = match.get("name")
                if type(name) is not str or name not in authorized:
                    continue
                try:
                    trusted = trusted_definitions.get(name)
                    if trusted is None:
                        descriptor = catalogue.descriptor_for_name(name)
                        raw_definition = _tool_discovery_definition_for_descriptor(descriptor)
                        [provider_definition] = _redacted_provider_tool_definitions(
                            (raw_definition,),
                            redactor=redactor,
                            field_name="discovery_tools",
                        )
                        trusted = (descriptor, raw_definition, provider_definition)
                        trusted_definitions[name] = trusted
                    descriptor, raw_definition, provider_definition = trusted
                    if (
                        match.get("description") != provider_definition["description"]
                        or match.get("input_schema") != provider_definition["input_schema"]
                    ):
                        raise ValueError("Provider-visible discovery definition changed.")
                    validated = ToolDiscoverySearchMatch.model_validate(
                        {
                            **match,
                            "description": raw_definition["description"],
                            "input_schema": raw_definition["input_schema"],
                        }
                    )
                except (KeyError, TypeError, ValueError) as exc:
                    raise RuntimeError(
                        "Native discovery search evidence is malformed or stale."
                    ) from exc
                if not tool_discovery_search_match_matches_descriptor(validated, descriptor):
                    raise RuntimeError(
                        "Native discovery search evidence conflicts with the current catalogue."
                    )
                previous = evidenced.setdefault(name, provider_definition)
                if previous != provider_definition:
                    raise RuntimeError("Native discovery search evidence is inconsistent.")
    return tuple(evidenced[name] for name in sorted(evidenced))


def _hosted_replay_loaded_tools(
    call: Mapping[str, Any],
    output: Mapping[str, Any],
) -> tuple[dict[str, Any], ...]:
    """Decode only the canonical provider state persisted by the OpenAI adapter."""

    if set(call) not in (
        {"type", "execution", "call_id", "status", "arguments"},
        {"type", "id", "execution", "call_id", "status", "arguments"},
    ) or (
        call.get("type") != "tool_search_call"
        or call.get("execution") != "server"
        or call.get("call_id") is not None
        or call.get("status") != "completed"
        or type(call.get("arguments")) is not dict
    ):
        raise RuntimeError("Hosted Tool Search replay contains a malformed server call.")
    if set(output) not in (
        {"type", "execution", "call_id", "status", "tools"},
        {"type", "id", "execution", "call_id", "status", "tools"},
    ) or (
        output.get("type") != "tool_search_output"
        or output.get("execution") != "server"
        or output.get("call_id") is not None
        or output.get("status") != "completed"
        or type(output.get("tools")) is not list
    ):
        raise RuntimeError("Hosted Tool Search replay contains a malformed loaded output.")
    loaded: list[dict[str, Any]] = []
    for raw_tool in cast("list[Any]", output["tools"]):
        if type(raw_tool) is not dict or set(raw_tool) != {
            "type",
            "name",
            "description",
            "parameters",
            "strict",
            "defer_loading",
        }:
            raise RuntimeError("Hosted Tool Search replay contains a malformed loaded function.")
        if (
            raw_tool.get("type") != "function"
            or type(raw_tool.get("name")) is not str
            or type(raw_tool.get("description")) is not str
            or type(raw_tool.get("parameters")) is not dict
            or raw_tool.get("strict") is not False
            or raw_tool.get("defer_loading") is not True
        ):
            raise RuntimeError("Hosted Tool Search replay contains a malformed loaded function.")
        loaded.append(
            {
                "name": raw_tool["name"],
                "description": raw_tool["description"],
                "input_schema": raw_tool["parameters"],
            }
        )
    try:
        return ToolDiscoveryProjectionResult(loaded_tools=tuple(loaded)).loaded_tools
    except (TypeError, ValueError) as exc:
        raise RuntimeError("Hosted Tool Search replay is not bounded and canonical.") from exc


def _redacted_hosted_discovery_tools_with_replay_evidence(
    messages: Iterable[Message],
    *,
    authorized_names: Iterable[str],
    catalogue: ToolCatalogSnapshot,
    redactor: SecretRedactor,
) -> tuple[dict[str, Any], ...]:
    """Keep durable hosted grants whose exact loaded output remains in context."""

    authorized = frozenset(authorized_names)
    evidenced: dict[str, dict[str, Any]] = {}
    trusted_definitions: dict[str, dict[str, Any]] = {}
    for message in messages:
        if message.role is not MessageRole.ASSISTANT:
            continue
        states = [
            part.state
            for part in message.content
            if type(part) is ProviderStatePart and part.provider == "openai"
        ]
        pair_count = 0
        for index, state in enumerate(states):
            item_type = state.get("type")
            if item_type == "tool_search_output":
                if (
                    index == 0
                    or states[index - 1].get("type") != "tool_search_call"
                    or states[index - 1].get("execution") != "server"
                ):
                    raise RuntimeError("Hosted Tool Search replay contains an orphan output.")
                continue
            if item_type != "tool_search_call":
                continue
            if (
                state.get("execution") != "server"
                or index + 1 >= len(states)
                or states[index + 1].get("type") != "tool_search_output"
            ):
                raise RuntimeError("Hosted Tool Search replay contains an incomplete pair.")
            pair_count += 1
            if pair_count > 1:
                raise RuntimeError("Hosted Tool Search replay repeats a selection pair.")
            for loaded in _hosted_replay_loaded_tools(state, states[index + 1]):
                name = cast("str", loaded["name"])
                if name not in authorized:
                    continue
                trusted = trusted_definitions.get(name)
                if trusted is None:
                    descriptor = catalogue.descriptor_for_name(name)
                    [trusted] = _redacted_provider_tool_definitions(
                        (_tool_discovery_definition_for_descriptor(descriptor),),
                        redactor=redactor,
                        field_name="hosted_discovery_replay_tools",
                    )
                    trusted_definitions[name] = trusted
                if loaded != trusted:
                    raise RuntimeError(
                        "Hosted Tool Search replay conflicts with the current catalogue."
                    )
                previous = evidenced.setdefault(name, trusted)
                if previous != trusted:
                    raise RuntimeError("Hosted Tool Search replay is inconsistent.")
    return tuple(evidenced[name] for name in sorted(evidenced))


def _native_tool_grant_ids_for_request(
    request: ModelRequest,
    *,
    available: Mapping[str, str],
) -> dict[str, str]:
    """Select only native grant identities actually projected in one request."""

    projected_names = {
        tool["name"]
        for tool in (
            ()
            if request.targeted_tool_projection is None
            else request.targeted_tool_projection.tools
        )
    }
    if request.tool_discovery_projection is not None:
        projected_names.update(request.tool_discovery_projection.loaded_tool_names)
    missing = sorted(projected_names.difference(available))
    if missing:
        raise RuntimeError(
            "A native tool projection has no matching durable grant: " + ", ".join(missing)
        )
    return {name: available[name] for name in sorted(projected_names)}


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
    """Return the registration-time expose-all snapshot."""

    return _require_frozen_tool_exposure(
        registered_agent.all_registered_tool_exposure,
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
        catalogue_revision=registered_agent.tool_catalogue.revision,
        tools=selected,
        registered_count=len(capabilities),
        ceiling_count=len(selected),
    )


def _model_request_messages(
    *,
    messages: list[Message],
    structured_output: StructuredOutputSpec | None,
    targeted_tool_gateway: TargetedToolGatewayProjection | None = None,
) -> list[Message]:
    """Return the complete statically determined message surface for one request."""

    projected = messages
    if (
        structured_output is not None
        and structured_output.strategy == StructuredOutputStrategy.TOOL
    ):
        projected = _with_structured_output_tool_instruction(projected, structured_output)
    if targeted_tool_gateway is None:
        return projected
    return _with_targeted_tool_gateway_instruction(
        projected,
        targeted_tool_gateway=targeted_tool_gateway,
    )


def _with_targeted_tool_gateway_instruction(
    messages: list[Message],
    *,
    targeted_tool_gateway: TargetedToolGatewayProjection,
    redactor: SecretRedactor | None = None,
) -> list[Message]:
    """Append one detached runtime-authored gateway descriptor message."""

    if type(targeted_tool_gateway) is not TargetedToolGatewayProjection:
        raise TypeError("targeted_tool_gateway must be a TargetedToolGatewayProjection.")
    copied = [detach_message(message) for message in messages]
    copied.append(
        targeted_tool_gateway.context_message(redactor=redactor),
    )
    return copied


def _context_pressure_overhead(
    *,
    registered_provider: runtime_records.RegisteredProvider,
    registered_agent: runtime_records.RegisteredAgentState,
    registered_environment: runtime_records.RegisteredEnvironment | None,
    structured_output: StructuredOutputSpec | None,
    thinking: ThinkingConfig | None,
    step: int,
    tool_exposure: ResolvedToolExposure,
    targeted_tool_projection_kind: TargetedToolProjectionKind | None = None,
    targeted_tool_gateway: TargetedToolGatewayProjection | None = None,
    targeted_tool_native: TargetedToolProjectionRequest | None = None,
    tool_discovery_projection_kind: ToolDiscoveryProjectionKind | None = None,
    tool_discovery_native_tool_names: Iterable[str] = (),
) -> ContextPressureOverhead:
    profile = copy_model_context_pressure_profile(
        registered_provider.provider.context_pressure_profile
    )
    tools = _model_request_tools(
        tool_exposure=tool_exposure,
        structured_output=structured_output,
        targeted_tool_projection=targeted_tool_projection_kind,
        tool_discovery_mode=registered_agent.tool_discovery_mode,
    )
    if targeted_tool_native is not None:
        tools.extend(copy_json_value(list(targeted_tool_native.tools), "targeted tools"))
    if tool_discovery_projection_kind is ToolDiscoveryProjectionKind.OPENAI_TOOL_SEARCH_CLIENT:
        discovery_names = tuple(sorted(tool_discovery_native_tool_names))
        if len(discovery_names) != len(set(discovery_names)):
            raise ValueError("Native discovery tool names must be unique.")
        if (
            targeted_tool_projection_kind is not TargetedToolProjectionKind.OPENAI_ADDITIONAL_TOOLS
            and targeted_tool_gateway is None
        ):
            tools = [tool for tool in tools if tool.get("name") != CALL_TOOL_NAME]
        existing_names = {tool.get("name") for tool in tools}
        for name in discovery_names:
            if name in existing_names:
                raise RuntimeError(
                    "A native discovery tool cannot also be directly or dynamically exposed: "
                    f"{name}"
                )
            descriptor = registered_agent.tool_catalogue.descriptor_for_name(name)
            tools.append(
                {
                    "name": descriptor.name,
                    "description": descriptor.description,
                    "input_schema": descriptor.input_schema_copy(),
                }
            )
            existing_names.add(name)
    structured_output_instruction: str | None = None
    if (
        structured_output is not None
        and structured_output.strategy == StructuredOutputStrategy.TOOL
    ):
        structured_output_instruction = structured_output_tool_instruction(structured_output)
    if targeted_tool_gateway is not None:
        gateway_instruction = targeted_tool_gateway.instruction_text()
        structured_output_instruction = (
            gateway_instruction
            if structured_output_instruction is None
            else structured_output_instruction + "\n" + gateway_instruction
        )

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


@timed_phase("context_policy")
async def _build_context(
    *,
    context_policy: ContextPolicy,
    session_store: SessionStore,
    session: Session,
    agent_spec: AgentSpec,
    messages: list[Message],
    step: int,
    interaction_id: str | None,
    model_step_id: str,
    evidence_key: MemoryEvidenceKey | None,
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
    MemoryEvidenceReference | None,
]:
    from cayu.resource_access import current_binding

    if (
        current_binding() is not None
        and knowledge_store is not None
        and type(knowledge_store).__dict__.get("resource_knowledge_access_version") != 1
    ):
        raise NotImplementedError("Knowledge store cannot enforce scoped context retrieval.")
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
        interaction_id=interaction_id,
        model_step_id=model_step_id,
        environment_name=environment_name,
        session_store=session_store,
        knowledge_store=knowledge_store,
        knowledge_access_scope=knowledge_access_scope,
        metadata=copy_durable_metadata(request_metadata, "metadata"),
        context_usage=context_usage,
        pressure_overhead=pressure_overhead,
        count_input_tokens=count_input_tokens,
        build_cache_prefix_request=build_cache_prefix_request,
        force_bounded_compaction=force_bounded_compaction,
    )
    if isinstance(context_policy, RuntimeManagedContextPolicy):
        checkpoint = project_runtime_managed_context_checkpoint(
            await session_store.load_checkpoint(session.id)
        )
        try:
            with (
                _context_secret_redactor_scope(secret_redactor),
                _context_recall_telemetry_publisher_scope(publish_recall_telemetry),
                _defer_billing_identity_cancellation_scope(),
                _automatic_compaction_runner_scope(run_compaction),
                memory_evidence_key_scope(evidence_key),
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
            memory_evidence_reference_from_checkpoint(
                safe_checkpoint if safe_checkpoint is not None else checkpoint
            ),
        )

    with _context_secret_redactor_scope(secret_redactor):
        result = await context_policy.build(request)
    return copy_context_messages(result), None, None, [], [], None


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
    input_coverage = None
    if "input_coverage" in event.payload:
        try:
            input_coverage = ContextInputCoverage.model_validate(event.payload["input_coverage"])
        except ValueError:
            # Imported or historical evidence can still report actual usage,
            # but malformed coverage must never become an estimator anchor.
            input_coverage = None
    metrics = usage_metrics_from_event_payload(event.payload)
    if metrics is None:
        return ContextUsageState(
            last_transcript_cursor=_transcript_cursor_from_model_completed_event(event),
            input_coverage=input_coverage,
        )
    return ContextUsageState(
        last_input_tokens=metrics.input_tokens,
        last_output_tokens=metrics.output_tokens,
        last_total_tokens=metrics.total_tokens,
        last_transcript_cursor=_transcript_cursor_from_model_completed_event(event),
        input_coverage=input_coverage,
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


class _FileAttachmentUnavailable(RuntimeError):
    """An attachment reference cannot be resolved in its declared scope."""


def _model_file_attachment_attestations(
    request: ModelRequest,
) -> str | None:
    """Hash the exact runtime-resolved bytes without materializing them twice."""

    raw_attachments = request.options.get(RESOLVED_FILE_ATTACHMENTS_OPTION)
    if raw_attachments is None:
        return None
    if type(raw_attachments) is not dict:
        raise RuntimeError("Resolved file attachment request material must be an object.")
    attestations: list[dict[str, str]] = []
    for artifact_id in sorted(raw_attachments):
        attachment = raw_attachments[artifact_id]
        if type(artifact_id) is not str or not artifact_id or type(attachment) is not dict:
            raise RuntimeError("Resolved file attachment request material is malformed.")
        if attachment.get("artifact_id") != artifact_id:
            raise RuntimeError("Resolved file attachment request identity is inconsistent.")
        digest = attachment.get("content_sha256")
        if (
            type(digest) is not str
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise RuntimeError("Resolved file attachment request digest is unavailable.")
        attestations.append(
            {
                "artifact_id": artifact_id,
                "content_sha256": digest,
            }
        )
    return "v1:" + canonical_durable_json_bytes(
        attestations,
        "model file attachment attestations",
    ).decode("utf-8")


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

    from cayu.resource_access import current_binding

    if (
        current_binding() is not None
        and type(artifact_store).__dict__.get("artifact_access_version") != 1
    ):
        raise NotImplementedError("Artifact store cannot enforce attachment access.")

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
            visual_policy = artifact.metadata.get("visual_publication")
            expected_digest = attachment.metadata.get("browser_visual_screenshot_sha256")
            if visual_policy is not None or expected_digest is not None:
                from cayu.tools.browser_visual import BrowserVisualPolicy

                try:
                    policy = BrowserVisualPolicy.model_validate(visual_policy)
                except (TypeError, ValueError):
                    raise _FileAttachmentUnavailable(
                        "Visual screenshot publication authority is unavailable."
                    ) from None
                if (
                    not policy.publish_to_model
                    or artifact_store.id != policy.artifact_store_id
                    or artifact.size_bytes > policy.max_bytes
                    or type(expected_digest) is not str
                    or expected_digest != artifact.metadata.get("content_sha256")
                    or sha256(result.content).hexdigest() != expected_digest
                ):
                    raise _FileAttachmentUnavailable(
                        "Visual screenshot publication authority changed."
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
    # Native image/PDF derivations are cached by source bytes and read options,
    # so distinct source snapshots may legitimately name the same derived file.
    # Keep per-read provenance in ordered_refs and the durable transcript; it
    # is not authority to resolve the source or part of the derived identity.
    # All other metadata stays strict, including page selections, content
    # digests, browser publication constraints, and unknown extension fields.
    from cayu.artifacts.attachments import same_file_attachment_reference

    return same_file_attachment_reference(left, right)


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
        targeted_tool_projection=request.targeted_tool_projection,
        tool_discovery_projection=request.tool_discovery_projection,
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
    model: str,
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
                "model": model,
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
    model: str,
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
                "model": model,
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


def reconstruct_assistant_step_result(
    *,
    stage: ModelCompletionStage,
    pointer: model_completion_publication.ModelStepPublicationCheckpoint,
    pending_round: pending_rounds.PendingToolRound | None,
    session_id: str,
    interaction_id: str,
    source_run_epoch: int,
) -> AssistantStepResult | None:
    """Decode a reconciled publication, without granting execution authority.

    The recovery owner must first validate the stage, receipt and transcript.
    Expected identities come from the admitted invocation, not the publication.
    A durable non-turn returns None, even if its earlier completion metadata
    describes a normal stop. The caller must retain that non-success outcome.
    """

    stage = _copy_model_completion_stage(stage)
    publication = stage.publication
    if (
        type(pointer) is not model_completion_publication.ModelStepPublicationCheckpoint
        or type(source_run_epoch) is not int
        or stage.session_id != session_id
        or stage.source_run_epoch != source_run_epoch
        or stage.state != "completed"
        or stage.purpose != "assistant-turn"
        or publication is None
        or publication.kind != "model-step"
        or publication.interaction_id != interaction_id
        or publication.publication_id != stage.logical_step_id
        or stage.intent.get("model_step_id") != stage.logical_step_id
        or pointer.stage_id != stage.stage_id
        or pointer.logical_step_id != stage.logical_step_id
        or pointer.source_transcript_cursor != stage.source_transcript_cursor
        or len(publication.events) != 1
    ):
        raise ValueError("Recovered model result conflicts with its expected publication.")
    event = publication.events[0]
    if (
        event.type is not EventType.MODEL_COMPLETED
        or event.id != pointer.completion_event_id
        or event.session_id != session_id
        or event.interaction_id != interaction_id
        or event.payload.get("step_classification") != pointer.classification
        or event.payload.get("transcript_cursor") != pointer.transcript_end_cursor
        or event.payload.get("model_step_id") != stage.logical_step_id
        or event.payload.get("model_attempt_id") != stage.intent.get("model_attempt_id")
        or event.payload.get("tool_round_id") != pointer.tool_round_id
    ):
        raise ValueError("Recovered model result conflicts with its completion event.")
    classification = pointer.classification.get("type")
    if classification in {"failed", "filtered", "invalid", "length"}:
        return None
    if classification not in {"continue", "final", "think_only"}:
        raise ValueError("Recovered model result has an unsupported classification.")
    step = event.payload.get("step")
    if type(step) is not int or step < 1:
        raise ValueError("Recovered model result requires its original positive step.")
    metadata = event.payload.get("completion")
    if type(metadata) is not dict:
        raise ValueError("Recovered model result requires explicit completion metadata.")
    # Do not normalize a provider payload or supply a default finish reason.
    # A portable fallback which omitted invalid metadata is not replay evidence.
    try:
        completion = ModelCompletion.model_validate(metadata)
    except (TypeError, ValueError):
        raise ValueError("Recovered model result has invalid completion metadata.") from None
    identity = ModelAttemptIdentity(
        model_step_id=event.payload["model_step_id"],
        model_attempt_id=event.payload["model_attempt_id"],
    )
    if len(publication.transcript_messages) != int(pointer.assistant_message_published):
        raise ValueError("Recovered model result conflicts with its message disposition.")
    assistant_message = (
        publication.transcript_messages[0] if pointer.assistant_message_published else None
    )
    tool_calls: list[runtime_records.ToolCallRequest] = []
    tool_identity = None
    if pointer.tool_round_id is not None:
        if type(pending_round) is not pending_rounds.PendingToolRound:
            raise ValueError("Recovered model tool calls require their original pending round.")
        tool_identity = pending_rounds.pending_tool_round_identity(pending_round)
        if (
            tool_identity.model_step_id != identity.model_step_id
            or tool_identity.model_attempt_id != identity.model_attempt_id
            or tool_identity.tool_round_id != pointer.tool_round_id
            or pending_round.source_model_step_id != stage.logical_step_id
            or pending_round.source_transcript_cursor != pointer.source_transcript_cursor
            or pending_round.model_step != step
            or (pending_round.assistant_message_state == "quarantined")
            != pointer.assistant_message_deferred
        ):
            raise ValueError("Recovered model result conflicts with its pending tool round.")
        tool_calls = tool_round_recovery.pending_round_tool_calls(pending_round)
        if pointer.assistant_message_deferred:
            assistant_message = pending_round.quarantined_assistant_message
    elif pending_round is not None or pointer.assistant_message_deferred:
        raise ValueError("Recovered model result has an unexpected pending tool round.")
    if assistant_message is not None:
        assistant_message = detach_message(assistant_message)
    text_content = assistant_text_content(assistant_message)
    result = AssistantStepResult(
        session_id=session_id,
        step=step,
        model_step_id=identity.model_step_id,
        model_attempt_id=identity.model_attempt_id,
        tool_round_identity=tool_identity,
        assistant_message=assistant_message,
        tool_calls=tool_calls,
        completion=completion,
        text_content=text_content,
        has_user_visible_content=bool(text_content.strip()),
        provider_state_count=provider_state_count(assistant_message),
        thinking_count=thinking_count(assistant_message),
    )
    if classify_assistant_step(result).type.value != classification:
        raise ValueError("Recovered model result conflicts with its durable classification.")
    return result


def _attempt_retry_suppression(exc: ModelAttemptFailed) -> RetrySuppression | None:
    if exc.completion_observed:
        return RetrySuppression.COMPLETION_OBSERVED
    if isinstance(exc.cause, ModelStreamDeadlineError):
        return RetrySuppression.DEADLINE
    if exc.automatic_retry_disabled and exc.retry_suppression is not None:
        return exc.retry_suppression
    if exc.automatic_retry_disabled:
        # This flag is also used for unresolved effects and publication failures.
        # Do not invent a more specific operation authority when it is unavailable.
        return RetrySuppression.AUTOMATIC_RETRY_DISABLED
    return None


def _typed_retry_fields(
    exc: ModelAttemptFailed,
) -> tuple[int | None, bool | None, float | None, bool]:
    cause = exc.cause
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
    model: str,
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
        model=model,
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
    model: str,
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
                "model": model,
                "step": step,
                "attempt": decision.attempt,
                "next_attempt": decision.next_attempt,
                "max_attempts": decision.max_attempts,
                **retry_diagnostic_payload(decision),
                "effective_max_attempts": decision.effective_max_attempts,
                "reason": None if decision.reason is None else decision.reason.value,
                "status_code": decision.status_code,
                **copy_model_attempt_identity(model_attempt_identity).payload(),
            },
        ),
        model_attempt_identity,
    )

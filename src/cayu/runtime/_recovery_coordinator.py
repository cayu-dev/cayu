"""Durable runtime continuation and crash-recovery ownership.

The coordinator owns paused-round continuation, recorded tool-round repair,
incomplete-session recovery, subagent reattachment, and abandoned-stream
finalization without importing or accepting :class:`CayuApp`. Public request
validation and registry ownership remain on the application façade; the
coordinator resolves registrations through narrow callbacks. Session execution,
interruption, turn accounting, and terminal hook orchestration are likewise
supplied through narrow callbacks by the composition root.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import contextlib
import json
import logging
import time
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from typing import Any, Literal, Protocol, cast
from uuid import UUID, uuid4, uuid5

from cayu._exception_groups import (
    add_exception_note_safely,
    exception_cause,
    exception_context,
    exception_group_children,
    exception_suppresses_context,
    iter_exception_tree,
    set_exception_cause,
)
from cayu._task_wait import (
    CapturedAwaitableOutcome,
    await_shielded_task_outcome,
    capture_awaitable_outcome,
    unexpected_child_cancellation_error,
)
from cayu._validation import (
    canonical_durable_json_bytes,
    copy_durable_json_value,
    copy_json_value,
    require_clean_nonblank,
)
from cayu.artifacts import ArtifactReadResult, copy_artifact_read_result
from cayu.core.events import (
    Event,
    EventType,
    copy_event,
    event_with_runtime_envelope_authority,
    event_with_runtime_generated_id,
    event_with_runtime_payload_authority,
)
from cayu.core.messages import Message, MessageRole, ToolCallPart, ToolResultPart, detach_message
from cayu.core.thinking import ThinkingConfig
from cayu.core.tools import _TOOL_POLICY_DENIAL_SOURCE, ToolResult
from cayu.environments import EnvironmentFactoryOperation
from cayu.environments.bindings import _runtime_owned_workspace_observer_name
from cayu.providers import (
    ProviderOperationAdapter,
    ProviderOperationMode,
    ProviderOperationSnapshot,
    ProviderOperationStatus,
)
from cayu.runtime import _approval_publication as approval_publication
from cayu.runtime import _approval_support as approval_support
from cayu.runtime import _invocation_secrets as invocation_secrets
from cayu.runtime import _model_completion_publication as model_completion_publication
from cayu.runtime import _resume_ledger as resume_ledger
from cayu.runtime import _runtime_records as runtime_records
from cayu.runtime import _structured_output_tool_round as structured_output_tool_round
from cayu.runtime import _tool_argument_publication as tool_argument_publication
from cayu.runtime import _tool_execution as tool_execution
from cayu.runtime import _tool_results as tool_results
from cayu.runtime import _tool_round_publication as tool_round_publication
from cayu.runtime import _tool_round_recovery as tool_round_recovery
from cayu.runtime import _transcript as transcript_helpers
from cayu.runtime import _web_access_results as web_access_results
from cayu.runtime import pending_actions
from cayu.runtime._child_session_identity import (
    ChildSessionKind,
    child_session_id_prefix,
    generate_child_session_id,
)
from cayu.runtime._diagnostics import (
    bound_diagnostic_text,
    exception_diagnostic,
    task_failure_payload_from_diagnostic,
    task_update_error_payload,
)
from cayu.runtime._durable_subagents import (
    durable_subagent_submission_from_checkpoint,
    durable_subagent_submission_receipt_from_checkpoint,
    durable_subagent_submission_seed_from_checkpoint,
    durable_subagent_submissions_from_checkpoint,
    require_durable_subagent_intent_matches_seed,
    require_durable_subagent_receipt_matches_intent,
    require_durable_subagent_receipt_matches_seed,
)
from cayu.runtime._environment_lifecycle import (
    EnvironmentLifecycle,
    exception_failure_payload,
)
from cayu.runtime._event_writer import (
    RuntimeEventWriter,
    _reconcile_exact_persisted_event,
    prepare_runtime_event,
)
from cayu.runtime._interruption_coordinator import (
    _PENDING_INTERRUPTION_CASCADE_CHECKPOINT_KEY,
    _PENDING_SESSION_INTERRUPT_CHECKPOINT_KEY,
)
from cayu.runtime._message_redaction import redact_runtime_message_for_boundary
from cayu.runtime._model_errors import (
    _FallbackBillingCancellationStateCheckFailed,
    detach_billing_identity_cancellation_group,
)
from cayu.runtime._model_step_executor import (
    ModelCompletionRecoveryContext,
    model_completion_recovery_context_from_stage,
)
from cayu.runtime._provider_operation_cancellation_claim import (
    active_provider_operation_cancellation_claim_from_checkpoint,
)
from cayu.runtime._run_limit_accounting import (
    RunLimitAccountingContext,
    rebase_run_limit_accounting_context,
    restore_run_limit_accounting_context,
)
from cayu.runtime._run_limits import (
    RunLimitController,
    SessionUsageTracker,
)
from cayu.runtime._session_control import (
    ActiveSessionRun,
    SessionControl,
)
from cayu.runtime._session_queries import query_all_sessions
from cayu.runtime._terminal_evidence import (
    TERMINAL_EVIDENCE_EVENT_TYPES,
    TERMINAL_EVIDENCE_QUERY_LIMIT,
    classify_current_terminal_evidence,
    interruption_request_id_from_payload,
)
from cayu.runtime._tool_round_executor import (
    DeferredTerminalStager,
    InterruptedToolRoundRequest,
    ToolApprovalRequired,
    ToolRoundExecutor,
    _interrupted_tool_call_event,
    _interrupted_tool_call_outcome,
    _staged_terminal_argument_projections,
    _ToolRoundPublicationCoordinator,
    _workspace_mutation_incomplete_event,
    policy_denial_payload_fields,
    restore_staged_terminal_authority,
)
from cayu.runtime.approvals import (
    PendingToolApproval,
    PendingToolCallApproval,
    ToolApprovalDecision,
    ToolApprovalRecoveryOutcome,
    ToolApprovalRecoveryRequest,
    ToolApprovalRequest,
    ToolPolicyEvidence,
    expiry_resolution_actor,
    resolution_actor_payload,
)
from cayu.runtime.budgets import (
    BudgetLimit,
    BudgetPolicy,
    copy_budget_policy,
    copy_request_budget_limits,
    request_budget_limits_for_session,
)
from cayu.runtime.costs import SessionCostSummary
from cayu.runtime.dispatch import (
    _new_prepared_subagent_dispatch_envelope,
    _require_dispatch_task_authority,
    _task_matches_queued_dispatch,
)
from cayu.runtime.errors import InteractionLifecyclePublicationRejected
from cayu.runtime.execution_profiles import (
    EXECUTION_PROFILE_METADATA_KEY,
    ActiveInvocationExecutionProfile,
    ExecutionProfileIdentity,
    active_invocation_execution_profile_from_checkpoint,
    active_invocation_execution_profile_is_released,
    active_invocation_execution_profile_matches_session_epoch,
    checkpoint_with_active_invocation_execution_profile,
    event_with_execution_profile_authority,
    event_with_execution_profile_fingerprint_authority,
    execution_profile_from_session_metadata,
)
from cayu.runtime.execution_units import (
    ModelAttemptIdentity,
    ModelStepIdentity,
    ToolRoundIdentity,
    copy_tool_round_identity,
)
from cayu.runtime.hooks import RuntimeHookPhase
from cayu.runtime.interactions import (
    INTERACTION_LIFECYCLE_EVENT_TYPES,
    INTERACTION_TERMINAL_EVENT_TYPES,
    InteractionStatus,
    InteractionSummaryEvidence,
)
from cayu.runtime.invocation import SessionExecutionSource, inherited_session_invocation
from cayu.runtime.loop_policies import LoopPolicy
from cayu.runtime.provider_operations import (
    ProviderOperationEvidenceError,
    ProviderOperationPendingDisposition,
    ProviderOperationRecoveryResult,
    ProviderOperationRecoveryStatus,
    ProviderOperationResolutionAction,
    ProviderOperationResolutionRequest,
    ProviderOperationResolutionResult,
    ProviderOperationUnavailableReason,
    RecoverableProviderOperation,
    RecoverableProviderOperationStart,
    clear_pending_provider_operation_disposition,
    load_pending_provider_operation_disposition,
    load_recoverable_provider_operation,
    load_recoverable_provider_operation_start,
    prepare_provider_operation_resolution_request,
    provider_operation_duplicate_request_risk,
    provider_operation_resolution_outcome_event_id,
    resolve_provider_operation_stage,
    validate_provider_operation_resolution_outcome_event,
)
from cayu.runtime.retry_policy import RetryPolicy
from cayu.runtime.sessions import (
    _INCOMPLETE_RECOVERY_CLAIM_CHECKPOINT_KEY,
    _SESSION_RUN_OPERATION_CHECKPOINT_KEY,
    MAX_INCOMPLETE_SESSIONS_RECOVERY_CURSOR_BYTES,
    MAX_SESSION_LIST_CURSOR_BYTES,
    RUNTIME_PUBLICATION_MAX_EVENT_BINDINGS,
    ActiveModelCompletionStage,
    CheckpointTransform,
    EventOrder,
    EventQuery,
    EventRecord,
    IncompleteSessionRecoveryAction,
    IncompleteSessionRecoveryRequest,
    IncompleteSessionRecoveryResult,
    IncompleteSessionsRecoveryPage,
    IncompleteSessionsRecoveryRequest,
    InteractionTransitionReceiptResult,
    InteractionTransitionSpec,
    ModelCompletionStage,
    RuntimePublicationReceipt,
    Session,
    SessionOrder,
    SessionQuery,
    SessionRunFenced,
    SessionRuntimePublicationConflict,
    SessionStatus,
    SessionStatusConflict,
    SessionStore,
    _activate_session_interaction,
    _activate_session_run_fence,
    _checkpoint_after_session_run_operation_cleanup,
    _checkpoint_with_session_run_operation,
    _current_session_run_epoch,
    _deactivate_session_interaction,
    _deactivate_session_run_fence,
    _event_with_session_run_operation,
    _incomplete_recovery_claim_from_checkpoint,
    _queued_dispatch_session_instance_fingerprint,
    _session_run_operation_from_checkpoint,
    _SessionRunOperation,
    copy_interaction_transition_spec,
    runtime_publication_checkpoint_value_digest,
)
from cayu.runtime.stop_policy import RunLimits, StopDecision, copy_run_limits, has_run_limits
from cayu.runtime.structured_output import (
    STRUCTURED_OUTPUT_TOOL_NAME,
    StructuredOutputSpec,
    copy_structured_output_spec,
    require_secret_free_structured_output_spec,
)
from cayu.runtime.structured_output import (
    _require_native_structured_output_support as _require_provider_native_output_support,
)
from cayu.runtime.tasks import Task, TaskStatus, TaskStore
from cayu.runtime.tool_exposure import (
    ALL_REGISTERED_TOOLS_PROFILE_ID,
    NOT_EXPOSED_IN_REQUEST_REASON,
    ResolvedToolExposureAuthority,
    tool_capability_ceiling_from_session_metadata,
    unexposed_tool_result,
    validate_resolved_tool_exposure_authority,
)
from cayu.runtime.tool_policy import ToolPolicyDecision
from cayu.runtime.tool_rounds import ToolRoundRecoveryRequest
from cayu.runtime.usage import SessionUsageSummary, session_usage_summary
from cayu.runtime.user_input import (
    PENDING_USER_INPUT_CHECKPOINT_KEY,
    PendingUserInput,
    UserInputRecoveryRequest,
    UserInputResponse,
    pending_user_input_from_checkpoint,
    pending_user_input_interruption_payload,
)
from cayu.runtime.workspace_observation_recovery import (
    WorkspaceObservationArtifactState,
    WorkspaceObservationEvidenceState,
    WorkspaceObservationLifecycle,
    WorkspaceObservationPhase,
    WorkspaceObservationTerminalStatus,
    await_workspace_observation_store_mutation,
    await_workspace_observation_store_read,
    publish_workspace_observation_transition,
    raise_workspace_observation_concurrent_control,
    restore_workspace_observation_cancellation_requests,
    workspace_observation_artifact_metadata_matches,
    workspace_observation_authority_matches,
    workspace_observation_event_digest,
    workspace_observation_observer_authority_matches,
    workspace_observation_terminal_from_delta_status,
    workspace_observations_from_checkpoint,
)
from cayu.tools._operation_boundary import BoundedInvocationOperationRegistry
from cayu.tools._redaction import InvocationRedactorSnapshot
from cayu.vaults import SecretRedactor

_INTERRUPTION_TYPE_TOOL_APPROVAL_REQUIRED = "tool_approval_required"
_INTERRUPTION_TYPE_USER_INPUT_REQUIRED = "user_input_required"
_INTERRUPTION_TYPE_RUNTIME_INTERRUPTED = "runtime_interrupted"
_INTERRUPTION_TYPE_OPERATOR_REQUESTED = "operator_requested"
_DEFAULT_APPROVAL_MAX_STEPS = 16
_ABANDONED_RUN_REASON = "event_stream_closed"
_INCOMPLETE_RECOVERY_CLAIM_LEASE = timedelta(minutes=5)
_INCOMPLETE_RECOVERY_CLAIM_HEARTBEAT_INTERVAL_SECONDS = 30.0
_INCOMPLETE_RECOVERY_CLAIM_HEARTBEAT_RETRY_SECONDS = 5.0
_MANUAL_RECOVERY_INTERRUPT_POLL_INTERVAL_SECONDS = 0.25
_TERMINAL_EVIDENCE_REPAIR_NAMESPACE = UUID("bd021bef-ec8f-4e1e-950d-734e2c9ac513")
_INCOMPLETE_RECOVERY_CURSOR_VERSION = 1
# Opaque store cursors require consuming each fetched page completely. When
# only one result slot remains, that can force one-candidate pages; cap the
# database round trips and continue through Cayu's outer cursor instead.
_INCOMPLETE_RECOVERY_MAX_STORE_PAGES = 10
_INCOMPLETE_RECOVERY_STATUS_ORDER = (
    # Process the target status before states recovery can move into it, so a
    # continuation page cannot rediscover a session this sweep interrupted.
    SessionStatus.INTERRUPTED,
    SessionStatus.INTERRUPTING,
    SessionStatus.RUNNING,
    SessionStatus.PENDING,
    SessionStatus.FAILED,
    SessionStatus.COMPLETED,
)
_RECOVERY_RESUMABLE_SESSION_STATUSES = {
    SessionStatus.COMPLETED,
    SessionStatus.FAILED,
    SessionStatus.INTERRUPTED,
}
_TERMINAL_EVENT_TYPE_BY_STATUS = {
    SessionStatus.COMPLETED: EventType.SESSION_COMPLETED,
    SessionStatus.FAILED: EventType.SESSION_FAILED,
    SessionStatus.INTERRUPTED: EventType.SESSION_INTERRUPTED,
}
_TOOL_ROUND_RECOVERABLE_SESSION_STATUSES = {
    SessionStatus.RUNNING,
    SessionStatus.INTERRUPTING,
    SessionStatus.INTERRUPTED,
    SessionStatus.FAILED,
}
_MODEL_BOUNDARY_TOOL_TERMINAL_EVENT_TYPES = frozenset(
    {
        EventType.TOOL_CALL_COMPLETED,
        EventType.TOOL_CALL_FAILED,
        EventType.TOOL_CALL_BLOCKED,
        EventType.TOOL_CALL_APPROVAL_DENIED,
    }
)

_MANUAL_RECOVERY_SECRET_SCOPE_UNAVAILABLE = (
    "Externally verified tool output is unavailable because the invocation "
    "secret scope could not be reconstructed."
)


def _public_manual_recovery_result(
    result: ToolResult,
    *,
    secret_resolution_scope: invocation_secrets.SecretResolutionScope,
) -> ToolResult:
    """Fail closed when recovery cannot positively prove a static secret scope."""

    if secret_resolution_scope == "static":
        return result.model_copy(deep=True)
    return ToolResult(
        content=_MANUAL_RECOVERY_SECRET_SCOPE_UNAVAILABLE,
        structured={
            "error": "invalid_tool_output",
            "manual_recovery": True,
            "outcome_unknown": False,
            "reason": "invocation_secret_scope_unavailable",
        },
        is_error=result.is_error,
    )


def _public_resolution_audit_fields(
    *,
    secret_resolution_scope: invocation_secrets.SecretResolutionScope,
    reason: str | None,
    metadata: dict[str, Any],
    redactor: SecretRedactor,
) -> dict[str, Any]:
    """Project operator audit text only with positive static-scope evidence."""

    if secret_resolution_scope != "static":
        return {"reason": None, "metadata": {}}
    return {
        "reason": reason,
        **approval_support.bounded_resolution_metadata_payload(
            metadata,
            redactor=redactor,
        ),
    }


def _receiptless_pause_event_identity(
    event: Event,
) -> tuple[Literal["approval", "user-input"], str]:
    """Return one exact pause identity without accepting ordinary tool evidence."""

    has_approval_id = "approval_id" in event.payload
    has_input_id = "input_id" in event.payload
    if has_approval_id == has_input_id:
        raise RuntimeError(
            "Receipt-less tool evidence must carry exactly one approval_id or input_id."
        )
    field_name = "approval_id" if has_approval_id else "input_id"
    try:
        pause_id = require_clean_nonblank(event.payload[field_name], field_name)
    except (KeyError, TypeError, ValueError):
        raise RuntimeError("Receipt-less tool evidence has a malformed pause identity.") from None
    return ("approval" if has_approval_id else "user-input"), pause_id


def _receiptless_exact_execution_evidence(
    records: list[EventRecord],
    *,
    identity: ToolRoundIdentity,
    expected_call_ids: set[str],
) -> list[tuple[int, Event]]:
    """Select exact round evidence while rejecting partial or conflicting identity."""

    evidence: list[tuple[int, Event]] = []
    for record in records:
        event = record.event
        tool_call_id = event.payload.get("tool_call_id")
        event_matches_call = type(tool_call_id) is str and tool_call_id in expected_call_ids
        event_matches_round = event.payload.get("tool_round_id") == identity.tool_round_id
        event_matches_attempt = (
            event.payload.get("model_step_id") == identity.model_step_id
            and event.payload.get("model_attempt_id") == identity.model_attempt_id
        )
        if not (event_matches_call or event_matches_round or event_matches_attempt):
            continue
        try:
            event_identity = ToolRoundIdentity.model_validate(
                {
                    "model_step_id": event.payload.get("model_step_id"),
                    "model_attempt_id": event.payload.get("model_attempt_id"),
                    "tool_round_id": event.payload.get("tool_round_id"),
                }
            )
        except (TypeError, ValueError):
            if any(
                event.payload.get(field_name) == expected
                for field_name, expected in (
                    ("model_step_id", identity.model_step_id),
                    ("model_attempt_id", identity.model_attempt_id),
                    ("tool_round_id", identity.tool_round_id),
                )
            ):
                raise RuntimeError(
                    "Durable tool lifecycle evidence has a partial execution identity."
                ) from None
            # Provider call ids can be reused. An unscoped historical event
            # is not evidence for this round and cannot contradict exact
            # round-owned terminal material.
            continue
        if event_identity != identity:
            if (
                event_matches_call
                and event_identity.tool_round_id != identity.tool_round_id
                and (
                    event_identity.model_step_id,
                    event_identity.model_attempt_id,
                )
                != (
                    identity.model_step_id,
                    identity.model_attempt_id,
                )
            ):
                continue
            raise RuntimeError(
                "Durable tool lifecycle evidence conflicts with its source model step."
            )
        evidence.append((record.sequence, event))
    return evidence


def _optional_exception_type_name(
    error: BaseException | None,
    *,
    redactor: SecretRedactor,
) -> str:
    return (
        "Exception" if error is None else exception_diagnostic(error, redactor=redactor).error_type
    )


def _environment_factory_resolution_error_payload(
    error: BaseException,
    *,
    redactor: SecretRedactor,
) -> dict[str, Any]:
    """Project one factory reconnect failure for durable recovery records."""

    return exception_diagnostic(
        error,
        empty_message="environment factory resolution failed",
        nonportable_message=(
            "Environment factory resolution failed with a non-portable diagnostic."
        ),
        redactor=redactor,
    ).payload_fields()


logger = logging.getLogger(__name__)

CheckpointTransformFactory = Callable[[dict[str, Any]], CheckpointTransform]
EffectiveRetryPolicy = Callable[[RetryPolicy | None], RetryPolicy]
RecoveryCleanup = Callable[[], Awaitable[None]]


def _pending_approval_and_round_for_atomic_claim(
    checkpoint: dict[str, Any] | None,
    *,
    approval_id: str,
    tool_round_id: str,
    gating_tool_call_id: str | None = None,
    recovery_tool_call_id: str | None = None,
    redactor: SecretRedactor,
) -> tuple[PendingToolApproval, tool_round_recovery.PendingToolRound]:
    if (gating_tool_call_id is None) == (recovery_tool_call_id is None):
        raise TypeError("Exactly one approval gating or recovery tool-call identity is required.")
    approval = approval_support.pending_approval_from_checkpoint(
        checkpoint,
        redactor=redactor,
    )
    if approval is None:
        raise RuntimeError("Session has no pending tool approval.")
    if approval.approval_id != approval_id or approval.tool_round_id != tool_round_id:
        raise ValueError("Tool approval identity does not match the current pending approval.")
    pending_round = tool_round_recovery.pending_tool_round_from_checkpoint(
        checkpoint,
        redactor=redactor,
    )
    reconstructed_approval_only_round = pending_round is None
    if reconstructed_approval_only_round:
        # Compatibility boundary for checkpoints written before the paired
        # approval/round contract. PendingToolApproval is itself validated and
        # carries the complete policy-planned call list; the atomic claim below
        # persists this projection before any resolution work can begin.
        pending_round = approval_support.planned_tool_round_from_pending_approval(approval)
    if pending_round.policy_state != "planned":
        raise RuntimeError("Pending tool approval has no durable policy plan.")
    if (
        pending_round.tool_round_id != approval.tool_round_id
        or pending_round.model_step_id != approval.model_step_id
        or pending_round.model_attempt_id != approval.model_attempt_id
        or (
            not reconstructed_approval_only_round
            and not approval_support.pending_approval_scope_matches_round(
                approval,
                pending_round,
            )
        )
        or [call.model_dump(mode="json") for call in pending_round.tool_calls]
        != [call.model_dump(mode="json") for call in approval.tool_calls]
    ):
        raise RuntimeError("Pending tool approval conflicts with its durable tool round.")
    gating_calls = [
        call for call in pending_round.tool_calls if call.tool_call_id == approval.tool_call_id
    ]
    gating_evidence = (
        None
        if len(gating_calls) != 1
        else approval_support.effective_tool_policy_evidence(gating_calls[0])
    )
    if len(gating_calls) != 1 or not (
        (
            gating_evidence is ToolPolicyEvidence.AUTHORITATIVE
            and gating_calls[0].policy_decision == ToolPolicyDecision.REQUIRE_APPROVAL.value
        )
        or gating_evidence is ToolPolicyEvidence.AMBIGUOUS
    ):
        raise RuntimeError(
            "Pending approval call is neither authoritatively approval-gated "
            "nor explicitly ambiguous."
        )
    resolution_intent = approval_support.approval_resolution_intent_from_checkpoint(
        checkpoint,
        redactor=redactor,
    )
    if resolution_intent is not None:
        approval_support.require_resolution_intent_matches_approval(
            resolution_intent,
            approval=approval,
        )
    if gating_tool_call_id is not None and approval.tool_call_id != gating_tool_call_id:
        raise ValueError("Tool approval identity does not match the current pending approval.")
    if recovery_tool_call_id is not None and not any(
        call.tool_call_id == recovery_tool_call_id for call in pending_round.tool_calls
    ):
        raise ValueError("Recovery tool call is not part of the pending approval round.")
    return approval, pending_round


def _pending_approval_for_atomic_claim(
    checkpoint: dict[str, Any] | None,
    *,
    approval_id: str,
    tool_round_id: str,
    gating_tool_call_id: str | None = None,
    recovery_tool_call_id: str | None = None,
    redactor: SecretRedactor,
) -> PendingToolApproval:
    approval, _pending_round = _pending_approval_and_round_for_atomic_claim(
        checkpoint,
        approval_id=approval_id,
        tool_round_id=tool_round_id,
        gating_tool_call_id=gating_tool_call_id,
        recovery_tool_call_id=recovery_tool_call_id,
        redactor=redactor,
    )
    return approval


def _checkpoint_with_legacy_approval_round(
    checkpoint: dict[str, Any] | None,
    *,
    approval: PendingToolApproval,
    redactor: SecretRedactor,
) -> dict[str, Any] | None:
    """Atomically upgrade an approval-only checkpoint at its exact claim."""

    pending_round = tool_round_recovery.pending_tool_round_from_checkpoint(
        checkpoint,
        redactor=redactor,
    )
    if pending_round is not None:
        return checkpoint
    current_approval = approval_support.pending_approval_from_checkpoint(
        checkpoint,
        redactor=redactor,
    )
    if current_approval != approval:
        raise RuntimeError("Pending tool approval changed before legacy round migration.")
    copied = {} if checkpoint is None else copy_json_value(checkpoint, "checkpoint")
    copied[tool_round_recovery.PENDING_TOOL_ROUND_CHECKPOINT_KEY] = (
        approval_support.planned_tool_round_from_pending_approval(approval).model_dump(mode="json")
    )
    return copied


def _approval_interrupt_close_intent_matches(
    checkpoint: dict[str, Any] | None,
    *,
    pending_round: tool_round_recovery.PendingToolRound,
) -> bool:
    """Require exact durable proof before recovering a cleared approval as interrupted."""

    if checkpoint is None:
        return False
    interrupt_payload = checkpoint.get(_PENDING_SESSION_INTERRUPT_CHECKPOINT_KEY)
    if (
        type(interrupt_payload) is not dict
        or interrupt_payload.get("interruption_type") != _INTERRUPTION_TYPE_OPERATOR_REQUESTED
        or type(interrupt_payload.get("interruption_request_id")) is not str
        or not interrupt_payload["interruption_request_id"].strip()
        or interrupt_payload["interruption_request_id"].strip()
        != interrupt_payload["interruption_request_id"]
    ):
        return False
    intent = interrupt_payload.get(approval_support.APPROVAL_INTERRUPT_CLOSE_INTENT_KEY)
    if type(intent) is not dict:
        return False
    identity = tool_round_recovery.pending_tool_round_identity(pending_round)
    expected = {
        "tool_call_id": _pending_round_policy_gate_call_id(pending_round),
        **identity.payload(),
    }
    return (
        all(intent.get(key) == value for key, value in expected.items())
        and type(intent.get("approval_id")) is str
    )


def _pending_round_policy_gate_call_id(
    pending_round: tool_round_recovery.PendingToolRound,
) -> str | None:
    """Return the call that must own this round's visible policy gate."""

    for call in pending_round.tool_calls:
        if (
            approval_support.effective_tool_policy_evidence(call)
            is ToolPolicyEvidence.AUTHORITATIVE
            and call.policy_decision == ToolPolicyDecision.REQUIRE_APPROVAL.value
        ):
            return call.tool_call_id
    for call in pending_round.tool_calls:
        if approval_support.effective_tool_policy_evidence(call) is ToolPolicyEvidence.AMBIGUOUS:
            return call.tool_call_id
    return None


def _require_native_structured_output_support(
    structured_output: StructuredOutputSpec | None,
    *,
    registered_provider: runtime_records.RegisteredProvider,
) -> None:
    _require_provider_native_output_support(
        structured_output,
        provider_name=registered_provider.name,
        provider=registered_provider.provider,
    )


def _recovery_abandonment_signal(
    error: BaseException | None,
    *,
    cancellation_baseline: int = 0,
) -> GeneratorExit | asyncio.CancelledError | None:
    """Find explicit abandonment, preferring cancellation for cleanup shielding."""
    if isinstance(error, GeneratorExit | asyncio.CancelledError):
        return error
    if isinstance(error, BaseExceptionGroup):
        try:
            task = asyncio.current_task()
        except RuntimeError:
            task = None
        cancellation_delivered = task is None or task.cancelling() > cancellation_baseline
        generator_exit: GeneratorExit | None = None
        for candidate in iter_exception_tree(error):
            if isinstance(candidate, asyncio.CancelledError) and cancellation_delivered:
                return candidate
            if isinstance(candidate, GeneratorExit) and generator_exit is None:
                generator_exit = candidate
        return generator_exit
    return None


def _task_cancellation_count() -> int:
    """Return the current task's cancellation generation for boundary tracking."""
    task = asyncio.current_task()
    return 0 if task is None else task.cancelling()


def _prepend_exception_cause(error: BaseException, cause: BaseException) -> None:
    """Preserve a new structured cause without discarding an existing chain."""
    set_exception_cause(cause, exception_cause(error))
    set_exception_cause(error, cause)


def _exception_graph_contains_identity(
    error: BaseException,
    target: BaseException,
) -> bool:
    """Inspect one base-owned exception graph without invoking extension accessors."""

    pending = [error]
    visited: set[int] = set()
    while pending:
        candidate = pending.pop()
        if candidate is target:
            return True
        candidate_id = id(candidate)
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
        elif not exception_suppresses_context(candidate):
            context = exception_context(candidate)
            if context is not None:
                pending.append(context)
    return False


def _attach_exception_cause_preserving_graph(
    error: BaseException,
    cause: BaseException,
) -> bool:
    """Attach one cause without mutating it or discarding either existing graph."""

    if _exception_graph_contains_identity(error, cause):
        return True
    existing = exception_cause(error)
    if existing is None and not exception_suppresses_context(error):
        existing = exception_context(error)
    if existing is None:
        return set_exception_cause(error, cause)
    if _exception_graph_contains_identity(cause, existing):
        return set_exception_cause(error, cause)
    combined = BaseExceptionGroup(
        "Continuation recovery retained prior and concurrent failure evidence",
        [cause, existing],
    )
    return set_exception_cause(error, combined)


async def _run_recovery_cleanup_steps(
    *,
    authoritative_failure: BaseException | None,
    steps: tuple[tuple[str, RecoveryCleanup], ...],
    cancellation_baseline: int = 0,
) -> tuple[tuple[str, BaseException], ...]:
    """Run every handoff cleanup without obscuring its triggering failure.

    Once task cancellation starts a continuation handoff, a later ``cancel()``
    must not interrupt finalization or fence release. Run that cleanup in a
    shielded child task which inherits the current run-fence context, and wait
    through any repeated cancellation requests. ``GeneratorExit`` is different:
    an explicit ``aclose()`` consumes it, so a cleanup failure must remain visible
    to the caller instead of being reduced to an exception note.
    """

    async def run_steps() -> list[tuple[str, BaseException]]:
        cleanup_failures: list[tuple[str, BaseException]] = []
        for operation, cleanup in steps:
            try:
                await cleanup()
            except BaseException as cleanup_failure:
                cleanup_failures.append((operation, cleanup_failure))
        return cleanup_failures

    abandonment = _recovery_abandonment_signal(
        authoritative_failure,
        cancellation_baseline=cancellation_baseline,
    )
    if isinstance(abandonment, asyncio.CancelledError):
        cleanup_task = asyncio.create_task(run_steps())
        while not cleanup_task.done():
            try:
                await asyncio.shield(cleanup_task)
            except asyncio.CancelledError:
                # Preserve the first cancellation as the caller-visible outcome,
                # but do not let later cancellation requests strand durable state.
                continue
        cleanup_failures = cleanup_task.result()
    else:
        cleanup_failures = await run_steps()

    if not cleanup_failures:
        return ()
    if isinstance(abandonment, asyncio.CancelledError) and authoritative_failure is not None:
        fatal_cleanup_failures = [
            failure
            for _operation, failure in cleanup_failures
            if any(
                not isinstance(candidate, BaseExceptionGroup)
                and not isinstance(candidate, (Exception, asyncio.CancelledError))
                for candidate in iter_exception_tree(failure)
            )
        ]
        if fatal_cleanup_failures:
            fatal_failure: BaseException
            if len(cleanup_failures) == 1:
                fatal_failure = cleanup_failures[0][1]
            else:
                fatal_failure = BaseExceptionGroup(
                    "Continuation recovery cleanup and process-control failures",
                    [failure for _operation, failure in cleanup_failures],
                )
            if not _attach_exception_cause_preserving_graph(
                fatal_failure,
                authoritative_failure,
            ):
                fatal_failure = BaseExceptionGroup(
                    "Continuation recovery cancellation and fatal cleanup failure",
                    [fatal_failure, authoritative_failure],
                )
            raise fatal_failure
    if authoritative_failure is not None and not isinstance(authoritative_failure, GeneratorExit):
        failures = tuple(cleanup_failures)
        for operation, cleanup_failure in cleanup_failures:
            authoritative_failure.add_note(
                "Continuation recovery cleanup failed during "
                f"{operation}: {type(cleanup_failure).__name__}. "
                "The original failure remains authoritative."
            )
        cleanup_group = BaseExceptionGroup(
            "Continuation recovery cleanup failures",
            [failure for _operation, failure in failures],
        )
        _prepend_exception_cause(authoritative_failure, cleanup_group)
        return failures

    operation, first_failure = cleanup_failures[0]
    for later_operation, later_failure in cleanup_failures[1:]:
        first_failure.add_note(
            "Additional continuation recovery cleanup failure during "
            f"{later_operation}: {later_failure!r}."
        )
    first_failure.add_note(f"Continuation recovery cleanup failed during {operation}.")
    if len(cleanup_failures) > 1:
        _prepend_exception_cause(
            first_failure,
            BaseExceptionGroup(
                "Additional continuation recovery cleanup failures",
                [failure for _operation, failure in cleanup_failures[1:]],
            ),
        )
    raise first_failure


@dataclass(frozen=True)
class RecoverySessionRunRequest:
    session: Session
    registered_agent: runtime_records.RegisteredAgentState
    registered_provider: runtime_records.RegisteredProvider
    registered_environment: runtime_records.RegisteredEnvironment | None
    active_invocation_profile: ActiveInvocationExecutionProfile
    messages: list[Message]
    messages_to_append: list[Message]
    max_steps: int
    limits: RunLimits
    budget_limits: tuple[BudgetLimit, ...]
    budget_policy: BudgetPolicy | None
    retry_policy: RetryPolicy
    structured_output: StructuredOutputSpec | None
    thinking: ThinkingConfig | None
    request_loop_policies: tuple[LoopPolicy, ...]
    request_metadata: dict[str, Any]
    task_id: str | None
    task_worker_id: str | None
    start_event_type: EventType | None
    start_event_payload: dict[str, Any]
    start_task_on_enter: bool
    release_run_fence_on_exit: bool
    run_limit_accounting: RunLimitAccountingContext | None = None
    initial_model_step_identity: ModelStepIdentity | None = None
    initial_model_step_number: int | None = None
    initial_model_step_tool_exposure: ResolvedToolExposureAuthority | None = None
    previous_tool_exposure_profile_id: str | None = None
    preserve_failure_until_initial_provider_dispatch: bool = False


def _rebound_active_invocation_profile(
    session: Session,
    snapshot: ActiveInvocationExecutionProfile,
) -> ActiveInvocationExecutionProfile:
    """Carry one validated profile into the run epoch claimed for recovery."""

    if snapshot.session_id != session.id:
        raise RuntimeError("Recovery profile authority belongs to a different session.")
    return snapshot.model_copy(update={"run_epoch": session.run_epoch})


def _continued_tool_exposure_profile_id(
    authority: ResolvedToolExposureAuthority | None,
) -> str:
    """Return the profile preceding a post-tool continuation model step."""

    if authority is None:
        # Pending rounds written before compact exposure authority was added used
        # the byte-compatible expose-all policy. Execution-profile validation
        # guarantees that the registered catalog and policy did not drift.
        return ALL_REGISTERED_TOOLS_PROFILE_ID
    if type(authority) is not ResolvedToolExposureAuthority:
        raise TypeError("authority must be a ResolvedToolExposureAuthority or None.")
    return authority.profile_id


def _retried_model_step_tool_exposure_authority(
    authority: ResolvedToolExposureAuthority | None,
    registered_agent: runtime_records.RegisteredAgentState,
    session: Session,
) -> ResolvedToolExposureAuthority:
    """Return exact exposure authority for a same-model-step retry."""

    if authority is None:
        raise ProviderOperationEvidenceError(
            "Provider-operation fallback has no durable tool-exposure authority."
        )
    try:
        validated = validate_resolved_tool_exposure_authority(
            authority,
            registered_agent.tool_capabilities,
        )
        capability_ceiling = tool_capability_ceiling_from_session_metadata(session.metadata)
    except (TypeError, ValueError) as exc:
        raise ProviderOperationEvidenceError(
            "Provider-operation fallback has invalid durable tool-exposure authority."
        ) from exc
    ceiling_names = frozenset(capability_ceiling.tool_names)
    if validated.ceiling_count != len(capability_ceiling.tool_names) or any(
        name not in ceiling_names for name in validated.tool_names
    ):
        raise ProviderOperationEvidenceError(
            "Provider-operation fallback tool exposure conflicts with the session capability "
            "ceiling."
        )
    return validated


@dataclass(frozen=True)
class DeferredInputMaterialization:
    messages: list[Message]
    cancellation: asyncio.CancelledError | None


@dataclass(frozen=True)
class RecoveryTerminalEventRequest:
    event: Event
    phase: RuntimeHookPhase
    session: Session
    registered_agent: runtime_records.RegisteredAgentState
    registered_environment: runtime_records.RegisteredEnvironment | None
    execution_profile: ExecutionProfileIdentity | None = None


@dataclass(frozen=True)
class ProviderOperationFailureRequest:
    resolution_event: Event
    session: Session
    registered_agent: runtime_records.RegisteredAgentState
    registered_environment: runtime_records.RegisteredEnvironment | None
    execution_profile: ExecutionProfileIdentity
    legacy_resolution_without_profile: bool = False


@dataclass(frozen=True)
class RecoveryLimitStopRequest:
    session: Session
    registered_agent: runtime_records.RegisteredAgentState
    registered_environment: runtime_records.RegisteredEnvironment | None
    environment_name: str | None
    decision: StopDecision
    usage_summary: SessionUsageSummary
    cost_summary: SessionCostSummary | None
    messages: list[Message]
    tool_calls: list[runtime_records.ToolCallRequest]
    completed_tool_outcomes: list[runtime_records.ToolCallOutcome]
    pending_approval_to_clear: PendingToolApproval | None
    deferred_messages: list[Message]
    requested_approval_decision: ToolApprovalDecision | None
    approval_resolution_request_digest: str | None
    execution_profile: ExecutionProfileIdentity | None = None


@dataclass(frozen=True)
class RecoveryTaskEventRequest:
    event_type: EventType
    task: Task
    session: Session
    registered_agent: runtime_records.RegisteredAgentState
    registered_environment: runtime_records.RegisteredEnvironment | None


@dataclass(frozen=True)
class RecoveryInterruptionRequest:
    session: Session
    registered_agent: runtime_records.RegisteredAgentState
    registered_environment: runtime_records.RegisteredEnvironment | None
    environment_name: str | None
    execution_profile: ExecutionProfileIdentity | None = None


@dataclass(frozen=True)
class RecoveryAbandonedTurnRequest:
    session: Session
    registered_agent: runtime_records.RegisteredAgentState
    registered_environment: runtime_records.RegisteredEnvironment | None
    environment_name: str | None
    run_started_at: float | None
    usage_tracker: SessionUsageTracker | None
    active_run: ActiveSessionRun[SessionUsageTracker] | None
    execution_profile: ExecutionProfileIdentity | None = None


@dataclass(frozen=True)
class RecoveryAbandonedSessionRequest:
    session: Session
    registered_agent: runtime_records.RegisteredAgentState
    registered_environment: runtime_records.RegisteredEnvironment | None
    environment_name: str | None
    run_started_at: float | None = None
    turn_usage_tracker: SessionUsageTracker | None = None
    active_run: ActiveSessionRun[SessionUsageTracker] | None = None
    interaction_transition_failures: tuple[dict[str, Any], ...] = ()
    interaction_transition: InteractionTransitionSpec | None = None
    execution_profile: ExecutionProfileIdentity | None = None


@dataclass(frozen=True)
class _IncompleteRecoveryClaim:
    claim_id: str
    claim_expires_at: datetime
    session_before_fence: Session
    session: Session
    run_operation: _SessionRunOperation | None = None


@dataclass(frozen=True)
class _TerminalEvidenceInspection:
    event: Event | None
    pending_interrupt_payload: dict[str, Any] | None
    pending_action_interrupt_payload: dict[str, Any] | None
    run_operation: _SessionRunOperation | None
    terminal_event_required: bool


class _IncompleteRecoveryClaimLost(RuntimeError):
    """The durable incomplete-session recovery lease is no longer owned."""


class ModelCompletionManualRecoveryRequired(RuntimeError):
    """A model dispatch cannot be reconstructed safely without operator input."""


@dataclass(frozen=True)
class ModelCompletionBoundaryReconciliation:
    """Verified state at the durable model-completion publication boundary."""

    state: Literal[
        "none",
        "promoted",
        "already_promoted",
        "provider_operation_pending",
        "provider_operation_reconciled",
        "provider_operation_unavailable",
    ]
    session: Session
    pointer: model_completion_publication.ModelStepPublicationCheckpoint | None = None
    completion_event: Event | None = None
    pending_tool_round: tool_round_recovery.PendingToolRound | None = None
    transcript_cursor: int = 0
    recovery_events: tuple[Event, ...] = ()

    @property
    def blocks_provider_dispatch(self) -> bool:
        return (
            self.pointer is not None
            and self.pending_tool_round is None
            and self.transcript_cursor == self.pointer.transcript_end_cursor
        )


class _ManualRecoveryInterrupted(RuntimeError):
    """A durable interruption won before manual recovery could claim the session."""


class _ManualRecoveryCascadePending(RuntimeError):
    """Descendant interruption must finish before manual recovery can continue."""


@dataclass(frozen=True)
class _ManualRecoveryInterruptionFence:
    session: Session
    claim_id: str
    error: BaseException | None


@dataclass(frozen=True)
class _ManualRecoveryEventDelivery:
    event: Event
    consumed: asyncio.Event


@dataclass(frozen=True)
class _ManualRecoveryStreamOutcome:
    error: BaseException | None
    interrupted_event: Event | None = None


@dataclass(frozen=True)
class _ManualRecoverySupervisorResult:
    error: BaseException | None
    cleanup_failure: BaseException | None


@dataclass(frozen=True)
class _ManualRecoveryPersistenceReconciliation:
    persisted: bool | None
    error: Exception | None = None
    cancellation: asyncio.CancelledError | None = None


@dataclass(frozen=True)
class _RecoveryInvocationSemantics:
    """Exact provider-dispatch semantics reconstructed for one continuation."""

    max_steps: int
    limits: RunLimits
    budget_limits: tuple[BudgetLimit, ...]
    retry_policy: RetryPolicy
    structured_output: StructuredOutputSpec | None
    thinking: ThinkingConfig | None


RunSession = Callable[[RecoverySessionRunRequest], AsyncGenerator[Event, None]]
TerminalEventStream = Callable[[RecoveryTerminalEventRequest], AsyncIterator[Event]]
ProviderOperationFailureStream = Callable[[ProviderOperationFailureRequest], AsyncIterator[Event]]
LimitStopEventStream = Callable[[RecoveryLimitStopRequest], AsyncIterator[Event]]
TaskEventFactory = Callable[[RecoveryTaskEventRequest], Event]
RegisteredAgentResolver = Callable[[str], runtime_records.RegisteredAgentState]
RegisteredProviderResolver = Callable[[str], runtime_records.RegisteredProvider]
RegisteredEnvironmentResolver = Callable[[str | None], runtime_records.RegisteredEnvironment | None]
BudgetPolicyResolver = Callable[[], BudgetPolicy | None]


class ExecutionProfileContinuationValidator(Protocol):
    def __call__(
        self,
        session: Session,
        checkpoint: dict[str, Any] | None,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_provider: runtime_records.RegisteredProvider,
        request_loop_policies: tuple[LoopPolicy, ...] | None = None,
        frozen_candidate_profile: ExecutionProfileIdentity | None = None,
        *,
        budget_policy: BudgetPolicy | None,
        request_budget_limits: tuple[BudgetLimit, ...] = (),
        structured_output: StructuredOutputSpec | None = None,
        thinking: ThinkingConfig | None = None,
        max_steps: int = 16,
        limits: RunLimits | None = None,
        retry_policy: RetryPolicy | None = None,
        invocation_semantics_available: bool = False,
        require_open_interaction: bool = True,
        additional_profile_fingerprints: tuple[str, ...] = (),
    ) -> Awaitable[ActiveInvocationExecutionProfile]: ...


RecoveryInterruptionStream = Callable[[RecoveryInterruptionRequest], AsyncIterator[Event]]
PendingSessionInterruptCheckpoint = Callable[[dict[str, Any], datetime], CheckpointTransform]
AbandonedTurnCompleted = Callable[[RecoveryAbandonedTurnRequest], Awaitable[Session]]
IncompleteRecoveryScopeHook = Callable[[str], Awaitable[None]]
RecoveryMutationHook = Callable[[], Awaitable[None]]
IncompleteRecoveryResultHook = Callable[
    [IncompleteSessionRecoveryResult],
    Awaitable[IncompleteSessionRecoveryResult],
]
MaterializeDeferredInteractionInput = Callable[[str], Awaitable[bool]]
ResumeInteraction = Callable[
    [
        Session,
        runtime_records.RegisteredAgentState,
        runtime_records.RegisteredEnvironment | None,
    ],
    Awaitable[Event | None],
]
RecoverProviderOperation = Callable[
    [
        Session,
        ModelCompletionStage,
        RecoverableProviderOperation,
        runtime_records.RegisteredAgentState,
        runtime_records.RegisteredProvider,
        runtime_records.RegisteredEnvironment | None,
    ],
    Awaitable[ProviderOperationRecoveryResult],
]
RecoverProviderOperationStart = Callable[
    [
        Session,
        ModelCompletionStage,
        RecoverableProviderOperationStart,
        runtime_records.RegisteredAgentState,
        runtime_records.RegisteredProvider,
        runtime_records.RegisteredEnvironment | None,
    ],
    Awaitable[ProviderOperationRecoveryResult],
]
CancelProviderOperation = Callable[
    [
        Session,
        ModelCompletionStage,
        RecoverableProviderOperation,
        runtime_records.RegisteredAgentState,
        runtime_records.RegisteredProvider,
        runtime_records.RegisteredEnvironment | None,
    ],
    Awaitable[ProviderOperationSnapshot | None],
]
InteractionTransitionReplayFailures = Callable[
    [BaseException],
    tuple[Exception, ...] | None,
]


class RecoveryCoordinator:
    """Continue paused work and repair incomplete sessions from durable state."""

    def __init__(
        self,
        *,
        session_store: SessionStore,
        task_store: TaskStore | None,
        event_writer: RuntimeEventWriter,
        session_control: SessionControl[SessionUsageTracker],
        environment_lifecycle: EnvironmentLifecycle,
        run_limit_controller: RunLimitController,
        tool_round_executor: ToolRoundExecutor,
        secret_redactor: SecretRedactor,
        clock: Callable[[], datetime],
        checkpoint_transform: CheckpointTransformFactory,
        effective_retry_policy: EffectiveRetryPolicy,
        run_session: RunSession,
        emit_terminal_event_with_hooks: TerminalEventStream,
        fail_provider_operation: ProviderOperationFailureStream,
        stop_session_for_limit_reached: LimitStopEventStream,
        task_event: TaskEventFactory,
        resolve_registered_agent: RegisteredAgentResolver,
        resolve_registered_provider: RegisteredProviderResolver,
        resolve_registered_environment: RegisteredEnvironmentResolver,
        resolve_budget_policy: BudgetPolicyResolver,
        validate_execution_profile_continuation: ExecutionProfileContinuationValidator,
        interrupt_session_for_recovery: RecoveryInterruptionStream,
        pending_session_interrupt_checkpoint: PendingSessionInterruptCheckpoint,
        abandoned_turn_completed: AbandonedTurnCompleted,
        resume_interaction: ResumeInteraction,
        recover_provider_operation: RecoverProviderOperation,
        recover_provider_operation_start: RecoverProviderOperationStart,
        cancel_provider_operation: CancelProviderOperation,
        interaction_transition_replay_failures: InteractionTransitionReplayFailures,
    ) -> None:
        self._session_store = session_store
        self._task_store = task_store
        self._event_writer = event_writer
        self._session_control = session_control
        self._environment_lifecycle = environment_lifecycle
        self._run_limit_controller = run_limit_controller
        self._tool_round_executor = tool_round_executor
        self._secret_redactor = secret_redactor
        self._clock = clock
        self._checkpoint_transform = checkpoint_transform
        self._effective_retry_policy = effective_retry_policy
        self._run_session = run_session
        self._emit_terminal_event_with_hooks = emit_terminal_event_with_hooks
        self._fail_provider_operation = fail_provider_operation
        self._stop_session_for_limit_reached = stop_session_for_limit_reached
        self._task_event = task_event
        self._resolve_registered_agent = resolve_registered_agent
        self._resolve_registered_provider = resolve_registered_provider
        self._resolve_registered_environment = resolve_registered_environment
        self._resolve_budget_policy = resolve_budget_policy
        self._validate_execution_profile_continuation = validate_execution_profile_continuation
        self._interrupt_session_for_recovery = interrupt_session_for_recovery
        self._pending_session_interrupt_checkpoint = pending_session_interrupt_checkpoint
        self._abandoned_turn_completed = abandoned_turn_completed
        self._resume_interaction = resume_interaction
        self._recover_provider_operation = recover_provider_operation
        self._recover_provider_operation_start = recover_provider_operation_start
        self._cancel_provider_operation = cancel_provider_operation
        self._interaction_transition_replay_failures = interaction_transition_replay_failures
        self._workspace_artifact_recovery_operations = BoundedInvocationOperationRegistry(
            max_operations=64
        )

    async def _recoverable_provider_operation(
        self,
        stage: ModelCompletionStage,
        *,
        registered_provider: runtime_records.RegisteredProvider | None = None,
    ) -> (
        tuple[
            RecoverableProviderOperation | RecoverableProviderOperationStart,
            runtime_records.RegisteredProvider,
        ]
        | None
    ):
        try:
            operation = await load_recoverable_provider_operation(self._session_store, stage)
            if operation is None:
                operation = await load_recoverable_provider_operation_start(
                    self._session_store,
                    stage,
                )
        except ProviderOperationEvidenceError as evidence_error:
            raise ModelCompletionManualRecoveryRequired(
                "Provider-operation recovery cannot continue because provider output already "
                "crossed Cayu's durable acceptance boundary."
            ) from evidence_error
        if operation is None:
            return None
        if registered_provider is None:
            try:
                registered_provider = self._resolve_registered_provider(operation.provider)
            except KeyError:
                return None
        elif registered_provider.name != operation.provider:
            return None
        provider = registered_provider.provider
        if (
            provider.provider_operation_mode is not ProviderOperationMode.BACKGROUND
            or not isinstance(
                provider.provider_operations,
                ProviderOperationAdapter,
            )
        ):
            return None
        return operation, registered_provider

    async def load_model_completion_boundary(
        self,
        session: Session,
    ) -> ActiveModelCompletionStage | None:
        """Load and validate durable model work without consulting a provider."""

        active = await self._session_store.load_active_model_completion_stage(session.id)
        if active is not None:
            self._validate_active_model_completion_stage(session, active.stage)
        return active

    async def preflight_model_completion_boundary(
        self,
        session: Session,
        *,
        registered_provider: runtime_records.RegisteredProvider | None = None,
        active_stage: ActiveModelCompletionStage | None = None,
    ) -> bool:
        """Reject a non-terminal dispatch and report terminal work to promote."""

        if active_stage is None:
            active = await self.load_model_completion_boundary(session)
        else:
            if type(active_stage) is not ActiveModelCompletionStage:
                raise TypeError("active_stage must be an ActiveModelCompletionStage.")
            active = active_stage.model_copy(deep=True)
        if active is None:
            return False
        stage = active.stage
        self._validate_active_model_completion_stage(session, stage)
        if stage.state == "in_flight":
            if (
                await self._recoverable_provider_operation(
                    stage,
                    registered_provider=registered_provider,
                )
                is not None
            ):
                return True
            raise ModelCompletionManualRecoveryRequired(
                "The active model-completion dispatch has no durable terminal response. "
                "Its provider outcome and linked budget reservations require manual "
                f"reconciliation before retrying: {stage.stage_id}"
            )
        return True

    async def cancel_provider_operation_for_interruption(
        self,
        session: Session,
        *,
        registered_agent: runtime_records.RegisteredAgentState | None = None,
        registered_provider: runtime_records.RegisteredProvider | None = None,
        registered_environment: runtime_records.RegisteredEnvironment | None = None,
    ) -> ActiveInvocationExecutionProfile | None:
        """Address one durable in-flight provider operation without a live worker."""

        if registered_agent is None and registered_provider is not None:
            raise ValueError("Frozen provider-operation cancellation requires the original agent.")

        active = await self._session_store.load_active_model_completion_stage(session.id)
        if active is None or active.stage.state != "in_flight":
            return None
        stage = active.stage
        self._validate_active_model_completion_stage(session, stage)
        if registered_agent is not None and registered_provider is None:
            raise ModelCompletionManualRecoveryRequired(
                "Provider-operation cancellation requires the original provider registration."
            )
        recoverable = await self._recoverable_provider_operation(
            stage,
            registered_provider=registered_provider,
        )
        if recoverable is None:
            return None
        operation, recovered_provider = recoverable
        if isinstance(operation, RecoverableProviderOperationStart):
            # There is no durable provider operation identity to cancel yet.
            # The normal boundary reconciler may replay an exact-idempotent
            # start; an unsupported start remains explicitly ambiguous.
            return None
        if registered_agent is None or registered_provider is None:
            try:
                registered_agent = self._resolve_registered_agent(session.agent_name)
                registered_provider = recovered_provider
                registered_environment = self._resolve_registered_environment(
                    session.environment_name
                )
            except KeyError as registration_error:
                raise ModelCompletionManualRecoveryRequired(
                    "Provider-operation cancellation requires the original agent, provider, "
                    "and environment registrations."
                ) from registration_error
        elif registered_provider is not recovered_provider:
            if registered_provider.name != recovered_provider.name:
                raise ModelCompletionManualRecoveryRequired(
                    "Provider-operation cancellation resolved a different provider identity."
                )
            recovered_provider = registered_provider
        checkpoint = await self._session_store.load_checkpoint(session.id)
        budget_policy_snapshot = copy_budget_policy(self._resolve_budget_policy())
        execution_profile_snapshot = await self._validate_execution_profile_continuation(
            session,
            checkpoint,
            registered_agent,
            recovered_provider,
            None,
            budget_policy=budget_policy_snapshot,
        )
        cancellation = await self._cancel_provider_operation(
            session,
            stage,
            operation,
            registered_agent,
            recovered_provider,
            registered_environment,
        )
        if cancellation is not None and cancellation.status is ProviderOperationStatus.COMPLETED:
            recovered = await self._recover_provider_operation(
                session,
                stage,
                operation,
                registered_agent,
                recovered_provider,
                registered_environment,
            )
            if recovered.status is not ProviderOperationRecoveryStatus.RECONCILED:
                raise ModelCompletionManualRecoveryRequired(
                    "Provider completion won cancellation but could not be reconciled."
                )
        return execution_profile_snapshot

    async def claim_provider_operation_interruption(
        self,
        session: Session,
        execution_profile_snapshot: ActiveInvocationExecutionProfile,
        *,
        interruption_request_id: str,
    ) -> Session:
        """Fence a dead provider worker and bind interruption to one recovery epoch."""

        if type(execution_profile_snapshot) is not ActiveInvocationExecutionProfile:
            raise TypeError(
                "Provider-operation interruption requires active invocation profile authority."
            )
        interruption_request_id = require_clean_nonblank(
            interruption_request_id,
            "interruption_request_id",
        )

        def claim_interruption(
            current_session: Session,
            checkpoint: dict[str, Any] | None,
        ) -> dict[str, Any]:
            current_profile = active_invocation_execution_profile_from_checkpoint(checkpoint)
            pending_interrupt = (
                None
                if checkpoint is None
                else checkpoint.get(_PENDING_SESSION_INTERRUPT_CHECKPOINT_KEY)
            )
            if (
                type(pending_interrupt) is not dict
                or interruption_request_id_from_payload(pending_interrupt)
                != interruption_request_id
                or _incomplete_recovery_claim_from_checkpoint(checkpoint) is not None
            ):
                raise SessionRunFenced(
                    "Provider-operation interruption request ownership changed before claim."
                )
            if (
                type(current_profile) is not ActiveInvocationExecutionProfile
                or current_profile != execution_profile_snapshot
            ):
                raise RuntimeError(
                    "Provider-operation interruption profile changed before recovery claimed it."
                )
            if (
                current_profile.session_id != current_session.id
                or current_profile.run_epoch != current_session.run_epoch
            ):
                raise SessionRunFenced(
                    "Provider-operation interruption no longer owns the active invocation epoch."
                )
            return checkpoint_with_active_invocation_execution_profile(
                checkpoint,
                session_id=current_session.id,
                interaction_id=current_profile.interaction_id,
                run_epoch=current_session.run_epoch + 1,
                profile=current_profile.profile,
                expected=current_profile,
            )

        claim_task = asyncio.create_task(
            self._session_store.fence_run_and_transform_checkpoint(
                session.id,
                statuses={SessionStatus.INTERRUPTING},
                checkpoint_transform=claim_interruption,
            )
        )
        outcome = await await_shielded_task_outcome(claim_task)
        error = outcome.error
        if isinstance(error, asyncio.CancelledError) and outcome.cancellation is None:
            error = unexpected_child_cancellation_error(
                error,
                operation="Provider-operation interruption claim",
            )
        cancellation = outcome.cancellation
        claimed = outcome.result
        if error is not None:
            reconciliation = await await_shielded_task_outcome(
                asyncio.create_task(
                    self._load_claimed_provider_operation_interruption(
                        session_id=session.id,
                        interaction_id=execution_profile_snapshot.interaction_id,
                        run_epoch=session.run_epoch + 1,
                        execution_profile=execution_profile_snapshot.profile,
                        interruption_request_id=interruption_request_id,
                    )
                ),
                cancellation=cancellation,
            )
            cancellation = reconciliation.cancellation or cancellation
            reconciliation_error = reconciliation.error
            if (
                isinstance(reconciliation_error, asyncio.CancelledError)
                and reconciliation.cancellation is None
            ):
                reconciliation_error = unexpected_child_cancellation_error(
                    reconciliation_error,
                    operation="Provider-operation interruption claim reconciliation",
                )
            if reconciliation_error is not None:
                if cancellation is not None:
                    cancellation.add_note(
                        "Provider-operation interruption claim reconciliation also failed: "
                        f"{type(reconciliation_error).__name__}."
                    )
                    _prepend_exception_cause(
                        cancellation,
                        BaseExceptionGroup(
                            "Provider-operation interruption claim failures",
                            [error, reconciliation_error],
                        ),
                    )
                    raise cancellation
                if not isinstance(reconciliation_error, Exception):
                    raise reconciliation_error from error
                error.add_note(
                    "Provider-operation interruption claim reconciliation failed: "
                    f"{type(reconciliation_error).__name__}."
                )
                raise error from reconciliation_error
            claimed = reconciliation.result
            if claimed is None:
                if cancellation is not None:
                    cancellation.add_note(
                        f"Provider-operation interruption claim also failed: {type(error).__name__}."
                    )
                    raise cancellation from error
                raise error
            _activate_session_run_fence(claimed)
            authoritative_failure = cancellation or error
            try:
                await _run_recovery_cleanup_steps(
                    authoritative_failure=authoritative_failure,
                    steps=(
                        (
                            "failed provider-operation interruption claim release",
                            lambda: (
                                self._environment_lifecycle.release_run_fence_after_environment_cleanup(
                                    session_id=claimed.id,
                                    execution_profile=execution_profile_snapshot.profile,
                                )
                            ),
                        ),
                    ),
                )
            finally:
                _deactivate_session_run_fence(claimed.id)
            if cancellation is not None:
                cancellation.add_note(
                    f"Provider-operation interruption claim also failed: {type(error).__name__}."
                )
                raise cancellation from error
            raise error
        if claimed is None:
            missing = RuntimeError(
                "Provider-operation interruption claim returned no session authority."
            )
            if cancellation is not None:
                cancellation.add_note(str(missing))
                raise cancellation from missing
            raise missing
        _activate_session_run_fence(claimed)
        if cancellation is not None:
            try:
                await _run_recovery_cleanup_steps(
                    authoritative_failure=cancellation,
                    steps=(
                        (
                            "cancelled provider-operation interruption run-fence release",
                            lambda: (
                                self._environment_lifecycle.release_run_fence_after_environment_cleanup(
                                    session_id=claimed.id,
                                    execution_profile=execution_profile_snapshot.profile,
                                )
                            ),
                        ),
                    ),
                )
            finally:
                _deactivate_session_run_fence(claimed.id)
            raise cancellation
        return claimed

    async def _load_claimed_provider_operation_interruption(
        self,
        *,
        session_id: str,
        interaction_id: str,
        run_epoch: int,
        execution_profile: ExecutionProfileIdentity,
        interruption_request_id: str,
    ) -> Session | None:
        """Reconcile acknowledgement loss only against the complete claimed identity."""

        current = await self._session_store.load(session_id)
        if (
            current is None
            or current.status is not SessionStatus.INTERRUPTING
            or current.run_epoch != run_epoch
        ):
            return None
        checkpoint = await self._session_store.load_checkpoint(session_id)
        current_profile = active_invocation_execution_profile_from_checkpoint(checkpoint)
        pending_interrupt = (
            None
            if checkpoint is None
            else checkpoint.get(_PENDING_SESSION_INTERRUPT_CHECKPOINT_KEY)
        )
        if (
            current_profile is None
            or current_profile.session_id != session_id
            or current_profile.interaction_id != interaction_id
            or current_profile.run_epoch != run_epoch
            or current_profile.profile != execution_profile
            or type(pending_interrupt) is not dict
            or interruption_request_id_from_payload(pending_interrupt) != interruption_request_id
            or _incomplete_recovery_claim_from_checkpoint(checkpoint) is not None
        ):
            return None
        return current

    async def reconcile_model_completion_boundary(
        self,
        session: Session,
        *,
        registered_agent: runtime_records.RegisteredAgentState | None = None,
        registered_provider: runtime_records.RegisteredProvider | None = None,
        registered_environment: runtime_records.RegisteredEnvironment | None = None,
    ) -> ModelCompletionBoundaryReconciliation:
        """Promote terminal model evidence or verify an already-published boundary."""

        if (registered_agent is None) != (registered_provider is None):
            raise ValueError(
                "Frozen model-completion reconciliation requires both agent and provider."
            )

        # Fail closed for this session's own accounting before scanning the
        # shared ledger. The global pass may retain rows owned by another
        # session-store publication domain without blocking this boundary.
        await self._run_limit_controller.recover_pending_budget_settlements(session_id=session.id)
        await self._run_limit_controller.recover_pending_budget_settlements()
        active = await self._session_store.load_active_model_completion_stage(session.id)
        state: Literal[
            "none",
            "promoted",
            "already_promoted",
            "provider_operation_pending",
            "provider_operation_unavailable",
            "provider_operation_reconciled",
        ] = "none"
        recovery_events: tuple[Event, ...] = ()
        if active is not None:
            stage = active.stage
            self._validate_active_model_completion_stage(session, stage)
            if session.status in {
                SessionStatus.COMPLETED,
                SessionStatus.INTERRUPTED,
            }:
                raise ModelCompletionManualRecoveryRequired(
                    "A terminal session retains an active model-completion stage. "
                    "Provider output and linked reservations require explicit manual "
                    f"reconciliation before the stage can be cleared: {stage.stage_id}"
                )
            if stage.state == "in_flight":
                recoverable = await self._recoverable_provider_operation(
                    stage,
                    registered_provider=registered_provider,
                )
                if recoverable is None:
                    raise ModelCompletionManualRecoveryRequired(
                        "The active model-completion dispatch has no durable terminal response. "
                        "Its provider outcome and linked budget reservations require manual "
                        f"reconciliation before retrying: {stage.stage_id}"
                    )
                operation, recovered_provider = recoverable
                if registered_agent is None or registered_provider is None:
                    try:
                        registered_agent = self._resolve_registered_agent(session.agent_name)
                        registered_provider = recovered_provider
                        registered_environment = self._resolve_registered_environment(
                            session.environment_name
                        )
                    except KeyError as registration_error:
                        raise ModelCompletionManualRecoveryRequired(
                            "Provider-operation recovery requires the original agent, provider, "
                            "and environment registrations."
                        ) from registration_error
                elif registered_provider is not recovered_provider:
                    if registered_provider.name != recovered_provider.name:
                        raise ModelCompletionManualRecoveryRequired(
                            "Provider-operation recovery resolved a different provider identity."
                        )
                    recovered_provider = registered_provider
                checkpoint = await self._session_store.load_checkpoint(session.id)
                budget_policy_snapshot = copy_budget_policy(self._resolve_budget_policy())
                await self._validate_execution_profile_continuation(
                    session,
                    checkpoint,
                    registered_agent,
                    recovered_provider,
                    None,
                    budget_policy=budget_policy_snapshot,
                )
                if isinstance(operation, RecoverableProviderOperationStart):
                    recovered = await self._recover_provider_operation_start(
                        session,
                        stage,
                        operation,
                        registered_agent,
                        recovered_provider,
                        registered_environment,
                    )
                else:
                    recovered = await self._recover_provider_operation(
                        session,
                        stage,
                        operation,
                        registered_agent,
                        recovered_provider,
                        registered_environment,
                    )
                recovery_events = recovered.events
                if recovered.status is ProviderOperationRecoveryStatus.PENDING:
                    transcript = await self._session_store.load_transcript(session.id)
                    return ModelCompletionBoundaryReconciliation(
                        state="provider_operation_pending",
                        session=session,
                        transcript_cursor=len(transcript),
                        recovery_events=recovery_events,
                    )
                if recovered.status is ProviderOperationRecoveryStatus.UNAVAILABLE:
                    transcript = await self._session_store.load_transcript(session.id)
                    return ModelCompletionBoundaryReconciliation(
                        state="provider_operation_unavailable",
                        session=session,
                        transcript_cursor=len(transcript),
                        recovery_events=recovery_events,
                    )
                if recovered.status is not ProviderOperationRecoveryStatus.RECONCILED:
                    raise RuntimeError("Provider-operation recovery returned an unknown state.")
                active = await self._session_store.load_active_model_completion_stage(session.id)
                if active is not None:
                    raise RuntimeError(
                        "Reconciled provider operation retained an active model-completion stage."
                    )
                loaded_session = await self._session_store.load(session.id)
                if loaded_session is None:
                    raise KeyError(f"Session not found: {stage.session_id}")
                session = loaded_session
                state = "provider_operation_reconciled"
            elif session.status not in {
                stage.source_status,
                SessionStatus.INTERRUPTING,
                SessionStatus.FAILED,
            }:
                raise ModelCompletionManualRecoveryRequired(
                    "The completed model stage cannot be promoted from the current session "
                    f"status ({session.status.value}); expected {stage.source_status.value}."
                )
            else:
                session = await self._promote_completed_model_stage(
                    session=session,
                    stage_id=stage.stage_id,
                )
                state = "promoted"

        checkpoint = await self._session_store.load_checkpoint(session.id)
        pointer = model_completion_publication.model_step_publication_from_checkpoint(checkpoint)
        pending_round = tool_round_recovery.pending_tool_round_from_checkpoint(checkpoint)
        pending_approval = approval_support.pending_approval_from_checkpoint(checkpoint)
        pending_user_input = pending_user_input_from_checkpoint(checkpoint)
        if pending_approval is not None and pending_user_input is not None:
            raise RuntimeError(
                "The checkpoint contains conflicting pending approval and user-input pauses."
            )
        if pending_round is not None and pending_user_input is not None:
            raise RuntimeError(
                "The checkpoint contains conflicting pending tool-round and pause markers."
            )
        if pending_round is not None and pending_approval is not None:
            _pending_approval_for_atomic_claim(
                checkpoint,
                approval_id=pending_approval.approval_id,
                tool_round_id=pending_approval.tool_round_id,
                gating_tool_call_id=pending_approval.tool_call_id,
                redactor=self._secret_redactor,
            )
        if pointer is None:
            if active is not None:
                raise RuntimeError(
                    "Promoted model completion did not publish its durable model-step pointer."
                )
            if pending_round is not None and pending_round.source_model_step_id is not None:
                raise RuntimeError(
                    "A pending tool round exists without a durable source model-step pointer."
                )
            return ModelCompletionBoundaryReconciliation(
                state=state,
                session=session,
                recovery_events=recovery_events,
            )

        receipt = await self._session_store.load_runtime_publication_receipt(
            session.id,
            pointer.logical_step_id,
        )
        if receipt is None:
            raise RuntimeError("The durable model-step pointer has no publication receipt.")
        if (
            receipt.kind != "model-step"
            or receipt.transcript_start_cursor != pointer.source_transcript_cursor
            or receipt.transcript_end_cursor != pointer.transcript_end_cursor
            or receipt.appended_event_ids != (pointer.completion_event_id,)
            or receipt.referenced_events
        ):
            raise RuntimeError(
                "The durable model-step pointer conflicts with its publication receipt."
            )

        event_records = await self._session_store.query_events(
            EventQuery(
                session_id=session.id,
                event_id=pointer.completion_event_id,
                limit=1,
            )
        )
        if len(event_records) != 1:
            raise RuntimeError("The durable model-step pointer has no exact completion event.")
        completion_event = event_records[0].event
        if (
            completion_event.type != EventType.MODEL_COMPLETED
            or completion_event.payload.get("step_classification") != pointer.classification
            or completion_event.payload.get("transcript_cursor") != pointer.transcript_end_cursor
        ):
            raise RuntimeError(
                "The durable model-step pointer conflicts with its completion event."
            )
        completed_stage = await self._session_store.load_model_completion_stage(
            session.id,
            pointer.stage_id,
        )
        if (
            completed_stage is None
            or completed_stage.state != "completed"
            or completed_stage.logical_step_id != pointer.logical_step_id
            or completed_stage.publication is None
            or completed_stage.publication.events != (completion_event,)
        ):
            raise RuntimeError(
                "The durable model-step pointer conflicts with its completion stage."
            )
        if completed_stage.reservation_ids:
            await self._run_limit_controller.reconcile_model_completion_settlements(
                completion_event,
                reservation_ids=completed_stage.reservation_ids,
            )

        transcript_window = await self._session_store.load_transcript_window(
            session.id,
            start_index=(
                pointer.transcript_end_cursor
                if pointer.assistant_message_deferred
                else max(0, pointer.transcript_end_cursor - 1)
            ),
            limit=2,
        )
        if transcript_window.cursor < pointer.transcript_end_cursor:
            raise RuntimeError("The durable model-step pointer extends beyond the transcript.")
        if pointer.assistant_message_published and (
            not transcript_window.records
            or transcript_window.records[0].index != pointer.transcript_end_cursor - 1
            or transcript_window.records[0].message.role != MessageRole.ASSISTANT
        ):
            raise RuntimeError(
                "The durable model-step pointer does not identify its assistant message."
            )

        if pending_round is not None:
            if (
                pointer.tool_round_id != pending_round.tool_round_id
                or pending_round.source_model_step_id != pointer.logical_step_id
                or pending_round.source_transcript_cursor != pointer.source_transcript_cursor
            ):
                raise RuntimeError(
                    "The pending tool round conflicts with its durable source model step."
                )
        elif pointer.tool_round_id is not None:
            pending_pause = pending_approval if pending_approval is not None else pending_user_input
            assistant_message = None
            if pointer.assistant_message_deferred and pending_pause is not None:
                assistant_message = getattr(
                    pending_pause,
                    "quarantined_assistant_message",
                    None,
                )
            elif transcript_window.records:
                assistant_message = transcript_window.records[0].message
            assistant_call_parts = tuple(
                part
                for part in (() if assistant_message is None else assistant_message.content)
                if type(part) is ToolCallPart
            )
            assistant_calls = tuple(
                (part.tool_call_id, part.tool_name, part.arguments) for part in assistant_call_parts
            )
            if transcript_window.cursor == pointer.transcript_end_cursor:
                if pending_pause is None:
                    raise RuntimeError(
                        "The latest model completion requires a pending tool round, but its "
                        "durable marker is missing."
                    )
                if (
                    pending_pause.agent_name != session.agent_name
                    or pending_pause.environment_name != session.environment_name
                ):
                    raise RuntimeError(
                        "The pending tool pause conflicts with its durable source model step."
                    )
                pending_calls = tuple(
                    (call.tool_call_id, call.tool_name, call.arguments)
                    for call in pending_pause.tool_calls
                )
                pending_target = (
                    pending_pause.tool_call_id,
                    pending_pause.tool_name,
                    pending_pause.arguments,
                )
                if (
                    not (
                        pointer.assistant_message_published
                        or (
                            pointer.assistant_message_deferred
                            and getattr(
                                pending_pause,
                                "assistant_message_state",
                                None,
                            )
                            == "quarantined"
                        )
                    )
                    or not assistant_calls
                    or assistant_calls != pending_calls
                    or pending_calls.count(pending_target) != 1
                ):
                    raise RuntimeError(
                        "The pending tool pause conflicts with its durable source model step."
                    )
            else:
                if pending_pause is not None:
                    raise RuntimeError(
                        "A pending tool pause remains after its source model step advanced."
                    )
                next_record = (
                    transcript_window.records[1] if len(transcript_window.records) > 1 else None
                )
                tool_results = (
                    tuple(
                        (part.tool_call_id, part.tool_name)
                        for part in next_record.message.content
                        if type(part) is ToolResultPart
                    )
                    if next_record is not None
                    and next_record.index
                    == pointer.transcript_end_cursor + int(pointer.assistant_message_deferred)
                    and next_record.message.role == MessageRole.TOOL
                    else ()
                )
                expected_results = tuple(
                    (tool_call_id, tool_name)
                    for tool_call_id, tool_name, _arguments in assistant_calls
                )
                if (
                    not (pointer.assistant_message_published or pointer.assistant_message_deferred)
                    or not expected_results
                    or next_record is None
                    or len(tool_results) != len(next_record.message.content)
                    or tool_results != expected_results
                ):
                    raise RuntimeError(
                        "The transcript after the durable model step does not exactly close "
                        "its assistant tool calls."
                    )
                assistant_identities = {
                    (
                        part.model_step_id,
                        part.model_attempt_id,
                        part.tool_round_id,
                    )
                    for part in assistant_call_parts
                }
                if len(assistant_identities) != 1:
                    raise RuntimeError(
                        "The assistant tool calls do not share one complete execution identity."
                    )
                model_step_id, model_attempt_id, tool_round_id = next(iter(assistant_identities))
                if (
                    type(model_step_id) is not str
                    or type(model_attempt_id) is not str
                    or type(tool_round_id) is not str
                ):
                    raise RuntimeError(
                        "The assistant tool-call identity is incomplete at recovery."
                    )
                if (
                    model_step_id != pointer.logical_step_id
                    or tool_round_id != pointer.tool_round_id
                ):
                    raise RuntimeError(
                        "The assistant tool-call identity conflicts with its durable model step."
                    )
                tool_publication_id = f"tool-round:{pointer.tool_round_id}"
                tool_receipt = await self._session_store.load_runtime_publication_receipt(
                    session.id,
                    tool_publication_id,
                )
                if tool_receipt is None:
                    if not await self._tool_result_tail_has_durable_lifecycle_provenance(
                        session=session,
                        identity=ToolRoundIdentity(
                            model_step_id=model_step_id,
                            model_attempt_id=model_attempt_id,
                            tool_round_id=tool_round_id,
                        ),
                        assistant_call_parts=assistant_call_parts,
                        tool_result_message=next_record.message,
                        arguments_deferred=pointer.assistant_message_deferred,
                    ):
                        raise RuntimeError(
                            "The transcript contains tool results without durable tool-round "
                            "publication provenance."
                        )
                elif (
                    tool_receipt.publication_id != tool_publication_id
                    or tool_receipt.kind != "tool-round"
                    or tool_receipt.transcript_start_cursor != pointer.transcript_end_cursor
                    or tool_receipt.transcript_end_cursor
                    != pointer.transcript_end_cursor
                    + (2 if pointer.assistant_message_deferred else 1)
                    or tool_receipt.intent.get("round_id") != pointer.tool_round_id
                    or tool_receipt.intent.get("model_step_id") != model_step_id
                    or tool_receipt.intent.get("model_attempt_id") != model_attempt_id
                    or tool_receipt.intent.get("tool_round_id") != tool_round_id
                    or tool_receipt.intent.get("tool_call_ids")
                    != [tool_call_id for tool_call_id, _tool_name, _arguments in assistant_calls]
                ):
                    raise RuntimeError(
                        "The durable tool-round publication receipt conflicts with its "
                        "source model step."
                    )

        await self._event_writer.fan_out_persisted([completion_event])
        return ModelCompletionBoundaryReconciliation(
            state=("already_promoted" if state == "none" else state),
            session=session,
            pointer=pointer,
            completion_event=copy_event(completion_event),
            pending_tool_round=pending_round,
            transcript_cursor=transcript_window.cursor,
            recovery_events=recovery_events,
        )

    async def _tool_result_tail_has_durable_lifecycle_provenance(
        self,
        *,
        session: Session,
        identity: ToolRoundIdentity,
        assistant_call_parts: tuple[ToolCallPart, ...],
        tool_result_message: Message,
        arguments_deferred: bool,
    ) -> bool:
        """Verify pause-resume results published outside the ordinary round protocol.

        Approval and user-input continuations atomically append their grouped
        result while clearing their pause checkpoint. They do not create an
        ordinary tool-round receipt, so recovery reconstructs their exact
        transcript message from bounded, round-scoped terminal evidence.
        """

        pending_calls = [
            PendingToolCallApproval(
                tool_call_id=part.tool_call_id,
                tool_name=part.tool_name,
                arguments=part.arguments,
            )
            for part in assistant_call_parts
        ]
        lifecycle_records = await self._session_store.query_events(
            EventQuery(
                session_id=session.id,
                event_types=(
                    EventType.TOOL_CALL_STARTED,
                    *_MODEL_BOUNDARY_TOOL_TERMINAL_EVENT_TYPES,
                ),
                limit=RUNTIME_PUBLICATION_MAX_EVENT_BINDINGS + 1,
                order_by=EventOrder.SEQUENCE_DESC,
            )
        )
        resume_records = await self._session_store.query_events(
            EventQuery(
                session_id=session.id,
                event_type=EventType.SESSION_RESUMED,
                limit=RUNTIME_PUBLICATION_MAX_EVENT_BINDINGS + 1,
                order_by=EventOrder.SEQUENCE_DESC,
            )
        )
        interruption_records = await self._session_store.query_events(
            EventQuery(
                session_id=session.id,
                event_type=EventType.SESSION_INTERRUPTED,
                limit=RUNTIME_PUBLICATION_MAX_EVENT_BINDINGS + 1,
                order_by=EventOrder.SEQUENCE_DESC,
            )
        )
        expected_call_ids = {call.tool_call_id for call in pending_calls}
        lifecycle_evidence = _receiptless_exact_execution_evidence(
            lifecycle_records,
            identity=identity,
            expected_call_ids=expected_call_ids,
        )
        resume_evidence = _receiptless_exact_execution_evidence(
            resume_records,
            identity=identity,
            expected_call_ids=expected_call_ids,
        )
        interruption_evidence = _receiptless_exact_execution_evidence(
            interruption_records,
            identity=identity,
            expected_call_ids=expected_call_ids,
        )
        if len(lifecycle_evidence) > RUNTIME_PUBLICATION_MAX_EVENT_BINDINGS:
            raise RuntimeError("Durable tool lifecycle evidence exceeds the recovery bound.")
        if len(resume_evidence) > RUNTIME_PUBLICATION_MAX_EVENT_BINDINGS:
            raise RuntimeError("Durable pause-resume evidence exceeds the recovery bound.")
        if len(interruption_evidence) > RUNTIME_PUBLICATION_MAX_EVENT_BINDINGS:
            raise RuntimeError("Durable pause-origin evidence exceeds the recovery bound.")
        if not resume_evidence:
            return False

        pending_by_call_id = {call.tool_call_id: call for call in pending_calls}
        if len(pending_by_call_id) != len(pending_calls):
            raise RuntimeError("The assistant tool-call tail contains duplicate call identities.")

        pause_kind: Literal["approval", "user-input"] | None = None
        pause_id: str | None = None
        resume_tool_call_id: str | None = None
        approval_decision: str | None = None
        for _sequence, event in resume_evidence:
            if (
                event.session_id != session.id
                or event.agent_name != session.agent_name
                or event.environment_name != session.environment_name
                or not identity.matches_payload(event.payload)
            ):
                raise RuntimeError(
                    "Durable pause-continuation anchor conflicts with its source model step."
                )
            event_pause_kind, event_pause_id = _receiptless_pause_event_identity(event)
            if pause_kind is None:
                pause_kind = event_pause_kind
                pause_id = event_pause_id
            elif pause_kind != event_pause_kind or pause_id != event_pause_id:
                raise RuntimeError(
                    "Durable pause-continuation evidence has conflicting pause identity."
                )
            tool_call_id = event.payload.get("tool_call_id")
            if type(tool_call_id) is not str:
                raise RuntimeError(
                    "Durable pause-continuation anchor has no valid tool-call identity."
                )
            if tool_call_id not in pending_by_call_id:
                raise RuntimeError(
                    "Durable pause-continuation anchor conflicts with its assistant tool call."
                )
            if event.tool_name is not None:
                raise RuntimeError(
                    "Durable pause-continuation anchor contains an unexpected tool name."
                )
            if resume_tool_call_id is None:
                resume_tool_call_id = tool_call_id
            elif resume_tool_call_id != tool_call_id:
                raise RuntimeError(
                    "Durable pause-continuation anchors identify different tool calls."
                )
            if event_pause_kind == "approval":
                decision = event.payload.get("decision")
                if decision not in {
                    ToolApprovalDecision.APPROVE.value,
                    ToolApprovalDecision.DENY.value,
                }:
                    raise RuntimeError("Durable approval continuation has no valid decision.")
                if approval_decision is None:
                    approval_decision = decision
                elif approval_decision != decision:
                    raise RuntimeError(
                        "Durable approval continuations contain conflicting decisions."
                    )
            elif event.payload.get("interruption_type") != _INTERRUPTION_TYPE_USER_INPUT_REQUIRED:
                raise RuntimeError(
                    "Durable user-input continuation has the wrong interruption type."
                )

        if pause_kind is None or pause_id is None or resume_tool_call_id is None:
            raise RuntimeError("Durable pause-continuation anchor has no pause identity.")

        expected_interruption_type = (
            _INTERRUPTION_TYPE_TOOL_APPROVAL_REQUIRED
            if pause_kind == "approval"
            else _INTERRUPTION_TYPE_USER_INPUT_REQUIRED
        )
        expected_origin_field = "approval" if pause_kind == "approval" else "user_input"
        expected_calls = [
            {
                "tool_call_id": call.tool_call_id,
                "tool_name": call.tool_name,
                "arguments": call.arguments,
            }
            for call in pending_calls
        ]
        origin_sequences: list[int] = []
        for sequence, event in interruption_evidence:
            if event.payload.get("interruption_type") != expected_interruption_type:
                continue
            if (
                event.session_id != session.id
                or event.agent_name != session.agent_name
                or event.environment_name != session.environment_name
                or not identity.matches_payload(event.payload)
            ):
                raise RuntimeError(
                    "Durable pause-origin evidence conflicts with its source model step."
                )
            try:
                pause_checkpoint, origin_arguments_quarantined = (
                    tool_argument_publication.pause_checkpoint_validation_view(
                        event.payload.get(expected_origin_field),
                        pause_kind=pause_kind,
                    )
                )
                if origin_arguments_quarantined:
                    projected_calls = pause_checkpoint.get("tool_calls")
                    if type(projected_calls) is not list or len(projected_calls) != len(
                        pending_calls
                    ):
                        raise ValueError("Projected pause tool calls conflict with the round.")
                    for projected_call, pending_call in zip(
                        projected_calls,
                        pending_calls,
                        strict=True,
                    ):
                        if type(projected_call) is not dict:
                            raise ValueError("Projected pause tool calls conflict with the round.")
                        typed_projected_call = cast("dict[str, Any]", projected_call)
                        if typed_projected_call.get("tool_name") != pending_call.tool_name:
                            raise ValueError("Projected pause tool calls conflict with the round.")
                        typed_projected_call["tool_call_id"] = pending_call.tool_call_id
                    pause_checkpoint.update(
                        {
                            "tool_round_id": identity.tool_round_id,
                            "model_step_id": identity.model_step_id,
                            "model_attempt_id": identity.model_attempt_id,
                            "tool_call_id": resume_tool_call_id,
                            ("approval_id" if pause_kind == "approval" else "input_id"): pause_id,
                        }
                    )
                    assistant_publication = pause_checkpoint.get("assistant_publication")
                    if type(assistant_publication) is dict:
                        safe_assistant_message = assistant_publication.get("message")
                        if type(safe_assistant_message) is not dict:
                            raise ValueError(
                                "Projected pause has no safe assistant publication evidence."
                            )
                        pause_checkpoint["assistant_message_state"] = "quarantined"
                        pause_checkpoint["quarantined_assistant_message"] = safe_assistant_message
                if pause_kind == "approval":
                    pending_pause = PendingToolApproval.model_validate(pause_checkpoint)
                    origin_pause_id = pending_pause.approval_id
                else:
                    pending_pause = PendingUserInput.model_validate(pause_checkpoint)
                    origin_pause_id = pending_pause.input_id
            except (TypeError, ValueError):
                raise RuntimeError(
                    "Durable pause-origin evidence contains an invalid pause checkpoint."
                ) from None
            origin_identity = ToolRoundIdentity(
                model_step_id=pending_pause.model_step_id,
                model_attempt_id=pending_pause.model_attempt_id,
                tool_round_id=pending_pause.tool_round_id,
            )
            origin_calls = [
                {
                    "tool_call_id": call.tool_call_id,
                    "tool_name": call.tool_name,
                    "arguments": call.arguments,
                }
                for call in pending_pause.tool_calls
            ]
            comparable_origin_calls = origin_calls
            comparable_expected_calls = expected_calls
            if arguments_deferred or origin_arguments_quarantined:
                comparable_origin_calls = [
                    {
                        "tool_call_id": call["tool_call_id"],
                        "tool_name": call["tool_name"],
                    }
                    for call in origin_calls
                ]
                comparable_expected_calls = [
                    {
                        "tool_call_id": call["tool_call_id"],
                        "tool_name": call["tool_name"],
                    }
                    for call in expected_calls
                ]
            if (
                origin_pause_id != pause_id
                or origin_identity != identity
                or pending_pause.agent_name != session.agent_name
                or pending_pause.environment_name != session.environment_name
                or pending_pause.tool_call_id != resume_tool_call_id
                or canonical_durable_json_bytes(
                    comparable_origin_calls,
                    "pause_origin_tool_calls",
                )
                != canonical_durable_json_bytes(
                    comparable_expected_calls,
                    "assistant_tool_calls",
                )
            ):
                raise RuntimeError("Durable pause-origin evidence conflicts with its continuation.")
            origin_sequences.append(sequence)

        if not origin_sequences:
            raise RuntimeError("Receipt-less tool evidence has no authoritative pause origin.")
        if min(origin_sequences) >= min(sequence for sequence, _event in resume_evidence):
            raise RuntimeError("Durable pause-continuation evidence precedes its pause origin.")
        if lifecycle_evidence and min(sequence for sequence, _event in lifecycle_evidence) <= min(
            sequence for sequence, _event in resume_evidence
        ):
            raise RuntimeError("Durable tool lifecycle evidence precedes its pause continuation.")

        started_by_call_id: dict[str, tuple[int, Event]] = {}
        terminal_by_call_id: dict[str, tuple[int, Event]] = {}
        for sequence, event in lifecycle_evidence:
            if (
                event.session_id != session.id
                or event.agent_name != session.agent_name
                or event.environment_name != session.environment_name
                or not identity.matches_payload(event.payload)
            ):
                raise RuntimeError(
                    "Durable tool lifecycle evidence conflicts with its source model step."
                )
            event_pause_kind, event_pause_id = _receiptless_pause_event_identity(event)
            if pause_kind != event_pause_kind or pause_id != event_pause_id:
                raise RuntimeError(
                    "Durable pause-continuation evidence has conflicting pause identity."
                )
            tool_call_id = event.payload.get("tool_call_id")
            if type(tool_call_id) is not str:
                raise RuntimeError(
                    "Durable tool lifecycle evidence has no valid tool-call identity."
                )
            pending_call = pending_by_call_id.get(tool_call_id)
            if pending_call is None:
                raise RuntimeError(
                    "Durable tool lifecycle evidence conflicts with its assistant tool call."
                )
            if event.tool_name != pending_call.tool_name:
                raise RuntimeError(
                    "Durable tool lifecycle evidence conflicts with its assistant tool call."
                )
            expected_idempotency_key = tool_execution.tool_idempotency_key(
                session_id=session.id,
                tool_round_id=identity.tool_round_id,
                tool_call_id=tool_call_id,
                approval_id=pause_id if event_pause_kind == "approval" else None,
                pause_id=pause_id if event_pause_kind == "user-input" else None,
            )
            if event.payload.get("idempotency_key") != expected_idempotency_key:
                raise RuntimeError(
                    "Durable pause-continuation evidence has a conflicting idempotency key."
                )
            if event.type == EventType.TOOL_CALL_STARTED:
                if tool_call_id in started_by_call_id:
                    raise RuntimeError(
                        "Durable pause-continuation evidence contains duplicate started events."
                    )
                if not tool_argument_publication.started_arguments_match_private_call(
                    event.payload,
                    private_arguments=pending_call.arguments,
                ):
                    raise RuntimeError(
                        "Durable pause-continuation started arguments conflict with "
                        "the assistant tool call."
                    )
                started_by_call_id[tool_call_id] = (sequence, event)
                continue
            if event.type not in _MODEL_BOUNDARY_TOOL_TERMINAL_EVENT_TYPES:
                continue
            if tool_call_id in terminal_by_call_id:
                raise RuntimeError(
                    "Durable tool lifecycle evidence contains duplicate terminal results."
                )
            terminal_by_call_id[tool_call_id] = (sequence, event)

        if set(terminal_by_call_id) != set(pending_by_call_id):
            return False

        outcomes: list[runtime_records.ToolCallOutcome] = []
        for pending_call in pending_calls:
            terminal_sequence, terminal = terminal_by_call_id[pending_call.tool_call_id]
            started = started_by_call_id.get(pending_call.tool_call_id)
            if (
                terminal.type
                in {
                    EventType.TOOL_CALL_COMPLETED,
                    EventType.TOOL_CALL_FAILED,
                }
                and started is None
                and not (
                    terminal.type is EventType.TOOL_CALL_FAILED
                    and terminal.payload.get("registration_state") == "unregistered_at_policy_plan"
                )
            ):
                raise RuntimeError("Durable executed tool evidence has no preceding started event.")
            if (
                terminal.payload.get("registration_state") == "unregistered_at_policy_plan"
                and started is not None
            ):
                raise RuntimeError(
                    "Durable unregistered-at-plan evidence contains a started event."
                )
            if terminal.type == EventType.TOOL_CALL_APPROVAL_DENIED and started is not None:
                raise RuntimeError(
                    "Durable approval-denied tool evidence contains a started event."
                )
            if started is not None and started[0] >= terminal_sequence:
                raise RuntimeError(
                    "Durable pause-continuation terminal evidence precedes its start."
                )
            if (
                pause_kind == "approval"
                and approval_decision == ToolApprovalDecision.DENY.value
                and terminal.type
                not in {
                    EventType.TOOL_CALL_BLOCKED,
                    EventType.TOOL_CALL_APPROVAL_DENIED,
                }
            ):
                raise RuntimeError(
                    "Durable denied approval evidence contains an executed tool result."
                )
            if (
                pause_kind == "approval"
                and approval_decision == ToolApprovalDecision.DENY.value
                and started is not None
            ):
                raise RuntimeError("Durable denied approval evidence contains a started event.")
            if (
                (
                    pause_kind == "approval"
                    and approval_decision == ToolApprovalDecision.APPROVE.value
                )
                or pause_kind == "user-input"
            ) and terminal.type == EventType.TOOL_CALL_APPROVAL_DENIED:
                raise RuntimeError(
                    "Durable pause-continuation decision conflicts with its terminal result."
                )
            outcome = resume_ledger.tool_call_outcome_from_terminal_event(
                event=terminal,
                pending_tool_call=pending_call,
            )
            completed = terminal.type == EventType.TOOL_CALL_COMPLETED
            if completed == outcome.result.is_error:
                raise RuntimeError(
                    "Durable terminal tool evidence conflicts with its result status."
                )
            outcomes.append(outcome)

        expected_messages = transcript_helpers.tool_result_messages(
            outcomes,
            tool_round_identity=identity,
        )
        if canonical_durable_json_bytes(
            [message.model_dump(mode="json") for message in expected_messages],
            "expected_tool_result_messages",
        ) != canonical_durable_json_bytes(
            [tool_result_message.model_dump(mode="json")],
            "tool_result_messages",
        ):
            raise RuntimeError(
                "The tool-result transcript conflicts with its durable terminal evidence."
            )
        return True

    @staticmethod
    def _validate_active_model_completion_stage(session: Session, stage) -> None:
        if stage.session_id != session.id:
            raise RuntimeError("The active model-completion stage belongs to another session.")
        if stage.source_run_epoch > session.run_epoch:
            raise RuntimeError(
                "The active model-completion stage was prepared by a future run epoch."
            )

    async def _promote_completed_model_stage(
        self,
        *,
        session: Session,
        stage_id: str,
    ) -> Session:
        async def commit_once():
            return await self._session_store.promote_model_completion_stage(
                session.id,
                stage_id=stage_id,
                expected_run_epoch=session.run_epoch,
            )

        async def commit():
            try:
                return await commit_once()
            except Exception as first_error:
                try:
                    return await commit_once()
                except Exception as replay_error:
                    replay_error.add_note(
                        "Exact recovered model-completion promotion also failed after "
                        f"{type(first_error).__name__}: {first_error}"
                    )
                    raise replay_error from first_error

        task = asyncio.create_task(commit())
        outcome = await await_shielded_task_outcome(task)
        cancellation = outcome.cancellation
        error = outcome.error
        if isinstance(error, asyncio.CancelledError) and cancellation is None:
            error = unexpected_child_cancellation_error(
                error,
                operation="Recovered model-completion promotion",
            )
        if error is not None:
            if cancellation is not None:
                cancellation.add_note(
                    "Recovered model-completion promotion also failed: "
                    f"{type(error).__name__}: {error}"
                )
                raise cancellation from error
            raise error
        if outcome.result is None:
            result_error = RuntimeError(
                "Recovered model-completion promotion returned no acknowledgement."
            )
            if cancellation is not None:
                cancellation.add_note(str(result_error))
                raise cancellation from result_error
            raise result_error
        if cancellation is not None:
            raise cancellation
        return outcome.result.session

    async def _cleanup_recovery_handoff(
        self,
        *,
        stream: AsyncGenerator[Event, None] | None,
        session_id: str,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_environment: runtime_records.RegisteredEnvironment | None,
        authoritative_failure: BaseException | None,
        finalize_abandoned: bool,
        release_run_fence: bool,
        abort_environment_setup: bool = True,
        execution_profile: ExecutionProfileIdentity | None = None,
    ) -> None:
        cleanup_steps: list[tuple[str, RecoveryCleanup]] = []
        if stream is not None:
            cleanup_steps.append(("nested stream close", stream.aclose))
        if finalize_abandoned:
            cleanup_steps.append(
                (
                    "abandoned session finalization",
                    lambda: self.finalize_abandoned_session_by_id(
                        session_id,
                        registered_agent=registered_agent,
                        registered_environment=registered_environment,
                        execution_profile=execution_profile,
                    ),
                )
            )
        if abort_environment_setup and authoritative_failure is not None:
            cleanup_steps.append(
                (
                    "environment setup abort",
                    lambda: self._environment_lifecycle.abort_environment_setup(
                        session_id=session_id,
                        original_error=authoritative_failure,
                        execution_profile=execution_profile,
                    ),
                )
            )
        if release_run_fence:
            cleanup_steps.append(
                (
                    "run fence release",
                    lambda: self._environment_lifecycle.release_run_fence_after_environment_cleanup(
                        session_id=session_id,
                        execution_profile=execution_profile,
                    ),
                )
            )
        await _run_recovery_cleanup_steps(
            authoritative_failure=authoritative_failure,
            steps=tuple(cleanup_steps),
        )

    async def _cleanup_entrypoint_handoff(
        self,
        *,
        stream: AsyncGenerator[Event, None] | None,
        session_id: str,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_environment: runtime_records.RegisteredEnvironment | None,
        authoritative_failure: BaseException | None,
        finalize_abandoned: bool,
        release_run_fence: bool,
        abort_environment_setup: bool = True,
        execution_profile: ExecutionProfileIdentity | None = None,
    ) -> None:
        try:
            await self._cleanup_recovery_handoff(
                stream=stream,
                session_id=session_id,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
                authoritative_failure=authoritative_failure,
                finalize_abandoned=finalize_abandoned,
                release_run_fence=release_run_fence,
                abort_environment_setup=abort_environment_setup,
                execution_profile=execution_profile,
            )
        finally:
            # Cleanup can run in a shielded child task with copied context. The
            # public caller must never retain a stale run epoch after handoff.
            _deactivate_session_run_fence(session_id)
            _deactivate_session_interaction(session_id)

    async def _activate_latest_open_interaction(self, session_id: str) -> str | None:
        records = await self._session_store.query_events(
            EventQuery(
                session_id=session_id,
                event_types=INTERACTION_LIFECYCLE_EVENT_TYPES,
                order_by=EventOrder.SEQUENCE_DESC,
                limit=1,
            )
        )
        if not records or records[0].event.type in INTERACTION_TERMINAL_EVENT_TYPES:
            return None
        interaction_id = records[0].event.interaction_id
        if interaction_id is None:
            raise RuntimeError("Interaction lifecycle event has no interaction identity.")
        _activate_session_interaction(session_id, interaction_id)
        return interaction_id

    async def _transition_recovery_session_to_running(
        self,
        loaded_session: Session,
        *,
        checkpoint: dict[str, Any] | None,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_environment: runtime_records.RegisteredEnvironment | None,
        from_statuses: set[SessionStatus] | None = None,
        checkpoint_transform: CheckpointTransform | None = None,
        execution_profile_snapshot: ActiveInvocationExecutionProfile | None = None,
        preserve_open_interaction_on_failure: bool = False,
        before_mutation: RecoveryMutationHook | None = None,
    ) -> tuple[Session, Event | None]:
        """Claim a paused session without leaving cancellation outcome-uncertain.

        Session stores activate run fences in task-local context after the durable
        transition commits. Keeping the transition in a shielded child task lets it
        reach a definite result even if the caller is cancelled at that boundary.
        A successful claim is then activated in the caller's context. If cancellation
        arrived, the claim is finalized and released before that cancellation is
        propagated.
        """
        if type(preserve_open_interaction_on_failure) is not bool:
            raise TypeError("preserve_open_interaction_on_failure must be a bool.")
        expected_statuses = (
            {SessionStatus.INTERRUPTED} if from_statuses is None else set(from_statuses)
        )
        if before_mutation is not None:
            preflight_events = await self._session_store.query_events(
                EventQuery(
                    session_id=loaded_session.id,
                    event_types=INTERACTION_LIFECYCLE_EVENT_TYPES,
                    order_by=EventOrder.SEQUENCE_DESC,
                    limit=1,
                )
            )
            if (
                not preflight_events
                or preflight_events[0].event.type in INTERACTION_TERMINAL_EVENT_TYPES
            ):
                raise RuntimeError(
                    "Pending recovery state has no open interaction. "
                    "Pre-interaction prerelease recovery state is unsupported."
                )
            if preflight_events[0].event.interaction_id is None:
                raise RuntimeError("Interaction lifecycle event has no interaction identity.")
            await before_mutation()
        if loaded_session.status in expected_statuses:
            (
                loaded_session,
                checkpoint,
            ) = await self._reconcile_terminal_evidence_before_continuation(
                session=loaded_session,
                checkpoint=checkpoint,
            )
            if execution_profile_snapshot is not None:
                reconciled_profile = active_invocation_execution_profile_from_checkpoint(checkpoint)
                if (
                    reconciled_profile is None
                    or reconciled_profile.session_id != execution_profile_snapshot.session_id
                    or reconciled_profile.interaction_id
                    != execution_profile_snapshot.interaction_id
                    or reconciled_profile.profile != execution_profile_snapshot.profile
                ):
                    raise RuntimeError(
                        "Active invocation profile changed during terminal-evidence recovery."
                    )
                execution_profile_snapshot = execution_profile_snapshot.model_copy(
                    update={"run_epoch": reconciled_profile.run_epoch}
                )
        session_id = loaded_session.id
        latest_events = await self._session_store.query_events(
            EventQuery(
                session_id=session_id,
                event_types=INTERACTION_LIFECYCLE_EVENT_TYPES,
                order_by=EventOrder.SEQUENCE_DESC,
                limit=1,
            )
        )
        if not latest_events or latest_events[0].event.type in INTERACTION_TERMINAL_EVENT_TYPES:
            raise RuntimeError(
                "Pending recovery state has no open interaction. "
                "Pre-interaction prerelease recovery state is unsupported."
            )
        interaction_id = latest_events[0].event.interaction_id
        if interaction_id is None:
            raise RuntimeError("Interaction lifecycle event has no interaction identity.")
        if (
            execution_profile_snapshot is not None
            and execution_profile_snapshot.interaction_id != interaction_id
        ):
            raise RuntimeError(
                "Active invocation execution profile belongs to another interaction."
            )
        run_operation_id = str(uuid4())

        def reject_active_incomplete_recovery(
            current_session: Session,
            checkpoint: dict[str, Any] | None,
        ) -> dict[str, Any] | None:
            checkpoint = _checkpoint_without_active_incomplete_recovery_claim(
                checkpoint,
                now=self._clock(),
            )
            if checkpoint_transform is not None:
                checkpoint = checkpoint_transform(current_session, checkpoint)
            if execution_profile_snapshot is not None:
                current_profile = active_invocation_execution_profile_from_checkpoint(checkpoint)
                if current_profile is None or current_profile != execution_profile_snapshot:
                    raise RuntimeError(
                        "Active invocation profile changed before continuation claimed it."
                    )
                if not active_invocation_execution_profile_is_released(
                    current_profile,
                    session_id=current_session.id,
                    run_epoch=current_session.run_epoch,
                ):
                    raise SessionRunFenced(
                        "The previous invocation still owns terminal hooks or trailing cleanup."
                    )
                checkpoint = checkpoint_with_active_invocation_execution_profile(
                    checkpoint,
                    session_id=current_session.id,
                    interaction_id=interaction_id,
                    run_epoch=current_session.run_epoch + 1,
                    profile=execution_profile_snapshot.profile,
                    expected=execution_profile_snapshot,
                )
            return _checkpoint_with_session_run_operation(
                checkpoint=checkpoint,
                current_session=current_session,
                operation_id=run_operation_id,
            )

        transition_task = asyncio.create_task(
            self._session_store.transition_status_and_checkpoint(
                session_id,
                from_statuses=expected_statuses,
                to_status=SessionStatus.RUNNING,
                checkpoint_transform=reject_active_incomplete_recovery,
            )
        )
        cancellation: asyncio.CancelledError | None = None
        transition_failure: BaseException | None = None
        while not transition_task.done():
            try:
                await asyncio.shield(transition_task)
            except asyncio.CancelledError as exc:
                if transition_task.cancelled():
                    transition_failure = exc
                    break
                if cancellation is None:
                    cancellation = exc
            except BaseException as exc:
                transition_failure = exc
                break

        session: Session | None = None
        if transition_failure is None:
            try:
                session = transition_task.result()
            except BaseException as exc:
                transition_failure = exc

        if session is None:
            if cancellation is not None:
                if transition_failure is not None:
                    cancellation.add_note(
                        "Continuation recovery transition also failed after cancellation: "
                        f"{type(transition_failure).__name__}."
                    )
                raise cancellation from transition_failure
            if transition_failure is None:
                raise RuntimeError("Continuation recovery transition completed without a session.")
            raise transition_failure

        # The transition activated this epoch only in the child task's copied
        # context. The caller owns all subsequent writes and cleanup.
        _activate_session_run_fence(session)
        _activate_session_interaction(session.id, interaction_id)
        try:
            resumed_event = await self._resume_interaction(
                session,
                registered_agent,
                registered_environment,
            )
        except BaseException as exc:
            try:
                cleanup_steps: list[tuple[str, RecoveryCleanup]] = []
                if not preserve_open_interaction_on_failure:
                    cleanup_steps.append(
                        (
                            "abandoned session finalization",
                            lambda: self.finalize_abandoned_session_by_id(
                                session.id,
                                registered_agent=registered_agent,
                                registered_environment=registered_environment,
                                execution_profile=(
                                    None
                                    if execution_profile_snapshot is None
                                    else execution_profile_snapshot.profile
                                ),
                            ),
                        )
                    )
                cleanup_steps.append(
                    (
                        "run fence release",
                        lambda: (
                            self._environment_lifecycle.release_run_fence_after_environment_cleanup(
                                session_id=session.id,
                                execution_profile=(
                                    None
                                    if execution_profile_snapshot is None
                                    else execution_profile_snapshot.profile
                                ),
                            )
                        ),
                    )
                )
                await _run_recovery_cleanup_steps(
                    authoritative_failure=exc,
                    steps=tuple(cleanup_steps),
                )
            finally:
                _deactivate_session_run_fence(session.id)
                _deactivate_session_interaction(session.id)
            raise
        if cancellation is None:
            return session, resumed_event

        try:
            cleanup_steps = []
            if not preserve_open_interaction_on_failure:
                cleanup_steps.append(
                    (
                        "abandoned session finalization",
                        lambda: self.finalize_abandoned_session_by_id(
                            session.id,
                            registered_agent=registered_agent,
                            registered_environment=registered_environment,
                            execution_profile=(
                                None
                                if execution_profile_snapshot is None
                                else execution_profile_snapshot.profile
                            ),
                        ),
                    )
                )
            cleanup_steps.append(
                (
                    "run fence release",
                    lambda: self._environment_lifecycle.release_run_fence_after_environment_cleanup(
                        session_id=session.id,
                        execution_profile=(
                            None
                            if execution_profile_snapshot is None
                            else execution_profile_snapshot.profile
                        ),
                    ),
                )
            )
            await _run_recovery_cleanup_steps(
                authoritative_failure=cancellation,
                steps=tuple(cleanup_steps),
            )
        finally:
            # Shielded cleanup runs in a copied context. Never leave the caller's
            # task-local epoch active if it catches and handles the cancellation.
            _deactivate_session_run_fence(session.id)
            _deactivate_session_interaction(session.id)
        raise cancellation

    async def resolve_user_input(
        self,
        response: UserInputResponse,
        *,
        before_mutation: RecoveryMutationHook | None = None,
    ) -> AsyncGenerator[Event, None]:
        """Resume a session paused by ``ask_user`` with the user's answer.

        The answer becomes the ``ask_user`` tool result; any other tool calls in the same
        round (none ran before the pause) execute now, and the session continues.
        """
        loaded_session = await self._session_store.load(response.session_id)
        if loaded_session is None:
            raise KeyError(f"Session not found: {response.session_id}")

        checkpoint = await self._session_store.load_checkpoint(loaded_session.id)
        pending = pending_user_input_from_checkpoint(
            checkpoint,
            redactor=self._secret_redactor,
            consume_on_rejection=True,
        )
        if pending is None:
            raise RuntimeError("Session has no pending user input.")
        if pending.input_id != response.input_id:
            raise ValueError(f"User input id does not match pending input: {response.input_id}")
        # The output-schema contract is fixed by the paused run's provider history; a resolver
        # cannot swap it (a spec matching or absent is fine; a differing one is rejected). Checked
        # before the status transition so it surfaces to the caller rather than being caught by the
        # resume's failure handler. All provider-dispatch semantics are then
        # compared with the frozen invocation profile before status changes.
        effective_structured_output = _effective_user_input_structured_output(
            structured_output=response.structured_output,
            pending=pending,
        )
        invocation_semantics = _effective_user_input_invocation_semantics(
            response=response,
            pending=pending,
            structured_output=effective_structured_output,
            effective_retry_policy=self._effective_retry_policy,
        )
        require_secret_free_structured_output_spec(
            effective_structured_output,
            redactor=self._secret_redactor,
            field_name="UserInputResponse.structured_output",
        )

        registered_agent = self._resolve_registered_agent(loaded_session.agent_name)
        registered_provider = self._resolve_registered_provider(loaded_session.provider_name)
        budget_policy_snapshot = copy_budget_policy(self._resolve_budget_policy())
        execution_profile_snapshot = await self._validate_execution_profile_continuation(
            loaded_session,
            checkpoint,
            registered_agent,
            registered_provider,
            response.loop_policies,
            budget_policy=budget_policy_snapshot,
            request_budget_limits=invocation_semantics.budget_limits,
            structured_output=invocation_semantics.structured_output,
            thinking=invocation_semantics.thinking,
            max_steps=invocation_semantics.max_steps,
            limits=invocation_semantics.limits,
            retry_policy=invocation_semantics.retry_policy,
            invocation_semantics_available=True,
        )
        _require_native_structured_output_support(
            effective_structured_output, registered_provider=registered_provider
        )
        registered_environment = self._resolve_registered_environment(
            loaded_session.environment_name
        )
        invocation_secrets.require_continuation_secret_resolution_compatibility(
            (
                "unknown"
                if pending.assistant_publication is None
                else pending.assistant_publication.secret_resolution_scope
            ),
            registered_environment,
        )
        session, resumed_event = await self._transition_recovery_session_to_running(
            loaded_session,
            checkpoint=checkpoint,
            registered_agent=registered_agent,
            registered_environment=registered_environment,
            execution_profile_snapshot=execution_profile_snapshot,
            before_mutation=before_mutation,
        )
        if resumed_event is not None:
            yield resumed_event

        continuation_stream = self.continue_user_input_resolution(
            response=response,
            session=session,
            pending=pending,
            registered_agent=registered_agent,
            registered_provider=registered_provider,
            registered_environment=registered_environment,
            execution_profile_snapshot=execution_profile_snapshot,
            budget_policy=budget_policy_snapshot,
        )
        authoritative_failure: BaseException | None = None
        abandoned = False
        try:
            async for event in continuation_stream:
                yield event
        except BaseException as exc:
            authoritative_failure = exc
            abandoned = _recovery_abandonment_signal(exc) is not None
            raise
        finally:
            await self._cleanup_entrypoint_handoff(
                stream=continuation_stream,
                session_id=session.id,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
                authoritative_failure=authoritative_failure,
                finalize_abandoned=abandoned,
                release_run_fence=True,
                execution_profile=execution_profile_snapshot.profile,
            )

    async def recover_user_input_request(
        self,
        request: UserInputRecoveryRequest,
        *,
        before_mutation: RecoveryMutationHook | None = None,
    ) -> AsyncGenerator[Event, None]:
        """Recover a user-input round stuck on `manual_recovery_required`.

        A tool in the paused round started on a prior resume but recorded no terminal event
        (a crash mid-tool), so it cannot be re-run automatically. The caller supplies the
        externally verified outcome for that `tool_call_id`; Cayu persists it as the tool's
        terminal result and continues the round (re-supplying `answer` in case the `ask_user`
        result was not recorded before the crash). Cayu does not infer the outcome itself.
        """
        loaded_session = await self._session_store.load(request.session_id)
        if loaded_session is None:
            raise KeyError(f"Session not found: {request.session_id}")

        checkpoint = await self._session_store.load_checkpoint(loaded_session.id)
        pending = pending_user_input_from_checkpoint(
            checkpoint,
            redactor=self._secret_redactor,
            consume_on_rejection=True,
        )
        if pending is None:
            raise RuntimeError("Session has no pending user input.")
        if pending.input_id != request.input_id:
            raise ValueError(f"User input id does not match pending input: {request.input_id}")
        effective_structured_output = _effective_user_input_structured_output(
            structured_output=request.structured_output,
            pending=pending,
        )
        invocation_semantics = _effective_user_input_invocation_semantics(
            response=request,
            pending=pending,
            structured_output=effective_structured_output,
            effective_retry_policy=self._effective_retry_policy,
        )
        require_secret_free_structured_output_spec(
            effective_structured_output,
            redactor=self._secret_redactor,
            field_name="UserInputRecoveryRequest.structured_output",
        )

        pending_tool_call = approval_support.round_tool_call_for_recovery(
            pending_calls=pending.tool_calls,
            tool_call_id=request.tool_call_id,
        )
        registered_agent = self._resolve_registered_agent(loaded_session.agent_name)
        registered_provider = self._resolve_registered_provider(loaded_session.provider_name)
        budget_policy_snapshot = copy_budget_policy(self._resolve_budget_policy())
        execution_profile_snapshot = await self._validate_execution_profile_continuation(
            loaded_session,
            checkpoint,
            registered_agent,
            registered_provider,
            request.loop_policies,
            budget_policy=budget_policy_snapshot,
            request_budget_limits=invocation_semantics.budget_limits,
            structured_output=invocation_semantics.structured_output,
            thinking=invocation_semantics.thinking,
            max_steps=invocation_semantics.max_steps,
            limits=invocation_semantics.limits,
            retry_policy=invocation_semantics.retry_policy,
            invocation_semantics_available=True,
        )
        _require_native_structured_output_support(
            effective_structured_output, registered_provider=registered_provider
        )
        registered_environment = self._resolve_registered_environment(
            loaded_session.environment_name
        )
        invocation_secrets.require_continuation_secret_resolution_compatibility(
            (
                "unknown"
                if pending.assistant_publication is None
                else pending.assistant_publication.secret_resolution_scope
            ),
            registered_environment,
        )
        session, resumed_event = await self._transition_recovery_session_to_running(
            loaded_session,
            checkpoint=checkpoint,
            registered_agent=registered_agent,
            registered_environment=registered_environment,
            execution_profile_snapshot=execution_profile_snapshot,
            before_mutation=before_mutation,
        )
        if resumed_event is not None:
            yield resumed_event
        recovery_stream = self.recover_user_input(
            request=request,
            loaded_session=loaded_session,
            session=session,
            pending=pending,
            pending_tool_call=pending_tool_call,
            registered_agent=registered_agent,
            registered_provider=registered_provider,
            registered_environment=registered_environment,
            execution_profile_snapshot=execution_profile_snapshot,
            budget_policy=budget_policy_snapshot,
        )
        authoritative_failure: BaseException | None = None
        try:
            async for event in recovery_stream:
                yield event
        except BaseException as exc:
            authoritative_failure = exc
            raise
        finally:
            await self._cleanup_entrypoint_handoff(
                stream=recovery_stream,
                session_id=session.id,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
                authoritative_failure=authoritative_failure,
                finalize_abandoned=False,
                release_run_fence=False,
                execution_profile=execution_profile_snapshot.profile,
            )

    async def resolve_tool_approval(
        self,
        request: ToolApprovalRequest,
        *,
        before_mutation: RecoveryMutationHook | None = None,
    ) -> AsyncGenerator[Event, None]:
        loaded_session = await self._session_store.load(request.session_id)
        if loaded_session is None:
            raise KeyError(f"Session not found: {request.session_id}")

        close_receipt = await self._session_store.load_runtime_publication_receipt(
            loaded_session.id,
            f"approval-close:{request.approval_id}",
        )
        if close_receipt is not None:
            expected_identity = {
                "approval_id": request.approval_id,
                "tool_call_id": request.tool_call_id,
                "tool_round_id": request.tool_round_id,
                "requested_decision": request.decision.value,
                "resolution_request_digest": (
                    approval_support.approval_resolution_request_digest(request)
                ),
            }
            if close_receipt.kind != "approval-close" or any(
                close_receipt.intent.get(key) != value for key, value in expected_identity.items()
            ):
                raise RuntimeError(
                    "Tool approval was already closed with a conflicting identity or decision."
                )
            if len(close_receipt.appended_event_ids) != 1:
                raise SessionRuntimePublicationConflict(
                    "Tool approval closure receipt has invalid event evidence."
                )
            closure_records = await self._session_store.query_events(
                EventQuery(
                    session_id=loaded_session.id,
                    event_id=close_receipt.appended_event_ids[0],
                    limit=2,
                )
            )
            if (
                len(closure_records) != 1
                or closure_records[0].event.id != close_receipt.appended_event_ids[0]
            ):
                raise SessionRuntimePublicationConflict(
                    "Tool approval closure event is missing from durable history."
                )
            if before_mutation is not None:
                await before_mutation()
            await self.materialize_deferred_input_for_receipt(close_receipt)
            closure_event = closure_records[0].event
            yield closure_event
            return

        checkpoint = await self._session_store.load_checkpoint(loaded_session.id)
        candidate_approval, candidate_round = _pending_approval_and_round_for_atomic_claim(
            checkpoint,
            approval_id=request.approval_id,
            tool_round_id=request.tool_round_id,
            gating_tool_call_id=request.tool_call_id,
            redactor=self._secret_redactor,
        )
        candidate_intent = approval_support.approval_resolution_intent_from_checkpoint(
            checkpoint,
            redactor=self._secret_redactor,
        )
        resolution_request_digest = approval_support.approval_resolution_request_digest(request)
        can_create_resolution_intent = True
        try:
            candidate_events = await self._session_store.load_events(loaded_session.id)
            candidate_history = approval_support.approval_resolution_history(
                events=candidate_events,
                approval=candidate_approval,
            )
            candidate_decision = request.decision
            if (
                approval_support.pending_approval_expired(
                    candidate_approval,
                    self._clock(),
                )
                and not candidate_history.has_granted_activity
            ):
                candidate_decision = ToolApprovalDecision.DENY
            approval_support.validate_retry_decision(
                history=candidate_history,
                approval=candidate_approval,
                decision=candidate_decision,
            )
            candidate_outcomes = approval_support.recorded_tool_outcomes(
                events=candidate_events,
                approval=candidate_approval,
            )
            if candidate_history.has_resolution_activity or candidate_outcomes:
                # Durable resolution activity without an existing request digest
                # cannot prove which audit-bearing request authorized it. Never
                # infer that identity from redacted or bounded event payloads.
                can_create_resolution_intent = False
        except Exception:
            # The continuation repeats this validation inside its established
            # interruption-event boundary. Do not let a rejected legacy retry
            # become durable authority before that happens.
            can_create_resolution_intent = False
        effective_structured_output = _effective_approval_structured_output(
            structured_output=request.structured_output,
            pending_approval=candidate_approval,
        )
        invocation_semantics = _effective_approval_invocation_semantics(
            request=request,
            pending_approval=candidate_approval,
            structured_output=effective_structured_output,
            effective_retry_policy=self._effective_retry_policy,
        )
        require_secret_free_structured_output_spec(
            effective_structured_output,
            redactor=self._secret_redactor,
            field_name="ToolApprovalRequest.structured_output",
        )
        registered_agent = self._resolve_registered_agent(loaded_session.agent_name)
        registered_provider = self._resolve_registered_provider(loaded_session.provider_name)
        budget_policy_snapshot = copy_budget_policy(self._resolve_budget_policy())
        execution_profile_snapshot = await self._validate_execution_profile_continuation(
            loaded_session,
            checkpoint,
            registered_agent,
            registered_provider,
            request.loop_policies,
            budget_policy=budget_policy_snapshot,
            request_budget_limits=invocation_semantics.budget_limits,
            structured_output=invocation_semantics.structured_output,
            thinking=invocation_semantics.thinking,
            max_steps=invocation_semantics.max_steps,
            limits=invocation_semantics.limits,
            retry_policy=invocation_semantics.retry_policy,
            invocation_semantics_available=True,
        )
        _require_native_structured_output_support(
            effective_structured_output, registered_provider=registered_provider
        )
        registered_environment = self._resolve_registered_environment(
            loaded_session.environment_name
        )
        invocation_secrets.require_continuation_secret_resolution_compatibility(
            candidate_approval.secret_resolution_scope,
            registered_environment,
        )
        pending_approval: PendingToolApproval | None = None
        pending_round: tool_round_recovery.PendingToolRound | None = None
        claimed_intent: approval_support.ApprovalResolutionIntent | None = None

        def claim_exact_approval(
            _current_session: Session,
            checkpoint: dict[str, Any] | None,
        ) -> dict[str, Any] | None:
            nonlocal claimed_intent, pending_approval, pending_round
            pending_approval, pending_round = _pending_approval_and_round_for_atomic_claim(
                checkpoint,
                approval_id=request.approval_id,
                tool_round_id=request.tool_round_id,
                gating_tool_call_id=request.tool_call_id,
                redactor=self._secret_redactor,
            )
            if pending_approval != candidate_approval or pending_round != candidate_round:
                raise RuntimeError("Pending tool approval changed before it was claimed.")
            current_intent = approval_support.approval_resolution_intent_from_checkpoint(
                checkpoint,
                redactor=self._secret_redactor,
            )
            if current_intent != candidate_intent:
                raise RuntimeError(
                    "Approval resolution intent changed before the approval was claimed."
                )
            claimed_checkpoint = _checkpoint_with_legacy_approval_round(
                checkpoint,
                approval=pending_approval,
                redactor=self._secret_redactor,
            )
            if current_intent is not None:
                claimed_intent = current_intent
                return claimed_checkpoint
            intent_decision = request.decision if can_create_resolution_intent else None
            if intent_decision is None:
                claimed_intent = None
                return claimed_checkpoint
            claimed_checkpoint = approval_support.checkpoint_with_approval_resolution_intent(
                claimed_checkpoint,
                approval=pending_approval,
                decision=intent_decision,
                resolution_request_digest=resolution_request_digest,
                redactor=self._secret_redactor,
            )
            claimed_intent = approval_support.approval_resolution_intent_from_checkpoint(
                claimed_checkpoint,
                redactor=self._secret_redactor,
            )
            return claimed_checkpoint

        session, resumed_event = await self._transition_recovery_session_to_running(
            loaded_session,
            checkpoint=checkpoint,
            registered_agent=registered_agent,
            registered_environment=registered_environment,
            checkpoint_transform=claim_exact_approval,
            execution_profile_snapshot=execution_profile_snapshot,
            before_mutation=before_mutation,
        )
        if pending_approval is None or pending_round is None:
            raise RuntimeError("Tool approval claim completed without approval state.")
        if resumed_event is not None:
            yield resumed_event
        continuation_stream = self.continue_tool_approval_resolution(
            request=request,
            session=session,
            pending_approval=pending_approval,
            registered_agent=registered_agent,
            registered_provider=registered_provider,
            registered_environment=registered_environment,
            execution_profile_snapshot=execution_profile_snapshot,
            budget_policy=budget_policy_snapshot,
            deferred_messages=pending_round.deferred_messages,
            claimed_resolution_intent=claimed_intent,
        )
        authoritative_failure: BaseException | None = None
        abandoned = False
        try:
            async for event in continuation_stream:
                yield event
        except BaseException as exc:
            authoritative_failure = exc
            abandoned = _recovery_abandonment_signal(exc) is not None
            raise
        finally:
            await self._cleanup_entrypoint_handoff(
                stream=continuation_stream,
                session_id=session.id,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
                authoritative_failure=authoritative_failure,
                finalize_abandoned=abandoned,
                release_run_fence=True,
                execution_profile=execution_profile_snapshot.profile,
            )

    async def resolve_provider_operation(
        self,
        request: ProviderOperationResolutionRequest,
        *,
        before_mutation: RecoveryMutationHook | None = None,
    ) -> AsyncGenerator[Event, None]:
        """Accept one disposition and drive its durable effect to a recovery boundary."""

        request = prepare_provider_operation_resolution_request(
            request,
            redactor=self._secret_redactor,
        )
        result = await resolve_provider_operation_stage(
            self._session_store,
            request,
            redactor=self._secret_redactor,
            before_resolution=before_mutation,
        )
        if not result.replayed:
            await self._event_writer.fan_out_persisted([result.event])
        yield result.event

        pending_resolution = await load_pending_provider_operation_disposition(
            self._session_store,
            request.session_id,
        )
        if pending_resolution is None:
            return
        pending, durable_result = pending_resolution
        if durable_result.record.request_digest != result.record.request_digest:
            raise ProviderOperationEvidenceError(
                "Pending provider-operation disposition changed after acceptance."
            )
        if await self._provider_operation_disposition_execution_started(
            pending=pending,
            result=durable_result,
        ):
            for settlement_event in await self._settle_provider_operation_disposition_reservations(
                pending=pending,
                result=durable_result,
            ):
                yield settlement_event
            return
        disposition_stream = self._finish_pending_provider_operation_disposition(
            pending=pending,
            result=durable_result,
        )
        try:
            try:
                async for event in disposition_stream:
                    yield event
            except ExceptionGroup as replay_failure:
                if self._interaction_transition_replay_failures(
                    replay_failure
                ) is None or not await self._provider_operation_disposition_execution_started(
                    pending=pending,
                    result=durable_result,
                ):
                    raise
            except (SessionRunFenced, SessionStatusConflict):
                if not await self._provider_operation_disposition_execution_started(
                    pending=pending,
                    result=durable_result,
                ):
                    raise
        finally:
            await disposition_stream.aclose()

    async def _provider_operation_disposition_execution_started(
        self,
        *,
        pending: ProviderOperationPendingDisposition,
        result: ProviderOperationResolutionResult,
    ) -> bool:
        """Recognize exact durable progress owned by another disposition caller."""

        session = await self._session_store.load(pending.session_id)
        if session is None:
            raise KeyError(f"Session not found: {pending.session_id}")
        if pending.action is ProviderOperationResolutionAction.FALLBACK_RETRY:
            if session.status is not SessionStatus.RUNNING:
                return False
            checkpoint = await self._session_store.load_checkpoint(pending.session_id)
            active_profile = active_invocation_execution_profile_from_checkpoint(checkpoint)
            return bool(
                active_profile is not None
                and active_profile.session_id == pending.session_id
                and active_profile.interaction_id == result.event.interaction_id
                and active_profile.run_epoch == session.run_epoch
                and active_profile.profile.fingerprint == pending.execution_profile_fingerprint
            )

        if session.status is not SessionStatus.FAILED:
            return False
        interaction_event_id = provider_operation_resolution_outcome_event_id(
            result.record.resolution_id,
            "interaction_failed",
        )
        interaction_records = await self._session_store.query_events(
            EventQuery(
                session_id=pending.session_id,
                event_id=interaction_event_id,
                limit=2,
            )
        )
        if not interaction_records:
            return False
        if len(interaction_records) != 1:
            raise ProviderOperationEvidenceError(
                "Provider-operation failure has duplicate interaction evidence."
            )
        validate_provider_operation_resolution_outcome_event(
            interaction_records[0].event,
            resolution_event=result.event,
            outcome="interaction_failed",
            expected_execution_profile_fingerprint=pending.execution_profile_fingerprint,
        )
        return True

    async def _provider_operation_disposition_effect_is_durable(
        self,
        *,
        pending: ProviderOperationPendingDisposition,
        result: ProviderOperationResolutionResult,
    ) -> bool:
        if pending.action is ProviderOperationResolutionAction.FAIL:
            terminal_event_id = provider_operation_resolution_outcome_event_id(
                result.record.resolution_id,
                "session_failed",
            )
            terminal_records = await self._session_store.query_events(
                EventQuery(
                    session_id=pending.session_id,
                    event_id=terminal_event_id,
                    limit=2,
                )
            )
            if not terminal_records:
                return False
            if len(terminal_records) != 1:
                raise ProviderOperationEvidenceError(
                    "Provider-operation failure has duplicate terminal evidence."
                )
            validate_provider_operation_resolution_outcome_event(
                terminal_records[0].event,
                resolution_event=result.event,
                outcome="session_failed",
                expected_execution_profile_fingerprint=pending.execution_profile_fingerprint,
            )
            for outcome in ("model_error", "interaction_failed"):
                event_id = provider_operation_resolution_outcome_event_id(
                    result.record.resolution_id,
                    outcome,
                )
                records = await self._session_store.query_events(
                    EventQuery(
                        session_id=pending.session_id,
                        event_id=event_id,
                        limit=2,
                    )
                )
                if len(records) != 1:
                    raise ProviderOperationEvidenceError(
                        "Provider-operation failure has incomplete durable evidence."
                    )
                validate_provider_operation_resolution_outcome_event(
                    records[0].event,
                    resolution_event=result.event,
                    outcome=outcome,
                    expected_execution_profile_fingerprint=(pending.execution_profile_fingerprint),
                )
            session = await self._session_store.load(pending.session_id)
            if session is None or session.status is not SessionStatus.FAILED:
                raise ProviderOperationEvidenceError(
                    "Provider-operation failure conflicts with durable session status."
                )
            return True

        if await self._provider_operation_fallback_terminal_outcome_is_durable(
            pending=pending,
            result=result,
        ):
            return True

        target_ordinal = pending.target_dispatch_ordinal
        if target_ordinal is None:
            raise ProviderOperationEvidenceError(
                "Fallback disposition lost its target dispatch ordinal."
            )
        target_stage_id = f"{pending.logical_step_id}:dispatch:{target_ordinal}"
        target_stage = await self._session_store.load_model_completion_stage(
            pending.session_id,
            target_stage_id,
        )
        if target_stage is not None:
            target_context = model_completion_recovery_context_from_stage(target_stage)
            if (
                target_stage.logical_step_id != pending.logical_step_id
                or target_stage.dispatch_ordinal != target_ordinal
                or target_context is None
                or target_context.execution_profile_fingerprint
                != pending.execution_profile_fingerprint
            ):
                raise ProviderOperationEvidenceError(
                    "Fallback disposition target stage has contradictory identity."
                )
            return True
        return False

    async def _provider_operation_fallback_terminal_outcome_is_durable(
        self,
        *,
        pending: ProviderOperationPendingDisposition,
        result: ProviderOperationResolutionResult,
    ) -> bool:
        """Recognize a typed pre-dispatch stop owned by this disposition."""

        session = await self._session_store.load(pending.session_id)
        if session is None or session.status is not SessionStatus.INTERRUPTED:
            return False
        resolution_records = await self._session_store.query_events(
            EventQuery(
                session_id=pending.session_id,
                event_id=result.event.id,
                limit=2,
            )
        )
        if len(resolution_records) != 1:
            raise ProviderOperationEvidenceError(
                "Fallback terminal outcome has missing or duplicate resolution evidence."
            )
        resolution_sequence = resolution_records[0].sequence
        interaction_records = await self._session_store.query_events(
            EventQuery(
                session_id=pending.session_id,
                interaction_id=result.event.interaction_id,
                event_type=EventType.INTERACTION_INTERRUPTED,
                after_sequence=resolution_sequence,
                order_by=EventOrder.SEQUENCE_DESC,
                limit=2,
            )
        )
        terminal_records = await self._session_store.query_events(
            EventQuery(
                session_id=pending.session_id,
                event_type=EventType.SESSION_INTERRUPTED,
                after_sequence=resolution_sequence,
                order_by=EventOrder.SEQUENCE_DESC,
                limit=2,
            )
        )
        if not interaction_records or not terminal_records:
            return False
        if len(interaction_records) != 1 or len(terminal_records) != 1:
            raise ProviderOperationEvidenceError(
                "Fallback terminal outcome has contradictory terminal evidence."
            )
        try:
            interaction = InteractionSummaryEvidence.model_validate(
                interaction_records[0].event.payload
            )
        except (TypeError, ValueError):
            raise ProviderOperationEvidenceError(
                "Fallback terminal outcome has malformed interaction evidence."
            ) from None
        if interaction.status is not InteractionStatus.INTERRUPTED:
            raise ProviderOperationEvidenceError(
                "Fallback terminal outcome has contradictory interaction status."
            )
        terminal_payload = terminal_records[0].event.payload
        interruption_type = terminal_payload.get("interruption_type")
        if interruption_type == "operator_requested":
            return True
        if (
            interruption_type != "limit_reached"
            and terminal_payload.get("terminal_evidence_repaired") is not True
        ):
            return False
        limit_records = await self._session_store.query_events(
            EventQuery(
                session_id=pending.session_id,
                event_type=EventType.SESSION_LIMIT_REACHED,
                after_sequence=resolution_sequence,
                order_by=EventOrder.SEQUENCE_DESC,
                limit=1,
            )
        )
        return bool(limit_records)

    async def _retire_completed_provider_operation_disposition(
        self,
        *,
        pending: ProviderOperationPendingDisposition,
        result: ProviderOperationResolutionResult,
    ) -> bool:
        await self._settle_provider_operation_disposition_reservations(
            pending=pending,
            result=result,
        )
        if not await self._provider_operation_disposition_effect_is_durable(
            pending=pending,
            result=result,
        ):
            return False
        await clear_pending_provider_operation_disposition(self._session_store, pending)
        return True

    async def _settle_provider_operation_disposition_reservations(
        self,
        *,
        pending: ProviderOperationPendingDisposition,
        result: ProviderOperationResolutionResult,
    ) -> tuple[Event, ...]:
        """Settle the source dispatch before replacement or terminalization."""

        stage = await self._session_store.load_model_completion_stage(
            pending.session_id,
            pending.stage_id,
        )
        if stage is None:
            raise ProviderOperationEvidenceError(
                "Provider-operation disposition lost its source stage."
            )
        recovery_context = model_completion_recovery_context_from_stage(stage)
        if pending.action is ProviderOperationResolutionAction.FALLBACK_RETRY:
            if recovery_context is None:
                raise ProviderOperationEvidenceError(
                    "Provider-operation fallback requires durable model-completion context."
                )
            session = await self._session_store.load(pending.session_id)
            if session is None:
                raise KeyError(f"Session not found: {pending.session_id}")
            registered_agent = self._resolve_registered_agent(session.agent_name)
            _retried_model_step_tool_exposure_authority(
                recovery_context.tool_exposure,
                registered_agent,
                session,
            )
        if not stage.reservation_ids:
            return ()
        if recovery_context is None:
            raise ProviderOperationEvidenceError(
                "Budgeted provider-operation disposition has no accounting context."
            )
        model_attempt_id = stage.intent.get("model_attempt_id")
        provider_name = stage.intent.get("provider_name")
        if type(model_attempt_id) is not str or type(provider_name) is not str:
            raise ProviderOperationEvidenceError(
                "Provider-operation disposition lost its dispatch identity."
            )
        session = await self._session_store.load(pending.session_id)
        if session is None:
            raise KeyError(f"Session not found: {pending.session_id}")
        reason = (
            "provider operation explicitly failed; usage unknown; charged reserved amount"
            if pending.action is ProviderOperationResolutionAction.FAIL
            else (
                "provider operation fallback accepted; original usage unknown; "
                "charged reserved amount"
            )
        )
        try:
            events = await (
                self._run_limit_controller.reconcile_unavailable_provider_operation_reservations(
                    reservation_ids=stage.reservation_ids,
                    recovery_contexts=recovery_context.budget_reservations,
                    session=session,
                    provider_name=provider_name,
                    model_attempt_identity=ModelAttemptIdentity(
                        model_step_id=stage.logical_step_id,
                        model_attempt_id=model_attempt_id,
                    ),
                    dispatch_id=stage.stage_id,
                    request_billing_identity=recovery_context.billing_identity,
                    reason=reason,
                    occurred_at=result.record.resolved_at,
                )
            )
        except (KeyError, NotImplementedError, TypeError, ValueError) as accounting_error:
            raise ProviderOperationEvidenceError(
                "Provider-operation disposition could not reconstruct its original budget "
                "reservation and pricing context."
            ) from accounting_error
        return tuple(events)

    async def _finish_pending_provider_operation_disposition(
        self,
        *,
        pending: ProviderOperationPendingDisposition,
        result: ProviderOperationResolutionResult,
    ) -> AsyncGenerator[Event, None]:
        """Finish one accepted disposition without replaying its old provider request."""

        for settlement_event in await self._settle_provider_operation_disposition_reservations(
            pending=pending,
            result=result,
        ):
            yield settlement_event

        if await self._retire_completed_provider_operation_disposition(
            pending=pending,
            result=result,
        ):
            return

        loaded_session = await self._session_store.load(pending.session_id)
        if loaded_session is None:
            raise KeyError(f"Session not found: {pending.session_id}")
        if pending.action is ProviderOperationResolutionAction.FALLBACK_RETRY:
            if loaded_session.status is not SessionStatus.INTERRUPTED:
                raise SessionStatusConflict(
                    "Fallback retry can continue only from an interrupted session."
                )
        elif loaded_session.status not in {
            SessionStatus.INTERRUPTED,
            SessionStatus.FAILED,
        }:
            raise SessionStatusConflict("Fail resolution requires interrupted provider work.")
        registered_agent = self._resolve_registered_agent(loaded_session.agent_name)
        registered_provider = self._resolve_registered_provider(loaded_session.provider_name)
        registered_environment = self._resolve_registered_environment(
            loaded_session.environment_name
        )
        checkpoint = await self._session_store.load_checkpoint(loaded_session.id)
        recovery_context: ModelCompletionRecoveryContext | None = None
        if pending.action is ProviderOperationResolutionAction.FALLBACK_RETRY:
            stage = await self._session_store.load_model_completion_stage(
                pending.session_id,
                pending.stage_id,
            )
            if stage is None:
                raise RuntimeError("Resolved provider-operation stage is missing.")
            recovery_context = model_completion_recovery_context_from_stage(stage)
            if recovery_context is None:
                raise RuntimeError(
                    "Provider-operation fallback requires durable model-completion context."
                )
        budget_policy_snapshot = copy_budget_policy(self._resolve_budget_policy())
        execution_profile_snapshot = await self._validate_execution_profile_continuation(
            loaded_session,
            checkpoint,
            registered_agent,
            registered_provider,
            budget_policy=budget_policy_snapshot,
            request_budget_limits=(
                () if recovery_context is None else recovery_context.budget_limits
            ),
            structured_output=(
                None if recovery_context is None else recovery_context.structured_output
            ),
            thinking=None if recovery_context is None else recovery_context.thinking,
            max_steps=(
                _DEFAULT_APPROVAL_MAX_STEPS
                if recovery_context is None
                else recovery_context.max_steps
            ),
            limits=None if recovery_context is None else recovery_context.limits,
            retry_policy=(None if recovery_context is None else recovery_context.retry_policy),
            invocation_semantics_available=recovery_context is not None,
        )
        if pending.action is ProviderOperationResolutionAction.FAIL:
            async for event in self._fail_provider_operation(
                ProviderOperationFailureRequest(
                    resolution_event=result.event,
                    session=loaded_session,
                    registered_agent=registered_agent,
                    registered_environment=registered_environment,
                    execution_profile=execution_profile_snapshot.profile,
                    legacy_resolution_without_profile=(
                        result.record.execution_profile_fingerprint is None
                    ),
                )
            ):
                yield event
            if not await self._retire_completed_provider_operation_disposition(
                pending=pending,
                result=result,
            ):
                raise RuntimeError(
                    "Provider-operation failure disposition has no durable terminal outcome."
                )
            return

        if recovery_context is None:
            raise RuntimeError(
                "Provider-operation fallback requires durable model-completion context."
            )
        session: Session | None = None
        fallback_stream: AsyncGenerator[Event, None] | None = None
        authoritative_failure: BaseException | None = None
        detached_billing_failure: BaseExceptionGroup | None = None
        billing_state_check_failure: RuntimeError | None = None
        try:
            session, resumed_event = await self._transition_recovery_session_to_running(
                loaded_session,
                checkpoint=checkpoint,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
                execution_profile_snapshot=execution_profile_snapshot,
                preserve_open_interaction_on_failure=True,
            )
            if resumed_event is not None:
                yield resumed_event

            fallback_stream = self._run_pending_provider_operation_fallback(
                pending=pending,
                result=result,
                session=session,
                registered_agent=registered_agent,
                registered_provider=registered_provider,
                registered_environment=registered_environment,
                execution_profile_snapshot=execution_profile_snapshot,
                recovery_context=recovery_context,
                budget_policy=budget_policy_snapshot,
                release_run_fence_on_cleanup=False,
            )
            try:
                async for event in fallback_stream:
                    yield event
            finally:
                await fallback_stream.aclose()
        except BaseExceptionGroup as exc:
            detached_billing_failure = detach_billing_identity_cancellation_group(exc)
            if detached_billing_failure is None:
                authoritative_failure = exc
                raise
            authoritative_failure = detached_billing_failure
        except _FallbackBillingCancellationStateCheckFailed:
            billing_state_check_failure = RuntimeError(
                "Session interruption state check failed after provider billing cancellation"
            )
            authoritative_failure = billing_state_check_failure
        except BaseException as exc:
            authoritative_failure = exc
            raise
        finally:
            if session is not None:
                await self._cleanup_entrypoint_handoff(
                    stream=None,
                    session_id=session.id,
                    registered_agent=registered_agent,
                    registered_environment=registered_environment,
                    authoritative_failure=authoritative_failure,
                    finalize_abandoned=False,
                    release_run_fence=True,
                    execution_profile=execution_profile_snapshot.profile,
                )
        if detached_billing_failure is not None:
            authoritative_failure = None
            fallback_stream = None
            del registered_provider, registered_environment
            raise detached_billing_failure from None
        if billing_state_check_failure is not None:
            authoritative_failure = None
            fallback_stream = None
            del registered_provider, registered_environment
            raise billing_state_check_failure from None

    async def _run_pending_provider_operation_fallback(
        self,
        *,
        pending: ProviderOperationPendingDisposition,
        result: ProviderOperationResolutionResult,
        session: Session,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_provider: runtime_records.RegisteredProvider,
        registered_environment: runtime_records.RegisteredEnvironment | None,
        execution_profile_snapshot: ActiveInvocationExecutionProfile,
        recovery_context: ModelCompletionRecoveryContext,
        budget_policy: BudgetPolicy | None,
        release_run_fence_on_cleanup: bool,
    ) -> AsyncGenerator[Event, None]:
        """Run the accepted fallback from an already fenced running session."""

        transcript = await self._session_store.load_transcript(session.id)
        recovery_events = await self._session_store.load_events(session.id)
        continued_accounting = recovery_context.run_limit_accounting
        if continued_accounting is not None:
            continued_accounting = rebase_run_limit_accounting_context(
                continued_accounting,
                session_id=session.id,
                limits=recovery_context.limits,
                budget_limits=request_budget_limits_for_session(
                    limits=recovery_context.budget_limits,
                    agent_name=registered_agent.spec.name,
                    causal_budget_id=session.causal_budget_id,
                ),
                events=recovery_events,
                reset_run_limits=False,
                reset_budgets=False,
                now=self._clock(),
            )
        if pending.source_step > recovery_context.max_steps:
            raise ProviderOperationEvidenceError(
                "Provider-operation fallback step exceeds its durable run limit."
            )
        session_stream = self._run_session(
            RecoverySessionRunRequest(
                session=session,
                registered_agent=registered_agent,
                registered_provider=registered_provider,
                registered_environment=registered_environment,
                active_invocation_profile=_rebound_active_invocation_profile(
                    session,
                    execution_profile_snapshot,
                ),
                messages=transcript,
                messages_to_append=[],
                max_steps=recovery_context.max_steps,
                limits=recovery_context.limits,
                budget_limits=recovery_context.budget_limits,
                budget_policy=copy_budget_policy(budget_policy),
                retry_policy=recovery_context.retry_policy,
                structured_output=recovery_context.structured_output,
                thinking=recovery_context.thinking,
                request_loop_policies=(),
                request_metadata=recovery_context.request_metadata,
                task_id=recovery_context.task_id,
                task_worker_id=None,
                start_event_type=None,
                start_event_payload={},
                start_task_on_enter=False,
                release_run_fence_on_exit=False,
                run_limit_accounting=continued_accounting,
                initial_model_step_identity=ModelStepIdentity(
                    model_step_id=pending.logical_step_id,
                ),
                initial_model_step_number=pending.source_step,
                initial_model_step_tool_exposure=(
                    _retried_model_step_tool_exposure_authority(
                        recovery_context.tool_exposure,
                        registered_agent,
                        session,
                    )
                ),
                preserve_failure_until_initial_provider_dispatch=True,
            )
        )
        authoritative_failure: BaseException | None = None
        replacement_dispatch_durable = False
        limit_outcome_started = False
        try:
            async for event in session_stream:
                if event.type is EventType.PROVIDER_OPERATION_STARTING:
                    if not await self._retire_completed_provider_operation_disposition(
                        pending=pending,
                        result=result,
                    ):
                        raise RuntimeError(
                            "Provider-operation fallback started without its exact durable stage."
                        )
                    replacement_dispatch_durable = True
                elif event.type is EventType.SESSION_LIMIT_REACHED:
                    limit_outcome_started = True
                yield event
            if not await self._retire_completed_provider_operation_disposition(
                pending=pending,
                result=result,
            ):
                raise RuntimeError("Provider-operation fallback has no exact durable target stage.")
        except BaseException as exc:
            authoritative_failure = exc
            raise
        finally:
            if (
                limit_outcome_started
                and isinstance(authoritative_failure, GeneratorExit)
                and not replacement_dispatch_durable
            ):
                # The limit decision is already durable and cannot lead to
                # provider dispatch. Finish its typed terminal evidence during
                # stream cleanup so recovery neither duplicates the limit event
                # nor mistakes this accepted disposition for pending work.
                async for _event in session_stream:
                    pass
                if not await self._retire_completed_provider_operation_disposition(
                    pending=pending,
                    result=result,
                ):
                    raise RuntimeError(
                        "Provider-operation fallback limit has no durable terminal outcome."
                    )
            cleanup = (
                self._cleanup_entrypoint_handoff
                if release_run_fence_on_cleanup
                else self._cleanup_recovery_handoff
            )
            await cleanup(
                stream=session_stream,
                session_id=session.id,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
                authoritative_failure=authoritative_failure,
                finalize_abandoned=(
                    replacement_dispatch_durable
                    and _recovery_abandonment_signal(authoritative_failure) is not None
                ),
                release_run_fence=release_run_fence_on_cleanup,
                execution_profile=execution_profile_snapshot.profile,
            )

    async def recover_tool_approval_request(
        self,
        request: ToolApprovalRecoveryRequest,
        *,
        before_mutation: RecoveryMutationHook | None = None,
    ) -> AsyncGenerator[Event, None]:
        loaded_session = await self._session_store.load(request.session_id)
        if loaded_session is None:
            raise KeyError(f"Session not found: {request.session_id}")

        checkpoint = await self._session_store.load_checkpoint(loaded_session.id)
        candidate_approval, candidate_round = _pending_approval_and_round_for_atomic_claim(
            checkpoint,
            approval_id=request.approval_id,
            tool_round_id=request.tool_round_id,
            recovery_tool_call_id=request.tool_call_id,
            redactor=self._secret_redactor,
        )
        candidate_intent = approval_support.approval_resolution_intent_from_checkpoint(
            checkpoint,
            redactor=self._secret_redactor,
        )
        effective_structured_output = _effective_approval_structured_output(
            structured_output=request.structured_output,
            pending_approval=candidate_approval,
        )
        invocation_semantics = _effective_approval_invocation_semantics(
            request=request,
            pending_approval=candidate_approval,
            structured_output=effective_structured_output,
            effective_retry_policy=self._effective_retry_policy,
        )
        require_secret_free_structured_output_spec(
            effective_structured_output,
            redactor=self._secret_redactor,
            field_name="ToolApprovalRecoveryRequest.structured_output",
        )
        pending_tool_call = approval_support.pending_tool_call_for_recovery(
            approval=candidate_approval,
            tool_call_id=request.tool_call_id,
        )
        registered_agent = self._resolve_registered_agent(loaded_session.agent_name)
        registered_provider = self._resolve_registered_provider(loaded_session.provider_name)
        budget_policy_snapshot = copy_budget_policy(self._resolve_budget_policy())
        execution_profile_snapshot = await self._validate_execution_profile_continuation(
            loaded_session,
            checkpoint,
            registered_agent,
            registered_provider,
            request.loop_policies,
            budget_policy=budget_policy_snapshot,
            request_budget_limits=invocation_semantics.budget_limits,
            structured_output=invocation_semantics.structured_output,
            thinking=invocation_semantics.thinking,
            max_steps=invocation_semantics.max_steps,
            limits=invocation_semantics.limits,
            retry_policy=invocation_semantics.retry_policy,
            invocation_semantics_available=True,
        )
        _require_native_structured_output_support(
            effective_structured_output, registered_provider=registered_provider
        )
        registered_environment = self._resolve_registered_environment(
            loaded_session.environment_name
        )
        invocation_secrets.require_continuation_secret_resolution_compatibility(
            candidate_approval.secret_resolution_scope,
            registered_environment,
        )
        pending_approval: PendingToolApproval | None = None
        pending_round: tool_round_recovery.PendingToolRound | None = None
        claimed_resolution_intent: approval_support.ApprovalResolutionIntent | None = None

        def claim_exact_approval(
            _current_session: Session,
            checkpoint: dict[str, Any] | None,
        ) -> dict[str, Any] | None:
            nonlocal claimed_resolution_intent, pending_approval, pending_round
            pending_approval, pending_round = _pending_approval_and_round_for_atomic_claim(
                checkpoint,
                approval_id=request.approval_id,
                tool_round_id=request.tool_round_id,
                recovery_tool_call_id=request.tool_call_id,
                redactor=self._secret_redactor,
            )
            if pending_approval != candidate_approval or pending_round != candidate_round:
                raise RuntimeError("Pending tool approval changed before it was claimed.")
            current_intent = approval_support.approval_resolution_intent_from_checkpoint(
                checkpoint,
                redactor=self._secret_redactor,
            )
            if current_intent != candidate_intent:
                raise RuntimeError(
                    "Approval resolution intent changed before recovery was claimed."
                )
            if current_intent is not None:
                approval_support.require_resolution_intent_matches_approval(
                    current_intent,
                    approval=pending_approval,
                )
            claimed_resolution_intent = current_intent
            return _checkpoint_with_legacy_approval_round(
                checkpoint,
                approval=pending_approval,
                redactor=self._secret_redactor,
            )

        session, resumed_event = await self._transition_recovery_session_to_running(
            loaded_session,
            checkpoint=checkpoint,
            registered_agent=registered_agent,
            registered_environment=registered_environment,
            checkpoint_transform=claim_exact_approval,
            execution_profile_snapshot=execution_profile_snapshot,
            before_mutation=before_mutation,
        )
        if pending_approval is None or pending_round is None:
            raise RuntimeError("Tool approval recovery claim completed without approval state.")
        if resumed_event is not None:
            yield resumed_event
        recovery_stream = self.recover_tool_approval(
            request=request,
            loaded_session=loaded_session,
            session=session,
            pending_approval=pending_approval,
            pending_tool_call=pending_tool_call,
            registered_agent=registered_agent,
            registered_provider=registered_provider,
            registered_environment=registered_environment,
            execution_profile_snapshot=execution_profile_snapshot,
            budget_policy=budget_policy_snapshot,
            deferred_messages=pending_round.deferred_messages,
            claimed_resolution_intent=claimed_resolution_intent,
        )
        authoritative_failure: BaseException | None = None
        try:
            async for event in recovery_stream:
                yield event
        except BaseException as exc:
            authoritative_failure = exc
            raise
        finally:
            await self._cleanup_entrypoint_handoff(
                stream=recovery_stream,
                session_id=session.id,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
                authoritative_failure=authoritative_failure,
                finalize_abandoned=False,
                release_run_fence=False,
                execution_profile=execution_profile_snapshot.profile,
            )

    def _reject_approval_owned_tool_round_recovery(
        self,
        checkpoint: dict[str, Any] | None,
    ) -> None:
        pending_approval = approval_support.pending_approval_from_checkpoint(
            checkpoint,
            redactor=self._secret_redactor,
        )
        if pending_approval is not None:
            raise RuntimeError(
                "Pending approval-owned tool rounds must be recovered with "
                "ToolApprovalRecoveryRequest."
            )

    async def recover_tool_round_request(
        self,
        request: ToolRoundRecoveryRequest,
        *,
        before_mutation: RecoveryMutationHook | None = None,
    ) -> AsyncGenerator[Event, None]:
        """Recover a crashed ordinary tool round with an operator-verified outcome.

        A tool call in a non-approval round started but recorded no terminal event
        (a crash mid-tool), so an automatic resume would close it as an
        unknown-outcome failure. The caller supplies the externally verified outcome
        for that `tool_call_id`; Cayu persists it as the call's terminal result and
        never re-runs the tool. One call per invocation: if other
        started-but-unresolved calls remain, the session returns to INTERRUPTED with
        `manual_recovery_required` naming the next call; otherwise the round closes
        from the recorded outcomes and the model loop continues. A crashed round can
        leave the session FAILED (an in-process persistence error) or in a stale live
        status (a process kill), so FAILED and RUNNING are accepted alongside
        INTERRUPTED. An existing INTERRUPTING transition wins rather than being
        reopened by recovery. The in-process claim registered while this recovery
        streams blocks duplicate work in this process, while a durable recovery
        claim serializes other workers and fences an expired owner. If this call
        fails after claiming a stale live session, the session closes to the
        resumable INTERRUPTED state. When the recovered terminal event is already
        durable, the evidence remains authoritative: do not retry the same
        `tool_call_id` — `resume(...)` finishes the round from the persisted outcome.
        """
        loaded_session = await self._session_store.load(request.session_id)
        if loaded_session is None:
            raise KeyError(f"Session not found: {request.session_id}")

        checkpoint = await self._session_store.load_checkpoint(loaded_session.id)
        pending_round = tool_round_recovery.pending_tool_round_from_checkpoint(
            checkpoint,
            redactor=self._secret_redactor,
            consume_on_rejection=True,
        )
        if pending_round is None:
            raise RuntimeError("Session has no pending tool round.")
        self._reject_approval_owned_tool_round_recovery(checkpoint)
        if pending_round.tool_round_id != request.round_id:
            raise ValueError(f"Tool round id does not match pending round: {request.round_id}")
        effective_structured_output = _effective_tool_round_structured_output(
            structured_output=request.structured_output,
            pending_round=pending_round,
        )
        invocation_semantics = _effective_tool_round_invocation_semantics(
            request=request,
            pending_round=pending_round,
            structured_output=effective_structured_output,
            effective_retry_policy=self._effective_retry_policy,
        )
        require_secret_free_structured_output_spec(
            effective_structured_output,
            redactor=self._secret_redactor,
            field_name="ToolRoundRecoveryRequest.structured_output",
        )

        pending_tool_call = approval_support.round_tool_call_for_recovery(
            pending_calls=pending_round.tool_calls,
            tool_call_id=request.tool_call_id,
        )
        registered_agent = self._resolve_registered_agent(loaded_session.agent_name)
        if pending_round.agent_name != registered_agent.spec.name:
            raise RuntimeError(
                f"Pending tool round belongs to a different agent: {pending_round.agent_name}."
            )
        registered_provider = self._resolve_registered_provider(loaded_session.provider_name)
        budget_policy_snapshot = copy_budget_policy(self._resolve_budget_policy())
        execution_profile_snapshot = await self._validate_execution_profile_continuation(
            loaded_session,
            checkpoint,
            registered_agent,
            registered_provider,
            request.loop_policies,
            budget_policy=budget_policy_snapshot,
            request_budget_limits=invocation_semantics.budget_limits,
            structured_output=invocation_semantics.structured_output,
            thinking=invocation_semantics.thinking,
            max_steps=invocation_semantics.max_steps,
            limits=invocation_semantics.limits,
            retry_policy=invocation_semantics.retry_policy,
            invocation_semantics_available=True,
        )
        _require_native_structured_output_support(
            effective_structured_output, registered_provider=registered_provider
        )
        registered_environment = self._resolve_registered_environment(
            loaded_session.environment_name
        )
        invocation_secrets.require_continuation_secret_resolution_compatibility(
            approval_support.tool_round_secret_resolution_scope(pending_round),
            registered_environment,
        )
        pending_operator_interruption = (
            checkpoint is not None and _PENDING_SESSION_INTERRUPT_CHECKPOINT_KEY in checkpoint
        )
        if (
            checkpoint is not None
            and _PENDING_INTERRUPTION_CASCADE_CHECKPOINT_KEY in checkpoint
            and not pending_operator_interruption
        ):
            raise _ManualRecoveryCascadePending(
                "Session has an incomplete background interruption cascade."
            )
        if before_mutation is not None:
            if self._session_control.has_active_tasks(loaded_session.id):
                raise RuntimeError(f"Session has active work in this process: {loaded_session.id}")
            interaction_records = await self._session_store.query_events(
                EventQuery(
                    session_id=loaded_session.id,
                    event_types=INTERACTION_LIFECYCLE_EVENT_TYPES,
                    order_by=EventOrder.SEQUENCE_DESC,
                    limit=1,
                )
            )
            if (
                not interaction_records
                or interaction_records[0].event.type in INTERACTION_TERMINAL_EVENT_TYPES
                or interaction_records[0].event.interaction_id is None
            ):
                raise RuntimeError(
                    "Pending tool recovery state has no open interaction. "
                    "Pre-interaction prerelease recovery state is unsupported."
                )
            await before_mutation()
        if (
            loaded_session.status in _RECOVERY_RESUMABLE_SESSION_STATUSES
            and not pending_operator_interruption
        ):
            original_pending_round = pending_round
            (
                loaded_session,
                checkpoint,
            ) = await self._reconcile_terminal_evidence_before_continuation(
                session=loaded_session,
                checkpoint=checkpoint,
            )
            pending_round = tool_round_recovery.pending_tool_round_from_checkpoint(
                checkpoint,
                redactor=self._secret_redactor,
                consume_on_rejection=True,
            )
            if pending_round is None or pending_round != original_pending_round:
                raise RuntimeError(
                    "Pending tool round changed while terminal evidence was reconciled."
                )
            pending_tool_call = approval_support.round_tool_call_for_recovery(
                pending_calls=pending_round.tool_calls,
                tool_call_id=request.tool_call_id,
            )
            execution_profile_snapshot = await self._validate_execution_profile_continuation(
                loaded_session,
                checkpoint,
                registered_agent,
                registered_provider,
                request.loop_policies,
                execution_profile_snapshot.profile,
                budget_policy=budget_policy_snapshot,
                request_budget_limits=invocation_semantics.budget_limits,
                structured_output=invocation_semantics.structured_output,
                thinking=invocation_semantics.thinking,
                max_steps=invocation_semantics.max_steps,
                limits=invocation_semantics.limits,
                retry_policy=invocation_semantics.retry_policy,
                invocation_semantics_available=True,
            )
        if self._session_control.has_active_tasks(loaded_session.id):
            raise RuntimeError(f"Session has active work in this process: {loaded_session.id}")
        # Reserve the in-process slot before awaiting the durable transition. The
        # check and registration are await-free, so another local recovery cannot
        # advance the run epoch while this claimant is waiting on storage.
        current_task = asyncio.current_task()
        if current_task is not None:
            self._session_control.register_active_task(
                loaded_session.id,
                current_task,
                task_id=None,
                task_started=False,
                task_finished=False,
            )
        interaction_id: str | None = None
        recovery_stream: AsyncGenerator[Event, None] | None = None
        authoritative_failure: BaseException | None = None
        try:
            interaction_id = await self._activate_latest_open_interaction(loaded_session.id)
            if interaction_id is None:
                raise RuntimeError(
                    "Pending tool recovery state has no open interaction. "
                    "Pre-interaction prerelease recovery state is unsupported."
                )
            recovery_stream = self.recover_tool_round(
                request=request,
                loaded_session=loaded_session,
                pending_round=pending_round,
                pending_tool_call=pending_tool_call,
                registered_agent=registered_agent,
                registered_provider=registered_provider,
                registered_environment=registered_environment,
                invocation_semantics=invocation_semantics,
                execution_profile_snapshot=execution_profile_snapshot,
                budget_policy=budget_policy_snapshot,
            )
            async for event in recovery_stream:
                yield event
        except BaseException as exc:
            authoritative_failure = exc
            raise
        finally:
            try:
                await self._cleanup_entrypoint_handoff(
                    stream=recovery_stream,
                    session_id=loaded_session.id,
                    registered_agent=registered_agent,
                    registered_environment=registered_environment,
                    authoritative_failure=authoritative_failure,
                    finalize_abandoned=False,
                    release_run_fence=False,
                    abort_environment_setup=False,
                    execution_profile=execution_profile_snapshot.profile,
                )
            finally:
                if interaction_id is not None:
                    _deactivate_session_interaction(loaded_session.id)
                if current_task is not None:
                    self._session_control.unregister_active_task(
                        loaded_session.id,
                        current_task,
                    )

    async def continue_user_input_resolution(
        self,
        *,
        response: UserInputResponse,
        session: Session,
        pending: PendingUserInput,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_provider: runtime_records.RegisteredProvider,
        registered_environment: runtime_records.RegisteredEnvironment | None,
        execution_profile_snapshot: ActiveInvocationExecutionProfile,
        budget_policy: BudgetPolicy | None,
        emit_resume_event: bool = True,
    ) -> AsyncGenerator[Event, None]:
        environment_name = _environment_name(registered_environment)
        tool_round_identity = ToolRoundIdentity(
            tool_round_id=pending.tool_round_id,
            model_step_id=pending.model_step_id,
            model_attempt_id=pending.model_attempt_id,
        )
        if pending.tool_exposure is not None:
            validate_resolved_tool_exposure_authority(
                pending.tool_exposure,
                registered_agent.tool_capabilities,
            )
        pending_cleared = False
        tool_outcomes: list[runtime_records.ToolCallOutcome] = []
        # Restore the original run's config persisted on the pending input. Explicit
        # values have already been checked against the frozen invocation profile.
        invocation_semantics = _effective_user_input_invocation_semantics(
            response=response,
            pending=pending,
            structured_output=_effective_user_input_structured_output(
                structured_output=response.structured_output,
                pending=pending,
            ),
            effective_retry_policy=self._effective_retry_policy,
        )
        effective_max_steps = invocation_semantics.max_steps
        effective_limits = invocation_semantics.limits
        effective_budget_limits = invocation_semantics.budget_limits
        effective_retry_policy = invocation_semantics.retry_policy
        continued_run_limit_accounting = pending.run_limit_accounting
        try:
            transcript = await self._session_store.load_transcript(session.id)
            resume_events = await self._session_store.load_events(session.id)
            if continued_run_limit_accounting is not None:
                continued_run_limit_accounting = rebase_run_limit_accounting_context(
                    continued_run_limit_accounting,
                    session_id=session.id,
                    limits=effective_limits,
                    budget_limits=request_budget_limits_for_session(
                        limits=effective_budget_limits,
                        agent_name=registered_agent.spec.name,
                        causal_budget_id=session.causal_budget_id,
                    ),
                    events=resume_events,
                    # Profile admission permits only an exact restatement of the
                    # frozen invocation semantics. It must not reset the durable
                    # accounting origin merely because the caller restated it.
                    reset_run_limits=False,
                    reset_budgets=False,
                    now=self._clock(),
                )
            factory_started_event = await self._environment_lifecycle.emit_factory_started(
                session=session,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
                execution_profile=execution_profile_snapshot.profile,
            )
            if factory_started_event is not None:
                yield factory_started_event
            factory_resolution = await self._environment_lifecycle.resolve_factory(
                session=session,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
                started_event=factory_started_event,
                operation=EnvironmentFactoryOperation.RECONNECT,
                execution_profile=execution_profile_snapshot.profile,
            )
            registered_environment = factory_resolution.registered_environment
            environment_name = _environment_name(registered_environment)
            for event in factory_resolution.events:
                yield event

            if factory_resolution.error is not None:
                raise factory_resolution.error
            if emit_resume_event:
                yield await self._event_writer.emit(
                    event_with_execution_profile_authority(
                        event_with_runtime_payload_authority(
                            Event(
                                type=EventType.SESSION_RESUMED,
                                session_id=session.id,
                                agent_name=registered_agent.spec.name,
                                environment_name=environment_name,
                                payload={
                                    **tool_round_identity.payload(),
                                    "interruption_type": _INTERRUPTION_TYPE_USER_INPUT_REQUIRED,
                                    "input_id": pending.input_id,
                                    "tool_call_id": pending.tool_call_id,
                                    "resolved_by": resolution_actor_payload(response.resolved_by),
                                },
                            ),
                            "model_step_id",
                            "model_attempt_id",
                            "tool_round_id",
                            "input_id",
                        ),
                        execution_profile_snapshot.profile,
                    )
                )
            binding_started_event = await self._environment_lifecycle.emit_binding_started(
                session=session,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
                execution_profile=execution_profile_snapshot.profile,
            )
            if binding_started_event is not None:
                yield binding_started_event
            binding_result = await self._environment_lifecycle.bind(
                session=session,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
                started_event=binding_started_event,
                execution_profile=execution_profile_snapshot.profile,
            )
            registered_environment = binding_result.registered_environment
            for event in binding_result.events:
                yield event
            if binding_result.error is not None:
                raise binding_result.error

            round_tool_calls = [
                runtime_records.ToolCallRequest(
                    id=pending_call.tool_call_id,
                    name=pending_call.tool_name,
                    arguments=copy_json_value(pending_call.arguments, "arguments"),
                )
                for pending_call in pending.tool_calls
            ]
            publish_arguments_as_unavailable = len(round_tool_calls) > 1
            base_round_redactor = self._tool_round_executor.redactor_for_tool_calls(
                registered_agent=registered_agent,
                tool_calls=round_tool_calls,
            )
            legacy_publication_scope = (
                pending.assistant_message_state == "quarantined"
                and pending.assistant_publication is None
            )
            persisted_secret_resolution_scope = (
                "unknown"
                if pending.assistant_publication is None
                else pending.assistant_publication.secret_resolution_scope
            )
            pause_secret_resolution_scope = invocation_secrets.continuation_secret_resolution_scope(
                persisted_secret_resolution_scope,
                registered_environment,
            )
            defer_round_terminals = (
                len(round_tool_calls) > 1 and pause_secret_resolution_scope != "static"
            ) or any(
                registered is not None and registered.workspace_mutation
                for registered in (
                    registered_agent.tools.get(tool_call.name) for tool_call in round_tool_calls
                )
            )
            publication_coordinator = (
                _ToolRoundPublicationCoordinator(
                    session_id=session.id,
                    tool_round_identity=tool_round_identity,
                    session_store=self._session_store,
                    redactor=base_round_redactor,
                    execution_profile=execution_profile_snapshot.profile,
                    tool_exposure=pending.tool_exposure,
                )
                if defer_round_terminals
                else None
            )
            staged_hook_modes: dict[str, tuple[bool, bool]] = {}

            async def record_round_publication_snapshot(
                tool_call_id: str,
                snapshot: invocation_secrets.InvocationPublicationSnapshot,
            ) -> None:
                if publication_coordinator is not None:
                    await publication_coordinator.seal_call(
                        tool_call_id=tool_call_id,
                        snapshot=snapshot,
                    )
                    return
                await self._session_store.transform_checkpoint(
                    session.id,
                    tool_round_recovery.assistant_publication_snapshot_transform(
                        tool_round_identity=tool_round_identity,
                        tool_call_id=tool_call_id,
                        redactor=snapshot.redactor,
                        unsafe_output=snapshot.secret_scope_incomplete,
                    ),
                )

            async def record_round_redactor(
                tool_call_id: str,
                snapshot: InvocationRedactorSnapshot,
            ) -> None:
                if publication_coordinator is None:
                    raise AssertionError("Continuation redactor observer has no coordinator.")
                await publication_coordinator.register_redactor(
                    tool_call_id=tool_call_id,
                    redactor=snapshot.redactor,
                )

            async def stage_round_terminal(
                event: Event,
                outcome: runtime_records.ToolCallOutcome,
                allow_modification: bool,
                publish_before_hooks: bool,
                snapshot: invocation_secrets.InvocationPublicationSnapshot,
            ) -> Event:
                if publication_coordinator is None:
                    raise AssertionError("Continuation terminal staging has no coordinator.")
                prepared_event = self._event_writer.prepare(event)
                exposure_blocked = (
                    prepared_event.type is EventType.TOOL_CALL_BLOCKED
                    and prepared_event.payload.get("blocked_by") == "tool_exposure"
                )
                staged = await publication_coordinator.stage_terminal(
                    tool_call_id=outcome.call.id,
                    event=prepared_event,
                    snapshot=snapshot,
                    hooks_state=(
                        "completed"
                        if exposure_blocked
                        else (
                            "observational"
                            if publish_before_hooks
                            else ("pending" if allow_modification else "finalized")
                        )
                    ),
                )
                staged_hook_modes[outcome.call.id] = (
                    allow_modification,
                    publish_before_hooks,
                )
                return staged

            async def record_round_workspace_capture(event: Event) -> Event:
                if publication_coordinator is None:
                    raise AssertionError("Workspace capture recording has no coordinator.")
                return await publication_coordinator.record_workspace_capture(event)

            async def record_static_publication_scope(
                tool_call_id: str,
                *,
                execution_scope_unknown: bool = False,
            ) -> None:
                await record_round_publication_snapshot(
                    tool_call_id,
                    invocation_secrets.InvocationPublicationSnapshot(
                        redactor=base_round_redactor,
                        unsafe_output=execution_scope_unknown,
                        secret_scope_incomplete=execution_scope_unknown,
                    ),
                )

            # Reuse any outcomes already recorded for this round — e.g. a prior resume attempt
            # that ran some tools before a mid-resume failure — so a retry never re-executes a
            # side-effecting tool. The round was already projected against limits at pause time;
            # its remaining tools run on resume without a fresh budget projection (so the user's
            # answer is never discarded by a limit check here).
            recorded_outcomes = approval_support.recorded_round_tool_outcomes(
                events=resume_events,
                pending_calls=pending.tool_calls,
                input_id=pending.input_id,
                tool_round_identity=tool_round_identity,
                staged_terminals=pending.staged_terminals,
            )
            restarted_staged_ids = (
                await self._fence_restarted_continuation_stages(
                    coordinator=publication_coordinator,
                    session=session,
                    registered_agent=registered_agent,
                    environment_name=environment_name,
                    tool_calls=round_tool_calls,
                    recorded_ids=set(recorded_outcomes),
                    pause_payload={"input_id": pending.input_id},
                    idempotency_options={"pause_id": pending.input_id},
                    execution_profile=execution_profile_snapshot.profile,
                )
                if publication_coordinator is not None
                else set()
            )
            pending_by_id = {call.tool_call_id: call for call in pending.tool_calls}

            # Build the round's outcomes in model order: a call already recorded (retry) is
            # reused; the answered ask_user call gets the injected answer; every other allowed
            # call executes now (none ran before the pause); a denied call is blocked.
            for tool_call in round_tool_calls:
                recorded_outcome = recorded_outcomes.get(tool_call.id)
                if recorded_outcome is not None:
                    # A terminal record from an earlier process proves the
                    # call is complete, but an additive pre-field checkpoint
                    # does not prove which invocation secrets it resolved.
                    await record_static_publication_scope(
                        tool_call.id,
                        execution_scope_unknown=legacy_publication_scope,
                    )
                    tool_outcomes.append(recorded_outcome)
                    continue
                if tool_call.id in restarted_staged_ids:
                    continue

                pending_call = pending_by_id[tool_call.id]
                registered_tool = registered_agent.tools.get(tool_call.name)
                policy_evidence = approval_support.effective_tool_policy_evidence(pending_call)
                policy_result = approval_support.policy_result_from_pending_tool_call(pending_call)
                if tool_call.id == pending.tool_call_id:
                    if policy_evidence is not ToolPolicyEvidence.AUTHORITATIVE:
                        raise RuntimeError(
                            "Pending user-input call has no authoritative policy decision."
                        )
                    idempotency_key = tool_execution.tool_idempotency_key(
                        session_id=session.id,
                        tool_round_id=tool_round_identity.tool_round_id,
                        tool_call_id=tool_call.id,
                        pause_id=pending.input_id,
                    )
                    result = ToolResult(
                        content=response.answer,
                        structured=response.structured,
                        artifacts=response.artifacts,
                        is_error=False,
                    )
                    await record_static_publication_scope(tool_call.id)
                    started_payload: dict[str, Any] = {
                        **tool_round_identity.payload(),
                        "tool_call_id": tool_call.id,
                        "idempotency_key": idempotency_key,
                        **tool_argument_publication.quarantined_argument_fields(),
                        "input_id": pending.input_id,
                    }
                    if registered_tool is not None:
                        started_payload["effect"] = registered_tool.effect.value
                    yield await self._event_writer.emit(
                        event_with_execution_profile_authority(
                            event_with_runtime_payload_authority(
                                Event(
                                    type=EventType.TOOL_CALL_STARTED,
                                    session_id=session.id,
                                    agent_name=registered_agent.spec.name,
                                    environment_name=environment_name,
                                    tool_name=tool_call.name,
                                    payload=started_payload,
                                ),
                                "model_step_id",
                                "model_attempt_id",
                                "tool_round_id",
                                "input_id",
                            ),
                            execution_profile_snapshot.profile,
                        )
                    )
                    async for (
                        event,
                        outcome,
                    ) in self._tool_round_executor.emit_tool_call_result_with_hooks(
                        event=Event(
                            type=EventType.TOOL_CALL_COMPLETED,
                            session_id=session.id,
                            agent_name=registered_agent.spec.name,
                            environment_name=environment_name,
                            tool_name=tool_call.name,
                            payload={
                                **tool_round_identity.payload(),
                                "tool_call_id": tool_call.id,
                                "idempotency_key": idempotency_key,
                                "input_id": pending.input_id,
                                "resolved_by": resolution_actor_payload(response.resolved_by),
                                "result": result.model_dump(),
                            },
                        ),
                        session=session,
                        registered_agent=registered_agent,
                        registered_environment=registered_environment,
                        tool_call=tool_call,
                        result=result,
                        task_id=pending.task_id,
                        execution_profile=execution_profile_snapshot.profile,
                        redactor=(
                            None
                            if publication_coordinator is None
                            else publication_coordinator.redactor
                        ),
                        output_redactor=(
                            None
                            if publication_coordinator is None
                            else publication_coordinator.redactor
                        ),
                        deferred_terminal_stager=(
                            None if publication_coordinator is None else stage_round_terminal
                        ),
                        publication_snapshot=invocation_secrets.InvocationPublicationSnapshot(
                            redactor=base_round_redactor,
                            unsafe_output=False,
                            secret_scope_incomplete=False,
                        ),
                    ):
                        yield event
                        if outcome is not None:
                            tool_outcomes.append(outcome)
                    continue

                call_taint_labels = approval_support.taint_labels_from_pending_tool_call(
                    pending_call
                )
                # `ToolRoundExecutor.execute_tool_call(check_policy=False)` does not re-enforce
                # the decision, so a DENY must be blocked here explicitly (mirroring the approval
                # resume) — otherwise a policy-denied sibling would execute. REQUIRE_APPROVAL
                # cannot occur: it would have preempted the ask_user pause with an approval pause.
                if (
                    policy_evidence is ToolPolicyEvidence.AUTHORITATIVE
                    and policy_result is not None
                    and policy_result.decision == ToolPolicyDecision.DENY
                ):
                    await record_static_publication_scope(tool_call.id)
                    public_policy_result = approval_support.public_policy_denial_result(
                        secret_resolution_scope=pause_secret_resolution_scope,
                        policy_result=policy_result,
                        publish_arguments=(
                            registered_tool is not None and registered_tool.publish_arguments
                        ),
                    )
                    reason = tool_execution.policy_denial_reason(public_policy_result)
                    blocked_result = tool_execution.blocked_tool_result(
                        public_policy_result,
                        reason=reason,
                    )
                    idempotency_key = tool_execution.tool_idempotency_key(
                        session_id=session.id,
                        tool_round_id=tool_round_identity.tool_round_id,
                        tool_call_id=tool_call.id,
                        pause_id=pending.input_id,
                    )
                    async for (
                        event,
                        outcome,
                    ) in self._tool_round_executor.emit_tool_call_result_with_hooks(
                        event=Event(
                            type=EventType.TOOL_CALL_BLOCKED,
                            session_id=session.id,
                            agent_name=registered_agent.spec.name,
                            environment_name=environment_name,
                            tool_name=tool_call.name,
                            payload={
                                **tool_round_identity.payload(),
                                "tool_call_id": tool_call.id,
                                "idempotency_key": idempotency_key,
                                "input_id": pending.input_id,
                                **policy_denial_payload_fields(
                                    tool_name=tool_call.name,
                                    denied_by=_TOOL_POLICY_DENIAL_SOURCE,
                                    decision=public_policy_result.decision.value,
                                    reason=reason,
                                    metadata=public_policy_result.metadata,
                                ),
                                "result": blocked_result.model_dump(),
                            },
                        ),
                        session=session,
                        registered_agent=registered_agent,
                        registered_environment=registered_environment,
                        tool_call=tool_call,
                        result=blocked_result,
                        task_id=pending.task_id,
                        execution_profile=execution_profile_snapshot.profile,
                        redactor=(
                            None
                            if publication_coordinator is None
                            else publication_coordinator.redactor
                        ),
                        output_redactor=(
                            None
                            if publication_coordinator is None
                            else publication_coordinator.redactor
                        ),
                        deferred_terminal_stager=(
                            None if publication_coordinator is None else stage_round_terminal
                        ),
                        publication_snapshot=invocation_secrets.InvocationPublicationSnapshot(
                            redactor=base_round_redactor,
                            unsafe_output=False,
                            secret_scope_incomplete=False,
                        ),
                    ):
                        yield event
                        if outcome is not None:
                            tool_outcomes.append(outcome)
                    continue

                if policy_evidence in {
                    ToolPolicyEvidence.AMBIGUOUS,
                    ToolPolicyEvidence.UNREGISTERED,
                    ToolPolicyEvidence.UNEXPOSED,
                }:
                    await record_static_publication_scope(tool_call.id)
                    async for (
                        event,
                        outcome,
                    ) in self._emit_non_authoritative_policy_call(
                        session=session,
                        registered_agent=registered_agent,
                        registered_environment=registered_environment,
                        environment_name=environment_name,
                        tool_call=tool_call,
                        policy_evidence=policy_evidence,
                        tool_exposure=pending.tool_exposure,
                        tool_round_identity=tool_round_identity,
                        task_id=pending.task_id,
                        execution_profile=execution_profile_snapshot.profile,
                        input_id=pending.input_id,
                        deferred_terminal_stager=(
                            None if publication_coordinator is None else stage_round_terminal
                        ),
                        publication_snapshot=invocation_secrets.InvocationPublicationSnapshot(
                            redactor=base_round_redactor,
                            unsafe_output=False,
                            secret_scope_incomplete=False,
                        ),
                    ):
                        yield event
                        if outcome is not None:
                            tool_outcomes.append(outcome)
                    continue

                if policy_evidence is not ToolPolicyEvidence.AUTHORITATIVE:
                    raise RuntimeError(
                        "Pending user-input sibling has no executable policy authority."
                    )

                async for event, outcome in self._tool_round_executor.execute_tool_call(
                    session=session,
                    registered_agent=registered_agent,
                    registered_environment=registered_environment,
                    tool_call=tool_call,
                    request_metadata=response.metadata,
                    task_id=pending.task_id,
                    execution_profile=execution_profile_snapshot.profile,
                    check_policy=False,
                    policy_result=policy_result,
                    policy_output_secret_resolution_scope=pause_secret_resolution_scope,
                    input_id=pending.input_id,
                    tool_round_identity=tool_round_identity,
                    model_step=pending.model_step,
                    taint_labels=call_taint_labels,
                    publish_arguments_as_unavailable=publish_arguments_as_unavailable,
                    deferred_terminal_stager=(
                        None if publication_coordinator is None else stage_round_terminal
                    ),
                    deferred_terminal_capture_recorder=(
                        None if publication_coordinator is None else record_round_workspace_capture
                    ),
                    resolved_redactor_observer=(
                        None if publication_coordinator is None else record_round_redactor
                    ),
                    publication_snapshot_observer=record_round_publication_snapshot,
                ):
                    yield event
                    if outcome is not None:
                        tool_outcomes.append(outcome)

            if publication_coordinator is not None:
                expected_staged_ids = {
                    call.id for call in round_tool_calls if call.id not in recorded_outcomes
                } | (set(recorded_outcomes) & restarted_staged_ids)
                current_stages = tool_round_recovery.checkpoint_staged_terminals(
                    await self._session_store.load_checkpoint(session.id),
                    tool_round_identity=tool_round_identity,
                )
                if {item.tool_call_id for item in current_stages} != expected_staged_ids:
                    raise RuntimeError(
                        "Dynamic user-input continuation requires one private terminal "
                        "stage per unresolved call."
                    )
                async for event, outcome in self._publish_continuation_staged_terminals(
                    coordinator=publication_coordinator,
                    session=session,
                    registered_agent=registered_agent,
                    registered_environment=registered_environment,
                    tool_calls=round_tool_calls,
                    task_id=pending.task_id,
                    execution_profile=execution_profile_snapshot.profile,
                    hook_modes=staged_hook_modes,
                    pause_authority={"input_id": pending.input_id},
                    already_published_ids=set(recorded_outcomes),
                ):
                    yield event
                    if outcome is not None:
                        tool_outcomes.append(outcome)
                outcomes_by_id = {outcome.call.id: outcome for outcome in tool_outcomes}
                tool_outcomes = [outcomes_by_id[call.id] for call in round_tool_calls]

            # The resume executes the round's tools sequentially in model order, so the outcome
            # list already lines up with the assistant tool-call parts.
            durable_round = pending_actions.pending_action_evidence_round_from_checkpoint(
                await self._session_store.load_checkpoint(session.id)
            )
            if (
                durable_round is None
                or tool_round_recovery.pending_tool_round_identity(durable_round)
                != tool_round_identity
            ):
                raise RuntimeError("Pending user-input round changed before transcript closure.")
            tool_result_messages = transcript_helpers.tool_result_messages(
                tool_outcomes,
                tool_round_identity=tool_round_identity,
            )
            transcript_messages = list(tool_result_messages)
            if durable_round.assistant_message_state == "quarantined":
                transcript_messages.insert(
                    0,
                    transcript_helpers.assistant_message_with_projected_tool_arguments(
                        tool_round_recovery.ready_assistant_publication_message(durable_round),
                        tool_outcomes,
                    ),
                )
            transcript.extend(transcript_messages)
            cleared_checkpoint = await self._checkpoint_without_pending_user_input(session.id)
            await self._session_store.append_transcript_messages_and_transform_checkpoint(
                session.id,
                transcript_messages,
                self._checkpoint_transform(cleared_checkpoint),
            )
            pending_cleared = True

            session_stream = self._run_session(
                RecoverySessionRunRequest(
                    session=session,
                    registered_agent=registered_agent,
                    registered_provider=registered_provider,
                    registered_environment=registered_environment,
                    active_invocation_profile=_rebound_active_invocation_profile(
                        session,
                        execution_profile_snapshot,
                    ),
                    messages=transcript,
                    messages_to_append=[],
                    max_steps=effective_max_steps,
                    limits=effective_limits,
                    budget_limits=effective_budget_limits,
                    budget_policy=copy_budget_policy(budget_policy),
                    retry_policy=effective_retry_policy,
                    structured_output=invocation_semantics.structured_output,
                    thinking=invocation_semantics.thinking,
                    request_loop_policies=response.loop_policies,
                    request_metadata=response.metadata,
                    task_id=pending.task_id,
                    task_worker_id=None,
                    start_event_type=None,
                    start_event_payload={},
                    start_task_on_enter=False,
                    release_run_fence_on_exit=False,
                    run_limit_accounting=continued_run_limit_accounting,
                    previous_tool_exposure_profile_id=(
                        _continued_tool_exposure_profile_id(pending.tool_exposure)
                    ),
                )
            )
            forwarded_stream = self._session_control.stream_with_out_of_band_events(
                session.id,
                session_stream,
            )
            try:
                async for event in forwarded_stream:
                    yield event
            except GeneratorExit:
                await forwarded_stream.aclose()
                raise
        except Exception as exc:
            if not pending_cleared:
                # The pending_user_input checkpoint is still present, so restore the resumable
                # INTERRUPTED state and emit a terminal event for closure (a SESSION_RESUMED was
                # already emitted). The caller can retry resolve_user_input; recorded outcomes
                # prevent re-running a tool that already completed. A tool that started with no
                # terminal (a crash mid-tool) cannot be re-run safely — flag it as needing manual
                # recovery so the retry is not a silent double-execution.
                # Carry the failure so a caller can distinguish "your answer failed, retry" from a
                # fresh pause (whose interrupted event has no error fields).
                payload: dict[str, Any] = {
                    **exception_failure_payload(
                        exc,
                        redactor=self._secret_redactor,
                    ),
                    **tool_round_identity.payload(),
                    "interruption_type": _INTERRUPTION_TYPE_USER_INPUT_REQUIRED,
                    **pending_user_input_interruption_payload(pending),
                }
                if isinstance(exc, approval_support.RoundToolManualRecoveryRequired):
                    payload["manual_recovery_required"] = True
                    payload["tool_call_id"] = exc.tool_call_id
                    payload["tool_name"] = exc.tool_name
                if isinstance(exc, resume_ledger.ToolCallEvidenceConflict):
                    payload[resume_ledger.TOOL_EVIDENCE_CONFLICT_PAYLOAD_KEY] = True
                session = await self._session_store.update_status(
                    session.id, SessionStatus.INTERRUPTED
                )
                async for event in self._emit_terminal_event_with_hooks(
                    RecoveryTerminalEventRequest(
                        event=event_with_execution_profile_authority(
                            Event(
                                type=EventType.SESSION_INTERRUPTED,
                                session_id=session.id,
                                agent_name=registered_agent.spec.name,
                                environment_name=environment_name,
                                payload=payload,
                            ),
                            execution_profile_snapshot.profile,
                        ),
                        phase=RuntimeHookPhase.AFTER_SESSION_INTERRUPTED,
                        session=session,
                        registered_agent=registered_agent,
                        registered_environment=registered_environment,
                        execution_profile=execution_profile_snapshot.profile,
                    )
                ):
                    yield event
                return
            raise

    async def _checkpoint_without_pending_user_input(
        self,
        session_id: str,
    ) -> dict[str, Any]:
        checkpoint = await self._session_store.load_checkpoint(session_id)
        checkpoint = {} if checkpoint is None else copy_json_value(checkpoint, "checkpoint")
        checkpoint.pop(PENDING_USER_INPUT_CHECKPOINT_KEY, None)
        return checkpoint

    async def _publish_continuation_staged_terminals(
        self,
        *,
        coordinator: _ToolRoundPublicationCoordinator,
        session: Session,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_environment: runtime_records.RegisteredEnvironment | None,
        tool_calls: list[runtime_records.ToolCallRequest],
        task_id: str | None,
        execution_profile: ExecutionProfileIdentity,
        hook_modes: dict[str, tuple[bool, bool]],
        pause_authority: dict[str, str],
        already_published_ids: set[str],
    ) -> AsyncIterator[tuple[Event, runtime_records.ToolCallOutcome | None]]:
        """Publish continuation results only after the round scope is final."""

        identity = coordinator.tool_round_identity
        checkpoint = await self._session_store.load_checkpoint(session.id)
        staged_records = tool_round_recovery.checkpoint_staged_terminals(
            checkpoint,
            tool_round_identity=identity,
        )
        staged_by_id = {item.tool_call_id: item for item in staged_records}
        calls_by_id = {call.id: call for call in tool_calls}
        if not set(staged_by_id).issubset(calls_by_id):
            raise RuntimeError("Continuation stages contain a call outside their tool round.")
        if not already_published_ids.issubset(calls_by_id):
            raise RuntimeError("Published continuation evidence names an unknown tool call.")

        async def complete_hooks(event: Event) -> Event:
            return await coordinator.complete_terminal_hooks(event)

        async def record_projection(event: Event) -> Event:
            return await coordinator.record_projected_terminal(event)

        for tool_call in tool_calls:
            staged = staged_by_id.get(tool_call.id)
            if staged is None:
                continue
            if tool_call.id in already_published_ids:
                continue
            staged_event = coordinator.restore_staged_event_authority(staged.event)
            authority_fields: list[str] = []
            for field_name, expected_value in pause_authority.items():
                if staged_event.payload.get(field_name) != expected_value:
                    raise RuntimeError(
                        "Continuation stage conflicts with its pending pause identity."
                    )
                authority_fields.append(field_name)
            if authority_fields:
                staged_event = event_with_runtime_payload_authority(
                    staged_event,
                    *authority_fields,
                )
            result_payload = staged_event.payload.get("result")
            if type(result_payload) is not dict:
                raise RuntimeError("Continuation stage lost its tool result.")
            result = tool_results.tool_result_from_payload(result_payload)
            argument_projection, hook_argument_projection = _staged_terminal_argument_projections(
                staged_event
            )
            hooks_already_completed = staged.hooks_state == "completed"
            allow_modification, publish_before_hooks = (
                (False, False)
                if hooks_already_completed
                else hook_modes.get(
                    tool_call.id,
                    (
                        staged.hooks_state == "pending",
                        staged.hooks_state == "observational",
                    ),
                )
            )
            async for event, outcome in self._tool_round_executor.emit_tool_call_result_with_hooks(
                event=staged_event,
                session=session,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
                tool_call=tool_call,
                result=result,
                task_id=task_id,
                execution_profile=execution_profile,
                redactor=coordinator.redactor,
                output_redactor=coordinator.redactor,
                argument_projection=argument_projection,
                hook_argument_projection=hook_argument_projection,
                allow_modification=allow_modification,
                publish_before_hooks=publish_before_hooks,
                deferred_terminal_projection_recorder=(
                    record_projection
                    if publish_before_hooks and not hooks_already_completed
                    else None
                ),
                deferred_terminal_finalizer=(None if hooks_already_completed else complete_hooks),
                hooks_already_completed=hooks_already_completed,
            ):
                yield event, outcome

    async def _fence_restarted_continuation_stages(
        self,
        *,
        coordinator: _ToolRoundPublicationCoordinator,
        session: Session,
        registered_agent: runtime_records.RegisteredAgentState,
        environment_name: str | None,
        tool_calls: list[runtime_records.ToolCallRequest],
        recorded_ids: set[str],
        pause_payload: dict[str, str],
        idempotency_options: dict[str, str],
        execution_profile: ExecutionProfileIdentity,
    ) -> set[str]:
        """Close a partially staged continuation without re-executing siblings."""

        checkpoint = await self._session_store.load_checkpoint(session.id)
        stages = tool_round_recovery.checkpoint_staged_terminals(
            checkpoint,
            tool_round_identity=coordinator.tool_round_identity,
        )
        if not stages:
            return set()
        staged_ids = {item.tool_call_id for item in stages}
        for staged in stages:
            if staged.tool_call_id in recorded_ids or staged.hooks_state == "completed":
                continue
            unavailable = tool_round_recovery.hook_scope_unavailable_recovery_event(staged.event)
            await self._session_store.transform_checkpoint(
                session.id,
                tool_round_recovery.completed_staged_terminal_transform(
                    tool_round_identity=coordinator.tool_round_identity,
                    event=unavailable,
                ),
            )

        for tool_call in tool_calls:
            if tool_call.id in recorded_ids or tool_call.id in staged_ids:
                continue
            result = ToolResult(
                content=(
                    "Tool call was not executed because recovery could not reconstruct "
                    "the complete sibling invocation-secret scope."
                ),
                structured={
                    "error": "invalid_tool_output",
                    "executed": False,
                    "outcome_unknown": False,
                    "recovered": True,
                    "reason": "continuation_secret_scope_unavailable",
                },
                is_error=True,
            )
            event = event_with_execution_profile_authority(
                Event(
                    type=EventType.TOOL_CALL_BLOCKED,
                    session_id=session.id,
                    agent_name=registered_agent.spec.name,
                    environment_name=environment_name,
                    tool_name=tool_call.name,
                    payload={
                        **coordinator.tool_round_identity.payload(),
                        **pause_payload,
                        "tool_call_id": tool_call.id,
                        "idempotency_key": tool_execution.tool_idempotency_key(
                            session_id=session.id,
                            tool_round_id=coordinator.tool_round_identity.tool_round_id,
                            tool_call_id=tool_call.id,
                            **idempotency_options,
                        ),
                        "recovered": True,
                        "result": result.model_dump(mode="json"),
                    },
                ),
                execution_profile,
            )
            staged_event = await coordinator.stage_terminal(
                tool_call_id=tool_call.id,
                event=self._event_writer.prepare(event),
                snapshot=invocation_secrets.InvocationPublicationSnapshot(
                    redactor=coordinator.redactor,
                    unsafe_output=False,
                    secret_scope_incomplete=False,
                ),
                hooks_state="finalized",
            )
            await coordinator.complete_terminal_hooks(staged_event)
            staged_ids.add(tool_call.id)
        return staged_ids

    async def _emit_non_authoritative_policy_call(
        self,
        *,
        session: Session,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_environment: runtime_records.RegisteredEnvironment | None,
        environment_name: str | None,
        tool_call: runtime_records.ToolCallRequest,
        policy_evidence: ToolPolicyEvidence,
        tool_exposure: ResolvedToolExposureAuthority | None,
        tool_round_identity: ToolRoundIdentity,
        task_id: str | None,
        execution_profile: ExecutionProfileIdentity,
        approval_id: str | None = None,
        input_id: str | None = None,
        requested_decision: ToolApprovalDecision | None = None,
        resolved_by_payload: dict[str, Any] | None = None,
        resolution_reason: str | None = None,
        resolution_metadata: dict[str, Any] | None = None,
        deferred_terminal_stager: DeferredTerminalStager | None = None,
        publication_snapshot: invocation_secrets.InvocationPublicationSnapshot | None = None,
    ) -> AsyncIterator[tuple[Event, runtime_records.ToolCallOutcome | None]]:
        """Close a call that lacks positive policy authority without dispatch."""

        if (approval_id is None) == (input_id is None):
            raise TypeError("Exactly one approval or user-input identity is required.")
        if policy_evidence is ToolPolicyEvidence.UNEXPOSED:
            async for event, outcome in self._tool_round_executor.execute_tool_call(
                session=session,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
                tool_call=tool_call,
                request_metadata={},
                task_id=task_id,
                execution_profile=execution_profile,
                check_policy=False,
                emit_started=False,
                policy_evidence=policy_evidence,
                tool_exposure=tool_exposure,
                approval_id=approval_id,
                input_id=input_id,
                tool_round_identity=tool_round_identity,
                deferred_terminal_stager=deferred_terminal_stager,
            ):
                yield event, outcome
            return
        evidence_payload: dict[str, Any]
        structured: dict[str, Any]
        if policy_evidence is ToolPolicyEvidence.AMBIGUOUS:
            event_type = EventType.TOOL_CALL_BLOCKED
            reason = (
                "Tool policy evaluation did not produce a durable decision; "
                "the call was not executed."
            )
            evidence_payload = {
                "decision": "ambiguous",
                "blocked_by": "policy_evaluation_ambiguous",
                "reason": reason,
            }
            structured = {
                "decision": "ambiguous",
                "blocked_by": "policy_evaluation_ambiguous",
            }
        elif policy_evidence is ToolPolicyEvidence.UNREGISTERED:
            event_type = EventType.TOOL_CALL_FAILED
            reason = f"Tool was not registered when the policy plan was recorded: {tool_call.name}"
            evidence_payload = {
                "registration_state": "unregistered_at_policy_plan",
            }
            structured = {
                "registration_state": "unregistered_at_policy_plan",
            }
        else:
            raise ValueError(
                "Non-authoritative closure requires ambiguous, unregistered, or unexposed evidence."
            )

        pause_payload: dict[str, Any]
        idempotency_options: dict[str, str]
        if approval_id is not None:
            pause_payload = {"approval_id": approval_id}
            idempotency_options = {"approval_id": approval_id}
            if requested_decision is not None:
                evidence_payload["requested_decision"] = requested_decision.value
                evidence_payload["resolution_reason"] = resolution_reason
                evidence_payload.update(
                    approval_support.bounded_resolution_metadata_payload(
                        {} if resolution_metadata is None else resolution_metadata,
                        redactor=self._secret_redactor,
                    )
                )
            evidence_payload["resolved_by"] = resolved_by_payload
        else:
            assert input_id is not None
            pause_payload = {"input_id": input_id}
            idempotency_options = {"pause_id": input_id}

        result = ToolResult(
            content=reason,
            structured={
                **tool_round_identity.payload(),
                **pause_payload,
                "tool_call_id": tool_call.id,
                "tool_name": tool_call.name,
                **structured,
            },
            is_error=True,
        )
        idempotency_key = tool_execution.tool_idempotency_key(
            session_id=session.id,
            tool_round_id=tool_round_identity.tool_round_id,
            tool_call_id=tool_call.id,
            **idempotency_options,
        )
        async for event, outcome in self._tool_round_executor.emit_tool_call_result_with_hooks(
            event=Event(
                type=event_type,
                session_id=session.id,
                agent_name=registered_agent.spec.name,
                environment_name=environment_name,
                tool_name=tool_call.name,
                payload={
                    **tool_round_identity.payload(),
                    **pause_payload,
                    "tool_call_id": tool_call.id,
                    "idempotency_key": idempotency_key,
                    **evidence_payload,
                    "result": result.model_dump(),
                },
            ),
            session=session,
            registered_agent=registered_agent,
            registered_environment=registered_environment,
            tool_call=tool_call,
            result=result,
            task_id=task_id,
            execution_profile=execution_profile,
            deferred_terminal_stager=deferred_terminal_stager,
            publication_snapshot=publication_snapshot,
        ):
            yield event, outcome

    async def continue_tool_approval_resolution(
        self,
        *,
        request: ToolApprovalRequest,
        session: Session,
        pending_approval: PendingToolApproval,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_provider: runtime_records.RegisteredProvider,
        registered_environment: runtime_records.RegisteredEnvironment | None,
        execution_profile_snapshot: ActiveInvocationExecutionProfile,
        budget_policy: BudgetPolicy | None,
        deferred_messages: list[Message] | None = None,
        emit_resume_event: bool = True,
        enforce_expiry: bool = True,
        claimed_resolution_intent: approval_support.ApprovalResolutionIntent | None = None,
        recovery_closure_only: bool = False,
    ) -> AsyncGenerator[Event, None]:
        environment_name = _environment_name(registered_environment)
        tool_round_identity = ToolRoundIdentity(
            tool_round_id=pending_approval.tool_round_id,
            model_step_id=pending_approval.model_step_id,
            model_attempt_id=pending_approval.model_attempt_id,
        )
        pending_approval_cleared = False
        tool_outcomes: list[runtime_records.ToolCallOutcome] = []
        expired = False
        original_resolution_decision = request.decision
        resolution_request_digest = approval_support.approval_resolution_request_digest(request)
        deferred_messages = (
            []
            if deferred_messages is None
            else [detach_message(message) for message in deferred_messages]
        )
        # Restore the original run's config persisted on the pending approval.
        # Explicit values have already been checked against the frozen profile.
        invocation_semantics = _effective_approval_invocation_semantics(
            request=request,
            pending_approval=pending_approval,
            structured_output=_effective_approval_structured_output(
                structured_output=request.structured_output,
                pending_approval=pending_approval,
            ),
            effective_retry_policy=self._effective_retry_policy,
        )
        effective_max_steps = invocation_semantics.max_steps
        effective_limits = invocation_semantics.limits
        effective_budget_limits = invocation_semantics.budget_limits
        effective_retry_policy = invocation_semantics.retry_policy
        continued_run_limit_accounting = pending_approval.run_limit_accounting
        try:
            transcript_snapshot = await self._session_store.load_transcript_snapshot(session.id)
            try:
                transcript = [
                    detach_message(record.message) for record in transcript_snapshot.records
                ]
                approval_transcript_cursor = transcript_snapshot.cursor
            finally:
                del transcript_snapshot
            approval_events = await self._session_store.load_events(session.id)
            resolved_budget_limits = request_budget_limits_for_session(
                limits=effective_budget_limits,
                agent_name=registered_agent.spec.name,
                causal_budget_id=session.causal_budget_id,
            )
            if continued_run_limit_accounting is not None:
                continued_run_limit_accounting = rebase_run_limit_accounting_context(
                    continued_run_limit_accounting,
                    session_id=session.id,
                    limits=effective_limits,
                    budget_limits=resolved_budget_limits,
                    events=approval_events,
                    # Profile admission permits only an exact restatement of the
                    # frozen invocation semantics. It must not reset the durable
                    # accounting origin merely because the caller restated it.
                    reset_run_limits=False,
                    reset_budgets=False,
                    now=self._clock(),
                )
            history = approval_support.approval_resolution_history(
                events=approval_events,
                approval=pending_approval,
            )
            # Expiry gates the FIRST grant only: a retry of an approval that
            # already has granted or executed activity was authorized
            # in-window before a crash, so coercing it to a denial would
            # contradict the recorded grant (and trip validate_retry_decision).
            if (
                enforce_expiry
                and approval_support.pending_approval_expired(pending_approval, self._clock())
                and not history.has_granted_activity
            ):
                expired = True
                # Captured before the coercion below replaces them on the request.
                requested_decision = request.decision
                triggered_by = request.resolved_by
                assert pending_approval.expires_at is not None
                expired_at_iso = pending_approval.expires_at.isoformat()
                request = ToolApprovalRequest(
                    session_id=request.session_id,
                    approval_id=request.approval_id,
                    tool_round_id=request.tool_round_id,
                    tool_call_id=request.tool_call_id,
                    decision=ToolApprovalDecision.DENY,
                    reason=f"Tool approval expired at {expired_at_iso}.",
                    metadata=copy_json_value(request.metadata, "metadata"),
                    resolved_by=expiry_resolution_actor(),
                    max_steps=request.max_steps,
                    limits=request.limits,
                    budget_limits=request.budget_limits,
                    retry_policy=request.retry_policy,
                    structured_output=request.structured_output,
                    thinking=request.thinking,
                    loop_policies=request.loop_policies,
                )
            approval_support.validate_retry_decision(
                history=history,
                approval=pending_approval,
                decision=request.decision,
            )
            resolved_by_payload = resolution_actor_payload(request.resolved_by)
            current_checkpoint = await self._session_store.load_checkpoint(session.id)
            publication_round = tool_round_recovery.pending_tool_round_from_checkpoint(
                current_checkpoint,
                redactor=self._secret_redactor,
                consume_on_rejection=True,
            )
            if (
                publication_round is None
                or tool_round_recovery.pending_tool_round_identity(publication_round)
                != tool_round_identity
            ):
                raise RuntimeError(
                    "Pending approval round changed before publication-scope recovery."
                )
            if publication_round.tool_exposure is not None:
                validate_resolved_tool_exposure_authority(
                    publication_round.tool_exposure,
                    registered_agent.tool_capabilities,
                )
            recorded_outcomes = approval_support.recorded_tool_outcomes(
                events=approval_events,
                approval=pending_approval,
                staged_terminals=publication_round.staged_terminals,
            )
            terminal_outcomes_cover_round = set(recorded_outcomes) == {
                pending_call.tool_call_id
                for pending_call in approval_support.pending_round_tool_calls(pending_approval)
            }
            if claimed_resolution_intent is None:
                if history.has_resolution_activity or recorded_outcomes:
                    raise RuntimeError(
                        "Tool approval cannot be retried automatically because prior durable "
                        "resolution activity has no exact resolution request identity."
                    )
                raise RuntimeError(
                    "Tool approval resolution request identity was not durably claimed."
                )
            current_intent = approval_support.approval_resolution_intent_from_checkpoint(
                current_checkpoint,
                redactor=self._secret_redactor,
            )
            if current_intent != claimed_resolution_intent:
                raise RuntimeError(
                    "Approval resolution intent changed after the approval was claimed."
                )
            if claimed_resolution_intent.decision is not original_resolution_decision:
                raise RuntimeError(
                    "Tool approval was already claimed with a different resolution decision."
                )
            if claimed_resolution_intent.resolution_request_digest is None:
                raise RuntimeError(
                    "Tool approval cannot be retried automatically because its durable "
                    "resolution intent predates exact resolution request identity."
                )
            if recovery_closure_only:
                if claimed_resolution_intent.decision is not ToolApprovalDecision.APPROVE:
                    raise RuntimeError(
                        "Manual tool approval recovery has no durable approval grant."
                    )
                if not terminal_outcomes_cover_round:
                    raise RuntimeError(
                        "Manual tool approval recovery cannot authorize pending sibling "
                        "execution; retry the exact original approval request."
                    )
                resolution_request_digest = claimed_resolution_intent.resolution_request_digest
            elif claimed_resolution_intent.resolution_request_digest != (resolution_request_digest):
                raise RuntimeError(
                    "Tool approval was already claimed with a different resolution request."
                )
            factory_started_event = await self._environment_lifecycle.emit_factory_started(
                session=session,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
                execution_profile=execution_profile_snapshot.profile,
            )
            if factory_started_event is not None:
                yield factory_started_event
            factory_resolution = await self._environment_lifecycle.resolve_factory(
                session=session,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
                started_event=factory_started_event,
                operation=EnvironmentFactoryOperation.RECONNECT,
                execution_profile=execution_profile_snapshot.profile,
            )
            registered_environment = factory_resolution.registered_environment
            environment_name = _environment_name(registered_environment)
            for event in factory_resolution.events:
                yield event
            if factory_resolution.error is not None:
                raise factory_resolution.error
            if emit_resume_event:
                yield await self._event_writer.emit(
                    approval_support.resumed_event(
                        session=session,
                        agent_name=registered_agent.spec.name,
                        environment_name=environment_name,
                        approval=pending_approval,
                        decision=request.decision,
                        resolved_by=request.resolved_by,
                        expired=expired,
                    )
                )
            if expired:
                yield await self._event_writer.emit(
                    event_with_runtime_payload_authority(
                        Event(
                            type=EventType.TOOL_CALL_APPROVAL_EXPIRED,
                            session_id=session.id,
                            agent_name=registered_agent.spec.name,
                            environment_name=environment_name,
                            tool_name=pending_approval.tool_name,
                            payload={
                                **tool_round_identity.payload(),
                                "approval_id": pending_approval.approval_id,
                                "tool_call_id": pending_approval.tool_call_id,
                                **(
                                    {
                                        "execution_profile_fingerprint": (
                                            pending_approval.execution_profile_fingerprint
                                        )
                                    }
                                    if pending_approval.execution_profile_fingerprint is not None
                                    else {}
                                ),
                                "expires_at": expired_at_iso,
                                "requested_decision": requested_decision.value,
                                "resolved_by": resolved_by_payload,
                                "triggered_by": resolution_actor_payload(triggered_by),
                            },
                        ),
                        "model_step_id",
                        "model_attempt_id",
                        "tool_round_id",
                        "approval_id",
                        *(
                            ("execution_profile_fingerprint",)
                            if pending_approval.execution_profile_fingerprint is not None
                            else ()
                        ),
                    )
                )

            if request.decision not in {
                ToolApprovalDecision.APPROVE,
                ToolApprovalDecision.DENY,
            }:
                raise ValueError(f"Unsupported tool approval decision: {request.decision}")

            binding_started_event = await self._environment_lifecycle.emit_binding_started(
                session=session,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
                execution_profile=execution_profile_snapshot.profile,
            )
            if binding_started_event is not None:
                yield binding_started_event
            binding_result = await self._environment_lifecycle.bind(
                session=session,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
                started_event=binding_started_event,
                execution_profile=execution_profile_snapshot.profile,
            )
            registered_environment = binding_result.registered_environment
            for event in binding_result.events:
                yield event
            if binding_result.error is not None:
                raise binding_result.error

            if request.decision == ToolApprovalDecision.APPROVE:
                run_started_at = time.monotonic()
                limits = copy_run_limits(effective_limits)
                budget_limits = resolved_budget_limits
                if continued_run_limit_accounting is not None:
                    run_started_at, run_baseline, run_budget_authorities = (
                        restore_run_limit_accounting_context(
                            continued_run_limit_accounting,
                            session_id=session.id,
                            budget_limits=budget_limits,
                            now=self._clock(),
                        )
                    )
                else:
                    run_budget_authorities = None
                    run_baseline = (
                        session_usage_summary(session.id, approval_events)
                        if limits.scope == "run" and has_run_limits(limits)
                        else None
                    )
                budget_baseline_events = (
                    approval_events if _has_run_budget_limit(budget_limits) else []
                )
                request_budget_notify_events: list[Event] = []
                recorded_tool_outcomes = list(recorded_outcomes.values())
                pending_tool_calls: list[runtime_records.ToolCallRequest] = []
                executable_pending_tool_calls = 0
                for pending_tool_call in approval_support.pending_round_tool_calls(
                    pending_approval
                ):
                    if pending_tool_call.tool_call_id in recorded_outcomes:
                        continue
                    tool_call = runtime_records.ToolCallRequest(
                        id=pending_tool_call.tool_call_id,
                        name=pending_tool_call.tool_name,
                        arguments=copy_json_value(pending_tool_call.arguments, "arguments"),
                    )
                    pending_tool_calls.append(tool_call)
                    policy_evidence = approval_support.effective_tool_policy_evidence(
                        pending_tool_call
                    )
                    policy_result = approval_support.policy_result_from_pending_tool_call(
                        pending_tool_call
                    )
                    if (
                        policy_evidence is not ToolPolicyEvidence.AUTHORITATIVE
                        or policy_result is None
                        or policy_result.decision == ToolPolicyDecision.DENY
                    ):
                        continue
                    executable_pending_tool_calls += 1
                limit_evaluation = await self._run_limit_controller.evaluate_request_limits(
                    session=session,
                    agent_name=registered_agent.spec.name,
                    environment_name=environment_name,
                    limits=limits,
                    budget_limits=budget_limits,
                    run_started_at=run_started_at,
                    run_baseline=run_baseline,
                    budget_baseline_events=budget_baseline_events,
                    run_budget_authorities=run_budget_authorities,
                    pending_tool_calls=executable_pending_tool_calls,
                    budget_notify_events=request_budget_notify_events,
                    pricing_provider_name=(
                        registered_provider.provider.billing_provider_name
                        or registered_provider.name
                    ),
                    execution_identity=ModelAttemptIdentity(
                        model_step_id=tool_round_identity.model_step_id,
                        model_attempt_id=tool_round_identity.model_attempt_id,
                    ),
                    execution_profile_fingerprint=(execution_profile_snapshot.profile.fingerprint),
                )
                for event in limit_evaluation.events:
                    yield event
                if limit_evaluation.decision is not None:
                    async for event in self._stop_session_for_limit_reached(
                        RecoveryLimitStopRequest(
                            session=session,
                            registered_agent=registered_agent,
                            registered_environment=registered_environment,
                            environment_name=environment_name,
                            decision=limit_evaluation.decision,
                            usage_summary=limit_evaluation.usage_summary,
                            cost_summary=limit_evaluation.cost_summary,
                            messages=transcript,
                            tool_calls=pending_tool_calls,
                            completed_tool_outcomes=recorded_tool_outcomes,
                            pending_approval_to_clear=pending_approval,
                            deferred_messages=deferred_messages,
                            requested_approval_decision=original_resolution_decision,
                            approval_resolution_request_digest=resolution_request_digest,
                            execution_profile=execution_profile_snapshot.profile,
                        )
                    ):
                        yield event
                    return

            pending_round_tool_calls = approval_support.pending_round_tool_calls(pending_approval)
            publish_arguments_as_unavailable = len(pending_round_tool_calls) > 1
            round_tool_calls = [
                runtime_records.ToolCallRequest(
                    id=pending_tool_call.tool_call_id,
                    name=pending_tool_call.tool_name,
                    arguments=copy_json_value(pending_tool_call.arguments, "arguments"),
                )
                for pending_tool_call in pending_round_tool_calls
            ]
            base_round_redactor = self._tool_round_executor.redactor_for_tool_calls(
                registered_agent=registered_agent,
                tool_calls=round_tool_calls,
            )
            legacy_publication_scope = (
                publication_round.assistant_message_state == "quarantined"
                and publication_round.assistant_publication is None
            )
            pause_secret_resolution_scope = invocation_secrets.continuation_secret_resolution_scope(
                pending_approval.secret_resolution_scope,
                registered_environment,
            )
            defer_round_terminals = (
                len(round_tool_calls) > 1 and pause_secret_resolution_scope != "static"
            ) or any(
                registered is not None and registered.workspace_mutation
                for registered in (
                    registered_agent.tools.get(tool_call.name) for tool_call in round_tool_calls
                )
            )
            publication_coordinator = (
                _ToolRoundPublicationCoordinator(
                    session_id=session.id,
                    tool_round_identity=tool_round_identity,
                    session_store=self._session_store,
                    redactor=base_round_redactor,
                    execution_profile=execution_profile_snapshot.profile,
                    tool_exposure=publication_round.tool_exposure,
                )
                if defer_round_terminals
                else None
            )
            staged_hook_modes: dict[str, tuple[bool, bool]] = {}

            async def record_round_publication_snapshot(
                tool_call_id: str,
                snapshot: invocation_secrets.InvocationPublicationSnapshot,
            ) -> None:
                if publication_coordinator is not None:
                    await publication_coordinator.seal_call(
                        tool_call_id=tool_call_id,
                        snapshot=snapshot,
                    )
                    return
                await self._session_store.transform_checkpoint(
                    session.id,
                    tool_round_recovery.assistant_publication_snapshot_transform(
                        tool_round_identity=tool_round_identity,
                        tool_call_id=tool_call_id,
                        redactor=snapshot.redactor,
                        unsafe_output=snapshot.secret_scope_incomplete,
                    ),
                )

            async def record_round_redactor(
                tool_call_id: str,
                snapshot: InvocationRedactorSnapshot,
            ) -> None:
                if publication_coordinator is None:
                    raise AssertionError("Continuation redactor observer has no coordinator.")
                await publication_coordinator.register_redactor(
                    tool_call_id=tool_call_id,
                    redactor=snapshot.redactor,
                )

            async def stage_round_terminal(
                event: Event,
                outcome: runtime_records.ToolCallOutcome,
                allow_modification: bool,
                publish_before_hooks: bool,
                snapshot: invocation_secrets.InvocationPublicationSnapshot,
            ) -> Event:
                if publication_coordinator is None:
                    raise AssertionError("Continuation terminal staging has no coordinator.")
                prepared_event = self._event_writer.prepare(event)
                exposure_blocked = (
                    prepared_event.type is EventType.TOOL_CALL_BLOCKED
                    and prepared_event.payload.get("blocked_by") == "tool_exposure"
                )
                staged = await publication_coordinator.stage_terminal(
                    tool_call_id=outcome.call.id,
                    event=prepared_event,
                    snapshot=snapshot,
                    hooks_state=(
                        "completed"
                        if exposure_blocked
                        else (
                            "observational"
                            if publish_before_hooks
                            else ("pending" if allow_modification else "finalized")
                        )
                    ),
                )
                staged_hook_modes[outcome.call.id] = (
                    allow_modification,
                    publish_before_hooks,
                )
                return staged

            async def record_round_workspace_capture(event: Event) -> Event:
                if publication_coordinator is None:
                    raise AssertionError("Workspace capture recording has no coordinator.")
                return await publication_coordinator.record_workspace_capture(event)

            async def record_static_publication_scope(
                tool_call_id: str,
                *,
                execution_scope_unknown: bool = False,
            ) -> None:
                await record_round_publication_snapshot(
                    tool_call_id,
                    invocation_secrets.InvocationPublicationSnapshot(
                        redactor=base_round_redactor,
                        unsafe_output=execution_scope_unknown,
                        secret_scope_incomplete=execution_scope_unknown,
                    ),
                )

            restarted_staged_ids = (
                await self._fence_restarted_continuation_stages(
                    coordinator=publication_coordinator,
                    session=session,
                    registered_agent=registered_agent,
                    environment_name=environment_name,
                    tool_calls=round_tool_calls,
                    recorded_ids=set(recorded_outcomes),
                    pause_payload={"approval_id": pending_approval.approval_id},
                    idempotency_options={"approval_id": pending_approval.approval_id},
                    execution_profile=execution_profile_snapshot.profile,
                )
                if publication_coordinator is not None
                else set()
            )

            for pending_tool_call, tool_call in zip(
                pending_round_tool_calls,
                round_tool_calls,
                strict=True,
            ):
                registered_tool = registered_agent.tools.get(tool_call.name)
                policy_result = approval_support.policy_result_from_pending_tool_call(
                    pending_tool_call
                )
                policy_evidence = approval_support.effective_tool_policy_evidence(pending_tool_call)
                call_taint_labels = approval_support.taint_labels_from_pending_tool_call(
                    pending_tool_call
                )
                recorded_outcome = recorded_outcomes.get(tool_call.id)
                if recorded_outcome is not None:
                    await record_static_publication_scope(
                        tool_call.id,
                        execution_scope_unknown=legacy_publication_scope,
                    )
                    tool_outcomes.append(recorded_outcome)
                    continue
                if tool_call.id in restarted_staged_ids:
                    continue

                if policy_evidence is ToolPolicyEvidence.UNEXPOSED:
                    await record_static_publication_scope(tool_call.id)
                    async for event, outcome in self._emit_non_authoritative_policy_call(
                        session=session,
                        registered_agent=registered_agent,
                        registered_environment=registered_environment,
                        environment_name=environment_name,
                        tool_call=tool_call,
                        policy_evidence=policy_evidence,
                        tool_exposure=publication_round.tool_exposure,
                        tool_round_identity=tool_round_identity,
                        task_id=pending_approval.task_id,
                        execution_profile=execution_profile_snapshot.profile,
                        approval_id=pending_approval.approval_id,
                        requested_decision=request.decision,
                        resolved_by_payload=resolved_by_payload,
                        resolution_reason=request.reason,
                        resolution_metadata=request.metadata,
                        deferred_terminal_stager=(
                            None if publication_coordinator is None else stage_round_terminal
                        ),
                    ):
                        yield event
                        if outcome is not None:
                            tool_outcomes.append(outcome)
                    continue

                if (
                    policy_evidence is ToolPolicyEvidence.AUTHORITATIVE
                    and policy_result is not None
                    and policy_result.decision == ToolPolicyDecision.DENY
                ):
                    await record_static_publication_scope(tool_call.id)
                    public_policy_result = approval_support.public_policy_denial_result(
                        secret_resolution_scope=pause_secret_resolution_scope,
                        policy_result=policy_result,
                        publish_arguments=(
                            registered_tool is not None and registered_tool.publish_arguments
                        ),
                    )
                    reason = tool_execution.policy_denial_reason(public_policy_result)
                    result = tool_execution.blocked_tool_result(
                        public_policy_result,
                        reason=reason,
                    )
                    idempotency_key = tool_execution.tool_idempotency_key(
                        session_id=session.id,
                        tool_round_id=tool_round_identity.tool_round_id,
                        tool_call_id=tool_call.id,
                        approval_id=pending_approval.approval_id,
                    )
                    async for (
                        event,
                        outcome,
                    ) in self._tool_round_executor.emit_tool_call_result_with_hooks(
                        event=Event(
                            type=EventType.TOOL_CALL_BLOCKED,
                            session_id=session.id,
                            agent_name=registered_agent.spec.name,
                            environment_name=environment_name,
                            tool_name=tool_call.name,
                            payload={
                                **tool_round_identity.payload(),
                                "approval_id": pending_approval.approval_id,
                                "tool_call_id": tool_call.id,
                                "idempotency_key": idempotency_key,
                                **policy_denial_payload_fields(
                                    tool_name=tool_call.name,
                                    denied_by=_TOOL_POLICY_DENIAL_SOURCE,
                                    decision=public_policy_result.decision.value,
                                    reason=reason,
                                    metadata=public_policy_result.metadata,
                                ),
                                "result": result.model_dump(),
                            },
                        ),
                        session=session,
                        registered_agent=registered_agent,
                        registered_environment=registered_environment,
                        tool_call=tool_call,
                        result=result,
                        task_id=pending_approval.task_id,
                        execution_profile=execution_profile_snapshot.profile,
                        redactor=(
                            None
                            if publication_coordinator is None
                            else publication_coordinator.redactor
                        ),
                        output_redactor=(
                            None
                            if publication_coordinator is None
                            else publication_coordinator.redactor
                        ),
                        deferred_terminal_stager=(
                            None if publication_coordinator is None else stage_round_terminal
                        ),
                        publication_snapshot=invocation_secrets.InvocationPublicationSnapshot(
                            redactor=base_round_redactor,
                            unsafe_output=False,
                            secret_scope_incomplete=False,
                        ),
                    ):
                        yield event
                        if outcome is not None:
                            tool_outcomes.append(outcome)
                    continue

                if (
                    policy_evidence is ToolPolicyEvidence.AUTHORITATIVE
                    and policy_result is not None
                    and policy_result.decision == ToolPolicyDecision.REQUIRE_APPROVAL
                    and request.decision == ToolApprovalDecision.APPROVE
                ):
                    yield await self._publish_tool_approval_granted_once(
                        session=session,
                        registered_agent=registered_agent,
                        environment_name=environment_name,
                        pending_approval=pending_approval,
                        tool_call=tool_call,
                        tool_round_identity=tool_round_identity,
                        reason=request.reason,
                        metadata=request.metadata,
                        resolved_by_payload=resolved_by_payload,
                        durable_events=approval_events,
                    )

                if request.decision == ToolApprovalDecision.DENY:
                    await record_static_publication_scope(tool_call.id)
                    approval_required = (
                        policy_evidence is ToolPolicyEvidence.AUTHORITATIVE
                        and policy_result is not None
                        and policy_result.decision == ToolPolicyDecision.REQUIRE_APPROVAL
                    ) or (
                        policy_evidence is ToolPolicyEvidence.AMBIGUOUS
                        and tool_call.id == pending_approval.tool_call_id
                    )
                    result = approval_support.approval_denied_tool_result(
                        request,
                        approval=pending_approval,
                        tool_call=tool_call,
                        approval_required=approval_required,
                    )
                    idempotency_key = tool_execution.tool_idempotency_key(
                        session_id=session.id,
                        tool_round_id=tool_round_identity.tool_round_id,
                        tool_call_id=tool_call.id,
                        approval_id=pending_approval.approval_id,
                    )
                    async for (
                        event,
                        outcome,
                    ) in self._tool_round_executor.emit_tool_call_result_with_hooks(
                        event=Event(
                            type=EventType.TOOL_CALL_APPROVAL_DENIED,
                            session_id=session.id,
                            agent_name=registered_agent.spec.name,
                            environment_name=environment_name,
                            tool_name=tool_call.name,
                            payload={
                                **tool_round_identity.payload(),
                                "approval_id": pending_approval.approval_id,
                                "tool_call_id": tool_call.id,
                                "idempotency_key": idempotency_key,
                                "approval_required": approval_required,
                                "reason": request.reason,
                                **approval_support.bounded_resolution_metadata_payload(
                                    request.metadata,
                                    redactor=self._secret_redactor,
                                ),
                                "resolved_by": resolved_by_payload,
                                "expired": expired,
                                "result": result.model_dump(),
                            },
                        ),
                        session=session,
                        registered_agent=registered_agent,
                        registered_environment=registered_environment,
                        tool_call=tool_call,
                        result=result,
                        task_id=pending_approval.task_id,
                        execution_profile=execution_profile_snapshot.profile,
                        redactor=(
                            None
                            if publication_coordinator is None
                            else publication_coordinator.redactor
                        ),
                        output_redactor=(
                            None
                            if publication_coordinator is None
                            else publication_coordinator.redactor
                        ),
                        deferred_terminal_stager=(
                            None if publication_coordinator is None else stage_round_terminal
                        ),
                        publication_snapshot=invocation_secrets.InvocationPublicationSnapshot(
                            redactor=base_round_redactor,
                            unsafe_output=False,
                            secret_scope_incomplete=False,
                        ),
                    ):
                        yield event
                        if outcome is not None:
                            tool_outcomes.append(outcome)
                    continue

                if request.decision == ToolApprovalDecision.APPROVE and policy_evidence in {
                    ToolPolicyEvidence.AMBIGUOUS,
                    ToolPolicyEvidence.UNREGISTERED,
                }:
                    await record_static_publication_scope(tool_call.id)
                    async for (
                        event,
                        outcome,
                    ) in self._emit_non_authoritative_policy_call(
                        session=session,
                        registered_agent=registered_agent,
                        registered_environment=registered_environment,
                        environment_name=environment_name,
                        tool_call=tool_call,
                        policy_evidence=policy_evidence,
                        tool_exposure=publication_round.tool_exposure,
                        tool_round_identity=tool_round_identity,
                        task_id=pending_approval.task_id,
                        execution_profile=execution_profile_snapshot.profile,
                        approval_id=pending_approval.approval_id,
                        requested_decision=request.decision,
                        resolved_by_payload=resolved_by_payload,
                        resolution_reason=request.reason,
                        resolution_metadata=request.metadata,
                        deferred_terminal_stager=(
                            None if publication_coordinator is None else stage_round_terminal
                        ),
                        publication_snapshot=invocation_secrets.InvocationPublicationSnapshot(
                            redactor=base_round_redactor,
                            unsafe_output=False,
                            secret_scope_incomplete=False,
                        ),
                    ):
                        yield event
                        if outcome is not None:
                            tool_outcomes.append(outcome)
                    continue

                if policy_evidence is not ToolPolicyEvidence.AUTHORITATIVE:
                    raise RuntimeError("Pending tool call has no executable policy authority.")

                async for event, outcome in self._tool_round_executor.execute_tool_call(
                    session=session,
                    registered_agent=registered_agent,
                    registered_environment=registered_environment,
                    tool_call=tool_call,
                    request_metadata=request.metadata,
                    task_id=pending_approval.task_id,
                    execution_profile=execution_profile_snapshot.profile,
                    check_policy=False,
                    emit_started=True,
                    policy_output_secret_resolution_scope=pause_secret_resolution_scope,
                    approval_id=pending_approval.approval_id,
                    tool_round_identity=tool_round_identity,
                    model_step=publication_round.model_step,
                    taint_labels=call_taint_labels,
                    publish_arguments_as_unavailable=publish_arguments_as_unavailable,
                    deferred_terminal_stager=(
                        None if publication_coordinator is None else stage_round_terminal
                    ),
                    deferred_terminal_capture_recorder=(
                        None if publication_coordinator is None else record_round_workspace_capture
                    ),
                    resolved_redactor_observer=(
                        None if publication_coordinator is None else record_round_redactor
                    ),
                    publication_snapshot_observer=record_round_publication_snapshot,
                ):
                    yield event
                    if outcome is not None:
                        tool_outcomes.append(outcome)

            if publication_coordinator is not None:
                expected_staged_ids = {
                    call.id for call in round_tool_calls if call.id not in recorded_outcomes
                } | (set(recorded_outcomes) & restarted_staged_ids)
                current_stages = tool_round_recovery.checkpoint_staged_terminals(
                    await self._session_store.load_checkpoint(session.id),
                    tool_round_identity=tool_round_identity,
                )
                if {item.tool_call_id for item in current_stages} != expected_staged_ids:
                    raise RuntimeError(
                        "Dynamic approval continuation requires one private terminal "
                        "stage per unresolved call."
                    )
                async for event, outcome in self._publish_continuation_staged_terminals(
                    coordinator=publication_coordinator,
                    session=session,
                    registered_agent=registered_agent,
                    registered_environment=registered_environment,
                    tool_calls=round_tool_calls,
                    task_id=pending_approval.task_id,
                    execution_profile=execution_profile_snapshot.profile,
                    hook_modes=staged_hook_modes,
                    pause_authority={"approval_id": pending_approval.approval_id},
                    already_published_ids=set(recorded_outcomes),
                ):
                    yield event
                    if outcome is not None:
                        tool_outcomes.append(outcome)

            source_checkpoint = await self._session_store.load_checkpoint(session.id)
            durable_round = tool_round_recovery.pending_tool_round_from_checkpoint(
                source_checkpoint,
                redactor=self._secret_redactor,
                consume_on_rejection=True,
            )
            if durable_round is None or (
                durable_round.tool_round_id,
                durable_round.model_step_id,
                durable_round.model_attempt_id,
            ) != (
                pending_approval.tool_round_id,
                pending_approval.model_step_id,
                pending_approval.model_attempt_id,
            ):
                raise RuntimeError("Pending approval round changed before atomic closure.")
            final_events = await self._session_store.load_events(session.id)
            final_outcomes = approval_support.recorded_tool_outcomes(
                events=final_events,
                approval=pending_approval,
            )
            lifecycle_event_types = {
                EventType.TOOL_CALL_STARTED,
                EventType.TOOL_CALL_COMPLETED,
                EventType.TOOL_CALL_FAILED,
                EventType.TOOL_CALL_BLOCKED,
                EventType.TOOL_CALL_APPROVAL_DENIED,
            }
            lifecycle_events = [
                event
                for event in final_events
                if event.type in lifecycle_event_types
                and event.payload.get("approval_id") == pending_approval.approval_id
                and tool_round_identity.matches_payload(event.payload)
            ]
            ordered_outcomes = [
                final_outcomes[call.tool_call_id] for call in durable_round.tool_calls
            ]
            tool_result_messages = transcript_helpers.tool_result_messages(
                ordered_outcomes,
                tool_round_identity=tool_round_identity,
            )
            transcript_messages = list(tool_result_messages)
            if durable_round.assistant_message_state == "quarantined":
                transcript_messages.insert(
                    0,
                    transcript_helpers.assistant_message_with_projected_tool_arguments(
                        tool_round_recovery.ready_assistant_publication_message(durable_round),
                        ordered_outcomes,
                    ),
                )
            target_checkpoint = approval_support.checkpoint_without_exact_pending_approval_round(
                source_checkpoint,
                approval=pending_approval,
                redactor=self._secret_redactor,
            )
            clear_event = approval_support.cleared_event(
                session=session,
                agent_name=registered_agent.spec.name,
                environment_name=environment_name,
                approval=pending_approval,
            )
            prepared_close = approval_publication.prepare_approval_publication(
                session_id=session.id,
                publication_id=f"approval-close:{pending_approval.approval_id}",
                kind="approval-close",
                intent={
                    "schema_version": 1,
                    "approval_id": pending_approval.approval_id,
                    "tool_call_id": pending_approval.tool_call_id,
                    **tool_round_identity.payload(),
                    "decision": request.decision.value,
                    "requested_decision": original_resolution_decision.value,
                    "resolution_request_digest": resolution_request_digest,
                    "tool_call_ids": [call.tool_call_id for call in durable_round.tool_calls],
                    "approval_digest": runtime_publication_checkpoint_value_digest(
                        pending_approval.model_dump(mode="json")
                    ),
                    "pending_round_digest": runtime_publication_checkpoint_value_digest(
                        durable_round.model_dump(mode="json")
                    ),
                    "event_ids": [clear_event.id],
                    "referenced_event_ids": [event.id for event in lifecycle_events],
                },
                source_checkpoint=source_checkpoint,
                target_checkpoint=target_checkpoint,
                transcript_messages=transcript_messages,
                events=[clear_event],
                referenced_events=lifecycle_events,
                expected_statuses={SessionStatus.RUNNING},
                expected_run_epoch=session.run_epoch,
                expected_transcript_cursor=approval_transcript_cursor,
            )
            prepared_events = prepared_close.request.events
            if len(prepared_events) != 1:
                raise AssertionError("Approval closure must publish one checkpoint event.")
            clear_event = prepared_events[0]
            close_cancellation = await approval_publication.publish_approval_with_exact_replay(
                prepared_close,
                session_store=self._session_store,
                event_writer=self._event_writer,
                fan_out=False,
            )
            pending_approval_cleared = True
            materialized = await self.materialize_expected_deferred_input(
                session.id,
                deferred_messages,
                cancellation=close_cancellation,
            )
            transcript = materialized.messages
            close_cancellation = materialized.cancellation
            await self._event_writer.fan_out_persisted([clear_event])
            yield clear_event
            if close_cancellation is not None:
                raise close_cancellation

            session_stream = self._run_session(
                RecoverySessionRunRequest(
                    session=session,
                    registered_agent=registered_agent,
                    registered_provider=registered_provider,
                    registered_environment=registered_environment,
                    active_invocation_profile=_rebound_active_invocation_profile(
                        session,
                        execution_profile_snapshot,
                    ),
                    messages=transcript,
                    messages_to_append=[],
                    max_steps=effective_max_steps,
                    limits=effective_limits,
                    budget_limits=effective_budget_limits,
                    budget_policy=copy_budget_policy(budget_policy),
                    retry_policy=effective_retry_policy,
                    structured_output=invocation_semantics.structured_output,
                    thinking=invocation_semantics.thinking,
                    request_loop_policies=request.loop_policies,
                    request_metadata=request.metadata,
                    task_id=pending_approval.task_id,
                    task_worker_id=None,
                    start_event_type=None,
                    start_event_payload={},
                    start_task_on_enter=False,
                    release_run_fence_on_exit=False,
                    run_limit_accounting=continued_run_limit_accounting,
                    previous_tool_exposure_profile_id=(
                        _continued_tool_exposure_profile_id(durable_round.tool_exposure)
                    ),
                )
            )
            forwarded_stream = self._session_control.stream_with_out_of_band_events(
                session.id,
                session_stream,
            )
            try:
                async for event in forwarded_stream:
                    yield event
            except GeneratorExit:
                await forwarded_stream.aclose()
                raise
        except GeneratorExit:
            await self.finalize_abandoned_session_by_id(
                session.id,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
                execution_profile=execution_profile_snapshot.profile,
            )
            raise
        except Exception as exc:
            failure_diagnostic = exception_diagnostic(
                exc,
                redactor=self._secret_redactor,
            )
            if isinstance(exc, approval_support.ToolApprovalManualRecoveryRequired):
                session = await self._session_store.update_status(
                    session.id,
                    SessionStatus.INTERRUPTED,
                )
                async for event in self._emit_terminal_event_with_hooks(
                    RecoveryTerminalEventRequest(
                        event=Event(
                            type=EventType.SESSION_INTERRUPTED,
                            session_id=session.id,
                            agent_name=registered_agent.spec.name,
                            environment_name=environment_name,
                            payload={
                                **exception_failure_payload(
                                    exc,
                                    redactor=self._secret_redactor,
                                ),
                                **tool_round_identity.payload(),
                                "interruption_type": _INTERRUPTION_TYPE_TOOL_APPROVAL_REQUIRED,
                                **approval_support.bounded_pending_approval_event_payload(
                                    pending_approval,
                                    redactor=self._secret_redactor,
                                ),
                                "approval_id": pending_approval.approval_id,
                                "tool_call_id": exc.tool_call_id,
                                "tool_name": exc.tool_name,
                                "manual_recovery_required": True,
                            },
                        ),
                        phase=RuntimeHookPhase.AFTER_SESSION_INTERRUPTED,
                        session=session,
                        registered_agent=registered_agent,
                        registered_environment=registered_environment,
                        execution_profile=execution_profile_snapshot.profile,
                    )
                ):
                    yield event
                return

            if not pending_approval_cleared:
                try:
                    pending_approval_cleared = await self._exact_approval_close_receipt_exists(
                        session_id=session.id,
                        approval=pending_approval,
                        requested_decision=original_resolution_decision,
                        resolution_request_digest=resolution_request_digest,
                    )
                except Exception as receipt_error:
                    exc.add_note(
                        "Exact approval-close receipt reconciliation failed; the approval "
                        "remains fail-closed: "
                        f"{type(receipt_error).__name__}: {receipt_error}"
                    )

            if not pending_approval_cleared:
                session = await self._session_store.update_status(
                    session.id,
                    SessionStatus.INTERRUPTED,
                )
                async for event in self._emit_terminal_event_with_hooks(
                    RecoveryTerminalEventRequest(
                        event=Event(
                            type=EventType.SESSION_INTERRUPTED,
                            session_id=session.id,
                            agent_name=registered_agent.spec.name,
                            environment_name=environment_name,
                            payload={
                                **exception_failure_payload(
                                    exc,
                                    redactor=self._secret_redactor,
                                ),
                                **tool_round_identity.payload(),
                                "interruption_type": _INTERRUPTION_TYPE_TOOL_APPROVAL_REQUIRED,
                                **approval_support.bounded_pending_approval_event_payload(
                                    pending_approval,
                                    redactor=self._secret_redactor,
                                ),
                                **(
                                    {
                                        resume_ledger.TOOL_EVIDENCE_CONFLICT_PAYLOAD_KEY: True,
                                    }
                                    if isinstance(exc, resume_ledger.ToolCallEvidenceConflict)
                                    else {}
                                ),
                            },
                        ),
                        phase=RuntimeHookPhase.AFTER_SESSION_INTERRUPTED,
                        session=session,
                        registered_agent=registered_agent,
                        registered_environment=registered_environment,
                        execution_profile=execution_profile_snapshot.profile,
                    )
                ):
                    yield event
                return

            task_failure_error: Exception | None = None
            if pending_approval.task_id is not None and self._task_store is not None:
                try:
                    task = await self._task_store.fail_task(
                        pending_approval.task_id,
                        task_failure_payload_from_diagnostic(
                            failure_diagnostic,
                            session_id=session.id,
                            additional_fields={
                                "approval_id": pending_approval.approval_id,
                            },
                        ),
                    )
                    yield await self._event_writer.emit(
                        self._task_event(
                            RecoveryTaskEventRequest(
                                event_type=EventType.TASK_FAILED,
                                task=task,
                                session=session,
                                registered_agent=registered_agent,
                                registered_environment=registered_environment,
                            )
                        )
                    )
                except Exception as task_exc:
                    task_failure_error = task_exc
            session = await self._session_store.update_status(session.id, SessionStatus.FAILED)
            payload: dict[str, Any] = {
                **exception_failure_payload(
                    exc,
                    diagnostic=failure_diagnostic,
                    redactor=self._secret_redactor,
                ),
                "approval_id": pending_approval.approval_id,
                "tool_call_id": pending_approval.tool_call_id,
            }
            if task_failure_error is not None:
                payload.update(
                    task_update_error_payload(
                        task_failure_error,
                        redactor=self._secret_redactor,
                    )
                )
            async for event in self._emit_terminal_event_with_hooks(
                RecoveryTerminalEventRequest(
                    event=Event(
                        type=EventType.SESSION_FAILED,
                        session_id=session.id,
                        agent_name=registered_agent.spec.name,
                        environment_name=environment_name,
                        payload=payload,
                    ),
                    phase=RuntimeHookPhase.AFTER_SESSION_FAILED,
                    session=session,
                    registered_agent=registered_agent,
                    registered_environment=registered_environment,
                    execution_profile=execution_profile_snapshot.profile,
                )
            ):
                yield event

    async def _exact_approval_close_receipt_exists(
        self,
        *,
        session_id: str,
        approval: PendingToolApproval,
        requested_decision: ToolApprovalDecision,
        resolution_request_digest: str,
    ) -> bool:
        """Prove that this exact resolution crossed its atomic close boundary."""

        receipt = await self._session_store.load_runtime_publication_receipt(
            session_id,
            f"approval-close:{approval.approval_id}",
        )
        if receipt is None:
            return False
        expected_identity = {
            "approval_id": approval.approval_id,
            "tool_call_id": approval.tool_call_id,
            "tool_round_id": approval.tool_round_id,
            "requested_decision": requested_decision.value,
            "resolution_request_digest": resolution_request_digest,
        }
        if receipt.kind != "approval-close" or any(
            receipt.intent.get(key) != value for key, value in expected_identity.items()
        ):
            raise SessionRuntimePublicationConflict(
                "Approval-close receipt conflicts with the claimed resolution request."
            )
        if len(receipt.appended_event_ids) != 1:
            raise SessionRuntimePublicationConflict(
                "Approval-close receipt has invalid event evidence."
            )
        return True

    async def _publish_tool_approval_granted_once(
        self,
        *,
        session: Session,
        registered_agent: runtime_records.RegisteredAgentState,
        environment_name: str | None,
        pending_approval: PendingToolApproval,
        tool_call: runtime_records.ToolCallRequest,
        tool_round_identity: ToolRoundIdentity,
        reason: str | None,
        metadata: dict[str, Any],
        resolved_by_payload: dict[str, Any] | None,
        durable_events: list[Event],
    ) -> Event:
        existing = [
            event
            for event in durable_events
            if event.type == EventType.TOOL_CALL_APPROVED
            and event.payload.get("approval_id") == pending_approval.approval_id
            and event.payload.get("tool_call_id") == tool_call.id
        ]
        if len(existing) > 1:
            raise resume_ledger.ToolCallEvidenceConflict(
                "Tool approval history contains duplicate approval-granted events."
            )
        if existing:
            # The continuation has already matched the retry against the
            # private immutable request digest, and approval_resolution_history
            # has validated this event's complete approval/round/call
            # descriptor. Reuse that durable grant without reconstructing
            # identity from public, bounded audit fields.
            persisted = existing[0]
        else:
            intended = self._event_writer.prepare(
                event_with_runtime_payload_authority(
                    Event(
                        type=EventType.TOOL_CALL_APPROVED,
                        session_id=session.id,
                        agent_name=registered_agent.spec.name,
                        environment_name=environment_name,
                        tool_name=tool_call.name,
                        payload={
                            **tool_round_identity.payload(),
                            "approval_id": pending_approval.approval_id,
                            "tool_call_id": tool_call.id,
                            **(
                                {
                                    "execution_profile_fingerprint": (
                                        pending_approval.execution_profile_fingerprint
                                    )
                                }
                                if pending_approval.execution_profile_fingerprint is not None
                                else {}
                            ),
                            **_public_resolution_audit_fields(
                                secret_resolution_scope=(pending_approval.secret_resolution_scope),
                                reason=reason,
                                metadata=metadata,
                                redactor=self._secret_redactor,
                            ),
                            "resolved_by": resolved_by_payload,
                        },
                    ),
                    "model_step_id",
                    "model_attempt_id",
                    "tool_round_id",
                    "approval_id",
                    *(
                        ("execution_profile_fingerprint",)
                        if pending_approval.execution_profile_fingerprint is not None
                        else ()
                    ),
                )
            )
            persisted = await self._event_writer.persist_exact_replay(intended)
        await self._event_writer.fan_out_persisted([persisted])
        return persisted

    async def _interrupt_for_resumable_manual_recovery(
        self,
        *,
        session: Session,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_environment: runtime_records.RegisteredEnvironment | None,
        execution_profile: ExecutionProfileIdentity,
        payload: dict[str, Any],
    ) -> AsyncGenerator[Event, None]:
        """Close a durable or acknowledgement-ambiguous recovery to resumable state."""
        try:
            interrupted = await self._session_store.transition_status(
                session.id,
                from_statuses={SessionStatus.RUNNING},
                to_status=SessionStatus.INTERRUPTED,
            )
        except SessionStatusConflict:
            current = await self._require_session(session.id)
            if current.status not in {SessionStatus.INTERRUPTING, SessionStatus.INTERRUPTED}:
                raise
            # An operator interruption won the status transition. Finalize its
            # durable request so its identity, reason, and cascade are preserved.
            async for event in self._interrupt_session_for_recovery(
                RecoveryInterruptionRequest(
                    session=current,
                    registered_agent=registered_agent,
                    registered_environment=registered_environment,
                    environment_name=_environment_name(registered_environment),
                    execution_profile=execution_profile,
                )
            ):
                yield event
            return
        async for event in self._emit_terminal_event_with_hooks(
            RecoveryTerminalEventRequest(
                event=event_with_execution_profile_authority(
                    Event(
                        type=EventType.SESSION_INTERRUPTED,
                        session_id=interrupted.id,
                        agent_name=registered_agent.spec.name,
                        environment_name=_environment_name(registered_environment),
                        payload=payload,
                    ),
                    execution_profile,
                ),
                phase=RuntimeHookPhase.AFTER_SESSION_INTERRUPTED,
                session=interrupted,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
                execution_profile=execution_profile,
            )
        ):
            yield event

    async def _reconcile_manual_recovery_persistence(
        self,
        event: Event,
    ) -> _ManualRecoveryPersistenceReconciliation:
        """Classify an append failure using the preassigned durable event id."""
        outcome = await await_shielded_task_outcome(
            asyncio.create_task(self._event_writer.is_persisted(event))
        )
        if outcome.error is None:
            return _ManualRecoveryPersistenceReconciliation(
                persisted=bool(outcome.result),
                cancellation=outcome.cancellation,
            )
        if isinstance(outcome.error, asyncio.CancelledError):
            return _ManualRecoveryPersistenceReconciliation(
                persisted=None,
                cancellation=outcome.cancellation or outcome.error,
            )
        if not isinstance(outcome.error, Exception):
            raise outcome.error
        return _ManualRecoveryPersistenceReconciliation(
            persisted=None,
            error=outcome.error,
            cancellation=outcome.cancellation,
        )

    async def fence_expired_incomplete_recovery_claim(
        self,
        *,
        session: Session,
        claim_id: str,
    ) -> bool:
        """Fence and clear one observed expired recovery owner.

        The claim id makes the takeover conditional: a concurrent heartbeat or
        claimant that changes ownership causes this operation to leave the
        session untouched.
        """
        claim: _IncompleteRecoveryClaim | None = None
        authoritative_failure: BaseException | None = None
        try:
            claim = await self._claim_incomplete_recovery(
                session=session,
                inactive_before=None,
                required_expired_claim_id=claim_id,
            )
            return claim is not None
        except BaseException as exc:
            authoritative_failure = exc
            raise
        finally:
            if claim is not None:
                await self._cleanup_incomplete_recovery_claim(
                    session_id=session.id,
                    claim_id=claim.claim_id,
                    authoritative_failure=authoritative_failure,
                )

    async def recover_user_input(
        self,
        *,
        request: UserInputRecoveryRequest,
        loaded_session: Session,
        session: Session,
        pending: PendingUserInput,
        pending_tool_call: PendingToolCallApproval,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_provider: runtime_records.RegisteredProvider,
        registered_environment: runtime_records.RegisteredEnvironment | None,
        execution_profile_snapshot: ActiveInvocationExecutionProfile,
        budget_policy: BudgetPolicy | None,
    ) -> AsyncGenerator[Event, None]:
        tool_round_identity = ToolRoundIdentity(
            tool_round_id=pending.tool_round_id,
            model_step_id=pending.model_step_id,
            model_attempt_id=pending.model_attempt_id,
        )
        recovery_prepared = False
        recovery_persisted = False
        cancellation_baseline = _task_cancellation_count()
        recovery_event_to_reconcile: Event | None = None
        authoritative_failure: BaseException | None = None
        abandoned = False
        try:
            recovered_result = ToolResult(
                content=request.message,
                structured=request.structured,
                artifacts=request.artifacts,
                is_error=request.outcome == ToolApprovalRecoveryOutcome.FAILED,
            )
            recovery_secret_resolution_scope = (
                "unknown"
                if pending.assistant_publication is None
                else pending.assistant_publication.secret_resolution_scope
            )
            public_recovered_result = _public_manual_recovery_result(
                recovered_result,
                secret_resolution_scope=recovery_secret_resolution_scope,
            )
            event_type = (
                EventType.TOOL_CALL_FAILED
                if recovered_result.is_error
                else EventType.TOOL_CALL_COMPLETED
            )
            events = await self._session_store.load_events(session.id)
            approval_support.validate_round_recovery_target(
                events=events,
                pending_calls=pending.tool_calls,
                tool_call_id=request.tool_call_id,
                input_id=pending.input_id,
                tool_round_identity=tool_round_identity,
            )
            factory_started_event = await self._environment_lifecycle.emit_factory_started(
                session=session,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
                execution_profile=execution_profile_snapshot.profile,
            )
            if factory_started_event is not None:
                yield factory_started_event
            factory_resolution = await self._environment_lifecycle.resolve_factory(
                session=session,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
                started_event=factory_started_event,
                operation=EnvironmentFactoryOperation.RECONNECT,
                execution_profile=execution_profile_snapshot.profile,
            )
            registered_environment = factory_resolution.registered_environment
            environment_name = _environment_name(registered_environment)
            for event in factory_resolution.events:
                yield event
            if factory_resolution.error is not None:
                session = await self._session_store.update_status(
                    session.id,
                    SessionStatus.INTERRUPTED,
                )
                async for event in self._emit_terminal_event_with_hooks(
                    RecoveryTerminalEventRequest(
                        event=event_with_execution_profile_authority(
                            Event(
                                type=EventType.SESSION_INTERRUPTED,
                                session_id=session.id,
                                agent_name=registered_agent.spec.name,
                                environment_name=environment_name,
                                payload={
                                    **tool_round_identity.payload(),
                                    "interruption_type": _INTERRUPTION_TYPE_USER_INPUT_REQUIRED,
                                    **pending_user_input_interruption_payload(pending),
                                    **_environment_factory_resolution_error_payload(
                                        factory_resolution.error,
                                        redactor=self._secret_redactor,
                                    ),
                                },
                            ),
                            execution_profile_snapshot.profile,
                        ),
                        phase=RuntimeHookPhase.AFTER_SESSION_INTERRUPTED,
                        session=session,
                        registered_agent=registered_agent,
                        registered_environment=registered_environment,
                        execution_profile=execution_profile_snapshot.profile,
                    )
                ):
                    yield event
                return
            recovery_tool_event, public_recovered_result = tool_results.redact_tool_result_event(
                event=Event(
                    type=event_type,
                    session_id=session.id,
                    agent_name=registered_agent.spec.name,
                    environment_name=environment_name,
                    tool_name=pending_tool_call.tool_name,
                    payload={
                        **tool_round_identity.payload(),
                        "tool_call_id": pending_tool_call.tool_call_id,
                        "idempotency_key": tool_execution.tool_idempotency_key(
                            session_id=session.id,
                            tool_round_id=tool_round_identity.tool_round_id,
                            tool_call_id=pending_tool_call.tool_call_id,
                            pause_id=pending.input_id,
                        ),
                        "input_id": pending.input_id,
                        "manual_recovery": True,
                        **tool_argument_publication.unavailable_argument_projection().payload_fields(),
                        **_public_resolution_audit_fields(
                            secret_resolution_scope=recovery_secret_resolution_scope,
                            reason=request.reason,
                            metadata=request.metadata,
                            redactor=self._secret_redactor,
                        ),
                        "resolved_by": resolution_actor_payload(request.resolved_by),
                        "result": public_recovered_result.model_dump(),
                    },
                ),
                result=public_recovered_result,
                redactor=self._secret_redactor,
            )
            recovery_tool_event = event_with_execution_profile_authority(
                recovery_tool_event,
                execution_profile_snapshot.profile,
            )
            recovery_event_to_reconcile = recovery_tool_event
            recovery_events = [
                event_with_execution_profile_authority(
                    Event(
                        type=EventType.SESSION_RESUMED,
                        session_id=session.id,
                        agent_name=registered_agent.spec.name,
                        environment_name=environment_name,
                        payload={
                            **tool_round_identity.payload(),
                            "interruption_type": _INTERRUPTION_TYPE_USER_INPUT_REQUIRED,
                            "input_id": pending.input_id,
                            "tool_call_id": pending.tool_call_id,
                            "resolved_by": resolution_actor_payload(request.resolved_by),
                        },
                    ),
                    execution_profile_snapshot.profile,
                ),
                recovery_tool_event,
            ]
            emitted_recovery_events = await self._event_writer.persist_many(
                session.id, recovery_events
            )
            recovery_persisted = True
            await self._event_writer.fan_out_persisted(emitted_recovery_events)
            for event in emitted_recovery_events:
                yield event
            tool_call = runtime_records.ToolCallRequest(
                id=pending_tool_call.tool_call_id,
                name=pending_tool_call.tool_name,
                arguments={},
            )
            tool_event = emitted_recovery_events[-1]
            # Manual recovery persists the operator-supplied result before hooks run, so
            # after_tool_call is observe-only here (v1): the threaded modification is ignored.
            async for event, _modified in self._tool_round_executor.run_tool_call_hooks(
                session=session,
                tool_event=tool_event,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
                tool_call=tool_call,
                result=public_recovered_result,
                task_id=pending.task_id,
                execution_profile=execution_profile_snapshot.profile,
                redactor=self._secret_redactor,
                output_redactor=self._secret_redactor,
                allow_modification=False,
            ):
                yield event
            recovery_prepared = True
        except (GeneratorExit, asyncio.CancelledError) as exc:
            authoritative_failure = exc
            abandoned = True
            raise
        except Exception as exc:
            authoritative_failure = exc
            reconciliation_error: Exception | None = None
            if not recovery_persisted and recovery_event_to_reconcile is not None:
                try:
                    reconciliation = await self._reconcile_manual_recovery_persistence(
                        recovery_event_to_reconcile
                    )
                except BaseException as reconciliation_failure:
                    authoritative_failure = reconciliation_failure
                    abandoned = (
                        _recovery_abandonment_signal(
                            reconciliation_failure,
                            cancellation_baseline=cancellation_baseline,
                        )
                        is not None
                    )
                    raise
                if reconciliation.cancellation is not None:
                    reconciliation.cancellation.add_note(
                        "Manual user-input recovery append failed while persistence "
                        "reconciliation was running."
                    )
                    authoritative_failure = reconciliation.cancellation
                    abandoned = True
                    raise reconciliation.cancellation from exc
                recovery_persisted = reconciliation.persisted is True
                reconciliation_error = reconciliation.error
            if recovery_persisted or reconciliation_error is not None:
                persistence_payload = (
                    {"manual_recovery_persisted": True}
                    if recovery_persisted
                    else {
                        "manual_recovery_persistence_unknown": True,
                        "persistence_reconciliation_error_type": (
                            _optional_exception_type_name(
                                reconciliation_error,
                                redactor=self._secret_redactor,
                            )
                        ),
                    }
                )
                diagnostic = exception_diagnostic(
                    exc,
                    redactor=self._secret_redactor,
                )
                try:
                    async for event in self._interrupt_for_resumable_manual_recovery(
                        session=session,
                        registered_agent=registered_agent,
                        registered_environment=registered_environment,
                        execution_profile=execution_profile_snapshot.profile,
                        payload={
                            **tool_round_identity.payload(),
                            "interruption_type": _INTERRUPTION_TYPE_USER_INPUT_REQUIRED,
                            **pending_user_input_interruption_payload(pending),
                            "input_id": pending.input_id,
                            "tool_call_id": pending_tool_call.tool_call_id,
                            **persistence_payload,
                            **diagnostic.payload_fields(),
                        },
                    ):
                        yield event
                except BaseException as interruption_failure:
                    authoritative_failure = interruption_failure
                    abandoned = (
                        _recovery_abandonment_signal(
                            interruption_failure,
                            cancellation_baseline=cancellation_baseline,
                        )
                        is not None
                    )
                    raise
                # The original failure is now represented by durable interrupted
                # state. It must not suppress a later fence-release failure.
                authoritative_failure = None
                return
            await self._session_store.update_status(session.id, loaded_session.status)
            raise
        except BaseExceptionGroup as exc:
            authoritative_failure = exc
            abandoned = (
                _recovery_abandonment_signal(
                    exc,
                    cancellation_baseline=cancellation_baseline,
                )
                is not None
            )
            if recovery_persisted and not abandoned:
                async for event in self._interrupt_session_for_recovery(
                    RecoveryInterruptionRequest(
                        session=session,
                        registered_agent=registered_agent,
                        registered_environment=registered_environment,
                        environment_name=_environment_name(registered_environment),
                        execution_profile=execution_profile_snapshot.profile,
                    )
                ):
                    yield event
            raise
        finally:
            if not recovery_prepared:
                await self._cleanup_recovery_handoff(
                    stream=None,
                    session_id=session.id,
                    registered_agent=registered_agent,
                    registered_environment=registered_environment,
                    authoritative_failure=authoritative_failure,
                    finalize_abandoned=abandoned,
                    release_run_fence=True,
                    execution_profile=execution_profile_snapshot.profile,
                )

        continuation_stream: AsyncGenerator[Event, None] | None = None
        authoritative_failure = None
        abandoned = False
        try:
            response = UserInputResponse(
                session_id=request.session_id,
                input_id=request.input_id,
                answer=request.answer,
                structured=request.structured,
                artifacts=request.artifacts,
                metadata=request.metadata,
                resolved_by=request.resolved_by,
                max_steps=request.max_steps,
                limits=request.limits,
                budget_limits=request.budget_limits,
                retry_policy=request.retry_policy,
                structured_output=request.structured_output,
                thinking=request.thinking,
                loop_policies=request.loop_policies,
            )
            continuation_stream = self.continue_user_input_resolution(
                response=response,
                session=session,
                pending=pending,
                registered_agent=registered_agent,
                registered_provider=registered_provider,
                registered_environment=registered_environment,
                execution_profile_snapshot=execution_profile_snapshot,
                budget_policy=budget_policy,
                emit_resume_event=False,
            )
            async for event in continuation_stream:
                yield event
        except BaseException as exc:
            authoritative_failure = exc
            abandoned = (
                _recovery_abandonment_signal(
                    exc,
                    cancellation_baseline=cancellation_baseline,
                )
                is not None
            )
            raise
        finally:
            await self._cleanup_recovery_handoff(
                stream=continuation_stream,
                session_id=session.id,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
                authoritative_failure=authoritative_failure,
                finalize_abandoned=abandoned,
                release_run_fence=True,
                execution_profile=execution_profile_snapshot.profile,
            )

    async def recover_tool_approval(
        self,
        *,
        request: ToolApprovalRecoveryRequest,
        loaded_session: Session,
        session: Session,
        pending_approval: PendingToolApproval,
        pending_tool_call: PendingToolCallApproval,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_provider: runtime_records.RegisteredProvider,
        registered_environment: runtime_records.RegisteredEnvironment | None,
        execution_profile_snapshot: ActiveInvocationExecutionProfile,
        budget_policy: BudgetPolicy | None,
        deferred_messages: list[Message],
        claimed_resolution_intent: approval_support.ApprovalResolutionIntent | None,
    ) -> AsyncGenerator[Event, None]:
        tool_round_identity = ToolRoundIdentity(
            tool_round_id=pending_approval.tool_round_id,
            model_step_id=pending_approval.model_step_id,
            model_attempt_id=pending_approval.model_attempt_id,
        )
        recovery_prepared = False
        recovery_persisted = False
        cancellation_baseline = _task_cancellation_count()
        recovery_event_to_reconcile: Event | None = None
        authoritative_failure: BaseException | None = None
        abandoned = False
        try:
            recovered_result = approval_support.recovered_tool_result(
                request=request,
            )
            recovery_secret_resolution_scope = pending_approval.secret_resolution_scope
            public_recovered_result = _public_manual_recovery_result(
                recovered_result,
                secret_resolution_scope=recovery_secret_resolution_scope,
            )
            event_type = (
                EventType.TOOL_CALL_FAILED
                if recovered_result.is_error
                else EventType.TOOL_CALL_COMPLETED
            )
            # Recovery reconciles an externally executed side effect that was
            # authorized before the crash, so an expired window does not block it
            # (an expired-never-approved approval has no started tool to recover).
            # The out-of-window reconciliation is still stamped for the audit trail.
            recovered_after_expiry = approval_support.pending_approval_expired(
                pending_approval, self._clock()
            )
            events = await self._session_store.load_events(session.id)
            approval_support.validate_recovery_target(
                events=events,
                approval=pending_approval,
                tool_call_id=request.tool_call_id,
            )
            factory_started_event = await self._environment_lifecycle.emit_factory_started(
                session=session,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
                execution_profile=execution_profile_snapshot.profile,
            )
            if factory_started_event is not None:
                yield factory_started_event
            factory_resolution = await self._environment_lifecycle.resolve_factory(
                session=session,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
                started_event=factory_started_event,
                operation=EnvironmentFactoryOperation.RECONNECT,
                execution_profile=execution_profile_snapshot.profile,
            )
            registered_environment = factory_resolution.registered_environment
            environment_name = _environment_name(registered_environment)
            for event in factory_resolution.events:
                yield event
            if factory_resolution.error is not None:
                session = await self._session_store.update_status(
                    session.id,
                    SessionStatus.INTERRUPTED,
                )
                async for event in self._emit_terminal_event_with_hooks(
                    RecoveryTerminalEventRequest(
                        event=Event(
                            type=EventType.SESSION_INTERRUPTED,
                            session_id=session.id,
                            agent_name=registered_agent.spec.name,
                            environment_name=environment_name,
                            payload={
                                **tool_round_identity.payload(),
                                "interruption_type": _INTERRUPTION_TYPE_TOOL_APPROVAL_REQUIRED,
                                **approval_support.bounded_pending_approval_event_payload(
                                    pending_approval,
                                    redactor=self._secret_redactor,
                                ),
                                **_environment_factory_resolution_error_payload(
                                    factory_resolution.error,
                                    redactor=self._secret_redactor,
                                ),
                                "approval_id": pending_approval.approval_id,
                            },
                        ),
                        phase=RuntimeHookPhase.AFTER_SESSION_INTERRUPTED,
                        session=session,
                        registered_agent=registered_agent,
                        registered_environment=registered_environment,
                        execution_profile=execution_profile_snapshot.profile,
                    )
                ):
                    yield event
                return
            recovery_tool_event, public_recovered_result = tool_results.redact_tool_result_event(
                event=Event(
                    type=event_type,
                    session_id=session.id,
                    agent_name=registered_agent.spec.name,
                    environment_name=environment_name,
                    tool_name=pending_tool_call.tool_name,
                    payload={
                        **tool_round_identity.payload(),
                        "approval_id": pending_approval.approval_id,
                        "tool_call_id": pending_tool_call.tool_call_id,
                        "idempotency_key": tool_execution.tool_idempotency_key(
                            session_id=session.id,
                            tool_round_id=tool_round_identity.tool_round_id,
                            tool_call_id=pending_tool_call.tool_call_id,
                            approval_id=pending_approval.approval_id,
                        ),
                        "manual_recovery": True,
                        **tool_argument_publication.unavailable_argument_projection().payload_fields(),
                        **_public_resolution_audit_fields(
                            secret_resolution_scope=recovery_secret_resolution_scope,
                            reason=request.reason,
                            metadata=request.metadata,
                            redactor=self._secret_redactor,
                        ),
                        "resolved_by": resolution_actor_payload(request.resolved_by),
                        "expired": recovered_after_expiry,
                        "result": public_recovered_result.model_dump(),
                    },
                ),
                result=public_recovered_result,
                redactor=self._secret_redactor,
            )
            recovery_tool_event = event_with_execution_profile_authority(
                recovery_tool_event,
                execution_profile_snapshot.profile,
            )
            recovery_event_to_reconcile = recovery_tool_event
            recovery_events = [
                approval_support.resumed_event(
                    session=session,
                    agent_name=registered_agent.spec.name,
                    environment_name=environment_name,
                    approval=pending_approval,
                    decision=ToolApprovalDecision.APPROVE,
                    resolved_by=request.resolved_by,
                    expired=recovered_after_expiry,
                ),
                recovery_tool_event,
            ]
            emitted_recovery_events = await self._event_writer.persist_many(
                session.id, recovery_events
            )
            recovery_persisted = True
            await self._event_writer.fan_out_persisted(emitted_recovery_events)
            for event in emitted_recovery_events:
                yield event
            tool_call = runtime_records.ToolCallRequest(
                id=pending_tool_call.tool_call_id,
                name=pending_tool_call.tool_name,
                arguments={},
            )
            tool_event = emitted_recovery_events[-1]
            # Manual recovery persists the operator-supplied result before hooks run, so
            # after_tool_call is observe-only here (v1): the threaded modification is ignored.
            async for event, _modified in self._tool_round_executor.run_tool_call_hooks(
                session=session,
                tool_event=tool_event,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
                tool_call=tool_call,
                result=public_recovered_result,
                task_id=pending_approval.task_id,
                execution_profile=execution_profile_snapshot.profile,
                redactor=self._secret_redactor,
                output_redactor=self._secret_redactor,
                allow_modification=False,
            ):
                yield event
            recovery_prepared = True
        except (GeneratorExit, asyncio.CancelledError) as exc:
            authoritative_failure = exc
            abandoned = True
            raise
        except Exception as exc:
            authoritative_failure = exc
            reconciliation_error: Exception | None = None
            if not recovery_persisted and recovery_event_to_reconcile is not None:
                try:
                    reconciliation = await self._reconcile_manual_recovery_persistence(
                        recovery_event_to_reconcile
                    )
                except BaseException as reconciliation_failure:
                    authoritative_failure = reconciliation_failure
                    abandoned = (
                        _recovery_abandonment_signal(
                            reconciliation_failure,
                            cancellation_baseline=cancellation_baseline,
                        )
                        is not None
                    )
                    raise
                if reconciliation.cancellation is not None:
                    reconciliation.cancellation.add_note(
                        "Manual tool-approval recovery append failed while persistence "
                        "reconciliation was running."
                    )
                    authoritative_failure = reconciliation.cancellation
                    abandoned = True
                    raise reconciliation.cancellation from exc
                recovery_persisted = reconciliation.persisted is True
                reconciliation_error = reconciliation.error
            if recovery_persisted or reconciliation_error is not None:
                persistence_payload = (
                    {"manual_recovery_persisted": True}
                    if recovery_persisted
                    else {
                        "manual_recovery_persistence_unknown": True,
                        "persistence_reconciliation_error_type": (
                            _optional_exception_type_name(
                                reconciliation_error,
                                redactor=self._secret_redactor,
                            )
                        ),
                    }
                )
                diagnostic = exception_diagnostic(
                    exc,
                    redactor=self._secret_redactor,
                )
                try:
                    async for event in self._interrupt_for_resumable_manual_recovery(
                        session=session,
                        registered_agent=registered_agent,
                        registered_environment=registered_environment,
                        execution_profile=execution_profile_snapshot.profile,
                        payload={
                            **tool_round_identity.payload(),
                            "interruption_type": _INTERRUPTION_TYPE_TOOL_APPROVAL_REQUIRED,
                            **approval_support.bounded_pending_approval_event_payload(
                                pending_approval,
                                redactor=self._secret_redactor,
                            ),
                            "approval_id": pending_approval.approval_id,
                            "tool_call_id": pending_tool_call.tool_call_id,
                            **persistence_payload,
                            **diagnostic.payload_fields(),
                        },
                    ):
                        yield event
                except BaseException as interruption_failure:
                    authoritative_failure = interruption_failure
                    abandoned = (
                        _recovery_abandonment_signal(
                            interruption_failure,
                            cancellation_baseline=cancellation_baseline,
                        )
                        is not None
                    )
                    raise
                # The original failure is now represented by durable interrupted
                # state. It must not suppress a later fence-release failure.
                authoritative_failure = None
                return
            await self._session_store.update_status(session.id, loaded_session.status)
            raise
        except BaseExceptionGroup as exc:
            authoritative_failure = exc
            abandoned = (
                _recovery_abandonment_signal(
                    exc,
                    cancellation_baseline=cancellation_baseline,
                )
                is not None
            )
            if recovery_persisted and not abandoned:
                async for event in self._interrupt_session_for_recovery(
                    RecoveryInterruptionRequest(
                        session=session,
                        registered_agent=registered_agent,
                        registered_environment=registered_environment,
                        environment_name=_environment_name(registered_environment),
                        execution_profile=execution_profile_snapshot.profile,
                    )
                ):
                    yield event
            raise
        finally:
            if not recovery_prepared:
                await self._cleanup_recovery_handoff(
                    stream=None,
                    session_id=session.id,
                    registered_agent=registered_agent,
                    registered_environment=registered_environment,
                    authoritative_failure=authoritative_failure,
                    finalize_abandoned=abandoned,
                    release_run_fence=True,
                    execution_profile=execution_profile_snapshot.profile,
                )

        continuation_stream: AsyncGenerator[Event, None] | None = None
        authoritative_failure = None
        abandoned = False
        try:
            approval_request = ToolApprovalRequest(
                session_id=request.session_id,
                approval_id=request.approval_id,
                tool_round_id=request.tool_round_id,
                tool_call_id=request.tool_call_id,
                decision=ToolApprovalDecision.APPROVE,
                reason=request.reason,
                metadata=request.metadata,
                resolved_by=request.resolved_by,
                max_steps=request.max_steps,
                limits=request.limits,
                budget_limits=request.budget_limits,
                retry_policy=request.retry_policy,
                structured_output=request.structured_output,
                thinking=request.thinking,
                loop_policies=request.loop_policies,
            )
            continuation_stream = self.continue_tool_approval_resolution(
                request=approval_request,
                session=session,
                pending_approval=pending_approval,
                registered_agent=registered_agent,
                registered_provider=registered_provider,
                registered_environment=registered_environment,
                execution_profile_snapshot=execution_profile_snapshot,
                budget_policy=budget_policy,
                deferred_messages=deferred_messages,
                emit_resume_event=False,
                enforce_expiry=False,
                claimed_resolution_intent=claimed_resolution_intent,
                recovery_closure_only=True,
            )
            async for event in continuation_stream:
                yield event
        except BaseException as exc:
            authoritative_failure = exc
            abandoned = (
                _recovery_abandonment_signal(
                    exc,
                    cancellation_baseline=cancellation_baseline,
                )
                is not None
            )
            raise
        finally:
            await self._cleanup_recovery_handoff(
                stream=continuation_stream,
                session_id=session.id,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
                authoritative_failure=authoritative_failure,
                finalize_abandoned=abandoned,
                release_run_fence=True,
                execution_profile=execution_profile_snapshot.profile,
            )

    async def _claim_manual_tool_round_recovery(
        self,
        *,
        session: Session,
        pending_round: tool_round_recovery.PendingToolRound,
        pending_tool_call: PendingToolCallApproval,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_environment: runtime_records.RegisteredEnvironment | None,
        execution_profile_snapshot: ActiveInvocationExecutionProfile,
    ) -> _IncompleteRecoveryClaim | _ManualRecoveryInterruptionFence:
        """Claim recovery or fence an operator interruption that won the race."""
        claim_id = str(uuid4())
        run_operation_id = str(uuid4())
        claim_expires_at: datetime | None = None
        claim_run_epoch: int | None = None
        claimed_run_operation_id: str | None = None
        session_before_fence: Session | None = None

        def require_matching_pending_call(checkpoint: dict[str, Any] | None) -> None:
            self._reject_approval_owned_tool_round_recovery(checkpoint)
            current_round = tool_round_recovery.pending_tool_round_from_checkpoint(
                checkpoint,
                redactor=self._secret_redactor,
                consume_on_rejection=True,
            )
            if current_round is None:
                raise RuntimeError("Session has no pending tool round.")
            if current_round.tool_round_id != pending_round.tool_round_id:
                raise RuntimeError("Pending tool round changed before recovery claimed it.")
            if current_round != pending_round:
                raise RuntimeError("Pending tool round changed before recovery claimed it.")
            current_tool_call = approval_support.round_tool_call_for_recovery(
                pending_calls=current_round.tool_calls,
                tool_call_id=pending_tool_call.tool_call_id,
            )
            if current_tool_call != pending_tool_call:
                raise RuntimeError("Pending tool call changed before recovery claimed it.")

        def claim_checkpoint(
            current_session: Session,
            checkpoint: dict[str, Any] | None,
        ) -> dict[str, Any]:
            nonlocal claim_expires_at, claim_run_epoch
            nonlocal claimed_run_operation_id, session_before_fence
            claimed_at = self._clock()
            _require_aware_datetime(claimed_at, "manual recovery claim clock")
            pending_operator_interruption = (
                checkpoint is not None and _PENDING_SESSION_INTERRUPT_CHECKPOINT_KEY in checkpoint
            )
            interruption_advanced = (
                pending_operator_interruption
                or current_session.status == SessionStatus.INTERRUPTING
                or (
                    current_session.status == SessionStatus.INTERRUPTED
                    and (
                        session.status != SessionStatus.INTERRUPTED
                        or current_session.run_epoch != session.run_epoch
                    )
                )
            )
            if interruption_advanced:
                raise _ManualRecoveryInterrupted(
                    "Session interruption became durable before manual recovery claimed it."
                )
            if (
                checkpoint is not None
                and _PENDING_INTERRUPTION_CASCADE_CHECKPOINT_KEY in checkpoint
            ):
                raise _ManualRecoveryCascadePending(
                    "Session has an incomplete background interruption cascade."
                )
            checkpoint = _checkpoint_without_active_incomplete_recovery_claim(
                checkpoint,
                now=claimed_at,
            )
            require_matching_pending_call(checkpoint)
            claim_expires_at = claimed_at + _INCOMPLETE_RECOVERY_CLAIM_LEASE
            claim_run_epoch = current_session.run_epoch + 1
            session_before_fence = current_session.model_copy(deep=True)
            updated = {} if checkpoint is None else copy_json_value(checkpoint, "checkpoint")
            updated = checkpoint_with_active_invocation_execution_profile(
                updated,
                session_id=current_session.id,
                interaction_id=execution_profile_snapshot.interaction_id,
                run_epoch=current_session.run_epoch + 1,
                profile=execution_profile_snapshot.profile,
                expected=execution_profile_snapshot,
            )
            updated[_INCOMPLETE_RECOVERY_CLAIM_CHECKPOINT_KEY] = {
                "version": 1,
                "claim_id": claim_id,
                "claimed_at": claimed_at.isoformat(),
                "claim_expires_at": claim_expires_at.isoformat(),
                "operation": "manual_tool_round_recovery",
                **tool_round_recovery.pending_tool_round_identity(pending_round).payload(),
                "tool_call_id": pending_tool_call.tool_call_id,
            }
            existing_operation = _session_run_operation_from_checkpoint(updated)
            if current_session.status == SessionStatus.RUNNING and existing_operation is not None:
                claimed_run_operation_id = existing_operation.operation_id
                return _checkpoint_with_rebased_session_run_operation(
                    updated,
                    previous_run_epoch=current_session.run_epoch,
                    run_epoch=claim_run_epoch,
                )
            claimed_run_operation_id = run_operation_id
            return _checkpoint_with_session_run_operation(
                checkpoint=updated,
                current_session=current_session,
                operation_id=run_operation_id,
            )

        transition_task = asyncio.create_task(
            self._session_store.transition_status_and_checkpoint(
                session.id,
                from_statuses=_TOOL_ROUND_RECOVERABLE_SESSION_STATUSES,
                to_status=SessionStatus.RUNNING,
                checkpoint_transform=claim_checkpoint,
            )
        )
        outcome = await await_shielded_task_outcome(transition_task)

        if outcome.error is not None:
            if isinstance(outcome.error, _ManualRecoveryCascadePending):
                if outcome.cancellation is not None:
                    outcome.cancellation.add_note(
                        "Manual tool-round recovery was blocked by an incomplete "
                        "background interruption cascade."
                    )
                    raise outcome.cancellation from outcome.error
                raise outcome.error
            if isinstance(outcome.error, _ManualRecoveryInterrupted):

                def fence_interruption(
                    _current_session: Session,
                    checkpoint: dict[str, Any] | None,
                ) -> dict[str, Any]:
                    nonlocal claim_expires_at, claim_run_epoch
                    require_matching_pending_call(checkpoint)
                    claimed_at = self._clock()
                    _require_aware_datetime(claimed_at, "manual recovery fence clock")
                    claim_expires_at = claimed_at + _INCOMPLETE_RECOVERY_CLAIM_LEASE
                    claim_run_epoch = _current_session.run_epoch + 1
                    updated = (
                        {} if checkpoint is None else copy_json_value(checkpoint, "checkpoint")
                    )
                    current_profile = active_invocation_execution_profile_from_checkpoint(updated)
                    if (
                        current_profile is None
                        or current_profile.interaction_id
                        != execution_profile_snapshot.interaction_id
                        or current_profile.profile != execution_profile_snapshot.profile
                        or not active_invocation_execution_profile_matches_session_epoch(
                            current_profile,
                            session_id=_current_session.id,
                            run_epoch=_current_session.run_epoch,
                        )
                    ):
                        raise _ManualRecoveryInterrupted(
                            "Active invocation profile changed before the interrupted "
                            "manual recovery was fenced."
                        )
                    updated = checkpoint_with_active_invocation_execution_profile(
                        updated,
                        session_id=_current_session.id,
                        interaction_id=current_profile.interaction_id,
                        run_epoch=claim_run_epoch,
                        profile=current_profile.profile,
                        expected=current_profile,
                    )
                    updated[_INCOMPLETE_RECOVERY_CLAIM_CHECKPOINT_KEY] = {
                        "version": 1,
                        "claim_id": claim_id,
                        "claimed_at": claimed_at.isoformat(),
                        "claim_expires_at": claim_expires_at.isoformat(),
                        "operation": "manual_tool_round_interruption_fence",
                        **tool_round_recovery.pending_tool_round_identity(pending_round).payload(),
                        "tool_call_id": pending_tool_call.tool_call_id,
                    }
                    return _checkpoint_with_rebased_session_run_operation(
                        updated,
                        previous_run_epoch=_current_session.run_epoch,
                        run_epoch=claim_run_epoch,
                    )

                fence_outcome = await await_shielded_task_outcome(
                    asyncio.create_task(
                        self._session_store.fence_run_and_transform_checkpoint(
                            session.id,
                            statuses={
                                SessionStatus.INTERRUPTING,
                                SessionStatus.INTERRUPTED,
                            },
                            checkpoint_transform=fence_interruption,
                        )
                    ),
                    cancellation=outcome.cancellation,
                )
                interruption_error: BaseException | None = fence_outcome.cancellation
                if fence_outcome.error is not None:
                    interruption_error = fence_outcome.cancellation or fence_outcome.error
                    reconciliation_outcome = await await_shielded_task_outcome(
                        asyncio.create_task(
                            self._load_owned_incomplete_recovery_claim(
                                session.id,
                                claim_id,
                                expected_run_epoch=claim_run_epoch,
                            )
                        ),
                        cancellation=fence_outcome.cancellation,
                    )
                    reconciliation_cancellation = reconciliation_outcome.cancellation
                    reconciliation_failure = reconciliation_outcome.error
                    if reconciliation_failure is not None:
                        if not isinstance(
                            reconciliation_failure,
                            Exception | asyncio.CancelledError,
                        ):
                            raise reconciliation_failure from fence_outcome.error
                        interruption_error.add_note(
                            "Could not reconcile whether the interrupted manual recovery "
                            "fence committed: "
                            f"{type(reconciliation_failure).__name__}."
                        )
                        if reconciliation_cancellation is not None:
                            reconciliation_cancellation.add_note(
                                "Interrupted manual recovery fence transition also failed: "
                                f"{type(fence_outcome.error).__name__}."
                            )
                            raise reconciliation_cancellation from fence_outcome.error
                        raise fence_outcome.error
                    elif reconciliation_outcome.result is not None:
                        fenced_session = reconciliation_outcome.result
                        if reconciliation_cancellation is not None:
                            interruption_error = reconciliation_cancellation
                            interruption_error.add_note(
                                "Interrupted manual recovery fence transition also failed: "
                                f"{type(fence_outcome.error).__name__}."
                            )
                    else:
                        if reconciliation_cancellation is not None:
                            reconciliation_cancellation.add_note(
                                "Interrupted manual recovery fence transition also failed: "
                                f"{type(fence_outcome.error).__name__}."
                            )
                            raise reconciliation_cancellation from fence_outcome.error
                        raise fence_outcome.error
                else:
                    fenced_session = fence_outcome.result
                if fenced_session is None:
                    raise RuntimeError("Interrupted manual recovery fence returned no session.")
                if claim_expires_at is None or claim_run_epoch is None:
                    raise RuntimeError("Interrupted manual recovery fence persisted no claim.")
                if fenced_session.run_epoch != claim_run_epoch:
                    raise RuntimeError(
                        "Interrupted manual recovery fence returned an unexpected run epoch."
                    )
                _activate_session_run_fence(fenced_session)
                return _ManualRecoveryInterruptionFence(
                    session=fenced_session,
                    claim_id=claim_id,
                    error=interruption_error,
                )

            reconciliation_outcome = await await_shielded_task_outcome(
                asyncio.create_task(
                    self._load_owned_incomplete_recovery_claim(
                        session.id,
                        claim_id,
                        expected_run_epoch=claim_run_epoch,
                    )
                ),
                cancellation=outcome.cancellation,
            )
            reconciliation_cancellation = reconciliation_outcome.cancellation
            reconciliation_failure = reconciliation_outcome.error
            if reconciliation_cancellation is None and isinstance(
                reconciliation_failure,
                asyncio.CancelledError,
            ):
                reconciliation_cancellation = reconciliation_failure
            authoritative_failure = reconciliation_cancellation or outcome.error
            if reconciliation_failure is not None:
                if not isinstance(reconciliation_failure, Exception | asyncio.CancelledError):
                    raise reconciliation_failure from outcome.error
                authoritative_failure.add_note(
                    "Could not reconcile whether the manual tool-round recovery claim "
                    f"committed: {type(reconciliation_failure).__name__}."
                )
            elif reconciliation_outcome.result is not None:
                reconciled_session = reconciliation_outcome.result
                _activate_session_run_fence(reconciled_session)
                await _run_recovery_cleanup_steps(
                    authoritative_failure=authoritative_failure,
                    steps=(
                        (
                            "ambiguous manual recovery claim finalization",
                            lambda: self.finalize_abandoned_session_by_id(
                                reconciled_session.id,
                                registered_agent=registered_agent,
                                registered_environment=registered_environment,
                                execution_profile=execution_profile_snapshot.profile,
                            ),
                        ),
                        (
                            "ambiguous manual recovery claim cleanup",
                            lambda: self._cleanup_incomplete_recovery_claim(
                                session_id=reconciled_session.id,
                                claim_id=claim_id,
                                authoritative_failure=authoritative_failure,
                            ),
                        ),
                    ),
                )
            if reconciliation_cancellation is not None:
                reconciliation_cancellation.add_note(
                    "Manual tool-round recovery claim transition also failed: "
                    f"{type(outcome.error).__name__}."
                )
                raise reconciliation_cancellation from outcome.error
            raise outcome.error
        claimed_session = outcome.result
        if claimed_session is None:
            raise RuntimeError("Manual tool-round recovery claim returned no session.")

        # The durable transition ran in a shielded child task. Bind its epoch to
        # the caller that will perform recovery writes and eventual cleanup.
        _activate_session_run_fence(claimed_session)
        if (
            claim_expires_at is None
            or claim_run_epoch is None
            or claimed_run_operation_id is None
            or session_before_fence is None
            or claimed_session.run_epoch != claim_run_epoch
        ):
            invariant_failure = RuntimeError(
                "Manual tool-round recovery transition did not persist its claim."
            )
            await _run_recovery_cleanup_steps(
                authoritative_failure=invariant_failure,
                steps=(
                    (
                        "abandoned manual recovery finalization",
                        lambda: self.finalize_abandoned_session_by_id(
                            claimed_session.id,
                            registered_agent=registered_agent,
                            registered_environment=registered_environment,
                            execution_profile=execution_profile_snapshot.profile,
                        ),
                    ),
                    (
                        "manual recovery claim cleanup",
                        lambda: self._cleanup_incomplete_recovery_claim(
                            session_id=claimed_session.id,
                            claim_id=claim_id,
                            authoritative_failure=invariant_failure,
                        ),
                    ),
                ),
            )
            raise invariant_failure

        claim = _IncompleteRecoveryClaim(
            claim_id=claim_id,
            claim_expires_at=claim_expires_at,
            session_before_fence=session_before_fence,
            session=claimed_session,
            run_operation=_SessionRunOperation(
                operation_id=claimed_run_operation_id,
                run_epoch=claimed_session.run_epoch,
            ),
        )
        if outcome.cancellation is None:
            return claim

        await _run_recovery_cleanup_steps(
            authoritative_failure=outcome.cancellation,
            steps=(
                (
                    "abandoned manual recovery finalization",
                    lambda: self.finalize_abandoned_session_by_id(
                        claimed_session.id,
                        registered_agent=registered_agent,
                        registered_environment=registered_environment,
                        execution_profile=execution_profile_snapshot.profile,
                    ),
                ),
                (
                    "manual recovery claim cleanup",
                    lambda: self._cleanup_incomplete_recovery_claim(
                        session_id=claimed_session.id,
                        claim_id=claim.claim_id,
                        authoritative_failure=outcome.cancellation,
                    ),
                ),
            ),
        )
        raise outcome.cancellation

    async def recover_tool_round(
        self,
        *,
        request: ToolRoundRecoveryRequest,
        loaded_session: Session,
        pending_round: tool_round_recovery.PendingToolRound,
        pending_tool_call: PendingToolCallApproval,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_provider: runtime_records.RegisteredProvider,
        registered_environment: runtime_records.RegisteredEnvironment | None,
        invocation_semantics: _RecoveryInvocationSemantics,
        execution_profile_snapshot: ActiveInvocationExecutionProfile,
        budget_policy: BudgetPolicy | None,
    ) -> AsyncGenerator[Event, None]:
        """Claim one manual recovery durably and stream its owned continuation."""
        caller_runtime_task = asyncio.current_task()
        interrupted_baseline = await self._session_control.latest_interrupted_event(
            loaded_session.id
        )
        interrupted_baseline_id = None if interrupted_baseline is None else interrupted_baseline.id
        claim = await self._claim_manual_tool_round_recovery(
            session=loaded_session,
            pending_round=pending_round,
            pending_tool_call=pending_tool_call,
            registered_agent=registered_agent,
            registered_environment=registered_environment,
            execution_profile_snapshot=execution_profile_snapshot,
        )
        if isinstance(claim, _ManualRecoveryInterruptionFence):
            authoritative_failure = claim.error
            interruption_request = RecoveryInterruptionRequest(
                session=claim.session,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
                environment_name=_environment_name(registered_environment),
                execution_profile=execution_profile_snapshot.profile,
            )

            async def finalize_owned_interruption() -> None:
                async for _ in self._interrupt_session_for_recovery(interruption_request):
                    pass

            try:
                if claim.error is not None:
                    # Reconciliation proved this exact fence committed. Finish
                    # its operator interruption directly even when the session
                    # was already INTERRUPTED; the generic abandoned-session
                    # finalizer deliberately ignores terminal statuses.
                    await _run_recovery_cleanup_steps(
                        authoritative_failure=claim.error,
                        steps=(
                            (
                                "interrupted manual recovery finalization",
                                finalize_owned_interruption,
                            ),
                        ),
                    )
                    raise claim.error
                async for event in self._interrupt_session_for_recovery(interruption_request):
                    yield event
            except BaseException as exc:
                authoritative_failure = exc
                raise
            finally:
                await _run_recovery_cleanup_steps(
                    authoritative_failure=authoritative_failure,
                    steps=(
                        (
                            "interrupted manual recovery claim release",
                            lambda: self._cleanup_incomplete_recovery_claim(
                                session_id=claim.session.id,
                                claim_id=claim.claim_id,
                                authoritative_failure=authoritative_failure,
                            ),
                        ),
                    ),
                )
            return
        stop_heartbeat = asyncio.Event()
        heartbeat_task = asyncio.create_task(
            self._heartbeat_incomplete_recovery_claim(
                session_id=claim.session.id,
                claim_id=claim.claim_id,
                claim_expires_at=claim.claim_expires_at,
                stop=stop_heartbeat,
            )
        )
        stop_interruption_watch = asyncio.Event()
        interruption_watch_task = asyncio.create_task(
            self._watch_manual_recovery_interruption(
                session_id=claim.session.id,
                interrupted_baseline_id=interrupted_baseline_id,
                stop=stop_interruption_watch,
            )
        )
        recovery_stream = self._recover_tool_round_claimed(
            request=request,
            loaded_session=claim.session_before_fence,
            session=claim.session,
            run_operation=claim.run_operation,
            pending_round=pending_round,
            pending_tool_call=pending_tool_call,
            registered_agent=registered_agent,
            registered_provider=registered_provider,
            registered_environment=registered_environment,
            invocation_semantics=invocation_semantics,
            execution_profile_snapshot=execution_profile_snapshot,
            budget_policy=budget_policy,
        )
        deliveries: asyncio.Queue[_ManualRecoveryEventDelivery | _ManualRecoveryStreamOutcome] = (
            asyncio.Queue(maxsize=2)
        )
        consumer_stopped = asyncio.Event()
        supervisor_started = asyncio.Event()
        consumer_stop_failure: BaseException | None = None
        forwarded_interrupted_event_ids: set[str] = set()

        def heartbeat_failure() -> BaseException | None:
            if not heartbeat_task.done():
                return None
            if heartbeat_task.cancelled():
                return _IncompleteRecoveryClaimLost(
                    "Manual tool-round recovery claim heartbeat was cancelled unexpectedly."
                )
            failure = heartbeat_task.exception()
            if failure is not None:
                return failure
            return _IncompleteRecoveryClaimLost(
                "Manual tool-round recovery claim heartbeat stopped unexpectedly."
            )

        def interruption_watch_failure() -> BaseException | None:
            if not interruption_watch_task.done():
                return None
            if interruption_watch_task.cancelled():
                return RuntimeError(
                    "Manual tool-round recovery interruption watcher was cancelled unexpectedly."
                )
            failure = interruption_watch_task.exception()
            if failure is not None:
                return failure
            if interruption_watch_task.result():
                return asyncio.CancelledError(
                    "Manual tool-round recovery was interrupted by a durable request."
                )
            return RuntimeError(
                "Manual tool-round recovery interruption watcher stopped unexpectedly."
            )

        async def stop_claim_heartbeat() -> None:
            stop_heartbeat.set()
            await heartbeat_task

        async def stop_interruption_watcher() -> None:
            stop_interruption_watch.set()
            await interruption_watch_task

        async def forward_recovery_events() -> None:
            async for event in recovery_stream:
                delivery = _ManualRecoveryEventDelivery(
                    event=event,
                    consumed=asyncio.Event(),
                )
                await deliveries.put(delivery)
                if event.type == EventType.SESSION_INTERRUPTED:
                    forwarded_interrupted_event_ids.add(event.id)
                await delivery.consumed.wait()
                if consumer_stopped.is_set():
                    raise asyncio.CancelledError

        async def supervise_recovery() -> _ManualRecoverySupervisorResult:
            recovery_task = asyncio.create_task(forward_recovery_events())
            supervisor_runtime_task = asyncio.current_task()
            authoritative_failure: BaseException | None = None
            cleanup_failure: BaseException | None = None
            durable_interruption_observed = False
            if supervisor_runtime_task is not None:
                self._session_control.register_active_control_task(
                    claim.session.id,
                    supervisor_runtime_task,
                )
            supervisor_started.set()

            async def stop_recovery_worker() -> None:
                if not recovery_task.done():
                    recovery_task.cancel()
                await asyncio.gather(recovery_task, return_exceptions=True)

            try:
                done, _pending = await asyncio.wait(
                    {recovery_task, heartbeat_task, interruption_watch_task},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if interruption_watch_task in done:
                    failure = interruption_watch_failure()
                    if failure is None:  # pragma: no cover - task completion is exhaustive.
                        raise AssertionError(
                            "Recovery interruption watcher completed without an outcome."
                        )
                    durable_interruption_observed = (
                        not interruption_watch_task.cancelled()
                        and interruption_watch_task.exception() is None
                        and interruption_watch_task.result()
                    )
                    raise failure
                if heartbeat_task in done:
                    failure = heartbeat_failure()
                    if failure is None:  # pragma: no cover - task completion is exhaustive.
                        raise AssertionError("Recovery heartbeat completed without an outcome.")
                    raise failure
                recovery_task.result()
            except BaseException as exc:
                authoritative_failure = (
                    consumer_stop_failure
                    if (
                        consumer_stopped.is_set()
                        and consumer_stop_failure is not None
                        and isinstance(exc, asyncio.CancelledError)
                    )
                    else exc
                )

            try:
                try:
                    cleanup_failures = await _run_recovery_cleanup_steps(
                        authoritative_failure=authoritative_failure,
                        steps=(
                            ("manual tool-round recovery event worker stop", stop_recovery_worker),
                            (
                                "manual tool-round recovery interruption watcher stop",
                                stop_interruption_watcher,
                            ),
                            (
                                "manual tool-round recovery handoff cleanup",
                                lambda: self._cleanup_recovery_handoff(
                                    stream=recovery_stream,
                                    session_id=claim.session.id,
                                    registered_agent=registered_agent,
                                    registered_environment=registered_environment,
                                    authoritative_failure=authoritative_failure,
                                    finalize_abandoned=authoritative_failure is not None,
                                    release_run_fence=False,
                                    execution_profile=execution_profile_snapshot.profile,
                                ),
                            ),
                            ("manual tool-round recovery heartbeat stop", stop_claim_heartbeat),
                            (
                                "manual tool-round recovery claim release",
                                lambda: self._cleanup_incomplete_recovery_claim(
                                    session_id=claim.session.id,
                                    claim_id=claim.claim_id,
                                    authoritative_failure=authoritative_failure,
                                ),
                            ),
                        ),
                    )
                    if cleanup_failures:
                        cleanup_failure = BaseExceptionGroup(
                            "Manual recovery cleanup failed",
                            [failure for _operation, failure in cleanup_failures],
                        )
                except BaseException as cleanup_error:
                    cleanup_failure = cleanup_error
                    authoritative_failure = cleanup_error
                interrupted_event: Event | None = None
                if durable_interruption_observed and cleanup_failure is None:
                    try:
                        candidate = await self._session_control.wait_for_interrupted_event(
                            claim.session.id
                        )
                    except BaseException as lookup_failure:
                        if authoritative_failure is not None:
                            authoritative_failure.add_note(
                                "The durable operator interruption was finalized, but its "
                                "terminal event could not be reconstructed."
                            )
                            _prepend_exception_cause(authoritative_failure, lookup_failure)
                        else:  # pragma: no cover - the watcher always supplies a failure.
                            authoritative_failure = lookup_failure
                    else:
                        if (
                            candidate is not None
                            and candidate.id != interrupted_baseline_id
                            and candidate.payload.get("interruption_type")
                            == _INTERRUPTION_TYPE_OPERATOR_REQUESTED
                        ):
                            if candidate.id not in forwarded_interrupted_event_ids:
                                interrupted_event = candidate
                            authoritative_failure = None
                await deliveries.put(
                    _ManualRecoveryStreamOutcome(
                        error=authoritative_failure,
                        interrupted_event=interrupted_event,
                    )
                )
            finally:
                if supervisor_runtime_task is not None:
                    self._session_control.unregister_active_control_task(
                        claim.session.id,
                        supervisor_runtime_task,
                    )
            return _ManualRecoverySupervisorResult(
                error=authoritative_failure,
                cleanup_failure=cleanup_failure,
            )

        supervisor_task = asyncio.create_task(supervise_recovery())
        supervisor_start_outcome = await await_shielded_task_outcome(
            asyncio.create_task(supervisor_started.wait())
        )
        if caller_runtime_task is not None:
            # CayuApp reserves the caller task before the durable claim. Once
            # the supervisor is live, transfer process-local ownership so an
            # operator interrupt targets one recovery layer rather than both.
            self._session_control.unregister_active_task(
                claim.session.id,
                caller_runtime_task,
            )
        pending_delivery: _ManualRecoveryEventDelivery | None = None
        authoritative_failure: BaseException | None = None

        async def stop_supervisor() -> None:
            nonlocal consumer_stop_failure
            consumer_stop_failure = authoritative_failure
            consumer_stopped.set()
            if pending_delivery is not None:
                pending_delivery.consumed.set()
            if not supervisor_task.done():
                supervisor_task.cancel()
            await asyncio.gather(supervisor_task, return_exceptions=True)
            if isinstance(consumer_stop_failure, GeneratorExit):
                supervisor_result = supervisor_task.result()
                if supervisor_result.cleanup_failure is not None:
                    raise supervisor_result.cleanup_failure

        try:
            if supervisor_start_outcome.error is not None:
                raise supervisor_start_outcome.error
            if supervisor_start_outcome.cancellation is not None:
                raise supervisor_start_outcome.cancellation
            while True:
                item = await deliveries.get()
                if isinstance(item, _ManualRecoveryStreamOutcome):
                    if item.error is not None:
                        raise item.error
                    if item.interrupted_event is not None:
                        yield item.interrupted_event
                    return
                pending_delivery = item
                yield item.event
                item.consumed.set()
                pending_delivery = None
        except BaseException as exc:
            authoritative_failure = exc
            raise
        finally:
            await _run_recovery_cleanup_steps(
                authoritative_failure=authoritative_failure,
                steps=(("manual tool-round recovery supervisor stop", stop_supervisor),),
            )

    async def _recover_tool_round_claimed(
        self,
        *,
        request: ToolRoundRecoveryRequest,
        loaded_session: Session,
        session: Session,
        run_operation: _SessionRunOperation | None,
        pending_round: tool_round_recovery.PendingToolRound,
        pending_tool_call: PendingToolCallApproval,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_provider: runtime_records.RegisteredProvider,
        registered_environment: runtime_records.RegisteredEnvironment | None,
        invocation_semantics: _RecoveryInvocationSemantics,
        execution_profile_snapshot: ActiveInvocationExecutionProfile,
        budget_policy: BudgetPolicy | None,
    ) -> AsyncGenerator[Event, None]:
        """Persist one operator-verified ordinary tool outcome and continue safely."""
        if run_operation is None:
            raise RuntimeError("Manual tool-round recovery has no durable run operation.")
        recovered_result = ToolResult(
            content=request.message,
            structured=request.structured,
            artifacts=request.artifacts,
            is_error=request.outcome == ToolApprovalRecoveryOutcome.FAILED,
        )
        recovery_secret_resolution_scope = approval_support.tool_round_secret_resolution_scope(
            pending_round
        )
        public_recovered_result = _public_manual_recovery_result(
            recovered_result,
            secret_resolution_scope=recovery_secret_resolution_scope,
        )
        event_type = (
            EventType.TOOL_CALL_FAILED
            if recovered_result.is_error
            else EventType.TOOL_CALL_COMPLETED
        )
        environment_name = _environment_name(registered_environment)
        recovery_persisted = False
        cancellation_baseline = _task_cancellation_count()
        recovery_event_to_reconcile: Event | None = None

        try:
            events = await self._session_store.load_events(session.id)
            tool_round_recovery.validate_tool_round_recovery_target(
                events=events,
                pending_round=pending_round,
                tool_call_id=request.tool_call_id,
            )
            factory_started_event = await self._environment_lifecycle.emit_factory_started(
                session=session,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
                execution_profile=execution_profile_snapshot.profile,
            )
            if factory_started_event is not None:
                yield factory_started_event
            factory_resolution = await self._environment_lifecycle.resolve_factory(
                session=session,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
                started_event=factory_started_event,
                operation=EnvironmentFactoryOperation.RECONNECT,
                execution_profile=execution_profile_snapshot.profile,
            )
            registered_environment = factory_resolution.registered_environment
            environment_name = _environment_name(registered_environment)
            for event in factory_resolution.events:
                yield event
            if factory_resolution.error is not None:
                async for event in self._interrupt_for_resumable_manual_recovery(
                    session=session,
                    registered_agent=registered_agent,
                    registered_environment=registered_environment,
                    execution_profile=execution_profile_snapshot.profile,
                    payload={
                        "interruption_type": _INTERRUPTION_TYPE_RUNTIME_INTERRUPTED,
                        **tool_round_recovery.pending_tool_round_identity(pending_round).payload(),
                        **_environment_factory_resolution_error_payload(
                            factory_resolution.error,
                            redactor=self._secret_redactor,
                        ),
                    },
                ):
                    yield event
                return
            recovery_tool_event, public_recovered_result = tool_results.redact_tool_result_event(
                event=Event(
                    type=event_type,
                    session_id=session.id,
                    agent_name=registered_agent.spec.name,
                    environment_name=environment_name,
                    tool_name=pending_tool_call.tool_name,
                    payload={
                        **tool_round_recovery.pending_tool_round_identity(pending_round).payload(),
                        "tool_call_id": pending_tool_call.tool_call_id,
                        "idempotency_key": tool_execution.tool_idempotency_key(
                            session_id=session.id,
                            tool_round_id=pending_round.tool_round_id,
                            tool_call_id=pending_tool_call.tool_call_id,
                        ),
                        "manual_recovery": True,
                        **tool_argument_publication.unavailable_argument_projection().payload_fields(),
                        **_public_resolution_audit_fields(
                            secret_resolution_scope=recovery_secret_resolution_scope,
                            reason=request.reason,
                            metadata=request.metadata,
                            redactor=self._secret_redactor,
                        ),
                        "resolved_by": resolution_actor_payload(request.resolved_by),
                        "result": public_recovered_result.model_dump(),
                    },
                ),
                result=public_recovered_result,
                redactor=self._secret_redactor,
            )
            recovery_tool_event = event_with_execution_profile_authority(
                recovery_tool_event,
                execution_profile_snapshot.profile,
            )
            recovery_event_to_reconcile = recovery_tool_event
            recovery_events = [
                event_with_execution_profile_authority(
                    Event(
                        type=EventType.SESSION_RESUMED,
                        session_id=session.id,
                        agent_name=registered_agent.spec.name,
                        environment_name=environment_name,
                        payload={
                            "interruption_type": _INTERRUPTION_TYPE_RUNTIME_INTERRUPTED,
                            **tool_round_recovery.pending_tool_round_identity(
                                pending_round
                            ).payload(),
                            "tool_call_id": pending_tool_call.tool_call_id,
                            "resolved_by": resolution_actor_payload(request.resolved_by),
                        },
                    ),
                    execution_profile_snapshot.profile,
                ),
                recovery_tool_event,
            ]
            emitted_recovery_events = await self._event_writer.persist_many(
                session.id, recovery_events
            )
            recovery_persisted = True
            await self._event_writer.fan_out_persisted(emitted_recovery_events)
            for event in emitted_recovery_events:
                yield event
            tool_call = runtime_records.ToolCallRequest(
                id=pending_tool_call.tool_call_id,
                name=pending_tool_call.tool_name,
                arguments={},
            )
            tool_event = emitted_recovery_events[-1]
            # The operator outcome is durable before hooks run. Recovery hooks are
            # observe-only so they cannot rewrite externally verified evidence.
            async for event, _modified in self._tool_round_executor.run_tool_call_hooks(
                session=session,
                tool_event=tool_event,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
                tool_call=tool_call,
                result=public_recovered_result,
                task_id=pending_round.task_id,
                execution_profile=execution_profile_snapshot.profile,
                redactor=self._secret_redactor,
                output_redactor=self._secret_redactor,
                allow_modification=False,
            ):
                yield event

            events = await self._session_store.load_events(session.id)
            recorded_outcomes, started_ids = tool_round_recovery.recorded_tool_outcomes(
                events=events,
                pending_round=pending_round,
            )
            remaining_ids = started_ids - set(recorded_outcomes)
            if remaining_ids:
                next_call = next(
                    call for call in pending_round.tool_calls if call.tool_call_id in remaining_ids
                )
                async for event in self._interrupt_for_resumable_manual_recovery(
                    session=session,
                    registered_agent=registered_agent,
                    registered_environment=registered_environment,
                    execution_profile=execution_profile_snapshot.profile,
                    payload={
                        "interruption_type": _INTERRUPTION_TYPE_RUNTIME_INTERRUPTED,
                        "manual_recovery_required": True,
                        **tool_round_recovery.pending_tool_round_identity(pending_round).payload(),
                        "tool_call_id": next_call.tool_call_id,
                        "tool_name": next_call.tool_name,
                    },
                ):
                    yield event
                return
        except (GeneratorExit, asyncio.CancelledError) as abandonment:
            await _run_recovery_cleanup_steps(
                authoritative_failure=abandonment,
                steps=(
                    (
                        "abandoned session finalization",
                        lambda: self.finalize_abandoned_session_by_id(
                            session.id,
                            registered_agent=registered_agent,
                            registered_environment=registered_environment,
                            execution_profile=execution_profile_snapshot.profile,
                        ),
                    ),
                ),
            )
            raise
        except Exception as exc:
            reconciliation_error: Exception | None = None
            if not recovery_persisted and recovery_event_to_reconcile is not None:
                try:
                    reconciliation = await self._reconcile_manual_recovery_persistence(
                        recovery_event_to_reconcile
                    )
                except BaseException as reconciliation_failure:
                    if (
                        _recovery_abandonment_signal(
                            reconciliation_failure,
                            cancellation_baseline=cancellation_baseline,
                        )
                        is not None
                    ):
                        await _run_recovery_cleanup_steps(
                            authoritative_failure=reconciliation_failure,
                            steps=(
                                (
                                    "abandoned session finalization",
                                    lambda: self.finalize_abandoned_session_by_id(
                                        session.id,
                                        registered_agent=registered_agent,
                                        registered_environment=registered_environment,
                                        execution_profile=execution_profile_snapshot.profile,
                                    ),
                                ),
                            ),
                        )
                    raise
                if reconciliation.cancellation is not None:
                    reconciliation.cancellation.add_note(
                        "Manual tool-round recovery append failed while persistence "
                        "reconciliation was running."
                    )
                    await _run_recovery_cleanup_steps(
                        authoritative_failure=reconciliation.cancellation,
                        steps=(
                            (
                                "abandoned session finalization",
                                lambda: self.finalize_abandoned_session_by_id(
                                    session.id,
                                    registered_agent=registered_agent,
                                    registered_environment=registered_environment,
                                    execution_profile=execution_profile_snapshot.profile,
                                ),
                            ),
                        ),
                    )
                    raise reconciliation.cancellation from exc
                recovery_persisted = reconciliation.persisted is True
                reconciliation_error = reconciliation.error
            if not recovery_persisted and reconciliation_error is None:
                if isinstance(exc, SessionRunFenced):
                    raise
                if loaded_session.status in {
                    SessionStatus.RUNNING,
                    SessionStatus.INTERRUPTING,
                }:
                    if loaded_session.status == SessionStatus.INTERRUPTING:
                        session = await self._session_store.transition_status(
                            session.id,
                            from_statuses={SessionStatus.RUNNING},
                            to_status=SessionStatus.INTERRUPTING,
                        )
                    diagnostic = exception_diagnostic(
                        exc,
                        redactor=self._secret_redactor,
                    )
                    async for event in self._interrupt_for_resumable_manual_recovery(
                        session=session,
                        registered_agent=registered_agent,
                        registered_environment=registered_environment,
                        execution_profile=execution_profile_snapshot.profile,
                        payload={
                            "interruption_type": _INTERRUPTION_TYPE_RUNTIME_INTERRUPTED,
                            **tool_round_recovery.pending_tool_round_identity(
                                pending_round
                            ).payload(),
                            "tool_call_id": pending_tool_call.tool_call_id,
                            "manual_recovery_stale_live_failure": True,
                            **diagnostic.payload_fields(),
                            "resolved_by": resolution_actor_payload(request.resolved_by),
                        },
                    ):
                        yield event
                    return
                try:

                    def restore_checkpoint(
                        _current_session: Session,
                        checkpoint: dict[str, Any] | None,
                    ) -> dict[str, Any] | None:
                        current_operation = _session_run_operation_from_checkpoint(checkpoint)
                        if checkpoint is None or current_operation != run_operation:
                            raise RuntimeError(
                                "Manual recovery run operation changed before rollback."
                            )
                        updated = copy_json_value(checkpoint, "checkpoint")
                        updated.pop(_SESSION_RUN_OPERATION_CHECKPOINT_KEY)
                        return updated

                    await self._session_store.transition_status_and_checkpoint(
                        session.id,
                        from_statuses={SessionStatus.RUNNING},
                        to_status=loaded_session.status,
                        checkpoint_transform=restore_checkpoint,
                    )
                except SessionStatusConflict:
                    current = await self._require_session(session.id)
                    if current.status not in {
                        SessionStatus.INTERRUPTING,
                        SessionStatus.INTERRUPTED,
                    }:
                        raise
                    async for event in self._interrupt_session_for_recovery(
                        RecoveryInterruptionRequest(
                            session=current,
                            registered_agent=registered_agent,
                            registered_environment=registered_environment,
                            environment_name=_environment_name(registered_environment),
                            execution_profile=execution_profile_snapshot.profile,
                        )
                    ):
                        yield event
                    return
                raise
            persistence_payload = (
                {"manual_recovery_persisted": True}
                if recovery_persisted
                else {
                    "manual_recovery_persistence_unknown": True,
                    "persistence_reconciliation_error_type": (
                        _optional_exception_type_name(
                            reconciliation_error,
                            redactor=self._secret_redactor,
                        )
                    ),
                }
            )
            diagnostic = exception_diagnostic(
                exc,
                redactor=self._secret_redactor,
            )
            async for event in self._interrupt_for_resumable_manual_recovery(
                session=session,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
                execution_profile=execution_profile_snapshot.profile,
                payload={
                    "interruption_type": _INTERRUPTION_TYPE_RUNTIME_INTERRUPTED,
                    **tool_round_recovery.pending_tool_round_identity(pending_round).payload(),
                    "tool_call_id": pending_tool_call.tool_call_id,
                    **persistence_payload,
                    **diagnostic.payload_fields(),
                    "resolved_by": resolution_actor_payload(request.resolved_by),
                },
            ):
                yield event
            return
        except BaseExceptionGroup as exc:
            abandonment = _recovery_abandonment_signal(
                exc,
                cancellation_baseline=cancellation_baseline,
            )
            if recovery_persisted and abandonment is None:
                async for event in self._interrupt_session_for_recovery(
                    RecoveryInterruptionRequest(
                        session=session,
                        registered_agent=registered_agent,
                        registered_environment=registered_environment,
                        environment_name=_environment_name(registered_environment),
                        execution_profile=execution_profile_snapshot.profile,
                    )
                ):
                    yield event
            if abandonment is not None:
                await _run_recovery_cleanup_steps(
                    authoritative_failure=exc,
                    steps=(
                        (
                            "abandoned session finalization",
                            lambda: self.finalize_abandoned_session_by_id(
                                session.id,
                                registered_agent=registered_agent,
                                registered_environment=registered_environment,
                                execution_profile=execution_profile_snapshot.profile,
                            ),
                        ),
                    ),
                )
            raise

        session_stream: AsyncGenerator[Event, None] | None = None
        authoritative_failure: BaseException | None = None
        try:
            transcript = await self._session_store.load_transcript(session.id)
            continued_run_limit_accounting = pending_round.run_limit_accounting
            if continued_run_limit_accounting is not None:
                recovery_events = await self._session_store.load_events(session.id)
                continued_run_limit_accounting = rebase_run_limit_accounting_context(
                    continued_run_limit_accounting,
                    session_id=session.id,
                    limits=invocation_semantics.limits,
                    budget_limits=request_budget_limits_for_session(
                        limits=invocation_semantics.budget_limits,
                        agent_name=registered_agent.spec.name,
                        causal_budget_id=session.causal_budget_id,
                    ),
                    events=recovery_events,
                    reset_run_limits=False,
                    reset_budgets=False,
                    now=self._clock(),
                )
            session_stream = self._run_session(
                RecoverySessionRunRequest(
                    session=session,
                    registered_agent=registered_agent,
                    registered_provider=registered_provider,
                    registered_environment=registered_environment,
                    active_invocation_profile=_rebound_active_invocation_profile(
                        session,
                        execution_profile_snapshot,
                    ),
                    messages=transcript,
                    messages_to_append=[],
                    max_steps=invocation_semantics.max_steps,
                    limits=invocation_semantics.limits,
                    budget_limits=invocation_semantics.budget_limits,
                    budget_policy=copy_budget_policy(budget_policy),
                    retry_policy=invocation_semantics.retry_policy,
                    structured_output=invocation_semantics.structured_output,
                    thinking=invocation_semantics.thinking,
                    request_loop_policies=request.loop_policies,
                    request_metadata=request.metadata,
                    task_id=pending_round.task_id,
                    task_worker_id=None,
                    start_event_type=None,
                    start_event_payload={},
                    start_task_on_enter=False,
                    release_run_fence_on_exit=False,
                    run_limit_accounting=continued_run_limit_accounting,
                    previous_tool_exposure_profile_id=(
                        _continued_tool_exposure_profile_id(pending_round.tool_exposure)
                    ),
                )
            )
            async for event in session_stream:
                yield event
        except BaseException as exc:
            authoritative_failure = exc
            raise
        finally:
            await self._cleanup_recovery_handoff(
                stream=session_stream,
                session_id=session.id,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
                authoritative_failure=authoritative_failure,
                finalize_abandoned=(
                    _recovery_abandonment_signal(
                        authoritative_failure,
                        cancellation_baseline=cancellation_baseline,
                    )
                    is not None
                ),
                release_run_fence=False,
                abort_environment_setup=False,
                execution_profile=execution_profile_snapshot.profile,
            )

    async def close_interrupted_tool_round(
        self,
        request: InterruptedToolRoundRequest,
    ) -> AsyncGenerator[Event, None]:
        """Close an interrupted round without replaying unfinished tools."""
        tool_round_identity = copy_tool_round_identity(request.tool_round_identity)
        publication_id = f"tool-round:{tool_round_identity.tool_round_id}"
        if (
            await self._session_store.load_runtime_publication_receipt(
                request.session.id,
                publication_id,
            )
            is not None
        ):
            await self.materialize_deferred_input_if_present(request.session.id)
            request.messages[:] = await self._session_store.load_transcript(request.session.id)
            return
        expected_transcript_cursor = await self._session_store.load_transcript_cursor(
            request.session.id
        )
        source_checkpoint = await self._session_store.load_checkpoint(request.session.id)
        pending_round = tool_round_recovery.pending_tool_round_from_checkpoint(source_checkpoint)
        if (
            pending_round is None
            or tool_round_recovery.pending_tool_round_identity(pending_round) != tool_round_identity
        ):
            raise RuntimeError("Interrupted tool round lost its durable pending marker.")
        pending_tool_calls = tool_round_recovery.pending_round_tool_calls(pending_round)
        if [(tool_call.id, tool_call.name) for tool_call in request.tool_calls] != [
            (tool_call.id, tool_call.name) for tool_call in pending_tool_calls
        ]:
            raise RuntimeError("Interrupted tool calls conflict with the durable pending round.")
        if any(call.tool_name == STRUCTURED_OUTPUT_TOOL_NAME for call in pending_round.tool_calls):
            async for event in self._recover_structured_output_tool_round(
                session=request.session,
                registered_agent=request.registered_agent,
                registered_environment=request.registered_environment,
                messages=request.messages,
                insert_at=len(request.messages),
                pending_round=pending_round,
                source_checkpoint=source_checkpoint,
                retry_allowed=False,
                expected_transcript_cursor=expected_transcript_cursor,
                execution_profile=request.execution_profile,
            ):
                yield event
            return
        lifecycle_events = await self._load_tool_round_lifecycle_events(
            session_id=request.session.id,
            pending_round=pending_round,
        )
        recorded_outcomes, started_ids = tool_round_recovery.recorded_tool_outcomes(
            events=lifecycle_events,
            pending_round=pending_round,
        )
        interrupted_results = _interrupted_tool_round_results(
            tool_calls=pending_tool_calls,
            completed_outcomes=list(recorded_outcomes.values()),
            tool_round_identity=tool_round_identity,
            cancellation_artifacts=request.cancellation_artifacts,
            cancellation_artifacts_by_id=request.cancellation_artifacts_by_id,
        )
        interrupted_results = await self.reattach_subagent_children_in_outcomes(
            session=request.session,
            registered_agent=request.registered_agent,
            tool_round_id=request.tool_round_identity.tool_round_id,
            outcomes=interrupted_results,
            source_checkpoint=source_checkpoint,
        )
        cancellation_redactors = request.cancellation_redactors_by_id or {}
        interrupted_results = [
            tool_results.redact_tool_call_outcomes(
                [outcome],
                cancellation_redactors.get(outcome.call.id, self._secret_redactor),
            )[0]
            for outcome in interrupted_results
        ]
        source_checkpoint, pending_round = await self._complete_recovery_assistant_publication(
            session_id=request.session.id,
            registered_agent=request.registered_agent,
            pending_round=pending_round,
            execution_scope_unknown_ids=started_ids,
        )
        if pending_round.assistant_message_state == "quarantined":
            tool_round_recovery.ready_assistant_publication_message(pending_round)
        planned_terminal_events = [
            _interrupted_tool_call_event(
                session=request.session,
                registered_agent=request.registered_agent,
                registered_environment=request.registered_environment,
                tool_call_outcome=interrupted_result,
                tool_round_identity=tool_round_identity,
            )
            for interrupted_result in interrupted_results
        ]
        tool_round_publication.collect_tool_round_publication_evidence(
            session_id=request.session.id,
            pending_round=pending_round,
            durable_events=[*lifecycle_events, *planned_terminal_events],
        )

        emitted_events: list[Event] = []
        for interrupted_result, terminal_event in zip(
            interrupted_results,
            planned_terminal_events,
            strict=True,
        ):
            expected_public_outcome = runtime_records.ToolCallOutcome(
                call=runtime_records.ToolCallRequest(
                    id=interrupted_result.call.id,
                    name=interrupted_result.call.name,
                    arguments={},
                ),
                result=interrupted_result.result,
            )
            async for event, outcome in self._tool_round_executor.emit_tool_call_result_with_hooks(
                event=terminal_event,
                session=request.session,
                registered_agent=request.registered_agent,
                registered_environment=request.registered_environment,
                tool_call=interrupted_result.call,
                result=interrupted_result.result,
                task_id=pending_round.task_id,
                execution_profile=request.execution_profile,
            ):
                emitted_events.append(event)
                if outcome is not None and outcome != expected_public_outcome:
                    raise RuntimeError("Interrupted tool-round hooks changed terminal evidence.")

        lifecycle_events = await self._load_tool_round_lifecycle_events(
            session_id=request.session.id,
            pending_round=pending_round,
        )
        prepared = tool_round_publication.prepare_tool_round_publication(
            session_id=request.session.id,
            pending_round=pending_round,
            source_checkpoint=source_checkpoint,
            durable_events=lifecycle_events,
            expected_statuses={
                SessionStatus.RUNNING,
                SessionStatus.INTERRUPTING,
                SessionStatus.INTERRUPTED,
            },
            expected_run_epoch=request.session.run_epoch,
            expected_transcript_cursor=expected_transcript_cursor,
        )
        cancellation = await self._publish_tool_round_with_exact_replay(prepared)
        materialized = await self.materialize_expected_deferred_input(
            request.session.id,
            pending_round.deferred_messages,
            cancellation=cancellation,
        )
        request.messages[:] = materialized.messages
        cancellation = materialized.cancellation
        for event in emitted_events:
            yield event
        if cancellation is not None:
            raise cancellation

    async def _load_tool_round_lifecycle_events(
        self,
        *,
        session_id: str,
        pending_round: tool_round_recovery.PendingToolRound,
    ) -> list[Event]:
        """Load bounded lifecycle evidence and scope reused call IDs by round."""
        candidates = await self._session_store.load_tool_round_lifecycle_events_for_round(
            session_id,
            [call.tool_call_id for call in pending_round.tool_calls],
            tool_round_identity=tool_round_recovery.pending_tool_round_identity(pending_round),
        )
        lifecycle_events: list[Event] = []
        for event in candidates:
            event_round_id = event.payload.get("tool_round_id")
            if event_round_id == pending_round.tool_round_id:
                lifecycle_events.append(event)
                continue
            if (
                type(event_round_id) is not str
                or not event_round_id.strip()
                or event_round_id.strip() != event_round_id
            ):
                raise RuntimeError(
                    "Indexed tool-round lifecycle evidence has no valid round identity."
                )
            raise RuntimeError(
                "Round-scoped lifecycle lookup returned evidence for a different tool round."
            )
        return lifecycle_events

    async def _complete_recovery_assistant_publication(
        self,
        *,
        session_id: str,
        registered_agent: runtime_records.RegisteredAgentState,
        pending_round: tool_round_recovery.PendingToolRound,
        execution_scope_unknown_ids: set[str] | frozenset[str] = frozenset(),
    ) -> tuple[dict[str, Any], tool_round_recovery.PendingToolRound]:
        """Finalize calls after every returned secret was durably projected."""

        if pending_round.assistant_message_state == "published":
            checkpoint = await self._session_store.load_checkpoint(session_id)
            return checkpoint or {}, pending_round
        identity = tool_round_recovery.pending_tool_round_identity(pending_round)
        tool_calls = tool_round_recovery.pending_round_tool_calls(pending_round)
        base_redactor = self._tool_round_executor.redactor_for_tool_calls(
            registered_agent=registered_agent,
            tool_calls=tool_calls,
        )
        publication = pending_round.assistant_publication
        covered_ids = set() if publication is None else set(publication.covered_tool_call_ids)
        # Recovery trusts only the capability recorded with the original model
        # completion. Current environment registration may differ after a
        # restart; missing legacy evidence is therefore treated as unknown.
        secret_resolution_scope = (
            "unknown" if publication is None else publication.secret_resolution_scope
        )
        expected_ids = {tool_call.id for tool_call in tool_calls}
        if not execution_scope_unknown_ids <= expected_ids:
            raise RuntimeError("Assistant recovery evidence names a call outside its tool round.")
        for tool_call in tool_calls:
            if tool_call.id in covered_ids:
                continue
            await self._session_store.transform_checkpoint(
                session_id,
                tool_round_recovery.assistant_publication_snapshot_transform(
                    tool_round_identity=identity,
                    tool_call_id=tool_call.id,
                    redactor=base_redactor,
                    unsafe_output=(
                        secret_resolution_scope != "static"
                        and tool_call.id in execution_scope_unknown_ids
                    ),
                ),
            )
        checkpoint = await self._session_store.load_checkpoint(session_id)
        recovered_round = tool_round_recovery.pending_tool_round_from_checkpoint(checkpoint)
        if (
            recovered_round is None
            or tool_round_recovery.pending_tool_round_identity(recovered_round) != identity
        ):
            raise RuntimeError(
                "Pending tool round changed while sealing its recovery publication projection."
            )
        return checkpoint or {}, recovered_round

    async def _recover_structured_output_tool_round(
        self,
        *,
        session: Session,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_environment: runtime_records.RegisteredEnvironment | None,
        messages: list[Message],
        insert_at: int,
        pending_round: tool_round_recovery.PendingToolRound,
        source_checkpoint: dict[str, Any] | None,
        retry_allowed: bool,
        expected_transcript_cursor: int,
        execution_profile: ExecutionProfileIdentity | None,
    ) -> AsyncGenerator[Event, None]:
        """Rebuild one reserved finalizer round from its durable model output."""

        spec = pending_round.structured_output
        step = pending_round.model_step
        attempt = pending_round.structured_output_attempt
        if spec is None or step is None or attempt is None:
            raise RuntimeError(
                "Structured-output recovery requires durable config, step, and attempt."
            )
        if attempt > spec.max_retries + 1:
            raise RuntimeError(
                "Structured-output recovery attempt exceeds the durable retry policy."
            )
        tool_calls = tool_round_recovery.pending_round_tool_calls(pending_round)
        validation = pending_round.structured_output_validation
        if validation is None:
            raise RuntimeError(
                "Structured-output recovery requires authoritative durable validation."
            )
        validation = validation.model_copy(deep=True)
        expected_outcomes = structured_output_tool_round._structured_output_tool_round_outcomes(
            tool_calls=tool_calls,
            spec=spec,
            validation=validation,
        )
        expected_outcomes = tool_results.redact_tool_call_outcomes(
            expected_outcomes,
            self._secret_redactor,
        )
        tool_round_identity = tool_round_recovery.pending_tool_round_identity(pending_round)
        structured_round_redactor = self._tool_round_executor.redactor_for_tool_calls(
            registered_agent=registered_agent,
            tool_calls=tool_calls,
        )
        for expected_outcome in expected_outcomes:
            await self._session_store.transform_checkpoint(
                session.id,
                tool_round_recovery.assistant_publication_snapshot_transform(
                    tool_round_identity=tool_round_identity,
                    tool_call_id=expected_outcome.call.id,
                    redactor=structured_round_redactor,
                    unsafe_output=False,
                ),
            )
        source_checkpoint = await self._session_store.load_checkpoint(session.id)
        reloaded_pending_round = tool_round_recovery.pending_tool_round_from_checkpoint(
            source_checkpoint
        )
        if (
            reloaded_pending_round is None
            or tool_round_recovery.pending_tool_round_identity(reloaded_pending_round)
            != tool_round_identity
        ):
            raise RuntimeError(
                "Structured-output tool round changed while sealing its publication projection."
            )
        pending_round = reloaded_pending_round
        environment_name = _environment_name(registered_environment)
        lifecycle_events = await self._load_tool_round_lifecycle_events(
            session_id=session.id,
            pending_round=pending_round,
        )
        recorded_outcomes, _started_ids = tool_round_recovery.recorded_tool_outcomes(
            events=lifecycle_events,
            pending_round=pending_round,
        )
        terminal_events_by_call = {
            event.payload["tool_call_id"]: event
            for event in lifecycle_events
            if event.type
            in {
                EventType.TOOL_CALL_COMPLETED,
                EventType.TOOL_CALL_FAILED,
                EventType.TOOL_CALL_BLOCKED,
                EventType.TOOL_CALL_APPROVAL_DENIED,
            }
        }
        planned_terminal_events: list[Event] = []
        for expected_outcome in expected_outcomes:
            expected_event = event_with_execution_profile_authority(
                structured_output_tool_round._structured_output_tool_terminal_event(
                    session=session,
                    registered_agent=registered_agent,
                    environment_name=environment_name,
                    tool_round_identity=tool_round_recovery.pending_tool_round_identity(
                        pending_round
                    ),
                    outcome=expected_outcome,
                ),
                execution_profile,
            )
            recorded_outcome = recorded_outcomes.get(expected_outcome.call.id)
            if recorded_outcome is None:
                planned_terminal_events.append(expected_event)
                continue
            recorded_event = terminal_events_by_call.get(expected_outcome.call.id)
            expected_payload = expected_event.payload
            legacy_expected_payload = dict(expected_payload)
            legacy_expected_payload.pop(tool_argument_publication.ARGUMENTS_STATE_FIELD, None)
            recorded_payload_matches = recorded_event is not None and (
                recorded_event.payload == expected_payload
                or (
                    tool_argument_publication.ARGUMENTS_STATE_FIELD not in recorded_event.payload
                    and recorded_event.payload == legacy_expected_payload
                )
            )
            expected_recorded_outcome = expected_outcome
            if recorded_event is not None:
                recorded_projection = tool_argument_publication.terminal_argument_projection(
                    recorded_event.payload,
                    legacy_arguments=expected_outcome.call.arguments,
                )
                expected_recorded_outcome = runtime_records.ToolCallOutcome(
                    call=runtime_records.ToolCallRequest(
                        id=expected_outcome.call.id,
                        name=expected_outcome.call.name,
                        arguments=recorded_projection.transcript_arguments(),
                    ),
                    result=expected_outcome.result,
                )
            if (
                recorded_outcome != expected_recorded_outcome
                or recorded_event is None
                or recorded_event.id != expected_event.id
                or recorded_event.type != expected_event.type
                or recorded_event.session_id != expected_event.session_id
                or recorded_event.agent_name != expected_event.agent_name
                or recorded_event.environment_name != expected_event.environment_name
                or recorded_event.tool_name != expected_event.tool_name
                or not recorded_payload_matches
            ):
                raise RuntimeError(
                    "Durable structured-output terminal evidence conflicts with "
                    f"the pending call: {expected_outcome.call.id}"
                )

        tool_round_publication.collect_tool_round_publication_evidence(
            session_id=session.id,
            pending_round=pending_round,
            durable_events=[*lifecycle_events, *planned_terminal_events],
        )
        emitted_terminal_events: list[Event] = []
        for terminal_event in planned_terminal_events:
            emitted_terminal_events.append(await self._event_writer.emit(terminal_event))

        lifecycle_events = await self._load_tool_round_lifecycle_events(
            session_id=session.id,
            pending_round=pending_round,
        )
        retry_scheduled = not validation.valid and retry_allowed and attempt <= spec.max_retries
        validating_event = structured_output_tool_round._structured_output_validating_event(
            session=session,
            registered_agent=registered_agent,
            environment_name=environment_name,
            spec=spec,
            step=step,
            attempt=attempt,
            tool_round_identity=tool_round_identity,
        )
        outcome_event = structured_output_tool_round._structured_output_event(
            event_type=(
                EventType.STRUCTURED_OUTPUT_VALIDATED
                if validation.valid
                else EventType.STRUCTURED_OUTPUT_FAILED
            ),
            session=session,
            registered_agent=registered_agent,
            environment_name=environment_name,
            spec=spec,
            validation=validation,
            step=step,
            attempt=attempt,
            redactor=self._secret_redactor,
            tool_round_identity=tool_round_identity,
        )
        auxiliary_events = [validating_event, outcome_event]
        if retry_scheduled:
            auxiliary_events.append(
                structured_output_tool_round._structured_output_event(
                    event_type=EventType.STRUCTURED_OUTPUT_RETRY,
                    session=session,
                    registered_agent=registered_agent,
                    environment_name=environment_name,
                    spec=spec,
                    validation=validation,
                    step=step,
                    attempt=attempt,
                    redactor=self._secret_redactor,
                    tool_round_identity=tool_round_identity,
                )
            )
        auxiliary_events = self._event_writer.prepare_many(
            [
                event_with_execution_profile_authority(event, execution_profile)
                for event in auxiliary_events
            ]
        )
        extension = structured_output_tool_round._StructuredOutputToolRoundPublicationExtension(
            intent={
                "schema_version": 1,
                "kind": "structured-output-validation",
                "step": step,
                "attempt": attempt,
                "valid": validation.valid,
                "retry_scheduled": retry_scheduled,
                "event_ids": [event.id for event in auxiliary_events],
            },
            events=tuple(auxiliary_events),
        )
        prepared = tool_round_publication.prepare_tool_round_publication(
            session_id=session.id,
            pending_round=pending_round,
            source_checkpoint=source_checkpoint,
            durable_events=lifecycle_events,
            expected_statuses={
                SessionStatus.RUNNING,
                SessionStatus.INTERRUPTING,
                SessionStatus.INTERRUPTED,
            },
            expected_run_epoch=session.run_epoch,
            expected_transcript_cursor=expected_transcript_cursor,
            extension=extension,
        )
        cancellation = await self._publish_tool_round_with_exact_replay(prepared)
        materialized = await self.materialize_expected_deferred_input(
            session.id,
            pending_round.deferred_messages,
            cancellation=cancellation,
        )
        messages[:] = materialized.messages
        cancellation = materialized.cancellation
        for event in emitted_terminal_events:
            yield event
        for event in auxiliary_events:
            yield copy_event(event)
        if cancellation is not None:
            raise cancellation

    async def _publish_tool_round_with_exact_replay(
        self,
        prepared: tool_round_publication.PreparedToolRoundPublication,
    ) -> asyncio.CancelledError | None:
        """Reconcile an ambiguous publication by replaying its retained request."""
        return await tool_round_publication.publish_tool_round_with_exact_replay(
            prepared,
            session_store=self._session_store,
            event_writer=self._event_writer,
        )

    async def materialize_deferred_input_if_present(self, session_id: str) -> bool:
        """Materialize one private interaction tail using its durable owner identity."""

        deferred = await self._session_store.load_deferred_interaction_input(session_id)
        if deferred is None:
            return False
        return await self._session_store.materialize_deferred_interaction_input(
            session_id,
            interaction_id=deferred.interaction_id,
        )

    async def materialize_deferred_input_for_receipt(
        self,
        receipt: RuntimePublicationReceipt,
    ) -> bool:
        """Finish only the deferred tail owned by an exact approval receipt."""

        if receipt.kind != "approval-close":
            raise ValueError("Deferred receipt materialization requires an approval-close receipt.")
        session = await self._session_store.load(receipt.session_id)
        if session is None:
            return False
        deferred = await self._session_store.load_deferred_interaction_input(receipt.session_id)
        if (
            deferred is None
            or receipt.interaction_id is None
            or deferred.interaction_id != receipt.interaction_id
        ):
            return False
        transcript = await self._session_store.load_transcript(receipt.session_id)
        if len(transcript) != receipt.transcript_end_cursor:
            return False
        _activate_session_run_fence(session)
        try:
            return await self._session_store.materialize_deferred_interaction_input(
                receipt.session_id,
                interaction_id=receipt.interaction_id,
            )
        except SessionRunFenced:
            return False
        finally:
            _deactivate_session_run_fence(receipt.session_id)

    async def materialize_expected_deferred_input(
        self,
        session_id: str,
        expected_messages: list[Message],
        *,
        cancellation: asyncio.CancelledError | None = None,
    ) -> DeferredInputMaterialization:
        """Append a retained recovery tail without losing caller cancellation."""

        expected = [detach_message(message) for message in expected_messages]

        def redacted_transcript_view(messages: list[Message]) -> list[Message]:
            return [
                redact_runtime_message_for_boundary(
                    message,
                    redactor=self._secret_redactor,
                    field_name=f"session.transcript[{index}]",
                )
                for index, message in enumerate(messages)
            ]

        async def materialize() -> list[Message]:
            deferred = await self._session_store.load_deferred_interaction_input(session_id)
            if (
                expected
                and deferred is not None
                and redacted_transcript_view(deferred.source_messages) != expected
            ):
                raise RuntimeError(
                    "Deferred interaction input conflicts with its pending tool-round tail "
                    f"(durable_count={len(deferred.source_messages)}, "
                    f"expected_count={len(expected)})."
                )
            if deferred is not None:
                await self._session_store.materialize_deferred_interaction_input(
                    session_id,
                    interaction_id=deferred.interaction_id,
                )
            transcript = await self._session_store.load_transcript(session_id)
            transcript_view = redacted_transcript_view(transcript)
            if expected and (
                len(transcript_view) < len(expected)
                or transcript_view[-len(expected) :] != expected
            ):
                raise RuntimeError(
                    "Deferred interaction input was not materialized after its tool result."
                )
            return transcript_view

        outcome = await await_shielded_task_outcome(
            asyncio.create_task(materialize()),
            cancellation=cancellation,
        )
        cancellation = outcome.cancellation
        error = outcome.error
        if isinstance(error, asyncio.CancelledError):
            error = unexpected_child_cancellation_error(
                error,
                operation="Deferred interaction input materialization",
            )
        if error is not None:
            if cancellation is not None:
                cancellation.add_note(
                    "Deferred interaction input materialization also failed: "
                    f"{type(error).__name__}."
                )
                raise cancellation from error
            raise error
        if outcome.result is None:
            missing_result = RuntimeError(
                "Deferred interaction input materialization returned no transcript."
            )
            if cancellation is not None:
                cancellation.add_note(str(missing_result))
                raise cancellation from missing_result
            raise missing_result
        return DeferredInputMaterialization(
            messages=outcome.result,
            cancellation=cancellation,
        )

    async def reattach_subagent_children_in_outcomes(
        self,
        *,
        session: Session,
        registered_agent: runtime_records.RegisteredAgentState,
        tool_round_id: str | None,
        outcomes: list[runtime_records.ToolCallOutcome],
        source_checkpoint: dict[str, Any] | None,
    ) -> list[runtime_records.ToolCallOutcome]:
        """Replace unfinished spawn outcomes with matching durable child references."""
        if tool_round_id is None or not outcomes:
            return outcomes
        children = await self._subagent_children_by_idempotency_key(session.id)
        reattached: list[runtime_records.ToolCallOutcome] = []
        for outcome in outcomes:
            idempotency_key = tool_execution.tool_idempotency_key(
                session_id=session.id,
                tool_round_id=tool_round_id,
                tool_call_id=outcome.call.id,
            )
            marker_backed = (
                durable_subagent_submission_seed_from_checkpoint(
                    source_checkpoint,
                    idempotency_key=idempotency_key,
                )
                is not None
                or durable_subagent_submission_from_checkpoint(
                    source_checkpoint,
                    idempotency_key=idempotency_key,
                )
                is not None
                or durable_subagent_submission_receipt_from_checkpoint(
                    source_checkpoint,
                    idempotency_key=idempotency_key,
                )
                is not None
            )
            if idempotency_key not in children and not marker_backed:
                reattached.append(outcome)
                continue
            recovery_arguments = self._subagent_recovery_arguments(
                checkpoint=source_checkpoint,
                parent_session=session,
                tool_name=outcome.call.name,
                tool_round_id=tool_round_id,
                tool_call_id=outcome.call.id,
                idempotency_key=idempotency_key,
                fallback=outcome.call.arguments,
            )
            reconciled_result = await self._reconcile_subagent_child(
                children,
                idempotency_key=idempotency_key,
                tool_call_id=outcome.call.id,
                tool_name=outcome.call.name,
                tool_round_id=tool_round_id,
                arguments=recovery_arguments,
                parent_session=session,
                registered_agent=registered_agent,
            )
            result = reconciled_result
            if result is None:
                result = self._reattached_subagent_result(
                    children,
                    idempotency_key,
                    tool_call_id=outcome.call.id,
                    tool_name=outcome.call.name,
                    tool_round_id=tool_round_id,
                    arguments=recovery_arguments,
                    parent_session=session,
                    registered_agent=registered_agent,
                )
            if result is not None and outcome.result.artifacts:
                result = result.model_copy(
                    update={
                        "artifacts": copy_json_value(
                            outcome.result.artifacts,
                            "reattached_subagent_artifacts",
                        )
                    },
                    deep=True,
                )
            reattached.append(
                outcome
                if result is None
                else runtime_records.ToolCallOutcome(call=outcome.call, result=result)
            )
        return reattached

    async def recover_pending_tool_round(
        self,
        *,
        session: Session,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_environment: runtime_records.RegisteredEnvironment | None,
        messages: list[Message],
        execution_profile: ExecutionProfileIdentity | None = None,
        tail_message_count: int = 0,
        incomplete_recovery_claimed: bool = False,
        expected_transcript_cursor: int | None = None,
    ) -> AsyncGenerator[Event, None]:
        """Repair one durable pending round strictly from recorded evidence."""
        checkpoint = await self._session_store.load_checkpoint(session.id)
        pending_round = tool_round_recovery.pending_tool_round_from_checkpoint(
            checkpoint,
            redactor=self._secret_redactor,
            consume_on_rejection=True,
        )
        if pending_round is None:
            return
        if expected_transcript_cursor is None:
            expected_transcript_cursor = await self._session_store.load_transcript_cursor(
                session.id
            )
        environment_name = _environment_name(registered_environment)
        if pending_round.agent_name != registered_agent.spec.name:
            raise RuntimeError(
                f"Pending tool round belongs to a different agent: {pending_round.agent_name}."
            )
        if pending_round.environment_name != environment_name:
            raise RuntimeError(
                "Pending tool round belongs to a different environment: "
                f"{pending_round.environment_name}."
            )
        if pending_round.tool_exposure is not None:
            validate_resolved_tool_exposure_authority(
                pending_round.tool_exposure,
                registered_agent.tool_capabilities,
            )
        registered_tool_names = frozenset(registered_agent.tools)
        durable_policy_decisions = frozenset(decision.value for decision in ToolPolicyDecision)
        ambiguous_interrupt_close_intent = (
            session.status == SessionStatus.INTERRUPTING
            and _approval_interrupt_close_intent_matches(
                checkpoint,
                pending_round=pending_round,
            )
        )
        invalid_planned_calls = [
            call
            for call in pending_round.tool_calls
            if pending_round.policy_state == "planned"
            and call.tool_name in registered_tool_names
            and not (
                (
                    call.policy_evidence is ToolPolicyEvidence.AUTHORITATIVE
                    and call.policy_decision in durable_policy_decisions
                )
                or call.policy_evidence is ToolPolicyEvidence.UNREGISTERED
                or call.policy_evidence is ToolPolicyEvidence.UNEXPOSED
                or (
                    call.policy_evidence is ToolPolicyEvidence.AMBIGUOUS
                    and ambiguous_interrupt_close_intent
                )
                or (
                    call.policy_evidence is None
                    and call.policy_decision in durable_policy_decisions
                )
            )
        ]
        if invalid_planned_calls:
            raise RuntimeError(
                "Policy-planned pending tool round has no authoritative decision for "
                f"registered call {invalid_planned_calls[0].tool_call_id}."
            )
        pending_tool_calls = tool_round_recovery.pending_round_tool_calls(pending_round)
        tool_round_identity = tool_round_recovery.pending_tool_round_identity(pending_round)
        if await transcript_helpers.tool_round_has_result_messages(
            self._session_store,
            session.id,
            pending_tool_calls,
            tool_round_identity=tool_round_identity,
        ):
            raise SessionRuntimePublicationConflict(
                "The durable transcript already closes the pending tool round without "
                "its atomic checkpoint publication."
            )
        insert_at = len(messages) - tail_message_count
        if insert_at < 0:
            raise RuntimeError("Pending tool round recovery received an invalid tail size.")
        if any(call.tool_name == STRUCTURED_OUTPUT_TOOL_NAME for call in pending_round.tool_calls):
            async for event in self._recover_structured_output_tool_round(
                session=session,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
                messages=messages,
                insert_at=insert_at,
                pending_round=pending_round,
                source_checkpoint=checkpoint,
                retry_allowed=session.status == SessionStatus.RUNNING,
                expected_transcript_cursor=expected_transcript_cursor,
                execution_profile=execution_profile,
            ):
                yield event
            return
        legacy_policy_plan_is_ambiguous = pending_round.policy_context_version is None and any(
            call.tool_name in registered_tool_names
            and call.policy_decision not in durable_policy_decisions
            for call in pending_round.tool_calls
        )
        legacy_policy_plan_requires_approval = (
            pending_round.policy_context_version is None
            and any(
                call.tool_name in registered_tool_names
                and call.policy_decision == ToolPolicyDecision.REQUIRE_APPROVAL.value
                for call in pending_round.tool_calls
            )
            and approval_support.pending_approval_from_checkpoint(
                checkpoint,
                redactor=self._secret_redactor,
            )
            is None
        )
        if pending_round.policy_state == "unplanned" and (
            pending_round.policy_context_version == 1
            or legacy_policy_plan_is_ambiguous
            or legacy_policy_plan_requires_approval
        ):
            # A raw round proves only that model output was durably staged. It
            # cannot prove whether a stateful policy had already returned before
            # the process stopped. Replaying authorize() and accepting ALLOW
            # could erase an earlier REQUIRE_APPROVAL result, so recovery
            # constructs a fail-closed manual gate without invoking policy.
            #
            # Legacy unversioned rounds are treated the same way whenever a
            # registered call lacks a complete recognized decision. Absence of
            # a decision is not positive authorization. Unversioned rounds with
            # complete decisions retain those authoritative legacy outcomes.
            if (
                session.status
                not in {
                    SessionStatus.RUNNING,
                    SessionStatus.INTERRUPTING,
                }
                and not incomplete_recovery_claimed
            ):
                raise RuntimeError(
                    "Pending tool round has no durable policy plan; resume it under a "
                    "claimed run fence before recovering tool results."
                )
            replanned_tool_calls = [
                runtime_records.ToolCallRequest(
                    id=call.tool_call_id,
                    name=call.tool_name,
                    arguments=copy_json_value(call.arguments, "arguments"),
                )
                for call in pending_round.tool_calls
            ]
            policy_plan = await self._tool_round_executor.fail_closed_recovery_policy_plan(
                session=session,
                registered_agent=registered_agent,
                tool_calls=replanned_tool_calls,
                request_metadata=pending_round.request_metadata,
                durable_tool_calls=(
                    pending_round.tool_calls
                    if pending_round.policy_context_version is None
                    else None
                ),
                tool_exposure=pending_round.tool_exposure,
            )
            if policy_plan.pending_approval is not None:
                approval_plan = policy_plan.pending_approval
                (
                    approval,
                    approval_events,
                ) = await self._tool_round_executor.checkpoint_pending_tool_approval(
                    session=session,
                    registered_agent=registered_agent,
                    registered_environment=registered_environment,
                    tool_call=approval_plan.call,
                    tool_calls=approval_plan.calls,
                    policy_outcomes=approval_plan.policy_outcomes,
                    active_taint_by_id=policy_plan.active_taint_labels,
                    task_id=pending_round.task_id,
                    policy_result=approval_plan.policy_result,
                    structured_output=pending_round.structured_output,
                    thinking=pending_round.thinking,
                    max_steps=pending_round.max_steps,
                    limits=pending_round.limits,
                    budget_limits=pending_round.budget_limits,
                    retry_policy=pending_round.retry_policy,
                    tool_round_identity=tool_round_recovery.pending_tool_round_identity(
                        pending_round
                    ),
                    deferred_messages=messages[insert_at:],
                    recovered=True,
                )
                for approval_event in approval_events:
                    yield approval_event
                raise ToolApprovalRequired(approval)
            pending_round = await self._tool_round_executor.checkpoint_tool_round_policy_plan(
                session=session,
                registered_agent=registered_agent,
                tool_calls=replanned_tool_calls,
                policy_outcomes=policy_plan.outcomes,
                active_taint_by_id=policy_plan.active_taint_labels,
                tool_round_identity=tool_round_recovery.pending_tool_round_identity(pending_round),
                recovered=True,
            )
            checkpoint = await self._session_store.load_checkpoint(session.id)
        approval_required_calls = [
            call
            for call in pending_round.tool_calls
            if approval_support.effective_tool_policy_evidence(call)
            is ToolPolicyEvidence.AUTHORITATIVE
            and call.policy_decision == ToolPolicyDecision.REQUIRE_APPROVAL.value
        ]
        if approval_required_calls:
            current_checkpoint = await self._session_store.load_checkpoint(session.id)
            paired_approval = approval_support.pending_approval_from_checkpoint(
                current_checkpoint,
                redactor=self._secret_redactor,
            )
            if paired_approval is None:
                if not (
                    session.status == SessionStatus.INTERRUPTING
                    and _approval_interrupt_close_intent_matches(
                        current_checkpoint,
                        pending_round=pending_round,
                    )
                ):
                    raise RuntimeError(
                        "Policy-planned REQUIRE_APPROVAL round has no matching pending approval."
                    )
            elif (
                paired_approval.tool_round_id != pending_round.tool_round_id
                or paired_approval.tool_call_id != approval_required_calls[0].tool_call_id
            ):
                raise RuntimeError(
                    "Policy-planned REQUIRE_APPROVAL round has no matching pending approval."
                )
            else:
                raise RuntimeError(
                    "Pending tool approval must be resolved before recovering its tool round."
                )
        lifecycle_events = await self._load_tool_round_lifecycle_events(
            session_id=session.id,
            pending_round=pending_round,
        )
        recorded_outcomes, started_ids = tool_round_recovery.recorded_tool_outcomes(
            events=lifecycle_events,
            pending_round=pending_round,
        )
        subagent_children: dict[str, Session | None] = {}
        subagent_recovery_checkpoint: dict[str, Any] | None = None
        if any(
            recorded_outcomes.get(call.tool_call_id) is None for call in pending_round.tool_calls
        ):
            subagent_children = await self._subagent_children_by_idempotency_key(session.id)
            subagent_recovery_checkpoint = await self._session_store.load_checkpoint(session.id)
        synthesized_outcomes: list[runtime_records.ToolCallOutcome] = []
        for pending_tool_call in pending_round.tool_calls:
            recorded_outcome = recorded_outcomes.get(pending_tool_call.tool_call_id)
            if recorded_outcome is not None:
                continue

            tool_call = runtime_records.ToolCallRequest(
                id=pending_tool_call.tool_call_id,
                name=pending_tool_call.tool_name,
                arguments=copy_json_value(pending_tool_call.arguments, "arguments"),
            )
            policy_evidence = approval_support.effective_tool_policy_evidence(pending_tool_call)
            if policy_evidence is ToolPolicyEvidence.UNEXPOSED:
                exposure = pending_round.tool_exposure
                if exposure is None:
                    raise RuntimeError(
                        "Unexposed recovered tool call lost its frozen exposure snapshot."
                    )
                if (
                    tool_call.name not in registered_agent.tools
                    or tool_call.name in exposure.tool_names
                ):
                    raise RuntimeError(
                        "Unexposed recovered tool call conflicts with its frozen exposure."
                    )
                synthesized_outcomes.append(
                    runtime_records.ToolCallOutcome(
                        call=runtime_records.ToolCallRequest(
                            id=tool_call.id,
                            name=tool_call.name,
                            arguments={},
                        ),
                        result=unexposed_tool_result(),
                    )
                )
                continue
            expected_idempotency_key = tool_execution.tool_idempotency_key(
                session_id=session.id,
                tool_round_id=pending_round.tool_round_id,
                tool_call_id=pending_tool_call.tool_call_id,
            )
            recovery_arguments = self._subagent_recovery_arguments(
                checkpoint=subagent_recovery_checkpoint,
                parent_session=session,
                tool_name=pending_tool_call.tool_name,
                tool_round_id=pending_round.tool_round_id,
                tool_call_id=pending_tool_call.tool_call_id,
                idempotency_key=expected_idempotency_key,
                fallback=tool_call.arguments,
            )
            registered_tool = registered_agent.tools.get(pending_tool_call.tool_name)
            result: ToolResult | None = None
            if registered_tool is not None and registered_tool.durable_tool_recovery is not None:

                async def load_durable_tool_operation(
                    storage_key: str,
                ) -> dict[str, Any] | None:
                    return await self._session_store.load_session_operation(
                        session.id,
                        storage_key,
                    )

                result = await registered_tool.durable_tool_recovery.reconcile_durable_tool_call(
                    parent_session_id=session.id,
                    parent_run_epoch=(pending_round.source_run_epoch or session.run_epoch),
                    execution_profile_fingerprint=(
                        None if execution_profile is None else execution_profile.fingerprint
                    ),
                    environment_name=environment_name,
                    environment_allocation_fingerprint=(
                        None
                        if registered_environment is None
                        else registered_environment.live_allocation_fingerprint
                    ),
                    model_step_id=pending_round.model_step_id,
                    model_attempt_id=pending_round.model_attempt_id,
                    tool_round_id=pending_round.tool_round_id,
                    tool_call_id=pending_tool_call.tool_call_id,
                    idempotency_key=expected_idempotency_key,
                    arguments=copy_json_value(
                        tool_call.arguments,
                        "durable_tool_recovery.arguments",
                    ),
                    started=pending_tool_call.tool_call_id in started_ids,
                    load_operation=load_durable_tool_operation,
                )
                if result is not None and type(result) is not ToolResult:
                    raise TypeError("Durable tool recovery must return ToolResult or None.")
            reconciled_result = None
            if result is None:
                reconciled_result = await self._reconcile_subagent_child(
                    subagent_children,
                    idempotency_key=expected_idempotency_key,
                    tool_call_id=pending_tool_call.tool_call_id,
                    tool_name=pending_tool_call.tool_name,
                    tool_round_id=pending_round.tool_round_id,
                    arguments=recovery_arguments,
                    parent_session=session,
                    registered_agent=registered_agent,
                )
                result = reconciled_result
            if result is None:
                result = self._reattached_subagent_result(
                    subagent_children,
                    expected_idempotency_key,
                    tool_call_id=pending_tool_call.tool_call_id,
                    tool_name=pending_tool_call.tool_name,
                    tool_round_id=pending_round.tool_round_id,
                    arguments=recovery_arguments,
                    parent_session=session,
                    registered_agent=registered_agent,
                )
            if result is None:
                result = tool_round_recovery.unknown_recovered_tool_result(
                    pending_tool_call=pending_tool_call,
                    pending_round=pending_round,
                    started=pending_tool_call.tool_call_id in started_ids,
                )
            synthesized_outcomes.append(
                runtime_records.ToolCallOutcome(call=tool_call, result=result)
            )

        synthesized_outcomes = tool_results.redact_tool_call_outcomes(
            synthesized_outcomes,
            self._secret_redactor,
        )
        # Classify staged evidence against the durable coverage exactly as it
        # existed when recovery took ownership. Completing the assistant
        # projection below may conservatively add coverage for calls that never
        # produced terminal evidence; that must not retroactively authenticate
        # a sibling result staged under an incomplete dynamic-secret scope.
        recovery_staged_records = tool_round_recovery.staged_terminal_records(pending_round)
        if any(item.event.session_id != session.id for item in recovery_staged_records):
            raise RuntimeError("Staged recovery evidence belongs to a different session.")
        checkpoint, pending_round = await self._complete_recovery_assistant_publication(
            session_id=session.id,
            registered_agent=registered_agent,
            pending_round=pending_round,
            execution_scope_unknown_ids=started_ids,
        )
        if pending_round.assistant_message_state == "quarantined":
            tool_round_recovery.ready_assistant_publication_message(pending_round)
        publication_scope = (
            "unknown"
            if pending_round.assistant_publication is None
            else pending_round.assistant_publication.secret_resolution_scope
        )
        if publication_scope != "static":
            quarantined_records: list[tool_round_recovery.StagedToolCallTerminal] = []
            for staged in recovery_staged_records:
                if staged.hooks_state == "completed":
                    quarantined_records.append(staged)
                    continue
                quarantined_event = tool_round_recovery.hook_scope_unavailable_recovery_event(
                    staged.event
                )
                await self._session_store.transform_checkpoint(
                    session.id,
                    tool_round_recovery.completed_staged_terminal_transform(
                        tool_round_identity=tool_round_recovery.pending_tool_round_identity(
                            pending_round
                        ),
                        event=quarantined_event,
                    ),
                )
                quarantined_records.append(
                    tool_round_recovery.StagedToolCallTerminal(
                        tool_call_id=staged.tool_call_id,
                        event=quarantined_event,
                        hooks_state="completed",
                    )
                )
            recovery_staged_records = quarantined_records
        synthesized_by_id = {outcome.call.id: outcome for outcome in synthesized_outcomes}
        staged_events_by_id = {item.tool_call_id: item.event for item in recovery_staged_records}
        staged_hook_states_by_id = {
            item.tool_call_id: item.hooks_state for item in recovery_staged_records
        }
        durable_terminal_ids = {
            event.payload.get("tool_call_id")
            for event in lifecycle_events
            if event.type in tool_round_recovery._TOOL_ROUND_TERMINAL_EVENT_TYPES
        }
        pending_calls_by_id = {call.tool_call_id: call for call in pending_round.tool_calls}
        started_interaction_by_id = {
            tool_call_id: event.interaction_id
            for event in lifecycle_events
            if event.type is EventType.TOOL_CALL_STARTED
            and type(tool_call_id := event.payload.get("tool_call_id")) is str
        }
        planned_terminal_events: list[Event] = []
        planned_outcomes: list[runtime_records.ToolCallOutcome] = []
        planned_hook_states: list[
            Literal["pending", "finalized", "observational", "completed"]
        ] = []
        for pending_call in pending_round.tool_calls:
            if pending_call.tool_call_id in durable_terminal_ids:
                continue
            staged_event = staged_events_by_id.get(pending_call.tool_call_id)
            if staged_event is not None:
                planned_terminal_events.append(staged_event)
                planned_outcomes.append(
                    resume_ledger.tool_call_outcome_from_terminal_event(
                        event=staged_event,
                        pending_tool_call=pending_calls_by_id[pending_call.tool_call_id],
                    )
                )
                planned_hook_states.append(staged_hook_states_by_id[pending_call.tool_call_id])
                continue
            outcome = synthesized_by_id.get(pending_call.tool_call_id)
            if outcome is None:
                raise RuntimeError("Recovery lost terminal evidence for a pending tool call.")
            policy_evidence = approval_support.effective_tool_policy_evidence(pending_call)
            is_unexposed = policy_evidence is ToolPolicyEvidence.UNEXPOSED
            event_type = (
                EventType.TOOL_CALL_BLOCKED
                if is_unexposed
                else (
                    EventType.TOOL_CALL_FAILED
                    if outcome.result.is_error
                    else EventType.TOOL_CALL_COMPLETED
                )
            )
            exposure = pending_round.tool_exposure if is_unexposed else None
            if is_unexposed and exposure is None:
                raise RuntimeError(
                    "Unexposed recovered terminal lost its frozen exposure snapshot."
                )
            planned_terminal_events.append(
                Event(
                    type=event_type,
                    session_id=session.id,
                    interaction_id=started_interaction_by_id.get(pending_call.tool_call_id),
                    agent_name=registered_agent.spec.name,
                    environment_name=environment_name,
                    tool_name=outcome.call.name,
                    payload={
                        **tool_round_recovery.pending_tool_round_identity(pending_round).payload(),
                        "tool_call_id": outcome.call.id,
                        "idempotency_key": tool_execution.tool_idempotency_key(
                            session_id=session.id,
                            tool_round_id=pending_round.tool_round_id,
                            tool_call_id=outcome.call.id,
                        ),
                        "recovered": True,
                        **(
                            {}
                            if exposure is None
                            else {
                                "blocked_by": "tool_exposure",
                                "reason": NOT_EXPOSED_IN_REQUEST_REASON,
                                "profile_id": exposure.profile_id,
                                "exposure_fingerprint": exposure.fingerprint,
                            }
                        ),
                        **(
                            tool_argument_publication.unavailable_argument_projection().payload_fields()
                            if is_unexposed
                            else {}
                        ),
                        "result": outcome.result.model_dump(),
                    },
                )
            )
            planned_outcomes.append(outcome)
            planned_hook_states.append("completed" if is_unexposed else "finalized")
        tool_round_publication.collect_tool_round_publication_evidence(
            session_id=session.id,
            pending_round=pending_round,
            durable_events=[*lifecycle_events, *planned_terminal_events],
        )

        emitted_events: list[Event] = []
        tool_round_identity = tool_round_recovery.pending_tool_round_identity(pending_round)

        async def complete_recovered_terminal_hooks(event: Event) -> Event:
            await self._session_store.transform_checkpoint(
                session.id,
                tool_round_recovery.completed_staged_terminal_transform(
                    tool_round_identity=tool_round_identity,
                    event=event,
                ),
            )
            return copy_event(event)

        async def record_recovered_terminal_projection(event: Event) -> Event:
            await self._session_store.transform_checkpoint(
                session.id,
                tool_round_recovery.projected_staged_terminal_transform(
                    tool_round_identity=tool_round_identity,
                    event=event,
                ),
            )
            return copy_event(event)

        for expected_outcome, terminal_event, hooks_state in zip(
            planned_outcomes,
            planned_terminal_events,
            planned_hook_states,
            strict=True,
        ):
            if expected_outcome.call.id in staged_events_by_id:
                terminal_event = restore_staged_terminal_authority(
                    terminal_event,
                    session_id=session.id,
                    tool_round_identity=tool_round_identity,
                    tool_exposure=pending_round.tool_exposure,
                )
                terminal_event = web_access_results.restore_persisted_web_access_result_authority(
                    terminal_event
                )
            expected_public_outcome = runtime_records.ToolCallOutcome(
                call=runtime_records.ToolCallRequest(
                    id=expected_outcome.call.id,
                    name=expected_outcome.call.name,
                    arguments={},
                ),
                result=expected_outcome.result,
            )
            async for (
                event,
                emitted_outcome,
            ) in self._tool_round_executor.emit_tool_call_result_with_hooks(
                event=terminal_event,
                session=session,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
                tool_call=expected_outcome.call,
                result=expected_outcome.result,
                task_id=pending_round.task_id,
                execution_profile=execution_profile,
                allow_modification=hooks_state == "pending",
                publish_before_hooks=hooks_state == "observational",
                deferred_terminal_projection_recorder=(
                    record_recovered_terminal_projection
                    if hooks_state == "observational"
                    and expected_outcome.call.id in staged_events_by_id
                    else None
                ),
                deferred_terminal_finalizer=(
                    complete_recovered_terminal_hooks
                    if hooks_state in {"pending", "finalized", "observational"}
                    and expected_outcome.call.id in staged_events_by_id
                    else None
                ),
                hooks_already_completed=hooks_state == "completed",
            ):
                emitted_events.append(event)
                if (
                    emitted_outcome is not None
                    and hooks_state != "pending"
                    and emitted_outcome != expected_public_outcome
                ):
                    raise RuntimeError("Recovered tool-round hooks changed terminal evidence.")

        lifecycle_events = await self._load_tool_round_lifecycle_events(
            session_id=session.id,
            pending_round=pending_round,
        )
        checkpoint = await self._session_store.load_checkpoint(session.id)
        refreshed_pending_round = tool_round_recovery.pending_tool_round_from_checkpoint(checkpoint)
        if (
            refreshed_pending_round is None
            or tool_round_recovery.pending_tool_round_identity(refreshed_pending_round)
            != tool_round_identity
        ):
            raise RuntimeError("Recovered hooks lost their pending tool-round owner.")
        pending_round = refreshed_pending_round
        prepared = tool_round_publication.prepare_tool_round_publication(
            session_id=session.id,
            pending_round=pending_round,
            source_checkpoint=checkpoint,
            durable_events=lifecycle_events,
            expected_statuses={
                SessionStatus.RUNNING,
                SessionStatus.INTERRUPTING,
                SessionStatus.INTERRUPTED,
            },
            expected_run_epoch=session.run_epoch,
            expected_transcript_cursor=expected_transcript_cursor,
        )
        cancellation = await self._publish_tool_round_with_exact_replay(prepared)
        materialized = await self.materialize_expected_deferred_input(
            session.id,
            pending_round.deferred_messages,
            cancellation=cancellation,
        )
        messages[:] = materialized.messages
        cancellation = materialized.cancellation
        for event in emitted_events:
            yield event
        if cancellation is not None:
            raise cancellation

    async def finalize_abandoned_session_run(
        self,
        request: RecoveryAbandonedSessionRequest,
    ) -> None:
        """Best-effort finalization for a live session whose event stream closed."""
        if (
            request.interaction_transition_failures
            and request.interaction_transition is not None
            and await self._record_committed_interaction_transition_cancellation(request)
        ):
            return
        try:
            finalized = await self._abandoned_turn_completed(
                RecoveryAbandonedTurnRequest(
                    session=request.session,
                    registered_agent=request.registered_agent,
                    registered_environment=request.registered_environment,
                    environment_name=request.environment_name,
                    run_started_at=request.run_started_at,
                    usage_tracker=request.turn_usage_tracker,
                    active_run=request.active_run,
                    execution_profile=request.execution_profile,
                )
            )
        except InteractionLifecyclePublicationRejected:
            raise
        except BaseException:
            try:
                finalized = await self._session_store.transition_status(
                    request.session.id,
                    from_statuses={
                        SessionStatus.PENDING,
                        SessionStatus.RUNNING,
                        SessionStatus.INTERRUPTING,
                    },
                    to_status=SessionStatus.INTERRUPTED,
                )
            except KeyError:
                return
            except ValueError:
                loaded = await self._session_store.load(request.session.id)
                if loaded is None or loaded.status is not SessionStatus.INTERRUPTED:
                    return
                finalized = loaded
        payload: dict[str, Any] = {
            "interruption_type": _INTERRUPTION_TYPE_RUNTIME_INTERRUPTED,
            "reason": _ABANDONED_RUN_REASON,
            "abandoned": True,
        }
        if request.interaction_transition_failures:
            copied_failures = copy_durable_json_value(
                list(request.interaction_transition_failures),
                "interaction transition cancellation diagnostics",
            )
            if type(copied_failures) is not list:
                raise TypeError("Interaction transition cancellation diagnostics must be a list.")
            payload["interaction_transition_failures"] = copied_failures

        async def emit_interrupted() -> None:
            async for _ in self._emit_terminal_event_with_hooks(
                RecoveryTerminalEventRequest(
                    event=Event(
                        type=EventType.SESSION_INTERRUPTED,
                        session_id=finalized.id,
                        agent_name=request.registered_agent.spec.name,
                        environment_name=request.environment_name,
                        payload=payload,
                    ),
                    phase=RuntimeHookPhase.AFTER_SESSION_INTERRUPTED,
                    session=finalized,
                    registered_agent=request.registered_agent,
                    registered_environment=request.registered_environment,
                    execution_profile=request.execution_profile,
                )
            ):
                pass

        if request.interaction_transition_failures:
            # These failures are the only durable explanation for an ambiguous
            # transition that exact readback proved absent. Let the owned
            # cancellation cleanup preserve a publication failure instead of
            # silently discarding both pieces of evidence.
            await emit_interrupted()
        else:
            with contextlib.suppress(BaseException):
                await emit_interrupted()

    async def _record_committed_interaction_transition_cancellation(
        self,
        request: RecoveryAbandonedSessionRequest,
    ) -> bool:
        """Record acknowledgement failures only after exact durable readback."""

        if request.interaction_transition is None:
            return False
        return await self._record_durable_interaction_transition_cancellation(
            session=request.session,
            transition=request.interaction_transition,
            failures=request.interaction_transition_failures,
            agent_name=request.registered_agent.spec.name,
            environment_name=request.environment_name,
        )

    async def _record_durable_interaction_transition_cancellation(
        self,
        *,
        session: Session,
        transition: InteractionTransitionSpec,
        failures: tuple[dict[str, Any], ...],
        agent_name: str,
        environment_name: str | None,
    ) -> bool:
        """Record an exact transition failure without resolving mutable registrations."""

        if agent_name != session.agent_name or environment_name != session.environment_name:
            raise RuntimeError(
                "Interaction transition cancellation identity conflicts with its session."
            )
        expected = copy_interaction_transition_spec(transition)
        if (
            expected.event.session_id != session.id
            or expected.event.interaction_id is None
            or expected.event.type
            not in {
                *INTERACTION_TERMINAL_EVENT_TYPES,
                EventType.INTERACTION_PAUSED,
            }
        ):
            raise RuntimeError(
                "Interaction transition cancellation evidence is not a settled event "
                "for the abandoned session."
            )
        receipt = await self._session_store.load_interaction_transition_receipt(
            session.id,
            transition=expected,
        )
        if receipt is None:
            return False
        receipt = InteractionTransitionReceiptResult.model_validate(receipt)
        if receipt.transition != expected or receipt.session.id != session.id:
            raise RuntimeError(
                "Interaction transition cancellation evidence conflicts with its durable receipt."
            )
        copied_failures = copy_durable_json_value(
            list(failures),
            "interaction transition cancellation diagnostics",
        )
        if type(copied_failures) is not list:
            raise TypeError("Interaction transition cancellation diagnostics must be a list.")
        await self._event_writer.persist(
            event_with_runtime_envelope_authority(
                Event(
                    type=(EventType.RUNTIME_INTERACTION_TRANSITION_ACKNOWLEDGEMENT_FAILED),
                    session_id=session.id,
                    agent_name=agent_name,
                    environment_name=environment_name,
                    payload={
                        "transition_event_type": str(expected.event.type),
                        "interaction_transition_failures": copied_failures,
                    },
                ),
                "session_id",
            )
        )
        return True

    async def finalize_abandoned_session_by_id(
        self,
        session_id: str,
        *,
        propagate_interaction_publication_rejection: bool = False,
        registered_agent: runtime_records.RegisteredAgentState | None = None,
        registered_environment: runtime_records.RegisteredEnvironment | None = None,
        execution_profile: ExecutionProfileIdentity | None = None,
    ) -> None:
        """Idempotently finalize a live session when setup-time streaming is abandoned."""
        try:
            session = await self._session_store.load(session_id)
        except Exception:
            return
        if session is None or session.status not in {
            SessionStatus.PENDING,
            SessionStatus.RUNNING,
            SessionStatus.INTERRUPTING,
        }:
            return
        if registered_agent is None:
            try:
                registered_agent = self._resolve_registered_agent(session.agent_name)
            except Exception:
                await self._finalize_abandoned_without_registered_runtime(session.id)
                return
            try:
                registered_environment = self._resolve_registered_environment(
                    session.environment_name
                )
            except Exception:
                await self._finalize_abandoned_without_registered_runtime(session.id)
                return
        elif (
            registered_agent.spec.name != session.agent_name
            or _environment_name(registered_environment) != session.environment_name
        ):
            raise RuntimeError(
                "Frozen abandonment runtime does not match the durable session identity."
            )
        if session.status == SessionStatus.INTERRUPTING:
            try:
                async for _ in self._interrupt_session_for_recovery(
                    RecoveryInterruptionRequest(
                        session=session,
                        registered_agent=registered_agent,
                        registered_environment=registered_environment,
                        environment_name=_environment_name(registered_environment),
                        execution_profile=execution_profile,
                    )
                ):
                    pass
                return
            except BaseException:
                # Preserve the existing best-effort fallback if the durable
                # operator-interruption payload cannot be finalized.
                pass
        try:
            await self.finalize_abandoned_session_run(
                RecoveryAbandonedSessionRequest(
                    session=session,
                    registered_agent=registered_agent,
                    registered_environment=registered_environment,
                    environment_name=_environment_name(registered_environment),
                    execution_profile=execution_profile,
                )
            )
        except InteractionLifecyclePublicationRejected:
            if propagate_interaction_publication_rejection:
                raise
        except BaseException:
            pass

    async def _finalize_abandoned_without_registered_runtime(self, session_id: str) -> None:
        try:
            finalized = await self._session_store.transition_status(
                session_id,
                from_statuses={
                    SessionStatus.PENDING,
                    SessionStatus.RUNNING,
                    SessionStatus.INTERRUPTING,
                },
                to_status=SessionStatus.INTERRUPTED,
            )
        except Exception:
            return
        with contextlib.suppress(BaseException):
            terminal_event = Event(
                type=EventType.SESSION_INTERRUPTED,
                session_id=finalized.id,
                agent_name=finalized.agent_name,
                environment_name=finalized.environment_name,
                payload={
                    "interruption_type": _INTERRUPTION_TYPE_RUNTIME_INTERRUPTED,
                    "reason": _ABANDONED_RUN_REASON,
                    "abandoned": True,
                },
            )
            checkpoint = await self._session_store.load_checkpoint(finalized.id)
            run_operation = _session_run_operation_from_checkpoint(checkpoint)
            if run_operation is not None:
                if run_operation.run_epoch != finalized.run_epoch:
                    raise RuntimeError(
                        "Abandoned session run operation does not match the active run epoch."
                    )
                terminal_event = _event_with_session_run_operation(
                    terminal_event,
                    run_operation,
                )
            await self._event_writer.emit(terminal_event)
            if run_operation is not None:
                await self._clear_session_run_operation(
                    session_id=finalized.id,
                    operation=run_operation,
                    terminal_evidence_durable=True,
                )

    async def recover_incomplete_session(
        self,
        request: IncompleteSessionRecoveryRequest,
        *,
        before_mutation: RecoveryMutationHook | None = None,
    ) -> IncompleteSessionRecoveryResult:
        """Repair one incomplete session without executing providers or tools."""
        session = await self._session_store.load(request.session_id)
        if session is None:
            raise KeyError(f"Session not found: {request.session_id}") from None
        recovered = await self._recover_incomplete_session_scoped(
            session=session,
            inactive_before=request.inactive_before,
            reason=request.reason,
            metadata=request.metadata,
            before_mutation=before_mutation,
        )
        return await self._finish_provider_operation_disposition_after_recovery(
            recovered,
            before_mutation=before_mutation,
        )

    async def _finish_provider_operation_disposition_after_recovery(
        self,
        recovered: IncompleteSessionRecoveryResult,
        *,
        before_mutation: RecoveryMutationHook | None = None,
    ) -> IncompleteSessionRecoveryResult:
        pending_resolution = await load_pending_provider_operation_disposition(
            self._session_store,
            recovered.session_id,
        )
        if pending_resolution is None:
            return recovered
        pending, result = pending_resolution
        if before_mutation is not None:
            await before_mutation()
        if await self._retire_completed_provider_operation_disposition(
            pending=pending,
            result=result,
        ):
            return recovered
        session = await self._require_session(recovered.session_id)
        if (
            pending.action is ProviderOperationResolutionAction.FALLBACK_RETRY
            and session.status is not SessionStatus.INTERRUPTED
        ):
            return recovered

        disposition_events = [
            event
            async for event in self._finish_pending_provider_operation_disposition(
                pending=pending,
                result=result,
            )
        ]
        current = await self._require_session(recovered.session_id)
        retained_actions = tuple(
            action
            for action in recovered.actions
            if action
            not in {
                IncompleteSessionRecoveryAction.SKIPPED_ACTIVE,
                IncompleteSessionRecoveryAction.SKIPPED_TERMINAL,
            }
        )
        return recovered.model_copy(
            update={
                "status": current.status,
                "actions": (
                    *retained_actions,
                    IncompleteSessionRecoveryAction.REPAIRED_PROVIDER_OPERATION_RESOLUTION,
                ),
                "events": (*recovered.events, *disposition_events),
                "message": "Finished the accepted provider-operation resolution.",
            }
        )

    async def _pending_durable_subagent_recovery_guard(
        self,
        *,
        session: Session,
        checkpoint: dict[str, Any] | None,
        previous_status: SessionStatus,
    ) -> IncompleteSessionRecoveryResult | None:
        """Keep one prepared child under its queue owner instead of abandoning it."""

        if session.status is not SessionStatus.PENDING or session.run_epoch != 0:
            return None
        intents = durable_subagent_submissions_from_checkpoint(checkpoint)
        subagent = session.metadata.get("subagent")
        if not intents:
            if type(subagent) is dict and subagent.get("mode") == "durable":
                raise RuntimeError("Pending durable subagent has no prepared execution intent.")
            return None
        if len(intents) != 1:
            raise RuntimeError("Pending durable subagent has ambiguous execution intents.")
        intent = intents[0]
        idempotency_key = intent.idempotency_key
        if (
            type(subagent) is not dict
            or subagent.get("mode") != "durable"
            or subagent.get("idempotency_key") != idempotency_key
        ):
            raise RuntimeError("Pending durable subagent has conflicting spawn metadata.")
        parent = await self._session_store.load(intent.parent_session_id)
        if parent is None:
            raise RuntimeError("Pending durable subagent has no durable parent session.")
        parent_checkpoint = await self._session_store.load_checkpoint(parent.id)
        parent_intent = durable_subagent_submission_from_checkpoint(
            parent_checkpoint,
            idempotency_key=idempotency_key,
        )
        parent_seed = durable_subagent_submission_seed_from_checkpoint(
            parent_checkpoint,
            idempotency_key=idempotency_key,
        )
        parent_receipt = durable_subagent_submission_receipt_from_checkpoint(
            parent_checkpoint,
            idempotency_key=idempotency_key,
        )
        if parent_intent is not None and parent_seed is not None and parent_receipt is None:
            require_durable_subagent_intent_matches_seed(parent_intent, parent_seed)
            parent_authority_matches = parent_intent == intent
        elif parent_intent is None and parent_receipt is not None:
            require_durable_subagent_receipt_matches_intent(parent_receipt, intent)
            if parent_seed is not None:
                require_durable_subagent_intent_matches_seed(intent, parent_seed)
                require_durable_subagent_receipt_matches_seed(parent_receipt, parent_seed)
            parent_authority_matches = True
        else:
            raise RuntimeError(
                "Pending durable subagent has incomplete parent submission authority."
            )
        if (
            not parent_authority_matches
            or intent.child_session_id != session.id
            or intent.parent_session_instance_fingerprint
            != _queued_dispatch_session_instance_fingerprint(parent)
            or session.parent_session_id != parent.id
            or session.causal_budget_id != intent.causal_budget_id
            or session.agent_name != intent.agent_name
            or session.provider_name != intent.child_provider_name
            or session.model != intent.child_model
            or session.environment_name != intent.environment_name
            or session.invocation
            != inherited_session_invocation(
                parent.invocation,
                source=SessionExecutionSource.SUBAGENT,
            )
            or execution_profile_from_session_metadata(session.metadata)
            != intent.child_execution_profile
            or session.metadata.get("subagent") != intent.request.metadata.get("subagent")
        ):
            raise RuntimeError(
                "Pending durable subagent conflicts with its prepared execution authority."
            )

        if self._task_store is None:
            return IncompleteSessionRecoveryResult(
                session_id=session.id,
                previous_status=previous_status,
                status=session.status,
                actions=(IncompleteSessionRecoveryAction.SKIPPED_ACTIVE,),
                events=(),
                message=(
                    "Prepared durable subagent requires its task store for reconciliation; "
                    "generic abandonment recovery skipped."
                ),
            )
        task = await self._task_store.load_task(intent.queue_task_id)
        if task is None:
            return IncompleteSessionRecoveryResult(
                session_id=session.id,
                previous_status=previous_status,
                status=session.status,
                actions=(IncompleteSessionRecoveryAction.SKIPPED_ACTIVE,),
                events=(),
                message=(
                    "Prepared durable subagent is awaiting recoverable parent queue "
                    "publication; generic abandonment recovery skipped."
                ),
            )
        envelope = _new_prepared_subagent_dispatch_envelope(
            intent=intent,
            session_instance_fingerprint=_queued_dispatch_session_instance_fingerprint(session),
        )
        if not _task_matches_queued_dispatch(
            task,
            task_type=intent.queue_task_type,
            parent_task_id=intent.parent_task_id,
            envelope=envelope,
        ):
            raise RuntimeError("Pending durable subagent queue task has conflicting authority.")
        parent_binding = await self._session_store.load_invocation_snapshot(parent.id)
        if parent_binding is None:
            raise RuntimeError("Pending durable subagent parent invocation is unavailable.")
        _require_dispatch_task_authority(
            task,
            envelope=envelope,
            session_binding=parent_binding,
            task_type=intent.queue_task_type,
        )
        if task.status is TaskStatus.COMPLETED:
            raise RuntimeError("Pending durable subagent conflicts with a terminal queue task.")
        if task.status in {TaskStatus.FAILED, TaskStatus.CANCELLED}:
            # Exact queue authority proves that no worker can admit this child.
            # Let ordinary claimed recovery durably close the pristine pending
            # session rather than preserving it as live work forever.
            return None
        return IncompleteSessionRecoveryResult(
            session_id=session.id,
            previous_status=previous_status,
            status=session.status,
            actions=(IncompleteSessionRecoveryAction.SKIPPED_ACTIVE,),
            events=(),
            message=(
                "Prepared durable subagent remains owned by its exact nonterminal queue task; "
                "generic abandonment recovery skipped."
            ),
        )

    async def recover_incomplete_sessions(
        self,
        request: IncompleteSessionsRecoveryRequest,
        *,
        before_recovery: IncompleteRecoveryScopeHook | None = None,
        before_mutation: IncompleteRecoveryScopeHook | None = None,
        after_recovery: IncompleteRecoveryScopeHook | None = None,
        reconcile_result: IncompleteRecoveryResultHook | None = None,
    ) -> IncompleteSessionsRecoveryPage:
        """Fault-isolate one bounded, resumable recovery page."""
        requested_statuses = tuple(
            status for status in _INCOMPLETE_RECOVERY_STATUS_ORDER if status in request.statuses
        )
        start_index = 0
        initial_session_cursor: str | None = None
        if request.cursor is not None:
            cursor_status, initial_session_cursor = _decode_incomplete_recovery_cursor(
                request.cursor,
                request=request,
            )
            start_index = requested_statuses.index(cursor_status)

        results: list[IncompleteSessionRecoveryResult] = []
        result_session_ids: set[str] = set()
        inspected_session_count = 0
        store_page_count = 0

        def continuation_after_status(status_index: int) -> str | None:
            next_index = status_index + 1
            if next_index >= len(requested_statuses):
                return None
            return _encode_incomplete_recovery_cursor(
                status=requested_statuses[next_index],
                session_cursor=None,
                request=request,
            )

        for status_index in range(start_index, len(requested_statuses)):
            status = requested_statuses[status_index]
            terminal_status = status in _RECOVERY_RESUMABLE_SESSION_STATUSES
            cursor = initial_session_cursor if status_index == start_index else None
            seen_cursors = set() if cursor is None else {cursor}
            while (
                len(results) < request.limit and inspected_session_count < request.inspection_limit
            ):
                inspection_remaining = request.inspection_limit - inspected_session_count
                result_remaining = request.limit - len(results)
                # SessionStore cursors are opaque. Never stop partway through a
                # store page and try to synthesize a cursor from a Session:
                # custom stores may not use Cayu's built-in cursor encoding.
                # With at most one result per candidate and a page no larger
                # than the remaining result capacity, either bound can be
                # reached only on the final candidate in this store page.
                query_limit = min(1000, inspection_remaining, result_remaining)
                page = await self._session_store.list_sessions(
                    SessionQuery(
                        status=status,
                        last_activity_before=request.inactive_before,
                        limit=query_limit,
                        cursor=cursor,
                        order_by=SessionOrder.UPDATED_AT_DESC,
                    )
                )
                store_page_count += 1
                if not page.sessions:
                    if page.next_cursor is not None:
                        raise RuntimeError(
                            "Session store returned an empty recovery page with a cursor."
                        )
                    if store_page_count >= _INCOMPLETE_RECOVERY_MAX_STORE_PAGES:
                        return IncompleteSessionsRecoveryPage(
                            results=tuple(results),
                            inspected_session_count=inspected_session_count,
                            next_cursor=continuation_after_status(status_index),
                        )
                    break
                if len(page.sessions) > query_limit:
                    raise RuntimeError(
                        "Session store returned more recovery candidates than requested."
                    )
                encoded_page_cursor: str | None = None
                if page.next_cursor is not None:
                    if page.next_cursor in seen_cursors:
                        raise RuntimeError(
                            "Session store returned a repeated cursor during "
                            "incomplete-session recovery."
                        )
                    encoded_page_cursor = _encode_incomplete_recovery_cursor(
                        status=status,
                        session_cursor=page.next_cursor,
                        request=request,
                    )

                for candidate_index, candidate in enumerate(page.sessions):
                    inspected_session_count += 1
                    if (
                        request.inactive_before is not None
                        and candidate.last_activity_at > request.inactive_before
                    ) or candidate.id in result_session_ids:
                        result = None
                    else:
                        result = await self._recover_incomplete_session_fault_isolated(
                            session=candidate,
                            request=request,
                            before_recovery=before_recovery,
                            before_mutation=before_mutation,
                            after_recovery=after_recovery,
                            reconcile_result=reconcile_result,
                        )
                    if result is not None and not (
                        terminal_status
                        and result.actions == (IncompleteSessionRecoveryAction.SKIPPED_TERMINAL,)
                    ):
                        results.append(result)
                        result_session_ids.add(result.session_id)

                    result_limit_reached = len(results) >= request.limit
                    inspection_limit_reached = inspected_session_count >= request.inspection_limit
                    if not result_limit_reached and not inspection_limit_reached:
                        continue

                    if candidate_index + 1 < len(page.sessions):
                        raise RuntimeError(
                            "Incomplete-session recovery reached a page bound before "
                            "consuming the store page."
                        )
                    next_cursor = (
                        encoded_page_cursor
                        if encoded_page_cursor is not None
                        else continuation_after_status(status_index)
                    )
                    return IncompleteSessionsRecoveryPage(
                        results=tuple(results),
                        inspected_session_count=inspected_session_count,
                        next_cursor=next_cursor,
                    )

                if store_page_count >= _INCOMPLETE_RECOVERY_MAX_STORE_PAGES:
                    next_cursor = (
                        encoded_page_cursor
                        if encoded_page_cursor is not None
                        else continuation_after_status(status_index)
                    )
                    return IncompleteSessionsRecoveryPage(
                        results=tuple(results),
                        inspected_session_count=inspected_session_count,
                        next_cursor=next_cursor,
                    )
                if page.next_cursor is None:
                    break
                seen_cursors.add(page.next_cursor)
                cursor = page.next_cursor

            initial_session_cursor = None

        return IncompleteSessionsRecoveryPage(
            results=tuple(results),
            inspected_session_count=inspected_session_count,
            next_cursor=None,
        )

    async def _recover_incomplete_session_fault_isolated(
        self,
        *,
        session: Session,
        request: IncompleteSessionsRecoveryRequest,
        before_recovery: IncompleteRecoveryScopeHook | None,
        before_mutation: IncompleteRecoveryScopeHook | None,
        after_recovery: IncompleteRecoveryScopeHook | None,
        reconcile_result: IncompleteRecoveryResultHook | None,
    ) -> IncompleteSessionRecoveryResult:
        try:
            if before_recovery is not None:
                await before_recovery(session.id)
            try:
                result = await self._recover_incomplete_session_scoped(
                    session=session,
                    inactive_before=request.inactive_before,
                    reason=request.reason,
                    metadata=request.metadata,
                    before_mutation=(
                        None if before_mutation is None else lambda: before_mutation(session.id)
                    ),
                )
                result = await self._finish_provider_operation_disposition_after_recovery(
                    result,
                    before_mutation=(
                        None if before_mutation is None else lambda: before_mutation(session.id)
                    ),
                )
                return result if reconcile_result is None else await reconcile_result(result)
            finally:
                if after_recovery is not None:
                    await after_recovery(session.id)
        except Exception as exc:
            diagnostic = exception_diagnostic(
                exc,
                empty_message="recovery failed",
                nonportable_message="Recovery failed with a non-portable diagnostic.",
                redactor=self._secret_redactor,
            )
            logger.warning(
                "Recovery failed for session %s (agent %s): error_type=%s error=%s",
                session.id,
                session.agent_name,
                diagnostic.error_type,
                diagnostic.message,
            )
            try:
                reloaded = await self._session_store.load(session.id)
            except Exception:
                reloaded = None
            return IncompleteSessionRecoveryResult(
                session_id=session.id,
                previous_status=session.status,
                status=session.status if reloaded is None else reloaded.status,
                actions=(IncompleteSessionRecoveryAction.FAILED,),
                message=bound_diagnostic_text(
                    f"Recovery failed: {diagnostic.error_type}: {diagnostic.message}"
                ),
            )

    async def _recover_incomplete_session_scoped(
        self,
        *,
        session: Session,
        inactive_before: datetime | None,
        reason: str,
        metadata: dict[str, Any],
        before_mutation: RecoveryMutationHook | None = None,
    ) -> IncompleteSessionRecoveryResult:
        reason = require_clean_nonblank(reason, "reason")
        metadata = copy_json_value(metadata, "metadata")
        previous_status = session.status

        if self._session_control.has_active_tasks(session.id):
            return IncompleteSessionRecoveryResult(
                session_id=session.id,
                previous_status=previous_status,
                status=session.status,
                actions=(IncompleteSessionRecoveryAction.SKIPPED_ACTIVE,),
                events=(),
                message="Session has active work in this CayuApp process; recovery skipped.",
            )

        return await self._recover_incomplete_session_owned(
            session=session,
            inactive_before=inactive_before,
            reason=reason,
            metadata=metadata,
            previous_status=previous_status,
            before_mutation=before_mutation,
        )

    async def _recover_incomplete_session_owned(
        self,
        *,
        session: Session,
        inactive_before: datetime | None,
        reason: str,
        metadata: dict[str, Any],
        previous_status: SessionStatus,
        before_mutation: RecoveryMutationHook | None,
    ) -> IncompleteSessionRecoveryResult:

        mutation_admitted = False

        async def admit_before_mutation() -> None:
            nonlocal mutation_admitted
            if mutation_admitted or before_mutation is None:
                return
            await before_mutation()
            mutation_admitted = True

        checkpoint = await self._session_store.load_checkpoint(session.id)
        active_invocation_profile = active_invocation_execution_profile_from_checkpoint(checkpoint)
        pending_provider_disposition = await load_pending_provider_operation_disposition(
            self._session_store,
            session.id,
            checkpoint=checkpoint,
        )
        pending_provider_disposition_effect_is_durable = False
        if pending_provider_disposition is not None:
            (
                pending_disposition_record,
                pending_disposition_result,
            ) = pending_provider_disposition
            pending_provider_disposition_effect_is_durable = (
                await self._provider_operation_disposition_effect_is_durable(
                    pending=pending_disposition_record,
                    result=pending_disposition_result,
                )
            )
        if active_invocation_profile is not None and not (
            active_invocation_execution_profile_matches_session_epoch(
                active_invocation_profile,
                session_id=session.id,
                run_epoch=session.run_epoch,
            )
        ):
            raise RuntimeError(
                "Active invocation execution profile does not match the recovery epoch."
            )
        durable_child_guard = await self._pending_durable_subagent_recovery_guard(
            session=session,
            checkpoint=checkpoint,
            previous_status=previous_status,
        )
        if durable_child_guard is not None:
            return durable_child_guard
        pending_approval = approval_support.pending_approval_from_checkpoint(
            checkpoint,
            redactor=self._secret_redactor,
            consume_on_rejection=True,
        )
        pending_user_input = pending_user_input_from_checkpoint(
            checkpoint,
            redactor=self._secret_redactor,
            consume_on_rejection=True,
        )
        pending_tool_round = tool_round_recovery.pending_tool_round_from_checkpoint(
            checkpoint,
            redactor=self._secret_redactor,
            consume_on_rejection=True,
        )
        workspace_observations = workspace_observations_from_checkpoint(checkpoint)
        self._validate_workspace_observation_recovery_authority(
            session=session,
            observations=workspace_observations,
            pending_round=pending_tool_round,
            execution_profile_snapshot=active_invocation_profile,
        )
        deferred_input = await self._session_store.load_deferred_interaction_input(session.id)
        active_model_completion = await self._session_store.load_active_model_completion_stage(
            session.id
        )
        terminal_repair_required = False
        if session.status in _RECOVERY_RESUMABLE_SESSION_STATUSES:
            terminal_repair = await self._terminal_evidence_repair_required(
                session=session,
                checkpoint=checkpoint,
            )
            terminal_repair_required = terminal_repair
            has_pending_work = (
                pending_approval is not None
                or pending_user_input is not None
                or pending_tool_round is not None
                or deferred_input is not None
                or active_model_completion is not None
                or bool(workspace_observations)
                or pending_provider_disposition is not None
            )
            if not terminal_repair and not has_pending_work:
                if active_invocation_profile is not None and not (
                    active_invocation_execution_profile_is_released(
                        active_invocation_profile,
                        session_id=session.id,
                        run_epoch=session.run_epoch,
                    )
                ):
                    await admit_before_mutation()
                    return await self._settle_terminal_invocation_closure_owned(
                        session=session,
                        inactive_before=inactive_before,
                        previous_status=previous_status,
                        execution_profile_snapshot=active_invocation_profile,
                    )
                return IncompleteSessionRecoveryResult(
                    session_id=session.id,
                    previous_status=previous_status,
                    status=session.status,
                    actions=(IncompleteSessionRecoveryAction.SKIPPED_TERMINAL,),
                    events=(),
                    message="Session is terminal and has durable terminal evidence.",
                )
            if terminal_repair and not has_pending_work:
                await admit_before_mutation()
                return await self._repair_terminal_evidence_owned(
                    session=session,
                    inactive_before=inactive_before,
                    previous_status=previous_status,
                )

        try:
            registered_agent = self._resolve_registered_agent(session.agent_name)
        except KeyError:
            if terminal_repair_required:
                await admit_before_mutation()
                return await self._repair_terminal_evidence_owned(
                    session=session,
                    inactive_before=inactive_before,
                    previous_status=previous_status,
                )
            return IncompleteSessionRecoveryResult(
                session_id=session.id,
                previous_status=previous_status,
                status=session.status,
                actions=(IncompleteSessionRecoveryAction.SKIPPED_UNREGISTERED_AGENT,),
                events=(),
                message=(f"Agent not registered: {session.agent_name!r}; session left untouched."),
            )
        try:
            registered_environment = self._resolve_registered_environment(session.environment_name)
        except KeyError:
            if terminal_repair_required:
                await admit_before_mutation()
                return await self._repair_terminal_evidence_owned(
                    session=session,
                    inactive_before=inactive_before,
                    previous_status=previous_status,
                )
            raise
        registered_provider = self._resolve_registered_provider(session.provider_name)
        if workspace_observations:
            # A static registration already provides the complete stable
            # environment authority. Reject a foreign lifecycle before profile
            # continuation or the recovery claim can mutate durable ownership.
            # Factory templates remain deliberately unmaterialized here.
            self._validate_workspace_observation_recovery_authority(
                session=session,
                observations=workspace_observations,
                pending_round=pending_tool_round,
                execution_profile_snapshot=active_invocation_profile,
                registered_environment=registered_environment,
            )
        pre_admission_profiled_session = False
        if (
            session.status is SessionStatus.PENDING
            and session.run_epoch == 0
            and active_invocation_profile is None
            and pending_approval is None
            and pending_user_input is None
            and pending_tool_round is None
            and deferred_input is None
            and active_model_completion is None
            and not workspace_observations
            and pending_provider_disposition is None
            and EXECUTION_PROFILE_METADATA_KEY in session.metadata
        ):
            interaction_records = await self._session_store.query_events(
                EventQuery(
                    session_id=session.id,
                    event_types=INTERACTION_LIFECYCLE_EVENT_TYPES,
                    order_by=EventOrder.SEQUENCE_DESC,
                    limit=1,
                )
            )
            if not interaction_records:
                # Presence of the reserved metadata key is not authority. A
                # pristine session is exempt only when its complete baseline
                # survives defensive reconstruction.
                execution_profile_from_session_metadata(session.metadata)
                pre_admission_profiled_session = True
        requires_execution_profile = (
            pending_approval is not None
            or pending_user_input is not None
            or pending_tool_round is not None
            or deferred_input is not None
            or active_model_completion is not None
            or bool(workspace_observations)
            or pending_provider_disposition is not None
            or active_invocation_profile is not None
            or (
                not pre_admission_profiled_session
                and session.status not in _RECOVERY_RESUMABLE_SESSION_STATUSES
                and EXECUTION_PROFILE_METADATA_KEY in session.metadata
            )
        )
        execution_profile_snapshot = None
        budget_policy_snapshot: BudgetPolicy | None = None
        if requires_execution_profile:
            budget_policy_snapshot = copy_budget_policy(self._resolve_budget_policy())
            pending_disposition = (
                None if pending_provider_disposition is None else pending_provider_disposition[0]
            )
            execution_profile_snapshot = await self._validate_execution_profile_continuation(
                session,
                checkpoint,
                registered_agent,
                registered_provider,
                budget_policy=budget_policy_snapshot,
                require_open_interaction=not (
                    pending_provider_disposition_effect_is_durable
                    or (
                        pending_disposition is not None
                        and pending_disposition.action is ProviderOperationResolutionAction.FAIL
                        and session.status is SessionStatus.FAILED
                    )
                ),
                additional_profile_fingerprints=(
                    ()
                    if pending_disposition is None
                    else (pending_disposition.execution_profile_fingerprint,)
                ),
            )
        claim: _IncompleteRecoveryClaim | None = None
        authoritative_failure: BaseException | None = None
        try:
            await admit_before_mutation()
            claim = await self._claim_incomplete_recovery(
                session=session,
                inactive_before=inactive_before,
                execution_profile_snapshot=execution_profile_snapshot,
            )
            if claim is None:
                current = await self._require_session(session.id)
                return IncompleteSessionRecoveryResult(
                    session_id=session.id,
                    previous_status=previous_status,
                    status=current.status,
                    actions=(IncompleteSessionRecoveryAction.SKIPPED_ACTIVE,),
                    events=(),
                    message="Session activity or recovery ownership changed; recovery skipped.",
                )
            return await self._recover_incomplete_session_with_heartbeat(
                claim=claim,
                recovery=lambda: self._recover_incomplete_session(
                    session=claim.session,
                    session_before_fence=claim.session_before_fence,
                    previous_status=previous_status,
                    inactive_before=inactive_before,
                    reason=reason,
                    metadata=metadata,
                    registered_agent=registered_agent,
                    registered_provider=registered_provider,
                    registered_environment=registered_environment,
                    claim_id=claim.claim_id,
                    execution_profile_snapshot=execution_profile_snapshot,
                    budget_policy=budget_policy_snapshot,
                ),
            )
        except BaseException as exc:
            authoritative_failure = exc
            raise
        finally:
            if claim is not None:
                await self._cleanup_incomplete_recovery_claim(
                    session_id=session.id,
                    claim_id=claim.claim_id,
                    authoritative_failure=authoritative_failure,
                    release_environment_cleanup=(
                        pending_provider_disposition is not None
                        and pending_provider_disposition[0].action
                        is ProviderOperationResolutionAction.FALLBACK_RETRY
                    ),
                    execution_profile=(
                        None
                        if execution_profile_snapshot is None
                        else execution_profile_snapshot.profile
                    ),
                )

    async def _settle_terminal_invocation_closure_owned(
        self,
        *,
        session: Session,
        inactive_before: datetime | None,
        previous_status: SessionStatus,
        execution_profile_snapshot: ActiveInvocationExecutionProfile,
    ) -> IncompleteSessionRecoveryResult:
        """Fence a dead terminal owner whose hooks never released its run epoch."""

        claim: _IncompleteRecoveryClaim | None = None
        authoritative_failure: BaseException | None = None
        try:
            claim = await self._claim_incomplete_recovery(
                session=session,
                inactive_before=inactive_before,
                execution_profile_snapshot=execution_profile_snapshot,
            )
            if claim is None:
                current = await self._require_session(session.id)
                return IncompleteSessionRecoveryResult(
                    session_id=session.id,
                    previous_status=previous_status,
                    status=current.status,
                    actions=(IncompleteSessionRecoveryAction.SKIPPED_ACTIVE,),
                    events=(),
                    message="Terminal invocation ownership is still active.",
                )
            return IncompleteSessionRecoveryResult(
                session_id=session.id,
                previous_status=previous_status,
                status=claim.session.status,
                actions=(IncompleteSessionRecoveryAction.REPAIRED_TERMINAL_OWNERSHIP,),
                events=(),
                message="Recovered terminal invocation ownership after worker loss.",
            )
        except BaseException as exc:
            authoritative_failure = exc
            raise
        finally:
            if claim is not None:
                await self._cleanup_incomplete_recovery_claim(
                    session_id=session.id,
                    claim_id=claim.claim_id,
                    authoritative_failure=authoritative_failure,
                )

    async def _terminal_evidence_repair_required(
        self,
        *,
        session: Session,
        checkpoint: dict[str, Any] | None,
    ) -> bool:
        inspection = await self._inspect_terminal_evidence(
            session=session,
            checkpoint=checkpoint,
        )
        return (
            (inspection.event is None and inspection.terminal_event_required)
            or inspection.pending_interrupt_payload is not None
            or inspection.run_operation is not None
            or (checkpoint is not None and _INCOMPLETE_RECOVERY_CLAIM_CHECKPOINT_KEY in checkpoint)
        )

    async def _reconcile_terminal_evidence_before_continuation(
        self,
        *,
        session: Session,
        checkpoint: dict[str, Any] | None,
    ) -> tuple[Session, dict[str, Any] | None]:
        """Finish a previous run's terminal publication before claiming the next run."""
        existing_claim = _incomplete_recovery_claim_from_checkpoint(checkpoint)
        now = self._clock()
        _require_aware_datetime(now, "terminal evidence reconciliation clock")
        if existing_claim is not None and existing_claim[1] > now:
            raise RuntimeError("Session has an active incomplete-session recovery operation.")
        if session.status not in _RECOVERY_RESUMABLE_SESSION_STATUSES:
            if _session_run_operation_from_checkpoint(checkpoint) is None:
                return session, checkpoint
            raise RuntimeError(
                "A non-terminal session retains incomplete terminal evidence for a prior run."
            )
        if not await self._terminal_evidence_repair_required(
            session=session,
            checkpoint=checkpoint,
        ):
            return session, checkpoint
        if self._session_control.has_active_tasks(session.id):
            raise RuntimeError(
                f"Session has active work while terminal evidence is incomplete: {session.id}"
            )
        await self._repair_terminal_evidence_owned(
            session=session,
            inactive_before=None,
            previous_status=session.status,
        )
        current = await self._require_session(session.id)
        current_checkpoint = await self._session_store.load_checkpoint(session.id)
        if await self._terminal_evidence_repair_required(
            session=current,
            checkpoint=current_checkpoint,
        ):
            current_claim = _incomplete_recovery_claim_from_checkpoint(current_checkpoint)
            current_now = self._clock()
            _require_aware_datetime(
                current_now,
                "terminal evidence reconciliation clock",
            )
            if current_claim is not None and current_claim[1] > current_now:
                raise RuntimeError("Session has an active incomplete-session recovery operation.")
            raise RuntimeError(
                "Session terminal evidence recovery did not finish the previous run boundary."
            )
        return current, current_checkpoint

    async def _inspect_terminal_evidence(
        self,
        *,
        session: Session,
        checkpoint: dict[str, Any] | None,
    ) -> _TerminalEvidenceInspection:
        expected_event_type = _TERMINAL_EVENT_TYPE_BY_STATUS.get(session.status)
        if expected_event_type is None:
            raise ValueError(f"Session is not terminal: {session.status}.")
        run_operation = _session_run_operation_from_checkpoint(checkpoint)

        pending_interrupt_payload: dict[str, Any] | None = None
        pending_interrupt_request_id: str | None = None
        if checkpoint is not None and _PENDING_SESSION_INTERRUPT_CHECKPOINT_KEY in checkpoint:
            marker = checkpoint[_PENDING_SESSION_INTERRUPT_CHECKPOINT_KEY]
            if type(marker) is not dict:
                raise ValueError("Pending session interrupt checkpoint must be an object.")
            pending_interrupt_payload = copy_json_value(marker, "pending_session_interrupt")
            if session.status != SessionStatus.INTERRUPTED:
                raise RuntimeError(
                    "Terminal evidence is contradictory: a non-interrupted session retains "
                    "a pending interruption marker."
                )
            pending_interrupt_request_id = interruption_request_id_from_payload(
                pending_interrupt_payload
            )
            if pending_interrupt_request_id is None:
                raise RuntimeError(
                    "Terminal evidence is not repairable: the pending interruption marker "
                    "has no stable request identity."
                )

        pending_approval = approval_support.pending_approval_from_checkpoint(
            checkpoint,
            redactor=self._secret_redactor,
            consume_on_rejection=True,
        )
        pending_user_input = pending_user_input_from_checkpoint(
            checkpoint,
            redactor=self._secret_redactor,
            consume_on_rejection=True,
        )
        pending_tool_round = tool_round_recovery.pending_tool_round_from_checkpoint(
            checkpoint,
            redactor=self._secret_redactor,
            consume_on_rejection=True,
        )
        approval_owns_tool_round = False
        if pending_approval is not None and pending_tool_round is not None:
            _pending_approval_and_round_for_atomic_claim(
                checkpoint,
                approval_id=pending_approval.approval_id,
                tool_round_id=pending_approval.tool_round_id,
                gating_tool_call_id=pending_approval.tool_call_id,
                redactor=self._secret_redactor,
            )
            approval_owns_tool_round = True
        pending_actions = tuple(
            action
            for action in (
                pending_approval,
                pending_user_input,
                None if approval_owns_tool_round else pending_tool_round,
            )
            if action is not None
        )
        if len(pending_actions) > 1:
            raise RuntimeError(
                "Terminal evidence is not repairable: the checkpoint contains "
                "conflicting pending actions."
            )
        pending_action_interrupt_payload: dict[str, Any] | None = None
        if pending_approval is not None and session.status == SessionStatus.INTERRUPTED:
            pending_action_interrupt_payload = {
                "interruption_type": _INTERRUPTION_TYPE_TOOL_APPROVAL_REQUIRED,
                "model_step_id": pending_approval.model_step_id,
                "model_attempt_id": pending_approval.model_attempt_id,
                "tool_round_id": pending_approval.tool_round_id,
                **approval_support.bounded_pending_approval_event_payload(
                    pending_approval,
                    redactor=self._secret_redactor,
                ),
            }
        elif pending_user_input is not None and session.status == SessionStatus.INTERRUPTED:
            pending_action_interrupt_payload = {
                "interruption_type": _INTERRUPTION_TYPE_USER_INPUT_REQUIRED,
                "model_step_id": pending_user_input.model_step_id,
                "model_attempt_id": pending_user_input.model_attempt_id,
                "tool_round_id": pending_user_input.tool_round_id,
                **pending_user_input_interruption_payload(pending_user_input),
            }
        elif pending_tool_round is not None and session.status == SessionStatus.INTERRUPTED:
            pending_action_interrupt_payload = {
                **tool_round_recovery.pending_tool_round_identity(pending_tool_round).payload(),
                "interruption_type": _INTERRUPTION_TYPE_RUNTIME_INTERRUPTED,
                "reason": "terminal_event_evidence_repaired",
                "recovered": True,
            }

        evidence_records = await self._session_store.query_events(
            EventQuery(
                session_id=session.id,
                event_types=TERMINAL_EVIDENCE_EVENT_TYPES,
                order_by=EventOrder.SEQUENCE_DESC,
                limit=TERMINAL_EVIDENCE_QUERY_LIMIT,
            )
        )
        classification = classify_current_terminal_evidence(
            evidence_events=tuple(record.event for record in evidence_records),
            expected_event_type=expected_event_type,
            run_operation_id=(None if run_operation is None else run_operation.operation_id),
            interruption_request_id=pending_interrupt_request_id,
        )
        terminal_events = classification.events
        if classification.run_operation_conflict:
            raise RuntimeError(
                "Terminal evidence is contradictory: the interruption event and "
                "pending run operation have different identities."
            )
        if any(event.type != expected_event_type for event in terminal_events):
            raise RuntimeError(
                "Terminal evidence is contradictory: the durable event type does not "
                f"match session status {session.status.value}."
            )
        if len(terminal_events) > 1:
            raise RuntimeError(
                "Terminal evidence is contradictory: more than one terminal event exists "
                "for the current run."
            )

        existing_event = None if not terminal_events else terminal_events[0].model_copy(deep=True)
        return _TerminalEvidenceInspection(
            event=existing_event,
            pending_interrupt_payload=pending_interrupt_payload,
            pending_action_interrupt_payload=pending_action_interrupt_payload,
            run_operation=run_operation,
            terminal_event_required=(
                run_operation is not None
                or pending_interrupt_payload is not None
                or pending_action_interrupt_payload is not None
                or classification.latest_lifecycle_event_type != EventType.SESSION_FORKED
            ),
        )

    async def _repair_terminal_evidence_owned(
        self,
        *,
        session: Session,
        inactive_before: datetime | None,
        previous_status: SessionStatus,
    ) -> IncompleteSessionRecoveryResult:
        claim: _IncompleteRecoveryClaim | None = None
        authoritative_failure: BaseException | None = None
        try:
            claim = await self._claim_incomplete_recovery(
                session=session,
                inactive_before=inactive_before,
            )
            if claim is None:
                current = await self._require_session(session.id)
                return IncompleteSessionRecoveryResult(
                    session_id=session.id,
                    previous_status=previous_status,
                    status=current.status,
                    actions=(IncompleteSessionRecoveryAction.SKIPPED_ACTIVE,),
                    events=(),
                    message="Session activity or recovery ownership changed; recovery skipped.",
                )
            return await self._recover_incomplete_session_with_heartbeat(
                claim=claim,
                recovery=lambda: self._repair_terminal_evidence(
                    session=claim.session,
                    terminal_run_epoch=claim.session_before_fence.run_epoch,
                    terminal_timestamp=claim.session_before_fence.updated_at,
                    previous_status=previous_status,
                    claim_id=claim.claim_id,
                ),
            )
        except BaseException as exc:
            authoritative_failure = exc
            raise
        finally:
            if claim is not None:
                await self._cleanup_incomplete_recovery_claim(
                    session_id=session.id,
                    claim_id=claim.claim_id,
                    authoritative_failure=authoritative_failure,
                )

    async def _repair_terminal_evidence(
        self,
        *,
        session: Session,
        terminal_run_epoch: int,
        terminal_timestamp: datetime,
        previous_status: SessionStatus,
        claim_id: str,
    ) -> IncompleteSessionRecoveryResult:
        checkpoint = await self._session_store.load_checkpoint(session.id)
        inspection = await self._inspect_terminal_evidence(
            session=session,
            checkpoint=checkpoint,
        )
        terminal_event = inspection.event
        if terminal_event is None and inspection.terminal_event_required:
            terminal_event = await self._persist_terminal_evidence_repair_event(
                self._terminal_evidence_repair_event(
                    session=session,
                    terminal_run_epoch=terminal_run_epoch,
                    terminal_timestamp=terminal_timestamp,
                    pending_interrupt_payload=inspection.pending_interrupt_payload,
                    pending_action_interrupt_payload=(inspection.pending_action_interrupt_payload),
                    run_operation=inspection.run_operation,
                )
            )

        if inspection.pending_interrupt_payload is not None:
            await self._clear_repaired_pending_interrupt(
                session_id=session.id,
                claim_id=claim_id,
                expected_payload=inspection.pending_interrupt_payload,
            )
        if inspection.run_operation is not None:
            await self._clear_session_run_operation(
                session_id=session.id,
                operation=inspection.run_operation,
                required_claim_id=claim_id,
                terminal_evidence_durable=True,
            )

        if terminal_event is not None:
            try:
                await self._event_writer.fan_out_persisted([terminal_event])
            except Exception as exc:
                logger.warning(
                    "Terminal evidence was repaired but durable side-effect delivery remains "
                    "pending: session_id=%s event_id=%s error_type=%s",
                    session.id,
                    terminal_event.id,
                    type(exc).__name__,
                )

        current = await self._require_session(session.id)
        if current.status != session.status or current.run_epoch != session.run_epoch:
            raise RuntimeError("Terminal session changed while its evidence was repaired.")
        return IncompleteSessionRecoveryResult(
            session_id=session.id,
            previous_status=previous_status,
            status=current.status,
            actions=(IncompleteSessionRecoveryAction.REPAIRED_TERMINAL_EVIDENCE,),
            events=(() if terminal_event is None else (terminal_event,)),
            message=(
                "Reconciled durable terminal evidence."
                if terminal_event is None
                else "Repaired durable terminal event evidence."
            ),
        )

    def _terminal_evidence_repair_event(
        self,
        *,
        session: Session,
        terminal_run_epoch: int,
        terminal_timestamp: datetime,
        pending_interrupt_payload: dict[str, Any] | None,
        pending_action_interrupt_payload: dict[str, Any] | None,
        run_operation: _SessionRunOperation | None,
    ) -> Event:
        event_type = _TERMINAL_EVENT_TYPE_BY_STATUS[session.status]
        pending_interrupt_request_id = (
            None
            if pending_interrupt_payload is None
            else interruption_request_id_from_payload(pending_interrupt_payload)
        )
        if pending_interrupt_request_id is not None:
            operation_identity = f"interrupt_request:{pending_interrupt_request_id}"
        elif run_operation is not None:
            operation_identity = run_operation.operation_id
        else:
            operation_identity = f"run_epoch:{terminal_run_epoch}"
        event_id = (
            run_operation.terminal_event_id
            if run_operation is not None and run_operation.terminal_event_id is not None
            else str(
                uuid5(
                    _TERMINAL_EVIDENCE_REPAIR_NAMESPACE,
                    f"{session.id}\0{operation_identity}\0{session.status.value}",
                )
            )
        )
        if session.status == SessionStatus.COMPLETED:
            payload: dict[str, Any] = {
                "recovered": True,
                "terminal_evidence_repaired": True,
            }
        elif session.status == SessionStatus.FAILED:
            payload = {
                "error": "Original terminal failure details were not durably recorded.",
                "error_type": "TerminalFailureEvidenceUnavailable",
                "recovered": True,
                "terminal_evidence_repaired": True,
            }
        elif pending_interrupt_payload is not None:
            payload = copy_json_value(
                pending_interrupt_payload,
                "pending_session_interrupt",
            )
        elif pending_action_interrupt_payload is not None:
            payload = copy_json_value(
                pending_action_interrupt_payload,
                "pending_action_interrupt",
            )
        else:
            payload = {
                "interruption_type": _INTERRUPTION_TYPE_RUNTIME_INTERRUPTED,
                "reason": "terminal_event_evidence_repaired",
                "recovered": True,
                "terminal_evidence_repaired": True,
            }
        event = event_with_runtime_envelope_authority(
            event_with_runtime_generated_id(
                Event(
                    id=event_id,
                    type=event_type,
                    session_id=session.id,
                    timestamp=terminal_timestamp,
                    agent_name=session.agent_name,
                    environment_name=session.environment_name,
                    payload=payload,
                )
            ),
            "session_id",
        )
        if type(payload.get("interruption_request_id")) is str:
            event = event_with_runtime_payload_authority(
                event,
                "interruption_request_id",
            )
        raw_profile_fingerprint = payload.get("execution_profile_fingerprint")
        if raw_profile_fingerprint is None:
            profile_fingerprint = None
        elif type(raw_profile_fingerprint) is str:
            profile_fingerprint = raw_profile_fingerprint
        else:
            raise TypeError("execution_profile_fingerprint must be a string or None.")
        event = event_with_execution_profile_fingerprint_authority(event, profile_fingerprint)
        return (
            event
            if run_operation is None
            else _event_with_session_run_operation(event, run_operation)
        )

    async def _persist_terminal_evidence_repair_event(self, event: Event) -> Event:
        # Freeze the publication-safe shape before the append attempt. The
        # writer may redact workload secrets, so acknowledgement-loss
        # reconciliation must compare durable evidence with this prepared
        # snapshot rather than with the raw checkpoint-derived payload.
        event = self._event_writer.prepare(event)
        try:
            return await self._event_writer.persist(event)
        except Exception as append_failure:
            try:
                records = await self._session_store.query_events(
                    EventQuery(
                        session_id=event.session_id,
                        event_id=event.id,
                        limit=1,
                    )
                )
            except Exception as reconciliation_failure:
                add_exception_note_safely(
                    append_failure,
                    "Terminal evidence append reconciliation failed: "
                    f"{type(reconciliation_failure).__name__}.",
                )
                raise ExceptionGroup(
                    "Terminal evidence append and reconciliation both failed.",
                    [append_failure, reconciliation_failure],
                ) from None
            try:
                persisted = _reconcile_exact_persisted_event(
                    event,
                    records,
                    conflict_message=(
                        "Terminal evidence repair event identity is already used by "
                        "different durable evidence."
                    ),
                )
            except RuntimeError as conflict:
                raise conflict from append_failure
            if persisted is None:
                raise
            return persisted

    async def _clear_repaired_pending_interrupt(
        self,
        *,
        session_id: str,
        claim_id: str,
        expected_payload: dict[str, Any],
    ) -> None:
        def clear_marker(
            current_session: Session,
            checkpoint: dict[str, Any] | None,
        ) -> dict[str, Any]:
            if current_session.status != SessionStatus.INTERRUPTED:
                raise RuntimeError("Session status changed during terminal interruption repair.")
            if checkpoint is None:
                raise _IncompleteRecoveryClaimLost(
                    "Terminal evidence recovery checkpoint disappeared."
                )
            updated = copy_json_value(checkpoint, "checkpoint")
            claim = _incomplete_recovery_claim_from_checkpoint(updated)
            if claim is None or claim[0] != claim_id:
                raise _IncompleteRecoveryClaimLost(
                    "Terminal evidence recovery ownership changed before marker cleanup."
                )
            current_payload = updated.get(_PENDING_SESSION_INTERRUPT_CHECKPOINT_KEY)
            if current_payload != expected_payload:
                raise RuntimeError(
                    "Pending interruption identity changed during terminal evidence repair."
                )
            updated.pop(_PENDING_SESSION_INTERRUPT_CHECKPOINT_KEY)
            return updated

        await self._session_store.transform_checkpoint(session_id, clear_marker)

    async def _clear_session_run_operation(
        self,
        *,
        session_id: str,
        operation: _SessionRunOperation,
        required_claim_id: str | None = None,
        terminal_evidence_durable: bool = False,
    ) -> None:
        def clear_operation(
            _session: Session,
            checkpoint: dict[str, Any] | None,
        ) -> dict[str, Any] | None:
            current_operation = _session_run_operation_from_checkpoint(checkpoint)
            if current_operation is None:
                return checkpoint
            if current_operation != operation:
                raise RuntimeError(
                    "Session run operation changed before terminal evidence cleanup."
                )
            if required_claim_id is not None:
                claim = _incomplete_recovery_claim_from_checkpoint(checkpoint)
                if claim is None or claim[0] != required_claim_id:
                    raise _IncompleteRecoveryClaimLost(
                        "Terminal evidence recovery ownership changed before run cleanup."
                    )
            return _checkpoint_after_session_run_operation_cleanup(
                checkpoint,
                operation=operation,
                retain_terminal_receipt=terminal_evidence_durable,
            )

        await self._session_store.transform_checkpoint(session_id, clear_operation)

    async def _cleanup_incomplete_recovery_claim(
        self,
        *,
        session_id: str,
        claim_id: str,
        authoritative_failure: BaseException | None,
        release_environment_cleanup: bool = False,
        execution_profile: ExecutionProfileIdentity | None = None,
    ) -> None:
        recovery_run_epoch = _current_session_run_epoch(session_id)

        async def release_recovery_run_fence() -> None:
            if release_environment_cleanup:
                await self._environment_lifecycle.release_run_fence_after_environment_cleanup(
                    session_id=session_id,
                    execution_profile=execution_profile,
                )
            else:
                await self._session_store.release_run_fence(session_id)
            if recovery_run_epoch is not None:
                self._environment_lifecycle.retire_repaired_run_fence_releases(
                    session_id=session_id,
                    repaired_run_epoch=recovery_run_epoch,
                )

        try:
            await _run_recovery_cleanup_steps(
                authoritative_failure=authoritative_failure,
                steps=(
                    (
                        "run fence release",
                        release_recovery_run_fence,
                    ),
                    (
                        "incomplete recovery claim release",
                        lambda: self._release_incomplete_recovery_claim(
                            session_id,
                            claim_id,
                        ),
                    ),
                ),
            )
        finally:
            _deactivate_session_run_fence(session_id)

    async def _load_owned_incomplete_recovery_claim(
        self,
        session_id: str,
        claim_id: str,
        *,
        expected_run_epoch: int | None,
    ) -> Session | None:
        """Return the session only while the exact claim and its epoch are owned."""
        if expected_run_epoch is None:
            return None
        checkpoint = await self._session_store.load_checkpoint(session_id)
        persisted_claim = _incomplete_recovery_claim_from_checkpoint(checkpoint)
        if persisted_claim is None or persisted_claim[0] != claim_id:
            return None
        session = await self._require_session(session_id)
        if session.run_epoch != expected_run_epoch:
            return None
        return session

    async def _claim_incomplete_recovery(
        self,
        *,
        session: Session,
        inactive_before: datetime | None,
        required_expired_claim_id: str | None = None,
        execution_profile_snapshot: ActiveInvocationExecutionProfile | None = None,
    ) -> _IncompleteRecoveryClaim | None:
        if required_expired_claim_id is not None:
            required_expired_claim_id = require_clean_nonblank(
                required_expired_claim_id,
                "required_expired_claim_id",
            )
        claim_id = str(uuid4())
        claim_expires_at: datetime | None = None
        claim_run_epoch: int | None = None

        if required_expired_claim_id is not None:
            session_before_fence: Session | None = None

            def replace_expired_claim(
                current_session: Session,
                checkpoint: dict[str, Any] | None,
            ) -> dict[str, Any]:
                nonlocal claim_expires_at, claim_run_epoch, session_before_fence
                claimed_at = self._clock()
                _require_aware_datetime(claimed_at, "recovery claim clock")
                if (
                    active_provider_operation_cancellation_claim_from_checkpoint(
                        checkpoint,
                        now=datetime.now(UTC),
                    )
                    is not None
                ):
                    raise _IncompleteRecoveryClaimLost(
                        "Provider-operation cancellation still owns the session epoch."
                    )
                existing = _incomplete_recovery_claim_from_checkpoint(checkpoint)
                if (
                    existing is None
                    or existing[0] != required_expired_claim_id
                    or existing[1] > claimed_at
                ):
                    raise _IncompleteRecoveryClaimLost(
                        "Expired incomplete-session recovery ownership changed."
                    )
                claim_expires_at = claimed_at + _INCOMPLETE_RECOVERY_CLAIM_LEASE
                next_run_epoch = current_session.run_epoch + 1
                claim_run_epoch = next_run_epoch
                session_before_fence = current_session.model_copy(deep=True)
                updated = {} if checkpoint is None else copy_json_value(checkpoint, "checkpoint")
                current_profile = active_invocation_execution_profile_from_checkpoint(updated)
                if execution_profile_snapshot is not None and (
                    current_profile != execution_profile_snapshot
                ):
                    raise _IncompleteRecoveryClaimLost(
                        "Active invocation profile changed before recovery takeover."
                    )
                if current_profile is not None:
                    updated = checkpoint_with_active_invocation_execution_profile(
                        updated,
                        session_id=current_session.id,
                        interaction_id=current_profile.interaction_id,
                        run_epoch=next_run_epoch,
                        profile=current_profile.profile,
                        expected=current_profile,
                    )
                updated[_INCOMPLETE_RECOVERY_CLAIM_CHECKPOINT_KEY] = {
                    "version": 1,
                    "claim_id": claim_id,
                    "claimed_at": claimed_at.isoformat(),
                    "claim_expires_at": claim_expires_at.isoformat(),
                }
                return _checkpoint_with_rebased_session_run_operation(
                    updated,
                    previous_run_epoch=current_session.run_epoch,
                    run_epoch=next_run_epoch,
                )

            fence_task = asyncio.create_task(
                self._session_store.fence_run_and_transform_checkpoint(
                    session.id,
                    statuses={session.status},
                    checkpoint_transform=replace_expired_claim,
                )
            )
            outcome = await await_shielded_task_outcome(fence_task)
            if isinstance(
                outcome.error,
                _IncompleteRecoveryClaimLost | SessionStatusConflict,
            ):
                if outcome.cancellation is not None:
                    outcome.cancellation.add_note(
                        "Expired recovery takeover was rejected while cancellation was pending."
                    )
                    raise outcome.cancellation from outcome.error
                return None

            authoritative_failure = outcome.cancellation or outcome.error
            fenced = outcome.result
            if outcome.error is not None:
                reconciliation_outcome = await await_shielded_task_outcome(
                    asyncio.create_task(
                        self._load_owned_incomplete_recovery_claim(
                            session.id,
                            claim_id,
                            expected_run_epoch=claim_run_epoch,
                        )
                    ),
                    cancellation=outcome.cancellation,
                )
                reconciliation_cancellation = reconciliation_outcome.cancellation
                reconciliation_failure = reconciliation_outcome.error
                if reconciliation_cancellation is None and isinstance(
                    reconciliation_failure,
                    asyncio.CancelledError,
                ):
                    reconciliation_cancellation = reconciliation_failure
                authoritative_failure = (
                    reconciliation_cancellation or outcome.cancellation or outcome.error
                )
                if reconciliation_failure is not None:
                    if not isinstance(
                        reconciliation_failure,
                        Exception | asyncio.CancelledError,
                    ):
                        raise reconciliation_failure from outcome.error
                    authoritative_failure.add_note(
                        "Could not reconcile whether the expired recovery takeover "
                        f"committed: {type(reconciliation_failure).__name__}."
                    )
                    if reconciliation_cancellation is not None:
                        reconciliation_cancellation.add_note(
                            "Expired recovery takeover also failed: "
                            f"{type(outcome.error).__name__}."
                        )
                        raise reconciliation_cancellation from outcome.error
                    raise outcome.error
                fenced = reconciliation_outcome.result
                if fenced is None:
                    if reconciliation_cancellation is not None:
                        reconciliation_cancellation.add_note(
                            "Expired recovery takeover also failed: "
                            f"{type(outcome.error).__name__}."
                        )
                        raise reconciliation_cancellation from outcome.error
                    raise outcome.error

            if fenced is None:
                raise RuntimeError("Expired recovery takeover returned no session.")
            _activate_session_run_fence(fenced)
            if (
                claim_expires_at is None
                or claim_run_epoch is None
                or session_before_fence is None
                or fenced.run_epoch != claim_run_epoch
            ):
                invariant_failure = RuntimeError(
                    "Expired recovery takeover did not persist its claim."
                )
                await self._cleanup_incomplete_recovery_claim(
                    session_id=session.id,
                    claim_id=claim_id,
                    authoritative_failure=invariant_failure,
                )
                raise invariant_failure
            claim = _IncompleteRecoveryClaim(
                claim_id=claim_id,
                claim_expires_at=claim_expires_at,
                session_before_fence=session_before_fence,
                session=fenced,
            )
            if authoritative_failure is None:
                return claim
            await self._cleanup_incomplete_recovery_claim(
                session_id=session.id,
                claim_id=claim_id,
                authoritative_failure=authoritative_failure,
            )
            if outcome.cancellation is not None:
                if outcome.error is not None:
                    outcome.cancellation.add_note(
                        f"Expired recovery takeover also failed: {type(outcome.error).__name__}."
                    )
                raise outcome.cancellation from outcome.error
            raise outcome.error

        session_before_fence: Session | None = None

        def claim_checkpoint(
            current_session: Session,
            checkpoint: dict[str, Any] | None,
        ) -> dict[str, Any]:
            nonlocal claim_expires_at, claim_run_epoch, session_before_fence
            claimed_at = self._clock()
            _require_aware_datetime(claimed_at, "recovery claim clock")
            if (
                active_provider_operation_cancellation_claim_from_checkpoint(
                    checkpoint,
                    now=datetime.now(UTC),
                )
                is not None
            ):
                raise _IncompleteRecoveryClaimLost(
                    "Provider-operation cancellation still owns the session epoch."
                )
            existing = _incomplete_recovery_claim_from_checkpoint(checkpoint)
            if (
                current_session.status != session.status
                or (
                    inactive_before is not None
                    and current_session.last_activity_at > inactive_before
                )
                or (existing is not None and existing[1] > claimed_at)
            ):
                raise _IncompleteRecoveryClaimLost(
                    "Incomplete-session recovery ownership changed before it was claimed."
                )
            claim_expires_at = claimed_at + _INCOMPLETE_RECOVERY_CLAIM_LEASE
            next_run_epoch = current_session.run_epoch + 1
            claim_run_epoch = next_run_epoch
            session_before_fence = current_session.model_copy(deep=True)
            updated = {} if checkpoint is None else copy_json_value(checkpoint, "checkpoint")
            current_profile = active_invocation_execution_profile_from_checkpoint(updated)
            if execution_profile_snapshot is not None and (
                current_profile != execution_profile_snapshot
            ):
                raise _IncompleteRecoveryClaimLost(
                    "Active invocation profile changed before recovery was claimed."
                )
            if current_profile is not None:
                updated = checkpoint_with_active_invocation_execution_profile(
                    updated,
                    session_id=current_session.id,
                    interaction_id=current_profile.interaction_id,
                    run_epoch=next_run_epoch,
                    profile=current_profile.profile,
                    expected=current_profile,
                )
            updated[_INCOMPLETE_RECOVERY_CLAIM_CHECKPOINT_KEY] = {
                "version": 1,
                "claim_id": claim_id,
                "claimed_at": claimed_at.isoformat(),
                "claim_expires_at": claim_expires_at.isoformat(),
            }
            return _checkpoint_with_rebased_session_run_operation(
                updated,
                previous_run_epoch=current_session.run_epoch,
                run_epoch=next_run_epoch,
            )

        fence_claimed = False
        try:
            fence_task = asyncio.create_task(
                self._session_store.fence_run_and_transform_checkpoint(
                    session.id,
                    statuses={session.status},
                    checkpoint_transform=claim_checkpoint,
                )
            )
            outcome = await await_shielded_task_outcome(fence_task)
            if isinstance(
                outcome.error,
                _IncompleteRecoveryClaimLost | SessionStatusConflict,
            ):
                if outcome.cancellation is not None:
                    outcome.cancellation.add_note(
                        "Incomplete-session recovery claim was rejected while cancellation "
                        "was pending."
                    )
                    raise outcome.cancellation from outcome.error
                return None

            authoritative_failure = outcome.cancellation or outcome.error
            fenced = outcome.result
            if outcome.error is not None:
                reconciliation_outcome = await await_shielded_task_outcome(
                    asyncio.create_task(
                        self._load_owned_incomplete_recovery_claim(
                            session.id,
                            claim_id,
                            expected_run_epoch=claim_run_epoch,
                        )
                    ),
                    cancellation=outcome.cancellation,
                )
                reconciliation_cancellation = reconciliation_outcome.cancellation
                reconciliation_failure = reconciliation_outcome.error
                if reconciliation_cancellation is None and isinstance(
                    reconciliation_failure,
                    asyncio.CancelledError,
                ):
                    reconciliation_cancellation = reconciliation_failure
                authoritative_failure = (
                    reconciliation_cancellation or outcome.cancellation or outcome.error
                )
                if reconciliation_failure is not None:
                    if not isinstance(
                        reconciliation_failure,
                        Exception | asyncio.CancelledError,
                    ):
                        raise reconciliation_failure from outcome.error
                    authoritative_failure.add_note(
                        "Could not reconcile whether the incomplete-session recovery claim "
                        f"committed: {type(reconciliation_failure).__name__}."
                    )
                    if reconciliation_cancellation is not None:
                        reconciliation_cancellation.add_note(
                            "Incomplete-session recovery claim also failed: "
                            f"{type(outcome.error).__name__}."
                        )
                        raise reconciliation_cancellation from outcome.error
                    raise outcome.error
                fenced = reconciliation_outcome.result
                if fenced is None:
                    if reconciliation_cancellation is not None:
                        reconciliation_cancellation.add_note(
                            "Incomplete-session recovery claim also failed: "
                            f"{type(outcome.error).__name__}."
                        )
                        raise reconciliation_cancellation from outcome.error
                    raise outcome.error

            if fenced is None:
                raise RuntimeError("Incomplete-session recovery claim returned no session.")
            _activate_session_run_fence(fenced)
            fence_claimed = True
            if (
                claim_expires_at is None
                or claim_run_epoch is None
                or session_before_fence is None
                or fenced.run_epoch != claim_run_epoch
            ):
                raise RuntimeError(
                    "Incomplete-session recovery claim was not persisted atomically."
                )
            if authoritative_failure is not None:
                if outcome.cancellation is not None:
                    if outcome.error is not None:
                        outcome.cancellation.add_note(
                            "Incomplete-session recovery claim also failed: "
                            f"{type(outcome.error).__name__}."
                        )
                    raise outcome.cancellation from outcome.error
                raise outcome.error

            try:
                renewed_until = await self._renew_incomplete_recovery_claim(
                    session.id,
                    claim_id,
                )
            except SessionRunFenced:
                await self._cleanup_incomplete_recovery_claim(
                    session_id=session.id,
                    claim_id=claim_id,
                    authoritative_failure=None,
                )
                fence_claimed = False
                return None
            if renewed_until is None:
                await self._cleanup_incomplete_recovery_claim(
                    session_id=session.id,
                    claim_id=claim_id,
                    authoritative_failure=None,
                )
                fence_claimed = False
                return None
            return _IncompleteRecoveryClaim(
                claim_id=claim_id,
                claim_expires_at=renewed_until,
                session_before_fence=session_before_fence,
                session=fenced,
            )
        except BaseException as exc:
            cleanup_steps: list[tuple[str, RecoveryCleanup]] = []
            if fence_claimed:
                cleanup_steps.append(
                    (
                        "run fence release",
                        lambda: self._session_store.release_run_fence(session.id),
                    )
                )
            cleanup_steps.append(
                (
                    "incomplete recovery claim release",
                    lambda: self._release_incomplete_recovery_claim(session.id, claim_id),
                )
            )
            try:
                await _run_recovery_cleanup_steps(
                    authoritative_failure=exc,
                    steps=tuple(cleanup_steps),
                )
            finally:
                if fence_claimed:
                    _deactivate_session_run_fence(session.id)
            raise

    async def _recover_incomplete_session_with_heartbeat(
        self,
        *,
        claim: _IncompleteRecoveryClaim,
        recovery: Callable[[], Awaitable[IncompleteSessionRecoveryResult]],
    ) -> IncompleteSessionRecoveryResult:
        stop_heartbeat = asyncio.Event()
        recovery_outcome_observed = False

        async def run_recovery() -> CapturedAwaitableOutcome[IncompleteSessionRecoveryResult]:
            return await capture_awaitable_outcome(recovery)

        def recovery_outcome() -> IncompleteSessionRecoveryResult:
            nonlocal recovery_outcome_observed
            captured = recovery_task.result()
            recovery_outcome_observed = True
            if captured.error is not None:
                raise captured.error
            if captured.result is None:
                raise RuntimeError("Incomplete-session recovery returned no result.")
            return captured.result

        recovery_task = asyncio.create_task(run_recovery())
        heartbeat_task = asyncio.create_task(
            self._heartbeat_incomplete_recovery_claim(
                session_id=claim.session.id,
                claim_id=claim.claim_id,
                claim_expires_at=claim.claim_expires_at,
                stop=stop_heartbeat,
            )
        )
        authoritative_failure: BaseException | None = None

        async def stop_workers() -> None:
            stop_heartbeat.set()
            for task in (recovery_task, heartbeat_task):
                if not task.done():
                    task.cancel()
            await asyncio.gather(recovery_task, heartbeat_task, return_exceptions=True)
            if recovery_outcome_observed or recovery_task.cancelled():
                return
            captured = recovery_task.result()
            if captured.error is None:
                return
            if isinstance(captured.error, asyncio.CancelledError):
                secondary = exception_cause(captured.error)
                if secondary is not None:
                    raise secondary
                return
            raise captured.error

        try:
            done, _pending = await asyncio.wait(
                {recovery_task, heartbeat_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if heartbeat_task in done:
                heartbeat_failure = heartbeat_task.exception()
                if heartbeat_failure is None:
                    raise RuntimeError(
                        "Incomplete-session recovery claim heartbeat stopped unexpectedly."
                    )
                raise heartbeat_failure
            result = recovery_outcome()
            stop_heartbeat.set()
            await heartbeat_task
            return result
        except BaseException as exc:
            authoritative_failure = exc
            raise
        finally:
            await _run_recovery_cleanup_steps(
                authoritative_failure=authoritative_failure,
                steps=(("incomplete recovery worker shutdown", stop_workers),),
            )

    async def _heartbeat_incomplete_recovery_claim(
        self,
        *,
        session_id: str,
        claim_id: str,
        claim_expires_at: datetime,
        stop: asyncio.Event,
    ) -> None:
        sleep_seconds = _INCOMPLETE_RECOVERY_CLAIM_HEARTBEAT_INTERVAL_SECONDS
        while not stop.is_set():
            try:
                await asyncio.wait_for(
                    stop.wait(),
                    timeout=sleep_seconds,
                )
            except TimeoutError:
                try:
                    renewed_until = await self._renew_incomplete_recovery_claim(
                        session_id,
                        claim_id,
                    )
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    now = self._clock()
                    _require_aware_datetime(now, "recovery claim clock")
                    if now >= claim_expires_at:
                        raise _IncompleteRecoveryClaimLost(
                            "Incomplete-session recovery claim could not be renewed before "
                            f"expiry for session {session_id}."
                        ) from exc
                    sleep_seconds = min(
                        _INCOMPLETE_RECOVERY_CLAIM_HEARTBEAT_RETRY_SECONDS,
                        max(0.0, (claim_expires_at - now).total_seconds()),
                    )
                    continue
                if renewed_until is None:
                    raise _IncompleteRecoveryClaimLost(
                        f"Incomplete-session recovery claim lost for session {session_id}."
                    ) from None
                claim_expires_at = renewed_until
                sleep_seconds = _INCOMPLETE_RECOVERY_CLAIM_HEARTBEAT_INTERVAL_SECONDS

    async def _watch_manual_recovery_interruption(
        self,
        *,
        session_id: str,
        interrupted_baseline_id: str | None,
        stop: asyncio.Event,
    ) -> bool:
        """Observe another worker's durable stop request while delivery is paused."""
        while not stop.is_set():
            session = await self._require_session(session_id)
            if session.status == SessionStatus.INTERRUPTING:
                return True
            if session.status == SessionStatus.INTERRUPTED:
                latest_interrupted = await self._session_control.latest_interrupted_event(
                    session_id
                )
                if (
                    latest_interrupted is not None
                    and latest_interrupted.id != interrupted_baseline_id
                    and latest_interrupted.payload.get("interruption_type")
                    == _INTERRUPTION_TYPE_OPERATOR_REQUESTED
                ):
                    return True
            try:
                await asyncio.wait_for(
                    stop.wait(),
                    timeout=_MANUAL_RECOVERY_INTERRUPT_POLL_INTERVAL_SECONDS,
                )
            except TimeoutError:
                continue
        return False

    async def _renew_incomplete_recovery_claim(
        self,
        session_id: str,
        claim_id: str,
    ) -> datetime | None:
        renewed_until: datetime | None = None

        def renew_claim(
            _session: Session,
            checkpoint: dict[str, Any] | None,
        ) -> dict[str, Any] | None:
            nonlocal renewed_until
            existing = _incomplete_recovery_claim_from_checkpoint(checkpoint)
            now = self._clock()
            _require_aware_datetime(now, "recovery claim clock")
            if existing is None or existing[0] != claim_id or existing[1] <= now:
                return None
            updated = copy_json_value(checkpoint, "checkpoint")
            marker = copy_json_value(
                updated[_INCOMPLETE_RECOVERY_CLAIM_CHECKPOINT_KEY],
                "incomplete_session_recovery_claim",
            )
            renewed_until = now + _INCOMPLETE_RECOVERY_CLAIM_LEASE
            marker["claim_expires_at"] = renewed_until.isoformat()
            marker["renewed_at"] = now.isoformat()
            updated[_INCOMPLETE_RECOVERY_CLAIM_CHECKPOINT_KEY] = marker
            return updated

        await self._session_store.transform_checkpoint(session_id, renew_claim)
        return renewed_until

    async def _release_incomplete_recovery_claim(
        self,
        session_id: str,
        claim_id: str,
    ) -> None:
        def release_claim(
            _session: Session,
            checkpoint: dict[str, Any] | None,
        ) -> dict[str, Any] | None:
            if checkpoint is None:
                return None
            existing = _incomplete_recovery_claim_from_checkpoint(checkpoint)
            if existing is None or existing[0] != claim_id:
                return None
            updated = copy_json_value(checkpoint, "checkpoint")
            updated.pop(_INCOMPLETE_RECOVERY_CLAIM_CHECKPOINT_KEY, None)
            return updated

        await self._session_store.transform_checkpoint(session_id, release_claim)

    async def _recover_workspace_observations(
        self,
        *,
        session: Session,
        registered_environment: runtime_records.RegisteredEnvironment | None,
        execution_profile_snapshot: ActiveInvocationExecutionProfile | None,
    ) -> tuple[Event, ...]:
        """Close crash-interrupted observation state without redispatching effects."""

        recovered_events: list[Event] = []
        checkpoint = await await_workspace_observation_store_read(
            lambda: self._session_store.load_checkpoint(session.id),
            operation="Workspace observation recovery checkpoint read",
        )
        observations = workspace_observations_from_checkpoint(checkpoint)
        pending_round = tool_round_recovery.pending_tool_round_from_checkpoint(
            checkpoint,
            redactor=self._secret_redactor,
            consume_on_rejection=True,
        )
        # Revalidate the complete aggregate after the recovery claim. Otherwise
        # a valid record ordered before a foreign record could be terminalized
        # (and could trigger artifact-store reads) before the malformed
        # aggregate is rejected.
        environment_authority_available = self._validate_workspace_observation_recovery_authority(
            session=session,
            observations=observations,
            pending_round=pending_round,
            execution_profile_snapshot=execution_profile_snapshot,
            registered_environment=registered_environment,
        )
        for window_id in sorted(observations):
            durable_lifecycle = observations[window_id]
            reconstructed_stage = self._reconstruct_workspace_observation_staged_outcome(
                session=session,
                checkpoint=checkpoint,
                lifecycle=durable_lifecycle,
            )
            if reconstructed_stage is not None:
                await publish_workspace_observation_transition(
                    session_store=self._session_store,
                    event_writer=self._event_writer,
                    session=session,
                    previous=durable_lifecycle,
                    current=reconstructed_stage,
                    phase="recovered-tool-outcome",
                )
                durable_lifecycle = reconstructed_stage
                checkpoint = await await_workspace_observation_store_read(
                    lambda: self._session_store.load_checkpoint(session.id),
                    operation="Workspace observation recovery checkpoint read",
                )
            tool_outcome_evidence_valid = True
            if durable_lifecycle.tool_outcome_event_id is not None:
                tool_outcome_evidence_valid = (
                    await self._workspace_observation_tool_outcome_evidence_valid(
                        session=session,
                        checkpoint=checkpoint,
                        lifecycle=durable_lifecycle,
                    )
                )

            delta_evidence_valid = False
            delta_evidence_conflict = False
            terminal_status: WorkspaceObservationTerminalStatus | None = None
            terminal_detail: str | None = None
            if durable_lifecycle.phase is WorkspaceObservationPhase.DELTA_PUBLISHED:
                (
                    delta_evidence_valid,
                    delta_evidence_conflict,
                    terminal_status,
                    terminal_detail,
                ) = await self._workspace_observation_delta_evidence(
                    session=session,
                    lifecycle=durable_lifecycle,
                )

            if (
                environment_authority_available
                and tool_outcome_evidence_valid
                and not delta_evidence_conflict
            ):
                lifecycle = await self._reconcile_workspace_observation_artifacts(
                    durable_lifecycle,
                    registered_environment=registered_environment,
                )
            else:
                # An extension-owned ArtifactStore must not be entered until the
                # content-bound tool and delta evidence prove that this lifecycle
                # owns the requested artifacts. Retain their exact identities but
                # fail verification closed for the terminal diagnostic.
                lifecycle = self._workspace_observation_unverified_artifacts(durable_lifecycle)
            if lifecycle.phase is WorkspaceObservationPhase.DELTA_PUBLISHED:
                if terminal_status is None:
                    raise AssertionError("Published workspace delta lost its classification.")
                if not environment_authority_available and terminal_status not in {
                    WorkspaceObservationTerminalStatus.AMBIGUOUS,
                    WorkspaceObservationTerminalStatus.FAILED,
                }:
                    terminal_status = WorkspaceObservationTerminalStatus.INCOMPLETE
                    terminal_detail = "workspace_revision_evidence_incomplete"
                if terminal_status not in {
                    WorkspaceObservationTerminalStatus.AMBIGUOUS,
                    WorkspaceObservationTerminalStatus.FAILED,
                }:
                    if any(
                        artifact.state is WorkspaceObservationArtifactState.MISSING
                        for artifact in lifecycle.artifacts
                    ):
                        terminal_status = WorkspaceObservationTerminalStatus.INCOMPLETE
                        terminal_detail = "referenced_workspace_artifact_missing"
                    elif any(
                        artifact.state is WorkspaceObservationArtifactState.FAILED
                        for artifact in lifecycle.artifacts
                    ):
                        terminal_status = WorkspaceObservationTerminalStatus.INCOMPLETE
                        terminal_detail = "workspace_artifact_verification_failed"
                    elif (
                        any(
                            artifact.state
                            in {
                                WorkspaceObservationArtifactState.INTENT,
                                WorkspaceObservationArtifactState.ORPHANED,
                            }
                            for artifact in lifecycle.artifacts
                        )
                        or lifecycle.before_state is not WorkspaceObservationEvidenceState.PUBLISHED
                        or lifecycle.after_state is not WorkspaceObservationEvidenceState.PUBLISHED
                        or lifecycle.delta_state is not WorkspaceObservationEvidenceState.PUBLISHED
                    ):
                        terminal_status = WorkspaceObservationTerminalStatus.INCOMPLETE
                        terminal_detail = "workspace_revision_evidence_incomplete"
                tool_outcome_available = await self._repair_workspace_observation_terminal_stage(
                    session=session,
                    durable_lifecycle=durable_lifecycle,
                    lifecycle=lifecycle,
                    capture_status=("recorded" if delta_evidence_valid else "failed"),
                    capture_detail_code=(None if delta_evidence_valid else terminal_detail),
                )
                if not tool_outcome_available:
                    terminal_status = WorkspaceObservationTerminalStatus.AMBIGUOUS
                    terminal_detail = "durable_tool_outcome_evidence_missing"
                final_event = prepare_runtime_event(
                    _workspace_mutation_incomplete_event(
                        lifecycle=lifecycle,
                        session=session,
                        execution_profile=(
                            None
                            if execution_profile_snapshot is None
                            else execution_profile_snapshot.profile
                        ),
                        status=terminal_status,
                        detail_code=terminal_detail,
                    ),
                    redactor=self._secret_redactor,
                )
                published = await publish_workspace_observation_transition(
                    session_store=self._session_store,
                    event_writer=self._event_writer,
                    session=session,
                    previous=durable_lifecycle,
                    current=None,
                    phase="terminal",
                    terminal_status=terminal_status,
                    terminal_detail_code=terminal_detail,
                    terminal_artifacts=lifecycle.artifacts,
                    events=(final_event,),
                )
                recovered_events.extend(published)
                checkpoint = await await_workspace_observation_store_read(
                    lambda: self._session_store.load_checkpoint(session.id),
                    operation="Workspace observation recovery checkpoint read",
                )
                observations = workspace_observations_from_checkpoint(checkpoint)
                continue

            terminal_status = (
                WorkspaceObservationTerminalStatus.INCOMPLETE
                if lifecycle.phase is WorkspaceObservationPhase.TOOL_OUTCOME_STAGED
                or lifecycle.phase is WorkspaceObservationPhase.AFTER_CAPTURED
                else WorkspaceObservationTerminalStatus.AMBIGUOUS
            )
            detail_code = (
                "worker_lost_before_workspace_observation_completed"
                if terminal_status is WorkspaceObservationTerminalStatus.INCOMPLETE
                else "worker_lost_before_tool_outcome_was_durable"
            )

            if lifecycle.tool_outcome_event_id is not None and not (
                await self._repair_workspace_observation_terminal_stage(
                    session=session,
                    durable_lifecycle=durable_lifecycle,
                    lifecycle=lifecycle,
                    capture_status="failed",
                    capture_detail_code=detail_code,
                )
            ):
                terminal_status = WorkspaceObservationTerminalStatus.AMBIGUOUS
                detail_code = "durable_tool_outcome_evidence_missing"

            terminal_event = prepare_runtime_event(
                _workspace_mutation_incomplete_event(
                    lifecycle=lifecycle,
                    session=session,
                    execution_profile=(
                        None
                        if execution_profile_snapshot is None
                        else execution_profile_snapshot.profile
                    ),
                    status=terminal_status,
                    detail_code=detail_code,
                ),
                redactor=self._secret_redactor,
            )
            published = await publish_workspace_observation_transition(
                session_store=self._session_store,
                event_writer=self._event_writer,
                session=session,
                previous=durable_lifecycle,
                current=None,
                phase="terminal",
                terminal_status=terminal_status,
                terminal_detail_code=detail_code,
                terminal_artifacts=lifecycle.artifacts,
                events=(terminal_event,),
            )
            recovered_events.extend(published)
            checkpoint = await await_workspace_observation_store_read(
                lambda: self._session_store.load_checkpoint(session.id),
                operation="Workspace observation recovery checkpoint read",
            )
            observations = workspace_observations_from_checkpoint(checkpoint)
        return tuple(recovered_events)

    def _validate_workspace_observation_recovery_authority(
        self,
        *,
        session: Session,
        observations: dict[str, WorkspaceObservationLifecycle],
        pending_round: tool_round_recovery.PendingToolRound | None,
        execution_profile_snapshot: ActiveInvocationExecutionProfile | None,
        registered_environment: runtime_records.RegisteredEnvironment | None = None,
    ) -> bool:
        """Reject lifecycle authority conflicts before recovery side effects."""

        if not observations:
            return True
        if pending_round is None:
            raise RuntimeError("Workspace observation has no authoritative pending tool round.")
        if execution_profile_snapshot is None:
            raise RuntimeError(
                "Workspace observation has no authoritative active invocation profile."
            )
        pending_identity = tool_round_recovery.pending_tool_round_identity(pending_round)
        pending_calls = {call.tool_call_id: call for call in pending_round.tool_calls}
        current_workspace_id: str | None = None
        current_observer = "UnconfiguredEnvironment"
        current_observer_is_runtime_owned = True
        current_artifact_store_id: str | None = None
        factory_template_unavailable = (
            registered_environment is not None
            and registered_environment.factory_backed
            and registered_environment.factory is not None
        )
        if registered_environment is not None and not factory_template_unavailable:
            try:
                workspace = registered_environment.environment.workspace
                workspace_id = None if workspace is None else getattr(workspace, "id", None)
                current_workspace_id = (
                    None
                    if workspace_id is None
                    else require_clean_nonblank(workspace_id, "workspace.id")
                )
                binding = registered_environment.environment.binding
                current_observer = (
                    "UnconfiguredWorkspaceBinding" if binding is None else type(binding).__name__
                )
                current_observer_is_runtime_owned = (
                    binding is None or _runtime_owned_workspace_observer_name(binding) is not None
                )
                artifact_store = registered_environment.environment.artifact_store
                artifact_store_id = (
                    None if artifact_store is None else getattr(artifact_store, "id", None)
                )
                current_artifact_store_id = (
                    None
                    if artifact_store_id is None
                    else require_clean_nonblank(artifact_store_id, "artifact_store.id")
                )
            except Exception:
                raise RuntimeError(
                    "Workspace observation current environment authority is invalid."
                ) from None
        claimed_tool_calls: set[tuple[str, str]] = set()
        for lifecycle in observations.values():
            if lifecycle.session_id != session.id:
                raise RuntimeError("Workspace observation belongs to a different session.")
            if (
                lifecycle.agent_name != session.agent_name
                or lifecycle.agent_name != pending_round.agent_name
                or lifecycle.environment_name != session.environment_name
                or lifecycle.environment_name != pending_round.environment_name
            ):
                raise RuntimeError("Workspace observation conflicts with its invocation scope.")
            if lifecycle.source_run_epoch > session.run_epoch:
                raise RuntimeError("Workspace observation belongs to a future run epoch.")
            # ``binding_generation_id`` identifies the historical concrete
            # in-process binding that owned the observation window. A fresh
            # worker necessarily registers a new generation, so equality with
            # the current process would make restart repair impossible. The
            # frozen lifecycle/pending-round tuple authenticates the historical
            # owner; only stable workspace, observer, and artifact-store
            # authority can be rebound positively across processes.
            if (
                registered_environment is not None
                and not factory_template_unavailable
                and (
                    not workspace_observation_authority_matches(
                        lifecycle.workspace_id,
                        current_workspace_id or "workspace-unavailable",
                        field_name="workspace_id",
                        session_id=lifecycle.session_id,
                        public_authority_alias_codec=(
                            self._session_store.public_authority_alias_codec
                        ),
                    )
                    or not workspace_observation_observer_authority_matches(
                        lifecycle.observer,
                        lifecycle.observer_authority,
                        current_observer,
                        configured_observer_is_runtime_owned=current_observer_is_runtime_owned,
                        session_id=lifecycle.session_id,
                        public_authority_alias_codec=(
                            self._session_store.public_authority_alias_codec
                        ),
                    )
                    or not workspace_observation_authority_matches(
                        lifecycle.artifact_store_id,
                        current_artifact_store_id,
                        field_name="artifact_store_id",
                        session_id=lifecycle.session_id,
                        public_authority_alias_codec=(
                            self._session_store.public_authority_alias_codec
                        ),
                    )
                )
            ):
                raise RuntimeError(
                    "Workspace observation conflicts with its registered environment authority."
                )
            if (
                lifecycle.interaction_id != execution_profile_snapshot.interaction_id
                or lifecycle.source_run_epoch != execution_profile_snapshot.run_epoch
            ):
                raise RuntimeError(
                    "Workspace observation conflicts with its active invocation profile."
                )
            if (
                lifecycle.model_step_id != pending_identity.model_step_id
                or lifecycle.model_attempt_id != pending_identity.model_attempt_id
                or lifecycle.tool_round_id != pending_identity.tool_round_id
                or lifecycle.model_step != pending_round.model_step
            ):
                raise RuntimeError("Workspace observation conflicts with its pending tool round.")
            pending_call = pending_calls.get(lifecycle.tool_call_id)
            if pending_call is None or pending_call.tool_name != lifecycle.tool_name:
                raise RuntimeError("Workspace observation conflicts with its pending tool call.")
            tool_call_owner = (lifecycle.tool_round_id, lifecycle.tool_call_id)
            if tool_call_owner in claimed_tool_calls:
                raise RuntimeError(
                    "Workspace observation has duplicate active lifecycles for one tool call."
                )
            claimed_tool_calls.add(tool_call_owner)
        # Factory registrations deliberately retain only an unmaterialized
        # template.  A fresh worker must not call the factory merely to inspect
        # a crashed mutation window, and the template's placeholder workspace
        # and binding are not evidence that the historical concrete authority
        # conflicts.  The authenticated lifecycle/profile/pending-round tuple
        # above is sufficient to close the durable lifecycle, but not to enter
        # extension-owned observers or artifact stores.
        return registered_environment is not None and not factory_template_unavailable

    def _reconstruct_workspace_observation_staged_outcome(
        self,
        *,
        session: Session,
        checkpoint: dict[str, Any] | None,
        lifecycle: WorkspaceObservationLifecycle,
    ) -> WorkspaceObservationLifecycle | None:
        """Bind a pre-crash private terminal stage to its exact observation owner."""

        if (
            lifecycle.phase is not WorkspaceObservationPhase.BEFORE_CAPTURED
            or lifecycle.tool_outcome_event_id is not None
        ):
            return None
        pending_round = tool_round_recovery.pending_tool_round_from_checkpoint(
            checkpoint,
            redactor=self._secret_redactor,
        )
        if pending_round is None:
            return None
        identity = ToolRoundIdentity(
            model_step_id=lifecycle.model_step_id,
            model_attempt_id=lifecycle.model_attempt_id,
            tool_round_id=lifecycle.tool_round_id,
        )
        if tool_round_recovery.pending_tool_round_identity(pending_round) != identity:
            return None
        matches = [
            item
            for item in tool_round_recovery.staged_terminal_records(pending_round)
            if item.tool_call_id == lifecycle.tool_call_id
        ]
        if not matches:
            return None
        if len(matches) != 1:
            raise RuntimeError("Workspace observation has duplicate staged tool outcomes.")
        staged_event = self._validated_workspace_observation_tool_outcome(
            matches[0].event,
            lifecycle=lifecycle,
            require_bound_identity=False,
        )
        if staged_event is None:
            raise RuntimeError(
                "Workspace observation staged tool outcome conflicts with its owner."
            )
        return WorkspaceObservationLifecycle.model_validate(
            {
                **lifecycle.model_dump(mode="json"),
                "phase": WorkspaceObservationPhase.TOOL_OUTCOME_STAGED.value,
                "tool_outcome_event_id": staged_event.id,
                "tool_outcome_event_digest": workspace_observation_event_digest(staged_event),
            }
        )

    async def _repair_workspace_observation_terminal_stage(
        self,
        *,
        session: Session,
        durable_lifecycle: WorkspaceObservationLifecycle,
        lifecycle: WorkspaceObservationLifecycle,
        capture_status: Literal["recorded", "failed"],
        capture_detail_code: str | None,
    ) -> bool:
        """Repair one staged tool outcome, or verify its exact durable event."""

        if lifecycle.tool_outcome_event_id is None or lifecycle.tool_outcome_event_digest is None:
            return False
        checkpoint = await await_workspace_observation_store_read(
            lambda: self._session_store.load_checkpoint(session.id),
            operation="Workspace observation terminal-stage checkpoint read",
        )
        pending_round = tool_round_recovery.pending_tool_round_from_checkpoint(
            checkpoint,
            redactor=self._secret_redactor,
        )
        matching_stage = None
        if pending_round is not None:
            matching_stage = next(
                (
                    item
                    for item in tool_round_recovery.staged_terminal_records(pending_round)
                    if item.tool_call_id == lifecycle.tool_call_id
                ),
                None,
            )
        if matching_stage is None:
            durable = await await_workspace_observation_store_read(
                lambda: self._session_store.query_events(
                    EventQuery(
                        session_id=session.id,
                        event_id=lifecycle.tool_outcome_event_id,
                        limit=2,
                    )
                ),
                operation="Workspace observation tool-outcome event read",
            )
            return (
                len(durable) == 1
                and self._validated_workspace_observation_tool_outcome(
                    durable[0].event,
                    lifecycle=lifecycle,
                )
                is not None
            )
        staged_event = self._validated_workspace_observation_tool_outcome(
            matching_stage.event,
            lifecycle=lifecycle,
        )
        if staged_event is None:
            raise RuntimeError("Workspace observation tool outcome conflicts with its stage.")
        identity = ToolRoundIdentity(
            model_step_id=lifecycle.model_step_id,
            model_attempt_id=lifecycle.model_attempt_id,
            tool_round_id=lifecycle.tool_round_id,
        )
        payload = dict(staged_event.payload)
        payload["workspace_mutation_capture_status"] = capture_status
        if capture_detail_code is None:
            payload.pop("workspace_mutation_capture_detail_code", None)
        else:
            payload["workspace_mutation_capture_detail_code"] = capture_detail_code
        staged_event = staged_event.model_copy(update={"payload": payload}, deep=True)
        stage_transform = tool_round_recovery.projected_staged_terminal_transform(
            tool_round_identity=identity,
            event=staged_event,
        )

        def guarded_stage_transform(
            current_session: Session,
            current_checkpoint: dict[str, Any] | None,
        ) -> dict[str, Any]:
            if current_session.run_epoch != session.run_epoch:
                raise RuntimeError("Workspace observation recovery ownership changed.")
            current = workspace_observations_from_checkpoint(current_checkpoint).get(
                durable_lifecycle.window_id
            )
            if current != durable_lifecycle:
                raise RuntimeError("Workspace observation changed before stage repair.")
            return stage_transform(current_session, current_checkpoint)

        await await_workspace_observation_store_mutation(
            lambda: self._session_store.transform_checkpoint(
                session.id,
                guarded_stage_transform,
            ),
            operation="Workspace observation terminal-stage repair",
        )
        return True

    @staticmethod
    def _validated_workspace_observation_tool_outcome(
        event: Event,
        *,
        lifecycle: WorkspaceObservationLifecycle,
        require_bound_identity: bool = True,
    ) -> Event | None:
        """Return one detached terminal event only when its complete owner matches."""

        if type(require_bound_identity) is not bool:
            raise TypeError("require_bound_identity must be a boolean.")

        identity = ToolRoundIdentity(
            model_step_id=lifecycle.model_step_id,
            model_attempt_id=lifecycle.model_attempt_id,
            tool_round_id=lifecycle.tool_round_id,
        )
        try:
            staged_event = restore_staged_terminal_authority(
                event,
                session_id=lifecycle.session_id,
                tool_round_identity=identity,
            )
        except Exception:
            return None
        if (
            staged_event.type
            not in {
                EventType.TOOL_CALL_COMPLETED,
                EventType.TOOL_CALL_FAILED,
                EventType.TOOL_CALL_BLOCKED,
            }
            or staged_event.session_id != lifecycle.session_id
            or staged_event.interaction_id != lifecycle.interaction_id
            or staged_event.agent_name != lifecycle.agent_name
            or staged_event.environment_name != lifecycle.environment_name
            or staged_event.tool_name != lifecycle.tool_name
            or staged_event.payload.get("tool_call_id") != lifecycle.tool_call_id
            or not identity.matches_payload(staged_event.payload)
        ):
            return None
        if require_bound_identity and (
            staged_event.id != lifecycle.tool_outcome_event_id
            or workspace_observation_event_digest(staged_event)
            != lifecycle.tool_outcome_event_digest
        ):
            return None
        return staged_event

    async def _workspace_observation_tool_outcome_evidence_valid(
        self,
        *,
        session: Session,
        checkpoint: dict[str, Any] | None,
        lifecycle: WorkspaceObservationLifecycle,
    ) -> bool:
        """Validate exact tool evidence before any extension-owned artifact read."""

        pending_round = tool_round_recovery.pending_tool_round_from_checkpoint(
            checkpoint,
            redactor=self._secret_redactor,
        )
        matching_stages = (
            []
            if pending_round is None
            else [
                item
                for item in tool_round_recovery.staged_terminal_records(pending_round)
                if item.tool_call_id == lifecycle.tool_call_id
            ]
        )
        if len(matching_stages) > 1:
            raise RuntimeError("Workspace observation has duplicate staged tool outcomes.")
        if matching_stages:
            if (
                self._validated_workspace_observation_tool_outcome(
                    matching_stages[0].event,
                    lifecycle=lifecycle,
                )
                is None
            ):
                raise RuntimeError("Workspace observation tool outcome conflicts with its stage.")
            return True
        if lifecycle.tool_outcome_event_id is None:
            return False
        durable = await await_workspace_observation_store_read(
            lambda: self._session_store.query_events(
                EventQuery(
                    session_id=session.id,
                    event_id=lifecycle.tool_outcome_event_id,
                    limit=2,
                )
            ),
            operation="Workspace observation tool-outcome event read",
        )
        return (
            len(durable) == 1
            and self._validated_workspace_observation_tool_outcome(
                durable[0].event,
                lifecycle=lifecycle,
            )
            is not None
        )

    async def _workspace_observation_delta_evidence(
        self,
        *,
        session: Session,
        lifecycle: WorkspaceObservationLifecycle,
    ) -> tuple[
        bool,
        bool,
        WorkspaceObservationTerminalStatus,
        str | None,
    ]:
        """Classify exact durable delta evidence before artifact reconciliation."""

        if lifecycle.mutation_event_id is None or lifecycle.mutation_event_digest is None:
            raise RuntimeError("Published workspace delta lost its event identity.")
        records = await await_workspace_observation_store_read(
            lambda: self._session_store.query_events(
                EventQuery(
                    session_id=session.id,
                    event_id=lifecycle.mutation_event_id,
                    limit=2,
                )
            ),
            operation="Workspace observation delta event read",
        )
        if not records:
            return (
                False,
                False,
                WorkspaceObservationTerminalStatus.INCOMPLETE,
                "workspace_delta_evidence_missing",
            )
        if len(records) != 1 or (
            workspace_observation_event_digest(records[0].event) != lifecycle.mutation_event_digest
        ):
            return (
                False,
                True,
                WorkspaceObservationTerminalStatus.AMBIGUOUS,
                "workspace_delta_evidence_conflict",
            )
        delta_event = self._validated_workspace_observation_delta_event(
            records[0].event,
            lifecycle=lifecycle,
        )
        if delta_event is None:
            return (
                False,
                True,
                WorkspaceObservationTerminalStatus.AMBIGUOUS,
                "workspace_delta_evidence_conflict",
            )
        delta_status = delta_event.payload.get("status")
        delta_detail_code = delta_event.payload.get("detail_code")
        if type(delta_status) is not str or (
            delta_detail_code is not None and type(delta_detail_code) is not str
        ):
            return (
                False,
                True,
                WorkspaceObservationTerminalStatus.AMBIGUOUS,
                "workspace_delta_evidence_conflict",
            )
        try:
            terminal_status, terminal_detail = workspace_observation_terminal_from_delta_status(
                delta_status,
                detail_code=delta_detail_code,
            )
        except (TypeError, ValueError):
            return (
                False,
                True,
                WorkspaceObservationTerminalStatus.AMBIGUOUS,
                "workspace_delta_evidence_conflict",
            )
        return True, False, terminal_status, terminal_detail

    @staticmethod
    def _workspace_observation_unverified_artifacts(
        lifecycle: WorkspaceObservationLifecycle,
    ) -> WorkspaceObservationLifecycle:
        """Retain exact artifact identities without entering an unproven store owner."""

        artifacts = tuple(
            artifact
            if artifact.state
            in {
                WorkspaceObservationArtifactState.INTENT,
                WorkspaceObservationArtifactState.FAILED,
            }
            else artifact.model_copy(update={"state": WorkspaceObservationArtifactState.FAILED})
            for artifact in lifecycle.artifacts
        )
        if artifacts == lifecycle.artifacts:
            return lifecycle
        return WorkspaceObservationLifecycle.model_validate(
            {
                **lifecycle.model_dump(mode="json"),
                "artifacts": [artifact.model_dump(mode="json") for artifact in artifacts],
            }
        )

    @staticmethod
    def _validated_workspace_observation_delta_event(
        event: Event,
        *,
        lifecycle: WorkspaceObservationLifecycle,
    ) -> Event | None:
        """Return a detached delta event only when its complete owner matches."""

        try:
            delta_event = copy_event(event)
        except Exception:
            return None
        payload = delta_event.payload
        if (
            delta_event.type is not EventType.WORKSPACE_MUTATION_RECORDED
            or delta_event.id != lifecycle.mutation_event_id
            or workspace_observation_event_digest(delta_event) != lifecycle.mutation_event_digest
            or delta_event.session_id != lifecycle.session_id
            or delta_event.interaction_id != lifecycle.interaction_id
            or delta_event.agent_name != lifecycle.agent_name
            or delta_event.environment_name != lifecycle.environment_name
            or delta_event.tool_name != lifecycle.tool_name
            or payload.get("window_id") != lifecycle.window_id
            or payload.get("session_run_epoch") != lifecycle.source_run_epoch
            or payload.get("binding_generation_id") != lifecycle.binding_generation_id
            or payload.get("workspace_id") != lifecycle.workspace_id
            or payload.get("observer") != lifecycle.observer
            or payload.get("artifact_store_id") != lifecycle.artifact_store_id
            or payload.get("tool_call_id") != lifecycle.tool_call_id
            or payload.get("model_step_id") != lifecycle.model_step_id
            or payload.get("model_attempt_id") != lifecycle.model_attempt_id
            or payload.get("tool_round_id") != lifecycle.tool_round_id
            or payload.get("model_step") != lifecycle.model_step
            or payload.get("before_observation_id") != lifecycle.before_observation_id
            or payload.get("after_observation_id") != lifecycle.after_observation_id
            or payload.get("tool_outcome_event_id") != lifecycle.tool_outcome_event_id
            or payload.get("tool_outcome_event_digest") != lifecycle.tool_outcome_event_digest
        ):
            return None
        delta_artifacts = tuple(
            artifact
            for artifact in lifecycle.artifacts
            if artifact.evidence_kind == "revision-delta"
            and artifact.state is WorkspaceObservationArtifactState.REFERENCED
        )
        manifest_fields = (
            payload.get("manifest_artifact_id"),
            payload.get("manifest_artifact_sha256"),
            payload.get("manifest_artifact_size_bytes"),
        )
        if not delta_artifacts:
            if any(value is not None for value in manifest_fields):
                return None
        elif len(delta_artifacts) != 1 or manifest_fields != (
            delta_artifacts[0].artifact_id,
            delta_artifacts[0].sha256,
            delta_artifacts[0].size_bytes,
        ):
            return None
        return delta_event

    async def _reconcile_workspace_observation_artifacts(
        self,
        lifecycle: WorkspaceObservationLifecycle,
        *,
        registered_environment: runtime_records.RegisteredEnvironment | None,
    ) -> WorkspaceObservationLifecycle:
        if not lifecycle.artifacts:
            return lifecycle
        artifact_store = None
        if registered_environment is not None:
            candidate_store = registered_environment.environment.artifact_store
            try:
                candidate_store_id = (
                    None if candidate_store is None else getattr(candidate_store, "id", None)
                )
            except Exception:
                candidate_store_id = None
            if type(candidate_store_id) is str and workspace_observation_authority_matches(
                lifecycle.artifact_store_id,
                candidate_store_id,
                field_name="artifact_store_id",
                session_id=lifecycle.session_id,
                public_authority_alias_codec=(self._session_store.public_authority_alias_codec),
            ):
                artifact_store = candidate_store
        reconciled = []
        for artifact in lifecycle.artifacts:
            if artifact.state is WorkspaceObservationArtifactState.FAILED:
                reconciled.append(artifact)
                continue
            exists = False
            valid = False
            verification_failed = artifact_store is None
            if artifact_store is not None:
                if self._workspace_artifact_recovery_operations.reserve():
                    try:
                        read_task = asyncio.create_task(
                            capture_awaitable_outcome(
                                lambda artifact_id=artifact.artifact_id, artifact_size=artifact.size_bytes: (
                                    artifact_store.read_bytes(
                                        artifact_id,
                                        max_bytes=artifact_size,
                                    )
                                )
                            )
                        )
                    except BaseException:
                        self._workspace_artifact_recovery_operations.release_reservation()
                        raise
                    self._workspace_artifact_recovery_operations.track(read_task)
                    outcome = await await_shielded_task_outcome(
                        read_task,
                        timeout_s=30.0,
                        timeout_after_cancellation_s=0.0,
                    )
                    if read_task.done():
                        self._workspace_artifact_recovery_operations.release(read_task)
                    if outcome.cancellation is not None and not read_task.done():
                        read_task.cancel("workspace artifact recovery abandoned")
                    read_result: object = None
                    read_error = outcome.error
                    if read_error is None:
                        captured = outcome.result
                        if type(captured) is not CapturedAwaitableOutcome:
                            read_error = RuntimeError(
                                "Workspace artifact recovery returned an invalid owned outcome."
                            )
                        else:
                            read_result = captured.result
                            read_error = captured.error
                    if outcome.cancellation is not None:
                        restore_workspace_observation_cancellation_requests(
                            outcome.cancellation_requests_consumed
                        )
                    raise_workspace_observation_concurrent_control(
                        cancellation=outcome.cancellation,
                        error=read_error,
                        operation="Workspace observation artifact recovery",
                        cancellation_requests_pending=(outcome.cancellation_requests_consumed),
                    )
                    if outcome.timed_out:
                        verification_failed = True
                        read_task.cancel("workspace artifact recovery timed out")
                    elif read_error is not None and any(
                        isinstance(candidate, (KeyboardInterrupt, SystemExit, GeneratorExit))
                        for candidate in iter_exception_tree(read_error)
                    ):
                        raise read_error
                    elif read_error is not None and not isinstance(
                        read_error,
                        FileNotFoundError,
                    ):
                        verification_failed = True
                    elif read_error is None:
                        try:
                            result = copy_artifact_read_result(
                                cast("ArtifactReadResult", read_result),
                                expected_artifact_id=artifact.artifact_id,
                                max_content_bytes=artifact.size_bytes,
                            )
                            metadata = result.metadata
                            exists = True
                            valid = (
                                not result.truncated
                                and not result.redaction_truncated
                                and result.total_bytes == artifact.size_bytes
                                and result.source_bytes_read == artifact.size_bytes
                                and metadata.size_bytes == artifact.size_bytes
                                and len(result.content) == artifact.size_bytes
                                and sha256(result.content).hexdigest() == artifact.sha256
                                and workspace_observation_artifact_metadata_matches(
                                    metadata,
                                    artifact=artifact,
                                    session_id=lifecycle.session_id,
                                    agent_name=lifecycle.agent_name,
                                    environment_name=lifecycle.environment_name,
                                    window_id=lifecycle.window_id,
                                )
                            )
                        except Exception:
                            verification_failed = True
                else:
                    verification_failed = True
            if artifact.state is WorkspaceObservationArtifactState.REFERENCED:
                if verification_failed:
                    state = WorkspaceObservationArtifactState.FAILED
                elif exists and valid:
                    state = WorkspaceObservationArtifactState.REFERENCED
                else:
                    state = WorkspaceObservationArtifactState.MISSING
            elif exists and valid:
                state = WorkspaceObservationArtifactState.ORPHANED
            elif artifact.state is WorkspaceObservationArtifactState.INTENT:
                # An unacknowledged cancellation-opaque put may still finish
                # after this read. Absence is not positive evidence of failure;
                # retain the exact content identity as a possible late orphan.
                state = WorkspaceObservationArtifactState.INTENT
            else:
                state = WorkspaceObservationArtifactState.FAILED
            reconciled.append(artifact.model_copy(update={"state": state}))
        return WorkspaceObservationLifecycle.model_validate(
            {
                **lifecycle.model_dump(mode="json"),
                "artifacts": [artifact.model_dump(mode="json") for artifact in reconciled],
            }
        )

    async def _recover_incomplete_session(
        self,
        *,
        session: Session,
        session_before_fence: Session,
        previous_status: SessionStatus,
        inactive_before: datetime | None,
        reason: str,
        metadata: dict[str, Any],
        registered_agent: runtime_records.RegisteredAgentState,
        registered_provider: runtime_records.RegisteredProvider,
        registered_environment: runtime_records.RegisteredEnvironment | None,
        claim_id: str,
        execution_profile_snapshot: ActiveInvocationExecutionProfile | None,
        budget_policy: BudgetPolicy | None,
    ) -> IncompleteSessionRecoveryResult:
        actions: list[IncompleteSessionRecoveryAction] = []
        events: list[Event] = []
        checkpoint = await self._session_store.load_checkpoint(session.id)
        pending_approval = approval_support.pending_approval_from_checkpoint(
            checkpoint,
            redactor=self._secret_redactor,
            consume_on_rejection=True,
        )
        pending_user_input = pending_user_input_from_checkpoint(
            checkpoint,
            redactor=self._secret_redactor,
            consume_on_rejection=True,
        )
        pending_tool_round = tool_round_recovery.pending_tool_round_from_checkpoint(
            checkpoint,
            redactor=self._secret_redactor,
            consume_on_rejection=True,
        )
        environment_name = _environment_name(registered_environment)

        pending_provider_resolution = await load_pending_provider_operation_disposition(
            self._session_store,
            session.id,
            checkpoint=checkpoint,
        )
        if pending_provider_resolution is not None:
            pending_disposition, resolution_result = pending_provider_resolution
            if await self._retire_completed_provider_operation_disposition(
                pending=pending_disposition,
                result=resolution_result,
            ):
                actions.append(
                    IncompleteSessionRecoveryAction.REPAIRED_PROVIDER_OPERATION_RESOLUTION
                )
                checkpoint = await self._session_store.load_checkpoint(session.id)
            elif pending_disposition.action is ProviderOperationResolutionAction.FAIL:
                if execution_profile_snapshot is None:
                    raise RuntimeError(
                        "Provider-operation failure recovery has no execution profile."
                    )
                async for event in self._fail_provider_operation(
                    ProviderOperationFailureRequest(
                        resolution_event=resolution_result.event,
                        session=session,
                        registered_agent=registered_agent,
                        registered_environment=registered_environment,
                        execution_profile=execution_profile_snapshot.profile,
                        legacy_resolution_without_profile=(
                            resolution_result.record.execution_profile_fingerprint is None
                        ),
                    )
                ):
                    events.append(event)
                if not await self._retire_completed_provider_operation_disposition(
                    pending=pending_disposition,
                    result=resolution_result,
                ):
                    raise RuntimeError(
                        "Recovered provider-operation failure has no terminal outcome."
                    )
                current = await self._require_session(session.id)
                return IncompleteSessionRecoveryResult(
                    session_id=session.id,
                    previous_status=previous_status,
                    status=current.status,
                    actions=(
                        IncompleteSessionRecoveryAction.REPAIRED_PROVIDER_OPERATION_RESOLUTION,
                    ),
                    events=tuple(events),
                    message="Finished the accepted provider-operation failure.",
                )
            elif session.status is SessionStatus.INTERRUPTED:
                return IncompleteSessionRecoveryResult(
                    session_id=session.id,
                    previous_status=previous_status,
                    status=session.status,
                    actions=(IncompleteSessionRecoveryAction.SKIPPED_ACTIVE,),
                    events=tuple(events),
                    message=(
                        "Accepted provider-operation fallback is ready for fenced continuation."
                    ),
                )
            elif session.status is SessionStatus.RUNNING:
                if execution_profile_snapshot is None:
                    raise RuntimeError(
                        "Provider-operation fallback recovery has no execution profile."
                    )
                source_stage = await self._session_store.load_model_completion_stage(
                    session.id,
                    pending_disposition.stage_id,
                )
                if source_stage is None:
                    raise RuntimeError("Resolved provider-operation stage is missing.")
                recovery_context = model_completion_recovery_context_from_stage(source_stage)
                if recovery_context is None:
                    raise RuntimeError(
                        "Provider-operation fallback requires durable model-completion context."
                    )
                interaction_id = await self._activate_latest_open_interaction(session.id)
                if interaction_id is None:
                    raise RuntimeError(
                        "Provider-operation fallback recovery has no open interaction."
                    )
                async for event in self._run_pending_provider_operation_fallback(
                    pending=pending_disposition,
                    result=resolution_result,
                    session=session,
                    registered_agent=registered_agent,
                    registered_provider=registered_provider,
                    registered_environment=registered_environment,
                    execution_profile_snapshot=execution_profile_snapshot,
                    recovery_context=recovery_context,
                    budget_policy=budget_policy,
                    release_run_fence_on_cleanup=False,
                ):
                    events.append(event)
                current = await self._require_session(session.id)
                return IncompleteSessionRecoveryResult(
                    session_id=session.id,
                    previous_status=previous_status,
                    status=current.status,
                    actions=(
                        IncompleteSessionRecoveryAction.REPAIRED_PROVIDER_OPERATION_RESOLUTION,
                    ),
                    events=tuple(events),
                    message="Finished the accepted provider-operation fallback.",
                )

        if session.status in _RECOVERY_RESUMABLE_SESSION_STATUSES and (
            await self._terminal_evidence_repair_required(
                session=session,
                checkpoint=checkpoint,
            )
        ):
            repaired = await self._repair_terminal_evidence(
                session=session,
                terminal_run_epoch=session_before_fence.run_epoch,
                terminal_timestamp=session_before_fence.updated_at,
                previous_status=previous_status,
                claim_id=claim_id,
            )
            actions.extend(repaired.actions)
            events.extend(repaired.events)
            session = await self._require_session(session.id)
            checkpoint = await self._session_store.load_checkpoint(session.id)

        if inactive_before is not None:
            events.append(
                await self._event_writer.emit(
                    Event(
                        type=EventType.SESSION_RUN_FENCED,
                        session_id=session.id,
                        agent_name=session.agent_name,
                        environment_name=environment_name,
                        payload={
                            "previous_run_epoch": session_before_fence.run_epoch,
                            "run_epoch": session.run_epoch,
                            "inactive_before": inactive_before.isoformat(),
                            "reason": reason,
                            "metadata": metadata,
                        },
                    )
                )
            )

        provider_operation_addressed = False
        if session.status is SessionStatus.INTERRUPTING:
            provider_operation_addressed = (
                await self.cancel_provider_operation_for_interruption(
                    session,
                    registered_agent=registered_agent,
                    registered_provider=registered_provider,
                    registered_environment=registered_environment,
                )
                is not None
            )
            session = await self._require_session(session.id)
        active_model_stage = await self._session_store.load_active_model_completion_stage(
            session.id
        )
        if provider_operation_addressed and active_model_stage is not None:
            # Cancellation already resolved the exact in-flight operation. A
            # cancelled, pending, unavailable, or unconfirmed provider outcome
            # must not be reinterpreted as recoverable model completion before
            # the local interruption is finalized. If completion won, the
            # cancellation path promoted it and cleared the active stage above.
            model_boundary = ModelCompletionBoundaryReconciliation(
                state="none",
                session=session,
            )
        else:
            model_boundary = await self.reconcile_model_completion_boundary(
                session,
                registered_agent=registered_agent,
                registered_provider=registered_provider,
                registered_environment=registered_environment,
            )
        session = model_boundary.session
        events.extend(copy_event(event) for event in model_boundary.recovery_events)
        if model_boundary.state == "provider_operation_pending":
            return IncompleteSessionRecoveryResult(
                session_id=session.id,
                previous_status=previous_status,
                status=session.status,
                actions=(IncompleteSessionRecoveryAction.SKIPPED_ACTIVE,),
                events=tuple(events),
                message=(
                    "Provider operation is still pending; its exact durable dispatch remains "
                    "eligible for later recovery."
                ),
            )
        if model_boundary.state == "provider_operation_unavailable":
            if active_model_stage is None:
                raise RuntimeError(
                    "Unavailable provider operation has no active model-completion stage."
                )
            unavailable_event = next(
                (
                    event
                    for event in reversed(model_boundary.recovery_events)
                    if event.type == EventType.PROVIDER_OPERATION_RECOVERY_REQUIRED
                ),
                None,
            )
            if unavailable_event is None:
                raise RuntimeError(
                    "Unavailable provider operation has no durable recovery evidence."
                )
            try:
                recovery_reason = ProviderOperationUnavailableReason(
                    unavailable_event.payload.get("recovery_reason")
                )
            except (TypeError, ValueError):
                raise RuntimeError(
                    "Unavailable provider operation has malformed recovery evidence."
                ) from None
            if session.status in {SessionStatus.PENDING, SessionStatus.RUNNING}:
                session = await self._session_store.transition_status(
                    session.id,
                    from_statuses={session.status},
                    to_status=SessionStatus.INTERRUPTED,
                )
            elif session.status is not SessionStatus.INTERRUPTED:
                raise RuntimeError(
                    "Unavailable provider operation cannot pause the current session status."
                )
            interrupted_event = await self._event_writer.emit(
                Event(
                    type=EventType.SESSION_INTERRUPTED,
                    session_id=session.id,
                    interaction_id=unavailable_event.interaction_id,
                    agent_name=session.agent_name,
                    environment_name=environment_name,
                    payload={
                        "interruption_type": "provider_operation_unavailable",
                        "stage_id": active_model_stage.stage.stage_id,
                        "recovery_reason": recovery_reason.value,
                        "duplicate_request_risk": provider_operation_duplicate_request_risk(
                            recovery_reason
                        ),
                    },
                )
            )
            events.append(interrupted_event)
            return IncompleteSessionRecoveryResult(
                session_id=session.id,
                previous_status=previous_status,
                status=session.status,
                actions=(IncompleteSessionRecoveryAction.INTERRUPTED_ABANDONED,),
                events=tuple(events),
                message=(
                    "Exact provider continuation is unavailable; explicit fallback retry or "
                    "failure is required."
                ),
            )
        if (
            model_boundary.state
            in {
                "promoted",
                "provider_operation_reconciled",
            }
            and model_boundary.completion_event is not None
        ):
            events.append(copy_event(model_boundary.completion_event))
        checkpoint = await self._session_store.load_checkpoint(session.id)
        pending_approval = approval_support.pending_approval_from_checkpoint(checkpoint)
        pending_user_input = pending_user_input_from_checkpoint(checkpoint)
        pending_tool_round = tool_round_recovery.pending_tool_round_from_checkpoint(checkpoint)
        if pending_tool_round is None and await self.materialize_deferred_input_if_present(
            session.id
        ):
            actions.append(IncompleteSessionRecoveryAction.REPAIRED_TOOL_ROUND)

        if (
            pending_tool_round is not None
            and pending_approval is None
            and pending_user_input is None
        ):
            pending_durable_children = await self._pending_durable_subagent_children(
                session=session,
                checkpoint=checkpoint,
                pending_round=pending_tool_round,
                registered_agent=registered_agent,
            )
            if pending_durable_children:
                statuses = sorted({child.status.value for child in pending_durable_children})
                return IncompleteSessionRecoveryResult(
                    session_id=session.id,
                    previous_status=previous_status,
                    status=session.status,
                    actions=(IncompleteSessionRecoveryAction.SKIPPED_ACTIVE,),
                    events=tuple(events),
                    message=(
                        "Durable subagent work is still active; the parent tool round "
                        "remains pending until child completion is durable "
                        f"(children={len(pending_durable_children)}, "
                        f"statuses={','.join(statuses)})."
                    ),
                )

        if session.status in {SessionStatus.PENDING, SessionStatus.RUNNING}:
            if pending_approval is not None:
                interrupt_payload = {
                    "model_step_id": pending_approval.model_step_id,
                    "model_attempt_id": pending_approval.model_attempt_id,
                    "tool_round_id": pending_approval.tool_round_id,
                    "interruption_type": _INTERRUPTION_TYPE_TOOL_APPROVAL_REQUIRED,
                    **approval_support.bounded_pending_approval_event_payload(
                        pending_approval,
                        redactor=self._secret_redactor,
                    ),
                    "recovered": True,
                    "reason": reason,
                    "metadata": metadata,
                }
            elif pending_user_input is not None:
                interrupt_payload = {
                    "model_step_id": pending_user_input.model_step_id,
                    "model_attempt_id": pending_user_input.model_attempt_id,
                    "tool_round_id": pending_user_input.tool_round_id,
                    "interruption_type": _INTERRUPTION_TYPE_USER_INPUT_REQUIRED,
                    **pending_user_input_interruption_payload(pending_user_input),
                    "recovered": True,
                    "reason": reason,
                    "metadata": metadata,
                }
            elif pending_tool_round is not None:
                interrupt_payload = {
                    **tool_round_recovery.pending_tool_round_identity(pending_tool_round).payload(),
                    "reason": reason,
                    "metadata": metadata,
                    "interruption_type": _INTERRUPTION_TYPE_RUNTIME_INTERRUPTED,
                    "recovered": True,
                }
            else:
                interrupt_payload = {
                    "reason": reason,
                    "metadata": metadata,
                    "interruption_type": _INTERRUPTION_TYPE_RUNTIME_INTERRUPTED,
                    "recovered": True,
                }
            interrupt_payload["interruption_request_id"] = str(uuid4())
            try:
                session = await self._session_store.transition_status_and_checkpoint(
                    session.id,
                    from_statuses={session.status},
                    to_status=SessionStatus.INTERRUPTING,
                    checkpoint_transform=self._pending_session_interrupt_checkpoint(
                        interrupt_payload,
                        self._clock(),
                    ),
                )
            except ValueError:
                session = await self._require_session(session.id)
                if session.status in _RECOVERY_RESUMABLE_SESSION_STATUSES:
                    return IncompleteSessionRecoveryResult(
                        session_id=session.id,
                        previous_status=previous_status,
                        status=session.status,
                        actions=(IncompleteSessionRecoveryAction.SKIPPED_TERMINAL,),
                        events=(),
                        message="Session changed during recovery; recovery skipped.",
                    )
                raise
            session = await self._require_session(session.id)
            checkpoint = await self._session_store.load_checkpoint(session.id)
            pending_approval = approval_support.pending_approval_from_checkpoint(
                checkpoint,
                redactor=self._secret_redactor,
                consume_on_rejection=True,
            )
            pending_tool_round = tool_round_recovery.pending_tool_round_from_checkpoint(
                checkpoint,
                redactor=self._secret_redactor,
                consume_on_rejection=True,
            )

        workspace_recovery_events = await self._recover_workspace_observations(
            session=session,
            registered_environment=registered_environment,
            execution_profile_snapshot=execution_profile_snapshot,
        )
        if workspace_recovery_events:
            events.extend(workspace_recovery_events)
            actions.append(IncompleteSessionRecoveryAction.REPAIRED_WORKSPACE_OBSERVATION)
            checkpoint = await await_workspace_observation_store_read(
                lambda: self._session_store.load_checkpoint(session.id),
                operation="Post-recovery workspace observation checkpoint read",
            )
            pending_tool_round = tool_round_recovery.pending_tool_round_from_checkpoint(
                checkpoint,
                redactor=self._secret_redactor,
                consume_on_rejection=True,
            )

        if pending_tool_round is not None and pending_approval is None:
            transcript_snapshot = await self._session_store.load_transcript_snapshot(session.id)
            try:
                transcript = [
                    detach_message(record.message) for record in transcript_snapshot.records
                ]
                expected_transcript_cursor = transcript_snapshot.cursor
            finally:
                del transcript_snapshot
            try:
                async for event in self.recover_pending_tool_round(
                    session=session,
                    registered_agent=registered_agent,
                    registered_environment=registered_environment,
                    messages=transcript,
                    execution_profile=(
                        None
                        if execution_profile_snapshot is None
                        else execution_profile_snapshot.profile
                    ),
                    incomplete_recovery_claimed=True,
                    expected_transcript_cursor=expected_transcript_cursor,
                ):
                    events.append(event)
            except ToolApprovalRequired:
                # Fail-closed planning of an ambiguous crash boundary may
                # atomically restore the human gate. Incomplete-session
                # recovery owns the outer interrupt transition, so retain the
                # paired approval and finish that transition below instead of
                # treating the pause as failure.
                pass
            actions.append(IncompleteSessionRecoveryAction.REPAIRED_TOOL_ROUND)
            session = await self._require_session(session.id)
            checkpoint = await self._session_store.load_checkpoint(session.id)

        pending_approval = approval_support.pending_approval_from_checkpoint(
            checkpoint,
            redactor=self._secret_redactor,
            consume_on_rejection=True,
        )
        if pending_approval is not None:
            if session.status == SessionStatus.FAILED:
                interrupt_payload = {
                    "model_step_id": pending_approval.model_step_id,
                    "model_attempt_id": pending_approval.model_attempt_id,
                    "tool_round_id": pending_approval.tool_round_id,
                    "interruption_type": _INTERRUPTION_TYPE_TOOL_APPROVAL_REQUIRED,
                    **approval_support.bounded_pending_approval_event_payload(
                        pending_approval,
                        redactor=self._secret_redactor,
                    ),
                    "recovered": True,
                    "reason": reason,
                    "metadata": metadata,
                    "interruption_request_id": str(uuid4()),
                }
                session = await self._session_store.transition_status_and_checkpoint(
                    session.id,
                    from_statuses={SessionStatus.FAILED},
                    to_status=SessionStatus.INTERRUPTING,
                    checkpoint_transform=self._pending_session_interrupt_checkpoint(
                        interrupt_payload,
                        self._clock(),
                    ),
                )
            session = await self._finalize_interrupting_for_recovery(
                session=session,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
                environment_name=environment_name,
                events=events,
                execution_profile=(
                    None
                    if execution_profile_snapshot is None
                    else execution_profile_snapshot.profile
                ),
            )
            actions.append(IncompleteSessionRecoveryAction.PENDING_APPROVAL)
            return IncompleteSessionRecoveryResult(
                session_id=session.id,
                previous_status=previous_status,
                status=session.status,
                actions=tuple(actions),
                events=tuple(events),
                pending_approval_id=pending_approval.approval_id,
                message="Session has a pending tool approval; resolve it with ToolApprovalRequest.",
            )

        pending_user_input = pending_user_input_from_checkpoint(
            checkpoint,
            redactor=self._secret_redactor,
            consume_on_rejection=True,
        )
        if pending_user_input is not None:
            session = await self._finalize_interrupting_for_recovery(
                session=session,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
                environment_name=environment_name,
                events=events,
                execution_profile=(
                    None
                    if execution_profile_snapshot is None
                    else execution_profile_snapshot.profile
                ),
            )
            actions.append(IncompleteSessionRecoveryAction.PENDING_USER_INPUT)
            return IncompleteSessionRecoveryResult(
                session_id=session.id,
                previous_status=previous_status,
                status=session.status,
                actions=tuple(actions),
                events=tuple(events),
                pending_user_input_id=pending_user_input.input_id,
                message="Session is awaiting user input; answer it with UserInputResponse.",
            )

        if session.status == SessionStatus.INTERRUPTING:
            session = await self._finalize_interrupting_for_recovery(
                session=session,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
                environment_name=environment_name,
                events=events,
                execution_profile=(
                    None
                    if execution_profile_snapshot is None
                    else execution_profile_snapshot.profile
                ),
            )
            actions.append(
                IncompleteSessionRecoveryAction.FINALIZED_INTERRUPT
                if previous_status == SessionStatus.INTERRUPTING
                else IncompleteSessionRecoveryAction.INTERRUPTED_ABANDONED
            )
        elif not actions:
            actions.append(IncompleteSessionRecoveryAction.SKIPPED_TERMINAL)

        message = "Recovered incomplete session."
        if actions == [IncompleteSessionRecoveryAction.SKIPPED_TERMINAL]:
            message = "Session is terminal; recovery skipped."
        return IncompleteSessionRecoveryResult(
            session_id=session.id,
            previous_status=previous_status,
            status=session.status,
            actions=tuple(actions),
            events=tuple(events),
            message=message,
        )

    async def _finalize_interrupting_for_recovery(
        self,
        *,
        session: Session,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_environment: runtime_records.RegisteredEnvironment | None,
        environment_name: str | None,
        events: list[Event],
        execution_profile: ExecutionProfileIdentity | None = None,
    ) -> Session:
        if session.status == SessionStatus.INTERRUPTING:
            async for event in self._interrupt_session_for_recovery(
                RecoveryInterruptionRequest(
                    session=session,
                    registered_agent=registered_agent,
                    registered_environment=registered_environment,
                    environment_name=environment_name,
                    execution_profile=execution_profile,
                )
            ):
                events.append(event)
            session = await self._require_session(session.id)
        return session

    async def _require_session(self, session_id: str) -> Session:
        loaded = await self._session_store.load(session_id)
        if loaded is None:
            raise KeyError(f"Session not found: {session_id}") from None
        return loaded

    async def _subagent_children_by_idempotency_key(
        self,
        parent_session_id: str,
    ) -> dict[str, Session | None]:
        children: dict[str, Session | None] = {}
        sessions = await query_all_sessions(
            self._session_store,
            SessionQuery(
                parent_session_id=parent_session_id,
                order_by=SessionOrder.CREATED_AT_ASC,
            ),
        )
        for child in sessions:
            idempotency_key = tool_round_recovery.subagent_child_idempotency_key(child)
            if idempotency_key is not None:
                # Contradictory durable claims must not make recovery attach
                # whichever child happened to be listed last.
                children[idempotency_key] = None if idempotency_key in children else child
        return children

    async def _pending_durable_subagent_children(
        self,
        *,
        session: Session,
        checkpoint: dict[str, Any] | None,
        pending_round: tool_round_recovery.PendingToolRound,
        registered_agent: runtime_records.RegisteredAgentState,
    ) -> tuple[Session, ...]:
        """Reconcile durable submissions without closing live child work as unknown."""

        lifecycle_events = await self._load_tool_round_lifecycle_events(
            session_id=session.id,
            pending_round=pending_round,
        )
        recorded_outcomes, _started_ids = tool_round_recovery.recorded_tool_outcomes(
            events=lifecycle_events,
            pending_round=pending_round,
        )
        children = await self._subagent_children_by_idempotency_key(session.id)
        pending: list[Session] = []
        for call in pending_round.tool_calls:
            if recorded_outcomes.get(call.tool_call_id) is not None:
                continue
            idempotency_key = tool_execution.tool_idempotency_key(
                session_id=session.id,
                tool_round_id=pending_round.tool_round_id,
                tool_call_id=call.tool_call_id,
            )
            recovery_arguments = self._subagent_recovery_arguments(
                checkpoint=checkpoint,
                parent_session=session,
                tool_name=call.tool_name,
                tool_round_id=pending_round.tool_round_id,
                tool_call_id=call.tool_call_id,
                idempotency_key=idempotency_key,
                fallback=call.arguments,
            )
            await self._reconcile_subagent_child(
                children,
                idempotency_key=idempotency_key,
                tool_call_id=call.tool_call_id,
                tool_name=call.tool_name,
                tool_round_id=pending_round.tool_round_id,
                arguments=recovery_arguments,
                parent_session=session,
                registered_agent=registered_agent,
            )
            child = children.get(idempotency_key)
            if child is None:
                continue
            subagent = child.metadata.get("subagent")
            if (
                isinstance(subagent, dict)
                and subagent.get("mode") == "durable"
                and child.status not in _RECOVERY_RESUMABLE_SESSION_STATUSES
            ):
                pending.append(child.model_copy(deep=True))
        return tuple(pending)

    @staticmethod
    def _subagent_recovery_arguments(
        *,
        checkpoint: dict[str, Any] | None,
        parent_session: Session,
        tool_name: str,
        tool_round_id: str,
        tool_call_id: str,
        idempotency_key: str,
        fallback: dict[str, Any],
    ) -> dict[str, Any]:
        """Restore post-hook arguments only from the exact durable spawn seed."""

        seed = durable_subagent_submission_seed_from_checkpoint(
            checkpoint,
            idempotency_key=idempotency_key,
        )
        if seed is None:
            intent = durable_subagent_submission_from_checkpoint(
                checkpoint,
                idempotency_key=idempotency_key,
            )
            if intent is not None:
                raise RuntimeError(
                    "Durable subagent submission intent has no effective-argument seed."
                )
            copied = copy_json_value(fallback, "subagent_recovery.arguments")
            if type(copied) is not dict:
                raise TypeError("Subagent recovery arguments must be an object.")
            return copied
        if (
            seed.parent_session_id != parent_session.id
            or seed.parent_session_instance_fingerprint
            != _queued_dispatch_session_instance_fingerprint(parent_session)
            or seed.tool_name != tool_name
            or seed.tool_round_id != tool_round_id
            or seed.tool_call_id != tool_call_id
            or seed.idempotency_key != idempotency_key
        ):
            raise RuntimeError(
                "Durable subagent effective-argument authority conflicts with its tool call."
            )
        copied = copy_json_value(
            seed.effective_arguments,
            "durable_subagent_recovery.effective_arguments",
        )
        if type(copied) is not dict:
            raise AssertionError("Durable subagent effective arguments must be an object.")
        return copied

    @staticmethod
    async def _reconcile_subagent_child(
        children: dict[str, Session | None],
        *,
        idempotency_key: str,
        tool_call_id: str,
        tool_name: str,
        tool_round_id: str,
        arguments: dict[str, Any],
        parent_session: Session,
        registered_agent: runtime_records.RegisteredAgentState,
    ) -> ToolResult | None:
        if idempotency_key in children and children[idempotency_key] is None:
            # Multiple children already claim this exact spawn identity. A
            # repair callback must not erase that contradiction by returning
            # whichever deterministic child it can load independently.
            return None
        registered_tool = registered_agent.tools.get(tool_name)
        if registered_tool is None or registered_tool.child_session_recovery is None:
            return None
        matcher = registered_tool.child_session_recovery
        reconciled = await matcher.reconcile_recoverable_child(
            children.get(idempotency_key),
            parent_session=parent_session,
            tool_name=tool_name,
            tool_round_id=tool_round_id,
            tool_call_id=tool_call_id,
            idempotency_key=idempotency_key,
            arguments=copy_json_value(arguments, "subagent_recovery.arguments"),
        )
        if reconciled is not None and type(reconciled) not in {Session, ToolResult}:
            raise TypeError(
                "Child-session reconciliation must return a Session, ToolResult, or None."
            )
        if type(reconciled) is ToolResult:
            return reconciled.model_copy(deep=True)
        if type(reconciled) is Session:
            existing = children.get(idempotency_key)
            if existing is not None and existing.id != reconciled.id:
                children[idempotency_key] = None
            else:
                children[idempotency_key] = reconciled
        return None

    @staticmethod
    def _reattached_subagent_result(
        children: dict[str, Session | None],
        idempotency_key: str,
        *,
        tool_call_id: str,
        tool_name: str,
        tool_round_id: str,
        arguments: dict[str, Any],
        parent_session: Session,
        registered_agent: runtime_records.RegisteredAgentState,
    ) -> ToolResult | None:
        child = children.get(idempotency_key)
        if child is None or not _matches_recoverable_subagent_child(
            child,
            idempotency_key=idempotency_key,
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            arguments=arguments,
            parent_session=parent_session,
            registered_agent=registered_agent,
        ):
            return None
        return tool_round_recovery.recovered_subagent_tool_result(
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            tool_round_id=tool_round_id,
            child=child,
        )


def _matches_recoverable_subagent_child(
    child: Session,
    *,
    idempotency_key: str,
    tool_call_id: str,
    tool_name: str,
    arguments: dict[str, Any],
    parent_session: Session,
    registered_agent: runtime_records.RegisteredAgentState,
) -> bool:
    registered_tool = registered_agent.tools.get(tool_name)
    if registered_tool is None or registered_tool.child_session_recovery is None:
        return False
    generated_prefix = child_session_id_prefix(ChildSessionKind.SUBAGENT)
    generated_identity = child.id.startswith(generated_prefix)
    if generated_identity and child.id != generate_child_session_id(
        kind=ChildSessionKind.SUBAGENT,
        parent_session_id=parent_session.id,
        logical_spawn_id=idempotency_key,
    ):
        return False
    matched = registered_tool.child_session_recovery.matches_recoverable_child(
        child,
        parent_invocation=parent_session.invocation,
        parent_session_id=parent_session.id,
        causal_budget_id=parent_session.causal_budget_id,
        environment_name=parent_session.environment_name,
        tool_call_id=tool_call_id,
        idempotency_key=idempotency_key,
        arguments=arguments,
        require_fingerprint=generated_identity,
    )
    if type(matched) is not bool:
        raise TypeError("Child-session recovery matchers must return bool.")
    return matched


def _checkpoint_without_active_incomplete_recovery_claim(
    checkpoint: dict[str, Any] | None,
    *,
    now: datetime,
) -> dict[str, Any] | None:
    """Reject live recovery ownership and remove an expired internal marker."""
    _require_aware_datetime(now, "now")
    if checkpoint is None:
        return None
    updated = copy_json_value(checkpoint, "checkpoint")
    existing = _incomplete_recovery_claim_from_checkpoint(updated)
    if existing is None:
        return updated
    if existing[1] > now:
        raise RuntimeError("Session has an active incomplete-session recovery operation.")
    updated.pop(_INCOMPLETE_RECOVERY_CLAIM_CHECKPOINT_KEY, None)
    return updated


def _incomplete_recovery_request_fingerprint(
    request: IncompleteSessionsRecoveryRequest,
) -> str:
    material = {
        "statuses": sorted(status.value for status in request.statuses),
        "inactive_before": (
            None if request.inactive_before is None else request.inactive_before.isoformat()
        ),
        "reason": request.reason,
        "metadata": request.metadata,
    }
    return sha256(
        canonical_durable_json_bytes(material, "incomplete sessions recovery cursor")
    ).hexdigest()


def _encode_incomplete_recovery_cursor(
    *,
    status: SessionStatus,
    session_cursor: str | None,
    request: IncompleteSessionsRecoveryRequest,
) -> str:
    if status not in request.statuses:
        raise ValueError("Recovery cursor status is not part of the request.")
    encoded_session_cursor: str | None = None
    if session_cursor is not None:
        session_cursor = require_clean_nonblank(session_cursor, "session cursor")
        session_cursor_bytes = session_cursor.encode("utf-8")
        if len(session_cursor_bytes) > MAX_SESSION_LIST_CURSOR_BYTES:
            raise ValueError(
                "Session-store recovery cursor exceeds its "
                f"{MAX_SESSION_LIST_CURSOR_BYTES}-byte contract."
            )
        encoded_session_cursor = base64.urlsafe_b64encode(session_cursor_bytes).decode("ascii")
    material = {
        "version": _INCOMPLETE_RECOVERY_CURSOR_VERSION,
        "status": status.value,
        "session_cursor_b64": encoded_session_cursor,
        "request_fingerprint": _incomplete_recovery_request_fingerprint(request),
    }
    encoded = base64.urlsafe_b64encode(
        canonical_durable_json_bytes(material, "incomplete sessions recovery cursor")
    ).decode("ascii")
    if len(encoded.encode("ascii")) > MAX_INCOMPLETE_SESSIONS_RECOVERY_CURSOR_BYTES:
        raise RuntimeError("Incomplete-session recovery cursor exceeds its byte limit.")
    return encoded


def _decode_incomplete_recovery_cursor(
    cursor: str,
    *,
    request: IncompleteSessionsRecoveryRequest,
) -> tuple[SessionStatus, str | None]:
    try:
        if len(cursor.encode("utf-8")) > MAX_INCOMPLETE_SESSIONS_RECOVERY_CURSOR_BYTES:
            raise ValueError("Incomplete-session recovery cursor exceeds its byte limit.")
        encoded_cursor = cursor.encode("ascii")
        raw = base64.b64decode(
            encoded_cursor,
            altchars=b"-_",
            validate=True,
        )
        if base64.urlsafe_b64encode(raw) != encoded_cursor:
            raise ValueError("Non-canonical incomplete-session recovery cursor.")
        decoded = json.loads(raw.decode("utf-8"))
    except (UnicodeError, binascii.Error, ValueError) as exc:
        raise ValueError("Invalid incomplete-session recovery cursor.") from exc
    expected_keys = {
        "version",
        "status",
        "session_cursor_b64",
        "request_fingerprint",
    }
    if type(decoded) is not dict or set(decoded) != expected_keys:
        raise ValueError("Invalid incomplete-session recovery cursor.")
    if (
        type(decoded["version"]) is not int
        or decoded["version"] != _INCOMPLETE_RECOVERY_CURSOR_VERSION
    ):
        raise ValueError("Unsupported incomplete-session recovery cursor version.")
    try:
        status = SessionStatus(decoded["status"])
    except (TypeError, ValueError) as exc:
        raise ValueError("Invalid incomplete-session recovery cursor status.") from exc
    if status not in request.statuses:
        raise ValueError("Incomplete-session recovery cursor does not match the request.")
    if type(decoded["request_fingerprint"]) is not str or decoded[
        "request_fingerprint"
    ] != _incomplete_recovery_request_fingerprint(request):
        raise ValueError("Incomplete-session recovery cursor does not match the request.")
    encoded_session_cursor = decoded["session_cursor_b64"]
    session_cursor: str | None = None
    if encoded_session_cursor is not None:
        if type(encoded_session_cursor) is not str:
            raise ValueError("Invalid incomplete-session recovery cursor.")
        try:
            encoded_session_cursor_bytes = encoded_session_cursor.encode("ascii")
            session_cursor_bytes = base64.b64decode(
                encoded_session_cursor_bytes,
                altchars=b"-_",
                validate=True,
            )
            if base64.urlsafe_b64encode(session_cursor_bytes) != encoded_session_cursor_bytes:
                raise ValueError("Non-canonical session-store recovery cursor.")
            if len(session_cursor_bytes) > MAX_SESSION_LIST_CURSOR_BYTES:
                raise ValueError("Session-store recovery cursor exceeds its byte limit.")
            session_cursor = require_clean_nonblank(
                session_cursor_bytes.decode("utf-8"),
                "session cursor",
            )
        except (UnicodeError, binascii.Error, ValueError) as exc:
            raise ValueError("Invalid incomplete-session recovery cursor.") from exc
    return status, session_cursor


def _checkpoint_with_rebased_session_run_operation(
    checkpoint: dict[str, Any],
    *,
    previous_run_epoch: int,
    run_epoch: int,
) -> dict[str, Any]:
    """Transfer an unfinished run publication to a newly fenced recovery epoch."""
    if run_epoch != previous_run_epoch + 1:
        raise ValueError(
            "A session run operation can be rebased only to the next fenced run epoch."
        )
    operation = _session_run_operation_from_checkpoint(checkpoint)
    if operation is None:
        return checkpoint
    if operation.run_epoch > previous_run_epoch:
        raise RuntimeError(
            "Session run operation belongs to a future run epoch and cannot be recovered."
        )
    updated = copy_json_value(checkpoint, "checkpoint")
    marker: dict[str, Any] = {
        "version": 1,
        "operation_id": operation.operation_id,
        "run_epoch": run_epoch,
    }
    if operation.terminal_event_id is not None:
        marker["terminal_event_id"] = operation.terminal_event_id
    if operation.queue_task_id is not None:
        marker["queue_task_id"] = operation.queue_task_id
    updated[_SESSION_RUN_OPERATION_CHECKPOINT_KEY] = marker
    return updated


def _require_aware_datetime(value: datetime, field_name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware.")
    return value


def _interrupted_tool_round_results(
    *,
    tool_calls: list[runtime_records.ToolCallRequest],
    completed_outcomes: list[runtime_records.ToolCallOutcome],
    tool_round_identity: ToolRoundIdentity,
    cancellation_artifacts: list[dict[str, Any]] | None = None,
    cancellation_artifacts_by_id: dict[str, list[dict[str, Any]]] | None = None,
) -> list[runtime_records.ToolCallOutcome]:
    completed_ids = {outcome.call.id for outcome in completed_outcomes}
    artifacts_for_interrupted_tool = (
        [] if cancellation_artifacts is None else cancellation_artifacts
    )
    interrupted_outcomes: list[runtime_records.ToolCallOutcome] = []
    for tool_call in tool_calls:
        if tool_call.id in completed_ids:
            continue
        if cancellation_artifacts_by_id is not None:
            result_artifacts = cancellation_artifacts_by_id.get(tool_call.id, [])
        else:
            result_artifacts = artifacts_for_interrupted_tool
            artifacts_for_interrupted_tool = []
        interrupted_outcomes.append(
            _interrupted_tool_call_outcome(
                tool_call=tool_call,
                tool_round_identity=tool_round_identity,
                artifacts=result_artifacts,
            )
        )
    return interrupted_outcomes


def _effective_tool_round_structured_output(
    *,
    structured_output: StructuredOutputSpec | None,
    pending_round: tool_round_recovery.PendingToolRound,
) -> StructuredOutputSpec | None:
    if type(pending_round) is not tool_round_recovery.PendingToolRound:
        raise TypeError("Pending tool round must be a PendingToolRound.")
    if structured_output is None:
        return copy_structured_output_spec(pending_round.structured_output)
    if pending_round.structured_output is None:
        return copy_structured_output_spec(structured_output)
    if structured_output.model_dump(mode="json") != pending_round.structured_output.model_dump(
        mode="json"
    ):
        raise ValueError("structured_output does not match the crashed run contract.")
    return copy_structured_output_spec(pending_round.structured_output)


def _effective_tool_round_invocation_semantics(
    *,
    request: ToolRoundRecoveryRequest,
    pending_round: tool_round_recovery.PendingToolRound,
    structured_output: StructuredOutputSpec | None,
    effective_retry_policy: EffectiveRetryPolicy,
) -> _RecoveryInvocationSemantics:
    if type(request) is not ToolRoundRecoveryRequest:
        raise TypeError("Tool-round recovery requires a ToolRoundRecoveryRequest.")
    if type(pending_round) is not tool_round_recovery.PendingToolRound:
        raise TypeError("Pending tool round must be a PendingToolRound.")
    return _RecoveryInvocationSemantics(
        max_steps=(
            request.max_steps
            if request.max_steps is not None
            else pending_round.max_steps or _DEFAULT_APPROVAL_MAX_STEPS
        ),
        limits=copy_run_limits(
            request.limits if request.limits is not None else pending_round.limits or RunLimits()
        ),
        budget_limits=copy_request_budget_limits(
            request.budget_limits
            if request.budget_limits is not None
            else pending_round.budget_limits or ()
        ),
        retry_policy=effective_retry_policy(
            request.retry_policy if request.retry_policy is not None else pending_round.retry_policy
        ),
        structured_output=copy_structured_output_spec(structured_output),
        thinking=request.thinking if request.thinking is not None else pending_round.thinking,
    )


def _effective_user_input_max_steps(
    *,
    max_steps: int | None,
    pending: PendingUserInput,
) -> int:
    # Restore the original run's max_steps on a user-input continuation. Pending
    # states written before run config was checkpointed fall back to the historical
    # request default; profile admission decides whether an explicit value is valid.
    if type(pending) is not PendingUserInput:
        raise TypeError("Pending user input must be a PendingUserInput.")
    if max_steps is not None:
        return max_steps
    if pending.max_steps is not None:
        return pending.max_steps
    return _DEFAULT_APPROVAL_MAX_STEPS


def _effective_user_input_run_limits(
    *,
    limits: RunLimits | None,
    pending: PendingUserInput,
) -> RunLimits:
    if type(pending) is not PendingUserInput:
        raise TypeError("Pending user input must be a PendingUserInput.")
    if limits is not None:
        return copy_run_limits(limits)
    if pending.limits is not None:
        return copy_run_limits(pending.limits)
    return RunLimits()


def _effective_user_input_budget_limits(
    *,
    budget_limits: tuple[BudgetLimit, ...] | None,
    pending: PendingUserInput,
) -> tuple[BudgetLimit, ...]:
    if type(pending) is not PendingUserInput:
        raise TypeError("Pending user input must be a PendingUserInput.")
    if budget_limits is not None:
        return copy_request_budget_limits(budget_limits)
    if pending.budget_limits is not None:
        return copy_request_budget_limits(pending.budget_limits)
    return ()


def _effective_user_input_retry_policy(
    *,
    retry_policy: RetryPolicy | None,
    pending: PendingUserInput,
) -> RetryPolicy | None:
    # RetryPolicy is frozen, so the persisted reference is safe to reuse.
    if type(pending) is not PendingUserInput:
        raise TypeError("Pending user input must be a PendingUserInput.")
    if retry_policy is not None:
        return retry_policy
    return pending.retry_policy


def _effective_user_input_structured_output(
    *,
    structured_output: StructuredOutputSpec | None,
    pending: PendingUserInput,
) -> StructuredOutputSpec | None:
    # Mirror _effective_approval_structured_output: inherit the paused run's spec when
    # the resolver supplies none; profile admission owns the final equality decision.
    if type(pending) is not PendingUserInput:
        raise TypeError("Pending user input must be a PendingUserInput.")
    if structured_output is None:
        return copy_structured_output_spec(pending.structured_output)
    if pending.structured_output is None:
        return copy_structured_output_spec(structured_output)
    if not _structured_output_specs_equal(structured_output, pending.structured_output):
        raise ValueError("structured_output does not match the paused run contract.")
    return copy_structured_output_spec(pending.structured_output)


def _effective_user_input_invocation_semantics(
    *,
    response: UserInputResponse | UserInputRecoveryRequest,
    pending: PendingUserInput,
    structured_output: StructuredOutputSpec | None,
    effective_retry_policy: EffectiveRetryPolicy,
) -> _RecoveryInvocationSemantics:
    if type(response) not in (UserInputResponse, UserInputRecoveryRequest):
        raise TypeError("User-input continuation requires a validated response.")
    return _RecoveryInvocationSemantics(
        max_steps=_effective_user_input_max_steps(max_steps=response.max_steps, pending=pending),
        limits=_effective_user_input_run_limits(limits=response.limits, pending=pending),
        budget_limits=_effective_user_input_budget_limits(
            budget_limits=response.budget_limits,
            pending=pending,
        ),
        retry_policy=effective_retry_policy(
            _effective_user_input_retry_policy(
                retry_policy=response.retry_policy,
                pending=pending,
            )
        ),
        structured_output=copy_structured_output_spec(structured_output),
        thinking=response.thinking if response.thinking is not None else pending.thinking,
    )


def _effective_approval_thinking(
    *,
    thinking: ThinkingConfig | None,
    pending_approval: PendingToolApproval,
) -> ThinkingConfig | None:
    # Restore the original run's thinking config on an approval continuation. Profile
    # admission decides whether an explicit value preserves the frozen invocation.
    if type(pending_approval) is not PendingToolApproval:
        raise TypeError("Pending approval must be a PendingToolApproval.")
    if thinking is not None:
        return thinking
    return pending_approval.thinking


def _effective_approval_max_steps(
    *,
    max_steps: int | None,
    pending_approval: PendingToolApproval,
) -> int:
    # Restore the original run's max_steps on an approval continuation. Approvals
    # persisted before run config was checkpointed fall back to the historical
    # request default; profile admission validates explicit values.
    if type(pending_approval) is not PendingToolApproval:
        raise TypeError("Pending approval must be a PendingToolApproval.")
    if max_steps is not None:
        return max_steps
    if pending_approval.max_steps is not None:
        return pending_approval.max_steps
    return _DEFAULT_APPROVAL_MAX_STEPS


def _effective_approval_run_limits(
    *,
    limits: RunLimits | None,
    pending_approval: PendingToolApproval,
) -> RunLimits:
    if type(pending_approval) is not PendingToolApproval:
        raise TypeError("Pending approval must be a PendingToolApproval.")
    if limits is not None:
        return copy_run_limits(limits)
    if pending_approval.limits is not None:
        return copy_run_limits(pending_approval.limits)
    return RunLimits()


def _effective_approval_budget_limits(
    *,
    budget_limits: tuple[BudgetLimit, ...] | None,
    pending_approval: PendingToolApproval,
) -> tuple[BudgetLimit, ...]:
    if type(pending_approval) is not PendingToolApproval:
        raise TypeError("Pending approval must be a PendingToolApproval.")
    if budget_limits is not None:
        return copy_request_budget_limits(budget_limits)
    if pending_approval.budget_limits is not None:
        return copy_request_budget_limits(pending_approval.budget_limits)
    return ()


def _effective_approval_retry_policy(
    *,
    retry_policy: RetryPolicy | None,
    pending_approval: PendingToolApproval,
) -> RetryPolicy | None:
    # RetryPolicy is frozen, so the persisted reference is safe to reuse.
    if type(pending_approval) is not PendingToolApproval:
        raise TypeError("Pending approval must be a PendingToolApproval.")
    if retry_policy is not None:
        return retry_policy
    return pending_approval.retry_policy


def _effective_approval_structured_output(
    *,
    structured_output: StructuredOutputSpec | None,
    pending_approval: PendingToolApproval,
) -> StructuredOutputSpec | None:
    if type(pending_approval) is not PendingToolApproval:
        raise TypeError("Pending approval must be a PendingToolApproval.")
    if structured_output is None:
        return copy_structured_output_spec(pending_approval.structured_output)
    if pending_approval.structured_output is None:
        return copy_structured_output_spec(structured_output)
    if not _structured_output_specs_equal(
        structured_output,
        pending_approval.structured_output,
    ):
        raise ValueError("Tool approval structured_output does not match the pending run contract.")
    return copy_structured_output_spec(pending_approval.structured_output)


def _effective_approval_invocation_semantics(
    *,
    request: ToolApprovalRequest | ToolApprovalRecoveryRequest,
    pending_approval: PendingToolApproval,
    structured_output: StructuredOutputSpec | None,
    effective_retry_policy: EffectiveRetryPolicy,
) -> _RecoveryInvocationSemantics:
    if type(request) not in (ToolApprovalRequest, ToolApprovalRecoveryRequest):
        raise TypeError("Tool-approval continuation requires a validated request.")
    return _RecoveryInvocationSemantics(
        max_steps=_effective_approval_max_steps(
            max_steps=request.max_steps,
            pending_approval=pending_approval,
        ),
        limits=_effective_approval_run_limits(
            limits=request.limits,
            pending_approval=pending_approval,
        ),
        budget_limits=_effective_approval_budget_limits(
            budget_limits=request.budget_limits,
            pending_approval=pending_approval,
        ),
        retry_policy=effective_retry_policy(
            _effective_approval_retry_policy(
                retry_policy=request.retry_policy,
                pending_approval=pending_approval,
            )
        ),
        structured_output=copy_structured_output_spec(structured_output),
        thinking=_effective_approval_thinking(
            thinking=request.thinking,
            pending_approval=pending_approval,
        ),
    )


def _structured_output_specs_equal(
    left: StructuredOutputSpec,
    right: StructuredOutputSpec,
) -> bool:
    if type(left) is not StructuredOutputSpec or type(right) is not StructuredOutputSpec:
        raise TypeError("Structured output comparison requires StructuredOutputSpec values.")
    return left.model_dump(mode="json") == right.model_dump(mode="json")


def _environment_name(
    registered_environment: runtime_records.RegisteredEnvironment | None,
) -> str | None:
    if registered_environment is None:
        return None
    return registered_environment.spec.name


def _has_run_budget_limit(limits: tuple[BudgetLimit, ...]) -> bool:
    return any(limit.scope == "run" for limit in limits)

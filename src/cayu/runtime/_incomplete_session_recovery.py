"""Recover incomplete sessions under shared claim and finalization authority."""

from __future__ import annotations

import base64
import binascii
import contextlib
import json
import logging
from collections.abc import Awaitable, Callable
from datetime import datetime
from hashlib import sha256
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4, uuid5

from cayu._validation import (
    canonical_durable_json_bytes,
    copy_durable_metadata,
    copy_durable_record,
    copy_json_value,
    require_clean_nonblank,
)
from cayu.approvals.tools import (
    PendingToolApproval,
)
from cayu.approvals.user_input import (
    PendingUserInput,
    UserInputPauseState,
    ambiguous_pending_user_input_from_checkpoint,
    pending_user_input_interruption_payload,
    user_input_lifecycle_authority_from_checkpoint,
)
from cayu.budgets.base import (
    BudgetPolicy,
    copy_budget_policy,
)
from cayu.collaboration.access import CollaborationAccessContext
from cayu.environments.factory import EnvironmentFactoryOperation
from cayu.events import (
    Event,
    EventType,
    copy_event,
    event_with_runtime_generated_id,
)
from cayu.execution_profiles import (
    ExecutionProfileIdentity,
)
from cayu.messages import detach_message
from cayu.observability.hooks import RuntimeHookPhase
from cayu.providers.retry_policy import RetryPolicy
from cayu.resource_access import ResourceAccessPolicy, resource_recovery
from cayu.runtime import _approval_support as approval_support
from cayu.runtime import _runtime_records as runtime_records
from cayu.runtime import _tool_call_replay as tool_call_replay
from cayu.runtime import _tool_execution as tool_execution
from cayu.runtime import _tool_round_recovery as tool_round_recovery
from cayu.runtime._continuation_task_failure import (
    load_direct_task_failure_replay,
    provider_operation_task_failure_payload,
    runtime_task_terminalization_idempotency_key,
)
from cayu.runtime._delegated_event_stream import _close_delegated_event_stream
from cayu.runtime._diagnostics import (
    ExceptionDiagnostic,
    bound_diagnostic_text,
    exception_diagnostic,
    task_failure_payload_from_diagnostic,
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
from cayu.runtime._durable_tool_round import (
    DeferredInteractionInput,
)
from cayu.runtime._environment_lifecycle import (
    EnvironmentLifecycle,
)
from cayu.runtime._event_writer import (
    RuntimeEventWriter,
)
from cayu.runtime._execution_profile_continuation import ExecutionProfileContinuation
from cayu.runtime._foreground_subagent_recovery import ForegroundSubagentRecoveryRequired
from cayu.runtime._invocation_lifecycle import (
    InvocationContext,
    reconstruct_invocation_context,
)
from cayu.runtime._model_completion_contracts import (
    ModelCompletionBoundaryReconciliation,
    ModelCompletionManualRecoveryRequired,
    model_completion_recovery_context_from_stage,
)
from cayu.runtime._model_completion_recovery import ModelCompletionRecovery
from cayu.runtime._pending_tool_round_recovery import (
    PendingToolRoundRecovery,
    RegisteredAgentResolver,
    RegisteredEnvironmentResolver,
    _environment_name,
    _matches_recoverable_subagent_child,
)
from cayu.runtime._provider_disposition_recovery import (
    ProviderDispositionRecovery,
)
from cayu.runtime._recovery_admission import (
    RecoveryAdmission,
    RecoveryMutationHook,
    _RecoveryInvocationSemantics,
)
from cayu.runtime._recovery_claims import (
    _IncompleteRecoveryClaim,
    _IncompleteRecoveryClaimLost,
)
from cayu.runtime._recovery_ownership import (
    RecoveryOwnership,
)
from cayu.runtime._recovery_requests import (
    ProviderOperationFailureRequest,
    RecoveryInterruptionRequest,
    RecoverySessionRunRequest,
    RecoveryTaskEventRequest,
    RecoveryTerminalEventRequest,
)
from cayu.runtime._run_limits import (
    SessionUsageTracker,
)
from cayu.runtime._session_control import (
    SessionControl,
)
from cayu.runtime._session_engine import SessionEngine
from cayu.runtime._session_finalization import (
    SessionFinalization,
    _checkpoint_with_pending_session_interrupt,
    _recovery_task_event,
)
from cayu.runtime._structured_output_tool_round import has_recoverable_structured_output_round
from cayu.runtime._terminal_evidence_finalization import (
    _RECOVERY_RESUMABLE_SESSION_STATUSES,
    TerminalEvidenceFinalization,
)
from cayu.runtime._terminal_evidence_reader import (
    _provider_cancellation_interrupt_payload,
)
from cayu.runtime._tool_completion import (
    load_recorded_tool_completion_policy,
    pending_round_has_completion_success,
    recorded_terminal_tool_completion_payload,
    recorded_tool_completion_result,
    tool_completion_requires_execution,
)
from cayu.runtime._tool_effect_preparation_recovery import (
    settle_prepared_tool_effects,
)
from cayu.runtime._tool_effect_state import (
    ToolEffectReconciliationRequired,
    ToolEffectStateOwner,
)
from cayu.runtime._tool_invocation.admission import (
    ToolApprovalRequired,
)
from cayu.runtime._user_input_recovery_evidence import UserInputRecoveryEvidence
from cayu.runtime._work_attempt_invocation import WorkAttemptInvocationAuthority
from cayu.runtime._work_attempt_session_mutation import record_work_attempt_execution_stop
from cayu.runtime._workspace_observation_recovery import (
    WorkspaceObservationRecovery,
)
from cayu.runtime.loop_policies import LoopPolicy
from cayu.runtime.provider_operations import (
    ProviderOperationEvidenceError,
    ProviderOperationPendingDisposition,
    ProviderOperationResolutionAction,
    ProviderOperationUnavailableReason,
    checkpoint_with_provider_operation_disposition_execution_owner,
    load_pending_provider_operation_disposition,
    provider_operation_duplicate_request_risk,
    provider_operation_resolution_outcome_event_id,
)
from cayu.sessions import _completion_finalization as completion_finalization
from cayu.sessions import _model_completion_publication as model_completion_publication
from cayu.sessions import _pending_approval_reader as pending_approval_reader
from cayu.sessions import _pending_tool_round as pending_rounds
from cayu.sessions import _pending_tool_round_reader as pending_round_reader
from cayu.sessions import _staged_tool_terminal_reader as staged_terminal_reader
from cayu.sessions._durable_operation_ownership import DurableOperationOwnership
from cayu.sessions._execution_profile_checkpoint import (
    EXECUTION_PROFILE_METADATA_KEY,
    ActiveInvocationExecutionProfile,
    active_invocation_execution_profile_from_checkpoint,
    active_invocation_execution_profile_is_released,
    active_invocation_execution_profile_matches_session_epoch,
    execution_profile_from_session_metadata,
)
from cayu.sessions._invocation_lifecycle import (
    invocation_lifecycle_receipt_history_present,
)
from cayu.sessions._invocation_terminal_decision import (
    settled_invocation_terminal_decision_from_checkpoint,
)
from cayu.sessions._terminal_evidence import (
    _INTERRUPTION_TYPE_RUNTIME_INTERRUPTED,
    _INTERRUPTION_TYPE_TOOL_APPROVAL_REQUIRED,
    _INTERRUPTION_TYPE_USER_INPUT_REQUIRED,
)
from cayu.sessions.base import (
    CheckpointTransform,
    SessionRunFenced,
    SessionRuntimePublicationConflict,
    SessionStore,
    _incomplete_recovery_claim_from_checkpoint,
    runtime_publication_checkpoint_value_digest,
)
from cayu.sessions.event_queries import EventOrder, EventQuery
from cayu.sessions.interactions import (
    INTERACTION_LIFECYCLE_EVENT_TYPES,
)
from cayu.sessions.invocation import (
    SessionExecutionSource,
    SessionInvocationBinding,
    inherited_session_invocation,
)
from cayu.sessions.queries import MAX_SESSION_LIST_CURSOR_BYTES, SessionOrder, SessionQuery
from cayu.sessions.records import (
    Session,
    SessionStatus,
    _queued_dispatch_session_instance_fingerprint,
)
from cayu.sessions.recovery import (
    MAX_INCOMPLETE_SESSIONS_RECOVERY_CURSOR_BYTES,
    IncompleteSessionRecoveryAction,
    IncompleteSessionRecoveryRequest,
    IncompleteSessionRecoveryResult,
    IncompleteSessionsRecoveryPage,
    IncompleteSessionsRecoveryRequest,
)
from cayu.tasks.dispatch import (
    _new_prepared_subagent_dispatch_envelope,
    _require_dispatch_task_authority,
    _task_matches_queued_dispatch,
)
from cayu.tasks.queries import TaskQuery
from cayu.tasks.records import Task, TaskStatus, copy_task
from cayu.tasks.store import TaskStore
from cayu.tasks.terminalization import TaskTerminalizationRequest, TaskTerminalKind
from cayu.vaults.redaction import SecretRedactor
from cayu.workspaces.observation_recovery import (
    await_workspace_observation_store_read,
    workspace_observations_from_checkpoint,
)

_ABANDONED_UNREPLAYABLE_TOOL_ROUND_CHECKPOINT_KEY = "abandoned_unreplayable_tool_round"

_COMPLETION_FINALIZATION_TASK_EVENT_NAMESPACE = UUID("ae86f400-31e6-4cd2-95f3-7d6f115c21a1")

_PROVIDER_OPERATION_UNAVAILABLE_INTERRUPT_NAMESPACE = UUID("c7b311fa-d36b-4ecb-a93a-c96e4c047f01")

_INCOMPLETE_RECOVERY_CURSOR_VERSION = 1

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

_UNREPLAYABLE_TOOL_ROUND_ARCHIVE_SESSION_STATUSES = frozenset(
    {
        SessionStatus.INTERRUPTING,
        SessionStatus.INTERRUPTED,
        SessionStatus.FAILED,
    }
)


if TYPE_CHECKING:
    from cayu.runtime._external_execution_to_wait import _ExternalExecutionToWait


def _retain_abandoned_unreplayable_tool_round(
    checkpoint: dict[str, Any],
    durable_round: dict[str, Any],
) -> dict[str, Any]:
    """Retain every opaque round while making repeated archival idempotent."""

    copied = copy_durable_record(checkpoint, "checkpoint")
    abandoned = copied.get(_ABANDONED_UNREPLAYABLE_TOOL_ROUND_CHECKPOINT_KEY)
    if abandoned is None:
        copied[_ABANDONED_UNREPLAYABLE_TOOL_ROUND_CHECKPOINT_KEY] = {
            "schema_version": 1,
            "reason": "opaque_provider_state",
            "tool_round": durable_round,
        }
        return copied
    if type(abandoned) is not dict or type(abandoned.get("tool_round")) is not dict:
        raise RuntimeError("Session retains malformed abandoned tool-round evidence.")
    if abandoned["tool_round"] == durable_round:
        return copied
    prior = abandoned.get("prior_tool_rounds", [])
    if type(prior) is not list or any(type(item) is not dict for item in prior):
        raise RuntimeError("Session retains malformed abandoned tool-round history.")
    abandoned["prior_tool_rounds"] = [*prior, abandoned["tool_round"]]
    abandoned["tool_round"] = durable_round
    return copied


logger = logging.getLogger(__name__)


class _RecoveryPreflightMutationRequired(RuntimeError):
    """Internal sentinel proving that recovery reached its first write boundary."""


IncompleteRecoveryScopeHook = Callable[[str], Awaitable[None]]

IncompleteRecoveryResultHook = Callable[
    [IncompleteSessionRecoveryResult, InvocationContext | None],
    Awaitable[IncompleteSessionRecoveryResult],
]


def _incomplete_recovery_request_fingerprint(
    request: IncompleteSessionsRecoveryRequest,
) -> str:
    material = {
        "statuses": sorted(status.value for status in request.statuses),
        "inactive_for_seconds": request.inactive_for_seconds,
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


def _require_recovery_max_steps(value: int | None) -> int:
    if value is None:
        raise ValueError("Recovery requires recorded invocation max_steps.")
    return value


class IncompleteSessionRecovery:
    """Recover incomplete sessions under shared claim and finalization authority."""

    def __init__(
        self,
        *,
        clock: Callable[[], datetime],
        deferred_input: DeferredInteractionInput,
        effective_retry_policy: Callable[[RetryPolicy | None], RetryPolicy],
        engine: SessionEngine,
        environment_lifecycle: EnvironmentLifecycle,
        event_writer: RuntimeEventWriter,
        execution_profile_continuation: ExecutionProfileContinuation,
        loop_policies: tuple[LoopPolicy, ...],
        model_completion_recovery: ModelCompletionRecovery,
        pending_tool_round_recovery: PendingToolRoundRecovery,
        provider_disposition: ProviderDispositionRecovery,
        recovery_admission: RecoveryAdmission,
        recovery_ownership: RecoveryOwnership,
        require_participant_execution: Callable[
            [Session, CollaborationAccessContext | None], Awaitable[None]
        ],
        resolve_budget_policy: Callable[[], BudgetPolicy | None],
        resolve_registered_agent: RegisteredAgentResolver,
        resolve_registered_environment: RegisteredEnvironmentResolver,
        resolve_registered_provider: Callable[[str], runtime_records.RegisteredProvider],
        runtime_hooks: tuple[runtime_records.RegisteredRuntimeHook, ...],
        secret_redactor: SecretRedactor,
        session_control: SessionControl[SessionUsageTracker],
        session_finalization: SessionFinalization,
        session_store: SessionStore,
        resource_access_policy: ResourceAccessPolicy | None,
        task_store: TaskStore | None,
        user_input_evidence: UserInputRecoveryEvidence,
        workspace_observation_recovery: WorkspaceObservationRecovery,
        terminal_finalization: TerminalEvidenceFinalization,
    ) -> None:
        self._clock = clock
        self._deferred_input = deferred_input
        self._effective_retry_policy = effective_retry_policy
        self._engine = engine
        self._environment_lifecycle = environment_lifecycle
        self._event_writer = event_writer
        self._execution_profile_continuation = execution_profile_continuation
        self._loop_policies = loop_policies
        self._model_completion_recovery = model_completion_recovery
        self._pending_tool_round_recovery = pending_tool_round_recovery
        self._provider_disposition = provider_disposition
        self._recovery_admission = recovery_admission
        self._recovery_ownership = recovery_ownership
        self._require_participant_execution = require_participant_execution
        self._resolve_budget_policy = resolve_budget_policy
        self._resolve_registered_agent = resolve_registered_agent
        self._resolve_registered_environment = resolve_registered_environment
        self._resolve_registered_provider = resolve_registered_provider
        self._runtime_hooks = runtime_hooks
        self._secret_redactor = secret_redactor
        self._session_control = session_control
        self._session_finalization = session_finalization
        self._session_store = session_store
        self._resource_access_policy = resource_access_policy
        self._task_store = task_store
        self._user_input_evidence = user_input_evidence
        self._workspace_observation_recovery = workspace_observation_recovery
        self.terminal_finalization = terminal_finalization

    async def _automatic_provider_disposition_task_context(
        self,
        pending: ProviderOperationPendingDisposition,
    ) -> tuple[str | None, bool]:
        """Resolve direct task authority or require an elected typed continuation.

        Generic session recovery carries no task-worker credential. It may finish
        an ordinary workerless attachment, but it must never consume or bypass an
        interrupted-handoff generation owned by an elected worker. A terminal
        task likewise requires its exact receipt-authenticated typed replay.
        """

        stage = await self._session_store.load_model_completion_stage(
            pending.session_id,
            pending.stage_id,
        )
        if stage is None:
            raise RuntimeError("Resolved provider-operation stage is missing.")
        recovery_context = model_completion_recovery_context_from_stage(stage)
        if recovery_context is None:
            raise RuntimeError(
                "Provider-operation disposition requires durable model-completion context."
            )
        task_id = recovery_context.task_id
        if task_id is None:
            return None, False
        if self._task_store is None:
            raise RuntimeError("Attached provider disposition requires a task store.")
        task = await self._task_store.load_task(task_id)
        session = await self._session_store.load(pending.session_id)
        if task is None or session is None:
            raise RuntimeError("Attached provider disposition lost its task or session.")
        if task.session_id != session.id or task.session_instance_id != session.instance_id:
            raise RuntimeError("Attached provider disposition changed task-session identity.")
        if task.status is TaskStatus.FAILED:
            direct_failure = await load_direct_task_failure_replay(
                self._task_store,
                task_id=task_id,
                session_id=session.id,
                session_instance_id=session.instance_id,
                expected_error=provider_operation_task_failure_payload(session_id=session.id),
                claimed_terminalization_idempotency_key=(
                    runtime_task_terminalization_idempotency_key(
                        task_id=task_id,
                        session_id=session.id,
                        kind=TaskTerminalKind.FAILED,
                    )
                ),
            )
            return task_id, direct_failure is None
        if task.status is not TaskStatus.RUNNING or task.worker_id is not None:
            return task_id, True
        try:
            await self._task_store.load_direct_attached_task_resume(
                task_id,
                session_id=session.id,
                session_instance_id=session.instance_id,
            )
        except (KeyError, NotImplementedError, ValueError):
            return task_id, True
        return task_id, False

    async def recover_incomplete_session(
        self,
        request: IncompleteSessionRecoveryRequest,
        *,
        before_mutation: RecoveryMutationHook | None = None,
        participant_context: CollaborationAccessContext | None = None,
        retain_open_interaction_invocation: bool = False,
        retain_invocation_context: Callable[[InvocationContext], None] | None = None,
        preserve_interaction_id: str | None = None,
        _work_attempt: WorkAttemptInvocationAuthority | None = None,
        execution_to_wait: _ExternalExecutionToWait | None = None,
    ) -> IncompleteSessionRecoveryResult:
        """Repair incomplete state, including an exact post-close child continuation.

        Closed tools are never replayed. A retained foreground action-close
        marker can authorize the next model step only after profile admission
        and a fenced recovery claim; uncertain model work retains its own gate.
        """
        if _work_attempt is not None:
            if type(_work_attempt) is not WorkAttemptInvocationAuthority:
                raise TypeError("Governed recovery requires authenticated work-attempt authority.")
            admission = _work_attempt.admission
            if (
                before_mutation is None
                or preserve_interaction_id != admission.interaction_id
                or request.session_id != admission.session_id
            ):
                raise ValueError("Governed recovery conflicts with its exact claim boundary.")
        if retain_invocation_context is not None and not retain_open_interaction_invocation:
            raise ValueError("Invocation context retention requires an open recovery invocation.")
        if preserve_interaction_id is not None:
            require_clean_nonblank(preserve_interaction_id, "preserve_interaction_id")
            if before_mutation is None or retain_open_interaction_invocation:
                raise ValueError(
                    "Execution-owner replacement requires an exact claim hook and full cleanup."
                )
        retained_invocation_context: InvocationContext | None = None

        def retain_context(context: InvocationContext) -> None:
            nonlocal retained_invocation_context
            retained_invocation_context = context
            if retain_invocation_context is not None:
                retain_invocation_context(context)

        session = await self._session_store.load(request.session_id)
        if session is None:
            raise KeyError(f"Session not found: {request.session_id}") from None
        if _work_attempt is not None:
            admitted = _work_attempt.admission
            profile = execution_profile_from_session_metadata(session.metadata)
            if (
                SessionInvocationBinding(
                    id=session.id,
                    session_instance_id=session.instance_id,
                    invocation=session.invocation,
                )
                != admitted.session_invocation
                or profile is None
                or profile.fingerprint != admitted.source_execution_profile_fingerprint
            ):
                raise ValueError("Governed recovery session authority changed before admission.")
        recovered = await self.recover_incomplete_session_scoped(
            execution_to_wait=execution_to_wait,
            participant_context=participant_context,
            _work_attempt=_work_attempt,
            preserve_interaction_id=preserve_interaction_id,
            session=session,
            inactive_for_seconds=request.inactive_for_seconds,
            reason=request.reason,
            metadata=request.metadata,
            before_mutation=before_mutation,
            retain_open_interaction_invocation=retain_open_interaction_invocation,
            retain_invocation_context=(
                retain_context if retain_open_interaction_invocation else None
            ),
        )
        return await self._finish_provider_operation_disposition_after_recovery(
            recovered,
            before_mutation=before_mutation,
            invocation_context=retained_invocation_context,
        )

    async def interrupt_incomplete_session_for_manual_tool_recovery(
        self,
        request: IncompleteSessionRecoveryRequest,
    ) -> IncompleteSessionRecoveryResult:
        """Fence a stale run and retain its pending tool effect for a typed decision."""

        session = await self._session_store.load(request.session_id)
        if session is None:
            raise KeyError(f"Session not found: {request.session_id}") from None
        return await self.recover_incomplete_session_scoped(
            session=session,
            inactive_for_seconds=request.inactive_for_seconds,
            reason=request.reason,
            metadata=request.metadata,
            interrupt_for_manual_tool_recovery=True,
        )

    async def _finish_provider_operation_disposition_after_recovery(
        self,
        recovered: IncompleteSessionRecoveryResult,
        *,
        before_mutation: RecoveryMutationHook | None = None,
        invocation_context: InvocationContext | None = None,
    ) -> IncompleteSessionRecoveryResult:
        if IncompleteSessionRecoveryAction.PENDING_ALLOCATION_CLEANUP in recovered.actions:
            return recovered
        pending_resolution = await load_pending_provider_operation_disposition(
            self._session_store,
            recovered.session_id,
        )
        if pending_resolution is None:
            return recovered
        pending, result = pending_resolution
        (
            task_id,
            requires_typed_continuation,
        ) = await self._automatic_provider_disposition_task_context(pending)
        if requires_typed_continuation:
            return recovered
        if before_mutation is not None:
            await before_mutation()
        disposition_events = [
            event
            async for event in self._provider_disposition.finish_pending_provider_operation_disposition(
                pending=pending,
                result=result,
                invocation_context=invocation_context,
                task_id=task_id,
            )
        ]
        current = await self._recovery_ownership.require_session(recovered.session_id)
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
            or session.runtime_name != intent.child_runtime_name
            or session.runtime_version != intent.child_runtime_version
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
                        inactive_for_seconds=request.inactive_for_seconds,
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
                    if candidate.id in result_session_ids:
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

    async def terminalize_zero_work_interruption(
        self,
        *,
        session: Session,
        inactive_for_seconds: int | None,
        commit: bool = False,
        recovery_ownership: DurableOperationOwnership | None = None,
    ) -> IncompleteSessionRecoveryResult | None:
        from cayu.runtime._zero_work_interruption import ZeroWorkInterruptionRequest

        if session.status not in {SessionStatus.INTERRUPTING, SessionStatus.INTERRUPTED}:
            return None
        if self._session_control.has_active_tasks(session.id):
            return None
        if self._task_store is not None and await self._task_store.list_tasks(
            TaskQuery(session_id=session.id, limit=1)
        ):
            return None
        checkpoint = await self._session_store.load_checkpoint(session.id)
        request = ZeroWorkInterruptionRequest(
            session, checkpoint, inactive_for_seconds, commit, recovery_ownership
        )
        try:
            publication = await self._session_store._terminalize_zero_work_interruption(request)
        except Exception:
            if not commit:
                raise
            readback = await self._session_store._terminalize_zero_work_interruption(
                ZeroWorkInterruptionRequest(
                    session, checkpoint, inactive_for_seconds, False, recovery_ownership
                )
            )
            if readback is None or not readback.replayed:
                raise
            publication = readback
        if publication is None:
            return None
        return IncompleteSessionRecoveryResult(
            session_id=session.id,
            previous_status=session.status,
            status=publication.session.status,
            actions=(IncompleteSessionRecoveryAction.TERMINALIZED_ZERO_WORK,),
            events=publication.events,
            message="Terminalized proven zero work without reconstructing an executable profile.",
        )

    async def preflight_incomplete_session(
        self,
        *,
        session: Session,
        inactive_for_seconds: int | None,
        participant_context: CollaborationAccessContext | None = None,
    ) -> IncompleteSessionRecoveryResult | None:
        """Validate one recovery path without acquiring claims or mutating state.

        ``None`` means the exact current state reached the coordinator's guarded
        mutation boundary. A returned result is a read-only disposition such as
        an active owner, a pending approval, or already-complete terminal state.
        The sentinel is raised only by the same ``before_mutation`` hook used by
        verifier-aware recovery admission, so registration and execution-profile
        incompatibilities are reported before an operator plan can authorize a
        write. Historical accounting defers this sentinel until continuation
        preflight; profile rejection diagnostics are not published by planning.
        """

        if type(session) is not Session:
            raise TypeError("session must be a Session.")

        async def prevent_mutation() -> None:
            raise _RecoveryPreflightMutationRequired

        try:
            return await self.recover_incomplete_session_scoped(
                session=session.model_copy(deep=True),
                inactive_for_seconds=inactive_for_seconds,
                reason="operator_recovery_plan_preflight",
                metadata={"source": "registered_application_recovery_plan"},
                before_mutation=prevent_mutation,
                record_profile_rejection=False,
                participant_context=participant_context,
            )
        except _RecoveryPreflightMutationRequired:
            return None

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
        retained_invocation_context: InvocationContext | None = None

        def retain_invocation_context(context: InvocationContext) -> None:
            nonlocal retained_invocation_context
            if (
                retained_invocation_context is not None
                and retained_invocation_context is not context
            ):
                raise RuntimeError("Batch recovery produced conflicting live invocation authority.")
            retained_invocation_context = context

        try:
            if before_recovery is not None:
                await before_recovery(session.id)
            try:
                result = await self.recover_incomplete_session_scoped(
                    session=session,
                    inactive_for_seconds=request.inactive_for_seconds,
                    reason=request.reason,
                    metadata=request.metadata,
                    before_mutation=(
                        None if before_mutation is None else lambda: before_mutation(session.id)
                    ),
                    retain_open_interaction_invocation=(reconcile_result is not None),
                    retain_invocation_context=(
                        retain_invocation_context if reconcile_result is not None else None
                    ),
                )
                result = await self._finish_provider_operation_disposition_after_recovery(
                    result,
                    before_mutation=(
                        None if before_mutation is None else lambda: before_mutation(session.id)
                    ),
                    invocation_context=retained_invocation_context,
                )
                return (
                    result
                    if reconcile_result is None
                    else await reconcile_result(result, retained_invocation_context)
                )
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

    @resource_recovery
    async def recover_incomplete_session_scoped(
        self,
        *,
        session: Session,
        inactive_for_seconds: int | None,
        reason: str,
        metadata: dict[str, Any],
        before_mutation: RecoveryMutationHook | None = None,
        record_profile_rejection: bool = True,
        retain_open_interaction_invocation: bool = False,
        retain_invocation_context: Callable[[InvocationContext], None] | None = None,
        provider_disposition_task_id: str | None = None,
        provider_disposition_task_worker_id: str | None = None,
        provider_disposition_task_handoff_id: str | None = None,
        provider_disposition_after_admission: RecoveryMutationHook | None = None,
        participant_context: CollaborationAccessContext | None = None,
        interrupt_for_manual_tool_recovery: bool = False,
        preserve_interaction_id: str | None = None,
        _work_attempt: WorkAttemptInvocationAuthority | None = None,
        execution_to_wait: _ExternalExecutionToWait | None = None,
    ) -> IncompleteSessionRecoveryResult:
        from cayu.runtime._abandoned_session_recovery import require_abandoned_execution_matches

        require_abandoned_execution_matches(session)
        reason = require_clean_nonblank(reason, "reason")
        metadata = copy_durable_metadata(metadata)
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

        if inactive_for_seconds is not None and self._session_store.supports_session_execution:
            execution = await self._session_store.inspect_session_execution(session.id)
            if execution.state == "executing":
                if execution.lease_expires_at is None:
                    raise RuntimeError("Executing session presence lacks its lease expiry.")
                return IncompleteSessionRecoveryResult(
                    session_id=session.id,
                    previous_status=previous_status,
                    status=session.status,
                    actions=(IncompleteSessionRecoveryAction.SKIPPED_EXECUTION_OWNER,),
                    execution_lease_expires_at=execution.lease_expires_at,
                    message=(
                        "Session execution owner lease is active until "
                        f"{execution.lease_expires_at.isoformat()}; retry recovery after expiry."
                    ),
                )

        return await self._recover_incomplete_session_owned(
            execution_to_wait=execution_to_wait,
            _work_attempt=_work_attempt,
            preserve_interaction_id=preserve_interaction_id,
            session=session,
            inactive_for_seconds=inactive_for_seconds,
            reason=reason,
            metadata=metadata,
            previous_status=previous_status,
            before_mutation=before_mutation,
            record_profile_rejection=record_profile_rejection,
            retain_open_interaction_invocation=retain_open_interaction_invocation,
            retain_invocation_context=retain_invocation_context,
            provider_disposition_task_id=provider_disposition_task_id,
            provider_disposition_task_worker_id=provider_disposition_task_worker_id,
            provider_disposition_task_handoff_id=provider_disposition_task_handoff_id,
            provider_disposition_after_admission=provider_disposition_after_admission,
            participant_context=participant_context,
            interrupt_for_manual_tool_recovery=interrupt_for_manual_tool_recovery,
        )

    async def _recover_incomplete_session_owned(
        self,
        *,
        session: Session,
        inactive_for_seconds: int | None,
        reason: str,
        metadata: dict[str, Any],
        previous_status: SessionStatus,
        before_mutation: RecoveryMutationHook | None,
        retain_open_interaction_invocation: bool,
        retain_invocation_context: Callable[[InvocationContext], None] | None,
        record_profile_rejection: bool = True,
        provider_disposition_task_id: str | None = None,
        provider_disposition_task_worker_id: str | None = None,
        provider_disposition_task_handoff_id: str | None = None,
        provider_disposition_after_admission: RecoveryMutationHook | None = None,
        participant_context: CollaborationAccessContext | None = None,
        interrupt_for_manual_tool_recovery: bool = False,
        preserve_interaction_id: str | None = None,
        _work_attempt: WorkAttemptInvocationAuthority | None = None,
        execution_to_wait: _ExternalExecutionToWait | None = None,
    ) -> IncompleteSessionRecoveryResult:

        external_recovery_record = None
        external_recovery_cleanup = False
        if execution_to_wait is not None:
            from cayu.runtime._external_execution_to_wait import _ExternalExecutionToWait

            if type(execution_to_wait) is not _ExternalExecutionToWait:
                raise PermissionError("External recovery requires its registered runtime owner.")
            (
                external_recovery_record,
                external_recovery_cleanup,
            ) = await execution_to_wait.inspect_recovery(session)

        if (provider_disposition_task_id is None) != (
            provider_disposition_task_worker_id is None
        ) or (
            provider_disposition_task_handoff_id is not None
            and provider_disposition_task_worker_id is None
        ):
            raise ValueError(
                "Typed provider recovery requires task and worker identities; handoff "
                "authority additionally requires both."
            )

        mutation_admitted = False

        async def admit_before_mutation() -> None:
            nonlocal mutation_admitted
            if mutation_admitted or before_mutation is None:
                return
            await before_mutation()
            mutation_admitted = True

        zero_work = await self.terminalize_zero_work_interruption(
            session=session,
            inactive_for_seconds=inactive_for_seconds,
        )
        if zero_work is not None:
            await admit_before_mutation()
            committed = await self.terminalize_zero_work_interruption(
                session=session,
                inactive_for_seconds=inactive_for_seconds,
                commit=True,
            )
            if committed is not None:
                return committed
            raise RuntimeError("Zero-work interruption authority changed before publication.")

        checkpoint = await self._session_store.load_checkpoint(session.id)
        recovered_runtime_failure = await self._engine.recover_committed_runtime_task_failure(
            session,
            checkpoint,
            previous_status,
            admit_before_mutation,
        )
        if recovered_runtime_failure is not None:
            return recovered_runtime_failure
        from cayu.runtime._producer_completion_replay import producer_completion_requires_execution

        if session.status is SessionStatus.RUNNING and producer_completion_requires_execution(
            checkpoint
        ):
            # Historical model publication permits neither policy execution nor
            # destructive fallback cleanup under an unauthenticated caller.
            # Refuse before claiming a replacement epoch; explicit stop/cleanup
            # retains its separate owner and does not require answer replay.
            await self._require_participant_execution(session, participant_context)
        pending_completion_finalization = (
            completion_finalization.pending_completion_finalization_from_checkpoint(checkpoint)
        )
        ambiguous_user_input = ambiguous_pending_user_input_from_checkpoint(checkpoint)
        if ambiguous_user_input is not None:
            return IncompleteSessionRecoveryResult(
                session_id=session.id,
                previous_status=previous_status,
                status=session.status,
                actions=(IncompleteSessionRecoveryAction.AMBIGUOUS_PENDING_USER_INPUT,),
                events=(),
                message=(
                    "Session has a historical user-input pause without exact authority; "
                    "explicitly interrupt the session before starting new work."
                ),
            )
        pending_provider_interrupt = _provider_cancellation_interrupt_payload(checkpoint)
        active_invocation_profile = active_invocation_execution_profile_from_checkpoint(checkpoint)
        if active_invocation_profile is None and invocation_lifecycle_receipt_history_present(
            checkpoint
        ):
            raise RuntimeError(
                "Incomplete-session recovery lost durable invocation profile authority."
            )
        pending_provider_disposition = await load_pending_provider_operation_disposition(
            self._session_store,
            session.id,
            checkpoint=checkpoint,
        )
        pending_provider_disposition_effect_is_durable = False
        if pending_provider_disposition is not None:
            # A durable disposition records intent, not current permission to
            # execute its fallback or terminal hooks after reconstruction.
            await self._require_participant_execution(session, participant_context)
            (
                pending_disposition_record,
                pending_disposition_result,
            ) = pending_provider_disposition
            pending_provider_disposition_effect_is_durable = (
                await self._provider_disposition.provider_operation_disposition_effect_is_durable(
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
            if external_recovery_record is not None:
                current = await self._recovery_ownership.require_session(session.id)
                if (
                    current.instance_id == session.instance_id
                    and current.run_epoch != session.run_epoch
                ):
                    # A competing owner advanced between the session and
                    # checkpoint reads. Do not reinterpret that newer profile
                    # or acquire another epoch from the stale wait snapshot.
                    # The adapter separately reconciles the exact native ticket.
                    return IncompleteSessionRecoveryResult(
                        session_id=session.id,
                        previous_status=previous_status,
                        status=current.status,
                        actions=(IncompleteSessionRecoveryAction.SKIPPED_ACTIVE,),
                        events=(),
                        message="Another owner advanced external wait recovery.",
                    )
            raise RuntimeError(
                "Active invocation execution profile does not match the recovery epoch."
            )
        pending_allocations = await self._environment_lifecycle.pending_allocation_names(session)
        unsettled_environment_lifecycle = await self._environment_lifecycle.has_unsettled_progress(
            session_id=session.id,
            interaction_id=(
                None
                if active_invocation_profile is None
                else active_invocation_profile.interaction_id
            ),
        )
        durable_child_guard = await self._pending_durable_subagent_recovery_guard(
            session=session,
            checkpoint=checkpoint,
            previous_status=previous_status,
        )
        if durable_child_guard is not None:
            return durable_child_guard
        pending_approval = pending_approval_reader.pending_approval_from_checkpoint(
            checkpoint,
            redactor=self._secret_redactor,
            consume_on_rejection=True,
        )
        pending_user_input, _resolution_intent = user_input_lifecycle_authority_from_checkpoint(
            checkpoint,
            redactor=self._secret_redactor,
            consume_on_rejection=True,
            current_run_epoch=session.run_epoch,
            runtime_session=session,
        )
        pending_tool_round = pending_round_reader.pending_tool_round_from_checkpoint(
            checkpoint,
            redactor=self._secret_redactor,
            consume_on_rejection=True,
            runtime_session=session,
        )
        from cayu.sessions._foreground_child_checkpoint import (
            FOREGROUND_CHILD_POST_ACTION_CONTINUATION_KEY,
            post_action_continuation_from_checkpoint,
        )

        post_action_transfer: CheckpointTransform | None = None
        post_action_round: pending_rounds.PendingToolRound | None = None
        post_action = post_action_continuation_from_checkpoint(checkpoint)
        if post_action is not None:
            latest_model = await self._session_store.query_events(
                EventQuery(
                    session_id=session.id,
                    event_type=EventType.MODEL_STARTED,
                    order_by=EventOrder.SEQUENCE_DESC,
                    limit=1,
                )
            )
            # Once a later model attempt is durable, its normal recovery owner
            # takes over. A retained close marker must never rewind that work.
            if (
                session.status is not SessionStatus.RUNNING
                or await self._session_store.load_active_model_completion_stage(session.id)
                is not None
                or (
                    latest_model
                    and latest_model[0].event.payload.get("model_step_id")
                    != post_action.pending_tool_round.get("model_step_id")
                )
            ):
                post_action = None
        if post_action is not None:
            if (
                post_action.wait.child_session_id != session.id
                or post_action.wait.child_session_instance_id != session.instance_id
            ):
                raise RuntimeError("Post-action continuation targets another child session.")
            parent = await self._session_store.load(post_action.wait.parent_effect.session_id)
            if (
                parent is None
                or parent.id != session.parent_session_id
                or parent.instance_id != post_action.wait.parent_effect.session_instance_id
            ):
                raise RuntimeError("Post-action continuation lost its parent incarnation.")
            parent_effect = await ToolEffectStateOwner(self._session_store).resolve_call(
                parent,
                tool_round_id=post_action.wait.parent_effect.tool_round_id,
                tool_call_id=post_action.wait.parent_effect.tool_call_id,
            )
            if parent_effect is None or parent_effect.intent != post_action.wait.parent_effect:
                raise RuntimeError("Post-action continuation lost its exact parent effect.")
            close_receipt = await self._session_store.load_runtime_publication_receipt(
                session.id, post_action.close_publication_id
            )
            expected_kind = (
                "approval-close"
                if post_action.action_kind == "tool_approval"
                else "user-input-close"
            )
            if (
                close_receipt is None
                or close_receipt.kind != expected_kind
                or close_receipt.session_id != session.id
                or close_receipt.publication_id != post_action.close_publication_id
                or close_receipt.intent.get("post_action_continuation_digest")
                != runtime_publication_checkpoint_value_digest(post_action.model_dump(mode="json"))
                or close_receipt.interaction_id != post_action.wait.child_interaction_id
                or any(
                    close_receipt.intent.get(field) != post_action.pending_tool_round.get(field)
                    for field in ("tool_round_id", "model_step_id", "model_attempt_id")
                )
                or post_action.continuation_revision != post_action.wait.revision
                or close_receipt.intent.get(
                    "approval_id" if expected_kind == "approval-close" else "input_id"
                )
                != post_action.action_id
            ):
                raise RuntimeError("Post-action continuation lacks exact close authority.")
            if pending_tool_round is not None:
                raise RuntimeError("Post-action continuation conflicts with a live tool round.")
            marker_value = post_action.model_dump(mode="json")
            restored = copy_json_value(post_action.pending_tool_round, "post_action.pending_round")

            def claim_post_action(
                current_session: Session, current: dict[str, Any] | None
            ) -> dict[str, Any] | None:
                if (
                    current_session.instance_id != session.instance_id
                    or current_session.run_epoch != session.run_epoch
                    or current is None
                    or current.get(FOREGROUND_CHILD_POST_ACTION_CONTINUATION_KEY) != marker_value
                    or pending_rounds.PENDING_TOOL_ROUND_CHECKPOINT_KEY in current
                ):
                    raise _IncompleteRecoveryClaimLost(
                        "Post-action continuation was already claimed or changed."
                    )
                updated = copy_json_value(current, "post_action.claimed_checkpoint")
                # Keep the marker across claim acknowledgement loss or process
                # death before the next model dispatch. The closed round is not
                # pending work and must never be republished or replayed.
                return updated

            # Inspect the completed round without publishing pending work.
            # Its exact marker is checked inside the admitted epoch claim below;
            # recovery planning must remain entirely read-only.
            post_action_transfer = claim_post_action
            post_action_round = pending_round_reader.pending_tool_round_from_checkpoint(
                {pending_rounds.PENDING_TOOL_ROUND_CHECKPOINT_KEY: restored},
                redactor=self._secret_redactor,
                consume_on_rejection=True,
                runtime_session=session,
            )
            if (
                post_action_round is None
                or active_invocation_profile is None
                or post_action_round.execution_profile_fingerprint
                != active_invocation_profile.profile.fingerprint
                or post_action.wait.child_interaction_id != active_invocation_profile.interaction_id
            ):
                raise RuntimeError("Post-action continuation conflicts with its invocation.")
        workspace_observations = workspace_observations_from_checkpoint(checkpoint)
        self._workspace_observation_recovery.validate_workspace_observation_recovery_authority(
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
        terminal_binding_finalization = None
        if session.status in _RECOVERY_RESUMABLE_SESSION_STATUSES:
            terminal_repair = await self.terminal_finalization.repair_required(
                session=session,
                checkpoint=checkpoint,
            )
            terminal_repair_required = terminal_repair
            has_pending_work = (
                execution_to_wait is not None
                or pending_approval is not None
                or pending_user_input is not None
                or pending_tool_round is not None
                or deferred_input is not None
                or active_model_completion is not None
                or bool(workspace_observations)
                or pending_provider_disposition is not None
                or pending_completion_finalization is not None
                or unsettled_environment_lifecycle
                or bool(pending_allocations)
            )
            if (
                not has_pending_work
                and active_invocation_profile is not None
                and not active_invocation_execution_profile_is_released(
                    active_invocation_profile,
                    session_id=session.id,
                    run_epoch=session.run_epoch,
                )
            ):
                terminal_binding_finalization = (
                    await self._environment_lifecycle.prepare_terminal_binding_finalization(
                        session=session,
                        execution_profile=active_invocation_profile.profile,
                        registered_environment=self._resolve_registered_environment(
                            session.environment_name
                        ),
                    )
                )
                if terminal_binding_finalization is not None:
                    pending_completion_finalization = terminal_binding_finalization.marker()
                    has_pending_work = True
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
                        inactive_for_seconds=inactive_for_seconds,
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
            if terminal_repair and not has_pending_work and active_invocation_profile is None:
                # A session that never acquired invocation authority can repair
                # its durable terminal evidence directly.  Once an invocation
                # profile exists, continue through registration/profile
                # validation and context reconstruction below so the matching
                # interaction settlement cannot be published from durable
                # session fields alone.
                await admit_before_mutation()
                return await self.terminal_finalization.repair_owned(
                    session=session,
                    inactive_for_seconds=inactive_for_seconds,
                    previous_status=previous_status,
                )

        try:
            registered_agent = self._resolve_registered_agent(session.agent_name)
        except KeyError:
            if terminal_repair_required:
                await admit_before_mutation()
                return await self.terminal_finalization.repair_owned(
                    session=session,
                    inactive_for_seconds=inactive_for_seconds,
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
                return await self.terminal_finalization.repair_owned(
                    session=session,
                    inactive_for_seconds=inactive_for_seconds,
                    previous_status=previous_status,
                )
            raise
        registered_provider = self._resolve_registered_provider(session.provider_name)
        if (
            session.status is SessionStatus.RUNNING
            and _work_attempt is None
            and await tool_completion_requires_execution(
                self._session_store, session, checkpoint, registered_agent
            )
        ):
            await self._require_participant_execution(session, participant_context)
        if workspace_observations:
            # A static registration already provides the complete stable
            # environment authority. Reject a foreign lifecycle before profile
            # continuation or the recovery claim can mutate durable ownership.
            # Factory templates remain deliberately unmaterialized here.
            self._workspace_observation_recovery.validate_workspace_observation_recovery_authority(
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
            and not unsettled_environment_lifecycle
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
            bool(pending_allocations)
            or pending_approval is not None
            or pending_user_input is not None
            or pending_tool_round is not None
            or deferred_input is not None
            or active_model_completion is not None
            or bool(workspace_observations)
            or pending_provider_disposition is not None
            or (
                pending_provider_interrupt is not None
                and session.status not in _RECOVERY_RESUMABLE_SESSION_STATUSES
            )
            or active_invocation_profile is not None
            or pending_completion_finalization is not None
            or (
                not pre_admission_profiled_session
                and session.status not in _RECOVERY_RESUMABLE_SESSION_STATUSES
                and EXECUTION_PROFILE_METADATA_KEY in session.metadata
            )
        )
        execution_profile_snapshot = None
        budget_policy_snapshot: BudgetPolicy | None = None
        if requires_execution_profile:
            published_model = model_completion_publication.model_step_publication_from_checkpoint(
                checkpoint
            )
            if published_model is not None and active_model_completion is None:
                # This accounts for already-published work under its original
                # reservation identity. It neither promotes a model stage nor
                # permits continuation under the current configuration.
                # Planning skips this accounting write but must still validate
                # continuation compatibility below. Its mutation gate remains
                # unadmitted, so subsequent writes still stop preflight.
                # Real recovery settles first.
                with contextlib.suppress(_RecoveryPreflightMutationRequired):
                    await self._model_completion_recovery.reconcile_published_model_budget(
                        session=session,
                        pointer=published_model,
                        before_mutation=admit_before_mutation,
                    )
            budget_policy_snapshot = copy_budget_policy(self._resolve_budget_policy())
            pending_disposition = (
                None if pending_provider_disposition is None else pending_provider_disposition[0]
            )
            if pending_completion_finalization is not None:
                # This recovery dispatches neither provider nor tools. Reuse
                # the frozen invocation identity across process-local object
                # replacement, then validate the current binding and target
                # against their dedicated durable recovery record.
                if active_invocation_profile is None:
                    raise RuntimeError(
                        "Pending completion finalization lost its frozen invocation profile."
                    )
                execution_profile_snapshot = active_invocation_profile
            else:
                execution_profile_snapshot = await self._execution_profile_continuation.validate(
                    session=session,
                    checkpoint=checkpoint,
                    registered_agent=registered_agent,
                    registered_provider=registered_provider,
                    budget_policy=budget_policy_snapshot,
                    request_loop_policies=(
                        None
                        if execution_to_wait is None
                        else execution_to_wait.request_loop_policies
                    ),
                    require_open_interaction=not (
                        execution_to_wait is not None
                        or (
                            active_model_completion is not None
                            and active_model_completion.stage.purpose == "auxiliary-inference"
                            and session.status in _RECOVERY_RESUMABLE_SESSION_STATUSES
                        )
                        or (
                            (terminal_repair_required or bool(pending_allocations))
                            and session.status in _RECOVERY_RESUMABLE_SESSION_STATUSES
                        )
                        or pending_provider_disposition_effect_is_durable
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
                    record_rejection=record_profile_rejection,
                )
            if pending_completion_finalization is not None:
                expected_terminal_status = (
                    SessionStatus.INTERRUPTED
                    if pending_completion_finalization["outcome"] == "interrupted"
                    else SessionStatus.FAILED
                )
                if session.status not in {SessionStatus.RUNNING, expected_terminal_status}:
                    raise RuntimeError(
                        "Pending finalization conflicts with the original terminal outcome."
                    )
                if registered_environment is None or (
                    registered_environment.spec.name
                    != pending_completion_finalization["environment_name"]
                ):
                    raise RuntimeError(
                        "Pending completion finalization resolved a different environment."
                    )
                if (
                    execution_profile_snapshot.profile.fingerprint
                    != pending_completion_finalization["execution_profile_fingerprint"]
                ):
                    raise RuntimeError("Pending completion finalization execution profile changed.")
        claim: _IncompleteRecoveryClaim | None = None
        invocation_context: InvocationContext | None = None
        authoritative_failure: BaseException | None = None
        provider_execution_transfer: CheckpointTransform | None = None
        command_recovery_context: InvocationContext | None = None
        if provider_disposition_task_id is not None:
            if pending_provider_disposition is None:
                raise ProviderOperationEvidenceError(
                    "Typed provider recovery lost its pending disposition."
                )
            expected_pending = pending_provider_disposition[0]
            if not expected_pending.execution_claimed or (
                expected_pending.execution_task_worker_id,
                expected_pending.execution_task_handoff_id,
            ) == (
                provider_disposition_task_worker_id,
                provider_disposition_task_handoff_id,
            ):
                raise ProviderOperationEvidenceError(
                    "Typed provider recovery has no predecessor execution owner to fence."
                )

            def transfer_provider_execution(
                _session: Session,
                current_checkpoint: dict[str, Any] | None,
            ) -> dict[str, Any]:
                return checkpoint_with_provider_operation_disposition_execution_owner(
                    current_checkpoint,
                    expected=expected_pending,
                    task_worker_id=provider_disposition_task_worker_id,
                    task_handoff_id=provider_disposition_task_handoff_id,
                )

            provider_execution_transfer = transfer_provider_execution

        def transfer_recovery_checkpoint(
            current_session: Session, current: dict[str, Any] | None
        ) -> dict[str, Any] | None:
            if external_recovery_record is not None:
                from cayu.runtime._continuation_recovery import (
                    require_recovery_frontier,
                    require_resolved_native_pause,
                )

                require_recovery_frontier(external_recovery_record, current_session, current)
                # Recheck under the native claim transaction: the original turn
                # may have paused after the read-only recovery inspection.
                # A retained terminal control can still request ordinary cleanup.
                if not external_recovery_cleanup:
                    require_resolved_native_pause(current)
            if post_action_transfer is not None:
                current = post_action_transfer(current_session, current)
            if provider_execution_transfer is not None:
                current = provider_execution_transfer(current_session, current)
            return current

        try:
            await admit_before_mutation()
            claim = await self._recovery_ownership.claim(
                session=session,
                inactive_for_seconds=inactive_for_seconds,
                execution_profile_snapshot=execution_profile_snapshot,
                target_status=SessionStatus.RUNNING if execution_to_wait is not None else None,
                checkpoint_transform=(
                    transfer_recovery_checkpoint
                    if post_action_transfer is not None
                    or provider_execution_transfer is not None
                    or external_recovery_record is not None
                    else None
                ),
            )
            if claim is None:
                current = await self._recovery_ownership.require_session(session.id)
                return IncompleteSessionRecoveryResult(
                    session_id=session.id,
                    previous_status=previous_status,
                    status=current.status,
                    actions=(IncompleteSessionRecoveryAction.SKIPPED_ACTIVE,),
                    events=(),
                    message="Session activity or recovery ownership changed; recovery skipped.",
                )
            if execution_profile_snapshot is not None:
                invocation_context = reconstruct_invocation_context(
                    runtime_hooks=self._runtime_hooks,
                    loop_policies=self._loop_policies,
                    session=claim.session,
                    execution_profile_snapshot=execution_profile_snapshot,
                    registered_agent=registered_agent,
                    registered_provider=registered_provider,
                    registered_environment=registered_environment,
                    budget_policy=budget_policy_snapshot,
                    recovery_claim_id=claim.claim_id,
                    work_attempt=_work_attempt,
                    request_loop_policies=(
                        () if execution_to_wait is None else execution_to_wait.request_loop_policies
                    ),
                )
            if provider_disposition_after_admission is not None:
                await provider_disposition_after_admission()

            def retain_cleanup_context(context: InvocationContext) -> None:
                nonlocal invocation_context
                invocation_context = context

            async def recover_claimed_session() -> IncompleteSessionRecoveryResult:
                nonlocal registered_environment, invocation_context, command_recovery_context
                assert claim is not None
                recovery_session = claim.session
                if terminal_binding_finalization is not None:
                    if invocation_context is None:
                        raise RuntimeError("Terminal binding cleanup lost invocation authority.")
                    await self._environment_lifecycle.checkpoint_terminal_binding_finalization(
                        session=claim.session,
                        invocation_context=invocation_context,
                        expected=terminal_binding_finalization,
                    )
                external_cleanup = False
                if execution_to_wait is not None:
                    if invocation_context is None:
                        raise RuntimeError("External wait recovery lost invocation authority.")
                    external_cleanup = await execution_to_wait.recovery_cleanup_requested(
                        invocation_context
                    )
                    if external_cleanup:
                        await execution_to_wait.attach_cleanup_writer(invocation_context)
                # Terminal external controls use ordinary interruption recovery,
                # including its effect/environment cleanup guards, not another
                # execution of the unfinished turn or its stop policies.
                if execution_to_wait is not None and not external_cleanup:
                    assert invocation_context is not None
                    boundary = (
                        await self._model_completion_recovery.reconcile_model_completion_boundary(
                            claim.session,
                            invocation_context=invocation_context,
                            registered_agent=registered_agent,
                            registered_provider=registered_provider,
                            registered_environment=registered_environment,
                        )
                    )
                    semantics = await execution_to_wait.require_recovered_result(
                        invocation_context, boundary
                    )
                    resumed = await self._event_writer.emit(
                        Event(
                            type=EventType.SESSION_RESUMED,
                            session_id=session.id,
                            agent_name=session.agent_name,
                            environment_name=_environment_name(registered_environment),
                            payload={
                                "recovered_external_wait": True,
                                "run_epoch": claim.session.run_epoch,
                            },
                        )
                    )
                    stream = self._engine.continue_run(
                        RecoverySessionRunRequest(
                            session=claim.session,
                            invocation_context=invocation_context,
                            messages=await self._session_store.load_transcript(session.id),
                            messages_to_append=[],
                            max_steps=semantics.max_steps,
                            limits=semantics.limits,
                            budget_limits=semantics.budget_limits,
                            retry_policy=semantics.retry_policy,
                            structured_output=semantics.structured_output,
                            tool_completion=semantics.tool_completion,
                            thinking=semantics.thinking,
                            request_metadata=semantics.request_metadata,
                            task_id=None,
                            task_worker_id=None,
                            task_handoff_id=None,
                            start_event_type=None,
                            start_event_payload={},
                            start_task_on_enter=False,
                            release_run_fence_on_exit=False,
                            run_limit_accounting=semantics.run_limit_accounting,
                            execution_to_wait=execution_to_wait,
                        )
                    )
                    events = [*boundary.recovery_events, resumed]
                    async with _close_delegated_event_stream(stream) as owned:
                        async for event in owned:
                            events.append(event)
                    current = await self._recovery_ownership.require_session(session.id)
                    return IncompleteSessionRecoveryResult(
                        session_id=session.id,
                        previous_status=previous_status,
                        status=current.status,
                        actions=(IncompleteSessionRecoveryAction.INTERRUPTED_ABANDONED,),
                        events=tuple(events),
                        message="Recovered the external wait's committed whole-turn result.",
                    )
                if (
                    not external_cleanup
                    and post_action_round is not None
                    and post_action is not None
                ):
                    if invocation_context is None:
                        raise RuntimeError("Post-action continuation lost invocation authority.")
                    if post_action_round.limits is None or post_action_round.budget_limits is None:
                        raise RuntimeError("Post-action continuation lost its original limits.")
                    continuation = (
                        await self._recovery_admission.prepare_recovered_tool_round_continuation(
                            session=claim.session,
                            pending_round=post_action_round,
                            invocation_semantics=_RecoveryInvocationSemantics(
                                max_steps=_require_recovery_max_steps(post_action_round.max_steps),
                                limits=post_action_round.limits,
                                budget_limits=post_action_round.budget_limits,
                                retry_policy=self._effective_retry_policy(
                                    post_action_round.retry_policy
                                ),
                                structured_output=post_action_round.structured_output,
                                thinking=post_action_round.thinking,
                            ),
                            invocation_context=invocation_context,
                            request_metadata=post_action.request_metadata,
                            task_worker_id=None,
                            task_handoff_id=None,
                            completed_model_step=post_action.completed_model_step,
                        )
                    )
                    stream = self._engine.continue_run(continuation)
                    async with _close_delegated_event_stream(stream) as owned_stream:
                        events = [event async for event in owned_stream]

                    def retire_close_marker(
                        _session: Session, current: dict[str, Any] | None
                    ) -> dict[str, Any] | None:
                        if (
                            current is None
                            or current.get(FOREGROUND_CHILD_POST_ACTION_CONTINUATION_KEY)
                            != marker_value
                        ):
                            return current
                        updated = copy_json_value(current, "post_action.completed_checkpoint")
                        updated.pop(FOREGROUND_CHILD_POST_ACTION_CONTINUATION_KEY)
                        return updated

                    await self._session_store.transform_checkpoint(session.id, retire_close_marker)
                    current_session = await self._recovery_ownership.require_session(session.id)
                    return IncompleteSessionRecoveryResult(
                        session_id=session.id,
                        previous_status=previous_status,
                        status=current_session.status,
                        actions=(IncompleteSessionRecoveryAction.REPAIRED_TOOL_ROUND,),
                        events=tuple(events),
                        message="Continued the foreground child after its committed action close.",
                    )
                if pending_allocations:
                    if (
                        registered_environment is None
                        or pending_allocations != (registered_environment.spec.name,)
                        or execution_profile_snapshot is None
                        or invocation_context is None
                    ):
                        raise RuntimeError("Pending allocation lost exact recovery authority.")
                    try:
                        await self._environment_lifecycle.reap_pending_allocation(
                            session=claim.session,
                            registered_agent=registered_agent,
                            registered_environment=registered_environment,
                            execution_profile=execution_profile_snapshot.profile,
                            invocation_context=invocation_context,
                        )
                    except Exception as exc:
                        diagnostic = exception_diagnostic(exc, redactor=self._secret_redactor)
                        return IncompleteSessionRecoveryResult(
                            session_id=session.id,
                            previous_status=previous_status,
                            status=claim.session.status,
                            actions=(IncompleteSessionRecoveryAction.PENDING_ALLOCATION_CLEANUP,),
                            message=bound_diagnostic_text(
                                "Allocation cleanup remains durably owned; retry "
                                "recover_incomplete_session. " + diagnostic.message
                            ),
                        )
                lifecycle_events = list(
                    await self._environment_lifecycle.reconcile_orphaned_progress(
                        session_id=claim.session.id,
                        interaction_id=(
                            None
                            if active_invocation_profile is None
                            else active_invocation_profile.interaction_id
                        ),
                        observed_at=self._clock(),
                    )
                )
                # Factory registrations have no live runner/workspace. Reuse the
                # round owner's exact recorded/staged evidence before inspecting
                # missing command results. Completed sibling tools need not
                # implement command recovery; unknown effects remain fenced.
                pending = pending_round_reader.pending_tool_round_from_checkpoint(
                    await self._session_store.load_checkpoint(claim.session.id)
                )
                settled_call_ids: set[str] = set()
                if (
                    not interrupt_for_manual_tool_recovery
                    and registered_environment is not None
                    and registered_environment.factory_backed
                    and invocation_context is not None
                    and pending is not None
                    and pending.source_run_epoch is not None
                    and pending.tool_calls
                ):
                    lifecycle_events_for_round = (
                        await self._pending_tool_round_recovery.load_tool_round_lifecycle_events(
                            session_id=claim.session.id, pending_round=pending
                        )
                    )
                    recorded_outcomes, _ = tool_round_recovery.recorded_tool_outcomes(
                        events=lifecycle_events_for_round, pending_round=pending
                    )
                    terminal_events = [
                        *lifecycle_events_for_round,
                        *(
                            item.event
                            for item in staged_terminal_reader.staged_terminal_records(pending)
                        ),
                    ]
                    settled_call_ids = {
                        call_id
                        for call_id in recorded_outcomes
                        if self._pending_tool_round_recovery.tool_terminals_are_settled(
                            [
                                event
                                for event in terminal_events
                                if event.payload.get("tool_call_id") == call_id
                            ],
                            [call_id],
                        )
                    }
                if (
                    not interrupt_for_manual_tool_recovery
                    and registered_environment is not None
                    and registered_environment.factory_backed
                    and invocation_context is not None
                    and pending is not None
                    and pending.source_run_epoch is not None
                    and pending.tool_calls
                    # Fully settled rounds need only native evidence repair,
                    # not a runner reconnect or Docker binding authority.
                    and any(
                        call.tool_call_id not in settled_call_ids for call in pending.tool_calls
                    )
                    and all(
                        [
                            call.tool_call_id in settled_call_ids
                            or await self._pending_tool_round_recovery.has_recoverable_durable_tool_result(
                                session=recovery_session,
                                tool_round_id=pending.tool_round_id,
                                tool_call_id=call.tool_call_id,
                            )
                            for call in pending.tool_calls
                        ]
                    )
                ):
                    try:
                        command_binding_authority = (
                            await self._environment_lifecycle.authorize_command_binding_recovery(
                                session=claim.session,
                                invocation_context=invocation_context,
                                source_run_epoch=pending.source_run_epoch,
                            )
                        )
                        factory_started = await self._environment_lifecycle.emit_factory_started(
                            session=claim.session,
                            registered_agent=registered_agent,
                            registered_environment=registered_environment,
                            execution_profile=invocation_context.profile,
                            invocation_context=invocation_context,
                        )
                        if factory_started is not None:
                            lifecycle_events.append(factory_started)
                        resolution = await self._environment_lifecycle.resolve_factory(
                            session=claim.session,
                            registered_agent=registered_agent,
                            registered_environment=registered_environment,
                            started_event=factory_started,
                            operation=EnvironmentFactoryOperation.RECONNECT,
                            execution_profile=invocation_context.profile,
                            invocation_context=invocation_context,
                            command_recovery=command_binding_authority,
                        )
                        lifecycle_events.extend(resolution.events)
                        registered_environment = resolution.registered_environment
                        if registered_environment is not None:
                            invocation_context = invocation_context.with_registered_environment(
                                registered_environment, validated_profile=invocation_context.profile
                            )
                        if resolution.error is not None:
                            raise resolution.error
                        binding_started = await self._environment_lifecycle.emit_binding_started(
                            session=claim.session,
                            registered_agent=registered_agent,
                            registered_environment=registered_environment,
                            execution_profile=invocation_context.profile,
                            invocation_context=invocation_context,
                        )
                        if binding_started is not None:
                            lifecycle_events.append(binding_started)
                        bound = await self._environment_lifecycle.bind(
                            session=claim.session,
                            registered_agent=registered_agent,
                            registered_environment=registered_environment,
                            started_event=binding_started,
                            execution_profile=invocation_context.profile,
                            invocation_context=invocation_context,
                            command_recovery=command_binding_authority,
                        )
                        lifecycle_events.extend(bound.events)
                        registered_environment = bound.registered_environment
                        if registered_environment is not None:
                            invocation_context = invocation_context.with_registered_environment(
                                registered_environment, validated_profile=invocation_context.profile
                            )
                        if bound.error is not None:
                            raise bound.error
                        command_recovery_context = invocation_context
                    except BaseException as error:
                        await self._environment_lifecycle.abort_environment_setup(
                            session_id=claim.session.id,
                            original_error=error,
                            execution_profile=invocation_context.profile,
                            invocation_context=invocation_context,
                        )
                        raise
                recovered = await self._recover_incomplete_session(
                    retain_cleanup_context=retain_cleanup_context,
                    preserve_interaction_id=preserve_interaction_id,
                    participant_context=participant_context,
                    session=claim.session,
                    session_before_fence=claim.session_before_fence,
                    previous_status=previous_status,
                    inactive_for_seconds=inactive_for_seconds,
                    reason=reason,
                    metadata=metadata,
                    registered_agent=registered_agent,
                    registered_provider=registered_provider,
                    registered_environment=registered_environment,
                    invocation_context=invocation_context,
                    claim_id=claim.claim_id,
                    execution_profile_snapshot=execution_profile_snapshot,
                    budget_policy=budget_policy_snapshot,
                    provider_disposition_task_id=provider_disposition_task_id,
                    provider_disposition_task_worker_id=(provider_disposition_task_worker_id),
                    provider_disposition_task_handoff_id=(provider_disposition_task_handoff_id),
                    interrupt_for_manual_tool_recovery=(interrupt_for_manual_tool_recovery),
                )
                if (
                    registered_environment is not None
                    and not pending_allocations
                    and await self._environment_lifecycle.release_deferred_materialization(
                        session=claim.session,
                        registered_agent=registered_agent,
                        registered_environment=registered_environment,
                    )
                ):
                    # A deferred container left by a crash holds nothing the
                    # runtime must publish; the session materializes anew.
                    recovered = recovered.model_copy(
                        update={
                            "actions": (
                                IncompleteSessionRecoveryAction.REAPED_ALLOCATION,
                                *recovered.actions,
                            ),
                        }
                    )
                if pending_allocations:
                    recovered = recovered.model_copy(
                        update={
                            "actions": (
                                IncompleteSessionRecoveryAction.REAPED_ALLOCATION,
                                *(
                                    action
                                    for action in recovered.actions
                                    if action
                                    is not IncompleteSessionRecoveryAction.SKIPPED_TERMINAL
                                ),
                            ),
                        }
                    )
                if not lifecycle_events:
                    return recovered
                retained_actions = tuple(
                    action
                    for action in recovered.actions
                    if action is not IncompleteSessionRecoveryAction.SKIPPED_TERMINAL
                )
                return recovered.model_copy(
                    update={
                        "actions": (
                            IncompleteSessionRecoveryAction.RECONCILED_ENVIRONMENT_LIFECYCLE,
                            *retained_actions,
                        ),
                        "events": (*lifecycle_events, *recovered.events),
                        "message": (
                            "Reconciled orphaned environment lifecycle progress. "
                            + recovered.message
                        ),
                    },
                    deep=True,
                )

            return await self._recovery_ownership.run_with_heartbeat(
                claim=claim,
                recovery=recover_claimed_session,
            )
        except BaseException as exc:
            authoritative_failure = exc
            if command_recovery_context is not None:
                try:
                    await self._environment_lifecycle.abort_environment_setup(
                        session_id=session.id,
                        original_error=exc,
                        execution_profile=command_recovery_context.profile,
                        invocation_context=command_recovery_context,
                    )
                except BaseException as cleanup_error:
                    authoritative_failure = cleanup_error
                    raise
            raise
        finally:
            if claim is not None:
                await self._recovery_ownership.cleanup_claim(
                    authority=claim.require_authority(),
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
                    invocation_context=invocation_context,
                    retain_open_interaction_invocation=(retain_open_interaction_invocation),
                    retain_invocation_context=retain_invocation_context,
                )

    async def _settle_terminal_invocation_closure_owned(
        self,
        *,
        session: Session,
        inactive_for_seconds: int | None,
        previous_status: SessionStatus,
        execution_profile_snapshot: ActiveInvocationExecutionProfile,
    ) -> IncompleteSessionRecoveryResult:
        """Fence a dead terminal owner whose hooks never released its run epoch."""

        claim: _IncompleteRecoveryClaim | None = None
        authoritative_failure: BaseException | None = None
        try:
            claim = await self._recovery_ownership.claim(
                session=session,
                inactive_for_seconds=inactive_for_seconds,
                execution_profile_snapshot=execution_profile_snapshot,
            )
            if claim is None:
                current = await self._recovery_ownership.require_session(session.id)
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
                await self._recovery_ownership.cleanup_claim(
                    authority=claim.require_authority(),
                    authoritative_failure=authoritative_failure,
                )

    async def _require_governed_completion_task(
        self,
        *,
        session: Session,
        marker: dict[str, Any],
        invocation_context: InvocationContext,
    ) -> Task:
        """Validate the exact task before governed workspace recovery effects."""
        if (
            type(invocation_context) is not InvocationContext
            or invocation_context.work_attempt is None
        ):
            raise TypeError("Governed completion recovery requires its invocation authority.")
        admission = invocation_context.work_attempt.admission
        if (
            admission.execution_entry is None
            or marker.get("task_id") != admission.task_id
            or session.id != admission.session_id
            or session.instance_id != admission.session_invocation.session_instance_id
            or invocation_context.binding.session_id != session.id
            or invocation_context.binding.session_instance_id != session.instance_id
            or invocation_context.profile.fingerprint
            != admission.source_execution_profile_fingerprint
        ):
            raise RuntimeError("Governed completion recovery has conflicting task authority.")
        if self._task_store is None:
            raise RuntimeError("Governed completion recovery requires its task store.")
        raw_task = await self._task_store.load_task(admission.task_id)
        if type(raw_task) is not Task:
            raise RuntimeError("Governed completion recovery lost its exact running task.")
        task = copy_task(raw_task)
        if (
            task.id != admission.task_id
            or task.work_contract != admission.contract
            or task.session_id != admission.session_id
            or task.session_instance_id != admission.session_invocation.session_instance_id
            or task.status is not TaskStatus.RUNNING
        ):
            raise RuntimeError("Governed completion recovery lost its exact running task.")
        return task

    async def _record_governed_completion_stop(self, context: InvocationContext) -> None:
        """Retain the stop before the workspace marker or its reply can disappear."""
        if context.work_attempt is None or self._task_store is None:
            raise RuntimeError("Governed completion stop has no attempt owner.")
        await record_work_attempt_execution_stop(
            self._task_store,
            admission=context.work_attempt.admission,
            reason="workspace_finalization_recovery",
            redactor=self._secret_redactor,
        )

    async def _settle_recovered_completion_task(
        self,
        *,
        session: Session,
        marker: dict[str, Any],
        registered_agent: runtime_records.RegisteredAgentState,
        registered_environment: runtime_records.RegisteredEnvironment,
        invocation_context: InvocationContext | None = None,
    ) -> Event | None:
        """Settle ordinary tasks; governed tasks retain their verified owner."""

        if invocation_context is not None and invocation_context.work_attempt is not None:
            await self._require_governed_completion_task(
                session=session, marker=marker, invocation_context=invocation_context
            )
            return None

        marker_has_task_identity = "task_id" in marker
        marker_task_id = marker.get("task_id")
        if self._task_store is None:
            if marker_has_task_identity and marker_task_id is not None:
                raise RuntimeError(
                    "Completion finalization recovery requires its durable task store."
                )
            return None
        if not marker_has_task_identity:
            tasks = await self._task_store.list_tasks(
                TaskQuery(
                    status=TaskStatus.RUNNING,
                    session_id=session.id,
                    limit=2,
                )
            )
            if len(tasks) > 1:
                raise RuntimeError(
                    "Legacy completion finalization marker has multiple running attached tasks."
                )
            if not tasks:
                return None
            task = tasks[0]
        else:
            if marker_task_id is None:
                return None
            if type(marker_task_id) is not str:
                raise RuntimeError("Completion finalization marker has an invalid task identity.")
            task = await self._task_store.load_task(marker_task_id)
            if task is None:
                raise RuntimeError("Completion finalization task is missing from durable storage.")
        if task.session_instance_id != session.instance_id:
            raise RuntimeError(
                "Completion finalization task belongs to a different session incarnation."
            )
        if task.status is TaskStatus.CANCELLED:
            return None
        if task.status is TaskStatus.COMPLETED:
            raise RuntimeError(
                "Completion finalization task was published before workspace output committed."
            )

        failure_payload = task_failure_payload_from_diagnostic(
            ExceptionDiagnostic(
                message=(
                    "Workspace output committed during recovery after the original "
                    "completion owner became unavailable."
                ),
                error_type="WorkspaceCompletionFinalizationRecovered",
            ),
            session_id=session.id,
            additional_fields={
                "phase": "workspace_finalize_recovery",
                "workspace_output_committed": True,
            },
        )
        if task.status is TaskStatus.FAILED:
            if task.error != failure_payload:
                return None
        elif task.status is not TaskStatus.RUNNING:
            raise RuntimeError(
                "Completion finalization task is not running or terminal during recovery."
            )
        elif task.worker_id is None:
            replayed = await load_direct_task_failure_replay(
                self._task_store,
                task_id=task.id,
                session_id=session.id,
                session_instance_id=session.instance_id,
                expected_error=failure_payload,
                claimed_terminalization_idempotency_key=(
                    runtime_task_terminalization_idempotency_key(
                        task_id=task.id,
                        session_id=session.id,
                        kind=TaskTerminalKind.FAILED,
                    )
                ),
            )
            task = (
                replayed
                if replayed is not None
                else await self._task_store.fail_task(
                    task.id,
                    failure_payload,
                    worker_id=None,
                )
            )
        else:
            if task.lease_expires_at is None:
                raise RuntimeError(
                    "Claimed completion finalization task lost its lease generation."
                )
            if not self._task_store.supports_attached_task_recovery_terminalization:
                raise RuntimeError(
                    "Task store cannot atomically settle an expired attached task owner."
                )
            task = await self._task_store.recover_attached_task_failure(
                TaskTerminalizationRequest(
                    task_id=task.id,
                    worker_id=task.worker_id,
                    lease_expires_at=task.lease_expires_at,
                    handoff_id=task.interrupted_handoff_id,
                    kind=TaskTerminalKind.FAILED,
                    error=failure_payload,
                    idempotency_key=runtime_task_terminalization_idempotency_key(
                        task_id=task.id,
                        session_id=session.id,
                        kind=TaskTerminalKind.FAILED,
                    ),
                ),
                session_id=session.id,
                session_instance_id=session.instance_id,
            )

        event_id = str(
            uuid5(
                _COMPLETION_FINALIZATION_TASK_EVENT_NAMESPACE,
                f"{session.id}\0{session.instance_id}\0{task.id}\0failed",
            )
        )
        event_template = _recovery_task_event(
            RecoveryTaskEventRequest(
                event_type=EventType.TASK_FAILED,
                task=task,
                session=session,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
            )
        )
        intended = event_with_runtime_generated_id(
            event_template.model_copy(
                update={
                    "id": event_id,
                    "timestamp": task.completed_at,
                    "payload": {
                        **event_template.payload,
                        "failure_phase": "workspace_finalize_recovery",
                        "workspace_output_committed": True,
                    },
                },
                deep=True,
            )
        )
        persisted = await self._event_writer.persist_exact_replay(intended)
        return (await self._event_writer.fan_out_persisted([persisted]))[0]

    async def _recover_pending_completion_finalization(
        self,
        *,
        session: Session,
        session_before_fence: Session,
        previous_status: SessionStatus,
        claim_id: str,
        marker: dict[str, Any],
        registered_agent: runtime_records.RegisteredAgentState,
        registered_environment: runtime_records.RegisteredEnvironment,
        execution_profile: ExecutionProfileIdentity,
        invocation_context: InvocationContext,
        retain_cleanup_context: Callable[[InvocationContext], None],
    ) -> IncompleteSessionRecoveryResult:
        """Reconnect and retry finalization without changing its terminal outcome."""

        completing = marker["outcome"] == "completed"
        terminal_status = (
            SessionStatus.INTERRUPTED
            if marker["outcome"] == "interrupted"
            else SessionStatus.FAILED
        )
        terminal_event_type = {
            "completed": EventType.SESSION_COMPLETED,
            "failed": EventType.SESSION_FAILED,
            "interrupted": EventType.SESSION_INTERRUPTED,
        }[marker["outcome"]]
        if completing and invocation_context.work_attempt is not None:
            await self._require_governed_completion_task(
                session=session, marker=marker, invocation_context=invocation_context
            )
            await self._record_governed_completion_stop(invocation_context)

        if session.status is SessionStatus.RUNNING:
            recovery_claim_id = invocation_context.recovery_claim_id
            if recovery_claim_id is None:
                raise RuntimeError(
                    "Running completion finalization recovery lost its durable claim."
                )

            def fail_pending_completion(
                current_session: Session,
                checkpoint: dict[str, Any] | None,
                store_now: datetime,
            ) -> dict[str, Any]:
                if (
                    current_session.instance_id != session.instance_id
                    or current_session.run_epoch != session.run_epoch
                ):
                    raise SessionRunFenced(
                        "Completion finalization recovery lost its session authority."
                    )
                claim = _incomplete_recovery_claim_from_checkpoint(checkpoint)
                if (
                    claim is None
                    or claim[0] != recovery_claim_id
                    or claim[1] <= store_now
                    or completion_finalization.pending_completion_finalization_from_checkpoint(
                        checkpoint
                    )
                    != marker
                ):
                    raise SessionRunFenced(
                        "Completion finalization recovery lost its exact durable marker."
                    )
                return copy_durable_record(checkpoint, "checkpoint")

            session = await self._session_store.transition_status_and_checkpoint(
                session.id,
                from_statuses={SessionStatus.RUNNING},
                to_status=terminal_status,
                store_time_checkpoint_transform=fail_pending_completion,
            )
        elif session.status is not terminal_status:
            raise RuntimeError("Pending finalization conflicts with the session terminal outcome.")

        completion_recovery = await self._environment_lifecycle.authorize_completion_recovery(
            session=session, invocation_context=invocation_context, marker=marker
        )
        events: list[Event] = []
        resolved_environment = registered_environment
        resolved_context = invocation_context
        authoritative_error: BaseException | None = None
        try:
            disposal_recovered = await self._environment_lifecycle.recover_completion_disposal(
                session=session,
                registered_agent=registered_agent,
                registered_environment=resolved_environment,
                execution_profile=execution_profile,
                marker=marker,
                completion_recovery=completion_recovery,
                invocation_context=resolved_context,
            )
            if not disposal_recovered:
                factory_started = await self._environment_lifecycle.emit_factory_started(
                    session=session,
                    registered_agent=registered_agent,
                    registered_environment=resolved_environment,
                    execution_profile=execution_profile,
                    invocation_context=resolved_context,
                )
                if factory_started is not None:
                    events.append(factory_started)
                factory_resolution = await self._environment_lifecycle.resolve_factory(
                    session=session,
                    registered_agent=registered_agent,
                    registered_environment=resolved_environment,
                    started_event=factory_started,
                    operation=EnvironmentFactoryOperation.RECONNECT,
                    execution_profile=execution_profile,
                    invocation_context=resolved_context,
                    completion_recovery=completion_recovery,
                )
                events.extend(factory_resolution.events)
                resolved_environment = factory_resolution.registered_environment
                if resolved_environment is None:
                    raise RuntimeError("Completion finalization recovery resolved no environment.")
                resolved_context = resolved_context.with_registered_environment(
                    resolved_environment,
                    validated_profile=execution_profile,
                )
                retain_cleanup_context(resolved_context)
                if factory_resolution.error is not None:
                    raise factory_resolution.error
                binding_started = await self._environment_lifecycle.emit_binding_started(
                    session=session,
                    registered_agent=registered_agent,
                    registered_environment=resolved_environment,
                    execution_profile=execution_profile,
                    invocation_context=resolved_context,
                )
                if binding_started is not None:
                    events.append(binding_started)
                binding_result = await self._environment_lifecycle.bind(
                    session=session,
                    registered_agent=registered_agent,
                    registered_environment=resolved_environment,
                    started_event=binding_started,
                    execution_profile=execution_profile,
                    invocation_context=resolved_context,
                    completion_recovery=completion_recovery,
                )
                events.extend(binding_result.events)
                resolved_environment = binding_result.registered_environment
                if resolved_environment is None:
                    raise RuntimeError(
                        "Completion finalization recovery lost its bound environment."
                    )
                resolved_context = resolved_context.with_registered_environment(
                    resolved_environment,
                    validated_profile=execution_profile,
                )
                retain_cleanup_context(resolved_context)
                if binding_result.error is not None:
                    raise binding_result.error
                finalized = await self._environment_lifecycle.finalize_terminal_event(
                    event=Event(
                        type=terminal_event_type,
                        session_id=session.id,
                        agent_name=registered_agent.spec.name,
                        environment_name=resolved_environment.spec.name,
                        payload={
                            "completion_finalization_recovery": True,
                            **await recorded_terminal_tool_completion_payload(
                                self._session_store, session
                            ),
                        },
                    ),
                    session=session,
                    registered_environment=resolved_environment,
                    execution_profile=execution_profile,
                    invocation_context=resolved_context,
                )
                events.extend(finalized.events)
                finalize_error = finalized.event.payload.get("binding_finalize_error")
                if type(finalize_error) is dict:
                    return IncompleteSessionRecoveryResult(
                        session_id=session.id,
                        previous_status=previous_status,
                        status=session.status,
                        actions=(IncompleteSessionRecoveryAction.FAILED,),
                        events=tuple(events),
                        message=(
                            "Workspace finalization remains pending; the terminal "
                            "session was not re-executed."
                        ),
                    )
            if completing:
                task_failed_event = await self._settle_recovered_completion_task(
                    session=session,
                    marker=marker,
                    registered_agent=registered_agent,
                    registered_environment=resolved_environment,
                    invocation_context=resolved_context,
                )
                if task_failed_event is not None:
                    events.append(task_failed_event)
            terminal_repair = await self.terminal_finalization.repair(
                session=session,
                terminal_run_epoch=session_before_fence.run_epoch,
                terminal_timestamp=session_before_fence.updated_at,
                previous_status=previous_status,
                claim_id=claim_id,
            )
            events.extend(terminal_repair.events)
            await self._environment_lifecycle.clear_completion_finalization(
                session_id=session.id,
                expected_marker=marker,
            )
            await self._environment_lifecycle.abort_environment_setup(
                session_id=session.id,
                original_error=None,
                allow_deferred_settlement=True,
                execution_profile=execution_profile,
                invocation_context=resolved_context,
            )
        except BaseException as exc:
            authoritative_error = exc
            raise
        finally:
            if authoritative_error is not None:
                try:
                    await self._environment_lifecycle.abort_environment_setup(
                        session_id=session.id,
                        original_error=authoritative_error,
                        execution_profile=execution_profile,
                        invocation_context=resolved_context,
                    )
                except BaseException as cleanup_error:
                    if cleanup_error is not authoritative_error:
                        raise BaseExceptionGroup(
                            "Completion finalization recovery and cleanup failed.",
                            [authoritative_error, cleanup_error],
                        ) from cleanup_error
        return IncompleteSessionRecoveryResult(
            session_id=session.id,
            previous_status=previous_status,
            status=session.status,
            actions=(IncompleteSessionRecoveryAction.REPAIRED_WORKSPACE_FINALIZATION,),
            events=tuple(events),
            message=(
                "Recovered committed workspace output without re-running model or tool effects."
            ),
        )

    async def _recover_incomplete_session(
        self,
        *,
        retain_cleanup_context: Callable[[InvocationContext], None],
        session: Session,
        session_before_fence: Session,
        previous_status: SessionStatus,
        inactive_for_seconds: int | None,
        reason: str,
        metadata: dict[str, Any],
        registered_agent: runtime_records.RegisteredAgentState,
        registered_provider: runtime_records.RegisteredProvider,
        registered_environment: runtime_records.RegisteredEnvironment | None,
        invocation_context: InvocationContext | None,
        claim_id: str,
        execution_profile_snapshot: ActiveInvocationExecutionProfile | None,
        budget_policy: BudgetPolicy | None,
        provider_disposition_task_id: str | None = None,
        provider_disposition_task_worker_id: str | None = None,
        provider_disposition_task_handoff_id: str | None = None,
        participant_context: CollaborationAccessContext | None = None,
        interrupt_for_manual_tool_recovery: bool = False,
        preserve_interaction_id: str | None = None,
    ) -> IncompleteSessionRecoveryResult:
        if (execution_profile_snapshot is None) != (invocation_context is None):
            raise RuntimeError(
                "Incomplete recovery requires profile authority and its context together."
            )
        if invocation_context is not None:
            if execution_profile_snapshot is None:
                raise RuntimeError("Incomplete recovery lost its execution-profile authority.")
            if (
                invocation_context.binding.session_id != session.id
                or invocation_context.binding.session_instance_id != session.instance_id
                or invocation_context.binding.run_epoch != session.run_epoch
                or invocation_context.profile is not execution_profile_snapshot.profile
            ):
                raise RuntimeError("Incomplete recovery lost its reconstructed invocation context.")
        actions: list[IncompleteSessionRecoveryAction] = []
        events: list[Event] = []
        checkpoint = await self._session_store.load_checkpoint(session.id)
        pending_completion_finalization = (
            completion_finalization.pending_completion_finalization_from_checkpoint(checkpoint)
        )
        provider_interrupt_payload = _provider_cancellation_interrupt_payload(checkpoint)
        pending_approval = pending_approval_reader.pending_approval_from_checkpoint(
            checkpoint,
            redactor=self._secret_redactor,
            consume_on_rejection=True,
        )
        pending_user_input, _resolution_intent = user_input_lifecycle_authority_from_checkpoint(
            checkpoint,
            redactor=self._secret_redactor,
            consume_on_rejection=True,
            current_run_epoch=session.run_epoch,
            runtime_session=session,
        )
        pending_tool_round = pending_round_reader.pending_tool_round_from_checkpoint(
            checkpoint,
            redactor=self._secret_redactor,
            consume_on_rejection=True,
            runtime_session=session,
        )
        environment_name = _environment_name(registered_environment)

        if pending_completion_finalization is not None:
            if (
                registered_environment is None
                or execution_profile_snapshot is None
                or invocation_context is None
            ):
                raise RuntimeError("Pending completion finalization lost recovery authority.")
            if any(
                item is not None
                for item in (
                    provider_interrupt_payload,
                    pending_approval,
                    pending_user_input,
                    pending_tool_round,
                )
            ):
                raise RuntimeError(
                    "Pending completion finalization conflicts with other recovery work."
                )
            return await self._recover_pending_completion_finalization(
                retain_cleanup_context=retain_cleanup_context,
                session=session,
                session_before_fence=session_before_fence,
                previous_status=previous_status,
                claim_id=claim_id,
                marker=pending_completion_finalization,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
                execution_profile=execution_profile_snapshot.profile,
                invocation_context=invocation_context,
            )

        if interrupt_for_manual_tool_recovery:
            if (
                pending_tool_round is None
                or pending_approval is not None
                or pending_user_input is not None
                or provider_interrupt_payload is not None
            ):
                raise RuntimeError(
                    "Manual tool-recovery handoff requires one pending ordinary tool round."
                )
            if session.status in {SessionStatus.PENDING, SessionStatus.RUNNING}:
                interrupt_payload = {
                    **pending_rounds.pending_tool_round_identity(pending_tool_round).payload(),
                    "reason": reason,
                    "metadata": metadata,
                    "interruption_type": _INTERRUPTION_TYPE_RUNTIME_INTERRUPTED,
                    "recovered": True,
                    "manual_recovery_required": True,
                    "interruption_request_id": str(uuid4()),
                }
                session = await self._session_store.transition_status_and_checkpoint(
                    session.id,
                    from_statuses={session.status},
                    to_status=SessionStatus.INTERRUPTING,
                    checkpoint_transform=_checkpoint_with_pending_session_interrupt(
                        interrupt_payload, cascade_created_at=self._clock()
                    ),
                )
            if session.status is SessionStatus.INTERRUPTING:
                session = await self._finalize_interrupting_for_recovery(
                    recovery_claim_id=claim_id,
                    preserve_interaction_id=preserve_interaction_id,
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
                    invocation_context=invocation_context,
                )
            if session.status is not SessionStatus.INTERRUPTED:
                raise RuntimeError(
                    "Manual tool-recovery handoff did not reach an interrupted session."
                )
            return IncompleteSessionRecoveryResult(
                session_id=session.id,
                previous_status=previous_status,
                status=session.status,
                actions=(IncompleteSessionRecoveryAction.INTERRUPTED_ABANDONED,),
                events=tuple(events),
                message="Fenced the stale run and retained its pending manual tool recovery.",
            )

        if provider_interrupt_payload is not None:
            if session.status is SessionStatus.INTERRUPTED:
                return await self.terminal_finalization.repair(
                    session=session,
                    terminal_run_epoch=session_before_fence.run_epoch,
                    terminal_timestamp=session_before_fence.updated_at,
                    previous_status=previous_status,
                    claim_id=claim_id,
                )
            if session.status in {SessionStatus.PENDING, SessionStatus.RUNNING}:
                session = await self._session_store.transition_status_and_checkpoint(
                    session.id,
                    from_statuses={session.status},
                    to_status=SessionStatus.INTERRUPTING,
                    checkpoint_transform=_checkpoint_with_pending_session_interrupt(
                        provider_interrupt_payload, cascade_created_at=self._clock()
                    ),
                )
            elif session.status is not SessionStatus.INTERRUPTING:
                raise RuntimeError(
                    "Provider cancellation interruption marker conflicts with session status."
                )
            session = await self._finalize_interrupting_for_recovery(
                recovery_claim_id=claim_id,
                preserve_interaction_id=preserve_interaction_id,
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
                invocation_context=invocation_context,
            )
            return IncompleteSessionRecoveryResult(
                session_id=session.id,
                previous_status=previous_status,
                status=session.status,
                actions=(IncompleteSessionRecoveryAction.INTERRUPTED_ABANDONED,),
                events=tuple(events),
                message="Finalized a durable provider cancellation interruption.",
            )

        pending_provider_resolution = await load_pending_provider_operation_disposition(
            self._session_store,
            session.id,
            checkpoint=checkpoint,
        )
        if pending_provider_resolution is not None:
            pending_disposition, resolution_result = pending_provider_resolution
            if provider_disposition_task_id is not None:
                source_stage = await self._session_store.load_model_completion_stage(
                    session.id,
                    pending_disposition.stage_id,
                )
                recovery_context = (
                    None
                    if source_stage is None
                    else model_completion_recovery_context_from_stage(source_stage)
                )
                if (
                    recovery_context is None
                    or recovery_context.task_id != provider_disposition_task_id
                    or not pending_disposition.execution_claimed
                    or pending_disposition.execution_task_worker_id
                    != provider_disposition_task_worker_id
                    or pending_disposition.execution_task_handoff_id
                    != provider_disposition_task_handoff_id
                ):
                    raise ProviderOperationEvidenceError(
                        "Typed provider recovery conflicts with its transferred authority."
                    )
            if await self._provider_disposition.retire_completed_provider_operation_disposition(
                pending=pending_disposition,
                result=resolution_result,
            ):
                actions.append(
                    IncompleteSessionRecoveryAction.REPAIRED_PROVIDER_OPERATION_RESOLUTION
                )
                checkpoint = await self._session_store.load_checkpoint(session.id)
            elif pending_disposition.action is ProviderOperationResolutionAction.FAIL:
                if provider_disposition_task_id is None:
                    (
                        task_id,
                        requires_typed_continuation,
                    ) = await self._automatic_provider_disposition_task_context(pending_disposition)
                    task_worker_id = None
                else:
                    task_id = provider_disposition_task_id
                    task_worker_id = provider_disposition_task_worker_id
                    requires_typed_continuation = False
                if requires_typed_continuation:
                    return IncompleteSessionRecoveryResult(
                        session_id=session.id,
                        previous_status=previous_status,
                        status=session.status,
                        actions=(IncompleteSessionRecoveryAction.SKIPPED_ACTIVE,),
                        events=tuple(events),
                        message=(
                            "Accepted provider-operation failure awaits its elected "
                            "attached-task continuation."
                        ),
                    )
                if execution_profile_snapshot is None:
                    raise RuntimeError(
                        "Provider-operation failure recovery has no execution profile."
                    )
                async for event in self._session_finalization.fail_recovered_provider_operation(
                    ProviderOperationFailureRequest(
                        resolution_event=resolution_result.event,
                        session=session,
                        registered_agent=registered_agent,
                        registered_environment=registered_environment,
                        execution_profile=execution_profile_snapshot.profile,
                        task_id=task_id,
                        task_worker_id=task_worker_id,
                        task_handoff_id=provider_disposition_task_handoff_id,
                        legacy_resolution_without_profile=(
                            resolution_result.record.execution_profile_fingerprint is None
                        ),
                        invocation_context=invocation_context,
                    )
                ):
                    events.append(event)
                failed_session = await self._recovery_ownership.require_session(
                    pending_disposition.session_id
                )
                terminal_event_id = provider_operation_resolution_outcome_event_id(
                    resolution_result.record.resolution_id,
                    "session_failed",
                )
                terminal_records = await self._session_store.query_events(
                    EventQuery(
                        session_id=pending_disposition.session_id,
                        event_id=terminal_event_id,
                        limit=2,
                    )
                )
                if len(terminal_records) != 1:
                    raise ProviderOperationEvidenceError(
                        "Provider-operation failure has incomplete terminal evidence."
                    )
                terminal_hook_authority = RecoveryTerminalEventRequest(
                    event=copy_event(terminal_records[0].event),
                    phase=RuntimeHookPhase.AFTER_SESSION_FAILED,
                    session=failed_session,
                    registered_agent=registered_agent,
                    registered_environment=registered_environment,
                    execution_profile=execution_profile_snapshot.profile,
                    invocation_context=invocation_context,
                    terminal_event_already_durable=True,
                    yield_durable_terminal_event=False,
                )
                if not await self._provider_disposition.retire_completed_provider_operation_disposition(
                    pending=pending_disposition,
                    result=resolution_result,
                    terminal_hook_authority=terminal_hook_authority,
                ):
                    raise RuntimeError(
                        "Recovered provider-operation failure has no terminal outcome."
                    )
                current = await self._recovery_ownership.require_session(session.id)
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
                if provider_disposition_task_id is None:
                    (
                        _task_id,
                        requires_typed_continuation,
                    ) = await self._automatic_provider_disposition_task_context(pending_disposition)
                    task_worker_id = None
                else:
                    _task_id = provider_disposition_task_id
                    task_worker_id = provider_disposition_task_worker_id
                    requires_typed_continuation = False
                if requires_typed_continuation:
                    return IncompleteSessionRecoveryResult(
                        session_id=session.id,
                        previous_status=previous_status,
                        status=session.status,
                        actions=(IncompleteSessionRecoveryAction.SKIPPED_ACTIVE,),
                        events=tuple(events),
                        message=(
                            "Accepted provider-operation fallback awaits its elected "
                            "attached-task continuation."
                        ),
                    )
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
                interaction_id = await self._recovery_admission.activate_latest_open_interaction(
                    session.id
                )
                if interaction_id is None:
                    raise RuntimeError(
                        "Provider-operation fallback recovery has no open interaction."
                    )
                async for (
                    event
                ) in self._provider_disposition.run_pending_provider_operation_fallback(
                    pending=pending_disposition,
                    participant_context=participant_context,
                    result=resolution_result,
                    session=session,
                    registered_agent=registered_agent,
                    registered_provider=registered_provider,
                    registered_environment=registered_environment,
                    execution_profile_snapshot=execution_profile_snapshot,
                    recovery_context=recovery_context,
                    budget_policy=budget_policy,
                    release_run_fence_on_cleanup=False,
                    task_worker_id=task_worker_id,
                    task_handoff_id=provider_disposition_task_handoff_id,
                    invocation_context=invocation_context,
                ):
                    events.append(event)
                current = await self._recovery_ownership.require_session(session.id)
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
            await self.terminal_finalization.repair_required(
                session=session,
                checkpoint=checkpoint,
            )
        ):
            repaired = await self.terminal_finalization.repair(
                session=session,
                terminal_run_epoch=session_before_fence.run_epoch,
                terminal_timestamp=session_before_fence.updated_at,
                previous_status=previous_status,
                claim_id=claim_id,
            )
            actions.extend(repaired.actions)
            events.extend(repaired.events)
            session = await self._recovery_ownership.require_session(session.id)
            checkpoint = await self._session_store.load_checkpoint(session.id)

            if settled_invocation_terminal_decision_from_checkpoint(checkpoint) is not None:
                # Terminal repair already authenticated and published the exact
                # winner. Its retired human gate is not missing executable work:
                # do not re-enter model reconciliation after repairing closure.
                return IncompleteSessionRecoveryResult(
                    session_id=session.id,
                    previous_status=previous_status,
                    status=session.status,
                    actions=tuple(actions),
                    events=tuple(events),
                    message="Recovered the committed terminal decision without redispatch.",
                )

        if inactive_for_seconds is not None:
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
                            "inactive_for_seconds": inactive_for_seconds,
                            "reason": reason,
                            "metadata": metadata,
                        },
                    )
                )
            )

        provider_operation_addressed = False
        if session.status is SessionStatus.INTERRUPTING:
            provider_operation_addressed = (
                await self._model_completion_recovery.cancel_provider_operation_for_interruption(
                    session,
                    registered_agent=registered_agent,
                    registered_provider=registered_provider,
                    registered_environment=registered_environment,
                    invocation_context=invocation_context,
                )
                is not None
            )
            session = await self._recovery_ownership.require_session(session.id)
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
            model_boundary = (
                await self._model_completion_recovery.reconcile_model_completion_boundary(
                    session,
                    invocation_context=invocation_context,
                    registered_agent=registered_agent,
                    registered_provider=registered_provider,
                    registered_environment=registered_environment,
                )
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
            interrupted_event = event_with_runtime_generated_id(
                Event(
                    id=str(
                        uuid5(
                            _PROVIDER_OPERATION_UNAVAILABLE_INTERRUPT_NAMESPACE,
                            f"{session.id}\0{active_model_stage.stage.stage_id}\0"
                            f"{unavailable_event.id}",
                        )
                    ),
                    type=EventType.SESSION_INTERRUPTED,
                    session_id=session.id,
                    agent_name=session.agent_name,
                    environment_name=environment_name,
                    timestamp=unavailable_event.timestamp,
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
            persisted_interrupted = await self._event_writer.persist_exact_replay(interrupted_event)
            [interrupted_event] = await self._event_writer.fan_out_persisted(
                [persisted_interrupted]
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
        pending_approval = pending_approval_reader.pending_approval_from_checkpoint(checkpoint)
        pending_user_input, _resolution_intent = user_input_lifecycle_authority_from_checkpoint(
            checkpoint,
            redactor=self._secret_redactor,
            consume_on_rejection=True,
            current_run_epoch=session.run_epoch,
            runtime_session=session,
        )
        pending_tool_round = pending_round_reader.pending_tool_round_from_checkpoint(checkpoint)
        if pending_user_input is not None:
            pause_state = await self._user_input_evidence.classify_pause(
                session=session,
                checkpoint=checkpoint,
                input_id=pending_user_input.input_id,
            )
            if pause_state not in {
                UserInputPauseState.ACTIVE,
                UserInputPauseState.ANSWERING,
            }:
                raise SessionRuntimePublicationConflict(
                    "Pending user-input recovery authority is ambiguous."
                )
        if pending_tool_round is None and await self._deferred_input.materialize_if_present(
            session.id
        ):
            actions.append(IncompleteSessionRecoveryAction.REPAIRED_TOOL_ROUND)

        if (
            session.status is SessionStatus.RUNNING
            and invocation_context is not None
            and invocation_context.work_attempt is None
            and pending_tool_round is None
            and pending_approval is None
            and pending_user_input is None
        ):
            from cayu.runtime._producer_completion_replay import prepare_producer_completion_replay

            producer_replay = await prepare_producer_completion_replay(
                self._session_store, session, checkpoint, invocation_context, model_boundary
            )
            if producer_replay is not None:
                stage = model_boundary.completed_stage
                assert stage is not None
                semantics = model_completion_recovery_context_from_stage(stage)
                if (
                    semantics is None
                    or semantics.execution_profile_fingerprint
                    != invocation_context.profile.fingerprint
                    or semantics.interaction_id != invocation_context.binding.interaction_id
                    or semantics.task_id is not None
                ):
                    raise ModelCompletionManualRecoveryRequired(
                        "Producer completion lacks its original native run semantics."
                    )
                stream = self._engine.continue_run(
                    RecoverySessionRunRequest(
                        session=session,
                        invocation_context=invocation_context,
                        participant_context=participant_context,
                        messages=await self._session_store.load_transcript(session.id),
                        messages_to_append=[],
                        max_steps=semantics.max_steps,
                        limits=semantics.limits,
                        budget_limits=semantics.budget_limits,
                        retry_policy=semantics.retry_policy,
                        structured_output=semantics.structured_output,
                        tool_completion=semantics.tool_completion,
                        thinking=semantics.thinking,
                        request_metadata=semantics.request_metadata,
                        task_id=None,
                        task_worker_id=None,
                        task_handoff_id=None,
                        start_event_type=None,
                        start_event_payload={},
                        start_task_on_enter=False,
                        release_run_fence_on_exit=False,
                        run_limit_accounting=semantics.run_limit_accounting,
                        producer_replay=producer_replay,
                    )
                )
                async with _close_delegated_event_stream(stream) as owned:
                    async for event in owned:
                        events.append(event)
                current = await self._recovery_ownership.require_session(session.id)
                return IncompleteSessionRecoveryResult(
                    session_id=session.id,
                    previous_status=previous_status,
                    status=current.status,
                    actions=(
                        *actions,
                        IncompleteSessionRecoveryAction.REPAIRED_TERMINAL_EVIDENCE
                        if current.status is SessionStatus.COMPLETED
                        else IncompleteSessionRecoveryAction.INTERRUPTED_ABANDONED
                        if current.status is SessionStatus.INTERRUPTED
                        else IncompleteSessionRecoveryAction.FAILED,
                    ),
                    events=tuple(events),
                    message="Replayed the retained producer model result without provider redispatch.",
                )

        if (
            session.status is SessionStatus.RUNNING
            and invocation_context is not None
            and invocation_context.work_attempt is not None
            and preserve_interaction_id == invocation_context.binding.interaction_id
            and pending_tool_round is not None
            and pending_approval is None
            and pending_user_input is None
            and has_recoverable_structured_output_round(pending_tool_round)
        ):
            # Replacement preserves this governed interaction. Reconcile its
            # reserved validation before interrupting the predecessor epoch so
            # that interruption cannot consume the original repair allowance.
            # This publishes validation evidence; only the replacement model
            # loop may dispatch a later repair under its execution claim.
            snapshot = await self._session_store.load_transcript_snapshot(session.id)
            transcript = [detach_message(record.message) for record in snapshot.records]
            async for event in self._pending_tool_round_recovery.recover_pending_tool_round(
                session=session,
                invocation_context=invocation_context,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
                messages=transcript,
                execution_profile=invocation_context.profile,
                incomplete_recovery_claimed=True,
                expected_transcript_cursor=snapshot.cursor,
            ):
                events.append(event)
            checkpoint, pending_tool_round = await pending_round_reader.load_pending_tool_round(
                self._session_store,
                session.id,
            )
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

        if (
            session.status is SessionStatus.RUNNING
            and invocation_context is not None
            and invocation_context.work_attempt is None
            and pending_approval is None
            and pending_user_input is None
            and model_boundary.completed_stage is not None
        ):
            semantics = model_completion_recovery_context_from_stage(model_boundary.completed_stage)
            if semantics is not None and semantics.tool_completion is not None:
                policy = await load_recorded_tool_completion_policy(
                    self._session_store,
                    session,
                    checkpoint,
                    execution_profile=invocation_context.profile,
                    max_steps=semantics.max_steps,
                    limits=semantics.limits,
                    retry_policy=semantics.retry_policy,
                    context=semantics,
                )
                if (
                    policy is not None
                    and pending_tool_round is not None
                    and await pending_round_has_completion_success(
                        self._session_store,
                        session,
                        pending_tool_round,
                        policy=policy,
                        interaction_id=invocation_context.binding.interaction_id,
                    )
                ):
                    snapshot = await self._session_store.load_transcript_snapshot(session.id)
                    async for event in self._pending_tool_round_recovery.recover_pending_tool_round(
                        session=session,
                        invocation_context=invocation_context,
                        registered_agent=registered_agent,
                        registered_environment=registered_environment,
                        messages=[detach_message(record.message) for record in snapshot.records],
                        execution_profile=invocation_context.profile,
                        incomplete_recovery_claimed=True,
                        expected_transcript_cursor=snapshot.cursor,
                    ):
                        events.append(event)
                    actions.append(IncompleteSessionRecoveryAction.REPAIRED_TOOL_ROUND)
                result = await recorded_tool_completion_result(
                    self._session_store,
                    session,
                    policy=policy,
                    execution_profile=invocation_context.profile,
                    registered_agent=registered_agent,
                )
                if result is not None:
                    stream = self._engine.continue_run(
                        RecoverySessionRunRequest(
                            session=session,
                            invocation_context=invocation_context,
                            participant_context=participant_context,
                            messages=await self._session_store.load_transcript(session.id),
                            messages_to_append=[],
                            max_steps=semantics.max_steps,
                            limits=semantics.limits,
                            budget_limits=semantics.budget_limits,
                            retry_policy=semantics.retry_policy,
                            structured_output=semantics.structured_output,
                            tool_completion=policy,
                            tool_completion_replay=result,
                            thinking=semantics.thinking,
                            request_metadata=semantics.request_metadata,
                            task_id=semantics.task_id,
                            task_worker_id=None,
                            task_handoff_id=None,
                            start_event_type=None,
                            start_event_payload={},
                            start_task_on_enter=False,
                            release_run_fence_on_exit=False,
                            run_limit_accounting=semantics.run_limit_accounting,
                        )
                    )
                    async with _close_delegated_event_stream(stream) as owned:
                        async for event in owned:
                            events.append(event)
                    current = await self._recovery_ownership.require_session(session.id)
                    return IncompleteSessionRecoveryResult(
                        session_id=session.id,
                        previous_status=previous_status,
                        status=current.status,
                        actions=(
                            *actions,
                            IncompleteSessionRecoveryAction.REPAIRED_TERMINAL_EVIDENCE,
                        ),
                        events=tuple(events),
                        message="Completed from the recorded final tool without another tool or provider dispatch.",
                    )

        failed_with_recoverable_tool_round = (
            session.status is SessionStatus.FAILED
            and pending_tool_round is not None
            and pending_approval is None
            and pending_user_input is None
        )
        if session.status in {SessionStatus.PENDING, SessionStatus.RUNNING} or (
            failed_with_recoverable_tool_round
        ):
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
                    **pending_rounds.pending_tool_round_identity(pending_tool_round).payload(),
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
                    checkpoint_transform=_checkpoint_with_pending_session_interrupt(
                        interrupt_payload, cascade_created_at=self._clock()
                    ),
                )
            except ValueError:
                session = await self._recovery_ownership.require_session(session.id)
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
            session = await self._recovery_ownership.require_session(session.id)
            checkpoint = await self._session_store.load_checkpoint(session.id)
            pending_approval = pending_approval_reader.pending_approval_from_checkpoint(
                checkpoint,
                redactor=self._secret_redactor,
                consume_on_rejection=True,
            )
            pending_tool_round = pending_round_reader.pending_tool_round_from_checkpoint(
                checkpoint,
                redactor=self._secret_redactor,
                consume_on_rejection=True,
                runtime_session=session,
            )

        workspace_recovery_events = (
            await self._workspace_observation_recovery.recover_workspace_observations(
                session=session,
                registered_environment=registered_environment,
                execution_profile_snapshot=execution_profile_snapshot,
                invocation_context=invocation_context,
            )
        )
        if workspace_recovery_events:
            events.extend(workspace_recovery_events)
            actions.append(IncompleteSessionRecoveryAction.REPAIRED_WORKSPACE_OBSERVATION)
            checkpoint = await await_workspace_observation_store_read(
                lambda: self._session_store.load_checkpoint(session.id),
                operation="Post-recovery workspace observation checkpoint read",
            )
            pending_tool_round = pending_round_reader.pending_tool_round_from_checkpoint(
                checkpoint,
                redactor=self._secret_redactor,
                consume_on_rejection=True,
                runtime_session=session,
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
                async for event in self._pending_tool_round_recovery.recover_pending_tool_round(
                    session=session,
                    invocation_context=invocation_context,
                    registered_agent=registered_agent,
                    registered_environment=registered_environment,
                    messages=transcript,
                    execution_profile=(
                        None
                        if execution_profile_snapshot is None
                        else execution_profile_snapshot.profile
                    ),
                    incomplete_recovery_claimed=True,
                    # This caller finishes the takeover when replay is elected.
                    elect_replay=True,
                    expected_transcript_cursor=expected_transcript_cursor,
                ):
                    events.append(event)
            except tool_call_replay.ToolCallReplayRequired as replay:
                # The continuation that took over replays the interrupted calls,
                # so finish this owner's interruption with the round still open.
                session = await self._recovery_ownership.require_session(session.id)
                session = await self._finalize_interrupting_for_recovery(
                    recovery_claim_id=claim_id,
                    preserve_interaction_id=preserve_interaction_id,
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
                    invocation_context=invocation_context,
                )
                return IncompleteSessionRecoveryResult(
                    session_id=session.id,
                    previous_status=previous_status,
                    status=session.status,
                    actions=(*actions, IncompleteSessionRecoveryAction.PENDING_TOOL_EFFECT),
                    events=tuple(events),
                    message=str(replay),
                )
            except ToolEffectReconciliationRequired as unresolved:
                # Unknown external effects are a retained recovery pause. Finish
                # this owner's interruption before exposing the next explicit
                # receipt action; do not make that action drain our old marker.
                session = await self._recovery_ownership.require_session(session.id)
                session = await self._finalize_interrupting_for_recovery(
                    recovery_claim_id=claim_id,
                    preserve_interaction_id=preserve_interaction_id,
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
                    invocation_context=invocation_context,
                )
                return IncompleteSessionRecoveryResult(
                    session_id=session.id,
                    previous_status=previous_status,
                    status=session.status,
                    actions=(
                        *actions,
                        (
                            IncompleteSessionRecoveryAction.PENDING_SUBAGENT
                            if isinstance(unresolved, ForegroundSubagentRecoveryRequired)
                            else IncompleteSessionRecoveryAction.PENDING_TOOL_EFFECT
                        ),
                    ),
                    events=tuple(events),
                    pending_subagent_session_ids=(
                        (unresolved.child_session_id,)
                        if isinstance(unresolved, ForegroundSubagentRecoveryRequired)
                        else ()
                    ),
                    message=(
                        str(unresolved)
                        if isinstance(unresolved, ForegroundSubagentRecoveryRequired)
                        else "External effect requires explicit receipt reconciliation or continuation."
                    ),
                )
            except ToolApprovalRequired:
                # Fail-closed planning of an ambiguous crash boundary may
                # atomically restore the human gate. Incomplete-session
                # recovery owns the outer interrupt transition, so retain the
                # paired approval and finish that transition below instead of
                # treating the pause as failure.
                pass
            except tool_round_recovery.UnsafeToolRoundContinuationError:
                if session.status not in _UNREPLAYABLE_TOOL_ROUND_ARCHIVE_SESSION_STATUSES:
                    raise
                await self._archive_unreplayable_tool_round(
                    session=session,
                    pending_round=pending_tool_round,
                )
                session = await self._finalize_interrupting_for_recovery(
                    recovery_claim_id=claim_id,
                    preserve_interaction_id=preserve_interaction_id,
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
                    invocation_context=invocation_context,
                )
                actions.append(IncompleteSessionRecoveryAction.INTERRUPTED_ABANDONED)
                return IncompleteSessionRecoveryResult(
                    session_id=session.id,
                    previous_status=previous_status,
                    status=session.status,
                    actions=tuple(actions),
                    events=tuple(events),
                    message=(
                        "Archived an abandoned tool round whose opaque provider state "
                        "could not be replayed safely."
                    ),
                )
            actions.append(IncompleteSessionRecoveryAction.REPAIRED_TOOL_ROUND)
            session = await self._recovery_ownership.require_session(session.id)
            checkpoint = await self._session_store.load_checkpoint(session.id)

        pending_approval = pending_approval_reader.pending_approval_from_checkpoint(
            checkpoint,
            redactor=self._secret_redactor,
            consume_on_rejection=True,
        )
        if pending_approval is not None:
            # Approval recovery deliberately skips ordinary/native round
            # recovery, but a consumed external call still needs the same
            # durable uncertainty transition after worker loss. Keep the
            # human gate and selected terminals intact; never dispatch here.
            events.extend(
                await settle_prepared_tool_effects(
                    store=self._session_store,
                    writer=self._event_writer,
                    session=session,
                    pending=pending_approval,
                    profile=None
                    if execution_profile_snapshot is None
                    else execution_profile_snapshot.profile,
                )
            )
            unresolved_effect = await ToolEffectStateOwner(self._session_store).preserve_unresolved(
                session,
                tool_round_id=pending_approval.tool_round_id,
                tool_call_ids=tuple(call.tool_call_id for call in pending_approval.tool_calls),
            )
            pending_child_ids: tuple[str, ...] = ()
            if unresolved_effect:
                events.extend(
                    await self._pending_tool_round_recovery.deliver_pending_tool_effect_uncertainty(
                        session
                    )
                )
                actions.append(IncompleteSessionRecoveryAction.PENDING_TOOL_EFFECT)
                pending_child_ids = await self._pending_foreground_children_at_human_gate(
                    session=session,
                    checkpoint=checkpoint,
                    pending=pending_approval,
                    registered_agent=registered_agent,
                )
                if pending_child_ids:
                    actions.append(IncompleteSessionRecoveryAction.PENDING_SUBAGENT)
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
                    checkpoint_transform=_checkpoint_with_pending_session_interrupt(
                        interrupt_payload, cascade_created_at=self._clock()
                    ),
                )
            session = await self._finalize_interrupting_for_recovery(
                recovery_claim_id=claim_id,
                preserve_interaction_id=preserve_interaction_id,
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
                invocation_context=invocation_context,
            )
            actions.append(IncompleteSessionRecoveryAction.PENDING_APPROVAL)
            return IncompleteSessionRecoveryResult(
                session_id=session.id,
                previous_status=previous_status,
                status=session.status,
                actions=tuple(actions),
                events=tuple(events),
                pending_approval_id=pending_approval.approval_id,
                pending_subagent_session_ids=pending_child_ids,
                message=(
                    "Foreground children require recovery; the pending approval remains intact."
                    if pending_child_ids
                    else "External effect remains unresolved; explicit receipt reconciliation is required."
                    if unresolved_effect
                    else "Session has a pending tool approval; resolve it with ToolApprovalRequest."
                ),
            )

        pending_user_input, _resolution_intent = user_input_lifecycle_authority_from_checkpoint(
            checkpoint,
            redactor=self._secret_redactor,
            consume_on_rejection=True,
            current_run_epoch=session.run_epoch,
            runtime_session=session,
        )
        if pending_user_input is not None:
            pause_state = await self._user_input_evidence.classify_pause(
                session=session,
                checkpoint=checkpoint,
                input_id=pending_user_input.input_id,
            )
            if pause_state not in {
                UserInputPauseState.ACTIVE,
                UserInputPauseState.ANSWERING,
            }:
                raise SessionRuntimePublicationConflict(
                    "Pending user-input recovery authority changed before finalization."
                )
            # Like approval-owned rounds, a user-input pause bypasses ordinary
            # round recovery. Classify dispatched sibling effects without
            # consuming the answer or authorizing another tool invocation.
            events.extend(
                await settle_prepared_tool_effects(
                    store=self._session_store,
                    writer=self._event_writer,
                    session=session,
                    pending=pending_user_input,
                    profile=None
                    if execution_profile_snapshot is None
                    else execution_profile_snapshot.profile,
                )
            )
            unresolved_effect = await ToolEffectStateOwner(self._session_store).preserve_unresolved(
                session,
                tool_round_id=pending_user_input.tool_round_id,
                tool_call_ids=tuple(call.tool_call_id for call in pending_user_input.tool_calls),
            )
            pending_child_ids = ()
            if unresolved_effect:
                events.extend(
                    await self._pending_tool_round_recovery.deliver_pending_tool_effect_uncertainty(
                        session
                    )
                )
                actions.append(IncompleteSessionRecoveryAction.PENDING_TOOL_EFFECT)
                pending_child_ids = await self._pending_foreground_children_at_human_gate(
                    session=session,
                    checkpoint=checkpoint,
                    pending=pending_user_input,
                    registered_agent=registered_agent,
                )
                if pending_child_ids:
                    actions.append(IncompleteSessionRecoveryAction.PENDING_SUBAGENT)
            session = await self._finalize_interrupting_for_recovery(
                recovery_claim_id=claim_id,
                preserve_interaction_id=preserve_interaction_id,
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
                invocation_context=invocation_context,
            )
            actions.append(IncompleteSessionRecoveryAction.PENDING_USER_INPUT)
            return IncompleteSessionRecoveryResult(
                session_id=session.id,
                previous_status=previous_status,
                status=session.status,
                actions=tuple(actions),
                events=tuple(events),
                pending_user_input_id=pending_user_input.input_id,
                pending_subagent_session_ids=pending_child_ids,
                message=(
                    "Foreground children require recovery; the pending user input remains intact."
                    if pending_child_ids
                    else "External effect remains unresolved; explicit receipt reconciliation is required."
                    if unresolved_effect
                    else "Session is awaiting user input; answer it with UserInputResponse."
                ),
            )

        if session.status == SessionStatus.INTERRUPTING:
            session = await self._finalize_interrupting_for_recovery(
                recovery_claim_id=claim_id,
                preserve_interaction_id=preserve_interaction_id,
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
                invocation_context=invocation_context,
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

    async def _archive_unreplayable_tool_round(
        self,
        *,
        session: Session,
        pending_round: pending_rounds.PendingToolRound,
    ) -> None:
        """Retain quarantined evidence while removing it from resumable work."""

        expected_identity = pending_rounds.pending_tool_round_identity(pending_round)
        if session.status not in _UNREPLAYABLE_TOOL_ROUND_ARCHIVE_SESSION_STATUSES:
            raise RuntimeError(
                "An unreplayable tool round can only be abandoned from a recoverable "
                "terminal or interrupting session."
            )

        def archive(
            current_session: Session,
            checkpoint: dict[str, Any] | None,
        ) -> dict[str, Any]:
            if (
                current_session.status is not session.status
                or current_session.run_epoch != session.run_epoch
            ):
                raise RuntimeError(
                    "Session changed before its unreplayable tool round was abandoned."
                )
            current = pending_round_reader.pending_tool_round_from_checkpoint(
                checkpoint,
                redactor=self._secret_redactor,
                consume_on_rejection=True,
                runtime_session=current_session,
            )
            if (
                current is None
                or pending_rounds.pending_tool_round_identity(current) != expected_identity
            ):
                raise RuntimeError(
                    "Pending tool round changed before its unreplayable state was abandoned."
                )
            copied = copy_durable_record(checkpoint, "checkpoint")
            durable_round = copied.pop(pending_rounds.PENDING_TOOL_ROUND_CHECKPOINT_KEY)
            pointer = model_completion_publication.model_step_publication_from_checkpoint(copied)
            if pointer is not None:
                if (
                    not pointer.assistant_message_deferred
                    or pointer.logical_step_id != expected_identity.model_step_id
                    or pointer.tool_round_id != expected_identity.tool_round_id
                ):
                    raise RuntimeError(
                        "Unreplayable tool round conflicts with its durable model-step pointer."
                    )
                copied.pop(model_completion_publication.LAST_MODEL_STEP_PUBLICATION_CHECKPOINT_KEY)
            return _retain_abandoned_unreplayable_tool_round(copied, durable_round)

        await self._session_store.transform_checkpoint(session.id, archive)

    async def _finalize_interrupting_for_recovery(
        self,
        *,
        session: Session,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_environment: runtime_records.RegisteredEnvironment | None,
        environment_name: str | None,
        events: list[Event],
        execution_profile: ExecutionProfileIdentity | None = None,
        invocation_context: InvocationContext | None = None,
        preserve_interaction_id: str | None = None,
        recovery_claim_id: str | None = None,
    ) -> Session:
        if session.status == SessionStatus.INTERRUPTING:
            async for event in self._session_finalization.interrupt_recovery(
                RecoveryInterruptionRequest(
                    recovery_claim_id=recovery_claim_id,
                    preserve_interaction_id=preserve_interaction_id,
                    session=session,
                    registered_agent=registered_agent,
                    registered_environment=registered_environment,
                    environment_name=environment_name,
                    execution_profile=execution_profile,
                    invocation_context=invocation_context,
                )
            ):
                events.append(event)
            session = await self._recovery_ownership.require_session(session.id)
        return session

    async def _pending_durable_subagent_children(
        self,
        *,
        session: Session,
        checkpoint: dict[str, Any] | None,
        pending_round: pending_rounds.PendingToolRound,
        registered_agent: runtime_records.RegisteredAgentState,
    ) -> tuple[Session, ...]:
        """Reconcile durable submissions without closing live child work as unknown."""

        lifecycle_events = await self._pending_tool_round_recovery.load_tool_round_lifecycle_events(
            session_id=session.id,
            pending_round=pending_round,
        )
        recorded_outcomes, _started_ids = tool_round_recovery.recorded_tool_outcomes(
            events=lifecycle_events,
            pending_round=pending_round,
        )
        children = await self._pending_tool_round_recovery.subagent_children_by_idempotency_key(
            session.id
        )
        pending: list[Session] = []
        for call in pending_round.tool_calls:
            if recorded_outcomes.get(call.tool_call_id) is not None:
                continue
            idempotency_key = tool_execution.tool_idempotency_key(
                session_id=session.id,
                tool_round_id=pending_round.tool_round_id,
                tool_call_id=call.tool_call_id,
            )
            recovery_arguments = (
                await self._pending_tool_round_recovery.subagent_recovery_arguments(
                    checkpoint=checkpoint,
                    parent_session=session,
                    tool_name=call.tool_name,
                    tool_round_id=pending_round.tool_round_id,
                    tool_call_id=call.tool_call_id,
                    idempotency_key=idempotency_key,
                    fallback=call.arguments,
                )
            )
            reconciled_result = await self._pending_tool_round_recovery.reconcile_subagent_child(
                children,
                idempotency_key=idempotency_key,
                tool_call_id=call.tool_call_id,
                tool_name=call.tool_name,
                tool_round_id=pending_round.tool_round_id,
                arguments=recovery_arguments,
                parent_session=session,
                registered_agent=registered_agent,
            )
            if reconciled_result is not None:
                # The child recovery owner has confirmed the exact handoff.
                # Its queued child need not finish before the parent can settle;
                # normal round recovery below revalidates and publishes the result.
                continue
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

    async def _pending_foreground_children_at_human_gate(
        self,
        *,
        session: Session,
        checkpoint: dict[str, Any] | None,
        pending: PendingToolApproval | PendingUserInput,
        registered_agent: runtime_records.RegisteredAgentState,
    ) -> tuple[str, ...]:
        """Read exact child obligations without repairing submissions or consuming a gate."""
        children = await self._pending_tool_round_recovery.subagent_children_by_idempotency_key(
            session.id
        )
        selected: list[str] = []
        owner = ToolEffectStateOwner(self._session_store)
        for call in pending.tool_calls:
            effect = await owner.resolve_call(
                session, tool_round_id=pending.tool_round_id, tool_call_id=call.tool_call_id
            )
            if effect is None or effect.state != "outcome_unknown":
                continue
            key = tool_execution.tool_idempotency_key(
                session_id=session.id,
                tool_round_id=pending.tool_round_id,
                tool_call_id=call.tool_call_id,
                approval_id=effect.intent.approval_id,
                pause_id=effect.intent.pause_id,
            )
            if effect.intent.idempotency_key != key or effect.intent.tool_name != call.tool_name:
                raise RuntimeError(
                    "Foreground child effect conflicts with its gated call identity."
                )
            child = children.get(key)
            if child is None:
                continue
            arguments = await self._pending_tool_round_recovery.subagent_recovery_arguments(
                checkpoint=checkpoint,
                parent_session=session,
                tool_name=call.tool_name,
                tool_round_id=pending.tool_round_id,
                tool_call_id=call.tool_call_id,
                idempotency_key=key,
                fallback=call.arguments,
            )
            if not _matches_recoverable_subagent_child(
                child,
                idempotency_key=key,
                tool_call_id=call.tool_call_id,
                tool_name=call.tool_name,
                arguments=arguments,
                parent_session=session,
                registered_agent=registered_agent,
            ):
                continue
            metadata = child.metadata.get("subagent")
            if (
                type(metadata) is dict
                and metadata.get("mode") == "foreground"
                and child.status
                in {SessionStatus.PENDING, SessionStatus.RUNNING, SessionStatus.INTERRUPTING}
            ):
                selected.append(child.id)
        return tuple(selected)

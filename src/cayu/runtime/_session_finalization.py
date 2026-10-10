"""Own session terminal transitions, interruption settlement and run-limit closure."""

from __future__ import annotations

import asyncio
import contextlib
import json
import sys
import time
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, NoReturn, TypeVar
from uuid import uuid4

from pydantic import (
    ValidationError,
)

from cayu._exception_groups import (
    add_exception_note_safely,
    exception_cause,
    exception_group_children,
    iter_exception_tree,
    rebuild_exception_group,
    set_exception_cause,
)
from cayu._exception_state import pop_exception_state, set_exception_state
from cayu._task_wait import (
    await_shielded_task_outcome,
    unexpected_child_cancellation_error,
)
from cayu._validation import (
    copy_durable_json_value,
    copy_durable_record,
    copy_json_value,
    require_clean_nonblank,
)
from cayu.approvals.tools import (
    PendingToolApproval,
    ToolApprovalDecision,
)
from cayu.approvals.user_input import (
    AMBIGUOUS_USER_INPUT_SUPERSESSION_INTENT_KEY,
    PENDING_USER_INPUT_CHECKPOINT_KEY,
    USER_INPUT_RESOLUTION_INTENT_CHECKPOINT_KEY,
    USER_INPUT_SUPERSESSION_INTENT_KEY,
    AmbiguousPendingUserInput,
    AmbiguousUserInputSupersessionIntent,
    PendingUserInput,
    UserInputSupersessionIntent,
    ambiguous_pending_user_input_from_checkpoint,
    ambiguous_user_input_supersession_intent_for,
    event_with_ambiguous_user_input_supersession_authority,
    event_with_user_input_supersession_authority,
    pending_user_input_identity,
    user_input_lifecycle_authority_from_checkpoint,
    user_input_supersession_intent_for,
)
from cayu.budgets.base import (
    BudgetCheck,
    BudgetReservationResult,
    budget_reservation_payload,
)
from cayu.budgets.pricing import (
    SessionCostTotals,
)
from cayu.budgets.usage import (
    SessionUsageSummary,
    aggregate_usage_metrics_payload,
    combine_session_usage_summaries,
    session_usage_summary_payload,
)
from cayu.deadlines import (
    current_execution_deadline,
    effective_deadline,
    expired_execution_deadline,
)
from cayu.events import (
    Event,
    EventType,
    copy_event,
    event_with_runtime_envelope_authority,
    event_with_runtime_generated_id,
    event_with_runtime_nested_payload_authority,
    event_with_runtime_payload_authority,
)
from cayu.exceptions import (
    InteractionLifecyclePublicationRejected,
    _runtime_interaction_lifecycle_publication_rejected,
)
from cayu.execution_profiles import (
    ExecutionProfileIdentity,
    event_with_execution_profile_authority,
)
from cayu.execution_units import (
    ModelAttemptIdentity,
    ToolRoundIdentity,
)
from cayu.failure_evidence import FailureEvidence
from cayu.messages import (
    Message,
    MessageRole,
    detach_message,
)
from cayu.observability.hooks import (
    RuntimeHookPhase,
)
from cayu.providers._credential_boundary import (
    copy_provider_cancellation_failures,
)
from cayu.runtime import _approval_publication as approval_publication
from cayu.runtime import _approval_support as approval_support
from cayu.runtime import _runtime_records as runtime_records
from cayu.runtime import _tool_results as tool_results
from cayu.runtime import _tool_round_recovery as tool_round_recovery
from cayu.runtime import _transcript as transcript_helpers
from cayu.runtime._continuation_task_failure import (
    load_direct_task_failure_replay,
    provider_operation_task_failure_payload,
    runtime_task_failure_terminalization_request,
    runtime_task_terminalization_idempotency_key,
)
from cayu.runtime._delegated_event_stream import (
    _close_delegated_event_stream as _close_delegated_event_stream,
)
from cayu.runtime._diagnostics import (
    exception_diagnostic,
)
from cayu.runtime._durable_tool_round import (
    DeferredInteractionInput,
    DurableToolRound,
)
from cayu.runtime._durable_tool_round import _environment_name as _environment_name
from cayu.runtime._durable_tool_round import (
    _interrupted_tool_round_results as _interrupted_tool_round_results,
)
from cayu.runtime._durable_tool_round import (
    _limit_reached_tool_call_event as _limit_reached_tool_call_event,
)
from cayu.runtime._durable_tool_round import (
    _limit_reached_tool_round_results as _limit_reached_tool_round_results,
)
from cayu.runtime._durable_tool_round import _limit_value_for_payload as _limit_value_for_payload
from cayu.runtime._environment_lifecycle import (
    EnvironmentLifecycle,
)
from cayu.runtime._event_writer import RuntimeEventWriter
from cayu.runtime._foreground_gate_continuation import ForegroundGatePolicyOwner
from cayu.runtime._foreground_subagent_recovery import ForegroundSubagentRecoveryRequired
from cayu.runtime._interruption_coordinator import (
    _PENDING_INTERRUPTION_CASCADE_CHECKPOINT_KEY,
    _PENDING_SESSION_INTERRUPT_CHECKPOINT_KEY,
    BackgroundInterruptionCoordinator,
    interruption_cascade_suppressed,
)
from cayu.runtime._invocation_lifecycle import (
    InvocationContext,
)
from cayu.runtime._live_model_attempt import _provider_failure_proves_no_model_effect
from cayu.runtime._memory_evidence import (
    close_context_exposure_without_provider_effect,
    close_unrecoverable_context_exposure,
)
from cayu.runtime._model_completion_contracts import (
    model_completion_recovery_context_from_stage,
)
from cayu.runtime._model_event_authority import _event_with_model_identity_authority
from cayu.runtime._model_step_executor import (
    ModelStepBudgetEvaluationRequest,
    ModelStepBudgetReservationFailureRequest,
    ModelStepLimitEvaluationRequest,
)
from cayu.runtime._recovery_claims import (
    _IncompleteRecoveryClaimLost,
)
from cayu.runtime._recovery_ownership import (
    RecoveryOwnership,
    _run_recovery_cleanup_steps,
)
from cayu.runtime._recovery_requests import (
    ProviderOperationFailureRequest,
    RecoveryAbandonedSessionRequest,
    RecoveryAbandonedTurnRequest,
    RecoveryInterruptionRequest,
    RecoveryLimitStopRequest,
    RecoveryTerminalEventRequest,
)
from cayu.runtime._run_limits import (
    BudgetEvaluation,
    LimitEvaluation,
    RunLimitController,
    SessionUsageTracker,
    budget_limit_reached_payload,
)
from cayu.runtime._session_control import (
    ActiveSessionRun,
    SessionControl,
    SessionInterruptedByRequest,
    clear_current_task_cancellation,
)
from cayu.runtime._terminal_event_publication import TerminalEventPublication
from cayu.runtime._terminal_evidence_finalization import (
    TerminalEvidenceFinalization,
)
from cayu.runtime._terminal_evidence_reader import (
    TerminalEvidenceReader,
)
from cayu.runtime._tool_round_executor import (
    ToolRoundLimitRequest,
    ordered_tool_result_messages,
)
from cayu.runtime._tool_round_staging import _redactor_for_tool_calls
from cayu.runtime._work_attempt_session_mutation import (
    record_work_attempt_execution_stop,
)
from cayu.runtime.provider_operations import (
    ProviderOperationEvidenceError,
    load_recoverable_provider_operation,
    provider_operation_resolution_outcome_event_id,
    validate_provider_operation_resolution_outcome_event,
)
from cayu.runtime.stop_policy import StopDecision, StopLimit
from cayu.runtime.tool_completion import (
    ToolCompletionResult,
)
from cayu.sessions import _model_completion_publication as model_completion_publication
from cayu.sessions import _pending_approval_reader as pending_approval_reader
from cayu.sessions import _pending_tool_round_reader as pending_round_reader
from cayu.sessions._checkpoint_preservation import (
    _invocation_lifecycle_authority_mutation_scope,
    _invocation_lifecycle_authority_read_scope,
)
from cayu.sessions._execution_profile_checkpoint import (
    active_invocation_execution_profile_from_checkpoint,
    active_invocation_execution_profile_is_released,
    active_invocation_execution_profile_matches_session_epoch,
    checkpoint_with_active_invocation_execution_profile,
    execution_profile_from_session_metadata,
)
from cayu.sessions._foreground_child_checkpoint import (
    ForegroundChildWait,
    foreground_child_state_from_checkpoint,
)
from cayu.sessions._invocation_lifecycle import (
    AdmittedInvocationBinding,
    SettleInvocationCommand,
)
from cayu.sessions._invocation_terminal_decision import (
    InvocationTerminalDecision,
    InvocationTerminalOutcome,
    build_invocation_terminal_decision,
    checkpoint_after_invocation_terminal_decision,
    checkpoint_with_invocation_terminal_decision,
    invocation_terminal_decision_from_checkpoint,
    invocation_terminal_decision_matches_active_profile,
    invocation_terminal_decision_matches_recovery_profile,
    invocation_terminal_event_id,
    settled_invocation_terminal_decision_from_checkpoint,
)
from cayu.sessions._terminal_evidence import (
    _INTERRUPTION_TYPE_OPERATOR_REQUESTED,
    _INTERRUPTION_TYPE_RUNTIME_INTERRUPTED,
    _session_run_operation_from_checkpoint,
    interruption_request_id_from_payload,
    require_interruption_event_matches_pending_marker,
)
from cayu.sessions.base import (
    _SESSION_RUN_OPERATION_ID_PAYLOAD_KEY,
    InteractionTransitionReceiptResult,
    InteractionTransitionResult,
    InteractionTransitionSpec,
    ModelCompletionStageDisposition,
    ModelCompletionStageSettlementRequest,
    SessionModelCompletionStageConflict,
    SessionRunFenced,
    SessionRuntimePublicationConflict,
    SessionStatusConflict,
    SessionStore,
    _activate_session_interaction,
    _clear_session_interaction_recovered_active_through,
    _close_session_interaction,
    _current_session_interaction_id,
    _current_session_interaction_recovered_active_through,
    _current_session_interaction_started_at,
    _current_session_invocation_interaction_ids,
    _deactivate_session_interaction,
    _event_with_session_run_operation,
    _incomplete_recovery_claim_from_checkpoint,
    _latest_session_invocation_interaction_is_settled,
    _mark_session_interaction_settled,
    _mark_session_invocation_terminal_event,
    copy_interaction_transition_spec,
    model_completion_stage_settlement_request,
    runtime_publication_checkpoint_value_digest,
)
from cayu.sessions.checkpoints import (
    AMBIGUOUS_PENDING_USER_INPUT_CHECKPOINT_KEY,
)
from cayu.sessions.cleanup import (
    RecoveryCleanupSupervisor,
)
from cayu.sessions.event_queries import EventOrder, EventQuery
from cayu.sessions.interactions import (
    INTERACTION_LIFECYCLE_EVENT_TYPES,
    INTERACTION_SUMMARY_EVENT_TYPES,
    INTERACTION_TERMINAL_EVENT_TYPES,
    InteractionStatus,
    InteractionSummaryEvidence,
    interaction_usage_summary,
)
from cayu.sessions.records import (
    Session,
    SessionStatus,
)
from cayu.sessions.transcript_queries import (
    TranscriptQuery,
)
from cayu.tasks._terminalization import _terminalize_claimed_task
from cayu.tasks.records import Task
from cayu.tasks.store import TaskStore
from cayu.tasks.terminalization import (
    TaskTerminalKind,
)
from cayu.vaults.redaction import SecretRedactor
from cayu.workspaces.observation_recovery import (
    retain_workspace_observation_pending_cancellation_requests,
)

_INTERACTION_TRANSITION_REPLAY_MAX_ATTEMPTS = 3

_INTERACTION_TRANSITION_REPLAY_WINDOW_SECONDS = 30.0

_INTERACTION_TRANSITION_RUN_FENCE_ATTRIBUTE = "_cayu_interaction_transition_run_fence"

_INTERACTION_TRANSITION_RUN_FENCE_AUTHORITY = object()

_INTERACTION_TRANSITION_REPLAY_FAILURE_AUTHORITY = object()

_INTERACTION_TRANSITION_REPLAY_ATTEMPTS_ATTRIBUTE = "_cayu_interaction_transition_replay_attempts"

_INTERACTION_TRANSITION_CANCELLATION_OUTCOME_ATTRIBUTE = (
    "_cayu_interaction_transition_cancellation_outcome"
)

_INTERACTION_TRANSITION_CANCELLATION_OUTCOME_AUTHORITY = object()

_INTERACTION_TRANSITION_DIAGNOSTIC_MAX_DEPTH = 8

_INTERACTION_TRANSITION_DIAGNOSTIC_MAX_DESCENDANTS = 32

_ExceptionT = TypeVar("_ExceptionT", bound=BaseException)


_INTERRUPTION_TYPE_LIMIT_REACHED = "limit_reached"


@dataclass(frozen=True, slots=True)
class _InteractionTransitionCancellationOutcome:
    """Authenticated immutable evidence carried to the outer run boundary."""

    authority: object
    transition_settled: bool
    encoded_failure_diagnostics: str
    encoded_transition_spec: str | None


def _interaction_transition_failure_contains_run_fence(error: BaseException) -> bool:
    return any(isinstance(candidate, SessionRunFenced) for candidate in iter_exception_tree(error))


def _mark_interaction_transition_run_fence(error: BaseException) -> None:
    BaseException.__setattr__(
        error,
        _INTERACTION_TRANSITION_RUN_FENCE_ATTRIBUTE,
        _INTERACTION_TRANSITION_RUN_FENCE_AUTHORITY,
    )


def _is_interaction_transition_run_fence(error: BaseException) -> bool:
    try:
        authority = BaseException.__getattribute__(
            error,
            _INTERACTION_TRANSITION_RUN_FENCE_ATTRIBUTE,
        )
    except (AttributeError, TypeError):
        return False
    return authority is _INTERACTION_TRANSITION_RUN_FENCE_AUTHORITY


def _interaction_transition_replay_failure(
    failures: list[Exception],
) -> Exception:
    """Return one ordered terminal failure without flattening attempt groups."""

    if not failures:
        raise AssertionError("Interaction transition replay had no failures.")
    unique_failures = _unique_exception_identities(failures)
    if len(failures) == 1:
        failure = unique_failures[0]
    else:
        failure = ExceptionGroup(
            "Interaction transition publication failed across replay attempts.",
            unique_failures,
        )
        BaseException.__setattr__(
            failure,
            "_cayu_interaction_transition_replay_failure",
            _INTERACTION_TRANSITION_REPLAY_FAILURE_AUTHORITY,
        )
        BaseException.__setattr__(
            failure,
            _INTERACTION_TRANSITION_REPLAY_ATTEMPTS_ATTRIBUTE,
            tuple(failures),
        )
    return failure


def _interaction_transition_failure_diagnostics(
    failures: tuple[BaseException, ...],
    *,
    redactor: SecretRedactor,
) -> list[dict[str, Any]]:
    """Return bounded ordered diagnostics without flattening attempt groups."""

    remaining_descendants = _INTERACTION_TRANSITION_DIAGNOSTIC_MAX_DESCENDANTS
    root_identities = {id(failure) for failure in failures}
    seen_identities = set(root_identities)

    def project(error: BaseException, *, depth: int) -> dict[str, Any]:
        nonlocal remaining_descendants
        projected = exception_diagnostic(error, redactor=redactor).payload_fields()
        if not isinstance(error, BaseExceptionGroup):
            return projected
        children = exception_group_children(error)
        if children is None:
            projected["children_unavailable"] = True
            return projected
        if depth >= _INTERACTION_TRANSITION_DIAGNOSTIC_MAX_DEPTH:
            projected["children_truncated"] = len(children)
            return projected

        projected_children: list[dict[str, Any]] = []
        duplicate_children = 0
        truncated_children = 0
        for child in children:
            child_identity = id(child)
            if child_identity in seen_identities:
                duplicate_children += 1
                continue
            if remaining_descendants <= 0:
                truncated_children += 1
                continue
            seen_identities.add(child_identity)
            remaining_descendants -= 1
            projected_children.append(project(child, depth=depth + 1))
        if projected_children:
            projected["children"] = projected_children
        if duplicate_children:
            projected["duplicate_children_omitted"] = duplicate_children
        if truncated_children:
            projected["children_truncated"] = truncated_children
        return projected

    return [project(failure, depth=0) for failure in failures]


def _unique_exception_identities(
    failures: Iterable[_ExceptionT],
) -> list[_ExceptionT]:
    """Preserve first-seen order without recording one exception object twice."""

    unique: list[_ExceptionT] = []
    for failure in failures:
        if not any(existing is failure for existing in unique):
            unique.append(failure)
    return unique


async def _run_interaction_transition_cancellation_cleanup_steps(
    cancellation: asyncio.CancelledError,
    *,
    steps: tuple[tuple[str, Callable[[], Awaitable[None]]], ...],
    supervisor: RecoveryCleanupSupervisor,
) -> tuple[tuple[str, BaseException], ...]:
    """Keep this cancellation's attempt and cleanup failures jointly visible.

    The shared recovery helper intentionally keeps historical causes behind the
    current cleanup group. Interaction-transition acknowledgement failures are
    different: they and the cancellation cleanup belong to one bounded replay
    handoff, so this narrow boundary exposes both without changing sibling
    recovery classification contracts.
    """

    attempt_failures = exception_cause(cancellation)
    cleanup_failures = await _run_recovery_cleanup_steps(
        authoritative_failure=cancellation,
        steps=steps,
        supervisor=supervisor,
    )
    cleanup_cause = exception_cause(cancellation)
    if (
        cleanup_failures
        and attempt_failures is not None
        and cleanup_cause is not None
        and cleanup_cause is not attempt_failures
    ):
        set_exception_cause(cleanup_cause, None)
        set_exception_cause(
            cancellation,
            BaseExceptionGroup(
                "Interaction transition attempt and cancellation cleanup failures.",
                _unique_exception_identities([attempt_failures, cleanup_cause]),
            ),
        )
    return cleanup_failures


def _normalize_interaction_transition_child_cancellation(
    error: BaseException | None,
) -> BaseException | None:
    """Classify store-owned cancellation without fabricating caller cancellation."""

    if isinstance(error, asyncio.CancelledError):
        return unexpected_child_cancellation_error(
            error,
            operation="Interaction transition publication",
        )
    if not isinstance(error, BaseExceptionGroup) or not any(
        isinstance(candidate, asyncio.CancelledError) for candidate in iter_exception_tree(error)
    ):
        return error
    return rebuild_exception_group(
        error,
        group_message="Interaction transition publication child failures.",
        leaf_mapper=lambda child: (
            unexpected_child_cancellation_error(
                child,
                operation="Interaction transition publication",
            )
            if isinstance(child, asyncio.CancelledError)
            else child
        ),
        invalid_leaf_factory=lambda: RuntimeError(
            "Interaction transition publication returned an unreadable exception group."
        ),
    )


def _validated_interaction_transition_result(
    value: object,
    *,
    transition: InteractionTransitionSpec,
) -> InteractionTransitionResult:
    """Detach and verify one extension-controlled publication result."""

    if type(value) is not InteractionTransitionResult:
        raise TypeError("Session store returned an invalid interaction transition result.")
    try:
        result = InteractionTransitionResult(
            session=value.session,
            event=value.event,
            terminal_event=value.terminal_event,
            status_changed=value.status_changed,
            replayed=value.replayed,
        )
    except (AttributeError, TypeError, ValueError):
        raise TypeError(
            "Session store returned an invalid interaction transition result."
        ) from None
    if result.event != transition.event:
        raise RuntimeError("Interaction transition publication returned a conflicting event.")
    if result.terminal_event != transition.terminal_event:
        raise RuntimeError(
            "Interaction transition publication returned conflicting terminal evidence."
        )
    if result.session.id != transition.event.session_id:
        raise RuntimeError("Interaction transition publication returned a conflicting session.")
    if result.status_changed:
        if result.session.status is not transition.to_status:
            raise RuntimeError(
                "Interaction transition publication changed status does not match its session."
            )
    elif (
        not transition.only_if_no_queued_messages
        or result.session.status not in transition.from_statuses
    ):
        raise RuntimeError("Interaction transition publication unchanged status is inconsistent.")
    return result


def _attach_interaction_transition_cancellation_outcome(
    cancellation: asyncio.CancelledError,
    *,
    transition_settled: bool,
    failure_diagnostics: list[dict[str, Any]],
    transition: InteractionTransitionSpec | None,
) -> None:
    """Carry only authenticated, durable-safe transition settlement evidence."""

    copied_diagnostics = copy_durable_json_value(
        failure_diagnostics,
        "interaction transition cancellation diagnostics",
    )
    if type(copied_diagnostics) is not list:
        raise TypeError("Interaction transition cancellation diagnostics must be a list.")
    outcome = _InteractionTransitionCancellationOutcome(
        authority=_INTERACTION_TRANSITION_CANCELLATION_OUTCOME_AUTHORITY,
        transition_settled=transition_settled,
        encoded_failure_diagnostics=json.dumps(
            copied_diagnostics,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ),
        encoded_transition_spec=(
            None
            if transition is None
            else json.dumps(
                copy_interaction_transition_spec(transition).model_dump(mode="json"),
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            )
        ),
    )
    if not set_exception_state(
        cancellation,
        _INTERACTION_TRANSITION_CANCELLATION_OUTCOME_ATTRIBUTE,
        outcome,
    ):
        raise RuntimeError("Could not attach interaction transition cancellation outcome.")


def _pop_interaction_transition_cancellation_outcome(
    cancellation: asyncio.CancelledError,
) -> tuple[bool, list[dict[str, Any]], InteractionTransitionSpec | None] | None:
    """Consume transition evidence only when it came from the owned store boundary."""

    outcome = pop_exception_state(
        cancellation,
        _INTERACTION_TRANSITION_CANCELLATION_OUTCOME_ATTRIBUTE,
    )
    if (
        type(outcome) is not _InteractionTransitionCancellationOutcome
        or outcome.authority is not _INTERACTION_TRANSITION_CANCELLATION_OUTCOME_AUTHORITY
        or type(outcome.transition_settled) is not bool
        or type(outcome.encoded_failure_diagnostics) is not str
        or (
            outcome.encoded_transition_spec is not None
            and type(outcome.encoded_transition_spec) is not str
        )
    ):
        return None
    try:
        decoded = json.loads(outcome.encoded_failure_diagnostics)
        copied = copy_durable_json_value(
            decoded,
            "interaction transition cancellation diagnostics",
        )
        transition = (
            None
            if outcome.encoded_transition_spec is None
            else InteractionTransitionSpec.model_validate(
                json.loads(outcome.encoded_transition_spec)
            )
        )
    except Exception:
        return None
    if type(copied) is not list or any(type(item) is not dict for item in copied):
        return None
    return outcome.transition_settled, copied, transition


def _raise_interaction_transition_cancellation(
    cancellation: asyncio.CancelledError,
    secondary_failures: Iterable[BaseException] = (),
    *,
    transition_settled: bool = False,
    failure_diagnostics: list[dict[str, Any]] | None = None,
    transition: InteractionTransitionSpec | None = None,
) -> NoReturn:
    """Redeliver caller cancellation with every settled secondary outcome."""

    current_failures = list(secondary_failures)
    if any(
        _interaction_transition_failure_contains_run_fence(failure) for failure in current_failures
    ):
        _mark_interaction_transition_run_fence(cancellation)
    _attach_interaction_transition_cancellation_outcome(
        cancellation,
        transition_settled=transition_settled,
        failure_diagnostics=[] if failure_diagnostics is None else failure_diagnostics,
        transition=transition,
    )
    preserved: list[BaseException] = []
    existing_cause = exception_cause(cancellation)
    if existing_cause is not None:
        preserved.append(existing_cause)
    preserved = _unique_exception_identities([*preserved, *current_failures])
    if preserved:
        cause: BaseException = (
            preserved[0]
            if len(preserved) == 1
            else BaseExceptionGroup(
                "Interaction transition cancellation and store failures.",
                preserved,
            )
        )
        if not set_exception_cause(cancellation, cause):
            raise BaseExceptionGroup(
                "Interaction transition cancellation and store failures.",
                [cancellation, cause],
            ) from None
    raise cancellation from exception_cause(cancellation)


def _interaction_transition_replay_is_authoritatively_rejected(
    error: BaseException,
) -> bool:
    """Return whether retry cannot reconcile the rejected transition."""

    from cayu.runtime._session_steering import SessionSteeringBoundaryReached

    leaves = tuple(
        candidate
        for candidate in iter_exception_tree(error)
        if not isinstance(candidate, BaseExceptionGroup)
    )
    return bool(leaves) and all(
        isinstance(leaf, (SessionStatusConflict, SessionSteeringBoundaryReached)) for leaf in leaves
    )


def _checkpoint_with_pending_session_interrupt(
    payload: dict[str, Any],
    *,
    include_interruption_cascade: bool = True,
    cascade_created_at: datetime | None = None,
    expected_interrupted_user_input: PendingUserInput | None = None,
    expected_ambiguous_user_input: AmbiguousPendingUserInput | None = None,
    expected_foreground_child_wait: ForegroundChildWait | None = None,
    expected_interrupted_approval: PendingToolApproval | None = None,
    terminal_decision: InvocationTerminalDecision | None = None,
    redactor: SecretRedactor | None = None,
):
    copied_payload = copy_json_value(payload, "interrupt_payload")
    if (
        expected_interrupted_user_input is not None
        and type(expected_interrupted_user_input) is not PendingUserInput
    ):
        raise TypeError("expected_interrupted_user_input must be PendingUserInput or None.")
    if (
        expected_ambiguous_user_input is not None
        and type(expected_ambiguous_user_input) is not AmbiguousPendingUserInput
    ):
        raise TypeError("expected_ambiguous_user_input must be AmbiguousPendingUserInput or None.")
    if cascade_created_at is not None and (
        cascade_created_at.tzinfo is None or cascade_created_at.utcoffset() is None
    ):
        raise ValueError("cascade_created_at must be timezone-aware.")
    resolved_cascade_created_at = (
        datetime.now(UTC) if cascade_created_at is None else cascade_created_at.astimezone(UTC)
    )

    def transform(session: Session, checkpoint: dict[str, Any] | None) -> dict[str, Any]:
        transition_payload = copy_json_value(copied_payload, "interrupt_payload")
        copied_checkpoint = (
            {} if checkpoint is None else copy_durable_record(checkpoint, "checkpoint")
        )
        foreground_wait, _ = foreground_child_state_from_checkpoint(copied_checkpoint)
        if (
            foreground_wait is not None
            and transition_payload.get("interruption_type") == _INTERRUPTION_TYPE_OPERATOR_REQUESTED
        ):
            intent = foreground_wait.parent_effect
            if intent.session_id != session.id or intent.session_instance_id != session.instance_id:
                raise SessionRunFenced(
                    "Operator interruption has foreign foreground wait evidence."
                )
            # Index the stop against the same original call as the human pause,
            # so bounded pending-action queries cannot keep showing that pause.
            transition_payload.update(
                model_step_id=intent.model_step_id,
                model_attempt_id=intent.model_attempt_id,
                tool_round_id=intent.tool_round_id,
                tool_call_id=intent.tool_call_id,
            )
            if terminal_decision is not None and any(
                terminal_decision.terminal_payload.get(key) != transition_payload[key]
                for key in ("model_step_id", "model_attempt_id", "tool_round_id", "tool_call_id")
            ):
                raise SessionRunFenced("Foreground stop evidence changed after terminal election.")
        if expected_foreground_child_wait is not None:
            current_wait, _ = foreground_child_state_from_checkpoint(copied_checkpoint)
            if (
                current_wait is None
                or current_wait != expected_foreground_child_wait
                or current_wait.parent_effect.session_id != session.id
                or current_wait.parent_effect.session_instance_id != session.instance_id
                or terminal_decision is None
                or terminal_decision.interaction_id != current_wait.parent_effect.interaction_id
            ):
                raise SessionRunFenced("Foreground parent interruption lost its exact wait.")
        active_profile = active_invocation_execution_profile_from_checkpoint(copied_checkpoint)
        if expected_interrupted_approval is not None:
            if redactor is None:
                raise TypeError("Approval stop requires its runtime redactor.")
            current_approval = pending_approval_reader.pending_approval_from_checkpoint(
                copied_checkpoint
            )
            approval_round = pending_round_reader.pending_tool_round_from_checkpoint(
                copied_checkpoint
            )
            if (
                current_approval != expected_interrupted_approval
                or approval_round is None
                or approval_round.assistant_message_state != "quarantined"
                or terminal_decision is None
                or pending_approval_reader.approval_resolution_intent_from_checkpoint(
                    copied_checkpoint
                )
                is not None
            ):
                raise SessionRunFenced("Approval stop lost its exact unclaimed pause.")
            close_intent = approval_support.approval_interrupt_close_intent(
                expected_interrupted_approval
            )
            if (
                terminal_decision.terminal_payload.get(
                    approval_support.APPROVAL_INTERRUPT_CLOSE_INTENT_KEY
                )
                != close_intent
            ):
                raise SessionRunFenced("Approval stop lost its exact terminal close intent.")
            # This unclaimed gate has never exposed its assistant/tool round.
            # Retire both together; retaining the planned round after closing
            # the interaction would incorrectly request new execution authority.
            copied_checkpoint = approval_support.checkpoint_without_exact_pending_approval_round(
                copied_checkpoint,
                approval=expected_interrupted_approval,
                redactor=redactor,
                runtime_session=session,
            )
            transition_payload[approval_support.APPROVAL_INTERRUPT_CLOSE_INTENT_KEY] = close_intent
        existing_terminal_decision = invocation_terminal_decision_from_checkpoint(copied_checkpoint)
        if active_profile is not None:
            if not active_invocation_execution_profile_matches_session_epoch(
                active_profile,
                session_id=session.id,
                run_epoch=session.run_epoch,
            ):
                raise RuntimeError(
                    "Active invocation execution profile does not match the interrupted "
                    "session run epoch."
                )
            copied_checkpoint = checkpoint_with_active_invocation_execution_profile(
                copied_checkpoint,
                session_id=session.id,
                interaction_id=active_profile.interaction_id,
                run_epoch=session.run_epoch,
                profile=active_profile.profile,
                expected=active_profile,
            )
        if terminal_decision is None and existing_terminal_decision is not None:
            raise SessionRunFenced(
                "Session interruption conflicts with an elected terminal decision."
            )
        if terminal_decision is not None:
            if (
                active_profile is None
                or terminal_decision.outcome is not InvocationTerminalOutcome.INTERRUPTED
                or not invocation_terminal_decision_matches_active_profile(
                    terminal_decision,
                    session_id=session.id,
                    session_instance_id=session.instance_id,
                    run_epoch=session.run_epoch,
                    interaction_id=active_profile.interaction_id,
                    execution_profile_fingerprint=active_profile.profile.fingerprint,
                )
            ):
                raise SessionRunFenced(
                    "Interruption terminal decision lost its invocation authority."
                )
            copied_checkpoint = checkpoint_with_invocation_terminal_decision(
                copied_checkpoint,
                terminal_decision,
            )
        ambiguous_user_input = ambiguous_pending_user_input_from_checkpoint(copied_checkpoint)
        if ambiguous_user_input is None:
            pending_user_input, resolution_intent = user_input_lifecycle_authority_from_checkpoint(
                copied_checkpoint,
                current_run_epoch=session.run_epoch,
                runtime_session=session,
            )
        else:
            pending_user_input = None
            resolution_intent = None
        if (
            expected_ambiguous_user_input is not None
            and session.status is SessionStatus.INTERRUPTED
            and ambiguous_user_input != expected_ambiguous_user_input
        ):
            raise SessionRuntimePublicationConflict(
                "Interrupted ambiguous user-input pause changed before supersession."
            )
        if (
            expected_interrupted_user_input is not None
            and session.status is SessionStatus.INTERRUPTED
            and (
                pending_user_input is None
                or pending_user_input_identity(pending_user_input)
                != pending_user_input_identity(expected_interrupted_user_input)
            )
        ):
            raise SessionRuntimePublicationConflict(
                "Interrupted user-input pause changed before supersession."
            )
        delegated_input_stop = (
            expected_foreground_child_wait is not None
            and terminal_decision is not None
            and pending_user_input is not None
            and resolution_intent is not None
            and resolution_intent.execution_state == "executing"
            and pending_user_input.session_id == session.id
            and pending_user_input.session_instance_id == session.instance_id
            and pending_user_input.input_id == expected_foreground_child_wait.parent_effect.pause_id
            and pending_user_input.source_interaction_id == terminal_decision.interaction_id
            and pending_user_input.execution_profile_fingerprint
            == expected_foreground_child_wait.parent_effect.execution_profile_fingerprint
        )
        # The exact wait was compared above under this transaction. Stopping
        # that delegated interaction retains its already accepted answer; it is
        # not authority to supersede an executing user-input resolution.
        if (
            pending_user_input is not None
            and not delegated_input_stop
            and transition_payload.get("interruption_type") == _INTERRUPTION_TYPE_OPERATOR_REQUESTED
        ):
            stored_profile = execution_profile_from_session_metadata(session.metadata)
            expected_current_run_epoch = (
                pending_user_input.source_run_epoch
                if resolution_intent is None
                else resolution_intent.claim_run_epoch
            )
            current_epoch_matches = session.run_epoch == expected_current_run_epoch or (
                session.status is SessionStatus.INTERRUPTED
                and expected_interrupted_user_input is not None
                and session.run_epoch == expected_current_run_epoch + 1
            )
            if (
                pending_user_input.session_id != session.id
                or pending_user_input.session_instance_id != session.instance_id
                or not current_epoch_matches
                or pending_user_input.execution_profile_fingerprint != stored_profile.fingerprint
                or (
                    active_profile is not None
                    and active_profile.profile.fingerprint != stored_profile.fingerprint
                )
            ):
                raise RuntimeError(
                    "Pending user input does not belong to the interrupted invocation."
                )
            if resolution_intent is not None and resolution_intent.execution_state == "executing":
                raise SessionRuntimePublicationConflict(
                    "User-input continuation is already executing and cannot be superseded."
                )
            supersession_intent = user_input_supersession_intent_for(
                pending_user_input,
                resolution_intent=resolution_intent,
            )
            transition_payload[USER_INPUT_SUPERSESSION_INTENT_KEY] = supersession_intent.model_dump(
                mode="json", exclude_none=True
            )
            copied_checkpoint.pop(PENDING_USER_INPUT_CHECKPOINT_KEY)
            copied_checkpoint.pop(USER_INPUT_RESOLUTION_INTENT_CHECKPOINT_KEY, None)
        elif (
            ambiguous_user_input is not None
            and transition_payload.get("interruption_type") == _INTERRUPTION_TYPE_OPERATOR_REQUESTED
        ):
            supersession_intent = ambiguous_user_input_supersession_intent_for(
                ambiguous_user_input,
                session_id=session.id,
                session_instance_id=session.instance_id,
            )
            transition_payload[AMBIGUOUS_USER_INPUT_SUPERSESSION_INTENT_KEY] = (
                supersession_intent.model_dump(mode="json")
            )
            copied_checkpoint.pop(AMBIGUOUS_PENDING_USER_INPUT_CHECKPOINT_KEY)
        copied_checkpoint[_PENDING_SESSION_INTERRUPT_CHECKPOINT_KEY] = copy_json_value(
            transition_payload,
            "interrupt_payload",
        )
        if (
            include_interruption_cascade
            and transition_payload.get("interruption_type") == _INTERRUPTION_TYPE_OPERATOR_REQUESTED
        ):
            copied_checkpoint[_PENDING_INTERRUPTION_CASCADE_CHECKPOINT_KEY] = {
                "attempt_id": str(uuid4()),
                "interrupt_payload": copy_json_value(
                    transition_payload,
                    "interrupt_payload",
                ),
                "created_at": resolved_cascade_created_at.isoformat(),
            }
        return copied_checkpoint

    return transform


def _runtime_interruption_event(
    event: Event,
    *,
    execution_profile: ExecutionProfileIdentity | None = None,
) -> Event:
    """Attest runtime-owned interruption identities copied through checkpoints."""

    fields = tuple(
        field_name
        for field_name in ("interruption_request_id", "retry_request_id", "attempt_id")
        if type(event.payload.get(field_name)) is str
    )
    event = event_with_runtime_payload_authority(event, *fields)
    supersession_payload = event.payload.get(USER_INPUT_SUPERSESSION_INTENT_KEY)
    if supersession_payload is not None:
        try:
            supersession_intent = UserInputSupersessionIntent.model_validate(supersession_payload)
        except (TypeError, ValueError) as exc:
            raise RuntimeError("User-input supersession evidence is malformed.") from exc
        event = event_with_user_input_supersession_authority(event, supersession_intent)
    ambiguous_supersession_payload = event.payload.get(AMBIGUOUS_USER_INPUT_SUPERSESSION_INTENT_KEY)
    if ambiguous_supersession_payload is not None:
        try:
            ambiguous_supersession_intent = AmbiguousUserInputSupersessionIntent.model_validate(
                ambiguous_supersession_payload
            )
        except (TypeError, ValueError) as exc:
            raise RuntimeError("Ambiguous user-input supersession evidence is malformed.") from exc
        event = event_with_ambiguous_user_input_supersession_authority(
            event,
            ambiguous_supersession_intent,
        )
    profile_fingerprint = event.payload.get("execution_profile_fingerprint")
    if profile_fingerprint is None:
        return event
    if execution_profile is None:
        raise RuntimeError(
            "An interrupted event references an execution profile without validated authority."
        )
    if profile_fingerprint != execution_profile.fingerprint:
        raise RuntimeError(
            "An interrupted event references a different execution profile than recovery."
        )
    return event_with_execution_profile_authority(event, execution_profile)


def _limit_reached_payload(
    *,
    decision: StopDecision,
    usage_summary: SessionUsageSummary,
    cost_summary: SessionCostTotals | None,
) -> dict[str, Any]:
    payload = {
        "reason": "limit_reached",
        "limit": decision.limit.value,
        "maximum": _limit_value_for_payload(decision.maximum),
        "actual": _limit_value_for_payload(decision.actual),
        "message": decision.message,
        "usage_summary": session_usage_summary_payload(usage_summary),
    }
    if cost_summary is not None:
        payload["cost_summary"] = cost_summary.model_dump(mode="json")
    return payload


async def _close_async_iterator(iterator: AsyncIterator[Any]) -> None:
    close = getattr(iterator, "aclose", None)
    if close is not None:
        await close()


async def _collect_through_event_type(
    iterator: AsyncIterator[Event],
    event_type: EventType | tuple[EventType, ...],
    *,
    missing_message: str,
) -> tuple[list[Event], Event]:
    events: list[Event] = []
    async for event in iterator:
        events.append(event)
        if event.type in ((event_type,) if isinstance(event_type, EventType) else event_type):
            return events, event
    raise RuntimeError(missing_message)


def _task_event(
    *,
    event_type: EventType,
    task: Task,
    session: Session,
    registered_agent: runtime_records.RegisteredAgentState,
    registered_environment: runtime_records.RegisteredEnvironment | None,
) -> Event:
    event = Event(
        type=event_type,
        session_id=session.id,
        agent_name=registered_agent.spec.name,
        environment_name=_environment_name(registered_environment),
        payload={
            "task_id": task.id,
            "task_type": task.type,
            "task_status": task.status.value,
            "task_session_id": task.session_id,
            "assigned_agent_name": task.assigned_agent_name,
            "parent_task_id": task.parent_task_id,
        },
    )
    return event_with_runtime_payload_authority(
        event,
        *(
            field_name
            for field_name in ("task_id", "task_session_id", "parent_task_id")
            if event.payload.get(field_name) is not None
        ),
    )


def _task_terminalization_idempotency_key(
    *,
    task_id: str,
    session_id: str,
    kind: TaskTerminalKind,
) -> str:
    return runtime_task_terminalization_idempotency_key(
        task_id=task_id,
        session_id=session_id,
        kind=kind,
    )


def _raise_primary_with_secondary_failure(
    primary: BaseException,
    secondary: BaseException,
    *,
    group_message: str,
) -> NoReturn:
    existing_cause = exception_cause(primary)
    combined_cause: BaseException = secondary
    if existing_cause is not None:
        combined_cause = BaseExceptionGroup(
            group_message,
            [existing_cause, secondary],
        )
    if not set_exception_cause(primary, combined_cause):
        raise BaseExceptionGroup(
            group_message,
            [primary, secondary],
        )
    raise primary


def _pending_interaction_action_kind(
    checkpoint: dict[str, Any] | None, *, run_epoch: int
) -> str | None:
    """Use the same durable gate classification for pause and terminal election."""

    from cayu.sessions._foreground_child_checkpoint import foreground_child_state_from_checkpoint

    if foreground_child_state_from_checkpoint(checkpoint)[0] is not None:
        return "waiting_on_child_action"

    if pending_approval_reader.pending_approval_from_checkpoint(checkpoint) is not None:
        return "tool_approval"
    if (
        user_input_lifecycle_authority_from_checkpoint(checkpoint, current_run_epoch=run_epoch)[0]
        is not None
    ):
        return "user_input"
    if pending_round_reader.pending_tool_round_from_checkpoint(checkpoint) is not None:
        return "tool_recovery"
    from cayu.sessions._session_continuation_store import has_unparked_external_interaction

    if has_unparked_external_interaction(checkpoint, run_epoch=run_epoch):
        return "external_wait_recovery"
    return None


_ABANDONED_RUN_REASON = "event_stream_closed"


class SessionFinalization:
    """Own session terminal transitions, interruption settlement and run-limit closure."""

    def __init__(
        self,
        *,
        session_store: SessionStore,
        task_store: TaskStore | None,
        event_writer: RuntimeEventWriter,
        session_control: SessionControl[SessionUsageTracker],
        environment_lifecycle: EnvironmentLifecycle,
        run_limit_controller: RunLimitController,
        recovery_ownership: RecoveryOwnership,
        recovery_cleanup_supervisor: RecoveryCleanupSupervisor,
        terminal_evidence: TerminalEvidenceReader,
        terminal_finalization: TerminalEvidenceFinalization,
        terminal_event_publication: TerminalEventPublication,
        foreground_gate_policy_owner: ForegroundGatePolicyOwner,
        background_interruption_coordinator: BackgroundInterruptionCoordinator,
        deferred_input: DeferredInteractionInput,
        secret_redactor: SecretRedactor,
        clock: Callable[[], datetime],
        resolve_registered_agent: Callable[[str], runtime_records.RegisteredAgentState],
        resolve_registered_environment: Callable[
            [str | None], runtime_records.RegisteredEnvironment | None
        ],
    ) -> None:
        self.session_store = session_store
        self.task_store = task_store
        self._event_writer = event_writer
        self._session_control = session_control
        self._environment_lifecycle = environment_lifecycle
        self._run_limit_controller = run_limit_controller
        self._recovery_ownership = recovery_ownership
        self._recovery_cleanup_supervisor = recovery_cleanup_supervisor
        self._terminal_evidence = terminal_evidence
        self._terminal_finalization = terminal_finalization
        self._terminal_event_publication = terminal_event_publication
        self._foreground_gate_policy_owner = foreground_gate_policy_owner
        self._background_interruption_coordinator = background_interruption_coordinator
        self._deferred_input = deferred_input
        self._secret_redactor = secret_redactor
        self._clock = clock
        self._resolve_registered_agent = resolve_registered_agent
        self._resolve_registered_environment = resolve_registered_environment
        self.interaction_publication_authority = object()

    def schedule_background_interruption_cascade(
        self,
        *,
        parent_session_id: str,
        interrupt_payload: dict[str, Any],
        create_if_missing: bool,
        retry_request: dict[str, Any] | None = None,
        allow_during_drain: bool = False,
    ) -> asyncio.Task[None] | None:
        return self._background_interruption_coordinator.schedule(
            parent_session_id=parent_session_id,
            interrupt_payload=interrupt_payload,
            create_if_missing=create_if_missing,
            retry_request=retry_request,
            allow_during_drain=allow_during_drain,
        )

    def supports_terminal_interaction_publication_protocol(self) -> bool:
        supports_terminal_publication = getattr(
            self.session_store,
            "_supports_terminal_interaction_publication_protocol",
            None,
        )
        return bool(callable(supports_terminal_publication) and supports_terminal_publication())

    def require_terminal_interaction_publication_protocol(self) -> None:
        if not self.supports_terminal_interaction_publication_protocol():
            raise NotImplementedError(
                "The session store does not support atomic terminal interaction publication."
            )

    async def activate_latest_open_interaction(self, session_id: str) -> str | None:
        records = await self.session_store.query_events(
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

    async def _activate_terminal_decision_interaction(
        self,
        decision: InvocationTerminalDecision,
    ) -> str | None:
        """Restore task-local attribution from exact durable winner evidence."""

        records = await self.session_store.query_events(
            EventQuery(
                session_id=decision.session_id,
                event_types=INTERACTION_LIFECYCLE_EVENT_TYPES,
                order_by=EventOrder.SEQUENCE_DESC,
                limit=1,
            )
        )
        if not records:
            return None
        latest = records[0].event
        if latest.interaction_id != decision.interaction_id:
            return None
        if decision.interaction_event_id is None:
            if (
                latest.type not in INTERACTION_TERMINAL_EVENT_TYPES
                or latest.id != decision.predecessor_interaction_event_id
            ):
                return None
            return None
        if latest.type in INTERACTION_TERMINAL_EVENT_TYPES:
            expected_type = (
                EventType.INTERACTION_INTERRUPTED
                if decision.outcome is InvocationTerminalOutcome.INTERRUPTED
                else EventType.INTERACTION_FAILED
            )
            if (
                latest.type is not expected_type
                or latest.id != decision.interaction_event_id
                or latest.timestamp != decision.observed_at
            ):
                return None
        _activate_session_interaction(decision.session_id, decision.interaction_id)
        return decision.interaction_id

    async def _prepare_interaction_state_event(
        self,
        *,
        session: Session,
        agent_name: str,
        environment_name: str | None,
        event_type: EventType,
        status: InteractionStatus,
        pending_action_kind: str | None = None,
        recovered_active_through: datetime | None = None,
        observed_at: datetime | None = None,
        event_id: str | None = None,
        settlement_status: SessionStatus | None = None,
        tool_completion_result: ToolCompletionResult | None = None,
    ) -> Event | None:
        interaction_id = _current_session_interaction_id(session.id)
        if interaction_id is None:
            raise RuntimeError("Interaction identity is not active for the session.")
        start_records = await self.session_store.query_events(
            EventQuery(
                session_id=session.id,
                interaction_id=interaction_id,
                event_type=EventType.INTERACTION_STARTED,
                order_by=EventOrder.SEQUENCE_ASC,
                limit=1,
            )
        )
        if not start_records:
            raise RuntimeError(f"Interaction has no durable start: {interaction_id}")
        latest_records = await self.session_store.query_events(
            EventQuery(
                session_id=session.id,
                interaction_id=interaction_id,
                event_types=INTERACTION_LIFECYCLE_EVENT_TYPES,
                order_by=EventOrder.SEQUENCE_DESC,
                limit=1,
            )
        )
        latest = latest_records[0]
        if latest.event.type in INTERACTION_TERMINAL_EVENT_TYPES:
            return None
        start_record = start_records[0]
        start_evidence = InteractionSummaryEvidence.model_validate(start_record.event.payload)
        prior_evidence = InteractionSummaryEvidence.model_validate(latest.event.payload)
        if (
            event_type == EventType.INTERACTION_PAUSED
            and latest.event.type == EventType.INTERACTION_PAUSED
            and prior_evidence.pending_action_kind == pending_action_kind
        ):
            if settlement_status is None:
                return None
            checkpoint = await self.session_store.load_checkpoint(session.id)
            active_profile = active_invocation_execution_profile_from_checkpoint(checkpoint)
            if active_profile is not None:
                settlement = await self.session_store.load_invocation_settlement_transition(
                    session.id,
                    expected_session_instance_id=session.instance_id,
                    expected_active_invocation_profile=active_profile,
                )
                if settlement is not None and settlement.to_status is settlement_status:
                    return None
            # An equal pause reason does not prove equal session settlement.
            # Recovery may retain the same round while changing FAILED to
            # INTERRUPTED; publish new exact evidence before releasing it.
        usage_pages: list[SessionUsageSummary] = []
        after_sequence: int | None = None
        while True:
            usage_records = await self.session_store.query_events(
                EventQuery(
                    session_id=session.id,
                    interaction_id=interaction_id,
                    event_types=INTERACTION_SUMMARY_EVENT_TYPES,
                    after_sequence=after_sequence,
                    order_by=EventOrder.SEQUENCE_ASC,
                    limit=5000,
                )
            )
            usage_pages.append(
                interaction_usage_summary(
                    session.id,
                    [record.event for record in usage_records],
                )
            )
            if len(usage_records) < 5000:
                break
            after_sequence = usage_records[-1].sequence
        usage_summary = combine_session_usage_summaries(session.id, usage_pages)

        async def transcript_range(role: MessageRole) -> tuple[int | None, int | None]:
            first_page = await self.session_store.query_transcript(
                TranscriptQuery(
                    session_id=session.id,
                    interaction_id=interaction_id,
                    role=role,
                    limit=1,
                )
            )
            if not first_page.records:
                return None, None
            first = first_page.records[0].index
            if first_page.total_records == 1:
                return first, first
            last_page = await self.session_store.query_transcript(
                TranscriptQuery(
                    session_id=session.id,
                    interaction_id=interaction_id,
                    role=role,
                    offset=first_page.total_records - 1,
                    limit=1,
                )
            )
            if not last_page.records:
                raise RuntimeError("Interaction transcript changed during summary projection.")
            return first, last_page.records[0].index

        source_start, source_end = await transcript_range(MessageRole.USER)
        result_start, result_end = await transcript_range(MessageRole.ASSISTANT)
        segment_started_at = _current_session_interaction_started_at(session.id)
        observed_at = self._clock() if observed_at is None else observed_at
        if recovered_active_through is None:
            recovered_active_through = _current_session_interaction_recovered_active_through(
                session.id
            )
        if recovered_active_through is not None:
            if (
                recovered_active_through.tzinfo is None
                or recovered_active_through.utcoffset() is None
            ):
                raise ValueError("recovered_active_through must be timezone-aware.")
            current_segment_ms = (
                max(
                    0,
                    int(
                        (
                            min(recovered_active_through, observed_at) - latest.event.timestamp
                        ).total_seconds()
                        * 1000
                    ),
                )
                if prior_evidence.status is InteractionStatus.ACTIVE
                else 0
            )
        else:
            current_segment_ms = (
                0
                if segment_started_at is None or event_type == EventType.INTERACTION_RESUMED
                else max(0, int((time.monotonic() - segment_started_at) * 1000))
            )
        terminal = event_type in INTERACTION_TERMINAL_EVENT_TYPES
        try:
            evidence = InteractionSummaryEvidence(
                status=status,
                tool_completion=tool_completion_result,
                start_event_id=start_record.event.id,
                start_event_sequence=start_record.sequence,
                source_transcript_start=source_start,
                source_transcript_end=source_end,
                result_transcript_start=result_start,
                result_transcript_end=result_end,
                started_at=start_evidence.started_at,
                completed_at=observed_at if terminal else None,
                active_duration_ms=prior_evidence.active_duration_ms + current_segment_ms,
                wall_duration_ms=(
                    max(
                        0,
                        int((observed_at - start_evidence.started_at).total_seconds() * 1000),
                    )
                    if terminal
                    else None
                ),
                model_step_count=usage_summary.model_steps,
                tool_call_count=usage_summary.tool_calls,
                token_usage=usage_summary.usage,
                provider_names=usage_summary.provider_names,
                models=usage_summary.models,
                pending_action_kind=pending_action_kind,
            )
        except ValidationError as exc:
            if terminal and observed_at < start_evidence.started_at:
                raise _runtime_interaction_lifecycle_publication_rejected(
                    session_id=session.id,
                    interaction_id=interaction_id,
                    runtime_authority=self.interaction_publication_authority,
                ) from exc
            raise
        event_payload = evidence.model_dump(mode="json")
        event = event_with_runtime_payload_authority(
            event_with_runtime_envelope_authority(
                Event(
                    type=event_type,
                    session_id=session.id,
                    interaction_id=interaction_id,
                    timestamp=observed_at,
                    agent_name=agent_name,
                    environment_name=environment_name,
                    payload=event_payload,
                )
                if event_id is None
                else Event(
                    id=event_id,
                    type=event_type,
                    session_id=session.id,
                    interaction_id=interaction_id,
                    timestamp=observed_at,
                    agent_name=agent_name,
                    environment_name=environment_name,
                    payload=event_payload,
                ),
                "session_id",
                "interaction_id",
            ),
            "start_event_id",
        )
        if event_type == EventType.INTERACTION_RESUMED:
            _clear_session_interaction_recovered_active_through(session.id)
        return event

    async def failure_interaction_observed_at(
        self,
        session: Session,
        *,
        observed_at: datetime | None = None,
    ) -> datetime | None:
        interaction_id = _current_session_interaction_id(session.id)
        if interaction_id is None:
            return None
        records = await self.session_store.query_events(
            EventQuery(
                session_id=session.id,
                interaction_id=interaction_id,
                event_types=(EventType.INTERACTION_STARTED,),
                order_by=EventOrder.SEQUENCE_ASC,
                limit=1,
            )
        )
        if not records:
            return None
        start_evidence = InteractionSummaryEvidence.model_validate(records[0].event.payload)
        if observed_at is None:
            observed_at = self._clock()
        if observed_at < start_evidence.started_at:
            raise _runtime_interaction_lifecycle_publication_rejected(
                session_id=session.id,
                interaction_id=interaction_id,
                runtime_authority=self.interaction_publication_authority,
            )
        return observed_at

    async def emit_interaction_state(
        self,
        *,
        session: Session,
        registered_agent: runtime_records.RegisteredAgentState,
        environment_name: str | None,
        event_type: EventType,
        status: InteractionStatus,
        pending_action_kind: str | None = None,
        recovered_active_through: datetime | None = None,
    ) -> Event | None:
        event = await self._prepare_interaction_state_event(
            session=session,
            agent_name=registered_agent.spec.name,
            environment_name=environment_name,
            event_type=event_type,
            status=status,
            pending_action_kind=pending_action_kind,
            recovered_active_through=recovered_active_through,
        )
        if event is None:
            return None
        return await self._event_writer.emit(event)

    async def _transition_status_under_terminal_finalization_claim(
        self,
        *,
        session: Session,
        from_statuses: set[SessionStatus],
        to_status: SessionStatus,
        claim_id: str,
    ) -> Session:
        """Atomically fence a status-only transition by its exact live claim."""

        def retain_exact_claim(
            current_session: Session,
            checkpoint: dict[str, Any] | None,
            store_now: datetime,
        ) -> dict[str, Any]:
            if (
                current_session.instance_id != session.instance_id
                or current_session.run_epoch != session.run_epoch
                or checkpoint is None
            ):
                raise SessionRunFenced(
                    "Status-only interaction transition lost its exact session authority."
                )
            claim = _incomplete_recovery_claim_from_checkpoint(checkpoint)
            if claim is None or claim[0] != claim_id or claim[1] <= store_now:
                raise SessionRunFenced(
                    "Status-only interaction transition lost its exact terminal recovery claim."
                )
            return copy_durable_record(checkpoint, "checkpoint")

        return await self.session_store.transition_status_and_checkpoint(
            session.id,
            from_statuses=from_statuses,
            to_status=to_status,
            store_time_checkpoint_transform=retain_exact_claim,
        )

    async def stop_at_requested_tool_round_boundary(self, session: Session) -> None:
        """Promote accepted steering only from the live owner's settled boundary.

        Admission never signals cancellation. The same durable interaction key
        survives recovery epoch changes, but cannot name a later interaction.
        Pending approvals, user input, or tool recovery remain authoritative.
        """

        from cayu.runtime._session_steering import (
            steering_operation_key,
            steering_receipt_from_record,
        )
        from cayu.runtime.session_steering import SessionSteeringConflict

        checkpoint = await self.session_store.load_checkpoint(session.id)
        profile = active_invocation_execution_profile_from_checkpoint(checkpoint)
        if profile is None:
            return
        key = steering_operation_key(session.instance_id, profile.interaction_id)
        stored = await self.session_store.load_session_operation(session.id, key)
        if stored is None:
            return
        receipt = steering_receipt_from_record(stored)
        if receipt is None:
            return
        request = receipt.request
        if (
            request.session_id != session.id
            or request.session_instance_id != session.instance_id
            or request.interaction_id != profile.interaction_id
            or request.expected_run_epoch > session.run_epoch
            or receipt.execution_profile_fingerprint != profile.profile.fingerprint
            or profile.run_epoch != session.run_epoch
        ):
            raise SessionSteeringConflict()
        if _pending_interaction_action_kind(checkpoint, run_epoch=session.run_epoch) is not None:
            return
        if await self.session_store.load_active_model_completion_stage(session.id) is not None:
            # A recovered provider operation still owns its outcome. Steering
            # must not replace that reconciliation with a new terminal result.
            return
        payload = {
            "interruption_type": _INTERRUPTION_TYPE_OPERATOR_REQUESTED,
            "interruption_request_id": key,
            "reason": "Stopped at the requested complete tool-round boundary.",
        }
        prepare_interrupt = _checkpoint_with_pending_session_interrupt(
            payload, include_interruption_cascade=False
        )

        def promote(current: Session, current_checkpoint: dict[str, Any] | None) -> dict[str, Any]:
            current_profile = active_invocation_execution_profile_from_checkpoint(
                current_checkpoint
            )
            if (
                current.instance_id != session.instance_id
                or current.run_epoch != session.run_epoch
                or current_profile != profile
                or invocation_terminal_decision_from_checkpoint(current_checkpoint) is not None
                or _pending_interaction_action_kind(current_checkpoint, run_epoch=current.run_epoch)
                is not None
            ):
                raise SessionSteeringConflict()
            return prepare_interrupt(current, current_checkpoint)

        try:
            with _invocation_lifecycle_authority_mutation_scope():
                await self.session_store.transition_status_and_checkpoint(
                    session.id,
                    from_statuses={SessionStatus.RUNNING},
                    to_status=SessionStatus.INTERRUPTING,
                    checkpoint_transform=promote,
                    require_no_active_model_completion_dispatch=True,
                )
        except Exception:
            # An ordinary interruption may win the race, or publication may
            # commit before losing its acknowledgement. Neither permits a
            # generic failure to replace the durable interruption owner.
            await self._session_control.raise_if_interrupted(session.id)
            raise
        raise SessionInterruptedByRequest(session.id)

    async def promote_rejected_completion_stop(self, session: Session) -> None:
        """Let the existing interruption owner settle an atomic stop rejection."""

        try:
            await self.stop_at_requested_tool_round_boundary(session)
        except SessionInterruptedByRequest:
            return
        raise SessionRunFenced("Safe steering did not establish its terminal boundary.")

    async def ensure_interruption_terminal_decision(
        self,
        *,
        session: Session,
        terminal_payload: dict[str, Any],
        interruption_request_id: str,
    ) -> InvocationTerminalDecision | None:
        """Elect interruption after any earlier provider dispatch has quiesced."""

        checkpoint = await self.session_store.load_checkpoint(session.id)
        active_profile = active_invocation_execution_profile_from_checkpoint(checkpoint)
        existing = invocation_terminal_decision_from_checkpoint(checkpoint)
        settled = settled_invocation_terminal_decision_from_checkpoint(checkpoint)
        if active_profile is None:
            raise SessionRunFenced(
                "Interrupted invocation has no active execution-profile authority."
            )
        if existing is not None:
            if (
                existing.outcome is not InvocationTerminalOutcome.INTERRUPTED
                or existing.interruption_request_id != interruption_request_id
                or not invocation_terminal_decision_matches_active_profile(
                    existing,
                    session_id=session.id,
                    session_instance_id=session.instance_id,
                    run_epoch=session.run_epoch,
                    interaction_id=active_profile.interaction_id,
                    execution_profile_fingerprint=active_profile.profile.fingerprint,
                )
            ):
                raise SessionRunFenced(
                    "Interrupted invocation has a conflicting terminal decision."
                )
            return existing
        if settled is not None:
            if (
                settled.outcome is not InvocationTerminalOutcome.INTERRUPTED
                or settled.interruption_request_id != interruption_request_id
                or not invocation_terminal_decision_matches_recovery_profile(
                    settled,
                    session_id=session.id,
                    session_instance_id=session.instance_id,
                    current_run_epoch=session.run_epoch,
                    interaction_id=active_profile.interaction_id,
                    execution_profile_fingerprint=active_profile.profile.fingerprint,
                )
            ):
                raise SessionRunFenced(
                    "Interrupted invocation has a conflicting settled terminal decision."
                )
            return settled
        if (
            session.status is not SessionStatus.INTERRUPTING
            or active_profile.session_id != session.id
            or active_profile.run_epoch != session.run_epoch
        ):
            raise SessionRunFenced(
                "Interrupted invocation lost its exact session/profile authority."
            )
        latest_records = await self.session_store.query_events(
            EventQuery(
                session_id=session.id,
                event_types=INTERACTION_LIFECYCLE_EVENT_TYPES,
                order_by=EventOrder.SEQUENCE_DESC,
                limit=1,
            )
        )
        if not latest_records:
            raise SessionRunFenced("Interrupted invocation has no open interaction to terminalize.")
        latest = latest_records[0].event
        if latest.interaction_id is None:
            raise SessionRuntimePublicationConflict(
                "Interrupted invocation interaction has no durable identity."
            )
        run_operation = _session_run_operation_from_checkpoint(checkpoint)
        if run_operation is not None and run_operation.run_epoch != session.run_epoch:
            raise SessionRunFenced(
                "Interruption terminal decision lost its queued run-operation epoch."
            )
        if run_operation is not None:
            terminal_payload = {
                **terminal_payload,
                _SESSION_RUN_OPERATION_ID_PAYLOAD_KEY: run_operation.operation_id,
            }
        observed_at = self._clock()
        decision = build_invocation_terminal_decision(
            outcome=InvocationTerminalOutcome.INTERRUPTED,
            session_id=session.id,
            session_instance_id=session.instance_id,
            run_epoch=session.run_epoch,
            profile_interaction_id=active_profile.interaction_id,
            interaction_id=latest.interaction_id,
            execution_profile_fingerprint=active_profile.profile.fingerprint,
            interaction_event_id=(
                None
                if latest.type in INTERACTION_TERMINAL_EVENT_TYPES
                else invocation_terminal_event_id(
                    outcome=InvocationTerminalOutcome.INTERRUPTED,
                    session_id=session.id,
                    session_instance_id=session.instance_id,
                    run_epoch=session.run_epoch,
                    interaction_id=latest.interaction_id,
                    source_id=interruption_request_id,
                    event_kind="interaction",
                )
            ),
            predecessor_interaction_event_id=(
                latest.id if latest.type in INTERACTION_TERMINAL_EVENT_TYPES else None
            ),
            terminal_event_id=(
                run_operation.terminal_event_id
                if run_operation is not None and run_operation.terminal_event_id is not None
                else invocation_terminal_event_id(
                    outcome=InvocationTerminalOutcome.INTERRUPTED,
                    session_id=session.id,
                    session_instance_id=session.instance_id,
                    run_epoch=session.run_epoch,
                    interaction_id=latest.interaction_id,
                    source_id=interruption_request_id,
                    event_kind="session",
                )
            ),
            observed_at=observed_at,
            terminal_payload=terminal_payload,
            interruption_request_id=interruption_request_id,
        )

        def install_decision(
            current_session: Session,
            current_checkpoint: dict[str, Any] | None,
        ) -> dict[str, Any]:
            current_profile = active_invocation_execution_profile_from_checkpoint(
                current_checkpoint
            )
            if (
                current_session.instance_id != session.instance_id
                or current_session.run_epoch != session.run_epoch
                or current_session.status is not SessionStatus.INTERRUPTING
                or current_profile != active_profile
                or _session_run_operation_from_checkpoint(current_checkpoint) != run_operation
            ):
                raise SessionRunFenced(
                    "Interrupted invocation changed before terminal decision election."
                )
            return checkpoint_with_invocation_terminal_decision(
                current_checkpoint,
                decision,
            )

        with _invocation_lifecycle_authority_mutation_scope():
            await self.session_store.transition_status_and_checkpoint(
                session.id,
                from_statuses={SessionStatus.INTERRUPTING},
                to_status=SessionStatus.INTERRUPTING,
                checkpoint_transform=install_decision,
                expected_latest_interaction_event_id=latest.id,
            )
        return decision

    async def publish_closed_interaction_terminal_decision(
        self,
        *,
        session: Session,
        decision: InvocationTerminalDecision,
        terminal_event: Event,
    ) -> tuple[Session, Event]:
        """Publish a session terminal after its interaction already closed.

        The terminal event and winner settlement commit before terminal status
        becomes observable. A restart between those transactions replays the
        settled winner and completes only the remaining status transition.
        """

        if (
            decision.outcome is not InvocationTerminalOutcome.INTERRUPTED
            or decision.interaction_event_id is not None
            or decision.predecessor_interaction_event_id is None
            or terminal_event.id != decision.terminal_event_id
        ):
            raise ValueError("Closed-interaction publication has invalid decision authority.")

        def settle_decision(
            current_session: Session,
            checkpoint: dict[str, Any] | None,
        ) -> dict[str, Any] | None:
            active_profile = active_invocation_execution_profile_from_checkpoint(checkpoint)
            if (
                current_session.instance_id != decision.session_instance_id
                or current_session.run_epoch != decision.run_epoch
                or current_session.status is not SessionStatus.INTERRUPTING
                or active_profile is None
                or not invocation_terminal_decision_matches_recovery_profile(
                    decision,
                    session_id=current_session.id,
                    session_instance_id=current_session.instance_id,
                    current_run_epoch=current_session.run_epoch,
                    interaction_id=active_profile.interaction_id,
                    execution_profile_fingerprint=active_profile.profile.fingerprint,
                )
            ):
                raise SessionRunFenced(
                    "Closed-interaction terminal publication lost invocation authority."
                )
            return checkpoint_after_invocation_terminal_decision(
                checkpoint,
                expected=decision,
            )

        checkpoint = await self.session_store.load_checkpoint(session.id)
        settled = settled_invocation_terminal_decision_from_checkpoint(checkpoint)
        if settled is None:
            with _invocation_lifecycle_authority_mutation_scope():
                try:
                    await self.session_store.publish_checkpoint_and_events(
                        session.id,
                        checkpoint_transform=settle_decision,
                        events=[terminal_event],
                        expected_statuses={SessionStatus.INTERRUPTING},
                        expected_run_epoch=decision.run_epoch,
                    )
                except Exception:
                    checkpoint = await self.session_store.load_checkpoint(session.id)
                    if settled_invocation_terminal_decision_from_checkpoint(checkpoint) != decision:
                        raise
            settled = decision
        if settled != decision:
            raise SessionRunFenced(
                "Closed-interaction terminal publication has a conflicting settled winner."
            )
        durable_event = await self._terminal_event_publication.reconcile_persisted_terminal_event(
            terminal_event
        )
        if durable_event is None:
            raise SessionRuntimePublicationConflict(
                "Settled closed-interaction winner has no terminal event."
            )

        def retain_settled_decision(
            current_session: Session,
            current_checkpoint: dict[str, Any] | None,
        ) -> dict[str, Any] | None:
            if (
                current_session.instance_id != decision.session_instance_id
                or current_session.run_epoch != decision.run_epoch
                or settled_invocation_terminal_decision_from_checkpoint(current_checkpoint)
                != decision
            ):
                raise SessionRunFenced(
                    "Closed-interaction terminal status lost its settled winner."
                )
            return current_checkpoint

        current = await self.session_store.load(session.id)
        if current is None:
            raise KeyError(f"Session not found: {session.id}")
        if current.status is SessionStatus.INTERRUPTING:
            with _invocation_lifecycle_authority_mutation_scope():
                try:
                    current = await self.session_store.transition_status_and_checkpoint(
                        session.id,
                        from_statuses={SessionStatus.INTERRUPTING},
                        to_status=SessionStatus.INTERRUPTED,
                        checkpoint_transform=retain_settled_decision,
                        expected_latest_interaction_event_id=(
                            decision.predecessor_interaction_event_id
                        ),
                    )
                except Exception:
                    reconciled = await self.session_store.load(session.id)
                    reconciled_checkpoint = await self.session_store.load_checkpoint(session.id)
                    if (
                        reconciled is None
                        or reconciled.instance_id != decision.session_instance_id
                        or reconciled.run_epoch != decision.run_epoch
                        or reconciled.status is not SessionStatus.INTERRUPTED
                        or settled_invocation_terminal_decision_from_checkpoint(
                            reconciled_checkpoint
                        )
                        != decision
                    ):
                        raise
                    current = reconciled
        elif current.status is not SessionStatus.INTERRUPTED:
            raise SessionRunFenced(
                "Closed-interaction terminal publication found a conflicting session status."
            )
        await self._event_writer.fan_out_persisted([durable_event])
        _mark_session_invocation_terminal_event(durable_event)
        return current, durable_event

    async def publish_interaction_transition(
        self,
        *,
        session: Session,
        invocation_context: InvocationContext | None = None,
        allow_released_invocation_authority: bool = False,
        agent_name: str,
        environment_name: str | None,
        to_status: SessionStatus,
        only_if_no_queued_messages: bool = False,
        from_statuses: set[SessionStatus] | None = None,
        recovered_active_through: datetime | None = None,
        observed_at: datetime | None = None,
        event_id: str | None = None,
        execution_profile: ExecutionProfileIdentity | None = None,
        model_completion_failure: BaseException | None = None,
        expected_recovery_claim_id: str | None = None,
        checkpoint_mutation: dict[str, Any] | None = None,
        terminal_event: Event | None = None,
        terminal_decision: InvocationTerminalDecision | None = None,
        tool_completion_result: ToolCompletionResult | None = None,
    ) -> tuple[Session, Event | None, bool]:
        if invocation_context is not None:
            if type(invocation_context) is not InvocationContext or not isinstance(
                invocation_context.binding,
                AdmittedInvocationBinding,
            ):
                raise TypeError(
                    "Interaction transition requires an authenticated admitted context."
                )
            binding = invocation_context.binding
            if (
                binding.session_id != session.id
                or binding.session_instance_id != session.instance_id
                or binding.run_epoch != session.run_epoch
                or binding.agent_name != agent_name
                or binding.environment_name != environment_name
            ):
                raise SessionRunFenced("Interaction transition lost its frozen invocation binding.")
            if (
                execution_profile is not None
                and execution_profile is not invocation_context.profile
            ):
                raise RuntimeError(
                    "Interaction transition substituted its validated execution profile."
                )
            execution_profile = invocation_context.profile
            context_recovery_claim_id = invocation_context.recovery_claim_id
            if expected_recovery_claim_id is None:
                expected_recovery_claim_id = context_recovery_claim_id
            elif (
                context_recovery_claim_id is not None
                and expected_recovery_claim_id != context_recovery_claim_id
            ):
                raise SessionRunFenced(
                    "Interaction transition substituted its recovery claim authority."
                )
        interaction_id = _current_session_interaction_id(session.id)
        if terminal_decision is not None:
            if interaction_id != terminal_decision.interaction_id:
                interaction_id = await self._activate_terminal_decision_interaction(
                    terminal_decision
                )
            if interaction_id != terminal_decision.interaction_id:
                raise SessionRunFenced(
                    "Terminal interaction publication changed its interaction authority."
                )
            # Offline recovery and a losing sibling path can enter without the
            # winner's process-local interaction state. Reconstruct mismatched
            # local state only from the durable open interaction and require it
            # to match the authenticated decision.
        active_model_completion = await self.session_store.load_active_model_completion_stage(
            session.id
        )
        model_completion_settlement: ModelCompletionStageSettlementRequest | None = None
        active_provider_operation = (
            None
            if active_model_completion is None
            else active_model_completion.stage.intent.get("provider_operation_start")
        )
        provider_operation_owned = type(active_provider_operation) is dict
        model_completion_dispatched = False
        if active_model_completion is not None and not provider_operation_owned:
            try:
                provider_operation_owned = (
                    await load_recoverable_provider_operation(
                        self.session_store,
                        active_model_completion.stage,
                    )
                    is not None
                )
            except ProviderOperationEvidenceError:
                provider_operation_owned = True
        if active_model_completion is not None and not provider_operation_owned:
            model_completion_dispatched = (
                await self.session_store.load_model_completion_stage_dispatch(
                    session.id,
                    active_model_completion.stage.stage_id,
                )
                is not None
            )
        if active_model_completion is not None and not provider_operation_owned:
            recovery_context = model_completion_recovery_context_from_stage(
                active_model_completion.stage
            )
            await (
                self._run_limit_controller.reconcile_borrowed_automatic_compaction_budget_authority(
                    session=session,
                    stage=active_model_completion.stage,
                )
            )
            budget_recovery_contexts = (
                () if recovery_context is None else recovery_context.budget_reservations
            )
            if active_model_completion.stage.purpose == "auxiliary-inference":
                from cayu.runtime._auxiliary_inference_contract import (
                    auxiliary_budget_recovery_contexts,
                )

                budget_recovery_contexts = auxiliary_budget_recovery_contexts(
                    active_model_completion.stage
                )
            budget_dispatch_id = active_model_completion.stage.stage_id
            if active_model_completion.stage.purpose in {
                "context-compaction",
                "auxiliary-inference",
            }:
                model_attempt_id = active_model_completion.stage.intent.get("model_attempt_id")
                if type(model_attempt_id) is not str:
                    raise SessionModelCompletionStageConflict(
                        "Model-stage terminalization lost its budget dispatch identity."
                    )
                budget_dispatch_id = require_clean_nonblank(
                    model_attempt_id,
                    "model_attempt_id",
                )
            if interaction_id is None:
                raise SessionModelCompletionStageConflict(
                    "An active model-completion stage has no active interaction identity."
                )
            if to_status is SessionStatus.COMPLETED:
                raise SessionModelCompletionStageConflict(
                    "A session cannot complete while a model-completion stage is active."
                )
            if to_status not in {SessionStatus.FAILED, SessionStatus.INTERRUPTED}:
                raise SessionModelCompletionStageConflict(
                    "An active model-completion stage requires a terminal disposition."
                )
            provider_rejected_before_effect = (
                model_completion_failure is not None
                and _provider_failure_proves_no_model_effect(model_completion_failure)
            )
            failed_before_effect = (
                not model_completion_dispatched or provider_rejected_before_effect
            )
            if active_model_completion.stage.state == "in_flight":
                if failed_before_effect:
                    await close_context_exposure_without_provider_effect(
                        store=self.session_store,
                        session_id=session.id,
                        stage_id=active_model_completion.stage.stage_id,
                        stage_intent=active_model_completion.stage.intent,
                        evidence_ref_suffix="provider-effect-absent",
                    )
                else:
                    await close_unrecoverable_context_exposure(
                        store=self.session_store,
                        session_id=session.id,
                        stage_id=active_model_completion.stage.stage_id,
                        stage_intent=active_model_completion.stage.intent,
                    )
                if not model_completion_dispatched:
                    await self._run_limit_controller.release_pre_provider_dispatch_reservations(
                        reservation_ids=active_model_completion.stage.reservation_ids,
                        recovery_contexts=budget_recovery_contexts,
                        dispatch_id=budget_dispatch_id,
                    )
            await self._run_limit_controller.require_model_completion_reservation_settlements(
                reservation_ids=active_model_completion.stage.reservation_ids,
                recovery_contexts=budget_recovery_contexts,
                dispatch_id=budget_dispatch_id,
            )
            # Accepted completions still require exact accounting settlement,
            # but retain their publication evidence instead of receiving an
            # in-flight provider-effect disposition.
            if active_model_completion.stage.state == "in_flight":
                model_completion_settlement = model_completion_stage_settlement_request(
                    active_model_completion.stage,
                    interaction_id=interaction_id,
                    disposition=(
                        ModelCompletionStageDisposition.FAILED_BEFORE_PROVIDER_EFFECT
                        if failed_before_effect
                        else ModelCompletionStageDisposition.PROVIDER_EFFECT_OUTCOME_UNKNOWN
                    ),
                    reason_code=(
                        "provider_authentication_failed"
                        if provider_rejected_before_effect
                        else (
                            "model_attempt_failed"
                            if to_status is SessionStatus.FAILED
                            else "model_attempt_interrupted"
                        )
                    ),
                    execution_profile_fingerprint=(
                        None if execution_profile is None else execution_profile.fingerprint
                    ),
                    settlement_run_epoch=session.run_epoch,
                    settled_reservation_ids=active_model_completion.stage.reservation_ids,
                )
        if interaction_id is None:
            if expected_recovery_claim_id is not None and only_if_no_queued_messages:
                raise ValueError(
                    "A context-free terminal recovery claim cannot guard a "
                    "queue-conditional transition."
                )
            if expected_recovery_claim_id is not None:
                transitioned = await self._transition_status_under_terminal_finalization_claim(
                    session=session,
                    from_statuses={session.status},
                    to_status=to_status,
                    claim_id=expected_recovery_claim_id,
                )
            else:
                transitioned = (
                    await self.session_store.transition_status_if_no_queued_messages(
                        session.id,
                        from_statuses={SessionStatus.RUNNING},
                        to_status=to_status,
                    )
                    if only_if_no_queued_messages
                    else await self.session_store.update_status(session.id, to_status)
                )
            return transitioned, None, True

        pending_action_kind: str | None = None
        if to_status is not SessionStatus.COMPLETED and terminal_decision is None:
            checkpoint = await self.session_store.load_checkpoint(session.id)
            pending_action_kind = _pending_interaction_action_kind(
                checkpoint, run_epoch=session.run_epoch
            )
            if (
                pending_action_kind is None
                and to_status is SessionStatus.INTERRUPTED
                and provider_operation_owned
            ):
                # A locally interrupted invocation can retain remote work.
                # Keep its interaction available for exact operation recovery
                # until that work and its original accounting have settled.
                pending_action_kind = "provider_operation_recovery"
        completion_pending_finalization = (
            to_status is SessionStatus.RUNNING and checkpoint_mutation is not None
        )
        if to_status is SessionStatus.COMPLETED or completion_pending_finalization:
            producer_publication = (
                model_completion_publication.model_step_publication_from_checkpoint(
                    await self.session_store.load_checkpoint(session.id)
                )
            )
            if producer_publication is not None:
                await self.session_store._retain_native_producer_output(
                    session.id,
                    invocation=invocation_context,
                    stage_id=producer_publication.stage_id,
                )
        if to_status is SessionStatus.COMPLETED or completion_pending_finalization:
            event_type = EventType.INTERACTION_COMPLETED
            interaction_status = InteractionStatus.COMPLETED
        elif pending_action_kind is not None:
            event_type = EventType.INTERACTION_PAUSED
            interaction_status = InteractionStatus.PAUSED
        elif to_status is SessionStatus.FAILED:
            event_type = EventType.INTERACTION_FAILED
            interaction_status = InteractionStatus.FAILED
        elif to_status is SessionStatus.INTERRUPTED:
            event_type = EventType.INTERACTION_INTERRUPTED
            interaction_status = InteractionStatus.INTERRUPTED
        else:
            raise ValueError(f"Unsupported interaction terminal session status: {to_status}")

        event = await self._prepare_interaction_state_event(
            session=session,
            agent_name=agent_name,
            environment_name=environment_name,
            event_type=event_type,
            status=interaction_status,
            settlement_status=to_status,
            tool_completion_result=tool_completion_result,
            pending_action_kind=pending_action_kind,
            recovered_active_through=recovered_active_through,
            observed_at=observed_at,
            event_id=event_id,
        )
        if event is None:
            loaded = await self.session_store.load(session.id)
            if loaded is None:
                raise KeyError(f"Session not found: {session.id}") from None
            if loaded.status is to_status:
                if expected_recovery_claim_id is None:
                    return loaded, None, True
                transitioned = await self._transition_status_under_terminal_finalization_claim(
                    session=session,
                    from_statuses={to_status},
                    to_status=to_status,
                    claim_id=expected_recovery_claim_id,
                )
                return transitioned, None, True
            if (
                from_statuses is None
                and to_status is SessionStatus.FAILED
                and loaded.status in {SessionStatus.COMPLETED, SessionStatus.INTERRUPTED}
            ):
                # Environment finalization or a terminal hook can fail after the
                # response interaction and its original session outcome are
                # already durable. Keep that interaction terminal evidence
                # truthful while recording the later session-scoped failure.
                if expected_recovery_claim_id is not None:
                    transitioned = await self._transition_status_under_terminal_finalization_claim(
                        session=session,
                        from_statuses={loaded.status},
                        to_status=SessionStatus.FAILED,
                        claim_id=expected_recovery_claim_id,
                    )
                else:
                    transitioned = await self.session_store.transition_status(
                        session.id,
                        from_statuses={loaded.status},
                        to_status=SessionStatus.FAILED,
                    )
                return transitioned, None, True
            allowed_statuses = (
                {SessionStatus.RUNNING, SessionStatus.INTERRUPTING}
                if from_statuses is None
                else from_statuses
            )
            if expected_recovery_claim_id is not None:
                transitioned = await self._transition_status_under_terminal_finalization_claim(
                    session=session,
                    from_statuses=allowed_statuses,
                    to_status=to_status,
                    claim_id=expected_recovery_claim_id,
                )
            else:
                transitioned = (
                    await self.session_store.transition_status_if_no_queued_messages(
                        session.id,
                        from_statuses=allowed_statuses,
                        to_status=to_status,
                    )
                    if only_if_no_queued_messages
                    else await self.session_store.transition_status(
                        session.id,
                        from_statuses=allowed_statuses,
                        to_status=to_status,
                    )
                )
            return transitioned, None, True

        replay_event = copy_event(event)
        from cayu.runtime._session_steering import SessionSteeringBoundaryReached

        replay_from_statuses = frozenset(
            {SessionStatus.RUNNING, SessionStatus.INTERRUPTING}
            if from_statuses is None
            else from_statuses
        )
        replay_transition = InteractionTransitionSpec(
            event=replay_event,
            from_statuses=tuple(replay_from_statuses),
            to_status=to_status,
            only_if_no_queued_messages=only_if_no_queued_messages,
            model_completion_stage_settlement=model_completion_settlement,
            checkpoint_mutation=checkpoint_mutation,
            terminal_event=terminal_event,
            terminal_decision=terminal_decision,
        )
        if terminal_event is not None:
            self.require_terminal_interaction_publication_protocol()
        active_invocation_profile = (
            invocation_context.active_profile
            if invocation_context is not None
            else active_invocation_execution_profile_from_checkpoint(
                await self.session_store.load_checkpoint(session.id)
            )
        )
        settlement_command: SettleInvocationCommand | None = None
        if active_invocation_profile is not None:
            if (
                execution_profile is not None
                and active_invocation_profile.profile != execution_profile
            ):
                raise SessionRunFenced(
                    "Interaction settlement lost its active invocation authority."
                )
            released_invocation_authority = (
                invocation_context is None
                and allow_released_invocation_authority
                and active_invocation_execution_profile_is_released(
                    active_invocation_profile,
                    session_id=session.id,
                    run_epoch=session.run_epoch,
                )
            )
            if released_invocation_authority:
                # The predecessor epoch already completed its typed release.
                # Terminal recovery may repair the interaction event only
                # while the exact released profile and successor epoch remain
                # authoritative at the store's atomic publication boundary.
                settlement_command = SettleInvocationCommand(
                    session_id=session.id,
                    expected_session_instance_id=session.instance_id,
                    expected_run_epoch=session.run_epoch,
                    expected_active_profile=active_invocation_profile,
                    expected_authority_state="released",
                    transition=replay_transition,
                )
            elif (
                active_invocation_profile.session_id != session.id
                or active_invocation_profile.run_epoch != session.run_epoch
            ):
                raise SessionRunFenced(
                    "Interaction settlement lost its active invocation authority."
                )
            if not released_invocation_authority:
                settlement_command = SettleInvocationCommand(
                    session_id=session.id,
                    expected_session_instance_id=session.instance_id,
                    expected_run_epoch=session.run_epoch,
                    expected_active_profile=active_invocation_profile,
                    transition=replay_transition,
                )
        result: InteractionTransitionResult | None = None
        pending_cancellation: asyncio.CancelledError | None = None
        replay_failures: list[Exception] = []
        replay_deadline: float | None = None
        loop = asyncio.get_running_loop()
        for attempt in range(_INTERACTION_TRANSITION_REPLAY_MAX_ATTEMPTS):
            if settlement_command is not None and expected_recovery_claim_id is None:
                transition_awaitable = self.session_store.apply_invocation_lifecycle_command(
                    settlement_command
                )
            else:
                transition_kwargs: dict[str, Any] = {
                    "event": copy_event(replay_transition.event),
                    "from_statuses": set(replay_transition.from_statuses),
                    "to_status": replay_transition.to_status,
                    "only_if_no_queued_messages": replay_transition.only_if_no_queued_messages,
                    "model_completion_stage_settlement": (
                        replay_transition.model_completion_stage_settlement
                    ),
                    "terminal_event": replay_transition.terminal_event,
                    "terminal_decision": replay_transition.terminal_decision,
                    "expected_recovery_claim_id": expected_recovery_claim_id,
                }
                if replay_transition.checkpoint_mutation is not None:
                    transition_kwargs["checkpoint_mutation"] = replay_transition.checkpoint_mutation
                if settlement_command is not None:
                    transition_kwargs.update(
                        {
                            "expected_session_instance_id": (
                                settlement_command.expected_session_instance_id
                            ),
                            "expected_active_invocation_profile": (
                                settlement_command.expected_active_profile
                            ),
                            "expected_invocation_authority_state": (
                                settlement_command.expected_authority_state
                            ),
                        }
                    )
                transition_awaitable = self.session_store.publish_interaction_transition(
                    session.id,
                    **transition_kwargs,
                )
            transition_task = asyncio.create_task(transition_awaitable)
            outcome = await await_shielded_task_outcome(transition_task)
            attempt_failure = _normalize_interaction_transition_child_cancellation(outcome.error)
            attempt_result: InteractionTransitionResult | None = None
            if attempt_failure is None and outcome.result is not None:
                try:
                    attempt_result = _validated_interaction_transition_result(
                        outcome.result,
                        transition=replay_transition,
                    )
                except (RuntimeError, TypeError) as result_failure:
                    attempt_failure = result_failure
            if outcome.cancellation is not None:
                if attempt_failure is not None:
                    if isinstance(attempt_failure, SessionSteeringBoundaryReached):
                        await _run_interaction_transition_cancellation_cleanup_steps(
                            outcome.cancellation,
                            supervisor=self._recovery_cleanup_supervisor,
                            steps=(
                                (
                                    "cooperative stop promotion after completion rejection",
                                    lambda: self.promote_rejected_completion_stop(session),
                                ),
                            ),
                        )
                    cancellation_failures = [*replay_failures, attempt_failure]
                    _raise_interaction_transition_cancellation(
                        outcome.cancellation,
                        cancellation_failures,
                        failure_diagnostics=_interaction_transition_failure_diagnostics(
                            tuple(cancellation_failures),
                            redactor=self._secret_redactor,
                        ),
                        transition=replay_transition,
                    )
                if attempt_result is None:
                    cancellation_failures = [
                        *replay_failures,
                        RuntimeError("Interaction transition publication returned no result."),
                    ]
                    _raise_interaction_transition_cancellation(
                        outcome.cancellation,
                        cancellation_failures,
                        failure_diagnostics=_interaction_transition_failure_diagnostics(
                            tuple(cancellation_failures),
                            redactor=self._secret_redactor,
                        ),
                        transition=replay_transition,
                    )
                result = attempt_result
                pending_cancellation = outcome.cancellation
                break
            if attempt_failure is None:
                if attempt_result is None:
                    attempt_failure = RuntimeError(
                        "Interaction transition publication returned no result."
                    )
                else:
                    result = attempt_result
                    break
            if not isinstance(attempt_failure, Exception):
                if replay_failures:
                    raise BaseExceptionGroup(
                        "Interaction transition replay and fatal store failure.",
                        _unique_exception_identities([*replay_failures, attempt_failure]),
                    ) from None
                raise attempt_failure
            prior_failures = list(replay_failures)
            replay_failures.append(attempt_failure)
            if _interaction_transition_failure_contains_run_fence(attempt_failure):
                _mark_interaction_transition_run_fence(attempt_failure)
                if prior_failures:
                    _raise_primary_with_secondary_failure(
                        attempt_failure,
                        _interaction_transition_replay_failure(prior_failures),
                        group_message=("Interaction transition replay failure and run-fence loss."),
                    )
                raise attempt_failure
            terminal_failure = _interaction_transition_replay_failure(replay_failures)
            if isinstance(attempt_failure, SessionSteeringBoundaryReached):
                if prior_failures:
                    _raise_primary_with_secondary_failure(
                        attempt_failure,
                        _interaction_transition_replay_failure(prior_failures),
                        group_message="Cooperative stop rejection and prior publication failures.",
                    )
                raise attempt_failure
            if _interaction_transition_replay_is_authoritatively_rejected(attempt_failure):
                raise terminal_failure from None
            if attempt + 1 >= _INTERACTION_TRANSITION_REPLAY_MAX_ATTEMPTS:
                raise terminal_failure from None
            if replay_deadline is None:
                replay_deadline = loop.time() + _INTERACTION_TRANSITION_REPLAY_WINDOW_SECONDS
            if loop.time() >= replay_deadline:
                raise terminal_failure from None
            # Give cancellation and a competing run-fence owner a boundary
            # before replay. The completed store task above is positively
            # settled before ownership cleanup or another attempt may begin.
            try:
                await asyncio.sleep(0)
            except asyncio.CancelledError as cancellation:
                _raise_interaction_transition_cancellation(
                    cancellation,
                    replay_failures,
                    failure_diagnostics=_interaction_transition_failure_diagnostics(
                        tuple(replay_failures),
                        redactor=self._secret_redactor,
                    ),
                    transition=replay_transition,
                )
            if loop.time() >= replay_deadline:
                raise terminal_failure from None
        if result is None:  # pragma: no cover - the bounded loop returns or raises
            raise AssertionError("Interaction transition replay produced no result.")
        if result.event.interaction_id is None:
            raise RuntimeError("Interaction transition result lost its interaction identity.")
        _mark_session_interaction_settled(
            session.id,
            result.event.interaction_id,
            replay_transition,
        )
        if result.event.type in INTERACTION_TERMINAL_EVENT_TYPES:
            _close_session_interaction(session.id)
            self._foreground_gate_policy_owner.release_interaction(
                session=result.session, interaction_id=result.event.interaction_id
            )
        if pending_cancellation is not None:
            # The atomic store publication already owns a durable side-effect
            # handoff. Redeliver cancellation at this settled boundary instead
            # of beginning extension-controlled budget or sink work.
            _raise_interaction_transition_cancellation(
                pending_cancellation,
                replay_failures,
                transition_settled=True,
                failure_diagnostics=_interaction_transition_failure_diagnostics(
                    tuple(replay_failures),
                    redactor=self._secret_redactor,
                ),
                transition=replay_transition,
            )
        persisted_events = [result.event]
        if result.terminal_event is not None:
            persisted_events.append(result.terminal_event)
            _mark_session_invocation_terminal_event(result.terminal_event)
        await self._event_writer.fan_out_persisted(persisted_events)
        return result.session, result.event, result.status_changed

    async def _reconcile_sibling_interaction_transition_cancellation(
        self,
        cancellation: asyncio.CancelledError,
        *,
        session: Session,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_environment: runtime_records.RegisteredEnvironment | None,
        environment_name: str | None,
        execution_profile: ExecutionProfileIdentity | None,
        invocation_context: InvocationContext | None,
        finalize_unsettled: bool,
        expected_recovery_claim_id: str | None,
    ) -> None:
        """Consume transition handoff evidence outside the main run loop."""

        if _is_interaction_transition_run_fence(cancellation):
            pop_exception_state(
                cancellation,
                _INTERACTION_TRANSITION_CANCELLATION_OUTCOME_ATTRIBUTE,
            )
            pop_exception_state(
                cancellation,
                _INTERACTION_TRANSITION_RUN_FENCE_ATTRIBUTE,
            )
            return

        transition_cancellation_outcome = _pop_interaction_transition_cancellation_outcome(
            cancellation
        )
        interaction_transition_failures: list[dict[str, Any]] = []
        interaction_transition: InteractionTransitionSpec | None = None
        if transition_cancellation_outcome is not None:
            (
                transition_settled,
                interaction_transition_failures,
                interaction_transition,
            ) = transition_cancellation_outcome
        else:
            transition_settled = False

        if not interaction_transition_failures and (
            transition_settled or _latest_session_invocation_interaction_is_settled(session.id)
        ):
            return
        if not interaction_transition_failures and not finalize_unsettled:
            return

        recovery_request = RecoveryAbandonedSessionRequest(
            session=session,
            registered_agent=registered_agent,
            registered_environment=registered_environment,
            environment_name=environment_name,
            interaction_transition_failures=tuple(interaction_transition_failures),
            interaction_transition=interaction_transition,
            interaction_transition_recovery_claim_id=expected_recovery_claim_id,
            execution_profile=execution_profile,
            invocation_context=invocation_context,
        )
        if interaction_transition_failures and interaction_transition is not None:
            committed_transition_recorded = False

            async def record_committed_transition_cancellation() -> None:
                nonlocal committed_transition_recorded
                committed_transition_recorded = (
                    await self.record_committed_interaction_transition_cancellation(
                        recovery_request
                    )
                )

            diagnostic_failures = await _run_interaction_transition_cancellation_cleanup_steps(
                cancellation,
                supervisor=self._recovery_cleanup_supervisor,
                steps=(
                    (
                        "committed sibling interaction-transition cancellation diagnostics",
                        record_committed_transition_cancellation,
                    ),
                ),
            )
            if committed_transition_recorded:
                interaction_id = interaction_transition.event.interaction_id
                if interaction_id is None:
                    raise RuntimeError(
                        "Committed interaction transition lost its interaction identity."
                    )
                _mark_session_interaction_settled(
                    session.id,
                    interaction_id,
                    interaction_transition,
                )
            if committed_transition_recorded or diagnostic_failures:
                return
        if transition_settled or _latest_session_invocation_interaction_is_settled(session.id):
            return
        if not finalize_unsettled:
            return

        async def finalize_unsettled_interaction() -> None:
            await self.finalize_abandoned_session_run(recovery_request)

        cancellation_cleanup_steps = (
            (
                "cancelled sibling interaction-transition finalization",
                finalize_unsettled_interaction,
            ),
        )
        if interaction_transition_failures:
            await _run_interaction_transition_cancellation_cleanup_steps(
                cancellation,
                steps=cancellation_cleanup_steps,
                supervisor=self._recovery_cleanup_supervisor,
            )
        else:
            await self._recovery_ownership.run_cleanup_steps(
                authoritative_failure=cancellation,
                steps=cancellation_cleanup_steps,
            )

    async def publish_sibling_interaction_transition(
        self,
        *,
        session: Session,
        invocation_context: InvocationContext | None = None,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_environment: runtime_records.RegisteredEnvironment | None,
        environment_name: str | None,
        to_status: SessionStatus,
        only_if_no_queued_messages: bool = False,
        from_statuses: set[SessionStatus] | None = None,
        recovered_active_through: datetime | None = None,
        observed_at: datetime | None = None,
        event_id: str | None = None,
        execution_profile: ExecutionProfileIdentity | None = None,
        model_completion_failure: BaseException | None = None,
        finalize_unsettled_cancellation: bool = True,
        expected_recovery_claim_id: str | None = None,
        terminal_event: Event | None = None,
        terminal_decision: InvocationTerminalDecision | None = None,
        allow_released_invocation_authority: bool = False,
    ) -> tuple[Session, Event | None, bool]:
        """Publish from a caller not enclosed by ``_run_session`` cleanup."""

        if invocation_context is not None and (
            registered_agent is not invocation_context.registered_agent
            or registered_environment is not invocation_context.registered_environment
        ):
            raise RuntimeError("Sibling interaction transition substituted a frozen collaborator.")
        try:
            return await self.publish_interaction_transition(
                session=session,
                invocation_context=invocation_context,
                agent_name=registered_agent.spec.name,
                environment_name=environment_name,
                to_status=to_status,
                only_if_no_queued_messages=only_if_no_queued_messages,
                from_statuses=from_statuses,
                recovered_active_through=recovered_active_through,
                observed_at=observed_at,
                event_id=event_id,
                execution_profile=execution_profile,
                allow_released_invocation_authority=allow_released_invocation_authority,
                model_completion_failure=model_completion_failure,
                expected_recovery_claim_id=expected_recovery_claim_id,
                terminal_event=terminal_event,
                terminal_decision=terminal_decision,
            )
        except asyncio.CancelledError as cancellation:
            await self._reconcile_sibling_interaction_transition_cancellation(
                cancellation,
                session=session,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
                environment_name=environment_name,
                execution_profile=execution_profile,
                invocation_context=invocation_context,
                finalize_unsettled=finalize_unsettled_cancellation,
                expected_recovery_claim_id=expected_recovery_claim_id,
            )
            raise

    async def resume_interaction(
        self,
        session: Session,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_environment: runtime_records.RegisteredEnvironment | None,
    ) -> Event | None:
        return await self.emit_interaction_state(
            session=session,
            registered_agent=registered_agent,
            environment_name=_environment_name(registered_environment),
            event_type=EventType.INTERACTION_RESUMED,
            status=InteractionStatus.ACTIVE,
        )

    async def fail_provider_operation_resolution(
        self,
        *,
        resolution_event: Event,
        session: Session,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_environment: runtime_records.RegisteredEnvironment | None,
        execution_profile: ExecutionProfileIdentity,
        task_id: str | None = None,
        task_worker_id: str | None = None,
        task_handoff_id: str | None = None,
        legacy_resolution_without_profile: bool = False,
        invocation_context: InvocationContext | None = None,
    ) -> AsyncIterator[Event]:
        """Terminalize one explicit provider failure through normal lifecycle hooks."""

        if invocation_context is not None and (
            invocation_context.binding.session_id != session.id
            or registered_agent is not invocation_context.registered_agent
            or registered_environment is not invocation_context.registered_environment
            or execution_profile is not invocation_context.profile
        ):
            raise RuntimeError(
                "Provider-operation failure substituted frozen invocation authority."
            )
        if resolution_event.type is not EventType.PROVIDER_OPERATION_RESOLVED:
            raise TypeError("Provider-operation failure requires a resolution event.")
        if resolution_event.session_id != session.id:
            raise ValueError("Provider-operation resolution belongs to another session.")
        if task_worker_id is not None and task_id is None:
            raise ValueError("Provider-operation task worker authority requires a task_id.")
        if task_handoff_id is not None and task_worker_id is None:
            raise ValueError("Provider-operation task handoff requires a task worker.")
        if task_id is not None and self.task_store is None:
            raise RuntimeError("task_store is required for attached provider failure.")
        resolution_id = resolution_event.payload.get("resolution_id")
        recovery_reason = resolution_event.payload.get("recovery_reason")
        if type(resolution_id) is not str or type(recovery_reason) is not str:
            raise ValueError("Provider-operation resolution evidence is malformed.")
        resolution_has_profile = "execution_profile_fingerprint" in resolution_event.payload
        resolution_profile = resolution_event.payload.get("execution_profile_fingerprint")
        if resolution_profile != execution_profile.fingerprint and not (
            legacy_resolution_without_profile and not resolution_has_profile
        ):
            raise ValueError("Provider-operation resolution belongs to another execution profile.")
        try:
            model_attempt_identity = ModelAttemptIdentity.model_validate(
                {
                    "model_step_id": resolution_event.payload.get("model_step_id"),
                    "model_attempt_id": resolution_event.payload.get("model_attempt_id"),
                }
            )
        except ValidationError:
            raise ValueError("Provider-operation resolution lost model identity.") from None

        environment_name = _environment_name(registered_environment)
        common_payload = {
            key: resolution_event.payload[key]
            for key in (
                "provider",
                "model",
                "step",
                "attempt",
                "max_attempts",
                "model_step_id",
                "model_attempt_id",
                "source_run_epoch",
                "run_epoch",
                "stage_id",
                "resolution_id",
                "resolution_action",
                "recovery_reason",
                "duplicate_request_risk",
                "reason",
                "metadata",
                "resolved_by",
            )
        }
        model_error_id = provider_operation_resolution_outcome_event_id(
            resolution_id,
            "model_error",
        )
        model_error_records = await self.session_store.query_events(
            EventQuery(session_id=session.id, event_id=model_error_id, limit=2)
        )
        if model_error_records:
            if len(model_error_records) != 1:
                raise ProviderOperationEvidenceError(
                    "Provider-operation failure has duplicate model evidence."
                )
            validate_provider_operation_resolution_outcome_event(
                model_error_records[0].event,
                resolution_event=resolution_event,
                outcome="model_error",
                expected_execution_profile_fingerprint=execution_profile.fingerprint,
            )
        else:
            model_error = event_with_execution_profile_authority(
                _event_with_model_identity_authority(
                    Event(
                        id=model_error_id,
                        type=EventType.MODEL_ERROR,
                        session_id=session.id,
                        interaction_id=resolution_event.interaction_id,
                        agent_name=registered_agent.spec.name,
                        environment_name=environment_name,
                        timestamp=resolution_event.timestamp,
                        payload={
                            **common_payload,
                            "error": "Provider operation was explicitly failed after recovery.",
                            "error_type": "provider_operation_unavailable",
                            "stage": "provider_operation_recovery",
                        },
                    ),
                    model_attempt_identity,
                ),
                execution_profile,
            )
            yield await self._event_writer.emit(model_error)

        if task_id is not None:
            task = await self.fail_task(
                task_id=task_id,
                task_worker_id=task_worker_id,
                task_handoff_id=task_handoff_id,
                session=session,
                error=provider_operation_task_failure_payload(session_id=session.id),
            )
            task_failed_template = _task_event(
                event_type=EventType.TASK_FAILED,
                task=task,
                session=session,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
            )
            task_failed = task_failed_template.model_copy(
                update={
                    "id": provider_operation_resolution_outcome_event_id(
                        resolution_id,
                        "task_failed",
                    ),
                    "interaction_id": resolution_event.interaction_id,
                    "timestamp": resolution_event.timestamp,
                    "payload": {
                        **task_failed_template.payload,
                        "resolution_id": resolution_id,
                        "stage_id": resolution_event.payload["stage_id"],
                        "failure_type": "provider_operation_unavailable",
                    },
                }
            )
            task_failed = event_with_runtime_generated_id(
                event_with_execution_profile_authority(
                    event_with_runtime_payload_authority(
                        task_failed,
                        "resolution_id",
                        "stage_id",
                    ),
                    execution_profile,
                )
            )
            persisted_task_failed = await self._event_writer.persist_exact_replay(task_failed)
            yield (await self._event_writer.fan_out_persisted([persisted_task_failed]))[0]

        interaction_failed_id = provider_operation_resolution_outcome_event_id(
            resolution_id,
            "interaction_failed",
        )
        interaction_failed_records = await self.session_store.query_events(
            EventQuery(session_id=session.id, event_id=interaction_failed_id, limit=2)
        )
        transitioned_session = await self.session_store.load(session.id)
        if transitioned_session is None:
            raise KeyError(f"Session not found: {session.id}")
        if interaction_failed_records:
            if len(interaction_failed_records) != 1:
                raise ProviderOperationEvidenceError(
                    "Provider-operation failure has duplicate interaction evidence."
                )
            validate_provider_operation_resolution_outcome_event(
                interaction_failed_records[0].event,
                resolution_event=resolution_event,
                outcome="interaction_failed",
                expected_execution_profile_fingerprint=execution_profile.fingerprint,
            )
            if transitioned_session.status is not SessionStatus.FAILED:
                raise ProviderOperationEvidenceError(
                    "Provider-operation interaction failure conflicts with session status."
                )
        else:
            interaction_id = await self.activate_latest_open_interaction(session.id)
            if interaction_id is None:
                raise RuntimeError("Provider-operation failure has no open durable interaction.")
            observed_at = await self.failure_interaction_observed_at(
                transitioned_session,
                observed_at=resolution_event.timestamp,
            )
            # Failure does not dispatch new work. Preserve the actual released
            # epoch instead of treating a reconstructed context as a live owner.
            stored_profile = active_invocation_execution_profile_from_checkpoint(
                await self.session_store.load_checkpoint(session.id)
            )
            released_authority = (
                stored_profile is not None
                and stored_profile.profile == execution_profile
                and stored_profile.interaction_id == resolution_event.interaction_id
                and active_invocation_execution_profile_is_released(
                    stored_profile,
                    session_id=session.id,
                    run_epoch=transitioned_session.run_epoch,
                )
            )
            (
                transitioned_session,
                interaction_failed_event,
                _,
            ) = await self.publish_sibling_interaction_transition(
                session=transitioned_session,
                invocation_context=None if released_authority else invocation_context,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
                environment_name=environment_name,
                to_status=SessionStatus.FAILED,
                from_statuses={SessionStatus.INTERRUPTED},
                observed_at=observed_at,
                event_id=interaction_failed_id,
                execution_profile=execution_profile,
                allow_released_invocation_authority=released_authority,
            )
            if interaction_failed_event is not None:
                yield interaction_failed_event

        session_failed_id = provider_operation_resolution_outcome_event_id(
            resolution_id,
            "session_failed",
        )
        terminal_records = await self.session_store.query_events(
            EventQuery(session_id=session.id, event_id=session_failed_id, limit=2)
        )
        if terminal_records:
            if len(terminal_records) != 1:
                raise ProviderOperationEvidenceError(
                    "Provider-operation failure has duplicate session evidence."
                )
            validate_provider_operation_resolution_outcome_event(
                terminal_records[0].event,
                resolution_event=resolution_event,
                outcome="session_failed",
                expected_execution_profile_fingerprint=execution_profile.fingerprint,
            )
            if transitioned_session.status is not SessionStatus.FAILED:
                raise ProviderOperationEvidenceError(
                    "Provider-operation terminal failure conflicts with session status."
                )
            try:
                async for hook_event in self._terminal_event_publication.replay(
                    event=copy_event(terminal_records[0].event),
                    phase=RuntimeHookPhase.AFTER_SESSION_FAILED,
                    session=transitioned_session,
                    registered_agent=registered_agent,
                    registered_environment=registered_environment,
                    execution_profile=execution_profile,
                    invocation_context=invocation_context,
                    yield_terminal_event=False,
                ):
                    yield hook_event
            finally:
                _deactivate_session_interaction(session.id)
            return
        try:
            async for terminal_event in self._terminal_event_publication.emit(
                event=event_with_execution_profile_authority(
                    Event(
                        id=session_failed_id,
                        type=EventType.SESSION_FAILED,
                        session_id=session.id,
                        agent_name=registered_agent.spec.name,
                        environment_name=environment_name,
                        timestamp=resolution_event.timestamp,
                        payload={
                            **common_payload,
                            "failure_type": "provider_operation_unavailable",
                            "error": "Provider operation was explicitly failed after recovery.",
                        },
                    ),
                    execution_profile,
                ),
                phase=RuntimeHookPhase.AFTER_SESSION_FAILED,
                session=transitioned_session,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
                execution_profile=execution_profile,
                invocation_context=invocation_context,
            ):
                yield terminal_event
        finally:
            _deactivate_session_interaction(session.id)

    async def prepare_turn_completed_event(
        self,
        *,
        session: Session,
        registered_agent: runtime_records.RegisteredAgentState,
        environment_name: str | None,
        status: SessionStatus,
        run_started_at: float,
        usage_tracker: SessionUsageTracker,
        event_id: str | None = None,
        timestamp: datetime | None = None,
    ) -> Event:
        """Prepare one invocation summary before a crash-sensitive terminal boundary."""

        summary = await usage_tracker.usage_summary()
        duration_ms = max(0, int((time.monotonic() - run_started_at) * 1000))
        interaction_ids = _current_session_invocation_interaction_ids(session.id)
        turn_completed = event_with_runtime_nested_payload_authority(
            Event(
                id=event_id or str(uuid4()),
                type=EventType.TURN_COMPLETED,
                session_id=session.id,
                interaction_id=None,
                agent_name=registered_agent.spec.name,
                environment_name=environment_name,
                timestamp=datetime.now(UTC) if timestamp is None else timestamp,
                payload={
                    "status": status.value,
                    "duration_ms": duration_ms,
                    "step_count": summary.model_steps,
                    "tool_call_count": summary.tool_calls,
                    "token_usage": aggregate_usage_metrics_payload(summary.usage),
                    "provider_names": summary.provider_names,
                    "models": summary.models,
                    "interaction_ids": list(interaction_ids),
                },
            ),
            ("interaction_ids", "*"),
        )
        # Both the deterministic replay identity and the fresh UUID fallback are
        # generated inside this runtime boundary. Preserve that provenance so a
        # configured secret fragment cannot make Cayu reject its own event ID.
        return event_with_runtime_generated_id(turn_completed)

    async def _emit_turn_completed(
        self,
        *,
        session: Session,
        registered_agent: runtime_records.RegisteredAgentState,
        environment_name: str | None,
        status: SessionStatus,
        run_started_at: float,
        usage_tracker: SessionUsageTracker,
        invocation_context: InvocationContext | None = None,
        prepared_turn_completed: Event | None = None,
        settle_interaction: bool = True,
    ) -> tuple[Event, ...]:
        registered_environment = None
        execution_profile = None
        if invocation_context is not None:
            if (
                invocation_context.binding.session_id != session.id
                or invocation_context.registered_agent is not registered_agent
                or invocation_context.binding.environment_name != environment_name
            ):
                raise RuntimeError("Turn completion substituted frozen invocation authority.")
            registered_environment = invocation_context.registered_environment
            execution_profile = invocation_context.profile
        interaction_id = _current_session_interaction_id(session.id)
        if settle_interaction and interaction_id is not None and status != SessionStatus.COMPLETED:
            _, interaction_event, _ = await self.publish_sibling_interaction_transition(
                session=session,
                invocation_context=invocation_context,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
                environment_name=environment_name,
                to_status=status,
                from_statuses={status},
                execution_profile=execution_profile,
                finalize_unsettled_cancellation=False,
            )
        else:
            interaction_event = None
        turn_completed = prepared_turn_completed
        if turn_completed is None:
            turn_completed = await self.prepare_turn_completed_event(
                session=session,
                registered_agent=registered_agent,
                environment_name=environment_name,
                status=status,
                run_started_at=run_started_at,
                usage_tracker=usage_tracker,
            )
        elif (
            turn_completed.type is not EventType.TURN_COMPLETED
            or turn_completed.session_id != session.id
            or turn_completed.interaction_id is not None
            or turn_completed.agent_name != registered_agent.spec.name
            or turn_completed.environment_name != environment_name
            or turn_completed.payload.get("status") != status.value
        ):
            raise RuntimeError("Prepared turn completion conflicts with its invocation.")
        turn_completed_event = await self._event_writer.emit(turn_completed)
        if interaction_event is None:
            return (turn_completed_event,)
        return (interaction_event, turn_completed_event)

    async def emit_turn_completed_once(
        self,
        *,
        session: Session,
        registered_agent: runtime_records.RegisteredAgentState,
        environment_name: str | None,
        status: SessionStatus,
        run_started_at: float,
        usage_tracker: SessionUsageTracker,
        active_run: ActiveSessionRun[SessionUsageTracker] | None,
        invocation_context: InvocationContext | None = None,
        prepared_turn_completed: Event | None = None,
        settle_interaction: bool = True,
    ) -> tuple[Event, ...]:
        if active_run is None:
            return await self._emit_turn_completed(
                session=session,
                registered_agent=registered_agent,
                environment_name=environment_name,
                status=status,
                run_started_at=run_started_at,
                usage_tracker=usage_tracker,
                invocation_context=invocation_context,
                prepared_turn_completed=prepared_turn_completed,
                settle_interaction=settle_interaction,
            )
        async with active_run.turn_completed_lock:
            if active_run.turn_completed_event is not None:
                return (active_run.turn_completed_event,)
            events = await self._emit_turn_completed(
                session=session,
                registered_agent=registered_agent,
                environment_name=environment_name,
                status=status,
                run_started_at=run_started_at,
                usage_tracker=usage_tracker,
                invocation_context=invocation_context,
                prepared_turn_completed=prepared_turn_completed,
                settle_interaction=settle_interaction,
            )
            active_run.turn_completed_event = events[-1]
            return events

    async def fail_task(
        self,
        *,
        task_id: str,
        task_worker_id: str | None,
        task_handoff_id: str | None,
        session: Session,
        error: dict[str, Any],
    ) -> Task:
        if self.task_store is None:
            raise RuntimeError("task_store is required when RunRequest.task_id is set.")
        if task_worker_id is not None:
            return await _terminalize_claimed_task(
                self.task_store,
                runtime_task_failure_terminalization_request(
                    task_id=task_id,
                    task_worker_id=task_worker_id,
                    task_handoff_id=task_handoff_id,
                    session_id=session.id,
                    error=error,
                ),
            )
        replayed = await load_direct_task_failure_replay(
            self.task_store,
            task_id=task_id,
            session_id=session.id,
            session_instance_id=session.instance_id,
            expected_error=error,
            claimed_terminalization_idempotency_key=_task_terminalization_idempotency_key(
                task_id=task_id,
                session_id=session.id,
                kind=TaskTerminalKind.FAILED,
            ),
        )
        if replayed is not None:
            return replayed
        return await self.task_store.fail_task(
            task_id,
            error,
            worker_id=task_worker_id,
        )

    async def apply_model_step_budget_evaluation(
        self,
        request: ModelStepBudgetEvaluationRequest,
    ) -> AsyncGenerator[Event, None]:
        events = self.apply_budget_evaluation(
            evaluation=request.evaluation,
            session=request.session,
            registered_agent=request.registered_agent,
            registered_environment=request.registered_environment,
            environment_name=request.environment_name,
            messages=request.messages,
            run_started_at=request.run_started_at,
            turn_usage_tracker=request.turn_usage_tracker,
            active_run=request.active_run,
            execution_profile=request.execution_profile,
            invocation_context=request.invocation_context,
        )
        try:
            async for event in events:
                yield event
        finally:
            await _close_async_iterator(events)

    async def apply_model_step_limit_evaluation(
        self,
        request: ModelStepLimitEvaluationRequest,
    ) -> AsyncGenerator[Event, None]:
        events = self.apply_limit_evaluation(
            evaluation=request.evaluation,
            session=request.session,
            registered_agent=request.registered_agent,
            registered_environment=request.registered_environment,
            environment_name=request.environment_name,
            messages=request.messages,
            run_started_at=request.run_started_at,
            turn_usage_tracker=request.turn_usage_tracker,
            active_run=request.active_run,
            execution_profile=request.execution_profile,
            invocation_context=request.invocation_context,
        )
        try:
            async for event in events:
                yield event
        finally:
            await _close_async_iterator(events)

    async def stop_for_model_step_budget_reservation_failure(
        self,
        request: ModelStepBudgetReservationFailureRequest,
    ) -> AsyncGenerator[Event, None]:
        events = self._stop_session_for_budget_reservation_failed(
            session=request.session,
            registered_agent=request.registered_agent,
            registered_environment=request.registered_environment,
            environment_name=request.environment_name,
            result=request.result,
            messages=request.messages,
            run_started_at=request.run_started_at,
            turn_usage_tracker=request.turn_usage_tracker,
            active_run=request.active_run,
            execution_profile=request.execution_profile,
            invocation_context=request.invocation_context,
        )
        try:
            async for event in events:
                yield event
        finally:
            await _close_async_iterator(events)

    async def apply_tool_round_limit(
        self,
        request: ToolRoundLimitRequest,
    ) -> AsyncGenerator[Event, None]:
        async for event in self.apply_limit_evaluation(
            evaluation=request.evaluation,
            session=request.session,
            registered_agent=request.registered_agent,
            registered_environment=request.registered_environment,
            environment_name=request.environment_name,
            messages=request.messages,
            tool_calls=request.tool_calls,
            completed_tool_outcomes=request.completed_tool_outcomes,
            tool_round_identity=request.tool_round_identity,
            run_started_at=request.run_started_at,
            turn_usage_tracker=request.turn_usage_tracker,
            active_run=request.active_run,
            execution_profile=request.execution_profile,
            invocation_context=request.invocation_context,
        ):
            yield event

    async def apply_limit_evaluation(
        self,
        *,
        evaluation: LimitEvaluation,
        session: Session,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_environment: runtime_records.RegisteredEnvironment | None,
        environment_name: str | None,
        messages: list[Message],
        tool_calls: list[runtime_records.ToolCallRequest] | None = None,
        completed_tool_outcomes: list[runtime_records.ToolCallOutcome] | None = None,
        tool_round_identity: ToolRoundIdentity | None = None,
        run_started_at: float | None = None,
        turn_usage_tracker: SessionUsageTracker | None = None,
        active_run: ActiveSessionRun[SessionUsageTracker] | None = None,
        execution_profile: ExecutionProfileIdentity | None,
        invocation_context: InvocationContext | None = None,
    ) -> AsyncGenerator[Event, None]:
        for event in evaluation.events:
            yield event
        if evaluation.decision is None:
            return
        async for event in self.stop_session_for_limit_reached(
            session=session,
            registered_agent=registered_agent,
            registered_environment=registered_environment,
            environment_name=environment_name,
            decision=evaluation.decision,
            usage_summary=evaluation.usage_summary,
            cost_summary=evaluation.cost_summary,
            messages=messages,
            tool_calls=tool_calls if tool_calls is not None else [],
            completed_tool_outcomes=(
                completed_tool_outcomes if completed_tool_outcomes is not None else []
            ),
            tool_round_identity=tool_round_identity,
            run_started_at=run_started_at,
            turn_usage_tracker=turn_usage_tracker,
            active_run=active_run,
            execution_profile=execution_profile,
            invocation_context=invocation_context,
            reconcile_transition_cancellation=False,
        ):
            yield event

    async def apply_budget_evaluation(
        self,
        *,
        evaluation: BudgetEvaluation,
        session: Session,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_environment: runtime_records.RegisteredEnvironment | None,
        environment_name: str | None,
        messages: list[Message],
        tool_calls: list[runtime_records.ToolCallRequest] | None = None,
        tool_round_identity: ToolRoundIdentity | None = None,
        run_started_at: float | None = None,
        turn_usage_tracker: SessionUsageTracker | None = None,
        active_run: ActiveSessionRun[SessionUsageTracker] | None = None,
        execution_profile: ExecutionProfileIdentity | None,
        invocation_context: InvocationContext | None = None,
    ) -> AsyncGenerator[Event, None]:
        for event in evaluation.events:
            yield event
        if evaluation.check is None:
            return
        async for event in self._stop_session_for_budget_limit_reached(
            session=session,
            registered_agent=registered_agent,
            registered_environment=registered_environment,
            environment_name=environment_name,
            check=evaluation.check,
            messages=messages,
            tool_calls=tool_calls if tool_calls is not None else [],
            completed_tool_outcomes=[],
            tool_round_identity=tool_round_identity,
            run_started_at=run_started_at,
            turn_usage_tracker=turn_usage_tracker,
            active_run=active_run,
            execution_profile=execution_profile,
            invocation_context=invocation_context,
        ):
            yield event

    async def stop_session_for_limit_reached(
        self,
        *,
        session: Session,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_environment: runtime_records.RegisteredEnvironment | None,
        environment_name: str | None,
        decision: StopDecision,
        usage_summary: SessionUsageSummary,
        cost_summary: SessionCostTotals | None,
        messages: list[Message],
        tool_calls: list[runtime_records.ToolCallRequest],
        completed_tool_outcomes: list[runtime_records.ToolCallOutcome],
        pending_approval_to_clear: PendingToolApproval | None = None,
        deferred_messages: list[Message] | None = None,
        requested_approval_decision: ToolApprovalDecision | None = None,
        approval_resolution_request_digest: str | None = None,
        tool_round_identity: ToolRoundIdentity | None = None,
        run_started_at: float | None = None,
        turn_usage_tracker: SessionUsageTracker | None = None,
        active_run: ActiveSessionRun[SessionUsageTracker] | None = None,
        execution_profile: ExecutionProfileIdentity | None,
        invocation_context: InvocationContext | None = None,
        reconcile_transition_cancellation: bool,
    ) -> AsyncGenerator[Event, None]:
        if type(reconcile_transition_cancellation) is not bool:
            raise TypeError("reconcile_transition_cancellation must be a bool.")
        if invocation_context is not None and (
            registered_agent is not invocation_context.registered_agent
            or registered_environment is not invocation_context.registered_environment
            or execution_profile is not invocation_context.profile
        ):
            raise RuntimeError("Limit terminalization substituted frozen invocation authority.")
        if (
            decision.limit is StopLimit.ESTIMATED_COST
            and invocation_context is not None
            and invocation_context.work_attempt is not None
        ):
            if self.task_store is None:
                raise RuntimeError("Governed budget stop requires its task store.")
            await record_work_attempt_execution_stop(
                self.task_store,
                admission=invocation_context.work_attempt.admission,
                reason="budget_limit",
                redactor=self._secret_redactor,
            )
        if tool_round_identity is None and pending_approval_to_clear is not None:
            tool_round_identity = ToolRoundIdentity(
                tool_round_id=pending_approval_to_clear.tool_round_id,
                model_step_id=pending_approval_to_clear.model_step_id,
                model_attempt_id=pending_approval_to_clear.model_attempt_id,
            )
        limit_payload = _limit_reached_payload(
            decision=decision,
            usage_summary=usage_summary,
            cost_summary=cost_summary,
        )
        yield await self._event_writer.emit(
            Event(
                type=EventType.SESSION_LIMIT_REACHED,
                session_id=session.id,
                agent_name=registered_agent.spec.name,
                environment_name=environment_name,
                payload=limit_payload,
            )
        )
        if tool_calls or completed_tool_outcomes or pending_approval_to_clear is not None:
            async for event in self._close_limited_tool_round(
                session=session,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
                messages=messages,
                tool_calls=tool_calls,
                completed_tool_outcomes=completed_tool_outcomes,
                decision=decision,
                pending_approval_to_clear=pending_approval_to_clear,
                deferred_messages=deferred_messages,
                requested_approval_decision=requested_approval_decision,
                approval_resolution_request_digest=approval_resolution_request_digest,
                tool_round_identity=tool_round_identity,
                execution_profile=execution_profile,
            ):
                yield event

        if reconcile_transition_cancellation:
            (
                interrupted_session,
                interaction_interrupted_event,
                _,
            ) = await self.publish_sibling_interaction_transition(
                session=session,
                invocation_context=invocation_context,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
                environment_name=environment_name,
                to_status=SessionStatus.INTERRUPTED,
                execution_profile=execution_profile,
            )
        else:
            (
                interrupted_session,
                interaction_interrupted_event,
                _,
            ) = await self.publish_interaction_transition(
                session=session,
                invocation_context=invocation_context,
                agent_name=registered_agent.spec.name,
                environment_name=environment_name,
                to_status=SessionStatus.INTERRUPTED,
                execution_profile=execution_profile,
            )
        if interaction_interrupted_event is not None:
            yield interaction_interrupted_event
        terminal_payload = {
            "interruption_type": _INTERRUPTION_TYPE_LIMIT_REACHED,
            **limit_payload,
            # The transition can advance the stored epoch. Correlate the stop
            # with the executing session, never the post-transition snapshot.
            "failure_evidence": FailureEvidence(
                classification="interruption",
                session_id=session.id,
                run_epoch=session.run_epoch,
            ).model_dump(mode="json"),
        }
        if tool_round_identity is not None:
            terminal_payload.update(tool_round_identity.payload())
        if run_started_at is not None and turn_usage_tracker is not None:
            for event in await self.emit_turn_completed_once(
                session=interrupted_session,
                registered_agent=registered_agent,
                environment_name=environment_name,
                status=SessionStatus.INTERRUPTED,
                run_started_at=run_started_at,
                usage_tracker=turn_usage_tracker,
                active_run=active_run,
                invocation_context=invocation_context,
            ):
                yield event
        async for event in self._terminal_event_publication.emit(
            event=Event(
                type=EventType.SESSION_INTERRUPTED,
                session_id=interrupted_session.id,
                agent_name=registered_agent.spec.name,
                environment_name=environment_name,
                payload=terminal_payload,
            ),
            phase=RuntimeHookPhase.AFTER_SESSION_INTERRUPTED,
            session=interrupted_session,
            registered_agent=registered_agent,
            registered_environment=registered_environment,
            execution_profile=execution_profile,
            invocation_context=invocation_context,
        ):
            yield event

    async def _stop_session_for_budget_reservation_failed(
        self,
        *,
        session: Session,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_environment: runtime_records.RegisteredEnvironment | None,
        environment_name: str | None,
        result: BudgetReservationResult,
        messages: list[Message],
        run_started_at: float | None = None,
        turn_usage_tracker: SessionUsageTracker | None = None,
        active_run: ActiveSessionRun[SessionUsageTracker] | None = None,
        execution_profile: ExecutionProfileIdentity | None = None,
        invocation_context: InvocationContext | None = None,
    ) -> AsyncGenerator[Event, None]:
        payload = budget_reservation_payload(result)
        yield await self._event_writer.emit(
            event_with_execution_profile_authority(
                Event(
                    type=EventType.BUDGET_LIMIT_REACHED,
                    session_id=session.id,
                    agent_name=registered_agent.spec.name,
                    environment_name=environment_name,
                    payload=payload,
                ),
                execution_profile,
            )
        )
        usage_summary = await self._run_limit_controller.session_usage_summary(session.id)
        decision = StopDecision(
            limit=StopLimit.ESTIMATED_COST,
            maximum=result.maximum,
            actual=result.actual,
            message=result.message,
        )
        async for event in self.stop_session_for_limit_reached(
            session=session,
            registered_agent=registered_agent,
            registered_environment=registered_environment,
            environment_name=environment_name,
            decision=decision,
            usage_summary=usage_summary,
            cost_summary=None,
            messages=messages,
            tool_calls=[],
            completed_tool_outcomes=[],
            run_started_at=run_started_at,
            turn_usage_tracker=turn_usage_tracker,
            active_run=active_run,
            execution_profile=execution_profile,
            invocation_context=invocation_context,
            reconcile_transition_cancellation=False,
        ):
            yield event

    async def _stop_session_for_budget_limit_reached(
        self,
        *,
        session: Session,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_environment: runtime_records.RegisteredEnvironment | None,
        environment_name: str | None,
        check: BudgetCheck,
        messages: list[Message],
        tool_calls: list[runtime_records.ToolCallRequest],
        completed_tool_outcomes: list[runtime_records.ToolCallOutcome],
        tool_round_identity: ToolRoundIdentity | None = None,
        run_started_at: float | None = None,
        turn_usage_tracker: SessionUsageTracker | None = None,
        active_run: ActiveSessionRun[SessionUsageTracker] | None = None,
        execution_profile: ExecutionProfileIdentity | None = None,
        invocation_context: InvocationContext | None = None,
    ) -> AsyncGenerator[Event, None]:
        payload = budget_limit_reached_payload(check)
        yield await self._event_writer.emit(
            event_with_execution_profile_authority(
                Event(
                    type=EventType.BUDGET_LIMIT_REACHED,
                    session_id=session.id,
                    agent_name=registered_agent.spec.name,
                    environment_name=environment_name,
                    payload=payload,
                ),
                execution_profile,
            )
        )
        decision = StopDecision(
            limit=StopLimit.ESTIMATED_COST,
            maximum=check.maximum,
            actual=check.actual,
            message=check.message,
        )
        usage_summary = await self._run_limit_controller.session_usage_summary(session.id)
        async for event in self.stop_session_for_limit_reached(
            session=session,
            registered_agent=registered_agent,
            registered_environment=registered_environment,
            environment_name=environment_name,
            decision=decision,
            usage_summary=usage_summary,
            cost_summary=check.cost_summary,
            messages=messages,
            tool_calls=tool_calls,
            completed_tool_outcomes=completed_tool_outcomes,
            tool_round_identity=tool_round_identity,
            run_started_at=run_started_at,
            turn_usage_tracker=turn_usage_tracker,
            active_run=active_run,
            execution_profile=execution_profile,
            invocation_context=invocation_context,
            reconcile_transition_cancellation=False,
        ):
            yield event

    async def _close_limited_tool_round(
        self,
        *,
        session: Session,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_environment: runtime_records.RegisteredEnvironment | None,
        messages: list[Message],
        tool_calls: list[runtime_records.ToolCallRequest],
        completed_tool_outcomes: list[runtime_records.ToolCallOutcome],
        decision: StopDecision,
        pending_approval_to_clear: PendingToolApproval | None = None,
        deferred_messages: list[Message] | None = None,
        requested_approval_decision: ToolApprovalDecision | None = None,
        approval_resolution_request_digest: str | None = None,
        tool_round_identity: ToolRoundIdentity | None = None,
        execution_profile: ExecutionProfileIdentity | None,
    ) -> AsyncGenerator[Event, None]:
        if tool_round_identity is None and pending_approval_to_clear is not None:
            tool_round_identity = ToolRoundIdentity(
                tool_round_id=pending_approval_to_clear.tool_round_id,
                model_step_id=pending_approval_to_clear.model_step_id,
                model_attempt_id=pending_approval_to_clear.model_attempt_id,
            )
        if tool_round_identity is None:
            raise RuntimeError("Closing a tool round requires its durable identity.")
        if pending_approval_to_clear is not None:
            if (
                type(requested_approval_decision) is not ToolApprovalDecision
                or type(approval_resolution_request_digest) is not str
                or not approval_resolution_request_digest
            ):
                raise RuntimeError(
                    "Closing a limited approval requires its original resolution request identity."
                )
            deferred_messages = (
                []
                if deferred_messages is None
                else [detach_message(message) for message in deferred_messages]
            )
            # The existing transcript may be a legacy, pre-quarantine assistant
            # message. Validate that boundary against the private durable pause
            # descriptor, not against terminal outcomes whose argument projection
            # can intentionally be unavailable.
            expected_tool_calls = [
                approval_support.tool_call_request_from_pending(call)
                for call in pending_approval_to_clear.tool_calls
            ]
            if await transcript_helpers.tool_round_has_result_messages(
                self.session_store,
                session.id,
                expected_tool_calls,
                tool_round_identity=tool_round_identity,
            ):

                def clear_exact_approval_round(
                    _current_session: Session,
                    checkpoint: dict[str, Any] | None,
                ) -> dict[str, Any]:
                    return approval_support.checkpoint_without_exact_pending_approval_round(
                        checkpoint,
                        approval=pending_approval_to_clear,
                        redactor=self._secret_redactor,
                        runtime_session=_current_session,
                    )

                await self.session_store.transform_checkpoint(
                    session.id,
                    clear_exact_approval_round,
                )
                materialized = await self._deferred_input.materialize_expected(
                    session.id,
                    deferred_messages,
                )
                messages[:] = materialized.messages
                yield await self._event_writer.emit(
                    approval_support.cleared_event(
                        session=session,
                        agent_name=registered_agent.spec.name,
                        environment_name=_environment_name(registered_environment),
                        approval=pending_approval_to_clear,
                    )
                )
                if materialized.cancellation is not None:
                    raise materialized.cancellation
                return
            async for event in self._close_limited_approval_tool_round(
                session=session,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
                messages=messages,
                tool_calls=tool_calls,
                completed_tool_outcomes=completed_tool_outcomes,
                decision=decision,
                pending_approval_to_clear=pending_approval_to_clear,
                deferred_messages=deferred_messages,
                requested_approval_decision=requested_approval_decision,
                approval_resolution_request_digest=approval_resolution_request_digest,
                tool_round_identity=tool_round_identity,
                execution_profile=execution_profile,
            ):
                yield event
            return

        round_owner = DurableToolRound(
            session=session,
            tool_round_identity=tool_round_identity,
            session_store=self.session_store,
            event_writer=self._event_writer,
        )
        async with contextlib.aclosing(
            round_owner.close_for_limit(
                messages=messages,
                tool_calls=tool_calls,
                completed_tool_outcomes=completed_tool_outcomes,
                decision=decision,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
                execution_profile=execution_profile,
                redactor=self._secret_redactor,
                materialize_deferred_input_if_present=(self._deferred_input.materialize_if_present),
                materialize_expected_deferred_input=(self._deferred_input.materialize_expected),
                raise_if_interrupted=self._session_control.raise_if_interrupted,
            )
        ) as closure:
            async for event in closure:
                yield event

    async def _close_limited_approval_tool_round(
        self,
        *,
        session: Session,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_environment: runtime_records.RegisteredEnvironment | None,
        messages: list[Message],
        tool_calls: list[runtime_records.ToolCallRequest],
        completed_tool_outcomes: list[runtime_records.ToolCallOutcome],
        decision: StopDecision,
        pending_approval_to_clear: PendingToolApproval,
        deferred_messages: list[Message],
        requested_approval_decision: ToolApprovalDecision,
        approval_resolution_request_digest: str,
        tool_round_identity: ToolRoundIdentity,
        execution_profile: ExecutionProfileIdentity | None,
    ) -> AsyncGenerator[Event, None]:
        completed_ids = {outcome.call.id for outcome in completed_tool_outcomes}
        remaining_tool_calls = [
            tool_call for tool_call in tool_calls if tool_call.id not in completed_ids
        ]
        skipped_outcomes = _limit_reached_tool_round_results(
            tool_calls=remaining_tool_calls,
            decision=decision,
            tool_round_identity=tool_round_identity,
        )
        completed_tool_outcomes = tool_results.redact_tool_call_outcomes(
            completed_tool_outcomes,
            self._secret_redactor,
        )
        skipped_outcomes = tool_results.redact_tool_call_outcomes(
            skipped_outcomes,
            self._secret_redactor,
        )
        base_round_redactor = _redactor_for_tool_calls(
            self._secret_redactor,
            registered_agent=registered_agent,
            tool_calls=tool_calls,
        )
        for skipped_outcome in skipped_outcomes:
            await self.session_store.transform_checkpoint(
                session.id,
                lambda _current_session, current_checkpoint, call_id=skipped_outcome.call.id: (
                    tool_round_recovery.checkpoint_with_assistant_publication_snapshot(
                        current_checkpoint,
                        tool_round_identity=tool_round_identity,
                        tool_call_id=call_id,
                        redactor=base_round_redactor,
                        unsafe_output=False,
                    )
                ),
            )
            yield await self._event_writer.emit(
                _limit_reached_tool_call_event(
                    session=session,
                    registered_agent=registered_agent,
                    registered_environment=registered_environment,
                    tool_call_outcome=skipped_outcome,
                    decision=decision,
                    tool_round_identity=tool_round_identity,
                    execution_profile=execution_profile,
                    approval_id=pending_approval_to_clear.approval_id,
                )
            )
        skipped_outcomes = [
            runtime_records.ToolCallOutcome(
                call=runtime_records.copy_tool_call_request(
                    outcome.call,
                    arguments={},
                ),
                result=outcome.result,
            )
            for outcome in skipped_outcomes
        ]
        source_checkpoint, pending_round = await pending_round_reader.load_pending_tool_round(
            self.session_store,
            session.id,
            redactor=self._secret_redactor,
            consume_on_rejection=True,
            runtime_session=session,
        )
        if (
            pending_round is None
            or pending_round.tool_round_id != tool_round_identity.tool_round_id
        ):
            raise RuntimeError("Pending approval round changed before limit closure.")
        lifecycle_events = await self.session_store.load_tool_round_lifecycle_events_for_round(
            session.id,
            [call.tool_call_id for call in pending_round.tool_calls],
            tool_round_identity=tool_round_identity,
        )
        tool_result_messages = ordered_tool_result_messages(
            tool_calls,
            [*completed_tool_outcomes, *skipped_outcomes],
            parallel=True,
            tool_round_identity=tool_round_identity,
        )
        transcript_messages = list(tool_result_messages)
        if pending_round.assistant_message_state == "quarantined":
            transcript_messages.insert(
                0,
                transcript_helpers.assistant_message_with_projected_tool_arguments(
                    tool_round_recovery.ready_assistant_publication_message(pending_round),
                    [*completed_tool_outcomes, *skipped_outcomes],
                ),
            )
        target_checkpoint = approval_support.checkpoint_without_exact_pending_approval_round(
            source_checkpoint,
            approval=pending_approval_to_clear,
            redactor=self._secret_redactor,
            runtime_session=session,
        )
        clear_event = approval_support.cleared_event(
            session=session,
            agent_name=registered_agent.spec.name,
            environment_name=_environment_name(registered_environment),
            approval=pending_approval_to_clear,
        )
        prepared_close = approval_publication.prepare_approval_publication(
            session_id=session.id,
            publication_id=f"approval-close:{pending_approval_to_clear.approval_id}",
            kind="approval-close",
            intent={
                "schema_version": 1,
                "approval_id": pending_approval_to_clear.approval_id,
                "tool_call_id": pending_approval_to_clear.tool_call_id,
                **tool_round_identity.payload(),
                "decision": "limit_reached",
                "requested_decision": requested_approval_decision.value,
                "resolution_request_digest": approval_resolution_request_digest,
                "tool_call_ids": [call.tool_call_id for call in pending_round.tool_calls],
                "approval_digest": runtime_publication_checkpoint_value_digest(
                    pending_approval_to_clear.model_dump(mode="json")
                ),
                "pending_round_digest": runtime_publication_checkpoint_value_digest(
                    pending_round.model_dump(mode="json")
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
            expected_transcript_cursor=(
                await self.session_store.load_transcript_cursor(session.id)
            ),
        )
        prepared_events = prepared_close.request.events
        if len(prepared_events) != 1:
            raise AssertionError("Approval limit closure must publish one checkpoint event.")
        clear_event = prepared_events[0]
        cancellation = await approval_publication.publish_approval_with_exact_replay(
            prepared_close,
            session_store=self.session_store,
            event_writer=self._event_writer,
            fan_out=False,
        )
        materialized = await self._deferred_input.materialize_expected(
            session.id,
            deferred_messages,
            cancellation=cancellation,
        )
        messages[:] = materialized.messages
        cancellation = materialized.cancellation
        await self._event_writer.fan_out_persisted([clear_event])
        yield clear_event
        if cancellation is not None:
            raise cancellation

    async def clear_pending_session_interrupt(
        self,
        session_id: str,
        *,
        expected_payload: dict[str, Any] | None = None,
        expected_run_epoch: int | None = None,
    ) -> None:
        def transform(_session: Session, checkpoint: dict[str, Any] | None) -> dict[str, Any]:
            copied = {} if checkpoint is None else copy_durable_record(checkpoint, "checkpoint")
            if expected_payload is not None:
                current = copied.get(_PENDING_SESSION_INTERRUPT_CHECKPOINT_KEY)
                if current != expected_payload:
                    raise RuntimeError("Pending session interrupt identity changed before clear.")
            copied.pop(_PENDING_SESSION_INTERRUPT_CHECKPOINT_KEY, None)
            return copied

        if expected_run_epoch is None:
            await self.session_store.transform_checkpoint(session_id, transform)
            return
        await self.session_store.publish_checkpoint_and_events(
            session_id,
            checkpoint_transform=transform,
            events=[],
            expected_statuses={SessionStatus.INTERRUPTED},
            expected_run_epoch=expected_run_epoch,
        )

    async def clear_claimed_pending_interrupt_if_retained(
        self,
        *,
        session_id: str,
        claim_id: str,
        expected_payload: dict[str, Any],
    ) -> None:
        """Clear an exact claimed marker unless its terminal event did so atomically."""

        checkpoint = await self.session_store.load_checkpoint(session_id)
        if checkpoint is not None and _PENDING_SESSION_INTERRUPT_CHECKPOINT_KEY not in checkpoint:
            return
        await self._terminal_finalization.clear_pending_interrupt(
            session_id=session_id,
            claim_id=claim_id,
            expected_payload=expected_payload,
        )

    async def publish_terminal_event_under_finalization_claim(
        self,
        *,
        event: Event,
        session: Session,
        claim_id: str,
        expected_payload: dict[str, Any],
    ) -> Event:
        """Atomically bind one terminal event to its exact unexpired owner."""

        claim_expires_at: datetime | None = None

        def commit_under_exact_claim(
            current_session: Session,
            checkpoint: dict[str, Any] | None,
            store_now: datetime,
        ) -> dict[str, Any]:
            nonlocal claim_expires_at
            if (
                current_session.instance_id != session.instance_id
                or current_session.status is not SessionStatus.INTERRUPTED
                or current_session.run_epoch != session.run_epoch
                or checkpoint is None
            ):
                raise SessionRuntimePublicationConflict(
                    "Terminal event lost its exact session authority."
                )
            updated = copy_durable_record(checkpoint, "checkpoint")
            if updated.get(_PENDING_SESSION_INTERRUPT_CHECKPOINT_KEY) != expected_payload:
                raise SessionRuntimePublicationConflict(
                    "Terminal event lost its exact interruption authority."
                )
            claim = _incomplete_recovery_claim_from_checkpoint(updated)
            if claim is None or claim[0] != claim_id or claim[1] <= store_now:
                raise _IncompleteRecoveryClaimLost(
                    "Terminal event lost its exact finalization claim."
                )
            claim_expires_at = claim[1]
            updated.pop(_PENDING_SESSION_INTERRUPT_CHECKPOINT_KEY)
            return updated

        def require_unexpired_finalization_commit(commit_at: datetime) -> None:
            if claim_expires_at is None or claim_expires_at <= commit_at:
                raise _IncompleteRecoveryClaimLost(
                    "Terminal event finalization claim expired before durable publication."
                )

        await self.session_store.publish_checkpoint_and_events_with_store_time(
            session.id,
            idempotency_key=f"terminal-finalization:{event.id}",
            checkpoint_transform=commit_under_exact_claim,
            commit_time_guard=require_unexpired_finalization_commit,
            events=[event],
            expected_statuses={SessionStatus.INTERRUPTED},
            expected_run_epoch=session.run_epoch,
        )
        delivered = await self._event_writer.fan_out_persisted([event])
        if len(delivered) != 1:
            raise RuntimeError("Terminal event side-effect delivery returned no exact event.")
        return delivered[0]

    async def handle_session_interrupted(
        self,
        *,
        session: Session,
        foreground_subagent_pending: ForegroundSubagentRecoveryRequired | None = None,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_environment: runtime_records.RegisteredEnvironment | None,
        environment_name: str | None,
        execution_profile: ExecutionProfileIdentity | None = None,
        invocation_context: InvocationContext | None = None,
        run_started_at: float | None = None,
        turn_usage_tracker: SessionUsageTracker | None = None,
        active_run: ActiveSessionRun[SessionUsageTracker] | None = None,
        interaction_transition_failures: tuple[dict[str, Any], ...] = (),
        provider_cancellation_failures: tuple[dict[str, Any], ...] = (),
        terminal_finalization_handoff_source_task: asyncio.Task[Any] | None = None,
        run_terminal_hooks: bool = True,
        preserve_interaction_id: str | None = None,
        recovery_claim_id: str | None = None,
    ) -> AsyncGenerator[Event, None]:
        clear_current_task_cancellation()
        current_task = asyncio.current_task()
        if current_task is not None:
            self._session_control.unregister_active_task(session.id, current_task)
        self._session_control.begin_emitting_interrupted(session.id)
        finalization = self._terminal_finalization.interrupted_run(
            session=session,
            task=current_task,
            transferred_from=terminal_finalization_handoff_source_task,
        )

        try:
            loaded_interrupted = await finalization.await_operation(
                lambda: self.session_store.load(session.id),
                operation_name="Live interruption session read",
            )
            if loaded_interrupted is None:
                raise KeyError(f"Session not found: {session.id}") from None
            payload = await finalization.await_operation(
                lambda: (
                    self._background_interruption_coordinator.load_pending_session_interrupt_payload(
                        session.id,
                        default={},
                    )
                ),
                operation_name="Live interruption payload read",
            )
            if foreground_subagent_pending is not None:
                payload.update(foreground_subagent_pending.interruption_evidence())
            user_input_supersession_retained = (
                USER_INPUT_SUPERSESSION_INTENT_KEY in payload
                or AMBIGUOUS_USER_INPUT_SUPERSESSION_INTENT_KEY in payload
            )
            payload.setdefault("interruption_type", _INTERRUPTION_TYPE_RUNTIME_INTERRUPTED)
            payload.setdefault("interruption_request_id", str(uuid4()))
            interruption_request_id = interruption_request_id_from_payload(payload)
            if interruption_request_id is None:
                raise SessionRuntimePublicationConflict(
                    "Pending session interruption has no request identity."
                )
            prepared_finalization = await finalization.prepare(
                session=loaded_interrupted,
                payload=payload,
                interruption_request_id=interruption_request_id,
            )
            if isinstance(prepared_finalization, Event):
                yield prepared_finalization
                return
            loaded_interrupted = prepared_finalization
            exact_interrupt_marker_retained = (
                bool(provider_cancellation_failures) or user_input_supersession_retained
            )
            expected_diagnostic_payload = copy_json_value(payload, "interrupt_payload")
            if interaction_transition_failures:
                copied_failures = copy_durable_json_value(
                    list(interaction_transition_failures),
                    "interaction transition cancellation diagnostics",
                )
                if type(copied_failures) is not list:
                    raise TypeError(
                        "Interaction transition cancellation diagnostics must be a list."
                    )
                payload["interaction_transition_failures"] = copied_failures
            if provider_cancellation_failures:
                copied_provider_failures = copy_provider_cancellation_failures(
                    provider_cancellation_failures
                )
                payload["provider_cancellation_failures"] = [
                    dict(item) for item in copied_provider_failures
                ]
            if interaction_transition_failures or provider_cancellation_failures:

                def retain_interruption_diagnostics(
                    _session: Session, checkpoint: dict[str, Any] | None
                ) -> dict[str, Any]:
                    current = (
                        {} if checkpoint is None else copy_durable_record(checkpoint, "checkpoint")
                    )
                    marker = current.get(_PENDING_SESSION_INTERRUPT_CHECKPOINT_KEY)
                    if (
                        settled_invocation_terminal_decision_from_checkpoint(current) is not None
                        and marker is None
                    ):
                        raise SessionRunFenced(
                            "Interrupted invocation has already been terminalized."
                        )
                    if marker is not None and marker != expected_diagnostic_payload:
                        raise SessionRunFenced(
                            "Interruption changed before diagnostic publication."
                        )
                    current[_PENDING_SESSION_INTERRUPT_CHECKPOINT_KEY] = copy_json_value(
                        payload, "interrupt_payload"
                    )
                    return current

                # Persist exact repair authority before the status or sibling
                # interaction changes. A process loss after either mutation
                # must not leave terminal state with no diagnostic evidence.
                # Inspect the store-owned winner inside the atomic transform;
                # ordinary callbacks cannot see private lifecycle authority.
                # Read access does not authorize changing that winner.
                with _invocation_lifecycle_authority_read_scope():
                    await finalization.await_operation(
                        lambda: self.session_store.publish_checkpoint_and_events(
                            session.id,
                            checkpoint_transform=retain_interruption_diagnostics,
                            events=[],
                            expected_statuses={loaded_interrupted.status},
                            expected_run_epoch=loaded_interrupted.run_epoch,
                        ),
                        operation_name="Live interruption diagnostic checkpoint publication",
                    )
            decision_checkpoint = await finalization.await_operation(
                lambda: self.session_store.load_checkpoint(session.id),
                operation_name="Live interruption terminal-decision read",
            )
            terminal_decision = invocation_terminal_decision_from_checkpoint(decision_checkpoint)
            if (
                terminal_decision is None
                and settled_invocation_terminal_decision_from_checkpoint(decision_checkpoint)
                is not None
                and _PENDING_SESSION_INTERRUPT_CHECKPOINT_KEY not in (decision_checkpoint or {})
            ):
                # Another finalizer already published and acknowledged this
                # invocation's exact terminal winner. A late execution owner
                # must not invent a fresh interruption from an empty marker.
                raise SessionRunFenced("Interrupted invocation has already been terminalized.")
            if preserve_interaction_id is not None:
                active_profile = active_invocation_execution_profile_from_checkpoint(
                    decision_checkpoint
                )
                if (
                    invocation_context is None
                    or invocation_context.recovery_claim_id is None
                    or invocation_context.binding.session_id != loaded_interrupted.id
                    or invocation_context.binding.session_instance_id
                    != loaded_interrupted.instance_id
                    or invocation_context.binding.run_epoch != loaded_interrupted.run_epoch
                    or invocation_context.active_profile != active_profile
                    or active_profile is None
                    or active_profile.interaction_id != preserve_interaction_id
                ):
                    raise SessionRunFenced(
                        "Execution-owner replacement lost its exact recovery interaction authority."
                    )

            if (
                terminal_decision is None
                and preserve_interaction_id is None
                and loaded_interrupted.status is SessionStatus.INTERRUPTING
                and active_invocation_execution_profile_from_checkpoint(decision_checkpoint)
                is not None
                and not user_input_supersession_retained
                and _pending_interaction_action_kind(
                    decision_checkpoint, run_epoch=loaded_interrupted.run_epoch
                )
                is None
                and self.supports_terminal_interaction_publication_protocol()
            ):
                terminal_decision = await finalization.await_operation(
                    lambda: self.ensure_interruption_terminal_decision(
                        session=loaded_interrupted,
                        terminal_payload=payload,
                        interruption_request_id=interruption_request_id,
                    ),
                    operation_name="Live post-dispatch terminal-decision election",
                )
                decision_checkpoint = await finalization.await_operation(
                    lambda: self.session_store.load_checkpoint(session.id),
                    operation_name="Live interruption elected terminal-decision read",
                )
            if terminal_decision is not None:
                decision_active_profile = active_invocation_execution_profile_from_checkpoint(
                    decision_checkpoint
                )
                if (
                    terminal_decision.outcome is not InvocationTerminalOutcome.INTERRUPTED
                    or decision_active_profile is None
                    or decision_active_profile.session_id != loaded_interrupted.id
                    or decision_active_profile.run_epoch != loaded_interrupted.run_epoch
                    or not invocation_terminal_decision_matches_recovery_profile(
                        terminal_decision,
                        session_id=loaded_interrupted.id,
                        session_instance_id=loaded_interrupted.instance_id,
                        current_run_epoch=loaded_interrupted.run_epoch,
                        interaction_id=decision_active_profile.interaction_id,
                        execution_profile_fingerprint=(decision_active_profile.profile.fingerprint),
                    )
                    or terminal_decision.interruption_request_id != interruption_request_id
                ):
                    raise SessionRunFenced("Live interruption lost its exact terminal decision.")
                interruption_turn_events: tuple[Event, ...] = ()
                if run_started_at is not None and turn_usage_tracker is not None:
                    interruption_turn_events = await self.emit_turn_completed_once(
                        session=loaded_interrupted,
                        registered_agent=registered_agent,
                        environment_name=environment_name,
                        status=SessionStatus.INTERRUPTED,
                        run_started_at=run_started_at,
                        usage_tracker=turn_usage_tracker,
                        active_run=active_run,
                        invocation_context=invocation_context,
                        settle_interaction=False,
                    )
                decided_terminal_event = _runtime_interruption_event(
                    Event(
                        id=terminal_decision.terminal_event_id,
                        type=EventType.SESSION_INTERRUPTED,
                        session_id=loaded_interrupted.id,
                        agent_name=registered_agent.spec.name,
                        environment_name=environment_name,
                        timestamp=terminal_decision.observed_at,
                        payload=payload,
                    ),
                    execution_profile=execution_profile,
                )
                (
                    terminal_finalize_result,
                    prepared_terminal_event,
                ) = await self.prepare_terminal_event_for_atomic_transition(
                    event=decided_terminal_event,
                    session=loaded_interrupted,
                    registered_environment=registered_environment,
                    execution_profile=execution_profile,
                    invocation_context=invocation_context,
                )
                decided_interaction_event: Event | None = None
                if terminal_decision.interaction_event_id is None:
                    (
                        loaded_interrupted,
                        prepared_terminal_event,
                    ) = await finalization.await_operation(
                        lambda: self.publish_closed_interaction_terminal_decision(
                            session=loaded_interrupted,
                            decision=terminal_decision,
                            terminal_event=prepared_terminal_event,
                        ),
                        operation_name=(
                            "Live interruption closed-interaction terminal publication"
                        ),
                    )
                else:
                    (
                        loaded_interrupted,
                        decided_interaction_event,
                        _,
                    ) = await finalization.await_operation(
                        lambda: self.publish_sibling_interaction_transition(
                            session=loaded_interrupted,
                            invocation_context=invocation_context,
                            registered_agent=registered_agent,
                            registered_environment=registered_environment,
                            environment_name=environment_name,
                            to_status=SessionStatus.INTERRUPTED,
                            from_statuses={SessionStatus.INTERRUPTING},
                            observed_at=terminal_decision.observed_at,
                            event_id=terminal_decision.interaction_event_id,
                            execution_profile=execution_profile,
                            finalize_unsettled_cancellation=False,
                            terminal_event=prepared_terminal_event,
                            terminal_decision=terminal_decision,
                            expected_recovery_claim_id=(finalization.claim_id or recovery_claim_id),
                        ),
                        operation_name="Live interruption atomic terminal publication",
                    )
                if decided_interaction_event is not None:
                    yield decided_interaction_event
                for turn_event in interruption_turn_events:
                    yield turn_event
                await self.clear_pending_session_interrupt(
                    session.id,
                    expected_payload=(payload if exact_interrupt_marker_retained else None),
                    expected_run_epoch=(
                        loaded_interrupted.run_epoch if exact_interrupt_marker_retained else None
                    ),
                )
                if not interruption_cascade_suppressed():
                    self.schedule_background_interruption_cascade(
                        parent_session_id=session.id,
                        interrupt_payload=prepared_terminal_event.payload,
                        create_if_missing=False,
                    )
                async for emitted in self.emit_atomically_persisted_terminal_event_with_hooks(
                    finalize_result=terminal_finalize_result,
                    event=prepared_terminal_event,
                    phase=RuntimeHookPhase.AFTER_SESSION_INTERRUPTED,
                    session=loaded_interrupted,
                    registered_agent=registered_agent,
                    registered_environment=registered_environment,
                    execution_profile=execution_profile,
                    invocation_context=invocation_context,
                    run_runtime_hooks=run_terminal_hooks,
                ):
                    yield emitted
                return
            interaction_event: Event | None = None
            if preserve_interaction_id is not None:
                assert invocation_context is not None
                recovery_claim_id = invocation_context.recovery_claim_id
                assert recovery_claim_id is not None
                loaded_interrupted = await finalization.await_operation(
                    lambda: self._transition_status_under_terminal_finalization_claim(
                        session=loaded_interrupted,
                        from_statuses={loaded_interrupted.status},
                        to_status=SessionStatus.INTERRUPTED,
                        claim_id=recovery_claim_id,
                    ),
                    operation_name="Execution-owner replacement epoch interruption",
                )
            elif loaded_interrupted.status != SessionStatus.INTERRUPTED:
                (
                    loaded_interrupted,
                    interaction_event,
                    _,
                ) = await finalization.await_operation(
                    lambda: self.publish_sibling_interaction_transition(
                        session=loaded_interrupted,
                        invocation_context=invocation_context,
                        registered_agent=registered_agent,
                        registered_environment=registered_environment,
                        environment_name=environment_name,
                        to_status=SessionStatus.INTERRUPTED,
                        execution_profile=execution_profile,
                        finalize_unsettled_cancellation=False,
                        expected_recovery_claim_id=(finalization.claim_id or recovery_claim_id),
                    ),
                    operation_name="Live interruption interaction transition",
                )
            if (
                preserve_interaction_id is None
                and interaction_event is None
                and _current_session_interaction_id(session.id) is not None
            ):
                _, interaction_event, _ = await finalization.await_operation(
                    lambda: self.publish_sibling_interaction_transition(
                        session=loaded_interrupted,
                        invocation_context=invocation_context,
                        registered_agent=registered_agent,
                        registered_environment=registered_environment,
                        environment_name=environment_name,
                        to_status=SessionStatus.INTERRUPTED,
                        from_statuses={SessionStatus.INTERRUPTED},
                        execution_profile=execution_profile,
                        finalize_unsettled_cancellation=False,
                        expected_recovery_claim_id=(finalization.claim_id or recovery_claim_id),
                    ),
                    operation_name="Live interruption residual interaction transition",
                )
            if interaction_event is not None:
                yield interaction_event

            async def finalize_interrupted_session(
                loaded_interrupted: Session,
            ) -> AsyncGenerator[Event, None]:
                existing_interrupt_event = await self._session_control.wait_for_interrupted_event(
                    session.id,
                    interruption_request_id=interruption_request_id,
                )
                if existing_interrupt_event is not None:
                    if exact_interrupt_marker_retained:
                        require_interruption_event_matches_pending_marker(
                            existing_interrupt_event,
                            payload,
                        )
                    if user_input_supersession_retained:
                        if finalization.claim_id is None:
                            raise RuntimeError(
                                "User-input supersession finalization lost its durable owner."
                            )
                        await self.clear_claimed_pending_interrupt_if_retained(
                            session_id=session.id,
                            claim_id=finalization.claim_id,
                            expected_payload=payload,
                        )
                    else:
                        await self.clear_pending_session_interrupt(
                            session.id,
                            expected_payload=(payload if exact_interrupt_marker_retained else None),
                            expected_run_epoch=(
                                loaded_interrupted.run_epoch
                                if exact_interrupt_marker_retained
                                else None
                            ),
                        )
                    if not interruption_cascade_suppressed():
                        self.schedule_background_interruption_cascade(
                            parent_session_id=session.id,
                            interrupt_payload=existing_interrupt_event.payload,
                            create_if_missing=False,
                        )
                    turn_completed_event = (
                        active_run.turn_completed_event
                        if active_run is not None and active_run.turn_completed_event is not None
                        else self._session_control.active_turn_completed_event(session.id)
                    )
                    if turn_completed_event is not None:
                        yield turn_completed_event
                    yield existing_interrupt_event
                    return
                if run_started_at is not None and turn_usage_tracker is not None:
                    for turn_event in await self.emit_turn_completed_once(
                        session=loaded_interrupted,
                        registered_agent=registered_agent,
                        environment_name=environment_name,
                        status=SessionStatus.INTERRUPTED,
                        run_started_at=run_started_at,
                        usage_tracker=turn_usage_tracker,
                        active_run=active_run,
                        invocation_context=invocation_context,
                    ):
                        yield turn_event
                terminal_event_publisher: Callable[[Event], Awaitable[Event]] | None = None
                if user_input_supersession_retained:
                    if finalization.claim_id is None:
                        raise RuntimeError(
                            "User-input supersession finalization lost its durable owner."
                        )
                    owned_claim_id = finalization.claim_id

                    async def publish_owned_terminal_event(event: Event) -> Event:
                        return await self.publish_terminal_event_under_finalization_claim(
                            event=event,
                            session=loaded_interrupted,
                            claim_id=owned_claim_id,
                            expected_payload=payload,
                        )

                    terminal_event_publisher = publish_owned_terminal_event
                terminal_event_stream = self._terminal_event_publication.emit(
                    event=_runtime_interruption_event(
                        Event(
                            type=EventType.SESSION_INTERRUPTED,
                            session_id=loaded_interrupted.id,
                            agent_name=registered_agent.spec.name,
                            environment_name=environment_name,
                            payload=payload,
                        ),
                        execution_profile=execution_profile,
                    ),
                    phase=RuntimeHookPhase.AFTER_SESSION_INTERRUPTED,
                    session=loaded_interrupted,
                    registered_agent=registered_agent,
                    registered_environment=registered_environment,
                    execution_profile=execution_profile,
                    invocation_context=invocation_context,
                    terminal_event_publisher=terminal_event_publisher,
                    run_runtime_hooks=run_terminal_hooks,
                )
                terminal_prefix, interrupted_event = await _collect_through_event_type(
                    terminal_event_stream,
                    EventType.SESSION_INTERRUPTED,
                    missing_message="Session interruption produced no terminal event.",
                )

                if exact_interrupt_marker_retained:
                    require_interruption_event_matches_pending_marker(
                        interrupted_event,
                        payload,
                    )
                if user_input_supersession_retained:
                    if finalization.claim_id is None:
                        raise RuntimeError(
                            "User-input supersession finalization lost its durable owner."
                        )
                    await self.clear_claimed_pending_interrupt_if_retained(
                        session_id=session.id,
                        claim_id=finalization.claim_id,
                        expected_payload=payload,
                    )
                else:
                    await self.clear_pending_session_interrupt(
                        session.id,
                        expected_payload=(payload if exact_interrupt_marker_retained else None),
                        expected_run_epoch=(
                            loaded_interrupted.run_epoch
                            if exact_interrupt_marker_retained
                            else None
                        ),
                    )
                if not interruption_cascade_suppressed():
                    self.schedule_background_interruption_cascade(
                        parent_session_id=session.id,
                        interrupt_payload=interrupted_event.payload,
                        create_if_missing=False,
                    )
                for event in terminal_prefix:
                    yield event
                async for event in terminal_event_stream:
                    yield event

            if user_input_supersession_retained:
                async with finalization.finalize(
                    session=loaded_interrupted,
                    expected_payload=payload,
                    finalization=finalize_interrupted_session,
                ) as owned:
                    async for event in owned:
                        yield event
            else:
                async for event in finalize_interrupted_session(loaded_interrupted):
                    yield event
        finally:
            try:
                await finalization.release(sys.exception())
            finally:
                self._session_control.end_emitting_interrupted(session.id)

    async def prepare_terminal_event_for_atomic_transition(
        self,
        *,
        event: Event,
        session: Session,
        registered_environment: runtime_records.RegisteredEnvironment | None,
        execution_profile: ExecutionProfileIdentity | None,
        invocation_context: InvocationContext | None,
        expected_run_operation_epoch: int | None = None,
    ) -> tuple[Any, Event]:
        """Finalize external resources and prepare terminal evidence without publishing it."""

        bound = await self._terminal_event_publication.bind_to_run_operation(
            event,
            session=session,
            expected_run_operation_epoch=expected_run_operation_epoch,
        )
        finalize_result = await self._environment_lifecycle.finalize_terminal_event(
            event=bound,
            session=session,
            registered_environment=registered_environment,
            execution_profile=execution_profile,
            invocation_context=invocation_context,
        )
        prepared = self._event_writer.prepare(
            event_with_runtime_envelope_authority(
                finalize_result.event,
                "session_id",
            )
        )
        return finalize_result, prepared

    async def emit_atomically_persisted_terminal_event_with_hooks(
        self,
        *,
        finalize_result: Any,
        event: Event,
        phase: RuntimeHookPhase,
        session: Session,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_environment: runtime_records.RegisteredEnvironment | None,
        execution_profile: ExecutionProfileIdentity | None,
        invocation_context: InvocationContext | None,
        run_runtime_hooks: bool = True,
    ) -> AsyncGenerator[Event, None]:
        """Expose one already-atomic terminal event, then run its terminal hooks."""

        try:
            for binding_event in finalize_result.events:
                yield binding_event
            async for emitted in self._terminal_event_publication.replay(
                event=event,
                phase=phase,
                session=session,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
                execution_profile=execution_profile,
                invocation_context=invocation_context,
                run_runtime_hooks=run_runtime_hooks,
            ):
                yield emitted
        except BaseException as post_finalize_failure:
            cancellation = finalize_result.cancellation
            if cancellation is None or any(
                candidate is cancellation
                for candidate in iter_exception_tree(post_finalize_failure)
            ):
                raise
            if isinstance(post_finalize_failure, Exception):
                if finalize_result.cancellation_requests_consumed:
                    retain_workspace_observation_pending_cancellation_requests(
                        cancellation,
                        finalize_result.cancellation_requests_consumed,
                    )
                raise cancellation from post_finalize_failure
            concurrent_control = BaseExceptionGroup(
                "Terminal finalization received concurrent control after egress parking.",
                [post_finalize_failure, cancellation],
            )
            if finalize_result.cancellation_requests_consumed:
                retain_workspace_observation_pending_cancellation_requests(
                    concurrent_control,
                    finalize_result.cancellation_requests_consumed,
                )
            raise concurrent_control from None
        if finalize_result.cancellation is not None:
            if finalize_result.cancellation_requests_consumed:
                retain_workspace_observation_pending_cancellation_requests(
                    finalize_result.cancellation,
                    finalize_result.cancellation_requests_consumed,
                )
            raise finalize_result.cancellation

    async def finalize_abandoned_session_run(
        self,
        request: RecoveryAbandonedSessionRequest,
    ) -> Event | None:
        """Best-effort finalization for a live session whose event stream closed."""
        if (
            request.interaction_transition_failures
            and request.interaction_transition is not None
            and await self.record_committed_interaction_transition_cancellation(request)
        ):
            return None
        payload: dict[str, Any] = {
            "interruption_type": _INTERRUPTION_TYPE_RUNTIME_INTERRUPTED,
            "reason": _ABANDONED_RUN_REASON,
            "abandoned": True,
        }
        terminal_repair = request.retain_terminal_publication_repair
        retained_repair_payload: dict[str, Any] | None = None
        if not terminal_repair:
            checkpoint = await self.session_store.load_checkpoint(request.session.id)
            marker = (
                None
                if checkpoint is None
                else checkpoint.get(_PENDING_SESSION_INTERRUPT_CHECKPOINT_KEY)
            )
            terminal_repair = (
                isinstance(marker, dict) and marker.get("terminal_publication_repair") is True
            )
            if terminal_repair:
                if not isinstance(marker, dict):
                    raise ValueError("Terminal publication repair marker must be an object.")
                retained_repair_payload = copy_durable_record(
                    marker, "terminal publication repair marker"
                )
        if terminal_repair:
            payload["terminal_publication_repair"] = True
        deadline = effective_deadline(
            current_execution_deadline(), request.session.execution_deadline
        )
        if deadline.expires_at is not None:
            payload["execution_deadline"] = deadline.inspection()
        expired_boundary = request.native_admission_deadline or expired_execution_deadline()
        payload["failure_evidence"] = FailureEvidence(
            classification="deadline" if expired_boundary is not None else "interruption",
            deadline=expired_boundary,
            deadline_phase=(
                "admission"
                if request.native_admission_deadline is not None
                else "in_flight"
                if expired_boundary is not None
                else None
            ),
            session_id=request.session.id,
            run_epoch=request.session.run_epoch,
            secondary_failures=bool(
                request.interaction_transition_failures or request.provider_cancellation_failures
            ),
        ).model_dump(mode="json")
        if request.interaction_transition_failures:
            copied_failures = copy_durable_json_value(
                list(request.interaction_transition_failures),
                "interaction transition cancellation diagnostics",
            )
            if type(copied_failures) is not list:
                raise TypeError("Interaction transition cancellation diagnostics must be a list.")
            payload["interaction_transition_failures"] = copied_failures
        if request.provider_cancellation_failures or terminal_repair:
            if terminal_repair:
                payload["terminal_publication_repair"] = True
            if not request.provider_cancellation_failures:
                payload["interruption_request_id"] = str(uuid4())
            else:
                copied_provider_failures = copy_provider_cancellation_failures(
                    request.provider_cancellation_failures
                )
                payload["provider_cancellation_failures"] = [
                    dict(item) for item in copied_provider_failures
                ]
                payload["interruption_request_id"] = str(uuid4())
            if retained_repair_payload is not None:
                payload = retained_repair_payload
            # The status transition and terminal-event publication are
            # separate durable operations. Persist exact repair authority
            # before either operation so a publication failure or process loss
            # cannot leave an interrupted session with no diagnostic evidence.
            await self.session_store.publish_checkpoint_and_events(
                request.session.id,
                checkpoint_transform=_checkpoint_with_pending_session_interrupt(
                    payload, cascade_created_at=self._clock()
                ),
                events=[],
                expected_statuses={request.session.status},
                expected_run_epoch=request.session.run_epoch,
            )

        async def clear_provider_interrupt_marker(
            *,
            require_interrupted: bool,
        ) -> None:
            if not request.provider_cancellation_failures and not terminal_repair:
                return

            def clear_published_interrupt(
                current_session: Session,
                checkpoint: dict[str, Any] | None,
            ) -> dict[str, Any] | None:
                if checkpoint is None:
                    return None
                updated = copy_durable_record(checkpoint, "checkpoint")
                current = updated.get(_PENDING_SESSION_INTERRUPT_CHECKPOINT_KEY)
                if current is None:
                    return updated
                if current != payload:
                    raise RuntimeError(
                        "Pending provider cancellation interruption identity changed."
                    )
                if require_interrupted and current_session.status is not SessionStatus.INTERRUPTED:
                    raise RuntimeError(
                        "Provider cancellation interruption published before terminal status."
                    )
                updated.pop(_PENDING_SESSION_INTERRUPT_CHECKPOINT_KEY)
                return updated

            await self.session_store.publish_checkpoint_and_events(
                request.session.id,
                checkpoint_transform=clear_published_interrupt,
                events=[],
                expected_statuses=({SessionStatus.INTERRUPTED} if require_interrupted else None),
                expected_run_epoch=request.session.run_epoch,
            )

        try:
            finalized = await self.complete_abandoned_turn(
                RecoveryAbandonedTurnRequest(
                    session=request.session,
                    registered_agent=request.registered_agent,
                    registered_environment=request.registered_environment,
                    environment_name=request.environment_name,
                    run_started_at=request.run_started_at,
                    usage_tracker=request.turn_usage_tracker,
                    active_run=request.active_run,
                    execution_profile=request.execution_profile,
                    invocation_context=request.invocation_context,
                )
            )
        except InteractionLifecyclePublicationRejected:
            raise
        except BaseException as transition_failure:
            try:
                active_model_completion = (
                    await self.session_store.load_active_model_completion_stage(request.session.id)
                )
            except BaseException as inspection_failure:
                add_exception_note_safely(
                    transition_failure,
                    "Active model-completion recovery authority inspection also failed: "
                    f"{type(inspection_failure).__name__}.",
                )
                raise transition_failure from inspection_failure
            if active_model_completion is not None:
                raise
            try:
                finalized = await self.session_store.transition_status(
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
                loaded = await self.session_store.load(request.session.id)
                if loaded is None or loaded.status is not SessionStatus.INTERRUPTED:
                    if loaded is not None:
                        await clear_provider_interrupt_marker(require_interrupted=False)
                    return
                finalized = loaded

        terminal_event: Event | None = None

        async def emit_interrupted() -> None:
            nonlocal terminal_event
            async for emitted in self._terminal_event_publication.publish_recovered(
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
                    invocation_context=request.invocation_context,
                    run_runtime_hooks=request.run_terminal_hooks,
                )
            ):
                if (
                    emitted.type is EventType.SESSION_INTERRUPTED
                    and emitted.session_id == finalized.id
                ):
                    terminal_event = copy_event(emitted)

        if (
            request.interaction_transition_failures
            or request.provider_cancellation_failures
            or terminal_repair
        ):
            # These failures are the only durable explanation for an ambiguous
            # transition that exact readback proved absent. Let the owned
            # cancellation cleanup preserve a publication failure instead of
            # silently discarding both pieces of evidence.
            await emit_interrupted()
            if request.provider_cancellation_failures:
                if terminal_event is None:
                    raise RuntimeError(
                        "Provider cancellation interruption produced no terminal evidence."
                    )
                require_interruption_event_matches_pending_marker(
                    terminal_event,
                    payload,
                )
            await clear_provider_interrupt_marker(require_interrupted=True)
        else:
            with contextlib.suppress(BaseException):
                await emit_interrupted()
        return terminal_event

    async def record_committed_interaction_transition_cancellation(
        self,
        request: RecoveryAbandonedSessionRequest,
    ) -> bool:
        """Record acknowledgement failures only after exact durable readback."""

        if request.interaction_transition is None:
            return False
        return await self.record_durable_interaction_transition_cancellation(
            session=request.session,
            transition=request.interaction_transition,
            failures=request.interaction_transition_failures,
            agent_name=request.registered_agent.spec.name,
            environment_name=request.environment_name,
            expected_recovery_claim_id=(request.interaction_transition_recovery_claim_id),
        )

    async def record_durable_interaction_transition_cancellation(
        self,
        *,
        session: Session,
        transition: InteractionTransitionSpec,
        failures: tuple[dict[str, Any], ...],
        agent_name: str,
        environment_name: str | None,
        expected_recovery_claim_id: str | None,
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
        if expected_recovery_claim_id is None:
            receipt = await self.session_store.load_interaction_transition_receipt(
                session.id,
                transition=expected,
            )
        else:
            receipt = await self.session_store.load_interaction_transition_receipt(
                session.id,
                transition=expected,
                expected_recovery_claim_id=expected_recovery_claim_id,
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
        invocation_context: InvocationContext | None = None,
        run_terminal_hooks: bool = True,
    ) -> None:
        """Idempotently finalize a live session when setup-time streaming is abandoned."""
        try:
            session = await self.session_store.load(session_id)
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
                await self.finalize_abandoned_without_registered_runtime(session.id)
                return
            try:
                registered_environment = self._resolve_registered_environment(
                    session.environment_name
                )
            except Exception:
                await self.finalize_abandoned_without_registered_runtime(session.id)
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
                async for _ in self.interrupt_recovery(
                    RecoveryInterruptionRequest(
                        session=session,
                        registered_agent=registered_agent,
                        registered_environment=registered_environment,
                        environment_name=_environment_name(registered_environment),
                        execution_profile=execution_profile,
                        invocation_context=invocation_context,
                        run_terminal_hooks=run_terminal_hooks,
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
                    invocation_context=invocation_context,
                    run_terminal_hooks=run_terminal_hooks,
                )
            )
        except InteractionLifecyclePublicationRejected:
            if propagate_interaction_publication_rejection:
                raise
        except BaseException:
            pass

    async def finalize_abandoned_without_registered_runtime(self, session_id: str) -> None:
        try:
            finalized = await self.session_store.transition_status(
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
            checkpoint = await self.session_store.load_checkpoint(finalized.id)
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
                await self._terminal_finalization.clear_run_operation(
                    session_id=finalized.id,
                    operation=run_operation,
                    terminal_evidence_durable=True,
                )

    def fail_recovered_provider_operation(
        self,
        request: ProviderOperationFailureRequest,
    ) -> AsyncIterator[Event]:
        return self.fail_provider_operation_resolution(
            resolution_event=request.resolution_event,
            session=request.session,
            registered_agent=request.registered_agent,
            registered_environment=request.registered_environment,
            execution_profile=request.execution_profile,
            task_id=request.task_id,
            task_worker_id=request.task_worker_id,
            task_handoff_id=request.task_handoff_id,
            legacy_resolution_without_profile=request.legacy_resolution_without_profile,
            invocation_context=request.invocation_context,
        )

    def stop_recovered_session_for_limit(
        self,
        request: RecoveryLimitStopRequest,
    ) -> AsyncIterator[Event]:
        return self.stop_session_for_limit_reached(
            session=request.session,
            registered_agent=request.registered_agent,
            registered_environment=request.registered_environment,
            environment_name=request.environment_name,
            decision=request.decision,
            usage_summary=request.usage_summary,
            cost_summary=request.cost_summary,
            messages=request.messages,
            tool_calls=request.tool_calls,
            completed_tool_outcomes=request.completed_tool_outcomes,
            pending_approval_to_clear=request.pending_approval_to_clear,
            deferred_messages=request.deferred_messages,
            requested_approval_decision=request.requested_approval_decision,
            approval_resolution_request_digest=request.approval_resolution_request_digest,
            execution_profile=request.execution_profile,
            invocation_context=request.invocation_context,
            reconcile_transition_cancellation=True,
        )

    def interrupt_recovery(
        self,
        request: RecoveryInterruptionRequest,
    ) -> AsyncIterator[Event]:
        return self.handle_session_interrupted(
            session=request.session,
            registered_agent=request.registered_agent,
            registered_environment=request.registered_environment,
            environment_name=request.environment_name,
            execution_profile=request.execution_profile,
            invocation_context=request.invocation_context,
            run_terminal_hooks=request.run_terminal_hooks,
            preserve_interaction_id=request.preserve_interaction_id,
            recovery_claim_id=request.recovery_claim_id,
        )

    async def complete_abandoned_turn(
        self,
        request: RecoveryAbandonedTurnRequest,
    ) -> Session:
        finalized, _, _ = await self.publish_sibling_interaction_transition(
            session=request.session,
            invocation_context=request.invocation_context,
            registered_agent=request.registered_agent,
            registered_environment=request.registered_environment,
            environment_name=request.environment_name,
            to_status=SessionStatus.INTERRUPTED,
            execution_profile=request.execution_profile,
        )
        if request.run_started_at is not None and request.usage_tracker is not None:
            await self.emit_turn_completed_once(
                session=finalized,
                registered_agent=request.registered_agent,
                environment_name=request.environment_name,
                status=SessionStatus.INTERRUPTED,
                run_started_at=request.run_started_at,
                usage_tracker=request.usage_tracker,
                active_run=request.active_run,
                invocation_context=request.invocation_context,
            )
        return finalized

    async def resume_recovery_interaction(
        self,
        session: Session,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_environment: runtime_records.RegisteredEnvironment | None,
    ) -> Event | None:
        return await self.resume_interaction(
            session,
            registered_agent,
            registered_environment,
        )

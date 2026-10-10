"""Explicit compaction with session-operation claim ownership."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import threading
import time
from collections.abc import AsyncGenerator, Awaitable, Callable, Iterable
from datetime import datetime, timedelta
from typing import Any
from uuid import uuid4

from cayu._exception_groups import (
    exception_cause,
)
from cayu._task_wait import (
    LateFailures,
    ShieldedTaskOutcome,
    await_shielded_task_outcome,
    retained_task_failure,
    unexpected_child_cancellation_error,
    wait_until_idle,
)
from cayu._validation import (
    copy_durable_record,
    copy_json_value,
    require_clean_nonblank,
)
from cayu.approvals.tools import (
    resolution_actor_payload,
)
from cayu.budgets.base import (
    BudgetCheck,
    BudgetLimit,
    BudgetPolicy,
    BudgetReservationResult,
    _effective_budget_limit_id,
    budget_check_payload,
    budget_limits_for_session,
    budget_reservation_payload,
    copy_budget_policy,
    has_deferred_contextual_price,
    request_budget_limits_for_session,
)
from cayu.budgets.billing import (
    UNRESOLVED_BILLING_IDENTITY,
    BillingIdentity,
    BillingIdentityState,
    ResolvedBillingIdentity,
    resolved_billing_identity,
)
from cayu.budgets.run_limits import has_run_limits
from cayu.budgets.usage import (
    ModelCompletionPurpose,
)
from cayu.context.base import (
    _COMPACTION_ATTEMPT_ID_KEY,
    CheckpointCompactionContextPolicy,
    ContextBuildError,
    ContextBuildResult,
    ContextCompactionTelemetry,
    ContextRequest,
    _attach_context_build_termination_diagnostics,
    _automatic_compaction_dispatch_runner_scope,
    _compaction_completion_publisher_scope,
    _compaction_model_attempt_identity_scope,
    _compaction_model_completed_payload,
    _CompactionAccountingUsageError,
    _context_secret_redactor_scope,
    _defer_billing_identity_cancellation_scope,
    _durable_compaction_completion_evidence,
    context_build_termination_compaction_telemetry,
    project_compaction_invocation_checkpoint,
    sanitize_context_build_error_checkpoint,
    sanitize_context_build_result_checkpoint,
    sanitize_context_compaction_telemetry,
)
from cayu.context.footprints import (
    RequestFootprintConfig,
    RequestVariant,
    analyze_request_footprint,
    copy_request_footprint_config,
)
from cayu.events import (
    Event,
    EventType,
    copy_event,
    event_with_runtime_payload_authority,
)
from cayu.execution_profiles import (
    ExecutionProfileComponentClass,
    ExecutionProfileIdentity,
    changed_execution_profile_components,
    event_with_execution_profile_authority,
    execution_profile_with_component,
)
from cayu.execution_units import (
    ModelAttemptIdentity,
    ModelStepIdentity,
    copy_model_attempt_identity,
    copy_model_step_identity,
    new_model_step_identity,
    strip_runtime_owned_execution_identity,
)
from cayu.providers.base import (
    ModelProvider,
    ModelRequest,
    UsageDialect,
    _detach_model_request,
    copy_usage_dialect,
)
from cayu.runtime import (
    _prompt_transition,
    _session_checkpoint_admission,
    _session_execution_profile,
    _session_operation_state,
    _transcript,
)
from cayu.runtime import _runtime_records as runtime_records
from cayu.runtime import _session_request_boundary as session_request_boundary
from cayu.runtime._compaction.identity import _CompactionExecutionIdentityLedger
from cayu.runtime._durable_tool_round import _environment_name
from cayu.runtime._event_writer import (
    RuntimeEventWriter,
)
from cayu.runtime._model_execution_selection import _session_agent_spec
from cayu.runtime._recovery_ownership import (
    RecoveryOwnership,
    _checkpoint_without_active_incomplete_recovery_claim,
)
from cayu.runtime._run_limits import (
    BudgetReservationIdentityGuard,
    BudgetReservationLeaseLost,
    BudgetReservationLeaseLostBeforeModelDispatch,
    BudgetStepReservation,
    RunLimitController,
    add_budget_failure_note,
)
from cayu.runtime._task_execution_admission import verifier_aware_task_execution_outcome
from cayu.runtime._task_store_operation_boundary import (
    raise_task_store_operation_failure,
)
from cayu.runtime.execution_identity import ExecutionProfileBehaviorIdentity
from cayu.runtime.execution_profiles import (
    ExecutionProfileMismatchError,
)
from cayu.runtime.loop_policies import (
    LoopPolicy,
)
from cayu.sessions._execution_profile_checkpoint import (
    execution_profile_from_session_metadata,
)
from cayu.sessions._execution_profile_rules import session_model_projection_cursor
from cayu.sessions.base import (
    SessionOperationPublication,
    SessionStore,
    _incomplete_recovery_claim_from_checkpoint,
    attribute_event_to_current_interaction,
    attribute_events_to_current_interaction,
)
from cayu.sessions.event_queries import EventQuery
from cayu.sessions.records import (
    Session,
)
from cayu.sessions.requests import (
    CompactSessionRequest,
    copy_compact_session_request,
)
from cayu.tasks.contracts import TaskCompletionDecisionRequired
from cayu.tasks.store import TaskStore
from cayu.vaults.redaction import SecretRedactor


class _SessionCompactionReplay(Exception):
    def __init__(self, event_ids: Iterable[str]) -> None:
        self.event_ids = tuple(event_ids)
        super().__init__("Replay an existing durable session compaction outcome.")


class SessionCompactionAttemptSuperseded(RuntimeError):
    """A recovered compaction attempt owns the durable operation claim."""


_CONTEXT_COMPACTION_OPERATION_KIND = "context_compaction"

_SESSION_OPERATION_CLAIM_LEASE = timedelta(minutes=5)

_SESSION_OPERATION_CLAIM_HEARTBEAT_INTERVAL_SECONDS = 30.0

_SESSION_OPERATION_CLAIM_HEARTBEAT_RETRY_SECONDS = 5.0

_SESSION_OPERATION_CLAIM_HEARTBEAT_STOP_TIMEOUT_SECONDS = 0.1

_SESSION_OPERATION_STORE_WAIT_TIMEOUT_SECONDS = 5.0


def _session_operation_heartbeat_failure(task: asyncio.Task[None]) -> BaseException:
    if task.cancelled():
        try:
            task.result()
        except asyncio.CancelledError as cancellation:
            concurrent_failure = exception_cause(cancellation)
            if concurrent_failure is not None:
                return concurrent_failure
        return SessionCompactionAttemptSuperseded(
            "Session compaction operation heartbeat was cancelled unexpectedly."
        )
    failure = task.exception()
    if failure is None:
        return SessionCompactionAttemptSuperseded(
            "Session compaction operation heartbeat stopped unexpectedly."
        )
    return failure


def _completed_session_operation_child_failure(
    task: asyncio.Task[Any],
    *,
    operation: str,
) -> BaseException | None:
    try:
        task.result()
    except asyncio.CancelledError as child_cancellation:
        return unexpected_child_cancellation_error(
            child_cancellation,
            operation=operation,
        )
    except BaseException as child_failure:
        return child_failure
    return None


def _consume_detached_session_operation_task(task: asyncio.Task[Any]) -> None:
    """Observe a renewal task that was detached after its lease deadline."""

    # Cancelled tasks do not emit an un-retrieved-exception warning, and calling
    # result() here would consume their cancellation message before the owning
    # heartbeat can classify and preserve it.
    if task.cancelled():
        return
    with contextlib.suppress(BaseException):
        task.result()


class _SessionOperationClaimHeartbeatState:
    """Expose a heartbeat store call that may still own the session boundary."""

    def __init__(self) -> None:
        self.pending_store_task: asyncio.Task[Any] | None = None
        self.confirmed_claim_expires_at: datetime | None = None
        self.claim_deadline_monotonic: float | None = None

    def confirm_claim(
        self,
        expires_at: datetime,
        *,
        claim_deadline_monotonic: float,
    ) -> None:
        confirmed = self.confirmed_claim_expires_at
        if confirmed is None or expires_at > confirmed:
            self.confirmed_claim_expires_at = expires_at
            self.claim_deadline_monotonic = claim_deadline_monotonic

    def remaining_claim_seconds(self) -> float:
        deadline = self.claim_deadline_monotonic
        if deadline is None:
            raise AssertionError("Session operation claim has no local monotonic deadline.")
        return deadline - time.monotonic()

    def observe_store_task(self, task: asyncio.Task[Any]) -> None:
        if self.pending_store_task is task:
            self.pending_store_task = None
        _consume_detached_session_operation_task(task)


async def _run_while_session_operation_claimed(
    operation: Callable[[], Awaitable[tuple[ContextBuildResult, BaseException | None]]],
    *,
    heartbeat_task: asyncio.Task[None],
) -> tuple[ContextBuildResult, BaseException | None]:
    heartbeat_observed = False
    heartbeat_failure: BaseException | None = None

    def observe_heartbeat_failure() -> BaseException:
        nonlocal heartbeat_failure
        nonlocal heartbeat_observed
        if not heartbeat_observed:
            heartbeat_failure = _session_operation_heartbeat_failure(heartbeat_task)
            heartbeat_observed = True
        if heartbeat_failure is None:
            raise AssertionError("Completed session operation heartbeat had no outcome.")
        return heartbeat_failure

    async def run_operation() -> tuple[ContextBuildResult, BaseException | None]:
        if heartbeat_task.done():
            raise observe_heartbeat_failure()
        return await operation()

    if heartbeat_task.done():
        raise observe_heartbeat_failure()
    operation_task = asyncio.create_task(run_operation())
    try:
        done, _pending = await asyncio.wait(
            {operation_task, heartbeat_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
        if heartbeat_task not in done:
            try:
                return operation_task.result()
            except asyncio.CancelledError as child_cancellation:
                raise unexpected_child_cancellation_error(
                    child_cancellation,
                    operation="Session compaction operation",
                ) from child_cancellation

        heartbeat_failure = observe_heartbeat_failure()
        if not operation_task.done():
            operation_task.cancel()
        outcome = await await_shielded_task_outcome(operation_task)
        if outcome.result is not None:
            _attach_context_build_termination_diagnostics(
                heartbeat_failure,
                compaction_telemetry=outcome.result[0].compaction_telemetry,
            )
        if outcome.error is not None:
            _preserve_context_build_termination_diagnostics(
                outcome.error,
                heartbeat_failure,
            )
        if outcome.cancellation is not None:
            outcome.cancellation.add_note(
                "Session compaction operation ownership was also lost: "
                f"{type(heartbeat_failure).__name__}: {heartbeat_failure}"
            )
            raise outcome.cancellation from heartbeat_failure
        if outcome.error is not None and not isinstance(
            outcome.error,
            asyncio.CancelledError,
        ):
            heartbeat_failure.add_note(
                "Compaction also failed while operation ownership loss was handled: "
                f"{type(outcome.error).__name__}: {outcome.error}"
            )
            raise heartbeat_failure from outcome.error
        raise heartbeat_failure
    except asyncio.CancelledError as exc:
        if not operation_task.done():
            operation_task.cancel()
        outcome = await await_shielded_task_outcome(
            operation_task,
            cancellation=exc,
        )
        cancellation = outcome.cancellation or exc
        if outcome.result is not None:
            _attach_context_build_termination_diagnostics(
                cancellation,
                compaction_telemetry=outcome.result[0].compaction_telemetry,
            )
        if outcome.error is not None:
            _preserve_context_build_termination_diagnostics(
                outcome.error,
                cancellation,
            )
        if heartbeat_task.done():
            heartbeat_failure = observe_heartbeat_failure()
            cancellation.add_note(
                "Session compaction operation ownership was also lost during caller "
                f"cancellation: {type(heartbeat_failure).__name__}: {heartbeat_failure}"
            )
        if outcome.error is not None and not isinstance(
            outcome.error,
            asyncio.CancelledError,
        ):
            cancellation.add_note(
                "Compaction also failed while caller cancellation was handled: "
                f"{type(outcome.error).__name__}: {outcome.error}"
            )
        raise cancellation from exception_cause(cancellation)
    finally:
        if not operation_task.done():
            operation_task.cancel()
            await asyncio.gather(operation_task, return_exceptions=True)


def _preserve_context_build_termination_diagnostics(
    source: BaseException,
    target: BaseException,
) -> None:
    telemetry = (
        source.compaction_telemetry
        if isinstance(source, ContextBuildError)
        else context_build_termination_compaction_telemetry(source)
    )
    if telemetry:
        _attach_context_build_termination_diagnostics(
            target,
            compaction_telemetry=list(telemetry),
        )


def _compact_session_request_digest(request: CompactSessionRequest) -> str:
    payload = request.model_dump(mode="json")
    # The authenticated caller is audit data for each attempt, not part of the
    # operation's semantic identity. A different operator must be able to
    # recover the same idempotent request after its lease expires.
    payload.pop("requested_by", None)
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _optional_text_digest(value: str | None) -> str | None:
    if value is None:
        return None
    return hashlib.sha256(value.encode()).hexdigest()


def _session_compaction_replay_event_ids(
    record: dict[str, Any],
) -> tuple[str, ...] | None:
    status = record.get("status")
    if status in {"running", "abandoned"}:
        return None
    if status not in {"completed", "failed"}:
        raise RuntimeError(f"Unsupported session compaction operation status: {status}")
    stored_event_ids = record.get("event_ids")
    if type(stored_event_ids) is not list or not all(
        type(event_id) is str and event_id for event_id in stored_event_ids
    ):
        raise RuntimeError("Completed session compaction operation is missing replay event ids.")
    return tuple(stored_event_ids)


def _application_compaction_causal_payload(
    *,
    request: CompactSessionRequest,
    operation_id: str,
    attempt_id: str,
    source_cursor: int,
    compactor: str,
    result_cursor: Any = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "operation_id": operation_id,
        "attempt_id": attempt_id,
        "request_id": request.idempotency_key,
        "reason": request.reason,
        "source_run_epoch": request.expected_run_epoch,
        "source_transcript_cursor": source_cursor,
        "compactor": _application_compaction_event_text(compactor) or "ContextCompactor",
        "mode": "bounded",
        "instruction_present": request.instructions is not None,
        "instruction_digest": _optional_text_digest(request.instructions),
        "actor": resolution_actor_payload(request.requested_by),
    }
    if type(result_cursor) is int and result_cursor >= 0:
        payload["result_transcript_cursor"] = result_cursor
    return payload


_APPLICATION_COMPACTION_EVENT_TEXT_MAX_BYTES = 512


def _application_compaction_event_text(value: Any) -> str | None:
    if type(value) is not str or not value or value != value.strip():
        return None
    if any(
        0xD800 <= ord(char) <= 0xDFFF or ord(char) < 0x20 or ord(char) == 0x7F for char in value
    ):
        return None
    if len(value.encode("utf-8")) > _APPLICATION_COMPACTION_EVENT_TEXT_MAX_BYTES:
        return None
    return value


def _model_execution_identity_payload(
    identity: ModelStepIdentity | ModelAttemptIdentity,
) -> dict[str, str]:
    if type(identity) is ModelAttemptIdentity:
        return copy_model_attempt_identity(identity).payload()
    if type(identity) is ModelStepIdentity:
        return copy_model_step_identity(identity).payload()
    raise TypeError("Model execution identity has an unsupported type.")


def _require_application_compaction_event_text(value: Any, field_name: str) -> str:
    value = require_clean_nonblank(value, field_name)
    if _application_compaction_event_text(value) is None:
        raise ValueError(
            f"`{field_name}` must contain valid Unicode without control characters "
            f"and be at most {_APPLICATION_COMPACTION_EVENT_TEXT_MAX_BYTES} UTF-8 bytes."
        )
    return value


def _application_compaction_event(
    *,
    telemetry: ContextCompactionTelemetry,
    request: CompactSessionRequest,
    operation_id: str,
    attempt_id: str,
    session: Session,
    registered_agent: runtime_records.RegisteredAgentState,
    environment_name: str | None,
    compactor: str,
    execution_identity: ModelStepIdentity | ModelAttemptIdentity | None = None,
) -> Event:
    sanitized = sanitize_context_compaction_telemetry(telemetry)
    payload = copy_json_value(sanitized.payload, "compaction_telemetry_payload")
    strip_runtime_owned_execution_identity(payload)
    if sanitized.event_type == EventType.CONTEXT_COMPACTION_FAILED:
        payload.pop("compacted_transcript_cursor", None)
    if execution_identity is not None:
        payload.update(_model_execution_identity_payload(execution_identity))
    payload.update(
        _application_compaction_causal_payload(
            request=request,
            operation_id=operation_id,
            attempt_id=attempt_id,
            source_cursor=request.expected_transcript_cursor,
            result_cursor=payload.get("compacted_transcript_cursor"),
            compactor=compactor,
        )
    )
    return Event(
        type=sanitized.event_type,
        session_id=session.id,
        agent_name=registered_agent.spec.name,
        environment_name=environment_name,
        payload=payload,
    )


def _application_compaction_budget_event(
    *,
    check: BudgetCheck,
    request: CompactSessionRequest,
    operation_id: str,
    attempt_id: str,
    session: Session,
    registered_agent: runtime_records.RegisteredAgentState,
    environment_name: str | None,
    compactor: str,
    execution_identity: ModelStepIdentity | ModelAttemptIdentity,
) -> Event:
    return Event(
        type=EventType.BUDGET_LIMIT_REACHED,
        session_id=session.id,
        agent_name=registered_agent.spec.name,
        environment_name=environment_name,
        payload={
            **budget_check_payload(check),
            **_model_execution_identity_payload(execution_identity),
            **_application_compaction_causal_payload(
                request=request,
                operation_id=operation_id,
                attempt_id=attempt_id,
                source_cursor=request.expected_transcript_cursor,
                compactor=compactor,
            ),
        },
    )


def _application_compaction_ledger_event(
    *,
    event_type: EventType,
    payload: dict[str, Any],
    request: CompactSessionRequest,
    operation_id: str,
    attempt_id: str,
    session: Session,
    registered_agent: runtime_records.RegisteredAgentState,
    environment_name: str | None,
    compactor: str,
) -> Event:
    return Event(
        type=event_type,
        session_id=session.id,
        agent_name=registered_agent.spec.name,
        environment_name=environment_name,
        payload={
            **copy_json_value(payload, "compaction_budget_payload"),
            **_application_compaction_causal_payload(
                request=request,
                operation_id=operation_id,
                attempt_id=attempt_id,
                source_cursor=request.expected_transcript_cursor,
                compactor=compactor,
            ),
        },
    )


def _budget_check_identity(
    check: BudgetCheck,
) -> str:
    return check.budget_limit_id


def _complete_session_operation_checkpoint(
    *,
    checkpoint: dict[str, Any] | None,
    persisted_record: dict[str, Any] | None,
    compacted_checkpoint: dict[str, Any],
    idempotency_key: str,
    operation_id: str,
    attempt_id: str,
    event_ids: list[str],
    result_cursor: Any,
    completed_at: datetime,
    on_terminalize: Callable[[datetime], None],
) -> SessionOperationPublication:
    if completed_at.tzinfo is None or completed_at.utcoffset() is None:
        raise ValueError("Session compaction completion time must be timezone-aware.")
    if checkpoint is None:
        raise SessionCompactionAttemptSuperseded(
            "Session compaction attempt was superseded before publication."
        )
    updated = copy_durable_record(checkpoint, "checkpoint")
    operations = _session_operation_state._session_operation_state(updated)
    record = operations["records"].get(idempotency_key)
    if type(record) is not dict or record.get("operation_id") != operation_id:
        if persisted_record is not None and persisted_record.get("operation_id") == operation_id:
            raise SessionCompactionAttemptSuperseded(
                "Session compaction attempt was superseded before publication."
            )
        raise RuntimeError("Session compaction operation claim was lost before publication.")
    if (
        record.get("status") != "running"
        or record.get("current_attempt_id") != attempt_id
        or operations.get("active_operation_id") != operation_id
    ):
        raise SessionCompactionAttemptSuperseded(
            "Session compaction attempt was superseded before publication."
        )
    claim_expires_at = _session_operation_state._operation_claim_expiry(record)
    if claim_expires_at is None or claim_expires_at <= completed_at:
        raise SessionCompactionAttemptSuperseded(
            "Session compaction operation claim expired before terminal publication."
        )
    if persisted_record is not None:
        if persisted_record.get("operation_id") != operation_id:
            raise RuntimeError(
                "Session compaction operation replay record does not match its claim."
            )
        if persisted_record.get("status") != "abandoned":
            raise SessionCompactionAttemptSuperseded(
                "Session compaction attempt was superseded before publication."
            )
    on_terminalize(claim_expires_at)
    compacted_state = copy_durable_record(compacted_checkpoint, "compacted_checkpoint")
    compacted_context = compacted_state.get(_CONTEXT_COMPACTION_OPERATION_KIND)
    if type(compacted_context) is not dict:
        raise ValueError("Compacted checkpoint is missing context compaction state.")
    updated[_CONTEXT_COMPACTION_OPERATION_KIND] = compacted_context
    record["status"] = "completed"
    existing_event_ids = record.get("event_ids", [])
    record["event_ids"] = [
        *existing_event_ids,
        *(event_id for event_id in event_ids if event_id not in existing_event_ids),
    ]
    record["result_transcript_cursor"] = result_cursor
    record["completed_at"] = completed_at.isoformat()
    record["updated_at"] = completed_at.isoformat()
    record.pop("claim_expires_at", None)
    operations["active_operation_id"] = None
    terminal_record = operations["records"].pop(idempotency_key)
    _session_operation_state._store_session_operation_state(updated, operations)
    return SessionOperationPublication(
        checkpoint=updated,
        operation_records={idempotency_key: terminal_record},
    )


def _renew_session_operation_claim_publication(
    *,
    checkpoint: dict[str, Any] | None,
    idempotency_key: str,
    operation_id: str,
    attempt_id: str,
    event_ids: list[str],
    renewed_at: datetime,
    persisted_record: dict[str, Any] | None,
    on_renew: Callable[[datetime], None] | None = None,
    on_renewed: Callable[[datetime], None] | None = None,
) -> tuple[SessionOperationPublication, datetime]:
    if renewed_at.tzinfo is None or renewed_at.utcoffset() is None:
        raise ValueError("Session compaction claim renewal time must be timezone-aware.")
    if checkpoint is None:
        raise SessionCompactionAttemptSuperseded(
            "Session compaction attempt was superseded before claim renewal."
        )
    updated = copy_durable_record(checkpoint, "checkpoint")
    operations = _session_operation_state._session_operation_state(updated)
    record = operations["records"].get(idempotency_key)
    if type(record) is not dict or record.get("operation_id") != operation_id:
        if persisted_record is not None and persisted_record.get("operation_id") == operation_id:
            raise SessionCompactionAttemptSuperseded(
                "Session compaction attempt was superseded before claim renewal."
            )
        raise RuntimeError("Session compaction operation claim was lost before renewal.")
    if (
        record.get("status") != "running"
        or record.get("current_attempt_id") != attempt_id
        or operations.get("active_operation_id") != operation_id
    ):
        raise SessionCompactionAttemptSuperseded(
            "Session compaction attempt was superseded before claim renewal."
        )
    current_expiry = _session_operation_state._operation_claim_expiry(record)
    if current_expiry is None or current_expiry <= renewed_at:
        raise SessionCompactionAttemptSuperseded(
            "Session compaction operation claim expired before renewal."
        )
    if on_renew is not None:
        on_renew(current_expiry)
    existing_event_ids = record.get("event_ids", [])
    record["event_ids"] = [
        *existing_event_ids,
        *(event_id for event_id in event_ids if event_id not in existing_event_ids),
    ]
    renewed_until = renewed_at + _SESSION_OPERATION_CLAIM_LEASE
    record["updated_at"] = renewed_at.isoformat()
    record["claim_expires_at"] = renewed_until.isoformat()
    if on_renewed is not None:
        on_renewed(renewed_until)
    updated[_session_operation_state._SESSION_OPERATIONS_CHECKPOINT_KEY] = operations
    return SessionOperationPublication(checkpoint=updated), renewed_until


def _append_session_operation_attempt_events(
    *,
    idempotency_key: str,
    operation_id: str,
    attempt_id: str,
    event_ids: list[str],
    clock: Callable[[], datetime],
    on_renew: Callable[[datetime], None],
    on_renewed: Callable[[datetime], None],
) -> Callable[
    [Session, dict[str, Any] | None, dict[str, Any] | None],
    SessionOperationPublication,
]:
    def append_events(
        _session: Session,
        checkpoint: dict[str, Any] | None,
        persisted_record: dict[str, Any] | None,
    ) -> SessionOperationPublication:
        publication, _renewed_until = _renew_session_operation_claim_publication(
            checkpoint=checkpoint,
            idempotency_key=idempotency_key,
            operation_id=operation_id,
            attempt_id=attempt_id,
            event_ids=event_ids,
            renewed_at=clock(),
            persisted_record=persisted_record,
            on_renew=on_renew,
            on_renewed=on_renewed,
        )
        return publication

    return append_events


def _fail_session_operation_checkpoint(
    *,
    idempotency_key: str,
    operation_id: str,
    attempt_id: str,
    failed_event_id: str,
    attempt_event_ids: list[str],
    error_type: str,
    clock: Callable[[], datetime],
    on_terminalize: Callable[[datetime], None],
) -> Callable[
    [Session, dict[str, Any] | None, dict[str, Any] | None],
    SessionOperationPublication,
]:
    def fail(
        _session: Session,
        checkpoint: dict[str, Any] | None,
        persisted_record: dict[str, Any] | None,
    ) -> SessionOperationPublication:
        completed_at = clock()
        if completed_at.tzinfo is None or completed_at.utcoffset() is None:
            raise ValueError("Session compaction failure time must be timezone-aware.")
        updated = {} if checkpoint is None else copy_durable_record(checkpoint, "checkpoint")
        operations = _session_operation_state._session_operation_state(updated)
        record = operations["records"].get(idempotency_key)
        if type(record) is not dict or record.get("operation_id") != operation_id:
            if (
                persisted_record is not None
                and persisted_record.get("operation_id") == operation_id
            ):
                if (
                    persisted_record.get("status") in {"completed", "failed"}
                    and persisted_record.get("current_attempt_id") == attempt_id
                ):
                    raise SessionCompactionAttemptSuperseded(
                        "Session compaction attempt was already terminal before failure "
                        "publication."
                    )
                terminal_record = copy_json_value(persisted_record, "session_operation")
                existing_event_ids = terminal_record.get("event_ids", [])
                terminal_record["event_ids"] = [
                    *existing_event_ids,
                    *(
                        event_id
                        for event_id in [*attempt_event_ids, failed_event_id]
                        if event_id not in existing_event_ids
                    ),
                ]
                return SessionOperationPublication(
                    checkpoint=updated,
                    operation_records={idempotency_key: terminal_record},
                )
            raise RuntimeError("Session compaction operation claim was lost before failure.")
        existing_event_ids = record.get("event_ids", [])
        record["event_ids"] = [
            *existing_event_ids,
            *(
                event_id
                for event_id in [*attempt_event_ids, failed_event_id]
                if event_id not in existing_event_ids
            ),
        ]
        if (
            record.get("status") != "running"
            or record.get("current_attempt_id") != attempt_id
            or operations.get("active_operation_id") != operation_id
        ):
            updated[_session_operation_state._SESSION_OPERATIONS_CHECKPOINT_KEY] = operations
            return SessionOperationPublication(checkpoint=updated)
        claim_expires_at = _session_operation_state._operation_claim_expiry(record)
        if claim_expires_at is None or claim_expires_at <= completed_at:
            updated[_session_operation_state._SESSION_OPERATIONS_CHECKPOINT_KEY] = operations
            return SessionOperationPublication(checkpoint=updated)
        on_terminalize(claim_expires_at)
        record["status"] = "failed"
        record["error_type"] = error_type
        record["completed_at"] = completed_at.isoformat()
        record["updated_at"] = completed_at.isoformat()
        record.pop("claim_expires_at", None)
        operations["active_operation_id"] = None
        terminal_record = operations["records"].pop(idempotency_key)
        _session_operation_state._store_session_operation_state(updated, operations)
        return SessionOperationPublication(
            checkpoint=updated,
            operation_records={idempotency_key: terminal_record},
        )

    return fail


class SessionCompaction:
    """Own explicit compaction, its durable claim, and outstanding accounting."""

    def __init__(
        self,
        *,
        session_store: SessionStore,
        task_store: TaskStore | None,
        get_budget_policy: Callable[[], BudgetPolicy | None],
        event_writer: RuntimeEventWriter,
        run_limit_controller: RunLimitController,
        request_footprint: RequestFootprintConfig,
        recovery_ownership: RecoveryOwnership,
        secret_redactor: SecretRedactor,
        clock: Callable[[], datetime],
        runtime_hooks: tuple[runtime_records.RegisteredRuntimeHook, ...],
        loop_policies: tuple[LoopPolicy, ...],
        loop_policy_execution_profile_identities: tuple[
            ExecutionProfileBehaviorIdentity | None, ...
        ],
        get_registered_agent: Callable[[str], runtime_records.RegisteredAgentState],
        get_registered_provider: Callable[[str | None], runtime_records.RegisteredProvider],
        get_registered_environment_for_session: Callable[
            [str | None], runtime_records.RegisteredEnvironment | None
        ],
        execution_profile_process_identity: str,
    ) -> None:
        self.session_store = session_store
        self._task_store = task_store
        self._get_budget_policy = get_budget_policy
        self._event_writer = event_writer
        self._run_limit_controller = run_limit_controller
        self._request_footprint = copy_request_footprint_config(request_footprint)
        self._recovery_ownership = recovery_ownership
        self._secret_redactor = secret_redactor
        self._clock = clock
        self._runtime_hooks = runtime_hooks
        self._loop_policies = loop_policies
        if len(loop_policies) != len(loop_policy_execution_profile_identities):
            raise ValueError("Loop-policy execution-profile identities are inconsistent.")
        self._loop_policy_execution_profile_identities = loop_policy_execution_profile_identities
        self._get_registered_agent = get_registered_agent
        self._get_registered_provider = get_registered_provider
        self._get_registered_environment_for_session = get_registered_environment_for_session
        self._execution_profile_process_identity = require_clean_nonblank(
            execution_profile_process_identity, "execution_profile_process_identity"
        )
        self._detached_session_operation_tasks: set[asyncio.Task[Any]] = set()
        self._session_operation_failures = LateFailures("Final session accounting")

    async def _require_session(self, session_id: str) -> Session:
        loaded = await self.session_store.load(session_id)
        if loaded is None:
            raise KeyError(f"Session not found: {session_id}") from None
        return loaded

    def _running_session_operations(self) -> set[asyncio.Future[Any]]:
        return set(self._detached_session_operation_tasks)

    @property
    def session_operations_pending(self) -> bool:
        """Whether detached session work still runs, or final accounting failed unreported."""

        return self._session_operation_failures.pending or any(
            not task.done() for task in self._detached_session_operation_tasks
        )

    async def wait_for_session_operations(self, *, timeout_s: float) -> bool:
        """Wait up to ``timeout_s`` for detached session work, without cancelling it."""

        return await wait_until_idle(self._running_session_operations, timeout_s=timeout_s)

    def raise_session_operation_failures(self) -> None:
        """Raise, once, for final session accounting that failed unreported."""

        self._session_operation_failures.raise_once()

    def _track_detached_session_operation_task(
        self, task: asyncio.Task[Any], *, report_failure: bool
    ) -> None:
        """Retain and observe best-effort work that cannot block its caller.

        ``report_failure`` says whether nobody else owns how the task ends, so a
        failure is reported once at shutdown.
        """

        if task in self._detached_session_operation_tasks:
            return
        self._detached_session_operation_tasks.add(task)

        def observe(completed: asyncio.Task[Any]) -> None:
            self._detached_session_operation_tasks.discard(completed)
            if not report_failure:
                _consume_detached_session_operation_task(completed)
                return
            # Reading a cancelled task would consume the cancellation its owning
            # heartbeat classifies; a cancellation is not a failure either.
            failure = retained_task_failure(completed)
            # A superseded attempt now belongs to the owner that took it over.
            if failure is not None and not isinstance(failure, SessionCompactionAttemptSuperseded):
                self._session_operation_failures.record(type(failure).__qualname__)

        task.add_done_callback(observe)

    async def _await_session_operation_store_task(
        self,
        task: asyncio.Task[Any],
        *,
        cancellation: asyncio.CancelledError | None = None,
    ) -> ShieldedTaskOutcome[Any]:
        """Bound a shielded store wait and retain any write that outlives it."""

        outcome = await await_shielded_task_outcome(
            task,
            cancellation=cancellation,
            timeout_s=_SESSION_OPERATION_STORE_WAIT_TIMEOUT_SECONDS,
        )
        if outcome.timed_out:
            task.cancel()
            # Cancelled with an unknown outcome, which its caller reconciles.
            self._track_detached_session_operation_task(task, report_failure=False)
        return outcome

    async def _fan_out_reconciled_session_operation_events(
        self,
        events: list[Event],
        *,
        cancellation: asyncio.CancelledError | None = None,
    ) -> tuple[BaseException | None, asyncio.CancelledError | None]:
        """Retry durable side effects without letting an unavailable sink hang cleanup."""

        if not events:
            return None, cancellation

        async def acknowledge_and_fan_out() -> list[Event]:
            await self._run_limit_controller.acknowledge_budget_settlement_events(events)
            return await self._event_writer.fan_out_persisted(events)

        fan_out_task = asyncio.create_task(acknowledge_and_fan_out())
        outcome = await self._await_session_operation_store_task(
            fan_out_task,
            cancellation=cancellation,
        )
        if outcome.timed_out:
            return (
                TimeoutError(
                    "Session operation event side-effect delivery exceeded "
                    f"{_SESSION_OPERATION_STORE_WAIT_TIMEOUT_SECONDS:g} seconds."
                ),
                outcome.cancellation,
            )
        error = outcome.error
        if isinstance(error, asyncio.CancelledError):
            error = unexpected_child_cancellation_error(
                error,
                operation="Session operation event side-effect delivery",
            )
        return error, outcome.cancellation

    async def run(
        self,
        request: CompactSessionRequest,
    ) -> AsyncGenerator[Event, None]:
        """Compact model-facing context without appending a conversation turn."""

        request = copy_compact_session_request(request)
        loaded_session = await self.session_store.load(request.session_id)
        if loaded_session is None:
            raise KeyError(f"Session not found: {request.session_id}")
        for field_name in (
            "agent_name",
            "provider_name",
            "model",
            "environment_name",
        ):
            session_request_boundary.require_secret_free_session_authority(
                getattr(loaded_session, field_name),
                field_name=field_name,
                redactor=self._secret_redactor,
            )
        request_digest = _compact_session_request_digest(request)
        checkpoint_before_claim = await self.session_store.load_checkpoint(loaded_session.id)
        checkpoint_before_claim = (
            await _prompt_transition._reconcile_committed_prompt_transition_intents(
                session_store=self.session_store,
                source_session_id=loaded_session.id,
                checkpoint=checkpoint_before_claim,
                clock=self._clock,
                persist=False,
            )
        )

        async def reconcile_after_admission() -> dict[str, Any] | None:
            current_checkpoint = await self.session_store.load_checkpoint(loaded_session.id)
            return await _prompt_transition._reconcile_committed_prompt_transition_intents(
                session_store=self.session_store,
                source_session_id=loaded_session.id,
                checkpoint=current_checkpoint,
                clock=self._clock,
            )

        persisted_before_claim = await self.session_store.load_session_operation(
            loaded_session.id,
            request.idempotency_key,
        )
        if type(persisted_before_claim) is dict:
            if persisted_before_claim.get("request_digest") != request_digest:
                raise ValueError(
                    "Session compaction idempotency key was already used for a different request."
                )
            replay_event_ids = _session_compaction_replay_event_ids(persisted_before_claim)
            if replay_event_ids is not None:
                (
                    requires_completion_decision,
                    admission_failure,
                ) = await verifier_aware_task_execution_outcome(
                    task_store=self._task_store,
                    redactor=self._secret_redactor,
                    task_id=None,
                    session_id=loaded_session.id,
                )
                if admission_failure is not None:
                    del loaded_session, request
                    raise_task_store_operation_failure(admission_failure)
                if requires_completion_decision:
                    del loaded_session, request
                    raise TaskCompletionDecisionRequired(
                        "Contracted tasks require the verifier-aware execution entrance."
                    ) from None
                await reconcile_after_admission()
                for event in await self._load_session_compaction_replay_events(
                    session_id=loaded_session.id,
                    event_ids=replay_event_ids,
                ):
                    yield event
                return
        if (
            checkpoint_before_claim is not None
            and _session_operation_state._SESSION_OPERATIONS_CHECKPOINT_KEY
            in checkpoint_before_claim
        ):
            existing_before_claim = _session_operation_state._session_operation_state(
                checkpoint_before_claim
            )["records"].get(request.idempotency_key)
            if type(existing_before_claim) is dict:
                if existing_before_claim.get("request_digest") != request_digest:
                    raise ValueError(
                        "Session compaction idempotency key was already used for a "
                        "different request."
                    )
                replay_event_ids = _session_compaction_replay_event_ids(existing_before_claim)
                if replay_event_ids is not None:
                    (
                        requires_completion_decision,
                        admission_failure,
                    ) = await verifier_aware_task_execution_outcome(
                        task_store=self._task_store,
                        redactor=self._secret_redactor,
                        task_id=None,
                        session_id=loaded_session.id,
                    )
                    if admission_failure is not None:
                        del loaded_session, request
                        raise_task_store_operation_failure(admission_failure)
                    if requires_completion_decision:
                        del loaded_session, request
                        raise TaskCompletionDecisionRequired(
                            "Contracted tasks require the verifier-aware execution entrance."
                        ) from None
                    await reconcile_after_admission()
                    for event in await self._load_session_compaction_replay_events(
                        session_id=loaded_session.id,
                        event_ids=replay_event_ids,
                    ):
                        yield event
                    return
        _session_checkpoint_admission._reject_unresumable_session_checkpoint(
            loaded_session,
            checkpoint_before_claim,
            redactor=self._secret_redactor,
            allow_active_operation=True,
        )
        if loaded_session.status not in _session_checkpoint_admission._RESUMABLE_SESSION_STATUSES:
            raise ValueError(
                f"Session compaction requires a resumable session boundary: {loaded_session.status}"
            )
        if loaded_session.run_epoch != request.expected_run_epoch:
            raise ValueError(
                "Session compaction source run epoch is stale: expected "
                f"{request.expected_run_epoch}, current {loaded_session.run_epoch}."
            )

        registered_agent = self._get_registered_agent(loaded_session.agent_name)
        context_policy = registered_agent.context_policy
        from cayu.memory.context import AutomaticRecallContextPolicy

        if type(context_policy) is AutomaticRecallContextPolicy:
            context_policy = context_policy.base_policy
        if not isinstance(context_policy, CheckpointCompactionContextPolicy):
            raise ValueError(
                "Explicit session compaction requires a configured "
                "CheckpointCompactionContextPolicy."
            )
        compactor_name = _require_application_compaction_event_text(
            type(context_policy.compactor).__name__,
            "compactor",
        )
        registered_environment = self._get_registered_environment_for_session(
            loaded_session.environment_name
        )
        environment_name = _environment_name(registered_environment)
        transcript_snapshot = await self.session_store.load_transcript_snapshot(loaded_session.id)
        try:
            source_transcript_cursor = transcript_snapshot.cursor
            if source_transcript_cursor != request.expected_transcript_cursor:
                raise ValueError(
                    "Session compaction source transcript cursor is stale: expected "
                    f"{request.expected_transcript_cursor}, current {source_transcript_cursor}."
                )
            projection_cursor = session_model_projection_cursor(loaded_session)
            if projection_cursor:
                transcript = list(
                    _transcript._project_model_target_snapshot(
                        transcript_snapshot,
                        projection_cursor,
                    ).messages
                )
            else:
                transcript = _transcript._transcript_snapshot_messages(transcript_snapshot)
        finally:
            del transcript_snapshot
        transcript, transcript_is_valid = session_request_boundary.redact_transcript(
            transcript,
            redactor=self._secret_redactor,
            field_name="session.transcript",
            reject_secret_bearing_runtime_projection_authority=True,
        )
        if not transcript_is_valid:
            raise ValueError(
                "Session transcript contains a workload secret in execution authority "
                "and cannot be compacted."
            ) from None
        (
            requires_completion_decision,
            admission_failure,
        ) = await verifier_aware_task_execution_outcome(
            task_store=self._task_store,
            redactor=self._secret_redactor,
            task_id=None,
            session_id=loaded_session.id,
        )
        if admission_failure is not None:
            del loaded_session, request, transcript
            raise_task_store_operation_failure(admission_failure)
        if requires_completion_decision:
            del loaded_session, request, transcript
            raise TaskCompletionDecisionRequired(
                "Contracted tasks require the verifier-aware execution entrance."
            ) from None
        checkpoint_before_claim = await reconcile_after_admission()
        compaction_budget_policy = copy_budget_policy(self._get_budget_policy())
        candidate_app_policy_budget_limits = budget_limits_for_session(
            policy=compaction_budget_policy,
            agent_name=registered_agent.spec.name,
            causal_budget_id=loaded_session.causal_budget_id,
        )
        candidate_request_budget_limits = request_budget_limits_for_session(
            limits=request.budget_limits,
            agent_name=registered_agent.spec.name,
            causal_budget_id=loaded_session.causal_budget_id,
        )
        compactor_provider_name: str | None = None
        compactor_model: str | None = None
        uses_forced_dispatch_boundary = (
            context_policy.compactor._uses_runtime_provider_dispatch_runner_for_forced_compaction()
        )
        has_compaction_accounting_limits = bool(
            candidate_app_policy_budget_limits
            or candidate_request_budget_limits
            or has_run_limits(request.limits)
        )
        if (
            uses_forced_dispatch_boundary
            or has_compaction_accounting_limits
            or self._request_footprint.enabled
        ):
            try:
                provider_budget_identity = context_policy.compactor.provider_budget_identity(
                    loaded_session
                )
            except NotImplementedError as exc:
                if self._request_footprint.enabled:
                    raise RuntimeError(
                        "Explicit compaction with request footprints requires the "
                        "ContextCompactor to explicitly declare "
                        "provider_budget_identity(session), returning provider/model or None "
                        "for deterministic execution."
                    ) from exc
                if not has_compaction_accounting_limits:
                    provider_budget_identity = None
                else:
                    raise RuntimeError(
                        "Explicit provider-backed compaction under run or cost limits requires "
                        "the ContextCompactor to declare provider_budget_identity(session), "
                        "returning provider/model or None for deterministic execution."
                    ) from exc
            if provider_budget_identity is None and uses_forced_dispatch_boundary:
                raise RuntimeError(
                    "Provider-backed compaction cannot declare a deterministic budget identity."
                )
            if provider_budget_identity is not None:
                if (
                    type(provider_budget_identity) is not tuple
                    or len(provider_budget_identity) != 2
                ):
                    raise TypeError(
                        "ContextCompactor.provider_budget_identity must return a "
                        "(provider_name, model) tuple or None."
                    )
                compactor_provider_name = _require_application_compaction_event_text(
                    provider_budget_identity[0],
                    "compactor_provider_name",
                )
                compactor_model = _require_application_compaction_event_text(
                    provider_budget_identity[1],
                    "compactor_model",
                )
                if self._request_footprint.enabled and not uses_forced_dispatch_boundary:
                    raise RuntimeError(
                        "Explicit provider-backed compaction with request footprints cannot "
                        f"safely run opaque provider-backed compactor "
                        f"{type(context_policy.compactor).__name__}: Cayu cannot observe each "
                        "provider dispatch independently. Use an unmodified built-in provider "
                        "compactor or disable request footprints."
                    )
        if compactor_provider_name is None:
            app_policy_budget_limits: tuple[BudgetLimit, ...] = ()
            budget_limits: tuple[BudgetLimit, ...] = ()
        else:
            app_policy_budget_limits = candidate_app_policy_budget_limits
            budget_limits = (*app_policy_budget_limits, *candidate_request_budget_limits)
        dispatch_settlement_budget_limits = tuple(
            limit
            for limit in budget_limits
            if (
                limit.reservation is not None
                or has_deferred_contextual_price(
                    limit.pricing,
                    provider_name=compactor_provider_name,
                    model=compactor_model,
                )
            )
        )
        dispatch_settlement_budget_limit_ids = {
            _effective_budget_limit_id(limit) for limit in dispatch_settlement_budget_limits
        }
        outer_budget_limits = tuple(
            limit
            for limit in budget_limits
            if _effective_budget_limit_id(limit) not in dispatch_settlement_budget_limit_ids
        )
        outer_app_policy_budget_limits = tuple(
            limit
            for limit in app_policy_budget_limits
            if _effective_budget_limit_id(limit) not in dispatch_settlement_budget_limit_ids
        )
        needs_dispatch_admission = compactor_provider_name is not None and (
            has_run_limits(request.limits) or bool(budget_limits)
        )
        if needs_dispatch_admission and not uses_forced_dispatch_boundary:
            raise RuntimeError(
                "Explicit provider-backed compaction under run or cost limits requires "
                "an unmodified built-in provider compactor so Cayu can admit every "
                "provider dispatch before execution."
            )
        expected_profile = execution_profile_from_session_metadata(loaded_session.metadata)
        try:
            registered_provider = self._get_registered_provider(loaded_session.provider_name)
        except KeyError:
            registered_provider = None

        def current_compaction_profile_candidate() -> ExecutionProfileIdentity:
            candidate = _session_execution_profile._execution_profile_identity(
                registered_agent=registered_agent,
                provider_name=loaded_session.provider_name,
                registered_provider=registered_provider,
                model=loaded_session.model,
                durable_system_prompt=None,
                redactor=self._secret_redactor,
                registered_environment=registered_environment,
                process_identity=self._execution_profile_process_identity,
                runtime_hooks=self._runtime_hooks,
                loop_policies=self._loop_policies,
                loop_policy_execution_profile_identities=(
                    self._loop_policy_execution_profile_identities
                ),
                budget_policy=compaction_budget_policy,
                request_budget_limits=request.budget_limits,
                causal_budget_id=loaded_session.causal_budget_id,
                limits=request.limits,
                finalization_material={
                    "kind": "cayu:explicit-context-compaction:v1",
                    "reason": request.reason,
                    "instruction_present": request.instructions is not None,
                    "instruction_digest": _optional_text_digest(request.instructions),
                    "limits": request.limits.model_dump(mode="json"),
                    "provider_name": compactor_provider_name,
                    "model": compactor_model,
                },
                tool_capability_ceiling=_session_execution_profile._session_tool_capability_ceiling(
                    loaded_session,
                ),
            )
            return execution_profile_with_component(
                candidate,
                expected_profile.component(
                    ExecutionProfileComponentClass.DURABLE_SYSTEM_PROJECTION
                ),
            )

        profile_candidate = current_compaction_profile_candidate()
        governed_registration_components = (
            ExecutionProfileComponentClass.CONTEXT_SELECTION,
            ExecutionProfileComponentClass.AUTOMATIC_RECALL,
            ExecutionProfileComponentClass.CONTEXT_COMPACTION,
            ExecutionProfileComponentClass.EXECUTION_ENVIRONMENT,
        )
        changed_registration_components = (
            ()
            if expected_profile.schema_version < 3
            else tuple(
                component
                for component in governed_registration_components
                if expected_profile.component(component) != profile_candidate.component(component)
            )
        )
        if changed_registration_components:
            raise ExecutionProfileMismatchError(
                session_id=loaded_session.id,
                expected_profile_fingerprint=expected_profile.fingerprint,
                candidate_profile_fingerprint=profile_candidate.fingerprint,
                changed_component_classes=changed_registration_components,
                expected_profile=expected_profile,
                candidate_profile=profile_candidate,
            )
        if expected_profile.schema_version < 3:
            compaction_execution_profile = profile_candidate
        else:
            compaction_execution_profile = expected_profile
            for component_class in (
                *governed_registration_components,
                ExecutionProfileComponentClass.APPLICATION_BUDGET_POLICY,
                ExecutionProfileComponentClass.INVOCATION_BUDGET_POLICY,
                ExecutionProfileComponentClass.FINALIZATION,
            ):
                compaction_execution_profile = execution_profile_with_component(
                    compaction_execution_profile,
                    profile_candidate.component(component_class),
                )

        def validate_live_compaction_semantics() -> None:
            candidate = current_compaction_profile_candidate()
            changed = tuple(
                component
                for component in governed_registration_components
                if candidate.component(component)
                != compaction_execution_profile.component(component)
            )
            if changed:
                raise ExecutionProfileMismatchError(
                    session_id=loaded_session.id,
                    expected_profile_fingerprint=compaction_execution_profile.fingerprint,
                    candidate_profile_fingerprint=candidate.fingerprint,
                    changed_component_classes=changed,
                    expected_profile=compaction_execution_profile,
                    candidate_profile=candidate,
                )

        def governed_compaction_events(events: Iterable[Event]) -> list[Event]:
            return [
                event_with_execution_profile_authority(event, compaction_execution_profile)
                for event in events
            ]

        operation_started_at = time.monotonic()

        operation_id = str(uuid4())
        attempt_id = str(uuid4())
        existing_before_claim: dict[str, Any] | None = None
        if (
            type(persisted_before_claim) is dict
            and persisted_before_claim.get("request_digest") == request_digest
            and persisted_before_claim.get("status") == "abandoned"
        ):
            persisted_operation_id = persisted_before_claim.get("operation_id")
            if type(persisted_operation_id) is not str:
                raise ValueError("Persisted session operation is missing its operation id.")
            operation_id = require_clean_nonblank(
                persisted_operation_id,
                "operation_id",
            )
        if (
            checkpoint_before_claim is not None
            and _session_operation_state._SESSION_OPERATIONS_CHECKPOINT_KEY
            in checkpoint_before_claim
        ):
            existing_candidate = _session_operation_state._session_operation_state(
                checkpoint_before_claim
            )["records"].get(request.idempotency_key)
            existing_before_claim = existing_candidate if type(existing_candidate) is dict else None
            if (
                type(existing_before_claim) is dict
                and existing_before_claim.get("request_digest") == request_digest
                and existing_before_claim.get("status") == "running"
            ):
                operation_id = require_clean_nonblank(
                    existing_before_claim.get("operation_id"),
                    "operation_id",
                )
        stored_model_step_identity: ModelStepIdentity | None = None
        for existing_record in (persisted_before_claim, existing_before_claim):
            if (
                type(existing_record) is not dict
                or existing_record.get("request_digest") != request_digest
                or existing_record.get("status") not in {"running", "abandoned"}
            ):
                continue
            raw_model_step_id = existing_record.get("model_step_id")
            if type(raw_model_step_id) is not str:
                raise RuntimeError("Existing session compaction lacks logical model-step identity.")
            candidate_identity = ModelStepIdentity(model_step_id=raw_model_step_id)
            if (
                stored_model_step_identity is not None
                and stored_model_step_identity != candidate_identity
            ):
                raise RuntimeError(
                    "Session compaction records disagree on logical model-step identity."
                )
            stored_model_step_identity = candidate_identity
            raw_profile = existing_record.get("execution_profile")
            if type(raw_profile) is not dict:
                raise RuntimeError(
                    "Existing session compaction lacks its governing execution profile."
                )
            stored_profile = ExecutionProfileIdentity.model_validate(raw_profile)
            if stored_profile != compaction_execution_profile:
                raise ExecutionProfileMismatchError(
                    session_id=loaded_session.id,
                    expected_profile_fingerprint=stored_profile.fingerprint,
                    candidate_profile_fingerprint=compaction_execution_profile.fingerprint,
                    changed_component_classes=changed_execution_profile_components(
                        stored_profile,
                        compaction_execution_profile,
                    ),
                    expected_profile=stored_profile,
                    candidate_profile=compaction_execution_profile,
                )
        model_step_identity = stored_model_step_identity or new_model_step_identity()
        reservation_identity_guard = self._run_limit_controller.reservation_identity_guard()
        started_event = self._event_writer.prepare(
            event_with_execution_profile_authority(
                Event(
                    type=EventType.CONTEXT_COMPACTION_STARTED,
                    session_id=loaded_session.id,
                    agent_name=registered_agent.spec.name,
                    environment_name=environment_name,
                    payload={
                        **_application_compaction_causal_payload(
                            request=request,
                            operation_id=operation_id,
                            attempt_id=attempt_id,
                            source_cursor=source_transcript_cursor,
                            compactor=compactor_name,
                        ),
                        **model_step_identity.payload(),
                    },
                ),
                compaction_execution_profile,
            )
        )
        claimed_checkpoint: dict[str, Any] | None = None
        claimed_operation_expires_at: datetime | None = None

        def claim_operation(
            current_session: Session,
            checkpoint: dict[str, Any] | None,
            persisted_record: dict[str, Any] | None,
            claim_now: datetime,
        ) -> SessionOperationPublication:
            nonlocal operation_id, claimed_checkpoint, claimed_operation_expires_at
            claim_expires_at = claim_now + _SESSION_OPERATION_CLAIM_LEASE
            if current_session.run_epoch != request.expected_run_epoch:
                raise ValueError(
                    "Session compaction source run epoch is stale: expected "
                    f"{request.expected_run_epoch}, current {current_session.run_epoch}."
                )
            recovery_claim = _incomplete_recovery_claim_from_checkpoint(checkpoint)
            if recovery_claim is not None and recovery_claim[1] <= claim_now:
                raise _session_checkpoint_admission._ExpiredIncompleteRecoveryClaim(
                    recovery_claim[0]
                )
            checkpoint = _checkpoint_without_active_incomplete_recovery_claim(
                checkpoint,
                now=claim_now,
            )
            _session_checkpoint_admission._reject_unresumable_session_checkpoint(
                current_session,
                checkpoint,
                redactor=self._secret_redactor,
                allow_active_operation=True,
            )
            updated = {} if checkpoint is None else copy_durable_record(checkpoint, "checkpoint")
            operations = _session_operation_state._session_operation_state(updated)
            _session_operation_state._abandon_expired_session_operation(operations, now=claim_now)
            records = operations["records"]
            existing = records.get(request.idempotency_key)
            if existing is None and persisted_record is not None:
                existing = copy_json_value(persisted_record, "session_operation")
                if existing.get("status") == "abandoned":
                    records[request.idempotency_key] = existing
            if existing is not None:
                if existing.get("request_digest") != request_digest:
                    raise ValueError(
                        "Session compaction idempotency key was already used for a "
                        "different request."
                    )
                status = existing.get("status")
                if status == "running":
                    raise RuntimeError(
                        "Equivalent session compaction operation is already running: "
                        f"{existing.get('operation_id')}"
                    )
                if status == "abandoned":
                    operation_id = require_clean_nonblank(
                        existing.get("operation_id"),
                        "operation_id",
                    )
                    if started_event.payload.get("operation_id") != operation_id:
                        raise RuntimeError(
                            "Abandoned session compaction changed during claim; retry it."
                        )
                    existing_model_step_identity = ModelStepIdentity(
                        model_step_id=existing.get("model_step_id")
                    )
                    if existing_model_step_identity != model_step_identity:
                        raise RuntimeError(
                            "Abandoned session compaction changed logical model-step identity."
                        )
                    if operations.get("active_operation_id") is not None:
                        raise RuntimeError(
                            "Session already has an active durable operation: "
                            f"{operations.get('active_operation_id')}"
                        )
                    operations["active_operation_id"] = operation_id
                    existing["status"] = "running"
                    existing["attempt_count"] = existing.get("attempt_count", 1) + 1
                    existing["current_attempt_id"] = attempt_id
                    existing["event_ids"] = [
                        *existing.get("event_ids", []),
                        started_event.id,
                    ]
                    existing["claim_expires_at"] = claim_expires_at.isoformat()
                    existing["updated_at"] = claim_now.isoformat()
                    existing.pop("abandoned_at", None)
                    updated[_session_operation_state._SESSION_OPERATIONS_CHECKPOINT_KEY] = (
                        operations
                    )
                    claimed_checkpoint = copy_durable_record(updated, "checkpoint")
                    claimed_operation_expires_at = claim_expires_at
                    archived_records = (
                        _session_operation_state._archive_inactive_session_operation_records(
                            updated,
                            except_idempotency_key=request.idempotency_key,
                        )
                    )
                    return SessionOperationPublication(
                        checkpoint=updated,
                        operation_records=archived_records,
                    )
                operation_id = require_clean_nonblank(
                    existing.get("operation_id"),
                    "operation_id",
                )
                stored_event_ids = _session_compaction_replay_event_ids(existing)
                if stored_event_ids is None:
                    raise RuntimeError("Running session compaction changed during claim.")
                raise _SessionCompactionReplay(stored_event_ids)
            active_operation_id = operations.get("active_operation_id")
            if active_operation_id is not None:
                raise RuntimeError(
                    f"Session already has an active durable operation: {active_operation_id}"
                )
            operations["active_operation_id"] = operation_id
            records[request.idempotency_key] = {
                "operation_id": operation_id,
                "model_step_id": model_step_identity.model_step_id,
                "kind": _CONTEXT_COMPACTION_OPERATION_KIND,
                "reason": request.reason,
                "request_digest": request_digest,
                "status": "running",
                "source_run_epoch": request.expected_run_epoch,
                "source_transcript_cursor": request.expected_transcript_cursor,
                "attempt_count": 1,
                "current_attempt_id": attempt_id,
                "event_ids": [started_event.id],
                "instruction_present": request.instructions is not None,
                "instruction_digest": _optional_text_digest(request.instructions),
                "execution_profile": compaction_execution_profile.model_dump(mode="json"),
                "claim_expires_at": claim_expires_at.isoformat(),
                "created_at": claim_now.isoformat(),
                "updated_at": claim_now.isoformat(),
            }
            updated[_session_operation_state._SESSION_OPERATIONS_CHECKPOINT_KEY] = operations
            claimed_checkpoint = copy_durable_record(updated, "checkpoint")
            claimed_operation_expires_at = claim_expires_at
            archived_records = _session_operation_state._archive_inactive_session_operation_records(
                updated,
                except_idempotency_key=request.idempotency_key,
            )
            return SessionOperationPublication(
                checkpoint=updated,
                operation_records=archived_records,
            )

        started_event = attribute_event_to_current_interaction(started_event)
        replay_event_ids: tuple[str, ...] | None = None

        def require_unexpired_initial_claim_commit(commit_at: datetime) -> None:
            if claimed_operation_expires_at is None:
                raise AssertionError(
                    "Session compaction initial publication did not capture its claim."
                )
            if claimed_operation_expires_at <= commit_at:
                raise SessionCompactionAttemptSuperseded(
                    "Session compaction initial publication claim expired before commit."
                )

        async def publish_initial_claim() -> None:
            await self.session_store.publish_session_operation_guarded_with_store_time(
                loaded_session.id,
                idempotency_key=request.idempotency_key,
                operation_transform=claim_operation,
                commit_guard=lambda: None,
                commit_time_guard=require_unexpired_initial_claim_commit,
                events=[started_event],
                expected_statuses=_session_checkpoint_admission._RESUMABLE_SESSION_STATUSES,
                expected_run_epoch=request.expected_run_epoch,
                expected_transcript_cursor=request.expected_transcript_cursor,
            )

        initial_claim_started_monotonic = time.monotonic()
        initial_claim_reconciled = False
        try:
            initial_task = asyncio.create_task(publish_initial_claim())
            initial_outcome = await self._await_session_operation_store_task(initial_task)
            initial_error = initial_outcome.error
            if initial_outcome.timed_out:
                initial_error = TimeoutError(
                    "Session compaction initial claim exceeded its bounded store wait."
                )
            if initial_outcome.cancellation is not None:
                raise initial_outcome.cancellation from initial_error
            if isinstance(initial_error, asyncio.CancelledError):
                initial_error = unexpected_child_cancellation_error(
                    initial_error,
                    operation="Session compaction initial claim publication",
                )
            if initial_error is not None:
                raise initial_error
        except _session_checkpoint_admission._ExpiredIncompleteRecoveryClaim as expired_claim:
            current = await self._require_session(loaded_session.id)
            fenced = await self._recovery_ownership.fence_expired_incomplete_recovery_claim(
                session=current,
                claim_id=expired_claim.claim_id,
            )
            if not fenced:
                raise RuntimeError(
                    "Expired incomplete-session recovery ownership changed while compaction "
                    "was fencing it; retry with current session state."
                ) from None
            current = await self._require_session(loaded_session.id)
            raise ValueError(
                "Session compaction fenced an expired incomplete-session recovery owner; "
                f"retry with run epoch {current.run_epoch}."
            ) from None
        except _SessionCompactionReplay as replay:
            replay_event_ids = replay.event_ids
        except BaseException as publication_failure:
            reconciliation = asyncio.create_task(
                self._reconcile_initial_compaction_claim(
                    session=loaded_session,
                    request=request,
                    operation_id=operation_id,
                    attempt_id=attempt_id,
                    model_step_id=model_step_identity.model_step_id,
                    request_digest=request_digest,
                    started_event=started_event,
                )
            )
            outcome = await self._await_session_operation_store_task(
                reconciliation,
                cancellation=(
                    publication_failure
                    if isinstance(publication_failure, asyncio.CancelledError)
                    else None
                ),
            )
            if outcome.cancellation is not None:
                raise outcome.cancellation from (
                    publication_failure if outcome.cancellation is not publication_failure else None
                )
            if outcome.timed_out or outcome.error is not None or outcome.result is None:
                publication_failure.add_note(
                    "Initial compaction claim acknowledgement could not be reconciled; "
                    "ownership remains unproven and no provider work was started."
                )
                raise publication_failure
            if not isinstance(publication_failure, Exception):
                raise publication_failure
            claimed_checkpoint, claimed_operation_expires_at = outcome.result
            initial_claim_reconciled = True
        if replay_event_ids is not None:
            for event in await self._load_session_compaction_replay_events(
                session_id=loaded_session.id,
                event_ids=replay_event_ids,
            ):
                yield event
            return
        if claimed_checkpoint is None:
            raise AssertionError("New session compaction did not persist its operation claim.")
        if claimed_operation_expires_at is None:
            raise AssertionError("New session compaction did not persist its claim expiry.")
        if not initial_claim_reconciled:
            claimed_operation_expires_at = await self._reconcile_compaction_operation_claim_expiry(
                session_id=loaded_session.id,
                idempotency_key=request.idempotency_key,
                operation_id=operation_id,
                attempt_id=attempt_id,
            )
        compaction_invocation_checkpoint = project_compaction_invocation_checkpoint(
            claimed_checkpoint,
            redactor=self._secret_redactor,
        )
        checkpoint_before_claim = None
        claimed_checkpoint = None

        operation_published = False
        operation_failure: BaseException | None = None
        stop_operation_heartbeat = asyncio.Event()
        operation_heartbeat_state = _SessionOperationClaimHeartbeatState()
        operation_heartbeat_state.confirm_claim(
            claimed_operation_expires_at,
            claim_deadline_monotonic=(
                initial_claim_started_monotonic + _SESSION_OPERATION_CLAIM_LEASE.total_seconds()
            ),
        )
        if operation_heartbeat_state.remaining_claim_seconds() <= 0:
            raise SessionCompactionAttemptSuperseded(
                "Session compaction operation claim acknowledgement exceeded its lease."
            )
        operation_heartbeat_task = asyncio.create_task(
            self._heartbeat_compaction_operation_claim(
                session=loaded_session,
                request=request,
                operation_id=operation_id,
                attempt_id=attempt_id,
                claim_expires_at=claimed_operation_expires_at,
                stop=stop_operation_heartbeat,
                state=operation_heartbeat_state,
            )
        )

        async def stop_claim_heartbeat() -> asyncio.Task[Any] | None:
            stop_operation_heartbeat.set()
            if not operation_heartbeat_task.done():
                operation_heartbeat_task.cancel()
            outcome = await await_shielded_task_outcome(
                operation_heartbeat_task,
                timeout_s=_SESSION_OPERATION_CLAIM_HEARTBEAT_STOP_TIMEOUT_SECONDS,
            )
            pending_renewal = operation_heartbeat_state.pending_store_task
            if outcome.timed_out:
                if pending_renewal is not None and not pending_renewal.done():
                    pending_outcome = await await_shielded_task_outcome(
                        pending_renewal,
                        timeout_s=_SESSION_OPERATION_CLAIM_HEARTBEAT_STOP_TIMEOUT_SECONDS,
                    )
                    if pending_outcome.timed_out:
                        return pending_renewal
                    if pending_outcome.cancellation is not None:
                        raise pending_outcome.cancellation
                outcome = await await_shielded_task_outcome(
                    operation_heartbeat_task,
                    timeout_s=_SESSION_OPERATION_CLAIM_HEARTBEAT_STOP_TIMEOUT_SECONDS,
                )
                if outcome.timed_out:
                    return operation_heartbeat_task
            if pending_renewal is not None and not pending_renewal.done():
                pending_outcome = await await_shielded_task_outcome(
                    pending_renewal,
                    timeout_s=_SESSION_OPERATION_CLAIM_HEARTBEAT_STOP_TIMEOUT_SECONDS,
                )
                if pending_outcome.timed_out:
                    return pending_renewal
                if pending_outcome.cancellation is not None:
                    raise pending_outcome.cancellation
            if outcome.cancellation is not None:
                raise outcome.cancellation
            heartbeat_failure = outcome.error
            if heartbeat_failure is None:
                return None
            if isinstance(heartbeat_failure, asyncio.CancelledError):
                concurrent_failure = exception_cause(heartbeat_failure)
                if concurrent_failure is None:
                    return None
                heartbeat_failure = concurrent_failure
            if operation_published or heartbeat_failure is operation_failure:
                return None
            raise heartbeat_failure

        async def stop_claim_heartbeat_and_track() -> None:
            blocker = await stop_claim_heartbeat()
            if blocker is not None:
                # Work of the stopped heartbeat, which nothing still depends on.
                self._track_detached_session_operation_task(blocker, report_failure=False)

        initial_event_delivered = False
        initial_delivery_failure: BaseException | None = None
        try:
            await self._event_writer.fan_out_persisted([started_event])
            yield started_event
            initial_event_delivered = True
        except BaseException as exc:
            initial_delivery_failure = exc
            raise
        finally:
            if not initial_event_delivered:
                await self._recovery_ownership.run_cleanup_steps(
                    authoritative_failure=initial_delivery_failure,
                    steps=(
                        (
                            "explicit compaction claim heartbeat stop",
                            stop_claim_heartbeat_and_track,
                        ),
                    ),
                )

        attempt_events: list[Event] = []
        prepublished_dispatch_events: list[Event] = []
        deferred_dispatch_settlement_events: list[Event] = []
        observed_dispatch_completion_events: list[Event] = []
        materialized_compaction_attempt_ids: set[str] = set()
        persisted_attempt_event_ids: set[str] = set()
        yielded_attempt_event_ids: set[str] = set()
        attempt_event_inventory: dict[str, Event] = {}
        completion_event_attempt_ids: dict[str, str] = {}
        unresolved_attempt_publication_tasks: set[asyncio.Task[Any]] = set()
        unresolved_completion_publication_tasks: set[asyncio.Task[Any]] = set()
        reached_budget_keys: set[str] = set()
        budget_reservations: list[BudgetStepReservation] = []
        budget_reservations_settled = False
        outer_model_attempt_identity = model_step_identity.new_attempt()
        try:
            limit_decision = await self._run_limit_controller.evaluate_operation_run_limit(
                session=loaded_session,
                limits=request.limits,
                operation_events=attempt_events,
                operation_started_at=operation_started_at,
            )
            if limit_decision is not None:
                raise RuntimeError(f"Compaction limit reached: {limit_decision.message}")
            budget_error = await self._enforce_compaction_budget_limits(
                session=loaded_session,
                budget_limits=outer_budget_limits,
                app_policy_budget_limits=outer_app_policy_budget_limits,
                attempt_events=attempt_events,
                reached_budget_keys=reached_budget_keys,
                request=request,
                operation_id=operation_id,
                attempt_id=attempt_id,
                registered_agent=registered_agent,
                environment_name=environment_name,
                compactor=compactor_name,
                provider_name=compactor_provider_name,
                model=compactor_model,
                execution_identity=model_step_identity,
            )
            if attempt_events:
                persisted_attempt_events = governed_compaction_events(attempt_events)
                await self._persist_compaction_attempt_events(
                    session=loaded_session,
                    request=request,
                    operation_id=operation_id,
                    attempt_id=attempt_id,
                    events=persisted_attempt_events,
                    heartbeat_state=operation_heartbeat_state,
                    persisted_event_ids=persisted_attempt_event_ids,
                    event_inventory=attempt_event_inventory,
                    unresolved_store_tasks=unresolved_attempt_publication_tasks,
                )
                attempt_events.clear()
                await self._run_limit_controller.acknowledge_budget_settlement_events(
                    persisted_attempt_events
                )
                await self._event_writer.fan_out_persisted(persisted_attempt_events)
                for event in persisted_attempt_events:
                    yielded_attempt_event_ids.add(event.id)
                    yield event
            if budget_error is not None:
                raise budget_error

            reservation_failure = await self._reserve_compaction_budget(
                session=loaded_session,
                registered_agent=registered_agent,
                environment_name=environment_name,
                budget_limits=outer_budget_limits,
                provider_name=compactor_provider_name,
                model=compactor_model,
                model_attempt_identity=outer_model_attempt_identity,
                request=request,
                operation_id=operation_id,
                attempt_id=attempt_id,
                compactor=compactor_name,
                reservations=budget_reservations,
                events=attempt_events,
                reservation_identity_guard=reservation_identity_guard,
                execution_profile_fingerprint=compaction_execution_profile.fingerprint,
            )
            reservation_error: RuntimeError | None = None
            if reservation_failure is not None:
                budget_reservations_settled = True
                attempt_events.append(
                    _application_compaction_ledger_event(
                        event_type=EventType.BUDGET_LIMIT_REACHED,
                        payload=budget_reservation_payload(reservation_failure),
                        request=request,
                        operation_id=operation_id,
                        attempt_id=attempt_id,
                        session=loaded_session,
                        registered_agent=registered_agent,
                        environment_name=environment_name,
                        compactor=compactor_name,
                    )
                )
                reservation_error = RuntimeError(
                    f"Compaction budget reservation failed: {reservation_failure.message}"
                )
            if attempt_events:
                persisted_attempt_events = governed_compaction_events(attempt_events)
                await self._persist_compaction_attempt_events(
                    session=loaded_session,
                    request=request,
                    operation_id=operation_id,
                    attempt_id=attempt_id,
                    events=persisted_attempt_events,
                    heartbeat_state=operation_heartbeat_state,
                    persisted_event_ids=persisted_attempt_event_ids,
                    event_inventory=attempt_event_inventory,
                    unresolved_store_tasks=unresolved_attempt_publication_tasks,
                )
                attempt_events.clear()
                await self._run_limit_controller.acknowledge_budget_settlement_events(
                    persisted_attempt_events
                )
                await self._event_writer.fan_out_persisted(persisted_attempt_events)
                for event in persisted_attempt_events:
                    yielded_attempt_event_ids.add(event.id)
                    yield event
            if reservation_error is not None:
                raise reservation_error

            async def publish_dispatch_budget_events() -> None:
                if not attempt_events:
                    return
                events = governed_compaction_events(attempt_events)
                await self._persist_compaction_attempt_events(
                    session=loaded_session,
                    request=request,
                    operation_id=operation_id,
                    attempt_id=attempt_id,
                    events=events,
                    heartbeat_state=operation_heartbeat_state,
                    persisted_event_ids=persisted_attempt_event_ids,
                    event_inventory=attempt_event_inventory,
                    unresolved_store_tasks=unresolved_attempt_publication_tasks,
                )
                attempt_events.clear()
                await self._run_limit_controller.acknowledge_budget_settlement_events(events)
                await self._event_writer.fan_out_persisted(events)
                prepublished_dispatch_events.extend(events)

            compaction_identity_ledger = _CompactionExecutionIdentityLedger(model_step_identity)

            async def record_compaction_footprint(
                *,
                provider: ModelProvider,
                provider_name: str,
                model_request: ModelRequest,
                dispatch_attempt: int,
                max_dispatch_attempts: int,
                model_attempt_identity: ModelAttemptIdentity,
            ) -> None:
                detached_request = _detach_model_request(model_request)
                if self._request_footprint.enabled:
                    footprint = analyze_request_footprint(
                        detached_request,
                        provider=provider,
                        provider_name=provider_name,
                        step=None,
                        attempt=dispatch_attempt,
                        max_attempts=max_dispatch_attempts,
                        request_variant=RequestVariant.CONTEXT_COMPACTION,
                        observation_id=str(uuid4()),
                        model_step_id=model_attempt_identity.model_step_id,
                        model_attempt_id=model_attempt_identity.model_attempt_id,
                        config=self._request_footprint,
                        operation_id=operation_id,
                        operation_attempt_id=attempt_id,
                        execution_profile_fingerprint=(compaction_execution_profile.fingerprint),
                    )
                    attempt_events.append(
                        event_with_execution_profile_authority(
                            event_with_runtime_payload_authority(
                                Event(
                                    type=EventType.REQUEST_FOOTPRINT_RECORDED,
                                    session_id=loaded_session.id,
                                    agent_name=registered_agent.spec.name,
                                    environment_name=environment_name,
                                    payload=footprint.model_dump(mode="json", exclude_none=True),
                                ),
                                "observation_id",
                                "model_step_id",
                                "model_attempt_id",
                                "operation_id",
                                "attempt_id",
                            ),
                            compaction_execution_profile,
                        )
                    )
                attempt_events.append(
                    event_with_execution_profile_authority(
                        event_with_runtime_payload_authority(
                            _application_compaction_ledger_event(
                                event_type=EventType.MODEL_STARTED,
                                payload={
                                    "model": detached_request.model,
                                    "provider": provider_name,
                                    "attempt": dispatch_attempt,
                                    "max_attempts": max_dispatch_attempts,
                                    "purpose": ModelCompletionPurpose.CONTEXT_COMPACTION.value,
                                    **model_attempt_identity.payload(),
                                },
                                request=request,
                                operation_id=operation_id,
                                attempt_id=attempt_id,
                                session=loaded_session,
                                registered_agent=registered_agent,
                                environment_name=environment_name,
                                compactor=compactor_name,
                            ),
                            "model_step_id",
                            "model_attempt_id",
                            "operation_id",
                            "attempt_id",
                        ),
                        compaction_execution_profile,
                    )
                )
                await publish_dispatch_budget_events()

            def identify_compaction_completion_payload(
                payload: dict[str, Any],
                *,
                expected_identity: ModelAttemptIdentity | None = None,
            ) -> dict[str, Any]:
                copied_payload = copy_json_value(payload, "compaction_model_completed_payload")
                identified = compaction_identity_ledger.identify_payload(
                    copied_payload,
                    expected_identity=expected_identity,
                )
                identity = {key: identified[key] for key in ("model_step_id", "model_attempt_id")}
                strip_runtime_owned_execution_identity(identified)
                identified.update(identity)
                return identified

            def application_compaction_telemetry_event(
                telemetry: ContextCompactionTelemetry,
            ) -> Event:
                execution_identity: ModelStepIdentity | ModelAttemptIdentity = model_step_identity
                if telemetry.event_type == EventType.MODEL_COMPLETED:
                    identified_payload = identify_compaction_completion_payload(telemetry.payload)
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
                return event_with_execution_profile_authority(
                    _application_compaction_event(
                        telemetry=telemetry,
                        request=request,
                        operation_id=operation_id,
                        attempt_id=attempt_id,
                        session=loaded_session,
                        registered_agent=registered_agent,
                        environment_name=environment_name,
                        compactor=compactor_name,
                        execution_identity=execution_identity,
                    ),
                    compaction_execution_profile,
                )

            def dispatch_completion_event(
                *,
                actual_pricing_provider_name: str,
                actual_model: str,
                actual_usage_dialect: UsageDialect,
                completed_metadata: dict[str, Any],
                model_attempt_identity: ModelAttemptIdentity,
            ) -> Event:
                provider_name = actual_pricing_provider_name
                compactor = compactor_name
                try:
                    payload = _compaction_model_completed_payload(
                        completed_payload=completed_metadata,
                        provider_name=provider_name,
                        fallback_model=actual_model,
                        compactor=compactor,
                        usage_dialect=actual_usage_dialect,
                    )
                except _CompactionAccountingUsageError as exc:
                    # A provider call still completed when normalized counters
                    # overflowed the durable int64 contract. Preserve the safe
                    # rejected-usage evidence so its reservation is settled at
                    # the reserved amount instead of being left uncertain.
                    payload = exc.payload
                durable_payload = _durable_compaction_completion_evidence(
                    payload,
                    provider_name=provider_name,
                    fallback_model=actual_model,
                    compactor=compactor,
                )
                durable_payload.update(
                    copy_model_attempt_identity(model_attempt_identity).payload()
                )
                return event_with_execution_profile_authority(
                    Event(
                        type=EventType.MODEL_COMPLETED,
                        session_id=loaded_session.id,
                        agent_name=registered_agent.spec.name,
                        environment_name=environment_name,
                        payload=durable_payload,
                    ),
                    compaction_execution_profile,
                )

            async def publish_compaction_completions(
                payloads: list[dict[str, Any]],
            ) -> None:
                pending: list[tuple[str, Event]] = []
                for raw_payload in payloads:
                    payload = identify_compaction_completion_payload(raw_payload)
                    execution_identity = ModelAttemptIdentity.model_validate(
                        {
                            "model_step_id": payload.get("model_step_id"),
                            "model_attempt_id": payload.get("model_attempt_id"),
                        }
                    )
                    compaction_attempt_id = payload.get(_COMPACTION_ATTEMPT_ID_KEY)
                    if type(compaction_attempt_id) is not str:
                        raise RuntimeError(
                            "Compaction completion evidence lost its attempt identity."
                        )
                    if compaction_attempt_id in materialized_compaction_attempt_ids:
                        continue
                    pending.append(
                        (
                            compaction_attempt_id,
                            _application_compaction_event(
                                telemetry=ContextCompactionTelemetry(
                                    event_type=EventType.MODEL_COMPLETED,
                                    payload=payload,
                                ),
                                request=request,
                                operation_id=operation_id,
                                attempt_id=attempt_id,
                                session=loaded_session,
                                registered_agent=registered_agent,
                                environment_name=environment_name,
                                compactor=compactor_name,
                                execution_identity=execution_identity,
                            ),
                        )
                    )
                if not pending and not deferred_dispatch_settlement_events:
                    return

                completion_events = governed_compaction_events(
                    event for _attempt_id, event in pending
                )
                completion_event_attempt_ids.update(
                    (event.id, compaction_attempt_id) for compaction_attempt_id, event in pending
                )
                events = [
                    *completion_events,
                    *deferred_dispatch_settlement_events,
                ]

                def record_durable_completion_batch() -> None:
                    materialized_compaction_attempt_ids.update(
                        compaction_attempt_id for compaction_attempt_id, _event in pending
                    )
                    observed_dispatch_completion_events.extend(completion_events)
                    prepublished_dispatch_events.extend(events)
                    deferred_dispatch_settlement_events.clear()

                try:
                    await self._persist_compaction_attempt_events(
                        session=loaded_session,
                        request=request,
                        operation_id=operation_id,
                        attempt_id=attempt_id,
                        events=events,
                        heartbeat_state=operation_heartbeat_state,
                        persisted_event_ids=persisted_attempt_event_ids,
                        event_inventory=attempt_event_inventory,
                        unresolved_store_tasks=unresolved_attempt_publication_tasks,
                        cost_bearing_store_tasks=(unresolved_completion_publication_tasks),
                    )
                except BaseException:
                    # The helper records IDs only after an ambiguous write is
                    # confirmed atomic and durable. Preserve that exact attempt
                    # identity before propagating the lost acknowledgement.
                    if all(event.id in persisted_attempt_event_ids for event in events):
                        record_durable_completion_batch()
                    raise
                record_durable_completion_batch()
                (
                    fan_out_error,
                    cancellation,
                ) = await self._fan_out_reconciled_session_operation_events(events)
                if cancellation is not None:
                    if fan_out_error is not None:
                        cancellation.add_note(
                            "Explicit compaction completion side-effect delivery also "
                            f"failed during cancellation: {type(fan_out_error).__name__}: "
                            f"{fan_out_error}"
                        )
                        raise cancellation from fan_out_error
                    raise cancellation
                if fan_out_error is not None:
                    raise fan_out_error

            async def settle_provider_dispatch(
                dispatch_reservations: list[BudgetStepReservation],
                *,
                completed_event: Event | None,
                uncertain_reason: str,
                release_reason: str | None = None,
            ) -> None:
                dispatch_reservation_ids = {
                    reservation.record.reservation_id for reservation in dispatch_reservations
                }
                if release_reason is not None:
                    settlement = self._release_compaction_budget_reservations(
                        dispatch_reservations,
                        session=loaded_session,
                        registered_agent=registered_agent,
                        environment_name=environment_name,
                        request=request,
                        operation_id=operation_id,
                        attempt_id=attempt_id,
                        compactor=compactor_name,
                        reason=release_reason,
                    )
                elif completed_event is None:
                    settlement = self._reconcile_uncertain_compaction_budget_reservations(
                        dispatch_reservations,
                        session=loaded_session,
                        registered_agent=registered_agent,
                        environment_name=environment_name,
                        request=request,
                        operation_id=operation_id,
                        attempt_id=attempt_id,
                        compactor=compactor_name,
                    )
                else:
                    settlement = self._reconcile_compaction_budget_reservations(
                        dispatch_reservations,
                        model_completed_events=[completed_event],
                        session=loaded_session,
                        registered_agent=registered_agent,
                        environment_name=environment_name,
                        request=request,
                        operation_id=operation_id,
                        attempt_id=attempt_id,
                        compactor=compactor_name,
                    )
                settlement_events: list[Event] = []
                try:
                    async for event in settlement:
                        settlement_events.append(event)
                except Exception as exc:
                    exc.add_note(uncertain_reason)
                    raise
                finally:
                    # Provider completion telemetry is finalized by the context
                    # policy (including outcome annotations such as
                    # ``invalid_summary``). Keep reconciliation events in memory
                    # until that authoritative completion evidence can be
                    # published first.
                    deferred_dispatch_settlement_events.extend(settlement_events)
                    unsettled_ids = {
                        reservation.record.reservation_id for reservation in dispatch_reservations
                    }
                    settled_ids = dispatch_reservation_ids - unsettled_ids
                    budget_reservations[:] = [
                        reservation
                        for reservation in budget_reservations
                        if reservation.record.reservation_id not in settled_ids
                    ]

            async def _run_provider_dispatch(
                provider: ModelProvider,
                actual_provider_name: str,
                actual_pricing_provider_name: str,
                actual_model: str,
                actual_usage_dialect: UsageDialect,
                billing_identity: BillingIdentity | None,
                model_request: ModelRequest,
                dispatch_attempt: int,
                max_dispatch_attempts: int,
                dispatch: Callable[[], Awaitable[tuple[str, dict[str, Any]]]],
                *,
                model_attempt_identity: ModelAttemptIdentity,
            ) -> tuple[str, dict[str, Any]]:
                model_attempt_identity = copy_model_attempt_identity(model_attempt_identity)
                validate_live_compaction_semantics()
                actual_pricing_provider_name = _require_application_compaction_event_text(
                    actual_pricing_provider_name,
                    "compactor_provider_name",
                )
                actual_model = _require_application_compaction_event_text(
                    actual_model,
                    "compactor_model",
                )
                actual_usage_dialect = copy_usage_dialect(
                    actual_usage_dialect,
                    "provider.usage_dialect",
                )
                if actual_pricing_provider_name != compactor_provider_name:
                    raise RuntimeError(
                        "Compaction dispatch provider identity differs from its admitted identity."
                    )
                if actual_model != compactor_model:
                    raise RuntimeError(
                        "Compaction dispatch model identity differs from its admitted identity."
                    )
                limit_decision = await self._run_limit_controller.evaluate_operation_run_limit(
                    session=loaded_session,
                    limits=request.limits,
                    operation_events=observed_dispatch_completion_events,
                    operation_started_at=operation_started_at,
                )
                if limit_decision is not None:
                    raise RuntimeError(f"Compaction limit reached: {limit_decision.message}")
                budget_operation_events = list(attempt_events)
                budget_operation_event_count = len(budget_operation_events)
                budget_error = await self._enforce_compaction_budget_limits(
                    session=loaded_session,
                    budget_limits=budget_limits,
                    app_policy_budget_limits=app_policy_budget_limits,
                    attempt_events=budget_operation_events,
                    reached_budget_keys=reached_budget_keys,
                    request=request,
                    operation_id=operation_id,
                    attempt_id=attempt_id,
                    registered_agent=registered_agent,
                    environment_name=environment_name,
                    compactor=compactor_name,
                    provider_name=actual_pricing_provider_name,
                    model=actual_model,
                    execution_identity=model_attempt_identity,
                    billing_identity_state=resolved_billing_identity(billing_identity),
                )
                attempt_events.extend(budget_operation_events[budget_operation_event_count:])
                if budget_error is not None:
                    await publish_dispatch_budget_events()
                    raise budget_error

                reservation_start = len(budget_reservations)
                reservation_failure = await self._reserve_compaction_budget(
                    session=loaded_session,
                    registered_agent=registered_agent,
                    environment_name=environment_name,
                    budget_limits=dispatch_settlement_budget_limits,
                    provider_name=actual_pricing_provider_name,
                    model=actual_model,
                    model_attempt_identity=model_attempt_identity,
                    request=request,
                    operation_id=operation_id,
                    attempt_id=attempt_id,
                    compactor=compactor_name,
                    reservations=budget_reservations,
                    events=attempt_events,
                    billing_identity=billing_identity,
                    reservation_identity_guard=reservation_identity_guard,
                    execution_profile_fingerprint=(compaction_execution_profile.fingerprint),
                )
                dispatch_reservations = budget_reservations[reservation_start:]
                if reservation_failure is not None:
                    attempt_events.append(
                        _application_compaction_ledger_event(
                            event_type=EventType.BUDGET_LIMIT_REACHED,
                            payload=budget_reservation_payload(reservation_failure),
                            request=request,
                            operation_id=operation_id,
                            attempt_id=attempt_id,
                            session=loaded_session,
                            registered_agent=registered_agent,
                            environment_name=environment_name,
                            compactor=compactor_name,
                        )
                    )
                    await publish_dispatch_budget_events()
                    raise RuntimeError(
                        f"Compaction budget reservation failed: {reservation_failure.message}"
                    )
                await publish_dispatch_budget_events()

                provider_dispatch_started = False

                async def tracked_dispatch() -> tuple[str, dict[str, Any]]:
                    nonlocal provider_dispatch_started
                    validate_live_compaction_semantics()
                    await record_compaction_footprint(
                        provider=provider,
                        provider_name=actual_provider_name,
                        model_request=model_request,
                        dispatch_attempt=dispatch_attempt,
                        max_dispatch_attempts=max_dispatch_attempts,
                        model_attempt_identity=model_attempt_identity,
                    )
                    validate_live_compaction_semantics()
                    deferred_dispatch_failure = (
                        await self._run_limit_controller.mark_reservations_dispatched(
                            dispatch_reservations,
                            dispatch_id=model_attempt_identity.model_attempt_id,
                        )
                    )
                    if deferred_dispatch_failure is not None:
                        provider_dispatch_started = True
                        raise deferred_dispatch_failure
                    validate_live_compaction_semantics()
                    provider_dispatch_started = True
                    with _compaction_model_attempt_identity_scope(model_attempt_identity):
                        return await dispatch()

                try:
                    (
                        dispatch_result,
                        lease_failure,
                    ) = await self._run_limit_controller.run_operation_with_reservation_heartbeat(
                        tracked_dispatch,
                        reservations=dispatch_reservations,
                        authoritative_failure_types=(ContextBuildError,),
                        lease_lost_before_dispatch_message=(
                            "Compaction budget reservation lease was lost before provider dispatch."
                        ),
                        authoritative_failure_note=(
                            "Budget reservation lease was also lost as compaction failed"
                        ),
                        concurrent_failure_note=(
                            "Compactor also failed while reservation lease loss was handled"
                        ),
                        completed_metadata_from_result=lambda completed: completed[1],
                    )
                except asyncio.CancelledError as exc:
                    raw_completed = getattr(exc, "completed_metadata", None)
                    completed_event = (
                        dispatch_completion_event(
                            actual_pricing_provider_name=actual_pricing_provider_name,
                            actual_model=actual_model,
                            actual_usage_dialect=actual_usage_dialect,
                            completed_metadata=raw_completed,
                            model_attempt_identity=model_attempt_identity,
                        )
                        if type(raw_completed) is dict
                        else None
                    )

                    async def settle_cancelled_dispatch() -> None:
                        await settle_provider_dispatch(
                            dispatch_reservations,
                            completed_event=completed_event,
                            uncertain_reason=(
                                "Compaction provider dispatch settlement failed "
                                "during cancellation."
                            ),
                            release_reason=(
                                None
                                if provider_dispatch_started
                                else "compaction cancelled before provider dispatch"
                            ),
                        )
                        await publish_dispatch_budget_events()

                    settlement_task = asyncio.create_task(settle_cancelled_dispatch())
                    settlement_failure: BaseException | None = None
                    while not settlement_task.done():
                        try:
                            await asyncio.shield(settlement_task)
                        except asyncio.CancelledError:
                            continue
                        except BaseException as candidate:
                            settlement_failure = candidate
                            break
                    if settlement_failure is None:
                        try:
                            settlement_task.result()
                        except BaseException as candidate:
                            settlement_failure = candidate
                    if settlement_failure is not None:
                        exc.add_note(
                            "Compaction provider cancellation settlement also failed: "
                            f"{type(settlement_failure).__name__}: {settlement_failure}"
                        )
                    raise
                except BaseException as exc:
                    raw_completed = getattr(exc, "completed_metadata", None)
                    completed_event = (
                        dispatch_completion_event(
                            actual_pricing_provider_name=actual_pricing_provider_name,
                            actual_model=actual_model,
                            actual_usage_dialect=actual_usage_dialect,
                            completed_metadata=raw_completed,
                            model_attempt_identity=model_attempt_identity,
                        )
                        if type(raw_completed) is dict
                        else None
                    )
                    try:
                        await settle_provider_dispatch(
                            dispatch_reservations,
                            completed_event=completed_event,
                            uncertain_reason=(
                                "Compaction provider dispatch settlement failed after "
                                "provider error."
                            ),
                            release_reason=(
                                "compaction reservation lease lost before provider dispatch"
                                if not provider_dispatch_started
                                else None
                            ),
                        )
                        await publish_dispatch_budget_events()
                    except BaseException as settlement_failure:
                        if isinstance(settlement_failure, Exception):
                            add_budget_failure_note(
                                exc,
                                operation="compaction provider dispatch settlement",
                                accounting_failure=settlement_failure,
                            )
                        else:
                            exc.add_note(
                                "Compaction provider dispatch settlement also ended with "
                                f"{type(settlement_failure).__name__}; "
                                "the original failure remains authoritative."
                            )
                    raise

                completed_event = dispatch_completion_event(
                    actual_pricing_provider_name=actual_pricing_provider_name,
                    actual_model=actual_model,
                    actual_usage_dialect=actual_usage_dialect,
                    completed_metadata=dispatch_result[1],
                    model_attempt_identity=model_attempt_identity,
                )

                async def settle_completed_dispatch() -> None:
                    await settle_provider_dispatch(
                        dispatch_reservations,
                        completed_event=completed_event,
                        uncertain_reason=(
                            "Compaction provider dispatch settlement failed after completion."
                        ),
                    )
                    await publish_dispatch_budget_events()

                settlement_task = asyncio.create_task(settle_completed_dispatch())
                settlement_outcome = await await_shielded_task_outcome(settlement_task)
                settlement_cancellation = settlement_outcome.cancellation
                if settlement_outcome.error is not None:
                    settlement_failure = settlement_outcome.error
                    if (
                        isinstance(settlement_failure, asyncio.CancelledError)
                        and settlement_cancellation is None
                    ):
                        settlement_failure = unexpected_child_cancellation_error(
                            settlement_failure,
                            operation="Completed compaction provider dispatch settlement",
                        )
                    if settlement_cancellation is not None:
                        settlement_cancellation.__dict__["completed_metadata"] = copy_json_value(
                            dispatch_result[1],
                            "completed_metadata",
                        )
                        settlement_cancellation.add_note(
                            "Completed compaction provider dispatch settlement also failed: "
                            f"{type(settlement_failure).__name__}: {settlement_failure}"
                        )
                        raise settlement_cancellation from settlement_failure
                    raise settlement_failure
                if settlement_outcome.result is not None:
                    raise RuntimeError(
                        "Completed compaction provider dispatch settlement returned an "
                        "unexpected result."
                    )
                if settlement_cancellation is not None:
                    settlement_cancellation.__dict__["completed_metadata"] = copy_json_value(
                        dispatch_result[1],
                        "completed_metadata",
                    )
                    raise settlement_cancellation
                if lease_failure is not None:
                    raise lease_failure
                return dispatch_result

            async def run_provider_dispatch(
                provider: ModelProvider,
                actual_provider_name: str,
                actual_pricing_provider_name: str,
                actual_model: str,
                actual_usage_dialect: UsageDialect,
                billing_identity: BillingIdentity | None,
                model_request: ModelRequest,
                dispatch_attempt: int,
                max_dispatch_attempts: int,
                dispatch: Callable[[], Awaitable[tuple[str, dict[str, Any]]]],
            ) -> tuple[str, dict[str, Any]]:
                validate_live_compaction_semantics()
                model_attempt_identity = compaction_identity_ledger.begin_dispatch(
                    model_step_identity.new_attempt()
                )
                try:
                    return await _run_provider_dispatch(
                        provider,
                        actual_provider_name,
                        actual_pricing_provider_name,
                        actual_model,
                        actual_usage_dialect,
                        billing_identity,
                        model_request,
                        dispatch_attempt,
                        max_dispatch_attempts,
                        dispatch,
                        model_attempt_identity=model_attempt_identity,
                    )
                finally:
                    compaction_identity_ledger.end_dispatch(model_attempt_identity)

            async def run_identity_only_dispatch(
                provider: ModelProvider,
                actual_provider_name: str,
                actual_pricing_provider_name: str,
                actual_model: str,
                actual_usage_dialect: UsageDialect,
                billing_identity: BillingIdentity | None,
                model_request: ModelRequest,
                dispatch_attempt: int,
                max_dispatch_attempts: int,
                dispatch: Callable[[], Awaitable[tuple[str, dict[str, Any]]]],
            ) -> tuple[str, dict[str, Any]]:
                """Identify a built-in dispatch reached through an opaque compactor."""

                del (
                    actual_pricing_provider_name,
                    actual_model,
                    actual_usage_dialect,
                    billing_identity,
                )
                validate_live_compaction_semantics()
                if has_compaction_accounting_limits:
                    raise RuntimeError(
                        "Explicit compaction declared deterministic execution but "
                        "attempted provider work under run or cost limits."
                    )
                model_attempt_identity = compaction_identity_ledger.begin_dispatch(
                    model_step_identity.new_attempt()
                )
                try:
                    await record_compaction_footprint(
                        provider=provider,
                        provider_name=actual_provider_name,
                        model_request=model_request,
                        dispatch_attempt=dispatch_attempt,
                        max_dispatch_attempts=max_dispatch_attempts,
                        model_attempt_identity=model_attempt_identity,
                    )
                    validate_live_compaction_semantics()
                    with _compaction_model_attempt_identity_scope(model_attempt_identity):
                        return await dispatch()
                finally:
                    compaction_identity_ledger.end_dispatch(model_attempt_identity)

            async def execute_compaction() -> ContextBuildResult:
                try:
                    with (
                        _compaction_completion_publisher_scope(publish_compaction_completions),
                        _context_secret_redactor_scope(self._secret_redactor),
                        _defer_billing_identity_cancellation_scope(),
                    ):
                        result = await context_policy.build_with_checkpoint(
                            ContextRequest(
                                session=loaded_session,
                                agent=_session_agent_spec(
                                    registered_agent=registered_agent,
                                    session=loaded_session,
                                ),
                                messages=transcript,
                                step=1,
                                environment_name=environment_name,
                                metadata={
                                    "operation_id": operation_id,
                                    "reason": request.reason,
                                },
                                force_compaction=True,
                                force_bounded_compaction=True,
                                compaction_instructions=request.instructions,
                            ),
                            checkpoint=compaction_invocation_checkpoint,
                        )
                except ContextBuildError as error:
                    sanitize_context_build_error_checkpoint(
                        error,
                        redactor=self._secret_redactor,
                    )
                    raise
                safe_checkpoint, safe_checkpoint_event_payload = (
                    sanitize_context_build_result_checkpoint(
                        result,
                        redactor=self._secret_redactor,
                    )
                )
                return result.model_copy(
                    update={
                        "checkpoint": safe_checkpoint,
                        "checkpoint_event_payload": safe_checkpoint_event_payload,
                    },
                    deep=True,
                )

            async def execute_compaction_with_limits() -> tuple[
                ContextBuildResult,
                BaseException | None,
            ]:
                dispatch_runner = (
                    run_provider_dispatch
                    if compactor_provider_name is not None
                    else run_identity_only_dispatch
                )
                with _automatic_compaction_dispatch_runner_scope(dispatch_runner):
                    if compactor_provider_name is not None:
                        return await execute_compaction(), None
                    return (
                        await self._run_limit_controller.run_operation_with_reservation_heartbeat(
                            execute_compaction,
                            reservations=budget_reservations,
                            authoritative_failure_types=(ContextBuildError,),
                            lease_lost_before_dispatch_message=(
                                "Compaction budget reservation lease was lost before provider "
                                "dispatch."
                            ),
                            authoritative_failure_note=(
                                "Budget reservation lease was also lost as compaction failed"
                            ),
                            concurrent_failure_note=(
                                "Compactor also failed while reservation lease loss was handled"
                            ),
                        )
                    )

            (
                result,
                reservation_lease_failure,
            ) = await _run_while_session_operation_claimed(
                execute_compaction_with_limits,
                heartbeat_task=operation_heartbeat_task,
            )
            compacted_checkpoint = result.checkpoint
            checkpoint_event_payload = result.checkpoint_event_payload
            if compacted_checkpoint is None or checkpoint_event_payload is None:
                raise ValueError("Session has no complete older context to compact.")

            telemetry_events = [
                application_compaction_telemetry_event(telemetry)
                for telemetry in result.compaction_telemetry
                if telemetry.event_type != EventType.CONTEXT_COMPACTION_STARTED
                and not (
                    telemetry.event_type == EventType.MODEL_COMPLETED
                    and telemetry.payload.get(_COMPACTION_ATTEMPT_ID_KEY)
                    in materialized_compaction_attempt_ids
                )
            ]
            attempt_events.extend(
                event for event in telemetry_events if event.type == EventType.MODEL_COMPLETED
            )
            materialized_compaction_attempt_ids.update(
                completion_attempt_id
                for telemetry in result.compaction_telemetry
                if telemetry.event_type == EventType.MODEL_COMPLETED
                and type(completion_attempt_id := telemetry.payload.get(_COMPACTION_ATTEMPT_ID_KEY))
                is str
            )
            attempt_events.extend(deferred_dispatch_settlement_events)
            deferred_dispatch_settlement_events.clear()
            async for event in self._reconcile_compaction_budget_reservations(
                budget_reservations,
                model_completed_events=[
                    event for event in attempt_events if event.type == EventType.MODEL_COMPLETED
                ],
                session=loaded_session,
                registered_agent=registered_agent,
                environment_name=environment_name,
                request=request,
                operation_id=operation_id,
                attempt_id=attempt_id,
                compactor=compactor_name,
            ):
                attempt_events.append(event)
            budget_reservations_settled = True
            if reservation_lease_failure is not None:
                raise reservation_lease_failure
            limit_decision = await self._run_limit_controller.evaluate_operation_run_limit(
                session=loaded_session,
                limits=request.limits,
                operation_events=[
                    *attempt_events,
                    *observed_dispatch_completion_events,
                ],
                operation_started_at=operation_started_at,
            )
            if limit_decision is not None:
                raise RuntimeError(f"Compaction limit reached: {limit_decision.message}")
            final_budget_operation_events = list(attempt_events)
            final_budget_operation_event_count = len(final_budget_operation_events)
            budget_error = await self._enforce_compaction_budget_limits(
                session=loaded_session,
                budget_limits=budget_limits,
                app_policy_budget_limits=app_policy_budget_limits,
                attempt_events=final_budget_operation_events,
                reached_budget_keys=reached_budget_keys,
                request=request,
                operation_id=operation_id,
                attempt_id=attempt_id,
                registered_agent=registered_agent,
                environment_name=environment_name,
                compactor=compactor_name,
                provider_name=compactor_provider_name,
                model=compactor_model,
                execution_identity=model_step_identity,
            )
            attempt_events.extend(
                final_budget_operation_events[final_budget_operation_event_count:]
            )
            if budget_error is not None:
                raise budget_error
            checkpoint_event = Event(
                type=EventType.SESSION_CHECKPOINTED,
                session_id=loaded_session.id,
                agent_name=registered_agent.spec.name,
                environment_name=environment_name,
                payload={
                    **copy_json_value(
                        checkpoint_event_payload,
                        "checkpoint_event_payload",
                    ),
                    **model_step_identity.payload(),
                    **_application_compaction_causal_payload(
                        request=request,
                        operation_id=operation_id,
                        attempt_id=attempt_id,
                        source_cursor=request.expected_transcript_cursor,
                        result_cursor=checkpoint_event_payload.get("compacted_transcript_cursor"),
                        compactor=compactor_name,
                    ),
                },
            )
            published_events = self._event_writer.prepare_many(
                attribute_events_to_current_interaction(
                    governed_compaction_events(
                        [
                            *attempt_events,
                            *[
                                event
                                for event in telemetry_events
                                if event.type != EventType.MODEL_COMPLETED
                            ],
                            checkpoint_event,
                        ]
                    )
                )
            )
            event_ids = [started_event.id, *[event.id for event in published_events]]
            try:
                heartbeat_blocker = await stop_claim_heartbeat()
            except BaseException as heartbeat_failure:
                _attach_context_build_termination_diagnostics(
                    heartbeat_failure,
                    compaction_telemetry=result.compaction_telemetry,
                )
                raise heartbeat_failure
            if heartbeat_blocker is not None:
                heartbeat_failure = SessionCompactionAttemptSuperseded(
                    "Session compaction operation heartbeat could not stop before "
                    "terminal publication."
                )
                _attach_context_build_termination_diagnostics(
                    heartbeat_failure,
                    compaction_telemetry=result.compaction_telemetry,
                )
                raise heartbeat_failure
            terminal_claim_expires_at: datetime | None = None

            def capture_terminal_claim_expiry(expires_at: datetime) -> None:
                nonlocal terminal_claim_expires_at
                terminal_claim_expires_at = expires_at

            def require_unexpired_terminal_commit(commit_at: datetime) -> None:
                if terminal_claim_expires_at is None:
                    raise AssertionError(
                        "Session compaction terminal publication did not capture its claim."
                    )
                if terminal_claim_expires_at <= commit_at:
                    raise SessionCompactionAttemptSuperseded(
                        "Session compaction terminal publication claim expired before commit."
                    )

            async def publish_terminal_success() -> None:
                await self.session_store.publish_session_operation_guarded_with_store_time(
                    loaded_session.id,
                    idempotency_key=request.idempotency_key,
                    operation_transform=lambda _session, checkpoint, persisted_record, store_now: (
                        _complete_session_operation_checkpoint(
                            checkpoint=checkpoint,
                            persisted_record=persisted_record,
                            compacted_checkpoint=compacted_checkpoint,
                            idempotency_key=request.idempotency_key,
                            operation_id=operation_id,
                            attempt_id=attempt_id,
                            event_ids=event_ids,
                            result_cursor=checkpoint_event_payload.get(
                                "compacted_transcript_cursor"
                            ),
                            completed_at=store_now,
                            on_terminalize=capture_terminal_claim_expiry,
                        )
                    ),
                    commit_guard=lambda: None,
                    commit_time_guard=require_unexpired_terminal_commit,
                    events=published_events,
                    expected_statuses=_session_checkpoint_admission._RESUMABLE_SESSION_STATUSES,
                    expected_run_epoch=request.expected_run_epoch,
                    expected_transcript_cursor=request.expected_transcript_cursor,
                    context_view_compaction_cursor=request.expected_transcript_cursor,
                )

            terminal_publication_task = asyncio.create_task(publish_terminal_success())
            terminal_outcome = await self._await_session_operation_store_task(
                terminal_publication_task,
            )
            terminal_cancellation = terminal_outcome.cancellation
            terminal_publication_unknown = terminal_outcome.timed_out
            if terminal_publication_unknown:
                publication_failure: BaseException | None = TimeoutError(
                    "Session compaction terminal publication exceeded its bounded store wait."
                )
            else:
                publication_failure = terminal_outcome.error
            if isinstance(publication_failure, asyncio.CancelledError):
                publication_failure = unexpected_child_cancellation_error(
                    publication_failure,
                    operation="Session compaction terminal publication",
                )
            if publication_failure is not None:

                async def reconcile_terminal_success() -> tuple[
                    list[bool],
                    dict[str, Any] | None,
                    dict[str, Any] | None,
                ]:
                    commit_states = [
                        await self._event_writer.is_persisted(event) for event in published_events
                    ]
                    operation_record = await self.session_store.load_session_operation(
                        loaded_session.id,
                        request.idempotency_key,
                    )
                    checkpoint = await self.session_store.load_checkpoint(loaded_session.id)
                    return commit_states, operation_record, checkpoint

                reconciliation_task = asyncio.create_task(reconcile_terminal_success())
                reconciliation_outcome = await self._await_session_operation_store_task(
                    reconciliation_task,
                    cancellation=terminal_cancellation,
                )
                terminal_cancellation = reconciliation_outcome.cancellation
                reconciliation_error: BaseException | None
                if reconciliation_outcome.timed_out:
                    reconciliation_error = TimeoutError(
                        "Session compaction terminal publication reconciliation exceeded "
                        "its bounded store wait."
                    )
                else:
                    reconciliation_error = reconciliation_outcome.error
                if isinstance(reconciliation_error, asyncio.CancelledError):
                    reconciliation_error = unexpected_child_cancellation_error(
                        reconciliation_error,
                        operation="Session compaction terminal publication reconciliation",
                    )
                if reconciliation_error is not None:
                    if terminal_publication_unknown:
                        # The guarded write can still resolve after this caller
                        # returns. Leave the durable claim for replay/recovery
                        # instead of racing it with a contradictory failure.
                        operation_published = True
                    publication_failure.add_note(
                        "Session compaction terminal publication reconciliation also "
                        f"failed: {type(reconciliation_error).__name__}: "
                        f"{reconciliation_error}"
                    )
                    if terminal_cancellation is not None:
                        terminal_cancellation.add_note(
                            "Terminal publication and reconciliation failed during cancellation."
                        )
                        raise terminal_cancellation from publication_failure
                    _attach_context_build_termination_diagnostics(
                        publication_failure,
                        compaction_telemetry=result.compaction_telemetry,
                    )
                    raise publication_failure from reconciliation_error
                reconciled = reconciliation_outcome.result
                if reconciled is None:
                    raise AssertionError(
                        "Session compaction terminal reconciliation lost its result."
                    ) from publication_failure
                commit_states, operation_record, durable_checkpoint = reconciled
                durable_event_ids = (
                    operation_record.get("event_ids", []) if type(operation_record) is dict else []
                )
                operation_completed = (
                    type(operation_record) is dict
                    and operation_record.get("status") == "completed"
                    and operation_record.get("operation_id") == operation_id
                    and operation_record.get("current_attempt_id") == attempt_id
                    and operation_record.get("result_transcript_cursor")
                    == checkpoint_event_payload.get("compacted_transcript_cursor")
                    and all(event_id in durable_event_ids for event_id in event_ids)
                )
                expected_compaction_checkpoint = compacted_checkpoint.get(
                    _CONTEXT_COMPACTION_OPERATION_KIND
                )
                checkpoint_completed = (
                    type(durable_checkpoint) is dict
                    and durable_checkpoint.get(_CONTEXT_COMPACTION_OPERATION_KIND)
                    == expected_compaction_checkpoint
                    and _session_operation_state._active_session_operation_id(durable_checkpoint)
                    != operation_id
                )
                events_completed = all(commit_states)
                fully_committed = operation_completed and checkpoint_completed and events_completed
                anything_committed = (
                    operation_completed or checkpoint_completed or any(commit_states)
                )
                if anything_committed and not fully_committed:
                    operation_published = True
                    committed_events = [
                        event
                        for event, committed in zip(
                            published_events,
                            commit_states,
                            strict=True,
                        )
                        if committed
                    ]
                    (
                        fan_out_error,
                        terminal_cancellation,
                    ) = await self._fan_out_reconciled_session_operation_events(
                        committed_events,
                        cancellation=terminal_cancellation,
                    )
                    atomicity_error = RuntimeError(
                        "The session store violated atomic terminal compaction publication."
                    )
                    if fan_out_error is not None:
                        atomicity_error.add_note(
                            "Committed terminal event side-effect delivery also failed: "
                            f"{type(fan_out_error).__name__}: {fan_out_error}"
                        )
                    if terminal_cancellation is not None:
                        terminal_cancellation.add_note(str(atomicity_error))
                        raise terminal_cancellation from publication_failure
                    raise atomicity_error from publication_failure
                if not fully_committed:
                    _attach_context_build_termination_diagnostics(
                        publication_failure,
                        compaction_telemetry=result.compaction_telemetry,
                    )
                    if terminal_publication_unknown:
                        operation_published = True
                    if terminal_cancellation is not None:
                        raise terminal_cancellation from publication_failure
                    raise publication_failure
            operation_published = True
            (
                fan_out_error,
                terminal_cancellation,
            ) = await self._fan_out_reconciled_session_operation_events(
                published_events,
                cancellation=terminal_cancellation,
            )
            if terminal_cancellation is not None:
                if fan_out_error is not None:
                    terminal_cancellation.add_note(
                        "Terminal event side-effect delivery also failed: "
                        f"{type(fan_out_error).__name__}: {fan_out_error}"
                    )
                raise terminal_cancellation
            if publication_failure is not None and not isinstance(publication_failure, Exception):
                raise publication_failure
            if fan_out_error is not None:
                raise fan_out_error
            for event in prepublished_dispatch_events:
                yield event
            prepublished_dispatch_events.clear()
            for event in published_events:
                yield event
        except GeneratorExit as exc:
            operation_failure = exc
            if operation_published:
                raise
            termination_telemetry = context_build_termination_compaction_telemetry(exc)
            if not termination_telemetry:
                if budget_reservations and not budget_reservations_settled:
                    release_events: list[Event] = []
                    try:
                        async for event in self._release_compaction_budget_reservations(
                            budget_reservations,
                            session=loaded_session,
                            registered_agent=registered_agent,
                            environment_name=environment_name,
                            request=request,
                            operation_id=operation_id,
                            attempt_id=attempt_id,
                            compactor=compactor_name,
                            reason="compaction operation abandoned",
                        ):
                            release_events.append(event)
                    finally:
                        if release_events:
                            await self._persist_compaction_attempt_events(
                                session=loaded_session,
                                request=request,
                                operation_id=operation_id,
                                attempt_id=attempt_id,
                                events=release_events,
                                heartbeat_state=operation_heartbeat_state,
                                persisted_event_ids=persisted_attempt_event_ids,
                                event_inventory=attempt_event_inventory,
                                unresolved_store_tasks=(unresolved_attempt_publication_tasks),
                            )
                            await self._event_writer.fan_out_persisted(release_events)
                        budget_reservations_settled = not budget_reservations
                raise
            try:
                failed_model_events = [
                    application_compaction_telemetry_event(telemetry)
                    for telemetry in termination_telemetry
                    if telemetry.event_type == EventType.MODEL_COMPLETED
                    and telemetry.payload.get(_COMPACTION_ATTEMPT_ID_KEY)
                    not in materialized_compaction_attempt_ids
                ]
                attempt_events.extend(failed_model_events)
                attempt_events.extend(deferred_dispatch_settlement_events)
                deferred_dispatch_settlement_events.clear()
                if budget_reservations and not budget_reservations_settled:
                    model_completed_events = [
                        event for event in attempt_events if event.type == EventType.MODEL_COMPLETED
                    ]
                    settlement_stream = (
                        self._reconcile_compaction_budget_reservations(
                            budget_reservations,
                            model_completed_events=model_completed_events,
                            session=loaded_session,
                            registered_agent=registered_agent,
                            environment_name=environment_name,
                            request=request,
                            operation_id=operation_id,
                            attempt_id=attempt_id,
                            compactor=compactor_name,
                        )
                        if model_completed_events
                        else self._release_compaction_budget_reservations(
                            budget_reservations,
                            session=loaded_session,
                            registered_agent=registered_agent,
                            environment_name=environment_name,
                            request=request,
                            operation_id=operation_id,
                            attempt_id=attempt_id,
                            compactor=compactor_name,
                            reason="compaction operation abandoned",
                        )
                    )
                    async for event in settlement_stream:
                        attempt_events.append(event)
                    budget_reservations_settled = True
                failed_telemetry = next(
                    (
                        telemetry
                        for telemetry in reversed(termination_telemetry)
                        if telemetry.event_type == EventType.CONTEXT_COMPACTION_FAILED
                    ),
                    None,
                )
                failed_payload = _application_compaction_causal_payload(
                    request=request,
                    operation_id=operation_id,
                    attempt_id=attempt_id,
                    source_cursor=request.expected_transcript_cursor,
                    compactor=compactor_name,
                )
                failed_payload.update(model_step_identity.payload())
                if failed_telemetry is not None:
                    failed_payload = application_compaction_telemetry_event(
                        failed_telemetry
                    ).payload
                safe_error_type = (
                    _application_compaction_event_text(type(exc).__name__) or "BaseException"
                )
                failed_payload["error_type"] = safe_error_type
                failed_event = Event(
                    type=EventType.CONTEXT_COMPACTION_FAILED,
                    session_id=loaded_session.id,
                    agent_name=registered_agent.spec.name,
                    environment_name=environment_name,
                    payload=failed_payload,
                )
                unpublished_attempt_events = [
                    event for event in attempt_events if event.id not in persisted_attempt_event_ids
                ]
                failed_events = self._event_writer.prepare_many(
                    governed_compaction_events([*unpublished_attempt_events, failed_event])
                )
                failed_terminal_claim_expires_at: datetime | None = None

                def capture_failed_terminal_claim_expiry(expires_at: datetime) -> None:
                    nonlocal failed_terminal_claim_expires_at
                    failed_terminal_claim_expires_at = expires_at

                def require_unexpired_failed_terminal_commit(commit_at: datetime) -> None:
                    if failed_terminal_claim_expires_at is None:
                        return
                    if failed_terminal_claim_expires_at <= commit_at:
                        raise SessionCompactionAttemptSuperseded(
                            "Session compaction failure publication claim expired before commit."
                        )

                await self.session_store.publish_session_operation_guarded_with_store_time(
                    loaded_session.id,
                    idempotency_key=request.idempotency_key,
                    operation_transform=(
                        lambda session, checkpoint, persisted_record, store_now: (
                            _fail_session_operation_checkpoint(
                                idempotency_key=request.idempotency_key,
                                operation_id=operation_id,
                                attempt_id=attempt_id,
                                failed_event_id=failed_event.id,
                                attempt_event_ids=[
                                    event.id for event in unpublished_attempt_events
                                ],
                                error_type=safe_error_type,
                                clock=lambda: store_now,
                                on_terminalize=capture_failed_terminal_claim_expiry,
                            )(session, checkpoint, persisted_record)
                        )
                    ),
                    commit_guard=lambda: None,
                    commit_time_guard=require_unexpired_failed_terminal_commit,
                    events=failed_events,
                )
                operation_published = True
                await self._run_limit_controller.acknowledge_budget_settlement_events(failed_events)
                await self._event_writer.fan_out_persisted(failed_events)
            except BaseException as cleanup_error:
                cleanup_error.add_note(
                    "Compaction was abandoned by GeneratorExit, but its usage and "
                    "failure accounting could not be durably published. "
                    "The accounting failure is authoritative."
                )
                operation_failure = cleanup_error
                raise cleanup_error from exc
            raise
        except BaseException as exc:
            operation_failure = exc
            if operation_published:
                raise
            authoritative_failure = exc
            failed_events: list[Event] = []
            failure_accounting_published = False

            async def finalize_failed_operation() -> None:
                nonlocal budget_reservations_settled
                nonlocal failed_events
                nonlocal failure_accounting_published
                nonlocal operation_published

                failure_telemetry = (
                    authoritative_failure.compaction_telemetry
                    if isinstance(authoritative_failure, ContextBuildError)
                    else context_build_termination_compaction_telemetry(authoritative_failure)
                )
                if failure_telemetry:
                    failed_model_events = [
                        application_compaction_telemetry_event(telemetry)
                        for telemetry in failure_telemetry
                        if telemetry.event_type == EventType.MODEL_COMPLETED
                        and telemetry.payload.get(_COMPACTION_ATTEMPT_ID_KEY)
                        not in materialized_compaction_attempt_ids
                    ]
                    existing_completion_attempt_ids = {
                        completion_attempt_id
                        for event in attempt_events
                        if event.type == EventType.MODEL_COMPLETED
                        and type(
                            completion_attempt_id := event.payload.get(_COMPACTION_ATTEMPT_ID_KEY)
                        )
                        is str
                    }
                    for event in failed_model_events:
                        completion_attempt_id = event.payload.get(_COMPACTION_ATTEMPT_ID_KEY)
                        if (
                            type(completion_attempt_id) is str
                            and completion_attempt_id in existing_completion_attempt_ids
                        ):
                            continue
                        attempt_events.append(event)
                        if type(completion_attempt_id) is str:
                            existing_completion_attempt_ids.add(completion_attempt_id)
                attempt_events.extend(deferred_dispatch_settlement_events)
                deferred_dispatch_settlement_events.clear()
                if budget_reservations and not budget_reservations_settled:
                    model_completed_events = [
                        event for event in attempt_events if event.type == EventType.MODEL_COMPLETED
                    ]
                    if model_completed_events:
                        settlement_stream = self._reconcile_compaction_budget_reservations(
                            budget_reservations,
                            model_completed_events=model_completed_events,
                            session=loaded_session,
                            registered_agent=registered_agent,
                            environment_name=environment_name,
                            request=request,
                            operation_id=operation_id,
                            attempt_id=attempt_id,
                            compactor=compactor_name,
                        )
                    elif isinstance(
                        authoritative_failure,
                        BudgetReservationLeaseLostBeforeModelDispatch,
                    ):
                        settlement_stream = self._release_compaction_budget_reservations(
                            budget_reservations,
                            session=loaded_session,
                            registered_agent=registered_agent,
                            environment_name=environment_name,
                            request=request,
                            operation_id=operation_id,
                            attempt_id=attempt_id,
                            compactor=compactor_name,
                            reason="compaction reservation lease lost before provider dispatch",
                        )
                    elif isinstance(authoritative_failure, BudgetReservationLeaseLost):
                        settlement_stream = (
                            self._reconcile_uncertain_compaction_budget_reservations(
                                budget_reservations,
                                session=loaded_session,
                                registered_agent=registered_agent,
                                environment_name=environment_name,
                                request=request,
                                operation_id=operation_id,
                                attempt_id=attempt_id,
                                compactor=compactor_name,
                            )
                        )
                    else:
                        settlement_stream = self._release_compaction_budget_reservations(
                            budget_reservations,
                            session=loaded_session,
                            registered_agent=registered_agent,
                            environment_name=environment_name,
                            request=request,
                            operation_id=operation_id,
                            attempt_id=attempt_id,
                            compactor=compactor_name,
                            reason="compaction provider step did not complete",
                        )
                    settlement_failure: BaseException | None = None
                    try:
                        async for event in settlement_stream:
                            attempt_events.append(event)
                    except BaseException as candidate:
                        settlement_failure = candidate
                    budget_reservations_settled = not budget_reservations
                    if settlement_failure is not None:
                        if isinstance(settlement_failure, Exception):
                            add_budget_failure_note(
                                authoritative_failure,
                                operation="compaction settlement",
                                accounting_failure=settlement_failure,
                            )
                        else:
                            authoritative_failure.add_note(
                                "Compaction settlement also ended with "
                                f"{type(settlement_failure).__name__}; "
                                "the original failure remains authoritative."
                            )
                failed_payload = _application_compaction_causal_payload(
                    request=request,
                    operation_id=operation_id,
                    attempt_id=attempt_id,
                    source_cursor=request.expected_transcript_cursor,
                    compactor=compactor_name,
                )
                failed_payload.update(model_step_identity.payload())
                if failure_telemetry:
                    failed_telemetry = next(
                        (
                            telemetry
                            for telemetry in reversed(failure_telemetry)
                            if telemetry.event_type == EventType.CONTEXT_COMPACTION_FAILED
                        ),
                        None,
                    )
                    if failed_telemetry is not None:
                        failed_payload = application_compaction_telemetry_event(
                            failed_telemetry
                        ).payload
                safe_error_type = (
                    _application_compaction_event_text(type(authoritative_failure).__name__)
                    or "BaseException"
                )
                failed_payload["error_type"] = safe_error_type
                failed_event = Event(
                    type=EventType.CONTEXT_COMPACTION_FAILED,
                    session_id=loaded_session.id,
                    agent_name=registered_agent.spec.name,
                    environment_name=environment_name,
                    payload=failed_payload,
                )
                unpublished_attempt_events = [
                    event for event in attempt_events if event.id not in persisted_attempt_event_ids
                ]
                failed_events = self._event_writer.prepare_many(
                    attribute_events_to_current_interaction(
                        governed_compaction_events([*unpublished_attempt_events, failed_event])
                    )
                )
                failed_terminal_claim_expires_at: datetime | None = None

                def capture_failed_terminal_claim_expiry(expires_at: datetime) -> None:
                    nonlocal failed_terminal_claim_expires_at
                    failed_terminal_claim_expires_at = expires_at

                def require_unexpired_failed_terminal_commit(commit_at: datetime) -> None:
                    if failed_terminal_claim_expires_at is None:
                        return
                    if failed_terminal_claim_expires_at <= commit_at:
                        raise SessionCompactionAttemptSuperseded(
                            "Session compaction failure publication claim expired before commit."
                        )

                await self.session_store.publish_session_operation_guarded_with_store_time(
                    loaded_session.id,
                    idempotency_key=request.idempotency_key,
                    operation_transform=(
                        lambda session, checkpoint, persisted_record, store_now: (
                            _fail_session_operation_checkpoint(
                                idempotency_key=request.idempotency_key,
                                operation_id=operation_id,
                                attempt_id=attempt_id,
                                failed_event_id=failed_event.id,
                                attempt_event_ids=[
                                    event.id for event in unpublished_attempt_events
                                ],
                                error_type=safe_error_type,
                                clock=lambda: store_now,
                                on_terminalize=capture_failed_terminal_claim_expiry,
                            )(session, checkpoint, persisted_record)
                        )
                    ),
                    commit_guard=lambda: None,
                    commit_time_guard=require_unexpired_failed_terminal_commit,
                    events=failed_events,
                )
                operation_published = True
                failure_accounting_published = True
                await self._run_limit_controller.acknowledge_budget_settlement_events(failed_events)
                await self._event_writer.fan_out_persisted(failed_events)

            async def stop_heartbeat_and_finalize_failed_operation() -> None:
                heartbeat_blocker = await stop_claim_heartbeat()
                uncertain_publication_tasks = set(unresolved_attempt_publication_tasks)
                publication_blockers = {
                    task for task in uncertain_publication_tasks if not task.done()
                }
                cost_bearing_publication_blockers = {
                    task for task in unresolved_completion_publication_tasks if not task.done()
                }
                if not uncertain_publication_tasks and heartbeat_blocker is None:
                    await finalize_failed_operation()
                    return

                store_blockers = set(publication_blockers)
                if heartbeat_blocker is not None:
                    store_blockers.add(heartbeat_blocker)

                async def finalize_after_pending_store_writes() -> None:
                    await asyncio.gather(*store_blockers, return_exceptions=True)
                    await asyncio.gather(operation_heartbeat_task, return_exceptions=True)
                    reconciled_events: list[Event] = []
                    existing_attempt_event_ids = {event.id for event in attempt_events}
                    for event in attempt_event_inventory.values():
                        durable = event.id in persisted_attempt_event_ids
                        if not durable:
                            durable = await self._event_writer.is_persisted(event)
                        if durable:
                            persisted_attempt_event_ids.add(event.id)
                            reconciled_events.append(event)
                            if (
                                all(
                                    published.id != event.id
                                    for published in prepublished_dispatch_events
                                )
                                and event.id not in yielded_attempt_event_ids
                            ):
                                prepublished_dispatch_events.append(event.model_copy(deep=True))
                            compaction_attempt_id = completion_event_attempt_ids.get(event.id)
                            if (
                                event.type == EventType.MODEL_COMPLETED
                                and type(compaction_attempt_id) is str
                            ):
                                materialized_compaction_attempt_ids.add(compaction_attempt_id)
                                if event.id not in existing_attempt_event_ids:
                                    attempt_events.append(event.model_copy(deep=True))
                                    existing_attempt_event_ids.add(event.id)
                                if all(
                                    observed.id != event.id
                                    for observed in observed_dispatch_completion_events
                                ):
                                    observed_dispatch_completion_events.append(
                                        event.model_copy(deep=True)
                                    )
                    fan_out_failure: BaseException | None = None
                    if reconciled_events:
                        try:
                            await self._event_writer.fan_out_persisted(reconciled_events)
                        except BaseException as exc:
                            fan_out_failure = exc
                    try:
                        await finalize_failed_operation()
                    except BaseException as finalization_failure:
                        if fan_out_failure is not None:
                            raise BaseExceptionGroup(
                                "Explicit compaction failure accounting and reconciled "
                                "event delivery both failed.",
                                [fan_out_failure, finalization_failure],
                            ) from None
                        raise
                    if fan_out_failure is not None:
                        raise fan_out_failure

                authoritative_failure.add_note(
                    "Explicit compaction failure accounting waited for an in-flight "
                    "operation write to release the session store."
                )
                if cost_bearing_publication_blockers:
                    # A completion publication can be the sole durable owner of
                    # already-incurred provider spend. Do not return while only
                    # volatile background work owns that evidence.
                    await finalize_after_pending_store_writes()
                    return

                if (
                    uncertain_publication_tasks
                    and not publication_blockers
                    and heartbeat_blocker is None
                ):
                    # A timed-out write can finish after its first reconciliation
                    # but before failure cleanup begins. Rescan the complete event
                    # inventory even when no task remains live so a late durable
                    # completion retains its original identity and accounting.
                    await finalize_after_pending_store_writes()
                    return

                deferred = asyncio.create_task(finalize_after_pending_store_writes())
                # Nobody else observes how this final accounting ends.
                self._track_detached_session_operation_task(deferred, report_failure=True)

            await self._recovery_ownership.run_cleanup_steps(
                authoritative_failure=authoritative_failure,
                steps=(
                    (
                        "explicit compaction failure accounting",
                        stop_heartbeat_and_finalize_failed_operation,
                    ),
                ),
            )
            # Fatal BaseExceptions cannot safely yield from an async generator.
            # Ordinary exceptions retain the explicit API's durable replay stream.
            if isinstance(authoritative_failure, Exception) and failure_accounting_published:
                for event in prepublished_dispatch_events:
                    yield event
                prepublished_dispatch_events.clear()
                for event in failed_events:
                    yield event
            raise
        finally:
            await self._recovery_ownership.run_cleanup_steps(
                authoritative_failure=operation_failure,
                steps=(
                    (
                        "explicit compaction claim heartbeat stop",
                        stop_claim_heartbeat_and_track,
                    ),
                ),
            )

    async def _load_session_compaction_replay_events(
        self,
        *,
        session_id: str,
        event_ids: tuple[str, ...],
    ) -> list[Event]:
        events: list[Event] = []
        for event_id in event_ids:
            records = await self.session_store.query_events(
                EventQuery(
                    session_id=session_id,
                    event_id=event_id,
                    limit=1,
                )
            )
            if len(records) != 1:
                raise RuntimeError(
                    f"Session compaction replay event is missing from durable history: {event_id}"
                )
            events.append(copy_event(records[0].event))
        return events

    async def _enforce_compaction_budget_limits(
        self,
        *,
        session: Session,
        budget_limits: tuple[BudgetLimit, ...],
        app_policy_budget_limits: tuple[BudgetLimit, ...],
        attempt_events: list[Event],
        reached_budget_keys: set[str],
        request: CompactSessionRequest,
        operation_id: str,
        attempt_id: str,
        registered_agent: runtime_records.RegisteredAgentState,
        environment_name: str | None,
        compactor: str,
        provider_name: str | None,
        model: str | None,
        execution_identity: ModelStepIdentity | ModelAttemptIdentity,
        billing_identity_state: BillingIdentityState = UNRESOLVED_BILLING_IDENTITY,
    ) -> RuntimeError | None:
        execution_payload = _model_execution_identity_payload(execution_identity)
        app_policy_budget_limit_ids = {
            _effective_budget_limit_id(limit) for limit in app_policy_budget_limits
        }
        checks = await self._run_limit_controller.evaluate_operation_budgets(
            session=session,
            budget_limits=budget_limits,
            operation_events=attempt_events,
            operation_model_step_id=execution_identity.model_step_id,
            operation_attempt_id=attempt_id,
            provider_name=provider_name,
            model=model,
            billing_identity_state=billing_identity_state,
        )
        interrupt_error: RuntimeError | None = None
        for outcome in checks:
            budget_limit, check = outcome.limit, outcome.check
            deferred_contextual_check = (
                not isinstance(billing_identity_state, ResolvedBillingIdentity)
                and not check.limit_reached
                and has_deferred_contextual_price(
                    budget_limit.pricing,
                    provider_name=provider_name,
                    model=model,
                )
            )
            if (
                check.budget_limit_id in app_policy_budget_limit_ids
                and not deferred_contextual_check
            ):
                attempt_events.append(
                    _application_compaction_ledger_event(
                        event_type=EventType.BUDGET_CHECKED,
                        payload={
                            **budget_check_payload(check),
                            **execution_payload,
                        },
                        request=request,
                        operation_id=operation_id,
                        attempt_id=attempt_id,
                        session=session,
                        registered_agent=registered_agent,
                        environment_name=environment_name,
                        compactor=compactor,
                    )
                )
            if not check.limit_reached:
                continue
            budget_key = _budget_check_identity(check)
            if budget_key not in reached_budget_keys:
                attempt_events.append(
                    _application_compaction_budget_event(
                        check=check,
                        request=request,
                        operation_id=operation_id,
                        attempt_id=attempt_id,
                        session=session,
                        registered_agent=registered_agent,
                        environment_name=environment_name,
                        compactor=compactor,
                        execution_identity=execution_identity,
                    )
                )
                reached_budget_keys.add(budget_key)
            if budget_limit.action == "interrupt" and interrupt_error is None:
                interrupt_error = RuntimeError(f"Compaction budget limit reached: {check.message}")
        return interrupt_error

    async def _renew_compaction_operation_claim(
        self,
        *,
        session: Session,
        request: CompactSessionRequest,
        operation_id: str,
        attempt_id: str,
        claim_expires_at: datetime,
        commit_started: threading.Event,
    ) -> datetime:
        renewed_until: datetime | None = None

        def require_unexpired_claim_commit(commit_at: datetime) -> None:
            if claim_expires_at <= commit_at:
                raise SessionCompactionAttemptSuperseded(
                    "Session compaction operation renewal claim expired before commit."
                )

        def renew(
            _session: Session,
            checkpoint: dict[str, Any] | None,
            persisted_record: dict[str, Any] | None,
            store_now: datetime,
        ) -> SessionOperationPublication:
            nonlocal renewed_until
            publication, renewed_until = _renew_session_operation_claim_publication(
                checkpoint=checkpoint,
                idempotency_key=request.idempotency_key,
                operation_id=operation_id,
                attempt_id=attempt_id,
                event_ids=[],
                renewed_at=store_now,
                persisted_record=persisted_record,
            )
            return publication

        await self.session_store.publish_session_operation_guarded_with_store_time(
            session.id,
            idempotency_key=request.idempotency_key,
            operation_transform=renew,
            commit_guard=commit_started.set,
            commit_time_guard=require_unexpired_claim_commit,
            events=[],
            expected_statuses=_session_checkpoint_admission._RESUMABLE_SESSION_STATUSES,
            expected_run_epoch=request.expected_run_epoch,
            expected_transcript_cursor=request.expected_transcript_cursor,
        )
        if renewed_until is None:
            raise AssertionError("Session compaction claim renewal did not update its lease.")
        return renewed_until

    async def _reconcile_initial_compaction_claim(
        self,
        *,
        session: Session,
        request: CompactSessionRequest,
        operation_id: str,
        attempt_id: str,
        model_step_id: str,
        request_digest: str,
        started_event: Event,
    ) -> tuple[dict[str, Any], datetime]:
        observed: tuple[dict[str, Any], datetime] | None = None

        def inspect(
            current: Session, checkpoint: dict[str, Any] | None, store_now: datetime
        ) -> None:
            nonlocal observed
            if checkpoint is None:
                raise SessionCompactionAttemptSuperseded("Initial compaction claim is absent.")
            operations = _session_operation_state._session_operation_state(checkpoint)
            record = operations["records"].get(request.idempotency_key)
            if (
                current.instance_id != session.instance_id
                or current.run_epoch != request.expected_run_epoch
                or current.status not in _session_checkpoint_admission._RESUMABLE_SESSION_STATUSES
                or type(record) is not dict
                or record.get("kind") != _CONTEXT_COMPACTION_OPERATION_KIND
                or record.get("status") != "running"
                or record.get("operation_id") != operation_id
                or record.get("current_attempt_id") != attempt_id
                or record.get("model_step_id") != model_step_id
                or record.get("request_digest") != request_digest
                or record.get("source_run_epoch") != request.expected_run_epoch
                or record.get("source_transcript_cursor") != request.expected_transcript_cursor
                or operations.get("active_operation_id") != operation_id
                or started_event.id not in record.get("event_ids", ())
            ):
                raise SessionCompactionAttemptSuperseded(
                    "Initial compaction claim does not prove this request's exact ownership."
                )
            expiry = _session_operation_state._operation_claim_expiry(record)
            if expiry is None or expiry <= store_now:
                raise SessionCompactionAttemptSuperseded(
                    "Initial compaction claim expired before acknowledgement reconciliation."
                )
            observed = (copy_durable_record(checkpoint, "checkpoint"), expiry)
            return None

        await self.session_store.transform_checkpoint_with_store_time(session.id, inspect)
        if observed is None or not await self._event_writer.is_persisted(started_event):
            raise RuntimeError("Initial compaction claim is missing its durable start event.")
        return observed

    async def _reconcile_compaction_operation_claim_expiry(
        self,
        *,
        session_id: str,
        idempotency_key: str,
        operation_id: str,
        attempt_id: str,
    ) -> datetime:
        claim_expires_at: datetime | None = None

        def inspect(
            _session: Session,
            checkpoint: dict[str, Any] | None,
            store_now: datetime,
        ) -> None:
            nonlocal claim_expires_at
            if checkpoint is None:
                raise SessionCompactionAttemptSuperseded(
                    "Session compaction operation claim disappeared during renewal reconciliation."
                )
            operations = _session_operation_state._session_operation_state(checkpoint)
            record = operations["records"].get(idempotency_key)
            if (
                type(record) is not dict
                or record.get("operation_id") != operation_id
                or record.get("status") != "running"
                or record.get("current_attempt_id") != attempt_id
                or operations.get("active_operation_id") != operation_id
            ):
                raise SessionCompactionAttemptSuperseded(
                    "Session compaction operation ownership changed during renewal reconciliation."
                )
            observed_expiry = _session_operation_state._operation_claim_expiry(record)
            if observed_expiry is None:
                raise RuntimeError(
                    "Session compaction operation claim has no valid expiry after renewal."
                )
            if observed_expiry <= store_now:
                raise SessionCompactionAttemptSuperseded(
                    "Session compaction operation claim expired before reconciliation."
                )
            claim_expires_at = observed_expiry
            return None

        await self.session_store.transform_checkpoint_with_store_time(session_id, inspect)
        if claim_expires_at is None:
            raise AssertionError("Session compaction claim reconciliation returned no expiry.")
        return claim_expires_at

    async def _reconcile_compaction_operation_claim_before_deadline(
        self,
        *,
        session_id: str,
        idempotency_key: str,
        operation_id: str,
        attempt_id: str,
        state: _SessionOperationClaimHeartbeatState,
        candidate_claim_deadline_monotonic: float | None = None,
    ) -> datetime:
        remaining_seconds = state.remaining_claim_seconds()
        if remaining_seconds <= 0:
            raise SessionCompactionAttemptSuperseded(
                "Session compaction operation claim expired before reconciliation."
            )
        reconciliation_task = asyncio.create_task(
            self._reconcile_compaction_operation_claim_expiry(
                session_id=session_id,
                idempotency_key=idempotency_key,
                operation_id=operation_id,
                attempt_id=attempt_id,
            )
        )
        state.pending_store_task = reconciliation_task
        reconciliation_task.add_done_callback(state.observe_store_task)
        try:
            done, _pending = await asyncio.wait(
                {reconciliation_task},
                timeout=remaining_seconds,
            )
        except asyncio.CancelledError as cancellation:
            reconciliation_failure: BaseException | None = None
            if reconciliation_task.done():
                reconciliation_failure = _completed_session_operation_child_failure(
                    reconciliation_task,
                    operation="Session compaction claim reconciliation",
                )
            else:
                reconciliation_task.cancel()
            if reconciliation_failure is not None:
                cancellation.add_note(
                    "Session compaction claim reconciliation also failed during "
                    f"cancellation: {type(reconciliation_failure).__name__}: "
                    f"{reconciliation_failure}"
                )
                raise cancellation from reconciliation_failure
            raise
        if reconciliation_task not in done:
            reconciliation_task.cancel()
            raise SessionCompactionAttemptSuperseded(
                "Session compaction operation claim reconciliation was not confirmed "
                "before its lease deadline."
            )
        try:
            durable_claim_expires_at = reconciliation_task.result()
        except asyncio.CancelledError as child_cancellation:
            raise unexpected_child_cancellation_error(
                child_cancellation,
                operation="Session compaction claim reconciliation",
            ) from child_cancellation
        confirmed_claim_expires_at = state.confirmed_claim_expires_at
        if candidate_claim_deadline_monotonic is not None and (
            confirmed_claim_expires_at is None
            or durable_claim_expires_at > confirmed_claim_expires_at
        ):
            state.confirm_claim(
                durable_claim_expires_at,
                claim_deadline_monotonic=candidate_claim_deadline_monotonic,
            )
        return state.confirmed_claim_expires_at or durable_claim_expires_at

    async def _heartbeat_compaction_operation_claim(
        self,
        *,
        session: Session,
        request: CompactSessionRequest,
        operation_id: str,
        attempt_id: str,
        claim_expires_at: datetime,
        stop: asyncio.Event,
        state: _SessionOperationClaimHeartbeatState,
    ) -> None:
        sleep_seconds = _SESSION_OPERATION_CLAIM_HEARTBEAT_INTERVAL_SECONDS
        while not stop.is_set():
            remaining_claim_seconds = state.remaining_claim_seconds()
            if remaining_claim_seconds <= 0:
                raise SessionCompactionAttemptSuperseded(
                    "Session compaction operation claim expired before renewal."
                )
            try:
                await asyncio.wait_for(
                    stop.wait(),
                    timeout=min(sleep_seconds, remaining_claim_seconds),
                )
            except TimeoutError:
                confirmed_claim_expires_at = state.confirmed_claim_expires_at
                if (
                    confirmed_claim_expires_at is not None
                    and confirmed_claim_expires_at > claim_expires_at
                ):
                    claim_expires_at = confirmed_claim_expires_at
                try:
                    claim_expires_at = (
                        await self._reconcile_compaction_operation_claim_before_deadline(
                            session_id=session.id,
                            idempotency_key=request.idempotency_key,
                            operation_id=operation_id,
                            attempt_id=attempt_id,
                            state=state,
                        )
                    )
                except SessionCompactionAttemptSuperseded:
                    raise
                except Exception:
                    # The renewal transaction below revalidates the exact claim
                    # against store time. A failed read cannot make a worker
                    # clock authoritative for lease ownership.
                    pass
                remaining_seconds = state.remaining_claim_seconds()
                if remaining_seconds <= 0:
                    raise SessionCompactionAttemptSuperseded(
                        "Session compaction operation claim expired before renewal."
                    ) from None
                renewal_started_monotonic = time.monotonic()
                renewal_task = asyncio.create_task(
                    self._renew_compaction_operation_claim(
                        session=session,
                        request=request,
                        operation_id=operation_id,
                        attempt_id=attempt_id,
                        claim_expires_at=claim_expires_at,
                        commit_started=(renewal_commit_started := threading.Event()),
                    )
                )
                state.pending_store_task = renewal_task
                renewal_task.add_done_callback(state.observe_store_task)
                try:
                    done, _pending = await asyncio.wait(
                        {renewal_task},
                        timeout=remaining_seconds,
                    )
                    if renewal_task not in done:
                        if not renewal_commit_started.is_set():
                            renewal_task.cancel()
                        confirmed_claim_expires_at = state.confirmed_claim_expires_at
                        if (
                            confirmed_claim_expires_at is not None
                            and confirmed_claim_expires_at > claim_expires_at
                        ):
                            claim_expires_at = (
                                await self._reconcile_compaction_operation_claim_before_deadline(
                                    session_id=session.id,
                                    idempotency_key=request.idempotency_key,
                                    operation_id=operation_id,
                                    attempt_id=attempt_id,
                                    state=state,
                                )
                            )
                            sleep_seconds = _SESSION_OPERATION_CLAIM_HEARTBEAT_INTERVAL_SECONDS
                            continue
                        raise SessionCompactionAttemptSuperseded(
                            "Session compaction operation claim renewal was not confirmed "
                            "before its lease deadline."
                        )
                    claim_expires_at = renewal_task.result()
                    state.confirm_claim(
                        claim_expires_at,
                        claim_deadline_monotonic=(
                            renewal_started_monotonic
                            + _SESSION_OPERATION_CLAIM_LEASE.total_seconds()
                        ),
                    )
                except asyncio.CancelledError as cancellation:
                    renewal_failure: BaseException | None = None
                    if renewal_task.done():
                        renewal_failure = _completed_session_operation_child_failure(
                            renewal_task,
                            operation="Session compaction claim renewal",
                        )
                    elif not renewal_commit_started.is_set():
                        renewal_task.cancel()
                    if renewal_failure is not None:
                        cancellation.add_note(
                            "Session compaction claim renewal also failed during "
                            f"cancellation: {type(renewal_failure).__name__}: "
                            f"{renewal_failure}"
                        )
                        raise cancellation from renewal_failure
                    raise
                except SessionCompactionAttemptSuperseded:
                    raise
                except Exception as exc:
                    try:
                        durable_claim_expires_at = (
                            await self._reconcile_compaction_operation_claim_before_deadline(
                                session_id=session.id,
                                idempotency_key=request.idempotency_key,
                                operation_id=operation_id,
                                attempt_id=attempt_id,
                                state=state,
                                candidate_claim_deadline_monotonic=(
                                    renewal_started_monotonic
                                    + _SESSION_OPERATION_CLAIM_LEASE.total_seconds()
                                ),
                            )
                        )
                    except SessionCompactionAttemptSuperseded as ownership_failure:
                        raise ownership_failure from exc
                    except Exception as reconciliation_failure:
                        exc.add_note(
                            "Session compaction claim renewal reconciliation also failed: "
                            f"{type(reconciliation_failure).__name__}: "
                            f"{reconciliation_failure}"
                        )
                    else:
                        if durable_claim_expires_at > claim_expires_at:
                            claim_expires_at = durable_claim_expires_at
                            sleep_seconds = _SESSION_OPERATION_CLAIM_HEARTBEAT_INTERVAL_SECONDS
                            continue
                    sleep_seconds = _SESSION_OPERATION_CLAIM_HEARTBEAT_RETRY_SECONDS
                    continue
                sleep_seconds = _SESSION_OPERATION_CLAIM_HEARTBEAT_INTERVAL_SECONDS

    async def _persist_compaction_attempt_events(
        self,
        *,
        session: Session,
        request: CompactSessionRequest,
        operation_id: str,
        attempt_id: str,
        events: list[Event],
        heartbeat_state: _SessionOperationClaimHeartbeatState,
        persisted_event_ids: set[str],
        event_inventory: dict[str, Event],
        unresolved_store_tasks: set[asyncio.Task[Any]],
        cost_bearing_store_tasks: set[asyncio.Task[Any]] | None = None,
    ) -> None:
        if not events:
            return
        events[:] = self._event_writer.prepare_many(attribute_events_to_current_interaction(events))
        event_inventory.update((event.id, event.model_copy(deep=True)) for event in events)
        publication_claim_expires_at: datetime | None = None
        renewed_claim_expires_at: datetime | None = None

        def capture_publication_claim_expiry(expires_at: datetime) -> None:
            nonlocal publication_claim_expires_at
            publication_claim_expires_at = expires_at

        def capture_renewed_claim_expiry(expires_at: datetime) -> None:
            nonlocal renewed_claim_expires_at
            if renewed_claim_expires_at is None or expires_at > renewed_claim_expires_at:
                renewed_claim_expires_at = expires_at

        def require_unexpired_publication_commit(commit_at: datetime) -> None:
            if publication_claim_expires_at is None:
                raise AssertionError(
                    "Session compaction event publication did not capture its claim."
                )
            if publication_claim_expires_at <= commit_at:
                raise SessionCompactionAttemptSuperseded(
                    "Session compaction event publication claim expired before commit."
                )

        event_ids = [event.id for event in events]

        async def publish() -> None:
            await self.session_store.publish_session_operation_guarded_with_store_time(
                session.id,
                idempotency_key=request.idempotency_key,
                operation_transform=(
                    lambda callback_session, checkpoint, persisted_record, store_now: (
                        _append_session_operation_attempt_events(
                            idempotency_key=request.idempotency_key,
                            operation_id=operation_id,
                            attempt_id=attempt_id,
                            event_ids=event_ids,
                            clock=lambda: store_now,
                            on_renew=capture_publication_claim_expiry,
                            on_renewed=capture_renewed_claim_expiry,
                        )(callback_session, checkpoint, persisted_record)
                    )
                ),
                commit_guard=lambda: None,
                commit_time_guard=require_unexpired_publication_commit,
                events=events,
                expected_statuses=_session_checkpoint_admission._RESUMABLE_SESSION_STATUSES,
                expected_run_epoch=request.expected_run_epoch,
                expected_transcript_cursor=request.expected_transcript_cursor,
            )

        publication_started_monotonic = time.monotonic()
        publication_task = asyncio.create_task(publish())
        publication_outcome = await self._await_session_operation_store_task(
            publication_task,
        )
        if publication_outcome.timed_out:
            unresolved_store_tasks.add(publication_task)
            if cost_bearing_store_tasks is not None:
                cost_bearing_store_tasks.add(publication_task)
        cancellation = publication_outcome.cancellation
        if publication_outcome.timed_out:
            publication_error: BaseException | None = TimeoutError(
                "Session compaction event publication exceeded its bounded store wait."
            )
        else:
            publication_error = publication_outcome.error
        if isinstance(publication_error, asyncio.CancelledError):
            publication_error = unexpected_child_cancellation_error(
                publication_error,
                operation="Session compaction event publication",
            )

        async def confirm_current_claim(record: dict[str, Any] | None = None) -> None:
            nonlocal renewed_claim_expires_at
            if renewed_claim_expires_at is None and record is not None:
                renewed_claim_expires_at = _session_operation_state._operation_claim_expiry(record)
            confirmed_expiry = heartbeat_state.confirmed_claim_expires_at
            if renewed_claim_expires_at is not None and (
                confirmed_expiry is None or renewed_claim_expires_at > confirmed_expiry
            ):
                heartbeat_state.confirm_claim(
                    renewed_claim_expires_at,
                    claim_deadline_monotonic=(
                        publication_started_monotonic
                        + _SESSION_OPERATION_CLAIM_LEASE.total_seconds()
                    ),
                )
                confirmed_expiry = renewed_claim_expires_at
            if confirmed_expiry is None:
                raise AssertionError(
                    "Session compaction event publication did not renew its claim."
                )

        if publication_error is not None:

            async def reconcile() -> tuple[list[bool], dict[str, Any] | None]:
                commit_states = [await self._event_writer.is_persisted(event) for event in events]
                checkpoint = await self.session_store.load_checkpoint(session.id)
                record = None
                if type(checkpoint) is dict:
                    record = _session_operation_state._session_operation_state(checkpoint)[
                        "records"
                    ].get(request.idempotency_key)
                return commit_states, record

            reconciliation_task = asyncio.create_task(reconcile())
            reconciliation_outcome = await self._await_session_operation_store_task(
                reconciliation_task,
                cancellation=cancellation,
            )
            cancellation = reconciliation_outcome.cancellation
            reconciliation_error: BaseException | None
            if reconciliation_outcome.timed_out:
                reconciliation_error = TimeoutError(
                    "Session compaction event publication reconciliation exceeded its "
                    "bounded store wait."
                )
            else:
                reconciliation_error = reconciliation_outcome.error
            if isinstance(reconciliation_error, asyncio.CancelledError):
                reconciliation_error = unexpected_child_cancellation_error(
                    reconciliation_error,
                    operation="Session compaction event publication reconciliation",
                )
            if reconciliation_error is not None:
                publication_error.add_note(
                    "Session compaction event publication reconciliation also failed: "
                    f"{type(reconciliation_error).__name__}: {reconciliation_error}"
                )
                if cancellation is not None:
                    cancellation.add_note(
                        "Session compaction event publication and reconciliation "
                        "failed during cancellation."
                    )
                    raise cancellation from publication_error
                raise publication_error from reconciliation_error
            reconciled = reconciliation_outcome.result
            if reconciled is None:
                raise AssertionError(
                    "Session compaction event publication reconciliation lost its result."
                ) from publication_error
            commit_states, record = reconciled
            record_event_ids = record.get("event_ids", []) if type(record) is dict else []
            record_has_all_events = all(event_id in record_event_ids for event_id in event_ids)
            # The guarded store commits the operation record and event batch
            # atomically. A per-event read can be stale when the physical write
            # lands between that read and the later checkpoint read, so the
            # operation record is the authoritative confirmation of the batch.
            fully_committed = record_has_all_events
            anything_committed = any(commit_states) or any(
                event_id in record_event_ids for event_id in event_ids
            )
            if anything_committed and not fully_committed:
                atomicity_error = RuntimeError(
                    "The session store violated atomic compaction event publication."
                )
                if cancellation is not None:
                    cancellation.add_note(str(atomicity_error))
                    raise cancellation from publication_error
                raise atomicity_error from publication_error
            if not fully_committed:
                if cancellation is not None:
                    raise cancellation from publication_error
                raise publication_error
            persisted_event_ids.update(event_ids)
            fan_out_error, cancellation = await self._fan_out_reconciled_session_operation_events(
                events,
                cancellation=cancellation,
            )
            if fan_out_error is not None:
                publication_error.add_note(
                    "Committed compaction event side-effect delivery also failed: "
                    f"{type(fan_out_error).__name__}: {fan_out_error}"
                )
            try:
                await confirm_current_claim(record)
            except BaseException as claim_error:
                if cancellation is not None:
                    cancellation.add_note(
                        "The compaction claim became unusable while publication "
                        "cancellation was being reconciled."
                    )
                    raise cancellation from claim_error
                raise claim_error from publication_error
            publication_error.add_note(
                "Session compaction events were durable; the operation will fail "
                "closed after the lost acknowledgement."
            )
            if cancellation is not None:
                raise cancellation from publication_error
            raise publication_error

        persisted_event_ids.update(event_ids)
        try:
            await confirm_current_claim()
        except BaseException as claim_error:
            authoritative_cancellation = cancellation
            (
                fan_out_error,
                later_cancellation,
            ) = await self._fan_out_reconciled_session_operation_events(
                events,
                cancellation=authoritative_cancellation,
            )
            cancellation = later_cancellation or authoritative_cancellation
            if fan_out_error is not None:
                claim_error.add_note(
                    "Committed compaction event side-effect delivery also failed "
                    "before claim rejection: "
                    f"{type(fan_out_error).__name__}: {fan_out_error}"
                )
            if cancellation is not None:
                cancellation.add_note(
                    "The compaction claim became unusable after event publication."
                )
                raise cancellation from claim_error
            raise
        if cancellation is not None:
            authoritative_cancellation = cancellation
            (
                fan_out_error,
                later_cancellation,
            ) = await self._fan_out_reconciled_session_operation_events(
                events,
                cancellation=authoritative_cancellation,
            )
            cancellation = later_cancellation or authoritative_cancellation
            if fan_out_error is not None:
                cancellation.add_note(
                    "Committed compaction event side-effect delivery also failed "
                    f"during cancellation: {type(fan_out_error).__name__}: "
                    f"{fan_out_error}"
                )
            raise cancellation

    async def _reserve_compaction_budget(
        self,
        *,
        session: Session,
        registered_agent: runtime_records.RegisteredAgentState,
        environment_name: str | None,
        budget_limits: tuple[BudgetLimit, ...],
        provider_name: str | None,
        model: str | None,
        model_attempt_identity: ModelAttemptIdentity,
        request: CompactSessionRequest,
        operation_id: str,
        attempt_id: str,
        compactor: str,
        reservations: list[BudgetStepReservation],
        events: list[Event],
        reservation_identity_guard: BudgetReservationIdentityGuard,
        billing_identity: BillingIdentity | None = None,
        execution_profile_fingerprint: str | None = None,
    ) -> BudgetReservationResult | None:
        model_attempt_identity = copy_model_attempt_identity(model_attempt_identity)
        setup = await self._run_limit_controller.reserve_operation_budgets(
            budget_limits=budget_limits,
            session_id=session.id,
            session_instance_id=session.instance_id,
            agent_name=registered_agent.spec.name,
            provider_name=provider_name,
            model=model,
            model_attempt_identity=model_attempt_identity,
            environment_name=environment_name,
            settlement_event_payload=self._event_writer.prepare_budget_settlement_template(
                Event(
                    type=EventType.BUDGET_RECONCILED,
                    session_id=session.id,
                    agent_name=registered_agent.spec.name,
                    environment_name=environment_name,
                    payload=_application_compaction_causal_payload(
                        request=request,
                        operation_id=operation_id,
                        attempt_id=attempt_id,
                        source_cursor=request.expected_transcript_cursor,
                        compactor=compactor,
                    ),
                )
            ).payload,
            billing_identity=billing_identity,
            execution_profile_fingerprint=execution_profile_fingerprint,
            reservation_identity_guard=reservation_identity_guard,
            rejection_release_reason="compaction budget reservation failed",
            accepted_record_error="Accepted compaction budget reservation has no record.",
            reservation_event_factory=lambda result: _application_compaction_ledger_event(
                event_type=(
                    EventType.BUDGET_RESERVED
                    if result.accepted
                    else EventType.BUDGET_RESERVATION_FAILED
                ),
                payload=budget_reservation_payload(result),
                request=request,
                operation_id=operation_id,
                attempt_id=attempt_id,
                session=session,
                registered_agent=registered_agent,
                environment_name=environment_name,
                compactor=compactor,
            ),
        )
        reservations.extend(setup.reservations)
        events.extend(setup.events)
        events.extend(
            [
                await self._run_limit_controller.budget_settlement_event(reconciliation)
                for reconciliation in setup.releases
            ]
        )
        if setup.error is not None:
            raise setup.error
        return setup.failure

    async def _reconcile_compaction_budget_reservations(
        self,
        reservations: list[BudgetStepReservation],
        *,
        model_completed_events: list[Event],
        session: Session,
        registered_agent: runtime_records.RegisteredAgentState,
        environment_name: str | None,
        request: CompactSessionRequest,
        operation_id: str,
        attempt_id: str,
        compactor: str,
    ) -> AsyncGenerator[Event, None]:
        async for reconciliation in self._run_limit_controller.reconcile_operation_reservations(
            reservations,
            model_completed_events=model_completed_events,
            completed_reason="compaction model completed",
            missing_usage_reason=(
                "compaction completed without priced usage; charged reserved amount"
            ),
        ):
            yield await self._run_limit_controller.budget_settlement_event(reconciliation)

    async def _reconcile_uncertain_compaction_budget_reservations(
        self,
        reservations: list[BudgetStepReservation],
        *,
        session: Session,
        registered_agent: runtime_records.RegisteredAgentState,
        environment_name: str | None,
        request: CompactSessionRequest,
        operation_id: str,
        attempt_id: str,
        compactor: str,
    ) -> AsyncGenerator[Event, None]:
        async for (
            reconciliation
        ) in self._run_limit_controller.reconcile_uncertain_operation_reservations(
            reservations,
            reason="compaction reservation lease lost; charged reserved amount",
        ):
            yield await self._run_limit_controller.budget_settlement_event(reconciliation)

    async def _release_compaction_budget_reservations(
        self,
        reservations: list[BudgetStepReservation],
        *,
        session: Session,
        registered_agent: runtime_records.RegisteredAgentState,
        environment_name: str | None,
        request: CompactSessionRequest,
        operation_id: str,
        attempt_id: str,
        compactor: str,
        reason: str,
    ) -> AsyncGenerator[Event, None]:
        async for reconciliation in self._run_limit_controller.release_operation_reservations(
            reservations,
            reason=reason,
        ):
            yield await self._run_limit_controller.budget_settlement_event(reconciliation)

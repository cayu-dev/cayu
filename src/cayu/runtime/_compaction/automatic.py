"""Automatic compaction dispatch, durable evidence and context publication."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field, replace
from typing import Any, cast
from uuid import uuid4

from cayu._exception_groups import (
    add_exception_note_safely,
)
from cayu._task_wait import (
    _consume_detached_task_outcome,
    await_shielded_task_outcome,
    consume_pending_task_cancellation,
    unexpected_child_cancellation_error,
    wait_until_idle,
)
from cayu._validation import (
    copy_durable_json_object,
    require_clean_nonblank,
    require_durable_clean_nonblank,
)
from cayu.budgets.base import (
    BudgetLimit,
    BudgetPolicy,
    BudgetReservationResult,
    budget_limits_for_session,
    has_deferred_contextual_price,
)
from cayu.budgets.billing import BillingIdentity, resolved_billing_identity
from cayu.budgets.usage import (
    ModelCompletionPurpose,
)
from cayu.context.base import (
    _COMPACTION_ATTEMPT_ID_KEY,
    CompactionRequest,
    CompactionResult,
    ContextBuildError,
    ContextCompactionTelemetry,
    ContextCompactor,
    ContextRecallTelemetry,
    _attach_automatic_compaction_failure_disposition,
    _automatic_compaction_dispatch_runner_scope,
    _AutomaticCompactionDispatchDisposition,
    _AutomaticCompactionFailureDisposition,
    _AutomaticCompactionFailureReason,
    _AutomaticCompactionLifecyclePhase,
    _AutomaticCompactionRecoveryAction,
    _compaction_completion_publisher_scope,
    _compaction_environment_admission_scope,
    _compaction_model_attempt_identity_scope,
    automatic_compaction_failure_disposition_payload,
    context_build_termination_checkpoint_error,
    context_build_termination_compaction_telemetry,
    sanitize_context_build_error_checkpoint,
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
)
from cayu.providers.base import (
    ModelProvider,
    ModelRequest,
    ModelStreamDeadlineError,
    UsageDialect,
    _detach_model_request,
)
from cayu.providers.deadlines import (
    ProviderStreamDeadlineAdmission,
    bind_provider_deadline_admission,
    reset_provider_deadline_admission,
)
from cayu.runtime import _context_events
from cayu.runtime import _runtime_records as runtime_records
from cayu.runtime._compaction import identity as _compaction_identity
from cayu.runtime._event_writer import RuntimeEventWriter
from cayu.runtime._live_model_attempt import (
    _deadline_with_runtime_recovery_authority,
    _model_request_fingerprint,
)
from cayu.runtime._model_completion_contracts import (
    ModelCompletionDispatchNotAuthorized,
    ModelCompletionRecoveryContext,
    ModelCompletionRecoveryContextFactory,
)
from cayu.runtime._model_completion_delivery import (
    _combine_authoritative_model_failure,
)
from cayu.runtime._model_errors import (
    model_provider_error_from_payload,
)
from cayu.runtime._model_event_authority import _event_with_model_identity_authority
from cayu.runtime._model_execution_selection import (
    ModelExecutionSelection,
    ModelFailoverTransition,
)
from cayu.runtime._model_stage_abandonment import abandon_pre_dispatch_model_stage
from cayu.runtime._run_limits import (
    BudgetedOperationFailed,
    BudgetedOperationRejected,
    BudgetedOperationSucceeded,
    BudgetEvaluation,
    BudgetReservationIdentityGuard,
    BudgetStepReservation,
    LimitEvaluation,
    RunLimitController,
    RunLimitGate,
)
from cayu.sessions._model_failover import (
    MODEL_FAILOVER_CHECKPOINT_KEY,
)
from cayu.sessions.base import (
    ModelCompletionStage,
    ModelCompletionStageRequest,
    RuntimePublicationRequest,
    SessionStore,
    _current_session_interaction_id,
    runtime_publication_checkpoint_mutation,
)
from cayu.sessions.event_queries import EventOrder, EventQuery
from cayu.sessions.records import CheckpointTransform, Session, SessionStatus
from cayu.vaults.redaction import SecretRedactor


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


_CONTEXT_TERMINATION_PERSIST_TIMEOUT_S = 5.0

_COMPACTION_UNREPRESENTED_CALLS_KEY = "compaction_model_calls_unrepresented"

_CONTEXT_EVENT_STORE_WAIT_TIMEOUT_S = 5.0

_AUTOMATIC_COMPACTION_PREDISPATCH_EVENT_STORE_WAIT_TIMEOUT_S = 60.0

_CONTEXT_EVENT_STORE_WAIT_AFTER_CANCELLATION_TIMEOUT_S = 5.0

_AUTOMATIC_COMPACTION_PREDISPATCH_PUBLICATION_ATTEMPTS = 3


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


class AutomaticCompaction:
    """Own automatic compaction services and writes that outlive an attempt."""

    def __init__(
        self,
        *,
        session_store: SessionStore,
        event_writer: RuntimeEventWriter,
        run_limit_controller: RunLimitController,
        request_footprint: RequestFootprintConfig,
        secret_redactor: SecretRedactor,
        checkpoint_transform: Callable[[dict[str, Any]], CheckpointTransform],
    ) -> None:
        self._session_store = session_store
        self._event_writer = event_writer
        self._run_limit_controller = run_limit_controller
        self._request_footprint = copy_request_footprint_config(request_footprint)
        self._secret_redactor = secret_redactor
        self._checkpoint_transform = checkpoint_transform
        self._detached_tasks: set[asyncio.Task[Any]] = set()

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

    @property
    def detached_writes_pending(self) -> bool:
        """Whether a kept store write still runs."""

        return any(not task.done() for task in self._detached_tasks)

    async def drain_detached_writes(self, *, timeout_s: float) -> bool:
        """Wait up to ``timeout_s`` for detached store writes, without cancelling them."""

        return await wait_until_idle(self._running_detached_writes, timeout_s=timeout_s)


class AutomaticCompactionRun:
    """Publish a context outcome and own every compactor dispatch within it."""

    def __init__(
        self,
        owner: AutomaticCompaction,
        *,
        session: Session,
        registered_agent: runtime_records.RegisteredAgentState,
        environment_name: str | None,
        execution_profile: ExecutionProfileIdentity | None,
        model_execution_selection: ModelExecutionSelection | None,
        model_completion_recovery_context_factory: ModelCompletionRecoveryContextFactory,
        budget_policy: BudgetPolicy | None,
        request_budget_limits: tuple[BudgetLimit, ...],
        limit_gate: RunLimitGate,
        reservation_identity_guard: BudgetReservationIdentityGuard,
        validate_live_model_semantics: Callable[[], None],
        refresh_live_model_semantics: Callable[[], Awaitable[None]],
        model_step_identity: ModelStepIdentity,
        source_transcript_cursor: int,
        step: int,
        allow_borrowed_stage: bool,
        fallback: ModelFailoverTransition | None = None,
    ) -> None:
        self._owner = owner
        self._session = session
        self._registered_agent = registered_agent
        self._environment_name = environment_name
        self._execution_profile = execution_profile
        self._model_execution_selection = model_execution_selection
        self._model_completion_recovery_context_factory = model_completion_recovery_context_factory
        self._budget_policy = budget_policy
        self._request_budget_limits = request_budget_limits
        self._limit_gate = limit_gate
        self._reservation_identity_guard = reservation_identity_guard
        self._validate_live_model_semantics = validate_live_model_semantics
        self._refresh_live_model_semantics = refresh_live_model_semantics
        self._model_step_identity = copy_model_step_identity(model_step_identity)
        self._source_transcript_cursor = source_transcript_cursor
        self._step = step
        self._allow_borrowed_stage = allow_borrowed_stage
        self._fallback = fallback
        self.events: list[Event] = []
        self._published_attempt_ids: set[str] = set()
        self._start_events: list[Event] = []
        self._completion_events: dict[str, Event] = {}
        self._identity_ledger = _compaction_identity._CompactionExecutionIdentityLedger(
            model_step_identity
        )
        self._dispatch_authorities: dict[str, _AutomaticCompactionDispatchAuthority] = {}
        self._settled_attempt_ids: set[str] = set()
        self._lifecycle = _AutomaticCompactionLifecycle()

    async def publish_recall(self, telemetry: ContextRecallTelemetry) -> None:
        event = _context_events._context_recall_telemetry_event(
            telemetry=telemetry,
            session=self._session,
            registered_agent=self._registered_agent,
            environment_name=self._environment_name,
            model_step_identity=self._model_step_identity,
            execution_profile=self._execution_profile,
        )
        self.events.append(await self._owner._event_writer.emit(event))

    async def run(
        self,
        compactor: ContextCompactor,
        compaction_request: CompactionRequest,
        compaction_started: ContextCompactionTelemetry,
        execute: Callable[[], Awaitable[CompactionResult]],
        completed_payloads: Callable[[], list[dict[str, Any]]],
    ) -> CompactionResult:
        if not self._published_attempt_ids:
            await self._require_no_uncheckpointed_automatic_compaction(
                parent_model_step_identity=self._model_step_identity,
            )

        await self._persist_automatic_compaction_start_with_retry(
            compaction_started,
            published_events=self.events,
            start_events=self._start_events,
            model_step_identity=self._model_step_identity,
            lifecycle=self._lifecycle,
        )

        async def publish_completions(payloads: list[dict[str, Any]]) -> None:
            try:
                await self._persist_automatic_compaction_completions(
                    self._identity_ledger.identify_payloads(payloads),
                    published_attempt_ids=self._published_attempt_ids,
                    published_events=self.events,
                    completion_events=self._completion_events,
                    dispatch_authorities=self._dispatch_authorities,
                    settled_attempt_ids=self._settled_attempt_ids,
                )
            except BaseException as error:
                self._lifecycle.attach_failure(
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
                budget_events=self.events,
                step=self._step,
                model_step_identity=self._model_step_identity,
                compaction_identity_ledger=self._identity_ledger,
                dispatch_authorities=self._dispatch_authorities,
                settled_attempt_ids=self._settled_attempt_ids,
                source_transcript_cursor=self._source_transcript_cursor,
                allow_borrowed_stage=self._allow_borrowed_stage,
                fallback=self._fallback,
                lifecycle=self._lifecycle,
            )

        with _compaction_completion_publisher_scope(publish_completions):
            if self._allow_borrowed_stage:
                return await run()
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
            return [await self._owner._event_writer.is_persisted(event) for event in events]

        reconciliation_task = asyncio.create_task(reconcile())
        outcome = await await_shielded_task_outcome(
            reconciliation_task,
            cancellation=cancellation,
            timeout_s=_CONTEXT_EVENT_STORE_WAIT_TIMEOUT_S,
            timeout_after_cancellation_s=(_CONTEXT_EVENT_STORE_WAIT_AFTER_CANCELLATION_TIMEOUT_S),
        )
        if outcome.timed_out:
            reconciliation_task.cancel()
            self._owner._retain_detached_task(reconciliation_task)
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

        fan_out_task = asyncio.create_task(self._owner._event_writer.fan_out_persisted(events))
        outcome = await await_shielded_task_outcome(
            fan_out_task,
            cancellation=cancellation,
            timeout_s=_CONTEXT_EVENT_STORE_WAIT_TIMEOUT_S,
            timeout_after_cancellation_s=(_CONTEXT_EVENT_STORE_WAIT_AFTER_CANCELLATION_TIMEOUT_S),
        )
        if outcome.timed_out:
            fan_out_task.cancel()
            self._owner._retain_detached_task(fan_out_task)
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
            records = await self._owner._session_store.query_events(
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

        prepared = self._owner._event_writer.prepare(event)
        last_error: BaseException | None = None
        for attempt in range(1, _AUTOMATIC_COMPACTION_PREDISPATCH_PUBLICATION_ATTEMPTS + 1):
            persistence_task = asyncio.create_task(
                self._owner._event_writer.persist_exact_replay(prepared)
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
                self._owner._retain_detached_task(persistence_task)
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
                    self._owner._event_writer.fan_out_persisted([persisted])
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
                    self._owner._retain_detached_task(fan_out_task)
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
        failure_event = _context_events._context_compaction_telemetry_event(
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
            return await self._owner._event_writer.emit(failure_event)
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
            event = _context_events._context_compaction_telemetry_event(
                telemetry=telemetry,
                session=self._session,
                registered_agent=self._registered_agent,
                environment_name=self._environment_name,
                execution_identity=model_step_identity,
                execution_profile=self._execution_profile,
            )
            start_events.append(event.model_copy(deep=True))
        persistence_task = asyncio.create_task(
            self._owner._event_writer.emit_many(self._session.id, [event])
        )
        outcome = await await_shielded_task_outcome(
            persistence_task,
            timeout_s=_AUTOMATIC_COMPACTION_PREDISPATCH_EVENT_STORE_WAIT_TIMEOUT_S,
            timeout_after_cancellation_s=(_CONTEXT_EVENT_STORE_WAIT_AFTER_CANCELLATION_TIMEOUT_S),
        )
        cancellation = outcome.cancellation
        if outcome.timed_out:
            persistence_task.cancel()
            self._owner._retain_detached_task(persistence_task)
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
        active = await self._owner._session_store.load_active_model_completion_stage(
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
            dispatch = await self._owner._session_store.load_model_completion_stage_dispatch(
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
        prepared = await self._owner._session_store.prepare_model_completion_stage(
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
            await self._owner._session_store.mark_model_completion_stage_dispatched(
                self._session.id,
                stage=authority.stage,
            )
            return None
        except BaseException as dispatch_failure:
            try:
                dispatch = await self._owner._session_store.load_model_completion_stage_dispatch(
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
        prepared_events = self._owner._event_writer.prepare_many(events)
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
            await self._owner._session_store.complete_model_completion_stage(
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
        active = await self._owner._session_store.load_active_model_completion_stage(
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
            await self._owner._session_store.promote_model_completion_stage(
                self._session.id,
                stage_id=authority.stage.stage_id,
                expected_run_epoch=self._session.run_epoch,
            )
        except BaseException as promotion_failure:
            reconciled = await self._owner._session_store.load_active_model_completion_stage(
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
                event = _context_events._context_compaction_telemetry_event(
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
                        event_durable = await self._owner._event_writer.is_persisted(event)
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
            self._owner._event_writer.persist_many(self._session.id, events)
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
            return await self._owner._event_writer.emit_many(self._session.id, events)
        except BaseException as publication_error:
            if (
                not isinstance(publication_error, ValueError)
                or compaction_start_event is None
                or all(event.id != compaction_start_event.id for event in events)
            ):
                raise
            try:
                start_durable = await self._owner._event_writer.is_persisted(compaction_start_event)
                remaining = [event for event in events if event.id != compaction_start_event.id]
                remaining_states = [
                    await self._owner._event_writer.is_persisted(event) for event in remaining
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
                    persisted_remaining = await self._owner._event_writer.persist_many(
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
            await self._owner._event_writer.fan_out_persisted(reconciled)
            return [
                next(event for event in reconciled if event.id == requested.id).model_copy(
                    deep=True
                )
                for requested in events
            ]

    async def _persist_context_events(
        self,
        *,
        compaction_telemetry: list[ContextCompactionTelemetry],
        recall_telemetry: list[ContextRecallTelemetry],
        checkpoint_update: dict[str, Any] | None,
        checkpoint_event_payload: dict[str, Any] | None,
        checkpoint_invariant_cause: BaseException | None = None,
        shield_persistence: bool = True,
        compaction_started_published: bool | None = None,
    ) -> tuple[list[Event], BaseException | None]:
        """Persist one context outcome completely before exposing its first event."""
        model_step_identity = self._model_step_identity
        compaction_identity_ledger = self._identity_ledger
        published_compaction_attempt_ids = self._published_attempt_ids
        compaction_completion_events = self._completion_events
        compaction_start_event = self._start_events[0] if self._start_events else None
        if compaction_started_published is None:
            compaction_started_published = any(
                event.type == EventType.CONTEXT_COMPACTION_STARTED for event in self.events
            )

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
            _context_events._context_recall_telemetry_event(
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
                event = _context_events._context_compaction_telemetry_event(
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
            _context_events._context_recall_telemetry_event(
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
                    await self._owner._event_writer.fan_out_persisted(reconciled_start_events)
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
            atomic_events = self._owner._event_writer.prepare_many(
                [*prepared_events, checkpoint_event]
            )
            checkpoint_transform = self._owner._checkpoint_transform(checkpoint_update)
            try:
                await self._owner._session_store.publish_checkpoint_and_events(
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
                        await self._owner._event_writer.is_persisted(event)
                        for event in atomic_events
                    ]
                    events_committed = all(event_commit_states)
                    durable_checkpoint = await self._owner._session_store.load_checkpoint(
                        self._session.id
                    )
                    durable_session = await self._owner._session_store.load(self._session.id)
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
            await self._owner._event_writer.fan_out_persisted(
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

    async def context_failed(
        self,
        error: ContextBuildError,
    ) -> tuple[list[Event], BaseException | None]:
        return await self._persist_context_events(
            compaction_telemetry=list(error.compaction_telemetry),
            recall_telemetry=list(error.recall_telemetry),
            checkpoint_update=error.checkpoint,
            checkpoint_event_payload=error.checkpoint_event_payload,
            checkpoint_invariant_cause=error,
        )

    async def context_terminated(
        self,
        error: BaseException,
        *,
        cancellation_requests_before_build: int,
    ) -> None:
        """Persist completed compaction evidence without replacing a fatal signal."""
        model_step_identity = self._model_step_identity
        compaction_identity_ledger = self._identity_ledger
        published_compaction_attempt_ids = self._published_attempt_ids
        compaction_completion_events = self._completion_events
        compaction_start_event = self._start_events[0] if self._start_events else None
        compaction_started_published = any(
            event.type == EventType.CONTEXT_COMPACTION_STARTED for event in self.events
        )

        model_step_identity = copy_model_step_identity(model_step_identity)
        telemetry = context_build_termination_compaction_telemetry(error)
        progress_error = context_build_termination_checkpoint_error(error)
        if progress_error is not None:
            sanitize_context_build_error_checkpoint(
                progress_error, redactor=self._owner._secret_redactor
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
                    compaction_telemetry=list(telemetry),
                    recall_telemetry=[],
                    checkpoint_update=progress_error.checkpoint,
                    checkpoint_event_payload=progress_error.checkpoint_event_payload,
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
                    _context_events._context_compaction_telemetry_event(
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
            self._owner._retain_detached_task(task)
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

    async def context_completed(
        self,
        *,
        checkpoint_update: dict[str, Any] | None,
        checkpoint_event_payload: dict[str, Any] | None,
        compaction_telemetry: list[ContextCompactionTelemetry],
        recall_telemetry: list[ContextRecallTelemetry],
    ) -> tuple[list[Event], BaseException | None]:
        events, failure = await self._persist_context_events(
            compaction_telemetry=compaction_telemetry,
            recall_telemetry=recall_telemetry,
            checkpoint_update=checkpoint_update,
            checkpoint_event_payload=checkpoint_event_payload,
        )
        if failure is not None and any(
            telemetry.event_type == EventType.CONTEXT_COMPACTION_COMPLETED
            for telemetry in compaction_telemetry
        ):
            self._lifecycle.attach_failure(
                failure,
                phase=_AutomaticCompactionLifecyclePhase.CHECKPOINT_INSTALLATION,
                reason=_AutomaticCompactionFailureReason.CHECKPOINT_FAILED,
                retryable=False,
                recovery_action=_AutomaticCompactionRecoveryAction.FAIL_CLOSED,
            )
            failure_event = await self._persist_automatic_compaction_checkpoint_failure(
                failure,
                model_step_identity=self._model_step_identity,
            )
            if failure_event is not None:
                events.append(failure_event)
        return events, failure

    async def _run_automatic_compaction_with_budget(
        self,
        *,
        compactor: ContextCompactor,
        compaction_request: CompactionRequest,
        execute: Callable[[], Awaitable[CompactionResult]],
        completed_payloads: Callable[[], list[dict[str, Any]]],
        budget_events: list[Event],
        step: int,
        model_step_identity: ModelStepIdentity,
        compaction_identity_ledger: _compaction_identity._CompactionExecutionIdentityLedger,
        dispatch_authorities: dict[str, _AutomaticCompactionDispatchAuthority],
        settled_attempt_ids: set[str],
        source_transcript_cursor: int,
        allow_borrowed_stage: bool,
        lifecycle: _AutomaticCompactionLifecycle,
        fallback: ModelFailoverTransition | None = None,
    ) -> CompactionResult:
        model_step_identity = copy_model_step_identity(model_step_identity)
        if compaction_identity_ledger.model_step_identity != model_step_identity:
            raise ValueError("Compaction identity ledger belongs to a different model step.")
        if type(lifecycle) is not _AutomaticCompactionLifecycle:
            raise TypeError("lifecycle must be an _AutomaticCompactionLifecycle.")
        controller = self._owner._run_limit_controller
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
            await abandon_pre_dispatch_model_stage(
                session_store=self._owner._session_store,
                run_limit_controller=self._owner._run_limit_controller,
                session=self._session,
                stage=authority.stage,
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
            if self._owner._request_footprint.enabled:
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
                    config=self._owner._request_footprint,
                    execution_profile_fingerprint=(
                        None
                        if self._execution_profile is None
                        else self._execution_profile.fingerprint
                    ),
                )
                budget_events.append(
                    await self._persist_automatic_compaction_predispatch_event(
                        event_with_execution_profile_authority(
                            _context_events._context_observation_event(
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

        # Each automatic compactor dispatch owns an independently settleable
        # model step. Sharing the assistant step would record its winner early.
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
            model_attempt_identity = compaction_identity_ledger.begin_dispatch(
                new_model_step_identity().new_attempt()
            )
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
            if self._owner._request_footprint.enabled:
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
            has_accounting_limits or self._owner._request_footprint.enabled
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
            model_attempt_identity = compaction_identity_ledger.begin_dispatch(
                new_model_step_identity().new_attempt()
            )
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

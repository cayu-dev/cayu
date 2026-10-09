"""Application work-attempt admission, claim ownership, recovery and proposals."""

from __future__ import annotations

import os
from collections.abc import AsyncGenerator, Awaitable, Callable
from datetime import datetime
from hashlib import sha256
from typing import Any, Literal, Protocol, cast
from uuid import uuid4

from cayu._validation import canonical_durable_json_bytes
from cayu.events import Event
from cayu.runtime._delegated_event_stream import _close_delegated_event_stream
from cayu.runtime._invocation_lifecycle import prepare_rebind_invocation_command
from cayu.runtime._session_request_boundary import _validate_resume_request, _validate_run_request
from cayu.runtime._task_store_operation_boundary import (
    TaskStoreOperationOutcome,
    capture_sensitive_result_validation,
    capture_sensitive_validation,
    capture_task_store_operation,
    raise_task_store_operation_failure,
    task_store_work_attempt_admission_capability_is_complete,
)
from cayu.runtime._work_attempt_invocation import (
    WorkAttemptRecoveryOwnership,
    _acknowledged_work_attempt_recovery,
    _WorkAttemptRecoveryAlreadyActive,
    _WorkAttemptRuntimeAuthority,
)
from cayu.runtime._work_attempt_session_mutation import (
    capture_work_attempt_checkpoint_result,
    capture_work_attempt_deferred_input_result,
    capture_work_attempt_session_result,
    read_work_attempt_session_store,
    settle_work_attempt_session_mutation,
)
from cayu.runtime.work_attempt_source import WorkAttemptSourceRequest
from cayu.sessions._execution_profile_checkpoint import (
    active_invocation_execution_profile_from_checkpoint,
    active_invocation_execution_profile_is_released,
    active_invocation_execution_profile_matches_session_epoch,
    checkpoint_with_active_invocation_execution_profile,
    execution_profile_from_session_metadata,
)
from cayu.sessions._invocation_lifecycle import (
    InvocationMutationResult,
    invocation_lifecycle_receipt_history_present,
)
from cayu.sessions._invocation_terminal_decision import (
    invocation_terminal_decision_from_checkpoint,
    settled_invocation_terminal_decision_from_checkpoint,
)
from cayu.sessions.base import (
    ResumeRequest,
    RunRequest,
    SessionRunFenced,
    SessionStore,
    _activate_session_interaction,
    _activate_session_run_fence,
    _deactivate_session_interaction,
    _deactivate_session_run_fence,
    _initial_transcript_pending_interaction_id,
)
from cayu.sessions.invocation import SessionInvocationBinding
from cayu.sessions.records import Session, SessionStatus, copy_session
from cayu.sessions.recovery import (
    IncompleteSessionRecoveryAction,
    IncompleteSessionRecoveryRequest,
    IncompleteSessionRecoveryResult,
)
from cayu.tasks.admission import (
    WORK_ATTEMPT_RECOVERY_CHECKPOINT_KEY,
    WORK_ATTEMPT_RENEWABLE_STATES,
    AdmittedCompletionProposalRequest,
    WorkAttemptAdmission,
    WorkAttemptAdmissionState,
    WorkAttemptClaimRenewalRequest,
    WorkAttemptExecutionClaimLost,
    WorkAttemptExecutionClaimRequest,
    WorkAttemptExecutionRequest,
    WorkAttemptProposalRequest,
    WorkAttemptRecoveryActivate,
    WorkAttemptRecoveryRequest,
    WorkAttemptRecoveryRequired,
    WorkAttemptRunRequest,
    copy_work_attempt_claim_renewal_request,
    copy_work_attempt_execution_request,
    copy_work_attempt_proposal_request,
    copy_work_attempt_recovery_request,
    copy_work_attempt_run_request,
    require_admitted_completion_proposal_result,
    require_work_attempt_admission_result,
    require_work_attempt_claim_result,
    require_work_attempt_execution_claim_result,
    require_work_attempt_recovery_activation_result,
    work_attempt_recovery_session_authority,
    work_attempt_recovery_session_authority_from_checkpoint,
)
from cayu.tasks.contracts import CompletionProposal
from cayu.tasks.store import TaskStore
from cayu.vaults.redaction import SecretRedactor


class WorkAttemptEngine(Protocol):
    """Session execution and settlement used by work-attempt coordination."""

    def work_attempt_source_request_sha256(
        self,
        request: RunRequest | ResumeRequest,
        *,
        kind: Literal["initial", "continuation"],
    ) -> str: ...

    def work_attempt_source_snapshot(
        self,
        request: RunRequest | ResumeRequest,
        *,
        kind: Literal["initial", "continuation"],
        source_request_sha256: str,
    ) -> WorkAttemptSourceRequest | None: ...

    def reconstruct_preparing_work_attempt_source(
        self, admission: WorkAttemptAdmission
    ) -> RunRequest | ResumeRequest: ...

    async def admit_initial_work_attempt(
        self,
        request: RunRequest,
        *,
        authority: _WorkAttemptRuntimeAuthority,
    ) -> WorkAttemptAdmission: ...

    async def admit_continuation_work_attempt(
        self,
        request: ResumeRequest,
        *,
        authority: _WorkAttemptRuntimeAuthority,
    ) -> WorkAttemptAdmission: ...

    def execute_admitted_work_attempt(
        self,
        admission: WorkAttemptAdmission,
        *,
        lease_seconds: int,
    ) -> AsyncGenerator[Event, None]: ...

    async def recover_preparing_work_attempt(
        self, ownership: WorkAttemptRecoveryOwnership
    ) -> WorkAttemptAdmission: ...

    async def _recover_work_attempt_session(
        self,
        request: IncompleteSessionRecoveryRequest,
        *,
        interaction_id: str,
        before_mutation: Callable[[], Awaitable[None]],
        admission: WorkAttemptAdmission | None = None,
    ) -> IncompleteSessionRecoveryResult: ...

    async def has_recoverable_work_attempt_model_result(
        self, admission: WorkAttemptAdmission
    ) -> bool: ...

    async def reconcile_released_work_attempt_recovery(
        self, admission: WorkAttemptAdmission
    ) -> WorkAttemptAdmission | None: ...

    async def _fan_out_work_attempt_interaction_started(
        self,
        admission: WorkAttemptAdmission,
        *,
        lease_seconds: int,
    ) -> WorkAttemptAdmission: ...


class WorkAttemptCheckpointGuard(Protocol):
    """Validate resumability using the existing checkpoint authority rules."""

    def __call__(
        self,
        session: Session,
        checkpoint: dict[str, Any] | None,
        *,
        redactor: SecretRedactor,
        allow_pending_tool_round: bool = False,
        allowed_initial_transcript_interaction_id: str | None = None,
    ) -> None: ...


class WorkAttemptCoordinator:
    """Own local work-attempt authority and complete admission/recovery operations.

    TaskStore owns durable claims and receipts; the engine owns session execution
    and settlement. Callers apply their admission policy around these operations.
    Separate claim/recover phases let a worker maintain its acknowledged claim
    while recovery runs. Store access remains live so replacing the configured
    store cannot authorize a claim acknowledged by another store.
    """

    def __init__(
        self,
        *,
        get_task_store: Callable[[], TaskStore | None],
        session_store: SessionStore,
        engine: WorkAttemptEngine,
        redactor: SecretRedactor,
        apply_run_defaults: Callable[[RunRequest], RunRequest],
        resolve_session: Callable[[str], Awaitable[tuple[str, str | None]]],
        checkpoint_guard: WorkAttemptCheckpointGuard,
    ) -> None:
        self._get_task_store = get_task_store
        self._runtime_session_store = session_store
        self._session_engine = engine
        self._secret_redactor = redactor
        self._with_application_run_defaults = apply_run_defaults
        self._resolve_public_session_authority = resolve_session
        self._checkpoint_guard = checkpoint_guard
        self._work_attempt_execution_pid = os.getpid()
        self._work_attempt_execution_owner_id = uuid4().hex

    @property
    def task_store(self) -> TaskStore | None:
        return self._get_task_store()

    def current_execution_owner_id(self) -> str:
        pid = os.getpid()
        if pid != self._work_attempt_execution_pid:
            self._work_attempt_execution_pid = pid
            self._work_attempt_execution_owner_id = uuid4().hex
        return f"cayu-work-attempt:{pid}:{self._work_attempt_execution_owner_id}"

    async def admit(
        self,
        request: RunRequest | ResumeRequest,
        *,
        execution: WorkAttemptExecutionRequest,
    ) -> WorkAttemptAdmission:
        """Durably admit an initial or rejected/continue contracted attempt.

        Admission starts and profiles the exact interaction but deliberately
        dispatches no provider, tool, hook, or mutating environment work. The
        automatic verified-work worker consumes this authority separately.
        """

        source_request_type = type(request)
        execution_request_type = type(execution)
        source_validation = _copied_public_work_attempt_source_request(
            request,
            redactor=self._secret_redactor,
        )
        execution_validation = _copied_public_work_attempt_execution_request(
            execution,
            redactor=self._secret_redactor,
        )
        del request, execution
        source_failure = source_validation.failure
        prepared_request = source_validation.result
        execution_failure = execution_validation.failure
        stable = execution_validation.result
        del source_validation, execution_validation
        if source_failure is not None:
            del execution_failure, prepared_request, stable
            raise source_failure from None
        if execution_failure is not None:
            del prepared_request, stable
            raise execution_failure from None
        if prepared_request is None:
            del stable
            if source_request_type not in {RunRequest, ResumeRequest}:
                raise TypeError(
                    "Work-attempt admission requires a RunRequest or ResumeRequest."
                ) from None
            raise ValueError("Work-attempt source request is invalid.") from None
        if stable is None:
            del prepared_request
            if execution_request_type is not WorkAttemptExecutionRequest:
                raise TypeError(
                    "Work-attempt execution requires a WorkAttemptExecutionRequest."
                ) from None
            raise ValueError("Work-attempt execution request is invalid.") from None
        task_store = self.task_store
        if task_store is None:
            raise RuntimeError("task_store is required for work-attempt execution.")
        if not task_store_work_attempt_admission_capability_is_complete(task_store):
            raise NotImplementedError(
                f"{type(task_store).__name__} does not implement the complete "
                "work-attempt admission contract."
            )
        owner = self.current_execution_owner_id()
        if type(prepared_request) is RunRequest:
            prepared_request = self._with_application_run_defaults(prepared_request)
            if prepared_request.task_id is None:
                raise ValueError("Initial work-attempt admission requires RunRequest.task_id.")
            if prepared_request.session_id is None:
                raise ValueError(
                    "Initial work-attempt admission requires a caller-stable RunRequest.session_id."
                )
            if stable.predecessor_admission_id is not None:
                raise ValueError(
                    "Initial work-attempt admission cannot select a predecessor admission."
                )
            if stable.task_id is not None and stable.task_id != prepared_request.task_id:
                raise ValueError(
                    "Initial work-attempt task selection conflicts with RunRequest.task_id."
                )
            source_request_sha256 = self._session_engine.work_attempt_source_request_sha256(
                prepared_request,
                kind="initial",
            )
            snapshot_validation = capture_sensitive_validation(
                lambda: (
                    self._session_engine.work_attempt_source_snapshot(
                        prepared_request,
                        kind="initial",
                        source_request_sha256=source_request_sha256,
                    ),
                ),
                operation_name="Work-attempt source snapshot validation",
                redactor=self._secret_redactor,
            )
            if snapshot_validation.failure is not None:
                del prepared_request, stable
                raise_task_store_operation_failure(snapshot_validation.failure)
            if snapshot_validation.result is None:
                del prepared_request, stable
                raise ValueError(
                    "Work-attempt source snapshot is invalid or exceeds its portable limit."
                )
            authority = _WorkAttemptRuntimeAuthority(
                request=stable,
                execution_owner_id=owner,
                kind="initial",
                source_request_sha256=source_request_sha256,
                source_request=snapshot_validation.result[0],
            )
            del snapshot_validation
            admission_operation = self._session_engine.admit_initial_work_attempt(
                prepared_request,
                authority=authority,
            )
            del (
                stable,
                owner,
                prepared_request,
                source_request_sha256,
                authority,
            )
            return await admission_operation
        if type(prepared_request) is ResumeRequest:
            if stable.task_id is None or stable.predecessor_admission_id is None:
                raise ValueError(
                    "Continuation work-attempt admission requires task_id and "
                    "predecessor_admission_id."
                )
            session_resolution = await capture_task_store_operation(
                lambda session_id=prepared_request.session_id: (
                    self._resolve_public_session_authority(session_id)
                ),
                operation_name="Work-attempt continuation session resolution",
                redactor=self._secret_redactor,
            )
            if session_resolution.failure is not None:
                raise_task_store_operation_failure(session_resolution.failure)
            if session_resolution.result is None:
                raise RuntimeError(
                    "Work-attempt continuation session resolution returned no result."
                )
            session_id, _store_resolved_session_id = session_resolution.result
            del session_resolution
            prepared_request = prepared_request.model_copy(update={"session_id": session_id})
            source_request_sha256 = self._session_engine.work_attempt_source_request_sha256(
                prepared_request,
                kind="continuation",
            )
            snapshot_validation = capture_sensitive_validation(
                lambda: (
                    self._session_engine.work_attempt_source_snapshot(
                        prepared_request,
                        kind="continuation",
                        source_request_sha256=source_request_sha256,
                    ),
                ),
                operation_name="Work-attempt source snapshot validation",
                redactor=self._secret_redactor,
            )
            if snapshot_validation.failure is not None:
                del prepared_request, stable, session_id, _store_resolved_session_id
                raise_task_store_operation_failure(snapshot_validation.failure)
            if snapshot_validation.result is None:
                del prepared_request, stable, session_id, _store_resolved_session_id
                raise ValueError(
                    "Work-attempt source snapshot is invalid or exceeds its portable limit."
                )
            authority = _WorkAttemptRuntimeAuthority(
                request=stable,
                execution_owner_id=owner,
                kind="continuation",
                source_request_sha256=source_request_sha256,
                source_request=snapshot_validation.result[0],
            )
            del snapshot_validation
            admission_operation = self._session_engine.admit_continuation_work_attempt(
                prepared_request,
                authority=authority,
            )
            del (
                stable,
                owner,
                prepared_request,
                session_id,
                _store_resolved_session_id,
                source_request_sha256,
                authority,
            )
            return await admission_operation
        raise AssertionError("Validated work-attempt source request has an unknown type.")

    async def execute(self, request: WorkAttemptRunRequest) -> AsyncGenerator[Event, None]:
        """Resolve this process's exact claim and delegate governed execution.

        Kept internal until the complete verified-worker capability and
        recovery contract are exposed together. No caller admission object,
        source settings or execution-owner identity cross this entrance.
        """
        request_type = type(request)
        request_validation = capture_sensitive_validation(
            lambda value=request: copy_work_attempt_run_request(value),
            operation_name="Work-attempt run-request validation",
            redactor=self._secret_redactor,
        )
        del request
        if request_validation.failure is not None:
            raise request_validation.failure from None
        stable = request_validation.result
        if stable is None:
            if request_type is not WorkAttemptRunRequest:
                raise TypeError(
                    "Work-attempt execution requires a WorkAttemptRunRequest."
                ) from None
            raise ValueError("Work-attempt run request is invalid.")
        task_store = self.task_store
        if task_store is None:
            raise RuntimeError("task_store is required for work-attempt execution.")
        outcome = await capture_task_store_operation(
            lambda: task_store.load_work_attempt_admission(stable.admission_id),
            operation_name="Work-attempt run admission lookup",
            redactor=self._secret_redactor,
        )
        if outcome.failure is not None:
            raise_task_store_operation_failure(outcome.failure)
        validation = capture_sensitive_result_validation(
            lambda value=outcome.result: require_work_attempt_admission_result(
                value, operation_name="Work-attempt run admission lookup"
            ),
            operation_name="Work-attempt run authority validation",
            redactor=self._secret_redactor,
        )
        del outcome
        if validation.failure is not None:
            raise_task_store_operation_failure(validation.failure)
        admission = validation.result
        if admission is None:
            raise WorkAttemptRecoveryRequired("Work-attempt execution has no admission.")
        claim = admission.claim
        if (
            admission.admission_id != stable.admission_id
            or claim.claim_id != stable.claim_id
            or claim.worker_id != stable.worker_id
            or claim.generation != stable.generation
            or claim.execution_owner_id != self.current_execution_owner_id()
        ):
            raise WorkAttemptExecutionClaimLost("Work-attempt execution claim is not owned here.")
        # The delegated stream owns its execution context and restores this
        # caller's context between advances. Do not let admission's inherited
        # epoch escape the stream and fence later decision/result publication.
        # Removing local tokens is not durable release or a dispatch grant.
        _deactivate_session_interaction(admission.session_id)
        _deactivate_session_run_fence(admission.session_id)
        stream = self._session_engine.execute_admitted_work_attempt(
            admission, lease_seconds=stable.lease_seconds
        )
        async with _close_delegated_event_stream(stream) as owned_stream:
            async for event in owned_stream:
                yield event

    async def renew_claim(
        self,
        request: WorkAttemptClaimRenewalRequest,
    ) -> WorkAttemptAdmission:
        """Renew the exact live execution generation owned by this process."""

        request_type = type(request)
        request_validation = _copied_public_work_attempt_claim_renewal_request(
            request,
            redactor=self._secret_redactor,
        )
        del request
        request_failure = request_validation.failure
        stable = request_validation.result
        del request_validation
        if request_failure is not None:
            raise request_failure from None
        if stable is None:
            if request_type is not WorkAttemptClaimRenewalRequest:
                raise TypeError(
                    "Claim renewal requires a WorkAttemptClaimRenewalRequest."
                ) from None
            raise ValueError("Work-attempt claim-renewal request is invalid.") from None
        task_store = self.task_store
        if task_store is None:
            raise RuntimeError("task_store is required for work-attempt renewal.")
        if not task_store_work_attempt_admission_capability_is_complete(task_store):
            raise NotImplementedError(
                f"{type(task_store).__name__} does not implement the complete "
                "work-attempt admission contract."
            )
        claim_request = WorkAttemptExecutionClaimRequest(
            admission_id=stable.admission_id,
            claim_id=stable.claim_id,
            worker_id=stable.worker_id,
            execution_owner_id=self.current_execution_owner_id(),
            generation=stable.generation,
            lease_seconds=stable.lease_seconds,
        )
        prior_outcome = await capture_task_store_operation(
            lambda: task_store.load_work_attempt_admission(stable.admission_id),
            operation_name="Work-attempt renewal authority lookup",
            redactor=self._secret_redactor,
        )
        if prior_outcome.failure is not None:
            raise_task_store_operation_failure(prior_outcome.failure)
        if prior_outcome.result is None:
            raise KeyError(f"Work-attempt admission not found: {stable.admission_id}")
        prior_validation = capture_sensitive_result_validation(
            lambda value=prior_outcome.result: require_work_attempt_admission_result(
                value,
                operation_name="Work-attempt renewal authority lookup",
            ),
            operation_name="Work-attempt renewal authority validation",
            redactor=self._secret_redactor,
        )
        del prior_outcome
        if prior_validation.failure is not None:
            raise_task_store_operation_failure(prior_validation.failure)
        prior = prior_validation.result
        del prior_validation
        if prior is None:
            raise RuntimeError("Work-attempt renewal authority lookup returned no authority.")
        outcome = await capture_task_store_operation(
            lambda: task_store.renew_work_attempt_execution_claim(claim_request),
            operation_name="Work-attempt execution-claim renewal",
            redactor=self._secret_redactor,
            mutation_store=task_store,
            mutation_method_name="renew_work_attempt_execution_claim",
        )
        if outcome.failure is not None:
            raise_task_store_operation_failure(outcome.failure)
        validation = capture_sensitive_result_validation(
            lambda value=outcome.result, previous=prior: require_work_attempt_claim_result(
                value,
                previous,
                claim_request,
                operation_name="Work-attempt execution-claim renewal",
                allowed_states=WORK_ATTEMPT_RENEWABLE_STATES,
                allowed_previous_states=WORK_ATTEMPT_RENEWABLE_STATES,
                renewal=True,
            ),
            operation_name="Work-attempt execution-claim renewal result validation",
            redactor=self._secret_redactor,
        )
        del outcome, prior
        if validation.failure is not None:
            raise_task_store_operation_failure(validation.failure)
        renewed = validation.result
        del validation
        if renewed is None:
            raise RuntimeError("Work-attempt execution-claim renewal returned no authority.")
        current_outcome = await capture_task_store_operation(
            lambda: task_store.load_work_attempt_admission(stable.admission_id),
            operation_name="Work-attempt renewal reconciliation lookup",
            redactor=self._secret_redactor,
        )
        if current_outcome.failure is not None:
            raise_task_store_operation_failure(current_outcome.failure)
        if current_outcome.result is None:
            raise RuntimeError("Renewed work-attempt authority is no longer durable.")
        current_validation = capture_sensitive_result_validation(
            lambda value=current_outcome.result, previous=renewed: (
                require_work_attempt_claim_result(
                    value,
                    previous,
                    claim_request,
                    operation_name="Work-attempt execution-claim renewal reconciliation",
                    allowed_states=WORK_ATTEMPT_RENEWABLE_STATES,
                    allowed_previous_states=WORK_ATTEMPT_RENEWABLE_STATES,
                    renewal=True,
                )
            ),
            operation_name="Work-attempt renewal reconciliation result validation",
            redactor=self._secret_redactor,
        )
        del current_outcome, renewed
        if current_validation.failure is not None:
            raise_task_store_operation_failure(current_validation.failure)
        current = current_validation.result
        del current_validation
        if current is None:
            raise RuntimeError("Work-attempt renewal reconciliation returned no authority.")
        return current

    async def _settle_first_work_attempt_recovery_predecessor(
        self,
        *,
        recovering: WorkAttemptAdmission,
        claim_request: WorkAttemptExecutionClaimRequest,
    ) -> None:
        """Fence and settle the first crashed session owner before reactivation."""

        task_store = self.task_store
        if task_store is None:  # pragma: no cover - checked by the public owner
            raise RuntimeError("task_store is required for work-attempt recovery.")

        async def require_exact_recovery_claim() -> None:
            current_outcome = await capture_task_store_operation(
                lambda: task_store.claim_work_attempt_recovery(claim_request),
                operation_name="Work-attempt predecessor settlement authority replay",
                redactor=self._secret_redactor,
                mutation_store=task_store,
                mutation_method_name="claim_work_attempt_recovery",
            )
            if current_outcome.failure is not None:
                raise_task_store_operation_failure(current_outcome.failure)
            current_validation = capture_sensitive_result_validation(
                lambda value=current_outcome.result: require_work_attempt_claim_result(
                    value,
                    recovering,
                    claim_request,
                    operation_name="Work-attempt predecessor settlement authority",
                    allowed_states=frozenset(
                        {
                            WorkAttemptAdmissionState.RECOVERING,
                            WorkAttemptAdmissionState.ACTIVE,
                        }
                    ),
                    allowed_previous_states=frozenset({WorkAttemptAdmissionState.RECOVERING}),
                ),
                operation_name=("Work-attempt predecessor settlement authority validation"),
                redactor=self._secret_redactor,
            )
            del current_outcome
            if current_validation.failure is not None:
                raise_task_store_operation_failure(current_validation.failure)
            authenticated = current_validation.result
            del current_validation
            if authenticated is None:
                raise RuntimeError("Work-attempt predecessor settlement returned no authority.")
            if authenticated.state is WorkAttemptAdmissionState.ACTIVE:
                del authenticated
                raise _WorkAttemptRecoveryAlreadyActive
            del authenticated

        try:
            recovered = await self._session_engine._recover_work_attempt_session(
                IncompleteSessionRecoveryRequest(
                    session_id=recovering.session_id,
                    reason="work_attempt_predecessor_owner_expired",
                ),
                before_mutation=require_exact_recovery_claim,
                interaction_id=recovering.interaction_id,
                admission=recovering,
            )
        except _WorkAttemptRecoveryAlreadyActive:
            return
        if type(recovered) is not IncompleteSessionRecoveryResult:
            raise RuntimeError("Work-attempt predecessor settlement returned an invalid result.")
        if recovered.session_id != recovering.session_id:
            raise RuntimeError("Work-attempt predecessor settlement changed the session identity.")
        if IncompleteSessionRecoveryAction.SKIPPED_ACTIVE in recovered.actions:
            raise WorkAttemptRecoveryRequired(
                "Work-attempt predecessor settlement is still owned by another recovery."
            )
        if recovered.status not in {
            SessionStatus.COMPLETED,
            SessionStatus.FAILED,
            SessionStatus.INTERRUPTED,
        }:
            raise WorkAttemptRecoveryRequired(
                "Work-attempt predecessor settlement remains active or ambiguous."
            )

    async def _load_work_attempt_runtime_checkpoint(
        self,
        session_id: str,
        *,
        operation_name: str,
    ) -> dict[str, Any] | None:
        """Load authenticated runtime checkpoint authority for work recovery."""

        raw_checkpoint = await read_work_attempt_session_store(
            lambda: self._runtime_session_store.load_checkpoint(session_id),
            operation_name=operation_name,
            redactor=self._secret_redactor,
        )
        checkpoint_validation = capture_work_attempt_checkpoint_result(
            raw_checkpoint,
            operation_name=operation_name,
            redactor=self._secret_redactor,
        )
        del raw_checkpoint
        if checkpoint_validation.failure is not None:
            raise_task_store_operation_failure(checkpoint_validation.failure)
        checkpoint = checkpoint_validation.result
        del checkpoint_validation
        if active_invocation_execution_profile_from_checkpoint(
            checkpoint
        ) is None and invocation_lifecycle_receipt_history_present(checkpoint):
            raise WorkAttemptRecoveryRequired(
                "Work-attempt recovery lost durable invocation profile authority."
            )
        return checkpoint

    async def recover(
        self,
        request: WorkAttemptRecoveryRequest,
    ) -> WorkAttemptAdmission:
        """Replace an expired generation after positive session quiescence."""

        operation = self.claim_recovery(request)
        del request
        try:
            ownership = await operation
        finally:
            del operation
        return await self.recover_claimed(ownership)

    async def claim_recovery(
        self, request: WorkAttemptRecoveryRequest
    ) -> WorkAttemptRecoveryOwnership:
        """Acknowledge an exact claim before the worker starts its heartbeat."""

        request_type = type(request)
        request_validation = _copied_public_work_attempt_recovery_request(
            request,
            redactor=self._secret_redactor,
        )
        del request
        request_failure = request_validation.failure
        stable = request_validation.result
        del request_validation
        if request_failure is not None:
            raise request_failure from None
        if stable is None:
            if request_type is not WorkAttemptRecoveryRequest:
                raise TypeError(
                    "Work-attempt recovery requires a WorkAttemptRecoveryRequest."
                ) from None
            raise ValueError("Work-attempt recovery request is invalid.") from None
        task_store = self.task_store
        if task_store is None:
            raise RuntimeError("task_store is required for work-attempt recovery.")
        if not task_store_work_attempt_admission_capability_is_complete(task_store):
            raise NotImplementedError(
                f"{type(task_store).__name__} does not implement the complete "
                "work-attempt admission contract."
            )
        claim_request = WorkAttemptExecutionClaimRequest(
            admission_id=stable.admission_id,
            claim_id=stable.claim_id,
            worker_id=stable.worker_id,
            execution_owner_id=self.current_execution_owner_id(),
            generation=stable.generation,
            lease_seconds=stable.lease_seconds,
        )
        prior_outcome = await capture_task_store_operation(
            lambda: task_store.load_work_attempt_admission(stable.admission_id),
            operation_name="Work-attempt recovery authority lookup",
            redactor=self._secret_redactor,
        )
        if prior_outcome.failure is not None:
            raise_task_store_operation_failure(prior_outcome.failure)
        if prior_outcome.result is None:
            raise KeyError(f"Work-attempt admission not found: {stable.admission_id}")
        prior_validation = capture_sensitive_result_validation(
            lambda value=prior_outcome.result: require_work_attempt_admission_result(
                value,
                operation_name="Work-attempt recovery authority lookup",
            ),
            operation_name="Work-attempt recovery authority validation",
            redactor=self._secret_redactor,
        )
        del prior_outcome
        if prior_validation.failure is not None:
            raise_task_store_operation_failure(prior_validation.failure)
        prior = prior_validation.result
        del prior_validation
        if prior is None:
            raise RuntimeError("Work-attempt recovery authority lookup returned no authority.")
        if prior.state is WorkAttemptAdmissionState.PREPARING:
            validation = capture_sensitive_result_validation(
                lambda prior=prior: self._session_engine.reconstruct_preparing_work_attempt_source(
                    prior
                ),
                operation_name="Prepared recovery source validation",
                redactor=self._secret_redactor,
            )
            if validation.failure is not None:
                del prior
                raise_task_store_operation_failure(validation.failure)
            del validation
        # Reject ambiguous migrated lifecycle state before claiming the next
        # TaskStore generation.  The runtime adapter is the sole owner of root
        # checkpoint migration; bypassing it here could reinterpret pre-v5
        # caller-writable state as executable recovery authority.
        await self._load_work_attempt_runtime_checkpoint(
            prior.session_id,
            operation_name="Work-attempt recovery checkpoint preflight",
        )
        claimed_outcome = await capture_task_store_operation(
            lambda: task_store.claim_work_attempt_recovery(claim_request),
            operation_name="Work-attempt recovery claim",
            redactor=self._secret_redactor,
            mutation_store=task_store,
            mutation_method_name="claim_work_attempt_recovery",
        )
        if claimed_outcome.failure is not None:
            raise_task_store_operation_failure(claimed_outcome.failure)
        claim_validation = capture_sensitive_result_validation(
            lambda value=claimed_outcome.result, previous=prior: require_work_attempt_claim_result(
                value,
                previous,
                claim_request,
                operation_name="Work-attempt recovery claim",
                allowed_states=frozenset(
                    {
                        WorkAttemptAdmissionState.PREPARING,
                        WorkAttemptAdmissionState.RECOVERING,
                        WorkAttemptAdmissionState.ACTIVE,
                    }
                ),
                allowed_previous_states=frozenset(
                    {
                        WorkAttemptAdmissionState.PREPARING,
                        WorkAttemptAdmissionState.ACTIVE,
                        WorkAttemptAdmissionState.RECOVERING,
                    }
                ),
            ),
            operation_name="Work-attempt recovery claim result validation",
            redactor=self._secret_redactor,
        )
        del claimed_outcome, prior
        if claim_validation.failure is not None:
            raise_task_store_operation_failure(claim_validation.failure)
        claimed = claim_validation.result
        del claim_validation
        if claimed is None:
            raise RuntimeError("Work-attempt recovery claim returned no authority.")
        return _acknowledged_work_attempt_recovery(task_store, claimed, claim_request)

    async def recover_claimed(
        self, ownership: WorkAttemptRecoveryOwnership
    ) -> WorkAttemptAdmission:
        """Continue existing cleanup under acknowledged same-runtime ownership."""
        if type(ownership) is not WorkAttemptRecoveryOwnership:
            raise TypeError("Work-attempt recovery requires acknowledged runtime ownership.")
        task_store, claim_request, claimed = ownership.store, ownership.request, ownership.admission
        if (
            task_store is not self.task_store
            or claim_request.execution_owner_id != self.current_execution_owner_id()
        ):
            raise WorkAttemptExecutionClaimLost("Recovery ownership belongs to another runtime.")
        if claimed.state is WorkAttemptAdmissionState.PREPARING:
            return await self._session_engine.recover_preparing_work_attempt(ownership)
        del ownership
        stable = WorkAttemptRecoveryRequest(
            admission_id=claim_request.admission_id,
            claim_id=claim_request.claim_id,
            worker_id=claim_request.worker_id,
            generation=claim_request.generation,
            lease_seconds=claim_request.lease_seconds,
        )
        # A successful recovery claim replaces the prior execution generation.
        # Retire any predecessor epoch copied into this caller before either
        # replaying an already-active generation or mutating the recovered
        # session.  The active replay branch below reinstalls only the current
        # durable epoch after validating its complete authority.
        _deactivate_session_run_fence(claimed.session_id)
        released_recovery = await self._session_engine.reconcile_released_work_attempt_recovery(
            claimed
        )
        if released_recovery is not None:
            return released_recovery
        already_active = claimed.state is WorkAttemptAdmissionState.ACTIVE
        if already_active and claimed.execution_stop is not None:
            raise WorkAttemptRecoveryRequired("Stopped work-attempt cleanup is unproven.")
        if already_active and claimed.recovery_evidence_sha256 is None:
            raise RuntimeError("Recovered admission has no durable recovery evidence.")
        if already_active:
            raw_session = await read_work_attempt_session_store(
                lambda session_id=claimed.session_id: self._runtime_session_store.load(session_id),
                operation_name="Recovered work-attempt session lookup",
                redactor=self._secret_redactor,
            )
            if raw_session is None:
                raise WorkAttemptRecoveryRequired("Recovered admission has no durable session.")
            session_validation = capture_work_attempt_session_result(
                raw_session,
                operation_name="Recovered work-attempt session lookup",
                redactor=self._secret_redactor,
            )
            del raw_session
            if session_validation.failure is not None:
                raise_task_store_operation_failure(session_validation.failure)
            session = session_validation.result
            del session_validation
            if session is None:
                raise RuntimeError("Recovered work-attempt session lookup returned no session.")
            profile = execution_profile_from_session_metadata(session.metadata)
            checkpoint = await self._load_work_attempt_runtime_checkpoint(
                session.id,
                operation_name="Recovered work-attempt checkpoint lookup",
            )
            recovery_authority = work_attempt_recovery_session_authority(claimed.claim)
            checkpoint_recovery_authority = work_attempt_recovery_session_authority_from_checkpoint(
                checkpoint
            )
            active_profile = active_invocation_execution_profile_from_checkpoint(checkpoint)
            if (
                SessionInvocationBinding(
                    id=session.id,
                    session_instance_id=session.instance_id,
                    invocation=session.invocation,
                )
                != claimed.session_invocation
                or profile is None
                or profile.fingerprint != claimed.source_execution_profile_fingerprint
                or checkpoint_recovery_authority != recovery_authority
                or (
                    session.status is SessionStatus.RUNNING
                    and (
                        active_profile is None
                        or active_profile.interaction_id != claimed.interaction_id
                        or active_profile.profile != profile
                        or not active_invocation_execution_profile_matches_session_epoch(
                            active_profile,
                            session_id=session.id,
                            run_epoch=session.run_epoch,
                        )
                        or active_invocation_execution_profile_is_released(
                            active_profile,
                            session_id=session.id,
                            run_epoch=session.run_epoch,
                        )
                    )
                )
            ):
                raise WorkAttemptRecoveryRequired(
                    "Recovered receipt conflicts with immutable session authority."
                )
            if session.status is SessionStatus.RUNNING:
                _activate_session_run_fence(session)
                _activate_session_interaction(session.id, claimed.interaction_id)
            return await self._session_engine._fan_out_work_attempt_interaction_started(
                claimed,
                lease_seconds=stable.lease_seconds,
            )
        raw_session = await read_work_attempt_session_store(
            lambda session_id=claimed.session_id: self._runtime_session_store.load(session_id),
            operation_name="Work-attempt recovery session lookup",
            redactor=self._secret_redactor,
        )
        if raw_session is None:
            raise WorkAttemptRecoveryRequired(
                "Recovery cannot prove settlement because the admitted session is missing."
            )
        session_validation = capture_work_attempt_session_result(
            raw_session,
            operation_name="Work-attempt recovery session lookup",
            redactor=self._secret_redactor,
        )
        del raw_session
        if session_validation.failure is not None:
            raise_task_store_operation_failure(session_validation.failure)
        session = session_validation.result
        del session_validation
        if session is None:
            raise RuntimeError("Work-attempt recovery session lookup returned no session.")
        profile = execution_profile_from_session_metadata(session.metadata)
        if (
            profile is None
            or profile.fingerprint != claimed.source_execution_profile_fingerprint
            or SessionInvocationBinding(
                id=session.id,
                session_instance_id=session.instance_id,
                invocation=session.invocation,
            )
            != claimed.session_invocation
        ):
            raise WorkAttemptRecoveryRequired(
                "Recovery cannot prove the admitted session and source profile."
            )
        recovery_authority = work_attempt_recovery_session_authority(claimed.claim)
        recovery_marker = recovery_authority.checkpoint_value()
        checkpoint = await self._load_work_attempt_runtime_checkpoint(
            session.id,
            operation_name="Work-attempt recovery checkpoint lookup",
        )
        pending_initial_interaction_id = _initial_transcript_pending_interaction_id(checkpoint)
        if pending_initial_interaction_id is not None:
            raw_deferred_input = await read_work_attempt_session_store(
                lambda session_id=session.id: (
                    self._runtime_session_store.load_deferred_interaction_input(session_id)
                ),
                operation_name="Work-attempt recovery deferred-input lookup",
                redactor=self._secret_redactor,
            )
            deferred_validation = capture_work_attempt_deferred_input_result(
                raw_deferred_input,
                operation_name="Work-attempt recovery deferred-input lookup",
                redactor=self._secret_redactor,
            )
            del raw_deferred_input
            if deferred_validation.failure is not None:
                raise_task_store_operation_failure(deferred_validation.failure)
            deferred_authority = deferred_validation.result
            del deferred_validation
            if deferred_authority != (pending_initial_interaction_id, True):
                raise WorkAttemptRecoveryRequired(
                    "Recovery requires the authenticated complete initial transcript; "
                    "source-only migrated input must start a new session."
                )
        checkpoint_recovery_authority = work_attempt_recovery_session_authority_from_checkpoint(
            checkpoint
        )
        active_profile = active_invocation_execution_profile_from_checkpoint(checkpoint)
        active_model_completion = await read_work_attempt_session_store(
            lambda session_id=session.id: (
                self._runtime_session_store.load_active_model_completion_stage(session_id)
            ),
            operation_name="Work-attempt recovery active model-stage lookup",
            redactor=self._secret_redactor,
        )
        has_active_model_completion = active_model_completion is not None
        del active_model_completion
        exact_session_replay = (
            checkpoint_recovery_authority == recovery_authority
            and session.status is SessionStatus.RUNNING
            and active_profile is not None
            and active_invocation_execution_profile_matches_session_epoch(
                active_profile,
                session_id=session.id,
                run_epoch=session.run_epoch,
            )
            and not active_invocation_execution_profile_is_released(
                active_profile,
                session_id=session.id,
                run_epoch=session.run_epoch,
            )
            and active_profile.interaction_id == claimed.interaction_id
            and active_profile.profile == profile
        )
        predecessor_settlement_required = checkpoint_recovery_authority is None and (
            claimed.execution_stop is not None
            or session.status
            not in {
                SessionStatus.COMPLETED,
                SessionStatus.FAILED,
                SessionStatus.INTERRUPTED,
            }
            or (
                active_profile is not None
                and not active_invocation_execution_profile_is_released(
                    active_profile,
                    session_id=session.id,
                    run_epoch=session.run_epoch,
                )
            )
        )
        if predecessor_settlement_required:
            if (
                active_profile is None
                or not active_invocation_execution_profile_matches_session_epoch(
                    active_profile,
                    session_id=session.id,
                    run_epoch=session.run_epoch,
                )
                or active_profile.interaction_id != claimed.interaction_id
                or active_profile.profile != profile
            ):
                raise WorkAttemptRecoveryRequired(
                    "Recovery cannot prove the crashed predecessor invocation authority."
                )
            model_result_ready = (
                await self._session_engine.has_recoverable_work_attempt_model_result(claimed)
            )
            if has_active_model_completion and not model_result_ready:
                raise WorkAttemptRecoveryRequired(
                    "Recovery is fenced while model-completion settlement remains active."
                )
            self._checkpoint_guard(
                session,
                checkpoint,
                redactor=self._secret_redactor,
                allow_pending_tool_round=model_result_ready,
                allowed_initial_transcript_interaction_id=(
                    claimed.interaction_id if claimed.kind == "initial" else None
                ),
            )
            await self._settle_first_work_attempt_recovery_predecessor(
                recovering=claimed,
                claim_request=claim_request,
            )
            del (
                claim_request,
                claimed,
                session,
                profile,
                recovery_authority,
                recovery_marker,
                checkpoint,
                checkpoint_recovery_authority,
                active_profile,
                has_active_model_completion,
                exact_session_replay,
                predecessor_settlement_required,
            )
            return await self.recover(stable)
        if claimed.execution_stop is not None:
            raise WorkAttemptRecoveryRequired("Stopped work-attempt cleanup is unproven.")
        prior_recovery_transition = False
        if (
            not exact_session_replay
            and checkpoint_recovery_authority is not None
            and session.status is SessionStatus.RUNNING
            and active_profile is not None
            and active_invocation_execution_profile_matches_session_epoch(
                active_profile,
                session_id=session.id,
                run_epoch=session.run_epoch,
            )
            and not active_invocation_execution_profile_is_released(
                active_profile,
                session_id=session.id,
                run_epoch=session.run_epoch,
            )
            and active_profile.interaction_id == claimed.interaction_id
            and active_profile.profile == profile
        ):
            prior_claim_outcome = await capture_task_store_operation(
                lambda: task_store.load_work_attempt_execution_claim(
                    checkpoint_recovery_authority.claim_id
                ),
                operation_name="Prior work-attempt recovery claim lookup",
                redactor=self._secret_redactor,
            )
            if prior_claim_outcome.failure is not None:
                raise_task_store_operation_failure(prior_claim_outcome.failure)
            raw_prior_claim = prior_claim_outcome.result
            del prior_claim_outcome
            if raw_prior_claim is None:
                raise WorkAttemptRecoveryRequired(
                    "Running session recovery marker has no historical claim authority."
                )

            def validate_prior_claim(
                value: object = raw_prior_claim,
                *,
                expected_admission_id: str = claimed.admission_id,
                expected_generation: int = claimed.claim.generation,
                replacement_claimed_at: datetime = claimed.claim.claimed_at,
            ) -> bool:
                prior_claim = require_work_attempt_execution_claim_result(
                    value,
                    operation_name="Prior work-attempt recovery claim lookup",
                )
                if (
                    work_attempt_recovery_session_authority(prior_claim)
                    != checkpoint_recovery_authority
                    or prior_claim.admission_id != expected_admission_id
                    or prior_claim.generation >= expected_generation
                    or prior_claim.lease_expires_at > replacement_claimed_at
                ):
                    raise WorkAttemptRecoveryRequired(
                        "Running session recovery marker is not an expired predecessor "
                        "of the current claim."
                    )
                return True

            prior_claim_validation = capture_sensitive_result_validation(
                validate_prior_claim,
                operation_name="Prior work-attempt recovery claim result validation",
                redactor=self._secret_redactor,
            )
            del raw_prior_claim, validate_prior_claim
            if prior_claim_validation.failure is not None:
                raise_task_store_operation_failure(prior_claim_validation.failure)
            prior_claim_is_valid = prior_claim_validation.result
            del prior_claim_validation
            if prior_claim_is_valid is not True:
                raise RuntimeError("Prior work-attempt recovery claim validation failed closed.")
            prior_recovery_transition = True
        if has_active_model_completion:
            raise WorkAttemptRecoveryRequired(
                "Recovery is fenced while model-completion settlement remains active."
            )
        if not exact_session_replay:
            source_session_sha256 = _work_attempt_recovery_session_snapshot_sha256(session)
            source_checkpoint_sha256 = _work_attempt_recovery_checkpoint_snapshot_sha256(checkpoint)

            def claim_recovered_session_execution(
                current_session: Session,
                current_checkpoint: dict[str, Any] | None,
                *,
                interaction_id: str = claimed.interaction_id,
                admission_kind: str = claimed.kind,
            ) -> dict[str, Any]:
                if (
                    _work_attempt_recovery_session_snapshot_sha256(current_session)
                    != source_session_sha256
                    or _work_attempt_recovery_checkpoint_snapshot_sha256(current_checkpoint)
                    != source_checkpoint_sha256
                ):
                    raise WorkAttemptRecoveryRequired(
                        "Session authority changed after recovery was claimed."
                    )
                self._checkpoint_guard(
                    current_session,
                    current_checkpoint,
                    redactor=self._secret_redactor,
                    allowed_initial_transcript_interaction_id=(
                        interaction_id if admission_kind == "initial" else None
                    ),
                )
                # Replacing an execution owner cannot undo a terminal winner,
                # including one published while settling the expired epoch.
                for decision in (
                    invocation_terminal_decision_from_checkpoint(current_checkpoint),
                    settled_invocation_terminal_decision_from_checkpoint(current_checkpoint),
                ):
                    if decision is not None and decision.interaction_id == interaction_id:
                        raise WorkAttemptRecoveryRequired(
                            "Work-attempt interaction has a durable terminal decision."
                        )
                current_profile = active_invocation_execution_profile_from_checkpoint(
                    current_checkpoint
                )
                if (
                    current_profile is None
                    or current_profile.interaction_id != interaction_id
                    or current_profile.profile != profile
                    or not active_invocation_execution_profile_matches_session_epoch(
                        current_profile,
                        session_id=current_session.id,
                        run_epoch=current_session.run_epoch,
                    )
                ):
                    raise WorkAttemptRecoveryRequired(
                        "Recovery cannot prove the prior invocation profile authority."
                    )
                if current_session.status is SessionStatus.RUNNING:
                    if not prior_recovery_transition:
                        raise WorkAttemptRecoveryRequired(
                            "Recovery cannot replace a running session without positive "
                            "effect-settlement evidence."
                        )
                    if active_invocation_execution_profile_is_released(
                        current_profile,
                        session_id=current_session.id,
                        run_epoch=current_session.run_epoch,
                    ):
                        raise WorkAttemptRecoveryRequired(
                            "Running predecessor recovery has conflicting released authority."
                        )
                else:
                    if not active_invocation_execution_profile_is_released(
                        current_profile,
                        session_id=current_session.id,
                        run_epoch=current_session.run_epoch,
                    ):
                        raise WorkAttemptRecoveryRequired(
                            "Terminal recovery requires a released prior invocation run fence."
                        )
                    if current_session.status not in {
                        SessionStatus.COMPLETED,
                        SessionStatus.FAILED,
                        SessionStatus.INTERRUPTED,
                    }:
                        raise WorkAttemptRecoveryRequired(
                            "Recovery requires a terminal, resumable prior session."
                        )
                if not active_invocation_execution_profile_matches_session_epoch(
                    current_profile,
                    session_id=current_session.id,
                    run_epoch=current_session.run_epoch,
                ):
                    raise WorkAttemptRecoveryRequired(
                        "Recovery cannot prove the released prior invocation epoch."
                    )
                updated = checkpoint_with_active_invocation_execution_profile(
                    current_checkpoint,
                    session_id=current_session.id,
                    interaction_id=interaction_id,
                    run_epoch=current_session.run_epoch + 1,
                    profile=profile,
                    expected=current_profile,
                )
                updated[WORK_ATTEMPT_RECOVERY_CHECKPOINT_KEY] = recovery_marker
                return updated

            recovery_rebind = prepare_rebind_invocation_command(
                session,
                checkpoint,
                expected_statuses={
                    SessionStatus.RUNNING,
                    SessionStatus.COMPLETED,
                    SessionStatus.FAILED,
                    SessionStatus.INTERRUPTED,
                },
                target_status=SessionStatus.RUNNING,
                checkpoint_transform=claim_recovered_session_execution,
            )

            async def transition_recovered_session() -> Session:
                try:
                    result = await self._runtime_session_store.apply_invocation_lifecycle_command(
                        recovery_rebind
                    )
                except SessionRunFenced as exc:
                    raise WorkAttemptRecoveryRequired(
                        "Session authority changed after recovery was claimed."
                    ) from exc
                if type(result) is not InvocationMutationResult:
                    raise RuntimeError(
                        "Work-attempt recovery rebind returned an incompatible result."
                    )
                return result.session

            mutation = settle_work_attempt_session_mutation(
                transition_recovered_session,
                operation_name="work-attempt-session-recovery",
                preserved_failure_types=(
                    SessionRunFenced,
                    WorkAttemptRecoveryRequired,
                ),
                redactor=self._secret_redactor,
            )
            try:
                raw_recovered_session = await mutation
            except BaseException:
                del (
                    stable,
                    claim_request,
                    claimed,
                    session,
                    profile,
                    recovery_authority,
                    recovery_marker,
                    checkpoint,
                    checkpoint_recovery_authority,
                    active_profile,
                    has_active_model_completion,
                    source_session_sha256,
                    source_checkpoint_sha256,
                    claim_recovered_session_execution,
                    recovery_rebind,
                    transition_recovered_session,
                    mutation,
                )
                raise
            del (
                mutation,
                transition_recovered_session,
                recovery_rebind,
                claim_recovered_session_execution,
                session,
            )
            recovered_session_validation = capture_work_attempt_session_result(
                raw_recovered_session,
                operation_name="Work-attempt session recovery",
                redactor=self._secret_redactor,
            )
            del raw_recovered_session
            if recovered_session_validation.failure is not None:
                raise_task_store_operation_failure(recovered_session_validation.failure)
            session = recovered_session_validation.result
            del recovered_session_validation
            if session is None:
                raise RuntimeError("Work-attempt session recovery returned no session.")
            checkpoint = await self._load_work_attempt_runtime_checkpoint(
                session.id,
                operation_name="Recovered work-attempt checkpoint lookup",
            )
            if (
                work_attempt_recovery_session_authority_from_checkpoint(checkpoint)
                != recovery_authority
            ):
                raise WorkAttemptRecoveryRequired(
                    "Session recovery did not publish its exact durable authority."
                )
        recovery_evidence = sha256(
            canonical_durable_json_bytes(
                {
                    "admission_id": claimed.admission_id,
                    "session_id": session.id,
                    "session_status": session.status.value,
                    "run_epoch": session.run_epoch,
                    "session_updated_at": session.updated_at.isoformat(),
                    "session_last_activity_at": session.last_activity_at.isoformat(),
                    "claim_generation": stable.generation,
                    "checkpoint_sha256": sha256(
                        canonical_durable_json_bytes(
                            checkpoint,
                            "work_attempt_recovery_checkpoint",
                        )
                    ).hexdigest(),
                },
                "work_attempt_recovery_evidence",
            )
        ).hexdigest()
        _activate_session_run_fence(session)
        _activate_session_interaction(session.id, claimed.interaction_id)
        activation = WorkAttemptRecoveryActivate(
            admission_id=claimed.admission_id,
            claim_id=claimed.claim.claim_id,
            generation=claimed.claim.generation,
            recovery_evidence_sha256=recovery_evidence,
        )
        activated_outcome = await capture_task_store_operation(
            lambda: task_store.activate_work_attempt_recovery(activation),
            operation_name="Work-attempt recovery activation",
            redactor=self._secret_redactor,
            mutation_store=task_store,
            mutation_method_name="activate_work_attempt_recovery",
        )
        if activated_outcome.failure is not None:
            raise_task_store_operation_failure(activated_outcome.failure)
        activation_validation = capture_sensitive_result_validation(
            lambda value=activated_outcome.result, recovering=claimed: (
                require_work_attempt_recovery_activation_result(
                    value,
                    recovering,
                    activation,
                )
            ),
            operation_name="Work-attempt recovery activation result validation",
            redactor=self._secret_redactor,
        )
        del activated_outcome, claimed
        if activation_validation.failure is not None:
            raise_task_store_operation_failure(activation_validation.failure)
        active = activation_validation.result
        del activation_validation
        if active is None:
            raise RuntimeError("Work-attempt recovery activation returned no authority.")
        return await self._session_engine._fan_out_work_attempt_interaction_started(
            active,
            lease_seconds=stable.lease_seconds,
        )

    async def submit_proposal(
        self,
        request: WorkAttemptProposalRequest,
    ) -> CompletionProposal:
        """Publish one proposal under the exact live attempt execution claim."""

        request_type = type(request)
        request_validation = _copied_public_work_attempt_proposal_request(
            request,
            redactor=self._secret_redactor,
        )
        del request
        request_failure = request_validation.failure
        public_request = request_validation.result
        del request_validation
        if request_failure is not None:
            raise request_failure from None
        if public_request is None:
            if request_type is not WorkAttemptProposalRequest:
                raise TypeError(
                    "Work-attempt proposal requires a WorkAttemptProposalRequest."
                ) from None
            raise ValueError("Work-attempt proposal request is invalid.") from None
        task_store = self.task_store
        if task_store is None:
            raise RuntimeError("task_store is required for completion proposals.")
        if not task_store_work_attempt_admission_capability_is_complete(task_store):
            raise NotImplementedError(
                f"{type(task_store).__name__} does not implement the complete "
                "work-attempt admission contract."
            )
        execution_owner_id = self.current_execution_owner_id()
        admission_outcome = await capture_task_store_operation(
            lambda: task_store.load_work_attempt_admission(public_request.admission_id),
            operation_name="Admitted completion proposal reconciliation lookup",
            redactor=self._secret_redactor,
        )
        if admission_outcome.failure is not None:
            raise_task_store_operation_failure(admission_outcome.failure)
        if admission_outcome.result is None:
            raise KeyError(f"Work-attempt admission not found: {public_request.admission_id}")
        admission_validation = capture_sensitive_result_validation(
            lambda value=admission_outcome.result: require_work_attempt_admission_result(
                value,
                operation_name="Admitted completion proposal reconciliation lookup",
            ),
            operation_name="Admitted completion proposal authority validation",
            redactor=self._secret_redactor,
        )
        del admission_outcome
        if admission_validation.failure is not None:
            raise_task_store_operation_failure(admission_validation.failure)
        admission = admission_validation.result
        del admission_validation
        if admission is None:
            raise RuntimeError("Admitted completion proposal lookup returned no authority.")
        if (
            admission.state is WorkAttemptAdmissionState.RELEASED
            and admission.claim.claim_id == public_request.claim_id
            and admission.claim.generation == public_request.generation
            and admission.attempt_id == public_request.proposal.attempt_id
        ):
            # A committed release is positive store-owned replay evidence.
            # Reconstruct its private owner only for the content-bound,
            # non-mutating proposal receipt path after process replacement.
            execution_owner_id = admission.claim.execution_owner_id
        copied = AdmittedCompletionProposalRequest(
            **public_request.model_dump(mode="python", warnings=False),
            execution_owner_id=execution_owner_id,
        )
        outcome = await capture_task_store_operation(
            lambda: task_store.submit_admitted_completion_proposal(copied),
            operation_name="Admitted completion proposal publication",
            redactor=self._secret_redactor,
            mutation_store=task_store,
            mutation_method_name="submit_admitted_completion_proposal",
        )
        if outcome.failure is not None:
            raise_task_store_operation_failure(outcome.failure)
        proposal_validation = capture_sensitive_result_validation(
            lambda value=outcome.result, authority=admission: (
                require_admitted_completion_proposal_result(
                    value,
                    copied,
                    authority,
                )
            ),
            operation_name="Admitted completion proposal result validation",
            redactor=self._secret_redactor,
        )
        del outcome, admission
        if proposal_validation.failure is not None:
            raise_task_store_operation_failure(proposal_validation.failure)
        proposal = proposal_validation.result
        del proposal_validation
        if proposal is None:
            raise RuntimeError("Admitted completion proposal returned no receipt.")
        return proposal


def _work_attempt_recovery_session_snapshot_sha256(session: Session) -> str:
    return sha256(
        canonical_durable_json_bytes(
            copy_session(session).model_dump(mode="json", warnings=False),
            "work_attempt_recovery_source_session",
        )
    ).hexdigest()


def _work_attempt_recovery_checkpoint_snapshot_sha256(
    checkpoint: dict[str, Any] | None,
) -> str:
    return sha256(
        canonical_durable_json_bytes(
            checkpoint,
            "work_attempt_recovery_source_checkpoint",
        )
    ).hexdigest()


def _copied_public_work_attempt_source_request(
    value: object,
    *,
    redactor: SecretRedactor,
) -> TaskStoreOperationOutcome[RunRequest | ResumeRequest]:
    """Copy one work-attempt source request behind a detached boundary."""

    def copy_request() -> RunRequest | ResumeRequest:
        if type(value) is RunRequest:
            return _validate_run_request(value)
        if type(value) is ResumeRequest:
            return _validate_resume_request(value)
        raise TypeError("Work-attempt source request must be a RunRequest or ResumeRequest.")

    return capture_sensitive_validation(
        copy_request,
        operation_name="Work-attempt source-request validation",
        redactor=redactor,
    )


def _copied_public_work_attempt_execution_request(
    value: object,
    *,
    redactor: SecretRedactor,
) -> TaskStoreOperationOutcome[WorkAttemptExecutionRequest]:
    """Copy caller-owned initial or continuation execution authority safely."""

    return capture_sensitive_validation(
        lambda: copy_work_attempt_execution_request(cast("WorkAttemptExecutionRequest", value)),
        operation_name="Work-attempt execution-request validation",
        redactor=redactor,
    )


def _copied_public_work_attempt_claim_renewal_request(
    value: object,
    *,
    redactor: SecretRedactor,
) -> TaskStoreOperationOutcome[WorkAttemptClaimRenewalRequest]:
    """Copy caller-owned renewal authority safely."""

    return capture_sensitive_validation(
        lambda: copy_work_attempt_claim_renewal_request(
            cast("WorkAttemptClaimRenewalRequest", value)
        ),
        operation_name="Work-attempt claim-renewal request validation",
        redactor=redactor,
    )


def _copied_public_work_attempt_recovery_request(
    value: object,
    *,
    redactor: SecretRedactor,
) -> TaskStoreOperationOutcome[WorkAttemptRecoveryRequest]:
    """Copy caller-owned recovery authority safely."""

    return capture_sensitive_validation(
        lambda: copy_work_attempt_recovery_request(cast("WorkAttemptRecoveryRequest", value)),
        operation_name="Work-attempt recovery-request validation",
        redactor=redactor,
    )


def _copied_public_work_attempt_proposal_request(
    value: object,
    *,
    redactor: SecretRedactor,
) -> TaskStoreOperationOutcome[WorkAttemptProposalRequest]:
    """Copy one caller-owned admitted proposal safely."""

    return capture_sensitive_validation(
        lambda: copy_work_attempt_proposal_request(cast("WorkAttemptProposalRequest", value)),
        operation_name="Work-attempt proposal-request validation",
        redactor=redactor,
    )

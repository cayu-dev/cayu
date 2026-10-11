"""Governed work-attempt runtime operations, independent of turn scheduling."""

from __future__ import annotations

import asyncio
import hashlib
from collections.abc import AsyncGenerator, Callable
from datetime import datetime
from typing import Any, Literal, Protocol
from uuid import UUID

from cayu._task_wait import capture_awaitable_outcome
from cayu._validation import (
    canonical_durable_json_bytes,
    copy_durable_record,
)
from cayu.budgets.base import (
    BudgetLimit,
    BudgetPolicy,
)
from cayu.budgets.run_limits import RunLimits
from cayu.context.structured_output import StructuredOutputSpec
from cayu.context.thinking import ThinkingConfig
from cayu.deadlines import resumed_execution_deadline
from cayu.events import (
    Event,
    EventType,
)
from cayu.execution_profiles import ExecutionProfileIdentity
from cayu.messages import Message
from cayu.providers.retry_policy import RetryPolicy
from cayu.runtime import _runtime_records as runtime_records
from cayu.runtime import (
    _session_checkpoint_admission,
    _session_execution_profile,
    _session_operation_state,
)
from cayu.runtime import _session_request_boundary as session_request_boundary
from cayu.runtime import _transcript as transcript_helpers
from cayu.runtime._delegated_event_stream import _close_delegated_event_stream
from cayu.runtime._durable_tool_round import _environment_name
from cayu.runtime._environment_lifecycle import EnvironmentLifecycle
from cayu.runtime._event_writer import RuntimeEventWriter
from cayu.runtime._execution_profile_continuation import ExecutionProfileContinuation
from cayu.runtime._interaction_events import (
    build_interaction_started_event,
    build_interaction_started_event_from_identity,
)
from cayu.runtime._interruption_coordinator import _PENDING_SESSION_INTERRUPT_CHECKPOINT_KEY
from cayu.runtime._invocation_lifecycle import (
    InvocationContext,
    _authenticated_invocation_context,
    invocation_checkpoint_state_sha256,
)
from cayu.runtime._model_completion_contracts import model_completion_recovery_context_from_stage
from cayu.runtime._model_completion_recovery import ModelCompletionRecovery
from cayu.runtime._model_step_executor import reconstruct_assistant_step_result
from cayu.runtime._pending_tool_round_recovery import PendingToolRoundRecovery
from cayu.runtime._session_finalization import (
    _INTERRUPTION_TYPE_RUNTIME_INTERRUPTED,
    SessionFinalization,
)
from cayu.runtime._session_preparation import SessionPreparation
from cayu.runtime._session_tool_discovery import (
    _require_tool_discovery_view,
    _tool_discovery_operation_initializer,
)
from cayu.runtime._structured_output_tool_round import has_recoverable_structured_output_round
from cayu.runtime._task_lease_authority import managed_task_lease_mutation
from cayu.runtime._task_store_operation_boundary import (
    capture_sensitive_result_validation,
    capture_task_store_operation,
    raise_task_store_operation_failure,
    task_store_work_attempt_admission_capability_is_complete,
)
from cayu.runtime._work_attempt_invocation import (
    WorkAttemptRecoveryOwnership,
    _authenticated_work_attempt_invocation,
    _WorkAttemptRuntimeAuthority,
)
from cayu.runtime._work_attempt_session_mutation import (
    capture_work_attempt_checkpoint_result,
    capture_work_attempt_deferred_input_result,
    capture_work_attempt_event_records_result,
    capture_work_attempt_execution_input_result,
    capture_work_attempt_session_result,
    read_work_attempt_session_store,
    settle_work_attempt_owned_operation,
    settle_work_attempt_session_mutation,
)
from cayu.runtime.loop_policies import LoopPolicy
from cayu.runtime.tool_completion import ToolCompletionPolicy
from cayu.runtime.work_attempt_semantics import WorkAttemptRunSemantics
from cayu.runtime.work_attempt_source import WorkAttemptSourceRequest
from cayu.sessions import _completion_finalization as completion_finalization
from cayu.sessions import _model_completion_publication as model_completion_publication
from cayu.sessions import _pending_tool_round_reader as pending_round_reader
from cayu.sessions._checkpoint_mutations import runtime_publication_checkpoint_mutation
from cayu.sessions._checkpoint_preservation import _invocation_lifecycle_authority_read_scope
from cayu.sessions._checkpoint_publication import _initial_transcript_pending_interaction_id
from cayu.sessions._execution_profile_checkpoint import (
    ActiveInvocationExecutionProfile,
    active_invocation_execution_profile_from_checkpoint,
    active_invocation_execution_profile_is_released,
    active_invocation_execution_profile_matches_session_epoch,
    execution_profile_from_session_metadata,
)
from cayu.sessions._invocation_lifecycle import (
    AdmitInvocationCommand,
    AdmittedInvocationBinding,
    CreateInvocationCommand,
    InvocationCheckpointPatch,
    InvocationMutationResult,
    invocation_lifecycle_receipt_history_present,
    released_invocation_evidence,
)
from cayu.sessions._invocation_scope import (
    _activate_session_interaction,
    _activate_session_run_fence,
    _deactivate_session_interaction,
    _deactivate_session_run_fence,
)
from cayu.sessions._invocation_terminal_decision import (
    InvocationTerminalOutcome,
    invocation_terminal_decision_from_checkpoint,
    invocation_terminal_decision_matches_active_profile,
    settled_invocation_terminal_decision_from_checkpoint,
)
from cayu.sessions.authority import SessionRunFenced
from cayu.sessions.creation import (
    run_request_with_prepared_work_attempt_creation,
    session_invocation_for_run_request,
)
from cayu.sessions.creation_claims import bind_runtime_session_create_claim
from cayu.sessions.event_delivery import (
    PersistedEventSideEffectDelivery,
    PersistedEventSideEffectStatus,
)
from cayu.sessions.event_queries import EventQuery
from cayu.sessions.invocation import SessionInvocationBinding
from cayu.sessions.invocation_release import InvocationReleaseEvidence
from cayu.sessions.records import (
    Session,
    SessionIdentity,
    SessionStatus,
    SessionStatusConflict,
)
from cayu.sessions.requests import (
    ResumeRequest,
    RunRequest,
    run_request_with_runtime_generated_authority,
    run_request_with_runtime_initial_transcript_authority,
    run_request_with_runtime_session_instance_authority,
)
from cayu.sessions.store import SessionStore
from cayu.tasks.admission import (
    WORK_ATTEMPT_RECOVERY_CHECKPOINT_KEY,
    WorkAttemptAdmission,
    WorkAttemptAdmissionActivate,
    WorkAttemptAdmissionConflict,
    WorkAttemptAdmissionPrepare,
    WorkAttemptAdmissionState,
    WorkAttemptExecutionClaimLost,
    WorkAttemptExecutionClaimRequest,
    WorkAttemptExecutionEntryDisposition,
    WorkAttemptExecutionEntryRequest,
    WorkAttemptExecutionRequest,
    WorkAttemptRecoveryActivate,
    WorkAttemptRecoveryRequired,
    require_work_attempt_activation_result,
    require_work_attempt_admission_result,
    require_work_attempt_claim_result,
    require_work_attempt_execution_entry_result,
    require_work_attempt_preparation_result,
    require_work_attempt_recovery_activation_result,
    work_attempt_recovery_session_authority,
    work_attempt_recovery_session_authority_from_checkpoint,
)
from cayu.tasks.contracts import TaskCompletionDecisionRequired, WorkCompletionConflict
from cayu.tasks.records import Task, copy_task
from cayu.tasks.store import TaskStore
from cayu.tools.discovery import (
    ToolDiscoveryViewInitialization,
    tool_discovery_view_initialization,
)
from cayu.tools.exposure import tool_capability_ceiling_from_session_metadata
from cayu.vaults.redaction import SecretRedactor

_WORK_ATTEMPT_HANDOFF_RENEWAL_MAX_INTERVAL_SECONDS = 30.0
_WORK_ATTEMPT_HANDOFF_RENEWAL_MIN_INTERVAL_SECONDS = 0.05


class WorkAttemptTurnExecution(Protocol):
    """Execute an already elected invocation through the ordinary turn loop."""

    def __call__(
        self,
        *,
        session: Session,
        invocation_context: InvocationContext,
        messages: list[Message],
        messages_to_append: list[Message],
        max_steps: int,
        limits: RunLimits,
        budget_limits: tuple[BudgetLimit, ...],
        retry_policy: RetryPolicy,
        structured_output: StructuredOutputSpec | None,
        thinking: ThinkingConfig | None,
        request_metadata: dict[str, Any],
        task_id: str | None,
        task_worker_id: str | None,
        task_handoff_id: str | None,
        start_event_type: EventType | None,
        start_event_payload: dict[str, Any],
        tool_completion: ToolCompletionPolicy | None,
        messages_deferred: bool,
    ) -> AsyncGenerator[Event, None]: ...


class WorkAttemptExecution:
    """Own runtime admission, election, handoff and exact settlement evidence."""

    def __init__(
        self,
        *,
        session_store: SessionStore,
        task_store: TaskStore | None,
        session_preparation: SessionPreparation,
        execution_profile_continuation: ExecutionProfileContinuation,
        environment_lifecycle: EnvironmentLifecycle,
        event_writer: RuntimeEventWriter,
        model_completion_recovery: ModelCompletionRecovery,
        pending_tool_round_recovery: PendingToolRoundRecovery,
        session_finalization: SessionFinalization,
        secret_redactor: SecretRedactor,
        clock: Callable[[], datetime],
        runtime_hooks: tuple[runtime_records.RegisteredRuntimeHook, ...],
        loop_policies: tuple[LoopPolicy, ...],
        get_budget_policy: Callable[[], BudgetPolicy | None],
        get_registered_agent: Callable[[str], runtime_records.RegisteredAgentState],
        get_registered_provider: Callable[[str | None], runtime_records.RegisteredProvider],
        get_registered_environment_for_session: Callable[
            [str | None], runtime_records.RegisteredEnvironment | None
        ],
        effective_retry_policy: Callable[[RetryPolicy | None], RetryPolicy],
        execute_session: WorkAttemptTurnExecution,
    ) -> None:
        self.session_store = session_store
        self.task_store = task_store
        self._session_preparation = session_preparation
        self._execution_profile_continuation = execution_profile_continuation
        self._environment_lifecycle = environment_lifecycle
        self._event_writer = event_writer
        self._model_completion_recovery = model_completion_recovery
        self._pending_tool_round_recovery = pending_tool_round_recovery
        self._session_finalization = session_finalization
        self._secret_redactor = secret_redactor
        self._clock = clock
        self._runtime_hooks = runtime_hooks
        self._loop_policies = loop_policies
        self._get_budget_policy = get_budget_policy
        self._get_registered_agent = get_registered_agent

        def resolve_provider(name: str | None = None) -> runtime_records.RegisteredProvider:
            return get_registered_provider(name)

        self._get_registered_provider = resolve_provider
        self._get_registered_environment_for_session = get_registered_environment_for_session
        self._effective_retry_policy = effective_retry_policy
        self._execute_session = execute_session

    def work_attempt_source_request_sha256(
        self,
        request: RunRequest | ResumeRequest,
        *,
        kind: Literal["initial", "continuation"],
    ) -> str:
        return session_request_boundary.work_attempt_source_request_sha256(
            request,
            kind=kind,
            redactor=self._secret_redactor,
            policy_instance_identity=self._execution_profile_continuation.policy_identity_registry.identity_for,
        )

    def work_attempt_source_snapshot(
        self,
        request: RunRequest | ResumeRequest,
        *,
        kind: Literal["initial", "continuation"],
        source_request_sha256: str,
    ) -> WorkAttemptSourceRequest | None:
        """Retain the original validated source before admission can commit."""
        if request.loop_policies:
            return None
        fields_set = tuple(sorted(request.model_fields_set))
        if type(request) is RunRequest:
            prepared = session_request_boundary.prepare_run_request(
                request, redactor=self._secret_redactor
            )
        elif type(request) is ResumeRequest:
            prepared = session_request_boundary.prepare_resume_request(
                request, redactor=self._secret_redactor
            )
        else:
            raise TypeError("Work-attempt snapshot requires an exact RunRequest or ResumeRequest.")
        return WorkAttemptSourceRequest.capture(
            kind=kind,
            request=session_request_boundary.work_attempt_source_document(prepared),
            fields_set=fields_set,
            source_request_sha256=source_request_sha256,
        )

    def reconstruct_preparing_work_attempt_source(
        self, admission: WorkAttemptAdmission
    ) -> RunRequest | ResumeRequest:
        """Validate portable data, without conferring private invocation authority."""
        source = admission.source_request
        if (
            admission.state is not WorkAttemptAdmissionState.PREPARING
            or source is None
            or admission.run_semantics is None
        ):
            raise WorkAttemptRecoveryRequired("Prepared work has no portable executable source.")
        model = RunRequest if source.kind == "initial" else ResumeRequest
        if set(source.fields_set) - model.model_fields.keys():
            raise WorkAttemptRecoveryRequired("Prepared work source has unknown explicit controls.")
        request = model.model_validate(source.request)
        if session_request_boundary.work_attempt_source_document(request) != source.request:
            raise WorkAttemptRecoveryRequired("Prepared work source changed during reconstruction.")
        object.__setattr__(request, "__pydantic_fields_set__", set(source.fields_set))
        return request

    async def recover_preparing_work_attempt(
        self, ownership: WorkAttemptRecoveryOwnership
    ) -> WorkAttemptAdmission:
        """Finish original creation under the existing acknowledged claim owner."""
        if (
            type(ownership) is not WorkAttemptRecoveryOwnership
            or ownership.store is not self.task_store
        ):
            raise TypeError("Prepared recovery requires this runtime's acknowledged claim.")
        admission, claim = ownership.admission, ownership.request
        request = self.reconstruct_preparing_work_attempt_source(admission)
        assert admission.run_semantics is not None
        if type(request) is RunRequest:
            request = request.model_copy(
                update={
                    "task_worker_id": claim.worker_id,
                    "task_lease_expires_at": admission.claim.lease_expires_at,
                    "execution_deadline": admission.run_semantics.deadline,
                }
            )
            request = run_request_with_runtime_generated_authority(
                request,
                "session_id",
                "task_id",
                *(("causal_budget_id",) if request.causal_budget_id is not None else ()),
            )
        else:
            request = request.model_copy(update={"task_worker_id": claim.worker_id})
        authority = _WorkAttemptRuntimeAuthority(
            request=WorkAttemptExecutionRequest(
                admission_id=admission.admission_id,
                claim_id=claim.claim_id,
                attempt_id=admission.attempt_id,
                interaction_id=admission.interaction_id,
                worker_id=claim.worker_id,
                task_id=admission.task_id,
                task_lease_expires_at=admission.claim.lease_expires_at,
                predecessor_admission_id=(
                    admission.continuation.prior_admission_id
                    if admission.continuation is not None
                    else None
                ),
                generation=claim.generation,
                lease_seconds=claim.lease_seconds,
            ),
            execution_owner_id=claim.execution_owner_id,
            kind=admission.kind,
            source_request_sha256=admission.source_request_sha256,
            source_request=admission.source_request,
        )
        if type(request) is RunRequest:
            operation = self.admit_initial_work_attempt(request, authority=authority)
        elif type(request) is ResumeRequest:
            operation = self.admit_continuation_work_attempt(request, authority=authority)
        else:
            raise TypeError("Prepared recovery reconstructed an invalid request type.")
        del request, authority, admission, claim, ownership
        return await operation

    async def _prepare_runtime_work_attempt(
        self,
        *,
        authority: _WorkAttemptRuntimeAuthority,
        task_id: str | None,
        session_binding: SessionInvocationBinding,
        execution_profile: ExecutionProfileIdentity,
        run_semantics: WorkAttemptRunSemantics | None,
    ) -> WorkAttemptAdmission:
        if task_id is None:
            raise ValueError("Work-attempt execution requires RunRequest.task_id.")
        task_store = self.task_store
        if task_store is None:
            raise RuntimeError("task_store is required for work-attempt execution.")
        if not task_store_work_attempt_admission_capability_is_complete(task_store):
            raise NotImplementedError(
                f"{type(task_store).__name__} does not implement the complete "
                "work-attempt admission contract."
            )
        task_outcome = await capture_task_store_operation(
            lambda: task_store.load_task(task_id),
            operation_name="Work-attempt task lookup",
            redactor=self._secret_redactor,
        )
        if task_outcome.failure is not None:
            raise_task_store_operation_failure(task_outcome.failure)
        raw_task = task_outcome.result
        del task_outcome
        if raw_task is None:
            raise KeyError(f"Task not found: {task_id}")

        def validate_task_result(value: object = raw_task) -> Task:
            if type(value) is not Task:
                raise RuntimeError("Task store returned invalid work-attempt authority.")
            copied = copy_task(value)
            if copied.id != task_id:
                raise RuntimeError("Task store returned conflicting work-attempt authority.")
            if copied.work_contract is None:
                raise ValueError("Work-attempt execution requires a contract-bound task.")
            return copied

        task_validation = capture_sensitive_result_validation(
            validate_task_result,
            operation_name="Work-attempt task lookup result validation",
            redactor=self._secret_redactor,
        )
        del raw_task, validate_task_result
        if task_validation.failure is not None:
            raise_task_store_operation_failure(task_validation.failure)
        task = task_validation.result
        del task_validation
        if task is None:
            raise RuntimeError("Work-attempt task lookup returned no task authority.")
        contract = task.work_contract
        assert contract is not None
        del task
        stable = authority.request
        prepare = WorkAttemptAdmissionPrepare(
            admission_id=stable.admission_id,
            claim_id=stable.claim_id,
            attempt_id=stable.attempt_id,
            task_id=task_id,
            session_id=session_binding.id,
            interaction_id=stable.interaction_id,
            worker_id=stable.worker_id,
            task_lease_expires_at=stable.task_lease_expires_at,
            execution_owner_id=authority.execution_owner_id,
            generation=stable.generation,
            lease_seconds=stable.lease_seconds,
            kind=authority.kind,
            predecessor_admission_id=stable.predecessor_admission_id,
            source_request_sha256=authority.source_request_sha256,
            contract=contract,
            session_invocation=session_binding,
            source_execution_profile_fingerprint=execution_profile.fingerprint,
            run_semantics=run_semantics,
            source_request=authority.source_request,
        )
        del contract
        async with managed_task_lease_mutation(
            task_id=task_id,
            worker_id=stable.worker_id,
            handoff_id=None,
            presented_lease_expires_at=stable.task_lease_expires_at,
        ) as effective_lease:
            prepare = prepare.model_copy(
                update={"task_lease_expires_at": effective_lease},
                deep=True,
            )
            outcome = await capture_task_store_operation(
                lambda: task_store.prepare_work_attempt_admission(prepare),
                operation_name="Work-attempt admission preparation",
                redactor=self._secret_redactor,
                mutation_store=task_store,
                mutation_method_name="prepare_work_attempt_admission",
            )
        if outcome.failure is not None:
            raise_task_store_operation_failure(outcome.failure)
        validation = capture_sensitive_result_validation(
            lambda value=outcome.result: require_work_attempt_preparation_result(value, prepare),
            operation_name="Work-attempt admission preparation result validation",
            redactor=self._secret_redactor,
        )
        del outcome
        if validation.failure is not None:
            raise_task_store_operation_failure(validation.failure)
        admission = validation.result
        del validation
        if admission is None:
            raise RuntimeError("Work-attempt admission preparation returned no authority.")
        return admission

    async def _activate_runtime_work_attempt(
        self,
        *,
        admission: WorkAttemptAdmission,
        session: Session,
        execution_profile: ExecutionProfileIdentity,
        lease_seconds: int,
    ) -> WorkAttemptAdmission:
        task_store = self.task_store
        if task_store is None:
            raise RuntimeError("task_store is required for work-attempt execution.")
        stored_profile = execution_profile_from_session_metadata(session.metadata)
        raw_deferred_input = await read_work_attempt_session_store(
            lambda: self.session_store.load_deferred_interaction_input(session.id),
            operation_name="Work-attempt deferred-input lookup",
            redactor=self._secret_redactor,
        )
        deferred_validation = capture_work_attempt_deferred_input_result(
            raw_deferred_input,
            operation_name="Work-attempt deferred-input lookup",
            redactor=self._secret_redactor,
        )
        del raw_deferred_input
        if deferred_validation.failure is not None:
            raise_task_store_operation_failure(deferred_validation.failure)
        deferred_authority = deferred_validation.result
        del deferred_validation
        raw_checkpoint = await read_work_attempt_session_store(
            lambda: self.session_store.load_checkpoint(session.id),
            operation_name="Work-attempt activation checkpoint lookup",
            redactor=self._secret_redactor,
        )
        checkpoint_validation = capture_work_attempt_checkpoint_result(
            raw_checkpoint,
            operation_name="Work-attempt activation checkpoint lookup",
            redactor=self._secret_redactor,
        )
        del raw_checkpoint
        if checkpoint_validation.failure is not None:
            raise_task_store_operation_failure(checkpoint_validation.failure)
        checkpoint = checkpoint_validation.result
        del checkpoint_validation
        active_profile = active_invocation_execution_profile_from_checkpoint(checkpoint)
        active_model_completion = await read_work_attempt_session_store(
            lambda: self.session_store.load_active_model_completion_stage(session.id),
            operation_name="Work-attempt active model-stage lookup",
            redactor=self._secret_redactor,
        )
        has_active_model_completion = active_model_completion is not None
        del active_model_completion
        if (
            session.id != admission.session_id
            or session.status is not SessionStatus.RUNNING
            or SessionInvocationBinding(
                id=session.id,
                session_instance_id=session.instance_id,
                invocation=session.invocation,
            )
            != admission.session_invocation
            or stored_profile != execution_profile
            or execution_profile.fingerprint != admission.source_execution_profile_fingerprint
            or (
                admission.kind == "initial"
                and (
                    deferred_authority is None
                    or deferred_authority != (admission.interaction_id, True)
                )
            )
            or active_profile is None
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
            or active_profile.interaction_id != admission.interaction_id
            or active_profile.profile != execution_profile
            or has_active_model_completion
        ):
            raise WorkAttemptRecoveryRequired(
                "Session store did not publish the exact work-attempt interaction authority."
            )
        if admission.state not in {
            WorkAttemptAdmissionState.PREPARING,
            WorkAttemptAdmissionState.ACTIVE,
        }:
            raise WorkAttemptRecoveryRequired(
                "Only prepared or exact active admission authority can be activated."
            )
        evidence = hashlib.sha256(
            canonical_durable_json_bytes(
                {
                    "admission_id": admission.admission_id,
                    "session_id": session.id,
                    "interaction_id": admission.interaction_id,
                    "run_epoch": session.run_epoch,
                    "invocation": session.invocation.model_dump(mode="json", warnings=False),
                    "execution_profile_fingerprint": execution_profile.fingerprint,
                    "session_updated_at": session.updated_at.isoformat(),
                    "session_last_activity_at": session.last_activity_at.isoformat(),
                    "checkpoint_sha256": hashlib.sha256(
                        canonical_durable_json_bytes(
                            checkpoint,
                            "work_attempt_activation_checkpoint",
                        )
                    ).hexdigest(),
                },
                "work_attempt_session_evidence",
            )
        ).hexdigest()
        if (
            admission.state is WorkAttemptAdmissionState.ACTIVE
            and admission.session_evidence_sha256 != evidence
        ):
            raise WorkAttemptRecoveryRequired(
                "Active admission conflicts with its exact session activation evidence."
            )
        # Session mutation already owns the durable run epoch. Publish its
        # task-local interaction authority before the receipt write so an
        # acknowledgement-lost activation can be retried without losing event
        # attribution in the owning task.
        _activate_session_run_fence(session)
        _activate_session_interaction(session.id, admission.interaction_id)
        activation = WorkAttemptAdmissionActivate(
            admission_id=admission.admission_id,
            claim_id=admission.claim.claim_id,
            prepare_request_sha256=admission.prepare_request_sha256,
            session_evidence_sha256=evidence,
        )
        outcome = await capture_task_store_operation(
            lambda: task_store.activate_work_attempt_admission(activation),
            operation_name="Work-attempt admission activation",
            redactor=self._secret_redactor,
            mutation_store=task_store,
            mutation_method_name="activate_work_attempt_admission",
        )
        if outcome.failure is not None:
            raise_task_store_operation_failure(outcome.failure)
        validation = capture_sensitive_result_validation(
            lambda value=outcome.result, prepared=admission: require_work_attempt_activation_result(
                value,
                prepared,
                activation,
            ),
            operation_name="Work-attempt admission activation result validation",
            redactor=self._secret_redactor,
        )
        del outcome, admission
        if validation.failure is not None:
            raise_task_store_operation_failure(validation.failure)
        active = validation.result
        del validation
        if active is None:
            raise RuntimeError("Work-attempt admission activation returned no authority.")
        return await self.fan_out_work_attempt_interaction_started(
            active,
            lease_seconds=lease_seconds,
        )

    async def _renew_work_attempt_handoff_claim(
        self,
        admission: WorkAttemptAdmission,
        *,
        lease_seconds: int,
    ) -> WorkAttemptAdmission:
        """Refresh the exact active claim while its admission handoff is owned."""

        task_store = self.task_store
        if task_store is None:
            raise RuntimeError("task_store is required for work-attempt handoff renewal.")
        claim = admission.claim
        request = WorkAttemptExecutionClaimRequest(
            admission_id=admission.admission_id,
            claim_id=claim.claim_id,
            worker_id=claim.worker_id,
            execution_owner_id=claim.execution_owner_id,
            generation=claim.generation,
            lease_seconds=lease_seconds,
        )
        outcome = await capture_task_store_operation(
            lambda: task_store.renew_work_attempt_execution_claim(request),
            operation_name="Work-attempt interaction-handoff claim renewal",
            redactor=self._secret_redactor,
            mutation_store=task_store,
            mutation_method_name="renew_work_attempt_execution_claim",
        )
        if outcome.failure is not None:
            raise_task_store_operation_failure(outcome.failure)
        validation = capture_sensitive_result_validation(
            lambda value=outcome.result, previous=admission: require_work_attempt_claim_result(
                value,
                previous,
                request,
                operation_name="Work-attempt interaction-handoff claim renewal",
                allowed_states=frozenset({WorkAttemptAdmissionState.ACTIVE}),
                allowed_previous_states=frozenset({WorkAttemptAdmissionState.ACTIVE}),
                renewal=True,
            ),
            operation_name="Work-attempt interaction-handoff renewal result validation",
            redactor=self._secret_redactor,
        )
        del outcome, admission
        if validation.failure is not None:
            raise_task_store_operation_failure(validation.failure)
        renewed = validation.result
        del validation
        if renewed is None:
            raise RuntimeError("Work-attempt interaction-handoff renewal returned no authority.")
        return renewed

    async def _require_live_work_attempt_handoff_claim(
        self,
        admission: WorkAttemptAdmission,
        *,
        lease_seconds: int,
    ) -> WorkAttemptAdmission:
        """Reconcile the exact claim after handoff without extending a fast replay."""

        task_store = self.task_store
        if task_store is None:
            raise RuntimeError("task_store is required for work-attempt handoff reconciliation.")
        claim = admission.claim
        request = WorkAttemptExecutionClaimRequest(
            admission_id=admission.admission_id,
            claim_id=claim.claim_id,
            worker_id=claim.worker_id,
            execution_owner_id=claim.execution_owner_id,
            generation=claim.generation,
            lease_seconds=lease_seconds,
        )
        outcome = await capture_task_store_operation(
            lambda: task_store.claim_work_attempt_recovery(request),
            operation_name="Work-attempt interaction-handoff claim reconciliation",
            redactor=self._secret_redactor,
            mutation_store=task_store,
            mutation_method_name="claim_work_attempt_recovery",
        )
        if outcome.failure is not None:
            raise_task_store_operation_failure(outcome.failure)
        validation = capture_sensitive_result_validation(
            lambda value=outcome.result, previous=admission: require_work_attempt_claim_result(
                value,
                previous,
                request,
                operation_name="Work-attempt interaction-handoff claim reconciliation",
                allowed_states=frozenset({WorkAttemptAdmissionState.ACTIVE}),
                allowed_previous_states=frozenset({WorkAttemptAdmissionState.ACTIVE}),
                renewal=True,
            ),
            operation_name=(
                "Work-attempt interaction-handoff claim reconciliation result validation"
            ),
            redactor=self._secret_redactor,
        )
        del outcome, admission
        if validation.failure is not None:
            raise_task_store_operation_failure(validation.failure)
        current = validation.result
        del validation
        if current is None:
            raise RuntimeError(
                "Work-attempt interaction-handoff reconciliation returned no authority."
            )
        return current

    async def fan_out_work_attempt_interaction_started(
        self,
        admission: WorkAttemptAdmission,
        *,
        lease_seconds: int,
    ) -> WorkAttemptAdmission:
        """Deliver the persisted event without spending the returned execution lease."""

        async def finish_interaction_started_handoff() -> list[Event]:
            raw_records = await read_work_attempt_session_store(
                lambda: self.session_store.query_events(
                    EventQuery(
                        session_id=admission.session_id,
                        interaction_id=admission.interaction_id,
                        event_type=EventType.INTERACTION_STARTED,
                        limit=2,
                    )
                ),
                operation_name="Work-attempt interaction-event lookup",
                redactor=self._secret_redactor,
            )
            records_validation = capture_work_attempt_event_records_result(
                raw_records,
                operation_name="Work-attempt interaction-event lookup",
                redactor=self._secret_redactor,
            )
            del raw_records
            if records_validation.failure is not None:
                raise_task_store_operation_failure(records_validation.failure)
            records = records_validation.result
            del records_validation
            if records is None:
                raise RuntimeError("Work-attempt interaction-event lookup returned no records.")
            if (
                len(records) != 1
                or records[0].event.session_id != admission.session_id
                or records[0].event.interaction_id != admission.interaction_id
                or records[0].event.type != EventType.INTERACTION_STARTED
            ):
                raise WorkAttemptRecoveryRequired(
                    "Work-attempt admission has no unique durable interaction-start event."
                )
            interaction_started = records[0].event
            delivered = await self._event_writer.fan_out_persisted([interaction_started])
            while True:
                delivery = await read_work_attempt_session_store(
                    lambda: self.session_store.get_persisted_event_side_effect_delivery(
                        session_id=interaction_started.session_id,
                        event_id=interaction_started.id,
                    ),
                    operation_name="Work-attempt interaction-event delivery lookup",
                    redactor=self._secret_redactor,
                )
                if delivery is None:
                    raise WorkAttemptRecoveryRequired(
                        "Work-attempt interaction event has no durable delivery authority."
                    )
                if delivery.status not in {
                    PersistedEventSideEffectStatus.PENDING,
                    PersistedEventSideEffectStatus.LEASED,
                }:
                    return delivered
                await asyncio.sleep(0.01)
                delivered = await self._event_writer.fan_out_persisted([interaction_started])

        async def finish_live_interaction_handoff() -> WorkAttemptAdmission:
            current = admission
            handoff = asyncio.create_task(
                capture_awaitable_outcome(finish_interaction_started_handoff),
                name="cayu-work-attempt-interaction-event-delivery",
            )
            renewal_interval = min(
                _WORK_ATTEMPT_HANDOFF_RENEWAL_MAX_INTERVAL_SECONDS,
                max(
                    _WORK_ATTEMPT_HANDOFF_RENEWAL_MIN_INTERVAL_SECONDS,
                    lease_seconds / 3,
                ),
            )
            # Give an ordinary in-process handoff a short grace period to finish
            # without changing its exact replay receipt. A handoff that remains
            # pending receives an early renewal before settling into the regular
            # heartbeat cadence; this also protects a live claim that was already
            # close to expiry when an exact replay entered the handoff.
            next_renewal_delay = min(
                _WORK_ATTEMPT_HANDOFF_RENEWAL_MIN_INTERVAL_SECONDS,
                renewal_interval,
            )
            renewal_failure: BaseException | None = None
            while not handoff.done():
                completed, _pending = await asyncio.wait(
                    {handoff},
                    timeout=next_renewal_delay,
                )
                if completed:
                    break
                try:
                    current = await self._renew_work_attempt_handoff_claim(
                        current,
                        lease_seconds=lease_seconds,
                    )
                except BaseException as failure:
                    renewal_failure = failure
                    break
                next_renewal_delay = renewal_interval
            handoff_outcome = await handoff
            handoff_failure = handoff_outcome.error
            if renewal_failure is not None and handoff_failure is not None:
                raise BaseExceptionGroup(
                    "Work-attempt interaction handoff and claim renewal failed.",
                    [renewal_failure, handoff_failure],
                )
            if renewal_failure is not None:
                raise renewal_failure
            if handoff_failure is not None:
                raise handoff_failure
            return await self._require_live_work_attempt_handoff_claim(
                current,
                lease_seconds=lease_seconds,
            )

        return await settle_work_attempt_owned_operation(
            finish_live_interaction_handoff,
            operation_name="work-attempt-interaction-event-handoff",
            preserved_failure_types=(
                WorkAttemptAdmissionConflict,
                WorkAttemptExecutionClaimLost,
                WorkAttemptRecoveryRequired,
            ),
            redactor=self._secret_redactor,
        )

    async def _reconcile_runtime_work_attempt(
        self,
        authority: _WorkAttemptRuntimeAuthority,
        *,
        expected_task_id: str | None,
        expected_session_id: str | None,
    ) -> WorkAttemptAdmission | None:
        task_store = self.task_store
        if task_store is None:
            return None
        if not task_store_work_attempt_admission_capability_is_complete(task_store):
            raise NotImplementedError(
                f"{type(task_store).__name__} does not implement the complete "
                "work-attempt admission contract."
            )
        stable = authority.request
        outcome = await capture_task_store_operation(
            lambda: task_store.load_work_attempt_admission(stable.admission_id),
            operation_name="Work-attempt admission reconciliation lookup",
            redactor=self._secret_redactor,
        )
        if outcome.failure is not None:
            raise_task_store_operation_failure(outcome.failure)
        admission = outcome.result
        if admission is None:
            return None
        admission_validation = capture_sensitive_result_validation(
            lambda value=admission: require_work_attempt_admission_result(
                value,
                operation_name="Work-attempt admission reconciliation lookup",
            ),
            operation_name="Work-attempt admission reconciliation validation",
            redactor=self._secret_redactor,
        )
        del outcome, admission
        if admission_validation.failure is not None:
            raise_task_store_operation_failure(admission_validation.failure)
        admission = admission_validation.result
        del admission_validation
        if admission is None:
            raise RuntimeError("Work-attempt admission reconciliation returned no authority.")
        if (
            admission.attempt_id != stable.attempt_id
            or admission.interaction_id != stable.interaction_id
            or admission.kind != authority.kind
            or admission.source_request_sha256 != authority.source_request_sha256
            or admission.source_request != authority.source_request
            or (expected_task_id is not None and admission.task_id != expected_task_id)
            or (expected_session_id is not None and admission.session_id != expected_session_id)
            or (
                authority.kind == "continuation"
                and (
                    admission.continuation is None
                    or stable.task_id != admission.task_id
                    or stable.predecessor_admission_id != admission.continuation.prior_admission_id
                )
            )
        ):
            raise WorkCompletionConflict(
                "Work-attempt admission identity conflicts with this runtime request."
            )
        if admission.state is WorkAttemptAdmissionState.RELEASED:
            raise WorkAttemptExecutionClaimLost(
                "The work-attempt admission has already released execution authority."
            )
        exact_loaded_claim = (
            admission.claim.claim_id == stable.claim_id
            and admission.claim.worker_id == stable.worker_id
            and admission.claim.execution_owner_id == authority.execution_owner_id
            and admission.claim.generation == stable.generation
        )
        if admission.state is not WorkAttemptAdmissionState.PREPARING and not (
            admission.state is WorkAttemptAdmissionState.ACTIVE and exact_loaded_claim
        ):
            raise WorkAttemptRecoveryRequired(
                "Published work-attempt authority must be replaced through the "
                "dedicated recovery entrance."
            )
        claim_request = WorkAttemptExecutionClaimRequest(
            admission_id=stable.admission_id,
            claim_id=stable.claim_id,
            worker_id=stable.worker_id,
            execution_owner_id=authority.execution_owner_id,
            generation=stable.generation,
            lease_seconds=stable.lease_seconds,
        )
        claim_outcome = await capture_task_store_operation(
            lambda: task_store.claim_work_attempt_recovery(claim_request),
            operation_name="Work-attempt admission claim reconciliation",
            redactor=self._secret_redactor,
            mutation_store=task_store,
            mutation_method_name="claim_work_attempt_recovery",
        )
        if claim_outcome.failure is not None:
            raise_task_store_operation_failure(claim_outcome.failure)
        claim_validation = capture_sensitive_result_validation(
            lambda value=claim_outcome.result, previous=admission: (
                require_work_attempt_claim_result(
                    value,
                    previous,
                    claim_request,
                    operation_name="Work-attempt admission claim reconciliation",
                    allowed_states=frozenset(
                        {
                            WorkAttemptAdmissionState.PREPARING,
                            WorkAttemptAdmissionState.ACTIVE,
                        }
                    ),
                    allowed_previous_states=frozenset(
                        {
                            WorkAttemptAdmissionState.PREPARING,
                            WorkAttemptAdmissionState.ACTIVE,
                        }
                    ),
                )
            ),
            operation_name="Work-attempt admission claim reconciliation result validation",
            redactor=self._secret_redactor,
        )
        del claim_outcome, admission
        if claim_validation.failure is not None:
            raise_task_store_operation_failure(claim_validation.failure)
        claimed = claim_validation.result
        del claim_validation
        if claimed is None:
            raise RuntimeError("Work-attempt admission claim reconciliation returned no authority.")
        admission = claimed
        del claimed
        if admission.state is WorkAttemptAdmissionState.ACTIVE:
            raw_session = await read_work_attempt_session_store(
                lambda: self.session_store.load(admission.session_id),
                operation_name="Active work-attempt session lookup",
                redactor=self._secret_redactor,
            )
            if raw_session is None:
                raise WorkAttemptRecoveryRequired(
                    "Active work-attempt admission has no durable session."
                )
            session_validation = capture_work_attempt_session_result(
                raw_session,
                operation_name="Active work-attempt session lookup",
                redactor=self._secret_redactor,
            )
            del raw_session
            if session_validation.failure is not None:
                raise_task_store_operation_failure(session_validation.failure)
            session = session_validation.result
            del session_validation
            if session is None:
                raise RuntimeError("Active work-attempt session lookup returned no session.")
            profile = execution_profile_from_session_metadata(session.metadata)
            if (
                SessionInvocationBinding(
                    id=session.id,
                    session_instance_id=session.instance_id,
                    invocation=session.invocation,
                )
                != admission.session_invocation
                or profile is None
                or profile.fingerprint != admission.source_execution_profile_fingerprint
            ):
                raise WorkAttemptRecoveryRequired(
                    "Active work-attempt receipt conflicts with immutable session authority."
                )
            # Mutable transcript, checkpoint, timestamps, and terminal state do
            # not participate in exact receipt identity. Rebuild process-local
            # fencing only when the same durable invocation is still running.
            raw_checkpoint = await read_work_attempt_session_store(
                lambda session_id=session.id: self.session_store.load_checkpoint(session_id),
                operation_name="Active work-attempt checkpoint lookup",
                redactor=self._secret_redactor,
            )
            checkpoint_validation = capture_work_attempt_checkpoint_result(
                raw_checkpoint,
                operation_name="Active work-attempt checkpoint lookup",
                redactor=self._secret_redactor,
            )
            del raw_checkpoint
            if checkpoint_validation.failure is not None:
                raise_task_store_operation_failure(checkpoint_validation.failure)
            checkpoint = checkpoint_validation.result
            del checkpoint_validation
            active_profile = active_invocation_execution_profile_from_checkpoint(checkpoint)
            if (
                session.status is SessionStatus.RUNNING
                and active_profile is not None
                and active_profile.interaction_id == admission.interaction_id
                and active_profile.profile == profile
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
            ):
                _activate_session_run_fence(session)
                _activate_session_interaction(session.id, admission.interaction_id)
            return await self.fan_out_work_attempt_interaction_started(
                admission,
                lease_seconds=stable.lease_seconds,
            )
        if admission.state is not WorkAttemptAdmissionState.PREPARING:
            raise WorkAttemptRecoveryRequired(
                "Work-attempt admission is already owned by session recovery."
            )
        raw_session = await read_work_attempt_session_store(
            lambda: self.session_store.load(admission.session_id),
            operation_name="Prepared work-attempt session lookup",
            redactor=self._secret_redactor,
        )
        if raw_session is None:
            if admission.kind == "continuation":
                raise WorkAttemptRecoveryRequired(
                    "Prepared continuation authority has no source session."
                )
            return WorkAttemptAdmission.model_validate(
                admission.model_dump(mode="python", warnings=False)
            )
        session_validation = capture_work_attempt_session_result(
            raw_session,
            operation_name="Prepared work-attempt session lookup",
            redactor=self._secret_redactor,
        )
        del raw_session
        if session_validation.failure is not None:
            raise_task_store_operation_failure(session_validation.failure)
        session = session_validation.result
        del session_validation
        if session is None:
            raise RuntimeError("Prepared work-attempt session lookup returned no session.")
        profile = execution_profile_from_session_metadata(session.metadata)
        raw_deferred_input = await read_work_attempt_session_store(
            lambda session_id=session.id: self.session_store.load_deferred_interaction_input(
                session_id
            ),
            operation_name="Prepared work-attempt deferred-input lookup",
            redactor=self._secret_redactor,
        )
        deferred_validation = capture_work_attempt_deferred_input_result(
            raw_deferred_input,
            operation_name="Prepared work-attempt deferred-input lookup",
            redactor=self._secret_redactor,
        )
        del raw_deferred_input
        if deferred_validation.failure is not None:
            raise_task_store_operation_failure(deferred_validation.failure)
        deferred_authority = deferred_validation.result
        del deferred_validation
        if (
            profile is None
            or profile.fingerprint != admission.source_execution_profile_fingerprint
            or SessionInvocationBinding(
                id=session.id,
                session_instance_id=session.instance_id,
                invocation=session.invocation,
            )
            != admission.session_invocation
        ):
            raise WorkAttemptRecoveryRequired(
                "Prepared work-attempt admission has ambiguous session authority."
            )
        if (
            admission.kind == "continuation"
            and session.status in _session_checkpoint_admission._RESUMABLE_SESSION_STATUSES
        ):
            return WorkAttemptAdmission.model_validate(
                admission.model_dump(mode="python", warnings=False)
            )
        if session.status is not SessionStatus.RUNNING or (
            admission.kind == "initial"
            and (
                deferred_authority is None or deferred_authority != (admission.interaction_id, True)
            )
        ):
            raise WorkAttemptRecoveryRequired(
                "Prepared work-attempt admission has ambiguous interaction authority."
            )
        return await self._activate_runtime_work_attempt(
            admission=admission,
            session=session,
            execution_profile=profile,
            lease_seconds=stable.lease_seconds,
        )

    async def _load_work_attempt_execution_snapshot(
        self, admission: WorkAttemptAdmission
    ) -> tuple[Session, dict[str, Any] | None]:
        """Read detached invocation state, never infer authority from its status."""
        raw_session = await read_work_attempt_session_store(
            lambda: self.session_store.load(admission.session_id),
            operation_name="Work-attempt execution session lookup",
            redactor=self._secret_redactor,
        )
        session_validation = capture_work_attempt_session_result(
            raw_session,
            operation_name="Work-attempt execution session lookup",
            redactor=self._secret_redactor,
        )
        del raw_session
        if session_validation.failure is not None:
            raise_task_store_operation_failure(session_validation.failure)
        session = session_validation.result
        if (
            session is None
            or SessionInvocationBinding(
                id=session.id,
                session_instance_id=session.instance_id,
                invocation=session.invocation,
            )
            != admission.session_invocation
        ):
            raise WorkAttemptRecoveryRequired("Work-attempt invocation identity changed.")
        raw_checkpoint = await read_work_attempt_session_store(
            lambda session_id=session.id: self.session_store.load_checkpoint(session_id),
            operation_name="Work-attempt execution checkpoint lookup",
            redactor=self._secret_redactor,
        )
        checkpoint_validation = capture_work_attempt_checkpoint_result(
            raw_checkpoint,
            operation_name="Work-attempt execution checkpoint lookup",
            redactor=self._secret_redactor,
        )
        del raw_checkpoint
        if checkpoint_validation.failure is not None:
            raise_task_store_operation_failure(checkpoint_validation.failure)
        return session, checkpoint_validation.result

    async def has_recoverable_work_attempt_cleanup(self, admission: WorkAttemptAdmission) -> bool:
        """Advisory cleanup eligibility, explicitly separate from execution replay.

        A durable stop forbids redispatch even when its acknowledgement was
        lost before finalization or terminal publication. Limit stops additionally
        require no unresolved model, tool, human, or durable-operation boundary.
        The elected recovery owner still validates and settles the checkpoint.
        """
        stop = admission.execution_stop
        entry = admission.execution_entry
        if stop is None or entry is None:
            return False
        session, checkpoint = await self._load_work_attempt_execution_snapshot(admission)
        active = active_invocation_execution_profile_from_checkpoint(checkpoint)
        marker = completion_finalization.pending_completion_finalization_from_checkpoint(checkpoint)
        if not (
            active is not None
            and active_invocation_execution_profile_matches_session_epoch(
                active, session_id=session.id, run_epoch=session.run_epoch
            )
            and active.run_epoch >= entry.request.run_epoch
            and active.interaction_id == admission.interaction_id
            and active.profile.fingerprint == admission.source_execution_profile_fingerprint
        ):
            return False
        if stop.request.reason == "workspace_finalization_recovery":
            return active.run_epoch > entry.request.run_epoch and (
                marker is None
                or (
                    marker.get("task_id") == admission.task_id
                    and marker.get("execution_profile_fingerprint")
                    == admission.source_execution_profile_fingerprint
                )
            )
        if stop.request.reason not in {"budget_limit", "elapsed_limit"}:
            return False
        # A native non-success marker retains binding cleanup, not permission
        # to continue execution. Completed governed work has a different owner.
        if marker is not None and not (
            marker["outcome"] in {"failed", "interrupted"}
            and marker["execution_profile_fingerprint"]
            == admission.source_execution_profile_fingerprint
            and marker.get("task_id") is None
        ):
            return False
        # Stop intent is not quiescence. Model-stage publication and the existing
        # checkpoint guards retain uncertain external work with its current owner.
        if (
            await self._model_completion_recovery.load_model_completion_boundary(session)
            is not None
        ):
            return False
        try:
            _session_checkpoint_admission._reject_unresumable_session_checkpoint(
                session,
                checkpoint,
                redactor=self._secret_redactor,
                allowed_initial_transcript_interaction_id=(
                    admission.interaction_id if admission.kind == "initial" else None
                ),
            )
        except RuntimeError:
            return False
        return True

    async def has_recoverable_work_attempt_model_result(
        self, admission: WorkAttemptAdmission
    ) -> bool:
        """Read-only scanner eligibility, never a replacement claim or dispatch grant.

        Terminal model, closed tool publications, and exact background-operation
        evidence can enter the recovery owner.
        Unknown dispatches and pending tool/human work retain their current fence.
        The claim owner must revalidate all state before any recovery mutation.
        """
        if admission.execution_entry is None or admission.execution_stop is not None:
            return False
        session, checkpoint = await self._load_work_attempt_execution_snapshot(admission)
        active = await self._model_completion_recovery.load_model_completion_boundary(session)
        if active is not None:
            stage = active.stage
            if stage.state == "in_flight":
                recovery_context = model_completion_recovery_context_from_stage(stage)
                return (
                    recovery_context is not None
                    and recovery_context.execution_profile_fingerprint
                    == admission.source_execution_profile_fingerprint
                    and stage.source_run_epoch <= session.run_epoch
                    and await self._model_completion_recovery.has_recoverable_provider_operation(
                        stage
                    )
                )
            if stage.state != "completed" or stage.publication is None:
                return False
            published_checkpoint = {
                operation.key: operation.value
                for operation in stage.publication.mutation.operations
                if operation.action == "set"
            }
            pointer = model_completion_publication.model_step_publication_from_checkpoint(
                published_checkpoint
            )
        else:
            published_checkpoint = checkpoint
            pointer = model_completion_publication.model_step_publication_from_checkpoint(
                checkpoint
            )
            if pointer is None:
                return False
            stage = await self.session_store.load_model_completion_stage(
                session.id, pointer.stage_id
            )
        if stage is None or pointer is None:
            return False
        recovery_context = model_completion_recovery_context_from_stage(stage)
        if (
            recovery_context is None
            or recovery_context.execution_profile_fingerprint
            != admission.source_execution_profile_fingerprint
            or stage.source_run_epoch > session.run_epoch
        ):
            return False
        if pointer.tool_round_id is not None:
            pending_round = pending_round_reader.pending_tool_round_from_checkpoint(
                published_checkpoint
            )
            if pending_round is not None:
                reconstruct_assistant_step_result(
                    stage=stage,
                    pointer=pointer,
                    pending_round=pending_round,
                    session_id=admission.session_id,
                    interaction_id=admission.interaction_id,
                    source_run_epoch=stage.source_run_epoch,
                )
                if has_recoverable_structured_output_round(pending_round):
                    return True
                if active is not None:
                    return False
                return await self._pending_tool_round_recovery.has_completed_tool_round_results(
                    session_id=session.id, pending_round=pending_round
                )
            if active is not None:
                return False
            receipt = await self.session_store.load_runtime_publication_receipt(
                session.id, f"tool-round:{pointer.tool_round_id}"
            )
            # Advisory readiness only. The elected recovery coordinator owns
            # exact receipt/call/transcript validation before any effects.
            return (
                receipt is not None
                and await self._pending_tool_round_recovery.has_settled_published_tool_round_results(
                    session_id=session.id, receipt=receipt
                )
            )
        reconstruct_assistant_step_result(
            stage=stage,
            pointer=pointer,
            pending_round=None,
            session_id=admission.session_id,
            interaction_id=admission.interaction_id,
            source_run_epoch=stage.source_run_epoch,
        )
        cursor = await self.session_store.load_transcript_cursor(session.id)
        return cursor == (
            pointer.source_transcript_cursor
            if active is not None
            else pointer.transcript_end_cursor
        )

    async def settle_work_attempt_admission_handoff(self, admission: WorkAttemptAdmission) -> bool:
        return await self._settle_pre_entry_event_handoff(
            admission, event_type=EventType.INTERACTION_STARTED
        )

    async def close_unentered_work_attempt_invocation(
        self, admission: WorkAttemptAdmission
    ) -> InvocationReleaseEvidence:
        """Close the exact admitted interaction without entering its execution.

        The caller owns a group-cancelled, content-validated admission. Keep
        that admission discoverable until this cleanup and release have both
        committed; the TaskStore lifecycle receipt is published afterwards.
        """
        if admission.execution_entry is not None or admission.execution_stop is not None:
            raise WorkAttemptRecoveryRequired("Pre-entry cleanup cannot close entered work.")
        session, checkpoint = await self._load_work_attempt_execution_snapshot(admission)
        if not await self.settle_work_attempt_admission_handoff(admission):
            raise WorkAttemptRecoveryRequired("Pre-entry admission handoff is not settled.")
        active = active_invocation_execution_profile_from_checkpoint(checkpoint)
        if (
            active is None
            or active.session_id != session.id
            or active.interaction_id != admission.interaction_id
            or active.profile.fingerprint != admission.source_execution_profile_fingerprint
        ):
            raise WorkAttemptRecoveryRequired("Pre-entry cleanup lost its invocation profile.")
        if active_invocation_execution_profile_is_released(
            active, session_id=session.id, run_epoch=session.run_epoch
        ):
            return released_invocation_evidence(
                session,
                checkpoint,
                session_id=admission.session_id,
                session_instance_id=admission.session_invocation.session_instance_id,
                active_profile=active,
            )
        if active.run_epoch != session.run_epoch:
            raise WorkAttemptRecoveryRequired("Pre-entry cleanup lost its invocation epoch.")
        if session.status not in {
            SessionStatus.RUNNING,
            SessionStatus.INTERRUPTING,
            SessionStatus.INTERRUPTED,
        }:
            raise WorkAttemptRecoveryRequired(
                "Pre-entry cleanup has a conflicting session outcome."
            )
        stop_id = "pre-entry:" + hashlib.sha256(admission.admission_id.encode()).hexdigest()
        if session.status is not SessionStatus.RUNNING:
            decision = invocation_terminal_decision_from_checkpoint(
                checkpoint
            ) or settled_invocation_terminal_decision_from_checkpoint(checkpoint)
            marker = (checkpoint or {}).get(_PENDING_SESSION_INTERRUPT_CHECKPOINT_KEY)
            retained_stop_id = (
                decision.interruption_request_id
                if decision is not None
                else marker.get("interruption_request_id")
                if type(marker) is dict
                else None
            )
            if retained_stop_id != stop_id:
                # Another interruption owner may still be running terminal
                # hooks. Its terminal status is not proof of owner return.
                raise WorkAttemptRecoveryRequired("Pre-entry cleanup belongs to another stop.")
        context = await self._resolve_work_attempt_invocation_context(
            admission, session=session, checkpoint=checkpoint, require_open_interaction=False
        )
        _activate_session_run_fence(session)
        _activate_session_interaction(session.id, admission.interaction_id)
        try:
            if admission.kind == "initial":
                deferred = await read_work_attempt_session_store(
                    lambda: self.session_store.load_deferred_interaction_input(session.id),
                    operation_name="Pre-entry initial transcript lookup",
                    redactor=self._secret_redactor,
                )
                if deferred is not None:
                    validated = capture_work_attempt_execution_input_result(
                        [], deferred, admission=admission, redactor=self._secret_redactor
                    )
                    if validated.failure is not None:
                        raise_task_store_operation_failure(validated.failure)
                    prepared = validated.result
                    assert prepared is not None
                    await self.session_store.replace_initial_transcript_messages(
                        session.id,
                        list(prepared.messages_to_append),
                        list(prepared.messages),
                    )
                elif _initial_transcript_pending_interaction_id(checkpoint) is not None:
                    raise WorkAttemptRecoveryRequired("Pre-entry initial transcript is incomplete.")
            else:
                await self.session_store.materialize_deferred_interaction_input(session.id)
            if session.status is SessionStatus.RUNNING:
                payload = {
                    "interruption_type": _INTERRUPTION_TYPE_RUNTIME_INTERRUPTED,
                    "interruption_request_id": stop_id,
                }

                def retain_stop(current, current_checkpoint):
                    if (
                        current.instance_id != session.instance_id
                        or current.run_epoch != active.run_epoch
                        or active_invocation_execution_profile_from_checkpoint(current_checkpoint)
                        != active
                        or (current_checkpoint or {}).get(
                            _PENDING_SESSION_INTERRUPT_CHECKPOINT_KEY, payload
                        )
                        != payload
                    ):
                        raise SessionRunFenced("Pre-entry interruption authority changed.")
                    return {
                        **(current_checkpoint or {}),
                        _PENDING_SESSION_INTERRUPT_CHECKPOINT_KEY: payload,
                    }

                # Enter the existing durable terminal-election protocol. Its
                # exact event identities arbitrate overlapping cleanup owners
                # and survive failure between interaction and session writes.
                with _invocation_lifecycle_authority_read_scope():
                    session = await self.session_store.transition_status_and_checkpoint(
                        session.id,
                        from_statuses={SessionStatus.RUNNING},
                        to_status=SessionStatus.INTERRUPTING,
                        checkpoint_transform=retain_stop,
                        require_no_active_model_completion_dispatch=True,
                    )
            if (
                session.status is not SessionStatus.INTERRUPTED
                or settled_invocation_terminal_decision_from_checkpoint(checkpoint) is None
            ):
                async for _event in self._session_finalization.handle_session_interrupted(
                    session=session,
                    registered_agent=context.registered_agent,
                    registered_environment=context.registered_environment,
                    environment_name=session.environment_name,
                    execution_profile=context.profile,
                    invocation_context=context,
                    run_terminal_hooks=False,
                ):
                    pass

            terminal_checkpoint = await self.session_store.load_checkpoint(session.id)
            terminal_decision = settled_invocation_terminal_decision_from_checkpoint(
                terminal_checkpoint
            )
            if (
                terminal_decision is None
                or terminal_decision.outcome is not InvocationTerminalOutcome.INTERRUPTED
                or terminal_decision.interruption_request_id != stop_id
                or terminal_decision.interaction_event_id is None
                or not invocation_terminal_decision_matches_active_profile(
                    terminal_decision,
                    session_id=session.id,
                    session_instance_id=session.instance_id,
                    run_epoch=active.run_epoch,
                    interaction_id=admission.interaction_id,
                    execution_profile_fingerprint=active.profile.fingerprint,
                )
            ):
                raise WorkAttemptRecoveryRequired("Pre-entry terminal handoff lost its authority.")
            for event_type, event_id in (
                (EventType.INTERACTION_INTERRUPTED, terminal_decision.interaction_event_id),
                (EventType.SESSION_INTERRUPTED, terminal_decision.terminal_event_id),
            ):
                if not await self._settle_pre_entry_event_handoff(
                    admission, event_type=event_type, event_id=event_id, dispatch_pending=True
                ):
                    raise WorkAttemptRecoveryRequired("Pre-entry terminal handoff is not settled.")

            # Paired terminal publication precedes clearing its stop marker.
            # Reconcile that tail even when a previous attempt already closed
            # the interaction. A release receipt must not strand an unfinished
            # marker, and terminal status never authorizes clearing another stop.
            def reconcile_stop(current, current_checkpoint):
                decision = settled_invocation_terminal_decision_from_checkpoint(current_checkpoint)
                if (
                    current.instance_id != session.instance_id
                    or current.run_epoch != active.run_epoch
                    or active_invocation_execution_profile_from_checkpoint(current_checkpoint)
                    != active
                    or decision is None
                    or decision.outcome is not InvocationTerminalOutcome.INTERRUPTED
                    or decision.interruption_request_id != stop_id
                    or not invocation_terminal_decision_matches_active_profile(
                        decision,
                        session_id=session.id,
                        session_instance_id=session.instance_id,
                        run_epoch=active.run_epoch,
                        interaction_id=admission.interaction_id,
                        execution_profile_fingerprint=active.profile.fingerprint,
                    )
                ):
                    raise SessionRunFenced("Pre-entry marker cleanup lost its terminal authority.")
                updated = dict(current_checkpoint or {})
                marker = updated.get(_PENDING_SESSION_INTERRUPT_CHECKPOINT_KEY)
                if _PENDING_SESSION_INTERRUPT_CHECKPOINT_KEY in updated and marker != {
                    "interruption_type": _INTERRUPTION_TYPE_RUNTIME_INTERRUPTED,
                    "interruption_request_id": stop_id,
                }:
                    raise SessionRunFenced("Pre-entry marker belongs to another interruption.")
                updated.pop(_PENDING_SESSION_INTERRUPT_CHECKPOINT_KEY, None)
                return updated

            with _invocation_lifecycle_authority_read_scope():
                await self.session_store.publish_checkpoint_and_events(
                    session.id,
                    checkpoint_transform=reconcile_stop,
                    events=[],
                    expected_statuses={SessionStatus.INTERRUPTED},
                    expected_run_epoch=active.run_epoch,
                )
            await self._environment_lifecycle.release_run_fence_after_environment_cleanup(
                session_id=session.id,
                execution_profile=context.profile,
                invocation_context=context,
            )
        finally:
            _deactivate_session_interaction(session.id)
            _deactivate_session_run_fence(session.id)
        session, checkpoint = await self._load_work_attempt_execution_snapshot(admission)
        return released_invocation_evidence(
            session,
            checkpoint,
            session_id=admission.session_id,
            session_instance_id=admission.session_invocation.session_instance_id,
            active_profile=context.active_profile,
        )

    async def load_work_attempt_release_evidence(
        self, admission: WorkAttemptAdmission
    ) -> InvocationReleaseEvidence:
        """Reconcile exact durable release without resolving live collaborators.

        The expected epoch comes from the durable execution entry, not the
        session's newer epoch. Terminal status alone never proves quiescence.
        This is an internal read boundary, not public caller authentication.
        """
        entry = admission.execution_entry
        if entry is None:
            raise WorkAttemptRecoveryRequired("Work-attempt execution has no entry to reconcile.")
        session, checkpoint = await self._load_work_attempt_execution_snapshot(admission)

        def project_release() -> InvocationReleaseEvidence:
            profile = execution_profile_from_session_metadata(session.metadata)
            if (
                profile is None
                or profile.fingerprint != admission.source_execution_profile_fingerprint
            ):
                raise WorkAttemptRecoveryRequired("Work-attempt source profile changed.")
            expected_epoch = entry.request.run_epoch
            if admission.execution_stop is not None:
                # The immutable stop forbids every later execution/proposal.
                # Later epochs can therefore prove cleanup of this same attempt,
                # but still require the ordinary exact release receipt below.
                active = active_invocation_execution_profile_from_checkpoint(checkpoint)
                workspace_stop = (
                    admission.execution_stop.request.reason == "workspace_finalization_recovery"
                )
                if (
                    active is None
                    or active.run_epoch < expected_epoch
                    or (workspace_stop and active.run_epoch == expected_epoch)
                    or session.status not in {SessionStatus.FAILED, SessionStatus.INTERRUPTED}
                    or (workspace_stop and session.status is not SessionStatus.FAILED)
                    or completion_finalization.pending_completion_finalization_from_checkpoint(
                        checkpoint
                    )
                    is not None
                ):
                    raise WorkAttemptRecoveryRequired("Stopped work-attempt cleanup is unproven.")
                expected_epoch = active.run_epoch
            return released_invocation_evidence(
                session,
                checkpoint,
                session_id=admission.session_id,
                session_instance_id=admission.session_invocation.session_instance_id,
                active_profile=ActiveInvocationExecutionProfile(
                    session_id=admission.session_id,
                    interaction_id=admission.interaction_id,
                    run_epoch=expected_epoch,
                    profile=profile,
                ),
            )

        validation = capture_sensitive_result_validation(
            project_release,
            operation_name="Work-attempt exact release validation",
            redactor=self._secret_redactor,
        )
        if validation.failure is not None:
            raise_task_store_operation_failure(validation.failure)
        if validation.result is None:
            raise WorkAttemptRecoveryRequired("Work-attempt invocation release is unproven.")
        return validation.result

    async def load_work_attempt_released_recovery_evidence(
        self, admission: WorkAttemptAdmission
    ) -> InvocationReleaseEvidence | None:
        """Read exact completed-execution proof without acquiring a generation.

        An incomplete invocation or a distinct recovery cleanup epoch remains
        with the incomplete-recovery owner. Invalid evidence is an error, not
        permission to dispatch or to consume a replacement claim by polling.
        """
        entry = admission.execution_entry
        if entry is None:
            return None
        session, checkpoint = await self._load_work_attempt_execution_snapshot(admission)
        active = active_invocation_execution_profile_from_checkpoint(checkpoint)
        if active is None:
            raise WorkAttemptRecoveryRequired("Entered work-attempt recovery lost its profile.")
        if (
            active.interaction_id != admission.interaction_id
            or active.profile.fingerprint != admission.source_execution_profile_fingerprint
            or active.run_epoch < entry.request.run_epoch
        ):
            raise WorkAttemptRecoveryRequired("Entered work-attempt recovery profile conflicts.")
        if active.run_epoch > entry.request.run_epoch and admission.execution_stop is None:
            # Incomplete recovery fences into its own cleanup epoch. Its
            # release cannot stand in for completion of the original run;
            # the existing recovery/rebind owner must reconcile that epoch.
            return None
        if not active_invocation_execution_profile_is_released(
            active, session_id=session.id, run_epoch=session.run_epoch
        ):
            return None
        if admission.execution_stop is not None and (
            session.status not in {SessionStatus.FAILED, SessionStatus.INTERRUPTED}
            or (
                admission.execution_stop.request.reason == "workspace_finalization_recovery"
                and session.status is not SessionStatus.FAILED
            )
            or completion_finalization.pending_completion_finalization_from_checkpoint(checkpoint)
            is not None
        ):
            return None
        return await self.load_work_attempt_release_evidence(admission)

    async def reconcile_released_work_attempt_recovery(
        self, admission: WorkAttemptAdmission
    ) -> WorkAttemptAdmission | None:
        """Activate settlement ownership without reopening completed execution."""
        entry = admission.execution_entry
        if entry is None or entry.request.generation >= admission.claim.generation:
            return None
        release = await self.load_work_attempt_released_recovery_evidence(admission)
        if release is None:
            return None
        evidence = hashlib.sha256(
            canonical_durable_json_bytes(
                {
                    "domain": "cayu:work-attempt:released-recovery:v1",
                    "prepare_request_sha256": admission.prepare_request_sha256,
                    "execution_entry": entry.model_dump(mode="json", warnings=False),
                    "claim": admission.claim.model_dump(
                        mode="json", warnings=False, exclude={"lease_expires_at", "renewal"}
                    ),
                    "release": release.model_dump(mode="json", warnings=False),
                    **(
                        {}
                        if admission.execution_stop is None
                        else {
                            "execution_stop": admission.execution_stop.model_dump(
                                mode="json", warnings=False
                            )
                        }
                    ),
                },
                "work_attempt_released_recovery",
            )
        ).hexdigest()
        request = WorkAttemptRecoveryActivate(
            admission_id=admission.admission_id,
            claim_id=admission.claim.claim_id,
            generation=admission.claim.generation,
            recovery_evidence_sha256=evidence,
        )
        task_store = self.task_store
        if task_store is None:
            raise RuntimeError("task_store is required for work-attempt recovery.")
        outcome = await capture_task_store_operation(
            lambda: task_store.activate_work_attempt_recovery(request),
            operation_name="Released work-attempt recovery activation",
            redactor=self._secret_redactor,
            mutation_store=task_store,
            mutation_method_name="activate_work_attempt_recovery",
        )
        if outcome.failure is not None:
            raise_task_store_operation_failure(outcome.failure)
        validation = capture_sensitive_result_validation(
            lambda value=outcome.result: require_work_attempt_recovery_activation_result(
                value, admission, request
            ),
            operation_name="Released work-attempt recovery result validation",
            redactor=self._secret_redactor,
        )
        del outcome
        if validation.failure is not None:
            raise_task_store_operation_failure(validation.failure)
        if validation.result is None:
            raise WorkAttemptRecoveryRequired("Released recovery returned no authority.")
        return validation.result

    async def execute_admitted_work_attempt(
        self,
        admission: WorkAttemptAdmission,
        *,
        lease_seconds: int,
    ) -> AsyncGenerator[Event, None]:
        """Own a governed run without ordinary task terminalization.

        App/worker authority resolution supplies the admission; this entrance
        never authenticates a caller's detached admission. Reads and input
        preparation precede election. After election, the existing common run
        owns deferred materialization, environment work and terminal cleanup.
        Any failure before that owner starts leaves the durable entry fenced
        for exact recovery, not available for an implicit second dispatch.
        """
        if admission.execution_stop is not None:
            raise WorkAttemptRecoveryRequired("Work-attempt execution is durably stopped.")
        if (
            admission.execution_entry is not None
            and admission.execution_entry.request.generation == admission.claim.generation
        ):
            raise WorkAttemptRecoveryRequired(
                "Work-attempt execution already entered; reconcile its exact invocation."
            )
        session, checkpoint = await self._load_work_attempt_execution_snapshot(admission)
        if session.status is not SessionStatus.RUNNING:
            raise WorkAttemptRecoveryRequired("Work-attempt invocation is not running.")
        raw_transcript = await read_work_attempt_session_store(
            lambda: self.session_store.load_transcript(session.id),
            operation_name="Work-attempt execution transcript lookup",
            redactor=self._secret_redactor,
        )
        raw_deferred = await read_work_attempt_session_store(
            lambda: self.session_store.load_deferred_interaction_input(session.id),
            operation_name="Work-attempt execution deferred-input lookup",
            redactor=self._secret_redactor,
        )
        input_validation = capture_work_attempt_execution_input_result(
            raw_transcript,
            raw_deferred,
            admission=admission,
            redactor=self._secret_redactor,
        )
        del raw_transcript, raw_deferred
        if input_validation.failure is not None:
            raise_task_store_operation_failure(input_validation.failure)
        prepared_input = input_validation.result
        if prepared_input is None:
            raise RuntimeError("Work-attempt execution has no validated input.")
        context = await self._enter_work_attempt_invocation_context(
            admission,
            session=session,
            checkpoint=checkpoint,
            lease_seconds=lease_seconds,
        )
        work_attempt = context.work_attempt
        if work_attempt is None:  # excluded by the election owner
            raise RuntimeError("Work-attempt execution lacks elected authority.")
        elected = work_attempt.admission
        semantics = elected.run_semantics
        if semantics is None:  # excluded by the context authority constructor
            raise RuntimeError("Work-attempt execution lacks source semantics.")
        _activate_session_run_fence(session)
        _activate_session_interaction(session.id, elected.interaction_id)
        try:
            stream = self._execute_session(
                session=session,
                invocation_context=context,
                messages=list(prepared_input.messages),
                messages_to_append=list(prepared_input.messages_to_append),
                messages_deferred=prepared_input.deferred,
                max_steps=semantics.max_steps,
                limits=semantics.limits,
                budget_limits=semantics.budget_limits,
                retry_policy=semantics.retry_policy,
                structured_output=semantics.structured_output,
                tool_completion=semantics.tool_completion,
                thinking=semantics.thinking,
                request_metadata=semantics.request_metadata,
                task_id=elected.task_id,
                task_worker_id=elected.claim.worker_id,
                task_handoff_id=None,
                start_event_type=None,
                start_event_payload={},
            )
            async with _close_delegated_event_stream(stream) as owned_stream:
                async for event in owned_stream:
                    yield event
        finally:
            # Process-local context tokens are not durable release evidence.
            # The common owner alone publishes release after owned cleanup;
            # an early setup failure leaves the durable entry for recovery.
            _deactivate_session_interaction(session.id)
            _deactivate_session_run_fence(session.id)

    async def _enter_work_attempt_invocation_context(
        self,
        admission: WorkAttemptAdmission,
        *,
        session: Session,
        checkpoint: dict[str, Any] | None,
        lease_seconds: int,
    ) -> InvocationContext:
        """Elect one original dispatcher after read-only profile preflight.

        The outer runtime owner resolves this process's claim before calling
        here. An uncertain mutation or duplicate entry remains durably fenced;
        neither readback nor a reconstructed response authorizes redispatch.
        """
        task_store = self.task_store
        if task_store is None:
            raise RuntimeError("task_store is required for work-attempt execution.")
        current = await self._require_live_work_attempt_handoff_claim(
            admission, lease_seconds=lease_seconds
        )
        if (
            current.execution_entry is not None
            and current.execution_entry.request.generation == current.claim.generation
        ):
            raise WorkAttemptRecoveryRequired(
                "Work-attempt execution already entered; reconcile its exact invocation."
            )
        context = await self._resolve_work_attempt_invocation_context(
            current, session=session, checkpoint=checkpoint
        )
        claim = current.claim
        request = WorkAttemptExecutionEntryRequest(
            admission_id=current.admission_id,
            prepare_request_sha256=current.prepare_request_sha256,
            claim_id=claim.claim_id,
            worker_id=claim.worker_id,
            execution_owner_id=claim.execution_owner_id,
            generation=claim.generation,
            run_epoch=session.run_epoch,
        )
        outcome = await capture_task_store_operation(
            lambda: task_store.enter_work_attempt_execution(request),
            operation_name="Work-attempt execution entry",
            redactor=self._secret_redactor,
            mutation_store=task_store,
            mutation_method_name="enter_work_attempt_execution",
        )
        if outcome.failure is not None:
            raise_task_store_operation_failure(outcome.failure)
        validation = capture_sensitive_result_validation(
            lambda value=outcome.result: require_work_attempt_execution_entry_result(
                value, current, request
            ),
            operation_name="Work-attempt execution-entry result validation",
            redactor=self._secret_redactor,
        )
        del outcome
        if validation.failure is not None:
            raise_task_store_operation_failure(validation.failure)
        result = validation.result
        if result is None:
            raise RuntimeError("Work-attempt execution entry returned no authority.")
        if result.disposition is not WorkAttemptExecutionEntryDisposition.ENTERED:
            raise WorkAttemptRecoveryRequired(
                "Work-attempt execution already entered; reconcile its exact invocation."
            )
        # No await separates the positive election response from constructing
        # the live context. Registered collaborators remain those preflighted
        # above; caller data and readback cannot supply the election response.
        return _authenticated_invocation_context(
            active_profile=context.active_profile,
            binding=context.binding,
            validated_profile=context.profile,
            registered_agent=context.registered_agent,
            registered_provider=context.registered_provider,
            registered_environment=context.registered_environment,
            runtime_hooks=context.runtime_hooks,
            loop_policies=context.loop_policies,
            request_loop_policies=context.request_loop_policies,
            budget_policy=context.budget_policy,
            tool_capability_ceiling=context.tool_capability_ceiling,
            work_attempt=_authenticated_work_attempt_invocation(result.admission),
        )

    async def _resolve_work_attempt_invocation_context(
        self,
        admission: WorkAttemptAdmission,
        *,
        session: Session,
        checkpoint: dict[str, Any] | None,
        require_open_interaction: bool = True,
    ) -> InvocationContext:
        """Resolve durable source settings through the existing profile owner.

        The calling execution/recovery owner must fence the live claim before
        using this context. This read-only resolver does not acquire ownership.
        """
        authority = _authenticated_work_attempt_invocation(admission)
        admitted = authority.admission
        semantics = admitted.run_semantics
        if semantics is None:  # excluded by the authority constructor
            raise WorkAttemptRecoveryRequired("Admission has no executable source settings.")
        if (
            session.id != admitted.session_id
            or session.instance_id != admitted.session_invocation.session_instance_id
        ):
            raise WorkAttemptExecutionClaimLost("Work-attempt session incarnation changed.")
        registered_agent = self._get_registered_agent(session.agent_name)
        registered_provider = self._get_registered_provider(session.provider_name)
        registered_environment = self._get_registered_environment_for_session(
            session.environment_name
        )
        budget_policy = self._get_budget_policy()
        active = await self._execution_profile_continuation.validate(
            session=session,
            checkpoint=checkpoint,
            registered_agent=registered_agent,
            registered_provider=registered_provider,
            request_loop_policies=(),
            budget_policy=budget_policy,
            request_budget_limits=semantics.budget_limits,
            structured_output=semantics.structured_output,
            tool_completion=semantics.tool_completion,
            thinking=semantics.thinking,
            max_steps=semantics.max_steps,
            limits=semantics.limits,
            retry_policy=semantics.retry_policy,
            invocation_semantics_available=True,
            record_rejection=False,
            require_open_interaction=require_open_interaction,
        )
        return _authenticated_invocation_context(
            active_profile=active,
            binding=AdmittedInvocationBinding(
                session_id=session.id,
                session_instance_id=session.instance_id,
                interaction_id=admitted.interaction_id,
                run_epoch=session.run_epoch,
                agent_name=session.agent_name,
                provider_name=session.provider_name,
                model=session.model,
                runtime_name=session.runtime_name,
                runtime_version=session.runtime_version,
                runtime_build_provenance=session.runtime_build_provenance,
                environment_name=session.environment_name,
            ),
            validated_profile=active.profile,
            registered_agent=registered_agent,
            registered_provider=registered_provider,
            registered_environment=registered_environment,
            runtime_hooks=self._runtime_hooks,
            loop_policies=self._loop_policies,
            request_loop_policies=(),
            budget_policy=budget_policy,
            tool_capability_ceiling=_session_execution_profile._session_tool_capability_ceiling(
                session
            ),
            work_attempt=authority,
        )

    async def admit_initial_work_attempt(
        self,
        request: RunRequest,
        *,
        authority: _WorkAttemptRuntimeAuthority,
    ) -> WorkAttemptAdmission:
        """Create and activate one initial governed interaction without dispatching it."""

        if request.parent_session_id is not None:
            raise TaskCompletionDecisionRequired(
                "Contracted session forks are rejected before work-attempt admission."
            )
        reconciled = await self._reconcile_runtime_work_attempt(
            authority,
            expected_task_id=request.task_id,
            expected_session_id=request.session_id,
        )
        if reconciled is not None and reconciled.state is WorkAttemptAdmissionState.ACTIVE:
            return reconciled
        prepared = await self._session_preparation.prepare_initial_run(
            request,
            admit_session=False,
            allow_work_attempt_admission=True,
            prepared_work_attempt=reconciled,
        )
        if prepared is None:
            raise RuntimeError("Work-attempt initial preparation unexpectedly failed closed.")
        if prepared.targeted_tool_grants:
            raise ValueError("Work-attempt admission cannot issue interaction-scoped tool grants.")
        prepared_request = prepared.request
        semantics_validation = capture_sensitive_result_validation(
            lambda prepared_request=prepared_request, prepared=prepared: WorkAttemptRunSemantics(
                max_steps=prepared_request.max_steps,
                limits=prepared_request.limits,
                budget_limits=prepared_request.budget_limits,
                causal_budget_id=(
                    prepared_request.causal_budget_id
                    or prepared_request.task_id
                    or prepared_request.session_id
                ),
                tool_capability_ceiling=prepared.tool_capability_ceiling,
                retry_policy=self._effective_retry_policy(prepared_request.retry_policy),
                failover=prepared_request.failover,
                structured_output=prepared_request.structured_output,
                tool_completion=prepared_request.tool_completion,
                thinking=(
                    prepared_request.thinking
                    if prepared_request.thinking is not None
                    else prepared.registered_agent.spec.thinking
                ),
                request_metadata=prepared_request.metadata,
                deadline_expires_at=prepared_request.execution_deadline.expires_at,
                deadline_source=prepared_request.execution_deadline.source,
                deadline_scope=prepared_request.execution_deadline.scope,
            ),
            operation_name="Work-attempt run settings validation",
            redactor=self._secret_redactor,
        )
        if semantics_validation.failure is not None:
            raise_task_store_operation_failure(semantics_validation.failure)
        run_semantics = semantics_validation.result
        if run_semantics is None:
            raise RuntimeError("Work-attempt preparation returned no run settings.")
        del semantics_validation
        session_id = prepared_request.session_id
        if session_id is None:
            raise AssertionError("Work-attempt session identity was not assigned.")
        interaction_id = authority.request.interaction_id
        lease_seconds = authority.request.lease_seconds
        interaction_started_event = build_interaction_started_event_from_identity(
            clock=self._clock,
            session_id=session_id,
            agent_name=prepared.registered_agent.spec.name,
            environment_name=_environment_name(prepared.registered_environment),
            interaction_id=interaction_id,
            event_id=None,
            targeted_tool_grants=prepared.targeted_tool_grants,
            policy_evidence=prepared.policy_evidence,
        )
        bind_runtime_session_create_claim(
            prepared_request,
            identity=prepared.session_identity,
            interaction_started_event=interaction_started_event,
        )
        if reconciled is None:
            session_instance_material = bytearray(
                hashlib.sha256(
                    canonical_durable_json_bytes(
                        {
                            "record_type": "cayu.work-attempt-session-instance",
                            "admission_id": authority.request.admission_id,
                            "session_id": session_id,
                            "source_request_sha256": authority.source_request_sha256,
                        },
                        "work_attempt_session_instance",
                    )
                ).digest()[:16]
            )
            session_instance_material[6] = (session_instance_material[6] & 0x0F) | 0x40
            session_instance_material[8] = (session_instance_material[8] & 0x3F) | 0x80
            session_instance_id = str(UUID(bytes=bytes(session_instance_material)))
        else:
            session_instance_id = reconciled.session_invocation.session_instance_id
        prepared_request = run_request_with_runtime_session_instance_authority(
            prepared_request,
            session_instance_id=session_instance_id,
        )
        initial_transcript_messages = transcript_helpers.initial_messages(
            system_prompt=prepared.rendered_system_prompt,
            request_messages=prepared_request.messages,
        )
        prepared_request = run_request_with_runtime_initial_transcript_authority(
            prepared_request,
            interaction_id=interaction_id,
            initial_transcript_messages=initial_transcript_messages,
        )
        session_binding = SessionInvocationBinding(
            id=session_id,
            session_instance_id=session_instance_id,
            invocation=session_invocation_for_run_request(
                prepared_request,
                session_id=session_id,
                parent_session=None,
            ),
        )
        admission = reconciled
        if admission is None:
            admission = await self._prepare_runtime_work_attempt(
                authority=authority,
                task_id=prepared_request.task_id,
                session_binding=session_binding,
                execution_profile=prepared.execution_profile,
                run_semantics=run_semantics,
            )
        elif (
            admission.state is not WorkAttemptAdmissionState.PREPARING
            or admission.task_id != prepared_request.task_id
            or admission.run_semantics != run_semantics
            or admission.session_invocation != session_binding
            or admission.source_execution_profile_fingerprint
            != prepared.execution_profile.fingerprint
        ):
            raise WorkAttemptRecoveryRequired(
                "Prepared initial admission conflicts with reconstructed runtime authority."
            )
        if admission.continuation is not None:
            raise WorkCompletionConflict(
                "Rejected/continue work requires the continuation admission entrance."
            )
        execution_profile = prepared.execution_profile
        discovery_agent = prepared.registered_agent
        discovery_ceiling = prepared.tool_capability_ceiling
        tool_discovery_initializer = _tool_discovery_operation_initializer(
            discovery_agent,
            discovery_ceiling,
        )
        tool_discovery_initialization = (
            None
            if tool_discovery_initializer is None
            else tool_discovery_view_initialization(
                session_id=session_id,
                agent_name=discovery_agent.spec.name,
                catalogue=discovery_agent.tool_catalogue,
                ceiling=discovery_ceiling,
            )
        )
        initial_active_profile = ActiveInvocationExecutionProfile(
            session_id=session_id,
            interaction_id=interaction_id,
            run_epoch=1,
            profile=execution_profile,
        )

        prepared_request = run_request_with_prepared_work_attempt_creation(
            prepared_request, identity=prepared.session_identity, admission=admission
        )

        async def create_session(
            request_to_create: RunRequest = prepared_request,
            *,
            identity: SessionIdentity = prepared.session_identity,
            started_event: Event = interaction_started_event,
            source_messages: tuple[Message, ...] = tuple(prepared_request.messages),
            discovery_initialization: ToolDiscoveryViewInitialization | None = (
                tool_discovery_initialization
            ),
            active_profile: ActiveInvocationExecutionProfile = initial_active_profile,
        ) -> Session:
            result = await self.session_store.apply_invocation_lifecycle_command(
                CreateInvocationCommand(
                    session_id=request_to_create.session_id or "",
                    expected_session_instance_id=session_instance_id,
                    request=request_to_create,
                    identity=identity,
                    active_profile=active_profile,
                    interaction_started_event=started_event,
                    interaction_source_messages=source_messages,
                    tool_discovery_initialization=discovery_initialization,
                )
            )
            if type(result) is not InvocationMutationResult:
                raise RuntimeError("Work-attempt create returned an invalid result.")
            return result.session

        mutation = settle_work_attempt_session_mutation(
            create_session,
            operation_name="work-attempt-session-create",
            preserved_failure_types=(
                SessionRunFenced,
                SessionStatusConflict,
                WorkAttemptRecoveryRequired,
            ),
            redactor=self._secret_redactor,
        )
        del (
            request,
            authority,
            reconciled,
            prepared,
            prepared_request,
            session_id,
            interaction_id,
            interaction_started_event,
            session_binding,
            tool_discovery_initializer,
            create_session,
        )
        try:
            raw_session = await mutation
        except BaseException:
            del admission, discovery_agent, discovery_ceiling, execution_profile, mutation
            raise
        del mutation
        session_validation = capture_work_attempt_session_result(
            raw_session,
            operation_name="Work-attempt session creation",
            redactor=self._secret_redactor,
        )
        del raw_session
        if session_validation.failure is not None:
            del admission, discovery_agent, discovery_ceiling, execution_profile
            raise_task_store_operation_failure(session_validation.failure)
        session = session_validation.result
        del session_validation
        if session is None:
            del admission, discovery_agent, discovery_ceiling, execution_profile
            raise RuntimeError("Work-attempt session creation returned no session.")
        await _require_tool_discovery_view(
            self.session_store,
            session,
            discovery_agent,
            discovery_ceiling,
            require_empty=True,
        )
        del discovery_agent, discovery_ceiling
        return await self._activate_runtime_work_attempt(
            admission=admission,
            session=session,
            execution_profile=execution_profile,
            lease_seconds=lease_seconds,
        )

    async def admit_continuation_work_attempt(
        self,
        request: ResumeRequest,
        *,
        authority: _WorkAttemptRuntimeAuthority,
    ) -> WorkAttemptAdmission:
        """Admit one rejected/continue attempt without dispatching provider work."""

        request = session_request_boundary.prepare_resume_request(
            request,
            redactor=self._secret_redactor,
        )
        if (
            request.profile_adoption is not None
            or request.target is not None
            or request.failover is not None
            or request.tool_capability_ceiling is not None
            or request.metadata
            or "max_steps" in request.model_fields_set
            or request.limits != RunLimits()
            or request.budget_limits
            or request.retry_policy is not None
            or request.structured_output is not None
            or request.thinking is not None
            or request.loop_policies
        ):
            raise ValueError(
                "Work-attempt continuation accepts new messages only; execution controls "
                "remain bound to the admitted source profile."
            )
        if request.tool_grants:
            raise ValueError("Work-attempt continuation does not admit targeted tool grants.")
        task_id = authority.request.task_id
        predecessor_admission_id = authority.request.predecessor_admission_id
        if task_id is None or predecessor_admission_id is None:
            raise ValueError(
                "Work-attempt continuation requires exact task and predecessor authority."
            )
        reconciled = await self._reconcile_runtime_work_attempt(
            authority,
            expected_task_id=task_id,
            expected_session_id=request.session_id,
        )
        if reconciled is not None and reconciled.state is WorkAttemptAdmissionState.ACTIVE:
            return reconciled
        raw_session = await read_work_attempt_session_store(
            lambda session_id=request.session_id: self.session_store.load(session_id),
            operation_name="Continuation work-attempt session lookup",
            redactor=self._secret_redactor,
        )
        if raw_session is None:
            raise KeyError(f"Session not found: {request.session_id}")
        session_validation = capture_work_attempt_session_result(
            raw_session,
            operation_name="Work-attempt continuation session lookup",
            redactor=self._secret_redactor,
        )
        del raw_session
        if session_validation.failure is not None:
            raise_task_store_operation_failure(session_validation.failure)
        session = session_validation.result
        if session is not None and reconciled is None:
            resumed_execution_deadline(session.execution_deadline).require_admission("work_attempt")
        elif session is not None and (
            reconciled is None
            or reconciled.run_semantics is None
            or resumed_execution_deadline(session.execution_deadline).model_dump()
            != reconciled.run_semantics.deadline.model_dump()
        ):
            raise WorkAttemptRecoveryRequired("Prepared continuation deadline changed.")
        del session_validation
        if session is None:
            raise RuntimeError("Work-attempt continuation session lookup returned no session.")
        profile = execution_profile_from_session_metadata(session.metadata)
        if profile is None:
            raise RuntimeError("Continuation session has no durable execution profile.")
        session_binding = SessionInvocationBinding(
            id=session.id,
            session_instance_id=session.instance_id,
            invocation=session.invocation,
        )
        if session.status not in _session_checkpoint_admission._RESUMABLE_SESSION_STATUSES:
            raise SessionStatusConflict(
                "Work-attempt continuation requires a terminal resumable session."
            )
        task_store = self.task_store
        if task_store is None:
            raise RuntimeError("task_store is required for work-attempt execution.")
        predecessor_outcome = await capture_task_store_operation(
            lambda: task_store.load_work_attempt_admission(predecessor_admission_id),
            operation_name="Continuation predecessor admission lookup",
            redactor=self._secret_redactor,
        )
        if predecessor_outcome.failure is not None:
            raise_task_store_operation_failure(predecessor_outcome.failure)
        if predecessor_outcome.result is None:
            raise WorkAttemptAdmissionConflict("Continuation predecessor admission is missing.")
        predecessor_validation = capture_sensitive_result_validation(
            lambda value=predecessor_outcome.result: require_work_attempt_admission_result(
                value,
                operation_name="Continuation predecessor admission lookup",
            ),
            operation_name="Continuation predecessor admission validation",
            redactor=self._secret_redactor,
        )
        del predecessor_outcome
        if predecessor_validation.failure is not None:
            raise_task_store_operation_failure(predecessor_validation.failure)
        predecessor = predecessor_validation.result
        del predecessor_validation
        if predecessor is None:
            raise RuntimeError("Continuation predecessor validation returned no authority.")
        if (
            predecessor.admission_id != predecessor_admission_id
            or predecessor.task_id != task_id
            or predecessor.session_id != session.id
            or predecessor.state is not WorkAttemptAdmissionState.RELEASED
        ):
            raise WorkAttemptAdmissionConflict(
                "Continuation predecessor conflicts with released task-session authority."
            )
        predecessor_recovery_authority = work_attempt_recovery_session_authority(predecessor.claim)
        raw_checkpoint = await read_work_attempt_session_store(
            lambda session_id=session.id: self.session_store.load_checkpoint(session_id),
            operation_name="Continuation work-attempt checkpoint lookup",
            redactor=self._secret_redactor,
        )
        checkpoint_validation = capture_work_attempt_checkpoint_result(
            raw_checkpoint,
            operation_name="Work-attempt continuation checkpoint lookup",
            redactor=self._secret_redactor,
        )
        del raw_checkpoint
        if checkpoint_validation.failure is not None:
            raise_task_store_operation_failure(checkpoint_validation.failure)
        checkpoint = checkpoint_validation.result
        del checkpoint_validation
        checkpoint_recovery_authority = work_attempt_recovery_session_authority_from_checkpoint(
            checkpoint
        )
        if (
            checkpoint_recovery_authority is not None
            and checkpoint_recovery_authority != predecessor_recovery_authority
        ):
            raise WorkAttemptRecoveryRequired(
                "Continuation recovery marker belongs to another predecessor admission."
            )
        _session_checkpoint_admission._reject_unresumable_session_checkpoint(
            session,
            checkpoint,
            redactor=self._secret_redactor,
        )
        active_profile = active_invocation_execution_profile_from_checkpoint(checkpoint)
        if active_profile is None and invocation_lifecycle_receipt_history_present(checkpoint):
            raise SessionRunFenced(
                "Continuation lost its durable predecessor invocation authority."
            )
        if active_profile is not None and not active_invocation_execution_profile_is_released(
            active_profile,
            session_id=session.id,
            run_epoch=session.run_epoch,
        ):
            raise SessionRunFenced(
                "The previous invocation still owns terminal hooks or trailing cleanup."
            )
        admission = reconciled
        if admission is None:
            admission = await self._prepare_runtime_work_attempt(
                authority=authority,
                task_id=task_id,
                session_binding=session_binding,
                execution_profile=profile,
                run_semantics=predecessor.run_semantics,
            )
        elif (
            admission.state is not WorkAttemptAdmissionState.PREPARING
            or admission.task_id != task_id
            or admission.run_semantics != predecessor.run_semantics
            or admission.session_invocation != session_binding
            or admission.source_execution_profile_fingerprint != profile.fingerprint
        ):
            raise WorkAttemptRecoveryRequired(
                "Prepared continuation conflicts with reconstructed runtime authority."
            )
        if admission.continuation is None:
            raise WorkCompletionConflict(
                "Continuation admission requires a rejected/continue decision."
            )
        if (
            admission.continuation.prior_admission_id != predecessor.admission_id
            or admission.continuation.prior_attempt_id != predecessor.attempt_id
        ):
            raise WorkAttemptRecoveryRequired(
                "Continuation receipt conflicts with its selected predecessor admission."
            )
        interaction_started_event = build_interaction_started_event(
            clock=self._clock,
            session=session,
            registered_agent=self._get_registered_agent(session.agent_name),
            environment_name=session.environment_name,
            interaction_id=authority.request.interaction_id,
        )
        expected_session_id = session.id
        expected_run_epoch = session.run_epoch
        lease_seconds = authority.request.lease_seconds

        def validate_continuation_checkpoint(
            current_session: Session,
            current_checkpoint: dict[str, Any] | None,
            *,
            source_session_id: str = expected_session_id,
            source_run_epoch: int = expected_run_epoch,
            expected_recovery_authority=predecessor_recovery_authority,
        ) -> dict[str, Any] | None:
            _session_checkpoint_admission._reject_unresumable_session_checkpoint(
                current_session,
                current_checkpoint,
                redactor=self._secret_redactor,
            )
            current_active = active_invocation_execution_profile_from_checkpoint(current_checkpoint)
            if current_active is not None and not active_invocation_execution_profile_is_released(
                current_active,
                session_id=source_session_id,
                run_epoch=source_run_epoch,
            ):
                raise SessionRunFenced("The previous invocation still owns trailing cleanup.")
            if (
                _session_operation_state._active_session_operation_id(current_checkpoint)
                is not None
            ):
                raise RuntimeError("Session has an active durable operation.")
            current_recovery_authority = work_attempt_recovery_session_authority_from_checkpoint(
                current_checkpoint
            )
            if (
                current_recovery_authority is not None
                and current_recovery_authority != expected_recovery_authority
            ):
                raise WorkAttemptRecoveryRequired(
                    "Continuation recovery marker belongs to another predecessor admission."
                )
            if current_checkpoint is None:
                return None
            updated = copy_durable_record(current_checkpoint, "checkpoint")
            # Recovery authority is scoped to the admission that published it.
            # Retire the predecessor marker in the same session transaction that
            # admits a distinct continuation interaction.
            updated.pop(WORK_ATTEMPT_RECOVERY_CHECKPOINT_KEY, None)
            return updated or None

        claimed_continuation_checkpoint = validate_continuation_checkpoint(
            session,
            checkpoint,
        )
        continuation_active_profile = ActiveInvocationExecutionProfile(
            session_id=expected_session_id,
            interaction_id=authority.request.interaction_id,
            run_epoch=expected_run_epoch + 1,
            profile=profile,
        )
        continuation_command = AdmitInvocationCommand(
            session_id=expected_session_id,
            expected_session_instance_id=session.instance_id,
            expected_statuses=tuple(_session_checkpoint_admission._RESUMABLE_SESSION_STATUSES),
            expected_run_epoch=expected_run_epoch,
            expected_checkpoint_sha256=invocation_checkpoint_state_sha256(checkpoint),
            target_active_profile=continuation_active_profile,
            checkpoint_patch=InvocationCheckpointPatch(
                mutation=runtime_publication_checkpoint_mutation(
                    checkpoint,
                    claimed_continuation_checkpoint,
                )
            ),
            tool_capability_ceiling=tool_capability_ceiling_from_session_metadata(session.metadata),
            interaction_started_event=interaction_started_event,
            interaction_source_messages=tuple(request.messages),
            expected_active_profile=active_profile,
        )

        async def admit_session_invocation(
            command: AdmitInvocationCommand = continuation_command,
        ) -> Session:
            result = await self.session_store.apply_invocation_lifecycle_command(command)
            if type(result) is not InvocationMutationResult:
                raise RuntimeError("Work-attempt admission returned an invalid result.")
            return result.session

        mutation = settle_work_attempt_session_mutation(
            admit_session_invocation,
            operation_name="work-attempt-continuation-admission",
            preserved_failure_types=(
                SessionRunFenced,
                SessionStatusConflict,
                WorkAttemptRecoveryRequired,
            ),
            redactor=self._secret_redactor,
        )
        del (
            request,
            authority,
            reconciled,
            session,
            task_id,
            session_binding,
            checkpoint,
            active_profile,
            predecessor,
            predecessor_recovery_authority,
            interaction_started_event,
            expected_session_id,
            expected_run_epoch,
            validate_continuation_checkpoint,
            continuation_command,
            admit_session_invocation,
        )
        try:
            raw_resumed = await mutation
        except BaseException:
            del admission, profile, mutation
            raise
        del mutation
        resumed_validation = capture_work_attempt_session_result(
            raw_resumed,
            operation_name="Work-attempt continuation admission",
            redactor=self._secret_redactor,
        )
        del raw_resumed
        if resumed_validation.failure is not None:
            del admission, profile
            raise_task_store_operation_failure(resumed_validation.failure)
        resumed = resumed_validation.result
        del resumed_validation
        if resumed is None:
            del admission, profile
            raise RuntimeError("Work-attempt continuation admission returned no session.")
        return await self._activate_runtime_work_attempt(
            admission=admission,
            session=resumed,
            execution_profile=profile,
            lease_seconds=lease_seconds,
        )

    async def _settle_pre_entry_event_handoff(
        self,
        admission: WorkAttemptAdmission,
        *,
        event_type: EventType,
        event_id: str | None = None,
        dispatch_pending: bool = False,
    ) -> bool:
        """Require positive callback return, not expiry of its admission lease.

        A terminal first attempt cannot have an older delivery still running.
        Retire an exact failed first attempt atomically before permitting cleanup,
        so a future delivery retry cannot begin after finalizer release.
        After takeover/retry, delivery state alone cannot prove that every prior
        callback returned. Keep that uncertainty fenced rather than redelivering.
        """
        records = await read_work_attempt_session_store(
            lambda: self.session_store.query_events(
                EventQuery(
                    session_id=admission.session_id,
                    interaction_id=admission.interaction_id if event_id is None else None,
                    event_id=event_id,
                    event_type=event_type,
                    limit=2,
                )
            ),
            operation_name="Pre-entry event handoff events",
            redactor=self._secret_redactor,
        )
        validation = capture_work_attempt_event_records_result(
            records,
            operation_name="Pre-entry event handoff evidence",
            redactor=self._secret_redactor,
        )
        if validation.failure is not None:
            raise_task_store_operation_failure(validation.failure)
        records = validation.result
        if records is None or len(records) != 1:
            return False
        record = records[0]
        if (
            record.event.session_id != admission.session_id
            or (event_id is None and record.event.interaction_id != admission.interaction_id)
            or (event_id is not None and record.event.id != event_id)
            or record.event.type is not event_type
        ):
            return False
        delivery = await read_work_attempt_session_store(
            lambda: self.session_store.get_persisted_event_side_effect_delivery(
                session_id=admission.session_id, event_id=record.event.id
            ),
            operation_name="Pre-entry event handoff delivery",
            redactor=self._secret_redactor,
        )
        if (
            dispatch_pending
            and type(delivery) is PersistedEventSideEffectDelivery
            and delivery.status is PersistedEventSideEffectStatus.PENDING
            and type(delivery.attempts) is int
            and delivery.attempts == 0
            and delivery.session_id == admission.session_id
            and delivery.event_id == record.event.id
            and type(delivery.event_sequence) is int
            and delivery.event_sequence == record.sequence
            and delivery.claim_id is None
            and delivery.lease_expires_at is None
        ):
            # Compare-and-claim the entire pending snapshot. A publisher that
            # advanced it wins; this recovery must not dispatch a retry instead.
            await self._event_writer.fan_out_first_persisted(delivery)
            return await self._settle_pre_entry_event_handoff(
                admission, event_type=event_type, event_id=event_id
            )
        if not (
            type(delivery) is PersistedEventSideEffectDelivery
            and delivery.session_id == admission.session_id
            and delivery.event_id == record.event.id
            and type(delivery.event_sequence) is int
            and delivery.event_sequence == record.sequence
            and type(delivery.attempts) is int
            and delivery.attempts == 1
            and delivery.claim_id is None
            and delivery.lease_expires_at is None
        ):
            return False
        if delivery.status is PersistedEventSideEffectStatus.FAILED:
            retired = await read_work_attempt_session_store(
                lambda: self.session_store.retire_failed_first_event_delivery(delivery),
                operation_name="Pre-entry event handoff retirement",
                redactor=self._secret_redactor,
            )
            if retired is None:
                return False
            if type(
                retired
            ) is not PersistedEventSideEffectDelivery or retired != delivery.model_copy(
                update={
                    "status": PersistedEventSideEffectStatus.DEAD_LETTERED,
                    "next_attempt_at": None,
                    "updated_at": retired.updated_at,
                }
            ):
                raise WorkAttemptRecoveryRequired("Pre-entry event handoff retirement conflicts.")
            delivery = retired
        return delivery.status in {
            PersistedEventSideEffectStatus.DELIVERED,
            PersistedEventSideEffectStatus.DEAD_LETTERED,
        }

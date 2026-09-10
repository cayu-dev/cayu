"""Pure final-settlement policy shared by verified-task store transactions."""

from __future__ import annotations

from datetime import datetime

from cayu.runtime.tasks import (
    CompletionDecisionApplicationReceipt,
    Task,
    TaskStatus,
    WorkAttemptLifecycleReceipt,
    WorkAttemptPreparationHoldReceipt,
    _ensure_exact_owned_active_task_lease,
    _task_cancellation_requested,
    copy_task,
)
from cayu.runtime.work_attempt_admission import (
    WorkAttemptAdmission,
    WorkAttemptAdmissionConflict,
    WorkAttemptAdmissionState,
    WorkAttemptExecutionClaimLost,
    WorkAttemptExecutionEntry,
    WorkAttemptExecutionEntryDisposition,
    WorkAttemptExecutionEntryRequest,
    WorkAttemptExecutionEntryResult,
    WorkAttemptExecutionStop,
    WorkAttemptExecutionStopRequest,
    copy_work_attempt_execution_entry_request,
    copy_work_attempt_execution_stop_request,
    require_work_attempt_admission_result,
)
from cayu.runtime.work_attempt_lifecycle import (
    WorkAttemptLifecycleSettlement,
    WorkAttemptPreparationHold,
    copy_work_attempt_lifecycle_settlement,
    copy_work_attempt_preparation_hold,
    runtime_stop_reason_for_execution_stop,
    work_attempt_admission_authority_sha256,
    work_attempt_lifecycle_settlement_sha256,
    work_attempt_preparation_hold_sha256,
)
from cayu.runtime.work_contracts import CompletionDecision, CompletionProposal, CompletionVerdict


def plan_work_attempt_execution_entry(
    request: WorkAttemptExecutionEntryRequest,
    *,
    admission: WorkAttemptAdmission,
    task: Task,
    now: datetime,
) -> WorkAttemptExecutionEntryResult:
    """Elect one dispatch at the transaction owning the existing admission."""
    request = copy_work_attempt_execution_entry_request(request)
    admission = require_work_attempt_admission_result(admission, operation_name="Execution entry")
    task = copy_task(task)
    claim = admission.claim
    if (
        request.admission_id != admission.admission_id
        or request.prepare_request_sha256 != admission.prepare_request_sha256
        or request.claim_id != claim.claim_id
        or request.worker_id != claim.worker_id
        or request.execution_owner_id != claim.execution_owner_id
        or request.generation != claim.generation
    ):
        raise WorkAttemptExecutionClaimLost("Execution entry conflicts with the current claim.")
    prior = admission.execution_entry
    if prior is not None and prior.request.generation == claim.generation:
        if prior.request != request:
            raise WorkAttemptAdmissionConflict(
                "Execution entry conflicts with its original request."
            )
        return WorkAttemptExecutionEntryResult(
            disposition=WorkAttemptExecutionEntryDisposition.ALREADY_ENTERED,
            admission=admission,
        )
    if (
        admission.state is not WorkAttemptAdmissionState.ACTIVE
        or admission.execution_stop is not None
        or admission.run_semantics is None
        or task.id != admission.task_id
        or task.work_contract != admission.contract
        or task.session_id != admission.session_id
        or task.session_instance_id != admission.session_invocation.session_instance_id
        or task.status is not TaskStatus.RUNNING
        or _task_cancellation_requested(task)
        or (prior is not None and admission.recovery_evidence_sha256 is None)
    ):
        raise WorkAttemptExecutionClaimLost("Execution entry requires a live admitted task owner.")
    _ensure_exact_owned_active_task_lease(task, claim.worker_id, claim.lease_expires_at, now=now)
    entry = WorkAttemptExecutionEntry(request=request, entered_at=now)
    entered = admission.model_copy(update={"execution_entry": entry})
    deadline = admission.run_semantics.deadline_expires_at
    if deadline is not None and deadline <= now:
        # Expired preparation may finish its exact session mutation, but must
        # never leave an entered invocation without durable cleanup authority.
        entered = plan_work_attempt_execution_stop(
            WorkAttemptExecutionStopRequest(
                admission_id=admission.admission_id,
                prepare_request_sha256=admission.prepare_request_sha256,
                claim_id=claim.claim_id,
                worker_id=claim.worker_id,
                execution_owner_id=claim.execution_owner_id,
                generation=claim.generation,
                execution_entry=entry,
                reason="elapsed_limit",
            ),
            admission=entered,
            task=task,
            now=now,
        )
    return WorkAttemptExecutionEntryResult(
        disposition=WorkAttemptExecutionEntryDisposition.ENTERED,
        admission=entered,
    )


def plan_work_attempt_execution_stop(
    request: WorkAttemptExecutionStopRequest,
    *,
    admission: WorkAttemptAdmission,
    task: Task,
    now: datetime,
) -> WorkAttemptAdmission:
    """Keep terminal execution intent on the admission before cleanup effects."""
    request = copy_work_attempt_execution_stop_request(request)
    admission = require_work_attempt_admission_result(admission, operation_name="Execution stop")
    task = copy_task(task)
    if admission.execution_stop is not None:
        if admission.execution_stop.request != request:
            raise WorkAttemptAdmissionConflict(
                "Execution stop conflicts with its original request."
            )
        return admission
    claim = admission.claim
    expected_state = (
        WorkAttemptAdmissionState.RECOVERING
        if request.reason == "workspace_finalization_recovery"
        else WorkAttemptAdmissionState.ACTIVE
    )
    if (
        admission.state is not expected_state
        or request.admission_id != admission.admission_id
        or request.prepare_request_sha256 != admission.prepare_request_sha256
        or request.execution_entry != admission.execution_entry
        or request.claim_id != claim.claim_id
        or request.worker_id != claim.worker_id
        or request.execution_owner_id != claim.execution_owner_id
        or request.generation != claim.generation
        or task.id != admission.task_id
        or task.work_contract != admission.contract
        or task.session_id != admission.session_id
        or task.session_instance_id != admission.session_invocation.session_instance_id
        or task.status is not TaskStatus.RUNNING
    ):
        raise WorkAttemptExecutionClaimLost(
            "Execution stop requires exact entered claim authority."
        )
    _ensure_exact_owned_active_task_lease(task, claim.worker_id, claim.lease_expires_at, now=now)
    return admission.model_copy(
        update={"execution_stop": WorkAttemptExecutionStop(request=request, recorded_at=now)}
    )


def plan_work_attempt_preparation_hold(
    request: WorkAttemptPreparationHold,
    *,
    task: Task,
    has_attempt: bool,
    now: datetime,
) -> tuple[Task, WorkAttemptPreparationHoldReceipt]:
    """Produce no-success state only for the exact still-unattached live claim."""
    request = copy_work_attempt_preparation_hold(request)
    task = copy_task(task)
    if (
        task.id != request.task_id
        or task.work_contract != request.contract
        or task.status is not TaskStatus.CLAIMED
        or task.session_id is not None
        or task.session_instance_id is not None
        or has_attempt
        or _task_cancellation_requested(task)
        or (request.deadline_expires_at is not None and request.deadline_expires_at > now)
    ):
        raise WorkAttemptAdmissionConflict(
            "Preparation hold requires an unattached initial task claim."
        )
    _ensure_exact_owned_active_task_lease(
        task, request.worker_id, request.lease_expires_at, now=now
    )
    updated = task.model_copy(
        update={
            "status": TaskStatus.NEEDS_ATTENTION,
            "status_reason": request.reason,
            "status_payload": {},
            "worker_id": None,
            "lease_expires_at": None,
            "updated_at": now,
        }
    )
    receipt = WorkAttemptPreparationHoldReceipt(
        request=request,
        request_sha256=work_attempt_preparation_hold_sha256(request),
        task=updated,
    )
    return updated, receipt


def plan_work_attempt_lifecycle_settlement(
    request: WorkAttemptLifecycleSettlement,
    *,
    task: Task,
    admission: WorkAttemptAdmission,
    latest_admission_id: str,
    proposal: CompletionProposal | None,
    decision: CompletionDecision | None,
    application: CompletionDecisionApplicationReceipt | None,
    now: datetime,
) -> tuple[Task, WorkAttemptAdmission, WorkAttemptLifecycleReceipt]:
    """Validate all authority before producing any task/index mutation.

    Stores resolve inputs under their transaction owner. Exact receipt replay
    precedes this planner, since a committed result remains authoritative after
    its original lease expires or checkpoint history is pruned.
    """

    request = copy_work_attempt_lifecycle_settlement(request)
    task = copy_task(task)
    admission = require_work_attempt_admission_result(
        admission, operation_name="Work-attempt lifecycle settlement"
    )
    evidence = request.release_evidence
    if (
        request.task_id != task.id
        or admission.task_id != task.id
        or request.admission_id != admission.admission_id
        or latest_admission_id != admission.admission_id
        or request.expected_admission_sha256 != work_attempt_admission_authority_sha256(admission)
        or admission.run_semantics is None
        or task.work_contract != admission.contract
        or task.session_id != admission.session_id
        or task.session_instance_id != admission.session_invocation.session_instance_id
        or evidence.session_id != admission.session_id
        or evidence.session_instance_id != admission.session_invocation.session_instance_id
        or evidence.interaction_id != admission.interaction_id
        or evidence.profile_fingerprint != admission.source_execution_profile_fingerprint
    ):
        raise WorkAttemptAdmissionConflict(
            "Lifecycle settlement conflicts with exact admission authority."
        )

    if request.kind in {"decision_application", "continuation_deadline_stop"}:
        if (
            admission.state is not WorkAttemptAdmissionState.RELEASED
            or proposal is None
            or proposal.attempt_id != admission.attempt_id
            or proposal.task_id != task.id
            or proposal.contract != admission.contract
            or decision is None
            or decision.decision_id != request.decision_id
            or decision.attempt_id != admission.attempt_id
            or decision.task_id != task.id
            or decision.contract != admission.contract
            or decision.proposal_id != proposal.proposal_id
            or application is None
            or application.task_id != task.id
            or application.decision_id != decision.decision_id
            or application.verifier_profile_fingerprint != decision.verifier_profile_fingerprint
            or application.idempotency_key != request.application_idempotency_key
            or application.task != task
            or task.worker_id is not None
            or task.lease_expires_at is not None
        ):
            raise WorkAttemptAdmissionConflict(
                "Lifecycle settlement lacks exact applied-decision authority."
            )
        continuation_stop = request.kind == "continuation_deadline_stop"
        retired = decision.verdict is CompletionVerdict.ACCEPTED
        if continuation_stop:
            if (
                decision.verdict is not CompletionVerdict.REJECTED
                or task.status is not TaskStatus.RUNNING
                or admission.execution_entry is None
                or admission.execution_stop is not None
                or proposal.proposal_id != request.proposal_id
                or proposal.request_sha256 != request.proposal_request_sha256
                or admission.run_semantics.deadline_expires_at is None
                or admission.run_semantics.deadline_expires_at > now
                or _task_cancellation_requested(task)
            ):
                raise WorkAttemptAdmissionConflict(
                    "Continuation deadline settlement lacks exact expired applied authority."
                )
            updated = task.model_copy(
                update={
                    "status": TaskStatus.NEEDS_ATTENTION,
                    "status_reason": request.stop_reason,
                    "status_payload": {"settlement_id": request.settlement_id},
                    "updated_at": now,
                }
            )
        elif (retired and task.status is not TaskStatus.COMPLETED) or (
            not retired
            and task.status
            not in {TaskStatus.PAUSED, TaskStatus.BLOCKED, TaskStatus.NEEDS_ATTENTION}
        ):
            raise WorkAttemptAdmissionConflict(
                "A continuing decision is not final task settlement."
            )
        else:
            updated = task
        settled_admission = admission
    else:
        execution_stop_reason = runtime_stop_reason_for_execution_stop(admission)
        proposal_deadline_stop = request.kind == "proposal_deadline_stop"
        if (
            admission.state
            is not (
                WorkAttemptAdmissionState.RELEASED
                if proposal_deadline_stop
                else WorkAttemptAdmissionState.ACTIVE
            )
            or (execution_stop_reason is not None and request.stop_reason != execution_stop_reason)
            or (not proposal_deadline_stop and proposal is not None)
            or decision is not None
            or application is not None
            or task.status is not TaskStatus.RUNNING
            or _task_cancellation_requested(task)
        ):
            raise WorkAttemptAdmissionConflict(
                "Runtime stop cannot replace another lifecycle outcome."
            )
        if proposal_deadline_stop:
            if (
                admission.execution_entry is None
                or admission.execution_stop is not None
                or proposal is None
                or proposal.proposal_id != request.proposal_id
                or proposal.request_sha256 != request.proposal_request_sha256
                or proposal.attempt_id != admission.attempt_id
                or proposal.task_id != task.id
                or proposal.contract != admission.contract
                or admission.run_semantics.deadline_expires_at is None
                or admission.run_semantics.deadline_expires_at > now
                or task.worker_id is not None
                or task.lease_expires_at is not None
            ):
                raise WorkAttemptAdmissionConflict(
                    "Proposal deadline settlement lacks exact expired authority."
                )
        else:
            _ensure_exact_owned_active_task_lease(
                task,
                admission.claim.worker_id,
                admission.claim.lease_expires_at,
                now=now,
            )
        updated = task.model_copy(
            update={
                "status": TaskStatus.NEEDS_ATTENTION,
                "status_reason": request.stop_reason,
                "status_payload": {"settlement_id": request.settlement_id},
                "worker_id": None,
                "lease_expires_at": None,
                "updated_at": now,
            }
        )
        settled_admission = admission.model_copy(
            update={"state": WorkAttemptAdmissionState.RELEASED}
        )
        retired = False

    receipt = WorkAttemptLifecycleReceipt(
        request=request,
        request_sha256=work_attempt_lifecycle_settlement_sha256(request),
        task=updated,
        retired_contract_binding=retired,
        settled_at=now,
    )
    return receipt.task, settled_admission, receipt

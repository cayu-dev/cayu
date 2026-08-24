"""Backend-neutral verified-work persistence invariants.

The built-in task stores own different transaction mechanisms, but they must
make the same authority decision from the same durable snapshots.  This module
contains only pure validation and transition planning; it never performs I/O.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import UTC, datetime

from cayu.runtime.tasks import (
    CompletionDecisionApplicationReceipt,
    Task,
    TaskClaimLost,
    TaskStatus,
    _ensure_active_task_lease,
    _ensure_can_transition,
)
from cayu.runtime.work_contracts import (
    CompletionDecision,
    CompletionDecisionApplicationRequest,
    CompletionProposal,
    CompletionRejectionAction,
    CompletionVerdict,
    TaskCompletionDecisionRequired,
    WorkAttempt,
    WorkCompletionConflict,
    WorkContract,
    WorkContractConflict,
    WorkContractRef,
)


def require_contract_reference(
    contract: WorkContract | None,
    reference: WorkContractRef,
) -> WorkContract:
    if contract is None:
        raise WorkContractConflict("Referenced work contract has not been published.")
    if contract.reference() != reference:
        raise WorkContractConflict(
            "Work-contract reference conflicts with the published fingerprint."
        )
    return contract


def require_task_contract(
    task: Task,
    reference: WorkContractRef,
    contract: WorkContract | None,
) -> WorkContract:
    contract = require_contract_reference(contract, reference)
    if task.work_contract is None:
        raise WorkCompletionConflict("Task is not bound to a work contract.")
    if task.work_contract != reference:
        raise WorkCompletionConflict(
            "Work operation conflicts with the task's frozen contract binding."
        )
    return contract


def require_attempt_worker(task: Task, worker_id: str | None, *, now: datetime) -> None:
    if task.worker_id != worker_id:
        raise TaskClaimLost("Work attempt does not carry the task's current worker authority.")
    if worker_id is not None:
        _ensure_active_task_lease(task, worker_id, now=now)


def require_attempt_state_current(
    task: Task,
    attempt: WorkAttempt,
    *,
    latest_attempt_id: str | None,
    contract: WorkContract | None,
) -> WorkContract:
    contract = require_task_contract(task, attempt.contract, contract)
    if latest_attempt_id != attempt.attempt_id:
        raise WorkCompletionConflict("Work operation does not reference the latest task attempt.")
    if task.status is not TaskStatus.RUNNING or task.session_id != attempt.session_id:
        raise WorkCompletionConflict("Work attempt no longer owns the live task session.")
    return contract


def require_attempt_current(
    task: Task,
    attempt: WorkAttempt,
    *,
    latest_attempt_id: str | None,
    contract: WorkContract | None,
    now: datetime,
) -> WorkContract:
    contract = require_attempt_state_current(
        task,
        attempt,
        latest_attempt_id=latest_attempt_id,
        contract=contract,
    )
    require_attempt_worker(task, attempt.worker_id, now=now)
    return contract


def require_decision_attempt_current(
    task: Task,
    attempt: WorkAttempt,
    *,
    latest_attempt_id: str | None,
    contract: WorkContract | None,
) -> WorkContract:
    contract = require_attempt_state_current(
        task,
        attempt,
        latest_attempt_id=latest_attempt_id,
        contract=contract,
    )
    if task.worker_id not in {None, attempt.worker_id}:
        raise TaskClaimLost("Completion decision conflicts with replacement task-worker authority.")
    return contract


def require_proposal_chain(
    proposal: CompletionProposal,
    attempt: WorkAttempt,
    task: Task,
    *,
    latest_attempt_id: str | None,
    contract: WorkContract | None,
) -> WorkContract:
    if attempt.attempt_id != proposal.attempt_id:
        raise WorkCompletionConflict("Completion proposal conflicts with its durable work attempt.")
    if attempt.task_id != proposal.task_id or attempt.contract != proposal.contract:
        raise WorkCompletionConflict("Completion proposal conflicts with its durable work attempt.")
    return require_attempt_state_current(
        task,
        attempt,
        latest_attempt_id=latest_attempt_id,
        contract=contract,
    )


def require_contracted_completion_authority(
    task: Task,
    status: TaskStatus,
    *,
    accepted_decision_id: str | None = None,
) -> None:
    if (
        status is TaskStatus.COMPLETED
        and task.work_contract is not None
        and accepted_decision_id is None
    ):
        raise TaskCompletionDecisionRequired(
            "Contracted task completion requires an accepted durable verifier decision."
        )


def lifecycle_now(task: Task, now: datetime | None = None) -> datetime:
    now = datetime.now(UTC) if now is None else now.astimezone(UTC)
    timestamps = [now, task.created_at, task.updated_at]
    if task.started_at is not None:
        timestamps.append(task.started_at)
    if task.completed_at is not None:
        timestamps.append(task.completed_at)
    return max(timestamps)


def _held_task(
    task: Task,
    status: TaskStatus,
    *,
    decision: CompletionDecision,
    now: datetime,
    status_reason: str | None = None,
) -> Task:
    if status not in {TaskStatus.PAUSED, TaskStatus.BLOCKED, TaskStatus.NEEDS_ATTENTION}:
        raise ValueError("Completion decisions can only apply supported held statuses.")
    if task.status is not TaskStatus.RUNNING:
        raise WorkCompletionConflict("Completion decision no longer owns a running task.")
    return task.model_copy(
        update={
            "status": status,
            "status_reason": status_reason or f"work_contract_{decision.verdict.value}",
            "status_payload": {
                "completion_decision_id": decision.decision_id,
                "gap_fingerprint": decision.gap_fingerprint,
                "verifier_profile_fingerprint": decision.verifier_profile_fingerprint,
                "verdict": decision.verdict.value,
            },
            "worker_id": None,
            "lease_expires_at": None,
            "updated_at": now,
        }
    )


def _rejection_hold(
    contract: WorkContract,
    decision: CompletionDecision,
    attempt: WorkAttempt,
    *,
    matching_gap_count: int,
) -> tuple[TaskStatus, str] | None:
    policy = contract.continuation_policy
    if attempt.ordinal >= policy.max_attempts:
        return (TaskStatus.NEEDS_ATTENTION, "work_contract_attempt_limit")
    repeated_gap_count = max(0, matching_gap_count - 1)
    if repeated_gap_count >= policy.max_repeated_gap_count:
        return (TaskStatus.NEEDS_ATTENTION, "work_contract_repeated_gap_limit")
    if policy.rejection_action is CompletionRejectionAction.INTERRUPT:
        return (TaskStatus.PAUSED, "work_contract_rejected")
    return None


def plan_decision_application(
    request: CompletionDecisionApplicationRequest,
    *,
    request_sha256: str,
    task: Task,
    decision: CompletionDecision,
    proposal: CompletionProposal,
    attempt: WorkAttempt,
    contract: WorkContract,
    matching_gap_count: int,
    now: datetime,
) -> tuple[Task, CompletionDecisionApplicationReceipt]:
    applied_at = lifecycle_now(task, now)
    updated = task
    if decision.verdict is CompletionVerdict.ACCEPTED:
        if request.result is None or request.result_reference is None:
            raise ValueError("Accepted completion decisions require a verified task result.")
        if request.result_reference != proposal.result:
            raise WorkCompletionConflict(
                "Decision application result conflicts with the accepted proposal."
            )
        require_contracted_completion_authority(
            task,
            TaskStatus.COMPLETED,
            accepted_decision_id=decision.decision_id,
        )
        _ensure_can_transition(task, TaskStatus.COMPLETED)
        updated = task.model_copy(
            update={
                "status": TaskStatus.COMPLETED,
                "status_reason": None,
                "status_payload": None,
                "result": deepcopy(request.result),
                "error": None,
                "worker_id": None,
                "lease_expires_at": None,
                "started_at": task.started_at or applied_at,
                "completed_at": applied_at,
                "updated_at": applied_at,
                "retry_series": None,
            }
        )
    else:
        if request.result is not None:
            raise ValueError("Non-accepted completion decisions cannot carry a result.")
        if decision.verdict is CompletionVerdict.BLOCKED:
            updated = _held_task(
                task,
                TaskStatus.BLOCKED,
                decision=decision,
                now=applied_at,
            )
        elif decision.verdict is CompletionVerdict.NEEDS_REVIEW:
            updated = _held_task(
                task,
                TaskStatus.NEEDS_ATTENTION,
                decision=decision,
                now=applied_at,
            )
        elif decision.verdict is CompletionVerdict.REJECTED:
            rejection_hold = _rejection_hold(
                contract,
                decision,
                attempt,
                matching_gap_count=matching_gap_count,
            )
            if rejection_hold is not None:
                hold_status, status_reason = rejection_hold
                updated = _held_task(
                    task,
                    hold_status,
                    decision=decision,
                    now=applied_at,
                    status_reason=status_reason,
                )
            elif task.worker_id is not None or task.lease_expires_at is not None:
                updated = task.model_copy(
                    update={
                        "worker_id": None,
                        "lease_expires_at": None,
                        "updated_at": applied_at,
                    }
                )
    receipt = CompletionDecisionApplicationReceipt(
        task_id=updated.id,
        decision_id=decision.decision_id,
        verifier_profile_fingerprint=decision.verifier_profile_fingerprint,
        idempotency_key=request.idempotency_key,
        request_sha256=request_sha256,
        task=updated,
        applied_at=applied_at,
    )
    return updated, receipt

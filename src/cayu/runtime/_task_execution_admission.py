"""Verifier-aware task admission shared by session operations."""

from __future__ import annotations

from cayu.runtime._task_store_operation_boundary import (
    capture_sensitive_validation,
    capture_task_store_operation,
)
from cayu.tasks.contracts import TaskCompletionDecisionRequired
from cayu.tasks.records import Task, copy_task
from cayu.tasks.store import TaskStore
from cayu.vaults.redaction import SecretRedactor


async def verifier_aware_task_execution_outcome(
    task_id: str | None,
    *,
    task_store: TaskStore | None,
    redactor: SecretRedactor,
    session_id: str | None = None,
    admit_session: bool = True,
    allow_missing_task: bool = False,
) -> tuple[bool, BaseException | None]:
    if task_store is None:
        # Preserve the established run lifecycle: a missing TaskStore is
        # terminalized after session creation with ordinary durable failure
        # evidence. With no store there is no contract authority to inspect
        # or admit at this earlier verifier-aware boundary.
        return False, None

    # An explicit task is authoritative evidence in its own right. Inspect
    # it even when the store does not implement the complete verified-work
    # lifecycle: an adopted/custom store must not turn a materialized
    # contract binding into ordinary execution merely through its
    # capability flag.
    if task_id is not None:
        task_outcome = await capture_task_store_operation(
            lambda: task_store.load_task(task_id),
            operation_name="Verified-work task lookup",
            redactor=redactor,
        )
        if task_outcome.failure is not None:
            return False, task_outcome.failure
        task = task_outcome.result
        del task_outcome
        if task is None:
            if not allow_missing_task:
                return False, KeyError("Task not found.")
        else:
            validation = capture_sensitive_validation(
                lambda task=task: copy_task(task),
                operation_name="Verified-work task lookup result validation",
                redactor=redactor,
            )
            del task
            if validation.failure is not None:
                return False, validation.failure
            task = validation.result
            del validation
            if task is None:
                return False, RuntimeError("Task store returned an invalid task lookup result.")
            if task.id != task_id:
                del task
                return False, RuntimeError(
                    "Task store returned a task other than the exact requested task."
                )
            requires_completion_decision = task.work_contract is not None
            del task
            if requires_completion_decision:
                return True, None

    if session_id is None:
        return False, None
    session_contract_lookup_supported = (
        type(task_store).load_active_work_contract_task_for_session
        is not TaskStore.load_active_work_contract_task_for_session
    )
    session_admission_supported = (
        type(task_store).admit_ordinary_session_execution
        is not TaskStore.admit_ordinary_session_execution
    )
    if (
        not task_store.supports_verified_work_contracts
        and not session_contract_lookup_supported
        and not session_admission_supported
    ):
        return False, None
    if not admit_session:
        if (
            not task_store.supports_verified_work_contracts
            and not session_contract_lookup_supported
        ):
            return False, None
        task_outcome = await capture_task_store_operation(
            lambda: task_store.load_active_work_contract_task_for_session(session_id),
            operation_name="Verified-work session lookup",
            redactor=redactor,
        )
        if task_outcome.failure is not None:
            return False, task_outcome.failure
        task = task_outcome.result
        del task_outcome
        if task is not None and type(task) is not Task:
            del task
            return False, RuntimeError(
                "Task store returned an invalid session contract lookup result."
            )
        requires_completion_decision = task is not None
        del task
        return requires_completion_decision, None
    admission_outcome = await capture_task_store_operation(
        lambda: task_store.admit_ordinary_session_execution(session_id),
        operation_name="Ordinary session execution admission",
        redactor=redactor,
        mutation_store=task_store,
        mutation_method_name="admit_ordinary_session_execution",
    )
    if type(admission_outcome.failure) is TaskCompletionDecisionRequired:
        return True, None
    if admission_outcome.failure is not None:
        return False, admission_outcome.failure
    return False, None

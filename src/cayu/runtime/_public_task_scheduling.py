"""Application-facing scheduling validation and exact store-result checks."""

from __future__ import annotations

from functools import partial

from cayu._validation import (
    MAX_PORTABLE_JSON_INTEGER,
    canonical_durable_json_bytes,
    require_durable_clean_nonblank,
    revalidate_model_input,
)
from cayu.runtime._task_store_operation_boundary import (
    capture_sensitive_validation,
    capture_task_store_operation,
    raise_task_store_operation_failure,
)
from cayu.runtime._verified_work_authority import invocation_contains_secret_public_identity
from cayu.tasks._scheduling import schedule_creation_digest, schedule_mutation_digest
from cayu.tasks.base import (
    Task,
    TaskCreate,
    TaskInvocationSnapshot,
    TaskStore,
    copy_task,
    copy_task_create,
    task_invocation_for_create,
)
from cayu.tasks.scheduling import (
    TASK_SCHEDULE_ID_MAX_BYTES,
    TaskRescheduleRequest,
    TaskScheduleCancelRequest,
    TaskScheduleConflict,
    TaskScheduleEvent,
    TaskScheduleEventType,
    TaskScheduleReceipt,
)
from cayu.vaults.redaction import SecretRedactor


async def _load_schedule_invocation(
    store: TaskStore, task_id: str, *, redactor: SecretRedactor
) -> TaskInvocationSnapshot:
    outcome = await capture_task_store_operation(
        partial(store.load_invocation_snapshot, task_id),
        operation_name="Scheduled task invocation lookup",
        redactor=redactor,
    )
    if outcome.failure is not None:
        raise_task_store_operation_failure(outcome.failure)
    if type(outcome.result) is not TaskInvocationSnapshot:
        raise TaskScheduleConflict("Scheduled task invocation authority is unavailable.")
    validation = capture_sensitive_validation(
        lambda: revalidate_model_input(outcome.result, TaskInvocationSnapshot),
        operation_name="Scheduled task invocation validation",
        redactor=redactor,
    )
    if validation.failure is not None:
        raise_task_store_operation_failure(validation.failure)
    snapshot = validation.result
    if snapshot is None or snapshot.id != task_id:
        raise TaskScheduleConflict("Scheduled task invocation identity conflicts.")
    if invocation_contains_secret_public_identity(snapshot.invocation, redactor):
        raise TaskScheduleConflict("Scheduled task invocation identity is unsafe.")
    return snapshot


async def create_scheduled_task(
    store: TaskStore, request: TaskCreate, *, redactor: SecretRedactor
) -> Task:
    """Publish an already-validated application request through the owned boundary."""
    expected_id = request.task_id
    if expected_id is None or redactor.redact_text(expected_id) != expected_id:
        raise ValueError("Scheduled task requires a secret-free stable identity.")
    parent = (
        None
        if request.parent_task_id is None
        else await _load_schedule_invocation(store, request.parent_task_id, redactor=redactor)
    )
    # Use the native derivation rules rather than duplicating origin/source
    # policy here. A new root's generated UUID is store-owned; inherited root
    # identity, origin and execution source are request-bound.
    authority = capture_sensitive_validation(
        lambda: task_invocation_for_create(request, task_id=expected_id, parent_task=parent),
        operation_name="Scheduled task creation authority",
        redactor=redactor,
    )
    if authority.failure is not None:
        raise_task_store_operation_failure(authority.failure)
    expected_invocation = authority.result
    if expected_invocation is None:
        raise TaskScheduleConflict("Scheduled task creation authority is unavailable.")
    if (
        parent is not None or request._runtime_session_binding is not None
    ) and invocation_contains_secret_public_identity(expected_invocation, redactor):
        raise TaskScheduleConflict("Scheduled task creation authority is unsafe.")
    digest = schedule_creation_digest(request)
    dispatched = copy_task_create(request)
    outcome = await capture_task_store_operation(
        partial(store.create_task, dispatched),
        operation_name="Scheduled task creation",
        redactor=redactor,
        mutation_store=store,
        mutation_method_name="create_task",
    )
    if outcome.failure is not None:
        raise_task_store_operation_failure(outcome.failure)
    if type(outcome.result) is not Task:
        raise TaskScheduleConflict("Task store returned invalid scheduled task evidence.")
    validation = capture_sensitive_validation(
        lambda value=outcome.result: copy_task(value),
        operation_name="Scheduled task creation evidence",
        redactor=redactor,
    )
    if validation.failure is not None:
        raise_task_store_operation_failure(validation.failure)
    task = validation.result
    if (
        task is None
        or task.id != expected_id
        or task.schedule is None
        or task.schedule.creation_sha256 != digest
        or task.type != request.type
        or task.title != request.title
        or task.description != request.description
        or task.parent_task_id != request.parent_task_id
        or task.assigned_agent_name != request.assigned_agent_name
        or canonical_durable_json_bytes(task.input, "task input")
        != canonical_durable_json_bytes(request.input, "task input")
        or canonical_durable_json_bytes(task.metadata, "task metadata")
        != canonical_durable_json_bytes(request.metadata, "task metadata")
        or task.work_contract != request.work_contract
        or (None if task.retry_series is None else task.retry_series.policy) != request.retry_policy
    ):
        raise TaskScheduleConflict("Task store returned conflicting scheduled task evidence.")
    durable = await _load_schedule_invocation(store, expected_id, redactor=redactor)
    # A worker may attach after create_task captured an unattached snapshot.
    # Returning that earlier snapshot grants no session authority. A returned
    # attachment, however, must match the complete durable incarnation pair;
    # do not splice newer attachment fields into the older lifecycle snapshot.
    if (
        task.invocation != durable.invocation
        or (durable.session_id is None) != (durable.session_instance_id is None)
        or (
            task.session_id is not None
            and (
                task.session_instance_id is None
                or task.session_id != durable.session_id
                or task.session_instance_id != durable.session_instance_id
            )
        )
        or task.invocation.origin != expected_invocation.origin
        or task.invocation.source != expected_invocation.source
        or task.invocation.root_session_id != expected_invocation.root_session_id
        or (
            (parent is not None or request._runtime_session_binding is not None)
            and task.invocation.root_invocation_id != expected_invocation.root_invocation_id
        )
    ):
        raise TaskScheduleConflict("Task store returned conflicting scheduled task authority.")
    return task


async def publish_task_schedule(
    store: TaskStore,
    request: TaskRescheduleRequest | TaskScheduleCancelRequest,
    *,
    redactor: SecretRedactor,
) -> TaskScheduleReceipt:
    if type(request) not in {TaskRescheduleRequest, TaskScheduleCancelRequest}:
        raise TypeError("A typed schedule mutation request is required.")
    validation = capture_sensitive_validation(
        lambda: revalidate_model_input(request, TaskRescheduleRequest, TaskScheduleCancelRequest),
        operation_name="Task schedule request validation",
        redactor=redactor,
    )
    del request
    if validation.failure is not None:
        raise_task_store_operation_failure(validation.failure)
    if validation.result is None:
        raise ValueError("Task schedule mutation request is invalid.") from None
    request = validation.result
    for identity in (request.task_id, request.operation_id):
        if redactor.redact_text(identity) != identity:
            raise ValueError("Task schedule identity contains a workload secret.") from None
    if not store.supports_task_scheduling:
        raise NotImplementedError("Task store does not support managed task scheduling.")
    digest = schedule_mutation_digest(request)
    # Keep expected authority separate from the extension-owned argument. A
    # frozen model is not proof against an adapter mutating nested fields or
    # using object.__setattr__ before returning its receipt.
    dispatched_request = revalidate_model_input(
        request, TaskRescheduleRequest, TaskScheduleCancelRequest
    )
    if isinstance(request, TaskRescheduleRequest):
        operation = partial(store.reschedule_task, dispatched_request)
        method = "reschedule_task"
    else:
        operation = partial(store.cancel_scheduled_task, dispatched_request)
        method = "cancel_scheduled_task"
    outcome = await capture_task_store_operation(
        operation,
        operation_name="Task schedule mutation",
        redactor=redactor,
        mutation_store=store,
        mutation_method_name=method,
    )
    if outcome.failure is not None:
        raise_task_store_operation_failure(outcome.failure)
    if type(outcome.result) is not TaskScheduleReceipt:
        raise TaskScheduleConflict("Task store returned invalid schedule evidence.") from None
    validation = capture_sensitive_validation(
        lambda value=outcome.result: revalidate_model_input(value, TaskScheduleReceipt),
        operation_name="Task schedule receipt validation",
        redactor=redactor,
    )
    del outcome
    if validation.failure is not None:
        raise_task_store_operation_failure(validation.failure)
    if validation.result is None:
        raise TaskScheduleConflict("Task store returned invalid schedule evidence.") from None
    receipt = validation.result
    if (
        receipt.task_id != request.task_id
        or receipt.operation_id != request.operation_id
        or receipt.expected_revision != request.expected_revision
        or receipt.request_sha256 != digest
    ):
        raise TaskScheduleConflict(
            "Task store returned evidence for a different schedule mutation."
        )
    if isinstance(request, TaskRescheduleRequest):
        if (
            receipt.type is not TaskScheduleEventType.RESCHEDULED
            or receipt.available_at != request.available_at
            or receipt.schedule.policy != request.policy
        ):
            raise TaskScheduleConflict("Task store changed the requested schedule.")
    elif receipt.type not in {
        TaskScheduleEventType.CANCELLED,
        TaskScheduleEventType.CANCELLATION_REQUESTED,
    }:
        raise TaskScheduleConflict("Task store did not acknowledge schedule cancellation.")
    return receipt


async def inspect_task_schedule_events(
    store: TaskStore,
    task_id: str,
    *,
    after_sequence: int,
    limit: int,
    redactor: SecretRedactor,
) -> list[TaskScheduleEvent]:
    """Validate a bounded page's types, task binding and order before exposure."""
    if type(task_id) is not str or len(task_id) > TASK_SCHEDULE_ID_MAX_BYTES:
        raise ValueError("Invalid schedule task identity.")
    task_id = require_durable_clean_nonblank(task_id, "task_id")
    if len(task_id.encode("utf-8")) > TASK_SCHEDULE_ID_MAX_BYTES:
        raise ValueError("Invalid schedule task identity.")
    if redactor.redact_text(task_id) != task_id:
        raise ValueError("Task schedule identity contains a workload secret.")
    if type(after_sequence) is not int or not 0 <= after_sequence <= MAX_PORTABLE_JSON_INTEGER:
        raise ValueError("Invalid schedule event cursor.")
    if type(limit) is not int or not 1 <= limit <= 1000:
        raise ValueError("Invalid schedule event limit.")
    if not store.supports_task_scheduling:
        raise NotImplementedError("Task store does not support managed task scheduling.")
    outcome = await capture_task_store_operation(
        partial(
            store.list_task_schedule_events, task_id, after_sequence=after_sequence, limit=limit
        ),
        operation_name="Task schedule inspection",
        redactor=redactor,
    )
    if outcome.failure is not None:
        raise_task_store_operation_failure(outcome.failure)
    if type(outcome.result) is not list or len(outcome.result) > limit:
        raise TaskScheduleConflict("Task store returned invalid schedule history.")
    copied: list[TaskScheduleEvent] = []
    previous = after_sequence
    for event in outcome.result:
        if type(event) is not TaskScheduleEvent:
            raise TaskScheduleConflict("Task store returned invalid schedule history.")
        validation = capture_sensitive_validation(
            lambda value=event: revalidate_model_input(value, TaskScheduleEvent),
            operation_name="Task schedule event validation",
            redactor=redactor,
        )
        if validation.failure is not None:
            raise_task_store_operation_failure(validation.failure)
        checked = validation.result
        if checked is None or checked.task_id != task_id or checked.sequence <= previous:
            raise TaskScheduleConflict("Task store returned conflicting schedule history.")
        previous = checked.sequence
        copied.append(checked)
    return copied

"""Application task creation and work-contract publication with explicit dependencies.

The public application keeps lifecycle admission. These operations own request
validation, store dispatch and exact result authentication, using the existing
cancellation-quiescent boundary for contract-bound mutations.
"""

from __future__ import annotations

from typing import cast

from cayu.runtime._public_task_scheduling import create_scheduled_task
from cayu.runtime._task_store_operation_boundary import (
    TaskStoreOperationOutcome,
    capture_sensitive_validation,
    capture_task_store_operation,
    raise_task_store_operation_failure,
)
from cayu.sessions.base import SessionStore
from cayu.sessions.invocation import InvocationOrigin, InvocationOriginTrust, TaskExecutionSource
from cayu.tasks._scheduling import schedule_creation_digest
from cayu.tasks._verified_work_authority import invocation_contains_secret_public_identity
from cayu.tasks.contracts import (
    WorkCompletionConflict,
    WorkContract,
    WorkContractConflict,
    WorkContractDraft,
    WorkContractRef,
    copy_work_contract,
    copy_work_contract_ref,
    work_contract_from_draft,
)
from cayu.tasks.creation import (
    TaskCreate,
    TaskInvocationSnapshot,
    copy_task_create,
    preflight_contract_bound_task_creation,
    require_contract_bound_task_creation_snapshot,
    task_create_with_runtime_invocation,
)
from cayu.tasks.records import Task, TaskStatus, copy_task
from cayu.tasks.store import TaskStore
from cayu.vaults.redaction import SecretRedactor


async def create_work_contract(
    request: WorkContractDraft,
    *,
    task_store: TaskStore | None,
    redactor: SecretRedactor,
) -> WorkContract:
    try:
        if type(request) is not WorkContractDraft:
            del request
            raise TypeError("Work-contract creation requires a WorkContractDraft request.")
        if task_store is None:
            del request
            raise RuntimeError("task_store is required to create work contracts.")
        if not task_store.supports_verified_work_contracts:
            del request
            raise NotImplementedError(
                f"{type(task_store).__name__} does not support verified work contracts."
            )
        validation = _validated_public_work_contract(
            request,
            redactor=redactor,
        )
        del request
        validation_failure = validation.failure
        contract = validation.result
        del validation
        if validation_failure is not None:
            raise validation_failure from None
        if contract is None:
            raise ValueError("Work-contract creation request is invalid.") from None
        contains_secret_identity = _work_contract_contains_secret_public_identity(
            contract,
            redactor,
        )
        if contains_secret_identity:
            del contract
            raise ValueError(
                "Work-contract public identity contains a workload secret and cannot be published."
            ) from None
        published_value, publication_failure = await _publish_public_work_contract(
            task_store,
            contract,
            redactor=redactor,
        )
        if publication_failure is not None:
            del contract, published_value
            raise_task_store_operation_failure(publication_failure)
        validation = _copied_public_work_contract(
            published_value,
            redactor=redactor,
        )
        del published_value
        validation_failure = validation.failure
        published = validation.result
        del validation
        if validation_failure is not None:
            del contract
            raise validation_failure from None
        if published is None:
            del contract
            raise WorkContractConflict(
                "Task store returned an invalid published work contract."
            ) from None
        if published != contract:
            del contract, published
            raise WorkContractConflict(
                "Task store returned a work contract other than the exact published definition."
            ) from None
        return published
    finally:
        # Extension dependencies may have sensitive representations in tracebacks.
        del task_store, redactor


async def load_work_contract(
    reference: WorkContractRef,
    *,
    task_store: TaskStore | None,
    redactor: SecretRedactor,
) -> WorkContract | None:
    try:
        if type(reference) is not WorkContractRef:
            del reference
            raise TypeError("Work-contract lookup requires a WorkContractRef.")
        if task_store is None:
            del reference
            raise RuntimeError("task_store is required to load work contracts.")
        if not task_store.supports_verified_work_contracts:
            del reference
            raise NotImplementedError(
                f"{type(task_store).__name__} does not support verified work contracts."
            )
        validation = _copied_public_work_contract_ref(
            reference,
            redactor=redactor,
        )
        del reference
        validation_failure = validation.failure
        copied_reference = validation.result
        del validation
        if validation_failure is not None:
            raise validation_failure from None
        if copied_reference is None:
            raise ValueError("Work-contract lookup reference is invalid.") from None
        contains_secret_identity = (
            redactor.redact_text(copied_reference.contract_id) != copied_reference.contract_id
        )
        if contains_secret_identity:
            del copied_reference
            raise ValueError(
                "Work-contract identity contains a workload secret and cannot be used for lookup."
            ) from None
        loaded_value, lookup_failure = await _load_public_work_contract(
            task_store,
            copied_reference,
            redactor=redactor,
        )
        if lookup_failure is not None:
            del copied_reference, loaded_value
            raise_task_store_operation_failure(lookup_failure)
        if loaded_value is None:
            return None
        validation = _copied_public_work_contract(
            loaded_value,
            redactor=redactor,
        )
        del loaded_value
        validation_failure = validation.failure
        loaded = validation.result
        del validation
        if validation_failure is not None:
            del copied_reference
            raise validation_failure from None
        if loaded is None:
            del copied_reference
            raise WorkContractConflict("Task store returned an invalid work contract.") from None
        if loaded.reference() != copied_reference:
            del copied_reference, loaded
            raise WorkContractConflict(
                "Task store returned a work contract other than the exact requested version."
            ) from None
        contains_secret_identity = _work_contract_contains_secret_public_identity(
            loaded,
            redactor,
        )
        if contains_secret_identity:
            del copied_reference, loaded
            raise ValueError(
                "Loaded work contract contains a workload secret in a public identity."
            ) from None
        return loaded
    finally:
        # Extension dependencies may have sensitive representations in tracebacks.
        del task_store, redactor


async def create_task(
    request: TaskCreate,
    *,
    task_store: TaskStore | None,
    session_store: SessionStore,
    redactor: SecretRedactor,
) -> Task:
    try:
        if type(request) is not TaskCreate:
            del request
            raise TypeError("Task creation requires a TaskCreate request.")
        if request.work_contract is None:
            # Preserve the established ordinary-task validation contract,
            # including actionable Pydantic field diagnostics. Contract-bound
            # requests use the detached boundary below because their durable
            # authority fields may contain workload-sensitive material.
            request = copy_task_create(request)
        else:
            validation = _copied_public_task_create(
                request,
                redactor=redactor,
            )
            del request
            validation_failure = validation.failure
            copied_request = validation.result
            del validation
            if validation_failure is not None:
                raise validation_failure from None
            if copied_request is None:
                raise ValueError("Task creation request is invalid.") from None
            request = copied_request
            del copied_request
        if task_store is None:
            del request
            raise RuntimeError("task_store is required to create tasks.")
        if (
            request.retry_policy is not None
            and redactor.redact_uppercase_text(request.retry_policy.cost_currency)
            != request.retry_policy.cost_currency
        ):
            raise ValueError(
                "Task retry cost currency contains a workload secret and cannot be used "
                "as durable accounting authority."
            ) from None
        for origin in (request.invocation_origin, request._verified_invocation_origin):
            if origin is None:
                continue
            for value in (origin.subject, origin.tenant):
                if value is not None and redactor.redact_text(value) != value:
                    del origin, request, value
                    raise ValueError(
                        "Task invocation origin contains a workload secret and cannot be "
                        "used as durable task authority."
                    ) from None
        if request.work_contract is not None:
            if not task_store.supports_verified_work_contracts:
                del request
                raise NotImplementedError(
                    f"{type(task_store).__name__} does not support verified work contracts."
                )
            if request.task_id is None:
                del request
                raise ValueError(
                    "Contracted task creation requires a caller-stable task_id for "
                    "cancellation reconciliation."
                ) from None
            if redactor.redact_text(request.task_id) != request.task_id:
                del request
                raise ValueError(
                    "Task identity contains a workload secret and cannot be exposed "
                    "through durable task projections."
                ) from None
            contains_secret_identity = (
                redactor.redact_text(request.work_contract.contract_id)
                != request.work_contract.contract_id
            )
            if contains_secret_identity:
                del request
                raise ValueError(
                    "Work-contract identity contains a workload secret and cannot be exposed "
                    "through durable task projections."
                ) from None
        if (
            request.session_id is not None
            and request._verified_invocation_origin is None
            and request._runtime_session_binding is None
        ):
            snapshot = await session_store.load_invocation_snapshot(request.session_id)
            if snapshot is not None:
                request = task_create_with_runtime_invocation(
                    request,
                    source=(request._runtime_invocation_source or TaskExecutionSource.SDK_TASK),
                    session_invocation=snapshot,
                )
            del snapshot
        if request.available_at is not None and not task_store.supports_delayed_availability:
            del request
            raise NotImplementedError(
                f"{type(task_store).__name__} does not support delayed task availability."
            )
        if request.schedule_policy is not None and not task_store.supports_task_scheduling:
            del request
            raise NotImplementedError(
                f"{type(task_store).__name__} does not support managed task scheduling."
            )
        if request.retry_policy is not None and not task_store.supports_task_retry_series:
            del request
            raise NotImplementedError(
                f"{type(task_store).__name__} does not support task retry series."
            )
        if request.work_contract is None:
            if request.schedule_policy is not None:
                return await create_scheduled_task(task_store, request, redactor=redactor)
            return await task_store.create_task(request)
        parent_invocation_snapshot: TaskInvocationSnapshot | None = None
        if request.parent_task_id is not None:
            (
                parent_snapshot_value,
                parent_lookup_failure,
            ) = await _load_public_task_invocation_snapshot(
                task_store,
                request.parent_task_id,
                redactor=redactor,
            )
            if parent_lookup_failure is not None:
                del parent_snapshot_value, request
                raise_task_store_operation_failure(parent_lookup_failure)
            parent_validation = _copied_public_task_invocation_snapshot(
                parent_snapshot_value,
                redactor=redactor,
            )
            del parent_snapshot_value
            parent_validation_failure = parent_validation.failure
            parent_invocation_snapshot = parent_validation.result
            del parent_validation
            if parent_validation_failure is not None:
                del parent_invocation_snapshot, request
                raise parent_validation_failure from None
            if (
                parent_invocation_snapshot is None
                or parent_invocation_snapshot.id != request.parent_task_id
            ):
                del parent_invocation_snapshot, request
                raise WorkContractConflict(
                    "Task parent invocation authority is unavailable for contracted creation."
                ) from None
        session_invocation_contains_secret = (
            request._runtime_session_binding is not None
            and invocation_contains_secret_public_identity(
                request._runtime_session_binding.invocation,
                redactor,
            )
        )
        parent_invocation_contains_secret = (
            parent_invocation_snapshot is not None
            and invocation_contains_secret_public_identity(
                parent_invocation_snapshot.invocation,
                redactor,
            )
        )
        direct_root_session_contains_secret = (
            request._runtime_session_binding is None
            and parent_invocation_snapshot is None
            and request.session_id is not None
            and redactor.redact_text(request.session_id) != request.session_id
        )
        if (
            session_invocation_contains_secret
            or parent_invocation_contains_secret
            or direct_root_session_contains_secret
        ):
            del parent_invocation_snapshot, request
            raise ValueError(
                "Task invocation identity contains a workload secret and cannot be exposed "
                "through durable task projections."
            ) from None
        preflight_contract_bound_task_creation(
            request,
            parent_task=parent_invocation_snapshot,
        )
        task, creation_failure = await _create_public_contracted_task(
            task_store,
            request,
            redactor=redactor,
        )
        if creation_failure is not None:
            del parent_invocation_snapshot, request, task
            raise_task_store_operation_failure(creation_failure)
        validation = _copied_public_task(
            task,
            redactor=redactor,
        )
        del task
        validation_failure = validation.failure
        copied_task = validation.result
        del validation
        if validation_failure is not None:
            del parent_invocation_snapshot, request
            raise validation_failure from None
        if copied_task is None:
            del parent_invocation_snapshot, request
            raise WorkContractConflict("Task store returned an invalid contracted task.") from None
        task = copied_task
        del copied_task
        try:
            # The request was preflighted against creation headroom before
            # dispatch. A managed replay may now contain legitimate lifecycle
            # growth; copy_task already enforces the full task bound.
            if request.schedule_policy is None or (
                task.schedule is not None and task.schedule.revision == 1
            ):
                require_contract_bound_task_creation_snapshot(task)
        except (TypeError, ValueError):
            del parent_invocation_snapshot, request, task
            raise WorkContractConflict(
                "Task store returned a contracted task outside the creation-snapshot bounds."
            ) from None
        (
            invocation_snapshot,
            invocation_lookup_failure,
        ) = await _load_public_task_invocation_snapshot(
            task_store,
            task.id,
            redactor=redactor,
        )
        if invocation_lookup_failure is not None:
            del invocation_snapshot, parent_invocation_snapshot, request, task
            raise_task_store_operation_failure(invocation_lookup_failure)
        validation = _copied_public_task_invocation_snapshot(
            invocation_snapshot,
            redactor=redactor,
        )
        del invocation_snapshot
        validation_failure = validation.failure
        copied_invocation_snapshot = validation.result
        del validation
        if validation_failure is not None:
            del parent_invocation_snapshot, request, task
            raise validation_failure from None
        if not _contracted_task_creation_result_matches_request(
            task=task,
            request=request,
            invocation_snapshot=copied_invocation_snapshot,
            parent_invocation_snapshot=parent_invocation_snapshot,
            redactor=redactor,
        ):
            del copied_invocation_snapshot, parent_invocation_snapshot, request, task
            raise WorkContractConflict(
                "Task store did not preserve the exact contracted task creation request."
            ) from None
        del copied_invocation_snapshot, parent_invocation_snapshot
        return task
    finally:
        # Extension dependencies may have sensitive representations in tracebacks.
        del task_store, session_store, redactor


def _work_contract_contains_secret_public_identity(
    contract: WorkContract,
    redactor: SecretRedactor,
) -> bool:
    public_identities = (
        contract.contract_id,
        contract.verifier.verifier_id,
        contract.verifier.version,
        contract.verifier.configuration_fingerprint,
        contract.result_resolver.resolver_id,
        contract.result_resolver.version,
        contract.result_resolver.configuration_fingerprint,
        *(criterion.criterion_id for criterion in contract.criteria),
        *(constraint.constraint_id for constraint in contract.constraints),
        *(requirement.requirement_id for requirement in contract.evidence_requirements),
    )
    return any(redactor.redact_text(value) != value for value in public_identities)


async def _publish_public_work_contract(
    task_store: TaskStore,
    contract: WorkContract,
    *,
    redactor: SecretRedactor,
) -> tuple[WorkContract | None, BaseException | None]:
    """Capture a publication conflict without exporting its sensitive store traceback."""

    outcome = await capture_task_store_operation(
        lambda: task_store.publish_work_contract(contract),
        operation_name="Work-contract publication",
        redactor=redactor,
        mutation_store=task_store,
        mutation_method_name="publish_work_contract",
    )
    if type(outcome.failure) is WorkContractConflict:
        return None, WorkContractConflict(
            "Task store rejected the work-contract publication because its durable identity "
            "conflicts with existing state."
        )
    return outcome.result, outcome.failure


async def _load_public_work_contract(
    task_store: TaskStore,
    reference: WorkContractRef,
    *,
    redactor: SecretRedactor,
) -> tuple[WorkContract | None, BaseException | None]:
    """Capture a lookup conflict without exporting the stored contract traceback."""

    outcome = await capture_task_store_operation(
        lambda: task_store.load_work_contract(reference),
        operation_name="Work-contract lookup",
        redactor=redactor,
    )
    if type(outcome.failure) is WorkContractConflict:
        return None, WorkContractConflict(
            "Task store rejected the work-contract lookup because its durable identity "
            "conflicts with the requested reference."
        )
    return outcome.result, outcome.failure


async def _create_public_contracted_task(
    task_store: TaskStore,
    request: TaskCreate,
    *,
    redactor: SecretRedactor,
) -> tuple[Task | None, BaseException | None]:
    """Capture contract-binding conflicts without exporting the task payload traceback."""

    dispatched = copy_task_create(request) if request.schedule_policy is not None else request
    outcome = await capture_task_store_operation(
        lambda: task_store.create_task(dispatched),
        operation_name="Contracted task creation",
        redactor=redactor,
        mutation_store=task_store,
        mutation_method_name="create_task",
    )
    if type(outcome.failure) is WorkContractConflict:
        return None, WorkContractConflict(
            "Task store rejected the contracted task because its work contract conflicts "
            "with durable state."
        )
    if type(outcome.failure) is WorkCompletionConflict:
        return None, WorkCompletionConflict(
            "Task store rejected the contracted task because its session binding conflicts "
            "with durable work authority."
        )
    return outcome.result, outcome.failure


async def _load_public_task_invocation_snapshot(
    task_store: TaskStore,
    task_id: str,
    *,
    redactor: SecretRedactor,
) -> tuple[TaskInvocationSnapshot | None, BaseException | None]:
    """Read back one contracted task's durable provenance without leaking extension state."""

    outcome = await capture_task_store_operation(
        lambda: task_store.load_invocation_snapshot(task_id),
        operation_name="Contracted task invocation lookup",
        redactor=redactor,
    )
    return outcome.result, outcome.failure


def _validated_public_work_contract(
    draft: WorkContractDraft,
    *,
    redactor: SecretRedactor,
) -> TaskStoreOperationOutcome[WorkContract]:
    """Validate a caller-owned draft without retaining a rejected model traceback."""

    return capture_sensitive_validation(
        lambda: work_contract_from_draft(draft),
        operation_name="Work-contract request validation",
        redactor=redactor,
    )


def _copied_public_work_contract(
    value: object,
    *,
    redactor: SecretRedactor,
) -> TaskStoreOperationOutcome[WorkContract]:
    """Copy one extension-returned contract behind a detached validation boundary."""

    if type(value) is not WorkContract:
        return TaskStoreOperationOutcome()
    return capture_sensitive_validation(
        lambda: copy_work_contract(value),
        operation_name="Work-contract result validation",
        redactor=redactor,
    )


def _copied_public_work_contract_ref(
    value: WorkContractRef,
    *,
    redactor: SecretRedactor,
) -> TaskStoreOperationOutcome[WorkContractRef]:
    """Copy one caller-owned reference behind a detached validation boundary."""

    return capture_sensitive_validation(
        lambda: cast("WorkContractRef", copy_work_contract_ref(value)),
        operation_name="Work-contract reference validation",
        redactor=redactor,
    )


def _copied_public_task_create(
    value: TaskCreate,
    *,
    redactor: SecretRedactor,
) -> TaskStoreOperationOutcome[TaskCreate]:
    """Copy one caller-owned task request behind a detached validation boundary."""

    return capture_sensitive_validation(
        lambda: copy_task_create(value),
        operation_name="Task request validation",
        redactor=redactor,
    )


def _copied_public_task(
    value: object,
    *,
    redactor: SecretRedactor,
) -> TaskStoreOperationOutcome[Task]:
    """Copy one extension-returned task behind a detached validation boundary."""

    if type(value) is not Task:
        return TaskStoreOperationOutcome()
    return capture_sensitive_validation(
        lambda: copy_task(value),
        operation_name="Task result validation",
        redactor=redactor,
    )


def _copied_public_task_invocation_snapshot(
    value: object,
    *,
    redactor: SecretRedactor,
) -> TaskStoreOperationOutcome[TaskInvocationSnapshot]:
    """Copy extension-returned task provenance behind a detached validation boundary."""

    if type(value) is not TaskInvocationSnapshot:
        return TaskStoreOperationOutcome()
    return capture_sensitive_validation(
        lambda: TaskInvocationSnapshot(
            id=value.id,
            session_id=value.session_id,
            session_instance_id=value.session_instance_id,
            invocation=value.invocation,
        ),
        operation_name="Task invocation result validation",
        redactor=redactor,
    )


def _contracted_task_invocation_matches_request(
    *,
    task: Task,
    request: TaskCreate,
    invocation_snapshot: TaskInvocationSnapshot,
    parent_invocation_snapshot: TaskInvocationSnapshot | None,
    redactor: SecretRedactor,
) -> bool:
    """Authenticate durable provenance and every request-owned invocation field."""

    invocation = task.invocation
    if (
        invocation_snapshot.id != task.id
        or invocation_snapshot.session_id != task.session_id
        or invocation_snapshot.session_instance_id != task.session_instance_id
        or invocation_snapshot.invocation != invocation
        or invocation.source
        is not (request._runtime_invocation_source or TaskExecutionSource.SDK_TASK)
    ):
        return False
    if redactor.redact_text(task.id) != task.id or invocation_contains_secret_public_identity(
        invocation,
        redactor,
    ):
        return False
    session_binding = request._runtime_session_binding
    if session_binding is not None:
        matches_session = (
            task.session_instance_id == session_binding.session_instance_id
            and invocation_snapshot.session_instance_id == session_binding.session_instance_id
            and invocation.origin == session_binding.invocation.origin
            and invocation.root_invocation_id == session_binding.invocation.root_invocation_id
            and invocation.root_session_id == session_binding.invocation.root_session_id
        )
        if not matches_session:
            return False
        if request.parent_task_id is None:
            return True
        return (
            parent_invocation_snapshot is not None
            and parent_invocation_snapshot.id == request.parent_task_id
            and parent_invocation_snapshot.invocation.origin == invocation.origin
            and parent_invocation_snapshot.invocation.root_invocation_id
            == invocation.root_invocation_id
        )
    if request.parent_task_id is not None:
        return (
            parent_invocation_snapshot is not None
            and parent_invocation_snapshot.id == request.parent_task_id
            and parent_invocation_snapshot.invocation.origin == invocation.origin
            and parent_invocation_snapshot.invocation.root_invocation_id
            == invocation.root_invocation_id
            and parent_invocation_snapshot.invocation.root_session_id == invocation.root_session_id
        )
    if request._verified_invocation_origin is not None:
        expected_origin = request._verified_invocation_origin
    elif request.invocation_origin is not None:
        expected_origin = InvocationOrigin(
            trust=InvocationOriginTrust.HOST_ASSERTED,
            subject=request.invocation_origin.subject,
            tenant=request.invocation_origin.tenant,
        )
    else:
        expected_origin = InvocationOrigin(trust=InvocationOriginTrust.UNATTRIBUTED)
    return invocation.origin == expected_origin and invocation.root_session_id == request.session_id


def _contracted_task_creation_result_matches_request(
    *,
    task: Task,
    request: TaskCreate,
    invocation_snapshot: TaskInvocationSnapshot | None,
    parent_invocation_snapshot: TaskInvocationSnapshot | None,
    redactor: SecretRedactor,
) -> bool:
    """Authenticate a custom store's contracted-create result before publication."""

    if invocation_snapshot is None or not _contracted_task_invocation_matches_request(
        task=task,
        request=request,
        invocation_snapshot=invocation_snapshot,
        parent_invocation_snapshot=parent_invocation_snapshot,
        redactor=redactor,
    ):
        return False
    if request.task_id is not None and task.id != request.task_id:
        return False
    if (
        task.type != request.type
        or task.title != request.title
        or task.description != request.description
        or (request.schedule_policy is None and task.session_id != request.session_id)
        or task.parent_task_id != request.parent_task_id
        or task.assigned_agent_name != request.assigned_agent_name
        or (request.schedule_policy is None and task.available_at != request.available_at)
        or task.input != request.input
        or task.metadata != request.metadata
        or task.work_contract != request.work_contract
    ):
        return False
    if request.schedule_policy is not None:
        # Native replay returns the current occurrence, which may have moved,
        # attached, or settled. Authenticate immutable creation intent without
        # treating an old due time or pending status as current authority.
        return (
            task.schedule is not None
            and task.schedule.creation_sha256 == schedule_creation_digest(request)
        )
    return (
        task.status is TaskStatus.PENDING
        and task.worker_id is None
        and task.lease_expires_at is None
        and task.status_reason is None
        and task.status_payload is None
        and task.result is None
        and task.error is None
        and task.started_at is None
        and task.completed_at is None
    )

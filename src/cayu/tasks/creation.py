"""Task creation records and immutable invocation provenance."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from typing import Any, cast
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, field_validator, model_validator

from cayu._clock import normalize_utc_datetime
from cayu._validation import copy_durable_json_object, copy_durable_metadata, revalidate_model_input
from cayu._validation import require_durable_clean_nonblank as require_clean_nonblank
from cayu._validation import require_durable_nonblank as require_nonblank
from cayu.sessions.invocation import (
    InvocationOrigin,
    InvocationOriginClaim,
    InvocationOriginTrust,
    SessionInvocationBinding,
    TaskExecutionSource,
    TaskInvocation,
    copy_invocation_origin,
    copy_invocation_origin_claim,
    copy_session_invocation_binding,
    copy_task_invocation,
    inherited_task_invocation,
)
from cayu.tasks._scheduling import schedule_creation_digest
from cayu.tasks.contracts import (
    WORK_CONTRACT_TASK_CREATION_MAX_BYTES,
    WORK_CONTRACT_TASK_CREATION_MAX_ITEMS,
    WORK_CONTRACT_TASK_MAX_BYTES,
    WORK_CONTRACT_TASK_MAX_ITEMS,
    WorkContractRef,
    copy_work_contract_ref,
    require_bounded_work_completion_document,
    validate_work_completion_linked_id,
)
from cayu.tasks.records import (
    Task,
    TaskRetryPolicy,
    TaskStatus,
    _preflight_bounded_task_payloads,
    _task_retry_attempt_authority_sha256,
    _validate_task_retry_reconciliation_identity,
    copy_task,
)
from cayu.tasks.retry import _task_retry_series_id, _task_retry_series_snapshot
from cayu.tasks.scheduling import (
    TaskSchedulePolicy,
    TaskScheduleState,
    validate_task_schedule_window,
)


class TaskInvocationSnapshot(BaseModel):
    """Bounded task identity and immutable provenance for delegation boundaries."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    id: str
    session_id: str | None
    session_instance_id: str | None = None
    invocation: TaskInvocation

    @field_validator("id")
    @classmethod
    def validate_id(cls, value: str) -> str:
        return require_clean_nonblank(value, "id")

    @field_validator("session_id")
    @classmethod
    def validate_session_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return require_clean_nonblank(value, "session_id")

    @field_validator("session_instance_id")
    @classmethod
    def validate_session_instance_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return SessionInvocationBinding.validate_session_instance_id(value)

    @model_validator(mode="after")
    def validate_session_instance_binding(self) -> TaskInvocationSnapshot:
        if self.session_instance_id is not None and self.session_id is None:
            raise ValueError("Task invocation session instance requires a session_id.")
        return self

    @field_validator("invocation")
    @classmethod
    def copy_invocation(cls, value: TaskInvocation) -> TaskInvocation:
        return copy_task_invocation(value)


class TaskCreate(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    task_id: str | None = None
    type: str
    title: str | None = None
    description: str | None = None
    session_id: str | None = None
    parent_task_id: str | None = None
    assigned_agent_name: str | None = None
    available_at: datetime | None = None
    schedule_policy: TaskSchedulePolicy | None = None
    input: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    retry_policy: TaskRetryPolicy | None = None
    work_contract: WorkContractRef | None = None
    invocation_origin: InvocationOriginClaim | None = None
    _verified_invocation_origin: InvocationOrigin | None = PrivateAttr(default=None)
    _runtime_invocation_source: TaskExecutionSource | None = PrivateAttr(default=None)
    _runtime_session_binding: SessionInvocationBinding | None = PrivateAttr(default=None)

    @field_validator("schedule_policy", mode="before")
    @classmethod
    def copy_schedule_policy(cls, value: object) -> object:
        return revalidate_model_input(value, TaskSchedulePolicy)

    @model_validator(mode="after")
    def validate_schedule(self) -> TaskCreate:
        if self.schedule_policy is not None:
            if self.task_id is None or self.available_at is None:
                raise ValueError("Managed task schedules require task_id and available_at.")
            if self.session_id is not None:
                raise ValueError("Managed schedules start as unattached queue tasks.")
            validate_task_schedule_window(self.available_at, self.schedule_policy)
        return self

    @model_validator(mode="before")
    @classmethod
    def preflight_work_contract_payloads(cls, value: object) -> object:
        if type(value) is not dict:
            return value
        document = cast("dict[str, object]", value)
        if document.get("work_contract") is None:
            return value
        _preflight_bounded_task_payloads(document, ("input", "metadata"))
        return value

    @field_validator("input", "metadata", mode="before")
    @classmethod
    def copy_json_object(cls, value: dict[str, Any], info) -> dict[str, Any]:
        if info.field_name == "metadata":
            return copy_durable_metadata(value)
        return copy_durable_json_object(value, info.field_name)

    @field_validator("type")
    @classmethod
    def validate_nonblank_type(cls, value: str, info) -> str:
        return require_clean_nonblank(value, info.field_name)

    @field_validator(
        "task_id",
        "title",
        "description",
        "session_id",
        "parent_task_id",
        "assigned_agent_name",
    )
    @classmethod
    def validate_optional_nonblank_strings(
        cls,
        value: str | None,
        info,
    ) -> str | None:
        if value is None:
            return None
        if info.field_name in {"title", "description"}:
            return require_nonblank(value, info.field_name)
        return require_clean_nonblank(value, info.field_name)

    @field_validator("available_at")
    @classmethod
    def normalize_available_at(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        return normalize_utc_datetime(value, "available_at")

    @field_validator("work_contract", mode="before")
    @classmethod
    def copy_work_contract(cls, value: object) -> object:
        return revalidate_model_input(value, WorkContractRef)

    @model_validator(mode="after")
    def validate_retry_and_work_contract_shape(self) -> TaskCreate:
        if self.retry_policy is not None:
            if self.task_id is not None:
                _validate_task_retry_reconciliation_identity(self.task_id, "task_id")
            if self.session_id is not None:
                raise ValueError("Retry-series tasks must start as unattached queue work.")
            if self.work_contract is not None:
                raise ValueError("Retry-series tasks cannot use verified work contracts.")
        if self.work_contract is None:
            return self
        if self.task_id is not None:
            validate_work_completion_linked_id(self.task_id, "task_id")
        if self.session_id is not None:
            validate_work_completion_linked_id(self.session_id, "session_id")
        require_bounded_work_completion_document(
            self.model_dump(mode="json", warnings=False),
            "Contract-bound task creation request",
            max_bytes=WORK_CONTRACT_TASK_MAX_BYTES,
            max_items=WORK_CONTRACT_TASK_MAX_ITEMS,
        )
        return self


def copy_task_create(request: TaskCreate) -> TaskCreate:
    if type(request) is not TaskCreate:
        raise TypeError("Task creation requires a TaskCreate instance.")
    if request.work_contract is not None:
        _preflight_bounded_task_payloads(request, ("input", "metadata"))
    copied = TaskCreate(
        task_id=request.task_id,
        type=request.type,
        title=request.title,
        description=request.description,
        session_id=request.session_id,
        parent_task_id=request.parent_task_id,
        assigned_agent_name=request.assigned_agent_name,
        available_at=request.available_at,
        schedule_policy=request.schedule_policy,
        input=copy_durable_json_object(request.input, "input"),
        metadata=copy_durable_metadata(request.metadata),
        retry_policy=(
            None
            if request.retry_policy is None
            else TaskRetryPolicy.model_validate(
                request.retry_policy.model_dump(mode="python", warnings=False)
            )
        ),
        work_contract=copy_work_contract_ref(request.work_contract),
        invocation_origin=copy_invocation_origin_claim(request.invocation_origin),
    )
    copied._verified_invocation_origin = (
        None
        if request._verified_invocation_origin is None
        else copy_invocation_origin(request._verified_invocation_origin)
    )
    copied._runtime_invocation_source = request._runtime_invocation_source
    copied._runtime_session_binding = _copy_optional_session_binding(
        request._runtime_session_binding
    )
    return copied


def task_create_with_runtime_invocation(
    request: TaskCreate,
    *,
    source: TaskExecutionSource,
    verified_origin: InvocationOrigin | None = None,
    session_invocation: SessionInvocationBinding | None = None,
) -> TaskCreate:
    """Attach provenance authority minted by a trusted Cayu boundary."""

    if type(request) is not TaskCreate:
        raise TypeError("Runtime task invocation authority requires a TaskCreate request.")
    if type(source) is not TaskExecutionSource:
        raise TypeError("source must be a TaskExecutionSource.")
    if verified_origin is not None:
        verified_origin = copy_invocation_origin(verified_origin)
        if verified_origin.trust is not InvocationOriginTrust.SERVER_VERIFIED:
            raise ValueError("Runtime-verified task origins must use server_verified trust.")
        if request.invocation_origin is not None:
            raise ValueError("A verified task cannot also carry a host origin claim.")
    session_binding = _copy_optional_session_binding(session_invocation)
    if session_binding is not None and (
        request.invocation_origin is not None or verified_origin is not None
    ):
        raise ValueError("Session-derived tasks must inherit their root invocation origin.")
    copied = copy_task_create(request)
    copied._runtime_invocation_source = source
    copied._verified_invocation_origin = verified_origin
    copied._runtime_session_binding = session_binding
    return copied


def task_create_with_execution_source(
    request: TaskCreate,
    *,
    source: TaskExecutionSource,
) -> TaskCreate:
    """Classify work at a trusted direct-SDK host boundary.

    The source is private model state rather than request JSON. Server-owned
    sources are intentionally rejected; only Cayu's server/runtime adapters may
    mint those classifications.
    """

    if type(source) is not TaskExecutionSource:
        raise TypeError("source must be a TaskExecutionSource.")
    if source not in {
        TaskExecutionSource.SDK_TASK,
        TaskExecutionSource.SCHEDULED,
        TaskExecutionSource.WEBHOOK,
    }:
        raise ValueError("Direct SDK task sources must be sdk_task, scheduled, or webhook.")
    return task_create_with_runtime_invocation(request, source=source)


def task_invocation_for_create(request, *, task_id, parent_task, session_invocation=None):
    from cayu.tasks.access import creation_invocation

    invocation = _task_invocation_for_create_unscoped(
        request, task_id=task_id, parent_task=parent_task, session_invocation=session_invocation
    )
    return creation_invocation(invocation, request, parent_task)


def _task_invocation_for_create_unscoped(
    request: TaskCreate,
    *,
    task_id: str,
    parent_task: Task | TaskInvocationSnapshot | None,
    session_invocation: SessionInvocationBinding | None = None,
) -> TaskInvocation:
    """Derive exact provenance inside the atomic task-store create boundary."""

    if type(request) is not TaskCreate:
        raise TypeError("Task invocation derivation requires a TaskCreate request.")
    task_id = require_clean_nonblank(task_id, "task_id")
    source = request._runtime_invocation_source or TaskExecutionSource.SDK_TASK
    verified_origin = request._verified_invocation_origin
    request_session_binding = request._runtime_session_binding
    supplied_session_binding = _copy_optional_session_binding(session_invocation)
    if (
        request_session_binding is not None
        and supplied_session_binding is not None
        and request_session_binding != supplied_session_binding
    ):
        raise ValueError("Task creation carries contradictory session invocation bindings.")
    session_binding = supplied_session_binding or request_session_binding
    if request.invocation_origin is not None and verified_origin is not None:
        raise ValueError("A task cannot carry both host-asserted and verified origins.")
    if parent_task is not None:
        if type(parent_task) not in {Task, TaskInvocationSnapshot}:
            raise TypeError("Parent task provenance must be a task or invocation snapshot.")
        if request.parent_task_id != parent_task.id:
            raise ValueError("Parent task identity conflicts with invocation derivation.")
        if request.invocation_origin is not None or verified_origin is not None:
            raise ValueError("Derived tasks must inherit their root invocation origin.")
        if (
            session_binding is not None
            and request.session_id is not None
            and request.session_id != session_binding.id
        ):
            raise ValueError("Task session identity conflicts with its provenance binding.")
        if session_binding is not None and (
            parent_task.invocation.origin != session_binding.invocation.origin
            or parent_task.invocation.root_invocation_id
            != session_binding.invocation.root_invocation_id
        ):
            raise ValueError("Parent task and attached session invocation provenance conflict.")
        return inherited_task_invocation(
            parent_task.invocation,
            source=source,
            root_session_id=(
                None if session_binding is None else session_binding.invocation.root_session_id
            ),
        )
    if request.parent_task_id is not None:
        raise ValueError("Parent task not found for invocation provenance.")
    if session_binding is not None:
        if request.invocation_origin is not None or verified_origin is not None:
            raise ValueError("Session-derived tasks must inherit their root invocation origin.")
        if request.session_id is not None and request.session_id != session_binding.id:
            raise ValueError("Task session identity conflicts with its provenance binding.")
        return inherited_task_invocation(
            session_binding.invocation,
            source=source,
        )
    if source is TaskExecutionSource.TASK_DISPATCH:
        raise ValueError("Task dispatch provenance requires a parent task or session.")
    if verified_origin is not None:
        if source not in {TaskExecutionSource.HTTP_RUN, TaskExecutionSource.PRODUCT_OPERATION}:
            raise ValueError("Verified task origins require a server-owned task source.")
        origin = copy_invocation_origin(verified_origin)
    elif request.invocation_origin is not None:
        if source not in {
            TaskExecutionSource.SDK_TASK,
            TaskExecutionSource.SCHEDULED,
            TaskExecutionSource.WEBHOOK,
        }:
            raise ValueError("Host-asserted task origins require a trusted host source.")
        origin = InvocationOrigin(
            trust=InvocationOriginTrust.HOST_ASSERTED,
            subject=request.invocation_origin.subject,
            tenant=request.invocation_origin.tenant,
        )
    else:
        if source not in {
            TaskExecutionSource.SDK_TASK,
            TaskExecutionSource.SCHEDULED,
            TaskExecutionSource.WEBHOOK,
        }:
            raise ValueError(f"{source.value} task provenance requires a trusted origin.")
        origin = InvocationOrigin(trust=InvocationOriginTrust.UNATTRIBUTED)
    from cayu.resource_access import current_binding

    return TaskInvocation(
        resource_access=current_binding(),
        origin=origin,
        root_invocation_id=str(uuid4()),
        root_session_id=request.session_id,
        source=source,
    )


def _copy_optional_session_binding(
    value: SessionInvocationBinding | None,
) -> SessionInvocationBinding | None:
    if value is None:
        return None
    return copy_session_invocation_binding(value)


def _copy_required_session_binding(
    value: SessionInvocationBinding,
) -> SessionInvocationBinding:
    if value is None:
        raise TypeError("Running task creation requires session invocation provenance.")
    return copy_session_invocation_binding(value)


def _task_invocation_for_attachment(
    task_invocation: TaskInvocation,
    *,
    session_id: str | None,
    session_binding: SessionInvocationBinding | None,
) -> TaskInvocation:
    task_invocation = copy_task_invocation(task_invocation)
    if session_id is None:
        if session_binding is not None:
            raise ValueError("Session provenance binding requires a session_id attachment.")
        return task_invocation
    if session_binding is None:
        raise ValueError("Session provenance binding is required to attach this task.")
    if session_binding.id != session_id:
        raise ValueError("Task session identity conflicts with its provenance binding.")
    session_invocation = session_binding.invocation
    if (
        task_invocation.origin != session_invocation.origin
        or task_invocation.root_invocation_id != session_invocation.root_invocation_id
    ):
        raise ValueError("Task and session invocation provenance conflict.")
    if (
        task_invocation.root_session_id is not None
        and task_invocation.root_session_id != session_invocation.root_session_id
    ):
        raise ValueError("Task and session root identities conflict.")
    return task_invocation


def _task_session_instance_for_attachment(
    *,
    stored_session_instance_id: str | None,
    session_id: str | None,
    session_binding: SessionInvocationBinding | None,
) -> str | None:
    """Bind one task attachment to the exact durable session incarnation."""

    if session_id is None:
        if session_binding is not None or stored_session_instance_id is not None:
            raise ValueError("Session-instance authority requires a session attachment.")
        return None
    if session_binding is None:
        raise ValueError("Session-instance authority is required to attach this task.")
    if session_binding.id != session_id:
        raise ValueError("Task session identity conflicts with its instance authority.")
    if (
        stored_session_instance_id is not None
        and stored_session_instance_id != session_binding.session_instance_id
    ):
        raise ValueError("Task is already bound to another session instance.")
    return session_binding.session_instance_id


def _task_session_id_for_start(
    *,
    task_id: str,
    stored_session_id: str | None,
    requested_session_id: str | None,
) -> str | None:
    """Resolve one start transition's canonical session without allowing reassignment."""

    task_id = require_clean_nonblank(task_id, "task_id")
    if (
        stored_session_id is not None
        and requested_session_id is not None
        and stored_session_id != requested_session_id
    ):
        raise ValueError(f"Task {task_id} is already bound to a different session.")
    return stored_session_id if stored_session_id is not None else requested_session_id


def require_contract_bound_task_creation_snapshot(task: Task) -> None:
    """Enforce the initial-snapshot reserve shared by every supporting store."""

    if type(task) is not Task or task.work_contract is None:
        raise TypeError("Creation-snapshot validation requires a contract-bound Task.")
    require_bounded_work_completion_document(
        task.model_dump(mode="json", warnings=False),
        "Contract-bound task creation snapshot",
        max_bytes=WORK_CONTRACT_TASK_CREATION_MAX_BYTES,
        max_items=WORK_CONTRACT_TASK_CREATION_MAX_ITEMS,
    )


def _task_from_create(
    request: TaskCreate,
    *,
    task_id: str,
    parent_task: Task | TaskInvocationSnapshot | None,
    session_invocation: SessionInvocationBinding | None = None,
    retry_started_at: datetime | None = None,
    supports_verified_work_contracts: bool = False,
) -> Task:
    if request.work_contract is not None and not supports_verified_work_contracts:
        raise NotImplementedError(
            "This TaskStore does not support verified work-contract task bindings."
        )
    now = datetime.now(UTC)
    retry_started_at = (
        now
        if retry_started_at is None
        else normalize_utc_datetime(retry_started_at, "retry_started_at")
    )
    if request.schedule_policy is not None:
        now = retry_started_at
    invocation = task_invocation_for_create(
        request,
        task_id=task_id,
        parent_task=parent_task,
        session_invocation=session_invocation,
    )
    effective_session_binding = (
        session_invocation if session_invocation is not None else request._runtime_session_binding
    )
    retry_policy = request.retry_policy
    if retry_policy is None:
        retry_series = None
    else:
        retry_series_id = _task_retry_series_id(task_id)
        authority_sha256 = _task_retry_attempt_authority_sha256(
            task_id=task_id,
            task_type=request.type,
            title=request.title,
            description=request.description,
            parent_task_id=request.parent_task_id,
            assigned_agent_name=request.assigned_agent_name,
            available_at=request.available_at,
            created_at=now,
            task_input=request.input,
            metadata=request.metadata,
            invocation=invocation,
            series_id=retry_series_id,
            causal_budget_id=retry_series_id,
            attempt=1,
            policy=retry_policy,
            started_at=retry_started_at,
            cumulative_tokens=0,
            cumulative_estimated_cost=Decimal(0),
            predecessor_task_id=None,
        )
        retry_series = _task_retry_series_snapshot(
            series_id=retry_series_id,
            causal_budget_id=retry_series_id,
            authority_sha256=authority_sha256,
            attempt=1,
            policy=retry_policy,
            started_at=retry_started_at,
        )
    task = Task(
        id=task_id,
        type=request.type,
        title=request.title,
        description=request.description,
        status=TaskStatus.PENDING,
        session_id=request.session_id,
        session_instance_id=(
            None
            if request.session_id is None or effective_session_binding is None
            else effective_session_binding.session_instance_id
        ),
        parent_task_id=request.parent_task_id,
        assigned_agent_name=request.assigned_agent_name,
        available_at=request.available_at,
        input=copy_durable_json_object(request.input, "input"),
        metadata=copy_durable_metadata(request.metadata),
        created_at=now,
        updated_at=now,
        invocation=invocation,
        retry_series=retry_series,
        work_contract=copy_work_contract_ref(request.work_contract),
    )
    if request.schedule_policy is not None:
        task = task.model_copy(
            update={
                "schedule": TaskScheduleState(
                    policy=request.schedule_policy,
                    creation_sha256=schedule_creation_digest(request),
                )
            }
        )
    if task.work_contract is not None:
        require_contract_bound_task_creation_snapshot(task)
    return task


def _running_task_from_create(
    request: TaskCreate,
    *,
    task_id: str,
    parent_task: Task | TaskInvocationSnapshot | None,
    session_invocation: SessionInvocationBinding,
    retry_started_at: datetime | None = None,
    supports_verified_work_contracts: bool = False,
) -> Task:
    task = _task_from_create(
        request,
        task_id=task_id,
        parent_task=parent_task,
        session_invocation=session_invocation,
        retry_started_at=retry_started_at,
        supports_verified_work_contracts=supports_verified_work_contracts,
    )
    if task.session_id is None:
        raise ValueError("TaskCreate.session_id is required to create a running task.")
    running = task.model_copy(
        update={
            "status": TaskStatus.RUNNING,
            "started_at": task.created_at,
        }
    )
    return copy_task(running) if running.work_contract is not None else running


def preflight_contract_bound_task_creation(
    request: TaskCreate,
    *,
    parent_task: Task | TaskInvocationSnapshot | None,
) -> None:
    """Validate the authoritative pending snapshot before an extension mutates."""

    if type(request) is not TaskCreate or request.work_contract is None:
        raise TypeError("Creation preflight requires a contract-bound TaskCreate request.")
    if request.task_id is None:
        raise ValueError("Contract-bound task creation requires a caller-stable task_id.")
    preview = _task_from_create(
        request,
        task_id=request.task_id,
        parent_task=parent_task,
        supports_verified_work_contracts=True,
    )
    del preview

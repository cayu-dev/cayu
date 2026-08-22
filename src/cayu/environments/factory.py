from __future__ import annotations

import asyncio
import re
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from math import isfinite
from typing import TYPE_CHECKING, Any

from cayu._exception_groups import iter_exception_tree
from cayu._exception_state import exception_state, set_exception_state
from cayu._task_wait import await_shielded_task_outcome
from cayu._validation import (
    canonical_durable_json_bytes,
    copy_durable_json_object,
    copy_label_map,
    require_clean_nonblank,
    require_durable_clean_nonblank,
)
from cayu.artifacts import ArtifactStore
from cayu.core.execution_identity import ExecutionProfileBehaviorIdentity
from cayu.environments.admission import (
    ExecutionAdmissionCandidate,
    ExecutionEnvironmentAuthority,
    ExecutionRequirements,
)
from cayu.environments.base import Environment, copy_environment
from cayu.runners.base import RunnerWorkloadAuthority

if TYPE_CHECKING:
    from cayu.egress.authority import EgressAuthorityIdentity

DEFAULT_ENVIRONMENT_FACTORY_RELEASE_TIMEOUT_SECONDS = 15.0
ENVIRONMENT_ALLOCATION_INTENT_SCHEMA_VERSION = 1
ENVIRONMENT_ALLOCATION_PROVIDER_METADATA_MAX_BYTES = 16 * 1024
ENVIRONMENT_ALLOCATION_ID_MAX_BYTES = 72
ENVIRONMENT_ALLOCATION_SCOPE_FIELD_MAX_BYTES = 128
ENVIRONMENT_ALLOCATION_OWNER_FIELD_MAX_BYTES = 2048
_ENVIRONMENT_ALLOCATION_ID_PATTERN = re.compile(r"ealloc_[0-9a-f]{32}\Z")
_ENVIRONMENT_FACTORY_CLEANUP_SETTLEMENT_TASK_ATTRIBUTE = (
    "_cayu_environment_factory_cleanup_settlement_task"
)
_ENVIRONMENT_FACTORY_CLEANUP_SETTLEMENT_TASK_TOKEN = object()


EnvironmentFactoryCleanupRetry = Callable[[], asyncio.Task[None]]


@dataclass(slots=True, eq=False)
class _EnvironmentFactoryCleanupOwner:
    task: asyncio.Task[None]
    retry: EnvironmentFactoryCleanupRetry | None
    token: object

    def retry_failed_task(self) -> asyncio.Task[None]:
        current = self.task
        if not current.done():
            return current
        try:
            current.result()
        except BaseException:
            pass
        else:
            return current
        if self.retry is None:
            return current
        replacement = self.retry()
        if not isinstance(replacement, asyncio.Task):
            raise TypeError("Factory cleanup retry must return an asyncio Task.")
        if replacement is current:
            raise RuntimeError("Factory cleanup retry must return a new task.")
        self.task = replacement
        _ENVIRONMENT_FACTORY_CLEANUP_OWNERS_BY_TASK.pop(current, None)
        _register_environment_factory_cleanup_owner_task(replacement, self)
        return replacement


_ENVIRONMENT_FACTORY_CLEANUP_OWNERS_BY_TASK: dict[
    asyncio.Task[None],
    _EnvironmentFactoryCleanupOwner,
] = {}


@dataclass(frozen=True, slots=True)
class _EnvironmentFactoryCleanupSettlementTaskHandoff:
    owner: _EnvironmentFactoryCleanupOwner
    token: object


def register_environment_factory_cleanup_retry(
    task: asyncio.Task[None],
    retry: EnvironmentFactoryCleanupRetry,
) -> None:
    """Authenticate a callable that retries the exact cleanup owned by ``task``."""

    if not isinstance(task, asyncio.Task):
        raise TypeError("Factory cleanup settlement requires an asyncio Task.")
    if not callable(retry):
        raise TypeError("Factory cleanup retry must be callable.")
    existing = _environment_factory_cleanup_owner_for_task(task)
    if existing is not None:
        if existing.retry is not None and existing.retry is not retry:
            raise RuntimeError("Factory cleanup task already has a different retry owner.")
        existing.retry = retry
        return
    owner = _EnvironmentFactoryCleanupOwner(
        task=task,
        retry=retry,
        token=_ENVIRONMENT_FACTORY_CLEANUP_SETTLEMENT_TASK_TOKEN,
    )
    _register_environment_factory_cleanup_owner_task(task, owner)


def attach_environment_factory_cleanup_settlement_task(
    error: BaseException,
    task: asyncio.Task[None],
) -> None:
    """Carry an in-flight factory cleanup owner across a failed create boundary."""

    if not isinstance(error, BaseException):
        raise TypeError("Factory cleanup settlement requires an exception.")
    if not isinstance(task, asyncio.Task):
        raise TypeError("Factory cleanup settlement requires an asyncio Task.")
    owner = _environment_factory_cleanup_owner_for_task(task)
    if owner is None:
        owner = _EnvironmentFactoryCleanupOwner(
            task=task,
            retry=None,
            token=_ENVIRONMENT_FACTORY_CLEANUP_SETTLEMENT_TASK_TOKEN,
        )
        _register_environment_factory_cleanup_owner_task(task, owner)
    handoff = _EnvironmentFactoryCleanupSettlementTaskHandoff(
        owner=owner,
        token=_ENVIRONMENT_FACTORY_CLEANUP_SETTLEMENT_TASK_TOKEN,
    )
    if not set_exception_state(
        error,
        _ENVIRONMENT_FACTORY_CLEANUP_SETTLEMENT_TASK_ATTRIBUTE,
        handoff,
    ):
        raise RuntimeError("Could not attach factory cleanup settlement ownership.")


def environment_factory_cleanup_settlement_task(
    error: BaseException,
) -> asyncio.Task[None] | None:
    """Return the exact deferred cleanup task carried by a factory failure."""

    handoff = exception_state(
        error,
        _ENVIRONMENT_FACTORY_CLEANUP_SETTLEMENT_TASK_ATTRIBUTE,
    )
    if (
        type(handoff) is not _EnvironmentFactoryCleanupSettlementTaskHandoff
        or handoff.token is not _ENVIRONMENT_FACTORY_CLEANUP_SETTLEMENT_TASK_TOKEN
        or not _is_authenticated_environment_factory_cleanup_owner(handoff.owner)
    ):
        return None
    return handoff.owner.task


def retry_environment_factory_cleanup_settlement_task(
    task: asyncio.Task[None],
) -> asyncio.Task[None]:
    """Retry one authenticated failed cleanup task, if it carries a retry owner."""

    if not isinstance(task, asyncio.Task):
        raise TypeError("Factory cleanup settlement retry requires an asyncio Task.")
    owner = _environment_factory_cleanup_owner_for_task(task)
    if owner is None or owner.task is not task:
        return task
    return owner.retry_failed_task()


def environment_factory_cleanup_retry_available(
    task: asyncio.Task[None],
) -> bool:
    """Return whether ``task`` owns an authenticated cleanup retry."""

    if not isinstance(task, asyncio.Task):
        raise TypeError("Factory cleanup settlement lookup requires an asyncio Task.")
    owner = _environment_factory_cleanup_owner_for_task(task)
    return owner is not None and owner.task is task and owner.retry is not None


def environment_factory_cleanup_settlement_tasks(
    error: BaseException,
) -> tuple[asyncio.Task[None], ...]:
    """Collect every authenticated cleanup owner in one exception tree."""

    if not isinstance(error, BaseException):
        raise TypeError("Factory cleanup settlement lookup requires an exception.")
    return tuple(
        dict.fromkeys(
            task
            for candidate in iter_exception_tree(error)
            if (task := environment_factory_cleanup_settlement_task(candidate)) is not None
        )
    )


def combine_environment_factory_cleanup_settlement_tasks(
    tasks: Sequence[asyncio.Task[None]],
    *,
    task_name: str,
    failure_message: str,
) -> asyncio.Task[None] | None:
    """Return one owner that settles every distinct authenticated cleanup task."""

    unique_tasks = tuple(dict.fromkeys(tasks))
    if not unique_tasks:
        return None
    if len(unique_tasks) == 1:
        return unique_tasks[0]

    owners = tuple(_environment_factory_cleanup_owner_for_task(task) for task in unique_tasks)

    def combined_task() -> asyncio.Task[None]:
        return asyncio.create_task(
            settle_all(
                tuple(
                    owner.task if owner is not None else task
                    for task, owner in zip(unique_tasks, owners, strict=True)
                )
            ),
            name=task_name,
        )

    async def settle_all(tasks_to_settle: Sequence[asyncio.Task[None]]) -> None:
        failures: list[BaseException] = []
        for task in tasks_to_settle:
            outcome = await await_shielded_task_outcome(task)
            if outcome.error is not None:
                failures.append(outcome.error)
            if outcome.cancellation is not None and outcome.cancellation is not outcome.error:
                failures.append(outcome.cancellation)
        if len(failures) == 1:
            raise failures[0]
        if failures:
            raise BaseExceptionGroup(failure_message, failures)

    task = combined_task()

    if any(owner is not None and owner.retry is not None for owner in owners):

        def retry_combined() -> asyncio.Task[None]:
            for owner in owners:
                if owner is not None:
                    owner.retry_failed_task()
            return combined_task()

        register_environment_factory_cleanup_retry(task, retry_combined)
    return task


def _environment_factory_cleanup_owner_for_task(
    task: asyncio.Task[None],
) -> _EnvironmentFactoryCleanupOwner | None:
    try:
        owner = _ENVIRONMENT_FACTORY_CLEANUP_OWNERS_BY_TASK.get(task)
    except BaseException:
        return None
    return owner if _is_authenticated_environment_factory_cleanup_owner(owner) else None


def _register_environment_factory_cleanup_owner_task(
    task: asyncio.Task[None],
    owner: _EnvironmentFactoryCleanupOwner,
) -> None:
    _ENVIRONMENT_FACTORY_CLEANUP_OWNERS_BY_TASK[task] = owner

    def retire_completed(completed: asyncio.Task[None]) -> None:
        if owner.task is not completed:
            _ENVIRONMENT_FACTORY_CLEANUP_OWNERS_BY_TASK.pop(completed, None)
            return
        if owner.retry is None:
            _ENVIRONMENT_FACTORY_CLEANUP_OWNERS_BY_TASK.pop(completed, None)
            return
        if not completed.cancelled() and completed.exception() is None:
            _ENVIRONMENT_FACTORY_CLEANUP_OWNERS_BY_TASK.pop(completed, None)

    task.add_done_callback(retire_completed)


def _is_authenticated_environment_factory_cleanup_owner(
    owner: object,
) -> bool:
    return (
        type(owner) is _EnvironmentFactoryCleanupOwner
        and owner.token is _ENVIRONMENT_FACTORY_CLEANUP_SETTLEMENT_TASK_TOKEN
        and isinstance(owner.task, asyncio.Task)
        and (owner.retry is None or callable(owner.retry))
    )


class EnvironmentFactoryOperation(StrEnum):
    """Whether a factory must allocate a new environment or reconnect one."""

    CREATE = "create"
    RECONNECT = "reconnect"


class EnvironmentFactoryReleaseAction(StrEnum):
    """How an unadopted factory result must release its live resources."""

    DISCARD = "discard"
    PRESERVE = "preserve"


EnvironmentFactoryRelease = Callable[[EnvironmentFactoryReleaseAction], Awaitable[None]]


class EnvironmentAllocationState(StrEnum):
    """Durable progress of one remote allocation intent."""

    UNPREPARED = "unprepared"
    PREPARED = "prepared"
    DISPATCHED = "dispatched"
    ACKNOWLEDGED = "acknowledged"
    REAPING = "reaping"
    REAPED = "reaped"


class EnvironmentAllocationUnsupportedError(RuntimeError):
    """Raised before provider mutation when crash-safe allocation is unavailable."""


@dataclass(frozen=True)
class EnvironmentAllocationScope:
    """Stable provider and adapter generation that own one allocation."""

    provider: str
    adapter_generation: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "provider",
            _bounded_allocation_text(
                self.provider,
                "provider",
                max_bytes=ENVIRONMENT_ALLOCATION_SCOPE_FIELD_MAX_BYTES,
            ),
        )
        object.__setattr__(
            self,
            "adapter_generation",
            _bounded_allocation_text(
                self.adapter_generation,
                "adapter_generation",
                max_bytes=ENVIRONMENT_ALLOCATION_SCOPE_FIELD_MAX_BYTES,
            ),
        )


@dataclass(frozen=True)
class EnvironmentAllocationIntent:
    """Portable identity prepared before one remote provider mutation."""

    allocation_id: str
    provider: str
    adapter_generation: str
    session_id: str
    environment_name: str
    requested_operation: EnvironmentFactoryOperation
    provider_metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: int = ENVIRONMENT_ALLOCATION_INTENT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if (
            type(self.schema_version) is not int
            or self.schema_version != ENVIRONMENT_ALLOCATION_INTENT_SCHEMA_VERSION
        ):
            raise ValueError("Environment allocation intent schema version is unsupported.")
        allocation_id = _bounded_allocation_text(
            self.allocation_id,
            "allocation_id",
            max_bytes=ENVIRONMENT_ALLOCATION_ID_MAX_BYTES,
        )
        if _ENVIRONMENT_ALLOCATION_ID_PATTERN.fullmatch(allocation_id) is None:
            raise ValueError("allocation_id must be a canonical Cayu allocation identifier.")
        if not isinstance(self.requested_operation, EnvironmentFactoryOperation):
            raise TypeError("requested_operation must be an EnvironmentFactoryOperation.")
        if self.requested_operation is not EnvironmentFactoryOperation.CREATE:
            raise ValueError("Remote allocation intents may only authorize create operations.")
        metadata = copy_durable_json_object(self.provider_metadata, "provider_metadata")
        if (
            len(
                canonical_durable_json_bytes(
                    metadata,
                    "provider_metadata",
                )
            )
            > ENVIRONMENT_ALLOCATION_PROVIDER_METADATA_MAX_BYTES
        ):
            raise ValueError(
                "Environment allocation provider metadata exceeds the durable byte limit."
            )
        object.__setattr__(self, "allocation_id", allocation_id)
        object.__setattr__(
            self,
            "provider",
            _bounded_allocation_text(
                self.provider,
                "provider",
                max_bytes=ENVIRONMENT_ALLOCATION_SCOPE_FIELD_MAX_BYTES,
            ),
        )
        object.__setattr__(
            self,
            "adapter_generation",
            _bounded_allocation_text(
                self.adapter_generation,
                "adapter_generation",
                max_bytes=ENVIRONMENT_ALLOCATION_SCOPE_FIELD_MAX_BYTES,
            ),
        )
        object.__setattr__(
            self,
            "session_id",
            _bounded_allocation_text(
                self.session_id,
                "session_id",
                max_bytes=ENVIRONMENT_ALLOCATION_OWNER_FIELD_MAX_BYTES,
            ),
        )
        object.__setattr__(
            self,
            "environment_name",
            _bounded_allocation_text(
                self.environment_name,
                "environment_name",
                max_bytes=ENVIRONMENT_ALLOCATION_OWNER_FIELD_MAX_BYTES,
            ),
        )
        object.__setattr__(self, "provider_metadata", metadata)

    @property
    def scope(self) -> EnvironmentAllocationScope:
        return EnvironmentAllocationScope(
            provider=self.provider,
            adapter_generation=self.adapter_generation,
        )

    def with_provider_metadata(
        self,
        provider_metadata: Mapping[str, Any],
    ) -> EnvironmentAllocationIntent:
        if not isinstance(provider_metadata, Mapping):
            raise TypeError("provider_metadata must be a mapping.")
        return EnvironmentAllocationIntent(
            allocation_id=self.allocation_id,
            provider=self.provider,
            adapter_generation=self.adapter_generation,
            session_id=self.session_id,
            environment_name=self.environment_name,
            requested_operation=self.requested_operation,
            provider_metadata=dict(provider_metadata),
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "allocation_id": self.allocation_id,
            "provider": self.provider,
            "adapter_generation": self.adapter_generation,
            "session_id": self.session_id,
            "environment_name": self.environment_name,
            "requested_operation": self.requested_operation.value,
            "provider_metadata": copy_durable_json_object(
                self.provider_metadata,
                "provider_metadata",
            ),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> EnvironmentAllocationIntent:
        if not isinstance(payload, Mapping):
            raise TypeError("Environment allocation intent payload must be a mapping.")
        copied = copy_durable_json_object(dict(payload), "allocation_intent")
        expected = {
            "schema_version",
            "allocation_id",
            "provider",
            "adapter_generation",
            "session_id",
            "environment_name",
            "requested_operation",
            "provider_metadata",
        }
        if set(copied) != expected:
            raise ValueError("Environment allocation intent has an invalid schema.")
        try:
            operation = EnvironmentFactoryOperation(copied["requested_operation"])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Environment allocation intent requested_operation is invalid."
            ) from exc
        return cls(
            schema_version=copied["schema_version"],
            allocation_id=copied["allocation_id"],
            provider=copied["provider"],
            adapter_generation=copied["adapter_generation"],
            session_id=copied["session_id"],
            environment_name=copied["environment_name"],
            requested_operation=operation,
            provider_metadata=copied["provider_metadata"],
        )


class EnvironmentAllocationContext(ABC):
    """Runtime-owned durable coordinator passed to recoverable factories."""

    @property
    @abstractmethod
    def intent(self) -> EnvironmentAllocationIntent:
        """Return the current detached allocation intent."""

    @property
    @abstractmethod
    def state(self) -> EnvironmentAllocationState:
        """Return the last reconciled durable state."""

    @property
    @abstractmethod
    def acknowledged_reconnect_metadata(self) -> dict[str, Any] | None:
        """Return detached provider acknowledgement, when known."""

    @abstractmethod
    async def prepare(
        self,
        provider_metadata: Mapping[str, Any],
    ) -> EnvironmentAllocationIntent:
        """Persist the final intent before provider dispatch."""

    @abstractmethod
    async def mark_dispatched(self) -> None:
        """Fence recovery before the irreversible provider call."""

    @abstractmethod
    async def acknowledge(self, reconnect_metadata: Mapping[str, Any]) -> None:
        """Persist the exact provider result before returning it."""

    @abstractmethod
    async def mark_reaping(self) -> bool:
        """Fence cleanup against publication; false means publication already won."""

    @abstractmethod
    async def mark_reaped(self) -> None:
        """Record positively completed cleanup after the durable cleanup fence."""


@dataclass(frozen=True)
class EnvironmentFactoryRequest:
    """Durable session context used to create or attach an environment."""

    session_id: str
    agent_name: str
    environment_name: str
    parent_session_id: str | None = None
    causal_budget_id: str | None = None
    labels: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    reconnect_metadata: dict[str, Any] = field(default_factory=dict)
    operation: EnvironmentFactoryOperation = EnvironmentFactoryOperation.CREATE
    execution_requirements: ExecutionRequirements = field(
        default_factory=ExecutionRequirements.trusted
    )
    execution_profile_fingerprint: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.operation, EnvironmentFactoryOperation):
            raise TypeError("operation must be an EnvironmentFactoryOperation.")
        if not isinstance(self.execution_requirements, ExecutionRequirements):
            raise TypeError("execution_requirements must be ExecutionRequirements.")
        object.__setattr__(
            self, "session_id", require_clean_nonblank(self.session_id, "session_id")
        )
        object.__setattr__(
            self, "agent_name", require_clean_nonblank(self.agent_name, "agent_name")
        )
        object.__setattr__(
            self,
            "environment_name",
            require_clean_nonblank(self.environment_name, "environment_name"),
        )
        if self.execution_profile_fingerprint is not None and (
            type(self.execution_profile_fingerprint) is not str
            or len(self.execution_profile_fingerprint) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.execution_profile_fingerprint
            )
        ):
            raise ValueError("execution_profile_fingerprint must be a lowercase SHA-256 digest.")
        if self.parent_session_id is not None:
            object.__setattr__(
                self,
                "parent_session_id",
                require_clean_nonblank(self.parent_session_id, "parent_session_id"),
            )
        if self.causal_budget_id is not None:
            object.__setattr__(
                self,
                "causal_budget_id",
                require_clean_nonblank(self.causal_budget_id, "causal_budget_id"),
            )
        object.__setattr__(self, "labels", copy_label_map(self.labels, "labels"))
        object.__setattr__(
            self,
            "metadata",
            copy_durable_json_object(self.metadata, "metadata"),
        )
        object.__setattr__(
            self,
            "reconnect_metadata",
            copy_durable_json_object(self.reconnect_metadata, "reconnect_metadata"),
        )
        object.__setattr__(
            self,
            "execution_requirements",
            ExecutionRequirements.model_validate(
                self.execution_requirements.model_dump(mode="python")
            ),
        )


@dataclass(frozen=True)
class EnvironmentFactoryResult:
    """Concrete environment and pre-adoption release contract for a session.

    ``release`` owns factory-created resources until workspace binding succeeds.
    After successful binding, the binding owns the adopted environment lifecycle.
    """

    environment: Environment
    metadata: dict[str, Any] = field(default_factory=dict)
    reconnect_metadata: dict[str, Any] = field(default_factory=dict)
    release: EnvironmentFactoryRelease | None = None
    release_timeout_s: float = DEFAULT_ENVIRONMENT_FACTORY_RELEASE_TIMEOUT_SECONDS

    def __post_init__(self) -> None:
        if not isinstance(self.environment, Environment):
            raise TypeError("EnvironmentFactoryResult.environment must be an Environment.")
        if self.release is not None and not callable(self.release):
            raise TypeError("EnvironmentFactoryResult.release must be callable or None.")
        if type(self.release_timeout_s) not in {int, float}:
            raise TypeError("EnvironmentFactoryResult.release_timeout_s must be numeric.")
        if not isfinite(self.release_timeout_s) or self.release_timeout_s <= 0:
            raise ValueError(
                "EnvironmentFactoryResult.release_timeout_s must be finite and greater than zero."
            )
        object.__setattr__(self, "release_timeout_s", float(self.release_timeout_s))
        object.__setattr__(self, "environment", copy_environment(self.environment))
        object.__setattr__(
            self,
            "metadata",
            copy_durable_json_object(self.metadata, "metadata"),
        )
        object.__setattr__(
            self,
            "reconnect_metadata",
            copy_durable_json_object(self.reconnect_metadata, "reconnect_metadata"),
        )


class EnvironmentFactory(ABC):
    """Creates or attaches a concrete environment for a session."""

    @property
    def execution_profile_identity(self) -> ExecutionProfileBehaviorIdentity | None:
        """Return a stable application declaration, or ``None`` when non-portable."""

        return None

    @property
    def egress_authority_identity(self) -> EgressAuthorityIdentity | None:
        """Return the immutable egress authority proposed for future invocations."""

        return None

    def execution_admission_candidate(
        self,
        request: EnvironmentFactoryRequest,
    ) -> ExecutionAdmissionCandidate | None:
        """Return explicit pre-create evidence without allocating resources.

        The default deliberately makes no capability claim. Implementations
        must keep this hook side-effect free because Cayu calls it before
        ``create``.
        """

        del request
        return None

    def construction_admission_candidate(self) -> ExecutionAdmissionCandidate | None:
        """Return side-effect-free configured capability evidence at app setup."""

        return None

    def execution_environment_authority(self) -> ExecutionEnvironmentAuthority | None:
        """Return the exact security boundary every materialized runner preserves."""

        return None

    def workload_authority(self, name: str) -> RunnerWorkloadAuthority | None:
        """Return configured authority for a workload every result must preserve."""

        del name
        return None

    @property
    def configured_artifact_store(self) -> ArtifactStore | None:
        """Return the stable artifact store exposed by every factory result."""

        return None

    def allocation_scope(
        self,
        request: EnvironmentFactoryRequest,
    ) -> EnvironmentAllocationScope | None:
        """Declare one crash-recoverable remote allocation boundary.

        Returning ``None`` means the factory does not allocate a process-external
        resource through this seam. Implementations that return a scope must also
        override :meth:`create_recoverable`; Cayu will never fall back to
        ``create`` after accepting a recoverable scope.
        """

        del request
        return None

    async def create_recoverable(
        self,
        request: EnvironmentFactoryRequest,
        allocation: EnvironmentAllocationContext,
    ) -> EnvironmentFactoryResult:
        """Create or recover exactly one intent-owned remote allocation."""

        del request, allocation
        raise EnvironmentAllocationUnsupportedError(
            "Environment factory declared remote allocation without implementing "
            "crash-safe creation and recovery."
        )

    @abstractmethod
    async def create(self, request: EnvironmentFactoryRequest) -> EnvironmentFactoryResult:
        """Return a concrete environment for the requested session."""


def copy_environment_factory_request(
    request: EnvironmentFactoryRequest,
) -> EnvironmentFactoryRequest:
    if not isinstance(request, EnvironmentFactoryRequest):
        raise TypeError("Environment factory request copies require an EnvironmentFactoryRequest.")
    return EnvironmentFactoryRequest(
        session_id=request.session_id,
        agent_name=request.agent_name,
        environment_name=request.environment_name,
        execution_profile_fingerprint=request.execution_profile_fingerprint,
        operation=request.operation,
        parent_session_id=request.parent_session_id,
        causal_budget_id=request.causal_budget_id,
        labels=request.labels,
        metadata=request.metadata,
        reconnect_metadata=request.reconnect_metadata,
        execution_requirements=request.execution_requirements,
    )


def copy_environment_factory_result(result: EnvironmentFactoryResult) -> EnvironmentFactoryResult:
    if not isinstance(result, EnvironmentFactoryResult):
        raise TypeError("Environment factory result copies require an EnvironmentFactoryResult.")
    return EnvironmentFactoryResult(
        environment=result.environment,
        metadata=result.metadata,
        reconnect_metadata=result.reconnect_metadata,
        release=result.release,
        release_timeout_s=result.release_timeout_s,
    )


def _bounded_allocation_text(
    value: str,
    field_name: str,
    *,
    max_bytes: int,
) -> str:
    copied = require_durable_clean_nonblank(value, field_name)
    if len(copied.encode("utf-8")) > max_bytes:
        raise ValueError(f"{field_name} exceeds the durable byte limit.")
    return copied

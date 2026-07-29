from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from math import isfinite
from typing import Any

from cayu._exception_groups import iter_exception_tree
from cayu._exception_state import exception_state, set_exception_state
from cayu._task_wait import await_shielded_task_outcome
from cayu._validation import copy_json_value, copy_label_map, require_clean_nonblank
from cayu.environments.admission import (
    ExecutionAdmissionCandidate,
    ExecutionRequirements,
)
from cayu.environments.base import Environment, copy_environment

DEFAULT_ENVIRONMENT_FACTORY_RELEASE_TIMEOUT_SECONDS = 15.0
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
        object.__setattr__(self, "metadata", copy_json_value(self.metadata, "metadata"))
        object.__setattr__(
            self,
            "reconnect_metadata",
            copy_json_value(self.reconnect_metadata, "reconnect_metadata"),
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
        object.__setattr__(self, "metadata", copy_json_value(self.metadata, "metadata"))
        object.__setattr__(
            self,
            "reconnect_metadata",
            copy_json_value(self.reconnect_metadata, "reconnect_metadata"),
        )


class EnvironmentFactory(ABC):
    """Creates or attaches a concrete environment for a session."""

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

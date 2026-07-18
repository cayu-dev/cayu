from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import StrEnum
from math import isfinite
from typing import Any

from cayu._validation import copy_json_value, copy_label_map, require_clean_nonblank
from cayu.environments.admission import (
    ExecutionAdmissionCandidate,
    ExecutionRequirements,
)
from cayu.environments.base import Environment, copy_environment

DEFAULT_ENVIRONMENT_FACTORY_RELEASE_TIMEOUT_SECONDS = 15.0


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

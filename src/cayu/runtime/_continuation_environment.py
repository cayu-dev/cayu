"""Environment preparation shared by paused continuations and manual recovery."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from dataclasses import dataclass, field

from cayu.environments.factory import EnvironmentFactoryOperation
from cayu.events import Event
from cayu.execution_profiles import ExecutionProfileIdentity
from cayu.runtime import _runtime_records as runtime_records
from cayu.runtime._environment_lifecycle import (
    EnvironmentBindingResult,
    EnvironmentFactoryResolutionResult,
    EnvironmentLifecycle,
)
from cayu.runtime._invocation_lifecycle import InvocationContext
from cayu.sessions.records import Session


@dataclass(slots=True)
class ContinuationEnvironment:
    """Retain the latest environment authority before exposing lifecycle events.

    Reconnect and bind remain separate so the caller can publish its resume event
    and settle gate-specific evidence between them. The caller handles ``error``
    after draining a phase and owns finalization on failure or stream closure.
    """

    lifecycle: EnvironmentLifecycle
    session: Session
    agent: runtime_records.RegisteredAgentState
    profile: ExecutionProfileIdentity
    registered_environment: runtime_records.RegisteredEnvironment | None
    invocation_context: InvocationContext | None
    name: str | None = field(init=False)
    error: Exception | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.name = (
            None if self.registered_environment is None else self.registered_environment.spec.name
        )

    def _adopt(self, result: EnvironmentFactoryResolutionResult | EnvironmentBindingResult) -> None:
        # Retain the replacement even if rebinding fails: cleanup must see the
        # environment already returned by the lifecycle operation.
        self.registered_environment = result.registered_environment
        if self.registered_environment is not None and self.invocation_context is not None:
            self.invocation_context = self.invocation_context.with_registered_environment(
                self.registered_environment, validated_profile=self.profile
            )
        self.error = result.error

    async def reconnect(self) -> AsyncGenerator[Event, None]:
        started = await self.lifecycle.emit_factory_started(
            session=self.session,
            registered_agent=self.agent,
            registered_environment=self.registered_environment,
            execution_profile=self.profile,
            invocation_context=self.invocation_context,
        )
        if started is not None:
            yield started
        result = await self.lifecycle.resolve_factory(
            session=self.session,
            registered_agent=self.agent,
            registered_environment=self.registered_environment,
            started_event=started,
            operation=EnvironmentFactoryOperation.RECONNECT,
            execution_profile=self.profile,
            invocation_context=self.invocation_context,
        )
        self._adopt(result)
        self.name = (
            None if self.registered_environment is None else self.registered_environment.spec.name
        )
        for event in result.events:
            yield event

    async def bind(self) -> AsyncGenerator[Event, None]:
        started = await self.lifecycle.emit_binding_started(
            session=self.session,
            registered_agent=self.agent,
            registered_environment=self.registered_environment,
            execution_profile=self.profile,
            invocation_context=self.invocation_context,
        )
        if started is not None:
            yield started
        result = await self.lifecycle.bind(
            session=self.session,
            registered_agent=self.agent,
            registered_environment=self.registered_environment,
            started_event=started,
            execution_profile=self.profile,
            invocation_context=self.invocation_context,
        )
        self._adopt(result)
        for event in result.events:
            yield event

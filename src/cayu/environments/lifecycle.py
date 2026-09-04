"""Bounded, content-free environment lifecycle progress contracts."""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable, Mapping
from contextvars import ContextVar, Token
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from types import MappingProxyType
from typing import Literal, Protocol
from uuid import uuid4

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictFloat,
    StrictInt,
    field_serializer,
    field_validator,
    model_validator,
)

from cayu._validation import MAX_PORTABLE_JSON_INTEGER, require_durable_clean_nonblank
from cayu.core.events import Event, EventType

ENVIRONMENT_LIFECYCLE_PROGRESS_SCHEMA_VERSION = 1
DEFAULT_ENVIRONMENT_LIFECYCLE_TIMEOUT_SECONDS = 3600.0
DEFAULT_ENVIRONMENT_PHASE_TIMEOUT_SECONDS = 900.0
DEFAULT_ENVIRONMENT_PROGRESS_MIN_INTERVAL_SECONDS = 0.25
DEFAULT_MAX_ENVIRONMENT_PROGRESS_EVENTS = 128
MAX_ENVIRONMENT_PROGRESS_EVENTS = 1024
MAX_ENVIRONMENT_PROGRESS_COUNTER = MAX_PORTABLE_JSON_INTEGER


class EnvironmentLifecycleOperation(StrEnum):
    """One runtime-owned unit of environment lifecycle work."""

    FACTORY = "factory"
    BINDING = "binding"
    FINALIZATION = "finalization"
    RELEASE = "release"
    RETAINED_CLEANUP = "retained_cleanup"


class EnvironmentLifecyclePhase(StrEnum):
    """Content-free phases shared by factories, bindings, and cleanup."""

    OWNERSHIP_ADMISSION = "ownership_admission"
    FACTORY_PROVISIONING = "factory_provisioning"
    FACTORY_RECONNECT = "factory_reconnect"
    SOURCE_OBSERVATION = "source_observation"
    STAGING_ADMISSION = "staging_admission"
    ARCHIVE_PREPARATION = "archive_preparation"
    TRANSFER = "transfer"
    TARGET_MATERIALIZATION = "target_materialization"
    EXECUTION_READY_PUBLICATION = "execution_ready_publication"
    FINAL_SOURCE_OBSERVATION = "final_source_observation"
    FINAL_TARGET_OBSERVATION = "final_target_observation"
    COPY_BACK_CONFLICT_PREFLIGHT = "copy_back_conflict_preflight"
    COPY_BACK_PUBLICATION = "copy_back_publication"
    ENVIRONMENT_RELEASE = "environment_release"
    RETAINED_CLEANUP = "retained_cleanup"


class EnvironmentLifecycleProgressStatus(StrEnum):
    """Stable disposition for one bounded lifecycle projection."""

    STARTED = "started"
    ADVANCED = "advanced"
    COMPLETED = "completed"
    FAILED = "failed"
    DEADLINE_EXCEEDED = "deadline_exceeded"
    RETAINED = "retained"


class EnvironmentLifecycleDeadlineExceeded(TimeoutError):
    """A configured lifecycle or phase deadline elapsed at a safe progress boundary."""

    def __init__(
        self,
        *,
        operation_id: str,
        phase: EnvironmentLifecyclePhase,
        scope: Literal["lifecycle", "phase"],
    ) -> None:
        self.operation_id = operation_id
        self.phase = phase
        self.scope = scope
        super().__init__(
            f"Environment lifecycle {phase.value} exceeded its configured {scope} deadline."
        )


_TERMINAL_PROGRESS_STATUSES = frozenset(
    {
        EnvironmentLifecycleProgressStatus.COMPLETED,
        EnvironmentLifecycleProgressStatus.FAILED,
        EnvironmentLifecycleProgressStatus.DEADLINE_EXCEEDED,
        EnvironmentLifecycleProgressStatus.RETAINED,
    }
)


def _default_phase_timeouts() -> dict[EnvironmentLifecyclePhase, float]:
    return {phase: DEFAULT_ENVIRONMENT_PHASE_TIMEOUT_SECONDS for phase in EnvironmentLifecyclePhase}


class EnvironmentLifecyclePolicy(BaseModel):
    """Finite lifecycle, phase, and progress-volume ceilings for one environment."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    lifecycle_timeout_seconds: StrictFloat = Field(
        default=DEFAULT_ENVIRONMENT_LIFECYCLE_TIMEOUT_SECONDS,
        gt=0,
        le=86_400,
    )
    phase_timeout_seconds: Mapping[EnvironmentLifecyclePhase, StrictFloat] = Field(
        default_factory=_default_phase_timeouts
    )
    progress_min_interval_seconds: StrictFloat = Field(
        default=DEFAULT_ENVIRONMENT_PROGRESS_MIN_INTERVAL_SECONDS,
        ge=0,
        le=60,
    )
    max_progress_events: StrictInt = Field(
        default=DEFAULT_MAX_ENVIRONMENT_PROGRESS_EVENTS,
        ge=32,
        le=MAX_ENVIRONMENT_PROGRESS_EVENTS,
    )

    @model_validator(mode="before")
    @classmethod
    def complete_phase_timeouts(cls, value: object) -> object:
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            return value
        copied = dict(value)
        supplied = copied.get("phase_timeout_seconds")
        if supplied is None:
            return copied
        if not isinstance(supplied, Mapping):
            return copied
        merged: dict[EnvironmentLifecyclePhase | str, object] = {
            phase: timeout for phase, timeout in _default_phase_timeouts().items()
        }
        merged.update(dict(supplied))
        copied["phase_timeout_seconds"] = merged
        return copied

    @field_validator("phase_timeout_seconds")
    @classmethod
    def freeze_phase_timeouts(
        cls,
        value: Mapping[EnvironmentLifecyclePhase, float],
    ) -> Mapping[EnvironmentLifecyclePhase, float]:
        return MappingProxyType(dict(value))

    @field_serializer("phase_timeout_seconds")
    def serialize_phase_timeouts(
        self,
        value: Mapping[EnvironmentLifecyclePhase, float],
    ) -> dict[str, float]:
        return {phase.value: timeout for phase, timeout in value.items()}

    @model_validator(mode="after")
    def validate_phase_timeouts(self) -> EnvironmentLifecyclePolicy:
        if set(self.phase_timeout_seconds) != set(EnvironmentLifecyclePhase):
            raise ValueError("phase_timeout_seconds must define every lifecycle phase.")
        return self

    def timeout_for(self, phase: EnvironmentLifecyclePhase) -> float:
        if not isinstance(phase, EnvironmentLifecyclePhase):
            raise TypeError("phase must be an EnvironmentLifecyclePhase.")
        return float(self.phase_timeout_seconds[phase])

    def __deepcopy__(
        self,
        memo: dict[int, object] | None = None,
    ) -> EnvironmentLifecyclePolicy:
        """Copy through the public form because mapping proxies are not pickleable."""

        del memo
        return EnvironmentLifecyclePolicy.model_validate(self.model_dump(mode="json"))


def copy_environment_lifecycle_policy(
    policy: EnvironmentLifecyclePolicy | None,
) -> EnvironmentLifecyclePolicy | None:
    if policy is None:
        return None
    if type(policy) is not EnvironmentLifecyclePolicy:
        raise TypeError("lifecycle_policy must be an EnvironmentLifecyclePolicy or None.")
    return EnvironmentLifecyclePolicy.model_validate(policy.model_dump(mode="json"))


class EnvironmentLifecycleProgress(BaseModel):
    """One bounded aggregate lifecycle observation safe for durable events."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    schema_version: Literal[1] = ENVIRONMENT_LIFECYCLE_PROGRESS_SCHEMA_VERSION
    operation_id: str = Field(min_length=1, max_length=128)
    binding_generation_id: str | None = Field(default=None, min_length=1, max_length=128)
    operation: EnvironmentLifecycleOperation
    phase: EnvironmentLifecyclePhase
    status: EnvironmentLifecycleProgressStatus
    event_index: StrictInt = Field(ge=1, le=MAX_ENVIRONMENT_PROGRESS_EVENTS)
    elapsed_ms: StrictInt = Field(ge=0, le=86_400_000)
    phase_elapsed_ms: StrictInt = Field(ge=0, le=86_400_000)
    lifecycle_timeout_seconds: StrictFloat = Field(gt=0, le=86_400)
    phase_timeout_seconds: StrictFloat = Field(gt=0, le=86_400)
    deadline_enforcement: Literal["cooperative_progress_boundary"] = "cooperative_progress_boundary"
    deadline_scope: Literal["lifecycle", "phase"] | None = None
    recovery_disposition: Literal["orphaned_stale"] | None = None
    deadline: datetime
    last_progress_at: datetime
    items_completed: StrictInt | None = Field(
        default=None, ge=0, le=MAX_ENVIRONMENT_PROGRESS_COUNTER
    )
    items_total: StrictInt | None = Field(default=None, ge=0, le=MAX_ENVIRONMENT_PROGRESS_COUNTER)
    bytes_completed: StrictInt | None = Field(
        default=None, ge=0, le=MAX_ENVIRONMENT_PROGRESS_COUNTER
    )
    bytes_total: StrictInt | None = Field(default=None, ge=0, le=MAX_ENVIRONMENT_PROGRESS_COUNTER)
    active_count: StrictInt | None = Field(default=None, ge=0, le=MAX_ENVIRONMENT_PROGRESS_COUNTER)
    queued_count: StrictInt | None = Field(default=None, ge=0, le=MAX_ENVIRONMENT_PROGRESS_COUNTER)
    operation_terminal: bool = False
    retained_owner: bool = False

    @model_validator(mode="after")
    def validate_projection(self) -> EnvironmentLifecycleProgress:
        if self.schema_version != ENVIRONMENT_LIFECYCLE_PROGRESS_SCHEMA_VERSION:
            raise ValueError("Environment lifecycle progress schema version is unsupported.")
        require_durable_clean_nonblank(self.operation_id, "operation_id")
        if not self.operation_id.isprintable():
            raise ValueError("operation_id must not contain control characters.")
        if self.binding_generation_id is not None:
            require_durable_clean_nonblank(
                self.binding_generation_id,
                "binding_generation_id",
            )
            if not self.binding_generation_id.isprintable():
                raise ValueError("binding_generation_id must not contain control characters.")
        for name in ("deadline", "last_progress_at"):
            value = getattr(self, name)
            if value.tzinfo is None or value.utcoffset() is None:
                raise ValueError(f"{name} must be timezone-aware.")
        if (
            self.items_completed is not None
            and self.items_total is not None
            and self.items_completed > self.items_total
        ):
            raise ValueError("items_completed cannot exceed items_total.")
        if (
            self.bytes_completed is not None
            and self.bytes_total is not None
            and self.bytes_completed > self.bytes_total
        ):
            raise ValueError("bytes_completed cannot exceed bytes_total.")
        if self.retained_owner and self.status not in {
            EnvironmentLifecycleProgressStatus.DEADLINE_EXCEEDED,
            EnvironmentLifecycleProgressStatus.RETAINED,
        }:
            raise ValueError("retained_owner requires a retained or deadline-exceeded status.")
        if self.operation_terminal and self.status not in _TERMINAL_PROGRESS_STATUSES:
            raise ValueError("operation_terminal requires a terminal lifecycle status.")
        if self.status is EnvironmentLifecycleProgressStatus.DEADLINE_EXCEEDED:
            if self.deadline_scope is None:
                raise ValueError("deadline-exceeded progress requires deadline_scope.")
        elif self.deadline_scope is not None:
            raise ValueError("deadline_scope is only valid for deadline-exceeded progress.")
        if self.recovery_disposition is not None and not (
            self.status is EnvironmentLifecycleProgressStatus.RETAINED
            and self.operation_terminal
            and self.retained_owner
        ):
            raise ValueError("recovery_disposition requires terminal retained ownership evidence.")
        return self

    def to_payload(self) -> dict[str, object]:
        return self.model_dump(mode="json", exclude_none=True)


def environment_lifecycle_progress_from_event(event: Event) -> EnvironmentLifecycleProgress:
    """Validate one runtime lifecycle event as the public typed projection."""

    if type(event) is not Event:
        raise TypeError("event must be an Event.")
    if event.type != EventType.ENVIRONMENT_LIFECYCLE_PROGRESS:
        raise ValueError("event is not environment lifecycle progress.")
    payload = dict(event.payload)
    execution_profile_fingerprint = payload.pop("execution_profile_fingerprint", None)
    if execution_profile_fingerprint is not None and (
        type(execution_profile_fingerprint) is not str
        or len(execution_profile_fingerprint) != 64
        or any(character not in "0123456789abcdef" for character in execution_profile_fingerprint)
    ):
        raise ValueError("Environment lifecycle execution profile authority is invalid.")
    return EnvironmentLifecycleProgress.model_validate(payload)


class EnvironmentLifecycleProgressReporter(Protocol):
    """Runtime-provided reporter available to environment adapters during lifecycle work."""

    @property
    def policy(self) -> EnvironmentLifecyclePolicy:
        """Return the frozen policy governing this operation."""

    async def report(
        self,
        phase: EnvironmentLifecyclePhase,
        status: EnvironmentLifecycleProgressStatus,
        *,
        items_completed: int | None = None,
        items_total: int | None = None,
        bytes_completed: int | None = None,
        bytes_total: int | None = None,
        active_count: int | None = None,
        queued_count: int | None = None,
    ) -> EnvironmentLifecycleProgress | None:
        """Publish one aggregate observation, or coalesce it under configured bounds."""


_CURRENT_ENVIRONMENT_LIFECYCLE_REPORTER: ContextVar[EnvironmentLifecycleProgressReporter | None] = (
    ContextVar("cayu_environment_lifecycle_progress_reporter", default=None)
)


def current_environment_lifecycle_progress_reporter() -> (
    EnvironmentLifecycleProgressReporter | None
):
    """Return the current runtime reporter without exposing session-owned internals."""

    return _CURRENT_ENVIRONMENT_LIFECYCLE_REPORTER.get()


def _set_environment_lifecycle_progress_reporter(
    reporter: EnvironmentLifecycleProgressReporter,
) -> Token[EnvironmentLifecycleProgressReporter | None]:
    return _CURRENT_ENVIRONMENT_LIFECYCLE_REPORTER.set(reporter)


def _reset_environment_lifecycle_progress_reporter(
    token: Token[EnvironmentLifecycleProgressReporter | None],
) -> None:
    _CURRENT_ENVIRONMENT_LIFECYCLE_REPORTER.reset(token)


class RuntimeEnvironmentLifecycleProgressReporter:
    """Bounded runtime implementation used behind the public reporter protocol."""

    def __init__(
        self,
        *,
        operation: EnvironmentLifecycleOperation,
        policy: EnvironmentLifecyclePolicy,
        publish: Callable[[EnvironmentLifecycleProgress], Awaitable[None]],
        binding_generation_id: str | None = None,
        operation_id: str | None = None,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        if not isinstance(operation, EnvironmentLifecycleOperation):
            raise TypeError("operation must be an EnvironmentLifecycleOperation.")
        if type(policy) is not EnvironmentLifecyclePolicy:
            raise TypeError("policy must be an EnvironmentLifecyclePolicy.")
        if not callable(publish):
            raise TypeError("publish must be callable.")
        self._operation = operation
        copied_policy = copy_environment_lifecycle_policy(policy)
        if copied_policy is None:  # pragma: no cover - guarded by the exact-type check above
            raise RuntimeError("Environment lifecycle policy copy unexpectedly returned None.")
        self._policy: EnvironmentLifecyclePolicy = copied_policy
        self._publish = publish
        if binding_generation_id is not None:
            binding_generation_id = require_durable_clean_nonblank(
                binding_generation_id,
                "binding_generation_id",
            )
            if len(binding_generation_id) > 128:
                raise ValueError("binding_generation_id cannot exceed 128 characters.")
        self._binding_generation_id = binding_generation_id
        self._now = now
        self._monotonic = monotonic
        self._operation_id = (
            f"envop_{uuid4().hex}"
            if operation_id is None
            else require_durable_clean_nonblank(operation_id, "operation_id")
        )
        if len(self._operation_id) > 128:
            raise ValueError("operation_id cannot exceed 128 characters.")
        self._started_at = self._aware_now()
        self._started_monotonic = monotonic()
        self._phase: EnvironmentLifecyclePhase | None = None
        self._phase_completed = False
        self._phase_started_at = self._started_at
        self._phase_started_monotonic = self._started_monotonic
        self._last_progress_at = self._started_at
        self._last_emitted_monotonic: float | None = None
        self._event_count = 0
        self._finished = False

    @property
    def policy(self) -> EnvironmentLifecyclePolicy:
        return self._policy

    @property
    def operation_id(self) -> str:
        return self._operation_id

    @property
    def finished(self) -> bool:
        """Return whether this operation already emitted its terminal boundary."""

        return self._finished

    async def report(
        self,
        phase: EnvironmentLifecyclePhase,
        status: EnvironmentLifecycleProgressStatus,
        *,
        items_completed: int | None = None,
        items_total: int | None = None,
        bytes_completed: int | None = None,
        bytes_total: int | None = None,
        active_count: int | None = None,
        queued_count: int | None = None,
    ) -> EnvironmentLifecycleProgress | None:
        if not isinstance(phase, EnvironmentLifecyclePhase):
            raise TypeError("phase must be an EnvironmentLifecyclePhase.")
        if not isinstance(status, EnvironmentLifecycleProgressStatus):
            raise TypeError("status must be an EnvironmentLifecycleProgressStatus.")
        if self._finished:
            raise RuntimeError("Environment lifecycle progress is already terminal.")
        now = self._aware_now()
        observed_monotonic = self._monotonic()
        deadline_phase = self._phase or phase
        deadline_scope = self._deadline_scope(
            observed_monotonic,
            deadline_phase,
            include_phase=self._phase is not None and not self._phase_completed,
        )
        if deadline_scope is not None:
            self._finished = True
            await self._emit(
                deadline_phase,
                EnvironmentLifecycleProgressStatus.DEADLINE_EXCEEDED,
                now=now,
                observed_monotonic=observed_monotonic,
                last_progress_at=self._last_progress_at,
                deadline_scope=deadline_scope,
                operation_terminal=True,
                retained_owner=False,
            )
            raise EnvironmentLifecycleDeadlineExceeded(
                operation_id=self._operation_id,
                phase=deadline_phase,
                scope=deadline_scope,
            )
        if phase is not self._phase or (
            self._phase_completed and status is EnvironmentLifecycleProgressStatus.STARTED
        ):
            self._phase = phase
            self._phase_completed = False
            self._phase_started_at = now
            self._phase_started_monotonic = observed_monotonic
        if status is EnvironmentLifecycleProgressStatus.ADVANCED:
            if self._event_count >= self._policy.max_progress_events - 1:
                return None
            if (
                self._last_emitted_monotonic is not None
                and observed_monotonic - self._last_emitted_monotonic
                < self._policy.progress_min_interval_seconds
            ):
                return None
        elif self._event_count >= self._policy.max_progress_events - 1:
            return None
        progress = await self._emit(
            phase,
            status,
            now=now,
            observed_monotonic=observed_monotonic,
            items_completed=items_completed,
            items_total=items_total,
            bytes_completed=bytes_completed,
            bytes_total=bytes_total,
            active_count=active_count,
            queued_count=queued_count,
            retained_owner=False,
        )
        if status is EnvironmentLifecycleProgressStatus.COMPLETED:
            self._phase_completed = True
        return progress

    async def finish(
        self,
        *,
        status: EnvironmentLifecycleProgressStatus,
        phase: EnvironmentLifecyclePhase | None = None,
        retained_owner: bool = False,
    ) -> EnvironmentLifecycleProgress:
        if status not in _TERMINAL_PROGRESS_STATUSES:
            raise ValueError("Environment lifecycle finish status must be terminal.")
        if self._finished:
            raise RuntimeError("Environment lifecycle progress is already terminal.")
        terminal_phase = phase or self._phase or EnvironmentLifecyclePhase.OWNERSHIP_ADMISSION
        now = self._aware_now()
        observed_monotonic = self._monotonic()
        if terminal_phase is not self._phase:
            self._phase = terminal_phase
            self._phase_completed = False
            self._phase_started_at = now
            self._phase_started_monotonic = observed_monotonic
        deadline_scope = self._deadline_scope(
            observed_monotonic,
            terminal_phase,
            include_phase=not self._phase_completed,
        )
        if deadline_scope is not None:
            status = EnvironmentLifecycleProgressStatus.DEADLINE_EXCEEDED
        self._finished = True
        progress = await self._emit(
            terminal_phase,
            status,
            now=now,
            observed_monotonic=observed_monotonic,
            last_progress_at=(self._last_progress_at if deadline_scope is not None else None),
            deadline_scope=deadline_scope,
            operation_terminal=True,
            retained_owner=retained_owner,
        )
        if deadline_scope is not None:
            raise EnvironmentLifecycleDeadlineExceeded(
                operation_id=self._operation_id,
                phase=terminal_phase,
                scope=deadline_scope,
            )
        return progress

    def _deadline_scope(
        self,
        observed_monotonic: float,
        phase: EnvironmentLifecyclePhase,
        *,
        include_phase: bool,
    ) -> Literal["lifecycle", "phase"] | None:
        if observed_monotonic - self._started_monotonic >= self._policy.lifecycle_timeout_seconds:
            return "lifecycle"
        if include_phase and (
            phase is self._phase
            and observed_monotonic - self._phase_started_monotonic
            >= self._policy.timeout_for(phase)
        ):
            return "phase"
        return None

    async def _emit(
        self,
        phase: EnvironmentLifecyclePhase,
        status: EnvironmentLifecycleProgressStatus,
        *,
        now: datetime,
        observed_monotonic: float,
        items_completed: int | None = None,
        items_total: int | None = None,
        bytes_completed: int | None = None,
        bytes_total: int | None = None,
        active_count: int | None = None,
        queued_count: int | None = None,
        last_progress_at: datetime | None = None,
        deadline_scope: Literal["lifecycle", "phase"] | None = None,
        operation_terminal: bool = False,
        retained_owner: bool,
    ) -> EnvironmentLifecycleProgress:
        self._event_count += 1
        phase_timeout = self._policy.timeout_for(phase)
        lifecycle_deadline = self._started_at + timedelta(
            seconds=self._policy.lifecycle_timeout_seconds
        )
        phase_deadline = self._phase_started_at + timedelta(seconds=phase_timeout)
        progress = EnvironmentLifecycleProgress(
            operation_id=self._operation_id,
            binding_generation_id=self._binding_generation_id,
            operation=self._operation,
            phase=phase,
            status=status,
            event_index=self._event_count,
            elapsed_ms=_elapsed_ms(observed_monotonic - self._started_monotonic),
            phase_elapsed_ms=_elapsed_ms(observed_monotonic - self._phase_started_monotonic),
            lifecycle_timeout_seconds=self._policy.lifecycle_timeout_seconds,
            phase_timeout_seconds=phase_timeout,
            deadline_scope=deadline_scope,
            deadline=min(lifecycle_deadline, phase_deadline),
            last_progress_at=now if last_progress_at is None else last_progress_at,
            items_completed=items_completed,
            items_total=items_total,
            bytes_completed=bytes_completed,
            bytes_total=bytes_total,
            active_count=active_count,
            queued_count=queued_count,
            operation_terminal=operation_terminal,
            retained_owner=retained_owner,
        )
        await self._publish(progress)
        self._last_progress_at = now
        self._last_emitted_monotonic = observed_monotonic
        return progress

    def _aware_now(self) -> datetime:
        value = self._now()
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("Environment lifecycle clock must return a timezone-aware datetime.")
        return value.astimezone(UTC)


def _elapsed_ms(seconds: float) -> int:
    return min(86_400_000, max(0, round(seconds * 1000)))


__all__ = [
    "DEFAULT_ENVIRONMENT_LIFECYCLE_TIMEOUT_SECONDS",
    "DEFAULT_ENVIRONMENT_PHASE_TIMEOUT_SECONDS",
    "DEFAULT_ENVIRONMENT_PROGRESS_MIN_INTERVAL_SECONDS",
    "DEFAULT_MAX_ENVIRONMENT_PROGRESS_EVENTS",
    "ENVIRONMENT_LIFECYCLE_PROGRESS_SCHEMA_VERSION",
    "MAX_ENVIRONMENT_PROGRESS_COUNTER",
    "EnvironmentLifecycleDeadlineExceeded",
    "EnvironmentLifecycleOperation",
    "EnvironmentLifecyclePhase",
    "EnvironmentLifecyclePolicy",
    "EnvironmentLifecycleProgress",
    "EnvironmentLifecycleProgressReporter",
    "EnvironmentLifecycleProgressStatus",
    "copy_environment_lifecycle_policy",
    "current_environment_lifecycle_progress_reporter",
    "environment_lifecycle_progress_from_event",
]

"""Bounded policy and supervision for Runtime recovery cleanup."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from contextvars import Context, copy_context
from dataclasses import dataclass
from enum import StrEnum
from math import isfinite
from typing import Literal, cast

from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator

from cayu._exception_groups import (
    exception_cause,
    exception_context,
    exception_suppresses_context,
    set_exception_cause,
)

DEFAULT_RECOVERY_CLEANUP_STEP_TIMEOUT_SECONDS = 30.0
DEFAULT_RECOVERY_CLEANUP_OVERALL_TIMEOUT_SECONDS = 120.0
DEFAULT_RECOVERY_CLEANUP_MAX_SUPERVISED_TASKS = 256
RECOVERY_CLEANUP_MAX_TIMEOUT_SECONDS = 86_400.0
RECOVERY_CLEANUP_OPERATION_MAX_CHARS = 160
_RECOVERY_CLEANUP_MAX_STEPS = 64

RecoveryCleanup = Callable[[], Awaitable[None]]


@dataclass(frozen=True)
class RecoveryCleanupStep:
    """One cleanup operation and its explicit dependency relationship."""

    operation: str
    cleanup: RecoveryCleanup
    independent_with_previous: bool = False


RecoveryCleanupStepInput = tuple[str, RecoveryCleanup] | RecoveryCleanupStep

logger = logging.getLogger(__name__)


def _require_positive_finite_seconds(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{field_name} must be a number.")
    resolved = float(value)
    if not isfinite(resolved) or resolved <= 0 or resolved > RECOVERY_CLEANUP_MAX_TIMEOUT_SECONDS:
        raise ValueError(
            f"{field_name} must be greater than zero and at most "
            f"{RECOVERY_CLEANUP_MAX_TIMEOUT_SECONDS:g} seconds."
        )
    return resolved


def _require_operation_name(value: object, field_name: str) -> str:
    if type(value) is not str or not value or value.strip() != value:
        raise ValueError(f"{field_name} must be a non-blank stripped string.")
    if len(value) > RECOVERY_CLEANUP_OPERATION_MAX_CHARS:
        raise ValueError(
            f"{field_name} must be at most {RECOVERY_CLEANUP_OPERATION_MAX_CHARS} characters."
        )
    return value


class RecoveryCleanupPolicy(BaseModel):
    """Finite deadlines for one ordered Runtime recovery-cleanup sequence."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    step_timeout_seconds: float = DEFAULT_RECOVERY_CLEANUP_STEP_TIMEOUT_SECONDS
    overall_timeout_seconds: float = DEFAULT_RECOVERY_CLEANUP_OVERALL_TIMEOUT_SECONDS
    max_supervised_tasks: StrictInt = Field(
        default=DEFAULT_RECOVERY_CLEANUP_MAX_SUPERVISED_TASKS,
        ge=1,
        le=4096,
    )

    @field_validator("step_timeout_seconds", "overall_timeout_seconds", mode="before")
    @classmethod
    def validate_timeout_seconds(cls, value: object, info) -> float:
        return _require_positive_finite_seconds(value, info.field_name)


def copy_recovery_cleanup_policy(
    policy: RecoveryCleanupPolicy | None,
) -> RecoveryCleanupPolicy:
    if policy is None:
        return RecoveryCleanupPolicy()
    if type(policy) is not RecoveryCleanupPolicy:
        raise TypeError("recovery_cleanup_policy must be a RecoveryCleanupPolicy.")
    return RecoveryCleanupPolicy.model_validate(policy.model_dump(mode="python", warnings=False))


class RecoveryCleanupDeadlineScope(StrEnum):
    STEP = "step"
    OVERALL = "overall"


class RecoveryCleanupDeadlineEvidence(BaseModel):
    """Portable, content-free evidence for an outcome-unknown cleanup step."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    code: Literal["recovery_cleanup_deadline_exceeded"] = "recovery_cleanup_deadline_exceeded"
    operation: str
    scope: RecoveryCleanupDeadlineScope
    timeout_seconds: float
    outcome_unknown: Literal[True] = True

    @field_validator("operation", mode="before")
    @classmethod
    def validate_operation(cls, value: object) -> str:
        return _require_operation_name(value, "operation")

    @field_validator("timeout_seconds", mode="before")
    @classmethod
    def validate_timeout_seconds(cls, value: object) -> float:
        return _require_positive_finite_seconds(value, "timeout_seconds")


class RecoveryCleanupDeadlineExceeded(TimeoutError):
    """One cleanup step transferred to retained supervision at a finite deadline."""

    code: Literal["recovery_cleanup_deadline_exceeded"] = "recovery_cleanup_deadline_exceeded"

    def __init__(
        self,
        *,
        operation: str,
        scope: RecoveryCleanupDeadlineScope,
        timeout_seconds: float,
    ) -> None:
        operation = _require_operation_name(operation, "operation")
        if type(scope) is not RecoveryCleanupDeadlineScope:
            raise TypeError("scope must be a RecoveryCleanupDeadlineScope.")
        resolved_timeout = _require_positive_finite_seconds(
            timeout_seconds,
            "timeout_seconds",
        )
        self.operation = operation
        self.scope = scope
        self.timeout_seconds = resolved_timeout
        super().__init__(
            f"Recovery cleanup {scope.value} deadline exceeded during {operation} "
            f"after {resolved_timeout:g} seconds; exact ownership remains retained."
        )

    def evidence(self) -> RecoveryCleanupDeadlineEvidence:
        return RecoveryCleanupDeadlineEvidence(
            code=self.code,
            operation=self.operation,
            scope=self.scope,
            timeout_seconds=self.timeout_seconds,
        )


class RecoveryCleanupRetainedTaskSnapshot(BaseModel):
    """Bounded process-local identity for one retained cleanup task."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    operation: str
    scope: RecoveryCleanupDeadlineScope
    timeout_seconds: float
    outcome_unknown: Literal[True] = True
    caller_cancellation_observed: bool

    @field_validator("operation", mode="before")
    @classmethod
    def validate_operation(cls, value: object) -> str:
        return _require_operation_name(value, "operation")

    @field_validator("timeout_seconds", mode="before")
    @classmethod
    def validate_timeout_seconds(cls, value: object) -> float:
        return _require_positive_finite_seconds(value, "timeout_seconds")


class RecoveryCleanupSupervisorSnapshot(BaseModel):
    """Content-free process-local cleanup supervision state."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    active_tasks: StrictInt = Field(ge=0)
    retained_tasks: StrictInt = Field(ge=0)
    timed_out_steps: StrictInt = Field(ge=0)
    completed_after_timeout: StrictInt = Field(ge=0)
    failed_after_timeout: StrictInt = Field(ge=0)
    retained_after_cancellation: StrictInt = Field(ge=0)
    capacity_exhausted_steps: StrictInt = Field(ge=0)
    retained: tuple[RecoveryCleanupRetainedTaskSnapshot, ...] = ()


class RecoveryCleanupCapacityExceeded(RuntimeError):
    """A new cleanup cannot be supervised without exceeding its configured bound."""

    code: Literal["recovery_cleanup_capacity_exceeded"] = "recovery_cleanup_capacity_exceeded"

    def __init__(self, *, operation: str, max_supervised_tasks: int) -> None:
        self.operation = _require_operation_name(operation, "operation")
        if type(max_supervised_tasks) is not int or max_supervised_tasks <= 0:
            raise ValueError("max_supervised_tasks must be a positive integer.")
        self.max_supervised_tasks = max_supervised_tasks
        super().__init__(
            "Recovery cleanup supervision capacity was exhausted before "
            f"{operation}; at most {max_supervised_tasks} cleanup tasks may be supervised, "
            "and existing ownership remains unchanged."
        )


@dataclass(frozen=True)
class _RetainedCleanup:
    operation: str
    scope: RecoveryCleanupDeadlineScope
    timeout_seconds: float
    caller_cancellation_observed: bool
    cancellation_handle: asyncio.TimerHandle | None = None
    continuation_barrier: _CleanupContinuationBarrier | None = None


@dataclass
class _RunningCleanupStep:
    phase_ordinal: int
    step: RecoveryCleanupStep
    task: asyncio.Task[BaseException | None]
    context: Context
    started_at: float
    caller_cancellation_forwarded: bool = False


@dataclass(frozen=True)
class _CleanupExecutionGroup:
    steps: tuple[RecoveryCleanupStep, ...]
    independent: bool


@dataclass
class _SequentialCleanupProgress:
    phase_ordinal: int
    operation: str
    started_at: float
    failures: list[BaseException | None]
    abort_after_current: bool = False
    caller_cancellation: asyncio.CancelledError | None = None
    caller_cancellation_forwarded_ordinal: int | None = None


@dataclass(frozen=True)
class _ObservedCallerCancellation:
    step_ordinal: int
    operation: str
    error: asyncio.CancelledError
    forwarded_to_observed_step: bool


@dataclass
class _CleanupContinuationBarrier:
    pending_tasks: set[asyncio.Task[BaseException | None]]
    steps: tuple[RecoveryCleanupStep, ...]
    context: Context
    sealed: bool = False
    scheduled: bool = False


class RecoveryCleanupSupervisor:
    """Run bounded cleanup steps and retain outcome-unknown tasks until settlement."""

    def __init__(self, policy: RecoveryCleanupPolicy | None = None) -> None:
        self._policy = copy_recovery_cleanup_policy(policy)
        self._active_tasks: set[asyncio.Task[BaseException | None]] = set()
        self._retained_tasks: dict[asyncio.Task[BaseException | None], _RetainedCleanup] = {}
        self._continuation_tasks: set[asyncio.Task[None]] = set()
        self._timed_out_steps = 0
        self._completed_after_timeout = 0
        self._failed_after_timeout = 0
        self._retained_after_cancellation = 0
        self._capacity_exhausted_steps = 0

    @property
    def policy(self) -> RecoveryCleanupPolicy:
        return copy_recovery_cleanup_policy(self._policy)

    def snapshot(self) -> RecoveryCleanupSupervisorSnapshot:
        self._harvest_completed()
        return RecoveryCleanupSupervisorSnapshot(
            active_tasks=len(self._active_tasks) + len(self._continuation_tasks),
            retained_tasks=len(self._retained_tasks),
            timed_out_steps=self._timed_out_steps,
            completed_after_timeout=self._completed_after_timeout,
            failed_after_timeout=self._failed_after_timeout,
            retained_after_cancellation=self._retained_after_cancellation,
            capacity_exhausted_steps=self._capacity_exhausted_steps,
            retained=tuple(
                RecoveryCleanupRetainedTaskSnapshot(
                    operation=retained.operation,
                    scope=retained.scope,
                    timeout_seconds=retained.timeout_seconds,
                    caller_cancellation_observed=(retained.caller_cancellation_observed),
                )
                for retained in sorted(
                    self._retained_tasks.values(),
                    key=lambda item: (item.operation, item.scope.value),
                )
            ),
        )

    async def drain(self, *, timeout_s: float) -> bool:
        """Wait boundedly for supervised work without cancelling retained owners."""

        timeout_s = _require_positive_finite_seconds(
            timeout_s,
            "timeout_s",
        )
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout_s
        while True:
            self._harvest_completed()
            pending = tuple(
                self._active_tasks | self._retained_tasks.keys() | self._continuation_tasks
            )
            if not pending:
                return True
            remaining = deadline - loop.time()
            if remaining <= 0:
                return False
            done, _pending = await asyncio.wait(
                pending,
                timeout=remaining,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if not done:
                return False
            # Active step owners are transitioned by their run_steps caller.
            # Yield instead of removing that reservation here so drain cannot
            # observe a false empty window before a dependent successor starts.
            if any(task in self._active_tasks for task in done):
                await asyncio.sleep(0)

    async def run_steps(
        self,
        *,
        steps: tuple[RecoveryCleanupStepInput, ...],
        shield_caller_cancellation: bool,
    ) -> tuple[tuple[str, BaseException], ...]:
        """Run dependency phases within finite bounds and retain uncertain owners."""

        self._harvest_completed()
        validated_steps = self._validate_steps(steps)
        groups = self._group_execution_groups(validated_steps)
        loop = asyncio.get_running_loop()
        overall_deadline = loop.time() + self._policy.overall_timeout_seconds
        failures: list[tuple[str, BaseException]] = []
        sequence_context = copy_context()
        caller_cancellation: _ObservedCallerCancellation | None = None
        caller_cancellation_recorded = False

        def append_group_failures(
            *,
            group_start_ordinal: int,
            phase: tuple[RecoveryCleanupStep, ...],
            phase_failures: list[BaseException | None],
        ) -> None:
            """Publish one group's failures and cancellation in step order."""

            nonlocal caller_cancellation_recorded
            for phase_ordinal, (step, failure) in enumerate(
                zip(phase, phase_failures, strict=True)
            ):
                matching_cancellation = (
                    caller_cancellation
                    if caller_cancellation is not None
                    and not caller_cancellation_recorded
                    and caller_cancellation.step_ordinal == group_start_ordinal + phase_ordinal
                    else None
                )
                if (
                    matching_cancellation is not None
                    and matching_cancellation.forwarded_to_observed_step
                ):
                    failures.append(
                        (
                            matching_cancellation.operation,
                            matching_cancellation.error,
                        )
                    )
                if failure is not None:
                    failures.append((step.operation, failure))
                if (
                    matching_cancellation is not None
                    and not matching_cancellation.forwarded_to_observed_step
                ):
                    failures.append(
                        (
                            matching_cancellation.operation,
                            matching_cancellation.error,
                        )
                    )
                if matching_cancellation is not None:
                    caller_cancellation_recorded = True

        for group_index, group in enumerate(groups):
            self._harvest_completed()
            phase = group.steps
            group_start_ordinal = sum(
                len(earlier_group.steps) for earlier_group in groups[:group_index]
            )
            remaining_steps = tuple(
                step
                for remaining_group in groups[group_index + 1 :]
                for step in remaining_group.steps
            )
            if loop.time() >= overall_deadline:
                admitted_phase = phase if group.independent else phase[:1]
                deferred_steps = (
                    remaining_steps if group.independent else phase[1:] + remaining_steps
                )
                self._admit_phase_after_overall_deadline(
                    phase=admitted_phase,
                    remaining_steps=deferred_steps,
                    sequence_context=sequence_context,
                    failures=failures,
                )
                break

            if not group.independent:
                if not self._has_task_capacity():
                    first_step = phase[0]
                    self._capacity_exhausted_steps += 1
                    failures.append(
                        (
                            first_step.operation,
                            RecoveryCleanupCapacityExceeded(
                                operation=first_step.operation,
                                max_supervised_tasks=self._policy.max_supervised_tasks,
                            ),
                        )
                    )
                    break

                segment_context = sequence_context.copy()
                segment_failures: list[BaseException | None] = [None] * len(phase)
                progress = _SequentialCleanupProgress(
                    phase_ordinal=0,
                    operation=phase[0].operation,
                    started_at=loop.time(),
                    failures=segment_failures,
                    caller_cancellation=(
                        None if caller_cancellation is None else caller_cancellation.error
                    ),
                )
                task = asyncio.create_task(
                    self._run_sequential_group(steps=phase, progress=progress),
                    name=f"cayu-recovery-cleanup:{phase[0].operation}",
                    context=segment_context,
                )
                self._active_tasks.add(task)
                deadline_scope = RecoveryCleanupDeadlineScope.STEP
                while not task.done():
                    observed_ordinal = progress.phase_ordinal
                    step_deadline = progress.started_at + self._policy.step_timeout_seconds
                    deadline_scope = (
                        RecoveryCleanupDeadlineScope.OVERALL
                        if overall_deadline <= step_deadline
                        else RecoveryCleanupDeadlineScope.STEP
                    )
                    active_deadline = min(step_deadline, overall_deadline)
                    remaining = active_deadline - loop.time()
                    if remaining <= 0:
                        if progress.phase_ordinal != observed_ordinal:
                            continue
                        break
                    try:
                        await asyncio.wait({task}, timeout=remaining)
                    except asyncio.CancelledError as cancellation:
                        if shield_caller_cancellation:
                            continue
                        if caller_cancellation is None:
                            progress.caller_cancellation = cancellation
                            cancellation_was_forwarded = False
                            if (
                                progress.caller_cancellation_forwarded_ordinal
                                != progress.phase_ordinal
                            ):
                                cancellation_was_forwarded = task.cancel(*cancellation.args)
                                if cancellation_was_forwarded:
                                    progress.caller_cancellation_forwarded_ordinal = (
                                        progress.phase_ordinal
                                    )
                            caller_cancellation = _ObservedCallerCancellation(
                                step_ordinal=group_start_ordinal + progress.phase_ordinal,
                                operation=progress.operation,
                                error=cancellation,
                                forwarded_to_observed_step=cancellation_was_forwarded,
                            )
                        else:
                            progress.caller_cancellation = caller_cancellation.error
                            if (
                                progress.caller_cancellation_forwarded_ordinal
                                != progress.phase_ordinal
                                and task.cancel(*cancellation.args)
                            ):
                                progress.caller_cancellation_forwarded_ordinal = (
                                    progress.phase_ordinal
                                )
                        continue

                if task.done():
                    self._active_tasks.discard(task)
                    task_failure = self._completed_task_failure(task)
                    if (
                        not (
                            isinstance(task_failure, asyncio.CancelledError)
                            and progress.caller_cancellation_forwarded_ordinal is not None
                            and caller_cancellation is not None
                        )
                        and task_failure is not None
                    ):
                        segment_failures[progress.phase_ordinal] = task_failure
                    append_group_failures(
                        group_start_ordinal=group_start_ordinal,
                        phase=phase,
                        phase_failures=segment_failures,
                    )
                    sequence_context = segment_context
                    continue

                timed_out_ordinal = progress.phase_ordinal
                timed_out_step = phase[timed_out_ordinal]
                progress.abort_after_current = True
                continuation_steps = phase[timed_out_ordinal + 1 :] + remaining_steps
                continuation_barrier = (
                    _CleanupContinuationBarrier(
                        pending_tasks={task},
                        steps=continuation_steps,
                        context=segment_context,
                    )
                    if continuation_steps
                    else None
                )
                cancellation_was_forwarded = (
                    progress.caller_cancellation_forwarded_ordinal == timed_out_ordinal
                )
                deadline_failure = self._retain_timed_out(
                    task,
                    operation=timed_out_step.operation,
                    scope=deadline_scope,
                    timeout_seconds=(
                        self._policy.overall_timeout_seconds
                        if deadline_scope is RecoveryCleanupDeadlineScope.OVERALL
                        else self._policy.step_timeout_seconds
                    ),
                    request_cancellation=not cancellation_was_forwarded,
                    caller_cancellation_observed=(caller_cancellation is not None),
                    continuation_barrier=continuation_barrier,
                )
                if cancellation_was_forwarded and caller_cancellation is not None:
                    self._retained_after_cancellation += 1
                    cancellation = caller_cancellation.error
                    cancellation.add_note(
                        "Recovery cleanup remained outcome-unknown during "
                        f"{timed_out_step.operation}."
                    )
                    self._attach_cancellation_deadline_evidence(
                        cancellation,
                        (deadline_failure,),
                    )
                else:
                    segment_failures[timed_out_ordinal] = deadline_failure
                append_group_failures(
                    group_start_ordinal=group_start_ordinal,
                    phase=phase,
                    phase_failures=segment_failures,
                )
                if continuation_barrier is not None:
                    self._seal_continuation_barrier(continuation_barrier)
                break

            phase_states: list[_RunningCleanupStep] = []
            phase_failures: list[BaseException | None] = [None] * len(phase)
            phase_capacity_exhausted = False
            for phase_ordinal, step in enumerate(phase):
                self._harvest_completed()
                if not self._has_task_capacity():
                    phase_capacity_exhausted = True
                    self._capacity_exhausted_steps += 1
                    phase_failures[phase_ordinal] = RecoveryCleanupCapacityExceeded(
                        operation=step.operation,
                        max_supervised_tasks=self._policy.max_supervised_tasks,
                    )
                    continue
                step_context = sequence_context.copy()
                task = asyncio.create_task(
                    self._capture_cleanup(step.cleanup),
                    name=f"cayu-recovery-cleanup:{step.operation}",
                    context=step_context,
                )
                self._active_tasks.add(task)
                phase_states.append(
                    _RunningCleanupStep(
                        phase_ordinal=phase_ordinal,
                        step=step,
                        task=task,
                        context=step_context,
                        started_at=loop.time(),
                    )
                )

            if not phase_states:
                failures.extend(
                    (phase[ordinal].operation, failure)
                    for ordinal, failure in enumerate(phase_failures)
                    if failure is not None
                )
                break

            continuation_barrier: _CleanupContinuationBarrier | None = None
            if remaining_steps and not phase_capacity_exhausted:
                continuation_context = (
                    phase_states[0].context if len(phase) == 1 else sequence_context.copy()
                )
                continuation_barrier = _CleanupContinuationBarrier(
                    pending_tasks=set(),
                    steps=remaining_steps,
                    context=continuation_context,
                )

            running = list(phase_states)
            phase_timed_out = False
            cancellation_deadline_failures: list[RecoveryCleanupDeadlineExceeded] = []
            while running:
                completed = [state for state in running if state.task.done()]
                for state in completed:
                    running.remove(state)
                    self._active_tasks.discard(state.task)
                    failure = self._completed_task_failure(state.task)
                    if (
                        isinstance(failure, asyncio.CancelledError)
                        and state.caller_cancellation_forwarded
                        and caller_cancellation is not None
                    ):
                        # The original caller cancellation was recorded once at
                        # delivery; child-task cancellations are its mechanism.
                        failure = None
                    if failure is not None:
                        phase_failures[state.phase_ordinal] = failure
                if not running:
                    break

                now = loop.time()
                timed_out = [
                    state
                    for state in running
                    if now
                    >= min(
                        state.started_at + self._policy.step_timeout_seconds,
                        overall_deadline,
                    )
                ]
                if timed_out:
                    phase_timed_out = True
                    for state in timed_out:
                        running.remove(state)
                        step_deadline = state.started_at + self._policy.step_timeout_seconds
                        scope = (
                            RecoveryCleanupDeadlineScope.OVERALL
                            if overall_deadline <= step_deadline
                            else RecoveryCleanupDeadlineScope.STEP
                        )
                        if continuation_barrier is not None:
                            continuation_barrier.pending_tasks.add(state.task)
                        deadline_failure = self._retain_timed_out(
                            state.task,
                            operation=state.step.operation,
                            scope=scope,
                            timeout_seconds=(
                                self._policy.overall_timeout_seconds
                                if scope is RecoveryCleanupDeadlineScope.OVERALL
                                else self._policy.step_timeout_seconds
                            ),
                            request_cancellation=(not state.caller_cancellation_forwarded),
                            caller_cancellation_observed=(state.caller_cancellation_forwarded),
                            continuation_barrier=continuation_barrier,
                        )
                        if state.caller_cancellation_forwarded and caller_cancellation is not None:
                            self._retained_after_cancellation += 1
                            cancellation = caller_cancellation.error
                            cancellation.add_note(
                                "Recovery cleanup remained outcome-unknown during "
                                f"{state.step.operation}."
                            )
                            cancellation_deadline_failures.append(deadline_failure)
                        else:
                            phase_failures[state.phase_ordinal] = deadline_failure
                    continue

                active_deadline = min(
                    overall_deadline,
                    *(state.started_at + self._policy.step_timeout_seconds for state in running),
                )
                try:
                    await asyncio.wait(
                        {state.task for state in running},
                        timeout=max(0.0, active_deadline - loop.time()),
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                except asyncio.CancelledError as cancellation:
                    if shield_caller_cancellation:
                        continue
                    if caller_cancellation is None:
                        cancellation_step = next(
                            (state for state in running if not state.task.done()),
                            running[-1],
                        )
                        cancellation_was_forwarded = False
                        for state in running:
                            if state.caller_cancellation_forwarded:
                                continue
                            forwarded = state.task.cancel(*cancellation.args)
                            state.caller_cancellation_forwarded = forwarded
                            if state is cancellation_step:
                                cancellation_was_forwarded = forwarded
                        caller_cancellation = _ObservedCallerCancellation(
                            step_ordinal=(group_start_ordinal + cancellation_step.phase_ordinal),
                            operation=cancellation_step.step.operation,
                            error=cancellation,
                            forwarded_to_observed_step=cancellation_was_forwarded,
                        )
                        continue
                    for state in running:
                        if state.caller_cancellation_forwarded:
                            continue
                        state.caller_cancellation_forwarded = state.task.cancel(*cancellation.args)
                    # Give exact cleanup owners the remainder of their finite
                    # budgets to reconcile an acknowledgement-boundary cancel.
                    continue

            if cancellation_deadline_failures and caller_cancellation is not None:
                self._attach_cancellation_deadline_evidence(
                    caller_cancellation.error,
                    tuple(cancellation_deadline_failures),
                )
            append_group_failures(
                group_start_ordinal=group_start_ordinal,
                phase=phase,
                phase_failures=phase_failures,
            )
            if phase_timed_out:
                if continuation_barrier is not None:
                    self._seal_continuation_barrier(continuation_barrier)
                break
            if phase_capacity_exhausted:
                # A dependency phase was not fully attempted. Later phases must
                # remain durable for ordinary exact recovery rather than racing it.
                break
            if len(phase) == 1:
                sequence_context = phase_states[0].context

        self._install_context(sequence_context)
        return tuple(failures)

    def _admit_phase_after_overall_deadline(
        self,
        *,
        phase: tuple[RecoveryCleanupStep, ...],
        remaining_steps: tuple[RecoveryCleanupStep, ...],
        sequence_context: Context,
        failures: list[tuple[str, BaseException]],
    ) -> None:
        """Admit one dependency phase without extending the bounded caller."""

        admitted: list[_RunningCleanupStep] = []
        phase_failures: list[BaseException | None] = [None] * len(phase)
        capacity_exhausted = False
        loop = asyncio.get_running_loop()
        for phase_ordinal, step in enumerate(phase):
            self._harvest_completed()
            if not self._has_task_capacity():
                capacity_exhausted = True
                self._capacity_exhausted_steps += 1
                phase_failures[phase_ordinal] = RecoveryCleanupCapacityExceeded(
                    operation=step.operation,
                    max_supervised_tasks=self._policy.max_supervised_tasks,
                )
                continue
            step_context = sequence_context.copy()
            task = asyncio.create_task(
                self._capture_cleanup(step.cleanup),
                name=f"cayu-recovery-cleanup:{step.operation}",
                context=step_context,
            )
            self._active_tasks.add(task)
            admitted.append(
                _RunningCleanupStep(
                    phase_ordinal=phase_ordinal,
                    step=step,
                    task=task,
                    context=step_context,
                    started_at=loop.time(),
                )
            )

        continuation_barrier: _CleanupContinuationBarrier | None = None
        if admitted and len(admitted) == len(phase) and remaining_steps and not capacity_exhausted:
            continuation_barrier = _CleanupContinuationBarrier(
                pending_tasks={state.task for state in admitted},
                steps=remaining_steps,
                context=(admitted[0].context if len(phase) == 1 else sequence_context.copy()),
            )
        for state in admitted:
            failure = self._retain_timed_out(
                state.task,
                operation=state.step.operation,
                scope=RecoveryCleanupDeadlineScope.OVERALL,
                timeout_seconds=self._policy.overall_timeout_seconds,
                request_cancellation=False,
                cancel_after_seconds=self._policy.step_timeout_seconds,
                continuation_barrier=continuation_barrier,
            )
            phase_failures[state.phase_ordinal] = failure
        failures.extend(
            (phase[ordinal].operation, failure)
            for ordinal, failure in enumerate(phase_failures)
            if failure is not None
        )
        if continuation_barrier is not None:
            self._seal_continuation_barrier(continuation_barrier)

    @staticmethod
    def _install_context(context: Context) -> None:
        """Preserve the ContextVar effects that inline cleanup historically had."""

        for variable, value in context.items():
            variable.set(value)

    @staticmethod
    async def _capture_cleanup(cleanup: RecoveryCleanup) -> BaseException | None:
        try:
            await cleanup()
        except BaseException as error:
            return error
        return None

    @staticmethod
    async def _run_sequential_group(
        *,
        steps: tuple[RecoveryCleanupStep, ...],
        progress: _SequentialCleanupProgress,
    ) -> BaseException | None:
        loop = asyncio.get_running_loop()
        for phase_ordinal, step in enumerate(steps):
            progress.phase_ordinal = phase_ordinal
            progress.operation = step.operation
            progress.started_at = loop.time()
            settlement_failure: BaseException | None = None
            try:
                await step.cleanup()
            except BaseException as error:
                settlement_failure = error
                current_failure: BaseException | None = error
                if (
                    isinstance(error, asyncio.CancelledError)
                    and progress.caller_cancellation is not None
                    and progress.caller_cancellation_forwarded_ordinal == phase_ordinal
                ):
                    current_failure = None
                if current_failure is not None:
                    progress.failures[phase_ordinal] = current_failure
            if progress.abort_after_current:
                return settlement_failure
        return None

    @staticmethod
    def _attach_cancellation_deadline_evidence(
        cancellation: asyncio.CancelledError,
        deadline_failures: tuple[RecoveryCleanupDeadlineExceeded, ...],
    ) -> None:
        """Attach every retained owner without discarding prior causal evidence."""

        if not deadline_failures:
            return
        existing = exception_cause(cancellation)
        if existing is None and not exception_suppresses_context(cancellation):
            existing = exception_context(cancellation)
        causes: list[BaseException] = list(deadline_failures)
        if existing is not None and not any(existing is cause for cause in causes):
            causes.append(existing)
        combined = (
            causes[0]
            if len(causes) == 1
            else BaseExceptionGroup(
                "Recovery cleanup cancellation retained outcome-unknown owners",
                causes,
            )
        )
        set_exception_cause(cancellation, combined)

    @staticmethod
    def _validate_steps(
        steps: tuple[RecoveryCleanupStepInput, ...],
    ) -> tuple[RecoveryCleanupStep, ...]:
        if type(steps) is not tuple:
            raise TypeError("steps must be a tuple.")
        if len(steps) > _RECOVERY_CLEANUP_MAX_STEPS:
            raise ValueError(
                f"steps cannot contain more than {_RECOVERY_CLEANUP_MAX_STEPS} operations."
            )
        validated: list[RecoveryCleanupStep] = []
        for index, step in enumerate(steps):
            if type(step) is RecoveryCleanupStep:
                operation = step.operation
                cleanup = step.cleanup
                independent_with_previous = step.independent_with_previous
                if type(independent_with_previous) is not bool:
                    raise TypeError(f"steps[{index}].independent_with_previous must be a boolean.")
            elif type(step) is tuple and len(step) == 2:
                operation, cleanup = step
                independent_with_previous = False
            else:
                raise TypeError(
                    f"steps[{index}] must be an operation/cleanup tuple or RecoveryCleanupStep."
                )
            operation = _require_operation_name(operation, f"steps[{index}].operation")
            if not callable(cleanup):
                raise TypeError(f"steps[{index}].cleanup must be callable.")
            validated_cleanup = cast("RecoveryCleanup", cleanup)
            if index == 0 and independent_with_previous:
                raise ValueError("steps[0] cannot be independent with a previous step.")
            validated.append(
                RecoveryCleanupStep(
                    operation=operation,
                    cleanup=validated_cleanup,
                    independent_with_previous=independent_with_previous,
                )
            )
        return tuple(validated)

    @staticmethod
    def _group_execution_groups(
        steps: tuple[RecoveryCleanupStep, ...],
    ) -> tuple[_CleanupExecutionGroup, ...]:
        phases: list[list[RecoveryCleanupStep]] = []
        for step in steps:
            if step.independent_with_previous:
                if not phases:
                    raise AssertionError("Independent cleanup step had no preceding phase.")
                phases[-1].append(step)
            else:
                phases.append([step])
        groups: list[_CleanupExecutionGroup] = []
        sequential_steps: list[RecoveryCleanupStep] = []
        for phase in phases:
            if len(phase) == 1:
                sequential_steps.extend(phase)
                continue
            if sequential_steps:
                groups.append(
                    _CleanupExecutionGroup(
                        steps=tuple(sequential_steps),
                        independent=False,
                    )
                )
                sequential_steps.clear()
            groups.append(
                _CleanupExecutionGroup(
                    steps=tuple(phase),
                    independent=True,
                )
            )
        if sequential_steps:
            groups.append(
                _CleanupExecutionGroup(
                    steps=tuple(sequential_steps),
                    independent=False,
                )
            )
        return tuple(groups)

    def _retain_timed_out(
        self,
        task: asyncio.Task[BaseException | None],
        *,
        operation: str,
        scope: RecoveryCleanupDeadlineScope,
        timeout_seconds: float,
        request_cancellation: bool,
        cancel_after_seconds: float | None = None,
        caller_cancellation_observed: bool = False,
        continuation_barrier: _CleanupContinuationBarrier | None = None,
    ) -> RecoveryCleanupDeadlineExceeded:
        if task not in self._active_tasks:
            raise AssertionError("Recovery cleanup task had no supervision reservation.")
        self._active_tasks.remove(task)
        cancellation_handle: asyncio.TimerHandle | None = None
        if request_cancellation:
            task.cancel("recovery cleanup deadline exceeded")
        elif cancel_after_seconds is not None:
            cancellation_handle = asyncio.get_running_loop().call_later(
                cancel_after_seconds,
                task.cancel,
                "recovery cleanup step deadline exceeded after sequence return",
            )
        self._timed_out_steps += 1
        self._retain_task(
            task,
            operation=operation,
            scope=scope,
            timeout_seconds=timeout_seconds,
            caller_cancellation_observed=caller_cancellation_observed,
            cancellation_handle=cancellation_handle,
            continuation_barrier=continuation_barrier,
        )
        return RecoveryCleanupDeadlineExceeded(
            operation=operation,
            scope=scope,
            timeout_seconds=timeout_seconds,
        )

    def _retain_task(
        self,
        task: asyncio.Task[BaseException | None],
        *,
        operation: str,
        scope: RecoveryCleanupDeadlineScope,
        timeout_seconds: float,
        caller_cancellation_observed: bool,
        cancellation_handle: asyncio.TimerHandle | None = None,
        continuation_barrier: _CleanupContinuationBarrier | None = None,
    ) -> None:
        retained = _RetainedCleanup(
            operation,
            scope,
            timeout_seconds,
            caller_cancellation_observed,
            cancellation_handle,
            continuation_barrier,
        )
        if task.done():
            self._harvest_late_result(task, retained)
            return
        # Active tasks reserve their retention slot before starting. Moving one
        # reservation here must preserve the hard total task bound.
        if not self._has_task_capacity():
            raise AssertionError("Recovery cleanup retention capacity was exceeded.")
        self._retained_tasks[task] = retained
        task.add_done_callback(self._harvest_task)

    def _has_task_capacity(self) -> bool:
        return (
            len(self._active_tasks) + len(self._retained_tasks) + len(self._continuation_tasks)
            < self._policy.max_supervised_tasks
        )

    def _harvest_completed(self) -> None:
        for task in tuple(self._retained_tasks):
            if task.done():
                self._harvest_task(task)
        for task in tuple(self._continuation_tasks):
            if task.done():
                self._harvest_continuation(task)

    def _harvest_task(self, task: asyncio.Task[BaseException | None]) -> None:
        retained = self._retained_tasks.pop(task, None)
        if retained is None:
            return
        self._harvest_late_result(task, retained)

    def _harvest_late_result(
        self,
        task: asyncio.Task[BaseException | None],
        retained: _RetainedCleanup,
    ) -> None:
        if retained.cancellation_handle is not None:
            retained.cancellation_handle.cancel()
        failure = self._completed_task_failure(task)
        if failure is None:
            self._completed_after_timeout += 1
        else:
            self._failed_after_timeout += 1
        barrier = retained.continuation_barrier
        if barrier is not None:
            barrier.pending_tasks.discard(task)
            self._maybe_schedule_continuation(barrier)
        if failure is not None:
            logger.warning(
                "Retained recovery cleanup settled with failure: operation=%s error_type=%s",
                retained.operation,
                type(failure).__name__,
            )

    def _seal_continuation_barrier(self, barrier: _CleanupContinuationBarrier) -> None:
        barrier.sealed = True
        self._maybe_schedule_continuation(barrier)

    def _maybe_schedule_continuation(self, barrier: _CleanupContinuationBarrier) -> None:
        if barrier.scheduled or not barrier.sealed or barrier.pending_tasks:
            return
        if not self._has_task_capacity():
            raise AssertionError("Recovery cleanup continuation lost its reserved capacity.")
        barrier.scheduled = True
        first_operation = barrier.steps[0].operation
        task = asyncio.create_task(
            self._run_background_continuation(barrier.steps),
            name=f"cayu-recovery-cleanup-continuation:{first_operation}",
            context=barrier.context,
        )
        self._continuation_tasks.add(task)
        task.add_done_callback(self._harvest_continuation)

    async def _run_background_continuation(
        self,
        steps: tuple[RecoveryCleanupStep, ...],
    ) -> None:
        current = asyncio.current_task()
        if current is None or current not in self._continuation_tasks:
            raise RuntimeError("Recovery cleanup continuation lost supervision.")
        # Atomically exchange the continuation reservation for the first exact
        # step owner before this coroutine reaches its first suspension point.
        self._continuation_tasks.remove(current)
        try:
            failures = await self.run_steps(
                steps=steps,
                shield_caller_cancellation=True,
            )
        except BaseException as error:
            logger.warning(
                "Deferred recovery cleanup continuation failed: error_type=%s",
                type(error).__name__,
            )
            return
        for operation, failure in failures:
            logger.warning(
                "Deferred recovery cleanup step failed: operation=%s error_type=%s",
                operation,
                type(failure).__name__,
            )

    def _harvest_continuation(self, task: asyncio.Task[None]) -> None:
        self._continuation_tasks.discard(task)
        try:
            task.result()
        except asyncio.CancelledError:
            logger.warning("Deferred recovery cleanup continuation was cancelled.")
        except BaseException as error:
            logger.warning(
                "Deferred recovery cleanup continuation crashed: error_type=%s",
                type(error).__name__,
            )

    @staticmethod
    def _completed_task_failure(
        task: asyncio.Task[BaseException | None],
    ) -> BaseException | None:
        try:
            return task.result()
        except asyncio.CancelledError as cancellation:
            return cancellation

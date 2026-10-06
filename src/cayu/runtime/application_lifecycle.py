"""Process-local application shutdown: admission, one shared deadline, and outcome.

Each part is usable on its own. ``ShutdownBudget`` shares one deadline across any
sequence of bounded operations, ``ApplicationAdmission`` seals local entrances and
counts admitted work, and ``ApplicationShutdown`` composes caller-supplied steps in
order and reports a content-free ``ApplicationShutdownOutcome``. ``CayuApp.aclose``
assembles these parts around its own subsystem drains.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import time
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from contextvars import ContextVar, Token
from dataclasses import dataclass
from functools import wraps
from math import isfinite
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr, field_validator

from cayu._operation_context import current_operation_owner, operation_stack
from cayu._task_wait import capture_awaitable_outcome

DEFAULT_APPLICATION_SHUTDOWN_TIMEOUT_SECONDS = 30.0

# A step whose timeout path performs local cleanup (for example cancelling idle
# workers) still runs briefly after the shared deadline is exhausted.
_EXHAUSTED_STEP_FLOOR_SECONDS = 0.05
# A drain is given its budget; this bounds how long the composer waits for the
# drain itself to return after that budget before reporting an overrun.
_STEP_OVERRUN_GRACE_SECONDS = 0.25

ApplicationLifecycleState = Literal["open", "closing", "closed"]
ApplicationShutdownStatus = Literal["settled", "incomplete", "failed"]
ApplicationShutdownStepStatus = Literal["settled", "incomplete", "failed"]
ApplicationShutdownStepReason = Literal[
    "deadline_exhausted",
    "overran_budget",
    "late_work",
    "unowned_cancellations",
    "still_draining",
    "open_operations",
    "retained_until_settled",
]
OwnedResourcesStatus = Literal["none", "released", "retained", "failed"]

logger = logging.getLogger(__name__)

_ENTRANCE_KIND = "__cayu_entrance_kind__"
"""Attribute naming how ``_admitted_entrance``/``_tracked_entrance`` wrapped a method."""


class ApplicationAdmissionsSealed(RuntimeError):
    """A new local operation cannot start after the application began shutting down."""


class SupportsAsyncClose(Protocol):
    """A resource whose ownership was explicitly transferred to the application."""

    async def close(self) -> None: ...


class ShutdownBudget:
    """One deadline shared by a sequence of bounded shutdown operations."""

    def __init__(self, timeout_s: float) -> None:
        self.timeout_s = _validated_timeout(timeout_s)
        self._started = time.monotonic()
        self._deadline = self._started + self.timeout_s

    def remaining(self) -> float:
        """Seconds left before the shared deadline, never negative."""

        return max(0.0, self._deadline - time.monotonic())

    def expired(self) -> bool:
        return self.remaining() <= 0.0

    def elapsed(self) -> float:
        return max(0.0, time.monotonic() - self._started)

    def step_budget(self, *, floor_s: float = 0.0) -> float:
        """The remaining time, or ``floor_s`` when that is larger."""

        return max(self.remaining(), floor_s)


class ApplicationShutdownStep(BaseModel):
    """Content-free outcome of one shutdown step."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    subsystem: StrictStr
    status: ApplicationShutdownStepStatus
    budget_seconds: float = Field(ge=0)
    elapsed_seconds: float = Field(ge=0)
    reason: ApplicationShutdownStepReason | None = None
    failure_type: StrictStr | None = None

    @field_validator("subsystem")
    @classmethod
    def _non_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("subsystem must not be blank.")
        return value


class ApplicationShutdownOutcome(BaseModel):
    """Truthful, process-local result of one application shutdown attempt.

    ``settled`` means every step completed and no owned work remained. It says
    nothing about other processes sharing the same durable stores.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    status: ApplicationShutdownStatus
    attempt: StrictInt = Field(ge=1)
    timeout_seconds: float = Field(gt=0)
    elapsed_seconds: float = Field(ge=0)
    steps: tuple[ApplicationShutdownStep, ...]
    open_operations: StrictInt = Field(ge=0)
    owned_resources: OwnedResourcesStatus
    scope: Literal["process_local"] = "process_local"

    @property
    def settled(self) -> bool:
        return self.status == "settled"

    def step(self, subsystem: str) -> ApplicationShutdownStep | None:
        return next((step for step in self.steps if step.subsystem == subsystem), None)

    @property
    def unsettled(self) -> tuple[ApplicationShutdownStep, ...]:
        return tuple(step for step in self.steps if step.status != "settled")

    def summary(self) -> str:
        """One content-free line describing this attempt, for logs and terminals."""

        line = f"application shutdown {self.status} after {self.elapsed_seconds:.2f}s"
        if not self.unsettled:
            return line
        details = ", ".join(
            f"{step.subsystem}: {step.reason or step.failure_type or step.status}"
            for step in self.unsettled
        )
        return f"{line} ({details})"


# --- Admission --------------------------------------------------------------------


class _AdmissionLease:
    __slots__ = ("active", "admission", "counted", "task")

    def __init__(self, admission: ApplicationAdmission, *, counted: bool) -> None:
        self.admission = admission
        self.counted = counted
        self.task = asyncio.current_task()
        self.active = True


# Operations running in this context, admitted or tracked, innermost last.
_operation = operation_stack
# Admitted operations in this context: what lets work they start pass the seal.
_admitted: ContextVar[tuple[_AdmissionLease, ...]] = ContextVar(
    "cayu_application_admission", default=()
)


class ApplicationAdmission:
    """Seal-once gate for local entrances, counting operations still in flight.

    Each operation counts once: a call it makes in its own task is part of it.
    Work an admitted operation starts while still running, including in a
    background task that outlives it, is not refused after sealing, and that
    task is counted so shutdown waits for it. Once the operation finished, its
    leftover tasks are refused like any other new work.
    """

    def __init__(self) -> None:
        self._sealed = False
        self._closed = False
        self._in_flight = 0
        self._idle = asyncio.Event()
        self._idle.set()

    @property
    def sealed(self) -> bool:
        return self._sealed

    @property
    def in_flight(self) -> int:
        return self._in_flight

    @property
    def closed(self) -> bool:
        return self._closed

    def seal(self) -> None:
        """Refuse new top-level operations; idempotent and never reopened."""

        self._sealed = True

    def close(self) -> None:
        """Refuse every operation from now on, including tracked ones.

        The final transition before releasing what operations depend on: unlike
        ``seal``, nothing is admitted or tracked afterwards.
        """

        self._sealed = True
        self._closed = True

    def _descends_from_running_work(self) -> bool:
        # Tasks copy the context they were created in, so a background task an
        # admitted operation spawned carries that operation's lease.
        return any(lease.active and lease.admission is self for lease in _admitted.get())

    def _within_running_operation(self) -> bool:
        # A call awaited in the same task cannot outlive the operation making it.
        task = asyncio.current_task()
        return any(
            lease.active and lease.admission is self and lease.task is task
            for lease in _operation.get()
        )

    def acquire(self) -> _AdmissionLease:
        """Admit and count one operation, unless sealed and not started by admitted work."""

        self._refuse_if_closed()
        if self._sealed and not self._descends_from_running_work():
            raise ApplicationAdmissionsSealed(
                "The application is shutting down; new operations are not admitted."
            )
        return self._count()

    def track(self) -> _AdmissionLease:
        """Count one operation that stays allowed while the application drains."""

        self._refuse_if_closed()
        return self._count()

    def _refuse_if_closed(self) -> None:
        if self._closed:
            raise ApplicationAdmissionsSealed(
                "The application is closed; its resources may already be released."
            )

    def _count(self) -> _AdmissionLease:
        if self._within_running_operation():
            return _AdmissionLease(self, counted=False)
        self._in_flight += 1
        self._idle.clear()
        return _AdmissionLease(self, counted=True)

    def release(self, lease: _AdmissionLease) -> None:
        if not lease.active:
            return
        lease.active = False
        if not lease.counted:
            return
        self._in_flight -= 1
        if self._in_flight == 0:
            self._idle.set()

    async def wait_idle(self, timeout_s: float) -> bool:
        """Wait until no admitted operation remains, up to ``timeout_s``."""

        if self._in_flight == 0:
            return True
        try:
            async with asyncio.timeout(timeout_s):
                await self._idle.wait()
        except TimeoutError:
            return self._in_flight == 0
        return True


def _current_operation_admission() -> ApplicationAdmission | None:
    """The admission of the application operation this code runs in, if any.

    Identifies which application owns work started here, including from a
    background task the operation spawned.
    """

    owner = current_operation_owner()
    return owner if isinstance(owner, ApplicationAdmission) else None


def _admission_of(owner: object) -> ApplicationAdmission:
    admission = getattr(owner, "_admission", None)
    if not isinstance(admission, ApplicationAdmission):
        raise TypeError("Counted entrances require an ApplicationAdmission named _admission.")
    return admission


def _admitted_entrance(operation: Callable[..., Any]) -> Callable[..., Any]:
    """Refuse an entrance once its owner's admission is sealed, and count it.

    Apply outermost. Coroutine entrances are checked when called; streaming
    entrances when first advanced. The count is released when the operation
    finishes, including when a stream is closed or abandoned. Work it starts
    while running is admitted after sealing too, and counted.
    """

    return _counted_entrance(operation, kind="admitted")


def _tracked_entrance(operation: Callable[..., Any]) -> Callable[..., Any]:
    """Count an entrance that stays allowed while its owner drains.

    Shutdown waits for it like admitted work. It is not refused after sealing,
    only once admission closes for good, and it does not let work it starts
    bypass the seal.
    """

    return _counted_entrance(operation, kind="tracked")


def _counted_entrance(
    operation: Callable[..., Any], *, kind: Literal["admitted", "tracked"]
) -> Callable[..., Any]:
    admits = kind == "admitted"

    def acquire(owner: object) -> _AdmissionLease:
        admission = _admission_of(owner)
        return admission.acquire() if admits else admission.track()

    def mark(lease: _AdmissionLease) -> tuple[Token[tuple[_AdmissionLease, ...]], ...]:
        operation = _operation.set((*_operation.get(), lease))
        if admits:
            return (operation, _admitted.set((*_admitted.get(), lease)))
        return (operation,)

    def unmark(tokens: tuple[Token[tuple[_AdmissionLease, ...]], ...]) -> None:
        if admits:
            _admitted.reset(tokens[1])
        _operation.reset(tokens[0])

    if inspect.isasyncgenfunction(operation):

        @wraps(operation)
        async def stream(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            lease = acquire(self)
            try:
                iterator = aiter(operation(self, *args, **kwargs))
            except BaseException:
                lease.admission.release(lease)
                raise
            finally:
                # The owning entrance sanitizes rejected inputs. Do not retain a
                # second raw copy in this wrapper's traceback.
                del args, kwargs
            try:
                advance = anext(iterator)
                while True:
                    tokens = mark(lease)
                    try:
                        try:
                            event = await advance
                        except StopAsyncIteration:
                            return
                    finally:
                        unmark(tokens)
                    try:
                        yield event
                    except GeneratorExit:
                        raise
                    except BaseException as error:
                        advance = iterator.athrow(error)
                    else:
                        advance = anext(iterator)
            finally:
                try:
                    await iterator.aclose()
                finally:
                    lease.admission.release(lease)

        setattr(stream, _ENTRANCE_KIND, kind)
        return stream

    if not inspect.iscoroutinefunction(operation):
        raise TypeError("Counted entrances require a coroutine or async-generator function.")

    @wraps(operation)
    async def call(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        lease = acquire(self)
        try:
            coroutine = operation(self, *args, **kwargs)
        except BaseException:
            lease.admission.release(lease)
            raise
        finally:
            del args, kwargs
        tokens = mark(lease)
        try:
            return await coroutine
        finally:
            try:
                unmark(tokens)
            finally:
                # Resetting fails when the coroutine is closed from another
                # context; the count must still be released.
                lease.admission.release(lease)

    setattr(call, _ENTRANCE_KIND, kind)
    return call


# --- Composition ------------------------------------------------------------------


@dataclass(frozen=True)
class ShutdownStepSpec:
    """One bounded shutdown operation.

    ``run`` receives its time budget and returns True/None when settled or False
    when work remains. ``floor_protected`` steps still run after the shared
    deadline is exhausted because their timeout path performs local cleanup.
    ``seal`` synchronously refuses the subsystem's new work; it runs even when
    the deadline is exhausted and the drain itself is skipped.
    """

    subsystem: str
    run: Callable[[float], Awaitable[bool | None]]
    floor_protected: bool = False
    incomplete_reason: ApplicationShutdownStepReason = "still_draining"
    seal: Callable[[], None] | None = None


ShutdownStage = Sequence[ShutdownStepSpec]
"""Steps in one stage run concurrently; stages run in order."""


@dataclass
class _StepResult:
    step: ApplicationShutdownStep
    fatal: BaseException | None = None


def _validated_timeout(timeout_s: float) -> float:
    if (
        isinstance(timeout_s, bool)
        or not isinstance(timeout_s, int | float)
        or not isfinite(timeout_s)
        or timeout_s <= 0
    ):
        raise ValueError("timeout_s must be a finite positive number.")
    return float(timeout_s)


class ApplicationShutdown:
    """Run ordered shutdown stages under one deadline and report the outcome.

    Repeated and concurrent calls share one attempt at a time. A settled outcome
    is final; after an incomplete or failed outcome, a later call runs a new
    attempt that first rejoins steps still running from the previous one. A
    caller's cancellation propagates to that caller promptly while the attempt
    finishes within its own deadline.
    """

    def __init__(
        self,
        *,
        admission: ApplicationAdmission,
        stages: Callable[[], Sequence[ShutdownStage]],
        late_work: Callable[[], Mapping[str, ApplicationShutdownStepReason]] = dict,
        owned_resources: Iterable[SupportsAsyncClose] = (),
    ) -> None:
        self.admission = admission
        self._stages = stages
        self._late_work = late_work
        # Each resource closes once, even when it is handed over more than once.
        resources = tuple({id(resource): resource for resource in owned_resources}.values())
        for resource in resources:
            close = getattr(resource, "close", None)
            if not callable(close) or not inspect.iscoroutinefunction(close):
                raise TypeError("owned_resources must provide an async close() method.")
        self._owned_resources = resources
        self._released: set[int] = set()
        self._attempt = 0
        self._attempt_task: (
            asyncio.Task[tuple[ApplicationShutdownOutcome, BaseException | None]] | None
        ) = None
        self._overrunning: dict[str, asyncio.Task[Any]] = {}
        self._closing: dict[int, asyncio.Task[Any]] = {}
        self.outcome: ApplicationShutdownOutcome | None = None

    @property
    def state(self) -> ApplicationLifecycleState:
        if self.outcome is not None and self.outcome.settled:
            return "closed"
        return "closing" if self.admission.sealed else "open"

    async def aclose(self, *, timeout_s: float) -> ApplicationShutdownOutcome:
        timeout = _validated_timeout(timeout_s)
        self.admission.seal()
        if self.outcome is not None and self.outcome.settled:
            return self.outcome
        task = self._attempt_task
        if task is None or task.done():
            self._attempt += 1
            task = asyncio.create_task(
                self._run_attempt(ShutdownBudget(timeout), self._attempt),
                name="cayu-application-shutdown",
            )
            task.add_done_callback(_consume_task_outcome)
            self._attempt_task = task
        outcome, fatal = await asyncio.shield(task)
        if fatal is not None:
            raise fatal
        return outcome

    async def _run_attempt(
        self, budget: ShutdownBudget, attempt: int
    ) -> tuple[ApplicationShutdownOutcome, BaseException | None]:
        open_operations_step = ShutdownStepSpec(
            subsystem="open_operations",
            run=self.admission.wait_idle,
            incomplete_reason="open_operations",
        )
        results = [await self._run_step(open_operations_step, budget)]
        for stage in self._stages():
            results.extend(await asyncio.gather(*(self._run_step(spec, budget) for spec in stage)))
        # Work a step reported settled may have been started again by a later
        # step (for example a cleanup scheduling an interruption cascade).
        late = dict(self._late_work())
        open_operations = self.admission.in_flight
        if open_operations:
            late.setdefault("open_operations", "late_work")
        steps = [
            result.step.model_copy(
                update={"status": "incomplete", "reason": late[result.step.subsystem]}
            )
            if result.step.subsystem in late and result.step.status == "settled"
            else result.step
            for result in results
        ]
        fatal = next((result.fatal for result in results if result.fatal is not None), None)
        work_settled = all(step.status == "settled" for step in steps) and fatal is None
        if work_settled:
            # Close admission in the same synchronous run as the final count:
            # no operation can start between seeing none in flight and
            # releasing what operations depend on.
            self.admission.close()
        owned_status, owned_step, owned_fatal = await self._release_owned(
            budget, work_settled=work_settled
        )
        if owned_step is not None:
            steps.append(owned_step)
        fatal = fatal or owned_fatal
        if any(step.status == "failed" for step in steps):
            status: ApplicationShutdownStatus = "failed"
        elif all(step.status == "settled" for step in steps):
            status = "settled"
        else:
            status = "incomplete"
        outcome = ApplicationShutdownOutcome(
            status=status,
            attempt=attempt,
            timeout_seconds=budget.timeout_s,
            elapsed_seconds=budget.elapsed(),
            steps=tuple(steps),
            open_operations=open_operations,
            owned_resources=owned_status,
        )
        self.outcome = outcome
        return outcome, fatal

    async def _run_step(self, spec: ShutdownStepSpec, budget: ShutdownBudget) -> _StepResult:
        def step_budget_now() -> float:
            if spec.floor_protected:
                return budget.step_budget(floor_s=_EXHAUSTED_STEP_FLOOR_SECONDS)
            return budget.remaining()

        step_budget = step_budget_now()
        result = _StepReport(spec.subsystem, lambda: step_budget).build

        previous = self._overrunning.get(spec.subsystem)
        if previous is not None:
            # A drain from an earlier attempt overran; wait for it before
            # starting another so the subsystem is never drained twice at once.
            if not previous.done():
                await asyncio.wait((previous,), timeout=step_budget)
                if not previous.done():
                    return _StepResult(result("incomplete", reason="overran_budget"))
            del self._overrunning[spec.subsystem]
            late_error = previous.result().error
            if late_error is not None:
                # Report how the earlier drain ended; the next attempt drains anew.
                return self._failed(spec, result, late_error)
        step_budget = step_budget_now()
        if spec.seal is not None:
            # Sealing is part of the step: a failing seal is reported like a
            # failing drain, and later steps still run.
            try:
                spec.seal()
            except BaseException as error:
                return self._failed(spec, result, error)
        if step_budget <= 0:
            return _StepResult(result("incomplete", reason="deadline_exhausted"))
        child = asyncio.create_task(
            capture_awaitable_outcome(lambda: spec.run(step_budget)),
            name=f"cayu-application-shutdown-{spec.subsystem}",
        )
        done, _ = await asyncio.wait((child,), timeout=step_budget + _STEP_OVERRUN_GRACE_SECONDS)
        if not done:
            child.add_done_callback(_consume_task_outcome)
            self._overrunning[spec.subsystem] = child
            return _StepResult(result("incomplete", reason="overran_budget"))
        captured = child.result()
        error = captured.error
        if error is None:
            if captured.result is False:
                return _StepResult(result("incomplete", reason=spec.incomplete_reason))
            return _StepResult(result("settled"))
        return self._failed(spec, result, error)

    @staticmethod
    def _failed(
        spec: ShutdownStepSpec,
        result: Callable[..., ApplicationShutdownStep],
        error: BaseException,
    ) -> _StepResult:
        if not _is_process_control(error):
            # Content-free: an exception message can carry secrets, and this
            # log is not redacted.
            logger.warning(
                "Application shutdown step %s failed (%s).",
                spec.subsystem,
                type(error).__qualname__,
            )
            return _StepResult(result("failed", failure=error))
        # Process-control signals are recorded and re-raised after the
        # remaining steps ran.
        return _StepResult(result("failed", failure=error), fatal=error)

    async def _release_owned(
        self, budget: ShutdownBudget, *, work_settled: bool
    ) -> tuple[OwnedResourcesStatus, ApplicationShutdownStep | None, BaseException | None]:
        if not self._owned_resources:
            return "none", None, None
        # Report the budget the closes were given, not what is left afterwards.
        given = budget.step_budget(floor_s=_EXHAUSTED_STEP_FLOOR_SECONDS)
        step = _StepReport("owned_resources", lambda: given).build

        pending = [
            resource for resource in self._owned_resources if id(resource) not in self._released
        ]
        if not pending:
            return "released", step("settled"), None
        if not work_settled:
            return "retained", step("incomplete", reason="retained_until_settled"), None
        # Close in reverse order and stop at the first resource that has not
        # closed, so a resource is never closed before one that depends on it.
        for resource in reversed(pending):
            key = id(resource)
            closing = self._closing.get(key)
            if closing is None:
                closing = asyncio.create_task(
                    capture_awaitable_outcome(resource.close),
                    name="cayu-application-owned-resource-close",
                )
                closing.add_done_callback(_consume_task_outcome)
                self._closing[key] = closing
            await asyncio.wait(
                (closing,), timeout=budget.step_budget(floor_s=_EXHAUSTED_STEP_FLOOR_SECONDS)
            )
            if not closing.done():
                # Still closing: the next attempt rejoins this close.
                return "retained", step("incomplete", reason="overran_budget"), None
            del self._closing[key]
            error = closing.result().error
            if error is not None:
                # Not released: a later attempt closes it again.
                if not _is_process_control(error):
                    logger.warning(
                        "Application shutdown step owned_resources failed (%s).",
                        type(error).__qualname__,
                    )
                    return "failed", step("failed", failure=error), None
                return "failed", step("failed", failure=error), error
            self._released.add(key)
        return "released", step("settled"), None


class _StepReport:
    """Builds one step's outcome, timing it from creation."""

    def __init__(self, subsystem: str, budget_seconds: Callable[[], float]) -> None:
        self._subsystem = subsystem
        self._budget_seconds = budget_seconds
        self._started = time.monotonic()

    def build(
        self,
        status: ApplicationShutdownStepStatus,
        *,
        reason: ApplicationShutdownStepReason | None = None,
        failure: BaseException | None = None,
    ) -> ApplicationShutdownStep:
        return ApplicationShutdownStep(
            subsystem=self._subsystem,
            status=status,
            budget_seconds=self._budget_seconds(),
            elapsed_seconds=max(0.0, time.monotonic() - self._started),
            reason=reason,
            failure_type=None if failure is None else type(failure).__qualname__,
        )


def _is_process_control(error: BaseException) -> bool:
    # A drain runs in its own child task and a seal runs synchronously, so a
    # CancelledError from either is the step's own, never the caller's. A group
    # is a process-control signal only when one of its members is.
    if isinstance(error, BaseExceptionGroup):
        return any(_is_process_control(member) for member in error.exceptions)
    return not isinstance(error, Exception | asyncio.CancelledError)


def _consume_task_outcome(task: asyncio.Future[Any]) -> None:
    if not task.cancelled():
        task.exception()


__all__ = [
    "DEFAULT_APPLICATION_SHUTDOWN_TIMEOUT_SECONDS",
    "ApplicationAdmission",
    "ApplicationAdmissionsSealed",
    "ApplicationLifecycleState",
    "ApplicationShutdown",
    "ApplicationShutdownOutcome",
    "ApplicationShutdownStatus",
    "ApplicationShutdownStep",
    "ApplicationShutdownStepReason",
    "ApplicationShutdownStepStatus",
    "OwnedResourcesStatus",
    "ShutdownBudget",
    "ShutdownStage",
    "ShutdownStepSpec",
    "SupportsAsyncClose",
]

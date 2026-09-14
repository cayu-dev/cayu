"""Explicit frozen run policy carried into ordinary and resumed tool execution."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass

from cayu._exception_groups import exception_cause, set_exception_cause
from cayu._task_wait import (
    CapturedAwaitableOutcome,
    await_shielded_task_outcome,
    capture_awaitable_outcome,
    restore_task_cancellation_requests,
    unexpected_child_cancellation_error,
)
from cayu.providers import ModelRequest
from cayu.providers.base import _copy_auxiliary_request
from cayu.providers.response import ModelResponse
from cayu.runtime._run_limit_accounting import RunLimitAccountingContext
from cayu.runtime.retry_policy import RetryPolicy, copy_retry_policy
from cayu.runtime.stop_policy import RunLimits, copy_run_limits
from cayu.tools.inference import InferenceLimits, copy_inference_limits


class AuxiliaryInferenceScope:
    """One tool lifetime, with one owned logical inference operation.

    The callback remains private and runtime-bound. The handle may be awaited in
    a child task, but ending the tool lifetime expires admission and drains the
    actual callback task before releasing its owner.
    """

    def __init__(
        self,
        invoke: Callable[[ModelRequest, str, InferenceLimits], Awaitable[ModelResponse]],
    ) -> None:
        self._invoke = invoke
        self._active = False
        self._entered = False
        self._used = False
        self._observed = False
        self._task: asyncio.Task[CapturedAwaitableOutcome[ModelResponse]] | None = None

    def __copy__(self) -> AuxiliaryInferenceScope:
        return self

    def __deepcopy__(self, memo: dict[int, object]) -> AuxiliaryInferenceScope:
        # Context snapshots share this expiring capability; copying must never
        # clone admission flags or try to duplicate an owned asyncio task.
        return self

    async def invoke(
        self, request: ModelRequest, *, purpose: str, limits: InferenceLimits
    ) -> ModelResponse:
        if not self._active or self._used:
            raise RuntimeError("Auxiliary inference handle is expired or already consumed.")
        request = _copy_auxiliary_request(request)
        limits = copy_inference_limits(limits)
        self._used = True

        async def execute() -> ModelResponse:
            try:
                return await self._invoke(request, purpose, limits)
            except asyncio.CancelledError as error:
                owner = asyncio.current_task()
                # This freshly created task owns the entire callback, including
                # setup and durable preparation before a provider attempt exists.
                # A child signal alone cannot cancel its public caller.
                if owner is not None and owner.cancelling():
                    raise
                raise unexpected_child_cancellation_error(
                    error, operation="Auxiliary inference callback"
                ) from error

        # Preserve one callback outcome for both the invoke waiter and lifetime
        # owner. Reading a cancelled Task twice can produce a fresh CancelledError
        # without the original accounting/cleanup cause.
        task = asyncio.create_task(
            capture_awaitable_outcome(execute), name="cayu-auxiliary-inference"
        )
        self._task = task
        try:
            # Cancellation must signal the callback without cancelling its
            # terminal-accounting drain or losing the captured outcome.
            captured = await asyncio.shield(task)
        except asyncio.CancelledError as cancellation:
            if not task.done() and task.cancelling() == 0:
                task.cancel()
            outcome = await await_shielded_task_outcome(task)
            self._observed = True
            failure = outcome.error if outcome.result is None else outcome.result.error
            if isinstance(failure, asyncio.CancelledError):
                # The callback received our stop request. Its diagnostics are
                # secondary evidence, not a new cancellation of the tool.
                failure = exception_cause(failure)
            if failure is not None:
                _retain_cancellation_failure(cancellation, failure)
            restore_task_cancellation_requests(
                outcome.cancellation_requests_consumed, cancellation=cancellation
            )
            raise cancellation
        self._observed = True
        if captured.error is not None:
            raise captured.error
        response = captured.result
        if response is None:
            raise RuntimeError("Auxiliary inference completed without a response.")
        if not self._active:
            raise RuntimeError("Auxiliary inference completed after its tool lifetime.")
        return response

    @asynccontextmanager
    async def lifetime(self) -> AsyncIterator[None]:
        if self._entered:
            raise RuntimeError("Auxiliary inference scope cannot be reopened.")
        self._entered = True
        self._active = True
        primary: BaseException | None = None
        try:
            yield
        except BaseException as exc:
            primary = exc
        finally:
            self._active = False
        task = self._task
        secondary: BaseException | None = None
        if task is not None and not self._observed:
            abandoned = not task.done()
            # invoke explicitly forwards waiter cancellation to its callback.
            # Do not issue a second request here:
            # cancellation-resistant providers may use the first request to
            # produce authoritative terminal evidence.
            if abandoned and task.cancelling() == 0:
                task.cancel()
            outcome = await await_shielded_task_outcome(task)
            if outcome.cancellation_requests_consumed:
                restore_task_cancellation_requests(
                    outcome.cancellation_requests_consumed, cancellation=outcome.cancellation
                )
            drain_failure = outcome.error if outcome.result is None else outcome.result.error
            if abandoned and isinstance(drain_failure, asyncio.CancelledError):
                # This is our own stop request, not a second caller signal.
                # Accounting/cleanup failures carried by it remain evidence.
                drain_failure = exception_cause(drain_failure)
            if outcome.cancellation is not None:
                secondary = outcome.cancellation
                if drain_failure is not None:
                    _retain_cancellation_failure(secondary, drain_failure)
            elif drain_failure is not None:
                secondary = drain_failure
            elif abandoned and primary is None:
                secondary = RuntimeError("Tool returned with unfinished auxiliary inference.")
        if primary is not None:
            if secondary is None:
                raise primary
            if isinstance(primary, asyncio.CancelledError):
                if isinstance(secondary, asyncio.CancelledError):
                    secondary = exception_cause(secondary)
                if secondary is not None:
                    _retain_cancellation_failure(primary, secondary)
                raise primary
            if isinstance(primary, Exception) and isinstance(secondary, asyncio.CancelledError):
                prior = exception_cause(secondary)
                set_exception_cause(
                    secondary,
                    primary
                    if prior is None
                    else BaseExceptionGroup(
                        "Tool execution and auxiliary cleanup failed", [primary, prior]
                    ),
                )
                raise secondary
            raise BaseExceptionGroup(
                "Tool execution and auxiliary cleanup failed", [primary, secondary]
            )
        if secondary is not None:
            raise secondary


def _retain_cancellation_failure(
    cancellation: asyncio.CancelledError, failure: BaseException
) -> None:
    prior = exception_cause(cancellation)
    if prior is failure:
        return
    set_exception_cause(
        cancellation,
        failure
        if prior is None
        else BaseExceptionGroup("Auxiliary cancellation and cleanup failed", [prior, failure]),
    )


@dataclass(frozen=True, slots=True, init=False)
class AuxiliaryInvocationPolicy:
    """Policy data, not invocation authority or permission to dispatch.

    InvocationContext and the store-owned attempt still authenticate execution.
    Keep this explicit at tool entrances so recovery cannot accidentally use the
    application's current defaults instead of the admitted run's semantics.
    """

    _limits: RunLimits
    _retry_policy: RetryPolicy
    _accounting: RunLimitAccountingContext | None

    def __init__(
        self,
        *,
        limits: RunLimits,
        retry_policy: RetryPolicy,
        accounting: RunLimitAccountingContext | None = None,
    ) -> None:
        if type(limits) is not RunLimits or type(retry_policy) is not RetryPolicy:
            raise TypeError("Auxiliary invocation requires resolved limits and retry policy.")
        object.__setattr__(self, "_limits", copy_run_limits(limits))
        object.__setattr__(self, "_retry_policy", copy_retry_policy(retry_policy))
        object.__setattr__(self, "_accounting", _copy_accounting(accounting))

    @property
    def limits(self) -> RunLimits:
        return copy_run_limits(self._limits)

    @property
    def retry_policy(self) -> RetryPolicy:
        return copy_retry_policy(self._retry_policy)

    @property
    def accounting(self) -> RunLimitAccountingContext | None:
        return _copy_accounting(self._accounting)


def _copy_accounting(value: RunLimitAccountingContext | None) -> RunLimitAccountingContext | None:
    if value is None:
        return None
    if type(value) is not RunLimitAccountingContext:
        raise TypeError("Auxiliary invocation requires resolved run accounting.")
    return RunLimitAccountingContext(
        started_at=value.started_at,
        baseline=value.baseline,
        run_budget_authorities=value.run_budget_authorities,
    )

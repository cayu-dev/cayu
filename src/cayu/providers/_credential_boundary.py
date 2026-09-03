"""Provider exception boundaries that do not retain credential-bearing receivers."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from functools import wraps
from threading import Lock
from typing import Any, NoReturn, ParamSpec, TypeVar, cast
from weakref import WeakKeyDictionary

from cayu._exception_groups import (
    add_exception_note_safely,
    exception_group_children,
    rebuild_exception_group,
)
from cayu._exception_state import (
    exception_state,
    exception_state_contains,
    set_exception_state,
)
from cayu._validation import require_durable_nonblank
from cayu.providers.base import ModelProviderError, ModelStreamDeadlineError
from cayu.providers.deadlines import (
    DEFAULT_MAX_CONCURRENT_PROVIDER_STREAMS,
    ProviderStreamDeadlineExceeded,
)

_P = ParamSpec("_P")
_T = TypeVar("_T")
_CREDENTIAL_SAFE_CANCELLATION_STATE = "_cayu_credential_safe_provider_cancellation"
_CREDENTIAL_SAFE_CANCELLATION_TOKEN = object()
_STREAM_CLEANUP_CANCELLATION_STATE = "_cayu_provider_stream_cleanup_cancellation"
_STREAM_CLEANUP_CANCELLATION_TOKEN = object()
_STREAM_CLEANUP_CANCELLATION_NOTE = (
    "Provider stream cleanup was cancelled after a provider operation failure."
)
_PROVIDER_CANCELLATION_FAILURE_CLASSIFICATIONS = {
    "model_stream": (
        "Model provider stream failed before cancellation.",
        "ModelProviderStreamError",
    ),
    "provider_stream_cleanup": (
        "Provider stream cleanup did not complete normally.",
        "ProviderStreamCleanupError",
    ),
    "billing_identity_for_request": (
        "Provider billing identity cleanup failed during caller cancellation.",
        "BillingIdentityCleanupError",
    ),
}
_MAX_OWNED_PROVIDER_STREAM_DEADLINE_CLEANUPS = 64
_PROVIDER_STREAM_CLEANUP_REGISTRIES_LOCK = Lock()
_PROVIDER_STREAM_CLEANUP_REGISTRIES: WeakKeyDictionary[
    asyncio.AbstractEventLoop,
    set[_ProviderStreamCleanupOwnership],
] = WeakKeyDictionary()
_PROVIDER_STREAM_DEADLINE_CLEANUP_REGISTRIES: WeakKeyDictionary[
    asyncio.AbstractEventLoop,
    set[_ProviderStreamCleanupOwnership],
] = WeakKeyDictionary()


@dataclass(frozen=True, slots=True)
class _CredentialSafeCancellationHandoff:
    """Authenticated runtime-owned state for one sanitized cancellation."""

    message: str
    stream_cleanup_cancelled_after_failure: bool
    provider_cancellation_failures: tuple[dict[str, Any], ...]
    token: object


@dataclass(frozen=True, slots=True)
class _StreamCleanupCancellationHandoff:
    """Authenticated evidence that real cancellation interrupted stream cleanup."""

    token: object


@dataclass(frozen=True, slots=True)
class _ProviderStreamCleanupOutcome:
    error: BaseException | None = None


class _ProviderStreamCleanupOwnership:
    """Explicit live-model ownership for one possibly opaque stream close."""

    __slots__ = ("_loop", "_registries", "_released", "_task")

    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        registries: WeakKeyDictionary[
            asyncio.AbstractEventLoop,
            set[_ProviderStreamCleanupOwnership],
        ],
    ) -> None:
        self._loop = loop
        self._registries = registries
        self._released = False
        self._task: asyncio.Task[_ProviderStreamCleanupOutcome] | None = None

    @property
    def reserved(self) -> bool:
        return not self._released

    def track(self, task: asyncio.Task[_ProviderStreamCleanupOutcome]) -> None:
        if self._released or self._task is not None or task.get_loop() is not self._loop:
            raise RuntimeError("Provider stream cleanup ownership is invalid.")
        self._task = task
        task.add_done_callback(lambda _completed: self.release())

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        with _PROVIDER_STREAM_CLEANUP_REGISTRIES_LOCK:
            owners = self._registries.get(self._loop)
            if owners is not None:
                owners.discard(self)
                if not owners:
                    self._registries.pop(self._loop, None)


def reserve_provider_stream_cleanup(
    max_concurrent_streams: int | None = None,
) -> _ProviderStreamCleanupOwnership:
    """Reserve bounded cleanup ownership for one live model dispatch."""

    if max_concurrent_streams is None:
        max_concurrent_streams = DEFAULT_MAX_CONCURRENT_PROVIDER_STREAMS
    if type(max_concurrent_streams) is not int:
        raise TypeError("max_concurrent_streams must be an int.")
    if max_concurrent_streams < 1:
        raise ValueError("max_concurrent_streams must be >= 1.")
    loop = asyncio.get_running_loop()
    with _PROVIDER_STREAM_CLEANUP_REGISTRIES_LOCK:
        owners = _PROVIDER_STREAM_CLEANUP_REGISTRIES.setdefault(loop, set())
        if len(owners) >= max_concurrent_streams:
            raise RuntimeError("Provider stream cleanup capacity is exhausted.")
        ownership = _ProviderStreamCleanupOwnership(
            loop,
            _PROVIDER_STREAM_CLEANUP_REGISTRIES,
        )
        owners.add(ownership)
    return ownership


def _reserve_provider_stream_deadline_cleanup() -> _ProviderStreamCleanupOwnership:
    """Reserve bounded ownership for cleanup that outlives a stream deadline."""

    loop = asyncio.get_running_loop()
    with _PROVIDER_STREAM_CLEANUP_REGISTRIES_LOCK:
        owners = _PROVIDER_STREAM_DEADLINE_CLEANUP_REGISTRIES.setdefault(loop, set())
        if len(owners) >= _MAX_OWNED_PROVIDER_STREAM_DEADLINE_CLEANUPS:
            raise RuntimeError("Provider stream deadline cleanup capacity is exhausted.")
        ownership = _ProviderStreamCleanupOwnership(
            loop,
            _PROVIDER_STREAM_DEADLINE_CLEANUP_REGISTRIES,
        )
        owners.add(ownership)
    return ownership


def release_provider_stream_cleanup(
    ownership: _ProviderStreamCleanupOwnership,
) -> None:
    """Release an unused live-model cleanup reservation."""

    if type(ownership) is not _ProviderStreamCleanupOwnership:
        raise TypeError("Provider stream cleanup ownership is invalid.")
    ownership.release()


class ProviderStreamCleanupError(ModelProviderError):
    """Terminal failure proving a provider stream could not be safely closed."""


def credential_safe_provider_cancellation(
    message: str,
    *,
    preserve_empty_artifacts: bool,
    stream_cleanup_cancelled_after_failure: bool = False,
    provider_cancellation_failures: tuple[dict[str, Any], ...] = (),
) -> asyncio.CancelledError:
    """Create a cancellation whose safe projection survives the outer boundary."""

    try:
        message = require_durable_nonblank(message, "provider cancellation message")
    except (TypeError, ValueError):
        message = "Provider operation cancelled"
    cancellation = asyncio.CancelledError(message)
    failures = _copy_provider_cancellation_failures(provider_cancellation_failures)
    handoff = _CredentialSafeCancellationHandoff(
        message=message,
        stream_cleanup_cancelled_after_failure=(stream_cleanup_cancelled_after_failure is True),
        provider_cancellation_failures=failures,
        token=_CREDENTIAL_SAFE_CANCELLATION_TOKEN,
    )
    if not set_exception_state(
        cancellation,
        _CREDENTIAL_SAFE_CANCELLATION_STATE,
        handoff,
    ):
        cancellation = asyncio.CancelledError("Provider operation cancelled")
    if preserve_empty_artifacts:
        set_exception_state(cancellation, "artifacts", [])
    return cancellation


def _credential_safe_cancellation_handoff(
    failure: asyncio.CancelledError,
) -> _CredentialSafeCancellationHandoff | None:
    """Return only cancellation state authenticated by the runtime-owned handoff."""

    handoff = exception_state(failure, _CREDENTIAL_SAFE_CANCELLATION_STATE)
    if (
        type(handoff) is not _CredentialSafeCancellationHandoff
        or handoff.token is not _CREDENTIAL_SAFE_CANCELLATION_TOKEN
        or type(handoff.message) is not str
        or not handoff.message.strip()
        or type(handoff.stream_cleanup_cancelled_after_failure) is not bool
        or type(handoff.provider_cancellation_failures) is not tuple
    ):
        return None
    return handoff


def copy_provider_cancellation_failures(
    failures: object,
) -> tuple[dict[str, Any], ...]:
    """Validate reconstructed bounded provider-cancellation diagnostics."""

    if type(failures) not in {list, tuple}:
        raise ValueError("Provider cancellation failures must be a list or tuple.")
    copied_source = cast("list[object] | tuple[object, ...]", failures)
    if len(copied_source) > 2:
        raise ValueError("Provider cancellation failures must contain at most two entries.")
    copied: list[dict[str, Any]] = []
    seen_phases: set[str] = set()
    for index, failure in enumerate(copied_source):
        if type(failure) is not dict:
            raise TypeError(f"Provider cancellation failure {index} must be a dict.")
        failure = cast("dict[object, object]", failure)
        if set(failure) != {"phase", "error", "error_type"}:
            raise ValueError("Provider cancellation failure fields are invalid.")
        phase = failure.get("phase")
        error = failure.get("error")
        error_type = failure.get("error_type")
        if type(phase) is not str or phase not in _PROVIDER_CANCELLATION_FAILURE_CLASSIFICATIONS:
            raise ValueError("Provider cancellation failure phase is invalid.")
        if phase in seen_phases:
            raise ValueError("Provider cancellation failure phases must be unique.")
        expected_error, expected_error_type = _PROVIDER_CANCELLATION_FAILURE_CLASSIFICATIONS[phase]
        if type(error) is not str or error != expected_error:
            raise ValueError("Provider cancellation failure error is invalid.")
        if type(error_type) is not str or error_type != expected_error_type:
            raise ValueError("Provider cancellation failure error_type is invalid.")
        seen_phases.add(phase)
        copied.append({"phase": phase, "error": error, "error_type": error_type})
    return tuple(copied)


def _copy_provider_cancellation_failures(
    failures: tuple[dict[str, Any], ...],
) -> tuple[dict[str, Any], ...]:
    if type(failures) is not tuple:
        raise ValueError("Provider cancellation failures must be a tuple.")
    return copy_provider_cancellation_failures(failures)


def _merge_provider_cancellation_failures(
    *failure_sets: tuple[dict[str, Any], ...],
) -> tuple[dict[str, Any], ...]:
    merged: list[dict[str, Any]] = []
    seen_phases: set[str] = set()
    for failures in failure_sets:
        for failure in _copy_provider_cancellation_failures(failures):
            phase = failure["phase"]
            if phase in seen_phases:
                continue
            seen_phases.add(phase)
            merged.append(failure)
            if len(merged) == 2:
                return tuple(merged)
    return tuple(merged)


def provider_cancellation_failures(
    failure: asyncio.CancelledError,
) -> tuple[dict[str, Any], ...]:
    """Return authenticated bounded diagnostics from a provider cancellation."""

    handoff = _credential_safe_cancellation_handoff(failure)
    if handoff is None:
        return ()
    return _copy_provider_cancellation_failures(handoff.provider_cancellation_failures)


def detach_credential_safe_provider_cancellation(
    failure: asyncio.CancelledError,
) -> asyncio.CancelledError | None:
    """Return a fresh public cancellation without provider traceback state."""

    handoff = _credential_safe_cancellation_handoff(failure)
    if handoff is None:
        return None
    return asyncio.CancelledError(handoff.message)


def _mark_stream_cleanup_cancellation(failure: asyncio.CancelledError) -> None:
    """Authenticate a real task cancellation observed while closing a stream."""

    set_exception_state(
        failure,
        _STREAM_CLEANUP_CANCELLATION_STATE,
        _StreamCleanupCancellationHandoff(token=_STREAM_CLEANUP_CANCELLATION_TOKEN),
    )


def stream_cleanup_cancelled_after_provider_failure(
    failure: asyncio.CancelledError,
) -> bool:
    """Return whether Cayu observed this cancellation during provider cleanup."""

    handoff = exception_state(failure, _STREAM_CLEANUP_CANCELLATION_STATE)
    return (
        type(handoff) is _StreamCleanupCancellationHandoff
        and handoff.token is _STREAM_CLEANUP_CANCELLATION_TOKEN
    )


def _provider_stream_cleanup_error(
    operation_failure: BaseException | None = None,
) -> ProviderStreamCleanupError:
    """Build a terminal detached classification for unsafe stream cleanup."""

    message = (
        "Provider stream cleanup failed."
        if operation_failure is None
        else "Provider stream cleanup failed after a provider operation failure."
    )
    provider = "unknown"
    status_code: int | None = None
    error_type: str | None = "ProviderStreamCleanupError" if operation_failure is None else None
    error_code: str | None = None
    request_id: str | None = None
    if isinstance(operation_failure, ModelProviderError):
        try:
            candidate_provider = operation_failure.provider
            candidate_status = operation_failure.status_code
            candidate_error_type = operation_failure.error_type
            candidate_error_code = operation_failure.error_code
            candidate_request_id = operation_failure.request_id
            if type(candidate_provider) is str and candidate_provider.strip():
                provider = candidate_provider
            if type(candidate_status) is int and 100 <= candidate_status <= 599:
                status_code = candidate_status
            if type(candidate_error_type) is str:
                error_type = candidate_error_type
            if type(candidate_error_code) is str:
                error_code = candidate_error_code
            if type(candidate_request_id) is str:
                request_id = candidate_request_id
        except BaseException:
            provider = "unknown"
            status_code = None
            error_type = None
            error_code = None
            request_id = None
    try:
        return ProviderStreamCleanupError(
            message,
            provider=provider,
            status_code=status_code,
            error_type=error_type,
            error_code=error_code,
            request_id=request_id,
            retryable=False,
        )
    except BaseException:
        return ProviderStreamCleanupError(
            message,
            provider="unknown",
            retryable=False,
        )


def _raise_detached_provider_stream_cleanup_error(
    failure: ProviderStreamCleanupError,
) -> NoReturn:
    """Raise a cleanup failure without retaining Python's implicit context edge."""

    try:
        raise failure from None
    finally:
        # ``from None`` only hides implicit context from formatting; it does not
        # remove the stored reference that can retain a provider traceback.
        failure.__cause__ = None
        failure.__context__ = None


def _deadline_with_cleanup_failure(
    failure: BaseException,
) -> ProviderStreamDeadlineExceeded | ModelStreamDeadlineError | None:
    if type(failure) is ProviderStreamDeadlineExceeded:
        return ProviderStreamDeadlineExceeded(
            failure.evidence,
            stream_cleanup_failed=True,
        )
    if type(failure) is ModelStreamDeadlineError:
        return ModelStreamDeadlineError(
            provider=failure.provider,
            evidence=failure.deadline_evidence,
            stream_cleanup_failed=True,
        )
    return None


def _raise_detached_stream_deadline_failure(
    failure: ProviderStreamDeadlineExceeded | ModelStreamDeadlineError,
) -> NoReturn:
    """Raise preserved deadline evidence without either cleanup traceback."""

    try:
        raise failure from None
    finally:
        failure.__cause__ = None
        failure.__context__ = None


def _detached_cleanup_failure(failure: BaseException) -> BaseException:
    """Rebuild a provider cleanup failure without retaining provider-controlled data."""

    if isinstance(failure, BaseExceptionGroup):
        return rebuild_exception_group(
            failure,
            group_message="Provider stream cleanup failed",
            leaf_mapper=_detached_cleanup_failure,
            invalid_leaf_factory=lambda: RuntimeError("Provider stream cleanup failed"),
        )
    if isinstance(failure, asyncio.CancelledError):
        return asyncio.CancelledError("Provider stream cleanup cancelled")
    if isinstance(failure, GeneratorExit):
        return GeneratorExit("Provider stream cleanup terminated")
    if isinstance(failure, KeyboardInterrupt):
        return KeyboardInterrupt("Provider stream cleanup interrupted")
    if isinstance(failure, SystemExit):
        return SystemExit("Provider stream cleanup exited")
    if isinstance(failure, Exception):
        return RuntimeError("Provider stream cleanup failed")
    return BaseException("Provider stream cleanup failed")


def _contains_fatal_signal(failure: BaseException) -> bool:
    """Return whether a failure contains a process-level control signal."""

    pending = [failure]
    visited: set[int] = set()
    while pending:
        candidate = pending.pop()
        if id(candidate) in visited:
            continue
        visited.add(id(candidate))
        if isinstance(candidate, (GeneratorExit, KeyboardInterrupt, SystemExit)):
            return True
        if isinstance(candidate, BaseExceptionGroup):
            children = exception_group_children(candidate)
            if children is None:
                return True
            pending.extend(children)
    return False


async def close_provider_stream_after_deadline(source: AsyncIterator[object]) -> bool:
    """Start bounded retained cleanup without delaying an established deadline.

    The returned flag is true when cleanup failed, did not settle immediately,
    or could not obtain bounded retained ownership. Genuine caller cancellation
    and process-level control signals remain authoritative.
    """

    try:
        close = getattr(source, "aclose", None)
    except (GeneratorExit, KeyboardInterrupt, SystemExit):
        raise
    except BaseException:
        return True
    if not callable(close):
        return False
    try:
        ownership = _reserve_provider_stream_deadline_cleanup()
    except RuntimeError:
        return True
    close_operation = cast("Callable[[], Awaitable[None]]", close)

    async def capture_close() -> _ProviderStreamCleanupOutcome:
        try:
            await close_operation()
        except BaseException as exc:
            return _ProviderStreamCleanupOutcome(error=exc)
        return _ProviderStreamCleanupOutcome()

    try:
        cleanup_task = asyncio.create_task(capture_close())
        ownership.track(cleanup_task)
    except BaseException:
        ownership.release()
        raise
    await asyncio.sleep(0)
    if not cleanup_task.done():
        # Let a caller already awakened by cleanup startup deliver real
        # cancellation before the bounded handoff completes.
        await asyncio.sleep(0)
    if not cleanup_task.done():
        return True
    cleanup_failure = cleanup_task.result().error
    if cleanup_failure is not None and _contains_fatal_signal(cleanup_failure):
        raise cleanup_failure
    return cleanup_failure is not None


@asynccontextmanager
async def aclosing_provider_stream(
    source: AsyncIterator[object],
    *,
    cancellation_baseline: int | None = None,
    cleanup_ownership: _ProviderStreamCleanupOwnership | None = None,
) -> AsyncIterator[AsyncIterator[object]]:
    """Close a nested provider stream before propagating its outcome.

    Provider translators and transports are layered async generators. Closing
    only the outer generator does not close a suspended inner iterator, while
    ``contextlib.aclosing`` lets an ordinary close-hook failure replace the
    typed provider error that initiated cleanup. A successful close preserves
    that typed outcome. A failed close is terminal because retrying while the
    old stream may remain live is unsafe; it retains the primary typed identity
    without exposing provider-controlled cleanup details. Genuine task
    cancellation and process-level cleanup signals remain authoritative.
    """

    operation_failure: BaseException | None = None
    cleanup_failure: BaseException | None = None
    cleanup_unsettled = False
    cleanup_task_tracked = False
    closing = False
    task = asyncio.current_task()
    if cancellation_baseline is None:
        cancellation_baseline = task.cancelling() if task is not None else 0
    elif type(cancellation_baseline) is not int or cancellation_baseline < 0:
        raise ValueError("cancellation_baseline must be a non-negative int or None.")
    if cleanup_ownership is not None and (
        type(cleanup_ownership) is not _ProviderStreamCleanupOwnership
        or not cleanup_ownership.reserved
    ):
        raise TypeError("Provider stream cleanup ownership is invalid.")
    cleanup_cancellation_baseline = cancellation_baseline
    try:
        try:
            yield source
        except GeneratorExit:
            closing = True
        except BaseException as exc:
            operation_failure = exc
    finally:
        try:
            task = asyncio.current_task()
            cleanup_cancellation_baseline = task.cancelling() if task is not None else 0
            deadline_failure = (
                type(operation_failure) is ProviderStreamDeadlineExceeded
                or type(operation_failure) is ModelStreamDeadlineError
            )
            close = None
            if deadline_failure and cleanup_ownership is None:
                cleanup_unsettled = await close_provider_stream_after_deadline(source)
            else:
                close = getattr(source, "aclose", None)
            if callable(close):
                close_operation = cast("Callable[[], Awaitable[None]]", close)
                if cleanup_ownership is None:
                    await close_operation()
                else:

                    async def capture_close() -> _ProviderStreamCleanupOutcome:
                        try:
                            await close_operation()
                        except BaseException as exc:
                            return _ProviderStreamCleanupOutcome(error=exc)
                        return _ProviderStreamCleanupOutcome()

                    cleanup_task = asyncio.create_task(capture_close())
                    cleanup_ownership.track(cleanup_task)
                    cleanup_task_tracked = True
                    current_count = task.cancelling() if task is not None else 0
                    if deadline_failure or current_count > cancellation_baseline:
                        # Start the retained close before releasing the live-model
                        # caller. The child owns any later physical settlement.
                        await asyncio.sleep(0)
                        if deadline_failure and not cleanup_task.done():
                            # Let a caller awakened by cleanup startup deliver
                            # cancellation before the bounded handoff completes.
                            await asyncio.sleep(0)
                        cleanup_unsettled = not cleanup_task.done()
                        if cleanup_task.done():
                            cleanup_failure = cleanup_task.result().error
                    else:
                        try:
                            outcome = await asyncio.shield(cleanup_task)
                        except asyncio.CancelledError as exc:
                            cleanup_failure = exc
                            cleanup_unsettled = not cleanup_task.done()
                            if cleanup_task.done():
                                cleanup_failure = cleanup_task.result().error
                        else:
                            cleanup_failure = outcome.error
            elif cleanup_ownership is not None:
                cleanup_ownership.release()
        except BaseException as exc:
            cleanup_failure = exc
            if (
                cleanup_ownership is not None
                and cleanup_ownership.reserved
                and not cleanup_task_tracked
            ):
                cleanup_ownership.release()
        finally:
            close = None
            del source

    if cleanup_failure is not None and _contains_fatal_signal(cleanup_failure):
        if operation_failure is not None:
            raise cleanup_failure from operation_failure
        raise cleanup_failure

    task = asyncio.current_task()
    cancellation_count = task.cancelling() if task is not None else cancellation_baseline
    if cancellation_count > cancellation_baseline:
        cancellation_during_cleanup = cancellation_count > cleanup_cancellation_baseline
        inherited_handoff = (
            _credential_safe_cancellation_handoff(operation_failure)
            if isinstance(operation_failure, asyncio.CancelledError)
            else None
        )
        if isinstance(cleanup_failure, asyncio.CancelledError):
            cancellation = cleanup_failure
        elif isinstance(operation_failure, asyncio.CancelledError):
            cancellation = operation_failure
        else:
            cancellation = asyncio.CancelledError(
                "Provider stream cleanup cancelled"
                if cancellation_during_cleanup
                else "Provider operation cancelled"
            )
        diagnostics: list[dict[str, Any]] = []
        if operation_failure is not None and not isinstance(
            operation_failure,
            asyncio.CancelledError,
        ):
            diagnostics.append(
                {
                    "phase": "model_stream",
                    "error": "Model provider stream failed before cancellation.",
                    "error_type": "ModelProviderStreamError",
                }
            )
        if cleanup_failure is not None or cleanup_unsettled:
            diagnostics.append(
                {
                    "phase": "provider_stream_cleanup",
                    "error": "Provider stream cleanup did not complete normally.",
                    "error_type": "ProviderStreamCleanupError",
                }
            )
        cancellation = credential_safe_provider_cancellation(
            "Provider stream cleanup cancelled"
            if isinstance(cleanup_failure, asyncio.CancelledError)
            else "Provider operation cancelled",
            preserve_empty_artifacts=False,
            stream_cleanup_cancelled_after_failure=(
                operation_failure is not None and (cleanup_failure is not None or cleanup_unsettled)
            ),
            provider_cancellation_failures=_merge_provider_cancellation_failures(
                ()
                if inherited_handoff is None
                else inherited_handoff.provider_cancellation_failures,
                tuple(diagnostics),
            ),
        )
        if operation_failure is not None and (cleanup_failure is not None or cleanup_unsettled):
            _mark_stream_cleanup_cancellation(cancellation)
        if operation_failure is cancellation:
            cause = cleanup_failure
        elif operation_failure is not None:
            cause = operation_failure
        else:
            cause = cleanup_failure
        if cause is not None and cause is not cancellation:
            raise cancellation from cause
        raise cancellation

    if operation_failure is not None and _contains_fatal_signal(operation_failure):
        if cleanup_failure is not None:
            raise operation_failure from cleanup_failure
        raise operation_failure
    if (cleanup_failure is not None or cleanup_unsettled) and operation_failure is not None:
        deadline_failure = _deadline_with_cleanup_failure(operation_failure)
        if deadline_failure is not None:
            operation_failure = None
            cleanup_failure = None
            task = None
            _raise_detached_stream_deadline_failure(deadline_failure)
        terminal_failure = _provider_stream_cleanup_error(operation_failure)
        operation_failure = None
        cleanup_failure = None
        task = None
        _raise_detached_provider_stream_cleanup_error(terminal_failure)
    if operation_failure is not None:
        raise operation_failure
    if cleanup_failure is not None:
        terminal_failure = _provider_stream_cleanup_error()
        cleanup_failure = None
        task = None
        _raise_detached_provider_stream_cleanup_error(terminal_failure)
    if closing:
        return


def _detached_provider_failure(failure: BaseException) -> BaseException:
    """Detach one grouped provider failure from receiver-bearing traceback state."""

    if isinstance(failure, BaseExceptionGroup):
        return rebuild_exception_group(
            failure,
            group_message="Provider operation failed",
            leaf_mapper=_detached_provider_failure,
            invalid_leaf_factory=lambda: RuntimeError("Provider operation failed"),
        )
    if isinstance(failure, asyncio.CancelledError):
        handoff = _credential_safe_cancellation_handoff(failure)
        if handoff is None:
            cancellation = asyncio.CancelledError("Provider operation cancelled")
            if exception_state_contains(failure, "artifacts"):
                set_exception_state(cancellation, "artifacts", [])
            return cancellation
        cancellation = credential_safe_provider_cancellation(
            handoff.message,
            preserve_empty_artifacts=exception_state_contains(failure, "artifacts"),
            stream_cleanup_cancelled_after_failure=(handoff.stream_cleanup_cancelled_after_failure),
            provider_cancellation_failures=handoff.provider_cancellation_failures,
        )
        if handoff is not None and handoff.stream_cleanup_cancelled_after_failure:
            add_exception_note_safely(cancellation, _STREAM_CLEANUP_CANCELLATION_NOTE)
        return cancellation
    if isinstance(failure, GeneratorExit):
        return GeneratorExit("Provider operation terminated")
    if isinstance(failure, KeyboardInterrupt):
        return KeyboardInterrupt("Provider operation interrupted")
    if isinstance(failure, SystemExit):
        return SystemExit("Provider operation exited")
    if isinstance(failure, Exception):
        return RuntimeError("Provider operation failed")
    return BaseException("Provider operation failed")


def _clear_traceback_safely(failure: BaseException) -> BaseException:
    """Prepare a direct failure without exposing provider-controlled signals."""

    if not isinstance(failure, Exception):
        return _detached_provider_failure(failure)
    try:
        BaseException.with_traceback(failure, None)
    except BaseException:
        return _detached_provider_failure(failure)
    return failure


def detach_provider_call_traceback(
    method: Callable[_P, Awaitable[_T]],
) -> Callable[_P, Awaitable[_T]]:
    """Keep a failed async provider call from retaining its receiver or arguments."""

    @wraps(method)
    async def boundary(*args: _P.args, **kwargs: _P.kwargs) -> _T:
        failure: BaseException | None = None
        try:
            try:
                return await method(*args, **kwargs)
            except BaseExceptionGroup as exc:
                failure = _detached_provider_failure(exc)
            except BaseException as exc:
                failure = _clear_traceback_safely(exc)
        finally:
            del args, kwargs
        if failure is None:  # pragma: no cover - the try branch returns
            raise AssertionError("provider call failure was not captured")
        raise failure from None

    return boundary


def detach_provider_stream_traceback(
    method: Callable[_P, AsyncIterator[_T]],
) -> Callable[_P, AsyncIterator[_T]]:
    """Keep a failed provider stream from retaining its receiver or arguments."""

    @wraps(method)
    async def boundary(
        *args: _P.args,
        **kwargs: _P.kwargs,
    ) -> AsyncIterator[_T]:
        failure: BaseException | None = None
        cleanup_failure: BaseException | None = None
        completed = False
        closing = False
        source: AsyncIterator[_T] | None = None
        cleanup_cancellation_baseline = 0
        try:
            try:
                source = method(*args, **kwargs)
                async for item in source:
                    yield item
                completed = True
            except GeneratorExit:
                closing = True
            except BaseExceptionGroup as exc:
                failure = _detached_provider_failure(exc)
            except BaseException as exc:
                failure = _clear_traceback_safely(exc)
        finally:
            close: object | None = None
            task = asyncio.current_task()
            cleanup_cancellation_baseline = task.cancelling() if task is not None else 0
            try:
                try:
                    close = None if source is None else getattr(source, "aclose", None)
                    if callable(close):
                        await close()
                except BaseException as exc:
                    cleanup_failure = _detached_cleanup_failure(exc)
            except BaseException:
                # Sanitization is a security boundary too. It must never expose
                # an extension failure or retain the source after lookup/close.
                cleanup_failure = RuntimeError("Provider stream cleanup failed")
            finally:
                close = None
                source = None
                del args, kwargs
        if cleanup_failure is not None and _contains_fatal_signal(cleanup_failure):
            failure = None
            raise cleanup_failure from None
        if failure is not None:
            cleanup_failure = None
            raise failure from None
        if cleanup_failure is not None:
            task = asyncio.current_task()
            cancellation_count = (
                task.cancelling() if task is not None else cleanup_cancellation_baseline
            )
            if (
                isinstance(cleanup_failure, asyncio.CancelledError)
                and cancellation_count <= cleanup_cancellation_baseline
            ):
                terminal_failure = _provider_stream_cleanup_error()
                cleanup_failure = None
                task = None
                _raise_detached_provider_stream_cleanup_error(terminal_failure)
            raise cleanup_failure from None
        if closing:
            return
        if not completed:  # pragma: no cover - non-caught base exceptions propagate
            raise AssertionError("provider stream termination was not captured")

    return boundary


__all__ = [
    "ProviderStreamCleanupError",
    "aclosing_provider_stream",
    "copy_provider_cancellation_failures",
    "credential_safe_provider_cancellation",
    "detach_credential_safe_provider_cancellation",
    "detach_provider_call_traceback",
    "detach_provider_stream_traceback",
    "provider_cancellation_failures",
    "release_provider_stream_cleanup",
    "reserve_provider_stream_cleanup",
    "stream_cleanup_cancelled_after_provider_failure",
]

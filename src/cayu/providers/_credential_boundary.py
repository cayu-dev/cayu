"""Provider exception boundaries that do not retain credential-bearing receivers."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from functools import wraps
from typing import NoReturn, ParamSpec, TypeVar

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
from cayu.providers.base import ModelProviderError

_P = ParamSpec("_P")
_T = TypeVar("_T")
_CREDENTIAL_SAFE_CANCELLATION_STATE = "_cayu_credential_safe_provider_cancellation"
_CREDENTIAL_SAFE_CANCELLATION_TOKEN = object()
_STREAM_CLEANUP_CANCELLATION_STATE = "_cayu_provider_stream_cleanup_cancellation"
_STREAM_CLEANUP_CANCELLATION_TOKEN = object()
_STREAM_CLEANUP_CANCELLATION_NOTE = (
    "Provider stream cleanup was cancelled after a provider operation failure."
)


@dataclass(frozen=True, slots=True)
class _CredentialSafeCancellationHandoff:
    """Authenticated runtime-owned state for one sanitized cancellation."""

    message: str
    stream_cleanup_cancelled_after_failure: bool
    token: object


@dataclass(frozen=True, slots=True)
class _StreamCleanupCancellationHandoff:
    """Authenticated evidence that real cancellation interrupted stream cleanup."""

    token: object


class ProviderStreamCleanupError(ModelProviderError):
    """Terminal failure proving a provider stream could not be safely closed."""


def credential_safe_provider_cancellation(
    message: str,
    *,
    preserve_empty_artifacts: bool,
    stream_cleanup_cancelled_after_failure: bool = False,
) -> asyncio.CancelledError:
    """Create a cancellation whose safe projection survives the outer boundary."""

    try:
        message = require_durable_nonblank(message, "provider cancellation message")
    except (TypeError, ValueError):
        message = "Provider operation cancelled"
    cancellation = asyncio.CancelledError(message)
    handoff = _CredentialSafeCancellationHandoff(
        message=message,
        stream_cleanup_cancelled_after_failure=(stream_cleanup_cancelled_after_failure is True),
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
    ):
        return None
    return handoff


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


@asynccontextmanager
async def aclosing_provider_stream(
    source: AsyncIterator[object],
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
    closing = False
    task = asyncio.current_task()
    cancellation_baseline = task.cancelling() if task is not None else 0
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
            close = getattr(source, "aclose", None)
            if callable(close):
                await close()
        except BaseException as exc:
            cleanup_failure = exc
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
        if operation_failure is not None and (
            cancellation_during_cleanup or cleanup_failure is not None
        ):
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
    if cleanup_failure is not None and operation_failure is not None:
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
        cancellation = asyncio.CancelledError(
            handoff.message if handoff is not None else "Provider operation cancelled"
        )
        if exception_state_contains(failure, "artifacts"):
            set_exception_state(cancellation, "artifacts", [])
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
    "credential_safe_provider_cancellation",
    "detach_provider_call_traceback",
    "detach_provider_stream_traceback",
    "stream_cleanup_cancelled_after_provider_failure",
]

"""Provider exception boundaries that do not retain credential-bearing receivers."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from functools import wraps
from typing import ParamSpec, TypeVar

from cayu._exception_groups import (
    exception_group_children,
    rebuild_exception_group,
)
from cayu._exception_state import (
    exception_state,
    exception_state_contains,
    set_exception_state,
)
from cayu._validation import require_durable_nonblank

_P = ParamSpec("_P")
_T = TypeVar("_T")
_CREDENTIAL_SAFE_CANCELLATION_STATE = "_cayu_credential_safe_provider_cancellation"
_CREDENTIAL_SAFE_CANCELLATION_TOKEN = object()


@dataclass(frozen=True, slots=True)
class _CredentialSafeCancellationHandoff:
    """Authenticated runtime-owned state for one sanitized cancellation."""

    message: str
    token: object


def credential_safe_provider_cancellation(
    message: str,
    *,
    preserve_empty_artifacts: bool,
) -> asyncio.CancelledError:
    """Create a cancellation whose safe projection survives the outer boundary."""

    try:
        message = require_durable_nonblank(message, "provider cancellation message")
    except (TypeError, ValueError):
        message = "Provider operation cancelled"
    cancellation = asyncio.CancelledError(message)
    handoff = _CredentialSafeCancellationHandoff(
        message=message,
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


def _credential_safe_cancellation_message(
    failure: asyncio.CancelledError,
) -> str | None:
    """Return only a message authenticated by the runtime-owned handoff."""

    handoff = exception_state(failure, _CREDENTIAL_SAFE_CANCELLATION_STATE)
    if (
        type(handoff) is not _CredentialSafeCancellationHandoff
        or handoff.token is not _CREDENTIAL_SAFE_CANCELLATION_TOKEN
        or type(handoff.message) is not str
        or not handoff.message.strip()
    ):
        return None
    return handoff.message


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


def _contains_fatal_cleanup_signal(failure: BaseException) -> bool:
    """Return whether cleanup failed with a process-level control signal."""

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
        message = _credential_safe_cancellation_message(failure)
        cancellation = asyncio.CancelledError(message or "Provider operation cancelled")
        if exception_state_contains(failure, "artifacts"):
            set_exception_state(cancellation, "artifacts", [])
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
        if cleanup_failure is not None and _contains_fatal_cleanup_signal(cleanup_failure):
            failure = None
            raise cleanup_failure from None
        if failure is not None:
            cleanup_failure = None
            raise failure from None
        if cleanup_failure is not None:
            raise cleanup_failure from None
        if closing:
            return
        if not completed:  # pragma: no cover - non-caught base exceptions propagate
            raise AssertionError("provider stream termination was not captured")

    return boundary


__all__ = [
    "detach_provider_call_traceback",
    "detach_provider_stream_traceback",
]

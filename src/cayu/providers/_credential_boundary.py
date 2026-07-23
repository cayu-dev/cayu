"""Provider exception boundaries that do not retain credential-bearing receivers."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from functools import wraps
from typing import ParamSpec, TypeVar

_P = ParamSpec("_P")
_T = TypeVar("_T")


def _detached_cleanup_failure(failure: BaseException) -> BaseException:
    """Rebuild a provider cleanup failure without retaining provider-controlled data."""

    if isinstance(failure, BaseExceptionGroup):
        children = [_detached_cleanup_failure(child) for child in failure.exceptions]
        del failure
        return BaseExceptionGroup("Provider stream cleanup failed", children)
    if isinstance(failure, asyncio.CancelledError):
        del failure
        return asyncio.CancelledError("Provider stream cleanup cancelled")
    if isinstance(failure, GeneratorExit):
        del failure
        return GeneratorExit("Provider stream cleanup terminated")
    if isinstance(failure, KeyboardInterrupt):
        del failure
        return KeyboardInterrupt("Provider stream cleanup interrupted")
    if isinstance(failure, SystemExit):
        del failure
        return SystemExit("Provider stream cleanup exited")
    if isinstance(failure, Exception):
        del failure
        return RuntimeError("Provider stream cleanup failed")
    del failure
    return BaseException("Provider stream cleanup failed")


def _contains_fatal_cleanup_signal(failure: BaseException) -> bool:
    """Return whether cleanup failed with a process-level control signal."""

    if isinstance(failure, (GeneratorExit, KeyboardInterrupt, SystemExit)):
        return True
    if isinstance(failure, BaseExceptionGroup):
        return any(_contains_fatal_cleanup_signal(child) for child in failure.exceptions)
    return False


def detach_provider_call_traceback(
    method: Callable[_P, Awaitable[_T]],
) -> Callable[_P, Awaitable[_T]]:
    """Keep a failed async provider call from retaining its receiver or arguments."""

    @wraps(method)
    async def boundary(*args: _P.args, **kwargs: _P.kwargs) -> _T:
        failure: BaseException | None = None
        try:
            return await method(*args, **kwargs)
        except asyncio.CancelledError as exc:
            failure = exc.with_traceback(None)
        except Exception as exc:
            failure = exc.with_traceback(None)
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
        source: AsyncIterator[_T] | None = method(*args, **kwargs)
        try:
            async for item in source:
                yield item
            completed = True
        except GeneratorExit:
            closing = True
        except asyncio.CancelledError as exc:
            failure = exc.with_traceback(None)
        except Exception as exc:
            failure = exc.with_traceback(None)
        finally:
            close = getattr(source, "aclose", None)
            try:
                if callable(close):
                    await close()
            except BaseException as exc:
                cleanup_failure = _detached_cleanup_failure(exc)
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

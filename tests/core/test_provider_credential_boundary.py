from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import pytest
from tests.provider_traceback_assertions import assert_cayu_traceback_does_not_retain

from cayu.providers._credential_boundary import detach_provider_stream_traceback


class _FailingCloseStream:
    def __init__(self) -> None:
        self._yielded = False

    def __aiter__(self) -> _FailingCloseStream:
        return self

    async def __anext__(self) -> str:
        if not self._yielded:
            self._yielded = True
            return "started"
        raise RuntimeError("authoritative provider failure")

    async def aclose(self) -> None:
        raise RuntimeError("provider cleanup failure")


class _CancellationGroupCloseStream(_FailingCloseStream):
    async def aclose(self) -> None:
        raise BaseExceptionGroup(
            "provider cleanup cancelled",
            [asyncio.CancelledError()],
        )


class _GroupedCloseStream(_FailingCloseStream):
    def __init__(self, failure: BaseExceptionGroup) -> None:
        super().__init__()
        self._failure = failure

    async def aclose(self) -> None:
        raise self._failure


class _ProviderWithGroupedCleanup:
    def __init__(self, failure: BaseExceptionGroup) -> None:
        self._failure = failure

    @detach_provider_stream_traceback
    def stream(self) -> AsyncIterator[str]:
        return _GroupedCloseStream(self._failure)


def _exception_texts(failure: BaseException) -> list[str]:
    if isinstance(failure, BaseExceptionGroup):
        return [
            str(failure),
            *[text for child in failure.exceptions for text in _exception_texts(child)],
        ]
    return [str(failure)]


def _assert_detached_exception_tree(failure: BaseException) -> None:
    assert failure.__cause__ is None
    assert failure.__context__ is None
    assert getattr(failure, "__notes__", ()) == ()
    if isinstance(failure, BaseExceptionGroup):
        for child in failure.exceptions:
            assert child.__traceback__ is None
            _assert_detached_exception_tree(child)


@pytest.mark.anyio
async def test_detached_provider_stream_closes_its_source_on_early_close() -> None:
    source_closed = asyncio.Event()

    @detach_provider_stream_traceback
    async def stream() -> AsyncIterator[str]:
        try:
            yield "started"
            await asyncio.Event().wait()
        finally:
            source_closed.set()

    events = stream()
    assert await anext(events) == "started"

    await events.aclose()

    assert source_closed.is_set()


@pytest.mark.anyio
async def test_detached_provider_stream_does_not_mask_provider_failure_during_close() -> None:
    @detach_provider_stream_traceback
    def stream() -> AsyncIterator[str]:
        return _FailingCloseStream()

    events = stream()
    assert await anext(events) == "started"

    with pytest.raises(RuntimeError, match="authoritative provider failure"):
        await anext(events)


@pytest.mark.anyio
async def test_detached_provider_stream_reports_early_close_failure() -> None:
    @detach_provider_stream_traceback
    def stream() -> AsyncIterator[str]:
        return _FailingCloseStream()

    events = stream()
    assert await anext(events) == "started"

    with pytest.raises(RuntimeError, match="Provider stream cleanup failed"):
        await events.aclose()


@pytest.mark.anyio
async def test_detached_provider_stream_keeps_primary_over_cleanup_cancellation_group() -> None:
    @detach_provider_stream_traceback
    def stream() -> AsyncIterator[str]:
        return _CancellationGroupCloseStream()

    events = stream()
    assert await anext(events) == "started"

    with pytest.raises(RuntimeError, match="authoritative provider failure"):
        await anext(events)


@pytest.mark.anyio
async def test_detached_provider_stream_scrubs_fatal_cleanup_group_after_clearing_receiver() -> (
    None
):
    credential = "provider-secret-cleanup-canary"
    provider = _ProviderWithGroupedCleanup(
        BaseExceptionGroup(
            f"raw cleanup group {credential}",
            [KeyboardInterrupt(f"raw fatal child {credential}")],
        )
    )
    events = provider.stream()
    assert await anext(events) == "started"

    with pytest.raises(BaseExceptionGroup) as exc_info:
        await events.aclose()

    assert all(credential not in text for text in _exception_texts(exc_info.value))
    _assert_detached_exception_tree(exc_info.value)
    assert_cayu_traceback_does_not_retain(exc_info.value, provider)


@pytest.mark.anyio
async def test_detached_provider_stream_rebuilds_nonfatal_cleanup_group() -> None:
    credential = "provider-secret-cleanup-canary"
    provider = _ProviderWithGroupedCleanup(
        ExceptionGroup(
            f"raw cleanup group {credential}",
            [RuntimeError(f"raw cleanup child {credential}")],
        )
    )
    events = provider.stream()
    assert await anext(events) == "started"

    with pytest.raises(ExceptionGroup) as exc_info:
        await events.aclose()

    assert all(credential not in text for text in _exception_texts(exc_info.value))
    _assert_detached_exception_tree(exc_info.value)
    assert_cayu_traceback_does_not_retain(exc_info.value, provider)

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import pytest
from tests.provider_traceback_assertions import assert_cayu_traceback_does_not_retain

from cayu._exception_groups import iter_exception_tree
from cayu.providers._credential_boundary import (
    ProviderStreamCleanupError,
    aclosing_provider_stream,
    detach_provider_call_traceback,
    detach_provider_stream_traceback,
)
from cayu.providers._http import sanitize_provider_cancellation
from cayu.providers.base import ModelProviderError


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


class _CancelledCloseStream(_FailingCloseStream):
    async def aclose(self) -> None:
        raise asyncio.CancelledError("provider child cleanup cancelled")


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


def _deep_group(leaf: BaseException, *, depth: int = 1_500) -> BaseExceptionGroup:
    failure: BaseException = leaf
    for _ in range(depth):
        failure = BaseExceptionGroup("provider failure", [failure])
    assert isinstance(failure, BaseExceptionGroup)
    return failure


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
async def test_aclosing_provider_stream_fails_closed_and_retains_primary_identity() -> None:
    source = _FailingCloseStream()
    primary = ModelProviderError(
        "authoritative provider failure",
        provider="test-provider",
        status_code=500,
        error_type="server_error",
        error_code="internal_error",
        retryable=True,
    )

    with pytest.raises(ProviderStreamCleanupError) as exc_info:
        async with aclosing_provider_stream(source):
            raise primary

    assert exc_info.value is not primary
    assert exc_info.value.provider == "test-provider"
    assert exc_info.value.status_code == 500
    assert exc_info.value.error_type == "server_error"
    assert exc_info.value.error_code == "internal_error"
    assert exc_info.value.retryable is False
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None
    assert_cayu_traceback_does_not_retain(exc_info.value, primary)
    assert_cayu_traceback_does_not_retain(exc_info.value, source)


@pytest.mark.anyio
@pytest.mark.parametrize(
    "primary",
    [
        pytest.param(KeyboardInterrupt("primary interrupt"), id="keyboard-interrupt"),
        pytest.param(SystemExit("primary exit"), id="system-exit"),
        pytest.param(
            BaseExceptionGroup(
                "primary fatal group",
                [RuntimeError("ordinary child"), KeyboardInterrupt("fatal child")],
            ),
            id="fatal-group",
        ),
    ],
)
async def test_aclosing_provider_stream_preserves_primary_fatal_signal(
    primary: BaseException,
) -> None:
    source = _FailingCloseStream()

    with pytest.raises(type(primary)) as exc_info:
        async with aclosing_provider_stream(source):
            raise primary

    assert exc_info.value is primary
    assert isinstance(exc_info.value.__cause__, RuntimeError)
    assert str(exc_info.value.__cause__) == "provider cleanup failure"


@pytest.mark.anyio
async def test_aclosing_provider_stream_preserves_new_task_cancellation() -> None:
    close_started = asyncio.Event()

    class BlockingCloseStream(_FailingCloseStream):
        async def aclose(self) -> None:
            close_started.set()
            await asyncio.Event().wait()

    primary = RuntimeError("authoritative provider failure")

    async def consume() -> None:
        async with aclosing_provider_stream(BlockingCloseStream()):
            raise primary

    task = asyncio.create_task(consume())
    await close_started.wait()
    task.cancel()
    assert task.cancelling() == 1

    cancellation: asyncio.CancelledError | None = None
    try:
        await task
    except asyncio.CancelledError as exc:
        cancellation = exc

    assert cancellation is not None
    assert cancellation.__cause__ is primary
    assert task.cancelled()


@pytest.mark.anyio
async def test_aclosing_provider_stream_reclassifies_cleanup_only_child_cancellation() -> None:
    task = asyncio.current_task()
    assert task is not None
    cancellation_baseline = task.cancelling()

    with pytest.raises(ProviderStreamCleanupError) as exc_info:
        async with aclosing_provider_stream(_CancelledCloseStream()):
            pass

    assert exc_info.value.provider == "unknown"
    assert exc_info.value.retryable is False
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None
    assert task.cancelling() == cancellation_baseline
    assert not task.cancelled()


@pytest.mark.anyio
async def test_aclosing_provider_stream_reclassifies_cleanup_only_failure() -> None:
    source = _FailingCloseStream()

    with pytest.raises(ProviderStreamCleanupError) as exc_info:
        async with aclosing_provider_stream(source):
            pass

    assert str(exc_info.value) == "Provider stream cleanup failed."
    assert exc_info.value.provider == "unknown"
    assert exc_info.value.error_type == "ProviderStreamCleanupError"
    assert exc_info.value.retryable is False
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None
    assert_cayu_traceback_does_not_retain(exc_info.value, source)


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
async def test_detached_provider_stream_reclassifies_cleanup_only_child_cancellation() -> None:
    @detach_provider_stream_traceback
    def stream() -> AsyncIterator[str]:
        return _CancelledCloseStream()

    events = stream()
    assert await anext(events) == "started"
    task = asyncio.current_task()
    assert task is not None
    cancellation_baseline = task.cancelling()

    with pytest.raises(ProviderStreamCleanupError) as exc_info:
        await events.aclose()

    assert exc_info.value.provider == "unknown"
    assert exc_info.value.retryable is False
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None
    assert task.cancelling() == cancellation_baseline
    assert not task.cancelled()


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


@pytest.mark.anyio
async def test_detached_provider_stream_bypasses_hostile_group_accessors() -> None:
    credential = "provider-secret-group-accessor-canary"

    class HostileCleanupGroup(ExceptionGroup):
        def __getattribute__(self, name: str):
            if name in {"exceptions", "__cause__", "__context__"}:
                raise RuntimeError(f"{credential}-{name}")
            return super().__getattribute__(name)

        def subgroup(self, _condition):
            raise RuntimeError(f"{credential}-subgroup")

        def split(self, _condition):
            raise RuntimeError(f"{credential}-split")

    provider = _ProviderWithGroupedCleanup(
        HostileCleanupGroup(
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


@pytest.mark.anyio
async def test_detached_provider_call_rebuilds_mixed_exception_group() -> None:
    credential = "provider-secret-call-group-canary"

    class Provider:
        @detach_provider_call_traceback
        async def call(self) -> None:
            raise BaseExceptionGroup(
                f"provider call failed near {credential}",
                [
                    asyncio.CancelledError(f"cancelled near {credential}"),
                    RuntimeError(f"request failed near {credential}"),
                ],
            )

    provider = Provider()
    with pytest.raises(BaseExceptionGroup) as exc_info:
        await provider.call()

    assert credential not in repr(exc_info.value)
    assert [type(candidate) for candidate in iter_exception_tree(exc_info.value)] == [
        BaseExceptionGroup,
        asyncio.CancelledError,
        RuntimeError,
    ]
    _assert_detached_exception_tree(exc_info.value)
    assert_cayu_traceback_does_not_retain(exc_info.value, provider)


@pytest.mark.anyio
async def test_detached_provider_call_preserves_fatal_child_hidden_by_descriptor() -> None:
    credential = "provider-secret-group-descriptor-canary"

    class HostileFatalGroup(BaseExceptionGroup):
        @property
        def exceptions(self) -> tuple[BaseException, ...]:
            raise RuntimeError(credential)

    class Provider:
        @detach_provider_call_traceback
        async def call(self) -> None:
            raise HostileFatalGroup(
                f"provider call failed near {credential}",
                [KeyboardInterrupt(f"interrupted near {credential}")],
            )

    provider = Provider()
    with pytest.raises(BaseExceptionGroup) as exc_info:
        await provider.call()

    assert [type(candidate) for candidate in iter_exception_tree(exc_info.value)] == [
        BaseExceptionGroup,
        KeyboardInterrupt,
    ]
    assert credential not in repr(exc_info.value)
    _assert_detached_exception_tree(exc_info.value)
    assert_cayu_traceback_does_not_retain(exc_info.value, provider)


@pytest.mark.anyio
async def test_detached_provider_call_bypasses_hostile_with_traceback() -> None:
    credential = "provider-secret-with-traceback-canary"

    class HostileFailure(RuntimeError):
        def with_traceback(self, _traceback):
            raise RuntimeError(credential)

    class Provider:
        @detach_provider_call_traceback
        async def call(self) -> None:
            raise HostileFailure("safe provider failure")

    provider = Provider()
    with pytest.raises(HostileFailure, match="safe provider failure") as exc_info:
        await provider.call()

    assert credential not in repr(exc_info.value)
    assert_cayu_traceback_does_not_retain(exc_info.value, provider)


@pytest.mark.anyio
async def test_detached_provider_call_scrubs_direct_fatal_failure() -> None:
    credential = "provider-secret-direct-fatal-canary"

    class Provider:
        @detach_provider_call_traceback
        async def call(self) -> None:
            failure = KeyboardInterrupt(credential)
            failure.authorization = credential
            raise failure

    provider = Provider()
    with pytest.raises(KeyboardInterrupt, match="Provider operation interrupted") as exc_info:
        await provider.call()

    assert type(exc_info.value) is KeyboardInterrupt
    assert credential not in repr(exc_info.value)
    assert vars(exc_info.value) == {}
    _assert_detached_exception_tree(exc_info.value)
    assert_cayu_traceback_does_not_retain(exc_info.value, provider)


@pytest.mark.anyio
async def test_detached_provider_call_preserves_authenticated_cancellation_message() -> None:
    credential = "provider-secret-sanitized-cancellation-canary"
    safe_message = "OpenAI provider request cancelled"

    class Provider:
        @detach_provider_call_traceback
        async def call(self) -> None:
            raise sanitize_provider_cancellation(
                asyncio.CancelledError(credential),
                provider_label="OpenAI",
                credential_values=(credential,),
                safe_message=safe_message,
            )

    provider = Provider()
    with pytest.raises(asyncio.CancelledError, match=safe_message) as exc_info:
        await provider.call()

    assert type(exc_info.value) is asyncio.CancelledError
    assert exc_info.value.args == (safe_message,)
    assert credential not in repr(exc_info.value)
    assert vars(exc_info.value) == {}
    _assert_detached_exception_tree(exc_info.value)
    assert_cayu_traceback_does_not_retain(exc_info.value, provider)


@pytest.mark.anyio
async def test_detached_provider_stream_scrubs_direct_cancellation() -> None:
    credential = "provider-secret-direct-cancellation-canary"

    class Provider:
        @detach_provider_stream_traceback
        async def stream(self) -> AsyncIterator[str]:
            failure = asyncio.CancelledError(credential)
            failure.authorization = credential
            failure.artifacts = [{"credential": credential}]
            failure.__dict__["_cayu_credential_safe_provider_cancellation"] = {
                "message": credential,
            }
            raise failure
            yield "unreachable"

    provider = Provider()
    with pytest.raises(asyncio.CancelledError, match="Provider operation cancelled") as exc_info:
        async for _ in provider.stream():
            pass

    assert type(exc_info.value) is asyncio.CancelledError
    assert credential not in repr(exc_info.value)
    assert vars(exc_info.value) == {"artifacts": []}
    _assert_detached_exception_tree(exc_info.value)
    assert_cayu_traceback_does_not_retain(exc_info.value, provider)


@pytest.mark.anyio
async def test_detached_provider_stream_contains_hostile_close_lookup() -> None:
    credential = "provider-secret-close-lookup-canary"

    class HostileCloseLookupStream:
        def __init__(self, provider: object) -> None:
            self.provider = provider

        def __aiter__(self):
            return self

        async def __anext__(self) -> str:
            raise StopAsyncIteration

        def __getattribute__(self, name: str):
            if name == "aclose":
                raise RuntimeError(credential)
            return super().__getattribute__(name)

    class Provider:
        @detach_provider_stream_traceback
        def stream(self) -> AsyncIterator[str]:
            return HostileCloseLookupStream(self)

    provider = Provider()
    with pytest.raises(RuntimeError, match="Provider stream cleanup failed") as exc_info:
        async for _ in provider.stream():
            pass

    assert credential not in repr(exc_info.value)
    assert_cayu_traceback_does_not_retain(exc_info.value, provider)


@pytest.mark.anyio
async def test_detached_provider_stream_rebuilds_deep_cleanup_group_iteratively() -> None:
    provider = _ProviderWithGroupedCleanup(_deep_group(RuntimeError("provider cleanup failed")))
    events = provider.stream()
    assert await anext(events) == "started"

    with pytest.raises(ExceptionGroup) as exc_info:
        await events.aclose()

    assert sum(1 for _ in iter_exception_tree(exc_info.value)) == 1_501
    assert_cayu_traceback_does_not_retain(exc_info.value, provider)

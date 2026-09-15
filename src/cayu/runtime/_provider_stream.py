"""Shared admitted provider stream and cancellation/cleanup ownership boundary."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable
from typing import Any, Never, cast

from cayu._exception_groups import exception_tree_contains, rebuild_exception_group
from cayu.deadlines import ExecutionDeadlineExceeded, current_execution_deadline
from cayu.providers import (
    ModelProvider,
    ModelProviderError,
    ModelRequest,
    ModelStreamEvent,
    ModelStreamEventType,
)
from cayu.providers._credential_boundary import (
    _ProviderStreamCleanupOwnership,
    aclosing_provider_stream,
    credential_safe_provider_cancellation,
    native_model_admission_deadline,
    provider_cancellation_admission_deadline,
    provider_cancellation_failures,
    release_provider_stream_cleanup,
    reserve_provider_stream_cleanup,
    retain_native_model_admission_expiry,
    stream_cleanup_cancelled_after_provider_failure,
)
from cayu.providers._stream_cleanup import (
    _local_http_cleanup_observer,
    _LocalHttpCleanupObserver,
)
from cayu.providers.deadlines import (
    ProviderStreamDeadlineAdmission,
    bind_provider_deadline_admission,
    reset_provider_deadline_admission,
)


class _ProviderStreamSelfCancellation(asyncio.CancelledError):
    """Authenticated handoff for provider-originated stream cancellation."""


def _provider_stream_self_cancellation_error(provider_name: str) -> ModelProviderError:
    """Project provider-only cancellation without granting caller control authority."""

    return ModelProviderError(
        "Model provider stream cancelled itself.",
        provider=provider_name,
        error_type="ProviderStreamCancellationError",
        error_code="provider_stream_cancelled_itself",
        retryable=False,
    )


class _ProviderStreamGeneratorExit(BaseException):
    """Carry a provider-raised GeneratorExit through context-manager closure."""

    def __init__(self, failure: GeneratorExit) -> None:
        super().__init__()
        self.failure = failure


def _sanitize_provider_cancellation_group(failure: BaseException) -> BaseException:
    """Detach a provider-owned group whose cancellation has no caller authority."""

    if isinstance(failure, BaseExceptionGroup):
        return rebuild_exception_group(
            failure,
            group_message="Model provider stream failed",
            leaf_mapper=_sanitize_provider_cancellation_group,
            invalid_leaf_factory=lambda: RuntimeError("Model provider stream failed"),
        )
    if isinstance(failure, asyncio.CancelledError):
        return RuntimeError("Model provider stream cancelled itself")
    if isinstance(failure, GeneratorExit):
        return GeneratorExit("Model provider stream terminated")
    if isinstance(failure, KeyboardInterrupt):
        return KeyboardInterrupt("Model provider stream interrupted")
    if isinstance(failure, SystemExit):
        raw_args: object
        try:
            raw_args = BaseException.__dict__["args"].__get__(failure, BaseException)
        except BaseException:
            raw_args = ()
        if type(raw_args) is tuple and len(raw_args) == 1:
            exit_code = cast("tuple[object]", raw_args)[0]
            if type(exit_code) is int or exit_code is None:
                return SystemExit(exit_code)
        return SystemExit("Model provider stream exited")
    if isinstance(failure, Exception):
        return RuntimeError("Model provider stream failed")
    return BaseException("Model provider stream failed")


def _raise_model_provider_stream_boundary_failure(
    failure: BaseException,
    *,
    cancellation_baseline: int,
) -> Never:
    task = asyncio.current_task()
    native_expiry = native_model_admission_deadline(failure)
    if task is not None and task.cancelling() > cancellation_baseline:
        inherited_diagnostics = (
            provider_cancellation_failures(failure)
            if isinstance(failure, asyncio.CancelledError)
            else ()
        )
        diagnostics = inherited_diagnostics or (
            ()
            if isinstance(failure, asyncio.CancelledError) or native_expiry is not None
            else (
                {
                    "phase": "model_stream",
                    "error": "Model provider stream failed before cancellation.",
                    "error_type": "ModelProviderStreamError",
                },
            )
        )
        raise credential_safe_provider_cancellation(
            "Provider operation cancelled",
            preserve_empty_artifacts=False,
            stream_cleanup_cancelled_after_failure=(
                isinstance(failure, asyncio.CancelledError)
                and stream_cleanup_cancelled_after_provider_failure(failure)
            ),
            provider_cancellation_failures=diagnostics,
            native_admission_deadline=(
                native_expiry
                if native_expiry is not None
                else provider_cancellation_admission_deadline(failure)
            ),
        ) from None
    if native_expiry is not None:
        raise failure
    if isinstance(failure, asyncio.CancelledError):
        raise _ProviderStreamSelfCancellation() from None
    if isinstance(failure, BaseExceptionGroup) and exception_tree_contains(
        failure,
        asyncio.CancelledError,
    ):
        raise _sanitize_provider_cancellation_group(failure) from None
    raise failure


async def _admitted_model_provider_events(
    provider: ModelProvider,
    request: ModelRequest,
    admission: ProviderStreamDeadlineAdmission,
    refresh_live_model_semantics: Callable[[], Awaitable[None]],
    cleanup_observer: _LocalHttpCleanupObserver | None = None,
) -> AsyncGenerator[ModelStreamEvent, None]:
    """Transfer one pre-dispatch deadline admission into the provider stream."""

    await refresh_live_model_semantics()
    deadline = current_execution_deadline()
    try:
        deadline.require_admission("model")
    except ExecutionDeadlineExceeded as failure:
        # Only the runtime's pre-dispatch check proves zero-dispatch expiry.
        retain_native_model_admission_expiry(failure, deadline)
        raise
    events = provider.runtime_stream(request)
    iterator = aiter(events)
    try:
        while True:
            token = bind_provider_deadline_admission(admission)
            cleanup_token = _local_http_cleanup_observer.set(cleanup_observer)
            try:
                try:
                    event = await anext(iterator)
                except StopAsyncIteration:
                    return
            finally:
                reset_provider_deadline_admission(token)
                _local_http_cleanup_observer.reset(cleanup_token)
            yield event
    finally:
        # The outer aclosing_provider_stream owns classification, redaction and
        # retained cleanup. Do not suppress an inner close failure here: doing
        # so would present failed cleanup as settled and allow another dispatch.
        close = getattr(iterator, "aclose", None)
        if close is not None:
            await close()


async def _owned_model_provider_events(
    source_factory: Callable[[], AsyncIterator[ModelStreamEvent]],
    *,
    cancellation_baseline: int,
    max_concurrent_streams: int,
    cleanup_ownership: _ProviderStreamCleanupOwnership | None = None,
) -> AsyncGenerator[ModelStreamEvent, None]:
    """Authenticate cancellation at the live model-stream boundary."""

    if cleanup_ownership is None:
        cleanup_ownership = reserve_provider_stream_cleanup(max_concurrent_streams)
    try:
        source = source_factory()
    except BaseException as failure:
        release_provider_stream_cleanup(cleanup_ownership)
        _raise_model_provider_stream_boundary_failure(
            failure,
            cancellation_baseline=cancellation_baseline,
        )
    try:
        async with aclosing_provider_stream(
            source,
            cancellation_baseline=cancellation_baseline,
            cleanup_ownership=cleanup_ownership,
        ) as events:
            try:
                iterator = aiter(events)
            except BaseException as failure:
                _raise_model_provider_stream_boundary_failure(
                    failure,
                    cancellation_baseline=cancellation_baseline,
                )
            while True:
                try:
                    event = cast("ModelStreamEvent", await anext(iterator))
                except StopAsyncIteration:
                    return
                except GeneratorExit as failure:
                    raise _ProviderStreamGeneratorExit(failure) from None
                except BaseException as failure:
                    _raise_model_provider_stream_boundary_failure(
                        failure,
                        cancellation_baseline=cancellation_baseline,
                    )
                task = asyncio.current_task()
                if (
                    task is not None
                    and task.cancelling() > cancellation_baseline
                    and event.type is not ModelStreamEventType.COMPLETED
                ):
                    # A provider may catch the injected CancelledError and return a
                    # value instead. Task state remains the positive caller-owned
                    # authority, so do not let nonterminal output resume ordinary
                    # execution. A validated completion is different: runtime must
                    # durably publish terminal provider evidence that won the race,
                    # then restore the same caller cancellation.
                    raise credential_safe_provider_cancellation(
                        "Provider operation cancelled",
                        preserve_empty_artifacts=False,
                    )
                yield event
    except _ProviderStreamGeneratorExit as failure:
        raise failure.failure from None


async def _close_async_iterator(iterator: AsyncIterator[Any]) -> None:
    # Iterator disposal runs while a more authoritative provider, budget,
    # cancellation, or GeneratorExit outcome is already propagating. Attribute
    # lookup is provider-controlled too, so it belongs inside the same boundary
    # as invoking the close hook.
    try:
        close = getattr(iterator, "aclose", None)
        if close is not None:
            await close()
    except ExceptionGroup:
        # Ordinary cleanup failures remain secondary to the outcome that
        # caused iterator disposal.
        pass
    except BaseExceptionGroup:
        # A mixed group also carries a fatal signal such as cancellation.
        # Preserve the complete group for the caller to classify.
        raise
    except Exception:
        pass

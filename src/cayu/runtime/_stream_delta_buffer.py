"""Bound display-only provider deltas before the durable Runtime event boundary."""

from __future__ import annotations

import sys
from collections.abc import AsyncGenerator, Awaitable, Callable
from dataclasses import dataclass
from time import monotonic

from cayu.providers.base import ModelStreamEvent, ModelStreamEventType, copy_model_stream_event

_MAX_DELTA_CHARS = 4096
_MAX_DELTA_DELAY_SECONDS = 1.0


@dataclass(frozen=True, slots=True)
class BufferedModelCompletion:
    """Hand terminal evidence to the attempt before it publishes the delta tail."""

    delta: ModelStreamEvent
    completion: ModelStreamEvent


def _is_plain_delta(event: object) -> bool:
    # Never merge signed blocks, opaque state, or provider recovery cursors. Those
    # boundaries carry semantics beyond the concatenated display text.
    return (
        type(event) is ModelStreamEvent
        and event.type in (ModelStreamEventType.TEXT_DELTA, ModelStreamEventType.THINKING)
        and type(event.delta) is str
        and bool(event.delta)
        and type(event.payload) is dict
        and not event.payload
        and event.completion is None
        and event.tool_discovery_result is None
        and event.recovery_metadata is None
        and event.provider_operation_status is None
    )


async def coalesce_model_stream_deltas(
    source: AsyncGenerator[ModelStreamEvent, None],
    *,
    before_accept: Callable[[], Awaitable[None]] | None = None,
) -> AsyncGenerator[ModelStreamEvent | BufferedModelCompletion, None]:
    """Keep the first delta prompt, then publish bounded consecutive text batches.

    The source has already observed every chunk for semantic-progress deadlines
    and rejected provider output that lost a cancellation race. Buffer only those
    accepted observations, so cancellation can flush them without admitting late
    provider output. ``before_accept`` checks session interruption at that same
    boundary, before a delta enters the buffer rather than when it is flushed.
    Time is checked on arrival, without creating another reader or cancellation
    owner. Exhaustion, failure and semantic boundaries flush accepted text before
    proceeding. A completion and its pending tail are handed off together so the
    attempt can retain terminal evidence before cancellable delta publication.
    Closing the consumer discards its unpublished buffer.
    """

    parts: list[str] = []
    size = 0
    kind: ModelStreamEventType | None = None
    published_at = monotonic()

    def flush() -> ModelStreamEvent:
        nonlocal size, published_at
        assert kind is not None
        event = ModelStreamEvent(type=kind, delta="".join(parts))
        parts.clear()
        size = 0
        published_at = monotonic()
        return event

    try:
        try:
            async for event in source:
                if type(event) is not ModelStreamEvent:
                    raise TypeError("Model providers must yield ModelStreamEvent instances.")
                if before_accept is not None:
                    await before_accept()
                if not _is_plain_delta(event):
                    if parts:
                        if event.type is ModelStreamEventType.COMPLETED:
                            batch = BufferedModelCompletion(flush(), event)
                            kind = None
                            yield batch
                            continue
                        yield flush()
                    kind = None
                    yield event
                    continue
                # Validate each untrusted input before concatenation; merging must
                # not repair malformed provider values or retain mutable objects.
                event = copy_model_stream_event(event)
                if event.type is not kind:
                    if parts:
                        yield flush()
                    kind = event.type
                    published_at = monotonic()
                    yield event
                    continue
                if parts and size + len(event.delta) > _MAX_DELTA_CHARS:
                    yield flush()
                parts.append(event.delta)
                size += len(event.delta)
                if (
                    size >= _MAX_DELTA_CHARS
                    or monotonic() - published_at >= _MAX_DELTA_DELAY_SECONDS
                ):
                    yield flush()
        except GeneratorExit:
            raise
        except BaseException:
            if parts:
                yield flush()
            raise
        else:
            if parts:
                yield flush()
    finally:
        primary_failure = sys.exception()
        try:
            await source.aclose()
        except BaseException as cleanup_failure:
            if (
                primary_failure is None
                or isinstance(primary_failure, GeneratorExit)
                or primary_failure is cleanup_failure
            ):
                raise
            raise BaseExceptionGroup(
                "Model stream buffering and cleanup both failed.",
                [primary_failure, cleanup_failure],
            ) from None

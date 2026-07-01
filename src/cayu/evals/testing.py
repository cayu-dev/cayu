from __future__ import annotations

from collections.abc import AsyncIterator, Iterable

from cayu.providers import (
    ModelProvider,
    ModelRequest,
    ModelStreamEvent,
    ModelStreamEventType,
)


class ScriptedModelProvider(ModelProvider):
    """Deterministic provider for local evals and tests.

    Pass either one batch of `ModelStreamEvent` values or multiple batches. Each
    model request consumes the next batch.
    """

    name = "scripted"

    def __init__(
        self,
        events: Iterable[ModelStreamEvent] | Iterable[Iterable[ModelStreamEvent]],
        *,
        name: str = "scripted",
    ) -> None:
        self.name = name
        self._batches: tuple[tuple[ModelStreamEvent, ...], ...] = _normalize_batches(events)
        self.requests: list[ModelRequest] = []

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        self.requests.append(request)
        index = len(self.requests) - 1
        if index >= len(self._batches):
            raise AssertionError(f"No scripted model event batch for request {index}.")
        for event in self._batches[index]:
            yield event


def _normalize_batches(
    events: Iterable[ModelStreamEvent] | Iterable[Iterable[ModelStreamEvent]],
) -> tuple[tuple[ModelStreamEvent, ...], ...]:
    values = list(events)
    if not values:
        return ()
    if all(type(value) is ModelStreamEvent for value in values):
        return (_require_complete_batch(tuple(_require_model_event(value) for value in values)),)
    batches: list[tuple[ModelStreamEvent, ...]] = []
    for batch in values:
        if isinstance(batch, ModelStreamEvent):
            raise TypeError("ScriptedModelProvider events must be one batch or multiple batches.")
        batches.append(
            _require_complete_batch(tuple(_require_model_event(event) for event in batch))
        )
    return tuple(batches)


def _require_model_event(value: object) -> ModelStreamEvent:
    if type(value) is not ModelStreamEvent:
        raise TypeError("ScriptedModelProvider batches must contain ModelStreamEvent values.")
    return value


def _require_complete_batch(batch: tuple[ModelStreamEvent, ...]) -> tuple[ModelStreamEvent, ...]:
    # The runtime requires every model step's stream to end with a COMPLETED event;
    # reject a script that would otherwise fail the run in a confusing way.
    if not batch:
        raise ValueError("ScriptedModelProvider batch must not be empty.")
    if batch[-1].type is not ModelStreamEventType.COMPLETED:
        raise ValueError("ScriptedModelProvider batch must end with a COMPLETED event.")
    return batch

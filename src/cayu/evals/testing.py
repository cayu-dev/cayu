from __future__ import annotations

from collections.abc import AsyncIterator, Iterable
from dataclasses import dataclass
from typing import Any

from cayu._validation import copy_json_value
from cayu.providers import (
    ModelProvider,
    ModelRequest,
    ModelStreamEvent,
    ModelStreamEventType,
    ProviderOperationAdapter,
    ProviderOperationConnection,
    ProviderOperationMode,
    ProviderOperationRecoveryMetadata,
    ProviderOperationSnapshot,
    ProviderOperationStartRequest,
    ProviderOperationState,
    ProviderOperationStatus,
    copy_model_stream_event,
)
from cayu.runtime.structured_output import STRUCTURED_OUTPUT_TOOL_NAME


def scripted_structured_output(
    output: Any,
    *,
    id: str | None = None,
) -> tuple[ModelStreamEvent, ModelStreamEvent]:
    """Build one complete provider script that submits structured JSON output.

    This is the public test/eval seam for Cayu's provider-neutral structured-output
    tool protocol. Callers supply only the JSON value; Cayu owns the reserved tool
    name and argument envelope.
    """
    return (
        ModelStreamEvent.tool_call(
            id=id,
            name=STRUCTURED_OUTPUT_TOOL_NAME,
            arguments={"output": copy_json_value(output, "output")},
        ),
        ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
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
        background: bool = False,
    ) -> None:
        self.name = name
        self._batches: tuple[tuple[ModelStreamEvent, ...], ...] = _normalize_batches(events)
        self.requests: list[ModelRequest] = []
        self._operation_adapter = _ScriptedProviderOperationAdapter(self) if background else None

    @property
    def provider_operation_mode(self) -> ProviderOperationMode:
        return (
            ProviderOperationMode.BACKGROUND
            if self._operation_adapter is not None
            else ProviderOperationMode.SYNCHRONOUS
        )

    @property
    def provider_operations(self) -> ProviderOperationAdapter | None:
        return self._operation_adapter

    @property
    def background_operation_ids(self) -> tuple[str, ...]:
        """Return deterministic operation ids created by this scripted provider."""

        if self._operation_adapter is None:
            return ()
        return tuple(self._operation_adapter.operations)

    def complete_background_operation(self, operation_id: str | None = None) -> str:
        """Simulate one scripted provider operation finishing while Cayu is offline."""

        if self._operation_adapter is None:
            raise RuntimeError("ScriptedModelProvider background mode is not enabled.")
        return self._operation_adapter.complete(operation_id)

    def _consume_batch(self, request: ModelRequest) -> tuple[ModelStreamEvent, ...]:
        self.requests.append(request)
        index = len(self.requests) - 1
        if index >= len(self._batches):
            raise AssertionError(f"No scripted model event batch for request {index}.")
        return self._batches[index]

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        for event in self._consume_batch(request):
            yield event


@dataclass
class _ScriptedProviderOperation:
    state: ProviderOperationState
    events: tuple[ModelStreamEvent, ...]
    status: ProviderOperationStatus = ProviderOperationStatus.IN_PROGRESS


class _ScriptedProviderOperationAdapter(ProviderOperationAdapter):
    def __init__(self, provider: ScriptedModelProvider) -> None:
        self.provider = provider
        self.operations: dict[str, _ScriptedProviderOperation] = {}

    def _require_operation(self, state: ProviderOperationState) -> _ScriptedProviderOperation:
        try:
            operation = self.operations[state.operation_id]
        except KeyError:
            raise KeyError(f"Unknown scripted provider operation: {state.operation_id}") from None
        if (
            operation.state.version != state.version
            or operation.state.operation_id != state.operation_id
            or operation.state.stream_protocol != state.stream_protocol
        ):
            raise ValueError("Scripted provider operation state does not match its identity.")
        cursor = state.recovery_metadata.cursor
        if cursor is None or cursor > len(operation.events):
            raise ValueError("Scripted provider operation cursor is unavailable.")
        expected_metadata = (
            operation.state.recovery_metadata
            if cursor == 0
            else operation.events[cursor - 1].recovery_metadata
        )
        if expected_metadata is None or state.recovery_metadata != expected_metadata:
            raise ValueError("Scripted provider operation cursor state is inconsistent.")
        return operation

    async def start(self, request: ProviderOperationStartRequest) -> ProviderOperationConnection:
        batch = tuple(
            copy_model_stream_event(event).model_copy(
                update={
                    "recovery_metadata": ProviderOperationRecoveryMetadata(cursor=cursor),
                }
            )
            for cursor, event in enumerate(
                self.provider._consume_batch(request.request),
                start=1,
            )
        )
        operation_id = f"scripted-operation-{len(self.operations)}"
        state = ProviderOperationState(
            operation_id=operation_id,
            stream_protocol="scripted-v1",
            recovery_metadata=ProviderOperationRecoveryMetadata(cursor=0),
        )
        operation = _ScriptedProviderOperation(state=state, events=batch)
        self.operations[operation_id] = operation

        async def events() -> AsyncIterator[ModelStreamEvent]:
            for event in operation.events:
                if event.type is ModelStreamEventType.COMPLETED:
                    operation.status = ProviderOperationStatus.COMPLETED
                yield event

        return ProviderOperationConnection(
            state=state,
            status=operation.status,
            events=events(),
        )

    async def retrieve(self, state: ProviderOperationState) -> ProviderOperationSnapshot:
        operation = self._require_operation(state)
        return ProviderOperationSnapshot(
            state=state,
            status=operation.status,
            events=(
                operation.events if operation.status is ProviderOperationStatus.COMPLETED else ()
            ),
        )

    async def reconnect(self, state: ProviderOperationState) -> ProviderOperationConnection:
        operation = self._require_operation(state)
        cursor = state.recovery_metadata.cursor
        if cursor is None:  # pragma: no cover - _require_operation owns this invariant
            raise AssertionError("Validated scripted cursor disappeared.")
        start_index = 0 if cursor == 0 else cursor - 1

        async def events() -> AsyncIterator[ModelStreamEvent]:
            for event in operation.events[start_index:]:
                if event.type is ModelStreamEventType.COMPLETED:
                    operation.status = ProviderOperationStatus.COMPLETED
                yield event

        return ProviderOperationConnection(
            state=state,
            status=operation.status,
            events=events(),
        )

    async def cancel(self, state: ProviderOperationState) -> ProviderOperationSnapshot:
        operation = self._require_operation(state)
        if not operation.status.terminal:
            operation.status = ProviderOperationStatus.CANCELLED
        return ProviderOperationSnapshot(
            state=state,
            status=operation.status,
        )

    def complete(self, operation_id: str | None) -> str:
        if operation_id is None:
            if not self.operations:
                raise RuntimeError("No scripted background operation has been started.")
            operation_id = next(reversed(self.operations))
        try:
            operation = self.operations[operation_id]
        except KeyError:
            raise KeyError(f"Unknown scripted provider operation: {operation_id}") from None
        if operation.status in {
            ProviderOperationStatus.CANCELLED,
            ProviderOperationStatus.FAILED,
            ProviderOperationStatus.EXPIRED,
        }:
            raise RuntimeError(
                f"Cannot complete scripted operation in state {operation.status.value}."
            )
        operation.status = ProviderOperationStatus.COMPLETED
        return operation_id


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
    if batch[-1].type != ModelStreamEventType.COMPLETED:
        raise ValueError("ScriptedModelProvider batch must end with a COMPLETED event.")
    return batch

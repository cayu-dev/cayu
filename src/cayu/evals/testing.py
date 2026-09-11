from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Iterable
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

from cayu._validation import copy_json_value
from cayu.core.messages import Message
from cayu.providers import (
    ModelProvider,
    ModelRequest,
    ModelStreamEvent,
    ModelStreamEventType,
    ProviderOperationAdapter,
    ProviderOperationCancellationSupport,
    ProviderOperationConnection,
    ProviderOperationMode,
    ProviderOperationRecoveryMetadata,
    ProviderOperationSnapshot,
    ProviderOperationStartRequest,
    ProviderOperationState,
    ProviderOperationStatus,
    copy_model_stream_event,
)
from cayu.providers.base import _preflight_provider_portable_messages
from cayu.runtime.structured_output import STRUCTURED_OUTPUT_TOOL_NAME


@dataclass
class _ScriptedEvalTrial:
    parent: _ScriptedEvalTrial | None = None
    active: bool = True


_eval_trial_owner: ContextVar[_ScriptedEvalTrial | None] = ContextVar(
    "scripted_eval_trial_owner", default=None
)
_eval_allowed_providers: ContextVar[tuple[ScriptedModelProvider, ...]] = ContextVar(
    "scripted_eval_allowed_providers", default=()
)
_eval_requires_owned_providers: ContextVar[bool] = ContextVar(
    "scripted_eval_requires_owned_providers", default=False
)


@dataclass
class _ScriptedEvalInvocation:
    providers: dict[int, ScriptedModelProvider] = field(default_factory=dict)
    parent: _ScriptedEvalInvocation | None = None
    delegated: dict[int, _ScriptedEvalInvocation] = field(default_factory=dict)
    active: bool = True


_eval_invocation: ContextVar[_ScriptedEvalInvocation | None] = ContextVar(
    "scripted_eval_invocation", default=None
)


def _claim_positional_provider(
    provider: ScriptedModelProvider, *, delegate_from: _ScriptedEvalInvocation | None = None
) -> None:
    owner = _eval_trial_owner.get()
    trial = owner
    while trial is not None:
        if not trial.active:
            raise ValueError("A completed eval trial cannot consume positional scripted batches.")
        trial = trial.parent
    if (
        owner is not None
        and _eval_requires_owned_providers.get()
        and provider._positional_eval_owner is not owner
        and not any(provider is allowed for allowed in _eval_allowed_providers.get())
    ):
        raise ValueError(
            "Eval trials require case-owned or predeclared positional ScriptedModelProvider "
            "instances; use response_factory or construct a provider per trial."
        )
    invocation = _eval_invocation.get()
    ancestor = invocation
    while ancestor is not None:
        if not ancestor.active:
            raise ValueError(
                "A completed eval invocation cannot consume positional scripted batches."
            )
        ancestor = ancestor.parent
    if invocation is None:
        if provider._active_eval_invocation is not None:
            raise ValueError("A positional ScriptedModelProvider is reserved by an active eval.")
        return
    if (
        provider._active_eval_invocation is not None
        and provider._active_eval_invocation is not invocation
    ):
        if (
            delegate_from is not None
            and delegate_from is invocation.parent
            and provider._active_eval_invocation is delegate_from
        ):
            invocation.delegated[id(provider)] = delegate_from
        else:
            raise ValueError(
                "Concurrent eval invocations cannot share a positional ScriptedModelProvider."
            )
    provider._active_eval_invocation = invocation
    invocation.providers[id(provider)] = provider


@contextmanager
def _scripted_eval_invocation(providers: Iterable[ScriptedModelProvider]):
    parent = _eval_invocation.get()
    owned = _ScriptedEvalInvocation(parent=parent)
    token = _eval_invocation.set(owned)
    try:
        # Reserve before any case setup can yield, not on first stream arrival.
        for provider in providers:
            # Only explicitly selected providers may be delegated, and only
            # from the direct parent. A sibling's active lease cannot be taken.
            _claim_positional_provider(provider, delegate_from=parent)
        yield
    finally:
        owned.active = False
        for provider in owned.providers.values():
            if provider._active_eval_invocation is owned:
                previous = owned.delegated.get(id(provider))
                # An intermediate parent may already have unwound while an
                # owned descendant was draining. Preserve an ancestor's fence.
                while previous is not None and not previous.active:
                    previous = previous.parent
                provider._active_eval_invocation = (
                    previous
                    if previous is not None and id(provider) in previous.providers
                    else None
                )
        owned.providers.clear()
        owned.delegated.clear()
        _eval_invocation.reset(token)


@contextmanager
def _scripted_eval_trial(
    *,
    concurrent: bool,
    providers: tuple[ScriptedModelProvider, ...] = (),
    require_owned: bool = False,
):
    owned = _ScriptedEvalTrial(parent=_eval_trial_owner.get())
    token = _eval_trial_owner.set(owned)
    allowed_token = _eval_allowed_providers.set(providers)
    ownership_token = _eval_requires_owned_providers.set(
        concurrent or require_owned or _eval_requires_owned_providers.get()
    )
    try:
        yield
    finally:
        owned.active = False
        _eval_requires_owned_providers.reset(ownership_token)
        _eval_allowed_providers.reset(allowed_token)
        _eval_trial_owner.reset(token)


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

    Pass positional event batches for sequential scripts, or a synchronous
    `response_factory` selecting one complete batch from a detached `ModelRequest`.
    Shared concurrent evals require request-aware selection; per-trial workflow
    factories may instead construct independent positional providers. Native
    structured-output support is explicitly opt-in.
    """

    name = "scripted"
    supports_native_structured_output = False

    def preflight_portable_messages(
        self,
        *,
        model: str,
        messages: list[Message],
        tools: list[dict[str, Any]],
    ) -> None:
        _preflight_provider_portable_messages(
            model=model,
            messages=messages,
            tools=tools,
            supports_system_messages=True,
            supports_tool_history=True,
            supports_tool_definitions=True,
            supports_file_attachments=True,
        )

    def __init__(
        self,
        events: Iterable[ModelStreamEvent] | Iterable[Iterable[ModelStreamEvent]] | None = None,
        *,
        name: str = "scripted",
        background: bool = False,
        response_factory: Callable[[ModelRequest], Iterable[ModelStreamEvent]] | None = None,
        supports_native_structured_output: bool = False,
    ) -> None:
        if events is not None and response_factory is not None:
            raise ValueError("ScriptedModelProvider accepts events or response_factory, not both.")
        if events is None and response_factory is None:
            raise ValueError("ScriptedModelProvider requires events or response_factory.")
        if response_factory is not None and not callable(response_factory):
            raise TypeError("response_factory must be callable.")
        if type(supports_native_structured_output) is not bool:
            raise TypeError("supports_native_structured_output must be a bool.")
        self.name = name
        self.supports_native_structured_output = supports_native_structured_output
        self._batches = _normalize_batches(events) if events is not None else None
        self._response_factory = response_factory
        self._positional_eval_owner = _eval_trial_owner.get()
        self._active_eval_invocation: _ScriptedEvalInvocation | None = None
        if self._response_factory is None:
            # A factory-created provider is owned immediately, including while
            # the factory is suspended before handing its app to the runner.
            _claim_positional_provider(self)
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
        if self._response_factory is None:
            _claim_positional_provider(self)
        self.requests.append(_copy_model_request(request))
        if self._response_factory is not None:
            return _require_complete_batch(
                tuple(
                    _require_model_event(event)
                    for event in self._response_factory(_copy_model_request(request))
                )
            )
        assert self._batches is not None
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

    @property
    def cancellation_support(self) -> ProviderOperationCancellationSupport:
        return ProviderOperationCancellationSupport.SUPPORTED

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
    return ModelStreamEvent.model_validate(value.model_dump(mode="python"))


def _copy_model_request(request: ModelRequest) -> ModelRequest:
    if type(request) is not ModelRequest:
        raise TypeError("ScriptedModelProvider requests must be ModelRequest values.")
    return ModelRequest.model_validate(request.model_dump(mode="python"))


def _require_complete_batch(batch: tuple[ModelStreamEvent, ...]) -> tuple[ModelStreamEvent, ...]:
    # The runtime requires every model step's stream to end with a COMPLETED event;
    # reject a script that would otherwise fail the run in a confusing way.
    if not batch:
        raise ValueError("ScriptedModelProvider batch must not be empty.")
    if batch[-1].type != ModelStreamEventType.COMPLETED:
        raise ValueError("ScriptedModelProvider batch must end with a COMPLETED event.")
    return batch


def _execution_profile_material(provider: ScriptedModelProvider) -> dict[str, Any] | None:
    """Bounded configuration material for runtime exact-type identity selection."""
    if provider._response_factory is not None:
        # An arbitrary application callback is not transparent built-in behavior.
        return None
    return {
        "background": provider.provider_operations is not None,
        "supports_native_structured_output": provider.supports_native_structured_output,
    }

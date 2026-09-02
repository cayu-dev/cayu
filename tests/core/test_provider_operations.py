from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from uuid import uuid4

import pytest
from pydantic import ValidationError

from cayu._exception_groups import exception_cause, set_exception_cause
from cayu.core import AgentSpec, EventType, Message, ThinkingConfig, ToolCallPart
from cayu.core.billing import BillingIdentity
from cayu.core.events import Event
from cayu.providers import (
    ModelContextOverflowError,
    ModelProvider,
    ModelProviderError,
    ModelRequest,
    ModelStreamEvent,
    ProviderOperationAdapter,
    ProviderOperationCancellationSupport,
    ProviderOperationConnection,
    ProviderOperationMode,
    ProviderOperationSnapshot,
    ProviderOperationStartIdempotencySupport,
    ProviderOperationStartRequest,
    ProviderOperationState,
    ProviderOperationStatus,
)
from cayu.runtime import (
    CayuApp,
    EventQuery,
    EventRecord,
    InMemorySessionStore,
    InterruptSessionRequest,
    PersistedEventSideEffectClaim,
    RecentTurnsContextPolicy,
    RetryPolicy,
    RunLimits,
    RunRequest,
    SessionIdentity,
    SessionRunFenced,
    SessionStatus,
)
from cayu.runtime._model_step_executor import (
    MAX_MODEL_COMPLETION_RECOVERY_BUDGET_LIMITS,
    MAX_MODEL_COMPLETION_RECOVERY_CONTEXT_BYTES,
    MAX_MODEL_COMPLETION_RECOVERY_METADATA_ENTRIES,
    ModelAttemptFailed,
    ModelCompletionRecoveryContext,
    _raise_terminal_model_attempt_failure,
)
from cayu.runtime.provider_operations import (
    ProviderOperationAccountingStatus,
    ProviderOperationCancellationStatus,
    ProviderOperationEvidenceError,
    ProviderOperationInspectionStatus,
    inspect_provider_operation,
)
from cayu.runtime.sessions import ModelCompletionStageRequest, ModelCompletionStageResult
from cayu.vaults import SecretRedactor


class _ReconnectableAdapter(ProviderOperationAdapter):
    def __init__(self, *, cursor: int = 0, start_error: Exception | None = None) -> None:
        self.cursor = cursor
        self.start_error = start_error
        self.start_calls = 0
        self.start_requests: list[ProviderOperationStartRequest] = []
        self.cancel_calls = 0

    @property
    def cancellation_support(self) -> ProviderOperationCancellationSupport:
        return ProviderOperationCancellationSupport.SUPPORTED

    async def start(self, request: ProviderOperationStartRequest) -> ProviderOperationConnection:
        self.start_calls += 1
        self.start_requests.append(request)
        if self.start_error is not None:
            raise self.start_error

        async def events() -> AsyncIterator[ModelStreamEvent]:
            yield ModelStreamEvent.text_delta(
                "done",
                recovery_metadata={"cursor": self.cursor + 1},
            )
            yield ModelStreamEvent.completed(
                {"finish_reason": "stop"},
                recovery_metadata={"cursor": self.cursor + 2},
            )

        return ProviderOperationConnection(
            state=ProviderOperationState(
                operation_id="response_123",
                stream_protocol="responses-v1",
                recovery_metadata={"cursor": self.cursor},
            ),
            status=ProviderOperationStatus.IN_PROGRESS,
            events=events(),
        )

    async def retrieve(self, state: ProviderOperationState) -> ProviderOperationSnapshot:
        raise AssertionError("retrieve is not used for an initial dispatch")

    async def reconnect(self, state: ProviderOperationState) -> ProviderOperationConnection:
        raise AssertionError("reconnect is not used for an initial dispatch")

    async def cancel(self, state: ProviderOperationState) -> ProviderOperationSnapshot:
        self.cancel_calls += 1
        return ProviderOperationSnapshot(
            state=state,
            status=ProviderOperationStatus.CANCELLED,
        )


class _InterruptibleOperationAdapter(_ReconnectableAdapter):
    def __init__(self) -> None:
        super().__init__()
        self.stream_started = asyncio.Event()
        self.cancelled_states: list[ProviderOperationState] = []

    async def start(self, request: ProviderOperationStartRequest) -> ProviderOperationConnection:
        self.start_calls += 1
        self.start_requests.append(request)
        state = ProviderOperationState(
            operation_id="response_interruptible",
            stream_protocol="responses-v1",
            recovery_metadata={"cursor": 0},
        )

        async def events() -> AsyncIterator[ModelStreamEvent]:
            self.stream_started.set()
            await asyncio.Event().wait()
            raise AssertionError("unreachable")
            yield  # pragma: no cover - keeps this an async generator

        return ProviderOperationConnection(
            state=state,
            status=ProviderOperationStatus.IN_PROGRESS,
            events=events(),
        )

    async def cancel(self, state: ProviderOperationState) -> ProviderOperationSnapshot:
        self.cancel_calls += 1
        self.cancelled_states.append(state)
        return ProviderOperationSnapshot(state=state, status=ProviderOperationStatus.CANCELLED)


class _CompletionWinsLiveCancellationAdapter(_InterruptibleOperationAdapter):
    def __init__(self) -> None:
        super().__init__()
        self.completed = False

    def _completed_snapshot(
        self,
        state: ProviderOperationState,
    ) -> ProviderOperationSnapshot:
        return ProviderOperationSnapshot(
            state=state,
            status=ProviderOperationStatus.COMPLETED,
            events=(
                ModelStreamEvent.text_delta(
                    "completed during cancellation",
                    recovery_metadata={"cursor": 1},
                ),
                ModelStreamEvent.completed(
                    {
                        "finish_reason": "stop",
                        "usage": {"input_tokens": 3, "output_tokens": 4},
                    },
                    recovery_metadata={"cursor": 2},
                ),
            ),
        )

    async def cancel(self, state: ProviderOperationState) -> ProviderOperationSnapshot:
        self.cancel_calls += 1
        self.cancelled_states.append(state)
        self.completed = True
        return self._completed_snapshot(state)

    async def retrieve(self, state: ProviderOperationState) -> ProviderOperationSnapshot:
        assert self.completed
        return self._completed_snapshot(state)


class _GeneratedToolIdAdapter(_ReconnectableAdapter):
    async def start(self, request: ProviderOperationStartRequest) -> ProviderOperationConnection:
        self.start_calls += 1
        self.start_requests.append(request)

        async def events() -> AsyncIterator[ModelStreamEvent]:
            yield ModelStreamEvent.tool_call(
                name="missing-tool",
                arguments={"query": "durable"},
                recovery_metadata={"cursor": 1},
            )
            yield ModelStreamEvent.completed(
                {"finish_reason": "tool_calls"},
                recovery_metadata={"cursor": 2},
            )

        return ProviderOperationConnection(
            state=ProviderOperationState(
                operation_id="response_generated_tool_id",
                stream_protocol="responses-v1",
                recovery_metadata={"cursor": 0},
            ),
            status=ProviderOperationStatus.IN_PROGRESS,
            events=events(),
        )


class _StartValidationEvents:
    def __init__(self) -> None:
        self.closed = False
        self.closed_event = asyncio.Event()

    def __aiter__(self) -> _StartValidationEvents:
        return self

    async def __anext__(self) -> ModelStreamEvent:
        raise StopAsyncIteration

    async def aclose(self) -> None:
        self.closed = True
        self.closed_event.set()


class _InvalidStartConnectionAdapter(_ReconnectableAdapter):
    def __init__(self) -> None:
        super().__init__()
        self.events = _StartValidationEvents()

    async def start(self, request: ProviderOperationStartRequest) -> ProviderOperationConnection:
        self.start_calls += 1
        self.start_requests.append(request)
        return ProviderOperationConnection(
            state=ProviderOperationState(
                operation_id="valid-before-copy",
                stream_protocol="responses-v1",
                recovery_metadata={"cursor": 0},
            ).model_copy(
                update={"operation_id": ""},
            ),
            status=ProviderOperationStatus.IN_PROGRESS,
            events=self.events,
        )


class _TerminalBlockingStartEvents:
    def __init__(self, *, close_error: BaseException | None = None) -> None:
        self._yielded = False
        self._close_error = close_error
        self.closed = False

    def __aiter__(self) -> _TerminalBlockingStartEvents:
        return self

    async def __anext__(self) -> ModelStreamEvent:
        if not self._yielded:
            self._yielded = True
            return ModelStreamEvent.completed(
                {
                    "finish_reason": "stop",
                    "usage": {"input_tokens": 3, "output_tokens": 2},
                },
                recovery_metadata={"cursor": 1},
            )
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    async def aclose(self) -> None:
        self.closed = True
        if self._close_error is not None:
            raise self._close_error


class _TerminalBlockingStartAdapter(_ReconnectableAdapter):
    def __init__(self, *, close_error: BaseException | None = None) -> None:
        super().__init__()
        self.events = _TerminalBlockingStartEvents(close_error=close_error)

    async def start(self, request: ProviderOperationStartRequest) -> ProviderOperationConnection:
        self.start_calls += 1
        self.start_requests.append(request)
        return ProviderOperationConnection(
            state=ProviderOperationState(
                operation_id="terminal-blocking-start",
                stream_protocol="responses-v1",
                recovery_metadata={"cursor": 0},
            ),
            status=ProviderOperationStatus.IN_PROGRESS,
            events=self.events,
        )


class _TerminalCursorGapAdapter(_ReconnectableAdapter):
    async def start(self, request: ProviderOperationStartRequest) -> ProviderOperationConnection:
        self.start_calls += 1
        self.start_requests.append(request)

        async def events() -> AsyncIterator[ModelStreamEvent]:
            yield ModelStreamEvent.completed(
                {
                    "finish_reason": "stop",
                    "usage": {"input_tokens": 3, "output_tokens": 2},
                },
                recovery_metadata={"cursor": 2},
            )

        return ProviderOperationConnection(
            state=ProviderOperationState(
                operation_id="terminal-cursor-gap",
                stream_protocol="responses-v1",
                recovery_metadata={"cursor": 0},
            ),
            status=ProviderOperationStatus.IN_PROGRESS,
            events=events(),
        )


class _CursorOverflowErrorAdapter(_ReconnectableAdapter):
    async def start(self, request: ProviderOperationStartRequest) -> ProviderOperationConnection:
        self.start_calls += 1
        self.start_requests.append(request)

        async def events() -> AsyncIterator[ModelStreamEvent]:
            yield ModelStreamEvent.error(
                "provider context limit exceeded",
                cause=ModelContextOverflowError(
                    "provider context limit exceeded",
                    provider="reconnectable",
                    status_code=400,
                    error_code="context_length_exceeded",
                ),
                recovery_metadata={"cursor": 1},
            )

        return ProviderOperationConnection(
            state=ProviderOperationState(
                operation_id="cursor-overflow-error",
                stream_protocol="responses-v1",
                recovery_metadata={"cursor": 0},
            ),
            status=ProviderOperationStatus.IN_PROGRESS,
            events=events(),
        )


class _BlockingCancelAdapter(_ReconnectableAdapter):
    def __init__(self) -> None:
        super().__init__()
        self.cancel_entered = asyncio.Event()
        self.cancel_release = asyncio.Event()

    async def cancel(self, state: ProviderOperationState) -> ProviderOperationSnapshot:
        self.cancel_calls += 1
        self.cancel_entered.set()
        await self.cancel_release.wait()
        return ProviderOperationSnapshot(
            state=state,
            status=ProviderOperationStatus.CANCELLED,
        )


class _CancellationResistantCancelAdapter(_ReconnectableAdapter):
    def __init__(self) -> None:
        super().__init__()
        self.cancel_entered = asyncio.Event()
        self.local_cancellation_observed = asyncio.Event()
        self.cancel_release = asyncio.Event()

    async def cancel(self, state: ProviderOperationState) -> ProviderOperationSnapshot:
        self.cancel_calls += 1
        self.cancel_entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.local_cancellation_observed.set()
            await self.cancel_release.wait()
            raise


class _FailingOperationStreamAdapter(_ReconnectableAdapter):
    async def start(self, request: ProviderOperationStartRequest) -> ProviderOperationConnection:
        connection = await super().start(request)

        async def events() -> AsyncIterator[ModelStreamEvent]:
            raise ModelProviderError(
                "provider stream disconnected",
                provider="reconnectable",
                retryable=True,
            )
            yield  # pragma: no cover - keeps this an async generator

        return ProviderOperationConnection(
            state=connection.state,
            status=connection.status,
            events=events(),
        )


class _ErrorEventOperationStreamAdapter(_ReconnectableAdapter):
    async def start(self, request: ProviderOperationStartRequest) -> ProviderOperationConnection:
        connection = await super().start(request)

        async def events() -> AsyncIterator[ModelStreamEvent]:
            yield ModelStreamEvent.error(
                "provider stream disconnected",
                cause=ModelProviderError(
                    "provider stream disconnected",
                    provider="reconnectable",
                    retryable=True,
                ),
            )

        return ProviderOperationConnection(
            state=connection.state,
            status=connection.status,
            events=events(),
        )


class _EmptyOperationStreamAdapter(_ReconnectableAdapter):
    async def start(self, request: ProviderOperationStartRequest) -> ProviderOperationConnection:
        connection = await super().start(request)

        async def events() -> AsyncIterator[ModelStreamEvent]:
            return
            yield  # pragma: no cover - keeps this an async generator

        return ProviderOperationConnection(
            state=connection.state,
            status=connection.status,
            events=events(),
        )


class _OverflowingOperationStreamAdapter(_ReconnectableAdapter):
    async def start(self, request: ProviderOperationStartRequest) -> ProviderOperationConnection:
        connection = await super().start(request)

        async def events() -> AsyncIterator[ModelStreamEvent]:
            raise ModelContextOverflowError(
                "provider context limit exceeded",
                provider="reconnectable",
                status_code=400,
                error_code="context_length_exceeded",
            )
            yield  # pragma: no cover - keeps this an async generator

        return ProviderOperationConnection(
            state=connection.state,
            status=connection.status,
            events=events(),
        )


class _ProviderIdentityAdapter(_ReconnectableAdapter):
    def __init__(self, *, operation_id: str, stream_protocol: str = "responses-v1") -> None:
        super().__init__()
        self.operation_id = operation_id
        self.stream_protocol = stream_protocol

    async def start(self, request: ProviderOperationStartRequest) -> ProviderOperationConnection:
        connection = await super().start(request)
        return ProviderOperationConnection(
            state=ProviderOperationState(
                operation_id=self.operation_id,
                stream_protocol=self.stream_protocol,
                recovery_metadata=connection.state.recovery_metadata,
            ),
            status=connection.status,
            events=connection.events,
        )


class _BlockingStartAdapter(_ReconnectableAdapter):
    def __init__(self) -> None:
        super().__init__()
        self.start_entered = asyncio.Event()
        self.start_release = asyncio.Event()

    async def start(self, request: ProviderOperationStartRequest) -> ProviderOperationConnection:
        self.start_calls += 1
        self.start_requests.append(request)
        self.start_entered.set()
        await self.start_release.wait()

        async def events() -> AsyncIterator[ModelStreamEvent]:
            yield ModelStreamEvent.text_delta("done")
            yield ModelStreamEvent.completed({"finish_reason": "stop"})

        return ProviderOperationConnection(
            state=ProviderOperationState(
                operation_id="response_123",
                stream_protocol="responses-v1",
            ),
            status=ProviderOperationStatus.IN_PROGRESS,
            events=events(),
        )


class _CancellationResistantStartAdapter(_BlockingStartAdapter):
    def __init__(self) -> None:
        super().__init__()
        self.local_cancellation_observed = asyncio.Event()

    async def start(self, request: ProviderOperationStartRequest) -> ProviderOperationConnection:
        try:
            return await super().start(request)
        except asyncio.CancelledError:
            self.local_cancellation_observed.set()
            await self.start_release.wait()
            raise


class _LateSuccessfulStartAdapter(_BlockingStartAdapter):
    def __init__(self) -> None:
        super().__init__()
        self.local_cancellation_observed = asyncio.Event()
        self.late_cancel_observed = asyncio.Event()
        self.cancelled_states: list[ProviderOperationState] = []

    async def start(self, request: ProviderOperationStartRequest) -> ProviderOperationConnection:
        self.start_calls += 1
        self.start_requests.append(request)
        self.start_entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.local_cancellation_observed.set()
            await self.start_release.wait()

        async def events() -> AsyncIterator[ModelStreamEvent]:
            yield ModelStreamEvent.text_delta("late")

        return ProviderOperationConnection(
            state=ProviderOperationState(
                operation_id="response_late",
                stream_protocol="responses-v1",
            ),
            status=ProviderOperationStatus.IN_PROGRESS,
            events=events(),
        )

    async def cancel(self, state: ProviderOperationState) -> ProviderOperationSnapshot:
        self.cancel_calls += 1
        self.cancelled_states.append(state)
        self.late_cancel_observed.set()
        return ProviderOperationSnapshot(state=state, status=ProviderOperationStatus.CANCELLED)


class _LateInvalidStartConnectionAdapter(_LateSuccessfulStartAdapter):
    def __init__(self) -> None:
        super().__init__()
        self.events = _StartValidationEvents()

    async def start(self, request: ProviderOperationStartRequest) -> ProviderOperationConnection:
        self.start_calls += 1
        self.start_requests.append(request)
        self.start_entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.local_cancellation_observed.set()
            await self.start_release.wait()
        return ProviderOperationConnection(
            state=ProviderOperationState(
                operation_id="late-invalid-before-copy",
                stream_protocol="responses-v1",
            ).model_copy(update={"operation_id": ""}),
            status=ProviderOperationStatus.IN_PROGRESS,
            events=self.events,
        )


class _LateFailingCancelAdapter(_LateSuccessfulStartAdapter):
    async def cancel(self, state: ProviderOperationState) -> ProviderOperationSnapshot:
        self.cancel_calls += 1
        self.cancelled_states.append(state)
        self.late_cancel_observed.set()
        raise TimeoutError("late provider cancellation failed")


class _SynchronousFailingCancelAdapter(_ReconnectableAdapter):
    def cancel(  # ty: ignore[invalid-method-override]
        self,
        state: ProviderOperationState,
    ) -> ProviderOperationSnapshot:
        self.cancel_calls += 1
        raise TimeoutError("cancel failed before returning an awaitable")


class _ReconnectableProvider(ModelProvider):
    name = "reconnectable"

    def __init__(
        self,
        *,
        background: bool = False,
        cursor: int = 0,
        start_error: Exception | None = None,
    ) -> None:
        self.background = background
        self.adapter = _ReconnectableAdapter(cursor=cursor, start_error=start_error)
        self.stream_calls = 0

    @property
    def provider_operation_mode(self) -> ProviderOperationMode:
        if self.background:
            return ProviderOperationMode.BACKGROUND
        return ProviderOperationMode.SYNCHRONOUS

    @property
    def provider_operations(self) -> ProviderOperationAdapter:
        return self.adapter

    def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        self.stream_calls += 1

        async def events() -> AsyncIterator[ModelStreamEvent]:
            yield ModelStreamEvent.text_delta("done")
            yield ModelStreamEvent.completed({"finish_reason": "stop"})

        return events()


class _ContextualReconnectableProvider(_ReconnectableProvider):
    def __init__(self, identity: BillingIdentity) -> None:
        super().__init__(background=True)
        self.identity = identity

    async def billing_identity_for_request(
        self,
        request: ModelRequest,
    ) -> BillingIdentity | None:
        del request
        return self.identity

    def billing_identity_for_completion(
        self,
        identity: BillingIdentity | None,
        payload: dict,
    ) -> BillingIdentity | None:
        del payload
        assert identity == self.identity
        return identity


class _CaptureStageStore(InMemorySessionStore):
    invocation_lifecycle_command_version = 1

    def __init__(self) -> None:
        super().__init__()
        self.stage_requests: list[ModelCompletionStageRequest] = []

    async def prepare_model_completion_stage(
        self,
        session_id: str,
        *,
        request: ModelCompletionStageRequest,
        expected_statuses: set[SessionStatus],
        expected_run_epoch: int,
        expected_transcript_cursor: int,
    ) -> ModelCompletionStageResult:
        self.stage_requests.append(request.model_copy(deep=True))
        return await super().prepare_model_completion_stage(
            session_id,
            request=request,
            expected_statuses=expected_statuses,
            expected_run_epoch=expected_run_epoch,
            expected_transcript_cursor=expected_transcript_cursor,
        )


class _SynchronousProvider(ModelProvider):
    name = "synchronous"

    def __init__(self) -> None:
        self.stream_calls = 0

    def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        self.stream_calls += 1

        async def events() -> AsyncIterator[ModelStreamEvent]:
            yield ModelStreamEvent.text_delta("done")
            yield ModelStreamEvent.completed({"finish_reason": "stop"})

        return events()


def test_provider_operation_state_is_versioned_portable_and_bounded() -> None:
    state = ProviderOperationState(
        operation_id="response_123",
        stream_protocol="responses-v1",
        recovery_metadata={"cursor": 7},
    )

    assert state.version == 1
    assert state.model_dump(mode="json") == {
        "version": 1,
        "operation_id": "response_123",
        "stream_protocol": "responses-v1",
        "recovery_metadata": {"cursor": 7},
    }

    with pytest.raises(ValueError, match="operation_id"):
        ProviderOperationState(operation_id="x" * 513, stream_protocol="responses-v1")
    with pytest.raises(ValueError, match="recovery_metadata"):
        ProviderOperationState(
            operation_id="response_123",
            stream_protocol="responses-v1",
            recovery_metadata={"cursor": 2**63},
        )
    with pytest.raises(ValueError, match="recovery_metadata"):
        ProviderOperationState(
            operation_id="response_123",
            stream_protocol="responses-v1",
            recovery_metadata={"cursor": object()},
        )

    opaque = {"continuation": {"page": "next"}}
    reconnectable = ProviderOperationState(
        operation_id="response_opaque",
        stream_protocol="responses-v1",
        recovery_metadata={"cursor": 8, "opaque": opaque},
    )
    opaque["continuation"]["page"] = "mutated"  # type: ignore[index]
    assert reconnectable.recovery_metadata.opaque == {"continuation": {"page": "next"}}
    with pytest.raises(ValueError, match="byte limit|recovery_metadata"):
        ProviderOperationState(
            operation_id="response_opaque",
            stream_protocol="responses-v1",
            recovery_metadata={"cursor": 8, "opaque": {"continuation": "x" * 5000}},
        )


@pytest.mark.parametrize(
    "recovery_metadata",
    [
        {"bearer_token": "credential"},
        {"private_key": "credential"},
        {"request": {"messages": []}},
        {"analysis": "private chain of thought"},
        {"response_body": "unbounded provider response"},
        {"cursor": "Bearer sk-test-secret"},
        {"cursor": "raw request: user private message"},
        {"cursor": "hidden reasoning: chain of thought"},
        {"cursor": -1},
        {"cursor": True},
        {"cursor": 1.0},
    ],
)
def test_provider_operation_state_rejects_unsafe_recovery_content(
    recovery_metadata: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="must not contain"):
        ProviderOperationState(
            operation_id="response_123",
            stream_protocol="responses-v1",
            recovery_metadata=recovery_metadata,
        )


def test_provider_operation_status_interprets_terminal_states() -> None:
    assert ProviderOperationStatus.IN_PROGRESS.terminal is False
    assert ProviderOperationStatus.QUEUED.terminal is False
    assert ProviderOperationStatus.COMPLETED.terminal is True
    assert ProviderOperationStatus.FAILED.terminal is True


def test_model_provider_defaults_to_no_provider_operation_capability() -> None:
    provider = _SynchronousProvider()

    assert provider.provider_operations is None
    assert provider.provider_operation_mode is ProviderOperationMode.SYNCHRONOUS


def test_provider_operation_adapter_requires_explicit_start_idempotency_support() -> None:
    adapter = _ReconnectableAdapter()

    assert adapter.start_idempotency_support is ProviderOperationStartIdempotencySupport.UNSUPPORTED


def test_capability_support_does_not_enable_background_dispatch() -> None:
    provider = _ReconnectableProvider()
    app = CayuApp(enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    events = asyncio.run(
        _collect_run_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="supported_but_synchronous",
                messages=[Message.text("user", "hello")],
            ),
        )
    )

    assert provider.stream_calls == 1
    assert provider.adapter.start_calls == 0
    assert not {
        EventType.PROVIDER_OPERATION_STARTING,
        EventType.PROVIDER_OPERATION_STARTED,
    }.intersection(event.type for event in events)


def test_reconnectable_dispatch_persists_identity_before_model_output() -> None:
    provider = _ReconnectableProvider(background=True)
    app = CayuApp(enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    async def run():
        public_events = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="provider_operation_start",
                    messages=[Message.text("user", "hello")],
                )
            )
        ]
        stored_events = await app.session_store.load_events("provider_operation_start")
        return public_events, stored_events

    public_events, stored_events = asyncio.run(run())

    assert provider.adapter.start_calls == 1
    assert provider.stream_calls == 0
    public_types = [event.type for event in public_events]
    assert (
        public_types.index(EventType.MODEL_STARTED)
        < public_types.index(EventType.PROVIDER_OPERATION_STARTING)
        < public_types.index(EventType.PROVIDER_OPERATION_STARTED)
        < public_types.index(EventType.MODEL_TEXT_DELTA)
    )
    public_starting = next(
        event for event in public_events if event.type == EventType.PROVIDER_OPERATION_STARTING
    )
    assert "start_id" not in public_starting.payload
    public_start = next(
        event for event in public_events if event.type == EventType.PROVIDER_OPERATION_STARTED
    )
    public_text = next(event for event in public_events if event.type == EventType.MODEL_TEXT_DELTA)
    public_completed = next(
        event for event in public_events if event.type == EventType.MODEL_COMPLETED
    )
    assert public_start.payload["operation_id"] == "response_123"
    assert public_start.payload["stream_protocol"] == "responses-v1"
    assert public_start.payload["status"] == "in_progress"
    assert public_start.payload["state_version"] == 1
    assert "start_id" not in public_start.payload
    assert "recovery_metadata" not in public_start.payload
    assert "provider_operation_progress" not in public_text.payload
    assert "provider_operation_progress" not in public_completed.payload

    stored_starting = next(
        event for event in stored_events if event.type == EventType.PROVIDER_OPERATION_STARTING
    )
    stored_start = next(
        event for event in stored_events if event.type == EventType.PROVIDER_OPERATION_STARTED
    )
    stored_text = next(event for event in stored_events if event.type == EventType.MODEL_TEXT_DELTA)
    stored_completed = next(
        event for event in stored_events if event.type == EventType.MODEL_COMPLETED
    )
    assert stored_start.payload["state_version"] == 1
    assert stored_start.payload["recovery_metadata"] == {"cursor": 0}
    assert stored_start.payload["provider"] == "reconnectable"
    assert stored_start.payload["source_run_epoch"] == 1
    assert stored_start.payload["model_step_id"]
    assert stored_start.payload["model_attempt_id"]
    assert stored_start.interaction_id
    assert stored_starting.payload["start_id"] == stored_start.payload["start_id"]
    assert stored_starting.payload["start_idempotency_support"] == "unsupported"
    assert provider.adapter.start_requests[0].idempotency_key == stored_start.payload["start_id"]
    assert (
        stored_text.payload["provider_operation_progress"]["stream_event"]["recovery_metadata"][
            "cursor"
        ]
        == 1
    )
    assert (
        stored_completed.payload["provider_operation_progress"]["stream_event"][
            "recovery_metadata"
        ]["cursor"]
        == 2
    )


def test_background_tool_call_without_id_persists_one_generated_identity() -> None:
    provider = _ReconnectableProvider(background=True)
    provider.adapter = _GeneratedToolIdAdapter()
    app = CayuApp(enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    async def run() -> tuple[list[Event], list[Message]]:
        await _collect_run_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="provider_operation_generated_tool_id",
                messages=[Message.text("user", "call a tool")],
                max_steps=1,
            ),
        )
        return (
            await app.session_store.load_events("provider_operation_generated_tool_id"),
            await app.session_store.load_transcript("provider_operation_generated_tool_id"),
        )

    stored_events, transcript = asyncio.run(run())

    progress = next(
        event for event in stored_events if event.type is EventType.PROVIDER_OPERATION_PROGRESS
    )
    persisted_payload = progress.payload["provider_operation_progress"]["stream_event"]["payload"]
    assert persisted_payload["id"] == progress.id
    tool_call = next(
        part for message in transcript for part in message.content if isinstance(part, ToolCallPart)
    )
    assert tool_call.tool_call_id == persisted_payload["id"]


def test_invalid_started_connection_is_closed_before_failure_publication() -> None:
    provider = _ReconnectableProvider(background=True)
    adapter = _InvalidStartConnectionAdapter()
    provider.adapter = adapter
    app = CayuApp(enable_logging=False, retry_policy=_two_attempt_retry_policy())
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    events = asyncio.run(
        _collect_run_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="invalid_started_connection_cleanup",
                messages=[Message.text("user", "hello")],
            ),
        )
    )

    assert adapter.start_calls == 1
    assert adapter.events.closed is True
    assert EventType.MODEL_RETRY not in {event.type for event in events}
    assert EventType.PROVIDER_OPERATION_STARTED not in {event.type for event in events}


def test_background_completion_closes_blocked_tail_before_publication() -> None:
    async def scenario() -> None:
        provider = _ReconnectableProvider(background=True)
        adapter = _TerminalBlockingStartAdapter()
        provider.adapter = adapter
        app = CayuApp(enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))

        events = await asyncio.wait_for(
            _collect_run_events(
                app,
                RunRequest(
                    agent_name="assistant",
                    session_id="background_terminal_blocked_tail",
                    messages=[Message.text("user", "hello")],
                ),
            ),
            timeout=1.0,
        )

        assert adapter.start_calls == 1
        assert adapter.events.closed is True
        completed = [event for event in events if event.type is EventType.MODEL_COMPLETED]
        assert len(completed) == 1
        assert completed[0].payload["usage_metrics"]["total_tokens"] == 5

    asyncio.run(scenario())


def test_background_cleanup_failure_preserves_non_turn_completion_and_usage() -> None:
    async def scenario() -> None:
        provider = _ReconnectableProvider(background=True)
        adapter = _TerminalBlockingStartAdapter(close_error=RuntimeError("provider close failed"))
        provider.adapter = adapter
        app = CayuApp(enable_logging=False, retry_policy=_two_attempt_retry_policy())
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))

        events = await _collect_run_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="background_terminal_cleanup_failure",
                messages=[Message.text("user", "hello")],
            ),
        )

        assert adapter.start_calls == 1
        assert adapter.events.closed is True
        assert EventType.MODEL_RETRY not in {event.type for event in events}
        completed = [event for event in events if event.type is EventType.MODEL_COMPLETED]
        assert len(completed) == 1
        assert completed[0].payload["usage_metrics"]["total_tokens"] == 5
        assert completed[0].payload["step_classification"]["type"] == "failed"
        transcript = await app.session_store.load_transcript("background_terminal_cleanup_failure")
        assert len(transcript) == 1
        assert transcript[0].role.value == "user"

    asyncio.run(scenario())


def test_invalid_terminal_cursor_preserves_non_turn_completion_and_usage() -> None:
    async def scenario() -> None:
        provider = _ReconnectableProvider(background=True)
        adapter = _TerminalCursorGapAdapter()
        provider.adapter = adapter
        app = CayuApp(enable_logging=False, retry_policy=_two_attempt_retry_policy())
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        session_id = "background_invalid_terminal_cursor"

        events = await _collect_run_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "hello")],
            ),
        )

        assert adapter.start_calls == 1
        assert EventType.MODEL_RETRY not in {event.type for event in events}
        stored = await app.session_store.load_events(session_id)
        completed = [event for event in stored if event.type is EventType.MODEL_COMPLETED]
        assert len(completed) == 1
        assert completed[0].payload["usage_metrics"]["total_tokens"] == 5
        assert completed[0].payload["step_classification"]["type"] == "failed"
        assert "provider_operation_progress" not in completed[0].payload
        transcript = await app.session_store.load_transcript(session_id)
        assert len(transcript) == 1
        assert transcript[0].role.value == "user"

    asyncio.run(scenario())


def test_model_stage_persists_secret_free_offline_recovery_context() -> None:
    identity = BillingIdentity(
        provider_name="reconnectable",
        resource_id="model-in-us-test-1",
        request_evidence={"region": "us-test-1"},
    )
    store = _CaptureStageStore()
    provider = _ContextualReconnectableProvider(identity)
    app = CayuApp(session_store=store, enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))
    metadata = {"traceparent": "00-" + "1" * 32 + "-" + "2" * 16 + "-01"}

    asyncio.run(
        _collect_run_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="provider_operation_recovery_context",
                messages=[Message.text("user", "hello")],
                metadata=metadata,
                max_steps=3,
                limits=RunLimits(max_tool_calls=2),
                retry_policy=RetryPolicy(max_attempts=2),
                thinking=ThinkingConfig(include_in_transcript=False),
            ),
        )
    )

    assert len(store.stage_requests) == 1
    context = ModelCompletionRecoveryContext.model_validate(
        store.stage_requests[0].intent["recovery_context"]
    )
    assert context.request_metadata == metadata
    assert context.max_steps == 3
    assert context.limits == RunLimits(max_tool_calls=2)
    assert context.retry_policy.max_attempts == 2
    assert context.thinking == ThinkingConfig(include_in_transcript=False)
    assert context.billing_identity == identity


@pytest.mark.parametrize("secret_location", ["value", "object_key"])
def test_background_dispatch_rejects_unrecoverable_secret_semantics_before_start(
    secret_location: str,
) -> None:
    secret = "provider-recovery-workload-secret"
    identity = BillingIdentity(
        provider_name="reconnectable",
        resource_id="model-in-us-test-1",
        request_evidence=({"opaque": secret} if secret_location == "value" else {secret: "safe"}),
    )
    store = _CaptureStageStore()
    provider = _ContextualReconnectableProvider(identity)
    app = CayuApp(
        session_store=store,
        secret_redactor=SecretRedactor(secret),
        enable_logging=False,
    )
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    events = asyncio.run(
        _collect_run_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id=f"provider_operation_secret_{secret_location}",
                messages=[Message.text("user", "hello")],
            ),
        )
    )

    assert events[-1].type is EventType.SESSION_FAILED
    assert provider.adapter.start_calls == 0
    assert store.stage_requests == []
    assert secret not in "".join(event.model_dump_json() for event in events)


def test_offline_recovery_context_is_bounded_before_stage_storage() -> None:
    with pytest.raises(ValidationError, match="interaction_id.*cannot be blank"):
        ModelCompletionRecoveryContext(interaction_id=" ")
    with pytest.raises(ValidationError, match="less than or equal to 256"):
        ModelCompletionRecoveryContext(max_steps=257)
    with pytest.raises(ValidationError, match="request_metadata cannot contain more than"):
        ModelCompletionRecoveryContext(
            request_metadata={
                f"key-{index}": index
                for index in range(MAX_MODEL_COMPLETION_RECOVERY_METADATA_ENTRIES + 1)
            }
        )
    with pytest.raises(ValidationError, match="budget_limits cannot contain more than"):
        ModelCompletionRecoveryContext.model_validate(
            {"budget_limits": [None] * (MAX_MODEL_COMPLETION_RECOVERY_BUDGET_LIMITS + 1)}
        )
    with pytest.raises(ValidationError, match="exceeds the durable byte limit"):
        ModelCompletionRecoveryContext(
            request_metadata={"payload": "x" * MAX_MODEL_COMPLETION_RECOVERY_CONTEXT_BYTES}
        )


def test_synchronous_dispatch_is_unchanged() -> None:
    provider = _SynchronousProvider()
    store = _CaptureStageStore()
    app = CayuApp(session_store=store, enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    events = asyncio.run(
        _collect_run_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="synchronous_provider",
                messages=[Message.text("user", "hello")],
            ),
        )
    )

    assert provider.stream_calls == 1
    assert EventType.PROVIDER_OPERATION_STARTED not in {event.type for event in events}
    recovery_context = ModelCompletionRecoveryContext.model_validate(
        store.stage_requests[0].intent["recovery_context"]
    )
    assert recovery_context.execution_profile_fingerprint is not None
    assert recovery_context.task_id is None


def test_operator_inspection_distinguishes_sync_and_in_progress() -> None:
    provider = _SynchronousProvider()
    app = CayuApp(enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    async def scenario():
        await _collect_run_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="inspection_sync",
                messages=[Message.text("user", "hello")],
            ),
        )
        sync = await inspect_provider_operation(app.session_store, "inspection_sync")
        await app.session_store.create(
            RunRequest(
                agent_name="assistant",
                session_id="inspection_in_progress",
                messages=[Message.text("user", "hello")],
            ),
            identity=SessionIdentity(provider_name="reconnectable", model="fake-model"),
        )
        await app.session_store.append_events(
            "inspection_in_progress",
            [
                _model_event(EventType.MODEL_STARTED, sequence_identity="attempt-a"),
                _model_event(
                    EventType.PROVIDER_OPERATION_STARTED,
                    sequence_identity="attempt-a",
                    payload={
                        "provider": "reconnectable",
                        "start_id": "provider-operation:attempt-a",
                        "state_version": 1,
                        "operation_id": "response_123",
                        "stream_protocol": "responses-v1",
                        "status": "in_progress",
                        "recovery_metadata": {"cursor": 0},
                    },
                ),
            ],
        )
        in_progress = await inspect_provider_operation(app.session_store, "inspection_in_progress")
        return sync, in_progress

    sync, in_progress = asyncio.run(scenario())

    assert sync.status is ProviderOperationInspectionStatus.SYNCHRONOUS
    assert sync.operation_id is None
    assert in_progress.status is ProviderOperationInspectionStatus.PROVIDER_OPERATION_IN_PROGRESS
    assert in_progress.operation_id == "response_123"
    assert in_progress.provider == "reconnectable"
    assert in_progress.stream_protocol == "responses-v1"


def test_operator_inspection_ignores_independently_scoped_compaction_completion() -> None:
    app = CayuApp(enable_logging=False)

    async def scenario():
        session_id = "inspection_after_compaction"
        await app.session_store.create(
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "hello")],
            ),
            identity=SessionIdentity(provider_name="reconnectable", model="fake-model"),
        )
        await app.session_store.append_events(
            session_id,
            [
                _model_event(
                    EventType.MODEL_STARTED,
                    sequence_identity="attempt-a",
                    session_id=session_id,
                ),
                _operation_event(operation_id="response_a", session_id=session_id),
                Event(
                    type=EventType.MODEL_COMPLETED,
                    session_id=session_id,
                    payload={
                        "purpose": "context_compaction",
                        "provider_name": "compactor",
                        "requested_model": "summary-model",
                        "model": "summary-model",
                        "model_step_id": "mstep_" + "b" * 32,
                        "model_attempt_id": "matt_" + "c" * 32,
                    },
                ),
            ],
        )
        return await inspect_provider_operation(app.session_store, session_id)

    inspection = asyncio.run(scenario())

    assert inspection.status is ProviderOperationInspectionStatus.PROVIDER_OPERATION_IN_PROGRESS
    assert inspection.operation_id == "response_a"


@pytest.mark.parametrize(
    ("fence_on", "expected_start_calls", "expected_cancel_calls"),
    [
        (EventType.PROVIDER_OPERATION_STARTING, 0, 0),
        (EventType.PROVIDER_OPERATION_STARTED, 1, 1),
    ],
)
def test_stale_run_owner_cannot_start_or_publish_provider_operation_identity(
    fence_on: EventType,
    expected_start_calls: int,
    expected_cancel_calls: int,
) -> None:
    store = _FenceOnEventStore(fence_on)
    provider = _ReconnectableProvider(background=True)
    app = CayuApp(session_store=store, enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    with pytest.raises(SessionRunFenced):
        asyncio.run(
            _collect_run_events(
                app,
                RunRequest(
                    agent_name="assistant",
                    session_id=f"stale_{fence_on.value}",
                    messages=[Message.text("user", "hello")],
                ),
            )
        )

    events = asyncio.run(store.load_events(f"stale_{fence_on.value}"))
    assert store.fenced
    assert provider.adapter.start_calls == expected_start_calls
    assert provider.adapter.cancel_calls == expected_cancel_calls
    assert EventType.PROVIDER_OPERATION_STARTED not in {event.type for event in events}


def test_ambiguous_provider_operation_start_is_never_retried() -> None:
    provider = _ReconnectableProvider(
        background=True,
        start_error=TimeoutError("provider submission timed out"),
    )
    app = CayuApp(
        enable_logging=False,
        retry_policy=_two_attempt_retry_policy(),
    )
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    events = asyncio.run(
        _collect_run_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="ambiguous_provider_start",
                messages=[Message.text("user", "hello")],
            ),
        )
    )

    assert provider.adapter.start_calls == 1
    assert EventType.MODEL_RETRY not in {event.type for event in events}
    interrupted = next(event for event in events if event.type == EventType.SESSION_INTERRUPTED)
    assert interrupted.payload["recovery_reason"] == "ambiguous_submission"
    stored = asyncio.run(app.session_store.load_events("ambiguous_provider_start"))
    assert sum(event.type == EventType.PROVIDER_OPERATION_STARTING for event in stored) == 1


@pytest.mark.parametrize(
    "adapter_type",
    [
        _FailingOperationStreamAdapter,
        _ErrorEventOperationStreamAdapter,
        _EmptyOperationStreamAdapter,
    ],
)
def test_started_provider_operation_stream_failure_is_never_retried(adapter_type) -> None:
    provider = _ReconnectableProvider(background=True)
    adapter = adapter_type()
    provider.adapter = adapter
    app = CayuApp(enable_logging=False, retry_policy=_two_attempt_retry_policy())
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    events = asyncio.run(
        _collect_run_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="failed_background_stream",
                messages=[Message.text("user", "hello")],
            ),
        )
    )

    assert adapter.start_calls == 1
    assert EventType.MODEL_RETRY not in {event.type for event in events}
    assert EventType.MODEL_ATTEMPT_DISCARDED not in {event.type for event in events}
    for error in (event for event in events if event.type is EventType.MODEL_ERROR):
        assert error.payload["max_attempts"] == 2
        assert error.payload["effective_max_attempts"] == error.payload["attempt"]
        assert "reason" not in error.payload
    inspection = asyncio.run(
        inspect_provider_operation(app.session_store, "failed_background_stream")
    )
    assert inspection.status is ProviderOperationInspectionStatus.PROVIDER_OPERATION_IN_PROGRESS


def test_started_provider_operation_overflow_does_not_dispatch_recovery_attempt() -> None:
    provider = _ReconnectableProvider(background=True)
    adapter = _OverflowingOperationStreamAdapter()
    provider.adapter = adapter
    app = CayuApp(enable_logging=False, retry_policy=_two_attempt_retry_policy())
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        context_overflow_policy=RecentTurnsContextPolicy(max_user_turns=1),
    )

    events = asyncio.run(
        _collect_run_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="background_context_overflow",
                messages=[Message.text("user", "hello")],
            ),
        )
    )

    assert adapter.start_calls == 1
    assert EventType.MODEL_RETRY not in {event.type for event in events}
    assert EventType.CONTEXT_OVERFLOW_RECOVERING not in {event.type for event in events}


def test_cursor_bearing_overflow_error_is_committed_once_before_terminal_failure() -> None:
    async def scenario() -> None:
        provider = _ReconnectableProvider(background=True)
        adapter = _CursorOverflowErrorAdapter()
        provider.adapter = adapter
        app = CayuApp(enable_logging=False, retry_policy=_two_attempt_retry_policy())
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_overflow_policy=RecentTurnsContextPolicy(max_user_turns=1),
        )
        session_id = "background_cursor_context_overflow"

        events = await _collect_run_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "hello")],
            ),
        )

        assert adapter.start_calls == 1
        assert EventType.MODEL_RETRY not in {event.type for event in events}
        assert EventType.CONTEXT_OVERFLOW_RECOVERING not in {event.type for event in events}
        stored = await app.session_store.load_events(session_id)
        errors = [event for event in stored if event.type is EventType.MODEL_ERROR]
        assert len(errors) == 1
        assert (
            errors[0].payload["provider_operation_progress"]["stream_event"]["recovery_metadata"][
                "cursor"
            ]
            == 1
        )

    asyncio.run(scenario())


def test_disabled_retry_preserves_authoritative_provider_failure_cause() -> None:
    overflow = ModelContextOverflowError(
        "provider context limit exceeded",
        provider="reconnectable",
        status_code=400,
        error_code="context_length_exceeded",
    )
    public_failure = ModelProviderError(
        "background operation failed after dispatch",
        provider="reconnectable",
        retryable=False,
    )
    assert set_exception_cause(public_failure, overflow)
    attempt_failure = ModelAttemptFailed(
        message=str(public_failure),
        payload={"error": str(public_failure)},
        emitted_error_event=False,
        cause=public_failure,
        automatic_retry_disabled=True,
    )

    with pytest.raises(ModelProviderError) as captured:
        _raise_terminal_model_attempt_failure(attempt_failure)

    assert captured.value is public_failure
    assert exception_cause(captured.value) is overflow


def test_provider_owned_operation_identity_cannot_embed_a_workload_secret() -> None:
    secret = "workload-secret-canary-ABCDEFGHIJKLMNOP"
    provider = _ReconnectableProvider(background=True)
    adapter = _ProviderIdentityAdapter(operation_id=f"op-{secret}-tail")
    provider.adapter = adapter
    app = CayuApp(enable_logging=False, secret_redactor=SecretRedactor(secret))
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    events = asyncio.run(
        _collect_run_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="secret_provider_identity",
                messages=[Message.text("user", "hello")],
            ),
        )
    )

    assert adapter.start_calls == 1
    assert adapter.cancel_calls == 1
    stored = asyncio.run(app.session_store.load_events("secret_provider_identity"))
    assert EventType.PROVIDER_OPERATION_STARTED not in {event.type for event in stored}
    assert secret not in repr(events) + repr(stored)


def test_session_interruption_cancels_the_exact_durable_provider_operation() -> None:
    async def scenario():
        provider = _ReconnectableProvider(background=True)
        adapter = _InterruptibleOperationAdapter()
        provider.adapter = adapter
        app = CayuApp(enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))

        async def run_session() -> list[Event]:
            return [
                event
                async for event in app.run(
                    RunRequest(
                        agent_name="assistant",
                        session_id="interrupt_provider_operation",
                        messages=[Message.text("user", "wait for cancellation")],
                    )
                )
            ]

        run_task = asyncio.create_task(run_session())
        await asyncio.wait_for(adapter.stream_started.wait(), timeout=1)
        interrupt_events = [
            event
            async for event in app.interrupt_session(
                InterruptSessionRequest(
                    session_id="interrupt_provider_operation",
                    reason="operator requested cancellation",
                )
            )
        ]
        run_events = await run_task
        inspection = await inspect_provider_operation(
            app.session_store,
            "interrupt_provider_operation",
        )
        return run_events, interrupt_events, adapter, inspection

    run_events, interrupt_events, adapter, inspection = asyncio.run(scenario())

    assert adapter.cancel_calls == 1
    assert adapter.cancelled_states == [
        ProviderOperationState(
            operation_id="response_interruptible",
            stream_protocol="responses-v1",
            recovery_metadata={"cursor": 0},
        )
    ]
    assert EventType.SESSION_INTERRUPTED in {event.type for event in run_events}
    assert [event.type for event in interrupt_events] == [EventType.SESSION_INTERRUPTED]
    assert inspection.cancellation_status is ProviderOperationCancellationStatus.CANCELLED
    assert inspection.accounting_status is ProviderOperationAccountingStatus.NOT_APPLICABLE
    assert inspection.reservation_count == 0


def test_provider_completion_before_interruption_is_never_cancelled() -> None:
    async def scenario() -> tuple[list[Event], _ReconnectableAdapter]:
        provider = _ReconnectableProvider(background=True)
        app = CayuApp(enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        events = await _collect_run_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="provider_completion_wins_interruption_race",
                messages=[Message.text("user", "finish first")],
            ),
        )
        with pytest.raises(ValueError, match="cannot be interrupted"):
            async for _event in app.interrupt_session(
                InterruptSessionRequest(
                    session_id="provider_completion_wins_interruption_race",
                    reason="too late",
                )
            ):
                pass
        return events, provider.adapter

    events, adapter = asyncio.run(scenario())

    assert sum(event.type is EventType.MODEL_COMPLETED for event in events) == 1
    assert adapter.cancel_calls == 0


def test_provider_completion_winning_live_cancellation_is_reconciled() -> None:
    async def scenario() -> tuple[
        list[Event],
        list[Event],
        list[Event],
        _CompletionWinsLiveCancellationAdapter,
    ]:
        provider = _ReconnectableProvider(background=True)
        adapter = _CompletionWinsLiveCancellationAdapter()
        provider.adapter = adapter
        app = CayuApp(enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        session_id = "provider_completion_wins_live_cancellation"

        run_task = asyncio.create_task(
            _collect_run_events(
                app,
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "complete while cancellation races")],
                ),
            )
        )
        await asyncio.wait_for(adapter.stream_started.wait(), timeout=1)
        interrupt_events = [
            event
            async for event in app.interrupt_session(
                InterruptSessionRequest(
                    session_id=session_id,
                    reason="provider may have completed",
                )
            )
        ]
        run_events = await run_task
        durable_events = await app.session_store.load_events(session_id)
        return run_events, interrupt_events, durable_events, adapter

    run_events, interrupt_events, durable_events, adapter = asyncio.run(scenario())

    assert adapter.cancel_calls == 1
    assert sum(event.type is EventType.MODEL_COMPLETED for event in durable_events) == 1
    assert EventType.SESSION_INTERRUPTED in {event.type for event in run_events}
    assert [event.type for event in interrupt_events] == [EventType.SESSION_INTERRUPTED]
    completed = next(event for event in durable_events if event.type is EventType.MODEL_COMPLETED)
    assert completed.interaction_id is not None


def test_cancellation_during_provider_start_waits_for_identity_before_releasing_run() -> None:
    async def scenario() -> tuple[bool, int, int, ProviderOperationInspectionStatus]:
        provider = _ReconnectableProvider(background=True)
        adapter = _BlockingStartAdapter()
        provider.adapter = adapter
        app = CayuApp(enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        task = asyncio.create_task(
            _collect_run_events(
                app,
                RunRequest(
                    agent_name="assistant",
                    session_id="cancelled_provider_start",
                    messages=[Message.text("user", "hello")],
                ),
            )
        )
        await asyncio.wait_for(adapter.start_entered.wait(), timeout=1)
        task.cancel("cancel after provider dispatch")
        assert task.cancelling() == 1
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert not task.done()
        adapter.start_release.set()
        with pytest.raises(asyncio.CancelledError, match="Provider operation cancelled"):
            await task
        inspection = await inspect_provider_operation(
            app.session_store,
            "cancelled_provider_start",
        )
        return task.cancelled(), task.cancelling(), adapter.start_calls, inspection.status

    cancelled, cancelling, start_calls, status = asyncio.run(scenario())

    assert cancelled
    assert cancelling == 0
    assert start_calls == 1
    assert status is ProviderOperationInspectionStatus.PROVIDER_OPERATION_IN_PROGRESS


def test_cancellation_during_provider_start_is_raised_before_started_event_yield() -> None:
    async def scenario() -> tuple[bool, list[EventType]]:
        provider = _ReconnectableProvider(background=True)
        adapter = _BlockingStartAdapter()
        provider.adapter = adapter
        app = CayuApp(enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        seen: list[EventType] = []

        async def collect_until_started() -> None:
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="cancel_before_started_yield",
                    messages=[Message.text("user", "hello")],
                )
            ):
                seen.append(event.type)
                if event.type == EventType.PROVIDER_OPERATION_STARTED:
                    return

        task = asyncio.create_task(collect_until_started())
        await asyncio.wait_for(adapter.start_entered.wait(), timeout=1)
        task.cancel("cancel before started event delivery")
        adapter.start_release.set()
        with pytest.raises(asyncio.CancelledError, match="Provider operation cancelled"):
            await task
        return task.cancelled(), seen

    cancelled, seen = asyncio.run(scenario())

    assert cancelled
    assert EventType.PROVIDER_OPERATION_STARTED not in seen


def test_cancellation_during_unsettled_provider_start_is_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> tuple[bool, int, ProviderOperationInspectionStatus]:
        monkeypatch.setattr(
            "cayu.runtime._model_step_executor._PROVIDER_OPERATION_START_SETTLEMENT_TIMEOUT_SECONDS",
            0.0,
        )
        provider = _ReconnectableProvider(background=True)
        adapter = _CancellationResistantStartAdapter()
        provider.adapter = adapter
        app = CayuApp(enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        task = asyncio.create_task(
            _collect_run_events(
                app,
                RunRequest(
                    agent_name="assistant",
                    session_id="cancel_unsettled_provider_start",
                    messages=[Message.text("user", "hello")],
                ),
            )
        )
        await asyncio.wait_for(adapter.start_entered.wait(), timeout=1)
        task.cancel("cancel unsettled provider start")
        with pytest.raises(asyncio.CancelledError, match="Provider operation cancelled"):
            await asyncio.wait_for(task, timeout=1)
        await asyncio.wait_for(adapter.local_cancellation_observed.wait(), timeout=1)
        inspection = await inspect_provider_operation(
            app.session_store,
            "cancel_unsettled_provider_start",
        )
        adapter.start_release.set()
        await asyncio.sleep(0)
        return task.cancelled(), adapter.start_calls, inspection.status

    cancelled, start_calls, status = asyncio.run(scenario())

    assert cancelled
    assert start_calls == 1
    assert status is ProviderOperationInspectionStatus.AMBIGUOUS_SUBMISSION


def test_late_successful_start_acknowledgement_cancels_exact_returned_operation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> tuple[
        int,
        list[ProviderOperationState],
        ProviderOperationInspectionStatus,
        list[Event],
    ]:
        monkeypatch.setattr(
            "cayu.runtime._model_step_executor._PROVIDER_OPERATION_START_SETTLEMENT_TIMEOUT_SECONDS",
            0.0,
        )
        provider = _ReconnectableProvider(background=True)
        adapter = _LateSuccessfulStartAdapter()
        provider.adapter = adapter
        app = CayuApp(enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        task = asyncio.create_task(
            _collect_run_events(
                app,
                RunRequest(
                    agent_name="assistant",
                    session_id="late_successful_provider_start",
                    messages=[Message.text("user", "hello")],
                ),
            )
        )
        await asyncio.wait_for(adapter.start_entered.wait(), timeout=1)
        task.cancel("cancel before late start acknowledgement")
        with pytest.raises(asyncio.CancelledError):
            await task
        await asyncio.wait_for(adapter.local_cancellation_observed.wait(), timeout=1)
        adapter.start_release.set()
        await asyncio.wait_for(adapter.late_cancel_observed.wait(), timeout=1)
        inspection = await inspect_provider_operation(
            app.session_store,
            "late_successful_provider_start",
        )
        stored = await app.session_store.load_events("late_successful_provider_start")
        return (
            adapter.cancel_calls,
            adapter.cancelled_states,
            inspection.status,
            stored,
        )

    cancel_calls, cancelled_states, status, stored = asyncio.run(scenario())

    assert cancel_calls == 1
    assert cancelled_states == [
        ProviderOperationState(
            operation_id="response_late",
            stream_protocol="responses-v1",
        )
    ]
    assert status is ProviderOperationInspectionStatus.PROVIDER_OPERATION_UNAVAILABLE
    reconciled = [event for event in stored if event.type == EventType.PROVIDER_OPERATION_STARTED]
    assert len(reconciled) == 1
    assert reconciled[0].payload["operation_id"] == "response_late"
    assert reconciled[0].payload["status"] == ProviderOperationStatus.CANCELLED.value


def test_late_invalid_start_acknowledgement_closes_returned_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        monkeypatch.setattr(
            "cayu.runtime._model_step_executor._PROVIDER_OPERATION_START_SETTLEMENT_TIMEOUT_SECONDS",
            0.0,
        )
        provider = _ReconnectableProvider(background=True)
        adapter = _LateInvalidStartConnectionAdapter()
        provider.adapter = adapter
        app = CayuApp(enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        session_id = "late_invalid_provider_start"
        task = asyncio.create_task(
            _collect_run_events(
                app,
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "hello")],
                ),
            )
        )
        await asyncio.wait_for(adapter.start_entered.wait(), timeout=1)
        task.cancel("cancel before invalid late start acknowledgement")
        with pytest.raises(asyncio.CancelledError):
            await task
        await asyncio.wait_for(adapter.local_cancellation_observed.wait(), timeout=1)
        adapter.start_release.set()
        await asyncio.wait_for(adapter.events.closed_event.wait(), timeout=1)

        stored = await app.session_store.load_events(session_id)
        assert adapter.events.closed is True
        assert not any(event.type is EventType.PROVIDER_OPERATION_STARTED for event in stored)

    asyncio.run(scenario())


def test_late_start_acknowledgement_cannot_publish_after_run_epoch_moves(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> tuple[ProviderOperationInspectionStatus, list[Event]]:
        monkeypatch.setattr(
            "cayu.runtime._model_step_executor._PROVIDER_OPERATION_START_SETTLEMENT_TIMEOUT_SECONDS",
            0.0,
        )
        provider = _ReconnectableProvider(background=True)
        adapter = _LateSuccessfulStartAdapter()
        provider.adapter = adapter
        app = CayuApp(enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        session_id = "late_start_after_newer_epoch"
        task = asyncio.create_task(
            _collect_run_events(
                app,
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "hello")],
                ),
            )
        )
        await asyncio.wait_for(adapter.start_entered.wait(), timeout=1)
        task.cancel("cancel before late start acknowledgement")
        with pytest.raises(asyncio.CancelledError):
            await task
        await asyncio.wait_for(adapter.local_cancellation_observed.wait(), timeout=1)

        prior = await app.session_store.load(session_id)
        assert prior is not None
        newer = await app.session_store.transition_status(
            session_id,
            from_statuses={prior.status},
            to_status=SessionStatus.RUNNING,
        )
        assert newer.run_epoch == prior.run_epoch + 1
        stored = await app.session_store.load_events(session_id)
        earlier_started = next(event for event in stored if event.type == EventType.MODEL_STARTED)
        await app.session_store.append_event(
            session_id,
            Event(
                type=EventType.MODEL_STARTED,
                session_id=session_id,
                interaction_id=earlier_started.interaction_id,
                agent_name=earlier_started.agent_name,
                environment_name=earlier_started.environment_name,
                payload={
                    **earlier_started.payload,
                    "model_step_id": "mstep_" + ("c" * 32),
                    "model_attempt_id": "matt_" + ("d" * 32),
                },
            ),
        )
        await app.session_store.transition_status(
            session_id,
            from_statuses={SessionStatus.RUNNING},
            to_status=SessionStatus.FAILED,
        )

        adapter.start_release.set()
        await asyncio.wait_for(adapter.late_cancel_observed.wait(), timeout=1)
        for _ in range(10):
            await asyncio.sleep(0)

        inspection = await inspect_provider_operation(app.session_store, session_id)
        return inspection.status, await app.session_store.load_events(session_id)

    status, stored = asyncio.run(scenario())

    assert status is ProviderOperationInspectionStatus.SYNCHRONOUS
    assert not any(
        event.type == EventType.PROVIDER_OPERATION_STARTED
        and event.payload.get("operation_id") == "response_late"
        for event in stored
    )


def test_late_start_cancellation_failure_preserves_exact_in_progress_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> tuple[ProviderOperationInspectionStatus, str | None]:
        monkeypatch.setattr(
            "cayu.runtime._model_step_executor._PROVIDER_OPERATION_START_SETTLEMENT_TIMEOUT_SECONDS",
            0.0,
        )
        provider = _ReconnectableProvider(background=True)
        adapter = _LateFailingCancelAdapter()
        provider.adapter = adapter
        app = CayuApp(enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        task = asyncio.create_task(
            _collect_run_events(
                app,
                RunRequest(
                    agent_name="assistant",
                    session_id="late_failed_provider_cancellation",
                    messages=[Message.text("user", "hello")],
                ),
            )
        )
        await asyncio.wait_for(adapter.start_entered.wait(), timeout=1)
        task.cancel("cancel before late failed cancellation")
        with pytest.raises(asyncio.CancelledError):
            await task
        adapter.start_release.set()
        await asyncio.wait_for(adapter.late_cancel_observed.wait(), timeout=1)
        for _ in range(10):
            inspection = await inspect_provider_operation(
                app.session_store,
                "late_failed_provider_cancellation",
            )
            if inspection.operation_id is not None:
                return inspection.status, inspection.operation_id
            await asyncio.sleep(0)
        return inspection.status, inspection.operation_id

    status, operation_id = asyncio.run(scenario())

    assert status is ProviderOperationInspectionStatus.PROVIDER_OPERATION_IN_PROGRESS
    assert operation_id == "response_late"


def test_store_cancellation_during_started_identity_publication_fails_closed() -> None:
    store = _CancelOnEventStore(EventType.PROVIDER_OPERATION_STARTED)
    provider = _ReconnectableProvider(background=True)
    app = CayuApp(session_store=store, enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    events = asyncio.run(
        _collect_run_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="cancelled_started_publication",
                messages=[Message.text("user", "hello")],
            ),
        )
    )

    assert provider.adapter.start_calls == 1
    assert provider.adapter.cancel_calls == 1
    assert EventType.MODEL_RETRY not in {event.type for event in events}
    interrupted = next(event for event in events if event.type == EventType.SESSION_INTERRUPTED)
    assert interrupted.payload["recovery_reason"] == "ambiguous_submission"


def test_started_identity_timeout_without_commit_cleans_up_and_never_retries() -> None:
    store = _FailBeforeCommitOnEventStore(
        EventType.PROVIDER_OPERATION_STARTED,
        TimeoutError("started identity append timed out"),
    )
    provider = _ReconnectableProvider(background=True)
    app = CayuApp(
        session_store=store,
        enable_logging=False,
        retry_policy=_two_attempt_retry_policy(),
    )
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    events = asyncio.run(
        _collect_run_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="started_identity_timeout",
                messages=[Message.text("user", "hello")],
            ),
        )
    )

    assert provider.adapter.start_calls == 1
    assert provider.adapter.cancel_calls == 1
    assert EventType.MODEL_RETRY not in {event.type for event in events}
    stored = asyncio.run(app.session_store.load_events("started_identity_timeout"))
    assert EventType.PROVIDER_OPERATION_STARTED not in {event.type for event in stored}


def test_synchronous_cleanup_failure_never_retries_provider_start() -> None:
    store = _FailBeforeCommitOnEventStore(
        EventType.PROVIDER_OPERATION_STARTED,
        TimeoutError("started identity append timed out"),
    )
    provider = _ReconnectableProvider(background=True)
    adapter = _SynchronousFailingCancelAdapter()
    provider.adapter = adapter
    app = CayuApp(
        session_store=store,
        enable_logging=False,
        retry_policy=_two_attempt_retry_policy(),
    )
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    events = asyncio.run(
        _collect_run_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="synchronous_cleanup_failure",
                messages=[Message.text("user", "hello")],
            ),
        )
    )

    assert adapter.start_calls == 1
    assert adapter.cancel_calls == 1
    assert EventType.MODEL_RETRY not in {event.type for event in events}


def test_cleanup_failure_does_not_invoke_extension_exception_cause_accessors() -> None:
    store = _FailBeforeCommitOnEventStore(
        EventType.PROVIDER_OPERATION_STARTED,
        _HostileCauseError("started identity append failed"),
    )
    provider = _ReconnectableProvider(background=True)
    adapter = _SynchronousFailingCancelAdapter()
    provider.adapter = adapter
    app = CayuApp(session_store=store, enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    events = asyncio.run(
        _collect_run_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="hostile_cleanup_failure_cause",
                messages=[Message.text("user", "hello")],
            ),
        )
    )

    assert adapter.start_calls == 1
    assert adapter.cancel_calls == 1
    interrupted = next(event for event in events if event.type == EventType.SESSION_INTERRUPTED)
    assert interrupted.payload["recovery_reason"] == "ambiguous_submission"


def test_post_commit_delivery_failure_leaves_operation_recoverable_without_retry() -> None:
    store = _FailClaimAfterStartedCommitStore()
    provider = _ReconnectableProvider(background=True)
    app = CayuApp(
        session_store=store,
        enable_logging=False,
        retry_policy=_two_attempt_retry_policy(),
    )
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    events = asyncio.run(
        _collect_run_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="post_commit_delivery_failure",
                messages=[Message.text("user", "hello")],
            ),
        )
    )

    assert provider.adapter.start_calls == 1
    assert provider.adapter.cancel_calls == 0
    assert EventType.MODEL_RETRY not in {event.type for event in events}
    assert EventType.PROVIDER_OPERATION_STARTED not in {event.type for event in events}
    stored = asyncio.run(app.session_store.load_events("post_commit_delivery_failure"))
    assert EventType.PROVIDER_OPERATION_STARTED in {event.type for event in stored}
    inspection = asyncio.run(
        inspect_provider_operation(app.session_store, "post_commit_delivery_failure")
    )
    assert inspection.status is ProviderOperationInspectionStatus.PROVIDER_OPERATION_IN_PROGRESS


def test_unavailable_started_identity_readback_never_cancels_or_retries() -> None:
    store = _UnverifiableStartedCommitStore()
    provider = _ReconnectableProvider(background=True)
    app = CayuApp(
        session_store=store,
        enable_logging=False,
        retry_policy=_two_attempt_retry_policy(),
    )
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    events = asyncio.run(
        _collect_run_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="unverifiable_started_identity",
                messages=[Message.text("user", "hello")],
            ),
        )
    )

    assert provider.adapter.start_calls == 1
    assert provider.adapter.cancel_calls == 0
    assert EventType.MODEL_RETRY not in {event.type for event in events}


def test_store_cancellation_during_started_identity_readback_fails_closed() -> None:
    store = _CancelOuterStartedReadbackStore()
    provider = _ReconnectableProvider(background=True)
    app = CayuApp(
        session_store=store,
        enable_logging=False,
        retry_policy=_two_attempt_retry_policy(),
    )
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    events = asyncio.run(
        _collect_run_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="cancelled_started_readback",
                messages=[Message.text("user", "hello")],
            ),
        )
    )

    assert provider.adapter.start_calls == 1
    assert provider.adapter.cancel_calls == 0
    assert EventType.MODEL_RETRY not in {event.type for event in events}
    interrupted = next(event for event in events if event.type == EventType.SESSION_INTERRUPTED)
    assert interrupted.payload["recovery_reason"] == "ambiguous_submission"


def test_cancellation_during_definite_absence_cleanup_propagates() -> None:
    async def scenario() -> tuple[int, int]:
        store = _FailBeforeCommitOnEventStore(
            EventType.PROVIDER_OPERATION_STARTED,
            TimeoutError("started identity append timed out"),
        )
        provider = _ReconnectableProvider(background=True)
        blocking_adapter = _BlockingCancelAdapter()
        provider.adapter = blocking_adapter
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))

        run_task = asyncio.create_task(
            _collect_run_events(
                app,
                RunRequest(
                    agent_name="assistant",
                    session_id="cancelled_started_cleanup",
                    messages=[Message.text("user", "hello")],
                ),
            )
        )
        await asyncio.wait_for(blocking_adapter.cancel_entered.wait(), timeout=1)
        run_task.cancel("caller cancelled during provider cleanup")
        blocking_adapter.cancel_release.set()
        with pytest.raises(asyncio.CancelledError, match="Provider operation cancelled"):
            await run_task
        assert run_task.cancelled()
        return blocking_adapter.start_calls, blocking_adapter.cancel_calls

    start_calls, cancel_calls = asyncio.run(scenario())

    assert start_calls == 1
    assert cancel_calls == 1


def test_timed_out_provider_cancellation_releases_with_uncertain_cleanup_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> tuple[bool, int]:
        monkeypatch.setattr(
            "cayu.runtime._model_step_executor._PROVIDER_OPERATION_START_CLEANUP_TIMEOUT_SECONDS",
            0.0,
        )
        store = _FailBeforeCommitOnEventStore(
            EventType.PROVIDER_OPERATION_STARTED,
            TimeoutError("started identity append timed out"),
        )
        provider = _ReconnectableProvider(background=True)
        adapter = _CancellationResistantCancelAdapter()
        provider.adapter = adapter
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        task = asyncio.create_task(
            _collect_run_events(
                app,
                RunRequest(
                    agent_name="assistant",
                    session_id="timed_out_provider_cancellation",
                    messages=[Message.text("user", "hello")],
                ),
            )
        )
        await asyncio.wait_for(adapter.local_cancellation_observed.wait(), timeout=1)
        await asyncio.wait_for(task, timeout=1)
        assert task.done()
        adapter.cancel_release.set()
        await asyncio.sleep(0)
        return task.done(), adapter.cancel_calls

    done, cancel_calls = asyncio.run(scenario())

    assert done
    assert cancel_calls == 1


def test_operator_inspection_rejects_contradictory_operation_identity() -> None:
    app = CayuApp(enable_logging=False)

    async def scenario() -> None:
        await app.session_store.create(
            RunRequest(
                agent_name="assistant",
                session_id="contradictory_operation",
                messages=[Message.text("user", "hello")],
            ),
            identity=SessionIdentity(provider_name="reconnectable", model="fake-model"),
        )
        await app.session_store.append_events(
            "contradictory_operation",
            [
                _model_event(
                    EventType.MODEL_STARTED,
                    sequence_identity="attempt-a",
                    session_id="contradictory_operation",
                ),
                _operation_event(
                    operation_id="response_a",
                    session_id="contradictory_operation",
                ),
                _operation_event(
                    operation_id="response_b",
                    session_id="contradictory_operation",
                ),
            ],
        )
        with pytest.raises(ProviderOperationEvidenceError, match="contradictory"):
            await inspect_provider_operation(app.session_store, "contradictory_operation")

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "order", ["valid_then_mismatched", "mismatched_then_valid", "mismatched_only"]
)
def test_operator_inspection_rejects_mismatched_complete_owning_identity(order: str) -> None:
    async def scenario() -> None:
        session_id = f"mismatched_scope_{order}"
        app = CayuApp(enable_logging=False)
        await app.session_store.create(
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "hello")],
            ),
            identity=SessionIdentity(provider_name="reconnectable", model="fake-model"),
        )
        valid = _operation_event(operation_id="response_a", session_id=session_id)
        mismatched = valid.model_copy(
            update={
                "id": str(uuid4()),
                "payload": {
                    **valid.payload,
                    "model_step_id": "mstep_" + "b" * 32,
                    "source_run_epoch": 2,
                },
            },
            deep=True,
        )
        operation_events = {
            "valid_then_mismatched": [valid, mismatched],
            "mismatched_then_valid": [mismatched, valid],
            "mismatched_only": [mismatched],
        }[order]
        await app.session_store.append_events(
            session_id,
            [
                _model_event(
                    EventType.MODEL_STARTED,
                    sequence_identity="attempt-a",
                    session_id=session_id,
                ),
                *operation_events,
            ],
        )
        with pytest.raises(ProviderOperationEvidenceError, match="mismatched identity"):
            await inspect_provider_operation(app.session_store, session_id)

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("interaction_id", "interaction-b"),
        ("provider", "other-provider"),
        ("model", "other-model"),
        ("step", 2),
        ("attempt", 2),
        ("max_attempts", 3),
        ("model_step_id", "mstep_" + "b" * 32),
        ("model_attempt_id", "attempt-b"),
    ],
)
def test_operator_inspection_validates_each_owning_identity_field(
    field: str,
    value: object,
) -> None:
    async def scenario() -> None:
        session_id = f"mismatched_{field}"
        app = CayuApp(enable_logging=False)
        await app.session_store.create(
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "hello")],
            ),
            identity=SessionIdentity(provider_name="reconnectable", model="fake-model"),
        )
        operation = _operation_event(operation_id="response_a", session_id=session_id)
        update = {"id": str(uuid4())}
        if field == "interaction_id":
            update["interaction_id"] = value
        else:
            update["payload"] = {**operation.payload, field: value}
        mismatched = operation.model_copy(update=update, deep=True)
        await app.session_store.append_events(
            session_id,
            [
                _model_event(
                    EventType.MODEL_STARTED,
                    sequence_identity="attempt-a",
                    session_id=session_id,
                ),
                mismatched,
            ],
        )
        with pytest.raises(ProviderOperationEvidenceError):
            await inspect_provider_operation(app.session_store, session_id)

    asyncio.run(scenario())


def test_operator_inspection_rejects_contradictory_provider_operation_epochs() -> None:
    async def scenario() -> None:
        session_id = "contradictory_provider_epochs"
        app = CayuApp(enable_logging=False)
        await app.session_store.create(
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "hello")],
            ),
            identity=SessionIdentity(provider_name="reconnectable", model="fake-model"),
        )
        await app.session_store.append_events(
            session_id,
            [
                _model_event(
                    EventType.MODEL_STARTED,
                    sequence_identity="attempt-a",
                    session_id=session_id,
                ),
                _model_event(
                    EventType.PROVIDER_OPERATION_STARTING,
                    sequence_identity="attempt-a",
                    session_id=session_id,
                    payload={"start_id": "provider-operation:attempt-a"},
                ),
                _operation_event(operation_id="response_a", session_id=session_id).model_copy(
                    update={
                        "payload": {
                            **_operation_event(
                                operation_id="response_a",
                                session_id=session_id,
                            ).payload,
                            "source_run_epoch": 2,
                        }
                    },
                    deep=True,
                ),
            ],
        )
        with pytest.raises(ProviderOperationEvidenceError, match="run epochs"):
            await inspect_provider_operation(app.session_store, session_id)

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "order",
    ["valid_then_conflicting", "conflicting_then_valid", "conflicting_only"],
)
def test_operator_inspection_rejects_conflicting_completion_scope(order: str) -> None:
    async def scenario() -> None:
        session_id = f"conflicting_completion_scope_{order}"
        app = CayuApp(enable_logging=False)
        await app.session_store.create(
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "hello")],
            ),
            identity=SessionIdentity(provider_name="reconnectable", model="fake-model"),
        )
        valid = _model_event(
            EventType.MODEL_COMPLETED,
            sequence_identity="attempt-a",
            session_id=session_id,
        )
        conflicting = valid.model_copy(
            update={
                "id": str(uuid4()),
                "payload": {
                    **valid.payload,
                    "provider_name": "other-provider",
                    "requested_model": "other-model",
                },
            },
            deep=True,
        )
        completions = {
            "valid_then_conflicting": [valid, conflicting],
            "conflicting_then_valid": [conflicting, valid],
            "conflicting_only": [conflicting],
        }[order]
        await app.session_store.append_events(
            session_id,
            [
                _model_event(
                    EventType.MODEL_STARTED,
                    sequence_identity="attempt-a",
                    session_id=session_id,
                ),
                _operation_event(operation_id="response_a", session_id=session_id),
                *completions,
            ],
        )
        with pytest.raises(ProviderOperationEvidenceError, match="different provider or model"):
            await inspect_provider_operation(app.session_store, session_id)

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("include_started", "terminal_type", "expected_status"),
    [
        (False, None, ProviderOperationInspectionStatus.AMBIGUOUS_SUBMISSION),
        (
            False,
            EventType.MODEL_ERROR,
            ProviderOperationInspectionStatus.AMBIGUOUS_SUBMISSION,
        ),
        (
            True,
            EventType.MODEL_ERROR,
            ProviderOperationInspectionStatus.PROVIDER_OPERATION_IN_PROGRESS,
        ),
        (
            True,
            EventType.MODEL_COMPLETED,
            ProviderOperationInspectionStatus.PROVIDER_OPERATION_RECONCILED,
        ),
    ],
)
def test_operator_inspection_classifies_ambiguous_start_and_terminal_error(
    include_started: bool,
    terminal_type: EventType | None,
    expected_status: ProviderOperationInspectionStatus,
) -> None:
    async def scenario():
        session_id = (
            "inspection_ambiguous_start" if terminal_type is None else "inspection_terminal_error"
        )
        app = CayuApp(enable_logging=False)
        await app.session_store.create(
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "hello")],
            ),
            identity=SessionIdentity(provider_name="reconnectable", model="fake-model"),
        )
        events = [
            _model_event(
                EventType.MODEL_STARTED,
                sequence_identity="attempt-a",
                session_id=session_id,
            ),
            _model_event(
                EventType.PROVIDER_OPERATION_STARTING,
                sequence_identity="attempt-a",
                session_id=session_id,
                payload={
                    "provider": "reconnectable",
                    "source_run_epoch": 1,
                    "start_id": "provider-operation:attempt-a",
                },
            ),
        ]
        if include_started:
            events.append(_operation_event(operation_id="response_a", session_id=session_id))
        if terminal_type is not None:
            events.append(
                _model_event(
                    terminal_type,
                    sequence_identity="attempt-a",
                    session_id=session_id,
                )
            )
        await app.session_store.append_events(session_id, events)
        return await inspect_provider_operation(app.session_store, session_id)

    inspection = asyncio.run(scenario())

    assert inspection.status is expected_status
    if expected_status is ProviderOperationInspectionStatus.AMBIGUOUS_SUBMISSION:
        assert inspection.provider == "reconnectable"
        assert inspection.operation_id is None
        assert inspection.stream_protocol is None
        assert inspection.recovery_reason == "ambiguous_submission"
        assert inspection.duplicate_request_risk is True
        assert inspection.allowed_resolutions == ("fallback_retry", "fail")
    if expected_status is ProviderOperationInspectionStatus.PROVIDER_OPERATION_IN_PROGRESS:
        assert inspection.provider == "reconnectable"
        assert inspection.operation_id == ("response_a" if include_started else None)
        assert inspection.stream_protocol == ("responses-v1" if include_started else None)


@pytest.mark.parametrize(
    ("provider_status", "recovery_reason", "duplicate_request_risk"),
    [
        (ProviderOperationStatus.FAILED, "failed", False),
        (ProviderOperationStatus.EXPIRED, "expired", False),
        (ProviderOperationStatus.CANCELLED, "cancelled", False),
        (ProviderOperationStatus.UNAVAILABLE, "unavailable", True),
        (ProviderOperationStatus.COMPLETED, "malformed", True),
    ],
)
def test_operator_inspection_exposes_terminal_provider_operation_for_resolution(
    provider_status: ProviderOperationStatus,
    recovery_reason: str,
    duplicate_request_risk: bool,
) -> None:
    async def scenario():
        session_id = f"inspection_{provider_status.value}"
        app = CayuApp(enable_logging=False)
        await app.session_store.create(
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "hello")],
            ),
            identity=SessionIdentity(provider_name="reconnectable", model="fake-model"),
        )
        await app.session_store.append_events(
            session_id,
            [
                _model_event(
                    EventType.MODEL_STARTED,
                    sequence_identity="attempt-a",
                    session_id=session_id,
                ),
                _operation_event(operation_id="response_a", session_id=session_id).model_copy(
                    update={
                        "payload": {
                            **_operation_event(
                                operation_id="response_a",
                                session_id=session_id,
                            ).payload,
                            "status": provider_status.value,
                        }
                    },
                    deep=True,
                ),
            ],
        )
        return await inspect_provider_operation(app.session_store, session_id)

    inspection = asyncio.run(scenario())

    assert inspection.status is ProviderOperationInspectionStatus.PROVIDER_OPERATION_UNAVAILABLE
    assert inspection.provider == "reconnectable"
    assert inspection.operation_id == "response_a"
    assert inspection.recovery_reason == recovery_reason
    assert inspection.duplicate_request_risk is duplicate_request_risk
    assert inspection.allowed_resolutions == ("fallback_retry", "fail")


def test_operator_inspection_uses_latest_reconnect_transition() -> None:
    async def scenario():
        session_id = "inspection_latest_reconnect_transition"
        app = CayuApp(enable_logging=False)
        await app.session_store.create(
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "hello")],
            ),
            identity=SessionIdentity(provider_name="reconnectable", model="fake-model"),
        )
        recovery_payload = {
            "operation_id": "response_a",
            "stream_protocol": "responses-v1",
            "status": "in_progress",
        }
        await app.session_store.append_events(
            session_id,
            [
                _model_event(
                    EventType.MODEL_STARTED,
                    sequence_identity="attempt-a",
                    session_id=session_id,
                ),
                _operation_event(operation_id="response_a", session_id=session_id),
                _model_event(
                    EventType.PROVIDER_OPERATION_RECONNECT_SCHEDULED,
                    sequence_identity="attempt-a",
                    session_id=session_id,
                    payload=recovery_payload,
                ),
                _model_event(
                    EventType.PROVIDER_OPERATION_RECONNECT_STARTED,
                    sequence_identity="attempt-a",
                    session_id=session_id,
                    payload=recovery_payload,
                ),
                _model_event(
                    EventType.PROVIDER_OPERATION_RECONNECT_SCHEDULED,
                    sequence_identity="attempt-a",
                    session_id=session_id,
                    payload=recovery_payload,
                ),
            ],
        )
        return await inspect_provider_operation(app.session_store, session_id)

    inspection = asyncio.run(scenario())

    assert inspection.status is ProviderOperationInspectionStatus.RECONNECT_SCHEDULED
    assert inspection.operation_id == "response_a"


class _FenceOnEventStore(InMemorySessionStore):
    invocation_lifecycle_command_version = 1

    def __init__(self, fence_on: EventType) -> None:
        super().__init__()
        self.fence_on = fence_on
        self.fenced = False

    async def append_event(self, session_id: str, event: Event) -> None:
        if event.type == self.fence_on and not self.fenced:
            self.fenced = True
            fenced = await asyncio.create_task(
                self.fence_stalled_run(
                    session_id,
                    statuses={SessionStatus.RUNNING},
                    inactive_for_seconds=0,
                )
            )
            assert fenced is not None
        await super().append_event(session_id, event)


class _CancelOnEventStore(InMemorySessionStore):
    invocation_lifecycle_command_version = 1

    def __init__(self, cancel_on: EventType) -> None:
        super().__init__()
        self.cancel_on = cancel_on

    async def append_event(self, session_id: str, event: Event) -> None:
        if event.type == self.cancel_on:
            raise asyncio.CancelledError("publication cancelled")
        await super().append_event(session_id, event)


class _FailBeforeCommitOnEventStore(InMemorySessionStore):
    invocation_lifecycle_command_version = 1

    def __init__(self, fail_on: EventType, failure: Exception) -> None:
        super().__init__()
        self.fail_on = fail_on
        self.failure = failure
        self.failed = False

    async def append_event(self, session_id: str, event: Event) -> None:
        if event.type == self.fail_on and not self.failed:
            self.failed = True
            raise self.failure
        await super().append_event(session_id, event)


class _HostileCauseError(Exception):
    @property
    def __cause__(self):
        raise RuntimeError("extension cause getter must not run")

    @__cause__.setter
    def __cause__(self, value):
        raise RuntimeError("extension cause setter must not run")


class _FailClaimAfterStartedCommitStore(InMemorySessionStore):
    invocation_lifecycle_command_version = 1

    def __init__(self) -> None:
        super().__init__()
        self.started_event_id: str | None = None
        self.failed_claim = False

    async def append_event(self, session_id: str, event: Event) -> None:
        await super().append_event(session_id, event)
        if event.type == EventType.PROVIDER_OPERATION_STARTED:
            self.started_event_id = event.id

    async def claim_persisted_event_side_effect(
        self,
        *,
        session_id: str | None = None,
        event_id: str | None = None,
        lease_seconds: float = 300.0,
    ) -> PersistedEventSideEffectClaim | None:
        if event_id == self.started_event_id and not self.failed_claim:
            self.failed_claim = True
            raise TimeoutError("post-commit side-effect claim timed out")
        return await super().claim_persisted_event_side_effect(
            session_id=session_id,
            event_id=event_id,
            lease_seconds=lease_seconds,
        )


class _UnverifiableStartedCommitStore(InMemorySessionStore):
    invocation_lifecycle_command_version = 1

    def __init__(self) -> None:
        super().__init__()
        self.started_event_id: str | None = None

    async def append_event(self, session_id: str, event: Event) -> None:
        if event.type == EventType.PROVIDER_OPERATION_STARTED:
            self.started_event_id = event.id
            raise TimeoutError("started identity commit is unknown")
        await super().append_event(session_id, event)

    async def query_events(self, query: EventQuery | None = None) -> list[EventRecord]:
        if (
            self.started_event_id is not None
            and query is not None
            and query.event_id == self.started_event_id
        ):
            raise TimeoutError("started identity readback is unavailable")
        return await super().query_events(query)


class _CancelOuterStartedReadbackStore(InMemorySessionStore):
    invocation_lifecycle_command_version = 1

    def __init__(self) -> None:
        super().__init__()
        self.started_event_id: str | None = None
        self.readbacks = 0

    async def append_event(self, session_id: str, event: Event) -> None:
        if event.type == EventType.PROVIDER_OPERATION_STARTED:
            self.started_event_id = event.id
            raise TimeoutError("started identity commit is unknown")
        await super().append_event(session_id, event)

    async def query_events(self, query: EventQuery | None = None) -> list[EventRecord]:
        if (
            self.started_event_id is not None
            and query is not None
            and query.event_id == self.started_event_id
        ):
            self.readbacks += 1
            if self.readbacks == 2:
                raise asyncio.CancelledError("caller cancelled during exact readback")
        return await super().query_events(query)


async def _collect_run_events(app: CayuApp, request: RunRequest):
    return [event async for event in app.run(request)]


def _two_attempt_retry_policy() -> RetryPolicy:
    return RetryPolicy(
        max_attempts=2,
        initial_delay_s=0.0,
        max_delay_s=0.0,
        backoff_multiplier=1.0,
        jitter_s=0.0,
    )


def _model_event(
    event_type: EventType,
    *,
    sequence_identity: str,
    payload=None,
    session_id: str = "inspection_in_progress",
):
    completion_payload = (
        {
            "transcript_cursor": 1,
            "provider_name": "reconnectable",
            "requested_model": "fake-model",
        }
        if event_type == EventType.MODEL_COMPLETED
        else {}
    )
    return Event(
        type=event_type,
        session_id=session_id,
        interaction_id="interaction-a",
        payload={
            "provider": "reconnectable",
            "model": "fake-model",
            "step": 1,
            "attempt": 1,
            "max_attempts": 2,
            "model_step_id": "mstep_" + "a" * 32,
            "model_attempt_id": sequence_identity,
            "source_run_epoch": 1,
            **completion_payload,
            **(payload or {}),
        },
    )


def _operation_event(*, operation_id: str, session_id: str) -> Event:
    return _model_event(
        EventType.PROVIDER_OPERATION_STARTED,
        sequence_identity="attempt-a",
        session_id=session_id,
        payload={
            "provider": "reconnectable",
            "source_run_epoch": 1,
            "start_id": "provider-operation:attempt-a",
            "state_version": 1,
            "operation_id": operation_id,
            "stream_protocol": "responses-v1",
            "status": "in_progress",
            "recovery_metadata": {"cursor": 0},
        },
    )

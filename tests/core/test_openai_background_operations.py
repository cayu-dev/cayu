from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Mapping
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
import pytest

from cayu import AgentSpec, CayuApp, InMemorySessionStore, RunRequest
from cayu.core import EventType, Message
from cayu.core.messages import MessageRole, ProviderStatePart
from cayu.providers import (
    HttpxOpenAITransport,
    ModelRequest,
    ModelStreamEventType,
    OpenAIAPIError,
    OpenAIProtocolError,
    OpenAIProvider,
    ProviderOperationCancellationSupport,
    ProviderOperationMalformedError,
    ProviderOperationMode,
    ProviderOperationStartIdempotencySupport,
    ProviderOperationStartRequest,
    ProviderOperationState,
    ProviderOperationStatus,
    TargetedToolProjectionRequest,
    ToolDiscoveryProjectionRequest,
)
from cayu.providers.base import (
    OPENAI_ADDITIONAL_TOOLS_PROTOCOL,
    TARGETED_TOOL_NATIVE_CACHE_ANCHOR_OPTION,
    TARGETED_TOOL_PROJECTION_MARKER_TYPE,
)
from cayu.runtime import (
    IncompleteSessionRecoveryRequest,
    ModelCompletionManualRecoveryRequired,
    SessionStatus,
)
from cayu.runtime.provider_operations import (
    ProviderOperationInspectionStatus,
    ProviderOperationUnavailableReason,
    inspect_provider_operation,
)
from cayu.runtime.tool_catalogue import CALL_TOOL_NAME
from cayu.runtime.tool_discovery import search_tools_spec
from cayu.runtime.tool_gateway import call_tool_spec


def _request() -> ModelRequest:
    return ModelRequest(
        model="gpt-test",
        messages=[Message.text("user", "finish this in the background")],
    )


def _targeted_request() -> tuple[ModelRequest, str]:
    marker_id = f"sha256:{'a' * 64}"
    marker = Message(
        role=MessageRole.ASSISTANT,
        content=(
            ProviderStatePart(
                provider="openai",
                state={
                    "type": TARGETED_TOOL_PROJECTION_MARKER_TYPE,
                    "protocol": OPENAI_ADDITIONAL_TOOLS_PROTOCOL,
                    "marker_id": marker_id,
                },
            ),
        ),
    )
    return (
        ModelRequest(
            model="gpt-test",
            messages=[Message.text("user", "remember this"), marker],
            tools=[call_tool_spec()],
            targeted_tool_projection=TargetedToolProjectionRequest(
                marker_id=marker_id,
                tools=(
                    {
                        "name": "remember",
                        "description": "Remember one reviewed fact.",
                        "input_schema": {
                            "type": "object",
                            "properties": {"fact": {"type": "string"}},
                            "required": ["fact"],
                            "additionalProperties": False,
                        },
                    },
                ),
            ),
            options={TARGETED_TOOL_NATIVE_CACHE_ANCHOR_OPTION: CALL_TOOL_NAME},
        ),
        marker_id,
    )


def _native_discovery_request() -> ModelRequest:
    return ModelRequest(
        model="gpt-test",
        messages=[Message.text("user", "find a memory tool")],
        tools=[search_tools_spec(), call_tool_spec()],
        tool_discovery_projection=ToolDiscoveryProjectionRequest(),
    )


def _created(*, response_id: str = "resp_background_123", sequence_number: int = 0):
    return {
        "type": "response.created",
        "sequence_number": sequence_number,
        "response": {
            "id": response_id,
            "model": "gpt-test",
            "status": "in_progress",
            "output": [],
        },
    }


def _completed(*, sequence_number: int = 2, text: str = "finished"):
    return {
        "type": "response.completed",
        "sequence_number": sequence_number,
        "response": {
            "id": "resp_background_123",
            "model": "gpt-test",
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "id": "msg_1",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": text}],
                }
            ],
            "usage": {"input_tokens": 4, "output_tokens": 1, "total_tokens": 5},
        },
    }


class BackgroundTransport:
    def __init__(self) -> None:
        self.start_batches: list[list[Mapping[str, Any] | BaseException]] = []
        self.reconnect_batches: list[list[Mapping[str, Any] | BaseException]] = []
        self.retrieve_responses: list[Mapping[str, Any] | BaseException] = []
        self.cancel_responses: list[Mapping[str, Any] | BaseException] = []
        self.start_calls: list[dict[str, Any]] = []
        self.reconnect_calls: list[dict[str, Any]] = []
        self.retrieve_calls: list[dict[str, Any]] = []
        self.cancel_calls: list[dict[str, Any]] = []

    async def create_response(
        self,
        *,
        url: str,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
        timeout_s: float,
    ) -> Mapping[str, Any]:
        raise AssertionError(f"unexpected non-streaming create: {url} {payload} {timeout_s}")

    async def stream_response_events(
        self,
        *,
        url: str,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
        timeout_s: float,
        stream_idle_timeout_s: float,
    ) -> AsyncIterator[Mapping[str, Any]]:
        self.start_calls.append(
            {
                "url": url,
                "headers": dict(headers),
                "payload": dict(payload),
                "timeout_s": timeout_s,
                "stream_idle_timeout_s": stream_idle_timeout_s,
            }
        )
        if not self.start_batches:
            raise AssertionError("No fake OpenAI background start queued.")
        for event in self.start_batches.pop(0):
            if isinstance(event, BaseException):
                raise event
            yield event

    async def retrieve_response(
        self,
        *,
        url: str,
        headers: Mapping[str, str],
        timeout_s: float,
    ) -> Mapping[str, Any]:
        self.retrieve_calls.append({"url": url, "headers": dict(headers), "timeout_s": timeout_s})
        if not self.retrieve_responses:
            raise AssertionError("No fake OpenAI retrieval queued.")
        result = self.retrieve_responses.pop(0)
        if isinstance(result, BaseException):
            raise result
        return result

    async def reconnect_response_events(
        self,
        *,
        url: str,
        headers: Mapping[str, str],
        starting_after: int,
        timeout_s: float,
        stream_idle_timeout_s: float,
    ) -> AsyncIterator[Mapping[str, Any]]:
        self.reconnect_calls.append(
            {
                "url": url,
                "headers": dict(headers),
                "starting_after": starting_after,
                "timeout_s": timeout_s,
                "stream_idle_timeout_s": stream_idle_timeout_s,
            }
        )
        if not self.reconnect_batches:
            raise AssertionError("No fake OpenAI reconnect queued.")
        for event in self.reconnect_batches.pop(0):
            if isinstance(event, BaseException):
                raise event
            yield event

    async def cancel_response(
        self,
        *,
        url: str,
        headers: Mapping[str, str],
        timeout_s: float,
    ) -> Mapping[str, Any]:
        self.cancel_calls.append({"url": url, "headers": dict(headers), "timeout_s": timeout_s})
        if not self.cancel_responses:
            raise AssertionError("No fake OpenAI cancellation queued.")
        result = self.cancel_responses.pop(0)
        if isinstance(result, BaseException):
            raise result
        return result


class SimulatedWorkerLoss(BaseException):
    pass


class _LoseFirstCompletionAcknowledgementStore(InMemorySessionStore):
    def __init__(self) -> None:
        super().__init__()
        self.completion_calls = 0

    async def complete_model_completion_stage(
        self,
        session_id,
        *,
        stage_id,
        publication,
    ):
        result = await super().complete_model_completion_stage(
            session_id,
            stage_id=stage_id,
            publication=publication,
        )
        self.completion_calls += 1
        if self.completion_calls == 1:
            raise SimulatedWorkerLoss("worker disappeared after terminal commit")
        return result


class _SSEByteStream(httpx.AsyncByteStream):
    def __init__(self, content: bytes) -> None:
        self.content = content

    async def __aiter__(self):
        yield self.content


def test_openai_background_operations_are_explicit_and_provider_scoped() -> None:
    synchronous = OpenAIProvider(api_key="test-key", transport=BackgroundTransport())
    background = OpenAIProvider(
        api_key="test-key",
        background=True,
        transport=BackgroundTransport(),
    )

    assert synchronous.provider_operation_mode is ProviderOperationMode.SYNCHRONOUS
    assert synchronous.provider_operations is None
    assert background.provider_operation_mode is ProviderOperationMode.BACKGROUND
    assert background.provider_operations is not None
    assert (
        background.provider_operations.start_idempotency_support
        is ProviderOperationStartIdempotencySupport.UNSUPPORTED
    )
    assert (
        background.provider_operations.cancellation_support
        is ProviderOperationCancellationSupport.SUPPORTED
    )

    with pytest.raises(TypeError, match="background must be a bool"):
        OpenAIProvider(api_key="test-key", background=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="official OpenAI API base URL"):
        OpenAIProvider(
            api_key="test-key",
            background=True,
            base_url="https://gateway.example.test",
        )


@pytest.mark.anyio
async def test_openai_background_start_publishes_identity_before_output() -> None:
    transport = BackgroundTransport()
    transport.start_batches.append(
        [
            _created(),
            {
                "type": "response.output_text.delta",
                "sequence_number": 1,
                "delta": "finished",
            },
            _completed(),
        ]
    )
    provider = OpenAIProvider(api_key="test-key", background=True, transport=transport)
    adapter = provider.provider_operations
    assert adapter is not None

    connection = await adapter.start(
        ProviderOperationStartRequest(request=_request(), idempotency_key="stable-start-id")
    )

    assert connection.state.operation_id == "resp_background_123"
    assert connection.state.stream_protocol == "openai-responses-background-v1"
    assert connection.state.recovery_metadata.cursor == 0
    assert connection.state.recovery_metadata.opaque["sequence_number"] == 0
    assert connection.status is ProviderOperationStatus.IN_PROGRESS
    assert transport.start_calls[0]["url"] == "https://api.openai.com/v1/responses"
    assert transport.start_calls[0]["payload"]["background"] is True
    assert transport.start_calls[0]["payload"]["stream"] is True
    assert transport.start_calls[0]["payload"]["store"] is True
    assert "stable-start-id" not in repr(transport.start_calls[0])

    events = [event async for event in connection.events]

    assert [event.type for event in events] == [
        ModelStreamEventType.TEXT_DELTA,
        ModelStreamEventType.COMPLETED,
    ]
    assert [event.recovery_metadata.cursor for event in events if event.recovery_metadata] == [1, 2]
    assert [
        event.recovery_metadata.opaque["sequence_number"]
        for event in events
        if event.recovery_metadata
    ] == [1, 2]


@pytest.mark.anyio
async def test_openai_background_retains_targeted_tool_ownership_through_stream() -> None:
    transport = BackgroundTransport()
    transport.start_batches.append([_created(), _completed(sequence_number=1)])
    provider = OpenAIProvider(
        api_key="test-key",
        background=True,
        reasoning_state="server",
        additional_tools_models=("gpt-test",),
        transport=transport,
    )
    adapter = provider.provider_operations
    assert adapter is not None
    request, marker_id = _targeted_request()

    connection = await adapter.start(
        ProviderOperationStartRequest(request=request, idempotency_key="targeted-start-id")
    )
    events = [event async for event in connection.events]

    assert connection.state.recovery_metadata.opaque["targeted_tool_marker_id"] == marker_id
    assert transport.start_calls[0]["payload"]["input"][-1]["type"] == "additional_tools"
    completed = next(event for event in events if event.type is ModelStreamEventType.COMPLETED)
    assert completed.recovery_metadata is not None
    assert completed.recovery_metadata.opaque["targeted_tool_marker_id"] == marker_id
    response_ref = next(
        item["state"]
        for item in completed.payload["provider_state"]
        if item["state"].get("type") == "response_ref"
    )
    assert response_ref["targeted_tool_marker_id"] == marker_id


@pytest.mark.anyio
async def test_openai_background_rejects_unverified_native_projection_before_transport() -> None:
    transport = BackgroundTransport()
    provider = OpenAIProvider(
        api_key="test-key",
        background=True,
        transport=transport,
    )
    adapter = provider.provider_operations
    assert adapter is not None
    request, _marker_id = _targeted_request()

    with pytest.raises(ValueError, match="not established"):
        await adapter.start(
            ProviderOperationStartRequest(request=request, idempotency_key="targeted-start-id")
        )

    assert transport.start_calls == []


@pytest.mark.anyio
async def test_openai_background_reconnect_starts_after_last_accepted_sequence() -> None:
    transport = BackgroundTransport()
    transport.start_batches.append(
        [
            _created(),
            {
                "type": "response.output_text.delta",
                "sequence_number": 1,
                "delta": "part one",
            },
        ]
    )
    provider = OpenAIProvider(api_key="test-key", background=True, transport=transport)
    adapter = provider.provider_operations
    assert adapter is not None
    started = await adapter.start(
        ProviderOperationStartRequest(request=_request(), idempotency_key="start-id")
    )
    [accepted] = [event async for event in started.events]
    assert accepted.recovery_metadata is not None
    recovered_state = ProviderOperationState(
        operation_id=started.state.operation_id,
        stream_protocol=started.state.stream_protocol,
        recovery_metadata=accepted.recovery_metadata,
    )
    transport.reconnect_batches.append([_completed(sequence_number=2)])

    reconnected = await adapter.reconnect(recovered_state)
    events = [event async for event in reconnected.events]

    assert reconnected.state == recovered_state
    assert reconnected.status is ProviderOperationStatus.COMPLETED
    assert transport.reconnect_calls == [
        {
            "url": "https://api.openai.com/v1/responses/resp_background_123",
            "headers": {
                "content-type": "application/json",
                "authorization": "Bearer test-key",
            },
            "starting_after": 1,
            "timeout_s": 60.0,
            "stream_idle_timeout_s": 120.0,
        }
    ]
    assert len(events) == 1
    assert events[0].type is ModelStreamEventType.COMPLETED
    assert events[0].recovery_metadata is not None
    assert events[0].recovery_metadata.cursor == 2
    assert events[0].recovery_metadata.opaque["sequence_number"] == 2


@pytest.mark.anyio
@pytest.mark.parametrize("terminal_status", ["cancelled", "expired"])
async def test_openai_background_late_terminal_stream_state_fails_truthfully(
    terminal_status: str,
) -> None:
    transport = BackgroundTransport()
    provider = OpenAIProvider(api_key="test-key", background=True, transport=transport)
    adapter = provider.provider_operations
    assert adapter is not None
    state = ProviderOperationState.model_validate(
        {
            "operation_id": "resp_background_123",
            "stream_protocol": "openai-responses-background-v1",
            "recovery_metadata": {"cursor": 0, "opaque": {"sequence_number": 0}},
        }
    )
    transport.reconnect_batches.append(
        [
            {
                "type": "response.output_text.delta",
                "sequence_number": 1,
                "delta": "partial",
            },
            {
                "type": f"response.{terminal_status}",
                "sequence_number": 2,
                "response": {
                    "id": state.operation_id,
                    "model": "gpt-test",
                    "status": terminal_status,
                },
            },
        ]
    )

    connection = await adapter.reconnect(state)
    events = [event async for event in connection.events]

    assert connection.status is ProviderOperationStatus.IN_PROGRESS
    assert [event.type for event in events] == [
        ModelStreamEventType.TEXT_DELTA,
        ModelStreamEventType.ERROR,
    ]
    assert events[-1].payload["provider_error_code"] == f"response_{terminal_status}"
    assert events[-1].provider_operation_status is ProviderOperationStatus(terminal_status)
    assert events[-1].recovery_metadata is not None
    assert events[-1].recovery_metadata.opaque["sequence_number"] == 2


@pytest.mark.anyio
async def test_openai_background_generic_stream_error_remains_reconnectable() -> None:
    transport = BackgroundTransport()
    transport.start_batches.append(
        [
            _created(),
            {
                "type": "response.output_text.delta",
                "sequence_number": 1,
                "delta": "finished",
            },
            SimulatedWorkerLoss("worker disappeared after cursor publication"),
        ]
    )
    provider = OpenAIProvider(api_key="test-key", background=True, transport=transport)
    store = InMemorySessionStore()
    app = CayuApp(session_store=store, enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="gpt-test"))
    session_id = "openai-generic-stream-error"
    with pytest.raises(SimulatedWorkerLoss):
        _ = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "continue after stream error")],
                )
            )
        ]
    transport.reconnect_batches.append(
        [
            {
                "type": "error",
                "sequence_number": 2,
                "code": "stream_interrupted",
                "message": "stream interrupted while response may still run",
            }
        ]
    )

    await app.recover_incomplete_session(
        IncompleteSessionRecoveryRequest(
            session_id=session_id,
            inactive_before=datetime.now(UTC) + timedelta(seconds=1),
        )
    )
    pending = await store.load(session_id)
    assert pending is not None
    assert pending.status in {SessionStatus.PENDING, SessionStatus.RUNNING}
    inspection = await inspect_provider_operation(store, session_id)
    assert inspection.status is ProviderOperationInspectionStatus.RECONNECT_SCHEDULED

    transport.reconnect_batches.append([_completed(sequence_number=3)])
    await app.recover_incomplete_session(IncompleteSessionRecoveryRequest(session_id=session_id))

    transcript = await store.load_transcript(session_id)
    assert transcript[-1].content[0].text == "finished"
    assert len(transport.reconnect_calls) == 2
    inspection = await inspect_provider_operation(store, session_id)
    assert inspection.status is ProviderOperationInspectionStatus.PROVIDER_OPERATION_RECONCILED


@pytest.mark.anyio
async def test_rejected_terminal_event_cannot_self_classify_provider_status() -> None:
    transport = BackgroundTransport()
    transport.start_batches.append(
        [
            _created(),
            {
                "type": "response.output_text.delta",
                "sequence_number": 1,
                "delta": "accepted",
            },
            SimulatedWorkerLoss("worker disappeared after cursor publication"),
        ]
    )
    transport.reconnect_batches.append(
        [
            {
                "type": "response.cancelled",
                "sequence_number": 1,
                "response": {
                    "id": "resp_background_123",
                    "model": "gpt-test",
                    "status": "cancelled",
                },
            }
        ]
    )
    provider = OpenAIProvider(api_key="test-key", background=True, transport=transport)
    store = InMemorySessionStore()
    app = CayuApp(session_store=store, enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="gpt-test"))
    session_id = "openai-rejected-terminal-status"
    with pytest.raises(SimulatedWorkerLoss):
        _ = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "reject forged terminal status")],
                )
            )
        ]

    await app.recover_incomplete_session(
        IncompleteSessionRecoveryRequest(
            session_id=session_id,
            inactive_before=datetime.now(UTC) + timedelta(seconds=1),
        )
    )

    inspection = await inspect_provider_operation(store, session_id)
    assert inspection.status is ProviderOperationInspectionStatus.PROVIDER_OPERATION_UNAVAILABLE
    assert inspection.recovery_reason is ProviderOperationUnavailableReason.MALFORMED
    assert inspection.duplicate_request_risk is True


@pytest.mark.anyio
async def test_openai_background_retrieves_completion_that_finished_offline() -> None:
    transport = BackgroundTransport()
    provider = OpenAIProvider(api_key="test-key", background=True, transport=transport)
    adapter = provider.provider_operations
    assert adapter is not None
    state = ProviderOperationState(
        operation_id="resp_background_123",
        stream_protocol="openai-responses-background-v1",
        recovery_metadata={"cursor": 0, "opaque": {"sequence_number": 0}},
    )
    transport.retrieve_responses.append(_completed()["response"])

    snapshot = await adapter.retrieve(state)

    assert snapshot.state == state
    assert snapshot.status is ProviderOperationStatus.COMPLETED
    assert [event.type for event in snapshot.events] == [
        ModelStreamEventType.TEXT_DELTA,
        ModelStreamEventType.COMPLETED,
    ]
    assert snapshot.events[0].delta == "finished"
    assert snapshot.events[-1].payload["usage"] == {
        "input_tokens": 4,
        "output_tokens": 1,
        "total_tokens": 5,
    }
    assert transport.retrieve_calls[0]["url"].endswith("/v1/responses/resp_background_123")


@pytest.mark.anyio
async def test_openai_background_retrieval_preserves_explicit_targeted_clear() -> None:
    transport = BackgroundTransport()
    provider = OpenAIProvider(
        api_key="test-key",
        background=True,
        reasoning_state="server",
        transport=transport,
    )
    adapter = provider.provider_operations
    assert adapter is not None
    state = ProviderOperationState(
        operation_id="resp_background_123",
        stream_protocol="openai-responses-background-v1",
        recovery_metadata={
            "cursor": 0,
            "opaque": {
                "sequence_number": 0,
                "targeted_tool_marker_id": None,
            },
        },
    )
    transport.retrieve_responses.append(_completed()["response"])

    snapshot = await adapter.retrieve(state)

    completed = next(
        event for event in snapshot.events if event.type is ModelStreamEventType.COMPLETED
    )
    response_ref = next(
        item["state"]
        for item in completed.payload["provider_state"]
        if item["state"].get("type") == "response_ref"
    )
    assert response_ref["targeted_tool_marker_id"] is None
    assert completed.recovery_metadata is not None
    assert completed.recovery_metadata.opaque["targeted_tool_marker_id"] is None


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("openai_status", "expected"),
    [
        ("queued", ProviderOperationStatus.QUEUED),
        ("in_progress", ProviderOperationStatus.IN_PROGRESS),
        ("failed", ProviderOperationStatus.FAILED),
        ("cancelled", ProviderOperationStatus.CANCELLED),
        ("expired", ProviderOperationStatus.EXPIRED),
    ],
)
async def test_openai_background_maps_provider_terminal_states(
    openai_status: str,
    expected: ProviderOperationStatus,
) -> None:
    transport = BackgroundTransport()
    provider = OpenAIProvider(api_key="test-key", background=True, transport=transport)
    adapter = provider.provider_operations
    assert adapter is not None
    state = ProviderOperationState(
        operation_id="resp_background_123",
        stream_protocol="openai-responses-background-v1",
        recovery_metadata={"cursor": 0, "opaque": {"sequence_number": 0}},
    )
    transport.retrieve_responses.append(
        {"id": state.operation_id, "model": "gpt-test", "status": openai_status}
    )

    snapshot = await adapter.retrieve(state)

    assert snapshot.status is expected
    assert snapshot.events == ()


@pytest.mark.anyio
async def test_openai_background_cancels_the_same_response_idempotently() -> None:
    transport = BackgroundTransport()
    provider = OpenAIProvider(api_key="test-key", background=True, transport=transport)
    adapter = provider.provider_operations
    assert adapter is not None
    state = ProviderOperationState(
        operation_id="resp_background_123",
        stream_protocol="openai-responses-background-v1",
        recovery_metadata={"cursor": 0, "opaque": {"sequence_number": 0}},
    )
    cancelled = {"id": state.operation_id, "model": "gpt-test", "status": "cancelled"}
    transport.cancel_responses.extend([cancelled, cancelled])

    first = await adapter.cancel(state)
    second = await adapter.cancel(state)

    assert first.status is ProviderOperationStatus.CANCELLED
    assert second == first
    assert [call["url"] for call in transport.cancel_calls] == [
        "https://api.openai.com/v1/responses/resp_background_123/cancel",
        "https://api.openai.com/v1/responses/resp_background_123/cancel",
    ]


@pytest.mark.anyio
async def test_openai_background_retrieval_sanitizes_transport_failures() -> None:
    secret = "sk-secret-provider-canary"
    transport = BackgroundTransport()
    transport.retrieve_responses.append(RuntimeError(f"transport leaked {secret}"))
    provider = OpenAIProvider(api_key=secret, background=True, transport=transport)
    adapter = provider.provider_operations
    assert adapter is not None
    state = ProviderOperationState(
        operation_id="resp_background_123",
        stream_protocol="openai-responses-background-v1",
        recovery_metadata={"cursor": 0, "opaque": {"sequence_number": 0}},
    )

    with pytest.raises(OpenAIAPIError) as exc_info:
        await adapter.retrieve(state)

    diagnostics = repr((str(exc_info.value), vars(exc_info.value)))
    assert secret not in diagnostics
    assert exc_info.value.response_body is None


@pytest.mark.anyio
@pytest.mark.parametrize("operation", ["start", "retrieve", "cancel"])
async def test_openai_background_operation_failures_are_credential_safe(
    operation: str,
) -> None:
    secret = "sk-background-operation-canary"
    transport = BackgroundTransport()
    failure = RuntimeError(f"transport leaked {secret}")
    state = ProviderOperationState(
        operation_id="resp_background_123",
        stream_protocol="openai-responses-background-v1",
        recovery_metadata={"cursor": 0, "opaque": {"sequence_number": 0}},
    )
    if operation == "start":
        transport.start_batches.append([failure])
    elif operation == "retrieve":
        transport.retrieve_responses.append(failure)
    else:
        transport.cancel_responses.append(failure)
    provider = OpenAIProvider(api_key=secret, background=True, transport=transport)
    adapter = provider.provider_operations
    assert adapter is not None

    with pytest.raises(OpenAIAPIError) as exc_info:
        if operation == "start":
            await adapter.start(
                ProviderOperationStartRequest(request=_request(), idempotency_key="start-id")
            )
        elif operation == "retrieve":
            await adapter.retrieve(state)
        else:
            await adapter.cancel(state)

    assert secret not in repr((str(exc_info.value), vars(exc_info.value)))
    assert exc_info.value.response_body is None


@pytest.mark.anyio
@pytest.mark.parametrize("operation", ["start", "reconnect"])
@pytest.mark.parametrize("failure_kind", ["exception", "cancellation"])
async def test_openai_background_stream_construction_failures_are_credential_safe(
    operation: str,
    failure_kind: str,
) -> None:
    secret = "sk-background-stream-construction-canary"
    transport = BackgroundTransport()
    failure: BaseException = (
        asyncio.CancelledError(f"transport leaked {secret}")
        if failure_kind == "cancellation"
        else RuntimeError(f"transport leaked {secret}")
    )

    def fail_stream_construction(**_: Any) -> AsyncIterator[Mapping[str, Any]]:
        raise failure

    if operation == "start":
        transport.stream_response_events = fail_stream_construction  # type: ignore[method-assign]
    else:
        transport.reconnect_response_events = fail_stream_construction  # type: ignore[method-assign]
    provider = OpenAIProvider(api_key=secret, background=True, transport=transport)
    adapter = provider.provider_operations
    assert adapter is not None
    state = ProviderOperationState(
        operation_id="resp_background_123",
        stream_protocol="openai-responses-background-v1",
        recovery_metadata={"cursor": 0, "opaque": {"sequence_number": 0}},
    )

    expected_failure = asyncio.CancelledError if failure_kind == "cancellation" else OpenAIAPIError
    with pytest.raises(expected_failure) as exc_info:
        if operation == "start":
            await adapter.start(
                ProviderOperationStartRequest(request=_request(), idempotency_key="start-id")
            )
        else:
            await adapter.reconnect(state)

    assert secret not in repr((str(exc_info.value), vars(exc_info.value)))
    if isinstance(exc_info.value, OpenAIAPIError):
        assert exc_info.value.response_body is None


@pytest.mark.anyio
async def test_openai_background_reconnect_failure_is_credential_safe_unavailable() -> None:
    secret = "sk-background-reconnect-canary"
    transport = BackgroundTransport()
    transport.reconnect_batches.append([RuntimeError(f"transport leaked {secret}")])
    provider = OpenAIProvider(api_key=secret, background=True, transport=transport)
    adapter = provider.provider_operations
    assert adapter is not None
    state = ProviderOperationState(
        operation_id="resp_background_123",
        stream_protocol="openai-responses-background-v1",
        recovery_metadata={"cursor": 0, "opaque": {"sequence_number": 0}},
    )

    connection = await adapter.reconnect(state)

    assert connection.status is ProviderOperationStatus.UNAVAILABLE
    assert secret not in repr(connection)


@pytest.mark.anyio
async def test_openai_background_resanitizes_typed_transport_failures() -> None:
    secret = "sk-typed-error-canary"
    transport = BackgroundTransport()
    transport.retrieve_responses.append(
        OpenAIAPIError(
            f"provider leaked {secret}",
            status_code=500,
            response_body=f'{{"error":"{secret}"}}',
        )
    )
    provider = OpenAIProvider(api_key=secret, background=True, transport=transport)
    adapter = provider.provider_operations
    assert adapter is not None
    state = ProviderOperationState(
        operation_id="resp_background_123",
        stream_protocol="openai-responses-background-v1",
        recovery_metadata={"cursor": 0, "opaque": {"sequence_number": 0}},
    )

    with pytest.raises(OpenAIAPIError) as exc_info:
        await adapter.retrieve(state)

    assert secret not in repr((str(exc_info.value), vars(exc_info.value)))
    assert exc_info.value.status_code == 500
    assert exc_info.value.response_body is None


@pytest.mark.anyio
async def test_openai_background_http_transport_uses_retrieve_resume_and_cancel_routes() -> None:
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path.endswith("/cancel"):
            return httpx.Response(
                200,
                json={"id": "resp_background_123", "status": "cancelled"},
                request=request,
            )
        if request.url.params.get("stream") == "true":
            event = _completed(sequence_number=8)
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                stream=_SSEByteStream(f"data: {json.dumps(event)}\n\n".encode()),
                request=request,
            )
        return httpx.Response(
            200,
            json={"id": "resp_background_123", "status": "in_progress"},
            request=request,
        )

    transport = HttpxOpenAITransport()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        transport._client._client = client
        retrieved = await transport.retrieve_response(
            url="https://api.openai.com/v1/responses/resp_background_123",
            headers={"authorization": "Bearer test-key"},
            timeout_s=1.0,
        )
        reconnected = [
            event
            async for event in transport.reconnect_response_events(
                url="https://api.openai.com/v1/responses/resp_background_123",
                headers={"authorization": "Bearer test-key"},
                starting_after=7,
                timeout_s=1.0,
                stream_idle_timeout_s=1.0,
            )
        ]
        cancelled = await transport.cancel_response(
            url="https://api.openai.com/v1/responses/resp_background_123/cancel",
            headers={"authorization": "Bearer test-key"},
            timeout_s=1.0,
        )

    assert retrieved["status"] == "in_progress"
    assert reconnected == [_completed(sequence_number=8)]
    assert cancelled["status"] == "cancelled"
    assert [(request.method, request.url.path) for request in requests] == [
        ("GET", "/v1/responses/resp_background_123"),
        ("GET", "/v1/responses/resp_background_123"),
        ("POST", "/v1/responses/resp_background_123/cancel"),
    ]
    assert dict(requests[1].url.params) == {
        "stream": "true",
        "starting_after": "7",
    }
    assert all(request.content == b"" for request in requests)


@pytest.mark.anyio
async def test_openai_background_checkpoints_parser_state_between_tool_events() -> None:
    transport = BackgroundTransport()
    transport.start_batches.append(
        [
            _created(),
            {
                "type": "response.output_item.added",
                "sequence_number": 1,
                "output_index": 0,
                "item": {
                    "type": "function_call",
                    "id": "fc_1",
                    "call_id": "call_1",
                    "name": "echo",
                    "arguments": "",
                },
            },
            {
                "type": "response.function_call_arguments.delta",
                "sequence_number": 2,
                "output_index": 0,
                "item_id": "fc_1",
                "delta": '{"text":"hello"}',
            },
        ]
    )
    provider = OpenAIProvider(api_key="test-key", background=True, transport=transport)
    adapter = provider.provider_operations
    assert adapter is not None
    started = await adapter.start(
        ProviderOperationStartRequest(request=_request(), idempotency_key="start-id")
    )
    checkpoints = [event async for event in started.events]
    assert [event.type for event in checkpoints] == [
        ModelStreamEventType.THINKING,
        ModelStreamEventType.THINKING,
    ]
    assert [event.delta for event in checkpoints] == ["", ""]
    latest_metadata = checkpoints[-1].recovery_metadata
    assert latest_metadata is not None
    parser = latest_metadata.opaque["parser"]
    assert isinstance(parser, dict)
    assert parser["pending_function_calls"] == [
        {
            "output_index": 0,
            "item_id": "fc_1",
            "call_id": "call_1",
            "name": "echo",
        }
    ]
    assert "hello" not in repr(latest_metadata.opaque)
    recovered_state = ProviderOperationState(
        operation_id=started.state.operation_id,
        stream_protocol=started.state.stream_protocol,
        recovery_metadata=latest_metadata,
    )
    transport.reconnect_batches.append(
        [
            {
                "type": "response.function_call_arguments.done",
                "sequence_number": 3,
                "output_index": 0,
                "item_id": "fc_1",
                "name": "echo",
                "arguments": '{"text":"hello"}',
            },
            {
                "type": "response.completed",
                "sequence_number": 4,
                "response": {
                    "id": "resp_background_123",
                    "model": "gpt-test",
                    "status": "completed",
                    "output": [
                        {
                            "type": "function_call",
                            "id": "fc_1",
                            "call_id": "call_1",
                            "name": "echo",
                            "arguments": '{"text":"hello"}',
                            "status": "completed",
                        }
                    ],
                    "usage": {"input_tokens": 4, "output_tokens": 2},
                },
            },
        ]
    )

    reconnected = await adapter.reconnect(recovered_state)
    recovered = [event async for event in reconnected.events]

    assert transport.reconnect_calls[0]["starting_after"] == 2
    assert [event.type for event in recovered] == [
        ModelStreamEventType.TOOL_CALL,
        ModelStreamEventType.COMPLETED,
    ]
    assert recovered[0].payload == {
        "id": "call_1",
        "name": "echo",
        "arguments": {"text": "hello"},
    }
    assert [event.recovery_metadata.cursor for event in recovered] == [3, 4]  # type: ignore[union-attr]
    assert "parser" not in recovered[0].recovery_metadata.opaque  # type: ignore[union-attr]


@pytest.mark.anyio
async def test_openai_background_reconnect_does_not_repeat_completed_tool_search_call() -> None:
    transport = BackgroundTransport()
    completed_tool_search = {
        "type": "tool_search_call",
        "id": "ts_1",
        "call_id": "call_1",
        "execution": "client",
        "arguments": {"query": "durable memory", "limit": 1},
        "status": "completed",
    }
    transport.start_batches.append(
        [
            _created(),
            {
                "type": "response.output_item.added",
                "sequence_number": 1,
                "output_index": 0,
                "item": {
                    "type": "tool_search_call",
                    "id": "ts_1",
                    "call_id": "call_1",
                    "execution": "client",
                    "arguments": {},
                    "status": "in_progress",
                },
            },
            {
                "type": "response.output_item.done",
                "sequence_number": 2,
                "output_index": 0,
                "item": completed_tool_search,
            },
        ]
    )
    provider = OpenAIProvider(
        api_key="test-key",
        background=True,
        reasoning_state="server",
        client_tool_search_models=("gpt-test",),
        transport=transport,
    )
    adapter = provider.provider_operations
    assert adapter is not None
    started = await adapter.start(
        ProviderOperationStartRequest(
            request=_native_discovery_request(),
            idempotency_key="native-discovery-start",
        )
    )

    started_events = [event async for event in started.events]

    assert [event.type for event in started_events] == [
        ModelStreamEventType.THINKING,
        ModelStreamEventType.TOOL_CALL,
    ]
    checkpoint = started_events[-1].recovery_metadata
    assert checkpoint is not None
    assert checkpoint.opaque["tool_discovery_loaded_tool_names"] == []
    parser = checkpoint.opaque["parser"]
    assert isinstance(parser, dict)
    assert parser["pending_tool_search_calls"] == []
    assert parser["completed_tool_search_items"] == [
        {"output_index": 0, "item": completed_tool_search}
    ]
    recovered_state = ProviderOperationState(
        operation_id=started.state.operation_id,
        stream_protocol=started.state.stream_protocol,
        recovery_metadata=checkpoint,
    )
    transport.reconnect_batches.append(
        [
            {
                "type": "response.completed",
                "sequence_number": 3,
                "response": {
                    "id": "resp_background_123",
                    "model": "gpt-test",
                    "status": "completed",
                    "output": [completed_tool_search],
                    "usage": {"input_tokens": 4, "output_tokens": 2},
                },
            }
        ]
    )

    reconnected = await adapter.reconnect(recovered_state)
    recovered_events = [event async for event in reconnected.events]

    assert [event.type for event in recovered_events] == [ModelStreamEventType.COMPLETED]
    response_ref = next(
        item["state"]
        for item in recovered_events[0].payload["provider_state"]
        if item["state"].get("type") == "response_ref"
    )
    assert response_ref["tool_discovery_loaded_tool_names"] == []


@pytest.mark.anyio
async def test_openai_background_rejects_nonadvancing_provider_sequence() -> None:
    transport = BackgroundTransport()
    transport.start_batches.append(
        [
            _created(),
            {
                "type": "response.output_text.delta",
                "sequence_number": 0,
                "delta": "replayed created boundary",
            },
        ]
    )
    provider = OpenAIProvider(api_key="test-key", background=True, transport=transport)
    adapter = provider.provider_operations
    assert adapter is not None
    started = await adapter.start(
        ProviderOperationStartRequest(request=_request(), idempotency_key="start-id")
    )

    with pytest.raises(ProviderOperationMalformedError, match="OpenAI provider failed"):
        _ = [event async for event in started.events]


@pytest.mark.anyio
async def test_openai_background_maps_not_found_to_expired() -> None:
    transport = BackgroundTransport()
    provider = OpenAIProvider(api_key="test-key", background=True, transport=transport)
    adapter = provider.provider_operations
    assert adapter is not None
    state = ProviderOperationState(
        operation_id="resp_background_123",
        stream_protocol="openai-responses-background-v1",
        recovery_metadata={"cursor": 0, "opaque": {"sequence_number": 0}},
    )
    transport.retrieve_responses.append(OpenAIAPIError("not found", status_code=404))
    transport.cancel_responses.append(OpenAIAPIError("not found", status_code=404))

    retrieved = await adapter.retrieve(state)
    cancelled = await adapter.cancel(state)

    assert retrieved.status is ProviderOperationStatus.EXPIRED
    assert cancelled.status is ProviderOperationStatus.EXPIRED


@pytest.mark.anyio
async def test_openai_background_completion_wins_cancellation_race() -> None:
    transport = BackgroundTransport()
    provider = OpenAIProvider(api_key="test-key", background=True, transport=transport)
    adapter = provider.provider_operations
    assert adapter is not None
    state = ProviderOperationState(
        operation_id="resp_background_123",
        stream_protocol="openai-responses-background-v1",
        recovery_metadata={"cursor": 0, "opaque": {"sequence_number": 0}},
    )
    transport.cancel_responses.append(_completed()["response"])

    result = await adapter.cancel(state)

    assert result.status is ProviderOperationStatus.COMPLETED
    assert [event.type for event in result.events] == [
        ModelStreamEventType.TEXT_DELTA,
        ModelStreamEventType.COMPLETED,
    ]


@pytest.mark.anyio
async def test_openai_background_request_options_cannot_disable_runtime_authority() -> None:
    transport = BackgroundTransport()
    provider = OpenAIProvider(api_key="test-key", background=True, transport=transport)
    adapter = provider.provider_operations
    assert adapter is not None
    request = ModelRequest(
        model="gpt-test",
        messages=[Message.text("user", "do not weaken background authority")],
        options={"openai": {"background": False}},
    )

    with pytest.raises(ValueError, match="reserved: background"):
        await adapter.start(
            ProviderOperationStartRequest(request=request, idempotency_key="start-id")
        )


@pytest.mark.anyio
@pytest.mark.parametrize("loss_after_cursor", [False, True])
async def test_openai_background_worker_loss_recovers_without_resubmission(
    loss_after_cursor: bool,
) -> None:
    transport = BackgroundTransport()
    start_batch: list[Mapping[str, Any] | BaseException] = [_created()]
    if loss_after_cursor:
        start_batch.append(
            {
                "type": "response.output_text.delta",
                "sequence_number": 1,
                "delta": "finished",
            }
        )
    start_batch.append(SimulatedWorkerLoss("worker disappeared"))
    transport.start_batches.append(start_batch)
    provider = OpenAIProvider(api_key="test-key", background=True, transport=transport)
    store = InMemorySessionStore()

    app = CayuApp(session_store=store, enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="gpt-test"))

    session_id = f"openai-worker-loss-{'cursor' if loss_after_cursor else 'identity'}"
    with pytest.raises(SimulatedWorkerLoss):
        _ = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "finish durably")],
                )
            )
        ]

    if loss_after_cursor:
        transport.reconnect_batches.append([_completed(sequence_number=2)])
    else:
        transport.retrieve_responses.append(_completed()["response"])
    recovered = await app.recover_incomplete_session(
        IncompleteSessionRecoveryRequest(
            session_id=session_id,
            inactive_before=datetime.now(UTC) + timedelta(seconds=1),
        )
    )

    # Generic incomplete-session recovery interrupts the abandoned interaction
    # after reconciling its terminal provider result.
    assert recovered.status is SessionStatus.INTERRUPTED
    assert len(transport.start_calls) == 1
    assert len(transport.reconnect_calls) == int(loss_after_cursor)
    assert len(transport.retrieve_calls) == int(not loss_after_cursor)
    transcript = await store.load_transcript(session_id)
    assert len(transcript) == 2
    assert transcript[1].content[0].text == "finished"
    durable_events = await store.load_events(session_id)
    assert sum(event.type is EventType.PROVIDER_OPERATION_STARTED for event in durable_events) == 1
    assert sum(event.type is EventType.MODEL_TEXT_DELTA for event in durable_events) == int(
        loss_after_cursor
    )
    assert sum(event.type is EventType.MODEL_COMPLETED for event in durable_events) == 1
    inspection = await inspect_provider_operation(store, session_id)
    assert inspection.status is ProviderOperationInspectionStatus.PROVIDER_OPERATION_RECONCILED
    assert inspection.recovery_reason is None


@pytest.mark.anyio
@pytest.mark.parametrize("terminal_status", ["cancelled", "expired"])
async def test_openai_background_recovery_does_not_leave_late_terminal_state_pending(
    terminal_status: str,
) -> None:
    transport = BackgroundTransport()
    transport.start_batches.append(
        [
            _created(),
            {
                "type": "response.output_text.delta",
                "sequence_number": 1,
                "delta": "accepted",
            },
            SimulatedWorkerLoss("worker disappeared after cursor publication"),
        ]
    )
    provider = OpenAIProvider(api_key="test-key", background=True, transport=transport)
    store = InMemorySessionStore()
    app = CayuApp(session_store=store, enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="gpt-test"))
    session_id = f"openai-late-{terminal_status}"
    with pytest.raises(SimulatedWorkerLoss):
        _ = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "recover a terminal provider state")],
                )
            )
        ]
    transport.reconnect_batches.append(
        [
            {
                "type": "response.output_text.delta",
                "sequence_number": 2,
                "delta": " but not completed",
            },
            {
                "type": f"response.{terminal_status}",
                "sequence_number": 3,
                "response": {
                    "id": "resp_background_123",
                    "model": "gpt-test",
                    "status": terminal_status,
                },
            },
        ]
    )

    await app.recover_incomplete_session(
        IncompleteSessionRecoveryRequest(
            session_id=session_id,
            inactive_before=datetime.now(UTC) + timedelta(seconds=1),
        )
    )
    with pytest.raises(ModelCompletionManualRecoveryRequired):
        await app.recover_incomplete_session(
            IncompleteSessionRecoveryRequest(session_id=session_id)
        )

    assert len(transport.reconnect_calls) == 1
    inspection = await inspect_provider_operation(store, session_id)
    assert inspection.status is ProviderOperationInspectionStatus.PROVIDER_OPERATION_UNAVAILABLE
    assert inspection.recovery_reason is ProviderOperationUnavailableReason(terminal_status)


@pytest.mark.anyio
async def test_openai_background_live_failure_is_visible_in_generic_inspection() -> None:
    transport = BackgroundTransport()
    transport.start_batches.append(
        [
            _created(),
            {
                "type": "response.failed",
                "sequence_number": 1,
                "response": {
                    "id": "resp_background_123",
                    "model": "gpt-test",
                    "status": "failed",
                    "error": {"message": "provider rejected background execution"},
                },
            },
        ]
    )
    provider = OpenAIProvider(api_key="test-key", background=True, transport=transport)
    store = InMemorySessionStore()
    app = CayuApp(session_store=store, enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="gpt-test"))
    session_id = "openai-live-background-failure"

    _ = [
        event
        async for event in app.run(
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "fail after identity publication")],
            )
        )
    ]

    inspection = await inspect_provider_operation(store, session_id)
    assert inspection.status is ProviderOperationInspectionStatus.PROVIDER_OPERATION_UNAVAILABLE
    assert inspection.recovery_reason is ProviderOperationUnavailableReason.FAILED


@pytest.mark.anyio
@pytest.mark.parametrize("loss_after_cursor", [False, True])
async def test_openai_background_protocol_failure_maps_to_malformed_recovery(
    loss_after_cursor: bool,
) -> None:
    transport = BackgroundTransport()
    start_batch: list[Mapping[str, Any] | BaseException] = [_created()]
    if loss_after_cursor:
        start_batch.append(
            {
                "type": "response.output_text.delta",
                "sequence_number": 1,
                "delta": "accepted",
            }
        )
    start_batch.append(SimulatedWorkerLoss("worker disappeared"))
    transport.start_batches.append(start_batch)
    malformed = OpenAIProtocolError("invalid response object")
    if loss_after_cursor:
        transport.reconnect_batches.append([malformed])
    else:
        transport.retrieve_responses.append(malformed)
    provider = OpenAIProvider(api_key="test-key", background=True, transport=transport)
    store = InMemorySessionStore()
    app = CayuApp(session_store=store, enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="gpt-test"))
    session_id = f"openai-malformed-{'reconnect' if loss_after_cursor else 'retrieval'}"
    with pytest.raises(SimulatedWorkerLoss):
        _ = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "recover malformed provider data")],
                )
            )
        ]

    await app.recover_incomplete_session(
        IncompleteSessionRecoveryRequest(
            session_id=session_id,
            inactive_before=datetime.now(UTC) + timedelta(seconds=1),
        )
    )

    inspection = await inspect_provider_operation(store, session_id)
    assert inspection.status is ProviderOperationInspectionStatus.PROVIDER_OPERATION_UNAVAILABLE
    assert inspection.recovery_reason is ProviderOperationUnavailableReason.MALFORMED
    assert len(transport.reconnect_calls) == int(loss_after_cursor)
    assert len(transport.retrieve_calls) == int(not loss_after_cursor)


@pytest.mark.anyio
async def test_openai_background_lost_start_acknowledgement_stays_ambiguous() -> None:
    transport = BackgroundTransport()
    transport.start_batches.append([SimulatedWorkerLoss("response id acknowledgement lost")])
    provider = OpenAIProvider(api_key="test-key", background=True, transport=transport)
    store = InMemorySessionStore()
    app = CayuApp(session_store=store, enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="gpt-test"))
    session_id = "openai-lost-start-acknowledgement"

    with pytest.raises(SimulatedWorkerLoss):
        _ = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "do not submit twice")],
                )
            )
        ]

    await app.recover_incomplete_session(
        IncompleteSessionRecoveryRequest(
            session_id=session_id,
            inactive_before=datetime.now(UTC) + timedelta(seconds=1),
        )
    )

    assert len(transport.start_calls) == 1
    inspection = await inspect_provider_operation(store, session_id)
    assert inspection.status is ProviderOperationInspectionStatus.AMBIGUOUS_SUBMISSION
    assert inspection.recovery_reason is ProviderOperationUnavailableReason.AMBIGUOUS_SUBMISSION
    assert inspection.duplicate_request_risk is True


@pytest.mark.anyio
async def test_openai_background_terminal_commit_acknowledgement_replays_exactly_once() -> None:
    transport = BackgroundTransport()
    transport.start_batches.append(
        [
            _created(),
            {
                "type": "response.output_text.delta",
                "sequence_number": 1,
                "delta": "settled once",
            },
            _completed(text="settled once"),
        ]
    )
    provider = OpenAIProvider(api_key="test-key", background=True, transport=transport)
    store = _LoseFirstCompletionAcknowledgementStore()
    app = CayuApp(session_store=store, enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="gpt-test"))
    session_id = "openai-lost-terminal-acknowledgement"

    with pytest.raises(SimulatedWorkerLoss, match="terminal commit"):
        _ = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "settle once")],
                )
            )
        ]

    await app.recover_incomplete_session(
        IncompleteSessionRecoveryRequest(
            session_id=session_id,
            inactive_before=datetime.now(UTC) + timedelta(seconds=1),
        )
    )
    await app.recover_incomplete_session(IncompleteSessionRecoveryRequest(session_id=session_id))

    assert len(transport.start_calls) == 1
    assert transport.retrieve_calls == []
    assert transport.reconnect_calls == []
    transcript = await store.load_transcript(session_id)
    assert [message.content[0].text for message in transcript] == [
        "settle once",
        "settled once",
    ]
    events = await store.load_events(session_id)
    assert sum(event.type is EventType.MODEL_COMPLETED for event in events) == 1


@pytest.mark.anyio
async def test_openai_background_malformed_terminal_requires_typed_resolution() -> None:
    transport = BackgroundTransport()
    transport.start_batches.append([_created(), SimulatedWorkerLoss("worker disappeared")])
    transport.retrieve_responses.append(
        {"id": "resp_background_123", "model": "gpt-test", "status": "completed"}
    )
    provider = OpenAIProvider(api_key="test-key", background=True, transport=transport)
    store = InMemorySessionStore()
    app = CayuApp(session_store=store, enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="gpt-test"))
    session_id = "openai-malformed-terminal"
    with pytest.raises(SimulatedWorkerLoss):
        _ = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "malformed terminal")],
                )
            )
        ]

    recovered = await app.recover_incomplete_session(
        IncompleteSessionRecoveryRequest(
            session_id=session_id,
            inactive_before=datetime.now(UTC) + timedelta(seconds=1),
        )
    )

    assert recovered.status is SessionStatus.INTERRUPTED
    inspection = await inspect_provider_operation(store, session_id)
    assert inspection.recovery_reason is ProviderOperationUnavailableReason.MALFORMED

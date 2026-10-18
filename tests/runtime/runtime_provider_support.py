"""Provider support."""

from __future__ import annotations

from collections.abc import AsyncIterator

from cayu.messages import ToolCallPart, ToolResultPart
from cayu.providers.base import (
    InputTokenCountResult,
    ModelContextOverflowError,
    ModelProvider,
    ModelRequest,
    ModelStreamEvent,
    UsageDialect,
)
from cayu.runtime.execution_identity import ExecutionProfileBehaviorIdentity
from cayu.sessions.memory import InMemorySessionStore


class FakeProvider(ModelProvider):
    name = "fake"

    def output_token_option_path(self) -> tuple[str, ...]:
        return ("fake", "max_output_tokens")

    def __init__(
        self,
        events: list[ModelStreamEvent] | list[list[ModelStreamEvent]],
    ) -> None:
        if events and isinstance(events[0], list):
            self.event_batches = events  # type: ignore[assignment]
        else:
            self.event_batches = [events]  # type: ignore[list-item]
        self.requests: list[ModelRequest] = []

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        self.requests.append(request)
        batch_index = len(self.requests) - 1
        if batch_index >= len(self.event_batches):
            raise AssertionError(f"No fake provider event batch for request {batch_index}")
        for event in self.event_batches[batch_index]:
            yield event


class StageRecordingFakeProvider(FakeProvider):
    def __init__(
        self,
        events: list[ModelStreamEvent] | list[list[ModelStreamEvent]],
        *,
        store: InMemorySessionStore,
        session_id: str,
    ) -> None:
        super().__init__(events)
        self.store = store
        self.session_id = session_id
        self.stage_ids: list[str] = []

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        active = await self.store.load_active_model_completion_stage(self.session_id)
        assert active is not None
        self.stage_ids.append(active.stage.stage_id)
        async for event in super().stream(request):
            yield event


class UsageDialectMutatingProvider(FakeProvider):
    usage_dialect = UsageDialect.ANTHROPIC

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        self.usage_dialect = UsageDialect.GENERIC
        async for event in super().stream(request):
            yield event


class StrictToolHistoryProvider(FakeProvider):
    """Reject tool history that is not backed by a declared request tool."""

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        declared_tools = {
            tool["name"] for tool in request.tools if isinstance(tool.get("name"), str)
        }
        calls_by_id: dict[str, str] = {}
        for message in request.messages:
            for part in message.content:
                if type(part) is ToolCallPart:
                    if part.tool_name not in declared_tools:
                        raise AssertionError(
                            f"Undeclared tool call in provider history: {part.tool_name}"
                        )
                    calls_by_id[part.tool_call_id] = part.tool_name
                elif type(part) is ToolResultPart:
                    if part.tool_name not in declared_tools:
                        raise AssertionError(
                            f"Undeclared tool result in provider history: {part.tool_name}"
                        )
                    if calls_by_id.get(part.tool_call_id) != part.tool_name:
                        raise AssertionError(
                            f"Unmatched tool result in provider history: {part.tool_call_id}"
                        )
        async for event in super().stream(request):
            yield event


class CountingProvider(FakeProvider):
    def __init__(
        self,
        events: list[ModelStreamEvent] | list[list[ModelStreamEvent]],
        *,
        count_result: InputTokenCountResult | None = None,
        count_error: Exception | None = None,
    ) -> None:
        super().__init__(events)
        self.count_result = count_result
        self.count_error = count_error
        self.count_requests: list[ModelRequest] = []

    async def count_input_tokens(
        self,
        request: ModelRequest,
    ) -> InputTokenCountResult | None:
        self.count_requests.append(request)
        if self.count_error is not None:
            raise self.count_error
        return self.count_result


class ContextOverflowProvider(ModelProvider):
    name = "overflow"

    def __init__(
        self,
        *,
        overflow_requests: int = 1,
        success_events: list[ModelStreamEvent] | None = None,
    ) -> None:
        self.overflow_requests = overflow_requests
        self.success_events = success_events or [
            ModelStreamEvent.text_delta("recovered"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ]
        self.requests: list[ModelRequest] = []

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        self.requests.append(request)
        if len(self.requests) <= self.overflow_requests:
            raise ModelContextOverflowError(
                "context too large",
                provider=self.name,
                status_code=400,
                error_code="context_length_exceeded",
            )
        for event in self.success_events:
            yield event


class VersionedFakeProvider(FakeProvider):
    """Opaque fake whose request behavior is stable across test app instances."""

    @property
    def execution_profile_identity(self) -> ExecutionProfileBehaviorIdentity:
        return ExecutionProfileBehaviorIdentity(
            name="tests:runtime:versioned-fake-provider",
            behavior_version="1",
            implementation_version="1",
        )


class NativeStructuredOutputFakeProvider(FakeProvider):
    supports_native_structured_output = True

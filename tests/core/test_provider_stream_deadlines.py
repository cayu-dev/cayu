from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Mapping
from contextlib import suppress

import pytest

import cayu.providers.deadlines as provider_deadlines_module
from cayu import Message
from cayu.providers import (
    ChatCompletionsProtocolError,
    ChatCompletionsProvider,
    OpenAIProtocolError,
    OpenAIProvider,
    ProviderOperationState,
    chat_completions_stream_events,
)
from cayu.providers.base import (
    OPENAI_HOSTED_TOOL_SEARCH_PROTOCOL,
    ModelProvider,
    ModelProviderError,
    ModelRequest,
    ModelStreamDeadlineError,
    ModelStreamEvent,
    ModelStreamEventType,
    ToolDiscoveryProjectionRequest,
)
from cayu.providers.deadlines import (
    ProviderDeadlineKind,
    ProviderProgressKind,
    ProviderStreamDeadlineController,
    ProviderStreamDeadlineExceeded,
    ProviderStreamDeadlines,
    bind_provider_deadline_controller,
    reset_provider_deadline_controller,
)
from cayu.providers.openai import (
    _openai_background_stream_events,
    openai_stream_events,
)
from cayu.runtime._model_errors import (
    copy_provider_exception_control,
    model_provider_error_from_payload,
)
from cayu.runtime.tool_discovery import search_tools_spec
from cayu.runtime.tool_gateway import call_tool_spec


class _DeadlineProvider(ModelProvider):
    name = "deadline-test"

    def __init__(self, events: AsyncIterator[ModelStreamEvent], deadlines: ProviderStreamDeadlines):
        self._events = events
        self._deadlines = deadlines

    @property
    def stream_deadlines(self) -> ProviderStreamDeadlines:
        return self._deadlines

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        del request
        async for event in self._events:
            yield event


def _request() -> ModelRequest:
    return ModelRequest(model="test-model", messages=[Message.text("user", "hello")])


async def _blocked_events() -> AsyncIterator[ModelStreamEvent]:
    await asyncio.Event().wait()
    yield ModelStreamEvent.completed({})  # pragma: no cover


@pytest.mark.anyio
async def test_simultaneous_semantic_and_absolute_expiry_prefers_absolute() -> None:
    provider = _DeadlineProvider(
        _blocked_events(),
        ProviderStreamDeadlines(
            transport_idle_timeout_s=1,
            protocol_idle_timeout_s=1,
            semantic_progress_timeout_s=0.01,
            absolute_stream_timeout_s=0.01,
        ),
    )

    with pytest.raises(ModelStreamDeadlineError) as captured:
        await anext(provider.runtime_stream(_request()))

    assert captured.value.deadline_evidence.deadline_kind is ProviderDeadlineKind.ABSOLUTE
    assert captured.value.retryable is False


@pytest.mark.anyio
async def test_simultaneous_expiry_uses_stable_all_clock_precedence() -> None:
    controller = ProviderStreamDeadlineController(
        ProviderStreamDeadlines(
            transport_idle_timeout_s=0.01,
            protocol_idle_timeout_s=0.01,
            semantic_progress_timeout_s=0.01,
            absolute_stream_timeout_s=0.01,
        )
    )

    with pytest.raises(ProviderStreamDeadlineExceeded) as captured:
        await controller.wait_for(
            asyncio.Event().wait(),
            kinds=tuple(ProviderDeadlineKind),
        )

    assert captured.value.evidence.deadline_kind is ProviderDeadlineKind.ABSOLUTE


@pytest.mark.anyio
async def test_normalized_noop_events_do_not_refresh_semantic_progress() -> None:
    async def noops() -> AsyncIterator[ModelStreamEvent]:
        while True:
            await asyncio.sleep(0.005)
            yield ModelStreamEvent.thinking()

    provider = _DeadlineProvider(
        noops(),
        ProviderStreamDeadlines(
            transport_idle_timeout_s=1,
            protocol_idle_timeout_s=1,
            semantic_progress_timeout_s=0.03,
            absolute_stream_timeout_s=1,
        ),
    )

    with pytest.raises(ModelStreamDeadlineError) as captured:
        async for _ in provider.runtime_stream(_request()):
            pass

    assert captured.value.deadline_evidence.deadline_kind is ProviderDeadlineKind.SEMANTIC_IDLE


@pytest.mark.anyio
async def test_duplicate_metadata_empty_deltas_and_unknown_events_do_not_refresh_progress() -> None:
    class NoopMetadataTransport:
        async def create_response(
            self,
            *,
            url: str,
            headers: Mapping[str, str],
            payload: Mapping[str, object],
            timeout_s: float,
        ) -> Mapping[str, object]:
            del url, headers, payload, timeout_s
            raise AssertionError("non-streaming response was not expected")

        async def stream_response_events(
            self,
            *,
            url: str,
            headers: Mapping[str, str],
            payload: Mapping[str, object],
            timeout_s: float,
            transport_idle_timeout_s: float,
            protocol_idle_timeout_s: float,
            semantic_progress_timeout_s: float,
            absolute_stream_timeout_s: float,
        ) -> AsyncIterator[Mapping[str, object]]:
            del (
                url,
                headers,
                payload,
                timeout_s,
                transport_idle_timeout_s,
                protocol_idle_timeout_s,
                semantic_progress_timeout_s,
                absolute_stream_timeout_s,
            )
            created = {"type": "response.created", "response": {"id": "resp-1"}}
            yield created
            while True:
                await asyncio.sleep(0.003)
                yield created
                yield {"type": "response.output_text.delta", "delta": ""}
                yield {"type": "response.keepalive"}

    provider = OpenAIProvider(
        api_key="test-key",
        transport=NoopMetadataTransport(),
        transport_idle_timeout_s=1,
        protocol_idle_timeout_s=1,
        semantic_progress_timeout_s=0.03,
        absolute_stream_timeout_s=1,
    )

    with pytest.raises(ModelStreamDeadlineError) as captured:
        async for _ in provider.runtime_stream(_request()):
            pass

    evidence = captured.value.deadline_evidence
    assert evidence.deadline_kind is ProviderDeadlineKind.SEMANTIC_IDLE
    assert evidence.last_progress_kind is ProviderProgressKind.RESPONSE_IDENTITY


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("item", "expected_kind"),
    [
        (
            {
                "type": "reasoning",
                "id": "rs_1",
                "status": "in_progress",
                "summary": [],
            },
            ProviderProgressKind.REASONING,
        ),
        (
            {
                "type": "function_call",
                "id": "fc_1",
                "call_id": "call_1",
                "name": "lookup",
                "arguments": "",
                "status": "in_progress",
            },
            ProviderProgressKind.TOOL_CALL,
        ),
        (
            {
                "type": "tool_search_call",
                "id": "ts_1",
                "call_id": "search_1",
                "execution": "client",
                "arguments": {},
                "status": "in_progress",
            },
            ProviderProgressKind.TOOL_CALL,
        ),
        (
            {
                "type": "tool_search_call",
                "call_id": None,
                "execution": "server",
                "arguments": {},
                "status": "in_progress",
            },
            ProviderProgressKind.HOSTED_TOOL,
        ),
    ],
)
async def test_openai_accepted_output_item_start_records_semantic_progress(
    item: Mapping[str, object],
    expected_kind: ProviderProgressKind,
) -> None:
    async def raw_events() -> AsyncIterator[Mapping[str, object]]:
        yield {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": item,
        }

    controller = ProviderStreamDeadlineController(ProviderStreamDeadlines())
    token = bind_provider_deadline_controller(controller)
    try:
        with pytest.raises(OpenAIProtocolError, match="ended before response.completed"):
            async for _ in openai_stream_events(raw_events()):
                pass
    finally:
        reset_provider_deadline_controller(token)

    evidence = controller.evidence((ProviderDeadlineKind.SEMANTIC_IDLE,))
    assert evidence.last_progress_kind is expected_kind
    assert evidence.last_progress_elapsed_s is not None


@pytest.mark.anyio
async def test_openai_hosted_tool_search_output_refreshes_runtime_semantic_deadline() -> None:
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {"fact": {"type": "string"}},
        "required": ["fact"],
    }
    completed_search = {
        "type": "tool_search_call",
        "execution": "server",
        "call_id": None,
        "status": "completed",
        "arguments": {"paths": ["remember_knowledge"]},
    }
    completed_output = {
        "type": "tool_search_output",
        "execution": "server",
        "call_id": None,
        "status": "completed",
        "tools": [
            {
                "type": "function",
                "name": "remember_knowledge",
                "description": "Save durable knowledge.",
                "parameters": schema,
                "strict": False,
                "defer_loading": True,
                "output_schema": None,
            }
        ],
    }

    class HostedToolSearchTransport:
        async def create_response(self, **kwargs):
            del kwargs
            raise AssertionError("non-streaming response was not expected")

        async def stream_response_events(self, **kwargs):
            del kwargs
            yield {"type": "response.created", "response": {"id": "resp-search"}}
            yield {
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {
                    "type": "tool_search_call",
                    "execution": "server",
                    "call_id": None,
                    "status": "in_progress",
                    "arguments": {},
                },
            }
            yield {
                "type": "response.output_item.done",
                "output_index": 0,
                "item": completed_search,
            }
            await asyncio.sleep(0.12)
            yield {
                "type": "response.output_item.added",
                "output_index": 1,
                "item": {
                    **completed_output,
                    "status": "in_progress",
                },
            }
            yield {
                "type": "response.output_item.done",
                "output_index": 1,
                "item": completed_output,
            }
            await asyncio.sleep(0.12)
            yield {
                "type": "response.completed",
                "response": {
                    "id": "resp-search",
                    "model": "gpt-test",
                    "status": "completed",
                    "output": [],
                },
            }

    provider = OpenAIProvider(
        api_key="test-key",
        transport=HostedToolSearchTransport(),
        hosted_tool_search_models=("gpt-test",),
        transport_idle_timeout_s=1,
        protocol_idle_timeout_s=1,
        semantic_progress_timeout_s=0.2,
        absolute_stream_timeout_s=1,
    )
    request = ModelRequest(
        model="gpt-test",
        messages=[Message.text("user", "Find a memory tool.")],
        tools=[search_tools_spec(), call_tool_spec()],
        tool_discovery_projection=ToolDiscoveryProjectionRequest(
            protocol=OPENAI_HOSTED_TOOL_SEARCH_PROTOCOL,
            generation_id=f"sha256:{'d' * 64}",
            candidate_tools=(
                {
                    "name": "remember_knowledge",
                    "description": "Save durable knowledge.",
                    "input_schema": schema,
                },
            ),
        ),
    )

    events = [event async for event in provider.runtime_stream(request)]

    assert events[-1].type is ModelStreamEventType.COMPLETED


@pytest.mark.anyio
async def test_openai_repeated_function_start_is_rejected_without_progress_refresh() -> None:
    item = {
        "type": "function_call",
        "id": "fc_1",
        "call_id": "call_1",
        "name": "lookup",
        "arguments": "",
        "status": "in_progress",
    }
    first_processed = asyncio.Event()
    release_duplicate = asyncio.Event()

    async def raw_events() -> AsyncIterator[Mapping[str, object]]:
        event = {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": item,
        }
        yield event
        first_processed.set()
        await release_duplicate.wait()
        yield event

    controller = ProviderStreamDeadlineController(ProviderStreamDeadlines())
    token = bind_provider_deadline_controller(controller)
    stream = openai_stream_events(raw_events())
    task = asyncio.create_task(anext(stream))
    try:
        await asyncio.wait_for(first_processed.wait(), timeout=1)
        before = controller.evidence((ProviderDeadlineKind.SEMANTIC_IDLE,))
        release_duplicate.set()
        with pytest.raises(OpenAIProtocolError, match="output_item.added was repeated"):
            await task
        after = controller.evidence((ProviderDeadlineKind.SEMANTIC_IDLE,))
    finally:
        release_duplicate.set()
        if not task.done():
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
        reset_provider_deadline_controller(token)

    assert before.last_progress_kind is ProviderProgressKind.TOOL_CALL
    assert after.last_progress_kind is ProviderProgressKind.TOOL_CALL
    assert after.last_progress_elapsed_s == before.last_progress_elapsed_s


@pytest.mark.anyio
async def test_openai_reasoning_lifecycle_records_only_new_semantic_progress() -> None:
    added_processed = asyncio.Event()
    release_done = asyncio.Event()
    done_processed = asyncio.Event()
    release_duplicate = asyncio.Event()
    added_item = {
        "type": "reasoning",
        "id": "rs_1",
        "status": "in_progress",
        "summary": [],
    }
    done_item = {
        "type": "reasoning",
        "id": "rs_1",
        "status": "completed",
        "summary": [],
        "encrypted_content": "opaque-reasoning",
    }

    async def raw_events() -> AsyncIterator[Mapping[str, object]]:
        yield {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": added_item,
        }
        added_processed.set()
        await release_done.wait()
        done_event = {
            "type": "response.output_item.done",
            "output_index": 0,
            "item": done_item,
        }
        yield done_event
        done_processed.set()
        await release_duplicate.wait()
        yield done_event

    controller = ProviderStreamDeadlineController(ProviderStreamDeadlines())
    token = bind_provider_deadline_controller(controller)
    stream = openai_stream_events(raw_events())
    task = asyncio.create_task(anext(stream))
    try:
        await asyncio.wait_for(added_processed.wait(), timeout=1)
        after_added = controller.evidence((ProviderDeadlineKind.SEMANTIC_IDLE,))
        await asyncio.sleep(0.001)
        release_done.set()
        await asyncio.wait_for(done_processed.wait(), timeout=1)
        after_done = controller.evidence((ProviderDeadlineKind.SEMANTIC_IDLE,))
        release_duplicate.set()
        with pytest.raises(OpenAIProtocolError, match="output_item.done was repeated"):
            await task
        after_duplicate = controller.evidence((ProviderDeadlineKind.SEMANTIC_IDLE,))
    finally:
        release_done.set()
        release_duplicate.set()
        if not task.done():
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
        reset_provider_deadline_controller(token)

    assert after_added.last_progress_kind is ProviderProgressKind.REASONING
    assert after_added.last_progress_elapsed_s is not None
    assert after_done.last_progress_kind is ProviderProgressKind.REASONING
    assert after_done.last_progress_elapsed_s is not None
    assert after_done.last_progress_elapsed_s > after_added.last_progress_elapsed_s
    assert after_duplicate.last_progress_elapsed_s == after_done.last_progress_elapsed_s


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("item", "match"),
    [
        pytest.param(
            {
                "type": "reasoning",
                "status": "in_progress",
                "summary": [],
            },
            "requires nonblank id",
            id="missing-id",
        ),
        pytest.param(
            {
                "type": "reasoning",
                "id": "rs_1",
                "status": "completed",
                "summary": [],
            },
            "invalid lifecycle status",
            id="invalid-status",
        ),
        pytest.param(
            {
                "type": "reasoning",
                "id": "rs_1",
                "status": "in_progress",
                "summary": {},
            },
            "summary must be a list",
            id="invalid-summary",
        ),
    ],
)
async def test_openai_background_rejects_malformed_reasoning_without_progress(
    item: Mapping[str, object],
    match: str,
) -> None:
    async def raw_events() -> AsyncIterator[Mapping[str, object]]:
        yield {
            "type": "response.output_item.added",
            "sequence_number": 1,
            "output_index": 0,
            "item": item,
        }

    state = ProviderOperationState.model_validate(
        {
            "operation_id": "resp_malformed_reasoning",
            "stream_protocol": "openai-responses-background-v1",
            "recovery_metadata": {"cursor": 0, "opaque": {"sequence_number": 0}},
        }
    )
    controller = ProviderStreamDeadlineController(ProviderStreamDeadlines())
    token = bind_provider_deadline_controller(controller)
    try:
        with pytest.raises(OpenAIProtocolError, match=match):
            async for _ in _openai_background_stream_events(
                raw_events(),
                state=state,
                first=None,
                reasoning_state="inline",
            ):
                pass
        evidence = controller.evidence((ProviderDeadlineKind.SEMANTIC_IDLE,))
    finally:
        reset_provider_deadline_controller(token)

    assert evidence.last_progress_kind is None
    assert evidence.last_progress_elapsed_s is None


@pytest.mark.anyio
async def test_openai_background_reasoning_progress_is_exact_across_reconnect() -> None:
    controller = ProviderStreamDeadlineController(ProviderStreamDeadlines())
    added_item = {
        "type": "reasoning",
        "id": "rs_1",
        "status": "in_progress",
        "summary": [],
    }
    done_item = {
        "type": "reasoning",
        "id": "rs_1",
        "status": "completed",
        "summary": [],
        "encrypted_content": "opaque-reasoning",
    }

    async def added_events() -> AsyncIterator[Mapping[str, object]]:
        yield {
            "type": "response.output_item.added",
            "sequence_number": 1,
            "output_index": 0,
            "item": added_item,
        }

    initial_state = ProviderOperationState.model_validate(
        {
            "operation_id": "resp_reasoning",
            "stream_protocol": "openai-responses-background-v1",
            "recovery_metadata": {"cursor": 0, "opaque": {"sequence_number": 0}},
        }
    )
    token = bind_provider_deadline_controller(controller)
    try:
        added = [
            event
            async for event in _openai_background_stream_events(
                added_events(),
                state=initial_state,
                first=None,
                reasoning_state="inline",
            )
        ]
        after_added = controller.evidence((ProviderDeadlineKind.SEMANTIC_IDLE,))
        assert added[-1].recovery_metadata is not None
        pending_state = ProviderOperationState(
            operation_id=initial_state.operation_id,
            stream_protocol=initial_state.stream_protocol,
            recovery_metadata=added[-1].recovery_metadata,
        )

        async def conflicting_done() -> AsyncIterator[Mapping[str, object]]:
            yield {
                "type": "response.output_item.done",
                "sequence_number": 2,
                "output_index": 0,
                "item": {**done_item, "id": "rs_conflict"},
            }

        with pytest.raises(OpenAIProtocolError, match="identity conflicts"):
            async for _ in _openai_background_stream_events(
                conflicting_done(),
                state=pending_state,
                first=None,
                reasoning_state="inline",
            ):
                pass
        after_conflict = controller.evidence((ProviderDeadlineKind.SEMANTIC_IDLE,))

        async def valid_done() -> AsyncIterator[Mapping[str, object]]:
            yield {
                "type": "response.output_item.done",
                "sequence_number": 2,
                "output_index": 0,
                "item": done_item,
            }

        completed = [
            event
            async for event in _openai_background_stream_events(
                valid_done(),
                state=pending_state,
                first=None,
                reasoning_state="inline",
            )
        ]
        after_done = controller.evidence((ProviderDeadlineKind.SEMANTIC_IDLE,))
        assert completed[-1].recovery_metadata is not None
        completed_state = ProviderOperationState(
            operation_id=initial_state.operation_id,
            stream_protocol=initial_state.stream_protocol,
            recovery_metadata=completed[-1].recovery_metadata,
        )

        async def duplicate_done() -> AsyncIterator[Mapping[str, object]]:
            yield {
                "type": "response.output_item.done",
                "sequence_number": 3,
                "output_index": 0,
                "item": done_item,
            }

        with pytest.raises(OpenAIProtocolError, match="output_item.done was repeated"):
            async for _ in _openai_background_stream_events(
                duplicate_done(),
                state=completed_state,
                first=None,
                reasoning_state="inline",
            ):
                pass
        after_duplicate = controller.evidence((ProviderDeadlineKind.SEMANTIC_IDLE,))
    finally:
        reset_provider_deadline_controller(token)

    assert len(added) == 1
    assert len(completed) == 1
    assert after_added.last_progress_kind is ProviderProgressKind.REASONING
    assert after_added.last_progress_elapsed_s is not None
    assert after_conflict.last_progress_elapsed_s == after_added.last_progress_elapsed_s
    assert after_done.last_progress_kind is ProviderProgressKind.REASONING
    assert after_done.last_progress_elapsed_s is not None
    assert after_done.last_progress_elapsed_s > after_added.last_progress_elapsed_s
    assert after_duplicate.last_progress_elapsed_s == after_done.last_progress_elapsed_s
    assert pending_state.recovery_metadata.opaque["parser"]["pending_reasoning_output_indexes"] == [
        0
    ]
    assert pending_state.recovery_metadata.opaque["parser"]["pending_reasoning_items"] == [
        {"output_index": 0, "item_id": "rs_1"}
    ]
    assert completed_state.recovery_metadata.opaque["parser"][
        "completed_reasoning_output_indexes"
    ] == [0]


@pytest.mark.anyio
async def test_openai_background_restores_legacy_pending_reasoning_after_restart() -> None:
    legacy_state = ProviderOperationState.model_validate(
        {
            "operation_id": "resp_legacy_reasoning",
            "stream_protocol": "openai-responses-background-v1",
            "recovery_metadata": {
                "cursor": 1,
                "opaque": {
                    "sequence_number": 1,
                    "parser": {
                        "pending_function_calls": [],
                        "pending_reasoning_output_indexes": [0],
                        "pending_tool_search_calls": [],
                        "completed_tool_search_items": [],
                        "completed_function_call_digests": [],
                    },
                },
            },
        }
    )

    async def premature_completion() -> AsyncIterator[Mapping[str, object]]:
        yield {
            "type": "response.completed",
            "sequence_number": 2,
        }

    with pytest.raises(OpenAIProtocolError, match="unfinished output items"):
        async for _ in _openai_background_stream_events(
            premature_completion(),
            state=legacy_state,
            first=None,
            reasoning_state="inline",
        ):
            pass

    async def valid_done() -> AsyncIterator[Mapping[str, object]]:
        yield {
            "type": "response.output_item.done",
            "sequence_number": 2,
            "output_index": 0,
            "item": {
                "type": "reasoning",
                "id": "rs_legacy",
                "status": "completed",
                "summary": [],
                "encrypted_content": "opaque-reasoning",
            },
        }

    completed = [
        event
        async for event in _openai_background_stream_events(
            valid_done(),
            state=legacy_state,
            first=None,
            reasoning_state="inline",
        )
    ]

    assert len(completed) == 1
    recovery = completed[-1].recovery_metadata
    assert recovery is not None
    parser = recovery.opaque["parser"]
    assert isinstance(parser, dict)
    assert parser["pending_reasoning_output_indexes"] == []
    assert parser["pending_reasoning_items"] == []
    assert parser["completed_reasoning_output_indexes"] == [0]


@pytest.mark.anyio
async def test_deadline_rejects_value_returned_after_provider_suppresses_cancellation() -> None:
    async def cancellation_resistant_read() -> str:
        try:
            await asyncio.Event().wait()
            raise AssertionError("the provider read unexpectedly resumed")
        except asyncio.CancelledError:
            return "late provider value"

    controller = ProviderStreamDeadlineController(
        ProviderStreamDeadlines(
            transport_idle_timeout_s=1,
            protocol_idle_timeout_s=1,
            semantic_progress_timeout_s=0.01,
            absolute_stream_timeout_s=1,
        )
    )

    with pytest.raises(ProviderStreamDeadlineExceeded) as captured:
        await asyncio.wait_for(
            controller.wait_for(
                cancellation_resistant_read(),
                kinds=(ProviderDeadlineKind.SEMANTIC_IDLE,),
            ),
            timeout=0.5,
        )

    assert captured.value.evidence.deadline_kind is ProviderDeadlineKind.SEMANTIC_IDLE
    assert captured.value.stream_cleanup_failed is True


@pytest.mark.anyio
async def test_runtime_retains_cancellation_resistant_provider_read_until_late_settlement() -> None:
    cancellation_seen = asyncio.Event()
    release = asyncio.Event()
    settled = asyncio.Event()
    close_called = asyncio.Event()

    class CancellationResistantEvents:
        def __aiter__(self) -> CancellationResistantEvents:
            return self

        async def __anext__(self) -> ModelStreamEvent:
            try:
                await asyncio.Event().wait()
                raise AssertionError("the provider read unexpectedly resumed")
            except asyncio.CancelledError:
                cancellation_seen.set()
                try:
                    await release.wait()
                    return ModelStreamEvent.text_delta("late provider value")
                finally:
                    settled.set()

        async def aclose(self) -> None:
            close_called.set()

    events = CancellationResistantEvents()

    class CancellationResistantProvider(ModelProvider):
        name = "cancellation-resistant"

        @property
        def stream_deadlines(self) -> ProviderStreamDeadlines:
            return ProviderStreamDeadlines(
                transport_idle_timeout_s=1,
                protocol_idle_timeout_s=1,
                semantic_progress_timeout_s=0.01,
                absolute_stream_timeout_s=1,
            )

        def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
            del request
            return events

    stream = CancellationResistantProvider().runtime_stream(_request())
    try:
        with pytest.raises(ModelStreamDeadlineError) as captured:
            await asyncio.wait_for(anext(stream), timeout=0.5)

        assert captured.value.deadline_evidence.deadline_kind is ProviderDeadlineKind.SEMANTIC_IDLE
        assert captured.value.stream_cleanup_failed is True
        assert cancellation_seen.is_set()
        assert close_called.is_set()
        assert not settled.is_set()
    finally:
        release.set()
        await asyncio.wait_for(settled.wait(), timeout=0.5)
        await asyncio.sleep(0)

    with pytest.raises(StopAsyncIteration):
        await anext(stream)


@pytest.mark.anyio
async def test_nonsettling_deadline_read_exhausts_capacity_before_provider_entry() -> None:
    existing_owners = set(provider_deadlines_module._PROVIDER_DEADLINE_AWAIT_OWNERS)
    capacity = len(existing_owners) + 1
    cancellation_seen = asyncio.Event()
    release = asyncio.Event()
    settled = asyncio.Event()

    class NonsettlingEvents:
        def __aiter__(self) -> NonsettlingEvents:
            return self

        async def __anext__(self) -> ModelStreamEvent:
            try:
                await asyncio.Event().wait()
                raise AssertionError("the provider read unexpectedly resumed")
            except asyncio.CancelledError:
                cancellation_seen.set()
                try:
                    await release.wait()
                    return ModelStreamEvent.text_delta("late provider value")
                finally:
                    settled.set()

        async def aclose(self) -> None:
            return None

    class ProbeProvider(ModelProvider):
        name = "deadline-capacity-probe"

        def __init__(self, events: AsyncIterator[ModelStreamEvent]) -> None:
            self.events = events
            self.entered = 0

        @property
        def stream_deadlines(self) -> ProviderStreamDeadlines:
            return ProviderStreamDeadlines(
                transport_idle_timeout_s=1,
                protocol_idle_timeout_s=1,
                semantic_progress_timeout_s=0.01,
                absolute_stream_timeout_s=1,
                max_concurrent_streams=capacity,
            )

        def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
            del request
            self.entered += 1
            return self.events

    async def completed_events() -> AsyncIterator[ModelStreamEvent]:
        yield ModelStreamEvent.completed({"finish_reason": "stop"})

    first = ProbeProvider(NonsettlingEvents())
    with pytest.raises(ModelStreamDeadlineError):
        await asyncio.wait_for(
            anext(first.runtime_stream(_request())),
            timeout=0.5,
        )
    assert first.entered == 1
    assert cancellation_seen.is_set()
    assert not settled.is_set()
    retained_owners = provider_deadlines_module._PROVIDER_DEADLINE_AWAIT_OWNERS - existing_owners
    assert len(retained_owners) == 1
    retained_owner = next(iter(retained_owners))
    rejected = ProbeProvider(completed_events())
    with pytest.raises(RuntimeError, match="deadline-read capacity is exhausted"):
        await anext(rejected.runtime_stream(_request()))
    assert rejected.entered == 0

    release.set()
    await asyncio.wait_for(settled.wait(), timeout=0.5)
    await asyncio.sleep(0)
    assert retained_owner not in provider_deadlines_module._PROVIDER_DEADLINE_AWAIT_OWNERS

    admitted = ProbeProvider(completed_events())
    admitted_events = [event async for event in admitted.runtime_stream(_request())]
    assert [event.type for event in admitted_events] == [ModelStreamEventType.COMPLETED]
    assert admitted.entered == 1


def test_default_stream_capacity_admits_more_than_legacy_sixty_four() -> None:
    admissions = [
        provider_deadlines_module.ProviderStreamDeadlineAdmission(ProviderStreamDeadlines())
        for _ in range(65)
    ]
    try:
        assert len(admissions) == 65
    finally:
        for admission in admissions:
            admission.close()


def test_deadline_admission_rejects_capacity_drift() -> None:
    admitted_capacity = len(provider_deadlines_module._PROVIDER_DEADLINE_AWAIT_OWNERS) + 1
    admission = provider_deadlines_module.ProviderStreamDeadlineAdmission(
        ProviderStreamDeadlines(max_concurrent_streams=admitted_capacity)
    )
    try:
        with pytest.raises(ValueError, match="capacity changed after dispatch admission"):
            admission.claim(ProviderStreamDeadlines(max_concurrent_streams=admitted_capacity + 1))
    finally:
        admission.close()


@pytest.mark.parametrize(
    ("value", "error", "message"),
    [
        pytest.param(0, ValueError, "max_concurrent_streams must be >= 1", id="zero"),
        pytest.param(1.5, TypeError, "max_concurrent_streams must be an int", id="float"),
    ],
)
def test_stream_capacity_validation(value: int, error: type[Exception], message: str) -> None:
    with pytest.raises(error, match=message):
        ProviderStreamDeadlines(max_concurrent_streams=value)


@pytest.mark.anyio
async def test_repeated_chat_tool_identity_metadata_does_not_refresh_semantic_progress() -> None:
    class RepeatedToolIdentityTransport:
        async def stream_chat_completions(
            self,
            *,
            url: str,
            headers: Mapping[str, str],
            payload: Mapping[str, object],
            timeout_s: float,
            transport_idle_timeout_s: float,
            protocol_idle_timeout_s: float,
            semantic_progress_timeout_s: float,
            absolute_stream_timeout_s: float,
        ) -> AsyncIterator[Mapping[str, object]]:
            del (
                url,
                headers,
                payload,
                timeout_s,
                transport_idle_timeout_s,
                protocol_idle_timeout_s,
                semantic_progress_timeout_s,
                absolute_stream_timeout_s,
            )
            repeated = {
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call-repeated",
                                    "type": "function",
                                    "function": {},
                                }
                            ]
                        },
                    }
                ]
            }
            while True:
                await asyncio.sleep(0.003)
                yield repeated

    provider = ChatCompletionsProvider(
        api_key="test-key",
        transport=RepeatedToolIdentityTransport(),
        transport_idle_timeout_s=1,
        protocol_idle_timeout_s=1,
        semantic_progress_timeout_s=0.03,
        absolute_stream_timeout_s=1,
    )

    with pytest.raises(ModelStreamDeadlineError) as captured:
        async for _ in provider.runtime_stream(_request()):
            pass

    evidence = captured.value.deadline_evidence
    assert evidence.deadline_kind is ProviderDeadlineKind.SEMANTIC_IDLE
    assert evidence.last_progress_kind is ProviderProgressKind.TOOL_CALL


@pytest.mark.anyio
async def test_continuous_semantic_progress_cannot_extend_absolute_lifetime() -> None:
    async def progress() -> AsyncIterator[ModelStreamEvent]:
        while True:
            await asyncio.sleep(0.005)
            yield ModelStreamEvent.text_delta("x")

    provider = _DeadlineProvider(
        progress(),
        ProviderStreamDeadlines(
            transport_idle_timeout_s=1,
            protocol_idle_timeout_s=1,
            semantic_progress_timeout_s=0.03,
            absolute_stream_timeout_s=0.05,
        ),
    )

    with pytest.raises(ModelStreamDeadlineError) as captured:
        async for _ in provider.runtime_stream(_request()):
            pass

    evidence = captured.value.deadline_evidence
    assert evidence.deadline_kind is ProviderDeadlineKind.ABSOLUTE
    assert evidence.last_progress_kind is not None


@pytest.mark.anyio
async def test_real_task_cancellation_remains_authoritative() -> None:
    provider = _DeadlineProvider(_blocked_events(), ProviderStreamDeadlines())
    task = asyncio.create_task(anext(provider.runtime_stream(_request())))
    await asyncio.sleep(0)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert task.cancelled()


@pytest.mark.anyio
async def test_bundled_terminal_preservation_is_not_inherited_by_stream_override() -> None:
    class OverridingChatProvider(ChatCompletionsProvider):
        def __init__(self) -> None:
            self._stream_deadlines = ProviderStreamDeadlines()

        async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
            del request
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                # An opaque override cannot turn cancellation into trusted
                # terminal evidence merely by inheriting a bundled adapter.
                yield ModelStreamEvent.completed({"finish_reason": "stop"})

    task = asyncio.create_task(anext(OverridingChatProvider().runtime_stream(_request())))
    await asyncio.sleep(0)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert task.cancelled()


@pytest.mark.anyio
async def test_terminal_created_only_after_deadline_cancellation_is_not_accepted() -> None:
    controller = ProviderStreamDeadlineController(
        ProviderStreamDeadlines(
            transport_idle_timeout_s=1,
            protocol_idle_timeout_s=1,
            semantic_progress_timeout_s=0.01,
            absolute_stream_timeout_s=1,
        )
    )

    async def fabricate_terminal_after_cancellation() -> ModelStreamEvent:
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            controller.observe_semantic(ProviderProgressKind.TERMINAL)
            return ModelStreamEvent.completed({"finish_reason": "stop"})

    try:
        with pytest.raises(ProviderStreamDeadlineExceeded):
            await controller.wait_for(
                fabricate_terminal_after_cancellation(),
                kinds=(ProviderDeadlineKind.SEMANTIC_IDLE,),
                accept_cancelled_result=lambda event: event.type is ModelStreamEventType.COMPLETED,
            )
    finally:
        controller.close()


@pytest.mark.anyio
async def test_terminal_preservation_never_downgrades_process_control() -> None:
    controller = ProviderStreamDeadlineController(
        ProviderStreamDeadlines(
            transport_idle_timeout_s=1,
            protocol_idle_timeout_s=1,
            semantic_progress_timeout_s=0.01,
            absolute_stream_timeout_s=1,
        )
    )
    controller.observe_semantic(ProviderProgressKind.TERMINAL)

    async def process_control_during_cancellation() -> ModelStreamEvent:
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            raise GeneratorExit from None

    try:
        with pytest.raises(GeneratorExit):
            await controller.wait_for(
                process_control_during_cancellation(),
                kinds=(ProviderDeadlineKind.SEMANTIC_IDLE,),
                accept_cancelled_result=lambda event: event.type is ModelStreamEventType.COMPLETED,
            )
    finally:
        controller.close()


@pytest.mark.anyio
async def test_deadline_remains_authoritative_when_cancelled_read_cleanup_fails() -> None:
    secret = "provider-cleanup-secret-must-not-survive"

    async def transformed_cleanup() -> ModelStreamEvent:
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            raise RuntimeError(secret) from None

    controller = ProviderStreamDeadlineController(
        ProviderStreamDeadlines(
            transport_idle_timeout_s=1,
            protocol_idle_timeout_s=1,
            semantic_progress_timeout_s=0.01,
            absolute_stream_timeout_s=1,
        )
    )

    with pytest.raises(ProviderStreamDeadlineExceeded) as captured:
        await controller.wait_for(
            transformed_cleanup(),
            kinds=(ProviderDeadlineKind.SEMANTIC_IDLE,),
        )

    assert captured.value.evidence.deadline_kind is ProviderDeadlineKind.SEMANTIC_IDLE
    assert captured.value.stream_cleanup_failed is True
    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None
    assert secret not in repr(captured.value)


@pytest.mark.anyio
async def test_runtime_deadline_survives_explicit_inner_close_failure() -> None:
    secret = "provider-close-secret-must-not-survive"

    class FailingCloseEvents:
        def __aiter__(self) -> FailingCloseEvents:
            return self

        async def __anext__(self) -> ModelStreamEvent:
            await asyncio.Event().wait()
            raise StopAsyncIteration  # pragma: no cover

        async def aclose(self) -> None:
            raise RuntimeError(secret)

    class FailingCloseProvider(ModelProvider):
        name = "failing-close"

        @property
        def stream_deadlines(self) -> ProviderStreamDeadlines:
            return ProviderStreamDeadlines(
                transport_idle_timeout_s=1,
                protocol_idle_timeout_s=1,
                semantic_progress_timeout_s=0.01,
                absolute_stream_timeout_s=1,
            )

        def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
            del request
            return FailingCloseEvents()

    with pytest.raises(ModelStreamDeadlineError) as captured:
        await anext(FailingCloseProvider().runtime_stream(_request()))

    failure = captured.value
    assert failure.deadline_evidence.deadline_kind is ProviderDeadlineKind.SEMANTIC_IDLE
    assert failure.stream_cleanup_failed is True
    assert failure.error_payload_fields()["stream_cleanup_failed"] is True
    assert failure.__cause__ is None
    assert failure.__context__ is None
    assert secret not in repr(failure)
    assert secret not in repr(failure.error_payload_fields())


@pytest.mark.anyio
async def test_runtime_deadline_does_not_wait_for_nonsettling_inner_close() -> None:
    close_started = asyncio.Event()
    close_release = asyncio.Event()
    close_finished = asyncio.Event()

    class BlockingCloseEvents:
        def __aiter__(self) -> BlockingCloseEvents:
            return self

        async def __anext__(self) -> ModelStreamEvent:
            await asyncio.Event().wait()
            raise StopAsyncIteration  # pragma: no cover

        async def aclose(self) -> None:
            close_started.set()
            await close_release.wait()
            close_finished.set()

    class BlockingCloseProvider(ModelProvider):
        name = "blocking-close"

        @property
        def stream_deadlines(self) -> ProviderStreamDeadlines:
            return ProviderStreamDeadlines(
                transport_idle_timeout_s=1,
                protocol_idle_timeout_s=1,
                semantic_progress_timeout_s=0.01,
                absolute_stream_timeout_s=1,
            )

        def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
            del request
            return BlockingCloseEvents()

    stream = BlockingCloseProvider().runtime_stream(_request())
    try:
        with pytest.raises(ModelStreamDeadlineError) as captured:
            await asyncio.wait_for(anext(stream), timeout=0.5)

        assert captured.value.deadline_evidence.deadline_kind is (
            ProviderDeadlineKind.SEMANTIC_IDLE
        )
        assert captured.value.stream_cleanup_failed is True
        assert close_started.is_set()
        assert not close_finished.is_set()
    finally:
        close_release.set()
        await asyncio.wait_for(close_finished.wait(), timeout=0.5)


@pytest.mark.anyio
async def test_deadline_error_copy_retains_validated_content_free_evidence() -> None:
    provider = _DeadlineProvider(
        _blocked_events(),
        ProviderStreamDeadlines(
            transport_idle_timeout_s=1,
            protocol_idle_timeout_s=1,
            semantic_progress_timeout_s=0.01,
            absolute_stream_timeout_s=1,
        ),
    )
    with pytest.raises(ModelStreamDeadlineError) as captured:
        await anext(provider.runtime_stream(_request()))

    copied = copy_provider_exception_control(captured.value).cause

    assert type(copied) is ModelStreamDeadlineError
    assert copied.error_payload_fields() == captured.value.error_payload_fields()
    assert copied.error_payload_fields()["provider_effect_outcome"] == "unknown"
    assert (
        copied.error_payload_fields()["provider_recovery_disposition"]
        == "manual_settlement_required"
    )


@pytest.mark.anyio
async def test_runtime_guard_closes_the_inner_custom_provider_stream() -> None:
    closed = False

    class ClosingProvider(ModelProvider):
        name = "closing-provider"

        async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
            nonlocal closed
            del request
            try:
                yield ModelStreamEvent.text_delta("one")
                await asyncio.Event().wait()
            finally:
                closed = True

    stream = ClosingProvider().runtime_stream(_request())

    assert (await anext(stream)).delta == "one"
    await stream.aclose()

    assert closed is True


@pytest.mark.anyio
async def test_downstream_processing_time_does_not_consume_semantic_idle_budget() -> None:
    async def events() -> AsyncIterator[ModelStreamEvent]:
        yield ModelStreamEvent.text_delta("one")
        yield ModelStreamEvent.text_delta("two")
        yield ModelStreamEvent.completed({"finish_reason": "stop"})

    provider = _DeadlineProvider(
        events(),
        ProviderStreamDeadlines(
            transport_idle_timeout_s=1,
            protocol_idle_timeout_s=1,
            semantic_progress_timeout_s=0.01,
            absolute_stream_timeout_s=1,
        ),
    )
    stream = provider.runtime_stream(_request())

    assert (await anext(stream)).delta == "one"
    await asyncio.sleep(0.03)
    assert (await anext(stream)).delta == "two"
    await stream.aclose()


@pytest.mark.anyio
async def test_downstream_pause_does_not_rewrite_last_progress_time() -> None:
    async def events() -> AsyncIterator[ModelStreamEvent]:
        yield ModelStreamEvent.text_delta("one")
        await asyncio.Event().wait()

    provider = _DeadlineProvider(
        events(),
        ProviderStreamDeadlines(
            transport_idle_timeout_s=1,
            protocol_idle_timeout_s=1,
            semantic_progress_timeout_s=0.01,
            absolute_stream_timeout_s=1,
        ),
    )
    stream = provider.runtime_stream(_request())

    assert (await anext(stream)).delta == "one"
    await asyncio.sleep(0.03)
    with pytest.raises(ModelStreamDeadlineError) as captured:
        await anext(stream)

    evidence = captured.value.deadline_evidence
    assert evidence.last_progress_elapsed_s is not None
    assert evidence.elapsed_s - evidence.last_progress_elapsed_s >= 0.03


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("event", "expected_kind"),
    [
        pytest.param(
            ModelStreamEvent.thinking("reasoning"),
            ProviderProgressKind.REASONING,
            id="reasoning",
        ),
        pytest.param(
            ModelStreamEvent.text_delta("content"),
            ProviderProgressKind.CONTENT,
            id="content",
        ),
        pytest.param(
            ModelStreamEvent.tool_call(name="lookup", arguments={}, id="call-1"),
            ProviderProgressKind.TOOL_CALL,
            id="tool-call",
        ),
        pytest.param(
            ModelStreamEvent.hosted_tool_call({"status": "searching"}),
            ProviderProgressKind.HOSTED_TOOL,
            id="hosted-tool",
        ),
        pytest.param(
            ModelStreamEvent.citation({"url": "https://example.test"}),
            ProviderProgressKind.CITATION,
            id="citation",
        ),
        pytest.param(
            ModelStreamEvent.completed({"finish_reason": "stop"}),
            ProviderProgressKind.TERMINAL,
            id="terminal",
        ),
    ],
)
async def test_normalized_progress_transitions_have_explicit_semantics(
    event: ModelStreamEvent,
    expected_kind: ProviderProgressKind,
) -> None:
    async def events() -> AsyncIterator[ModelStreamEvent]:
        yield event
        await asyncio.Event().wait()

    provider = _DeadlineProvider(
        events(),
        ProviderStreamDeadlines(
            transport_idle_timeout_s=1,
            protocol_idle_timeout_s=1,
            semantic_progress_timeout_s=0.01,
            absolute_stream_timeout_s=1,
        ),
    )
    stream = provider.runtime_stream(_request())

    assert await anext(stream) == event
    with pytest.raises(ModelStreamDeadlineError) as captured:
        await anext(stream)

    assert captured.value.deadline_evidence.last_progress_kind is expected_kind


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("raw_event", "expected_kind"),
    [
        pytest.param(
            {"id": "response-1", "choices": []},
            ProviderProgressKind.RESPONSE_IDENTITY,
            id="response-identity",
        ),
        pytest.param(
            {"usage": {"input_tokens": 1}, "choices": []},
            ProviderProgressKind.USAGE,
            id="permitted-usage-tail",
        ),
    ],
)
async def test_chat_lifecycle_only_refreshes_declared_metadata_progress(
    raw_event: dict[str, object],
    expected_kind: ProviderProgressKind,
) -> None:
    async def raw_events() -> AsyncIterator[dict[str, object]]:
        yield raw_event

    controller = ProviderStreamDeadlineController(ProviderStreamDeadlines())
    token = bind_provider_deadline_controller(controller)
    try:
        with pytest.raises(ChatCompletionsProtocolError, match="finish_reason"):
            async for _ in chat_completions_stream_events(raw_events()):
                pass
    finally:
        reset_provider_deadline_controller(token)

    evidence = controller.evidence((ProviderDeadlineKind.SEMANTIC_IDLE,))
    assert evidence.last_progress_kind is expected_kind


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("reasoning_details", "expected_kind"),
    [
        pytest.param(
            [{"type": "reasoning.encrypted", "data": "opaque"}],
            ProviderProgressKind.REASONING,
            id="accepted-reasoning-state",
        ),
        pytest.param([], None, id="empty-reasoning-state"),
    ],
)
async def test_chat_reasoning_details_refresh_only_accepted_nonempty_state(
    reasoning_details: list[dict[str, str]],
    expected_kind: ProviderProgressKind | None,
) -> None:
    async def raw_events() -> AsyncIterator[dict[str, object]]:
        yield {
            "choices": [
                {
                    "index": 0,
                    "delta": {"reasoning_details": reasoning_details},
                }
            ]
        }

    controller = ProviderStreamDeadlineController(ProviderStreamDeadlines())
    token = bind_provider_deadline_controller(controller)
    try:
        with pytest.raises(ChatCompletionsProtocolError, match="finish_reason"):
            async for _ in chat_completions_stream_events(
                raw_events(),
                provider_state_target_sha256="a" * 64,
            ):
                pass
    finally:
        reset_provider_deadline_controller(token)

    evidence = controller.evidence((ProviderDeadlineKind.SEMANTIC_IDLE,))
    assert evidence.last_progress_kind is expected_kind


@pytest.mark.anyio
async def test_deadline_error_event_reconstructs_exact_typed_recovery_control() -> None:
    provider = _DeadlineProvider(
        _blocked_events(),
        ProviderStreamDeadlines(
            transport_idle_timeout_s=1,
            protocol_idle_timeout_s=1,
            semantic_progress_timeout_s=0.01,
            absolute_stream_timeout_s=1,
        ),
    )
    with pytest.raises(ModelStreamDeadlineError) as captured:
        await anext(provider.runtime_stream(_request()))
    deadline = ModelStreamDeadlineError(
        provider=provider.name,
        evidence=captured.value.deadline_evidence,
        stream_cleanup_failed=True,
    )
    payload = ModelStreamEvent.error(str(deadline), cause=deadline).payload

    reconstructed = model_provider_error_from_payload(
        payload,
        fallback_provider=provider.name,
    )

    assert type(reconstructed) is ModelStreamDeadlineError
    assert reconstructed.error_payload_fields() == deadline.error_payload_fields()
    assert reconstructed.stream_cleanup_failed is True


def test_malformed_deadline_error_event_fails_closed_without_timeout_authority() -> None:
    reconstructed = model_provider_error_from_payload(
        {
            "error": "untrusted deadline",
            "error_type": "ModelStreamDeadlineError",
            "provider": "untrusted",
            "provider_deadline_kind": "semantic_idle",
            "provider_deadline_timeout_s": 1.0,
            "provider_stream_elapsed_s": 1.0,
            "provider_effect_outcome": "none",
            "provider_recovery_disposition": "retry",
            "retryable": True,
        },
        fallback_provider="fallback",
    )

    assert type(reconstructed) is ModelProviderError
    assert reconstructed.provider == "fallback"
    assert reconstructed.error_code == "invalid_provider_stream_deadline_evidence"
    assert reconstructed.retryable is False


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("provider_deadline_kind", "semantic_idle"),
        ("provider_deadline_timeout_s", 1.0),
        ("provider_stream_elapsed_s", 1.0),
        ("provider_last_progress_kind", "reasoning"),
        ("provider_last_progress_elapsed_s", 0.5),
        ("provider_last_progress_at", "2026-01-01T00:00:00+00:00"),
        ("provider_effect_outcome", "unknown"),
        ("provider_recovery_disposition", "manual_settlement_required"),
        ("provider_error_code", "provider_stream_semantic_idle_timeout"),
    ],
)
def test_partial_deadline_claim_fails_closed_without_timeout_authority(
    field: str, value: object
) -> None:
    reconstructed = model_provider_error_from_payload(
        {
            "error": "untrusted deadline",
            "provider": "untrusted",
            field: value,
        },
        fallback_provider="fallback",
    )

    assert type(reconstructed) is ModelProviderError
    assert reconstructed.provider == "fallback"
    assert reconstructed.error_code == "invalid_provider_stream_deadline_evidence"
    assert reconstructed.retryable is False


def test_cleanup_failure_diagnostic_does_not_claim_stream_deadline() -> None:
    reconstructed = model_provider_error_from_payload(
        {
            "error": "Provider stream cleanup failed.",
            "error_type": "ProviderStreamCleanupError",
            "provider": "cleanup-provider",
            "provider_error_type": "ProviderStreamCleanupError",
            "provider_error_code": "stream_cleanup_failed",
            "retryable": False,
            "stream_cleanup_failed": True,
        },
        fallback_provider="fallback",
    )

    assert type(reconstructed) is ModelProviderError
    assert reconstructed.provider == "cleanup-provider"
    assert reconstructed.error_type == "ProviderStreamCleanupError"
    assert reconstructed.error_code == "stream_cleanup_failed"
    assert reconstructed.retryable is False


@pytest.mark.parametrize(
    "provider_error_code",
    [
        pytest.param([], id="list"),
        pytest.param({}, id="object"),
    ],
)
def test_wrong_type_provider_error_code_cannot_bypass_deadline_claim(
    provider_error_code: object,
) -> None:
    reconstructed = model_provider_error_from_payload(
        {
            "error": "untrusted deadline",
            "provider": "untrusted",
            "provider_error_code": provider_error_code,
            "provider_deadline_kind": "semantic_idle",
        },
        fallback_provider="fallback",
    )

    assert type(reconstructed) is ModelProviderError
    assert reconstructed.provider == "fallback"
    assert reconstructed.error_code == "invalid_provider_stream_deadline_evidence"
    assert reconstructed.retryable is False

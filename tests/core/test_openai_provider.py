from __future__ import annotations

import asyncio
import gzip
import json
from collections.abc import Mapping
from decimal import Decimal
from typing import Any

import httpx
import pytest
from tests.provider_traceback_assertions import assert_cayu_traceback_does_not_retain

import cayu.providers.openai as openai_module
from cayu import (
    RESOLVED_FILE_ATTACHMENTS_OPTION,
    AgentSpec,
    BudgetLimit,
    CayuApp,
    Event,
    EventType,
    FileAttachmentKind,
    InMemorySessionStore,
    Message,
    ModelPrice,
    PriceBook,
    RecentTurnsContextPolicy,
    ResumeRequest,
    RetryPolicy,
    RunRequest,
    file_attachment,
)
from cayu.core.messages import FilePart, MessageRole, ProviderStatePart, TextPart, ToolCallPart
from cayu.core.tools import Tool, ToolContext, ToolResult, ToolSpec
from cayu.embeddings import TextEmbeddingRequest
from cayu.providers import (
    HttpxOpenAITransport,
    InputTokenCountConfidence,
    InputTokenCountMethod,
    ModelContextOverflowError,
    ModelFinishReason,
    ModelProvider,
    ModelRequest,
    ModelStreamEvent,
    ModelStreamEventType,
    NativeStructuredOutputSchemaInvalid,
    OpenAIAPIError,
    OpenAIContextOverflowError,
    OpenAIProtocolError,
    OpenAIProvider,
    UsageDialect,
    build_openai_embedding_payload,
    build_openai_payload,
    openai_embedding_result,
    openai_response_events,
    preflight_openai_native_structured_output_schema,
)
from cayu.providers._http import MAX_PROVIDER_ERROR_BODY_CHARS
from cayu.providers._sse import aiter_sse_json_events
from cayu.providers.openai import openai_stream_events
from cayu.runtime.execution_profiles import execution_profile_from_session_metadata


async def _collect_events(app: CayuApp, request: RunRequest) -> list[Event]:
    return [event async for event in app.run(request)]


class RecordingTransport:
    def __init__(
        self,
        stream_events: list[list[Mapping[str, Any]]] | None = None,
        count_responses: list[Mapping[str, Any]] | None = None,
    ) -> None:
        self.stream_event_batches = list(stream_events or [])
        self.count_responses = list(count_responses or [])
        self.calls: list[dict[str, Any]] = []
        self.count_calls: list[dict[str, Any]] = []

    async def create_response(
        self,
        *,
        url: str,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
        timeout_s: float,
    ) -> Mapping[str, Any]:
        self.count_calls.append(
            {
                "url": url,
                "headers": dict(headers),
                "payload": dict(payload),
                "timeout_s": timeout_s,
            }
        )
        if not self.count_responses:
            raise AssertionError("No fake OpenAI count response queued.")
        return self.count_responses.pop(0)

    async def stream_response_events(
        self,
        *,
        url: str,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
        timeout_s: float,
        stream_idle_timeout_s: float,
    ):
        self.calls.append(
            {
                "url": url,
                "headers": dict(headers),
                "payload": dict(payload),
                "timeout_s": timeout_s,
                "stream_idle_timeout_s": stream_idle_timeout_s,
            }
        )
        if not self.stream_event_batches:
            raise AssertionError("No fake OpenAI stream queued.")
        for event in self.stream_event_batches.pop(0):
            yield event


class BlankFailingTransport:
    async def create_response(
        self,
        *,
        url: str,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
        timeout_s: float,
    ) -> Mapping[str, Any]:
        raise RuntimeError()

    async def stream_response_events(
        self,
        *,
        url: str,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
        timeout_s: float,
        stream_idle_timeout_s: float,
    ):
        raise RuntimeError()
        yield {}


class EchoTool(Tool):
    spec = ToolSpec(
        name="echo",
        description="Echo text.",
        input_schema={
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    )

    async def run(self, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
        return ToolResult(
            content=args["text"],
            structured={"agent": ctx.agent_name, "echoed": args["text"]},
        )


@pytest.mark.anyio
async def test_agent_hosted_web_search_projects_native_openai_tool() -> None:
    from cayu.providers import OpenAIWebSearch

    transport = RecordingTransport(
        stream_events=[
            [
                {
                    "type": "response.completed",
                    "response": {
                        "id": "resp_1",
                        "model": "gpt-test",
                        "status": "completed",
                        "output": [
                            {
                                "type": "message",
                                "role": "assistant",
                                "content": [{"type": "output_text", "text": "done"}],
                            }
                        ],
                        "usage": {
                            "input_tokens": 1,
                            "output_tokens": 1,
                            "total_tokens": 2,
                        },
                    },
                }
            ]
        ]
    )
    app = CayuApp()
    app.register_provider(OpenAIProvider(api_key="test-key", transport=transport), default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="gpt-5.6"),
        tools=[EchoTool()],
        hosted_tools=[
            OpenAIWebSearch(
                search_context_size="medium",
                external_web_access=False,
                allowed_domains=("openai.com",),
                return_token_budget="default",
                include_sources=True,
            )
        ],
    )

    events = await _collect_events(
        app,
        RunRequest(
            agent_name="assistant",
            messages=[Message.text("user", "Search the official docs.")],
            max_steps=1,
        ),
    )

    assert events[-1].type == EventType.SESSION_COMPLETED
    assert transport.calls[0]["payload"]["tools"] == [
        {
            "type": "function",
            "name": "echo",
            "description": "Echo text.",
            "parameters": EchoTool.spec.input_schema,
            "strict": False,
        },
        {
            "type": "web_search",
            "search_context_size": "medium",
            "external_web_access": False,
            "filters": {"allowed_domains": ["openai.com"]},
        },
    ]
    assert transport.calls[0]["payload"]["include"] == [
        "reasoning.encrypted_content",
        "web_search_call.action.sources",
    ]


@pytest.mark.anyio
async def test_agent_hosted_web_search_rejects_unsupported_provider_before_dispatch() -> None:
    from cayu.providers import OpenAIWebSearch

    class UnsupportedProvider(ModelProvider):
        name = "unsupported"

        def __init__(self) -> None:
            self.calls = 0

        async def stream(self, request: ModelRequest):
            self.calls += 1
            yield ModelStreamEvent.completed(
                {
                    "model": request.model,
                    "status": "completed",
                    "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                }
            )

    provider = UnsupportedProvider()
    app = CayuApp()
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="gpt-5.6"),
        hosted_tools=[OpenAIWebSearch()],
    )

    with pytest.raises(ValueError, match="hosted.*not supported"):
        await _collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                messages=[Message.text("user", "Search the web.")],
                max_steps=1,
            ),
        )

    assert provider.calls == 0


def test_openai_hosted_web_search_custom_endpoint_requires_explicit_capability() -> None:
    from cayu.providers import HostedToolCapabilityError, OpenAIWebSearch

    provider = OpenAIProvider(
        api_key="test-key",
        base_url="https://compatible.example.test/v1",
        transport=RecordingTransport(),
    )

    with pytest.raises(HostedToolCapabilityError, match="not established for this custom endpoint"):
        provider.preflight_hosted_tools(
            model="gpt-5.6",
            hosted_tools=(OpenAIWebSearch(),),
            options={},
        )

    verified_provider = OpenAIProvider(
        api_key="test-key",
        base_url="https://compatible.example.test/v1",
        transport=RecordingTransport(),
        hosted_web_search_supported=True,
    )
    verified_provider.preflight_hosted_tools(
        model="gpt-5.6",
        hosted_tools=(OpenAIWebSearch(),),
        options={},
    )


def test_openai_hosted_web_search_unlimited_budget_requires_gpt5_reasoning() -> None:
    from cayu.providers import HostedToolCapabilityError, OpenAIWebSearch

    provider = OpenAIProvider(api_key="test-key", transport=RecordingTransport())

    with pytest.raises(HostedToolCapabilityError, match="requires a GPT-5"):
        provider.preflight_hosted_tools(
            model="chat-latest",
            hosted_tools=(OpenAIWebSearch(return_token_budget="unlimited"),),
            options={},
        )


@pytest.mark.parametrize("model", ["gpt-4.1", "gpt-5.6-unverified", "gpt-5.7"])
def test_openai_hosted_web_search_rejects_model_without_verified_support(model: str) -> None:
    from cayu.providers import HostedToolCapabilityError, OpenAIWebSearch

    provider = OpenAIProvider(api_key="test-key", transport=RecordingTransport())

    with pytest.raises(HostedToolCapabilityError, match=f"not established for model {model!r}"):
        provider.preflight_hosted_tools(
            model=model,
            hosted_tools=(OpenAIWebSearch(),),
            options={},
        )


@pytest.mark.anyio
async def test_agent_hosted_web_search_rejects_strict_budget_without_call_ceiling() -> None:
    from cayu.providers import HostedToolCapabilityError, OpenAIWebSearch

    transport = RecordingTransport()
    app = CayuApp()
    app.register_provider(OpenAIProvider(api_key="test-key", transport=transport), default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="gpt-5.6"),
        hosted_tools=[OpenAIWebSearch()],
    )

    with pytest.raises(HostedToolCapabilityError, match="no hard per-response"):
        await _collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                messages=[Message.text("user", "Search the web.")],
                max_steps=1,
                budget_limits=(
                    BudgetLimit(
                        max_estimated_cost=Decimal("1"),
                        pricing=PriceBook(
                            prices=(
                                ModelPrice.fixed(
                                    provider_name="openai",
                                    model="gpt-5.6",
                                    input_per_million=Decimal("1"),
                                    output_per_million=Decimal("1"),
                                    web_search_per_thousand=Decimal("10"),
                                ),
                            )
                        ),
                    ),
                ),
            ),
        )

    assert transport.calls == []


@pytest.mark.anyio
async def test_hosted_web_search_configuration_changes_execution_profile_identity() -> None:
    from cayu.providers import OpenAIWebSearch

    async def run_with(search_context_size: str) -> str:
        transport = RecordingTransport(
            stream_events=[
                [
                    {
                        "type": "response.completed",
                        "response": {
                            "id": f"resp_{search_context_size}",
                            "model": "gpt-test",
                            "status": "completed",
                            "output": [
                                {
                                    "type": "message",
                                    "role": "assistant",
                                    "content": [{"type": "output_text", "text": "done"}],
                                }
                            ],
                            "usage": {
                                "input_tokens": 1,
                                "output_tokens": 1,
                                "total_tokens": 2,
                            },
                        },
                    }
                ]
            ]
        )
        store = InMemorySessionStore()
        app = CayuApp(session_store=store)
        app.register_provider(
            OpenAIProvider(api_key="test-key", transport=transport),
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="gpt-5.6"),
            hosted_tools=[OpenAIWebSearch(search_context_size=search_context_size)],
        )
        events = await _collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                messages=[Message.text("user", "Search.")],
                max_steps=1,
            ),
        )
        session = await store.load(events[0].session_id)
        assert session is not None
        return execution_profile_from_session_metadata(session.metadata).fingerprint

    assert await run_with("low") != await run_with("high")


@pytest.mark.anyio
async def test_runtime_persists_web_search_lifecycle_citation_sources_and_replay() -> None:
    from cayu import CitationPart, HostedToolCallPart, OpenAIWebSearch

    transport = RecordingTransport(
        stream_events=[
            [
                {
                    "type": "response.output_item.added",
                    "output_index": 0,
                    "item": {
                        "type": "web_search_call",
                        "id": "ws_1",
                        "status": "in_progress",
                    },
                },
                {
                    "type": "response.web_search_call.searching",
                    "output_index": 0,
                    "item_id": "ws_1",
                },
                {
                    "type": "response.output_item.done",
                    "output_index": 0,
                    "item": {
                        "type": "web_search_call",
                        "id": "ws_1",
                        "status": "completed",
                        "action": {
                            "type": "search",
                            "query": "Cayu runtime",
                            "sources": [
                                {
                                    "type": "url",
                                    "url": "https://github.com/example/cayu",
                                    "title": "Cayu",
                                }
                            ],
                        },
                    },
                },
                {
                    "type": "response.output_text.delta",
                    "output_index": 1,
                    "content_index": 0,
                    "delta": "Prefix ",
                },
                {
                    "type": "response.output_text.delta",
                    "output_index": 1,
                    "content_index": 1,
                    "delta": "Cayu is a runtime.",
                },
                {
                    "type": "response.output_text.annotation.added",
                    "output_index": 1,
                    "content_index": 1,
                    "annotation": {
                        "type": "url_citation",
                        "url": "https://github.com/example/cayu",
                        "title": "Cayu",
                        "start_index": 0,
                        "end_index": 4,
                    },
                },
                {
                    "type": "response.completed",
                    "response": {
                        "id": "resp_1",
                        "model": "gpt-test",
                        "status": "completed",
                        "usage": {
                            "input_tokens": 3,
                            "output_tokens": 4,
                            "total_tokens": 7,
                        },
                    },
                },
            ],
            [
                {"type": "response.output_text.delta", "delta": "Follow-up."},
                {
                    "type": "response.completed",
                    "response": {
                        "id": "resp_2",
                        "model": "gpt-test",
                        "status": "completed",
                        "output": [
                            {
                                "type": "message",
                                "role": "assistant",
                                "content": [
                                    {
                                        "type": "output_text",
                                        "text": "Follow-up.",
                                        "annotations": [],
                                    }
                                ],
                            }
                        ],
                        "usage": {
                            "input_tokens": 5,
                            "output_tokens": 2,
                            "total_tokens": 7,
                        },
                    },
                },
            ],
        ]
    )
    store = InMemorySessionStore()
    app = CayuApp(session_store=store)
    app.register_provider(OpenAIProvider(api_key="test-key", transport=transport), default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="gpt-5.6"),
        hosted_tools=[OpenAIWebSearch()],
    )

    run_events = await _collect_events(
        app,
        RunRequest(
            session_id="sess_web_search",
            agent_name="assistant",
            messages=[Message.text("user", "Search for Cayu.")],
            max_steps=1,
        ),
    )
    resume_events = [
        event
        async for event in app.resume(
            ResumeRequest(
                session_id="sess_web_search",
                messages=[Message.text("user", "Continue.")],
                max_steps=1,
            )
        )
    ]

    hosted_events = [
        event for event in run_events if event.type == EventType.MODEL_HOSTED_TOOL_CALL
    ]
    assert [event.payload["status"] for event in hosted_events] == [
        "in_progress",
        "searching",
        "completed",
    ]
    assert all(event.payload["provider_name"] == "openai" for event in hosted_events)
    assert all(event.payload["model"] == "gpt-5.6" for event in hosted_events)
    assert len({event.payload["model_attempt_id"] for event in hosted_events}) == 1
    assert len({event.payload["provider_operation_id"] for event in hosted_events}) == 1
    citation_event = next(event for event in run_events if event.type == EventType.MODEL_CITATION)
    assert (
        citation_event.payload["provider_operation_id"]
        == hosted_events[0].payload["provider_operation_id"]
    )
    assert citation_event.payload["provenance"] == {
        "provider_name": "openai",
        "hosted_tool": "web_search",
        "untrusted_external_evidence": True,
    }
    completed = next(event for event in run_events if event.type == EventType.MODEL_COMPLETED)
    assert "hosted_tools" not in completed.payload["usage_metrics"]
    assert completed.payload["hosted_tool_usage"] == {
        "web_search_calls": 1,
        "web_search_outcome_unknown": 0,
    }

    transcript = await store.load_transcript("sess_web_search")
    searched_message = next(
        message
        for message in transcript
        if message.role == MessageRole.ASSISTANT
        and any(type(part) is HostedToolCallPart for part in message.content)
    )
    hosted_part = next(
        part for part in searched_message.content if type(part) is HostedToolCallPart
    )
    citation_part = next(part for part in searched_message.content if type(part) is CitationPart)
    assert hosted_part.action.sources[0].url == "https://github.com/example/cayu"
    stored_hosted_events = [
        event
        for event in await store.load_events("sess_web_search")
        if event.type == EventType.MODEL_HOSTED_TOOL_CALL
    ]
    assert hosted_part.model_attempt_id == stored_hosted_events[-1].payload["model_attempt_id"]
    assert citation_part.start_index == 7
    assert citation_part.end_index == 11
    assert resume_events[-1].type == EventType.SESSION_COMPLETED
    usage = await app.get_session_usage("sess_web_search")
    assert usage.usage.hosted_tools.web_search_calls == 1
    assert usage.usage.hosted_tools.web_search_outcome_unknown == 0
    assert any(
        item.get("type") == "web_search_call" for item in transport.calls[1]["payload"]["input"]
    )


@pytest.mark.anyio
async def test_runtime_accounts_terminal_only_web_search_evidence_once() -> None:
    from cayu import OpenAIWebSearch

    transport = RecordingTransport(
        stream_events=[
            [
                {
                    "type": "response.completed",
                    "response": {
                        "id": "resp_terminal_only",
                        "model": "gpt-test",
                        "status": "completed",
                        "output": [
                            {
                                "type": "web_search_call",
                                "id": "ws_terminal_only",
                                "status": "completed",
                                "action": {"type": "search", "query": "Cayu"},
                            }
                        ],
                        "usage": {
                            "input_tokens": 1,
                            "output_tokens": 1,
                            "total_tokens": 2,
                        },
                    },
                }
            ]
        ]
    )
    store = InMemorySessionStore()
    app = CayuApp(session_store=store)
    app.register_provider(OpenAIProvider(api_key="test-key", transport=transport), default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="gpt-5.6"),
        hosted_tools=[OpenAIWebSearch()],
    )

    events = await _collect_events(
        app,
        RunRequest(
            session_id="sess_terminal_only_search",
            agent_name="assistant",
            messages=[Message.text("user", "Search")],
            max_steps=1,
        ),
    )

    hosted = [event for event in events if event.type == EventType.MODEL_HOSTED_TOOL_CALL]
    assert [event.payload["status"] for event in hosted] == ["completed"]
    stored_hosted = [
        event
        for event in await store.load_events("sess_terminal_only_search")
        if event.type == EventType.MODEL_HOSTED_TOOL_CALL
    ]
    assert [(event.payload["call_id"], event.payload["status"]) for event in stored_hosted] == [
        ("ws_terminal_only", "completed")
    ]
    completed = next(event for event in events if event.type == EventType.MODEL_COMPLETED)
    assert completed.payload["hosted_tool_usage"] == {
        "web_search_calls": 1,
        "web_search_outcome_unknown": 0,
    }
    usage = await app.get_session_usage("sess_terminal_only_search")
    assert usage.usage.hosted_tools.web_search_calls == 1
    assert usage.usage.hosted_tools.web_search_outcome_unknown == 0


@pytest.mark.anyio
async def test_runtime_accounts_for_ambiguous_search_before_provider_retry() -> None:
    from cayu import OpenAIWebSearch

    transport = RecordingTransport(
        stream_events=[
            [
                {
                    "type": "response.output_item.added",
                    "output_index": 0,
                    "item": {
                        "type": "web_search_call",
                        "id": "ws_first",
                        "status": "in_progress",
                    },
                },
                {
                    "type": "response.failed",
                    "response": {
                        "error": {
                            "type": "server_error",
                            "code": "internal_error",
                            "message": "transient",
                        }
                    },
                },
            ],
            [
                {
                    "type": "response.output_item.added",
                    "output_index": 0,
                    "item": {
                        "type": "web_search_call",
                        "id": "ws_retry",
                        "status": "in_progress",
                    },
                },
                {
                    "type": "response.output_item.done",
                    "output_index": 0,
                    "item": {
                        "type": "web_search_call",
                        "id": "ws_retry",
                        "status": "completed",
                        "action": {"type": "search", "query": "Cayu"},
                    },
                },
                {
                    "type": "response.completed",
                    "response": {
                        "id": "resp_retry",
                        "model": "gpt-test",
                        "status": "completed",
                        "usage": {"input_tokens": 2, "output_tokens": 1, "total_tokens": 3},
                    },
                },
            ],
        ]
    )
    store = InMemorySessionStore()
    app = CayuApp(session_store=store)
    app.register_provider(OpenAIProvider(api_key="test-key", transport=transport), default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="gpt-5.6"),
        hosted_tools=[OpenAIWebSearch()],
    )

    events = await _collect_events(
        app,
        RunRequest(
            session_id="sess_web_search_retry",
            agent_name="assistant",
            messages=[Message.text("user", "Search")],
            max_steps=1,
            retry_policy=RetryPolicy(max_attempts=2, initial_delay_s=0.0),
        ),
    )

    assert len(transport.calls) == 2
    hosted = [event for event in events if event.type == EventType.MODEL_HOSTED_TOOL_CALL]
    assert [event.payload["status"] for event in hosted] == [
        "in_progress",
        "outcome_unknown",
        "in_progress",
        "completed",
    ]
    stored_hosted = [
        event
        for event in await store.load_events("sess_web_search_retry")
        if event.type == EventType.MODEL_HOSTED_TOOL_CALL
    ]
    assert [(event.payload["call_id"], event.payload["status"]) for event in stored_hosted] == [
        ("ws_first", "in_progress"),
        ("ws_first", "outcome_unknown"),
        ("ws_retry", "in_progress"),
        ("ws_retry", "completed"),
    ]
    assert EventType.MODEL_RETRY in [event.type for event in events]
    usage = await app.get_session_usage("sess_web_search_retry")
    assert usage.usage.hosted_tools.web_search_calls == 1
    assert usage.usage.hosted_tools.web_search_outcome_unknown == 1


@pytest.mark.anyio
async def test_runtime_accounts_started_search_before_later_parser_failure() -> None:
    from cayu import OpenAIWebSearch

    transport = RecordingTransport(
        stream_events=[
            [
                {
                    "type": "response.output_item.added",
                    "output_index": 0,
                    "item": {
                        "type": "web_search_call",
                        "id": "ws_parser_failure",
                        "status": "in_progress",
                    },
                },
                {
                    "type": "response.web_search_call.searching",
                    "output_index": 0,
                    "item_id": "ws_wrong",
                },
            ]
        ]
    )
    store = InMemorySessionStore()
    app = CayuApp(session_store=store)
    app.register_provider(OpenAIProvider(api_key="test-key", transport=transport), default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="gpt-5.6"),
        hosted_tools=[OpenAIWebSearch()],
    )

    events = await _collect_events(
        app,
        RunRequest(
            session_id="sess_web_search_parser_failure",
            agent_name="assistant",
            messages=[Message.text("user", "Search")],
            max_steps=1,
            retry_policy=RetryPolicy(max_attempts=1),
        ),
    )

    hosted = [event for event in events if event.type == EventType.MODEL_HOSTED_TOOL_CALL]
    assert [event.payload["status"] for event in hosted] == [
        "in_progress",
        "outcome_unknown",
    ]
    stored_hosted = [
        event
        for event in await store.load_events("sess_web_search_parser_failure")
        if event.type == EventType.MODEL_HOSTED_TOOL_CALL
    ]
    assert [(event.payload["call_id"], event.payload["status"]) for event in stored_hosted] == [
        ("ws_parser_failure", "in_progress"),
        ("ws_parser_failure", "outcome_unknown"),
    ]
    usage = await app.get_session_usage("sess_web_search_parser_failure")
    assert usage.usage.hosted_tools.web_search_calls == 0
    assert usage.usage.hosted_tools.web_search_outcome_unknown == 1
    assert events[-1].type == EventType.SESSION_FAILED


def test_build_openai_payload_translates_cayu_messages() -> None:
    request = ModelRequest(
        model="gpt-test",
        messages=[
            Message.text("system", "You are a careful assistant."),
            Message.text("user", "Read a file."),
            Message(
                role=MessageRole.ASSISTANT,
                content=[
                    TextPart(text="I will inspect it."),
                    ToolCallPart(
                        tool_call_id="call_1",
                        tool_name="read_file",
                        arguments={"path": "README.md"},
                    ),
                ],
            ),
            Message.tool_result(
                tool_call_id="call_1",
                tool_name="read_file",
                content="README content",
                structured={"ignored_by_provider": True},
            ),
        ],
        tools=[
            {
                "name": "read_file",
                "description": "Read a file.",
                "input_schema": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                },
            }
        ],
        options={"openai": {"temperature": 0.2}},
    )

    payload = build_openai_payload(request)

    assert payload == {
        "model": "gpt-test",
        "instructions": "You are a careful assistant.",
        "input": [
            {
                "role": "user",
                "content": [{"type": "input_text", "text": "Read a file."}],
            },
            {
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {"type": "output_text", "text": "I will inspect it."},
                ],
            },
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "read_file",
                "arguments": '{"path":"README.md"}',
                "status": "completed",
            },
            {
                "type": "function_call_output",
                "call_id": "call_1",
                "output": "README content",
            },
        ],
        "tools": [
            {
                "type": "function",
                "name": "read_file",
                "description": "Read a file.",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                },
                "strict": False,
            }
        ],
        "store": False,
        # Requested so reasoning items carry encrypted_content and survive replay
        # across stateless (store=false) calls instead of 404-ing on their rs_ id.
        "include": ["reasoning.encrypted_content"],
        "temperature": 0.2,
    }


def test_build_openai_payload_passes_provider_cache_options() -> None:
    request = ModelRequest(
        model="gpt-test",
        messages=[Message.text("user", "hello")],
        options={
            "openai": {
                "prompt_cache_key": "tenant-a-agent",
                "prompt_cache_retention": "24h",
            }
        },
    )

    payload = build_openai_payload(request)

    assert payload["prompt_cache_key"] == "tenant-a-agent"
    assert payload["prompt_cache_retention"] == "24h"


def test_build_openai_payload_maps_native_structured_output_to_text_format() -> None:
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
        "additionalProperties": False,
    }
    request = ModelRequest(
        model="gpt-test",
        messages=[Message.text("user", "answer")],
        options={
            "structured_output": {
                "name": "answer_schema",
                "schema": schema,
                "strategy": "native",
                "max_retries": 1,
                "repair_prompt": None,
            }
        },
    )

    payload = build_openai_payload(request)

    assert payload["text"] == {
        "format": {
            "type": "json_schema",
            "name": "answer_schema",
            "schema": schema,
            "strict": True,
        }
    }
    assert "tools" not in payload


def test_build_openai_payload_ignores_tool_strategy_structured_output_option() -> None:
    request = ModelRequest(
        model="gpt-test",
        messages=[Message.text("user", "answer")],
        options={
            "structured_output": {
                "name": "answer_schema",
                "schema": {"type": "object"},
                "strategy": "tool",
                "max_retries": 1,
                "repair_prompt": None,
            }
        },
    )

    payload = build_openai_payload(request)

    assert "text" not in payload


def test_build_openai_payload_rejects_openai_text_override_with_native_structured_output() -> None:
    request = ModelRequest(
        model="gpt-test",
        messages=[Message.text("user", "answer")],
        options={
            "openai": {"text": {"format": {"type": "text"}}},
            "structured_output": {
                "name": "answer_schema",
                "schema": {"type": "object"},
                "strategy": "native",
                "max_retries": 1,
                "repair_prompt": None,
            },
        },
    )

    with pytest.raises(ValueError, match="cannot be combined with native structured output"):
        build_openai_payload(request)


def test_openai_native_schema_preflight_accepts_strict_compliant_schema() -> None:
    preflight_openai_native_structured_output_schema(
        {
            "type": "object",
            "properties": {
                "answer": {"type": "string"},
                "address": {
                    "type": ["object", "null"],
                    "properties": {"street": {"type": "string"}},
                    "required": ["street"],
                    "additionalProperties": False,
                },
                "tags": {"type": "array", "items": {"type": "string"}},
                "value": {
                    "anyOf": [
                        {"type": "string"},
                        {
                            "type": "object",
                            "properties": {"code": {"type": "integer"}},
                            "required": ["code"],
                            "additionalProperties": False,
                        },
                    ]
                },
            },
            "required": ["answer", "address", "tags", "value"],
            "additionalProperties": False,
        }
    )


def test_openai_native_schema_preflight_accepts_recursive_refs() -> None:
    preflight_openai_native_structured_output_schema(
        {
            "type": "object",
            "properties": {
                "whole": {"$ref": "#"},
                "node": {"$ref": "#/$defs/node"},
            },
            "required": ["whole", "node"],
            "additionalProperties": False,
            "$defs": {
                "node": {
                    "type": "object",
                    "properties": {
                        "children": {"type": "array", "items": {"$ref": "#/$defs/node"}},
                    },
                    "required": ["children"],
                    "additionalProperties": False,
                }
            },
        }
    )


def test_openai_native_schema_preflight_resolves_root_ref() -> None:
    preflight_openai_native_structured_output_schema(
        {
            "$ref": "#/$defs/answer",
            "$defs": {
                "answer": {
                    "type": "object",
                    "properties": {"answer": {"type": "string"}},
                    "required": ["answer"],
                    "additionalProperties": False,
                }
            },
        }
    )


def test_openai_native_schema_preflight_requires_root_object() -> None:
    with pytest.raises(
        NativeStructuredOutputSchemaInvalid,
        match=r"\$: OpenAI native structured output requires the root schema",
    ):
        preflight_openai_native_structured_output_schema(
            {"type": "array", "items": {"type": "string"}}
        )


def test_openai_native_schema_preflight_requires_additional_properties_false() -> None:
    with pytest.raises(
        NativeStructuredOutputSchemaInvalid,
        match=r"\$/properties/address: .*additionalProperties: false",
    ):
        preflight_openai_native_structured_output_schema(
            {
                "type": "object",
                "properties": {
                    "address": {
                        "type": "object",
                        "properties": {"street": {"type": "string"}},
                        "required": ["street"],
                    }
                },
                "required": ["address"],
                "additionalProperties": False,
            }
        )


def test_openai_native_schema_preflight_requires_all_properties_required() -> None:
    with pytest.raises(
        NativeStructuredOutputSchemaInvalid,
        match=r"\$: .*listed in required; missing: note\.",
    ):
        preflight_openai_native_structured_output_schema(
            {
                "type": "object",
                "properties": {"answer": {"type": "string"}, "note": {"type": "string"}},
                "required": ["answer"],
                "additionalProperties": False,
            }
        )


def test_openai_native_schema_preflight_rejects_external_ref() -> None:
    with pytest.raises(
        NativeStructuredOutputSchemaInvalid,
        match=r"\$/properties/item/\$ref: .*internal \$refs",
    ):
        preflight_openai_native_structured_output_schema(
            {
                "type": "object",
                "properties": {"item": {"$ref": "https://example.com/schema.json"}},
                "required": ["item"],
                "additionalProperties": False,
            }
        )


def test_openai_native_schema_preflight_rejects_unresolvable_ref() -> None:
    with pytest.raises(
        NativeStructuredOutputSchemaInvalid,
        match=r"\$/properties/item/\$ref: .*does not resolve",
    ):
        preflight_openai_native_structured_output_schema(
            {
                "type": "object",
                "properties": {"item": {"$ref": "#/$defs/missing"}},
                "required": ["item"],
                "additionalProperties": False,
                "$defs": {},
            }
        )


def test_openai_native_schema_preflight_rejects_ref_sibling_keys() -> None:
    with pytest.raises(
        NativeStructuredOutputSchemaInvalid,
        match=r"\$/properties/item: .*sibling keys \(found: description\)",
    ):
        preflight_openai_native_structured_output_schema(
            {
                "type": "object",
                "properties": {"item": {"$ref": "#/$defs/base", "description": "sibling"}},
                "required": ["item"],
                "additionalProperties": False,
                "$defs": {
                    "base": {
                        "type": "object",
                        "properties": {"name": {"type": "string"}},
                        "required": ["name"],
                        "additionalProperties": False,
                    }
                },
            }
        )


def test_openai_native_schema_preflight_walks_ref_targets() -> None:
    # A referenced schema must satisfy the strict rules even when it lives
    # under a container keyword the structural walk does not descend into.
    with pytest.raises(
        NativeStructuredOutputSchemaInvalid,
        match=r"\$/x-container/obj: .*additionalProperties: false",
    ):
        preflight_openai_native_structured_output_schema(
            {
                "type": "object",
                "properties": {"item": {"$ref": "#/x-container/obj"}},
                "required": ["item"],
                "additionalProperties": False,
                "x-container": {
                    "obj": {
                        "type": "object",
                        "properties": {"answer": {"type": "string"}},
                        "required": ["answer"],
                    }
                },
            }
        )


def test_openai_native_schema_preflight_accepts_mutually_recursive_ref_targets() -> None:
    preflight_openai_native_structured_output_schema(
        {
            "type": "object",
            "properties": {"tree": {"$ref": "#/$defs/a"}},
            "required": ["tree"],
            "additionalProperties": False,
            "$defs": {
                "a": {
                    "type": "object",
                    "properties": {"b": {"anyOf": [{"type": "null"}, {"$ref": "#/$defs/b"}]}},
                    "required": ["b"],
                    "additionalProperties": False,
                },
                "b": {
                    "type": "object",
                    "properties": {"a": {"anyOf": [{"type": "null"}, {"$ref": "#/$defs/a"}]}},
                    "required": ["a"],
                    "additionalProperties": False,
                },
            },
        }
    )


def test_openai_native_schema_preflight_array_index_refs_follow_rfc_6901() -> None:
    def schema_with_ref(pointer: str) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {"item": {"$ref": pointer}},
            "required": ["item"],
            "additionalProperties": False,
            "anyOf": [{"type": "string"}, {"type": "integer"}],
        }

    preflight_openai_native_structured_output_schema(schema_with_ref("#/anyOf/1"))

    # Leading zeros and non-decimal Unicode "digits" are not RFC 6901 array
    # indices; both must fail with the typed unresolvable-ref error (the
    # latter previously escaped as a raw int() ValueError).
    for bad_index in ("01", "²"):
        with pytest.raises(
            NativeStructuredOutputSchemaInvalid,
            match=r"\$/properties/item/\$ref: .*does not resolve",
        ):
            preflight_openai_native_structured_output_schema(
                schema_with_ref(f"#/anyOf/{bad_index}")
            )


def test_openai_native_schema_preflight_rejects_overdeep_nesting() -> None:
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {"leaf": {"type": "string"}},
        "required": ["leaf"],
        "additionalProperties": False,
    }
    for _ in range(200):
        schema = {
            "type": "object",
            "properties": {"child": schema},
            "required": ["child"],
            "additionalProperties": False,
        }

    with pytest.raises(
        NativeStructuredOutputSchemaInvalid,
        match=r"exceeds the preflight depth limit",
    ):
        preflight_openai_native_structured_output_schema(schema)


def test_openai_native_schema_preflight_walks_anyof_and_items() -> None:
    with pytest.raises(
        NativeStructuredOutputSchemaInvalid,
        match=r"\$/properties/entries/items/anyOf\[1\]: .*missing: code\.",
    ):
        preflight_openai_native_structured_output_schema(
            {
                "type": "object",
                "properties": {
                    "entries": {
                        "type": "array",
                        "items": {
                            "anyOf": [
                                {"type": "string"},
                                {
                                    "type": "object",
                                    "properties": {"code": {"type": "integer"}},
                                    "required": [],
                                    "additionalProperties": False,
                                },
                            ]
                        },
                    }
                },
                "required": ["entries"],
                "additionalProperties": False,
            }
        )


def test_openai_provider_preflights_native_structured_output_schema() -> None:
    provider = OpenAIProvider(api_key="test-key", transport=RecordingTransport())

    with pytest.raises(
        NativeStructuredOutputSchemaInvalid,
        match=r"\$: .*additionalProperties: false",
    ):
        provider.preflight_native_structured_output_schema(
            {
                "type": "object",
                "properties": {"answer": {"type": "string"}},
                "required": ["answer"],
            }
        )


def test_build_openai_payload_translates_file_attachments() -> None:
    attachment = file_attachment(
        artifact_id="art_pdf",
        kind=FileAttachmentKind.DOCUMENT,
        filename="invoice.pdf",
        content_type="application/pdf",
        size_bytes=5,
    )
    request = ModelRequest(
        model="gpt-test",
        messages=[
            Message.text("user", "Read the invoice."),
            Message.tool_call(
                tool_call_id="call_1",
                tool_name="read_file",
                arguments={"artifact_id": "art_pdf"},
            ),
            Message.tool_result(
                tool_call_id="call_1",
                tool_name="read_file",
                content="Attached PDF artifact art_pdf: invoice.pdf.",
                artifacts=[attachment],
            ),
        ],
        options={
            RESOLVED_FILE_ATTACHMENTS_OPTION: {
                "art_pdf": {
                    "artifact_id": "art_pdf",
                    "kind": "document",
                    "filename": "invoice.pdf",
                    "content_type": "application/pdf",
                    "data_base64": "aGVsbG8=",
                    "metadata": {},
                }
            }
        },
    )

    payload = build_openai_payload(request)

    assert payload["input"][2] == {
        "type": "function_call_output",
        "call_id": "call_1",
        "output": "Attached PDF artifact art_pdf: invoice.pdf.",
    }
    assert payload["input"][3] == {
        "role": "user",
        "content": [
            {
                "type": "input_text",
                "text": "The previous tool result returned file content for inspection.",
            },
            {
                "type": "input_file",
                "filename": "invoice.pdf",
                "file_data": "data:application/pdf;base64,aGVsbG8=",
            },
        ],
    }


def test_build_openai_payload_translates_user_file_parts() -> None:
    image = file_attachment(
        artifact_id="art_image",
        kind=FileAttachmentKind.IMAGE,
        filename="invoice.png",
        content_type="image/png",
        size_bytes=5,
    )
    document = file_attachment(
        artifact_id="art_pdf",
        kind=FileAttachmentKind.DOCUMENT,
        filename="contract.pdf",
        content_type="application/pdf",
        size_bytes=9,
    )
    request = ModelRequest(
        model="gpt-test",
        messages=[
            Message(
                role="user",
                content=[
                    TextPart(text="Read the invoice and the contract."),
                    FilePart(attachment=image),
                    FilePart(attachment=document),
                ],
            ),
        ],
        options={
            RESOLVED_FILE_ATTACHMENTS_OPTION: {
                "art_image": {
                    "artifact_id": "art_image",
                    "kind": "image",
                    "filename": "invoice.png",
                    "content_type": "image/png",
                    "data_base64": "aGVsbG8=",
                    "metadata": {},
                },
                "art_pdf": {
                    "artifact_id": "art_pdf",
                    "kind": "document",
                    "filename": "contract.pdf",
                    "content_type": "application/pdf",
                    "data_base64": "JVBERi0xLjQ=",
                    "metadata": {},
                },
            }
        },
    )

    payload = build_openai_payload(request)

    assert payload["input"][0] == {
        "role": "user",
        "content": [
            {"type": "input_text", "text": "Read the invoice and the contract."},
            {
                "type": "input_image",
                "image_url": "data:image/png;base64,aGVsbG8=",
            },
            {
                "type": "input_file",
                "filename": "contract.pdf",
                "file_data": "data:application/pdf;base64,JVBERi0xLjQ=",
            },
        ],
    }


def test_build_openai_payload_requires_resolved_user_file_parts() -> None:
    image = file_attachment(
        artifact_id="art_missing",
        kind=FileAttachmentKind.IMAGE,
        filename="invoice.png",
        content_type="image/png",
        size_bytes=5,
    )
    request = ModelRequest(
        model="gpt-test",
        messages=[
            Message(
                role="user",
                content=[TextPart(text="Read it."), FilePart(attachment=image)],
            ),
        ],
    )

    with pytest.raises(OpenAIProtocolError, match="Missing resolved file attachment"):
        build_openai_payload(request)


@pytest.mark.anyio
async def test_openai_provider_emits_text_and_completed_events() -> None:
    transport = RecordingTransport(
        stream_events=[
            [
                {"type": "response.created", "response": {"id": "resp_1"}},
                {"type": "response.output_text.delta", "delta": "hello"},
                {
                    "type": "response.completed",
                    "response": {
                        "id": "resp_1",
                        "model": "gpt-test",
                        "status": "completed",
                        "output": [
                            {
                                "type": "message",
                                "id": "msg_1",
                                "status": "completed",
                                "role": "assistant",
                                "content": [
                                    {
                                        "type": "output_text",
                                        "text": "hello",
                                        "annotations": [],
                                    }
                                ],
                            }
                        ],
                        "usage": {"input_tokens": 10, "output_tokens": 2},
                    },
                },
            ]
        ]
    )
    provider = OpenAIProvider(api_key="test-key", transport=transport)
    request = ModelRequest(
        model="gpt-test",
        messages=[Message.text("user", "Say hello.")],
    )

    events = [event async for event in provider.stream(request)]

    assert events[-1].completion is not None
    assert events[-1].completion.end_turn is None
    assert [event.type for event in events] == [
        ModelStreamEventType.TEXT_DELTA,
        ModelStreamEventType.COMPLETED,
    ]
    assert events[0].delta == "hello"
    assert events[1].payload["status"] == "completed"
    assert events[1].completion.finish_reason == ModelFinishReason.STOP
    assert transport.calls[0]["url"] == "https://api.openai.com/v1/responses"
    assert transport.calls[0]["headers"]["authorization"] == "Bearer test-key"
    assert transport.calls[0]["payload"]["store"] is False
    assert transport.calls[0]["payload"]["stream"] is True


@pytest.mark.anyio
@pytest.mark.parametrize("end_turn", [True, False])
async def test_openai_stream_events_preserves_end_turn(end_turn: bool) -> None:
    async def raw_events():
        yield {
            "type": "response.completed",
            "response": {
                "id": "resp_turn_control",
                "model": "gpt-test",
                "status": "completed",
                "end_turn": end_turn,
                "output": [],
            },
        }

    events = [event async for event in openai_stream_events(raw_events())]

    assert events[-1].completion is not None
    assert events[-1].completion.end_turn is end_turn


@pytest.mark.parametrize("timeout_s", [float("nan"), float("inf"), float("-inf")])
def test_openai_provider_rejects_nonfinite_timeout(timeout_s: float) -> None:
    with pytest.raises(ValueError, match="timeout_s"):
        OpenAIProvider(api_key="test-key", timeout_s=timeout_s)


@pytest.mark.anyio
async def test_openai_stream_events_stops_at_first_completed_event() -> None:
    yielded = 0

    async def raw_events():
        nonlocal yielded
        yielded += 1
        yield {
            "type": "response.completed",
            "response": {
                "id": "response-local",
                "model": "model-local",
                "status": "completed",
                "output": [],
                "usage": {},
            },
        }
        yielded += 1
        yield {"type": "response.output_text.delta", "delta": "late"}

    events = [event async for event in openai_stream_events(raw_events())]

    assert [event.type for event in events] == [ModelStreamEventType.COMPLETED]
    assert yielded == 1


@pytest.mark.anyio
@pytest.mark.parametrize("end_turn", ["false", 0, 1, [], {}])
async def test_openai_stream_events_rejects_malformed_end_turn(end_turn: object) -> None:
    async def raw_events():
        yield {
            "type": "response.completed",
            "response": {
                "id": "resp_turn_control",
                "model": "gpt-test",
                "status": "completed",
                "end_turn": end_turn,
                "output": [],
            },
        }

    with pytest.raises(OpenAIProtocolError, match="end_turn must be a boolean"):
        [event async for event in openai_stream_events(raw_events())]


@pytest.mark.anyio
async def test_openai_provider_counts_input_tokens_with_official_endpoint() -> None:
    transport = RecordingTransport(
        count_responses=[{"object": "response.input_tokens", "input_tokens": 42}]
    )
    provider = OpenAIProvider(api_key="test-key", transport=transport)
    request = ModelRequest(
        model="gpt-test",
        messages=[Message.text("user", "Count this.")],
    )

    result = await provider.count_input_tokens(request)

    assert result is not None
    assert result.input_tokens == 42
    assert result.method == InputTokenCountMethod.OFFICIAL
    assert result.confidence == InputTokenCountConfidence.HIGH
    assert result.metadata == {
        "endpoint": "responses/input_tokens",
        "provider_billing_status": "not_documented",
    }
    assert transport.count_calls[0]["url"] == "https://api.openai.com/v1/responses/input_tokens"
    assert transport.count_calls[0]["headers"]["authorization"] == "Bearer test-key"
    assert transport.count_calls[0]["payload"]["model"] == "gpt-test"
    assert transport.count_calls[0]["payload"]["input"] == [
        {"role": "user", "content": [{"type": "input_text", "text": "Count this."}]}
    ]
    assert "store" not in transport.count_calls[0]["payload"]
    assert "stream" not in transport.count_calls[0]["payload"]
    assert "include" not in transport.count_calls[0]["payload"]


@pytest.mark.anyio
@pytest.mark.parametrize("operation", ["count", "embed"])
async def test_openai_non_stream_failures_drop_provider_credential_and_exception_graph(
    operation: str,
) -> None:
    canary = "provider-openai-key-canary-0123456789"

    class CredentialFailingTransport(RecordingTransport):
        async def create_response(self, **_kwargs: Any) -> Mapping[str, Any]:
            error = RuntimeError(f"Authorization: Bearer {canary}")
            error.headers = {"Authorization": f"Bearer {canary}"}
            error.add_note(f"transport retained {canary}")
            raise error

    provider = OpenAIProvider(api_key=canary, transport=CredentialFailingTransport())

    with pytest.raises(OpenAIAPIError) as exc_info:
        if operation == "count":
            await provider.count_input_tokens(
                ModelRequest(model="gpt-test", messages=[Message.text("user", "Count this.")])
            )
        else:
            await provider.embed_texts(
                TextEmbeddingRequest(model="text-embedding-test", texts=["embed this"])
            )

    assert_cayu_traceback_does_not_retain(exc_info.value, provider)
    retained = str(exc_info.value) + repr(exc_info.value) + repr(vars(exc_info.value))
    assert canary not in retained
    assert str(exc_info.value) == "RuntimeError: OpenAI provider failed"
    assert exc_info.value.response_body is None
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None


def test_build_openai_token_count_payload_keeps_only_count_supported_fields() -> None:
    request = ModelRequest(
        model="gpt-test",
        messages=[Message.text("user", "Count this.")],
        tools=[EchoTool.spec.model_dump()],
        options={
            "openai": {
                "parallel_tool_calls": False,
                "reasoning": {"effort": "low"},
                "temperature": 0.2,
                "tool_choice": "auto",
                "truncation": "auto",
            }
        },
    )

    payload = openai_module.build_openai_token_count_payload(request)

    assert payload == {
        "model": "gpt-test",
        "input": [{"role": "user", "content": [{"type": "input_text", "text": "Count this."}]}],
        "tools": [
            {
                "type": "function",
                "name": "echo",
                "description": "Echo text.",
                "parameters": {
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                },
                "strict": False,
            }
        ],
        "reasoning": {"effort": "low"},
        "truncation": "auto",
        "tool_choice": "auto",
        "parallel_tool_calls": False,
    }


def test_build_openai_embedding_payload() -> None:
    request = TextEmbeddingRequest(
        model="text-embedding-test",
        texts=["first", "second"],
        dimensions=256,
        options={"user": "user_123"},
    )

    payload = build_openai_embedding_payload(request)

    assert payload == {
        "model": "text-embedding-test",
        "input": ["first", "second"],
        "encoding_format": "float",
        "dimensions": 256,
        "user": "user_123",
    }


def test_build_openai_embedding_payload_rejects_reserved_overrides() -> None:
    request = TextEmbeddingRequest(
        model="text-embedding-test",
        texts=["first"],
        options={"input": "override"},
    )

    with pytest.raises(ValueError, match="reserved keys"):
        build_openai_embedding_payload(request)


@pytest.mark.anyio
async def test_openai_provider_embeds_texts() -> None:
    transport = RecordingTransport(
        count_responses=[
            {
                "object": "list",
                "model": "text-embedding-test",
                "data": [
                    {"object": "embedding", "index": 0, "embedding": [1.0, 0.0]},
                    {"object": "embedding", "index": 1, "embedding": [0.0, 1.0]},
                ],
                "usage": {"prompt_tokens": 7, "total_tokens": 7},
            }
        ]
    )
    provider = OpenAIProvider(api_key="test-key", transport=transport)

    result = await provider.embed_texts(
        TextEmbeddingRequest(model="text-embedding-test", texts=["first", "second"])
    )

    assert result.model == "text-embedding-test"
    assert [embedding.vector for embedding in result.embeddings] == [[1.0, 0.0], [0.0, 1.0]]
    assert result.usage is not None
    assert result.usage.input_tokens == 7
    assert result.usage.total_tokens == 7
    assert result.metadata == {"provider": "openai", "endpoint": "embeddings"}
    assert transport.count_calls[0]["url"] == "https://api.openai.com/v1/embeddings"
    assert transport.count_calls[0]["payload"] == {
        "model": "text-embedding-test",
        "input": ["first", "second"],
        "encoding_format": "float",
    }


def test_openai_embedding_result_rejects_count_mismatch() -> None:
    with pytest.raises(OpenAIProtocolError, match="count"):
        openai_embedding_result(
            {
                "object": "list",
                "model": "text-embedding-test",
                "data": [],
            },
            requested_count=1,
        )


def test_openai_embedding_result_rejects_wrong_indexes() -> None:
    with pytest.raises(OpenAIProtocolError, match="indexes"):
        openai_embedding_result(
            {
                "object": "list",
                "model": "text-embedding-test",
                "data": [{"object": "embedding", "index": 1, "embedding": [1.0]}],
            },
            requested_count=1,
        )


def test_openai_embedding_result_rejects_invalid_usage() -> None:
    with pytest.raises(OpenAIProtocolError, match="prompt_tokens"):
        openai_embedding_result(
            {
                "object": "list",
                "model": "text-embedding-test",
                "data": [{"object": "embedding", "index": 0, "embedding": [1.0]}],
                "usage": {"prompt_tokens": True, "total_tokens": 1},
            },
            requested_count=1,
        )


@pytest.mark.anyio
async def test_openai_provider_rejects_invalid_token_count_response() -> None:
    transport = RecordingTransport(
        count_responses=[{"object": "response.input_tokens", "input_tokens": "42"}]
    )
    provider = OpenAIProvider(api_key="test-key", transport=transport)

    with pytest.raises(OpenAIProtocolError, match="input_tokens"):
        await provider.count_input_tokens(
            ModelRequest(model="gpt-test", messages=[Message.text("user", "Count this.")])
        )


@pytest.mark.anyio
async def test_openai_provider_emits_tool_call_events() -> None:
    transport = RecordingTransport(
        stream_events=[
            [
                {
                    "type": "response.output_item.added",
                    "output_index": 1,
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
                    "item_id": "fc_1",
                    "output_index": 1,
                    "delta": '{"text":',
                },
                {
                    "type": "response.function_call_arguments.delta",
                    "item_id": "fc_1",
                    "output_index": 1,
                    "delta": '"hello"}',
                },
                {
                    "type": "response.function_call_arguments.done",
                    "item_id": "fc_1",
                    "output_index": 1,
                    "name": "echo",
                    "arguments": '{"text":"hello"}',
                },
                {
                    "type": "response.completed",
                    "response": {
                        "id": "resp_1",
                        "model": "gpt-test",
                        "status": "completed",
                        "output": [
                            {
                                "type": "reasoning",
                                "id": "rs_1",
                                "summary": [],
                                "phase": "tool_use",
                            },
                            {
                                "type": "function_call",
                                "id": "fc_1",
                                "call_id": "call_1",
                                "name": "echo",
                                "arguments": '{"text":"hello"}',
                                "status": "completed",
                            },
                        ],
                    },
                },
            ]
        ]
    )
    provider = OpenAIProvider(api_key="test-key", transport=transport)
    request = ModelRequest(
        model="gpt-test",
        messages=[Message.text("user", "Use a tool.")],
    )

    events = [event async for event in provider.stream(request)]

    assert [event.type for event in events] == [
        ModelStreamEventType.TOOL_CALL,
        ModelStreamEventType.COMPLETED,
    ]
    assert events[0].payload == {
        "id": "call_1",
        "name": "echo",
        "arguments": {"text": "hello"},
    }
    assert events[1].payload["status"] == "completed"
    assert events[1].completion is not None
    assert events[1].completion.finish_reason == ModelFinishReason.TOOL_CALLS


@pytest.mark.anyio
async def test_openai_provider_round_trips_runtime_tool_results() -> None:
    transport = RecordingTransport(
        stream_events=[
            [
                {
                    "type": "response.output_item.added",
                    "output_index": 1,
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
                    "item_id": "fc_1",
                    "output_index": 1,
                    "delta": '{"text":"hello from openai"}',
                },
                {
                    "type": "response.function_call_arguments.done",
                    "item_id": "fc_1",
                    "output_index": 1,
                    "name": "echo",
                    "arguments": '{"text":"hello from openai"}',
                },
                {
                    "type": "response.completed",
                    "response": {
                        "id": "resp_1",
                        "model": "gpt-test",
                        "status": "completed",
                        "output": [
                            {
                                "type": "reasoning",
                                "id": "rs_1",
                                "summary": [],
                                "phase": "tool_use",
                            },
                            {
                                "type": "function_call",
                                "id": "fc_1",
                                "call_id": "call_1",
                                "name": "echo",
                                "arguments": '{"text":"hello from openai"}',
                                "status": "completed",
                            },
                        ],
                    },
                },
            ],
            [
                {"type": "response.output_text.delta", "delta": "final answer"},
                {
                    "type": "response.completed",
                    "response": {
                        "id": "resp_2",
                        "model": "gpt-test",
                        "status": "completed",
                        "output": [
                            {
                                "type": "message",
                                "id": "msg_2",
                                "status": "completed",
                                "role": "assistant",
                                "content": [
                                    {
                                        "type": "output_text",
                                        "text": "final answer",
                                        "annotations": [],
                                    }
                                ],
                            }
                        ],
                    },
                },
            ],
        ]
    )
    provider = OpenAIProvider(api_key="test-key", transport=transport)
    app = CayuApp()
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(
            name="assistant",
            model="gpt-test",
            system_prompt="Use tools when needed.",
        ),
        tools=[EchoTool()],
    )

    events = [
        event
        async for event in app.run(
            RunRequest(
                agent_name="assistant",
                messages=[Message.text("user", "Echo this.")],
            )
        )
    ]

    assert events[-1].type == "session.completed"
    model_completed_events = [event for event in events if event.type == "model.completed"]
    assert "provider_state" not in model_completed_events[0].payload
    assert len(transport.calls) == 2
    assert transport.calls[0]["payload"]["input"] == [
        {
            "role": "user",
            "content": [{"type": "input_text", "text": "Echo this."}],
        }
    ]
    assert transport.calls[1]["payload"]["input"] == [
        {
            "role": "user",
            "content": [{"type": "input_text", "text": "Echo this."}],
        },
        {
            "type": "reasoning",
            "id": "rs_1",
            "summary": [],
            "phase": "tool_use",
        },
        {
            "type": "function_call",
            "id": "fc_1",
            "call_id": "call_1",
            "name": "echo",
            "arguments": '{"text":"hello from openai"}',
            "status": "completed",
        },
        {
            "type": "function_call_output",
            "call_id": "call_1",
            "output": "hello from openai",
        },
    ]


@pytest.mark.anyio
async def test_openai_provider_replays_streamed_function_call_when_completed_lacks_output() -> None:
    transport = RecordingTransport(
        stream_events=[
            [
                {
                    "type": "response.output_item.added",
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
                    "item_id": "fc_1",
                    "output_index": 0,
                    "delta": '{"text":"hello from openai"}',
                },
                {
                    "type": "response.function_call_arguments.done",
                    "item_id": "fc_1",
                    "output_index": 0,
                    "name": "echo",
                    "arguments": '{"text":"hello from openai"}',
                },
                {
                    "type": "response.completed",
                    "response": {
                        "id": "resp_1",
                        "model": "gpt-test",
                        "status": "completed",
                    },
                },
            ],
            [
                {"type": "response.output_text.delta", "delta": "final answer"},
                {
                    "type": "response.completed",
                    "response": {
                        "id": "resp_2",
                        "model": "gpt-test",
                        "status": "completed",
                        "output": [
                            {
                                "type": "message",
                                "role": "assistant",
                                "content": [{"type": "output_text", "text": "final answer"}],
                            }
                        ],
                    },
                },
            ],
        ]
    )
    provider = OpenAIProvider(api_key="test-key", transport=transport)
    app = CayuApp()
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="gpt-test", system_prompt="Use tools."),
        tools=[EchoTool()],
    )

    events = [
        event
        async for event in app.run(
            RunRequest(
                agent_name="assistant",
                messages=[Message.text("user", "Echo this.")],
            )
        )
    ]

    assert events[-1].type == "session.completed"
    assert transport.calls[1]["payload"]["input"] == [
        {
            "role": "user",
            "content": [{"type": "input_text", "text": "Echo this."}],
        },
        {
            "type": "function_call",
            "call_id": "call_1",
            "name": "echo",
            "arguments": '{"text":"hello from openai"}',
            "status": "completed",
            "id": "fc_1",
        },
        {
            "type": "function_call_output",
            "call_id": "call_1",
            "output": "hello from openai",
        },
    ]


@pytest.mark.anyio
async def test_openai_provider_replays_streamed_text_when_completed_lacks_output() -> None:
    transport = RecordingTransport(
        stream_events=[
            [
                {
                    "type": "response.output_item.added",
                    "output_index": 0,
                    "item": {
                        "type": "message",
                        "id": "msg_1",
                        "status": "in_progress",
                        "role": "assistant",
                        "content": [],
                    },
                },
                {"type": "response.output_text.delta", "delta": "hello"},
                {
                    "type": "response.output_item.done",
                    "output_index": 0,
                    "item": {
                        "type": "message",
                        "id": "msg_1",
                        "status": "completed",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "hello"}],
                    },
                },
                {
                    "type": "response.completed",
                    "response": {
                        "id": "resp_1",
                        "model": "gpt-test",
                        "status": "completed",
                    },
                },
            ],
            [
                {"type": "response.output_text.delta", "delta": "second answer"},
                {
                    "type": "response.completed",
                    "response": {
                        "id": "resp_2",
                        "model": "gpt-test",
                        "status": "completed",
                        "output": [
                            {
                                "type": "message",
                                "role": "assistant",
                                "content": [{"type": "output_text", "text": "second answer"}],
                            }
                        ],
                    },
                },
            ],
        ]
    )
    provider = OpenAIProvider(api_key="test-key", transport=transport)
    app = CayuApp()
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="gpt-test", system_prompt="Be direct."))

    run_events = [
        event
        async for event in app.run(
            RunRequest(
                agent_name="assistant",
                session_id="sess_openai_text_replay",
                messages=[Message.text("user", "Say hello.")],
            )
        )
    ]
    resume_events = [
        event
        async for event in app.resume(
            ResumeRequest(
                session_id="sess_openai_text_replay",
                messages=[Message.text("user", "Again.")],
            )
        )
    ]

    assert run_events[-1].type == "session.completed"
    assert resume_events[-1].type == "session.completed"
    assert transport.calls[1]["payload"]["input"] == [
        {
            "role": "user",
            "content": [{"type": "input_text", "text": "Say hello."}],
        },
        {
            "type": "message",
            "id": "msg_1",
            "status": "completed",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "hello"}],
        },
        {
            "role": "user",
            "content": [{"type": "input_text", "text": "Again."}],
        },
    ]


def test_openai_response_events_rejects_malformed_function_call() -> None:
    with pytest.raises(OpenAIProtocolError, match="arguments were not valid JSON"):
        openai_response_events(
            {
                "output": [
                    {
                        "type": "function_call",
                        "call_id": "call_1",
                        "name": "echo",
                        "arguments": "{",
                    }
                ]
            }
        )


@pytest.mark.parametrize("end_turn", [True, False, None])
def test_openai_response_events_preserves_end_turn(end_turn: bool | None) -> None:
    events = openai_response_events(
        {
            "id": "resp_turn_control",
            "model": "gpt-test",
            "status": "completed",
            "end_turn": end_turn,
            "output": [],
        }
    )

    assert events[-1].completion is not None
    assert events[-1].completion.end_turn is end_turn


def test_openai_response_events_sanitizes_response_error() -> None:
    with pytest.raises(OpenAIAPIError) as exc_info:
        openai_response_events(
            {
                "request_id": "req_buffered",
                "error": {
                    "type": "invalid_request_error",
                    "code": "bad_request",
                    "param": "input",
                    "message": "bad request",
                    "debug": "not persisted",
                },
                "output": [],
            }
        )

    message = str(exc_info.value)
    assert message == "OpenAI response error: [provider response body omitted]"
    assert "debug" not in message
    assert "not persisted" not in message
    assert exc_info.value.error_type == "invalid_request_error"
    assert exc_info.value.error_code == "bad_request"
    assert exc_info.value.param == "input"
    assert exc_info.value.request_id == "req_buffered"
    assert exc_info.value.response_body is None


@pytest.mark.parametrize(
    "secret_start",
    [
        MAX_PROVIDER_ERROR_BODY_CHARS - len("workload-secret-canary-ABCDEFGHIJKLMNOP"),
        MAX_PROVIDER_ERROR_BODY_CHARS - 10,
        MAX_PROVIDER_ERROR_BODY_CHARS,
    ],
    ids=("ends-at-old-bound", "crosses-old-bound", "starts-at-old-bound"),
)
def test_openai_buffered_error_omits_raw_message_at_old_cutoff(secret_start: int) -> None:
    secret = "workload-secret-canary-ABCDEFGHIJKLMNOP"
    raw_response = {
        "request_id": "req_buffered_cutoff",
        "error": {
            "type": "server_error",
            "code": "internal_error",
            "message": "x" * secret_start + secret,
        },
    }

    with pytest.raises(OpenAIAPIError) as exc_info:
        openai_response_events(raw_response)

    rendered = repr((str(exc_info.value), vars(exc_info.value)))
    assert str(exc_info.value) == "OpenAI response error: [provider response body omitted]"
    assert not any(secret[:size] in rendered for size in range(8, len(secret) + 1))
    assert exc_info.value.error_type == "server_error"
    assert exc_info.value.error_code == "internal_error"
    assert exc_info.value.request_id == "req_buffered_cutoff"
    assert_cayu_traceback_does_not_retain(exc_info.value, raw_response)


def test_openai_buffered_error_classifies_overflow_after_old_cutoff() -> None:
    secret = "workload-secret-canary-ABCDEFGHIJKLMNOP"

    with pytest.raises(OpenAIContextOverflowError) as exc_info:
        openai_response_events(
            {
                "request_id": "req_buffered_overflow",
                "error": {
                    "type": "invalid_request_error",
                    "code": "context_length_exceeded",
                    "message": (
                        "x" * (MAX_PROVIDER_ERROR_BODY_CHARS + 1)
                        + " Context length exceeded. "
                        + secret
                    ),
                },
            }
        )

    assert str(exc_info.value) == "OpenAI model context overflow"
    assert secret not in repr((str(exc_info.value), vars(exc_info.value)))
    assert exc_info.value.error_type == "invalid_request_error"
    assert exc_info.value.error_code == "context_length_exceeded"
    assert exc_info.value.request_id == "req_buffered_overflow"
    assert exc_info.value.response_body is None


def test_openai_response_events_emits_refusal_text() -> None:
    events = openai_response_events(
        {
            "id": "resp_1",
            "model": "gpt-test",
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "id": "msg_1",
                    "status": "completed",
                    "role": "assistant",
                    "content": [
                        {
                            "type": "refusal",
                            "refusal": "I cannot help with that.",
                        }
                    ],
                }
            ],
        }
    )

    assert [event.type for event in events] == [
        ModelStreamEventType.TEXT_DELTA,
        ModelStreamEventType.COMPLETED,
    ]
    assert events[0].delta == "I cannot help with that."


@pytest.mark.anyio
async def test_openai_stream_events_emits_incomplete_terminal_response() -> None:
    async def raw_events():
        yield {"type": "response.output_text.delta", "delta": "partial"}
        yield {
            "type": "response.incomplete",
            "response": {
                "id": "resp_1",
                "model": "gpt-test",
                "status": "incomplete",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "partial"}],
                    }
                ],
                "incomplete_details": {"reason": "max_output_tokens"},
                "usage": {"input_tokens": 10, "output_tokens": 5},
            },
        }

    events = [event async for event in openai_module.openai_stream_events(raw_events())]

    assert [event.type for event in events] == [
        ModelStreamEventType.TEXT_DELTA,
        ModelStreamEventType.COMPLETED,
    ]
    assert events[0].delta == "partial"
    assert events[1].payload["status"] == "incomplete"
    assert events[1].payload["incomplete_details"] == {"reason": "max_output_tokens"}
    assert events[1].completion is not None
    assert events[1].completion.finish_reason == ModelFinishReason.LENGTH
    assert events[1].completion.raw_finish_reason == "max_output_tokens"


@pytest.mark.anyio
async def test_openai_stream_events_emits_web_search_lifecycle_and_citation() -> None:
    async def raw_events():
        yield {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {"type": "web_search_call", "id": "ws_1", "status": "in_progress"},
        }
        yield {
            "type": "response.web_search_call.in_progress",
            "output_index": 0,
            "item_id": "ws_1",
        }
        yield {
            "type": "response.web_search_call.searching",
            "output_index": 0,
            "item_id": "ws_1",
        }
        yield {
            "type": "response.web_search_call.completed",
            "output_index": 0,
            "item_id": "ws_1",
        }
        yield {
            "type": "response.output_item.done",
            "output_index": 0,
            "item": {
                "type": "web_search_call",
                "id": "ws_1",
                "status": "completed",
                "action": {
                    "type": "search",
                    "query": "Cayu",
                    "sources": [
                        {
                            "type": "url",
                            "url": "https://github.com/example/cayu",
                            "title": "Cayu",
                        }
                    ],
                },
            },
        }
        yield {
            "type": "response.output_text.delta",
            "output_index": 1,
            "content_index": 0,
            "delta": "Cayu is a runtime.",
        }
        yield {
            "type": "response.output_text.annotation.added",
            "output_index": 1,
            "content_index": 0,
            "annotation": {
                "type": "url_citation",
                "url": "https://github.com/example/cayu",
                "title": "Cayu",
                "start_index": 0,
                "end_index": 4,
            },
        }
        yield {
            "type": "response.completed",
            "response": {
                "id": "resp_1",
                "model": "gpt-test",
                "status": "completed",
                "usage": {"input_tokens": 3, "output_tokens": 4, "total_tokens": 7},
            },
        }

    events = [event async for event in openai_stream_events(raw_events())]

    assert [event.type for event in events] == [
        ModelStreamEventType.HOSTED_TOOL_CALL,
        ModelStreamEventType.HOSTED_TOOL_CALL,
        ModelStreamEventType.HOSTED_TOOL_CALL,
        ModelStreamEventType.TEXT_DELTA,
        ModelStreamEventType.CITATION,
        ModelStreamEventType.COMPLETED,
    ]
    assert [event.payload["status"] for event in events[:3]] == [
        "in_progress",
        "searching",
        "completed",
    ]
    assert events[2].payload["action"]["sources"][0]["title"] == "Cayu"
    assert events[4].payload["end_index"] == 4
    assert events[-1].payload["provider_state"][0]["state"]["type"] == "web_search_call"


@pytest.mark.anyio
async def test_openai_stream_events_projects_citation_offsets_over_final_text() -> None:
    async def raw_events():
        yield {
            "type": "response.output_text.delta",
            "output_index": 0,
            "content_index": 0,
            "delta": "Prefix ",
        }
        yield {
            "type": "response.output_text.delta",
            "output_index": 0,
            "content_index": 1,
            "delta": "Cayu",
        }
        yield {
            "type": "response.output_text.annotation.added",
            "output_index": 0,
            "content_index": 1,
            "annotation": {
                "type": "url_citation",
                "url": "https://github.com/example/cayu",
                "start_index": 0,
                "end_index": 4,
            },
        }
        yield {
            "type": "response.completed",
            "response": {
                "id": "resp_1",
                "model": "gpt-test",
                "status": "completed",
                "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
            },
        }

    events = [event async for event in openai_stream_events(raw_events())]

    citation = next(event for event in events if event.type is ModelStreamEventType.CITATION)
    assert citation.payload["start_index"] == 7
    assert citation.payload["end_index"] == 11


@pytest.mark.anyio
async def test_openai_stream_events_marks_started_search_unknown_on_transport_end() -> None:
    async def raw_events():
        yield {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {"type": "web_search_call", "id": "ws_1", "status": "in_progress"},
        }

    stream = openai_stream_events(raw_events())
    started = await anext(stream)
    unknown = await anext(stream)

    assert started.payload["status"] == "in_progress"
    assert unknown.payload == {
        "tool_type": "web_search",
        "call_id": "ws_1",
        "status": "outcome_unknown",
    }
    with pytest.raises(OpenAIProtocolError, match="ended before response.completed"):
        await anext(stream)


@pytest.mark.anyio
async def test_openai_stream_events_marks_started_search_unknown_before_protocol_error() -> None:
    async def raw_events():
        yield {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {"type": "web_search_call", "id": "ws_1", "status": "in_progress"},
        }
        yield {
            "type": "response.web_search_call.searching",
            "output_index": 0,
            "item_id": "ws_wrong",
        }

    stream = openai_stream_events(raw_events())
    started = await anext(stream)
    unknown = await anext(stream)

    assert started.payload["status"] == "in_progress"
    assert unknown.payload == {
        "tool_type": "web_search",
        "call_id": "ws_1",
        "status": "outcome_unknown",
    }
    with pytest.raises(OpenAIProtocolError, match="item_id mismatch"):
        await anext(stream)


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("second_output_index", "second_call_id", "expected_error"),
    [
        (0, "ws_2", "output_item.added was repeated"),
        (1, "ws_1", "call identity was reused"),
    ],
)
async def test_openai_stream_events_rejects_rebound_web_search_identity(
    second_output_index: int,
    second_call_id: str,
    expected_error: str,
) -> None:
    async def raw_events():
        yield {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {"type": "web_search_call", "id": "ws_1", "status": "in_progress"},
        }
        yield {
            "type": "response.output_item.added",
            "output_index": second_output_index,
            "item": {
                "type": "web_search_call",
                "id": second_call_id,
                "status": "in_progress",
            },
        }

    stream = openai_stream_events(raw_events())
    assert (await anext(stream)).payload["status"] == "in_progress"
    assert (await anext(stream)).payload == {
        "tool_type": "web_search",
        "call_id": "ws_1",
        "status": "outcome_unknown",
    }
    with pytest.raises(OpenAIProtocolError, match=expected_error):
        await anext(stream)


@pytest.mark.anyio
async def test_openai_stream_events_rejects_duplicate_identity_in_terminal_output() -> None:
    async def raw_events():
        yield {
            "type": "response.completed",
            "response": {
                "id": "resp_1",
                "model": "gpt-test",
                "status": "completed",
                "output": [
                    {
                        "type": "web_search_call",
                        "id": "ws_same",
                        "status": "completed",
                        "action": {"type": "search", "query": query},
                    }
                    for query in ("Cayu", "Cayu docs")
                ],
            },
        }

    stream = openai_stream_events(raw_events())
    with pytest.raises(OpenAIProtocolError, match="duplicated across output items"):
        await anext(stream)


@pytest.mark.anyio
async def test_openai_stream_events_reconciles_matching_terminal_search_evidence() -> None:
    search_call = {
        "type": "web_search_call",
        "id": "ws_1",
        "status": "completed",
        "action": {"type": "search", "query": "Cayu"},
    }

    async def raw_events():
        yield {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {"type": "web_search_call", "id": "ws_1", "status": "in_progress"},
        }
        yield {
            "type": "response.output_item.done",
            "output_index": 0,
            "item": search_call,
        }
        yield {
            "type": "response.completed",
            "response": {
                "id": "resp_1",
                "model": "gpt-test",
                "status": "completed",
                "output": [search_call],
            },
        }

    events = [event async for event in openai_stream_events(raw_events())]

    hosted = [event for event in events if event.type is ModelStreamEventType.HOSTED_TOOL_CALL]
    assert [event.payload["status"] for event in hosted] == ["in_progress", "completed"]
    assert events[-1].type is ModelStreamEventType.COMPLETED
    assert events[-1].payload["hosted_tool_usage"] == {
        "web_search_calls": 1,
        "web_search_outcome_unknown": 0,
    }


@pytest.mark.anyio
@pytest.mark.parametrize(
    "terminal_output",
    [
        pytest.param([], id="omitted"),
        pytest.param(
            [
                {
                    "type": "web_search_call",
                    "id": "ws_2",
                    "status": "completed",
                    "action": {"type": "search", "query": "Cayu"},
                }
            ],
            id="rebound-call-id",
        ),
        pytest.param(
            [
                {"type": "message", "role": "assistant", "content": []},
                {
                    "type": "web_search_call",
                    "id": "ws_1",
                    "status": "completed",
                    "action": {"type": "search", "query": "Cayu"},
                },
            ],
            id="rebound-output-index",
        ),
    ],
)
async def test_openai_stream_events_rejects_terminal_output_that_omits_or_rebinds_search(
    terminal_output: list[dict[str, object]],
) -> None:
    lifecycle_call = {
        "type": "web_search_call",
        "id": "ws_1",
        "status": "completed",
        "action": {"type": "search", "query": "Cayu"},
    }

    async def raw_events():
        yield {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {"type": "web_search_call", "id": "ws_1", "status": "in_progress"},
        }
        yield {
            "type": "response.output_item.done",
            "output_index": 0,
            "item": lifecycle_call,
        }
        yield {
            "type": "response.completed",
            "response": {
                "id": "resp_1",
                "model": "gpt-test",
                "status": "completed",
                "output": terminal_output,
            },
        }

    stream = openai_stream_events(raw_events())
    assert (await anext(stream)).payload["status"] == "in_progress"
    assert (await anext(stream)).payload["status"] == "completed"
    with pytest.raises(OpenAIProtocolError, match="omitted|conflicts"):
        await anext(stream)


@pytest.mark.anyio
async def test_openai_stream_events_marks_started_search_unknown_before_cancellation() -> None:
    blocker = asyncio.Event()

    async def raw_events():
        yield {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {"type": "web_search_call", "id": "ws_1", "status": "in_progress"},
        }
        await blocker.wait()

    stream = openai_stream_events(raw_events())
    started = await anext(stream)
    pending = asyncio.create_task(anext(stream))
    await asyncio.sleep(0)
    pending.cancel()

    unknown = await pending
    assert started.payload["status"] == "in_progress"
    assert unknown.payload == {
        "tool_type": "web_search",
        "call_id": "ws_1",
        "status": "outcome_unknown",
    }
    with pytest.raises(asyncio.CancelledError):
        await anext(stream)


@pytest.mark.anyio
async def test_openai_stream_events_marks_unfinished_search_unknown_on_incomplete() -> None:
    async def raw_events():
        yield {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {"type": "web_search_call", "id": "ws_1", "status": "in_progress"},
        }
        yield {
            "type": "response.incomplete",
            "response": {
                "id": "resp_1",
                "model": "gpt-test",
                "status": "incomplete",
                "incomplete_details": {"reason": "max_output_tokens"},
                "usage": {"input_tokens": 1, "output_tokens": 0, "total_tokens": 1},
            },
        }

    events = [event async for event in openai_stream_events(raw_events())]

    assert [event.type for event in events] == [
        ModelStreamEventType.HOSTED_TOOL_CALL,
        ModelStreamEventType.HOSTED_TOOL_CALL,
        ModelStreamEventType.COMPLETED,
    ]
    assert events[1].payload["status"] == "outcome_unknown"
    assert events[-1].payload["provider_state"] == []


@pytest.mark.anyio
async def test_openai_incomplete_response_does_not_apply_end_turn_false() -> None:
    async def raw_events():
        yield {
            "type": "response.incomplete",
            "response": {
                "id": "resp_incomplete_turn_control",
                "model": "gpt-test",
                "status": "incomplete",
                "end_turn": False,
                "incomplete_details": {"reason": "max_output_tokens"},
                "output": [],
            },
        }

    events = [event async for event in openai_stream_events(raw_events())]

    assert events[-1].completion is not None
    assert events[-1].completion.finish_reason == ModelFinishReason.LENGTH
    assert events[-1].completion.end_turn is None


@pytest.mark.anyio
@pytest.mark.parametrize("pending_item_type", ["function_call", "reasoning"])
async def test_openai_stream_events_discards_unfinished_items_on_incomplete_response(
    pending_item_type: str,
) -> None:
    if pending_item_type == "function_call":
        item = {
            "type": "function_call",
            "id": "fc_partial",
            "call_id": "call_partial",
            "name": "echo",
            "arguments": '{"text":',
            "status": "incomplete",
        }
        partial_events = [
            {
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {**item, "arguments": "", "status": "in_progress"},
            },
            {
                "type": "response.function_call_arguments.delta",
                "item_id": "fc_partial",
                "output_index": 0,
                "delta": '{"text":',
            },
        ]
    else:
        item = {
            "type": "reasoning",
            "id": "rs_partial",
            "status": "incomplete",
            "summary": [],
        }
        partial_events = [
            {
                "type": "response.output_item.added",
                "output_index": 0,
                "item": item,
            },
            {
                "type": "response.reasoning_summary_text.delta",
                "delta": "partial thought",
            },
        ]

    async def raw_events():
        for event in partial_events:
            yield event
        yield {
            "type": "response.incomplete",
            "response": {
                "id": "resp_partial",
                "model": "gpt-test",
                "status": "incomplete",
                "output": [item],
                "incomplete_details": {"reason": "max_output_tokens"},
            },
        }

    events = [event async for event in openai_stream_events(raw_events())]

    assert ModelStreamEventType.TOOL_CALL not in [event.type for event in events]
    completed = [event for event in events if event.type == ModelStreamEventType.COMPLETED]
    assert len(completed) == 1
    assert completed[0].completion is not None
    assert completed[0].completion.finish_reason == ModelFinishReason.LENGTH
    assert completed[0].payload["provider_state"] == []


@pytest.mark.anyio
async def test_openai_stream_events_uses_done_function_call_arguments() -> None:
    async def raw_events():
        yield {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {
                "type": "function_call",
                "id": "fc_1",
                "call_id": "call_1",
                "name": "echo",
                "arguments": "",
            },
        }
        yield {
            "type": "response.function_call_arguments.done",
            "item_id": "fc_1",
            "output_index": 0,
            "name": "echo",
            "arguments": '{"text":"from done event"}',
            "sequence_number": 2,
        }
        yield {
            "type": "response.completed",
            "response": {
                "id": "resp_1",
                "model": "gpt-test",
                "status": "completed",
            },
        }

    events = [event async for event in openai_module.openai_stream_events(raw_events())]

    assert [event.type for event in events] == [
        ModelStreamEventType.TOOL_CALL,
        ModelStreamEventType.COMPLETED,
    ]
    assert events[0].payload == {
        "id": "call_1",
        "name": "echo",
        "arguments": {"text": "from done event"},
    }


@pytest.mark.anyio
async def test_openai_stream_completion_uses_fallback_output_items_for_finish_reason() -> None:
    async def raw_events():
        yield {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {
                "type": "function_call",
                "id": "fc_1",
                "call_id": "call_1",
                "name": "echo",
                "arguments": "",
            },
        }
        yield {
            "type": "response.function_call_arguments.done",
            "item_id": "fc_1",
            "output_index": 0,
            "name": "echo",
            "arguments": '{"text":"from done event"}',
            "sequence_number": 2,
        }
        yield {
            "type": "response.completed",
            "response": {
                "id": "resp_1",
                "model": "gpt-test",
                "status": "completed",
            },
        }

    events = [event async for event in openai_module.openai_stream_events(raw_events())]

    assert [event.type for event in events] == [
        ModelStreamEventType.TOOL_CALL,
        ModelStreamEventType.COMPLETED,
    ]
    assert events[1].completion is not None
    assert events[1].completion.finish_reason == ModelFinishReason.TOOL_CALLS
    assert events[1].payload["provider_state"] == [
        {
            "provider": "openai",
            "state": {
                "type": "function_call",
                "id": "fc_1",
                "call_id": "call_1",
                "name": "echo",
                "arguments": '{"text":"from done event"}',
                "status": "completed",
            },
        }
    ]


def test_openai_completion_respects_explicit_empty_output_items() -> None:
    response = {
        "id": "resp_1",
        "model": "gpt-test",
        "status": "completed",
        "output": [
            {
                "type": "function_call",
                "id": "fc_1",
                "call_id": "call_1",
                "name": "echo",
                "arguments": "{}",
            }
        ],
    }

    event = openai_module._completed_event_from_response(
        response,
        completion_output_items=[],
    )

    assert event.completion is not None
    assert event.completion.finish_reason == ModelFinishReason.STOP


@pytest.mark.anyio
async def test_openai_stream_events_emits_refusal_text() -> None:
    async def raw_events():
        yield {
            "type": "response.refusal.delta",
            "item_id": "msg_1",
            "output_index": 0,
            "content_index": 0,
            "delta": "I cannot help with that.",
            "sequence_number": 1,
        }
        yield {
            "type": "response.refusal.done",
            "item_id": "msg_1",
            "output_index": 0,
            "content_index": 0,
            "refusal": "I cannot help with that.",
            "sequence_number": 2,
        }
        yield {
            "type": "response.completed",
            "response": {
                "id": "resp_1",
                "model": "gpt-test",
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "id": "msg_1",
                        "role": "assistant",
                        "content": [
                            {
                                "type": "refusal",
                                "refusal": "I cannot help with that.",
                            }
                        ],
                    }
                ],
            },
        }

    events = [event async for event in openai_module.openai_stream_events(raw_events())]

    assert [event.type for event in events] == [
        ModelStreamEventType.TEXT_DELTA,
        ModelStreamEventType.COMPLETED,
    ]
    assert events[0].delta == "I cannot help with that."


@pytest.mark.anyio
async def test_openai_stream_events_rejects_function_call_done_without_call_id() -> None:
    async def raw_events():
        yield {
            "type": "response.function_call_arguments.done",
            "item_id": "fc_1",
            "output_index": 0,
            "name": "echo",
            "arguments": '{"text":"hello"}',
            "sequence_number": 1,
        }

    with pytest.raises(OpenAIProtocolError, match="arrived before output_item.added"):
        [event async for event in openai_module.openai_stream_events(raw_events())]


@pytest.mark.anyio
async def test_openai_stream_events_extracts_response_failed_error() -> None:
    async def raw_events():
        yield {
            "type": "response.failed",
            "response": {
                "id": "resp_1",
                "request_id": "req_stream",
                "status": "failed",
                "error": {
                    "code": "server_error",
                    "type": "server_error",
                    "param": "input",
                    "message": "The model failed.",
                    "debug": "not persisted",
                },
            },
            "sequence_number": 1,
        }

    with pytest.raises(OpenAIAPIError) as exc_info:
        [event async for event in openai_module.openai_stream_events(raw_events())]

    message = str(exc_info.value)
    assert message == "OpenAI streaming error: [provider response body omitted]"
    assert "debug" not in message
    assert "not persisted" not in message
    assert exc_info.value.error_type == "server_error"
    assert exc_info.value.error_code == "server_error"
    assert exc_info.value.param == "input"
    assert exc_info.value.request_id == "req_stream"
    assert exc_info.value.status_code == 500
    assert exc_info.value.retryable is True
    assert exc_info.value.response_body is None


@pytest.mark.anyio
async def test_openai_stream_events_extracts_top_level_error_event() -> None:
    async def raw_events():
        yield {
            "type": "error",
            "code": "rate_limit_exceeded",
            "message": "Too many requests.",
            "param": None,
            "request_id": "req_top_level",
            "sequence_number": 1,
        }

    with pytest.raises(OpenAIAPIError) as exc_info:
        [event async for event in openai_module.openai_stream_events(raw_events())]

    assert str(exc_info.value) == "OpenAI streaming error: [provider response body omitted]"
    assert exc_info.value.error_type == "error"
    assert exc_info.value.error_code == "rate_limit_exceeded"
    assert exc_info.value.request_id == "req_top_level"
    assert exc_info.value.status_code == 429
    assert exc_info.value.retryable is True


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("error_type", "error_code", "expected_status", "expected_retryable"),
    [
        pytest.param("authentication_error", None, 401, False, id="authentication"),
        pytest.param("invalid_request_error", "bad_request", 400, False, id="request"),
        pytest.param(
            "invalid_request_error",
            "previous_response_not_found",
            404,
            False,
            id="stale-chain",
        ),
        pytest.param("future_error", "future_code", None, None, id="unknown"),
        pytest.param("authentication_error", "server_error", None, False, id="conflict"),
        pytest.param(
            "authentication_error",
            "previous_response_not_found",
            None,
            False,
            id="stale-chain-conflict",
        ),
    ],
)
async def test_openai_stream_error_classifies_only_consistent_known_identity(
    error_type: str,
    error_code: str | None,
    expected_status: int | None,
    expected_retryable: bool | None,
) -> None:
    async def raw_events():
        yield {
            "type": "response.failed",
            "response": {
                "error": {
                    "type": error_type,
                    "code": error_code,
                    "message": "server_error overloaded authentication_error",
                }
            },
        }

    with pytest.raises(OpenAIAPIError) as exc_info:
        [event async for event in openai_stream_events(raw_events())]

    assert exc_info.value.status_code == expected_status
    assert exc_info.value.retryable is expected_retryable


@pytest.mark.anyio
@pytest.mark.parametrize("event_type", ["response.failed", "error"])
@pytest.mark.parametrize(
    "secret_start",
    [
        MAX_PROVIDER_ERROR_BODY_CHARS - len("workload-secret-canary-ABCDEFGHIJKLMNOP"),
        MAX_PROVIDER_ERROR_BODY_CHARS - 10,
        MAX_PROVIDER_ERROR_BODY_CHARS,
    ],
    ids=("ends-at-old-bound", "crosses-old-bound", "starts-at-old-bound"),
)
async def test_openai_stream_error_omits_raw_message_at_old_cutoff(
    event_type: str,
    secret_start: int,
) -> None:
    secret = "workload-secret-canary-ABCDEFGHIJKLMNOP"
    error = {
        "type": "server_error",
        "code": "internal_error",
        "message": "x" * secret_start + secret,
    }
    event = (
        {
            "type": "response.failed",
            "response": {
                "request_id": "req_cutoff",
                "error": error,
            },
        }
        if event_type == "response.failed"
        else {**error, "type": "error", "request_id": "req_cutoff"}
    )

    async def raw_events():
        yield event

    with pytest.raises(OpenAIAPIError) as exc_info:
        [event async for event in openai_stream_events(raw_events())]

    rendered = repr((str(exc_info.value), vars(exc_info.value)))
    assert str(exc_info.value) == "OpenAI streaming error: [provider response body omitted]"
    assert not any(secret[:size] in rendered for size in range(8, len(secret) + 1))
    assert exc_info.value.error_code == "internal_error"
    assert exc_info.value.request_id == "req_cutoff"
    assert_cayu_traceback_does_not_retain(exc_info.value, event)


@pytest.mark.anyio
async def test_openai_provider_stream_omits_cutoff_secret_fragment() -> None:
    secret = "workload-secret-canary-ABCDEFGHIJKLMNOP"
    transport = RecordingTransport(
        stream_events=[
            [
                {
                    "type": "response.failed",
                    "response": {
                        "request_id": "req_provider_cutoff",
                        "error": {
                            "type": "server_error",
                            "code": "internal_error",
                            "message": ("x" * (MAX_PROVIDER_ERROR_BODY_CHARS - 10) + secret),
                        },
                    },
                }
            ]
        ]
    )
    provider = OpenAIProvider(api_key="test-key", transport=transport)
    request = ModelRequest(model="test-model", messages=[Message.text("user", "hello")])

    events = [event async for event in provider.stream(request)]

    rendered = repr([event.model_dump(mode="json") for event in events])
    assert [event.type for event in events] == [ModelStreamEventType.ERROR]
    assert not any(secret[:size] in rendered for size in range(8, len(secret) + 1))
    assert events[0].payload["provider_error_type"] == "server_error"
    assert events[0].payload["provider_error_code"] == "internal_error"
    assert "request_id" not in events[0].payload


@pytest.mark.anyio
async def test_runtime_retries_openai_stream_server_error_then_completes() -> None:
    transport = RecordingTransport(
        stream_events=[
            [
                {
                    "type": "response.failed",
                    "response": {
                        "error": {
                            "type": "server_error",
                            "code": "server_error",
                            "message": "temporary provider failure",
                        }
                    },
                }
            ],
            [
                {"type": "response.output_text.delta", "delta": "recovered"},
                {
                    "type": "response.completed",
                    "response": {
                        "id": "resp_retry",
                        "model": "gpt-test",
                        "status": "completed",
                        "output": [
                            {
                                "type": "message",
                                "id": "msg_retry",
                                "status": "completed",
                                "role": "assistant",
                                "content": [
                                    {
                                        "type": "output_text",
                                        "text": "recovered",
                                        "annotations": [],
                                    }
                                ],
                            }
                        ],
                        "usage": {},
                    },
                },
            ],
        ]
    )
    provider = OpenAIProvider(api_key="test-key", transport=transport)
    app = CayuApp()
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="gpt-test"))

    events = [
        event
        async for event in app.run(
            RunRequest(
                agent_name="assistant",
                messages=[Message.text("user", "hello")],
                max_steps=1,
                retry_policy=RetryPolicy(max_attempts=2, initial_delay_s=0.0),
            )
        )
    ]

    assert len(transport.calls) == 2
    retry = next(event for event in events if event.type == EventType.MODEL_RETRY)
    assert retry.payload["status_code"] == 500
    assert retry.payload["reason"] == "http_status"
    discarded = next(event for event in events if event.type == EventType.MODEL_ATTEMPT_DISCARDED)
    assert discarded.payload["status_code"] == 500
    assert events[-1].type == EventType.SESSION_COMPLETED


@pytest.mark.anyio
async def test_runtime_honors_bounded_retry_after_from_successful_http_stream() -> None:
    requests: list[httpx.Request] = []
    response_events = [
        {
            "type": "response.failed",
            "response": {
                "error": {
                    "type": "server_error",
                    "code": "server_error",
                    "message": "temporary provider failure",
                }
            },
        },
        _completed_sse("resp_retry_after"),
    ]

    class ResponseStream(httpx.AsyncByteStream):
        def __init__(self, body: bytes) -> None:
            self._body = body

        async def __aiter__(self):
            yield self._body

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        event = response_events[len(requests) - 1]
        body = f"data: {json.dumps(event)}\n\n".encode()
        headers = {"retry-after": "120"} if len(requests) == 1 else {}
        return httpx.Response(
            200,
            headers=headers,
            stream=ResponseStream(body),
            request=request,
        )

    http_transport = HttpxOpenAITransport()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        http_transport._client._client = client
        provider = OpenAIProvider(api_key="test-key", transport=http_transport)
        app = CayuApp()
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="gpt-test"))

        events = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    messages=[Message.text("user", "hello")],
                    max_steps=1,
                    retry_policy=RetryPolicy(
                        max_attempts=2,
                        initial_delay_s=0.0,
                        max_delay_s=0.01,
                    ),
                )
            )
        ]

    assert len(requests) == 2
    assert all(request.headers["accept-encoding"] == "identity" for request in requests)
    model_error = next(event for event in events if event.type == EventType.MODEL_ERROR)
    assert model_error.payload["retry_after_s"] == 120.0
    retry = next(event for event in events if event.type == EventType.MODEL_RETRY)
    assert retry.payload["delay_seconds"] == 0.01
    assert events[-1].type == EventType.SESSION_COMPLETED


@pytest.mark.anyio
@pytest.mark.parametrize("status_code", [429, 500])
async def test_runtime_retries_compressed_http_error_without_reading_body(
    status_code: int,
) -> None:
    requests: list[httpx.Request] = []

    class ResponseStream(httpx.AsyncByteStream):
        def __init__(self, body: bytes) -> None:
            self._body = body
            self.yielded_chunks = 0

        async def __aiter__(self):
            self.yielded_chunks += 1
            yield self._body

    compressed_stream = ResponseStream(
        gzip.compress(b'{"error":{"message":"temporary provider failure"}}')
    )

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(
                status_code,
                headers={
                    "content-encoding": "gzip",
                    "content-type": "application/json",
                    "retry-after": "120",
                },
                stream=compressed_stream,
                request=request,
            )
        body = f"data: {json.dumps(_completed_sse('resp_compressed_retry'))}\n\n".encode()
        return httpx.Response(
            200,
            stream=ResponseStream(body),
            request=request,
        )

    http_transport = HttpxOpenAITransport()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        http_transport._client._client = client
        provider = OpenAIProvider(api_key="test-key", transport=http_transport)
        app = CayuApp()
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="gpt-test"))

        events = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    messages=[Message.text("user", "hello")],
                    max_steps=1,
                    retry_policy=RetryPolicy(
                        max_attempts=2,
                        initial_delay_s=0.0,
                        max_delay_s=0.01,
                    ),
                )
            )
        ]

    assert len(requests) == 2
    assert compressed_stream.yielded_chunks == 0
    model_error = next(event for event in events if event.type == EventType.MODEL_ERROR)
    assert model_error.payload["status_code"] == status_code
    assert model_error.payload["retry_after_s"] == 120.0
    retry = next(event for event in events if event.type == EventType.MODEL_RETRY)
    assert retry.payload["status_code"] == status_code
    assert retry.payload["reason"] == "http_status"
    assert retry.payload["delay_seconds"] == 0.01
    assert events[-1].type == EventType.SESSION_COMPLETED


@pytest.mark.anyio
async def test_runtime_does_not_recover_conflicting_openai_context_identity() -> None:
    transport = RecordingTransport(
        stream_events=[
            [
                {
                    "type": "response.failed",
                    "response": {
                        "error": {
                            "type": "authentication_error",
                            "code": "context_length_exceeded",
                            "message": "This model's maximum context length was exceeded.",
                        }
                    },
                }
            ]
        ]
    )
    provider = OpenAIProvider(api_key="test-key", transport=transport)
    app = CayuApp()
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="gpt-test"),
        context_overflow_policy=RecentTurnsContextPolicy(max_user_turns=1),
    )

    events = [
        event
        async for event in app.run(
            RunRequest(
                agent_name="assistant",
                messages=[Message.text("user", "hello")],
                max_steps=1,
                retry_policy=RetryPolicy(max_attempts=2, initial_delay_s=0.0),
            )
        )
    ]

    assert len(transport.calls) == 1
    assert EventType.MODEL_RETRY not in [event.type for event in events]
    assert EventType.CONTEXT_OVERFLOW_RECOVERING not in [event.type for event in events]
    model_error = next(event for event in events if event.type == EventType.MODEL_ERROR)
    assert model_error.payload["retryable"] is False
    assert model_error.payload["provider_error_type"] == "authentication_error"
    assert model_error.payload["provider_error_code"] == "context_length_exceeded"
    assert events[-1].type == EventType.SESSION_FAILED


def test_openai_response_events_normalizes_web_search_sources_and_citations() -> None:
    events = openai_response_events(
        {
            "id": "resp_1",
            "model": "gpt-test",
            "status": "completed",
            "output": [
                {
                    "type": "web_search_call",
                    "id": "ws_1",
                    "status": "completed",
                    "action": {
                        "type": "search",
                        "query": "Cayu runtime",
                        "sources": [
                            {
                                "type": "url",
                                "url": "https://github.com/example/cayu",
                                "title": "Cayu",
                            }
                        ],
                    },
                },
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "Cayu is a runtime.",
                            "annotations": [
                                {
                                    "type": "url_citation",
                                    "url": "https://github.com/example/cayu",
                                    "title": "Cayu",
                                    "start_index": 0,
                                    "end_index": 4,
                                }
                            ],
                        }
                    ],
                },
            ],
            "usage": {"input_tokens": 3, "output_tokens": 4, "total_tokens": 7},
        }
    )

    assert [event.type for event in events] == [
        ModelStreamEventType.HOSTED_TOOL_CALL,
        ModelStreamEventType.TEXT_DELTA,
        ModelStreamEventType.CITATION,
        ModelStreamEventType.COMPLETED,
    ]
    assert events[0].payload == {
        "tool_type": "web_search",
        "call_id": "ws_1",
        "status": "completed",
        "source_count": 1,
        "action": {
            "type": "search",
            "query": "Cayu runtime",
            "sources": [
                {
                    "type": "url",
                    "url": "https://github.com/example/cayu",
                    "title": "Cayu",
                }
            ],
        },
    }
    assert events[2].payload == {
        "citation_type": "url_citation",
        "url": "https://github.com/example/cayu",
        "title": "Cayu",
        "start_index": 0,
        "end_index": 4,
    }
    assert events[-1].payload["provider_state"][0] == {
        "provider": "openai",
        "state": {
            "type": "web_search_call",
            "id": "ws_1",
            "status": "completed",
            "action": {
                "type": "search",
                "query": "Cayu runtime",
                "sources": [
                    {
                        "type": "url",
                        "url": "https://github.com/example/cayu",
                        "title": "Cayu",
                    }
                ],
            },
        },
    }
    assert events[-1].payload["hosted_tool_usage"] == {
        "web_search_calls": 1,
        "web_search_outcome_unknown": 0,
    }


def test_openai_response_events_preserves_multi_query_and_optional_citation_metadata() -> None:
    events = openai_response_events(
        {
            "id": "resp_1",
            "model": "gpt-test",
            "status": "completed",
            "output": [
                {
                    "type": "web_search_call",
                    "id": "ws_1",
                    "status": "completed",
                    "action": {
                        "type": "search",
                        "queries": ["Cayu runtime", "Cayu documentation"],
                        "sources": [{"type": "url", "url": "https://github.com/example/cayu"}],
                    },
                },
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "Cayu",
                            "annotations": [
                                {
                                    "type": "url_citation",
                                    "url": "https://github.com/example/cayu",
                                }
                            ],
                        }
                    ],
                },
            ],
        }
    )

    assert events[0].payload["action"] == {
        "type": "search",
        "queries": ["Cayu runtime", "Cayu documentation"],
        "sources": [{"type": "url", "url": "https://github.com/example/cayu"}],
    }
    assert events[2].payload == {
        "citation_type": "url_citation",
        "url": "https://github.com/example/cayu",
    }


def test_openai_response_events_projects_citation_offsets_over_final_text() -> None:
    events = openai_response_events(
        {
            "id": "resp_1",
            "model": "gpt-test",
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [
                        {"type": "output_text", "text": "Prefix ", "annotations": []},
                        {
                            "type": "output_text",
                            "text": "Cayu",
                            "annotations": [
                                {
                                    "type": "url_citation",
                                    "url": "https://github.com/example/cayu",
                                    "start_index": 0,
                                    "end_index": 4,
                                }
                            ],
                        },
                    ],
                }
            ],
        }
    )

    citation = next(event for event in events if event.type is ModelStreamEventType.CITATION)
    assert citation.payload["start_index"] == 7
    assert citation.payload["end_index"] == 11


@pytest.mark.parametrize(
    "url",
    [
        "javascript:alert(1)",
        "//example.com/no-scheme",
        "https:///missing-host",
        "https://@",
        "https://:443",
        "https://user@example.com/private",
        "https://bad host.example/source",
        "https://_service.example/source",
    ],
)
def test_openai_response_events_rejects_unsafe_web_search_urls(url: str) -> None:
    with pytest.raises(OpenAIProtocolError, match="must use http or https"):
        openai_response_events(
            {
                "id": "resp_1",
                "model": "gpt-test",
                "status": "completed",
                "output": [
                    {
                        "type": "web_search_call",
                        "id": "ws_1",
                        "status": "completed",
                        "action": {
                            "type": "search",
                            "query": "Cayu",
                            "sources": [{"type": "url", "url": url}],
                        },
                    }
                ],
            }
        )


def test_openai_response_events_counts_but_does_not_replay_incomplete_search() -> None:
    events = openai_response_events(
        {
            "id": "resp_1",
            "model": "gpt-test",
            "status": "incomplete",
            "output": [
                {
                    "type": "web_search_call",
                    "id": "ws_1",
                    "status": "incomplete",
                }
            ],
            "usage": {"input_tokens": 1, "output_tokens": 0, "total_tokens": 1},
        }
    )

    assert events[0].payload["status"] == "incomplete"
    assert events[-1].payload["hosted_tool_usage"] == {
        "web_search_calls": 1,
        "web_search_outcome_unknown": 0,
    }
    assert events[-1].payload["provider_state"] == []


@pytest.mark.parametrize("status", ["in_progress", "searching"])
def test_openai_response_events_rejects_nonterminal_search_in_completed_response(
    status: str,
) -> None:
    with pytest.raises(
        OpenAIProtocolError,
        match="terminal response contains a nonterminal web search call",
    ):
        openai_response_events(
            {
                "id": "resp_1",
                "model": "gpt-test",
                "status": "completed",
                "output": [
                    {
                        "type": "web_search_call",
                        "id": "ws_1",
                        "status": status,
                    }
                ],
            }
        )


def test_openai_response_events_marks_nonterminal_search_unknown_in_incomplete_response() -> None:
    events = openai_response_events(
        {
            "id": "resp_1",
            "model": "gpt-test",
            "status": "incomplete",
            "output": [
                {
                    "type": "web_search_call",
                    "id": "ws_1",
                    "status": "in_progress",
                }
            ],
            "usage": {"input_tokens": 1, "output_tokens": 0, "total_tokens": 1},
        }
    )

    assert events[0].payload == {
        "tool_type": "web_search",
        "call_id": "ws_1",
        "status": "outcome_unknown",
    }
    assert events[-1].payload["hosted_tool_usage"] == {
        "web_search_calls": 0,
        "web_search_outcome_unknown": 1,
    }
    assert events[-1].payload["provider_state"] == []


def test_openai_response_events_rejects_duplicate_web_search_call_identity() -> None:
    with pytest.raises(OpenAIProtocolError, match="duplicated across output items"):
        openai_response_events(
            {
                "id": "resp_1",
                "model": "gpt-test",
                "status": "completed",
                "output": [
                    {
                        "type": "web_search_call",
                        "id": "ws_same",
                        "status": "completed",
                        "action": {"type": "search", "query": query},
                    }
                    for query in ("Cayu", "Cayu docs")
                ],
            }
        )


def test_openai_response_events_counts_multiple_search_calls_independently() -> None:
    events = openai_response_events(
        {
            "id": "resp_1",
            "model": "gpt-test",
            "status": "completed",
            "output": [
                {
                    "type": "web_search_call",
                    "id": f"ws_{index}",
                    "status": "completed",
                    "action": {"type": "search", "query": query},
                }
                for index, query in enumerate(("Cayu", "Cayu docs"), start=1)
            ],
            "usage": {"input_tokens": 3, "output_tokens": 1, "total_tokens": 4},
        }
    )

    assert [event.payload["call_id"] for event in events[:-1]] == ["ws_1", "ws_2"]
    assert events[-1].payload["hosted_tool_usage"] == {
        "web_search_calls": 2,
        "web_search_outcome_unknown": 0,
    }
    assert len(events[-1].payload["provider_state"]) == 2


def test_openai_response_events_ignores_reasoning_items() -> None:
    events = openai_response_events(
        {
            "id": "resp_1",
            "model": "gpt-test",
            "status": "completed",
            "output": [
                {"type": "reasoning", "id": "rs_1", "summary": []},
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "done"}],
                },
            ],
        }
    )

    assert [event.type for event in events] == [
        ModelStreamEventType.TEXT_DELTA,
        ModelStreamEventType.COMPLETED,
    ]
    assert events[0].delta == "done"


@pytest.mark.parametrize("reserved_option", ["include", "input", "store", "previous_response_id"])
def test_openai_options_must_not_override_reserved_payload_fields(
    reserved_option: str,
) -> None:
    request = ModelRequest(
        model="gpt-test",
        messages=[Message.text("user", "hello")],
        options={"openai": {reserved_option: "bad"}},
    )

    with pytest.raises(ValueError, match="reserved"):
        build_openai_payload(request)


def test_openai_payload_replays_provider_state_items() -> None:
    request = ModelRequest(
        model="gpt-test",
        messages=[
            Message.text("user", "hello"),
            Message(
                role=MessageRole.ASSISTANT,
                content=[
                    ProviderStatePart(
                        provider="openai",
                        state={
                            "type": "reasoning",
                            "id": "rs_1",
                            "summary": [],
                            "phase": "tool_use",
                        },
                    ),
                    ProviderStatePart(
                        provider="openai",
                        state={
                            "type": "function_call",
                            "id": "fc_1",
                            "call_id": "call_1",
                            "name": "echo",
                            "arguments": '{"text":"hello"}',
                            "status": "completed",
                        },
                    ),
                    ToolCallPart(
                        tool_call_id="call_1",
                        tool_name="echo",
                        arguments={"text": "hello"},
                    ),
                ],
            ),
            Message.tool_result(
                tool_call_id="call_1",
                tool_name="echo",
                content="hello",
            ),
        ],
    )

    payload = build_openai_payload(request)

    assert payload["input"] == [
        {
            "role": "user",
            "content": [{"type": "input_text", "text": "hello"}],
        },
        {
            "type": "reasoning",
            "id": "rs_1",
            "summary": [],
            "phase": "tool_use",
        },
        {
            "type": "function_call",
            "id": "fc_1",
            "call_id": "call_1",
            "name": "echo",
            "arguments": '{"text":"hello"}',
            "status": "completed",
        },
        {
            "type": "function_call_output",
            "call_id": "call_1",
            "output": "hello",
        },
    ]


def test_openai_payload_ignores_other_provider_state_items() -> None:
    request = ModelRequest(
        model="gpt-test",
        messages=[
            Message.text("user", "hello"),
            Message(
                role=MessageRole.ASSISTANT,
                content=[
                    ProviderStatePart(
                        provider="other-provider",
                        state={"type": "opaque", "id": "state_1"},
                    ),
                    TextPart(text="assistant text"),
                ],
            ),
        ],
    )

    payload = build_openai_payload(request)

    assert payload["input"] == [
        {
            "role": "user",
            "content": [{"type": "input_text", "text": "hello"}],
        },
        {
            "type": "message",
            "status": "completed",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "assistant text"}],
        },
    ]


def test_openai_provider_rejects_invalid_tool_names() -> None:
    request = ModelRequest(
        model="gpt-test",
        messages=[Message.text("user", "hello")],
        tools=[
            {
                "name": "invalid tool name",
                "description": "Bad name.",
                "input_schema": {"type": "object"},
            }
        ],
    )

    with pytest.raises(ValueError, match="tool names"):
        build_openai_payload(request)


def test_openai_provider_rejects_protected_extra_headers() -> None:
    with pytest.raises(ValueError, match="extra_headers cannot override"):
        OpenAIProvider(
            api_key="test-key",
            extra_headers={"authorization": "other-key"},
        )


def test_openai_provider_requires_https_base_url() -> None:
    with pytest.raises(ValueError, match="https"):
        OpenAIProvider(
            api_key="test-key",
            base_url="http://api.openai.com",
        )


def test_openai_provider_declares_usage_dialect_when_renamed_for_a_gateway() -> None:
    provider = OpenAIProvider(
        api_key="test-key",
        name="company-gateway",
        base_url="https://gateway.example.com/openai",
    )

    assert provider.usage_dialect is UsageDialect.OPENAI


@pytest.mark.anyio
async def test_httpx_openai_transport_requires_https_url() -> None:
    with pytest.raises(ValueError, match="https"):
        await HttpxOpenAITransport().create_response(
            url="http://api.openai.com/v1/responses",
            headers={},
            payload={},
            timeout_s=1,
        )


@pytest.mark.anyio
async def test_httpx_openai_transport_includes_url_in_network_errors(monkeypatch) -> None:
    class FailingClient:
        def __init__(self, **kwargs: Any) -> None:
            pass

        async def __aenter__(self) -> FailingClient:
            return self

        async def __aexit__(self, *args: Any) -> None:
            return None

        async def post(
            self,
            url: str,
            *,
            headers: dict[str, str],
            json: dict[str, Any],
            timeout: Any = None,
        ) -> httpx.Response:
            request = httpx.Request("POST", url)
            raise httpx.ConnectError(
                "[Errno 8] nodename nor servname provided, or not known",
                request=request,
            )

    monkeypatch.setattr(
        "cayu.providers._http.httpx.AsyncClient",
        FailingClient,
    )

    with pytest.raises(OpenAIAPIError, match="https://api.openai.com/v1/responses"):
        await HttpxOpenAITransport().create_response(
            url="https://api.openai.com/v1/responses",
            headers={},
            payload={},
            timeout_s=1,
        )


@pytest.mark.anyio
async def test_httpx_openai_transport_sanitizes_error_body(monkeypatch) -> None:
    class FailingClient:
        def __init__(self, **kwargs: Any) -> None:
            pass

        async def __aenter__(self) -> FailingClient:
            return self

        async def __aexit__(self, *args: Any) -> None:
            return None

        async def post(
            self,
            url: str,
            *,
            headers: dict[str, str],
            json: dict[str, Any],
            timeout: Any = None,
        ) -> httpx.Response:
            request = httpx.Request("POST", url)
            response = httpx.Response(
                400,
                request=request,
                headers={"content-type": "application/json"},
                json={
                    "error": {
                        "type": "invalid_request_error",
                        "code": "bad_request",
                        "message": "bad request",
                        "debug": "not persisted",
                    },
                    "request_id": "req_123",
                    "extra": "not persisted",
                },
            )
            raise httpx.HTTPStatusError(
                "bad request",
                request=request,
                response=response,
            )

    monkeypatch.setattr(
        "cayu.providers._http.httpx.AsyncClient",
        FailingClient,
    )

    with pytest.raises(OpenAIAPIError) as exc_info:
        await HttpxOpenAITransport().create_response(
            url="https://api.openai.com/v1/responses",
            headers={},
            payload={},
            timeout_s=1,
        )

    message = str(exc_info.value)
    assert message == ("OpenAI API request failed with HTTP 400: [provider response body omitted]")
    assert exc_info.value.status_code == 400
    assert exc_info.value.error_type == "invalid_request_error"
    assert exc_info.value.error_code == "bad_request"
    assert exc_info.value.request_id == "req_123"
    assert "debug" not in message
    assert "not persisted" not in message


@pytest.mark.anyio
async def test_httpx_openai_transport_classifies_context_overflow(monkeypatch) -> None:
    class FailingClient:
        def __init__(self, **kwargs: Any) -> None:
            pass

        async def __aenter__(self) -> FailingClient:
            return self

        async def __aexit__(self, *args: Any) -> None:
            return None

        async def post(
            self,
            url: str,
            *,
            headers: dict[str, str],
            json: dict[str, Any],
            timeout: Any = None,
        ) -> httpx.Response:
            request = httpx.Request("POST", url)
            response = httpx.Response(
                400,
                request=request,
                headers={"content-type": "application/json"},
                json={
                    "error": {
                        "type": "invalid_request_error",
                        "code": "context_length_exceeded",
                        "message": "This model's maximum context length was exceeded.",
                    },
                    "request_id": "req_context",
                },
            )
            raise httpx.HTTPStatusError(
                "bad request",
                request=request,
                response=response,
            )

    monkeypatch.setattr(
        "cayu.providers._http.httpx.AsyncClient",
        FailingClient,
    )

    with pytest.raises(OpenAIContextOverflowError) as exc_info:
        await HttpxOpenAITransport().create_response(
            url="https://api.openai.com/v1/responses",
            headers={},
            payload={},
            timeout_s=1,
        )

    assert exc_info.value.provider == "openai"
    assert isinstance(exc_info.value, ModelContextOverflowError)
    assert isinstance(exc_info.value, OpenAIAPIError)
    assert exc_info.value.status_code == 400
    assert exc_info.value.error_type == "invalid_request_error"
    assert exc_info.value.error_code == "context_length_exceeded"
    assert exc_info.value.request_id == "req_context"
    assert isinstance(exc_info.value.__cause__, httpx.HTTPStatusError)


@pytest.mark.anyio
async def test_httpx_openai_transport_rejects_conflicting_context_identity(monkeypatch) -> None:
    class FailingClient:
        def __init__(self, **kwargs: Any) -> None:
            del kwargs

        async def post(
            self,
            url: str,
            *,
            headers: dict[str, str],
            json: dict[str, Any],
            timeout: Any = None,
        ) -> httpx.Response:
            del headers, json, timeout
            request = httpx.Request("POST", url)
            response = httpx.Response(
                401,
                request=request,
                json={
                    "error": {
                        "type": "authentication_error",
                        "code": "context_length_exceeded",
                        "message": "This model's maximum context length was exceeded.",
                    }
                },
            )
            raise httpx.HTTPStatusError("unauthorized", request=request, response=response)

    monkeypatch.setattr("cayu.providers._http.httpx.AsyncClient", FailingClient)

    with pytest.raises(OpenAIAPIError) as exc_info:
        await HttpxOpenAITransport().create_response(
            url="https://api.openai.com/v1/responses",
            headers={},
            payload={},
            timeout_s=1,
        )

    assert not isinstance(exc_info.value, OpenAIContextOverflowError)
    assert exc_info.value.status_code == 401
    assert exc_info.value.error_type == "authentication_error"
    assert exc_info.value.error_code == "context_length_exceeded"


@pytest.mark.anyio
async def test_openai_stream_events_classifies_context_overflow() -> None:
    secret = "workload-secret-canary-ABCDEFGHIJKLMNOP"

    async def raw_events():
        yield {
            "type": "response.failed",
            "response": {
                "id": "resp_1",
                "status": "failed",
                "error": {
                    "type": "invalid_request_error",
                    "code": "context_length_exceeded",
                    "message": (
                        "x" * (MAX_PROVIDER_ERROR_BODY_CHARS + 1)
                        + " Context length exceeded. "
                        + secret
                    ),
                },
            },
            "sequence_number": 1,
        }

    with pytest.raises(OpenAIContextOverflowError) as exc_info:
        [event async for event in openai_module.openai_stream_events(raw_events())]

    assert exc_info.value.provider == "openai"
    assert isinstance(exc_info.value, ModelContextOverflowError)
    assert isinstance(exc_info.value, OpenAIAPIError)
    assert exc_info.value.error_code == "context_length_exceeded"
    assert str(exc_info.value) == "OpenAI model context overflow"
    assert secret not in repr((str(exc_info.value), vars(exc_info.value)))
    assert exc_info.value.response_body is None


@pytest.mark.anyio
async def test_openai_provider_emits_nonblank_error_for_blank_exception() -> None:
    provider = OpenAIProvider(api_key="test-key", transport=BlankFailingTransport())
    request = ModelRequest(
        model="gpt-test",
        messages=[Message.text("user", "hello")],
    )

    events = [event async for event in provider.stream(request)]

    assert [event.type for event in events] == [ModelStreamEventType.ERROR]
    assert events[0].payload == {
        "error": "RuntimeError: OpenAI provider failed",
        "error_type": "RuntimeError",
    }


@pytest.mark.anyio
async def test_openai_provider_stream_propagates_context_overflow() -> None:
    overflow = OpenAIContextOverflowError(
        "OpenAI model context overflow",
        status_code=400,
        error_code="context_length_exceeded",
    )

    class OverflowTransport:
        async def create_response(
            self,
            *,
            url: str,
            headers: Mapping[str, str],
            payload: Mapping[str, Any],
            timeout_s: float,
        ) -> Mapping[str, Any]:
            raise AssertionError("create_response should not be called.")

        async def stream_response_events(
            self,
            *,
            url: str,
            headers: Mapping[str, str],
            payload: Mapping[str, Any],
            timeout_s: float,
            stream_idle_timeout_s: float,
        ):
            raise overflow
            yield {}

    provider = OpenAIProvider(api_key="test-key", transport=OverflowTransport())
    request = ModelRequest(
        model="gpt-test",
        messages=[Message.text("user", "hello")],
    )

    with pytest.raises(OpenAIContextOverflowError) as exc_info:
        [event async for event in provider.stream(request)]

    # Overflow escapes as a fresh detached typed exception so runtime recovery
    # can retry without retaining transport traceback locals containing headers.
    assert exc_info.value is not overflow
    assert isinstance(exc_info.value, ModelContextOverflowError)
    assert exc_info.value.status_code == 400
    assert exc_info.value.error_code == "context_length_exceeded"
    assert exc_info.value.retryable is False


@pytest.mark.anyio
async def test_openai_sse_parser_lets_keepalive_heartbeats_refresh_idle_timer() -> None:
    # OpenAI streams parse through the shared SSE parser, so `:` keep-alive
    # heartbeats count as stream activity (same semantics as Chat Completions;
    # previously the two parsers had drifted apart on this).
    async def lines():
        for _ in range(5):
            await asyncio.sleep(0.04)
            yield ": keepalive"
        yield 'data: {"ok": true}'
        yield ""

    events = [
        event
        async for event in aiter_sse_json_events(
            lines(),
            idle_timeout_s=0.1,
            provider_label="OpenAI",
            protocol_error=OpenAIProtocolError,
        )
    ]
    assert events == [{"ok": True}]


@pytest.mark.anyio
async def test_openai_sse_parser_times_out_a_silent_stream() -> None:
    async def lines():
        await asyncio.sleep(0.05)
        yield ""

    with pytest.raises(TimeoutError, match="OpenAI streaming response produced no SSE events"):
        [
            event
            async for event in aiter_sse_json_events(
                lines(),
                idle_timeout_s=0.001,
                provider_label="OpenAI",
                protocol_error=OpenAIProtocolError,
            )
        ]


def _simple_request():
    return ModelRequest(
        model="gpt-test",
        messages=[Message.text("user", "hi")],
    )


def test_inline_payload_sets_store_false_and_include():
    payload = build_openai_payload(_simple_request(), reasoning_state="inline")
    assert payload["store"] is False
    assert payload["include"] == ["reasoning.encrypted_content"]
    assert "previous_response_id" not in payload


def test_reasoning_state_defaults_to_inline():
    p = OpenAIProvider(api_key="k")
    assert p.reasoning_state == "inline"


def test_reasoning_state_accepts_server():
    p = OpenAIProvider(api_key="k", reasoning_state="server")
    assert p.reasoning_state == "server"


def test_reasoning_state_rejects_unknown():
    with pytest.raises(ValueError):
        OpenAIProvider(api_key="k", reasoning_state="bogus")


def test_build_openai_payload_rejects_unknown_reasoning_state():
    with pytest.raises(ValueError, match="reasoning_state"):
        build_openai_payload(_simple_request(), reasoning_state="bogus")


def test_build_openai_payload_rejects_non_boolean_chain():
    bad_chain: Any = "yes"
    with pytest.raises(TypeError, match="chain"):
        build_openai_payload(_simple_request(), chain=bad_chain)


def _assistant_with_reasoning():
    return Message(
        role=MessageRole.ASSISTANT,
        content=[
            ProviderStatePart(
                provider="openai",
                state={"type": "reasoning", "id": "rs_1", "encrypted_content": "blob"},
            ),
            ProviderStatePart(
                provider="openai",
                state={
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "f",
                    "arguments": "{}",
                    "status": "completed",
                },
            ),
        ],
    )


def test_server_first_call_payload():
    request = ModelRequest(
        model="gpt-test",
        messages=[Message.text("user", "go"), _assistant_with_reasoning()],
    )
    payload = build_openai_payload(request, reasoning_state="server")
    assert payload["store"] is True
    assert "include" not in payload
    assert "previous_response_id" not in payload
    types = [item.get("type") for item in payload["input"]]
    assert "reasoning" not in types
    assert "function_call" in types


async def _drain(events, **kwargs):
    async def gen():
        for e in events:
            yield e

    out = []
    async for ev in openai_stream_events(gen(), **kwargs):
        out.append(ev)
    return out


def _completed_sse(resp_id):
    return {
        "type": "response.completed",
        "response": {
            "id": resp_id,
            "model": "gpt-test",
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "hello"}],
                }
            ],
            "usage": {"input_tokens": 1, "output_tokens": 1},
        },
    }


@pytest.mark.anyio
async def test_server_mode_emits_response_ref() -> None:
    events = await _drain([_completed_sse("resp_123")], reasoning_state="server")
    completed = events[-1]
    states = [s["state"] for s in completed.payload["provider_state"]]
    assert {"type": "response_ref", "id": "resp_123"} in states


@pytest.mark.anyio
async def test_inline_mode_emits_no_response_ref() -> None:
    events = await _drain([_completed_sse("resp_123")], reasoning_state="inline")
    completed = events[-1]
    types = [s["state"].get("type") for s in completed.payload["provider_state"]]
    assert "response_ref" not in types


def test_server_later_call_sends_delta_and_previous_response_id():
    prior_assistant = Message(
        role=MessageRole.ASSISTANT,
        content=[
            ProviderStatePart(
                provider="openai",
                state={
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "ok"}],
                },
            ),
            ProviderStatePart(provider="openai", state={"type": "response_ref", "id": "resp_prev"}),
        ],
    )
    request = ModelRequest(
        model="gpt-test",
        messages=[
            Message.text("user", "first"),
            prior_assistant,
            Message.text("user", "second"),  # the new turn
        ],
    )
    payload = build_openai_payload(request, reasoning_state="server")
    assert payload["previous_response_id"] == "resp_prev"
    assert len(payload["input"]) == 1
    assert payload["input"][0]["content"][0]["text"] == "second"


class StaleChainRecoveryTransport:
    """404s on the first call (stale chain), succeeds on the second."""

    def __init__(self):
        self.payloads = []
        self._calls = 0

    async def stream_response_events(
        self, *, url, headers, payload, timeout_s, stream_idle_timeout_s
    ):
        self.payloads.append(payload)
        self._calls += 1
        if self._calls == 1:
            raise OpenAIAPIError(
                "OpenAI API request failed with HTTP 404: "
                '{"message":"Previous response with id resp_prev not found.","type":"invalid_request_error"}',
                status_code=404,
                error_type="invalid_request_error",
                error_code="previous_response_not_found",
                param="previous_response_id",
            )
            yield {}  # unreachable; makes this an async generator
        yield {
            "type": "response.completed",
            "response": {
                "id": "resp_new",
                "model": "gpt-test",
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "ok"}],
                    }
                ],
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
        }


class NonStalePreviousResponseErrorTransport:
    def __init__(self):
        self.payloads = []

    async def stream_response_events(
        self, *, url, headers, payload, timeout_s, stream_idle_timeout_s
    ):
        self.payloads.append(payload)
        raise OpenAIAPIError(
            "OpenAI API request failed with HTTP 400: "
            '{"message":"previous_response_id cannot be used with conversation.",'
            '"type":"invalid_request_error"}',
            status_code=400,
            error_type="invalid_request_error",
            param="previous_response_id",
        )
        yield {}  # unreachable; makes this an async generator


@pytest.mark.anyio
async def test_server_mode_recovers_from_stale_previous_response_id() -> None:
    transport = StaleChainRecoveryTransport()
    provider = OpenAIProvider(api_key="k", reasoning_state="server", transport=transport)
    request = ModelRequest(
        model="gpt-test",
        messages=[
            Message.text("user", "first"),
            Message(
                role=MessageRole.ASSISTANT,
                content=[
                    ProviderStatePart(
                        provider="openai", state={"type": "response_ref", "id": "resp_prev"}
                    )
                ],
            ),
            Message.text("user", "second"),
        ],
    )
    events = [e async for e in provider.stream(request)]
    assert any(e.type.name == "COMPLETED" for e in events)
    assert transport.payloads[0].get("previous_response_id") == "resp_prev"
    assert "previous_response_id" not in transport.payloads[1]
    assert transport.payloads[1]["store"] is True
    assert len(transport.payloads) == 2


@pytest.mark.anyio
async def test_server_mode_stale_chain_rebuilds_completed_hosted_search_in_order() -> None:
    from cayu import (
        CitationPart,
        CitationProvenance,
        HostedToolCallPart,
        WebSearchAction,
        WebSearchSource,
    )

    model_step_id = "mstep_00000000000000000000000000000000"
    model_attempt_id = "matt_00000000000000000000000000000000"
    prior_assistant = Message(
        role=MessageRole.ASSISTANT,
        content=[
            HostedToolCallPart(
                call_id="ws_unknown",
                status="outcome_unknown",
                provider_name="openai",
                model="gpt-test",
                model_step_id=model_step_id,
                model_attempt_id=model_attempt_id,
            ),
            HostedToolCallPart(
                call_id="ws_completed",
                status="completed",
                action=WebSearchAction(
                    type="search",
                    query="Cayu",
                    sources=(
                        WebSearchSource(
                            url="https://github.com/example/cayu",
                            title="Cayu",
                        ),
                    ),
                ),
                provider_name="openai",
                model="gpt-test",
                model_step_id=model_step_id,
                model_attempt_id=model_attempt_id,
            ),
            TextPart(text="Cayu source"),
            CitationPart(
                url="https://github.com/example/cayu",
                title="Cayu",
                start_index=0,
                end_index=4,
                provenance=CitationProvenance(provider_name="openai"),
                model_step_id=model_step_id,
                model_attempt_id=model_attempt_id,
            ),
            ToolCallPart(
                tool_call_id="call_1",
                tool_name="read_file",
                arguments={"path": "README.md"},
            ),
            ProviderStatePart(provider="openai", state={"type": "response_ref", "id": "resp_prev"}),
        ],
    )
    transport = StaleChainRecoveryTransport()
    provider = OpenAIProvider(api_key="k", reasoning_state="server", transport=transport)
    request = ModelRequest(
        model="gpt-test",
        messages=[
            Message.text("user", "first"),
            prior_assistant,
            Message.text("user", "second"),
        ],
    )

    events = [event async for event in provider.stream(request)]

    assert any(event.type is ModelStreamEventType.COMPLETED for event in events)
    replay_input = transport.payloads[1]["input"]
    assert [item.get("type") for item in replay_input] == [
        None,
        "web_search_call",
        "message",
        "function_call",
        None,
    ]
    assert replay_input[1] == {
        "type": "web_search_call",
        "id": "ws_completed",
        "status": "completed",
        "action": {
            "type": "search",
            "query": "Cayu",
            "sources": [
                {
                    "type": "url",
                    "url": "https://github.com/example/cayu",
                    "title": "Cayu",
                }
            ],
        },
    }
    assert replay_input[2]["content"] == [
        {
            "type": "output_text",
            "text": "Cayu source",
            "annotations": [
                {
                    "type": "url_citation",
                    "url": "https://github.com/example/cayu",
                    "title": "Cayu",
                    "start_index": 0,
                    "end_index": 4,
                }
            ],
        }
    ]
    assert replay_input[3]["call_id"] == "call_1"


@pytest.mark.anyio
async def test_server_mode_does_not_recover_non_stale_previous_response_error() -> None:
    transport = NonStalePreviousResponseErrorTransport()
    provider = OpenAIProvider(api_key="k", reasoning_state="server", transport=transport)
    request = ModelRequest(
        model="gpt-test",
        messages=[
            Message(
                role=MessageRole.ASSISTANT,
                content=[
                    ProviderStatePart(
                        provider="openai", state={"type": "response_ref", "id": "resp_prev"}
                    )
                ],
            ),
            Message.text("user", "second"),
        ],
    )

    events = [e async for e in provider.stream(request)]

    assert len(transport.payloads) == 1
    assert any(e.type.name == "ERROR" for e in events)


@pytest.mark.anyio
async def test_server_mode_does_not_recover_conflicting_stale_chain_identity() -> None:
    transport = RecordingTransport(
        stream_events=[
            [
                {
                    "type": "response.failed",
                    "response": {
                        "error": {
                            "type": "authentication_error",
                            "code": "previous_response_not_found",
                            "param": "previous_response_id",
                            "message": "Conflicting error identity.",
                        }
                    },
                }
            ]
        ]
    )
    provider = OpenAIProvider(api_key="k", reasoning_state="server", transport=transport)
    request = ModelRequest(
        model="gpt-test",
        messages=[
            Message(
                role=MessageRole.ASSISTANT,
                content=[
                    ProviderStatePart(
                        provider="openai", state={"type": "response_ref", "id": "resp_prev"}
                    )
                ],
            ),
            Message.text("user", "second"),
        ],
    )

    events = [event async for event in provider.stream(request)]

    assert len(transport.calls) == 1
    assert [event.type for event in events] == [ModelStreamEventType.ERROR]
    assert events[0].payload["retryable"] is False
    assert events[0].payload["provider_error_type"] == "authentication_error"
    assert events[0].payload["provider_error_code"] == "previous_response_not_found"


@pytest.mark.anyio
async def test_server_mode_recovers_from_in_band_stale_chain_identity() -> None:
    transport = RecordingTransport(
        stream_events=[
            [
                {
                    "type": "response.failed",
                    "response": {
                        "error": {
                            "type": "invalid_request_error",
                            "code": "previous_response_not_found",
                            "param": "previous_response_id",
                            "message": "Previous response not found.",
                        }
                    },
                }
            ],
            [_completed_sse("resp_new")],
        ]
    )
    provider = OpenAIProvider(api_key="k", reasoning_state="server", transport=transport)
    request = ModelRequest(
        model="gpt-test",
        messages=[
            Message(
                role=MessageRole.ASSISTANT,
                content=[
                    ProviderStatePart(
                        provider="openai", state={"type": "response_ref", "id": "resp_prev"}
                    )
                ],
            ),
            Message.text("user", "second"),
        ],
    )

    events = [event async for event in provider.stream(request)]

    assert len(transport.calls) == 2
    assert transport.calls[0]["payload"]["previous_response_id"] == "resp_prev"
    assert "previous_response_id" not in transport.calls[1]["payload"]
    assert events[-1].type == ModelStreamEventType.COMPLETED


def test_is_stale_chain_error_uses_typed_identity_not_message_text() -> None:
    assert openai_module._is_stale_chain_error(
        OpenAIAPIError(
            "chain gone",
            status_code=404,
            error_type="invalid_request_error",
            error_code="previous_response_not_found",
        )
    )
    assert openai_module._is_stale_chain_error(
        OpenAIAPIError("chain gone", status_code=404, param="previous_response_id")
    )
    # A stale code without the provider's 404 classification is insufficient.
    assert not openai_module._is_stale_chain_error(
        OpenAIAPIError("chain gone", error_code="previous_response_not_found")
    )
    # Recognized identities that contradict a stale 404 fail closed.
    assert not openai_module._is_stale_chain_error(
        OpenAIAPIError(
            "chain gone",
            status_code=404,
            error_type="authentication_error",
            error_code="previous_response_not_found",
            param="previous_response_id",
        )
    )
    assert not openai_module._is_stale_chain_error(
        OpenAIAPIError(
            "chain gone",
            status_code=404,
            error_code="server_error",
            param="previous_response_id",
        )
    )
    assert not openai_module._is_stale_chain_error(
        OpenAIAPIError(
            "chain gone",
            status_code=404,
            error_code="previous_response_not_found",
            param="api_key",
        )
    )
    # Message text alone no longer classifies: only structured identity does.
    assert not openai_module._is_stale_chain_error(
        OpenAIAPIError("Previous response with id resp_prev not found.")
    )
    # A different error about the same param (e.g. invalid combination) is not
    # a stale chain even though its message names previous_response_id.
    assert not openai_module._is_stale_chain_error(
        OpenAIAPIError(
            "previous_response_id resp_prev not found in conversation state.",
            status_code=400,
            error_type="invalid_request_error",
            param="previous_response_id",
        )
    )
    assert not openai_module._is_stale_chain_error(RuntimeError("previous response not found"))


def test_openai_api_error_from_response_captures_typed_identity() -> None:
    response = httpx.Response(
        404,
        headers={"content-type": "application/json", "retry-after": "3.5"},
        json={
            "error": {
                "message": "Previous response with id 'resp_prev' not found.",
                "type": "invalid_request_error",
                "param": "previous_response_id",
                "code": "previous_response_not_found",
            },
            "request_id": "req_123",
        },
    )

    error = openai_module._openai_api_error_from_response(
        response,
        "OpenAI API request failed",
        3.5,
    )

    assert isinstance(error, OpenAIAPIError)
    assert str(error) == "OpenAI API request failed"
    assert error.status_code == 404
    assert error.error_type == "invalid_request_error"
    assert error.error_code == "previous_response_not_found"
    assert error.param == "previous_response_id"
    assert error.request_id == "req_123"
    assert error.retryable is False
    assert error.retry_after_s == 3.5
    assert openai_module._is_stale_chain_error(error)


def test_openai_api_error_from_response_tolerates_non_json_body() -> None:
    response = httpx.Response(502, headers={"content-type": "text/html"}, text="bad gateway")

    error = openai_module._openai_api_error_from_response(
        response,
        "OpenAI API request failed",
        None,
    )

    assert error.status_code == 502
    assert error.error_type is None
    assert error.error_code is None
    assert error.param is None
    assert error.retryable is None
    assert not openai_module._is_stale_chain_error(error)


@pytest.mark.parametrize(
    ("status_code", "error_type", "error_code", "expected_retryable"),
    [
        pytest.param(401, "authentication_error", None, False, id="permanent"),
        pytest.param(500, "server_error", "server_error", True, id="transient"),
        pytest.param(500, "future_error", "future_code", None, id="unknown"),
        pytest.param(500, "authentication_error", "server_error", False, id="body-conflict"),
        pytest.param(429, "server_error", "server_error", False, id="status-conflict"),
    ],
)
def test_openai_http_error_uses_stream_identity_classification(
    status_code: int,
    error_type: str,
    error_code: str | None,
    expected_retryable: bool | None,
) -> None:
    response = httpx.Response(
        status_code,
        headers={"content-type": "application/json", "retry-after": "4"},
        json={"error": {"type": error_type, "code": error_code}},
    )

    error = openai_module._openai_api_error_from_response(response, "OpenAI failed", 4.0)

    assert error.status_code == status_code
    assert error.error_type == error_type
    assert error.error_code == error_code
    assert error.retryable is expected_retryable
    assert error.retry_after_s == 4.0


@pytest.mark.anyio
async def test_inline_mode_does_not_recover_on_stale_chain_error() -> None:
    transport = StaleChainRecoveryTransport()
    provider = OpenAIProvider(api_key="k", reasoning_state="inline", transport=transport)
    request = ModelRequest(model="gpt-test", messages=[Message.text("user", "hi")])
    events = [e async for e in provider.stream(request)]
    # Inline mode never recovers: exactly one transport call, error surfaces as MODEL_ERROR.
    assert len(transport.payloads) == 1
    assert any(e.type.name == "ERROR" for e in events)


def test_server_recovery_payload_drops_chain_and_provider_state():
    request = ModelRequest(
        model="gpt-test",
        messages=[
            Message.text("user", "first"),
            Message(
                role=MessageRole.ASSISTANT,
                content=[
                    ProviderStatePart(
                        provider="openai",
                        state={"type": "reasoning", "id": "rs_1", "encrypted_content": "b"},
                    ),
                    ProviderStatePart(
                        provider="openai", state={"type": "response_ref", "id": "resp_prev"}
                    ),
                ],
            ),
            Message.text("user", "second"),
        ],
    )
    payload = build_openai_payload(request, reasoning_state="server", chain=False)
    assert "previous_response_id" not in payload
    assert payload["store"] is True
    texts = [c.get("text") for item in payload["input"] for c in item.get("content", [])]
    assert "first" in texts and "second" in texts
    types = [item.get("type") for item in payload["input"]]
    assert "reasoning" not in types and "response_ref" not in types


@pytest.mark.parametrize(
    "failure_kind",
    ["cleanup", "post_completion_provider", "post_completion_overflow"],
)
def test_cayu_app_preserves_completion_and_closes_without_reading_tail(
    failure_kind: str,
) -> None:
    class CompletedThenFailureStream:
        def __init__(self) -> None:
            self._index = 0
            self.closed = False
            self.tail_reads = 0

        def __aiter__(self) -> CompletedThenFailureStream:
            return self

        async def __anext__(self) -> Mapping[str, Any]:
            self._index += 1
            if self._index == 1:
                return {"type": "response.output_text.delta", "delta": "ok"}
            if self._index == 2:
                return {
                    "type": "response.completed",
                    "response": {
                        "id": "response-1",
                        "model": "fake-model",
                        "status": "completed",
                        "output": [],
                        "usage": {"input_tokens": 2, "output_tokens": 1},
                    },
                }
            if self._index == 3 and failure_kind.startswith("post_completion_"):
                self.tail_reads += 1
                error = (
                    {
                        "type": "invalid_request_error",
                        "code": "context_length_exceeded",
                        "message": "context overflow after completion",
                    }
                    if failure_kind == "post_completion_overflow"
                    else {
                        "type": "server_error",
                        "code": "server_error",
                        "message": "retryable failure after completion",
                    }
                )
                return {
                    "type": "response.failed",
                    "response": {"error": error},
                }
            raise StopAsyncIteration

        async def aclose(self) -> None:
            self.closed = True
            if failure_kind == "cleanup":
                raise RuntimeError("provider stream close failed")

    class CompletingTransport:
        def __init__(self) -> None:
            self.calls = 0
            self.source = CompletedThenFailureStream()

        def stream_response_events(self, **_kwargs: Any) -> CompletedThenFailureStream:
            self.calls += 1
            return self.source

    store = InMemorySessionStore()
    transport = CompletingTransport()
    provider = OpenAIProvider(api_key="test-key", transport=transport)
    app = CayuApp(
        session_store=store,
        retry_policy=RetryPolicy(max_attempts=2, initial_delay_s=0.0),
    )
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    session_id = f"sess_completion_cleanup_{failure_kind}"
    events = asyncio.run(
        _collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "hi")],
            ),
        )
    )
    transcript = asyncio.run(store.load_transcript(session_id))
    durable_events = asyncio.run(store.load_events(session_id))

    assert transport.calls == 1
    assert transport.source.closed
    assert transport.source.tail_reads == 0
    assert EventType.MODEL_RETRY not in {event.type for event in events}
    assert EventType.MODEL_ERROR not in {event.type for event in events}
    model_completions = [
        event for event in durable_events if event.type == EventType.MODEL_COMPLETED
    ]
    assert len(model_completions) == 1
    completion = model_completions[0]
    assert completion.payload["usage"] == {
        "input_tokens": 2,
        "output_tokens": 1,
    }
    assert completion.payload["usage_metrics"]["input_tokens"] == 2
    assert completion.payload["usage_metrics"]["output_tokens"] == 1
    assert completion.payload["usage_metrics"]["total_tokens"] == 3
    if failure_kind == "cleanup":
        assert completion.payload["step_classification"]["type"] == "failed"
        assert events[-1].type == EventType.SESSION_FAILED
        assert [message.role for message in transcript] == ["user"]
        return

    assert completion.payload["step_classification"]["type"] == "final"
    assert events[-1].type == EventType.SESSION_COMPLETED
    assert [message.role for message in transcript] == ["user", "assistant"]
    assert transcript[-1] == Message.text("assistant", "ok")

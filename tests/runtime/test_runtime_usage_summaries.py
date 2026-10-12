"""Usage summaries contracts."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from decimal import Decimal
from uuid import UUID

import pytest
from pydantic import ValidationError
from tests.runtime.runtime_provider_support import CountingProvider, FakeProvider
from tests.runtime.runtime_session_support import (
    collect_events,
    collect_resume_events,
    collect_tool_approval_events,
)
from tests.runtime.runtime_tool_support import RequireApprovalPolicy, SideEffectTool, UpperTool

from cayu.agents import AgentSpec
from cayu.applications import CayuApp
from cayu.approvals.tools import ToolApprovalDecision, ToolApprovalRequest
from cayu.budgets.pricing import ModelPrice, PriceBook
from cayu.budgets.run_limits import RunLimits
from cayu.context.counting import ContextCountingConfig, ContextCountingMode
from cayu.events import Event, EventType
from cayu.messages import Message, TextPart
from cayu.providers.base import (
    InputTokenCountConfidence,
    InputTokenCountMethod,
    InputTokenCountResult,
    ModelRequest,
    ModelStreamEvent,
    UsageDialect,
)
from cayu.runtime._event_projection import PRIVATE_EVENT_AUTHORITY
from cayu.sessions.event_queries import EventOrder, EventQuery
from cayu.sessions.memory import InMemorySessionStore
from cayu.sessions.records import SessionIdentity
from cayu.sessions.requests import ResumeRequest, RunRequest
from cayu.vaults.redaction import SecretRedactor


def test_context_counting_is_off_by_default() -> None:
    provider = CountingProvider(
        [
            ModelStreamEvent.text_delta("done"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ],
        count_result=InputTokenCountResult(
            input_tokens=12,
            method=InputTokenCountMethod.OFFICIAL,
            confidence=InputTokenCountConfidence.HIGH,
        ),
    )
    app = CayuApp(enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_context_counting_default_off",
                messages=[Message.text("user", "hello")],
            ),
        )
    )
    assert provider.count_requests == []
    assert EventType.CONTEXT_COUNTED not in {event.type for event in events}
    assert EventType.CONTEXT_COUNT_RECONCILED not in {event.type for event in events}


def test_cayu_app_emits_turn_completed_with_aggregated_usage() -> None:
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_1",
                    name="side_effect",
                    arguments={},
                ),
                ModelStreamEvent.completed(
                    {
                        "finish_reason": "tool_calls",
                        "model": "fake-model",
                        "usage": {
                            "input_tokens": 10,
                            "output_tokens": 4,
                            "total_tokens": 14,
                        },
                    }
                ),
            ],
            [
                ModelStreamEvent.text_delta("done"),
                ModelStreamEvent.completed(
                    {
                        "finish_reason": "stop",
                        "model": "fake-model",
                        "usage": {
                            "input_tokens": 12,
                            "output_tokens": 3,
                            "total_tokens": 15,
                        },
                    }
                ),
            ],
        ]
    )
    app = CayuApp(enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[SideEffectTool()],
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_turn_completed",
                messages=[Message.text("user", "use the tool")],
            ),
        )
    )

    event_types = [event.type for event in events]
    turn_index = event_types.index(EventType.TURN_COMPLETED)
    assert turn_index < event_types.index(EventType.SESSION_COMPLETED)
    turn = events[turn_index]
    assert turn.payload["status"] == "completed"
    assert turn.payload["step_count"] == 2
    assert turn.payload["tool_call_count"] == 1
    assert turn.payload["token_usage"]["input_tokens"] == 22
    assert turn.payload["token_usage"]["output_tokens"] == 7
    assert turn.payload["token_usage"]["total_tokens"] == 29
    assert turn.payload["models"] == ["fake-model"]
    assert turn.payload["provider_names"] == ["fake"]
    assert isinstance(turn.payload["duration_ms"], int)


def test_cayu_app_turn_completed_on_resume_excludes_prior_turn_usage() -> None:
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.text_delta("first"),
                ModelStreamEvent.completed(
                    {
                        "finish_reason": "stop",
                        "model": "fake-model",
                        "usage": {
                            "input_tokens": 10,
                            "output_tokens": 4,
                            "total_tokens": 14,
                        },
                    }
                ),
            ],
            [
                ModelStreamEvent.text_delta("second"),
                ModelStreamEvent.completed(
                    {
                        "finish_reason": "stop",
                        "model": "fake-model",
                        "usage": {
                            "input_tokens": 12,
                            "output_tokens": 3,
                            "total_tokens": 15,
                        },
                    }
                ),
            ],
        ]
    )
    app = CayuApp(enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    async def run() -> tuple[list[Event], list[Event]]:
        first_events = await collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_turn_completed_resume",
                messages=[Message.text("user", "first")],
            ),
        )
        resumed_events = await collect_resume_events(
            app,
            ResumeRequest(
                session_id="sess_turn_completed_resume",
                messages=[Message.text("user", "second")],
            ),
        )
        return first_events, resumed_events

    first_events, resumed_events = asyncio.run(run())

    first_turn = next(event for event in first_events if event.type == EventType.TURN_COMPLETED)
    resumed_turn = next(event for event in resumed_events if event.type == EventType.TURN_COMPLETED)

    assert first_turn.payload["step_count"] == 1
    assert first_turn.payload["tool_call_count"] == 0
    assert first_turn.payload["token_usage"]["input_tokens"] == 10
    assert first_turn.payload["token_usage"]["output_tokens"] == 4
    assert first_turn.payload["token_usage"]["total_tokens"] == 14

    assert resumed_turn.payload["step_count"] == 1
    assert resumed_turn.payload["tool_call_count"] == 0
    assert resumed_turn.payload["token_usage"]["input_tokens"] == 12
    assert resumed_turn.payload["token_usage"]["output_tokens"] == 3
    assert resumed_turn.payload["token_usage"]["total_tokens"] == 15
    assert len(first_turn.payload["interaction_ids"]) == 1
    assert len(resumed_turn.payload["interaction_ids"]) == 1
    assert first_turn.payload["interaction_ids"] != resumed_turn.payload["interaction_ids"]


def test_context_pressure_estimate_reconciles_against_actual_input_usage() -> None:
    user_text = "secret pressure text should not leak into pressure event payload"
    provider = FakeProvider(
        [
            ModelStreamEvent.text_delta("done"),
            ModelStreamEvent.completed(
                {
                    "finish_reason": "stop",
                    "model": "fake-model",
                    "usage": {
                        "input_tokens": 20,
                        "output_tokens": 3,
                        "total_tokens": 23,
                    },
                }
            ),
        ]
    )
    app = CayuApp(
        context_counting=ContextCountingConfig(mode=ContextCountingMode.OBSERVE),
        enable_logging=False,
    )
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_context_pressure_reconcile",
                messages=[Message.text("user", user_text)],
            ),
        )
    )

    event_types = [event.type for event in events]
    estimated_index = event_types.index(EventType.CONTEXT_PRESSURE_ESTIMATED)
    started_index = event_types.index(EventType.MODEL_STARTED)
    completed_index = event_types.index(EventType.MODEL_COMPLETED)
    reconciled_index = event_types.index(EventType.CONTEXT_PRESSURE_RECONCILED)
    assert estimated_index < started_index < completed_index < reconciled_index

    estimated = events[estimated_index]
    assert estimated.payload["provider"] == "fake"
    assert estimated.payload["model"] == "fake-model"
    assert estimated.payload["messages"] == {"count": 1, "roles": ["user"]}
    assert estimated.payload["tools"] == {"count": 0}
    assert estimated.payload["estimate"]["method"] == "local_full_request_estimate"
    assert estimated.payload["estimate"]["estimated_context_input_tokens"] > 0
    assert (
        estimated.payload["estimate"]["estimated_context_window_tokens"]
        == estimated.payload["estimate"]["estimated_context_input_tokens"]
    )
    assert user_text not in json.dumps(estimated.payload, sort_keys=True)

    reconciled = events[reconciled_index]
    assert reconciled.payload["observation_id"] == estimated.payload["observation_id"]
    pre_call_estimate = reconciled.payload["pre_call_estimate"]
    expected_delta = 20 - pre_call_estimate["estimated_context_input_tokens"]
    assert reconciled.payload["actual_input_tokens"] == 20
    assert reconciled.payload["delta_tokens"] == expected_delta
    assert reconciled.payload["relative_error"] == expected_delta / 20
    assert reconciled.payload["reconciled"] is True


def test_context_counting_observe_emits_count_and_reconciliation_events() -> None:
    user_text = "secret phrase should not be in context count event payload"
    provider = CountingProvider(
        [
            ModelStreamEvent.text_delta("done"),
            ModelStreamEvent.completed(
                {
                    "finish_reason": "stop",
                    "model": "fakemodel",
                    "usage": {
                        "input_tokens": 15,
                        "output_tokens": 2,
                        "total_tokens": 17,
                    },
                }
            ),
        ],
        count_result=InputTokenCountResult(
            input_tokens=12,
            method=InputTokenCountMethod.OFFICIAL,
            confidence=InputTokenCountConfidence.HIGH,
            components={"messages": 10, "tools": 2},
        ),
    )
    app = CayuApp(
        context_counting=ContextCountingConfig(mode=ContextCountingMode.OBSERVE),
        secret_redactor=SecretRedactor("-"),
        enable_logging=False,
    )
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fakemodel"))

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_context_counting_observe",
                messages=[Message.text("user", user_text)],
            ),
        )
    )

    event_types = [event.type for event in events]
    counted_index = event_types.index(EventType.CONTEXT_COUNTED)
    started_index = event_types.index(EventType.MODEL_STARTED)
    completed_index = event_types.index(EventType.MODEL_COMPLETED)
    reconciled_index = event_types.index(EventType.CONTEXT_COUNT_RECONCILED)
    assert counted_index < started_index < completed_index < reconciled_index
    assert len(provider.count_requests) == 1

    counted = events[counted_index]
    assert counted.payload["provider"] == "fake"
    assert counted.payload["model"] == "fakemodel"
    assert isinstance(counted.payload["observation_id"], str)
    assert counted.payload["observation_id"]
    assert counted.payload["messages"] == {"count": 1, "roles": ["user"]}
    assert counted.payload["tools"] == {"count": 0}
    assert counted.payload["count"] == {
        "input_tokens": 12,
        "method": "official",
        "confidence": "high",
        "components": {"messages": 10, "tools": 2},
        "metadata": {},
    }
    assert user_text not in json.dumps(counted.payload, sort_keys=True)

    reconciled = events[reconciled_index]
    assert reconciled.payload["observation_id"] == counted.payload["observation_id"]
    assert reconciled.payload["pre_call_count"]["input_tokens"] == 12
    assert reconciled.payload["actual_input_tokens"] == 15
    assert reconciled.payload["delta_tokens"] == 3
    assert reconciled.payload["relative_error"] == 0.2
    assert reconciled.payload["reconciled"] is True

    private_records = asyncio.run(
        app.session_store.query_events(
            EventQuery(
                session_id="sess_context_counting_observe",
                event_types=(
                    EventType.CONTEXT_PRESSURE_ESTIMATED,
                    EventType.CONTEXT_PRESSURE_RECONCILED,
                    EventType.CONTEXT_COUNTED,
                    EventType.CONTEXT_COUNT_RECONCILED,
                ),
            )
        )
    )
    private_observation_ids = {record.event.payload["observation_id"] for record in private_records}
    assert len(private_observation_ids) == 2
    assert all(
        str(UUID(observation_id)) == observation_id for observation_id in private_observation_ids
    )
    assert all(
        event.payload["observation_id"] == PRIVATE_EVENT_AUTHORITY
        for event in events
        if event.type
        in {
            EventType.CONTEXT_PRESSURE_ESTIMATED,
            EventType.CONTEXT_PRESSURE_RECONCILED,
            EventType.CONTEXT_COUNTED,
            EventType.CONTEXT_COUNT_RECONCILED,
        }
    )


def test_context_counting_uses_defensive_request_copy_for_provider_counter() -> None:
    class MutatingCountingProvider(CountingProvider):
        async def count_input_tokens(
            self,
            request: ModelRequest,
        ) -> InputTokenCountResult | None:
            result = await super().count_input_tokens(request)
            part = request.messages[0].content[0]
            assert isinstance(part, TextPart)
            # Message parts are frozen; a misbehaving counter cannot corrupt
            # the shared transcript at all.
            with pytest.raises(ValidationError):
                part.text = "mutated by counter"  # type: ignore[misc]
            request.tools.append({"name": "mutated_tool"})
            request.options["temperature"] = 1
            return result

    provider = MutatingCountingProvider(
        [
            ModelStreamEvent.text_delta("done"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ],
        count_result=InputTokenCountResult(
            input_tokens=12,
            method=InputTokenCountMethod.OFFICIAL,
            confidence=InputTokenCountConfidence.HIGH,
        ),
    )
    app = CayuApp(
        context_counting=ContextCountingConfig(mode=ContextCountingMode.OBSERVE),
        enable_logging=False,
    )
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_context_counting_copy",
                messages=[Message.text("user", "original prompt")],
            ),
        )
    )

    count_part = provider.count_requests[0].messages[0].content[0]
    stream_part = provider.requests[0].messages[0].content[0]
    assert isinstance(count_part, TextPart)
    assert isinstance(stream_part, TextPart)
    assert count_part.text == "original prompt"
    assert stream_part.text == "original prompt"
    assert provider.count_requests[0].tools == [{"name": "mutated_tool"}]
    assert provider.requests[0].tools == []
    assert provider.count_requests[0].options["temperature"] == 1
    assert "temperature" not in provider.requests[0].options


def test_context_counting_failure_is_observable_and_does_not_block_model_call() -> None:
    provider = CountingProvider(
        [
            ModelStreamEvent.text_delta("done"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ],
        count_error=RuntimeError("counter endpoint unavailable"),
    )
    app = CayuApp(
        context_counting=ContextCountingConfig(mode=ContextCountingMode.OBSERVE),
        enable_logging=False,
    )
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_context_counting_failure",
                messages=[Message.text("user", "hello")],
            ),
        )
    )

    event_types = [event.type for event in events]
    assert EventType.CONTEXT_COUNT_FAILED in event_types
    assert EventType.MODEL_COMPLETED in event_types
    assert EventType.CONTEXT_COUNTED not in event_types
    assert EventType.CONTEXT_COUNT_RECONCILED not in event_types

    failed = next(event for event in events if event.type == EventType.CONTEXT_COUNT_FAILED)
    assert failed.payload["error"] == "counter endpoint unavailable"
    assert failed.payload["error_type"] == "RuntimeError"


def test_runtime_adds_usage_metrics_to_model_completed_events():
    class RenamedOpenAIProvider(FakeProvider):
        name = "renamed-openai"
        usage_dialect = UsageDialect.OPENAI

    provider = RenamedOpenAIProvider(
        [
            ModelStreamEvent.text_delta("done"),
            ModelStreamEvent.completed(
                {
                    "model": "fake-model-version",
                    "usage": {
                        "input_tokens": 12,
                        "input_tokens_details": {"cached_tokens": 5},
                        "output_tokens": 3,
                    },
                }
            ),
        ]
    )
    app = CayuApp()
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="usage_runtime",
                messages=[Message.text("user", "hello")],
            ),
        )
    )

    completed = next(event for event in events if event.type == EventType.MODEL_COMPLETED)
    assert completed.payload["usage"] == {
        "input_tokens": 12,
        "input_tokens_details": {"cached_tokens": 5},
        "output_tokens": 3,
    }
    assert completed.payload["usage_metrics"] == {
        "provider_name": "renamed-openai",
        "requested_model": "fake-model",
        "model": "fake-model-version",
        "input_tokens": 12,
        "output_tokens": 3,
        "total_tokens": 15,
        "reasoning_output_tokens": 0,
        "cache": {
            "read_tokens": 5,
            "write_tokens": 0,
            "cached_input_tokens": 5,
            "uncached_input_tokens": 7,
        },
    }


def test_runtime_keeps_raw_model_completed_usage_when_it_cannot_normalize_usage_metrics():
    provider = FakeProvider(
        [
            ModelStreamEvent.text_delta("done"),
            ModelStreamEvent.completed(
                {
                    "model": "fake-model-version",
                    "usage": {
                        "provider_specific_counter": 123,
                        "provider_specific_cache_mode": "hit",
                    },
                }
            ),
        ]
    )
    app = CayuApp()
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="usage_runtime_raw_only",
                messages=[Message.text("user", "hello")],
            ),
        )
    )

    completed = next(event for event in events if event.type == EventType.MODEL_COMPLETED)
    assert completed.payload["usage"] == {
        "provider_specific_counter": 123,
        "provider_specific_cache_mode": "hit",
    }
    assert "usage_metrics" not in completed.payload


def test_cayu_app_get_session_usage_summarizes_durable_events():
    class OpenAIUsageProvider(FakeProvider):
        usage_dialect = UsageDialect.OPENAI

    provider = OpenAIUsageProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_1",
                    name="upper",
                    arguments={"text": "hello"},
                ),
                ModelStreamEvent.completed(
                    {
                        "usage": {
                            "input_tokens": 20,
                            "output_tokens": 4,
                            "input_tokens_details": {"cached_tokens": 8},
                        }
                    }
                ),
            ],
            [
                ModelStreamEvent.text_delta("done"),
                ModelStreamEvent.completed({"usage": {"input_tokens": 10, "output_tokens": 2}}),
            ],
        ]
    )
    app = CayuApp()
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[UpperTool()],
    )

    asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="usage_summary",
                messages=[Message.text("user", "uppercase hello")],
                max_steps=3,
            ),
        )
    )
    summary = asyncio.run(app.get_session_usage("usage_summary"))

    assert summary.model_steps == 2
    assert summary.tool_calls == 1
    assert summary.provider_names == ["fake"]
    assert summary.models == ["fake-model"]
    assert summary.usage.input_tokens == 30
    assert summary.usage.output_tokens == 6
    assert summary.usage.total_tokens == 36
    assert summary.usage.cache.cached_input_tokens == 8
    assert summary.usage.cache.uncached_input_tokens == 22


def test_cayu_app_reconstructs_gemini_thinking_usage() -> None:
    class GeminiUsageProvider(FakeProvider):
        name = "gemini"
        usage_dialect = UsageDialect.GEMINI

    provider = GeminiUsageProvider(
        [
            ModelStreamEvent.text_delta("OK."),
            ModelStreamEvent.completed(
                {
                    "model": "gemini-3.5-flash",
                    "usage": {
                        "prompt_tokens": 5,
                        "completion_tokens": 2,
                        "total_tokens": 60,
                    },
                }
            ),
        ]
    )
    app = CayuApp()
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="gemini-3.5-flash"))

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="gemini_thinking_usage",
                messages=[Message.text("user", "Reply with OK.")],
                limits=RunLimits(max_total_tokens=59),
            ),
        )
    )

    completed = next(event for event in events if event.type == EventType.MODEL_COMPLETED)
    assert "usage_normalization_failed" not in completed.payload
    assert completed.payload["usage_metrics"]["input_tokens"] == 5
    assert completed.payload["usage_metrics"]["output_tokens"] == 55
    assert completed.payload["usage_metrics"]["reasoning_output_tokens"] == 53
    assert completed.payload["usage_metrics"]["total_tokens"] == 60
    limit_reached = next(event for event in events if event.type == EventType.SESSION_LIMIT_REACHED)
    assert limit_reached.payload["limit"] == "total_tokens"
    assert limit_reached.payload["actual"] == 60
    assert limit_reached.payload["maximum"] == 59

    summary = asyncio.run(app.get_session_usage("gemini_thinking_usage"))
    assert summary.model_steps == 1
    assert summary.provider_names == ["gemini"]
    assert summary.models == ["gemini-3.5-flash"]
    assert summary.usage.input_tokens == 5
    assert summary.usage.output_tokens == 55
    assert summary.usage.reasoning_output_tokens == 53
    assert summary.usage.total_tokens == 60

    cost = asyncio.run(
        app.get_session_cost(
            "gemini_thinking_usage",
            PriceBook(
                prices=(
                    ModelPrice.fixed(
                        provider_name="gemini",
                        model="gemini-3.5-flash",
                        input_per_million=Decimal("1"),
                        output_per_million=Decimal("10"),
                    ),
                )
            ),
        )
    )
    assert cost.priced_model_steps == 1
    assert cost.line_items[0].input_tokens == 5
    assert cost.line_items[0].output_tokens == 55
    assert cost.total_cost == Decimal("0.000555")


def test_cayu_app_get_session_usage_queries_only_usage_relevant_events():
    class TrackingStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1

        def __init__(self) -> None:
            super().__init__()
            self.load_events_called = False
            self.event_queries: list[EventQuery] = []
            self.accounting_queries: list[EventQuery] = []

        async def read_usage_accounting(self, query: EventQuery, *, by_session: bool = False):
            self.accounting_queries.append(query)
            return await super().read_usage_accounting(query, by_session=by_session)

        async def load_events(self, session_id: str) -> list[Event]:
            self.load_events_called = True
            return await super().load_events(session_id)

        async def query_events(self, query: EventQuery | None = None):
            if query is not None:
                self.event_queries.append(query)
            return await super().query_events(query)

    store = TrackingStore()
    app = CayuApp(session_store=store)

    async def run() -> None:
        await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="usage_query",
                messages=[Message.text("user", "hello")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        await store.append_events(
            "usage_query",
            [
                Event(type=EventType.SESSION_STARTED, session_id="usage_query"),
                Event(
                    type=EventType.MODEL_COMPLETED,
                    session_id="usage_query",
                    payload={"usage": {"input_tokens": 10, "output_tokens": 2}},
                ),
                Event(type=EventType.MODEL_TEXT_DELTA, session_id="usage_query"),
                Event(type=EventType.TOOL_CALL_STARTED, session_id="usage_query"),
                Event(type=EventType.TOOL_CALL_COMPLETED, session_id="usage_query"),
            ],
        )

        summary = await app.get_session_usage("usage_query")

        assert summary.model_steps == 1
        assert summary.tool_calls == 1
        assert summary.usage.input_tokens == 10
        assert summary.usage.output_tokens == 2

    asyncio.run(run())

    assert store.load_events_called is False
    # The store owns one snapshot and one watermark across all usage-bearing
    # types. The app never pages full event payloads into a retained history.
    assert store.event_queries == []
    assert len(store.accounting_queries) == 1
    assert store.accounting_queries[0].session_id == "usage_query"


def test_query_all_event_records_preserves_filters():
    from cayu.runtime._session_queries import query_all_event_records

    store = InMemorySessionStore()

    async def run() -> None:
        for session_id in ("query_all_a", "query_all_b"):
            await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "hello")],
                ),
                identity=SessionIdentity(provider_name="fake", model="fake-model"),
            )
        await store.append_event(
            "query_all_a",
            Event(
                type=EventType.MODEL_COMPLETED,
                session_id="query_all_a",
                timestamp=datetime(2026, 1, 1, 12, 0, tzinfo=UTC),
            ),
        )
        await store.append_event(
            "query_all_b",
            Event(
                type=EventType.MODEL_COMPLETED,
                session_id="query_all_b",
                timestamp=datetime(2026, 1, 1, 13, 0, tzinfo=UTC),
            ),
        )

        records = await query_all_event_records(
            store,
            EventQuery(
                session_ids=("query_all_b",),
                event_type=EventType.MODEL_COMPLETED,
                since=datetime(2026, 1, 1, 12, 30, tzinfo=UTC),
                until=datetime(2026, 1, 1, 13, 30, tzinfo=UTC),
                limit=1,
            ),
        )
        assert [record.event.session_id for record in records] == ["query_all_b"]

        # The paginator rebuilds the query field-by-field; a dropped
        # `event_types` would silently widen usage reads (the delta leaks in).
        delta_event = Event(
            type=EventType.MODEL_TEXT_DELTA,
            session_id="query_all_b",
            payload={"delta": "noise"},
        )
        await store.append_event("query_all_b", delta_event)
        multi_type_records = await query_all_event_records(
            store,
            EventQuery(
                session_ids=("query_all_b",),
                event_types=(EventType.MODEL_COMPLETED, EventType.TOOL_CALL_STARTED),
                limit=1,
            ),
        )
        assert [record.event.type for record in multi_type_records] == [EventType.MODEL_COMPLETED]

        descending_records = await query_all_event_records(
            store,
            EventQuery(
                session_id="query_all_b",
                order_by=EventOrder.SEQUENCE_DESC,
                limit=1,
            ),
        )
        assert [record.event.type for record in descending_records] == [
            EventType.MODEL_TEXT_DELTA,
            EventType.MODEL_COMPLETED,
        ]

        multi_session_descending_records = await query_all_event_records(
            store,
            EventQuery(
                session_ids=("query_all_a", "query_all_b"),
                order_by=EventOrder.SEQUENCE_DESC,
                limit=1,
            ),
        )
        assert [record.event.type for record in multi_session_descending_records] == [
            EventType.MODEL_TEXT_DELTA,
            EventType.MODEL_COMPLETED,
            EventType.MODEL_COMPLETED,
        ]
        assert [record.event.session_id for record in multi_session_descending_records] == [
            "query_all_b",
            "query_all_b",
            "query_all_a",
        ]

        bounded_records = await query_all_event_records(
            store,
            EventQuery(
                session_id="query_all_b",
                before_sequence=descending_records[0].sequence,
                order_by=EventOrder.SEQUENCE_DESC,
                limit=1,
            ),
        )
        assert [record.event.type for record in bounded_records] == [EventType.MODEL_COMPLETED]

        event_id_records = await query_all_event_records(
            store,
            EventQuery(
                session_id="query_all_b",
                event_id=delta_event.id,
                limit=1,
            ),
        )
        assert [record.event.id for record in event_id_records] == [delta_event.id]

        await store.append_events(
            "query_all_b",
            [
                Event(
                    id="workflow-target-1",
                    type=EventType.WORKFLOW_STEP_COMPLETED,
                    session_id="query_all_b",
                    interaction_id="workflow-interaction",
                    workflow_name="maintenance",
                    payload={"attempt_id": "attempt-2", "step_id": "step-a"},
                ),
                Event(
                    id="workflow-other-interaction",
                    type=EventType.WORKFLOW_STEP_COMPLETED,
                    session_id="query_all_b",
                    interaction_id="other-interaction",
                    workflow_name="maintenance",
                    payload={"attempt_id": "attempt-2", "step_id": "step-a"},
                ),
                Event(
                    id="workflow-other-step",
                    type=EventType.WORKFLOW_STEP_COMPLETED,
                    session_id="query_all_b",
                    interaction_id="workflow-interaction",
                    workflow_name="maintenance",
                    payload={"attempt_id": "attempt-2", "step_id": "step-b"},
                ),
                Event(
                    id="workflow-target-2",
                    type=EventType.WORKFLOW_STEP_COMPLETED,
                    session_id="query_all_b",
                    interaction_id="workflow-interaction",
                    workflow_name="maintenance",
                    payload={"attempt_id": "attempt-2", "step_id": "step-a"},
                ),
            ],
        )
        workflow_records = await query_all_event_records(
            store,
            EventQuery(
                session_id="query_all_b",
                interaction_id="workflow-interaction",
                event_type=EventType.WORKFLOW_STEP_COMPLETED,
                workflow_name="maintenance",
                workflow_attempt_id="attempt-2",
                workflow_step_id="step-a",
                limit=1,
            ),
        )
        assert [record.event.id for record in workflow_records] == [
            "workflow-target-1",
            "workflow-target-2",
        ]

        await store.append_events(
            "query_all_b",
            [
                Event(
                    id="workflow-attempt-1",
                    type="custom.cayu.workflow.attempt",
                    session_id="query_all_b",
                    workflow_name="maintenance",
                    payload={"attempt_id": "attempt-1"},
                ),
                Event(
                    id="workflow-valid-attempt-1",
                    type=EventType.WORKFLOW_STEP_STARTED,
                    session_id="query_all_b",
                    workflow_name="maintenance",
                    payload={"attempt_id": "attempt-1", "step_id": "step-a"},
                ),
                Event(
                    id="workflow-attempt-2",
                    type="custom.cayu.workflow.attempt",
                    session_id="query_all_b",
                    workflow_name="maintenance",
                    payload={"attempt_id": "attempt-2"},
                ),
                Event(
                    id="workflow-stale-attempt-1",
                    type=EventType.WORKFLOW_STEP_STARTED,
                    session_id="query_all_b",
                    workflow_name="maintenance",
                    payload={"attempt_id": "attempt-1", "step_id": "step-a"},
                ),
                Event(
                    id="workflow-valid-attempt-2",
                    type=EventType.WORKFLOW_STEP_STARTED,
                    session_id="query_all_b",
                    workflow_name="maintenance",
                    payload={"attempt_id": "attempt-2", "step_id": "step-a"},
                ),
            ],
        )
        fenced_records = await query_all_event_records(
            store,
            EventQuery(
                session_id="query_all_b",
                event_type=EventType.WORKFLOW_STEP_STARTED,
                workflow_name="maintenance",
                workflow_step_id="step-a",
                workflow_attempt_fenced=True,
                limit=1,
            ),
        )
        assert [record.event.id for record in fenced_records] == [
            "workflow-valid-attempt-1",
            "workflow-valid-attempt-2",
        ]

    asyncio.run(run())


def test_cayu_app_usage_summary_spans_tool_approval_pause_and_resume():
    # Composition pin: usage accounting rebuilds from the full event tail on
    # every entrance, so totals must span both segments of a paused run.
    store = InMemorySessionStore()
    tool = SideEffectTool()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_1",
                    name="side_effect",
                    arguments={"value": "pause here"},
                ),
                ModelStreamEvent.completed(
                    {
                        "finish_reason": "tool_calls",
                        "usage": {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12},
                    }
                ),
            ],
            [
                ModelStreamEvent.text_delta("done"),
                ModelStreamEvent.completed(
                    {
                        "finish_reason": "stop",
                        "usage": {"input_tokens": 20, "output_tokens": 4, "total_tokens": 24},
                    }
                ),
            ],
        ]
    )
    app = CayuApp(session_store=store)
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[tool],
        tool_policy=RequireApprovalPolicy(),
    )

    interrupt_events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_usage_across_pause",
                messages=[Message.text("user", "use the tool")],
            ),
        )
    )
    approval_event = next(
        event for event in interrupt_events if event.type == EventType.TOOL_CALL_APPROVAL_REQUESTED
    )
    approval_id = approval_event.payload["approval"]["approval_id"]

    resume_events = asyncio.run(
        collect_tool_approval_events(
            app,
            ToolApprovalRequest(
                session_id="sess_usage_across_pause",
                approval_id=approval_id,
                tool_round_id=approval_event.payload["tool_round_id"],
                tool_call_id=approval_event.payload["tool_call_id"],
                decision=ToolApprovalDecision.APPROVE,
            ),
        )
    )
    assert resume_events[-1].type == EventType.SESSION_COMPLETED

    summary = asyncio.run(app.get_session_usage("sess_usage_across_pause"))
    assert summary.model_steps == 2
    assert summary.tool_calls == 1
    assert summary.usage.total_tokens == 36

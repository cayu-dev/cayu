"""Budget limits contracts."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from decimal import Decimal

from tests.core._execution_profile_fixtures import (
    profiled_session_identity,
    run_request_with_registered_tool_ceiling,
)
from tests.runtime.runtime_budget_support import fake_budget_limit
from tests.runtime.runtime_provider_support import FakeProvider
from tests.runtime.runtime_session_support import collect_events, collect_resume_events
from tests.runtime.runtime_tool_support import SideEffectTool, UpperTool

from cayu.agents import AgentSpec
from cayu.applications import CayuApp
from cayu.budgets.base import BudgetLimit, BudgetPolicy, BudgetWindow
from cayu.budgets.pricing import ModelPrice, PriceBook
from cayu.budgets.run_limits import RunLimits
from cayu.events import Event, EventType
from cayu.messages import Message
from cayu.providers.base import ModelStreamEvent
from cayu.sessions.memory import InMemorySessionStore
from cayu.sessions.records import SessionStatus
from cayu.sessions.requests import ResumeRequest, RunRequest


def test_cayu_app_request_notify_budget_emits_event_without_stopping_session():
    store = InMemorySessionStore()
    provider = FakeProvider(
        [
            ModelStreamEvent.text_delta("final answer"),
            ModelStreamEvent.completed(
                {
                    "finish_reason": "stop",
                    "usage": {
                        "input_tokens": 1000,
                        "output_tokens": 100,
                        "total_tokens": 1100,
                    },
                }
            ),
        ]
    )
    app = CayuApp(session_store=store)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_request_notify_budget",
                messages=[Message.text("user", "answer")],
                budget_limits=(fake_budget_limit("0.002", action="notify"),),
            ),
        )
    )
    session = asyncio.run(store.load("sess_request_notify_budget"))

    assert [event.type for event in events] == [
        EventType.SESSION_STARTED,
        EventType.MODEL_STARTED,
        EventType.MODEL_TEXT_DELTA,
        EventType.MODEL_COMPLETED,
        EventType.BUDGET_LIMIT_REACHED,
        EventType.TURN_COMPLETED,
        EventType.SESSION_COMPLETED,
    ]
    assert events[4].payload["action"] == "notify"
    assert events[4].payload["limit_reached"] is True
    assert events[4].payload["maximum"] == "0.002"
    assert events[4].payload["actual"] == "0.002"
    assert session is not None
    assert session.status == SessionStatus.COMPLETED


def test_cayu_app_request_notify_budget_dedupes_during_tool_round():
    store = InMemorySessionStore()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_1",
                    name="upper",
                    arguments={"text": "one"},
                ),
                ModelStreamEvent.tool_call(
                    id="call_2",
                    name="upper",
                    arguments={"text": "two"},
                ),
                ModelStreamEvent.completed(
                    {
                        "finish_reason": "tool_calls",
                        "usage": {
                            "input_tokens": 1000,
                            "output_tokens": 100,
                            "total_tokens": 1100,
                        },
                    }
                ),
            ],
            [
                ModelStreamEvent.text_delta("done"),
                ModelStreamEvent.completed(
                    {
                        "finish_reason": "stop",
                        "usage": {
                            "input_tokens": 1,
                            "output_tokens": 1,
                            "total_tokens": 2,
                        },
                    }
                ),
            ],
        ]
    )
    app = CayuApp(session_store=store)
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[UpperTool()],
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_request_notify_budget_tool_round",
                messages=[Message.text("user", "call tools")],
                budget_limits=(fake_budget_limit("0.002", action="notify"),),
                max_steps=2,
            ),
        )
    )
    session = asyncio.run(store.load("sess_request_notify_budget_tool_round"))
    budget_events = [event for event in events if event.type == EventType.BUDGET_LIMIT_REACHED]

    assert [event.payload["action"] for event in budget_events] == ["notify"]
    assert [event.type for event in events].count(EventType.TOOL_CALL_COMPLETED) == 2
    assert session is not None
    assert session.status == SessionStatus.COMPLETED


def test_cayu_app_request_session_budget_uses_rolling_window():
    async def run():
        store = InMemorySessionStore()
        rolling_limit = fake_budget_limit(
            "0.50",
            window=BudgetWindow.rolling(seconds=60),
        )
        provider = FakeProvider(
            [
                ModelStreamEvent.text_delta("new answer"),
                ModelStreamEvent.completed(
                    {
                        "finish_reason": "stop",
                        "usage": {
                            "input_tokens": 1_000,
                            "output_tokens": 0,
                            "total_tokens": 1_000,
                        },
                    }
                ),
            ]
        )
        app = CayuApp(session_store=store)
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        await store.create(
            run_request_with_registered_tool_ceiling(
                app,
                agent_name="assistant",
                session_id="sess_rolling_request_budget",
                messages=[Message.text("user", "old")],
                budget_limits=(rolling_limit,),
            ),
            identity=profiled_session_identity(
                provider_name="fake",
                model="fake-model",
                request_budget_limits=(rolling_limit,),
                causal_budget_id="sess_rolling_request_budget",
                app=app,
            ),
        )
        await store.append_event(
            "sess_rolling_request_budget",
            Event(
                type=EventType.MODEL_COMPLETED,
                session_id="sess_rolling_request_budget",
                timestamp=datetime.now(UTC) - timedelta(seconds=120),
                payload={
                    "usage_metrics": {
                        "provider_name": "fake",
                        "model": "fake-model",
                        "input_tokens": 1_000_000,
                        "output_tokens": 0,
                        "total_tokens": 1_000_000,
                    }
                },
            ),
        )
        await store.update_status("sess_rolling_request_budget", SessionStatus.COMPLETED)

        events = await collect_resume_events(
            app,
            ResumeRequest(
                session_id="sess_rolling_request_budget",
                messages=[Message.text("user", "continue")],
                budget_limits=(rolling_limit,),
            ),
        )
        session = await store.load("sess_rolling_request_budget")
        return events, session, provider

    events, session, provider = asyncio.run(run())

    assert [event.type for event in events] == [
        EventType.SESSION_RESUMED,
        EventType.MODEL_STARTED,
        EventType.MODEL_TEXT_DELTA,
        EventType.MODEL_COMPLETED,
        EventType.TURN_COMPLETED,
        EventType.SESSION_COMPLETED,
    ]
    assert session is not None
    assert session.status == SessionStatus.COMPLETED
    assert len(provider.requests) == 1


def test_cayu_app_budget_limit_stops_before_tool_side_effects():
    store = InMemorySessionStore()
    tool = SideEffectTool()
    provider = FakeProvider(
        [
            ModelStreamEvent.tool_call(
                id="call_1",
                name="side_effect",
                arguments={},
            ),
            ModelStreamEvent.completed(
                {
                    "finish_reason": "tool_calls",
                    "usage": {
                        "input_tokens": 1000,
                        "output_tokens": 100,
                        "total_tokens": 1100,
                    },
                }
            ),
        ]
    )
    app = CayuApp(session_store=store)
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[tool],
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_cost_limit_tool",
                messages=[Message.text("user", "do it")],
                budget_limits=(fake_budget_limit("0.002"),),
            ),
        )
    )

    assert [event.type for event in events] == [
        EventType.SESSION_STARTED,
        EventType.MODEL_STARTED,
        EventType.MODEL_COMPLETED,
        EventType.SESSION_LIMIT_REACHED,
        EventType.TOOL_CALL_FAILED,
        EventType.TURN_COMPLETED,
        EventType.SESSION_INTERRUPTED,
    ]
    assert events[3].payload["limit"] == "estimated_cost"
    assert events[4].payload["reason"] == "limit_reached"
    assert tool.calls == []

    transcript = asyncio.run(store.load_transcript("sess_cost_limit_tool"))
    assert [message.role for message in transcript] == ["user", "assistant", "tool"]


def test_cayu_app_budget_limit_fails_closed_when_model_step_is_unpriced():
    store = InMemorySessionStore()
    provider = FakeProvider(
        [
            ModelStreamEvent.text_delta("final answer"),
            ModelStreamEvent.completed(
                {
                    "finish_reason": "stop",
                    "usage": {
                        "input_tokens": 1,
                        "output_tokens": 1,
                        "total_tokens": 2,
                    },
                }
            ),
        ]
    )
    app = CayuApp(session_store=store)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_cost_limit_unpriced",
                messages=[Message.text("user", "answer")],
                budget_limits=(
                    BudgetLimit(
                        max_estimated_cost=Decimal("100"),
                        pricing=PriceBook(
                            prices=(
                                ModelPrice.fixed(
                                    provider_name="fake",
                                    model="other-model",
                                    input_per_million=Decimal("1"),
                                    output_per_million=Decimal("1"),
                                ),
                            )
                        ),
                    ),
                ),
            ),
        )
    )

    assert [event.type for event in events] == [
        EventType.SESSION_STARTED,
        EventType.SESSION_LIMIT_REACHED,
        EventType.TURN_COMPLETED,
        EventType.SESSION_INTERRUPTED,
    ]
    assert len(provider.requests) == 0
    assert events[1].payload["limit"] == "estimated_cost"
    assert "no matching pricing" in events[1].payload["message"]
    assert events[1].payload["cost_summary"]["unpriced_model_steps"] == 0


def test_cayu_app_budget_limit_allows_unpriced_steps_when_explicitly_configured():
    provider = FakeProvider(
        [
            ModelStreamEvent.text_delta("final answer"),
            ModelStreamEvent.completed(
                {
                    "finish_reason": "stop",
                    "usage": {
                        "input_tokens": 1,
                        "output_tokens": 1,
                        "total_tokens": 2,
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
                session_id="sess_cost_limit_unpriced_allowed",
                messages=[Message.text("user", "answer")],
                budget_limits=(
                    BudgetLimit(
                        max_estimated_cost=Decimal("100"),
                        pricing=PriceBook(
                            prices=(
                                ModelPrice.fixed(
                                    provider_name="fake",
                                    model="other-model",
                                    input_per_million=Decimal("1"),
                                    output_per_million=Decimal("1"),
                                ),
                            )
                        ),
                        allow_unpriced=True,
                    ),
                ),
            ),
        )
    )

    assert [event.type for event in events] == [
        EventType.SESSION_STARTED,
        EventType.MODEL_STARTED,
        EventType.MODEL_TEXT_DELTA,
        EventType.MODEL_COMPLETED,
        EventType.TURN_COMPLETED,
        EventType.SESSION_COMPLETED,
    ]


def test_cayu_app_app_budget_applies_across_sessions():
    store = InMemorySessionStore()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.text_delta("first"),
                ModelStreamEvent.completed(
                    {
                        "finish_reason": "stop",
                        "usage": {
                            "input_tokens": 1_000_000,
                            "output_tokens": 0,
                            "total_tokens": 1_000_000,
                        },
                    }
                ),
            ],
            [
                ModelStreamEvent.text_delta("second"),
                ModelStreamEvent.completed(
                    {
                        "finish_reason": "stop",
                        "usage": {
                            "input_tokens": 1_000_000,
                            "output_tokens": 0,
                            "total_tokens": 1_000_000,
                        },
                    }
                ),
            ],
        ]
    )
    app = CayuApp(
        session_store=store,
        budget_policy=BudgetPolicy(
            limits=(
                BudgetLimit(
                    scope="app",
                    max_estimated_cost=Decimal("1"),
                    pricing=fake_budget_limit("10").pricing,
                ),
            )
        ),
    )
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    first_events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_app_budget_first",
                messages=[Message.text("user", "first")],
            ),
        )
    )
    second_events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_app_budget_second",
                messages=[Message.text("user", "second")],
            ),
        )
    )
    second_session = asyncio.run(store.load("sess_app_budget_second"))

    assert [event.type for event in first_events] == [
        EventType.SESSION_STARTED,
        EventType.BUDGET_CHECKED,
        EventType.MODEL_STARTED,
        EventType.MODEL_TEXT_DELTA,
        EventType.MODEL_COMPLETED,
        EventType.BUDGET_CHECKED,
        EventType.BUDGET_LIMIT_REACHED,
        EventType.SESSION_LIMIT_REACHED,
        EventType.TURN_COMPLETED,
        EventType.SESSION_INTERRUPTED,
    ]
    assert [event.type for event in second_events] == [
        EventType.SESSION_STARTED,
        EventType.BUDGET_CHECKED,
        EventType.BUDGET_LIMIT_REACHED,
        EventType.SESSION_LIMIT_REACHED,
        EventType.TURN_COMPLETED,
        EventType.SESSION_INTERRUPTED,
    ]
    assert second_events[1].payload["scope"] == "app"
    assert second_events[1].payload["actual"] == "1"
    assert second_events[2].payload["limit_reached"] is True
    assert second_session is not None
    assert second_session.status == SessionStatus.INTERRUPTED
    assert len(provider.requests) == 1


def test_cayu_app_notify_budget_emits_event_without_stopping_session():
    store = InMemorySessionStore()
    provider = FakeProvider(
        [
            ModelStreamEvent.text_delta("first"),
            ModelStreamEvent.completed(
                {
                    "finish_reason": "stop",
                    "usage": {
                        "input_tokens": 1_000_000,
                        "output_tokens": 0,
                        "total_tokens": 1_000_000,
                    },
                }
            ),
        ]
    )
    app = CayuApp(
        session_store=store,
        budget_policy=BudgetPolicy(
            limits=(
                BudgetLimit(
                    scope="app",
                    max_estimated_cost=Decimal("1"),
                    pricing=fake_budget_limit("10").pricing,
                    action="notify",
                ),
            )
        ),
    )
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_app_budget_notify",
                messages=[Message.text("user", "first")],
            ),
        )
    )
    session = asyncio.run(store.load("sess_app_budget_notify"))

    assert [event.type for event in events] == [
        EventType.SESSION_STARTED,
        EventType.BUDGET_CHECKED,
        EventType.MODEL_STARTED,
        EventType.MODEL_TEXT_DELTA,
        EventType.MODEL_COMPLETED,
        EventType.BUDGET_CHECKED,
        EventType.BUDGET_LIMIT_REACHED,
        EventType.TURN_COMPLETED,
        EventType.SESSION_COMPLETED,
    ]
    assert events[6].payload["action"] == "notify"
    assert events[6].payload["limit_reached"] is True
    assert session is not None
    assert session.status == SessionStatus.COMPLETED
    assert len(provider.requests) == 1


def test_cayu_app_policy_can_notify_before_stricter_interrupt_budget():
    store = InMemorySessionStore()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.text_delta("first"),
                ModelStreamEvent.completed(
                    {
                        "finish_reason": "stop",
                        "usage": {
                            "input_tokens": 1_000_000,
                            "output_tokens": 0,
                            "total_tokens": 1_000_000,
                        },
                    }
                ),
            ],
            [
                ModelStreamEvent.text_delta("second"),
                ModelStreamEvent.completed(
                    {
                        "finish_reason": "stop",
                        "usage": {
                            "input_tokens": 1_000_000,
                            "output_tokens": 0,
                            "total_tokens": 1_000_000,
                        },
                    }
                ),
            ],
        ]
    )
    app = CayuApp(
        session_store=store,
        budget_policy=BudgetPolicy(
            limits=(
                BudgetLimit(
                    scope="app",
                    max_estimated_cost=Decimal("1"),
                    pricing=fake_budget_limit("10").pricing,
                    action="notify",
                ),
                BudgetLimit(
                    scope="app",
                    max_estimated_cost=Decimal("2"),
                    pricing=fake_budget_limit("10").pricing,
                    action="interrupt",
                ),
            )
        ),
    )
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    first_events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_budget_notify_then_interrupt_first",
                messages=[Message.text("user", "first")],
            ),
        )
    )
    second_events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_budget_notify_then_interrupt_second",
                messages=[Message.text("user", "second")],
            ),
        )
    )
    first_session = asyncio.run(store.load("sess_budget_notify_then_interrupt_first"))
    second_session = asyncio.run(store.load("sess_budget_notify_then_interrupt_second"))

    assert EventType.BUDGET_LIMIT_REACHED in [event.type for event in first_events]
    first_notify_events = [
        event
        for event in first_events
        if event.type == EventType.BUDGET_LIMIT_REACHED and event.payload["action"] == "notify"
    ]
    assert len(first_notify_events) == 1
    assert first_events[-1].type == EventType.SESSION_COMPLETED
    assert [event.type for event in second_events] == [
        EventType.SESSION_STARTED,
        EventType.BUDGET_CHECKED,
        EventType.BUDGET_CHECKED,
        EventType.MODEL_STARTED,
        EventType.MODEL_TEXT_DELTA,
        EventType.MODEL_COMPLETED,
        EventType.BUDGET_CHECKED,
        EventType.BUDGET_CHECKED,
        EventType.BUDGET_LIMIT_REACHED,
        EventType.SESSION_LIMIT_REACHED,
        EventType.TURN_COMPLETED,
        EventType.SESSION_INTERRUPTED,
    ]
    assert second_events[8].payload["action"] == "interrupt"
    assert first_session is not None
    assert first_session.status == SessionStatus.COMPLETED
    assert second_session is not None
    assert second_session.status == SessionStatus.INTERRUPTED
    assert len(provider.requests) == 2


def test_cayu_app_request_app_budget_limit_applies_across_sessions():
    store = InMemorySessionStore()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.text_delta("first"),
                ModelStreamEvent.completed(
                    {
                        "finish_reason": "stop",
                        "usage": {
                            "input_tokens": 1_000_000,
                            "output_tokens": 0,
                            "total_tokens": 1_000_000,
                        },
                    }
                ),
            ],
            [
                ModelStreamEvent.text_delta("should not run"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    app = CayuApp(session_store=store)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    first_events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_request_app_budget_first",
                messages=[Message.text("user", "first")],
            ),
        )
    )
    second_events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_request_app_budget_second",
                messages=[Message.text("user", "second")],
                budget_limits=(
                    BudgetLimit(
                        scope="app",
                        max_estimated_cost=Decimal("1"),
                        pricing=fake_budget_limit("10").pricing,
                    ),
                ),
            ),
        )
    )

    assert first_events[-1].type == EventType.SESSION_COMPLETED
    assert [event.type for event in second_events] == [
        EventType.SESSION_STARTED,
        EventType.SESSION_LIMIT_REACHED,
        EventType.TURN_COMPLETED,
        EventType.SESSION_INTERRUPTED,
    ]
    assert second_events[1].payload["limit"] == "estimated_cost"
    assert second_events[1].payload["actual"] == "1"
    assert second_events[1].payload["cost_summary"]["total_cost"] == "1"
    assert len(provider.requests) == 1


def test_cayu_app_agent_budget_only_applies_to_matching_agent():
    store = InMemorySessionStore()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.text_delta("builder"),
                ModelStreamEvent.completed(
                    {
                        "finish_reason": "stop",
                        "usage": {
                            "input_tokens": 1_000_000,
                            "output_tokens": 0,
                            "total_tokens": 1_000_000,
                        },
                    }
                ),
            ],
            [
                ModelStreamEvent.text_delta("researcher"),
                ModelStreamEvent.completed(
                    {
                        "finish_reason": "stop",
                        "usage": {
                            "input_tokens": 1,
                            "output_tokens": 0,
                            "total_tokens": 1,
                        },
                    }
                ),
            ],
        ]
    )
    app = CayuApp(
        session_store=store,
        budget_policy=BudgetPolicy(
            limits=(
                BudgetLimit(
                    scope="agent",
                    key="builder",
                    max_estimated_cost=Decimal("1"),
                    pricing=fake_budget_limit("10").pricing,
                ),
            )
        ),
    )
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="builder", model="fake-model"))
    app.register_agent(AgentSpec(name="researcher", model="fake-model"))

    builder_events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="builder",
                session_id="sess_agent_budget_builder",
                messages=[Message.text("user", "builder")],
            ),
        )
    )
    researcher_events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="researcher",
                session_id="sess_agent_budget_researcher",
                messages=[Message.text("user", "researcher")],
            ),
        )
    )

    assert EventType.BUDGET_CHECKED in [event.type for event in builder_events]
    assert EventType.BUDGET_CHECKED not in [event.type for event in researcher_events]
    assert researcher_events[-1].type == EventType.SESSION_COMPLETED
    assert len(provider.requests) == 2


def test_cayu_app_request_agent_budget_limit_applies_to_matching_agent_history():
    store = InMemorySessionStore()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.text_delta("first"),
                ModelStreamEvent.completed(
                    {
                        "finish_reason": "stop",
                        "usage": {
                            "input_tokens": 1_000_000,
                            "output_tokens": 0,
                            "total_tokens": 1_000_000,
                        },
                    }
                ),
            ],
            [
                ModelStreamEvent.text_delta("should not run"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    app = CayuApp(session_store=store)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="builder", model="fake-model"))

    first_events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="builder",
                session_id="sess_request_agent_budget_first",
                messages=[Message.text("user", "first")],
            ),
        )
    )
    second_events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="builder",
                session_id="sess_request_agent_budget_second",
                messages=[Message.text("user", "second")],
                budget_limits=(
                    BudgetLimit(
                        scope="agent",
                        key="builder",
                        max_estimated_cost=Decimal("1"),
                        pricing=fake_budget_limit("10").pricing,
                    ),
                ),
            ),
        )
    )

    assert first_events[-1].type == EventType.SESSION_COMPLETED
    assert [event.type for event in second_events] == [
        EventType.SESSION_STARTED,
        EventType.SESSION_LIMIT_REACHED,
        EventType.TURN_COMPLETED,
        EventType.SESSION_INTERRUPTED,
    ]
    assert second_events[1].payload["actual"] == "1"
    assert len(provider.requests) == 1


def test_cayu_app_budget_fails_closed_for_unpriced_model_steps():
    store = InMemorySessionStore()
    provider = FakeProvider(
        [
            ModelStreamEvent.text_delta("first"),
            ModelStreamEvent.completed(
                {
                    "finish_reason": "stop",
                    "usage": {
                        "input_tokens": 1,
                        "output_tokens": 1,
                        "total_tokens": 2,
                    },
                }
            ),
        ]
    )
    app = CayuApp(
        session_store=store,
        budget_policy=BudgetPolicy(
            limits=(
                BudgetLimit(
                    scope="app",
                    max_estimated_cost=Decimal("100"),
                    pricing=PriceBook(
                        prices=(
                            ModelPrice.fixed(
                                provider_name="fake",
                                model="other-model",
                                input_per_million=Decimal("1"),
                                output_per_million=Decimal("1"),
                            ),
                        )
                    ),
                ),
            )
        ),
    )
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    first_events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_budget_unpriced_first",
                messages=[Message.text("user", "first")],
            ),
        )
    )
    second_events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_budget_unpriced_second",
                messages=[Message.text("user", "second")],
            ),
        )
    )

    assert [event.type for event in first_events] == [
        EventType.SESSION_STARTED,
        EventType.BUDGET_CHECKED,
        EventType.BUDGET_LIMIT_REACHED,
        EventType.SESSION_LIMIT_REACHED,
        EventType.TURN_COMPLETED,
        EventType.SESSION_INTERRUPTED,
    ]
    assert "no matching model pricing" in first_events[2].payload["message"]
    assert first_events[2].payload["unpriced_model_steps"] == 0
    assert [event.type for event in second_events] == [
        EventType.SESSION_STARTED,
        EventType.BUDGET_CHECKED,
        EventType.BUDGET_LIMIT_REACHED,
        EventType.SESSION_LIMIT_REACHED,
        EventType.TURN_COMPLETED,
        EventType.SESSION_INTERRUPTED,
    ]
    assert "no matching model pricing" in second_events[2].payload["message"]
    assert second_events[2].payload["unpriced_model_steps"] == 0
    assert len(provider.requests) == 0


def test_cayu_app_run_scoped_budget_limit_ignores_prior_session_cost():
    store = InMemorySessionStore()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.text_delta("first answer"),
                ModelStreamEvent.completed(
                    {
                        "finish_reason": "stop",
                        "usage": {
                            "input_tokens": 1000,
                            "output_tokens": 100,
                            "total_tokens": 1100,
                        },
                    }
                ),
            ],
            [
                ModelStreamEvent.text_delta("second answer"),
                ModelStreamEvent.completed(
                    {
                        "finish_reason": "stop",
                        "usage": {
                            "input_tokens": 1,
                            "output_tokens": 1,
                            "total_tokens": 2,
                        },
                    }
                ),
            ],
        ]
    )
    app = CayuApp(session_store=store)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    run_budget = fake_budget_limit("0.002", scope="run")
    asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_run_scope_cost",
                messages=[Message.text("user", "first")],
                budget_limits=(run_budget,),
            ),
        )
    )

    events = asyncio.run(
        collect_resume_events(
            app,
            ResumeRequest(
                session_id="sess_run_scope_cost",
                messages=[Message.text("user", "second")],
                budget_limits=(run_budget,),
            ),
        )
    )

    assert [event.type for event in events] == [
        EventType.SESSION_RESUMED,
        EventType.MODEL_STARTED,
        EventType.MODEL_TEXT_DELTA,
        EventType.MODEL_COMPLETED,
        EventType.TURN_COMPLETED,
        EventType.SESSION_COMPLETED,
    ]
    assert len(provider.requests) == 2


def test_cayu_app_run_scoped_limit_still_trips_within_a_single_run():
    store = InMemorySessionStore()
    tool = SideEffectTool()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_1",
                    name="side_effect",
                    arguments={"step": 1},
                ),
                ModelStreamEvent.completed(
                    {
                        "finish_reason": "tool_calls",
                        "usage": {"input_tokens": 1, "output_tokens": 1},
                    }
                ),
            ],
            [
                ModelStreamEvent.tool_call(
                    id="call_2",
                    name="side_effect",
                    arguments={"step": 2},
                ),
                ModelStreamEvent.completed(
                    {
                        "finish_reason": "tool_calls",
                        "usage": {"input_tokens": 1, "output_tokens": 1},
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
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_run_scope_runaway",
                messages=[Message.text("user", "do it")],
                limits=RunLimits(max_tool_calls=1, scope="run"),
            ),
        )
    )

    assert EventType.SESSION_LIMIT_REACHED in [event.type for event in events]
    assert tool.calls == [{"step": 1}]

from __future__ import annotations

# ruff: noqa: E402
import asyncio
from collections.abc import AsyncIterator

import pytest

fastapi = pytest.importorskip("fastapi")
pytest.importorskip("sse_starlette")

from fastapi.testclient import TestClient

from cayu import AgentSpec, CayuApp, InMemoryTaskStore, Message, TaskCreate
from cayu.core.events import Event, EventType
from cayu.providers import ModelProvider, ModelRequest, ModelStreamEvent
from cayu.runtime import (
    EventQuery,
    InMemorySessionStore,
    RunRequest,
    SessionIdentity,
    SessionStatus,
)
from cayu.server import create_server


class OneShotProvider(ModelProvider):
    name = "fake"

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        yield ModelStreamEvent.text_delta("done")
        yield ModelStreamEvent.completed({"finish_reason": "stop"})


class UsageProvider(ModelProvider):
    name = "fake"

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        yield ModelStreamEvent.text_delta("done")
        yield ModelStreamEvent.completed(
            {
                "usage": {
                    "input_tokens": 10,
                    "input_tokens_details": {"cached_tokens": 4},
                    "output_tokens": 2,
                }
            }
        )


async def _collect_run(app: CayuApp, request: RunRequest) -> list[Event]:
    return [event async for event in app.run(request)]


def test_server_uses_app_task_store_for_runs_and_task_list() -> None:
    app = CayuApp(task_store=InMemoryTaskStore())
    app.register_provider(OneShotProvider(), default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    client = TestClient(create_server(app))

    dashboard = client.get("/")
    assert dashboard.status_code == 200
    assert "root" in dashboard.text

    with client.stream("POST", "/api/run", json={"prompt": "hello"}) as response:
        assert response.status_code == 200
        list(response.iter_lines())

    tasks = client.get("/api/tasks").json()
    assert len(tasks) == 1
    assert tasks[0]["type"] == "run"
    assert tasks[0]["status"] == "completed"
    assert tasks[0]["worker_id"] is None
    assert tasks[0]["lease_expires_at"] is None


def test_server_task_list_exposes_worker_lease_state() -> None:
    task_store = InMemoryTaskStore()

    async def setup_task() -> None:
        await task_store.create_task(
            TaskCreate(
                task_id="leased_task",
                type="review",
                assigned_agent_name="assistant",
            )
        )
        claimed = await task_store.claim_task("worker_a", lease_seconds=300)
        assert claimed is not None

    asyncio.run(setup_task())

    client = TestClient(create_server(CayuApp(task_store=task_store)))
    tasks = client.get("/api/tasks").json()

    assert tasks == [
        {
            "id": "leased_task",
            "type": "review",
            "title": None,
            "status": "running",
            "session_id": None,
            "worker_id": "worker_a",
            "lease_expires_at": tasks[0]["lease_expires_at"],
            "created_at": tasks[0]["created_at"],
            "completed_at": None,
        }
    ]
    assert isinstance(tasks[0]["lease_expires_at"], str)


def test_server_exposes_session_usage_summary() -> None:
    app = CayuApp()
    app.register_provider(UsageProvider(), default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))
    asyncio.run(
        _collect_run(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="usage_1",
                messages=[Message.text("user", "hello")],
            ),
        )
    )

    client = TestClient(create_server(app))
    response = client.get("/api/sessions/usage_1/usage")

    assert response.status_code == 200
    assert response.json() == {
        "session_id": "usage_1",
        "model_steps": 1,
        "tool_calls": 0,
        "provider_names": ["fake"],
        "models": ["fake-model"],
        "usage": {
            "provider_name": None,
            "model": None,
            "input_tokens": 10,
            "output_tokens": 2,
            "total_tokens": 12,
            "reasoning_output_tokens": 0,
            "cache": {
                "read_tokens": 0,
                "write_tokens": 0,
                "cached_input_tokens": 4,
                "uncached_input_tokens": 6,
            },
        },
    }


def test_server_run_accepts_budget_limits() -> None:
    app = CayuApp()
    app.register_provider(UsageProvider(), default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))
    client = TestClient(create_server(app))

    with client.stream(
        "POST",
        "/api/run",
        json={
            "prompt": "hello",
            "budget_limits": [
                {
                    "scope": "session",
                    "max_estimated_cost": "0.000001",
                    "pricing": {
                        "prices": [
                            {
                                "provider_name": "fake",
                                "model": "fake-model",
                                "input_per_million": "1",
                                "output_per_million": "1",
                            }
                        ]
                    },
                }
            ],
        },
    ) as response:
        assert response.status_code == 200
        list(response.iter_lines())

    sessions = client.get("/api/sessions").json()
    assert len(sessions) == 1
    assert sessions[0]["status"] == "interrupted"


def test_server_lists_sessions_with_label_filters() -> None:
    store = InMemorySessionStore()
    app = CayuApp(session_store=store)
    client = TestClient(create_server(app))

    async def seed() -> None:
        await store.create(
            RunRequest(
                agent_name="builder",
                session_id="sess_invoice",
                labels={"organization": "org_123", "project": "ap_q2"},
                messages=[Message.text("user", "invoice")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        await store.create(
            RunRequest(
                agent_name="builder",
                session_id="sess_research",
                labels={"organization": "org_123", "project": "research"},
                messages=[Message.text("user", "research")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        await store.create(
            RunRequest(
                agent_name="reviewer",
                session_id="sess_other_org",
                labels={"organization": "org_999", "project": "ap_q2"},
                messages=[Message.text("user", "review")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )

    asyncio.run(seed())

    org_response = client.get("/api/sessions?label=organization=org_123&limit=10")
    exact_response = client.get(
        "/api/sessions?label=organization=org_123&label=project=ap_q2&limit=10"
    )
    missing_response = client.get("/api/sessions?label=organization=missing&limit=10")

    assert org_response.status_code == 200
    assert {session["id"] for session in org_response.json()} == {
        "sess_invoice",
        "sess_research",
    }
    assert exact_response.status_code == 200
    assert [session["id"] for session in exact_response.json()] == ["sess_invoice"]
    assert exact_response.json()[0]["labels"] == {
        "organization": "org_123",
        "project": "ap_q2",
    }
    assert missing_response.status_code == 200
    assert missing_response.json() == []


def test_server_lists_sessions_with_typed_filters() -> None:
    store = InMemorySessionStore()
    app = CayuApp(session_store=store)
    client = TestClient(create_server(app))

    async def seed() -> None:
        await store.create(
            RunRequest(
                agent_name="builder",
                session_id="sess_builder_local",
                environment_name="local",
                messages=[Message.text("user", "build")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        await store.create(
            RunRequest(
                agent_name="builder",
                session_id="sess_builder_prod",
                environment_name="prod",
                causal_budget_id="budget_123",
                messages=[Message.text("user", "build prod")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        await store.create(
            RunRequest(
                agent_name="reviewer",
                session_id="sess_reviewer_prod",
                environment_name="prod",
                messages=[Message.text("user", "review")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        await store.update_status("sess_builder_prod", SessionStatus.COMPLETED)

    asyncio.run(seed())

    builder_response = client.get(
        "/api/sessions?agent_name=builder&order_by=created_at_asc&limit=10"
    )
    completed_response = client.get("/api/sessions?status=completed&limit=10")
    env_response = client.get("/api/sessions?environment_name=prod&agent_name=builder&limit=10")
    causal_response = client.get("/api/sessions?causal_budget_id=budget_123&limit=10")

    assert builder_response.status_code == 200
    assert [session["id"] for session in builder_response.json()] == [
        "sess_builder_local",
        "sess_builder_prod",
    ]
    assert completed_response.status_code == 200
    assert [session["id"] for session in completed_response.json()] == ["sess_builder_prod"]
    assert env_response.status_code == 200
    assert [session["id"] for session in env_response.json()] == ["sess_builder_prod"]
    assert causal_response.status_code == 200
    assert [session["id"] for session in causal_response.json()] == ["sess_builder_prod"]


def test_server_lists_sessions_with_typed_and_label_filters_together() -> None:
    store = InMemorySessionStore()
    app = CayuApp(session_store=store)
    client = TestClient(create_server(app))

    async def seed() -> None:
        await store.create(
            RunRequest(
                agent_name="builder",
                session_id="sess_builder_invoice",
                labels={"organization": "org_123"},
                messages=[Message.text("user", "invoice")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        await store.create(
            RunRequest(
                agent_name="reviewer",
                session_id="sess_reviewer_invoice",
                labels={"organization": "org_123"},
                messages=[Message.text("user", "review")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )

    asyncio.run(seed())

    response = client.get("/api/sessions?agent_name=builder&label=organization=org_123")

    assert response.status_code == 200
    assert [session["id"] for session in response.json()] == ["sess_builder_invoice"]


def test_server_lists_sessions_with_label_selectors() -> None:
    store = InMemorySessionStore()
    app = CayuApp(session_store=store)
    client = TestClient(create_server(app))

    async def seed() -> None:
        await store.create(
            RunRequest(
                agent_name="builder",
                session_id="sess_selector_invoice",
                labels={"organization": "org_123", "project": "ap_q2", "workflow": "invoice"},
                messages=[Message.text("user", "invoice")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        await store.create(
            RunRequest(
                agent_name="builder",
                session_id="sess_selector_research",
                labels={"organization": "org_123", "project": "research"},
                messages=[Message.text("user", "research")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        await store.create(
            RunRequest(
                agent_name="reviewer",
                session_id="sess_selector_unowned",
                labels={"project": "ap_q2"},
                messages=[Message.text("user", "review")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )

    asyncio.run(seed())

    exists_response = client.get(
        "/api/sessions",
        params={"label_selector": "workflow"},
    )
    in_response = client.get(
        "/api/sessions",
        params={"label_selector": "project in (ap_q2,research)"},
    )
    equals_response = client.get(
        "/api/sessions",
        params={"label_selector": "project==ap_q2"},
    )
    not_in_response = client.get(
        "/api/sessions",
        params=[
            ("label", "organization=org_123"),
            ("label_selector", "project notin (research)"),
        ],
    )
    not_exists_response = client.get(
        "/api/sessions",
        params={"label_selector": "!organization"},
    )

    assert exists_response.status_code == 200
    assert [session["id"] for session in exists_response.json()] == ["sess_selector_invoice"]
    assert in_response.status_code == 200
    assert {session["id"] for session in in_response.json()} == {
        "sess_selector_invoice",
        "sess_selector_research",
        "sess_selector_unowned",
    }
    assert equals_response.status_code == 200
    assert {session["id"] for session in equals_response.json()} == {
        "sess_selector_invoice",
        "sess_selector_unowned",
    }
    assert not_in_response.status_code == 200
    assert [session["id"] for session in not_in_response.json()] == ["sess_selector_invoice"]
    assert not_exists_response.status_code == 200
    assert [session["id"] for session in not_exists_response.json()] == ["sess_selector_unowned"]


def test_server_session_label_filters_allow_reserved_query_keys() -> None:
    app = CayuApp()
    client = TestClient(create_server(app))

    response = client.get("/api/sessions?label=cayu:agent=builder")

    assert response.status_code == 200
    assert response.json() == []


def test_server_rejects_invalid_session_label_filters() -> None:
    app = CayuApp()
    client = TestClient(create_server(app))

    assert client.get("/api/sessions?label=missing_separator").status_code == 422
    assert client.get("/api/sessions?label=%20=org_123").status_code == 422
    assert client.get("/api/sessions?label=owner=org_123&label=owner=org_456").status_code == 422
    assert client.get("/api/sessions?agent_name=%20").status_code == 422
    assert client.get("/api/sessions?status=not-a-status").status_code == 422
    assert client.get("/api/sessions?label_selector=project%20in%20ap_q2").status_code == 422


def test_server_exposes_filtered_sessions_summary() -> None:
    app = CayuApp()
    app.register_provider(UsageProvider(), default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))
    for session_id, labels in (
        ("summary_filter_invoice", {"organization": "org_123", "project": "ap_q2"}),
        ("summary_filter_research", {"organization": "org_123", "project": "research"}),
        ("summary_filter_other", {"organization": "org_999", "project": "ap_q2"}),
    ):
        asyncio.run(
            _collect_run(
                app,
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    labels=labels,
                    messages=[Message.text("user", "hello")],
                ),
            )
        )

    client = TestClient(create_server(app))
    response = client.post(
        "/api/sessions/summary",
        params=[
            ("label", "organization=org_123"),
            ("label_selector", "project in (ap_q2,research)"),
            ("order_by", "created_at_asc"),
        ],
        json={
            "pricing": {
                "prices": [
                    {
                        "provider_name": "fake",
                        "model": "fake-model",
                        "input_per_million": "1",
                        "output_per_million": "2",
                        "cache_read_input_per_million": "0.25",
                    }
                ]
            }
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["session_count"] == 2
    assert [item["session"]["id"] for item in body["sessions"]] == [
        "summary_filter_invoice",
        "summary_filter_research",
    ]
    assert body["usage"]["session_count"] == 2
    assert body["usage"]["usage"]["total_tokens"] == 24
    assert body["cost"]["session_count"] == 2
    assert body["cost"]["total_cost"] == "0.000020"
    assert [item["session_id"] for item in body["cost"]["session_costs"]] == [
        "summary_filter_invoice",
        "summary_filter_research",
    ]


def test_server_filtered_sessions_summary_queries_events_in_one_batch() -> None:
    app = CayuApp()
    app.register_provider(UsageProvider(), default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))
    for session_id in ("summary_batch_one", "summary_batch_two"):
        asyncio.run(
            _collect_run(
                app,
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    labels={"organization": "org_123"},
                    messages=[Message.text("user", "hello")],
                ),
            )
        )

    queries: list[EventQuery] = []
    original_query_events = app.session_store.query_events

    async def query_events(query: EventQuery | None = None):
        copied = EventQuery() if query is None else query
        queries.append(copied)
        return await original_query_events(query)

    app.session_store.query_events = query_events

    client = TestClient(create_server(app))
    response = client.post(
        "/api/sessions/summary",
        params={"label": "organization=org_123", "order_by": "created_at_asc"},
    )

    assert response.status_code == 200
    assert response.json()["session_count"] == 2
    assert len(queries) == 1
    assert queries[0].session_ids == ("summary_batch_one", "summary_batch_two")
    assert queries[0].session_id is None


def test_server_sessions_summary_allows_omitted_body() -> None:
    app = CayuApp()
    app.register_provider(UsageProvider(), default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))
    asyncio.run(
        _collect_run(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="summary_no_body",
                labels={"organization": "org_123"},
                messages=[Message.text("user", "hello")],
            ),
        )
    )

    client = TestClient(create_server(app))
    response = client.post(
        "/api/sessions/summary",
        params={"label": "organization=org_123"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["session_count"] == 1
    assert body["sessions"][0]["session"]["id"] == "summary_no_body"
    assert body["usage"]["usage"]["total_tokens"] == 12
    assert body["cost"] is None


def test_server_run_rejects_request_budget_reservations() -> None:
    app = CayuApp()
    app.register_provider(UsageProvider(), default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))
    client = TestClient(create_server(app))

    response = client.post(
        "/api/run",
        json={
            "prompt": "hello",
            "budget_limits": [
                {
                    "scope": "session",
                    "max_estimated_cost": "0.01",
                    "pricing": {
                        "prices": [
                            {
                                "provider_name": "fake",
                                "model": "fake-model",
                                "input_per_million": "1",
                                "output_per_million": "1",
                            }
                        ]
                    },
                    "reservation": {
                        "max_input_tokens": 1,
                        "max_output_tokens": 0,
                    },
                }
            ],
        },
    )

    assert response.status_code == 422
    assert "Request budget limits must not use reservations" in response.text


def test_server_session_usage_returns_404_for_missing_session() -> None:
    app = CayuApp()
    app.register_provider(UsageProvider(), default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    client = TestClient(create_server(app))
    response = client.get("/api/sessions/missing/usage")

    assert response.status_code == 404
    assert response.json() == {"detail": "Session not found"}


def test_server_session_usage_rejects_blank_session_id() -> None:
    app = CayuApp()
    app.register_provider(UsageProvider(), default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    client = TestClient(create_server(app))
    response = client.get("/api/sessions/%20/usage")

    assert response.status_code == 422


def test_server_exposes_session_cost_estimate() -> None:
    app = CayuApp()
    app.register_provider(UsageProvider(), default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))
    asyncio.run(
        _collect_run(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="cost_1",
                messages=[Message.text("user", "hello")],
            ),
        )
    )

    client = TestClient(create_server(app))
    response = client.post(
        "/api/sessions/cost_1/cost",
        json={
            "pricing": {
                "prices": [
                    {
                        "provider_name": "fake",
                        "model": "fake-model",
                        "input_per_million": "1",
                        "output_per_million": "2",
                        "cache_read_input_per_million": "0.25",
                    }
                ]
            }
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "session_id": "cost_1",
        "currency": "USD",
        "model_steps": 1,
        "priced_model_steps": 1,
        "unpriced_model_steps": 0,
        "total_cost": "0.000010",
        "line_items": [
            {
                "model_step": 1,
                "provider_name": "fake",
                "model": "fake-model",
                "pricing_provider_name": "fake",
                "pricing_model": "fake-model",
                "pricing_match": "exact",
                "priced": True,
                "currency": "USD",
                "input_tokens": 10,
                "output_tokens": 2,
                "cache_read_input_tokens": 0,
                "cache_write_input_tokens": 0,
                "uncached_input_tokens": 6,
                "input_cost": "0.000006",
                "output_cost": "0.000004",
                "cache_read_input_cost": "0.00",
                "cache_write_input_cost": "0",
                "total_cost": "0.000010",
                "missing_pricing_reason": None,
            }
        ],
    }


def test_server_exposes_causal_budget_usage_and_cost() -> None:
    app = CayuApp()
    app.register_provider(UsageProvider(), default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))
    for session_id in ("causal_parent", "causal_child"):
        asyncio.run(
            _collect_run(
                app,
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    causal_budget_id="job_shared",
                    messages=[Message.text("user", "hello")],
                ),
            )
        )

    client = TestClient(create_server(app))
    usage_response = client.get("/api/causal-budgets/job_shared/usage")
    pricing_body = {
        "pricing": {
            "prices": [
                {
                    "provider_name": "fake",
                    "model": "fake-model",
                    "input_per_million": "1",
                    "output_per_million": "2",
                    "cache_read_input_per_million": "0.25",
                }
            ]
        },
    }
    cost_response = client.post(
        "/api/causal-budgets/job_shared/cost",
        json=pricing_body,
    )

    async def unexpected_app_summary_call(*args, **kwargs):
        raise AssertionError("causal summary route must use one session snapshot")

    app.get_causal_budget_usage = unexpected_app_summary_call
    app.get_causal_budget_cost = unexpected_app_summary_call

    summary_response = client.post(
        "/api/causal-budgets/job_shared/summary",
        json=pricing_body,
    )

    assert usage_response.status_code == 200
    assert usage_response.json() == {
        "causal_budget_id": "job_shared",
        "session_ids": ["causal_parent", "causal_child"],
        "session_count": 2,
        "model_steps": 2,
        "tool_calls": 0,
        "provider_names": ["fake"],
        "models": ["fake-model"],
        "usage": {
            "provider_name": None,
            "model": None,
            "input_tokens": 20,
            "output_tokens": 4,
            "total_tokens": 24,
            "reasoning_output_tokens": 0,
            "cache": {
                "read_tokens": 0,
                "write_tokens": 0,
                "cached_input_tokens": 8,
                "uncached_input_tokens": 12,
            },
        },
        "session_summaries": [
            {
                "session_id": "causal_parent",
                "model_steps": 1,
                "tool_calls": 0,
                "provider_names": ["fake"],
                "models": ["fake-model"],
                "usage": {
                    "provider_name": None,
                    "model": None,
                    "input_tokens": 10,
                    "output_tokens": 2,
                    "total_tokens": 12,
                    "reasoning_output_tokens": 0,
                    "cache": {
                        "read_tokens": 0,
                        "write_tokens": 0,
                        "cached_input_tokens": 4,
                        "uncached_input_tokens": 6,
                    },
                },
            },
            {
                "session_id": "causal_child",
                "model_steps": 1,
                "tool_calls": 0,
                "provider_names": ["fake"],
                "models": ["fake-model"],
                "usage": {
                    "provider_name": None,
                    "model": None,
                    "input_tokens": 10,
                    "output_tokens": 2,
                    "total_tokens": 12,
                    "reasoning_output_tokens": 0,
                    "cache": {
                        "read_tokens": 0,
                        "write_tokens": 0,
                        "cached_input_tokens": 4,
                        "uncached_input_tokens": 6,
                    },
                },
            },
        ],
    }
    assert cost_response.status_code == 200
    assert cost_response.json()["causal_budget_id"] == "job_shared"
    assert cost_response.json()["session_ids"] == ["causal_parent", "causal_child"]
    assert cost_response.json()["session_count"] == 2
    assert cost_response.json()["model_steps"] == 2
    assert cost_response.json()["total_cost"] == "0.000020"
    assert [item["session_id"] for item in cost_response.json()["session_costs"]] == [
        "causal_parent",
        "causal_child",
    ]
    assert summary_response.status_code == 200
    summary_body = summary_response.json()
    assert summary_body["causal_budget_id"] == "job_shared"
    assert summary_body["session_count"] == 2
    assert [item["session"]["id"] for item in summary_body["sessions"]] == [
        "causal_parent",
        "causal_child",
    ]
    assert [item["outcome"]["reason"] for item in summary_body["sessions"]] == [
        "completed",
        "completed",
    ]
    for item in summary_body["sessions"]:
        assert item["events"]["total_events"] > 0
        assert item["events"]["counts_by_type"]["model.completed"] == 1
        assert item["events"]["counts_by_type"]["session.completed"] == 1
        assert item["events"]["latest_event"]["type"] == "session.completed"
    assert summary_body["usage"]["usage"]["total_tokens"] == 24
    assert summary_body["cost"]["total_cost"] == "0.000020"

    missing_summary_response = client.post(
        "/api/causal-budgets/missing/summary",
        json=pricing_body,
    )
    assert missing_summary_response.status_code == 404
    assert missing_summary_response.json() == {"detail": "Causal budget not found"}


def test_server_session_cost_reports_unpriced_steps() -> None:
    app = CayuApp()
    app.register_provider(UsageProvider(), default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))
    asyncio.run(
        _collect_run(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="cost_unpriced",
                messages=[Message.text("user", "hello")],
            ),
        )
    )

    client = TestClient(create_server(app))
    response = client.post(
        "/api/sessions/cost_unpriced/cost",
        json={
            "pricing": {
                "prices": [
                    {
                        "provider_name": "other-provider",
                        "model": "other-model",
                        "input_per_million": "1",
                        "output_per_million": "1",
                    }
                ]
            }
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["total_cost"] == "0"
    assert body["priced_model_steps"] == 0
    assert body["unpriced_model_steps"] == 1
    assert body["line_items"][0]["missing_pricing_reason"] == "no matching model pricing"


def test_server_session_cost_returns_404_for_missing_session() -> None:
    app = CayuApp()
    app.register_provider(UsageProvider(), default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    client = TestClient(create_server(app))
    response = client.post(
        "/api/sessions/missing/cost",
        json={
            "pricing": {
                "prices": [
                    {
                        "provider_name": "fake",
                        "model": "fake-model",
                        "input_per_million": "1",
                        "output_per_million": "1",
                    }
                ]
            }
        },
    )

    assert response.status_code == 404
    assert response.json() == {"detail": "Session not found"}


def test_server_session_cost_validates_pricing_body() -> None:
    app = CayuApp()
    app.register_provider(UsageProvider(), default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    client = TestClient(create_server(app))
    response = client.post(
        "/api/sessions/session_1/cost",
        json={
            "pricing": {
                "prices": [
                    {
                        "provider_name": "fake",
                        "model": "fake-model",
                        "input_per_million": "-1",
                        "output_per_million": "1",
                    }
                ]
            }
        },
    )

    assert response.status_code == 422


def test_server_exposes_session_summary() -> None:
    app = CayuApp()
    app.register_provider(UsageProvider(), default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))
    asyncio.run(
        _collect_run(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="summary_1",
                messages=[Message.text("user", "hello")],
            ),
        )
    )

    client = TestClient(create_server(app))
    response = client.get("/api/sessions/summary_1/summary")

    assert response.status_code == 200
    body = response.json()
    assert body["session"]["id"] == "summary_1"
    assert body["session"]["status"] == "completed"
    assert body["session"]["agent_name"] == "assistant"
    assert body["session"]["provider_name"] == "fake"
    assert body["session"]["model"] == "fake-model"
    assert body["session"]["environment_name"] is None
    assert body["events"]["total_events"] == 5
    assert body["events"]["counts_by_type"] == {
        "model.completed": 1,
        "model.started": 1,
        "model.text.delta": 1,
        "session.completed": 1,
        "session.started": 1,
    }
    assert body["events"]["latest_event"]["type"] == "session.completed"
    assert body["transcript"] == {"total_messages": 2}
    assert body["outcome"]["session_id"] == "summary_1"
    assert body["outcome"]["status"] == "completed"
    assert body["outcome"]["reason"] == "completed"
    assert body["outcome"]["details"] == {}
    assert body["outcome"]["retry"] is None
    assert body["outcome"]["terminal_event"]["type"] == "session.completed"
    assert body["outcome"]["latest_retry_event"] is None
    assert body["usage"] == {
        "session_id": "summary_1",
        "model_steps": 1,
        "tool_calls": 0,
        "provider_names": ["fake"],
        "models": ["fake-model"],
        "usage": {
            "provider_name": None,
            "model": None,
            "input_tokens": 10,
            "output_tokens": 2,
            "total_tokens": 12,
            "reasoning_output_tokens": 0,
            "cache": {
                "read_tokens": 0,
                "write_tokens": 0,
                "cached_input_tokens": 4,
                "uncached_input_tokens": 6,
            },
        },
    }


def test_server_session_summary_exposes_interrupted_outcome_and_retry() -> None:
    app = CayuApp()

    async def seed() -> None:
        await app.session_store.create(
            RunRequest(
                agent_name="assistant",
                session_id="summary_interrupted",
                messages=[Message.text("user", "hello")],
            ),
            identity=SessionIdentity(
                provider_name="fake",
                model="fake-model",
                runtime_name="cayu",
                runtime_version=None,
            ),
        )
        await app.session_store.update_status(
            "summary_interrupted",
            SessionStatus.INTERRUPTED,
        )
        await app.session_store.append_events(
            "summary_interrupted",
            [
                Event(
                    id="summary_retry",
                    type=EventType.MODEL_RETRY,
                    session_id="summary_interrupted",
                    payload={
                        "provider": "fake",
                        "model": "fake-model",
                        "step": 1,
                        "attempt": 1,
                        "next_attempt": 2,
                        "max_attempts": 2,
                        "reason": "timeout",
                        "delay_seconds": 0.0,
                        "error": "stream idle timeout",
                    },
                ),
                Event(
                    id="summary_terminal",
                    type=EventType.SESSION_INTERRUPTED,
                    session_id="summary_interrupted",
                    payload={
                        "interruption_type": "limit_reached",
                        "limit": "total_tokens",
                        "actual": 12,
                        "maximum": 10,
                        "message": "Run limit reached.",
                    },
                ),
                Event(
                    id="summary_hook",
                    type=EventType.HOOK_COMPLETED,
                    session_id="summary_interrupted",
                    payload={"hook": "after_session_interrupted"},
                ),
            ],
        )

    asyncio.run(seed())

    client = TestClient(create_server(app))
    response = client.get("/api/sessions/summary_interrupted/summary")

    assert response.status_code == 200
    body = response.json()
    outcome = body["outcome"]
    assert outcome["status"] == "interrupted"
    assert outcome["reason"] == "limit_reached"
    assert outcome["details"] == {
        "interruption_type": "limit_reached",
        "limit": "total_tokens",
        "maximum": 10,
        "actual": 12,
        "message": "Run limit reached.",
    }
    assert outcome["retry"] == {
        "provider": "fake",
        "model": "fake-model",
        "step": 1,
        "attempt": 1,
        "next_attempt": 2,
        "max_attempts": 2,
        "delay_seconds": 0.0,
        "reason": "timeout",
    }
    assert outcome["terminal_event"]["id"] == "summary_terminal"
    assert outcome["latest_retry_event"]["id"] == "summary_retry"
    assert body["events"]["latest_event"]["id"] == "summary_hook"


def test_server_session_summary_returns_404_for_missing_session() -> None:
    app = CayuApp()
    app.register_provider(UsageProvider(), default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    client = TestClient(create_server(app))
    response = client.get("/api/sessions/missing/summary")

    assert response.status_code == 404
    assert response.json() == {"detail": "Session not found"}


def test_server_session_summary_rejects_blank_session_id() -> None:
    app = CayuApp()
    app.register_provider(UsageProvider(), default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    client = TestClient(create_server(app))
    response = client.get("/api/sessions/%20/summary")

    assert response.status_code == 422


def test_server_exposes_paginated_session_events() -> None:
    app = CayuApp()
    app.register_provider(OneShotProvider(), default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    async def seed_events() -> None:
        await app.session_store.create(
            RunRequest(
                agent_name="assistant",
                session_id="events_1",
                messages=[Message.text("user", "hello")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        await app.session_store.append_events(
            "events_1",
            [
                Event(
                    id="event_1",
                    type=EventType.SESSION_STARTED,
                    session_id="events_1",
                    agent_name="assistant",
                ),
                Event(
                    id="event_2",
                    type=EventType.TOOL_CALL_COMPLETED,
                    session_id="events_1",
                    agent_name="assistant",
                    tool_name="read_file",
                    payload={"path": "notes/result.txt"},
                ),
                Event(
                    id="event_3",
                    type=EventType.MODEL_COMPLETED,
                    session_id="events_1",
                    agent_name="assistant",
                    payload={"finish_reason": "stop"},
                ),
            ],
        )

    asyncio.run(seed_events())

    client = TestClient(create_server(app))

    first_page = client.get("/api/sessions/events_1/events?limit=2")
    assert first_page.status_code == 200
    first_body = first_page.json()
    assert first_body["session_id"] == "events_1"
    assert first_body["has_more"] is True
    assert first_body["next_sequence"] == 2
    assert [event["id"] for event in first_body["events"]] == ["event_1", "event_2"]
    assert first_body["events"][1] == {
        "sequence": 2,
        "id": "event_2",
        "type": "tool.call.completed",
        "session_id": "events_1",
        "agent_name": "assistant",
        "environment_name": None,
        "workflow_name": None,
        "tool_name": "read_file",
        "payload": {"path": "notes/result.txt"},
        "timestamp": first_body["events"][1]["timestamp"],
    }

    second_page = client.get("/api/sessions/events_1/events?after_sequence=2&limit=2")
    assert second_page.status_code == 200
    second_body = second_page.json()
    assert second_body["has_more"] is False
    assert second_body["next_sequence"] == 3
    assert [event["id"] for event in second_body["events"]] == ["event_3"]


def test_server_filters_session_events() -> None:
    app = CayuApp()
    app.register_provider(OneShotProvider(), default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    async def seed_events() -> None:
        await app.session_store.create(
            RunRequest(
                agent_name="assistant",
                session_id="events_filters",
                environment_name="local",
                messages=[Message.text("user", "hello")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        await app.session_store.append_events(
            "events_filters",
            [
                Event(
                    id="event_filter_1",
                    type=EventType.TOOL_CALL_COMPLETED,
                    session_id="events_filters",
                    agent_name="assistant",
                    environment_name="local",
                    tool_name="read_file",
                ),
                Event(
                    id="event_filter_2",
                    type=EventType.TOOL_CALL_COMPLETED,
                    session_id="events_filters",
                    agent_name="assistant",
                    environment_name="local",
                    tool_name="write_file",
                ),
                Event(
                    id="event_filter_3",
                    type=EventType.MODEL_COMPLETED,
                    session_id="events_filters",
                    agent_name="assistant",
                    environment_name="local",
                ),
            ],
        )

    asyncio.run(seed_events())

    client = TestClient(create_server(app))
    response = client.get(
        "/api/sessions/events_filters/events",
        params={
            "event_type": "tool.call.completed",
            "tool_name": "read_file",
            "environment_name": "local",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["has_more"] is False
    assert body["next_sequence"] == 1
    assert [event["id"] for event in body["events"]] == ["event_filter_1"]


def test_server_session_events_returns_404_for_missing_session() -> None:
    app = CayuApp()
    app.register_provider(OneShotProvider(), default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    client = TestClient(create_server(app))
    response = client.get("/api/sessions/missing/events")

    assert response.status_code == 404
    assert response.json() == {"detail": "Session not found"}


def test_server_session_events_validates_query() -> None:
    app = CayuApp()
    app.register_provider(OneShotProvider(), default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    async def create_session() -> None:
        await app.session_store.create(
            RunRequest(
                agent_name="assistant",
                session_id="events_validation",
                messages=[Message.text("user", "hello")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )

    asyncio.run(create_session())

    client = TestClient(create_server(app))

    assert client.get("/api/sessions/events_validation/events?limit=0").status_code == 422
    assert (
        client.get("/api/sessions/events_validation/events?event_type=not.valid").status_code == 422
    )
    assert client.get("/api/sessions/%20/events").status_code == 422


def test_server_exposes_paginated_session_transcript() -> None:
    app = CayuApp()
    app.register_provider(OneShotProvider(), default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    async def seed_transcript() -> None:
        await app.session_store.create(
            RunRequest(
                agent_name="assistant",
                session_id="transcript_1",
                messages=[Message.text("user", "hello")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        await app.session_store.append_transcript_messages(
            "transcript_1",
            [
                Message.text("user", "hello"),
                Message.tool_call(
                    tool_call_id="call_1",
                    tool_name="read_file",
                    arguments={"path": "notes/result.txt"},
                ),
                Message.tool_result(
                    tool_call_id="call_1",
                    tool_name="read_file",
                    content="file contents",
                ),
                Message.text("assistant", "done"),
            ],
        )

    asyncio.run(seed_transcript())

    client = TestClient(create_server(app))
    first_page = client.get("/api/sessions/transcript_1/transcript?limit=2")

    assert first_page.status_code == 200
    first_body = first_page.json()
    assert first_body["session_id"] == "transcript_1"
    assert first_body["offset"] == 0
    assert first_body["next_offset"] == 2
    assert first_body["has_more"] is True
    assert first_body["total_messages"] == 4
    assert [message["index"] for message in first_body["messages"]] == [0, 1]
    assert [message["role"] for message in first_body["messages"]] == ["user", "assistant"]
    assert first_body["messages"][1]["content"] == [
        {
            "type": "tool_call",
            "tool_call_id": "call_1",
            "tool_name": "read_file",
            "arguments": {"path": "notes/result.txt"},
        }
    ]

    second_page = client.get("/api/sessions/transcript_1/transcript?offset=2&limit=2")
    assert second_page.status_code == 200
    second_body = second_page.json()
    assert second_body["next_offset"] == 4
    assert second_body["has_more"] is False
    assert [message["role"] for message in second_body["messages"]] == ["tool", "assistant"]


def test_server_filters_session_transcript_by_role() -> None:
    app = CayuApp()
    app.register_provider(OneShotProvider(), default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    async def seed_transcript() -> None:
        await app.session_store.create(
            RunRequest(
                agent_name="assistant",
                session_id="transcript_roles",
                messages=[Message.text("user", "hello")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        await app.session_store.append_transcript_messages(
            "transcript_roles",
            [
                Message.text("user", "first"),
                Message.text("assistant", "reply"),
                Message.text("user", "second"),
            ],
        )

    asyncio.run(seed_transcript())

    client = TestClient(create_server(app))
    response = client.get("/api/sessions/transcript_roles/transcript?role=user")

    assert response.status_code == 200
    body = response.json()
    assert body["total_messages"] == 2
    assert body["has_more"] is False
    assert [message["index"] for message in body["messages"]] == [0, 2]
    assert [message["content"][0]["text"] for message in body["messages"]] == [
        "first",
        "second",
    ]


def test_server_session_transcript_returns_404_for_missing_session() -> None:
    app = CayuApp()
    app.register_provider(OneShotProvider(), default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    client = TestClient(create_server(app))
    response = client.get("/api/sessions/missing/transcript")

    assert response.status_code == 404
    assert response.json() == {"detail": "Session not found"}


def test_server_session_transcript_validates_query() -> None:
    app = CayuApp()
    app.register_provider(OneShotProvider(), default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    async def create_session() -> None:
        await app.session_store.create(
            RunRequest(
                agent_name="assistant",
                session_id="transcript_validation",
                messages=[Message.text("user", "hello")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )

    asyncio.run(create_session())

    client = TestClient(create_server(app))

    assert client.get("/api/sessions/transcript_validation/transcript?limit=0").status_code == 422
    assert (
        client.get("/api/sessions/transcript_validation/transcript?role=invalid").status_code == 422
    )
    assert client.get("/api/sessions/%20/transcript").status_code == 422


def test_dashboard_routes_fall_back_to_index_without_masking_api_or_assets() -> None:
    app = CayuApp()
    app.register_provider(OneShotProvider(), default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    client = TestClient(create_server(app))

    for path in ["/sessions", "/run", "/sessions/session-abc"]:
        response = client.get(path)
        assert response.status_code == 200
        assert '<div id="root"></div>' in response.text

    assert client.get("/api/missing").status_code == 404
    assert client.get("/assets/missing.js").status_code == 404


def test_run_rejects_blank_prompt_and_agent_before_runtime() -> None:
    app = CayuApp()
    app.register_provider(OneShotProvider(), default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    client = TestClient(create_server(app))

    assert client.post("/api/run", json={"prompt": " "}).status_code == 422
    assert client.post("/api/run", json={"prompt": "hello", "agent": " "}).status_code == 422
    assert (
        client.post("/api/resume", json={"session_id": " ", "prompt": "hello"}).status_code == 422
    )
    assert (
        client.post(
            "/api/tool-approvals/resolve",
            json={
                "session_id": " ",
                "approval_id": "approval_1",
                "decision": "approve",
            },
        ).status_code
        == 422
    )
    assert (
        client.post(
            "/api/tool-approvals/recover",
            json={
                "session_id": " ",
                "approval_id": "approval_1",
                "tool_call_id": "call_1",
                "outcome": "completed",
                "message": "confirmed externally",
            },
        ).status_code
        == 422
    )
    assert (
        client.post(
            "/api/tool-approvals/recover",
            json={
                "session_id": "session_1",
                "approval_id": "approval_1",
                "tool_call_id": " ",
                "outcome": "completed",
                "message": "confirmed externally",
            },
        ).status_code
        == 422
    )
    assert (
        client.post(
            "/api/tool-approvals/recover",
            json={
                "session_id": "session_1",
                "approval_id": "approval_1",
                "tool_call_id": "call_1",
                "outcome": "completed",
                "message": " ",
            },
        ).status_code
        == 422
    )
    assert (
        client.post(
            "/api/tool-approvals/resolve",
            json={
                "session_id": "session_1",
                "approval_id": "approval_1",
                "decision": "maybe",
            },
        ).status_code
        == 422
    )


def test_run_endpoint_passes_retry_policy_to_runtime() -> None:
    app = CayuApp()
    app.register_provider(OneShotProvider(), default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))
    captured_requests = []

    async def run(request):
        captured_requests.append(request)
        yield Event(
            type=EventType.SESSION_STARTED,
            session_id=request.session_id,
            agent_name=request.agent_name,
        )

    app.run = run
    client = TestClient(create_server(app))

    with client.stream(
        "POST",
        "/api/run",
        json={
            "prompt": "hello",
            "retry_policy": {
                "max_attempts": 2,
                "initial_delay_s": 0,
                "retry_on_status_codes": [429],
            },
        },
    ) as response:
        assert response.status_code == 200
        list(response.iter_lines())

    assert len(captured_requests) == 1
    retry_policy = captured_requests[0].retry_policy
    assert retry_policy is not None
    assert retry_policy.max_attempts == 2
    assert retry_policy.initial_delay_s == 0.0
    assert retry_policy.retry_on_status_codes == (429,)


def test_tool_approval_endpoints_preserve_metadata() -> None:
    app = CayuApp()
    app.register_provider(OneShotProvider(), default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    async def create_interrupted_session(session_id: str) -> None:
        await app.session_store.create(
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "hello")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        await app.session_store.update_status(session_id, SessionStatus.INTERRUPTED)

    asyncio.run(create_interrupted_session("session_resolve_metadata"))
    asyncio.run(create_interrupted_session("session_recover_metadata"))

    resolved_requests = []
    recovered_requests = []

    async def resolve_tool_approval(request):
        resolved_requests.append(request)
        if False:
            yield None

    async def recover_tool_approval(request):
        recovered_requests.append(request)
        if False:
            yield None

    app.resolve_tool_approval = resolve_tool_approval
    app.recover_tool_approval = recover_tool_approval
    client = TestClient(create_server(app))

    with client.stream(
        "POST",
        "/api/tool-approvals/resolve",
        json={
            "session_id": "session_resolve_metadata",
            "approval_id": "approval_1",
            "decision": "approve",
            "metadata": {"actor": "operator"},
        },
    ) as response:
        assert response.status_code == 200
        list(response.iter_lines())

    with client.stream(
        "POST",
        "/api/tool-approvals/recover",
        json={
            "session_id": "session_recover_metadata",
            "approval_id": "approval_2",
            "tool_call_id": "call_1",
            "outcome": "completed",
            "message": "confirmed externally",
            "metadata": {"actor": "operator", "source": "dashboard"},
        },
    ) as response:
        assert response.status_code == 200
        list(response.iter_lines())

    assert resolved_requests[0].metadata == {"actor": "operator"}
    assert recovered_requests[0].metadata == {
        "actor": "operator",
        "source": "dashboard",
    }


def test_interrupt_session_endpoint_streams_interrupted_event() -> None:
    app = CayuApp()
    app.register_provider(OneShotProvider(), default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    async def create_pending_session() -> None:
        await app.session_store.create(
            RunRequest(
                agent_name="assistant",
                session_id="session_interrupt_endpoint",
                messages=[Message.text("user", "hello")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )

    asyncio.run(create_pending_session())
    client = TestClient(create_server(app))

    with client.stream(
        "POST",
        "/api/sessions/session_interrupt_endpoint/interrupt",
        json={"reason": "operator requested stop", "metadata": {"actor": "operator"}},
    ) as response:
        assert response.status_code == 200
        lines = list(response.iter_lines())

    body = "\n".join(lines)
    assert "session.interrupted" in body
    assert "operator requested stop" in body

    session = asyncio.run(app.session_store.load("session_interrupt_endpoint"))
    assert session is not None
    assert session.status == SessionStatus.INTERRUPTED


def test_interrupt_session_endpoint_rejects_completed_session_before_streaming() -> None:
    app = CayuApp()
    app.register_provider(OneShotProvider(), default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    async def create_completed_session() -> None:
        await app.session_store.create(
            RunRequest(
                agent_name="assistant",
                session_id="session_interrupt_completed",
                messages=[Message.text("user", "hello")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        await app.session_store.update_status(
            "session_interrupt_completed", SessionStatus.COMPLETED
        )

    asyncio.run(create_completed_session())
    client = TestClient(create_server(app))

    response = client.post("/api/sessions/session_interrupt_completed/interrupt")

    assert response.status_code == 409
    assert response.json() == {
        "detail": "Session cannot be interrupted from status: completed",
    }


def test_interrupt_session_endpoint_rejects_completion_race_before_streaming() -> None:
    class CompletingRaceStore(InMemorySessionStore):
        def __init__(self) -> None:
            super().__init__()
            self.loads = 0

        async def load(self, session_id: str):
            self.loads += 1
            if session_id == "session_interrupt_race" and self.loads == 2:
                await self.update_status(session_id, SessionStatus.COMPLETED)
            return await super().load(session_id)

    store = CompletingRaceStore()
    app = CayuApp(session_store=store)
    app.register_provider(OneShotProvider(), default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    async def create_running_session() -> None:
        await app.session_store.create(
            RunRequest(
                agent_name="assistant",
                session_id="session_interrupt_race",
                messages=[Message.text("user", "hello")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        await app.session_store.update_status("session_interrupt_race", SessionStatus.RUNNING)

    asyncio.run(create_running_session())
    client = TestClient(create_server(app))

    response = client.post("/api/sessions/session_interrupt_race/interrupt")

    assert response.status_code == 409
    assert response.json()["detail"] == "Session cannot be interrupted from status: completed"


def test_interrupt_session_endpoint_returns_conflict_while_interruption_finalizes() -> None:
    app = CayuApp()
    app.register_provider(OneShotProvider(), default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    async def create_interrupting_session() -> None:
        await app.session_store.create(
            RunRequest(
                agent_name="assistant",
                session_id="session_interrupt_finalizing",
                messages=[Message.text("user", "hello")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        await app.session_store.update_status(
            "session_interrupt_finalizing",
            SessionStatus.INTERRUPTING,
        )

    asyncio.run(create_interrupting_session())
    client = TestClient(create_server(app))

    response = client.post("/api/sessions/session_interrupt_finalizing/interrupt")

    assert response.status_code == 409
    assert response.json() == {
        "detail": "Session interruption is still finalizing: session_interrupt_finalizing",
    }


def test_interrupt_session_endpoint_is_idempotent_for_interrupted_session() -> None:
    app = CayuApp()
    app.register_provider(OneShotProvider(), default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    async def create_interrupted_session() -> None:
        await app.session_store.create(
            RunRequest(
                agent_name="assistant",
                session_id="session_interrupt_idempotent",
                messages=[Message.text("user", "hello")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        await app.session_store.update_status(
            "session_interrupt_idempotent", SessionStatus.INTERRUPTED
        )
        await app.session_store.append_event(
            "session_interrupt_idempotent",
            Event(
                type=EventType.SESSION_INTERRUPTED,
                session_id="session_interrupt_idempotent",
                agent_name="assistant",
                payload={"reason": "already interrupted", "metadata": {}},
            ),
        )

    asyncio.run(create_interrupted_session())
    client = TestClient(create_server(app))

    with client.stream(
        "POST",
        "/api/sessions/session_interrupt_idempotent/interrupt",
    ) as response:
        assert response.status_code == 200
        lines = list(response.iter_lines())

    body = "\n".join(lines)
    assert "session.interrupted" in body
    assert "already interrupted" in body

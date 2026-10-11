"""HTTP server usage cost contracts."""

from __future__ import annotations

# ruff: noqa: E402
import pytest

fastapi = pytest.importorskip("fastapi")
pytest.importorskip("sse_starlette")
httpx = pytest.importorskip("httpx")

import asyncio
from collections.abc import AsyncIterator
from typing import Any

from fastapi.testclient import TestClient
from tests.server.server_test_support import (
    _LOCAL_SERVER_CONFIG,
    UsageProvider,
    _aggregate_usage_json,
    _collect_run,
    _price_book_payload,
    _sse_events,
)

from cayu.agents import AgentSpec
from cayu.applications import CayuApp
from cayu.budgets.billing import BillingIdentity
from cayu.budgets.pricing import default_price_book
from cayu.events import EventType
from cayu.messages import Message
from cayu.providers.base import ModelProvider, ModelRequest, ModelStreamEvent
from cayu.providers.bedrock import bedrock_billing_identity, completed_bedrock_billing_identity
from cayu.server import DashboardConfig, ServerConfig, create_server
from cayu.sessions.requests import RunRequest
from cayu.vaults.redaction import SecretRedactor


class BedrockUsageProvider(ModelProvider):
    name = "bedrock"
    identity = bedrock_billing_identity(
        invoked_model="global.anthropic.claude-sonnet-4-6",
        source_region="us-east-1",
        resource_type="inference_profile",
        profile_scope="global",
    )

    async def billing_identity_for_request(
        self,
        request: ModelRequest,
    ) -> BillingIdentity:
        assert request.model == self.identity.resource_id
        return self.identity

    def billing_identity_for_completion(
        self,
        identity: BillingIdentity | None,
        payload: dict[str, Any],
    ) -> BillingIdentity | None:
        assert identity == self.identity
        return completed_bedrock_billing_identity(
            self.identity,
            effective_service_tier=payload["bedrock_service_tier"],
        )

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        yield ModelStreamEvent.completed(
            {
                "model": request.model,
                "usage": {"input_tokens": 1_000_000, "output_tokens": 1_000_000},
                "bedrock_service_tier": "default",
            }
        )


def test_server_dashboard_accepts_default_price_book_config() -> None:
    app = CayuApp()
    price_book = default_price_book()
    client = TestClient(
        create_server(
            app,
            config=ServerConfig.local_development(
                dashboard=DashboardConfig(runtime_config={"priceBook": price_book})
            ),
        )
    )

    dashboard = client.get("/cayu/")

    assert dashboard.status_code == 200
    assert f'"price_book_version":"{price_book.price_book_version}"' in dashboard.text
    assert '"prices":[' in dashboard.text


def test_server_session_usage_returns_404_for_missing_session() -> None:
    app = CayuApp()
    app.register_provider(UsageProvider(), default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    client = TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG))
    response = client.get("/api/sessions/missing/usage")

    assert response.status_code == 404
    assert response.json() == {"detail": "Session not found"}


def test_server_session_usage_rejects_blank_session_id() -> None:
    app = CayuApp()
    app.register_provider(UsageProvider(), default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    client = TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG))
    response = client.get("/api/sessions/%20/usage")

    assert response.status_code == 422


def test_server_exposes_session_cost_estimate() -> None:
    app = CayuApp()
    app.register_provider(UsageProvider(), default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))
    run_events = asyncio.run(
        _collect_run(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="cost_1",
                messages=[Message.text("user", "hello")],
            ),
        )
    )

    completed = next(event for event in run_events if event.type is EventType.MODEL_COMPLETED)
    client = TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG))
    response = client.post(
        "/api/sessions/cost_1/cost",
        json={
            "pricing": _price_book_payload(
                output_per_million="2",
                cache_read_input_per_million="0.25",
            )
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "session_id": "cost_1",
        "auxiliary_attempts": 0,
        "unpriced_auxiliary_attempts": 0,
        "currency": "USD",
        "model_steps": 1,
        "priced_model_steps": 1,
        "unpriced_model_steps": 0,
        "missing_usage_model_steps": 0,
        "missing_pricing_model_steps": 0,
        "unsupported_pricing_model_steps": 0,
        "total_cost": "0.000011",
        "line_items": [
            {
                "model_step": 1,
                "auxiliary_attempt": False,
                "execution_profile_fingerprint": completed.payload["execution_profile_fingerprint"],
                "provider_name": "fake",
                "model": "fake-model",
                "requested_model": "fake-model",
                "pricing_provider_name": "fake",
                "pricing_model": "fake-model",
                "pricing_match": "prefix",
                "pricing_provenance": {
                    "source": "official",
                    "url": "https://example.com/pricing",
                    "as_of": "2026-07-13",
                },
                "pricing_effective_from": None,
                "pricing_effective_through": None,
                "pricing_tier_max_input_tokens": None,
                "priced": True,
                "currency": "USD",
                "input_tokens": 10,
                "output_tokens": 2,
                "cache_read_input_tokens": 4,
                "cache_write_input_tokens": 0,
                "web_search_calls": 0,
                "web_search_outcome_unknown": 0,
                "uncached_input_tokens": 6,
                "input_cost": "0.000006",
                "output_cost": "0.000004",
                "cache_read_input_cost": "0.000001",
                "cache_write_input_cost": "0",
                "web_search_cost": "0",
                "total_cost": "0.000011",
                "missing_pricing_reason": None,
                "unpriced_reason": None,
            }
        ],
    }


def test_server_preserves_priced_and_unpriced_bedrock_identity() -> None:
    app = CayuApp()
    provider = BedrockUsageProvider()
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model=provider.identity.resource_id))
    asyncio.run(
        _collect_run(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="bedrock_cost_1",
                messages=[Message.text("user", "hello")],
            ),
        )
    )
    pricing = _price_book_payload(
        provider_name="bedrock",
        model=provider.identity.resource_id,
        input_per_million="3",
        output_per_million="15",
    )
    price = pricing["prices"][0]
    assert isinstance(price, dict)
    price.update(
        {
            "match": "exact",
            "pricing_context": {
                "dimensions": {
                    "source_region": ["us-east-1"],
                    "service_tier": ["default"],
                }
            },
        }
    )

    client = TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG))
    priced = client.post(
        "/api/sessions/bedrock_cost_1/cost",
        json={"pricing": pricing},
    ).json()
    pricing_context = price["pricing_context"]
    assert isinstance(pricing_context, dict)
    dimensions = pricing_context["dimensions"]
    assert isinstance(dimensions, dict)
    dimensions["source_region"] = ["us-west-2"]
    unpriced = client.post(
        "/api/sessions/bedrock_cost_1/cost",
        json={"pricing": pricing},
    ).json()

    completed_identity = completed_bedrock_billing_identity(
        provider.identity,
        effective_service_tier="default",
    ).model_dump(mode="json")
    assert priced["total_cost"] == "18"
    assert priced["line_items"][0]["billing_identity"] == completed_identity
    assert priced["line_items"][0]["pricing_model"] == provider.identity.resource_id
    assert unpriced["total_cost"] == "0"
    assert unpriced["unpriced_model_steps"] == 1
    assert unpriced["line_items"][0]["billing_identity"] == completed_identity
    assert unpriced["line_items"][0]["missing_pricing_reason"] == "no matching model pricing"


def test_server_cost_accepts_tiered_price_book() -> None:
    app = CayuApp()
    app.register_provider(UsageProvider(), default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))
    asyncio.run(
        _collect_run(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="tiered_cost",
                messages=[Message.text("user", "hello")],
            ),
        )
    )
    price_book = {
        "price_book_version": "test",
        "generated_at": "2026-07-13",
        "prices": [
            {
                "provider_name": "fake",
                "model": "fake-model",
                "schedules": [
                    {
                        "pricing": {
                            "standard": [
                                {
                                    "max_input_tokens": 5,
                                    "input_per_million": "1",
                                    "output_per_million": "2",
                                    "cache_read_input_per_million": "0.25",
                                },
                                {
                                    "max_input_tokens": None,
                                    "input_per_million": "10",
                                    "output_per_million": "20",
                                    "cache_read_input_per_million": "2",
                                },
                            ]
                        },
                        "provenance": {
                            "source": "official",
                            "url": "https://example.com/pricing",
                            "as_of": "2026-07-13",
                        },
                    }
                ],
            }
        ],
    }

    client = TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG))
    response = client.post(
        "/api/sessions/tiered_cost/cost",
        json={"pricing": price_book},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["total_cost"] == "0.000108"
    assert body["line_items"][0]["pricing_tier_max_input_tokens"] is None
    assert (
        body["line_items"][0]["pricing_provenance"]
        == price_book["prices"][0]["schedules"][0]["provenance"]
    )


def test_server_exposes_causal_budget_usage_and_cost_with_tiered_price_book() -> None:
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

    client = TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG))
    usage_response = client.get("/api/causal-budgets/job_shared/usage")
    price_book = {
        "price_book_version": "test",
        "generated_at": "2026-07-13",
        "prices": [
            {
                "provider_name": "fake",
                "model": "fake-model",
                "schedules": [
                    {
                        "pricing": {
                            "standard": [
                                {
                                    "max_input_tokens": 5,
                                    "input_per_million": "1",
                                    "output_per_million": "2",
                                },
                                {
                                    "max_input_tokens": None,
                                    "input_per_million": "10",
                                    "output_per_million": "20",
                                },
                            ]
                        },
                        "provenance": {
                            "source": "official",
                            "url": "https://example.com/pricing",
                            "as_of": "2026-07-13",
                        },
                    }
                ],
            }
        ],
    }
    pricing_body = {"pricing": price_book}
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
        "unmeasured_model_attempts": 0,
        "provider_names": ["fake"],
        "models": ["fake-model"],
        "usage": _aggregate_usage_json(
            input_tokens=20,
            output_tokens=4,
            total_tokens=24,
            cache_read_tokens=8,
            cached_input_tokens=8,
            uncached_input_tokens=12,
        ),
        "session_summaries": [
            {
                "session_id": "causal_parent",
                "model_steps": 1,
                "tool_calls": 0,
                "unmeasured_model_attempts": 0,
                "provider_names": ["fake"],
                "models": ["fake-model"],
                "usage": _aggregate_usage_json(
                    input_tokens=10,
                    output_tokens=2,
                    total_tokens=12,
                    cache_read_tokens=4,
                    cached_input_tokens=4,
                    uncached_input_tokens=6,
                ),
            },
            {
                "session_id": "causal_child",
                "model_steps": 1,
                "tool_calls": 0,
                "unmeasured_model_attempts": 0,
                "provider_names": ["fake"],
                "models": ["fake-model"],
                "usage": _aggregate_usage_json(
                    input_tokens=10,
                    output_tokens=2,
                    total_tokens=12,
                    cache_read_tokens=4,
                    cached_input_tokens=4,
                    uncached_input_tokens=6,
                ),
            },
        ],
    }
    assert cost_response.status_code == 200
    assert cost_response.json()["causal_budget_id"] == "job_shared"
    assert cost_response.json()["session_ids"] == ["causal_parent", "causal_child"]
    assert cost_response.json()["session_count"] == 2
    assert cost_response.json()["model_steps"] == 2
    assert cost_response.json()["total_cost"] == "0.00028"
    assert all(
        item["line_items"][0]["pricing_provenance"]
        == price_book["prices"][0]["schedules"][0]["provenance"]
        for item in cost_response.json()["session_costs"]
    )
    assert all(
        item["line_items"][0]["pricing_tier_max_input_tokens"] is None
        for item in cost_response.json()["session_costs"]
    )
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
    assert summary_body["usage"]["usage"]["total_tokens"] == "24"
    assert summary_body["cost"]["total_cost"] == "0.00028"

    missing_summary_response = client.post(
        "/api/causal-budgets/missing/summary",
        json=pricing_body,
    )
    assert missing_summary_response.status_code == 404
    assert missing_summary_response.json() == {"detail": "Causal budget not found"}


def test_server_projects_session_authority_across_session_and_causal_budget_views() -> None:
    app = CayuApp(secret_redactor=SecretRedactor("-"), enable_logging=False)
    app.register_provider(UsageProvider(), default=True)
    app.register_agent(AgentSpec(name="assistant", model="fakemodel"))
    client = TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG))

    events = _sse_events(
        client,
        "/api/run",
        {
            "agent": "assistant",
            "prompt": "hello",
            "causal_budget_id": "job_shared",
        },
    )
    public_session_id = events[0]["session_id"]
    private_session_id = asyncio.run(app.session_store.list_sessions()).sessions[0].id
    assert public_session_id == app.project_session_id_for_exposure(private_session_id)

    pricing_body = {"pricing": _price_book_payload(model="fakemodel")}
    session_summary = client.post("/api/sessions/summary", json=pricing_body)
    session_usage = client.get(f"/api/sessions/{public_session_id}/usage")
    session_cost = client.post(
        f"/api/sessions/{public_session_id}/cost",
        json=pricing_body,
    )
    usage = client.get("/api/causal-budgets/job_shared/usage")
    cost = client.post("/api/causal-budgets/job_shared/cost", json=pricing_body)
    causal_summary = client.post(
        "/api/causal-budgets/job_shared/summary",
        json=pricing_body,
    )

    assert session_summary.status_code == 200
    session_item = session_summary.json()["sessions"][0]
    assert session_item["session"]["id"] == public_session_id
    assert session_item["outcome"]["session_id"] == public_session_id
    assert session_summary.json()["usage"]["session_ids"] == [public_session_id]
    assert [
        item["session_id"] for item in session_summary.json()["usage"]["session_summaries"]
    ] == [public_session_id]
    assert session_summary.json()["cost"]["session_ids"] == [public_session_id]
    assert [item["session_id"] for item in session_summary.json()["cost"]["session_costs"]] == [
        public_session_id
    ]

    assert session_usage.status_code == 200
    assert session_usage.json()["session_id"] == public_session_id
    assert session_cost.status_code == 200
    assert session_cost.json()["session_id"] == public_session_id

    assert usage.status_code == 200
    assert usage.json()["session_ids"] == [public_session_id]
    assert [item["session_id"] for item in usage.json()["session_summaries"]] == [public_session_id]

    assert cost.status_code == 200
    assert cost.json()["session_ids"] == [public_session_id]
    assert [item["session_id"] for item in cost.json()["session_costs"]] == [public_session_id]

    assert causal_summary.status_code == 200
    causal_body = causal_summary.json()
    assert causal_body["sessions"][0]["session"]["id"] == public_session_id
    assert causal_body["sessions"][0]["outcome"]["session_id"] == public_session_id
    assert causal_body["usage"]["session_ids"] == [public_session_id]
    assert causal_body["cost"]["session_ids"] == [public_session_id]


def test_server_rejects_a_merged_flat_and_model_catalog_pricing_body() -> None:
    app = CayuApp()
    client = TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG))
    ambiguous = {
        "prices": [],
        "price_book_version": "test",
        "generated_at": "2026-07-13",
        "models": [],
    }

    response = client.post(
        "/api/causal-budgets/missing/cost",
        json={"pricing": ambiguous},
    )

    assert response.status_code == 422


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

    client = TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG))
    response = client.post(
        "/api/sessions/cost_unpriced/cost",
        json={
            "pricing": _price_book_payload(
                provider_name="other-provider",
                model="other-model",
            )
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

    client = TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG))
    response = client.post(
        "/api/sessions/missing/cost",
        json={"pricing": _price_book_payload()},
    )

    assert response.status_code == 404
    assert response.json() == {"detail": "Session not found"}


def test_server_session_cost_validates_pricing_body() -> None:
    app = CayuApp()
    app.register_provider(UsageProvider(), default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    client = TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG))
    response = client.post(
        "/api/sessions/session_1/cost",
        json={"pricing": _price_book_payload(input_per_million="-1")},
    )

    assert response.status_code == 422
